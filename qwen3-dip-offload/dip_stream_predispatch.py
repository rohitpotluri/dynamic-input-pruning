"""
dip_stream_predispatch.py — Selective-streaming DIP built BEFORE accelerate
dispatch. This is the proven-safe structure: accelerate never hooks the
streaming MLPs (they don't hold up/down as managed params), so no hook
collision and no _old_forward leftover.

Load sequence:
  1. build empty model, swap Linears -> QuantLinear
  2. load ALL int4 weights to CPU (real data, from checkpoint)
  3. replace each MLP with StreamingDIPMlp, handing it the up/down weights
     (pinned on CPU). Remove up/down from the module tree so accelerate's
     device-map never assigns/hooks them.
  4. infer device map + dispatch: accelerate only manages gate/attn/norms.
     The StreamingDIPMlp modules exist BEFORE dispatch, so if accelerate
     hooks them at all, it hooks OUR module cleanly (no post-hoc surgery).

Only gate_proj / attention / norms are offloaded by accelerate. up/down are
owned by us and streamed selectively (method C: pinned-gather).
"""

import ctypes, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from qlinear import (QuantLinear, dequantize, _replace_linears_with_quant,
                     GROUP_SIZE)

_here = os.path.dirname(os.path.abspath(__file__))
_so_dir = os.path.join(_here, "kernels", "int4_gather_gemv")
_use_gather = False
if os.path.isdir(_so_dir):
    _sos = [f for f in os.listdir(_so_dir)
            if f.startswith("int4_gather_ops") and f.endswith(".so")]
    if _sos:
        ctypes.CDLL(os.path.join(_so_dir, _sos[0]))
        @torch.library.register_fake("int4_gather_ops::int4_gather_gemv")
        def _(qweight, scales, x, indices):
            return torch.empty(indices.shape[0], dtype=torch.bfloat16, device=x.device)
        _use_gather = True


class SharedBuffers:
    def __init__(self, k, in_packed, n_groups_up, out_features, device):
        self.k = k
        self.device = torch.device(device)
        self.gpu_up_qw = torch.empty(k, in_packed, dtype=torch.int8, device=device)
        self.gpu_up_sc = torch.empty(k, n_groups_up, dtype=torch.bfloat16, device=device)
        self.pin_up_qw = torch.empty(k, in_packed, dtype=torch.int8).pin_memory()
        self.pin_up_sc = torch.empty(k, n_groups_up, dtype=torch.bfloat16).pin_memory()
        self.gpu_down = torch.empty(k, out_features, dtype=torch.bfloat16, device=device)
        self.pin_down = torch.empty(k, out_features, dtype=torch.bfloat16).pin_memory()
        self.local_idx = torch.arange(k, dtype=torch.int32, device=device)
        self.copy_stream = torch.cuda.Stream(device=device)


class StreamingDIPMlp(nn.Module):
    """
    Self-contained MLP: owns up/down pinned on CPU, gate_proj is a normal
    (resident) QuantLinear on GPU. No accelerate management of up/down.
    NOTE: we keep gate_proj RESIDENT on GPU (small) so this module needs no
    accelerate hook at all.
    """
    def __init__(self, gate_qw, gate_sc, up_qw, up_sc, down_qw, down_sc,
                 shared, keep_ratio=0.32, device="cuda:0"):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.device = torch.device(device)
        self.buf = shared

        # gate resident on GPU (small: inter x dim int4)
        self.gate_qw = gate_qw.to(device)
        self.gate_sc = gate_sc.to(device)

        # up pinned on CPU
        self.up_qw = up_qw.contiguous().pin_memory()
        self.up_sc = up_sc.contiguous().pin_memory()
        self.inter = up_qw.shape[0]

        # down dequantized+transposed, pinned on CPU
        dn = dequantize(down_qw, down_sc)          # [out, inter]
        self.down_t = dn.t().contiguous().pin_memory()   # [inter, out]
        self.out_features = dn.shape[0]
        del dn

        self.k = shared.k

    def _gate(self, x):
        w = dequantize(self.gate_qw, self.gate_sc).to(x.dtype)
        return F.linear(x, w)

    def forward(self, x):
        if (not _use_gather or x.dim() != 3 or x.shape[1] != 1
                or self.keep_ratio >= 0.999):
            return self._dense(x)
        b = self.buf
        xdt = x.dtype
        g = F.silu(self._gate(x)).reshape(-1)         # [inter]
        k = self.k
        idx = g.abs().topk(k).indices
        idx_cpu = idx.to("cpu")

        with torch.cuda.stream(b.copy_stream):
            torch.index_select(self.up_qw, 0, idx_cpu, out=b.pin_up_qw)
            torch.index_select(self.up_sc, 0, idx_cpu, out=b.pin_up_sc)
            b.gpu_up_qw.copy_(b.pin_up_qw, non_blocking=True)
            b.gpu_up_sc.copy_(b.pin_up_sc, non_blocking=True)
            torch.index_select(self.down_t, 0, idx_cpu, out=b.pin_down)
            b.gpu_down.copy_(b.pin_down, non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(b.copy_stream)

        xf = x.reshape(-1).to(torch.bfloat16)
        up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(
            b.gpu_up_qw, b.gpu_up_sc, xf, b.local_idx)   # [k]
        h_sel = g[idx] * up_sel.to(xdt)                  # [k]
        y = (h_sel.unsqueeze(0).to(b.gpu_down.dtype) @ b.gpu_down).squeeze(0)  # [out]
        return y.to(xdt).reshape(x.shape[0], 1, -1)

    def _dense(self, x):
        g = F.silu(self._gate(x))                                    # [b,s,inter]
        up_w = dequantize(self.up_qw.to(self.device),
                          self.up_sc.to(self.device)).to(x.dtype)
        u = F.linear(x, up_w); del up_w                              # [b,s,inter]
        h = g * u
        y = h @ self.down_t.to(self.device).to(x.dtype)             # [b,s,out]
        return y


def load_streaming_dip(ckpt_dir, gpu_mem_gib=10, cpu_mem_gib=60, keep_ratio=0.32,
                       device="cuda:0"):
    """Full loader: streaming MLPs built BEFORE dispatch."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
    from accelerate.utils import set_module_tensor_to_device
    from collections import Counter

    ckpt_dir = Path(ckpt_dir)
    config = AutoConfig.from_pretrained(ckpt_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    _replace_linears_with_quant(model)

    st_path = ckpt_dir / "model_int4.safetensors"

    # find MLPs
    mlp_names = [(n, m) for n, m in model.named_modules()
                 if all(hasattr(m, p) for p in ("gate_proj", "up_proj", "down_proj"))
                 and isinstance(m.up_proj, QuantLinear)]

    # build streaming MLPs (pre-dispatch). Read gate/up/down from checkpoint.
    shared = None
    sdips = []
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name, mlp in mlp_names:
            g_qw = f.get_tensor(f"{name}.gate_proj.qweight")
            g_sc = f.get_tensor(f"{name}.gate_proj.scales")
            u_qw = f.get_tensor(f"{name}.up_proj.qweight")
            u_sc = f.get_tensor(f"{name}.up_proj.scales")
            d_qw = f.get_tensor(f"{name}.down_proj.qweight")
            d_sc = f.get_tensor(f"{name}.down_proj.scales")
            if shared is None:
                k = max(1, int(round(u_qw.shape[0] * keep_ratio)))
                shared = SharedBuffers(k, u_qw.shape[1], u_sc.shape[1],
                                       d_qw.shape[0], device)
                print(f"Shared buffers: k={k}")
            sdip = StreamingDIPMlp(g_qw, g_sc, u_qw, u_sc, d_qw, d_sc,
                                   shared, keep_ratio, device)
            # replace the whole mlp module with our streaming module BEFORE dispatch
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], sdip)
            sdips.append(sdip)
    print(f"Built {len(sdips)} streaming MLPs (pre-dispatch)")

    # now the model tree has StreamingDIPMlp (self-contained, GPU/CPU managed by
    # us) in place of the original MLPs. accelerate will map the REMAINING
    # params (embeddings, attention, norms, lm_head).
    max_memory = {0: f"{gpu_mem_gib}GiB", "cpu": f"{cpu_mem_gib}GiB"}
    device_map = infer_auto_device_map(
        model, max_memory=max_memory,
        no_split_module_classes=["Qwen3DecoderLayer"], dtype=torch.bfloat16)
    print(f"Device map: {dict(Counter(str(v) for v in device_map.values()))}")

    # load remaining (non-MLP) weights to their mapped devices
    def device_for(pn):
        best = None
        for mn, dev in device_map.items():
            if pn == mn or pn.startswith(mn + "."):
                if best is None or len(mn) > len(best[0]):
                    best = (mn, dev)
        return best[1] if best else "cpu"

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            # skip MLP up/down/gate — already owned by StreamingDIPMlp
            if any(s in key for s in (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
                continue
            try:
                dev = device_for(key)
                target = "cuda:0" if dev == 0 or dev == "cuda:0" else "cpu"
                set_module_tensor_to_device(model, key, target, value=f.get_tensor(key))
            except Exception as e:
                print(f"  warn: could not place {key}: {e}")

    model = dispatch_model(model, device_map=device_map, offload_buffers=True)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tok, sdips