"""
dip_stream_predispatch.py — Selective-streaming DIP (pre-dispatch), now with
BOTH up_proj and down_proj running on custom int4 gather kernels.

up_proj  : int4_gather_gemv  (row gather -> dot products)   -> up_sel [k]
down_proj: int4_gather_down  (row gather of transpose -> weighted sum) -> y [out]

down is stored TRANSPOSED + int4 so selected channels are contiguous int4 rows
(halves down's pinned memory + streamed bytes vs the old bf16 transpose).

Load sequence unchanged: build empty -> QuantLinear -> replace MLPs with
StreamingDIPMlp BEFORE dispatch -> load non-MLP weights -> dispatch.
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
_use_down = False
if os.path.isdir(_so_dir):
    for prefix, flag in (("int4_gather_ops", "up"), ("int4_down_ops", "down")):
        sos = [f for f in os.listdir(_so_dir)
               if f.startswith(prefix) and f.endswith(".so")]
        if sos:
            ctypes.CDLL(os.path.join(_so_dir, sos[0]))
    # register fakes if the ops exist
    try:
        @torch.library.register_fake("int4_gather_ops::int4_gather_gemv")
        def _(qweight, scales, x, indices):
            return torch.empty(indices.shape[0], dtype=torch.bfloat16, device=x.device)
        _use_gather = True
    except Exception:
        pass
    try:
        @torch.library.register_fake("int4_down_ops::int4_gather_down")
        def _(down_t_sel, scales, h_sel):
            out = down_t_sel.shape[1] * 2
            return torch.empty(out, dtype=torch.bfloat16, device=h_sel.device)
        _use_down = True
    except Exception:
        pass


def _unpack_int4_rows(qweight):
    """int8 [R, C/2] packed -> int8 [R, C] values in [-8,7]."""
    q = qweight.to(torch.uint8)
    low = q & 0x0F
    high = (q >> 4) & 0x0F
    def sext(x):
        x = x.to(torch.int8)
        return torch.where(x >= 8, x - 16, x)
    low = sext(low); high = sext(high)
    R, half = q.shape
    out = torch.empty(R, half * 2, dtype=torch.int8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


def _pack_int4_rows(q):
    """int8 [R, C] in [-8,7] -> packed int8 [R, C/2]."""
    lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
    hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def _transpose_down_to_int4(down_qw, down_sc):
    """
    down_proj stored as [out, inter] int4 (packed along inter), scales
    [out, inter/128]. We want down_t int4 [inter, out] with scales
    [inter, out/128], so channel c is a contiguous packed int4 row.

    Simplest correct route: dequantize -> transpose -> re-quantize int4 g128
    along the new last dim (out). One-time cost at load.
    """
    w = dequantize(down_qw, down_sc)          # [out, inter] bf16
    wt = w.t().contiguous().float()           # [inter, out]
    inter, out_f = wt.shape
    assert out_f % GROUP_SIZE == 0
    n_groups = out_f // GROUP_SIZE
    wg = wt.reshape(inter, n_groups, GROUP_SIZE)
    max_abs = wg.abs().amax(dim=2, keepdim=True)
    scales = (max_abs / 7.0).clamp(min=1e-8)
    q = torch.round(wg / scales).clamp(-8, 7).to(torch.int8).reshape(inter, out_f)
    packed = _pack_int4_rows(q)               # [inter, out/2]
    scales = scales.squeeze(-1).to(torch.bfloat16)   # [inter, n_groups]
    return packed, scales


class SharedBuffers:
    def __init__(self, k, up_in_packed, up_ngroups, down_out, down_ngroups, device):
        self.k = k
        self.device = torch.device(device)
        # up staging
        self.gpu_up_qw = torch.empty(k, up_in_packed, dtype=torch.int8, device=device)
        self.gpu_up_sc = torch.empty(k, up_ngroups, dtype=torch.bfloat16, device=device)
        self.pin_up_qw = torch.empty(k, up_in_packed, dtype=torch.int8).pin_memory()
        self.pin_up_sc = torch.empty(k, up_ngroups, dtype=torch.bfloat16).pin_memory()
        # down staging (int4 transpose rows)
        self.gpu_dn_qw = torch.empty(k, down_out // 2, dtype=torch.int8, device=device)
        self.gpu_dn_sc = torch.empty(k, down_ngroups, dtype=torch.bfloat16, device=device)
        self.pin_dn_qw = torch.empty(k, down_out // 2, dtype=torch.int8).pin_memory()
        self.pin_dn_sc = torch.empty(k, down_ngroups, dtype=torch.bfloat16).pin_memory()
        self.local_idx = torch.arange(k, dtype=torch.int32, device=device)
        self.copy_stream = torch.cuda.Stream(device=device)


class StreamingDIPMlp(nn.Module):
    def __init__(self, gate_qw, gate_sc, up_qw, up_sc, down_qw, down_sc,
                 shared, keep_ratio=0.6, device="cuda:0"):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.device = torch.device(device)
        self.buf = shared

        self.gate_qw = gate_qw.to(device)
        self.gate_sc = gate_sc.to(device)

        self.up_qw = up_qw.contiguous().pin_memory()
        self.up_sc = up_sc.contiguous().pin_memory()
        self.inter = up_qw.shape[0]

        # down: transpose + repack int4, pin
        dn_t_qw, dn_t_sc = _transpose_down_to_int4(down_qw, down_sc)
        self.down_t_qw = dn_t_qw.contiguous().pin_memory()   # [inter, out/2]
        self.down_t_sc = dn_t_sc.contiguous().pin_memory()   # [inter, out/128]
        self.out_features = dn_t_qw.shape[1] * 2

        self.k = shared.k

    def _gate(self, x):
        w = dequantize(self.gate_qw, self.gate_sc).to(x.dtype)
        return F.linear(x, w)

    def forward(self, x):
        if (not (_use_gather and _use_down) or x.dim() != 3 or x.shape[1] != 1
                or self.keep_ratio >= 0.999):
            return self._dense(x)
        b = self.buf
        xdt = x.dtype
        g = F.silu(self._gate(x)).reshape(-1)          # [inter]
        k = self.k
        idx = g.abs().topk(k).indices
        idx_cpu = idx.to("cpu")

        with torch.cuda.stream(b.copy_stream):
            torch.index_select(self.up_qw, 0, idx_cpu, out=b.pin_up_qw)
            torch.index_select(self.up_sc, 0, idx_cpu, out=b.pin_up_sc)
            b.gpu_up_qw.copy_(b.pin_up_qw, non_blocking=True)
            b.gpu_up_sc.copy_(b.pin_up_sc, non_blocking=True)
            torch.index_select(self.down_t_qw, 0, idx_cpu, out=b.pin_dn_qw)
            torch.index_select(self.down_t_sc, 0, idx_cpu, out=b.pin_dn_sc)
            b.gpu_dn_qw.copy_(b.pin_dn_qw, non_blocking=True)
            b.gpu_dn_sc.copy_(b.pin_dn_sc, non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(b.copy_stream)

        xf = x.reshape(-1).to(torch.bfloat16)
        up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(
            b.gpu_up_qw, b.gpu_up_sc, xf, b.local_idx)      # [k]
        h_sel = (g[idx] * up_sel.to(xdt)).to(torch.bfloat16)  # [k]
        y = torch.ops.int4_down_ops.int4_gather_down(
            b.gpu_dn_qw, b.gpu_dn_sc, h_sel)                # [out]
        return y.to(xdt).reshape(x.shape[0], 1, -1)

    def _dense(self, x):
        g = F.silu(self._gate(x))
        up_w = dequantize(self.up_qw.to(self.device),
                          self.up_sc.to(self.device)).to(x.dtype)
        u = F.linear(x, up_w); del up_w
        h = g * u
        # dense down: reconstruct from transposed int4
        dn_t = dequantize(self.down_t_qw.to(self.device),
                          self.down_t_sc.to(self.device)).to(x.dtype)  # [inter, out]
        y = h @ dn_t
        return y


def load_streaming_dip(ckpt_dir, gpu_mem_gib=10, cpu_mem_gib=60, keep_ratio=0.6,
                       device="cuda:0"):
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
    mlp_names = [(n, m) for n, m in model.named_modules()
                 if all(hasattr(m, p) for p in ("gate_proj", "up_proj", "down_proj"))
                 and isinstance(m.up_proj, QuantLinear)]

    shared = None
    sdips = []
    print("Building streaming MLPs (repacking down to int4 transpose; one-time) ...")
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for i, (name, mlp) in enumerate(mlp_names):
            g_qw = f.get_tensor(f"{name}.gate_proj.qweight")
            g_sc = f.get_tensor(f"{name}.gate_proj.scales")
            u_qw = f.get_tensor(f"{name}.up_proj.qweight")
            u_sc = f.get_tensor(f"{name}.up_proj.scales")
            d_qw = f.get_tensor(f"{name}.down_proj.qweight")
            d_sc = f.get_tensor(f"{name}.down_proj.scales")
            if shared is None:
                k = max(1, int(round(u_qw.shape[0] * keep_ratio)))
                out_f = d_qw.shape[0]
                shared = SharedBuffers(k, u_qw.shape[1], u_sc.shape[1],
                                       out_f, out_f // GROUP_SIZE, device)
                print(f"Shared buffers: k={k}, out={out_f}")
            sdip = StreamingDIPMlp(g_qw, g_sc, u_qw, u_sc, d_qw, d_sc,
                                   shared, keep_ratio, device)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], sdip)
            sdips.append(sdip)
            if (i + 1) % 16 == 0:
                print(f"  ... {i+1}/{len(mlp_names)} MLPs built")
    print(f"Built {len(sdips)} streaming MLPs")

    max_memory = {0: f"{gpu_mem_gib}GiB", "cpu": f"{cpu_mem_gib}GiB"}
    device_map = infer_auto_device_map(
        model, max_memory=max_memory,
        no_split_module_classes=["Qwen3DecoderLayer"], dtype=torch.bfloat16)
    print(f"Device map: {dict(Counter(str(v) for v in device_map.values()))}")

    def device_for(pn):
        best = None
        for mn, dev in device_map.items():
            if pn == mn or pn.startswith(mn + "."):
                if best is None or len(mn) > len(best[0]):
                    best = (mn, dev)
        return best[1] if best else "cpu"

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if any(s in key for s in (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
                continue
            try:
                dev = device_for(key)
                target = "cuda:0" if dev == 0 or dev == "cuda:0" else "cpu"
                set_module_tensor_to_device(model, key, target, value=f.get_tensor(key))
            except Exception as e:
                print(f"  warn: {key}: {e}")

    model = dispatch_model(model, device_map=device_map, offload_buffers=True)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tok, sdips