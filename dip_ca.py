"""
dip_ca.py — Cache-aware DIP for GPU offload.

Extends selective streaming with a resident channel cache: the
most-frequently-selected channels per layer keep their up/down int4 weights
resident on the GPU. Each token, the selected channels split into:
  * hits   : resident -> no streaming
  * misses : not resident -> streamed by pinned-gather

The cache is populated by a short calibration pass that records channel pick
frequency over a few tokens and keeps the top cache_frac per layer resident.
In the offload setting the bottleneck is CPU->GPU transfer, so reducing misses
reduces PCIe traffic.
"""

import ctypes, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from qlinear import QuantLinear, dequantize, _replace_linears_with_quant, GROUP_SIZE
# importing this also loads the int4_gather / int4_down kernels used below
from dip_stream_predispatch import (_transpose_down_to_int4, _use_gather, _use_down)

try:
    from topk_kernel import select_topk, _use_topk
except Exception:
    _use_topk = False
    def select_topk(s, k):
        return s.abs().topk(k).indices

try:
    from gate_kernel import gate_gemv, _use_gate
except Exception:
    _use_gate = False
    def gate_gemv(qweight, scales, x):
        raise RuntimeError("gate kernel unavailable")

# cache-aware fused GEMV kernel (dual-source gather + dequant + GEMV for up_proj)
_ca_so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "ca_fused")
_use_ca_fused = False
if os.path.isdir(_ca_so):
    _s = [f for f in os.listdir(_ca_so)
          if f.startswith("ca_fused_ops") and f.endswith(".so")]
    if _s:
        ctypes.CDLL(os.path.join(_ca_so, _s[0]))
        try:
            @torch.library.register_fake("ca_fused_ops::ca_fused_gemv")
            def _(res_qw, res_sc, str_qw, str_sc, x, src, row):
                return torch.empty(src.shape[0], dtype=torch.bfloat16, device=x.device)
            _use_ca_fused = True
        except Exception:
            pass


class CABuffers:
    """Shared staging buffers for streaming the misses (sized to worst-case k)."""
    def __init__(self, k, up_in_packed, up_ngroups, down_out, down_ngroups, device):
        self.device = torch.device(device)
        self.gpu_up_qw = torch.empty(k, up_in_packed, dtype=torch.int8, device=device)
        self.gpu_up_sc = torch.empty(k, up_ngroups, dtype=torch.bfloat16, device=device)
        self.pin_up_qw = torch.empty(k, up_in_packed, dtype=torch.int8).pin_memory()
        self.pin_up_sc = torch.empty(k, up_ngroups, dtype=torch.bfloat16).pin_memory()
        self.gpu_dn_qw = torch.empty(k, down_out // 2, dtype=torch.int8, device=device)
        self.gpu_dn_sc = torch.empty(k, down_ngroups, dtype=torch.bfloat16, device=device)
        self.pin_dn_qw = torch.empty(k, down_out // 2, dtype=torch.int8).pin_memory()
        self.pin_dn_sc = torch.empty(k, down_ngroups, dtype=torch.bfloat16).pin_memory()
        self.copy_stream = torch.cuda.Stream(device=device)


class CADIPMlp(nn.Module):
    """MLP with a resident channel cache: resident channels' up/down int4
    weights live on the GPU; misses are streamed per token."""
    def __init__(self, gate_qw, gate_sc, up_qw, up_sc, down_qw, down_sc,
                 buf, keep_ratio=0.6, cache_frac=0.3, device="cuda:0"):
        super().__init__()
        self.device = torch.device(device)
        self.keep_ratio = keep_ratio
        self.buf = buf

        self.gate_qw = gate_qw.to(device)
        self.gate_sc = gate_sc.to(device)

        # full up/down pinned on CPU (source for streaming misses)
        self.up_qw = up_qw.contiguous().pin_memory()
        self.up_sc = up_sc.contiguous().pin_memory()
        self.inter = up_qw.shape[0]

        dn_t_qw, dn_t_sc = _transpose_down_to_int4(down_qw, down_sc)
        self.down_t_qw = dn_t_qw.contiguous().pin_memory()
        self.down_t_sc = dn_t_sc.contiguous().pin_memory()
        self.out_features = dn_t_qw.shape[1] * 2

        self.k = int(round(self.inter * keep_ratio))
        self.cache_n = int(round(self.inter * cache_frac))

        self.register_buffer("cache_ids", torch.empty(0, dtype=torch.long), persistent=False)
        # channel_id -> row in the resident cache, or -1 if not cached
        self.register_buffer("id2slot", torch.full((self.inter,), -1, dtype=torch.long), persistent=False)
        self._res_up_qw = None
        self._res_up_sc = None
        self._res_dn_qw = None
        self._res_dn_sc = None

    def set_cache(self, hot_ids):
        """Load the given channel ids' up/down weights resident on the GPU."""
        hot_ids = hot_ids[: self.cache_n].sort().values
        self.cache_ids = hot_ids.to(self.device)
        id2 = torch.full((self.inter,), -1, dtype=torch.long)
        id2[hot_ids] = torch.arange(len(hot_ids))
        self.id2slot = id2.to(self.device)
        self._res_up_qw = self.up_qw[hot_ids].to(self.device)
        self._res_up_sc = self.up_sc[hot_ids].to(self.device)
        self._res_dn_qw = self.down_t_qw[hot_ids].to(self.device)
        self._res_dn_sc = self.down_t_sc[hot_ids].to(self.device)

    def _gate(self, x):
        if _use_gate and x.dim() == 3 and x.shape[1] == 1:
            xf = x.reshape(-1)
            out = gate_gemv(self.gate_qw, self.gate_sc, xf)
            return out.reshape(x.shape[0], 1, -1)
        w = dequantize(self.gate_qw, self.gate_sc).to(x.dtype)
        return F.linear(x, w)

    def forward(self, x):
        if (not (_use_gather and _use_down) or x.dim() != 3 or x.shape[1] != 1
                or self.keep_ratio >= 0.999 or self._res_up_qw is None):
            return self._dense(x)
        b = self.buf
        xdt = x.dtype
        g = F.silu(self._gate(x)).reshape(-1)          # [inter]
        idx = select_topk(g, self.k)                   # [k]

        # split selected channels into hits (resident) and misses (to stream)
        slots = self.id2slot[idx]                      # [k], -1 if miss
        hit_mask = slots >= 0
        miss_ids = idx[~hit_mask]
        hit_ids = idx[hit_mask]
        hit_slots = slots[hit_mask]

        n_miss = miss_ids.numel()
        miss_ids_cpu = miss_ids.to("cpu")

        if n_miss > 0:
            with torch.cuda.stream(b.copy_stream):
                torch.index_select(self.up_qw, 0, miss_ids_cpu, out=b.pin_up_qw[:n_miss])
                torch.index_select(self.up_sc, 0, miss_ids_cpu, out=b.pin_up_sc[:n_miss])
                b.gpu_up_qw[:n_miss].copy_(b.pin_up_qw[:n_miss], non_blocking=True)
                b.gpu_up_sc[:n_miss].copy_(b.pin_up_sc[:n_miss], non_blocking=True)
                torch.index_select(self.down_t_qw, 0, miss_ids_cpu, out=b.pin_dn_qw[:n_miss])
                torch.index_select(self.down_t_sc, 0, miss_ids_cpu, out=b.pin_dn_sc[:n_miss])
                b.gpu_dn_qw[:n_miss].copy_(b.pin_dn_qw[:n_miss], non_blocking=True)
                b.gpu_dn_sc[:n_miss].copy_(b.pin_dn_sc[:n_miss], non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(b.copy_stream)

        # order is [hits..., misses...] in combined_ids; the src/row tables and
        # g_sel below all follow the same order so rows stay aligned.
        combined_ids = torch.cat([hit_ids, miss_ids], dim=0)
        k_tot = combined_ids.numel()
        xf = x.reshape(-1).to(torch.bfloat16)

        if _use_ca_fused:
            n_hit = hit_ids.numel()
            src = torch.empty(k_tot, dtype=torch.int32, device=self.device)
            row = torch.empty(k_tot, dtype=torch.int32, device=self.device)
            src[:n_hit] = 0                                   # hit -> resident cache
            row[:n_hit] = hit_slots.to(torch.int32)
            src[n_hit:] = 1                                   # miss -> streamed buffer
            row[n_hit:] = torch.arange(n_miss, dtype=torch.int32, device=self.device)
            up_sel = torch.ops.ca_fused_ops.ca_fused_gemv(
                self._res_up_qw, self._res_up_sc,
                b.gpu_up_qw, b.gpu_up_sc, xf, src, row)       # [k_tot]
            dn_qw = torch.cat([self._res_dn_qw[hit_slots], b.gpu_dn_qw[:n_miss]], dim=0)
            dn_sc = torch.cat([self._res_dn_sc[hit_slots], b.gpu_dn_sc[:n_miss]], dim=0)
        else:
            up_qw = torch.cat([self._res_up_qw[hit_slots], b.gpu_up_qw[:n_miss]], dim=0)
            up_sc = torch.cat([self._res_up_sc[hit_slots], b.gpu_up_sc[:n_miss]], dim=0)
            dn_qw = torch.cat([self._res_dn_qw[hit_slots], b.gpu_dn_qw[:n_miss]], dim=0)
            dn_sc = torch.cat([self._res_dn_sc[hit_slots], b.gpu_dn_sc[:n_miss]], dim=0)
            local_idx = torch.arange(k_tot, dtype=torch.int32, device=self.device)
            up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(up_qw, up_sc, xf, local_idx)

        g_sel = g[combined_ids]
        h_sel = (g_sel * up_sel.to(xdt)).to(torch.bfloat16)
        y = torch.ops.int4_down_ops.int4_gather_down(dn_qw, dn_sc, h_sel)   # [out]
        return y.to(xdt).reshape(x.shape[0], 1, -1)

    def _dense(self, x):
        g = F.silu(self._gate(x))
        up_w = dequantize(self.up_qw.to(self.device), self.up_sc.to(self.device)).to(x.dtype)
        u = F.linear(x, up_w); del up_w
        h = g * u
        dn_t = dequantize(self.down_t_qw.to(self.device), self.down_t_sc.to(self.device)).to(x.dtype)
        return h @ dn_t


def load_dip_ca(ckpt_dir, gpu_mem_gib=10, cpu_mem_gib=60, keep_ratio=0.6,
                cache_frac=0.3, device="cuda:0"):
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

    buf = None
    cas = []
    print(f"Building CA-DIP MLPs (keep={keep_ratio}, cache_frac={cache_frac}) ...")
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for i, (name, mlp) in enumerate(mlp_names):
            g_qw = f.get_tensor(f"{name}.gate_proj.qweight")
            g_sc = f.get_tensor(f"{name}.gate_proj.scales")
            u_qw = f.get_tensor(f"{name}.up_proj.qweight")
            u_sc = f.get_tensor(f"{name}.up_proj.scales")
            d_qw = f.get_tensor(f"{name}.down_proj.qweight")
            d_sc = f.get_tensor(f"{name}.down_proj.scales")
            if buf is None:
                k = int(round(u_qw.shape[0] * keep_ratio))
                out_f = d_qw.shape[0]
                buf = CABuffers(k, u_qw.shape[1], u_sc.shape[1], out_f,
                                out_f // GROUP_SIZE, device)
            ca = CADIPMlp(g_qw, g_sc, u_qw, u_sc, d_qw, d_sc, buf,
                          keep_ratio, cache_frac, device)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], ca)
            cas.append(ca)
            if (i + 1) % 16 == 0:
                print(f"  ... {i+1}/{len(mlp_names)}")
    print(f"Built {len(cas)} CA-DIP MLPs")

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
    return model, tok, cas


def calibrate_cache(model, tok, cas, prompt, calib_tokens=30):
    """Record channel selection frequency over a short pass, then load each
    layer's most-frequent channels resident."""
    counts = [torch.zeros(c.inter, dtype=torch.long) for c in cas]

    handles = []
    def mk(lid, c):
        def hook(module, inp, out):
            x = inp[0]
            if x.dim() == 3 and x.shape[1] == 1:
                g = F.silu(c._gate(x)).reshape(-1)
                idx = select_topk(g, c.k).to("cpu")
                counts[lid][idx] += 1
        return hook
    for lid, c in enumerate(cas):
        handles.append(c.register_forward_hook(mk(lid, c)))

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    ids = ids.to("cuda:0")
    print(f"Calibrating cache over {calib_tokens} tokens ...")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=calib_tokens, do_sample=False,
                       use_cache=True, pad_token_id=tok.eos_token_id)
    for h in handles:
        h.remove()

    for lid, c in enumerate(cas):
        cnt = counts[lid]
        # Deterministic hot-set: order by (frequency desc, channel id asc) so
        # ties resolve the same way every run. A plain argsort is unstable
        # across ties; encode a single key freq*inter - id for a total order.
        inter = c.inter
        key = cnt.to(torch.long) * inter - torch.arange(inter, dtype=torch.long)
        hot = key.argsort(descending=True)
        c.set_cache(hot)
    print("Cache loaded resident.")