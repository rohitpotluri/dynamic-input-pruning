"""
dip_linear.py — DIP injected BEFORE accelerate dispatch (the correct fix).

Root cause of earlier garbage: accelerate binds offload hooks to the module
objects present at dispatch time. Swapping modules AFTER dispatch leaves the
new modules unhooked -> offloaded weights never materialize -> garbage.

Fix: build empty model -> swap Linears to QuantLinear -> swap the MLP
gate/up projections to DIP versions -> THEN infer device map, load weights,
and dispatch. accelerate now hooks the DIP modules natively.

This module provides:
  * DIPGate / DIPUp   : QuantLinear subclasses (buffers identical, so weight
                        loading + hooks treat them exactly like QuantLinear)
  * DIPContext        : shared per-MLP scratch for the selected indices
  * load_quantized_qwen3_dip(...) : loader that does DIP-before-dispatch
"""

import ctypes, os, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlinear import QuantLinear, dequantize, _replace_linears_with_quant, GROUP_SIZE

# ---- load gather kernel ----
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


class DIPContext:
    def __init__(self, keep_ratio=0.32):
        self.keep_ratio = keep_ratio
        self.idx = None


def _dequant_linear(qlin, x):
    w = dequantize(qlin.qweight, qlin.scales).to(x.dtype)
    return F.linear(x, w)


class DIPGate(QuantLinear):
    """gate_proj: normal output + stash top-k indices on the shared ctx."""
    ctx: DIPContext = None
    def forward(self, x):
        out = _dequant_linear(self, x)
        c = self.ctx
        if (_use_gather and c is not None and x.dim() == 3 and x.shape[1] == 1
                and c.keep_ratio < 0.999):
            g = F.silu(out).reshape(-1)
            k = max(1, int(round(g.shape[0] * c.keep_ratio)))
            c.idx = g.abs().topk(k).indices.to(torch.int32)
        elif c is not None:
            c.idx = None
        return out


class DIPUp(QuantLinear):
    """up_proj: gather-compute only selected rows when indices are stashed."""
    ctx: DIPContext = None
    def forward(self, x):
        c = self.ctx
        if c is None or c.idx is None:
            return _dequant_linear(self, x)
        xf = x.reshape(-1).to(torch.bfloat16)
        up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(
            self.qweight, self.scales, xf, c.idx)
        full = torch.zeros(self.qweight.shape[0], dtype=x.dtype, device=x.device)
        full[c.idx.long()] = up_sel.to(x.dtype)
        return full.reshape(x.shape[0], 1, -1)


def _retype(module, new_cls, ctx):
    """Change an existing QuantLinear's class in-place to new_cls, attach ctx.
    In-place __class__ reassignment keeps the SAME object (and its buffers),
    so device_map inference + weight loading + hooks all still target it."""
    assert isinstance(module, QuantLinear)
    module.__class__ = new_cls
    module.ctx = ctx
    return module


def convert_mlps_before_dispatch(model, keep_ratio=0.32):
    """Retype gate/up QuantLinears to DIP versions IN PLACE (pre-dispatch).
    Returns list of DIPContext (shared per MLP)."""
    ctxs = []
    for name, mlp in model.named_modules():
        if all(hasattr(mlp, p) for p in ("gate_proj", "up_proj", "down_proj")):
            if not isinstance(mlp.up_proj, QuantLinear):
                continue
            ctx = DIPContext(keep_ratio)
            _retype(mlp.gate_proj, DIPGate, ctx)
            _retype(mlp.up_proj, DIPUp, ctx)
            ctxs.append(ctx)
    return ctxs


def load_quantized_qwen3_dip(ckpt_dir, gpu_mem_gib=10, cpu_mem_gib=60, keep_ratio=0.32):
    """Loader that swaps in DIP modules BEFORE dispatch, so accelerate hooks them."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file
    from collections import Counter

    ckpt_dir = Path(ckpt_dir)
    json.load(open(ckpt_dir / "quant_config.json"))  # sanity
    config = AutoConfig.from_pretrained(ckpt_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    _replace_linears_with_quant(model)
    ctxs = convert_mlps_before_dispatch(model, keep_ratio)   # <-- BEFORE dispatch
    print(f"DIP contexts: {len(ctxs)} MLPs converted (keep={keep_ratio})")

    max_memory = {0: f"{gpu_mem_gib}GiB", "cpu": f"{cpu_mem_gib}GiB"}
    device_map = infer_auto_device_map(
        model, max_memory=max_memory,
        no_split_module_classes=["Qwen3DecoderLayer"], dtype=torch.bfloat16)
    print(f"Device map distribution: {dict(Counter(str(v) for v in device_map.values()))}")

    state = load_file(str(ckpt_dir / "model_int4.safetensors"), device="cpu")

    def device_for(pn):
        best = None
        for mn, dev in device_map.items():
            if pn == mn or pn.startswith(mn + "."):
                if best is None or len(mn) > len(best[0]):
                    best = (mn, dev)
        return best[1] if best else "cpu"

    for pname, tensor in state.items():
        dev = device_for(pname)
        target = "cuda:0" if dev == 0 or dev == "cuda:0" else "cpu"
        set_module_tensor_to_device(model, pname, target, value=tensor)

    model = dispatch_model(model, device_map=device_map, offload_buffers=True)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tok, ctxs