"""
dip_mlp.py — DIP MLP module using the real gather+int4 GEMV kernel.

Per token, decode path (seqlen == 1):
  1. gate = SiLU(gate_proj(x))            [full, PyTorch]  -> importance signal
  2. idx  = top-k channels by |gate|      [PyTorch topk]
  3. up_sel = gather_gemv(up_proj, x)[idx]  [YOUR KERNEL] -> only kept rows
  4. h_sel  = gate[idx] * up_sel
  5. y = sum_k h_sel[k] * down_proj[:, idx[k]]   [PyTorch col-select for now]

Only up_proj uses the custom kernel here; down_proj column-gather is a later
kernel. This proves the gate->select->gather-gemv path is correct end-to-end.

Falls back to the full dense MLP for prefill (seqlen > 1) and if the kernel
isn't loaded.
"""

import ctypes, os
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlinear import QuantLinear, dequantize

# load the gather kernel if present
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


def _w(proj):
    return dequantize(proj.qweight, proj.scales) if isinstance(proj, QuantLinear) else proj.weight


class DIPMlp(nn.Module):
    """Wraps a Qwen3 MLP (gate_proj/up_proj/down_proj) with DIP pruning."""

    def __init__(self, mlp, keep_ratio=0.32):
        super().__init__()
        self.mlp = mlp
        self.keep_ratio = keep_ratio

    def forward(self, x):
        # only prune single-token decode; prefill uses dense path
        if not _use_gather or x.dim() != 3 or x.shape[1] != 1 or self.keep_ratio >= 0.999:
            return self._dense(x)

        gp, up, dn = self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj
        # only support the quantized kernel path when up_proj is QuantLinear
        if not isinstance(up, QuantLinear):
            return self._dense(x)

        xdt = x.dtype
        x_flat = x.reshape(-1)                       # [in]  (bsz=1, seq=1)

        # 1) gate (full) -> importance
        gate_w = _w(gp).to(xdt)
        g = F.silu(F.linear(x_flat.unsqueeze(0), gate_w)).squeeze(0)   # [inter]
        del gate_w
        inter = g.shape[-1]
        k = max(1, int(round(inter * self.keep_ratio)))

        # 2) top-k channels by |gate|
        idx = g.abs().topk(k).indices.to(torch.int32)        # [k]

        # 3) up_proj selected rows via YOUR kernel
        up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(
            up.qweight, up.scales, x_flat.to(torch.bfloat16), idx)   # [k]

        # 4) combine with gate
        h_sel = g[idx.long()] * up_sel.to(xdt)               # [k]

        # 5) down_proj: only selected columns (PyTorch for now)
        dn_w = _w(dn).to(xdt)                                 # [out, inter]
        dn_sel = dn_w[:, idx.long()]                         # [out, k]
        del dn_w
        y = F.linear(h_sel.unsqueeze(0), dn_sel).squeeze(0)  # [out]
        return y.reshape(x.shape[0], 1, -1)

    def _dense(self, x):
        gp, up, dn = self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj
        xdt = x.dtype
        g = F.silu(F.linear(x, _w(gp).to(xdt)))
        u = F.linear(x, _w(up).to(xdt))
        h = g * u
        return F.linear(h, _w(dn).to(xdt))


def patch_dip(model, keep_ratio=0.32):
    """Replace every decoder MLP's forward with a DIP wrapper."""
    wrappers = {}
    for name, mod in model.named_modules():
        if all(hasattr(mod, p) for p in ("gate_proj", "up_proj", "down_proj")):
            w = DIPMlp(mod, keep_ratio)
            wrappers[name] = mod.forward
            mod.forward = w.forward
    return wrappers


def unpatch_dip(model, wrappers):
    for name, mod in model.named_modules():
        if name in wrappers:
            mod.forward = wrappers[name]