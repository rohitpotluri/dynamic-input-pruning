"""
gate_kernel.py — Loads the int4 dequant-GEMV kernel (int4_ops) and provides
gate_gemv() for the MLP gate projection.

Gate is SiLU(gate_proj(x)) where gate_proj is an int4 g128 linear, so it reuses
the int4_gemv kernel: the gate projection runs on the CUDA kernel rather than
dequantize + F.linear.
"""
import ctypes, os
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_so_dir = os.path.join(_here, "kernels", "int4_gemv")
_use_gate = False
if os.path.isdir(_so_dir):
    sos = [f for f in os.listdir(_so_dir)
           if f.startswith("int4_ops") and f.endswith(".so")]
    if sos:
        ctypes.CDLL(os.path.join(_so_dir, sos[0]))
        try:
            @torch.library.register_fake("int4_ops::int4_gemv")
            def _(qweight, scales, x):
                return torch.empty(qweight.shape[0], dtype=torch.bfloat16, device=x.device)
            _use_gate = True
        except Exception:
            pass


def gate_gemv(qweight, scales, x):
    """
    Compute gate_proj(x) via the int4 kernel. x is [in] (decode, seqlen 1);
    returns [out] bf16.
    """
    return torch.ops.int4_ops.int4_gemv(qweight, scales, x.to(torch.bfloat16).contiguous())