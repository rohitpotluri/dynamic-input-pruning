"""
test_int4_gemv.py — Verify the int4 GEMV kernel matches PyTorch dequant.

Loads the compiled kernel, builds a random int4-packed weight + scales + input,
runs both the kernel and the reference (qlinear.dequantize + F.linear), and
reports max/mean error. They should match closely (bf16 rounding aside).

Run from kernels/int4_gather_gemv/ after building:
    python test_int4_gemv.py
"""

import ctypes, os, sys
import torch
import torch.nn.functional as F

# load the compiled .so
here = os.path.dirname(os.path.abspath(__file__))
so = [f for f in os.listdir(here) if f.startswith("int4_ops") and f.endswith(".so")]
assert so, "build first: python int4_gemv_setup.py build_ext --inplace"
ctypes.CDLL(os.path.join(here, so[0]))

# bring in the reference dequant from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
from qlinear import dequantize, GROUP_SIZE


def pack_int4(q):  # q: int8 [out, in] in [-8,7] -> [out, in/2]
    lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
    hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def main():
    torch.manual_seed(0)
    out_f, in_f = 4096, 5120          # a realistic proj shape (in_f % 128 == 0)
    assert in_f % GROUP_SIZE == 0

    q = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8)
    qw = pack_int4(q).cuda()
    n_groups = in_f // GROUP_SIZE
    scales = (torch.rand(out_f, n_groups) * 0.01 + 0.001).to(torch.bfloat16).cuda()
    x = torch.randn(in_f, dtype=torch.bfloat16).cuda()

    # kernel
    y_kernel = torch.ops.int4_ops.int4_gemv(qw, scales, x)

    # reference
    w = dequantize(qw, scales).cuda()          # [out, in] bf16
    y_ref = F.linear(x.unsqueeze(0), w).squeeze(0)

    err = (y_kernel.float() - y_ref.float()).abs()
    print(f"shape out={out_f} in={in_f}")
    print(f"max abs err : {err.max().item():.4f}")
    print(f"mean abs err: {err.mean().item():.5f}")
    rel = err.max().item() / (y_ref.float().abs().max().item() + 1e-6)
    print(f"max rel err : {rel:.4%}")
    print("PASS" if rel < 0.02 else "CHECK — error higher than expected")


if __name__ == "__main__":
    main()