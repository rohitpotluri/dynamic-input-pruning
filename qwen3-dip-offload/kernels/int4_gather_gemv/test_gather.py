"""
test_gather.py — Verify the gather+int4-GEMV kernel.

Checks that computing only the selected rows matches the reference full GEMV
restricted to those same rows. Run from kernels/int4_gather_gemv/ after build:
    python setup_gather.py build_ext --inplace
    python test_gather.py
"""
import ctypes, os, sys
import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
so = [f for f in os.listdir(here) if f.startswith("int4_gather_ops") and f.endswith(".so")]
assert so, "build first: python setup_gather.py build_ext --inplace"
ctypes.CDLL(os.path.join(here, so[0]))

sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
from qlinear import dequantize, GROUP_SIZE


def pack_int4(q):
    lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
    hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def main():
    torch.manual_seed(0)
    out_f, in_f = 4096, 5120
    assert in_f % GROUP_SIZE == 0

    q = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8)
    qw = pack_int4(q).cuda()
    scales = (torch.rand(out_f, in_f // GROUP_SIZE) * 0.01 + 0.001).to(torch.bfloat16).cuda()
    x = torch.randn(in_f, dtype=torch.bfloat16).cuda()

    # DIP keeps 32% of rows, chosen randomly here (real DIP picks by gate)
    k = int(out_f * 0.32)
    idx = torch.randperm(out_f)[:k].sort().values.to(torch.int32).cuda()

    # kernel: only selected rows
    y_kernel = torch.ops.int4_gather_ops.int4_gather_gemv(qw, scales, x, idx)

    # reference: full dequant GEMV, then pick same rows
    w = dequantize(qw, scales).cuda()
    y_full = F.linear(x.unsqueeze(0), w).squeeze(0)
    y_ref = y_full[idx.long()]

    err = (y_kernel.float() - y_ref.float()).abs()
    print(f"out={out_f} in={in_f} | selected k={k} ({100*k/out_f:.0f}%)")
    print(f"max abs err : {err.max().item():.4f}")
    print(f"mean abs err: {err.mean().item():.5f}")
    rel = err.max().item() / (y_ref.float().abs().max().item() + 1e-6)
    print(f"max rel err : {rel:.4%}")
    print("PASS" if rel < 0.02 else "CHECK — error higher than expected")


if __name__ == "__main__":
    main()