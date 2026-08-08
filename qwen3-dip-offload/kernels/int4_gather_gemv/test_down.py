"""
test_down.py — Verify the down_proj gather kernel.

Reference: y[o] = sum_j h_sel[j] * down_t[idx[j], o]  (dequantized).
We stream selected rows of down_t (int4) into a contiguous buffer, run the
kernel, and compare to a bf16 reference.

Run from kernels/int4_gather_gemv/ after building:
    python setup_down.py build_ext --inplace
    python test_down.py
"""
import ctypes, os, sys
import torch

here = os.path.dirname(os.path.abspath(__file__))
so = [f for f in os.listdir(here) if f.startswith("int4_down_ops") and f.endswith(".so")]
assert so, "build first: python setup_down.py build_ext --inplace"
ctypes.CDLL(os.path.join(here, so[0]))

sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
from qlinear import dequantize, GROUP_SIZE


def pack_int4(q):
    lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
    hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def main():
    torch.manual_seed(0)
    inter, out_f = 25600, 5120           # down_t is [inter, out]
    assert out_f % GROUP_SIZE == 0
    k = int(inter * 0.6)                  # selected channels

    # random down_t (transposed down), int4
    q = torch.randint(-8, 8, (k, out_f), dtype=torch.int8)   # already the k selected rows
    dtw = pack_int4(q).cuda()
    scales = (torch.rand(k, out_f // GROUP_SIZE) * 0.01 + 0.001).to(torch.bfloat16).cuda()
    h_sel = torch.randn(k, dtype=torch.bfloat16).cuda()

    # kernel
    y_kernel = torch.ops.int4_down_ops.int4_gather_down(dtw, scales, h_sel)

    # reference
    w = dequantize(dtw, scales).cuda()        # [k, out] bf16
    y_ref = (h_sel.float().unsqueeze(0) @ w.float()).squeeze(0)

    err = (y_kernel.float() - y_ref).abs()
    print(f"k={k} out={out_f}")
    print(f"max abs err : {err.max().item():.4f}")
    print(f"mean abs err: {err.mean().item():.5f}")
    rel = err.max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"max rel err : {rel:.4%}")
    print("PASS" if rel < 0.02 else "CHECK")


if __name__ == "__main__":
    main()