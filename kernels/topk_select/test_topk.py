"""
test_topk.py — Check that the threshold-based top-k kernel selects essentially
the same channel set as torch.topk (order-independent).

DIP tolerates a few differences from threshold granularity and ties, so the
test checks that the overlap between the kernel and torch selections is high.

Run from kernels/topk_select/ after building:
    python setup_topk.py build_ext --inplace
    python test_topk.py
"""
import ctypes, os
import torch

here = os.path.dirname(os.path.abspath(__file__))
so = [f for f in os.listdir(here) if f.startswith("topk_ops") and f.endswith(".so")]
assert so, "build first: python setup_topk.py build_ext --inplace"
ctypes.CDLL(os.path.join(here, so[0]))


def main():
    torch.manual_seed(0)
    inter = 25600
    k = int(inter * 0.6)              # 15360, the DIP keep count
    s = torch.randn(inter, dtype=torch.bfloat16).cuda()

    idx_kernel = torch.ops.topk_ops.topk_select(s, k)
    set_kernel = set(idx_kernel.cpu().tolist())

    # reference: torch.topk on |s|
    idx_ref = s.abs().topk(k).indices
    set_ref = set(idx_ref.cpu().tolist())

    overlap = len(set_kernel & set_ref)
    print(f"inter={inter}, k={k}")
    print(f"kernel selected: {len(set_kernel)}")
    print(f"overlap with torch.topk: {overlap} / {k} = {100 * overlap / k:.2f}%")
    print("PASS" if overlap / k > 0.98 else "FAIL")


if __name__ == "__main__":
    main()