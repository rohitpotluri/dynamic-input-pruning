"""
test_ca_fused.py — Check the cache-aware fused GEMV: reading each row from the
resident cache or the streamed buffer must match a single dense reference.

Run from kernels/ca_fused/ after building:
    python setup_ca_fused.py build_ext --inplace
    python test_ca_fused.py
"""
import ctypes, os, sys
import torch

here = os.path.dirname(os.path.abspath(__file__))
so = [f for f in os.listdir(here) if f.startswith("ca_fused_ops") and f.endswith(".so")]
assert so, "build the kernel first (python setup_ca_fused.py build_ext --inplace)"
ctypes.CDLL(os.path.join(here, so[0]))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
from qlinear import dequantize, GROUP_SIZE


def pack(q):
    lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
    hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def main():
    torch.manual_seed(0)
    in_f = 5120
    k = 15360
    cache_n = 7680
    assert in_f % GROUP_SIZE == 0

    # true weight for the k selected channels, used to build the reference
    q_full = torch.randint(-8, 8, (k, in_f), dtype=torch.int8)
    sc_full = (torch.rand(k, in_f // GROUP_SIZE) * 0.01 + 0.001).to(torch.bfloat16)
    x = torch.randn(in_f, dtype=torch.bfloat16).cuda()

    # split: first n_hit rows are hits (in cache), the rest are misses (streamed)
    n_hit = 5855

    # resident cache holds the hit rows at cache slots 0..n_hit-1
    res_q = pack(q_full[:n_hit]).cuda()
    res_sc = sc_full[:n_hit].cuda()
    # streamed buffer holds the miss rows at staging rows 0..(k-n_hit-1)
    str_q = pack(q_full[n_hit:]).cuda()
    str_sc = sc_full[n_hit:].cuda()

    # pad both buffers to the sizes the kernel indexes into
    str_q_full = torch.zeros(k, in_f // 2, dtype=torch.int8).cuda()
    str_sc_full = torch.zeros(k, in_f // GROUP_SIZE, dtype=torch.bfloat16).cuda()
    str_q_full[:k - n_hit] = str_q
    str_sc_full[:k - n_hit] = str_sc
    res_q_full = torch.zeros(cache_n, in_f // 2, dtype=torch.int8).cuda()
    res_sc_full = torch.zeros(cache_n, in_f // GROUP_SIZE, dtype=torch.bfloat16).cuda()
    res_q_full[:n_hit] = res_q
    res_sc_full[:n_hit] = res_sc

    # src/row tables: hits read cache slots, misses read staging positions
    src = torch.zeros(k, dtype=torch.int32)
    row = torch.zeros(k, dtype=torch.int32)
    src[:n_hit] = 0
    row[:n_hit] = torch.arange(n_hit, dtype=torch.int32)
    src[n_hit:] = 1
    row[n_hit:] = torch.arange(k - n_hit, dtype=torch.int32)
    src = src.cuda(); row = row.cuda()

    y_kernel = torch.ops.ca_fused_ops.ca_fused_gemv(
        res_q_full, res_sc_full, str_q_full, str_sc_full, x, src, row)

    # reference: dequantize the full true weight and do a dense GEMV
    w = dequantize(pack(q_full).cuda(), sc_full.cuda())
    y_ref = (x.unsqueeze(0).float() @ w.float().t()).squeeze(0)

    err = (y_kernel.float() - y_ref).abs()
    rel = err.max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"k={k}, n_hit={n_hit}, n_miss={k - n_hit}, in={in_f}")
    print(f"max abs err : {err.max().item():.4f}")
    print(f"max rel err : {rel:.4%}")
    print("PASS" if rel < 0.02 else "FAIL")


if __name__ == "__main__":
    main()