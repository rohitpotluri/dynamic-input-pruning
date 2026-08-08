"""
bench_selective_stream.py — The load-bearing measurement.

Question: on THIS T4 / PCIe-x8 link, to get 32% of an up_proj's rows onto the
GPU, is it faster to:

  (A) BULK      : stream the whole weight (100%) in one contiguous copy
                  (what accelerate does now)
  (B) GATHER    : index the 32% selected rows on CPU -> contiguous buffer,
                  then one transfer of just those rows
  (C) PINNED-B  : same as B but the CPU source is pinned (page-locked)
  (D) ROWWISE   : copy each selected row individually (many small transfers)
                  -- included to show why scattered small copies are bad

If (B)/(C) beat (A), selective streaming can win and we build it.
If not, selective streaming won't help on this hardware and we rethink.

Uses one realistic up_proj shape for Qwen3-32B int4:
  qweight int8 [25600, 2560]  (packed int4)  ~ 65 MB
Simulated over many iters to average out noise.

Usage:
    python bench_selective_stream.py --keep 0.32 --iters 50
"""

import argparse, time
import torch


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep", type=float, default=0.32)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=int, default=25600)   # up_proj rows (inter)
    p.add_argument("--inpacked", type=int, default=2560)  # in_features/2
    args = p.parse_args()

    out_f, in_p = args.out, args.inpacked
    k = int(round(out_f * args.keep))
    dev = "cuda:0"

    # CPU source weight (int8 packed), pageable and pinned versions
    w_cpu = torch.randint(-128, 127, (out_f, in_p), dtype=torch.int8)
    w_cpu_pinned = w_cpu.pin_memory()

    # a fixed random selection of k rows (sorted, like real top-k output)
    idx = torch.randperm(out_f)[:k].sort().values

    # preallocated GPU destinations (reused, no per-iter malloc)
    dst_full = torch.empty(out_f, in_p, dtype=torch.int8, device=dev)
    dst_sel = torch.empty(k, in_p, dtype=torch.int8, device=dev)
    # pinned staging for the gathered rows (CPU side)
    gather_buf = torch.empty(k, in_p, dtype=torch.int8).pin_memory()

    bytes_full = w_cpu.numel()
    bytes_sel = k * in_p

    # (A) BULK: whole weight, pageable
    def bulk():
        dst_full.copy_(w_cpu, non_blocking=False)

    # (B) GATHER pageable: index rows on CPU then one transfer
    def gather_pageable():
        sel = w_cpu[idx]                    # CPU gather -> new contiguous tensor
        dst_sel.copy_(sel, non_blocking=False)

    # (C) GATHER pinned: gather into pinned buffer, async transfer
    def gather_pinned():
        torch.index_select(w_cpu_pinned, 0, idx, out=gather_buf)
        dst_sel.copy_(gather_buf, non_blocking=True)

    # (D) ROWWISE: copy each selected row separately (worst case)
    def rowwise():
        for j in range(k):
            dst_sel[j].copy_(w_cpu[idx[j]], non_blocking=True)

    tA = timeit(bulk, args.iters)
    tB = timeit(gather_pageable, args.iters)
    tC = timeit(gather_pinned, args.iters)
    tD = timeit(rowwise, max(5, args.iters // 10))  # rowwise is slow; fewer iters

    def gbps(nbytes, t): return nbytes / 1e9 / t

    print("=" * 62)
    print(f"up_proj [{out_f}, {in_p}] int8 = {bytes_full/1e6:.0f} MB full, "
          f"keep {args.keep:.0%} -> {bytes_sel/1e6:.0f} MB selected")
    print("=" * 62)
    print(f"(A) BULK   100% pageable : {tA*1e3:7.2f} ms  | {gbps(bytes_full,tA):5.1f} GB/s")
    print(f"(B) GATHER  32% pageable : {tB*1e3:7.2f} ms  | {gbps(bytes_sel,tB):5.1f} GB/s")
    print(f"(C) GATHER  32% pinned   : {tC*1e3:7.2f} ms  | {gbps(bytes_sel,tC):5.1f} GB/s")
    print(f"(D) ROWWISE 32% (small)  : {tD*1e3:7.2f} ms  | {gbps(bytes_sel,tD):5.1f} GB/s")
    print("-" * 62)
    print(f"Best selective vs BULK   : {tA/min(tB,tC):.2f}x faster"
          if min(tB, tC) < tA else
          f"Selective NOT faster (best {min(tB,tC)*1e3:.2f}ms vs bulk {tA*1e3:.2f}ms)")
    print("=" * 62)
    print("\nVERDICT:")
    if min(tB, tC) < tA * 0.9:
        print("  Selective streaming WINS -> build it.")
    else:
        print("  Selective streaming does NOT clearly beat bulk on this link.")
        print("  The CPU-side gather cost may be eating the transfer savings.")
        print("  -> rethink (e.g. resident hot rows via DIP-CA instead of per-token gather).")


if __name__ == "__main__":
    main()