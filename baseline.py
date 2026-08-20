"""
baseline.py — Offloaded inference baseline for the int4 Qwen3-32B on a single T4.

This is the throughput the DIP / CA-DIP configurations are compared against. It
uses the same int4 g128 weights, the dense dequant path in QuantLinear.forward
(no gather/GEMV kernels), and accelerate's stock offload (no DIP selection), so
the difference between this and the DIP configurations isolates the effect of
pruning + selective streaming + the kernels, not quantization.

Usage:
    python baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10
"""

import argparse
import time

import torch

from qlinear import load_quantized_qwen3
from gen_utils import benchmark_decode, DEFAULT_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_runs", type=int, default=3)
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    print("Loading int4 Qwen3-32B with offload ...")
    t0 = time.time()
    model, tok = load_quantized_qwen3(
        args.ckpt, gpu_mem_gib=args.gpu_mem, cpu_mem_gib=args.cpu_mem
    )
    model.eval()
    print(f"Load time: {time.time() - t0:.1f}s")

    res = benchmark_decode(model, tok, prompt=args.prompt,
                           timed_tokens=args.max_new_tokens, runs=args.num_runs,
                           thinking=False)

    print("\n" + "=" * 50)
    print(f"Baseline throughput: {res['mean_tps']:.3f} +/- {res['std_tps']:.3f} tok/s")
    print("=" * 50)
    print("\nSample output:")
    print(res["sample"])


if __name__ == "__main__":
    main()