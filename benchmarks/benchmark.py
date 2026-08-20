"""
benchmark.py — Steady-state decode throughput for one config, warmup excluded.

Runs one config per invocation (clean memory, no cross-contamination):
    python benchmarks/benchmark.py --config baseline --ckpt checkpoints/qwen3-32b-int4
    python benchmarks/benchmark.py --config streaming --ckpt checkpoints/qwen3-32b-int4
    python benchmarks/benchmark.py --config ca --ckpt checkpoints/qwen3-32b-int4

Collect the three printed lines into the results table.
"""
import argparse, sys, os
# run as a plain script from benchmarks/, so add the repo root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from gen_utils import benchmark_decode, DEFAULT_PROMPT


def load_config(cfg, ckpt, gpu_mem, cpu_mem, keep, cache_frac):
    if cfg == "baseline":
        from qlinear import load_quantized_qwen3
        model, tok = load_quantized_qwen3(ckpt, gpu_mem, cpu_mem)
        return model, tok
    if cfg == "streaming":
        from dip_stream_predispatch import load_streaming_dip
        model, tok, _ = load_streaming_dip(ckpt, gpu_mem, cpu_mem, keep)
        return model, tok
    if cfg == "ca":
        from dip_ca import load_dip_ca, calibrate_cache
        model, tok, cas = load_dip_ca(ckpt, gpu_mem, cpu_mem, keep, cache_frac)
        calibrate_cache(model, tok, cas, DEFAULT_PROMPT, calib_tokens=30)
        return model, tok
    raise ValueError(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=["baseline", "streaming", "ca"])
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--keep", type=float, default=0.6)
    p.add_argument("--cache_frac", type=float, default=0.3)
    p.add_argument("--warmup_tokens", type=int, default=8)
    p.add_argument("--timed_tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    print(f"=== Benchmark: {args.config} ===")
    model, tok = load_config(args.config, args.ckpt, args.gpu_mem, args.cpu_mem,
                             args.keep, args.cache_frac)
    model.eval()

    res = benchmark_decode(model, tok, prompt=DEFAULT_PROMPT,
                           warmup_tokens=args.warmup_tokens,
                           timed_tokens=args.timed_tokens, runs=args.runs,
                           thinking=False)

    print("\n" + "=" * 55)
    print(f"config: {args.config}")
    print(f"keep={args.keep}  cache_frac={args.cache_frac if args.config == 'ca' else '-'}")
    print(f"mean throughput: {res['mean_tps']:.3f} +/- {res['std_tps']:.3f} tok/s "
          f"(over {args.runs} runs of {args.timed_tokens} tok, warmup excluded)")
    print("=" * 55)
    print(f"\nSample output:\n{res['sample']}")


if __name__ == "__main__":
    main()