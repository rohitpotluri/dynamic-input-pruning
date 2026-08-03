"""
profile_baseline.py — Operator-level profiling of the offloaded int4 baseline,
using torch.profiler (sees through accelerate's hooks).

Tells you which OPERATIONS dominate per-token time:
  * aten::copy_ / aten::to / Memcpy HtoD -> PCIe streaming (offload cost)
  * bitwise/mul/where (int4 unpack) -> dequant work
  * aten::mm / addmm -> the matmuls
  * attention / softmax kernels

Usage:
    python profile_baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 4
"""

import argparse
import time
import torch
from torch.profiler import profile, ProfilerActivity

from qlinear import load_quantized_qwen3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--prompt", type=str, default="Explain gravity in simple terms.")
    args = p.parse_args()

    print("Loading ...")
    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to("cuda:0")

    gen = dict(max_new_tokens=args.tokens, do_sample=False,
               use_cache=True, pad_token_id=tok.eos_token_id)

    print("Warmup ...")
    with torch.no_grad():
        _ = model.generate(ids, **gen)
    torch.cuda.synchronize()

    print(f"Profiling {args.tokens} tokens ...")
    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        with torch.no_grad():
            out = model.generate(ids, **gen)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new_tok = out.shape[-1] - ids.shape[-1]

    print("\n" + "=" * 60)
    print(f"Wall: {wall:.2f}s | {new_tok} tok | {new_tok/wall:.3f} tok/s")
    print("=" * 60)

    print("\n--- TOP OPS BY TOTAL CUDA TIME ---")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25))

    print("\n--- TOP OPS BY TOTAL CPU TIME ---")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=25))

    prof.export_chrome_trace("baseline_trace.json")
    print("\nFull trace -> baseline_trace.json")


if __name__ == "__main__":
    main()