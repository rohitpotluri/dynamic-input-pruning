"""
test_dip_ca.py — Run CA-DIP (resident cache + stream the misses) and report
decode throughput.

Usage:
    python -m tests.test_dip_ca --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 \
        --tokens 60 --keep 0.6 --cache_frac 0.3
"""
import argparse
import torch
from dip_ca import load_dip_ca, calibrate_cache
from gen_utils import generate_text, DEFAULT_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=60)
    p.add_argument("--keep", type=float, default=0.6)
    p.add_argument("--cache_frac", type=float, default=0.3)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = p.parse_args()

    model, tok, cas = load_dip_ca(args.ckpt, args.gpu_mem, args.cpu_mem,
                                  args.keep, args.cache_frac)
    model.eval()
    print(f"GPU after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    calibrate_cache(model, tok, cas, args.prompt, calib_tokens=30)
    print(f"GPU after cache load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("Warmup ...")
    generate_text(model, tok, args.prompt, 2, thinking=False)
    print("Timed run ...")
    txt, ntok, dt = generate_text(model, tok, args.prompt, args.tokens, thinking=False)

    print(f"\nOutput:\n{txt}")
    print("\n" + "=" * 50)
    print(f"CA-DIP (cache_frac={args.cache_frac}): {ntok} tok in {dt:.2f}s "
          f"-> {ntok / dt:.3f} tok/s")
    print("=" * 50)


if __name__ == "__main__":
    main()