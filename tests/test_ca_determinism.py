"""
test_ca_determinism.py — Run CA-DIP generation twice with the same config and
check the outputs are identical. Greedy decoding plus the deterministic cache
should give byte-identical text; a difference means the selection or cache is
not reproducible.

Usage:
    python -m tests.test_ca_determinism --ckpt checkpoints/qwen3-32b-int4 \
        --gpu_mem 10 --tokens 50 --keep 0.6 --cache_frac 0.3
"""
import argparse
from dip_ca import load_dip_ca, calibrate_cache
from gen_utils import generate_text, DEFAULT_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=50)
    p.add_argument("--keep", type=float, default=0.6)
    p.add_argument("--cache_frac", type=float, default=0.3)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = p.parse_args()

    model, tok, cas = load_dip_ca(args.ckpt, args.gpu_mem, args.cpu_mem,
                                  args.keep, args.cache_frac)
    model.eval()
    calibrate_cache(model, tok, cas, args.prompt, calib_tokens=30)

    print("\n=== RUN 1 ===")
    o1, _, _ = generate_text(model, tok, args.prompt, args.tokens, thinking=False)
    print(o1)
    print("\n=== RUN 2 ===")
    o2, _, _ = generate_text(model, tok, args.prompt, args.tokens, thinking=False)
    print(o2)

    print("\n" + "=" * 50)
    if o1 == o2:
        print("IDENTICAL -> deterministic")
    else:
        for i, (a, b) in enumerate(zip(o1, o2)):
            if a != b:
                print(f"diverges at char {i}: run1='{o1[i:i+20]}' vs run2='{o2[i:i+20]}'")
                break
        print("NOT IDENTICAL -> selection or cache is not reproducible")
    print("=" * 50)


if __name__ == "__main__":
    main()