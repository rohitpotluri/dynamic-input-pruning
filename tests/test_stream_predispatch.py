"""
test_stream_predispatch.py — Run selective-streaming DIP (built pre-dispatch)
and report decode throughput.

Usage:
    python -m tests.test_stream_predispatch --ckpt checkpoints/qwen3-32b-int4 \
        --gpu_mem 10 --tokens 30 --keep 0.6
"""
import argparse
from dip_stream_predispatch import load_streaming_dip, _use_gather
from gen_utils import generate_text, DEFAULT_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=30)
    p.add_argument("--keep", type=float, default=0.6)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = p.parse_args()

    print(f"gather kernel loaded: {_use_gather}")
    model, tok, sdips = load_streaming_dip(
        args.ckpt, args.gpu_mem, args.cpu_mem, args.keep)
    model.eval()

    print("Warmup ...")
    generate_text(model, tok, args.prompt, 2, thinking=False)
    print("Timed run ...")
    txt, ntok, dt = generate_text(model, tok, args.prompt, args.tokens, thinking=False)

    print(f"\nOutput:\n{txt}")
    print("\n" + "=" * 50)
    print(f"Streaming DIP: {ntok} tok in {dt:.2f}s -> {ntok / dt:.3f} tok/s")
    print("=" * 50)


if __name__ == "__main__":
    main()