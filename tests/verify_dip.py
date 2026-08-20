"""
verify_dip.py — Confirm selective-streaming DIP is active and generate a longer
sample to judge coherence.

Checks:
  1. the MLPs are StreamingDIPMlp instances
  2. instruments each MLP to count how often the streaming (decode) path runs
     vs the dense fallback, how often the gather kernel is called, and how many
     rows are streamed (k) vs total (inter)
  3. prints GPU memory (up/down live on CPU, so this should leave headroom)
  4. generates a longer output to read for coherence

Usage:
    python -m tests.verify_dip --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 \
        --tokens 120 --keep 0.6
"""
import argparse, time
import torch
from dip_stream_predispatch import load_streaming_dip, StreamingDIPMlp, _use_gather
from gen_utils import build_inputs, DEFAULT_PROMPT

STATS = {"decode_calls": 0, "dense_calls": 0, "rows_streamed": 0,
         "rows_total": 0, "kernel_calls": 0}


def instrument(sdips):
    """Wrap each StreamingDIPMlp.forward to count path usage."""
    for m in sdips:
        orig = m.forward
        def make(m, orig):
            def wrapped(x):
                is_decode = (x.dim() == 3 and x.shape[1] == 1
                             and m.keep_ratio < 0.999 and _use_gather)
                if is_decode:
                    STATS["decode_calls"] += 1
                    STATS["rows_streamed"] += m.k
                    STATS["rows_total"] += m.inter
                    STATS["kernel_calls"] += 1
                else:
                    STATS["dense_calls"] += 1
                return orig(x)
            return wrapped
        m.forward = make(m, orig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--keep", type=float, default=0.6)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = p.parse_args()

    print(f"gather kernel loaded: {_use_gather}")
    model, tok, sdips = load_streaming_dip(
        args.ckpt, args.gpu_mem, args.cpu_mem, args.keep)
    model.eval()

    n_sdip = sum(1 for m in sdips if isinstance(m, StreamingDIPMlp))
    print(f"\n[1] StreamingDIPMlp count: {n_sdip}")

    instrument(sdips)

    print(f"[3] GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
          f"(up/down are on CPU, so this leaves headroom)")

    ids = build_inputs(tok, args.prompt, thinking=False)

    print(f"\nGenerating {args.tokens} tokens ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    ntok = out.shape[-1] - ids.shape[-1]

    print("\n" + "=" * 60)
    print("Path usage:")
    print(f"  decode (streaming) calls : {STATS['decode_calls']}")
    print(f"  dense (fallback) calls   : {STATS['dense_calls']}")
    print(f"  gather kernel calls      : {STATS['kernel_calls']}")
    if STATS["rows_total"]:
        pct = 100 * STATS["rows_streamed"] / STATS["rows_total"]
        print(f"  rows streamed / total    : {STATS['rows_streamed']} / "
              f"{STATS['rows_total']}  ({pct:.1f}% streamed, "
              f"{100 - pct:.1f}% skipped)")
    print("=" * 60)
    print(f"Throughput: {ntok} tok in {dt:.1f}s -> {ntok / dt:.3f} tok/s")
    print("=" * 60)
    print("\nOutput:\n")
    print(txt)


if __name__ == "__main__":
    main()