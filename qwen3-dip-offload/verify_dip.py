"""
verify_dip.py — Confirm DIP + selective streaming is REALLY happening, and
generate a longer sample to judge coherence.

Checks:
  1. Are the MLPs actually StreamingDIPMlp? (not the original)
  2. Instrument one MLP: count how many times the streaming (decode) path runs
     vs the dense fallback, and confirm the gather kernel is called + how many
     rows are streamed (k) vs total (inter).
  3. Print GPU memory (should leave room, since up/down live on CPU).
  4. Generate a LONG output (many tokens) to read for coherence.

Usage:
    python verify_dip.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 \
        --tokens 120 --keep 0.32
"""
import argparse, time
import torch
from dip_stream_predispatch import load_streaming_dip, StreamingDIPMlp, _use_gather


# global counters
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
    p.add_argument("--keep", type=float, default=0.32)
    p.add_argument("--prompt", default="Explain the theory of relativity in detail, covering both special and general relativity.")
    args = p.parse_args()

    print(f"gather kernel loaded: {_use_gather}")
    model, tok, sdips = load_streaming_dip(
        args.ckpt, args.gpu_mem, args.cpu_mem, args.keep)
    model.eval()

    # 1) confirm MLPs are StreamingDIPMlp
    n_sdip = sum(1 for m in sdips if isinstance(m, StreamingDIPMlp))
    print(f"\n[1] StreamingDIPMlp count: {n_sdip} (expect 64)")

    # 2) instrument
    instrument(sdips)

    # 3) GPU memory
    print(f"[3] GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB "
          f"(up/down should be on CPU, so this leaves headroom)")

    # 4) long generation
    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    ids = ids.to("cuda:0")

    print(f"\nGenerating {args.tokens} tokens ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    ntok = out.shape[-1] - ids.shape[-1]

    print("\n" + "=" * 60)
    print("PATH USAGE (proves DIP streaming is active):")
    print(f"  decode (streaming) calls : {STATS['decode_calls']}")
    print(f"  dense (fallback) calls   : {STATS['dense_calls']}")
    print(f"  gather kernel calls      : {STATS['kernel_calls']}")
    if STATS["rows_total"]:
        pct = 100 * STATS["rows_streamed"] / STATS["rows_total"]
        print(f"  rows streamed / total    : {STATS['rows_streamed']} / "
              f"{STATS['rows_total']}  ({pct:.1f}% streamed, "
              f"{100-pct:.1f}% skipped)")
    print("=" * 60)
    print(f"Throughput: {ntok} tok in {dt:.1f}s -> {ntok/dt:.3f} tok/s")
    print("=" * 60)
    print("\nFULL OUTPUT:\n")
    print(txt)


if __name__ == "__main__":
    main()