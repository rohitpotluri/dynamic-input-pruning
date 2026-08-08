"""
test_stream_predispatch.py — Selective-streaming DIP built pre-dispatch.
Usage:
    python test_stream_predispatch.py --ckpt checkpoints/qwen3-32b-int4 \
        --gpu_mem 10 --tokens 8 --keep 0.32
"""
import argparse, time
import torch
from dip_stream_predispatch import load_streaming_dip, _use_gather


def gen_timed(model, tok, prompt, tokens):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    ids = ids.to("cuda:0")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return txt, out.shape[-1] - ids.shape[-1], dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--keep", type=float, default=0.32)
    p.add_argument("--prompt", default="Explain gravity in simple terms.")
    args = p.parse_args()

    print(f"gather kernel loaded: {_use_gather}")
    model, tok, sdips = load_streaming_dip(
        args.ckpt, args.gpu_mem, args.cpu_mem, args.keep)
    model.eval()

    print("Warmup ...")
    gen_timed(model, tok, args.prompt, 2)
    print("Timed run ...")
    txt, ntok, dt = gen_timed(model, tok, args.prompt, args.tokens)
    print(f"\nOutput:\n{txt}")
    print("\n" + "=" * 50)
    print(f"Streaming DIP (pre-dispatch): {ntok} tok in {dt:.2f}s -> {ntok/dt:.3f} tok/s")
    print(f"(baseline 0.088)")
    print("=" * 50)


if __name__ == "__main__":
    main()