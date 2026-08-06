"""
test_dip_mlp.py — End-to-end DIP MLP test on Qwen3-8B-int4 (fits in VRAM).

Compares:
  * reference (dense, no prune)
  * DIP MLP using the real gather kernel at keep=0.32

Confirms the kernel-in-the-loop path produces coherent, on-topic output.

Usage:
    python test_dip_mlp.py --ckpt checkpoints/qwen3-8b-int4 --tokens 60 --keep 0.32
"""
import argparse
import torch

from dip_quality_test import load_full_gpu       # reuse the full-GPU loader
from dip_mlp import patch_dip, unpatch_dip, _use_gather


def gen(model, tok, prompt, tokens):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-8b-int4")
    p.add_argument("--tokens", type=int, default=60)
    p.add_argument("--keep", type=float, default=0.32)
    p.add_argument("--prompt", type=str,
                   default="Explain the theory of relativity in detail.")
    args = p.parse_args()

    print(f"gather kernel loaded: {_use_gather}")
    assert _use_gather, "gather kernel .so not found — build it first"

    print("Loading Qwen3-8B-int4 fully on GPU ...")
    model, tok = load_full_gpu(args.ckpt)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    print("=" * 60); print("REFERENCE (dense)"); print("=" * 60)
    print(gen(model, tok, args.prompt, args.tokens))

    print("\n" + "=" * 60)
    print(f"DIP MLP via kernel (keep {args.keep:.2f})")
    print("=" * 60)
    w = patch_dip(model, args.keep)
    try:
        print(gen(model, tok, args.prompt, args.tokens))
    finally:
        unpatch_dip(model, w)


if __name__ == "__main__":
    main()