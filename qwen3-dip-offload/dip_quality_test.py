"""
dip_quality_test.py — Verify DIP pruning preserves output quality, on a model
that fits ENTIRELY in VRAM (no accelerate offload, no hooks to collide with).

We load Qwen3-8B-int4 fully onto the GPU, then compare:
  * reference output (no pruning)
  * DIP-pruned output at several keep-ratios

Since the same SwiGLU MLP + sparsity structure is shared with Qwen3-32B, a
quality result here transfers: if 68% pruning reads fine on 8B, it will on 32B.

This directly answers the go/no-go question for the kernels:
    "Does keeping only the top-K% of MLP channels per token still produce
     coherent text?"

Usage:
    python dip_quality_test.py --ckpt checkpoints/qwen3-8b-int4 \
        --tokens 60 --keeps 1.0 0.5 0.32 0.2
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlinear import QuantLinear, dequantize, _replace_linears_with_quant


def load_full_gpu(ckpt_dir):
    """Load the int4 model fully onto the GPU (no offload)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file
    from pathlib import Path

    ckpt_dir = Path(ckpt_dir)
    config = AutoConfig.from_pretrained(ckpt_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    _replace_linears_with_quant(model)

    state = load_file(str(ckpt_dir / "model_int4.safetensors"), device="cpu")
    for pname, t in state.items():
        set_module_tensor_to_device(model, pname, "cuda:0", value=t)

    model.eval()
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tok


def _weight(proj):
    if isinstance(proj, QuantLinear):
        return dequantize(proj.qweight, proj.scales)
    return proj.weight


def make_pruned_forward(mlp, keep_ratio):
    def forward(x):
        xdt = x.dtype
        gate_w = _weight(mlp.gate_proj).to(xdt)
        g = F.silu(F.linear(x, gate_w)); del gate_w
        inter = g.shape[-1]

        up_w = _weight(mlp.up_proj).to(xdt)
        u = F.linear(x, up_w); del up_w
        h = g * u                                    # full intermediate

        if keep_ratio < 0.999:
            k = max(1, int(round(inter * keep_ratio)))
            h2 = h.reshape(-1, inter)
            # zero out all but top-k channels by |h| per token
            thresh = h2.abs().topk(k, dim=-1).values[:, -1:].clamp(min=0)
            mask = (h2.abs() >= thresh)
            h2 = h2 * mask
            h = h2.reshape(h.shape)

        down_w = _weight(mlp.down_proj).to(xdt)
        y = F.linear(h, down_w); del down_w
        return y
    return forward


def patch(model, keep):
    orig = {}
    for name, mod in model.named_modules():
        if all(hasattr(mod, p) for p in ("gate_proj", "up_proj", "down_proj")):
            orig[name] = mod.forward
            mod.forward = make_pruned_forward(mod, keep)
    return orig


def unpatch(model, orig):
    for name, mod in model.named_modules():
        if name in orig:
            mod.forward = orig[name]


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
    p.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5, 0.32, 0.2])
    p.add_argument("--prompt", type=str,
                   default="Explain the theory of relativity in detail.")
    args = p.parse_args()

    print("Loading Qwen3-8B-int4 fully on GPU ...")
    model, tok = load_full_gpu(args.ckpt)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    ref = None
    for keep in args.keeps:
        label = "REFERENCE (no prune)" if keep >= 0.999 else f"keep {keep:.2f} (prune {100*(1-keep):.0f}%)"
        print("=" * 60)
        print(label)
        print("=" * 60)
        if keep >= 0.999:
            txt = gen(model, tok, args.prompt, args.tokens)
            ref = txt
        else:
            o = patch(model, keep)
            try:
                txt = gen(model, tok, args.prompt, args.tokens)
            finally:
                unpatch(model, o)
        print(txt)
        if ref and keep < 0.999:
            same = sum(1 for a, b in zip(ref.split(), txt.split()) if a == b)
            print(f"\n[word-match vs reference: {100*same/max(1,len(ref.split())):.0f}%]")
        print()


if __name__ == "__main__":
    main()