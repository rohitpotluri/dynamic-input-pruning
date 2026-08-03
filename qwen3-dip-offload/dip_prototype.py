"""
dip_prototype.py — Prove DIP works in PyTorch before writing CUDA kernels.

Gate-predicted pruning per token in the MLP:
    gate = SiLU(gate_proj(x)); keep top-k channels by |gate|
    only compute up_proj[keep] and down_proj[:,keep]

FIX vs v1: we do NOT pre-dequantize/hold all 64 layers' weights (that OOMs the
T4). Instead each patched MLP dequantizes ITS weights lazily inside forward,
one layer at a time, and lets them free right after — same one-layer-resident
pattern the offloaded model already uses.

Usage:
    python dip_prototype.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 \
        --tokens 40 --keep 0.32
"""

import argparse
import torch
import torch.nn.functional as F

from qlinear import load_quantized_qwen3, dequantize, QuantLinear


def _weight(proj):
    if isinstance(proj, QuantLinear):
        return dequantize(proj.qweight, proj.scales)
    return proj.weight


def make_pruned_forward(mlp, keep_ratio):
    """
    Prune intermediate channels to top keep_ratio by |SiLU(gate(x))|.
    Weights are dequantized lazily inside forward (no hoarding).
    """
    def forward(x):
        xdt = x.dtype

        # 1) gate — dequant gate_proj only, compute, free
        gate_w = _weight(mlp.gate_proj).to(xdt)
        g = F.silu(F.linear(x, gate_w))
        del gate_w

        inter = g.shape[-1]
        k = max(1, int(round(inter * keep_ratio)))

        if keep_ratio >= 0.999:
            up_w = _weight(mlp.up_proj).to(xdt)
            u = F.linear(x, up_w); del up_w
            h = g * u
            down_w = _weight(mlp.down_proj).to(xdt)
            y = F.linear(h, down_w); del down_w
            return y

        # 2) top-k channels by |gate| per token position
        b_shape = g.shape[:-1]
        g2 = g.reshape(-1, inter)                     # [T, inter]
        T = g2.shape[0]
        idx = g2.abs().topk(k, dim=-1).indices        # [T, k]
        g_sel = torch.gather(g2, 1, idx)              # [T, k]

        x2 = x.reshape(T, -1)                         # [T, dim]

        # 3) up_proj: only kept rows. Dequant full up_w, gather rows, free.
        up_w = _weight(mlp.up_proj).to(xdt)           # [inter, dim]
        up_sel = up_w[idx]                            # [T, k, dim]
        del up_w
        u_sel = torch.einsum("tkd,td->tk", up_sel, x2)   # [T, k]
        del up_sel
        h_sel = g_sel * u_sel                         # [T, k]

        # 4) down_proj: only kept cols. Dequant, gather cols, free.
        down_w = _weight(mlp.down_proj).to(xdt)       # [dim_out, inter]
        down_sel = down_w.t()[idx]                    # [T, k, dim_out]
        del down_w
        y = torch.einsum("tk,tko->to", h_sel, down_sel)  # [T, dim_out]
        del down_sel
        return y.reshape(*b_shape, -1)

    return forward


def patch_mlps(model, keep_ratio):
    originals = {}
    for name, mod in model.named_modules():
        if all(hasattr(mod, p) for p in ("gate_proj", "up_proj", "down_proj")):
            originals[name] = mod.forward
            mod.forward = make_pruned_forward(mod, keep_ratio)
    return originals


def restore_mlps(model, originals):
    for name, mod in model.named_modules():
        if name in originals:
            mod.forward = originals[name]


def generate_text(model, tok, prompt, tokens):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=40)
    p.add_argument("--keep", type=float, default=0.32)
    p.add_argument("--prompt", type=str,
                   default="Explain the theory of relativity in detail.")
    args = p.parse_args()

    print("Loading model ...")
    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    print("\n--- REFERENCE (no pruning) ---")
    ref = generate_text(model, tok, args.prompt, args.tokens)
    print(ref)
    torch.cuda.empty_cache()

    keep = args.keep
    print(f"\n--- DIP PRUNED (keep {keep:.2f} = prune {100*(1-keep):.0f}%) ---")
    orig = patch_mlps(model, keep)
    try:
        pruned = generate_text(model, tok, args.prompt, args.tokens)
    finally:
        restore_mlps(model, orig)
    print(pruned)

    same = sum(1 for a, b in zip(ref.split(), pruned.split()) if a == b)
    denom = max(1, len(ref.split()))
    print(f"\n[rough word-match vs reference: {100*same/denom:.0f}%]")


if __name__ == "__main__":
    main()