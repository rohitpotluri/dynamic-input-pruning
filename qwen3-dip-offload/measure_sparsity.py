"""
measure_sparsity.py — How prunable is Qwen3-32B's MLP, per token?

DIP's whole premise: for a given token, the MLP intermediate activation
(after SiLU(gate) * up) has many near-zero channels. A near-zero intermediate
channel means the corresponding up_proj row and down_proj column didn't matter
for this token -> we could have skipped streaming them.

This script hooks every decoder layer's MLP, captures the intermediate
activation during real generation, and reports what fraction of the ~25600
intermediate channels are effectively zero per token.

That fraction is your DIP ceiling:
    70%+ dead -> big win, build DIP
    40-60%    -> moderate win
    <20%      -> DIP can't save enough, rethink

Usage:
    python measure_sparsity.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 8
"""

import argparse
import torch
import torch.nn.functional as F

from qlinear import load_quantized_qwen3, dequantize, QuantLinear


# thresholds (relative to per-token max abs activation) to call a channel "dead"
REL_THRESHOLDS = [0.001, 0.01, 0.05]


def find_mlp_modules(model):
    """Return list of (layer_idx, mlp_module) for every decoder layer's MLP."""
    mlps = []
    for name, mod in model.named_modules():
        # Qwen3 MLP has gate_proj, up_proj, down_proj children
        if all(hasattr(mod, p) for p in ("gate_proj", "up_proj", "down_proj")):
            mlps.append((name, mod))
    return mlps


class SparsityHook:
    """
    Wraps an MLP module's forward to compute the intermediate activation
    h = SiLU(gate(x)) * up(x)  and record per-channel sparsity, WITHOUT
    changing the module's real output.
    """
    def __init__(self):
        # counts[thr] = [total_channels, dead_channels] accumulated over tokens
        self.counts = {t: [0, 0] for t in REL_THRESHOLDS}
        self.n_token_positions = 0

    def __call__(self, module, inp, out):
        x = inp[0]
        # only measure single-token decode steps (seqlen == 1); skip prefill
        if x.dim() == 3 and x.shape[1] != 1:
            return
        with torch.no_grad():
            # recompute the intermediate the same way the MLP does
            def lin(proj, xx):
                if isinstance(proj, QuantLinear):
                    w = dequantize(proj.qweight, proj.scales).to(xx.dtype)
                    return F.linear(xx, w)
                return proj(xx)
            gate = lin(module.gate_proj, x)
            up = lin(module.up_proj, x)
            h = F.silu(gate) * up                    # [.., inter]
            h = h.reshape(-1, h.shape[-1])           # [tokens, inter]
            maxabs = h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            rel = h.abs() / maxabs                   # 0..1 per channel
            for thr in REL_THRESHOLDS:
                dead = (rel < thr).sum().item()
                total = rel.numel()
                self.counts[thr][0] += total
                self.counts[thr][1] += dead
            self.n_token_positions += h.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--prompt", type=str,
                   default="Explain the theory of relativity in detail.")
    args = p.parse_args()

    print("Loading model ...")
    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    mlps = find_mlp_modules(model)
    print(f"Found {len(mlps)} MLP modules.")

    hook = SparsityHook()
    handles = [m.register_forward_hook(hook) for _, m in mlps]

    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to("cuda:0")

    print(f"Generating {args.tokens} tokens (measuring MLP sparsity) ...")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=args.tokens, do_sample=False,
                       use_cache=True, pad_token_id=tok.eos_token_id)

    for h in handles:
        h.remove()

    print("\n" + "=" * 60)
    print("MLP INTERMEDIATE SPARSITY (per decode token, across all layers)")
    print("=" * 60)
    print(f"Token-positions measured: {hook.n_token_positions}")
    print(f"Intermediate width      : {mlps[0][1].up_proj.qweight.shape[0] if isinstance(mlps[0][1].up_proj, QuantLinear) else '?'}")
    print("-" * 60)
    print(f"{'threshold':>12} | {'% channels dead':>16} | {'DIP verdict':>15}")
    print("-" * 60)
    for thr in REL_THRESHOLDS:
        total, dead = hook.counts[thr]
        pct = 100.0 * dead / total if total else 0.0
        verdict = ("BIG WIN" if pct >= 70 else
                   "MODERATE" if pct >= 40 else
                   "WEAK")
        print(f"{thr:>12} | {pct:>15.1f}% | {verdict:>15}")
    print("=" * 60)
    print("\nInterpretation: 'dead' = channel activation < threshold * max.")
    print("The higher the % dead, the more weight-streaming DIP can skip.")


if __name__ == "__main__":
    main()