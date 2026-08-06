"""
dip_offload.py — Run DIP + gather kernel on the OFFLOADED 32B model.

The problem: accelerate's offload hooks materialize a layer's weights to GPU
right before its forward, then evict after. Our earlier monkey-patch fought
this and hit 'Tensor on device meta'. 

The fix here: we do NOT replace forward in a way that bypasses accelerate.
Instead we let accelerate place the MLP's weights on GPU as usual (its hook
runs first), and our DIP forward reads them AFTER they're materialized -- but
only does the pruned compute. So we don't save streaming yet (accelerate still
moves the whole layer), but we DO prove the kernel runs correctly in the
offloaded 32B pipeline and measure the compute effect.

This is the safe first integration step. The bandwidth win (streaming only
selected rows) is the NEXT step, once this runs clean.

Usage:
    python dip_offload.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 \
        --tokens 8 --keep 0.32
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlinear import load_quantized_qwen3, QuantLinear, dequantize
from dip_mlp import _use_gather


def _w(proj):
    return dequantize(proj.qweight, proj.scales) if isinstance(proj, QuantLinear) else proj.weight


def make_dip_forward(mlp, keep_ratio):
    """
    DIP forward that assumes weights are ALREADY on GPU (accelerate's hook put
    them there). Reads them, prunes, uses the gather kernel for up_proj.
    """
    def forward(x):
        if (not _use_gather or x.dim() != 3 or x.shape[1] != 1
                or keep_ratio >= 0.999 or not isinstance(mlp.up_proj, QuantLinear)):
            # dense fallback (prefill, or kernel unavailable)
            g = F.silu(F.linear(x, _w(mlp.gate_proj).to(x.dtype)))
            u = F.linear(x, _w(mlp.up_proj).to(x.dtype))
            return F.linear(g * u, _w(mlp.down_proj).to(x.dtype))

        xdt = x.dtype
        xf = x.reshape(-1)
        gate_w = _w(mlp.gate_proj).to(xdt)
        g = F.silu(F.linear(xf.unsqueeze(0), gate_w)).squeeze(0)
        del gate_w
        inter = g.shape[-1]
        k = max(1, int(round(inter * keep_ratio)))
        idx = g.abs().topk(k).indices.to(torch.int32)

        up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(
            mlp.up_proj.qweight, mlp.up_proj.scales,
            xf.to(torch.bfloat16), idx)
        h_sel = g[idx.long()] * up_sel.to(xdt)

        dn_w = _w(mlp.down_proj).to(xdt)
        y = F.linear(h_sel.unsqueeze(0), dn_w[:, idx.long()]).squeeze(0)
        del dn_w
        return y.reshape(x.shape[0], 1, -1)

    return forward


def patch(model, keep):
    orig = {}
    for name, mod in model.named_modules():
        if all(hasattr(mod, p) for p in ("gate_proj", "up_proj", "down_proj")):
            orig[name] = mod.forward
            mod.forward = make_dip_forward(mod, keep)
    return orig


def gen_timed(model, tok, prompt, tokens):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to("cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    ntok = out.shape[-1] - ids.shape[-1]
    return txt, ntok, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--keep", type=float, default=0.32)
    p.add_argument("--prompt", type=str, default="Explain gravity in simple terms.")
    args = p.parse_args()

    print(f"gather kernel loaded: {_use_gather}")
    print("Loading offloaded 32B ...")
    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    print("\n--- Warmup (dense) ---")
    _ = gen_timed(model, tok, args.prompt, 2)

    print("\n--- DIP + kernel on offloaded 32B ---")
    orig = patch(model, args.keep)
    try:
        txt, ntok, dt = gen_timed(model, tok, args.prompt, args.tokens)
    finally:
        pass  # leave patched; we're done

    print(f"\nOutput:\n{txt}")
    print("\n" + "=" * 50)
    print(f"DIP+kernel: {ntok} tok in {dt:.2f}s -> {ntok/dt:.3f} tok/s")
    print(f"(baseline was ~0.086 tok/s)")
    print("=" * 50)


if __name__ == "__main__":
    main()