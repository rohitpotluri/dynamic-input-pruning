"""
profile_baseline.py — Find where the per-token time actually goes.

Breaks the cost into three buckets so we can attribute future speedups:
  1. DEQUANT  — time spent in QuantLinear unpacking/scaling int4 -> bf16
  2. STREAM   — time accelerate spends moving CPU-offloaded layers to GPU
  3. COMPUTE  — everything else (matmuls, attention, norms)

It works by monkey-patching QuantLinear.forward and accelerate's device hook
to accumulate timers, then running a short generation.

Usage:
    python profile_baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 4
"""

import argparse
import time
import torch

import qlinear
from qlinear import load_quantized_qwen3, QuantLinear, dequantize


# ---- global accumulators ----
T = {"dequant": 0.0, "linear_total": 0.0, "calls": 0}


def patch_quantlinear():
    """Wrap QuantLinear.forward to time the dequant vs the matmul."""
    orig = QuantLinear.forward

    def timed_forward(self, x):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w = dequantize(self.qweight, self.scales)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = torch.nn.functional.linear(x, w.to(x.dtype))
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        T["dequant"] += (t1 - t0)
        T["linear_total"] += (t2 - t0)
        T["calls"] += 1
        return out

    QuantLinear.forward = timed_forward
    return orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--prompt", type=str, default="Explain gravity in simple terms.")
    args = p.parse_args()

    print("Loading ...")
    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    patch_quantlinear()

    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to("cuda:0")

    gen = dict(max_new_tokens=args.tokens, do_sample=False,
               use_cache=True, pad_token_id=tok.eos_token_id)

    # warmup (compiles caches etc.) — reset timers after
    print("Warmup ...")
    with torch.no_grad():
        _ = model.generate(ids, **gen)
    T["dequant"] = T["linear_total"] = 0.0
    T["calls"] = 0

    print(f"Timed run ({args.tokens} tokens) ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, **gen)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    new_tok = out.shape[-1] - ids.shape[-1]
    dq = T["dequant"]
    lin = T["linear_total"]
    matmul = lin - dq
    other = wall - lin  # attention, norms, streaming overhead, sampling

    print("\n" + "=" * 55)
    print(f"Tokens generated     : {new_tok}")
    print(f"Wall time            : {wall:.2f}s   ({new_tok/wall:.3f} tok/s)")
    print(f"QuantLinear calls    : {T['calls']}")
    print("-" * 55)
    print(f"DEQUANT (int4->bf16) : {dq:6.2f}s  ({100*dq/wall:4.1f}%)")
    print(f"MATMUL  (F.linear)   : {matmul:6.2f}s  ({100*matmul/wall:4.1f}%)")
    print(f"OTHER   (attn/stream): {other:6.2f}s  ({100*other/wall:4.1f}%)")
    print("=" * 55)
    print("\nNote: STREAM (PCIe) time is folded into OTHER and into the")
    print("matmul/dequant of CPU-resident layers (hooks move weights just")
    print("before the layer runs). The big lever is whichever bucket dominates.")


if __name__ == "__main__":
    main()