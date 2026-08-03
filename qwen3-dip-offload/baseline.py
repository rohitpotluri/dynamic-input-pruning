"""
baseline.py — Naive offloaded inference baseline for our self-quantized
INT4 Qwen3-32B on a single T4.

This is the number DIP/DIP-CA + custom kernels must beat. It uses:
  * our int4 g128 weights (same weights the kernels will use)
  * naive PyTorch dequant in QuantLinear.forward (no fused kernel)
  * accelerate's stock offload (no smart prefetch, no DIP selection)

So any speedup later is attributable to OUR work, not to quantization.

Usage:
    python baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10
"""

import argparse
import time

import torch

from qlinear import load_quantized_qwen3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/qwen3-32b-int4")
    p.add_argument("--prompt", type=str, default="Explain gravity in simple terms.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_runs", type=int, default=3)
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    print("Loading int4 Qwen3-32B with offload ...")
    t0 = time.time()
    model, tok = load_quantized_qwen3(
        args.ckpt, gpu_mem_gib=args.gpu_mem, cpu_mem_gib=args.cpu_mem
    )
    model.eval()
    print(f"Load time: {time.time() - t0:.1f}s")

    messages = [{"role": "user", "content": args.prompt}]
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    # some transformers versions return a dict-like; normalize to a tensor
    if not torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids["input_ids"]
    inputs = prompt_ids.to("cuda:0")
    prompt_len = inputs.shape[-1]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,                 # greedy -> deterministic/comparable
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )

    print("\nWarmup (not timed) ...")
    with torch.no_grad():
        _ = model.generate(inputs, **gen_kwargs)
    torch.cuda.synchronize()

    tps_list = []
    for r in range(args.num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(inputs, **gen_kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new_tok = out.shape[-1] - prompt_len
        tps = new_tok / dt
        tps_list.append(tps)
        print(f"[run {r+1}] {new_tok} tok in {dt:.2f}s -> {tps:.3f} tok/s")

    avg = sum(tps_list) / len(tps_list)
    print("\n" + "=" * 50)
    print(f"Avg throughput: {avg:.3f} tok/s   <-- BASELINE TO BEAT")
    print("=" * 50)
    print("\nSample output:")
    print(tok.decode(out[0][prompt_len:], skip_special_tokens=True))


if __name__ == "__main__":
    main()