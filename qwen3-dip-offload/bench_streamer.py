"""
bench_streamer.py — Isolated benchmark of the streaming manager vs naive .to().

Simulates the offloaded portion of Qwen3-32B: ~37 decoder layers worth of int4
weights sitting in CPU RAM, streamed to a T4 every token. Compares:

  A. NAIVE   — pageable CPU tensor -> .to("cuda") per layer (what happens now)
  B. STREAMER — pinned + reused buffers + double-buffered async prefetch

If (B) is meaningfully faster here, integrating it into the model is worth it.

Usage:
    python bench_streamer.py --layers 37 --iters 10
"""

import argparse
import time
import torch

from offload.prefetch import LayerWeightStreamer


def make_fake_layers(n_layers, device_cpu="cpu"):
    """
    Approximate one Qwen3-32B decoder layer's *quantized* weight footprint.
    dim=5120, intermediate ~= 25600. int4 packed = /2 bytes on last dim.
    We use int8 buffers sized to the packed int4 layout + bf16 scales, roughly.
    Total per layer ~ a few hundred MB (close enough for a PCIe benchmark).
    """
    dim = 5120
    inter = 25600
    layers = []
    for _ in range(n_layers):
        d = {}
        # attention projections (packed int4: out x in/2 int8)
        d["q_proj.qweight"] = torch.randint(-128, 127, (dim, dim // 2), dtype=torch.int8)
        d["k_proj.qweight"] = torch.randint(-128, 127, (1024, dim // 2), dtype=torch.int8)
        d["v_proj.qweight"] = torch.randint(-128, 127, (1024, dim // 2), dtype=torch.int8)
        d["o_proj.qweight"] = torch.randint(-128, 127, (dim, dim // 2), dtype=torch.int8)
        # mlp projections
        d["gate_proj.qweight"] = torch.randint(-128, 127, (inter, dim // 2), dtype=torch.int8)
        d["up_proj.qweight"] = torch.randint(-128, 127, (inter, dim // 2), dtype=torch.int8)
        d["down_proj.qweight"] = torch.randint(-128, 127, (dim, inter // 2), dtype=torch.int8)
        layers.append(d)
    return layers


def bench_naive(layers, device, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for d in layers:
            gpu = {k: v.to(device, non_blocking=False) for k, v in d.items()}
            # touch to ensure copy really lands
            _ = next(iter(gpu.values()))[0, 0].item()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_streamer(layers, device, iters, n_buffers):
    streamer = LayerWeightStreamer(layers, device=device, n_buffers=n_buffers)
    streamer.pin_all()
    n = len(layers)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        streamer.prefetch(0)
        for i in range(n):
            streamer.prefetch(i + 1)     # kick next while we "use" i
            gpu = streamer.get(i)        # blocks until i ready
            _ = next(iter(gpu.values()))[0, 0].item()  # touch
            streamer.release(i)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=37)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--buffers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    print(f"Building {args.layers} fake offloaded layers ...")
    layers = make_fake_layers(args.layers)
    bytes_per_iter = sum(v.numel() * v.element_size()
                         for d in layers for v in d.values())
    print(f"Per-iter transfer volume: {bytes_per_iter/1e9:.2f} GB")

    # separate copies so pinning in one doesn't affect the other
    import copy
    layers_naive = [{k: v.clone() for k, v in d.items()} for d in layers]
    layers_stream = [{k: v.clone() for k, v in d.items()} for d in layers]

    print("\nBenchmarking NAIVE (pageable .to) ...")
    t_naive = bench_naive(layers_naive, args.device, args.iters)
    gbps_naive = bytes_per_iter * args.iters / 1e9 / t_naive

    print("Benchmarking STREAMER (pinned + reuse + double-buffer) ...")
    t_stream = bench_streamer(layers_stream, args.device, args.iters, args.buffers)
    gbps_stream = bytes_per_iter * args.iters / 1e9 / t_stream

    print("\n" + "=" * 55)
    print(f"NAIVE    : {t_naive:.2f}s total | {gbps_naive:.1f} GB/s effective")
    print(f"STREAMER : {t_stream:.2f}s total | {gbps_stream:.1f} GB/s effective")
    print(f"Speedup  : {t_naive / t_stream:.2f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()