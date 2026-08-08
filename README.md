# dynamic-input-pruning
Custom CUDA kernels for Dynamic Input Pruning (DIP) and cache-aware DIP,
for running an offloaded int4 Qwen3-32B on a single 15GB GPU (Tesla T4).

## Setup

Get SSH key (local machine):
    type C:\Users\realr\ .ssh\id_vast.pub

Install deps:
    uv pip install -r requirements.txt

## Prepare the model

Download bf16 Qwen3-32B (~62GB):
    python download.py --repo_id Qwen/Qwen3-32B

Quantize to int4 g128 symmetric (our custom format):
    python quantize.py --model_dir checkpoints/Qwen/Qwen3-32B --out_dir checkpoints/qwen3-32b-int4

## Baseline

Profile where per-token time goes (streaming vs dequant vs compute):
    python profile_baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 4

Measure baseline throughput:
    python baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --max_new_tokens 8 --num_runs 1

## Build the CUDA kernels (Tesla T4 / sm_75)

    cd kernels/int4_gather_gemv
    python setup.py build_ext --inplace            # plain int4 dequant-GEMV
    python test_int4_gemv.py                        # correctness check
    python setup_gather.py build_ext --inplace      # gather + int4 dequant-GEMV (up_proj)
    python test_gather.py                           # correctness check
    python setup_down.py build_ext --inplace        # gather + int4 dequant (down_proj)
    python test_down.py                             # correctness check
    cd ../..

## Verify DIP prunability + quality

Measure MLP sparsity (how prunable each token is):
    python measure_sparsity.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 8

Verify pruned output quality on a fits-in-VRAM model (Qwen3-8B):
    python download.py --repo_id Qwen/Qwen3-8B
    python quantize.py --model_dir checkpoints/Qwen/Qwen3-8B --out_dir checkpoints/qwen3-8b-int4
    python dip_quality_test.py --ckpt checkpoints/qwen3-8b-int4 --tokens 60 --keeps 1.0 0.5 0.32 0.2

## Run selective-streaming DIP on the offloaded 32B

Both MLP projections (up + down) run on custom int4 gather kernels; only the
DIP-selected rows are streamed CPU->GPU per token. keep=0.6 is the coherent
operating point (prune 40%):

    python test_stream_predispatch.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 30 --keep 0.6

Verify DIP is active (path usage, % rows streamed) + longer-form coherence:

    python verify_dip.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 120 --keep 0.6

## Status

- [x] int4 g128 self-quantization (custom format)
- [x] GPU/CPU offload loader (31/37 split on T4)
- [x] Baseline: 0.088 tok/s
- [x] Profiled: ~60% of time is CPU->GPU weight streaming
- [x] MLP is ~68% prunable per token; keep=0.6 preserves coherence in streaming
- [x] Three verified CUDA kernels: int4 GEMV, gather-GEMV (up), gather-down (down)
- [x] DIP + kernels correct on offloaded 32B
- [x] Selective streaming (both projections, int4, pinned-gather): ~2.5x over baseline
- [ ] DIP-CA cache policy (keep hot channels resident)
- [ ] Prefetch / double-buffering overlap
- [ ] Benchmark with warmup + per-optimization attribution

## Notes

- Selective streaming uses pinned CPU weights + reused GPU staging buffers +
  gather-into-pinned-buffer (validated ~3.2x faster than bulk-streaming on the
  T4's PCIe link via bench_selective_stream.py).
- down_proj is stored transposed + repacked to int4 at load time so selected
  channels are contiguous rows (fast gather), halving down's streamed bytes.
- Streaming MLPs are swapped in BEFORE accelerate dispatch so accelerate never
  hooks them (avoids the offload-hook collision).