# dynamic-input-pruning

Custom CUDA kernels for Dynamic Input Pruning (DIP) and cache-aware DIP,
running an offloaded int4 Qwen3-32B on a single 15 GB NVIDIA Tesla T4.

Qwen3-32B in int4 is ~18 GB, larger than the T4's 15 GB of usable memory.
Weights are held in CPU RAM and streamed to the GPU over PCIe per forward pass.
Streaming the full MLP weights dominates decode time (~60% of per-token latency
at baseline). DIP exploits the fact that, per token, most SwiGLU MLP
intermediate channels contribute little after the gate: for hidden state `x` it
computes the gate activation `g = SiLU(gate_proj(x))` (length = intermediate
size `I`), keeps only the `k = keep_ratio · I` channels with the largest `|g|`,
and streams only those channels' up/down weight rows. Cache-aware DIP
additionally keeps the most-frequently-selected channels resident on the GPU so
they are never streamed. The entire decode-time MLP path (gate, selection, up,
down, and the cache-aware assembly) runs on custom int4 CUDA kernels.

## Hardware

NVIDIA Tesla T4 (Turing, TU104, compute capability sm_75):

- 16 GB GDDR6 (~15 GB usable), ~320 GB/s device memory bandwidth
- 2560 CUDA cores, 320 Turing Tensor cores, 70 W, single slot, no NVLink
- Host interface: PCIe Gen3 (the T4 supports x8 and x16). Nominal Gen3 x16 is
  ~16 GB/s theoretical / ~12-13 GB/s achievable each way.

The offload regime makes the host-to-device link the bottleneck. On the cloud
instance used here, measured effective host-to-device bandwidth for a bulk
weight transfer was ~5.6 GB/s (consistent with a Gen3 x8 link), so PCIe
transfer, not compute, is what limits decode.

This is what DIP attacks: it reduces bytes crossing PCIe per token rather than
arithmetic. Each MLP streams gate + up + down; DIP keeps the gate resident and
streams only k of the I up/down rows, so per-layer streamed bytes for those two
projections scale as `keep_ratio` (0.6 here, a 40% reduction), and cache-aware
DIP removes a further fraction equal to the cache hit rate.

## Results

Qwen3-32B, int4 g128, single T4, keep_ratio=0.6 (prune 40%), cache_frac=0.3.
Steady-state decode throughput, warmup excluded, mean of 3 runs of 64 tokens:

| Configuration                | tok/s | Speedup |
|------------------------------|-------|---------|
| Baseline (offloaded int4)    | 0.089 | 1.0x    |
| Selective-streaming DIP      | 0.417 | 4.7x    |
| Cache-aware DIP              | 0.477 | 5.4x    |

Selective streaming (transferring only the selected channels' rows instead of
the full weight) provides the bulk of the gain; the resident cache removes a
further ~38% of the remaining transfers (the measured hit rate at cache_frac=0.3),
for an additional ~15%.

## Architecture

### Quantization and offload

`quantize.py` converts the seven MLP/attention projection weights per layer to
int4, group size 128, symmetric. For each group of 128 weights the scale is
`s = max(|w|) / 7` (signed int4 spans [-8, 7]), the stored integer is
`q = round(w / s)`, and dequantization is `w ≈ q · s`. Two int4 values are
packed per byte (low nibble = even column, high nibble = odd column); scales are
bf16, one per group. Embeddings, norms, QK-norms and lm_head stay bf16.

`qlinear.py` builds the model on the meta device, replaces every `nn.Linear`
with a `QuantLinear`, infers an accelerate device map (with
`no_split_module_classes=["Qwen3DecoderLayer"]`), and dispatches. On the T4 this
places 31 decoder layers on the GPU and offloads 37 to CPU; accelerate streams
each offloaded layer's weights to the GPU immediately before its forward and
evicts them after.

### Selective streaming (DIP)

`dip_stream_predispatch.py` replaces each MLP with a `StreamingDIPMlp` *before*
accelerate dispatch, so accelerate never attaches an offload hook to it (a hook
on a module whose forward is later replaced corrupts the offloaded path). The
gate weight is kept resident on the GPU; up and down weights are held pinned in
CPU RAM and streamed by the module itself, not by accelerate.

The standard SwiGLU MLP is `y = down( SiLU(gate(x)) ⊙ up(x) )`. Writing
`g = SiLU(gate(x))` and `u = up(x)`, the intermediate is `h = g ⊙ u` and
`y = down(h)`. DIP keeps only the k channels where `|g|` is largest, so `u` and
`h` need only be computed on those k rows, and only those rows of `up` and
`down` are needed.

Per decode token (sequence length 1):

1. `g = SiLU(gate_proj(x))` via the int4 GEMV kernel.
2. `idx = top-k(|g|, k)`, `k = keep_ratio · I`, via the top-k selection kernel.
3. The selected rows of up_proj and down_proj are gathered on the CPU into
   pinned buffers (`index_select`), then copied to preallocated GPU buffers with
   an async transfer. Only k of the I intermediate channels cross PCIe. Pinned
   gather + a single contiguous copy measured ~3.2x faster than transferring the
   full weight, and far faster than per-row copies.
4. `u_sel = up_sel(x)` via `int4_gather_gemv`, then
   `y = int4_gather_down( g[idx] ⊙ u_sel )` — the second kernel folds the
   `g ⊙ u` product and the down projection together.

down_proj is stored transposed and repacked to int4 at load, so the selected
intermediate channels are contiguous rows (a fast gather) rather than strided
columns; this also halves down_proj's streamed bytes versus a bf16 transpose.

Staging buffers on the GPU are allocated once and shared across all 64 layers
(the layers run sequentially, so per-layer buffers are unnecessary and would
exhaust VRAM).

### Cache-aware DIP

`dip_ca.py` adds a resident channel cache. A short calibration pass records
per-layer channel selection frequency; the top `cache_frac · I` channels per
layer have their up/down int4 weights loaded resident on the GPU. The hot set is
chosen with a deterministic tie-break (sort key `freq · I − channel_id`, i.e.
frequency first, then lowest channel id) so the cache — and therefore
generation — is reproducible.

Per token the k selected channels split into hits (resident, `id2slot[c] ≥ 0`)
and misses (streamed). Only the misses cross PCIe. The up projection is then a
single fused kernel: for each selected channel a source flag picks the resident
cache or the streamed buffer, and the kernel reads that row, dequantizes int4
inline, and computes the dot product with x in one pass. At cache_frac=0.3 the
measured hit rate is ~38%, so roughly that fraction of up/down streaming is
avoided.

### CUDA kernels (sm_75)

- **int4 dequant-GEMV** — computes `out[r] = Σ_c (q[r,c]·s[r,c]) · x[c]`; one
  block per output row, nibble unpack + per-group scale, warp-reduced. Used for
  the gate projection.
- **gather-GEMV (up_proj)** — same GEMV but only for the selected rows given by
  an index list; one block per selected row.
- **gather-down (down_proj)** — reads selected rows of the int4 transpose; each
  block computes one output element `y[o] = Σ_j h[j] · dequant(down_t[j, o])` as
  a weighted sum over the k selected rows.
- **top-k selection** — DIP needs the *set* of the k largest `|g|`, not a sorted
  order, so instead of a full sort the kernel finds a threshold `t` such that
  `#{|g| > t} ≈ k` by bisection (parallel max-abs reduction to bound the search,
  parallel count per step), then emits the qualifying indices by deterministic
  parallel stream compaction: a shared-memory prefix-sum scan places each
  qualifying element at slot = (number of qualifiers before it), strict-greater
  rows first, then the tie band lowest-index-first. Returns exactly k valid
  indices and is a pure function (identical input → identical output).
- **cache-aware fused GEMV** — per selected channel a flag selects the resident
  cache or the streamed buffer as the source; the kernel reads from the chosen
  source, dequantizes, and computes the GEMV in a single pass.

## Setup

    uv pip install -r requirements.txt

## Prepare the model

    python scripts/download.py --repo_id Qwen/Qwen3-32B
    python scripts/quantize.py --model_dir checkpoints/Qwen/Qwen3-32B --out_dir checkpoints/qwen3-32b-int4

## Build the kernels (Tesla T4 / sm_75)

    cd kernels/int4_gemv
    python setup.py build_ext --inplace
    cd ../int4_gather
    python setup_gather.py build_ext --inplace
    cd ../int4_down
    python setup_down.py build_ext --inplace
    cd ../topk_select
    python setup_topk.py build_ext --inplace
    cd ../ca_fused
    python setup_ca_fused.py build_ext --inplace
    cd ../..

Each kernel directory has a standalone correctness test (e.g. `test_gather.py`)
that checks the kernel against a PyTorch reference.

## Run

Baseline (offloaded int4, no pruning):

    python baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --max_new_tokens 64

Selective-streaming DIP:

    python -m tests.test_stream_predispatch --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 30 --keep 0.6

Cache-aware DIP:

    python -m tests.test_dip_ca --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 50 --keep 0.6 --cache_frac 0.3

Benchmark a configuration (warmup-excluded steady-state throughput):

    python benchmarks/benchmark.py --config ca --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --keep 0.6 --cache_frac 0.3

## Notes

keep_ratio=0.6 (prune 40%) is the operating point where the streaming pipeline
stays coherent; more aggressive pruning is possible but degrades output because
the gate is an imperfect predictor of final channel importance (selecting on
`|g|` alone approximates, but does not equal, selecting on the true `|g ⊙ u|`).
cache_frac=0.3 is the largest resident cache that fits alongside the model in
15 GB.