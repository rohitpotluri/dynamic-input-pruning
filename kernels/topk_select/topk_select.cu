// topk_select.cu — Threshold-based top-k selection for DIP.
// Deterministic and parallel: identical input -> identical output, no races.
//
// Pipeline:
//   1. max_abs   : max|s|                          (parallel reduction)
//   2. bisection : threshold t (host loop over count_gt)
//   3. compact   : deterministic parallel stream compaction via prefix-sum
//
// Compaction: for each element i that qualifies, its output slot is the number
// of qualifying elements with index < i (an exclusive prefix-sum over the
// qualify mask). Slots are data-determined, with no atomics, so the result is
// reproducible. Two tiers fill exactly k:
//   tier A: strict winners  |s| >  t_strict            (<= k of these)
//   tier B: tie band        t_lo < |s| <= t_strict     (fills the remainder,
//           lowest-index-first, since prefix-sum preserves index order)
//
// The scan runs in a single block of BLOCK threads walking the array in
// grid-stride tiles, carrying a running offset in shared memory. A single
// block avoids cross-block carry logic; for one MLP width (n ~ 25600) this is
// fast enough. Built for sm_75.

#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define BLOCK 1024

__global__ void max_abs_kernel(const __nv_bfloat16* __restrict__ s, int n,
                               float* __restrict__ out_max) {
    __shared__ float sh[BLOCK];
    float m = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(__bfloat162float(s[i])));
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + st]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax((unsigned int*)out_max, __float_as_uint(sh[0]));
}

__global__ void count_gt_kernel(const __nv_bfloat16* __restrict__ s, int n,
                                float thresh, int* __restrict__ count) {
    __shared__ int sh;
    if (threadIdx.x == 0) sh = 0;
    __syncthreads();
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (fabsf(__bfloat162float(s[i])) > thresh) local++;
    atomicAdd(&sh, local);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(count, sh);
}

// Single-block deterministic compaction. One block walks the array in tiles of
// BLOCK: for each tile it computes the qualify mask, exclusive-scans it, and
// scatters qualifying indices to (running_offset + local_exclusive_rank), with
// running_offset carried in shared memory across tiles. Two passes: strict
// winners, then the tie band, until exactly k are emitted.
__global__ void parallel_compact_kernel(const __nv_bfloat16* __restrict__ s, int n,
                                         float t_lo, float t_strict,
                                         int* __restrict__ idx, int k) {
    __shared__ int scan[BLOCK];
    __shared__ int running;      // number emitted so far
    if (threadIdx.x == 0) running = 0;
    __syncthreads();

    int tid = threadIdx.x;

    // pass A: strict winners |s| > t_strict
    for (int base = 0; base < n; base += BLOCK) {
        int i = base + tid;
        int q = 0;
        if (i < n) {
            float v = fabsf(__bfloat162float(s[i]));
            q = (v > t_strict) ? 1 : 0;
        }
        scan[tid] = q;
        __syncthreads();
        for (int off = 1; off < BLOCK; off <<= 1) {
            int add = (tid >= off) ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        int inclusive = scan[tid];
        int excl = inclusive - q;
        int tile_total = scan[BLOCK - 1];
        int base_off = running;
        if (q) {
            int pos = base_off + excl;
            if (pos < k) idx[pos] = i;
        }
        __syncthreads();
        if (tid == 0) running += tile_total;
        __syncthreads();
    }

    // pass B: tie band (t_lo, t_strict], fill until k
    for (int base = 0; base < n; base += BLOCK) {
        if (running >= k) break;
        int i = base + tid;
        int q = 0;
        if (i < n) {
            float v = fabsf(__bfloat162float(s[i]));
            q = (v > t_lo && v <= t_strict) ? 1 : 0;
        }
        scan[tid] = q;
        __syncthreads();
        for (int off = 1; off < BLOCK; off <<= 1) {
            int add = (tid >= off) ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        int inclusive = scan[tid];
        int excl = inclusive - q;
        int tile_total = scan[BLOCK - 1];
        int base_off = running;
        if (q) {
            int pos = base_off + excl;
            if (pos < k) idx[pos] = i;
        }
        __syncthreads();
        if (tid == 0) running += tile_total;
        __syncthreads();
    }
}

torch::Tensor topk_select_impl(const torch::Tensor& s_in, int64_t k) {
    auto s = s_in.contiguous();
    int n = s.size(0);
    auto dev = s.device();
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(s.data_ptr());

    int grid = (n + BLOCK - 1) / BLOCK; if (grid > 256) grid = 256;

    auto maxbuf = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    max_abs_kernel<<<grid, BLOCK>>>(sp, n, maxbuf.data_ptr<float>());
    float hi = maxbuf.item<float>();
    float lo = 0.0f;

    auto cbuf = torch::zeros({1}, iopt);
    float thresh = 0.5f * (lo + hi);
    for (int step = 0; step < 25; ++step) {
        cbuf.zero_();
        count_gt_kernel<<<grid, BLOCK>>>(sp, n, thresh, cbuf.data_ptr<int>());
        int cnt = cbuf.item<int>();
        if (cnt > k) lo = thresh; else hi = thresh;
        thresh = 0.5f * (lo + hi);
    }
    float t_strict = hi;
    float t_lo = lo;

    auto idx = torch::zeros({k}, iopt);
    parallel_compact_kernel<<<1, BLOCK>>>(sp, n, t_lo, t_strict,
                                          idx.data_ptr<int>(), (int)k);
    return idx;
}

TORCH_LIBRARY(topk_ops, m) {
    m.def("topk_select(Tensor s, int k) -> Tensor");
}
TORCH_LIBRARY_IMPL(topk_ops, CUDA, m) {
    m.impl("topk_select", &topk_select_impl);
}