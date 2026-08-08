// int4_gather_down.cu — Fused GATHER + INT4 dequant + weighted-row-sum for
// down_proj in the DIP MLP.
//
// down_proj computes: y[o] = sum over selected channels c of h[c] * down[o, c]
// We store down TRANSPOSED and int4-packed: down_t [inter, out], so channel c
// is a contiguous ROW down_t[c, :] of length `out`. DIP selects k channels.
//
// This kernel computes:
//   y[o] = sum_{j=0..k-1} h_sel[j] * dequant(down_t[ idx_local[j] , o ])
// but since we STREAM only the selected rows into a contiguous buffer first,
// idx_local is just 0..k-1 (rows already gathered on host side).
//
//   down_t_sel : int8  [k, out/2]   (k selected rows, two int4 per byte)
//   scales     : bf16  [k, out/128]
//   h_sel      : bf16  [k]          (gate*up for selected channels)
//   y          : bf16  [out]
//
// One block per OUTPUT element o. Each block sums k contributions.
// Actually: one block per output col o; threads loop over k. out=5120.
// Built for sm_75 (T4).

#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define BLOCK_THREADS 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_THREADS / WARP_SIZE)
#define GROUP_SIZE 128

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ int sext4(int v) { return (v >= 8) ? (v - 16) : v; }

// One block per output element o. Threads split the k selected channels.
// For output o: read down_t_sel[j, o] (int4) for each selected row j, dequant,
// multiply by h_sel[j], accumulate.
__global__ void int4_gather_down_kernel(
    const int8_t*        __restrict__ down_t_sel,  // [k, out/2]
    const __nv_bfloat16* __restrict__ scales,      // [k, out/GROUP_SIZE]
    const __nv_bfloat16* __restrict__ h_sel,       // [k]
    __nv_bfloat16*       __restrict__ y,           // [out]
    int k, int out_features
) {
    int o = blockIdx.x;                 // which output element (0..out-1)
    int packed_col = o / 2;             // which byte in the row
    int nibble = o & 1;                 // low (even) or high (odd)
    int n_groups = out_features / GROUP_SIZE;
    int group = o / GROUP_SIZE;

    float acc = 0.0f;
    // each thread handles a subset of the k selected channels
    for (int j = threadIdx.x; j < k; j += BLOCK_THREADS) {
        int8_t byte = down_t_sel[j * (out_features / 2) + packed_col];
        int q = nibble ? sext4((byte >> 4) & 0x0F) : sext4(byte & 0x0F);
        float s = __bfloat162float(scales[j * n_groups + group]);
        float h = __bfloat162float(h_sel[j]);
        acc += (q * s) * h;
    }

    __shared__ float warp_sums[NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    float wsum = warpReduceSum(acc);
    if (lane_id == 0) warp_sums[warp_id] = wsum;
    __syncthreads();
    if (threadIdx.x < NUM_WARPS) {
        float fsum = warp_sums[threadIdx.x];
        fsum = warpReduceSum(fsum);
        if (threadIdx.x == 0) y[o] = __float2bfloat16_rn(fsum);
    }
}

torch::Tensor int4_gather_down_impl(
    const torch::Tensor& down_t_sel,   // int8 [k, out/2]
    const torch::Tensor& scales,       // bf16 [k, out/128]
    const torch::Tensor& h_sel         // bf16 [k]
) {
    int k = down_t_sel.size(0);
    int out_features = down_t_sel.size(1) * 2;
    auto y = torch::empty({out_features},
        torch::TensorOptions().dtype(torch::kBFloat16).device(h_sel.device()));

    int4_gather_down_kernel<<<out_features, BLOCK_THREADS>>>(
        down_t_sel.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h_sel.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        k, out_features
    );
    return y;
}

TORCH_LIBRARY(int4_down_ops, m) {
    m.def("int4_gather_down(Tensor down_t_sel, Tensor scales, Tensor h_sel) -> Tensor");
}
TORCH_LIBRARY_IMPL(int4_down_ops, CUDA, m) {
    m.impl("int4_gather_down", &int4_gather_down_impl);
}
