// int4_gather_gemv.cu — Fused gather + int4 dequant + GEMV for up_proj.
//
// Same int4 g128 symmetric layout as int4_gemv.cu, but computes only the
// selected rows given by `indices` rather than all output rows.
//
//   qweight : int8  [out_features, in_features/2]   (two int4 per byte)
//   scales  : bf16  [out_features, in_features/128]
//   x       : bf16  [in_features]
//   indices : int32 [k]     rows to compute (DIP-selected)
//   out     : bf16  [k]     out[j] = row indices[j] of (dequant(W) @ x)
//
// One block per selected row; grid = k blocks (not out_features). In the DIP
// pipeline the k selected rows are streamed into a contiguous buffer first and
// `indices` is 0..k-1, so the kernel reads only the streamed rows. Built for
// sm_75.

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

__device__ __forceinline__ int sext4(int v) {
    return (v >= 8) ? (v - 16) : v;
}

__global__ void int4_gather_gemv_kernel(
    const int8_t*        __restrict__ qweight,  // [out, in/2]
    const __nv_bfloat16* __restrict__ scales,   // [out, in/GROUP_SIZE]
    const __nv_bfloat16* __restrict__ x,        // [in]
    const int*           __restrict__ indices,  // [k] selected rows
    __nv_bfloat16*       __restrict__ out,      // [k]
    int in_features
) {
    __shared__ float warp_sums[NUM_WARPS];

    int j = blockIdx.x;              // selected output (0..k-1)
    int row = indices[j];            // weight row to use

    int packed_row = row * (in_features / 2);
    int n_groups = in_features / GROUP_SIZE;
    int scale_row = row * n_groups;

    float acc = 0.0f;
    int n_packed = in_features / 2;
    for (int pc = threadIdx.x; pc < n_packed; pc += BLOCK_THREADS) {
        int8_t byte = qweight[packed_row + pc];
        int lo = sext4(byte & 0x0F);
        int hi = sext4((byte >> 4) & 0x0F);

        int col_even = 2 * pc;
        int col_odd  = 2 * pc + 1;
        float s_even = __bfloat162float(scales[scale_row + (col_even / GROUP_SIZE)]);
        float s_odd  = __bfloat162float(scales[scale_row + (col_odd  / GROUP_SIZE)]);
        float xe = __bfloat162float(x[col_even]);
        float xo = __bfloat162float(x[col_odd]);
        acc += (lo * s_even) * xe;
        acc += (hi * s_odd)  * xo;
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    float wsum = warpReduceSum(acc);
    if (lane_id == 0) warp_sums[warp_id] = wsum;
    __syncthreads();

    if (threadIdx.x < NUM_WARPS) {
        float fsum = warp_sums[threadIdx.x];
        fsum = warpReduceSum(fsum);
        if (threadIdx.x == 0)
            out[j] = __float2bfloat16_rn(fsum);
    }
}

torch::Tensor int4_gather_gemv_impl(
    const torch::Tensor& qweight,   // int8  [out, in/2]
    const torch::Tensor& scales,    // bf16  [out, in/128]
    const torch::Tensor& x,         // bf16  [in]
    const torch::Tensor& indices    // int32 [k]
) {
    int in_features = qweight.size(1) * 2;
    int k = indices.size(0);

    auto out = torch::empty({k},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

    int4_gather_gemv_kernel<<<k, BLOCK_THREADS>>>(
        qweight.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        in_features
    );
    return out;
}

TORCH_LIBRARY(int4_gather_ops, m) {
    m.def("int4_gather_gemv(Tensor qweight, Tensor scales, Tensor x, Tensor indices) -> Tensor");
}
TORCH_LIBRARY_IMPL(int4_gather_ops, CUDA, m) {
    m.impl("int4_gather_gemv", &int4_gather_gemv_impl);
}