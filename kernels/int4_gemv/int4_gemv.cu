// int4_gemv.cu — int4 (g128, symmetric) dequant + GEMV for decode (seqlen == 1).
//
// Weight layout (matches quantize.py):
//   qweight : int8 [out_features, in_features/2]   two int4 packed per byte
//             low nibble  = even input column
//             high nibble = odd  input column
//             each nibble is a signed int4 in [-8, 7] (two's complement)
//   scales  : bf16 [out_features, in_features/128]  one scale per group of 128
//
// Per output row r:
//   out[r] = sum_c dequant(qweight[r, c]) * x[c],  dequant uses scales[r, c/128]
//
// One block per output row; the block reduces over the input dimension.
// bf16 in/out. Built for sm_75.

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

// sign-extend a 4-bit value (0..15) to signed int (-8..7)
__device__ __forceinline__ int sext4(int v) {
    return (v >= 8) ? (v - 16) : v;
}

__global__ void int4_dequant_gemv_kernel(
    const int8_t*        __restrict__ qweight,  // [out, in/2]
    const __nv_bfloat16* __restrict__ scales,   // [out, in/GROUP_SIZE]
    const __nv_bfloat16* __restrict__ x,        // [in]
    __nv_bfloat16*       __restrict__ out,      // [out]
    int in_features
) {
    __shared__ float warp_sums[NUM_WARPS];

    int row = blockIdx.x;
    int packed_row = row * (in_features / 2);
    int n_groups = in_features / GROUP_SIZE;
    int scale_row = row * n_groups;

    float acc = 0.0f;

    // each thread walks packed bytes; each byte covers two input columns
    int n_packed = in_features / 2;
    for (int pc = threadIdx.x; pc < n_packed; pc += BLOCK_THREADS) {
        int8_t byte = qweight[packed_row + pc];
        int lo = sext4(byte & 0x0F);          // even column = 2*pc
        int hi = sext4((byte >> 4) & 0x0F);   // odd  column = 2*pc + 1

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
            out[row] = __float2bfloat16_rn(fsum);
    }
}

torch::Tensor int4_gemv_impl(
    const torch::Tensor& qweight,   // int8  [out, in/2]
    const torch::Tensor& scales,    // bf16  [out, in/128]
    const torch::Tensor& x          // bf16  [in]
) {
    int out_features = qweight.size(0);
    int in_features  = qweight.size(1) * 2;

    auto out = torch::empty({out_features},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

    int4_dequant_gemv_kernel<<<out_features, BLOCK_THREADS>>>(
        qweight.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        in_features
    );
    return out;
}

TORCH_LIBRARY(int4_ops, m) {
    m.def("int4_gemv(Tensor qweight, Tensor scales, Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(int4_ops, CUDA, m) {
    m.impl("int4_gemv", &int4_gemv_impl);
}