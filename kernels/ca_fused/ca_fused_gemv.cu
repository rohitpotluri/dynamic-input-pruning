// ca_fused_gemv.cu — Cache-aware fused int4 GEMV for CA-DIP up_proj.
//
// For each of the k selected channels, its weight row lives in one of two
// places: the resident GPU cache (a "hit") or the streamed staging buffer
// (a "miss"). This kernel reads each row from the correct source, dequantizes
// int4 inline, and computes the dot product with x, fusing the dual-source
// gather + dequant + GEMV in a single pass.
//
// Per selected output j (0..k-1):
//   src[j]  : 0 = hit (resident cache), 1 = miss (streamed buffer)
//   row[j]  : row index within that source
//   out[j]  = sum_c dequant(W_src[row][c]) * x[c]
//
// Inputs:
//   res_qw  int8  [cache_n, in/2]      resident cache weights
//   res_sc  bf16  [cache_n, in/128]
//   str_qw  int8  [k,       in/2]      streamed (miss) weights (first n_miss used)
//   str_sc  bf16  [k,       in/128]
//   x       bf16  [in]
//   src     int32 [k]                  0 = hit, 1 = miss
//   row     int32 [k]                  row within the chosen source
//   out     bf16  [k]
//
// One block per selected output j; the block reduces over in. Built for sm_75.

#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define BT 256
#define WARP 32
#define NW (BT / WARP)
#define GSZ 128

__device__ __forceinline__ float wred(float v) {
    #pragma unroll
    for (int o = WARP/2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ int sx4(int v) { return (v >= 8) ? (v - 16) : v; }

__global__ void ca_fused_gemv_kernel(
    const int8_t*        __restrict__ res_qw, const __nv_bfloat16* __restrict__ res_sc,
    const int8_t*        __restrict__ str_qw, const __nv_bfloat16* __restrict__ str_sc,
    const __nv_bfloat16* __restrict__ x,
    const int*           __restrict__ src, const int* __restrict__ row,
    __nv_bfloat16*       __restrict__ out,
    int in_features)
{
    __shared__ float wsum[NW];
    int j = blockIdx.x;
    int is_miss = src[j];
    int r = row[j];

    // source pointers for this row: resident cache on a hit, staging on a miss
    const int8_t*        qw = is_miss ? str_qw : res_qw;
    const __nv_bfloat16*  sc = is_miss ? str_sc : res_sc;

    int n_packed = in_features / 2;
    int n_groups = in_features / GSZ;
    int prow = r * n_packed;
    int srow = r * n_groups;

    float acc = 0.0f;
    for (int pc = threadIdx.x; pc < n_packed; pc += BT) {
        int8_t byte = qw[prow + pc];
        int lo = sx4(byte & 0x0F);
        int hi = sx4((byte >> 4) & 0x0F);
        int ce = 2*pc, co = 2*pc + 1;
        float se = __bfloat162float(sc[srow + (ce / GSZ)]);
        float so = __bfloat162float(sc[srow + (co / GSZ)]);
        acc += (lo * se) * __bfloat162float(x[ce]);
        acc += (hi * so) * __bfloat162float(x[co]);
    }
    int wid = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    float w = wred(acc);
    if (lane == 0) wsum[wid] = w;
    __syncthreads();
    if (threadIdx.x < NW) {
        float f = wred(wsum[threadIdx.x]);
        if (threadIdx.x == 0) out[j] = __float2bfloat16_rn(f);
    }
}

torch::Tensor ca_fused_gemv_impl(
    const torch::Tensor& res_qw, const torch::Tensor& res_sc,
    const torch::Tensor& str_qw, const torch::Tensor& str_sc,
    const torch::Tensor& x,
    const torch::Tensor& src, const torch::Tensor& row)
{
    int in_features = x.size(0);
    int k = src.size(0);
    auto out = torch::empty({k},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));
    ca_fused_gemv_kernel<<<k, BT>>>(
        res_qw.data_ptr<int8_t>(), reinterpret_cast<const __nv_bfloat16*>(res_sc.data_ptr()),
        str_qw.data_ptr<int8_t>(), reinterpret_cast<const __nv_bfloat16*>(str_sc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        src.data_ptr<int>(), row.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        in_features);
    return out;
}

TORCH_LIBRARY(ca_fused_ops, m) {
    m.def("ca_fused_gemv(Tensor res_qw, Tensor res_sc, Tensor str_qw, Tensor str_sc, Tensor x, Tensor src, Tensor row) -> Tensor");
}
TORCH_LIBRARY_IMPL(ca_fused_ops, CUDA, m) {
    m.impl("ca_fused_gemv", &ca_fused_gemv_impl);
}