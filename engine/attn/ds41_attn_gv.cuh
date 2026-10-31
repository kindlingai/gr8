// DeepSeek-V4.1-Flash-1017 attention A, decode GEMV half: the two kernels and their launchers over densev.cuh's MXFP8 gv_gemv_body.
// Included by the rank-side glue (ds41_attn_ops.cuh, inside the m4sim TU that already has densev.cuh through kernels/bind.cuh) and by the
// single-TU test (tests/ds41_attn_test.cu). never include it from ds41_attn.cu: densev.cuh has non-inline __global__ functions (gv_multi,
// k_i4pair, k_warm), so a second TU holding it fails to link ("multiple definition"). The attention TU launches these through
// ds41a::g_fn_xfrag / g_fn_gemv (ds41_attn.cuh), set by ds41_attn_set_gv.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstddef>
#include "ds41_attn_types.h"
#include "../kernels/densev.cuh"   // gv_gemv_body<Mxfp8>, pack_h2: global scope

namespace ds41a
{
// same arithmetic as ds41_attn.cuh f_e4m3 / f_e8m0 (own copies so this header does not need that one)
__device__ __forceinline__ float gvf_e4m3(uint8_t c)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
    return __half2float(__half(h));
}
__device__ __forceinline__ float gvf_e8m0(uint8_t b) { return __uint_as_float(b ? ((uint32_t)b << 23) : 0x00400000u); }

// ------------------------------------------------------------------------------------------------ decode GEMV (M <= 8)
// The decode projections stream the weights once instead of running the 64 x 64 tiled scalar GEMM above (96% of a layer's time at
// n = 128, and only N/64 CTAs at M = 1): the pack's MXFP8 GV records ([N/16][K/32][528], densev.cuh Mxfp8) are the mma A operand, the
// activations are the B operand (tokens = the 8 columns). The mma takes fp16, so the B operand is the DEQUANTIZED activation
// e4m3(code) * 2^(e-127) (exact in fp16 unless it is below 2^-24 or above 65504: the official act-quant numerics are kept); the
// weight's row scale multiplies each 32-group's fp32 partial (exact, a power of two). The result equals the generic GEMM up to fp32
// summation order. M > 8 and prefill keep the generic GEMM.
//
// Fragment table (densev.cuh layout) of `groups` activation blocks: [g][K/16][32 lanes][2] u32, lane l = (token t = l/4, c = l%4),
// reg 0 = {x[t][k0], x[t][k0+1]}, reg 1 = {x[t][k0+8], x[t][k0+9]}, k0 = 16 ks + 2c; columns >= M are zero. Group g reads codes at
// codes + g gc, scales at sc + g gs (the block-diagonal o_a: gc = 4096, gs = 128). grid = (K/16*64/256, groups), block 256.
__global__ void k_xfrag_q(const uint8_t* __restrict__ codes, int ldc, const uint8_t* __restrict__ sc, int lds, int M, int K, size_t gc, size_t gs,
                          uint32_t* __restrict__ xf)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K / 16 * 64) return;
    const int g = blockIdx.y, ks = i >> 6, l = (i >> 1) & 31, reg = i & 1, t = l >> 2;
    const int k0 = ks * 16 + reg * 8 + 2 * (l & 3);
    uint32_t v = 0;
    if (t < M)
    {
        const uint8_t* c = codes + g * gc + (size_t)t * ldc + k0;
        const float s = gvf_e8m0(sc[g * gs + (size_t)t * lds + (k0 >> 5)]);
        const float a = fminf(fmaxf(gvf_e4m3(c[0]) * s, -65504.f), 65504.f), b = fminf(fmaxf(gvf_e4m3(c[1]) * s, -65504.f), 65504.f);
        v = pack_h2(a, b);
    }
    xf[(size_t)g * K * 4 + i] = v;
}
// y[g][tok][N] (bf16, row stride ystride, group stride y_gs elements) = group g's records (out_tiles x K/32 x 528 B, contiguous per group)
// x group g's table. CTA = tpc output tiles x ks k-parts (tpc * ks <= 16 warps). grid = (ceil(out_tiles / tpc), groups).
__global__ void __launch_bounds__(512) k_gv_mx(const uint8_t* __restrict__ stream, const uint32_t* __restrict__ xf, int K, int out_tiles, int tpc, int ks, int M,
                                               bf16* y, int ystride, size_t y_gs)
{
    const int g = blockIdx.y;
    gv_gemv_body<Mxfp8, 1>(stream + (size_t)g * out_tiles * (K / 32) * 528, xf + (size_t)g * K * 4, K, out_tiles, tpc, ks, M, y + g * y_gs, ystride, gv::O_BF16,
                           (int)blockIdx.x);
}

// largest k-part count <= 16 that divides the K / 32 records of a tile
inline int gv_pick_ks(int K)
{
    for (int k = 16; k > 1; k--)
        if ((K / 32) % k == 0) return k;
    return 1;
}
// XfragFn
inline void gv_xfrag_launch(const uint8_t* codes, int ldc, const uint8_t* sc, int lds, int M, int K, int groups, size_t gc, size_t gs, uint32_t* xf,
                            cudaStream_t st)
{
    k_xfrag_q<<<dim3((K / 16 * 64 + 255) / 256, groups), 256, 0, st>>>(codes, ldc, sc, lds, M, K, gc, gs, xf);
}
// GemvFn (tpc, ks: 0 = default; ks that does not divide K / 32 falls back to gv_pick_ks; tpc * ks <= 16 warps)
inline void gv_gemv_launch(const uint32_t* xf, const Fp8Mat& W, bf16* Y, int ldy, int M, int groups, size_t y_gs, cudaStream_t st, int tpc, int ks)
{
    const int K = W.K, ot = W.N / 16;
    if (!ks || (K / 32) % ks) ks = gv_pick_ks(K);
    if (tpc < 1) tpc = 1;
    if (tpc * ks > 16) tpc = 16 / ks > 0 ? 16 / ks : 1;
    k_gv_mx<<<dim3((ot + tpc - 1) / tpc, groups), tpc * ks * 32, 0, st>>>(W.gv, xf, K, ot, tpc, ks, M, Y, ldy, y_gs);
}
}   // namespace ds41a
