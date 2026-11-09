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
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
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

// ------------------------------------------------------------------------------------------------ quad records (GR8_DS41_GV_QUAD=1)
// GR8_DS41_GV_QUAD=1 (default off). The pack's MXFP8 group is 528 B (tile0 256 B, tile1 256 B, 8 x u16 scales = 16 B); 528 mod 64 = 16, so
// the groups sit off the 64 B bursts. A QUAD = 4 groups of one tile = 2112 B = 33 bursts, repacked in place at bind:
//   [j = 0..3: 512 B, lane l's 16 B = group j {tile0 bytes 8l..8l+7, tile1 bytes 8l..8l+7}] [64 B: row r = 0..7: {g0..g3 scale u16 of row r}]
// k_gv_mx_q reads a quad with 4 x ld.v4 + 1 x ld.v2 a lane. Each lane gets the same c[4] words and scale as GvGroup<Mxfp8>::load, the same mma
// order: outputs bit-identical to k_gv_mx. Needs tpc 1 and NGP = K/32/ks % 4 == 0 (whole quads a warp).
// g_gv_quad holds the repacked record pointers (host). gv_gemv_launch_q sends exactly those to k_gv_mx_q and every other weight to gv_gemv_launch.
// Nothing else may read a repacked record: the attention GEMV hook is the only reader of q_b / o_a / o_b .mx.gv (the GEMM reads .mx.codes).
// ds41_attn_bind (ds41_attn_ops.cuh) reads the knob, repacks and installs gv_gemv_launch_q.
// one CTA (256 thr) per quad, in place
__global__ void k_quad_pack(uint8_t* w, size_t nquads)
{
    __shared__ uint8_t sm[2112];
    for (size_t q = blockIdx.x; q < nquads; q += gridDim.x)
    {
        uint8_t* b = w + q * 2112;
        for (int i = threadIdx.x; i < 2112; i += blockDim.x) sm[i] = b[i];
        __syncthreads();
        for (int i = threadIdx.x; i < 2048; i += blockDim.x)   // tile words
        {
            const int j = i >> 9, o = i & 511, l = o >> 4, h = (o >> 3) & 1, k = o & 7;
            b[i] = sm[j * 528 + h * 256 + l * 8 + k];
        }
        for (int i = threadIdx.x; i < 64; i += blockDim.x)     // scales: row r, group j, byte k
        {
            const int r = i >> 3, j = (i >> 1) & 3, k = i & 1;
            b[2048 + i] = sm[j * 528 + 512 + r * 2 + k];
        }
        __syncthreads();
    }
}
// one warp: groups g0..g1-1 (g0 % 4 == 0, (g1 - g0) % 4 == 0) of one tile, one quad read ahead
template <int NT>
__device__ __forceinline__ void gv_quad_part(const uint8_t* tile, const u32* bf, int ts, int g0, int g1, float (&acc)[NT][4])
{
    const int lane = threadIdx.x & 31;
    const int NQ = (g1 - g0) / 4;
    const uint8_t* base = tile + (size_t)g0 * 528;
    u32 nw[4][4], ns[2];
    if (NQ > 0)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j) gv::ld_nc_v4(base + j * 512 + lane * 16, nw[j]);
        gv::ld_nc_v2(base + 2048 + (lane >> 2) * 8, ns);
    }
    for (int q = 0; q < NQ; ++q)
    {
        u32 cw[4][4], cs[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { cw[j][0] = nw[j][0]; cw[j][1] = nw[j][1]; cw[j][2] = nw[j][2]; cw[j][3] = nw[j][3]; }
        cs[0] = ns[0]; cs[1] = ns[1];
        if (q + 1 < NQ)
        {
            const uint8_t* nb = base + (size_t)(q + 1) * 2112;
            #pragma unroll
            for (int j = 0; j < 4; ++j) gv::ld_nc_v4(nb + j * 512 + lane * 16, nw[j]);
            gv::ld_nc_v2(nb + 2048 + (lane >> 2) * 8, ns);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            GvGroup<Mxfp8> gg;
            gg.c[0] = cw[j][0]; gg.c[1] = cw[j][1]; gg.c[2] = cw[j][2]; gg.c[3] = cw[j][3];
            gg.s = (cs[j >> 1] >> (16 * (j & 1))) & 0xffffu;
            gg.template run<NT>(bf, (g0 + 4 * q + j) * Mxfp8::G, acc, ts);
        }
    }
}
// k_gv_mx at tpc 1 with the quad warp part (the rest is gv_gemv_body<Mxfp8, 1>'s code). grid = (out_tiles, groups), block ks x 32.
__global__ void __launch_bounds__(512) k_gv_mx_q(const uint8_t* __restrict__ stream, const uint32_t* __restrict__ xf, int K, int out_tiles, int ks, int M,
                                                 bf16* y, int ystride, size_t y_gs)
{
    __shared__ float4 red[32][32];
    const int g = blockIdx.y;
    const uint8_t* st = stream + (size_t)g * out_tiles * (K / 32) * 528;
    const u32* xfrag = xf + (size_t)g * K * 4;
    bf16* yg = y + g * y_gs;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int part = warp % ks, ot = blockIdx.x;
    const int NG2 = K / 32, NGP = NG2 / ks;
    const bool on = ot < out_tiles;
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    if (on) gv_quad_part<1>(st + (size_t)ot * NG2 * 528, xfrag + lane * 2, K * 4, part * NGP, (part + 1) * NGP, acc);
    if (ks > 1) gv_reduce<1>(red, acc, on, part, warp, lane, ks);
    if (!on || part) return;
    const int r = lane >> 2, c = lane & 3;
    const size_t o = (size_t)ot * 16 + r;
    const int m0 = 2 * c;
    if (m0 < M) { gv::put(yg, (size_t)m0 * ystride + o, acc[0][0], gv::O_BF16); gv::put(yg, (size_t)m0 * ystride + o + 8, acc[0][2], gv::O_BF16); }
    if (m0 + 1 < M) { gv::put(yg, (size_t)(m0 + 1) * ystride + o, acc[0][1], gv::O_BF16); gv::put(yg, (size_t)(m0 + 1) * ystride + o + 8, acc[0][3], gv::O_BF16); }
}
inline std::unordered_set<const uint8_t*> g_gv_quad;   // record pointers in the quad layout (only k_gv_mx_q may read them)
// the launcher's tpc / ks rule (gv_gemv_launch), for the bind-time check
inline bool gv_quad_fits(int K, int tpc, int ks)
{
    if (!ks || (K / 32) % ks) ks = gv_pick_ks(K);
    if (tpc < 1) tpc = 1;
    return tpc == 1 && (K / 32) % 4 == 0 && ((K / 32) / ks) % 4 == 0;
}
// bind (before any capture): repack W's `groups` record blocks in place on st when the launch shape takes whole quads; 1 = repacked
inline int gv_quad_bind(const Fp8Mat& W, int groups, int tpc, int ks, cudaStream_t st)
{
    if (!W.gv || g_gv_quad.count(W.gv) || (W.N & 15) || !gv_quad_fits(W.K, tpc, ks)) return 0;
    const size_t bytes = (size_t)groups * (W.N / 16) * (W.K / 32) * 528;
    k_quad_pack<<<1024, 256, 0, st>>>(const_cast<uint8_t*>(W.gv), bytes / 2112);
    g_gv_quad.insert(W.gv);
    return 1;
}
// GemvFn under GR8_DS41_GV_QUAD=1: quad records -> k_gv_mx_q, every other weight -> the stock launcher
inline void gv_gemv_launch_q(const uint32_t* xf, const Fp8Mat& W, bf16* Y, int ldy, int M, int groups, size_t y_gs, cudaStream_t st, int tpc, int ks)
{
    if (!g_gv_quad.count(W.gv))
    {
        gv_gemv_launch(xf, W, Y, ldy, M, groups, y_gs, st, tpc, ks);
        return;
    }
    const int K = W.K, ot = W.N / 16;
    if (!gv_quad_fits(K, tpc, ks))
    {
        fprintf(stderr, "GR8_DS41_GV_QUAD: quad records N=%d K=%d launched with tpc %d ks %d (needs tpc 1, whole quads a k-part)\n", W.N, K, tpc, ks);
        abort();
    }
    if (!ks || (K / 32) % ks) ks = gv_pick_ks(K);
    static bool told = false;
    if (!told)
    {
        told = true;
        fprintf(stderr, "# GR8_DS41_GV_QUAD=1: %zu weight record sets in the quad layout (k_gv_mx_q)\n", g_gv_quad.size());
    }
    k_gv_mx_q<<<dim3(ot, groups), ks * 32, 0, st>>>(W.gv, xf, K, ot, ks, M, Y, ldy, y_gs);
}
// ------------------------------------------------------------------------------------------------ q_a + kv in one launch (GR8_DS_ATTN_PROJ_FUSE)
// GemvDualFn: matrix A (q_a, N = 1280) and matrix B (kv, N = 512) over the SAME fragment table (K = 5120). One grid: CTAs [0, gA) run A's tiles, the rest
// run B's, each through gv_gemv_body<Mxfp8, 1> exactly as k_gv_mx does (same tpc / ks, same body, B's tile index = blockIdx.x - gA), so both outputs are
// bit-identical to two k_gv_mx launches. The CTA shape (tpc x ks warps) is gv_gemv_launch's rule. Quad-layout records or different K: two stock launches.
__global__ void __launch_bounds__(512) k_gv_mx_dual(const uint8_t* __restrict__ sa, const uint8_t* __restrict__ sb, const uint32_t* __restrict__ xf, int K, int otA,
                                                    int otB, int gA, int tpc, int ks, int M, bf16* ya, int ldya, bf16* yb, int ldyb)
{
    if ((int)blockIdx.x < gA) gv_gemv_body<Mxfp8, 1>(sa, xf, K, otA, tpc, ks, M, ya, ldya, gv::O_BF16, (int)blockIdx.x);
    else gv_gemv_body<Mxfp8, 1>(sb, xf, K, otB, tpc, ks, M, yb, ldyb, gv::O_BF16, (int)blockIdx.x - gA);
}
inline void gv_gemv_dual_launch(const uint32_t* xf, const Fp8Mat& WA, bf16* YA, int ldya, const Fp8Mat& WB, bf16* YB, int ldyb, int M, cudaStream_t st, int tpc,
                                int ks)
{
    if (WA.K != WB.K || g_gv_quad.count(WA.gv) || g_gv_quad.count(WB.gv))
    {
        gv_gemv_launch_q(xf, WA, YA, ldya, M, 1, 0, st, tpc, ks);
        gv_gemv_launch_q(xf, WB, YB, ldyb, M, 1, 0, st, tpc, ks);
        return;
    }
    const int K = WA.K, otA = WA.N / 16, otB = WB.N / 16;
    if (!ks || (K / 32) % ks) ks = gv_pick_ks(K);
    if (tpc < 1) tpc = 1;
    if (tpc * ks > 16) tpc = 16 / ks > 0 ? 16 / ks : 1;
    const int gA = (otA + tpc - 1) / tpc, gB = (otB + tpc - 1) / tpc;
    k_gv_mx_dual<<<gA + gB, tpc * ks * 32, 0, st>>>(WA.gv, WB.gv, xf, K, otA, otB, gA, tpc, ks, M, YA, ldya, YB, ldyb);
}
}   // namespace ds41a
