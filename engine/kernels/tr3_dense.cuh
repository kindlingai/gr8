// Dense trellis GEMV (exlr8 v2 dense weights, #65; measurement stage): y[M][N] = H128(f16(H128(x * suh) @ What)) * svh
// for one rank's slice of q_b or o_proj stored as trellis K2-K4 tiles with the E4M3-rounded codebook (exlr8quant SPEC.md
// 5.10). The v2 dense file layout is not specified yet (SPEC 13.13); this takes the experts' fc1 order per 128-column
// output block, [K/16 k16][8 col16][32 BITS bytes], which SPEC 6.6 says dense trellis matrices share.
//
//   k_tr3d_rot   x fp16 [M][ldx] -> xa: f16(H128(x * suh)) as mma B fragment tables, one [K/16][32][2] table per group
//                of 8 tokens (the bind::Xfrag layout). The input rotation is done once for the whole matrix.
//   k_tr3d_gemv  CTA = (128-column block, k part), 8 warps = the block's 8 tiles of 16 columns. Each decoded tile is the
//                A operand of MT mma calls, one per token group, so 9-24 rows decode the weights once.
//                Partial sums fp32 -> part[k part][M][N].
//   k_tr3d_fin   sums the k parts in order, rounds to fp16, output Hadamard, times svh -> y fp16 [M][N].
#pragma once
#include "tr3.cuh"

namespace tr3d
{

// grid (K / 128, M), block 32: warp = (128-block, token m)
__global__ void __launch_bounds__(32) k_tr3d_rot(const __half* x, int ldx, int K, const __half* suh, u32* xa)
{
    const int blk = blockIdx.x, m = blockIdx.y, lane = threadIdx.x;
    const int k0 = blk * 128 + lane * 4;
    const uint2 xr = *reinterpret_cast<const uint2*>(x + static_cast<size_t>(m) * ldx + k0);
    const uint2 sr = *reinterpret_cast<const uint2*>(suh + k0);
    const float2 x01 = __half22float2(as_h2(xr.x)), x23 = __half22float2(as_h2(xr.y));
    const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
    float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
    had128(v, lane);
    // table[g][((ks * 32 + m8 * 4 + c) * 2 + r)] holds halfs (16 ks + 8 r + 2 c, + 1) of token m8 of group g
    __half* t = reinterpret_cast<__half*>(xa + static_cast<size_t>(m >> 3) * (K / 16) * 64);
    const int m8 = m & 7;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int k = k0 + i, ks = k >> 4, w = k & 15, r = w >> 3, c = (w & 7) >> 1;
        t[((static_cast<size_t>(ks) * 32 + m8 * 4 + c) * 2 + r) * 2 + (w & 1)] = __float2half_rn(v[i]);
    }
}

// grid (N / 128) * ks, block 256. w = the matrix's tiles, xa = MT fragment tables (rows >= M zero), part [ks][M][N].
template <int BITS, bool E4, int MT>
__global__ void __launch_bounds__(256) k_tr3d_gemv(const uint8_t* w, const u32* xa, int K, int N, int ks, int M, float* part)
{
    constexpr int TB = 32 * BITS;
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int nb = blockIdx.x / ks, p = blockIdx.x % ks;
    const int K16 = K / 16, steps = K16 / ks, s0 = p * steps;
    const uint8_t* base = w + static_cast<size_t>(nb) * K16 * 8 * TB + (static_cast<size_t>(s0) * 8 + j) * TB;
    const u32* xb = xa + static_cast<size_t>(s0) * 64 + lane * 2;
    Tr3Lane<BITS, E4> ld;
    ld.init(lane);
    float acc[MT][4];
    #pragma unroll
    for (int g = 0; g < MT; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    u32 na[TR3_UNR];
    #pragma unroll
    for (int q = 0; q < TR3_UNR; ++q) na[q] = q < steps ? ld.loadw(base + static_cast<size_t>(q) * 8 * TB, lane) : 0u;
    for (int s = 0; s < steps; s += TR3_UNR)
    {
        u32 wa[TR3_UNR];
        #pragma unroll
        for (int q = 0; q < TR3_UNR; ++q) wa[q] = na[q];
        if (s + TR3_UNR < steps)
        {
            #pragma unroll
            for (int q = 0; q < TR3_UNR; ++q) na[q] = ld.loadw(base + static_cast<size_t>(s + TR3_UNR + q) * 8 * TB, lane);
        }
        #pragma unroll
        for (int q = 0; q < TR3_UNR; ++q)
        {
            if (s + q >= steps) break;
            u32 f0[2], f1[2];
            ld.decodew(wa[q], f0, f1);
            #pragma unroll
            for (int g = 0; g < MT; ++g)
            {
                const u32* xg = xb + static_cast<size_t>(g) * K16 * 64 + static_cast<size_t>(s + q) * 64;
                mma16816(acc[g], f0[0], f1[0], f0[1], f1[1], __ldg(xg), __ldg(xg + 1));
            }
        }
    }
    // acc[g]: c0, c1 = (col l/4, tokens 8g + 2c, 2c + 1), c2, c3 = (col l/4 + 8, same)
    const int nl = lane >> 2, c = lane & 3, col = nb * 128 + j * 16 + nl;
    #pragma unroll
    for (int g = 0; g < MT; ++g)
    {
        const int m0 = g * 8 + 2 * c;
        float* q0 = part + (static_cast<size_t>(p) * M + m0) * N + col;
        if (m0 < M) { q0[0] = acc[g][0]; q0[8] = acc[g][2]; }
        if (m0 + 1 < M) { q0[N] = acc[g][1]; q0[N + 8] = acc[g][3]; }
    }
}

// grid (N / 128, M), block 32
__global__ void __launch_bounds__(32) k_tr3d_fin(const float* part, int ks, int M, int N, const __half* svh, __half* y)
{
    const int nb = blockIdx.x, m = blockIdx.y, lane = threadIdx.x, n0 = nb * 128 + lane * 4;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int p = 0; p < ks; ++p)
    {
        const float4 a = *reinterpret_cast<const float4*>(part + (static_cast<size_t>(p) * M + m) * N + n0);
        v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = rnd_h(v[i]);
    had128(v, lane);
    const uint2 sr = *reinterpret_cast<const uint2*>(svh + n0);
    const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
    __half2* o = reinterpret_cast<__half2*>(y + static_cast<size_t>(m) * N + n0);
    o[0] = __floats2half2_rn(v[0] * s01.x, v[1] * s01.y);
    o[1] = __floats2half2_rn(v[2] * s23.x, v[3] * s23.y);
}

struct Dense
{
    const uint8_t* w = nullptr;       // tiles
    const __half *suh = nullptr, *svh = nullptr;
    int K = 0, N = 0, bits = 4, ks = 8;
    u32* xa = nullptr;                // [ceil(M/8)][K/16][64] scratch, zero rows >= M once
    float* part = nullptr;            // [ks][M][N] scratch
};

// x [M][ldx] -> y [M][N]; M <= 24
inline void launch(const Dense& d, const __half* x, int ldx, int M, __half* y, cudaStream_t st)
{
    k_tr3d_rot<<<dim3(d.K / 128, M), 32, 0, st>>>(x, ldx, d.K, d.suh, d.xa);
    const int grid = d.N / 128 * d.ks, mt = (M + 7) / 8;
#define TR3D_GO(B, T) k_tr3d_gemv<B, true, T><<<grid, 256, 0, st>>>(d.w, d.xa, d.K, d.N, d.ks, M, d.part)
#define TR3D_MT(B) do { if (mt == 1) TR3D_GO(B, 1); else if (mt == 2) TR3D_GO(B, 2); else TR3D_GO(B, 3); } while (0)
    if (d.bits == 4) TR3D_MT(4);
    else if (d.bits == 3) TR3D_MT(3);
    else TR3D_MT(2);
#undef TR3D_MT
#undef TR3D_GO
    k_tr3d_fin<<<dim3(d.N / 128, M), 32, 0, st>>>(d.part, d.ks, M, d.N, d.svh, y);
}

}   // namespace tr3d
