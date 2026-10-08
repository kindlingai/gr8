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

// Single launch with no epilogue kernel. CTA = (128-column block nb, k part p) of 256 * SUB threads, warp = (sub-part q,
// tile column j). The SUB sub-parts of a CTA reduce through shared memory, each CTA writes its fp32 partial [M][128] to
// part[nb][p], and the last CTA of block nb (a counter, as k1_tr3_fused) sums the ks partials in order, rounds to fp16,
// applies H128 and svh and writes y. The counters return to zero. xa = the input already rotated (the producer's job;
// k_tr3d_rot stands in for it). UNR = tiles in flight per warp.
template <int BITS, int MT, int SUB, int UNR>
__global__ void __launch_bounds__(256 * SUB) k_tr3d_gemv1(const uint8_t* w, const u32* xa, int K, int N, int ks, int M, const __half* svh,
                                                          float* part, int* cnt, __half* y)
{
    constexpr int TB = 32 * BITS;
    __shared__ float4 red[SUB > 1 ? (SUB - 1) * 8 * MT * 32 : 1];
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = warp & 7, q = warp >> 3;
    const int nb = blockIdx.x / ks, p = blockIdx.x % ks;
    const int K16 = K / 16, steps = K16 / (ks * SUB), s0 = (p * SUB + q) * steps;
    const uint8_t* base = w + static_cast<size_t>(nb) * K16 * 8 * TB + (static_cast<size_t>(s0) * 8 + j) * TB;
    const u32* xb = xa + static_cast<size_t>(s0) * 64 + lane * 2;
    Tr3Lane<BITS, true> ld;
    ld.init(lane);
    float acc[MT][4];
    #pragma unroll
    for (int g = 0; g < MT; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    u32 na[UNR];
    #pragma unroll
    for (int u = 0; u < UNR; ++u) na[u] = u < steps ? ld.loadw(base + static_cast<size_t>(u) * 8 * TB, lane) : 0u;
    for (int s = 0; s < steps; s += UNR)
    {
        u32 wa[UNR];
        #pragma unroll
        for (int u = 0; u < UNR; ++u) wa[u] = na[u];
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
            if (s + UNR + u < steps) na[u] = ld.loadw(base + static_cast<size_t>(s + UNR + u) * 8 * TB, lane);
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
        {
            if (s + u >= steps) break;
            u32 f0[2], f1[2];
            ld.decodew(wa[u], f0, f1);
            #pragma unroll
            for (int g = 0; g < MT; ++g)
            {
                const u32* xg = xb + static_cast<size_t>(g) * K16 * 64 + static_cast<size_t>(s + u) * 64;
                mma16816(acc[g], f0[0], f1[0], f0[1], f1[1], __ldg(xg), __ldg(xg + 1));
            }
        }
    }
    if (SUB > 1)
    {
        if (q) 
            #pragma unroll
            for (int g = 0; g < MT; ++g) red[((q - 1) * 8 * MT + j * MT + g) * 32 + lane] = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        __syncthreads();
        if (!q)
            for (int r = 1; r < SUB; ++r)
                #pragma unroll
                for (int g = 0; g < MT; ++g)
                {
                    const float4 v = red[((r - 1) * 8 * MT + j * MT + g) * 32 + lane];
                    acc[g][0] += v.x; acc[g][1] += v.y; acc[g][2] += v.z; acc[g][3] += v.w;
                }
    }
    // the CTA's partial: part[nb][p][m][128]
    float* pp = part + (static_cast<size_t>(nb) * ks + p) * M * 128;
    if (!q)
    {
        const int nl = lane >> 2, c = lane & 3, col = j * 16 + nl;
        #pragma unroll
        for (int g = 0; g < MT; ++g)
        {
            const int m0 = g * 8 + 2 * c;
            if (m0 < M) { pp[m0 * 128 + col] = acc[g][0]; pp[m0 * 128 + col + 8] = acc[g][2]; }
            if (m0 + 1 < M) { pp[(m0 + 1) * 128 + col] = acc[g][1]; pp[(m0 + 1) * 128 + col + 8] = acc[g][3]; }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(cnt + nb, 1) == ks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float* pb = part + static_cast<size_t>(nb) * ks * M * 128;
    for (int m = warp; m < M; m += 8 * SUB)
    {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        for (int r = 0; r < ks; ++r)
        {
            const float4 a = __ldcg(reinterpret_cast<const float4*>(pb + (static_cast<size_t>(r) * M + m) * 128 + lane * 4));
            v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = rnd_h(v[i]);
        had128(v, lane);
        const int n0 = nb * 128 + lane * 4;
        const uint2 sr = *reinterpret_cast<const uint2*>(svh + n0);
        const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
        __half2* o = reinterpret_cast<__half2*>(y + static_cast<size_t>(m) * N + n0);
        o[0] = __floats2half2_rn(v[0] * s01.x, v[1] * s01.y);
        o[1] = __floats2half2_rn(v[2] * s23.x, v[3] * s23.y);
    }
    if (threadIdx.x == 0) cnt[nb] = 0;
}

struct Dense
{
    const uint8_t* w = nullptr;       // tiles
    const __half *suh = nullptr, *svh = nullptr;
    int K = 0, N = 0, bits = 4, ks = 8;
    u32* xa = nullptr;                // [ceil(M/8)][K/16][64] scratch, zero rows >= M once
    float* part = nullptr;            // [ks][M][N] scratch
    int* cnt = nullptr;               // [N / 128] zero (launch1)
    int sub = 2;                      // launch1: k sub-parts per CTA (1 or 2)
};

// one launch on input already rotated into d.xa (launch1); M <= 24
inline void launch1(const Dense& d, int M, __half* y, cudaStream_t st)
{
    const int mt = (M + 7) / 8, grid = d.N / 128 * d.ks;
#define TR3D_G1(B, T, S) k_tr3d_gemv1<B, T, S, 16><<<grid, 256 * S, 0, st>>>(d.w, d.xa, d.K, d.N, d.ks, M, d.svh, d.part, d.cnt, y)
#define TR3D_M1(B, S) do { if (mt == 1) TR3D_G1(B, 1, S); else if (mt == 2) TR3D_G1(B, 2, S); else TR3D_G1(B, 3, S); } while (0)
    if (d.sub == 2) { if (d.bits == 4) TR3D_M1(4, 2); else if (d.bits == 3) TR3D_M1(3, 2); else TR3D_M1(2, 2); }
    else { if (d.bits == 4) TR3D_M1(4, 1); else if (d.bits == 3) TR3D_M1(3, 1); else TR3D_M1(2, 1); }
#undef TR3D_M1
#undef TR3D_G1
}

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
