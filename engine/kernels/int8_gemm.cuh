// Native exlr8 int8 gG prefill GEMM, Y[M][N] = X[M][K] @ W^T, reading the exact same single-copy
// [N/16][K/G][GR] weight record as decode (densev.cuh Int8v<GRP>).
//
// Zero MXFP8 re-quantization, zero FP8 activation quantization, zero duplicate weight storage (~1.031 B/weight at g64).
// Activations X are read directly as FP16 through a 3-stage BK=32 cp.async shared-memory pipeline (two bank-conflict-free
// 48 B/row K16 slabs per stage; all 256 threads issue 1 cp.async.cg 16B load each at MT=8); each warp owns one 16-row
// weight output tile, loads coalesced 256-byte 16x16 INT8 tiles straight into registers (ld.global.nc.v2.u32), decodes
// q8 -> FP16 via PRMT+HSUB2, applies each group's FP16 scale s8 via 4x mul.f16x2 (exact FP16-TC precision, eliminating
// separate group accumulators so all GRP in {1,2,4,8} stay <= 64 regs/thread at 4 CTAs/SM), and runs 16 FP16-TC
// mma16816 tiles per barrier (fp32 accumulate).
//
// Works on both sm_121a (DGX Spark GB10) and sm_120a (RTX 5090).
#pragma once
#include "densev.cuh"

namespace i8gemm
{

struct Rs
{
    char* base = nullptr;
    size_t pstride = 0, ss = 0;
    const uint64_t* stepp = nullptr;
    int M = 0, nr = 0;
};

__device__ __forceinline__ uint16_t* rs_row(const Rs& o, int row, int ld)
{
    const int p = ((row + 1) * o.nr - 1) / o.M;
    return reinterpret_cast<uint16_t*>(o.base + (*o.stepp & 1) * o.pstride + p * o.ss) +
           static_cast<size_t>(row - p * o.M / o.nr) * ld;
}

struct Args
{
    const __half* x = nullptr;        // [M][ldx] FP16 activations (16-byte aligned rows, ldx % 8 == 0)
    const uint8_t* w = nullptr;       // [N/16][K/G][GR] native Int8v<GRP> records
    void* y = nullptr;                // [M][ldy] FP16 (bf16==0), BF16 (bf16==1), or FP32 (bf16==2)
    int M = 0, N = 0, K = 0;
    int ldx = 0, ldy = 0;
    int bf16 = 0;                     // 0 = FP16, 1 = BF16, 2 = FP32
    int grp = 0;                      // GRP = G / 16 in {1, 2, 4, 8}
    int rgrp = 0;                     // grouped raster width along M-blocks (0 = auto)
    // Optional paired matrix sharing the same X, K, M, and grp (launch_pair)
    const uint8_t* w2 = nullptr;
    void* y2 = nullptr;
    int N2 = 0, ldy2 = 0, bf162 = 0;
};

__device__ __forceinline__ void cp16(void* smem_dst, const void* glob_src)
{
    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(glob_src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ u32 hmul2_u32(u32 a, u32 b)
{
    u32 r;
    asm("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ const uint8_t* tile_addr(const uint8_t* wtile, int ks, int grp_shift, int grp_mask, int GR)
{
    return wtile + static_cast<size_t>(ks >> grp_shift) * GR + (ks & grp_mask) * 256;
}

__device__ __forceinline__ const uint8_t* grp_addr(const uint8_t* wtile, int ks, int grp_shift, int GR)
{
    return wtile + static_cast<size_t>(ks >> grp_shift) * GR;
}

__device__ __forceinline__ u32 ld_scale_i8(const uint8_t* gp, int lane, int grp)
{
    return ld_nc(gp + grp * 256 + (lane >> 2) * 4);
}

// CTA = 8 warps (256 threads): BN = 128 output cols (1 warp = 16 output cols = 1 output tile ot),
// BM_ROWS = MT * 8 prompt rows (MT = 4 -> 32 rows, MT = 8 -> 64 rows), BK = 32 (2 K16 slabs per stage).
// Static smem: 3 stages x 2 slabs x BM_ROWS x 48 B (9 KB for MT=4, 18 KB for MT=8).
template <int MT, bool RS = false, bool F32 = false>
__global__ void __launch_bounds__(256) k_gemm_i8g(Args a0, Rs rs = Rs())
{
    using C = Int8v<1>;
    constexpr int STAGES = 3;
    constexpr int BM_ROWS = MT * 8;
    constexpr int X_LD_U32 = 12;      // 48 B/row (24 halfs): 16B aligned for cp.async, 0 bank conflicts for 32-lane B loads
    constexpr int X_LD_U4 = 3;        // 48 B = 3 x uint4
    __shared__ __align__(16) u32 sm[STAGES][2][BM_ROWS * X_LD_U32];

    Args a = a0;
    const int grp = a.grp;
    const int grp_mask = grp - 1;
    const int grp_shift = __ffs(grp) - 1;
    const int GR = grp * 256 + 32;
    const int nb1 = (a0.N + 127) / 128;
    const int nb_tot = nb1 + (a0.w2 ? (a0.N2 + 127) / 128 : 0);
    const int mb_tot = (a0.M + BM_ROWS - 1) / BM_ROWS;
    int bx, by;
    if (a0.rgrp > 1 && nb_tot > 1 && mb_tot > 1)
    {
        const int lin = static_cast<int>(blockIdx.y) * gridDim.x + static_cast<int>(blockIdx.x);
        const int g = a0.rgrp;
        const int strip_size = g * nb_tot;
        const int strip = lin / strip_size;
        const int rem = lin % strip_size;
        const int m0_s = strip * g;
        const int mw = min(g, mb_tot - m0_s);
        bx = rem / mw;
        by = m0_s + (rem % mw);
    }
    else
    {
        bx = static_cast<int>(blockIdx.x);
        by = static_cast<int>(blockIdx.y);
    }
    if (a0.w2 && bx >= nb1)
    {
        bx -= nb1;
        a.w = a0.w2; a.N = a0.N2; a.y = a0.y2; a.ldy = a0.ldy2; a.bf16 = a0.bf162;
    }

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ot = bx * 8 + warp;
    const int m0 = by * BM_ROWS;
    const int K16 = a.K >> 4;
    const int K32 = (K16 + 1) >> 1;
    const int NG = K16 >> grp_shift;
    const bool on = ot < (a.N >> 4);
    const uint8_t* wtile = on ? (a.w + static_cast<size_t>(ot) * NG * GR) : nullptr;

    auto loadX32 = [&](int slot, int k32) {
        // Total uint4 transfers = BM_ROWS * 4 (256 for MT=8, 128 for MT=4): 1 load per thread, 0 idle at MT=8.
        if (tid < BM_ROWS * 4)
        {
            const int half_k = tid / (BM_ROWS * 2);
            const int rem = tid - half_k * (BM_ROWS * 2);
            const int row = rem >> 1, h8 = rem & 1;
            uint4* dst = reinterpret_cast<uint4*>(sm[slot][half_k]) + row * X_LD_U4 + h8;
            const int grow = m0 + row;
            const int ks = k32 * 2 + half_k;
            if (grow < a.M && ks < K16)
                cp16(dst, a.x + static_cast<size_t>(grow) * a.ldx + ks * 16 + h8 * 8);
            else
                *dst = make_uint4(0u, 0u, 0u, 0u);
        }
    };

    // Prologue: 2 BK=32 stages in flight + prefetch first pair of W tiles and group scale(s)
    #pragma unroll
    for (int p = 0; p < STAGES - 1; ++p)
    {
        if (p < K32) loadX32(p, p);
        cp_commit();
    }

    u32 nw0[2] = {0u, 0u}, nw1[2] = {0u, 0u};
    u32 nsc0 = 0u, nsc1 = 0u;
    if (on && K16 > 0)
    {
        C::ld_tile(wtile, lane, nw0);
        nsc0 = ld_scale_i8(wtile, lane, grp);
        if (K16 > 1)
        {
            C::ld_tile(tile_addr(wtile, 1, grp_shift, grp_mask, GR), lane, nw1);
            if (grp == 1) nsc1 = ld_scale_i8(grp_addr(wtile, 1, grp_shift, GR), lane, grp);
        }
    }

    float acc[MT][4];
    #pragma unroll
    for (int t = 0; t < MT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;

    const int r = lane >> 2, c = lane & 3;
    const int x_lane_off = r * X_LD_U32 + c;

    int stage = 0;
    for (int k32 = 0; k32 < K32; ++k32)
    {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int nk32 = k32 + STAGES - 1;
        if (nk32 < K32)
        {
            int nslot = stage + STAGES - 1;
            if (nslot >= STAGES) nslot -= STAGES;
            loadX32(nslot, nk32);
        }
        cp_commit();

        const int ks0 = k32 * 2;
        const int ks1 = ks0 + 1;

        if (on)
        {
            const u32 cw0[2] = {nw0[0], nw0[1]};
            const u32 cw1[2] = {nw1[0], nw1[1]};
            const u32 csc0 = nsc0;
            const u32 csc1 = (grp == 1) ? nsc1 : csc0;

            // Prefetch next BK=32 pair of W tiles and group scale(s)
            if (ks0 + 2 < K16)
            {
                C::ld_tile(tile_addr(wtile, ks0 + 2, grp_shift, grp_mask, GR), lane, nw0);
                if (grp == 1 || ((ks0 + 2) & grp_mask) == 0)
                    nsc0 = ld_scale_i8(grp_addr(wtile, ks0 + 2, grp_shift, GR), lane, grp);
            }
            if (ks1 + 2 < K16)
            {
                C::ld_tile(tile_addr(wtile, ks1 + 2, grp_shift, grp_mask, GR), lane, nw1);
                if (grp == 1)
                    nsc1 = ld_scale_i8(grp_addr(wtile, ks1 + 2, grp_shift, GR), lane, grp);
            }

            // Slab 0: ks0 (k0..15)
            {
                const u32 s0 = __byte_perm(csc0, csc0, 0x1010);
                const u32 s1 = __byte_perm(csc0, csc0, 0x3232);
                u32 af[4];
                C::dec(cw0, af);
                af[0] = hmul2_u32(af[0], s0);
                af[1] = hmul2_u32(af[1], s1);
                af[2] = hmul2_u32(af[2], s0);
                af[3] = hmul2_u32(af[3], s1);
                const u32* sx = sm[stage][0] + x_lane_off;
                #pragma unroll
                for (int t = 0; t < MT; ++t)
                {
                    const u32 b0 = sx[t * (8 * X_LD_U32)];
                    const u32 b1 = sx[t * (8 * X_LD_U32) + 4];
                    mma16816(acc[t], af[0], af[1], af[2], af[3], b0, b1);
                }
            }

            // Slab 1: ks1 (k16..31)
            if (ks1 < K16)
            {
                const u32 s0 = __byte_perm(csc1, csc1, 0x1010);
                const u32 s1 = __byte_perm(csc1, csc1, 0x3232);
                u32 af[4];
                C::dec(cw1, af);
                af[0] = hmul2_u32(af[0], s0);
                af[1] = hmul2_u32(af[1], s1);
                af[2] = hmul2_u32(af[2], s0);
                af[3] = hmul2_u32(af[3], s1);
                const u32* sx = sm[stage][1] + x_lane_off;
                #pragma unroll
                for (int t = 0; t < MT; ++t)
                {
                    const u32 b0 = sx[t * (8 * X_LD_U32)];
                    const u32 b1 = sx[t * (8 * X_LD_U32) + 4];
                    mma16816(acc[t], af[0], af[1], af[2], af[3], b0, b1);
                }
            }
        }
        if (++stage == STAGES) stage = 0;
    }
    cp_wait<0>();
    if (!on) return;

    const int col = ot * 16 + r;
    #pragma unroll
    for (int t = 0; t < MT; ++t)
    {
        const int row0 = m0 + 8 * t + 2 * c;
        const int row1 = row0 + 1;
        if constexpr (RS)
        {
            if (row0 < a.M)
            {
                uint16_t* dst = rs_row(rs, row0, a.ldy);
                dst[col] = f2bf(acc[t][0]);
                dst[col + 8] = f2bf(acc[t][2]);
            }
            if (row1 < a.M)
            {
                uint16_t* dst = rs_row(rs, row1, a.ldy);
                dst[col] = f2bf(acc[t][1]);
                dst[col + 8] = f2bf(acc[t][3]);
            }
        }
        else if constexpr (F32)
        {
            float* yp = static_cast<float*>(a.y);
            if (row0 < a.M) { yp[static_cast<size_t>(row0) * a.ldy + col] = acc[t][0]; yp[static_cast<size_t>(row0) * a.ldy + col + 8] = acc[t][2]; }
            if (row1 < a.M) { yp[static_cast<size_t>(row1) * a.ldy + col] = acc[t][1]; yp[static_cast<size_t>(row1) * a.ldy + col + 8] = acc[t][3]; }
        }
        else
        {
            const int odt = a.bf16 ? gv::O_BF16 : gv::O_F16;
            if (row0 < a.M) { gv::put(a.y, static_cast<size_t>(row0) * a.ldy + col, acc[t][0], odt); gv::put(a.y, static_cast<size_t>(row0) * a.ldy + col + 8, acc[t][2], odt); }
            if (row1 < a.M) { gv::put(a.y, static_cast<size_t>(row1) * a.ldy + col, acc[t][1], odt); gv::put(a.y, static_cast<size_t>(row1) * a.ldy + col + 8, acc[t][3], odt); }
        }
    }
}

inline void launch(Args a, cudaStream_t st, const Rs& rs = Rs())
{
    if (!a.w || a.M <= 0 || a.N <= 0 || a.K <= 0) return;
    if (a.grp != 1 && a.grp != 2 && a.grp != 4 && a.grp != 8) return;
    const bool big = a.M >= 64;
    const int bm = big ? 64 : 32;
    const int nb = (a.N + 127) / 128 + (a.w2 ? (a.N2 + 127) / 128 : 0);
    const int mb = (a.M + bm - 1) / bm;
    if (a.rgrp == 0 && mb >= 4) a.rgrp = 4;
    const dim3 grid(static_cast<unsigned>(nb), static_cast<unsigned>(mb));
    if (rs.base)
    {
        if (big) k_gemm_i8g<8, true, false><<<grid, 256, 0, st>>>(a, rs);
        else     k_gemm_i8g<4, true, false><<<grid, 256, 0, st>>>(a, rs);
    }
    else if (a.bf16 == 2)
    {
        if (big) k_gemm_i8g<8, false, true><<<grid, 256, 0, st>>>(a);
        else     k_gemm_i8g<4, false, true><<<grid, 256, 0, st>>>(a);
    }
    else
    {
        if (big) k_gemm_i8g<8, false, false><<<grid, 256, 0, st>>>(a);
        else     k_gemm_i8g<4, false, false><<<grid, 256, 0, st>>>(a);
    }
}

}   // namespace i8gemm
