// K1 entry kernels: 3 launches per layer per step (fc1, fc2, finalize); shared expert fused into the
// fc1/fc2 grids as extra CTAs; routing weights applied; fixed-order (deterministic) final sum.
#pragma once
#include "tr3.cuh"
#include "rowtile.cuh"
#include "densev.cuh"

// shared expert / dense MLP: int4 g32 records (one interleaved fc1 stream), or MXFP8 W8A16 records when A.s_mx
// (the MTP draft layer: gate and up are two streams, s_fc1 and s_up). M > 8: NT = ceil(M / 8) row tiles on the weights
// of each group (x fragment tables from A.xfrag, act rows [8 NT][SACTW]).
__device__ __forceinline__ void k1_shared_fc1(const K1Args& A, int cta, float* smem)
{
    __half* act = A.act + static_cast<size_t>(A.smax) * 8 * ACTW;
    g6mx::nt_dispatch_dev(g6mx::tiles(A.M), [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        if (A.s_mx) rt_fc1_cta<Mxfp8, GvGroup<Mxfp8>, NTv>(A.s_fc1, A.s_up, Mxfp8::GR, A.xfrag, cta * A.s_cw, A.s_cw, A.s_nch, A.s_ks, 1.f, 1.f, act, SACTW, smem);
        else rt_fc1_cta<Int4, RtGroup<Int4>, NTv>(A.s_fc1, A.s_fc1 + Int4::GR, 2 * Int4::GR, A.xfrag, cta * A.s_cw, A.s_cw, A.s_nch, A.s_ks, 1.f, 1.f, act, SACTW, smem);
    });
}

// fc2 of the shared expert / dense MLP. The B tables of s_ntp row tiles sit in smem at once; more row tiles than that
// (the dense layers, I = 2048: 32 KB per tile against 96 KB) take another pass over the weight stream per group of
// s_ntp tiles. Every pass is the same arithmetic on its own columns, so a row's bits do not depend on the grouping.
__device__ __forceinline__ void k1_shared_fc2(const K1Args& A, int os, u32* smem)
{
    const int OTU = KSH / A.s_ns2;
    __half* y = A.s_direct ? K1_OUT(A) : A.ypart + static_cast<size_t>(A.ysh) * HID;
    const int bf = A.s_direct && A.out_bf16;
    const u32 mask = A.M >= 32 ? 0xffffffffu : (1u << A.M) - 1u;
    const int ntot = g6mx::tiles(A.M);
    for (int t0 = 0; t0 < ntot; t0 += A.s_ntp)
    {
        const int n = ntot - t0 < A.s_ntp ? ntot - t0 : A.s_ntp;
        if (t0) __syncthreads();                       // the previous pass is done with the tables
        rt_build_bfrag(smem, A.act + static_cast<size_t>(A.smax) * 8 * ACTW + static_cast<size_t>(8 * t0) * SACTW, SACTW, A.s_nch, n);
        __syncthreads();
        g6mx::nt_dispatch_dev(n, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            if (A.s_mx) rt_gemv_body<Mxfp8, GvGroup<Mxfp8>, NTv>(A.s_fc2, smem, A.s_nch / 2, os * OTU, (os + 1) * OTU, nullptr, 1.f, mask >> (8 * t0), y + static_cast<size_t>(8 * t0) * HID, HID, bf);
            else rt_gemv_body<Int4, RtGroup<Int4>, NTv>(A.s_fc2, smem, A.s_nch / 2, os * OTU, (os + 1) * OTU, nullptr, 1.f, mask >> (8 * t0), y + static_cast<size_t>(8 * t0) * HID, HID, bf);
        });
    }
}

// Dedicated dense int8 gG MLP kernels (decoupled from k1_tr3_* so MoE kernels pay zero PTX/register cost).
template <int NT>
__device__ __forceinline__ void rt_fc1_cta_i8(const uint8_t* gate, const uint8_t* up, int grp, const u32* xfrag, int chunk0, int cw, int nch,
                                              int ks, __half* act, int aw, float* smem)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / ks, part = warp % ks, chunk = chunk0 + cl;
    const bool on = cl < cw && chunk < nch;
    const int NG = KSH / grp, NGP = NG / ks, g0 = part * NGP;
    const int GR = grp * 256 + 32;
    float ag[NT][4], au[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.f;
    if (on)
    {
        const size_t t_off = static_cast<size_t>(chunk) * NG * GR;
        gv_tile_part_i8<NT>(gate + t_off, xfrag + lane * 2, KSH * 64, g0, g0 + NGP, grp, ag);
        gv_tile_part_i8<NT>(up + t_off, xfrag + lane * 2, KSH * 64, g0, g0 + NGP, grp, au);
    }
    if (ks > 1)
    {
        float4* ps = reinterpret_cast<float4*>(smem) + (warp * 32 + lane) * 2 * NT;
        if (on)
        {
            #pragma unroll
            for (int t = 0; t < NT; ++t)
            {
                ps[2 * t] = make_float4(ag[t][0], ag[t][1], ag[t][2], ag[t][3]);
                ps[2 * t + 1] = make_float4(au[t][0], au[t][1], au[t][2], au[t][3]);
            }
        }
        __syncthreads();
        if (on && part == 0)
            for (int q = 1; q < ks; ++q)
            {
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const float4 g = ps[q * 64 * NT + 2 * t], u = ps[q * 64 * NT + 2 * t + 1];
                    ag[t][0] += g.x; ag[t][1] += g.y; ag[t][2] += g.z; ag[t][3] += g.w;
                    au[t][0] += u.x; au[t][1] += u.y; au[t][2] += u.z; au[t][3] += u.w;
                }
            }
    }
    if (!on || part) return;
    const int r = lane >> 2, c = lane & 3;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        __half* a0 = act + (8 * t + 2 * c) * aw + chunk * 16 + r;
        a0[0] = __float2half_rn(swiglu_f(ag[t][0], au[t][0]));
        a0[aw] = __float2half_rn(swiglu_f(ag[t][1], au[t][1]));
        a0[8] = __float2half_rn(swiglu_f(ag[t][2], au[t][2]));
        a0[aw + 8] = __float2half_rn(swiglu_f(ag[t][3], au[t][3]));
    }
}

template <int NT>
__device__ __forceinline__ void rt_gemv_body_i8(const uint8_t* stream, const u32* bfrag, int NG2, int grp, int ot0, int ot_end,
                                                u32 mask, __half* y, int ystride, int bf16)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int GR = grp * 256 + 32;
    const size_t tile_bytes = static_cast<size_t>(NG2) * GR;
    const u32* bf = bfrag + lane * 2;
    const int ts = NG2 * grp * 64;
    const int r = lane >> 2, c = lane & 3;
    for (int ot = ot0 + warp; ot < ot_end; ot += nwarps)
    {
        float acc[NT][4];
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        gv_tile_part_i8<NT>(stream + static_cast<size_t>(ot) * tile_bytes, bf, ts, 0, NG2, grp, acc);
        const int o = ot * 16 + r;
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            const int m0 = 8 * t + 2 * c;
            if ((mask >> m0) & 1)
            {
                if (bf16) { reinterpret_cast<uint16_t*>(y)[m0 * ystride + o] = f2bf(acc[t][0]); reinterpret_cast<uint16_t*>(y)[m0 * ystride + o + 8] = f2bf(acc[t][2]); }
                else { y[m0 * ystride + o] = __float2half_rn(acc[t][0]); y[m0 * ystride + o + 8] = __float2half_rn(acc[t][2]); }
            }
            if ((mask >> (m0 + 1)) & 1)
            {
                if (bf16) { reinterpret_cast<uint16_t*>(y)[(m0 + 1) * ystride + o] = f2bf(acc[t][1]); reinterpret_cast<uint16_t*>(y)[(m0 + 1) * ystride + o + 8] = f2bf(acc[t][3]); }
                else { y[(m0 + 1) * ystride + o] = __float2half_rn(acc[t][1]); y[(m0 + 1) * ystride + o + 8] = __float2half_rn(acc[t][3]); }
            }
        }
    }
}

__global__ __launch_bounds__(1024) void k1_dense_i8_fc1(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    __half* act = A.act + static_cast<size_t>(A.smax) * 8 * ACTW;
    g6mx::nt_dispatch_dev(g6mx::tiles(A.M), [&](auto N) {
        constexpr int NTv = decltype(N)::value;
        rt_fc1_cta_i8<NTv>(A.s_fc1, A.s_up, A.s_i8_fc1, A.xfrag, static_cast<int>(blockIdx.x) * A.s_cw, A.s_cw, A.s_nch, A.s_ks, act, SACTW, reinterpret_cast<float*>(smem));
    });
}

__global__ __launch_bounds__(512, 2) void k1_dense_i8_fc2(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int os = static_cast<int>(blockIdx.x);
    const int OTU = KSH / A.s_ns2;
    __half* y = A.s_direct ? K1_OUT(A) : A.ypart + static_cast<size_t>(A.ysh) * HID;
    const int bf = A.s_direct && A.out_bf16;
    const u32 mask = A.M >= 32 ? 0xffffffffu : (1u << A.M) - 1u;
    const int ntot = g6mx::tiles(A.M);
    for (int t0 = 0; t0 < ntot; t0 += A.s_ntp)
    {
        const int n = ntot - t0 < A.s_ntp ? ntot - t0 : A.s_ntp;
        if (t0) __syncthreads();
        rt_build_bfrag(smem, A.act + static_cast<size_t>(A.smax) * 8 * ACTW + static_cast<size_t>(8 * t0) * SACTW, SACTW, A.s_nch, n);
        __syncthreads();
        g6mx::nt_dispatch_dev(n, [&](auto N) {
            constexpr int NTv = decltype(N)::value;
            rt_gemv_body_i8<NTv>(A.s_fc2, smem, A.s_nch / A.s_i8_fc2, A.s_i8_fc2, os * OTU, (os + 1) * OTU, mask >> (8 * t0), y + static_cast<size_t>(8 * t0) * HID, HID, bf);
        });
    }
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

// ------------------------------------------------------------------ TR3 build
// one fc1 unit of the slot's expert e at its width (tr3_unit). K2 = the layer has 2-bit units (bind::scan_widths):
// only then is the K2 body compiled in. The K2 body makes K3 units about 3.5% slower at M = 5 (GB10). Layers
// without 2-bit units therefore take the K3 / K4 kernels. CB = the layer's codebook (tr3.cuh TR3_*).
template <bool E4, bool K2, bool BS = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc1_any(const K1Args& A, int plan, int unit, int e, int mask, u32* smem)
{
    const long long* tr = A.t_tab + 6 * e;
    const int s_ks1 = __ffs(A.KS1) - 1;
    const int u_ks1 = unit >> s_ks1, mat = u_ks1 & 1, u_b = u_ks1 >> 1;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const uint8_t* mw;
    const int k = tr3_fc1_unit(A.t_fc1, A.t_ptab, A.t_kmix, e, tr, b * 2 + mat, mw);
    if ((!BS || K2) && k == 3) tr3_fc1_body<3, E4, BS, CB>(A, plan, unit, tr, mw, mask, smem);
    else if (!K2 || k == 4) tr3_fc1_body<4, E4, BS, CB>(A, plan, unit, tr, mw, mask, smem);
    else tr3_fc1_body<K2 ? 2 : 4, E4, BS, CB>(A, plan, unit, tr, mw, mask, smem);
}

// one fc2 unit. Down has one width per expert (tr[4]).
template <bool E4, bool K2, bool BS = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc2_any(const K1Args& A, int plan, int unit, const long long* tr, int mask, u32* smem)
{
    if ((!BS || K2) && tr[4] == 3) tr3_fc2_body<3, E4, CB, BS>(A, plan, unit, tr, mask, smem);
    else if (!K2 || tr[4] == 4) tr3_fc2_body<4, E4, CB, BS>(A, plan, unit, tr, mask, smem);
    else tr3_fc2_body<K2 ? 2 : 4, E4, CB, BS>(A, plan, unit, tr, mask, smem);
}

// Multi-tile (nt in 2..4) dispatchers for fc1 and fc2 (M > 8 only; weights streamed strictly once across nt tiles).
template <bool E4, bool K2, bool BS = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc1_any_nt(const K1Args& A, int plan, int ss0, int b, int mat, int h, int e,
                                               int nt, const int (&masks)[4], u32* smem)
{
    const long long* tr = A.t_tab + 6 * e;
    if (b >= static_cast<int>(tr[3])) return;
    const uint8_t* mw;
    const int k = tr3_fc1_unit(A.t_fc1, A.t_ptab, A.t_kmix, e, tr, b * 2 + mat, mw);
    const int4 m4 = make_int4(masks[0], masks[1], masks[2], masks[3]);
    if ((!BS || K2) && k == 3) tr3_fc1_body_nt<3, E4, BS, CB>(A, plan, ss0, b, mat, h, tr, mw, nt, m4, smem);
    else if (!K2 || k == 4) tr3_fc1_body_nt<4, E4, BS, CB>(A, plan, ss0, b, mat, h, tr, mw, nt, m4, smem);
    else tr3_fc1_body_nt<K2 ? 2 : 4, E4, BS, CB>(A, plan, ss0, b, mat, h, tr, mw, nt, m4, smem);
}

template <bool E4, bool K2, bool BS = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc2_any_nt(const K1Args& A, int ss0, int b, int ns,
                                               const long long* tr, int nt, const int (&masks)[4], u32* smem)
{
    const int4 m4 = make_int4(masks[0], masks[1], masks[2], masks[3]);
    if ((!BS || K2) && tr[4] == 3) tr3_fc2_body_nt<3, E4, CB, BS>(A, ss0, b, ns, tr, nt, m4, smem);
    else if (!K2 || tr[4] == 4) tr3_fc2_body_nt<4, E4, CB, BS>(A, ss0, b, ns, tr, nt, m4, smem);
    else tr3_fc2_body_nt<K2 ? 2 : 4, E4, CB, BS>(A, ss0, b, ns, tr, nt, m4, smem);
}

// Trellis shared-expert slots of a step (SH): one per 8 rows (an n=8 tile each), so 1 at M <= 8; 0 without SH.
template <bool SH>
__device__ __forceinline__ int k1_nsh(const K1Args& A)
{
    if constexpr (!SH) return 0;
    else if constexpr (g6mx::NT == 1) return 1;
    else return g6mx::tiles(A.M);
}

// Work item slot -> expert, token mask and scratch slot. false = no work. Without SH the scratch slot is the item slot.
// SH: item slots 0 .. nsh-1 are the trellis shared expert, tile t with rows 8 t .. 8 t + 7 (every row at M <= 8), and
// its scratch slot is n_slots + t (the finalize reads it there). Item slot i >= nsh is routed slot i - nsh. Every step
// has the shared expert, so it is handed out first.
template <bool SH>
__device__ __forceinline__ bool k1_slot(const K1Args& A, int plan, int slot, int& e, int& mask, int& ss)
{
    const int nsh = k1_nsh<SH>(A);
    if (SH && slot < nsh)
    {
        const int r = A.M - 8 * slot;
        e = A.sh_e; mask = r >= 8 ? 0xff : (1 << r) - 1; ss = A.n_slots + slot;
        return true;
    }
    ss = slot - nsh;
    if (ss < A.n_slots_dev[plan])
    {
        e = A.slot_expert[plan * SMAX + ss];
        mask = A.slot_mask[plan * SMAX + ss];
        return true;
    }
    return false;
}

// Leader-dedup multi-tile slot resolver: groups up to 4 consecutive 8-token slots for the same expert
// (including the trellis shared expert at M > 8) onto the first slot of each run so weights are read once.
// Returns nt in 1..4 for a leader slot, or 0 for an inactive / non-leader slot.
template <bool SH>
__device__ __forceinline__ int k1_slot_nt(const K1Args& A, int plan, int slot, int& e, int (&masks)[4], int& ss0)
{
    if constexpr (g6mx::NT == 1)
    {
        int m = 0;
        if (!k1_slot<SH>(A, plan, slot, e, m, ss0)) return 0;
        masks[0] = m; masks[1] = masks[2] = masks[3] = 0;
        return 1;
    }
    const int nsh = k1_nsh<SH>(A);
    if (SH && slot < nsh)
    {
        if (slot & 3) return 0;
        const int nt = (nsh - slot < 4) ? (nsh - slot) : 4;
        e = A.sh_e;
        ss0 = A.n_slots + slot;
        #pragma unroll
        for (int ti = 0; ti < 4; ++ti)
        {
            if (ti < nt)
            {
                const int r = A.M - 8 * (slot + ti);
                masks[ti] = r >= 8 ? 0xff : (1 << r) - 1;
            }
            else masks[ti] = 0;
        }
        return nt;
    }
    const int ss = slot - nsh;
    const int ns = A.n_slots_dev[plan];
    if (ss >= ns) return 0;
    const int* se = A.slot_expert + plan * SMAX;
    e = se[ss];
    if (A.M <= 8)
    {
        ss0 = ss;
        masks[0] = A.slot_mask[plan * SMAX + ss];
        masks[1] = masks[2] = masks[3] = 0;
        return 1;
    }
    int back = 0;
    while (ss - back > 0 && se[ss - back - 1] == e) ++back;
    if (back & 3) return 0;
    int nt = 1;
    while (nt < 4 && ss + nt < ns && se[ss + nt] == e) ++nt;
    ss0 = ss;
    #pragma unroll
    for (int ti = 0; ti < 4; ++ti)
        masks[ti] = (ti < nt) ? A.slot_mask[plan * SMAX + ss + ti] : 0;
    return nt;
}

// GR8_K1_TLBW (A/B, bind.cuh make_k1): warp 0 of fc1 CTA s, s < the step's item slots, prefetches one L2 line per
// 2^tlbw-byte granule of item slot s's fc1 and fc2 ranges at the start of the launch. The GPU's address translations
// of every routed slot (fc1 units of the later CTA waves, all fc2 units) are then walked while fc1 runs, not when
// those CTAs start. prefetch.global.L2 has no result: no output changes.
template <bool SH>
__device__ __forceinline__ void k1_tlbw(const K1Args& A, int plan, int s)
{
    int e, mask, ss;
    if (threadIdx.x >= 32 || !k1_slot<SH>(A, plan, s, e, mask, ss)) return;
    const long long* tr = A.t_tab + 6 * e;
    const int nb = static_cast<int>(tr[3]);
    if (nb <= 0) return;                                           // an expert without blocks on this rank: no ranges
    constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;      // bytes of one fc1 matrix or fc2 block per bit of width
    const uintptr_t g = uintptr_t(1) << A.tlbw;
    auto warm = [&](const uint8_t* base, size_t lo, size_t hi) {
        if (hi <= lo) return;
        const uintptr_t a0 = reinterpret_cast<uintptr_t>(base + lo), a1 = reinterpret_cast<uintptr_t>(base + hi);
        for (uintptr_t a = (a0 & ~(g - 1)) + threadIdx.x * g; a < a1; a += 32 * g)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(a < a0 ? a0 : a) : "memory");
    };
    if (A.t_ptab)                                                  // the piece table: each piece's two slabs
    {
        const long long km = A.t_kmix ? A.t_kmix[e] : 0;
        for (int b = 0; b < nb; ++b)
        {
            warm(A.t_ptab[tr3_ptab_slot(e, b)], 0, tr3_piece_fc1_bytes(km, tr, b));
            warm(A.t_ptab[tr3_ptab_slot(e, b) + 1], 0, tr3_piece_fc2_bytes(tr));
        }
        return;
    }
    size_t f1lo, f1hi;
    tr3_unit(A.t_kmix, e, tr, 0, f1lo);                          // units are back to back in unit order
    const int kl = tr3_unit(A.t_kmix, e, tr, 2 * nb - 1, f1hi);
    f1hi += kl * MB;
    const size_t f2lo = static_cast<size_t>(tr[1]), f2hi = f2lo + static_cast<size_t>(nb) * static_cast<size_t>(tr[4]) * MB;
    warm(A.t_fc1, f1lo, f1hi);
    warm(A.t_fc2, f2lo, f2hi);
}

// E4 = the layer's routed experts are exlr8-e4m3 (codebook rounded to E4M3, tr3.cuh tr3_cb2); K2 = it has 2-bit units;
// SH = its shared expert is trellis row 256, run as one more slot, and its hidden-side vectors are in block-scale form
// (tr3.sg, tr3.v1b, tr3.v3b; exlr8 v2); CB = the codebook (tr3.cuh TR3_*, other than mcg on exlr8-e4m3 layers only).
// The binder picks the instantiation per layer (bind::make_k1); E4 = K2 = SH = false, CB = mcg is the TR3 kernel.
template <bool E4 = false, bool K2 = false, bool SH = false, int CB = TR3_MCG>
__global__ __launch_bounds__(1024) void k1_tr3_fc1(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[0];
    const int ups = g6nr::EB * 2 * A.KS1, nr = (A.n_slots + k1_nsh<SH>(A)) * ups;
    if (A.tlbw && static_cast<int>(blockIdx.x) < A.n_slots + k1_nsh<SH>(A)) k1_tlbw<SH>(A, plan, blockIdx.x);
    if (static_cast<int>(blockIdx.x) < nr)
    {
        int e, m, ss;
        const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (static_cast<int>(blockIdx.x) >> (__ffs(ups) - 1)) : (static_cast<int>(blockIdx.x) / ups);
        if (k1_slot<SH>(A, plan, slot, e, m, ss))
            tr3_fc1_any<E4, K2, SH, CB>(A, plan, blockIdx.x + (ss - slot) * ups, e, m, smem);
    }
    else if (!SH) k1_shared_fc1(A, blockIdx.x - nr, reinterpret_cast<float*>(smem));
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

template <bool E4 = false, bool K2 = false, bool SH = false, int CB = TR3_MCG>
__global__ __launch_bounds__(1024) void k1_tr3_fc1_nt(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[0];
    const int ups = g6nr::EB * 2 * A.KS1, nr = (A.n_slots + k1_nsh<SH>(A)) * ups;
    if (A.tlbw && static_cast<int>(blockIdx.x) < A.n_slots + k1_nsh<SH>(A)) k1_tlbw<SH>(A, plan, blockIdx.x);
    if (static_cast<int>(blockIdx.x) < nr)
    {
        int e, masks[4], ss;
        const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (static_cast<int>(blockIdx.x) >> (__ffs(ups) - 1)) : (static_cast<int>(blockIdx.x) / ups);
        const int nt = k1_slot_nt<SH>(A, plan, slot, e, masks, ss);
        if (nt == 1) tr3_fc1_any<E4, K2, SH, CB>(A, plan, blockIdx.x + (ss - slot) * ups, e, masks[0], smem);
        else if (nt > 1)
        {
            const int s_ks1 = __ffs(A.KS1) - 1;
            const int h = static_cast<int>(blockIdx.x) & (A.KS1 - 1);
            const int u_ks1 = static_cast<int>(blockIdx.x) >> s_ks1, mat = u_ks1 & 1, u_b = u_ks1 >> 1;
            const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
            tr3_fc1_any_nt<E4, K2, SH, CB>(A, plan, ss, b, mat, h, e, nt, masks, smem);
        }
    }
    else if (!SH) k1_shared_fc1(A, blockIdx.x - nr, reinterpret_cast<float*>(smem));
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

template <bool E4 = false, bool K2 = false, bool SH = false, int CB = TR3_MCG>
__global__ __launch_bounds__(512, 2) void k1_tr3_fc2(K1Args A)   // 68 -> 64 regs: 2 CTAs per SM (item 1h L1)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[1];
    const int ups = g6nr::EB * A.NS2, nr = (A.n_slots + k1_nsh<SH>(A)) * ups;
    if (static_cast<int>(blockIdx.x) < nr)
    {
        int e, m, ss;
        const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (static_cast<int>(blockIdx.x) >> (__ffs(ups) - 1)) : (static_cast<int>(blockIdx.x) / ups);
        if (k1_slot<SH>(A, plan, slot, e, m, ss))
            tr3_fc2_any<E4, K2, SH, CB>(A, plan, blockIdx.x + (ss - slot) * ups, A.t_tab + 6 * e, m, smem);
    }
    else if (!SH) k1_shared_fc2(A, blockIdx.x - nr, smem);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[2] = plan;
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

template <bool E4 = false, bool K2 = false, bool SH = false, int CB = TR3_MCG>
__global__ __launch_bounds__(512, 2) void k1_tr3_fc2_nt(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[1];
    const int ups = g6nr::EB * A.NS2, nr = (A.n_slots + k1_nsh<SH>(A)) * ups;
    if (static_cast<int>(blockIdx.x) < nr)
    {
        int e, masks[4], ss;
        const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (static_cast<int>(blockIdx.x) >> (__ffs(ups) - 1)) : (static_cast<int>(blockIdx.x) / ups);
        const int nt = k1_slot_nt<SH>(A, plan, slot, e, masks, ss);
        if (nt == 1) tr3_fc2_any<E4, K2, SH, CB>(A, plan, blockIdx.x + (ss - slot) * ups, A.t_tab + 6 * e, masks[0], smem);
        else if (nt > 1)
        {
            const int s_ns2 = __ffs(A.NS2) - 1;
            const int ns = static_cast<int>(blockIdx.x) & (A.NS2 - 1), u_b = static_cast<int>(blockIdx.x) >> s_ns2;
            const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
            tr3_fc2_any_nt<E4, K2, SH, CB>(A, ss, b, ns, A.t_tab + 6 * e, nt, masks, smem);
        }
    }
    else if (!SH) k1_shared_fc2(A, blockIdx.x - nr, smem);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[2] = plan;
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

template <bool SH = false>
__global__ __launch_bounds__(SH ? 288 : 256, 3) void k1_tr3_fin(K1Args A)
{
    const int plan = A.ctl[2];
    [[maybe_unused]] const bool corrected = tr3_fin_body<SH>(A, plan);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[0] = (plan + 1) % A.n_plans;
#if DEC_SLOT >= 2
    if (A.slot.send) dslot::publish(A.slot, threadIdx.x < 32 || corrected);
#endif
}

// ------------------------------------------------------------------ GR8_ROUTE_OVL: the trellis shared expert apart
// SH layers, separate fc1 / fc2 launches (bind.cuh k1_ovl_fork / launch_k1_ovl). The shared expert's units need nothing
// from the route plan (k1_slot: item slots 0 .. nsh-1, scratch slots n_slots + t; tr3_fc1_body reads the plan only for
// routed slots, tr3_fc2_body not at all), so they can run on a forked stream while the router GEMV and the plan kernel
// run. PART 1 = the shared expert's CTAs of k1_tr3_fc1 / k1_tr3_fc2 (the first nsh * ups of their grids), PART 2 = the
// routed ones (the rest: blockIdx.x is offset by nsh * ups). Same units, same bodies, same scratch, so the same bytes.
// PART 2 keeps the ctl bookkeeping (ctl[1], ctl[2]); PART 1 writes no ctl word and reads only ctl[0], which nothing
// writes between the fork and the join (the finalize, after the join, advances it).
template <bool E4, bool K2, int PART, int CB = TR3_MCG>
__global__ __launch_bounds__(1024) void k1_tr3_fc1_ovl(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[0];
    const int ups = A.B * 2 * A.KS1;
    const int cta = static_cast<int>(blockIdx.x) + (PART == 2 ? k1_nsh<true>(A) * ups : 0);
    if (PART == 2 && A.tlbw && static_cast<int>(blockIdx.x) < A.n_slots) k1_tlbw<true>(A, plan, blockIdx.x + k1_nsh<true>(A));
    int e, m, ss;
    const int slot = cta / ups;
    if (k1_slot<true>(A, plan, slot, e, m, ss))
        tr3_fc1_any<E4, K2, true, CB>(A, plan, cta + (ss - slot) * ups, e, m, smem);
    if (PART == 2 && blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

template <bool E4, bool K2, int PART, int CB = TR3_MCG, int MAXT = 1024, int MINB = 1>
__global__ __launch_bounds__(MAXT, MINB) void k1_tr3_fc1_ovl_nt(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[0];
    const int ups = A.B * 2 * A.KS1;
    const int cta = static_cast<int>(blockIdx.x) + (PART == 2 ? k1_nsh<true>(A) * ups : 0);
    if (PART == 2 && A.tlbw && static_cast<int>(blockIdx.x) < A.n_slots) k1_tlbw<true>(A, plan, blockIdx.x + k1_nsh<true>(A));
    int e, masks[4], ss;
    const int slot = cta / ups;
    const int nt = k1_slot_nt<true>(A, plan, slot, e, masks, ss);
    if (nt == 1) tr3_fc1_any<E4, K2, true, CB>(A, plan, cta + (ss - slot) * ups, e, masks[0], smem);
    else if (nt > 1)
    {
        const int s_ks1 = __ffs(A.KS1) - 1;
        const int h = cta & (A.KS1 - 1), u_ks1 = cta >> s_ks1, mat = u_ks1 & 1, u_b = u_ks1 >> 1;
        const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
        tr3_fc1_any_nt<E4, K2, true, CB>(A, plan, ss, b, mat, h, e, nt, masks, smem);
    }
    if (PART == 2 && blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

template <bool E4, bool K2, int PART, int CB = TR3_MCG>
__global__ __launch_bounds__(512, 2) void k1_tr3_fc2_ovl(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = PART == 2 ? A.ctl[1] : A.ctl[0];
    const int ups = A.B * A.NS2;
    const int cta = static_cast<int>(blockIdx.x) + (PART == 2 ? k1_nsh<true>(A) * ups : 0);
    int e, m, ss;
    const int slot = cta / ups;
    if (k1_slot<true>(A, plan, slot, e, m, ss))
        tr3_fc2_any<E4, K2, true, CB>(A, plan, cta + (ss - slot) * ups, A.t_tab + 6 * e, m, smem);
    if (PART == 2 && blockIdx.x == 0 && threadIdx.x == 0) A.ctl[2] = plan;
}

template <bool E4, bool K2, int PART, int CB = TR3_MCG>
__global__ __launch_bounds__(512, 2) void k1_tr3_fc2_ovl_nt(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = PART == 2 ? A.ctl[1] : A.ctl[0];
    const int ups = A.B * A.NS2;
    const int cta = static_cast<int>(blockIdx.x) + (PART == 2 ? k1_nsh<true>(A) * ups : 0);
    int e, masks[4], ss;
    const int slot = cta / ups;
    const int nt = k1_slot_nt<true>(A, plan, slot, e, masks, ss);
    if (nt == 1) tr3_fc2_any<E4, K2, true, CB>(A, plan, cta + (ss - slot) * ups, A.t_tab + 6 * e, masks[0], smem);
    else if (nt > 1)
    {
        const int s_ns2 = __ffs(A.NS2) - 1;
        const int ns = cta & (A.NS2 - 1), u_b = cta >> s_ns2;
        const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
        tr3_fc2_any_nt<E4, K2, true, CB>(A, ss, b, ns, A.t_tab + 6 * e, nt, masks, smem);
    }
    if (PART == 2 && blockIdx.x == 0 && threadIdx.x == 0) A.ctl[2] = plan;
}

// ------------------------------------------------------------------ TR3 fused fc1 + fc2: one persistent launch
// Work items in order: routed fc1 units, shared fc1 CTAs, routed fc2 units, shared fc2 CTAs. CTAs take items from a
// counter (A.fctl[0]); an fc2 item waits until the fc1 units of its (slot, block) are done (A.fctl[1 + slot*B + b],
// the shared expert's in A.fctl[1 + SMAX*3]). Items are handed out in order, so everything an item waits for is
// already running on some CTA: no deadlock at any grid size. The last CTA out zeroes the counters for the next step.
// Same bodies, same per-unit arithmetic as the three-launch path (the bytes of gpart, z, act, out are the same);
// what changes is that fc2 units start while fc1 units of other slots still run, and the fc1 -> fc2 launch boundary
// (drain, gap, ramp) is gone. Every CTA has 512 threads: the binder maps the fc1 split (KS1, WG) to (KS1*WG/2, 2),
// which keeps every k range and its partial index p = h*WG + q (bind.cuh). Dynamic smem = max of the fc1 and fc2 needs.
__device__ __forceinline__ int ld_acquire(const int* p)
{
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

template <bool E4, bool K2, bool SH, int CB>
__device__ __forceinline__ void k1_fused_body(const K1Args& A)
{
    extern __shared__ __align__(16) u32 smem[];
    __shared__ int s_item;
    const int t = threadIdx.x;
    const int plan = A.ctl[0];
    // items for the plan's own slots only (A.n_slots is the capacity, M x 8): the slots past them were empty items
    const int nsl = A.n_slots_dev[plan] + k1_nsh<SH>(A);
    const int u1 = g6nr::EB * 2 * A.KS1, n1 = nsl * u1;                  // fc1 units per slot, routed (+ SH shared) fc1 items
    const int sh1 = SH ? 0 : (A.s_nch + A.s_cw - 1) / A.s_cw;           // shared fc1 CTAs (s_* records)
    const int u2 = g6nr::EB * A.NS2, n2 = nsl * u2;                      // fc2 units per slot, routed (+ SH shared) fc2 items
    const int total = n1 + sh1 + n2 + (SH ? 0 : A.s_ns2);
    int* done = A.fctl + 1;
    int* sdone = A.fctl + 1 + SMAX * g6nr::EB;
    if (blockIdx.x == 0 && t == 0) A.ctl[2] = plan;
    if (A.tlbw)                                                         // every item slot, also past a smaller persistent grid
        for (int s = blockIdx.x; s < nsl; s += gridDim.x) k1_tlbw<SH>(A, plan, s);
    for (;;)
    {
        if (t == 0) s_item = atomicAdd(A.fctl, 1);
        __syncthreads();
        const int item = s_item;
        if (item >= total) break;
        if (item < n1)
        {
            const int u_b = item >> __ffs(A.KS1);
            const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (item / u1);
            const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
            int e, masks[4], ss = slot;
            const int nt = k1_slot_nt<SH>(A, plan, slot, e, masks, ss);
            if (nt > 0)
            {
                if (nt == 1) tr3_fc1_any<E4, K2, SH, CB>(A, plan, item + (ss - slot) * u1, e, masks[0], smem);
                else
                {
                    const int s_ks1 = __ffs(A.KS1) - 1;
                    const int h = item & (A.KS1 - 1), mat = (item >> s_ks1) & 1;
                    tr3_fc1_any_nt<E4, K2, SH, CB>(A, plan, ss, b, mat, h, e, nt, masks, smem);
                }
                __threadfence();
                __syncthreads();
                if (t == 0) atomicAdd(done + ss * g6nr::EB + b, 1);
            }
        }
        else if constexpr (!SH)
        {
            if (item < n1 + sh1)
            {
                k1_shared_fc1(A, item - n1, reinterpret_cast<float*>(smem));
                __threadfence();
                __syncthreads();
                if (t == 0) atomicAdd(sdone, 1);
            }
            else if (item < n1 + sh1 + n2)
            {
                const int unit = item - n1 - sh1;
                const int u_b = unit >> (__ffs(A.NS2) - 1);
                const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (unit / u2);
                const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
                int e, masks[4], ss;
                const int nt = k1_slot_nt<SH>(A, plan, slot, e, masks, ss);
                if (nt > 0)
                {
                    const long long* tr = A.t_tab + 6 * e;
                    if (b < static_cast<int>(tr[3]))
                    {
                        if (t == 0) while (ld_acquire(done + ss * g6nr::EB + b) < 2 * A.KS1) {}
                        __syncthreads();
                        if (nt == 1) tr3_fc2_any<E4, K2, SH, CB>(A, plan, unit + (ss - slot) * u2, tr, masks[0], smem);
                        else tr3_fc2_any_nt<E4, K2, SH, CB>(A, ss, b, unit & (A.NS2 - 1), tr, nt, masks, smem);
                    }
                }
            }
            else
            {
                if (t == 0) while (ld_acquire(sdone) < sh1) {}
                __syncthreads();
                k1_shared_fc2(A, item - n1 - sh1 - n2, smem);
            }
        }
        else
        {
            const int unit = item - n1;
            const int u_b = unit >> (__ffs(A.NS2) - 1);
            const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (unit / u2);
            const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
            int e, masks[4], ss;
            const int nt = k1_slot_nt<SH>(A, plan, slot, e, masks, ss);
            if (nt > 0)
            {
                const long long* tr = A.t_tab + 6 * e;
                if (b < static_cast<int>(tr[3]))
                {
                    if (t == 0) while (ld_acquire(done + ss * g6nr::EB + b) < 2 * A.KS1) {}
                    __syncthreads();
                    if (nt == 1) tr3_fc2_any<E4, K2, SH, CB>(A, plan, unit + (ss - slot) * u2, tr, masks[0], smem);
                    else tr3_fc2_any_nt<E4, K2, SH, CB>(A, ss, b, unit & (A.NS2 - 1), tr, nt, masks, smem);
                }
            }
        }
        __syncthreads();                                                // smem and s_item are reused
    }
    __shared__ int s_last;
    if (t == 0)
    {
        __threadfence();
        s_last = atomicAdd(A.fctl + FCTL_N - 1, 1) == static_cast<int>(gridDim.x) - 1;
    }
    __syncthreads();
    if (s_last)                                                          // the last CTA zeroes the counters, all its threads
    {
        for (int i = t; i < FCTL_N; i += blockDim.x) A.fctl[i] = 0;
        __threadfence();
    }
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

template <bool E4 = false, bool K2 = false, bool SH = false, int CB = TR3_MCG>
__global__ __launch_bounds__(512) void k1_tr3_fused(K1Args A)
{
    k1_fused_body<E4, K2, SH, CB>(A);
}

// The same with the trellis shared expert. 128 registers (1 CTA/SM of 512 threads) avoids 64-reg load-ahead throttling.
template <bool E4 = true, bool K2 = false, int CB = TR3_MCG>
__global__ __maxnreg__(128) void k1_tr3_fused_sh(K1Args A)
{
    k1_fused_body<E4, K2, true, CB>(A);
}

// ------------------------------------------------------------------ GR8_TR3_WIDE4: persistent fused wide-K4 decode kernels
// Exact 0-ULP equivalents of k1_tr3_fused_sh for block-scale trellis-shared layers without 2-bit units (k2 == 0).
template <bool P256>
__device__ __forceinline__ uint4 k1_ld_w4(const void* p)
{
    if constexpr (P256) return tr4w_ld128_p256(p);
    else return tr4w_ld128(p);
}

template <bool XCOMP, bool P256, int CB>
__device__ __forceinline__ void tr4_fc1_body_w4(const K1Args& A, int plan, int unit, const long long* tr, const uint8_t* mw, int mask, u32* smem)
{
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = blockDim.x >> 5;
    const int s_ks1 = __ffs(A.KS1) - 1, s_wg = __ffs(A.WG) - 1;
    const int h = unit & (A.KS1 - 1);
    const int u_ks1 = unit >> s_ks1;
    const int mat = u_ks1 & 1;
    const int u_b = u_ks1 >> 1;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (u_b / g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int KSU = KSH >> s_ks1;
    const int ks0 = h * KSU;
    const int q = warp >> 3, j = warp & 7;
    const int steps = KSU >> s_wg;
    const int ksq = q * steps;
    const int g_start = (ks0 + ksq) >> 3;
    const int ng = steps >> 3;
    const uint8_t* g0 = mw + static_cast<size_t>(g_start) * TR4W_GROUP + (j * 2) * 512 + lane * 16;

    uint4 nw0 = k1_ld_w4<P256>(g0);
    uint4 nw1 = k1_ld_w4<P256>(g0 + 512);

    {
        constexpr int PU = 2;
        const int nact = __popc(mask & 0xff), ntask = nact * (KSU >> 3);
        const int m0 = __ffs(mask) - 1;
        for (int task = warp; task < ntask; task += PU * nwarps)
        {
            uint2 xr[PU], sr[PU];
            int m[PU], blk[PU];
            #pragma unroll
            for (int u = 0; u < PU; ++u)
            {
                if (task + u * nwarps < ntask)
                {
                    const int tk = task + u * nwarps;
                    if (nact == 1) { m[u] = m0; blk[u] = tk; }
                    else { m[u] = __fns(mask, 0, tk % nact + 1); blk[u] = tk / nact; }
                    const int tok = A.M <= 8 ? m[u] : (slot < A.n_slots ? A.slot_tok[(plan * SMAX + slot) * 8 + m[u]] : (slot - A.n_slots) * 8 + m[u]);
                    const int k = ks0 * 16 + blk[u] * 128 + lane * 4;
                    xr[u] = *reinterpret_cast<const uint2*>(A.x + tok * HID + k);
                    sr[u] = tr3_hid4<true>(nullptr, A.t_sg, A.t_v1b, row * 2 + mat, k);
                }
            }
            #pragma unroll
            for (int u = 0; u < PU; ++u)
            {
                if (task + u * nwarps >= ntask) break;
                const float2 x01 = __half22float2(as_h2(xr[u].x)), x23 = __half22float2(as_h2(xr[u].y));
                const float2 s01 = __half22float2(as_h2(sr[u].x)), s23 = __half22float2(as_h2(sr[u].y));
                float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
                had128(v, lane);
                if constexpr (XCOMP)
                {
                    u32* xp = smem + (m[u] * (KSU * 16 + 8) + blk[u] * 128 + lane * 4) / 2;
                    xp[0] = pack_h2(v[0], v[1]);
                    xp[1] = pack_h2(v[2], v[3]);
                }
                else store_frag_row(smem, blk[u] * 8, m[u], lane, v);
            }
        }
    }
    __syncthreads();

    const u32* xa = XCOMP ? (smem + min(lane >> 2, A.M - 1) * (KSU * 8 + 4) + ksq * 8 + (lane & 3))
                          : (smem + (ksq * 32 + lane) * 2);
    constexpr int xs = XCOMP ? 8 : 64;
    constexpr int xo = XCOMP ? 4 : 1;
    const int prev_lane = (lane + 31) & 31;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < ng; ++g)
    {
        const u32 w[8] = {nw0.x, nw0.y, nw0.z, nw0.w, nw1.x, nw1.y, nw1.z, nw1.w};
        if (g + 1 < ng)
        {
            const uint8_t* gp = g0 + static_cast<size_t>(g + 1) * TR4W_GROUP;
            nw0 = k1_ld_w4<P256>(gp);
            nw1 = k1_ld_w4<P256>(gp + 512);
        }
        const int s = g * 8;
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const u32 pc = __shfl_sync(0xffffffffu, w[p], prev_lane);
            u32 f0[2], f1[2];
            Tr3Lane<4, true, CB>::decode_m((static_cast<u64>(pc) << 32) | w[p], 28, f0, f1);
            mma16816(acc, f0[0], f1[0], f0[1], f1[1], xa[(s + p) * xs], xa[(s + p) * xs + xo]);
        }
    }
    const int nl = lane >> 2, c = lane & 3;
    float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX + h * A.WG + q) * 8 * 128 + j * 16 + nl;
    if ((mask >> (2 * c)) & 1) { gp[2 * c * 128] = acc[0]; gp[2 * c * 128 + 8] = acc[2]; }
    if ((mask >> (2 * c + 1)) & 1) { gp[(2 * c + 1) * 128] = acc[1]; gp[(2 * c + 1) * 128 + 8] = acc[3]; }
}

__device__ __forceinline__ void tr_fc2_prologue_256(const K1Args& A, int slot, int b, size_t row, int mask, int warp, int lane, u32* smem)
{
    const int GP = A.KS1 * A.WG;
    const __half* v2 = A.t_v2 + row * 3 * 128 + lane * 4;
    const int m = warp;
    const bool active = ((mask >> m) & 1) != 0;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (active)
    {
        auto reduce_mat = [&](int mat, float (&out)[4]) {
            const float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX * 8) * 128 + m * 128 + lane * 4;
            float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int p = 0; p < GP; p += 2)
            {
                const float4 w0 = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p) * 8 * 128));
                if (p + 1 < GP)
                {
                    const float4 w1 = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p + 1) * 8 * 128));
                    s.x += w0.x; s.y += w0.y; s.z += w0.z; s.w += w0.w;
                    s.x += w1.x; s.y += w1.y; s.z += w1.z; s.w += w1.w;
                }
                else { s.x += w0.x; s.y += w0.y; s.z += w0.z; s.w += w0.w; }
            }
            float y[4] = {rnd_h(s.x), rnd_h(s.y), rnd_h(s.z), rnd_h(s.w)};
            had128(y, lane);
            const uint2 sr = *reinterpret_cast<const uint2*>(v2 + mat * 128);
            const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
            out[0] = rnd_h(y[0] * s01.x); out[1] = rnd_h(y[1] * s01.y);
            out[2] = rnd_h(y[2] * s23.x); out[3] = rnd_h(y[3] * s23.y);
        };
        float tmp[4];
        reduce_mat(0, tmp);
        reinterpret_cast<float4*>(smem + 512)[m * 32 + lane] = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
        reduce_mat(1, tmp);
        const float4 gu0 = reinterpret_cast<const float4*>(smem + 512)[m * 32 + lane];
        const uint2 sr = *reinterpret_cast<const uint2*>(v2 + 2 * 128);
        const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
        v[0] = rnd_h(swiglu_f(gu0.x, tmp[0])) * s01.x;
        v[1] = rnd_h(swiglu_f(gu0.y, tmp[1])) * s01.y;
        v[2] = rnd_h(swiglu_f(gu0.z, tmp[2])) * s23.x;
        v[3] = rnd_h(swiglu_f(gu0.w, tmp[3])) * s23.y;
        had128(v, lane);
    }
    store_frag_row(smem, 0, m, lane, v);
    __syncthreads();
}

__device__ __forceinline__ void tr_fc2_prologue_512(const K1Args& A, int slot, int b, size_t row, int mask, int warp, int lane, u32* smem)
{
    const int GP = A.KS1 * A.WG;
    const __half* v2 = A.t_v2 + row * 3 * 128 + lane * 4;
    const int m = warp & 7, mat = warp >> 3;
    const bool active = ((mask >> m) & 1) != 0;
    float gu0[4] = {0.f, 0.f, 0.f, 0.f};
    if (warp < 16 && active)
    {
        const float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX * 8) * 128 + m * 128 + lane * 4;
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = 0; p < GP; p += 2)
        {
            const float4 w0 = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p) * 8 * 128));
            if (p + 1 < GP)
            {
                const float4 w1 = __ldcg(reinterpret_cast<const float4*>(gp + static_cast<size_t>(p + 1) * 8 * 128));
                s.x += w0.x; s.y += w0.y; s.z += w0.z; s.w += w0.w;
                s.x += w1.x; s.y += w1.y; s.z += w1.z; s.w += w1.w;
            }
            else { s.x += w0.x; s.y += w0.y; s.z += w0.z; s.w += w0.w; }
        }
        float y[4] = {rnd_h(s.x), rnd_h(s.y), rnd_h(s.z), rnd_h(s.w)};
        had128(y, lane);
        const uint2 sr = *reinterpret_cast<const uint2*>(v2 + mat * 128);
        const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
        gu0[0] = rnd_h(y[0] * s01.x); gu0[1] = rnd_h(y[1] * s01.y);
        gu0[2] = rnd_h(y[2] * s23.x); gu0[3] = rnd_h(y[3] * s23.y);
        if (mat == 1)
            reinterpret_cast<float4*>(smem + 512)[m * 32 + lane] = make_float4(gu0[0], gu0[1], gu0[2], gu0[3]);
    }
    __syncthreads();
    if (warp < 8)
    {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if (active)
        {
            const float4 gu1 = reinterpret_cast<const float4*>(smem + 512)[m * 32 + lane];
            const uint2 sr = *reinterpret_cast<const uint2*>(v2 + 2 * 128);
            const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
            v[0] = rnd_h(swiglu_f(gu0[0], gu1.x)) * s01.x;
            v[1] = rnd_h(swiglu_f(gu0[1], gu1.y)) * s01.y;
            v[2] = rnd_h(swiglu_f(gu0[2], gu1.z)) * s23.x;
            v[3] = rnd_h(swiglu_f(gu0[3], gu1.w)) * s23.y;
            had128(v, lane);
        }
        store_frag_row(smem, 0, m, lane, v);
    }
    __syncthreads();
}

template <int NWARPS, bool P256, int CB>
__device__ __forceinline__ void tr4_fc2_body_w4(const K1Args& A, int unit, const long long* tr, int mask, u32* smem)
{
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int s_ns2 = __ffs(A.NS2) - 1;
    constexpr int s_nw = (NWARPS == 8) ? 3 : 4;
    const int ns = unit & (A.NS2 - 1);
    const int u_b = unit >> s_ns2;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (u_b / g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int NTU = KSH >> s_ns2;
    const int nt0 = ns * NTU + warp;
    const int ntiles = (NTU - warp + NWARPS - 1) >> s_nw;
    const uint8_t* wg = tr3_fc2_block(A.t_fc2, A.t_ptab, static_cast<int>((tr - A.t_tab) / 6), tr, b, 4) + static_cast<size_t>(nt0) * TR4W_NT + lane * 16;

    uint4 nw0 = make_uint4(0u, 0u, 0u, 0u), nw1 = make_uint4(0u, 0u, 0u, 0u);
    if (ntiles > 0)
    {
        nw0 = k1_ld_w4<P256>(wg);
        nw1 = k1_ld_w4<P256>(wg + 512);
    }

    if constexpr (NWARPS == 8) tr_fc2_prologue_256(A, slot, b, row, mask, warp, lane, smem);
    else tr_fc2_prologue_512(A, slot, b, row, mask, warp, lane, smem);

    const u32* ha = smem + lane * 2;
    const int nl = lane >> 2, c = lane & 3;
    const bool wr0 = (mask >> (2 * c)) & 1, wr1 = (mask >> (2 * c + 1)) & 1;
    __half* zb = A.z + ((static_cast<size_t>(slot) * A.B + b) * 8 + 2 * c) * HID + nl;
    const int prev_lane = (lane + 31) & 31;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int i = 0; i < ntiles; ++i)
    {
        const u32 w[8] = {nw0.x, nw0.y, nw0.z, nw0.w, nw1.x, nw1.y, nw1.z, nw1.w};
        if (i + 1 < ntiles)
        {
            const uint8_t* np = wg + static_cast<size_t>(i + 1) * NWARPS * TR4W_NT;
            nw0 = k1_ld_w4<P256>(np);
            nw1 = k1_ld_w4<P256>(np + 512);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const u32 pc = __shfl_sync(0xffffffffu, w[p], prev_lane);
            u32 f0[2], f1[2];
            Tr3Lane<4, true, CB>::decode_m((static_cast<u64>(pc) << 32) | w[p], 28, f0, f1);
            mma16816(acc, f0[0], f1[0], f0[1], f1[1], ha[p * 64], ha[p * 64 + 1]);
        }
        __half* zp = zb + (nt0 + i * NWARPS) * 16;
        if (wr0) { zp[0] = __float2half_rn(acc[0]); zp[8] = __float2half_rn(acc[2]); }
        if (wr1) { zp[HID] = __float2half_rn(acc[1]); zp[HID + 8] = __float2half_rn(acc[3]); }
        #pragma unroll
        for (int k = 0; k < 4; ++k) acc[k] = 0.f;
    }
}

template <bool XCOMP, int CB>
__device__ __forceinline__ void tr3_fc1_body_k3w(const K1Args& A, int plan, int unit, const long long* tr, const uint8_t* mw, int mask, u32* smem)
{
    constexpr int BITS = 3, TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = blockDim.x >> 5;
    const int s_ks1 = __ffs(A.KS1) - 1, s_wg = __ffs(A.WG) - 1;
    const int h = unit & (A.KS1 - 1);
    const int u_ks1 = unit >> s_ks1, mat = u_ks1 & 1, u_b = u_ks1 >> 1;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (u_b / g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int KSU = KSH >> s_ks1, ks0 = h * KSU;
    const int q = warp >> 3, j = warp & 7;
    const int steps = KSU >> s_wg, ksq = q * steps;
    const uint8_t* base = mw + (static_cast<size_t>(ks0 + ksq) * 8 + j) * TB;
    const uint8_t* g0 = base - static_cast<size_t>(j) * TB + j * 256 + lane * 8;
    u64 nw[3];
    #pragma unroll
    for (int n = 0; n < 3; ++n) nw[n] = ld_nc64(g0 + n * 2048);

    {
        constexpr int PU = 2;
        const int nact = __popc(mask & 0xff), ntask = nact * (KSU >> 3);
        const int m0 = __ffs(mask) - 1;
        for (int task = warp; task < ntask; task += PU * nwarps)
        {
            uint2 xr[PU], sr[PU];
            int m[PU], blk[PU];
            #pragma unroll
            for (int u = 0; u < PU; ++u)
            {
                if (task + u * nwarps < ntask)
                {
                    const int tk = task + u * nwarps;
                    if (nact == 1) { m[u] = m0; blk[u] = tk; }
                    else { m[u] = __fns(mask, 0, tk % nact + 1); blk[u] = tk / nact; }
                    const int tok = A.M <= 8 ? m[u] : (slot < A.n_slots ? A.slot_tok[(plan * SMAX + slot) * 8 + m[u]] : (slot - A.n_slots) * 8 + m[u]);
                    const int k = ks0 * 16 + blk[u] * 128 + lane * 4;
                    xr[u] = *reinterpret_cast<const uint2*>(A.x + tok * HID + k);
                    sr[u] = tr3_hid4<true>(nullptr, A.t_sg, A.t_v1b, row * 2 + mat, k);
                }
            }
            #pragma unroll
            for (int u = 0; u < PU; ++u)
            {
                if (task + u * nwarps >= ntask) break;
                const float2 x01 = __half22float2(as_h2(xr[u].x)), x23 = __half22float2(as_h2(xr[u].y));
                const float2 s01 = __half22float2(as_h2(sr[u].x)), s23 = __half22float2(as_h2(sr[u].y));
                float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
                had128(v, lane);
                if constexpr (XCOMP)
                {
                    u32* xp = smem + (m[u] * (KSU * 16 + 8) + blk[u] * 128 + lane * 4) / 2;
                    xp[0] = pack_h2(v[0], v[1]);
                    xp[1] = pack_h2(v[2], v[3]);
                }
                else store_frag_row(smem, blk[u] * 8, m[u], lane, v);
            }
        }
    }
    __syncthreads();

    const u32* xa = XCOMP ? (smem + min(lane >> 2, A.M - 1) * (KSU * 8 + 4) + ksq * 8 + (lane & 3))
                          : (smem + (ksq * 32 + lane) * 2);
    constexpr int xs = XCOMP ? 8 : 64, xo = XCOMP ? 4 : 1;
    const int prev_lane = (lane + 31) & 31;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int s = 0; s < steps; s += 8)
    {
        const u64 w[3] = {nw[0], nw[1], nw[2]};
        if (s + 8 < steps)
        {
            #pragma unroll
            for (int n = 0; n < 3; ++n) nw[n] = ld_nc64(g0 + static_cast<size_t>(s + 8) * 8 * TB + n * 2048);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const u32 c = trw_chunk<3>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, c, prev_lane);
            u32 f0[2], f1[2];
            Tr3Lane<3, true, CB>::decode_m((static_cast<u64>(pc) << 24) | c, 21, f0, f1);
            mma16816(acc, f0[0], f1[0], f0[1], f1[1], xa[(s + p) * xs], xa[(s + p) * xs + xo]);
        }
    }
    const int nl = lane >> 2, c = lane & 3;
    float* gp = A.gpart + (((static_cast<size_t>(slot) * A.B + b) * 2 + mat) * GPMAX + h * A.WG + q) * 8 * 128 + j * 16 + nl;
    if ((mask >> (2 * c)) & 1) { gp[2 * c * 128] = acc[0]; gp[2 * c * 128 + 8] = acc[2]; }
    if ((mask >> (2 * c + 1)) & 1) { gp[(2 * c + 1) * 128] = acc[1]; gp[(2 * c + 1) * 128 + 8] = acc[3]; }
}

template <int NWARPS, int CB>
__device__ __forceinline__ void tr3_fc2_body_k3w(const K1Args& A, int unit, const long long* tr, int mask, u32* smem)
{
    constexpr int BITS = 3, TB = 32 * BITS;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int s_ns2 = __ffs(A.NS2) - 1;
    constexpr int s_nw = (NWARPS == 8) ? 3 : 4;
    const int ns = unit & (A.NS2 - 1), u_b = unit >> s_ns2;
    const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
    const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (u_b / g6nr::EB);
    if (b >= static_cast<int>(tr[3])) return;
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const int NTU = KSH >> s_ns2, nt0 = ns * NTU + warp;
    const int ntiles = (NTU - warp + NWARPS - 1) >> s_nw;
    const uint8_t* wg = tr3_fc2_block(A.t_fc2, A.t_ptab, static_cast<int>((tr - A.t_tab) / 6), tr, b, BITS) + static_cast<size_t>(nt0) * 8 * TB + lane * 8;
    u64 nw[3] = {0, 0, 0};
    if (ntiles > 0)
    {
        #pragma unroll
        for (int n = 0; n < 3; ++n) nw[n] = ld_nc64(wg + n * 256);
    }

    if constexpr (NWARPS == 8) tr_fc2_prologue_256(A, slot, b, row, mask, warp, lane, smem);
    else tr_fc2_prologue_512(A, slot, b, row, mask, warp, lane, smem);

    const u32* ha = smem + lane * 2;
    const int nl = lane >> 2, c = lane & 3;
    const bool wr0 = (mask >> (2 * c)) & 1, wr1 = (mask >> (2 * c + 1)) & 1;
    __half* zb = A.z + ((static_cast<size_t>(slot) * A.B + b) * 8 + 2 * c) * HID + nl;
    const int prev_lane = (lane + 31) & 31;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int i = 0; i < ntiles; ++i)
    {
        const u64 w[3] = {nw[0], nw[1], nw[2]};
        if (i + 1 < ntiles)
        {
            #pragma unroll
            for (int n = 0; n < 3; ++n) nw[n] = ld_nc64(wg + static_cast<size_t>(i + 1) * NWARPS * 8 * TB + n * 256);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            const u32 ch = trw_chunk<3>(w, p);
            const u32 pc = __shfl_sync(0xffffffffu, ch, prev_lane);
            u32 f0[2], f1[2];
            Tr3Lane<3, true, CB>::decode_m((static_cast<u64>(pc) << 24) | ch, 21, f0, f1);
            mma16816(acc, f0[0], f1[0], f0[1], f1[1], ha[p * 64], ha[p * 64 + 1]);
        }
        __half* zp = zb + (nt0 + i * NWARPS) * 16;
        if (wr0) { zp[0] = __float2half_rn(acc[0]); zp[8] = __float2half_rn(acc[2]); }
        if (wr1) { zp[HID] = __float2half_rn(acc[1]); zp[HID + 8] = __float2half_rn(acc[3]); }
        #pragma unroll
        for (int k = 0; k < 4; ++k) acc[k] = 0.f;
    }
}

template <bool HAS_K3, int NWARPS, bool XCOMP, bool P256, int CB>
__device__ __forceinline__ void k1_fused_w4_body(const K1Args& A)
{
    extern __shared__ __align__(16) u32 smem[];
    __shared__ int s_item;
    __shared__ int s_meta[4];
    const int t = threadIdx.x;
    if (t == 0)
    {
        const int plan = A.ctl[0];
        const int nsl = A.n_slots_dev[plan] + k1_nsh<true>(A);
        const int n1 = nsl * (g6nr::EB * 2 * A.KS1);
        const int n2 = nsl * (g6nr::EB * A.NS2);
        s_meta[0] = plan;
        s_meta[1] = n1;
        s_meta[2] = n1 + n2;
        if (blockIdx.x == 0) A.ctl[2] = plan;
    }
    __syncthreads();
    if (A.tlbw)
    {
        const int plan = s_meta[0], nsl = A.n_slots_dev[plan] + k1_nsh<true>(A);
        for (int s = blockIdx.x; s < nsl; s += gridDim.x) k1_tlbw<true>(A, plan, s);
    }

    for (;;)
    {
        if (t == 0) s_item = atomicAdd(A.fctl, 1);
        __syncthreads();
        const int item = s_item;
        if (item >= s_meta[2]) break;
        const int plan = s_meta[0], n1 = s_meta[1];
        if (item < n1)
        {
            const int u1 = g6nr::EB * 2 * A.KS1;
            const int u_b = item >> __ffs(A.KS1);
            const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (item / u1);
            const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
            int e, masks[4], ss = slot;
            const int nt = k1_slot_nt<true>(A, plan, slot, e, masks, ss);
            if (nt > 0)
            {
                const long long* tr = A.t_tab + 6 * e;
                if (b < static_cast<int>(tr[3]))
                {
                    const int unit = item + (ss - slot) * u1;
                    const int s_ks1 = __ffs(A.KS1) - 1;
                    const int h = unit & (A.KS1 - 1);
                    const int mat = (unit >> s_ks1) & 1;
                    const uint8_t* mw;
                    const int k = tr3_fc1_unit(A.t_fc1, A.t_ptab, A.t_kmix, e, tr, b * 2 + mat, mw);
                    if constexpr (!XCOMP)
                    {
                        if (nt > 1)
                        {
                            const int4 m4 = make_int4(masks[0], masks[1], masks[2], masks[3]);
                            if constexpr (HAS_K3)
                            {
                                if (k == 3) tr3_fc1_body_nt<3, true, true, CB>(A, plan, ss, b, mat, h, tr, mw, nt, m4, smem);
                                else tr3_fc1_body_nt<4, true, true, CB>(A, plan, ss, b, mat, h, tr, mw, nt, m4, smem);
                            }
                            else tr3_fc1_body_nt<4, true, true, CB>(A, plan, ss, b, mat, h, tr, mw, nt, m4, smem);
                        }
                        else
                        {
                            if constexpr (HAS_K3)
                            {
                                if (k == 3) tr3_fc1_body_k3w<XCOMP, CB>(A, plan, unit, tr, mw, masks[0], smem);
                                else tr4_fc1_body_w4<XCOMP, P256, CB>(A, plan, unit, tr, mw, masks[0], smem);
                            }
                            else tr4_fc1_body_w4<XCOMP, P256, CB>(A, plan, unit, tr, mw, masks[0], smem);
                        }
                    }
                    else
                    {
                        if constexpr (HAS_K3)
                        {
                            if (k == 3) tr3_fc1_body_k3w<XCOMP, CB>(A, plan, unit, tr, mw, masks[0], smem);
                            else tr4_fc1_body_w4<XCOMP, P256, CB>(A, plan, unit, tr, mw, masks[0], smem);
                        }
                        else tr4_fc1_body_w4<XCOMP, P256, CB>(A, plan, unit, tr, mw, masks[0], smem);
                    }
                    __threadfence();
                    __syncthreads();
                    if (t == 0) atomicAdd(A.fctl + 1 + ss * g6nr::EB + b, 1);
                }
            }
        }
        else
        {
            const int u2 = g6nr::EB * A.NS2;
            const int unit0 = item - n1;
            const int u_b = unit0 >> (__ffs(A.NS2) - 1);
            const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b >> (__ffs(g6nr::EB) - 1)) : (unit0 / u2);
            const int b = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (u_b & (g6nr::EB - 1)) : (u_b % g6nr::EB);
            int e, masks[4], ss;
            const int nt = k1_slot_nt<true>(A, plan, slot, e, masks, ss);
            if (nt > 0)
            {
                const long long* tr = A.t_tab + 6 * e;
                if (b < static_cast<int>(tr[3]))
                {
                    if (t == 0) while (ld_acquire(A.fctl + 1 + ss * g6nr::EB + b) < 2 * A.KS1) {}
                    __syncthreads();
                    const int unit = unit0 + (ss - slot) * u2;
                    if constexpr (!XCOMP)
                    {
                        if (nt > 1)
                        {
                            const int ns = unit & (A.NS2 - 1);
                            const int4 m4 = make_int4(masks[0], masks[1], masks[2], masks[3]);
                            if constexpr (HAS_K3)
                            {
                                if (tr[4] == 3) tr3_fc2_body_nt<3, true, CB, true>(A, ss, b, ns, tr, nt, m4, smem);
                                else tr3_fc2_body_nt<4, true, CB, true>(A, ss, b, ns, tr, nt, m4, smem);
                            }
                            else tr3_fc2_body_nt<4, true, CB, true>(A, ss, b, ns, tr, nt, m4, smem);
                        }
                        else
                        {
                            if constexpr (HAS_K3)
                            {
                                if (tr[4] == 3) tr3_fc2_body_k3w<NWARPS, CB>(A, unit, tr, masks[0], smem);
                                else tr4_fc2_body_w4<NWARPS, P256, CB>(A, unit, tr, masks[0], smem);
                            }
                            else tr4_fc2_body_w4<NWARPS, P256, CB>(A, unit, tr, masks[0], smem);
                        }
                    }
                    else
                    {
                        if constexpr (HAS_K3)
                        {
                            if (tr[4] == 3) tr3_fc2_body_k3w<NWARPS, CB>(A, unit, tr, masks[0], smem);
                            else tr4_fc2_body_w4<NWARPS, P256, CB>(A, unit, tr, masks[0], smem);
                        }
                        else tr4_fc2_body_w4<NWARPS, P256, CB>(A, unit, tr, masks[0], smem);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (t == 0)
    {
        __threadfence();
        s_meta[3] = atomicAdd(A.fctl + FCTL_N - 1, 1) == static_cast<int>(gridDim.x) - 1;
    }
    __syncthreads();
    if (s_meta[3])
    {
        for (int i = t; i < FCTL_N; i += blockDim.x) A.fctl[i] = 0;
        __threadfence();
    }
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

template <bool P256, int CB = TR3_MCG>
__global__ __launch_bounds__(256, 3) void k1_tr3_fused256_c3(K1Args A)
{
    k1_fused_w4_body<false, 8, true, P256, CB>(A);
}

template <bool HAS_K3, bool XCOMP, int CB = TR3_MCG>
__global__ __launch_bounds__(256, 2) void k1_tr3_fused256_c2(K1Args A)
{
    k1_fused_w4_body<HAS_K3, 8, XCOMP, false, CB>(A);
}

template <int CB = TR3_MCG>
__global__ __maxnreg__(128) void k1_tr3_fused512_w4(K1Args A)
{
    k1_fused_w4_body<false, 16, true, false, CB>(A);
}

// ------------------------------------------------------------------ NVFP4 build
__global__ void k1_nv_fc1(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[0];
    const int ups = (A.n_nch + A.n_cw - 1) / A.n_cw;
    const int nr = A.n_slots * ups;
    if (static_cast<int>(blockIdx.x) < nr)
    {
        const int slot = blockIdx.x / ups;
        if (slot < A.n_slots_dev[plan])
        {
            const int e = A.slot_expert[plan * SMAX + slot];
            const uint8_t* f1 = A.n_fc1 + static_cast<size_t>(e) * A.n_nch * KSH * 2 * Nvfp4::GR;
            rt_fc1_cta<Nvfp4>(f1, f1 + Nvfp4::GR, 2 * Nvfp4::GR, A.xfrag, (blockIdx.x % ups) * A.n_cw, A.n_cw,
                              A.n_nch, A.n_ks, A.n_alpha[e * 3], A.n_alpha[e * 3 + 1], A.act + static_cast<size_t>(slot) * 8 * ACTW, ACTW, reinterpret_cast<float*>(smem));
        }
    }
    else k1_shared_fc1(A, blockIdx.x - nr, reinterpret_cast<float*>(smem));
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

__global__ void k1_nv_fc2(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[1];
    const int nr = A.n_slots * A.n_ns2;
    if (static_cast<int>(blockIdx.x) < nr)
    {
        const int slot = blockIdx.x / A.n_ns2, os = blockIdx.x % A.n_ns2;
        if (slot < A.n_slots_dev[plan])
        {
            const int e = A.slot_expert[plan * SMAX + slot];
            rt_build_bfrag(smem, A.act + static_cast<size_t>(slot) * 8 * ACTW, ACTW, A.n_nch);
            __syncthreads();
            const int OTU = KSH / A.n_ns2;
            rt_gemv_body<Nvfp4>(A.n_fc2 + static_cast<size_t>(e) * KSH * A.n_nch * Nvfp4::GR, smem, A.n_nch, os * OTU, (os + 1) * OTU,
                                A.slot_w + (static_cast<size_t>(plan) * SMAX + slot) * 8, A.n_alpha[e * 3 + 2],
                                A.slot_mask[plan * SMAX + slot], A.ypart + static_cast<size_t>(slot) * 8 * HID, HID);
        }
    }
    else k1_shared_fc2(A, blockIdx.x - nr, smem);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[2] = plan;
}

// (M <= 8 only, never launched) out[m] = f16 or bf16( sum_k ypart[slot(m,k)][m] + ypart[shared][m] ), fp32 sum, fixed k order. grid = M*6, 128 threads.
__global__ void k1_nv_fin(K1Args A)
{
    const int plan = A.ctl[2];
    const int m = blockIdx.x / 6;
    const int n0 = (blockIdx.x % 6) * 1024 + threadIdx.x * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k <= 8; ++k)
    {
        const int slot = k < 8 ? A.tok_slot[(plan * G6_MX + m) * 8 + k] : SMAX;
        if (slot < 0) continue;
        const uint4 v = *reinterpret_cast<const uint4*>(A.ypart + (static_cast<size_t>(slot) * 8 + m) * HID + n0);
        const u32 w[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float2 f = __half22float2(as_h2(w[i]));
            acc[2 * i] += f.x; acc[2 * i + 1] += f.y;
        }
    }
    uint4 o;
    o.x = pack_o2(acc[0], acc[1], A.out_bf16); o.y = pack_o2(acc[2], acc[3], A.out_bf16);
    o.z = pack_o2(acc[4], acc[5], A.out_bf16); o.w = pack_o2(acc[6], acc[7], A.out_bf16);
    *reinterpret_cast<uint4*>(A.out + m * HID + n0) = o;
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[0] = (plan + 1) % A.n_plans;
}
