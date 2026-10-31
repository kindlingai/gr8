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
    size_t off;
    const int k = tr3_unit(A.t_kmix, e, tr, b * 2 + mat, off);
    const uint8_t* mw = A.t_fc1 + off;
    if ((!BS || K2) && k == 3) tr3_fc1_body<3, E4, BS, CB>(A, plan, unit, tr, mw, mask, smem);
    else if (!K2 || k == 4) tr3_fc1_body<4, E4, BS, CB>(A, plan, unit, tr, mw, mask, smem);
    else tr3_fc1_body<K2 ? 2 : 4, E4, BS, CB>(A, plan, unit, tr, mw, mask, smem);
}

// one fc2 unit. Down has one width per expert (tr[4]).
template <bool E4, bool K2, bool BS = false, int CB = TR3_MCG>
__device__ __forceinline__ void tr3_fc2_any(const K1Args& A, int plan, int unit, const long long* tr, int mask, u32* smem)
{
    if ((!BS || K2) && tr[4] == 3) tr3_fc2_body<3, E4, CB>(A, plan, unit, tr, mask, smem);
    else if (!K2 || tr[4] == 4) tr3_fc2_body<4, E4, CB>(A, plan, unit, tr, mask, smem);
    else tr3_fc2_body<K2 ? 2 : 4, E4, CB>(A, plan, unit, tr, mask, smem);
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
    size_t f1lo, f1hi;
    tr3_unit(A.t_kmix, e, tr, 0, f1lo);                          // units are back to back in unit order
    const int kl = tr3_unit(A.t_kmix, e, tr, 2 * nb - 1, f1hi);
    f1hi += kl * MB;
    const size_t f2lo = static_cast<size_t>(tr[1]), f2hi = f2lo + static_cast<size_t>(nb) * static_cast<size_t>(tr[4]) * MB;
    const uintptr_t g = uintptr_t(1) << A.tlbw;
    auto warm = [&](const uint8_t* base, size_t lo, size_t hi) {
        if (hi <= lo) return;
        const uintptr_t a0 = reinterpret_cast<uintptr_t>(base + lo), a1 = reinterpret_cast<uintptr_t>(base + hi);
        for (uintptr_t a = (a0 & ~(g - 1)) + threadIdx.x * g; a < a1; a += 32 * g)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(a < a0 ? a0 : a) : "memory");
    };
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
        int e, mask, ss;
        const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (static_cast<int>(blockIdx.x) >> (__ffs(ups) - 1)) : (static_cast<int>(blockIdx.x) / ups);
        if (k1_slot<SH>(A, plan, slot, e, mask, ss)) tr3_fc1_any<E4, K2, SH, CB>(A, plan, blockIdx.x + (ss - slot) * ups, e, mask, smem);
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
        int e, mask, ss;
        const int slot = ((g6nr::EB & (g6nr::EB - 1)) == 0) ? (static_cast<int>(blockIdx.x) >> (__ffs(ups) - 1)) : (static_cast<int>(blockIdx.x) / ups);
        if (k1_slot<SH>(A, plan, slot, e, mask, ss)) tr3_fc2_any<E4, K2, SH, CB>(A, plan, blockIdx.x + (ss - slot) * ups, A.t_tab + 6 * e, mask, smem);
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
    int e, mask, ss;
    const int slot = cta / ups;
    if (k1_slot<true>(A, plan, slot, e, mask, ss)) tr3_fc1_any<E4, K2, true, CB>(A, plan, cta + (ss - slot) * ups, e, mask, smem);
    if (PART == 2 && blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

template <bool E4, bool K2, int PART, int CB = TR3_MCG>
__global__ __launch_bounds__(512, 2) void k1_tr3_fc2_ovl(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = PART == 2 ? A.ctl[1] : A.ctl[0];
    const int ups = A.B * A.NS2;
    const int cta = static_cast<int>(blockIdx.x) + (PART == 2 ? k1_nsh<true>(A) * ups : 0);
    int e, mask, ss;
    const int slot = cta / ups;
    if (k1_slot<true>(A, plan, slot, e, mask, ss)) tr3_fc2_any<E4, K2, false, CB>(A, plan, cta + (ss - slot) * ups, A.t_tab + 6 * e, mask, smem);
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
            int e, mask, ss = slot;
            if (k1_slot<SH>(A, plan, slot, e, mask, ss)) tr3_fc1_any<E4, K2, SH, CB>(A, plan, item + (ss - slot) * u1, e, mask, smem);
            __threadfence();
            __syncthreads();
            if (t == 0) atomicAdd(done + ss * g6nr::EB + b, 1);
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
                int e, mask, ss;
                if (k1_slot<SH>(A, plan, slot, e, mask, ss))
                {
                    const long long* tr = A.t_tab + 6 * e;
                    if (b < static_cast<int>(tr[3]))
                    {
                        if (t == 0) while (ld_acquire(done + ss * g6nr::EB + b) < 2 * A.KS1) {}
                        __syncthreads();
                        tr3_fc2_any<E4, K2, SH, CB>(A, plan, unit + (ss - slot) * u2, tr, mask, smem);
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
            int e, mask, ss;
            if (k1_slot<SH>(A, plan, slot, e, mask, ss))
            {
                const long long* tr = A.t_tab + 6 * e;
                if (b < static_cast<int>(tr[3]))
                {
                    if (t == 0) while (ld_acquire(done + ss * g6nr::EB + b) < 2 * A.KS1) {}
                    __syncthreads();
                    tr3_fc2_any<E4, K2, SH, CB>(A, plan, unit + (ss - slot) * u2, tr, mask, smem);
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
