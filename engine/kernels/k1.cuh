// K1 entry kernels: 3 launches per layer per step (fc1, fc2, finalize); shared expert fused into the
// fc1/fc2 grids as extra CTAs; routing weights applied; fixed-order (deterministic) final sum.
#pragma once
#include "tr3.cuh"
#include "rowtile.cuh"
#include "densev.cuh"

// shared expert / dense MLP: int4 g32 records (one interleaved fc1 stream), or MXFP8 W8A16 records when A.s_mx
// (the MTP draft layer: gate and up are two streams, s_fc1 and s_up)
__device__ __forceinline__ void k1_shared_fc1(const K1Args& A, int cta, float* smem)
{
    __half* act = A.act + static_cast<size_t>(SMAX) * 8 * ACTW;
    if (A.s_mx) rt_fc1_cta<Mxfp8, GvGroup<Mxfp8>>(A.s_fc1, A.s_up, Mxfp8::GR, A.xfrag, cta * A.s_cw, A.s_cw, A.s_nch, A.s_ks, 1.f, 1.f, act, SACTW, smem);
    else rt_fc1_cta<Int4>(A.s_fc1, A.s_fc1 + Int4::GR, 2 * Int4::GR, A.xfrag, cta * A.s_cw, A.s_cw, A.s_nch, A.s_ks, 1.f, 1.f, act, SACTW, smem);
}

__device__ __forceinline__ void k1_shared_fc2(const K1Args& A, int os, u32* smem)
{
    rt_build_bfrag(smem, A.act + static_cast<size_t>(SMAX) * 8 * ACTW, SACTW, A.s_nch);
    __syncthreads();
    const int OTU = KSH / A.s_ns2;
    __half* y = A.s_direct ? K1_OUT(A) : A.ypart + static_cast<size_t>(SMAX) * 8 * HID;
    const int bf = A.s_direct && A.out_bf16, mask = (1 << A.M) - 1;
    if (A.s_mx) rt_gemv_body<Mxfp8, GvGroup<Mxfp8>>(A.s_fc2, smem, A.s_nch / 2, os * OTU, (os + 1) * OTU, nullptr, 1.f, mask, y, HID, bf);
    else rt_gemv_body<Int4>(A.s_fc2, smem, A.s_nch / 2, os * OTU, (os + 1) * OTU, nullptr, 1.f, mask, y, HID, bf);
}

// ------------------------------------------------------------------ TR3 build
// one fc1 unit of the slot's expert e at its width (tr3_unit). K2 = the layer has 2-bit units (bind::scan_widths):
// only then is the K2 body compiled in. The K2 body makes K3 units about 3.5% slower at M = 5 (GB10, #65). Layers
// without 2-bit units therefore take the K3 / K4 kernels.
template <bool E4, bool K2>
__device__ __forceinline__ void tr3_fc1_any(const K1Args& A, int plan, int unit, int e, int mask, u32* smem)
{
    const long long* tr = A.t_tab + 6 * e;
    const int mat = (unit / A.KS1) & 1, b = (unit / (A.KS1 * 2)) % A.B;
    if (b >= static_cast<int>(tr[3])) return;
    size_t off;
    const int k = tr3_unit(A.t_kmix, e, tr, b * 2 + mat, off);
    const uint8_t* mw = A.t_fc1 + off;
    if (k == 3) tr3_fc1_body<3, E4>(A, plan, unit, tr, mw, mask, smem);
    else if (!K2 || k == 4) tr3_fc1_body<4, E4>(A, plan, unit, tr, mw, mask, smem);
    else tr3_fc1_body<K2 ? 2 : 4, E4>(A, plan, unit, tr, mw, mask, smem);
}

// one fc2 unit. Down has one width per expert (tr[4]).
template <bool E4, bool K2>
__device__ __forceinline__ void tr3_fc2_any(const K1Args& A, int plan, int unit, const long long* tr, int mask, u32* smem)
{
    if (tr[4] == 3) tr3_fc2_body<3, E4>(A, plan, unit, tr, mask, smem);
    else if (!K2 || tr[4] == 4) tr3_fc2_body<4, E4>(A, plan, unit, tr, mask, smem);
    else tr3_fc2_body<K2 ? 2 : 4, E4>(A, plan, unit, tr, mask, smem);
}

// Work item slot -> expert, token mask and scratch slot. false = no work. Without SH the scratch slot is the item slot.
// SH: item slot 0 is the trellis shared expert with every token, and its scratch slot is n_slots (the finalize reads it
// there). Item slot i > 0 is routed slot i - 1. Every step has the shared expert, so it is handed out first.
template <bool SH>
__device__ __forceinline__ bool k1_slot(const K1Args& A, int plan, int slot, int& e, int& mask, int& ss)
{
    if (SH && slot == 0) { e = A.sh_e; mask = A.sh_mask; ss = A.n_slots; return true; }
    ss = slot - (SH ? 1 : 0);
    if (ss < A.n_slots_dev[plan])
    {
        e = A.slot_expert[plan * SMAX + ss];
        mask = A.slot_mask[plan * SMAX + ss];
        return true;
    }
    return false;
}

// E4 = the layer's routed experts are exlr8-e4m3 (codebook rounded to E4M3, tr3.cuh tr3_cb2); K2 = it has 2-bit units;
// SH = its shared expert is trellis row 256, run as one more slot (exlr8 v2, #65). The binder picks the instantiation
// per layer (bind::make_k1); E4 = K2 = SH = false is the TR3 kernel.
template <bool E4 = false, bool K2 = false, bool SH = false>
__global__ __launch_bounds__(1024) void k1_tr3_fc1(K1Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[0];
    const int ups = A.B * 2 * A.KS1, nr = (A.n_slots + SH) * ups;
    if (static_cast<int>(blockIdx.x) < nr)
    {
        int e, mask, ss;
        const int slot = blockIdx.x / ups;
        if (k1_slot<SH>(A, plan, slot, e, mask, ss)) tr3_fc1_any<E4, K2>(A, plan, blockIdx.x + (ss - slot) * ups, e, mask, smem);
    }
    else if (!SH) k1_shared_fc1(A, blockIdx.x - nr, reinterpret_cast<float*>(smem));
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[1] = plan;
}

template <bool E4 = false, bool K2 = false, bool SH = false>
__global__ __launch_bounds__(512, 2) void k1_tr3_fc2(K1Args A)   // 68 -> 64 regs: 2 CTAs per SM (item 1h L1)
{
    extern __shared__ __align__(16) u32 smem[];
    const int plan = A.ctl[1];
    const int ups = A.B * A.NS2, nr = (A.n_slots + SH) * ups;
    if (static_cast<int>(blockIdx.x) < nr)
    {
        int e, mask, ss;
        const int slot = blockIdx.x / ups;
        if (k1_slot<SH>(A, plan, slot, e, mask, ss)) tr3_fc2_any<E4, K2>(A, plan, blockIdx.x + (ss - slot) * ups, A.t_tab + 6 * e, mask, smem);
    }
    else if (!SH) k1_shared_fc2(A, blockIdx.x - nr, smem);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[2] = plan;
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

template <bool SH = false>
__global__ void k1_tr3_fin(K1Args A)
{
    const int plan = A.ctl[2];
    tr3_fin_body<SH>(A, plan);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[0] = (plan + 1) % A.n_plans;
#if DEC_SLOT >= 2
    if (A.slot.send) dslot::publish(A.slot, threadIdx.x < 32);      // warp 0 stored (tr3_fin_body)
#endif
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

template <bool E4, bool K2, bool SH>
__device__ __forceinline__ void k1_fused_body(const K1Args& A)
{
    extern __shared__ __align__(16) u32 smem[];
    __shared__ int s_item;
    const int t = threadIdx.x;
    const int plan = A.ctl[0];
    const int u1 = A.B * 2 * A.KS1, n1 = (A.n_slots + SH) * u1;         // fc1 units per slot, routed (+ SH shared) fc1 items
    const int sh1 = SH ? 0 : (A.s_nch + A.s_cw - 1) / A.s_cw;           // shared fc1 CTAs (s_* records)
    const int u2 = A.B * A.NS2, n2 = (A.n_slots + SH) * u2;             // fc2 units per slot, routed (+ SH shared) fc2 items
    const int total = n1 + sh1 + n2 + (SH ? 0 : A.s_ns2);
    int* done = A.fctl + 1;
    int* sdone = A.fctl + 1 + SMAX * g6nr::EB;
    if (blockIdx.x == 0 && t == 0) A.ctl[2] = plan;
    for (;;)
    {
        if (t == 0) s_item = atomicAdd(A.fctl, 1);
        __syncthreads();
        const int item = s_item;
        if (item >= total) break;
        if (item < n1)
        {
            const int slot = item / u1, b = (item / (A.KS1 * 2)) % A.B;
            int e, mask, ss = slot;
            if (k1_slot<SH>(A, plan, slot, e, mask, ss)) tr3_fc1_any<E4, K2>(A, plan, item + (ss - slot) * u1, e, mask, smem);
            __threadfence();
            __syncthreads();
            if (t == 0) atomicAdd(done + ss * A.B + b, 1);
        }
        else if (item < n1 + sh1)
        {
            k1_shared_fc1(A, item - n1, reinterpret_cast<float*>(smem));
            __threadfence();
            __syncthreads();
            if (t == 0) atomicAdd(sdone, 1);
        }
        else if (item < n1 + sh1 + n2)
        {
            const int unit = item - n1 - sh1, slot = unit / u2, b = (unit / A.NS2) % A.B;
            int e, mask, ss;
            if (k1_slot<SH>(A, plan, slot, e, mask, ss))
            {
                const long long* tr = A.t_tab + 6 * e;
                if (b < static_cast<int>(tr[3]))
                {
                    if (t == 0) while (ld_acquire(done + ss * A.B + b) < 2 * A.KS1) {}
                    __syncthreads();
                    tr3_fc2_any<E4, K2>(A, plan, unit + (ss - slot) * u2, tr, mask, smem);
                }
            }
        }
        else
        {
            if (t == 0) while (ld_acquire(sdone) < sh1) {}
            __syncthreads();
            k1_shared_fc2(A, item - n1 - sh1 - n2, smem);
        }
        __syncthreads();                                                // smem and s_item are reused
    }
    if (t == 0)
    {
        __threadfence();
        if (atomicAdd(A.fctl + FCTL_N - 1, 1) == static_cast<int>(gridDim.x) - 1)
        {
            for (int i = 0; i < FCTL_N; ++i) A.fctl[i] = 0;
            __threadfence();
        }
    }
#if DEC_SLOT >= 2
    if (A.slot.send && A.s_direct) dslot::publish(A.slot, true);
#endif
}

template <bool E4 = false, bool K2 = false, bool SH = false>
__global__ __launch_bounds__(512) void k1_tr3_fused(K1Args A)
{
    k1_fused_body<E4, K2, SH>(A);
}

// The same with the trellis shared expert. Under __launch_bounds__(512) alone nvcc gives the no-K2 instantiation 64
// registers and spills (104 B stores, 132 B loads; the others get 101-128 and no spills), so this entry sets the cap.
template <bool E4 = true, bool K2 = false>
__global__ __maxnreg__(128) void k1_tr3_fused_sh(K1Args A)
{
    k1_fused_body<E4, K2, true>(A);
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

// out[m] = f16 or bf16( sum_k ypart[slot(m,k)][m] + ypart[shared][m] ), fp32 sum, fixed k order. grid = M*6, 128 threads.
__global__ void k1_nv_fin(K1Args A)
{
    const int plan = A.ctl[2];
    const int m = blockIdx.x / 6;
    const int n0 = (blockIdx.x % 6) * 1024 + threadIdx.x * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k <= 8; ++k)
    {
        const int slot = k < 8 ? A.tok_slot[(plan * 8 + m) * 8 + k] : SMAX;
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
