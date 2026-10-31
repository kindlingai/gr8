// gr8 DeepSeek-V4.1-Flash router. Our own code; standalone (no glue.cuh).
//
// Semantics (checkpoint config: scoring sqrtsoftplus, topk noaux_tc, norm_topk_prob, routed_scaling_factor 1.5):
//   logit[m][e]  = gate[e] . x[m]                    gate bf16 [384][5120], x bf16 [M][5120], fp32 accumulate
//   score[m][e]  = sqrt(softplus(logit)),  softplus(v) = v > 20 ? v : log1p(exp(v))
//   key[m][e]    = score + bias[e]                    bias = e_score_correction_bias, fp32 [384] (selection only)
//   routes       = the 6 largest keys, ties -> lowest expert id
//   weight[m][k] = score[sel k] / (sum of the 6 selected scores + 1e-20) * 1.5          (the UNBIASED score is normalized)
// Text only: gate.bias_vl (image tokens) is not used.
//
// Two kernels, M <= 8 (decode):
//   k_gate_logits<<<NEXP / 8, 256>>>(gate, x, M, logits)   warp per expert row, whole row in flight, fixed-order butterfly sum
//   k_plan<<<1, PLAN_THREADS>>>(logits, bias, M, plan)     top-6 per token and the slot plan the mx4 kernels read
// Plan layout = glue.cuh's RoutePlan for M <= 8: slot s = one expert's n=8 tile, experts ascending; column c of a slot = token c.
// The shared expert is not a slot of this plan (it runs its own kernel on every token).
// Determinism: fixed lane order everywhere, no atomics on floats. Versus the reference GEMM the fp32 summation order differs
// (rounding level only); a route flips only when two keys are within that rounding of each other.
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o ds_router_test tests/ds_router_test.cu
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include "ds_plan_mt.h"

namespace dsr
{
constexpr int NEXP = 384;                 // routed experts
constexpr int HIDDEN = 5120;
constexpr int TOPK = 6;
constexpr int MMAX = 8;                   // tokens per step (one n=8 tile)
constexpr int SMAX = MMAX * TOPK;         // slots per step
constexpr float ROUTE_SCALE = 1.5f;
constexpr int PLAN_THREADS = 1024;        // block of k_plan
// The router is generic in (NE, TK) = (routed experts, routes per token); the defaults are the main model (384, top-6). The DSpark
// draft layers 40..42 are (128, 3): k_gate_logits<128>, k_plan<128, 3>, PlanT<3>, dsh::k_combine<..., 3>.
template <int NE, int TK>
struct RouterGeom
{
    static constexpr int SM = MMAX * TK;      // slots per step
    static constexpr int RW = NE / 32;        // warps that own experts in k_plan's slot assignment; each lane holds RW experts
    static_assert(NE % 32 == 0 && NE % 8 == 0 && NE <= PLAN_THREADS && HIDDEN % 256 == 0, "router tiling");
    static_assert(SM * 8 <= PLAN_THREADS && MMAX <= PLAN_THREADS / 32 && TK <= 8 && TK <= NE && MMAX * NE % PLAN_THREADS == 0,
                  "k_plan: one thread per slot entry, one warp per token");
};
constexpr int RW = RouterGeom<NEXP, TOPK>::RW;

template <int TK>
struct PlanT
{
    static constexpr int SM = MMAX * TK;
    int n_slots;                 // distinct experts this step
    int pad_[3];
    int slot_expert[SM];         // ascending expert id; -1 beyond n_slots
    int slot_mask[SM];           // bit c = column c (token c) uses the slot
    float slot_w[SM][8];         // routing weight (x 1.5) of the token in column c, 0 if unused
    int slot_tok[SM][8];         // token in column c, -1 = unused
    int tok_slot[MMAX][8];       // slot of route k of token m; -1 for k >= TK or m >= M
    int tok_col[MMAX][8];        // column of token m in that slot (= m)
    float tok_w[MMAX][8];
    int tok_expert[MMAX][8];     // selected experts in selection order (best key first)
};
using Plan = PlanT<TOPK>;

#ifdef __CUDACC__
__host__ __device__ __forceinline__ float sqrt_softplus(float x) { return sqrtf(x > 20.f ? x : log1pf(expf(x))); }

// Programmatic dependent launch (the same two instructions as mx4::pdl_*; both are no-ops unless the kernel was launched with
// cudaLaunchAttributeProgrammaticStreamSerialization). The _p twins and the PDLM = 1 instances below follow the chain rules of
// ds41_moe_ops.cuh: every PDL-launched kernel executes pdl_wait on every path, and a kernel triggers only after its own wait when its
// dependents read data produced two kernels earlier.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void bf16x8(const uint4& v, float (&f)[8])
{
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}

// grid = NE / 8, block 256. Warp w of CTA b = expert 8 b + w. Lane l reads 16-byte pieces l, l + 32, ... of the row (all NV in
// flight), and the matching pieces of every token's x (L1/L2 hits: x is 80 KB for M = 8). logits [M][NEXP] fp32.
// PDLM = 1: the whole-layer PDL chain instance (launched under build_xfrag_p): triggers at once, waits just before its stores.
template <int NE = NEXP, int PDLM = 0>
__global__ void __launch_bounds__(256) k_gate_logits(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ x, int M, float* logits)
{
    static_assert(NE % 8 == 0, "k_gate_logits: 8 experts per CTA");
    constexpr int NV = HIDDEN / 8 / 32;
    if constexpr (PDLM != 0) pdl_trigger();
    const int lane = threadIdx.x & 31, e = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint4* wr = reinterpret_cast<const uint4*>(gate + static_cast<size_t>(e) * HIDDEN) + lane;
    const uint4* xr = reinterpret_cast<const uint4*>(x) + lane;
    uint4 w[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w[i].x), "=r"(w[i].y), "=r"(w[i].z), "=r"(w[i].w) : "l"(wr + 32 * i));
    #pragma unroll
    for (int i = 0; i < NV; ++i) asm volatile("" : "+r"(w[i].x), "+r"(w[i].y), "+r"(w[i].z), "+r"(w[i].w));   // all loads issued before any use
    float acc[MMAX];
    #pragma unroll
    for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
    #pragma unroll
    for (int i = 0; i < NV; ++i)
    {
        float wf[8];
        bf16x8(w[i], wf);
        #pragma unroll
        for (int m = 0; m < MMAX; ++m)
            if (m < M)
            {
                float xf[8];
                bf16x8(__ldg(xr + m * (HIDDEN / 8) + 32 * i), xf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] = fmaf(wf[j], xf[j], acc[m]);
            }
    }
    if constexpr (PDLM != 0) pdl_wait();      // tail wait: the gate and x loads above overlap the kernel before; only the stores need it
    #pragma unroll
    for (int m = 0; m < MMAX; ++m)
    {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
        if (lane == 0 && m < M) logits[m * NE + e] = acc[m];
    }
}

#ifdef DSR_PROF
__device__ long long g_dsr_prof[8];       // cycle stamps of warp 0 (test only): start, scored, picked, scatter
#define DSR_STAMP(i) do { if (t == 0) g_dsr_prof[i] = clock64(); } while (0)
#else
#define DSR_STAMP(i) do {} while (0)
#endif
// grid 1, block PLAN_THREADS. Phase 1: all threads score (token, expert) pairs into smem. Phase 2: warp w < M picks the routes of
// token w (argmax by redux.sync: the largest key, then the lowest expert id among lanes holding it). Phase 3: thread t < NEXP owns
// expert t for the slot assignment. logits [M][NEXP] fp32 (finite), bias [NEXP] fp32, 1 <= M <= 8.
__device__ __forceinline__ unsigned key_bits(float f)       // order-preserving float -> u32
{
    const unsigned u = __float_as_uint(f);
    return u ^ ((u >> 31) ? 0xffffffffu : 0x80000000u);
}
// KC = routes kept per token (the top-k cut, GR8_DS_MOE_TOPK): the first KC picks in selection order (best biased key first); the weights renormalize over the
// kept picks; a dropped route k >= KC is absent (the init above leaves tok_slot -1, tok_col 0, tok_w 0, tok_expert -1 for it). KC == TK is today's code.
template <int NE, int TK, int PDLM, int KC = TK>
__device__ __forceinline__ void plan_body(const float* __restrict__ logits, const float* __restrict__ bias, int M, PlanT<TK>* __restrict__ P)
{
    static_assert(KC >= 1 && KC <= TK, "plan_body: kept routes");
    constexpr int RW = RouterGeom<NE, TK>::RW, SM = RouterGeom<NE, TK>::SM;
    __shared__ unsigned emask[NE];                // bit m = token m selected the expert
    __shared__ int sel_e[MMAX][TK];
    __shared__ float sel_w[MMAX][TK];
    __shared__ int slot_of[NE];
    __shared__ int wcount[RW];
    __shared__ float s_sm[MMAX][NE], k_sm[MMAX][NE];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    DSR_STAMP(0);
    if (t < NE) emask[t] = 0u;
    if (t < SM * 8) { (&P->slot_w[0][0])[t] = 0.f; (&P->slot_tok[0][0])[t] = -1; }
    if (t < SM) { P->slot_expert[t] = -1; P->slot_mask[t] = 0; }
    if (t < MMAX * 8) { (&P->tok_slot[0][0])[t] = -1; (&P->tok_col[0][0])[t] = 0; (&P->tok_w[0][0])[t] = 0.f; (&P->tok_expert[0][0])[t] = -1; }
    if constexpr (PDLM != 0) { pdl_wait(); pdl_trigger(); }   // the logits come from the gate; the plan init above touches nothing the gate reads
    #pragma unroll
    for (int j = 0; j < MMAX * NE / PLAN_THREADS; ++j)
    {
        const int i = t + PLAN_THREADS * j;
        if (i < M * NE)
        {
            const float x = logits[i];
            const float sc = sqrt_softplus(x);
            (&s_sm[0][0])[i] = sc;
            (&k_sm[0][0])[i] = sc + bias[i % NE];
        }
    }
    __syncthreads();
    DSR_STAMP(1);
    if (w < M)
    {
        const int m = w;
        float s[RW], key[RW];
        #pragma unroll
        for (int i = 0; i < RW; ++i) { s[i] = s_sm[m][l + 32 * i]; key[i] = k_sm[m][l + 32 * i]; }
        unsigned taken = 0;
        float my_k = key[0];
        int my_e = l;
        #pragma unroll
        for (int i = 1; i < RW; ++i)
            if (key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }       // strict: the lowest id wins a tie on this lane
        float picked[KC];
        int pe[KC];
        #pragma unroll
        for (int k = 0; k < KC; ++k)
        {
            const unsigned kb = key_bits(my_k);
            const unsigned best = __reduce_max_sync(0xffffffffu, kb);
            const int be = static_cast<int>(__reduce_min_sync(0xffffffffu, kb == best ? static_cast<unsigned>(my_e) : 0xffffffffu));
            float sv = 0.f;
            if ((be & 31) == l)                  // the winner's lane: take the score, refill its local best
            {
                #pragma unroll
                for (int i = 0; i < RW; ++i) if (i == (be >> 5)) sv = s[i];
                taken |= 1u << (be >> 5);
                my_k = -INFINITY;
                my_e = 0x7fffffff;
                #pragma unroll
                for (int i = 0; i < RW; ++i)
                    if (!(taken >> i & 1) && key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }
            }
            picked[k] = __shfl_sync(0xffffffffu, sv, be & 31);
            pe[k] = be;
        }
        DSR_STAMP(2);
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < KC; ++k) sum += picked[k];                  // selection order, fixed
        if (l < KC)
        {
            float wv = 0.f; int ev = 0;
            #pragma unroll
            for (int k = 0; k < KC; ++k) if (k == l) { wv = picked[k]; ev = pe[k]; }
            sel_e[m][l] = ev;
            sel_w[m][l] = wv / (sum + 1e-20f) * ROUTE_SCALE;
            atomicOr(&emask[ev], 1u << m);
        }
    }
    __syncthreads();
    // thread t = expert t: its slot = the number of selected experts below it
    unsigned em = 0u;
    int used = 0;
    unsigned wm = 0u;
    if (t < NE)
    {
        em = emask[t];
        used = em != 0u;
        wm = __ballot_sync(0xffffffffu, used);          // warps 0..RW-1 are full and take this branch together
        if (l == 0) wcount[w] = __popc(wm);
    }
    __syncthreads();
    if (t < NE)
    {
        int base = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) if (i < w) base += wcount[i];
        if (used)
        {
            const int slot = base + __popc(wm & ((1u << l) - 1u));
            slot_of[t] = slot;
            P->slot_expert[slot] = t;
            P->slot_mask[slot] = static_cast<int>(em);
        }
    }
    if (t == 0)
    {
        int n = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) n += wcount[i];
        P->n_slots = n;
    }
    __syncthreads();
    DSR_STAMP(3);
    if (t < M * TK)
    {
        const int m = t / TK, k = t % TK;
        if constexpr (KC < TK) if (k >= KC) return;                     // dropped route: stays absent
        const int e = sel_e[m][k], slot = slot_of[e];
        P->tok_slot[m][k] = slot;
        P->tok_col[m][k] = m;
        P->tok_w[m][k] = sel_w[m][k];
        P->tok_expert[m][k] = e;
        P->slot_w[slot][m] = sel_w[m][k];
        P->slot_tok[slot][m] = m;
    }
}
template <int NE = NEXP, int TK = TOPK, int KC = TK>
__global__ void __launch_bounds__(PLAN_THREADS) k_plan(const float* __restrict__ logits, const float* __restrict__ bias, int M, PlanT<TK>* __restrict__ P)
{
    plan_body<NE, TK, 0, KC>(logits, bias, M, P);
}
// PDL twin: waits for the gate (head), then triggers so the shared expert's fc1 can start while the plan is still being built.
template <int NE = NEXP, int TK = TOPK, int KC = TK>
__global__ void __launch_bounds__(PLAN_THREADS) k_plan_p(const float* __restrict__ logits, const float* __restrict__ bias, int M, PlanT<TK>* __restrict__ P)
{
    plan_body<NE, TK, 1, KC>(logits, bias, M, P);
}

// ------------------------------------------------------------------------------------------------ multi-tile router, M <= 32
// New kernels only: k_gate_logits / k_plan above are unchanged (M <= 8).
// Same gate weight read as k_gate_logits (once, 20 x 16 B per lane, in registers), then token groups of 8 (x re-read from L1/L2).
// Per token the fma order is k_gate_logits's, so a token's logits are bit-identical to the M <= 8 kernel's.
// PDLM = 1: the PDL chain instance (trigger at once, wait before the first store), as gate_logits_body.
template <int NE = NEXP, int PDLM = 0>
__global__ void __launch_bounds__(256) k_gate_logits_mt(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ x, int M, float* logits)
{
    static_assert(NE % 8 == 0, "k_gate_logits_mt: 8 experts per CTA");
    constexpr int NV = HIDDEN / 8 / 32;
    if constexpr (PDLM != 0) pdl_trigger();
    const int lane = threadIdx.x & 31, e = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint4* wr = reinterpret_cast<const uint4*>(gate + static_cast<size_t>(e) * HIDDEN) + lane;
    const uint4* xr = reinterpret_cast<const uint4*>(x) + lane;
    uint4 w[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w[i].x), "=r"(w[i].y), "=r"(w[i].z), "=r"(w[i].w) : "l"(wr + 32 * i));
    #pragma unroll
    for (int i = 0; i < NV; ++i) asm volatile("" : "+r"(w[i].x), "+r"(w[i].y), "+r"(w[i].z), "+r"(w[i].w));
    #pragma unroll 1
    for (int p0 = 0; p0 < M; p0 += MMAX)
    {
        float acc[MMAX];
        #pragma unroll
        for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;
        #pragma unroll
        for (int i = 0; i < NV; ++i)
        {
            float wf[8];
            bf16x8(w[i], wf);
            #pragma unroll
            for (int m = 0; m < MMAX; ++m)
                if (p0 + m < M)
                {
                    float xf[8];
                    bf16x8(__ldg(xr + (p0 + m) * (HIDDEN / 8) + 32 * i), xf);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[m] = fmaf(wf[j], xf[j], acc[m]);
                }
        }
        if constexpr (PDLM != 0) pdl_wait();
        #pragma unroll
        for (int m = 0; m < MMAX; ++m)
        {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
            if (lane == 0 && p0 + m < M) logits[(p0 + m) * NE + e] = acc[m];
        }
    }
}

// ------------------------------------------------------------------------------------------------ read-once prefill gate (GR8_DS_GATE_RO=1)
// k_gate_logits_ro: the gate of the read-once prefill path (router_ro, M >= RO_FROM) as an 8-expert x 8-token register tile per warp.
// k_gate_logits_mt re-reads every x piece for one expert (8 FMA per 16-byte load, 8 bf16 -> fp32 converts per 8 FMA, 384 CTA-passes over x).
// Here a CTA keeps the bf16 rows of GRO_EXP = 8 experts in smem (80 KB, filled once) and each warp takes token groups of 8 (g = warp, warp + 8, ...):
// per K piece (lane + 32 i) it loads the 8 tokens' x pieces once, converts them once, and runs all 8 experts on them (64 independent chains).
// The bits are k_gate_logits_mt's: per (token, expert, lane) the chain is acc = 0.f, then for i = 0..19, j = 0..7: acc = fmaf(w[8 (lane + 32 i) + j], x[same], acc)
// (a loop order change between independent chains does not change any chain), then the same xor butterfly 16, 8, 4, 2, 1 and lane 0's store.
// gro_group() is the one body the kernel and the host test (tests/ds_gate_ro_test.cu) both run, lane by lane on the CPU.
constexpr int GRO_EXP = 8;                                              // experts per CTA (grid.x = NE / 8, as k_gate_logits_mt)
constexpr int GRO_TOK = 8;                                              // tokens per warp tile
constexpr int GRO_PIECES = HIDDEN / 8;                                  // 16-byte pieces per row
constexpr int GRO_NV = GRO_PIECES / 32;                                 // pieces per lane (20)
constexpr size_t GRO_SMEM = (size_t)GRO_EXP * GRO_PIECES * sizeof(uint4);   // 81920 B dynamic smem per CTA
static_assert(GRO_NV % 2 == 0 && GRO_EXP * GRO_PIECES % 256 == 0, "k_gate_logits_ro: two-buffer i loop, smem fill by 256 threads");
#ifdef __CUDA_ARCH__
#define GRO_UNROLL _Pragma("unroll")
#define GRO_UNROLL1 _Pragma("unroll 1")
#else
#define GRO_UNROLL
#define GRO_UNROLL1
#endif
__host__ __device__ __forceinline__ float gro_f(unsigned u)
{
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f; std::memcpy(&f, &u, 4); return f;
#endif
}
__host__ __device__ __forceinline__ void gro_bf16x8(const uint4& v, float (&f)[8])        // the same widening as bf16x8 above (exact)
{
    f[0] = gro_f(v.x << 16); f[1] = gro_f(v.x & 0xffff0000u);
    f[2] = gro_f(v.y << 16); f[3] = gro_f(v.y & 0xffff0000u);
    f[4] = gro_f(v.z << 16); f[5] = gro_f(v.z & 0xffff0000u);
    f[6] = gro_f(v.w << 16); f[7] = gro_f(v.w & 0xffff0000u);
}
__host__ __device__ __forceinline__ uint4 gro_ldx(const uint4* p)
{
#ifdef __CUDA_ARCH__
    return __ldg(p);
#else
    return *p;
#endif
}
// one K piece i for the whole tile: xr = the 8 tokens' pieces, sw = this lane's smem column of the 8 expert rows (sw[e * GRO_PIECES + 32 i]).
__host__ __device__ __forceinline__ void gro_piece(const uint4 (&xr)[GRO_TOK], const uint4* sw, float (&acc)[GRO_EXP][GRO_TOK])
{
    float xf[GRO_TOK][8];
    GRO_UNROLL
    for (int t = 0; t < GRO_TOK; ++t) gro_bf16x8(xr[t], xf[t]);
    GRO_UNROLL
    for (int e = 0; e < GRO_EXP; ++e)
    {
        float wf[8];
        gro_bf16x8(sw[e * GRO_PIECES], wf);
        GRO_UNROLL
        for (int t = 0; t < GRO_TOK; ++t)
        {
            GRO_UNROLL
            for (int j = 0; j < 8; ++j) acc[e][t] = fmaf(wf[j], xf[t][j], acc[e][t]);
        }
    }
}
// One token group for one lane: xb = x as 16-byte pieces + lane, ro[t] = piece offset of token t's row, swl = smem rows + lane. acc is the lane's partial sums.
__host__ __device__ __forceinline__ void gro_group(const uint4* xb, const int (&ro)[GRO_TOK], const uint4* swl, float (&acc)[GRO_EXP][GRO_TOK])
{
    GRO_UNROLL
    for (int e = 0; e < GRO_EXP; ++e)
    {
        GRO_UNROLL
        for (int t = 0; t < GRO_TOK; ++t) acc[e][t] = 0.f;
    }
    uint4 xa[GRO_TOK], xc[GRO_TOK];
    GRO_UNROLL
    for (int t = 0; t < GRO_TOK; ++t) xa[t] = gro_ldx(xb + ro[t]);
    GRO_UNROLL1
    for (int i = 0; i < GRO_NV; i += 2)
    {
        GRO_UNROLL
        for (int t = 0; t < GRO_TOK; ++t) xc[t] = gro_ldx(xb + ro[t] + 32 * (i + 1));
        gro_piece(xa, swl + 32 * i, acc);
        if (i + 2 < GRO_NV)
        {
            GRO_UNROLL
            for (int t = 0; t < GRO_TOK; ++t) xa[t] = gro_ldx(xb + ro[t] + 32 * (i + 2));
        }
        gro_piece(xc, swl + 32 * (i + 1), acc);
    }
}
// Row offsets of a group at row r0 of M rows: rows past M re-read the last live row (their sums are never stored).
__host__ __device__ __forceinline__ void gro_rows(int r0, int M, int (&ro)[GRO_TOK])
{
    const int last = (M - r0 < GRO_TOK ? M - r0 : GRO_TOK) - 1;
    GRO_UNROLL
    for (int t = 0; t < GRO_TOK; ++t) ro[t] = (r0 + (t < last ? t : last)) * GRO_PIECES;
}
#ifdef __CUDACC__
// grid = NE / 8, block 256, GRO_SMEM dynamic smem (cudaFuncSetAttribute MaxDynamicSharedMemorySize first: gro_attr in ds41_moe_ops.cuh). Plain launch (no PDL).
template <int NE = NEXP>
__global__ void __launch_bounds__(256, 1) k_gate_logits_ro(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ x, int M, float* logits)
{
    static_assert(NE % GRO_EXP == 0, "k_gate_logits_ro: 8 experts per CTA");
    extern __shared__ __align__(16) unsigned char gro_smem[];
    uint4* sw = reinterpret_cast<uint4*>(gro_smem);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, e0 = blockIdx.x * GRO_EXP;
    {
        const uint4* src = reinterpret_cast<const uint4*>(gate + static_cast<size_t>(e0) * HIDDEN);
        #pragma unroll
        for (int k = 0; k < GRO_EXP * GRO_PIECES / 256; ++k) sw[threadIdx.x + 256 * k] = __ldg(src + threadIdx.x + 256 * k);
    }
    __syncthreads();
    const uint4* xb = reinterpret_cast<const uint4*>(x) + lane;
    for (int g = wid; g * GRO_TOK < M; g += 8)
    {
        int ro[GRO_TOK];
        gro_rows(g * GRO_TOK, M, ro);
        float acc[GRO_EXP][GRO_TOK];
        gro_group(xb, ro, sw + lane, acc);
        #pragma unroll
        for (int e = 0; e < GRO_EXP; ++e)
        {
            #pragma unroll
            for (int t = 0; t < GRO_TOK; ++t)
            {
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc[e][t] += __shfl_xor_sync(0xffffffffu, acc[e][t], o);
                if (lane == 0 && g * GRO_TOK + t < M) logits[(g * GRO_TOK + t) * NE + e0 + e] = acc[e][t];
            }
        }
    }
}
#endif

// grid 1, block PLAN_THREADS. Warp w = token w (< M <= MM <= 32) scores its experts in registers (lane l holds experts l + 32 i), picks the
// TK routes exactly as k_plan does (same scores, same redux.sync argmax, ties -> lowest id, same 1.5 x score / sum), and marks the experts in
// emask (integer atomicOr). Then expert thread t: n = tokens of expert t, nt = ceil(n / 8) tiles; two warp scans give its slot (distinct
// experts below) and its first global tile (tiles below). Route thread (m, k): position = rank of token m among the expert's tokens.
// Output = PlanMT (ds_plan_mt.h); every field is written here (initial values as plan_ref).
// PDLM = 1: the PDL chain instance (wait for the gate before the logits are read, then trigger), as plan_body.
template <int NE = NEXP, int TK = TOPK, int MM = MT_MAX, int PDLM = 0, int KC = TK>
__global__ void __launch_bounds__(PLAN_THREADS) k_plan_mt(const float* __restrict__ logits, const float* __restrict__ bias, int M, PlanMT<TK, MM>* __restrict__ P)
{
    constexpr int RW = NE / 32, SM = PlanMT<TK, MM>::SM;
    static_assert(NE % 32 == 0 && NE <= PLAN_THREADS && MM <= PLAN_THREADS / 32 && MM <= 32 && TK <= 8 && TK <= NE && MM * TK <= PLAN_THREADS,
                  "k_plan_mt tiling");
    static_assert(KC >= 1 && KC <= TK, "k_plan_mt: kept routes");
    __shared__ unsigned emask[NE];                // bit m = token m selected the expert
    __shared__ int tile0_of[NE];
    __shared__ int sel_e[MM][TK];
    __shared__ float sel_w[MM][TK];
    __shared__ int wcount[RW], wtiles[RW];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    if (t < NE) emask[t] = 0u;
    for (int i = t; i < SM; i += PLAN_THREADS) { P->slot_expert[i] = -1; P->slot_tile0[i] = 0; P->slot_nt[i] = 0; P->tile_mask[i] = 0; }
    for (int i = t; i < SM * 8; i += PLAN_THREADS) { (&P->tile_w[0][0])[i] = 0.f; (&P->tile_tok[0][0])[i] = -1; }
    for (int i = t; i < MM * 8; i += PLAN_THREADS)
    {
        (&P->tok_slot[0][0])[i] = -1; (&P->tok_col[0][0])[i] = 0; (&P->tok_w[0][0])[i] = 0.f; (&P->tok_expert[0][0])[i] = -1;
    }
    if constexpr (PDLM != 0) { pdl_wait(); pdl_trigger(); }
    float s[RW], key[RW];
    if (w < M)
    {
        #pragma unroll
        for (int i = 0; i < RW; ++i)
        {
            const float sc = sqrt_softplus(logits[w * NE + l + 32 * i]);
            s[i] = sc;
            key[i] = sc + bias[l + 32 * i];
        }
    }
    __syncthreads();
    if (w < M)
    {
        const int m = w;
        unsigned taken = 0;
        float my_k = key[0];
        int my_e = l;
        #pragma unroll
        for (int i = 1; i < RW; ++i)
            if (key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }       // strict: the lowest id wins a tie on this lane
        float picked[KC];
        int pe[KC];
        #pragma unroll
        for (int k = 0; k < KC; ++k)
        {
            const unsigned kb = key_bits(my_k);
            const unsigned best = __reduce_max_sync(0xffffffffu, kb);
            const int be = static_cast<int>(__reduce_min_sync(0xffffffffu, kb == best ? static_cast<unsigned>(my_e) : 0xffffffffu));
            float sv = 0.f;
            if ((be & 31) == l)
            {
                #pragma unroll
                for (int i = 0; i < RW; ++i) if (i == (be >> 5)) sv = s[i];
                taken |= 1u << (be >> 5);
                my_k = -INFINITY;
                my_e = 0x7fffffff;
                #pragma unroll
                for (int i = 0; i < RW; ++i)
                    if (!(taken >> i & 1) && key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }
            }
            picked[k] = __shfl_sync(0xffffffffu, sv, be & 31);
            pe[k] = be;
        }
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < KC; ++k) sum += picked[k];                  // selection order, fixed
        if (l < KC)
        {
            float wv = 0.f; int ev = 0;
            #pragma unroll
            for (int k = 0; k < KC; ++k) if (k == l) { wv = picked[k]; ev = pe[k]; }
            sel_e[m][l] = ev;
            sel_w[m][l] = wv / (sum + 1e-20f) * ROUTE_SCALE;
            atomicOr(&emask[ev], 1u << m);
        }
    }
    __syncthreads();
    int n = 0, nt = 0, incl = 0;
    unsigned wm = 0u;
    if (t < NE)
    {
        n = __popc(emask[t]);
        nt = mt_ntiles(n);
        wm = __ballot_sync(0xffffffffu, n != 0);        // warps 0..RW-1 are full and take this branch together
        incl = nt;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            const int v = __shfl_up_sync(0xffffffffu, incl, o);
            if (l >= o) incl += v;
        }
        if (l == 31) wtiles[w] = incl;
        if (l == 0) wcount[w] = __popc(wm);
    }
    __syncthreads();
    if (t < NE)
    {
        int sb = 0, tb = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) if (i < w) { sb += wcount[i]; tb += wtiles[i]; }
        if (n)
        {
            const int slot = sb + __popc(wm & ((1u << l) - 1u)), tile0 = tb + incl - nt;
            tile0_of[t] = tile0;
            P->slot_expert[slot] = t;
            P->slot_tile0[slot] = tile0;
            P->slot_nt[slot] = nt;
            #pragma unroll
            for (int j = 0; j < MM / 8; ++j) if (j < nt) P->tile_mask[tile0 + j] = mt_tile_mask(n, j);
        }
    }
    if (t == 0)
    {
        int ns = 0, nts = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) { ns += wcount[i]; nts += wtiles[i]; }
        P->n_slots = ns;
        P->n_tiles = nts;
    }
    __syncthreads();
    if (t < M * TK)
    {
        const int m = t / TK, k = t % TK;
        if constexpr (KC < TK) if (k >= KC) return;                     // dropped route: stays absent
        const int e = sel_e[m][k];
        int tile, col;
        mt_route_pos(tile0_of[e], emask[e], m, tile, col);
        P->tok_slot[m][k] = tile;
        P->tok_col[m][k] = col;
        P->tok_w[m][k] = sel_w[m][k];
        P->tok_expert[m][k] = e;
        P->tile_w[tile][col] = sel_w[m][k];
        P->tile_tok[tile][col] = m;
    }
}

// ------------------------------------------------------------------------------------------------ W-row prefill plan, M <= MM <= 256
// k_plan_mt for M > 32 (a separate kernel: k_plan_mt and its M <= 32 SASS are untouched). Differences: the expert token mask is MW = MM / 32 words
// (emask[expert][word m >> 5], bit m & 31); the 32 warps score the tokens warp w, w + 32, ... (each token's picks are k_plan_mt's, instruction for
// instruction); an expert's tokens are cut into chunks of 32 ranks and every chunk is one SLOT (ds_plan_mt.h): the slot scan runs over ceil(n / 32)
// per expert, the tile scan over ceil(n / 8) as before, so an expert's tiles are laid out exactly as in the unchunked plan. Route threads loop
// (M * TK can exceed the 1024 threads). Output = PlanMT<TK, MM>, every field written (plan_ref_chunked is the specification). Plain launch only.
template <int NE = NEXP, int TK = TOPK, int MM = 128, int KC = TK>
__global__ void __launch_bounds__(PLAN_THREADS) k_plan_pw(const float* __restrict__ logits, const float* __restrict__ bias, int M, PlanMT<TK, MM>* __restrict__ P)
{
    constexpr int RW = NE / 32, SM = PlanMT<TK, MM>::SM, MW = MM / 32;
    static_assert(NE % 32 == 0 && NE <= PLAN_THREADS && MM % 32 == 0 && MM <= PW_MAX && TK <= 8 && TK <= NE, "k_plan_pw tiling");
    static_assert(KC >= 1 && KC <= TK, "k_plan_pw: kept routes");
    __shared__ unsigned emask[NE][MW];            // bit (m & 31) of word m >> 5 = token m selected the expert
    __shared__ int tile0_of[NE];
    __shared__ int sel_e[MM][TK];
    __shared__ float sel_w[MM][TK];
    __shared__ int wcount[RW], wtiles[RW];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    for (int i = t; i < NE * MW; i += PLAN_THREADS) (&emask[0][0])[i] = 0u;
    for (int i = t; i < SM; i += PLAN_THREADS) { P->slot_expert[i] = -1; P->slot_tile0[i] = 0; P->slot_nt[i] = 0; P->tile_mask[i] = 0; }
    for (int i = t; i < SM * 8; i += PLAN_THREADS) { (&P->tile_w[0][0])[i] = 0.f; (&P->tile_tok[0][0])[i] = -1; }
    for (int i = t; i < MM * 8; i += PLAN_THREADS)
    {
        (&P->tok_slot[0][0])[i] = -1; (&P->tok_col[0][0])[i] = 0; (&P->tok_w[0][0])[i] = 0.f; (&P->tok_expert[0][0])[i] = -1;
    }
    __syncthreads();
    #pragma unroll 1
    for (int m = w; m < M; m += PLAN_THREADS / 32)
    {
        float s[RW], key[RW];
        #pragma unroll
        for (int i = 0; i < RW; ++i)
        {
            const float sc = sqrt_softplus(logits[m * NE + l + 32 * i]);
            s[i] = sc;
            key[i] = sc + bias[l + 32 * i];
        }
        unsigned taken = 0;
        float my_k = key[0];
        int my_e = l;
        #pragma unroll
        for (int i = 1; i < RW; ++i)
            if (key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }       // strict: the lowest id wins a tie on this lane
        float picked[KC];
        int pe[KC];
        #pragma unroll
        for (int k = 0; k < KC; ++k)
        {
            const unsigned kb = key_bits(my_k);
            const unsigned best = __reduce_max_sync(0xffffffffu, kb);
            const int be = static_cast<int>(__reduce_min_sync(0xffffffffu, kb == best ? static_cast<unsigned>(my_e) : 0xffffffffu));
            float sv = 0.f;
            if ((be & 31) == l)
            {
                #pragma unroll
                for (int i = 0; i < RW; ++i) if (i == (be >> 5)) sv = s[i];
                taken |= 1u << (be >> 5);
                my_k = -INFINITY;
                my_e = 0x7fffffff;
                #pragma unroll
                for (int i = 0; i < RW; ++i)
                    if (!(taken >> i & 1) && key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }
            }
            picked[k] = __shfl_sync(0xffffffffu, sv, be & 31);
            pe[k] = be;
        }
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < KC; ++k) sum += picked[k];                  // selection order, fixed
        if (l < KC)
        {
            float wv = 0.f; int ev = 0;
            #pragma unroll
            for (int k = 0; k < KC; ++k) if (k == l) { wv = picked[k]; ev = pe[k]; }
            sel_e[m][l] = ev;
            sel_w[m][l] = wv / (sum + 1e-20f) * ROUTE_SCALE;
            atomicOr(&emask[ev][m >> 5], 1u << (m & 31));
        }
    }
    __syncthreads();
    int n = 0, nch = 0, nt = 0, in_s = 0, in_t = 0;
    if (t < NE)
    {
        #pragma unroll
        for (int i = 0; i < MW; ++i) n += __popc(emask[t][i]);
        nch = mt_nchunks(n);
        nt = mt_ntiles(n);
        in_s = nch;
        in_t = nt;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)                                // warp scans (NE % 32 == 0: every warp here is full)
        {
            const int vs = __shfl_up_sync(0xffffffffu, in_s, o), vt = __shfl_up_sync(0xffffffffu, in_t, o);
            if (l >= o) { in_s += vs; in_t += vt; }
        }
        if (l == 31) { wcount[w] = in_s; wtiles[w] = in_t; }
    }
    __syncthreads();
    if (t < NE)
    {
        int sb = 0, tb = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) if (i < w) { sb += wcount[i]; tb += wtiles[i]; }
        if (n)
        {
            const int slot0 = sb + in_s - nch, tile0 = tb + in_t - nt;
            tile0_of[t] = tile0;
            for (int c = 0; c < nch; ++c)
            {
                P->slot_expert[slot0 + c] = t;
                P->slot_tile0[slot0 + c] = tile0 + 4 * c;
                P->slot_nt[slot0 + c] = min(4, nt - 4 * c);
            }
            for (int j = 0; j < nt; ++j) P->tile_mask[tile0 + j] = mt_tile_mask(n, j);
        }
    }
    if (t == 0)
    {
        int ns = 0, nts = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) { ns += wcount[i]; nts += wtiles[i]; }
        P->n_slots = ns;
        P->n_tiles = nts;
    }
    __syncthreads();
    for (int r = t; r < M * TK; r += PLAN_THREADS)
    {
        const int m = r / TK, k = r % TK;
        if constexpr (KC < TK) if (k >= KC) continue;                   // dropped route: stays absent
        const int e = sel_e[m][k];
        int tile, col;
        mt_route_pos_w(tile0_of[e], emask[e], m, tile, col);
        P->tok_slot[m][k] = tile;
        P->tok_col[m][k] = col;
        P->tok_w[m][k] = sel_w[m][k];
        P->tok_expert[m][k] = e;
        P->tile_w[tile][col] = sel_w[m][k];
        P->tile_tok[tile][col] = m;
    }
}
#endif   // __CUDACC__
}   // namespace dsr
