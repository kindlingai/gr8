// g6 glue kernels: embedding, all-reduce consume + residual + RMSNorm, router top-8 -> K1 plan,
// vocab-split head (local argmax / LSE / Gumbel sample + merge), RoPE, fabric publish/wait helpers.
// Conventions (RUNTIME.md, SPEC §9): partials BF16 in fabric slots, residual FP32, activations FP16.
// Every rank runs these on identical bits with fixed reduction orders, so all decisions are bit-identical.
// GB10 rule (measured, results/m2/RESULTS-M2b.md): host-pinned memory is slow to touch from the GPU. Control words
// (step counter, tokens, sampling rows) live in DEVICE memory; only fabric slots, tails, doorbells and the error
// word are pinned. k_step_begin copies the host step counter once per graph launch; k_wait_peers (one CTA) is the
// only spinner on tails.
#pragma once
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "common.cuh"
#include "k1args.cuh"
#include "topk.cuh"
#include "../comms/fabric.h"

#ifndef G6_SPIN_TIMEOUT_CYC
#define G6_SPIN_TIMEOUT_CYC 20000000000LL   // ~10 s at GB10 clocks; tests override
#endif

namespace glue {

namespace cg = cooperative_groups;

constexpr int VOCAB = 154880;
constexpr int NEXP = 256, TOPK = 8, MMAX = G6_MX;   // MMAX: decode rows per step at most (8, or 32 under DEC_S2)
constexpr float RMS_EPS = 1e-5f;
constexpr float ROUTE_SCALE = 2.5f;
constexpr double ROPE_THETA = 8000000.0;
constexpr int ROPE_DIM = 64;

// error word codes (high byte), low bits = (collective << 8) | peer
constexpr unsigned ERR_AR = 0x61000000u, ERR_HEAD = 0x62000000u;

// ---------------------------------------------------------------- fabric helpers

__device__ __forceinline__ u64 ld_acquire_sys(const u64* p)
{
    u64 v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void st_release_sys(u64* p, u64 v)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ char* send_slot(const g6::FabricDev& f, u64 step, int c)
{
    return f.send + ((size_t)(step & 1) * f.ncoll + c) * f.stride;
}
// rank p's contribution to collective c: our own send slot for p == rank, else the recv slot p wrote into
__device__ __forceinline__ const char* peer_slot(const g6::FabricDev& f, u64 step, int c, int p)
{
    return p == f.rank ? send_slot(f, step, c)
                       : f.recv + (((size_t)(step & 1) * f.ncoll + c) * f.nr + p) * f.stride;
}

// All threads of every CTA call this after the CTA's last store into the send slot. The last CTA to arrive
// rings doorbell[c] = seq (release, system scope); the proxy then posts the RDMA writes. ticket: device int, 0.
__device__ __forceinline__ void publish(const g6::FabricDev& f, int c, u64 seq, unsigned* ticket)
{
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const unsigned n = gridDim.x * gridDim.y * gridDim.z;
        if (atomicAdd(ticket, 1u) == n - 1)
        {
            *ticket = 0;
            __threadfence_system();
            st_release_sys(reinterpret_cast<u64*>(f.doorbell) + c, seq);
        }
    }
}

#ifdef PF_RNPUB
// PF_RNPUB: publish once per cluster instead of once per CTA. Call from all threads of every CTA of a cluster kernel
// whose cluster ended with a cluster barrier (barrier.cluster arrive.release / wait.acquire) after its last store into
// the send slot (cluster_rms_store ends with cl.sync()). Thread 0 of cluster rank 0 then holds every store of the
// cluster in causality order; its system fence (cumulative) orders them before the ticket and the doorbell. Ticket
// counts clusters. Clusters must lie along x (gridDim.x / cluster size clusters per row of the grid).
__device__ __forceinline__ void publish_cluster(const g6::FabricDev& f, int c, u64 seq, unsigned* ticket)
{
    if (threadIdx.x == 0 && cg::this_cluster().block_rank() == 0)
    {
        __threadfence_system();
        const unsigned n = gridDim.x * gridDim.y * gridDim.z / cg::this_cluster().num_blocks();
        if (atomicAdd(ticket, 1u) == n - 1)
        {
            *ticket = 0;
            __threadfence_system();
            st_release_sys(reinterpret_cast<u64*>(f.doorbell) + c, seq);
        }
    }
}
#endif

// Threads 0..nr-1 each spin on one peer's tail (at byte offset `payload` of its slot); the CTA then proceeds.
// Timeout or an error already raised elsewhere -> error word, stop waiting (host aborts the step).
__device__ __forceinline__ void wait_peers(const g6::FabricDev& f, u64 step, int c, u64 seq, size_t payload,
                                           unsigned code)
{
    const int p = threadIdx.x;
    if (p < f.nr && p != f.rank)
    {
        const u64* tail = reinterpret_cast<const u64*>(peer_slot(f, step, c, p) + payload);
        const long long t0 = clock64();
        unsigned it = 0;
        while (ld_acquire_sys(tail) != seq)
        {
            if ((++it & 1023u) == 0 &&
                (*reinterpret_cast<volatile unsigned*>(f.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
            {
                atomicCAS(f.error, 0u, code | ((unsigned)c << 8) | (unsigned)p);
                break;
            }
        }
    }
    __syncthreads();
}

// first node of a step graph: host-pinned step counter -> device copy that every other kernel reads
__global__ void k_step_begin(const u64* host_step, u64* dev_step)
{
    if (threadIdx.x == 0) *dev_step = *reinterpret_cast<const volatile u64*>(host_step);
}

// copy n 32-bit words between pinned and device memory with one CTA (control data in / results out). block 32.
__global__ void __launch_bounds__(32) k_copy_words(const uint32_t* src, uint32_t* dst, int n)
{
    for (int i = threadIdx.x; i < n; i += 32) dst[i] = __ldcv(src + i);
}

// grid 1, block 32: wait until all peers' slots for collective c of this step have landed.
__global__ void __launch_bounds__(32) k_wait_peers(g6::FabricDev f, const u64* stepp, int c, size_t payload,
                                                   unsigned code)
{
    const u64 step = *stepp;
    wait_peers(f, step, c, g6::coll_seq(step, c, f.ncoll), payload, code);
}

// ---------------------------------------------------------------- RMSNorm over a 6144 row split across a cluster

constexpr int RN_CL = 8;                 // CTAs (cluster size) per row
constexpr int RN_COLS = HID / RN_CL;     // 768 columns per CTA
constexpr int RN_T = RN_COLS / 8;        // 96 threads, 8 consecutive columns each

__device__ __forceinline__ void bf16x8_to_f(const uint4& v, float (&o)[8])
{
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __bfloat162float(b[k]);
}

// x: this thread's 8 values of row m (columns col0..col0+7). Writes fp16(x * rsqrt(mean(x^2)+eps) * w).
// Fixed reduction order: in-thread 0..7, warp xor-tree, warps 0..2, cluster ranks 0..7.
__device__ __forceinline__ void cluster_rms_store(const float (&x)[8], const __nv_bfloat16* w, float eps,
                                                  __half* out_row, int col0, __half* out2_row = nullptr
#ifdef DEC_SLOT
                                                  , uint4* o_ret = nullptr
#endif
                                                  )
{
    __shared__ float wsum[RN_T / 32];
    __shared__ float cta_ss;
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) ss = fmaf(x[k], x[k], ss);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.f;
        #pragma unroll
        for (int i = 0; i < RN_T / 32; ++i) t += wsum[i];
        cta_ss = t;
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    float tot = 0.f;
    #pragma unroll
    for (int r = 0; r < RN_CL; ++r) tot += *cl.map_shared_rank(&cta_ss, r);
    const float inv = rsqrtf(tot * (1.f / HID) + eps);
    uint4 wv = *reinterpret_cast<const uint4*>(w + col0);
    float wf[8];
    bf16x8_to_f(wv, wf);
    uint4 o;
    o.x = pack_h2(x[0] * inv * wf[0], x[1] * inv * wf[1]);
    o.y = pack_h2(x[2] * inv * wf[2], x[3] * inv * wf[3]);
    o.z = pack_h2(x[4] * inv * wf[4], x[5] * inv * wf[5]);
    o.w = pack_h2(x[6] * inv * wf[6], x[7] * inv * wf[7]);
    *reinterpret_cast<uint4*>(out_row + col0) = o;
    if (out2_row) *reinterpret_cast<uint4*>(out2_row + col0) = o;      // same bytes (prefill RS: the all-gather slot)
#ifdef DEC_SLOT
    if (o_ret) *o_ret = o;
#endif
    cl.sync();   // peers may still be reading our cta_ss
}

// ---------------------------------------------------------------- embedding (+ first input RMSNorm)

struct EmbedArgs
{
    const int* tok;               // [M] token ids, device memory (written by the previous step's sampler)
    const __nv_bfloat16* emb;     // [VOCAB][6144]
    const __nv_bfloat16* w;       // [6144] layer-0 input_layernorm weight
    float* resid;                 // [M][6144]
    __half* out;                  // [M][6144]
    float eps;
};

// grid (RN_CL, M), block RN_T
__global__ void __cluster_dims__(RN_CL, 1, 1) __launch_bounds__(RN_T) k_embed_norm(EmbedArgs a)
{
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * RN_COLS + threadIdx.x * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(a.emb + (size_t)a.tok[m] * HID + col0);
    float x[8];
    bf16x8_to_f(v, x);
    float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
    r[0] = make_float4(x[0], x[1], x[2], x[3]);
    r[1] = make_float4(x[4], x[5], x[6], x[7]);
    cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0);
}

// ---------------------------------------------------------------- all-reduce consume + residual + RMSNorm

struct ArNormArgs
{
    g6::FabricDev f;
    const u64* stepp;             // DEVICE-memory step counter (k_step_begin refreshes it)
    int c;                        // collective index within the step
    int M;
    float* resid;                 // [M][6144] fp32, updated in place: resid += sum_p partial_p (rank order)
    const __nv_bfloat16* w;       // [6144] next norm weight
    __half* out;                  // [M][6144]
    float eps;
#ifdef DEC_SLOT
    u32* xf16;                    // nullptr, or the mma B fragment table of out (bind.cuh Xfrag layout, rows >= M zero)
    u32* xbf16;                   //   the same values rounded to BF16 (nullptr: not wanted)
#endif
};

#ifdef DEC_SLOT
__device__ __forceinline__ u32 h2w_to_bf2(u32 w)
{
    const float2 f = __half22float2(as_h2(w));
    return pack_bf2(f.x, f.y);
}
// This thread's 8 normed halves o (columns col0..col0+7, col0 % 8 == 0) of row m -> fragment table words
// [((ks * 32 + m * 4 + c) * 2 + r)], ks = col0 / 16, r = col0 / 8 % 2: threads 2j (r = 0) and 2j+1 (r = 1) hold the two
// halves of one k16 step and swap words, then store the two 16-byte quads k_xfrag stores. Rows 8 t .. 8 t + 7 are the
// tile t table (KSH * 16 uint4 after tile t - 1's). The CTAs of row M-1 also write the rest of its tile (rows M..8 t + 7)
// as zero. All threads of the warp call it.
__device__ __forceinline__ void frag_store(const uint4& o, int col0, int m, int M, u32* f16, u32* bf16)
{
    const bool odd = threadIdx.x & 1;
    const u32 pa = __shfl_xor_sync(0xffffffffu, odd ? o.x : o.z, 1);
    const u32 pb = __shfl_xor_sync(0xffffffffu, odd ? o.y : o.w, 1);
    const uint4 q = odd ? make_uint4(pa, o.z, pb, o.w) : make_uint4(o.x, pa, o.y, pb);
    const int tile = m >> 3, mr = m & 7;
    const size_t i = (size_t)tile * KSH * 16 + (size_t)(col0 >> 4) * 16 + odd;   // uint4 index of (tile, ks, row 0, this half)
    reinterpret_cast<uint4*>(f16)[i + mr * 2] = q;
    if (bf16)
        reinterpret_cast<uint4*>(bf16)[i + mr * 2] = make_uint4(h2w_to_bf2(q.x), h2w_to_bf2(q.y), h2w_to_bf2(q.z), h2w_to_bf2(q.w));
    if (m == M - 1)
        for (int z = mr + 1; z < 8; ++z)
        {
            reinterpret_cast<uint4*>(f16)[i + z * 2] = make_uint4(0u, 0u, 0u, 0u);
            if (bf16) reinterpret_cast<uint4*>(bf16)[i + z * 2] = make_uint4(0u, 0u, 0u, 0u);
        }
}
#endif

// grid (RN_CL, M), block RN_T. Slot payload = M x 6144 BF16 row-major. Launch k_wait_peers(c, M*6144*2) first.
__global__ void __cluster_dims__(RN_CL, 1, 1) __launch_bounds__(RN_T) k_ar_add_norm(ArNormArgs a)
{
    const u64 step = *a.stepp;
    const int m = blockIdx.y;
    const int col0 = blockIdx.x * RN_COLS + threadIdx.x * 8;
    const size_t off = ((size_t)m * HID + col0) * 2;
    float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
    if (a.f.nr == g6nr::NR)
    {   // every peer load in flight before the adds (one pinned-memory latency instead of NR); adds in rank order
        uint4 v[g6nr::NR];
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p) v[p] = __ldcv(reinterpret_cast<const uint4*>(peer_slot(a.f, step, a.c, p) + off));
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p)
        {
            float b[8];
            bf16x8_to_f(v[p], b);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x[k] += b[k];
        }
    }
    else
        for (int p = 0; p < a.f.nr; ++p)
        {
            const uint4 v = __ldcv(reinterpret_cast<const uint4*>(peer_slot(a.f, step, a.c, p) + off));
            float b[8];
            bf16x8_to_f(v, b);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x[k] += b[k];
        }
    const float4 r0 = r[0], r1 = r[1];
    x[0] += r0.x; x[1] += r0.y; x[2] += r0.z; x[3] += r0.w;
    x[4] += r1.x; x[5] += r1.y; x[6] += r1.z; x[7] += r1.w;
    r[0] = make_float4(x[0], x[1], x[2], x[3]);
    r[1] = make_float4(x[4], x[5], x[6], x[7]);
#ifdef DEC_SLOT
    uint4 o;
    cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0, nullptr, &o);
    if (a.xf16) frag_store(o, col0, m, a.M, a.xf16, a.xbf16);
#else
    cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0);
#endif
}

// ---------------------------------------------------------------- small-width RMSNorm (q_a 2048, kv latent 512)

// grid M, block 256. in row m at in + m*ld_in, width <= 4096. fp32 math, fixed order.
template <typename TIn>
__global__ void __launch_bounds__(256) k_rmsnorm_rows(const TIn* in, int ld_in, int width, const __nv_bfloat16* w,
                                                      float eps, __half* out, int ld_out)
{
    __shared__ float wsum[8];
    const TIn* row = in + (size_t)blockIdx.x * ld_in;
    float ss = 0.f;
    for (int i = threadIdx.x; i < width; i += 256)
    {
        const float v = (float)row[i];
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += wsum[i];
    const float inv = rsqrtf(tot / width + eps);
    for (int i = threadIdx.x; i < width; i += 256)
        out[(size_t)blockIdx.x * ld_out + i] = __float2half_rn((float)row[i] * inv * __bfloat162float(w[i]));
}

// ---------------------------------------------------------------- router: sigmoid top-8 -> K1 plan

// One plan per MoE layer call; K1Args points into it with n_plans = 1 (ctl stays 0).
// A slot is one expert's n=8 tile: at most 8 tokens, one per column. M <= 8: one slot per selected expert, column c = token c
// (slot_mask = the token mask). M > 8: an expert with more than 8 tokens takes ceil(n/8) consecutive slots, its tokens in
// ascending order packed 8 per slot, columns 0.. in that order (slot_tok maps column -> token). An mma column depends only
// on its own token, so where a token sits in a slot does not change its bits.
struct RoutePlan
{
    int n_slots;                  // slots this step (<= M*8; M <= 8: = distinct experts; above: an expert with >8 tokens counts ceil(n/8) times)
    int pad_[3];
    int slot_expert[SMAX];        // ascending expert id (an expert with several slots: consecutive)
    int slot_mask[SMAX];          // bit c = column c of the slot is used (M <= 8: bit m = token m routed to this slot)
    float slot_w[SMAX][8];        // routing weight of the token in column c of the slot (0 if unused)
    int slot_tok[SMAX][8];        // token in column c of the slot, -1 = unused
    int tok_slot[MMAX][8];        // slot of route k of token m, -1 for m >= M
    int tok_col[MMAX][8];         // column of token m in that slot
    float tok_w[MMAX][8];
    int tok_expert[MMAX][8];      // selected expert ids in selection order (checks / debug)
};

// grid 1, block 256 (warp w = tokens w, w + 8, ...; M <= MMAX). logits [M][256] fp32, bias [256] fp32 (e_score_correction_bias).
// Selection key = sigmoid(logit) + bias, ties -> lowest expert id. Weights = unbiased sigmoid of the
// selected experts / their sum * 2.5 (FACTS-M0 2(c)).
__global__ void __launch_bounds__(256) k_router_plan(const float* logits, const float* bias, int M, RoutePlan* P)
{
    __shared__ unsigned emask[NEXP];              // bit m = token m selected the expert
    __shared__ int sel_e[MMAX][TOPK];
    __shared__ float sel_w[MMAX][TOPK];
    __shared__ int slot_of[NEXP];                 // first slot of the expert
    __shared__ int wcount[8];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const bool dense = M > 8;
    emask[t] = 0u;
    for (int i = t; i < SMAX * 8; i += 256)
    {
        (&P->slot_w[0][0])[i] = 0.f;
        (&P->slot_tok[0][0])[i] = -1;
    }
    __syncthreads();
    for (int m = w; m < M; m += 8)
    {
        float s[8], key[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int e = l + 32 * i;
            s[i] = 1.f / (1.f + expf(-logits[m * NEXP + e]));
            key[i] = s[i] + bias[e];
        }
        unsigned taken = 0;
        for (int k = 0; k < TOPK; ++k)
        {
            float bk = -INFINITY;
            int be = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                if (!(taken >> i & 1) && key[i] > bk) { bk = key[i]; be = l + 32 * i; }   // i ascending = id ascending
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                if (ok > bk || (ok == bk && oe < be)) { bk = ok; be = oe; }
            }
            float sv = 0.f;
            if ((be & 31) == l) { taken |= 1u << (be >> 5); sv = s[be >> 5]; }
            sv = __shfl_sync(0xffffffffu, sv, be & 31);
            if (l == 0) { sel_e[m][k] = be; sel_w[m][k] = sv; }
        }
        __syncwarp();
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) sum += sel_w[m][k];
        __syncwarp();
        if (l < TOPK)
        {
            sel_w[m][l] = sel_w[m][l] / sum * ROUTE_SCALE;
            atomicOr(&emask[sel_e[m][l]], 1u << m);
        }
        __syncwarp();
    }
    __syncthreads();
    // thread t = expert t: cnt tokens -> ceil(cnt / 8) slots (one when M <= 8); first slot = exclusive prefix over experts
    const int cnt = __popc(emask[t]), nsl = (cnt + 7) >> 3;
    int scan = nsl;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        const int v = __shfl_up_sync(0xffffffffu, scan, o);
        if (l >= o) scan += v;
    }
    if (l == 31) wcount[w] = scan;
    __syncthreads();
    int base = 0;
    for (int i = 0; i < w; ++i) base += wcount[i];
    const int slot0 = base + scan - nsl;
    if (nsl)
    {
        slot_of[t] = slot0;
        for (int j = 0; j < nsl; ++j)
        {
            P->slot_expert[slot0 + j] = t;
            P->slot_mask[slot0 + j] = dense ? static_cast<int>((1u << (cnt - 8 * j < 8 ? cnt - 8 * j : 8)) - 1u) : static_cast<int>(emask[t]);
        }
    }
    if (t == 0)
    {
        int n = 0;
        for (int i = 0; i < 8; ++i) n += wcount[i];
        P->n_slots = n;
    }
    __syncthreads();
    for (int i = t; i < MMAX * TOPK; i += 256)
    {
        const int m = i >> 3, k = i & 7;
        if (m < M)
        {
            const int e = sel_e[m][k];
            const int r = __popc(emask[e] & ((1u << m) - 1u));      // rank of token m among the expert's tokens
            const int s2 = slot_of[e] + (dense ? r >> 3 : 0), col = dense ? (r & 7) : m;
            P->tok_slot[m][k] = s2;
            P->tok_col[m][k] = col;
            P->tok_w[m][k] = sel_w[m][k];
            P->tok_expert[m][k] = e;
            P->slot_w[s2][col] = sel_w[m][k];
            P->slot_tok[s2][col] = m;
        }
        else
        {
            P->tok_slot[m][k] = -1;
            P->tok_col[m][k] = 0;
            P->tok_w[m][k] = 0.f;
            P->tok_expert[m][k] = -1;
        }
    }
}

// ---------------------------------------------------------------- sampling: Philox4x32-10 + Gumbel-max

__host__ __device__ __forceinline__ uint32_t mulhi32(uint32_t a, uint32_t b)
{
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return (uint32_t)(((uint64_t)a * b) >> 32);
#endif
}

__host__ __device__ __forceinline__ uint32_t philox_x(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                                      uint32_t k0, uint32_t k1)
{
    #pragma unroll
    for (int r = 0; r < 10; ++r)
    {
        const uint32_t lo0 = c0 * 0xD2511F53u, hi0 = mulhi32(c0, 0xD2511F53u);
        const uint32_t lo1 = c2 * 0xCD9E8D57u, hi1 = mulhi32(c2, 0xCD9E8D57u);
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return c0;
}

// Per-row sampling state (fixed address, written by the control thread identically on every rank).
struct SampleRow
{
    float inv_temp;               // 1/T; 0 = greedy
    uint32_t seed_lo, seed_hi;    // request seed (rank 0 draws one at admission if the client gave none)
    int pos;                      // absolute position of the token being sampled
    int draw;                     // stream index (0 = main sample; MTP rejection draws use others)
    float top_p;                  // (0,1): sample from the nucleus of softmax(logit / T) (needs HeadArgs::nc > 0);
                                  // else the full vocabulary. Greedy rows ignore it.
    int topk;                     // > 0: the row also returns its top LP_K tokens and the ranks (HeadArgs::tk; topk.cuh)
    int nodraft;                  // rejection sampling: 1 = row m + 1 holds no draft for this row (S2: a session's last row
                                  // in a step that carries several sessions)
};
__host__ __device__ __forceinline__ bool topp_on(const SampleRow& s) { return s.inv_temp > 0.f && s.top_p > 0.f && s.top_p < 1.f; }

// Gumbel noise for vocab id v: u in (0,1) from the top 24 bits; g = -log(-log(u)).
__host__ __device__ __forceinline__ float gumbel(const SampleRow& s, int v)
{
    const uint32_t x = philox_x((uint32_t)v, (uint32_t)s.pos, (uint32_t)s.draw, 0x67364c75u, s.seed_lo, s.seed_hi);
    const float u = ((float)(x >> 8) + 0.5f) * (1.f / 16777216.f);
    return -logf(-logf(u));
}
__host__ __device__ __forceinline__ float sample_score(const SampleRow& s, float logit, int v)
{
    return s.inv_temp == 0.f ? logit : logit * s.inv_temp + gumbel(s, v);
}
// Rejection sampling (--spec-accept rejection): the uniform draw that accepts a draft at the row's position, from a Philox
// stream of its own (another counter word than gumbel), so it is independent of the Gumbel noise of every token.
__host__ __device__ __forceinline__ float uniform01(const SampleRow& s)
{
    const uint32_t x = philox_x(0u, (uint32_t)s.pos, (uint32_t)s.draw, 0x3c6ef372u, s.seed_lo, s.seed_hi);
    return ((float)(x >> 8) + 0.5f) * (1.f / 16777216.f);
}

// Per-row record each rank contributes to the head all-gather (32 bytes).
struct HeadRec
{
    float score;                  // best sample_score in this rank's vocab slice
    int id;                       // its global vocab id (ties -> lowest id)
    float logit;                  // its raw logit
    float lmax;                   // max raw logit in the slice
    float lsum;                   // sum exp(logit - lmax) over the slice (raw logits, T = 1: API logprobs)
    float pad_[3];                // pad_[0]: raw logit of the row's target id if it is in this slice, else -inf
};                                //          (only with HeadArgs::tgt; 0 otherwise)
static_assert(sizeof(HeadRec) == 32, "HeadRec is 32 bytes on the wire");
// top_p rows (HeadArgs::nc > 0): pad_[1] = sum exp((logit - lmax) / T) over the slice, pad_[2] = the largest raw logit
// of the slice that is NOT among its candidates. After the M records: [M][nc] candidates, the slice's highest raw
// logits (unused entries: id -1). Every token of the whole vocabulary with logit > max over ranks of pad_[2] is a candidate.
struct HeadCand
{
    float logit;
    int id;
};
constexpr int TP_BINS = 2048;             // local candidate cut: histogram of (lmax - logit), TP_RANGE raw logit units
constexpr float TP_RANGE = 32.f;
constexpr int TP_SORT = 2048;             // merge sort size: nr * nc must fit
__host__ __device__ __forceinline__ size_t head_bytes(int M, int nc) { return (size_t)M * (sizeof(HeadRec) + (size_t)nc * sizeof(HeadCand)); }
__device__ __forceinline__ int tp_bin(float d) { const float b = d * (TP_BINS / TP_RANGE); return b < (float)(TP_BINS - 1) ? (int)b : TP_BINS - 1; }

struct HeadArgs
{
    g6::FabricDev f;
    const u64* stepp;
    int c;                        // collective index of the head all-gather
    int M;
    const float* logits;          // [M][ldl] this rank's vocab slice, fp32
    int ldl, vlo, vn;             // row stride, first global vocab id, slice width
    const SampleRow* rows;        // [M] device memory
    unsigned* ticket;             // publish ticket (device int, 0)
    int* tok_out;                 // [M] sampled ids (feeds next step's embedding)
    float* lp_out;                // [M] logprob of the sampled id under the raw distribution
    const int* tgt = nullptr;     // optional [M] target ids (teacher forcing; < 0 = none), device memory
    float* lpt_out = nullptr;     // optional [M] logprob of the target id under the raw distribution (-inf if none)
    int nc = 0;                   // top_p candidates per rank per row (0 = top_p off; merge with k_head_merge_tp)
    unsigned* topp_miss = nullptr;// optional device counter: top_p rows whose nucleus reached past the candidates
    TkRes* tk = nullptr;          // optional: top lists of the rows with SampleRow::topk > 0 (the slot then carries them, see below)
};
// Rows with SampleRow::topk > 0 (and HeadArgs::tk set): every rank also sends, after the top_p candidates, M lists of
// LP_K TkEnt (its slice's best, sorted by tk_before); the slot payload grows by M * LP_K * 8 bytes whenever any row of
// the step asks (head_bytes_tk), which the host passes to the fabric and the merge kernels re-derive from the rows.
__host__ __device__ __forceinline__ size_t head_bytes_tk(int M, int nc, bool any_tk)
{
    return head_bytes(M, nc) + (any_tk ? (size_t)M * LP_K * sizeof(TkEnt) : 0);
}
__device__ __forceinline__ size_t head_wait_bytes(const HeadArgs& a)
{
    bool any = false;
    if (a.tk)
        for (int i = 0; i < a.M; ++i) any |= a.rows[i].topk > 0;
    return head_bytes_tk(a.M, a.nc, any);
}
// peer p's list of row m: entry i
__device__ __forceinline__ TkEnt head_tk_peer(const HeadArgs& a, u64 step, int p, int m, int i)
{
    const float2 v = __ldcv(reinterpret_cast<const float2*>(peer_slot(a.f, step, a.c, p) + head_bytes(a.M, a.nc)) + (size_t)m * LP_K + i);
    return TkEnt{v.x, __float_as_int(v.y)};
}

// top_p part of k_head_local (block 1024, after the record is stored; s_mx = the 32 warp maxima): the slice's
// highest raw logits (at most nc, whole histogram bins of lmax - logit) -> candidates, pad_[1] / pad_[2] of the record.
__device__ __forceinline__ void head_local_topp(const HeadArgs& a, int m, const float* row, const SampleRow sr, const float* s_mx)
{
    __shared__ unsigned s_hist[TP_BINS];
    __shared__ float s_z[32], s_u[32];
    __shared__ int s_cut, s_cnt;
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    float lmx = -INFINITY;
    for (int i = 0; i < 32; ++i) lmx = fmaxf(lmx, s_mx[i]);
    for (int i = t; i < TP_BINS; i += 1024) s_hist[i] = 0u;
    if (t == 0) s_cnt = 0;
    __syncthreads();
    float z = 0.f;
    for (int i = t; i < a.vn; i += 1024)
    {
        const float lg = row[i];
        z += expf((lg - lmx) * sr.inv_temp);
        atomicAdd(&s_hist[tp_bin(lmx - lg)], 1u);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (l == 0) s_z[w] = z;
    __syncthreads();
    if (w == 0)                          // cut = last bin b with count(bins 0..b) <= nc (-1: bin 0 alone is larger)
    {
        constexpr int PER = TP_BINS / 32;
        unsigned c = 0;
        for (int i = 0; i < PER; ++i) c += s_hist[l * PER + i];
        unsigned inc = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const unsigned y = __shfl_up_sync(0xffffffffu, inc, o); if (l >= o) inc += y; }
        const unsigned exc = inc - c;
        const unsigned over = __ballot_sync(0xffffffffu, inc > (unsigned)a.nc);
        const int L = over ? __ffs(over) - 1 : 31;        // first lane whose bins pass nc
        if (l == L)
        {
            int cut = over ? L * PER - 1 : TP_BINS - 1;
            if (over)
            {
                unsigned cum = exc;
                for (int i = 0; i < PER; ++i)
                {
                    cum += s_hist[L * PER + i];
                    if (cum > (unsigned)a.nc) break;
                    cut = L * PER + i;
                }
            }
            s_cut = cut;
        }
    }
    __syncthreads();
    HeadCand* cand = reinterpret_cast<HeadCand*>(send_slot(a.f, *a.stepp, a.c) + (size_t)a.M * sizeof(HeadRec)) + (size_t)m * a.nc;
    const int cut = s_cut;
    float um = -INFINITY;                // largest logit not sent
    for (int i = t; i < a.vn; i += 1024)
    {
        const float lg = row[i];
        if (tp_bin(lmx - lg) <= cut) cand[atomicAdd(&s_cnt, 1)] = HeadCand{lg, a.vlo + i};
        else um = fmaxf(um, lg);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) um = fmaxf(um, __shfl_xor_sync(0xffffffffu, um, o));
    if (l == 0) s_u[w] = um;
    __syncthreads();
    for (int i = s_cnt + t; i < a.nc; i += 1024) cand[i] = HeadCand{-INFINITY, -1};
    if (t == 0)
    {
        float zs = 0.f, us = -INFINITY;
        for (int i = 0; i < 32; ++i) { zs += s_z[i]; us = fmaxf(us, s_u[i]); }
        HeadRec* r = reinterpret_cast<HeadRec*>(send_slot(a.f, *a.stepp, a.c)) + m;
        r->pad_[1] = zs;
        r->pad_[2] = us;
    }
}

// k_head_local for rows with topk (block 1024, uniform per block): the slice's LP_K best (logit, id) in tk_before order,
// by LP_K rounds of "best entry strictly after the previous pick" (exact, ties by id), into the send slot.
__device__ __forceinline__ void head_local_topk(const HeadArgs& a, int m, const float* row)
{
    __shared__ float s_bl[32], s_ol[LP_K];
    __shared__ int s_bi[32], s_oi[LP_K];
    __shared__ float s_pl;
    __shared__ int s_pi;
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    float pl = INFINITY;
    int pi = -1;
    for (int k = 0; k < LP_K; ++k)
    {
        float bl = -INFINITY;
        int bi = TK_PAD_ID;
        for (int i = t; i < a.vn; i += 1024)
        {
            const float lg = row[i];
            const int v = a.vlo + i;
            if (tk_before(pl, pi, lg, v) && tk_before(lg, v, bl, bi)) { bl = lg; bi = v; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ol = __shfl_xor_sync(0xffffffffu, bl, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (tk_before(ol, oi, bl, bi)) { bl = ol; bi = oi; }
        }
        if (l == 0) { s_bl[w] = bl; s_bi[w] = bi; }
        __syncthreads();
        if (w == 0)
        {
            bl = s_bl[l]; bi = s_bi[l];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ol = __shfl_xor_sync(0xffffffffu, bl, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (tk_before(ol, oi, bl, bi)) { bl = ol; bi = oi; }
            }
            if (l == 0) { s_ol[k] = bl; s_oi[k] = bi; s_pl = bl; s_pi = bi; }
        }
        __syncthreads();
        pl = s_pl; pi = s_pi;
    }
    if (t < LP_K)
    {
        float2 v;
        v.x = s_ol[t]; v.y = __int_as_float(s_oi[t]);
        reinterpret_cast<float2*>(send_slot(a.f, *a.stepp, a.c) + head_bytes(a.M, a.nc))[(size_t)m * LP_K + t] = v;
    }
}

// Rejection sampling (RS, --spec-accept rejection; MTP and DFlash2 verify steps): row m's draft is the step's input token
// of row m + 1 (the draft of the position row m samples), or -1 for the last row, greedy rows and RS off. The RS kernels
// are k_head_local_rs, k_head_merge_rs and k_head_merge_tp_rs; with RS off the head runs today's kernels, unchanged.
template <bool RS> __device__ __forceinline__ int rs_draft(const HeadArgs& a, const int* tok, int m)
{
    return RS && m + 1 < a.M && !a.rows[m].nodraft && a.rows[m].inv_temp > 0.f ? tok[m + 1] : -1;
}

// grid M, block 1024: local record -> send slot c, then publish.
__global__ void __launch_bounds__(1024) k_head_local(HeadArgs a)
{
    __shared__ float s_sc[32], s_lg[32], s_mx[32], s_sm[32];
    __shared__ int s_id[32];
    const int m = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    const SampleRow sr = a.rows[m];
    const float* row = a.logits + (size_t)m * a.ldl;
    float bs = -INFINITY, blg = 0.f, mx = -INFINITY, sm = 0.f;
    int bid = 0x7fffffff;
    for (int i = t; i < a.vn; i += 1024)
    {
        const float lg = row[i];
        const int v = a.vlo + i;
        const float sc = sample_score(sr, lg, v);
        if (sc > bs) { bs = sc; bid = v; blg = lg; }   // i ascending per thread -> lowest id on ties
        if (lg > mx) { sm = sm * expf(mx - lg) + 1.f; mx = lg; }
        else sm += expf(lg - mx);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        const float os = __shfl_xor_sync(0xffffffffu, bs, o), ol = __shfl_xor_sync(0xffffffffu, blg, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
        if (os > bs || (os == bs && oi < bid)) { bs = os; bid = oi; blg = ol; }
        const float omx = __shfl_xor_sync(0xffffffffu, mx, o), osm = __shfl_xor_sync(0xffffffffu, sm, o);
        const float nmx = fmaxf(mx, omx);
        sm = (mx == -INFINITY ? 0.f : sm * expf(mx - nmx)) + (omx == -INFINITY ? 0.f : osm * expf(omx - nmx));
        mx = nmx;
    }
    if (l == 0) { s_sc[w] = bs; s_id[w] = bid; s_lg[w] = blg; s_mx[w] = mx; s_sm[w] = sm; }
    __syncthreads();
    if (t == 0)
    {
        HeadRec r;
        r.score = s_sc[0]; r.id = s_id[0]; r.logit = s_lg[0]; r.lmax = s_mx[0]; r.lsum = s_sm[0];
        for (int i = 1; i < 32; ++i)
        {
            if (s_sc[i] > r.score || (s_sc[i] == r.score && s_id[i] < r.id)) { r.score = s_sc[i]; r.id = s_id[i]; r.logit = s_lg[i]; }
            const float nmx = fmaxf(r.lmax, s_mx[i]);
            r.lsum = (r.lmax == -INFINITY ? 0.f : r.lsum * expf(r.lmax - nmx)) +
                     (s_mx[i] == -INFINITY ? 0.f : s_sm[i] * expf(s_mx[i] - nmx));
            r.lmax = nmx;
        }
        r.pad_[0] = r.pad_[1] = r.pad_[2] = 0.f;
        if (a.tgt)
        {
            const int i = a.tgt[m] - a.vlo;
            r.pad_[0] = i >= 0 && i < a.vn ? row[i] : -INFINITY;
        }
        reinterpret_cast<HeadRec*>(send_slot(a.f, *a.stepp, a.c))[m] = r;
    }
    if (a.nc > 0 && topp_on(sr)) head_local_topp(a, m, row, sr, s_mx);
    if (a.tk && sr.topk > 0) head_local_topk(a, m, row);
    const u64 step = *a.stepp;
    publish(a.f, a.c, g6::coll_seq(step, a.c, a.f.ncoll), a.ticket);
}

// k_head_local for rejection sampling (launch_head_local with Model::rs). tok: the step's input tokens, device memory.
// Its own kernel, so that k_head_local stays as it is. Row m with a draft d (rs_draft): the record's best score skips d (the
// replacement when d is rejected), pad_[0] holds d's raw logit and pad_[1] the slice's sum of exp((logit - lmax) / T),
// the normalizer of p(d). Rows without a draft get k_head_local's record.
__global__ void __launch_bounds__(1024) k_head_local_rs(HeadArgs a, const int* tok)
{
    __shared__ float s_sc[32], s_lg[32], s_mx[32], s_sm[32], s_zt[32];
    __shared__ int s_id[32];
    const int m = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    const SampleRow sr = a.rows[m];
    const float* row = a.logits + (size_t)m * a.ldl;
    const int d = rs_draft<true>(a, tok, m);
    float bs = -INFINITY, blg = 0.f, mx = -INFINITY, sm = 0.f, zt = 0.f;
    int bid = 0x7fffffff;
    for (int i = t; i < a.vn; i += 1024)
    {
        const float lg = row[i];
        const int v = a.vlo + i;
        const float sc = sample_score(sr, lg, v);
        if (sc > bs && v != d) { bs = sc; bid = v; blg = lg; }   // i ascending per thread -> lowest id on ties
        if (lg > mx) { sm = sm * expf(mx - lg) + 1.f; zt = zt * expf((mx - lg) * sr.inv_temp) + 1.f; mx = lg; }
        else { sm += expf(lg - mx); zt += expf((lg - mx) * sr.inv_temp); }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        const float os = __shfl_xor_sync(0xffffffffu, bs, o), ol = __shfl_xor_sync(0xffffffffu, blg, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
        if (os > bs || (os == bs && oi < bid)) { bs = os; bid = oi; blg = ol; }
        const float omx = __shfl_xor_sync(0xffffffffu, mx, o), osm = __shfl_xor_sync(0xffffffffu, sm, o);
        const float ozt = __shfl_xor_sync(0xffffffffu, zt, o);
        const float nmx = fmaxf(mx, omx);
        sm = (mx == -INFINITY ? 0.f : sm * expf(mx - nmx)) + (omx == -INFINITY ? 0.f : osm * expf(omx - nmx));
        zt = (mx == -INFINITY ? 0.f : zt * expf((mx - nmx) * sr.inv_temp)) + (omx == -INFINITY ? 0.f : ozt * expf((omx - nmx) * sr.inv_temp));
        mx = nmx;
    }
    if (l == 0) { s_sc[w] = bs; s_id[w] = bid; s_lg[w] = blg; s_mx[w] = mx; s_sm[w] = sm; s_zt[w] = zt; }
    __syncthreads();
    if (t == 0)
    {
        HeadRec r;
        r.score = s_sc[0]; r.id = s_id[0]; r.logit = s_lg[0]; r.lmax = s_mx[0]; r.lsum = s_sm[0];
        float rz = s_zt[0];
        for (int i = 1; i < 32; ++i)
        {
            if (s_sc[i] > r.score || (s_sc[i] == r.score && s_id[i] < r.id)) { r.score = s_sc[i]; r.id = s_id[i]; r.logit = s_lg[i]; }
            const float nmx = fmaxf(r.lmax, s_mx[i]);
            r.lsum = (r.lmax == -INFINITY ? 0.f : r.lsum * expf(r.lmax - nmx)) +
                     (s_mx[i] == -INFINITY ? 0.f : s_sm[i] * expf(s_mx[i] - nmx));
            rz = (r.lmax == -INFINITY ? 0.f : rz * expf((r.lmax - nmx) * sr.inv_temp)) +
                 (s_mx[i] == -INFINITY ? 0.f : s_zt[i] * expf((s_mx[i] - nmx) * sr.inv_temp));
            r.lmax = nmx;
        }
        r.pad_[0] = r.pad_[1] = r.pad_[2] = 0.f;
        if (a.tgt || d >= 0)              // a row with a draft carries d's logit, also when it has a target (then the
        {                                 // target logprob of that row is d's; prompt rows' targets are d anyway)
            const int i = (d >= 0 ? d : a.tgt[m]) - a.vlo;
            r.pad_[0] = i >= 0 && i < a.vn ? row[i] : -INFINITY;
        }
        if (d >= 0) r.pad_[1] = rz;
        reinterpret_cast<HeadRec*>(send_slot(a.f, *a.stepp, a.c))[m] = r;
    }
    if (a.nc > 0 && topp_on(sr)) head_local_topp(a, m, row, sr, s_mx);
    if (a.tk && sr.topk > 0) head_local_topk(a, m, row);
    const u64 step = *a.stepp;
    publish(a.f, a.c, g6::coll_seq(step, a.c, a.f.ncoll), a.ticket);
}

// RS, row m with draft d (rs_draft): accept d with probability p(d) = exp((l_d - mx) / T) / z under the request's
// distribution (z over the nucleus for top_p rows), with the row's uniform draw; else the record's best, which skipped d.
// tok / logit: the row's output token and its raw logit, set to d and ld when d is kept.
__device__ __forceinline__ void rs_accept(const SampleRow& sr, int d, float pd, int& tok, float& logit, float ld)
{
    if (uniform01(sr) < pd) { tok = d; logit = ld; }
}

// grid 1, block 32: wait for the 5 peer records, merge in rank order, write the sampled token + logprob.
// Rows with SampleRow::topk > 0 (HeadArgs::tk): then the ranks' top lists -> the row's TkRes.
template <bool RS> __device__ __forceinline__ void head_merge(const HeadArgs& a, const int* tok)
{
    const u64 step = *a.stepp;
    const u64 seq = g6::coll_seq(step, a.c, a.f.ncoll);
    wait_peers(a.f, step, a.c, seq, head_wait_bytes(a), ERR_HEAD);
    const int m = threadIdx.x;
    float bs = -INFINITY, blg = 0.f, mx = -INFINITY, sm = 0.f, tl = -INFINITY;
    int bid = 0x7fffffff;
    if (m < a.M)
    {
        for (int p = 0; p < a.f.nr; ++p)
        {
            const float4* q = reinterpret_cast<const float4*>(peer_slot(a.f, step, a.c, p) + (size_t)m * sizeof(HeadRec));
            const float4 v = __ldcv(q);
            const float4 u = __ldcv(q + 1);
            HeadRec r;
            r.score = v.x; r.id = __float_as_int(v.y); r.logit = v.z; r.lmax = v.w; r.lsum = u.x;
            tl = fmaxf(tl, u.y);
            if (r.score > bs || (r.score == bs && r.id < bid)) { bs = r.score; bid = r.id; blg = r.logit; }
            const float nmx = fmaxf(mx, r.lmax);
            sm = (mx == -INFINITY ? 0.f : sm * expf(mx - nmx)) + (r.lmax == -INFINITY ? 0.f : r.lsum * expf(r.lmax - nmx));
            mx = nmx;
        }
        if constexpr (RS)
            if (const int d = rs_draft<RS>(a, tok, m); d >= 0)
            {
                const SampleRow sr = a.rows[m];
                float z = 0.f;
                for (int p = 0; p < a.f.nr; ++p)
                {
                    const float4* q = reinterpret_cast<const float4*>(peer_slot(a.f, step, a.c, p) + (size_t)m * sizeof(HeadRec));
                    z += __ldcv(q + 1).z * expf((__ldcv(q).w - mx) * sr.inv_temp);
                }
                rs_accept(sr, d, expf((tl - mx) * sr.inv_temp) / z, bid, blg, tl);
            }
        a.tok_out[m] = bid;
        a.lp_out[m] = blg - mx - logf(sm);
        if (a.lpt_out) a.lpt_out[m] = tl - mx - logf(sm);
    }
    if (a.tk)
    {
        bool any = false;
        for (int i = 0; i < a.M; ++i) any |= a.rows[i].topk > 0;
        if (!any) return;
        __shared__ TkEnt s_te[TK_ROWS][8 * LP_K];
        const int nl = a.f.nr * LP_K;
        for (int x = m; x < a.M * nl; x += 32)       // the lists of every asking row, loaded by the whole warp
        {
            const int r = x / nl, p = (x - r * nl) / LP_K;
            if (a.rows[r].topk > 0) s_te[r][x - r * nl] = head_tk_peer(a, step, p, r, x - r * nl - p * LP_K);
        }
        __syncwarp();
        if (m < a.M && a.rows[m].topk > 0)
        {
            TkEnt top[LP_K];
            tk_merge(a.f.nr, [&](int p, int i) { return s_te[m][p * LP_K + i]; }, top);
            tk_result(a.tk, m, top, mx, sm, blg, tl);
        }
    }
}

__global__ void __launch_bounds__(32) k_head_merge(HeadArgs a) { head_merge<false>(a, nullptr); }
__global__ void __launch_bounds__(32) k_head_merge_rs(HeadArgs a, const int* tok) { head_merge<true>(a, tok); }

// k_head_merge with top_p candidates (HeadArgs::nc > 0). grid M, block 1024: one row per CTA. Rows without top_p
// give k_head_merge's result. top_p rows: the candidates above every rank's unsent maximum, sorted (logit desc, id
// asc); keep while the mass before a token is < top_p (softmax at T over the whole vocabulary: the ranks' pad_[1]);
// sample = argmax over the kept of logit / T + gumbel. If the kept candidates never reach top_p, all of them are
// kept and topp_miss counts the row. Same bytes on every rank -> same token.
// RS rows with a draft d: the replacement sample skips d among the kept, and p(d) is d's share of the kept mass (0 when d
// is not kept); a row that keeps nothing falls back to the full-vocabulary p(d), as its sample does.
template <bool RS> __device__ __forceinline__ void head_merge_tp(const HeadArgs& a, const int* tok)
{
    __shared__ float s_l[TP_SORT], s_ws[32];
    __shared__ int s_id[TP_SORT];
    __shared__ float s_mx, s_sm, s_tl, s_blg, s_z, s_cut, s_best[32], s_bl[32], s_kz[32], s_ld;
    __shared__ int s_bid, s_bi[32], s_keep, s_in;
    const u64 step = *a.stepp;
    const u64 seq = g6::coll_seq(step, a.c, a.f.ncoll);
    wait_peers(a.f, step, a.c, seq, head_wait_bytes(a), ERR_HEAD);
    const int m = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    const SampleRow sr = a.rows[m];
    const bool on = topp_on(sr);
    const int d = rs_draft<RS>(a, tok, m);
    if (t == 0)
    {
        float bs = -INFINITY, blg = 0.f, mx = -INFINITY, sm = 0.f, tl = -INFINITY, cut = -INFINITY;
        int bid = 0x7fffffff;
        for (int p = 0; p < a.f.nr; ++p)
        {
            const float4* q = reinterpret_cast<const float4*>(peer_slot(a.f, step, a.c, p) + (size_t)m * sizeof(HeadRec));
            const float4 v = __ldcv(q), u = __ldcv(q + 1);
            HeadRec r;
            r.score = v.x; r.id = __float_as_int(v.y); r.logit = v.z; r.lmax = v.w; r.lsum = u.x;
            tl = fmaxf(tl, u.y);
            cut = fmaxf(cut, u.w);
            if (r.score > bs || (r.score == bs && r.id < bid)) { bs = r.score; bid = r.id; blg = r.logit; }
            const float nmx = fmaxf(mx, r.lmax);
            sm = (mx == -INFINITY ? 0.f : sm * expf(mx - nmx)) + (r.lmax == -INFINITY ? 0.f : r.lsum * expf(r.lmax - nmx));
            mx = nmx;
        }
        float z = 0.f;                   // sum exp((logit - mx) / T) over the whole vocabulary
        if (on || (RS && d >= 0))
            for (int p = 0; p < a.f.nr; ++p)
            {
                const float4* q = reinterpret_cast<const float4*>(peer_slot(a.f, step, a.c, p) + (size_t)m * sizeof(HeadRec));
                const float4 v = __ldcv(q), u = __ldcv(q + 1);
                z += u.z * expf((v.w - mx) * sr.inv_temp);
            }
        s_mx = mx; s_sm = sm; s_tl = tl; s_bid = bid; s_blg = blg; s_z = z; s_cut = cut;
    }
    __syncthreads();
    if (on)
    {
        const float cut = s_cut, mx = s_mx;
        const int n = a.f.nr * a.nc;
        for (int i = t; i < TP_SORT; i += 1024)
        {
            float lg = -INFINITY;
            int id = 0x7fffffff;
            if (i < n)
            {
                const int p = i / a.nc, k = i - p * a.nc;
                const HeadCand c = reinterpret_cast<const HeadCand*>(peer_slot(a.f, step, a.c, p) + (size_t)a.M * sizeof(HeadRec))[(size_t)m * a.nc + k];
                if (c.id >= 0 && c.logit > cut) { lg = c.logit; id = c.id; }
            }
            s_l[i] = lg; s_id[i] = id;
        }
        __syncthreads();
        for (int k = 2; k <= TP_SORT; k <<= 1)             // bitonic: (logit desc, id asc)
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                for (int i = t; i < TP_SORT; i += 1024)
                {
                    const int x = i ^ j;
                    if (x <= i) continue;
                    const bool b_first = s_l[x] > s_l[i] || (s_l[x] == s_l[i] && s_id[x] < s_id[i]);
                    const bool a_first = s_l[i] > s_l[x] || (s_l[i] == s_l[x] && s_id[i] < s_id[x]);
                    if ((i & k) == 0 ? b_first : a_first)
                    {
                        const float tl = s_l[i]; s_l[i] = s_l[x]; s_l[x] = tl;
                        const int ti = s_id[i]; s_id[i] = s_id[x]; s_id[x] = ti;
                    }
                }
                __syncthreads();
            }
        // exclusive prefix of the mass, two entries per thread (2t, 2t+1)
        const float w0 = s_l[2 * t] == -INFINITY ? 0.f : expf((s_l[2 * t] - mx) * sr.inv_temp);
        const float w1 = s_l[2 * t + 1] == -INFINITY ? 0.f : expf((s_l[2 * t + 1] - mx) * sr.inv_temp);
        float inc = w0 + w1;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const float y = __shfl_up_sync(0xffffffffu, inc, o); if (l >= o) inc += y; }
        if (l == 31) s_ws[w] = inc;
        if (t == 0) s_keep = 0;
        __syncthreads();
        float base = 0.f;
        for (int i = 0; i < w; ++i) base += s_ws[i];
        const float pz = sr.top_p * s_z;
        const float e0 = base + inc - (w0 + w1), e1 = e0 + w0;       // mass before entries 2t, 2t+1
        const int k0 = s_l[2 * t] != -INFINITY && e0 < pz, k1 = s_l[2 * t + 1] != -INFINITY && e1 < pz;
        if (k0 + k1) atomicAdd(&s_keep, k0 + k1);
        if (RS && t == 0) s_in = 0;
        __syncthreads();
        const int keep = s_keep;
        if (RS && d >= 0)                // the kept mass and d's logit if d is kept
        {
            float kz = (2 * t < keep ? w0 : 0.f) + (2 * t + 1 < keep ? w1 : 0.f);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) kz += __shfl_xor_sync(0xffffffffu, kz, o);
            if (l == 0) s_kz[w] = kz;
            for (int i = 2 * t; i < 2 * t + 2 && i < keep; ++i)
                if (s_id[i] == d) { s_in = 1; s_ld = s_l[i]; }
            __syncthreads();
        }
        float total = 0.f;
        for (int i = 0; i < 32; ++i) total += s_ws[i];
        if (keep > 0)
        {
            if (t == 0 && total < pz && a.topp_miss) atomicAdd(a.topp_miss, 1u);
            float bs = -INFINITY, bl = 0.f;
            int bi = 0x7fffffff;
            for (int i = t; i < keep; i += 1024)
            {
                if (RS && s_id[i] == d) continue;
                const float sc = s_l[i] * sr.inv_temp + gumbel(sr, s_id[i]);
                if (sc > bs || (sc == bs && s_id[i] < bi)) { bs = sc; bi = s_id[i]; bl = s_l[i]; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                const float os = __shfl_xor_sync(0xffffffffu, bs, o), ol = __shfl_xor_sync(0xffffffffu, bl, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (os > bs || (os == bs && oi < bi)) { bs = os; bi = oi; bl = ol; }
            }
            if (l == 0) { s_best[w] = bs; s_bi[w] = bi; s_bl[w] = bl; }
            __syncthreads();
            if (t == 0)
            {
                for (int i = 1; i < 32; ++i)
                    if (s_best[i] > s_best[0] || (s_best[i] == s_best[0] && s_bi[i] < s_bi[0])) { s_best[0] = s_best[i]; s_bi[0] = s_bi[i]; s_bl[0] = s_bl[i]; }
                if (!RS || s_bi[0] != 0x7fffffff) { s_bid = s_bi[0]; s_blg = s_bl[0]; }   // RS: d was the only kept entry
                if (RS && d >= 0)
                {
                    float kz = 0.f;
                    for (int i = 0; i < 32; ++i) kz += s_kz[i];
                    int bt = s_bid;
                    float bg = s_blg;
                    rs_accept(sr, d, s_in ? expf((s_ld - mx) * sr.inv_temp) / kz : 0.f, bt, bg, s_in ? s_ld : 0.f);
                    s_bid = bt; s_blg = bg;
                }
            }
        }
        else if (t == 0 && a.topp_miss) atomicAdd(a.topp_miss, 1u);   // no candidate above the cut: full-vocab sample
        __syncthreads();
    }
    if (RS && d >= 0 && t == 0 && !(on && s_keep > 0))
    {
        int bt = s_bid;
        float bg = s_blg;
        rs_accept(sr, d, expf((s_tl - s_mx) * sr.inv_temp) / s_z, bt, bg, s_tl);
        s_bid = bt; s_blg = bg;
    }
    if (t == 0)
    {
        a.tok_out[m] = s_bid;
        a.lp_out[m] = s_blg - s_mx - logf(s_sm);
        if (a.lpt_out) a.lpt_out[m] = s_tl - s_mx - logf(s_sm);
    }
    if (a.tk && sr.topk > 0)                  // top list: loaded by the block, merged by thread 0
    {
        __shared__ TkEnt s_te[8 * LP_K];
        __syncthreads();
        if (t < a.f.nr * LP_K) s_te[t] = head_tk_peer(a, step, t / LP_K, m, t % LP_K);
        __syncthreads();
        if (t == 0)
        {
            TkEnt top[LP_K];
            tk_merge(a.f.nr, [&](int p, int i) { return s_te[p * LP_K + i]; }, top);
            tk_result(a.tk, m, top, s_mx, s_sm, s_blg, s_tl);
        }
    }
}

__global__ void __launch_bounds__(1024) k_head_merge_tp(HeadArgs a) { head_merge_tp<false>(a, nullptr); }
__global__ void __launch_bounds__(1024) k_head_merge_tp_rs(HeadArgs a, const int* tok) { head_merge_tp<true>(a, tok); }

// ---------------------------------------------------------------- RoPE (interleaved pairs, FACTS-M0 2(c))

// cos/sin of pos * theta^(-2j/64), angle formed and reduced in double so long positions keep fp32 accuracy.
__device__ __forceinline__ void rope_cs(int pos, int j, float& c, float& s)
{
    const double inv = exp(-(2.0 * j / ROPE_DIM) * log(ROPE_THETA));
    double sd, cd;
    sincos((double)pos * inv, &sd, &cd);
    c = (float)cd;
    s = (float)sd;
}

// In place on fp16 rows: for row r and head h, rotate the 64 channels at x + r*ld + h*hstride + off as
// 32 adjacent pairs. grid (rows), block (nheads*32 threads, <= 1024; thread = (head, pair)).
__global__ void k_rope(__half* x, int ld, int nheads, int hstride, int off, const int* pos)
{
    const int r = blockIdx.x, h = threadIdx.x >> 5, j = threadIdx.x & 31;
    if (h >= nheads) return;
    float c, s;
    rope_cs(pos[r], j, c, s);
    __half2* p = reinterpret_cast<__half2*>(x + (size_t)r * ld + (size_t)h * hstride + off) + j;
    const float2 v = __half22float2(*p);
    *p = __floats2half2_rn(v.x * c - v.y * s, v.x * s + v.y * c);
}

}  // namespace glue
