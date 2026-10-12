// gr8 rank prefill: one rank's prefill sub-batch (M rows of one sequence at consecutive positions, M up to
// Prefill::Mmax) over the model's layers, through the prefill binder (bind_prefill.cuh: MXFP8 dense, TR3 -> FP8
// expert GEMMs, BF16 via cuBLAS) and the same attention bindings as the decode step (rank.cuh), so the KV and index
// caches a prefill writes are the ones the decode steps read.
//
// No head: the prefill path takes every prompt token but the last; the last one goes through a normal decode step,
// which also samples the first token.
//
// Fabric: a second, larger region (slot = Mmax x 6144 BF16) with TWO collectives (0 = attention partial, 1 = MLP
// partial) and the LAYER as its step: layer-step number ls = ls0 + i for layer i of a sub-batch, slots alternate by
// parity of ls. A peer cannot publish layer-step ls + 2 before it has consumed every rank's MLP partial of ls + 1,
// which this rank only sends after consuming ls, so reusing the slot two layers later is safe.
// k_set_step writes the layer-step cell on the stream before each layer. Plain launches (not a graph).
//
// Reduce-scatter form (fabric with FIVE collectives, G6_PF_RS=1 in the drivers): rows are owned by rank,
// p: [pf_r0(M, p), pf_r0(M, p + 1)). Per layer-step in time order c0 = RS of the attention partial, c1 = AG of the
// owner's normed x2 rows, c2 = RS of the MLP partial, c3 = AG of the normed x rows, and after the last layer
// c4 = AG of the FP32 residual rows (only so pf_hash keeps its meaning). RS send slot = nr slices at p * pf_ss(f); the
// real fabric delivers slice p to peer p at offset 0 of its recv slot [c][me]; the loopback (P.loop, m4sim) keeps every
// slice at its offset. The owner runs k_ar_add_norm's exact op order on its rows (sum p0..p5 FP32, + resid, cluster
// RMS), so x / x2 are byte-equal to the all-reduce form; the resid rows of other owners are stale until c4.
//
// Row-owner projections (G6_PF_OWN=1, RS form only; phase 1a of the row-owner front end, design note
// parallelism.md section 3): for every layer but the first, the owner computes the next layer's q_a / kv_a / idx_k /
// idx_w and q_a_norm on its rows right after its c2 norm, and c3 all-gathers those records (OWN_REC = 5,568 B/row)
// instead of the normed x rows (12,288 B/row). The rest of the attention block (q_b, idx_q, attention) runs on all
// rows as before. The last layer's c3 still gathers x (the MTP pass reads the final-norm rows).
// G6_PF_OWN=2 (also --pfv2; phase 1b): for a full-index layer i + 1, after the c3 gather the owner also computes idx_q
// for its rows, every rank appends all rows' index keys (g6_attn_bind_index_append), the owner scores and selects
// top-2048 for its rows only (g6_attn_bind_prefill_index_rows), and c4 of layer-step i (free on every step but the
// last) all-gathers the ordered IDs (8,192 B/row) into idx_all; layer i + 1's attention then reuses them (pf_ix, called
// by the drivers after the consume, before the next pf_set_step). Two-stream lanes: that work touches the layer's shared
// attention state, so it waits for the other lane like the attention does.
//
// Row-owner QO attention (G6_PF_OWN=3, needs G6_PF_RS=1 and --pfv2; engine/runtime/QO-INTEGRATION.md): the owner of a
// row runs the whole attention block for all 64 heads of it (full-head q_b, UK absorb, the QO core kernel, UV,
// full-K o_proj), adds the result into its residual rows and norms them itself (k_own_add_norm), which publishes c1.
// c0 is never used; c3 carries only kv_a | idx_k (QO_REC); full-index layers append the index keys of every row and
// score the owner rows inside the attention block (no per-layer c4). Every rank still appends every row's latent KV
// and index keys, so the caches stay replicated for decode.
//
// Pair-owner QO attention (G6_PF_OWN=4, TP4 builds only, needs G6_PF_RS=1 and --pfv2; docs/DESIGN-QO-PAIR.md, qp.cuh):
// OWN=2 with, on the layers of G6_PF_QP_LAYERS, OWN=3's attention block split over the head pairs {0,1} and {2,3}.
#pragma once
#include <algorithm>
#include "envparse.h"
#include <ctime>
#include <array>
#include <map>
#include "rank.cuh"
#include "../kernels/bind_prefill.cuh"
#include "../attn/prefill.h"
#include "../attn/prefill-qo-bind.h"
#include "qo_weights.cuh"
#include "dtr_find.h"
#include "cp_pf_tail.h"

namespace g6rt
{

__global__ void k_set_step(u64* cell, u64 v) { if (threadIdx.x == 0) *cell = v; }
// row `row` of src [.][HID] fp16 -> dst row 0 (not used by the step itself; test helper). grid 1, block 256.
__global__ void k_copy_row(const __half* src, int row, __half* dst)
{
    for (int i = threadIdx.x; i < HID; i += 256) dst[i] = src[(size_t)row * HID + i];
}

// device staging [M][6144] BF16 -> this rank's send slot of collective c, then the doorbell. grid PF_PUT, block 128.
constexpr int PF_PUT = 32;
__global__ void __launch_bounds__(128) k_put_publish_n(const uint4* src, g6::FabricDev f, const u64* stepp, int c, int n16,
                                                       unsigned* ticket)
{
    const u64 step = *stepp;
    uint4* dst = reinterpret_cast<uint4*>(glue::send_slot(f, step, c));
    for (int i = blockIdx.x * 128 + threadIdx.x; i < n16; i += gridDim.x * 128) dst[i] = src[i];
    glue::publish(f, c, g6::coll_seq(step, c, f.ncoll), ticket);
}

// ---------------------------------------------------------------- reduce-scatter form helpers and kernels
// strict parse of the G6_PF_* choice variables, called first thing in main (m4sim, g6m4): unset = default, else exactly
// one of the listed values or a fatal exit before any load (atoi would fail open)
inline void pf_env_check()
{
    std::string e;
    if (env_choice("G6_PF_DRAFT_SHARE", 1, 0, &e) < 0) die(e);
    if (env_choice("G6_PF_QP_EARLY", 2, 0, &e) < 0) die(e);          // G6_PF_OWN=4 (qp.cuh)
    if (env_choice("G6_PF_QP_WARM", 1, 0, &e) < 0) die(e);
    const int roll = env_choice("G6_PF_QP_ROLL", 4, 0, &e);       // 0 = resident pair pack, 2..4 = rolling slots
    if (roll < 0) die(e);
    if (roll == 1) die("G6_PF_QP_ROLL='1': a rolling pack needs 2..4 slots (0 = resident)");
    if (env_choice("G6_PF_QP_ROLL_SER", 1, 0, &e) < 0) die(e);    // measurement: builds on the lane stream (fully exposed)
#if defined(G6_CP_PF_HASH2) && G6_CP_PF_HASH2
    if (env_choice("G6_CP_PF_HASH2", 1, 1, &e) < 0) die(e);   // HASH2 build: unset / 1 = hash2, 0 = k_cp_pf_hash, else exit
#endif
}
// G6_PF_DRAFT_SHARE=1: the MTP draft prefill borrows the target prefill's tensor-core attention scratch instead of
// allocating a second root of the same size. Unset / 0 = today's behaviour exactly.
inline bool pf_draft_share()
{
    std::string e;
    const int v = env_choice("G6_PF_DRAFT_SHARE", 1, 0, &e);
    if (v < 0) die(e);
    return v == 1;
}
__host__ __device__ inline int pf_r0(int M, int p, int nr) { return p * M / nr; }          // first row of rank p
__host__ __device__ inline int pf_rows(int M, int p, int nr) { return pf_r0(M, p + 1, nr) - pf_r0(M, p, nr); }
__host__ __device__ inline size_t pf_ss(const g6::FabricDev& f) { return (f.stride / f.nr) & ~size_t(255); }
inline size_t pf_ss_min(int Mmax, int nr) { return ((size_t)(Mmax + nr - 1) / nr * HID * 2 + 8 + 255) & ~size_t(255); }
// FabricConfig::slot_bytes of a prefill region
inline size_t pf_slot_bytes(int Mmax, bool rs, int nr = g6nr::NR)
{
    const size_t ar = (size_t)Mmax * HID * 2;
    return rs ? std::max(ar, nr * pf_ss_min(Mmax, nr)) : ar;
}

// partial [M][6144] BF16 -> slice p of the send slot of collective c (rows of rank p at p * ss), then the doorbell.
// grid PF_PUT, block 128.
__global__ void __launch_bounds__(128) k_put_rs(const uint4* src, g6::FabricDev f, const u64* stepp, int c, int M, unsigned* ticket)
{
    const u64 step = *stepp;
    char* dst = glue::send_slot(f, step, c);
    const size_t ss = pf_ss(f);
    constexpr int R16 = HID * 2 / 16;
    const int n16 = M * R16;
    for (int i = blockIdx.x * 128 + threadIdx.x; i < n16; i += gridDim.x * 128)
    {
        const int row = i / R16;
        const int p = ((row + 1) * f.nr - 1) / M;                 // the owner: largest p with pf_r0(M, p) <= row
        reinterpret_cast<uint4*>(dst + p * ss)[(size_t)(row - pf_r0(M, p, f.nr)) * R16 + i % R16] = src[i];
    }
    glue::publish(f, c, g6::coll_seq(step, c, f.ncoll), ticket);
}
// G6_PF_RSDIRECT=1 (RS form only): o_proj, the dense down GEMM and the MoE final sum write their rows straight into the
// send slot (bindp::RsOut), so pf_put only rings the doorbell. grid 1, block 32.
inline bool pf_rsdirect()
{
    static const bool on = getenv("G6_PF_RSDIRECT") && atoi(getenv("G6_PF_RSDIRECT")) == 1;
    return on;
}
__global__ void __launch_bounds__(32) k_publish(g6::FabricDev f, const u64* stepp, int c, unsigned* ticket)
{
    glue::publish(f, c, g6::coll_seq(*stepp, c, f.ncoll), ticket);
}
// where rank p's slice for THIS rank lives in collective c's slot of rank p
__device__ __forceinline__ const char* rs_slice(const g6::FabricDev& f, u64 step, int c, int p, bool loop)
{
    return glue::peer_slot(f, step, c, p) + (p == f.rank || loop ? f.rank * pf_ss(f) : 0);
}
// grid 1, block 32: every peer's slice of collective c for this rank has landed
__global__ void __launch_bounds__(32) k_wait_rs(g6::FabricDev f, const u64* stepp, int c, int M, bool loop)
{
    const u64 step = *stepp;
    const int p = threadIdx.x;
    if (p < f.nr && p != f.rank)
    {
        const u64 seq = g6::coll_seq(step, c, f.ncoll);
        const u64* tail = reinterpret_cast<const u64*>(rs_slice(f, step, c, p, loop) + (size_t)pf_rows(M, f.rank, f.nr) * HID * 2);
        const long long t0 = clock64();
        unsigned it = 0;
        while (glue::ld_acquire_sys(tail) != seq)
            if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(f.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
            {
                atomicCAS(f.error, 0u, glue::ERR_AR | ((unsigned)c << 8) | (unsigned)p);
                break;
            }
    }
}
struct RsNormArgs
{
    g6::FabricDev f;
    const u64* stepp;
    int c;                        // the RS collective; c + 1 = the all-gather of the normed rows
    int M;
    bool loop;
    float* resid;                 // [M][6144] fp32: this rank's rows updated in place
    const __nv_bfloat16* w;
    __half* out;                  // [M][6144]: this rank's rows
    float eps;
    unsigned* ticket;             // of collective c + 1
    bool ag = true;               // false: no AG rows and no doorbell of c + 1 (G6_PF_OWN sends records there instead)
};
// grid (RN_CL, max(rows, 1)), block RN_T: k_ar_add_norm on this rank's rows (same op order), the normed row also
// into the send slot of collective c + 1 (row j of the rank at j * 6144), then the doorbell of c + 1
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T) k_rs_add_norm(RsNormArgs a)
{
    const u64 step = *a.stepp;
    const int j = blockIdx.y;
    if (j < pf_rows(a.M, a.f.rank, a.f.nr))
    {
        const int m = pf_r0(a.M, a.f.rank, a.f.nr) + j;
        const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
        const size_t off = ((size_t)j * HID + col0) * 2;
        float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int p = 0; p < a.f.nr; ++p)
        {
            const uint4 v = __ldcv(reinterpret_cast<const uint4*>(rs_slice(a.f, step, a.c, p, a.loop) + off));
            float b[8];
            glue::bf16x8_to_f(v, b);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x[k] += b[k];
        }
        float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
        const float4 r0 = r[0], r1 = r[1];
        x[0] += r0.x; x[1] += r0.y; x[2] += r0.z; x[3] += r0.w;
        x[4] += r1.x; x[5] += r1.y; x[6] += r1.z; x[7] += r1.w;
        r[0] = make_float4(x[0], x[1], x[2], x[3]);
        r[1] = make_float4(x[4], x[5], x[6], x[7]);
        __half* ag = a.ag ? reinterpret_cast<__half*>(glue::send_slot(a.f, step, a.c + 1)) + (size_t)j * HID : nullptr;
        glue::cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0, ag);
    }
#ifdef PF_RNPUB
    // rows < 1 launches one cluster with no stores and no cluster barrier: nothing to order, only its ticket
    if (a.ag) glue::publish_cluster(a.f, a.c + 1, g6::coll_seq(step, a.c + 1, a.f.ncoll), a.ticket);
#else
    if (a.ag) glue::publish(a.f, a.c + 1, g6::coll_seq(step, a.c + 1, a.f.ncoll), a.ticket);
#endif
}
#ifdef PF_RNWIDE
// PF_RNWIDE: k_rs_add_norm with one 768-thread CTA per row instead of an 8-CTA cluster of 96 threads (8 columns per
// thread as before; thread t = old (cluster rank t / 96, thread t % 96)). The RMS reduction keeps glue::cluster_rms_store's
// order: in-thread, warp xor tree, each old CTA's 3 warps summed 0..2, then the 8 old-CTA sums in rank order, so the
// bits are those of k_rs_add_norm. grid max(rows, 1), block RN_CL * RN_T.
__global__ void __launch_bounds__(glue::RN_CL * glue::RN_T) k_rs_add_norm_w(RsNormArgs a)
{
    constexpr int NW = glue::RN_CL * glue::RN_T / 32, WPC = glue::RN_T / 32;
    __shared__ float wsum[NW];
    __shared__ float s_inv;
    const u64 step = *a.stepp;
    const int j = blockIdx.x;
    if (j < pf_rows(a.M, a.f.rank, a.f.nr))
    {
        const int m = pf_r0(a.M, a.f.rank, a.f.nr) + j;
        const int col0 = threadIdx.x * 8;
        const size_t off = ((size_t)j * HID + col0) * 2;
        float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#if PF_RNWIDE >= 2
        // all peer slices in flight first (nr <= RN_MAXR), then summed in the same p order
        constexpr int RN_MAXR = 8;
        uint4 vv[RN_MAXR];
        #pragma unroll
        for (int p = 0; p < RN_MAXR; ++p)
            if (p < a.f.nr) vv[p] = __ldcv(reinterpret_cast<const uint4*>(rs_slice(a.f, step, a.c, p, a.loop) + off));
        #pragma unroll
        for (int p = 0; p < RN_MAXR; ++p)
        {
            if (p >= a.f.nr) break;
            float b[8];
            glue::bf16x8_to_f(vv[p], b);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x[k] += b[k];
        }
#else
        for (int p = 0; p < a.f.nr; ++p)
        {
            const uint4 v = __ldcv(reinterpret_cast<const uint4*>(rs_slice(a.f, step, a.c, p, a.loop) + off));
            float b[8];
            glue::bf16x8_to_f(v, b);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x[k] += b[k];
        }
#endif
        float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
        const float4 r0 = r[0], r1 = r[1];
        x[0] += r0.x; x[1] += r0.y; x[2] += r0.z; x[3] += r0.w;
        x[4] += r1.x; x[5] += r1.y; x[6] += r1.z; x[7] += r1.w;
        r[0] = make_float4(x[0], x[1], x[2], x[3]);
        r[1] = make_float4(x[4], x[5], x[6], x[7]);
        float ss = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) ss = fmaf(x[k], x[k], ss);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            float tot = 0.f;
            #pragma unroll
            for (int c = 0; c < glue::RN_CL; ++c)
            {
                float t = 0.f;
                #pragma unroll
                for (int i = 0; i < WPC; ++i) t += wsum[c * WPC + i];
                tot += t;
            }
            s_inv = rsqrtf(tot * (1.f / HID) + a.eps);
        }
        __syncthreads();
        const float inv = s_inv;
        uint4 wv = *reinterpret_cast<const uint4*>(a.w + col0);
        float wf[8];
        glue::bf16x8_to_f(wv, wf);
        uint4 o;
        o.x = pack_h2(x[0] * inv * wf[0], x[1] * inv * wf[1]);
        o.y = pack_h2(x[2] * inv * wf[2], x[3] * inv * wf[3]);
        o.z = pack_h2(x[4] * inv * wf[4], x[5] * inv * wf[5]);
        o.w = pack_h2(x[6] * inv * wf[6], x[7] * inv * wf[7]);
        *reinterpret_cast<uint4*>(a.out + (size_t)m * HID + col0) = o;
        if (a.ag) *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(glue::send_slot(a.f, step, a.c + 1)) + (size_t)j * HID + col0) = o;
    }
    if (a.ag) glue::publish(a.f, a.c + 1, g6::coll_seq(step, a.c + 1, a.f.ncoll), a.ticket);
}
#endif
// G6_PF_OWN=3: k_rs_add_norm with ONE source, this rank's complete o_proj rows o [rows][6144] FP32 (row j = owner row
// pf_r0(M, rank) + j): resid += o, normed rows -> out and the send slot of collective c + 1 (= c1), then its doorbell.
// grid (RN_CL, max(rows, 1)), block RN_T.
__global__ void __cluster_dims__(glue::RN_CL, 1, 1) __launch_bounds__(glue::RN_T) k_own_add_norm(RsNormArgs a, const float* o)
{
    const u64 step = *a.stepp;
    const int j = blockIdx.y;
    if (j < pf_rows(a.M, a.f.rank, a.f.nr))
    {
        const int m = pf_r0(a.M, a.f.rank, a.f.nr) + j;
        const int col0 = blockIdx.x * glue::RN_COLS + threadIdx.x * 8;
        const float4* ov = reinterpret_cast<const float4*>(o + (size_t)j * HID + col0);
        const float4 o0 = ov[0], o1 = ov[1];
        float x[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
        float4* r = reinterpret_cast<float4*>(a.resid + (size_t)m * HID + col0);
        const float4 r0 = r[0], r1 = r[1];
        x[0] += r0.x; x[1] += r0.y; x[2] += r0.z; x[3] += r0.w;
        x[4] += r1.x; x[5] += r1.y; x[6] += r1.z; x[7] += r1.w;
        r[0] = make_float4(x[0], x[1], x[2], x[3]);
        r[1] = make_float4(x[4], x[5], x[6], x[7]);
        __half* ag = reinterpret_cast<__half*>(glue::send_slot(a.f, step, a.c + 1)) + (size_t)j * HID;
        glue::cluster_rms_store(x, a.w, a.eps, a.out + (size_t)m * HID, col0, ag);
    }
    glue::publish(a.f, a.c + 1, g6::coll_seq(step, a.c + 1, a.f.ncoll), a.ticket);
}
// grid 1, block 32: every peer's all-gather rows of collective c have landed (payload pf_rows(M, p) x row_bytes)
__global__ void __launch_bounds__(32) k_wait_ag(g6::FabricDev f, const u64* stepp, int c, int M, int row_bytes)
{
    const u64 step = *stepp;
    const int p = threadIdx.x;
    if (p < f.nr && p != f.rank)
    {
        const u64 seq = g6::coll_seq(step, c, f.ncoll);
        const u64* tail = reinterpret_cast<const u64*>(glue::peer_slot(f, step, c, p) + (size_t)pf_rows(M, p, f.nr) * row_bytes);
        const long long t0 = clock64();
        unsigned it = 0;
        while (glue::ld_acquire_sys(tail) != seq)
            if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(f.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
            {
                atomicCAS(f.error, 0u, glue::ERR_AR | ((unsigned)c << 8) | (unsigned)p);
                break;
            }
    }
}
// grid (PF_AG, nr), block 256, after k_wait_ag: peer p's rows (offset 0 of its slot) -> dst rows pf_r0(M, p)..
// (no spinning here, so a waiting lane holds one CTA, never a grid, while the other lane's stream runs)
constexpr int PF_AG = 32;
__global__ void __launch_bounds__(256) k_ag_copy(g6::FabricDev f, const u64* stepp, int c, int M, int row_bytes, char* dst)
{
    const int p = blockIdx.y;
    if (p == f.rank) return;
    const uint4* s16 = reinterpret_cast<const uint4*>(glue::peer_slot(f, *stepp, c, p));
    uint4* d16 = reinterpret_cast<uint4*>(dst + (size_t)pf_r0(M, p, f.nr) * row_bytes);
    const size_t n16 = (size_t)pf_rows(M, p, f.nr) * row_bytes / 16;
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n16; i += (size_t)gridDim.x * 256) d16[i] = __ldcv(s16 + i);
}
inline void pf_ag(const g6::FabricDev& f, const u64* stepp, int c, int M, int row_bytes, char* dst, cudaStream_t st)
{
    k_wait_ag<<<1, 32, 0, st>>>(f, stepp, c, M, row_bytes);
    k_ag_copy<<<dim3(PF_AG, f.nr), 256, 0, st>>>(f, stepp, c, M, row_bytes, dst);
}
// G6_PF_OWN record of one row: q_a_n [2048] | kv_a [576] | idx_k [128] | idx_w [32] fp16 (non-indexer layers carry
// stale idx_k / idx_w columns, never read)
constexpr int OWN_REC = (QA + KVA + IDXK + IDXW) * 2;
constexpr int QO_REC = (KVA + IDXK) * 2;           // G6_PF_OWN=3 record: kv_a [576] | idx_k [128] fp16
struct OwnRec { __half *q_a_n, *kv_a, *idx_k, *idx_w; };
// QO (G6_PF_OWN=3): a template parameter, so the OWN=2 instantiations are the served code unchanged
template <bool QO = false> __device__ __forceinline__ uint4* own_at(const OwnRec& r, int row, int k)     // k: uint4 index inside the record
{
    constexpr int Q = QA * 2 / 16, K = KVA * 2 / 16, I = IDXK * 2 / 16;
    if constexpr (QO) k += Q;                        // no q_a_n / idx_w columns in a QO record
    if (k < Q) return reinterpret_cast<uint4*>(r.q_a_n + (size_t)row * QA) + k;
    if ((k -= Q) < K) return reinterpret_cast<uint4*>(r.kv_a + (size_t)row * KVA) + k;
    if ((k -= K) < I) return reinterpret_cast<uint4*>(r.idx_k + (size_t)row * IDXK) + k;
    return reinterpret_cast<uint4*>(r.idx_w + (size_t)row * IDXW) + (k - I);
}
// this rank's rows -> send slot of collective c (row j at j * OWN_REC), then the doorbell. grid PF_PUT, block 128.
template <bool QO = false> __global__ void __launch_bounds__(128) k_own_put(OwnRec r, g6::FabricDev f, const u64* stepp, int c, int M, unsigned* ticket)
{
    const u64 step = *stepp;
    uint4* dst = reinterpret_cast<uint4*>(glue::send_slot(f, step, c));
    constexpr int R16 = (QO ? QO_REC : OWN_REC) / 16;
    const int r0 = pf_r0(M, f.rank, f.nr), n16 = pf_rows(M, f.rank, f.nr) * R16;
    for (int i = blockIdx.x * 128 + threadIdx.x; i < n16; i += gridDim.x * 128) dst[i] = *own_at<QO>(r, r0 + i / R16, i % R16);
    glue::publish(f, c, g6::coll_seq(step, c, f.ncoll), ticket);
}
// grid (PF_AG, nr), block 256, after k_wait_ag(.., OWN_REC): peer p's records -> rows pf_r0(M, p).. of the four buffers
template <bool QO = false> __global__ void __launch_bounds__(256) k_own_unpack(OwnRec r, g6::FabricDev f, const u64* stepp, int c, int M)
{
    const int p = blockIdx.y;
    if (p == f.rank) return;
    const uint4* s16 = reinterpret_cast<const uint4*>(glue::peer_slot(f, *stepp, c, p));
    constexpr int R16 = (QO ? QO_REC : OWN_REC) / 16;
    const int r0 = pf_r0(M, p, f.nr), n16 = pf_rows(M, p, f.nr) * R16;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n16; i += gridDim.x * 256) *own_at<QO>(r, r0 + i / R16, i % R16) = __ldcv(s16 + i);
}

struct Prefill
{
    int Mmax = 0;
    g6::FabricDev f{};                // ncoll = 2 (all-reduce) or 5 (RS form), stride >= pf_slot_bytes(Mmax, rs) + 8
    u64* stepp = nullptr;             // device layer-step cell
    unsigned* tickets = nullptr;      // [5]
    float* resid = nullptr;           // [Mmax][6144]
    __half *x = nullptr, *x2 = nullptr, *q_a = nullptr, *q_a_n = nullptr, *kv_a = nullptr, *q_b = nullptr;
    __half *idx_q = nullptr, *idx_k = nullptr, *idx_w = nullptr;
    float* logits = nullptr;          // [Mmax][256]
    bool rs = false;                  // reduce-scatter form (f.ncoll == 5)
    bool loop = false;                // set before build_prefill by a loopback driver (m4sim): RS slices stay at p * ss
    bool own = false;                 // set before build_prefill: G6_PF_OWN row-owner projections (RS form only)
    bool ownix = false;               //   G6_PF_OWN=2: row-owner index scoring too (needs own and v2)
    bool qo = false;                  //   G6_PF_OWN=3: row-owner QO attention (sets own; needs v2; QO-INTEGRATION.md)
    const std::vector<QoW>* qow = nullptr;   // set before build_prefill with qo: full-head weights per layer of the model
    std::vector<G6AttnQo*> qa;        // qo: one state per layer (shared by a second lane)
    int qo_rows = 0;                  // qo: owner rows per call, upper bound (ceil(Mmax / nr) + 16)
    __half *qo_qb = nullptr, *qo_out = nullptr;   // [qo_rows][16384]: full q_b of rows a0.., attention output of rows r0..
    uint8_t* qo_x8 = nullptr;         // MXFP8 records of qo_out (K = 16384)
    float* qo_o = nullptr;            // [qo_rows][6144] FP32 o_proj output (complete, not a partial)
    bool perm = false;                // qo + G6_PF_OWNPERM=1: rows in 16-row-block owner order (pf_perm_order), see pf_inputs
    int* ord = nullptr;               // perm: [Mmax] row j of the sub-batch = input row ord[j]
    float* perm_tmp = nullptr;        // perm: [Mmax][6144] scratch for putting resid / x back in input order
    std::map<int, std::pair<bindp::Gemm, bindp::Gemm>> qg;   // by sub-batch M: full q_b, full o_proj
    std::vector<char> ixl;            // per layer: full-index layer
    bool qp = false;                  // G6_PF_OWN=4 pair-owner QO (with own + ownix, needs v2, NR = 4; qp.cuh), this request
    std::vector<QpW>* qpw = nullptr;  //   set before build_prefill: pair weights per model layer (empty outside qpl)
    std::vector<char> qpl;            //   set before build_prefill: per model layer, 1 = pair-owner (G6_PF_QP_LAYERS), 0 = OWN=2
    QpRoll* roll = nullptr;           //   set before build_prefill (both lanes, one object): G6_PF_QP_ROLL, *qpw names its slots
    int qp_rows = 0;                  //   W rows per call, upper bound: 2 x (ceil(Mmax / nr) + 16)
    int qp_early = 0;                 //   G6_PF_QP_EARLY: 0 = O_c's partial sent after W's o_proj; 1 = after O_c's own xq +
                                      //   o_proj (overlaps O_r's o_proj); 2 = O_c's QO call, o_proj, send, then O_r's QO call
    __half *qp_qb = nullptr, *qp_out = nullptr;   // [qp_rows][8192]: q_b of W (+ alignment rows), attention output of W
    uint8_t* qp_x8 = nullptr;         //   MXFP8 records of qp_out (K = 8192)
    __nv_bfloat16* qp_part = nullptr; //   [qp_rows][6144] o_proj partials of W: O_c rows (sent to c), then O_r rows
    int* qp_pos = nullptr;            //   [qp_rows] positions of W
    int* qp_ids = nullptr;            //   [qp_rows][2048] selected IDs of W
    int qp_pend = -1;                 //   model layer whose pair norm the next c0 consume runs (pf_attn_block_qp sets it)
    const __nv_bfloat16* qp_mine = nullptr;       // its own-pair partial of O_r
    int lane = 0;                     //   0 = lane A (built first), 1 = lane B (built with share): its roll `used` events
    bool qp_built = false;            //   built with G6_PF_OWN=4 (qp is then chosen per request, pf_qp_select)
    int qp_ctx = 0;                   //   largest prompt + gen that may run OWN=4: the QO key / RoPE table size
    uint16_t* part[2] = {nullptr, nullptr};
    int *tok = nullptr, *pos = nullptr;
    void* scratch = nullptr;
    bindp::Scratch sc;
    cublasHandle_t blas = nullptr;
    std::vector<bindp::Tensors> T;                       // per layer of the model
    std::map<int, std::vector<bindp::Layer>> L;          // by row count
    std::map<int, std::vector<bindp::Layer>> Lown;       // G6_PF_OWN: bindings of this rank's rows of an M-row sub-batch
    int blas_rows = 0;                                   // G6_PF_BLAS_ROWS (read at build): cuBLAS members in chunks of
    std::map<std::pair<int, int>, std::vector<bindp::Layer>> Lblas;   //   this many rows; bindings by (first row, rows)
    bool v2 = false;                                     // set before build_prefill: the tensor-core prefill attention
    std::vector<G6AttnPrefill*> av;                      //   (FP16 operands, lossy; ATTN PREFILL-V2.md), one state per layer
    int* idx_keep = nullptr;                             // two-lane form: this lane's copy [Mmax][2048] of the last full-index
                                                         //   layer's list (the other lane overwrites the binding's own)
    int attn_rows = 0;                                   // set before build_prefill: attention in chunks of this many rows
    __half* attn_out = nullptr;                          //   (< Mmax: G6_PF_ATTN_ROWS) -> this lane's attention output
    int* idx_all = nullptr;                              //   [Mmax][H*256] and index lists [Mmax][2048], chunks copied in
    std::string dump_dir;                                // G6_PF_IDDUMP=<dir> (read at build, rank 0 only): after the
    std::vector<int> dump;                               //   attention call of the first / middle / last full-index layer
                                                         //   (positions in Model::layers) write its ordered IDs + positions
    // G6_PF_PDUMP=<dir> (read at build, every rank; measurement only, needs --pfv2 and G6_PF_OWN <= 1): the attention of
    // the layers in G6_PF_PDUMP_LAYERS (default 0,3,20,34,55,74) runs through the trace entry and writes, for the rows
    // whose position % G6_PF_PDUMP_EVERY (default 32) == 0, ids / probs / index scores (pf_pdump). Unset: nothing allocated.
    std::string pd_dir;
    std::vector<char> pd_lay;                            // per layer position: 1 = dump, 2 = index scores only (the
    int pd_every = 32;                                   //   full-index source of a dumped reuse layer)
    float *pd_probs = nullptr, *pd_isc = nullptr;        // [attn_rows][HMAX][2048], [attn_rows][capacity]
    std::string rc_file;                                 // G6_PF_RCDUMP=<dir> (read at build, every rank; measurement
                                                         //   only): <dir>/routes.r<rank>.jsonl, see pf_rcdump
    void (*dfl_tap)(Model&, Prefill&, int, int, cudaStream_t) = nullptr;   // --dflash: after layer i, keep the prompt rows the
                                                                            //   drafter reads (dflash.cuh dfl_pf_tap_launch)
};

// m: a built Model whose attention bindings were created with max_rows >= Mmax (build(..., max_rows)).
// share: a second lane (pf_step2) passes the first lane's Prefill: it reuses its tensor-core attention states and both
// keep their own copy of the index lists
// tc_root (share == nullptr, P.v2): this model's own per-layer states are created, but the first one borrows that
// state's scratch block instead of allocating a root of its own (one H16/full-index allocation for both models).
// Every use of the borrowed block must be serialized against the root's uses (attn/prefill.h create_shared contract).
// G6_PF_OWN=3 load-time warm-up (results/peak/qol0): cuBLAS's first call per (handle, shape) costs ~0.1-0.4 s on the
// host, paid by the first request otherwise. Absorb + UV at the owner row counts of an Mmax sub-batch over the ranks
// (682 / 683 at 4096 x 6), and this rank's full-head q_b GEMM at that sub-batch, on scratch; then a sync.
inline void pf_qo_warm(Model& m, Prefill& P, cudaStream_t st)
{
    const int nr = P.f.nr, me = P.f.rank, M = P.Mmax;
    int lo = M, hi = 0;
    for (int r = 0; r < nr; ++r) { const int n = pf_r0(M, r + 1, nr) - pf_r0(M, r, nr); lo = std::min(lo, n); hi = std::max(hi, n); }
    for (int rows = lo; rows <= hi; ++rows)
        if (rows > 0 && rows <= P.qo_rows)
            if (int e = g6_attn_qo_warm(P.qa.front(), rows, st)) die("g6_attn_qo_warm failed (" + std::to_string(e) + ")");
    const int r0 = pf_r0(M, me, nr), r1 = pf_r0(M, me + 1, nr), a0 = r0 & ~15;
    const QoW& w = (*P.qow)[0];
    bindp::Gemm g;
    if (const char* e = bindp::make_gemm(w.qb_c, w.qb_s, r1 - a0, P.sc.x8_small, P.qo_qb, 0, g)) die(std::string("QO q_b warm: ") + e);
    bindp::launch(g, st);
    G6RT_CK(cudaStreamSynchronize(st));
    if (m.rank == 0) fprintf(stderr, "# rank %d: G6_PF_OWN=3 warm-up: absorb + UV at %d..%d owner rows, q_b at %d rows\n", m.rank, lo, hi, r1 - a0);
}
// G6_PF_OWN=4 (qp.cuh in a TP4 build; a TP6 build exits)
inline void pf_qp_alloc(Prefill& P, Model& m, int Mmax, const g6::FabricDev& f, Prefill* share);
inline void pf_qp_states(Prefill& P, Model& m, cudaStream_t st, Prefill* share);
inline void pf_qp_warm(Model& m, Prefill& P, cudaStream_t st);
inline void build_prefill(Prefill& P, Model& m, const Find& find, int Mmax, const g6::FabricDev& f, u64* stepp,
                          cudaStream_t st = nullptr, Prefill* share = nullptr, G6AttnPrefill* tc_root = nullptr)
{
    if (f.ncoll != 2 && f.ncoll != 5) die("prefill fabric must have two collectives (five in the RS form)");
    P.rs = f.ncoll == 5;
    if (f.stride < pf_slot_bytes(Mmax, P.rs, f.nr) + 8) die("prefill fabric slots are smaller than Mmax rows");
    if (P.rs && pf_ss(f) < pf_ss_min(Mmax, f.nr)) die("prefill fabric RS slices are smaller than ceil(Mmax / nr) rows");
    if (P.own && !P.rs) die("G6_PF_OWN needs the reduce-scatter form (G6_PF_RS=1)");
    if (P.ownix && (!P.own || !(share ? share->v2 : P.v2))) die("G6_PF_OWN=2 needs G6_PF_RS=1 and --pfv2");
    if (P.qo && (!P.own || P.ownix || !(share ? share->v2 : P.v2) || !P.qow || (int)P.qow->size() != m.nl))
        die("G6_PF_OWN=3 needs G6_PF_RS=1, --pfv2 and the full-head weights of every layer");
    for (const Layer& y : m.layers) P.ixl.push_back(y.indexer);
    P.Mmax = Mmax; P.f = f; P.stepp = stepp;
    const size_t n = (size_t)Mmax;
    P.tickets = dmalloc<unsigned>(5);
    P.resid = dmalloc<float>(n * HID);
    P.x = dmalloc<__half>(n * HID);
    P.x2 = dmalloc<__half>(n * HID);
    P.q_a = dmalloc<__half>(n * QA);
    P.q_a_n = dmalloc<__half>(n * QA);
    P.kv_a = dmalloc<__half>(n * KVA);
    P.q_b = dmalloc<__half>(n * g6nr::QBW);
    P.idx_q = dmalloc<__half>(n * IDXQ);
    P.idx_k = dmalloc<__half>(n * IDXK);
    P.idx_w = dmalloc<__half>(n * IDXW);
    P.logits = dmalloc<float>(n * glue::NEXP);
    P.part[0] = dmalloc<uint16_t>(n * HID);
    P.part[1] = dmalloc<uint16_t>(n * HID);
    P.tok = dmalloc<int>(n);
    P.pos = dmalloc<int>(n);
    P.scratch = dmalloc<char>(bindp::scratch_bytes(Mmax));
    P.sc = bindp::scratch_at(P.scratch, Mmax);
    if (cublasCreate(&P.blas) != CUBLAS_STATUS_SUCCESS) die("cublasCreate failed");
    if (const char* e = getenv("G6_PF_BLAS_ROWS"); e && atoi(e) > 0)
    {
        P.blas_rows = atoi(e);
        if (P.blas_rows % 16) die("G6_PF_BLAS_ROWS must be a multiple of 16");
        if (m.rank == 0) fprintf(stderr, "# prefill cuBLAS members in chunks of %d rows (G6_PF_BLAS_ROWS)\n", P.blas_rows);
    }
    if (const char* d = getenv("G6_PF_IDDUMP"); d && *d && m.rank == 0 && m.nl > 1)
    {
        std::vector<int> ix;
        for (int i = 0; i < m.nl; ++i) if (m.layers[i].indexer) ix.push_back(i);
        if (!ix.empty()) for (int k : {0, (int)ix.size() / 2, (int)ix.size() - 1})
            if (std::find(P.dump.begin(), P.dump.end(), ix[k]) == P.dump.end()) P.dump.push_back(ix[k]);
        P.dump_dir = d;
    }
    const bool chunked = P.attn_rows > 0 && P.attn_rows < Mmax;
    if (!chunked) P.attn_rows = Mmax;
    else P.attn_out = dmalloc<__half>(n * g6nr::QBW);
    if (chunked || P.ownix || P.qo) P.idx_all = dmalloc<int>(n * 2048);
    if (P.qp) pf_qp_alloc(P, m, Mmax, f, share);
    if (P.qo)
    {
        P.qo_rows = (Mmax + f.nr - 1) / f.nr + 16;
        const size_t R = (size_t)P.qo_rows;
        P.qo_qb = dmalloc<__half>(R * 16384);
        P.qo_out = dmalloc<__half>(R * 16384);
        P.qo_x8 = dmalloc<uint8_t>((R + 15) / 16 * (16384 / 32) * bindp::REC);
        P.qo_o = dmalloc<float>(R * HID);
        const char* op = getenv("G6_PF_OWNPERM");
        P.perm = op && atoi(op) == 1;
        if (P.perm)
        {
            P.ord = dmalloc<int>(n);
            P.perm_tmp = dmalloc<float>(n * HID);
            if (f.rank == 0 && !share) fprintf(stderr, "# G6_PF_OWNPERM=1: QO owner rows interleaved in 16-row blocks\n");
        }
    }
    if (const char* d = getenv("G6_PF_RCDUMP"); d && *d)
    {
        P.rc_file = std::string(d) + "/routes.r" + std::to_string(m.rank) + ".jsonl";
        if (m.rank == 0) fprintf(stderr, "# G6_PF_RCDUMP %s: per-expert prefill route counts per MoE layer and sub-batch\n", d);
    }
    if (const char* d = getenv("G6_PF_PDUMP"); d && *d)
    {
        if (!(share ? share->v2 : P.v2) || P.ownix) die("G6_PF_PDUMP needs --pfv2 and G6_PF_OWN <= 1");
        if (const char* e = getenv("G6_PF_PDUMP_EVERY"); e && atoi(e) > 0) P.pd_every = atoi(e);
        std::string ls = getenv("G6_PF_PDUMP_LAYERS") ? getenv("G6_PF_PDUMP_LAYERS") : "0,3,20,34,55,74";
        P.pd_lay.assign(m.nl, 0);
        for (size_t k = 0; k < ls.size();)
        {
            const int l = atoi(ls.c_str() + k);
            for (int i = 0; i < m.nl; ++i)
                if (m.layers[i].l == l)
                {
                    P.pd_lay[i] = 1;
                    const int src = m.layers[i].index_src;
                    if (!m.layers[i].indexer && src >= 0 && !P.pd_lay[src]) P.pd_lay[src] = 2;
                }
            k = ls.find(',', k); if (k == std::string::npos) break; ++k;
        }
        P.pd_dir = d;
        P.pd_probs = dmalloc<float>((size_t)P.attn_rows * g6nr::HMAX * 2048);
        P.pd_isc = dmalloc<float>((size_t)P.attn_rows * m.layers[0].ab.capacity);
        if (m.rank == 0) fprintf(stderr, "# G6_PF_PDUMP %s: every %d rows, layers %s (+ index-score sources)\n", d, P.pd_every, ls.c_str());
    }
    for (const Layer& y : m.layers)
    {
        if (y.ab.max_rows < P.attn_rows) die("attention bindings were created for fewer rows than the prefill attention chunk");
        P.T.push_back(bindp::layer_tensors(y.l, [&](const char* name) { return as_bind(find(name)); }));
        P.T.back().qo_cover = P.qo ? 1 : 0;      // G6_PF_OWN=3 prefills q_b / o_proj from the QO pack (PF_QO_DEDUP packs omit them)
        P.T.back().heads = y.ab.args.heads;
        P.T.back().fc1_wide = y.T.fc1_wide;      // decode widened tr3.fc1 in place: prefill must read the same order
        P.T.back().fc2_wide = y.T.fc2_wide;      // ... and tr3.fc2
        P.T.back().k2 = y.T.k2;                  // 2-bit units: the K2 instantiations (bind::scan_widths)
        P.T.back().sh_trellis = y.T.sh_trellis;  // the shared expert as trellis row 256
        if (y.T.shfp8.codes[0])                  // ... and its FP8 copy for the prefill GEMMs (bind::make_shared_fp8)
        {
            bindp::Tensors& t = P.T.back();
            const bind::LayerTensors::ShFp8& F = y.T.shfp8;
            const int id[3] = {bindp::T_S_GATE, bindp::T_S_UP, bindp::T_S_DOWN};
            for (int j = 0; j < 3; ++j)
            {
                const long long n = j < 2 ? F.I : HID, k = j < 2 ? HID : F.I;
                t.t[id[j]] = bind::Tensor(); t.t[id[j]].p = F.codes[j]; t.t[id[j]].nd = 2; t.t[id[j]].shape[0] = n; t.t[id[j]].shape[1] = k;
                t.s[id[j]] = bind::Tensor(); t.s[id[j]].p = F.scales[j]; t.s[id[j]].nd = 2; t.s[id[j]].shape[0] = n; t.s[id[j]].shape[1] = k / 32;
            }
            t.sh_fp8 = 1;
            for (int i = 0; i < 6; ++i) t.sh_tr[i] = y.T.sh_tr[i];
        }
    }
    if (P.qp) pf_qp_states(P, m, st, share);
    if (P.qo && share) P.qa = share->qa;
    else if (P.qo)
    {
        size_t b = 0;
        for (int i = 0; i < m.nl; ++i)
        {
            G6AttnQo* q = nullptr;
            const QoW& w = (*P.qow)[i];
            if (int e = g6_attn_qo_create(&q, &m.layers[i].ab, &w.uk, &w.uv, P.qo_rows, st, P.qa.empty() ? nullptr : P.qa.front()))
                die("rank " + std::to_string(m.rank) + " layer " + std::to_string(m.layers[i].l) + ": g6_attn_qo_create failed (" + std::to_string(e) + ")");
            P.qa.push_back(q);
            b += g6_attn_qo_bytes(q);
        }
        if (m.rank == 0) fprintf(stderr, "# rank %d: QO attention states (G6_PF_OWN=3, kernel %s) for %d layers: %.3f GB\n", m.rank,
                                 getenv("G6_PF_QO") && *getenv("G6_PF_QO") ? getenv("G6_PF_QO") : "b", m.nl, b / 1e9);
    }
    if (share)
    {
        P.v2 = share->v2; P.av = share->av;
        if (!chunked && !P.ownix && !P.qo)    // chunked / G6_PF_OWN=2 / 3 lanes keep their own lists in idx_all already
        {
            P.idx_keep = dmalloc<int>(n * 2048);
            if (!share->idx_keep) share->idx_keep = dmalloc<int>(n * 2048);
        }
    }
    else if (P.v2)
    {
        size_t b = 0;
        for (Layer& y : m.layers)
        {
            G6AttnPrefill* a = nullptr;
            // private UK/UV per layer; one scratch block and one cuBLAS handle, owned by the first layer's state
            G6RT_CK(g6_attn_bind_prefill_create_shared(&a, &y.ab, st, P.av.empty() ? tc_root : P.av.front()));
            P.av.push_back(a);
            b += g6_attn_bind_prefill_bytes(a);
        }
        fprintf(stderr, "# rank %d: tensor-core prefill attention states for %d layers: %.3f GB\n", m.rank, m.nl, b / 1e9);
    }
    if (P.qo && !share) pf_qo_warm(m, P, st);   // lanes that share the states share the cuBLAS handle
    if (P.qp && !share)
    {
        std::string ee;
        const int warm = env_choice("G6_PF_QP_WARM", 1, 0, &ee);
        if (warm < 0) die(ee);
        if (warm) pf_qp_warm(m, P, st);
    }
}

// G6_PF_ATTN_ROWS=<r>: attention runs in chunks of r rows inside a prefill sub-batch of MP rows (the rest of the layer,
// comms included, still at MP rows). Returns the attention bindings' row count: r if 0 < r < MP, else MP.
inline int pf_attn_rows(int MP)
{
    const char* e = getenv("G6_PF_ATTN_ROWS");
    const int a = e ? atoi(e) : 0;
    return MP > 0 && a > 0 && a < MP ? a : MP;
}
inline const bindp::Layer& pf_layer(Model& m, Prefill& P, int i, int M)
{
    std::vector<bindp::Layer>& v = P.L[M];
    if (v.empty())
    {
        if (M < 1 || M > P.Mmax) die("prefill sub-batch larger than Mmax");
        v.resize(m.nl);
        for (int j = 0; j < m.nl; ++j)
        {
            bindp::Io io;
            io.x = P.x; io.q_a = P.q_a; io.q_a_n = P.q_a_n; io.kv_a = P.kv_a; io.q_b = P.q_b;
            io.idx_q = P.idx_q; io.idx_k = P.idx_k; io.idx_w = P.idx_w;
            io.attn = P.attn_out ? P.attn_out : (const __half*)m.layers[j].ab.args.output;
            io.attn_part = P.part[0];
            io.x2 = P.x2; io.logits = P.logits;
            io.mlp_part = P.part[1];
            if (const char* e = bindp::make_layer(P.T[j], M, P.sc, io, P.blas, v[j]))
                die("rank " + std::to_string(m.rank) + " layer " + std::to_string(m.layers[j].l) + " (prefill): " + e);
            if (P.rs && pf_rsdirect())
            {
                auto rs = [&](int c) {
                    bindp::RsOut o;
                    o.base = P.f.send + (size_t)c * P.f.stride; o.pstride = (size_t)P.f.ncoll * P.f.stride; o.ss = pf_ss(P.f);
                    o.stepp = P.stepp; o.M = M; o.nr = P.f.nr;
                    return o;
                };
                v[j].rs_o = rs(0);
                if (v[j].moe) v[j].ex.rs = rs(2); else v[j].rs_down = rs(2);
                if (v[j].moe && !v[j].ex.on) die("G6_PF_RSDIRECT: MoE layer without the expert path");
            }
            if (v[j].heads != m.layers[j].ab.args.heads) die("prefill binder and attention binder disagree on the head count");
        }
    }
    return v[i];
}
// G6_PF_OWN: layer i's bindings for rows pf_own_a0 .. pf_r0(M, rank + 1) - 1 of an M-row sub-batch: this rank's rows,
// started at the 16-row boundary at or below its first row so every row keeps its slot in the MXFP8 records and the
// mma tiles (k_xq8 / k_gemm_mx8b give the same bits for a row at any M then; the up to 15 leading rows belong
// to the previous owner, are computed from stale x and overwritten by the c3 gather). Only xq_x and q_a / kv_a are
// launched from these; idx_k / idx_w (cuBLAS, whose kernel choice depends on M) run at the full M.
inline int pf_own_a0(int M, const Prefill& P) { return pf_r0(M, P.f.rank, P.f.nr) & ~15; }
inline const bindp::Layer& pf_layer_own(Model& m, Prefill& P, int i, int M)
{
    std::vector<bindp::Layer>& v = P.Lown[M];
    if (v.empty())
    {
        const int r0 = pf_own_a0(M, P), rows = pf_r0(M, P.f.rank + 1, P.f.nr) - r0;
        if (rows < 1 || M > P.Mmax) die("G6_PF_OWN: no owner rows in this sub-batch");
        v.resize(m.nl);
        for (int j = 0; j < m.nl; ++j)
        {
            bindp::Io io;
            io.x = P.x + (size_t)r0 * HID; io.q_a = P.q_a + (size_t)r0 * QA; io.q_a_n = P.q_a_n + (size_t)r0 * QA;
            io.kv_a = P.kv_a + (size_t)r0 * KVA; io.q_b = P.q_b;
            io.idx_q = P.idx_q + (size_t)r0 * IDXQ; io.idx_k = P.idx_k + (size_t)r0 * IDXK; io.idx_w = P.idx_w + (size_t)r0 * IDXW;
            io.attn = P.attn_out ? P.attn_out : (const __half*)m.layers[j].ab.args.output;
            io.attn_part = P.part[0];
            io.x2 = P.x2; io.logits = P.logits;
            io.mlp_part = P.part[1];
            if (const char* e = bindp::make_layer(P.T[j], rows, P.sc, io, P.blas, v[j]))
                die("rank " + std::to_string(m.rank) + " layer " + std::to_string(m.layers[j].l) + " (prefill, owner rows): " + e);
        }
    }
    return v[i];
}

#define G6RT_BLAS(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) die("cuBLAS launch failed in the prefill step"); } while (0)

// G6_PF_BLAS_ROWS=<r>: layer i's bindings for rows r0 .. r0 + rows - 1 (only xb_x / idx_k / idx_w and xb_x2 / router
// are launched from these; the BF16 copy and the fp32 staging stay at row 0 of the scratch)
inline const bindp::Layer& pf_layer_rows(Model& m, Prefill& P, int i, int r0, int rows)
{
    std::vector<bindp::Layer>& v = P.Lblas[{r0, rows}];
    if (v.empty())
    {
        if (rows < 1 || r0 + rows > P.Mmax) die("G6_PF_BLAS_ROWS: chunk outside Mmax");
        v.resize(m.nl);
        for (int j = 0; j < m.nl; ++j)
        {
            bindp::Io io;
            io.x = P.x + (size_t)r0 * HID; io.q_a = P.q_a + (size_t)r0 * QA; io.q_a_n = P.q_a_n + (size_t)r0 * QA;
            io.kv_a = P.kv_a + (size_t)r0 * KVA; io.q_b = P.q_b;
            io.idx_q = P.idx_q + (size_t)r0 * IDXQ; io.idx_k = P.idx_k + (size_t)r0 * IDXK; io.idx_w = P.idx_w + (size_t)r0 * IDXW;
            io.attn = P.attn_out ? P.attn_out : (const __half*)m.layers[j].ab.args.output;
            io.attn_part = P.part[0];
            io.x2 = P.x2 + (size_t)r0 * HID; io.logits = P.logits + (size_t)r0 * glue::NEXP;
            io.mlp_part = P.part[1];
            if (const char* e = bindp::make_layer(P.T[j], rows, P.sc, io, P.blas, v[j]))
                die("rank " + std::to_string(m.rank) + " layer " + std::to_string(m.layers[j].l) + " (prefill, cuBLAS chunk): " + e);
        }
    }
    return v[i];
}
// the cuBLAS members of layer i on an M-row sub-batch. cuBLAS picks its kernel by M, so a row's bits depend on the
// sub-batch size; G6_PF_BLAS_ROWS=<r> runs them in r-row chunks at multiples of r instead, so they depend only on the
// row's chunk (the same for any lane size that is a multiple of r). Rows [a, e): only the chunks that overlap them.
inline void pf_blas_x(Model& m, Prefill& P, int i, int M, cudaStream_t st, int a = 0, int e = -1)
{
    const int R = P.blas_rows;
    if (R <= 0 || R >= M)
    {
        const bindp::Layer& b = pf_layer(m, P, i, M);
        bindp::launch(b.xb_x, st);
        G6RT_BLAS(bindp::launch(b.idx_k, st));
        G6RT_BLAS(bindp::launch(b.idx_w, st));
        return;
    }
    if (e < 0) e = M;
    for (int c0 = a / R * R; c0 < e; c0 += R)
    {
        const bindp::Layer& b = pf_layer_rows(m, P, i, c0, std::min(R, M - c0));
        bindp::launch(b.xb_x, st);
        G6RT_BLAS(bindp::launch(b.idx_k, st));
        G6RT_BLAS(bindp::launch(b.idx_w, st));
    }
}
inline void pf_blas_x2(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    const int R = P.blas_rows;
    if (R <= 0 || R >= M)
    {
        const bindp::Layer& b = pf_layer(m, P, i, M);
        bindp::launch(b.xb_x2, st);
        G6RT_BLAS(bindp::launch(b.router, st));
        return;
    }
    for (int c0 = 0; c0 < M; c0 += R)
    {
        const bindp::Layer& b = pf_layer_rows(m, P, i, c0, std::min(R, M - c0));
        bindp::launch(b.xb_x2, st);
        G6RT_BLAS(bindp::launch(b.router, st));
    }
}

// optional stage profile (G6_PF_PROF=1 in the drivers): synchronizes the stream at every mark, so use it for a
// breakdown only, never for a throughput number
struct PfProf
{
    bool on = false;
    double t[9] = {}, last = 0;   // [1] attention on reuse layers, [8] attention on full-index layers
    static double now() { timespec s; clock_gettime(CLOCK_MONOTONIC, &s); return s.tv_sec * 1e3 + s.tv_nsec * 1e-6; }
    void print(const char* tag) const
    {
        static const char* n[9] = {"projections", "attention", "o_proj", "put+publish", "wait+reduce+norm", "shared/dense MLP", "router+experts", "other", ""};
        double tot = 0;
        for (double v : t) tot += v;
        fprintf(stderr, "# %s prefill stage profile (ms, synchronized):", tag);
        for (int i = 0; i < 8; ++i)
        {
            const double v = i == 1 ? t[1] + t[8] : t[i];
            fprintf(stderr, " %s %.1f (%.0f%%),", n[i], v, 100 * v / (tot > 0 ? tot : 1));
            if (i == 1) fprintf(stderr, " [attention full-index %.1f, reuse %.1f],", t[8], t[1]);
        }
        fprintf(stderr, " total %.1f\n", tot);
    }
};
inline PfProf& pf_prof() { static PfProf p; return p; }
// gate (b), G6_PF_PROF with G6_PF_OWN=3: rank 0's QO block per layer, synchronized wall ms summed over sub-batches.
// Stages: q_b (+ q_a / kv_a / indexer projections on layer 0), cache append (+ owner index rows), absorb, prepass,
// core, UV, o_proj (records + GEMM), norm (k_own_add_norm).
struct QoProf
{
    static constexpr int NS = 8;
    std::map<int, std::array<double, NS>> t;
    std::map<int, int> calls;
    double last = 0;
    void tick(cudaStream_t st, int layer, int k)
    {
        G6RT_CK(cudaStreamSynchronize(st));
        const double n = PfProf::now();
        if (k >= 0) t[layer][k] += n - last;
        last = n;
    }
    void print() const
    {
        if (t.empty()) return;
        static const char* n[NS] = {"q_b", "append", "absorb", "prepass", "core", "UV", "o_proj", "norm"};
        fprintf(stderr, "# QO block rank 0 per layer (ms, synchronized, summed over sub-batches): layer calls");
        for (const char* s : n) fprintf(stderr, " %s", s);
        fprintf(stderr, " total\n");
        for (const auto& [l, v] : t)
        {
            double tot = 0;
            fprintf(stderr, "# qoprof L%03d %d", l, calls.at(l));
            for (double x : v) { fprintf(stderr, " %.3f", x); tot += x; }
            fprintf(stderr, " %.3f\n", tot);
        }
    }
};
inline QoProf& qo_prof() { static QoProf p; return p; }
inline void pf_mark(int k, cudaStream_t st)
{
    PfProf& p = pf_prof();
    if (!p.on) return;
    G6RT_CK(cudaStreamSynchronize(st));
    const double n = PfProf::now();
    if (p.last > 0) p.t[k] += n - p.last;
    p.last = n;
}

// host -> device: the sub-batch's token ids and positions (load-time style copy; the stream must be idle)
// G6_PF_OWNPERM=1 (OWN=3): the sub-batch's rows in 16-row blocks sorted by (block % nr, block), so the contiguous owner
// split p * M / nr gives every rank blocks from the whole position range (results/peak/qe/imbalance: in the first
// sub-batch the low ranks' rows attend to < 2048 keys). Every row carries its position; pf_finish_gather puts resid
// and x back in input order.
inline std::vector<int> pf_perm_order(int M, int nr)
{
    std::vector<int> b((M + 15) / 16), o;
    for (int i = 0; i < (int)b.size(); ++i) b[i] = i;
    std::stable_sort(b.begin(), b.end(), [nr](int x, int y) { return x % nr < y % nr; });
    for (int k : b)
        for (int r = k * 16; r < std::min(M, k * 16 + 16); ++r) o.push_back(r);
    return o;
}
inline void pf_inputs(Prefill& P, int M, const int* tok, const int* pos)
{
    if (P.perm)
    {
        const std::vector<int> o = pf_perm_order(M, P.f.nr);
        std::vector<int> t(M), p(M);
        for (int j = 0; j < M; ++j) { t[j] = tok[o[j]]; p[j] = pos[o[j]]; }
        G6RT_CK(g6h2d::copy(P.tok, t.data(), M * sizeof(int)));
        G6RT_CK(g6h2d::copy(P.pos, p.data(), M * sizeof(int)));
        G6RT_CK(g6h2d::copy(P.ord, o.data(), M * sizeof(int)));
        return;
    }
    G6RT_CK(g6h2d::copy(P.tok, tok, M * sizeof(int)));     // pageable sources: pinned staging (h2d.h)
    G6RT_CK(g6h2d::copy(P.pos, pos, M * sizeof(int)));
}
inline void pf_embed(Model& m, Prefill& P, int M, cudaStream_t st)
{
    glue::EmbedArgs a{P.tok, m.emb, m.layers[0].attn_norm, P.resid, P.x, glue::RMS_EPS};
    glue::k_embed_norm<<<dim3(glue::RN_CL, M), glue::RN_T, 0, st>>>(a);
}
inline void pf_set_step(Prefill& P, u64 ls, cudaStream_t st) { k_set_step<<<1, 1, 0, st>>>(P.stepp, ls); }
// G6_PF_RSHASH=1 (measurement only, synchronizes the stream): FNV of every owner slice of the RS send slot just filled
// for collective 2c, one stderr line per call (the received slices are copies of these; RSDIRECT must not change them)
inline void pf_rshash(Prefill& P, int c, int M, cudaStream_t st)
{
    static const bool on = getenv("G6_PF_RSHASH") && atoi(getenv("G6_PF_RSHASH")) == 1;
    if (!on || !P.rs) return;
    G6RT_CK(cudaStreamSynchronize(st));
    u64 step = 0;
    G6RT_CK(cudaMemcpy(&step, P.stepp, 8, cudaMemcpyDeviceToHost));
    const char* slot = P.f.send + ((size_t)(step & 1) * P.f.ncoll + 2 * c) * P.f.stride;
    uint64_t h = 0xcbf29ce484222325ull;
    for (int p = 0; p < P.f.nr; ++p)
    {
        std::vector<char> b((size_t)pf_rows(M, p, P.f.nr) * HID * 2);
        G6RT_CK(cudaMemcpy(b.data(), slot + p * pf_ss(P.f), b.size(), cudaMemcpyDeviceToHost));
        h = fnv(h, b.data(), b.size());
    }
    fprintf(stderr, "# rshash rank %d step %llu c %d M %d %016llx\n", P.f.rank, (unsigned long long)step, 2 * c, M, (unsigned long long)h);
}
inline void pf_put_launch(Prefill& P, int c, int M, cudaStream_t st)
{
    if (P.rs && pf_rsdirect())
    {
        k_publish<<<1, 32, 0, st>>>(P.f, P.stepp, 2 * c, P.tickets + 2 * c);
        return;
    }
    if (P.rs)
    {
        k_put_rs<<<PF_PUT, 128, 0, st>>>(reinterpret_cast<const uint4*>(P.part[c]), P.f, P.stepp, 2 * c, M, P.tickets + 2 * c);
        return;
    }
    k_put_publish_n<<<PF_PUT, 128, 0, st>>>(reinterpret_cast<const uint4*>(P.part[c]), P.f, P.stepp, c, M * (HID * 2 / 16),
                                            P.tickets + c);
}
inline void pf_put(Prefill& P, int c, int M, cudaStream_t st) { pf_put_launch(P, c, M, st); pf_rshash(P, c, M, st); }
// G6_PF_PDUMP (measurement only, synchronizes the stream): one attention chunk through the trace entry (same computation
// as g6_attn_bind_prefill_run plus FP32 copies), then for the sampled rows <dir>/L<layer>.p<first position of the
// chunk>.r<rank>.{pos,ids,probs,iscore}: int32 [R], int32 [R][2048] (-1 pad), float32 [R][H][2048] aligned with ids
// (dump layers only), float32 [R][2048] index score of each selected key, -inf on pads (full-index layers only);
// one line per file in <dir>/meta.r<rank>.jsonl
inline void pf_pdump(Model& m, Prefill& P, int i, const G6AttnStep& s, cudaStream_t st)
{
    Layer& y = m.layers[i];
    const int r = s.rows, H = y.ab.args.heads, C = s.context;
    const bool probs = P.pd_lay[i] == 1, isc = y.indexer && !s.reuse_indices;
    G6AttnTrace tr{};
    if (probs) tr.probs = P.pd_probs;
    if (isc) tr.index_scores = P.pd_isc;
    G6RT_CK(g6_attn_bind_prefill_trace(P.av[i], &y.ab, &s, &tr, st));
    G6RT_CK(cudaStreamSynchronize(st));
    std::vector<int> pos(r), ids((size_t)r * 2048);
    G6RT_CK(cudaMemcpy(pos.data(), s.positions, r * 4, cudaMemcpyDeviceToHost));
    G6RT_CK(cudaMemcpy(ids.data(), y.ab.args.indices, ids.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> rows;
    for (int k = 0; k < r; ++k) if (pos[k] % P.pd_every == 0) rows.push_back(k);
    if (rows.empty()) return;
    const std::string b = P.pd_dir + "/L" + std::to_string(y.l) + ".p" + std::to_string(pos[0]) + ".r" + std::to_string(m.rank);
    auto put = [&](const char* ext, const void* p, size_t bytes) {
        FILE* f = fopen((b + ext).c_str(), "ab");
        if (!f || fwrite(p, 1, bytes, f) != bytes) die("G6_PF_PDUMP: cannot write " + b + ext);
        fclose(f);
    };
    for (const char* ext : {".pos", ".ids", ".probs", ".iscore"}) remove((b + ext).c_str());
    std::vector<float> row((size_t)H * 2048), sc((size_t)C), g(2048);
    for (int k : rows)
    {
        put(".pos", &pos[k], 4);
        put(".ids", &ids[(size_t)k * 2048], 2048 * 4);
        if (probs)
        {
            G6RT_CK(cudaMemcpy(row.data(), P.pd_probs + (size_t)k * H * 2048, row.size() * 4, cudaMemcpyDeviceToHost));
            put(".probs", row.data(), row.size() * 4);
        }
        if (isc)
        {
            G6RT_CK(cudaMemcpy(sc.data(), P.pd_isc + (size_t)k * C, sc.size() * 4, cudaMemcpyDeviceToHost));
            for (int j = 0; j < 2048; ++j) { const int id = ids[(size_t)k * 2048 + j]; g[j] = id >= 0 && id < C ? sc[id] : -INFINITY; }
            put(".iscore", g.data(), 2048 * 4);
        }
    }
    if (FILE* fm = fopen((P.pd_dir + "/meta.r" + std::to_string(m.rank) + ".jsonl").c_str(), "a"))
    {
        fprintf(fm, "{\"layer\": %d, \"first\": %d, \"chunk_rows\": %d, \"rows\": %d, \"context\": %d, \"rank\": %d, \"heads\": %d, "
                    "\"full_index\": %d, \"index_src_layer\": %d, \"probs\": %d, \"iscore\": %d, \"every\": %d, \"own\": %d, \"tile\": \"%s\", \"dense\": \"%s\"}\n",
                y.l, pos[0], r, (int)rows.size(), C, m.rank, H, (int)y.indexer, y.indexer ? y.l : m.layers[y.index_src].l, (int)probs,
                (int)isc, P.pd_every, (int)P.own + (int)P.ownix, getenv("G6_PF_TILE") ? getenv("G6_PF_TILE") : "",
                getenv("G6_PF_DENSE") ? getenv("G6_PF_DENSE") : "");
        fclose(fm);
    }
}
// G6_PF_ADUMP=<dir> (measurement only, synchronizes): on the first prefill call of model layer position 0, every rank
// writes <dir>/<what>.r<rank>.bin of n rows x row_bytes from dev (QO-INTEGRATION.md gate a diagnostics)
inline void pf_adump(const Prefill& P, int i, const char* what, const void* dev, int n, size_t row_bytes, cudaStream_t st)
{
    const char* d = getenv("G6_PF_ADUMP");
    if (!d || !*d || i != 0 || n <= 0) return;
    static std::map<std::string, int> done;
    const std::string f = std::string(d) + "/" + what + ".r" + std::to_string(P.f.rank) + ".bin";
    if (done[f]++) return;
    G6RT_CK(cudaStreamSynchronize(st));
    std::vector<char> h((size_t)n * row_bytes);
    G6RT_CK(cudaMemcpy(h.data(), dev, h.size(), cudaMemcpyDeviceToHost));
    FILE* o = fopen(f.c_str(), "wb");
    if (!o || fwrite(h.data(), 1, h.size(), o) != h.size()) die("G6_PF_ADUMP: cannot write " + f);
    fclose(o);
}
#if G6_NR == 4
#include "qp.cuh"
#else
inline void pf_qp_alloc(Prefill&, Model&, int, const g6::FabricDev&, Prefill*) { die("G6_PF_OWN=4 is TP4 only (G6_NR=4)"); }
inline void pf_qp_states(Prefill&, Model&, cudaStream_t, Prefill*) { die("G6_PF_OWN=4 is TP4 only (G6_NR=4)"); }
inline void pf_qp_warm(Model&, Prefill&, cudaStream_t) { die("G6_PF_OWN=4 is TP4 only (G6_NR=4)"); }
inline void pf_attn_block_qp(Model&, Prefill&, int, int, int, cudaStream_t, cudaEvent_t, cudaEvent_t) { die("G6_PF_OWN=4 is TP4 only (G6_NR=4)"); }
inline void pf_qp_norm(Prefill&, int, const __nv_bfloat16*, __half*, cudaStream_t) { die("G6_PF_OWN=4 is TP4 only (G6_NR=4)"); }
inline bool pf_qp_select(Prefill&, Prefill*, int, int) { die("G6_PF_OWN=4 is TP4 only (G6_NR=4)"); return false; }
inline int pf_qp_min() { return 0; }
#endif
// G6_PF_OWN=3: layer i's attention block, owner rows r0..r1 (QO-INTEGRATION.md section 1). Ends with the owner's
// residual + norm and the c1 publish (no c0). wait_attn / used_attn as in pf_attn_block.
inline void pf_attn_block_qo(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st, cudaEvent_t wait_attn,
                             cudaEvent_t used_attn)
{
    Layer& y = m.layers[i];
    const bindp::Layer& b = pf_layer(m, P, i, M);
    const int r0 = pf_r0(M, P.f.rank, P.f.nr), r1 = pf_r0(M, P.f.rank + 1, P.f.nr), a0 = r0 & ~15, Ro = r1 - r0;
    const int ctx_own = P.perm ? context : context - (M - r1);   // last owner position + 1 (perm: rows span the sub-batch)
    pf_mark(7, st);
    const bool qp = pf_prof().on && P.f.rank == 0;
    QoProf& Q = qo_prof();
    if (qp) { Q.tick(st, i, -1); Q.calls[i]++; }
    if (i == 0)                  // later layers: q_a_n / kv_a / idx_k / idx_w of the owner rows from the c2 owner pass,
    {                            // kv_a / idx_k of the other rows from c3
        bindp::launch(b.xq_x, st);
        bindp::launch_pair(b.q_a, b.kv_a, st);
        pf_blas_x(m, P, i, M, st);
        glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(P.q_a, QA, QA, y.q_a_norm, glue::RMS_EPS, P.q_a_n, QA);
    }
    std::pair<bindp::Gemm, bindp::Gemm>& g = P.qg[M];
    if (Ro > 0 && !g.first.on)
    {
        const QoW& w = (*P.qow)[i];
        if (const char* e = bindp::make_gemm(w.qb_c, w.qb_s, r1 - a0, P.sc.x8_small, P.qo_qb, 0, g.first)) die(std::string("QO q_b: ") + e);
        if (const char* e = bindp::make_gemm(w.o_c, w.o_s, Ro, P.qo_x8, P.qo_o, 2, g.second)) die(std::string("QO o_proj: ") + e);
    }
    if (Ro > 0)
    {
        const bindp::Layer& bo = pf_layer_own(m, P, i, M);  // rows a0..r1: q_a_n records, idx_q
        bindp::launch(bo.xq_qa, st);
        if (y.indexer) bindp::launch(bo.idx_q, st);
        bindp::Gemm qb = g.first;                           // the full-head weights of THIS layer (the cached Gemm keeps
        const QoW& w = (*P.qow)[i];                         //   only the shapes / buffers of the row count)
        qb.w = static_cast<const uint8_t*>(w.qb_c.p); qb.ws = static_cast<const uint8_t*>(w.qb_s.p);
        bindp::launch(qb, st);
    }
    if (qp) Q.tick(st, i, 0);
    pf_mark(0, st);
    if (wait_attn) G6RT_CK(cudaStreamWaitEvent(st, wait_attn, 0));
    // every row's latent KV (and index keys on a full-index layer) into the replicated caches, in attention-sized chunks
    for (int c0 = 0; c0 < M; c0 += P.attn_rows)
    {
        const int r = std::min(P.attn_rows, M - c0);
        G6Attn a = y.ab.args;
        a.rows = r; a.context = P.perm ? context : context - (M - c0 - r);
        a.kv_a = P.kv_a + (size_t)c0 * KVA; a.positions = P.pos + c0;
        a.reuse_indices = P.idx_all;                        // non-null: latent KV only
        G6RT_CK(g6_attn_append(&a, st));
        if (y.indexer)
        {
            G6AttnStep s{};
            s.rows = r; s.context = a.context; s.positions = P.pos + c0; s.index_k = P.idx_k + (size_t)c0 * IDXK;
            G6RT_CK(g6_attn_bind_index_append(&y.ab, &s, st));
        }
    }
    if (y.indexer)               // owner rows' top-2048 into idx_all (reuse layers read them there)
        for (int c0 = r0; c0 < r1; c0 += P.attn_rows)
        {
            const int r = std::min(P.attn_rows, r1 - c0);
            G6AttnStep s{};
            s.rows = r; s.context = P.perm ? context : context - (M - c0 - r); s.positions = P.pos + c0;
            s.index_q = P.idx_q + (size_t)c0 * IDXQ; s.index_weight = P.idx_w + (size_t)c0 * IDXW;
            G6RT_CK(g6_attn_bind_prefill_index_rows(P.av[i], &y.ab, &s, P.idx_all + (size_t)c0 * 2048, nullptr, st));
        }
    if (qp) Q.tick(st, i, 1);
    pf_mark(y.indexer ? 8 : 1, st);
    if (Ro > 0)
    {
        G6AttnQoStep s{};
        s.rows = Ro; s.context = ctx_own;
        s.q_b = P.qo_qb + (size_t)(r0 - a0) * 16384; s.ld_qb = 16384;
        s.positions = P.pos + r0; s.ids = P.idx_all + (size_t)r0 * 2048; s.out = P.qo_out;
        if (qp) g6_attn_qo_prof(1, &Q.t[i][2]);
        if (int e = g6_attn_qo_run(P.qa[i], &y.ab, &s, st)) die("g6_attn_qo_run failed (" + std::to_string(e) + ")");
        if (qp) { g6_attn_qo_prof(0, nullptr); Q.tick(st, i, -1); }
        pf_adump(P, i, "attn", P.qo_out, Ro, 16384 * 2, st);
    }
    if (used_attn) G6RT_CK(cudaEventRecord(used_attn, st));
    pf_mark(y.indexer ? 8 : 1, st);
    if (Ro > 0)
    {
        bindp::Xq xq;
        xq.on = 1; xq.src = P.qo_out; xq.ld = 16384; xq.K = 16384; xq.M = Ro; xq.dst = P.qo_x8;
        bindp::launch(xq, st);
        bindp::Gemm op = g.second;
        const QoW& w = (*P.qow)[i];
        op.w = static_cast<const uint8_t*>(w.o_c.p); op.ws = static_cast<const uint8_t*>(w.o_s.p);
        bindp::launch(op, st);
        pf_adump(P, i, "oproj", P.qo_o, Ro, HID * 4, st);
        if (getenv("G6_PF_ADUMP") && i == 0)                 // diagnostic: the same GEMM with an FP16 output (8x finer)
        {
            static __half* o16 = nullptr;
            if (!o16) o16 = dmalloc<__half>((size_t)P.qo_rows * HID);
            bindp::Gemm g16 = op;
            g16.bf16 = 0; g16.y = o16;
            bindp::launch(g16, st);
            pf_adump(P, i, "oproj16", o16, Ro, HID * 2, st);
        }
    }
    if (qp) Q.tick(st, i, 6);
    pf_mark(2, st);
    RsNormArgs a{P.f, P.stepp, 0, M, P.loop, P.resid, y.mlp_norm, P.x2, glue::RMS_EPS, P.tickets + 1};
    k_own_add_norm<<<dim3(glue::RN_CL, Ro > 0 ? Ro : 1), glue::RN_T, 0, st>>>(a, P.qo_o);
    if (qp) Q.tick(st, i, 7);
    pf_mark(4, st);
}
// context = last position of the sub-batch + 1
// cache_only: stop after the attention call (the MTP layer's prefill pass needs only its caches written)
// two-stream lanes (pf_step2s): wait_attn = the other lane is done with this layer's attention binding (scratch,
// output, index list); used_attn = recorded once this lane is done with it (after xq_attn read the output)
inline void pf_attn_block(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st, bool cache_only = false,
                          cudaEvent_t wait_attn = nullptr, cudaEvent_t used_attn = nullptr)
{
    if (P.qo && !cache_only) { pf_attn_block_qo(m, P, i, M, context, st, wait_attn, used_attn); return; }
    if (P.qp && P.qpl[i] && !cache_only) { pf_attn_block_qp(m, P, i, M, context, st, wait_attn, used_attn); return; }
    Layer& y = m.layers[i];
    const bindp::Layer& b = pf_layer(m, P, i, M);
    pf_mark(7, st);
    if (!(P.own && i > 0))       // G6_PF_OWN: layers after the first got q_a_n / kv_a / idx_k / idx_w from c3
    {
        bindp::launch(b.xq_x, st);
        bindp::launch_pair(b.q_a, b.kv_a, st);
        pf_blas_x(m, P, i, M, st);
        glue::k_rmsnorm_rows<__half><<<M, 256, 0, st>>>(P.q_a, QA, QA, y.q_a_norm, glue::RMS_EPS, P.q_a_n, QA);
    }
    bindp::launch(b.xq_qa, st);
    bindp::launch(b.q_b, st);
    if (!(P.ownix && y.indexer && i > 0)) bindp::launch(b.idx_q, st);   // G6_PF_OWN=2: owner rows only, in pf_ix
    pf_mark(0, st);
    // attention in chunks of P.attn_rows rows (one chunk unless G6_PF_ATTN_ROWS < the sub-batch): chunk k sees the
    // context up to its own last row; a chunked lane copies each chunk's output and index list out of the binding
    const int H = b.heads, AR = P.attn_rows;
    const bool rix = P.ownix && y.indexer && i > 0;      // G6_PF_OWN=2: IDs gathered into idx_all by the previous consume
    if (wait_attn) G6RT_CK(cudaStreamWaitEvent(st, wait_attn, 0));
    for (int c0 = 0; c0 < M; c0 += AR)
    {
        const int r = std::min(AR, M - c0);
        G6AttnStep s{};
        s.rows = r; s.context = context - (M - c0 - r);
        s.q_b = P.q_b + (size_t)c0 * H * 256; s.kv_a = P.kv_a + (size_t)c0 * KVA; s.positions = P.pos + c0;
        if (y.indexer && !rix) { s.index_q = P.idx_q + (size_t)c0 * IDXQ; s.index_k = P.idx_k + (size_t)c0 * IDXK; s.index_weight = P.idx_w + (size_t)c0 * IDXW; }
        else if (P.attn_out || P.ownix) { s.reuse_indices = P.idx_all + (size_t)c0 * 2048; s.reuse_rows = r; }
        else { s.reuse_indices = P.idx_keep ? P.idx_keep : m.layers[y.index_src].ab.args.indices; s.reuse_rows = M; }
        if (!P.pd_dir.empty() && P.pd_lay[i]) pf_pdump(m, P, i, s, st);
        else if (P.v2) G6RT_CK(g6_attn_bind_prefill_run(P.av[i], &y.ab, &s, st));
        else G6RT_CK(g6_attn_bind_run(&y.ab, &s, st));
        if (!P.attn_out && !P.ownix) continue;
        if (y.indexer && !rix)
            G6RT_CK(cudaMemcpyAsync(P.idx_all + (size_t)c0 * 2048, y.ab.args.indices, (size_t)r * 2048 * sizeof(int), cudaMemcpyDeviceToDevice, st));
        if (!cache_only && P.attn_out)
            G6RT_CK(cudaMemcpyAsync(P.attn_out + (size_t)c0 * H * 256, y.ab.args.output, (size_t)r * H * 256 * sizeof(__half), cudaMemcpyDeviceToDevice, st));
    }
    if (y.indexer && P.idx_keep && !P.ownix)
        G6RT_CK(cudaMemcpyAsync(P.idx_keep, y.ab.args.indices, (size_t)M * 2048 * sizeof(int), cudaMemcpyDeviceToDevice, st));
    if (y.indexer && !P.dump_dir.empty() && std::find(P.dump.begin(), P.dump.end(), i) != P.dump.end())
    {
        // measurement only (synchronizes the stream): <dir>/L<layer>.p<first position>.{ids,pos}, int32 [M][2048] / [M]
        std::vector<int> ids((size_t)M * 2048), pos(M);
        G6RT_CK(cudaStreamSynchronize(st));
        G6RT_CK(cudaMemcpy(ids.data(), P.idx_all ? P.idx_all : y.ab.args.indices, ids.size() * 4, cudaMemcpyDeviceToHost));
        G6RT_CK(cudaMemcpy(pos.data(), P.pos, pos.size() * 4, cudaMemcpyDeviceToHost));
        const std::string b = P.dump_dir + "/L" + std::to_string(y.l) + ".p" + std::to_string(pos[0]);
        FILE* fi = fopen((b + ".ids").c_str(), "wb");
        FILE* fp = fopen((b + ".pos").c_str(), "wb");
        if (!fi || !fp) die("G6_PF_IDDUMP: cannot write " + b);
        fwrite(ids.data(), 4, ids.size(), fi); fwrite(pos.data(), 4, pos.size(), fp);
        fclose(fi); fclose(fp);
        if (FILE* fm = fopen((P.dump_dir + "/meta.jsonl").c_str(), "a"))
        {
            fprintf(fm, "{\"layer\": %d, \"first\": %d, \"rows\": %d, \"context\": %d, \"v2\": %d, \"tile\": \"%s\", \"dense\": \"%s\"}\n",
                    y.l, pos[0], M, context, (int)P.v2, getenv("G6_PF_TILE") ? getenv("G6_PF_TILE") : "",
                    getenv("G6_PF_DENSE") ? getenv("G6_PF_DENSE") : "");
            fclose(fm);
        }
    }
    pf_mark(y.indexer ? 8 : 1, st);
    if (cache_only) return;
    bindp::launch(b.xq_attn, st);
    if (used_attn) G6RT_CK(cudaEventRecord(used_attn, st));
    bindp::launch(b.o_proj, st, b.rs_o);
    if (P.attn_out) pf_adump(P, i, "attn", P.attn_out, M, (size_t)H * 256 * 2, st);
    pf_adump(P, i, "oproj", P.part[0], M, HID * 2, st);
    pf_mark(2, st);
    pf_put(P, 0, M, st);
    pf_mark(3, st);
}
// RS form in two halves (a loopback driver delivers collective 2c + 1 between them): owner rows, then the gather
inline void pf_consume_rs(Prefill& P, int c, int M, const __nv_bfloat16* w, __half* out, cudaStream_t st, bool ag = true)
{
    if (c == 0 && P.qp_pend >= 0) { pf_qp_norm(P, M, w, out, st); return; }   // G6_PF_OWN=4: the pair norm (qp.cuh)
    const int cr = 2 * c, rows = pf_rows(M, P.f.rank, P.f.nr);
    k_wait_rs<<<1, 32, 0, st>>>(P.f, P.stepp, cr, M, P.loop);
    RsNormArgs a{P.f, P.stepp, cr, M, P.loop, P.resid, w, out, glue::RMS_EPS, P.tickets + cr + 1};
    a.ag = ag;
#ifdef PF_RNWIDE
    k_rs_add_norm_w<<<rows > 0 ? rows : 1, glue::RN_CL * glue::RN_T, 0, st>>>(a);
#else
    k_rs_add_norm<<<dim3(glue::RN_CL, rows > 0 ? rows : 1), glue::RN_T, 0, st>>>(a);
#endif
}
inline void pf_consume_ag(Prefill& P, int c, int M, __half* out, cudaStream_t st)
{
    pf_ag(P.f, P.stepp, 2 * c + 1, M, HID * 2, reinterpret_cast<char*>(out), st);
}
inline void pf_consume(Prefill& P, int c, int M, const __nv_bfloat16* w, __half* out, cudaStream_t st)
{
    pf_mark(7, st);
    if (P.qo && c == 0)          // G6_PF_OWN=3: the owner normed its rows in pf_attn_block_qo; only the c1 gather
    {
        pf_consume_ag(P, 0, M, out, st);
        pf_mark(4, st);
        return;
    }
    if (P.rs)
    {
        pf_consume_rs(P, c, M, w, out, st);
        pf_consume_ag(P, c, M, out, st);
        pf_mark(4, st);
        return;
    }
    glue::ArNormArgs a{P.f, P.stepp, c, M, P.resid, w, out, glue::RMS_EPS};
    glue::k_wait_peers<<<1, 32, 0, st>>>(P.f, P.stepp, c, (size_t)M * HID * 2, glue::ERR_AR);
    glue::k_ar_add_norm<<<dim3(glue::RN_CL, M), glue::RN_T, 0, st>>>(a);
    pf_mark(4, st);
}
// ---- the consume of the MLP partial of layer i (collectives c2 / c3 in the RS form), G6_PF_OWN aware
inline bool pf_own_next(const Model& m, const Prefill& P, int i) { return P.own && i + 1 < m.nl; }
inline int pf_x_row_bytes(const Model& m, const Prefill& P, int i) { return pf_own_next(m, P, i) ? (P.qo ? QO_REC : OWN_REC) : HID * 2; }
inline OwnRec pf_own_rec(Prefill& P) { return OwnRec{P.q_a_n, P.kv_a, P.idx_k, P.idx_w}; }
// RS half: owner rows of x; G6_PF_OWN: then layer i + 1's projections on those rows and their records into c3
inline void pf_consume_x_rs(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    const bool own = pf_own_next(m, P, i);
    pf_consume_rs(P, 1, M, norm_after(m, i), P.x, st, !own);
    if (!own) return;
    pf_mark(4, st);
    const int r0 = pf_r0(M, P.f.rank, P.f.nr), rows = pf_rows(M, P.f.rank, P.f.nr);
    if (rows > 0)
    {
        const bindp::Layer& b = pf_layer_own(m, P, i + 1, M);
        bindp::launch(b.xq_x, st);
        bindp::launch_pair(b.q_a, b.kv_a, st);
        pf_blas_x(m, P, i + 1, M, st, r0, r0 + rows);        // same cuBLAS shapes as the non-owner path; owner rows kept
        glue::k_rmsnorm_rows<__half><<<rows, 256, 0, st>>>(P.q_a + (size_t)r0 * QA, QA, QA, m.layers[i + 1].q_a_norm, glue::RMS_EPS,
                                                           P.q_a_n + (size_t)r0 * QA, QA);
    }
    pf_mark(0, st);
    if (P.qo) k_own_put<true><<<PF_PUT, 128, 0, st>>>(pf_own_rec(P), P.f, P.stepp, 3, M, P.tickets + 3);
    else k_own_put<<<PF_PUT, 128, 0, st>>>(pf_own_rec(P), P.f, P.stepp, 3, M, P.tickets + 3);
}
// gather half: every row of x, or (G6_PF_OWN) every row's q_a_n / kv_a / idx_k / idx_w of layer i + 1
inline void pf_consume_x_ag(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    if (!pf_own_next(m, P, i)) { pf_consume_ag(P, 1, M, P.x, st); return; }
    k_wait_ag<<<1, 32, 0, st>>>(P.f, P.stepp, 3, M, P.qo ? QO_REC : OWN_REC);
    if (P.qo) k_own_unpack<true><<<dim3(PF_AG, P.f.nr), 256, 0, st>>>(pf_own_rec(P), P.f, P.stepp, 3, M);
    else k_own_unpack<<<dim3(PF_AG, P.f.nr), 256, 0, st>>>(pf_own_rec(P), P.f, P.stepp, 3, M);
}
inline void pf_consume_x(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    if (!P.rs) { pf_consume(P, 1, M, norm_after(m, i), P.x, st); return; }
    pf_mark(7, st);
    pf_consume_x_rs(m, P, i, M, st);
    pf_consume_x_ag(m, P, i, M, st);
    pf_mark(4, st);
}
// G6_PF_OWN=2, full-index layer i + 1, still layer-step i (before the next pf_set_step): idx_q of the owner rows, the
// index-key append of all rows, the owner's top-2048, its IDs into c4. wait_ix: the other lane is done with the shared
// attention state (two-stream lanes; the event must already be recorded on the host side).
inline bool pf_ix_next(const Model& m, const Prefill& P, int i) { return P.ownix && i + 1 < m.nl && m.layers[i + 1].indexer; }
constexpr int IX_ROW = 2048 * 4;
inline void pf_ix_score(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st, cudaEvent_t wait_ix = nullptr)
{
    if (!pf_ix_next(m, P, i)) return;
    const int j = i + 1, r0 = pf_r0(M, P.f.rank, P.f.nr), r1 = pf_r0(M, P.f.rank + 1, P.f.nr);
    Layer& y = m.layers[j];
    pf_mark(7, st);
    if (r1 > r0)
    {
        const bindp::Layer& b = pf_layer_own(m, P, j, M);
        bindp::launch(b.xq_qa, st);
        bindp::launch(b.idx_q, st);
    }
    pf_mark(0, st);
    if (wait_ix) G6RT_CK(cudaStreamWaitEvent(st, wait_ix, 0));
    for (int c0 = 0; c0 < M; c0 += P.attn_rows)          // index keys of every row, in attention-sized chunks
    {
        const int r = std::min(P.attn_rows, M - c0);
        G6AttnStep s{};
        s.rows = r; s.context = context - (M - c0 - r); s.positions = P.pos + c0; s.index_k = P.idx_k + (size_t)c0 * IDXK;
        G6RT_CK(g6_attn_bind_index_append(&y.ab, &s, st));
    }
    G6AttnStep s{};
    s.rows = r1 - r0; s.context = context - (M - r1); s.positions = P.pos + r0;
    s.index_q = P.idx_q + (size_t)r0 * IDXQ; s.index_weight = P.idx_w + (size_t)r0 * IDXW;
    G6RT_CK(g6_attn_bind_prefill_index_rows(P.av[j], &y.ab, &s, P.idx_all + (size_t)r0 * 2048, nullptr, st));
    k_put_publish_n<<<PF_PUT, 128, 0, st>>>(reinterpret_cast<const uint4*>(P.idx_all + (size_t)r0 * 2048), P.f, P.stepp, 4,
                                            (r1 - r0) * (IX_ROW / 16), P.tickets + 4);
    pf_mark(8, st);
}
// every owner's IDs of full-index layer i + 1 -> idx_all
inline void pf_ix_gather(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    if (!pf_ix_next(m, P, i)) return;
    pf_ag(P.f, P.stepp, 4, M, IX_ROW, reinterpret_cast<char*>(P.idx_all), st);
    pf_mark(4, st);
}
inline void pf_ix(Model& m, Prefill& P, int i, int M, int context, cudaStream_t st, cudaEvent_t wait_ix = nullptr)
{
    pf_ix_score(m, P, i, M, context, st, wait_ix);
    pf_ix_gather(m, P, i, M, st);
}
// RS form, after the last layer (same layer-step): c4 = this rank's FP32 residual rows to every peer, then the gather,
// so P.resid holds every row again (pf_hash). No-op in the all-reduce form.
inline void pf_finish_put(Prefill& P, int M, cudaStream_t st)
{
    if (!P.rs) return;
    const int me = P.f.rank, n16 = pf_rows(M, me, P.f.nr) * (HID * 4 / 16);
    k_put_publish_n<<<PF_PUT, 128, 0, st>>>(reinterpret_cast<const uint4*>(P.resid + (size_t)pf_r0(M, me, P.f.nr) * HID), P.f,
                                            P.stepp, 4, n16, P.tickets + 4);
}
template <class T> __global__ void k_pf_unperm(const T* src, T* dst, const int* ord, int n)   // grid M: dst row ord[j] = src row j
{
    const T* s = src + (size_t)blockIdx.x * n;
    T* d = dst + (size_t)ord[blockIdx.x] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}
inline void pf_finish_gather(Prefill& P, int M, cudaStream_t st)
{
    if (!P.rs) { pf_mark(7, st); return; }
    pf_ag(P.f, P.stepp, 4, M, HID * 4, reinterpret_cast<char*>(P.resid), st);
    if (!P.perm) { pf_mark(7, st); return; }     // G6_PF_OWNPERM: resid (hash, dump) and the final-norm x (MTP draft) back in input order
    G6RT_CK(cudaMemcpyAsync(P.perm_tmp, P.resid, (size_t)M * HID * 4, cudaMemcpyDeviceToDevice, st));
    k_pf_unperm<<<M, 256, 0, st>>>(P.perm_tmp, P.resid, P.ord, HID);
    __half* xt = reinterpret_cast<__half*>(P.perm_tmp);
    G6RT_CK(cudaMemcpyAsync(xt, P.x, (size_t)M * HID * 2, cudaMemcpyDeviceToDevice, st));
    k_pf_unperm<<<M, 256, 0, st>>>(xt, P.x, P.ord, HID);
    pf_mark(7, st);
}
inline void pf_finish(Prefill& P, int M, cudaStream_t st) { pf_finish_put(P, M, st); pf_finish_gather(P, M, st); pf_mark(7, st); }
// host: the proxy's jobs of one sub-batch (layer-steps ls0 .. ls0 + nl - 1) on a real fabric
template <class Fab>
inline void pf_enqueue(Fab& fab, const Prefill& P, u64 ls0, int nl, int M)
{
    if (!P.rs)
    {
        const std::vector<uint32_t> sizes(2, (uint32_t)(M * HID * 2));
        for (int i = 0; i < nl; ++i) fab.enqueue_step(ls0 + i, sizes);
        return;
    }
    const int nr = P.f.nr, me = P.f.rank;
    typename Fab::Rs rs;
    rs.mask = P.qo ? 0x4 : 0x5;          // c0, c2 (G6_PF_OWN=3: c2 only)
    rs.skip = P.qo ? 0x1 : 0;            // G6_PF_OWN=3: no collective 0 (nothing publishes it; the owner norms locally)
    rs.ss = pf_ss(P.f);
    for (int p = 0; p < nr; ++p) rs.bytes.push_back((uint32_t)(pf_rows(M, p, nr) * HID * 2));
    const uint32_t mine = (uint32_t)(pf_rows(M, me, nr) * HID * 2);
    for (int i = 0; i < nl; ++i)
    {
        const uint32_t c3 = P.own && i < nl - 1 ? (uint32_t)(pf_rows(M, me, nr) * (P.qo ? QO_REC : OWN_REC)) : mine;
        std::vector<uint32_t> sizes{0, mine, 0, c3};
        if (i == nl - 1) sizes.push_back(2 * mine);                  // c4: FP32 residual rows
        else if (P.ownix && P.ixl[i + 1]) sizes.push_back((uint32_t)(pf_rows(M, me, nr) * IX_ROW));   // c4: owner IDs
        typename Fab::Rs ri = rs;
        if (P.qp && P.qpl[i]) { ri.p2p = 0x1; ri.peer = me ^ 2; }   // G6_PF_OWN=4: c0 = O_c's partial, to c only
        fab.enqueue_step(ls0 + i, sizes, ri);
    }
}
// G6_PF_RCDUMP (measurement only, synchronizes the stream): after a MoE layer's routing, one line
// {"layer", "first" (position of the sub-batch's first row), "rows", "rank", "cnt": [256] routes per expert} (k_pf_count)
inline void pf_rcdump(Model& m, Prefill& P, int i, int M, const int* cnt, cudaStream_t st)
{
    int c[glue::NEXP], first = 0;
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaMemcpy(c, cnt, sizeof(c), cudaMemcpyDeviceToHost));
    G6RT_CK(cudaMemcpy(&first, P.pos, 4, cudaMemcpyDeviceToHost));
    FILE* f = fopen(P.rc_file.c_str(), "a");
    if (!f) die("G6_PF_RCDUMP: cannot write " + P.rc_file);
    fprintf(f, "{\"layer\": %d, \"first\": %d, \"rows\": %d, \"rank\": %d, \"cnt\": [", m.layers[i].l, first, M, m.rank);
    for (int e = 0; e < glue::NEXP; ++e) fprintf(f, e ? ",%d" : "%d", c[e]);
    fprintf(f, "]}\n");
    fclose(f);
}
inline void pf_mlp_block(Model& m, Prefill& P, int i, int M, cudaStream_t st)
{
    const bindp::Layer& b = pf_layer(m, P, i, M);
    pf_mark(7, st);
    // An exlr8 v2 layer whose shared expert runs from its FP8 copy (Layer::shx) launches these five itself,
    // inside launch_shared, after k_shx_rotq has written the rotated gate / up records. Running them here as
    // well read sc.x8_big / sc.x8_small as the previous stage left them and was overwritten by that pass:
    // same output, one shared gate + up + down GEMM per MoE layer of pure waste.
    if (!b.shx.on)
    {
        bindp::launch(b.xq_x2, st);
        bindp::launch_pair(b.gate, b.up, st);
        bindp::launch(b.act, st);
        bindp::launch(b.xq_act, st);
        bindp::launch(b.down, st, b.rs_down);
    }
    pf_mark(5, st);
    pf_blas_x2(m, P, i, M, st);
    bindp::launch_shared(b, st);                 // exlr8 v2 trellis shared expert; no-op otherwise
    bindp::launch_experts(b.ex, st);
    if (!P.rc_file.empty() && b.ex.on) pf_rcdump(m, P, i, M, b.ex.cnt, st);
    pf_mark(6, st);
    pf_put(P, 1, M, st);
    pf_mark(3, st);
}
// the whole sub-batch of one rank in stream order (real fabric); layer i runs as layer-step ls0 + i
#if defined(G6_CP) && G6_CP
// CP prefill staging over the real fabric (defined below); no-ops unless the model is staged
inline void cp_pf_pre(Model& m, int i, int start, cudaStream_t st);
inline void cp_pf_prefetch(Model& m, int i, int start, cudaStream_t st);
inline void cp_pf_post(Model& m, int i, int a0, int n0, int a1, int n1, cudaStream_t st);
#else
inline void cp_pf_pre(Model&, int, int, cudaStream_t) {}
inline void cp_pf_prefetch(Model&, int, int, cudaStream_t) {}
inline void cp_pf_post(Model&, int, int, int, int, int, cudaStream_t) {}
#endif
inline void pf_step(Model& m, Prefill& P, int M, int context, u64 ls0, cudaStream_t st)
{
    pf_prof().last = 0;
    pf_embed(m, P, M, st);
    pf_mark(7, st);
    for (int i = 0; i < m.nl; ++i)
    {
        pf_set_step(P, ls0 + i, st);
        cp_pf_pre(m, i, context - M, st);
        pf_attn_block(m, P, i, M, context, st);
        cp_pf_prefetch(m, i, context - M, st);
        cp_pf_post(m, i, context - M, M, 0, 0, st);
        pf_consume(P, 0, M, m.layers[i].mlp_norm, P.x2, st);
        pf_mlp_block(m, P, i, M, st);
        pf_consume_x(m, P, i, M, st);
        if (P.dfl_tap) P.dfl_tap(m, P, i, M, st);
        pf_ix(m, P, i, M, context, st);
    }
    pf_finish(P, M, st);
}
// Two sub-batches (A = the earlier rows, B = the next ones) as two lanes, each with its own Prefill (scratch, fabric
// region, step cell), interleaved so that every partial travels while the other lane computes: a consume comes one
// block of the other lane after its publish. Same kernels and the same per-row arithmetic as pf_step twice (A, then
// B); the shared attention bindings see A's rows of a layer before B's.
inline void pf_step2(Model& m, Prefill& A, Prefill& B, int MA, int MB, int ctxA, int ctxB, u64 lsA, u64 lsB, cudaStream_t st)
{
    pf_prof().last = 0;
    pf_embed(m, A, MA, st);
    pf_mark(7, st);
    pf_embed(m, B, MB, st);
    pf_mark(7, st);
    for (int i = 0; i < m.nl; ++i)
    {
        pf_set_step(A, lsA + i, st);
        cp_pf_pre(m, i, ctxA - MA, st);
        pf_attn_block(m, A, i, MA, ctxA, st);
        cp_pf_prefetch(m, i, ctxA - MA, st);
        if (i > 0) { pf_consume_x(m, B, i - 1, MB, st); if (B.dfl_tap) B.dfl_tap(m, B, i - 1, MB, st); pf_ix(m, B, i - 1, MB, ctxB, st); }
        pf_set_step(B, lsB + i, st);
        pf_attn_block(m, B, i, MB, ctxB, st);
        cp_pf_post(m, i, ctxA - MA, MA, ctxB - MB, MB, st);
        pf_consume(A, 0, MA, m.layers[i].mlp_norm, A.x2, st);
        pf_mlp_block(m, A, i, MA, st);
        pf_consume(B, 0, MB, m.layers[i].mlp_norm, B.x2, st);
        pf_mlp_block(m, B, i, MB, st);
        pf_consume_x(m, A, i, MA, st);
        if (A.dfl_tap) A.dfl_tap(m, A, i, MA, st);
        pf_ix(m, A, i, MA, ctxA, st);
    }
    pf_finish_put(A, MA, st);
    pf_consume_x(m, B, m.nl - 1, MB, st);
    if (B.dfl_tap) B.dfl_tap(m, B, m.nl - 1, MB, st);
    pf_finish(B, MB, st);
    pf_finish_gather(A, MA, st);
}
// Two sub-batches as two lanes on two streams (G6_PF_STREAMS=2 in g6m4): each lane runs pf_step's order on its own
// stream, so a lane waiting on the fabric does not hold up the other lane. The only state the lanes share is each
// layer's attention binding (tensor-core scratch, output, index list): B's attention of layer i waits until A is done
// with it (A.xq_attn), A's attention of layer i + 1 waits until B is done with layer i. That is also the causal
// order of the caches (B's rows are the later positions). Same kernels and per-row arithmetic as pf_step2.
struct PfLanes
{
    cudaStream_t sb = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    std::vector<cudaEvent_t> ea, eb;       // per layer: lane A / lane B done with the attention binding
};
inline void pf_lanes_init(PfLanes& L, int nl)
{
    G6RT_CK(cudaStreamCreateWithFlags(&L.sb, cudaStreamNonBlocking));
    G6RT_CK(cudaEventCreateWithFlags(&L.fork, cudaEventDisableTiming));
    G6RT_CK(cudaEventCreateWithFlags(&L.join, cudaEventDisableTiming));
    L.ea.resize(nl); L.eb.resize(nl);
    for (int i = 0; i < nl; ++i)
    {
        G6RT_CK(cudaEventCreateWithFlags(&L.ea[i], cudaEventDisableTiming));
        G6RT_CK(cudaEventCreateWithFlags(&L.eb[i], cudaEventDisableTiming));
    }
}
// G6_PF_STAGGER=<us> (PEAK exposed-comm row, start ramp): lane B's stream first runs a one-thread sleep of that many
// microseconds, so its embed, projections and first collectives start after lane A's. Writes nothing (outputs are
// unchanged); the lanes stay coupled per layer through the attention binding, so only the slack before B's first
// attention moves. Default off (unset or 0): no launch, today's path.
__global__ void k_pf_stagger(unsigned long long ns)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { __nanosleep(1000); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
inline void pf_stagger(cudaStream_t st)
{
    static const long us = getenv("G6_PF_STAGGER") ? atol(getenv("G6_PF_STAGGER")) : 0;
    if (us > 0) k_pf_stagger<<<1, 1, 0, st>>>(static_cast<unsigned long long>(us) * 1000);
}
inline void pf_step2s(Model& m, Prefill& A, Prefill& B, int MA, int MB, int ctxA, int ctxB, u64 lsA, u64 lsB, cudaStream_t sa,
                      PfLanes& L)
{
    pf_prof().last = 0;
    cudaStream_t sb = L.sb;
    G6RT_CK(cudaEventRecord(L.fork, sa));
    G6RT_CK(cudaStreamWaitEvent(sb, L.fork, 0));
    pf_embed(m, A, MA, sa);
    pf_mark(7, sa);
    pf_stagger(sb);
    pf_embed(m, B, MB, sb);
    pf_mark(7, sb);
    for (int i = 0; i < m.nl; ++i)
    {
        if (i > 0) pf_ix(m, A, i - 1, MA, ctxA, sa, L.eb[i - 1]);      // G6_PF_OWN=2: layer i's IDs (layer-step i - 1)
        pf_set_step(A, lsA + i, sa);
        cp_pf_pre(m, i, ctxA - MA, sa);
        pf_attn_block(m, A, i, MA, ctxA, sa, false, i > 0 ? L.eb[i - 1] : nullptr, L.ea[i]);
        cp_pf_prefetch(m, i, ctxA - MA, sa);
        pf_consume(A, 0, MA, m.layers[i].mlp_norm, A.x2, sa);
        pf_mlp_block(m, A, i, MA, sa);
        pf_consume_x(m, A, i, MA, sa);
        if (A.dfl_tap) A.dfl_tap(m, A, i, MA, sa);
        if (i > 0) pf_ix(m, B, i - 1, MB, ctxB, sb, L.ea[i]);
        pf_set_step(B, lsB + i, sb);
        pf_attn_block(m, B, i, MB, ctxB, sb, false, L.ea[i], L.eb[i]);
        cp_pf_post(m, i, ctxA - MA, MA, ctxB - MB, MB, sb);   // after B: A's rows are in the pool (B waited A.ea[i])
        pf_consume(B, 0, MB, m.layers[i].mlp_norm, B.x2, sb);
        pf_mlp_block(m, B, i, MB, sb);
        pf_consume_x(m, B, i, MB, sb);
        if (B.dfl_tap) B.dfl_tap(m, B, i, MB, sb);
    }
    pf_finish(A, MA, sa);
    pf_finish(B, MB, sb);
    G6RT_CK(cudaEventRecord(L.join, sb));
    G6RT_CK(cudaStreamWaitEvent(sa, L.join, 0));
}
// ---------------------------------------------------------------- MTP layer (78) over prefilled rows
// Layer 78 must see every row the target processed (rank.cuh "MTP draft pass"). For prefilled rows only its attention
// caches matter (nothing is predicted from them), so this pass computes the layer's input rows (eh_proj of
// [enorm(emb(next token)) | hnorm(target hidden)], the target hidden = the sub-batch's final-norm rows in P.x), the
// layer's projections and one attention call that appends the KV and index-key rows; its output is dropped. Local to
// the rank: no collective. v1: the attention call still scores (about one layer's attention of extra time).
struct DraftPrefill
{
    Prefill P;                        // for the draft Model (one layer): scratch, bindings, inputs; its fabric is never used
    __half *eh_in = nullptr, *eh_out = nullptr;      // [Mmax][12288], [Mmax][6144]
    uint8_t* x8 = nullptr;            // MXFP8 records of eh_in
    bind::Tensor codes, scales;       // L078.eh.mx.codes / .scales
    std::map<int, bindp::Gemm> eh;    // by row count
};
// D.d must have been built with max_rows >= Mmax (build_draft(..., max_rows)). f: any prefill-sized region (unused).
// tc: the target model's prefill of the same rank. With G6_PF_DRAFT_SHARE=1 the draft layer's tensor-core prefill
// state borrows that prefill's scratch root (-1 root allocation) instead of allocating a second identical one; the
// draft pass runs on the stream the target prefill was joined and synchronized on, so the uses stay serialized.
inline void build_draft_prefill(DraftPrefill& Q, Draft& D, const Find& find, int Mmax, const g6::FabricDev& f, bool v2, cudaStream_t st,
                                Prefill* tc = nullptr)
{
    Q.P.v2 = v2;
    G6AttnPrefill* root = nullptr;
    if (pf_draft_share())
    {
        if (!v2) die("G6_PF_DRAFT_SHARE=1 needs --pfv2 (without it there is no tensor-core prefill scratch to share)");
        if (!tc || tc->av.empty()) die("G6_PF_DRAFT_SHARE=1: the target prefill has no tensor-core attention states to share");
        root = tc->av.front();
        fprintf(stderr, "# rank %d: G6_PF_DRAFT_SHARE=1: MTP draft prefill borrows the target prefill's tensor-core scratch\n", D.d.rank);
    }
    build_prefill(Q.P, D.d, find, Mmax, f, dmalloc<u64>(1), st, nullptr, root);
    const size_t n = (size_t)Mmax;
    Q.eh_in = dmalloc<__half>(n * 2 * HID);
    Q.eh_out = dmalloc<__half>(n * HID);
    Q.x8 = dmalloc<uint8_t>((n + 15) / 16 * (2 * HID / 32) * bindp::REC);
    Q.codes = as_bind(find("L078.eh.mx.codes"));
    Q.scales = as_bind(find("L078.eh.mx.scales"));
    if (!Q.codes.p || !Q.scales.p) die("pack has no L078.eh.mx.codes / .scales (MTP prefill)");
}
// host -> device: the rows' NEXT tokens and positions (the stream must be idle)
inline void pf_draft_inputs(DraftPrefill& Q, int M, const int* next_tok, const int* pos) { pf_inputs(Q.P, M, next_tok, pos); }
#ifdef PF_DRAFTKV
// PF_DRAFTKV (default off): the MTP layer's prefill output is dropped, so fill only its caches: kv_a (+ idx_k / idx_w) and
// the latent-KV + index-key appends, as pf_attn_block_qo does. pf_attn_block(cache_only) also ran q_a, q_b, idx_q, the
// dense scoring and the top-k for every row (~68 of ~96 ms kernel per rank, results/peak/qehost/TABLE.md).
inline void pf_draft_kv(Model& d, Prefill& P, int M, int context, cudaStream_t st)
{
    Layer& y = d.layers[0];
    const bindp::Layer& b = pf_layer(d, P, 0, M);
    pf_mark(7, st);
    bindp::launch(b.xq_x, st);
    bindp::launch(b.kv_a, st);
    pf_blas_x(d, P, 0, M, st);
    pf_mark(0, st);
    for (int c0 = 0; c0 < M; c0 += P.attn_rows)
    {
        const int r = std::min(P.attn_rows, M - c0);
        G6Attn a = y.ab.args;
        a.rows = r; a.context = context - (M - c0 - r);
        a.kv_a = P.kv_a + (size_t)c0 * KVA; a.positions = P.pos + c0;
        a.reuse_indices = a.positions;                      // non-null: latent KV only (the append never reads it)
        G6RT_CK(g6_attn_append(&a, st));
        if (y.indexer)
        {
            G6AttnStep s{};
            s.rows = r; s.context = a.context; s.positions = P.pos + c0; s.index_k = P.idx_k + (size_t)c0 * IDXK;
            G6RT_CK(g6_attn_bind_index_append(&y.ab, &s, st));
        }
    }
    pf_mark(y.indexer ? 8 : 1, st);
}
#endif
// hid: the target's final-norm rows of the same sub-batch (Prefill::x after pf_step / pf_step2)
inline void pf_draft(Draft& D, DraftPrefill& Q, const __half* hid, int M, int context, cudaStream_t st)
{
    Model& d = D.d;
    Prefill& P = Q.P;
    if (M < 1 || M > P.Mmax) die("MTP prefill sub-batch larger than Mmax");
    bindp::Gemm& g = Q.eh[M];
    if (!g.on)
    {
        if (const char* e = bindp::make_gemm(Q.codes, Q.scales, M, Q.x8, Q.eh_out, 0, g)) die(std::string("MTP prefill eh_proj: ") + e);
        if (g.K != 2 * HID || g.N != HID) die("L078.eh.mx.codes is not [6144][12288]");
    }
    k_mtp_in<<<dim3(2, M), 256, 0, st>>>(MtpInArgs{P.tok, P.pos, d.emb, D.enorm, D.hnorm, hid, glue::RMS_EPS, Q.eh_in});
    bindp::Xq xq;
    xq.on = 1; xq.src = Q.eh_in; xq.ld = 2 * HID; xq.K = 2 * HID; xq.M = M; xq.dst = Q.x8;
    bindp::launch(xq, st);
    bindp::launch(g, st);
    k_mtp_resid_norm<<<dim3(glue::RN_CL, M), glue::RN_T, 0, st>>>(Q.eh_out, d.layers[0].attn_norm, P.resid, P.x, glue::RMS_EPS);
#ifdef PF_DRAFTKV
    pf_draft_kv(d, P, M, context, st);
#else
    pf_attn_block(d, P, 0, M, context, st, true);
#endif
}

// G6_PF_HASH_GPU=1: pf_hash on the GPU = sum over words of splitmix64(index << 32 | word), 8 B back to the host instead of
// the M x HID fp32 rows + a byte-serial FNV (PEAK row 5). splitmix64 is a bijection, so any one changed word changes the
// sum. A different value than the FNV form: every rank of a run must use the same setting. Default off.
__global__ void __launch_bounds__(256) k_pf_hash(const uint32_t* w, size_t n, unsigned long long* out)
{
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(256) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * 256)
    {
        uint64_t z = (static_cast<uint64_t>(i) << 32 | w[i]) + 0x9E3779B97F4A7C15ull;
        z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ z >> 27) * 0x94D049BB133111EBull;
        s += z ^ z >> 31;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// G6_PF_RESDUMP=<dir> (measurement only; after the stream is synchronized and pf_finish): the sub-batch's FP32
// residual rows -> <dir>/resid.s<seq>.p<first position>.f32, [M][6144] (QO-INTEGRATION.md gate a)
inline void pf_resdump(Prefill& P, int M, size_t seq, int first)
{
    const char* d = getenv("G6_PF_RESDUMP");
    if (!d || !*d) return;
    std::vector<float> r((size_t)M * HID);
    G6RT_CK(cudaMemcpy(r.data(), P.resid, r.size() * 4, cudaMemcpyDeviceToHost));
    const std::string f = std::string(d) + "/resid.s" + std::to_string(seq) + ".p" + std::to_string(first) + ".f32";
    FILE* o = fopen(f.c_str(), "wb");
    if (!o || fwrite(r.data(), 4, r.size(), o) != r.size()) die("G6_PF_RESDUMP: cannot write " + f);
    fclose(o);
}
#if defined(G6_CP) && G6_CP
// ---------------------------------------------------------------- CP prefill staging
// A prefill sub-batch runs today's kernels on a staged, identity-mapped copy of each layer's latent KV: cp_pf_enter swaps
// every layer's binding args (in place: the prefill binder checks the binding's address) to {mla_cache = pool[i % 2],
// block_table = identity, page_count = table_size, cp_nr = 0}. Index keys are stored as the latent (the replicated
// prefix and the rank's own pages past R), so a full-index layer stages them too: index_cache = the pool's index region
// (pf_ixoff: table_size pages of 64 x 132 B after the latent pages), filled, gathered, sent and copied back with the
// latent. An S2 run below R holds all its pages on every rank, so its staged view reads the index keys in place through
// index_table = the run's own table (Model::s2_pf_table) and stages none. Before layer i's attention, cp_pf_gather fills
// pool[i % 2] with every page holding positions < start (from its owner); the attention appends the chunk's rows into the
// pool; cp_pf_back then copies the rows this rank stores into its sharded store (before the pool is reused at i + 2).
// cp_pf_exit restores the CP args. Plain launches only (the args go by value into every launch). A missing owner page
// sets 0x63000400 in the fabric error word (the attention error word is cleared by every prepare).
constexpr int CP_KVB = 656;           // attn.cu KV_BYTES: one latent record
constexpr int CP_IXB = 132;           // attn.cu IDX_BYTES: one index key; a page is 64 x 128 codes, then 64 FP32 scales
constexpr int CP_PF_MAXNR = 8;
struct CpPfSrc { const char* kv[CP_PF_MAXNR]; const int* table[CP_PF_MAXNR]; const char* ix[CP_PF_MAXNR]; };
// one page of 64 index keys, src -> dst (uint4 pieces of 64 x 132 B = 528), by the block's threads
__device__ __forceinline__ void cp_pf_ix_page(const char* src, char* dst, int ps)
{
    const uint4* a = reinterpret_cast<const uint4*>(src);
    uint4* d = reinterpret_cast<uint4*>(dst);
    for (size_t k = threadIdx.x; k < (size_t)ps * CP_IXB / 16; k += blockDim.x) d[k] = a[k];
}
// grid npages, block 256: logical page lp of a layer -> dst page lp, from rank (lp < Rp ? me : lp % nr) at its table;
// ixoff != 0: its index keys too (s.ix[o], page pg) -> dst + ixoff, page lp
__global__ void __launch_bounds__(256) k_cp_pf_gather(CpPfSrc s, char* dst, int nr, int me, int Rp, int ps, unsigned* err, size_t ixoff)
{
    const int lp = blockIdx.x, o = lp < Rp ? me : lp % nr;
    const int pg = s.table[o][lp];
    if (pg < 0) { if (threadIdx.x == 0) atomicOr(err, 0x63000400u); return; }
    const size_t n = (size_t)ps * CP_KVB / 16;
    const uint4* a = reinterpret_cast<const uint4*>(s.kv[o] + (size_t)pg * ps * CP_KVB);
    uint4* d = reinterpret_cast<uint4*>(dst + (size_t)lp * ps * CP_KVB);
    for (size_t k = threadIdx.x; k < n; k += blockDim.x) d[k] = a[k];
    if (ixoff) cp_pf_ix_page(s.ix[o] + (size_t)pg * ps * CP_IXB, dst + ixoff + (size_t)lp * ps * CP_IXB, ps);
}
// grid n, block 64: row pos = a0 + blockIdx.x of the pool -> this rank's store, if it stores that page; ixoff != 0: the
// row's index key too (128 B of codes, then its FP32 scale in the page's scale block) -> ix
__global__ void __launch_bounds__(64) k_cp_pf_back(const char* pool, char* kv, const int* table, int a0, int ps, char* ix, size_t ixoff)
{
    const int pos = a0 + blockIdx.x, pg = table[pos / ps];
    if (pg == G6_CP_REMOTE) return;
    const uint4* a = reinterpret_cast<const uint4*>(pool + (size_t)pos * CP_KVB);
    uint4* d = reinterpret_cast<uint4*>(kv + ((size_t)pg * ps + pos % ps) * CP_KVB);
    if (threadIdx.x < CP_KVB / 16) d[threadIdx.x] = a[threadIdx.x];
    if (!ixoff) return;
    const char* sp = pool + ixoff + (size_t)(pos / ps) * ps * CP_IXB;
    char* dp = ix + (size_t)pg * ps * CP_IXB;
    const int off = pos % ps, t = threadIdx.x - CP_KVB / 16;   // threads 41..48: codes, 49: the scale (64 >= 50)
    if (t >= 0 && t < 8) reinterpret_cast<uint4*>(dp + off * 128)[t] = reinterpret_cast<const uint4*>(sp + off * 128)[t];
    if (t == 8) reinterpret_cast<float*>(dp + ps * 128)[off] = reinterpret_cast<const float*>(sp + ps * 128)[off];
}
static_assert(CP_KVB / 16 + 9 <= 64, "k_cp_pf_back: one thread per 16 B of a row and its index key");
inline size_t cp_pf_mla_bytes(const Model& m)
{
    const G6Attn& a = m.layers[0].ab.args;
    for (const Layer& y : m.layers)
        if (y.ab.args.table_size != a.table_size || y.ab.args.page_size != a.page_size) die("G6_CP prefill: layers differ in table size");
    if (a.page_size != 64) die("G6_CP prefill: page size " + std::to_string(a.page_size) + ", staging needs 64");
    return (size_t)a.table_size * a.page_size * CP_KVB;
}
// one pool: the latent pages, then (any full-index layer) the index pages of one layer, both at capacity
inline size_t cp_pf_pool_bytes(const Model& m)
{
    bool ix = false;
    for (const Layer& y : m.layers) ix = ix || y.ab.args.index_cache;
    const G6Attn& a = m.layers[0].ab.args;
    return cp_pf_mla_bytes(m) + (ix ? (size_t)a.table_size * a.page_size * CP_IXB : 0);
}
// layer i's index region offset in its pool while staged (0: the layer has no index keys, or an S2 run below R reads
// them in place)
inline size_t cp_pf_ixoff(const Model& m, int i)
{
    return m.cp.pf_staged && m.cp.pf_ixon && m.cp.pf_saved[i].index_cache ? m.cp.pf_ixoff : 0;
}
// load time, after build (m.cp.on): the two pools (device memory, or the two halves of `region`: g6m4's registered bulk
// fabric area, 2 x cp_pf_pool_bytes) and the identity table
inline void cp_pf_alloc(Model& m, cudaStream_t st, char* region = nullptr)
{
    const G6Attn& a = m.layers[0].ab.args;
    const size_t pool = cp_pf_pool_bytes(m);
    for (int k = 0; k < 2; ++k) m.cp.pf_pool[k] = region ? region + k * pool : dmalloc<char>(pool);
    m.cp.pf_ixoff = pool > cp_pf_mla_bytes(m) ? cp_pf_mla_bytes(m) : 0;
    m.cp.pf_ident = dmalloc<int>(a.table_size);
    std::vector<int> id(a.table_size);
    for (int k = 0; k < a.table_size; ++k) id[k] = k;
    G6RT_CK(g6h2d::copy(m.cp.pf_ident, id.data(), id.size() * sizeof(int), st));
    fprintf(stderr, "# rank %d: G6_CP prefill staging 2 x %.3f GB (%s)\n", m.rank, pool / 1e9, region ? "pinned, registered" : "device");
}
inline void cp_pf_enter(Model& m)
{
    if (!m.cp.on || m.cp.pf_staged) die("G6_CP prefill: enter without CP or twice");
    m.cp.pf_saved.clear();
    m.cp.pf_g0 = m.cp.pf_g;
    m.cp.pf_ixon = m.cp.pf_ixoff != 0;
#ifdef DEC_S2
    if (m.s2_pf_table) m.cp.pf_ixon = false;
#endif
    for (int i = 0; i < m.nl; ++i)
    {
        G6Attn& a = m.layers[i].ab.args;
        m.cp.pf_saved.push_back(a);
        a.mla_cache = m.cp.pf_pool[(m.cp.pf_g0 + i) % 2]; a.block_table = m.cp.pf_ident; a.page_count = a.table_size; a.cp_nr = 0; a.cp_rank = 0;
        if (m.cp.pf_ixon && a.index_cache) a.index_cache = (char*)a.mla_cache + m.cp.pf_ixoff;   // the staged index keys
#ifdef DEC_S2
        // S2 below R: the decode views read the run's index keys through its table, so the staged view does too
        a.index_table = m.s2_pf_table ? m.cp.pf_saved.back().block_table : nullptr;
#endif
    }
    m.cp.pf_staged = true;
}
inline void cp_pf_exit(Model& m)
{
    if (!m.cp.pf_staged) die("G6_CP prefill: exit without enter");
    for (int i = 0; i < m.nl; ++i) m.layers[i].ab.args = m.cp.pf_saved[i];
    m.cp.pf_staged = false;
    m.cp.pf_g = m.cp.pf_g0 + m.nl;
}
// layer i's stored latent of every rank (the saved CP args); loopback drivers only (one address space)
inline CpPfSrc cp_pf_src(const std::vector<Model>& R, int i)
{
    if ((int)R.size() > CP_PF_MAXNR) die("G6_CP prefill: too many ranks");
    CpPfSrc s{};
    for (const Model& m : R)
    {
        if (!m.cp.pf_staged) die("G6_CP prefill: gather outside the staged view");
        s.kv[m.rank] = (const char*)m.cp.pf_saved[i].mla_cache; s.table[m.rank] = m.cp.pf_saved[i].block_table;
        s.ix[m.rank] = (const char*)m.cp.pf_saved[i].index_cache;
    }
    return s;
}
inline int cp_pf_k(const Model& m, int i) { return (int)((m.cp.pf_g0 + i) % 2); }
// layer i's pool <- every page with positions < start (first position of the earliest lane)
inline void cp_pf_gather(Model& m, const CpPfSrc& s, int i, int start, cudaStream_t st)
{
    const G6Attn& a = m.cp.pf_saved[i];
    const int np = (start + a.page_size - 1) / a.page_size;
    if (np) k_cp_pf_gather<<<np, 256, 0, st>>>(s, m.cp.pf_pool[cp_pf_k(m, i)], g6nr::NR, m.rank, m.cp.R / a.page_size, a.page_size, m.f.error, cp_pf_ixoff(m, i));
}
// rows a0 .. a0 + n - 1 of layer i: pool -> this rank's store (pages it stores)
inline void cp_pf_back(Model& m, int i, int a0, int n, cudaStream_t st)
{
    const G6Attn& a = m.cp.pf_saved[i];
    if (n > 0) k_cp_pf_back<<<n, 64, 0, st>>>(m.cp.pf_pool[cp_pf_k(m, i)], (char*)a.mla_cache, a.block_table, a0, a.page_size, (char*)a.index_cache, cp_pf_ixoff(m, i));
}
// ---- real fabric (g6m4): the pools are the registered area of a bulk-mode Fabric, identical layout on
// every rank. Use g of pool k = g % 2: begin(g) = local fill of the pages I hold (lp < R/64 or my pages) + doorbell k;
// the proxy then waits every peer's ack of use g - 2 (flag 2 + k >= g - 1), RDMA-writes my tail pages' positions < start to every
// peer at the same offsets, and writes {g + 1, aux} to flag k of every peer. wait(g) = GPU spin until flag k of every peer
// >= g + 1 (G6_CP_PF_VERIFY=1: then the peer's pages are hashed and compared to its aux). end(g) = after both lanes'
// copy-back: doorbell 2 + k, the proxy writes ack {g + 1} to flag 2 + k of every peer. Proxy jobs are enqueued in the
// order their doorbells ring: S(0), S(1), A(0), S(2), A(1), ... (begin(g + 1) after the first lane's attention of g).
struct CpPfNet
{
    g6::Fabric* fab = nullptr;
    g6::BulkDev bd{};
    bool verify = false;
    cudaEvent_t done[2] = {};         // end(g) recorded: the next begin of pool g % 2 waits on it (two-stream lanes)
    unsigned long long* vcount = nullptr;   // device: verify mismatches (G6_CP_PF_VERIFY)
#if defined(G6_CP_PF_HASH2) && G6_CP_PF_HASH2
    bool hash2 = true;                      // G6_CP_PF_HASH2 build: on unless env G6_CP_PF_HASH2=0 (same-binary control arm)
    unsigned long long* acc = nullptr;      // device: my page hash of pools 0 / 1 before k_cp_pf_hfin copies it to aux_out
#endif
};
// grid npages, block 256: pages this rank holds (lp < Rp or lp % nr == me) of its store -> pool (ixoff != 0: and their
// index keys, ix -> dst + ixoff)
__global__ void __launch_bounds__(256) k_cp_pf_fill(const char* kv, const int* table, char* dst, int nr, int me, int Rp, int ps, unsigned* err,
                                                    const char* ix, size_t ixoff)
{
    const int lp = blockIdx.x;
    if (lp >= Rp && lp % nr != me) return;
    const int pg = table[lp];
    if (pg < 0) { if (threadIdx.x == 0) atomicOr(err, 0x63000400u); return; }
    const size_t n = (size_t)ps * CP_KVB / 16;
    const uint4* a = reinterpret_cast<const uint4*>(kv + (size_t)pg * ps * CP_KVB);
    uint4* d = reinterpret_cast<uint4*>(dst + (size_t)lp * ps * CP_KVB);
    for (size_t k = threadIdx.x; k < n; k += blockDim.x) d[k] = a[k];
    if (ixoff) cp_pf_ix_page(ix + (size_t)pg * ps * CP_IXB, dst + ixoff + (size_t)lp * ps * CP_IXB, ps);
}
// summands of page lp's index keys in the pool (ixoff != 0), positions < start only: the codes of those rows as 8-B
// words (word index from the pool start, as the latent's), then their FP32 scales as 4-B values (index 2^40 + 4-B offset)
__device__ __forceinline__ unsigned long long cp_pf_ix_hash(const char* pool, size_t ixoff, int lp, int ps, int rows)
{
    if (!ixoff || rows <= 0) return 0;
    const size_t pb = ixoff + (size_t)lp * ps * CP_IXB, base = pb / 8;
    const unsigned long long* w = reinterpret_cast<const unsigned long long*>(pool) + base;
    unsigned long long h = 0;
    for (size_t k = threadIdx.x; k < (size_t)rows * 16; k += blockDim.x) h += cp_pf_hw(w[k], base + k);
    const unsigned* sc = reinterpret_cast<const unsigned*>(pool + pb + (size_t)ps * 128);
    for (int r = threadIdx.x; r < rows; r += blockDim.x) h += cp_pf_hw(sc[r], (1ull << 40) + (pb + (size_t)ps * 128) / 4 + r);
    return h;
}
// grid (npages), block 256: order-independent hash of the tail pages of `owner` in [Rp, np) of pool -> atomicAdd *out;
// positions < start only (cp_pf_tail.h: the tail page's rows >= start are being appended while it is sent)
__global__ void __launch_bounds__(256) k_cp_pf_hash(const char* pool, int nr, int owner, int Rp, int ps, int start, unsigned long long* out,
                                                    size_t ixoff)
{
    const int lp = blockIdx.x;
    if (lp < Rp || lp % nr != owner) return;
    const size_t base = (size_t)lp * ps * CP_KVB / 8, n = (size_t)cp_pf_rows(lp, ps, start) * CP_KVB / 8;
    const unsigned long long* w = reinterpret_cast<const unsigned long long*>(pool) + base;
    unsigned long long h = 0;
    for (size_t k = threadIdx.x; k < n; k += blockDim.x) h += cp_pf_hw(w[k], base + k);
    h += cp_pf_ix_hash(pool, ixoff, lp, ps, cp_pf_rows(lp, ps, start));
    atomicAdd(out, h);
}
__global__ void k_cp_pf_zero(unsigned long long* p, int n) { if (threadIdx.x < n) p[threadIdx.x] = 0; }
#if defined(G6_CP_PF_HASH2) && G6_CP_PF_HASH2
// G6_CP_PF_HASH2: k_cp_pf_hash's value with one atomic per page into DEVICE memory. k_cp_pf_hash
// adds from every thread (256 per page) into *out; for my own hash out = aux_out (pinned host memory, fabric.cpp:156), ~0.5 us per
// atomic (S3). Same summands cp_pf_hw(word, idx) over the same words, summed mod 2^64 -> bit-identical; peers check it unchanged.
// first page >= Rp held by owner o (lp % nr == o), and the number of such pages below np
__host__ __device__ inline int cp_pf_first(int nr, int o, int Rp) { return Rp + ((o - Rp % nr) % nr + nr) % nr; }
inline int cp_pf_owned(int nr, int o, int Rp, int np) { const int f = cp_pf_first(nr, o, Rp); return f >= np ? 0 : (np - 1 - f) / nr + 1; }
struct CpPfOwners { int o[CP_PF_MAXNR]; };
// grid (owned pages, owner slots), block 256: page blockIdx.x of owner p = own.o[blockIdx.y] -> atomicAdd out[p * ostride]
// (ostride 0: one accumulator; 1: out[p] per owner, as the wait side's h[p])
__global__ void __launch_bounds__(256) k_cp_pf_hash2(const char* pool, int nr, CpPfOwners own, int Rp, int ps, int start, unsigned long long* out, int ostride,
                                                     size_t ixoff)
{
    const int lp = cp_pf_first(nr, own.o[blockIdx.y], Rp) + (int)blockIdx.x * nr;
    const int rows = cp_pf_rows(lp, ps, start);          // block-uniform: a page at or past start returns as a whole block
    if (rows <= 0) return;
    const size_t base = (size_t)lp * ps * CP_KVB / 8, n2 = (size_t)rows * CP_KVB / 16;   // CP_KVB = 656 = 41 x 16 B
    const ulonglong2* w = reinterpret_cast<const ulonglong2*>(reinterpret_cast<const unsigned long long*>(pool) + base);
    unsigned long long h = 0;
    for (size_t k = threadIdx.x; k < n2; k += blockDim.x) { const ulonglong2 v = w[k]; h += cp_pf_hw(v.x, base + 2 * k) + cp_pf_hw(v.y, base + 2 * k + 1); }
    h += cp_pf_ix_hash(pool, ixoff, lp, ps, rows);
    for (int s = 16; s; s >>= 1) h += __shfl_down_sync(0xffffffffu, h, s);
    __shared__ unsigned long long red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = h;
    __syncthreads();
    if (threadIdx.x == 0) { unsigned long long t = 0; for (int i = 0; i < 8; ++i) t += red[i]; atomicAdd(out + (size_t)own.o[blockIdx.y] * ostride, t); }
}
// grid 1, block 32: my finished hash (device) -> aux_out (pinned), one store, fenced to the system scope by the storing thread
// itself (k_cp_pf_bell's fence runs in another thread of a later kernel)
__global__ void k_cp_pf_hfin(const unsigned long long* acc, unsigned long long* out)
{
    if (threadIdx.x == 0) { *out = *acc; __threadfence_system(); }
}
#endif
// grid 1, block 32: GPU -> proxy doorbell (after the stream's earlier writes to pinned memory)
__global__ void __launch_bounds__(32) k_cp_pf_bell(unsigned long long* db, unsigned long long v)
{
    if (threadIdx.x == 0) { __threadfence_system(); glue::st_release_sys(reinterpret_cast<u64*>(db), v); }
}
// grid 1, block 32: lane p waits flag k of peer p >= want; verify: aux vs my hash of p's pages (hash[p], computed after)
__global__ void __launch_bounds__(32) k_cp_pf_wait(g6::BulkDev bd, int k, unsigned long long want, unsigned code)
{
    const int p = threadIdx.x;
    if (p >= bd.nr || p == bd.rank) return;
    const u64* f = reinterpret_cast<const u64*>(bd.flags + ((size_t)k * bd.nr + p) * 2);
    const long long t0 = clock64();
    unsigned it = 0;
    while (glue::ld_acquire_sys(f) < want)
        if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(bd.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
        {
            atomicCAS(bd.error, 0u, code | (unsigned)(k << 8) | (unsigned)p);
            return;
        }
}
// grid 1, block 32: verify: hash[p] (mine, of peer p's pages in my pool) == aux of flag k from p, else count + error
__global__ void __launch_bounds__(32) k_cp_pf_check(g6::BulkDev bd, int k, const unsigned long long* hash, unsigned long long* count)
{
    const int p = threadIdx.x;
    if (p >= bd.nr || p == bd.rank) return;
    const unsigned long long aux = glue::ld_acquire_sys(reinterpret_cast<const u64*>(bd.flags + ((size_t)k * bd.nr + p) * 2 + 1));
    if (aux != hash[p]) { atomicAdd(count, 1ull); atomicCAS(bd.error, 0u, 0x63000800u | (unsigned)(k << 4) | (unsigned)p); }
}
inline CpPfNet& cp_pf_net(Model& m) { return *static_cast<CpPfNet*>(m.cp.pf_net); }
// load time (g6m4): fabs = a started bulk-mode Fabric with bulk_bytes = 2 x cp_pf_pool_bytes(m) and bulk_flags = 4
inline void cp_pf_net_init(Model& m, g6::Fabric& fabs, cudaStream_t st)
{
    auto* n = new CpPfNet;
    n->fab = &fabs; n->bd = fabs.bulk_dev();
    n->verify = getenv("G6_CP_PF_VERIFY") && atoi(getenv("G6_CP_PF_VERIFY")) == 1;
    for (cudaEvent_t& e : n->done) G6RT_CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
#if defined(G6_CP_PF_HASH2) && G6_CP_PF_HASH2
    {
        std::string e;
        const int v = env_choice("G6_CP_PF_HASH2", 1, 1, &e);   // strict: unset / 1 = hash2, 0 = k_cp_pf_hash, else exit
        if (v < 0) die(e);
        n->hash2 = v == 1;
    }
    n->acc = dmalloc<unsigned long long>(2);
    fprintf(stderr, "# rank %d: G6_CP_PF_HASH2 build: page hash %s\n", m.rank, n->hash2 ? "hash2 (block-reduced, device accumulator)" : "k_cp_pf_hash (env G6_CP_PF_HASH2=0)");
#endif
    n->vcount = dmalloc<unsigned long long>(1 + 2 * CP_PF_MAXNR);   // [0] count, [1..] hash scratch of two pools
    G6RT_CK(cudaMemsetAsync(n->vcount, 0, (1 + 2 * CP_PF_MAXNR) * 8, st));
    m.cp.pf_net = n;
    cp_pf_alloc(m, st, n->bd.bulk);
}
inline unsigned long long cp_pf_mismatches(Model& m)
{
    unsigned long long c = 0;
    if (m.cp.pf_net) G6RT_CK(cudaMemcpy(&c, cp_pf_net(m).vcount, 8, cudaMemcpyDeviceToHost));
    return c;
}
// begin(layer i): local fill of pool k, (verify) my hash -> aux_out[k], doorbell k; proxy job S(g)
inline void cp_pf_begin(Model& m, int i, int start, cudaStream_t st)
{
    CpPfNet& n = cp_pf_net(m);
    const G6Attn& a = m.cp.pf_saved[i];
    const int k = cp_pf_k(m, i), ps = a.page_size, Rp = m.cp.R / ps, np = (start + ps - 1) / ps, nr = g6nr::NR;
    const unsigned long long g = m.cp.pf_g0 + i;
    G6RT_CK(cudaStreamWaitEvent(st, n.done[k], 0));    // my previous use of pool k (g - 2) is over (both lanes)
    const size_t ixoff = cp_pf_ixoff(m, i);
    if (np) k_cp_pf_fill<<<np, 256, 0, st>>>((const char*)a.mla_cache, a.block_table, m.cp.pf_pool[k], nr, m.rank, Rp, ps, n.bd.error,
                                             (const char*)a.index_cache, ixoff);
#if defined(G6_CP_PF_HASH2) && G6_CP_PF_HASH2
    if (n.verify && n.hash2)
    {
        const int own = cp_pf_owned(nr, m.rank, Rp, np);
        CpPfOwners o{}; o.o[0] = m.rank;
        k_cp_pf_zero<<<1, 32, 0, st>>>(n.acc + k, 1);
        if (own) k_cp_pf_hash2<<<dim3(own, 1), 256, 0, st>>>(m.cp.pf_pool[k], nr, o, Rp, ps, start, n.acc + k, 0, ixoff);
        k_cp_pf_hfin<<<1, 32, 0, st>>>(n.acc + k, n.bd.aux_out + k);
    }
    else
#endif
    if (n.verify)
    {
        k_cp_pf_zero<<<1, 32, 0, st>>>(n.bd.aux_out + k, 1);
        if (np) k_cp_pf_hash<<<np, 256, 0, st>>>(m.cp.pf_pool[k], nr, m.rank, Rp, ps, start, n.bd.aux_out + k, ixoff);
    }
    k_cp_pf_bell<<<1, 32, 0, st>>>(n.bd.doorbell + k, g + 1);
    g6::Fabric::BulkJob j;
    j.db = k; j.seq = g + 1;
    j.wait_flag = 2 + k; j.wait_min = g >= 2 ? g - 1 : 0;
    const size_t pb = (size_t)ps * CP_KVB, base = (size_t)k * cp_pf_pool_bytes(m);
    for (int lp = Rp; lp < np; ++lp)   // positions < start only: the tail page's first start % ps rows (cp_pf_tail.h)
        if (lp % nr == m.rank)
        {
            const int rows = cp_pf_rows(lp, ps, start);
            j.ranges.push_back({base + lp * pb, (uint32_t)((size_t)rows * CP_KVB)});
            if (ixoff)                 // the page's index keys: the rows' codes, and their scales after the page's codes
            {
                // ranges are whole 8-B words (fabric.cpp), so an odd row count also sends the scale of row `rows`, the
                // position `start`: every rank appends that row into its own pool after its wait for this use, which
                // the peer's flag orders after these bytes land, and no hash covers it
                const size_t ib = base + ixoff + (size_t)lp * ps * CP_IXB;
                j.ranges.push_back({ib, (uint32_t)((size_t)rows * 128)});
                j.ranges.push_back({ib + (size_t)ps * 128, (uint32_t)((size_t)(rows + 1) / 2 * 8)});
            }
        }
    j.post_flag = k; j.post_val = g + 1;
    n.fab->enqueue_bulk(std::move(j));
}
// wait(layer i): every peer's pages of use g have landed (verify: and hash to what the peer sent)
inline void cp_pf_wait(Model& m, int i, int start, cudaStream_t st)
{
    CpPfNet& n = cp_pf_net(m);
    const int k = cp_pf_k(m, i);
    const unsigned long long g = m.cp.pf_g0 + i;
    k_cp_pf_wait<<<1, 32, 0, st>>>(n.bd, k, g + 1, 0x63000c00u);
    if (n.verify)
    {
        const G6Attn& a = m.cp.pf_saved[i];
        const int ps = a.page_size, Rp = m.cp.R / ps, np = (start + ps - 1) / ps;
        const size_t ixoff = cp_pf_ixoff(m, i);
        unsigned long long* h = n.vcount + 1 + k * CP_PF_MAXNR;
        k_cp_pf_zero<<<1, 32, 0, st>>>(h, CP_PF_MAXNR);
#if defined(G6_CP_PF_HASH2) && G6_CP_PF_HASH2
        if (n.hash2)
        {   // every peer in one launch: blockIdx.y = the y-th peer, its hash -> h + p (ostride 1)
            CpPfOwners o{}; int y = 0, mx = 0;
            for (int p = 0; p < g6nr::NR; ++p)
                if (p != m.rank) { o.o[y++] = p; mx = std::max(mx, cp_pf_owned(g6nr::NR, p, Rp, np)); }
            if (mx) k_cp_pf_hash2<<<dim3(mx, y), 256, 0, st>>>(m.cp.pf_pool[k], g6nr::NR, o, Rp, ps, start, h, 1, ixoff);
        }
        else
#endif
        for (int p = 0; p < g6nr::NR; ++p)
            if (p != m.rank && np) k_cp_pf_hash<<<np, 256, 0, st>>>(m.cp.pf_pool[k], g6nr::NR, p, Rp, ps, start, h + p, ixoff);
        k_cp_pf_check<<<1, 32, 0, st>>>(n.bd, k, h, n.vcount);
    }
}
// end(layer i), after every lane's copy-back of layer i on st: doorbell 2 + k, done event; proxy job A(g)
inline void cp_pf_end(Model& m, int i, cudaStream_t st)
{
    CpPfNet& n = cp_pf_net(m);
    const int k = cp_pf_k(m, i);
    const unsigned long long g = m.cp.pf_g0 + i;
    k_cp_pf_bell<<<1, 32, 0, st>>>(n.bd.doorbell + 2 + k, g + 1);
    G6RT_CK(cudaEventRecord(n.done[k], st));
    g6::Fabric::BulkJob j;
    j.db = 2 + k; j.seq = g + 1; j.post_flag = 2 + k; j.post_val = g + 1;
    n.fab->enqueue_bulk(std::move(j));
}
// the hooks pf_step / pf_step2 / pf_step2s call (no-ops unless m.cp.pf_staged with a CpPfNet): before the first lane's
// attention of layer i (i == 0: begin + wait; later layers were begun by the prefetch), the prefetch of layer i + 1
// right after it, and after the last lane's attention: the copy-back of every lane's rows + end
inline bool cp_pf_live(const Model& m) { return m.cp.pf_staged && m.cp.pf_net; }
inline void cp_pf_pre(Model& m, int i, int start, cudaStream_t st)
{
    if (!cp_pf_live(m)) return;
    if (i == 0) cp_pf_begin(m, 0, start, st);
    cp_pf_wait(m, i, start, st);
}
inline void cp_pf_prefetch(Model& m, int i, int start, cudaStream_t st)
{
    if (cp_pf_live(m) && i + 1 < m.nl) cp_pf_begin(m, i + 1, start, st);
}
inline void cp_pf_post(Model& m, int i, int a0, int n0, int a1, int n1, cudaStream_t st)
{
    if (!cp_pf_live(m)) return;
    cp_pf_back(m, i, a0, n0, st);
    cp_pf_back(m, i, a1, n1, st);
    cp_pf_end(m, i, st);
}
#endif
// after the stream is synchronized: hash of the sub-batch's residual rows (every rank must print the same)
inline uint64_t pf_hash(Prefill& P, int M)
{
    static const bool gpu = getenv("G6_PF_HASH_GPU") && atoi(getenv("G6_PF_HASH_GPU")) == 1;
    if (gpu)
    {
        static thread_local unsigned long long* d = nullptr;
        if (!d) G6RT_CK(cudaMalloc(&d, 8));
        unsigned long long h = 0;
        G6RT_CK(cudaMemset(d, 0, 8));
        k_pf_hash<<<1024, 256>>>(reinterpret_cast<const uint32_t*>(P.resid), static_cast<size_t>(M) * HID, d);
        G6RT_CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
        return h;
    }
    std::vector<float> r((size_t)M * HID);
    G6RT_CK(cudaMemcpy(r.data(), P.resid, r.size() * 4, cudaMemcpyDeviceToHost));
    return fnv(0xcbf29ce484222325ull, r.data(), r.size() * 4);
}

}   // namespace g6rt
