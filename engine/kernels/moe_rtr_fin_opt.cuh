// Bit-exact fused MoE router+plan and reduced-warp K1 finalize decode kernels for GLM-5.3
// (GR8_MOE_RTR_FIN_OPT=1, default off).
//
// Preserves exact IEEE-754 left-to-right reduction order (0 ULP vs stock bind::launch(b.router) + glue::k_router_plan
// and k1_tr3_fin<true>):
//   - k_rtr_plan_fused_sk1<NT, PFB, EVICT>: 16 CTAs x 512t (16 warps/CTA = 1 warp per k-part, 0 global part traffic,
//     64 KB L2-resident scratch for sigmoid/key handoff + done_cnt), parallel RoutePlan zeroing, exact expf + bias, and
//     last-CTA 16-warp exact top-k/slot scatter. Cuts router+plan from 2 launches to 1 (-75 graph nodes per 75L step).
//   - k_fin_5w<EVICT, U4>: 5-warp (160t) 1-barrier exact k1_tr3_fin<true> with L1::evict_first loads and paired 16-byte
//     BF16/FP16 stores direct to send slot (M in {4, 8}).
//   - k_fin_1w<W, PREF, EVICT, U4>: 1 warp per (m, hb), W warps/CTA, 0 reduction barriers, software-pipelined route
//     prefetch, L1::evict_first loads, and paired 16-byte stores (M=16: W=8 at TP6, W=4 at TP4).
//
// Strict per-(TP, M) dispatch table (enabled only at (TP, M) with 5/5 non-overlapping 8L E2E wins vs GR8_TR3_WIDE4=1;
// TP4 M=1 and all other (TP, M) fall back to stock).
// Self-contained: leaves bind.cuh, k1.cuh, tr3.cuh, glue.cuh, densev.cuh, dtr.cuh, dec_dense_opt.cuh and
// dtr_exact_opt.cuh unmodified.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "bind.cuh"
#include "glue.cuh"

namespace moe_rtr_fin_opt
{

#if !G6_MODEL_FLASH && !G6_MODEL_DS41 && !G6_MODEL_INK2

// ------------------------------------------------------------------------------------------------ L1/L2 load helpers

__device__ __forceinline__ void ld_ef_v4(const void* p, u32* v)
{
    asm volatile("ld.global.nc.L1::evict_first.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                 : "l"(p));
}

__device__ __forceinline__ uint2 ld_ef_v2(const void* p)
{
    uint2 v;
    asm volatile("ld.global.nc.L1::evict_first.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ float ldcg_f32(const float* p)
{
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

// ------------------------------------------------------------------------------------------------ Router scratch (~64 KB)

struct RouterScratch
{
    float s_buf[32][glue::NEXP];    // 32 KB: sigmoid(logit)
    float k_buf[32][glue::NEXP];    // 32 KB: sigmoid(logit) + bias
    unsigned done_cnt;              // 16-tile arrival counter (auto-reset to 0 by last CTA)
};

// Exact Bf16w warp k-part body (identical mma_bf16 operands and order to densev::gv_tile_part<Bf16w, NT>)
template <int NT, int PFB, bool EVICT>
__device__ __forceinline__ void rtr_bf16_part(const uint8_t* tile, const u32* bf, int ts, int g0, int g1,
                                              float (&acc)[NT][4])
{
    const int lane = threadIdx.x & 31;
    const int Q = g1 - g0;
    u32 nx[PFB][4];
    #pragma unroll
    for (int p = 0; p < PFB; ++p)
    {
        if (p < Q)
        {
            const uint8_t* ptr = tile + static_cast<size_t>(g0 + p) * 512 + lane * 16;
            if constexpr (EVICT) ld_ef_v4(ptr, nx[p]);
            else gv::ld_nc_v4(ptr, nx[p]);
        }
    }
    for (int q = 0; q < Q; q += PFB)
    {
        u32 cur[PFB][4];
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
        {
            cur[p][0] = nx[p][0]; cur[p][1] = nx[p][1];
            cur[p][2] = nx[p][2]; cur[p][3] = nx[p][3];
        }
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
        {
            if (q + PFB + p < Q)
            {
                const uint8_t* ptr = tile + static_cast<size_t>(g0 + q + PFB + p) * 512 + lane * 16;
                if constexpr (EVICT) ld_ef_v4(ptr, nx[p]);
                else gv::ld_nc_v4(ptr, nx[p]);
            }
        }
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
        {
            if (q + p < Q)
            {
                const int ks = g0 + q + p;
                #pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    gv::mma_bf16(acc[t], cur[p][0], cur[p][1], cur[p][2], cur[p][3],
                                 bf[t * ts + ks * 64], bf[t * ts + ks * 64 + 1]);
                }
            }
        }
    }
}

__device__ __forceinline__ void rtr_zero_plan_512(int ot, int M, glue::RoutePlan* __restrict__ P)
{
    constexpr int TOPK = glue::TOPK;
    constexpr int GSTRIDE = 16 * 512;
    const int gtid = ot * 512 + threadIdx.x;
    const int n_sl = M * TOPK * 8;
    for (int i = gtid; i < n_sl; i += GSTRIDE)
    {
        (&P->slot_w[0][0])[i] = 0.f;
        (&P->slot_tok[0][0])[i] = -1;
    }
    const int n_tok = glue::MMAX * TOPK;
    for (int i = M * TOPK + gtid; i < n_tok; i += GSTRIDE)
    {
        const int m = i >> 3, k = i & 7;
        P->tok_slot[m][k] = -1;
        P->tok_col[m][k] = 0;
        P->tok_w[m][k] = 0.f;
        P->tok_expert[m][k] = -1;
    }
}

__device__ __forceinline__ void rtr_emit_tile(int ot, int t, int M, int lane, float b0, float b8,
                                              const float4& s, float* __restrict__ logits,
                                              RouterScratch* __restrict__ rsc)
{
    constexpr int NEXP = glue::NEXP;
    const int r = lane >> 2, c = lane & 3;
    const int o = ot * 16 + r;
    const int m0 = 8 * t + 2 * c;
    if (m0 < M)
    {
        logits[static_cast<size_t>(m0) * NEXP + o] = s.x;
        logits[static_cast<size_t>(m0) * NEXP + o + 8] = s.z;
        const float s0 = 1.f / (1.f + expf(-s.x));
        const float s8 = 1.f / (1.f + expf(-s.z));
        rsc->s_buf[m0][o] = s0;
        rsc->k_buf[m0][o] = s0 + b0;
        rsc->s_buf[m0][o + 8] = s8;
        rsc->k_buf[m0][o + 8] = s8 + b8;
    }
    if (m0 + 1 < M)
    {
        logits[static_cast<size_t>(m0 + 1) * NEXP + o] = s.y;
        logits[static_cast<size_t>(m0 + 1) * NEXP + o + 8] = s.w;
        const float s0 = 1.f / (1.f + expf(-s.y));
        const float s8 = 1.f / (1.f + expf(-s.w));
        rsc->s_buf[m0 + 1][o] = s0;
        rsc->k_buf[m0 + 1][o] = s0 + b0;
        rsc->s_buf[m0 + 1][o + 8] = s8;
        rsc->k_buf[m0 + 1][o + 8] = s8 + b8;
    }
}

// Exact last-CTA glue::k_router_plan (and moe_union_cap::k_router_plan_capped when cap > 0) body
// (16 warps/CTA -> M <= 16 in 1 pass, M <= 32 in 2 passes, 0-ULP identical to stock / union-capped plan).
__device__ __forceinline__ void rtr_plan_tail_16w(int M, glue::RoutePlan* __restrict__ P,
                                                  RouterScratch* __restrict__ rsc,
                                                  unsigned keep, const unsigned* keep_dev,
                                                  int cap, int cap_mode)
{
    constexpr int WPC = 16;
    constexpr int BD = WPC * 32;
    constexpr int NEXP = glue::NEXP;
    constexpr int TOPK = glue::TOPK;
    constexpr int RW = glue::RW;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ unsigned emask[NEXP];
    __shared__ int sel_e[glue::MMAX][TOPK];
    __shared__ float sel_w[glue::MMAX][TOPK];
    __shared__ int slot_of[NEXP];
    __shared__ float bs_val[NEXP];
    __shared__ int bs_id[NEXP];
    __shared__ unsigned char kept_exp[NEXP];
    __shared__ int wcount[RW];
    __shared__ int s_n_distinct;

    const bool dense = M > 8;
    const unsigned on = keep_dev ? *keep_dev : keep ? keep : 0xffffffffu;
    const unsigned m_bits = M >= 32 ? 0xffffffffu : ((1u << M) - 1u);
    const unsigned active_mask = on & m_bits;
    const bool do_cap = (cap > 0) && (M > 1) && (M * TOPK > cap);
    if (M > 1)
    {
        for (int i = tid; i < NEXP; i += BD)
        {
            emask[i] = 0u;
            kept_exp[i] = 1;
        }
        __syncthreads();
    }

    for (int m = warp; m < M; m += WPC)
    {
        if (!(active_mask >> m & 1u))
        {
            if (M == 1)
            {
                if (lane < TOPK)
                {
                    P->tok_slot[0][lane] = -1;
                    P->tok_col[0][lane] = 0;
                    P->tok_w[0][lane] = 0.f;
                    P->tok_expert[0][lane] = -1;
                }
                if (lane == 0) P->n_slots = 0;
            }
            continue;
        }
        float sv_arr[RW], key[RW];
        #pragma unroll
        for (int i = 0; i < RW; ++i)
        {
            const int e = lane + 32 * i;
            sv_arr[i] = ldcg_f32(&rsc->s_buf[m][e]);
            key[i] = ldcg_f32(&rsc->k_buf[m][e]);
        }
        unsigned taken = 0u;
        float my_k = key[0];
        int my_e = lane;
        #pragma unroll
        for (int i = 1; i < RW; ++i)
            if (key[i] > my_k) { my_k = key[i]; my_e = lane + 32 * i; }
        for (int k = 0; k < TOPK; ++k)
        {
            float bk = my_k;
            int be = my_e;
            #pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1)
            {
                const float ok = __shfl_xor_sync(0xffffffffu, bk, o2);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o2);
                if (ok > bk || (ok == bk && oe < be)) { bk = ok; be = oe; }
            }
            float sv = 0.f;
            if ((be & 31) == lane)
            {
                taken |= 1u << (be >> 5);
                sv = sv_arr[be >> 5];
                if (k + 1 < TOPK)
                {
                    my_k = -INFINITY;
                    my_e = 0x7fffffff;
                    #pragma unroll
                    for (int i = 0; i < RW; ++i)
                        if (!(taken >> i & 1u) && key[i] > my_k) { my_k = key[i]; my_e = lane + 32 * i; }
                }
            }
            sv = __shfl_sync(0xffffffffu, sv, be & 31);
            if (lane == 0) { sel_e[m][k] = be; sel_w[m][k] = sv; }
        }
        __syncwarp();
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) sum += sel_w[m][k];
        __syncwarp();
        if (lane < TOPK)
        {
            const float wv = sel_w[m][lane] / sum * glue::ROUTE_SCALE;
            sel_w[m][lane] = wv;
            if (M == 1)
            {
                const int e = sel_e[0][lane];
                int s2 = 0;
                #pragma unroll
                for (int j = 0; j < TOPK; ++j) s2 += (sel_e[0][j] < e);
                P->slot_expert[s2] = e;
                P->slot_mask[s2] = 1;
                P->tok_slot[0][lane] = s2;
                P->tok_col[0][lane] = 0;
                P->tok_w[0][lane] = wv;
                P->tok_expert[0][lane] = e;
                P->slot_w[s2][0] = wv;
                P->slot_tok[s2][0] = 0;
                if (lane == 0) P->n_slots = TOPK;
            }
            else
            {
                atomicOr(&emask[sel_e[m][lane]], 1u << m);
            }
        }
        __syncwarp();
    }
    if (M == 1) return;
    __syncthreads();

    // Exact moe_union_cap::k_router_plan_capped union cap (all 512t participate in barriers, tid < 256 runs bitonic sort)
    if (do_cap)
    {
        const unsigned em0 = tid < NEXP ? emask[tid] : 0u;
        if (tid < NEXP)
        {
            const unsigned wm0 = __ballot_sync(0xffffffffu, em0 != 0u);
            if (lane == 0) wcount[warp] = __popc(wm0);
        }
        __syncthreads();
        if (tid == 0)
        {
            int nd = 0;
            #pragma unroll
            for (int i = 0; i < RW; ++i) nd += wcount[i];
            s_n_distinct = nd;
        }
        __syncthreads();
        if (s_n_distinct > cap)
        {
            const int policy = cap_mode % 3;
            const int renorm = cap_mode / 3;
            if (tid < NEXP)
            {
                kept_exp[tid] = 0;
                float my_score = -INFINITY;
                if (em0 != 0u)
                {
                    float sc = 0.f;
                    bool is_anchor0 = false;
                    bool is_top1 = false;
                    unsigned tmp = em0;
                    while (tmp)
                    {
                        const int m = __ffs(tmp) - 1;
                        tmp &= tmp - 1u;
                        int rk = -1;
                        #pragma unroll
                        for (int k = 0; k < TOPK; ++k)
                            if (sel_e[m][k] == tid) { rk = k; break; }
                        if (rk >= 0)
                        {
                            const float wm = sel_w[m][rk];
                            if (policy == 1)
                            {
                                const float gamma = 1.f / (1.f + 0.25f * static_cast<float>(m));
                                sc += gamma * wm;
                                if (m == 0) is_anchor0 = true;
                                if (rk == 0) is_top1 = true;
                            }
                            else
                            {
                                sc += wm;
                                if (rk == 0) is_top1 = true;
                            }
                        }
                    }
                    if (policy == 0) { if (is_top1) sc += 100.f; }
                    else if (policy == 1) { if (is_anchor0) sc += 1000.f; if (is_top1) sc += 100.f; }
                    my_score = sc;
                }
                bs_val[tid] = my_score;
                bs_id[tid] = tid;
            }
            __syncthreads();
            for (int k = 2; k <= NEXP; k <<= 1)
            {
                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    if (tid < NEXP)
                    {
                        const int ixj = tid ^ j;
                        if (ixj > tid)
                        {
                            const bool desc_dir = (tid & k) == 0;
                            const float va = bs_val[tid], vb = bs_val[ixj];
                            const int ia = bs_id[tid], ib = bs_id[ixj];
                            const bool a_better = (va > vb) || (va == vb && ia < ib);
                            const bool swap = desc_dir ? !a_better : a_better;
                            if (swap)
                            {
                                bs_val[tid] = vb; bs_val[ixj] = va;
                                bs_id[tid] = ib; bs_id[ixj] = ia;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            if (tid < cap && bs_val[tid] > -INFINITY) kept_exp[bs_id[tid]] = 1;
            __syncthreads();
            if (tid < NEXP && !kept_exp[tid]) emask[tid] = 0u;
            __syncthreads();

            for (int m = warp; m < M; m += WPC)
            {
                if (!(active_mask >> m & 1u)) continue;
                if (lane == 0)
                {
                    int n_kept = 0;
                    float sum_w = 0.f, sum_w2 = 0.f, orig_w2 = 0.f;
                    #pragma unroll
                    for (int k = 0; k < TOPK; ++k)
                    {
                        const float wv = sel_w[m][k];
                        orig_w2 += wv * wv;
                        if (kept_exp[sel_e[m][k]])
                        {
                            ++n_kept;
                            sum_w += wv;
                            sum_w2 += wv * wv;
                        }
                    }
                    if (n_kept > 0 && n_kept < TOPK)
                    {
                        float scale = 1.f;
                        if (renorm == 0) scale = (sum_w2 > 0.f) ? sqrtf(orig_w2 / sum_w2) : 0.f;
                        else if (renorm == 1) scale = (sum_w > 0.f) ? (glue::ROUTE_SCALE / sum_w) : 0.f;
                        #pragma unroll
                        for (int k = 0; k < TOPK; ++k)
                        {
                            if (kept_exp[sel_e[m][k]]) sel_w[m][k] *= scale;
                            else sel_w[m][k] = 0.f;
                        }
                    }
                    else if (n_kept == 0)
                    {
                        #pragma unroll
                        for (int k = 0; k < TOPK; ++k) sel_w[m][k] = 0.f;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (warp == 0)
    {
        int base = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
        {
            const int e = g * 32 + lane;
            const unsigned em = emask[e];
            int nsl, exc, tot;
            if (!dense)
            {
                nsl = em != 0u;
                const unsigned wm = __ballot_sync(0xffffffffu, nsl);
                exc = __popc(wm & ((1u << lane) - 1u));
                tot = __popc(wm);
            }
            else
            {
                const int cnt = __popc(em);
                nsl = (cnt + 7) >> 3;
                int scan = nsl;
                #pragma unroll
                for (int o2 = 1; o2 < 32; o2 <<= 1)
                {
                    const int v = __shfl_up_sync(0xffffffffu, scan, o2);
                    if (lane >= o2) scan += v;
                }
                exc = scan - nsl;
                tot = __shfl_sync(0xffffffffu, scan, 31);
            }
            const int slot0 = base + exc;
            base += tot;
            if (nsl)
            {
                slot_of[e] = slot0;
                if (!dense)
                {
                    P->slot_expert[slot0] = e;
                    P->slot_mask[slot0] = static_cast<int>(em);
                }
                else
                {
                    const int cnt = __popc(em);
                    for (int j = 0; j < nsl; ++j)
                    {
                        P->slot_expert[slot0 + j] = e;
                        P->slot_mask[slot0 + j] = static_cast<int>((1u << (cnt - 8 * j < 8 ? cnt - 8 * j : 8)) - 1u);
                    }
                }
            }
        }
        if (lane == 0)
        {
            P->n_slots = base;
            if (base == 0) P->slot_expert[0] = 0;
        }
    }
    __syncthreads();

    for (int i = tid; i < M * TOPK; i += BD)
    {
        const int m = i >> 3, k = i & 7;
        if (!(active_mask >> m & 1u) || !kept_exp[sel_e[m][k]])
        {
            P->tok_slot[m][k] = -1;
            P->tok_col[m][k] = 0;
            P->tok_w[m][k] = 0.f;
            P->tok_expert[m][k] = -1;
            continue;
        }
        const int e = sel_e[m][k];
        const int rk = __popc(emask[e] & ((1u << m) - 1u));
        const int s2 = slot_of[e] + (dense ? rk >> 3 : 0);
        const int col = dense ? (rk & 7) : m;
        P->tok_slot[m][k] = s2;
        P->tok_col[m][k] = col;
        P->tok_w[m][k] = sel_w[m][k];
        P->tok_expert[m][k] = e;
        P->slot_w[s2][col] = sel_w[m][k];
        P->slot_tok[s2][col] = m;
    }
}

// 16 CTAs x 512t fused BF16 router + exact RoutePlan (with optional union cap, 0 global part traffic)
template <int NT, int PFB, bool EVICT>
__global__ __launch_bounds__(512)
void k_rtr_plan_fused_sk1(const uint8_t* __restrict__ stream, const u32* __restrict__ xfrag,
                          const float* __restrict__ bias, int M,
                          float* __restrict__ logits, glue::RoutePlan* __restrict__ P,
                          RouterScratch* __restrict__ rsc,
                          unsigned keep, const unsigned* keep_dev,
                          int cap, int cap_mode)
{
    constexpr int K = geom::HID, NG2 = K / 16, NGP = NG2 / 16, TS = K * 4;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ot = static_cast<int>(blockIdx.x);

    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    rtr_bf16_part<NT, PFB, EVICT>(stream + static_cast<size_t>(ot) * NG2 * 512,
                                  xfrag + lane * 2, TS, warp * NGP, (warp + 1) * NGP, acc);

    __shared__ float4 s_part[NT][16][32];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        s_part[t][warp][lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);

    rtr_zero_plan_512(ot, M, P);
    __syncthreads();

    if (warp < NT)
    {
        const int o = ot * 16 + (lane >> 2);
        const float b0 = bias[o], b8 = bias[o + 8];
        float4 s = s_part[warp][0][lane];
        #pragma unroll
        for (int p = 1; p < 16; ++p)
        {
            const float4 q = s_part[warp][p][lane];
            s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
        }
        rtr_emit_tile(ot, warp, M, lane, b0, b8, s, logits, rsc);
    }

    __shared__ int s_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(&rsc->done_cnt, 1u) == 15u);
    __syncthreads();
    if (!s_last) return;
    if (tid == 0) rsc->done_cnt = 0u;

    rtr_plan_tail_16w(M, P, rsc, keep, keep_dev, cap, cap_mode);
}

// ------------------------------------------------------------------------------------------------ Exact K1 Finalize

template <bool EVICT>
__device__ __forceinline__ void fin_route_acc(const K1Args& A, int plan, int m, int hb, int n0, int lane,
                                              uint2 sg, int route_k, float (&acc)[4])
{
    const int idx = (plan * G6_MX + m) * 8 + route_k;
    const int slot = A.tok_slot[idx];
    if (slot < 0) return;
    const float w = A.tok_w[idx];
    const int col = A.M <= 8 ? m : A.tok_col[idx];
    const long long* tr = A.t_tab + 6 * A.slot_expert[plan * SMAX + slot];
    const int nb = static_cast<int>(tr[3]);
    const size_t row = static_cast<size_t>(tr[2]);
    uint2 zr[g6nr::EB];
    __half sc[g6nr::EB];
    #pragma unroll
    for (int b = 0; b < g6nr::EB; ++b)
    {
        if (b < nb)
        {
            const void* zp = A.z + ((static_cast<size_t>(slot) * A.B + b) * 8 + col) * HID + n0;
            zr[b] = EVICT ? ld_ef_v2(zp) : *reinterpret_cast<const uint2*>(zp);
            sc[b] = A.t_v3b[(row + b) * geom::HB + hb];
        }
    }
    #pragma unroll
    for (int b = 0; b < g6nr::EB; ++b)
    {
        if (b < nb)
        {
            const float2 z01 = __half22float2(as_h2(zr[b].x));
            const float2 z23 = __half22float2(as_h2(zr[b].y));
            float v[4] = {z01.x, z01.y, z23.x, z23.y};
            had128(v, lane);
            const __half2 c = __half2half2(sc[b]);
            const uint2 s2 = make_uint2(as_u32(__hmul2(as_h2(sg.x), c)), as_u32(__hmul2(as_h2(sg.y), c)));
            const float2 s01 = __half22float2(as_h2(s2.x));
            const float2 s23 = __half22float2(as_h2(s2.y));
            acc[0] += w * (v[0] * s01.x); acc[1] += w * (v[1] * s01.y);
            acc[2] += w * (v[2] * s23.x); acc[3] += w * (v[3] * s23.y);
        }
    }
}

template <bool EVICT>
__device__ __forceinline__ void fin_shared_acc(const K1Args& A, int m, int hb, int n0, int lane,
                                               uint2 sg, float (&acc)[4])
{
    const long long* tr = A.t_tab + 6 * A.sh_e;
    const int snb = static_cast<int>(tr[3]);
    const size_t row = static_cast<size_t>(tr[2]);
    uint2 zr[g6nr::EB];
    __half sc[g6nr::EB];
    #pragma unroll
    for (int b = 0; b < g6nr::EB; ++b)
    {
        if (b < snb)
        {
            const void* zp = A.z + ((static_cast<size_t>(A.n_slots + (m >> 3)) * A.B + b) * 8 + (m & 7)) * HID + n0;
            zr[b] = EVICT ? ld_ef_v2(zp) : *reinterpret_cast<const uint2*>(zp);
            sc[b] = A.t_v3b[(row + b) * geom::HB + hb];
        }
    }
    #pragma unroll
    for (int b = 0; b < g6nr::EB; ++b)
    {
        if (b < snb)
        {
            const float2 z01 = __half22float2(as_h2(zr[b].x));
            const float2 z23 = __half22float2(as_h2(zr[b].y));
            float v[4] = {z01.x, z01.y, z23.x, z23.y};
            had128(v, lane);
            const __half2 c = __half2half2(sc[b]);
            const uint2 s2 = make_uint2(as_u32(__hmul2(as_h2(sg.x), c)), as_u32(__hmul2(as_h2(sg.y), c)));
            const float2 s01 = __half22float2(as_h2(s2.x));
            const float2 s23 = __half22float2(as_h2(s2.y));
            acc[0] += v[0] * s01.x; acc[1] += v[1] * s01.y;
            acc[2] += v[2] * s23.x; acc[3] += v[3] * s23.y;
        }
    }
}

__device__ __forceinline__ bool fin_store_u4(const K1Args& A, int m, int n0, int lane, const float (&acc)[4])
{
    uint2 o;
    o.x = pack_o2(acc[0], acc[1], A.out_bf16);
    o.y = pack_o2(acc[2], acc[3], A.out_bf16);
    const uint2 p = make_uint2(__shfl_xor_sync(0xffffffffu, o.x, 1),
                               __shfl_xor_sync(0xffffffffu, o.y, 1));
    if ((lane & 1) == 0)
    {
        *reinterpret_cast<uint4*>(K1_OUT(A) + static_cast<size_t>(m) * HID + n0) = make_uint4(o.x, o.y, p.x, p.y);
        return true;
    }
    return false;
}

// 5 warps (160t) per (m, hb), 1 barrier, L1::evict_first loads + 16B paired stores (M in {4, 8})
__global__ __launch_bounds__(160, 4)
void k_fin_5w(K1Args A)
{
    __shared__ float4 red[5][32];
    const int plan = A.ctl[2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m = static_cast<int>(blockIdx.x) / geom::HB;
    const int hb = static_cast<int>(blockIdx.x) % geom::HB;
    const int n0 = hb * 128 + lane * 4;
    const uint2 sg = *reinterpret_cast<const uint2*>(A.t_sg + HID + n0);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (warp < 4)
    {
        fin_route_acc<true>(A, plan, m, hb, n0, lane, sg, 2 * warp, acc);
        fin_route_acc<true>(A, plan, m, hb, n0, lane, sg, 2 * warp + 1, acc);
    }
    else
    {
        fin_shared_acc<true>(A, m, hb, n0, lane, sg, acc);
    }
    red[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();

    [[maybe_unused]] bool stored = false;
    if (warp == 0)
    {
        float tot[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int q = 0; q < 5; ++q)
        {
            const float4 v = red[q][lane];
            tot[0] += v.x; tot[1] += v.y; tot[2] += v.z; tot[3] += v.w;
        }
        stored = fin_store_u4(A, m, n0, lane, tot);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[0] = (plan + 1) % A.n_plans;
#if DEC_SLOT >= 2
    if (A.slot.send) dslot::publish(A.slot, stored);
#endif
}

// 1 warp per (m, hb), W warps/CTA, 0 reduction barriers, software-pipelined prefetch (M=16)
template <int W>
__global__ __launch_bounds__(W * 32)
void k_fin_1w(K1Args A)
{
    static_assert(geom::HB % W == 0, "k_fin_1w: W must divide geom::HB");
    const int plan = A.ctl[2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int item = static_cast<int>(blockIdx.x) * W + warp;
    const int m = item / geom::HB, hb = item % geom::HB;
    const int n0 = hb * 128 + lane * 4;
    const uint2 sg = *reinterpret_cast<const uint2*>(A.t_sg + HID + n0);

    auto load_route = [&](int rk, int& nb_o, float& w_o, uint2 (&zr)[g6nr::EB], __half (&sc)[g6nr::EB]) {
        if (rk < 8)
        {
            const int idx = (plan * G6_MX + m) * 8 + rk;
            const int slot = A.tok_slot[idx];
            if (slot < 0) { nb_o = 0; w_o = 0.f; return; }
            w_o = A.tok_w[idx];
            const int col = A.M <= 8 ? m : A.tok_col[idx];
            const long long* tr = A.t_tab + 6 * A.slot_expert[plan * SMAX + slot];
            nb_o = static_cast<int>(tr[3]);
            const size_t row = static_cast<size_t>(tr[2]);
            #pragma unroll
            for (int b = 0; b < g6nr::EB; ++b)
            {
                if (b < nb_o)
                {
                    const void* zp = A.z + ((static_cast<size_t>(slot) * A.B + b) * 8 + col) * HID + n0;
                    zr[b] = ld_ef_v2(zp);
                    sc[b] = A.t_v3b[(row + b) * geom::HB + hb];
                }
            }
        }
        else
        {
            const long long* tr = A.t_tab + 6 * A.sh_e;
            nb_o = static_cast<int>(tr[3]);
            w_o = 1.f;
            const size_t row = static_cast<size_t>(tr[2]);
            #pragma unroll
            for (int b = 0; b < g6nr::EB; ++b)
            {
                if (b < nb_o)
                {
                    const void* zp = A.z + ((static_cast<size_t>(A.n_slots + (m >> 3)) * A.B + b) * 8 + (m & 7)) * HID + n0;
                    zr[b] = ld_ef_v2(zp);
                    sc[b] = A.t_v3b[(row + b) * geom::HB + hb];
                }
            }
        }
    };

    uint2 czr[g6nr::EB], nzr[g6nr::EB];
    __half csc[g6nr::EB], nsc[g6nr::EB];
    int cnb = 0, nnb = 0;
    float cw = 0.f, nw = 0.f;
    load_route(0, cnb, cw, czr, csc);

    float tot[4] = {0.f, 0.f, 0.f, 0.f};
    float p[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int rk = 0; rk < 9; ++rk)
    {
        if (rk + 1 < 9) load_route(rk + 1, nnb, nw, nzr, nsc);
        if ((rk & 1) == 0) p[0] = p[1] = p[2] = p[3] = 0.f;
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b)
        {
            if (b < cnb)
            {
                const float2 z01 = __half22float2(as_h2(czr[b].x));
                const float2 z23 = __half22float2(as_h2(czr[b].y));
                float v[4] = {z01.x, z01.y, z23.x, z23.y};
                had128(v, lane);
                const __half2 c = __half2half2(csc[b]);
                const uint2 s2 = make_uint2(as_u32(__hmul2(as_h2(sg.x), c)), as_u32(__hmul2(as_h2(sg.y), c)));
                const float2 s01 = __half22float2(as_h2(s2.x));
                const float2 s23 = __half22float2(as_h2(s2.y));
                if (rk < 8)
                {
                    p[0] += cw * (v[0] * s01.x); p[1] += cw * (v[1] * s01.y);
                    p[2] += cw * (v[2] * s23.x); p[3] += cw * (v[3] * s23.y);
                }
                else
                {
                    p[0] += v[0] * s01.x; p[1] += v[1] * s01.y;
                    p[2] += v[2] * s23.x; p[3] += v[3] * s23.y;
                }
            }
        }
        if ((rk & 1) == 1 || rk == 8)
        {
            tot[0] += p[0]; tot[1] += p[1]; tot[2] += p[2]; tot[3] += p[3];
        }
        cnb = nnb; cw = nw;
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b) { czr[b] = nzr[b]; csc[b] = nsc[b]; }
    }

    const bool stored = fin_store_u4(A, m, n0, lane, tot);
    if (blockIdx.x == 0 && threadIdx.x == 0) A.ctl[0] = (plan + 1) % A.n_plans;
#if DEC_SLOT >= 2
    if (A.slot.send) dslot::publish(A.slot, stored);
#endif
}

// ------------------------------------------------------------------------------------------------ Strict per-(TP, M) table

inline int rtr_mode(int M)
{
    if constexpr (g6nr::NR == 6 && g6nr::EB == 3)
    {
        if (M == 1 || M == 2 || M == 4) return 1;  // EVICT=false (5/5 non-ovl WIN vs GR8_TR3_WIDE4=1)
        if (M == 8 || M == 16) return 2;           // EVICT=true  (5/5 non-ovl WIN vs GR8_TR3_WIDE4=1)
    }
    else if constexpr (g6nr::NR == 4 && g6nr::EB == 4)
    {
        // M == 1: stock (both arms overlapped on 1/5 rep vs GR8_TR3_WIDE4=1)
        if (M == 2 || M == 8) return 2;            // EVICT=true
        if (M >= 3 && M <= 32) return 1;           // EVICT=false across TP4 S2 widths (3..32, fused with union cap)
    }
    return 0;
}

inline int fin_mode(int M)
{
    if constexpr (g6nr::NR == 6 && g6nr::EB == 3)
    {
        if (M == 4 || M == 8) return 1;            // k_fin_5w
        if (M == 16) return 3;                     // k_fin_1w<8> (5/5 non-ovl WIN vs GR8_TR3_WIDE4=1)
    }
    else if constexpr (g6nr::NR == 4 && g6nr::EB == 4)
    {
        // M <= 8: stock k1_tr3_fin (k_fin_5w did not win 5/5 at NR=4)
        if (M >= 9 && M <= 32) return 2;           // k_fin_1w<4> (0-barrier software-pipelined for M=9..32)
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ Runtime state & launch hooks

struct State
{
    bool on = false;
    RouterScratch* rsc = nullptr;
};

inline cudaError_t setup(State& s)
{
    s = State{};
    if (!knob::moe_rtr_fin_opt()) return cudaSuccess;
    if constexpr ((g6nr::NR != 6 || g6nr::EB != 3) && (g6nr::NR != 4 || g6nr::EB != 4)) return cudaSuccess;
    cudaError_t e = cudaMalloc(&s.rsc, sizeof(RouterScratch));
    if (e != cudaSuccess) return e;
    e = cudaMemset(s.rsc, 0, sizeof(RouterScratch));
    if (e != cudaSuccess) { cudaFree(s.rsc); s.rsc = nullptr; return e; }
    s.on = true;
    return cudaSuccess;
}

inline bool launch_router_plan(const State& s, const bind::Gv& rtr, const float* bias, int M,
                               float* logits, glue::RoutePlan* plan,
                               unsigned keep, const unsigned* keep_dev, cudaStream_t st,
                               int cap_override = -1, int cap_mode_override = -1)
{
    if (!s.on || !s.rsc || M < 1 || M > G6_MX || M > 32 || !bias || !logits || !plan) return false;
    const int mode = rtr_mode(M);
    if (!mode) return false;
    if (!rtr.on || rtr.codec != bind::BF16W || rtr.O != glue::NEXP || rtr.K != geom::HID ||
        rtr.ks != 16 || rtr.M != M || rtr.odt != gv::O_F32 || rtr.ld != glue::NEXP || rtr.y != logits)
        return false;
#ifdef DEC_SLOT
    if (rtr.slot.send) return false;
#endif
    static const int env_cap = static_cast<int>(knob::moe_union_cap());
    static const int env_mode = static_cast<int>(knob::moe_union_mode());
    const int cap = cap_override >= 0 ? cap_override : env_cap;
    const int cap_mode = cap_mode_override >= 0 ? cap_mode_override : env_mode;

    if (M <= 8)
    {
        if (mode == 2) k_rtr_plan_fused_sk1<1, 4, true><<<16, 512, 0, st>>>(rtr.w, rtr.xfrag, bias, M, logits, plan, s.rsc, keep, keep_dev, cap, cap_mode);
        else k_rtr_plan_fused_sk1<1, 4, false><<<16, 512, 0, st>>>(rtr.w, rtr.xfrag, bias, M, logits, plan, s.rsc, keep, keep_dev, cap, cap_mode);
        return true;
    }
    if constexpr (G6_MX >= 16)
    {
        if (M <= 16)
        {
            constexpr bool EV16 = (g6nr::NR == 6 && g6nr::EB == 3);
            k_rtr_plan_fused_sk1<2, 4, EV16><<<16, 512, 0, st>>>(rtr.w, rtr.xfrag, bias, M, logits, plan, s.rsc, keep, keep_dev, cap, cap_mode);
            return true;
        }
    }
    if constexpr (G6_MX >= 24)
    {
        if (M <= 24)
        {
            k_rtr_plan_fused_sk1<3, 4, false><<<16, 512, 0, st>>>(rtr.w, rtr.xfrag, bias, M, logits, plan, s.rsc, keep, keep_dev, cap, cap_mode);
            return true;
        }
    }
    if constexpr (G6_MX >= 32)
    {
        if (M <= 32)
        {
            k_rtr_plan_fused_sk1<4, 4, false><<<16, 512, 0, st>>>(rtr.w, rtr.xfrag, bias, M, logits, plan, s.rsc, keep, keep_dev, cap, cap_mode);
            return true;
        }
    }
    return false;
}

inline bool fin_eligible(const State& s, const bind::K1& k)
{
    if (!s.on || !k.on || !k.sh || k.dense || k.a.steer_dir) return false;
    if (!fin_mode(k.a.M)) return false;
#if DEC_SLOT >= 2
    if (k.a.slot.send)
    {
        if ((reinterpret_cast<uintptr_t>(k.a.slot.send) & 15u) || (k.a.slot.half & 15u) || (k.a.slot.off & 15u))
            return false;
    }
    else
#endif
    {
        if (reinterpret_cast<uintptr_t>(k.a.out) & 15u) return false;
    }
    return true;
}

inline void launch_fin(const K1Args& a, cudaStream_t st)
{
    const int mode = fin_mode(a.M);
    if (mode == 1)
    {
        k_fin_5w<<<a.M * geom::HB, 160, 0, st>>>(a);
    }
    else if (mode == 2 || mode == 3)
    {
        if constexpr (G6_MX >= 16)
        {
            constexpr int W16 = (g6nr::NR == 6 && g6nr::EB == 3) ? 8 : 4;
            k_fin_1w<W16><<<a.M * (geom::HB / W16), W16 * 32, 0, st>>>(a);
        }
    }
}

inline bool launch_mlp(const State& s, const bind::Layer& L, const bind::K1Ovl* ovl, cudaStream_t st)
{
    if (!fin_eligible(s, L.k1)) return false;
    const bind::K1& k = L.k1;
    if (ovl)
    {
        const int g1 = k.a.n_slots * k.a.B * 2 * k.a.KS1, g2 = k.a.n_slots * k.a.B * k.a.NS2;
        bind::k1_pick(k, [&](auto, auto K2, auto, auto CB) {
            if constexpr (g6mx::NT > 1)
            {
                if (k.a.M > 8)
                {
                    if (k.b1 <= 256) k1_tr3_fc1_ovl_nt<true, K1_V(K2), 2, K1_V(CB), 256, 2><<<g1, k.b1, k.sm1, st>>>(k.a);
                    else k1_tr3_fc1_ovl_nt<true, K1_V(K2), 2, K1_V(CB)><<<g1, k.b1, k.sm1, st>>>(k.a);
                    if (ovl->mode == 1) k1_tr3_fc2_ovl_nt<true, K1_V(K2), 2, K1_V(CB)><<<g2, k.b2, k.sm2, st>>>(k.a);
                    return;
                }
            }
            k1_tr3_fc1_ovl<true, K1_V(K2), 2, K1_V(CB)><<<g1, k.b1, k.sm1, st>>>(k.a);
            if (ovl->mode == 1) k1_tr3_fc2_ovl<true, K1_V(K2), 2, K1_V(CB)><<<g2, k.b2, k.sm2, st>>>(k.a);
        });
        cudaStreamWaitEvent(st, ovl->join, 0);
        if (ovl->mode != 1) bind::launch_k1_fc2(k, st);
    }
    else
    {
        if (k.fuse) bind::launch_k1_fused(k, st);
        else
        {
            bind::launch_k1_fc1(k, st);
            bind::launch_k1_fc2(k, st);
        }
    }
    launch_fin(k.a, st);
    dtr::launch(L.glu_t, st);
    dtr::launch(L.down_t, st);
    return true;
}

#else

struct State { bool on = false; };
inline cudaError_t setup(State&) { return cudaSuccess; }
inline bool launch_router_plan(const State&, const bind::Gv&, const float*, int, float*, glue::RoutePlan*,
                               unsigned, const unsigned*, cudaStream_t, int = -1, int = -1) { return false; }
inline bool launch_mlp(const State&, const bind::Layer&, const bind::K1Ovl*, cudaStream_t) { return false; }

#endif

}  // namespace moe_rtr_fin_opt
