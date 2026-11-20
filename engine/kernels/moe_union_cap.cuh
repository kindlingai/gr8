// Batch-budgeted union-capped MoE router plan for GLM-5.3 decode
// (GR8_MOE_UNION_CAP=B, default 0 = off; GR8_MOE_UNION_MODE=0..8, default 0).
//
// When GR8_MOE_UNION_CAP == 0 (default), nothing in this header is launched and
// the stock glue::k_router_plan / moe_rtr_fin_opt path executes unchanged.
//
// When GR8_MOE_UNION_CAP = B > 0:
//   1. Each active token m computes its exact stock top-8 experts and normalized
//      gate weights (sum = ROUTE_SCALE).
//   2. If the union of distinct active experts across the batch is already <= B,
//      capping and renormalization are bypassed (0-ULP bit-exact to stock).
//   3. Otherwise, each active expert e in 0..255 receives a deterministic
//      batch-importance score (computed in fixed ascending m order, zero atomics,
//      identical across all TP ranks) and a 256-thread bitonic sort keeps the
//      exact top-B experts (tie-break: lower expert id wins).
//   4. Rows whose top-8 subset was truncated have surviving weights scaled by
//      GR8_MOE_UNION_MODE / 3:
//        0 = L2 variance-preserving (sqrt(sum_all w^2 / sum_kept w^2))
//        1 = L1 sum-preserving      (ROUTE_SCALE / sum_kept w)
//        2 = none                   (keep original w for surviving experts)
//      Untruncated rows (all 8 experts kept, including row 0 in anchor0 mode)
//      retain their exact original weights (0-ULP). Dropped routes emit the
//      standard RoutePlan no-route encoding (tok_slot = -1), which every K1
//      consumer (k1_tr3_fin, k_fin_5w, k_fin_1w) skips natively.
//
// Priority policy = GR8_MOE_UNION_MODE % 3:
//   0 = balanced:  S_e = sum_m w_{m,e}, plus +100 bonus for any active row's
//                  rank-0 expert (protects every session/row from starvation).
//   1 = anchor0:   S_e = sum_m (w_{m,e} / (1 + 0.25*m)), plus +1000 for row 0's
//                  top-8 experts (row 0 is 100% exact when B >= 8) and +100 for
//                  any active row's rank-0 expert. Ideal for solo spec verify.
//   2 = pure_mass: S_e = sum_m w_{m,e} with no priority bonuses (ablation).
//
// Self-contained: leaves glue.cuh, k1.cuh, tr3.cuh, bind.cuh, and
// moe_rtr_fin_opt.cuh unmodified.
#pragma once
#include <cmath>
#include <cstdint>
#include "bind.cuh"
#include "glue.cuh"
#include "../knobs-rs/gr8_knobs.h"

namespace moe_union_cap
{

#if !G6_MODEL_FLASH && !G6_MODEL_DS41 && !G6_MODEL_INK2

static_assert(glue::NEXP == 256, "k_router_plan_capped assumes NEXP == 256 for 256-thread bitonic sort");

__global__ void __launch_bounds__(glue::NEXP) k_router_plan_capped(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int M,
    glue::RoutePlan* __restrict__ P,
    int cap,
    int mode,
    unsigned keep = 0u,
    const unsigned* keep_dev = nullptr)
{
    constexpr int NEXP = glue::NEXP;
    constexpr int TOPK = glue::TOPK;
    constexpr int MMAX = glue::MMAX;
    constexpr int RW = glue::RW;

    __shared__ unsigned emask[NEXP];
    __shared__ int sel_e[MMAX][TOPK];
    __shared__ float sel_w[MMAX][TOPK];
    __shared__ int slot_of[NEXP];
    __shared__ int wcount[RW];
    __shared__ float s_m[8][NEXP], k_m[8][NEXP];
    __shared__ float bs_val[NEXP];
    __shared__ int bs_id[NEXP];
    __shared__ unsigned char kept_exp[NEXP];
    __shared__ int s_n_distinct;

    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const bool dense = M > 8;
    const unsigned on = keep_dev ? *keep_dev : keep ? keep : 0xffffffffu;
    const unsigned m_bits = M >= 32 ? 0xffffffffu : ((1u << M) - 1u);
    const unsigned active_mask = on & m_bits;

    emask[t] = 0u;
    kept_exp[t] = 0;

    if (!dense)
    {
        const float bt = bias[t];
        for (int m = 0; m < M; ++m)
        {
            const float st = 1.f / (1.f + expf(-logits[m * NEXP + t]));
            s_m[m][t] = st;
            k_m[m][t] = st + bt;
        }
    }
    for (int i = t; i < M * TOPK * 8; i += NEXP)
    {
        (&P->slot_w[0][0])[i] = 0.f;
        (&P->slot_tok[0][0])[i] = -1;
    }
    for (int i = M * TOPK + t; i < MMAX * TOPK; i += NEXP)
    {
        const int m = i >> 3, k = i & 7;
        P->tok_slot[m][k] = -1;
        P->tok_col[m][k] = 0;
        P->tok_w[m][k] = 0.f;
        P->tok_expert[m][k] = -1;
    }
    __syncthreads();

    // 1. Exact stock per-token top-8 selection & normalization
    for (int m = w; m < M; m += RW)
    {
        if (!(active_mask >> m & 1u)) continue;
        float s[RW], key[RW];
        #pragma unroll
        for (int i = 0; i < RW; ++i)
        {
            const int e = l + 32 * i;
            if (!dense)
            {
                key[i] = k_m[m][e];
            }
            else
            {
                s[i] = 1.f / (1.f + expf(-logits[m * NEXP + e]));
                key[i] = s[i] + bias[e];
            }
        }
        unsigned taken = 0u;
        float my_k = key[0];
        int my_e = l;
        #pragma unroll
        for (int i = 1; i < RW; ++i)
            if (key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }

        for (int k = 0; k < TOPK; ++k)
        {
            float bk = my_k;
            int be = my_e;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                if (ok > bk || (ok == bk && oe < be)) { bk = ok; be = oe; }
            }
            if (!dense)
            {
                if (l == 0) { sel_e[m][k] = be; sel_w[m][k] = s_m[m][be]; }
                if (k + 1 < TOPK && (be & 31) == l)
                {
                    taken |= 1u << (be >> 5);
                    my_k = -INFINITY;
                    my_e = 0x7fffffff;
                    #pragma unroll
                    for (int i = 0; i < RW; ++i)
                        if (!(taken >> i & 1u) && key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }
                }
            }
            else
            {
                float sv = 0.f;
                if ((be & 31) == l)
                {
                    taken |= 1u << (be >> 5);
                    sv = s[be >> 5];
                    my_k = -INFINITY;
                    my_e = 0x7fffffff;
                    #pragma unroll
                    for (int i = 0; i < RW; ++i)
                        if (!(taken >> i & 1u) && key[i] > my_k) { my_k = key[i]; my_e = l + 32 * i; }
                }
                sv = __shfl_sync(0xffffffffu, sv, be & 31);
                if (l == 0) { sel_e[m][k] = be; sel_w[m][k] = sv; }
            }
        }
        __syncwarp();
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) sum += sel_w[m][k];
        __syncwarp();
        if (l < TOPK)
        {
            const float wv = sel_w[m][l] / sum * glue::ROUTE_SCALE;
            sel_w[m][l] = wv;
            atomicOr(&emask[sel_e[m][l]], 1u << m);
        }
        __syncwarp();
    }
    __syncthreads();

    // 2. Count distinct experts in the batch union
    const unsigned em0 = emask[t];
    {
        const unsigned wm0 = __ballot_sync(0xffffffffu, em0 != 0u);
        if (l == 0) wcount[w] = __popc(wm0);
    }
    __syncthreads();
    if (t == 0)
    {
        int nd = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) nd += wcount[i];
        s_n_distinct = nd;
    }
    __syncthreads();

    // 3. Cap union to at most `cap` distinct experts if exceeded
    if (s_n_distinct <= cap)
    {
        kept_exp[t] = (em0 != 0u) ? 1 : 0;
        __syncthreads();
    }
    else
    {
        const int policy = mode % 3;
        const int renorm = mode / 3;
        float my_score = -INFINITY;
        if (em0 != 0u)
        {
            float sc = 0.f;
            bool is_anchor0 = false;
            bool is_top1 = false;
            unsigned tmp = em0;
            // Fixed ascending m order -> bitwise identical across all TP ranks (zero float atomics)
            while (tmp)
            {
                const int m = __ffs(tmp) - 1;
                tmp &= tmp - 1u;
                int rk = -1;
                #pragma unroll
                for (int k = 0; k < TOPK; ++k)
                {
                    if (sel_e[m][k] == t) { rk = k; break; }
                }
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
            if (policy == 0)
            {
                if (is_top1) sc += 100.f;
            }
            else if (policy == 1)
            {
                if (is_anchor0) sc += 1000.f;
                if (is_top1) sc += 100.f;
            }
            my_score = sc;
        }

        // 256-thread bitonic sort: (score DESC, expert_id ASC)
        bs_val[t] = my_score;
        bs_id[t] = t;
        __syncthreads();
        for (int k = 2; k <= NEXP; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                const int ixj = t ^ j;
                if (ixj > t)
                {
                    const bool desc_dir = (t & k) == 0;
                    const float va = bs_val[t], vb = bs_val[ixj];
                    const int ia = bs_id[t], ib = bs_id[ixj];
                    const bool a_better = (va > vb) || (va == vb && ia < ib);
                    const bool swap = desc_dir ? !a_better : a_better;
                    if (swap)
                    {
                        bs_val[t] = vb; bs_val[ixj] = va;
                        bs_id[t] = ib; bs_id[ixj] = ia;
                    }
                }
                __syncthreads();
            }
        }

        if (t < cap && bs_val[t] > -INFINITY) kept_exp[bs_id[t]] = 1;
        __syncthreads();

        if (!kept_exp[t]) emask[t] = 0u;
        __syncthreads();

        // Renormalize surviving gate weights for truncated rows (untruncated rows stay 0-ULP exact)
        for (int m = w; m < M; m += RW)
        {
            if (!(active_mask >> m & 1u)) continue;
            if (l == 0)
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

    // 4. Prefix-sum kept experts into ascending RoutePlan slots (identical layout to stock k_router_plan)
    const unsigned em = emask[t];
    int nsl, exc;
    if (!dense)
    {
        nsl = em != 0u;
        const unsigned wm = __ballot_sync(0xffffffffu, nsl);
        exc = __popc(wm & ((1u << l) - 1u));
        if (l == 0) wcount[w] = __popc(wm);
    }
    else
    {
        const int cnt = __popc(em);
        nsl = (cnt + 7) >> 3;
        int scan = nsl;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            const int v = __shfl_up_sync(0xffffffffu, scan, o);
            if (l >= o) scan += v;
        }
        exc = scan - nsl;
        if (l == 31) wcount[w] = scan;
    }
    __syncthreads();
    int base = 0;
    #pragma unroll
    for (int i = 0; i < RW; ++i) if (i < w) base += wcount[i];
    const int slot0 = base + exc;
    if (nsl)
    {
        slot_of[t] = slot0;
        if (!dense)
        {
            P->slot_expert[slot0] = t;
            P->slot_mask[slot0] = static_cast<int>(em);
        }
        else
        {
            const int cnt = __popc(em);
            for (int j = 0; j < nsl; ++j)
            {
                P->slot_expert[slot0 + j] = t;
                P->slot_mask[slot0 + j] = static_cast<int>((1u << (cnt - 8 * j < 8 ? cnt - 8 * j : 8)) - 1u);
            }
        }
    }
    if (t == 0)
    {
        int n = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) n += wcount[i];
        P->n_slots = n;
        if (n == 0) P->slot_expert[0] = 0;
    }
    __syncthreads();

    // 5. Scatter per-token routes (dropped routes carry tok_slot = -1)
    for (int i = t; i < M * TOPK; i += NEXP)
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
        const int r = __popc(emask[e] & ((1u << m) - 1u));
        const int s2 = slot_of[e] + (dense ? r >> 3 : 0);
        const int col = dense ? (r & 7) : m;
        P->tok_slot[m][k] = s2;
        P->tok_col[m][k] = col;
        P->tok_w[m][k] = sel_w[m][k];
        P->tok_expert[m][k] = e;
        P->slot_w[s2][col] = sel_w[m][k];
        P->slot_tok[s2][col] = m;
    }
}

inline bool launch_router_plan(const bind::Gv& rtr, const float* bias, int M,
                               float* logits, glue::RoutePlan* plan,
                               unsigned keep, const unsigned* keep_dev, cudaStream_t st)
{
    static const int cap = static_cast<int>(knob::moe_union_cap());
    if (cap <= 0) return false;
    // If M == 1 and cap >= TOPK (8), capping can never trigger -> fall through to exact stock / opt path.
    if (M <= 1 && cap >= glue::TOPK) return false;
    static const int mode = static_cast<int>(knob::moe_union_mode());
    bind::launch(rtr, st);
    k_router_plan_capped<<<1, glue::NEXP, 0, st>>>(logits, bias, M, plan, cap, mode, keep, keep_dev);
    return true;
}

#else

inline bool launch_router_plan(const bind::Gv&, const float*, int, float*, glue::RoutePlan*,
                               unsigned, const unsigned*, cudaStream_t)
{
    return false;
}

#endif

}  // namespace moe_union_cap
