// Inkling-2-Small router and expert plan (sm_121a). Our own code. The twin of ink2ref::Model::route
// (engine/ref/ink2/ink2_cpu_ref.h):
//   score[e] = sigmoid(gate_w[e] . x)                          BF16 operands, fp32 sum
//   pick     = the TK experts with the largest score[e] + bias[e], ties to the lower e, in pick order
//   wt[k]    = route_scale * score[pick k] / (sum of the picked scores + 1e-20)        (scores WITHOUT the bias)
// Shapes are arguments: NE % 16 == 0 and NE <= 1024, TK <= 8, H % 16 == 0.
//
// The plan groups the routes by expert so an expert's weights are read once per group of token tiles:
//   slot    one per expert with at least one token, in ascending expert order
//   tile    8 token columns of one expert; slot s owns tiles slot_tile0[s] .. + slot_nt[s]
//   tile_tok / tile_w [tile][8]   the token and the route weight of a column (dead columns: a valid token, weight 0)
//   tile_mask [tile]              the live columns
//   tok_slot / tok_col [m][k]     where route k of token m lives (tok_slot = its tile)
// One plan serves decode and prefill. With M <= 8 tokens an expert has at most 8 of them, so every slot has exactly
// one tile and tile == slot: tile_mask / tile_w / tok_slot are then the per-slot mask, weights and slot index of the
// decode path.
//
// route_pick and the three plan steps are host and device functions with one body; plan_ref runs them on the CPU
// (tests/ink2_moe_host_test.cpp). This header also compiles as plain C++.
#pragma once
#include "nvfp4.cuh"
#include <vector>

#define INK2_HD NV4_HD

namespace ink2
{
using nv4::u32;
using nv4::u64;

constexpr int MAX_NE = 1024, MAX_TK = 8;

INK2_HD float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

// The picks of one token from its scores. idx / wt [TK]; idx is read back while picking.
INK2_HD void route_pick(const float* score, const float* bias, int NE, int TK, float route_scale, int* idx, float* wt)
{
    float sum = 0.0f;
    for (int k = 0; k < TK; ++k)
    {
        int best = -1;
        float bkey = 0.0f;
        for (int e = 0; e < NE; ++e)
        {
            bool used = false;
            for (int j = 0; j < k; ++j) used = used || idx[j] == e;
            if (used) continue;
            const float key = score[e] + bias[e];
            if (best < 0 || key > bkey)
            {
                best = e;
                bkey = key;
            }
        }
        idx[k] = best;
        sum += score[best];
    }
    for (int k = 0; k < TK; ++k) wt[k] = route_scale * score[idx[k]] / (sum + 1e-20f);
}

// ---------------------------------------------------------------------------------------------- plan
struct Plan   // pointers into host or device memory
{
    int* slot_expert;   // [max_slots]
    int* slot_tile0;    // [max_slots]
    int* slot_nt;       // [max_slots]
    int* tile_expert;   // [max_tiles]
    int* tile_mask;     // [max_tiles]
    float* tile_w;      // [max_tiles][8]
    int* tile_tok;      // [max_tiles][8]
    int* tok_slot;      // [M][TK]
    int* tok_col;       // [M][TK]
    int* counts;        // [2] = slots, tiles
};
INK2_HD int plan_max_slots(int M, int TK, int NE) { return M * TK < NE ? M * TK : NE; }
// Every expert has at most one partial tile.
INK2_HD int plan_max_tiles(int M, int TK, int NE)
{
    const int a = M * TK, b = (M * TK + 7) / 8 + NE;
    return a < b ? a : b;
}
// Step 1 (per expert): its routes.
INK2_HD int plan_count(const int* idx, int n, int e)
{
    int c = 0;
    for (int i = 0; i < n; ++i) c += idx[i] == e;
    return c;
}
// Step 2 (serial): slots and first tiles in ascending expert order. tile0 [NE] out.
INK2_HD void plan_slots(const Plan& p, const int* cnt, int NE, int* tile0)
{
    int ns = 0, nt = 0;
    for (int e = 0; e < NE; ++e)
    {
        tile0[e] = nt;
        if (!cnt[e]) continue;
        const int n = (cnt[e] + 7) >> 3;
        p.slot_expert[ns] = e;
        p.slot_tile0[ns] = nt;
        p.slot_nt[ns] = n;
        ++ns;
        nt += n;
    }
    p.counts[0] = ns;
    p.counts[1] = nt;
}
// Step 3 (per expert with cnt > 0): its columns in token order.
INK2_HD void plan_place(const Plan& p, const int* idx, const float* wt, int M, int TK, int e, int tile0, int cnt)
{
    int pos = 0, last = 0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k)
        {
            const int i = m * TK + k;
            if (idx[i] != e) continue;
            const int tile = tile0 + (pos >> 3), col = pos & 7;
            p.tile_tok[tile * 8 + col] = m;
            p.tile_w[tile * 8 + col] = wt[i];
            p.tok_slot[i] = tile;
            p.tok_col[i] = col;
            last = m;
            ++pos;
        }
    const int nt = (cnt + 7) >> 3;
    for (int j = 0; j < nt; ++j)
    {
        p.tile_expert[tile0 + j] = e;
        p.tile_mask[tile0 + j] = j + 1 < nt ? 0xff : (1 << (cnt - 8 * (nt - 1))) - 1;
    }
    for (; pos < nt * 8; ++pos)
    {
        p.tile_tok[tile0 * 8 + pos] = last;
        p.tile_w[tile0 * 8 + pos] = 0.0f;
    }
}

// The plan on the CPU, in vectors.
struct PlanHost
{
    std::vector<int> slot_expert, slot_tile0, slot_nt, tile_expert, tile_mask, tile_tok, tok_slot, tok_col;
    std::vector<float> tile_w;
    int counts[2] = {0, 0};
    Plan view()
    {
        return Plan{slot_expert.data(), slot_tile0.data(), slot_nt.data(), tile_expert.data(), tile_mask.data(), tile_w.data(),
                    tile_tok.data(), tok_slot.data(), tok_col.data(), counts};
    }
    int n_slots() const { return counts[0]; }
    int n_tiles() const { return counts[1]; }
};
inline void plan_ref(const int* idx, const float* wt, int M, int TK, int NE, PlanHost& h)
{
    const size_t ms = static_cast<size_t>(plan_max_slots(M, TK, NE)), mt = static_cast<size_t>(plan_max_tiles(M, TK, NE));
    h.slot_expert.assign(ms, -1);
    h.slot_tile0.assign(ms, -1);
    h.slot_nt.assign(ms, 0);
    h.tile_expert.assign(mt, -1);
    h.tile_mask.assign(mt, 0);
    h.tile_tok.assign(mt * 8, -1);
    h.tile_w.assign(mt * 8, 0.0f);
    h.tok_slot.assign(static_cast<size_t>(M) * TK, -1);
    h.tok_col.assign(static_cast<size_t>(M) * TK, -1);
    std::vector<int> cnt(static_cast<size_t>(NE)), tile0(static_cast<size_t>(NE));
    for (int e = 0; e < NE; ++e) cnt[e] = plan_count(idx, M * TK, e);
    const Plan p = h.view();
    plan_slots(p, cnt.data(), NE, tile0.data());
    for (int e = 0; e < NE; ++e)
        if (cnt[e]) plan_place(p, idx, wt, M, TK, e, tile0[e], cnt[e]);
}

#ifdef __CUDACC__
// ---------------------------------------------------------------------------------------------- kernels
// mma m16n8k16, BF16 operands, fp32 accumulate (the fragment layout of the fp16 form):
//   a0 (row l / 4, k 2 c, 2 c + 1), a1 (row l / 4 + 8), a2 / a3 the same at k + 8;  b0 (k 2 c .., column l / 4), b1 (k + 8)
//   d0, d1 = (row l / 4, columns 2 c, 2 c + 1), d2, d3 = row l / 4 + 8
__device__ __forceinline__ void mma_bf16(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    gr8_sm70::mma16816_bf16(d, a0, a1, a2, a3, b0, b1);
}

// score [M][NE] = sigmoid(x gate_w^T). A warp owns 16 experts and NT token tiles; the gate rows are read once per
// 8 NT tokens. Templated on RW_CTA (warps per CTA) so small NE=128 launches 8 CTAs instead of 2 (100% bit-exact MMA).
constexpr int RW = 4;   // warps per CTA (default for large-N GEMMs in ink2_norm_res.cuh)
template <int NT, int RW_CTA = RW>
__global__ void __launch_bounds__(RW_CTA * 32) k_logits(const uint16_t* x, const uint16_t* gw, int M, int H, int NE, float* score)
{
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int e0 = (blockIdx.x * RW_CTA + (threadIdx.x >> 5)) * 16;
    if (e0 >= NE) return;
    const int m0 = blockIdx.y * NT * 8;
    float acc[NT][4];
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * H + 2 * c;
    }
    const uint16_t* w0 = gw + static_cast<size_t>(e0 + r) * H + 2 * c;
    const uint16_t* w1 = w0 + static_cast<size_t>(8) * H;
    for (int k = 0; k < H; k += 16)
    {
        const u32 a0 = nv4::rd32(w0 + k), a1 = nv4::rd32(w1 + k), a2 = nv4::rd32(w0 + k + 8), a3 = nv4::rd32(w1 + k + 8);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (m0 + t * 8 < M)
                mma_bf16(acc[t], a0, a1, a2, a3, *reinterpret_cast<const u32*>(xr[t] + k), *reinterpret_cast<const u32*>(xr[t] + k + 8));
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), e = e0 + r + 8 * (j >> 1);
            if (m < M) score[static_cast<size_t>(m) * NE + e] = sigmoid(acc[t][j]);
        }
}

// Coalesced 128B-line Tensor Core router logits (H % 64 == 0, 16B-aligned gw): 100% bit-exact with k_logits.
template <int NT, int RW_CTA = 1>
__global__ void __launch_bounds__(RW_CTA * 32) k_logits_c128(const uint16_t* __restrict__ x, const uint16_t* __restrict__ gw,
                                                             int M, int H, int NE, float* __restrict__ score)
{
    __shared__ alignas(16) uint32_t sm_w[RW_CTA][16][36];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e0 = (blockIdx.x * RW_CTA + warp) * 16;
    if (e0 >= NE) return;
    const int m0 = blockIdx.y * NT * 8;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    const int vr = (lane & 3) | ((lane & 16) >> 2), vqp = (lane >> 2) & 3;
    gr8_sm70::Mma16816Acc vacc[NT];
    const uint16_t* vxr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        vacc[t].zero();
        vxr[t] = x + static_cast<size_t>(min(m0 + t * 8 + vr, M - 1)) * H + 4 * vqp;
    }
#else
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * H + 2 * c;
    }
#endif
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = gw + static_cast<size_t>(e0 + rr * 4 + lg) * H;
    for (int k = 0; k < H; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
            const uint4 fv = make_uint4(gr8_sm70::bf16x2_to_f16x2(v.x), gr8_sm70::bf16x2_to_f16x2(v.y),
                                        gr8_sm70::bf16x2_to_f16x2(v.z), gr8_sm70::bf16x2_to_f16x2(v.w));
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 4]) = fv;
#else
            *reinterpret_cast<uint4*>(&sm_w[warp][rr * 4 + lg][lc * 4]) = v;
#endif
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
            const uint2 qa0 = *reinterpret_cast<const uint2*>(&sm_w[warp][vr][p * 8 + 2 * vqp]);
            const uint2 qa1 = *reinterpret_cast<const uint2*>(&sm_w[warp][vr + 8][p * 8 + 2 * vqp]);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M)
                {
                    const u32 xb0 = gr8_sm70::bf16x2_to_f16x2(*reinterpret_cast<const u32*>(vxr[t] + kp));
                    const u32 xb1 = gr8_sm70::bf16x2_to_f16x2(*reinterpret_cast<const u32*>(vxr[t] + kp + 2));
                    vacc[t].mma_prepared(qa0.x, qa0.y, qa1.x, qa1.y, xb0, xb1);
                }
#else
            const u32 a0 = sm_w[warp][r][p * 8 + c];
            const u32 a1 = sm_w[warp][r + 8][p * 8 + c];
            const u32 a2 = sm_w[warp][r][p * 8 + c + 4];
            const u32 a3 = sm_w[warp][r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M)
                    mma_bf16(acc[t], a0, a1, a2, a3, *reinterpret_cast<const u32*>(xr[t] + kp), *reinterpret_cast<const u32*>(xr[t] + kp + 8));
#endif
        }
        __syncwarp();
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    #pragma unroll
    for (int t = 0; t < NT; ++t) vacc[t].flush_store(acc[t]);
#endif
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), e = e0 + r + 8 * (j >> 1);
            if (m < M) score[static_cast<size_t>(m) * NE + e] = sigmoid(acc[t][j]);
        }
}

inline float*& splitk_buf()
{
    static float* p = nullptr;
    return p;
}
inline size_t& splitk_cap()
{
    static size_t n = 0;
    return n;
}
inline bool& splitk_enabled()
{
    static bool on = true;
    return on;
}
// Deterministic split-K factor depending ONLY on (total_tiles, K, total_n) — NEVER on M — for M <= 64!
// Guarantees 100% batch-size invariance (MTP=0 == MTP=K, c=1 == c=4 == c=8).
inline int pick_splitk(int total_tiles, int K, int M, size_t total_n)
{
    if (!splitk_enabled() || M > 64 || (K % 64) != 0) return 1;
    const int K64 = K >> 6;
    if (K64 < 4 || total_tiles >= 512 || total_tiles < 1) return 1;
    constexpr int TARGET_WARPS = 576;
    int sk = (TARGET_WARPS + total_tiles - 1) / total_tiles;
    const int max_sk_k = K64 >> 1;
    if (sk > 24) sk = 24;
    if (sk > max_sk_k) sk = max_sk_k;
    if (sk <= 1) return 1;
    const size_t need64 = static_cast<size_t>(sk) * 64 * total_n;
    if (splitk_buf() == nullptr || need64 > splitk_cap()) return 1;
    return sk;
}

// Split-K coalesced 128B-line Tensor Core router logits: grid (NE/16, ceil(M/(8*NT)), SK), block 32.
// Writes fp32 partials [SK][M][NE] for deterministic reduction + sigmoid.
template <int NT>
__global__ void __launch_bounds__(32) k_logits_c128_splitk(const uint16_t* __restrict__ x, const uint16_t* __restrict__ gw,
                                                           int M, int H, int NE, int SK, float* __restrict__ partial)
{
    __shared__ alignas(16) uint32_t sm_w[16][36];
    const int lane = threadIdx.x & 31;
    const int e0 = blockIdx.x * 16;
    if (e0 >= NE) return;
    const int m0 = blockIdx.y * NT * 8;
    const int sk_id = blockIdx.z;
    const int H64 = H >> 6;
    const int k0 = ((sk_id * H64) / SK) << 6;
    const int k1 = (((sk_id + 1) * H64) / SK) << 6;
    const int r = lane >> 2, c = lane & 3;
    const int lg = lane >> 3, lc = lane & 7;
    float acc[NT][4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    const int vr = (lane & 3) | ((lane & 16) >> 2), vqp = (lane >> 2) & 3;
    gr8_sm70::Mma16816Acc vacc[NT];
    const uint16_t* vxr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        vacc[t].zero();
        vxr[t] = x + static_cast<size_t>(min(m0 + t * 8 + vr, M - 1)) * H + 4 * vqp;
    }
#else
    const uint16_t* xr[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xr[t] = x + static_cast<size_t>(min(m0 + t * 8 + r, M - 1)) * H + 2 * c;
    }
#endif
    const uint16_t* wr[4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        wr[rr] = gw + static_cast<size_t>(e0 + rr * 4 + lg) * H;
    for (int k = k0; k < k1; k += 64)
    {
        #pragma unroll
        for (int rr = 0; rr < 4; ++rr)
        {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(wr[rr] + k) + lc);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
            const uint4 fv = make_uint4(gr8_sm70::bf16x2_to_f16x2(v.x), gr8_sm70::bf16x2_to_f16x2(v.y),
                                        gr8_sm70::bf16x2_to_f16x2(v.z), gr8_sm70::bf16x2_to_f16x2(v.w));
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 4]) = fv;
#else
            *reinterpret_cast<uint4*>(&sm_w[rr * 4 + lg][lc * 4]) = v;
#endif
        }
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            const int kp = k + p * 16;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
            const uint2 qa0 = *reinterpret_cast<const uint2*>(&sm_w[vr][p * 8 + 2 * vqp]);
            const uint2 qa1 = *reinterpret_cast<const uint2*>(&sm_w[vr + 8][p * 8 + 2 * vqp]);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M)
                {
                    const u32 xb0 = gr8_sm70::bf16x2_to_f16x2(*reinterpret_cast<const u32*>(vxr[t] + kp));
                    const u32 xb1 = gr8_sm70::bf16x2_to_f16x2(*reinterpret_cast<const u32*>(vxr[t] + kp + 2));
                    vacc[t].mma_prepared(qa0.x, qa0.y, qa1.x, qa1.y, xb0, xb1);
                }
#else
            const u32 a0 = sm_w[r][p * 8 + c];
            const u32 a1 = sm_w[r + 8][p * 8 + c];
            const u32 a2 = sm_w[r][p * 8 + c + 4];
            const u32 a3 = sm_w[r + 8][p * 8 + c + 4];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (m0 + t * 8 < M)
                    mma_bf16(acc[t], a0, a1, a2, a3, *reinterpret_cast<const u32*>(xr[t] + kp), *reinterpret_cast<const u32*>(xr[t] + kp + 8));
#endif
        }
        __syncwarp();
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    #pragma unroll
    for (int t = 0; t < NT; ++t) vacc[t].flush_store(acc[t]);
#endif
    float* pout = partial + static_cast<size_t>(sk_id) * M * NE;
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), e = e0 + r + 8 * (j >> 1);
            if (m < M) pout[static_cast<size_t>(m) * NE + e] = acc[t][j];
        }
}

__global__ void __launch_bounds__(256) k_logits_splitk_reduce(const float* __restrict__ partial, float* __restrict__ score, int MN, int SK)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= MN) return;
    float s = partial[i];
    for (int k = 1; k < SK; ++k) s += partial[static_cast<size_t>(k) * MN + i];
    score[i] = sigmoid(s);
}

// A thread per token (fallback for NE > 256).
__global__ void __launch_bounds__(128) k_pick(const float* score, const float* bias, int M, int NE, int TK, float route_scale, int* idx, float* wt)
{
    const int m = blockIdx.x * 128 + threadIdx.x;
    if (m < M) route_pick(score + static_cast<size_t>(m) * NE, bias, NE, TK, route_scale, idx + m * TK, wt + m * TK);
}

// Warp-parallel top-TK router pick: 1 warp per token, 4 warps/CTA (or 1 warp/CTA for small M).
// 100% bit-for-bit identical to route_pick (exact tie-breaking to lowest e, exact k=0..TK-1 sum order).
// Supports NE <= 256 (NE = 128 -> 4 experts per lane in registers).
template <int WPC>
__global__ void __launch_bounds__(WPC * 32) k_pick_warp(const float* __restrict__ score, const float* __restrict__ bias,
                                                        int M, int NE, int TK, float route_scale,
                                                        int* __restrict__ idx, float* __restrict__ wt)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int m = blockIdx.x * WPC + warp;
    if (m >= M) return;
    const float* sc_row = score + static_cast<size_t>(m) * NE;
    float sc[8], ky[8];
    unsigned used = 0u;
    const int n_per_lane = (NE + 31) >> 5;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int e = lane + i * 32;
        if (i < n_per_lane && e < NE)
        {
            const float s = sc_row[e];
            sc[i] = s;
            ky[i] = s + bias[e];
        }
        else
        {
            sc[i] = 0.0f;
            ky[i] = -INFINITY;
            used |= (1u << i);
        }
    }

    int my_id = 0;
    float my_sc = 0.0f;
    float sum = 0.0f;
    for (int k = 0; k < TK; ++k)
    {
        int be = -1;
        float bk = -INFINITY, bs = 0.0f;
        int bi = -1;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            if (i < n_per_lane && ((used >> i) & 1u) == 0u)
            {
                const int e = lane + i * 32;
                const float key = ky[i];
                if (be < 0 || key > bk)
                {
                    be = e;
                    bk = key;
                    bs = sc[i];
                    bi = i;
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            const float ok = __shfl_xor_sync(0xffffffffu, bk, off);
            const float os = __shfl_xor_sync(0xffffffffu, bs, off);
            if (oe >= 0 && (be < 0 || ok > bk || (ok == bk && oe < be)))
            {
                be = oe;
                bk = ok;
                bs = os;
            }
        }
        if (bi >= 0 && (lane + bi * 32) == be) used |= (1u << bi);
        if (lane == k)
        {
            my_id = be;
            my_sc = bs;
        }
        sum += bs;
    }
    if (lane < TK)
    {
        idx[m * TK + lane] = my_id;
        wt[m * TK + lane] = route_scale * my_sc / (sum + 1e-20f);
    }
}

// One CTA of NE threads, thread e = expert e (fallback for NE > 128).
__global__ void __launch_bounds__(MAX_NE) k_plan(Plan p, const int* idx, const float* wt, int M, int TK, int NE)
{
    __shared__ int cnt[MAX_NE], tile0[MAX_NE];
    const int e = threadIdx.x;
    cnt[e] = plan_count(idx, M * TK, e);
    __syncthreads();
    if (e == 0) plan_slots(p, cnt, NE, tile0);
    __syncthreads();
    if (cnt[e]) plan_place(p, idx, wt, M, TK, e, tile0[e], cnt[e]);
}

// Warp-parallel exact-order plan builder for NE <= 128: 256 threads (8 warps), zero atomics,
// __match_any_sync bucket counts + warp prefix sum + exact ascending (m, k) placement.
// 100% bit-for-bit identical to plan_ref / k_plan across all M = 1..512.
__global__ void __launch_bounds__(256) k_plan_fast(Plan p, const int* __restrict__ idx, const float* __restrict__ wt, int M, int TK, int NE)
{
    __shared__ int s_wcnt[8][128];
    __shared__ int s_cnt[128];
    __shared__ int s_tile0[128];
    __shared__ int s_last[128];
    __shared__ int s_wsum_s[4];
    __shared__ int s_wsum_t[4];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int N = M * TK;

    // Zero s_wcnt[8][128]
    for (int i = tid; i < 8 * 128; i += 256) (&s_wcnt[0][0])[i] = 0;
    __syncthreads();

    // Contiguous chunk per warp so ascending (m, k) order is strictly preserved!
    const int c0 = (N * warp) >> 3;
    const int c1 = (N * (warp + 1)) >> 3;

    // Pass 1: count occurrences of each expert inside warp's chunk
    for (int base = c0; base < c1; base += 32)
    {
        const int i = base + lane;
        const bool valid = (i < c1);
        const unsigned active = __ballot_sync(0xffffffffu, valid);
        if (valid)
        {
            const int e = idx[i];
            const unsigned m = __match_any_sync(active, e);
            if (lane == (__ffs(m) - 1)) s_wcnt[warp][e] += __popc(m);
        }
    }
    __syncthreads();

    // Prefix-sum across 8 warps per expert, then parallel warp prefix-sum across experts 0..NE-1
    if (tid < NE)
    {
        const int e = tid;
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w)
        {
            const int c = s_wcnt[w][e];
            s_wcnt[w][e] = sum;
            sum += c;
        }
        s_cnt[e] = sum;
    }
    __syncthreads();

    if (tid < 128)
    {
        const int e = tid;
        const int cnt = (e < NE) ? s_cnt[e] : 0;
        const int has = cnt > 0 ? 1 : 0;
        const int nt = cnt > 0 ? ((cnt + 7) >> 3) : 0;
        int ps = has, pt = nt;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1)
        {
            const int os = __shfl_up_sync(0xffffffffu, ps, off);
            const int ot = __shfl_up_sync(0xffffffffu, pt, off);
            if (lane >= off) { ps += os; pt += ot; }
        }
        if (lane == 31)
        {
            s_wsum_s[warp] = ps;
            s_wsum_t[warp] = pt;
        }
        __syncthreads();
        int base_s = 0, base_t = 0;
        for (int w = 0; w < warp; ++w) { base_s += s_wsum_s[w]; base_t += s_wsum_t[w]; }
        const int slot_e = base_s + ps - has;
        const int tile0_e = base_t + pt - nt;
        if (e < NE)
        {
            s_tile0[e] = tile0_e;
            if (cnt > 0)
            {
                p.slot_expert[slot_e] = e;
                p.slot_tile0[slot_e] = tile0_e;
                p.slot_nt[slot_e] = nt;
                for (int j = 0; j < nt; ++j)
                {
                    p.tile_expert[tile0_e + j] = e;
                    p.tile_mask[tile0_e + j] = (j + 1 < nt) ? 0xff : ((1 << (cnt - 8 * (nt - 1))) - 1);
                }
            }
            if (e == NE - 1)
            {
                p.counts[0] = slot_e + has;
                p.counts[1] = tile0_e + nt;
            }
        }
    }
    else
    {
        // Match the __syncthreads inside the if (tid < 128) branch!
        __syncthreads();
    }
    __syncthreads();

    // Pass 2: scatter in exact ascending (m, k) order
    for (int base = c0; base < c1; base += 32)
    {
        const int i = base + lane;
        const bool valid = (i < c1);
        const unsigned active = __ballot_sync(0xffffffffu, valid);
        if (valid)
        {
            const int e = idx[i];
            const unsigned m = __match_any_sync(active, e);
            const int r = __popc(m & ((1u << lane) - 1));
            const int base_pos = s_wcnt[warp][e];
            __syncwarp(active);
            if (lane == (__ffs(m) - 1)) s_wcnt[warp][e] = base_pos + __popc(m);
            const int pos = base_pos + r;
            const int tile = s_tile0[e] + (pos >> 3);
            const int col = pos & 7;
            const int tok_m = i / TK;
            p.tile_tok[tile * 8 + col] = tok_m;
            p.tile_w[tile * 8 + col] = wt[i];
            p.tok_slot[i] = tile;
            p.tok_col[i] = col;
            if (pos == s_cnt[e] - 1) s_last[e] = tok_m;
        }
    }
    __syncthreads();

    // Dead-column padding for partial final tiles (exact match to plan_place)
    if (tid < NE)
    {
        const int cnt = s_cnt[tid];
        if (cnt > 0)
        {
            const int nt = (cnt + 7) >> 3;
            const int t0 = s_tile0[tid];
            const int last = s_last[tid];
            for (int pos = cnt; pos < nt * 8; ++pos)
            {
                p.tile_tok[t0 * 8 + pos] = last;
                p.tile_w[t0 * 8 + pos] = 0.0f;
            }
        }
    }
}

inline void route_check(int M, int H, int NE, int TK)
{
    if (M < 1 || H % 16 || NE % 16 || NE < 16 || NE > MAX_NE || TK < 1 || TK > MAX_TK || TK > NE)
        throw std::runtime_error("ink2 router: M < 1, H % 16, NE % 16, NE > 1024 or TK > 8");
}
// routes -> plan
inline bool& fast_router_enabled()
{
    static bool on = true;
    return on;
}
inline void plan(const Plan& p, const int* idx, const float* wt, int M, int NE, int TK, cudaStream_t st)
{
    if (fast_router_enabled() && NE <= 128) k_plan_fast<<<1, 256, 0, st>>>(p, idx, wt, M, TK, NE);
    else k_plan<<<1, NE, 0, st>>>(p, idx, wt, M, TK, NE);
}
// x [M][H] BF16 -> score [M][NE], idx / wt [M][TK], and the plan.
inline void route(const uint16_t* x, const uint16_t* gate_w, const float* bias, int M, int H, int NE, int TK, float route_scale, float* score,
                  int* idx, float* wt, const Plan& p, cudaStream_t st)
{
    route_check(M, H, NE, TK);
    if (fast_router_enabled())
    {
        const unsigned gx1 = static_cast<unsigned>(NE / 16);
        const bool c128 = (H % 64 == 0) && ((reinterpret_cast<uintptr_t>(gate_w) & 15u) == 0);
        if (c128)
        {
            const int sk = pick_splitk(NE / 16, H, M, static_cast<size_t>(NE));
            if (sk > 1)
            {
                float* pbuf = splitk_buf();
                if (M <= 8) k_logits_c128_splitk<1><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, gate_w, M, H, NE, sk, pbuf);
                else if (M <= 16) k_logits_c128_splitk<2><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, gate_w, M, H, NE, sk, pbuf);
                else if (M <= 32) k_logits_c128_splitk<4><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, gate_w, M, H, NE, sk, pbuf);
                else k_logits_c128_splitk<8><<<dim3(gx1, 1, static_cast<unsigned>(sk)), 32, 0, st>>>(x, gate_w, M, H, NE, sk, pbuf);
                const int mn = M * NE;
                k_logits_splitk_reduce<<<(mn + 255) / 256, 256, 0, st>>>(pbuf, score, mn, sk);
            }
            else if (M <= 64) k_logits_c128<1, 1><<<dim3(gx1, (M + 7) / 8), 32, 0, st>>>(x, gate_w, M, H, NE, score);
            else k_logits_c128<4, 1><<<dim3(gx1, (M + 31) / 32), 32, 0, st>>>(x, gate_w, M, H, NE, score);
        }
        else
        {
            if (M <= 64) k_logits<1, 1><<<dim3(gx1, (M + 7) / 8), 32, 0, st>>>(x, gate_w, M, H, NE, score);
            else k_logits<4, 1><<<dim3(gx1, (M + 31) / 32), 32, 0, st>>>(x, gate_w, M, H, NE, score);
        }
        if (NE <= 256)
        {
            if (M <= 16) k_pick_warp<1><<<M, 32, 0, st>>>(score, bias, M, NE, TK, route_scale, idx, wt);
            else k_pick_warp<4><<<(M + 3) / 4, 128, 0, st>>>(score, bias, M, NE, TK, route_scale, idx, wt);
        }
        else k_pick<<<(M + 127) / 128, 128, 0, st>>>(score, bias, M, NE, TK, route_scale, idx, wt);
        plan(p, idx, wt, M, NE, TK, st);
        return;
    }
    if (M <= 8) k_logits<1><<<dim3((NE / 16 + RW - 1) / RW, 1), RW * 32, 0, st>>>(x, gate_w, M, H, NE, score);
    else k_logits<4><<<dim3((NE / 16 + RW - 1) / RW, (M + 31) / 32), RW * 32, 0, st>>>(x, gate_w, M, H, NE, score);
    k_pick<<<(M + 127) / 128, 128, 0, st>>>(score, bias, M, NE, TK, route_scale, idx, wt);
    plan(p, idx, wt, M, NE, TK, st);
}
#endif   // __CUDACC__
}   // namespace ink2
