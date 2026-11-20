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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// score [M][NE] = sigmoid(x gate_w^T). A warp owns 16 experts and NT token tiles; the gate rows are read once per
// 8 NT tokens.
constexpr int RW = 4;   // warps per CTA
template <int NT>
__global__ void __launch_bounds__(RW * 32) k_logits(const uint16_t* x, const uint16_t* gw, int M, int H, int NE, float* score)
{
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int e0 = (blockIdx.x * RW + (threadIdx.x >> 5)) * 16;
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

// A thread per token.
__global__ void __launch_bounds__(128) k_pick(const float* score, const float* bias, int M, int NE, int TK, float route_scale, int* idx, float* wt)
{
    const int m = blockIdx.x * 128 + threadIdx.x;
    if (m < M) route_pick(score + static_cast<size_t>(m) * NE, bias, NE, TK, route_scale, idx + m * TK, wt + m * TK);
}

// One CTA of NE threads, thread e = expert e.
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

inline void route_check(int M, int H, int NE, int TK)
{
    if (M < 1 || H % 16 || NE % 16 || NE < 16 || NE > MAX_NE || TK < 1 || TK > MAX_TK || TK > NE)
        throw std::runtime_error("ink2 router: M < 1, H % 16, NE % 16, NE > 1024 or TK > 8");
}
// routes -> plan
inline void plan(const Plan& p, const int* idx, const float* wt, int M, int NE, int TK, cudaStream_t st)
{
    k_plan<<<1, NE, 0, st>>>(p, idx, wt, M, TK, NE);
}
// x [M][H] BF16 -> score [M][NE], idx / wt [M][TK], and the plan.
inline void route(const uint16_t* x, const uint16_t* gate_w, const float* bias, int M, int H, int NE, int TK, float route_scale, float* score,
                  int* idx, float* wt, const Plan& p, cudaStream_t st)
{
    route_check(M, H, NE, TK);
    if (M <= 8) k_logits<1><<<dim3((NE / 16 + RW - 1) / RW, 1), RW * 32, 0, st>>>(x, gate_w, M, H, NE, score);
    else k_logits<4><<<dim3((NE / 16 + RW - 1) / RW, (M + 31) / 32), RW * 32, 0, st>>>(x, gate_w, M, H, NE, score);
    k_pick<<<(M + 127) / 128, 128, 0, st>>>(score, bias, M, NE, TK, route_scale, idx, wt);
    plan(p, idx, wt, M, NE, TK, st);
}
#endif   // __CUDACC__
}   // namespace ink2
