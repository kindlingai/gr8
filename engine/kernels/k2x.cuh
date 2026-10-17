// K2x: the alternative decode MoE design of issue #8, for the A/B against K1 (k1.cuh). exlr8 v2 layers only: E4M3
// experts, the trellis shared expert (tr3.table row 256), block-scale hidden-side vectors (tr3.sg / v1b / v3b).
// Written from the issue's description and the g6 tree; it reuses the trellis decode of tr3.cuh / tr3_wide.cuh.
//
//   plan     one CTA counts each expert's routes with shared-memory atomics and places every (token, expert) pair into
//            worst-case-sized grids (per-expert rows, route -> row). The other CTAs of the same launch rotate each
//            token once: xr = f16(H128(x * tr3.sg)). On block-scale layers the per-row scale is constant over a
//            128 block, so H128(x * sg * c) = c * H128(x * sg): fc1 multiplies the fp16 fragment by c (in fp16, after
//            the rounding; K1 rounds once, f16(H128(x * sg * c))).
//   fc1      work item = one expert x one R-row block of its intermediate (R = 16, 32 or 64), gate and up, full K.
//            Trellis tiles decode into registers as the mma B operand (the tokens are the A operand, 16 per mma), so an
//            expert's tokens (up to 32 at G6_MX = 32) share every decoded tile. Writes f16(gate), f16(up) per row.
//   fc2      work item = one expert x OB blocks of 128 output columns. Prologue as K1's (H128, svh, silu * up, suh_d,
//            H128); the per-block output scales are folded into the fp32 sum (sum_b c_b H(z_b) = H(sum_b c_b z_b)), one
//            H128 per output block (K1 rounds each block's z to fp16 before its H128; K2x keeps the fp32 sum); each route's row is written as int8 with one fp32 scale per 128 columns (or fp16
//            with f16rows = 1).
//   combine  out[m] = sum_k w[m][k] row(m, k) + shared row(m), fp32, fixed order, written as BF16 (or fp16).
// Every grid is sized for the worst case of M rows (CTAs past the plan's experts exit), so the step can be captured in
// a CUDA graph. Not bit-identical to K1 (other summation order, int8 rows); k2x_ab in row_budget_test measures the
// difference and the time.
#pragma once
#include "tr3.cuh"

namespace k2x
{
constexpr int NE = 257;                     // 256 routed experts + the trellis shared expert
constexpr int MAXR = 9 * G6_MX;             // rows: 8 routes + the shared expert per token
constexpr int EBW = g6nr::EB * 128;         // intermediate columns per expert on a rank
static_assert(G6_MX <= 32, "k2x: at most 32 rows per expert (two 16-row mma tiles, the 32-row fc2 stage)");

struct Plan
{
    int nex, nrows, pad_[2];
    int ex_e[NE], ex_n[NE], ex_base[NE];    // expert of list entry i, its rows, its first row
    int row_tok[MAXR];                      // row -> token
    int ridx[G6_MX * 9];                    // (token m, route k) -> row; k = 8: the shared expert's row
};

struct Args
{
    const int* tok_expert = nullptr;        // [G6_MX][8] (glue::RoutePlan::tok_expert)
    const float* tok_w = nullptr;           // [G6_MX][8]
    const __half* x = nullptr;              // [M][6144] mlp_norm output
    int M = 0;
    const long long* tab = nullptr;         // tr3.table [257][6]
    const long long* kmix = nullptr;        // tr3.kmix or nullptr
    const uint8_t *fc1 = nullptr, *fc2 = nullptr;
    int wide1 = 0, wide2 = 0;               // fc1 / fc2 in the wide order (bind::widen_fc1 / widen_fc2)
    const __half* v2 = nullptr;             // [rows][3][128]
    const __half* sg = nullptr;             // [2][6144] input, output signs
    const __half* v1b = nullptr;            // [rows][2][48]
    const __half* v3b = nullptr;            // [rows][48]
    Plan* plan = nullptr;
    u32* xr = nullptr;                      // [G6_MX][KSH][4][2] rotated tokens, (a0, a2) pairs per lane quad
    __half* h = nullptr;                    // [MAXR][2][EBW] f16(gate), f16(up) before the output-side H128
    int8_t* y8 = nullptr;                   // [MAXR][6144]
    float* ysc = nullptr;                   // [MAXR][48]
    __half* y16 = nullptr;                  // [MAXR][6144] (f16rows)
    void* out = nullptr;                    // [M][6144]
    int out_bf16 = 1;
    int R = 64, OB = 2, f16rows = 0;
};

inline size_t scratch_bytes()
{
    return sizeof(Plan) + 256 + static_cast<size_t>(G6_MX) * HID * 2 + static_cast<size_t>(MAXR) * 2 * EBW * 2 +
           static_cast<size_t>(MAXR) * HID + static_cast<size_t>(MAXR) * 48 * 4 + static_cast<size_t>(MAXR) * HID * 2 + 1024;
}
inline void scratch_at(uint8_t* p, Args& a)
{
    auto take = [&](size_t n) { uint8_t* q = p; p += (n + 255) & ~static_cast<size_t>(255); return q; };
    a.plan = reinterpret_cast<Plan*>(take(sizeof(Plan)));
    a.xr = reinterpret_cast<u32*>(take(static_cast<size_t>(G6_MX) * HID * 2));
    a.h = reinterpret_cast<__half*>(take(static_cast<size_t>(MAXR) * 2 * EBW * 2));
    a.y8 = reinterpret_cast<int8_t*>(take(static_cast<size_t>(MAXR) * HID));
    a.ysc = reinterpret_cast<float*>(take(static_cast<size_t>(MAXR) * 48 * 4));
    a.y16 = reinterpret_cast<__half*>(take(static_cast<size_t>(MAXR) * HID * 2));
}

// ------------------------------------------------------------------------------------------ plan + rotation
__global__ void __launch_bounds__(256) k_plan_rot(Args A)
{
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (blockIdx.x == 0)
    {
        __shared__ int cnt[NE], base[NE], rpos[G6_MX * 8];
        __shared__ int nex, nrows;
        Plan* P = A.plan;
        for (int i = t; i < NE; i += 256) cnt[i] = 0;
        if (t == 0) { nex = 0; nrows = 0; }
        __syncthreads();
        for (int i = t; i < A.M * 8; i += 256) rpos[i] = atomicAdd(&cnt[A.tok_expert[i]], 1);
        if (t == 0) cnt[256] = A.M;
        __syncthreads();
        for (int e = t; e < NE; e += 256)
            if (cnt[e] > 0)
            {
                const int i = atomicAdd(&nex, 1), b = atomicAdd(&nrows, cnt[e]);
                base[e] = b;
                P->ex_e[i] = e; P->ex_n[i] = cnt[e]; P->ex_base[i] = b;
            }
        __syncthreads();
        for (int i = t; i < A.M * 8; i += 256)
        {
            const int r = base[A.tok_expert[i]] + rpos[i];
            P->row_tok[r] = i >> 3;
            P->ridx[(i >> 3) * 9 + (i & 7)] = r;
        }
        for (int m = t; m < A.M; m += 256)
        {
            const int r = base[256] + m;
            P->row_tok[r] = m;
            P->ridx[m * 9 + 8] = r;
        }
        if (t == 0) { P->nex = nex; P->nrows = nrows; }
        return;
    }
    const int task = (blockIdx.x - 1) * 8 + warp, m = task / 48, blk = task % 48;
    if (m >= A.M) return;
    const int k = blk * 128 + lane * 4;
    const uint2 xr = *reinterpret_cast<const uint2*>(A.x + static_cast<size_t>(m) * HID + k);
    const uint2 sr = *reinterpret_cast<const uint2*>(A.sg + k);
    const float2 x01 = __half22float2(as_h2(xr.x)), x23 = __half22float2(as_h2(xr.y));
    const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
    float v[4] = {x01.x * s01.x, x01.y * s01.y, x23.x * s23.x, x23.y * s23.y};
    had128(v, lane);
    const int ks = blk * 8 + (lane >> 2), q0 = (lane & 3) * 4, reg = q0 >= 8, c0 = (q0 & 7) >> 1;
    u32* o = A.xr + (static_cast<size_t>(m) * KSH + ks) * 8;
    o[c0 * 2 + reg] = pack_h2(v[0], v[1]);
    o[(c0 + 1) * 2 + reg] = pack_h2(v[2], v[3]);
}

// ------------------------------------------------------------------------------------------ fc1
// One warp: tile column j of one matrix (gate | up) of block b, k-steps [ks0, ks0 + nks), all of the expert's rows.
template <int BITS, bool WIDE, int G>
__device__ __forceinline__ void fc1_warp(const Args& A, const uint8_t* mw, int j, int ks0, int nks, int base, int n,
                                         const __half* sc, float (&acc)[2][2][4])
{
    constexpr int TB = 32 * BITS;
    const int lane = threadIdx.x & 31, c = lane & 3, rl = lane >> 2;
    const Plan* P = A.plan;
    const uint2* xl[G];
    const uint2* xh[G];
    bool hon[G];
    #pragma unroll
    for (int g = 0; g < G; ++g)
    {
        const int r = g * 16 + rl;
        xl[g] = reinterpret_cast<const uint2*>(A.xr + static_cast<size_t>(P->row_tok[base + min(r, n - 1)]) * KSH * 8) + c;
        xh[g] = reinterpret_cast<const uint2*>(A.xr + static_cast<size_t>(P->row_tok[base + min(r + 8, n - 1)]) * KSH * 8) + c;
        hon[g] = g * 16 + 8 < n;
    }
    auto step = [&](int s, const u32 (&f0)[2], const u32 (&f1)[2], __half2 s2) {
        #pragma unroll
        for (int g = 0; g < G; ++g)
        {
            const uint2 lo = __ldg(xl[g] + s * 4);
            const u32 a0 = as_u32(__hmul2(as_h2(lo.x), s2)), a2 = as_u32(__hmul2(as_h2(lo.y), s2));
            u32 a1 = 0, a3 = 0;
            if (hon[g])
            {
                const uint2 hi = __ldg(xh[g] + s * 4);
                a1 = as_u32(__hmul2(as_h2(hi.x), s2)); a3 = as_u32(__hmul2(as_h2(hi.y), s2));
            }
            mma16816(acc[g][0], a0, a1, a2, a3, f0[0], f0[1]);
            mma16816(acc[g][1], a0, a1, a2, a3, f1[0], f1[1]);
        }
    };
    if constexpr (WIDE)
    {
        constexpr int NW = trw_nw<BITS>();
        const uint8_t* g0 = mw + static_cast<size_t>(ks0) * 8 * TB + j * 256 + lane * 8;
        u64 nw[3] = {0, 0, 0};
        #pragma unroll
        for (int q = 0; q < NW; ++q) nw[q] = ld_nc64(g0 + q * 2048);
        for (int s = 0; s < nks; s += 8)
        {
            const u64 w[3] = {nw[0], nw[1], nw[2]};
            if (s + 8 < nks)
            {
                #pragma unroll
                for (int q = 0; q < NW; ++q) nw[q] = ld_nc64(g0 + static_cast<size_t>(s + 8) * 8 * TB + q * 2048);
            }
            const __half2 s2 = __half2half2(sc[(ks0 + s) >> 3]);
            #pragma unroll
            for (int p = 0; p < 8; ++p)
            {
                const u32 ch = trw_chunk<BITS>(w, p);
                const u32 pc = __shfl_sync(0xffffffffu, ch, (lane + 31) & 31);
                u32 f0[2], f1[2];
                Tr3Lane<BITS, true>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
                step(ks0 + s + p, f0, f1, s2);
            }
        }
    }
    else
    {
        Tr3Lane<BITS, true> ld;
        ld.init(lane);
        const uint8_t* t0 = mw + (static_cast<size_t>(ks0) * 8 + j) * TB;
        u32 na[8];
        #pragma unroll
        for (int p = 0; p < 8; ++p) na[p] = ld.loadw(t0 + static_cast<size_t>(p) * 8 * TB, lane);
        for (int s = 0; s < nks; s += 8)
        {
            u32 wa[8];
            #pragma unroll
            for (int p = 0; p < 8; ++p) wa[p] = na[p];
            if (s + 8 < nks)
            {
                #pragma unroll
                for (int p = 0; p < 8; ++p) na[p] = ld.loadw(t0 + static_cast<size_t>(s + 8 + p) * 8 * TB, lane);
            }
            const __half2 s2 = __half2half2(sc[(ks0 + s) >> 3]);
            #pragma unroll
            for (int p = 0; p < 8; ++p)
            {
                u32 f0[2], f1[2];
                ld.decodew(wa[p], f0, f1);
                step(ks0 + s + p, f0, f1, s2);
            }
        }
    }
}

template <int BITS, int G>
__device__ __forceinline__ void fc1_pick_wide(const Args& A, const uint8_t* mw, int j, int ks0, int nks, int base, int n,
                                              const __half* sc, float (&acc)[2][2][4])
{
    if (trw_ok<BITS>() && A.wide1) fc1_warp<BITS, trw_ok<BITS>(), G>(A, mw, j, ks0, nks, base, n, sc, acc);
    else fc1_warp<BITS, false, G>(A, mw, j, ks0, nks, base, n, sc, acc);
}

// grid = (experts at most) x (EBW / R), block 256 = 8 warps = 2 matrices x R/16 tile columns x 64/R k parts
__global__ void __launch_bounds__(256) k_fc1(Args A)
{
    __shared__ float red[8][2][2][4][32];
    const int per = EBW / A.R, idx = blockIdx.x / per, ch = blockIdx.x % per;
    const Plan* P = A.plan;
    if (idx >= P->nex) return;
    const int e = P->ex_e[idx], n = P->ex_n[idx], base = P->ex_base[idx];
    const long long* tr = A.tab + 6 * e;
    const int col0 = ch * A.R, b = col0 >> 7;
    if (b >= static_cast<int>(tr[3])) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int TC = A.R / 16, KS = 4 / TC;
    const int mat = warp & 1, tc = (warp >> 1) % TC, hk = (warp >> 1) / TC;
    const int j = ((col0 & 127) >> 4) + tc;
    size_t off;
    const int bits = tr3_unit(A.kmix, e, tr, 2 * b + mat, off);
    const size_t row = static_cast<size_t>(tr[2]) + b;
    const __half* sc = A.v1b + (row * 2 + mat) * (HID / 128);
    const int nks = KSH / KS, ks0 = hk * nks;
    float acc[2][2][4] = {};
    const uint8_t* mw = A.fc1 + off;
    if (n > 16)
    {
        if (bits == 3) fc1_pick_wide<3, 2>(A, mw, j, ks0, nks, base, n, sc, acc);
        else if (bits == 2) fc1_pick_wide<2, 2>(A, mw, j, ks0, nks, base, n, sc, acc);
        else fc1_pick_wide<4, 2>(A, mw, j, ks0, nks, base, n, sc, acc);
    }
    else
    {
        if (bits == 3) fc1_pick_wide<3, 1>(A, mw, j, ks0, nks, base, n, sc, acc);
        else if (bits == 2) fc1_pick_wide<2, 1>(A, mw, j, ks0, nks, base, n, sc, acc);
        else fc1_pick_wide<4, 1>(A, mw, j, ks0, nks, base, n, sc, acc);
    }
    const int G = n > 16 ? 2 : 1;
    if (KS > 1)
    {
        for (int g = 0; g < G; ++g)
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf)
                #pragma unroll
                for (int i = 0; i < 4; ++i) red[warp][g][hf][i][lane] = acc[g][hf][i];
        __syncthreads();
        if (hk) return;
        for (int g = 0; g < G; ++g)
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf)
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float s = 0.f;
                    for (int q = 0; q < KS; ++q) s += red[mat + 2 * (tc + TC * q)][g][hf][i][lane];
                    acc[g][hf][i] = s;
                }
    }
    const int c = lane & 3, rl = lane >> 2;
    for (int g = 0; g < G; ++g)
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf)
        {
            const int col = b * 128 + j * 16 + hf * 8 + 2 * c, r = g * 16 + rl;
            __half* hp = A.h + (static_cast<size_t>(base + r) * 2 + mat) * EBW + col;
            if (r < n) *reinterpret_cast<u32*>(hp) = pack_h2(acc[g][hf][0], acc[g][hf][1]);
            if (r + 8 < n) *reinterpret_cast<u32*>(hp + 8 * 2 * EBW) = pack_h2(acc[g][hf][2], acc[g][hf][3]);
        }
}

// ------------------------------------------------------------------------------------------ fc2
// One warp: n-tile nt of every block of the expert. The words of all blocks are requested before the first decode.
template <int BITS, bool WIDE, int G>
__device__ __forceinline__ void fc2_ntile(const Args& A, const uint8_t* e0, int nb, int nt, const u32* act, const float* cb,
                                          float (&acc)[2][2][4])
{
    constexpr int TB = 32 * BITS, NW = WIDE ? trw_nw<BITS>() : 8;
    constexpr size_t BB = static_cast<size_t>(KSH) * 8 * TB;
    const int lane = threadIdx.x & 31;
    u64 wv[g6nr::EB][WIDE ? 3 : 4];
    Tr3Lane<BITS, true> ld;
    ld.init(lane);
    #pragma unroll
    for (int b = 0; b < g6nr::EB; ++b)
        if (b < nb)
        {
            const uint8_t* bb = e0 + b * BB;
            if constexpr (WIDE)
            {
                #pragma unroll
                for (int q = 0; q < NW; ++q) wv[b][q] = ld_nc64(bb + static_cast<size_t>(nt) * 8 * TB + lane * 8 + q * 256);
            }
            else
            {
                #pragma unroll
                for (int p = 0; p < 8; p += 2)
                    wv[b][p / 2] = ld.loadw(bb + (static_cast<size_t>(nt) * 8 + p) * TB, lane) |
                                   (static_cast<u64>(ld.loadw(bb + (static_cast<size_t>(nt) * 8 + p + 1) * TB, lane)) << 32);
            }
        }
    #pragma unroll
    for (int b = 0; b < g6nr::EB; ++b)
    {
        if (b >= nb) break;
        float ab[2][2][4] = {};
        #pragma unroll
        for (int p = 0; p < 8; ++p)
        {
            u32 f0[2], f1[2];
            if constexpr (WIDE)
            {
                const u64 w[3] = {wv[b][0], wv[b][1], NW > 2 ? wv[b][2] : 0ull};
                const u32 ch = trw_chunk<BITS>(w, p);
                const u32 pc = __shfl_sync(0xffffffffu, ch, (lane + 31) & 31);
                Tr3Lane<BITS, true>::decode_m((static_cast<u64>(pc) << (8 * BITS)) | ch, 7 * BITS, f0, f1);
            }
            else ld.decodew(static_cast<u32>(wv[b][p / 2] >> (32 * (p & 1))), f0, f1);
            #pragma unroll
            for (int g = 0; g < G; ++g)
            {
                const uint4 a = *reinterpret_cast<const uint4*>(act + ((static_cast<size_t>(g) * g6nr::EB + b) * 8 + p) * 128 + lane * 4);
                mma16816(ab[g][0], a.x, a.y, a.z, a.w, f0[0], f0[1]);
                mma16816(ab[g][1], a.x, a.y, a.z, a.w, f1[0], f1[1]);
            }
        }
        #pragma unroll
        for (int g = 0; g < G; ++g)
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf)
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[g][hf][i] += cb[b] * ab[g][hf][i];
    }
}

template <int BITS, int G>
__device__ __forceinline__ void fc2_body(const Args& A, const long long* tr, int ob0, int base, int n, u32* act, float* stage)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = lane & 3, rl = lane >> 2;
    const int nb = static_cast<int>(tr[3]);
    const size_t row0 = static_cast<size_t>(tr[2]);
    constexpr bool CW = trw_ok<BITS>();
    const bool wide = CW && A.wide2;
    const uint8_t* e0 = A.fc2 + tr[1];
    for (int ob = ob0; ob < ob0 + A.OB; ++ob)
    {
        float acc[2][2][4] = {};
        float cb[g6nr::EB];
        #pragma unroll
        for (int b = 0; b < g6nr::EB; ++b) cb[b] = b < nb ? __half2float(A.v3b[(row0 + b) * (HID / 128) + ob]) : 0.f;
        const int nt = ob * 8 + warp;
        if (wide) fc2_ntile<BITS, CW, G>(A, e0, nb, nt, act, cb, acc);
        else fc2_ntile<BITS, false, G>(A, e0, nb, nt, act, cb, acc);
        #pragma unroll
        for (int g = 0; g < G; ++g)
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf)
            {
                const int col = warp * 16 + hf * 8 + 2 * c, r = g * 16 + rl;
                stage[r * 128 + col] = acc[g][hf][0]; stage[r * 128 + col + 1] = acc[g][hf][1];
                stage[(r + 8) * 128 + col] = acc[g][hf][2]; stage[(r + 8) * 128 + col + 1] = acc[g][hf][3];
            }
        __syncthreads();
        for (int r = warp; r < n; r += 8)
        {
            const float4 s4 = *reinterpret_cast<const float4*>(stage + r * 128 + lane * 4);
            float v[4] = {s4.x, s4.y, s4.z, s4.w};
            had128(v, lane);
            const uint2 sr = *reinterpret_cast<const uint2*>(A.sg + HID + ob * 128 + lane * 4);
            const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
            v[0] *= s01.x; v[1] *= s01.y; v[2] *= s23.x; v[3] *= s23.y;
            const size_t yr = static_cast<size_t>(base + r);
            if (A.f16rows)
            {
                uint2 o;
                o.x = pack_h2(v[0], v[1]); o.y = pack_h2(v[2], v[3]);
                *reinterpret_cast<uint2*>(A.y16 + yr * HID + ob * 128 + lane * 4) = o;
            }
            else
            {
                float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                const float scl = am / 127.f, inv = am > 0.f ? 127.f / am : 0.f;
                u32 q = 0;
                #pragma unroll
                for (int i = 0; i < 4; ++i) q |= (static_cast<u32>(__float2int_rn(v[i] * inv)) & 0xffu) << (8 * i);
                *reinterpret_cast<u32*>(A.y8 + yr * HID + ob * 128 + lane * 4) = q;
                if (lane == 0) A.ysc[yr * (HID / 128) + ob] = scl;
            }
        }
        __syncthreads();
    }
}

// grid = (experts at most) x (48 / OB), block 256: warp j = n-tile j of each 128-column output block
__global__ void __launch_bounds__(256) k_fc2(Args A)
{
    extern __shared__ __align__(16) u32 smem[];
    const int per = (HID / 128) / A.OB, idx = blockIdx.x / per, og = blockIdx.x % per;
    const Plan* P = A.plan;
    if (idx >= P->nex) return;
    const int e = P->ex_e[idx], n = P->ex_n[idx], base = P->ex_base[idx];
    const long long* tr = A.tab + 6 * e;
    const int G = n > 16 ? 2 : 1, nb = static_cast<int>(tr[3]);
    u32* act = smem;                                                  // [G][EB][8 ks][32 lanes][4]
    float* stage = reinterpret_cast<float*>(smem + 2 * g6nr::EB * 8 * 128);   // [32 rows][128]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // prologue (K1's fc2 prologue per row and block): act = H128(f16(silu(g') * u') * suh_d), g' = f16(H128(g) * svh_g)
    for (int task = warp; task < G * 16 * nb; task += 8)
    {
        const int r = task / nb, b = task % nb;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if (r < n)
        {
            const __half* v2 = A.v2 + (static_cast<size_t>(tr[2]) + b) * 3 * 128 + lane * 4;
            float gu[2][4];
            #pragma unroll
            for (int mat = 0; mat < 2; ++mat)
            {
                const uint2 hr = *reinterpret_cast<const uint2*>(A.h + (static_cast<size_t>(base + r) * 2 + mat) * EBW + b * 128 + lane * 4);
                const float2 h01 = __half22float2(as_h2(hr.x)), h23 = __half22float2(as_h2(hr.y));
                float y[4] = {h01.x, h01.y, h23.x, h23.y};
                had128(y, lane);
                const uint2 sr = *reinterpret_cast<const uint2*>(v2 + mat * 128);
                const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
                gu[mat][0] = rnd_h(y[0] * s01.x); gu[mat][1] = rnd_h(y[1] * s01.y);
                gu[mat][2] = rnd_h(y[2] * s23.x); gu[mat][3] = rnd_h(y[3] * s23.y);
            }
            const uint2 sr = *reinterpret_cast<const uint2*>(v2 + 2 * 128);
            const float2 s01 = __half22float2(as_h2(sr.x)), s23 = __half22float2(as_h2(sr.y));
            v[0] = rnd_h(silu_f(gu[0][0]) * gu[1][0]) * s01.x;
            v[1] = rnd_h(silu_f(gu[0][1]) * gu[1][1]) * s01.y;
            v[2] = rnd_h(silu_f(gu[0][2]) * gu[1][2]) * s23.x;
            v[3] = rnd_h(silu_f(gu[0][3]) * gu[1][3]) * s23.y;
            had128(v, lane);
        }
        const int g = r >> 4, rr = r & 15, ks = lane >> 2, q0 = (lane & 3) * 4;
        const int reg = (q0 >= 8 ? 2 : 0) + (rr >= 8 ? 1 : 0), c0 = (q0 & 7) >> 1;
        u32* ap = act + ((static_cast<size_t>(g) * g6nr::EB + b) * 8 + ks) * 128;
        ap[((rr & 7) * 4 + c0) * 4 + reg] = pack_h2(v[0], v[1]);
        ap[((rr & 7) * 4 + c0 + 1) * 4 + reg] = pack_h2(v[2], v[3]);
    }
    __syncthreads();
    const int bits = static_cast<int>(tr[4]), ob0 = og * A.OB;
    if (G == 2)
    {
        if (bits == 3) fc2_body<3, 2>(A, tr, ob0, base, n, act, stage);
        else if (bits == 2) fc2_body<2, 2>(A, tr, ob0, base, n, act, stage);
        else fc2_body<4, 2>(A, tr, ob0, base, n, act, stage);
    }
    else
    {
        if (bits == 3) fc2_body<3, 1>(A, tr, ob0, base, n, act, stage);
        else if (bits == 2) fc2_body<2, 1>(A, tr, ob0, base, n, act, stage);
        else fc2_body<4, 1>(A, tr, ob0, base, n, act, stage);
    }
}
constexpr size_t FC2_SMEM = static_cast<size_t>(2) * g6nr::EB * 8 * 128 * 4 + 32 * 128 * 4;

// ------------------------------------------------------------------------------------------ combine
// grid ceil(M * 48 / 8), block 256: warp = (token m, 128-column block)
__global__ void __launch_bounds__(256) k_combine(Args A)
{
    const int lane = threadIdx.x & 31, task = blockIdx.x * 8 + (threadIdx.x >> 5), m = task / 48, ob = task % 48;
    if (m >= A.M) return;
    const Plan* P = A.plan;
    const int n0 = ob * 128 + lane * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < 9; ++k)
    {
        const size_t r = static_cast<size_t>(P->ridx[m * 9 + k]);
        const float w = k < 8 ? A.tok_w[m * 8 + k] : 1.f;
        if (A.f16rows)
        {
            const uint2 y = *reinterpret_cast<const uint2*>(A.y16 + r * HID + n0);
            const float2 y01 = __half22float2(as_h2(y.x)), y23 = __half22float2(as_h2(y.y));
            acc[0] += w * y01.x; acc[1] += w * y01.y; acc[2] += w * y23.x; acc[3] += w * y23.y;
        }
        else
        {
            const u32 q = *reinterpret_cast<const u32*>(A.y8 + r * HID + n0);
            const float s = w * A.ysc[r * (HID / 128) + ob];
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] += s * static_cast<float>(static_cast<int8_t>((q >> (8 * i)) & 0xffu));
        }
    }
    uint2 o;
    o.x = pack_o2(acc[0], acc[1], A.out_bf16);
    o.y = pack_o2(acc[2], acc[3], A.out_bf16);
    *reinterpret_cast<uint2*>(static_cast<__half*>(A.out) + static_cast<size_t>(m) * HID + n0) = o;
}

// once per process before the first launch
inline cudaError_t init() { return cudaFuncSetAttribute(k_fc2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(FC2_SMEM)); }

inline int max_experts(int M) { return (8 * M < 256 ? 8 * M : 256) + 1; }
// R: 16, 32 or 64 rows of the intermediate per fc1 item; OB: a divisor of 48. nullptr = launched.
inline const char* launch(const Args& A, cudaStream_t st)
{
    const int M = A.M, nx = max_experts(M);
    if (M < 1 || M > G6_MX) return "k2x: M outside 1..G6_MX";
    if (A.R != 16 && A.R != 32 && A.R != 64) return "k2x: R must be 16, 32 or 64";
    if (A.OB < 1 || (HID / 128) % A.OB) return "k2x: OB must divide 48";
    k_plan_rot<<<1 + (M * 48 + 7) / 8, 256, 0, st>>>(A);
    k_fc1<<<nx * (EBW / A.R), 256, 0, st>>>(A);
    k_fc2<<<nx * ((HID / 128) / A.OB), 256, FC2_SMEM, st>>>(A);
    k_combine<<<(M * 48 + 7) / 8, 256, 0, st>>>(A);
    return nullptr;
}
} // namespace k2x
