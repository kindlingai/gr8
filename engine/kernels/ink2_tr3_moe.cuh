// Inkling-2-Mini / Inkling-2 exlr8-v2 trellis routed experts (pmcgx / mul1 / mcg, E4M3-rounded, K=2/3/4) on sm_121a
// tensor cores, with Inkling-2 softcapped SwiGLU (4*tanh(g/4) * sigmoid(g) * 25*tanh(u/25)).
//
// Self-contained: parameterized on runtime (H, eb, NE, TK) with zero dependency on GLM compile-time constants
// (common.cuh / k1args.cuh). Uses ink2::Plan (ink2_router.cuh) directly and writes the same ypart [tile][8][H]
// layout as ink2_moe.cuh so k_reduce (TP=1) and k_ink2rt_picks + sum (TP>1) work identically.
//
// Expects standard v2plan::v2_layer tensors (no shared expert -> bs=false):
//   tr3.fc1    [fc1_bytes] u8    [nblk][2][H/16][8][32*K]
//   tr3.fc2    [fc2_bytes] u8    [nblk][H/16][8][32*K]  (already Item128-transposed to gr8 order)
//   tr3.v1     [nblk][2][H] f16  signed suh for gate (0) and up (1)
//   tr3.v2     [nblk][3][128] f16  svh gate (0), svh up (1), suh down (2)
//   tr3.v3     [nblk][H] f16     signed svh for down
//   tr3.table  [NE][6] i64       [fc1_off, fc2_off, row0, nblk, k_down, first_half]
//   tr3.kmix   [NE] i64          0 when uniform per expert, else 4 bits per fc1 unit (2*b + mat)
#pragma once
#include "ink2_moe.cuh"
#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace ink2
{
namespace tr3
{
enum { CB_MCG = 0, CB_MUL1 = 1, CB_PMCGX = 2 };

#ifdef __CUDACC__
static constexpr float SQ128 = 0.08838834764831844f;   // 1 / sqrt(128)

__device__ __forceinline__ __half2 as_h2(u32 v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ u32 as_u32(__half2 h) { return *reinterpret_cast<u32*>(&h); }
__device__ __forceinline__ float rnd_h(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ void mma16816_f16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void had128(float (&v)[4], int lane)
{
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float p = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? (p - v[e]) : (v[e] + p);
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] *= SQ128;
}

__device__ __forceinline__ u32 round_e4m3(u32 r)
{
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(r));
    return r;
}

__device__ __forceinline__ u32 cb_mcg2(u32 w0, u32 w1)
{
    u32 x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    const u32 lo = __byte_perm(x0, x1, 0x5410);
    const u32 hi = __byte_perm(x0, x1, 0x7632);
    return round_e4m3(as_u32(__hadd2(as_h2(lo), as_h2(hi))));
}

__device__ __forceinline__ u32 cb_mul1x2(u32 w0, u32 w1)
{
    const u32 s0 = __dp4a(w0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const u32 s1 = __dp4a(w1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half2 v = __hfma2(as_h2(__byte_perm(s0, s1, 0x5410)), as_h2(0x1eee1eeeu), as_h2(0xc931c931u));
    return round_e4m3(as_u32(v));
}

__device__ __forceinline__ u32 cb_pmcgx(u32 w)
{
    u32 x0 = w * 0xCBAC1FEDu, x1 = w * 0xBCA14C51u;
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x0) : "r"(x0), "r"(0x8fff8fffu));
    asm("lop3.b32 %0, %1, %2, 0x3b603b60, 0x6a;" : "=r"(x1) : "r"(x1), "r"(0x8fff8fffu));
    return round_e4m3(as_u32(__hadd2(as_h2(x0), as_h2(__byte_perm(x1, 0, 0x1032)))));
}

template <int BITS, int CB>
struct LaneDec
{
    int i0, i1, sh;
    __device__ __forceinline__ void init(int lane)
    {
        const int b0 = (lane * 8 + 1) * BITS - 16 + 256 * BITS;
        i0 = (b0 / 32) % (8 * BITS);
        i1 = (i0 + 1) % (8 * BITS);
        sh = 48 - (b0 % 32);
    }
    __device__ __forceinline__ u32 loadw(const uint8_t* tile, int lane) const
    {
        if constexpr (BITS == 4) return nv4::rd32(tile + lane * 4);
        else return nv4::rd32(tile + (lane % (8 * BITS)) * 4);
    }
    __device__ __forceinline__ void decodew(u32 raw, u32 (&f0)[2], u32 (&f1)[2]) const
    {
        u64 m;
        int s;
        if constexpr (BITS == 4)
        {
            const u32 a = __shfl_sync(0xffffffffu, raw, i0);
            m = (static_cast<u64>(a) << 32) | raw;
            s = 28;
        }
        else
        {
            const u32 a = __shfl_sync(0xffffffffu, raw, i0);
            const u32 b = __shfl_sync(0xffffffffu, raw, i1);
            m = (static_cast<u64>(a) << 32) | b;
            s = sh;
        }
        if constexpr (CB == CB_PMCGX)
        {
            u32 w[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) w[p] = static_cast<u32>(m >> (s - BITS * (2 * p + 1))) & 0xfffffu;
            f0[0] = cb_pmcgx(w[0]);
            f0[1] = cb_pmcgx(w[1]);
            f1[0] = cb_pmcgx(w[2]);
            f1[1] = cb_pmcgx(w[3]);
        }
        else
        {
            u32 w[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = static_cast<u32>(m >> (s - BITS * j)) & 0xffffu;
            if constexpr (CB == CB_MUL1)
            {
                f0[0] = cb_mul1x2(w[0], w[1]);
                f0[1] = cb_mul1x2(w[2], w[3]);
                f1[0] = cb_mul1x2(w[4], w[5]);
                f1[1] = cb_mul1x2(w[6], w[7]);
            }
            else
            {
                f0[0] = cb_mcg2(w[0], w[1]);
                f0[1] = cb_mcg2(w[2], w[3]);
                f1[0] = cb_mcg2(w[4], w[5]);
                f1[1] = cb_mcg2(w[6], w[7]);
            }
        }
    }
};

__device__ __forceinline__ int fc1_unit(const uint8_t* fc1, const long long* kmix, int e, const long long* tr, int u, int H, const uint8_t*& mw)
{
    const size_t mb = static_cast<size_t>(H) * 16;   // (H/16) * 8 * 32 bytes per bit of width
    const long long km = kmix ? kmix[e] : 0;
    if (!km)
    {
        const int k = static_cast<int>(tr[4]);
        mw = fc1 + static_cast<size_t>(tr[0]) + static_cast<size_t>(u) * static_cast<size_t>(k) * mb;
        return k;
    }
    size_t off = static_cast<size_t>(tr[0]);
    for (int i = 0; i < u; ++i) off += static_cast<size_t>((km >> (4 * i)) & 15) * mb;
    mw = fc1 + off;
    return static_cast<int>((km >> (4 * u)) & 15);
}

__device__ __forceinline__ const uint8_t* fc2_block(const uint8_t* fc2, const long long* tr, int b, int bits, int H)
{
    const size_t mb = static_cast<size_t>(H) * 16;
    return fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * static_cast<size_t>(bits) * mb;
}

template <int BITS, int CB>
__device__ __forceinline__ void fc1_hb_mma(const uint8_t* mw, int hb, int warp, int lane,
                                           const __half (&x_rot)[8][136], int mask, float (&c)[4])
{
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = ((mask >> row0) & 1) != 0;
    const size_t tb = static_cast<size_t>(32 * BITS);
    LaneDec<BITS, CB> tl;
    tl.init(lane);
    const int ks0 = hb * 8;
    #pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        const uint8_t* tile = mw + (static_cast<size_t>(ks0 + p) * 8 + warp) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const u32 b0 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col) : 0u;
        const u32 b1 = act ? *reinterpret_cast<const u32*>(x_rot[row0] + p * 16 + 2 * col + 8) : 0u;
        mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

template <int BITS, int CB>
__device__ __forceinline__ void fc2_ntile_mma(const uint8_t* mw, int n_tile, int lane,
                                              const __half (&x2)[8][136], int mask, float (&c)[4])
{
    const int row0 = lane >> 2, col = lane & 3;
    const bool act = ((mask >> row0) & 1) != 0;
    const size_t tb = static_cast<size_t>(32 * BITS);
    LaneDec<BITS, CB> tl;
    tl.init(lane);
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks)
    {
        const uint8_t* tile = mw + (static_cast<size_t>(n_tile) * 8 + ks) * tb;
        const u32 raw = tl.loadw(tile, lane);
        u32 f0[2], f1[2];
        tl.decodew(raw, f0, f1);
        const u32 b0 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col) : 0u;
        const u32 b1 = act ? *reinterpret_cast<const u32*>(x2[row0] + ks * 16 + 2 * col + 8) : 0u;
        mma16816_f16(c, f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

struct Tr3Args
{
    const uint8_t* fc1;
    const uint8_t* fc2;
    const __half* v1;         // [nblk][2][H]
    const __half* v2;         // [nblk][3][128]
    const __half* v3;         // [nblk][H]
    const long long* table;   // [NE][6]
    const long long* kmix;    // [NE] or nullptr
    const uint16_t* x;        // [M][H] BF16
    Plan plan;
    __half* a_rot;            // [max_tiles * 8][eb][128] f16
    void* ypart;              // [max_tiles * 8][H] fp32 (or BF16 if y_bf16)
    uint16_t* out;            // [M][H] BF16
    int H, eb, M, TK, y_bf16;
};

// grid: (max_slots, eb), block: 256 (8 warps).
// Computes fc1 (gate + up) -> H128 * svh -> Inkling-2 softcapped SwiGLU -> * suh_down -> H128 -> a_rot [row][eb][128].
template <int CB>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc1(Tr3Args a)
{
    const int slot = blockIdx.x, b = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    if (b >= static_cast<int>(tr[3])) return;
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const size_t brow = static_cast<size_t>(tr[2]) + b;
    const int HB = a.H / 128;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;

    const uint8_t *mw0 = nullptr, *mw1 = nullptr;
    const int bits0 = fc1_unit(a.fc1, a.kmix, e, tr, 2 * b + 0, a.H, mw0);
    const int bits1 = fc1_unit(a.fc1, a.kmix, e, tr, 2 * b + 1, a.H, mw1);

    __shared__ __half x_rot[8][136];
    __shared__ float y_gate[8][128];
    __shared__ float y_up[8][128];

    for (int tg = 0; tg < nt; ++tg)
    {
        const int tile = tile0 + tg;
        const int mask = a.plan.tile_mask[tile];
        const bool w_act = ((mask >> warp) & 1) != 0;
        const int w_tok = w_act ? a.plan.tile_tok[tile * 8 + warp] : 0;

        for (int mat = 0; mat < 2; ++mat)
        {
            const uint8_t* mw = mat == 0 ? mw0 : mw1;
            const int bits = mat == 0 ? bits0 : bits1;
            const __half* suh = a.v1 + (brow * 2 + mat) * a.H;
            float c[4] = {0.f, 0.f, 0.f, 0.f};

            for (int hb = 0; hb < HB; ++hb)
            {
                if (w_act)
                {
                    const uint16_t* xp = a.x + static_cast<size_t>(w_tok) * a.H + hb * 128 + lane * 4;
                    const __half* sp = suh + hb * 128 + lane * 4;
                    float v[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) v[i] = bf(xp[i]) * __half2float(sp[i]);
                    had128(v, lane);
                    __half* dst = x_rot[warp] + lane * 4;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(v[i]);
                }
                __syncthreads();
                if (bits == 3) fc1_hb_mma<3, CB>(mw, hb, warp, lane, x_rot, mask, c);
                else if (bits == 4) fc1_hb_mma<4, CB>(mw, hb, warp, lane, x_rot, mask, c);
                else fc1_hb_mma<2, CB>(mw, hb, warp, lane, x_rot, mask, c);
                __syncthreads();
            }

            float (*ydst)[128] = mat == 0 ? y_gate : y_up;
            const int t0 = 2 * col, t1 = 2 * col + 1;
            if ((mask >> t0) & 1)
            {
                ydst[t0][warp * 16 + row0] = rnd_h(c[0]);
                ydst[t0][warp * 16 + row0 + 8] = rnd_h(c[2]);
            }
            if ((mask >> t1) & 1)
            {
                ydst[t1][warp * 16 + row0] = rnd_h(c[1]);
                ydst[t1][warp * 16 + row0 + 8] = rnd_h(c[3]);
            }
            __syncthreads();
        }

        if (w_act)
        {
            float vg[4], vu[4];
            const __half* svh_g = a.v2 + (brow * 3 + 0) * 128 + lane * 4;
            const __half* svh_u = a.v2 + (brow * 3 + 1) * 128 + lane * 4;
            const __half* suh_d = a.v2 + (brow * 3 + 2) * 128 + lane * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                vg[i] = y_gate[warp][lane * 4 + i];
                vu[i] = y_up[warp][lane * 4 + i];
            }
            had128(vg, lane);
            had128(vu, lane);
            float va[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float g = rnd_h(vg[i] * __half2float(svh_g[i]));
                const float u = rnd_h(vu[i] * __half2float(svh_u[i]));
                const float act = bf(static_cast<uint16_t>(swiglu(g, u)));
                va[i] = act * __half2float(suh_d[i]);
            }
            had128(va, lane);
            __half* dst = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dst[i] = __float2half_rn(va[i]);
        }
        __syncthreads();
    }
}

// grid: (max_slots, HB), block: 256 (8 warps).
// Accumulates down projection over b = 0..nb-1, applies H128 * v3[brow] * route_weight, writes ypart [tile*8+col][H].
template <int CB>
__global__ void __launch_bounds__(256) k_ink2_tr3_fc2(Tr3Args a)
{
    const int slot = blockIdx.x, hb = blockIdx.y;
    if (slot >= a.plan.counts[0]) return;
    const int e = a.plan.slot_expert[slot];
    const long long* tr = a.table + static_cast<size_t>(e) * 6;
    const int nb = static_cast<int>(tr[3]);
    const int bits = static_cast<int>(tr[4]);
    const int tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = lane >> 2, col = lane & 3;
    const int n_tile = hb * 8 + warp;

    __shared__ __half x2_sm[8][136];
    __shared__ float y_raw[8][128];

    for (int tg = 0; tg < nt; ++tg)
    {
        const int tile = tile0 + tg;
        const int mask = a.plan.tile_mask[tile];
        const bool w_act = ((mask >> warp) & 1) != 0;
        const float rw = w_act ? a.plan.tile_w[tile * 8 + warp] : 0.f;
        float sum_v[4] = {0.f, 0.f, 0.f, 0.f};

        for (int b = 0; b < nb; ++b)
        {
            if (w_act)
            {
                const __half* src = a.a_rot + (static_cast<size_t>(tile * 8 + warp) * a.eb + b) * 128 + lane * 4;
                *reinterpret_cast<uint2*>(x2_sm[warp] + lane * 4) = *reinterpret_cast<const uint2*>(src);
            }
            __syncthreads();

            const uint8_t* mw = fc2_block(a.fc2, tr, b, bits, a.H);
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            if (bits == 3) fc2_ntile_mma<3, CB>(mw, n_tile, lane, x2_sm, mask, c);
            else if (bits == 4) fc2_ntile_mma<4, CB>(mw, n_tile, lane, x2_sm, mask, c);
            else fc2_ntile_mma<2, CB>(mw, n_tile, lane, x2_sm, mask, c);

            const int t0 = 2 * col, t1 = 2 * col + 1;
            if ((mask >> t0) & 1)
            {
                y_raw[t0][warp * 16 + row0] = rnd_h(c[0]);
                y_raw[t0][warp * 16 + row0 + 8] = rnd_h(c[2]);
            }
            if ((mask >> t1) & 1)
            {
                y_raw[t1][warp * 16 + row0] = rnd_h(c[1]);
                y_raw[t1][warp * 16 + row0 + 8] = rnd_h(c[3]);
            }
            __syncthreads();

            if (w_act)
            {
                float v[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = y_raw[warp][lane * 4 + i];
                had128(v, lane);
                const size_t brow = static_cast<size_t>(tr[2]) + b;
                const __half* svh = a.v3 + brow * a.H + hb * 128 + lane * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) sum_v[i] += rw * (v[i] * __half2float(svh[i]));
            }
            __syncthreads();
        }

        if (w_act)
        {
            const size_t o = static_cast<size_t>(tile * 8 + warp) * a.H + hb * 128 + lane * 4;
            if (a.y_bf16)
            {
                uint16_t* dst = static_cast<uint16_t*>(a.ypart) + o;
                #pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = static_cast<uint16_t>(nv4::bf16_rne(sum_v[i]));
            }
            else
            {
                *reinterpret_cast<float4*>(static_cast<float*>(a.ypart) + o) = make_float4(sum_v[0], sum_v[1], sum_v[2], sum_v[3]);
            }
        }
    }
}

template <int CB>
inline void launch_tr3(const Tr3Args& a, int NE, cudaStream_t st)
{
    const int slots = plan_max_slots(a.M, a.TK, NE);
    const int HB = a.H / 128;
    k_ink2_tr3_fc1<CB><<<dim3(slots, a.eb), 256, 0, st>>>(a);
    k_ink2_tr3_fc2<CB><<<dim3(slots, HB), 256, 0, st>>>(a);
    if (a.out != nullptr)
    {
        MoeArgs ma{};
        ma.H = a.H;
        ma.M = a.M;
        ma.TK = a.TK;
        ma.plan = a.plan;
        ma.ypart = a.ypart;
        ma.y_bf16 = a.y_bf16;
        ma.out = a.out;
        k_combine<<<static_cast<unsigned>((static_cast<size_t>(a.M) * a.H + 255) / 256), 256, 0, st>>>(ma);
    }
}

inline void moe_tr3(const Tr3Args& a, int cb, int NE, cudaStream_t st)
{
    if (a.H % 128 != 0 || a.eb < 1 || a.M < 1) throw std::runtime_error("ink2::tr3::moe_tr3: H, eb or M");
    if (cb == CB_PMCGX) launch_tr3<CB_PMCGX>(a, NE, st);
    else if (cb == CB_MUL1) launch_tr3<CB_MUL1>(a, NE, st);
    else launch_tr3<CB_MCG>(a, NE, st);
}
#endif   // __CUDACC__
}   // namespace tr3
}   // namespace ink2
