// Inkling-2-Small routed experts on sm_121a tensor cores, both stored formats. Our own code. The twin of
// ink2ref::Model::moe (engine/ref/ink2/ink2_cpu_ref.h) for one rank's intermediate slice:
//
//   x [M][H] BF16 -> quantize (the layer's activation format) -> fc1 (gate | up) -> softcapped SwiGLU, BF16
//     -> quantize each (expert, token) row of I values -> fc2 -> ypart = route weight * y -> sum in pick order, BF16
//
//   NVFP4 layers  records nv4g (576 B = 16 rows x 64 k), activations by the rule four over six with one outer scale
//                 per row; mma kind::mxf4nvf4 scale_vec::4X m16n8k64 (nv4::Frag in nvfp4.cuh)
//   MXFP8 layers  records of 528 B (16 rows x 32 k E4M3, one E8M0 scale per row), activations E4M3 with one E8M0
//                 scale per 32; mma kind::mxf8f6f4 scale_vec::1X m16n8k32, E4M3 x E4M3 (Mx8Frag below)
//
// The kernels are templates over the format (F = nv4::Frag or ink2::Mx8Frag) and over NT, the token tiles a warp
// carries. They walk the plan of ink2_router.cuh: a CTA = (slot, a few 16-row chunks); a warp reads its chunk's records
// of the slot's expert with ld.global.nc once per group of NT tiles and runs NT mma per record.
//   decode   M <= 8: NT = 1, every slot has one tile, ypart fp32
//   prefill  any M:  NT = 4, ypart fp32 or BF16 (MoeArgs::y_bf16)
// Both give the same sums for the same M (one mma per record and tile, the same order over k).
//
// Tensors: fc1 [NE][I / 16][H / KG][gate GR | up GR], fc2 [NE][H / 16][I / KG][GR], s1 / s2 [NE] fp32 (NVFP4 only).
// This header also compiles as plain C++ (Mx8Frag's maps and the exact MXFP8 quantizer), for the host tests.
#pragma once
#include "ink2_router.cuh"

namespace ink2
{
using nv4::bperm;
using nv4::rd16;
using nv4::rd32;
using nv4::rd64;

// ---------------------------------------------------------------------------------------------- exact scalars
// Softcapped SwiGLU, BF16 bits (ink2ref::Model::act): the sigmoid sees the raw gate.
INK2_HD u32 swiglu(float gate, float up)
{
    const float g_cap = 4.0f * tanhf(gate / 4.0f);
    const float sig = sigmoid(gate);
    const float u_cap = 25.0f * tanhf(up / 25.0f);
    return nv4::bf16_rne(g_cap * sig * u_cap);
}

// MXFP8 block scale (ink2ref::mx8::quantize): s = clamp(ceil(log2(fl32(amax / 448))) + 127, 0, 254), amax floored
// at FLT_MIN by the caller. The logarithm is read from the bits of the quotient, subnormal or not.
INK2_HD u32 mx8_scale(float amax)
{
    const u32 b = nv4::f2u(amax / 448.0f);
    const int ef = static_cast<int>((b >> 23) & 255u);
    const u32 mant = b & 0x7fffffu;
    int l;
    if (ef) l = ef - 127 + (mant != 0u);
    else
    {
#ifdef __CUDA_ARCH__
        const int top = 31 - __clz(static_cast<int>(mant));
#else
        const int top = 31 - __builtin_clz(mant);
#endif
        l = top - 149 + ((mant & (mant - 1u)) != 0u);
    }
    const int s = l + 127;
    return static_cast<u32>(s < 0 ? 0 : s > 254 ? 254 : s);
}
// code = e4m3(x * 2^(127 - s)); the factor is an exact fp32 (2^-127 is a subnormal), so one fp32 product is the
// reference's fp64 product rounded to fp32.
INK2_HD u32 mx8_code(float x, u32 s) { return nv4::e4m3_rne(x * nv4::u2f(s == 254u ? 0x00400000u : (254u - s) << 23)); }

// ---------------------------------------------------------------------------------------------- MXFP8 operand map
// mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
// Operand layout this file assumes (lane = 4 m + t):
//   a0 / a1  rows m / m + 8, physical k 4 t .. 4 t + 3 (one byte each);  a2 / a3 the same rows, k 16 + 4 t ..
//   b0 / b1  column m, physical k 4 t .. / 16 + 4 t ..;  d as in m16n8k16
//   scale A  byte 0 of lane 4 m's word scales row m, of lane 4 m + 1's word row m + 8;  scale B  lane 4 m, column m
// The record gives lane (r, c) of tile h (k 16 h ..) the 8 bytes (r, 2c), (r, 2c+1), (r+8, 2c), (r+8, 2c+1),
// (r, 2c+8), (r, 2c+9), (r+8, 2c+8), (r+8, 2c+9): two byte permutes split the rows, nothing crosses lanes. The k order
// inside the block of 32 (one scale, so any order shared by A and B is right): register half h, byte i ->
// k 16 h + 2 c + (i & 1) + 8 (i >> 1).
struct Mx8Frag
{
    static constexpr int GR = 528, KG = 32, SOFF = 512, XREC = 32;
    static constexpr bool OUTER = false;
    static constexpr u32 SEL_LO = 0x5410u;   // bytes 0, 1 of both words = row m
    static constexpr u32 SEL_HI = 0x7632u;   // bytes 2, 3 = row m + 8
    static constexpr int BID_A = 0, TID_A = 0, BID_B = 0, TID_B = 0;

    // Byte of an activation record that holds k (0 .. 31) of its block.
    static constexpr NV4_CX int xrec_byte(int k)
    {
        const int h = k >> 4, k16 = k & 15, c = (k16 & 7) >> 1, i = (k16 & 1) + 2 * (k16 >> 3);
        return c * 8 + h * 4 + i;
    }

    struct A
    {
        u32 a0, a1, a2, a3, s;
    };
    static INK2_HD A load(const uint8_t* rec, int lane)
    {
        const u64 v0 = rd64(rec + lane * 8), v1 = rd64(rec + 256 + lane * 8);
        const u32 l0 = static_cast<u32>(v0), h0 = static_cast<u32>(v0 >> 32), l1 = static_cast<u32>(v1), h1 = static_cast<u32>(v1 >> 32);
        A o;
        o.a0 = bperm(l0, h0, SEL_LO);
        o.a1 = bperm(l0, h0, SEL_HI);
        o.a2 = bperm(l1, h1, SEL_LO);
        o.a3 = bperm(l1, h1, SEL_HI);
        o.s = (rd16(rec + SOFF + (lane >> 2) * 2) >> (8 * (lane & 1))) & 0xffu;   // even lane: row r, odd lane: row r + 8
        return o;
    }
    static INK2_HD void loadB(const uint8_t* xr, int c, u32& b0, u32& b1) { nv4::Frag::loadB(xr, c, b0, b1); }
#ifdef __CUDACC__
    static __device__ __forceinline__ void mma(float (&d)[4], const A& a, u32 b0, u32 b1, u32 sb)
    {
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a.a0), "r"(a.a1), "r"(a.a2), "r"(a.a3), "r"(b0), "r"(b1),
                       "r"(a.s), "n"(BID_A), "n"(TID_A), "r"(sb), "n"(BID_B), "n"(TID_B));
    }
#endif
};

#ifdef __CUDACC__
// ---------------------------------------------------------------------------------------------- MXFP8 quantizer
// One CTA per row, a thread per block of 32. x [rows][K] 16-bit cells -> xq [rows][K / 32][32] (Mx8Frag::xrec_byte),
// xs [rows][K / 32] u32 = the E8M0 byte. n_live / live_mul as in nv4::k_nv4_quant.
__global__ void __launch_bounds__(nv4::QT) k_mx8_quant(const uint16_t* x, int K, int in_bf16, const int* n_live, int live_mul, uint8_t* xq, u32* xs)
{
    const int row = blockIdx.x;
    if (n_live && row >= *n_live * live_mul) return;
    const uint16_t* xr = x + static_cast<size_t>(row) * K;
    const int NG = K / 32;
    for (int b = threadIdx.x; b < NG; b += nv4::QT)
    {
        const uint4* p = reinterpret_cast<const uint4*>(xr + b * 32);
        u32 w[16];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const uint4 v = p[j];
            w[4 * j] = v.x; w[4 * j + 1] = v.y; w[4 * j + 2] = v.z; w[4 * j + 3] = v.w;
        }
        float amax = FLT_MIN;
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            amax = fmaxf(amax, fmaxf(fabsf(nv4::cell16(w[j] & 0xffffu, in_bf16)), fabsf(nv4::cell16(w[j] >> 16, in_bf16))));
        const u32 s = mx8_scale(amax);
        const size_t blk = static_cast<size_t>(row) * NG + b;
        uint8_t* o = xq + blk * Mx8Frag::XREC;
        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            o[Mx8Frag::xrec_byte(2 * j)] = static_cast<uint8_t>(mx8_code(nv4::cell16(w[j] & 0xffffu, in_bf16), s));
            o[Mx8Frag::xrec_byte(2 * j + 1)] = static_cast<uint8_t>(mx8_code(nv4::cell16(w[j] >> 16, in_bf16), s));
        }
        xs[blk] = s;
    }
}
inline void mx8_quant(const void* x, int rows, int K, bool in_bf16, const int* n_live, int live_mul, uint8_t* xq, u32* xs, cudaStream_t st)
{
    if (rows < 1 || K % 32) throw std::runtime_error("ink2::mx8_quant: rows < 1 or K % 32");
    k_mx8_quant<<<rows, nv4::QT, 0, st>>>(static_cast<const uint16_t*>(x), K, in_bf16 ? 1 : 0, n_live, live_mul, xq, xs);
}
// BF16 rows in the activation format of F.
template <class F>
inline void quant_rows(const uint16_t* x, int rows, int K, const int* n_live, int live_mul, uint8_t* xq, u32* xs, float* x_outer, cudaStream_t st)
{
    if constexpr (F::OUTER) nv4::quant(x, rows, K, true, nv4::ScaleRule::FourOverSixMae, nullptr, n_live, live_mul, xq, xs, x_outer, st);
    else mx8_quant(x, rows, K, true, n_live, live_mul, xq, xs, st);
}

// ---------------------------------------------------------------------------------------------- the experts
struct MoeArgs
{
    const uint8_t* fc1;
    const uint8_t* fc2;
    const float* s1;      // [NE], NVFP4 only
    const float* s2;
    int H, I, M, TK;
    uint8_t* xq;          // [M][H / KG][32]        x quantized
    u32* xs;              // [M][H / KG]
    float* x_outer;       // [M]                    NVFP4 only
    Plan plan;
    uint16_t* act;        // [tile][8][I] BF16      fc1 output
    uint8_t* aq;          // [tile * 8][I / KG][32]
    u32* as;              // [tile * 8][I / KG]
    float* a_outer;       // [tile * 8]             NVFP4 only
    void* ypart;          // [tile][8][H] fp32, or BF16 with y_bf16
    int y_bf16;
    uint16_t* out;        // [M][H] BF16
};

constexpr int CW1 = 4, CW2 = 4;   // warps (16-row chunks) per CTA of fc1 / fc2

// fc1 + SwiGLU. grid (max slots, ceil(I / 16 / CW1)).
template <class F, int NT>
__global__ void __launch_bounds__(CW1 * 32) k_fc1(MoeArgs a)
{
    const int slot = blockIdx.x;
    if (slot >= a.plan.counts[0]) return;
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int ch = blockIdx.y * CW1 + (threadIdx.x >> 5);
    if (ch >= a.I / 16) return;
    const int e = a.plan.slot_expert[slot], tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const int NG = a.H / F::KG;
    const uint8_t* rec0 = a.fc1 + (static_cast<size_t>(e) * (a.I / 16) + ch) * NG * 2 * F::GR;
    for (int tg = 0; tg < nt; tg += NT)
    {
        float ag[NT][4], au[NT][4];
        u32 xo[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            ag[t][0] = ag[t][1] = ag[t][2] = ag[t][3] = 0.0f;
            au[t][0] = au[t][1] = au[t][2] = au[t][3] = 0.0f;
            xo[t] = static_cast<u32>(a.plan.tile_tok[(tile0 + min(tg + t, nt - 1)) * 8 + r]) * NG;
        }
        for (int g = 0; g < NG; ++g)
        {
            const uint8_t* rec = rec0 + static_cast<size_t>(g) * 2 * F::GR;
            const typename F::A G = F::load(rec, lane), U = F::load(rec + F::GR, lane);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (tg + t < nt)
                {
                    u32 b0, b1;
                    F::loadB(a.xq + static_cast<size_t>(xo[t] + g) * F::XREC, c, b0, b1);
                    const u32 sb = a.xs[xo[t] + g];
                    F::mma(ag[t], G, b0, b1, sb);
                    F::mma(au[t], U, b0, b1, sb);
                }
        }
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (tg + t < nt)
            {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    const int row = (tile0 + tg + t) * 8 + 2 * c + (j & 1), i = ch * 16 + r + 8 * (j >> 1);
                    float sc = 1.0f;
                    if constexpr (F::OUTER) sc = a.s1[e] * a.x_outer[a.plan.tile_tok[row]];
                    a.act[static_cast<size_t>(row) * a.I + i] = static_cast<uint16_t>(swiglu(ag[t][j] * sc, au[t][j] * sc));
                }
            }
    }
}

// fc2 and the route weight. grid (max slots, ceil(H / 16 / CW2)).
template <class F, int NT>
__global__ void __launch_bounds__(CW2 * 32) k_fc2(MoeArgs a)
{
    const int slot = blockIdx.x;
    if (slot >= a.plan.counts[0]) return;
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int ot = blockIdx.y * CW2 + (threadIdx.x >> 5);
    if (ot >= a.H / 16) return;
    const int e = a.plan.slot_expert[slot], tile0 = a.plan.slot_tile0[slot], nt = a.plan.slot_nt[slot];
    const int NG = a.I / F::KG;
    const uint8_t* rec0 = a.fc2 + (static_cast<size_t>(e) * (a.H / 16) + ot) * NG * F::GR;
    for (int tg = 0; tg < nt; tg += NT)
    {
        float acc[NT][4];
        u32 xo[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
            xo[t] = static_cast<u32>((tile0 + min(tg + t, nt - 1)) * 8 + r) * NG;
        }
        for (int g = 0; g < NG; ++g)
        {
            const typename F::A A = F::load(rec0 + static_cast<size_t>(g) * F::GR, lane);
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                if (tg + t < nt)
                {
                    u32 b0, b1;
                    F::loadB(a.aq + static_cast<size_t>(xo[t] + g) * F::XREC, c, b0, b1);
                    F::mma(acc[t], A, b0, b1, a.as[xo[t] + g]);
                }
        }
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (tg + t < nt)
            {
                const int tile = tile0 + tg + t, mask = a.plan.tile_mask[tile];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    const int col = 2 * c + (j & 1), row = tile * 8 + col, n = ot * 16 + r + 8 * (j >> 1);
                    if (!((mask >> col) & 1)) continue;
                    float y = acc[t][j];
                    if constexpr (F::OUTER) y *= a.s2[e] * a.a_outer[row];
                    const float v = nv4::mul_rn(a.plan.tile_w[row], y);
                    const size_t o = static_cast<size_t>(row) * a.H + n;
                    if (a.y_bf16) static_cast<uint16_t*>(a.ypart)[o] = static_cast<uint16_t>(nv4::bf16_rne(v));
                    else static_cast<float*>(a.ypart)[o] = v;
                }
            }
    }
}

// out[m] = BF16(sum over the picks, in pick order, of ypart). A thread per output cell.
__global__ void __launch_bounds__(256) k_combine(MoeArgs a)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(a.M) * a.H) return;
    const int m = static_cast<int>(i / a.H), n = static_cast<int>(i % a.H);
    float s = 0.0f;
    for (int k = 0; k < a.TK; ++k)
    {
        const size_t o = static_cast<size_t>(a.plan.tok_slot[m * a.TK + k] * 8 + a.plan.tok_col[m * a.TK + k]) * a.H + n;
        s += a.y_bf16 ? nv4::u2f(static_cast<u32>(static_cast<const uint16_t*>(a.ypart)[o]) << 16) : static_cast<const float*>(a.ypart)[o];
    }
    a.out[i] = static_cast<uint16_t>(nv4::bf16_rne(s));
}

// x [M][H] BF16 and a built plan -> a.out. NE sizes the grids only.
template <class F, int NT>
inline void moe_run(const MoeArgs& a, const uint16_t* x, int NE, cudaStream_t st)
{
    if (a.M < 1 || a.H % 64 || a.I % 64 || a.TK < 1 || a.TK > MAX_TK) throw std::runtime_error("ink2 moe: M < 1, H % 64, I % 64 or TK > 8");
    const int ms = plan_max_slots(a.M, a.TK, NE), mt = plan_max_tiles(a.M, a.TK, NE);
    quant_rows<F>(x, a.M, a.H, nullptr, 0, a.xq, a.xs, a.x_outer, st);
    k_fc1<F, NT><<<dim3(ms, (a.I / 16 + CW1 - 1) / CW1), CW1 * 32, 0, st>>>(a);
    quant_rows<F>(a.act, mt * 8, a.I, a.plan.counts + 1, 8, a.aq, a.as, a.a_outer, st);
    k_fc2<F, NT><<<dim3(ms, (a.H / 16 + CW2 - 1) / CW2), CW2 * 32, 0, st>>>(a);
    k_combine<<<static_cast<unsigned>((static_cast<size_t>(a.M) * a.H + 255) / 256), 256, 0, st>>>(a);
}
// Decode: M <= 8, one tile per slot. ypart must be fp32 or BF16 as a.y_bf16 says.
template <class F>
inline void moe_decode(const MoeArgs& a, const uint16_t* x, int NE, cudaStream_t st)
{
    if (a.M > 8) throw std::runtime_error("ink2::moe_decode: M > 8");
    moe_run<F, 1>(a, x, NE, st);
}
// Read-once prefill: any M, an expert's records are read once per 4 of its tiles (32 tokens).
template <class F>
inline void moe_prefill(const MoeArgs& a, const uint16_t* x, int NE, cudaStream_t st)
{
    moe_run<F, 4>(a, x, NE, st);
}

// ---------------------------------------------------------------------------------------------- buffers
// The router's and the experts' work buffers for up to M tokens of one shape, one cudaMalloc each.
struct Work
{
    int M = 0, H = 0, I = 0, NE = 0, TK = 0, y_bf16 = 0;
    float* score = nullptr;
    int* idx = nullptr;
    float* wt = nullptr;
    Plan plan{};
    uint8_t* xq = nullptr;
    u32* xs = nullptr;
    float* x_outer = nullptr;
    uint16_t* act = nullptr;
    uint8_t* aq = nullptr;
    u32* as = nullptr;
    float* a_outer = nullptr;
    void* ypart = nullptr;
    std::vector<void*> owned;

    template <class T>
    T* get(size_t n)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) throw std::runtime_error("ink2::Work: cudaMalloc");
        owned.push_back(p);
        return static_cast<T*>(p);
    }
    // KG = the smaller k group of the two formats (32), so one Work serves both.
    void alloc(int M_, int H_, int I_, int NE_, int TK_, bool y_bf16_)
    {
        M = M_; H = H_; I = I_; NE = NE_; TK = TK_; y_bf16 = y_bf16_ ? 1 : 0;
        const size_t ms = static_cast<size_t>(plan_max_slots(M, TK, NE)), mt = static_cast<size_t>(plan_max_tiles(M, TK, NE)), m = static_cast<size_t>(M);
        score = get<float>(m * NE);
        idx = get<int>(m * TK);
        wt = get<float>(m * TK);
        plan.slot_expert = get<int>(ms);
        plan.slot_tile0 = get<int>(ms);
        plan.slot_nt = get<int>(ms);
        plan.tile_expert = get<int>(mt);
        plan.tile_mask = get<int>(mt);
        plan.tile_w = get<float>(mt * 8);
        plan.tile_tok = get<int>(mt * 8);
        plan.tok_slot = get<int>(m * TK);
        plan.tok_col = get<int>(m * TK);
        plan.counts = get<int>(2);
        xq = get<uint8_t>(m * (H / 32) * 32);
        xs = get<u32>(m * (H / 32));
        x_outer = get<float>(m);
        act = get<uint16_t>(mt * 8 * I);
        aq = get<uint8_t>(mt * 8 * (I / 32) * 32);
        as = get<u32>(mt * 8 * (I / 32));
        a_outer = get<float>(mt * 8);
        ypart = get<uint8_t>(mt * 8 * H * (y_bf16 ? 2 : 4));
    }
    void release()
    {
        for (void* p : owned) cudaFree(p);
        owned.clear();
    }
    MoeArgs args(const uint8_t* fc1, const uint8_t* fc2, const float* s1, const float* s2, int m, uint16_t* out) const
    {
        if (m > M) throw std::runtime_error("ink2::Work: more tokens than allocated");
        return MoeArgs{fc1, fc2, s1, s2, H, I, m, TK, xq, xs, x_outer, plan, act, aq, as, a_outer, ypart, y_bf16, out};
    }
};
#endif   // __CUDACC__
}   // namespace ink2
