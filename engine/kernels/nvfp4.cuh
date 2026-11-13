// NVFP4 on sm_121a: the reusable kernel layer over the record "nv4g" (nvfp4_pack.h). Our own code.
// Nothing here knows a model: every kernel takes its shape (N output rows, K inputs; N % 16 == 0, K % 64 == 0).
//
//   y[m][n] = sum_k W[n][k] x[m][k],   W = e2m1 * (w_outer * e4m3 per 16 k),   x = e2m1 * (x_outer[m] * e4m3 per 16 k)
//
// Parts
//   exact scalar functions   e2m1 / e4m3 conversion and the block quantizer of both scale rules. They are host and
//                            device functions with one body, in fp32 with a fixed order, and give the bytes of
//                            nv4pack::quantize (tests/ink2_moe_host_test.cpp compares them on the CPU).
//   nv4::Frag                the ONE place that knows how a record and an activation row map onto the operands of
//                            mma.sync kind::mxf4nvf4 scale_vec::4X m16n8k64 (SASS OMMA.SF.16864). A wrong guess about
//                            the hardware is a fix here and nowhere else; tests/nvfp4_mma_probe.cu checks it on a GPU.
//   nv4::k_nv4_quant         BF16 / FP16 rows -> activation records (rule Static6 or FourOverSixMae, or a fixed outer)
//   nv4::dense_gemv / _mt    dense y = x W^T over nv4g records, output FP16 / BF16 / FP32
//
// Activation storage ("XREC"), written by the quantizer and read as the B operand:
//   xq [row][K / 64][32]  code bytes; quad lane c reads the 8 bytes at c * 8 (Frag::xrec_byte is the map from k)
//   xs [row][K / 64]      u32 = the 4 E4M3 block scales of the group, block t in byte t
//   x_outer [row]         fp32
// Block membership is the true contiguous one (k 16 b .. 16 b + 15), as in nv4pack; only the place of a nibble inside
// its group follows the operand order.
//
// This header also compiles as plain C++ (the scalar functions and Frag's maps), for the host tests.
#pragma once
#include "nvfp4_pack.h"
#include <cfloat>
#include <climits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define NV4_HD __host__ __device__ __forceinline__
#define NV4_CX __host__ __device__
#else
#define NV4_HD inline
#define NV4_CX
#endif

namespace nv4
{
typedef uint32_t u32;
typedef uint64_t u64;
using nv4pack::ScaleRule;

// ---------------------------------------------------------------------------------------------- bits and loads
NV4_HD u32 f2u(float f)
{
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    u32 u;
    memcpy(&u, &f, 4);
    return u;
#endif
}
NV4_HD float u2f(u32 u)
{
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    memcpy(&f, &u, 4);
    return f;
#endif
}
// a * b rounded once. On the device a plain product that feeds a sum is contracted into an fma; the host is built with
// -ffp-contract=off.
NV4_HD float mul_rn(float a, float b)
{
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
// Weight loads: read once, kept out of L1 on the device.
NV4_HD u32 rd32(const void* p)
{
#ifdef __CUDA_ARCH__
    u32 v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
#else
    u32 v;
    memcpy(&v, p, 4);
    return v;
#endif
}
NV4_HD u64 rd64(const void* p)
{
#ifdef __CUDA_ARCH__
    u64 v;
    asm volatile("ld.global.nc.L1::no_allocate.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
#else
    u64 v;
    memcpy(&v, p, 8);
    return v;
#endif
}
NV4_HD u32 rd16(const void* p)
{
#ifdef __CUDA_ARCH__
    u32 v;
    asm volatile("{ .reg .u16 t; ld.global.nc.L1::no_allocate.u16 t, [%1]; cvt.u32.u16 %0, t; }" : "=r"(v) : "l"(p));
    return v;
#else
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
#endif
}
// prmt.b32: result byte i = byte (sel >> 4 i) & 7 of the 8 bytes {x, y}.
NV4_HD u32 bperm(u32 x, u32 y, u32 sel)
{
#ifdef __CUDA_ARCH__
    return __byte_perm(x, y, sel);
#else
    const u64 v = static_cast<u64>(x) | (static_cast<u64>(y) << 32);
    u32 r = 0;
    for (int i = 0; i < 4; ++i) r |= static_cast<u32>((v >> (8 * ((sel >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return r;
#endif
}
// One 16-bit activation cell -> fp32 (exact).
NV4_HD float cell16(u32 bits, int bf16)
{
#ifdef __CUDA_ARCH__
    return bf16 ? __uint_as_float(bits << 16) : __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
#else
    if (bf16) return u2f(bits << 16);
    const int e = (bits >> 10) & 31, m = bits & 1023;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24)
                  : e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                  : std::ldexp(static_cast<float>(m + 1024), e - 25);
    return bits & 0x8000u ? -v : v;
#endif
}
// fp32 -> BF16 bits, round to nearest even; inf and NaN keep their top 16 bits.
NV4_HD u32 bf16_rne(float f)
{
    u32 u = f2u(f);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

// ---------------------------------------------------------------------------------------------- E2M1 and E4M3
NV4_HD float e2m1_val(u32 code)
{
    const int m = static_cast<int>(code & 7u);
    const float v = m < 4 ? static_cast<float>(m) * 0.5f : m < 6 ? static_cast<float>(m - 2) : static_cast<float>((m - 4) * 2);
    return code & 8u ? -v : v;
}
// = nv4pack::f32_to_e2m1_rne
NV4_HD u32 e2m1_rne(float v)
{
    const float a = fminf(fabsf(v), 6.0f);
    u32 idx;
    if (a < 2.0f) idx = static_cast<u32>(rintf(a * 2.0f));
    else if (a < 4.0f) idx = 4u + static_cast<u32>(rintf(a - 2.0f));
    else idx = 6u + static_cast<u32>(rintf((a - 4.0f) * 0.5f));
    return ((f2u(v) >> 31) && idx != 0 ? 8u : 0u) | idx;
}
// = nv4pack::e4m3_to_f32
NV4_HD float e4m3_val(u32 b)
{
    const u32 e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = static_cast<float>(m) * (1.0f / 512.0f);
    else if (e == 15 && m == 7) return u2f(0x7fc00000u);
    else v = static_cast<float>(8u + m) * u2f((e + 117u) << 23);   // (8 + m) 2^(e - 10)
    return b & 0x80u ? -v : v;
}
// = nv4pack::f32_to_e4m3_rne. In the binade e = max(exponent, -6) the grid step is 2^(e - 3), so n = rint(a 2^(3 - e))
// is 0 .. 16 and the byte is ((e + 7) << 3) + n - 8: n < 8 only in the subnormal binade (byte = n), n = 16 is the next
// binade's first value. The product is exact (a power of two, no overflow), so fp32 gives the fp64 result.
NV4_HD u32 e4m3_rne(float v)
{
    const u32 b = f2u(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0x7fu;
    const float a = fminf(fabsf(v), 448.0f);
    int e = static_cast<int>((f2u(a) >> 23) & 255u) - 127;
    if (e < -6) e = -6;
    const int n = static_cast<int>(rintf(a * u2f(static_cast<u32>(130 - e) << 23)));
    if (n == 0) return 0u;
    u32 byte = (static_cast<u32>(e + 7) << 3) + static_cast<u32>(n) - 8u;
    if (byte > 0x7eu) byte = 0x7eu;
    return (b >> 31 ? 0x80u : 0u) | byte;
}

// ---------------------------------------------------------------------------------------------- block quantizer
NV4_HD float outer_of(float amax, bool four_six) { return amax / (four_six ? 1536.0f : 2688.0f); }

// One candidate of a block of 16 (nv4pack::block_candidate): q = 16 nibbles, k j in bits 4 j .. 4 j + 3, which is the
// 8 code bytes of the block as a little-endian u64. Returns the L1 error.
NV4_HD float block_candidate(const float* x, float b_max, float outer, float target, u32& s, u64& q)
{
    s = outer == 0.0f ? 0u : e4m3_rne((b_max / target) / outer);
    const float sf = mul_rn(outer, e4m3_val(s));
    float err = 0.0f;
    q = 0;
#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (int j = 0; j < 16; ++j)
    {
        const u32 c = sf != 0.0f ? e2m1_rne(x[j] / sf) : 0u;
        const float d = mul_rn(e2m1_val(c), sf) - x[j];
        err += fabsf(d);
        q |= static_cast<u64>(c) << (4 * j);
    }
    return err;
}
// The block under a rule (nv4pack::quantize's inner step): FourOverSixMae takes the / 4 candidate iff its error is
// strictly lower.
NV4_HD void quant_block(const float* x, float outer, bool four_six, u32& s, u64& q)
{
    float b_max = 0.0f;
#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (int j = 0; j < 16; ++j) b_max = fmaxf(b_max, fabsf(x[j]));
    const float e6 = block_candidate(x, b_max, outer, 6.0f, s, q);
    if (four_six)
    {
        u32 s4;
        u64 q4;
        const float e4 = block_candidate(x, b_max, outer, 4.0f, s4, q4);
        if (e4 < e6) { s = s4; q = q4; }
    }
}

// ---------------------------------------------------------------------------------------------- the operand map
// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// Operand layout this file assumes (lane = 4 m + t; "physical k" = the instruction's own k index):
//   a0 / a1  rows m / m + 8, physical k 8 t .. 8 t + 7 (byte i = k 8 t + 2 i in the low nibble, + 1 in the high)
//   a2 / a3  the same rows, physical k 32 + 8 t ..
//   b0 / b1  column m, physical k 8 t .. / 32 + 8 t ..
//   d0, d1 = (row m, columns 2 t, 2 t + 1);  d2, d3 = row m + 8
//   scale A  byte j of the word scales physical k 16 j .. 16 j + 15; selector {0, 0}: lane 4 m gives row m's word,
//            lane 4 m + 1 row m + 8's
//   scale B  lane 4 m gives column m's word
// A lane's a0 therefore lies in block t >> 1 and its a2 in block 2 + (t >> 1). The record gives quad lane c the bytes
// of k 2 c, 2 c + 1, 2 c + 8, 2 c + 9 of EVERY block, so MMA lane c = 2 hb + q takes block words hb and 2 + hb of the
// record lanes 2 q and 2 q + 1 and splits them by row with two byte permutes. Every record byte is read once.
// The k order inside a block that results (the same for A and B, which is all the sum needs):
//   register half (a0 or a2 / b0 or b1) -> block hb or 2 + hb;  byte i, nibble j -> k 4 q + 2 (i >> 1) + 8 (i & 1) + j
struct Frag
{
    static constexpr int GR = nv4pack::GR, KG = nv4pack::KG, SOFF = nv4pack::SOFF, XREC = 32;
    static constexpr bool OUTER = true;          // the format carries fp32 outer scales (weights and activations)
    static constexpr u32 SEL_LO = 0x6420u;       // {W0 byte 0, W0 byte 2, W1 byte 0, W1 byte 2} = row m
    static constexpr u32 SEL_HI = 0x7531u;       // the odd bytes = row m + 8
    static constexpr int BID_A = 0, TID_A = 0, BID_B = 0, TID_B = 0;

    // Byte of an activation record that holds k (0 .. 63) of its group; the nibble is k & 1.
    static constexpr NV4_CX int xrec_byte(int k)
    {
        const int t = k >> 4, kk = k & 15, hb = t & 1, reg = t >> 1;
        const int q = (kk >> 2) & 1, i = (((kk >> 1) & 1) << 1) | ((kk >> 3) & 1);
        return (2 * hb + q) * 8 + reg * 4 + i;
    }

    struct A
    {
        u32 a0, a1, a2, a3, s;
    };
    // One lane's share of the record at rec.
    static NV4_HD A load(const uint8_t* rec, int lane)
    {
        const int r = lane >> 2, c = lane & 3, q = c & 1, hb = c >> 1;
        const uint8_t* p = rec + (r * 4 + 2 * q) * 16 + hb * 4;
        const u32 w0l = rd32(p), w0h = rd32(p + 8), w1l = rd32(p + 16), w1h = rd32(p + 24);
        A o;
        o.a0 = bperm(w0l, w1l, SEL_LO);
        o.a1 = bperm(w0l, w1l, SEL_HI);
        o.a2 = bperm(w0h, w1h, SEL_LO);
        o.a3 = bperm(w0h, w1h, SEL_HI);
        o.s = rd32(rec + SOFF + r * 8 + q * 4);   // even lane: the scales of row r, odd lane: of row r + 8
        return o;
    }
    // One lane's share of an activation record (xr = the 32 bytes of (row, group)).
    static NV4_HD void loadB(const uint8_t* xr, int c, u32& b0, u32& b1)
    {
#ifdef __CUDA_ARCH__
        const u64 v = *reinterpret_cast<const u64*>(xr + c * 8);
#else
        u64 v;
        memcpy(&v, xr + c * 8, 8);
#endif
        b0 = static_cast<u32>(v);
        b1 = static_cast<u32>(v >> 32);
    }
#ifdef __CUDACC__
    // d += A (16 rows x 64 k) @ B (64 k x 8 columns), both block scaled.
    static __device__ __forceinline__ void mma(float (&d)[4], const A& a, u32 b0, u32 b1, u32 sb)
    {
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a.a0), "r"(a.a1), "r"(a.a2), "r"(a.a3), "r"(b0), "r"(b1),
                       "r"(a.s), "n"(BID_A), "n"(TID_A), "r"(sb), "n"(BID_B), "n"(TID_B));
    }
#endif
};

enum { O_F16 = 0, O_BF16 = 1, O_F32 = 2 };

#ifdef __CUDACC__
// ---------------------------------------------------------------------------------------------- quantizer kernel
// One CTA per row. x [rows][K] 16-bit cells (BF16 or FP16). outer = fixed_outer[row], or the row's amax over the
// rule's denominator. n_live (optional, a device int) with live_mul: rows >= *n_live * live_mul are skipped, so a
// caller can launch the largest grid before the row count is known on the host.
constexpr int QT = 256;
__global__ void __launch_bounds__(QT) k_nv4_quant(const uint16_t* x, int K, int in_bf16, int four_six, const float* fixed_outer,
                                                  const int* n_live, int live_mul, uint8_t* xq, u32* xs, float* x_outer)
{
    const int row = blockIdx.x, tid = threadIdx.x;
    if (n_live && row >= *n_live * live_mul) return;
    const uint16_t* xr = x + static_cast<size_t>(row) * K;
    __shared__ float red[QT / 32];
    float outer;
    if (fixed_outer) outer = fixed_outer[row];
    else
    {
        float a = 0.0f;
        for (int i = tid * 8; i < K; i += QT * 8)
        {
            const uint4 v = *reinterpret_cast<const uint4*>(xr + i);
            const u32 w[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) a = fmaxf(a, fmaxf(fabsf(cell16(w[j] & 0xffffu, in_bf16)), fabsf(cell16(w[j] >> 16, in_bf16))));
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        if ((tid & 31) == 0) red[tid >> 5] = a;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < QT / 32; ++j) a = fmaxf(a, red[j]);
        outer = outer_of(a, four_six != 0);
    }
    if (tid == 0) x_outer[row] = outer;
    const int NG = K / 64;
    for (int b = tid; b < K / 16; b += QT)
    {
        const uint4 v0 = *reinterpret_cast<const uint4*>(xr + b * 16), v1 = *reinterpret_cast<const uint4*>(xr + b * 16 + 8);
        const u32 w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float f[16];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            f[2 * j] = cell16(w[j] & 0xffffu, in_bf16);
            f[2 * j + 1] = cell16(w[j] >> 16, in_bf16);
        }
        u32 s;
        u64 q;
        quant_block(f, outer, four_six != 0, s, q);
        const int t = b & 3;
        const size_t grp = static_cast<size_t>(row) * NG + (b >> 2);
        uint8_t* o = xq + grp * Frag::XREC;
        #pragma unroll
        for (int p = 0; p < 8; ++p) o[Frag::xrec_byte(16 * t + 2 * p)] = static_cast<uint8_t>(q >> (8 * p));
        reinterpret_cast<uint8_t*>(xs)[grp * 4 + t] = static_cast<uint8_t>(s);
    }
}

// rows x K cells -> xq [rows][K / 64][32], xs [rows][K / 64], x_outer [rows].
inline void quant(const void* x, int rows, int K, bool in_bf16, ScaleRule rule, const float* fixed_outer, const int* n_live, int live_mul,
                  uint8_t* xq, u32* xs, float* x_outer, cudaStream_t st)
{
    if (rows < 1 || K % 64) throw std::runtime_error("nv4::quant: rows < 1 or K % 64");
    k_nv4_quant<<<rows, QT, 0, st>>>(static_cast<const uint16_t*>(x), K, in_bf16 ? 1 : 0, rule == ScaleRule::FourOverSixMae ? 1 : 0, fixed_outer,
                                     n_live, live_mul, xq, xs, x_outer);
}

// ---------------------------------------------------------------------------------------------- dense kernels
// y[m][n] = (w_outer * x_outer[m]) * sum_k codes. w = [N / 16][K / 64][GR]. A warp owns 16 output rows and NT token
// tiles of 8; it reads its records once (ld.global.nc) and runs NT mma per record.
constexpr int DW = 4;   // warps per CTA
template <int NT>
__global__ void __launch_bounds__(DW * 32) k_dense(const uint8_t* w, int N, int K, const uint8_t* xq, const u32* xs, const float* x_outer,
                                                   float w_outer, int M, void* y, int odt)
{
    const int lane = threadIdx.x & 31, r = lane >> 2, c = lane & 3;
    const int ot = blockIdx.x * DW + (threadIdx.x >> 5);
    if (ot >= N / 16) return;
    const int NG = K / 64, m0 = blockIdx.y * NT * 8;
    float acc[NT][4];
    u32 xo[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
        xo[t] = static_cast<u32>(min(m0 + t * 8 + r, M - 1)) * NG;
    }
    const uint8_t* rec = w + static_cast<size_t>(ot) * NG * Frag::GR;
    for (int g = 0; g < NG; ++g)
    {
        const Frag::A a = Frag::load(rec + static_cast<size_t>(g) * Frag::GR, lane);
        #pragma unroll
        for (int t = 0; t < NT; ++t)
            if (m0 + t * 8 < M)
            {
                u32 b0, b1;
                Frag::loadB(xq + static_cast<size_t>(xo[t] + g) * Frag::XREC, c, b0, b1);
                Frag::mma(acc[t], a, b0, b1, xs[xo[t] + g]);
            }
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int m = m0 + t * 8 + 2 * c + (j & 1), n = ot * 16 + r + 8 * (j >> 1);
            if (m >= M) continue;
            const float v = acc[t][j] * (w_outer * x_outer[m]);
            const size_t i = static_cast<size_t>(m) * N + n;
            if (odt == O_F32) static_cast<float*>(y)[i] = v;
            else if (odt == O_BF16) static_cast<uint16_t*>(y)[i] = static_cast<uint16_t>(bf16_rne(v));
            else static_cast<__half*>(y)[i] = __float2half_rn(v);
        }
}

inline void dense_check(int N, int K, int M)
{
    if (N % 16 || K % 64 || N < 16 || K < 64 || M < 1) throw std::runtime_error("nv4::dense: N % 16, K % 64 or M < 1");
}
// M <= 8 tokens: one token tile per warp.
inline void dense_gemv(const uint8_t* w, int N, int K, const uint8_t* xq, const u32* xs, const float* x_outer, float w_outer, int M, void* y,
                       int odt, cudaStream_t st)
{
    dense_check(N, K, M);
    if (M > 8) throw std::runtime_error("nv4::dense_gemv: M > 8");
    k_dense<1><<<dim3((N / 16 + DW - 1) / DW, 1), DW * 32, 0, st>>>(w, N, K, xq, xs, x_outer, w_outer, M, y, odt);
}
// Any M: 4 token tiles per warp, so the weights are read once per 32 tokens.
inline void dense_mt(const uint8_t* w, int N, int K, const uint8_t* xq, const u32* xs, const float* x_outer, float w_outer, int M, void* y,
                     int odt, cudaStream_t st)
{
    dense_check(N, K, M);
    k_dense<4><<<dim3((N / 16 + DW - 1) / DW, (M + 31) / 32), DW * 32, 0, st>>>(w, N, K, xq, xs, x_outer, w_outer, M, y, odt);
}
#endif   // __CUDACC__
}   // namespace nv4
