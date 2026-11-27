// Host CPU mul1 (0x83DCD12D E4M3FN) Wide-Order Routed MoE Expert Evaluator for ERNIE-5.5-Flash Hybrid Mode
// (`GR8_ERNIE55_HYBRID_CPU_OVERLAP=1`, M=1 decode).
//
// Evaluates cold K2/K3/K4 (+ per-matrix kmix) routed experts directly in host DDR5 RAM in parallel with
// GPU hot-expert execution and PCIe cold-expert H2D staging. Pure C++20 header (zero CUDA dependency) so
// both CUDA runtime code and host-only unit tests compile and verify the exact same implementation.
//
// Math invariants (bit-exact codebook + tile layout matching engine/kernels/ernie55_moe.cuh):
//   - mul1 codebook: prod = s * 0x83DCD12Du; bsum = sum of 4 unsigned bytes of prod (in [0, 1020]);
//     x = fp16(0x6400 + bsum) = 1024 + bsum; y = fp16_rn(fma(x, fp16(0x1eee), fp16(0xc931)));
//     w = e4m3fn_to_f32(f32_to_e4m3fn_sat(y)).
//   - Wide-order 16x128 tile GEMV for BITS in {2, 3, 4} matching ernie55_mul1_fc1_hb_wide &
//     ernie55_mul1_fc2_ntile_wide + mma.m16n8k16.row.col.f32.f16.f16.f32.
//   - Runtime-dispatched x86_64 AVX-512 VNNI (`vpdpbusd` + 4 KiB L1D-resident bsum_lut[1024]) fast path
//     with portable scalar fallback (zero -march=native build requirement, safe on any x86_64 or aarch64).
//   - Exact 7-stage Walsh-Hadamard had128, BF16-rounded SiTU-GLU(4, 25), FP16 fc1_act rounding, and
//     v1b/v2/v3b/sg/topk_w scaling -> outputs 2048 FP32 values identical to fc2_part_r[route * 2048].
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDA_ARCH__)
#include <immintrin.h>
#define ERNIE55_CPU_CAN_AVX512 1
#else
#define ERNIE55_CPU_CAN_AVX512 0
#endif

namespace ernie55cpu {

static constexpr int EH = 2048;
static constexpr int EHB = 16;
static constexpr int EB_MAX = 16;
static constexpr float SQ128 = 0.08838834764831844f; // 1 / sqrt(128)
static constexpr float SITU_BETA = 4.0f;
static constexpr float SITU_LINEAR_BETA = 25.0f;

// =================================================================================================
// Exact IEEE 754 FP16 / BF16 / E4M3FN Scalar Helpers
// =================================================================================================
inline float f16_to_f32(uint16_t h) {
    const uint32_t s = static_cast<uint32_t>(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    uint32_t u = 0;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            uint32_t mm = m;
            int exp = -14;
            while ((mm & 0x400u) == 0) { mm <<= 1; --exp; }
            mm &= 0x3ffu;
            u = s | (static_cast<uint32_t>(exp + 127) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_f16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint16_t s = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const int e = static_cast<int>((u >> 23) & 0xffu) - 127;
    const uint32_t m = u & 0x7fffffu;
    if (e == 128) return static_cast<uint16_t>(s | 0x7c00u | (m ? 0x0200u : 0u));
    if (e > 15) return static_cast<uint16_t>(s | 0x7c00u);
    if (e < -24) return s;
    if (e < -14) {
        const uint32_t full = m | 0x800000u;
        const int shift = -14 - e + 13; // in [14, 24]
        const uint32_t mant = full >> shift;
        const uint32_t rem = full & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        const uint32_t round_up = (rem > half || (rem == half && (mant & 1u))) ? 1u : 0u;
        return static_cast<uint16_t>(s | (mant + round_up));
    }
    uint32_t mant = m >> 13;
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
        ++mant;
        if (mant == 0x400u) {
            mant = 0;
            const int en = e + 1;
            if (en > 15) return static_cast<uint16_t>(s | 0x7c00u);
            return static_cast<uint16_t>(s | (static_cast<uint32_t>(en + 15) << 10));
        }
    }
    return static_cast<uint16_t>(s | (static_cast<uint32_t>(e + 15) << 10) | mant);
}

inline float f16_round_f32(float f) {
    return f16_to_f32(f32_to_f16_rn(f));
}

inline float bf16_round_f32(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        if (u & 0x007fffffu) u |= 0x00400000u;
    } else {
        const uint32_t lsb = (u >> 16) & 1u;
        u += 0x7fffu + lsb;
    }
    u &= 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

inline uint8_t f32_to_e4m3fn_sat(float v) {
    if (std::isnan(v)) return 0x7fu;
    const uint8_t sign = std::signbit(v) ? 0x80u : 0x00u;
    float av = std::fabs(v);
    if (av >= 448.0f) return static_cast<uint8_t>(sign | 0x7eu);
    if (av == 0.0f) return sign;
    int e = 0;
    float m = std::frexp(av, &e); // av = m * 2^e, m in [0.5, 1)
    int exp_unbiased = e - 1;
    if (exp_unbiased < -6) {
        float scaled = std::ldexp(av, 9); // av / 2^-9
        int q = static_cast<int>(std::nearbyint(scaled));
        if (q <= 0) return sign;
        if (q >= 8) return static_cast<uint8_t>(sign | (1u << 3));
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(q));
    }
    int biased = exp_unbiased + 7;
    float frac = (m * 2.0f - 1.0f) * 8.0f;
    int q = static_cast<int>(std::nearbyint(frac));
    if (q == 8) { q = 0; ++biased; }
    if (biased > 15 || (biased == 15 && q >= 7)) return static_cast<uint8_t>(sign | 0x7eu);
    return static_cast<uint8_t>(sign | (static_cast<uint8_t>(biased) << 3) | static_cast<uint8_t>(q));
}

inline float e4m3fn_to_f32(uint8_t q) {
    const float s = (q & 0x80u) ? -1.0f : 1.0f;
    const int exp = (q >> 3) & 0xf;
    const int mant = q & 0x7;
    if (exp == 0) return s * std::ldexp(static_cast<float>(mant), -9);
    if (exp == 15 && mant == 7) return std::nanf("");
    return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

// =================================================================================================
// Exact mul1 (0x83DCD12D -> FP16 FMA 0x1eee/0xc931 -> E4M3FN) Codebook Tables
// =================================================================================================
struct Mul1Tables {
    alignas(64) float bsum_lut[1024]; // 4 KiB (100% L1D-resident, indexed by unsigned byte-sum 0..1020)
    alignas(64) float lut[65536];     // 256 KiB (L2-resident, indexed directly by 16-bit trellis state)

    Mul1Tables() {
        const float c1 = f16_to_f32(0x1eee); // 0.00676727294921875f
        const float c0 = f16_to_f32(0xc931); // -10.3828125f
        for (uint32_t b = 0; b < 1024; ++b) {
            // 0x6400 + b is exact FP16 for 1024 + b (for b in 0..1020).
            const float x = 1024.0f + static_cast<float>(b);
            const float y = f16_to_f32(f32_to_f16_rn(std::fma(x, c1, c0)));
            bsum_lut[b] = e4m3fn_to_f32(f32_to_e4m3fn_sat(y));
        }
        for (uint32_t s = 0; s < 65536; ++s) {
            const uint32_t prod = s * 0x83DCD12Du;
            const uint32_t bsum = (prod & 0xffu) + ((prod >> 8) & 0xffu) +
                                  ((prod >> 16) & 0xffu) + ((prod >> 24) & 0xffu);
            lut[s] = bsum_lut[bsum];
        }
    }
};

inline const Mul1Tables& mul1_tables() {
    static const Mul1Tables t;
    return t;
}

inline int fc1_unit_width(long long km, int dk, int unit) {
    if (km == 0LL) return dk;
    if (km < 0LL) return static_cast<int>((km >> ((unit & 1) ? 4 : 0)) & 0xfull);
    return static_cast<int>((km >> (4 * unit)) & 0xfull);
}

// =================================================================================================
// Exact 7-stage Walsh-Hadamard had128 and SiTU-GLU(4, 25)
// =================================================================================================
inline void had128_cpu(float v[128]) {
    for (int len = 1; len < 128; len <<= 1) {
        const int step = len << 1;
        for (int i = 0; i < 128; i += step) {
            for (int j = 0; j < len; ++j) {
                const float a = v[i + j];
                const float b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) v[i] *= SQ128;
}

inline float situ_mul_cpu(float g, float u) {
    const float sig = 1.0f / (1.0f + std::exp(-g));
    const float ga = SITU_BETA * std::tanh(g * (1.0f / SITU_BETA)) * sig;
    const float ua = SITU_LINEAR_BETA * std::tanh(u * (1.0f / SITU_LINEAR_BETA));
    return ga * ua;
}

// =================================================================================================
// Pre-packed 128-element input layout matching mma.m16n8k16 operand B for M=1:
//   xp[p][0] = { x[p*16+0], x[p*16+2], x[p*16+4], x[p*16+6] }
//   xp[p][1] = { x[p*16+1], x[p*16+3], x[p*16+5], x[p*16+7] }
//   xp[p][2] = { x[p*16+8], x[p*16+10], x[p*16+12], x[p*16+14] }
//   xp[p][3] = { x[p*16+9], x[p*16+11], x[p*16+13], x[p*16+15] }
// =================================================================================================
struct alignas(64) PackedX128 {
    float v[8][4][4];
};

inline void pack_x128(const float x[128], PackedX128& px) {
    for (int p = 0; p < 8; ++p) {
        const float* xp = x + p * 16;
        px.v[p][0][0] = xp[0];  px.v[p][0][1] = xp[2];  px.v[p][0][2] = xp[4];  px.v[p][0][3] = xp[6];
        px.v[p][1][0] = xp[1];  px.v[p][1][1] = xp[3];  px.v[p][1][2] = xp[5];  px.v[p][1][3] = xp[7];
        px.v[p][2][0] = xp[8];  px.v[p][2][1] = xp[10]; px.v[p][2][2] = xp[12]; px.v[p][2][3] = xp[14];
        px.v[p][3][0] = xp[9];  px.v[p][3][1] = xp[11]; px.v[p][3][2] = xp[13]; px.v[p][3][3] = xp[15];
    }
}

// =================================================================================================
// Portable Scalar Wide-Order 16x128 Tile GEMV (K=2, K=3, K=4)
// =================================================================================================
template <int BITS, size_t PLANE_STRIDE>
inline void mul1_tile16x128_dot_scalar(const uint8_t* tile, const PackedX128& px, float out[16]) {
    const float* lut = mul1_tables().lut;
    uint32_t ch[8][32];
    if constexpr (BITS == 2) {
        const uint8_t* p0 = tile;
        const uint8_t* p1 = tile + PLANE_STRIDE;
        for (int lane = 0; lane < 32; ++lane) {
            uint64_t w0, w1;
            std::memcpy(&w0, p0 + lane * 8, 8);
            std::memcpy(&w1, p1 + lane * 8, 8);
            ch[0][lane] = static_cast<uint32_t>(w0 & 0xffffu);
            ch[1][lane] = static_cast<uint32_t>((w0 >> 16) & 0xffffu);
            ch[2][lane] = static_cast<uint32_t>((w0 >> 32) & 0xffffu);
            ch[3][lane] = static_cast<uint32_t>((w0 >> 48) & 0xffffu);
            ch[4][lane] = static_cast<uint32_t>(w1 & 0xffffu);
            ch[5][lane] = static_cast<uint32_t>((w1 >> 16) & 0xffffu);
            ch[6][lane] = static_cast<uint32_t>((w1 >> 32) & 0xffffu);
            ch[7][lane] = static_cast<uint32_t>((w1 >> 48) & 0xffffu);
        }
    } else if constexpr (BITS == 3) {
        const uint8_t* p0 = tile;
        const uint8_t* p1 = tile + PLANE_STRIDE;
        const uint8_t* p2 = tile + 2 * PLANE_STRIDE;
        for (int lane = 0; lane < 32; ++lane) {
            uint64_t w0, w1, w2;
            std::memcpy(&w0, p0 + lane * 8, 8);
            std::memcpy(&w1, p1 + lane * 8, 8);
            std::memcpy(&w2, p2 + lane * 8, 8);
            ch[0][lane] = static_cast<uint32_t>(w0 & 0xffffffu);
            ch[1][lane] = static_cast<uint32_t>((w0 >> 24) & 0xffffffu);
            ch[2][lane] = static_cast<uint32_t>(((w0 >> 48) | (w1 << 16)) & 0xffffffu);
            ch[3][lane] = static_cast<uint32_t>((w1 >> 8) & 0xffffffu);
            ch[4][lane] = static_cast<uint32_t>((w1 >> 32) & 0xffffffu);
            ch[5][lane] = static_cast<uint32_t>(((w1 >> 56) | (w2 << 8)) & 0xffffffu);
            ch[6][lane] = static_cast<uint32_t>((w2 >> 16) & 0xffffffu);
            ch[7][lane] = static_cast<uint32_t>((w2 >> 40) & 0xffffffu);
        }
    } else { // BITS == 4
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t w0[4], w1[4];
            std::memcpy(w0, tile + lane * 16, 16);
            std::memcpy(w1, tile + 512 + lane * 16, 16);
            ch[0][lane] = w0[0]; ch[1][lane] = w0[1]; ch[2][lane] = w0[2]; ch[3][lane] = w0[3];
            ch[4][lane] = w1[0]; ch[5][lane] = w1[1]; ch[6][lane] = w1[2]; ch[7][lane] = w1[3];
        }
    }

    for (int r = 0; r < 16; ++r) out[r] = 0.0f;
    constexpr int B0 = 7 * BITS;
    for (int p = 0; p < 8; ++p) {
        const float* x0 = px.v[p][0];
        const float* x1 = px.v[p][1];
        const float* x2 = px.v[p][2];
        const float* x3 = px.v[p][3];
        for (int lane = 0; lane < 32; ++lane) {
            const int g = lane >> 2;
            const int col = lane & 3;
            const int prev = (lane - 1) & 31;
            const uint64_t m = (static_cast<uint64_t>(ch[p][prev]) << (8 * BITS)) | ch[p][lane];
            const float v0 = x0[col], v1 = x1[col], v2 = x2[col], v3 = x3[col];
            const float lo = lut[(m >> (B0 - 0 * BITS)) & 0xffffu] * v0 +
                             lut[(m >> (B0 - 1 * BITS)) & 0xffffu] * v1 +
                             lut[(m >> (B0 - 2 * BITS)) & 0xffffu] * v2 +
                             lut[(m >> (B0 - 3 * BITS)) & 0xffffu] * v3;
            const float hi = lut[(m >> (B0 - 4 * BITS)) & 0xffffu] * v0 +
                             lut[(m >> (B0 - 5 * BITS)) & 0xffffu] * v1 +
                             lut[(m >> (B0 - 6 * BITS)) & 0xffffu] * v2 +
                             lut[(m >> (B0 - 7 * BITS)) & 0xffffu] * v3;
            out[g]     += lo;
            out[g + 8] += hi;
        }
    }
}

// =================================================================================================
// Runtime-Dispatched x86_64 AVX-512 VNNI Fast Path (Zen 5 / Sapphire Rapids+)
// =================================================================================================
#if ERNIE55_CPU_CAN_AVX512

enum class Avx512DecodeMode : int {
    VNNI_REG = 0, // 100% register-only vpdpbusd + IEEE FP16/E4M3FN bit-exact (ZERO gather / ZERO LUT!)
    VNNI_L1  = 1, // vpdpbusd + 4 KiB L1D bsum_lut[1024] gather
    LUT_L2   = 2, // direct 256 KiB L2 lut[65536] gather
};

inline bool cpu_has_avx512_vnni() {
    static const bool ok = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") &&
               __builtin_cpu_supports("avx512dq") &&
               __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") &&
               __builtin_cpu_supports("avx512vnni") &&
               __builtin_cpu_supports("fma");
    }();
    return ok;
}

#define ERNIE55_AVX512_TARGET __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx512vnni,fma")))
#define ERNIE55_AVX512_INLINE __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx512vnni,fma"), always_inline))

ERNIE55_AVX512_INLINE
inline void sum4_to4_avx512(__m512 v, float dst[4]) {
    const __m512 s1 = _mm512_add_ps(v, _mm512_shuffle_ps(v, v, 0xB1)); // swap pairs (0<->1, 2<->3)
    const __m512 s2 = _mm512_add_ps(s1, _mm512_shuffle_ps(s1, s1, 0x4E)); // swap halves (01<->23)
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, s2);
    dst[0] = tmp[0]; dst[1] = tmp[4]; dst[2] = tmp[8]; dst[3] = tmp[12];
}

ERNIE55_AVX512_INLINE
inline void reduce4_to_out16(__m512 lo0, __m512 lo1, __m512 hi0, __m512 hi1, float out[16]) {
    sum4_to4_avx512(lo0, out + 0);
    sum4_to4_avx512(lo1, out + 4);
    sum4_to4_avx512(hi0, out + 8);
    sum4_to4_avx512(hi1, out + 12);
}

template <Avx512DecodeMode MODE>
ERNIE55_AVX512_INLINE
inline __m512 mul1_decode16_avx512(__m512i w, const float* bsum_lut, const float* lut,
                                   __m512i k_mul, __m512i k_ones, __m512i k_zero) {
    if constexpr (MODE == Avx512DecodeMode::VNNI_REG) {
        (void)bsum_lut; (void)lut; (void)k_zero;
        // 100% register-only, branchless, 65536/65536 bit-exact mul1 -> FP16 -> E4M3FN decode!
        // Zero vpgatherdd, zero vcvtps2ph/vcvtph2ps: |yf| in [0.002, 3.453] is always normal FP16
        // and never hits an FP16 halfway tie, so (u + 0x1000) & 0xffffe000 matches FP16 RNE bit-for-bit.
        const __m512i prod = _mm512_mullo_epi32(w, k_mul);
        const __m512i x_i  = _mm512_dpbusd_epi32(_mm512_set1_epi32(1024), prod, k_ones);
        const __m512  yf   = _mm512_fmadd_ps(_mm512_cvtepi32_ps(x_i),
                                             _mm512_set1_ps(0.00676727294921875f),
                                             _mm512_set1_ps(-10.3828125f));
        const __m512i u    = _mm512_castps_si512(yf);
        const __m512i uh   = _mm512_and_si512(_mm512_add_epi32(u, _mm512_set1_epi32(0x1000)),
                                             _mm512_set1_epi32(static_cast<int>(0xffffe000u)));
        const __m512i l20  = _mm512_and_si512(_mm512_srli_epi32(uh, 20), _mm512_set1_epi32(1));
        const __m512i uq   = _mm512_and_si512(
            _mm512_add_epi32(_mm512_add_epi32(uh, _mm512_set1_epi32(0x7ffff)), l20),
            _mm512_set1_epi32(static_cast<int>(0xfff00000u)));
        const __m512  qn   = _mm512_castsi512_ps(uq);
        // E4M3FN subnormal path (|yh| < 2^-6): 1.5 * 2^14 = 24576.0f has FP32 ULP = 2^-9 -> exact RNE!
        const __m512i au   = _mm512_and_si512(uh, _mm512_set1_epi32(0x7fffffff));
        const __mmask16 sub = _mm512_cmplt_epu32_mask(au, _mm512_set1_epi32(0x3c800000));
        const __m512  k_mag = _mm512_set1_ps(24576.0f);
        const __m512  yh    = _mm512_castsi512_ps(uh);
        return _mm512_mask_sub_ps(qn, sub, _mm512_add_ps(yh, k_mag), k_mag);
    } else if constexpr (MODE == Avx512DecodeMode::VNNI_L1) {
        (void)lut;
        const __m512i prod = _mm512_mullo_epi32(w, k_mul);
        const __m512i bsum = _mm512_dpbusd_epi32(k_zero, prod, k_ones);
        return _mm512_i32gather_ps(bsum, bsum_lut, 4);
    } else {
        (void)bsum_lut; (void)k_mul; (void)k_ones; (void)k_zero;
        return _mm512_i32gather_ps(w, lut, 4);
    }
}

template <Avx512DecodeMode MODE>
ERNIE55_AVX512_INLINE
inline void k2_step_p_avx512(int p, __m512i ch0, __m512i ch1, const PackedX128& px,
                             const float* bsum_lut, const float* lut,
                             __m512i k_mul, __m512i k_ones, __m512i k_zero, __m512i k_m16,
                             __m512& acc_lo0, __m512& acc_lo1, __m512& acc_hi0, __m512& acc_hi1) {
    const __m512i pc0 = _mm512_alignr_epi32(ch0, ch1, 15);
    const __m512i pc1 = _mm512_alignr_epi32(ch1, ch0, 15);
    const __m512i m0  = _mm512_or_si512(_mm512_slli_epi32(pc0, 16), ch0);
    const __m512i m1  = _mm512_or_si512(_mm512_slli_epi32(pc1, 16), ch1);

    const __m512 vx0 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][0]));
    const __m512 vx1 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][1]));
    const __m512 vx2 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][2]));
    const __m512 vx3 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][3]));

    // Interleave m0/m1 chains for maximum out-of-order ILP across 4 independent accumulators
    acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0, 14), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_lo0);
    acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1, 14), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_lo1);
    acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0, 12), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_lo0);
    acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1, 12), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_lo1);
    acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0, 10), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_lo0);
    acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1, 10), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_lo1);
    acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0,  8), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_lo0);
    acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1,  8), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_lo1);

    acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0,  6), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_hi0);
    acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1,  6), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_hi1);
    acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0,  4), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_hi0);
    acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1,  4), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_hi1);
    acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m0,  2), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_hi0);
    acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(_mm512_srli_epi32(m1,  2), k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_hi1);
    acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(m0,                        k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_hi0);
    acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_and_si512(m1,                        k_m16), bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_hi1);
}

// AVX-512 VNNI wide-order K=2 16x128 tile GEMV (full 16-lane SIMD across lanes 0..15 and 16..31).
template <size_t PLANE_STRIDE, Avx512DecodeMode MODE = Avx512DecodeMode::VNNI_L1>
ERNIE55_AVX512_TARGET
inline void mul1_tile16x128_dot_k2_avx512(const uint8_t* tile, const PackedX128& px, float out[16]) {
    const Mul1Tables& tbl = mul1_tables();
    const float* bsum_lut = tbl.bsum_lut;
    const float* lut = tbl.lut;
    const __m512i k_mul  = _mm512_set1_epi32(static_cast<int>(0x83DCD12Du));
    const __m512i k_ones = _mm512_set1_epi32(0x01010101);
    const __m512i k_zero = _mm512_setzero_si512();
    const __m512i k_m16  = _mm512_set1_epi32(0xffff);
    const __m512i idx_ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i idx_od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    __m512 acc_lo0 = _mm512_setzero_ps();
    __m512 acc_lo1 = _mm512_setzero_ps();
    __m512 acc_hi0 = _mm512_setzero_ps();
    __m512 acc_hi1 = _mm512_setzero_ps();

    for (int pl_idx = 0; pl_idx < 2; ++pl_idx) {
        const uint8_t* pl = tile + static_cast<size_t>(pl_idx) * PLANE_STRIDE;
        const int p_base = pl_idx * 4;
        const __m512i a = _mm512_loadu_si512(pl + 0);
        const __m512i b = _mm512_loadu_si512(pl + 64);
        const __m512i c = _mm512_loadu_si512(pl + 128);
        const __m512i d = _mm512_loadu_si512(pl + 192);
        const __m512i ev0 = _mm512_permutex2var_epi32(a, idx_ev, b);
        const __m512i od0 = _mm512_permutex2var_epi32(a, idx_od, b);
        const __m512i ev1 = _mm512_permutex2var_epi32(c, idx_ev, d);
        const __m512i od1 = _mm512_permutex2var_epi32(c, idx_od, d);
        k2_step_p_avx512<MODE>(p_base + 0, _mm512_and_si512(ev0, k_m16), _mm512_and_si512(ev1, k_m16), px, bsum_lut, lut, k_mul, k_ones, k_zero, k_m16, acc_lo0, acc_lo1, acc_hi0, acc_hi1);
        k2_step_p_avx512<MODE>(p_base + 1, _mm512_srli_epi32(ev0, 16),   _mm512_srli_epi32(ev1, 16),   px, bsum_lut, lut, k_mul, k_ones, k_zero, k_m16, acc_lo0, acc_lo1, acc_hi0, acc_hi1);
        k2_step_p_avx512<MODE>(p_base + 2, _mm512_and_si512(od0, k_m16), _mm512_and_si512(od1, k_m16), px, bsum_lut, lut, k_mul, k_ones, k_zero, k_m16, acc_lo0, acc_lo1, acc_hi0, acc_hi1);
        k2_step_p_avx512<MODE>(p_base + 3, _mm512_srli_epi32(od0, 16),   _mm512_srli_epi32(od1, 16),   px, bsum_lut, lut, k_mul, k_ones, k_zero, k_m16, acc_lo0, acc_lo1, acc_hi0, acc_hi1);
    }
    reduce4_to_out16(acc_lo0, acc_lo1, acc_hi0, acc_hi1, out);
}

// AVX-512 VNNI wide-order K=3 and K=4 16x128 tile GEMV (packs 16-lane u32 chunks, 64-bit shift windows).
template <int BITS, size_t PLANE_STRIDE, Avx512DecodeMode MODE = Avx512DecodeMode::VNNI_L1>
ERNIE55_AVX512_TARGET
inline void mul1_tile16x128_dot_k34_avx512(const uint8_t* tile, const PackedX128& px, float out[16]) {
    const Mul1Tables& tbl = mul1_tables();
    const float* bsum_lut = tbl.bsum_lut;
    const float* lut = tbl.lut;
    const __m512i k_mul  = _mm512_set1_epi32(static_cast<int>(0x83DCD12Du));
    const __m512i k_ones = _mm512_set1_epi32(0x01010101);
    const __m512i k_zero = _mm512_setzero_si512();

    alignas(64) uint32_t ch[8][32];
    if constexpr (BITS == 3) {
        const uint8_t* p0 = tile;
        const uint8_t* p1 = tile + PLANE_STRIDE;
        const uint8_t* p2 = tile + 2 * PLANE_STRIDE;
        for (int lane = 0; lane < 32; ++lane) {
            uint64_t w0, w1, w2;
            std::memcpy(&w0, p0 + lane * 8, 8);
            std::memcpy(&w1, p1 + lane * 8, 8);
            std::memcpy(&w2, p2 + lane * 8, 8);
            ch[0][lane] = static_cast<uint32_t>(w0 & 0xffffffu);
            ch[1][lane] = static_cast<uint32_t>((w0 >> 24) & 0xffffffu);
            ch[2][lane] = static_cast<uint32_t>(((w0 >> 48) | (w1 << 16)) & 0xffffffu);
            ch[3][lane] = static_cast<uint32_t>((w1 >> 8) & 0xffffffu);
            ch[4][lane] = static_cast<uint32_t>((w1 >> 32) & 0xffffffu);
            ch[5][lane] = static_cast<uint32_t>(((w1 >> 56) | (w2 << 8)) & 0xffffffu);
            ch[6][lane] = static_cast<uint32_t>((w2 >> 16) & 0xffffffu);
            ch[7][lane] = static_cast<uint32_t>((w2 >> 40) & 0xffffffu);
        }
    } else { // BITS == 4
        for (int lane = 0; lane < 32; ++lane) {
            uint32_t w0[4], w1[4];
            std::memcpy(w0, tile + lane * 16, 16);
            std::memcpy(w1, tile + 512 + lane * 16, 16);
            ch[0][lane] = w0[0]; ch[1][lane] = w0[1]; ch[2][lane] = w0[2]; ch[3][lane] = w0[3];
            ch[4][lane] = w1[0]; ch[5][lane] = w1[1]; ch[6][lane] = w1[2]; ch[7][lane] = w1[3];
        }
    }

    __m512 acc_lo0 = _mm512_setzero_ps();
    __m512 acc_lo1 = _mm512_setzero_ps();
    __m512 acc_hi0 = _mm512_setzero_ps();
    __m512 acc_hi1 = _mm512_setzero_ps();
    constexpr int B0 = 7 * BITS;
    alignas(64) uint32_t wbuf[8][32];

    for (int p = 0; p < 8; ++p) {
        for (int lane = 0; lane < 32; ++lane) {
            const int prev = (lane - 1) & 31;
            const uint64_t m = (static_cast<uint64_t>(ch[p][prev]) << (8 * BITS)) | ch[p][lane];
            wbuf[0][lane] = static_cast<uint32_t>((m >> (B0 - 0 * BITS)) & 0xffffu);
            wbuf[1][lane] = static_cast<uint32_t>((m >> (B0 - 1 * BITS)) & 0xffffu);
            wbuf[2][lane] = static_cast<uint32_t>((m >> (B0 - 2 * BITS)) & 0xffffu);
            wbuf[3][lane] = static_cast<uint32_t>((m >> (B0 - 3 * BITS)) & 0xffffu);
            wbuf[4][lane] = static_cast<uint32_t>((m >> (B0 - 4 * BITS)) & 0xffffu);
            wbuf[5][lane] = static_cast<uint32_t>((m >> (B0 - 5 * BITS)) & 0xffffu);
            wbuf[6][lane] = static_cast<uint32_t>((m >> (B0 - 6 * BITS)) & 0xffffu);
            wbuf[7][lane] = static_cast<uint32_t>((m >> (B0 - 7 * BITS)) & 0xffffu);
        }
        const __m512 vx0 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][0]));
        const __m512 vx1 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][1]));
        const __m512 vx2 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][2]));
        const __m512 vx3 = _mm512_broadcast_f32x4(_mm_loadu_ps(px.v[p][3]));

        acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[0][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_lo0);
        acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[1][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_lo0);
        acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[2][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_lo0);
        acc_lo0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[3][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_lo0);
        acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[4][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_hi0);
        acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[5][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_hi0);
        acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[6][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_hi0);
        acc_hi0 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[7][0]),  bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_hi0);

        acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[0][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_lo1);
        acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[1][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_lo1);
        acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[2][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_lo1);
        acc_lo1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[3][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_lo1);
        acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[4][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx0, acc_hi1);
        acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[5][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx1, acc_hi1);
        acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[6][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx2, acc_hi1);
        acc_hi1 = _mm512_fmadd_ps(mul1_decode16_avx512<MODE>(_mm512_load_si512(&wbuf[7][16]), bsum_lut, lut, k_mul, k_ones, k_zero), vx3, acc_hi1);
    }
    reduce4_to_out16(acc_lo0, acc_lo1, acc_hi0, acc_hi1, out);
}

#undef ERNIE55_AVX512_TARGET
#undef ERNIE55_AVX512_INLINE

#endif // ERNIE55_CPU_CAN_AVX512

template <int BITS, size_t PLANE_STRIDE>
inline void mul1_tile16x128_dot_cpu(const uint8_t* tile, const PackedX128& px, float out[16]) {
#if ERNIE55_CPU_CAN_AVX512
    if (cpu_has_avx512_vnni()) {
        if constexpr (BITS == 2) {
            mul1_tile16x128_dot_k2_avx512<PLANE_STRIDE, Avx512DecodeMode::VNNI_REG>(tile, px, out);
        } else {
            mul1_tile16x128_dot_k34_avx512<BITS, PLANE_STRIDE, Avx512DecodeMode::VNNI_REG>(tile, px, out);
        }
        return;
    }
#endif
    mul1_tile16x128_dot_scalar<BITS, PLANE_STRIDE>(tile, px, out);
}

// =================================================================================================
// Single-Block (FC1 block b -> SiTU-GLU -> had128 -> FC2 block b accumulate) and Full Expert Eval
// =================================================================================================
struct CpuExpertDesc {
    const uint8_t*  exp_fc1 = nullptr; // wide-reordered base for expert e (ld.h_fc1 + tr[0])
    const uint8_t*  exp_fc2 = nullptr; // wide-reordered base for expert e (ld.h_fc2 + tr[1])
    const size_t*   piece_fc1_off = nullptr; // [nb]
    const size_t*   piece_fc2_off = nullptr; // [nb]
    const uint16_t* v1b = nullptr;     // [nblk * 2 * EHB] FP16
    const uint16_t* v2  = nullptr;     // [nblk * 3 * 128] FP16
    const uint16_t* v3b = nullptr;     // [nblk * EHB] FP16
    const uint16_t* sg_out = nullptr;  // [EH] FP16 (sg + EH)
    size_t          brow0 = 0;         // tr[2]
    long long       kmix = 0;          // kmix[e] (or 0)
    int             nb = 0;            // tr[3]
    int             dk = 2;            // tr[4]
    float           topk_w = 1.0f;     // routing weight for this route
    float*          out_2048 = nullptr;// destination [EH] FP32
};

inline void pack_x_had_layer(const uint16_t* x_had_r, PackedX128 px_layer[EHB]) {
    float xf[128];
    for (int hb = 0; hb < EHB; ++hb) {
        const uint16_t* src = x_had_r + hb * 128;
        for (int c = 0; c < 128; ++c) xf[c] = f16_to_f32(src[c]);
        pack_x128(xf, px_layer[hb]);
    }
}

// Evaluate blocks b in [b_lo, b_hi) of expert `d` and accumulate into `fc2_acc[EH]` (must be pre-zeroed).
inline void eval_expert_blocks_cpu(const CpuExpertDesc& d, const PackedX128 px_layer[EHB],
                                   int b_lo, int b_hi, float fc2_acc[EH]) {
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(EHB) * 2048;
    const int b_end = std::min(b_hi, d.nb);
    PackedX128 px2;

    for (int b = b_lo; b < b_end; ++b) {
        const size_t brow = d.brow0 + static_cast<size_t>(b);
        const int kg = fc1_unit_width(d.kmix, d.dk, 2 * b + 0);
        const int ku = fc1_unit_width(d.kmix, d.dk, 2 * b + 1);
        const uint8_t* mw_g = d.exp_fc1 + d.piece_fc1_off[b];
        const uint8_t* mw_u = mw_g + static_cast<size_t>(kg) * UNIT_PER_BIT;

        float g_out[128] = {};
        float u_out[128] = {};
        float t16[16];

        // 1. FC1 gate + up across hb=0..15, warp=0..7
        for (int hb = 0; hb < EHB; ++hb) {
            const float bsg = f16_to_f32(d.v1b[(brow * 2 + 0) * EHB + hb]);
            const float bsu = f16_to_f32(d.v1b[(brow * 2 + 1) * EHB + hb]);
            const uint8_t* grpg = mw_g + static_cast<size_t>(hb) * (2048 * kg);
            const uint8_t* grpu = mw_u + static_cast<size_t>(hb) * (2048 * ku);
            const PackedX128& px = px_layer[hb];

            for (int warp = 0; warp < 8; ++warp) {
                const uint8_t* tg = grpg + static_cast<size_t>(warp) * ((kg == 4) ? 1024u : 256u);
                if (kg == 2)      mul1_tile16x128_dot_cpu<2, 2048>(tg, px, t16);
                else if (kg == 3) mul1_tile16x128_dot_cpu<3, 2048>(tg, px, t16);
                else              mul1_tile16x128_dot_cpu<4, 2048>(tg, px, t16);
                float* dst_g = g_out + warp * 16;
                for (int r = 0; r < 16; ++r) dst_g[r] = std::fma(t16[r], bsg, dst_g[r]);

                const uint8_t* tu = grpu + static_cast<size_t>(warp) * ((ku == 4) ? 1024u : 256u);
                if (ku == 2)      mul1_tile16x128_dot_cpu<2, 2048>(tu, px, t16);
                else if (ku == 3) mul1_tile16x128_dot_cpu<3, 2048>(tu, px, t16);
                else              mul1_tile16x128_dot_cpu<4, 2048>(tu, px, t16);
                float* dst_u = u_out + warp * 16;
                for (int r = 0; r < 16; ++r) dst_u[r] = std::fma(t16[r], bsu, dst_u[r]);
            }
        }

        // 2. had128 -> BF16-rounded SiTU-GLU(4, 25) -> su_d -> had128 -> FP16 round (matches GPU fc1_act)
        had128_cpu(g_out);
        had128_cpu(u_out);
        const uint16_t* sv_g = d.v2 + (brow * 3 + 0) * 128;
        const uint16_t* sv_u = d.v2 + (brow * 3 + 1) * 128;
        const uint16_t* su_d = d.v2 + (brow * 3 + 2) * 128;
        float va[128];
        for (int c = 0; c < 128; ++c) {
            const float gf = bf16_round_f32(g_out[c] * f16_to_f32(sv_g[c]));
            const float uf = bf16_round_f32(u_out[c] * f16_to_f32(sv_u[c]));
            const float act = bf16_round_f32(situ_mul_cpu(gf, uf));
            va[c] = act * f16_to_f32(su_d[c]);
        }
        had128_cpu(va);
        for (int c = 0; c < 128; ++c) va[c] = f16_round_f32(va[c]);
        pack_x128(va, px2);

        // 3. FC2 block b across hb=0..15, warp=0..7 (100% sequential read of exp_fc2[b]!)
        const uint8_t* mw_fc2 = d.exp_fc2 + d.piece_fc2_off[b];
        const size_t nt_stride = static_cast<size_t>(256 * d.dk);
        for (int hb = 0; hb < EHB; ++hb) {
            const float sc = f16_to_f32(d.v3b[brow * EHB + hb]);
            float* acc_hb = fc2_acc + hb * 128;
            for (int warp = 0; warp < 8; ++warp) {
                const int n_tile = hb * 8 + warp;
                const uint8_t* nt = mw_fc2 + static_cast<size_t>(n_tile) * nt_stride;
                if (d.dk == 2)      mul1_tile16x128_dot_cpu<2, 256>(nt, px2, t16);
                else if (d.dk == 3) mul1_tile16x128_dot_cpu<3, 256>(nt, px2, t16);
                else                mul1_tile16x128_dot_cpu<4, 256>(nt, px2, t16);
                float* dst = acc_hb + warp * 16;
                for (int r = 0; r < 16; ++r) dst[r] = std::fma(t16[r], sc, dst[r]);
            }
        }
    }
}

// Finalize fc2_acc[EH] -> had128 per 128-elem block -> * sg_out * topk_w -> out_2048[EH].
inline void finalize_expert_cpu(const CpuExpertDesc& d, float fc2_acc[EH]) {
    const float rw = d.topk_w;
    for (int hb = 0; hb < EHB; ++hb) {
        float* v = fc2_acc + hb * 128;
        had128_cpu(v);
        const uint16_t* sg = d.sg_out + hb * 128;
        float* dst = d.out_2048 + hb * 128;
        for (int c = 0; c < 128; ++c) dst[c] = v[c] * f16_to_f32(sg[c]) * rw;
    }
}

// Convenience single-threaded full expert evaluation (for unit tests & standalone callers).
inline void eval_cold_expert_cpu(const CpuExpertDesc& d, const uint16_t* x_had_r) {
    PackedX128 px_layer[EHB];
    pack_x_had_layer(x_had_r, px_layer);
    float fc2_acc[EH] = {};
    eval_expert_blocks_cpu(d, px_layer, 0, d.nb, fc2_acc);
    finalize_expert_cpu(d, fc2_acc);
}

// =================================================================================================
// Persistent Low-Latency Spin-Then-Park Thread Pool for Parallel Cold Expert Evaluation
// Supports block-level splitting so all cores stay 100% utilized even when n_experts < n_threads!
// =================================================================================================
struct CpuExpertPool {
    static constexpr int MAX_EXPS = 16;
    static constexpr int MAX_SPLITS = 16;

    struct BlockTask {
        int exp_idx;
        int b_lo;
        int b_hi;
        float* acc_dst;
    };

    std::vector<std::thread> workers;
    int n_total_threads = 1; // workers.size() + 1 (caller thread participates!)

    // Current batch state (written by caller before gen++, read by workers)
    const CpuExpertDesc* cur_exps = nullptr;
    const PackedX128*    cur_px = nullptr;
    const BlockTask*     cur_tasks = nullptr;
    int                  cur_n_tasks = 0;

    std::atomic<int>      next_idx{0};
    std::atomic<int>      done_cnt{0};
    std::atomic<uint64_t> gen{0};
    std::atomic<bool>     stop{false};
    std::atomic<int>      sleeping{0};
    std::mutex            mu;
    std::condition_variable cv;

    // Pre-allocated scratch (zero per-layer heap allocs!)
    PackedX128 px_buf[EHB];
    std::vector<BlockTask> tasks_buf;
    std::vector<float>     part_acc; // [MAX_EXPS * MAX_SPLITS * EH]
    const CpuExpertDesc*   async_exps = nullptr;
    int                    async_n_exps = 0;
    int                    async_splits = 1;
    bool                   async_in_flight = false;

    static inline void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }

    void start(int total_threads) {
        shutdown();
        mul1_tables(); // force one-time LUT init before first step
        n_total_threads = std::max(1, total_threads);
        part_acc.resize(static_cast<size_t>(MAX_EXPS) * MAX_SPLITS * EH, 0.0f);
        tasks_buf.reserve(MAX_EXPS * MAX_SPLITS);
        const int n_workers = n_total_threads - 1;
        if (n_workers <= 0) return;

        stop.store(false, std::memory_order_relaxed);
        workers.reserve(static_cast<size_t>(n_workers));
        for (int w = 0; w < n_workers; ++w) {
            workers.emplace_back([this]() {
                uint64_t seen = 0;
                for (;;) {
                    uint64_t g = gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        for (int spin = 0; spin < 6000; ++spin) {
                            g = gen.load(std::memory_order_acquire);
                            if (g != seen || stop.load(std::memory_order_relaxed)) break;
                            cpu_pause();
                        }
                        if (g == seen) {
                            if (stop.load(std::memory_order_acquire)) return;
                            sleeping.fetch_add(1, std::memory_order_seq_cst);
                            std::unique_lock<std::mutex> lk(mu);
                            cv.wait(lk, [&]() {
                                return gen.load(std::memory_order_seq_cst) != seen ||
                                       stop.load(std::memory_order_seq_cst);
                            });
                            lk.unlock();
                            sleeping.fetch_sub(1, std::memory_order_acq_rel);
                            g = gen.load(std::memory_order_acquire);
                        }
                    }
                    if (stop.load(std::memory_order_relaxed)) return;
                    seen = g;
                    const BlockTask* tasks = cur_tasks;
                    const int n = cur_n_tasks;
                    const CpuExpertDesc* exps = cur_exps;
                    const PackedX128* px = cur_px;
                    int completed = 0;
                    for (;;) {
                        const int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= n) break;
                        const auto& tk = tasks[idx];
                        std::memset(tk.acc_dst, 0, EH * sizeof(float));
                        eval_expert_blocks_cpu(exps[tk.exp_idx], px, tk.b_lo, tk.b_hi, tk.acc_dst);
                        ++completed;
                    }
                    if (completed > 0) done_cnt.fetch_add(completed, std::memory_order_release);
                }
            });
        }
    }

    // Launch `n_exps` (<= MAX_EXPS) cold experts asynchronously on the worker threads so the
    // caller thread can concurrently stage GPU-bound cold experts & enqueue PCIe H2D transfers!
    // Must be paired with `wait_async()`. If `workers.empty()`, evaluates synchronously in `wait_async()`.
    void launch_async(const CpuExpertDesc* exps, int n_exps, const uint16_t* x_had_r) {
        if (n_exps <= 0) { async_in_flight = false; return; }
        if (n_exps > MAX_EXPS) n_exps = MAX_EXPS;
        pack_x_had_layer(x_had_r, px_buf);

        const int active_w = std::max(1, static_cast<int>(workers.size()));
        int splits = 1;
        if (active_w > 1) {
            const int target_tasks = (n_exps < active_w) ? active_w : (2 * active_w);
            const int need = (target_tasks + n_exps - 1) / n_exps;
            if (need >= 16)      splits = 16;
            else if (need >= 8)  splits = 8;
            else if (need >= 4)  splits = 4;
            else if (need >= 2)  splits = 2;
        }

        tasks_buf.clear();
        for (int i = 0; i < n_exps; ++i) {
            const int nb = exps[i].nb;
            const int s_eff = std::min(splits, std::max(1, nb));
            const int per = (nb + s_eff - 1) / s_eff;
            for (int s = 0; s < s_eff; ++s) {
                const int b_lo = s * per;
                const int b_hi = std::min(nb, b_lo + per);
                if (b_lo >= b_hi) break;
                float* dst = part_acc.data() + static_cast<size_t>(i * MAX_SPLITS + s) * EH;
                tasks_buf.push_back({i, b_lo, b_hi, dst});
            }
        }

        async_exps = exps;
        async_n_exps = n_exps;
        async_splits = splits;
        async_in_flight = true;

        if (!workers.empty() && !tasks_buf.empty()) {
            cur_exps = exps;
            cur_px = px_buf;
            cur_tasks = tasks_buf.data();
            cur_n_tasks = static_cast<int>(tasks_buf.size());
            next_idx.store(0, std::memory_order_relaxed);
            done_cnt.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_seq_cst);
            if (sleeping.load(std::memory_order_seq_cst) > 0) {
                std::lock_guard<std::mutex> lk(mu);
                cv.notify_all();
            }
        }
    }

    // Help drain any remaining tasks, wait for workers, reduce splits, and finalize outputs.
    void wait_async() {
        if (!async_in_flight) return;
        const int n_tasks = static_cast<int>(tasks_buf.size());
        if (workers.empty()) {
            for (int t = 0; t < n_tasks; ++t) {
                const auto& tk = tasks_buf[static_cast<size_t>(t)];
                std::memset(tk.acc_dst, 0, EH * sizeof(float));
                eval_expert_blocks_cpu(async_exps[tk.exp_idx], px_buf, tk.b_lo, tk.b_hi, tk.acc_dst);
            }
        } else if (n_tasks > 0) {
            int completed = 0;
            for (;;) {
                const int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_tasks) break;
                const auto& tk = tasks_buf[static_cast<size_t>(idx)];
                std::memset(tk.acc_dst, 0, EH * sizeof(float));
                eval_expert_blocks_cpu(async_exps[tk.exp_idx], px_buf, tk.b_lo, tk.b_hi, tk.acc_dst);
                ++completed;
            }
            if (completed > 0) done_cnt.fetch_add(completed, std::memory_order_release);
            while (done_cnt.load(std::memory_order_acquire) < n_tasks) cpu_pause();
        }

        for (int i = 0; i < async_n_exps; ++i) {
            const int nb = async_exps[i].nb;
            const int s_eff = std::min(async_splits, std::max(1, nb));
            float* acc0 = part_acc.data() + static_cast<size_t>(i * MAX_SPLITS) * EH;
            int actual_s = 0;
            const int per = (nb + s_eff - 1) / s_eff;
            for (int s = 0; s < s_eff; ++s) {
                if (s * per >= nb) break;
                ++actual_s;
            }
            for (int s = 1; s < actual_s; ++s) {
                const float* acc_s = part_acc.data() + static_cast<size_t>(i * MAX_SPLITS + s) * EH;
                for (int c = 0; c < EH; ++c) acc0[c] += acc_s[c];
            }
            finalize_expert_cpu(async_exps[i], acc0);
        }
        async_in_flight = false;
    }

    // Evaluate `n_exps` (<= MAX_EXPS) cold experts synchronously in parallel across all threads in the pool.
    void run_batch(const CpuExpertDesc* exps, int n_exps, const uint16_t* x_had_r) {
        launch_async(exps, n_exps, x_had_r);
        wait_async();
    }

    void shutdown() {
        if (workers.empty()) return;
        stop.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu);
            cv.notify_all();
        }
        for (auto& th : workers) if (th.joinable()) th.join();
        workers.clear();
        n_total_threads = 1;
    }

    ~CpuExpertPool() { shutdown(); }
};

} // namespace ernie55cpu
