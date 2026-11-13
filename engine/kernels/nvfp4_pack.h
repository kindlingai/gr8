// Host NVFP4 primitives and the packer of the record "nv4g". Our own code; no CUDA needed. The Rust twin is
// engine/loader-rs/src/nvfp4.rs; both produce the same bytes (engine/loader-rs/tests/nvfp4.rs).
//
// NVFP4 layout of one [N][K] matrix (K a multiple of 16):
//   packed [N][K / 2]  bytes, byte j of a row = k 2j in the low nibble, k 2j + 1 in the high nibble (E2M1: bit 3 sign,
//                      bits 2..0 -> 0, .5, 1, 1.5, 2, 3, 4, 6)
//   scales [N][K / 16] E4M3FN bytes, one per 16 k
//   outer  fp32 per matrix, stored directly or as amax / denominator
//   value(n, k) = e2m1(code) * (outer * e4m3(scale[n][k / 16]))
//
// Record nv4g: 16 rows x 64 k = 576 bytes (nine 64-byte bursts). Bytes 0 .. 512 are the code bytes in the lane / tile
// order of the MXFP4 record (mxfp4_pack.h); bytes 512 .. 576 are the 64 block scales (pack_scales_64).
//
// Quantization of an fp32 vector follows one of two rules:
//   Static6         block scale = e4m3((block amax / 6) / outer), outer = amax / 2688 unless fixed
//   FourOverSixMae  per block the lower L1 error of block amax / 6 and block amax / 4 (ties keep / 6),
//                   outer = amax / 1536 unless fixed
// All arithmetic is fp32 in a fixed order; build with -ffp-contract=off to keep it identical to the Rust twin.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

namespace nv4pack
{
constexpr int KG = 64, GR = 576, SOFF = 512;
constexpr float E2M1_MAX = 6.0f;

enum class ScaleRule { Static6, FourOverSixMae };
inline float outer_denominator(ScaleRule r) { return r == ScaleRule::Static6 ? 2688.0f : 1536.0f; }

// ---- E2M1
inline const float* e2m1_table() { static const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; return t; }
inline float e2m1(unsigned code) { const float m = e2m1_table()[code & 7]; return code & 8 ? -m : m; }

// fp32 -> E2M1 code, round to nearest even on the grid, saturating at 6. Magnitude 0 is code 0.
inline uint8_t f32_to_e2m1_rne(float v)
{
    const float a = std::fmin(std::fabs(v), E2M1_MAX);
    unsigned idx;
    if (a < 2.0f) idx = static_cast<unsigned>(std::nearbyintf(a * 2.0f));
    else if (a < 4.0f) idx = 4u + static_cast<unsigned>(std::nearbyintf(a - 2.0f));
    else idx = 6u + static_cast<unsigned>(std::nearbyintf((a - 4.0f) * 0.5f));
    return static_cast<uint8_t>((std::signbit(v) && idx != 0 ? 8u : 0u) | idx);
}

// ---- E4M3FN
inline float e4m3_to_f32(uint8_t b)
{
    const int e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = static_cast<float>(m) * (1.0f / 512.0f);
    else if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
    else v = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
    return b & 0x80 ? -v : v;
}

// fp32 -> E4M3FN byte, round to nearest even, saturating at 448. NaN gives 0x7f; a result of 0 is byte 0x00.
inline uint8_t f32_to_e4m3_rne(float v)
{
    if (std::isnan(v)) return 0x7f;
    const bool neg = std::signbit(v);
    double a = std::fabs(static_cast<double>(v));
    if (a > 448.0) a = 448.0;
    if (a == 0.0) return 0;
    int e = std::ilogb(a);
    if (e < -6) e = -6;
    const double step = std::ldexp(1.0, e - 3);
    a = std::nearbyint(a / step) * step;
    if (a > 448.0) a = 448.0;
    if (a == 0.0) return 0;
    const uint8_t s = neg ? 0x80 : 0;
    if (a < std::ldexp(1.0, -6)) return static_cast<uint8_t>(s | static_cast<uint8_t>(a * 512.0));
    const int e2 = std::ilogb(a);
    const unsigned m = static_cast<unsigned>(a / std::ldexp(1.0, e2) * 8.0) - 8u;
    return static_cast<uint8_t>(s | (static_cast<unsigned>(e2 + 7) << 3) | m);
}

// ---- outer scale conventions
struct OuterScale
{
    std::vector<float> v;       // direct scales, or amax when denominator != 0
    float denominator = 0.0f;   // 0 = v holds the scales themselves
    static OuterScale direct(std::vector<float> s) { return OuterScale{std::move(s), 0.0f}; }
    static OuterScale amax_denominator(std::vector<float> amax, float den) { return OuterScale{std::move(amax), den}; }
    float at(size_t e) const
    {
        const float x = v.size() == 1 ? v[0] : v[e];
        return denominator != 0.0f ? x / denominator : x;
    }
};

// ---- quantization
// One candidate of a 16-element block: target = the E2M1 magnitude the block amax maps to (6 or 4).
inline float block_candidate(const float* x, float b_max, float outer, float target, uint8_t& s, uint8_t q[16])
{
    s = outer == 0.0f ? uint8_t(0) : f32_to_e4m3_rne((b_max / target) / outer);
    const float sf = outer * e4m3_to_f32(s);
    float err = 0.0f;
    for (int j = 0; j < 16; ++j)
    {
        q[j] = sf != 0.0f ? f32_to_e2m1_rne(x[j] / sf) : uint8_t(0);
        const float d = e2m1(q[j]) * sf - x[j];
        err += std::fabs(d);
    }
    return err;
}

// Quantize a row-major [n][K] fp32 block under ONE outer scale: *fixed_outer, or amax of all n * K values over the
// rule's denominator. codes gets n * K / 2 bytes, scales n * K / 16. Returns the outer scale used.
inline float quantize(const float* x, size_t n, int K, ScaleRule rule, const float* fixed_outer, uint8_t* codes, uint8_t* scales)
{
    if (K % 16) throw std::runtime_error("nv4pack: K % 16");
    const size_t total = n * static_cast<size_t>(K);
    float amax = 0.0f;
    for (size_t i = 0; i < total; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    const float outer = fixed_outer ? *fixed_outer : amax / outer_denominator(rule);
    for (size_t b = 0; b < total / 16; ++b)
    {
        const float* blk = x + b * 16;
        float b_max = 0.0f;
        for (int j = 0; j < 16; ++j) b_max = std::fmax(b_max, std::fabs(blk[j]));
        uint8_t s, q[16];
        const float e6 = block_candidate(blk, b_max, outer, 6.0f, s, q);
        if (rule == ScaleRule::FourOverSixMae)
        {
            uint8_t s4, q4[16];
            const float e4 = block_candidate(blk, b_max, outer, 4.0f, s4, q4);
            if (e4 < e6) { s = s4; memcpy(q, q4, 16); }
        }
        scales[b] = s;
        for (int j = 0; j < 8; ++j) codes[b * 8 + j] = static_cast<uint8_t>(q[2 * j] | (q[2 * j + 1] << 4));
    }
    return outer;
}

// One vector of K values with its own outer scale (the per-token activation case).
inline float quantize_row(const float* x, int K, ScaleRule rule, uint8_t* codes, uint8_t* scales)
{
    return quantize(x, 1, K, rule, nullptr, codes, scales);
}

// Row-major dequantization, the definition of the format: nblk blocks of 16 -> out[nblk * 16].
inline void dequantize(const uint8_t* codes, const uint8_t* scales, size_t nblk, float outer, float* out)
{
    for (size_t b = 0; b < nblk; ++b)
    {
        const float sf = outer * e4m3_to_f32(scales[b]);
        for (int j = 0; j < 8; ++j)
        {
            const uint8_t c = codes[b * 8 + j];
            out[b * 16 + 2 * j] = e2m1(c & 15u) * sf;
            out[b * 16 + 2 * j + 1] = e2m1(c >> 4) * sf;
        }
    }
}

// ---- record nv4g
// The 64 scale bytes of a record: for r in 0..8, bytes r * 8 .. r * 8 + 4 = the 4 block scales of row r0 + r, bytes
// r * 8 + 4 .. r * 8 + 8 = those of row r0 + r + 8 (each an aligned u32). stride = scale bytes per source row, sb =
// first block-16 scale column.
inline void pack_scales_64(uint8_t* dst, const uint8_t* scales, size_t stride, int r0, int sb)
{
    for (int r = 0; r < 8; ++r)
    {
        memcpy(dst + r * 8, scales + static_cast<size_t>(r0 + r) * stride + sb, 4);
        memcpy(dst + r * 8 + 4, scales + static_cast<size_t>(r0 + r + 8) * stride + sb, 4);
    }
}

// Inverse: out[row 0..16][block 0..4].
inline void unpack_scales_64(const uint8_t* src, uint8_t out[16][4])
{
    for (int r = 0; r < 8; ++r)
    {
        memcpy(out[r], src + r * 8, 4);
        memcpy(out[r + 8], src + r * 8 + 4, 4);
    }
}

// One record: rows r0 .. r0 + 15 of the matrix, k k0 .. k0 + 63 (k0 % 64 == 0). K = row length in k of the SOURCE, so
// a column slice needs no copy.
inline void pack_group(uint8_t* dst, const uint8_t* packed, const uint8_t* scales, int K, int r0, int k0)
{
    const size_t pb = static_cast<size_t>(K) / 2, sb = static_cast<size_t>(K) / 16;
    for (int lane = 0; lane < 32; ++lane)
    {
        const int r = lane >> 2, c = lane & 3;
        for (int t = 0; t < 4; ++t)
        {
            const size_t j = static_cast<size_t>(k0 / 2 + t * 8 + c);
            uint8_t* o = dst + lane * 16 + t * 4;
            o[0] = packed[static_cast<size_t>(r0 + r) * pb + j];
            o[1] = packed[static_cast<size_t>(r0 + r + 8) * pb + j];
            o[2] = packed[static_cast<size_t>(r0 + r) * pb + j + 4];
            o[3] = packed[static_cast<size_t>(r0 + r + 8) * pb + j + 4];
        }
    }
    pack_scales_64(dst + SOFF, scales, sb, r0, k0 / 16);
}

// Columns [c0, c0 + Kn) of a [R][K] matrix -> [R / 16][Kn / 64][GR].
inline std::vector<uint8_t> pack_fc2(const uint8_t* packed, const uint8_t* scales, int R, int K, int c0, int Kn)
{
    if (R % 16 || K % 16 || c0 % KG || Kn % KG || c0 + Kn > K) throw std::runtime_error("nv4pack: R % 16, or the column slice");
    std::vector<uint8_t> out(static_cast<size_t>(R / 16) * (Kn / KG) * GR);
    for (int ot = 0; ot < R / 16; ++ot)
        for (int g = 0; g < Kn / KG; ++g)
            pack_group(out.data() + (static_cast<size_t>(ot) * (Kn / KG) + g) * GR, packed, scales, K, ot * 16, c0 + g * KG);
    return out;
}

// Any [R][K] matrix -> [R / 16][K / 64][GR].
inline std::vector<uint8_t> pack_matrix(const uint8_t* packed, const uint8_t* scales, int R, int K)
{
    return pack_fc2(packed, scales, R, K, 0, K);
}

// fc1: gate and up [R][K] -> [R / 16][K / 64][gate GR | up GR]
inline std::vector<uint8_t> pack_fc1(const uint8_t* gp, const uint8_t* gs, const uint8_t* up, const uint8_t* us, int R, int K)
{
    if (R % 16 || K % KG) throw std::runtime_error("nv4pack: R % 16 or K % 64");
    std::vector<uint8_t> out(static_cast<size_t>(R / 16) * (K / KG) * 2 * GR);
    for (int ch = 0; ch < R / 16; ++ch)
        for (int g = 0; g < K / KG; ++g)
        {
            uint8_t* d = out.data() + (static_cast<size_t>(ch) * (K / KG) + g) * 2 * GR;
            pack_group(d, gp, gs, K, ch * 16, g * KG);
            pack_group(d + GR, up, us, K, ch * 16, g * KG);
        }
    return out;
}

// ---- host decode, the layout oracle: out[16][64] = the dequantized weights of the record at g, read back from the
// lane / tile / byte positions the device reads.
inline void decode_group(const uint8_t* g, float outer, float (&out)[16][64])
{
    uint8_t sc[16][4];
    unpack_scales_64(g + SOFF, sc);
    for (int lane = 0; lane < 32; ++lane)
    {
        const int r = lane >> 2, c = lane & 3;
        for (int t = 0; t < 4; ++t)
        {
            const uint8_t* w = g + lane * 16 + t * 4;
            const float sl = outer * e4m3_to_f32(sc[r][t]), sh = outer * e4m3_to_f32(sc[r + 8][t]);
            const int kb = t * 16 + 2 * c;
            out[r][kb] = e2m1(w[0] & 15u) * sl;          out[r][kb + 1] = e2m1(w[0] >> 4) * sl;
            out[r + 8][kb] = e2m1(w[1] & 15u) * sh;      out[r + 8][kb + 1] = e2m1(w[1] >> 4) * sh;
            out[r][kb + 8] = e2m1(w[2] & 15u) * sl;      out[r][kb + 9] = e2m1(w[2] >> 4) * sl;
            out[r + 8][kb + 8] = e2m1(w[3] & 15u) * sh;  out[r + 8][kb + 9] = e2m1(w[3] >> 4) * sh;
        }
    }
}
}   // namespace nv4pack
