// Engram element maths shared by the device kernels (engram_dev.cuh) and the host tests: one definition, so the host test of these
// functions is a test of what the kernels compute. Reference: ref-official/inference/model.py ParallelEngramEmbedding.forward (dequant)
// and Engram.forward (gate), MIT. No FMA anywhere: the reference rounds every torch op to fp32 separately.
#pragma once
#include <cstdint>
#include <cstring>
#include "hash.h"

namespace ds41 { namespace engram {

ENGRAM_HD inline float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
ENGRAM_HD inline uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// E4M3 (finite-only "fn" format: bias 7, no inf, 0x7F/0xFF are NaN) -> float, exact
ENGRAM_HD inline float e4m3_to_f32(uint8_t b)
{
    const uint32_t sign = (uint32_t)(b >> 7) << 31, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return bits_f32(sign | 0x7FC00000u);
    if (e == 0) return bits_f32(sign | f32_bits((float)m * 0.001953125f));   // subnormal: m * 2^-9
    return bits_f32(sign | ((e - 7 + 127) << 23) | (m << 20));
}
// E8M0 scale byte -> float: 2^(s-127), 0xFF = NaN (torch float8_e8m0fnu)
ENGRAM_HD inline float e8m0_to_f32(uint8_t s)
{
    if (s == 255) return bits_f32(0x7FC00000u);
    if (s == 0) return bits_f32(0x00400000u);     // 2^-127, a float subnormal
    return bits_f32((uint32_t)s << 23);
}
// float -> bfloat16 bits, round to nearest even (torch .to(bfloat16)); NaN stays NaN
ENGRAM_HD inline uint16_t f32_to_bf16(float f)
{
    const uint32_t u = f32_bits(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
ENGRAM_HD inline float bf16_to_f32(uint16_t h) { return bits_f32((uint32_t)h << 16); }
ENGRAM_HD inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// one table element: bf16( float(e4m3 w) * float(e8m0 s) )   (values.float() * scales.float(), then .to(bfloat16))
ENGRAM_HD inline uint16_t dequant1(uint8_t w, uint8_t s) { return f32_to_bf16(e4m3_to_f32(w) * e8m0_to_f32(s)); }

constexpr int kHid = 5120;                        // dim
constexpr int kStreams = 4;                       // hc_mult
constexpr int kKvDim = kHid * (kStreams + 1);     // wkv out_features: 4 keys + 1 value = 25600
constexpr float kInvSqrtDim = 0.013975424859373686f;   // float(5120 ** -0.5)
constexpr float kGateClamp = 1e-6f;

// the gate of one (row, stream) from the three sums the kernel reduces: sum h^2, sum key^2, sum h*w*key (all fp32).
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps);  dot = sum * rstd * dim^-0.5;
//   gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
// M supplies M::rsqrt, M::sqrt, M::exp (std:: on the host, rsqrtf/sqrtf/expf on the device).
template <class M>
ENGRAM_HD inline float gate_from_sums(float sum_h2, float sum_k2, float sum_hwk, float eps)
{
    const float rstd = M::rsqrt(sum_h2 / (float)kHid + eps) * M::rsqrt(sum_k2 / (float)kHid + eps);
    const float dot = (sum_hwk * rstd) * kInvSqrtDim;
    float a = dot < 0.f ? -dot : dot;
    a = a < kGateClamp ? kGateClamp : a;
    const float s = M::sqrt(a);
    const float x = (f32_bits(dot) >> 31) ? -s : s;   // copysign(s, dot)
    return 1.f / (1.f + M::exp(-x));
}

}}  // namespace ds41::engram
