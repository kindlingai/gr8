// MXFP8 activation quantizer for the W-row wkv of Engram: the official act_quant (ref-official/inference/kernel.py: blocks of 32, E8M0 scale
// by fast_round_scale, e4m3 codes) as one host-and-device code path, so tests/test_act_quant.cpp runs the lines the kernel runs. These are the
// same formulas as the attention code's k_act_quant (engine/attn/ds41_attn.cuh, act_quant_lane): per 32-element block, a = max(max |v|, 1e-4),
// e = ceil(log2(a * fp32(1/448))) taken from the float's bits, code = e4m3(v * 2^-e) saturating, scale byte = e + 127.
// The W-row wkv (engram_dev.cuh Engram::project_wide) quantizes with this and multiplies with the tensor-core entry ds41_gemm_mx_tc,
// which takes codes and scales already made. The attention code has no W-row entry for the quantizer, so this copy of 25 lines stands in.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_fp8.h>
#include "hash.h"   // ENGRAM_HD

namespace ds41 { namespace engram {

constexpr float kInv448 = 0.002232142857142857f;   // fp32(1/448): the attention code's INV448 (ds41_attn_types.h)

// the block exponent e of a 32-block whose largest magnitude is amax (NaN-free: fmaxf drops a NaN operand, as the kernel's warp max does)
ENGRAM_HD inline int aq_exp(float amax)
{
    const float a = fmaxf(amax, 1e-4f);
    float x = a * kInv448;
    uint32_t b;
    memcpy(&b, &x, 4);
    return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0);
}
// 2^e as a float, e in -126..127
ENGRAM_HD inline float aq_pow2(int e)
{
    const uint32_t b = (uint32_t)(e + 127) << 23;
    float f;
    memcpy(&f, &b, 4);
    return f;
}
// the e4m3 code of one value of a block with exponent e
ENGRAM_HD inline uint8_t aq_code(float v, int e) { return (uint8_t)__nv_cvt_float_to_fp8(v * aq_pow2(-e), __NV_SATFINITE, __NV_E4M3); }
// the E8M0 scale byte of a block with exponent e
ENGRAM_HD inline uint8_t aq_scale_byte(int e) { return (uint8_t)(e + 127); }

}}   // namespace ds41::engram
