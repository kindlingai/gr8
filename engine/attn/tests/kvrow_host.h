// Host codec of the Flash MLA latent row (attn/kvfmt.h), the arithmetic of flash_attn.cu append_kv / latent() for a
// normalized 512-d latent v (no build flag needs a GPU):
//   G6_FLASH_KV_FP8=0  1024 B: v[d] as BF16 (RN)
//   G6_FLASH_KV_FP8=1   528 B: per 128 dims s = max(amax / 448, FLT_MIN), code = E4M3(v / s) (RN, satfinite), value e4 * s
// Tests use it to fill latent caches in the selected format and as the storage reference.
#pragma once
#include "../kvfmt.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace kvrow
{
constexpr int LAT = 512;
static_assert(G6_MODEL_FLASH, "kvrow_host.h is the Flash latent row");
static_assert(g6kv::ROW == (G6_FLASH_KV_FP8 ? LAT + 4 * 4 : 2 * LAT), "kvfmt.h row size");

inline void encode(const float* v, uint8_t* row)
{
#if G6_FLASH_KV_FP8
    for (int g = 0; g < LAT / 128; ++g)
    {
        float amax = 0.f;
        for (int d = 128 * g; d < 128 * (g + 1); ++d) amax = fmaxf(amax, fabsf(v[d]));
        const float s = fmaxf(amax / 448.f, FLT_MIN);
        for (int d = 128 * g; d < 128 * (g + 1); ++d) row[d] = __nv_fp8_e4m3(v[d] / s).__x;
        memcpy(row + g6kv::SCALE_OFF + 4 * g, &s, 4);
    }
#else
    for (int d = 0; d < LAT; ++d) { const __nv_bfloat16 b = __float2bfloat16_rn(v[d]); memcpy(row + 2 * d, &b, 2); }
#endif
}

inline float decode(const uint8_t* row, int d)
{
#if G6_FLASH_KV_FP8
    __nv_fp8_e4m3 c; c.__x = row[d];
    float s; memcpy(&s, row + g6kv::SCALE_OFF + 4 * (d >> 7), 4);
    return (float)c * s;
#else
    __nv_bfloat16 b; memcpy(&b, row + 2 * d, 2);
    return __bfloat162float(b);
#endif
}
}
