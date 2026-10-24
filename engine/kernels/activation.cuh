// Shared decode/prefill activation. The default GLM expression is unchanged.
#pragma once
#include <cmath>
#include "../model_geom.h"
#ifdef __CUDACC__
#define G6_ACT_HD __host__ __device__ __forceinline__
#else
#define G6_ACT_HD inline
#endif
G6_ACT_HD float silu_f(float x)
{
#ifdef __CUDA_ARCH__
    return x / (1.f + __expf(-x));
#else
    return x / (1.f + expf(-x));
#endif
}
#if G6_MODEL_FLASH
G6_ACT_HD float swiglu_f(float gate, float up)
{
    // Clamp BEFORE SiLU. Negative gate values have no lower limit.
    gate = fminf(gate, geom::SWIGLU_LIMIT);
    up = fminf(fmaxf(up, -geom::SWIGLU_LIMIT), geom::SWIGLU_LIMIT);
    return silu_f(gate) * up;
}
#else
// Preserve the GLM expression at the call site, including compiler operand scheduling.
#define swiglu_f(gate, up) (silu_f(gate) * (up))
#endif
#undef G6_ACT_HD
