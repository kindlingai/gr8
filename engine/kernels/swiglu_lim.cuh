// The shared / routed expert's swiglu with the clamp before silu (gate <= L, up in [-L, L], L = swiglu_limit 10). One definition for two TUs:
// mxfp4.cuh (the rank TU: every fc1 kernel of the routed and the shared expert) and attn/ds41_sh_tc.cuh (the attention TU: k_sh_swiglu_q of
// the tensor-core shared expert). No other header needed.
#pragma once

namespace mx4
{
__device__ __forceinline__ float swiglu_lim(float gate, float up, float lim)
{
    gate = fminf(gate, lim);
    up = fminf(fmaxf(up, -lim), lim);
    return gate / (1.f + __expf(-gate)) * up;
}
}   // namespace mx4
