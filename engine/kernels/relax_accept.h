// Relaxed greedy acceptance of DSpark drafts (GR8_DS_RELAXED_ACCEPT; DS41 verify steps). Not exact when on: greedy output
// changes. Off (the default) launches none of the code that uses this header. Same rule as SGLang's relaxed_accept.py
// (DSV41_RELAXED_ACCEPT): a draft d of a greedy row is accepted when it passes every rule given,
//   ratio:R   p(d) >= R * p(argmax)                 (R in (0, 1]; logit gap l_max - l_d <= ln(1/R))
//   topp:P    mass of the tokens with logit > l_d < P (d is in the top-p set at T = 1; P in (0, 1))
// Host + device: glue.cuh (k_head_local_rx / k_head_relax) and the host tests (engine/tests/test_ds41_dspark.cpp).
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define RELAX_HD __host__ __device__ __forceinline__
#else
#define RELAX_HD inline
#endif

namespace relax
{
struct Rule
{
    int on = 0;                   // 0 = exact acceptance (the usual head kernels)
    float log_ratio = -INFINITY;  // ln R; -inf = no ratio rule
    float topp = 0.f;             // P; 0 = no top-p rule (needs --top-p-cands > 0: the candidates carry the mass above l_d)
};
// ld: d's raw logit; mx: the row's max raw logit; z = sum exp(l - mx) over the vocabulary; above = sum exp(l - mx) over the
// known tokens with l > ld; complete: every token with l > ld is known (else the top-p rule refuses: never accepts on a guess)
RELAX_HD bool ok(const Rule& r, float ld, float mx, float above, float z, bool complete)
{
    if (!(ld > -INFINITY && ld < INFINITY)) return false;   // d's logit is -inf (masked) or NaN
    if (ld - mx < r.log_ratio) return false;
    if (r.topp > 0.f && !(complete && above < r.topp * z)) return false;
    return true;
}
}  // namespace relax
