// Shared host/device arithmetic for the two-part online-softmax merge.
#pragma once
#include <math.h>

#if defined(__CUDACC__)
#define DS41_SPLIT_HD __host__ __device__
#else
#define DS41_SPLIT_HD
#endif

namespace ds41a {
DS41_SPLIT_HD inline bool split_eligible(int rows, int window_slots, int compressed_slots)
{
    return rows >= 512 && ((window_slots + compressed_slots + 63) >> 6) >= 4;
}
struct SplitMergeWeights { float a, b, denom; };
DS41_SPLIT_HD inline SplitMergeWeights split_merge_weights(float m0, float sum0, float m1, float sum1, float sink)
{
    const float m = fmaxf(m0, m1);
    const float a = expf(m0 - m), b = expf(m1 - m);
    return {a, b, sum0 * a + sum1 * b + expf(sink - m)};
}
DS41_SPLIT_HD inline float split_merge_value(float acc0, float acc1, SplitMergeWeights w)
{
    return (acc0 * w.a + acc1 * w.b) / w.denom;
}
}  // namespace ds41a
#undef DS41_SPLIT_HD
