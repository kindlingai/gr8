// Four-stream mHC arithmetic shared by the CPU oracle and CUDA coefficient kernel.
#pragma once
#include <cmath>
#include "../model_geom.h"
#ifdef __CUDACC__
#define MHC_HD __host__ __device__
#else
#define MHC_HD
#endif
namespace flash_mhc {
#if G6_MODEL_FLASH
constexpr int STREAMS = geom::HC, MIX = geom::HC_MIX, ITERS = geom::HC_ITERS;
constexpr float HC_EPS = geom::HC_EPS;
#else
// The host oracle can run in a non-Flash test environment.
constexpr int STREAMS = 4, MIX = 24, ITERS = 20;
constexpr float HC_EPS = 1e-6f;
#endif
static_assert(STREAMS == 4 && MIX == 24, "four-stream kernel");
MHC_HD inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// comb[source][destination]. The initial row softmax counts as the first row
// normalization; do one column normalization, then 19 row/column pairs.
MHC_HD inline void coefficients(const float* mix, const float* base, const float* scale,
                                float* pre, float* post, float* comb) {
    for (int i = 0; i < STREAMS; ++i) {
        pre[i] = sigmoid(mix[i] * scale[0] + base[i]) + HC_EPS;
        post[i] = 2.f * sigmoid(mix[4+i] * scale[1] + base[4+i]);
        float mx = -INFINITY, sum = 0.f;
        for (int j = 0; j < STREAMS; ++j) {
            const int k = 4*i+j;
            comb[k] = mix[8+k] * scale[2] + base[8+k];
            mx = fmaxf(mx, comb[k]);
        }
        for (int j = 0; j < STREAMS; ++j) {
            comb[4*i+j] = expf(comb[4*i+j] - mx);
            sum += comb[4*i+j];
        }
        for (int j = 0; j < STREAMS; ++j) comb[4*i+j] = comb[4*i+j] / sum + HC_EPS;
    }
    for (int iter = 0; iter < ITERS; ++iter) {
        if (iter) for (int i = 0; i < STREAMS; ++i) {
            float sum = 0.f;
            for (int j = 0; j < STREAMS; ++j) sum += comb[4*i+j];
            for (int j = 0; j < STREAMS; ++j) comb[4*i+j] /= sum + HC_EPS;
        }
        for (int j = 0; j < STREAMS; ++j) {
            float sum = 0.f;
            for (int i = 0; i < STREAMS; ++i) sum += comb[4*i+j];
            for (int i = 0; i < STREAMS; ++i) comb[4*i+j] /= sum + HC_EPS;
        }
    }
}
} // namespace flash_mhc
#undef MHC_HD
