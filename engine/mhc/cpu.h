// Host-only numerical oracle. H is an argument so tiny and production shapes
// use the same implementation. Runtime geometry comes from model_geom.h.
#pragma once
#include "math.h"
#include <cstddef>
#include <vector>
namespace flash_mhc {
inline void prepare_cpu(const float* streams, const float* fn, const float* base,
                        const float* scale, const float* norm, int M, int H, float eps,
                        float* pre, float* post, float* comb, float* x, float* normalized) {
    for (int m = 0; m < M; ++m) {
        const float* r = streams + (size_t)m * 4 * H;
        float sq = 0.f, mix[MIX] = {};
        for (int k = 0; k < 4*H; ++k) sq += r[k] * r[k];
        const float inv = 1.f / sqrtf(sq / (4*H) + eps);
        for (int o = 0; o < MIX; ++o)
            for (int k = 0; k < 4*H; ++k) mix[o] += (r[k] * inv) * fn[(size_t)o*4*H+k];
        float *p = pre+4*m, *q = post+4*m, *c = comb+16*m;
        coefficients(mix, base, scale, p, q, c);
        sq = 0.f;
        for (int h = 0; h < H; ++h) {
            float v = 0.f;
            for (int i = 0; i < 4; ++i) v += p[i] * r[i*H+h];
            x[(size_t)m*H+h] = v;
            sq += v*v;
        }
        const float ni = 1.f / sqrtf(sq/H + eps);
        for (int h = 0; h < H; ++h) normalized[(size_t)m*H+h] = x[(size_t)m*H+h] * ni * norm[h];
    }
}
inline void output_cpu(float* streams, const float* block, const float* post, const float* comb, int M, int H) {
    for (int m = 0; m < M; ++m) for (int h = 0; h < H; ++h) {
        float old[4];
        for (int j = 0; j < 4; ++j) old[j] = streams[((size_t)m*4+j)*H+h];
        for (int i = 0; i < 4; ++i) {
            float sum = 0.f;
            for (int j = 0; j < 4; ++j) sum += comb[m*16+j*4+i] * old[j];
            streams[((size_t)m*4+i)*H+h] = post[m*4+i] * block[(size_t)m*H+h] + sum;
        }
    }
}
inline void final_cpu(const float* streams, const float* norm, int M, int H, float eps, float* out) {
    for (int m = 0; m < M; ++m) {
        float sq = 0.f;
        for (int h = 0; h < H; ++h) {
            float v = 0.f;
            for (int j = 0; j < 4; ++j) v += streams[((size_t)m*4+j)*H+h];
            v *= .25f; out[(size_t)m*H+h] = v; sq += v*v;
        }
        const float inv = 1.f / sqrtf(sq/H+eps);
        for (int h = 0; h < H; ++h) out[(size_t)m*H+h] *= inv * norm[h];
    }
}
} // namespace flash_mhc
