// Host oracle for the DeepSeek-V4.1 hyper-connection trunk (the SHIFTED variant: 1017 has no hc_head tensors).
// engine/mhc/ds_mhc.cuh is the device twin. The coefficient math is the Flash one (flash_mhc::coefficients), so the
// differences from engine/mhc/cpu.h are only the ones that matter:
//   * pre is CARRIED one sub-layer: the block input collapse uses the pre computed at the PREVIOUS sub-layer
//     (one-hot [1,0,0,0] at the start), the mixes of this sub-layer only produce the NEXT pre, post and comb;
//   * the mixes are dot(flat, fn) * rsqrt(mean(flat^2) + eps) (DeepGEMM mhc_reference order), fn is F32;
//   * the trunk is BF16-valued after every update, and the collapse is rounded to BF16 before the block RMSNorm;
//   * the end of the model applies the last carried pre (no 0.25 mean, no hc_head), then RMSNorm * norm.weight.
// Every float array below holds BF16-representable values wherever the DeepInfer tensor is BF16.
#pragma once
#include "math.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace ds_mhc {
inline float bf16r(float x) {   // round to nearest even BF16, returned as float
    uint32_t u; std::memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;   // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r; std::memcpy(&r, &u, 4);
    return r;
}
// streams [M,4,H] = 4 copies of the embedding row; pre = one-hot.
inline void embed_cpu(const int* tok, const float* embed, int M, int H, float* streams, float* pre) {
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < 4; ++j)
            for (int h = 0; h < H; ++h) streams[((size_t)m*4 + j)*H + h] = embed[(size_t)tok[m]*H + h];
        for (int j = 0; j < 4; ++j) pre[m*4 + j] = j == 0 ? 1.f : 0.f;
    }
}
// One sub-layer's input. pre is read (collapse) and then replaced by this row's next pre.
// y [M,H] = BF16(RMSNorm(BF16(sum_i pre_prev[i]*T[i])) * norm).
inline void prepare_cpu(const float* streams, float* pre, const float* fn, const float* base, const float* scale,
                        const float* norm, int M, int H, float eps, float* post, float* comb, float* y) {
    for (int m = 0; m < M; ++m) {
        const float* r = streams + (size_t)m * 4 * H;
        double sq = 0.0;
        for (int k = 0; k < 4*H; ++k) sq += (double)r[k] * r[k];
        const float inv = (float)(1.0 / std::sqrt(sq / (4.0*H) + eps));
        float mix[flash_mhc::MIX];
        for (int o = 0; o < flash_mhc::MIX; ++o) {
            double d = 0.0;
            for (int k = 0; k < 4*H; ++k) d += (double)r[k] * fn[(size_t)o*4*H + k];
            mix[o] = (float)d * inv;
        }
        float old[4] = {pre[m*4], pre[m*4+1], pre[m*4+2], pre[m*4+3]};
        flash_mhc::coefficients(mix, base, scale, pre + 4*m, post + 4*m, comb + 16*m);
        double s2 = 0.0;
        float* yr = y + (size_t)m * H;                      // y doubles as the collapse buffer
        for (int h = 0; h < H; ++h) {
            float v = 0.f;
            for (int i = 0; i < 4; ++i) v += old[i] * r[(size_t)i*H + h];
            v = bf16r(v);
            yr[h] = v; s2 += (double)v * v;
        }
        const float ni = (float)(1.0 / std::sqrt(s2 / H + eps));
        for (int h = 0; h < H; ++h) yr[h] = bf16r(yr[h] * ni * norm[h]);
    }
}
// T_new[j] = BF16(post[j]*out + sum_i comb[i][j]*T[i]); out is rounded to BF16 first when round_y (a BF16 module output).
inline void consume_cpu(float* streams, const float* block, const float* post, const float* comb, int M, int H,
                        bool round_y = true) {
    for (int m = 0; m < M; ++m) for (int h = 0; h < H; ++h) {
        float old[4];
        for (int j = 0; j < 4; ++j) old[j] = streams[((size_t)m*4 + j)*H + h];
        const float y = round_y ? bf16r(block[(size_t)m*H + h]) : block[(size_t)m*H + h];
        for (int i = 0; i < 4; ++i) {
            float sum = 0.f;
            for (int j = 0; j < 4; ++j) sum += comb[m*16 + j*4 + i] * old[j];
            streams[((size_t)m*4 + i)*H + h] = bf16r(post[m*4 + i] * y + sum);
        }
    }
}
// After the last consume: x = BF16(sum_i pre[i]*T[i]); out = BF16(RMSNorm(x) * norm). pre = the last sub-layer's next pre.
inline void final_cpu(const float* streams, const float* pre, const float* norm, int M, int H, float eps, float* out,
                      float* xpre = nullptr) {
    for (int m = 0; m < M; ++m) {
        double sq = 0.0;
        float* o = out + (size_t)m * H;
        for (int h = 0; h < H; ++h) {
            float v = 0.f;
            for (int i = 0; i < 4; ++i) v += pre[m*4 + i] * streams[((size_t)m*4 + i)*H + h];
            v = bf16r(v); o[h] = v; sq += (double)v * v;
            if (xpre) xpre[(size_t)m * H + h] = v;
        }
        const float inv = (float)(1.0 / std::sqrt(sq / H + eps));
        for (int h = 0; h < H; ++h) o[h] = bf16r(o[h] * inv * norm[h]);
    }
}
} // namespace ds_mhc
