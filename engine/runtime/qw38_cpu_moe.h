// Qwen3.8-Flash-Next Hybrid CPU AVX-512 BF16 K3 (TR3_PMCGX, E4M3) Cold-Expert Offload Engine.
//
// Runs cold routed experts on host DDR5 RAM using Zen 5 AVX-512 + AVX512_BF16 (vdpbf16ps) in parallel
// with GPU VRAM-hot experts + PCIe UVA cold experts.
// Zero extra host RAM: reads the exact same wide-reordered K3 pinned host buffers (`h_fc1`, `h_fc2`)
// already resident in StrataExpertCache, synchronized with the GPU via a fixed-address mapped pinned
// control block (`CpuMoeBridge`) so it is 100% CUDA Graph compatible and zero-stall on cache hits.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include "../model_geom_qw38.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace qw38rt {

// Fixed-address mapped pinned GPU <-> CPU bridge (cudaHostAllocMapped).
// All GPU kernels write/read through its device pointer; CPU worker pool polls `go_seq` directly.
struct CpuMoeBridge {
    static constexpr int MAX_M = 8;
    static constexpr int MAX_ROUTES = MAX_M * geom::TOPK; // 80
    static constexpr int MAX_LAYERS = geom::L_MAIN;       // 48

    // Cache-line aligned doorbells
    alignas(64) volatile uint32_t go_seq;        // GPU stores monotonic seq when cpu_n_groups > 0
    alignas(64) volatile uint32_t done_seq;      // CPU stores same seq after cpu_fc2_part is visible
    alignas(64) volatile int cur_layer;          // layer index 0..47
    volatile int cur_M;                          // active batch size 1..8
    volatile int cpu_n_groups;                   // number of cold expert groups assigned to CPU (0..80)
    volatile uint32_t step_counter;              // monotonic GPU-maintained dispatch id (CUDA Graph safe!)

    alignas(64) uint8_t route_on_cpu[MAX_ROUTES];
    alignas(64) int cpu_grp_exp[MAX_ROUTES];
    alignas(64) int cpu_grp_cnt[MAX_ROUTES];
    alignas(64) int cpu_grp_routes[MAX_ROUTES * 8];
    alignas(64) float topk_w[MAX_ROUTES];
    alignas(64) uint16_t x_had[MAX_M * geom::HID];            // FP16 [M, 2560]
    alignas(64) float cpu_fc2_part[MAX_ROUTES * geom::HID];   // FP32 [80, 2560] = 800 KB
};

struct CpuMoeLayerRef {
    const uint8_t* h_fc1 = nullptr;
    const uint8_t* h_fc2 = nullptr;
    const long long* h_tab = nullptr;   // [rows * 6]
    const long long* h_kmix = nullptr;  // [rows] or nullptr
    const uint16_t* h_sg = nullptr;     // [2 * 2560] FP16
    const uint16_t* h_v1b = nullptr;    // [n_blks * 2 * 20] FP16
    const uint16_t* h_v2 = nullptr;     // [n_blks * 3 * 128] FP16
    const uint16_t* h_v3b = nullptr;    // [n_blks * 20] FP16
    int eb = 5;
};

namespace cpumoe_detail {

static constexpr float SQ128_F = 0.08838834764831845f; // 1 / sqrt(128)

inline float fp16_bits_to_f32(uint16_t h) {
#if defined(__x86_64__) && defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t s = (uint32_t(h) & 0x8000u) << 16;
    const uint32_t e = (uint32_t(h) >> 10) & 0x1fu;
    const uint32_t m = uint32_t(h) & 0x3ffu;
    uint32_t u = 0;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            float f = float(m) * (1.0f / 16777216.0f);
            return s ? -f : f;
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &u, 4);
    return out;
#endif
}

inline void had128_cpu(float* v) {
    for (int len = 1; len < 128; len <<= 1) {
        const int step = len << 1;
        for (int i = 0; i < 128; i += step) {
            for (int j = 0; j < len; ++j) {
                const float a = v[i + j];
                const float b = v[i + j + len];
                v[i + j] = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) v[i] *= SQ128_F;
}

inline float silu_cpu(float x) {
    return x / (1.0f + std::exp(-x));
}

#if defined(__x86_64__) || defined(_M_X64)
#define QW38_AVX512_ATTR __attribute__((always_inline, target("avx512f,avx512bw,avx512vl,f16c,avx512bf16,avx512vbmi2"), optimize("O3")))
#define QW38_AVX512_FN   __attribute__((target("avx512f,avx512bw,avx512vl,f16c,avx512bf16,avx512vbmi2"), optimize("O3")))

// Bit-exact AVX-512 128-point normalized Walsh-Hadamard transform (8 ZMMs, zero scalar loops!).
QW38_AVX512_ATTR
static inline void had128_avx512(float* __restrict__ v) {
    __m512 r[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = _mm512_loadu_ps(v + i * 16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __m512 t;
        t = _mm512_permute_ps(r[i], 0xB1);
        r[i] = _mm512_mask_add_ps(_mm512_sub_ps(t, r[i]), 0x5555, r[i], t);
        t = _mm512_permute_ps(r[i], 0x4E);
        r[i] = _mm512_mask_add_ps(_mm512_sub_ps(t, r[i]), 0x3333, r[i], t);
        t = _mm512_shuffle_f32x4(r[i], r[i], 0xB1);
        r[i] = _mm512_mask_add_ps(_mm512_sub_ps(t, r[i]), 0x0F0F, r[i], t);
        t = _mm512_shuffle_f32x4(r[i], r[i], 0x4E);
        r[i] = _mm512_mask_add_ps(_mm512_sub_ps(t, r[i]), 0x00FF, r[i], t);
    }
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        __m512 a = r[i], b = r[i + 1];
        r[i] = _mm512_add_ps(a, b);
        r[i + 1] = _mm512_sub_ps(a, b);
    }
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        for (int j = 0; j < 2; ++j) {
            __m512 a = r[i + j], b = r[i + j + 2];
            r[i + j] = _mm512_add_ps(a, b);
            r[i + j + 2] = _mm512_sub_ps(a, b);
        }
    }
    const __m512 sq = _mm512_set1_ps(SQ128_F);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        __m512 a = r[j], b = r[j + 4];
        _mm512_storeu_ps(v + j * 16,       _mm512_mul_ps(_mm512_add_ps(a, b), sq));
        _mm512_storeu_ps(v + (j + 4) * 16, _mm512_mul_ps(_mm512_sub_ps(a, b), sq));
    }
}

// Decode 16 20-bit TR3_PMCGX states -> packed interleaved BF16 (__m512bh).
// Uses vpternlogd (0x6A) + lo256/hi256 + fast E4M3 round-to-nearest (+0x80000 & 0xfff00000, rel_rmse=0.0033) + vcvtne2ps2bf16.
QW38_AVX512_ATTR
static inline __m512bh pmcgx_decode16_bf16(__m512i w) {
    const __m512i c0 = _mm512_set1_epi32(0xCBAC1FEDu);
    const __m512i c1 = _mm512_set1_epi32(0xBCA14C51u);
    const __m512i mk = _mm512_set1_epi32(0x8fff8fffu);
    const __m512i xr = _mm512_set1_epi32(0x3b603b60u);
    const __m512i x0 = _mm512_ternarylogic_epi32(_mm512_mullo_epi32(w, c0), mk, xr, 0x6A);
    const __m512i x1 = _mm512_ternarylogic_epi32(_mm512_mullo_epi32(w, c1), mk, xr, 0x6A);
    const __m512i x1s = _mm512_rol_epi32(x1, 16);
    const __m512 ra = _mm512_add_ps(_mm512_cvtph_ps(_mm512_castsi512_si256(x0)),
                                    _mm512_cvtph_ps(_mm512_castsi512_si256(x1s)));
    const __m512 rb = _mm512_add_ps(_mm512_cvtph_ps(_mm512_extracti64x4_epi64(x0, 1)),
                                    _mm512_cvtph_ps(_mm512_extracti64x4_epi64(x1s, 1)));
    const __m512i rbias = _mm512_set1_epi32(0x00080000u);
    const __m512i emask = _mm512_set1_epi32(0xfff00000u);
    const __m512 a_e4 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_add_epi32(_mm512_castps_si512(ra), rbias), emask));
    const __m512 b_e4 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_add_epi32(_mm512_castps_si512(rb), rbias), emask));
    return _mm512_cvtne2ps_pbh(b_e4, a_e4);
}

// Fast horizontal sum of each 4-lane group (col=0..3) in a 16-lane __m512 -> 4 floats (rows 0..3)
QW38_AVX512_ATTR
static inline void hsum4x4_store(__m512 v, float* dst4) {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    dst4[0] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    dst4[1] = tmp[4] + tmp[5] + tmp[6] + tmp[7];
    dst4[2] = tmp[8] + tmp[9] + tmp[10] + tmp[11];
    dst4[3] = tmp[12] + tmp[13] + tmp[14] + tmp[15];
}

QW38_AVX512_ATTR
static inline void hsum4x4_add(__m512 v, float* dst4, float scale) {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    dst4[0] += (tmp[0] + tmp[1] + tmp[2] + tmp[3]) * scale;
    dst4[1] += (tmp[4] + tmp[5] + tmp[6] + tmp[7]) * scale;
    dst4[2] += (tmp[8] + tmp[9] + tmp[10] + tmp[11]) * scale;
    dst4[3] += (tmp[12] + tmp[13] + tmp[14] + tmp[15]) * scale;
}

// Pack 128 FP32 inputs -> broadcast BF16 pair vectors x0[8] (k=0..7) and x1[8] (k=8..15)
QW38_AVX512_ATTR
static inline void pack_x128_f32_to_bf16(const float* x128, __m512i* x0, __m512i* x1) {
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const __m512 v = _mm512_loadu_ps(x128 + p * 16);
        const __m256i h = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v));
        x0[p] = _mm512_broadcast_i32x4(_mm256_castsi256_si128(h));
        x1[p] = _mm512_broadcast_i32x4(_mm256_extracti128_si256(h, 1));
    }
}

// Pack 128 FP16 inputs directly -> broadcast BF16 pair vectors x0[8] and x1[8]
QW38_AVX512_FN
static void pack_x128_f16_to_bf16(const uint16_t* x128, __m512i* x0, __m512i* x1) {
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x128 + p * 16)));
        const __m256i h = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v));
        x0[p] = _mm512_broadcast_i32x4(_mm256_castsi256_si128(h));
        x1[p] = _mm512_broadcast_i32x4(_mm256_extracti128_si256(h, 1));
    }
}

// Fully-inlined, ZERO-call, register-only half-step for cnt == 1 using AVX512VBMI2 vpshldd!
QW38_AVX512_ATTR
static inline void k3_accum_half_m1(__m512i ch, __m512i pl, __m512i mask20,
                                    __m512i xp0, __m512i xp1,
                                    __m512& a0, __m512& a8) {
    const __m512bh x0b = reinterpret_cast<__m512bh>(xp0);
    const __m512bh x1b = reinterpret_cast<__m512bh>(xp1);
    const __m512i ch8 = _mm512_slli_epi32(ch, 8);
    const __m512i s0 = _mm512_and_si512(_mm512_shldi_epi32(pl, ch8, 6),  mask20);
    a0 = _mm512_dpbf16_ps(a0, pmcgx_decode16_bf16(s0), x0b);
    const __m512i s1 = _mm512_and_si512(_mm512_shldi_epi32(pl, ch8, 12), mask20);
    a0 = _mm512_dpbf16_ps(a0, pmcgx_decode16_bf16(s1), x1b);
    const __m512i s2 = _mm512_and_si512(_mm512_shldi_epi32(pl, ch8, 18), mask20);
    a8 = _mm512_dpbf16_ps(a8, pmcgx_decode16_bf16(s2), x0b);
    const __m512i s3 = _mm512_and_si512(ch, mask20);
    a8 = _mm512_dpbf16_ps(a8, pmcgx_decode16_bf16(s3), x1b);
}

// Fully-inlined, ZERO-call step for cnt == 1, staggered plane loads (max 8 live weight ZMMs -> ZERO spills!).
QW38_AVX512_ATTR
static inline void k3_step_p_m1(__m512i r0, __m512i r1, __m512i r2, __m512i r3,
                                __m512i idx_even, __m512i rot_prev, __m512i mask24, __m512i mask20,
                                __m512i xp0, __m512i xp1,
                                __m512& alo0, __m512& alo8, __m512& ahi0, __m512& ahi8) {
    const __m512i ch_lo = _mm512_and_si512(_mm512_permutex2var_epi32(r0, idx_even, r1), mask24);
    const __m512i ch_hi = _mm512_and_si512(_mm512_permutex2var_epi32(r2, idx_even, r3), mask24);
    const __m512i pl_lo = _mm512_mask_permutexvar_epi32(_mm512_permutexvar_epi32(rot_prev, ch_lo), 0x0001, rot_prev, ch_hi);
    const __m512i pl_hi = _mm512_mask_permutexvar_epi32(_mm512_permutexvar_epi32(rot_prev, ch_hi), 0x0001, rot_prev, ch_lo);
    k3_accum_half_m1(ch_lo, pl_lo, mask20, xp0, xp1, alo0, alo8);
    k3_accum_half_m1(ch_hi, pl_hi, mask20, xp0, xp1, ahi0, ahi8);
}

QW38_AVX512_ATTR
static inline void k3_wide_wg_m1(const uint8_t* __restrict__ p0,
                                 const __m512i* __restrict__ x0,
                                 const __m512i* __restrict__ x1,
                                 __m512& alo0, __m512& alo8, __m512& ahi0, __m512& ahi8,
                                 int plane_stride) {
    const __m512i mask24 = _mm512_set1_epi32(0xffffffu);
    const __m512i mask20 = _mm512_set1_epi32(0xfffffu);
    const __m512i rot_prev = _mm512_setr_epi32(15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
    const __m512i idx_even = _mm512_setr_epi32(0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);

    // Load plane 0 -> execute p=0, 1
    const __m512i w0_0 = _mm512_loadu_si512(p0 + 0);
    const __m512i w0_1 = _mm512_loadu_si512(p0 + 64);
    const __m512i w0_2 = _mm512_loadu_si512(p0 + 128);
    const __m512i w0_3 = _mm512_loadu_si512(p0 + 192);
    k3_step_p_m1(w0_0, w0_1, w0_2, w0_3, idx_even, rot_prev, mask24, mask20, x0[0], x1[0], alo0, alo8, ahi0, ahi8);
    k3_step_p_m1(_mm512_srli_epi64(w0_0, 24), _mm512_srli_epi64(w0_1, 24),
                 _mm512_srli_epi64(w0_2, 24), _mm512_srli_epi64(w0_3, 24),
                 idx_even, rot_prev, mask24, mask20, x0[1], x1[1], alo0, alo8, ahi0, ahi8);

    // Load plane 1 -> execute p=2, 3, 4 (vpshrdq fuses srli+slli+or!)
    const uint8_t* p1 = p0 + plane_stride;
    const __m512i w1_0 = _mm512_loadu_si512(p1 + 0);
    const __m512i w1_1 = _mm512_loadu_si512(p1 + 64);
    const __m512i w1_2 = _mm512_loadu_si512(p1 + 128);
    const __m512i w1_3 = _mm512_loadu_si512(p1 + 192);
    k3_step_p_m1(_mm512_shrdi_epi64(w0_0, w1_0, 48), _mm512_shrdi_epi64(w0_1, w1_1, 48),
                 _mm512_shrdi_epi64(w0_2, w1_2, 48), _mm512_shrdi_epi64(w0_3, w1_3, 48),
                 idx_even, rot_prev, mask24, mask20, x0[2], x1[2], alo0, alo8, ahi0, ahi8);
    k3_step_p_m1(_mm512_srli_epi64(w1_0, 8), _mm512_srli_epi64(w1_1, 8),
                 _mm512_srli_epi64(w1_2, 8), _mm512_srli_epi64(w1_3, 8),
                 idx_even, rot_prev, mask24, mask20, x0[3], x1[3], alo0, alo8, ahi0, ahi8);
    k3_step_p_m1(_mm512_srli_epi64(w1_0, 32), _mm512_srli_epi64(w1_1, 32),
                 _mm512_srli_epi64(w1_2, 32), _mm512_srli_epi64(w1_3, 32),
                 idx_even, rot_prev, mask24, mask20, x0[4], x1[4], alo0, alo8, ahi0, ahi8);

    // Load plane 2 -> execute p=5, 6, 7
    const uint8_t* p2 = p1 + plane_stride;
    const __m512i w2_0 = _mm512_loadu_si512(p2 + 0);
    const __m512i w2_1 = _mm512_loadu_si512(p2 + 64);
    const __m512i w2_2 = _mm512_loadu_si512(p2 + 128);
    const __m512i w2_3 = _mm512_loadu_si512(p2 + 192);
    k3_step_p_m1(_mm512_shrdi_epi64(w1_0, w2_0, 56), _mm512_shrdi_epi64(w1_1, w2_1, 56),
                 _mm512_shrdi_epi64(w1_2, w2_2, 56), _mm512_shrdi_epi64(w1_3, w2_3, 56),
                 idx_even, rot_prev, mask24, mask20, x0[5], x1[5], alo0, alo8, ahi0, ahi8);
    k3_step_p_m1(_mm512_srli_epi64(w2_0, 16), _mm512_srli_epi64(w2_1, 16),
                 _mm512_srli_epi64(w2_2, 16), _mm512_srli_epi64(w2_3, 16),
                 idx_even, rot_prev, mask24, mask20, x0[6], x1[6], alo0, alo8, ahi0, ahi8);
    k3_step_p_m1(_mm512_srli_epi64(w2_0, 40), _mm512_srli_epi64(w2_1, 40),
                 _mm512_srli_epi64(w2_2, 40), _mm512_srli_epi64(w2_3, 40),
                 idx_even, rot_prev, mask24, mask20, x0[7], x1[7], alo0, alo8, ahi0, ahi8);
}

// Multi-token (cnt = 2..8) fully-inlined half-step using AVX512VBMI2 vpshldd!
template <int MAX_CNT>
QW38_AVX512_ATTR
static inline void k3_accum_half_multi(__m512i ch, __m512i pl, __m512i mask20,
                                       const __m512i* const* x0_ptrs, const __m512i* const* x1_ptrs,
                                       int p, int cnt, __m512* a0, __m512* a8) {
    const __m512i ch8 = _mm512_slli_epi32(ch, 8);
    const __m512bh w0 = pmcgx_decode16_bf16(_mm512_and_si512(_mm512_shldi_epi32(pl, ch8, 6),  mask20));
    #pragma unroll
    for (int t = 0; t < MAX_CNT; ++t) { if (t < cnt) a0[t] = _mm512_dpbf16_ps(a0[t], w0, reinterpret_cast<__m512bh>(x0_ptrs[t][p])); }
    const __m512bh w1 = pmcgx_decode16_bf16(_mm512_and_si512(_mm512_shldi_epi32(pl, ch8, 12), mask20));
    #pragma unroll
    for (int t = 0; t < MAX_CNT; ++t) { if (t < cnt) a0[t] = _mm512_dpbf16_ps(a0[t], w1, reinterpret_cast<__m512bh>(x1_ptrs[t][p])); }
    const __m512bh w2 = pmcgx_decode16_bf16(_mm512_and_si512(_mm512_shldi_epi32(pl, ch8, 18), mask20));
    #pragma unroll
    for (int t = 0; t < MAX_CNT; ++t) { if (t < cnt) a8[t] = _mm512_dpbf16_ps(a8[t], w2, reinterpret_cast<__m512bh>(x0_ptrs[t][p])); }
    const __m512bh w3 = pmcgx_decode16_bf16(_mm512_and_si512(ch, mask20));
    #pragma unroll
    for (int t = 0; t < MAX_CNT; ++t) { if (t < cnt) a8[t] = _mm512_dpbf16_ps(a8[t], w3, reinterpret_cast<__m512bh>(x1_ptrs[t][p])); }
}

template <int MAX_CNT>
QW38_AVX512_ATTR
static inline void k3_step_p_multi(__m512i r0, __m512i r1, __m512i r2, __m512i r3,
                                   __m512i idx_even, __m512i rot_prev, __m512i mask24, __m512i mask20,
                                   const __m512i* const* x0_ptrs, const __m512i* const* x1_ptrs,
                                   int p, int cnt,
                                   __m512* alo0, __m512* alo8, __m512* ahi0, __m512* ahi8) {
    const __m512i ch_lo = _mm512_and_si512(_mm512_permutex2var_epi32(r0, idx_even, r1), mask24);
    const __m512i ch_hi = _mm512_and_si512(_mm512_permutex2var_epi32(r2, idx_even, r3), mask24);
    const __m512i pl_lo = _mm512_mask_permutexvar_epi32(_mm512_permutexvar_epi32(rot_prev, ch_lo), 0x0001, rot_prev, ch_hi);
    const __m512i pl_hi = _mm512_mask_permutexvar_epi32(_mm512_permutexvar_epi32(rot_prev, ch_hi), 0x0001, rot_prev, ch_lo);
    k3_accum_half_multi<MAX_CNT>(ch_lo, pl_lo, mask20, x0_ptrs, x1_ptrs, p, cnt, alo0, alo8);
    k3_accum_half_multi<MAX_CNT>(ch_hi, pl_hi, mask20, x0_ptrs, x1_ptrs, p, cnt, ahi0, ahi8);
}

template <int MAX_CNT>
QW38_AVX512_ATTR
static inline void k3_wide_wg_multi(const uint8_t* __restrict__ p0,
                                    const __m512i* const* x0_ptrs, const __m512i* const* x1_ptrs,
                                    int cnt,
                                    __m512* alo0, __m512* alo8, __m512* ahi0, __m512* ahi8,
                                    int plane_stride) {
    const __m512i mask24 = _mm512_set1_epi32(0xffffffu);
    const __m512i mask20 = _mm512_set1_epi32(0xfffffu);
    const __m512i rot_prev = _mm512_setr_epi32(15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
    const __m512i idx_even = _mm512_setr_epi32(0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);

    const __m512i w0_0 = _mm512_loadu_si512(p0 + 0);
    const __m512i w0_1 = _mm512_loadu_si512(p0 + 64);
    const __m512i w0_2 = _mm512_loadu_si512(p0 + 128);
    const __m512i w0_3 = _mm512_loadu_si512(p0 + 192);
    k3_step_p_multi<MAX_CNT>(w0_0, w0_1, w0_2, w0_3, idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 0, cnt, alo0, alo8, ahi0, ahi8);
    k3_step_p_multi<MAX_CNT>(_mm512_srli_epi64(w0_0, 24), _mm512_srli_epi64(w0_1, 24),
                             _mm512_srli_epi64(w0_2, 24), _mm512_srli_epi64(w0_3, 24),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 1, cnt, alo0, alo8, ahi0, ahi8);

    const uint8_t* p1 = p0 + plane_stride;
    const __m512i w1_0 = _mm512_loadu_si512(p1 + 0);
    const __m512i w1_1 = _mm512_loadu_si512(p1 + 64);
    const __m512i w1_2 = _mm512_loadu_si512(p1 + 128);
    const __m512i w1_3 = _mm512_loadu_si512(p1 + 192);
    k3_step_p_multi<MAX_CNT>(_mm512_shrdi_epi64(w0_0, w1_0, 48), _mm512_shrdi_epi64(w0_1, w1_1, 48),
                             _mm512_shrdi_epi64(w0_2, w1_2, 48), _mm512_shrdi_epi64(w0_3, w1_3, 48),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 2, cnt, alo0, alo8, ahi0, ahi8);
    k3_step_p_multi<MAX_CNT>(_mm512_srli_epi64(w1_0, 8), _mm512_srli_epi64(w1_1, 8),
                             _mm512_srli_epi64(w1_2, 8), _mm512_srli_epi64(w1_3, 8),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 3, cnt, alo0, alo8, ahi0, ahi8);
    k3_step_p_multi<MAX_CNT>(_mm512_srli_epi64(w1_0, 32), _mm512_srli_epi64(w1_1, 32),
                             _mm512_srli_epi64(w1_2, 32), _mm512_srli_epi64(w1_3, 32),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 4, cnt, alo0, alo8, ahi0, ahi8);

    const uint8_t* p2 = p1 + plane_stride;
    const __m512i w2_0 = _mm512_loadu_si512(p2 + 0);
    const __m512i w2_1 = _mm512_loadu_si512(p2 + 64);
    const __m512i w2_2 = _mm512_loadu_si512(p2 + 128);
    const __m512i w2_3 = _mm512_loadu_si512(p2 + 192);
    k3_step_p_multi<MAX_CNT>(_mm512_shrdi_epi64(w1_0, w2_0, 56), _mm512_shrdi_epi64(w1_1, w2_1, 56),
                             _mm512_shrdi_epi64(w1_2, w2_2, 56), _mm512_shrdi_epi64(w1_3, w2_3, 56),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 5, cnt, alo0, alo8, ahi0, ahi8);
    k3_step_p_multi<MAX_CNT>(_mm512_srli_epi64(w2_0, 16), _mm512_srli_epi64(w2_1, 16),
                             _mm512_srli_epi64(w2_2, 16), _mm512_srli_epi64(w2_3, 16),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 6, cnt, alo0, alo8, ahi0, ahi8);
    k3_step_p_multi<MAX_CNT>(_mm512_srli_epi64(w2_0, 40), _mm512_srli_epi64(w2_1, 40),
                             _mm512_srli_epi64(w2_2, 40), _mm512_srli_epi64(w2_3, 40),
                             idx_even, rot_prev, mask24, mask20, x0_ptrs, x1_ptrs, 7, cnt, alo0, alo8, ahi0, ahi8);
}

// Tiny dedicated L1i-clean M=1 FC1 single-warp kernel (~4 KB machine code, 100% resident in 32 KB L1i!).
QW38_AVX512_FN __attribute__((noinline))
static void run_fc1_1warp_m1(
    const uint8_t* __restrict__ mw,
    const __m512i (*cur_x0)[geom::HB][8],
    const __m512i (*cur_x1)[geom::HB][8],
    const uint16_t* __restrict__ v1b_row,
    int tok, int w,
    float* __restrict__ dst16) {
    __m512 a_lo0 = _mm512_setzero_ps(), a_lo8 = _mm512_setzero_ps();
    __m512 a_hi0 = _mm512_setzero_ps(), a_hi8 = _mm512_setzero_ps();
    for (int hb = 0; hb < geom::HB; ++hb) {
        const uint8_t* hb_base = mw + size_t(hb) * (2048 * 3) + size_t(w) * 256;
        if (hb + 2 < geom::HB) {
            const uint8_t* nxt = hb_base + (2048 * 3) * 2;
            #pragma unroll
            for (int pl = 0; pl < 3; ++pl) {
                const uint8_t* q = nxt + pl * 2048;
                _mm_prefetch(q + 0,   _MM_HINT_NTA);
                _mm_prefetch(q + 64,  _MM_HINT_NTA);
                _mm_prefetch(q + 128, _MM_HINT_NTA);
                _mm_prefetch(q + 192, _MM_HINT_NTA);
            }
        }
        const __m512 vbs = _mm512_set1_ps(_cvtsh_ss(v1b_row[hb]));
        __m512 t0 = _mm512_setzero_ps(), t8 = _mm512_setzero_ps();
        __m512 th0 = _mm512_setzero_ps(), th8 = _mm512_setzero_ps();
        k3_wide_wg_m1(hb_base, cur_x0[tok][hb], cur_x1[tok][hb], t0, t8, th0, th8, 2048);
        a_lo0 = _mm512_fmadd_ps(t0, vbs, a_lo0);
        a_lo8 = _mm512_fmadd_ps(t8, vbs, a_lo8);
        a_hi0 = _mm512_fmadd_ps(th0, vbs, a_hi0);
        a_hi8 = _mm512_fmadd_ps(th8, vbs, a_hi8);
    }
    hsum4x4_store(a_lo0, dst16 + 0);
    hsum4x4_store(a_hi0, dst16 + 4);
    hsum4x4_store(a_lo8, dst16 + 8);
    hsum4x4_store(a_hi8, dst16 + 12);
}

// Zero-spill specialized single-warp multi-token FC1 kernel (MAX_CNT=2 fits 100% in 32 ZMM registers!).
template <int MAX_CNT>
QW38_AVX512_FN __attribute__((noinline))
static void run_fc1_1warp_multi_t(
    const uint8_t* __restrict__ mw,
    const __m512i (*cur_x0)[geom::HB][8],
    const __m512i (*cur_x1)[geom::HB][8],
    const int* __restrict__ s_tok,
    const uint16_t* __restrict__ v1b_row,
    int cnt, int w,
    float* __restrict__ y_out) {
    __m512 alo0[MAX_CNT], alo8[MAX_CNT], ahi0[MAX_CNT], ahi8[MAX_CNT];
    #pragma unroll
    for (int t = 0; t < MAX_CNT; ++t) {
        alo0[t] = _mm512_setzero_ps(); alo8[t] = _mm512_setzero_ps();
        ahi0[t] = _mm512_setzero_ps(); ahi8[t] = _mm512_setzero_ps();
    }
    const __m512i* x0_ptrs[MAX_CNT];
    const __m512i* x1_ptrs[MAX_CNT];
    for (int hb = 0; hb < geom::HB; ++hb) {
        const uint8_t* p0 = mw + size_t(hb) * (2048 * 3) + size_t(w) * 256;
        if (hb + 2 < geom::HB) {
            const uint8_t* nxt = p0 + (2048 * 3) * 2;
            #pragma unroll
            for (int pl = 0; pl < 3; ++pl) {
                const uint8_t* q = nxt + pl * 2048;
                _mm_prefetch(q + 0,   _MM_HINT_NTA);
                _mm_prefetch(q + 64,  _MM_HINT_NTA);
                _mm_prefetch(q + 128, _MM_HINT_NTA);
                _mm_prefetch(q + 192, _MM_HINT_NTA);
            }
        }
        const __m512 vbs = _mm512_set1_ps(_cvtsh_ss(v1b_row[hb]));
        #pragma unroll
        for (int t = 0; t < MAX_CNT; ++t) {
            const int tk = (t < cnt) ? s_tok[t] : s_tok[0];
            x0_ptrs[t] = cur_x0[tk][hb];
            x1_ptrs[t] = cur_x1[tk][hb];
        }
        __m512 tl0[MAX_CNT], tl8[MAX_CNT], th0[MAX_CNT], th8[MAX_CNT];
        #pragma unroll
        for (int t = 0; t < MAX_CNT; ++t) {
            tl0[t] = _mm512_setzero_ps(); tl8[t] = _mm512_setzero_ps();
            th0[t] = _mm512_setzero_ps(); th8[t] = _mm512_setzero_ps();
        }
        k3_wide_wg_multi<MAX_CNT>(p0, x0_ptrs, x1_ptrs, cnt, tl0, tl8, th0, th8, 2048);
        #pragma unroll
        for (int t = 0; t < MAX_CNT; ++t) {
            if (t < cnt) {
                alo0[t] = _mm512_fmadd_ps(tl0[t], vbs, alo0[t]);
                alo8[t] = _mm512_fmadd_ps(tl8[t], vbs, alo8[t]);
                ahi0[t] = _mm512_fmadd_ps(th0[t], vbs, ahi0[t]);
                ahi8[t] = _mm512_fmadd_ps(th8[t], vbs, ahi8[t]);
            }
        }
    }
    for (int t = 0; t < cnt; ++t) {
        float* dst = y_out + size_t(t) * 128 + w * 16;
        hsum4x4_store(alo0[t], dst + 0);
        hsum4x4_store(ahi0[t], dst + 4);
        hsum4x4_store(alo8[t], dst + 8);
        hsum4x4_store(ahi8[t], dst + 12);
    }
}

static inline void run_fc1_pair_avx512(
    const uint8_t* __restrict__ mw,
    const __m512i (*cur_x0)[geom::HB][8],
    const __m512i (*cur_x1)[geom::HB][8],
    const int* __restrict__ s_tok,
    const uint16_t* __restrict__ v1b_row,
    int cnt, int wpair,
    float* __restrict__ y_out) {
    const int w0 = wpair * 2;
    if (cnt == 1) {
        run_fc1_1warp_m1(mw, cur_x0, cur_x1, v1b_row, s_tok[0], w0,     y_out + w0 * 16);
        run_fc1_1warp_m1(mw, cur_x0, cur_x1, v1b_row, s_tok[0], w0 + 1, y_out + (w0 + 1) * 16);
    } else if (cnt == 2) {
        run_fc1_1warp_multi_t<2>(mw, cur_x0, cur_x1, s_tok, v1b_row, 2, w0,     y_out);
        run_fc1_1warp_multi_t<2>(mw, cur_x0, cur_x1, s_tok, v1b_row, 2, w0 + 1, y_out);
    } else if (cnt <= 4) {
        run_fc1_1warp_multi_t<4>(mw, cur_x0, cur_x1, s_tok, v1b_row, cnt, w0,     y_out);
        run_fc1_1warp_multi_t<4>(mw, cur_x0, cur_x1, s_tok, v1b_row, cnt, w0 + 1, y_out);
    } else {
        run_fc1_1warp_multi_t<8>(mw, cur_x0, cur_x1, s_tok, v1b_row, cnt, w0,     y_out);
        run_fc1_1warp_multi_t<8>(mw, cur_x0, cur_x1, s_tok, v1b_row, cnt, w0 + 1, y_out);
    }
}

// Tiny dedicated L1i-clean M=1 FC2 kernel (~4 KB machine code), supports FP32 (decode) or FP16 (prefill)!
template <bool OUT_FP16>
QW38_AVX512_FN __attribute__((noinline))
static void run_fc2_hb_m1_t(
    const uint8_t* __restrict__ h_fc2,
    size_t fc2_off,
    int nb,
    size_t tr2_base,
    const uint16_t* __restrict__ h_v3b,
    const uint16_t* __restrict__ h_sg_out,
    const __m512i* __restrict__ act_x0_g,
    const __m512i* __restrict__ act_x1_g,
    int out_row, float rw, int hb,
    void* __restrict__ out_buf) {
    alignas(64) float y_raw[128];
    std::memset(y_raw, 0, sizeof(y_raw));
    constexpr size_t FC2_BLK_B = size_t(geom::KSH) * 8 * 32 * 3;
    for (int b = 0; b < nb; ++b) {
        const float bscale = _cvtsh_ss(h_v3b[(tr2_base + b) * geom::HB + hb]);
        const uint8_t* mw = h_fc2 + fc2_off + size_t(b) * FC2_BLK_B;
        const __m512i* xp0 = act_x0_g + size_t(b) * 8;
        const __m512i* xp1 = act_x1_g + size_t(b) * 8;
        for (int w = 0; w < 8; ++w) {
            const uint8_t* p0 = mw + size_t(hb * 8 + w) * 768;
            if (w + 2 < 8) {
                #pragma unroll
                for (int off = 1536; off < 2304; off += 64) {
                    _mm_prefetch(p0 + off, _MM_HINT_NTA);
                }
            }
            __m512 alo0 = _mm512_setzero_ps(), alo8 = _mm512_setzero_ps();
            __m512 ahi0 = _mm512_setzero_ps(), ahi8 = _mm512_setzero_ps();
            k3_wide_wg_m1(p0, xp0, xp1, alo0, alo8, ahi0, ahi8, 256);
            float* dst = y_raw + w * 16;
            hsum4x4_add(alo0, dst + 0, bscale);
            hsum4x4_add(ahi0, dst + 4, bscale);
            hsum4x4_add(alo8, dst + 8, bscale);
            hsum4x4_add(ahi8, dst + 12, bscale);
        }
    }
    had128_avx512(y_raw);
    const __m512 vrw = _mm512_set1_ps(rw);
    const uint16_t* sgp = h_sg_out + hb * 128;
    if constexpr (OUT_FP16) {
        uint16_t* dst = static_cast<uint16_t*>(out_buf) + size_t(out_row) * geom::HID + hb * 128;
        for (int k = 0; k < 128; k += 16) {
            const __m512 yv = _mm512_load_ps(y_raw + k);
            const __m512 sv = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sgp + k)));
            const __m256i h = _mm512_cvtps_ph(_mm512_mul_ps(vrw, _mm512_mul_ps(yv, sv)), _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k), h);
        }
    } else {
        float* dst = static_cast<float*>(out_buf) + size_t(out_row) * geom::HID + hb * 128;
        for (int k = 0; k < 128; k += 16) {
            const __m512 yv = _mm512_load_ps(y_raw + k);
            const __m512 sv = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sgp + k)));
            _mm512_storeu_ps(dst + k, _mm512_mul_ps(vrw, _mm512_mul_ps(yv, sv)));
        }
    }
}

template <int MAX_CNT, bool OUT_FP16>
QW38_AVX512_FN __attribute__((noinline))
static void run_fc2_hb_multi_t(
    const uint8_t* __restrict__ h_fc2,
    size_t fc2_off,
    int nb, int eb,
    size_t tr2_base,
    const uint16_t* __restrict__ h_v3b,
    const uint16_t* __restrict__ h_sg_out,
    const __m512i* __restrict__ act_x0_g,
    const __m512i* __restrict__ act_x1_g,
    const int* __restrict__ s_out_row,
    const float* __restrict__ s_rw,
    int cnt, int hb,
    void* __restrict__ out_buf) {
    alignas(64) float y_raw[MAX_CNT][128];
    constexpr size_t FC2_BLK_B = size_t(geom::KSH) * 8 * 32 * 3;
    for (int t = 0; t < cnt; ++t) std::memset(y_raw[t], 0, 128 * sizeof(float));
    const __m512i* x0_ptrs[MAX_CNT];
    const __m512i* x1_ptrs[MAX_CNT];
    for (int b = 0; b < nb; ++b) {
        const float bscale = _cvtsh_ss(h_v3b[(tr2_base + b) * geom::HB + hb]);
        const uint8_t* mw = h_fc2 + fc2_off + size_t(b) * FC2_BLK_B;
        #pragma unroll
        for (int t = 0; t < MAX_CNT; ++t) {
            const int tt = (t < cnt) ? t : 0;
            x0_ptrs[t] = act_x0_g + (size_t(tt) * eb + b) * 8;
            x1_ptrs[t] = act_x1_g + (size_t(tt) * eb + b) * 8;
        }
        for (int w = 0; w < 8; ++w) {
            const uint8_t* p0 = mw + size_t(hb * 8 + w) * 768;
            if (w + 2 < 8) {
                #pragma unroll
                for (int off = 1536; off < 2304; off += 64) {
                    _mm_prefetch(p0 + off, _MM_HINT_NTA);
                }
            }
            __m512 alo0[MAX_CNT], alo8[MAX_CNT], ahi0[MAX_CNT], ahi8[MAX_CNT];
            #pragma unroll
            for (int t = 0; t < MAX_CNT; ++t) {
                alo0[t] = _mm512_setzero_ps(); alo8[t] = _mm512_setzero_ps();
                ahi0[t] = _mm512_setzero_ps(); ahi8[t] = _mm512_setzero_ps();
            }
            k3_wide_wg_multi<MAX_CNT>(p0, x0_ptrs, x1_ptrs, cnt, alo0, alo8, ahi0, ahi8, 256);
            for (int t = 0; t < cnt; ++t) {
                float* dst = y_raw[t] + w * 16;
                hsum4x4_add(alo0[t], dst + 0, bscale);
                hsum4x4_add(ahi0[t], dst + 4, bscale);
                hsum4x4_add(alo8[t], dst + 8, bscale);
                hsum4x4_add(ahi8[t], dst + 12, bscale);
            }
        }
    }
    alignas(64) float sg_f[128];
    const uint16_t* sgp = h_sg_out + hb * 128;
    for (int k = 0; k < 128; k += 16) {
        _mm512_store_ps(sg_f + k, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sgp + k))));
    }
    for (int t = 0; t < cnt; ++t) {
        had128_avx512(y_raw[t]);
        const __m512 vrw = _mm512_set1_ps(s_rw[t]);
        if constexpr (OUT_FP16) {
            uint16_t* dst = static_cast<uint16_t*>(out_buf) + size_t(s_out_row[t]) * geom::HID + hb * 128;
            for (int k = 0; k < 128; k += 16) {
                const __m512 yv = _mm512_load_ps(y_raw[t] + k);
                const __m512 sv = _mm512_load_ps(sg_f + k);
                const __m256i h = _mm512_cvtps_ph(_mm512_mul_ps(vrw, _mm512_mul_ps(yv, sv)), _MM_FROUND_TO_NEAREST_INT);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k), h);
            }
        } else {
            float* dst = static_cast<float*>(out_buf) + size_t(s_out_row[t]) * geom::HID + hb * 128;
            for (int k = 0; k < 128; k += 16) {
                const __m512 yv = _mm512_load_ps(y_raw[t] + k);
                const __m512 sv = _mm512_load_ps(sg_f + k);
                _mm512_storeu_ps(dst + k, _mm512_mul_ps(vrw, _mm512_mul_ps(yv, sv)));
            }
        }
    }
}

template <bool OUT_FP16 = false>
static inline void run_fc2_hb_avx512(
    const uint8_t* __restrict__ h_fc2,
    size_t fc2_off,
    int nb, int eb,
    size_t tr2_base,
    const uint16_t* __restrict__ h_v3b,
    const uint16_t* __restrict__ h_sg_out,
    const __m512i* __restrict__ act_x0_g,
    const __m512i* __restrict__ act_x1_g,
    const int* __restrict__ s_out_row,
    const float* __restrict__ s_rw,
    int cnt, int hb,
    void* __restrict__ out_buf) {
    if (cnt == 1) {
        run_fc2_hb_m1_t<OUT_FP16>(h_fc2, fc2_off, nb, tr2_base, h_v3b, h_sg_out, act_x0_g, act_x1_g,
                                  s_out_row[0], s_rw[0], hb, out_buf);
    } else if (cnt == 2) {
        run_fc2_hb_multi_t<2, OUT_FP16>(h_fc2, fc2_off, nb, eb, tr2_base, h_v3b, h_sg_out, act_x0_g, act_x1_g,
                                        s_out_row, s_rw, 2, hb, out_buf);
    } else if (cnt <= 4) {
        run_fc2_hb_multi_t<4, OUT_FP16>(h_fc2, fc2_off, nb, eb, tr2_base, h_v3b, h_sg_out, act_x0_g, act_x1_g,
                                        s_out_row, s_rw, cnt, hb, out_buf);
    } else {
        run_fc2_hb_multi_t<8, OUT_FP16>(h_fc2, fc2_off, nb, eb, tr2_base, h_v3b, h_sg_out, act_x0_g, act_x1_g,
                                        s_out_row, s_rw, cnt, hb, out_buf);
    }
}

QW38_AVX512_FN __attribute__((noinline))
static void run_epilogue_gb_avx512(
    float* __restrict__ yg_base,
    float* __restrict__ yu_base,
    const uint16_t* __restrict__ svh_g,
    const uint16_t* __restrict__ svh_u,
    const uint16_t* __restrict__ suh_d,
    int cnt, int eb, int b,
    __m512i* __restrict__ act_x0_g,
    __m512i* __restrict__ act_x1_g) {
    alignas(64) float sg_f[128], su_f[128], sd_f[128], va[128];
    for (int j = 0; j < 128; j += 16) {
        _mm512_store_ps(sg_f + j, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(svh_g + j))));
        _mm512_store_ps(su_f + j, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(svh_u + j))));
        _mm512_store_ps(sd_f + j, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(suh_d + j))));
    }
    for (int t = 0; t < cnt; ++t) {
        float* vg = yg_base + size_t(t) * 128;
        float* vu = yu_base + size_t(t) * 128;
        had128_avx512(vg);
        had128_avx512(vu);
        for (int j = 0; j < 128; ++j) {
            const float gv = vg[j] * sg_f[j];
            const float uv = vu[j] * su_f[j];
            va[j] = (silu_cpu(gv) * uv) * sd_f[j];
        }
        had128_avx512(va);
        const size_t act_off = (size_t(t) * size_t(eb) + size_t(b)) * 8;
        pack_x128_f32_to_bf16(va, act_x0_g + act_off, act_x1_g + act_off);
    }
}
#endif // __x86_64__

} // namespace cpumoe_detail

// Persistent low-latency CPU worker pool for hybrid decode AND small/medium prefill cold-expert execution.
struct CpuMoePool {
    static constexpr int MAX_GROUPS = 512;   // supports up to 512 cold experts/layer (decode <= 80, prefill <= 512)
    static constexpr int MAX_PACK_M = 512;   // max prefill M for CPU cold-expert offload

    CpuMoeBridge* bridge = nullptr;          // mapped pinned host pointer
    CpuMoeBridge* d_bridge = nullptr;        // GPU UVA device pointer to same memory
    std::vector<CpuMoeLayerRef> layers;
    std::vector<std::vector<long long>> h_tab_store, h_kmix_store;
    std::vector<std::vector<uint16_t>> h_sg_store, h_v1b_store, h_v2_store, h_v3b_store;

    bool enabled = false;
    int n_threads = 14;
    float cpu_frac = 0.60f;
    bool avx512_ok = false;

    // Scratch per-group buffers
    std::vector<float> grp_y_gate;           // [MAX_GROUPS * eb * 8 * 128]
    std::vector<float> grp_y_up;             // [MAX_GROUPS * eb * 8 * 128]
#if defined(__x86_64__) || defined(_M_X64)
    // Pre-packed BF16 x_had per layer: [MAX_PACK_M][20][8] stored as __m512i (64-byte aligned, avoids nvcc ICE!)
    using XPackRow = __m512i[geom::HB][8];
    XPackRow* cur_x0 = nullptr;
    XPackRow* cur_x1 = nullptr;
    // Pre-packed BF16 fc1_act per group: [MAX_GROUPS * 8 * eb * 8] (64-byte aligned!)
    __m512i* grp_act_x0 = nullptr;
    __m512i* grp_act_x1 = nullptr;
#endif

    // Worker pool sync: 64-slot ring of task-completion work-stealing barriers!
    // Preemption-immune: if any thread is preempted by OS/CUDA, remaining threads steal all tasks
    // and finish the layer with ZERO 4ms CFS stall!
    std::vector<std::thread> workers;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> active_step{false};
    std::atomic<bool> prefill_mode{false};
    std::mutex wake_mu;
    std::condition_variable wake_cv;

    struct alignas(64) JobSlot {
        alignas(64) std::atomic<int> next_pack{0};
        alignas(64) std::atomic<int> done_pack{0};
        alignas(64) std::atomic<int> next_fc1{0};
        alignas(64) std::atomic<int> done_fc1{0};
        alignas(64) std::atomic<int> next_epi{0};
        alignas(64) std::atomic<int> done_epi{0};
        alignas(64) std::atomic<int> next_fc2{0};
        alignas(64) std::atomic<int> done_fc2{0};
        alignas(64) int n_pack = 0;
        int n_fc1 = 0;
        int n_epi = 0;
        int n_fc2 = 0;
    };
    JobSlot job_slots[64];
    alignas(64) std::atomic<uint64_t> job_gen{0};
    int cur_li = 0;
    int cur_ng = 0;
    bool cur_out_fp16 = false;
    const uint16_t* cur_pack_src = nullptr;  // [M, HID] FP16 for parallel prefill x_had packing
    void* cur_out_buf = nullptr;             // float* (decode) or uint16_t* (prefill)
    int cur_grp_exp[MAX_GROUPS] = {};
    int cur_grp_cnt[MAX_GROUPS] = {};
    int cur_grp_nb[MAX_GROUPS] = {};
    int cur_grp_toks[MAX_GROUPS][8] = {};
    int cur_grp_out_row[MAX_GROUPS][8] = {}; // decode: route id; prefill: compact output row 0..n_cpu_routes-1
    float cur_grp_rws[MAX_GROUPS][8] = {};

    // Telemetry
    uint64_t stat_cpu_groups = 0;
    uint64_t stat_cpu_pf_groups = 0;
    uint64_t stat_cpu_dispatches = 0;
    uint64_t stat_cpu_busy_ns = 0;

    ~CpuMoePool() { shutdown(); }

    static bool check_avx512() {
#if defined(__x86_64__) || defined(_M_X64)
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c") &&
               __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512vbmi2");
#else
        return false;
#endif
    }

    void init(int nl, int eb,
              const std::vector<const uint8_t*>& h_fc1_vec,
              const std::vector<const uint8_t*>& h_fc2_vec,
              const std::vector<std::vector<long long>>& h_tabs,
              const std::vector<std::vector<long long>>& h_kmixs,
              const std::vector<const __half*>& d_sg_vec,
              const std::vector<const __half*>& d_v1b_vec,
              const std::vector<const __half*>& d_v2_vec,
              const std::vector<const __half*>& d_v3b_vec,
              const std::vector<size_t>& v1b_cnt,
              const std::vector<size_t>& v2_cnt,
              const std::vector<size_t>& v3b_cnt,
              int nth, float cfrac) {
        avx512_ok = check_avx512();
        if (!avx512_ok || nth <= 0 || cfrac <= 0.0f) {
            enabled = false;
            return;
        }
        n_threads = std::max(1, std::min(nth, 16));
        cpu_frac = std::clamp(cfrac, 0.0f, 1.0f);

        cudaHostAlloc(&bridge, sizeof(CpuMoeBridge), cudaHostAllocMapped);
        std::memset(bridge, 0, sizeof(CpuMoeBridge));
        cudaHostGetDevicePointer(&d_bridge, bridge, 0);

        layers.resize(size_t(nl));
        h_tab_store = h_tabs;
        h_kmix_store = h_kmixs;
        h_sg_store.resize(size_t(nl));
        h_v1b_store.resize(size_t(nl));
        h_v2_store.resize(size_t(nl));
        h_v3b_store.resize(size_t(nl));
        for (int li = 0; li < nl; ++li) {
            h_sg_store[size_t(li)].resize(2 * geom::HID);
            h_v1b_store[size_t(li)].resize(v1b_cnt[size_t(li)]);
            h_v2_store[size_t(li)].resize(v2_cnt[size_t(li)]);
            h_v3b_store[size_t(li)].resize(v3b_cnt[size_t(li)]);
            cudaMemcpy(h_sg_store[size_t(li)].data(), d_sg_vec[size_t(li)], 2 * geom::HID * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v1b_store[size_t(li)].data(), d_v1b_vec[size_t(li)], v1b_cnt[size_t(li)] * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v2_store[size_t(li)].data(), d_v2_vec[size_t(li)], v2_cnt[size_t(li)] * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v3b_store[size_t(li)].data(), d_v3b_vec[size_t(li)], v3b_cnt[size_t(li)] * sizeof(uint16_t), cudaMemcpyDeviceToHost);

            auto& lr = layers[size_t(li)];
            lr.h_fc1 = h_fc1_vec[size_t(li)];
            lr.h_fc2 = h_fc2_vec[size_t(li)];
            lr.h_tab = h_tab_store[size_t(li)].data();
            lr.h_kmix = h_kmix_store[size_t(li)].empty() ? nullptr : h_kmix_store[size_t(li)].data();
            lr.h_sg = h_sg_store[size_t(li)].data();
            lr.h_v1b = h_v1b_store[size_t(li)].data();
            lr.h_v2 = h_v2_store[size_t(li)].data();
            lr.h_v3b = h_v3b_store[size_t(li)].data();
            lr.eb = eb;
        }

        grp_y_gate.assign(size_t(MAX_GROUPS) * size_t(eb) * 8 * 128, 0.0f);
        grp_y_up.assign(size_t(MAX_GROUPS) * size_t(eb) * 8 * 128, 0.0f);
#if defined(__x86_64__) || defined(_M_X64)
        const size_t x_bytes = size_t(MAX_PACK_M) * sizeof(XPackRow);
        cur_x0 = static_cast<XPackRow*>(std::aligned_alloc(64, x_bytes));
        cur_x1 = static_cast<XPackRow*>(std::aligned_alloc(64, x_bytes));
        std::memset(cur_x0, 0, x_bytes);
        std::memset(cur_x1, 0, x_bytes);
        const size_t act_bytes = size_t(MAX_GROUPS) * 8 * size_t(eb) * 8 * sizeof(__m512i);
        grp_act_x0 = static_cast<__m512i*>(std::aligned_alloc(64, act_bytes));
        grp_act_x1 = static_cast<__m512i*>(std::aligned_alloc(64, act_bytes));
        std::memset(grp_act_x0, 0, act_bytes);
        std::memset(grp_act_x1, 0, act_bytes);
#endif

        stop_flag.store(false, std::memory_order_release);
        active_step.store(false, std::memory_order_release);
        enabled = true;

        workers.reserve(size_t(n_threads));
        workers.emplace_back([this]() { coordinator_loop(); });
        for (int t = 1; t < n_threads; ++t) {
            workers.emplace_back([this, t]() { worker_loop(t); });
        }

        std::fprintf(stderr, "# qw38 strata CPU hybrid K3 MoE: ON (%d AVX-512 BF16/VBMI2 threads, cpu_frac=%.2f)\n",
                     n_threads, cpu_frac);
    }

    void shutdown() {
        if (!enabled) return;
        stop_flag.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(wake_mu);
            active_step.store(true, std::memory_order_release);
        }
        wake_cv.notify_all();
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto& th : workers) {
            if (th.joinable()) th.join();
        }
        workers.clear();
        if (bridge) {
            cudaFreeHost(bridge);
            bridge = nullptr;
            d_bridge = nullptr;
        }
#if defined(__x86_64__) || defined(_M_X64)
        if (cur_x0) { std::free(cur_x0); cur_x0 = nullptr; }
        if (cur_x1) { std::free(cur_x1); cur_x1 = nullptr; }
        if (grp_act_x0) { std::free(grp_act_x0); grp_act_x0 = nullptr; }
        if (grp_act_x1) { std::free(grp_act_x1); grp_act_x1 = nullptr; }
#endif
        enabled = false;
    }

    void begin_step(bool is_prefill = false) {
        if (!enabled) return;
        {
            std::lock_guard<std::mutex> lk(wake_mu);
            prefill_mode.store(is_prefill, std::memory_order_release);
            active_step.store(true, std::memory_order_release);
        }
        wake_cv.notify_all();
    }

    void end_step() {
        if (!enabled) return;
        active_step.store(false, std::memory_order_release);
        prefill_mode.store(false, std::memory_order_release);
    }

    // Parallel x_had BF16 packing task for prefill (t = token index 0..M-1)
    void exec_pack_task(int m) {
#if defined(__x86_64__) || defined(_M_X64)
        const uint16_t* row = cur_pack_src + size_t(m) * geom::HID;
        for (int hb = 0; hb < geom::HB; ++hb) {
            cpumoe_detail::pack_x128_f16_to_bf16(row + hb * 128, cur_x0[m][hb], cur_x1[m][hb]);
        }
#else
        (void)m;
#endif
    }

    // Execute one FC1 task: tidx = ((g * eb + b) * 2 + mat) * 4 + wpair (40 tasks/expert!)
    void exec_fc1_task(int tidx) {
#if defined(__x86_64__) || defined(_M_X64)
        const auto& lr = layers[size_t(cur_li)];
        const int eb = lr.eb;
        const int wpair = tidx & 3;
        const int mat = (tidx >> 2) & 1;
        const int gb = tidx >> 3;
        const int b = gb % eb;
        const int g = gb / eb;
        const int e = cur_grp_exp[g];
        const int cnt = cur_grp_cnt[g];
        const long long* tr = lr.h_tab + 6 * e;
        if (b >= int(tr[3])) return;
        const size_t brow = size_t(tr[2]) + b;
        constexpr size_t MB = size_t(geom::KSH) * 8 * 32; // 40960 B/bit
        const size_t unit_b = 3 * MB; // K3
        const uint8_t* mw = lr.h_fc1 + size_t(tr[0]) + size_t(2 * b + mat) * unit_b;
        const uint16_t* v1b_row = lr.h_v1b + (brow * 2 + mat) * geom::HB;
        float* y_dst = ((mat == 0) ? grp_y_gate.data() : grp_y_up.data()) + size_t(g * eb + b) * 8 * 128;
        cpumoe_detail::run_fc1_pair_avx512(mw, cur_x0, cur_x1, cur_grp_toks[g], v1b_row, cnt, wpair, y_dst);
#else
        (void)tidx;
#endif
    }

    // Epilogue for one (g, b): vectorized had128(gate)*svh_g, had128(up)*svh_u, silu(g)*u*suh_d, had128 -> BF16 act!
    void exec_epilogue_gb(int gb) {
#if defined(__x86_64__) || defined(_M_X64)
        const auto& lr = layers[size_t(cur_li)];
        const int eb = lr.eb;
        const int b = gb % eb;
        const int g = gb / eb;
        const int e = cur_grp_exp[g];
        const int cnt = cur_grp_cnt[g];
        const long long* tr = lr.h_tab + 6 * e;
        if (b >= int(tr[3])) return;
        const size_t brow = size_t(tr[2]) + b;
        const uint16_t* svh_g = lr.h_v2 + (brow * 3 + 0) * 128;
        const uint16_t* svh_u = lr.h_v2 + (brow * 3 + 1) * 128;
        const uint16_t* suh_d = lr.h_v2 + (brow * 3 + 2) * 128;
        float* yg_base = grp_y_gate.data() + size_t(g * eb + b) * 8 * 128;
        float* yu_base = grp_y_up.data() + size_t(g * eb + b) * 8 * 128;
        const size_t g_off = size_t(g) * 8 * size_t(eb) * 8;
        cpumoe_detail::run_epilogue_gb_avx512(yg_base, yu_base, svh_g, svh_u, suh_d,
                                              cnt, eb, b, grp_act_x0 + g_off, grp_act_x1 + g_off);
#else
        (void)gb;
#endif
    }

    // Execute one FC2 task: tidx = g * 20 + hb
    void exec_fc2_task(int tidx) {
#if defined(__x86_64__) || defined(_M_X64)
        const auto& lr = layers[size_t(cur_li)];
        const int hb = tidx % geom::HB;
        const int g = tidx / geom::HB;
        const int e = cur_grp_exp[g];
        const int cnt = cur_grp_cnt[g];
        const long long* tr = lr.h_tab + 6 * e;
        const int nb = cur_grp_nb[g];
        const size_t g_off = size_t(g) * 8 * size_t(lr.eb) * 8;
        if (cur_out_fp16) {
            cpumoe_detail::run_fc2_hb_avx512<true>(
                lr.h_fc2, size_t(tr[1]), nb, lr.eb, size_t(tr[2]),
                lr.h_v3b, lr.h_sg + geom::HID,
                grp_act_x0 + g_off, grp_act_x1 + g_off,
                cur_grp_out_row[g], cur_grp_rws[g], cnt, hb, cur_out_buf);
        } else {
            cpumoe_detail::run_fc2_hb_avx512<false>(
                lr.h_fc2, size_t(tr[1]), nb, lr.eb, size_t(tr[2]),
                lr.h_v3b, lr.h_sg + geom::HID,
                grp_act_x0 + g_off, grp_act_x1 + g_off,
                cur_grp_out_row[g], cur_grp_rws[g], cnt, hb, cur_out_buf);
        }
#else
        (void)tidx;
#endif
    }

    void participate_in_job(uint64_t g) {
        JobSlot& sl = job_slots[g & 63];
        const int n_pack = sl.n_pack;
        const int n_fc1 = sl.n_fc1;
        const int n_epi = sl.n_epi;
        const int n_fc2 = sl.n_fc2;
        if (n_pack > 0) {
            while (true) {
                const int m = sl.next_pack.fetch_add(1, std::memory_order_relaxed);
                if (m >= n_pack) break;
                exec_pack_task(m);
                sl.done_pack.fetch_add(1, std::memory_order_release);
            }
            while (sl.done_pack.load(std::memory_order_acquire) < n_pack) {
                if (job_gen.load(std::memory_order_relaxed) != g) return;
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
            }
        }
        // Phase 1: FC1 work-stealing (preemption-immune!)
        while (true) {
            const int t = sl.next_fc1.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_fc1) break;
            exec_fc1_task(t);
            sl.done_fc1.fetch_add(1, std::memory_order_release);
        }
        while (sl.done_fc1.load(std::memory_order_acquire) < n_fc1) {
            if (job_gen.load(std::memory_order_relaxed) != g) return;
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        // Phase 1.5: Epilogue + BF16 act pre-pack work-stealing
        while (true) {
            const int gb = sl.next_epi.fetch_add(1, std::memory_order_relaxed);
            if (gb >= n_epi) break;
            exec_epilogue_gb(gb);
            sl.done_epi.fetch_add(1, std::memory_order_release);
        }
        while (sl.done_epi.load(std::memory_order_acquire) < n_epi) {
            if (job_gen.load(std::memory_order_relaxed) != g) return;
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        // Phase 2: FC2 work-stealing
        while (true) {
            const int t = sl.next_fc2.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_fc2) break;
            exec_fc2_task(t);
            sl.done_fc2.fetch_add(1, std::memory_order_release);
        }
    }

    // Synchronous prefill cold-expert offload called on host thread in parallel with GPU prefill!
    // Uses the calling thread + (n_threads - 1) worker threads = n_threads total.
    void run_prefill_sync(int li, int M, const uint16_t* h_xhad,
                          int ng, const int* exps, const int* cnts,
                          const int (*toks)[8], const int (*out_rows)[8], const float (*rws)[8],
                          uint16_t* h_out_fp16) {
        if (!enabled || ng <= 0 || li < 0 || li >= int(layers.size())) return;
        const auto& lr = layers[size_t(li)];
        cur_li = li;
        cur_ng = ng;
        cur_out_fp16 = true;
        cur_pack_src = h_xhad;
        cur_out_buf = h_out_fp16;
        for (int g = 0; g < ng; ++g) {
            const int e = exps[g];
            const int c = cnts[g];
            cur_grp_exp[g] = e;
            cur_grp_cnt[g] = c;
            cur_grp_nb[g] = int(lr.h_tab[6 * e + 3]);
            for (int t = 0; t < c; ++t) {
                cur_grp_toks[g][t] = toks[g][t];
                cur_grp_out_row[g][t] = out_rows[g][t];
                cur_grp_rws[g][t] = rws[g][t];
            }
        }
        const uint64_t g = job_gen.load(std::memory_order_relaxed) + 1;
        JobSlot& sl = job_slots[g & 63];
        sl.n_pack = std::min(M, MAX_PACK_M);
        sl.n_fc1 = ng * lr.eb * 8;
        sl.n_epi = ng * lr.eb;
        sl.n_fc2 = ng * geom::HB;
        sl.next_pack.store(0, std::memory_order_relaxed);
        sl.done_pack.store(0, std::memory_order_relaxed);
        sl.next_fc1.store(0, std::memory_order_relaxed);
        sl.done_fc1.store(0, std::memory_order_relaxed);
        sl.next_epi.store(0, std::memory_order_relaxed);
        sl.done_epi.store(0, std::memory_order_relaxed);
        sl.next_fc2.store(0, std::memory_order_relaxed);
        sl.done_fc2.store(0, std::memory_order_relaxed);

        job_gen.store(g, std::memory_order_release);
        participate_in_job(g);

        while (sl.done_fc2.load(std::memory_order_acquire) < sl.n_fc2) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        stat_cpu_pf_groups += uint64_t(ng);
    }

    void worker_loop(int /*tid*/) {
        // Leave workers unpinned so Linux CFS can migrate around CUDA/OS/peer processes with zero stalls!
        uint64_t seen_gen = 0;
        while (!stop_flag.load(std::memory_order_acquire)) {
            if (!active_step.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(wake_mu);
                wake_cv.wait(lk, [&]() {
                    return stop_flag.load(std::memory_order_acquire) ||
                           active_step.load(std::memory_order_acquire);
                });
                if (stop_flag.load(std::memory_order_acquire)) break;
            }
            const uint64_t g = job_gen.load(std::memory_order_acquire);
            if (g != seen_gen) {
                seen_gen = g;
                if (stop_flag.load(std::memory_order_acquire)) break;
                participate_in_job(g);
            } else {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
            }
        }
    }

    void coordinator_loop() {
        uint32_t last_seq = 0;
        while (!stop_flag.load(std::memory_order_acquire)) {
            if (!active_step.load(std::memory_order_acquire) || prefill_mode.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(wake_mu);
                wake_cv.wait(lk, [&]() {
                    return stop_flag.load(std::memory_order_acquire) ||
                           (active_step.load(std::memory_order_acquire) && !prefill_mode.load(std::memory_order_acquire));
                });
                if (stop_flag.load(std::memory_order_acquire)) break;
            }
            const uint32_t seq = bridge->go_seq;
            if (seq == last_seq) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            last_seq = seq;

            const int li = bridge->cur_layer;
            const int ng = bridge->cpu_n_groups;
            const int cur_m = std::max(1, std::min(int(bridge->cur_M), CpuMoeBridge::MAX_M));
            if (ng > 0 && li >= 0 && li < int(layers.size())) {
                const auto t_c0 = std::chrono::steady_clock::now();
                const auto& lr = layers[size_t(li)];
                cur_li = li;
                cur_ng = ng;
                cur_out_fp16 = false;
                cur_out_buf = const_cast<float*>(bridge->cpu_fc2_part);
                // Pre-pack x_had[0..cur_m-1][0..19] ONCE per layer (<0.6 us)!
#if defined(__x86_64__) || defined(_M_X64)
                const uint16_t* xh_base = const_cast<const uint16_t*>(bridge->x_had);
                for (int m = 0; m < cur_m; ++m) {
                    for (int hb = 0; hb < geom::HB; ++hb) {
                        cpumoe_detail::pack_x128_f16_to_bf16(
                            xh_base + (size_t(m) * geom::HB + hb) * 128,
                            cur_x0[m][hb], cur_x1[m][hb]);
                    }
                }
#endif
                for (int g = 0; g < ng; ++g) {
                    const int e = bridge->cpu_grp_exp[g];
                    const int cnt = bridge->cpu_grp_cnt[g];
                    cur_grp_exp[g] = e;
                    cur_grp_cnt[g] = cnt;
                    cur_grp_nb[g] = int(lr.h_tab[6 * e + 3]);
                    for (int t = 0; t < cnt; ++t) {
                        const int r = bridge->cpu_grp_routes[g * 8 + t];
                        cur_grp_out_row[g][t] = r;
                        cur_grp_toks[g][t] = r / geom::TOPK;
                        cur_grp_rws[g][t] = bridge->topk_w[r];
                    }
                }
                const uint64_t g = job_gen.load(std::memory_order_relaxed) + 1;
                JobSlot& sl = job_slots[g & 63];
                sl.n_pack = 0;
                sl.n_fc1 = ng * lr.eb * 8; // 40 tasks/expert (2 warps each)
                sl.n_epi = ng * lr.eb;
                sl.n_fc2 = ng * geom::HB;  // 20 tasks/expert
                sl.next_fc1.store(0, std::memory_order_relaxed);
                sl.done_fc1.store(0, std::memory_order_relaxed);
                sl.next_epi.store(0, std::memory_order_relaxed);
                sl.done_epi.store(0, std::memory_order_relaxed);
                sl.next_fc2.store(0, std::memory_order_relaxed);
                sl.done_fc2.store(0, std::memory_order_relaxed);

                job_gen.store(g, std::memory_order_release);
                participate_in_job(g);

                while (sl.done_fc2.load(std::memory_order_acquire) < sl.n_fc2) {
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
#endif
                }
                stat_cpu_busy_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_c0).count());
                stat_cpu_groups += uint64_t(ng);
                ++stat_cpu_dispatches;
            }
            std::atomic_thread_fence(std::memory_order_release);
            bridge->done_seq = seq;
        }
    }
};

} // namespace qw38rt
