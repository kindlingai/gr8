// CPU AVX-512 micro-benchmark for Qwen3.8 K3 pmcgx (exlr8-e4m3) routed experts on Ryzen 7 9800X3D.
// Tests exact pmcgx E4M3 decoding + GEMM for M=1 and M=5 across 8/16 threads (both LUT-based and arithmetic).
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

static inline float fp16_to_f32(uint16_t h) {
    return _cvtsh_ss(h);
}
static inline uint16_t f32_to_fp16(float f) {
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
}
static inline uint16_t round_fp16_e4m3(uint16_t h) {
    float v = fp16_to_f32(h);
    uint16_t sign = h & 0x8000u;
    float av = std::fabs(v);
    if (av >= 448.0f) return uint16_t(sign | 0x5F00u);
    if (av == 0.0f) return sign;
    float q;
    if (av < (1.0f / 64.0f)) {
        q = std::nearbyint(av * 512.0f) / 512.0f;
    } else {
        int exp = 0;
        float frac = std::frexp(av, &exp);
        float m = std::nearbyint(frac * 16.0f);
        q = std::ldexp(m / 16.0f, exp);
        if (q > 448.0f) q = 448.0f;
    }
    return f32_to_fp16(sign ? -q : q);
}

// 4 MiB LUT: 2^20 states -> packed (fp16 w0, fp16 w1), exact E4M3-rounded pmcgx!
static uint32_t* build_pmcgx_lut() {
    uint32_t* lut = static_cast<uint32_t*>(aligned_alloc(64, (1u << 20) * sizeof(uint32_t)));
    for (uint32_t w = 0; w < (1u << 20); ++w) {
        uint32_t x0 = ((w * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t x1 = ((w * 0xBCA14C51u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        float v0 = fp16_to_f32(uint16_t(x0 & 0xFFFFu)) + fp16_to_f32(uint16_t(x1 >> 16));
        float v1 = fp16_to_f32(uint16_t(x0 >> 16)) + fp16_to_f32(uint16_t(x1 & 0xFFFFu));
        uint16_t h0 = round_fp16_e4m3(f32_to_fp16(v0));
        uint16_t h1 = round_fp16_e4m3(f32_to_fp16(v1));
        lut[w] = uint32_t(h0) | (uint32_t(h1) << 16);
    }
    return lut;
}

// Process one 16x16 K3 tile (96 bytes, 128 pmcgx pairs -> 256 weights) in pack order for M tokens.
// Wait: in wide order, 8 k-steps x 8 tiles = 6144 bytes = [3 u64][8 tiles][32 lanes]!
// Here we benchmark reading 1 full expert (1,843,200 bytes = 19,200 tiles of 96 B = 300 groups of 6144 B)
// using AVX-512 gather from the 4 MiB L3-resident LUT vs arithmetic!
template <int M, bool USE_LUT>
static float run_expert_slice(const uint8_t* __restrict__ data, size_t n_tiles,
                              const uint32_t* __restrict__ lut,
                              const float* __restrict__ x_m) {
    // Each 96-byte tile = 128 pairs (6 bits/pair = 768 bits).
    // Vectorize 16 pairs (32 weights) at a time with AVX-512!
    __m512 acc[M];
    for (int m = 0; m < M; ++m) acc[m] = _mm512_setzero_ps();

    const __m512i vmask20 = _mm512_set1_epi32(0xFFFFFu);
    const __m512i vc0 = _mm512_set1_epi32(int(0xCBAC1FEDu));
    const __m512i vc1 = _mm512_set1_epi32(int(0xBCA14C51u));
    const __m512i vmsk = _mm512_set1_epi32(int(0x8FFF8FFFu));
    const __m512i vxor = _mm512_set1_epi32(int(0x3B603B60u));
    const __m512i v16_mask = _mm512_set1_epi32(0xFFFF);

    // Precompute bit offsets for 16 consecutive pairs within a 96-bit (12-byte) chunk:
    // Actually 16 pairs * 6 bits = 96 bits = 12 bytes!
    // Each pair p in 0..15 has its 20-bit state ending at bit (p+1)*6 -> starts at (p+1)*6 - 20 = 6*p - 14.
    // Loading 4 uint32s (16 bytes, overlapping 4 bytes with previous) gives ALL bits for 16 pairs!
    alignas(64) static const int32_t k_widx[16] = {0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3};
    // Shift right amount in 64-bit word (u64 = (w[i]<<32)|w[i+1]) for each of 16 pairs
    alignas(64) static const int32_t k_sh[16] = {
        26, 20, 14, 8, 2,
        28, 22, 16, 10, 4,
        30, 24, 18, 12, 6, 0
    };
    const __m512i vsh = _mm512_load_si512(k_sh);

    for (size_t t = 0; t < n_tiles; ++t) {
        const uint8_t* tp = data + t * 96;
        // 8 chunks of 16 pairs (12 bytes each = 96 bytes total)
        // Load 24 uint32s = 96 bytes (plus wrap for tail-biting)
        const uint32_t* w = reinterpret_cast<const uint32_t*>(tp);
        for (int c = 0; c < 8; ++c) {
            // Build 16 20-bit states from w[c*3 - 1 .. c*3 + 3] (5 uint32s)
            const uint32_t w_prev = (c == 0) ? w[23] : w[c * 3 - 1];
            const uint32_t w0 = w[c * 3 + 0];
            const uint32_t w1 = w[c * 3 + 1];
            const uint32_t w2 = w[c * 3 + 2];
            // Pack hi/lo pairs into 16 lanes using 64-bit shifts (or 2x 8-lane u64)
            uint32_t st_arr[16];
            const uint64_t q0 = (uint64_t(w_prev) << 32) | w0;
            const uint64_t q1 = (uint64_t(w0) << 32) | w1;
            const uint64_t q2 = (uint64_t(w1) << 32) | w2;
            // 16 pairs: first 5 from q0, next 5 from q1, next 6 from q2
            st_arr[0]  = uint32_t(q0 >> 26) & 0xFFFFFu;
            st_arr[1]  = uint32_t(q0 >> 20) & 0xFFFFFu;
            st_arr[2]  = uint32_t(q0 >> 14) & 0xFFFFFu;
            st_arr[3]  = uint32_t(q0 >>  8) & 0xFFFFFu;
            st_arr[4]  = uint32_t(q0 >>  2) & 0xFFFFFu;
            st_arr[5]  = uint32_t(q1 >> 28) & 0xFFFFFu;
            st_arr[6]  = uint32_t(q1 >> 22) & 0xFFFFFu;
            st_arr[7]  = uint32_t(q1 >> 16) & 0xFFFFFu;
            st_arr[8]  = uint32_t(q1 >> 10) & 0xFFFFFu;
            st_arr[9]  = uint32_t(q1 >>  4) & 0xFFFFFu;
            st_arr[10] = uint32_t(q2 >> 30) & 0xFFFFFu;
            st_arr[11] = uint32_t(q2 >> 24) & 0xFFFFFu;
            st_arr[12] = uint32_t(q2 >> 18) & 0xFFFFFu;
            st_arr[13] = uint32_t(q2 >> 12) & 0xFFFFFu;
            st_arr[14] = uint32_t(q2 >>  6) & 0xFFFFFu;
            st_arr[15] = uint32_t(q2 >>  0) & 0xFFFFFu;
            (void)vsh; (void)k_widx; (void)vmask20;

            __m512i vst = _mm512_loadu_si512(st_arr);
            __m512 wf0, wf1;
            if constexpr (USE_LUT) {
                __m512i packed = _mm512_i32gather_epi32(vst, lut, 4);
                __m256i h0 = _mm512_cvtepi32_epi16(_mm512_and_si512(packed, v16_mask));
                __m256i h1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(packed, 16));
                wf0 = _mm512_cvtph_ps(h0);
                wf1 = _mm512_cvtph_ps(h1);
            } else {
                __m512i x0 = _mm512_mullo_epi32(vst, vc0);
                __m512i x1 = _mm512_mullo_epi32(vst, vc1);
                x0 = _mm512_ternarylogic_epi32(x0, vmsk, vxor, 0x6A); // (x0 & msk) ^ xor
                x1 = _mm512_ternarylogic_epi32(x1, vmsk, vxor, 0x6A);
                __m512 f0a = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_and_si512(x0, v16_mask)));
                __m512 f0b = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32(x1, 16)));
                __m512 f1a = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32(x0, 16)));
                __m512 f1b = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_and_si512(x1, v16_mask)));
                wf0 = _mm512_add_ps(f0a, f0b);
                wf1 = _mm512_add_ps(f1a, f1b);
            }
            for (int m = 0; m < M; ++m) {
                __m512 xv0 = _mm512_loadu_ps(x_m + m * 256 + c * 32);
                __m512 xv1 = _mm512_loadu_ps(x_m + m * 256 + c * 32 + 16);
                acc[m] = _mm512_fmadd_ps(wf0, xv0, acc[m]);
                acc[m] = _mm512_fmadd_ps(wf1, xv1, acc[m]);
            }
        }
    }
    float s = 0.0f;
    for (int m = 0; m < M; ++m) s += _mm512_reduce_add_ps(acc[m]);
    return s;
}

template <int M, bool USE_LUT>
static void bench(const char* name, int n_threads, const uint8_t* big_buf, size_t n_experts, const uint32_t* lut) {
    constexpr size_t EXP_BYTES = 1843200; // 19,200 tiles of 96 B
    constexpr size_t EXP_TILES = EXP_BYTES / 96;
    alignas(64) float x_m[M * 256];
    for (int i = 0; i < M * 256; ++i) x_m[i] = 0.01f * ((i % 17) - 8);

    std::atomic<size_t> next_exp{0};
    std::atomic<float> sink{0.0f};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> thrs;
    for (int t = 0; t < n_threads; ++t) {
        thrs.emplace_back([&]() {
            float loc = 0.0f;
            for (;;) {
                size_t e = next_exp.fetch_add(1, std::memory_order_relaxed);
                if (e >= n_experts) break;
                loc += run_expert_slice<M, USE_LUT>(big_buf + e * EXP_BYTES, EXP_TILES, lut, x_m);
            }
            float cur = sink.load();
            while (!sink.compare_exchange_weak(cur, cur + loc)) {}
        });
    }
    for (auto& th : thrs) th.join();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double gb = (double(n_experts) * EXP_BYTES) / 1e9;
    double us_per_exp = (sec * 1e6) / n_experts;
    printf("%-32s | T=%2d M=%d | %5.1f us/expert | %5.2f GB/s (sink=%.1f)\n",
           name, n_threads, M, us_per_exp, gb / sec, sink.load());
}

int main() {
    printf("Building 4 MiB pmcgx E4M3 LUT...\n");
    uint32_t* lut = build_pmcgx_lut();
    // Allocate 1.8 GB (1000 experts) so it comes from real DDR5, NOT L3 cache!
    const size_t N_EXP = 512;
    const size_t BYTES = N_EXP * 1843200ull;
    uint8_t* buf = static_cast<uint8_t*>(aligned_alloc(4096, BYTES));
    for (size_t i = 0; i < BYTES; i += 4096) buf[i] = uint8_t(i >> 12);

    bench<1, true>("LUT (4MB L3-res, exact E4M3)", 8,  buf, N_EXP, lut);
    bench<1, true>("LUT (4MB L3-res, exact E4M3)", 16, buf, N_EXP, lut);
    bench<5, true>("LUT (4MB L3-res, exact E4M3)", 8,  buf, N_EXP, lut);
    bench<5, true>("LUT (4MB L3-res, exact E4M3)", 16, buf, N_EXP, lut);
    bench<1, false>("Arith AVX-512 (no E4M3)",     8,  buf, N_EXP, lut);
    bench<1, false>("Arith AVX-512 (no E4M3)",     16, buf, N_EXP, lut);
    bench<5, false>("Arith AVX-512 (no E4M3)",     8,  buf, N_EXP, lut);
    bench<5, false>("Arith AVX-512 (no E4M3)",     16, buf, N_EXP, lut);
    free(buf);
    free(lut);
    return 0;
}
