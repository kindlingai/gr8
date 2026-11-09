// Host test of the W-row wkv's activation quantizer (engram/act_quant.h, the lines k_act_quant_rows runs) against an independent implementation
// of the official act_quant (ref-official/inference/kernel.py: amax = max(|x|, 1e-4) per block of 32, scale = 2^ceil(log2(amax / 448)) by
// fast_round_scale, y = clamp(x / scale, -448, 448) rounded to e4m3 nearest-even). The reference uses double arithmetic and a table search for the
// rounding: nothing of act_quant.h and not the cuda_fp8 intrinsic. Host only: no GPU. What this does not cover: the attention code's k_act_quant itself (a GPU test of
// the kernel against this lives in test_engram_gpu.cu, not yet run) and the warp shuffle of the kernel (max is order independent).
//   test_act_quant [seed]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "act_quant.h"
#include "engram_math.h"
using namespace ds41::engram;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; if (g_fail <= 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

// every finite e4m3 value with its code (positive codes 0..126; 0x7F is NaN), from the format definition (bias 7, 3 mantissa bits, subnormal m * 2^-9)
struct Tab { double v[127]; Tab() { for (int c = 0; c < 127; ++c) { const int e = c >> 3, m = c & 7; v[c] = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7); } } };
static uint8_t ref_e4m3(double x, const Tab& t)   // nearest, ties to the even code; |x| clamped to 448 first (the reference clamps)
{
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    double a = std::fabs(x);
    if (a > 448.0) a = 448.0;
    int best = 0;
    for (int c = 1; c < 127; ++c)
    {
        const double d = std::fabs(t.v[c] - a), db = std::fabs(t.v[best] - a);
        if (d < db || (d == db && (c & 1) == 0)) best = c;
    }
    return (uint8_t)(sign | best);
}
static void ref_block(const float* x, const Tab& t, uint8_t* code, uint8_t* sb)
{
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    amax = std::max(amax, 1e-4f);
    const float sin = amax * 0.002232142857142857f;                 // fp32 product, as the kernel
    const int e = (int)std::ceil(std::log2((double)sin));            // 2^e >= sin, the smallest such power of two (log2 of an exact power is exact)
    *sb = (uint8_t)(e + 127);
    for (int i = 0; i < 32; ++i) code[i] = ref_e4m3(std::ldexp((double)x[i], -e), t);
}

int main(int argc, char** argv)
{
    const uint64_t seed = argc > 1 ? strtoull(argv[1], nullptr, 10) : 20261026ull;
    std::mt19937_64 rng(seed);
    const Tab tab;
    auto u01 = [&] { return (double)(rng() >> 11) * (1.0 / 9007199254740992.0); };
    long blocks = 0, hit_clamp = 0, hit_sub = 0, hit_boundary = 0;
    const int kinds = 9;
    for (long n = 0; n < 60000; ++n)
    {
        float x[32];
        const int kind = (int)(n % kinds);
        const int ex = (int)(rng() % 40) - 20;
        for (int i = 0; i < 32; ++i)
        {
            double v;
            switch (kind)
            {
            case 0: v = std::ldexp(u01() * 2.0 - 1.0, ex); break;                                    // uniform at a random scale
            case 1: { double g = 0; for (int k = 0; k < 12; ++k) g += u01(); v = std::ldexp(g - 6.0, ex); break; }   // near normal
            case 2: v = (i == 7 ? 1.0 : 0.01 * (u01() - 0.5)) * std::ldexp(1.0, ex); break;           // one outlier
            case 3: v = 0.0; break;                                                                   // all zero: the 1e-4 floor
            case 4: v = std::ldexp(u01() - 0.5, -12); break;                                          // below the floor: scale from 1e-4
            case 5: v = (i == 3 ? 448.0 : u01()) * std::ldexp(1.0, ex); break;                        // amax exactly 448 * 2^ex: a power of two scale
            case 6: v = (i == 3 ? 448.0 * (1.0 + 1e-7 * (double)(int)(rng() % 7)) : u01()) * std::ldexp(1.0, ex); break;   // amax a few ulp above / at the boundary
            case 7: v = (u01() < 0.5 ? -1.0 : 1.0) * std::ldexp(1.0 + (double)(rng() % 8) / 8.0 + 1.0 / 16.0, ex); break;   // ties between e4m3 values
            default: v = std::ldexp(u01() * 2.0 - 1.0, ex) * (rng() % 50 == 0 ? 3000.0 : 1.0); break;   // heavy outliers: saturation after scaling
            }
            x[i] = round_bf16((float)v);     // the kernel's input is bf16
        }
        uint8_t rc[32], rs, kc[32], ks;
        ref_block(x, tab, rc, &rs);
        const int e = aq_exp([&] { float a = 0.f; for (int i = 0; i < 32; ++i) a = std::fmax(a, std::fabs(x[i])); return a; }());
        ks = aq_scale_byte(e);
        for (int i = 0; i < 32; ++i) kc[i] = aq_code(x[i], e);
        CHECK(ks == rs, "kind %d block %ld: scale byte %u, reference %u", kind, n, ks, rs);
        for (int i = 0; i < 32; ++i) CHECK(kc[i] == rc[i], "kind %d block %ld elem %d (x=%.9g): code 0x%02x, reference 0x%02x", kind, n, i, x[i], kc[i], rc[i]);
        ++blocks;
        for (int i = 0; i < 32; ++i) { if ((rc[i] & 0x7F) == 126) ++hit_clamp; if ((rc[i] & 0x7F) < 8 && (rc[i] & 0x7F) > 0) ++hit_sub; }
        if (kind == 5 || kind == 6) ++hit_boundary;
    }
    printf("act_quant: %ld blocks (%ld codes at the 448 end, %ld subnormal codes, %ld boundary blocks), seed %llu\n", blocks, hit_clamp, hit_sub, hit_boundary, (unsigned long long)seed);
    printf("%s (seed %llu, %d failed checks)\n", g_fail ? "FAIL" : "PASS", (unsigned long long)seed, g_fail);
    return g_fail ? 1 : 0;
}
