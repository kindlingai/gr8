// The sampler's uniform draw, CPU only (no GPU, no launch): for every 24-bit k, u01_top24() (glue.cuh), which gumbel() and
// uniform01() use, gives 0 < u < 1 and a finite Gumbel noise -log(-log(u)), and the same bits as the form it replaced
// wherever that form was below 1. Only k = 0xFFFFFF changes: the old form gave u = 1.0 (g = +inf), the new one 1 - 2^-24.
// build + run (scripts/build.sh does): nvcc -O2 -std=c++17 $GEN -I engine/kernels -o sampler_u01_test engine/kernels/tests/sampler_u01_test.cu
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "../glue.cuh"

using glue::u01_top24;

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { if (fails < 10) fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static float u01_before(uint32_t x) { return ((float)(x >> 8) + 0.5f) * (1.f / 16777216.f); }   // main before this change
static bool same_bits(float a, float b) { uint32_t i, j; memcpy(&i, &a, 4); memcpy(&j, &b, 4); return i == j; }

int main()
{
    uint32_t changed = 0;
    for (uint32_t k = 0; k < (1u << 24); ++k)
        for (uint32_t low : {0u, 0xFFu})   // the 8 dropped bits do not matter
        {
            const uint32_t x = k << 8 | low;
            const float u = u01_top24(x), was = u01_before(x);
            EXPECT(u > 0.f && u < 1.f);
            EXPECT(std::isfinite(-logf(-logf(u))));
            if (was < 1.f) EXPECT(same_bits(u, was));
            else ++changed;
        }
    EXPECT(changed == 2);   // k = 0xFFFFFF, both low words
    EXPECT(u01_before(0xFFFFFFFFu) == 1.f && std::isinf(-logf(-logf(u01_before(0xFFFFFFFFu)))));
    EXPECT(u01_top24(0xFFFFFFFFu) == 1.f - 1.f / 16777216.f);
    if (fails) { fprintf(stderr, "sampler_u01_test: %d FAILED\n", fails); return 1; }
    printf("sampler_u01_test: PASS (every 24-bit draw in (0,1) with finite noise; only k = 0xFFFFFF changed, 1.0 -> 1 - 2^-24)\n");
    return 0;
}
