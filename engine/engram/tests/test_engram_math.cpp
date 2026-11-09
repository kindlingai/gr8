// Host test of engram_math.h against the official ParallelEngramEmbedding (dequant) and Engram.forward (gate).
//   test_engram_math <vectors_math.bin>      (written by engine/engram/gen_engram_vectors.py)
// Dequant must be bit-exact. The gate sums are fp32 and the reduction order differs from torch's, so the gate is compared to
// 2e-5 absolute and the output row to 1 bf16 ulp.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "engram/engram_math.h"
using namespace ds41::engram;

struct HostMath {
    static float rsqrt(float x) { return 1.f / std::sqrt(x); }
    static float sqrt(float x) { return std::sqrt(x); }
    static float exp(float x) { return std::exp(x); }
};
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)
static int ordered(uint16_t b) { return (b & 0x8000) ? -(int)(b & 0x7FFF) : (int)b; }

template <class T> static std::vector<T> rd(FILE* f, size_t n)
{
    std::vector<T> v(n);
    if (fread(v.data(), sizeof(T), n, f) != n) { printf("short read\n"); exit(2); }
    return v;
}

int main(int argc, char** argv)
{
    if (argc != 2) { printf("usage: %s vectors_math.bin\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 2; }
    uint32_t N, R;
    if (fread(&N, 4, 1, f) != 1 || fread(&R, 4, 1, f) != 1) return 2;
    auto w = rd<uint8_t>(f, (size_t)N * 256), s = rd<uint8_t>(f, (size_t)N * 8);
    auto deq = rd<uint16_t>(f, (size_t)N * 256);
    auto hh = rd<uint16_t>(f, (size_t)R * 4 * kHid), kv = rd<uint16_t>(f, (size_t)R * kKvDim);
    auto q = rd<float>(f, 4 * kHid), k = rd<float>(f, 4 * kHid), gref = rd<float>(f, (size_t)R * 4);
    auto out = rd<uint16_t>(f, (size_t)R * 4 * kHid);
    fclose(f);

    // dequant: every element, bit-exact
    size_t bad = 0;
    for (uint32_t r = 0; r < N; ++r)
        for (int c = 0; c < 256; ++c) {
            const uint16_t g = dequant1(w[(size_t)r * 256 + c], s[(size_t)r * 8 + (c >> 5)]);
            if (g != deq[(size_t)r * 256 + c]) {
                if (bad++ < 5) printf("  dequant mismatch row %u col %d: w %02x s %02x got %04x want %04x\n", r, c, w[(size_t)r * 256 + c], s[(size_t)r * 8 + (c >> 5)], g, deq[(size_t)r * 256 + c]);
            }
        }
    CHECK(bad == 0, "%zu of %u dequant elements differ", bad, N * 256);

    // gate + output row
    float max_gate_err = 0.f; int max_ulp = 0; size_t over = 0;
    std::vector<float> wgt(4 * kHid);
    for (int i = 0; i < 4 * kHid; ++i) wgt[i] = q[i] * k[i];
    for (uint32_t r = 0; r < R; ++r) {
        const uint16_t* kvr = &kv[(size_t)r * kKvDim];
        for (int j = 0; j < 4; ++j) {
            const uint16_t* h = &hh[((size_t)r * 4 + j) * kHid];
            float sh = 0.f, sk = 0.f, shk = 0.f;
            for (int i = 0; i < kHid; ++i) {
                const float hv = bf16_to_f32(h[i]), kk = bf16_to_f32(kvr[j * kHid + i]);
                sh += hv * hv; sk += kk * kk; shk += (hv * wgt[j * kHid + i]) * kk;
            }
            const float g = gate_from_sums<HostMath>(sh, sk, shk, 1e-20f);
            const float ge = std::fabs(g - gref[r * 4 + j]);
            if (ge > max_gate_err) max_gate_err = ge;
            CHECK(ge <= 2e-5f, "gate row %u stream %d: got %.9g want %.9g", r, j, g, gref[r * 4 + j]);
            for (int i = 0; i < kHid; ++i) {
                const float t = bf16_to_f32(h[i]), val = bf16_to_f32(kvr[4 * kHid + i]);
                const uint16_t o = f32_to_bf16(t + g * val);          // two rounded fp32 ops, no FMA (-ffp-contract=off)
                const int d = std::abs(ordered(o) - ordered(out[((size_t)r * 4 + j) * kHid + i]));
                if (d > max_ulp) max_ulp = d;
                if (d > 1) ++over;
            }
        }
    }
    CHECK(over == 0, "%zu output elements differ by more than 1 bf16 ulp (max %d)", over, max_ulp);
    if (fails) { printf("FAILED %d checks\n", fails); return 1; }
    printf("PASS engram_math: %u x 256 dequant elements bit-exact; %u rows x 4 streams gate max abs err %.3g, output max %d bf16 ulp\n", N, R, max_gate_err, max_ulp);
    return 0;
}
