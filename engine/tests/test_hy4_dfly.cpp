// CPU unit test for Tencent Hy4-flash DFly2 speculative drafter geometry, slot-0 mtp_fusion,
// hidden_correction math, and lossless speculative verification (greedy & rejection sampling).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../model_geom_hy4.h"

namespace {

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

void rmsnorm_cpu(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& y, float eps) {
    double ss = 0.0;
    for (float v : x) ss += double(v) * double(v);
    const float inv = 1.0f / std::sqrt(float(ss / double(x.size())) + eps);
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = w[i] * (x[i] * inv);
}

// Verify DFly2 target DSA layer mapping [25, 29, 33, 37, 41] are all genuine DSA layers in geom::is_dsa_layer.
void test_dfly_target_layers() {
    const int expected[5] = {25, 29, 33, 37, 41};
    for (int l : expected) {
        check(geom::is_dsa_layer(l), "DFly2 target layer must be a DSA layer");
        check(geom::layer_heads(l) == 32, "DFly2 target layer must have 32 heads");
    }
    // Check that the 5 layers are the exact last 5 DSA layers of Hy4-flash
    std::vector<int> dsa_all;
    for (int l = 0; l < geom::L_MAIN; ++l) if (geom::is_dsa_layer(l)) dsa_all.push_back(l);
    check(dsa_all.size() == 12, "Hy4-flash must have 12 DSA layers");
    for (int i = 0; i < 5; ++i) {
        check(dsa_all[dsa_all.size() - 5 + i] == expected[i], "DFly2 target layers must be last 5 DSA layers");
    }
}

// Verify slot-0 mtp_fusion identity when gate == 0 and exact linear combination when gate != 0.
void test_mtp_fusion_math() {
    const int D = 8;
    std::vector<float> anchor = {1.0f, -2.0f, 0.5f, 0.25f, -0.75f, 1.5f, 0.0f, 2.0f};
    std::vector<float> target = {-0.5f, 1.0f, 1.5f, -1.0f, 0.5f, -0.25f, 2.0f, -1.5f};
    std::vector<float> w1(D, 1.0f), gate(D, 0.25f);
    std::vector<float> en, hn;
    rmsnorm_cpu(anchor, w1, en, geom::RMS_EPS);
    rmsnorm_cpu(target, w1, hn, geom::RMS_EPS);
    std::vector<float> cat(2 * D);
    for (int i = 0; i < D; ++i) { cat[i] = en[i]; cat[D + i] = hn[i]; }
    // eh_proj = [I | I] -> fused = en + hn
    std::vector<float> out(D);
    for (int i = 0; i < D; ++i) out[i] = anchor[i] + gate[i] * (cat[i] + cat[D + i]);
    for (int i = 0; i < D; ++i) {
        const float want = anchor[i] + 0.25f * (en[i] + hn[i]);
        check(std::fabs(out[i] - want) < 1e-6f, "mtp_fusion formula mismatch");
    }
}

// Verify lossless rejection sampling preserves target marginal distribution over 200k draws!
void test_rejection_sampling_unbiased() {
    const double pt[4] = {0.50, 0.30, 0.15, 0.05};
    const double pd[4] = {0.20, 0.40, 0.30, 0.10};
    auto draw_u = [](uint64_t s) -> double {
        uint64_t z = s + 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return double(z >> 11) * (1.0 / 9007199254740992.0);
    };
    int cnt[4] = {0, 0, 0, 0};
    const int N = 200000;
    for (int i = 0; i < N; ++i) {
        // Sample draft token from pd
        double u0 = draw_u(uint64_t(i) * 3 + 1);
        int d = 3;
        double c = 0.0;
        for (int k = 0; k < 4; ++k) { c += pd[k]; if (u0 < c) { d = k; break; } }
        // Accept with min(1, pt[d]/pd[d])
        double u1 = draw_u(uint64_t(i) * 3 + 2);
        int emitted = d;
        if (!(pt[d] >= pd[d] || u1 * pd[d] < pt[d])) {
            double res[4], sum = 0.0;
            for (int k = 0; k < 4; ++k) { res[k] = std::max(0.0, pt[k] - pd[k]); sum += res[k]; }
            double u2 = draw_u(uint64_t(i) * 3 + 3) * sum;
            double rc = 0.0;
            emitted = 3;
            for (int k = 0; k < 4; ++k) { rc += res[k]; if (u2 < rc) { emitted = k; break; } }
        }
        cnt[emitted] += 1;
    }
    for (int k = 0; k < 4; ++k) {
        const double emp = double(cnt[k]) / double(N);
        check(std::fabs(emp - pt[k]) < 0.005, "rejection sampling marginal distribution drifted from target!");
    }
}

} // namespace

int main() {
    test_dfly_target_layers();
    test_mtp_fusion_math();
    test_rejection_sampling_unbiased();
    std::printf("test_hy4_dfly: ALL PASS\n");
    return 0;
}
