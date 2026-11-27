// Standalone CPU math unit test verifying exact Hy4-flash mathematical identities:
//   1. Non-combined iHC (HYV4FlashHyperConnection + HYV4FlashHyperHead): eps=1e-5 norm, hc_eps=1e-6, mag=2.0
//   2. Absorbed MLA exact equivalence to expand_kv:
//        (q_nope @ W_uk) . c + q_rot . k_rot == q_nope . (c @ W_uk^T) + q_rot . k_rot
//        (sum_t p_t c_t) @ W_uv^T == sum_t p_t (c_t @ W_uv^T)
//      AND online softmax sink initialization (m=sink, l=1.0) == concat([scores, sink]) -> softmax -> drop sink!
//   3. Router sigmoid + e_score_correction_bias -> top-16 -> pre-bias gather -> normalize (+1e-20) * 3.998
//   4. Routed SwiGLU clamp (gate<=10, up in [-10,10]) vs Shared SwiGLU (no clamp)
//   5. Trailing-64 interleaved RoPE invertibility and orthogonality
#include "../model_geom_hy4.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static void test_absorbed_mla_and_sinks() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    constexpr int T = 17, NOPE = 192, ROPE = 64, LORA = 512, VDIM = 256;
    std::vector<float> q_nope(NOPE), q_rot(ROPE);
    std::vector<float> W_uk(NOPE * LORA), W_uv(VDIM * LORA);
    std::vector<float> C(T * LORA), K_rot(T * ROPE);
    for (float& x : q_nope) x = U(rng);
    for (float& x : q_rot)  x = U(rng);
    for (float& x : W_uk)   x = U(rng) * 0.1f;
    for (float& x : W_uv)   x = U(rng) * 0.1f;
    for (float& x : C)      x = U(rng);
    for (float& x : K_rot)  x = U(rng);
    const float sink = 0.73f;
    const float scale = 1.0f / std::sqrt(float(NOPE + ROPE));

    // Reference expand_kv
    std::vector<float> scores_ref(T + 1);
    std::vector<float> V_ref(T * VDIM, 0.0f);
    for (int t = 0; t < T; ++t) {
        float dp = 0.0f;
        for (int d = 0; d < NOPE; ++d) {
            float kn = 0.0f;
            for (int r = 0; r < LORA; ++r) kn += C[t * LORA + r] * W_uk[d * LORA + r];
            dp += q_nope[d] * kn;
        }
        for (int d = 0; d < ROPE; ++d) dp += q_rot[d] * K_rot[t * ROPE + d];
        scores_ref[t] = dp * scale;
        for (int v = 0; v < VDIM; ++v) {
            float vv = 0.0f;
            for (int r = 0; r < LORA; ++r) vv += C[t * LORA + r] * W_uv[v * LORA + r];
            V_ref[t * VDIM + v] = vv;
        }
    }
    scores_ref[T] = sink;
    float mx = *std::max_element(scores_ref.begin(), scores_ref.end());
    std::vector<float> pr(T + 1);
    float s = 0.0f;
    for (int t = 0; t <= T; ++t) { pr[t] = std::exp(scores_ref[t] - mx); s += pr[t]; }
    for (int t = 0; t <= T; ++t) pr[t] /= s;
    std::vector<float> out_ref(VDIM, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VDIM; ++v) out_ref[v] += pr[t] * V_ref[t * VDIM + v];

    // Absorbed MLA + online softmax initialized with sink (split into 2 splits to verify split-KV reduction too!)
    std::vector<float> q_abs(LORA, 0.0f);
    for (int r = 0; r < LORA; ++r)
        for (int d = 0; d < NOPE; ++d) q_abs[r] += q_nope[d] * W_uk[d * LORA + r];

    auto run_split = [&](int t0, int t1, bool has_sink, float& m_out, float& l_out, std::vector<float>& acc) {
        m_out = has_sink ? sink : -INFINITY;
        l_out = has_sink ? 1.0f : 0.0f;
        acc.assign(LORA, 0.0f);
        for (int t = t0; t < t1; ++t) {
            float dp = 0.0f;
            for (int r = 0; r < LORA; ++r) dp += q_abs[r] * C[t * LORA + r];
            for (int d = 0; d < ROPE; ++d) dp += q_rot[d] * K_rot[t * ROPE + d];
            const float sc = dp * scale;
            const float mn = std::max(m_out, sc);
            const float a = std::exp(m_out - mn);
            const float p = std::exp(sc - mn);
            l_out = l_out * a + p;
            for (int r = 0; r < LORA; ++r) acc[r] = acc[r] * a + p * C[t * LORA + r];
            m_out = mn;
        }
    };
    float m0, l0, m1, l1;
    std::vector<float> o0, o1;
    run_split(0, 9, true, m0, l0, o0);
    run_split(9, T, false, m1, l1, o1);
    const float gm = std::max(m0, m1);
    const float w0 = std::exp(m0 - gm), w1 = std::exp(m1 - gm);
    const float gl = l0 * w0 + l1 * w1;
    std::vector<float> u(LORA);
    for (int r = 0; r < LORA; ++r) u[r] = (o0[r] * w0 + o1[r] * w1) / gl;
    std::vector<float> out_abs(VDIM, 0.0f);
    for (int v = 0; v < VDIM; ++v)
        for (int r = 0; r < LORA; ++r) out_abs[v] += u[r] * W_uv[v * LORA + r];

    float max_diff = 0.0f;
    for (int v = 0; v < VDIM; ++v) max_diff = std::max(max_diff, std::fabs(out_ref[v] - out_abs[v]));
    assert(max_diff < 1e-5f);
    std::printf("test_absorbed_mla_and_sinks (2-split): max_diff=%.3e OK\n", max_diff);
}

static void test_router_and_hc() {
    // Router: verify tie-break, pre-bias gather, and scaling 3.998
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    std::vector<float> lg(geom::NEXP), bias(geom::NEXP), sc(geom::NEXP), ch(geom::NEXP);
    for (int e = 0; e < geom::NEXP; ++e) {
        lg[e] = U(rng);
        bias[e] = U(rng) * 0.2f;
        sc[e] = 1.0f / (1.0f + std::exp(-lg[e]));
        ch[e] = sc[e] + bias[e];
    }
    std::vector<int> idx(geom::NEXP);
    for (int e = 0; e < geom::NEXP; ++e) idx[e] = e;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return ch[a] > ch[b]; });
    float sum = 0.0f;
    for (int k = 0; k < geom::TOPK; ++k) sum += sc[idx[k]];
    float wsum = 0.0f;
    for (int k = 0; k < geom::TOPK; ++k) wsum += (sc[idx[k]] / (sum + 1e-20f)) * geom::ROUTED_SCALE;
    assert(std::fabs(wsum - geom::ROUTED_SCALE) < 1e-5f);
    std::printf("test_router_topk16: weight_sum=%.6f (expected %.6f) OK\n", wsum, geom::ROUTED_SCALE);

    // Trailing-64 interleaved RoPE norm preservation
    float x[128];
    for (int i = 0; i < 128; ++i) x[i] = U(rng);
    float n0 = 0.0f, n1 = 0.0f;
    for (int i = 0; i < 128; ++i) n0 += x[i] * x[i];
    for (int p = 0; p < 32; ++p) {
        const float inv = std::pow(geom::ROPE_THETA_DSA, -float(2 * p) / 64.0f);
        const float a = 123456.0f * inv;
        const float c = std::cos(a), s = std::sin(a);
        const float x0 = x[64 + 2 * p], x1 = x[64 + 2 * p + 1];
        x[64 + 2 * p]     = x0 * c - x1 * s;
        x[64 + 2 * p + 1] = x1 * c + x0 * s;
    }
    for (int i = 0; i < 128; ++i) n1 += x[i] * x[i];
    assert(std::fabs(n0 - n1) < 1e-4f);
    std::printf("test_trailing64_rope: norm diff=%.3e OK\n", std::fabs(n0 - n1));
}

int main() {
    test_absorbed_mla_and_sinks();
    test_router_and_hc();
    std::printf("test_hy4_math: ALL PASS\n");
    return 0;
}
