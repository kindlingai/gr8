// Shared by the KDA tests: gates, one layer's host weights from the dense export, the f_a/g_a/b projections in
// double, the loader's head rule.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "st.h"
#include "../math.h"
#include "../../model_geom.h"

static_assert(G6_MODEL_FLASH, "KDA tests are GLM 5.3 Flash only (HID 4096): build with -DG6_MODEL_FLASH=1");

namespace kda_test
{
using namespace kda;
constexpr int HID_ = geom::HID;

inline int g_fail = 0;

inline double rel(const float* a, const float* b, size_t n)
{
    double d = 0, r = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const double e = static_cast<double>(a[i]) - b[i];
        d += e * e;
        r += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(d / (r > 0 ? r : 1));
}
inline void gate(const std::string& name, double r, double g)
{
    const bool ok = r <= g;
    if (!ok) ++g_fail;
    printf("  %-40s rel-L2 %.3e  gate %.0e  %s\n", name.c_str(), r, g, ok ? "PASS" : "FAIL");
}
inline void same(const std::string& name, const std::vector<float>& a, const std::vector<float>& b)
{
    const bool ok = a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * 4);
    if (!ok) ++g_fail;
    printf("  %-40s bit-identical: %s\n", name.c_str(), ok ? "PASS" : "FAIL");
}

// one layer's weights, all heads (host, from the dense export file)
struct W
{
    st::File f;
    const uint16_t *fa, *ga, *b, *fb, *gb, *o, *onorm, *conv[3];
    const float *A_log, *dt_bias;
    void load(const std::string& dir, int l)
    {
        char p[64];
        snprintf(p, sizeof p, "/exlr8-dense-%03d.safetensors", l);
        if (!f.open(dir + p)) { fprintf(stderr, "cannot open %s%s\n", dir.c_str(), p); exit(2); }
        fa = f["self_attn.f_a_proj.bf16"].u16(); ga = f["self_attn.g_a_proj.bf16"].u16();
        b = f["self_attn.b_proj.bf16"].u16(); fb = f["self_attn.f_b_proj.bf16"].u16();
        gb = f["self_attn.g_b_proj.bf16"].u16(); o = f["self_attn.o_proj.bf16"].u16();
        onorm = f["self_attn.o_norm.weight"].u16();
        conv[0] = f["self_attn.q_conv1d.bf16"].u16(); conv[1] = f["self_attn.k_conv1d.bf16"].u16();
        conv[2] = f["self_attn.v_conv1d.bf16"].u16();
        A_log = f["self_attn.A_log"].f32(); dt_bias = f["self_attn.dt_bias"].f32();
    }
};

// x @ [f_a; g_a; b]^T in double for T rows -> [T][320]
inline std::vector<float> small_proj(const W& w, const float* x, int T)
{
    std::vector<float> y(static_cast<size_t>(T) * 320);
#pragma omp parallel for
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < 320; ++r)
        {
            const uint16_t* row = r < 128 ? w.fa + static_cast<size_t>(r) * HID_ : r < 256 ? w.ga + static_cast<size_t>(r - 128) * HID_
                                                                                           : w.b + static_cast<size_t>(r - 256) * HID_;
            double d = 0;
            for (int i = 0; i < HID_; ++i) d += static_cast<double>(bf2f(row[i])) * x[static_cast<size_t>(t) * HID_ + i];
            y[static_cast<size_t>(t) * 320 + r] = static_cast<float>(d);
        }
    return y;
}

inline void head_rows(int l, int nr, int r, int& h0, int& nh)
{
    std::vector<int> w(nr);
    for (int j = 0; j < nr; ++j) w[j] = HEADS / nr + (j < HEADS % nr);
    const int j = (r + nr - l % nr) % nr;
    h0 = 0;
    for (int i = 0; i < j; ++i) h0 += w[i];
    nh = w[j];
}

}   // namespace kda_test
