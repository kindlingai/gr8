// Host test of the parts of nvfp4.cuh / ink2_router.cuh / ink2_moe.cuh that are one body for the CPU and the GPU:
//   1. the exact scalar functions against nv4pack and ink2ref (E2M1, E4M3, both NVFP4 block rules, MXFP8, SwiGLU)
//   2. the operand maps (nv4::Frag, ink2::Mx8Frag): the real load code runs on the CPU and a model of the two
//      block-scaled mma instructions (the layout the headers state) must give the dot products of the decoded record.
//      The GPU probe tests/nvfp4_mma_probe.cu checks the same thing against the hardware.
//   3. the router's pick against ink2ref::Model::route, bit for bit, with exact ties
//   4. the plan: every (token, route) is covered once, nothing out of bounds, slots ascending
// Build: g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off (scripts/host_tests.sh). No GPU.
#include "../ink2_moe.cuh"
#include "../../ref/ink2/ink2_cpu_ref.h"
#include <cstdio>
#include <cstdlib>
#include <random>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (++fails <= 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static std::mt19937_64 rng(20261112);
static float urand(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(rng() >> 40) * (1.0f / 16777216.0f); }
static float nrand() { static std::normal_distribution<float> d(0.0f, 1.0f); return d(rng); }
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------------------------------------- 1. scalars
static void test_scalars()
{
    for (int b = 0; b < 256; ++b)
    {
        const float r = nv4pack::e4m3_to_f32(static_cast<uint8_t>(b)), v = nv4::e4m3_val(static_cast<uint32_t>(b));
        CHECK((std::isnan(r) && std::isnan(v)) || bits(r) == bits(v), "e4m3_val %d", b);
        if (!std::isnan(r)) CHECK(nv4::e4m3_rne(r) == (r == 0.0f ? 0u : static_cast<uint32_t>(b)), "e4m3_rne of grid value %d", b);
    }
    for (int c = 0; c < 16; ++c) CHECK(bits(nv4pack::e2m1(c)) == bits(nv4::e2m1_val(c)), "e2m1_val %d", c);
    const auto one = [](float v) {
        CHECK(nv4::e4m3_rne(v) == nv4pack::f32_to_e4m3_rne(v), "e4m3_rne(%a) = %u, reference %u", v, nv4::e4m3_rne(v), nv4pack::f32_to_e4m3_rne(v));
        CHECK(nv4::e2m1_rne(v) == nv4pack::f32_to_e2m1_rne(v), "e2m1_rne(%a)", v);
    };
    // every midpoint and grid value of both formats, with its neighbours
    for (int b = 0; b < 127; ++b)
    {
        const float lo = nv4pack::e4m3_to_f32(static_cast<uint8_t>(b)), hi = b == 126 ? 480.0f : nv4pack::e4m3_to_f32(static_cast<uint8_t>(b + 1));
        for (float v : {lo, hi, 0.5f * (lo + hi)})
            for (int d = -2; d <= 2; ++d)
            {
                const float w = from_bits(bits(v) + static_cast<uint32_t>(d));
                one(w);
                one(-w);
            }
    }
    for (float v = 0.0f; v <= 7.0f; v += 0.125f)
        for (int d = -2; d <= 2; ++d) { one(from_bits(bits(v) + static_cast<uint32_t>(d))); one(-from_bits(bits(v) + static_cast<uint32_t>(d))); }
    for (float v : {0.0f, -0.0f, FLT_MIN, FLT_TRUE_MIN, 1e-30f, 448.0f, 449.0f, 1e30f, std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        one(v);
    for (int i = 0; i < 20000000; ++i) one(from_bits(static_cast<uint32_t>(rng())));
    for (int i = 0; i < 2000000; ++i) one(urand(-8.0f, 8.0f));

    // 16-bit cells and BF16 rounding
    for (uint32_t h = 0; h < 65536; ++h) CHECK(bits(nv4::cell16(h, 1)) == bits(ink2ref::bf16_to_f32(static_cast<uint16_t>(h))), "bf16 cell");
    CHECK(nv4::cell16(0x3c00, 0) == 1.0f && nv4::cell16(0xc400, 0) == -4.0f && nv4::cell16(0x0001, 0) == std::ldexp(1.0f, -24), "fp16 cell");
    for (int i = 0; i < 2000000; ++i)
    {
        const float v = from_bits(static_cast<uint32_t>(rng()));
        CHECK(nv4::bf16_rne(v) == ink2ref::f32_to_bf16(v), "bf16_rne(%a)", v);
        const float g = urand(-30.0f, 30.0f), u = urand(-80.0f, 80.0f);
        CHECK(ink2::swiglu(g, u) == ink2ref::f32_to_bf16(ink2ref::Model::act(g, u)), "swiglu(%a, %a)", g, u);
    }
}

// A row of K values in a mix of magnitudes, with the cases the rules care about.
static std::vector<float> make_row(int K, int kind)
{
    std::vector<float> x(static_cast<size_t>(K));
    const float mag = std::ldexp(1.0f, static_cast<int>(rng() % 24) - 12);
    for (int k = 0; k < K; ++k) x[k] = ink2ref::rbf(nrand() * mag);
    if (kind == 1) std::fill(x.begin(), x.end(), 0.0f);
    if (kind == 2)   // spikes: a block with one large value takes the / 4 candidate or not
        for (int b = 0; b < K / 16; b += 2) x[b * 16 + static_cast<int>(rng() % 16)] = ink2ref::rbf(mag * urand(3.0f, 40.0f)) * (rng() & 1 ? 1.0f : -1.0f);
    if (kind == 3)
        for (int k = 0; k < K; ++k) x[k] = from_bits(static_cast<uint32_t>(rng() % 0x00800000u) | (rng() & 1 ? 0x80000000u : 0u));   // subnormals
    if (kind == 4)
        for (int k = 0; k < K; ++k) x[k] = (k % 16 < 3) ? 0.0f : x[k];
    if (kind == 5)
        for (int k = 0; k < K; ++k) x[k] = ink2ref::rbf(urand(-448.0f, 448.0f) * (k % 7 == 0 ? 1e6f : 1.0f));
    if (kind == 6)
        for (int k = 16; k < 48 && k < K; ++k) x[k] = 0.0f;   // whole zero blocks in a live row
    return x;
}

static void test_quantizers()
{
    for (int it = 0; it < 6000; ++it)
    {
        const int K = 64 * (1 + static_cast<int>(rng() % 4));
        const std::vector<float> x = make_row(K, it % 8);
        for (int fs = 0; fs < 2; ++fs)
        {
            const nv4pack::ScaleRule rule = fs ? nv4pack::ScaleRule::FourOverSixMae : nv4pack::ScaleRule::Static6;
            std::vector<uint8_t> rc(static_cast<size_t>(K) / 2), rs(static_cast<size_t>(K) / 16);
            const float router = nv4pack::quantize_row(x.data(), K, rule, rc.data(), rs.data());
            float amax = 0.0f;
            for (int k = 0; k < K; ++k) amax = fmaxf(amax, fabsf(x[k]));
            const float outer = nv4::outer_of(amax, fs != 0);
            CHECK(bits(outer) == bits(router), "outer scale, K %d rule %d", K, fs);
            for (int b = 0; b < K / 16; ++b)
            {
                uint32_t s;
                uint64_t q;
                nv4::quant_block(&x[static_cast<size_t>(b) * 16], outer, fs != 0, s, q);
                uint64_t rq;
                memcpy(&rq, &rc[static_cast<size_t>(b) * 8], 8);
                CHECK(s == rs[b] && q == rq, "NVFP4 block: iteration %d kind %d rule %d block %d", it, it % 8, fs, b);
            }
        }
        // a fixed outer scale, as a caller may pass
        {
            const float fixed = it % 3 == 0 ? 0.0f : std::ldexp(1.0f, static_cast<int>(rng() % 20) - 14);
            std::vector<uint8_t> rc(static_cast<size_t>(K) / 2), rs(static_cast<size_t>(K) / 16);
            nv4pack::quantize(x.data(), 1, K, nv4pack::ScaleRule::FourOverSixMae, &fixed, rc.data(), rs.data());
            for (int b = 0; b < K / 16; ++b)
            {
                uint32_t s;
                uint64_t q, rq;
                nv4::quant_block(&x[static_cast<size_t>(b) * 16], fixed, true, s, q);
                memcpy(&rq, &rc[static_cast<size_t>(b) * 8], 8);
                CHECK(s == rs[b] && q == rq, "NVFP4 block under a fixed outer scale");
            }
        }
        // MXFP8
        std::vector<uint8_t> mc(static_cast<size_t>(K)), msc(static_cast<size_t>(K) / 32);
        ink2ref::mx8::quantize(x.data(), static_cast<size_t>(K) / 32, mc.data(), msc.data());
        for (int b = 0; b < K / 32; ++b)
        {
            float amax = FLT_MIN;
            for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[static_cast<size_t>(b) * 32 + j]));
            const uint32_t s = ink2::mx8_scale(amax);
            CHECK(s == msc[b], "MXFP8 scale %u, reference %u (amax %a)", s, msc[b], amax);
            for (int j = 0; j < 32; ++j)
                CHECK(ink2::mx8_code(x[static_cast<size_t>(b) * 32 + j], s) == mc[static_cast<size_t>(b) * 32 + j], "MXFP8 code");
        }
    }
    // the scale at exact powers of two and next to them, over the whole exponent range
    for (int e = -149; e < 128; ++e)
        for (int d = -1; d <= 1; ++d)
        {
            const float a = fmaxf(from_bits(bits(std::ldexp(1.0f, e)) + static_cast<uint32_t>(d)), FLT_MIN);
            std::vector<float> x(32, 0.0f);
            x[5] = a;
            uint8_t c[32], s;
            ink2ref::mx8::quantize(x.data(), 1, c, &s);
            CHECK(ink2::mx8_scale(a) == s, "MXFP8 scale at %a", a);
        }
}

// ---------------------------------------------------------------------------------------------- 2. operand maps
static unsigned nib(uint32_t reg, int i, int j) { return (reg >> (8 * i + 4 * j)) & 15u; }
static unsigned byt(uint32_t reg, int i) { return (reg >> (8 * i)) & 255u; }

static void test_nv4_map()
{
    {   // xrec_byte is a bijection of the 32 code pairs
        bool seen[32] = {};
        for (int k = 0; k < 64; k += 2)
        {
            const int o = nv4::Frag::xrec_byte(k);
            CHECK(o >= 0 && o < 32 && !seen[o] && nv4::Frag::xrec_byte(k + 1) == o, "nv4 xrec_byte(%d)", k);
            if (o >= 0 && o < 32) seen[o] = true;
        }
    }
    for (int it = 0; it < 200; ++it)
    {
        // weights: 16 rows x 64 k, random codes and block scales in a narrow exponent range (exact sums in fp64)
        uint8_t wc[16][32], ws[16][4], rec[nv4pack::GR];
        for (auto& r : wc) for (uint8_t& b : r) b = static_cast<uint8_t>(rng());
        for (auto& r : ws) for (uint8_t& b : r) b = static_cast<uint8_t>(0x30 + rng() % 0x20);
        nv4pack::pack_group(rec, &wc[0][0], &ws[0][0], 64, 0, 0);
        float W[16][64];
        nv4pack::decode_group(rec, 1.0f, W);
        // activations: 8 columns
        uint8_t xc[8][32], xsc[8][4], xrec[8][32] = {};
        uint32_t xs[8];
        float X[8][64];
        for (int m = 0; m < 8; ++m)
        {
            for (uint8_t& b : xc[m]) b = static_cast<uint8_t>(rng());
            for (uint8_t& b : xsc[m]) b = static_cast<uint8_t>(0x30 + rng() % 0x20);
            nv4pack::dequantize(xc[m], xsc[m], 4, 1.0f, X[m]);
            for (int k = 0; k < 64; ++k)
                xrec[m][nv4::Frag::xrec_byte(k)] |= static_cast<uint8_t>(((xc[m][k / 2] >> (4 * (k & 1))) & 15u) << (4 * (k & 1)));
            memcpy(&xs[m], xsc[m], 4);
        }
        // the instruction as nvfp4.cuh states it
        unsigned wa[16][64], xb[8][64];
        uint32_t sa[16], sb[8];
        for (int lane = 0; lane < 32; ++lane)
        {
            const int m = lane >> 2, t = lane & 3;
            const nv4::Frag::A a = nv4::Frag::load(rec, lane);
            uint32_t b0, b1;
            nv4::Frag::loadB(xrec[m], t, b0, b1);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j)
                {
                    const int k = 8 * t + 2 * i + j;
                    wa[m][k] = nib(a.a0, i, j);
                    wa[m + 8][k] = nib(a.a1, i, j);
                    wa[m][32 + k] = nib(a.a2, i, j);
                    wa[m + 8][32 + k] = nib(a.a3, i, j);
                    xb[m][k] = nib(b0, i, j);
                    xb[m][32 + k] = nib(b1, i, j);
                }
            if (t == 0) { sa[m] = a.s; sb[m] = xs[m]; }
            if (t == 1) sa[m + 8] = a.s;
        }
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 8; ++c)
            {
                double got = 0.0, want = 0.0;
                for (int k = 0; k < 64; ++k)
                {
                    got += static_cast<double>(nv4pack::e2m1(wa[r][k])) * nv4pack::e4m3_to_f32(static_cast<uint8_t>(sa[r] >> (8 * (k / 16)))) *
                           static_cast<double>(nv4pack::e2m1(xb[c][k])) * nv4pack::e4m3_to_f32(static_cast<uint8_t>(sb[c] >> (8 * (k / 16))));
                    want += static_cast<double>(W[r][k]) * X[c][k];
                }
                CHECK(std::fabs(got - want) <= 1e-9 * (1.0 + std::fabs(want)), "nv4 operand map: row %d column %d: %g, expected %g", r, c, got, want);
            }
    }
}

static void test_mx8_map()
{
    {
        bool seen[32] = {};
        for (int k = 0; k < 32; ++k)
        {
            const int o = ink2::Mx8Frag::xrec_byte(k);
            CHECK(o >= 0 && o < 32 && !seen[o], "mx8 xrec_byte(%d)", k);
            if (o >= 0 && o < 32) seen[o] = true;
        }
    }
    const auto code = [] { uint8_t b; do b = static_cast<uint8_t>(rng()); while ((b & 0x7f) == 0x7f); return b; };
    for (int it = 0; it < 200; ++it)
    {
        uint8_t wc[16][32], ws[16], rec[ink2ref::mx8::GR];
        for (auto& r : wc) for (uint8_t& b : r) b = code();
        for (uint8_t& b : ws) b = static_cast<uint8_t>(120 + rng() % 12);
        ink2ref::mx8::pack_group(rec, &wc[0][0], ws, 32, 1, 0, 0, 0);
        float W[16][32];
        ink2ref::mx8::decode_group(rec, W);
        uint8_t xc[8][32], xsc[8], xrec[8][32];
        float X[8][32];
        for (int m = 0; m < 8; ++m)
        {
            for (uint8_t& b : xc[m]) b = code();
            xsc[m] = static_cast<uint8_t>(120 + rng() % 12);
            ink2ref::mx8::dequantize(xc[m], &xsc[m], 1, X[m]);
            for (int k = 0; k < 32; ++k) xrec[m][ink2::Mx8Frag::xrec_byte(k)] = xc[m][k];
        }
        unsigned wa[16][32], xb[8][32];
        uint32_t sa[16], sb[8];
        for (int lane = 0; lane < 32; ++lane)
        {
            const int m = lane >> 2, t = lane & 3;
            const ink2::Mx8Frag::A a = ink2::Mx8Frag::load(rec, lane);
            uint32_t b0, b1;
            ink2::Mx8Frag::loadB(xrec[m], t, b0, b1);
            for (int i = 0; i < 4; ++i)
            {
                const int k = 4 * t + i;
                wa[m][k] = byt(a.a0, i);
                wa[m + 8][k] = byt(a.a1, i);
                wa[m][16 + k] = byt(a.a2, i);
                wa[m + 8][16 + k] = byt(a.a3, i);
                xb[m][k] = byt(b0, i);
                xb[m][16 + k] = byt(b1, i);
            }
            if (t == 0) { sa[m] = a.s & 255u; sb[m] = xsc[m]; }
            if (t == 1) sa[m + 8] = a.s & 255u;
        }
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 8; ++c)
            {
                double got = 0.0, want = 0.0;
                for (int k = 0; k < 32; ++k)
                {
                    got += static_cast<double>(nv4pack::e4m3_to_f32(static_cast<uint8_t>(wa[r][k]))) * ink2ref::mx8::scale_f32(static_cast<uint8_t>(sa[r])) *
                           static_cast<double>(nv4pack::e4m3_to_f32(static_cast<uint8_t>(xb[c][k]))) * ink2ref::mx8::scale_f32(static_cast<uint8_t>(sb[c]));
                    want += static_cast<double>(W[r][k]) * X[c][k];
                }
                CHECK(std::fabs(got - want) <= 1e-9 * (1.0 + std::fabs(want)), "mx8 operand map: row %d column %d: %g, expected %g", r, c, got, want);
            }
    }
}

// ---------------------------------------------------------------------------------------------- 3. router
static void test_router(int NE, int TK, int H, int tokens)
{
    ink2ref::Config c;
    c.hidden = H; c.n_layers = 1; c.n_heads = c.n_kv_heads = 1; c.head_dim = 16; c.inter = 64; c.n_exp = NE; c.top_k = TK; c.tp = 1;
    c.global = {0}; c.mxfp8 = {0};
    ink2ref::Weights w;
    w.layers.resize(1);
    ink2ref::Layer& L = w.layers[0];
    L.router_w.resize(static_cast<size_t>(NE) * H);
    L.router_bias.resize(static_cast<size_t>(NE));
    for (float& v : L.router_w) v = ink2ref::rbf(nrand() * 0.2f);
    for (float& v : L.router_bias) v = urand(-0.3f, 0.3f);
    // exact ties: three experts with the same row and bias, and a run of experts with the same bias
    for (int e : {7, 19})
    {
        memcpy(&L.router_w[static_cast<size_t>(e) * H], &L.router_w[static_cast<size_t>(3) * H], sizeof(float) * H);
        L.router_bias[e] = L.router_bias[3];
    }
    for (int e = 24; e < 32; ++e) L.router_bias[e] = 0.125f;
    const ink2ref::Model model(c, w);
    std::vector<float> x(static_cast<size_t>(H)), score(static_cast<size_t>(NE)), wt(static_cast<size_t>(TK)), rw;
    std::vector<int> idx(static_cast<size_t>(TK)), ri;
    int ties = 0;
    for (int t = 0; t < tokens; ++t)
    {
        const float mag = t % 5 == 0 ? 8.0f : 1.0f;                        // saturated sigmoids: many equal scores
        for (float& v : x) v = t % 10 == 9 ? 0.0f : ink2ref::rbf(nrand() * mag);   // x = 0: every score is 0.5
        if (t % 4 == 1) L.router_bias[3] = L.router_bias[7] = L.router_bias[19] = 5.0f;   // the tied three are picked
        if (t % 4 == 3) L.router_bias[3] = L.router_bias[7] = L.router_bias[19] = 0.01f;
        model.route(0, x.data(), ri, rw);
        for (int e = 0; e < NE; ++e) score[e] = ink2::sigmoid(ink2ref::dot(x.data(), &L.router_w[static_cast<size_t>(e) * H], H));
        ink2::route_pick(score.data(), L.router_bias.data(), NE, TK, c.route_scale, idx.data(), wt.data());
        for (int k = 0; k < TK; ++k)
        {
            CHECK(idx[k] == ri[k], "router NE %d: token %d pick %d: %d, reference %d", NE, t, k, idx[k], ri[k]);
            CHECK(bits(wt[k]) == bits(rw[k]), "router NE %d: token %d weight %d", NE, t, k);
            for (int j = 0; j < k; ++j)
                ties += score[idx[j]] + L.router_bias[idx[j]] == score[idx[k]] + L.router_bias[idx[k]];
        }
    }
    CHECK(ties > 0, "router NE %d: the test met no exact tie", NE);
    printf("  router NE %d TK %d: %d tokens, %d tied pick pairs\n", NE, TK, tokens, ties);
}

// ---------------------------------------------------------------------------------------------- 4. plan
static void test_plan(int M, int NE, int TK, int skew)
{
    std::vector<float> score(static_cast<size_t>(NE)), bias(static_cast<size_t>(NE)), wt(static_cast<size_t>(M) * TK);
    std::vector<int> idx(static_cast<size_t>(M) * TK);
    for (int e = 0; e < NE; ++e) bias[e] = skew == 0 ? 0.0f : skew == 1 ? (e % 9 == 0 ? 0.6f : 0.0f) : (e < TK + 2 ? 4.0f : 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (float& v : score) v = urand(0.0f, 1.0f);
        ink2::route_pick(score.data(), bias.data(), NE, TK, 8.0f, &idx[static_cast<size_t>(m) * TK], &wt[static_cast<size_t>(m) * TK]);
    }
    ink2::PlanHost p;
    ink2::plan_ref(idx.data(), wt.data(), M, TK, NE, p);
    const int ns = p.n_slots(), nt = p.n_tiles();
    const int ms = ink2::plan_max_slots(M, TK, NE), mt = ink2::plan_max_tiles(M, TK, NE);
    CHECK(ns >= 1 && ns <= ms && nt >= ns && nt <= mt, "plan M %d: %d slots (max %d), %d tiles (max %d)", M, ns, ms, nt, mt);
    if (ns < 1 || ns > ms || nt > mt) return;
    std::vector<int> slot_of(static_cast<size_t>(NE), -1);
    int tiles = 0;
    for (int s = 0; s < ns; ++s)
    {
        CHECK(p.slot_expert[s] >= 0 && p.slot_expert[s] < NE && (s == 0 || p.slot_expert[s] > p.slot_expert[s - 1]), "plan M %d: slot_expert not ascending at %d", M, s);
        CHECK(p.slot_tile0[s] == tiles && p.slot_nt[s] >= 1, "plan M %d: slot %d tiles", M, s);
        if (M <= 8) CHECK(p.slot_nt[s] == 1 && p.slot_tile0[s] == s, "plan M %d: a decode slot is one tile", M);
        if (p.slot_expert[s] >= 0 && p.slot_expert[s] < NE) slot_of[p.slot_expert[s]] = s;
        for (int j = 0; j < p.slot_nt[s] && tiles + j < nt; ++j) CHECK(p.tile_expert[tiles + j] == p.slot_expert[s], "plan M %d: tile_expert", M);
        tiles += p.slot_nt[s];
    }
    CHECK(tiles == nt, "plan M %d: tiles of the slots %d, counted %d", M, tiles, nt);
    if (tiles != nt) return;
    std::vector<uint8_t> hit(static_cast<size_t>(nt) * 8, 0);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < TK; ++k)
        {
            const size_t i = static_cast<size_t>(m) * TK + k;
            const int s = slot_of[idx[i]], tile = p.tok_slot[i], col = p.tok_col[i];
            const bool ok = s >= 0 && tile >= p.slot_tile0[s] && tile < p.slot_tile0[s] + p.slot_nt[s] && col >= 0 && col < 8;
            CHECK(ok, "plan M %d: route (%d, %d) at tile %d column %d", M, m, k, tile, col);
            if (!ok) continue;
            const size_t c = static_cast<size_t>(tile) * 8 + col;
            CHECK(!hit[c] && p.tile_tok[c] == m && bits(p.tile_w[c]) == bits(wt[i]) && ((p.tile_mask[tile] >> col) & 1), "plan M %d: column of route (%d, %d)", M, m, k);
            hit[c] = 1;
        }
    size_t live = 0;
    for (int t = 0; t < nt; ++t)
        for (int col = 0; col < 8; ++col)
        {
            const size_t c = static_cast<size_t>(t) * 8 + col;
            const bool on = (p.tile_mask[t] >> col) & 1;
            live += on;
            CHECK(on == (hit[c] != 0), "plan M %d: mask of tile %d column %d", M, t, col);
            CHECK(p.tile_tok[c] >= 0 && p.tile_tok[c] < M && (on || p.tile_w[c] == 0.0f), "plan M %d: dead column %d of tile %d", M, col, t);
        }
    CHECK(live == static_cast<size_t>(M) * TK && (p.tile_mask[0] & ~0xff) == 0, "plan M %d: %zu live columns for %d routes", M, live, M * TK);
}

int main()
{
    test_scalars();
    printf("scalars: %s\n", fails ? "FAIL" : "ok");
    test_quantizers();
    printf("quantizers (NVFP4 both rules, MXFP8): %s\n", fails ? "FAIL" : "ok");
    test_nv4_map();
    test_mx8_map();
    printf("operand maps: %s\n", fails ? "FAIL" : "ok");
    test_router(384, 8, 128, 500);
    test_router(32, 4, 64, 500);
    for (int M = 1; M <= 8; ++M)
        for (int skew = 0; skew < 3; ++skew)
        {
            test_plan(M, 384, 8, skew);
            test_plan(M, 32, 4, skew);
        }
    for (int M : {1, 7, 8, 9, 33, 256, 1024, 4096})
        for (int skew = 0; skew < 3; ++skew)
        {
            test_plan(M, 384, 8, skew);
            test_plan(M, 32, 4, skew);
        }
    printf("plans: %s\n", fails ? "FAIL" : "ok");
    if (fails) { printf("ink2_moe_host_test: %d checks FAILED\n", fails); return 1; }
    printf("ink2_moe_host_test: PASS\n");
    return 0;
}
