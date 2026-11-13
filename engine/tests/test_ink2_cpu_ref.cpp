// Host test of the Inkling-2-Small CPU reference (engine/ref/ink2/ink2_cpu_ref.h) on a miniature model: 6 layers, 2 per
// depth-attention block, full attention on layers 2 and 5 (log scaling on from position 4), a sliding window of 5,
// NVFP4 experts on layers 0 .. 3 and MXFP8 on 4 and 5, 48 heads over 8 K / V heads, so a 6-rank split has the 8 + 2
// heads and the 64-wide intermediate slice of the full model's.
//   * one 12-token pass, a 5 + 4 + 3 split and 12 single-token steps give the same logits bit for bit, with two image
//     tokens in the sequence, at TP 1 and with the sums split over 6 ranks
//   * an 8-token speculative step, 3 tokens kept, rollback, then single-token steps: the same bits again
//   * the image feature row skips embed_norm; the window and the convolutions reach exactly as far as they should
//   * the MXFP8 helpers against their row-major definition
//   g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off engine/tests/test_ink2_cpu_ref.cpp -o /tmp/t && /tmp/t
#include "../ref/ink2/ink2_cpu_ref.h"
#include <cstdio>
#include <cstdlib>

using namespace ink2ref;

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

namespace
{
struct Rng
{
    uint64_t s;
    uint32_t next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 40);
    }
    float uni() { return static_cast<float>(next()) / 8388608.0f - 1.0f; }   // [-1, 1)
    // n BF16 values bias + scale * uni
    std::vector<float> bf(size_t n, float scale, float bias = 0.0f)
    {
        std::vector<float> v(n);
        for (float& x : v) x = rbf(bias + scale * uni());
        return v;
    }
    std::vector<float> f32(size_t n, float scale)
    {
        std::vector<float> v(n);
        for (float& x : v) x = scale * uni();
        return v;
    }
};

constexpr int H = 192, NL = 6, NH = 48, NKV = 8, HD = 16, RD = 4, RW = 8, INTER = 384, NE = 16, VOCAB = 512, VALID = 500, IMG = 7, NT = 12;

Config config(int tp)
{
    Config c;
    c.hidden = H;
    c.n_layers = NL;
    c.block = 2;
    c.n_heads = NH;
    c.n_kv_heads = NKV;
    c.head_dim = HD;
    c.rel_dim = RD;
    c.rel_win = RW;
    c.sliding_window = 5;
    c.log_scale_base = 4;
    c.inter = INTER;
    c.n_exp = NE;
    c.top_k = 4;
    c.route_scale = 4.0f;
    c.vocab = VOCAB;
    c.valid_vocab = VALID;
    c.image_token_id = IMG;
    c.tp = tp;
    c.global = {0, 0, 1, 0, 0, 1};
    c.mxfp8 = {0, 0, 0, 0, 1, 1};
    return c;
}

Weights weights(const Config& c)
{
    Rng r{0x1ce2b00c};
    Weights w;
    const float ih = 1.0f / std::sqrt(static_cast<float>(H));
    const size_t nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD;
    w.embed = r.bf(static_cast<size_t>(VOCAB) * H, 1.0f);
    w.embed_norm = r.bf(H, 0.2f, 1.0f);
    w.final_norm = r.bf(H, 0.2f, 1.0f);
    w.output_query = r.f32(H, 1.0f);
    w.lm_head = r.bf(static_cast<size_t>(VOCAB) * H, 0.3f);
    w.layers.resize(NL);
    for (int l = 0; l < NL; ++l)
    {
        Layer& y = w.layers[l];
        const bool g = c.global[l] != 0;
        if (l) y.attn_query = r.f32(static_cast<size_t>(g ? 3 : 1) * H, 1.0f);
        y.mlp_query = r.f32(H, 1.0f);
        if (g)
        {
            y.q_norm = r.bf(H, 0.2f, 1.0f);
            y.k_norm = r.bf(H, 0.2f, 1.0f);
            y.v_norm = r.bf(H, 0.2f, 1.0f);
            y.k_static = r.bf(4 * nk, 0.5f);
            y.v_static = r.bf(4 * nk, 0.5f);
            y.k_wproj = r.bf(static_cast<size_t>(NKV) * 4 * H, ih);
            y.v_wproj = r.bf(static_cast<size_t>(NKV) * 4 * H, ih);
        }
        else
        {
            y.attn_norm = r.bf(H, 0.2f, 1.0f);
            y.k_sconv = r.bf(nk * 4, 0.5f);
            y.v_sconv = r.bf(nk * 4, 0.5f);
        }
        y.wq = r.bf(nq * H, 4.0f * ih);
        y.gate = r.bf(nq * H, 2.0f * ih);
        y.wr = r.bf(static_cast<size_t>(NH) * RD * H, 4.0f * ih);
        y.sink = r.bf(static_cast<size_t>(NH) * H, 2.0f * ih);
        y.wk = r.bf(nk * H, 4.0f * ih);
        y.wv = r.bf(nk * H, 2.0f * ih);
        y.wo = r.bf(H * nq, 0.1f);
        y.q_clip = r.bf(nq, 0.5f, 1.0f);
        y.rel_clip = r.bf(static_cast<size_t>(NH) * RD, 0.5f, 1.0f);
        y.rel_proj = r.bf(static_cast<size_t>(RD) * RW, 0.3f);
        y.attn_sconv = r.bf(static_cast<size_t>(H) * 4, 0.5f);
        y.mlp_sconv = r.bf(static_cast<size_t>(H) * 4, 0.5f);
        y.mlp_norm = r.bf(H, 0.2f, 1.0f);
        y.router_w = r.bf(static_cast<size_t>(NE) * H, 4.0f * ih);
        y.router_bias = r.f32(NE, 0.1f);
        const size_t n = static_cast<size_t>(INTER) * H;
        for (int e = 0; e < NE; ++e)
        {
            const std::vector<float> w13 = r.f32(2 * n, 0.15f), w2 = r.f32(n, 0.05f);
            std::vector<uint8_t> c13(2 * n), s13(2 * n / 16), c2(n), s2(n / 16), f1, f2;
            if (c.mxfp8[l])
            {
                mx8::quantize(w13.data(), 2 * n / 32, c13.data(), s13.data());
                mx8::quantize(w2.data(), n / 32, c2.data(), s2.data());
                f1 = mx8::pack_fc1(c13.data(), s13.data(), c13.data() + n, s13.data() + n / 32, INTER, H);
                f2 = mx8::pack_fc2(c2.data(), s2.data(), H, INTER, 0, INTER);
            }
            else
            {
                y.s1.push_back(nv4pack::quantize(w13.data(), 2 * INTER, H, nv4pack::ScaleRule::Static6, nullptr, c13.data(), s13.data()));
                y.s2.push_back(nv4pack::quantize(w2.data(), H, INTER, nv4pack::ScaleRule::Static6, nullptr, c2.data(), s2.data()));
                f1 = nv4pack::pack_fc1(c13.data(), s13.data(), c13.data() + n / 2, s13.data() + n / 16, INTER, H);
                f2 = nv4pack::pack_fc2(c2.data(), s2.data(), H, INTER, 0, INTER);
            }
            y.fc1.insert(y.fc1.end(), f1.begin(), f1.end());
            y.fc2.insert(y.fc2.end(), f2.begin(), f2.end());
        }
    }
    return w;
}

bool same(const float* a, const float* b, size_t n) { return memcmp(a, b, n * sizeof(float)) == 0; }

double cosine(const float* a, const float* b, size_t n)
{
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(a[i]) && std::isfinite(b[i]))
        {
            ab += static_cast<double>(a[i]) * b[i];
            aa += static_cast<double>(a[i]) * a[i];
            bb += static_cast<double>(b[i]) * b[i];
        }
    return ab / std::sqrt(aa * bb);
}

void test_scalars()
{
    // BF16 round to nearest even: 1 + 2^-8 is half way between 1 and 1 + 2^-7 and goes to the even one
    CHECK(f32_to_bf16(1.0f) == 0x3f80);
    CHECK(f32_to_bf16(1.0f + 1.0f / 256.0f) == 0x3f80);
    CHECK(f32_to_bf16(1.0f + 3.0f / 256.0f) == 0x3f82);
    CHECK(f32_to_bf16(1.0f + 1.5f / 256.0f) == 0x3f81);
    CHECK(f32_to_bf16(-std::numeric_limits<float>::infinity()) == 0xff80);
    CHECK(bf16_to_f32(0x3f81) == 1.0f + 1.0f / 128.0f);
    // the activation: zero at gate 0, the caps 4 and 25, the sigmoid on the raw gate
    CHECK(Model::act(0.0f, 3.0f) == 0.0f);
    CHECK(Model::act(80.0f, 400.0f) == 4.0f * 25.0f);
    CHECK(Model::act(80.0f, -400.0f) == -4.0f * 25.0f);
    CHECK(std::fabs(Model::act(-8.0f, 400.0f)) < 0.04f);   // 4 tanh(-2) sigmoid(-8) 25: the raw gate closes it
    CHECK(Model::act(1.0f, 1.0f) == rbf((4.0f * std::tanh(0.25f)) * sigmoidf(1.0f) * (25.0f * std::tanh(0.04f))));
}

void test_mx8()
{
    Rng r{77};
    const int R = 32, K = 96;
    const std::vector<float> x = r.f32(static_cast<size_t>(R) * K, 3.0f), y = r.f32(static_cast<size_t>(R) * K, 0.01f);
    std::vector<uint8_t> cx(x.size()), sx(x.size() / 32), cy(x.size()), sy(x.size() / 32);
    mx8::quantize(x.data(), x.size() / 32, cx.data(), sx.data());
    mx8::quantize(y.data(), y.size() / 32, cy.data(), sy.data());
    std::vector<float> dx(x.size()), dy(y.size());
    mx8::dequantize(cx.data(), sx.data(), x.size() / 32, dx.data());
    mx8::dequantize(cy.data(), sy.data(), y.size() / 32, dy.data());
    for (size_t b = 0; b < x.size() / 32; ++b)
    {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        // the block maximum lands in (224, 448] codes, so the step is at most amax / 14 / 2 per element
        for (int j = 0; j < 32; ++j) CHECK(std::fabs(dx[b * 32 + j] - x[b * 32 + j]) <= amax / 16.0f);
        CHECK(amax * std::ldexp(1.0f, 127 - sx[b]) <= 448.0f && amax * std::ldexp(1.0f, 127 - sx[b]) > 224.0f);
    }
    // a power of two ratio takes the exact exponent
    std::vector<float> p(32, 0.0f);
    p[5] = -448.0f * 4.0f;
    uint8_t pc[32], ps;
    mx8::quantize(p.data(), 1, pc, &ps);
    CHECK(ps == 129 && pc[5] == 0xfe && pc[0] == 0);
    // records against the row-major dequantization
    const std::vector<uint8_t> f1 = mx8::pack_fc1(cx.data(), sx.data(), cy.data(), sy.data(), R, K);
    CHECK(f1.size() == static_cast<size_t>(R / 16) * (K / 32) * 2 * mx8::GR);
    float t[16][32];
    for (int ch = 0; ch < R / 16; ++ch)
        for (int g = 0; g < K / 32; ++g)
            for (int half = 0; half < 2; ++half)
            {
                mx8::decode_group(&f1[((static_cast<size_t>(ch) * (K / 32) + g) * 2 + half) * mx8::GR], t);
                const std::vector<float>& d = half ? dy : dx;
                for (int rr = 0; rr < 16; ++rr) CHECK(same(t[rr], &d[static_cast<size_t>(ch * 16 + rr) * K + g * 32], 32));
            }
    const std::vector<uint8_t> f2 = mx8::pack_fc2(cx.data(), sx.data(), R, K, 32, 64);
    CHECK(f2.size() == static_cast<size_t>(R / 16) * 2 * mx8::GR);
    for (int ot = 0; ot < R / 16; ++ot)
        for (int g = 0; g < 2; ++g)
        {
            mx8::decode_group(&f2[(static_cast<size_t>(ot) * 2 + g) * mx8::GR], t);
            for (int rr = 0; rr < 16; ++rr) CHECK(same(t[rr], &dx[static_cast<size_t>(ot * 16 + rr) * K + 32 + g * 32], 32));
        }
}

struct Seq
{
    int toks[NT];
    std::vector<float> vis;   // [NT][H]
};

Seq sequence()
{
    Rng r{4242};
    Seq s;
    for (int& t : s.toks)
        do t = static_cast<int>(r.next() % VALID);
        while (t == IMG);
    s.toks[3] = s.toks[9] = IMG;
    s.vis = r.bf(static_cast<size_t>(NT) * H, 1.5f);
    return s;
}

// The 12 tokens in the given chunks, from position 0.
std::vector<float> run(Model& m, const Seq& s, std::initializer_list<int> chunks)
{
    m.rollback(0);
    std::vector<float> out;
    int at = 0;
    for (int n : chunks)
    {
        const std::vector<float> lg = m.forward(s.toks + at, n, &s.vis[static_cast<size_t>(at) * H]);
        out.insert(out.end(), lg.begin(), lg.end());
        at += n;
    }
    CHECK(at == NT && m.pos() == NT);
    return out;
}

// One pass, chunks and single steps; returns the logits [NT][VOCAB].
std::vector<float> test_stateful(Model& m, const Seq& s)
{
    const std::vector<float> full = run(m, s, {12});
    CHECK(full.size() == static_cast<size_t>(NT) * VOCAB);
    for (int t = 0; t < NT; ++t)
        for (int v = 0; v < VOCAB; ++v)
        {
            const float z = full[static_cast<size_t>(t) * VOCAB + v];
            if (v < VALID) CHECK(std::isfinite(z) && std::fabs(z) <= 30.0f);
            else CHECK(std::isinf(z) && z < 0.0f);
        }
    CHECK(!same(&full[0], &full[VOCAB], VOCAB));
    const std::vector<float> chunked = run(m, s, {5, 4, 3});
    CHECK(same(full.data(), chunked.data(), full.size()));
    const std::vector<float> steps = run(m, s, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    CHECK(same(full.data(), steps.data(), full.size()));
    const std::vector<float> odd = run(m, s, {3, 1, 7, 1});   // the image tokens first and last in a chunk
    CHECK(same(full.data(), odd.data(), full.size()));
    return full;
}

// 4 prompt tokens, an 8-token draft whose first 3 tokens are right, rollback to position 7, then single steps.
void test_speculative(Model& m, const Seq& s, const std::vector<float>& want)
{
    m.rollback(0);
    m.forward(s.toks, 4, s.vis.data());
    int draft[8];
    for (int i = 0; i < 8; ++i) draft[i] = i < 3 ? s.toks[4 + i] : (s.toks[4 + i] == IMG ? 11 : (s.toks[4 + i] + 1 + i) % VALID == IMG ? 12 : (s.toks[4 + i] + 1 + i) % VALID);
    const std::vector<float> spec = m.forward(draft, 8, &s.vis[4 * H]);
    CHECK(m.pos() == 12);
    CHECK(same(spec.data(), &want[4 * VOCAB], 3 * VOCAB));                 // the kept tokens' logits were already right
    CHECK(!same(&spec[3 * VOCAB], &want[7 * VOCAB], VOCAB));               // the first rejected one's are not
    m.rollback(7);
    CHECK(m.pos() == 7);
    for (int t = 7; t < NT; ++t)
    {
        const std::vector<float> lg = m.forward(s.toks + t, 1, &s.vis[static_cast<size_t>(t) * H]);
        CHECK(same(lg.data(), &want[static_cast<size_t>(t) * VOCAB], VOCAB));
    }
    // without the rollback the rejected rows would stay in the caches and change the result
    m.rollback(0);
    m.forward(s.toks, 4, s.vis.data());
    m.forward(draft, 8, &s.vis[4 * H]);
    m.rollback(8);
    CHECK(!same(m.forward(s.toks + 8, 1, &s.vis[8 * H]).data(), &want[8 * VOCAB], VOCAB));
}

// The feature row of an image token replaces the embedding after embed_norm: the row embed_norm(embed[20]) gives the
// bits of text token 20, and the raw row embed[20] does not.
void test_image(Model& m, const Weights& w, const Seq& s)
{
    const int text[2] = {20, 21}, tok[2] = {IMG, 21};
    m.rollback(0);
    const std::vector<float> a = m.forward(text, 2);
    std::vector<float> f(s.vis.begin(), s.vis.begin() + 2 * H);
    m.wnorm(&w.embed[20 * H], w.embed_norm, f.data());
    m.rollback(0);
    const std::vector<float> b = m.forward(tok, 2, f.data());
    CHECK(same(a.data(), b.data(), a.size()));
    for (int c = 0; c < H; ++c) f[c] = w.embed[20 * H + c];
    m.rollback(0);
    const std::vector<float> d = m.forward(tok, 2, f.data());
    CHECK(!same(a.data(), d.data(), VOCAB));
    // the feature row of a text token is not read
    for (int c = 0; c < H; ++c) f[H + c] = -f[H + c];
    m.rollback(0);
    CHECK(same(d.data(), m.forward(tok, 2, f.data()).data(), d.size()));
    bool threw = false;
    m.rollback(0);
    try { m.forward(tok, 1, nullptr); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// A local layer alone: token 0 reaches K / V of positions 0 .. 3 through the 4-tap convolutions, and position p sees
// the keys p - 4 .. p. So a change of token 0 moves the output at position 7 and not at 8 or 9.
void test_window(const Config& c, const Weights& w)
{
    Model m(c, w);
    Rng r{9};
    const int M = 10;
    std::vector<float> x = r.bf(static_cast<size_t>(M) * H, 1.0f);
    const std::vector<float> a = m.attention(0, x, x, x, M);
    for (int i = 0; i < H; ++i) x[i] = rbf(-x[i] + 0.25f);
    m.rollback(0);
    const std::vector<float> b = m.attention(0, x, x, x, M);
    CHECK(same(&a[8 * H], &b[8 * H], 2 * H));
    for (int t = 0; t < 8; ++t) CHECK(!same(&a[static_cast<size_t>(t) * H], &b[static_cast<size_t>(t) * H], H));
    // a global layer has no window, and its log scaling starts where (pos + 1) / base passes 1
    m.rollback(0);
    const std::vector<float> g1 = m.attention(2, x, x, x, M);
    for (int i = 0; i < H; ++i) x[i] = rbf(0.5f * x[i] - 0.125f);
    m.rollback(0);
    const std::vector<float> g2 = m.attention(2, x, x, x, M);
    CHECK(!same(&g1[9 * H], &g2[9 * H], H));
    CHECK(m.tau(0) == 1.0f && m.tau(3) == 1.0f && m.tau(4) > 1.0f && m.tau(11) == 1.0f + 0.1f * std::log(3.0f));
}

// The router: top_k distinct experts, ties to the lower index, weights that sum to route_scale.
void test_router(const Config& c, const Weights& w)
{
    Model m(c, w);
    Rng r{31};
    const std::vector<float> x = r.bf(H, 1.0f);
    std::vector<int> idx;
    std::vector<float> wt;
    m.route(1, x.data(), idx, wt);
    CHECK(idx.size() == 4 && wt.size() == 4);
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k)
    {
        for (int j = 0; j < k; ++j) CHECK(idx[j] != idx[k]);
        sum += wt[k];
    }
    CHECK(std::fabs(sum - c.route_scale) < 1e-5f);
    Weights tie = w;
    std::fill(tie.layers[1].router_w.begin(), tie.layers[1].router_w.end(), 0.0f);
    std::fill(tie.layers[1].router_bias.begin(), tie.layers[1].router_bias.end(), 0.0f);
    tie.layers[1].router_bias[9] = 0.5f;
    Model mt(c, tie);
    mt.route(1, x.data(), idx, wt);
    CHECK(idx[0] == 9 && idx[1] == 0 && idx[2] == 1 && idx[3] == 2);
    for (int k = 0; k < 4; ++k) CHECK(wt[k] == 1.0f);
}

// One routed-expert sum with the activation quantized per rank against one rank. Returns the lowest NVFP4 cosine.
double test_moe_split(const Model& m1, const Model& m6)
{
    Rng r{5};
    double low = 1.0;
    for (int l : {0, 3, 4, 5})
        for (int k = 0; k < 3; ++k)
        {
            const std::vector<float> x = r.bf(H, 1.0f);
            std::vector<float> a(H), b(H);
            m1.moe(l, x.data(), a.data());
            m6.moe(l, x.data(), b.data());
            const double cs = cosine(a.data(), b.data(), H);
            float diff = 0.0f, big = 0.0f;
            for (int i = 0; i < H; ++i)
            {
                diff = std::fmax(diff, std::fabs(a[i] - b[i]));
                big = std::fmax(big, std::fabs(a[i]));
            }
            CHECK(big > 0.1f);
            if (l >= 4) CHECK(cs > 0.99999 && diff <= 1e-5f * std::fmax(1.0f, big) + big / 128.0f);   // at most one BF16 step
            else
            {
                CHECK(cs > 0.99 && !same(a.data(), b.data(), H));
                low = cs < low ? cs : low;
            }
        }
    return low;
}
}   // namespace

int main()
{
    test_scalars();
    test_mx8();
    const Config c1 = config(1), c6 = config(6);
    const Weights w = weights(c1);
    const Seq s = sequence();
    Model m1(c1, w), m6(c6, w);
    const std::vector<float> l1 = test_stateful(m1, s);
    test_speculative(m1, s, l1);
    test_image(m1, w, s);
    test_window(c1, w);
    test_router(c1, w);
    // the sums split over 6 ranks: the same state rules. Each rank quantizes its own slice of the expert activation:
    // MXFP8 blocks do not cross a slice, so those layers agree with one rank; an NVFP4 slice has its own outer scale,
    // so one expert sum agrees to a cosine near 0.998 and the logits of this random 6-layer model drift further.
    const double moe_nv = test_moe_split(m1, m6);
    const std::vector<float> l6 = test_stateful(m6, s);
    test_speculative(m6, s, l6);
    const double cs = cosine(l1.data(), l6.data(), l1.size());
    CHECK(!same(l1.data(), l6.data(), l1.size()));
    CHECK(cs > 0.5);
    printf("test_ink2_cpu_ref: ok (6 ranks against 1: NVFP4 expert sum cosine >= %.6f, logits cosine %.6f)\n", moe_nv, cs);
    return 0;
}
