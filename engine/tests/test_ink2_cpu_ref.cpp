// Host test of the Inkling-2-Small CPU reference (engine/ref/ink2/ink2_cpu_ref.h) on a miniature model: 6 layers, 2 per
// depth-attention block, full attention on layers 2 and 5 (log scaling on from position 4), a sliding window of 5,
// NVFP4 experts on layers 0 .. 3 and MXFP8 on 4 and 5, 48 heads over 8 K / V heads, so a 6-rank split has the 8 + 2
// heads and the 64-wide intermediate slice of the full model's.
//   * one 12-token pass, a 5 + 4 + 3 split and 12 single-token steps give the same logits bit for bit, with two image
//     tokens in the sequence, at TP 1 and with the sums split over 6 ranks
//   * an 8-token speculative step, 3 tokens kept, rollback, then single-token steps: the same bits again
//   * the image feature row skips embed_norm; the window and the convolutions reach exactly as far as they should
//   * the MXFP8 helpers against their row-major definition
//   * the MTP drafter (3 layers, the middle one with the whole context, a window of 5, 4 taps, rank 16) at TP 1, 4
//     and 6: a draft does not depend on how the context was forwarded, the window and the two-way block attention
//     reach as far as they should, and speculative decode with 1, 3 and 7 drafts gives the bits of plain decode
//   * the vision tower (2 layers of 64 wide, 4 heads of 16, 4-pixel patches, a 4 x 4 position table): the position
//     table read at its own size and at others, the rotary position by column and by row, the patch order of the
//     2 x 2 merge, the padding value, and an image inside a token sequence in one pass and in chunks
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

constexpr int H = 192, NL = 6, NH = 48, NKV = 8, HD = 16, RD = 4, RW = 8, INTER0 = 384, NE = 16, VOCAB = 512, VALID = 500, IMG = 7, NT = 12;

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
    c.inter = INTER0;
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
        const int INTER = c.inter;
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

// ---- the MTP drafter
constexpr int ML = 3, MI = 384, MR = 16, MW = 5, MB = 8, MK = 7, NGEN = 26;

// The drafter's test model: an intermediate size that splits over 4 and 6 ranks, MXFP8 experts on every layer.
Config mtp_config(int tp)
{
    Config c = config(tp);
    c.inter = 768;
    c.mxfp8 = {1, 1, 1, 1, 1, 1};
    c.mtp_layers = ML;
    c.mtp_inter = MI;
    c.mtp_window = MW;
    c.mtp_block = MB;
    c.mtp_drafts = MK;
    c.markov_rank = MR;
    c.mtp_global = {0, 1, 0};
    c.mtp_target_layers = {1, 3, 5, 6};
    return c;
}

Weights mtp_weights(const Config& c)
{
    Weights w = weights(c);
    Rng r{0x6d7470};
    const float ih = 1.0f / std::sqrt(static_cast<float>(H));
    const size_t nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD, nt = c.mtp_target_layers.size();
    MtpWeights& y = w.mtp;
    y.fuse_weight = r.bf((ML + 1) * nt, 1.0f);
    y.mask_embed = r.bf(H, 1.0f);
    y.norm = r.bf(H, 0.2f, 1.0f);
    y.layers.resize(ML);
    for (MtpLayer& l : y.layers)
    {
        l.dflare_norm = r.bf(H, 0.2f, 1.0f);
        l.k_ctx = r.bf(nk * H, 4.0f * ih);
        l.v_ctx = r.bf(nk * H, 2.0f * ih);
        l.k_norm_ctx = r.bf(HD, 0.2f, 1.0f);
        l.input_norm = r.bf(H, 0.2f, 1.0f);
        l.post_norm = r.bf(H, 0.2f, 1.0f);
        l.wq = r.bf(nq * H, 4.0f * ih);
        l.wk = r.bf(nk * H, 4.0f * ih);
        l.wv = r.bf(nk * H, 2.0f * ih);
        l.q_norm = r.bf(HD, 0.2f, 1.0f);
        l.k_norm = r.bf(HD, 0.2f, 1.0f);
        l.wo = r.bf(H * nq, 0.1f);
        l.gate = r.bf(static_cast<size_t>(MI) * H, 2.0f * ih);
        l.up = r.bf(static_cast<size_t>(MI) * H, 2.0f * ih);
        l.down = r.bf(static_cast<size_t>(H) * MI, 0.1f);
    }
    y.head_dflare_norm = r.bf(H, 0.2f, 1.0f);
    y.markov_w1 = r.bf(static_cast<size_t>(VOCAB) * MR, 1.0f);
    y.rnn_w = r.bf(static_cast<size_t>(3) * MR * (2 * MR + 2 * H), 2.0f * ih);
    y.rnn_b = r.bf(3 * MR, 0.5f);
    y.hidden_out_w = r.bf(static_cast<size_t>(H) * MR, 0.5f);
    y.hidden_out_b = r.bf(H, 0.5f);
    return w;
}

int argmax_text(const float* lg)
{
    int best = -1;
    for (int v = 0; v < VOCAB; ++v)
        if (v != IMG && (best < 0 || lg[v] > lg[best])) best = v;
    return best;
}

void test_rbf_f64()
{
    CHECK(rbf_f64(1.0) == 1.0f && rbf_f64(-0.375) == -0.375f);
    // just above the tie 1 + 2^-8: fp32 would round it onto the tie, and the tie then goes down to 1
    CHECK(rbf_f64(1.0 + 1.0 / 256.0 + 1e-12) == 1.0f + 1.0f / 128.0f);
    CHECK(rbf_f64(1.0 + 3.0 / 256.0 - 1e-12) == 1.0f + 1.0f / 128.0f);
    CHECK(rbf_f64(1.0 + 1.0 / 256.0) == 1.0f && rbf_f64(1.0 + 3.0 / 256.0) == 1.0f + 1.0f / 64.0f);
    CHECK(rbf_f64(-(1.0 + 1.0 / 256.0 + 1e-12)) == -(1.0f + 1.0f / 128.0f));
}

// A draft depends on the forwarded positions only, not on the chunks they came in; fewer drafts are a prefix.
// Returns the 7 step logits.
std::vector<float> test_mtp_state(Model& m, const Seq& s)
{
    std::vector<float> lg, lg2, hid;
    run(m, s, {12});
    const std::vector<int> d = m.mtp_draft(20, MK, &lg, &hid);
    CHECK(d.size() == MK && lg.size() == static_cast<size_t>(MK) * VOCAB && hid.size() == static_cast<size_t>(MB) * H);
    for (int k = 0; k < MK; ++k)
        for (int v = 0; v < VOCAB; ++v)
        {
            const float z = lg[static_cast<size_t>(k) * VOCAB + v];
            if (v < VALID) CHECK(std::isfinite(z) && std::fabs(z) <= 30.0f);
            else CHECK(std::isinf(z) && z < 0.0f);
        }
    CHECK(!same(&lg[0], &lg[VOCAB], VOCAB));
    CHECK(m.pos() == 12 && m.mtp_draft(20, MK, &lg2) == d && same(lg.data(), lg2.data(), lg.size()));   // no state change
    for (auto chunks : {std::initializer_list<int>{5, 4, 3}, std::initializer_list<int>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}})
    {
        run(m, s, chunks);
        CHECK(m.mtp_draft(20, MK, &lg2) == d && same(lg.data(), lg2.data(), lg.size()));
    }
    const std::vector<int> d3 = m.mtp_draft(20, 3, &lg2);
    CHECK(d3.size() == 3 && std::equal(d3.begin(), d3.end(), d.begin()) && same(lg.data(), lg2.data(), lg2.size()));
    // the token before the block and the last token both count
    CHECK(!same(lg.data(), (m.mtp_draft(21, MK, &lg2), lg2.data()), VOCAB));
    // a rejected speculative step leaves nothing behind
    int junk[5] = {30, 31, 32, 33, 34};
    m.forward(junk, 5);
    CHECK(!same(lg.data(), (m.mtp_draft(20, MK, &lg2), lg2.data()), VOCAB));
    m.rollback(12);
    CHECK(m.mtp_draft(20, MK, &lg2) == d && same(lg.data(), lg2.data(), lg.size()));
    bool threw = false;
    try { m.mtp_draft(20, MK + 1); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    return lg;
}

// With 12 context rows and a window of 5 a local drafter layer reads rows 7 .. 11 for every row of the block; the
// layer with the whole context reads row 0 too.
void test_mtp_window(Model& m, const Seq& s)
{
    std::vector<float> base, lg;
    run(m, s, {12});
    m.mtp_draft(20, MK, &base);
    const size_t nk = static_cast<size_t>(NKV) * HD;
    const auto moved = [&](int layer, int row, bool value) {
        std::vector<float>& kv = value ? m.mtp_ctx_v(layer) : m.mtp_ctx_k(layer);
        CHECK(kv.size() == 12 * nk);
        const std::vector<float> keep = kv;
        for (size_t i = 0; i < nk; ++i) kv[row * nk + i] = rbf(-kv[row * nk + i] + 0.25f);
        m.mtp_draft(20, MK, &lg);
        kv = keep;
        return !same(base.data(), lg.data(), base.size());
    };
    for (int layer : {0, 2})
        for (bool value : {false, true})
        {
            for (int row = 0; row < 7; ++row) CHECK(!moved(layer, row, value));
            for (int row = 7; row < 12; ++row) CHECK(moved(layer, row, value));
        }
    for (bool value : {false, true})
        for (int row : {0, 6, 11}) CHECK(moved(1, row, value));
    m.mtp_draft(20, MK, &lg);
    CHECK(same(base.data(), lg.data(), base.size()));
}

// Row 0 of the block is the last token, the other rows are the mask. A change of the mask moves the final hidden row 0,
// so a row attends to the rows after it.
void test_mtp_two_way(const Config& c, const Weights& w, const Seq& s)
{
    Weights w2 = w;
    for (float& x : w2.mtp.mask_embed) x = rbf(0.5f * x - 0.125f);
    Model a(c, w), b(c, w2);
    std::vector<float> ha, hb;
    run(a, s, {12});
    run(b, s, {12});
    a.mtp_draft(20, MK, nullptr, &ha);
    b.mtp_draft(20, MK, nullptr, &hb);
    CHECK(!same(ha.data(), hb.data(), H));
    // the mask rows share one input and differ by their position only
    for (int r = 2; r < MB; ++r) CHECK(!same(&ha[H], &ha[static_cast<size_t>(r) * H], H));
}

// Plain decode against speculative decode with k drafts per round. A round forwards the last token and the k drafts,
// keeps the rows whose input was right and rolls the rest back. The drafts of this random model are almost never
// right, so every third round takes half of the true tokens as its drafts and every third round all of them.
void test_mtp_decode(Model& m, const Seq& s, int tp)
{
    int prompt[4];
    for (int i = 0; i < 4; ++i) prompt[i] = s.toks[i] == IMG ? 40 : s.toks[i];
    // plain: tok[i] is the token at position 4 + i, lg[i] the logits it gives, dr[i] / dl[i] the drafts after it
    m.rollback(0);
    std::vector<int> tok;
    std::vector<std::vector<float>> lg, dl;
    std::vector<std::vector<int>> dr;
    tok.push_back(argmax_text(&m.forward(prompt, 4)[3 * VOCAB]));
    std::vector<float> first_dl;
    const std::vector<int> first_dr = m.mtp_draft(tok[0], MK, &first_dl);
    for (int i = 0; i < NGEN + MK + 1; ++i)
    {
        lg.push_back(m.forward(&tok[i], 1));
        tok.push_back(argmax_text(lg[i].data()));
        dl.emplace_back();
        dr.push_back(m.mtp_draft(tok[i + 1], MK, &dl[i]));
    }
    int natural = 0, rounds = 0, kept = 0;
    for (int k : {1, 3, 7})
    {
        m.rollback(0);
        CHECK(argmax_text(&m.forward(prompt, 4)[3 * VOCAB]) == tok[0]);
        int i = 0;   // tok[i] is the last token, not forwarded yet
        for (int round = 0; i < NGEN; ++round)
        {
            std::vector<float> l2;
            std::vector<int> d = m.mtp_draft(tok[i], k, &l2);
            const std::vector<int>& wd = i ? dr[i - 1] : first_dr;
            const std::vector<float>& wl = i ? dl[i - 1] : first_dl;
            CHECK(std::equal(d.begin(), d.end(), wd.begin()) && same(l2.data(), wl.data(), l2.size()));
            int want = -1;
            if (round % 3)
            {
                want = round % 3 == 1 ? k / 2 : k;
                for (int j = 0; j < k; ++j) d[j] = j < want ? tok[i + 1 + j] : (tok[i + 1 + j] + 1) % VALID;
            }
            std::vector<int> rows(1, tok[i]);
            for (int t : d) rows.push_back(t == IMG ? 11 : t);
            const std::vector<float> out = m.forward(rows.data(), k + 1);
            int a = 0;
            while (a < k && rows[a + 1] == argmax_text(&out[static_cast<size_t>(a) * VOCAB])) ++a;
            if (want >= 0) CHECK(a == want);
            else natural += a;
            for (int j = 0; j <= a; ++j)
            {
                CHECK(same(&out[static_cast<size_t>(j) * VOCAB], lg[i + j].data(), VOCAB));
                CHECK(argmax_text(&out[static_cast<size_t>(j) * VOCAB]) == tok[i + j + 1]);
            }
            m.rollback(4 + i + a + 1);
            i += a + 1;
            ++rounds;
            kept += a;
        }
    }
    printf("  mtp decode, %d rank%s: %d rounds, %d drafts kept (%d of them the drafter's own)\n", tp, tp == 1 ? "" : "s", rounds, kept + natural, natural);
}

// ---- the vision tower
constexpr int VD = 64, VL = 2, VH = 4, VHD = VD / VH, VI = 128, VP = 4, VM = 2, VG = 4, VPJ = 96;

Config vis_config()
{
    Config c = config(1);
    c.has_vision = true;
    c.vis_hidden = VD;
    c.vis_layers = VL;
    c.vis_heads = VH;
    c.vis_mlp = VI;
    c.vis_patch = VP;
    c.vis_merge = VM;
    c.vis_grid = VG;
    c.vis_proj = VPJ;
    return c;
}

void vis_weights(Weights& w)
{
    Rng r{0x76697331};
    VisionWeights& x = w.vis;
    const size_t vd = VD, pk = VP * VP * 3, md = VM * VM * VD;
    const float iv = 1.0f / std::sqrt(static_cast<float>(VD));
    x.patch_embedding = r.bf(vd * pk, 2.0f / std::sqrt(static_cast<float>(pk)));
    x.class_embedding = r.bf(vd, 0.5f);
    x.cls_position_embedding = r.bf(vd, 0.5f);
    x.position_embedding = r.bf(static_cast<size_t>(VG) * VG * vd, 0.5f);
    x.ln_pre_w = r.bf(vd, 0.2f, 1.0f);
    x.ln_pre_b = r.bf(vd, 0.1f);
    x.layers.resize(VL);
    for (VisionLayer& l : x.layers)
    {
        l.ln1_w = r.bf(vd, 0.2f, 1.0f);
        l.ln1_b = r.bf(vd, 0.1f);
        l.ln2_w = r.bf(vd, 0.2f, 1.0f);
        l.ln2_b = r.bf(vd, 0.1f);
        l.q_w = r.bf(vd * vd, 4.0f * iv);
        l.q_b = r.bf(vd, 0.1f);
        l.k_w = r.bf(vd * vd, 4.0f * iv);
        l.k_b = r.bf(vd, 0.1f);
        l.v_w = r.bf(vd * vd, 2.0f * iv);
        l.v_b = r.bf(vd, 0.1f);
        l.o_w = r.bf(vd * vd, 2.0f * iv);
        l.o_b = r.bf(vd, 0.1f);
        l.ls1 = r.bf(vd, 0.2f, 0.5f);
        l.ls2 = r.bf(vd, 0.2f, 0.5f);
        l.fc_w = r.bf(static_cast<size_t>(VI) * vd, 2.0f * iv);
        l.fc_b = r.bf(VI, 0.1f);
        l.proj_w = r.bf(vd * VI, 2.0f / std::sqrt(static_cast<float>(VI)));
        l.proj_b = r.bf(vd, 0.1f);
    }
    x.proj_in = r.bf(static_cast<size_t>(VPJ) * md, 2.0f / std::sqrt(static_cast<float>(md)));
    x.proj_out = r.bf(static_cast<size_t>(H) * VPJ, 2.0f / std::sqrt(static_cast<float>(VPJ)));
}

std::vector<uint8_t> pixels(int vit_h, int vit_w, uint64_t seed)
{
    Rng r{seed};
    std::vector<uint8_t> p(static_cast<size_t>(vit_h) * VP * vit_w * VP * 3);
    for (uint8_t& b : p) b = static_cast<uint8_t>(r.next() & 255);
    return p;
}

// The position table: read at its own size it is the table; at another size, half-pixel centres clamped at the border.
void test_vis_position(const Model& m, const Weights& w)
{
    int i0, i1;
    float w1;
    for (int i = 0; i < VG; ++i)
    {
        Model::vis_lerp(i, VG, VG, i0, i1, w1);
        CHECK(i0 == i && w1 == 0.0f);
    }
    for (int i = 0; i < 32; ++i)   // the full model's table at its own size
    {
        Model::vis_lerp(i, 32, 32, i0, i1, w1);
        CHECK(i0 == i && w1 == 0.0f);
    }
    Model::vis_lerp(0, 8, 4, i0, i1, w1);   // centre -0.25: both taps on the border
    CHECK(i0 == 0 && i1 == 0 && w1 == 0.75f);
    Model::vis_lerp(1, 8, 4, i0, i1, w1);   // centre 0.25
    CHECK(i0 == 0 && i1 == 1 && w1 == 0.25f);
    Model::vis_lerp(7, 8, 4, i0, i1, w1);   // centre 3.25: the second tap clamped
    CHECK(i0 == 3 && i1 == 3 && w1 == 0.25f);
    Model::vis_lerp(0, 2, 4, i0, i1, w1);   // a smaller grid: centre 0.5
    CHECK(i0 == 0 && i1 == 1 && w1 == 0.5f);
    Model::vis_lerp(1, 2, 4, i0, i1, w1);
    CHECK(i0 == 2 && i1 == 3 && w1 == 0.5f);
    const float* t = w.vis.position_embedding.data();
    std::vector<float> e(VD);
    for (int r = 0; r < VG; ++r)
        for (int c = 0; c < VG; ++c)
        {
            m.vis_pos_embed(r, c, VG, VG, e.data());
            CHECK(same(e.data(), t + static_cast<size_t>(r * VG + c) * VD, VD));
        }
    // an 8 x 6 grid against the definition in fp64 (the result is rounded once more, so within one BF16 step)
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 6; ++c)
        {
            m.vis_pos_embed(r, c, 8, 6, e.data());
            const double sy = (r + 0.5) * VG / 8.0 - 0.5, sx = (c + 0.5) * VG / 6.0 - 0.5, fy = std::floor(sy), fx = std::floor(sx);
            const auto cl = [](double v) { return static_cast<int>(v < 0 ? 0 : v > VG - 1 ? VG - 1 : v); };
            const int y0 = cl(fy), y1 = cl(fy + 1), x0 = cl(fx), x1 = cl(fx + 1);
            for (int d = 0; d < VD; ++d)
            {
                const double want = (1 - (sy - fy)) * ((1 - (sx - fx)) * t[(y0 * VG + x0) * VD + d] + (sx - fx) * t[(y0 * VG + x1) * VD + d]) +
                                    (sy - fy) * ((1 - (sx - fx)) * t[(y1 * VG + x0) * VD + d] + (sx - fx) * t[(y1 * VG + x1) * VD + d]);
                CHECK(std::fabs(e[d] - want) <= 1e-5 + std::fabs(want) / 128.0);
                CHECK(e[d] == rbf(e[d]));
            }
        }
    // a constant column of the table stays that constant at any size (the four weights sum to 1)
    Weights w2 = w;
    for (int g = 0; g < VG * VG; ++g) w2.vis.position_embedding[static_cast<size_t>(g) * VD + 3] = 0.75f;
    Model m2(vis_config(), w2);
    for (int r = 0; r < 6; ++r)
    {
        m2.vis_pos_embed(r, 9, 6, 10, e.data());
        CHECK(e[3] == 0.75f);
    }
}

// The rotary position: the class row at (0, 0) is not turned; the first half of a head turns with the column only,
// the second half with the row only; pair 0 of each half turns by exactly the position in radians.
void test_vis_rope(const Model& m)
{
    Rng r{611};
    const std::vector<float> x = r.bf(VD, 1.0f);
    std::vector<float> a = x, b = x, c = x;
    m.vis_rope(a.data(), 0, 0);
    CHECK(same(a.data(), x.data(), VD));
    a = x;
    m.vis_rope(a.data(), 3, 5);
    m.vis_rope(b.data(), 3, 6);   // one column to the right
    m.vis_rope(c.data(), 4, 5);   // one row down
    for (int h = 0; h < VH; ++h)
    {
        const float *xa = &a[h * VHD], *xb = &b[h * VHD], *xc = &c[h * VHD], *x0 = &x[h * VHD];
        CHECK(!same(xa, xb, VHD / 2) && same(xa + VHD / 2, xb + VHD / 2, VHD / 2));
        CHECK(same(xa, xc, VHD / 2) && !same(xa + VHD / 2, xc + VHD / 2, VHD / 2));
        // pair 0 of the column half: angle = 5; pair 0 of the row half: angle = 3
        CHECK(xa[0] == rbf(x0[0] * std::cos(5.0f) - x0[1] * std::sin(5.0f)) && xa[1] == rbf(x0[1] * std::cos(5.0f) + x0[0] * std::sin(5.0f)));
        const int o = VHD / 2;
        CHECK(xa[o] == rbf(x0[o] * std::cos(3.0f) - x0[o + 1] * std::sin(3.0f)) && xa[o + 1] == rbf(x0[o + 1] * std::cos(3.0f) + x0[o] * std::sin(3.0f)));
        // the last pair of the column half: angle = 5 / theta^((hd / 2 - 2) / (hd / 2))
        const float ang = 5.0f * (1.0f / std::pow(10000.0f, static_cast<float>(VHD / 2 - 2) / static_cast<float>(VHD / 2)));
        CHECK(xa[o - 2] == rbf(x0[o - 2] * std::cos(ang) - x0[o - 1] * std::sin(ang)));
        // a rotation keeps the length of a pair (to the BF16 rounding)
        for (int p = 0; p < VHD / 2; ++p)
            CHECK(std::fabs(std::hypot(xa[2 * p], xa[2 * p + 1]) - std::hypot(x0[2 * p], x0[2 * p + 1])) < 0.02f);
    }
}

// The patch order, the 2 x 2 merge, the padding value and the grid checks.
void test_vis_merge(const Model& m, const Weights& w)
{
    // the order: the 4 patches of an image token are consecutive, tokens row by row
    int row, col;
    const int want[8][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}};
    for (int n = 0; n < 8; ++n)
    {
        Model::vis_patch_rc(n, 6, 2, row, col);
        CHECK(row == want[n][0] && col == want[n][1]);
    }
    Model::vis_patch_rc(12, 6, 2, row, col);   // token 3 = the first of the second token row
    CHECK(row == 2 && col == 0);
    std::vector<int> seen(4 * 6, 0);
    for (int n = 0; n < 4 * 6; ++n)
    {
        Model::vis_patch_rc(n, 6, 2, row, col);
        CHECK(row >= 0 && row < 4 && col >= 0 && col < 6 && !seen[row * 6 + col]++);
    }
    // pixel values: u = 0 is -1, u = 255 is 1, the padding (u = -1) one BF16 step below -1
    CHECK(rbf((0.0f * (1.0f / 255.0f) - 0.5f) / 0.5f) == -1.0f && rbf((255.0f * (1.0f / 255.0f) - 0.5f) / 0.5f) == 1.0f);
    CHECK(f32_to_bf16((-1.0f * (1.0f / 255.0f) - 0.5f) / 0.5f) == 0xbf81);

    const int vh = 4, vw = 6, nt = 6;
    const std::vector<uint8_t> px = pixels(vh, vw, 91);
    const std::vector<float> out = m.encode_image(px.data(), vh, vw), hs = m.vision_states(px.data(), vh, vw);
    CHECK(out.size() == static_cast<size_t>(nt) * H && hs.size() == static_cast<size_t>(1 + vh * vw) * VD);
    for (float v : out) CHECK(std::isfinite(v) && v == rbf(v));
    // the projector by hand from the patch rows, each found by its (row, col): token (tr, tc) reads (2tr, 2tc),
    // (2tr, 2tc + 1), (2tr + 1, 2tc), (2tr + 1, 2tc + 1) in that order
    std::vector<int> at(static_cast<size_t>(vh * vw));
    for (int n = 0; n < vh * vw; ++n)
    {
        Model::vis_patch_rc(n, vw, 2, row, col);
        at[row * vw + col] = n;
    }
    for (int t = 0; t < nt; ++t)
    {
        const int tr = t / (vw / 2), tc = t % (vw / 2);
        std::vector<float> in;
        for (int k = 0; k < 4; ++k)
        {
            const float* p = &hs[static_cast<size_t>(1 + at[(2 * tr + k / 2) * vw + 2 * tc + k % 2]) * VD];
            in.insert(in.end(), p, p + VD);
        }
        std::vector<float> m1(VPJ);
        for (int j = 0; j < VPJ; ++j) m1[j] = Model::gelu(rbf(dot(in.data(), &w.vis.proj_in[static_cast<size_t>(j) * 4 * VD], 4 * VD)));
        for (int d = 0; d < H; ++d) CHECK(out[static_cast<size_t>(t) * H + d] == rbf(dot(m1.data(), &w.vis.proj_out[static_cast<size_t>(d) * VPJ], VPJ)));
    }
    CHECK(Model::gelu(0.0f) == 0.0f && Model::gelu(1.0f) == rbf(0.8413447f) && Model::gelu(-1.0f) == rbf(-0.15865525f));
    // every token sees every patch (the attention has no mask): one changed pixel in the last patch moves token 0
    std::vector<uint8_t> p2 = px;
    p2.back() ^= 0x80;
    const std::vector<float> o2 = m.encode_image(p2.data(), vh, vw);
    CHECK(!same(o2.data(), out.data(), H) && !same(o2.data() + 5 * H, out.data() + 5 * H, H));
    // padding: pixels outside the content are not read, and count as u = -1 (not as their bytes, and not as 0)
    std::vector<uint8_t> pad = px, zero = px;
    const int ch = vh * VP - 3, cw = vw * VP - 5;
    for (int y = 0; y < vh * VP; ++y)
        for (int x = 0; x < vw * VP; ++x)
            if (y >= ch || x >= cw)
                for (int k = 0; k < 3; ++k)
                {
                    pad[(static_cast<size_t>(y) * vw * VP + x) * 3 + k] = 99;
                    zero[(static_cast<size_t>(y) * vw * VP + x) * 3 + k] = 0;
                }
    const std::vector<float> a = m.encode_image(pad.data(), vh, vw, ch, cw), b = m.encode_image(zero.data(), vh, vw, ch, cw);
    const std::vector<float> c = m.encode_image(pad.data(), vh, vw), z = m.encode_image(zero.data(), vh, vw);
    CHECK(same(a.data(), b.data(), a.size()) && !same(a.data(), c.data(), a.size()));
    const double cz = cosine(a.data(), z.data(), a.size());
    printf("  vision: padding as -1 against zero bytes: cosine %.6f, %s\n", cz, same(a.data(), z.data(), a.size()) ? "the same bits" : "other bits");
    CHECK(cz > 0.999);
    // a grid the merge does not divide, and a model without the tower
    bool threw = false;
    try { m.encode_image(px.data(), 3, 6); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    threw = false;
    Model plain(config(1), w);
    try { plain.encode_image(px.data(), vh, vw); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    // a turn of the image is another image for the tower: a row and a column of the same index turn other pairs
    const std::vector<float> tall = m.encode_image(px.data(), vw, vh);
    CHECK(tall.size() == out.size() && !same(tall.data(), out.data(), out.size()));
}

// An image in a token sequence: its rows stand at its tokens without embed_norm, in one pass and in chunks.
void test_vis_forward(const Config& c, const Weights& w)
{
    Model m(c, w);
    const int vh = 2, vw = 6, nt = 3;
    const std::vector<uint8_t> px = pixels(vh, vw, 17), px2 = pixels(2, 2, 18);
    int toks[NT] = {11, 12, IMG, IMG, IMG, 13, 14, IMG, 15, 16, 17, 18};
    const std::vector<ImageSpan> images = {{2, vh, vw, px.data(), -1, -1}, {7, 2, 2, px2.data(), 37 % (2 * VP), 5}};
    const std::vector<float> full = m.forward_with_images(toks, NT, images);
    // by hand: the rows of both images at their tokens
    const std::vector<float> r1 = m.encode_image(px.data(), vh, vw), r2 = m.encode_image(px2.data(), 2, 2, 37 % (2 * VP), 5);
    CHECK(r1.size() == static_cast<size_t>(nt) * H && r2.size() == static_cast<size_t>(H));
    std::vector<float> vis(static_cast<size_t>(NT) * H, 0.0f);
    memcpy(&vis[2 * H], r1.data(), r1.size() * sizeof(float));
    memcpy(&vis[7 * H], r2.data(), r2.size() * sizeof(float));
    m.rollback(0);
    const std::vector<float> hand = m.forward(toks, NT, vis.data());
    CHECK(same(full.data(), hand.data(), full.size()));
    // chunks that cut the first image in two
    m.rollback(0);
    std::vector<float> parts = m.forward(toks, 4, vis.data());
    const std::vector<float> rest = m.forward(toks + 4, NT - 4, vis.data() + 4 * H);
    parts.insert(parts.end(), rest.begin(), rest.end());
    CHECK(same(full.data(), parts.data(), full.size()));
    // the image rows do not pass embed_norm: a row put through it is another input
    {
        std::vector<float> n(static_cast<size_t>(H));
        m.wnorm(&vis[2 * H], w.embed_norm, n.data());
        CHECK(!same(n.data(), &vis[2 * H], H));
        memcpy(&vis[2 * H], n.data(), n.size() * sizeof(float));
    }
    m.rollback(0);
    const std::vector<float> scaled = m.forward(toks, NT, vis.data());
    CHECK(!same(full.data() + 2 * VOCAB, scaled.data() + 2 * VOCAB, VOCAB));
    // other pixels: other logits from the first image token on, the same before it
    std::vector<uint8_t> p3 = px;
    p3[5] ^= 0x40;
    m.rollback(0);
    const std::vector<float> other = m.forward_with_images(toks, NT, {{2, vh, vw, p3.data(), -1, -1}, images[1]});
    CHECK(same(full.data(), other.data(), 2 * static_cast<size_t>(VOCAB)) && !same(full.data() + 2 * VOCAB, other.data() + 2 * VOCAB, VOCAB));
    // a span on a text token, and one past the end
    bool threw = false;
    m.rollback(0);
    try { m.forward_with_images(toks, NT, {{3, vh, vw, px.data(), -1, -1}}); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { m.forward_with_images(toks, 4, {{2, vh, vw, px.data(), -1, -1}}); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

void test_vision()
{
    const Config c = vis_config();
    Weights w = weights(c);
    vis_weights(w);
    const Model m(c, w);
    test_vis_position(m, w);
    test_vis_rope(m);
    test_vis_merge(m, w);
    test_vis_forward(c, w);
    printf("  vision: position table, rotary position, merge order, padding and the image splice ok\n");
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
    // the drafter at 1, 4 and 6 ranks: each agrees with itself exactly. A sum split over ranks is added in another
    // order, so between rank counts only the draft tokens and a cosine are required; on this model the BF16 rounding
    // takes the difference away and the logits come out the same bit for bit (printed).
    test_rbf_f64();
    test_vision();
    std::vector<float> dl[3];
    int at = 0;
    for (int tp : {1, 4, 6})
    {
        const Config c = mtp_config(tp);
        const Weights mw = mtp_weights(c);
        Model m(c, mw);
        dl[at++] = test_mtp_state(m, s);
        test_mtp_window(m, s);
        if (tp == 1) test_mtp_two_way(c, mw, s);
        test_mtp_decode(m, s, tp);
    }
    const double cs4 = cosine(dl[0].data(), dl[1].data(), dl[0].size()), cs6 = cosine(dl[0].data(), dl[2].data(), dl[0].size());
    printf("  mtp draft logits against 1 rank: cosine %.6f at 4 ranks, %.6f at 6\n", cs4, cs6);
    CHECK(cs4 > 0.9999 && cs6 > 0.9999);
    for (int k = 0; k < MK; ++k)
        for (int i = 1; i < 3; ++i) CHECK(argmax_text(&dl[i][static_cast<size_t>(k) * VOCAB]) == argmax_text(&dl[0][static_cast<size_t>(k) * VOCAB]));
    printf("  mtp draft logits bit for bit against 1 rank: %s at 4 ranks, %s at 6\n", same(dl[0].data(), dl[1].data(), dl[0].size()) ? "yes" : "no",
           same(dl[0].data(), dl[2].data(), dl[0].size()) ? "yes" : "no");
    printf("test_ink2_cpu_ref: ok (6 ranks against 1: NVFP4 expert sum cosine >= %.6f, logits cosine %.6f)\n", moe_nv, cs);
    return 0;
}
