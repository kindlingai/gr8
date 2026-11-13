// Inkling-2-Small trunk operators on one GPU against the CPU reference (sm_121a). Our own code.
//
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a \
//        -I engine -I engine/kernels -I engine/attn -I engine/ref/ink2 engine/tests/ink2_layers_gpu_test.cu -o /tmp/ink2_layers_gpu_test
//   /tmp/ink2_layers_gpu_test          exit 0 = every check passed
//
// Part 1, one operator at a time on the reference's own inputs (tight: a sum over H in another order may move a BF16
// cell by one step, nothing more; what has no such sum must give the reference's bits):
//   norms, embed + image rows, depth attention (1 and 3 queries, 1 .. 9 rows, with and without the fused norm), the
//   residual steps, the GEMM and the softcapped head with its -inf mask, the static and dynamic short convolutions,
//   a speculative convolution step (8 rows, 3 accepted, then 5 single steps == a sequential run, bit for bit), one
//   whole attention layer (local and global, 1, 4 and 6 ranks) against ink2ref::Model::attention, the expert sum
//   over ranks against ink2ref::Model::moe, and the attention core alone against a host twin: a global layer at
//   position 130000 (log scaling with base 128000, 508 key spans, rows in two launches) and a local layer whose
//   window of 600 splits into 3 spans across the ring and the batch.
//
// Part 2, a 6-layer model (hidden 256; 2 layers per depth-attention block; full attention on layers 2 and 5; a window
// of 8; NVFP4 experts on layers 0 .. 3 and MXFP8 on 4 and 5; 48 heads over 8 K / V heads; 16 experts, top 4; a
// vocabulary of 768 with 750 valid ids) against ink2ref::Model::forward: one prefill, single steps, chunks across the
// window, two slots in one batch, image tokens, a speculative step of 8 with 0 and with 3 rows accepted, and the same
// with every sharded sum split over 4 and 6 simulated ranks (reference Config::tp). The router biases of the 4 chosen
// experts of a layer are raised in steps of 2 so a rounding difference cannot change a pick. The experts quantize
// their activations to 4 or 8 bits, which turns a one-step BF16 difference at a layer's input into a larger one at
// its output (on this model about 75 times at layer 0, and more with every layer), so the free-running logits can
// only be held to wide thresholds. The tight end-to-end proof is the stage-by-stage run: every stage of every layer
// of the prefill is computed by the reference from the device's own inputs of that stage and held to the
// per-operator thresholds, which leaves no room for a wiring error and none for the amplification.
//
// What must hold bit for bit on the device itself: state after a speculative step with nothing accepted, the logits
// of the steps after it, and a step's logits whatever the batch it ran in (prefill, chunks, single steps, slots).
#include "nvfp4.cuh"
#include "ink2_moe.cuh"
#include "ink2_attn.cuh"
#include "ink2_cpu_ref.h"
#include <algorithm>
#include <cstdio>
#include <string>

using ink2ref::Config;
using ink2ref::Layer;
using ink2ref::Model;
using ink2ref::Weights;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } \
    } while (0)

template <class T>
static T* dev_alloc(size_t n)
{
    T* p = nullptr;
    CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    CK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
    return p;
}
template <class T>
static T* to_dev(const std::vector<T>& v)
{
    T* p = dev_alloc<T>(v.size());
    if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
static void put(T* p, const std::vector<T>& v)
{
    if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T>
static std::vector<T> to_host(const T* p, size_t n)
{
    std::vector<T> v(n);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    if (n) CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

struct Rng   // the generator of engine/tests/test_ink2_cpu_ref.cpp
{
    uint64_t s;
    uint32_t next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 40);
    }
    float uni() { return static_cast<float>(next()) / 8388608.0f - 1.0f; }   // [-1, 1)
    std::vector<float> bf(size_t n, float scale, float bias = 0.0f)
    {
        std::vector<float> v(n);
        for (float& x : v) x = ink2ref::rbf(bias + scale * uni());
        return v;
    }
    std::vector<float> f32(size_t n, float scale)
    {
        std::vector<float> v(n);
        for (float& x : v) x = scale * uni();
        return v;
    }
};

static std::vector<uint16_t> cells(const float* f, size_t n)
{
    std::vector<uint16_t> c(n);
    for (size_t i = 0; i < n; ++i) c[i] = ink2ref::f32_to_bf16(f[i]);
    return c;
}
static std::vector<uint16_t> cells(const std::vector<float>& f) { return cells(f.data(), f.size()); }
static std::vector<float> floats(const std::vector<uint16_t>& c)
{
    std::vector<float> f(c.size());
    for (size_t i = 0; i < c.size(); ++i) f[i] = ink2ref::bf16_to_f32(c[i]);
    return f;
}
// rows r0 .. r0 + nr of a [.][cols] matrix; columns c0 .. c0 + nc of a [rows][cols] matrix
static std::vector<float> row_slice(const std::vector<float>& v, size_t cols, size_t r0, size_t nr)
{
    return std::vector<float>(v.begin() + r0 * cols, v.begin() + (r0 + nr) * cols);
}
static std::vector<float> col_slice(const std::vector<float>& v, size_t rows, size_t cols, size_t c0, size_t nc)
{
    std::vector<float> o(rows * nc);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < nc; ++c) o[r * nc + c] = v[r * cols + c0 + c];
    return o;
}

static int g_fail = 0;
static void expect(bool ok, const char* what)
{
    if (ok) return;
    printf("    FAIL: %s\n", what);
    ++g_fail;
}

struct Stat
{
    double cos, rel;
    size_t differ;   // values that are not the reference's bits
    bool mask_ok;    // the same values are infinite, with the same sign
};
static Stat compare(const float* a, const float* b, size_t n)
{
    double ab = 0, aa = 0, bb = 0, dd = 0;
    Stat s{0, 0, 0, true};
    for (size_t i = 0; i < n; ++i)
    {
        if (memcmp(&a[i], &b[i], 4) != 0) ++s.differ;
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
        {
            s.mask_ok = s.mask_ok && a[i] == b[i];
            continue;
        }
        const double x = a[i], y = b[i];
        ab += x * y; aa += x * x; bb += y * y; dd += (x - y) * (x - y);
    }
    s.cos = ab / std::sqrt(aa * bb + 1e-300);
    s.rel = std::sqrt(dd / (bb + 1e-300));
    return s;
}
static Stat compare(const std::vector<uint16_t>& a, const std::vector<float>& b)
{
    const std::vector<float> f = floats(a);
    return compare(f.data(), b.data(), b.size());
}
// prints the line and checks cos > min_cos, rel < max_rel and the infinities
static bool check(const char* tag, const Stat& s, size_t n, double min_cos, double max_rel)
{
    const bool ok = s.mask_ok && s.cos > min_cos && s.rel < max_rel;
    printf("  %-58s cos %.7f  rel %.2e  differ %zu / %zu  %s\n", tag, s.cos, s.rel, s.differ, n, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
    return ok;
}
static void check_bits(const char* tag, const void* a, const void* b, size_t bytes)
{
    const bool ok = memcmp(a, b, bytes) == 0;
    printf("  %-58s %s\n", tag, ok ? "bit for bit" : "FAIL (bits differ)");
    if (!ok) ++g_fail;
}

// per-operator thresholds: BF16 results of an fp32 sum in another order; fp32 results
constexpr double OP_COS = 0.99999, OP_REL = 5e-3, F32_REL = 1e-4;
// end-to-end thresholds on the logits of the 6-layer model (see the head of the file)
constexpr double E2E_COS = 0.8, E2E_REL = 0.6;

struct DevRows
{
    int *slot = nullptr, *pos = nullptr, *run0 = nullptr;
    int cap = 0;
    void alloc(int n)
    {
        cap = n;
        slot = dev_alloc<int>(n);
        pos = dev_alloc<int>(n);
        run0 = dev_alloc<int>(n);
    }
    void set(const ink2::RowsHost& h)
    {
        if (h.rows() > cap) { printf("DevRows: too many rows\n"); exit(2); }
        put(slot, h.slot);
        put(pos, h.pos);
        put(run0, h.run0);
    }
    ink2::Rows view() const { return ink2::Rows{slot, pos, run0}; }
};
static ink2::RowsHost one_run(int slot, int pos0, int n)
{
    ink2::RowsHost r;
    r.add(slot, pos0, n);
    return r;
}

// ---------------------------------------------------------------------------------------------- the 6-layer model
constexpr int H = 256, NL = 6, NH = 48, NKV = 8, HD = 16, RD = 16, RELW = 8, WIN = 8, BASE = 6, INTER = 768, NE = 16, TK = 4, VOCAB = 768, VALID = 750, IMG = 7;
constexpr int MAX_PPS = 2;   // positions 0 .. 127 per slot

struct Mini
{
    Config c;
    Weights w;
    struct Codes { std::vector<uint8_t> c13, s13, c2, s2; };
    std::vector<std::vector<Codes>> codes;   // [layer][expert]: the experts' quantized matrices before packing
};

static Config mini_config(int tp)
{
    Config c;
    c.hidden = H;
    c.n_layers = NL;
    c.block = 2;
    c.n_heads = NH;
    c.n_kv_heads = NKV;
    c.head_dim = HD;
    c.rel_dim = RD;
    c.rel_win = RELW;
    c.sliding_window = WIN;
    c.log_scale_base = BASE;
    c.inter = INTER;
    c.n_exp = NE;
    c.top_k = TK;
    c.route_scale = 4.0f;
    c.vocab = VOCAB;
    c.valid_vocab = VALID;
    c.image_token_id = IMG;
    c.tp = tp;
    c.global = {0, 0, 1, 0, 0, 1};
    c.mxfp8 = {0, 0, 0, 0, 1, 1};
    return c;
}

// fc1 / fc2 records of every expert of layer l for rank r of T: intermediate rows / columns r * INTER / T ..
static void pack_experts(const Mini& mm, int l, int T, int r, std::vector<uint8_t>& fc1, std::vector<uint8_t>& fc2)
{
    const int is = INTER / T;
    const size_t n = static_cast<size_t>(INTER) * H, o = static_cast<size_t>(r) * is * H;
    fc1.clear();
    fc2.clear();
    for (int e = 0; e < NE; ++e)
    {
        const Mini::Codes& k = mm.codes[l][e];
        std::vector<uint8_t> f1, f2;
        if (mm.c.mxfp8[l])
        {
            f1 = ink2ref::mx8::pack_fc1(k.c13.data() + o, k.s13.data() + o / 32, k.c13.data() + n + o, k.s13.data() + (n + o) / 32, is, H);
            f2 = ink2ref::mx8::pack_fc2(k.c2.data(), k.s2.data(), H, INTER, r * is, is);
        }
        else
        {
            f1 = nv4pack::pack_fc1(k.c13.data() + o / 2, k.s13.data() + o / 16, k.c13.data() + (n + o) / 2, k.s13.data() + (n + o) / 16, is, H);
            f2 = nv4pack::pack_fc2(k.c2.data(), k.s2.data(), H, INTER, r * is, is);
        }
        fc1.insert(fc1.end(), f1.begin(), f1.end());
        fc2.insert(fc2.end(), f2.begin(), f2.end());
    }
}

static void build_mini(Mini& mm)
{
    mm.c = mini_config(1);
    const Config& c = mm.c;
    Rng r{0x1ce2b00c};
    Weights& w = mm.w;
    const float ih = 1.0f / std::sqrt(static_cast<float>(H));
    const size_t nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD;
    w.embed = r.bf(static_cast<size_t>(VOCAB) * H, 1.0f);
    w.embed_norm = r.bf(H, 0.2f, 1.0f);
    w.final_norm = r.bf(H, 0.2f, 1.0f);
    w.output_query = r.f32(H, 1.0f);
    w.lm_head = r.bf(static_cast<size_t>(VOCAB) * H, 0.3f);
    w.layers.resize(NL);
    mm.codes.resize(NL);
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
        y.rel_proj = r.bf(static_cast<size_t>(RD) * RELW, 0.3f);
        y.attn_sconv = r.bf(static_cast<size_t>(H) * 4, 0.5f);
        y.mlp_sconv = r.bf(static_cast<size_t>(H) * 4, 0.5f);
        y.mlp_norm = r.bf(H, 0.2f, 1.0f);
        y.router_w = r.bf(static_cast<size_t>(NE) * H, 4.0f * ih);
        y.router_bias = r.f32(NE, 0.1f);
        // the layer's 4 experts, in a fixed pick order: a score is in (0, 1), so steps of 2 leave no near tie
        for (int k = 0; k < TK; ++k) y.router_bias[(3 * l + 5 * k) % NE] += 2.0f * static_cast<float>(TK - k);
        const size_t n = static_cast<size_t>(INTER) * H;
        mm.codes[l].resize(NE);
        for (int e = 0; e < NE; ++e)
        {
            const std::vector<float> w13 = r.f32(2 * n, 0.15f), w2 = r.f32(n, 0.05f);
            Mini::Codes& k = mm.codes[l][e];
            if (c.mxfp8[l])
            {
                k.c13.resize(2 * n); k.s13.resize(2 * n / 32); k.c2.resize(n); k.s2.resize(n / 32);
                ink2ref::mx8::quantize(w13.data(), 2 * n / 32, k.c13.data(), k.s13.data());
                ink2ref::mx8::quantize(w2.data(), n / 32, k.c2.data(), k.s2.data());
            }
            else
            {
                k.c13.resize(n); k.s13.resize(2 * n / 16); k.c2.resize(n / 2); k.s2.resize(n / 16);
                y.s1.push_back(nv4pack::quantize(w13.data(), 2 * INTER, H, nv4pack::ScaleRule::Static6, nullptr, k.c13.data(), k.s13.data()));
                y.s2.push_back(nv4pack::quantize(w2.data(), H, INTER, nv4pack::ScaleRule::Static6, nullptr, k.c2.data(), k.s2.data()));
            }
        }
        pack_experts(mm, l, 1, 0, y.fc1, y.fc2);
    }
}

// ---------------------------------------------------------------------------------------------- the device model
// The expert sum over ranks: per pick the ranks' partials in rank order, then the picks in pick order.
struct MoeSumArgs
{
    const float* yp[12];   // per rank [tile][8][H]
    int R;
    ink2::Plan plan;
    int M, Hn, TKn;
    uint16_t* out;
};
__global__ void __launch_bounds__(256) k_test_moe_sum(MoeSumArgs a)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(a.M) * a.Hn) return;
    const int m = static_cast<int>(i / a.Hn), n = static_cast<int>(i % a.Hn);
    float s = 0.0f;
    for (int k = 0; k < a.TKn; ++k)
    {
        const size_t o = static_cast<size_t>(a.plan.tok_slot[m * a.TKn + k] * 8 + a.plan.tok_col[m * a.TKn + k]) * a.Hn + n;
        float y = a.yp[0][o];
        for (int r = 1; r < a.R; ++r) y = ink2::add_rn(y, a.yp[r][o]);
        s = ink2::add_rn(s, y);
    }
    a.out[i] = ink2::bf16(s);
}

static ink2::AttnGeom rank_geom(const Config& c, int l, int T, int r)
{
    ink2::AttnGeom g;
    g.H = c.hidden;
    g.hd = c.head_dim;
    g.rel_dim = c.rel_dim;
    g.rel_win = c.rel_win;
    g.gqa = c.n_heads / c.n_kv_heads;
    g.nh_q = c.n_heads / T;
    g.q0 = r * g.nh_q;
    g.kv0 = g.q0 / g.gqa;
    g.nh_kv = (g.q0 + g.nh_q - 1) / g.gqa + 1 - g.kv0;
    g.window = c.sliding_window;
    g.log_scale_base = c.log_scale_base;
    g.global = c.global[l] != 0;
    return g;
}
// The rank's slices of a layer's attention weights, on the device.
static ink2::AttnWeights rank_weights(const Layer& y, const ink2::AttnGeom& g, int n_heads)
{
    const size_t Hn = static_cast<size_t>(g.H), hd = static_cast<size_t>(g.hd), rd = static_cast<size_t>(g.rel_dim), q0 = static_cast<size_t>(g.q0), nq = static_cast<size_t>(g.nh_q),
                 k0 = static_cast<size_t>(g.kv0), nkv = static_cast<size_t>(g.nh_kv), nk_all = y.wk.size() / Hn;
    ink2::AttnWeights a;
    a.wq = to_dev(cells(row_slice(y.wq, Hn, q0 * hd, nq * hd)));
    a.gate = to_dev(cells(row_slice(y.gate, Hn, q0 * hd, nq * hd)));
    a.wr = to_dev(cells(row_slice(y.wr, Hn, q0 * rd, nq * rd)));
    a.sink = to_dev(cells(row_slice(y.sink, Hn, q0, nq)));
    a.wk = to_dev(cells(row_slice(y.wk, Hn, k0 * hd, nkv * hd)));
    a.wv = to_dev(cells(row_slice(y.wv, Hn, k0 * hd, nkv * hd)));
    a.wo = to_dev(cells(col_slice(y.wo, Hn, static_cast<size_t>(n_heads) * hd, q0 * hd, nq * hd)));
    a.q_clip = to_dev(cells(row_slice(y.q_clip, hd, q0, nq)));
    a.rel_clip = to_dev(cells(row_slice(y.rel_clip, rd, q0, nq)));
    a.rel_proj = to_dev(cells(y.rel_proj));
    if (g.global)
    {
        a.k_static = to_dev(cells(col_slice(y.k_static, 4, nk_all, k0 * hd, nkv * hd)));
        a.v_static = to_dev(cells(col_slice(y.v_static, 4, nk_all, k0 * hd, nkv * hd)));
        a.k_wproj = to_dev(cells(row_slice(y.k_wproj, Hn, k0 * 4, nkv * 4)));
        a.v_wproj = to_dev(cells(row_slice(y.v_wproj, Hn, k0 * 4, nkv * 4)));
    }
    else
    {
        a.k_sconv = to_dev(cells(row_slice(y.k_sconv, 4, k0 * hd, nkv * hd)));
        a.v_sconv = to_dev(cells(row_slice(y.v_sconv, 4, k0 * hd, nkv * hd)));
    }
    return a;
}
// The state of a layer on a rank: local = a ring; global = pages in a scrambled order behind the page table.
static ink2::AttnState rank_state(const ink2::AttnGeom& g, int slots, int max_pps)
{
    const size_t nk = static_cast<size_t>(g.nh_kv) * g.hd;
    ink2::AttnState s;
    s.hist_k = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * nk);
    s.hist_v = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * nk);
    if (!g.global)
    {
        s.ring = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 2 * g.window * nk);
        return s;
    }
    const int total = slots * max_pps, step = total % 3 ? 3 : (total % 5 ? 5 : 7);
    std::vector<int> table(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) table[i] = (i * step + 1) % total;
    s.pages = dev_alloc<uint16_t>(static_cast<size_t>(total) * 2 * ink2::KV_PAGE_TOKENS * nk);
    s.page_table = to_dev(table);
    s.max_pps = max_pps;
    return s;
}

struct SharedLayer   // what every rank holds a copy of
{
    const float *attn_query = nullptr, *mlp_query = nullptr, *router_bias = nullptr;
    const uint16_t *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr, *attn_sconv = nullptr, *mlp_sconv = nullptr, *mlp_norm = nullptr, *router_w = nullptr;
    uint16_t *hist_a = nullptr, *hist_m = nullptr;   // [slots][3][H]
    uint16_t *att = nullptr, *mo = nullptr;          // [M][H] the inputs of attn_sconv and mlp_sconv, kept until the commit
};
struct RankLayer
{
    ink2::AttnGeom g;
    ink2::AttnWeights aw;
    ink2::AttnState as;
    ink2::AttnWork wk;
    const uint8_t *fc1 = nullptr, *fc2 = nullptr;
    const float *s1 = nullptr, *s2 = nullptr;
};

// The forward of the model with every sharded sum split over T ranks, all on this GPU. What is replicated on the
// ranks of a real deployment (norms, depth attention, convolutions of the residual, the router) runs once.
class Runner
{
public:
    Runner(const Mini& mm, int T_, int slots_, int max_m) : c(mm.c), T(T_), slots(slots_), maxM(max_m)
    {
        const Weights& w = mm.w;
        const size_t mh = static_cast<size_t>(maxM) * H;
        embed = to_dev(cells(w.embed));
        embed_norm = to_dev(cells(w.embed_norm));
        final_norm = to_dev(cells(w.final_norm));
        output_query = to_dev(w.output_query);
        sh.resize(NL);
        rk.assign(static_cast<size_t>(T), std::vector<RankLayer>(NL));
        for (int l = 0; l < NL; ++l)
        {
            const Layer& y = w.layers[l];
            SharedLayer& s = sh[l];
            if (l) s.attn_query = to_dev(y.attn_query);
            s.mlp_query = to_dev(y.mlp_query);
            s.router_bias = to_dev(y.router_bias);
            if (c.global[l])
            {
                s.q_norm = to_dev(cells(y.q_norm));
                s.k_norm = to_dev(cells(y.k_norm));
                s.v_norm = to_dev(cells(y.v_norm));
            }
            else s.attn_norm = to_dev(cells(y.attn_norm));
            s.attn_sconv = to_dev(cells(y.attn_sconv));
            s.mlp_sconv = to_dev(cells(y.mlp_sconv));
            s.mlp_norm = to_dev(cells(y.mlp_norm));
            s.router_w = to_dev(cells(y.router_w));
            s.hist_a = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * H);
            s.hist_m = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * H);
            s.att = dev_alloc<uint16_t>(mh);
            s.mo = dev_alloc<uint16_t>(mh);
            for (int r = 0; r < T; ++r)
            {
                RankLayer& x = rk[r][l];
                x.g = rank_geom(c, l, T, r);
                x.aw = rank_weights(y, x.g, NH);
                x.as = rank_state(x.g, slots, MAX_PPS);
                x.wk.alloc(x.g, maxM, 0);
                std::vector<uint8_t> f1, f2;
                pack_experts(mm, l, T, r, f1, f2);
                x.fc1 = to_dev(f1);
                x.fc2 = to_dev(f2);
                x.s1 = to_dev(y.s1);
                x.s2 = to_dev(y.s2);
            }
        }
        mw.resize(static_cast<size_t>(T));
        for (int r = 0; r < T; ++r)
        {
            mw[r].alloc(maxM, H, INTER / T, NE, TK, false);
            lm.push_back(to_dev(cells(row_slice(w.lm_head, H, static_cast<size_t>(r) * (VOCAB / T), VOCAB / T))));
        }
        rows.alloc(maxM);
        toks = dev_alloc<int>(maxM);
        acc_dev = dev_alloc<uint8_t>(maxM);
        for (uint16_t** p : {&vis, &h, &qi, &ki, &vi, &a, &mid, &xm, &ms, &hf, &scratch}) *p = dev_alloc<uint16_t>(mh);
        for (int b = 0; b < (NL + c.block - 1) / c.block; ++b) states.push_back(dev_alloc<uint16_t>(mh));
        parts = dev_alloc<float>(static_cast<size_t>(T) * mh);
        logits = dev_alloc<float>(static_cast<size_t>(maxM) * VOCAB);
    }

    // The rows' logits [M][VOCAB]. vis_rows = [M][H] cells, read on the image tokens. No state changes before commit().
    std::vector<float> forward(const ink2::RowsHost& rh, const int* tk, const uint16_t* vis_rows)
    {
        M = rh.rows();
        const size_t n = static_cast<size_t>(M) * H;
        const ink2::Rows rw = rows.view();
        rows.set(rh);
        put(toks, std::vector<int>(tk, tk + M));
        if (vis_rows) put(vis, std::vector<uint16_t>(vis_rows, vis_rows + n));
        ink2::embed_norm(toks, embed, embed_norm, vis_rows ? vis : nullptr, IMG, h, M, H, 0);
        cap(h, n);
        for (int l = 0; l < NL; ++l)
        {
            SharedLayer& s = sh[l];
            const bool glob = c.global[l] != 0;
            const int done = (l + c.block - 1) / c.block;   // finished blocks before the attention of layer l
            if (l == 0) ink2::rmsnorm(h, s.attn_norm, qi, M, H, 0);
            else
            {
                ink2::AttnResArgs ar{};
                for (int d = 0; d < done; ++d) ar.rows[d] = states[d];
                ar.rows[done] = h;
                ar.query = s.attn_query;
                ar.D = done + 1;
                ar.Q = glob ? 3 : 1;
                ar.H = H;
                ar.norm[0] = glob ? s.q_norm : s.attn_norm;
                ar.norm[1] = s.k_norm;
                ar.norm[2] = s.v_norm;
                ar.out[0] = qi;
                ar.out[1] = ki;
                ar.out[2] = vi;
                ink2::attn_res(ar, M, 0);
            }
            cap(qi, n);
            if (glob)
            {
                cap(ki, n);
                cap(vi, n);
            }
            for (int r = 0; r < T; ++r)
            {
                RankLayer& x = rk[r][l];
                ink2::attn_forward(x.g, x.aw, x.as, x.wk, rw, rh.max_pos, qi, glob ? ki : qi, glob ? vi : qi, M, T == 1 ? s.att : nullptr, parts + static_cast<size_t>(r) * n, 0);
            }
            if (T > 1) ink2::sum_ranks(parts, T, n, s.att, 0);
            ink2::sconv_static(s.att, s.hist_a, rw, s.attn_sconv, M, H, a, 0);
            cap(s.att, n);
            cap(a, n);
            ink2::attn_block_step(h, a, l % c.block == 0, states[l / c.block], mid, n, 0);
            cap(mid, n);
            ink2::AttnResArgs ar{};
            const int d2 = l / c.block + 1;
            for (int d = 0; d < d2; ++d) ar.rows[d] = states[d];
            ar.rows[d2] = mid;
            ar.query = s.mlp_query;
            ar.D = d2 + 1;
            ar.Q = 1;
            ar.H = H;
            ar.norm[0] = s.mlp_norm;
            ar.out[0] = xm;
            ink2::attn_res(ar, M, 0);
            cap(xm, n);
            if (c.mxfp8[l]) moe<ink2::Mx8Frag>(l);
            else moe<nv4::Frag>(l);
            ink2::sconv_static(s.mo, s.hist_m, rw, s.mlp_sconv, M, H, ms, 0);
            cap(s.mo, n);
            cap(ms, n);
            ink2::add_bf16(mid, ms, h, n, 0);
            cap(h, n);
        }
        ink2::AttnResArgs ar{};
        const int nb = static_cast<int>(states.size());
        for (int d = 0; d < nb; ++d) ar.rows[d] = states[d];
        ar.rows[nb] = h;
        ar.query = output_query;
        ar.D = nb + 1;
        ar.Q = 1;
        ar.H = H;
        ar.norm[0] = final_norm;
        ar.out[0] = hf;
        ink2::attn_res(ar, M, 0);
        cap(hf, n);
        const int vr = VOCAB / T;
        std::vector<float> out(static_cast<size_t>(M) * VOCAB);
        for (int r = 0; r < T; ++r)
        {
            ink2::lm_head(hf, lm[r], M, H, vr, r * vr, VALID, logits, 0);
            const std::vector<float> part = to_host(logits, static_cast<size_t>(M) * vr);
            for (int m = 0; m < M; ++m) memcpy(&out[static_cast<size_t>(m) * VOCAB + r * vr], &part[static_cast<size_t>(m) * vr], sizeof(float) * vr);
        }
        return out;
    }

    // Keep the rows of the last forward: all of them, or the ones acc marks.
    void commit(const std::vector<uint8_t>* acc = nullptr)
    {
        const ink2::Rows rw = rows.view();
        if (acc) put(acc_dev, *acc);
        const uint8_t* am = acc ? acc_dev : nullptr;
        for (int l = 0; l < NL; ++l)
        {
            ink2::sconv_advance(sh[l].att, sh[l].hist_a, rw, am, M, H, 0);
            ink2::sconv_advance(sh[l].mo, sh[l].hist_m, rw, am, M, H, 0);
            for (int r = 0; r < T; ++r) ink2::attn_commit(rk[r][l].g, rk[r][l].as, rk[r][l].wk, rw, am, M, 0);
        }
        CK(cudaDeviceSynchronize());
    }

    // Every state cell a later step of a slot at position pos can read: the convolution rings, the K / V rings, and
    // the page entries of the positions below pos.
    std::vector<uint16_t> snapshot(int slot, int pos) const
    {
        std::vector<uint16_t> out;
        const auto add = [&out](const uint16_t* p, size_t n) {
            const std::vector<uint16_t> v = to_host(p, n);
            out.insert(out.end(), v.begin(), v.end());
        };
        for (int l = 0; l < NL; ++l)
        {
            add(sh[l].hist_a, static_cast<size_t>(slots) * 3 * H);
            add(sh[l].hist_m, static_cast<size_t>(slots) * 3 * H);
            for (int r = 0; r < T; ++r)
            {
                const RankLayer& x = rk[r][l];
                const size_t nk = static_cast<size_t>(x.g.nh_kv) * x.g.hd;
                add(x.as.hist_k, static_cast<size_t>(slots) * 3 * nk);
                add(x.as.hist_v, static_cast<size_t>(slots) * 3 * nk);
                if (!x.g.global)
                {
                    add(x.as.ring, static_cast<size_t>(slots) * 2 * x.g.window * nk);
                    continue;
                }
                const std::vector<int> table = to_host(x.as.page_table, static_cast<size_t>(slots) * MAX_PPS);
                for (int p = 0; p < pos; ++p)
                    for (int kv = 0; kv < 2; ++kv)
                        add(x.as.pages + ((static_cast<size_t>(table[slot * MAX_PPS + p / ink2::KV_PAGE_TOKENS]) * 2 + kv) * ink2::KV_PAGE_TOKENS + p % ink2::KV_PAGE_TOKENS) * nk, nk);
            }
        }
        return out;
    }

    const Config& c;
    int T, slots, maxM, M = 0;
    // When set, forward() appends every stage's result in order: h; per layer q_in, (k_in, v_in on a global layer),
    // att, a, mid, x_mlp, the expert sum, ms, h; then the input of the head. See stage_by_stage().
    std::vector<std::vector<float>>* trace = nullptr;

private:
    void cap(const uint16_t* p, size_t n) const
    {
        if (trace) trace->push_back(floats(to_host(p, n)));
    }

    template <class F>
    void moe(int l)
    {
        const SharedLayer& s = sh[l];
        ink2::Work& w0 = mw[0];
        ink2::route(xm, s.router_w, s.router_bias, M, H, NE, TK, c.route_scale, w0.score, w0.idx, w0.wt, w0.plan, 0);
        MoeSumArgs sa{};
        for (int r = 0; r < T; ++r)
        {
            const RankLayer& x = rk[r][l];
            ink2::MoeArgs ma = mw[r].args(x.fc1, x.fc2, x.s1, x.s2, M, T == 1 ? s.mo : scratch);
            ma.plan = w0.plan;   // one routing for every rank
            if (M <= 8) ink2::moe_decode<F>(ma, xm, NE, 0);
            else ink2::moe_prefill<F>(ma, xm, NE, 0);
            sa.yp[r] = static_cast<const float*>(mw[r].ypart);
        }
        if (T == 1) return;
        sa.R = T;
        sa.plan = w0.plan;
        sa.M = M;
        sa.Hn = H;
        sa.TKn = TK;
        sa.out = s.mo;
        k_test_moe_sum<<<ink2::cells_grid(static_cast<size_t>(M) * H), 256, 0, 0>>>(sa);
    }

    std::vector<SharedLayer> sh;
    std::vector<std::vector<RankLayer>> rk;   // [rank][layer]
    std::vector<ink2::Work> mw;               // [rank]
    std::vector<const uint16_t*> lm;          // [rank] lm_head rows r * VOCAB / T ..
    std::vector<uint16_t*> states;            // [block][M][H]
    const uint16_t *embed = nullptr, *embed_norm = nullptr, *final_norm = nullptr;
    const float* output_query = nullptr;
    DevRows rows;
    int* toks = nullptr;
    uint8_t* acc_dev = nullptr;
    uint16_t *vis = nullptr, *h = nullptr, *qi = nullptr, *ki = nullptr, *vi = nullptr, *a = nullptr, *mid = nullptr, *xm = nullptr, *ms = nullptr, *hf = nullptr, *scratch = nullptr;
    float *parts = nullptr, *logits = nullptr;
};

// ---------------------------------------------------------------------------------------------- part 1: operators
static void test_norms(const Mini& mm)
{
    printf("norms, embed, residual steps\n");
    Model ref(mm.c, mm.w);
    Rng r{11};
    const int M = 5;
    const std::vector<float> x = r.bf(static_cast<size_t>(M) * H, 3.0f), wn = r.bf(H, 0.2f, 1.0f);
    std::vector<float> want(x.size());
    for (int m = 0; m < M; ++m) ref.wnorm(&x[static_cast<size_t>(m) * H], wn, &want[static_cast<size_t>(m) * H]);
    const uint16_t* dx = to_dev(cells(x));
    const uint16_t* dw = to_dev(cells(wn));
    uint16_t* dy = dev_alloc<uint16_t>(x.size());
    ink2::rmsnorm(dx, dw, dy, M, H, 0);
    check("rmsnorm", compare(to_host(dy, x.size()), want), x.size(), OP_COS, OP_REL);

    // embed: rows 1 and 3 are image tokens and take their feature row as it is
    const int toks[M] = {5, IMG, 300, IMG, 767};
    const std::vector<float> vis = r.bf(static_cast<size_t>(M) * H, 1.5f);
    for (int m = 0; m < M; ++m)
        if (toks[m] == IMG) memcpy(&want[static_cast<size_t>(m) * H], &vis[static_cast<size_t>(m) * H], sizeof(float) * H);
        else ref.wnorm(&mm.w.embed[static_cast<size_t>(toks[m]) * H], mm.w.embed_norm, &want[static_cast<size_t>(m) * H]);
    const uint16_t* de = to_dev(cells(mm.w.embed));
    const uint16_t* den = to_dev(cells(mm.w.embed_norm));
    const uint16_t* dv = to_dev(cells(vis));
    const int* dt = to_dev(std::vector<int>(toks, toks + M));
    ink2::embed_norm(dt, de, den, dv, IMG, dy, M, H, 0);
    const std::vector<uint16_t> got = to_host(dy, x.size());
    check("embed + embed_norm, 2 image rows", compare(got, want), x.size(), OP_COS, OP_REL);
    const std::vector<uint16_t> vc = cells(vis);
    check_bits("  image rows are the feature rows", &got[H], &vc[H], 2 * H);
    check_bits("  image rows are the feature rows (row 3)", &got[3 * H], &vc[3 * H], 2 * H);
    // without feature rows the image token id is an ordinary token
    ink2::embed_norm(dt, de, den, nullptr, IMG, dy, M, H, 0);
    for (int m = 0; m < M; ++m) ref.wnorm(&mm.w.embed[static_cast<size_t>(toks[m]) * H], mm.w.embed_norm, &want[static_cast<size_t>(m) * H]);
    check("embed + embed_norm, no feature rows", compare(to_host(dy, x.size()), want), x.size(), OP_COS, OP_REL);

    // residual steps: exact
    const std::vector<float> b = r.bf(x.size(), 2.0f);
    const uint16_t* db = to_dev(cells(b));
    uint16_t* dst = dev_alloc<uint16_t>(x.size());
    uint16_t* dmid = dev_alloc<uint16_t>(x.size());
    std::vector<uint16_t> sum(x.size());
    for (size_t i = 0; i < x.size(); ++i) sum[i] = ink2ref::f32_to_bf16(x[i] + b[i]);
    ink2::attn_block_step(dx, db, false, dst, dmid, x.size(), 0);
    check_bits("mid = bf16(h + a) inside a block", to_host(dmid, x.size()).data(), sum.data(), 2 * x.size());
    ink2::attn_block_step(dx, db, true, dst, dmid, x.size(), 0);
    check_bits("block start: state = h", to_host(dst, x.size()).data(), cells(x).data(), 2 * x.size());
    check_bits("block start: mid = a", to_host(dmid, x.size()).data(), cells(b).data(), 2 * x.size());
    // the sum of rank partials, in rank order
    const int R = 6;
    const std::vector<float> parts = r.f32(static_cast<size_t>(R) * 700, 2.0f);
    for (size_t i = 0; i < 700; ++i)
    {
        float s = parts[i];
        for (int k = 1; k < R; ++k) s += parts[static_cast<size_t>(k) * 700 + i];
        sum[i] = ink2ref::f32_to_bf16(s);
    }
    ink2::sum_ranks(to_dev(parts), R, 700, dy, 0);
    check_bits("sum of 6 rank partials", to_host(dy, 700).data(), sum.data(), 2 * 700);
}

static void test_attn_res(const Mini& mm)
{
    printf("depth attention\n");
    Model ref(mm.c, mm.w);
    Rng r{12};
    const int M = 3;
    const size_t n = static_cast<size_t>(M) * H;
    std::vector<std::vector<float>> all;
    std::vector<const uint16_t*> dall;
    for (int d = 0; d < ink2::MAX_DEPTH; ++d)
    {
        all.push_back(r.bf(n, 0.5f + static_cast<float>(d)));   // rows of very different norms: the mixture must not care
        dall.push_back(to_dev(cells(all.back())));
    }
    const std::vector<float> query = r.f32(3 * static_cast<size_t>(H), 1.0f);
    const std::vector<float> norms = r.bf(3 * static_cast<size_t>(H), 0.2f, 1.0f);
    const float* dq = to_dev(query);
    const uint16_t* dn = to_dev(cells(norms));
    uint16_t* out[3] = {dev_alloc<uint16_t>(n), dev_alloc<uint16_t>(n), dev_alloc<uint16_t>(n)};
    for (int Q : {1, 3})
        for (int D : {1, 2, 4, 9})
            for (int fused = 0; fused < 2; ++fused)
            {
                const std::vector<std::vector<float>> states(all.begin(), all.begin() + (D - 1));
                const std::vector<float> mix = ref.attn_res(states, all[D - 1], query, Q, M);   // [M][Q][H]
                ink2::AttnResArgs a{};
                for (int d = 0; d < D; ++d) a.rows[d] = dall[d];
                a.query = dq;
                a.D = D;
                a.Q = Q;
                a.H = H;
                for (int q = 0; q < Q; ++q)
                {
                    a.out[q] = out[q];
                    a.norm[q] = fused ? dn + static_cast<size_t>(q) * H : nullptr;
                }
                ink2::attn_res(a, M, 0);
                for (int q = 0; q < Q; ++q)
                {
                    std::vector<float> want(n);
                    for (int m = 0; m < M; ++m)
                    {
                        const float* src = &mix[(static_cast<size_t>(m) * Q + q) * H];
                        if (fused) ref.wnorm(src, std::vector<float>(norms.begin() + q * H, norms.begin() + (q + 1) * H), &want[static_cast<size_t>(m) * H]);
                        else memcpy(&want[static_cast<size_t>(m) * H], src, sizeof(float) * H);
                    }
                    char tag[96];
                    snprintf(tag, sizeof tag, "Q %d  D %d  query %d  %s", Q, D, q, fused ? "+ norm" : "mixture");
                    check(tag, compare(to_host(out[q], n), want), n, OP_COS, OP_REL);
                }
            }
}

static void test_gemm_head()
{
    printf("GEMM and the softcapped head\n");
    Rng r{13};
    const int K = 256;
    for (int M : {1, 3, 8, 11, 37})
    {
        const int N = 100, v0 = 690, valid = 750;   // rows 60 .. 99 are past the valid ids; N is not a multiple of 16
        const std::vector<float> x = r.bf(static_cast<size_t>(M) * K, 1.0f), w = r.bf(static_cast<size_t>(N) * K, 0.6f);
        std::vector<float> z(static_cast<size_t>(M) * N), zb(z.size()), lg(z.size());
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
            {
                const float d = ink2ref::dot(&x[static_cast<size_t>(m) * K], &w[static_cast<size_t>(n) * K], K);
                z[static_cast<size_t>(m) * N + n] = d;
                zb[static_cast<size_t>(m) * N + n] = ink2ref::rbf(d);
                lg[static_cast<size_t>(m) * N + n] = v0 + n < valid ? 30.0f * std::tanh(d / 30.0f) : -std::numeric_limits<float>::infinity();
            }
        const uint16_t* dx = to_dev(cells(x));
        const uint16_t* dw = to_dev(cells(w));
        float* df = dev_alloc<float>(z.size() + 16);
        uint16_t* db = dev_alloc<uint16_t>(z.size() + 16);
        char tag[96];
        ink2::gemm_f32(dx, dw, M, K, N, df, 0);
        snprintf(tag, sizeof tag, "gemm fp32   M %2d  N %d", M, N);
        check(tag, compare(to_host(df, z.size()).data(), z.data(), z.size()), z.size(), OP_COS, F32_REL);
        ink2::gemm_bf16(dx, dw, M, K, N, db, 0);
        snprintf(tag, sizeof tag, "gemm BF16   M %2d  N %d", M, N);
        check(tag, compare(to_host(db, z.size()), zb), z.size(), OP_COS, OP_REL);
        expect(to_host(db, z.size() + 16)[z.size()] == 0, "the GEMM wrote past row N - 1");
        ink2::lm_head(dx, dw, M, K, N, v0, valid, df, 0);
        snprintf(tag, sizeof tag, "lm_head     M %2d  rows %d .. %d, valid %d", M, v0, v0 + N, valid);
        const std::vector<float> got = to_host(df, z.size());
        check(tag, compare(got.data(), lg.data(), z.size()), z.size(), OP_COS, F32_REL);
        bool cap = true;
        for (float v : got) cap = cap && (std::isinf(v) || std::fabs(v) <= 30.0f);
        expect(cap, "a logit is outside [-30, 30]");
    }
    // a large pre-activation saturates at 30
    const std::vector<float> x(static_cast<size_t>(K), 8.0f), w(static_cast<size_t>(16) * K, 1.0f);
    float* df = dev_alloc<float>(16);
    ink2::lm_head(to_dev(cells(x)), to_dev(cells(w)), 1, K, 16, 0, 16, df, 0);
    expect(to_host(df, 16)[3] == 30.0f, "softcap(2048) != 30");
}

// Reference convolution of the whole sequence x [n][C] from position 0.
static std::vector<float> ref_sconv_static(const std::vector<float>& w, const std::vector<float>& x, int n, int C)
{
    std::vector<float> hist;
    return Model::sconv_static(w, hist, x, n, C);
}

static void test_sconv(const Mini& mm)
{
    printf("short convolutions\n");
    Rng r{14};
    const int C = H, N = 14, SLOTS = 3;
    const std::vector<float> w = r.bf(static_cast<size_t>(C) * 4, 0.5f);
    const std::vector<float> xa = r.bf(static_cast<size_t>(N) * C, 1.0f), xb = r.bf(static_cast<size_t>(N) * C, 1.0f);
    const std::vector<uint16_t> wa = cells(ref_sconv_static(w, xa, N, C)), wb = cells(ref_sconv_static(w, xb, N, C));
    const std::vector<uint16_t> ca = cells(xa), cb = cells(xb);
    const uint16_t* dw = to_dev(cells(w));
    uint16_t* dx = dev_alloc<uint16_t>(2 * static_cast<size_t>(N) * C);   // a batch holds rows of both sequences
    uint16_t* dy = dev_alloc<uint16_t>(2 * static_cast<size_t>(N) * C);
    DevRows rows;
    rows.alloc(2 * N);
    uint8_t* dacc = dev_alloc<uint8_t>(2 * N);

    // sequence a on slot 2 and b on slot 0, in batches that mix the slots: {a 0..4, b 0..1}, {b 2..2, a 4..5}, {a 5..12, b 3..9}, ..
    {
        uint16_t* hist = dev_alloc<uint16_t>(static_cast<size_t>(SLOTS) * 3 * C);
        CK(cudaMemset(hist, 0xff, static_cast<size_t>(SLOTS) * 3 * C * 2));   // NaN cells: nothing may read past position 0
        std::vector<uint16_t> ga(ca.size()), gb(cb.size());
        int pa = 0, pb = 0;
        const int plan[][2] = {{4, 2}, {1, 1}, {8, 7}, {1, 4}};
        for (int step = 0; step < 4; ++step)
        {
            const int na = plan[step][0], nb = plan[step][1];
            ink2::RowsHost rh;
            std::vector<uint16_t> x;
            if (step & 1) rh.add(0, pb, nb), x.insert(x.end(), cb.begin() + static_cast<size_t>(pb) * C, cb.begin() + static_cast<size_t>(pb + nb) * C);
            rh.add(2, pa, na);
            x.insert(x.end(), ca.begin() + static_cast<size_t>(pa) * C, ca.begin() + static_cast<size_t>(pa + na) * C);
            if (!(step & 1)) rh.add(0, pb, nb), x.insert(x.end(), cb.begin() + static_cast<size_t>(pb) * C, cb.begin() + static_cast<size_t>(pb + nb) * C);
            rows.set(rh);
            put(dx, x);
            ink2::sconv_static(dx, hist, rows.view(), dw, rh.rows(), C, dy, 0);
            ink2::sconv_advance(dx, hist, rows.view(), nullptr, rh.rows(), C, 0);
            const std::vector<uint16_t> y = to_host(dy, static_cast<size_t>(rh.rows()) * C);
            const size_t oa = (step & 1) ? static_cast<size_t>(nb) * C : 0, ob = (step & 1) ? 0 : static_cast<size_t>(na) * C;
            memcpy(&ga[static_cast<size_t>(pa) * C], &y[oa], static_cast<size_t>(na) * C * 2);
            memcpy(&gb[static_cast<size_t>(pb) * C], &y[ob], static_cast<size_t>(nb) * C * 2);
            pa += na;
            pb += nb;
        }
        check_bits("static, 2 slots in mixed batches: slot 2", ga.data(), wa.data(), ga.size() * 2);
        check_bits("static, 2 slots in mixed batches: slot 0", gb.data(), wb.data(), gb.size() * 2);
    }
    // speculative: 5 rows kept, then 8 rows at positions 5 .. 12 of which rows 5, 6 and 7 are right; 3 accepted;
    // the rest in single steps
    {
        uint16_t* hist = dev_alloc<uint16_t>(static_cast<size_t>(SLOTS) * 3 * C);
        std::vector<uint16_t> g(ca.size());
        ink2::RowsHost rh = one_run(1, 0, 5);
        rows.set(rh);
        put(dx, std::vector<uint16_t>(ca.begin(), ca.begin() + 5 * C));
        ink2::sconv_static(dx, hist, rows.view(), dw, 5, C, dy, 0);
        ink2::sconv_advance(dx, hist, rows.view(), nullptr, 5, C, 0);
        memcpy(g.data(), to_host(dy, 5 * static_cast<size_t>(C)).data(), 5 * static_cast<size_t>(C) * 2);
        const std::vector<uint16_t> before = to_host(hist, static_cast<size_t>(SLOTS) * 3 * C);
        std::vector<uint16_t> draft(ca.begin() + 5 * C, ca.begin() + 13 * C);
        for (size_t i = 3 * static_cast<size_t>(C); i < draft.size(); ++i) draft[i] = cb[i];   // wrong rows
        rh = one_run(1, 5, 8);
        rows.set(rh);
        put(dx, draft);
        ink2::sconv_static(dx, hist, rows.view(), dw, 8, C, dy, 0);
        const std::vector<uint16_t> spec = to_host(dy, 8 * static_cast<size_t>(C));
        check_bits("speculative 8 rows: the forward leaves the state", to_host(hist, before.size()).data(), before.data(), before.size() * 2);
        check_bits("speculative 8 rows: the 3 right rows", spec.data(), &wa[5 * static_cast<size_t>(C)], 3 * static_cast<size_t>(C) * 2);
        put(dacc, std::vector<uint8_t>{1, 1, 1, 0, 0, 0, 0, 0});
        ink2::sconv_advance(dx, hist, rows.view(), dacc, 8, C, 0);
        for (int t = 8; t < 13; ++t)
        {
            rh = one_run(1, t, 1);
            rows.set(rh);
            put(dx, std::vector<uint16_t>(ca.begin() + static_cast<size_t>(t) * C, ca.begin() + static_cast<size_t>(t + 1) * C));
            ink2::sconv_static(dx, hist, rows.view(), dw, 1, C, dy, 0);
            ink2::sconv_advance(dx, hist, rows.view(), nullptr, 1, C, 0);
            memcpy(&g[static_cast<size_t>(t) * C], to_host(dy, C).data(), static_cast<size_t>(C) * 2);
        }
        check_bits("3 accepted, then 5 single steps == sequential", &g[8 * static_cast<size_t>(C)], &wa[8 * static_cast<size_t>(C)], 5 * static_cast<size_t>(C) * 2);
    }
    // dynamic: the K convolution of the model's first global layer on all 8 K / V heads, 9 + 5 rows
    {
        Model ref(mm.c, mm.w);
        const Layer& y = mm.w.layers[2];
        const int nk = NKV * HD;
        const std::vector<float> x = r.bf(static_cast<size_t>(N) * nk, 1.0f), in = r.bf(static_cast<size_t>(N) * H, 1.0f);
        std::vector<float> hist;
        const std::vector<float> want = ref.sconv_dynamic(y.k_static, y.k_wproj, hist, x, in, N);
        const uint16_t* dstat = to_dev(cells(y.k_static));
        const uint16_t* dproj = to_dev(cells(y.k_wproj));
        const uint16_t* din = to_dev(cells(in));
        const std::vector<uint16_t> cx = cells(x);
        const uint16_t* dxk = to_dev(cx);
        uint16_t* dh = dev_alloc<uint16_t>(3 * static_cast<size_t>(nk));
        uint16_t* dyk = dev_alloc<uint16_t>(static_cast<size_t>(N) * nk);
        float* dyn = dev_alloc<float>(static_cast<size_t>(N) * NKV * 4);
        ink2::gemm_f32(din, dproj, N, H, NKV * 4, dyn, 0);
        const std::vector<float> hdyn = to_host(dyn, static_cast<size_t>(N) * NKV * 4);
        ink2::RowsHost rh = one_run(0, 0, 9);
        rows.set(rh);
        ink2::sconv_dynamic(dxk, dh, rows.view(), dstat, dyn, 9, nk, HD, dyk, 0);
        ink2::sconv_advance(dxk, dh, rows.view(), nullptr, 9, nk, 0);
        CK(cudaDeviceSynchronize());
        rh = one_run(0, 9, 5);
        rows.set(rh);
        ink2::sconv_dynamic(dxk + 9 * nk, dh, rows.view(), dstat, dyn + 9 * NKV * 4, 5, nk, HD, dyk + 9 * nk, 0);
        const std::vector<uint16_t> got = to_host(dyk, static_cast<size_t>(N) * nk);
        check("dynamic, 9 + 5 rows", compare(got, want), want.size(), OP_COS, OP_REL);
        // from the device's own projections the taps and the sum are the reference's operations
        std::vector<uint16_t> twin(got.size());
        for (int t = 0; t < N; ++t)
            for (int c = 0; c < nk; ++c)
            {
                float s = 0.0f;
                for (int lag = 0; lag < 4 && lag <= t; ++lag) s += (y.k_static[static_cast<size_t>(lag) * nk + c] + hdyn[(static_cast<size_t>(t) * NKV + c / HD) * 4 + lag]) * x[static_cast<size_t>(t - lag) * nk + c];
                twin[static_cast<size_t>(t) * nk + c] = ink2ref::f32_to_bf16((x[static_cast<size_t>(t) * nk + c] + s) / 3.0f);
            }
        check_bits("dynamic, from the device's projections", got.data(), twin.data(), got.size() * 2);
    }
}

// One attention layer of the model against ink2ref::Model::attention, positions 0 .. n in chunks, on T ranks.
static void test_attention_layer(const Mini& mm, int l, int T)
{
    Config c = mm.c;
    c.tp = T;
    Model ref(c, mm.w);
    const Layer& y = mm.w.layers[l];
    const bool glob = c.global[l] != 0;
    Rng r{static_cast<uint64_t>(100 + l)};
    const int n = 21, maxm = 11;
    const std::vector<float> qi = r.bf(static_cast<size_t>(n) * H, 1.0f), ki = glob ? r.bf(qi.size(), 1.0f) : qi, vi = glob ? r.bf(qi.size(), 1.0f) : qi;
    const std::vector<float> want = ref.attention(l, qi, ki, vi, n);
    const uint16_t* dq = to_dev(cells(qi));
    const uint16_t* dk = to_dev(cells(ki));
    const uint16_t* dv = to_dev(cells(vi));
    std::vector<ink2::AttnGeom> g(static_cast<size_t>(T));
    std::vector<ink2::AttnWeights> w(static_cast<size_t>(T));
    std::vector<ink2::AttnState> s(static_cast<size_t>(T));
    std::vector<ink2::AttnWork> wk(static_cast<size_t>(T));
    for (int k = 0; k < T; ++k)
    {
        g[k] = rank_geom(c, l, T, k);
        w[k] = rank_weights(y, g[k], NH);
        s[k] = rank_state(g[k], 2, MAX_PPS);
        wk[k].alloc(g[k], maxm, 0);
    }
    DevRows rows;
    rows.alloc(maxm);
    uint16_t* out = dev_alloc<uint16_t>(static_cast<size_t>(n) * H);
    float* parts = dev_alloc<float>(static_cast<size_t>(T) * maxm * H);
    int at = 0;
    for (int m : {1, 6, 11, 1, 2})   // the window (8) and the scaling base (6) are crossed inside and between chunks
    {
        const ink2::RowsHost rh = one_run(1, at, m);
        rows.set(rh);
        const size_t o = static_cast<size_t>(at) * H;
        for (int k = 0; k < T; ++k)
            ink2::attn_forward(g[k], w[k], s[k], wk[k], rows.view(), rh.max_pos, dq + o, dk + o, dv + o, m, T == 1 ? out + o : nullptr, parts + static_cast<size_t>(k) * m * H, 0);
        if (T > 1) ink2::sum_ranks(parts, T, static_cast<size_t>(m) * H, out + o, 0);
        for (int k = 0; k < T; ++k) ink2::attn_commit(g[k], s[k], wk[k], rows.view(), nullptr, m, 0);
        CK(cudaDeviceSynchronize());
        at += m;
    }
    char tag[96];
    snprintf(tag, sizeof tag, "layer %d (%s), %d rank%s, 21 positions in 5 chunks", l, glob ? "global" : "local", T, T == 1 ? "" : "s");
    check(tag, compare(to_host(out, static_cast<size_t>(n) * H), want), want.size(), 0.9999, 0.015);
    for (int k = 0; k < T; ++k) wk[k].release();
}

// The expert sum over T ranks (the Runner's wiring) against ink2ref::Model::moe with Config::tp = T.
template <class F>
static void test_moe_ranks(const Mini& mm, int l, int T, int M)
{
    Config c = mm.c;
    c.tp = T;
    Model ref(c, mm.w);
    const Layer& y = mm.w.layers[l];
    Rng r{static_cast<uint64_t>(200 + l)};
    const std::vector<float> x = r.bf(static_cast<size_t>(M) * H, 1.0f);
    std::vector<float> want(x.size());
    for (int m = 0; m < M; ++m) ref.moe(l, &x[static_cast<size_t>(m) * H], &want[static_cast<size_t>(m) * H]);
    const uint16_t* dx = to_dev(cells(x));
    const uint16_t* gw = to_dev(cells(y.router_w));
    const float* bias = to_dev(y.router_bias);
    const float* s1 = to_dev(y.s1);
    const float* s2 = to_dev(y.s2);
    uint16_t* out = dev_alloc<uint16_t>(x.size());
    uint16_t* scratch = dev_alloc<uint16_t>(x.size());
    std::vector<ink2::Work> wk(static_cast<size_t>(T));
    MoeSumArgs sa{};
    for (int k = 0; k < T; ++k) wk[k].alloc(M, H, INTER / T, NE, TK, false);
    ink2::route(dx, gw, bias, M, H, NE, TK, c.route_scale, wk[0].score, wk[0].idx, wk[0].wt, wk[0].plan, 0);
    for (int k = 0; k < T; ++k)
    {
        std::vector<uint8_t> f1, f2;
        pack_experts(mm, l, T, k, f1, f2);
        ink2::MoeArgs ma = wk[k].args(to_dev(f1), to_dev(f2), s1, s2, M, T == 1 ? out : scratch);
        ma.plan = wk[0].plan;
        if (M <= 8) ink2::moe_decode<F>(ma, dx, NE, 0);
        else ink2::moe_prefill<F>(ma, dx, NE, 0);
        sa.yp[k] = static_cast<const float*>(wk[k].ypart);
    }
    if (T > 1)
    {
        sa.R = T;
        sa.plan = wk[0].plan;
        sa.M = M;
        sa.Hn = H;
        sa.TKn = TK;
        sa.out = out;
        k_test_moe_sum<<<ink2::cells_grid(x.size()), 256, 0, 0>>>(sa);
    }
    char tag[96];
    snprintf(tag, sizeof tag, "experts layer %d (%s), %d rank%s, M %d", l, F::OUTER ? "NVFP4" : "MXFP8", T, T == 1 ? "" : "s", M);
    check(tag, compare(to_host(out, x.size()), want), x.size(), 0.9995, 0.02);
    for (int k = 0; k < T; ++k) wk[k].release();
}

// Host twin of the attention core from the device's own projections and K / V rows: the reference's loop
// (ink2ref::Model::attention, "scores over the visible keys").
struct CoreIn
{
    ink2::AttnGeom g;
    std::vector<uint16_t> q_raw, r_raw, gate, sink, q_clip, rel_clip, rel_proj;
};
template <class KeyFn>   // key(row, kpos, lkv, is_v) -> the hd cells
static std::vector<float> host_core(const CoreIn& in, const ink2::RowsHost& rh, KeyFn&& key)
{
    const ink2::AttnGeom& g = in.g;
    const int hd = g.hd, rd = g.rel_dim, M = rh.rows();
    const float clip = 2.0f * std::sqrt(static_cast<float>(hd)), rclip = 2.0f * std::sqrt(static_cast<float>(rd));
    const auto f = [](uint16_t c) { return ink2ref::bf16_to_f32(c); };
    std::vector<float> og(static_cast<size_t>(M) * g.nh_q * hd), q(static_cast<size_t>(hd)), rr(static_cast<size_t>(rd)), tmp(static_cast<size_t>(hd > rd ? hd : rd)), sc;
    for (int m = 0; m < M; ++m)
    {
        const int p = rh.pos[m];
        const float tq = g.global ? 1.0f + 0.1f * std::log(std::fmax(1.0f, static_cast<float>(p + 1) / static_cast<float>(g.log_scale_base))) : 1.0f;
        for (int h = 0; h < g.nh_q; ++h)
        {
            const size_t qo = (static_cast<size_t>(m) * g.nh_q + h) * hd, ro = (static_cast<size_t>(m) * g.nh_q + h) * rd;
            for (int d = 0; d < hd; ++d) tmp[d] = f(in.q_raw[qo + d]) * f(in.q_clip[static_cast<size_t>(h) * hd + d]);
            const float qf = std::fmin(1.0f, clip / (ink2ref::l2_norm(tmp.data(), hd) + ink2ref::EPS));
            for (int d = 0; d < hd; ++d) q[d] = ink2ref::rbf(tmp[d] * qf);
            for (int d = 0; d < rd; ++d) tmp[d] = f(in.r_raw[ro + d]) * f(in.rel_clip[static_cast<size_t>(h) * rd + d]);
            const float rf = std::fmin(1.0f, rclip / (ink2ref::l2_norm(tmp.data(), rd) + ink2ref::EPS));
            for (int d = 0; d < rd; ++d) rr[d] = ink2ref::rbf(tmp[d] * rf);
            if (g.global)
            {
                for (int d = 0; d < hd; ++d) q[d] = ink2ref::rbf(q[d] * tq);
                for (int d = 0; d < rd; ++d) rr[d] = ink2ref::rbf(rr[d] * tq);
            }
            const int lkv = (g.q0 + h) / g.gqa - g.kv0, lo = g.global ? 0 : (p - (g.window - 1) > 0 ? p - (g.window - 1) : 0);
            const float sink = f(in.sink[static_cast<size_t>(m) * g.nh_q + h]);
            sc.assign(static_cast<size_t>(p - lo + 1), 0.0f);
            float mx = sink;
            for (int kpos = lo; kpos <= p; ++kpos)
            {
                const uint16_t* kr = key(m, kpos, lkv, false);
                float z = 0.0f;
                for (int d = 0; d < hd; ++d) z += q[d] * f(kr[d]);
                z = z / std::sqrt(static_cast<float>(hd));
                if (p - kpos < g.rel_win)
                {
                    float b = 0.0f;
                    for (int d = 0; d < rd; ++d) b += rr[d] * f(in.rel_proj[static_cast<size_t>(d) * g.rel_win + (p - kpos)]);
                    z += b;
                }
                sc[kpos - lo] = z;
                mx = std::fmax(mx, z);
            }
            double den = std::exp(static_cast<double>(sink) - mx);
            for (float& z : sc) den += std::exp(static_cast<double>(z) - mx);
            std::vector<double> o(static_cast<size_t>(hd), 0.0);
            for (int kpos = lo; kpos <= p; ++kpos)
            {
                const uint16_t* vr = key(m, kpos, lkv, true);
                const double e = std::exp(static_cast<double>(sc[kpos - lo]) - mx) / den;
                for (int d = 0; d < hd; ++d) o[d] += e * f(vr[d]);
            }
            for (int d = 0; d < hd; ++d) og[qo + d] = ink2ref::rbf(ink2ref::rbf(static_cast<float>(o[d])) * ink2ref::sigmoidf(f(in.gate[qo + d])));
        }
    }
    return og;
}

// The attention core where the model above cannot go: long key ranges.
static void test_core_long()
{
    printf("attention core, long key ranges (against the host twin)\n");
    Rng r{15};
    const int Hs = 64, hd = 16, M = 3;
    ink2::AttnGeom g;
    g.H = Hs;
    g.hd = hd;
    g.rel_dim = 16;
    g.rel_win = 1024;
    g.gqa = 3;
    g.nh_q = 6;
    g.q0 = 6;      // a rank that starts at query head 6 = K / V head 2
    g.nh_kv = 2;
    g.kv0 = 2;
    const int nk = g.nh_kv * hd, nq = g.nh_q * hd;
    const float ih = 1.0f / std::sqrt(static_cast<float>(Hs));
    const std::vector<float> wq = r.bf(static_cast<size_t>(nq) * Hs, 4.0f * ih), gate = r.bf(wq.size(), 2.0f * ih), wr = r.bf(static_cast<size_t>(g.nh_q) * g.rel_dim * Hs, 4.0f * ih),
                             sink = r.bf(static_cast<size_t>(g.nh_q) * Hs, 2.0f * ih), wkk = r.bf(static_cast<size_t>(nk) * Hs, 4.0f * ih), wv = r.bf(wkk.size(), 2.0f * ih),
                             wo = r.bf(static_cast<size_t>(Hs) * nq, 0.1f), q_clip = r.bf(nq, 0.5f, 1.0f), rel_clip = r.bf(static_cast<size_t>(g.nh_q) * g.rel_dim, 0.5f, 1.0f),
                             rel_proj = r.bf(static_cast<size_t>(g.rel_dim) * g.rel_win, 0.3f), ksc = r.bf(static_cast<size_t>(nk) * 4, 0.5f), kst = r.bf(4 * static_cast<size_t>(nk), 0.5f),
                             wproj = r.bf(static_cast<size_t>(g.nh_kv) * 4 * Hs, ih), xin = r.bf(static_cast<size_t>(M) * Hs, 1.0f);
    ink2::AttnWeights w;
    w.wq = to_dev(cells(wq));
    w.gate = to_dev(cells(gate));
    w.wr = to_dev(cells(wr));
    w.sink = to_dev(cells(sink));
    w.wk = to_dev(cells(wkk));
    w.wv = to_dev(cells(wv));
    w.wo = to_dev(cells(wo));
    w.q_clip = to_dev(cells(q_clip));
    w.rel_clip = to_dev(cells(rel_clip));
    w.rel_proj = to_dev(cells(rel_proj));
    w.k_sconv = w.v_sconv = to_dev(cells(ksc));
    w.k_static = w.v_static = to_dev(cells(kst));
    w.k_wproj = w.v_wproj = to_dev(cells(wproj));
    const uint16_t* dx = to_dev(cells(xin));
    uint16_t* out = dev_alloc<uint16_t>(static_cast<size_t>(M) * Hs);
    DevRows rows;
    rows.alloc(M);
    CoreIn in;
    in.q_clip = cells(q_clip);
    in.rel_clip = cells(rel_clip);
    in.rel_proj = cells(rel_proj);
    const auto grab = [&](const ink2::AttnWork& wk) {
        in.q_raw = to_host(wk.q_raw, static_cast<size_t>(M) * nq);
        in.r_raw = to_host(wk.r_raw, static_cast<size_t>(M) * g.nh_q * g.rel_dim);
        in.gate = to_host(wk.gate, static_cast<size_t>(M) * nq);
        in.sink = to_host(wk.sink, static_cast<size_t>(M) * g.nh_q);
    };

    // global: 3 rows at positions 130000 .. 130002 of slot 1; every earlier K / V row is random; base 128000
    {
        g.global = true;
        g.window = 1024;
        g.log_scale_base = 128000;
        const int P = 130000, slots = 2, pps = (P + M) / ink2::KV_PAGE_TOKENS + 1, total = slots * pps;
        const size_t page_cells = 2 * static_cast<size_t>(ink2::KV_PAGE_TOKENS) * nk;
        std::vector<uint16_t> pages = cells(r.bf(static_cast<size_t>(total) * page_cells, 1.0f));
        std::vector<int> table(static_cast<size_t>(total));
        for (int i = 0; i < total; ++i) table[i] = static_cast<int>((static_cast<long long>(i) * 7 + 3) % total);
        if (total % 7 == 0) { printf("page count is a multiple of 7\n"); exit(2); }
        ink2::AttnState s;
        s.hist_k = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * nk);
        s.hist_v = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * nk);
        s.pages = to_dev(pages);
        s.page_table = to_dev(table);
        s.max_pps = pps;
        const int smax = (P + M - 1 + ink2::ATTN_SPAN) / ink2::ATTN_SPAN;
        ink2::AttnWork wk;
        wk.alloc(g, M, static_cast<size_t>(2) * g.nh_q * smax * (hd + 2));   // room for 2 of the 3 rows: two launches
        const ink2::RowsHost rh = one_run(1, P, M);
        rows.set(rh);
        ink2::attn_forward(g, w, s, wk, rows.view(), rh.max_pos, dx, dx, dx, M, out, nullptr, 0);
        grab(wk);
        in.g = g;
        pages = to_host(s.pages, pages.size());
        const std::vector<uint16_t> kv_new = to_host(wk.kv_new, static_cast<size_t>(M) * 2 * nk);
        bool wrote = true;
        const auto key = [&](int m, int kpos, int lkv, bool is_v) {
            const size_t pg = static_cast<size_t>(table[rh.slot[m] * pps + kpos / ink2::KV_PAGE_TOKENS]);
            return &pages[((pg * 2 + (is_v ? 1 : 0)) * ink2::KV_PAGE_TOKENS + kpos % ink2::KV_PAGE_TOKENS) * nk + static_cast<size_t>(lkv) * hd];
        };
        for (int m = 0; m < M; ++m)
            for (int lkv = 0; lkv < g.nh_kv; ++lkv)
                for (int kv = 0; kv < 2; ++kv) wrote = wrote && memcmp(key(m, P + m, lkv, kv != 0), &kv_new[(static_cast<size_t>(m) * 2 + kv) * nk + static_cast<size_t>(lkv) * hd], hd * 2) == 0;
        expect(wrote, "the rows' K / V are not in their pages");
        const std::vector<float> want = host_core(in, rh, key);
        check("global, positions 130000 .. 130002, 508 spans, 2 launches", compare(to_host(wk.og, want.size()), want), want.size(), OP_COS, OP_REL);
        expect(ink2::attn_tau(P, g.log_scale_base) > 1.001f && ink2::attn_tau(127999, g.log_scale_base) == 1.0f, "tau");
        wk.release();
    }
    // local: window 600, 3 rows at positions 700 .. 702 of slot 1: spans of 256, 256 and 88 keys, the last rows of
    // the range from the batch itself and the rest from the ring
    {
        g.global = false;
        g.window = 600;
        g.log_scale_base = 128000;
        const int P = 700, slots = 2;
        const std::vector<uint16_t> ring = cells(r.bf(static_cast<size_t>(slots) * 2 * g.window * nk, 1.0f));
        ink2::AttnState s;
        s.hist_k = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * nk);
        s.hist_v = dev_alloc<uint16_t>(static_cast<size_t>(slots) * 3 * nk);
        s.ring = to_dev(ring);
        ink2::AttnWork wk;
        wk.alloc(g, M, static_cast<size_t>(M) * g.nh_q * 3 * (hd + 2));
        const ink2::RowsHost rh = one_run(1, P, M);
        rows.set(rh);
        ink2::attn_forward(g, w, s, wk, rows.view(), rh.max_pos, dx, dx, dx, M, out, nullptr, 0);
        grab(wk);
        in.g = g;
        const std::vector<uint16_t> kv_new = to_host(wk.kv_new, static_cast<size_t>(M) * 2 * nk);
        check_bits("local: the forward leaves the ring", to_host(s.ring, ring.size()).data(), ring.data(), ring.size() * 2);
        const auto key = [&](int m, int kpos, int lkv, bool is_v) {
            if (kpos >= P) return &kv_new[(static_cast<size_t>(kpos - P) * 2 + (is_v ? 1 : 0)) * nk + static_cast<size_t>(lkv) * hd];
            return &ring[((static_cast<size_t>(rh.slot[m]) * 2 + (is_v ? 1 : 0)) * g.window + kpos % g.window) * nk + static_cast<size_t>(lkv) * hd];
        };
        const std::vector<float> want = host_core(in, rh, key);
        check("local, window 600, positions 700 .. 702, 3 spans", compare(to_host(wk.og, want.size()), want), want.size(), OP_COS, OP_REL);
        // commit rows 0 and 1: their ring entries change, nothing else
        uint8_t* dacc = to_dev(std::vector<uint8_t>{1, 1, 0});
        ink2::attn_commit(g, s, wk, rows.view(), dacc, M, 0);
        std::vector<uint16_t> ring2 = ring;
        for (int m = 0; m < 2; ++m)
            for (int kv = 0; kv < 2; ++kv)
                memcpy(&ring2[((static_cast<size_t>(1) * 2 + kv) * g.window + (P + m) % g.window) * nk], &kv_new[(static_cast<size_t>(m) * 2 + kv) * nk], static_cast<size_t>(nk) * 2);
        check_bits("local: commit of 2 of 3 rows", to_host(s.ring, ring.size()).data(), ring2.data(), ring.size() * 2);
        wk.release();
    }
    // grouped global core vs scalar core: hd = 128, TP=6 rank-0 uneven GQA (gq = 6 and 2), two slots and span boundary 255..260 + 4090..4105
    {
        ink2::AttnGeom gg;
        gg.H = 128;
        gg.hd = 128;
        gg.rel_dim = 16;
        gg.rel_win = 1024;
        gg.gqa = 6;
        gg.nh_q = 8;
        gg.q0 = 0;
        gg.nh_kv = 2;
        gg.kv0 = 0;
        gg.window = 1024;
        gg.log_scale_base = 4096;
        gg.global = true;
        const int Mg = 22, nk2 = gg.nh_kv * gg.hd, nq2 = gg.nh_q * gg.hd;
        const int slots = 2, pps = 4200 / ink2::KV_PAGE_TOKENS + 2, total = slots * pps;
        const size_t page_cells = 2 * static_cast<size_t>(ink2::KV_PAGE_TOKENS) * nk2;
        const std::vector<uint16_t> pages = cells(r.bf(static_cast<size_t>(total) * page_cells, 1.0f));
        std::vector<int> table(static_cast<size_t>(total));
        for (int i = 0; i < total; ++i) table[i] = (i * 5 + 1) % total;
        ink2::RowsHost rh;
        rh.add(0, 254, 6);     // crosses 255 -> 256 (ns = 1 and ns = 2 in the same CTA)
        rh.add(1, 4090, 16);   // second slot, crosses page boundary 4095 -> 4096, total 22 rows * 6 = 132 > 4 * GRP_Q
        DevRows gr;
        gr.alloc(Mg);
        gr.set(rh);
        const int smax = ink2::attn_splits(true, gg.window, rh.max_pos);
        const uint16_t* q_raw = to_dev(cells(r.bf(static_cast<size_t>(Mg) * nq2, 1.0f)));
        const uint16_t* r_raw = to_dev(cells(r.bf(static_cast<size_t>(Mg) * gg.nh_q * gg.rel_dim, 1.0f)));
        const uint16_t* gate2 = to_dev(cells(r.bf(static_cast<size_t>(Mg) * nq2, 1.0f)));
        const uint16_t* sink2 = to_dev(cells(r.bf(static_cast<size_t>(Mg) * gg.nh_q, 1.0f)));
        const uint16_t* qc2 = to_dev(cells(r.bf(nq2, 0.5f, 1.0f)));
        const uint16_t* rc2 = to_dev(cells(r.bf(static_cast<size_t>(gg.nh_q) * gg.rel_dim, 0.5f, 1.0f)));
        const uint16_t* rp2 = to_dev(cells(r.bf(static_cast<size_t>(gg.rel_dim) * gg.rel_win, 0.3f)));
        uint16_t* og_ref = dev_alloc<uint16_t>(static_cast<size_t>(Mg) * nq2);
        uint16_t* og_grp = dev_alloc<uint16_t>(static_cast<size_t>(Mg) * nq2);
        float* part_buf = dev_alloc<float>(static_cast<size_t>(Mg) * gg.nh_q * smax * (gg.hd + 2));
        ink2::AttnCoreArgs ca{q_raw, r_raw, gate2, sink2, qc2, rc2, rp2, nullptr, nullptr, to_dev(pages), to_dev(table),
                              gr.view(), og_ref, part_buf, 0, Mg, gg.nh_q, gg.q0, gg.gqa, gg.kv0, nk2, gg.hd,
                              gg.rel_dim, gg.rel_win, gg.window, gg.log_scale_base, 1, pps, smax};
        ink2::k_ink2_attn_core<<<dim3(Mg * gg.nh_q, smax), ink2::ACT>>>(ca);
        ink2::k_ink2_attn_merge<<<Mg * gg.nh_q, ink2::ACT>>>(ca);
        CK(cudaMemset(part_buf, 0xff, static_cast<size_t>(Mg) * gg.nh_q * smax * (gg.hd + 2) * sizeof(float)));
        ca.og = og_grp;
        const int max_ngrp = (Mg * 6 + ink2::GRP_Q - 1) / ink2::GRP_Q;
        ink2::k_ink2_attn_global_grp<<<dim3(max_ngrp, smax, gg.nh_kv), ink2::ACT>>>(ca);
        ink2::k_ink2_attn_merge<<<Mg * gg.nh_q, ink2::ACT>>>(ca);
        check_bits("global grouped core == scalar core (hd 128, TP=6 GQA, 2 slots, 255/256 + 4096)",
                   to_host(og_grp, static_cast<size_t>(Mg) * nq2).data(),
                   to_host(og_ref, static_cast<size_t>(Mg) * nq2).data(),
                   static_cast<size_t>(Mg) * nq2 * 2);
    }
}

// ---------------------------------------------------------------------------------------------- part 2: the model
constexpr int NT = 24;
struct Seq
{
    std::vector<int> toks;        // [NT]
    std::vector<float> vis;       // [NT][H]
    std::vector<uint16_t> visc;
};
static Seq sequence(uint64_t seed)
{
    Rng r{seed};
    Seq s;
    s.toks.resize(NT);
    for (int& t : s.toks)
        do t = static_cast<int>(r.next() % VALID);
        while (t == IMG);
    s.toks[3] = s.toks[9] = s.toks[10] = s.toks[17] = IMG;
    s.vis = r.bf(static_cast<size_t>(NT) * H, 1.5f);
    s.visc = cells(s.vis);
    return s;
}
// Tokens at .. at + n of the sequence on a slot, kept. Returns the logits [n][VOCAB].
static std::vector<float> step(Runner& g, const Seq& s, int slot, int at, int n)
{
    const std::vector<float> lg = g.forward(one_run(slot, at, n), &s.toks[at], &s.visc[static_cast<size_t>(at) * H]);
    g.commit();
    return lg;
}
static std::vector<float> run(Runner& g, const Seq& s, int slot, std::initializer_list<int> chunks)
{
    std::vector<float> out;
    int at = 0;
    for (int n : chunks)
    {
        const std::vector<float> lg = step(g, s, slot, at, n);
        out.insert(out.end(), lg.begin(), lg.end());
        at += n;
    }
    return out;
}
static void check_valid_mask(const std::vector<float>& lg)
{
    bool ok = true;
    for (size_t i = 0; i < lg.size(); ++i)
    {
        const bool valid = static_cast<int>(i % VOCAB) < VALID;
        ok = ok && (valid ? std::isfinite(lg[i]) && std::fabs(lg[i]) <= 30.0f : std::isinf(lg[i]) && lg[i] < 0.0f);
    }
    expect(ok, "logits: finite and within the softcap below valid_vocab, -inf from there on");
}

// The reference's forward over the sequence, one stage at a time, every stage on the device's own inputs of that
// stage (dev = Runner::trace of the same prefill, lg = its logits). A difference cannot build up from stage to stage,
// so each stage holds its operator's thresholds. One line per kind of stage: the worst layer.
static void stage_by_stage(const Config& c, const Weights& W, const Seq& s, const std::vector<std::vector<float>>& dev, const std::vector<float>& lg)
{
    struct Kind
    {
        const char* name;
        double min_cos, max_rel;
        Stat worst;
        size_t n;
    };
    enum { K_IN, K_ATT, K_CONV, K_RES, K_MOE, K_HEAD, K_N };
    Kind kinds[K_N] = {{"inputs of a stage (norms, depth attention)", OP_COS, OP_REL, {2, 0, 0, true}, 0},
                       {"attention", 0.9999, 0.015, {2, 0, 0, true}, 0},
                       {"convolutions of the residual", OP_COS, OP_REL, {2, 0, 0, true}, 0},
                       {"residual steps", OP_COS, OP_REL, {2, 0, 0, true}, 0},
                       {"expert sums", 0.9995, 0.02, {2, 0, 0, true}, 0},
                       {"logits from the head's input", OP_COS, F32_REL, {2, 0, 0, true}, 0}};
    size_t at = 0;
    bool short_trace = false;
    // compares the reference's v with the device's next stage, then continues from the device's
    const auto stage = [&](int k, std::vector<float>& v) {
        if (at >= dev.size() || dev[at].size() != v.size())
        {
            short_trace = true;
            return;
        }
        const Stat st = compare(dev[at].data(), v.data(), v.size());
        Kind& q = kinds[k];
        q.worst.cos = std::min(q.worst.cos, st.cos);
        q.worst.rel = std::max(q.worst.rel, st.rel);
        q.worst.differ += st.differ;
        q.worst.mask_ok = q.worst.mask_ok && st.mask_ok;
        q.n += v.size();
        v = dev[at++];
    };
    Model R(c, W);
    const int M = NT;
    std::vector<float> h(static_cast<size_t>(M) * H);
    for (int m = 0; m < M; ++m)
        if (s.toks[m] == c.image_token_id)
            for (int i = 0; i < H; ++i) h[static_cast<size_t>(m) * H + i] = ink2ref::rbf(s.vis[static_cast<size_t>(m) * H + i]);
        else R.wnorm(&W.embed[static_cast<size_t>(s.toks[m]) * H], W.embed_norm, &h[static_cast<size_t>(m) * H]);
    stage(K_IN, h);
    std::vector<std::vector<float>> states, hist_a(c.n_layers), hist_m(c.n_layers);
    for (int l = 0; l < c.n_layers; ++l)
    {
        const Layer& w = W.layers[l];
        const bool glob = c.global[l] != 0;
        std::vector<float> qi(h.size()), ki, vi;
        if (l == 0 || !glob)
        {
            const std::vector<float> x = l == 0 ? h : R.attn_res(states, h, w.attn_query, 1, M);
            for (int m = 0; m < M; ++m) R.wnorm(&x[static_cast<size_t>(m) * H], w.attn_norm, &qi[static_cast<size_t>(m) * H]);
            stage(K_IN, qi);
        }
        else
        {
            const std::vector<float> x = R.attn_res(states, h, w.attn_query, 3, M);
            ki.resize(h.size());
            vi.resize(h.size());
            for (int m = 0; m < M; ++m)
            {
                const float* xr = &x[static_cast<size_t>(m) * 3 * H];
                R.wnorm(xr, w.q_norm, &qi[static_cast<size_t>(m) * H]);
                R.wnorm(xr + H, w.k_norm, &ki[static_cast<size_t>(m) * H]);
                R.wnorm(xr + 2 * H, w.v_norm, &vi[static_cast<size_t>(m) * H]);
            }
            stage(K_IN, qi);
            stage(K_IN, ki);
            stage(K_IN, vi);
        }
        std::vector<float> att = R.attention(l, qi, glob ? ki : qi, glob ? vi : qi, M);
        stage(K_ATT, att);
        std::vector<float> a = Model::sconv_static(w.attn_sconv, hist_a[l], att, M, H);
        stage(K_CONV, a);
        std::vector<float> mid(h.size());
        if (l % c.block == 0)
        {
            states.push_back(h);
            mid = a;
        }
        else
            for (size_t i = 0; i < h.size(); ++i) mid[i] = ink2ref::rbf(h[i] + a[i]);
        stage(K_RES, mid);
        const std::vector<float> xr = R.attn_res(states, mid, w.mlp_query, 1, M);
        std::vector<float> xn(h.size()), mo(h.size());
        for (int m = 0; m < M; ++m) R.wnorm(&xr[static_cast<size_t>(m) * H], w.mlp_norm, &xn[static_cast<size_t>(m) * H]);
        stage(K_IN, xn);
        for (int m = 0; m < M; ++m) R.moe(l, &xn[static_cast<size_t>(m) * H], &mo[static_cast<size_t>(m) * H]);
        stage(K_MOE, mo);
        std::vector<float> ms = Model::sconv_static(w.mlp_sconv, hist_m[l], mo, M, H);
        stage(K_CONV, ms);
        for (size_t i = 0; i < h.size(); ++i) h[i] = ink2ref::rbf(mid[i] + ms[i]);
        stage(K_RES, h);
    }
    const std::vector<float> xf = R.attn_res(states, h, W.output_query, 1, M);
    std::vector<float> hf(h.size()), want(static_cast<size_t>(M) * VOCAB);
    for (int m = 0; m < M; ++m) R.wnorm(&xf[static_cast<size_t>(m) * H], W.final_norm, &hf[static_cast<size_t>(m) * H]);
    stage(K_IN, hf);
    for (int m = 0; m < M; ++m)
        for (int v = 0; v < VOCAB; ++v)
        {
            const float z = ink2ref::dot(&hf[static_cast<size_t>(m) * H], &W.lm_head[static_cast<size_t>(v) * H], H);
            want[static_cast<size_t>(m) * VOCAB + v] = v < VALID ? 30.0f * std::tanh(z / 30.0f) : -std::numeric_limits<float>::infinity();
        }
    {
        const Stat st = compare(lg.data(), want.data(), want.size());
        kinds[K_HEAD].worst = st;
        kinds[K_HEAD].n = want.size();
    }
    expect(!short_trace && at == dev.size(), "stage by stage: the device's trace has every stage");
    char tag[96];
    for (const Kind& q : kinds)
    {
        snprintf(tag, sizeof tag, "  stage by stage: %s", q.name);
        check(tag, q.worst, q.n, q.min_cos, q.max_rel);
    }
}

static std::vector<float> test_model(const Mini& mm, int T, const std::vector<float>* tp1)
{
    printf("6-layer model, %d rank%s\n", T, T == 1 ? "" : "s");
    Config c = mm.c;
    c.tp = T;
    const Seq sa = sequence(4242), sb = sequence(777);
    const size_t all = static_cast<size_t>(NT) * VOCAB;
    std::vector<float> wa, wb;
    {
        Model ref(c, mm.w);
        wa = ref.forward(sa.toks.data(), NT, sa.vis.data());
    }
    {
        Model ref(c, mm.w);
        wb = ref.forward(sb.toks.data(), NT, sb.vis.data());
    }
    char tag[96];
    const auto name = [&](const char* what) {
        snprintf(tag, sizeof tag, "%s", what);
        return tag;
    };

    // one prefill of 24 tokens, 4 of them image tokens; the window (8) and the scaling base (6) are far behind
    Runner pre(mm, T, 1, NT);
    std::vector<std::vector<float>> trace;
    pre.trace = &trace;
    const std::vector<float> full = run(pre, sa, 0, {NT});
    pre.trace = nullptr;
    check_valid_mask(full);
    check(name("prefill of 24 tokens"), compare(full.data(), wa.data(), all), all, E2E_COS, E2E_REL);
    stage_by_stage(c, mm.w, sa, trace, full);
    if (tp1) check(name("  against the 1-rank device run (another sum order)"), compare(full.data(), tp1->data(), all), all, E2E_COS, E2E_REL);

    // 24 single steps, and chunks: the device's own bits do not depend on the batch
    Runner dec(mm, T, 2, 9);
    const std::vector<float> steps = run(dec, sa, 1, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    check(name("24 single steps"), compare(steps.data(), wa.data(), all), all, E2E_COS, E2E_REL);
    check_bits("  single steps == the prefill", steps.data(), full.data(), all * 4);
    Runner chk(mm, T, 1, 9);
    const std::vector<float> chunks = run(chk, sa, 0, {5, 9, 1, 7, 2});
    check(name("chunks 5 + 9 + 1 + 7 + 2"), compare(chunks.data(), wa.data(), all), all, E2E_COS, E2E_REL);
    check_bits("  chunks == the prefill", chunks.data(), full.data(), all * 4);

    // two slots, their rows in one batch in either order, at different positions
    {
        Runner two(mm, T, 3, 12);
        std::vector<float> ga(all), gb(all);
        int pa = 0, pb = 0, k = 0;
        const int plan[][2] = {{3, 4}, {5, 2}, {1, 1}, {7, 5}, {8, 4}, {0, 8}};
        for (const auto& st : plan)
        {
            ink2::RowsHost rh;
            std::vector<int> tk;
            std::vector<uint16_t> vis;
            const auto add = [&](const Seq& s, int slot, int at, int n) {
                rh.add(slot, at, n);
                tk.insert(tk.end(), s.toks.begin() + at, s.toks.begin() + at + n);
                vis.insert(vis.end(), s.visc.begin() + static_cast<size_t>(at) * H, s.visc.begin() + static_cast<size_t>(at + n) * H);
            };
            const bool a_first = (k++ & 1) == 0;
            if (a_first) add(sa, 2, pa, st[0]);
            add(sb, 0, pb, st[1]);
            if (!a_first) add(sa, 2, pa, st[0]);
            const std::vector<float> lg = two.forward(rh, tk.data(), vis.data());
            two.commit();
            const size_t oa = a_first ? 0 : static_cast<size_t>(st[1]) * VOCAB, ob = a_first ? static_cast<size_t>(st[0]) * VOCAB : 0;
            memcpy(&ga[static_cast<size_t>(pa) * VOCAB], &lg[oa], static_cast<size_t>(st[0]) * VOCAB * 4);
            memcpy(&gb[static_cast<size_t>(pb) * VOCAB], &lg[ob], static_cast<size_t>(st[1]) * VOCAB * 4);
            pa += st[0];
            pb += st[1];
        }
        check(name("two slots in one batch: slot 2"), compare(ga.data(), wa.data(), all), all, E2E_COS, E2E_REL);
        check(name("two slots in one batch: slot 0"), compare(gb.data(), wb.data(), all), all, E2E_COS, E2E_REL);
        check_bits("  slot 2 == the prefill", ga.data(), full.data(), all * 4);
    }

    // speculative steps on slot 1 of `dec`'s twin: 11 tokens kept (past the window), then
    //   a draft of 8 with nothing accepted: no state cell moves, and the next steps are the sequential bits
    //   a draft of 8 whose first 3 tokens are right: 3 accepted, then single steps
    {
        Runner sp(mm, T, 2, 9);
        run(sp, sa, 1, {4, 7});
        int draft[8];
        const auto make_draft = [&](int at, int right) {
            for (int i = 0; i < 8; ++i)
            {
                const int t = sa.toks[at + i];
                draft[i] = i < right ? t : (t == IMG ? 11 : ((t + 1 + i) % VALID == IMG ? 12 : (t + 1 + i) % VALID));
            }
        };
        const std::vector<uint16_t> before = sp.snapshot(1, 11);
        make_draft(11, 0);
        sp.forward(one_run(1, 11, 8), draft, &sa.visc[static_cast<size_t>(11) * H]);
        const std::vector<uint16_t> after = sp.snapshot(1, 11);
        check_bits("draft of 8 at position 11, 0 accepted: the state", after.data(), before.data(), before.size() * 2);
        std::vector<float> got;
        for (int t = 11; t < 14; ++t)
        {
            const std::vector<float> lg = step(sp, sa, 1, t, 1);
            got.insert(got.end(), lg.begin(), lg.end());
        }
        check(name("  the 3 steps after it"), compare(got.data(), &wa[static_cast<size_t>(11) * VOCAB], 3 * static_cast<size_t>(VOCAB)), 3 * static_cast<size_t>(VOCAB), E2E_COS, E2E_REL);
        check_bits("  the 3 steps after it == the prefill", got.data(), &full[static_cast<size_t>(11) * VOCAB], 3 * static_cast<size_t>(VOCAB) * 4);

        make_draft(14, 3);
        const std::vector<float> spec = sp.forward(one_run(1, 14, 8), draft, &sa.visc[static_cast<size_t>(14) * H]);
        check_bits("draft of 8 at position 14: its 3 right rows == the prefill", spec.data(), &full[static_cast<size_t>(14) * VOCAB], 3 * static_cast<size_t>(VOCAB) * 4);
        expect(memcmp(&spec[3 * static_cast<size_t>(VOCAB)], &full[static_cast<size_t>(17) * VOCAB], static_cast<size_t>(VOCAB) * 4) != 0, "the first wrong row of the draft has the right logits");
        const std::vector<uint8_t> acc{1, 1, 1, 0, 0, 0, 0, 0};
        sp.commit(&acc);
        got.clear();
        for (int t = 17; t < NT; ++t)
        {
            const std::vector<float> lg = step(sp, sa, 1, t, 1);
            got.insert(got.end(), lg.begin(), lg.end());
        }
        const size_t n = static_cast<size_t>(NT - 17) * VOCAB;
        check(name("3 accepted, then 7 single steps"), compare(got.data(), &wa[static_cast<size_t>(17) * VOCAB], n), n, E2E_COS, E2E_REL);
        check_bits("  == the prefill", got.data(), &full[static_cast<size_t>(17) * VOCAB], n * 4);
    }
    return full;
}

int main()
{
    Mini mm;
    build_mini(mm);
    printf("== part 1: one operator at a time ==\n");
    test_norms(mm);
    test_attn_res(mm);
    test_gemm_head();
    test_sconv(mm);
    printf("one attention layer against the reference's\n");
    for (int T : {1, 4, 6})
        for (int l : {0, 2, 3, 5}) test_attention_layer(mm, l, T);
    printf("the expert sum over ranks against the reference's\n");
    for (int T : {1, 4, 6})
        for (int M : {3, 11})
        {
            test_moe_ranks<nv4::Frag>(mm, 1, T, M);
            test_moe_ranks<ink2::Mx8Frag>(mm, 4, T, M);
        }
    test_core_long();
    printf("== part 2: the 6-layer model ==\n");
    const std::vector<float> one = test_model(mm, 1, nullptr);
    test_model(mm, 6, &one);
    test_model(mm, 4, &one);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    if (g_fail) printf("FAILED: %d check%s\n", g_fail, g_fail == 1 ? "" : "s");
    else printf("ALL PASS\n");
    return g_fail ? 1 : 0;
}
