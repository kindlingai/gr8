// Inkling-2-Small: the step of the 48 text layers over the operators of kernels/ink2_*.cuh and attn/ink2_attn.cuh.
//
// One Engine runs the ranks it holds: all of them on one GPU (ink2_sim.cu) or one rank of a cluster (ink2_m4.cu). What
// every rank holds a copy of (the norms, the depth attention, the convolutions of the residual, the router) runs once;
// what is sharded (the attention heads, the expert intermediate columns, the lm_head rows) runs per rank and leaves an
// fp32 partial. The partials of all ranks are summed in rank order, so the result does not depend on where a rank ran:
//   attention  [M][H] fp32 per rank           -> sum over the ranks                    -> BF16
//   experts    [M][TK][H] fp32 per rank       -> per pick: sum over the ranks, then the picks in order -> BF16
//   head       per row: the rank's top NC logits, its largest logit and sum of exp     -> merged on the host
// With one rank the attention and the experts write BF16 directly. `gather` (set by ink2_m4.cu) exchanges the partials
// with the other ranks between the per-rank work and the sum; without it every rank's partial is already in place.
//
// The step is eager: the attention launches depend on the largest position of the step, so no graph is captured.
// The dimensions are read from the checkpoint's config.json and the shapes of the planned image, so a miniature
// checkpoint runs through the same code.
#pragma once
#include "../kernels/nvfp4.cuh"
#include "../kernels/ink2_moe.cuh"
#include "../attn/ink2_attn.cuh"
#include "../model_geom_ink2.h"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"
#include "../knobs-rs/gr8_knobs.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace ink2rt
{
[[noreturn]] inline void die(const std::string& m)
{
    fprintf(stderr, "ink2: %s\n", m.c_str());
    exit(2);
}
inline void ck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}
#define INK2RT_CK(x) ::ink2rt::ck((x), #x)

template <class T> T* dalloc(size_t n)
{
    void* p = nullptr;
    INK2RT_CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    INK2RT_CK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
    return static_cast<T*>(p);
}

// ------------------------------------------------------------------------------------------------ config
// The first integer (or number) after "key" in text from `from` on; def when the key is absent.
inline double json_num(const std::string& text, size_t from, const char* key, double def)
{
    const std::string k = std::string("\"") + key + "\"";
    size_t p = text.find(k, from);
    if (p == std::string::npos) return def;
    p = text.find(':', p + k.size());
    if (p == std::string::npos) return def;
    const char* s = text.c_str() + p + 1;
    char* end = nullptr;
    const double v = strtod(s, &end);
    return end == s ? def : v;
}

struct Cfg
{
    int H = 0, heads = 0, kv_heads = 0, hd = 0, rel_dim = 0, rel_win = 0, NE = 0, TK = 0, I = 0, vocab = 0, valid = 0, n_layers = 0;
    int window = geom::WINDOW, log_base = geom::LOG_SCALE_BASE, block = geom::ATTN_RES_BLOCK, image_token = geom::TOK_IMAGE;
    float route_scale = geom::ROUTE_SCALE;
};

inline std::string json_section(const std::string& text, const char* key)
{
    const std::string k = std::string("\"") + key + "\"";
    size_t p = text.find(k);
    if (p == std::string::npos) return text;
    p = text.find('{', p + k.size());
    if (p == std::string::npos) return text;
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = p; i < text.size(); ++i)
    {
        const char ch = text[i];
        if (in_str)
        {
            if (esc) esc = false;
            else if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
        }
        else if (ch == '"') in_str = true;
        else if (ch == '{') ++depth;
        else if (ch == '}' && --depth == 0) return text.substr(p, i - p + 1);
    }
    return text.substr(p);
}

// config.json of the checkpoint: its text_config when it has one, else the top level.
inline Cfg read_config(const std::string& dir)
{
    std::ifstream f(dir + "/config.json");
    if (!f) die(dir + "/config.json: cannot open");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string sc = json_section(ss.str(), "text_config");
    const auto opt_int = [&](const char* key, int def) {
        return static_cast<int>(json_num(sc, 0, key, def));
    };
    const auto need = [&](const char* key) {
        const int v = opt_int(key, -1);
        if (v < 1) die(dir + "/config.json: no " + key);
        return v;
    };
    Cfg c;
    c.H = need("hidden_size");
    c.heads = need("num_attention_heads");
    c.kv_heads = need("num_key_value_heads");
    c.hd = need("head_dim");
    c.NE = need("n_routed_experts");
    c.TK = need("num_experts_per_tok");
    c.I = opt_int("moe_intermediate_size", -1);
    if (c.I < 1) c.I = need("intermediate_size");
    c.vocab = need("vocab_size");
    c.n_layers = need("num_hidden_layers");
    c.window = opt_int("sliding_window", opt_int("sliding_window_size", geom::WINDOW));
    c.log_base = opt_int("log_scaling_n_floor", geom::LOG_SCALE_BASE);
    c.block = opt_int("n_layers_per_attn_res_block", geom::ATTN_RES_BLOCK);
    c.route_scale = static_cast<float>(json_num(sc, 0, "route_scale", geom::ROUTE_SCALE));
    c.valid = opt_int("unpadded_vocab_size", std::min(c.vocab, static_cast<int>(geom::VALID_VOCAB)));
    if (c.valid > c.vocab || c.valid < 1) c.valid = c.vocab;
    if (c.heads % c.kv_heads || c.H % 32 || c.hd % 16 || c.hd > ink2::ATTN_MAX_HD) die(dir + "/config.json: heads, hidden_size or head_dim");
    return c;
}

// "all", or a list of layer numbers in ascending order
inline std::vector<int> parse_layers(const std::string& s, int n_layers)
{
    std::vector<int> v;
    if (s == "all" || s.empty())
    {
        for (int l = 0; l < n_layers; ++l) v.push_back(l);
        return v;
    }
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        char* end = nullptr;
        const long l = strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end || l < 0 || l >= n_layers || (!v.empty() && l <= v.back())) die("--layers " + s + ": want all, or ascending layer numbers below " + std::to_string(n_layers));
        v.push_back(static_cast<int>(l));
    }
    if (v.empty()) die("--layers: no layer");
    return v;
}

// ------------------------------------------------------------------------------------------------ the image of a rank
// The planned image of one rank (engine/loader-rs/src/ink2.rs): its manifest, and the tensors on the device. whole():
// the image in one allocation through the parallel loader. Without it a tensor is read when it is first asked for.
struct Image
{
    g6::Pack pack;
    void* plan = nullptr;
    uint8_t* dev = nullptr;
    int rank = 0;
    std::vector<std::pair<std::string, void*>> loaded;

    struct Sink
    {
        uint8_t* dev = nullptr;
        std::mutex mu;
        std::vector<cudaStream_t> streams;
        static void* alloc(void*, size_t n)
        {
            void* p = nullptr;
            return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
        }
        static void release(void*, void* p) { cudaFreeHost(p); }
        static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n)
        {
            Sink& s = *static_cast<Sink*>(ctx);
            cudaStream_t st;
            {
                std::lock_guard<std::mutex> lk(s.mu);
                while (static_cast<int>(s.streams.size()) <= w)
                {
                    cudaStream_t x;
                    if (cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking) != cudaSuccess) return 2;
                    s.streams.push_back(x);
                }
                st = s.streams[static_cast<size_t>(w)];
            }
            if (cudaMemcpyAsync(s.dev + off, src, n, cudaMemcpyHostToDevice, st) != cudaSuccess) return 3;
            return cudaStreamSynchronize(st) == cudaSuccess ? 0 : 4;
        }
    };

    void open(const std::string& dir, int nr, int r)
    {
        rank = r;
        const std::string sfx = ".manifest.json";
        std::string p = knob::tmpdir() + "/ink2-rank" + std::to_string(r) + ".XXXXXX" + sfx;
        const int fd = mkstemps(&p[0], static_cast<int>(sfx.size()));
        if (fd < 0) die(p + ": " + strerror(errno));
        close(fd);
        char err[1024] = "";
        plan = g6l_ink2_plan(dir.c_str(), static_cast<uint32_t>(nr), static_cast<uint32_t>(r), p.c_str(), err, sizeof err);
        if (!plan)
        {
            unlink(p.c_str());
            die("rank " + std::to_string(r) + ": plan of " + dir + ": " + err);
        }
        std::string e;
        const bool ok = pack.open_manifest(p, e);
        unlink(p.c_str());
        if (!ok) die("rank " + std::to_string(r) + ": planned manifest: " + e);
    }
    void whole()
    {
        void* d = nullptr;
        INK2RT_CK(cudaMalloc(&d, pack.bytes()));
        Sink sink;
        sink.dev = static_cast<uint8_t*>(d);
        const g6l_sink s{&sink, &Sink::alloc, &Sink::release, &Sink::copy};
        g6l_stats st{};
        char err[1024] = "";
        if (g6l_plan_load(plan, &s, &st, err, sizeof err)) die(std::string("rank ") + std::to_string(rank) + ": load: " + err);
        for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
        dev = sink.dev;
        fprintf(stderr, "# rank %d: checkpoint -> device image %.2f GB in %.1f s (%u threads)\n", rank, st.image_bytes / 1e9, st.seconds, st.threads);
    }
    const g6::Entry* find(const std::string& name) const { return pack.find(name); }
    const g6::Entry& entry(const std::string& name) const
    {
        const g6::Entry* e = pack.find(name);
        if (!e) die("rank " + std::to_string(rank) + ": the image has no " + name);
        return *e;
    }
    // the tensor on the device
    template <class T> const T* get(const std::string& name)
    {
        const g6::Entry& e = entry(name);
        if (dev) return reinterpret_cast<const T*>(dev + e.offset);
        for (const auto& kv : loaded)
            if (kv.first == name) return static_cast<const T*>(kv.second);
        std::vector<uint8_t> host(e.nbytes);
        char err[1024] = "";
        if (g6l_plan_read(plan, e.offset, e.nbytes, host.data(), err, sizeof err)) die("rank " + std::to_string(rank) + ": read of " + name + ": " + err);
        void* d = nullptr;
        INK2RT_CK(cudaMalloc(&d, e.nbytes ? e.nbytes : 1));
        INK2RT_CK(cudaMemcpy(d, host.data(), e.nbytes, cudaMemcpyHostToDevice));
        loaded.emplace_back(name, d);
        return static_cast<const T*>(d);
    }
    template <class T> const T* opt(const std::string& name) { return pack.find(name) ? get<T>(name) : nullptr; }
    int64_t dim(const std::string& name, size_t i) const
    {
        const g6::Entry& e = entry(name);
        if (i >= e.shape.size()) die(name + ": shape");
        return e.shape[i];
    }
};

// ------------------------------------------------------------------------------------------------ kernels of the sums
struct PickArgs
{
    const float* yp;   // [tile][8][H] of one rank
    ink2::Plan plan;
    int M, H, TK;
    float* out;        // [M][TK][H]
};
// The rank's expert outputs in the order of the tokens' picks.
__global__ void __launch_bounds__(256) k_ink2rt_picks(PickArgs a)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(a.M) * a.TK * a.H) return;
    const int n = static_cast<int>(i % a.H), mk = static_cast<int>(i / a.H);
    a.out[i] = a.yp[static_cast<size_t>(a.plan.tok_slot[mk] * 8 + a.plan.tok_col[mk]) * a.H + n];
}
// Per pick the ranks in order, then the picks in order.
__global__ void __launch_bounds__(256) k_ink2rt_pick_sum(const float* parts, int R, size_t stride, int M, int H, int TK, uint16_t* out)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<size_t>(M) * H) return;
    const size_t m = i / H, n = i % H;
    float s = 0.0f;
    for (int k = 0; k < TK; ++k)
    {
        const size_t o = (m * TK + k) * H + n;
        float y = parts[o];
        for (int r = 1; r < R; ++r) y = ink2::add_rn(y, parts[static_cast<size_t>(r) * stride + o]);
        s = ink2::add_rn(s, y);
    }
    out[i] = ink2::bf16(s);
}

// ------------------------------------------------------------------------------------------------ the head
struct Cand
{
    float logit;
    int id;
};
inline bool cand_before(const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); }
struct RowSample
{
    float inv_temp = 0, top_p = 0;   // inv_temp 0 = greedy
    uint32_t seed = 0, spos = 0;
};
struct RowOut
{
    int tok = 0;
    float lse = 0;            // log of the sum of exp over the valid vocabulary
    std::vector<Cand> top;    // the largest logits in order, at most NC
};
// A number in [0, 1) from the request's seed and the position of the sampled token.
inline double sample_u(uint32_t seed, uint32_t spos)
{
    uint64_t z = (static_cast<uint64_t>(seed) << 32 | spos) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}
// The token of a row from its merged candidates: the first one (greedy), or one drawn at the temperature from the
// candidates that hold top_p of their mass.
inline int sample_row(const std::vector<Cand>& top, const RowSample& s)
{
    if (top.empty()) return 0;
    if (!(s.inv_temp > 0.f) || top.size() == 1) return top[0].id;
    std::vector<double> p(top.size());
    double total = 0;
    for (size_t i = 0; i < top.size(); ++i) total += p[i] = std::exp(static_cast<double>(top[i].logit - top[0].logit) * s.inv_temp);
    size_t n = top.size();
    if (s.top_p > 0.f && s.top_p < 1.f)
    {
        double cum = 0;
        for (size_t i = 0; i < top.size(); ++i)
        {
            cum += p[i];
            if (cum >= static_cast<double>(s.top_p) * total) { n = i + 1; break; }
        }
        total = 0;
        for (size_t i = 0; i < n; ++i) total += p[i];
    }
    const double u = sample_u(s.seed, s.spos) * total;
    double cum = 0;
    for (size_t i = 0; i < n; ++i)
    {
        cum += p[i];
        if (u < cum) return top[i].id;
    }
    return top[n - 1].id;
}
// One rank's share of a row's head, as it crosses the fabric: [max f32][count i32][sum f64][NC x Cand].
inline size_t head_row_bytes(int NC) { return 16 + static_cast<size_t>(NC) * sizeof(Cand); }
inline void head_pack(const float* logits, int rows, int v0, int NC, uint8_t* out)
{
    std::vector<Cand> c;
    c.reserve(static_cast<size_t>(rows));
    float mx = -INFINITY;
    for (int j = 0; j < rows; ++j)
        if (std::isfinite(logits[j]))
        {
            c.push_back({logits[j], v0 + j});
            if (logits[j] > mx) mx = logits[j];
        }
    double sum = 0;
    for (const Cand& x : c) sum += std::exp(static_cast<double>(x.logit - mx));
    const size_t n = std::min(c.size(), static_cast<size_t>(NC));
    std::partial_sort(c.begin(), c.begin() + static_cast<long>(n), c.end(), cand_before);
    const int count = static_cast<int>(n);
    memset(out, 0, head_row_bytes(NC));
    memcpy(out, &mx, 4);
    memcpy(out + 4, &count, 4);
    memcpy(out + 8, &sum, 8);
    if (n) memcpy(out + 16, c.data(), n * sizeof(Cand));
}
// The ranks' shares of one row, in rank order -> the row's candidates and its log-sum-exp.
inline RowOut head_merge(const uint8_t* parts, size_t stride, int R, int NC)
{
    RowOut o;
    float gmx = -INFINITY;
    for (int r = 0; r < R; ++r)
    {
        float mx;
        memcpy(&mx, parts + static_cast<size_t>(r) * stride, 4);
        if (mx > gmx) gmx = mx;
    }
    double total = 0;
    for (int r = 0; r < R; ++r)
    {
        const uint8_t* p = parts + static_cast<size_t>(r) * stride;
        float mx;
        int count;
        double sum;
        memcpy(&mx, p, 4);
        memcpy(&count, p + 4, 4);
        memcpy(&sum, p + 8, 8);
        if (count < 0 || count > NC) die("head: a rank sent a bad candidate count");
        if (count) total += sum * std::exp(static_cast<double>(mx - gmx));
        const size_t at = o.top.size();
        o.top.resize(at + static_cast<size_t>(count));
        if (count) memcpy(o.top.data() + at, p + 16, static_cast<size_t>(count) * sizeof(Cand));
    }
    std::stable_sort(o.top.begin(), o.top.end(), cand_before);
    if (o.top.size() > static_cast<size_t>(NC)) o.top.resize(static_cast<size_t>(NC));
    o.lse = total > 0 ? gmx + static_cast<float>(std::log(total)) : 0.f;
    return o;
}

// ------------------------------------------------------------------------------------------------ the engine
struct RankPart
{
    ink2::AttnGeom g;
    ink2::AttnWeights aw;
    ink2::AttnState as;
    ink2::AttnWork wk;
    uint16_t* staging = nullptr;   // [slots][2][64][nk] on global layers, for partial-page KV state snapshots
    const uint8_t *fc1 = nullptr, *fc2 = nullptr;
    const float *s1 = nullptr, *s2 = nullptr;
};
struct Layer
{
    int l = 0;
    bool glob = false, mx8 = false;
    const float *attn_query = nullptr, *mlp_query = nullptr, *router_bias = nullptr;
    const uint16_t *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr, *attn_sconv = nullptr, *mlp_sconv = nullptr, *mlp_norm = nullptr, *router_w = nullptr;
    uint16_t *hist_a = nullptr, *hist_m = nullptr;   // [slots][3][H]
    uint16_t *att = nullptr, *mo = nullptr;          // [M][H]: the inputs of attn_sconv and mlp_sconv, kept until the commit
    std::vector<RankPart> rk;                        // the ranks this engine runs
};

struct Options
{
    int nr = 1;          // ranks of the model
    int rank0 = 0;       // the first rank this engine runs
    int nloc = 1;        // how many it runs: nr (the simulator) or 1
    int slots = 1;       // sessions
    int max_rows = 8;    // rows of a step
    int cap = 4096;      // positions of a session
    int nc = 256;        // candidates of a row's head
    std::string layers = "all";
    bool whole_image = false;
};

class Engine
{
public:
    Cfg c;
    Options o;
    std::vector<int> layer_ids;
    // Exchange with the other ranks (ink2_m4.cu). coll: the collective of the step, 2 per layer then the head.
    // parts: [nr][bytes] with this rank's share at index rank0; on return every rank's share is in place.
    std::function<void(int coll, float* dev_parts, size_t bytes)> gather;
    std::function<void(int coll, uint8_t* host_parts, size_t bytes)> gather_head;

    int collectives() const { return 2 * static_cast<int>(L.size()) + 1; }
    size_t attn_bytes(int M) const { return static_cast<size_t>(M) * c.H * 4; }
    size_t moe_bytes(int M) const { return static_cast<size_t>(M) * c.TK * c.H * 4; }
    size_t head_bytes(int M) const { return static_cast<size_t>(M) * head_row_bytes(o.nc); }
    // the largest share of one collective, for the fabric's slots
    size_t max_bytes() const { return std::max(moe_bytes(o.max_rows), head_bytes(o.max_rows)); }

    void init(const std::string& dir, const Options& opt)
    {
        o = opt;
        c = read_config(dir);
        if (c.heads % o.nr || c.I % o.nr || o.nloc < 1 || o.rank0 + o.nloc > o.nr) die("TP " + std::to_string(o.nr) + " does not divide the heads or the expert columns");
        layer_ids = parse_layers(o.layers, c.n_layers);
        for (int i = 0; i < o.nloc; ++i)
        {
            img.push_back(std::make_unique<Image>());
            img[i]->open(dir, o.nr, o.rank0 + i);
            if (o.whole_image) img[i]->whole();
        }
        Image& a = *img[0];
        if (a.dim("G.embed_norm", 0) != c.H) die("G.embed_norm does not have hidden_size cells");
        const int l0 = layer_ids[0];
        c.rel_dim = static_cast<int>(a.dim(lname(l0, "rel_l2_clip"), 1));
        c.rel_win = static_cast<int>(a.dim(lname(l0, "rel_logits_proj"), 1));
        vr = static_cast<int>(a.dim("G.lm_head", 0));
        max_pps = (o.cap + ink2::KV_PAGE_TOKENS - 1) / ink2::KV_PAGE_TOKENS;
        const size_t mh = static_cast<size_t>(o.max_rows) * c.H;
        embed = a.get<uint16_t>("G.embed");
        embed_norm = a.get<uint16_t>("G.embed_norm");
        final_norm = a.get<uint16_t>("G.final_norm");
        output_query = a.get<float>("G.attn_res.output_query");
        L.resize(layer_ids.size());
        for (size_t j = 0; j < L.size(); ++j) bind_layer(L[j], layer_ids[j]);
        mw.resize(static_cast<size_t>(o.nloc));
        for (int i = 0; i < o.nloc; ++i)
        {
            mw[i].alloc(o.max_rows, c.H, c.I / o.nr, c.NE, c.TK, false);
            lm.push_back(img[i]->get<uint16_t>("G.lm_head"));
            logits.push_back(dalloc<float>(static_cast<size_t>(o.max_rows) * vr));
        }
        d_slot = dalloc<int>(o.max_rows);
        d_pos = dalloc<int>(o.max_rows);
        d_run0 = dalloc<int>(o.max_rows);
        toks = dalloc<int>(o.max_rows);
        acc_dev = dalloc<uint8_t>(o.max_rows);
        for (uint16_t** p : {&h, &qi, &ki, &vi, &a_, &mid, &xm, &ms, &hf, &scratch}) *p = dalloc<uint16_t>(mh);
        const int nb = (static_cast<int>(L.size()) + c.block - 1) / c.block;
        if (nb + 1 > ink2::MAX_DEPTH) die("more blocks of layers than the depth attention reads");
        for (int b = 0; b < nb; ++b) states.push_back(dalloc<uint16_t>(mh));
        part_stride = static_cast<size_t>(o.max_rows) * c.TK * c.H;
        parts = dalloc<float>(static_cast<size_t>(o.nr) * part_stride);
        host_logits.resize(static_cast<size_t>(o.max_rows) * vr);
        head_parts.resize(static_cast<size_t>(o.nr) * head_bytes(o.max_rows));
        INK2RT_CK(cudaDeviceSynchronize());
        fprintf(stderr, "# ink2: ranks %d..%d of %d, %zu layers, H %d, %d experts top %d, vocab %d (%d rows a rank), %d slots, %d rows, cap %d\n", o.rank0,
                o.rank0 + o.nloc - 1, o.nr, L.size(), c.H, c.NE, c.TK, c.vocab, vr, o.slots, o.max_rows, o.cap);
    }

    // A new run on a slot: its convolution histories and rings read as zeros.
    void reset_slot(int slot)
    {
        if (slot < 0 || slot >= o.slots) die("reset_slot: slot");
        const size_t s = static_cast<size_t>(slot);
        for (Layer& y : L)
        {
            INK2RT_CK(cudaMemset(y.hist_a + s * 3 * c.H, 0, static_cast<size_t>(3) * c.H * 2));
            INK2RT_CK(cudaMemset(y.hist_m + s * 3 * c.H, 0, static_cast<size_t>(3) * c.H * 2));
            for (RankPart& x : y.rk)
            {
                const size_t nk = static_cast<size_t>(x.g.nh_kv) * x.g.hd;
                INK2RT_CK(cudaMemset(x.as.hist_k + s * 3 * nk, 0, 3 * nk * 2));
                INK2RT_CK(cudaMemset(x.as.hist_v + s * 3 * nk, 0, 3 * nk * 2));
                if (!x.g.global) INK2RT_CK(cudaMemset(x.as.ring + s * 2 * x.g.window * nk, 0, static_cast<size_t>(2) * x.g.window * nk * 2));
                else INK2RT_CK(cudaMemset(x.staging + s * 2 * ink2::KV_PAGE_TOKENS * nk, 0, static_cast<size_t>(2) * ink2::KV_PAGE_TOKENS * nk * 2));
            }
        }
    }

    // One step over the rows of rh. No state changes before commit(). When samples is null (prefill), the head is skipped.
    void forward(const ink2::RowsHost& rh, const int* tk, const RowSample* samples, std::vector<RowOut>& out)
    {
        M = rh.rows();
        if (M < 1 || M > o.max_rows) die("step of " + std::to_string(M) + " rows");
        if (rh.max_pos >= o.cap) die("position " + std::to_string(rh.max_pos) + " is past the cap " + std::to_string(o.cap));
        const size_t n = static_cast<size_t>(M) * c.H;
        const int T = o.nr;
        INK2RT_CK(cudaMemcpy(d_slot, rh.slot.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
        INK2RT_CK(cudaMemcpy(d_pos, rh.pos.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
        INK2RT_CK(cudaMemcpy(d_run0, rh.run0.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
        INK2RT_CK(cudaMemcpy(toks, tk, sizeof(int) * M, cudaMemcpyHostToDevice));
        const ink2::Rows rw{d_slot, d_pos, d_run0};
        ink2::embed_norm(toks, embed, embed_norm, nullptr, c.image_token, h, M, c.H, 0);
        for (size_t j = 0; j < L.size(); ++j)
        {
            Layer& s = L[j];
            const int ji = static_cast<int>(j);
            const int done = (ji + c.block - 1) / c.block;   // finished blocks before the attention of this layer
            if (!s.attn_query) ink2::rmsnorm(h, s.attn_norm, qi, M, c.H, 0);
            else
            {
                ink2::AttnResArgs ar{};
                for (int d = 0; d < done; ++d) ar.rows[d] = states[d];
                ar.rows[done] = h;
                ar.query = s.attn_query;
                ar.D = done + 1;
                ar.Q = s.glob ? 3 : 1;
                ar.H = c.H;
                ar.norm[0] = s.glob ? s.q_norm : s.attn_norm;
                ar.norm[1] = s.k_norm;
                ar.norm[2] = s.v_norm;
                ar.out[0] = qi;
                ar.out[1] = ki;
                ar.out[2] = vi;
                ink2::attn_res(ar, M, 0);
            }
            for (int i = 0; i < o.nloc; ++i)
            {
                RankPart& x = s.rk[i];
                ink2::attn_forward(x.g, x.aw, x.as, x.wk, rw, rh.max_pos, qi, s.glob ? ki : qi, s.glob ? vi : qi, M, T == 1 ? s.att : nullptr,
                                   parts + static_cast<size_t>(o.rank0 + i) * n, 0);
            }
            if (T > 1)
            {
                if (gather) gather(2 * ji, parts, n * 4);
                ink2::sum_ranks(parts, T, n, s.att, 0);
            }
            ink2::sconv_static(s.att, s.hist_a, rw, s.attn_sconv, M, c.H, a_, 0);
            ink2::attn_block_step(h, a_, ji % c.block == 0, states[ji / c.block], mid, n, 0);
            ink2::AttnResArgs ar{};
            const int d2 = ji / c.block + 1;
            for (int d = 0; d < d2; ++d) ar.rows[d] = states[d];
            ar.rows[d2] = mid;
            ar.query = s.mlp_query;
            ar.D = d2 + 1;
            ar.Q = 1;
            ar.H = c.H;
            ar.norm[0] = s.mlp_norm;
            ar.out[0] = xm;
            ink2::attn_res(ar, M, 0);
            if (s.mx8) moe<ink2::Mx8Frag>(s, ji);
            else moe<nv4::Frag>(s, ji);
            ink2::sconv_static(s.mo, s.hist_m, rw, s.mlp_sconv, M, c.H, ms, 0);
            ink2::add_bf16(mid, ms, h, n, 0);
        }
        if (!samples)
        {
            out.clear();
            return;
        }
        ink2::AttnResArgs ar{};
        const int nb = static_cast<int>(states.size());
        for (int d = 0; d < nb; ++d) ar.rows[d] = states[d];
        ar.rows[nb] = h;
        ar.query = output_query;
        ar.D = nb + 1;
        ar.Q = 1;
        ar.H = c.H;
        ar.norm[0] = final_norm;
        ar.out[0] = hf;
        ink2::attn_res(ar, M, 0);
        // the head: every rank's rows of lm_head, its candidates on the host, then the merge in rank order
        const size_t hb = head_bytes(M), rb = head_row_bytes(o.nc);
        for (int i = 0; i < o.nloc; ++i)
        {
            const int r = o.rank0 + i;
            ink2::lm_head(hf, lm[i], M, c.H, vr, r * vr, c.valid, logits[i], 0);
            INK2RT_CK(cudaMemcpy(host_logits.data(), logits[i], sizeof(float) * static_cast<size_t>(M) * vr, cudaMemcpyDeviceToHost));
            for (int m = 0; m < M; ++m) head_pack(&host_logits[static_cast<size_t>(m) * vr], vr, r * vr, o.nc, &head_parts[static_cast<size_t>(r) * hb + static_cast<size_t>(m) * rb]);
        }
        if (T > 1 && gather_head) gather_head(collectives() - 1, head_parts.data(), hb);
        out.resize(static_cast<size_t>(M));
        std::vector<uint8_t> row(static_cast<size_t>(T) * rb);
        for (int m = 0; m < M; ++m)
        {
            for (int r = 0; r < T; ++r) memcpy(&row[static_cast<size_t>(r) * rb], &head_parts[static_cast<size_t>(r) * hb + static_cast<size_t>(m) * rb], rb);
            out[m] = head_merge(row.data(), rb, T, o.nc);
            out[m].tok = sample_row(out[m].top, samples ? samples[m] : RowSample{});
        }
    }

    // Keep the rows of the last forward: all of them, or the ones acc marks.
    void commit(const std::vector<uint8_t>* acc = nullptr)
    {
        const ink2::Rows rw{d_slot, d_pos, d_run0};
        if (acc) INK2RT_CK(cudaMemcpy(acc_dev, acc->data(), static_cast<size_t>(M), cudaMemcpyHostToDevice));
        const uint8_t* am = acc ? acc_dev : nullptr;
        for (Layer& s : L)
        {
            ink2::sconv_advance(s.att, s.hist_a, rw, am, M, c.H, 0);
            ink2::sconv_advance(s.mo, s.hist_m, rw, am, M, c.H, 0);
            for (RankPart& x : s.rk) ink2::attn_commit(x.g, x.as, x.wk, rw, am, M, 0);
        }
        INK2RT_CK(cudaDeviceSynchronize());
    }

    // The state a KvStateDesc names (s2/kv_state_desc.h ink2_kv_state): bytes of one page row and of a slot's tail.
    int rank_nk(int i = 0) const { return L.empty() ? 0 : L[0].rk[static_cast<size_t>(i)].g.nh_kv * L[0].rk[static_cast<size_t>(i)].g.hd; }
    int max_rank_nk() const
    {
        int m = 0;
        for (int r = 0; r < o.nr; ++r) m = std::max(m, rank_geom(r, false).nh_kv * c.hd);
        return m;
    }
    int max_pages_per_slot() const { return max_pps; }

    struct KvPtrs
    {
        bool global = false;
        uint8_t *pages = nullptr, *staging = nullptr, *ring = nullptr;
        uint8_t *hist_k = nullptr, *hist_v = nullptr, *hist_a = nullptr, *hist_m = nullptr;
    };
    std::vector<KvPtrs> kv_ptrs(int i = 0) const
    {
        std::vector<KvPtrs> v;
        v.reserve(L.size());
        for (const Layer& y : L)
        {
            const RankPart& x = y.rk[static_cast<size_t>(i)];
            KvPtrs k;
            k.global = y.glob;
            k.pages = reinterpret_cast<uint8_t*>(x.as.pages);
            k.staging = reinterpret_cast<uint8_t*>(x.staging);
            k.ring = reinterpret_cast<uint8_t*>(x.as.ring);
            k.hist_k = reinterpret_cast<uint8_t*>(x.as.hist_k);
            k.hist_v = reinterpret_cast<uint8_t*>(x.as.hist_v);
            k.hist_a = reinterpret_cast<uint8_t*>(y.hist_a);
            k.hist_m = reinterpret_cast<uint8_t*>(y.hist_m);
            v.push_back(k);
        }
        return v;
    }

    // Copies the unfinished page at `pos` of each full-attention layer to `staging` (before capture) or back (after restore).
    void stage_partial_page(int slot, uint32_t pos, bool to_staging, int i = 0)
    {
        if (slot < 0 || slot >= o.slots) die("stage_partial_page: slot");
        const uint32_t rem = pos % static_cast<uint32_t>(ink2::KV_PAGE_TOKENS);
        const uint32_t p = pos / static_cast<uint32_t>(ink2::KV_PAGE_TOKENS);
        for (Layer& y : L)
        {
            if (!y.glob) continue;
            RankPart& x = y.rk[static_cast<size_t>(i)];
            const size_t nk = static_cast<size_t>(x.g.nh_kv) * x.g.hd;
            const size_t elems = static_cast<size_t>(2) * ink2::KV_PAGE_TOKENS * nk;
            uint16_t* stg = x.staging + static_cast<size_t>(slot) * elems;
            if (rem > 0 && p < static_cast<uint32_t>(max_pps))
            {
                uint16_t* pg = x.as.pages + (static_cast<size_t>(slot) * max_pps + p) * elems;
                if (to_staging) INK2RT_CK(cudaMemcpy(stg, pg, elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
                else INK2RT_CK(cudaMemcpy(pg, stg, elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
            }
            else if (to_staging) INK2RT_CK(cudaMemset(stg, 0, elems * sizeof(uint16_t)));
        }
        INK2RT_CK(cudaDeviceSynchronize());
    }

private:
    static std::string lname(int l, const char* what)
    {
        char b[96];
        snprintf(b, sizeof b, "L%03d.%s", l, what);
        return b;
    }
    ink2::AttnGeom rank_geom(int r, bool glob) const
    {
        ink2::AttnGeom g;
        g.H = c.H;
        g.hd = c.hd;
        g.rel_dim = c.rel_dim;
        g.rel_win = c.rel_win;
        g.gqa = c.heads / c.kv_heads;
        g.nh_q = c.heads / o.nr;
        g.q0 = r * g.nh_q;
        g.kv0 = g.q0 / g.gqa;
        g.nh_kv = (g.q0 + g.nh_q - 1) / g.gqa + 1 - g.kv0;
        g.window = c.window;
        g.log_scale_base = c.log_base;
        g.global = glob;
        return g;
    }
    void bind_layer(Layer& s, int l)
    {
        Image& a = *img[0];
        const auto N = [l](const char* w) { return lname(l, w); };
        s.l = l;
        s.glob = a.find(N("q_input_norm")) != nullptr;
        s.mx8 = a.entry(N("x.fc1")).expert_format == 6;
        if (!s.mx8 && a.entry(N("x.fc1")).expert_format != 5) die(N("x.fc1") + ": the experts are neither nv4g nor mxfp8");
        s.attn_query = a.opt<float>(N("attn_res.attn_query"));
        s.mlp_query = a.get<float>(N("attn_res.mlp_query"));
        s.router_bias = a.get<float>(N("router.bias"));
        if (s.glob)
        {
            s.q_norm = a.get<uint16_t>(N("q_input_norm"));
            s.k_norm = a.get<uint16_t>(N("k_input_norm"));
            s.v_norm = a.get<uint16_t>(N("v_input_norm"));
            if (!s.attn_query) die(N("attn_res.attn_query") + ": a full-attention layer needs it");
        }
        else s.attn_norm = a.get<uint16_t>(N("attn_norm"));
        s.attn_sconv = a.get<uint16_t>(N("attn_sconv"));
        s.mlp_sconv = a.get<uint16_t>(N("mlp_sconv"));
        s.mlp_norm = a.get<uint16_t>(N("mlp_norm"));
        s.router_w = a.get<uint16_t>(N("router.w"));
        if (a.dim(N("router.w"), 0) != c.NE || a.dim(N("x.fc1"), 1) * 16 != c.I / o.nr) die(N("x.fc1") + ": experts or columns differ from config.json");
        const size_t mh = static_cast<size_t>(o.max_rows) * c.H, sl = static_cast<size_t>(o.slots);
        s.hist_a = dalloc<uint16_t>(sl * 3 * c.H);
        s.hist_m = dalloc<uint16_t>(sl * 3 * c.H);
        s.att = dalloc<uint16_t>(mh);
        s.mo = dalloc<uint16_t>(mh);
        s.rk.resize(static_cast<size_t>(o.nloc));
        for (int i = 0; i < o.nloc; ++i)
        {
            Image& im = *img[i];
            RankPart& x = s.rk[i];
            x.g = rank_geom(o.rank0 + i, s.glob);
            const size_t nk = static_cast<size_t>(x.g.nh_kv) * x.g.hd;
            if (im.dim(N("wq"), 0) != static_cast<int64_t>(x.g.nh_q) * x.g.hd || im.dim(N("wk"), 0) != static_cast<int64_t>(nk)) die(N("wq") + ": heads of the rank differ from config.json");
            x.aw.wq = im.get<uint16_t>(N("wq"));
            x.aw.wr = im.get<uint16_t>(N("wr"));
            x.aw.gate = im.get<uint16_t>(N("gate_attn"));
            x.aw.sink = im.get<uint16_t>(N("attn_sink_proj"));
            x.aw.wk = im.get<uint16_t>(N("wk"));
            x.aw.wv = im.get<uint16_t>(N("wv"));
            x.aw.wo = im.get<uint16_t>(N("wo"));
            x.aw.q_clip = im.get<uint16_t>(N("q_l2_clip"));
            x.aw.rel_clip = im.get<uint16_t>(N("rel_l2_clip"));
            x.aw.rel_proj = im.get<uint16_t>(N("rel_logits_proj"));
            if (s.glob)
            {
                x.aw.k_static = im.get<uint16_t>(N("k_sconv.static_w"));
                x.aw.v_static = im.get<uint16_t>(N("v_sconv.static_w"));
                x.aw.k_wproj = im.get<uint16_t>(N("k_sconv.weight_proj"));
                x.aw.v_wproj = im.get<uint16_t>(N("v_sconv.weight_proj"));
            }
            else
            {
                x.aw.k_sconv = im.get<uint16_t>(N("k_sconv"));
                x.aw.v_sconv = im.get<uint16_t>(N("v_sconv"));
            }
            x.as.hist_k = dalloc<uint16_t>(sl * 3 * nk);
            x.as.hist_v = dalloc<uint16_t>(sl * 3 * nk);
            int keys = x.g.window;
            if (!s.glob) x.as.ring = dalloc<uint16_t>(sl * 2 * x.g.window * nk);
            else
            {
                // every slot owns max_pps pages, in order
                const int total = o.slots * max_pps;
                std::vector<int> table(static_cast<size_t>(total));
                for (int p = 0; p < total; ++p) table[p] = p;
                int* t = dalloc<int>(static_cast<size_t>(total));
                INK2RT_CK(cudaMemcpy(t, table.data(), sizeof(int) * total, cudaMemcpyHostToDevice));
                x.as.pages = dalloc<uint16_t>(static_cast<size_t>(total) * 2 * ink2::KV_PAGE_TOKENS * nk);
                x.staging = dalloc<uint16_t>(sl * 2 * ink2::KV_PAGE_TOKENS * nk);
                x.as.page_table = t;
                x.as.max_pps = max_pps;
                keys = o.cap;
            }
            // the split partials of one row over every span of keys; the forward takes as many rows a launch as fit
            const size_t spans = static_cast<size_t>(keys + ink2::ATTN_SPAN - 1) / ink2::ATTN_SPAN + 1;
            const size_t rows = s.glob ? 1 : static_cast<size_t>(o.max_rows);
            x.wk.alloc(x.g, o.max_rows, rows * x.g.nh_q * spans * (x.g.hd + 2));
            x.fc1 = im.get<uint8_t>(N("x.fc1"));
            x.fc2 = im.get<uint8_t>(N("x.fc2"));
            if (!s.mx8)
            {
                x.s1 = im.get<float>(N("x.s1"));
                x.s2 = im.get<float>(N("x.s2"));
            }
        }
    }
    template <class F> void moe(Layer& s, int j)
    {
        ink2::Work& w0 = mw[0];
        const int T = o.nr;
        ink2::route(xm, s.router_w, s.router_bias, M, c.H, c.NE, c.TK, c.route_scale, w0.score, w0.idx, w0.wt, w0.plan, 0);
        const size_t picks = static_cast<size_t>(M) * c.TK * c.H;
        for (int i = 0; i < o.nloc; ++i)
        {
            const RankPart& x = s.rk[i];
            ink2::MoeArgs ma = mw[i].args(x.fc1, x.fc2, x.s1, x.s2, M, T == 1 ? s.mo : scratch);
            ma.plan = w0.plan;   // one routing for every rank
            if (M <= 8) ink2::moe_decode<F>(ma, xm, c.NE, 0);
            else ink2::moe_prefill<F>(ma, xm, c.NE, 0);
            if (T == 1) return;
            PickArgs pa{static_cast<const float*>(mw[i].ypart), w0.plan, M, c.H, c.TK, parts + static_cast<size_t>(o.rank0 + i) * picks};
            k_ink2rt_picks<<<ink2::cells_grid(picks), 256, 0, 0>>>(pa);
        }
        if (gather) gather(2 * j + 1, parts, picks * 4);
        k_ink2rt_pick_sum<<<ink2::cells_grid(static_cast<size_t>(M) * c.H), 256, 0, 0>>>(parts, T, picks, M, c.H, c.TK, s.mo);
    }

    std::vector<std::unique_ptr<Image>> img;
    std::vector<Layer> L;
    std::vector<ink2::Work> mw;
    std::vector<const uint16_t*> lm;
    std::vector<float*> logits;
    std::vector<uint16_t*> states;
    std::vector<float> host_logits;
    std::vector<uint8_t> head_parts;
    const uint16_t *embed = nullptr, *embed_norm = nullptr, *final_norm = nullptr;
    const float* output_query = nullptr;
    int *d_slot = nullptr, *d_pos = nullptr, *d_run0 = nullptr, *toks = nullptr;
    uint8_t* acc_dev = nullptr;
    uint16_t *h = nullptr, *qi = nullptr, *ki = nullptr, *vi = nullptr, *a_ = nullptr, *mid = nullptr, *xm = nullptr, *ms = nullptr, *hf = nullptr, *scratch = nullptr;
    float* parts = nullptr;
    size_t part_stride = 0;
    int M = 0, vr = 0, max_pps = 0;
};
}   // namespace ink2rt
