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
#include "../kernels/ink2_loop.cuh"
#include "../kernels/ink2_tr3_moe.cuh"
#include "../attn/ink2_attn.cuh"
#include "../kernels/ink2_mtp.cuh"
#include "../kernels/ink2_head.cuh"
#include "ink2_prof.cuh"
#include "../model_geom_ink2.h"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"
#include "../knobs-rs/gr8_knobs.h"
#include "ink2_vision.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
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
    // looped core (Inkling-2-Mini text_config.loop_config); loop_passes <= 1 = unlooped
    int loop_start = 0, loop_end = 0, loop_passes = 1, loop_gate_rank = 0;
    bool is_looped() const { return loop_passes > 1 && loop_end > loop_start; }
    int n_loop_layers() const { return is_looped() ? (loop_end - loop_start) : 0; }
    int n_virtual_layers() const { return is_looped() ? n_layers + (loop_passes - 1) * (loop_end - loop_start) : n_layers; }
    // the MTP drafter (mtp_config); mtp_layers 0 = the checkpoint has none
    int mtp_layers = 0, mtp_inter = 0, mtp_window = 0, mtp_block = 8, mtp_drafts = 7, markov_rank = 0;
    int mtp_H = 0, mtp_heads = 0, mtp_kv_heads = 0, mtp_hd = 0;
    float mtp_theta = 1e6f;
    std::vector<uint8_t> mtp_global;   // [mtp_layers] 1 = the layer reads the whole context
    std::vector<int> mtp_targets;      // the taps: trunk layers whose input is read; n_layers = the final_norm output
    // the vision tower (vision_config; the defaults are Inkling-2-Small's). has_vision: config.json has a vision_config
    bool has_vision = false;
    int vis_hidden = 1024, vis_layers = 23, vis_heads = 16, vis_mlp = 4096, vis_patch = 20, vis_merge = 2, vis_grid = 32, vis_proj = 4096;
    float vis_eps = 1e-5f, vis_theta = 10000.0f;
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

// The integers of the list after "key" in text; empty when the key is absent or is not a list.
inline std::vector<int> json_ints(const std::string& text, const char* key)
{
    std::vector<int> v;
    const std::string k = std::string("\"") + key + "\"";
    size_t p = text.find(k);
    if (p == std::string::npos) return v;
    p = text.find_first_not_of(" \t\r\n:", p + k.size());
    if (p == std::string::npos || text[p] != '[') return v;
    const size_t end = text.find(']', p);
    if (end == std::string::npos) return v;
    const char* s = text.c_str() + p + 1;
    const char* stop = text.c_str() + end;
    while (s < stop)
    {
        char* e = nullptr;
        const long x = strtol(s, &e, 10);
        if (e == s) ++s;
        else
        {
            v.push_back(static_cast<int>(x));
            s = e;
        }
    }
    return v;
}

// config.json of the checkpoint: its text_config when it has one, else the top level. The drafter's mtp_config and
// the tower's vision_config are read apart, so a key of the same name in them never stands in for the trunk's.
inline Cfg read_config(const std::string& dir)
{
    std::ifstream f(dir + "/config.json");
    if (!f) die(dir + "/config.json: cannot open");
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str(), mtp, vis;
    if (text.find("\"vision_config\"") != std::string::npos)
    {
        vis = json_section(text, "vision_config");
        const size_t at = text.find(vis, text.find("\"vision_config\""));
        if (vis.empty() || vis[0] != '{' || at == std::string::npos) die(dir + "/config.json: vision_config is not an object");
        text.replace(at, vis.size(), "{}");
    }
    if (text.find("\"mtp_config\"") != std::string::npos)
    {
        mtp = json_section(text, "mtp_config");
        const size_t at = text.find(mtp, text.find("\"mtp_config\""));
        if (mtp.empty() || mtp[0] != '{' || at == std::string::npos) die(dir + "/config.json: mtp_config is not an object");
        text.replace(at, mtp.size(), "{}");
    }
    const std::string sc = json_section(text, "text_config");
    const auto opt_int = [&](const char* key, int def) {
        return static_cast<int>(json_num(sc, 0, key, def));
    };
    const auto need = [&](const char* key) {
        const int v = opt_int(key, -1);
        if (v < 1) die(dir + "/config.json: no " + key);
        return v;
    };
    Cfg c;
    c.image_token = static_cast<int>(json_num(text, 0, "image_token_id", geom::TOK_IMAGE));
    if (!vis.empty())
    {
        const auto vi = [&](const char* key, int def) { return static_cast<int>(json_num(vis, 0, key, def)); };
        c.has_vision = true;
        c.vis_hidden = vi("hidden_size", c.vis_hidden);
        c.vis_layers = vi("num_hidden_layers", c.vis_layers);
        c.vis_heads = vi("num_attention_heads", c.vis_heads);
        c.vis_mlp = static_cast<int>(std::lround(c.vis_hidden * json_num(vis, 0, "mlp_ratio", 4.0)));
        c.vis_patch = vi("native_patch_size", c.vis_patch);
        c.vis_merge = vi("adapter_merge_size", c.vis_merge);
        c.vis_grid = vi("posemb_grid_size", c.vis_grid);
        c.vis_proj = vi("projection_hidden_dim", c.vis_proj);
        c.vis_eps = static_cast<float>(json_num(vis, 0, "layer_norm_eps", c.vis_eps));
        c.vis_theta = static_cast<float>(json_num(vis, 0, "rope_theta", c.vis_theta));
    }
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
    if (sc.find("\"loop_config\"") != std::string::npos)
    {
        const std::string lc = json_section(sc, "loop_config");
        c.loop_start = static_cast<int>(json_num(lc, 0, "core_start", 0));
        c.loop_end = static_cast<int>(json_num(lc, 0, "core_end", 0));
        c.loop_passes = static_cast<int>(json_num(lc, 0, "num_passes", 1));
        c.loop_gate_rank = static_cast<int>(json_num(lc, 0, "gate_rank", 0));
        if (c.loop_passes > 1)
        {
            if (c.loop_start < c.block || c.loop_end > c.n_layers || c.loop_start >= c.loop_end ||
                c.loop_start % c.block || c.loop_end % c.block || c.loop_gate_rank < 1 || c.loop_gate_rank > ink2::LOOP_MAX_RANK)
                die(dir + "/config.json: invalid loop_config");
        }
    }
    if (c.heads % c.kv_heads || c.H % 32 || c.hd % 16 || c.hd > ink2::ATTN_MAX_HD) die(dir + "/config.json: heads, hidden_size or head_dim");
    if (!mtp.empty())
    {
        const auto mi = [&](const char* key, int def) { return static_cast<int>(json_num(mtp, 0, key, def)); };
        c.mtp_layers = mi("num_hidden_layers", 0);
        c.mtp_inter = mi("intermediate_size", 0);
        c.mtp_window = mi("sliding_window_size", mi("sliding_window", c.window));
        c.mtp_block = mi("block_size", 8);
        c.mtp_drafts = mi("num_speculative_tokens", c.mtp_block - 1);
        c.markov_rank = mi("markov_rank", 0);
        c.mtp_H = mi("hidden_size", c.H);
        c.mtp_heads = mi("num_attention_heads", c.heads);
        c.mtp_kv_heads = mi("num_key_value_heads", c.kv_heads);
        c.mtp_hd = mi("head_dim", c.hd);
        c.mtp_theta = static_cast<float>(json_num(mtp, 0, "rope_theta", 1e6));
        c.mtp_targets = json_ints(mtp, "target_layer_ids");
        const std::vector<int> local = json_ints(mtp, "local_layer_ids");
        c.mtp_global.assign(static_cast<size_t>(std::max(c.mtp_layers, 0)), 1);
        for (int l : local)
            if (l >= 0 && l < c.mtp_layers) c.mtp_global[static_cast<size_t>(l)] = 0;
    }
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

    void open(const std::string& dir, int nr, int r, const std::string& variant = "")
    {
        rank = r;
        const std::string sfx = ".manifest.json";
        std::string p = knob::tmpdir() + "/ink2-rank" + std::to_string(r) + ".XXXXXX" + sfx;
        const int fd = mkstemps(&p[0], static_cast<int>(sfx.size()));
        if (fd < 0) die(p + ": " + strerror(errno));
        close(fd);
        char err[1024] = "";
        plan = g6l_ink2_plan_ex(dir.c_str(), static_cast<uint32_t>(nr), static_cast<uint32_t>(r),
                                variant.empty() ? nullptr : variant.c_str(), p.c_str(), err, sizeof err);
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
    // exlr8-v2 trellis experts (exp_fmt in {1, 2, 3})
    const uint8_t *tr3_fc1 = nullptr, *tr3_fc2 = nullptr;
    const __half *tr3_v1 = nullptr, *tr3_v2 = nullptr, *tr3_v3 = nullptr;
    const long long *tr3_tab = nullptr, *tr3_kmix = nullptr;
    int tr3_eb = 0;
    int tr3_nblk = 0;
    const __half *tr3_sg = nullptr, *tr3_v1b = nullptr, *tr3_v3b = nullptr;
    int tr3_wide = 0;
};
struct CtrlWeights
{
    const uint16_t *down = nullptr, *forget = nullptr, *gain = nullptr;
};
struct Layer
{
    int l = 0;           // physical weight layer index
    int v = 0;           // virtual execution step index
    int loop_pass = 0;   // 0 for prelude / core pass 0 / coda; 1..loop_passes-1 for recurrent passes
    int loop_j = -1;     // 0..n_loop-1 for core layers, else -1
    int exp_fmt = 5;     // 5 = nv4g, 6 = mxfp8, 1 = exlr8 mcg, 2 = exlr8 mul1, 3 = exlr8 pmcgx
    bool glob = false, mx8 = false;
    const float *attn_query = nullptr, *mlp_query = nullptr, *router_bias = nullptr;
    const uint16_t *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr, *attn_sconv = nullptr, *mlp_sconv = nullptr, *mlp_norm = nullptr, *router_w = nullptr;
    CtrlWeights ctrl_q, ctrl_k, ctrl_v, ctrl_mlp;
    uint16_t *hist_a = nullptr, *hist_m = nullptr;   // [slots][3][H]
    uint16_t *att = nullptr, *mo = nullptr;          // [M][H]: the inputs of attn_sconv and mlp_sconv, kept until the commit
    std::vector<RankPart> rk;                        // the ranks this engine runs
};

// The drafter: one rank's slices of a layer and its context rows, a layer, and one rank's work buffers.
struct MtpRank
{
    const uint16_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wkc = nullptr, *wvc = nullptr, *wo = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
    int8_t *wq_i8 = nullptr, *wk_i8 = nullptr, *wv_i8 = nullptr, *wo_i8 = nullptr, *gate_i8 = nullptr, *up_i8 = nullptr, *down_i8 = nullptr;
    float *wq_sc = nullptr, *wk_sc = nullptr, *wv_sc = nullptr, *wo_sc = nullptr, *gate_sc = nullptr, *up_sc = nullptr, *down_sc = nullptr;
    uint16_t* ring = nullptr;      // [slots][2][window][nk] on a local layer
    uint16_t* pages = nullptr;     // [slots * max_pps][2][64][nk] on a global layer
    uint16_t* staging = nullptr;   // [slots][2][64][nk] on a global layer
};
struct MtpLayer
{
    bool glob = false;
    const uint16_t *dflare_norm = nullptr, *input_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *k_norm_ctx = nullptr;
    std::vector<MtpRank> rk;
};
struct MtpWork
{
    uint16_t *q = nullptr, *og = nullptr, *kb = nullptr, *vb = nullptr, *gate = nullptr, *up = nullptr, *ck = nullptr, *cv = nullptr, *base = nullptr;
    uint16_t *fold_w = nullptr, *fold_b = nullptr;   // [vr][R], [vr]: lm_head . hidden_out
    int8_t *lm_i8 = nullptr, *fold_w_i8 = nullptr;   // drafter-only per-row symmetric INT8 copies (2x less HBM)
    float *lm_scale = nullptr, *fold_scale = nullptr;
    float *lg = nullptr, *part = nullptr;
    size_t part_floats = 0;
};

// The prefill GEMMs through cuBLAS (Options::cublas_prefill): ink2::gemm's hook while a BlasScope is alive.
inline cublasHandle_t& blas_cur()
{
    static thread_local cublasHandle_t hd = nullptr;
    return hd;
}
inline bool blas_gemm(const uint16_t* x, const uint16_t* w, int M, int K, int N, void* y, int mode, cudaStream_t st)
{
    static thread_local bool warned = false;
    cublasHandle_t hd = blas_cur();
    if (!hd || M < 1 || N < 16 || (N & 7) != 0 || K < 16 || (K & 15) != 0) return false;
    if (((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(y)) & 15u) != 0) return false;
    const float one = 1.f, zero = 0.f;
    cublasStatus_t rc = cublasSetStream(hd, st);
    if (rc == CUBLAS_STATUS_SUCCESS)
        rc = cublasGemmEx(hd, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &zero, y, mode == ink2::GEMM_F32 ? CUDA_R_32F : CUDA_R_16BF, N,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (rc == CUBLAS_STATUS_SUCCESS) return true;
    if (!warned) fprintf(stderr, "# ink2: cuBLAS GEMM failed (%d); the engine's own GEMM runs instead\n", static_cast<int>(rc));
    warned = true;
    return false;
}
struct BlasScope
{
    explicit BlasScope(cublasHandle_t hd)
    {
        if (!hd) return;
        blas_cur() = hd;
        ink2::gemm_hook() = blas_gemm;
    }
    ~BlasScope()
    {
        blas_cur() = nullptr;
        ink2::gemm_hook() = nullptr;
    }
    BlasScope(const BlasScope&) = delete;
    BlasScope& operator=(const BlasScope&) = delete;
};
// A prefill chunk: the rows are one run of one slot at consecutive positions.
inline bool one_run(const ink2::RowsHost& rh)
{
    const int M = rh.rows();
    for (int m = 0; m < M; ++m)
        if (rh.slot[static_cast<size_t>(m)] != rh.slot[0] || rh.run0[static_cast<size_t>(m)] != 0 || rh.pos[static_cast<size_t>(m)] != rh.pos[0] + m) return false;
    return M > 0;
}

struct Options
{
    int nr = 1;          // ranks of the model
    int rank0 = 0;       // the first rank this engine runs
    int nloc = 1;        // how many it runs: nr (the simulator) or 1
    int slots = 1;       // sessions
    int max_rows = 8;    // rows of a step
    int cap = 4096;      // positions of a session
    int nc = 256;        // candidates of a row's head
    int mtp = 0;         // drafts of a round (0 = no drafter: no taps, no drafter state)
    std::string layers = "all";
    std::string variant; // exlr8 width-manifest variant (empty = auto / GR8_INK2_VARIANT)
    bool whole_image = false;
    bool vision = false; // run the vision tower when the checkpoint has one (image input)
    // the speed paths: -1 = what the knob says (GR8_INK2_*), 0 = off, 1 = on
    int fast_allreduce = -1, gpu_head_pack = -1, cublas_prefill = -1, tc_prefill_attn = -1;
    int use_graph = -1;  // -1 = env INK2_GRAPH (default 1 on TP=1), 0 = off, 1 = on
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
    // Options::fast_allreduce (ink2_m4.cu sets both or neither): the collectives of one forward, or of one mtp_draft,
    // are one fabric step. step_begin gets the bytes of its collectives in order, before the first gather; step_end
    // runs after the last one. gather then starts the exchange on the stream and does not wait for it.
    std::function<void(const std::vector<uint32_t>& bytes)> step_begin;
    std::function<void()> step_end;
    bool fast_allreduce() const { return use_far; }
    int max_collectives() const { return std::max(collectives(), o.mtp > 0 ? 2 * static_cast<int>(ML.size()) + o.mtp : 0); }

    int collectives() const { return 2 * static_cast<int>(L.size()) + 1; }
    size_t attn_bytes(int M) const { return static_cast<size_t>(M) * c.H * 4; }
    size_t moe_bytes(int M) const { return static_cast<size_t>(M) * c.TK * c.H * 4; }
    size_t head_bytes(int M) const { return static_cast<size_t>(M) * head_row_bytes(o.nc); }
    // the largest share of one collective, for the fabric's slots
    size_t max_bytes() const
    {
        size_t b = std::max(moe_bytes(o.max_rows), head_bytes(o.max_rows));
        if (o.mtp > 0) b = std::max(b, std::max(static_cast<size_t>(o.slots) * c.mtp_block * c.H * 4, head_bytes(o.slots)));
        return b;
    }

    void init(const std::string& dir, const Options& opt)
    {
        o = opt;
        c = read_config(dir);
        if (c.heads % o.nr || o.nloc < 1 || o.rank0 + o.nloc > o.nr) die("TP " + std::to_string(o.nr) + " does not divide the heads");
        layer_ids = parse_layers(o.layers, c.n_layers);
        for (int i = 0; i < o.nloc; ++i)
        {
            img.push_back(std::make_unique<Image>());
            img[i]->open(dir, o.nr, o.rank0 + i, o.variant);
            if (o.whole_image) img[i]->whole();
        }
        Image& a = *img[0];
        if (a.dim("G.embed_norm", 0) != c.H) die("G.embed_norm does not have hidden_size cells");
        const int l0 = layer_ids[0];
        c.rel_dim = static_cast<int>(a.dim(lname(l0, "rel_l2_clip"), 1));
        c.rel_win = static_cast<int>(a.dim(lname(l0, "rel_logits_proj"), 1));
        vr = static_cast<int>(a.dim("G.lm_head", 0));
        max_pps = (o.cap + ink2::KV_PAGE_TOKENS - 1) / ink2::KV_PAGE_TOKENS;
        use_far = o.fast_allreduce < 0 ? knob::ink2_fast_allreduce() : o.fast_allreduce != 0;
        use_ghp = (o.gpu_head_pack < 0 ? knob::ink2_gpu_head_pack() : o.gpu_head_pack != 0) && o.nc <= ink2::INK2_HEAD_MAX_NC;
        use_blas = o.cublas_prefill < 0 ? knob::ink2_cublas_prefill() : o.cublas_prefill != 0;
        use_tc = o.tc_prefill_attn < 0 ? knob::ink2_tc_prefill_attn() : o.tc_prefill_attn != 0;
        const size_t mh = static_cast<size_t>(o.max_rows) * c.H;
        embed = a.get<uint16_t>("G.embed");
        embed_norm = a.get<uint16_t>("G.embed_norm");
        final_norm = a.get<uint16_t>("G.final_norm");
        output_query = a.get<float>("G.attn_res.output_query");
        if (c.is_looped())
        {
            if (static_cast<int>(layer_ids.size()) != c.n_layers) die("looped Inkling-2 requires --layers all");
            msg_queries = a.get<float>("G.loop.msg_queries");
            L.resize(static_cast<size_t>(c.n_virtual_layers()));
            int v = 0;
            for (int l = 0; l < c.loop_start; ++l, ++v) bind_layer(L[v], l, v, 0, -1);
            const int nl = c.n_loop_layers();
            for (int pass = 0; pass < c.loop_passes; ++pass)
                for (int j = 0; j < nl; ++j, ++v) bind_layer(L[v], c.loop_start + j, v, pass, j);
            for (int l = c.loop_end; l < c.n_layers; ++l, ++v) bind_layer(L[v], l, v, 0, -1);
        }
        else
        {
            L.resize(layer_ids.size());
            for (size_t j = 0; j < L.size(); ++j) bind_layer(L[j], layer_ids[j], static_cast<int>(j), 0, -1);
        }
        int max_tr3_eb = 0;
        for (const Layer& y : L)
            if (y.exp_fmt >= 1 && y.exp_fmt <= 3)
                for (const RankPart& x : y.rk) max_tr3_eb = std::max(max_tr3_eb, x.tr3_eb);
        mw.resize(static_cast<size_t>(o.nloc));
        const int i_work = std::max(64, (c.I / o.nr + 31) / 32 * 32);
        for (int i = 0; i < o.nloc; ++i)
        {
            mw[i].alloc(o.max_rows, c.H, i_work, c.NE, c.TK, false);
            lm.push_back(img[i]->get<uint16_t>("G.lm_head"));
            logits.push_back(dalloc<float>(static_cast<size_t>(o.max_rows) * vr));
            if (max_tr3_eb > 0)
            {
                const size_t mt = static_cast<size_t>(ink2::plan_max_tiles(o.max_rows, c.TK, c.NE));
                tr3_a_rot.push_back(dalloc<__half>(mt * 8 * static_cast<size_t>(max_tr3_eb) * 128));
                tr3_x_had.push_back(dalloc<__half>(static_cast<size_t>(o.max_rows) * c.H));
            }
        }
        if (max_tr3_eb > 0) init_tr3_wide();
        d_slot = dalloc<int>(o.max_rows);
        d_pos = dalloc<int>(o.max_rows);
        d_run0 = dalloc<int>(o.max_rows);
        toks = dalloc<int>(o.max_rows);
        acc_dev = dalloc<uint8_t>(o.max_rows);
        for (uint16_t** p : {&h, &qi, &ki, &vi, &a_, &mid, &xm, &ms, &hf, &scratch}) *p = dalloc<uint16_t>(mh);
        const int phys_layers_for_blocks = c.is_looped() ? c.n_layers : static_cast<int>(L.size());
        const int nb = (phys_layers_for_blocks + c.block - 1) / c.block;
        if (nb + 1 > ink2::MAX_DEPTH) die("more blocks of layers than the depth attention reads");
        for (int b = 0; b < nb; ++b) states.push_back(dalloc<uint16_t>(mh));
        if (c.is_looped())
        {
            pre_h = dalloc<uint16_t>(mh);
            prev_h = dalloc<uint16_t>(mh);
            const int n_prev = c.n_loop_layers() / c.block - 1;
            for (int b = 0; b < n_prev; ++b) prev_states.push_back(dalloc<uint16_t>(mh));
            for (int q = 0; q < 3; ++q) f32_attn[q] = dalloc<float>(mh);
            for (int q = 0; q < 4; ++q) f32_msg[q] = dalloc<float>(mh);
            f32_mlp = dalloc<float>(mh);
            gate_cat = dalloc<uint16_t>(mh * 2);
            gate_g = dalloc<uint16_t>(static_cast<size_t>(o.max_rows) * c.loop_gate_rank);
            gate_sf = dalloc<float>(mh);
            gate_sm = dalloc<float>(mh);
        }
        part_stride = static_cast<size_t>(o.max_rows) * c.TK * c.H;
        parts = dalloc<float>(static_cast<size_t>(o.nr) * part_stride);
        host_logits.resize(static_cast<size_t>(o.max_rows) * vr);
        head_parts.resize(static_cast<size_t>(o.nr) * head_bytes(o.max_rows));
        if (use_blas && cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) die("cublasCreate failed");
        if (use_ghp) head_dev = dalloc<uint8_t>(head_bytes(std::max(o.max_rows, o.slots)));
        h_arg_val = dalloc<float>(static_cast<size_t>(o.max_rows) * ink2::HEAD_ARGMAX_NG);
        h_arg_id = dalloc<int>(static_cast<size_t>(o.max_rows) * ink2::HEAD_ARGMAX_NG);
        if (ink2::splitk_buf() == nullptr)
        {
            constexpr size_t SK_CAP = 2u * 1024u * 1024u;   // 8 MiB workspace for deterministic Split-K c128 MMA
            ink2::splitk_buf() = dalloc<float>(SK_CAP);
            ink2::splitk_cap() = SK_CAP;
        }
        if (o.nr == 1 && o.nloc == 1)
        {
            INK2RT_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            INK2RT_CK(cudaHostAlloc(&h_meta_pin, sizeof(int) * static_cast<size_t>(4) * o.max_rows, cudaHostAllocDefault));
            INK2RT_CK(cudaHostAlloc(&h_head_pin, head_bytes(std::max(o.max_rows, o.slots)), cudaHostAllocDefault));
            INK2RT_CK(cudaHostAlloc(&h_acc_pin, static_cast<size_t>(o.max_rows), cudaHostAllocDefault));
            const char* eg = std::getenv("INK2_GRAPH");
            if (o.use_graph == 0) use_graph = false;
            else if (o.use_graph == 1) use_graph = true;
            else use_graph = !(eg && eg[0] == '0' && eg[1] == '\0');
        }
        if (o.mtp > 0) init_mtp();
        if (o.nr == 1 && o.nloc == 1 && o.mtp > 0)
        {
            INK2RT_CK(cudaHostAlloc(&h_mtp_meta_pin, sizeof(int) * (static_cast<size_t>(4) * o.slots + static_cast<size_t>(o.slots) * c.mtp_block), cudaHostAllocDefault));
            INK2RT_CK(cudaHostAlloc(&h_draft_out_pin, sizeof(int) * static_cast<size_t>(o.slots) * o.mtp, cudaHostAllocDefault));
        }
        if (o.nloc == 1 && !L.empty())
        {
            std::vector<ink2::CommitLayerDesc> hdesc;
            hdesc.reserve(L.size());
            for (const Layer& s : L)
            {
                const RankPart& x = s.rk[0];
                ink2::CommitLayerDesc d{};
                d.att = s.att;
                d.mo = s.mo;
                d.kp = x.wk.kp;
                d.vp = x.wk.vp;
                d.kv_new = x.wk.kv_new;
                d.hist_a = s.hist_a;
                d.hist_m = s.hist_m;
                d.hist_k = x.as.hist_k;
                d.hist_v = x.as.hist_v;
                d.ring = x.as.ring;
                d.nk = x.g.nh_kv * x.g.hd;
                d.window = x.g.window;
                d.is_local = x.g.global ? 0 : 1;
                hdesc.push_back(d);
            }
            commit_descs = dalloc<ink2::CommitLayerDesc>(hdesc.size());
            INK2RT_CK(cudaMemcpy(commit_descs, hdesc.data(), hdesc.size() * sizeof(ink2::CommitLayerDesc), cudaMemcpyHostToDevice));
        }
        if (o.vision && a.find("V.proj_out.weight")) init_vision();
        prof.init();
        INK2RT_CK(cudaDeviceSynchronize());
        fprintf(stderr, "# ink2: ranks %d..%d of %d, %zu layers (%d phys%s), H %d, %d experts top %d, vocab %d (%d rows a rank), %d slots, %d rows, cap %d\n", o.rank0,
                o.rank0 + o.nloc - 1, o.nr, L.size(), c.n_layers, c.is_looped() ? " looped" : "", c.H, c.NE, c.TK, c.vocab, vr, o.slots, o.max_rows, o.cap);
    }

    // ---- image input (ink2_vision.cuh). The tower runs on this engine's first rank; every rank computes the same rows.
    bool vision_ready() const { return vis && vis->ready(); }
    // 16 hex digits over the tower's tensors as the image plans them (name, dtype, shape, bytes): the ranks compare it
    const std::string& vision_id() const { return vis_id_; }
    size_t vision_device_bytes() const { return vis ? vis->device_bytes() : 0; }
    long vision_encode_calls() const { return vis ? vis->encode_calls : 0; }
    // The image rows of a launch of `rows` rows at prompt positions first .. (forward's vis_rows), or null when no
    // image of the request overlaps them. The pointer is valid until the next call.
    const uint16_t* stage_vision(const std::vector<g6rt::ImageIn>& images, int first, int rows, uint64_t req_id)
    {
        if (images.empty()) return nullptr;
        if (!vision_ready()) die("an image request, and the engine runs no vision tower");
        try
        {
            return vis->stage(images, first, rows, static_cast<unsigned long long>(req_id));
        }
        catch (const std::exception& e)
        {
            die(e.what());
        }
    }
    // One image through the tower: BF16 [tokens][H] to the device pointer out (a test compares it with the reference).
    void encode_vision(const uint8_t* rgb, int vit_h, int vit_w, uint16_t* out, int content_h = -1, int content_w = -1)
    {
        if (!vision_ready()) die("encode_vision: the engine runs no vision tower");
        try
        {
            vis->encode(rgb, vit_h, vit_w, out, content_h, content_w);
        }
        catch (const std::exception& e)
        {
            die(e.what());
        }
    }

    // A new run on a slot: its convolution histories and rings read as zeros.
    void reset_slot(int slot)
    {
        if (stream) INK2RT_CK(cudaStreamSynchronize(stream));
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
        if (o.mtp < 1) return;
        for (MtpLayer& y : ML)
            for (int i = 0; i < o.nloc; ++i)
            {
                MtpRank& x = y.rk[static_cast<size_t>(i)];
                const size_t nk = static_cast<size_t>(rank_nk(i));
                if (!y.glob) INK2RT_CK(cudaMemset(x.ring + s * 2 * c.mtp_window * nk, 0, static_cast<size_t>(2) * c.mtp_window * nk * 2));
                else INK2RT_CK(cudaMemset(x.staging + s * 2 * ink2::KV_PAGE_TOKENS * nk, 0, static_cast<size_t>(2) * ink2::KV_PAGE_TOKENS * nk * 2));
            }
        INK2RT_CK(cudaMemset(last_taps + s * ntaps * c.H, 0, static_cast<size_t>(ntaps) * c.H * 2));
    }

    // One step over the rows of rh. No state changes before commit(). When samples is null (prefill), the head is skipped.
    // With the drafter on, the taps of the rows are kept for commit().
    // vis_rows [M][H] (stage_vision) or null: the row of an image token is read from it in place of its embedding.
    void forward(const ink2::RowsHost& rh, const int* tk, const RowSample* samples, std::vector<RowOut>& out, const uint16_t* vis_rows = nullptr, bool exact_head = false)
    {
        M = rh.rows();
        if (M < 1 || M > o.max_rows) die("step of " + std::to_string(M) + " rows");
        if (rh.max_pos >= o.cap) die("position " + std::to_string(rh.max_pos) + " is past the cap " + std::to_string(o.cap));
        const cudaStream_t st = stream;
        ink2::Prof* pf = prof.enabled ? &prof : nullptr;
        if (pf) pf->begin_op(!samples ? 0 : (M == 1 ? 1 : 2), M);
        const size_t n = static_cast<size_t>(M) * c.H;
        const int T = o.nr;
        {
            ink2::ProfScope _ps(pf, ink2::PS_H2D_META, st);
            if (h_meta_pin)
            {
                int* p_slot = h_meta_pin;
                int* p_pos = p_slot + o.max_rows;
                int* p_run0 = p_pos + o.max_rows;
                int* p_tok = p_run0 + o.max_rows;
                memcpy(p_slot, rh.slot.data(), sizeof(int) * M);
                memcpy(p_pos, rh.pos.data(), sizeof(int) * M);
                memcpy(p_run0, rh.run0.data(), sizeof(int) * M);
                memcpy(p_tok, tk, sizeof(int) * M);
                INK2RT_CK(cudaMemcpyAsync(d_slot, p_slot, sizeof(int) * M, cudaMemcpyHostToDevice, st));
                INK2RT_CK(cudaMemcpyAsync(d_pos, p_pos, sizeof(int) * M, cudaMemcpyHostToDevice, st));
                INK2RT_CK(cudaMemcpyAsync(d_run0, p_run0, sizeof(int) * M, cudaMemcpyHostToDevice, st));
                INK2RT_CK(cudaMemcpyAsync(toks, p_tok, sizeof(int) * M, cudaMemcpyHostToDevice, st));
            }
            else
            {
                INK2RT_CK(cudaMemcpy(d_slot, rh.slot.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
                INK2RT_CK(cudaMemcpy(d_pos, rh.pos.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
                INK2RT_CK(cudaMemcpy(d_run0, rh.run0.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
                INK2RT_CK(cudaMemcpy(toks, tk, sizeof(int) * M, cudaMemcpyHostToDevice));
            }
        }
        const ink2::Rows rw{d_slot, d_pos, d_run0};
        if (T > 1 && step_begin)
        {
            std::vector<uint32_t> sz;
            for (size_t j = 0; j < L.size(); ++j)
            {
                sz.push_back(static_cast<uint32_t>(n * 4));
                sz.push_back(static_cast<uint32_t>(n * c.TK * 4));
            }
            if (samples) sz.push_back(static_cast<uint32_t>((head_bytes(M) + 7) / 8 * 8));
            step_begin(sz);
        }
        const bool chunk = one_run(rh);   // a prefill chunk: the speed paths that depend on M are for these rows only
        const bool tc = use_tc && chunk && M >= 16;
        const bool want_blas = use_blas && chunk && M >= 32;
        bool all_greedy = (samples != nullptr);
        for (int m = 0; m < M && all_greedy; ++m)
            if (samples[m].inv_temp > 0.f) all_greedy = false;
        static const bool fast_greedy_head = []() {
            const char* e = std::getenv("INK2_FAST_GREEDY_HEAD");
            return !(e && e[0] == '0' && e[1] == '\0');
        }();
        const int head_mode = (T == 1 && o.nloc == 1 && use_ghp && fast_greedy_head && !exact_head && all_greedy) ? 1 : 0;
        const size_t hb = head_bytes(M), rb = head_row_bytes(o.nc);

        auto run_body_and_head = [&](bool async_head_d2h) {
            {
                ink2::ProfScope _ps(pf, ink2::PS_EMBED, st);
                ink2::embed_norm(toks, embed, embed_norm, vis_rows, c.image_token, h, M, c.H, st);
            }
            int cur_nb = 0;
            const int num_pre_blocks = c.is_looped() ? (c.loop_start / c.block) : 0;
            for (size_t j = 0; j < L.size(); ++j)
            {
                Layer& s = L[j];
                const int ji = static_cast<int>(j);
                if (c.is_looped() && (ji == c.loop_start || (s.loop_pass >= 1 && s.loop_j == 0)))
                {
                    ink2::ProfScope _ps(pf, ink2::PS_LOOP_COPY_TAPS, st);
                    if (ji == c.loop_start) ink2::copy_bf16(h, pre_h, n, st);
                    if (s.loop_pass >= 1 && s.loop_j == 0)
                    {
                        for (int d = num_pre_blocks + 1; d < cur_nb; ++d)
                            ink2::copy_bf16(states[static_cast<size_t>(d)], prev_states[static_cast<size_t>(d - (num_pre_blocks + 1))], n, st);
                        ink2::copy_bf16(h, prev_h, n, st);
                        ink2::copy_bf16(pre_h, h, n, st);
                        cur_nb = num_pre_blocks;
                    }
                }
                const int tap_id = c.is_looped() ? ji : s.l;
                for (int t = 0; t < ntaps; ++t)
                    if (c.mtp_targets[static_cast<size_t>(t)] == tap_id)
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_LOOP_COPY_TAPS, st);
                        ink2::copy_bf16(h, step_taps + static_cast<size_t>(t) * o.max_rows * c.H, n, st);
                    }
                const bool gated = s.loop_pass >= 1;
                if (gated)
                {
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_ATTN_RES, st);
                        ink2::AttnResF32Args ma{};
                        const int np = static_cast<int>(prev_states.size());
                        for (int d = 0; d < np; ++d) ma.rows[d] = prev_states[static_cast<size_t>(d)];
                        ma.rows[np] = prev_h;
                        ma.query = msg_queries + static_cast<size_t>(s.loop_j) * 4 * c.H;
                        ma.D = np + 1;
                        ma.Q = 4;
                        ma.H = c.H;
                        for (int q = 0; q < 4; ++q) ma.out[q] = f32_msg[q];
                        ink2::attn_res_f32(ma, M, st);

                        ink2::AttnResF32Args ar{};
                        for (int d = 0; d < cur_nb; ++d) ar.rows[d] = states[static_cast<size_t>(d)];
                        ar.rows[cur_nb] = h;
                        ar.query = s.attn_query;
                        ar.D = cur_nb + 1;
                        ar.Q = s.glob ? 3 : 1;
                        ar.H = c.H;
                        for (int q = 0; q < ar.Q; ++q) ar.out[q] = f32_attn[q];
                        ink2::attn_res_f32(ar, M, st);
                    }
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_LOOP_MSG_GATE, st);
                        if (!ink2::stock_msg_gate_enabled() && M <= 64 && c.loop_gate_rank == 64 && c.H <= 4096 && 3 * M <= o.max_rows && gate_sf != nullptr && gate_sm != nullptr)
                        {
                            ink2::MsgGate3Args mg3{};
                            mg3.h_f32[0] = f32_attn[0];
                            mg3.h_f32[1] = f32_attn[s.glob ? 1 : 0];
                            mg3.h_f32[2] = f32_attn[s.glob ? 2 : 0];
                            mg3.msg_f32[0] = f32_msg[0];
                            mg3.msg_f32[1] = f32_msg[1];
                            mg3.msg_f32[2] = f32_msg[2];
                            mg3.down_w[0] = s.ctrl_q.down;
                            mg3.down_w[1] = s.ctrl_k.down;
                            mg3.down_w[2] = s.ctrl_v.down;
                            mg3.forget_w[0] = s.ctrl_q.forget;
                            mg3.forget_w[1] = s.ctrl_k.forget;
                            mg3.forget_w[2] = s.ctrl_v.forget;
                            mg3.gain_w[0] = s.ctrl_q.gain;
                            mg3.gain_w[1] = s.ctrl_k.gain;
                            mg3.gain_w[2] = s.ctrl_v.gain;
                            mg3.norm_w[0] = s.glob ? s.q_norm : s.attn_norm;
                            mg3.norm_w[1] = s.glob ? s.k_norm : s.attn_norm;
                            mg3.norm_w[2] = s.glob ? s.v_norm : s.attn_norm;
                            mg3.out_bf16[0] = qi;
                            mg3.out_bf16[1] = ki;
                            mg3.out_bf16[2] = vi;
                            ink2::msg_gate_norm_3(mg3, gate_cat, gate_g, gate_sf, gate_sm, M, c.H, st);
                        }
                        else
                        {
                            ink2::msg_gate_norm(f32_attn[0], f32_msg[0], s.ctrl_q.down, s.ctrl_q.forget, s.ctrl_q.gain,
                                                s.glob ? s.q_norm : s.attn_norm, gate_cat, gate_g, qi, M, c.H, c.loop_gate_rank, st, gate_sf, gate_sm);
                            ink2::msg_gate_norm(f32_attn[s.glob ? 1 : 0], f32_msg[1], s.ctrl_k.down, s.ctrl_k.forget, s.ctrl_k.gain,
                                                s.glob ? s.k_norm : s.attn_norm, gate_cat, gate_g, ki, M, c.H, c.loop_gate_rank, st, gate_sf, gate_sm);
                            ink2::msg_gate_norm(f32_attn[s.glob ? 2 : 0], f32_msg[2], s.ctrl_v.down, s.ctrl_v.forget, s.ctrl_v.gain,
                                                s.glob ? s.v_norm : s.attn_norm, gate_cat, gate_g, vi, M, c.H, c.loop_gate_rank, st, gate_sf, gate_sm);
                        }
                    }
                }
                else if (!s.attn_query)
                {
                    ink2::ProfScope _ps(pf, ink2::PS_ATTN_RES, st);
                    ink2::rmsnorm(h, s.attn_norm, qi, M, c.H, st);
                }
                else
                {
                    ink2::ProfScope _ps(pf, ink2::PS_ATTN_RES, st);
                    ink2::AttnResArgs ar{};
                    for (int d = 0; d < cur_nb; ++d) ar.rows[d] = states[static_cast<size_t>(d)];
                    ar.rows[cur_nb] = h;
                    ar.query = s.attn_query;
                    ar.D = cur_nb + 1;
                    ar.Q = s.glob ? 3 : 1;
                    ar.H = c.H;
                    ar.norm[0] = s.glob ? s.q_norm : s.attn_norm;
                    ar.norm[1] = s.k_norm;
                    ar.norm[2] = s.v_norm;
                    ar.out[0] = qi;
                    ar.out[1] = ki;
                    ar.out[2] = vi;
                    ink2::attn_res(ar, M, st);
                }
                const bool split_kv = s.glob || gated;
                for (int i = 0; i < o.nloc; ++i)
                {
                    RankPart& x = s.rk[i];
                    ink2::attn_forward(x.g, x.aw, x.as, x.wk, rw, rh.max_pos, qi, split_kv ? ki : qi, split_kv ? vi : qi, M, T == 1 ? s.att : nullptr,
                                       parts + static_cast<size_t>(o.rank0 + i) * n, st, tc, pf);
                }
                if (T > 1)
                {
                    if (gather) gather(2 * ji, parts, n * 4);
                    ink2::sum_ranks(parts, T, n, s.att, st);
                }
                {
                    ink2::ProfScope _ps(pf, ink2::PS_POST_ATTN_SCONV_RES, st);
                    ink2::sconv_static(s.att, s.hist_a, rw, s.attn_sconv, M, c.H, a_, st);
                    const bool bstart = (c.is_looped() ? s.l : ji) % c.block == 0;
                    ink2::attn_block_step(h, a_, bstart, states[static_cast<size_t>(cur_nb)], mid, n, st);
                    if (bstart) ++cur_nb;
                }
                if (gated)
                {
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_ATTN_RES, st);
                        ink2::AttnResF32Args ar{};
                        for (int d = 0; d < cur_nb; ++d) ar.rows[d] = states[static_cast<size_t>(d)];
                        ar.rows[cur_nb] = mid;
                        ar.query = s.mlp_query;
                        ar.D = cur_nb + 1;
                        ar.Q = 1;
                        ar.H = c.H;
                        ar.out[0] = f32_mlp;
                        ink2::attn_res_f32(ar, M, st);
                    }
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_LOOP_MSG_GATE, st);
                        ink2::msg_gate_norm(f32_mlp, f32_msg[3], s.ctrl_mlp.down, s.ctrl_mlp.forget, s.ctrl_mlp.gain,
                                            s.mlp_norm, gate_cat, gate_g, xm, M, c.H, c.loop_gate_rank, st, gate_sf, gate_sm);
                    }
                }
                else
                {
                    ink2::ProfScope _ps(pf, ink2::PS_ATTN_RES, st);
                    ink2::AttnResArgs ar{};
                    for (int d = 0; d < cur_nb; ++d) ar.rows[d] = states[static_cast<size_t>(d)];
                    ar.rows[cur_nb] = mid;
                    ar.query = s.mlp_query;
                    ar.D = cur_nb + 1;
                    ar.Q = 1;
                    ar.H = c.H;
                    ar.norm[0] = s.mlp_norm;
                    ar.out[0] = xm;
                    ink2::attn_res(ar, M, st);
                }
                if (s.exp_fmt >= 1 && s.exp_fmt <= 3) moe_tr3(s, ji, st);
                else if (s.mx8) moe<ink2::Mx8Frag>(s, ji, st);
                else moe<nv4::Frag>(s, ji, st);
                {
                    ink2::ProfScope _ps(pf, ink2::PS_POST_MLP_SCONV_RES, st);
                    ink2::sconv_static(s.mo, s.hist_m, rw, s.mlp_sconv, M, c.H, ms, st);
                    ink2::add_bf16(mid, ms, h, n, st);
                }
            }
            if (!samples && ntaps == 0) return;
            {
                ink2::ProfScope _ps(pf, ink2::PS_FINAL_RES, st);
                ink2::AttnResArgs ar{};
                for (int d = 0; d < cur_nb; ++d) ar.rows[d] = states[static_cast<size_t>(d)];
                ar.rows[cur_nb] = h;
                ar.query = output_query;
                ar.D = cur_nb + 1;
                ar.Q = 1;
                ar.H = c.H;
                ar.norm[0] = final_norm;
                ar.out[0] = hf;
                ink2::attn_res(ar, M, st);
                for (int t = 0; t < ntaps; ++t)
                    if (c.mtp_targets[static_cast<size_t>(t)] == c.n_virtual_layers()) ink2::copy_bf16(hf, step_taps + static_cast<size_t>(t) * o.max_rows * c.H, n, st);
            }
            if (!samples) return;
            for (int i = 0; i < o.nloc; ++i)
            {
                const int r = o.rank0 + i;
                {
                    ink2::ProfScope _ps(pf, ink2::PS_TRUNK_LM_HEAD, st);
                    ink2::lm_head(hf, lm[i], M, c.H, vr, r * vr, c.valid, logits[i], st);
                }
                if (head_mode == 1)
                {
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_TRUNK_HEAD_PACK, st);
                        ink2::head_argmax_dev(logits[i], M, vr, r * vr, h_arg_val, h_arg_id, head_dev, rb, st);
                    }
                    if (async_head_d2h) INK2RT_CK(cudaMemcpyAsync(h_head_pin, head_dev, hb, cudaMemcpyDeviceToHost, st));
                    else
                    {
                        INK2RT_CK(cudaMemcpyAsync(&head_parts[static_cast<size_t>(r) * hb], head_dev, hb, cudaMemcpyDeviceToHost, st));
                        INK2RT_CK(cudaStreamSynchronize(st));
                    }
                    continue;
                }
                if (use_ghp)
                {
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_TRUNK_HEAD_PACK, st);
                        ink2::head_pack_dev(logits[i], M, vr, r * vr, o.nc, head_dev, rb, st);
                    }
                    if (async_head_d2h) INK2RT_CK(cudaMemcpyAsync(h_head_pin, head_dev, hb, cudaMemcpyDeviceToHost, st));
                    else
                    {
                        INK2RT_CK(cudaMemcpyAsync(&head_parts[static_cast<size_t>(r) * hb], head_dev, hb, cudaMemcpyDeviceToHost, st));
                        INK2RT_CK(cudaStreamSynchronize(st));
                    }
                    continue;
                }
                INK2RT_CK(cudaMemcpyAsync(host_logits.data(), logits[i], sizeof(float) * static_cast<size_t>(M) * vr, cudaMemcpyDeviceToHost, st));
                INK2RT_CK(cudaStreamSynchronize(st));
                for (int m = 0; m < M; ++m) head_pack(&host_logits[static_cast<size_t>(m) * vr], vr, r * vr, o.nc, &head_parts[static_cast<size_t>(r) * hb + static_cast<size_t>(m) * rb]);
            }
        };

        const bool graph_ok = use_graph && !prof.enabled && T == 1 && o.nloc == 1 && st != nullptr && h_head_pin != nullptr &&
                              samples != nullptr && !vis_rows && M <= 64 && !tc && !want_blas && use_ghp;
        if (graph_ok)
        {
            const int sm_loc = ink2::attn_splits(false, c.window, rh.max_pos);
            const int sm_glb = ink2::attn_splits(true, c.window, rh.max_pos);
            const int loc_flag = (rh.max_pos < c.window) ? 1 : 0;
            const FwdGraphKey gk{M, sm_loc, sm_glb, loc_flag, head_mode};
            auto it = fwd_graphs.find(gk);
            if (it == fwd_graphs.end())
            {
                cudaGraph_t g = nullptr;
                cudaGraphExec_t exec = nullptr;
                INK2RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
                run_body_and_head(true);
                INK2RT_CK(cudaStreamEndCapture(st, &g));
                INK2RT_CK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
                INK2RT_CK(cudaGraphDestroy(g));
                fwd_graphs.emplace(gk, exec);
                INK2RT_CK(cudaGraphLaunch(exec, st));
            }
            else
            {
                INK2RT_CK(cudaGraphLaunch(it->second, st));
            }
            INK2RT_CK(cudaStreamSynchronize(st));
            out.resize(static_cast<size_t>(M));
            for (int m = 0; m < M; ++m)
            {
                out[m] = head_merge(h_head_pin + static_cast<size_t>(m) * rb, rb, 1, o.nc);
                out[m].tok = sample_row(out[m].top, samples[m]);
            }
            return;
        }

        const BlasScope blas_scope(want_blas ? blas : nullptr);
        run_body_and_head(false);
        if (!samples)
        {
            if (T > 1 && step_end) step_end();
            if (pf) pf->end_op(st);
            out.clear();
            return;
        }
        if (T > 1 && gather_head) gather_head(collectives() - 1, head_parts.data(), hb);
        if (T > 1 && step_end) step_end();
        out.resize(static_cast<size_t>(M));
        std::vector<uint8_t> row(static_cast<size_t>(T) * rb);
        for (int m = 0; m < M; ++m)
        {
            for (int r = 0; r < T; ++r) memcpy(&row[static_cast<size_t>(r) * rb], &head_parts[static_cast<size_t>(r) * hb + static_cast<size_t>(m) * rb], rb);
            out[m] = head_merge(row.data(), rb, T, o.nc);
            out[m].tok = sample_row(out[m].top, samples ? samples[m] : RowSample{});
        }
        if (pf) pf->end_op(st);
    }

    // Keep the rows of the last forward: all of them, or the ones acc marks.
    void commit(const std::vector<uint8_t>* acc = nullptr)
    {
        const cudaStream_t st = stream;
        ink2::Prof* pf = prof.enabled ? &prof : nullptr;
        if (pf) pf->begin_op(3, M);
        const ink2::Rows rw{d_slot, d_pos, d_run0};
        if (acc)
        {
            if (h_acc_pin)
            {
                memcpy(h_acc_pin, acc->data(), static_cast<size_t>(M));
                INK2RT_CK(cudaMemcpyAsync(acc_dev, h_acc_pin, static_cast<size_t>(M), cudaMemcpyHostToDevice, st));
            }
            else INK2RT_CK(cudaMemcpy(acc_dev, acc->data(), static_cast<size_t>(M), cudaMemcpyHostToDevice));
        }
        const uint8_t* am = acc ? acc_dev : nullptr;
        {
            ink2::ProfScope _ps(pf, ink2::PS_COMMIT_TRUNK, st);
            static const bool batch_commit_on = []() {
                const char* e = std::getenv("INK2_BATCH_COMMIT");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            if (batch_commit_on && commit_descs != nullptr && o.nloc == 1)
            {
                ink2::commit_all(commit_descs, static_cast<int>(L.size()), rw, am, M, c.H, st);
            }
            else
            {
                for (Layer& s : L)
                {
                    ink2::sconv_advance(s.att, s.hist_a, rw, am, M, c.H, st);
                    ink2::sconv_advance(s.mo, s.hist_m, rw, am, M, c.H, st);
                    for (RankPart& x : s.rk) ink2::attn_commit(x.g, x.as, x.wk, rw, am, M, st);
                }
            }
        }
        if (o.mtp > 0)
        {
            ink2::ProfScope _ps(pf, ink2::PS_COMMIT_MTP, st);
            mtp_commit(rw, am, st);
        }
        INK2RT_CK(cudaStreamSynchronize(st));
        if (pf) pf->end_op(st);
    }

    // K drafts (1 .. Options::mtp) per session: the tokens after `golden`, the token sampled at the session's last
    // committed position q_pos. out [S][K]. Draft step of a session is sampled like the trunk samples position
    // q_pos + 1 + step (spos = q_pos + 2 + step), so a draft equals the verified token whenever the logits agree.
    // Reads the committed context rows and last taps only; changes no state and none of the last forward's rows.
    void mtp_draft(const std::vector<int>& slots, const std::vector<int>& q_pos, const std::vector<int>& golden, const std::vector<RowSample>& samples, int K,
                   std::vector<int>& out)
    {
        const int S = static_cast<int>(slots.size()), B = c.mtp_block, H = c.H, T = o.nr, R = c.markov_rank, G = 3 * R;
        if (o.mtp < 1) die("mtp_draft: the drafter is off");
        if (S < 1 || S > o.slots || K < 1 || K > o.mtp || q_pos.size() != slots.size() || golden.size() != slots.size() || samples.size() != slots.size())
            die("mtp_draft: sessions or drafts");
        const cudaStream_t st = stream;
        ink2::Prof* pf = prof.enabled ? &prof : nullptr;
        if (pf) pf->begin_op(4, S);
        const int MB = S * B;
        std::vector<int> nctx(static_cast<size_t>(S)), pos(static_cast<size_t>(MB));
        int nctx_max = 0;
        for (int s = 0; s < S; ++s)
        {
            if (slots[s] < 0 || slots[s] >= o.slots || q_pos[s] < 0 || q_pos[s] >= o.cap || golden[s] < 0 || golden[s] >= c.vocab) die("mtp_draft: slot, position or token");
            nctx[s] = q_pos[s] + 1;
            nctx_max = std::max(nctx_max, nctx[s]);
            for (int b = 0; b < B; ++b) pos[static_cast<size_t>(s) * B + b] = nctx[s] + b;
        }
        if (h_mtp_meta_pin)
        {
            int* p_slot = h_mtp_meta_pin;
            int* p_nctx = p_slot + o.slots;
            int* p_gold = p_nctx + o.slots;
            int* p_prev = p_gold + o.slots;
            int* p_pos = p_prev + o.slots;
            memcpy(p_slot, slots.data(), sizeof(int) * S);
            memcpy(p_nctx, nctx.data(), sizeof(int) * S);
            memcpy(p_gold, golden.data(), sizeof(int) * S);
            memcpy(p_prev, golden.data(), sizeof(int) * S);
            memcpy(p_pos, pos.data(), sizeof(int) * MB);
            INK2RT_CK(cudaMemcpyAsync(m_slot, p_slot, sizeof(int) * S, cudaMemcpyHostToDevice, st));
            INK2RT_CK(cudaMemcpyAsync(m_nctx, p_nctx, sizeof(int) * S, cudaMemcpyHostToDevice, st));
            INK2RT_CK(cudaMemcpyAsync(m_gold, p_gold, sizeof(int) * S, cudaMemcpyHostToDevice, st));
            INK2RT_CK(cudaMemcpyAsync(m_prev, p_prev, sizeof(int) * S, cudaMemcpyHostToDevice, st));
            INK2RT_CK(cudaMemcpyAsync(m_pos, p_pos, sizeof(int) * MB, cudaMemcpyHostToDevice, st));
        }
        else
        {
            INK2RT_CK(cudaMemcpy(m_slot, slots.data(), sizeof(int) * S, cudaMemcpyHostToDevice));
            INK2RT_CK(cudaMemcpy(m_nctx, nctx.data(), sizeof(int) * S, cudaMemcpyHostToDevice));
            INK2RT_CK(cudaMemcpy(m_gold, golden.data(), sizeof(int) * S, cudaMemcpyHostToDevice));
            INK2RT_CK(cudaMemcpy(m_prev, golden.data(), sizeof(int) * S, cudaMemcpyHostToDevice));
            INK2RT_CK(cudaMemcpy(m_pos, pos.data(), sizeof(int) * MB, cudaMemcpyHostToDevice));
        }
        const size_t n = static_cast<size_t>(MB) * H;
        if (T > 1 && step_begin)
        {
            std::vector<uint32_t> sz(2 * ML.size(), static_cast<uint32_t>(n * 4));
            sz.insert(sz.end(), static_cast<size_t>(K), static_cast<uint32_t>((head_bytes(S) + 7) / 8 * 8));
            step_begin(sz);
        }
        bool all_greedy = (T == 1 && R % 8 == 0 && R <= 512 && m_arg_val != nullptr);
        for (int s = 0; s < S && all_greedy; ++s)
            if (samples[static_cast<size_t>(s)].inv_temp > 0.f) all_greedy = false;
        static const bool mtp_fast_argmax_on = []() {
            const char* e = std::getenv("INK2_MTP_FAST_ARGMAX");
            return !(e && e[0] == '0' && e[1] == '\0');
        }();
        out.assign(static_cast<size_t>(S) * K, 0);

        auto run_mtp_body_and_lm = [&](bool allow_blas) {
            int coll = collectives();
            {
                ink2::ProfScope _ps(pf, ink2::PS_MTP_BODY, st);
                const BlasScope mtp_blas_scope(allow_blas && use_blas && MB >= 16 ? blas : nullptr);
                ink2::k_ink2_mtp_block_init<<<MB, ink2::NRT, 0, st>>>(m_gold, embed, embed_norm, mask_embed, m_res, B, H);
                for (size_t l = 0; l < ML.size(); ++l)
                {
                    MtpLayer& y = ML[l];
                    ink2::rmsnorm(m_res, y.input_norm, m_x, MB, H, st);
                    for (int i = 0; i < o.nloc; ++i)
                    {
                        const MtpRank& x = y.rk[static_cast<size_t>(i)];
                        MtpWork& w = mwk[static_cast<size_t>(i)];
                        const ink2::AttnGeom g = rank_geom(o.rank0 + i, false);
                        const int nq = g.nh_q * g.hd, nk = g.nh_kv * g.hd;
                        if (x.wq_i8 != nullptr && MB <= 64)
                        {
                            const ink2::GemmTargetI8 tqkv_i8[3] = {
                                {x.wq_i8, x.wq_sc, w.q, nq, 0},
                                {x.wk_i8, x.wk_sc, w.kb, nk, 0},
                                {x.wv_i8, x.wv_sc, w.vb, nk, 0},
                            };
                            ink2::gemm_multi_bf16_i8(m_x, tqkv_i8, 3, MB, H, st);
                        }
                        else
                        {
                            const ink2::GemmTarget tqkv[3] = {
                                {x.wq, w.q, nq, ink2::GEMM_BF16, 0},
                                {x.wk, w.kb, nk, ink2::GEMM_BF16, 0},
                                {x.wv, w.vb, nk, ink2::GEMM_BF16, 0},
                            };
                            ink2::gemm_multi(m_x, tqkv, 3, MB, H, st);
                        }
                        ink2::mtp_head_norm_rope(w.q, y.q_norm, m_pos, inv_freq, MB, g.nh_q, g.hd, st);
                        ink2::mtp_head_norm_rope(w.kb, y.k_norm, m_pos, inv_freq, MB, g.nh_kv, g.hd, st);
                        ink2::MtpAttnArgs a{};
                        a.q = w.q;
                        a.kb = w.kb;
                        a.vb = w.vb;
                        a.ring = x.ring;
                        a.pages = x.pages;
                        a.page_table = mtp_page_table;
                        a.slot = m_slot;
                        a.nctx = m_nctx;
                        a.og = w.og;
                        a.B = B;
                        a.nh_q = g.nh_q;
                        a.q0 = g.q0;
                        a.gqa = g.gqa;
                        a.kv0 = g.kv0;
                        a.nk = nk;
                        a.hd = g.hd;
                        a.window = c.mtp_window;
                        a.global = y.glob ? 1 : 0;
                        a.max_pps = max_pps;
                        ink2::mtp_attn(a, MB, nctx_max, w.part, w.part_floats, st);
                        if (T == 1)
                        {
                            if (x.wo_i8 != nullptr && MB <= 64) ink2::gemm_bf16_i8(w.og, x.wo_i8, x.wo_sc, MB, nq, H, m_red, st);
                            else ink2::gemm_bf16(w.og, x.wo, MB, nq, H, m_red, st);
                        }
                        else ink2::gemm_f32(w.og, x.wo, MB, nq, H, mtp_parts + static_cast<size_t>(o.rank0 + i) * n, st);
                    }
                    if (T > 1)
                    {
                        if (gather) gather(coll++, mtp_parts, n * 4);
                        ink2::sum_ranks(mtp_parts, T, n, m_red, st);
                    }
                    ink2::add_bf16(m_res, m_red, m_res, n, st);
                    ink2::rmsnorm(m_res, y.post_norm, m_x, MB, H, st);
                    const int iw = c.mtp_inter / T;
                    for (int i = 0; i < o.nloc; ++i)
                    {
                        const MtpRank& x = y.rk[static_cast<size_t>(i)];
                        MtpWork& w = mwk[static_cast<size_t>(i)];
                        if (x.gate_i8 != nullptr && MB <= 64)
                        {
                            const ink2::GemmTargetI8 tgu_i8[2] = {
                                {x.gate_i8, x.gate_sc, w.gate, iw, 0},
                                {x.up_i8, x.up_sc, w.up, iw, 0},
                            };
                            ink2::gemm_multi_bf16_i8(m_x, tgu_i8, 2, MB, H, st);
                        }
                        else
                        {
                            const ink2::GemmTarget tgu[2] = {
                                {x.gate, w.gate, iw, ink2::GEMM_BF16, 0},
                                {x.up, w.up, iw, ink2::GEMM_BF16, 0},
                            };
                            ink2::gemm_multi(m_x, tgu, 2, MB, H, st);
                        }
                        const size_t cells = static_cast<size_t>(MB) * iw;
                        ink2::k_ink2_mtp_swiglu<<<ink2::cells_grid(cells), 256, 0, st>>>(w.gate, w.up, cells);
                        if (T == 1)
                        {
                            if (x.down_i8 != nullptr && MB <= 64) ink2::gemm_bf16_i8(w.gate, x.down_i8, x.down_sc, MB, iw, H, m_red, st);
                            else ink2::gemm_bf16(w.gate, x.down, MB, iw, H, m_red, st);
                        }
                        else ink2::gemm_f32(w.gate, x.down, MB, iw, H, mtp_parts + static_cast<size_t>(o.rank0 + i) * n, st);
                    }
                    if (T > 1)
                    {
                        if (gather) gather(coll++, mtp_parts, n * 4);
                        ink2::sum_ranks(mtp_parts, T, n, m_red, st);
                    }
                    ink2::add_bf16(m_res, m_red, m_res, n, st);
                }
                ink2::rmsnorm(m_res, mtp_norm, m_x, MB, H, st);
                ink2::k_ink2_mtp_pick_rows<<<ink2::cells_grid(static_cast<size_t>(S) * K * H), 256, 0, st>>>(m_x, m_hid, S, B, K, H);
                ink2::mtp_mix_norm(last_taps, static_cast<size_t>(ntaps) * H, static_cast<size_t>(H), m_slot, fuse_alpha + ML.size() * ntaps, head_dflare_norm, m_frnn, S, ntaps, H, st);
                ink2::gemm_f32(m_hid, w_hid, S * K, H, G, m_hidp, st);
                ink2::gemm_f32(m_frnn, w_frnn, S, H, G, m_frp, st);
            }
            {
                ink2::ProfScope _ps(pf, ink2::PS_MTP_BASE_LM, st);
                const BlasScope mtp_lm_blas(allow_blas && use_blas && S * K >= 16 ? blas : nullptr);
                for (int i = 0; i < o.nloc; ++i)
                {
                    MtpWork& w = mwk[static_cast<size_t>(i)];
                    if (w.lm_i8 != nullptr && w.lm_scale != nullptr && S * K <= 64)
                        ink2::gemm_bf16_i8(m_hid, w.lm_i8, w.lm_scale, S * K, H, vr, w.base, st);
                    else
                        ink2::gemm_bf16(m_hid, lm[i], S * K, H, vr, w.base, st);
                }
            }
            return coll;
        };

        const bool mtp_graph_ok = use_graph && !prof.enabled && T == 1 && o.nloc == 1 && st != nullptr && h_draft_out_pin != nullptr && all_greedy && mtp_fast_argmax_on;
        if (mtp_graph_ok)
        {
            const int sm_loc = ink2::mtp_splits(false, c.mtp_window, nctx_max, B);
            const int sm_glb = ink2::mtp_splits(true, c.mtp_window, nctx_max, B);
            const MtpGraphKey mgk{S, K, sm_loc, sm_glb};
            auto it = mtp_graphs.find(mgk);
            if (it == mtp_graphs.end())
            {
                cudaGraph_t g = nullptr;
                cudaGraphExec_t exec = nullptr;
                INK2RT_CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
                (void)run_mtp_body_and_lm(false);
                INK2RT_CK(cudaMemsetAsync(m_state, 0, static_cast<size_t>(S) * R * 2, st));
                constexpr int NG = 256;
                MtpWork& w0 = mwk[0];
                for (int step = 0; step < K; ++step)
                {
                    ink2::k_ink2_mtp_inp<<<ink2::cells_grid(static_cast<size_t>(S) * 2 * R), 256, 0, st>>>(m_state, markov_w1, m_prev, m_inp, S, R);
                    ink2::gemm_f32(m_inp, w_rec, S, 2 * R, G, m_rec, st);
                    ink2::k_ink2_mtp_gate<<<ink2::cells_grid(static_cast<size_t>(S) * R), 256, 0, st>>>(rnn_bias, m_hidp, m_frp, m_rec, m_state, m_latent, S, K, step, R);
                    ink2::mtp_fold_argmax_launch(
                        m_latent, w0.fold_w, w0.fold_w_i8, w0.fold_scale, w0.fold_b, w0.base,
                        m_arg_val, m_arg_id, m_prev, m_draft_out, NG, S, K, step, R, vr, c.valid, st);
                }
                INK2RT_CK(cudaMemcpyAsync(h_draft_out_pin, m_draft_out, sizeof(int) * static_cast<size_t>(S) * K, cudaMemcpyDeviceToHost, st));
                INK2RT_CK(cudaStreamEndCapture(st, &g));
                INK2RT_CK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
                INK2RT_CK(cudaGraphDestroy(g));
                mtp_graphs.emplace(mgk, exec);
                INK2RT_CK(cudaGraphLaunch(exec, st));
            }
            else
            {
                INK2RT_CK(cudaGraphLaunch(it->second, st));
            }
            INK2RT_CK(cudaStreamSynchronize(st));
            memcpy(out.data(), h_draft_out_pin, sizeof(int) * static_cast<size_t>(S) * K);
            return;
        }

        int coll = run_mtp_body_and_lm(true);
        INK2RT_CK(cudaMemsetAsync(m_state, 0, static_cast<size_t>(S) * R * 2, st));
        if (all_greedy && mtp_fast_argmax_on)
        {
            constexpr int NG = 256;
            MtpWork& w0 = mwk[0];
            for (int step = 0; step < K; ++step)
            {
                ink2::ProfScope _ps(pf, ink2::PS_MTP_RNN_FOLD, st);
                ink2::k_ink2_mtp_inp<<<ink2::cells_grid(static_cast<size_t>(S) * 2 * R), 256, 0, st>>>(m_state, markov_w1, m_prev, m_inp, S, R);
                ink2::gemm_f32(m_inp, w_rec, S, 2 * R, G, m_rec, st);
                ink2::k_ink2_mtp_gate<<<ink2::cells_grid(static_cast<size_t>(S) * R), 256, 0, st>>>(rnn_bias, m_hidp, m_frp, m_rec, m_state, m_latent, S, K, step, R);
                ink2::mtp_fold_argmax_launch(
                    m_latent, w0.fold_w, w0.fold_w_i8, w0.fold_scale, w0.fold_b, w0.base,
                    m_arg_val, m_arg_id, m_prev, m_draft_out, NG, S, K, step, R, vr, c.valid, st);
            }
            INK2RT_CK(cudaMemcpyAsync(out.data(), m_draft_out, sizeof(int) * static_cast<size_t>(S) * K, cudaMemcpyDeviceToHost, st));
            INK2RT_CK(cudaStreamSynchronize(st));
            if (pf) pf->end_op(st);
            return;
        }
        const size_t hb = head_bytes(S), rb = head_row_bytes(o.nc);
        std::vector<uint8_t> row(static_cast<size_t>(T) * rb);
        std::vector<int> prev(static_cast<size_t>(S));
        for (int step = 0; step < K; ++step)
        {
            {
                ink2::ProfScope _ps(pf, ink2::PS_MTP_RNN_FOLD, st);
                ink2::k_ink2_mtp_inp<<<ink2::cells_grid(static_cast<size_t>(S) * 2 * R), 256, 0, st>>>(m_state, markov_w1, m_prev, m_inp, S, R);
                ink2::gemm_f32(m_inp, w_rec, S, 2 * R, G, m_rec, st);
                ink2::k_ink2_mtp_gate<<<ink2::cells_grid(static_cast<size_t>(S) * R), 256, 0, st>>>(rnn_bias, m_hidp, m_frp, m_rec, m_state, m_latent, S, K, step, R);
                for (int i = 0; i < o.nloc; ++i)
                {
                    const int r = o.rank0 + i;
                    MtpWork& w = mwk[static_cast<size_t>(i)];
                    ink2::gemm_f32(m_latent, w.fold_w, S, R, vr, w.lg, st);
                    ink2::k_ink2_mtp_logits<<<ink2::cells_grid(static_cast<size_t>(S) * vr), 256, 0, st>>>(w.base, w.fold_b, w.lg, S, K, step, vr, r * vr, c.valid);
                }
            }
            for (int i = 0; i < o.nloc; ++i)
            {
                const int r = o.rank0 + i;
                MtpWork& w = mwk[static_cast<size_t>(i)];
                if (use_ghp)
                {
                    {
                        ink2::ProfScope _ps(pf, ink2::PS_MTP_HEAD_PACK, st);
                        ink2::head_pack_dev(w.lg, S, vr, r * vr, o.nc, head_dev, rb, st);
                    }
                    INK2RT_CK(cudaMemcpyAsync(&mtp_head_parts[static_cast<size_t>(r) * hb], head_dev, hb, cudaMemcpyDeviceToHost, st));
                    INK2RT_CK(cudaStreamSynchronize(st));
                    continue;
                }
                INK2RT_CK(cudaMemcpyAsync(mtp_host_logits.data(), w.lg, sizeof(float) * static_cast<size_t>(S) * vr, cudaMemcpyDeviceToHost, st));
                INK2RT_CK(cudaStreamSynchronize(st));
                for (int s = 0; s < S; ++s)
                    head_pack(&mtp_host_logits[static_cast<size_t>(s) * vr], vr, r * vr, o.nc, &mtp_head_parts[static_cast<size_t>(r) * hb + static_cast<size_t>(s) * rb]);
            }
            if (T > 1 && gather_head) gather_head(coll++, mtp_head_parts.data(), hb);
            for (int s = 0; s < S; ++s)
            {
                for (int r = 0; r < T; ++r) memcpy(&row[static_cast<size_t>(r) * rb], &mtp_head_parts[static_cast<size_t>(r) * hb + static_cast<size_t>(s) * rb], rb);
                RowSample sm = samples[static_cast<size_t>(s)];
                sm.spos = static_cast<uint32_t>(q_pos[s] + 2 + step);
                prev[s] = sample_row(head_merge(row.data(), rb, T, o.nc).top, sm);
                out[static_cast<size_t>(s) * K + step] = prev[s];
            }
            if (step + 1 < K) INK2RT_CK(cudaMemcpyAsync(m_prev, prev.data(), sizeof(int) * S, cudaMemcpyHostToDevice, st));
        }
        if (T > 1 && step_end) step_end();
        if (pf) pf->end_op(st);
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
    bool gpu_head_pack_active() const { return use_ghp; }
    bool cublas_prefill_active() const { return use_blas && blas != nullptr; }

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
        if (stream) INK2RT_CK(cudaStreamSynchronize(stream));
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
        for (MtpLayer& y : ML)
        {
            if (!y.glob) continue;
            MtpRank& x = y.rk[static_cast<size_t>(i)];
            const size_t elems = static_cast<size_t>(2) * ink2::KV_PAGE_TOKENS * rank_nk(i);
            uint16_t* stg = x.staging + static_cast<size_t>(slot) * elems;
            if (rem > 0 && p < static_cast<uint32_t>(max_pps))
            {
                uint16_t* pg = x.pages + (static_cast<size_t>(slot) * max_pps + p) * elems;
                if (to_staging) INK2RT_CK(cudaMemcpy(stg, pg, elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
                else INK2RT_CK(cudaMemcpy(pg, stg, elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
            }
            else if (to_staging) INK2RT_CK(cudaMemset(stg, 0, elems * sizeof(uint16_t)));
        }
        INK2RT_CK(cudaDeviceSynchronize());
    }

    // The drafter's state for a KvStateDesc: its layers' context rows (no convolution histories) and, per slot, the
    // taps of the last committed row [slots][taps * H].
    std::vector<KvPtrs> mtp_kv_ptrs(int i = 0) const
    {
        std::vector<KvPtrs> v;
        for (const MtpLayer& y : ML)
        {
            const MtpRank& x = y.rk[static_cast<size_t>(i)];
            KvPtrs k;
            k.global = y.glob;
            k.pages = reinterpret_cast<uint8_t*>(x.pages);
            k.staging = reinterpret_cast<uint8_t*>(x.staging);
            k.ring = reinterpret_cast<uint8_t*>(x.ring);
            v.push_back(k);
        }
        return v;
    }
    uint8_t* mtp_last_taps() const { return reinterpret_cast<uint8_t*>(last_taps); }
    int mtp_taps() const { return ntaps; }

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
    void bind_layer(Layer& s, int l, int v, int loop_pass, int loop_j)
    {
        Image& a = *img[0];
        const auto N = [l](const char* w) { return lname(l, w); };
        s.l = l;
        s.v = v;
        s.loop_pass = loop_pass;
        s.loop_j = loop_j;
        s.glob = a.find(N("q_input_norm")) != nullptr;
        const bool is_tr3 = a.find(N("tr3.fc1")) != nullptr;
        if (is_tr3)
        {
            s.exp_fmt = a.entry(N("tr3.fc1")).expert_format;
            if (s.exp_fmt < 1 || s.exp_fmt > 3) die(N("tr3.fc1") + ": unsupported exlr8 expert_format " + std::to_string(s.exp_fmt));
            s.mx8 = false;
        }
        else
        {
            s.exp_fmt = a.entry(N("x.fc1")).expert_format;
            s.mx8 = s.exp_fmt == 6;
            if (!s.mx8 && s.exp_fmt != 5) die(N("x.fc1") + ": the experts are neither nv4g, mxfp8 nor exlr8 tr3");
            if (c.I % o.nr != 0) die("TP " + std::to_string(o.nr) + " does not divide the expert columns");
        }
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
        if (a.dim(N("router.w"), 0) != c.NE) die(N("router.w") + ": experts differ from config.json");
        if (!is_tr3 && a.dim(N("x.fc1"), 1) * 16 != c.I / o.nr) die(N("x.fc1") + ": columns differ from config.json");
        if (loop_j >= 0)
        {
            auto bind_ctrl = [&](CtrlWeights& cw, const char* nm) {
                cw.down = a.get<uint16_t>(N((std::string("ctrl.") + nm + ".down").c_str()));
                cw.forget = a.get<uint16_t>(N((std::string("ctrl.") + nm + ".forget").c_str()));
                cw.gain = a.get<uint16_t>(N((std::string("ctrl.") + nm + ".gain").c_str()));
            };
            bind_ctrl(s.ctrl_q, "q");
            bind_ctrl(s.ctrl_k, "k");
            bind_ctrl(s.ctrl_v, "v");
            bind_ctrl(s.ctrl_mlp, "mlp");
        }
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
            // the split partials over every span of keys; 16 global rows fill 32-query GQA CTAs on both TP=6 and TP=4
            const size_t spans = static_cast<size_t>(keys + ink2::ATTN_SPAN - 1) / ink2::ATTN_SPAN + 1;
            const size_t rows = s.glob ? std::min<size_t>(16, static_cast<size_t>(o.max_rows)) : static_cast<size_t>(o.max_rows);
            x.wk.alloc(x.g, o.max_rows, rows * x.g.nh_q * spans * (x.g.hd + 2));
            if (is_tr3)
            {
                x.tr3_fc1 = im.get<uint8_t>(N("tr3.fc1"));
                x.tr3_fc2 = im.get<uint8_t>(N("tr3.fc2"));
                x.tr3_v1 = im.get<__half>(N("tr3.v1"));
                x.tr3_v2 = im.get<__half>(N("tr3.v2"));
                x.tr3_v3 = im.get<__half>(N("tr3.v3"));
                x.tr3_tab = im.get<long long>(N("tr3.table"));
                x.tr3_kmix = im.opt<long long>(N("tr3.kmix"));
                x.tr3_nblk = static_cast<int>(im.dim(N("tr3.v1"), 0));
                const int nh_tot = c.I / 128;
                x.tr3_eb = (nh_tot + o.nr - 1) / o.nr;
            }
            else
            {
                x.fc1 = im.get<uint8_t>(N("x.fc1"));
                x.fc2 = im.get<uint8_t>(N("x.fc2"));
                if (!s.mx8)
                {
                    x.s1 = im.get<float>(N("x.s1"));
                    x.s2 = im.get<float>(N("x.s2"));
                }
            }
        }
    }
    static std::string mname(int l, const char* what)
    {
        char b[96];
        snprintf(b, sizeof b, "MTP.L%02d.%s", l, what);
        return b;
    }
    // The drafter of the checkpoint: its weights, its per-slot context rows and the buffers of a draft.
    void init_mtp()
    {
        Image& a = *img[0];
        const int H = c.H, R = c.markov_rank, B = c.mtp_block, T = o.nr, nl = c.mtp_layers;
        ntaps = static_cast<int>(c.mtp_targets.size());
        if (nl < 1 || !a.find("MTP.fuse_weight")) die("--mtp " + std::to_string(o.mtp) + ": the checkpoint has no MTP drafter");
        if (o.mtp > c.mtp_drafts) die("--mtp " + std::to_string(o.mtp) + ": the drafter makes at most " + std::to_string(c.mtp_drafts) + " drafts");
        if (c.mtp_H != H || c.mtp_heads != c.heads || c.mtp_kv_heads != c.kv_heads || c.mtp_hd != c.hd || c.hd % 2) die("mtp_config: hidden_size or heads differ from the trunk's");
        if (B < 2 || c.mtp_drafts > B - 1 || c.mtp_window < 1 || ntaps < 1) die("mtp_config: block_size, num_speculative_tokens, sliding_window_size or target_layer_ids");
        if (c.mtp_inter < 1 || c.mtp_inter % T || (c.mtp_inter / T) % 16 || R < 16 || R % 16) die("mtp_config: intermediate_size or markov_rank");
        if (o.slots > o.max_rows) die("the drafter needs max_rows >= slots");
        for (int t : c.mtp_targets)
            if (c.is_looped())
            {
                if (t < 0 || t > c.n_virtual_layers()) die("mtp_config: target layer " + std::to_string(t) + " out of virtual range");
            }
            else if (t != c.n_layers && std::find(layer_ids.begin(), layer_ids.end(), t) == layer_ids.end())
                die("mtp_config: target layer " + std::to_string(t) + " is not run");
        if (a.entry("MTP.fuse_weight").nbytes != static_cast<size_t>(nl + 1) * ntaps * 2) die("MTP.fuse_weight: not [layers + 1][taps]");
        if (a.dim("MTP.markov_head.rnn_proj.weight", 0) != 3 * R || a.dim("MTP.markov_head.rnn_proj.weight", 1) != 2 * R + 2 * H || a.dim("MTP.markov_head.markov_w1", 1) != R ||
            a.dim("MTP.markov_head.hidden_out.weight", 0) != H || a.dim("MTP.markov_head.hidden_out.weight", 1) != R)
            die("MTP.markov_head: shapes differ from mtp_config");
        const size_t sl = static_cast<size_t>(o.slots), G = static_cast<size_t>(3) * R;
        // softmax of the fuse weights per consumer, to BF16
        {
            const size_t nf = static_cast<size_t>(nl + 1) * ntaps;
            std::vector<uint16_t> fw(nf), al(nf);
            INK2RT_CK(cudaMemcpy(fw.data(), a.get<uint16_t>("MTP.fuse_weight"), nf * 2, cudaMemcpyDeviceToHost));
            for (int k = 0; k <= nl; ++k)
            {
                const uint16_t* f = &fw[static_cast<size_t>(k) * ntaps];
                float mx = ink2::bf(f[0]), sum = 0.0f;
                for (int t = 1; t < ntaps; ++t) mx = std::fmax(mx, ink2::bf(f[t]));
                for (int t = 0; t < ntaps; ++t) sum += std::exp(ink2::bf(f[t]) - mx);
                for (int t = 0; t < ntaps; ++t) al[static_cast<size_t>(k) * ntaps + t] = ink2::bf16(std::exp(ink2::bf(f[t]) - mx) / sum);
            }
            fuse_alpha = dalloc<uint16_t>(nf);
            INK2RT_CK(cudaMemcpy(fuse_alpha, al.data(), nf * 2, cudaMemcpyHostToDevice));
            std::vector<float> fr(static_cast<size_t>(c.hd / 2));
            for (int j = 0; j < c.hd / 2; ++j) fr[static_cast<size_t>(j)] = static_cast<float>(std::pow(static_cast<double>(c.mtp_theta), -2.0 * j / c.hd));
            inv_freq = dalloc<float>(fr.size());
            INK2RT_CK(cudaMemcpy(inv_freq, fr.data(), fr.size() * 4, cudaMemcpyHostToDevice));
        }
        mask_embed = a.get<uint16_t>("MTP.mask_embed");
        mtp_norm = a.get<uint16_t>("MTP.norm.weight");
        head_dflare_norm = a.get<uint16_t>("MTP.markov_head.dflare_norm.weight");
        markov_w1 = a.get<uint16_t>("MTP.markov_head.markov_w1");
        rnn_bias = a.get<uint16_t>("MTP.markov_head.rnn_proj.bias");
        // rnn_proj [3 R][2 R + 2 H] by columns: the recurrent input, the block row, the mixed taps
        {
            const uint16_t* w = a.get<uint16_t>("MTP.markov_head.rnn_proj.weight");
            const size_t pitch = (static_cast<size_t>(2) * R + 2 * H) * 2;
            w_rec = dalloc<uint16_t>(G * 2 * R);
            w_hid = dalloc<uint16_t>(G * H);
            w_frnn = dalloc<uint16_t>(G * H);
            INK2RT_CK(cudaMemcpy2D(w_rec, static_cast<size_t>(2) * R * 2, w, pitch, static_cast<size_t>(2) * R * 2, G, cudaMemcpyDeviceToDevice));
            INK2RT_CK(cudaMemcpy2D(w_hid, static_cast<size_t>(H) * 2, w + 2 * R, pitch, static_cast<size_t>(H) * 2, G, cudaMemcpyDeviceToDevice));
            INK2RT_CK(cudaMemcpy2D(w_frnn, static_cast<size_t>(H) * 2, w + 2 * R + H, pitch, static_cast<size_t>(H) * 2, G, cudaMemcpyDeviceToDevice));
        }
        // every slot owns max_pps pages, in order, as in the trunk
        bool any_glob = false;
        {
            const int total = o.slots * max_pps;
            std::vector<int> table(static_cast<size_t>(total));
            for (int p = 0; p < total; ++p) table[static_cast<size_t>(p)] = p;
            int* t = dalloc<int>(static_cast<size_t>(total));
            INK2RT_CK(cudaMemcpy(t, table.data(), sizeof(int) * total, cudaMemcpyHostToDevice));
            mtp_page_table = t;
        }
        ML.resize(static_cast<size_t>(nl));
        for (int l = 0; l < nl; ++l)
        {
            MtpLayer& y = ML[static_cast<size_t>(l)];
            const auto N = [l](const char* w) { return mname(l, w); };
            y.glob = c.mtp_global[static_cast<size_t>(l)] != 0;
            any_glob = any_glob || y.glob;
            y.dflare_norm = a.get<uint16_t>(N("dflare_norm.weight"));
            y.input_norm = a.get<uint16_t>(N("input_layernorm.weight"));
            y.post_norm = a.get<uint16_t>(N("post_attention_layernorm.weight"));
            y.q_norm = a.get<uint16_t>(N("self_attn.q_norm.weight"));
            y.k_norm = a.get<uint16_t>(N("self_attn.k_norm.weight"));
            y.k_norm_ctx = a.get<uint16_t>(N("self_attn.k_norm_ctx.weight"));
            y.rk.resize(static_cast<size_t>(o.nloc));
            for (int i = 0; i < o.nloc; ++i)
            {
                Image& im = *img[i];
                MtpRank& x = y.rk[static_cast<size_t>(i)];
                const ink2::AttnGeom g = rank_geom(o.rank0 + i, false);
                const int64_t nq = static_cast<int64_t>(g.nh_q) * g.hd, nk = static_cast<int64_t>(g.nh_kv) * g.hd, iw = c.mtp_inter / T;
                if (im.dim(N("self_attn.q_proj.weight"), 0) != nq || im.dim(N("self_attn.k_proj.weight"), 0) != nk || im.dim(N("self_attn.k_proj_ctx.weight"), 0) != nk ||
                    im.dim(N("self_attn.o_proj.weight"), 1) != nq || im.dim(N("mlp.gate_proj.weight"), 0) != iw || im.dim(N("mlp.down_proj.weight"), 1) != iw)
                    die(N("self_attn.q_proj.weight") + ": the rank's slices differ from mtp_config");
                x.wq = im.get<uint16_t>(N("self_attn.q_proj.weight"));
                x.wk = im.get<uint16_t>(N("self_attn.k_proj.weight"));
                x.wv = im.get<uint16_t>(N("self_attn.v_proj.weight"));
                x.wkc = im.get<uint16_t>(N("self_attn.k_proj_ctx.weight"));
                x.wvc = im.get<uint16_t>(N("self_attn.v_proj_ctx.weight"));
                x.wo = im.get<uint16_t>(N("self_attn.o_proj.weight"));
                x.gate = im.get<uint16_t>(N("mlp.gate_proj.weight"));
                x.up = im.get<uint16_t>(N("mlp.up_proj.weight"));
                x.down = im.get<uint16_t>(N("mlp.down_proj.weight"));
                if (!y.glob) x.ring = dalloc<uint16_t>(sl * 2 * c.mtp_window * static_cast<size_t>(nk));
                else
                {
                    x.pages = dalloc<uint16_t>(sl * max_pps * 2 * ink2::KV_PAGE_TOKENS * static_cast<size_t>(nk));
                    x.staging = dalloc<uint16_t>(sl * 2 * ink2::KV_PAGE_TOKENS * static_cast<size_t>(nk));
                }
            }
        }
        const size_t mh = static_cast<size_t>(o.max_rows) * H, MBm = sl * B, km = sl * o.mtp;
        step_taps = dalloc<uint16_t>(static_cast<size_t>(ntaps) * mh);
        last_taps = dalloc<uint16_t>(sl * ntaps * H);
        ctx_x = dalloc<uint16_t>(mh);
        m_res = dalloc<uint16_t>(MBm * H);
        m_x = dalloc<uint16_t>(MBm * H);
        m_red = dalloc<uint16_t>(MBm * H);
        m_hid = dalloc<uint16_t>(km * H);
        m_frnn = dalloc<uint16_t>(sl * H);
        m_hidp = dalloc<float>(km * G);
        m_frp = dalloc<float>(sl * G);
        m_rec = dalloc<float>(sl * G);
        m_state = dalloc<uint16_t>(sl * R);
        m_latent = dalloc<uint16_t>(sl * R);
        m_inp = dalloc<uint16_t>(sl * 2 * R);
        m_slot = dalloc<int>(sl);
        m_nctx = dalloc<int>(sl);
        m_gold = dalloc<int>(sl);
        m_prev = dalloc<int>(sl);
        m_pos = dalloc<int>(MBm);
        m_arg_val = dalloc<float>(sl * 256);
        m_arg_id = dalloc<int>(sl * 256);
        m_draft_out = dalloc<int>(sl * static_cast<size_t>(o.mtp));
        mtp_parts = T > 1 ? dalloc<float>(static_cast<size_t>(T) * MBm * H) : nullptr;
        mtp_host_logits.resize(sl * vr);
        mtp_head_parts.resize(static_cast<size_t>(T) * head_bytes(o.slots));
        const uint16_t* how = a.get<uint16_t>("MTP.markov_head.hidden_out.weight");
        const uint16_t* hob = a.get<uint16_t>("MTP.markov_head.hidden_out.bias");
        mwk.resize(static_cast<size_t>(o.nloc));
        for (int i = 0; i < o.nloc; ++i)
        {
            MtpWork& w = mwk[static_cast<size_t>(i)];
            const ink2::AttnGeom g = rank_geom(o.rank0 + i, false);
            const size_t nq = static_cast<size_t>(g.nh_q) * g.hd, nk = static_cast<size_t>(g.nh_kv) * g.hd, iw = static_cast<size_t>(c.mtp_inter / T);
            w.q = dalloc<uint16_t>(MBm * nq);
            w.og = dalloc<uint16_t>(MBm * nq);
            w.kb = dalloc<uint16_t>(MBm * nk);
            w.vb = dalloc<uint16_t>(MBm * nk);
            w.gate = dalloc<uint16_t>(MBm * iw);
            w.up = dalloc<uint16_t>(MBm * iw);
            w.ck = dalloc<uint16_t>(static_cast<size_t>(o.max_rows) * nk);
            w.cv = dalloc<uint16_t>(static_cast<size_t>(o.max_rows) * nk);
            w.base = dalloc<uint16_t>(km * vr);
            w.lg = dalloc<float>(sl * vr);
            // the split partials of 16 block rows over every span of keys a row can read
            const size_t spans = static_cast<size_t>((any_glob ? o.cap : std::min(c.mtp_window, o.cap)) + B + ink2::ATTN_SPAN - 1) / ink2::ATTN_SPAN;
            w.part_floats = static_cast<size_t>(16) * g.nh_q * spans * (g.hd + 2);
            w.part = dalloc<float>(w.part_floats);
            // lm_head . hidden_out, once
            w.fold_w = dalloc<uint16_t>(static_cast<size_t>(vr) * R);
            w.fold_b = dalloc<uint16_t>(static_cast<size_t>(vr));
            ink2::k_ink2_mtp_fold<<<vr, 256, 0, 0>>>(lm[static_cast<size_t>(i)], how, hob, w.fold_w, w.fold_b, H, R);
            static const bool mtp_i8_on = []() {
                const char* e = std::getenv("INK2_MTP_INT8_LM");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            if (mtp_i8_on && (H % 64 == 0) && (R % 8 == 0))
            {
                w.lm_i8 = dalloc<int8_t>(static_cast<size_t>(vr) * H);
                w.lm_scale = dalloc<float>(static_cast<size_t>(vr));
                w.fold_w_i8 = dalloc<int8_t>(static_cast<size_t>(vr) * R);
                w.fold_scale = dalloc<float>(static_cast<size_t>(vr));
                const unsigned gx_q = static_cast<unsigned>((vr + 3) / 4);
                ink2::k_ink2_quant_row_i8<<<gx_q, 128, 0, 0>>>(lm[static_cast<size_t>(i)], w.lm_i8, w.lm_scale, vr, H);
                ink2::k_ink2_quant_row_i8<<<gx_q, 128, 0, 0>>>(w.fold_w, w.fold_w_i8, w.fold_scale, vr, R);
            }
        }
        size_t mtp_body_i8_bytes = 0;
        {
            static const bool mtp_i8_on = []() {
                const char* e = std::getenv("INK2_MTP_INT8_LM");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            const int iw_l = c.mtp_inter / T;
            if (mtp_i8_on && T == 1 && (H % 128 == 0) && (iw_l % 128 == 0))
            {
                auto qmat = [&](const uint16_t* src, int N, int K, int8_t*& dst, float*& sc) {
                    dst = dalloc<int8_t>(static_cast<size_t>(N) * K);
                    sc = dalloc<float>(static_cast<size_t>(N));
                    ink2::k_ink2_quant_row_i8<<<static_cast<unsigned>((N + 3) / 4), 128, 0, 0>>>(src, dst, sc, N, K);
                    mtp_body_i8_bytes += static_cast<size_t>(N) * K + static_cast<size_t>(N) * sizeof(float);
                };
                for (int l = 0; l < nl; ++l)
                {
                    MtpLayer& y = ML[static_cast<size_t>(l)];
                    for (int i = 0; i < o.nloc; ++i)
                    {
                        MtpRank& x = y.rk[static_cast<size_t>(i)];
                        const ink2::AttnGeom g = rank_geom(o.rank0 + i, false);
                        const int nq_l = g.nh_q * g.hd, nk_l = g.nh_kv * g.hd;
                        if ((nq_l % 128 == 0) && (nk_l % 16 == 0))
                        {
                            qmat(x.wq, nq_l, H, x.wq_i8, x.wq_sc);
                            qmat(x.wk, nk_l, H, x.wk_i8, x.wk_sc);
                            qmat(x.wv, nk_l, H, x.wv_i8, x.wv_sc);
                            qmat(x.wo, H, nq_l, x.wo_i8, x.wo_sc);
                            qmat(x.gate, iw_l, H, x.gate_i8, x.gate_sc);
                            qmat(x.up, iw_l, H, x.up_i8, x.up_sc);
                            qmat(x.down, H, iw_l, x.down_i8, x.down_sc);
                        }
                    }
                }
            }
        }
        INK2RT_CK(cudaDeviceSynchronize());
        if (!mwk.empty() && mwk[0].lm_i8 != nullptr)
            fprintf(stderr, "# ink2: drafter INT8 lm_head + fold_w + body enabled (%.1f MiB, exact BF16 trunk verification)\n",
                    (static_cast<size_t>(vr) * (H + R) + static_cast<size_t>(vr) * 8 + mtp_body_i8_bytes) / (1024.0 * 1024.0));
        fprintf(stderr, "# ink2: drafter of %d layers, %d taps, block %d, %d drafts a round, window %d, rank %d\n", nl, ntaps, B, o.mtp, c.mtp_window, R);
    }
    // The context rows of the accepted rows of the last forward, per drafter layer, and the slots' last taps.
    void mtp_commit(const ink2::Rows& rw, const uint8_t* am, cudaStream_t st = 0)
    {
        const int H = c.H;
        const size_t tap_stride = static_cast<size_t>(o.max_rows) * H;
        for (size_t l = 0; l < ML.size(); ++l)
        {
            MtpLayer& y = ML[l];
            ink2::mtp_mix_norm(step_taps, static_cast<size_t>(H), tap_stride, nullptr, fuse_alpha + l * ntaps, y.dflare_norm, ctx_x, M, ntaps, H, st);
            for (int i = 0; i < o.nloc; ++i)
            {
                MtpRank& x = y.rk[static_cast<size_t>(i)];
                MtpWork& w = mwk[static_cast<size_t>(i)];
                const ink2::AttnGeom g = rank_geom(o.rank0 + i, false);
                const int nk = g.nh_kv * g.hd;
                const ink2::GemmTarget tkv[2] = {
                    {x.wkc, w.ck, nk, ink2::GEMM_BF16, 0},
                    {x.wvc, w.cv, nk, ink2::GEMM_BF16, 0},
                };
                ink2::gemm_multi(ctx_x, tkv, 2, M, H, st);
                ink2::mtp_head_norm_rope(w.ck, y.k_norm_ctx, rw.pos, inv_freq, M, g.nh_kv, g.hd, st);
                ink2::k_ink2_mtp_ctx_store<<<ink2::cells_grid(static_cast<size_t>(M) * nk), 256, 0, st>>>(w.ck, w.cv, x.ring, x.pages, mtp_page_table, max_pps, rw, am, M, nk,
                                                                                                         c.mtp_window);
            }
        }
        ink2::k_ink2_mtp_last_taps<<<ink2::cells_grid(static_cast<size_t>(M) * ntaps * H), 256, 0, st>>>(step_taps, tap_stride, last_taps, rw, am, M, ntaps, H);
    }
    void init_tr3_wide()
    {
        const char* e = std::getenv("INK2_TR3_WIDE");
        if (e && e[0] == '0' && e[1] == '\0') return;
        const int HB = c.H / 128;
        if (c.H % 128 != 0) return;
        struct PhysTr3 { const __half *sg = nullptr, *v1b = nullptr, *v3b = nullptr; int wide = 0; bool done = false; };
        std::vector<PhysTr3> phys(static_cast<size_t>(c.n_layers) * o.nloc);
        int* d_ok = dalloc<int>(1);
        int n_widened = 0;
        for (Layer& s : L)
        {
            if (s.exp_fmt != 3) continue;
            for (int i = 0; i < o.nloc; ++i)
            {
                RankPart& x = s.rk[i];
                PhysTr3& pt = phys[static_cast<size_t>(s.l) * o.nloc + i];
                if (!pt.done)
                {
                    pt.done = true;
                    if (x.tr3_nblk > 0 && x.tr3_eb > 0)
                    {
                        __half* sg = dalloc<__half>(static_cast<size_t>(2) * c.H);
                        __half* v1b = dalloc<__half>(static_cast<size_t>(x.tr3_nblk) * 2 * HB);
                        __half* v3b = dalloc<__half>(static_cast<size_t>(x.tr3_nblk) * HB);
                        int one = 1;
                        INK2RT_CK(cudaMemcpy(d_ok, &one, sizeof(int), cudaMemcpyHostToDevice));
                        ink2::tr3::k_ink2_tr3_extract_sg<<<(c.H + 255) / 256, 256, 0, 0>>>(x.tr3_v1, x.tr3_v3, sg, x.tr3_nblk, c.H);
                        ink2::tr3::k_ink2_tr3_extract_verify_bs<<<x.tr3_nblk * HB, 128, 0, 0>>>(x.tr3_v1, x.tr3_v3, sg, v1b, v3b, d_ok, x.tr3_nblk, c.H);
                        int h_ok = 0;
                        INK2RT_CK(cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
                        if (h_ok == 1)
                        {
                            ink2::tr3::k_ink2_tr3_widen_fc1<<<c.NE * x.tr3_eb * 2 * HB, 256, 0, 0>>>(
                                const_cast<uint8_t*>(x.tr3_fc1), x.tr3_tab, x.tr3_kmix, x.tr3_eb, c.H);
                            ink2::tr3::k_ink2_tr3_widen_fc2<<<c.NE * x.tr3_eb * HB, 256, 0, 0>>>(
                                const_cast<uint8_t*>(x.tr3_fc2), x.tr3_tab, x.tr3_eb, c.H);
                            pt.sg = sg;
                            pt.v1b = v1b;
                            pt.v3b = v3b;
                            pt.wide = 1;
                            ++n_widened;
                        }
                        else
                        {
                            cudaFree(sg);
                            cudaFree(v1b);
                            cudaFree(v3b);
                        }
                    }
                }
                x.tr3_sg = pt.sg;
                x.tr3_v1b = pt.v1b;
                x.tr3_v3b = pt.v3b;
                x.tr3_wide = pt.wide;
            }
        }
        cudaFree(d_ok);
        INK2RT_CK(cudaDeviceSynchronize());
        if (n_widened > 0)
            fprintf(stderr, "# ink2: tr3 wide+factorized enabled on %d physical PMCGX layers (H=%d, HB=%d)\n", n_widened, c.H, HB);
    }

    void moe_tr3(Layer& s, int j, cudaStream_t st = 0)
    {
        ink2::Work& w0 = mw[0];
        const int T = o.nr;
        ink2::Prof* pf = prof.enabled ? &prof : nullptr;
        {
            ink2::ProfScope _ps(pf, ink2::PS_ROUTER, st);
            ink2::route(xm, s.router_w, s.router_bias, M, c.H, c.NE, c.TK, c.route_scale, w0.score, w0.idx, w0.wt, w0.plan, st);
        }
        const int cb = s.exp_fmt == 3 ? ink2::tr3::CB_PMCGX : s.exp_fmt == 2 ? ink2::tr3::CB_MUL1 : ink2::tr3::CB_MCG;
        const size_t picks = static_cast<size_t>(M) * c.TK * c.H;
        for (int i = 0; i < o.nloc; ++i)
        {
            const RankPart& x = s.rk[i];
            ink2::tr3::Tr3Args ta{};
            ta.fc1 = x.tr3_fc1;
            ta.fc2 = x.tr3_fc2;
            ta.v1 = x.tr3_v1;
            ta.v2 = x.tr3_v2;
            ta.v3 = x.tr3_v3;
            ta.table = x.tr3_tab;
            ta.kmix = x.tr3_kmix;
            ta.x = xm;
            ta.plan = w0.plan;
            ta.a_rot = tr3_a_rot[static_cast<size_t>(i)];
            ta.ypart = mw[i].ypart;
            ta.out = T == 1 ? s.mo : nullptr;
            ta.H = c.H;
            ta.eb = x.tr3_eb;
            ta.M = M;
            ta.TK = c.TK;
            ta.y_bf16 = 0;
            ta.sg = x.tr3_sg;
            ta.v1b = x.tr3_v1b;
            ta.v3b = x.tr3_v3b;
            ta.x_had = tr3_x_had[static_cast<size_t>(i)];
            ta.wide = x.tr3_wide;
            ink2::tr3::moe_tr3(ta, cb, c.NE, st, pf);
            if (T == 1) return;
            PickArgs pa{static_cast<const float*>(mw[i].ypart), w0.plan, M, c.H, c.TK, parts + static_cast<size_t>(o.rank0 + i) * picks};
            k_ink2rt_picks<<<ink2::cells_grid(picks), 256, 0, st>>>(pa);
        }
        if (gather) gather(2 * j + 1, parts, picks * 4);
        k_ink2rt_pick_sum<<<ink2::cells_grid(static_cast<size_t>(M) * c.H), 256, 0, st>>>(parts, T, picks, M, c.H, c.TK, s.mo);
    }
    template <class F> void moe(Layer& s, int j, cudaStream_t st = 0)
    {
        ink2::Work& w0 = mw[0];
        const int T = o.nr;
        ink2::route(xm, s.router_w, s.router_bias, M, c.H, c.NE, c.TK, c.route_scale, w0.score, w0.idx, w0.wt, w0.plan, st);
        const size_t picks = static_cast<size_t>(M) * c.TK * c.H;
        for (int i = 0; i < o.nloc; ++i)
        {
            const RankPart& x = s.rk[i];
            ink2::MoeArgs ma = mw[i].args(x.fc1, x.fc2, x.s1, x.s2, M, T == 1 ? s.mo : scratch);
            ma.plan = w0.plan;   // one routing for every rank
            if (M <= 8) ink2::moe_decode<F>(ma, xm, c.NE, st);
            else ink2::moe_prefill<F>(ma, xm, c.NE, st);
            if (T == 1) return;
            PickArgs pa{static_cast<const float*>(mw[i].ypart), w0.plan, M, c.H, c.TK, parts + static_cast<size_t>(o.rank0 + i) * picks};
            k_ink2rt_picks<<<ink2::cells_grid(picks), 256, 0, st>>>(pa);
        }
        if (gather) gather(2 * j + 1, parts, picks * 4);
        k_ink2rt_pick_sum<<<ink2::cells_grid(static_cast<size_t>(M) * c.H), 256, 0, st>>>(parts, T, picks, M, c.H, c.TK, s.mo);
    }

    std::vector<std::unique_ptr<Image>> img;
    std::unique_ptr<Ink2Vision> vis;
    std::string vis_id_;

    void init_vision()
    {
        Image& a = *img[0];
        if (!c.has_vision) die("the checkpoint has V.* tensors and its config.json has no vision_config");
        VisDims vd;
        vd.vd = c.vis_hidden;
        vd.layers = c.vis_layers;
        vd.heads = c.vis_heads;
        vd.mlp = c.vis_mlp;
        vd.patch = c.vis_patch;
        vd.merge = c.vis_merge;
        vd.grid = c.vis_grid;
        vd.proj = c.vis_proj;
        vd.H = c.H;
        vd.eps = c.vis_eps;
        vd.theta = c.vis_theta;
        const auto get = [&](const std::string& name, const std::vector<int64_t>& shape) -> const uint16_t* {
            const g6::Entry& e = a.entry("V." + name);
            if (e.dtype != g6::DType::BF16 || e.shape != shape) die("V." + name + ": not the BF16 tensor of the vision_config");
            return a.get<uint16_t>("V." + name);
        };
        // the id: every V.* entry of the image in name order
        std::vector<const g6::Entry*> ve;
        for (const g6::Entry& e : a.pack.entries())
            if (e.name.compare(0, 2, "V.") == 0) ve.push_back(&e);
        std::sort(ve.begin(), ve.end(), [](const g6::Entry* x, const g6::Entry* y) { return x->name < y->name; });
        ds41::engram::Sha256 sha;
        for (const g6::Entry* e : ve)
        {
            std::string line = e->name + " " + std::to_string(static_cast<int>(e->dtype)) + " [";
            for (int64_t s : e->shape) line += std::to_string(s) + ",";
            line += "] " + std::to_string(e->nbytes) + "\n";
            sha.update(line.data(), line.size());
        }
        vis_id_ = sha.hex().substr(0, 16);
        vis = std::make_unique<Ink2Vision>();
        vis->log = o.rank0 == 0;
        try
        {
            vis->init(vd, get, o.max_rows);
        }
        catch (const std::exception& e)
        {
            die(e.what());
        }
    }
    std::vector<Layer> L;
    std::vector<ink2::Work> mw;
    std::vector<const uint16_t*> lm;
    std::vector<float*> logits;
    std::vector<uint16_t*> states;
    std::vector<float> host_logits;
    std::vector<uint8_t> head_parts;
    cublasHandle_t blas = nullptr;   // use_blas
    uint8_t* head_dev = nullptr;   // [rows][head_row_bytes]: the device's packed head rows (use_ghp)
    bool use_far = false, use_ghp = false, use_blas = false, use_tc = false;
    const uint16_t *embed = nullptr, *embed_norm = nullptr, *final_norm = nullptr;
    const float* output_query = nullptr;
    const float* msg_queries = nullptr;   // looped only: [n_loop][4][H] fp32
    uint16_t *pre_h = nullptr, *prev_h = nullptr, *gate_cat = nullptr, *gate_g = nullptr;
    float *gate_sf = nullptr, *gate_sm = nullptr;
    std::vector<uint16_t*> prev_states;
    float *f32_attn[3] = {}, *f32_msg[4] = {}, *f32_mlp = nullptr;
    std::vector<__half*> tr3_a_rot;       // exlr8 tr3 per-local-rank activation scratch
    std::vector<__half*> tr3_x_had;       // exlr8 tr3 per-local-rank pre-rotated input scratch
    int *d_slot = nullptr, *d_pos = nullptr, *d_run0 = nullptr, *toks = nullptr;
    uint8_t* acc_dev = nullptr;
    uint16_t *h = nullptr, *qi = nullptr, *ki = nullptr, *vi = nullptr, *a_ = nullptr, *mid = nullptr, *xm = nullptr, *ms = nullptr, *hf = nullptr, *scratch = nullptr;
    float* parts = nullptr;
    size_t part_stride = 0;
    int M = 0, vr = 0, max_pps = 0;
    // the drafter (Options::mtp > 0)
    std::vector<MtpLayer> ML;
    std::vector<MtpWork> mwk;
    std::vector<float> mtp_host_logits;
    std::vector<uint8_t> mtp_head_parts;
    int ntaps = 0;
    const uint16_t *mask_embed = nullptr, *mtp_norm = nullptr, *head_dflare_norm = nullptr, *markov_w1 = nullptr, *rnn_bias = nullptr;
    const int* mtp_page_table = nullptr;
    uint16_t *fuse_alpha = nullptr, *w_rec = nullptr, *w_hid = nullptr, *w_frnn = nullptr;
    float* inv_freq = nullptr;
    uint16_t *step_taps = nullptr, *last_taps = nullptr, *ctx_x = nullptr;   // [taps][max_rows][H], [slots][taps][H], [max_rows][H]
    uint16_t *m_res = nullptr, *m_x = nullptr, *m_red = nullptr, *m_hid = nullptr, *m_frnn = nullptr, *m_state = nullptr, *m_latent = nullptr, *m_inp = nullptr;
    float *m_hidp = nullptr, *m_frp = nullptr, *m_rec = nullptr, *mtp_parts = nullptr;
    int *m_slot = nullptr, *m_nctx = nullptr, *m_gold = nullptr, *m_prev = nullptr, *m_pos = nullptr;
    float *m_arg_val = nullptr;
    int *m_arg_id = nullptr, *m_draft_out = nullptr;
    float *h_arg_val = nullptr;
    int *h_arg_id = nullptr;
    cudaStream_t stream = nullptr;
    int *h_meta_pin = nullptr, *h_mtp_meta_pin = nullptr, *h_draft_out_pin = nullptr;
    uint8_t *h_head_pin = nullptr, *h_acc_pin = nullptr;
    bool use_graph = false;
    using FwdGraphKey = std::tuple<int, int, int, int, int>;
    using MtpGraphKey = std::tuple<int, int, int, int>;
    std::map<FwdGraphKey, cudaGraphExec_t> fwd_graphs;
    std::map<MtpGraphKey, cudaGraphExec_t> mtp_graphs;
    ink2::CommitLayerDesc* commit_descs = nullptr;
    ink2::Prof prof;
};
}   // namespace ink2rt
