// Qwen3.8-Flash-Next runtime step engine (qw38rt::Engine).
// Runs all NR ranks on one GPU (m4sim_qw38) or one rank over the RDMA fabric (g6m4_qw38).
// Partial sums from attention/GDN and MoE cross the fabric as BF16 [M, 2560] and are summed in rank order
// 0..nr-1, so m4sim_qw38 and multi-box g6m4_qw38 produce byte-identical logits and tokens.
//
// Features:
//   - Exlr8 v2 indexed release loading via g6l_qw38_plan + g6l_plan_load (with optional --layout from bin/g6layout)
//   - Weight paging (GR8_PTAB=0|1|2) via engine/kernels/ptab.cuh (per-layer fixed-address piece tables + slab Pool)
//   - Layer-1 PLE ~95.4 GiB NVMe row store via io_uring + O_DIRECT + set-associative RAM LRU cache
//   - Paged full-attention KV + QSA block-4 sparse indexer + Gated DeltaNet FP32 recurrent state
//   - Exposes exact device state pointers for StashPool + GR8_KV_REWIDTH (s2/qw38_ckpt_rewidth.h)
//   - Optional per-block CUDA event profiling (GR8_QW38_PROFILE=1) for iterative perf tuning
#pragma once
#include "../model_geom_qw38.h"
#include "../kernels/qw38_hc.cuh"
#include "../kernels/qw38_gdn.cuh"
#include "../attn/qw38_attn.cuh"
#include "../engram/qw38_ple.h"
#include "../engram/qw38_ple.cuh"
#include "../kernels/qw38_moe.cuh"
#include "../kernels/qw38_mtp.cuh"
#include "../kernels/qw38_head.cuh"
#include "../kernels/ink2_head.cuh"
#include "../kernels/ptab.cuh"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"
#include "../knobs-rs/gr8_knobs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace qw38rt {

[[noreturn]] inline void die(const std::string& m) {
    std::fprintf(stderr, "qw38: %s\n", m.c_str());
    std::exit(2);
}

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}
#define QW38_CK(x) ::qw38rt::ck((x), #x)

template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    QW38_CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    QW38_CK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
    return static_cast<T*>(p);
}

// ------------------------------------------------------------------------------------------------ head / sampling
struct Cand {
    float logit;
    int id;
};
inline bool cand_before(const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); }

struct RowSample {
    float inv_temp = 0.0f, top_p = 1.0f;
    uint32_t seed = 0, spos = 0;
};

struct RowOut {
    int tok = 0;
    float lse = 0.0f;
    std::vector<Cand> top;
};

inline double sample_u(uint32_t seed, uint32_t spos) {
    uint64_t z = (static_cast<uint64_t>(seed) << 32 | spos) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

inline int sample_row(const std::vector<Cand>& top, const RowSample& s) {
    if (top.empty()) return 0;
    if (!(s.inv_temp > 0.f) || top.size() == 1) return top[0].id;
    std::vector<double> p(top.size());
    double total = 0.0;
    for (size_t i = 0; i < top.size(); ++i)
        total += p[i] = std::exp(static_cast<double>(top[i].logit - top[0].logit) * s.inv_temp);
    size_t n = top.size();
    if (s.top_p > 0.f && s.top_p < 1.f) {
        double cum = 0.0;
        for (size_t i = 0; i < top.size(); ++i) {
            cum += p[i];
            if (cum >= static_cast<double>(s.top_p) * total) { n = i + 1; break; }
        }
        total = 0.0;
        for (size_t i = 0; i < n; ++i) total += p[i];
    }
    const double u = sample_u(s.seed, s.spos) * total;
    double cum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cum += p[i];
        if (u < cum) return top[i].id;
    }
    return top[n - 1].id;
}

inline size_t head_row_bytes(int NC) { return 16 + static_cast<size_t>(NC) * sizeof(Cand); }

inline RowOut head_merge(const uint8_t* parts, size_t stride, int R, int NC) {
    RowOut o;
    float gmx = -INFINITY;
    for (int r = 0; r < R; ++r) {
        float mx;
        std::memcpy(&mx, parts + static_cast<size_t>(r) * stride, 4);
        if (mx > gmx) gmx = mx;
    }
    double total = 0.0;
    for (int r = 0; r < R; ++r) {
        const uint8_t* p = parts + static_cast<size_t>(r) * stride;
        float mx;
        int count;
        double sum;
        std::memcpy(&mx, p, 4);
        std::memcpy(&count, p + 4, 4);
        std::memcpy(&sum, p + 8, 8);
        if (count < 0 || count > NC) die("head: bad candidate count");
        if (count) total += sum * std::exp(static_cast<double>(mx - gmx));
        const size_t at = o.top.size();
        o.top.resize(at + static_cast<size_t>(count));
        if (count) std::memcpy(o.top.data() + at, p + 16, static_cast<size_t>(count) * sizeof(Cand));
    }
    std::stable_sort(o.top.begin(), o.top.end(), cand_before);
    if (o.top.size() > static_cast<size_t>(NC)) o.top.resize(static_cast<size_t>(NC));
    o.lse = total > 0.0 ? gmx + static_cast<float>(std::log(total)) : 0.f;
    return o;
}

// ------------------------------------------------------------------------------------------------ rank image
struct Image {
    g6::Pack pack;
    void* plan = nullptr;
    uint8_t* dev = nullptr;
    int rank = 0;

    struct Sink {
        uint8_t* dev = nullptr;
        std::mutex mu;
        std::vector<cudaStream_t> streams;
        static void* alloc(void*, size_t n) {
            void* p = nullptr;
            return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
        }
        static void release(void*, void* p) { cudaFreeHost(p); }
        static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n) {
            Sink& s = *static_cast<Sink*>(ctx);
            cudaStream_t st;
            {
                std::lock_guard<std::mutex> lk(s.mu);
                while (static_cast<int>(s.streams.size()) <= w) {
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

    ~Image() {
        if (dev) cudaFree(dev);
        if (plan) g6l_plan_free(plan);
    }

    void open(const std::string& release_dir, const std::string& widths_path, int nr, int r,
              const std::string& layout_path, bool with_mtp) {
        rank = r;
        const std::string sfx = ".manifest.json";
        std::string p = knob::tmpdir() + "/qw38-rank" + std::to_string(r) + ".XXXXXX" + sfx;
        const int fd = mkstemps(&p[0], static_cast<int>(sfx.size()));
        if (fd < 0) die(p + ": " + std::strerror(errno));
        ::close(fd);
        char err[1024] = "";
        plan = g6l_qw38_plan(release_dir.c_str(), widths_path.c_str(),
                             static_cast<uint32_t>(nr), static_cast<uint32_t>(r),
                             with_mtp ? 1 : 0,
                             layout_path.empty() ? nullptr : layout_path.c_str(),
                             p.c_str(), err, sizeof err);
        if (!plan) {
            ::unlink(p.c_str());
            die("rank " + std::to_string(r) + ": g6l_qw38_plan failed: " + err);
        }
        std::string e;
        const bool ok = pack.open_manifest(p, e);
        ::unlink(p.c_str());
        if (!ok) die("rank " + std::to_string(r) + ": open manifest: " + e);
    }

    void whole() {
        void* d = nullptr;
        QW38_CK(cudaMalloc(&d, pack.bytes()));
        Sink sink;
        sink.dev = static_cast<uint8_t*>(d);
        const g6l_sink s{&sink, &Sink::alloc, &Sink::release, &Sink::copy};
        g6l_stats st{};
        char err[1024] = "";
        if (g6l_plan_load(plan, &s, &st, err, sizeof err))
            die("rank " + std::to_string(rank) + ": load: " + err);
        for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
        dev = sink.dev;
        std::fprintf(stderr, "# rank %d: qw38 device image %.2f GB in %.2f s (%u threads)\n",
                     rank, st.image_bytes / 1e9, st.seconds, st.threads);
    }

    const g6::Entry* find(const std::string& name) const { return pack.find(name); }
    const g6::Entry& entry(const std::string& name) const {
        const g6::Entry* e = pack.find(name);
        if (!e) die("rank " + std::to_string(rank) + ": missing tensor " + name);
        return *e;
    }
    template <class T> const T* get(const std::string& name) const {
        return reinterpret_cast<const T*>(dev + entry(name).offset);
    }
    template <class T> const T* opt(const std::string& name) const {
        const g6::Entry* e = pack.find(name);
        return e ? reinterpret_cast<const T*>(dev + e->offset) : nullptr;
    }
    template <class T> void read_host(const std::string& name, std::vector<T>& out) const {
        const g6::Entry& e = entry(name);
        out.resize(e.nbytes / sizeof(T));
        QW38_CK(cudaMemcpy(out.data(), dev + e.offset, e.nbytes, cudaMemcpyDeviceToHost));
    }
};

// ------------------------------------------------------------------------------------------------ Options & Profile
struct Options {
    int nr = 1;
    int rank0 = 0;
    int nloc = 1;          // nr for m4sim, 1 for g6m4
    int slots = 1;
    int max_rows = 512;
    int cap = 4096;
    int nc = 256;
    int mtp = 0;
    std::string layers = "all";
    std::string widths_path;    // defaults to <release>/exlr8-manifest-bit300.json
    std::string layout_path;    // optional bin/g6layout file
    size_t ple_cache_mib = 4096;
    int ple_uring = 128;
    bool ple_direct = true;
    ptab::Mode ptab_mode = ptab::OFF;
    bool wide = true;
    bool dense_fp8 = true;
    bool ple_async = true;
    bool profile = false;
};

struct ProfileAccum {
    double hc_ms = 0, gdn_ms = 0, attn_ms = 0, ple_io_ms = 0, ple_gpu_ms = 0, moe_ms = 0, ar_ms = 0, head_ms = 0;
    int steps = 0;
    int tokens = 0;
    void print(int rank) const {
        if (!steps) return;
        const double tot = hc_ms + gdn_ms + attn_ms + ple_io_ms + ple_gpu_ms + moe_ms + ar_ms + head_ms;
        std::fprintf(stderr,
                     "# qw38 profile r%d (%d steps, %d tok, %.2f ms total, %.2f ms/step): "
                     "hc=%.2f gdn=%.2f attn=%.2f ple_io=%.2f ple_gpu=%.2f moe=%.2f ar=%.2f head=%.2f\n",
                     rank, steps, tokens, tot, tot / steps,
                     hc_ms / steps, gdn_ms / steps, attn_ms / steps, ple_io_ms / steps,
                     ple_gpu_ms / steps, moe_ms / steps, ar_ms / steps, head_ms / steps);
    }
};

// ------------------------------------------------------------------------------------------------ Engine
struct Engine {
    Options o;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = 0;
    std::vector<std::unique_ptr<Image>> img;
    std::vector<int> layer_ids;
    int max_pps = 0;
    int max_blocks = 0;

    // Global weights (rank 0 copy for replicated tensors, per-rank for lm_head)
    const __nv_bfloat16* embed = nullptr;
    qw38::HcWeights final_hc;
    std::vector<const __nv_bfloat16*> lm_head;
    std::vector<qw38::Fp8Mat> fp8_lm_head;
    std::vector<int> v_lo, v_rows;

    // Weight paging pool (GR8_PTAB=1|2) and on-load FP8 dense arena (GR8_QW38_DENSE_FP8=1)
    ptab::Pool ptab_pool;
    qw38::Fp8Arena fp8_arena;

    // Layer-1 PLE NVMe store + host context per slot
    qw38::PleStore ple_store;
    bool has_ple = false;
    qw38::PleWeights ple_w;
    qw38::PleState ple_st;
    qw38::PleScratch ple_sc;
    std::vector<std::array<int32_t, 2>> ple_ctx; // [slots][2] prior tokens (init to TOK_EOS)
    __nv_bfloat16* ple_pinned = nullptr;         // [max_rows * 2560] pinned host staging
    // Persistent background thread to overlap PLE NVMe fetch with Layer-0 GPU compute (GR8_QW38_PLE_ASYNC=1)
    std::thread ple_thr;
    std::mutex ple_mu;
    std::condition_variable ple_cv, ple_done_cv;
    bool ple_stop = false, ple_has_job = false, ple_job_done = true, ple_job_ok = true;
    std::vector<uint64_t> ple_job_ids;

    struct RankLayer {
        qw38::GdnWeights gdn_w;
        qw38::GdnLayerState gdn_st;
        qw38::FullAttnWeights attn_w;
        qw38::FullLayerKv kv;
        qw38::MoeWeights moe_w;
        ptab::Table ptab_tbl;
    };
    struct Layer {
        int l = 0;
        bool is_full = false;
        qw38::HcWeights attn_hc;
        qw38::HcWeights mlp_hc;
        std::vector<RankLayer> rk;
    };
    std::vector<Layer> L;

    // Shared step scratch buffers
    __nv_bfloat16* hyper = nullptr;       // [max_rows, 10240]
    __nv_bfloat16* mixed = nullptr;       // [max_rows, 2560]
    __nv_bfloat16* parts = nullptr;       // [nr, max_rows, 2560] (BF16 partials for TP all-reduce)
    qw38::HcScratch hc_sc;
    qw38::GdnScratch gdn_sc;
    qw38::FullAttnScratch attn_sc;
    qw38::MoeScratch moe_sc;
    qw38::HeadPackScratch head_sc;
    float* head_logits = nullptr;         // [max_rows, max_v_rows] FP32
    uint8_t* head_dev_pkt = nullptr;      // [max_rows * head_row_bytes(nc)]
    std::vector<uint8_t> head_host_parts; // [nr * max_rows * head_row_bytes(nc)]
    int* d_tok = nullptr;                 // [max_rows]
    int* d_slot = nullptr;                // [max_rows]
    int* d_pos = nullptr;                 // [max_rows]

    // Fabric callbacks (set by g6m4_qw38; nullptr on m4sim_qw38 where all ranks are already in `parts`)
    std::function<void(int M, bool skip_head)> step_begin;
    std::function<void(int c, __nv_bfloat16* parts, size_t bytes)> gather_dev;
    std::function<void(int c, uint8_t* host_parts, size_t bytes)> gather_host;
    std::function<void()> step_end;

    ProfileAccum prof;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    ~Engine() {
        if (ple_thr.joinable()) {
            { std::lock_guard<std::mutex> lk(ple_mu); ple_stop = true; ple_has_job = true; }
            ple_cv.notify_one();
            ple_thr.join();
        }
        if (ple_pinned) cudaFreeHost(ple_pinned);
        if (blas) cublasDestroy(blas);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
    }

    int max_pages_per_slot() const { return max_pps; }

    static std::string lname(int l, const char* w) {
        char b[96];
        std::snprintf(b, sizeof b, "L%03d.%s", l, w);
        return b;
    }

    static std::vector<int> parse_layers(const std::string& s) {
        std::vector<int> v;
        if (s == "all" || s.empty()) {
            for (int l = 0; l < geom::L_MAIN; ++l) v.push_back(l);
            return v;
        }
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int l = std::atoi(item.c_str());
            if (l < 0 || l >= geom::L_MAIN) die("--layers: invalid layer " + item);
            v.push_back(l);
        }
        return v;
    }

    void init(const std::string& release_dir, const Options& opt) {
        o = opt;
        if (!geom::rank_count_ok(o.nr)) die("unsupported TP width " + std::to_string(o.nr));
        if (o.widths_path.empty()) {
            const char* env_w = std::getenv("GR8_QW38_WIDTHS");
            if (env_w && *env_w) o.widths_path = env_w;
            else {
                for (const char* cand : {"exlr8-manifest-bit300.json", "exlr8-manifest-bit400.json", "exlr8-manifest-bit200.json"}) {
                    const std::string p = release_dir + "/" + cand;
                    if (std::filesystem::exists(p)) { o.widths_path = p; break; }
                }
            }
            if (o.widths_path.empty()) die("no exlr8-manifest-bit*.json found in " + release_dir);
        }
        if (o.ptab_mode == ptab::OFF) o.ptab_mode = ptab::mode_env();
        if (const char* we = std::getenv("GR8_QW38_WIDE")) o.wide = (std::atoi(we) != 0);
        if (const char* fe = std::getenv("GR8_QW38_DENSE_FP8")) o.dense_fp8 = (std::atoi(fe) != 0);
        if (const char* ae = std::getenv("GR8_QW38_PLE_ASYNC")) o.ple_async = (std::atoi(ae) != 0);
        if (const char* pe = std::getenv("GR8_QW38_PROFILE")) o.profile = (std::atoi(pe) != 0);
        if (const char* cm = std::getenv("GR8_QW38_PLE_CACHE_MIB")) {
            long v = std::atol(cm);
            if (v >= 0) o.ple_cache_mib = size_t(v);
        }

        cublasCreate(&blas);
        QW38_CK(cudaEventCreate(&ev0));
        QW38_CK(cudaEventCreate(&ev1));
        layer_ids = parse_layers(o.layers);
        max_pps = (o.cap + geom::KV_PAGE - 1) / geom::KV_PAGE;
        max_blocks = max_pps * (geom::KV_PAGE / geom::IDX_COMPRESS);

        // 1. Plan and load each local rank's device image
        img.resize(size_t(o.nloc));
        lm_head.resize(size_t(o.nloc));
        v_lo.resize(size_t(o.nloc));
        v_rows.resize(size_t(o.nloc));
        int max_vr = 0;
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            img[size_t(i)] = std::make_unique<Image>();
            img[size_t(i)]->open(release_dir, o.widths_path, o.nr, r, o.layout_path, o.mtp > 0);
            img[size_t(i)]->whole();
            lm_head[size_t(i)] = img[size_t(i)]->get<__nv_bfloat16>("lm_head.weight");
            v_lo[size_t(i)] = geom::vocab_row0(o.nr, r);
            v_rows[size_t(i)] = geom::vocab_rows(o.nr, r);
            max_vr = std::max(max_vr, v_rows[size_t(i)]);
        }

        const Image& im0 = *img[0];
        embed = im0.get<__nv_bfloat16>("embed.weight");
        final_hc.norm = im0.get<__nv_bfloat16>("hc_final.norm");
        final_hc.down = im0.get<__nv_bfloat16>("hc_final.down");
        final_hc.up = im0.get<__nv_bfloat16>("hc_final.up");
        final_hc.inject = nullptr;

        // 2. Initialize Layer-1 PLE NVMe rowstore if layer 1 is active
        for (int l : layer_ids) if (geom::is_ple(l)) has_ple = true;
        ple_ctx.assign(size_t(o.slots), {geom::TOK_EOS, geom::TOK_EOS});
        if (has_ple) {
            ple_w.key_proj = im0.get<__nv_bfloat16>("L001.ple.key_proj");
            ple_w.value_proj = im0.get<__nv_bfloat16>("L001.ple.value_proj");
            ple_w.conv1d = im0.get<__nv_bfloat16>("L001.ple.conv1d");
            ple_w.norm_query = im0.get<__nv_bfloat16>("L001.ple.norm_query");
            ple_w.norm_key = im0.get<__nv_bfloat16>("L001.ple.norm_key");
            ple_w.norm_conv = im0.get<__nv_bfloat16>("L001.ple.norm_conv");

            std::vector<int64_t> mult, offs, vs;
            im0.read_host("L001.ple.layer_multipliers", mult);
            im0.read_host("L001.ple.ngram_heads_offsets", offs);
            im0.read_host("L001.ple.ngram_heads_vocab_sizes", vs);
            if (mult.size() != 3 || offs.size() != 16 || vs.size() != 16)
                die("L001.ple hash tables have unexpected sizes");
            ple_store.set_hash_params(mult.data(), offs.data(), vs.data(), geom::TOK_EOS);

            const std::string ple_bin = release_dir + "/ple/exlr8-ple-L001.bin";
            qw38::PleStoreOpts po;
            po.cache_bytes = o.ple_cache_mib << 20;
            po.io_uring = o.ple_uring;
            po.direct = o.ple_direct;
            std::string perr;
            if (!ple_store.open(ple_bin, po, &perr)) die(perr);
            std::fprintf(stderr, "# rank %d: qw38 PLE opened %s (%llu rows, direct=%d, uring=%d, cache=%zu MiB)\n",
                         o.rank0, ple_bin.c_str(), (unsigned long long)ple_store.total_rows(),
                         int(ple_store.is_direct()), int(ple_store.has_uring()), o.ple_cache_mib);

            ple_st.conv_hist = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::PLE_NHIST * geom::HC_D);
            ple_sc.emb = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::PLE_DIM);
            ple_sc.key = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            ple_sc.query = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            ple_sc.val = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HID);
            ple_sc.gv = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            ple_sc.gv_norm = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            QW38_CK(cudaHostAlloc(&ple_pinned, size_t(o.max_rows) * geom::PLE_DIM * 2, cudaHostAllocDefault));
            if (o.ple_async) {
                ple_thr = std::thread([this]() {
                    for (;;) {
                        std::unique_lock<std::mutex> lk(ple_mu);
                        ple_cv.wait(lk, [&]() { return ple_has_job; });
                        if (ple_stop) return;
                        ple_has_job = false;
                        lk.unlock();
                        const bool ok = ple_store.fetch_rows(ple_job_ids.data(), ple_job_ids.size(),
                                                             reinterpret_cast<uint8_t*>(ple_pinned));
                        lk.lock();
                        ple_job_ok = ok;
                        ple_job_done = true;
                        ple_done_cv.notify_one();
                    }
                });
            }
        }

        // 3. Bind layers + allocate per-layer KV / GDN states + build ptab::Table (GR8_PTAB=1|2)
        L.resize(layer_ids.size());
        size_t ptab_slab_total = 0;
        int ptab_reshuf_total = 0, ptab_reloc_total = 0;
        for (size_t li = 0; li < layer_ids.size(); ++li) {
            const int l = layer_ids[li];
            Layer& y = L[li];
            y.l = l;
            y.is_full = geom::is_full(l);
            y.attn_hc.norm = im0.get<__nv_bfloat16>(lname(l, "attn_hc.norm"));
            y.attn_hc.down = im0.get<__nv_bfloat16>(lname(l, "attn_hc.down"));
            y.attn_hc.up = im0.get<__nv_bfloat16>(lname(l, "attn_hc.up"));
            y.attn_hc.inject = im0.get<__nv_bfloat16>(lname(l, "attn_hc.inject"));
            y.mlp_hc.norm = im0.get<__nv_bfloat16>(lname(l, "mlp_hc.norm"));
            y.mlp_hc.down = im0.get<__nv_bfloat16>(lname(l, "mlp_hc.down"));
            y.mlp_hc.up = im0.get<__nv_bfloat16>(lname(l, "mlp_hc.up"));
            y.mlp_hc.inject = im0.get<__nv_bfloat16>(lname(l, "mlp_hc.inject"));

            y.rk.resize(size_t(o.nloc));
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                const Image& im = *img[size_t(i)];
                RankLayer& rl = y.rk[size_t(i)];

                if (y.is_full) {
                    rl.attn_w.q_proj = im.get<__nv_bfloat16>(lname(l, "attn.q_proj"));
                    rl.attn_w.k_proj = im.get<__nv_bfloat16>(lname(l, "attn.k_proj"));
                    rl.attn_w.v_proj = im.get<__nv_bfloat16>(lname(l, "attn.v_proj"));
                    rl.attn_w.o_proj = im.get<__nv_bfloat16>(lname(l, "attn.o_proj"));
                    rl.attn_w.q_norm = im.get<__nv_bfloat16>(lname(l, "attn.q_norm"));
                    rl.attn_w.k_norm = im.get<__nv_bfloat16>(lname(l, "attn.k_norm"));
                    rl.attn_w.idx_qk_proj = im.get<__nv_bfloat16>(lname(l, "idx.qk_proj"));
                    rl.attn_w.idx_q_norm = im.get<__nv_bfloat16>(lname(l, "idx.q_norm"));
                    rl.attn_w.idx_k_norm = im.get<__nv_bfloat16>(lname(l, "idx.k_norm"));

                    const int nh_kv = geom::nh_kv(o.nr, r);
                    rl.kv.kv_pages = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_pps) * 2 * geom::KV_PAGE * size_t(nh_kv) * geom::HEAD_DIM);
                    rl.kv.idx_pages = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_pps) * 16 * geom::IDX_DIM);
                    rl.kv.idx_raw = dalloc<__nv_bfloat16>(size_t(o.slots) * 4 * geom::IDX_DIM);
                    rl.kv.kv_staging = dalloc<__nv_bfloat16>(size_t(o.slots) * 2 * geom::KV_PAGE * size_t(nh_kv) * geom::HEAD_DIM);
                    rl.kv.idx_staging = dalloc<__nv_bfloat16>(size_t(o.slots) * 16 * geom::IDX_DIM);
                } else {
                    rl.gdn_w.in_proj_qkv = im.get<__nv_bfloat16>(lname(l, "gdn.in_proj_qkv"));
                    rl.gdn_w.conv1d = im.get<__nv_bfloat16>(lname(l, "gdn.conv1d"));
                    rl.gdn_w.in_proj_z = im.get<__nv_bfloat16>(lname(l, "gdn.in_proj_z"));
                    rl.gdn_w.in_proj_a = im.get<__nv_bfloat16>(lname(l, "gdn.in_proj_a"));
                    rl.gdn_w.in_proj_b = im.get<__nv_bfloat16>(lname(l, "gdn.in_proj_b"));
                    rl.gdn_w.A_log = im.get<__nv_bfloat16>(lname(l, "gdn.A_log"));
                    rl.gdn_w.dt_bias = im.get<__nv_bfloat16>(lname(l, "gdn.dt_bias"));
                    rl.gdn_w.norm = im.get<__nv_bfloat16>(lname(l, "gdn.norm"));
                    rl.gdn_w.out_proj = im.get<__nv_bfloat16>(lname(l, "gdn.out_proj"));

                    const int nh_v = geom::gdn_vh(o.nr);
                    const int qkv_loc = geom::gdn_qkv_local(o.nr, r);
                    rl.gdn_st.S = dalloc<float>(size_t(o.slots) * size_t(nh_v) * 128 * 128);
                    rl.gdn_st.conv = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::GDN_NHIST * size_t(qkv_loc));
                }

                // MoE weights (512 pmcgx routed experts + BF16 gated shared expert)
                rl.moe_w.router = im.get<__nv_bfloat16>(lname(l, "router"));
                rl.moe_w.sh_gate = im.get<__nv_bfloat16>(lname(l, "shared.gate_proj"));
                rl.moe_w.sh_up = im.get<__nv_bfloat16>(lname(l, "shared.up_proj"));
                rl.moe_w.sh_down = im.get<__nv_bfloat16>(lname(l, "shared.down_proj"));
                rl.moe_w.sh_exp_gate = im.get<__nv_bfloat16>(lname(l, "shared.gate_w"));
                rl.moe_w.sh_i = static_cast<int>(im.entry(lname(l, "shared.gate_proj")).shape[0]);

                rl.moe_w.is_tr3 = true;
                rl.moe_w.t_fc1 = im.get<uint8_t>(lname(l, "tr3.fc1"));
                rl.moe_w.t_fc2 = im.get<uint8_t>(lname(l, "tr3.fc2"));
                rl.moe_w.t_tab = im.get<long long>(lname(l, "tr3.table"));
                rl.moe_w.t_kmix = im.opt<long long>(lname(l, "tr3.kmix"));
                rl.moe_w.t_sg = im.get<__half>(lname(l, "tr3.sg"));
                rl.moe_w.t_v1b = im.get<__half>(lname(l, "tr3.v1b"));
                rl.moe_w.t_v2 = im.get<__half>(lname(l, "tr3.v2"));
                rl.moe_w.t_v3b = im.get<__half>(lname(l, "tr3.v3b"));
                rl.moe_w.t_rows = static_cast<int>(im.entry(lname(l, "tr3.table")).shape[0]);
                rl.moe_w.t_eb = g6nr::EB;

                if (o.wide) {
                    launch_tr3_widen(const_cast<uint8_t*>(rl.moe_w.t_fc1), rl.moe_w.t_tab, rl.moe_w.t_kmix,
                                     rl.moe_w.t_rows, rl.moe_w.t_eb, stream);
                    launch_tr4_widen(const_cast<uint8_t*>(rl.moe_w.t_fc1), rl.moe_w.t_tab, rl.moe_w.t_kmix,
                                     rl.moe_w.t_rows, rl.moe_w.t_eb, stream);
                    launch_tr3_widen2(const_cast<uint8_t*>(rl.moe_w.t_fc2), rl.moe_w.t_tab,
                                      rl.moe_w.t_rows, rl.moe_w.t_eb, stream);
                    launch_tr4_widen2(const_cast<uint8_t*>(rl.moe_w.t_fc2), rl.moe_w.t_tab,
                                      rl.moe_w.t_rows, rl.moe_w.t_eb, stream);
                    rl.moe_w.fc1_wide = 1;
                    rl.moe_w.fc2_wide = 1;
                }

                if (o.ptab_mode != ptab::OFF) {
                    if (const char* perr = ptab::build(rl.ptab_tbl, rl.moe_w.t_tab, rl.moe_w.t_kmix,
                                                       rl.moe_w.t_fc1, rl.moe_w.t_fc2,
                                                       rl.moe_w.t_rows, rl.moe_w.t_eb,
                                                       o.ptab_mode, ptab_pool, stream)) {
                        die("L" + std::to_string(l) + " ptab::build: " + perr);
                    }
                    rl.moe_w.t_ptab = rl.ptab_tbl.dev;
                    ptab_slab_total += rl.ptab_tbl.slab_bytes;
                    ptab_reshuf_total += rl.ptab_tbl.reshuffles;
                    ptab_reloc_total += rl.ptab_tbl.relocated;
                }
            }
        }
        if (o.ptab_mode != ptab::OFF) {
            std::fprintf(stderr, "# rank %d: GR8_PTAB=%d (%s): %zu layers, %.2f GB slabs (pool %.2f GB), reshuffles=%d, relocated=%d\n",
                         o.rank0, int(o.ptab_mode), o.ptab_mode == ptab::SCATTER ? "SCATTER" : "IMAGE",
                         L.size(), ptab_slab_total / 1e9, ptab_pool.held / 1e9, ptab_reshuf_total, ptab_reloc_total);
        }

        // 3b. On-load block-32 FP8 E4M3 quantization of dense decode projections (GR8_QW38_DENSE_FP8=1)
        if (o.dense_fp8) {
            fp8_lm_head.resize(size_t(o.nloc));
            for (int i = 0; i < o.nloc; ++i) {
                fp8_arena.add(lm_head[size_t(i)], fp8_lm_head[size_t(i)], v_rows[size_t(i)], geom::HID);
            }
            fp8_arena.add(final_hc.down, final_hc.fp8_down, geom::HC_RANK, geom::HC_D);
            fp8_arena.add(final_hc.up, final_hc.fp8_up, geom::HC_D, geom::HC_RANK, true);
            for (auto& y : L) {
                fp8_arena.add(y.attn_hc.down, y.attn_hc.fp8_down, geom::HC_RANK, geom::HC_D);
                fp8_arena.add(y.attn_hc.up, y.attn_hc.fp8_up, geom::HC_D, geom::HC_RANK, true);
                fp8_arena.add(y.mlp_hc.down, y.mlp_hc.fp8_down, geom::HC_RANK, geom::HC_D);
                fp8_arena.add(y.mlp_hc.up, y.mlp_hc.fp8_up, geom::HC_D, geom::HC_RANK, true);
                for (int i = 0; i < o.nloc; ++i) {
                    const int r = o.rank0 + i;
                    auto& rl = y.rk[size_t(i)];
                    if (y.is_full) {
                        const int nh_q = geom::nh_q(o.nr);
                        const int nh_kv = geom::nh_kv(o.nr, r);
                        fp8_arena.add(rl.attn_w.q_proj, rl.attn_w.fp8_q, nh_q * geom::HEAD_DIM * 2, geom::HID);
                        fp8_arena.add(rl.attn_w.k_proj, rl.attn_w.fp8_k, nh_kv * geom::HEAD_DIM, geom::HID);
                        fp8_arena.add(rl.attn_w.v_proj, rl.attn_w.fp8_v, nh_kv * geom::HEAD_DIM, geom::HID);
                        fp8_arena.add(rl.attn_w.idx_qk_proj, rl.attn_w.fp8_idx, geom::IDX_PROJ_OUT, geom::HID);
                        fp8_arena.add(rl.attn_w.o_proj, rl.attn_w.fp8_o, geom::HID, nh_q * geom::HEAD_DIM);
                    } else {
                        const int nh_v = geom::gdn_vh(o.nr);
                        const int qkv_loc = geom::gdn_qkv_local(o.nr, r);
                        fp8_arena.add(rl.gdn_w.in_proj_qkv, rl.gdn_w.fp8_qkv, qkv_loc, geom::HID);
                        fp8_arena.add(rl.gdn_w.in_proj_z, rl.gdn_w.fp8_z, nh_v * geom::GDN_DV, geom::HID);
                        fp8_arena.add(rl.gdn_w.out_proj, rl.gdn_w.fp8_out, geom::HID, nh_v * geom::GDN_DV);
                    }
                    fp8_arena.add(rl.moe_w.sh_gate, rl.moe_w.fp8_sh_gate, rl.moe_w.sh_i, geom::HID);
                    fp8_arena.add(rl.moe_w.sh_up, rl.moe_w.fp8_sh_up, rl.moe_w.sh_i, geom::HID);
                    fp8_arena.add(rl.moe_w.sh_down, rl.moe_w.fp8_sh_down, geom::HID, rl.moe_w.sh_i);
                }
            }
            const auto fp8_t0 = std::chrono::steady_clock::now();
            const size_t fp8_b = fp8_arena.build(stream);
            QW38_CK(cudaStreamSynchronize(stream));
            const double fp8_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fp8_t0).count();
            std::fprintf(stderr, "# rank %d: GR8_QW38_DENSE_FP8=1: quantized dense decode weights (%.2f GB in %.2f s)\n",
                         o.rank0, fp8_b / 1e9, fp8_s);
        }

        // 4. Allocate shared step scratch buffers
        const size_t M = size_t(o.max_rows);
        hyper = dalloc<__nv_bfloat16>(M * geom::HC_D);
        mixed = dalloc<__nv_bfloat16>(M * geom::HID);
        parts = dalloc<__nv_bfloat16>(size_t(o.nr) * M * geom::HID);

        hc_sc.hn = dalloc<__nv_bfloat16>(M * geom::HC_D);
        hc_sc.down = dalloc<__nv_bfloat16>(M * geom::HC_RANK);
        hc_sc.up = dalloc<__nv_bfloat16>(M * geom::HC_D);
        hc_sc.inj = dalloc<float>(M * geom::HC);

        const int max_nh_v = geom::gdn_vh(o.nr);
        const int max_qkv = (2 * geom::gdn_kh_max(o.nr) + max_nh_v) * geom::GDN_DK;
        gdn_sc.qkv_raw = dalloc<__nv_bfloat16>(M * size_t(max_qkv));
        gdn_sc.qkv_conv = dalloc<__nv_bfloat16>(M * size_t(max_qkv));
        gdn_sc.z = dalloc<__nv_bfloat16>(M * size_t(max_nh_v) * geom::GDN_DV);
        gdn_sc.a = dalloc<__nv_bfloat16>(M * size_t(max_nh_v));
        gdn_sc.b = dalloc<__nv_bfloat16>(M * size_t(max_nh_v));
        gdn_sc.core = dalloc<__nv_bfloat16>(M * size_t(max_nh_v) * geom::GDN_DV);

        const int max_nh_q = geom::nh_q(o.nr);
        const int max_nh_kv = geom::nh_kv_max(o.nr);
        attn_sc.q_gate = dalloc<__nv_bfloat16>(M * size_t(max_nh_q) * 512);
        attn_sc.q = dalloc<__nv_bfloat16>(M * size_t(max_nh_q) * 256);
        attn_sc.gate = dalloc<__nv_bfloat16>(M * size_t(max_nh_q) * 256);
        attn_sc.k = dalloc<__nv_bfloat16>(M * size_t(max_nh_kv) * 256);
        attn_sc.v = dalloc<__nv_bfloat16>(M * size_t(max_nh_kv) * 256);
        attn_sc.idx_qk = dalloc<__nv_bfloat16>(M * geom::IDX_PROJ_OUT);
        attn_sc.idx_q = dalloc<__nv_bfloat16>(M * geom::IDX_HEADS * geom::IDX_DIM);
        attn_sc.attn_out = dalloc<__nv_bfloat16>(M * size_t(max_nh_q) * 256);
        attn_sc.idx_scores = dalloc<float>(M * size_t(max_blocks));
        attn_sc.sel_blocks = dalloc<int>(M * geom::IDX_TOP_BLOCKS);
        attn_sc.sel_counts = dalloc<int>(M);
        attn_sc.split_m = dalloc<float>(8 * size_t(max_nh_q) * 24);
        attn_sc.split_l = dalloc<float>(8 * size_t(max_nh_q) * 24);
        attn_sc.split_o = dalloc<float>(8 * size_t(max_nh_q) * 24 * 256);

        moe_sc.router_logits = dalloc<float>(M * geom::NEXP);
        moe_sc.topk_idx = dalloc<int>(M * geom::TOPK);
        moe_sc.topk_w = dalloc<float>(M * geom::TOPK);
        moe_sc.d_n_groups = dalloc<int>(1);
        moe_sc.grp_exp = dalloc<int>(M * geom::TOPK);
        moe_sc.grp_cnt = dalloc<int>(M * geom::TOPK);
        moe_sc.grp_routes = dalloc<int>(M * geom::TOPK * 8);
        moe_sc.fc1_part = dalloc<float>(8 * geom::TOPK * size_t(g6nr::EB) * 2 * 2 * 128);
        moe_sc.fc1_act = dalloc<__half>(M * geom::TOPK * size_t(g6nr::EB) * 128);
        moe_sc.fc2_part = dalloc<float>(M * geom::TOPK * geom::HID);
        moe_sc.routed_acc = dalloc<float>(M * geom::HID);
        moe_sc.sh_gate_buf = dalloc<__nv_bfloat16>(M * geom::SHARED_I);
        moe_sc.sh_up_buf = dalloc<__nv_bfloat16>(M * geom::SHARED_I);
        moe_sc.sh_out = dalloc<__nv_bfloat16>(M * geom::HID);
        moe_sc.sh_scalar_gate = dalloc<float>(M);

        head_logits = dalloc<float>(M * size_t(max_vr));
        head_sc.blk_max = dalloc<float>(M * qw38::HEAD_BLKS);
        head_sc.blk_cnt = dalloc<int>(M * qw38::HEAD_BLKS);
        head_sc.blk_dsum = dalloc<double>(M * qw38::HEAD_BLKS);
        head_sc.row_max = dalloc<float>(M);
        head_sc.row_fin = dalloc<int>(M);
        head_sc.surv_cnt = dalloc<int>(M);
        head_sc.surv_val = dalloc<float>(M * qw38::HEAD_MAX_SURV);
        head_sc.surv_id = dalloc<int>(M * qw38::HEAD_MAX_SURV);
        const size_t hrb = head_row_bytes(o.nc);
        head_dev_pkt = dalloc<uint8_t>(M * hrb);
        head_host_parts.assign(size_t(o.nr) * M * hrb, 0);

        d_tok = dalloc<int>(M);
        d_slot = dalloc<int>(M);
        d_pos = dalloc<int>(M);
    }

    void reset_slot(int slot) {
        if (slot < 0 || slot >= o.slots) return;
        ple_ctx[size_t(slot)] = {geom::TOK_EOS, geom::TOK_EOS};
        if (has_ple) {
            const size_t pb = size_t(geom::PLE_NHIST) * geom::HC_D * 2;
            QW38_CK(cudaMemsetAsync(ple_st.conv_hist + size_t(slot) * geom::PLE_NHIST * geom::HC_D, 0, pb, stream));
        }
        for (auto& y : L) {
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                auto& rl = y.rk[size_t(i)];
                if (y.is_full) {
                    QW38_CK(cudaMemsetAsync(rl.kv.idx_raw + size_t(slot) * 4 * geom::IDX_DIM, 0, 4 * geom::IDX_DIM * 2, stream));
                } else {
                    const size_t sb = size_t(geom::gdn_vh(o.nr)) * 128 * 128 * 4;
                    const size_t cb = size_t(geom::GDN_NHIST) * size_t(geom::gdn_qkv_local(o.nr, r)) * 2;
                    QW38_CK(cudaMemsetAsync(rl.gdn_st.S + size_t(slot) * size_t(geom::gdn_vh(o.nr)) * 128 * 128, 0, sb, stream));
                    QW38_CK(cudaMemsetAsync(rl.gdn_st.conv + size_t(slot) * geom::GDN_NHIST * size_t(geom::gdn_qkv_local(o.nr, r)), 0, cb, stream));
                }
            }
        }
    }

    // Forward one step over `segs` (`M` total rows), optionally skipping lm_head (prefill chunks).
    void forward(const std::vector<qw38::StepSeg>& segs, const int* toks, const RowSample* samples,
                 std::vector<RowOut>& out, bool skip_head = false) {
        int M = 0;
        for (const auto& g : segs) M += g.len;
        if (M <= 0 || M > o.max_rows) die("forward: bad M=" + std::to_string(M));

        std::vector<int> h_slot(static_cast<size_t>(M)), h_pos(static_cast<size_t>(M));
        for (const auto& g : segs) {
            for (int t = 0; t < g.len; ++t) {
                h_slot[size_t(g.row0 + t)] = g.slot;
                h_pos[size_t(g.row0 + t)] = g.pos0 + t;
            }
        }
        QW38_CK(cudaMemcpyAsync(d_tok, toks, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_slot, h_slot.data(), size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_pos, h_pos.data(), size_t(M) * 4, cudaMemcpyHostToDevice, stream));

        // 0a. Notify multi-collective fabric step start (single enqueue_step for all 2*L+1 collectives on TP>=2)
        if (step_begin) step_begin(M, skip_head);

        // 0. Kick off async PLE NVMe fetch BEFORE Layer 0 GPU compute so it overlaps completely behind Layer 0!
        bool ple_in_flight = false;
        if (has_ple && o.ple_async && ple_thr.joinable()) {
            ple_job_ids.resize(size_t(M) * geom::PLE_HEADS);
            for (const auto& g : segs) {
                if (g.len <= 0) continue;
                if (g.pos0 == 0) ple_ctx[size_t(g.slot)] = {geom::TOK_EOS, geom::TOK_EOS};
                ple_store.compute_ids(toks + g.row0, g.len, ple_ctx[size_t(g.slot)].data(), true,
                                      ple_job_ids.data() + size_t(g.row0) * geom::PLE_HEADS);
            }
            {
                std::lock_guard<std::mutex> lk(ple_mu);
                ple_job_done = false;
                ple_has_job = true;
            }
            ple_cv.notify_one();
            ple_in_flight = true;
        }

        // 0b. Token embed -> 4-stream hyper state
        qw38::k_embed_hc4<<<M, 256, 0, stream>>>(d_tok, embed, hyper, M);

        auto t_start = [&]() { if (o.profile) cudaEventRecord(ev0, stream); };
        auto t_end = [&](double& acc) {
            if (!o.profile) return;
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            acc += ms;
        };

        const size_t part_stride = size_t(M) * geom::HID;
        int coll = 0;

        for (size_t li = 0; li < L.size(); ++li) {
            Layer& y = L[li];

            // Layer-1 PLE before attn_hyper_connection
            if (geom::is_ple(y.l) && has_ple) {
                const auto p0 = std::chrono::steady_clock::now();
                if (ple_in_flight) {
                    std::unique_lock<std::mutex> lk(ple_mu);
                    ple_done_cv.wait(lk, [&]() { return ple_job_done; });
                    if (!ple_job_ok) die("PLE async fetch_rows failed");
                    ple_in_flight = false;
                } else {
                    std::vector<uint64_t> all_ids(size_t(M) * geom::PLE_HEADS);
                    for (const auto& g : segs) {
                        if (g.len <= 0) continue;
                        if (g.pos0 == 0) ple_ctx[size_t(g.slot)] = {geom::TOK_EOS, geom::TOK_EOS};
                        ple_store.compute_ids(toks + g.row0, g.len, ple_ctx[size_t(g.slot)].data(), true,
                                              all_ids.data() + size_t(g.row0) * geom::PLE_HEADS);
                    }
                    if (!ple_store.fetch_rows(all_ids.data(), all_ids.size(), reinterpret_cast<uint8_t*>(ple_pinned)))
                        die("PLE fetch_rows failed");
                }
                const auto p1 = std::chrono::steady_clock::now();
                if (o.profile) prof.ple_io_ms += std::chrono::duration<double, std::milli>(p1 - p0).count();

                t_start();
                QW38_CK(cudaMemcpyAsync(ple_sc.emb, ple_pinned, size_t(M) * geom::PLE_DIM * 2, cudaMemcpyHostToDevice, stream));
                qw38::ple_forward(blas, ple_w, ple_st, ple_sc, hyper, segs.data(), int(segs.size()), M, true, stream);
                t_end(prof.ple_gpu_ms);
            }

            // Attention / GDN hyper-connection prepare
            t_start();
            qw38::hc_prepare(blas, y.attn_hc, hyper, mixed, hc_sc, M, stream);
            t_end(prof.hc_ms);

            // Attention or GDN per local rank
            t_start();
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                RankLayer& rl = y.rk[size_t(i)];
                __nv_bfloat16* p_dst = parts + size_t(r) * part_stride;
                if (y.is_full) {
                    qw38::full_attn_forward(blas, rl.attn_w, rl.kv, attn_sc, mixed, p_dst,
                                            d_slot, d_pos, segs.data(), int(segs.size()), M, o.nr, r,
                                            max_pps, max_blocks, stream);
                } else {
                    qw38::gdn_forward(blas, rl.gdn_w, rl.gdn_st, gdn_sc, mixed, p_dst,
                                      segs.data(), int(segs.size()), M, o.nr, r, true, stream, d_slot, d_pos);
                }
            }
            t_end(y.is_full ? prof.attn_ms : prof.gdn_ms);

            // Collective 1: exchange partials across ranks (no-op for TP=1 or m4sim)
            t_start();
            if (o.nr > 1 && gather_dev) gather_dev(coll++, parts, part_stride * 2);
            else ++coll;
            t_end(prof.ar_ms);

            // Sum rank-ordered partials + HC inject
            t_start();
            qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            t_end(prof.hc_ms);

            // MLP hyper-connection prepare
            t_start();
            qw38::hc_prepare(blas, y.mlp_hc, hyper, mixed, hc_sc, M, stream);
            t_end(prof.hc_ms);

            // MoE per local rank
            t_start();
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                RankLayer& rl = y.rk[size_t(i)];
                __nv_bfloat16* p_dst = parts + size_t(r) * part_stride;
                qw38::moe_forward(blas, rl.moe_w, moe_sc, mixed, p_dst, M, stream);
            }
            t_end(prof.moe_ms);

            // Collective 2: exchange MoE partials across ranks
            t_start();
            if (o.nr > 1 && gather_dev) gather_dev(coll++, parts, part_stride * 2);
            else ++coll;
            t_end(prof.ar_ms);

            // Sum rank-ordered partials + HC inject
            t_start();
            qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            t_end(prof.hc_ms);
        }

        if (skip_head) {
            if (step_end) step_end();
            if (o.profile) { ++prof.steps; prof.tokens += M; }
            return;
        }

        // Final HC mixer -> lm_head -> k_ink2_head_pack -> host merge & sample
        t_start();
        qw38::hc_prepare(blas, final_hc, hyper, mixed, hc_sc, M, stream);
        const size_t hrb = head_row_bytes(o.nc);
        const size_t rank_pkt_bytes = size_t(M) * hrb;
        cublasSetStream(blas, stream);
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            if (M <= 8) {
                if (i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q) {
                    qw38::launch_fp8_gemv_smallm<true>(mixed, fp8_lm_head[size_t(i)], head_logits, M, stream);
                } else {
                    qw38::launch_bf16_gemv_smallm<true>(mixed, lm_head[size_t(i)], head_logits, M, v_rows[size_t(i)], geom::HID, stream);
                }
            } else {
                qw38::gemm_bf16_nt_f32(blas, M, v_rows[size_t(i)], geom::HID, mixed, lm_head[size_t(i)], head_logits);
            }
            qw38::head_pack_fast(head_logits, M, v_rows[size_t(i)], v_lo[size_t(i)], o.nc, head_sc, head_dev_pkt, hrb, stream);
            QW38_CK(cudaMemcpyAsync(head_host_parts.data() + size_t(r) * rank_pkt_bytes,
                                    head_dev_pkt, rank_pkt_bytes, cudaMemcpyDeviceToHost, stream));
        }
        QW38_CK(cudaStreamSynchronize(stream));
        if (o.nr > 1 && gather_host) gather_host(coll++, head_host_parts.data(), rank_pkt_bytes);
        if (step_end) step_end();

        out.resize(size_t(M));
        for (int m = 0; m < M; ++m) {
            // Repack rank shares for row m contiguously for head_merge
            std::vector<uint8_t> row_parts(size_t(o.nr) * hrb);
            for (int r = 0; r < o.nr; ++r) {
                std::memcpy(row_parts.data() + size_t(r) * hrb,
                            head_host_parts.data() + size_t(r) * rank_pkt_bytes + size_t(m) * hrb, hrb);
            }
            out[size_t(m)] = head_merge(row_parts.data(), hrb, o.nr, o.nc);
            if (samples) out[size_t(m)].tok = sample_row(out[size_t(m)].top, samples[m]);
            else if (!out[size_t(m)].top.empty()) out[size_t(m)].tok = out[size_t(m)].top[0].id;
        }
        t_end(prof.head_ms);
        if (o.profile) { ++prof.steps; prof.tokens += M; }
    }
};

} // namespace qw38rt
