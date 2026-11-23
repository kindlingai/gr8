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
#include "../engram/qw38_ple_trellis.cuh"
#include "../kernels/qw38_moe.cuh"
#include "../kernels/qw38_mtp.cuh"
#include "../kernels/qw38_head.cuh"
#include "../kernels/ink2_head.cuh"
#include "../kernels/ptab.cuh"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"
#include "../knobs-rs/gr8_knobs.h"
#include "qw38_vision.cuh"

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
#include <map>
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

// Compute post-image isotropic 3D MRoPE offset (sum over images of max(llm_h, llm_w) - span).
// Matches transformers Qwen4ExpModel.get_rope_index verbatim; 0 when there are no images.
inline int qw38_mrope_delta(const std::vector<g6rt::ImageIn>& images) {
    int delta = 0;
    for (const auto& im : images) {
        const int lh = im.vit_h / qw38v::MERGE;
        const int lw = im.vit_w / qw38v::MERGE;
        delta += std::max(lh, lw) - im.span;
    }
    return delta;
}

// Compute exact 3D MRoPE (t, h, w) coordinates for prompt rows [first, first + n).
inline void qw38_build_mrope(const std::vector<g6rt::ImageIn>& images,
                             int first, int n, int3* out_thw) {
    int img_idx = 0;
    const int n_img = static_cast<int>(images.size());
    int delta = 0;
    while (img_idx < n_img && images[size_t(img_idx)].pos + images[size_t(img_idx)].span <= first) {
        const auto& im = images[size_t(img_idx)];
        const int lh = im.vit_h / qw38v::MERGE;
        const int lw = im.vit_w / qw38v::MERGE;
        delta += std::max(lh, lw) - im.span;
        ++img_idx;
    }
    for (int i = 0; i < n; ++i) {
        const int p = first + i;
        while (img_idx < n_img && images[size_t(img_idx)].pos + images[size_t(img_idx)].span <= p) {
            const auto& im = images[size_t(img_idx)];
            const int lh = im.vit_h / qw38v::MERGE;
            const int lw = im.vit_w / qw38v::MERGE;
            delta += std::max(lh, lw) - im.span;
            ++img_idx;
        }
        if (img_idx < n_img && p >= images[size_t(img_idx)].pos) {
            const auto& im = images[size_t(img_idx)];
            const int lw = im.vit_w / qw38v::MERGE;
            const int rel = p - im.pos;
            const int r = rel / lw;
            const int c = rel % lw;
            const int base = im.pos + delta;
            out_thw[i] = make_int3(base, base + r, base + c);
        } else {
            const int cp = p + delta;
            out_thw[i] = make_int3(cp, cp, cp);
        }
    }
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
    std::string ple_path;       // optional PLE table path (.bin or exlr8 row-trellis .safetensors)
    size_t ple_cache_mib = 4096;
    int ple_uring = 128;
    bool ple_direct = true;
    ptab::Mode ptab_mode = ptab::OFF;
    bool wide = true;
    bool dense_fp8 = true;
    bool ple_async = true;
    bool use_graph = true;
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
    void* cublas_ws = nullptr;
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
    __nv_bfloat16* ple_pinned[2] = {nullptr, nullptr}; // double-buffered [max_rows * 2560] pinned host staging (raw BF16 mode)
    uint8_t* ple_codes_pinned[2] = {nullptr, nullptr}; // double-buffered [max_rows * 16 * row_bytes] pinned staging (row-trellis mode)
    uint16_t* ple_svh_pinned[2] = {nullptr, nullptr};  // double-buffered [max_rows * 16] pinned row scales (row-trellis mode)
    uint8_t* d_ple_codes = nullptr;                    // device staging for compressed row-trellis rows
    __half* d_ple_svh = nullptr;                       // device staging for row scales
    __half* d_ple_suh = nullptr;                       // device column scales [nc] (uploaded once at init)
    cudaEvent_t ple_h2d_done[2] = {nullptr, nullptr};
    bool ple_h2d_pending[2] = {false, false};
    int ple_active_buf = 0;
    // Persistent background thread to overlap PLE NVMe fetch with GPU compute (GR8_QW38_PLE_ASYNC=1)
    std::thread ple_thr;
    std::mutex ple_mu;
    std::condition_variable ple_cv, ple_done_cv;
    bool ple_stop = false, ple_has_job = false, ple_job_done = true, ple_job_ok = true;
    int ple_job_buf = 0;
    bool ple_job_is_prefetch = false;
    std::vector<uint64_t> ple_job_ids;
    std::vector<uint64_t> ple_next_pf_ids;
    // Cross-chunk async prefetch tracking (never mutates ple_ctx until consumed by forward())
    bool ple_pf_valid = false;
    int ple_pf_buf = 0, ple_pf_slot = -1, ple_pf_pos0 = -1, ple_pf_len = 0;
    uint64_t ple_pf_hash = 0;
    std::array<int32_t, 2> ple_pf_ctx_in{geom::TOK_EOS, geom::TOK_EOS};
    std::array<int32_t, 2> ple_pf_ctx_out{geom::TOK_EOS, geom::TOK_EOS};

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
    uint8_t* head_host_parts = nullptr;   // [nr * max_rows * head_row_bytes(nc)] pinned host staging (graph-safe D2H!)
    int* h_meta_pinned = nullptr;         // [3 * max_rows] pinned host staging for tok/slot/pos
    int3* h_rope_pinned = nullptr;        // [max_rows] pinned host staging for 3D MRoPE (t, h, w)
    int* d_tok = nullptr;                 // [max_rows]
    int* d_slot = nullptr;                // [max_rows]
    int* d_pos = nullptr;                 // [max_rows]
    int3* d_rope = nullptr;               // [max_rows]
    std::vector<int> slot_mrope_delta;    // [slots] post-image isotropic MRoPE shift (0 for text-only)

    // Optional vision tower (loaded when GR8_QW38_VISION=1 and dense/vision-bf16.safetensors exists)
    Qw38Vision vision;
    void load_vision(const std::string& path) { vision.load(path, o.max_rows); }
    bool vision_ready() const { return vision.ready(); }
    const std::string& vision_id() const { return vision.id(); }
    size_t vision_device_bytes() const { return vision.device_bytes(); }
    void set_slot_mrope_delta(int slot, int delta) {
        if (slot >= 0 && slot < o.slots) slot_mrope_delta[size_t(slot)] = delta;
    }
    void set_slot_idx_raw_thw(int slot, int3 thw) {
        if (slot < 0 || slot >= o.slots) return;
        for (auto& y : L) {
            if (!y.is_full) continue;
            for (int i = 0; i < o.nloc; ++i)
                QW38_CK(cudaMemcpyAsync(y.rk[size_t(i)].kv.idx_raw_thw + slot, &thw, sizeof(int3), cudaMemcpyHostToDevice, stream));
        }
    }

    // Optional Layer-48 MTP drafter state (loaded when o.mtp > 0)
    bool has_mtp = false;
    qw38::MtpWeights mtp_w;
    qw38::MtpScratch mtp_sc;
    __nv_bfloat16* last_hyper = nullptr;         // [slots, 10240] BF16 (trunk final hyper state per slot)
    __nv_bfloat16* mtp_hid_in = nullptr;         // [slots, 10240] BF16 (gathered initial hyper for arbitrary S2 slot subsets)
    __nv_bfloat16* mtp_hid_cur = nullptr;        // [slots, 10240] BF16 (chained MTP draft hidden state)

    // Fabric callbacks (set by g6m4_qw38; nullptr on m4sim_qw38 where all ranks are already in `parts`)
    std::function<void(int M, bool skip_head, int n_layers_override, int head_M)> step_begin;
    std::function<void(int c, __nv_bfloat16* parts, size_t bytes)> gather_dev;
    std::function<void(int c, __nv_bfloat16* hyper, const __nv_bfloat16* parts, const float* inj, int M)> gather_inject;
    std::function<void(int c, uint8_t* host_parts, size_t bytes)> gather_host;
    // Optional zero-copy direct-fabric AR callbacks (g6m4_qw38 fast AR): write directly into f.send (M > 8),
    // or write to GPU HBM `parts` + coalesced copy to f.send (M <= 8 decode) and inject directly from own + __ldcv(f.recv).
    std::function<__nv_bfloat16*(int c)> direct_send_dst;
    std::function<void(int c, size_t bytes)> direct_ar_wait;
    std::function<void(int c, const __nv_bfloat16* own_part, size_t bytes)> direct_copy_ring_wait;
    std::function<void(int c, __nv_bfloat16* hyper, const float* inj, int M, const __nv_bfloat16* own_override)> direct_hc_inject;
    std::function<void(int c, __nv_bfloat16* hyper, const __nv_bfloat16* w_norm, const __nv_bfloat16* w_inj,
                       __nv_bfloat16* hn_out, float* inj_io, int M)> direct_hc_inject_norm;
    // Optional sequence-parallel HC callbacks (multi-rank prefill): RS(M/nr) -> local HC(M/nr) -> AG(M/nr).
    std::function<void(int M, int n_layers, int slice_rows)> step_begin_sp;
    std::function<void(int c, __nv_bfloat16* hyper_loc, const __nv_bfloat16* w_norm, const __nv_bfloat16* w_inj,
                       __nv_bfloat16* hn_loc, float* inj_loc, int r0, int m_loc)> direct_hc_rs_inject_norm;
    std::function<void(int c, __nv_bfloat16* mixed, int M, int slice_rows)> direct_ag_assemble;
    std::function<void()> step_end;
    std::function<int()> step_parity;     // returns 0 or 1 for double-buffered multi-box graph capture

    // CUDA Graph cache for decode & MTP-verify steps (100% compatible with GR8_PTAB=1|2 & FAST_AR!)
    struct GraphKey {
        int M, need_qsa, ns_attn, par, snap_k;
        bool operator<(const GraphKey& b) const {
            if (M != b.M) return M < b.M;
            if (need_qsa != b.need_qsa) return need_qsa < b.need_qsa;
            if (ns_attn != b.ns_attn) return ns_attn < b.ns_attn;
            if (par != b.par) return par < b.par;
            return snap_k < b.snap_k;
        }
    };
    struct GraphPair {
        cudaGraphExec_t exec_a = nullptr; // embed + Layer 0 (overlapped with PLE NVMe fetch)
        cudaGraphExec_t exec_b = nullptr; // L1 PLE + Layers 1..47 + Final HC + lm_head + head_pack + D2H
    };
    std::map<GraphKey, GraphPair> graphs;

    struct MtpGraphKey {
        int M, need_qsa, ns_attn, par, hin_kind; // hin_kind: 0 = last_hyper, 1 = mtp_hid_cur
        bool operator<(const MtpGraphKey& b) const {
            if (M != b.M) return M < b.M;
            if (need_qsa != b.need_qsa) return need_qsa < b.need_qsa;
            if (ns_attn != b.ns_attn) return ns_attn < b.ns_attn;
            if (par != b.par) return par < b.par;
            return hin_kind < b.hin_kind;
        }
    };
    std::map<MtpGraphKey, cudaGraphExec_t> mtp_graphs;

    ProfileAccum prof;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    ~Engine() {
        for (auto& kv : graphs) {
            if (kv.second.exec_a) cudaGraphExecDestroy(kv.second.exec_a);
            if (kv.second.exec_b) cudaGraphExecDestroy(kv.second.exec_b);
        }
        graphs.clear();
        for (auto& kv : mtp_graphs) {
            if (kv.second) cudaGraphExecDestroy(kv.second);
        }
        mtp_graphs.clear();
        if (ple_thr.joinable()) {
            {
                std::unique_lock<std::mutex> lk(ple_mu);
                ple_done_cv.wait(lk, [&]() { return ple_job_done; });
                ple_stop = true;
                ple_has_job = true;
            }
            ple_cv.notify_one();
            ple_thr.join();
        }
        for (int b = 0; b < 2; ++b) {
            if (ple_h2d_done[b]) cudaEventDestroy(ple_h2d_done[b]);
            if (ple_pinned[b]) cudaFreeHost(ple_pinned[b]);
            if (ple_codes_pinned[b]) cudaFreeHost(ple_codes_pinned[b]);
            if (ple_svh_pinned[b]) cudaFreeHost(ple_svh_pinned[b]);
        }
        if (d_ple_codes) cudaFree(d_ple_codes);
        if (d_ple_svh) cudaFree(d_ple_svh);
        if (d_ple_suh) cudaFree(d_ple_suh);
        if (head_host_parts) cudaFreeHost(head_host_parts);
        if (h_meta_pinned) cudaFreeHost(h_meta_pinned);
        if (h_rope_pinned) cudaFreeHost(h_rope_pinned);
        if (blas) cublasDestroy(blas);
        if (cublas_ws) cudaFree(cublas_ws);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        if (stream) cudaStreamDestroy(stream);
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
            const std::optional<std::string>& env_w = knob::qw38_widths();
            if (env_w && !env_w->empty()) o.widths_path = *env_w;
            else {
                for (const char* cand : {"manifests/exlr8-manifest-bit300.json", "manifests/exlr8-manifest-bit400.json",
                                         "manifests/exlr8-manifest-bit200.json", "exlr8-manifest-bit300.json",
                                         "exlr8-manifest-bit400.json", "exlr8-manifest-bit200.json"}) {
                    const std::string p = release_dir + "/" + cand;
                    if (std::filesystem::exists(p)) { o.widths_path = p; break; }
                }
            }
            if (o.widths_path.empty()) die("no exlr8-manifest-bit*.json found in " + release_dir);
        }
        if (o.ptab_mode == ptab::OFF) o.ptab_mode = ptab::mode_env();
        o.wide = knob::qw38_wide();
        o.dense_fp8 = knob::qw38_dense_fp8();
        o.ple_async = knob::qw38_ple_async();
        o.use_graph = knob::qw38_graph();
        o.profile = knob::qw38_profile();
        if (knob::qw38_ple_cache_mib() >= 0) o.ple_cache_mib = size_t(knob::qw38_ple_cache_mib());
        if (o.ple_path.empty()) {
            const std::optional<std::string>& env_ple = knob::qw38_ple_path();
            if (env_ple && !env_ple->empty()) o.ple_path = *env_ple;
        }

        QW38_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cublasCreate(&blas);
        {
            constexpr size_t CUBLAS_WS_BYTES = 32ull << 20;
            QW38_CK(cudaMalloc(&cublas_ws, CUBLAS_WS_BYTES));
            cublasSetWorkspace(blas, cublas_ws, CUBLAS_WS_BYTES);
        }
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

            const std::string ple_bin = qw38::resolve_ple_path(release_dir, o.ple_path);
            qw38::PleStoreOpts po;
            po.cache_bytes = o.ple_cache_mib << 20;
            po.io_uring = o.ple_uring;
            po.direct = o.ple_direct;
            std::string perr;
            if (!ple_store.open(ple_bin, po, &perr)) die(perr);
            if (ple_store.is_row_trellis()) {
                const auto& rt = ple_store.trellis_info();
                if (rt.nc < geom::PLE_HEAD_DIM)
                    die("L001.ple row-trellis nc=" + std::to_string(rt.nc) + " < PLE_HEAD_DIM");
                std::fprintf(stderr,
                             "# rank %d: qw38 PLE opened %s (row-trellis %s %s, %llu rows x %d cols, %zu B/row, "
                             "resident=%d, direct=%d, uring=%d, cache=%zu MiB)\n",
                             o.rank0, ple_bin.c_str(), rt.codebook.c_str(), rt.width_name.c_str(),
                             (unsigned long long)ple_store.total_rows(), rt.nc, rt.row_bytes,
                             int(ple_store.is_resident()), int(ple_store.is_direct()), int(ple_store.has_uring()),
                             o.ple_cache_mib);
            } else {
                std::fprintf(stderr, "# rank %d: qw38 PLE opened %s (%llu rows, direct=%d, uring=%d, cache=%zu MiB)\n",
                             o.rank0, ple_bin.c_str(), (unsigned long long)ple_store.total_rows(),
                             int(ple_store.is_direct()), int(ple_store.has_uring()), o.ple_cache_mib);
            }

            ple_st.conv_hist = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::PLE_NHIST * geom::HC_D);
            if (o.mtp > 0)
                ple_st.conv_hist_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(o.mtp) * geom::PLE_NHIST * geom::HC_D);
            ple_sc.emb = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::PLE_DIM);
            ple_sc.key = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            ple_sc.query = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            ple_sc.val = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HID);
            ple_sc.gv = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            ple_sc.gv_norm = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            if (ple_store.is_row_trellis()) {
                const size_t max_ple_rows = size_t(o.max_rows) * geom::PLE_HEADS;
                const size_t rb = ple_store.row_bytes();
                for (int b = 0; b < 2; ++b) {
                    QW38_CK(cudaHostAlloc(&ple_codes_pinned[b], max_ple_rows * rb, cudaHostAllocDefault));
                    QW38_CK(cudaHostAlloc(&ple_svh_pinned[b], max_ple_rows * sizeof(uint16_t), cudaHostAllocDefault));
                }
                d_ple_codes = dalloc<uint8_t>(max_ple_rows * rb);
                d_ple_svh = dalloc<__half>(max_ple_rows);
                d_ple_suh = dalloc<__half>(ple_store.suh().size());
                QW38_CK(cudaMemcpy(d_ple_suh, ple_store.suh().data(), ple_store.suh().size() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice));
            } else {
                for (int b = 0; b < 2; ++b) {
                    QW38_CK(cudaHostAlloc(&ple_pinned[b], size_t(o.max_rows) * geom::PLE_DIM * 2, cudaHostAllocDefault));
                }
            }
            for (int b = 0; b < 2; ++b) {
                QW38_CK(cudaEventCreateWithFlags(&ple_h2d_done[b], cudaEventDisableTiming));
                ple_h2d_pending[b] = false;
            }
            if (o.ple_async) {
                ple_thr = std::thread([this]() {
                    for (;;) {
                        std::unique_lock<std::mutex> lk(ple_mu);
                        ple_cv.wait(lk, [&]() { return ple_has_job; });
                        if (ple_stop) return;
                        ple_has_job = false;
                        const int buf = ple_job_buf;
                        const bool is_pf = ple_job_is_prefetch;
                        lk.unlock();
                        const bool ok = is_pf
                            ? ple_store.prefetch_rows(ple_job_ids.data(), ple_job_ids.size())
                            : (ple_store.is_row_trellis()
                                   ? ple_store.fetch_trellis_rows(ple_job_ids.data(), ple_job_ids.size(),
                                                                  ple_codes_pinned[buf], ple_svh_pinned[buf])
                                   : ple_store.fetch_rows(ple_job_ids.data(), ple_job_ids.size(),
                                                          reinterpret_cast<uint8_t*>(ple_pinned[buf])));
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
                    rl.kv.idx_raw_thw = dalloc<int3>(size_t(o.slots));
                    if (o.mtp > 0) {
                        rl.kv.idx_raw_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(o.mtp) * 4 * geom::IDX_DIM);
                        rl.kv.idx_raw_thw_snap = dalloc<int3>(size_t(o.slots) * size_t(o.mtp));
                    }
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
                    if (o.mtp > 0) {
                        rl.gdn_st.S_snap = dalloc<float>(size_t(o.slots) * size_t(o.mtp) * size_t(nh_v) * 128 * 128);
                        rl.gdn_st.conv_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(o.mtp) * geom::GDN_NHIST * size_t(qkv_loc));
                    }
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

        // 3a. Bind optional Layer-48 MTP drafter weights & KV cache (when --mtp > 0)
        if (o.mtp > 0 && im0.find("MTP.fc_embedding") != nullptr) {
            has_mtp = true;
            mtp_w.pre_fc_norm_emb = im0.get<__nv_bfloat16>("MTP.pre_fc_norm_embedding");
            mtp_w.pre_fc_norm_hid = im0.get<__nv_bfloat16>("MTP.pre_fc_norm_hidden");
            mtp_w.fc_emb = im0.get<__nv_bfloat16>("MTP.fc_embedding");
            mtp_w.fc_hid = im0.get<__nv_bfloat16>("MTP.fc_hidden");
            mtp_w.final_hc.norm = im0.get<__nv_bfloat16>("MTP.hc_final.norm");
            mtp_w.final_hc.down = im0.get<__nv_bfloat16>("MTP.hc_final.down");
            mtp_w.final_hc.up = im0.get<__nv_bfloat16>("MTP.hc_final.up");
            mtp_w.attn_hc.norm = im0.get<__nv_bfloat16>("L048.attn_hc.norm");
            mtp_w.attn_hc.down = im0.get<__nv_bfloat16>("L048.attn_hc.down");
            mtp_w.attn_hc.up = im0.get<__nv_bfloat16>("L048.attn_hc.up");
            mtp_w.attn_hc.inject = im0.get<__nv_bfloat16>("L048.attn_hc.inject");
            mtp_w.mlp_hc.norm = im0.get<__nv_bfloat16>("L048.mlp_hc.norm");
            mtp_w.mlp_hc.down = im0.get<__nv_bfloat16>("L048.mlp_hc.down");
            mtp_w.mlp_hc.up = im0.get<__nv_bfloat16>("L048.mlp_hc.up");
            mtp_w.mlp_hc.inject = im0.get<__nv_bfloat16>("L048.mlp_hc.inject");

            mtp_w.attn.resize(size_t(o.nloc));
            mtp_w.kv.resize(size_t(o.nloc));
            mtp_w.moe.resize(size_t(o.nloc));
            mtp_w.fp8_exp_gu.resize(size_t(o.nloc));
            mtp_w.fp8_exp_dn.resize(size_t(o.nloc));
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                const Image& im = *img[size_t(i)];
                auto& aw = mtp_w.attn[size_t(i)];
                aw.q_proj = im.get<__nv_bfloat16>("L048.attn.q_proj");
                aw.k_proj = im.get<__nv_bfloat16>("L048.attn.k_proj");
                aw.v_proj = im.get<__nv_bfloat16>("L048.attn.v_proj");
                aw.o_proj = im.get<__nv_bfloat16>("L048.attn.o_proj");
                aw.q_norm = im.get<__nv_bfloat16>("L048.attn.q_norm");
                aw.k_norm = im.get<__nv_bfloat16>("L048.attn.k_norm");
                aw.idx_qk_proj = im.get<__nv_bfloat16>("L048.idx.qk_proj");
                aw.idx_q_norm = im.get<__nv_bfloat16>("L048.idx.q_norm");
                aw.idx_k_norm = im.get<__nv_bfloat16>("L048.idx.k_norm");

                const int nh_kv = geom::nh_kv(o.nr, r);
                auto& kv = mtp_w.kv[size_t(i)];
                kv.kv_pages = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_pps) * 2 * geom::KV_PAGE * size_t(nh_kv) * geom::HEAD_DIM);
                kv.idx_pages = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_pps) * 16 * geom::IDX_DIM);
                kv.idx_raw = dalloc<__nv_bfloat16>(size_t(o.slots) * 4 * geom::IDX_DIM);
                kv.idx_raw_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * 4 * geom::IDX_DIM);
                kv.idx_raw_thw = dalloc<int3>(size_t(o.slots));
                kv.idx_raw_thw_snap = dalloc<int3>(size_t(o.slots));
                kv.kv_staging = dalloc<__nv_bfloat16>(size_t(o.slots) * 2 * geom::KV_PAGE * size_t(nh_kv) * geom::HEAD_DIM);
                kv.idx_staging = dalloc<__nv_bfloat16>(size_t(o.slots) * 16 * geom::IDX_DIM);

                auto& mw = mtp_w.moe[size_t(i)];
                mw.is_tr3 = false;
                mw.router = im.get<__nv_bfloat16>("L048.router");
                mw.sh_gate = im.get<__nv_bfloat16>("L048.shared.gate_proj");
                mw.sh_up = im.get<__nv_bfloat16>("L048.shared.up_proj");
                mw.sh_down = im.get<__nv_bfloat16>("L048.shared.down_proj");
                mw.sh_exp_gate = im.get<__nv_bfloat16>("L048.shared.gate_w");
                mw.sh_i = static_cast<int>(im.entry("L048.shared.gate_proj").shape[0]);
                mw.mtp_gate_up = im.get<__nv_bfloat16>("L048.mtp_exp.gate_up");
                mw.mtp_down = im.get<__nv_bfloat16>("L048.mtp_exp.down");
            }
            const size_t ms = size_t(std::max(o.max_rows, o.slots));
            last_hyper = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HC_D);
            mtp_hid_in = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HC_D);
            mtp_hid_cur = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HC_D);
            mtp_sc.emb_raw = dalloc<__nv_bfloat16>(ms * geom::HID);
            mtp_sc.emb_norm = dalloc<__nv_bfloat16>(ms * geom::HID);
            mtp_sc.emb_proj = dalloc<__nv_bfloat16>(ms * geom::HID);
            mtp_sc.hid_norm = dalloc<__nv_bfloat16>(ms * geom::HC_D);
            mtp_sc.hid_proj = dalloc<__nv_bfloat16>(ms * geom::HC_D);
            mtp_sc.mtp_hyper = dalloc<__nv_bfloat16>(ms * geom::HC_D);
            mtp_sc.exp_mid = dalloc<__nv_bfloat16>(ms * geom::TOPK * geom::SHARED_I);
            mtp_sc.exp_dn_part = dalloc<float>(ms * geom::TOPK * geom::HID);
            std::fprintf(stderr, "# rank %d: qw38 MTP L048 drafter loaded (K=%d)\n", o.rank0, o.mtp);
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
            if (has_mtp) {
                fp8_arena.add(mtp_w.fc_emb, mtp_w.fp8_fc_emb, geom::HID, geom::HID);
                fp8_arena.add(mtp_w.fc_hid, mtp_w.fp8_fc_hid, geom::HID, geom::HID);
                fp8_arena.add(mtp_w.attn_hc.down, mtp_w.attn_hc.fp8_down, geom::HC_RANK, geom::HC_D);
                fp8_arena.add(mtp_w.attn_hc.up, mtp_w.attn_hc.fp8_up, geom::HC_D, geom::HC_RANK, true);
                fp8_arena.add(mtp_w.mlp_hc.down, mtp_w.mlp_hc.fp8_down, geom::HC_RANK, geom::HC_D);
                fp8_arena.add(mtp_w.mlp_hc.up, mtp_w.mlp_hc.fp8_up, geom::HC_D, geom::HC_RANK, true);
                fp8_arena.add(mtp_w.final_hc.down, mtp_w.final_hc.fp8_down, geom::HC_RANK, geom::HC_D);
                fp8_arena.add(mtp_w.final_hc.up, mtp_w.final_hc.fp8_up, geom::HC_D, geom::HC_RANK, true);
                for (int i = 0; i < o.nloc; ++i) {
                    const int r = o.rank0 + i;
                    const int nh_q = geom::nh_q(o.nr);
                    const int nh_kv = geom::nh_kv(o.nr, r);
                    auto& aw = mtp_w.attn[size_t(i)];
                    fp8_arena.add(aw.q_proj, aw.fp8_q, nh_q * geom::HEAD_DIM * 2, geom::HID);
                    fp8_arena.add(aw.k_proj, aw.fp8_k, nh_kv * geom::HEAD_DIM, geom::HID);
                    fp8_arena.add(aw.v_proj, aw.fp8_v, nh_kv * geom::HEAD_DIM, geom::HID);
                    fp8_arena.add(aw.idx_qk_proj, aw.fp8_idx, geom::IDX_PROJ_OUT, geom::HID);
                    fp8_arena.add(aw.o_proj, aw.fp8_o, geom::HID, nh_q * geom::HEAD_DIM);
                    auto& mw = mtp_w.moe[size_t(i)];
                    fp8_arena.add(mw.sh_gate, mw.fp8_sh_gate, mw.sh_i, geom::HID);
                    fp8_arena.add(mw.sh_up, mw.fp8_sh_up, mw.sh_i, geom::HID);
                    fp8_arena.add(mw.sh_down, mw.fp8_sh_down, geom::HID, mw.sh_i);
                    // Quantize all 512 MTP BF16 routed experts to block-32 FP8!
                    fp8_arena.add(mw.mtp_gate_up, mtp_w.fp8_exp_gu[size_t(i)], geom::NEXP * 2 * mw.sh_i, geom::HID);
                    fp8_arena.add(mw.mtp_down, mtp_w.fp8_exp_dn[size_t(i)], geom::NEXP * geom::HID, mw.sh_i);
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
        hc_sc.up = nullptr;
        hc_sc.inj = dalloc<float>(M * geom::HC);

        const int max_nh_v = geom::gdn_vh(o.nr);
        const int max_qkv = (2 * geom::gdn_kh_max(o.nr) + max_nh_v) * geom::GDN_DK;
        gdn_sc.qkv_raw = dalloc<__nv_bfloat16>(M * size_t(max_qkv));
        gdn_sc.qkv_conv = dalloc<__nv_bfloat16>(M * size_t(max_qkv));
        gdn_sc.z = dalloc<__nv_bfloat16>(M * size_t(max_nh_v) * geom::GDN_DV);
        gdn_sc.a = dalloc<__nv_bfloat16>(M * size_t(max_nh_v));
        gdn_sc.b = dalloc<__nv_bfloat16>(M * size_t(max_nh_v));
        gdn_sc.core = dalloc<__nv_bfloat16>(M * size_t(max_nh_v) * geom::GDN_DV);
        const size_t max_wy_chunks = (M + qw38::GDN_WY_CH - 1) / qw38::GDN_WY_CH;
        gdn_sc.wy_qe = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DK);
        gdn_sc.wy_kd = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DK);
        gdn_sc.wy_w = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DK);
        gdn_sc.wy_u = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DV);
        gdn_sc.wy_aq = dalloc<uint16_t>(size_t(max_nh_v) * M * qw38::GDN_WY_CH);
        gdn_sc.wy_glast = dalloc<float>(size_t(max_nh_v) * max_wy_chunks);

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
        moe_sc.grp_off = dalloc<int>(M * geom::TOPK);
        moe_sc.grp_routes = dalloc<int>(M * geom::TOPK * 8);
        moe_sc.x_had = dalloc<__half>(M * geom::HID);
        moe_sc.fc1_part = dalloc<float>(8 * geom::TOPK * size_t(g6nr::EB) * 2 * 2 * 128);
        moe_sc.fc1_act = dalloc<__half>(M * geom::TOPK * size_t(g6nr::EB) * 128);
        moe_sc.fc2_part = dalloc<float>(M * geom::TOPK * geom::HID);
        moe_sc.routed_acc = dalloc<float>(M * geom::HID);
        moe_sc.sh_gate_buf = dalloc<__nv_bfloat16>(M * geom::SHARED_I);
        moe_sc.sh_up_buf = dalloc<__nv_bfloat16>(M * geom::SHARED_I);
        moe_sc.sh_out = dalloc<__nv_bfloat16>(M * geom::HID);
        moe_sc.sh_scalar_gate = dalloc<float>(M);

        constexpr size_t HEAD_CHUNK = 256;
        const size_t HM = std::min<size_t>(M, HEAD_CHUNK);
        head_logits = dalloc<float>(HM * size_t(max_vr));
        head_sc.blk_max = dalloc<float>(HM * qw38::HEAD_BLKS);
        head_sc.blk_cnt = dalloc<int>(HM * qw38::HEAD_BLKS);
        head_sc.blk_dsum = dalloc<double>(HM * qw38::HEAD_BLKS);
        head_sc.row_max = dalloc<float>(HM);
        head_sc.row_fin = dalloc<int>(HM);
        head_sc.surv_cnt = dalloc<int>(HM);
        head_sc.surv_val = dalloc<float>(HM * qw38::HEAD_MAX_SURV);
        head_sc.surv_id = dalloc<int>(HM * qw38::HEAD_MAX_SURV);
        const size_t hrb = head_row_bytes(o.nc);
        head_dev_pkt = dalloc<uint8_t>(M * hrb);
        QW38_CK(cudaHostAlloc(&head_host_parts, size_t(o.nr) * M * hrb, cudaHostAllocDefault));
        std::memset(head_host_parts, 0, size_t(o.nr) * M * hrb);
        QW38_CK(cudaHostAlloc(&h_meta_pinned, 3 * M * sizeof(int), cudaHostAllocDefault));
        QW38_CK(cudaHostAlloc(&h_rope_pinned, M * sizeof(int3), cudaHostAllocDefault));

        d_tok = dalloc<int>(M);
        d_slot = dalloc<int>(M);
        d_pos = dalloc<int>(M);
        d_rope = dalloc<int3>(M);
        slot_mrope_delta.assign(size_t(o.slots), 0);

        // Warm up cuBLAS handles/heuristics at init so step 0 does not pay ~500 ms lazy-init latency.
        {
            const int warm_m = std::min(o.max_rows, 16);
            const int warm_v = std::min(v_rows[0], 256);
            cublasSetStream(blas, stream);
            qw38::gemm_bf16_nt(blas, warm_m, geom::HC_RANK, geom::HC_D, hc_sc.hn, final_hc.down, hc_sc.down);
            qw38::gemm_bf16_nt_f32(blas, warm_m, warm_v, geom::HID, mixed, lm_head[0], head_logits);
            QW38_CK(cudaStreamSynchronize(stream));
        }
    }

    void reset_slot(int slot) {
        if (slot < 0 || slot >= o.slots) return;
        ple_ctx[size_t(slot)] = {geom::TOK_EOS, geom::TOK_EOS};
        slot_mrope_delta[size_t(slot)] = 0;
        if (has_ple) {
            const size_t pb = size_t(geom::PLE_NHIST) * geom::HC_D * 2;
            QW38_CK(cudaMemsetAsync(ple_st.conv_hist + size_t(slot) * geom::PLE_NHIST * geom::HC_D, 0, pb, stream));
        }
        if (has_mtp) {
            for (int i = 0; i < o.nloc; ++i) {
                QW38_CK(cudaMemsetAsync(mtp_w.kv[size_t(i)].idx_raw + size_t(slot) * 4 * geom::IDX_DIM, 0, 4 * geom::IDX_DIM * 2, stream));
                QW38_CK(cudaMemsetAsync(mtp_w.kv[size_t(i)].idx_raw_thw + size_t(slot), 0, sizeof(int3), stream));
            }
        }
        for (auto& y : L) {
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                auto& rl = y.rk[size_t(i)];
                if (y.is_full) {
                    QW38_CK(cudaMemsetAsync(rl.kv.idx_raw + size_t(slot) * 4 * geom::IDX_DIM, 0, 4 * geom::IDX_DIM * 2, stream));
                    QW38_CK(cudaMemsetAsync(rl.kv.idx_raw_thw + size_t(slot), 0, sizeof(int3), stream));
                } else {
                    const size_t sb = size_t(geom::gdn_vh(o.nr)) * 128 * 128 * 4;
                    const size_t cb = size_t(geom::GDN_NHIST) * size_t(geom::gdn_qkv_local(o.nr, r)) * 2;
                    QW38_CK(cudaMemsetAsync(rl.gdn_st.S + size_t(slot) * size_t(geom::gdn_vh(o.nr)) * 128 * 128, 0, sb, stream));
                    QW38_CK(cudaMemsetAsync(rl.gdn_st.conv + size_t(slot) * geom::GDN_NHIST * size_t(geom::gdn_qkv_local(o.nr, r)), 0, cb, stream));
                }
            }
        }
    }

    static uint64_t ple_tok_hash(const int* toks, int len) {
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < len; ++i) {
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(toks[i]));
            h *= 1099511628211ULL;
        }
        return h;
    }

    void ple_wait_h2d(int buf) {
        if (ple_h2d_pending[buf]) {
            QW38_CK(cudaEventSynchronize(ple_h2d_done[buf]));
            ple_h2d_pending[buf] = false;
        }
    }

    void ple_wait_job() {
        std::unique_lock<std::mutex> lk(ple_mu);
        ple_done_cv.wait(lk, [&]() { return ple_job_done; });
    }

    // Kick off next prefill chunk's PLE NVMe fetch into the alternate pinned buffer so it overlaps
    // behind the current chunk's Layers 1..47 GPU compute. Never mutates ple_ctx until consumed.
    void ple_prefetch_next(int slot, int pos0, const int* toks, int len) {
        if (!has_ple || !o.ple_async || !ple_thr.joinable()) return;
        if (slot < 0 || slot >= o.slots || len <= 0 || len > o.max_rows || !toks) return;
        ple_wait_job();
        const int buf = 1 - ple_active_buf;
        ple_wait_h2d(buf);
        const std::array<int32_t, 2> ctx_in = (pos0 == 0)
            ? std::array<int32_t, 2>{geom::TOK_EOS, geom::TOK_EOS}
            : ple_ctx[size_t(slot)];
        std::array<int32_t, 2> ctx_tmp = ctx_in;
        ple_job_ids.resize(size_t(len) * geom::PLE_HEADS);
        ple_store.compute_ids(toks, len, ctx_tmp.data(), true, ple_job_ids.data());
        ple_pf_valid = true;
        ple_pf_buf = buf;
        ple_pf_slot = slot;
        ple_pf_pos0 = pos0;
        ple_pf_len = len;
        ple_pf_hash = ple_tok_hash(toks, len);
        ple_pf_ctx_in = ctx_in;
        ple_pf_ctx_out = ctx_tmp;
        {
            std::lock_guard<std::mutex> lk(ple_mu);
            ple_job_buf = buf;
            ple_job_is_prefetch = false;
            ple_job_done = false;
            ple_has_job = true;
        }
        ple_cv.notify_one();
    }

    // Stage next prefill chunk's PLE row IDs so ple_thr warms PleRowCache during Layers 1..47 of this chunk.
    void stage_ple_prefetch(const int* toks, int n, int32_t ctx0, int32_t ctx1) {
        if (!has_ple || !o.ple_async || !ple_thr.joinable() || n <= 0) return;
        int32_t c[2] = {ctx0, ctx1};
        ple_next_pf_ids.resize(size_t(n) * geom::PLE_HEADS);
        ple_store.compute_ids(toks, n, c, false, ple_next_pf_ids.data());
    }

    // Forward one step over `segs` (`M` total rows), optionally skipping lm_head (prefill chunks).
    // When `snap_k > 0` (MTP verify step), saves post-token-t recurrent/conv/idx_raw snapshots for t in 0..snap_k-1!
    // When `vis_n > 0` (multimodal prefill), splices staged vision embeddings into HC streams and masks image tokens in PLE.
    // When `rope_thw != nullptr`, uses per-row 3D MRoPE (t, h, w) coords; else uses isotropic (pos + slot_mrope_delta[slot]).
    // When `head_last_only` is true (prefill final chunk), runs final HC + lm_head only on row `M - 1` into `out[M - 1]`.
    void forward(const std::vector<qw38::StepSeg>& segs, const int* toks, const RowSample* samples,
                 std::vector<RowOut>& out, bool skip_head = false, int snap_k = 0,
                 int vis_n = 0, const int3* rope_thw = nullptr, bool head_last_only = false) {
        int M = 0;
        int max_pos = 0;
        bool all_len1 = true;
        for (size_t s = 0; s < segs.size(); ++s) {
            const auto& g = segs[s];
            M += g.len;
            if (g.len > 0) {
                const int ep = g.pos0 + g.len - 1;
                if (ep > max_pos) max_pos = ep;
            }
            if (g.len != 1 || g.row0 != static_cast<int>(s)) all_len1 = false;
        }
        if (M <= 0 || M > o.max_rows) die("forward: bad M=" + std::to_string(M));
        all_len1 = all_len1 && (static_cast<int>(segs.size()) == M);

        int* h_tok_p = h_meta_pinned;
        int* h_slot_p = h_meta_pinned + o.max_rows;
        int* h_pos_p = h_meta_pinned + 2 * o.max_rows;
        std::memcpy(h_tok_p, toks, size_t(M) * sizeof(int));
        for (const auto& g : segs) {
            const int dlt = slot_mrope_delta[size_t(g.slot)];
            for (int t = 0; t < g.len; ++t) {
                h_slot_p[g.row0 + t] = g.slot;
                h_pos_p[g.row0 + t] = g.pos0 + t;
                if (!rope_thw) {
                    const int rp = g.pos0 + t + dlt;
                    h_rope_pinned[g.row0 + t] = make_int3(rp, rp, rp);
                }
            }
        }
        if (rope_thw) std::memcpy(h_rope_pinned, rope_thw, size_t(M) * sizeof(int3));
        QW38_CK(cudaMemcpyAsync(d_tok, h_tok_p, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_slot, h_slot_p, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_pos, h_pos_p, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_rope, h_rope_pinned, size_t(M) * sizeof(int3), cudaMemcpyHostToDevice, stream));

        const bool use_direct_ar = (o.nr > 1 && o.nloc == 1 && direct_send_dst && direct_ar_wait && direct_hc_inject);
        const bool use_direct_send = (use_direct_ar && (M > 8 || !direct_copy_ring_wait));
        const bool use_sp_hc = (use_direct_ar && skip_head && !has_mtp && M >= 64 && L.size() >= 2 &&
                                step_begin_sp && direct_hc_rs_inject_norm && direct_ag_assemble);
        const int sp_slice = use_sp_hc ? ((M + o.nr - 1) / o.nr) : M;
        const int sp_r0 = use_sp_hc ? std::min(o.rank0 * sp_slice, M) : 0;
        const int sp_m = use_sp_hc ? std::min(sp_slice, M - sp_r0) : M;
        const bool last_only = head_last_only && !skip_head && M > 1;
        const int head_M = last_only ? 1 : M;

        // 0a. Notify multi-collective fabric step start (single enqueue_step for all collectives on TP>=2)
        if (use_sp_hc) step_begin_sp(M, int(L.size()), sp_slice);
        else if (step_begin) step_begin(M, skip_head, -1, head_M);

        // Mask <|image_pad|> -> TOK_EOS for Layer-1 PLE n-gram hashing (matches Qwen4ExpTextModel conv_mask).
        std::vector<int> ple_tok_buf;
        const int* ple_toks = toks;
        if (has_ple && vis_n > 0) {
            ple_tok_buf.assign(toks, toks + M);
            for (int& tk : ple_tok_buf) if (tk == qw38v::TOK_IMAGE) tk = geom::TOK_EOS;
            ple_toks = ple_tok_buf.data();
        }

        // 0b. Consume cross-chunk PLE prefetch if it matches, else kick off async PLE NVMe fetch
        //     into the alternate double buffer BEFORE Layer 0 GPU compute!
        bool ple_in_flight = false;
        int ple_cur_buf = 0;
        if (has_ple && o.ple_async && ple_thr.joinable()) {
            bool pf_hit = false;
            if (ple_pf_valid && vis_n == 0 && segs.size() == 1 && segs[0].row0 == 0 &&
                segs[0].slot == ple_pf_slot && segs[0].pos0 == ple_pf_pos0 && segs[0].len == ple_pf_len) {
                const std::array<int32_t, 2> cur_ctx = (segs[0].pos0 == 0)
                    ? std::array<int32_t, 2>{geom::TOK_EOS, geom::TOK_EOS}
                    : ple_ctx[size_t(segs[0].slot)];
                if (cur_ctx == ple_pf_ctx_in && ple_tok_hash(ple_toks, M) == ple_pf_hash) {
                    pf_hit = true;
                }
            }
            if (pf_hit) {
                ple_ctx[size_t(segs[0].slot)] = ple_pf_ctx_out;
                ple_cur_buf = ple_pf_buf;
                ple_active_buf = ple_cur_buf;
                ple_pf_valid = false;
                ple_in_flight = true;
            } else {
                ple_wait_job();
                ple_pf_valid = false;
                ple_cur_buf = 1 - ple_active_buf;
                ple_active_buf = ple_cur_buf;
                ple_wait_h2d(ple_cur_buf);
                ple_job_ids.resize(size_t(M) * geom::PLE_HEADS);
                for (const auto& g : segs) {
                    if (g.len <= 0) continue;
                    if (g.pos0 == 0) ple_ctx[size_t(g.slot)] = {geom::TOK_EOS, geom::TOK_EOS};
                    ple_store.compute_ids(ple_toks + g.row0, g.len, ple_ctx[size_t(g.slot)].data(), true,
                                          ple_job_ids.data() + size_t(g.row0) * geom::PLE_HEADS);
                }
                {
                    std::lock_guard<std::mutex> lk(ple_mu);
                    ple_job_buf = ple_cur_buf;
                    ple_job_is_prefetch = false;
                    ple_job_done = false;
                    ple_has_job = true;
                }
                ple_cv.notify_one();
                ple_in_flight = true;
            }
        }

        auto finish_ple_io = [&]() {
            const auto p0 = std::chrono::steady_clock::now();
            if (ple_in_flight) {
                std::unique_lock<std::mutex> lk(ple_mu);
                ple_done_cv.wait(lk, [&]() { return ple_job_done; });
                if (!ple_job_ok) die("PLE async fetch_rows failed");
                ple_in_flight = false;
                if (!ple_next_pf_ids.empty()) {
                    ple_job_ids.swap(ple_next_pf_ids);
                    ple_next_pf_ids.clear();
                    ple_job_is_prefetch = true;
                    ple_job_done = false;
                    ple_has_job = true;
                    lk.unlock();
                    ple_cv.notify_one();
                }
            } else {
                ple_cur_buf = 1 - ple_active_buf;
                ple_active_buf = ple_cur_buf;
                ple_wait_h2d(ple_cur_buf);
                std::vector<uint64_t> all_ids(size_t(M) * geom::PLE_HEADS);
                for (const auto& g : segs) {
                    if (g.len <= 0) continue;
                    if (g.pos0 == 0) ple_ctx[size_t(g.slot)] = {geom::TOK_EOS, geom::TOK_EOS};
                    ple_store.compute_ids(ple_toks + g.row0, g.len, ple_ctx[size_t(g.slot)].data(), true,
                                          all_ids.data() + size_t(g.row0) * geom::PLE_HEADS);
                }
                if (ple_store.is_row_trellis()) {
                    if (!ple_store.fetch_trellis_rows(all_ids.data(), all_ids.size(),
                                                      ple_codes_pinned[ple_cur_buf], ple_svh_pinned[ple_cur_buf]))
                        die("PLE fetch_trellis_rows failed");
                } else {
                    if (!ple_store.fetch_rows(all_ids.data(), all_ids.size(),
                                              reinterpret_cast<uint8_t*>(ple_pinned[ple_cur_buf])))
                        die("PLE fetch_rows failed");
                }
            }
            const auto p1 = std::chrono::steady_clock::now();
            if (o.profile) prof.ple_io_ms += std::chrono::duration<double, std::milli>(p1 - p0).count();
        };

        auto upload_ple_to_dev = [&]() {
            if (ple_store.is_row_trellis()) {
                const size_t n_ple_rows = size_t(M) * geom::PLE_HEADS;
                QW38_CK(cudaMemcpyAsync(d_ple_codes, ple_codes_pinned[ple_cur_buf], n_ple_rows * ple_store.row_bytes(),
                                        cudaMemcpyHostToDevice, stream));
                QW38_CK(cudaMemcpyAsync(d_ple_svh, ple_svh_pinned[ple_cur_buf], n_ple_rows * sizeof(uint16_t),
                                        cudaMemcpyHostToDevice, stream));
            } else {
                QW38_CK(cudaMemcpyAsync(ple_sc.emb, ple_pinned[ple_cur_buf], size_t(M) * geom::PLE_DIM * 2,
                                        cudaMemcpyHostToDevice, stream));
            }
            QW38_CK(cudaEventRecord(ple_h2d_done[ple_cur_buf], stream));
            ple_h2d_pending[ple_cur_buf] = true;
        };

        auto run_ple_gpu = [&]() {
            if (ple_store.is_row_trellis()) {
                const auto& rt = ple_store.trellis_info();
                qw38::ple_rowtrellis_decode(d_ple_codes, d_ple_svh, d_ple_suh, ple_sc.emb,
                                            M * geom::PLE_HEADS, rt.n_tiles, geom::PLE_HEAD_DIM,
                                            rt.k_bits, rt.e4m3, stream);
            }
            qw38::ple_forward(blas, ple_w, ple_st, ple_sc, hyper, segs.data(), int(segs.size()), M, true, stream, d_slot, d_pos, snap_k);
        };

        const size_t part_stride = size_t(M) * geom::HID;
        const size_t hrb = head_row_bytes(o.nc);
        const size_t rank_pkt_bytes = size_t(head_M) * hrb;

        auto run_layer_gpu = [&](size_t li, int& coll) {
            Layer& y = L[li];
            qw38::hc_prepare(blas, y.attn_hc, hyper, mixed, hc_sc, M, stream);
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                RankLayer& rl = y.rk[size_t(i)];
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                if (y.is_full) {
                    qw38::full_attn_forward(blas, rl.attn_w, rl.kv, attn_sc, mixed, p_dst,
                                            d_slot, d_pos, segs.data(), int(segs.size()), M, o.nr, r,
                                            max_pps, max_blocks, stream, snap_k, d_rope);
                } else {
                    qw38::gdn_forward(blas, rl.gdn_w, rl.gdn_st, gdn_sc, mixed, p_dst,
                                      segs.data(), int(segs.size()), M, o.nr, r, true, stream, d_slot, d_pos, snap_k);
                }
            }
            if (use_direct_ar && use_direct_send) {
                direct_ar_wait(coll, part_stride * 2);
                direct_hc_inject(coll, hyper, hc_sc.inj, M, nullptr);
                ++coll;
            } else if (o.nr > 1 && gather_inject) {
                gather_inject(coll++, hyper, parts, hc_sc.inj, M);
            } else if (use_direct_ar) {
                const __nv_bfloat16* own_p = parts + size_t(o.rank0) * part_stride;
                direct_copy_ring_wait(coll, own_p, part_stride * 2);
                direct_hc_inject(coll, hyper, hc_sc.inj, M, own_p);
                ++coll;
            } else {
                if (o.nr > 1 && gather_dev) gather_dev(coll++, parts, part_stride * 2);
                else ++coll;
                qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            }

            qw38::hc_prepare(blas, y.mlp_hc, hyper, mixed, hc_sc, M, stream);
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                RankLayer& rl = y.rk[size_t(i)];
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                qw38::moe_forward(blas, rl.moe_w, moe_sc, mixed, p_dst, M, stream);
            }
            if (use_direct_ar && use_direct_send) {
                direct_ar_wait(coll, part_stride * 2);
                direct_hc_inject(coll, hyper, hc_sc.inj, M, nullptr);
                ++coll;
            } else if (o.nr > 1 && gather_inject) {
                gather_inject(coll++, hyper, parts, hc_sc.inj, M);
            } else if (use_direct_ar) {
                const __nv_bfloat16* own_p = parts + size_t(o.rank0) * part_stride;
                direct_copy_ring_wait(coll, own_p, part_stride * 2);
                direct_hc_inject(coll, hyper, hc_sc.inj, M, own_p);
                ++coll;
            } else {
                if (o.nr > 1 && gather_dev) gather_dev(coll++, parts, part_stride * 2);
                else ++coll;
                qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            }
        };

        auto run_head_gpu = [&](bool hn_ready = false) {
            if (last_only) {
                qw38::hc_prepare(blas, final_hc, hyper + size_t(M - 1) * geom::HC_D, mixed, hc_sc, 1, stream);
            } else if (hn_ready) {
                qw38::hc_prepare_from_hn(blas, final_hc, mixed, hc_sc, M, stream);
            } else {
                qw38::hc_prepare(blas, final_hc, hyper, mixed, hc_sc, M, stream);
            }
            constexpr int HEAD_CHUNK = 256;
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                for (int r0 = 0; r0 < head_M; r0 += HEAD_CHUNK) {
                    const int mc = std::min(head_M - r0, HEAD_CHUNK);
                    const __nv_bfloat16* mx_ptr = mixed + size_t(r0) * geom::HID;
                    if (head_M <= 20 && i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q) {
                        qw38::launch_fp8_gemv_smallm<true>(mx_ptr, fp8_lm_head[size_t(i)], head_logits, mc, stream);
                    } else if (head_M <= 8) {
                        qw38::launch_bf16_gemv_smallm<true>(mx_ptr, lm_head[size_t(i)], head_logits, mc, v_rows[size_t(i)], geom::HID, stream);
                    } else {
                        cublasSetStream(blas, stream);
                        qw38::gemm_bf16_nt_f32(blas, mc, v_rows[size_t(i)], geom::HID, mx_ptr, lm_head[size_t(i)], head_logits);
                    }
                    qw38::head_pack_fast(head_logits, mc, v_rows[size_t(i)], v_lo[size_t(i)], o.nc, head_sc,
                                         head_dev_pkt + size_t(r0) * hrb, hrb, stream);
                }
                QW38_CK(cudaMemcpyAsync(head_host_parts + size_t(r) * rank_pkt_bytes,
                                        head_dev_pkt, rank_pkt_bytes, cudaMemcpyDeviceToHost, stream));
            }
        };

        // Fast CUDA Graph path for decode (M <= 16, all_len1) AND MTP verify (M <= 64, all_verify)!
        const int V_chk = snap_k + 1;
        bool all_verify = (!all_len1 && snap_k > 0 && !segs.empty() && static_cast<int>(segs.size()) * V_chk == M);
        if (all_verify) {
            for (size_t s = 0; s < segs.size(); ++s) {
                if (segs[s].len != V_chk || segs[s].row0 != static_cast<int>(s) * V_chk) { all_verify = false; break; }
            }
        }
        const bool verify_graph_ok = all_verify && M <= 64;
        const bool graph_ok = o.use_graph && !o.profile && !skip_head && !last_only && vis_n == 0 && !rope_thw &&
                              ((snap_k == 0 && M <= 16 && all_len1) || verify_graph_ok) &&
                              (o.nr == 1 || o.nloc == o.nr || static_cast<bool>(step_parity));
        if (graph_ok) {
            const int need_qsa = (((max_pos + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS) ? 1 : 0;
            const int ns_attn = qw38::gqa_split_count(M, geom::nh_kv(o.nr, o.rank0), max_pos, attn_sc.split_o != nullptr);
            const int par = step_parity ? step_parity() : 0;
            const GraphKey gk{M, need_qsa, ns_attn, par, snap_k};
            const bool split_l0_ple = (L.size() >= 2 && geom::is_ple(L[1].l) && has_ple);

            auto it = graphs.find(gk);
            if (it == graphs.end()) {
                GraphPair gp{};
                int coll = 0;
                if (split_l0_ple) {
                    cudaGraph_t ga = nullptr;
                    QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                    qw38::k_embed_hc4<<<M, 256, 0, stream>>>(d_tok, embed, hyper, M);
                    run_layer_gpu(0, coll);
                    QW38_CK(cudaStreamEndCapture(stream, &ga));
                    QW38_CK(cudaGraphInstantiate(&gp.exec_a, ga, nullptr, nullptr, 0));
                    QW38_CK(cudaGraphDestroy(ga));
                    QW38_CK(cudaGraphLaunch(gp.exec_a, stream));

                    finish_ple_io();
                    upload_ple_to_dev();

                    cudaGraph_t gb = nullptr;
                    QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                    run_ple_gpu();
                    for (size_t li = 1; li < L.size(); ++li) run_layer_gpu(li, coll);
                    run_head_gpu();
                    QW38_CK(cudaStreamEndCapture(stream, &gb));
                    QW38_CK(cudaGraphInstantiate(&gp.exec_b, gb, nullptr, nullptr, 0));
                    QW38_CK(cudaGraphDestroy(gb));
                    QW38_CK(cudaGraphLaunch(gp.exec_b, stream));
                } else {
                    if (has_ple) {
                        finish_ple_io();
                        upload_ple_to_dev();
                    }
                    cudaGraph_t gb = nullptr;
                    QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                    qw38::k_embed_hc4<<<M, 256, 0, stream>>>(d_tok, embed, hyper, M);
                    for (size_t li = 0; li < L.size(); ++li) {
                        if (geom::is_ple(L[li].l) && has_ple)
                            run_ple_gpu();
                        run_layer_gpu(li, coll);
                    }
                    run_head_gpu();
                    QW38_CK(cudaStreamEndCapture(stream, &gb));
                    QW38_CK(cudaGraphInstantiate(&gp.exec_b, gb, nullptr, nullptr, 0));
                    QW38_CK(cudaGraphDestroy(gb));
                    QW38_CK(cudaGraphLaunch(gp.exec_b, stream));
                }
                graphs.emplace(gk, gp);
            } else {
                const GraphPair& gp = it->second;
                if (gp.exec_a) {
                    QW38_CK(cudaGraphLaunch(gp.exec_a, stream));
                    finish_ple_io();
                    upload_ple_to_dev();
                } else if (has_ple) {
                    finish_ple_io();
                    upload_ple_to_dev();
                }
                QW38_CK(cudaGraphLaunch(gp.exec_b, stream));
            }

            QW38_CK(cudaStreamSynchronize(stream));
            if (o.nr > 1 && gather_host) gather_host(static_cast<int>(2 * L.size()), head_host_parts, rank_pkt_bytes);
            if (step_end) step_end();

            out.resize(size_t(M));
            std::vector<uint8_t> row_parts(size_t(o.nr) * hrb);
            for (int m = 0; m < M; ++m) {
                for (int r = 0; r < o.nr; ++r) {
                    std::memcpy(row_parts.data() + size_t(r) * hrb,
                                head_host_parts + size_t(r) * rank_pkt_bytes + size_t(m) * hrb, hrb);
                }
                out[size_t(m)] = head_merge(row_parts.data(), hrb, o.nr, o.nc);
                if (samples) out[size_t(m)].tok = sample_row(out[size_t(m)].top, samples[m]);
                else if (!out[size_t(m)].top.empty()) out[size_t(m)].tok = out[size_t(m)].top[0].id;
            }
            return;
        }

        // Eager / profiled / prefill path
        if (vis_n > 0) {
            qw38::k_embed_hc4_vis<<<M, 256, 0, stream>>>(d_tok, embed, vision.row_map(), vision.pool(), hyper, M);
        } else {
            qw38::k_embed_hc4<<<M, 256, 0, stream>>>(d_tok, embed, hyper, M);
        }

        auto t_start = [&]() { if (o.profile) cudaEventRecord(ev0, stream); };
        auto t_end = [&](double& acc) {
            if (!o.profile) return;
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            acc += ms;
        };

        if (use_sp_hc) {
            // Sequence-Parallel HC prefill (skip_head && !has_mtp, M >= 64, L >= 2):
            // Layer 0 runs replicated so Layer-1 PLE sees full hyper[0..M-1].
            // Layers 1..L-1 shard hyper/HC across ranks (RS -> local M/nr HC -> AG), cutting HC HBM+compute by TP!
            int coll = 0;
            const size_t wire_rs = size_t(sp_m) * geom::HID * 2;
            const size_t wire_ag = size_t(sp_slice) * geom::HID * 2;
            __nv_bfloat16* hyper_loc = hyper + size_t(sp_r0) * geom::HC_D;
            const int r = o.rank0;

            // --- Layer 0 (replicated) ---
            {
                Layer& y = L[0];
                RankLayer& rl = y.rk[0];
                t_start();
                qw38::hc_prepare(blas, y.attn_hc, hyper, mixed, hc_sc, M, stream);
                t_end(prof.hc_ms);

                t_start();
                __nv_bfloat16* p_dst = direct_send_dst(coll);
                if (y.is_full) {
                    qw38::full_attn_forward(blas, rl.attn_w, rl.kv, attn_sc, mixed, p_dst,
                                            d_slot, d_pos, segs.data(), int(segs.size()), M, o.nr, r,
                                            max_pps, max_blocks, stream, snap_k, d_rope);
                } else {
                    qw38::gdn_forward(blas, rl.gdn_w, rl.gdn_st, gdn_sc, mixed, p_dst,
                                      segs.data(), int(segs.size()), M, o.nr, r, true, stream, d_slot, d_pos, snap_k);
                }
                t_end(y.is_full ? prof.attn_ms : prof.gdn_ms);

                t_start(); direct_ar_wait(coll, part_stride * 2); t_end(prof.ar_ms);
                t_start();
                if (direct_hc_inject_norm) {
                    direct_hc_inject_norm(coll, hyper, y.mlp_hc.norm, y.mlp_hc.inject, hc_sc.hn, hc_sc.inj, M);
                    ++coll;
                    qw38::hc_prepare_from_hn(blas, y.mlp_hc, mixed, hc_sc, M, stream);
                } else {
                    direct_hc_inject(coll++, hyper, hc_sc.inj, M, nullptr);
                    qw38::hc_prepare(blas, y.mlp_hc, hyper, mixed, hc_sc, M, stream);
                }
                t_end(prof.hc_ms);

                t_start();
                qw38::moe_forward(blas, rl.moe_w, moe_sc, mixed, direct_send_dst(coll), M, stream);
                t_end(prof.moe_ms);

                t_start(); direct_ar_wait(coll, part_stride * 2); t_end(prof.ar_ms);
                t_start(); direct_hc_inject(coll++, hyper, hc_sc.inj, M, nullptr); t_end(prof.hc_ms);
            }

            // --- Layer 1 PLE (on full replicated hyper) + Enter SP-HC ---
            {
                Layer& y1 = L[1];
                if (geom::is_ple(y1.l) && has_ple) {
                    finish_ple_io();
                    t_start();
                    QW38_CK(cudaMemcpyAsync(ple_sc.emb, ple_pinned, size_t(M) * geom::PLE_DIM * 2, cudaMemcpyHostToDevice, stream));
                    qw38::ple_forward(blas, ple_w, ple_st, ple_sc, hyper, segs.data(), int(segs.size()), M, true, stream, d_slot, d_pos, snap_k);
                    t_end(prof.ple_gpu_ms);
                }
                t_start();
                qw38::hc_prepare(blas, y1.attn_hc, hyper_loc, direct_send_dst(coll), hc_sc, sp_m, stream);
                t_end(prof.hc_ms);
                t_start(); direct_ar_wait(coll, wire_ag); t_end(prof.ar_ms);
                t_start(); direct_ag_assemble(coll++, mixed, M, sp_slice); t_end(prof.hc_ms);
            }

            // --- Layers 1 .. L-1 (Sequence-Parallel HC!) ---
            for (size_t li = 1; li < L.size(); ++li) {
                Layer& y = L[li];
                RankLayer& rl = y.rk[0];
                const bool last_layer = (li + 1 == L.size());

                t_start();
                __nv_bfloat16* p_dst = last_layer ? parts : direct_send_dst(coll);
                if (y.is_full) {
                    qw38::full_attn_forward(blas, rl.attn_w, rl.kv, attn_sc, mixed, p_dst,
                                            d_slot, d_pos, segs.data(), int(segs.size()), M, o.nr, r,
                                            max_pps, max_blocks, stream, snap_k, d_rope);
                } else {
                    qw38::gdn_forward(blas, rl.gdn_w, rl.gdn_st, gdn_sc, mixed, p_dst,
                                      segs.data(), int(segs.size()), M, o.nr, r, true, stream, d_slot, d_pos, snap_k);
                }
                t_end(y.is_full ? prof.attn_ms : prof.gdn_ms);

                // Last layer of a skip_head && !has_mtp chunk: KV/GDN state is written; MoE/hyper output is dead.
                if (last_layer) break;

                // Attn RS -> local fused inject+norm -> local MLP HC -> MLP AG
                t_start(); direct_ar_wait(coll, wire_rs); t_end(prof.ar_ms);
                t_start();
                direct_hc_rs_inject_norm(coll++, hyper_loc, y.mlp_hc.norm, y.mlp_hc.inject, hc_sc.hn, hc_sc.inj, sp_r0, sp_m);
                qw38::hc_prepare_from_hn(blas, y.mlp_hc, direct_send_dst(coll), hc_sc, sp_m, stream);
                t_end(prof.hc_ms);
                t_start(); direct_ar_wait(coll, wire_ag); t_end(prof.ar_ms);
                t_start(); direct_ag_assemble(coll++, mixed, M, sp_slice); t_end(prof.hc_ms);

                // MoE -> MoE RS -> local fused inject+norm -> local next-Attn HC -> next-Attn AG
                t_start();
                qw38::moe_forward(blas, rl.moe_w, moe_sc, mixed, direct_send_dst(coll), M, stream);
                t_end(prof.moe_ms);

                t_start(); direct_ar_wait(coll, wire_rs); t_end(prof.ar_ms);
                t_start();
                const qw38::HcWeights& next_hc = L[li + 1].attn_hc;
                direct_hc_rs_inject_norm(coll++, hyper_loc, next_hc.norm, next_hc.inject, hc_sc.hn, hc_sc.inj, sp_r0, sp_m);
                qw38::hc_prepare_from_hn(blas, next_hc, direct_send_dst(coll), hc_sc, sp_m, stream);
                t_end(prof.hc_ms);
                t_start(); direct_ar_wait(coll, wire_ag); t_end(prof.ar_ms);
                t_start(); direct_ag_assemble(coll++, mixed, M, sp_slice); t_end(prof.hc_ms);
            }

            if (step_end) step_end();
            if (o.profile) { ++prof.steps; prof.tokens += M; }
            return;
        }

        const bool fuse_inj_hc = (M > 12) && (!use_direct_ar || static_cast<bool>(direct_hc_inject_norm));
        auto inject_or_fuse = [&](int c, const qw38::HcWeights* next_hc) -> bool {
            if (fuse_inj_hc && next_hc != nullptr) {
                if (use_direct_ar) {
                    direct_hc_inject_norm(c, hyper, next_hc->norm, next_hc->inject, hc_sc.hn, hc_sc.inj, M);
                } else {
                    const __nv_bfloat16* p0 = parts;
                    const __nv_bfloat16* p1 = (o.nr > 1) ? (parts + part_stride) : nullptr;
                    const __nv_bfloat16* p2 = (o.nr > 2) ? (parts + 2 * part_stride) : nullptr;
                    const __nv_bfloat16* p3 = (o.nr > 3) ? (parts + 3 * part_stride) : nullptr;
                    qw38::k_hc_inject_rmsnorm_inj_ptrs<<<M, 320, 0, stream>>>(
                        hyper, p0, p1, p2, p3, next_hc->norm, next_hc->inject, hc_sc.hn, hc_sc.inj,
                        M, o.nr, -1, geom::RMS_EPS);
                }
                return true;
            }
            if (use_direct_ar) {
                const __nv_bfloat16* own_p = use_direct_send ? nullptr : (parts + size_t(o.rank0) * part_stride);
                direct_hc_inject(c, hyper, hc_sc.inj, M, own_p);
            } else {
                qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            }
            return false;
        };

        int coll = 0;
        bool hn_ready = false;
        for (size_t li = 0; li < L.size(); ++li) {
            Layer& y = L[li];
            if (geom::is_ple(y.l) && has_ple) {
                finish_ple_io();
                t_start();
                upload_ple_to_dev();
                run_ple_gpu();
                hn_ready = false;
                t_end(prof.ple_gpu_ms);
            }

            t_start();
            if (hn_ready) { qw38::hc_prepare_from_hn(blas, y.attn_hc, mixed, hc_sc, M, stream); hn_ready = false; }
            else qw38::hc_prepare(blas, y.attn_hc, hyper, mixed, hc_sc, M, stream);
            t_end(prof.hc_ms);

            t_start();
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                RankLayer& rl = y.rk[size_t(i)];
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                if (y.is_full) {
                    qw38::full_attn_forward(blas, rl.attn_w, rl.kv, attn_sc, mixed, p_dst,
                                            d_slot, d_pos, segs.data(), int(segs.size()), M, o.nr, r,
                                            max_pps, max_blocks, stream, snap_k, d_rope);
                } else {
                    qw38::gdn_forward(blas, rl.gdn_w, rl.gdn_st, gdn_sc, mixed, p_dst,
                                      segs.data(), int(segs.size()), M, o.nr, r, true, stream, d_slot, d_pos, snap_k);
                }
            }
            t_end(y.is_full ? prof.attn_ms : prof.gdn_ms);

            if (!use_direct_send && o.nr > 1 && gather_inject) {
                t_start();
                gather_inject(coll++, hyper, parts, hc_sc.inj, M);
                t_end(prof.ar_ms);
                hn_ready = false;
            } else {
                t_start();
                if (use_direct_ar) {
                    if (use_direct_send) direct_ar_wait(coll, part_stride * 2);
                    else direct_copy_ring_wait(coll, parts + size_t(o.rank0) * part_stride, part_stride * 2);
                } else if (o.nr > 1 && gather_dev) gather_dev(coll, parts, part_stride * 2);
                t_end(prof.ar_ms);

                t_start();
                hn_ready = inject_or_fuse(coll, &y.mlp_hc);
                ++coll;
                t_end(prof.hc_ms);
            }

            t_start();
            if (hn_ready) { qw38::hc_prepare_from_hn(blas, y.mlp_hc, mixed, hc_sc, M, stream); hn_ready = false; }
            else qw38::hc_prepare(blas, y.mlp_hc, hyper, mixed, hc_sc, M, stream);
            t_end(prof.hc_ms);

            t_start();
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                RankLayer& rl = y.rk[size_t(i)];
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                qw38::moe_forward(blas, rl.moe_w, moe_sc, mixed, p_dst, M, stream);
            }
            t_end(prof.moe_ms);

            if (!use_direct_send && o.nr > 1 && gather_inject) {
                t_start();
                gather_inject(coll++, hyper, parts, hc_sc.inj, M);
                t_end(prof.ar_ms);
                hn_ready = false;
            } else {
                t_start();
                if (use_direct_ar) {
                    if (use_direct_send) direct_ar_wait(coll, part_stride * 2);
                    else direct_copy_ring_wait(coll, parts + size_t(o.rank0) * part_stride, part_stride * 2);
                } else if (o.nr > 1 && gather_dev) gather_dev(coll, parts, part_stride * 2);
                t_end(prof.ar_ms);

                const qw38::HcWeights* next_hc = nullptr;
                if (li + 1 < L.size()) {
                    if (!(geom::is_ple(L[li + 1].l) && has_ple)) next_hc = &L[li + 1].attn_hc;
                } else if (!skip_head) {
                    next_hc = &final_hc;
                }
                t_start();
                hn_ready = inject_or_fuse(coll, next_hc);
                ++coll;
                t_end(prof.hc_ms);
            }
        }

        if (skip_head) {
            if (step_end) step_end();
            if (o.profile) { ++prof.steps; prof.tokens += M; }
            return;
        }

        t_start();
        run_head_gpu(hn_ready);
        QW38_CK(cudaStreamSynchronize(stream));
        if (o.nr > 1 && gather_host) gather_host(coll++, head_host_parts, rank_pkt_bytes);
        if (step_end) step_end();

        out.resize(size_t(M));
        std::vector<uint8_t> row_parts(size_t(o.nr) * hrb);
        if (last_only) {
            for (int r = 0; r < o.nr; ++r) {
                std::memcpy(row_parts.data() + size_t(r) * hrb,
                            head_host_parts + size_t(r) * rank_pkt_bytes, hrb);
            }
            out[size_t(M - 1)] = head_merge(row_parts.data(), hrb, o.nr, o.nc);
            if (samples) out[size_t(M - 1)].tok = sample_row(out[size_t(M - 1)].top, samples[M - 1]);
            else if (!out[size_t(M - 1)].top.empty()) out[size_t(M - 1)].tok = out[size_t(M - 1)].top[0].id;
        } else {
            for (int m = 0; m < M; ++m) {
                for (int r = 0; r < o.nr; ++r) {
                    std::memcpy(row_parts.data() + size_t(r) * hrb,
                                head_host_parts + size_t(r) * rank_pkt_bytes + size_t(m) * hrb, hrb);
                }
                out[size_t(m)] = head_merge(row_parts.data(), hrb, o.nr, o.nc);
                if (samples) out[size_t(m)].tok = sample_row(out[size_t(m)].top, samples[m]);
                else if (!out[size_t(m)].top.empty()) out[size_t(m)].tok = out[size_t(m)].top[0].id;
            }
        }
        t_end(prof.head_ms);
        if (o.profile) { ++prof.steps; prof.tokens += M; }
    }

    // -------------------------------------------------------------------------------------------- MTP (Layer 48)
    // Save trunk's final hyper state from row `src_row` of `hyper` into `last_hyper[slot]`.
    void mtp_save_last_hyper(int slot, int src_row) {
        if (!has_mtp) return;
        QW38_CK(cudaMemcpyAsync(last_hyper + size_t(slot) * geom::HC_D,
                                hyper + size_t(src_row) * geom::HC_D,
                                geom::HC_D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }

    // Populate L048 MTP KV cache for `len` consecutive positions `[pos0 .. pos0+len-1]` of `slot`,
    // using trunk's `hyper[row0 .. row0+len-1]` and next tokens `next_toks[0 .. len-1]`!
    // Only runs mtp_fuse + attn_hc + k/v/idx_k projections + KV scatter (ZERO q/attn/o/MoE/AR!).
    void mtp_sync_kv(int slot, int row0, int pos0, int len, const int* next_toks) {
        if (!has_mtp || len <= 0) return;
        int* h_tok_p = h_meta_pinned;
        int* h_slot_p = h_meta_pinned + o.max_rows;
        int* h_pos_p = h_meta_pinned + 2 * o.max_rows;
        const int dlt = slot_mrope_delta[size_t(slot)];
        for (int t = 0; t < len; ++t) {
            h_tok_p[t] = next_toks[t];
            h_slot_p[t] = slot;
            h_pos_p[t] = pos0 + t;
            const int rp = pos0 + t + dlt;
            h_rope_pinned[t] = make_int3(rp, rp, rp);
        }
        QW38_CK(cudaMemcpyAsync(d_tok, h_tok_p, size_t(len) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_slot, h_slot_p, size_t(len) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_pos, h_pos_p, size_t(len) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_rope, h_rope_pinned, size_t(len) * sizeof(int3), cudaMemcpyHostToDevice, stream));

        const __nv_bfloat16* h_src = hyper + size_t(row0) * geom::HC_D;
        qw38::k_gather_embed<<<len, 256, 0, stream>>>(d_tok, embed, mtp_sc.emb_raw, len);
        qw38::k_rmsnorm_1pw<geom::HID><<<len, 256, 0, stream>>>(mtp_sc.emb_raw, mtp_w.pre_fc_norm_emb, mtp_sc.emb_norm, len, 1, geom::RMS_EPS);
        qw38::k_rmsnorm_1pw<geom::HID><<<len * geom::HC, 256, 0, stream>>>(h_src, mtp_w.pre_fc_norm_hid, mtp_sc.hid_norm, len, geom::HC, geom::RMS_EPS);

        if (len <= 20 && mtp_w.fp8_fc_emb.q) {
            qw38::launch_fp8_gemv_smallm<false>(mtp_sc.emb_norm, mtp_w.fp8_fc_emb, mtp_sc.emb_proj, len, stream);
        } else {
            cublasSetStream(blas, stream);
            qw38::gemm_bf16_nt(blas, len, geom::HID, geom::HID, mtp_sc.emb_norm, mtp_w.fc_emb, mtp_sc.emb_proj);
        }
        if (len * geom::HC <= 20 && mtp_w.fp8_fc_hid.q) {
            qw38::launch_fp8_gemv_smallm<false>(mtp_sc.hid_norm, mtp_w.fp8_fc_hid, mtp_sc.hid_proj, len * geom::HC, stream);
        } else {
            cublasSetStream(blas, stream);
            qw38::gemm_bf16_nt(blas, len * geom::HC, geom::HID, geom::HID, mtp_sc.hid_norm, mtp_w.fc_hid, mtp_sc.hid_proj);
        }
        qw38::k_mtp_fuse_stream4<<<len, 256, 0, stream>>>(mtp_sc.hid_proj, mtp_sc.emb_proj, mtp_sc.mtp_hyper, len);
        qw38::hc_prepare(blas, mtp_w.attn_hc, mtp_sc.mtp_hyper, mixed, hc_sc, len, stream);

        qw38::StepSeg seg{slot, 0, pos0, len};
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            const int nh_kv = geom::nh_kv(o.nr, r);
            const auto& aw = mtp_w.attn[size_t(i)];
            auto& kv = mtp_w.kv[size_t(i)];
            if (len <= 20 && aw.fp8_k.q) {
                qw38::launch_fp8_gemv_smallm<false>(mixed, aw.fp8_k, attn_sc.k, len, stream);
                qw38::launch_fp8_gemv_smallm<false>(mixed, aw.fp8_v, attn_sc.v, len, stream);
                qw38::launch_fp8_gemv_smallm<false>(mixed, aw.fp8_idx, attn_sc.idx_qk, len, stream);
            } else {
                cublasSetStream(blas, stream);
                qw38::gemm_bf16_nt(blas, len, nh_kv * geom::HEAD_DIM, geom::HID, mixed, aw.k_proj, attn_sc.k);
                qw38::gemm_bf16_nt(blas, len, nh_kv * geom::HEAD_DIM, geom::HID, mixed, aw.v_proj, attn_sc.v);
                qw38::gemm_bf16_nt(blas, len, geom::IDX_PROJ_OUT, geom::HID, mixed, aw.idx_qk_proj, attn_sc.idx_qk);
            }
            qw38::k_qw38_kv_write<<<len * nh_kv, 256, 0, stream>>>(attn_sc.k, attn_sc.v, aw.k_norm, d_slot, d_pos, kv.kv_pages, len, nh_kv, max_pps, geom::RMS_EPS, d_rope);
            qw38::qsa_k_update_seg_fast(attn_sc.idx_qk, aw.idx_k_norm, kv.idx_raw, nullptr, kv.idx_pages,
                                        max_pps, seg.slot, seg.row0, seg.pos0, seg.len, 0, geom::RMS_EPS, stream,
                                        kv.idx_raw_thw, nullptr, d_rope);
        }
    }

    // Run ONE batched MTP draft step across `S` active sessions (1 token per session), CUDA-graphed when o.use_graph!
    //   inputs: hid_in [S, 10240], toks [S], pos_mtp [S]
    //   outputs: hid_out [S, 10240] (L048 output hyper state for chaining), out [S] sampled draft tokens.
    void mtp_draft_step(const std::vector<int>& slots, const int* toks, const int* pos_mtp,
                        const __nv_bfloat16* hid_in, __nv_bfloat16* hid_out,
                        const RowSample* samples, std::vector<RowOut>& out) {
        const int M = static_cast<int>(slots.size());
        int* h_tok_p = h_meta_pinned;
        int* h_slot_p = h_meta_pinned + o.max_rows;
        int* h_pos_p = h_meta_pinned + 2 * o.max_rows;
        std::vector<qw38::StepSeg> segs(static_cast<size_t>(M));
        int max_pos = 0;
        for (int s = 0; s < M; ++s) {
            h_tok_p[s] = toks[s];
            h_slot_p[s] = slots[size_t(s)];
            h_pos_p[s] = pos_mtp[s];
            const int rp = pos_mtp[s] + slot_mrope_delta[size_t(slots[size_t(s)])];
            h_rope_pinned[s] = make_int3(rp, rp, rp);
            if (pos_mtp[s] > max_pos) max_pos = pos_mtp[s];
            segs[size_t(s)] = {slots[size_t(s)], s, pos_mtp[s], 1};
        }
        QW38_CK(cudaMemcpyAsync(d_tok, h_tok_p, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_slot, h_slot_p, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_pos, h_pos_p, size_t(M) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_rope, h_rope_pinned, size_t(M) * sizeof(int3), cudaMemcpyHostToDevice, stream));

        // Arm 3-collective fabric step (1 layer = 2 dev AR + 1 head AR)
        if (step_begin) step_begin(M, false, 1, M);

        const size_t part_stride = size_t(M) * geom::HID;
        const size_t hrb = head_row_bytes(o.nc);
        const size_t rank_pkt_bytes = size_t(M) * hrb;

        auto run_mtp_gpu = [&]() {
            qw38::k_gather_embed<<<M, 256, 0, stream>>>(d_tok, embed, mtp_sc.emb_raw, M);
            qw38::k_rmsnorm_1pw<geom::HID><<<M, 256, 0, stream>>>(mtp_sc.emb_raw, mtp_w.pre_fc_norm_emb, mtp_sc.emb_norm, M, 1, geom::RMS_EPS);
            qw38::k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, stream>>>(hid_in, mtp_w.pre_fc_norm_hid, mtp_sc.hid_norm, M, geom::HC, geom::RMS_EPS);

            if (M <= 20 && mtp_w.fp8_fc_emb.q) {
                qw38::launch_fp8_gemv_smallm<false>(mtp_sc.emb_norm, mtp_w.fp8_fc_emb, mtp_sc.emb_proj, M, stream);
            } else {
                cublasSetStream(blas, stream);
                qw38::gemm_bf16_nt(blas, M, geom::HID, geom::HID, mtp_sc.emb_norm, mtp_w.fc_emb, mtp_sc.emb_proj);
            }
            if (M * geom::HC <= 20 && mtp_w.fp8_fc_hid.q) {
                qw38::launch_fp8_gemv_smallm<false>(mtp_sc.hid_norm, mtp_w.fp8_fc_hid, mtp_sc.hid_proj, M * geom::HC, stream);
            } else {
                cublasSetStream(blas, stream);
                qw38::gemm_bf16_nt(blas, M * geom::HC, geom::HID, geom::HID, mtp_sc.hid_norm, mtp_w.fc_hid, mtp_sc.hid_proj);
            }
            qw38::k_mtp_fuse_stream4<<<M, 256, 0, stream>>>(mtp_sc.hid_proj, mtp_sc.emb_proj, mtp_sc.mtp_hyper, M);

            const bool use_direct_ar = (o.nr > 1 && o.nloc == 1 && direct_send_dst && direct_ar_wait && direct_hc_inject);
        const bool use_direct_send = (use_direct_ar && (M > 8 || !direct_copy_ring_wait));
            int coll = 0;

            // L048 Full Attention + AR + Inject
            qw38::hc_prepare(blas, mtp_w.attn_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                qw38::full_attn_forward(blas, mtp_w.attn[size_t(i)], mtp_w.kv[size_t(i)], attn_sc, mixed, p_dst,
                                        d_slot, d_pos, segs.data(), int(segs.size()), M, o.nr, r, max_pps, max_blocks, stream, 0, d_rope);
            }
            if (use_direct_ar && use_direct_send) {
                direct_ar_wait(coll, part_stride * 2);
                direct_hc_inject(coll, mtp_sc.mtp_hyper, hc_sc.inj, M, nullptr);
                ++coll;
            } else if (o.nr > 1 && gather_inject) {
                gather_inject(coll++, mtp_sc.mtp_hyper, parts, hc_sc.inj, M);
            } else if (use_direct_ar) {
                const __nv_bfloat16* own_p = parts + size_t(o.rank0) * part_stride;
                direct_copy_ring_wait(coll, own_p, part_stride * 2);
                direct_hc_inject(coll, mtp_sc.mtp_hyper, hc_sc.inj, M, own_p);
                ++coll;
            } else {
                if (o.nr > 1 && gather_dev) gather_dev(coll++, parts, part_stride * 2);
                else ++coll;
                qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(mtp_sc.mtp_hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            }

            // L048 MoE (GPU-native FP8/BF16!) + AR + Inject
            qw38::hc_prepare(blas, mtp_w.mlp_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                qw38::mtp_moe_forward(blas, mtp_w.moe[size_t(i)], mtp_w.fp8_exp_gu[size_t(i)], mtp_w.fp8_exp_dn[size_t(i)],
                                      moe_sc, mtp_sc, mixed, p_dst, M, stream);
            }
            if (use_direct_ar && use_direct_send) {
                direct_ar_wait(coll, part_stride * 2);
                direct_hc_inject(coll, mtp_sc.mtp_hyper, hc_sc.inj, M, nullptr);
                ++coll;
            } else if (o.nr > 1 && gather_inject) {
                gather_inject(coll++, mtp_sc.mtp_hyper, parts, hc_sc.inj, M);
            } else if (use_direct_ar) {
                const __nv_bfloat16* own_p = parts + size_t(o.rank0) * part_stride;
                direct_copy_ring_wait(coll, own_p, part_stride * 2);
                direct_hc_inject(coll, mtp_sc.mtp_hyper, hc_sc.inj, M, own_p);
                ++coll;
            } else {
                if (o.nr > 1 && gather_dev) gather_dev(coll++, parts, part_stride * 2);
                else ++coll;
                qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(mtp_sc.mtp_hyper, parts, hc_sc.inj, M, o.nr, part_stride);
            }

            if (hid_out) {
                QW38_CK(cudaMemcpyAsync(hid_out, mtp_sc.mtp_hyper, size_t(M) * geom::HC_D * sizeof(__nv_bfloat16),
                                        cudaMemcpyDeviceToDevice, stream));
            }

            // MTP Final HC -> shared lm_head -> head_pack_fast -> D2H
            qw38::hc_prepare(blas, mtp_w.final_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
            constexpr int HEAD_CHUNK = 256;
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                for (int r0 = 0; r0 < M; r0 += HEAD_CHUNK) {
                    const int mc = std::min(M - r0, HEAD_CHUNK);
                    const __nv_bfloat16* mx_ptr = mixed + size_t(r0) * geom::HID;
                    if (M <= 20 && i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q) {
                        qw38::launch_fp8_gemv_smallm<true>(mx_ptr, fp8_lm_head[size_t(i)], head_logits, mc, stream);
                    } else if (M <= 8) {
                        qw38::launch_bf16_gemv_smallm<true>(mx_ptr, lm_head[size_t(i)], head_logits, mc, v_rows[size_t(i)], geom::HID, stream);
                    } else {
                        cublasSetStream(blas, stream);
                        qw38::gemm_bf16_nt_f32(blas, mc, v_rows[size_t(i)], geom::HID, mx_ptr, lm_head[size_t(i)], head_logits);
                    }
                    qw38::head_pack_fast(head_logits, mc, v_rows[size_t(i)], v_lo[size_t(i)], o.nc, head_sc,
                                         head_dev_pkt + size_t(r0) * hrb, hrb, stream);
                }
                QW38_CK(cudaMemcpyAsync(head_host_parts + size_t(r) * rank_pkt_bytes,
                                        head_dev_pkt, rank_pkt_bytes, cudaMemcpyDeviceToHost, stream));
            }
        };

        const int hin_kind = (hid_in == last_hyper) ? 0 : ((hid_in == mtp_hid_cur) ? 1 : ((hid_in == mtp_hid_in) ? 2 : -1));
        const bool mtp_graph_ok = o.use_graph && !o.profile && M <= 16 && hin_kind >= 0 && hid_out == mtp_hid_cur &&
                                  (o.nr == 1 || o.nloc == o.nr || static_cast<bool>(step_parity));
        if (mtp_graph_ok) {
            const int need_qsa = (((max_pos + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS) ? 1 : 0;
            const int ns_attn = qw38::gqa_split_count(M, geom::nh_kv(o.nr, o.rank0), max_pos, attn_sc.split_o != nullptr);
            const int par = step_parity ? step_parity() : 0;
            const MtpGraphKey mk{M, need_qsa, ns_attn, par, hin_kind};
            auto it = mtp_graphs.find(mk);
            if (it == mtp_graphs.end()) {
                cudaGraph_t mg = nullptr;
                cudaGraphExec_t mexec = nullptr;
                QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                run_mtp_gpu();
                QW38_CK(cudaStreamEndCapture(stream, &mg));
                QW38_CK(cudaGraphInstantiate(&mexec, mg, nullptr, nullptr, 0));
                QW38_CK(cudaGraphDestroy(mg));
                mtp_graphs.emplace(mk, mexec);
                QW38_CK(cudaGraphLaunch(mexec, stream));
            } else {
                QW38_CK(cudaGraphLaunch(it->second, stream));
            }
        } else {
            run_mtp_gpu();
        }

        QW38_CK(cudaStreamSynchronize(stream));
        if (o.nr > 1 && gather_host) gather_host(2, head_host_parts, rank_pkt_bytes);
        if (step_end) step_end();

        out.resize(size_t(M));
        std::vector<uint8_t> row_parts(size_t(o.nr) * hrb);
        for (int m = 0; m < M; ++m) {
            for (int r = 0; r < o.nr; ++r) {
                std::memcpy(row_parts.data() + size_t(r) * hrb,
                            head_host_parts + size_t(r) * rank_pkt_bytes + size_t(m) * hrb, hrb);
            }
            out[size_t(m)] = head_merge(row_parts.data(), hrb, o.nr, o.nc);
            if (samples) out[size_t(m)].tok = sample_row(out[size_t(m)].top, samples[m]);
            else if (!out[size_t(m)].top.empty()) out[size_t(m)].tok = out[size_t(m)].top[0].id;
        }
    }

    // Roll back slot `slot`'s recurrent/conv/idx_raw state to exact post-token-`snap_idx` snapshot (`0 <= snap_idx < o.mtp`).
    void mtp_rollback_slot(int slot, int snap_idx) {
        if (has_ple && ple_st.conv_hist_snap) {
            const size_t pb = size_t(geom::PLE_NHIST) * geom::HC_D * 2;
            QW38_CK(cudaMemcpyAsync(
                ple_st.conv_hist + size_t(slot) * geom::PLE_NHIST * geom::HC_D,
                ple_st.conv_hist_snap + (size_t(slot) * size_t(o.mtp) + size_t(snap_idx)) * geom::PLE_NHIST * geom::HC_D,
                pb, cudaMemcpyDeviceToDevice, stream));
        }
        for (auto& y : L) {
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                auto& rl = y.rk[size_t(i)];
                if (y.is_full) {
                    if (rl.kv.idx_raw_snap) {
                        const size_t ib = 4 * geom::IDX_DIM * 2;
                        QW38_CK(cudaMemcpyAsync(
                            rl.kv.idx_raw + size_t(slot) * 4 * geom::IDX_DIM,
                            rl.kv.idx_raw_snap + (size_t(slot) * size_t(o.mtp) + size_t(snap_idx)) * 4 * geom::IDX_DIM,
                            ib, cudaMemcpyDeviceToDevice, stream));
                    }
                    if (rl.kv.idx_raw_thw_snap && rl.kv.idx_raw_thw) {
                        QW38_CK(cudaMemcpyAsync(
                            rl.kv.idx_raw_thw + size_t(slot),
                            rl.kv.idx_raw_thw_snap + size_t(slot) * size_t(o.mtp) + size_t(snap_idx),
                            sizeof(int3), cudaMemcpyDeviceToDevice, stream));
                    }
                } else {
                    const size_t nh_v = size_t(geom::gdn_vh(o.nr));
                    const size_t qkv_l = size_t(geom::gdn_qkv_local(o.nr, r));
                    if (rl.gdn_st.S_snap) {
                        const size_t sb = nh_v * 128 * 128 * sizeof(float);
                        QW38_CK(cudaMemcpyAsync(
                            rl.gdn_st.S + size_t(slot) * nh_v * 128 * 128,
                            rl.gdn_st.S_snap + (size_t(slot) * size_t(o.mtp) + size_t(snap_idx)) * nh_v * 128 * 128,
                            sb, cudaMemcpyDeviceToDevice, stream));
                    }
                    if (rl.gdn_st.conv_snap) {
                        const size_t cb = geom::GDN_NHIST * qkv_l * 2;
                        QW38_CK(cudaMemcpyAsync(
                            rl.gdn_st.conv + size_t(slot) * geom::GDN_NHIST * qkv_l,
                            rl.gdn_st.conv_snap + (size_t(slot) * size_t(o.mtp) + size_t(snap_idx)) * geom::GDN_NHIST * qkv_l,
                            cb, cudaMemcpyDeviceToDevice, stream));
                    }
                }
            }
        }
    }

    // Draft K tokens per session for an arbitrary subset of S2 `slots`, returning flat [slots.size() * K] token IDs.
    // `pos_in[i]` is the last committed trunk position q (whose hyper state is in `last_hyper[slots[i]]`),
    // and `tok_in[i]` is the golden token at position q + 1. Preserves L048 idx_raw across speculative k >= 1 steps.
    void mtp_draft_k(const std::vector<int>& slots, const std::vector<int>& pos_in, const std::vector<int>& tok_in,
                     const std::vector<RowSample>& base_sm, int K, std::vector<int>& out_flat) {
        const size_t S = slots.size();
        out_flat.assign(S * size_t(K), 0);
        if (S == 0 || K <= 0) return;

        bool contiguous = true;
        for (size_t i = 0; i < S; ++i)
            if (slots[i] != int(i)) { contiguous = false; break; }
        const __nv_bfloat16* hin0 = last_hyper;
        if (!contiguous) {
            for (size_t i = 0; i < S; ++i) {
                QW38_CK(cudaMemcpyAsync(mtp_hid_in + i * geom::HC_D,
                                        last_hyper + size_t(slots[i]) * geom::HC_D,
                                        geom::HC_D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            }
            hin0 = mtp_hid_in;
        }

        std::vector<int> cur_tok = tok_in;
        std::vector<int> cur_pos = pos_in;
        std::vector<RowSample> dsm(S);
        std::vector<RowOut> dout;
        const size_t idx_slot_b = 4 * geom::IDX_DIM * sizeof(__nv_bfloat16);

        for (int k = 0; k < K; ++k) {
            for (size_t i = 0; i < S; ++i) {
                dsm[i] = base_sm[i];
                dsm[i].spos = uint32_t(pos_in[i] + k + 2);
            }
            const __nv_bfloat16* hin = (k == 0) ? hin0 : mtp_hid_cur;
            mtp_draft_step(slots, cur_tok.data(), cur_pos.data(), hin, mtp_hid_cur, dsm.data(), dout);
            if (k == 0 && K > 1) {
                // Save post-k0 (real committed pos_in) idx_raw before speculative k >= 1 steps touch it
                for (int li = 0; li < o.nloc; ++li) {
                    auto& kv = mtp_w.kv[size_t(li)];
                    if (kv.idx_raw_snap) {
                        for (size_t i = 0; i < S; ++i) {
                            const size_t off = size_t(slots[i]) * 4 * geom::IDX_DIM;
                            QW38_CK(cudaMemcpyAsync(kv.idx_raw_snap + off, kv.idx_raw + off, idx_slot_b, cudaMemcpyDeviceToDevice, stream));
                        }
                    }
                    if (kv.idx_raw_thw_snap && kv.idx_raw_thw) {
                        for (size_t i = 0; i < S; ++i) {
                            QW38_CK(cudaMemcpyAsync(kv.idx_raw_thw_snap + size_t(slots[i]), kv.idx_raw_thw + size_t(slots[i]),
                                                    sizeof(int3), cudaMemcpyDeviceToDevice, stream));
                        }
                    }
                }
            }
            for (size_t i = 0; i < S; ++i) {
                out_flat[i * size_t(K) + size_t(k)] = dout[i].tok;
                cur_tok[i] = dout[i].tok;
                cur_pos[i] += 1;
            }
        }
        if (K > 1) {
            for (int li = 0; li < o.nloc; ++li) {
                auto& kv = mtp_w.kv[size_t(li)];
                if (kv.idx_raw_snap) {
                    for (size_t i = 0; i < S; ++i) {
                        const size_t off = size_t(slots[i]) * 4 * geom::IDX_DIM;
                        QW38_CK(cudaMemcpyAsync(kv.idx_raw + off, kv.idx_raw_snap + off, idx_slot_b, cudaMemcpyDeviceToDevice, stream));
                    }
                }
                if (kv.idx_raw_thw_snap && kv.idx_raw_thw) {
                    for (size_t i = 0; i < S; ++i) {
                        QW38_CK(cudaMemcpyAsync(kv.idx_raw_thw + size_t(slots[i]), kv.idx_raw_thw_snap + size_t(slots[i]),
                                                sizeof(int3), cudaMemcpyDeviceToDevice, stream));
                    }
                }
            }
        }
    }

    // Execute one full MTP speculative round of `K` drafts + 1 batched trunk verify across `S` sessions!
    // `cur_tok[s]` is the token to feed at `pos[s]`; `last_hyper[s]` holds trunk hyper at `pos[s] - 1`.
    // Appends 1..K+1 exact accepted tokens to `emitted[s]` and updates `cur_tok[s]`, `pos[s]`, `last_hyper[s]`.
    void mtp_round(int S, int K, std::vector<int>& cur_tok, std::vector<int>& pos,
                   float inv_temp, float top_p, uint32_t seed,
                   std::vector<std::vector<int>>& emitted,
                   int& total_drafted, int& total_accepted) {
        std::vector<int> slots(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s) slots[size_t(s)] = s;

        // 1. Draft K tokens per session using Layer-48 MTP
        std::vector<std::vector<int>> drafts(static_cast<size_t>(S), std::vector<int>(static_cast<size_t>(K), 0));
        std::vector<int> d_tok_in = cur_tok;
        std::vector<int> d_pos_in(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s) d_pos_in[size_t(s)] = pos[size_t(s)] - 1;
        std::vector<RowOut> dout;
        std::vector<RowSample> dsm(static_cast<size_t>(S));

        for (int k = 0; k < K; ++k) {
            for (int s = 0; s < S; ++s) dsm[size_t(s)] = RowSample{inv_temp, top_p, seed, uint32_t(pos[size_t(s)] + k + 1)};
            const __nv_bfloat16* hin = (k == 0) ? last_hyper : mtp_hid_cur;
            mtp_draft_step(slots, d_tok_in.data(), d_pos_in.data(), hin, mtp_hid_cur, dsm.data(), dout);
            for (int s = 0; s < S; ++s) {
                drafts[size_t(s)][size_t(k)] = dout[size_t(s)].tok;
                d_tok_in[size_t(s)] = dout[size_t(s)].tok;
                d_pos_in[size_t(s)] += 1;
            }
        }
        total_drafted += S * K;

        // 2. Single batched trunk verify of M = S * (K + 1) tokens with exact post-token-t snapshots!
        const auto saved_ple_ctx = ple_ctx;
        const int V = K + 1;
        std::vector<qw38::StepSeg> segs(static_cast<size_t>(S));
        std::vector<int> vtoks(static_cast<size_t>(S * V));
        std::vector<RowSample> vsm(static_cast<size_t>(S * V));
        for (int s = 0; s < S; ++s) {
            segs[size_t(s)] = {s, s * V, pos[size_t(s)], V};
            vtoks[size_t(s * V)] = cur_tok[size_t(s)];
            vsm[size_t(s * V)] = RowSample{inv_temp, top_p, seed, uint32_t(pos[size_t(s)] + 1)};
            for (int k = 0; k < K; ++k) {
                vtoks[size_t(s * V + 1 + k)] = drafts[size_t(s)][size_t(k)];
                vsm[size_t(s * V + 1 + k)] = RowSample{inv_temp, top_p, seed, uint32_t(pos[size_t(s)] + 2 + k)};
            }
        }
        std::vector<RowOut> vout;
        forward(segs, vtoks.data(), vsm.data(), vout, false, K);

        // 3. Exact greedy verification + commit/rollback + MTP KV sync!
        for (int s = 0; s < S; ++s) {
            int n_acc = 0;
            for (int k = 0; k < K; ++k) {
                if (vout[size_t(s * V + k)].tok == drafts[size_t(s)][size_t(k)]) ++n_acc;
                else break;
            }
            total_accepted += n_acc;

            // Emitted tokens: accepted drafts [0..n_acc-1] + trunk token at row n_acc
            for (int k = 0; k < n_acc; ++k) emitted[size_t(s)].push_back(drafts[size_t(s)][size_t(k)]);
            const int bonus = vout[size_t(s * V + n_acc)].tok;
            emitted[size_t(s)].push_back(bonus);

            // Roll back slot if partial accept (< K)
            if (n_acc < K) {
                mtp_rollback_slot(s, n_acc);
                // Recompute exact host ple_ctx for the (n_acc + 1) actually committed tokens
                ple_ctx[size_t(s)] = saved_ple_ctx[size_t(s)];
                const int n_comm = n_acc + 1;
                if (n_comm >= 2) {
                    ple_ctx[size_t(s)][0] = vtoks[size_t(s * V + n_comm - 2)];
                    ple_ctx[size_t(s)][1] = vtoks[size_t(s * V + n_comm - 1)];
                } else {
                    ple_ctx[size_t(s)][0] = ple_ctx[size_t(s)][1];
                    ple_ctx[size_t(s)][1] = vtoks[size_t(s * V)];
                }
            }

            // Sync MTP KV cache with real trunk hyper states for accepted positions [pos..pos+n_acc-1]
            if (n_acc > 0) {
                mtp_sync_kv(s, s * V, pos[size_t(s)], n_acc, &drafts[size_t(s)][0]);
            }
            mtp_save_last_hyper(s, s * V + n_acc);
            cur_tok[size_t(s)] = bonus;
            pos[size_t(s)] += n_acc + 1;
        }
    }
};

} // namespace qw38rt
