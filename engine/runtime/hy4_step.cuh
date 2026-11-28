// Tencent Hy4-flash runtime step engine (hy4rt::Engine).
// Upgrades:
//   - 2-stage Blackwell MXFP8 dense quantization (GR8_HY4_MX8_DENSE=1): loads part=1 permanent tensors and
//     part=2 BF16 dense linears separately, quantizes dense projections to MXFP8 (E4M3 + UE8M0 1x32) on GPU,
//     and frees the ~10.5 GiB BF16 dense slab BEFORE allocating the KV pool (~5.1 GiB VRAM reclaimed + 2x dense BW!).
//   - Multi-precision paged DSA KV cache (GR8_HY4_KV_FORMAT=fp8|int4|bf16):
//       fp8  (default): 7.91 GiB / 1M (-44.5% VRAM vs BF16) + FP8 compressed indexer keys
//       int4 (KV4):     5.05 GiB / 1M (-64.6% VRAM vs BF16) + FP8 compressed indexer keys
//       bf16:          14.25 GiB / 1M (lossless rollback)
//   - Built-in 0.85B DFly2 speculative drafter (GR8_HY4_DFLY=1): 5 layers reusing target DSA layers
//     [25, 29, 33, 37, 41] KV cache & snapshotted top-2048 indices, lossless greedy + rejection-sampling verify,
//     and exact idx_raw + Engram context rollback on partial verify accept.
//   - 5-segment CUDA Graphs (GR8_HY4_GRAPHS=1) split at the 4 Engram layers so async NVMe io_uring prefetching
//     overlaps GPU graph execution.
#pragma once
#include "../model_geom_hy4.h"
#include "../knobs-rs/gr8_knobs.h"
#include "../kernels/hy4_hc.cuh"
#include "../kernels/hy4_mxfp8.cuh"
#include "../engram/hy4_engram.h"
#include "../engram/hy4_engram.cuh"
#include "../attn/hy4_attn.cuh"
#include "../kernels/hy4_moe.cuh"
#include "../kernels/hy4_dfly.cuh"
#include "../kernels/ink2_head.cuh"
#include "hy4_vision.cuh"
#include "image_input.h"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"

#include <algorithm>
#include <array>
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
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace hy4rt {

// Hy4-flash interleaved 3D M-RoPE position delta after all images in a prompt:
// Each image of patch grid (vit_h, vit_w) has H = vit_h/2, W = vit_w/2, span = H * (W + 1)
// (the +1 column is image_newline per row) and advances the 1D text position cursor by max(H, W + 1).
inline int hy4_mrope_delta(const std::vector<g6rt::ImageIn>& images) {
    int delta = 0;
    for (const auto& im : images) {
        const int H = im.vit_h / hy4v::MERGE;
        const int W = im.vit_w / hy4v::MERGE;
        const int span = H * (W + 1);
        delta += std::max(H, W + 1) - span;
    }
    return delta;
}

// Build per-row 3D M-RoPE (T, H, W) positions for prompt slice [pos0, pos0 + len).
inline void hy4_build_mrope(const std::vector<g6rt::ImageIn>& images, int pos0, int len, int3* out) {
    int cur_delta = 0;
    size_t img_i = 0;
    while (img_i < images.size() && images[img_i].pos + images[img_i].span <= pos0) {
        const int H = images[img_i].vit_h / hy4v::MERGE;
        const int W = images[img_i].vit_w / hy4v::MERGE;
        cur_delta += std::max(H, W + 1) - images[img_i].span;
        ++img_i;
    }
    for (int t = 0; t < len; ++t) {
        const int p = pos0 + t;
        while (img_i < images.size() && p >= images[img_i].pos + images[img_i].span) {
            const int H = images[img_i].vit_h / hy4v::MERGE;
            const int W = images[img_i].vit_w / hy4v::MERGE;
            cur_delta += std::max(H, W + 1) - images[img_i].span;
            ++img_i;
        }
        if (img_i < images.size() && p >= images[img_i].pos && p < images[img_i].pos + images[img_i].span) {
            const auto& im = images[img_i];
            const int W = im.vit_w / hy4v::MERGE;
            const int k = p - im.pos;
            const int r = k / (W + 1);
            const int c = k % (W + 1);
            const int start = im.pos + cur_delta;
            out[t] = make_int3(start, start + r, start + c);
        } else {
            const int rpos = p + cur_delta;
            out[t] = make_int3(rpos, rpos, rpos);
        }
    }
}

__global__ void k_hy4_bench_seed_bf16(__nv_bfloat16* dst, size_t n, uint32_t seed) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
        const float v = (static_cast<float>(int32_t(h & 0xffffu) - 32768)) * (1.0f / 65536.0f);
        dst[i] = __float2bfloat16(v);
    }
}

__global__ void k_hy4_bench_seed_u8(uint8_t* dst, size_t n, uint32_t seed) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352du;
        // Small benign E4M3FN byte (sign + exponent 3..6 + mantissa -> |x| ~ 0.06..0.5)
        dst[i] = static_cast<uint8_t>(((h & 1u) << 7) | (3u + ((h >> 1) & 3u)) << 3 | ((h >> 3) & 7u));
    }
}

__global__ void k_hy4_bench_seed_half(__half* dst, size_t n, float val) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const __half hv = __float2half_rn(val);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = hv;
    }
}

// Per-slot isolated scatter of post-final-norm hidden states [M, 3072] into dfly_target_hidden[d_snap_dst[m]].
// 100% CUDA-Graph-capture-safe AND prevents cross-slot clobbering during multi-session serving!
// Grid: (M), Block: 256.
__global__ void __launch_bounds__(256) k_hy4_scatter_hidden(
    const __nv_bfloat16* __restrict__ mixed_ln,
    const int* __restrict__ d_snap_dst,
    __nv_bfloat16* __restrict__ dst_hidden,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int di = d_snap_dst[m];
    if (di < 0) return;
    const uint4* src = reinterpret_cast<const uint4*>(mixed_ln + static_cast<size_t>(m) * geom::HID);
    uint4*       dst = reinterpret_cast<uint4*>(dst_hidden + static_cast<size_t>(di) * geom::HID);
    for (int i = threadIdx.x; i < geom::HID / 8; i += 256) dst[i] = src[i];
}

// Replay accepted idx_tk rows [0..n_acc-1] onto idx_raw[slot] starting from pos0 (exact partial-accept rollback!).
__global__ void __launch_bounds__(128) k_hy4_idx_raw_replay(
    const __nv_bfloat16* __restrict__ idx_raw_bak, // [4, 128]
    const __nv_bfloat16* __restrict__ idx_tk_ver,  // [n_acc, 128]
    __nv_bfloat16* __restrict__ idx_raw_dst,       // [4, 128]
    int pos0, int n_acc) {
    const int d = threadIdx.x;
    for (int k = 0; k < 4; ++k) idx_raw_dst[k * 128 + d] = idx_raw_bak[k * 128 + d];
    for (int t = 0; t < n_acc; ++t) {
        const int sub = (pos0 + t) & 3;
        idx_raw_dst[sub * 128 + d] = idx_tk_ver[static_cast<size_t>(t) * 128 + d];
    }
}

[[noreturn]] inline void die(const std::string& m) {
    std::fprintf(stderr, "hy4: %s\n", m.c_str());
    std::exit(2);
}

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}
#define HY4_CK(x) ::hy4rt::ck((x), #x)

template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    HY4_CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    HY4_CK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
    return static_cast<T*>(p);
}

inline hy4::KvFmt parse_kv_fmt(const std::string& s) {
    if (s == "bf16" || s == "BF16") return hy4::KvFmt::BF16;
    if (s == "fp8"  || s == "FP8")  return hy4::KvFmt::FP8;
    if (s == "int4" || s == "INT4" || s == "kv4") return hy4::KvFmt::INT4;
    die("bad --kv-format / GR8_HY4_KV_FORMAT '" + s + "' (want bf16|fp8|int4)");
}

inline const char* kv_fmt_name(hy4::KvFmt f) {
    if (f == hy4::KvFmt::FP8)  return "fp8";
    if (f == hy4::KvFmt::INT4) return "int4";
    return "bf16";
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

inline void build_probs(const std::vector<Cand>& top, const RowSample& s,
                        std::vector<double>& p, size_t& n_eff) {
    p.clear();
    n_eff = 0;
    if (top.empty()) return;
    if (!(s.inv_temp > 0.f) || top.size() == 1) {
        p.assign(1, 1.0);
        n_eff = 1;
        return;
    }
    p.resize(top.size());
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
    const double inv = (total > 0.0) ? (1.0 / total) : 0.0;
    for (size_t i = 0; i < n; ++i) p[i] *= inv;
    n_eff = n;
}

inline int sample_row(const std::vector<Cand>& top, const RowSample& s) {
    if (top.empty()) return 0;
    if (!(s.inv_temp > 0.f) || top.size() == 1) return top[0].id;
    std::vector<double> p;
    size_t n = 0;
    build_probs(top, s, p, n);
    const double u = sample_u(s.seed, s.spos);
    double cum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cum += p[i];
        if (u < cum) return top[i].id;
    }
    return top[n - 1].id;
}

// Exact lossless speculative verification for 1 step (greedy or rejection sampling when inv_temp > 0).
// Returns true if `draft_tok` is accepted; if rejected, sets `replacement_tok` from max(0, p_target - p_draft).
inline bool spec_verify_token(int draft_tok, const RowOut& target_row, const RowOut* draft_row,
                              const RowSample& s, int& replacement_tok) {
    if (!(s.inv_temp > 0.f) || draft_row == nullptr) {
        if (target_row.tok == draft_tok) return true;
        replacement_tok = target_row.tok;
        return false;
    }
    std::vector<double> pt, pd;
    size_t nt = 0, nd = 0;
    build_probs(target_row.top, s, pt, nt);
    build_probs(draft_row->top, s, pd, nd);
    double p_t = 0.0, p_d = 0.0;
    for (size_t i = 0; i < nt; ++i) if (target_row.top[i].id == draft_tok) { p_t = pt[i]; break; }
    for (size_t i = 0; i < nd; ++i) if (draft_row->top[i].id == draft_tok) { p_d = pd[i]; break; }
    if (p_t >= p_d && p_t > 0.0) return true;
    const double u = sample_u(s.seed ^ 0x517cc1b7u, s.spos);
    if (p_d > 0.0 && u * p_d < p_t) return true;

    // Residual sample from max(0, p_target - p_draft)
    std::vector<double> res(nt, 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < nt; ++i) {
        const int id = target_row.top[i].id;
        double qv = 0.0;
        for (size_t j = 0; j < nd; ++j) if (draft_row->top[j].id == id) { qv = pd[j]; break; }
        const double diff = std::max(0.0, pt[i] - qv);
        res[i] = diff;
        sum += diff;
    }
    if (sum <= 0.0) { replacement_tok = target_row.tok; return false; }
    const double ur = sample_u(s.seed ^ 0x9e3779b9u, s.spos) * sum;
    double cum = 0.0;
    for (size_t i = 0; i < nt; ++i) {
        cum += res[i];
        if (ur < cum) { replacement_tok = target_row.top[i].id; return false; }
    }
    replacement_tok = target_row.top[nt - 1].id;
    return false;
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
            if (cudaMemcpyAsync(s.dev + off, src, n, cudaMemcpyHostToDevice, st) != cudaSuccess) return 2;
            return cudaStreamSynchronize(st) == cudaSuccess ? 0 : 2;
        }
    };

    ~Image() {
        if (plan) g6l_plan_free(plan);
        if (dev) cudaFree(dev);
    }

    void load_part(const std::string& release_dir, const std::string& widths_path, int nr, int r, uint32_t part) {
        rank = r;
        char mf[256];
        std::snprintf(mf, sizeof(mf), "/tmp/hy4-manifest-p%d-r%d-pt%u.json", static_cast<int>(getpid()), r, part);
        char err[512] = {};
        plan = g6l_hy4_plan_part(release_dir.c_str(), widths_path.c_str(), static_cast<uint32_t>(nr),
                                 static_cast<uint32_t>(r), part, mf, err, sizeof(err));
        if (!plan) die("g6l_hy4_plan_part(r" + std::to_string(r) + ", part=" + std::to_string(part) + "): " + err);

        std::string perr;
        const bool pok = pack.open_manifest(mf, perr);
        ::unlink(mf);
        if (!pok) die("open_manifest(r" + std::to_string(r) + "): " + perr);
        HY4_CK(cudaMalloc(&dev, pack.bytes()));

        Sink sink;
        sink.dev = dev;
        const g6l_sink csink{&sink, Sink::alloc, Sink::release, Sink::copy};
        g6l_stats st{};
        const int rc = g6l_plan_load(plan, &csink, &st, err, sizeof(err));
        for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
        if (rc != 0) die("g6l_plan_load(r" + std::to_string(r) + "): " + err);
        g6l_plan_free(plan);
        plan = nullptr;
        static const char* pnames[4] = {"full", "perm", "dense-staging", "dfly"};
        std::fprintf(stderr, "# rank %d: hy4[%s] loaded %.2f GiB in %.2f s (%u threads)\n",
                     r, pnames[part < 4 ? part : 0], st.image_bytes / double(1ull << 30), st.seconds, st.threads);
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
    template <class T> T* get_mut(const std::string& name) const {
        return reinterpret_cast<T*>(dev + entry(name).offset);
    }
    template <class T> const T* opt(const std::string& name) const {
        const g6::Entry* e = pack.find(name);
        return e ? reinterpret_cast<const T*>(dev + e->offset) : nullptr;
    }
    template <class T> void read_host(const std::string& name, std::vector<T>& out) const {
        const g6::Entry& e = entry(name);
        out.resize(e.nbytes / sizeof(T));
        HY4_CK(cudaMemcpy(out.data(), dev + e.offset, e.nbytes, cudaMemcpyDeviceToHost));
    }
};

// ------------------------------------------------------------------------------------------------ Options & Profile
struct Options {
    int nr = 1;
    int rank0 = 0;
    int nloc = 1;                  // nr for m4sim, 1 for g6m4
    int slots = 1;
    int max_rows = 512;
    int cap = 16384;               // max sequence length per slot (up to 1048576+)
    int pool_rows = 0;             // global dynamic KV pool rows (0 -> auto = slots * cap)
    int nc = 256;
    std::string widths_path;
    std::string engram_dir;
    size_t engram_cache_mib = 2048;
    bool engram_cache_set = false;
    int engram_uring = 128;
    bool engram_direct = true;
    bool engram_mmap = false;
    bool engram_async = true;
    bool no_engram = false;
    bool profile = false;
    std::string dump_dir;

    hy4::KvFmt kv_fmt = hy4::KvFmt::FP8;
    bool kv_fmt_set = false;
    bool mx8_dense = true;
    bool mx8_set = false;
    bool use_dfly = true;
    bool dfly_set = false;
    int dfly_tokens = 5;
    bool dfly_tokens_set = false;
    bool use_graphs = true;
    bool graphs_set = false;
};

struct ProfileAccum {
    double hc_ms = 0, eng_io_ms = 0, eng_gpu_ms = 0, attn_ms = 0, moe_ms = 0, ar_ms = 0, head_ms = 0, dfly_ms = 0;
    int steps = 0, tokens = 0, dfly_steps = 0;
    void print(int rank) const {
        if (!steps) return;
        const double tot = hc_ms + eng_io_ms + eng_gpu_ms + attn_ms + moe_ms + ar_ms + head_ms;
        std::fprintf(stderr,
                     "# hy4 profile r%d (%d steps, %d tok, %.2f ms total, %.2f ms/step): "
                     "hc=%.2f eng_io=%.2f eng_gpu=%.2f attn=%.2f moe=%.2f ar=%.2f head=%.2f",
                     rank, steps, tokens, tot, tot / steps,
                     hc_ms / steps, eng_io_ms / steps, eng_gpu_ms / steps,
                     attn_ms / steps, moe_ms / steps, ar_ms / steps, head_ms / steps);
        if (dfly_steps > 0) std::fprintf(stderr, " | dfly=%.2f ms/draft (%d drafts)", dfly_ms / dfly_steps, dfly_steps);
        std::fprintf(stderr, "\n");
    }
};

// ------------------------------------------------------------------------------------------------ Engine
struct Engine {
    Options o;
    cublasHandle_t blas = nullptr;
    void* cublas_ws = nullptr;
    cudaStream_t stream = 0;
    std::vector<std::unique_ptr<Image>> img;
    std::vector<hy4::mx8::Mx8Arena> mx8_arena;

    int max_pps = 0;
    int pool_pages = 0;
    int max_groups = 0;

    // Global weights
    const __nv_bfloat16* embed = nullptr;
    hy4::HcHeadWeights hc_head;
    const __nv_bfloat16* final_norm = nullptr;
    std::vector<const __nv_bfloat16*> lm_head;
    std::vector<int> v_lo, v_rows;

    // 4-Layer Engram NVMe Store + async background prefetch thread
    hy4::EngramStore engram_store;
    bool has_engram = false;
    std::array<hy4::EngramWeights, 4> eng_w{};
    hy4::EngramScratch eng_sc{};
    std::vector<std::array<int32_t, 2>> eng_ctx;
    uint8_t* eng_pinned[4] = {};
    std::thread eng_thr;
    std::mutex eng_mu;
    std::condition_variable eng_cv, eng_done_cv;
    bool eng_stop = false;
    struct EngJob { int slot_idx = -1; int M = 0; std::vector<uint64_t> ids; bool done = true; bool ok = true; };
    std::array<EngJob, 4> eng_jobs{};
    int eng_queue_head = 0, eng_queue_tail = 0;
    std::array<int, 8> eng_queue{};

    struct RankLayer {
        hy4::AttnWeights attn_w;
        hy4::DsaLayerKv dsa_kv;
        hy4::SwaLayerKv swa_kv;
        hy4::DenseMlpWeights dense_w;
        hy4::MoeWeights moe_w;
    };
    struct Layer {
        int l = 0;
        bool is_dsa = false;
        int dsa_idx = -1; // 0..11 for DSA layers
        int dfly_idx = -1; // 0..4 for target DSA layers [25, 29, 33, 37, 41]
        bool is_moe = false;
        int eng_slot = -1;
        hy4::HcWeights attn_hc;
        hy4::HcWeights ffn_hc;
        const __nv_bfloat16* in_ln = nullptr;
        const __nv_bfloat16* post_attn_ln = nullptr;
        std::vector<RankLayer> rk;
    };
    std::vector<Layer> L;

    // Built-in DFly2 0.85B Speculative Drafter
    bool has_dfly = false;
    int dfly_w_max = 0;
    int dfly_snap_cap = 0;
    std::vector<std::unique_ptr<Image>> dfly_img;     // kept only when !mx8_dense
    std::vector<hy4::mx8::Mx8Arena> dfly_mx8_arena;   // MXFP8 DFly2 weights (~0.81 GiB!)
    std::vector<__nv_bfloat16*> dfly_perm_bf16;       // tiny permanent DFly2 norms/gate (~144 KiB!)
    std::vector<float*> dfly_sinks_f32;               // 5 * nh_loc FP32 sinks
    std::vector<hy4::dfly::RankWeights> dfly_w;
    std::vector<hy4::mx8::LinW> dfly_lm_head;         // MXFP8 lm_head for DFly2 drafting only (100% lossless target verify!)
    hy4::dfly::Scratch dfly_sc{};
    std::array<int*, hy4::dfly::NUM_LAYERS> dfly_snap_grp{}; // [dfly_snap_cap, 511]
    std::array<int*, hy4::dfly::NUM_LAYERS> dfly_snap_cnt{}; // [dfly_snap_cap]
    std::array<int*, hy4::dfly::NUM_LAYERS> dfly_snap_pos{}; // [dfly_snap_cap]
    __nv_bfloat16* dfly_target_hidden = nullptr;             // [dfly_snap_cap, 3072]
    // Exact idx_raw rollback state for verify partial accepts (12 DSA layers x nloc)
    std::vector<__nv_bfloat16*> idx_raw_bak;     // [12 * nloc], each [slots, 4, 128]
    std::vector<int3*>          idx_raw_thw_bak; // [12 * nloc], each [slots]
    std::vector<__nv_bfloat16*> idx_tk_ver;      // [12 * nloc], each [dfly_snap_cap, 128]

    // Native Hy4-flash vision tower + multimodal M-RoPE / Engram mask state
    Hy4Vision vision;
    std::vector<int> slot_mrope_delta;

    // Shared step scratch buffers
    __nv_bfloat16* hyper = nullptr;
    __nv_bfloat16* mixed_ln = nullptr;
    __nv_bfloat16* parts = nullptr;
    uint8_t* mx8_x8 = nullptr;
    hy4::HcScratch hc_sc{};
    hy4::AttnScratch attn_sc{};
    hy4::DenseMlpScratch dense_sc{};
    hy4::MoeScratch moe_sc{};

    float* head_logits = nullptr;
    uint8_t* head_dev_pkt = nullptr;
    uint8_t* head_host_parts = nullptr;
    int* d_tok = nullptr;
    int* d_slot = nullptr;
    int* d_pos = nullptr;
    int* d_rope_pos = nullptr;
    int* d_snap_dst = nullptr;
    int3* d_mrope = nullptr;
    uint8_t* d_eng_mask = nullptr;
    uint8_t* h_eng_mask_pinned = nullptr;
    int* h_meta_pinned = nullptr;
    int* d_page_table = nullptr;
    std::vector<int> h_page_table;

    // 5-segment CUDA Graphs for decode/verify M <= 12 (split at 4 Engram layers!)
    struct GraphKey {
        int M = 0;
        bool big_kv = false;
        bool sparse = false;
        int idx_bucket = 0;
        bool snap = false;
        bool save_tk = false;
        bool operator<(const GraphKey& r) const {
            return std::tie(M, big_kv, sparse, idx_bucket, snap, save_tk) <
                   std::tie(r.M, r.big_kv, r.sparse, r.idx_bucket, r.snap, r.save_tk);
        }
    };
    struct SegGraph {
        std::array<cudaGraphExec_t, 5> exec{};
        bool ready = false;
    };
    std::map<GraphKey, SegGraph> graphs;
    std::map<int, cudaGraphExec_t> dfly_graphs; // keyed by W (full DFly2 body + K greedy steps in 1 graph!)
    hy4::dfly::DflyGraphMeta* h_dfly_meta_pinned = nullptr;

    // Optional collective hooks (set by g6m4_hy4 for TP=2 over RDMA)
    std::function<void(int M, bool skip_head)> on_step_begin;
    std::function<void(int c, int M, const __nv_bfloat16* post_ln_w)> on_ar_norm_inject;
    std::function<void(int c, int M, uint8_t* host_parts, size_t row_bytes)> on_head_gather;
    std::function<void()> on_step_end;
    std::function<void(int M, bool dev_argmax)> on_dfly_begin;
    std::function<void(int c, int M, const __nv_bfloat16* post_ln_w)> on_dfly_ar_add;
    std::function<void(int c, const hy4::dfly::DflyArgPair* local_pair, int* dst_id)> on_dfly_argmax_tp2;
    std::function<void(int c, int M, uint8_t* host_parts, size_t row_bytes)> on_dfly_head_gather;
    std::function<void()> on_dfly_end;

    ProfileAccum prof{};
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;

    ~Engine() {
        clear_graphs();
        if (eng_thr.joinable()) {
            {
                std::lock_guard<std::mutex> lk(eng_mu);
                eng_stop = true;
            }
            eng_cv.notify_all();
            eng_thr.join();
        }
        for (int s = 0; s < 4; ++s) if (eng_pinned[s]) cudaFreeHost(eng_pinned[s]);
        if (head_host_parts) cudaFreeHost(head_host_parts);
        if (h_meta_pinned) cudaFreeHost(h_meta_pinned);
        if (h_eng_mask_pinned) cudaFreeHost(h_eng_mask_pinned);
        if (h_dfly_meta_pinned) cudaFreeHost(h_dfly_meta_pinned);
        for (auto* p : dfly_perm_bf16) if (p) cudaFree(p);
        for (auto* p : dfly_sinks_f32) if (p) cudaFree(p);
        if (ev_a) cudaEventDestroy(ev_a);
        if (ev_b) cudaEventDestroy(ev_b);
        if (blas) cublasDestroy(blas);
        if (cublas_ws) cudaFree(cublas_ws);
        if (stream) cudaStreamDestroy(stream);
    }

    void clear_graphs() {
        for (auto& kv : graphs) {
            for (auto& ex : kv.second.exec) {
                if (ex) { cudaGraphExecDestroy(ex); ex = nullptr; }
            }
        }
        graphs.clear();
        for (auto& kv : dfly_graphs) {
            if (kv.second) { cudaGraphExecDestroy(kv.second); kv.second = nullptr; }
        }
        dfly_graphs.clear();
    }

    static std::string lname(int l, const std::string& suf) {
        char b[96];
        std::snprintf(b, sizeof(b), "L%03d.%s", l, suf.c_str());
        return b;
    }

    static std::string dname(int l, const std::string& suf) {
        char b[96];
        std::snprintf(b, sizeof(b), "dfly.L%d.%s", l, suf.c_str());
        return b;
    }

    void eng_worker_loop() {
        while (true) {
            int sidx = -1;
            {
                std::unique_lock<std::mutex> lk(eng_mu);
                eng_cv.wait(lk, [&]() { return eng_stop || eng_queue_head != eng_queue_tail; });
                if (eng_stop) return;
                sidx = eng_queue[eng_queue_head & 7];
                ++eng_queue_head;
            }
            auto& job = eng_jobs[size_t(sidx)];
            const bool ok = engram_store.fetch_rows(sidx, job.ids.data(), job.ids.size(), eng_pinned[sidx]);
            {
                std::lock_guard<std::mutex> lk(eng_mu);
                job.ok = ok;
                job.done = true;
            }
            eng_done_cv.notify_all();
        }
    }

    void eng_enqueue(int sidx, int M, const std::vector<uint64_t>& ids) {
        if (!has_engram) return;
        if (!o.engram_async) {
            if (!engram_store.fetch_rows(sidx, ids.data(), ids.size(), eng_pinned[sidx]))
                die("engram_store.fetch_rows failed on slot " + std::to_string(sidx));
            eng_jobs[size_t(sidx)].done = true;
            eng_jobs[size_t(sidx)].ok = true;
            return;
        }
        {
            std::lock_guard<std::mutex> lk(eng_mu);
            eng_jobs[size_t(sidx)].slot_idx = sidx;
            eng_jobs[size_t(sidx)].M = M;
            eng_jobs[size_t(sidx)].ids = ids;
            eng_jobs[size_t(sidx)].done = false;
            eng_jobs[size_t(sidx)].ok = true;
            eng_queue[eng_queue_tail & 7] = sidx;
            ++eng_queue_tail;
        }
        eng_cv.notify_one();
    }

    void eng_wait(int sidx) {
        if (!has_engram || !o.engram_async) return;
        std::unique_lock<std::mutex> lk(eng_mu);
        eng_done_cv.wait(lk, [&]() { return eng_jobs[size_t(sidx)].done; });
        if (!eng_jobs[size_t(sidx)].ok) die("async engram fetch failed on slot " + std::to_string(sidx));
    }

    void reset_slot(int slot) {
        if (slot < 0 || slot >= o.slots) return;
        eng_ctx[size_t(slot)] = {geom::ENGRAM_PAD_ID, geom::ENGRAM_PAD_ID};
        if (size_t(slot) < slot_mrope_delta.size()) slot_mrope_delta[size_t(slot)] = 0;
    }

    void set_slot_mrope_delta(int slot, int delta) {
        if (slot >= 0 && size_t(slot) < slot_mrope_delta.size()) slot_mrope_delta[size_t(slot)] = delta;
    }

    bool load_vision(const std::string& path) { vision.load(path, std::max(o.max_rows, dfly_w_max)); return vision.ready(); }
    bool vision_ready() const { return vision.ready(); }
    const std::string& vision_id() const { return vision.id(); }
    size_t vision_device_bytes() const { return vision.device_bytes(); }

    void seed_bench_kv(int fill_pos, int sessions) {
        if (fill_pos <= 0) return;
        if (fill_pos >= o.cap) die("seed_bench_kv: fill_pos >= cap");
        const int need_pages = (fill_pos + geom::KV_PAGE - 1) / geom::KV_PAGE;
        if (need_pages * sessions > pool_pages)
            die("seed_bench_kv: sessions * pages exceeds pool_pages (" + std::to_string(need_pages * sessions) + " > " + std::to_string(pool_pages) + ")");
        for (int l = 0; l < geom::L_MAIN; ++l) {
            for (int i = 0; i < o.nloc; ++i) {
                auto& rl = L[size_t(l)].rk[size_t(i)];
                if (L[size_t(l)].is_dsa) {
                    const size_t n_idx = size_t(pool_pages) * 16 * geom::IDX_DIM;
                    if (o.kv_fmt == hy4::KvFmt::BF16) {
                        k_hy4_bench_seed_bf16<<<static_cast<unsigned>((n_idx + 255) / 256), 256, 0, stream>>>(
                            rl.dsa_kv.idx_bf16, n_idx, 0x12345678u + uint32_t(l * 17 + i));
                        const size_t n_kv = size_t(pool_pages) * geom::KV_PAGE * geom::KVA;
                        k_hy4_bench_seed_bf16<<<4096, 256, 0, stream>>>(
                            reinterpret_cast<__nv_bfloat16*>(rl.dsa_kv.kv_pool), n_kv, 0x9abcdef0u + uint32_t(l * 31 + i));
                    } else {
                        k_hy4_bench_seed_u8<<<4096, 256, 0, stream>>>(
                            rl.dsa_kv.idx_q, n_idx, 0x12345678u + uint32_t(l * 17 + i));
                        k_hy4_bench_seed_half<<<1024, 256, 0, stream>>>(
                            rl.dsa_kv.idx_sc, size_t(pool_pages) * 16 * 4, 0.05f);
                        const size_t n_kv_b = size_t(pool_pages) * geom::KV_PAGE * hy4::dsa_kv_tok_bytes(o.kv_fmt);
                        k_hy4_bench_seed_u8<<<4096, 256, 0, stream>>>(
                            rl.dsa_kv.kv_pool, n_kv_b, 0x9abcdef0u + uint32_t(l * 31 + i));
                    }
                } else {
                    const size_t n_sw = size_t(o.slots) * geom::SWA_RING * geom::KVA;
                    k_hy4_bench_seed_bf16<<<static_cast<unsigned>((n_sw + 255) / 256), 256, 0, stream>>>(
                        rl.swa_kv.ring, n_sw, 0x55aa33ccu + uint32_t(l * 13 + i));
                }
            }
        }
        HY4_CK(cudaStreamSynchronize(stream));
    }

    void set_slot_pages(int slot, const std::vector<int>& phys_pages) {
        if (slot < 0 || slot >= o.slots) return;
        const int n = std::min(static_cast<int>(phys_pages.size()), max_pps);
        if (n <= 0) return;
        std::memcpy(h_page_table.data() + static_cast<size_t>(slot) * max_pps, phys_pages.data(),
                    static_cast<size_t>(n) * sizeof(int));
        HY4_CK(cudaMemcpyAsync(d_page_table + static_cast<size_t>(slot) * max_pps,
                               phys_pages.data(), static_cast<size_t>(n) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    }

    void init(const std::string& release_dir, const Options& opts) {
        o = opts;
        if (o.nr != 1 && o.nr != 2 && o.nr != 4) die("unsupported TP nr=" + std::to_string(o.nr));
        if (o.widths_path.empty()) o.widths_path = release_dir + "/manifests/k3.json";
        if (o.engram_dir.empty()) o.engram_dir = release_dir + "/engram";

        // Apply typed knobs (CLI flags override knobs when explicitly set!)
        if (!o.kv_fmt_set)        o.kv_fmt = parse_kv_fmt(knob::hy4_kv_format());
        if (!o.mx8_set)           o.mx8_dense = knob::hy4_mx8_dense();
        if (!o.dfly_set)          o.use_dfly = knob::hy4_dfly();
        if (!o.dfly_tokens_set)   o.dfly_tokens = std::max(1, static_cast<int>(knob::hy4_dfly_tokens()));
        if (!o.graphs_set)        o.use_graphs = knob::hy4_graphs();
        if (!o.engram_cache_set) {
            if (auto v = knob::hy4_engram_cache_mb()) o.engram_cache_mib = static_cast<size_t>(*v);
        }
        o.engram_mmap   = knob::hy4_engram_mmap();
        o.engram_direct = knob::hy4_engram_direct();
        o.profile       = o.profile || knob::hy4_profile();

        HY4_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) die("cublasCreate failed");
        cublasSetStream(blas, stream);
        HY4_CK(cudaMalloc(&cublas_ws, 32ull << 20));
        cublasSetWorkspace(blas, cublas_ws, 32ull << 20);
        HY4_CK(cudaEventCreate(&ev_a));
        HY4_CK(cudaEventCreate(&ev_b));

        // Pre-configure dynamic shared-memory attributes BEFORE any CUDA Graph capture!
        hy4::ensure_mla_tc_smem();
        hy4::dfly::ensure_dfly_mla_smem();
        hy4::mx8::ensure_large_smem_attr();

        // 1. Stage 1: Load permanent weights (part=1 when MXFP8 dense is enabled, else part=0 all-in-one BF16)
        img.resize(size_t(o.nloc));
        mx8_arena.resize(size_t(o.nloc));
        for (int i = 0; i < o.nloc; ++i) {
            img[size_t(i)] = std::make_unique<Image>();
            img[size_t(i)]->load_part(release_dir, o.widths_path, o.nr, o.rank0 + i, o.mx8_dense ? 1u : 0u);
        }

        // 2. Bind global weights
        const Image& im0 = *img[0];
        embed = im0.get<__nv_bfloat16>("embed.weight");
        hc_head.fn    = im0.get<float>("hc_head.fn");
        hc_head.base  = im0.get<float>("hc_head.base");
        hc_head.scale = im0.get<float>("hc_head.scale");
        final_norm = im0.get<__nv_bfloat16>("norm.weight");

        lm_head.resize(size_t(o.nloc));
        v_lo.resize(size_t(o.nloc));
        v_rows.resize(size_t(o.nloc));
        int max_vr = 0;
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            lm_head[size_t(i)] = img[size_t(i)]->get<__nv_bfloat16>("lm_head.weight");
            const int t_per = geom::VTILES / o.nr;
            v_lo[size_t(i)] = r * t_per * 128;
            v_rows[size_t(i)] = static_cast<int>(img[size_t(i)]->entry("lm_head.weight").shape[0]);
            max_vr = std::max(max_vr, v_rows[size_t(i)]);
        }

        // 3. Bind permanent layer structures & widen PMCGX routed/shared experts
        L.resize(geom::L_MAIN);
        std::array<std::vector<int32_t>, 4> eng_luts;
        int dsa_cnt = 0;
        for (int l = 0; l < geom::L_MAIN; ++l) {
            Layer& y = L[size_t(l)];
            y.l = l;
            y.is_dsa = geom::is_dsa_layer(l);
            y.dsa_idx = y.is_dsa ? (dsa_cnt++) : -1;
            y.dfly_idx = hy4::dfly::target_dsa_to_dfly_idx(l);
            y.is_moe = (l >= geom::FIRST_MOE);
            y.eng_slot = -1;
            for (int es = 0; es < 4; ++es) if (hy4::ENGRAM_LAYER_IDX[es] == l) y.eng_slot = es;

            y.attn_hc.fn    = im0.get<float>(lname(l, "attn_hc.fn"));
            y.attn_hc.base  = im0.get<float>(lname(l, "attn_hc.base"));
            y.attn_hc.scale = im0.get<float>(lname(l, "attn_hc.scale"));
            y.ffn_hc.fn     = im0.get<float>(lname(l, "mlp_hc.fn"));
            y.ffn_hc.base   = im0.get<float>(lname(l, "mlp_hc.base"));
            y.ffn_hc.scale  = im0.get<float>(lname(l, "mlp_hc.scale"));
            y.in_ln         = im0.get<__nv_bfloat16>(lname(l, "input_norm"));
            y.post_attn_ln  = im0.get<__nv_bfloat16>(lname(l, "post_attn_norm"));

            if (y.eng_slot >= 0) {
                auto& ew = eng_w[size_t(y.eng_slot)];
                const __nv_bfloat16* hn_all = im0.get<__nv_bfloat16>(lname(l, "eng.head_norms"));
                for (int h = 0; h < 16; ++h) ew.head_norms[h] = hn_all + size_t(h) * geom::ENGRAM_HEAD_DIM;
                ew.value_b   = im0.get<__nv_bfloat16>(lname(l, "eng.val_b"));
                ew.pre_gamma = im0.get<__nv_bfloat16>(lname(l, "eng.pre_gamma"));
                const __nv_bfloat16* kb_all = im0.get<__nv_bfloat16>(lname(l, "eng.key_b"));
                const __nv_bfloat16* nk_all = im0.get<__nv_bfloat16>(lname(l, "eng.norm_k"));
                const __nv_bfloat16* nq_all = im0.get<__nv_bfloat16>(lname(l, "eng.norm_q"));
                const __nv_bfloat16* hs_all = im0.get<__nv_bfloat16>(lname(l, "eng.hc_scales"));
                for (int hc = 0; hc < 4; ++hc) {
                    ew.key_b[hc]      = kb_all + size_t(hc) * geom::HID;
                    ew.norm_key[hc]   = nk_all + size_t(hc) * geom::HID;
                    ew.norm_query[hc] = nq_all + size_t(hc) * geom::HID;
                    ew.hc_scales[hc]  = hs_all + size_t(hc) * geom::HID;
                }
                im0.read_host<int32_t>(lname(l, "eng.lookup_table"), eng_luts[size_t(y.eng_slot)]);
            }

            y.rk.resize(size_t(o.nloc));
            for (int i = 0; i < o.nloc; ++i) {
                const Image& im = *img[size_t(i)];
                RankLayer& rl = y.rk[size_t(i)];
                const int nh_loc = (y.is_dsa ? geom::HEADS_DSA : geom::HEADS_SWA) / o.nr;
                rl.attn_w.is_dsa       = y.is_dsa;
                rl.attn_w.nh_loc       = nh_loc;
                rl.attn_w.q_a_ln       = im.get<__nv_bfloat16>(lname(l, "attn.q_a_norm"));
                rl.attn_w.kv_a_ln      = im.get<__nv_bfloat16>(lname(l, "attn.kv_a_norm"));
                rl.attn_w.post_proj_ln = im.get<__nv_bfloat16>(lname(l, "attn.post_proj_norm"));
                rl.attn_w.sinks        = im.get<float>(lname(l, "attn.sinks"));

                if (y.is_dsa) {
                    rl.attn_w.idx_hw_proj = im.get<__nv_bfloat16>(lname(l, "idx.head_w"));
                    rl.attn_w.idx_tk_norm = im.get<__nv_bfloat16>(lname(l, "idx.tok_k_norm"));
                    rl.attn_w.idx_kv_proj = im.get<float>(lname(l, "idx.k_val_proj"));
                    rl.attn_w.idx_kg_proj = im.get<float>(lname(l, "idx.k_gate_proj"));
                    rl.attn_w.idx_k_norm  = im.get<float>(lname(l, "idx.k_norm"));
                    rl.attn_w.idx_ape     = im.get<float>(lname(l, "idx.ape"));
                }

                if (!y.is_moe) {
                    rl.dense_w.post_fc2_ln = im.get<__nv_bfloat16>(lname(l, "mlp.post_fc2_norm"));
                    rl.dense_w.dense_i_loc = geom::DENSE_I / o.nr;
                } else {
                    rl.moe_w.router_w    = im.get<__nv_bfloat16>(lname(l, "router.w"));
                    rl.moe_w.router_bias = im.get<float>(lname(l, "router.bias"));
                    rl.moe_w.combine_ln  = im.get<__nv_bfloat16>(lname(l, "moe.combine_norm"));

                    const int eb = geom::EXP_BLOCKS / o.nr;
                    rl.moe_w.routed.fc1  = im.get<uint8_t>(lname(l, "tr3.fc1"));
                    rl.moe_w.routed.fc2  = im.get<uint8_t>(lname(l, "tr3.fc2"));
                    rl.moe_w.routed.tab  = im.get<long long>(lname(l, "tr3.table"));
                    rl.moe_w.routed.kmix = im.opt<long long>(lname(l, "tr3.kmix"));
                    rl.moe_w.routed.sg   = im.get<__half>(lname(l, "tr3.sg"));
                    rl.moe_w.routed.v1b  = im.get<__half>(lname(l, "tr3.v1b"));
                    rl.moe_w.routed.v2   = im.get<__half>(lname(l, "tr3.v2"));
                    rl.moe_w.routed.v3b  = im.get<__half>(lname(l, "tr3.v3b"));
                    rl.moe_w.routed.ne   = geom::NEXP;
                    rl.moe_w.routed.eb   = eb;
                    hy4::k_hy4_tr3_widen_fc1<geom::EHB><<<geom::NEXP * 2 * eb * geom::EHB, 256, 0, stream>>>(
                        im.get_mut<uint8_t>(lname(l, "tr3.fc1")), rl.moe_w.routed.tab, rl.moe_w.routed.kmix, geom::NEXP, eb);
                    hy4::k_hy4_tr3_widen_fc2<geom::EHB><<<geom::NEXP * eb * geom::EHB, 256, 0, stream>>>(
                        im.get_mut<uint8_t>(lname(l, "tr3.fc2")), rl.moe_w.routed.tab, geom::NEXP, eb);

                    rl.moe_w.shared.fc1  = im.get<uint8_t>(lname(l, "sh_tr3.fc1"));
                    rl.moe_w.shared.fc2  = im.get<uint8_t>(lname(l, "sh_tr3.fc2"));
                    rl.moe_w.shared.tab  = im.get<long long>(lname(l, "sh_tr3.table"));
                    rl.moe_w.shared.kmix = im.opt<long long>(lname(l, "sh_tr3.kmix"));
                    rl.moe_w.shared.sg   = im.get<__half>(lname(l, "sh_tr3.sg"));
                    rl.moe_w.shared.v1b  = im.get<__half>(lname(l, "sh_tr3.v1b"));
                    rl.moe_w.shared.v2   = im.get<__half>(lname(l, "sh_tr3.v2"));
                    rl.moe_w.shared.v3b  = im.get<__half>(lname(l, "sh_tr3.v3b"));
                    rl.moe_w.shared.ne   = 1;
                    rl.moe_w.shared.eb   = eb;
                    hy4::k_hy4_tr3_widen_fc1<geom::HB><<<1 * 2 * eb * geom::HB, 256, 0, stream>>>(
                        im.get_mut<uint8_t>(lname(l, "sh_tr3.fc1")), rl.moe_w.shared.tab, rl.moe_w.shared.kmix, 1, eb);
                    hy4::k_hy4_tr3_widen_fc2<geom::HB><<<1 * eb * geom::HB, 256, 0, stream>>>(
                        im.get_mut<uint8_t>(lname(l, "sh_tr3.fc2")), rl.moe_w.shared.tab, 1, eb);
                }
            }
        }

        // 4. Stage 2: Bind or MXFP8-quantize dense linears (freeing BF16 dense staging BEFORE KV allocation!)
        __nv_bfloat16* kv_b_tmp = dalloc<__nv_bfloat16>(size_t(geom::HEADS_SWA) * geom::KVB_HEAD * geom::KV_LORA);
        for (int i = 0; i < o.nloc; ++i) {
            std::unique_ptr<Image> dense_tmp;
            if (o.mx8_dense) {
                dense_tmp = std::make_unique<Image>();
                dense_tmp->load_part(release_dir, o.widths_path, o.nr, o.rank0 + i, 2u);
            }
            const Image& dim = o.mx8_dense ? *dense_tmp : *img[size_t(i)];
            auto& ar = mx8_arena[size_t(i)];

            if (o.mx8_dense) {
                for (int l = 0; l < geom::L_MAIN; ++l) {
                    const Layer& y = L[size_t(l)];
                    const int nh_loc = (y.is_dsa ? geom::HEADS_DSA : geom::HEADS_SWA) / o.nr;
                    ar.reserve(geom::QA, geom::HID);
                    ar.reserve(nh_loc * geom::QK_HEAD, geom::QA);
                    ar.reserve(geom::KVA, geom::HID);
                    ar.reserve(nh_loc * geom::KV_LORA, geom::QK_NOPE);
                    ar.reserve(nh_loc * geom::V_HEAD, geom::KV_LORA);
                    ar.reserve(nh_loc * geom::V_HEAD, geom::HID);
                    ar.reserve(geom::HID, nh_loc * geom::V_HEAD);
                    if (y.is_dsa) {
                        ar.reserve(geom::IDX_HEADS * geom::IDX_DIM, geom::QA);
                        ar.reserve(geom::IDX_DIM, geom::HID);
                    }
                    if (i == 0 && y.eng_slot >= 0) {
                        ar.reserve(geom::HID, geom::ENGRAM_DIM);
                        for (int hc = 0; hc < 4; ++hc) ar.reserve(geom::HID, geom::ENGRAM_DIM);
                    }
                    if (!y.is_moe) {
                        const int di = geom::DENSE_I / o.nr;
                        ar.reserve(di, geom::HID);
                        ar.reserve(di, geom::HID);
                        ar.reserve(geom::HID, di);
                    } else {
                        ar.reserve(geom::EH, geom::HID);
                        ar.reserve(geom::HID, geom::EH);
                    }
                }
                if (!ar.alloc()) die("mx8_arena cudaMalloc failed");
            }

            auto bind_w = [&](const __nv_bfloat16* src, int N, int K) -> hy4::mx8::LinW {
                if (o.mx8_dense) return hy4::mx8::LinW::from_mx8(ar.add(src, N, K, stream));
                return hy4::mx8::LinW::from_bf16(src, N, K);
            };

            for (int l = 0; l < geom::L_MAIN; ++l) {
                Layer& y = L[size_t(l)];
                RankLayer& rl = y.rk[size_t(i)];
                const int nh_loc = rl.attn_w.nh_loc;

                rl.attn_w.q_a_proj  = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.q_a_proj")), geom::QA, geom::HID);
                rl.attn_w.q_b_proj  = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.q_b_proj")), nh_loc * geom::QK_HEAD, geom::QA);
                rl.attn_w.kv_a_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.kv_a_proj")), geom::KVA, geom::HID);
                rl.attn_w.gate_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.linear_gate")), nh_loc * geom::V_HEAD, geom::HID);
                rl.attn_w.o_proj    = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.o_proj")), geom::HID, nh_loc * geom::V_HEAD);

                // Unpack kv_b_proj [nh_loc, 448, 512] -> w_uk_t + w_uv (in-place in dim, then quantize if MX8!)
                __nv_bfloat16* kv_b = dim.get_mut<__nv_bfloat16>(lname(l, "attn.kv_b_proj"));
                const size_t kv_b_el = static_cast<size_t>(nh_loc) * geom::KVB_HEAD * geom::KV_LORA;
                HY4_CK(cudaMemcpyAsync(kv_b_tmp, kv_b, kv_b_el * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                __nv_bfloat16* w_uk_t = kv_b;
                __nv_bfloat16* w_uv   = kv_b + static_cast<size_t>(nh_loc) * geom::KV_LORA * geom::QK_NOPE;
                hy4::k_hy4_unpack_kv_b<<<nh_loc * geom::KVB_HEAD, 256, 0, stream>>>(kv_b_tmp, w_uk_t, w_uv, nh_loc);
                rl.attn_w.w_uk_t = bind_w(w_uk_t, nh_loc * geom::KV_LORA, geom::QK_NOPE);
                rl.attn_w.w_uv   = bind_w(w_uv,   nh_loc * geom::V_HEAD,  geom::KV_LORA);

                if (y.is_dsa) {
                    rl.attn_w.idx_q_proj  = bind_w(dim.get<__nv_bfloat16>(lname(l, "idx.q_proj")), geom::IDX_HEADS * geom::IDX_DIM, geom::QA);
                    rl.attn_w.idx_tk_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "idx.tok_k_proj")), geom::IDX_DIM, geom::HID);
                }
                if (i == 0 && y.eng_slot >= 0) {
                    auto& ew = eng_w[size_t(y.eng_slot)];
                    ew.value_w = bind_w(dim.get<__nv_bfloat16>(lname(l, "eng.val_w")), geom::HID, geom::ENGRAM_DIM);
                    const __nv_bfloat16* kw_all = dim.get<__nv_bfloat16>(lname(l, "eng.key_w"));
                    for (int hc = 0; hc < 4; ++hc) {
                        ew.key_w[hc] = bind_w(kw_all + size_t(hc) * geom::HID * geom::ENGRAM_DIM, geom::HID, geom::ENGRAM_DIM);
                    }
                }
                if (!y.is_moe) {
                    const int di = rl.dense_w.dense_i_loc;
                    rl.dense_w.gate_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "mlp.gate_proj")), di, geom::HID);
                    rl.dense_w.up_proj   = bind_w(dim.get<__nv_bfloat16>(lname(l, "mlp.up_proj")),   di, geom::HID);
                    rl.dense_w.down_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "mlp.down_proj")), geom::HID, di);
                } else {
                    rl.moe_w.routed_down = bind_w(dim.get<__nv_bfloat16>(lname(l, "moe.down_proj")), geom::EH, geom::HID);
                    rl.moe_w.routed_up   = bind_w(dim.get<__nv_bfloat16>(lname(l, "moe.up_proj")),   geom::HID, geom::EH);
                }
            }
            HY4_CK(cudaStreamSynchronize(stream));
            if (o.mx8_dense) {
                std::fprintf(stderr, "# rank %d: quantized dense linears to MXFP8 (%.2f GiB, freed %.2f GiB BF16 staging)\n",
                             o.rank0 + i, ar.total_bytes / double(1ull << 30), dense_tmp->pack.bytes() / double(1ull << 30));
            }
            // dense_tmp is destroyed here -> cudaFree BF16 dense slab BEFORE DFly2 / KV pool!
        }

        // 5. Stage 3: Optional built-in 0.85B DFly2 speculative drafter
        const bool dfly_exists = std::filesystem::exists(release_dir + "/dense/exlr8-dfly.safetensors") ||
                                 std::filesystem::exists(release_dir + "/exlr8-dfly.safetensors");
        if (o.use_dfly && dfly_exists) {
            has_dfly = true;
            dfly_w.resize(size_t(o.nloc));
            dfly_lm_head.resize(size_t(o.nloc));
            dfly_img.resize(size_t(o.nloc));
            dfly_mx8_arena.resize(size_t(o.nloc));
            dfly_perm_bf16.assign(size_t(o.nloc), nullptr);
            dfly_sinks_f32.assign(size_t(o.nloc), nullptr);

            const int df_nh = hy4::dfly::HEADS / o.nr;
            const int df_i  = hy4::dfly::MLP_I / o.nr;
            for (int i = 0; i < o.nloc; ++i) {
                auto tmp = std::make_unique<Image>();
                tmp->load_part(release_dir, o.widths_path, o.nr, o.rank0 + i, 3u);
                const Image& dm = *tmp;
                auto& rw = dfly_w[size_t(i)];
                auto& dar = dfly_mx8_arena[size_t(i)];

                // Copy small permanent DFly2 norms/gate/conf (~144 KiB) so we can free the 1.58 GiB BF16 slab when MX8=1!
                const size_t perm_el = 6 * geom::HID + (geom::HID + 8) + 256 +
                                       size_t(hy4::dfly::NUM_LAYERS) * (3 * geom::HID + geom::QA + geom::KV_LORA);
                dfly_perm_bf16[size_t(i)] = dalloc<__nv_bfloat16>(perm_el);
                dfly_sinks_f32[size_t(i)] = dalloc<float>(size_t(hy4::dfly::NUM_LAYERS) * df_nh);
                size_t pcur = 0;
                auto copy_perm = [&](const std::string& tname, size_t nel) -> const __nv_bfloat16* {
                    pcur = (pcur + 7) & ~size_t(7); // 16-byte aligned for uint2/uint4 vectorized norm loads
                    __nv_bfloat16* dst = dfly_perm_bf16[size_t(i)] + pcur;
                    pcur += nel;
                    HY4_CK(cudaMemcpyAsync(dst, dm.get<__nv_bfloat16>(tname), nel * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToDevice, stream));
                    return dst;
                };

                rw.mtp_enorm      = copy_perm("dfly.mtp.enorm", geom::HID);
                rw.mtp_hnorm      = copy_perm("dfly.mtp.hnorm", geom::HID);
                rw.mtp_gate       = copy_perm("dfly.mtp.gate", geom::HID);
                rw.final_norm     = copy_perm("dfly.final_norm", geom::HID);
                rw.hc_hidden_norm = copy_perm("dfly.hc.hidden_norm", geom::HID);
                rw.hc_embed_norm  = copy_perm("dfly.hc.embed_norm", geom::HID);
                rw.conf_w         = copy_perm("dfly.conf.w", geom::HID);
                rw.conf_b         = copy_perm("dfly.conf.b", 1);

                if (o.mx8_dense) {
                    dar.reserve(geom::HID, 2 * geom::HID); // mtp_eh_proj
                    dar.reserve(hy4::dfly::HC_I, 2 * geom::HID);
                    dar.reserve(hy4::dfly::HC_I, 2 * geom::HID);
                    dar.reserve(geom::HID, hy4::dfly::HC_I);
                    for (int d = 0; d < hy4::dfly::NUM_LAYERS; ++d) {
                        dar.reserve(geom::QA, geom::HID);
                        dar.reserve(df_nh * geom::QK_HEAD, geom::QA);
                        dar.reserve(geom::KVA, geom::HID);
                        dar.reserve(df_nh * geom::KV_LORA, geom::QK_NOPE);
                        dar.reserve(df_nh * geom::V_HEAD, geom::KV_LORA);
                        dar.reserve(df_nh * geom::V_HEAD, geom::HID);
                        dar.reserve(geom::HID, df_nh * geom::V_HEAD);
                        dar.reserve(df_i, geom::HID);
                        dar.reserve(df_i, geom::HID);
                        dar.reserve(geom::HID, df_i);
                    }
                    if (knob::hy4_dfly_mx8_head()) dar.reserve(v_rows[size_t(i)], geom::HID);
                    if (!dar.alloc()) die("dfly_mx8_arena cudaMalloc failed");
                }

                auto bind_df = [&](const __nv_bfloat16* src, int N, int K) -> hy4::mx8::LinW {
                    if (o.mx8_dense) return hy4::mx8::LinW::from_mx8(dar.add(src, N, K, stream));
                    return hy4::mx8::LinW::from_bf16(src, N, K);
                };

                if (o.mx8_dense && knob::hy4_dfly_mx8_head()) {
                    dfly_lm_head[size_t(i)] = hy4::mx8::LinW::from_mx8(dar.add(lm_head[size_t(i)], v_rows[size_t(i)], geom::HID, stream));
                } else {
                    dfly_lm_head[size_t(i)] = hy4::mx8::LinW::from_bf16(lm_head[size_t(i)], v_rows[size_t(i)], geom::HID);
                }
                rw.mtp_eh_proj  = bind_df(dm.get<__nv_bfloat16>("dfly.mtp.eh_proj"), geom::HID, 2 * geom::HID);
                rw.hc_gate_proj = bind_df(dm.get<__nv_bfloat16>("dfly.hc.gate_proj"), hy4::dfly::HC_I, 2 * geom::HID);
                rw.hc_up_proj   = bind_df(dm.get<__nv_bfloat16>("dfly.hc.up_proj"),   hy4::dfly::HC_I, 2 * geom::HID);
                rw.hc_down_proj = bind_df(dm.get<__nv_bfloat16>("dfly.hc.down_proj"), geom::HID, hy4::dfly::HC_I);

                for (int d = 0; d < hy4::dfly::NUM_LAYERS; ++d) {
                    auto& lw = rw.layers[d];
                    lw.nh_loc = df_nh;
                    lw.i_loc  = df_i;
                    lw.input_norm     = copy_perm(dname(d, "input_norm"), geom::HID);
                    lw.post_attn_norm = copy_perm(dname(d, "post_attn_norm"), geom::HID);
                    lw.q_a_ln         = copy_perm(dname(d, "attn.q_a_norm"), geom::QA);
                    lw.kv_a_ln        = copy_perm(dname(d, "attn.kv_a_norm"), geom::KV_LORA);
                    lw.post_proj_ln   = copy_perm(dname(d, "attn.post_proj_norm"), geom::HID);

                    float* s_dst = dfly_sinks_f32[size_t(i)] + size_t(d) * df_nh;
                    hy4::dfly::k_dfly_bf16_to_f32<<<1, 32, 0, stream>>>(dm.get<__nv_bfloat16>(dname(d, "attn.sinks")), s_dst, df_nh);
                    lw.sinks = s_dst;

                    lw.q_a_proj  = bind_df(dm.get<__nv_bfloat16>(dname(d, "attn.q_a_proj")), geom::QA, geom::HID);
                    lw.q_b_proj  = bind_df(dm.get<__nv_bfloat16>(dname(d, "attn.q_b_proj")), df_nh * geom::QK_HEAD, geom::QA);
                    lw.kv_a_proj = bind_df(dm.get<__nv_bfloat16>(dname(d, "attn.kv_a_proj")), geom::KVA, geom::HID);
                    lw.gate_proj = bind_df(dm.get<__nv_bfloat16>(dname(d, "attn.linear_gate")), df_nh * geom::V_HEAD, geom::HID);
                    lw.o_proj    = bind_df(dm.get<__nv_bfloat16>(dname(d, "attn.o_proj")), geom::HID, df_nh * geom::V_HEAD);

                    __nv_bfloat16* kv_b = dm.get_mut<__nv_bfloat16>(dname(d, "attn.kv_b_proj"));
                    const size_t kv_b_el = static_cast<size_t>(df_nh) * geom::KVB_HEAD * geom::KV_LORA;
                    HY4_CK(cudaMemcpyAsync(kv_b_tmp, kv_b, kv_b_el * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                    __nv_bfloat16* w_uk_t = kv_b;
                    __nv_bfloat16* w_uv   = kv_b + static_cast<size_t>(df_nh) * geom::KV_LORA * geom::QK_NOPE;
                    hy4::k_hy4_unpack_kv_b<<<df_nh * geom::KVB_HEAD, 256, 0, stream>>>(kv_b_tmp, w_uk_t, w_uv, df_nh);
                    lw.w_uk_t = bind_df(w_uk_t, df_nh * geom::KV_LORA, geom::QK_NOPE);
                    lw.w_uv   = bind_df(w_uv,   df_nh * geom::V_HEAD,  geom::KV_LORA);

                    lw.mlp_gate = bind_df(dm.get<__nv_bfloat16>(dname(d, "mlp.gate_proj")), df_i, geom::HID);
                    lw.mlp_up   = bind_df(dm.get<__nv_bfloat16>(dname(d, "mlp.up_proj")),   df_i, geom::HID);
                    lw.mlp_down = bind_df(dm.get<__nv_bfloat16>(dname(d, "mlp.down_proj")), geom::HID, df_i);
                }
                HY4_CK(cudaStreamSynchronize(stream));
                if (o.mx8_dense) {
                    std::fprintf(stderr, "# rank %d: DFly2 drafter ready in MXFP8 (%.2f GiB, K=%d)\n",
                                 o.rank0 + i, dar.total_bytes / double(1ull << 30), o.dfly_tokens);
                } else {
                    dfly_img[size_t(i)] = std::move(tmp);
                    std::fprintf(stderr, "# rank %d: DFly2 drafter ready in BF16 (%.2f GiB, K=%d)\n",
                                 o.rank0 + i, dfly_img[size_t(i)]->pack.bytes() / double(1ull << 30), o.dfly_tokens);
                }
            }
        }
        cudaFree(kv_b_tmp);

        // 6. Stage 4: Allocate dynamic multi-precision DSA + SWA KV pool (AFTER BF16 dense slabs are freed!)
        max_pps = (o.cap + geom::KV_PAGE - 1) / geom::KV_PAGE;
        max_groups = (o.cap + geom::IDX_RATIO - 1) / geom::IDX_RATIO;
        const int want_rows = (o.pool_rows > 0) ? o.pool_rows : (o.slots * o.cap);
        pool_pages = (want_rows + geom::KV_PAGE - 1) / geom::KV_PAGE;
        if (pool_pages < max_pps) pool_pages = max_pps;

        const size_t kv_tok_b = hy4::dsa_kv_tok_bytes(o.kv_fmt);
        const size_t idx_grp_b = (o.kv_fmt == hy4::KvFmt::BF16) ? (geom::IDX_DIM * 2) : (geom::IDX_DIM + 4 * 2); // 256B or 136B
        const size_t dsa_kv_per_page = size_t(geom::KV_PAGE) * kv_tok_b + size_t(16) * idx_grp_b;
        const size_t tot_dsa_bytes = size_t(geom::N_DSA) * size_t(pool_pages) * dsa_kv_per_page * size_t(o.nloc);
        const size_t tot_swa_bytes = size_t(geom::N_SWA) * size_t(o.slots) * geom::SWA_RING * geom::KVA * 2 * size_t(o.nloc);
        std::fprintf(stderr, "# hy4 KV pool [%s]: cap=%d (max_pps=%d), pool_pages=%d (%d tok), slots=%d | DSA=%.2f GiB + SWA=%.3f GiB\n",
                     kv_fmt_name(o.kv_fmt), o.cap, max_pps, pool_pages, pool_pages * geom::KV_PAGE, o.slots,
                     tot_dsa_bytes / double(1ull << 30), tot_swa_bytes / double(1ull << 30));

        for (int l = 0; l < geom::L_MAIN; ++l) {
            Layer& y = L[size_t(l)];
            for (int i = 0; i < o.nloc; ++i) {
                RankLayer& rl = y.rk[size_t(i)];
                if (y.is_dsa) {
                    rl.dsa_kv.fmt     = o.kv_fmt;
                    rl.dsa_kv.kv_pool = dalloc<uint8_t>(size_t(pool_pages) * geom::KV_PAGE * kv_tok_b);
                    if (o.kv_fmt == hy4::KvFmt::BF16) {
                        rl.dsa_kv.idx_bf16 = dalloc<__nv_bfloat16>(size_t(pool_pages) * 16 * geom::IDX_DIM);
                    } else {
                        rl.dsa_kv.idx_q  = dalloc<uint8_t>(size_t(pool_pages) * 16 * geom::IDX_DIM);
                        rl.dsa_kv.idx_sc = dalloc<__half>(size_t(pool_pages) * 16 * 4);
                    }
                    rl.dsa_kv.idx_raw = dalloc<__nv_bfloat16>(size_t(o.slots) * 4 * geom::IDX_DIM);
                    rl.dsa_kv.idx_raw_thw = dalloc<int3>(size_t(o.slots));
                } else {
                    rl.swa_kv.ring = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::SWA_RING * geom::KVA);
                }
            }
        }

        // 7. Open 4-Layer Engram Store + start async worker thread
        eng_ctx.assign(size_t(o.slots), {geom::ENGRAM_PAD_ID, geom::ENGRAM_PAD_ID});
        slot_mrope_delta.assign(size_t(o.slots), 0);
        if (!o.no_engram) {
            hy4::EngramStoreOpts eopts{};
            eopts.cache_bytes = o.engram_cache_mib << 20;
            eopts.io_uring = o.engram_uring;
            eopts.direct = o.engram_direct;
            eopts.use_mmap = o.engram_mmap;
            if (auto cd = knob::test_hy4_engram_compact_dir()) eopts.compact_dir = *cd;
            std::string eerr;
            if (!engram_store.open(o.engram_dir, eng_luts, eopts, eerr)) die(eerr);
            has_engram = true;
            for (int s = 0; s < 4; ++s) {
                HY4_CK(cudaHostAlloc(&eng_pinned[s], size_t(o.max_rows) * geom::ENGRAM_HEADS * geom::ENGRAM_ROW_BYTES,
                                     cudaHostAllocDefault));
            }
            if (o.engram_async) eng_thr = std::thread(&Engine::eng_worker_loop, this);
        }

        // 8. Allocate shared scratch buffers
        dfly_w_max = std::max(2, o.dfly_tokens + 1);
        const int Mmax = std::max(o.max_rows, dfly_w_max);
        const int max_nh = geom::HEADS_SWA / o.nr;
        const int eb = geom::EXP_BLOCKS / o.nr;
        const int dense_i_loc = geom::DENSE_I / o.nr;
        const int max_k = std::max({geom::DENSE_I / o.nr, max_nh * geom::V_HEAD, 2 * geom::HID, hy4::dfly::MLP_I / o.nr});

        hyper    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HC_D);
        mixed_ln = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        parts    = dalloc<__nv_bfloat16>(size_t(o.nr) * Mmax * geom::HID);
        mx8_x8   = dalloc<uint8_t>(hy4::mx8::x8_bytes(Mmax, max_k));

        hc_sc.flat_f32 = nullptr; // eliminated by direct BF16 k_hy4_hc_dot_tiled
        hc_sc.mixes    = dalloc<float>(size_t(Mmax) * geom::HC_MIX);
        hc_sc.post     = dalloc<float>(size_t(Mmax) * geom::HC);

        eng_sc.raw_rows = dalloc<uint8_t>(size_t(Mmax) * geom::ENGRAM_HEADS * geom::ENGRAM_ROW_BYTES);
        eng_sc.emb      = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::ENGRAM_DIM);
        eng_sc.val      = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        eng_sc.main_sig = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        eng_sc.key      = dalloc<__nv_bfloat16>(size_t(geom::HC) * Mmax * geom::HID);

        attn_sc.q_a      = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::QA);
        attn_sc.q_resid  = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::QA);
        attn_sc.q_raw    = dalloc<__nv_bfloat16>(size_t(Mmax) * max_nh * geom::QK_HEAD);
        attn_sc.q_nope_h = dalloc<__nv_bfloat16>(size_t(max_nh) * Mmax * geom::QK_NOPE);
        attn_sc.q_abs_h  = dalloc<__nv_bfloat16>(size_t(max_nh) * Mmax * geom::KV_LORA);
        attn_sc.q_full   = dalloc<__nv_bfloat16>(size_t(Mmax) * max_nh * geom::KVA);
        attn_sc.kv_raw   = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::KVA);
        attn_sc.gate     = dalloc<__nv_bfloat16>(size_t(Mmax) * max_nh * geom::V_HEAD);
        attn_sc.u_h      = dalloc<__nv_bfloat16>(size_t(max_nh) * Mmax * geom::KV_LORA);
        attn_sc.v_h      = dalloc<__nv_bfloat16>(size_t(max_nh) * Mmax * geom::V_HEAD);
        attn_sc.attn_out = dalloc<__nv_bfloat16>(size_t(Mmax) * max_nh * geom::V_HEAD);
        attn_sc.kv_b_tmp = dalloc<__nv_bfloat16>(size_t(max_nh) * geom::KV_LORA * geom::V_HEAD);
        const int max_dec_M = std::max(16, dfly_w_max);
        attn_sc.split_m  = dalloc<float>(size_t(max_dec_M) * hy4::MLA_MAX_SPLITS * max_nh);
        attn_sc.split_l  = dalloc<float>(size_t(max_dec_M) * hy4::MLA_MAX_SPLITS * max_nh);
        attn_sc.split_o  = dalloc<float>(size_t(max_dec_M) * hy4::MLA_MAX_SPLITS * max_nh * geom::KV_LORA);
        attn_sc.idx_tk   = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::IDX_DIM);
        attn_sc.idx_q    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::IDX_HEADS * geom::IDX_DIM);
        attn_sc.idx_hw   = dalloc<float>(size_t(Mmax) * geom::IDX_HEADS);
        attn_sc.idx_sc   = dalloc<float>(size_t(hy4::IDX_SCORE_BATCH) * max_groups);
        attn_sc.sel_grp  = dalloc<int>(size_t(Mmax) * (geom::IDX_GROUP_TOPK - 1));
        attn_sc.sel_cnt  = dalloc<int>(size_t(Mmax));

        dense_sc.gate = dalloc<__nv_bfloat16>(size_t(Mmax) * dense_i_loc);
        dense_sc.up   = dalloc<__nv_bfloat16>(size_t(Mmax) * dense_i_loc);

        moe_sc.router_logits = dalloc<float>(size_t(Mmax) * geom::NEXP);
        moe_sc.topk_idx      = dalloc<int>(size_t(Mmax) * geom::TOPK);
        moe_sc.topk_w        = dalloc<float>(size_t(Mmax) * geom::TOPK);
        moe_sc.routed_in     = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::EH);
        moe_sc.routed_lat    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::EH);
        moe_sc.routed_up_out = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        moe_sc.shared_out    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        moe_sc.d_n_groups    = dalloc<int>(1);
        moe_sc.grp_exp       = dalloc<int>(size_t(Mmax) * geom::TOPK);
        moe_sc.grp_cnt       = dalloc<int>(size_t(Mmax) * geom::TOPK);
        moe_sc.grp_routes    = dalloc<int>((size_t(Mmax) * geom::TOPK + geom::NEXP) * 16);
        moe_sc.x_had_r       = dalloc<__half>(size_t(Mmax) * geom::EHB * 128);
        moe_sc.x_had_s       = dalloc<__half>(size_t(Mmax) * geom::HB * 128);
        moe_sc.fc1_act_r     = dalloc<__half>(size_t(Mmax) * geom::TOPK * eb * 128);
        moe_sc.fc1_act_s     = dalloc<__half>(size_t(Mmax) * eb * 128);
        moe_sc.fc2_part_r    = dalloc<float>(size_t(Mmax) * geom::TOPK * geom::EH);

        if (has_dfly) {
            dfly_snap_cap = o.slots * dfly_w_max;
            dfly_sc.max_W      = dfly_w_max;
            dfly_sc.h          = dalloc<__nv_bfloat16>(size_t(dfly_w_max) * geom::HID);
            dfly_sc.x_ln       = dalloc<__nv_bfloat16>(size_t(dfly_w_max) * geom::HID);
            dfly_sc.kv_draft   = dalloc<__nv_bfloat16>(size_t(dfly_w_max) * geom::KVA);
            dfly_sc.mlp_gate   = dalloc<__nv_bfloat16>(size_t(dfly_w_max) * (hy4::dfly::MLP_I / o.nr));
            dfly_sc.mlp_up     = dalloc<__nv_bfloat16>(size_t(dfly_w_max) * (hy4::dfly::MLP_I / o.nr));
            dfly_sc.cat6144    = dalloc<__nv_bfloat16>(2 * geom::HID);
            dfly_sc.tmp3072    = dalloc<__nv_bfloat16>(geom::HID);
            dfly_sc.hc_gate    = dalloc<__nv_bfloat16>(hy4::dfly::HC_I);
            dfly_sc.hc_up      = dalloc<__nv_bfloat16>(hy4::dfly::HC_I);
            dfly_sc.d_q_pos    = dalloc<int>(size_t(dfly_w_max));
            dfly_sc.d_draft_id = dalloc<int>(size_t(dfly_w_max));
            dfly_sc.d_max_v    = dalloc<float>(256);
            dfly_sc.d_max_i    = dalloc<int>(256);
            dfly_sc.d_meta     = dalloc<hy4::dfly::DflyGraphMeta>(1);
            dfly_sc.d_arg_pair = dalloc<hy4::dfly::DflyArgPair>(1);
            HY4_CK(cudaMallocHost(&h_dfly_meta_pinned, sizeof(hy4::dfly::DflyGraphMeta)));

            constexpr int KSEL = geom::IDX_GROUP_TOPK - 1;
            for (int d = 0; d < hy4::dfly::NUM_LAYERS; ++d) {
                dfly_snap_grp[size_t(d)] = dalloc<int>(size_t(dfly_snap_cap) * KSEL);
                dfly_snap_cnt[size_t(d)] = dalloc<int>(size_t(dfly_snap_cap));
                dfly_snap_pos[size_t(d)] = dalloc<int>(size_t(dfly_snap_cap));
            }
            dfly_target_hidden = dalloc<__nv_bfloat16>(size_t(dfly_snap_cap) * geom::HID);
            idx_raw_bak.resize(size_t(geom::N_DSA * o.nloc));
            idx_raw_thw_bak.resize(size_t(geom::N_DSA * o.nloc));
            idx_tk_ver.resize(size_t(geom::N_DSA * o.nloc));
            for (int k = 0; k < geom::N_DSA * o.nloc; ++k) {
                idx_raw_bak[size_t(k)]     = dalloc<__nv_bfloat16>(size_t(o.slots) * 4 * geom::IDX_DIM);
                idx_raw_thw_bak[size_t(k)] = dalloc<int3>(size_t(o.slots));
                idx_tk_ver[size_t(k)]      = dalloc<__nv_bfloat16>(size_t(dfly_snap_cap) * geom::IDX_DIM);
            }
        }

        head_logits  = dalloc<float>(size_t(Mmax) * max_vr);
        head_dev_pkt = dalloc<uint8_t>(size_t(Mmax) * head_row_bytes(o.nc));
        HY4_CK(cudaHostAlloc(&head_host_parts, size_t(o.nr) * Mmax * head_row_bytes(o.nc), cudaHostAllocDefault));
        HY4_CK(cudaHostAlloc(&h_meta_pinned, size_t(5) * Mmax * sizeof(int), cudaHostAllocDefault));
        HY4_CK(cudaHostAlloc(&h_eng_mask_pinned, size_t(Mmax), cudaHostAllocDefault));
        d_tok      = dalloc<int>(size_t(Mmax));
        d_slot     = dalloc<int>(size_t(Mmax));
        d_pos      = dalloc<int>(size_t(Mmax));
        d_rope_pos = dalloc<int>(size_t(Mmax));
        d_snap_dst = dalloc<int>(size_t(Mmax));
        d_mrope    = dalloc<int3>(size_t(Mmax));
        d_eng_mask = dalloc<uint8_t>(size_t(Mmax));

        h_page_table.assign(size_t(o.slots) * max_pps, 0);
        for (int s = 0; s < o.slots; ++s) {
            for (int p = 0; p < max_pps; ++p) {
                h_page_table[size_t(s) * max_pps + p] = (s * max_pps + p) % pool_pages;
            }
        }
        d_page_table = dalloc<int>(size_t(o.slots) * max_pps);
        HY4_CK(cudaMemcpy(d_page_table, h_page_table.data(), h_page_table.size() * sizeof(int), cudaMemcpyHostToDevice));

        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr, "# hy4 ready: TP=%d rank0=%d kv=%s mx8=%d dfly=%d(K=%d) graphs=%d VRAM used=%.2f / %.2f GiB (free=%.2f GiB)\n",
                     o.nr, o.rank0, kv_fmt_name(o.kv_fmt), o.mx8_dense ? 1 : 0,
                     has_dfly ? 1 : 0, has_dfly ? o.dfly_tokens : 0, o.use_graphs ? 1 : 0,
                     (total_b - free_b) / double(1ull << 30),
                     total_b / double(1ull << 30), free_b / double(1ull << 30));
    }

    void dump_tensor(const std::string& name, const void* dptr, size_t nbytes) {
        if (o.dump_dir.empty()) return;
        std::filesystem::create_directories(o.dump_dir);
        std::vector<uint8_t> buf(nbytes);
        HY4_CK(cudaStreamSynchronize(stream));
        HY4_CK(cudaMemcpy(buf.data(), dptr, nbytes, cudaMemcpyDeviceToHost));
        std::ofstream f(o.dump_dir + "/" + name, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(nbytes));
    }

    void prepare_engram_step(const std::vector<hy4::StepSeg>& segs, const int* toks, int M) {
        if (!has_engram) return;
        std::vector<std::array<int32_t, 3>> tri(static_cast<size_t>(M));
        for (const auto& sg : segs) {
            int32_t c0 = (sg.pos0 == 0) ? geom::ENGRAM_PAD_ID : eng_ctx[size_t(sg.slot)][0];
            int32_t c1 = (sg.pos0 == 0) ? geom::ENGRAM_PAD_ID : eng_ctx[size_t(sg.slot)][1];
            for (int t = 0; t < sg.len; ++t) {
                const int m = sg.row0 + t;
                const int pos = sg.pos0 + t;
                const int32_t tok = toks[m];
                const int32_t t1 = (pos >= 1) ? c1 : geom::ENGRAM_PAD_ID;
                const int32_t t2 = (pos >= 2) ? c0 : geom::ENGRAM_PAD_ID;
                tri[size_t(m)] = {tok, t1, t2};
                c0 = c1;
                c1 = tok;
            }
            eng_ctx[size_t(sg.slot)] = {c0, c1};
        }
        for (int sidx = 0; sidx < 4; ++sidx) {
            std::vector<uint64_t> ids(size_t(M) * geom::ENGRAM_HEADS);
            for (int m = 0; m < M; ++m) {
                engram_store.hash_token(sidx, tri[size_t(m)][0], tri[size_t(m)][1], tri[size_t(m)][2],
                                        ids.data() + size_t(m) * geom::ENGRAM_HEADS);
            }
            eng_enqueue(sidx, M, ids);
        }
    }

    // Save idx_raw for slot `slot` across all 12 DSA layers before a DFly2 verify step.
    void dfly_save_idx_raw(int slot) {
        if (!has_dfly) return;
        for (int l = 0; l < geom::L_MAIN; ++l) {
            const Layer& y = L[size_t(l)];
            if (!y.is_dsa) continue;
            for (int i = 0; i < o.nloc; ++i) {
                const size_t k = static_cast<size_t>(y.dsa_idx) * o.nloc + i;
                HY4_CK(cudaMemcpyAsync(
                    idx_raw_bak[k] + static_cast<size_t>(slot) * 4 * geom::IDX_DIM,
                    y.rk[size_t(i)].dsa_kv.idx_raw + static_cast<size_t>(slot) * 4 * geom::IDX_DIM,
                    4 * geom::IDX_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                HY4_CK(cudaMemcpyAsync(
                    idx_raw_thw_bak[k] + slot,
                    y.rk[size_t(i)].dsa_kv.idx_raw_thw + slot,
                    sizeof(int3), cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    int snap_off_for_seg_len(int seg_len) const {
        return (seg_len > 0 && seg_len <= dfly_w_max) ? (seg_len - 1) : 0;
    }

    // Rollback idx_raw and eng_ctx after a partial DFly2 verify accept (`n_acc < verify_len`).
    void dfly_rollback_slot(int slot, int pos0, int n_acc, const int* toks_row0,
                            const std::array<int32_t, 2>& saved_eng_ctx) {
        // 1. Restore Engram 2-token history to exact last 2 accepted tokens!
        if (has_engram) {
            int32_t c0 = (pos0 == 0) ? geom::ENGRAM_PAD_ID : saved_eng_ctx[0];
            int32_t c1 = (pos0 == 0) ? geom::ENGRAM_PAD_ID : saved_eng_ctx[1];
            for (int t = 0; t < n_acc; ++t) {
                c0 = c1;
                c1 = toks_row0[t];
            }
            eng_ctx[size_t(slot)] = {c0, c1};
        }
        // 2. Restore exact idx_raw[slot][0..3] across all 12 DSA layers from slot's isolated idx_tk_ver slice!
        if (has_dfly) {
            const size_t tk_base = static_cast<size_t>(slot) * dfly_w_max * geom::IDX_DIM;
            for (int l = 0; l < geom::L_MAIN; ++l) {
                const Layer& y = L[size_t(l)];
                if (!y.is_dsa) continue;
                for (int i = 0; i < o.nloc; ++i) {
                    const size_t k = static_cast<size_t>(y.dsa_idx) * o.nloc + i;
                    k_hy4_idx_raw_replay<<<1, 128, 0, stream>>>(
                        idx_raw_bak[k] + static_cast<size_t>(slot) * 4 * geom::IDX_DIM,
                        idx_tk_ver[k]  + tk_base,
                        y.rk[size_t(i)].dsa_kv.idx_raw + static_cast<size_t>(slot) * 4 * geom::IDX_DIM,
                        pos0, n_acc);
                    // Restore idx_raw_thw[slot]: if pos0+n_acc stays in the pre-verify group, restore backup;
                    // else the new incomplete group starts at ((pos0+n_acc) & ~3) which is a decode 1D text token!
                    if (((pos0 + n_acc) >> 2) == (pos0 >> 2)) {
                        HY4_CK(cudaMemcpyAsync(
                            y.rk[size_t(i)].dsa_kv.idx_raw_thw + slot,
                            idx_raw_thw_bak[k] + slot,
                            sizeof(int3), cudaMemcpyDeviceToDevice, stream));
                    } else if (((pos0 + n_acc) & 3) != 0) {
                        const int gp0_r = (((pos0 + n_acc) >> 2) << 2) + slot_mrope_delta[size_t(slot)];
                        const int3 thw_val = make_int3(gp0_r, gp0_r, gp0_r);
                        HY4_CK(cudaMemcpyAsync(
                            y.rk[size_t(i)].dsa_kv.idx_raw_thw + slot,
                            &thw_val, sizeof(int3), cudaMemcpyHostToDevice, stream));
                    }
                }
            }
        }
    }

    // Emit one layer's GPU ops (for either eager execution or CUDA graph capture!)
    void emit_layer_gpu(int l, const hy4::StepSeg* segs, int nsegs, int M, int max_pos,
                        int& coll, bool capture_mode, int graph_idx_groups,
                        bool snap_on, bool save_tk,
                        const std::function<void(double&)>& prof_mark,
                        const int3* mrope_ptr = nullptr) {
        const Layer& y = L[size_t(l)];

        // 4b. Attention HC Pre + input_layernorm
        hy4::hc_pre_and_norm(blas, y.attn_hc, hc_sc, hyper, y.in_ln, mixed_ln, M, stream);
        if (!capture_mode) prof_mark(prof.hc_ms);

        // 4c. Attention (per local rank)
        const int g_ns = capture_mode
            ? ((max_pos + 1 < 256) ? 1
               : (y.is_dsa && max_pos >= geom::IDX_TOPK) ? std::max(1, 16 / M)
                                                         : std::max(1, 8 / M))
            : 0;
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            const RankLayer& rl = y.rk[size_t(i)];
            __nv_bfloat16* p_dst = parts + static_cast<size_t>(r) * M * geom::HID;
            const bool do_snap = snap_on && has_dfly && (y.dfly_idx >= 0) && (i == 0);
            __nv_bfloat16* tk_dst = (save_tk && has_dfly && y.is_dsa)
                ? idx_tk_ver[static_cast<size_t>(y.dsa_idx) * o.nloc + i]
                : nullptr;
            hy4::attn_forward(blas, rl.attn_w, attn_sc,
                              y.is_dsa ? const_cast<hy4::DsaLayerKv*>(&rl.dsa_kv) : nullptr,
                              y.is_dsa ? nullptr : const_cast<hy4::SwaLayerKv*>(&rl.swa_kv),
                              mixed_ln, p_dst,
                              segs, nsegs,
                              d_slot, d_pos, d_page_table,
                              M, max_pps, max_groups, max_pos, mx8_x8, stream,
                              (do_snap || tk_dst != nullptr) ? d_snap_dst : nullptr,
                              do_snap ? dfly_snap_grp[size_t(y.dfly_idx)] : nullptr,
                              do_snap ? dfly_snap_cnt[size_t(y.dfly_idx)] : nullptr,
                              do_snap ? dfly_snap_pos[size_t(y.dfly_idx)] : nullptr,
                              tk_dst, graph_idx_groups, g_ns,
                              d_rope_pos, mrope_ptr);
        }
        if (!capture_mode) prof_mark(prof.attn_ms);

        // 4d. Attention TP AllReduce + post_proj_norm + HC Inject
        const __nv_bfloat16* attn_post_ln = y.rk[0].attn_w.post_proj_ln;
        if (on_ar_norm_inject) {
            on_ar_norm_inject(coll++, M, attn_post_ln);
        } else {
            hy4::k_hy4_sum_norm_inject<<<M, 256, 0, stream>>>(
                parts, attn_post_ln, hc_sc.post, hyper, o.nr, M, M);
        }
        if (!capture_mode) {
            prof_mark(prof.ar_ms);
            if (!o.dump_dir.empty()) {
                char fn[64];
                std::snprintf(fn, sizeof(fn), "L%03d_post_attn.bf16", l);
                dump_tensor(fn, hyper, size_t(M) * geom::HC_D * 2);
            }
        }

        // 4e. FFN HC Pre + post_attention_layernorm
        hy4::hc_pre_and_norm(blas, y.ffn_hc, hc_sc, hyper, y.post_attn_ln, mixed_ln, M, stream);
        if (!capture_mode) prof_mark(prof.hc_ms);

        // 4f+4g. MLP (L0 dense) or MoE (L1..L41) + fused TP=1 add_norm_inject!
        const bool fuse_tp1_moe = (y.is_moe && o.nr == 1 && !on_ar_norm_inject);
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            const RankLayer& rl = y.rk[size_t(i)];
            __nv_bfloat16* p_dst = parts + static_cast<size_t>(r) * M * geom::HID;
            if (!y.is_moe) {
                hy4::dense_mlp_forward(blas, rl.dense_w, dense_sc, mixed_ln, p_dst, M, mx8_x8, stream);
            } else {
                hy4::moe_forward(blas, rl.moe_w, moe_sc, mixed_ln, p_dst, M, mx8_x8, stream, fuse_tp1_moe);
            }
        }
        if (!capture_mode) prof_mark(prof.moe_ms);

        const __nv_bfloat16* ffn_post_ln = y.is_moe ? y.rk[0].moe_w.combine_ln : y.rk[0].dense_w.post_fc2_ln;
        if (fuse_tp1_moe) {
            hy4::k_hy4_add_norm_inject<<<M, 256, 0, stream>>>(
                moe_sc.routed_up_out, moe_sc.shared_out, ffn_post_ln, hc_sc.post, hyper, M);
        } else if (on_ar_norm_inject) {
            on_ar_norm_inject(coll++, M, ffn_post_ln);
        } else {
            hy4::k_hy4_sum_norm_inject<<<M, 256, 0, stream>>>(
                parts, ffn_post_ln, hc_sc.post, hyper, o.nr, M, M);
        }
        if (!capture_mode) {
            prof_mark(prof.ar_ms);
            if (!o.dump_dir.empty()) {
                char fn[64];
                std::snprintf(fn, sizeof(fn), "L%03d_out.bf16", l);
                dump_tensor(fn, hyper, size_t(M) * geom::HC_D * 2);
            }
        }
    }

    static int idx_group_bucket(int max_pos, int max_groups) {
        if (max_pos < geom::IDX_TOPK) return 0;
        const int ng = std::min(max_groups, (max_pos >> 2));
        int b = 1024;
        while (b < ng && b < max_groups) b <<= 1;
        return std::min(b, max_groups);
    }

    // Forward pass across all 42 layers + optional lm_head sampling/logits dump.
    void forward(const std::vector<hy4::StepSeg>& segs, const int* toks, const RowSample* sm,
                 std::vector<RowOut>& out, bool skip_head = false,
                 std::vector<float>* out_full_logits = nullptr,
                 bool save_verify_idx_tk = false,
                 int vis_n = 0, const int3* h_mrope = nullptr) {
        int M = 0, max_pos = 0;
        for (const auto& sg : segs) {
            M = std::max(M, sg.row0 + sg.len);
            if (sg.len > 0) max_pos = std::max(max_pos, sg.pos0 + sg.len - 1);
        }
        if (M <= 0 || M > o.max_rows) die("forward: bad M=" + std::to_string(M));
        if (max_pos >= o.cap) die("forward: pos " + std::to_string(max_pos) + " >= cap " + std::to_string(o.cap));

        // 1. Kick off async 4-layer Engram NVMe prefetch IMMEDIATELY
        prepare_engram_step(segs, toks, M);

        // 2. Upload tok/slot/pos/rope_pos/snap_dst metadata + optional M-RoPE & Engram mask
        int* h_tok      = h_meta_pinned;
        int* h_slot     = h_meta_pinned + M;
        int* h_pos      = h_meta_pinned + 2 * M;
        int* h_rope_pos = h_meta_pinned + 3 * M;
        int* h_snap_dst = h_meta_pinned + 4 * M;
        bool any_eng_masked = false;
        for (int m = 0; m < M; ++m) { h_snap_dst[m] = -1; h_eng_mask_pinned[m] = 1; }
        for (const auto& sg : segs) {
            const bool fit_w = has_dfly && sg.slot >= 0 && sg.slot < o.slots && sg.len > 0 && sg.len <= dfly_w_max;
            const int rdelta = (sg.slot >= 0 && size_t(sg.slot) < slot_mrope_delta.size())
                ? slot_mrope_delta[size_t(sg.slot)] : 0;
            for (int t = 0; t < sg.len; ++t) {
                const int m = sg.row0 + t;
                const int tok = toks[m];
                h_tok[m]      = tok;
                h_slot[m]     = sg.slot;
                h_pos[m]      = sg.pos0 + t;
                h_rope_pos[m] = sg.pos0 + t + rdelta;
                if (tok == hy4v::TOK_IMAGE || tok == hy4v::TOK_VIDEO || tok == geom::ENGRAM_PAD_ID) {
                    h_eng_mask_pinned[m] = 0;
                    any_eng_masked = true;
                }
                if (fit_w) h_snap_dst[m] = sg.slot * dfly_w_max + t;
            }
            if (has_dfly && !fit_w && sg.slot >= 0 && sg.slot < o.slots && sg.len > 0) {
                h_snap_dst[sg.row0 + sg.len - 1] = sg.slot * dfly_w_max;
            }
        }
        HY4_CK(cudaMemcpyAsync(d_tok,      h_tok,      size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        HY4_CK(cudaMemcpyAsync(d_slot,     h_slot,     size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        HY4_CK(cudaMemcpyAsync(d_pos,      h_pos,      size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        HY4_CK(cudaMemcpyAsync(d_rope_pos, h_rope_pos, size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        HY4_CK(cudaMemcpyAsync(d_snap_dst, h_snap_dst, size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        const int3* mrope_ptr = nullptr;
        if (h_mrope != nullptr) {
            HY4_CK(cudaMemcpyAsync(d_mrope, h_mrope, size_t(M) * sizeof(int3), cudaMemcpyHostToDevice, stream));
            mrope_ptr = d_mrope;
        }
        const uint8_t* eng_mask_ptr = nullptr;
        if (any_eng_masked) {
            HY4_CK(cudaMemcpyAsync(d_eng_mask, h_eng_mask_pinned, size_t(M), cudaMemcpyHostToDevice, stream));
            eng_mask_ptr = d_eng_mask;
        }

        if (on_step_begin) on_step_begin(M, skip_head);

        const size_t rb = head_row_bytes(o.nc);
        const bool snap_on = has_dfly && !skip_head;
        const bool use_graph_now = o.use_graphs && o.nr <= 2 && o.nloc == 1 && !o.profile &&
                                   o.dump_dir.empty() && !skip_head && !out_full_logits &&
                                   vis_n == 0 && !mrope_ptr && !eng_mask_ptr && M <= 12;

        if (use_graph_now) {
            const bool is_big_kv = (max_pos + 1 >= 256);
            const bool is_sparse = (max_pos >= geom::IDX_TOPK);
            const int ib = idx_group_bucket(max_pos, max_groups);
            const GraphKey gk{M, is_big_kv, is_sparse, ib, snap_on, save_verify_idx_tk};
            auto& sg_entry = graphs[gk];
            static constexpr int SEG_LO[5] = {0, 1, 11, 21, 31};
            static constexpr int SEG_HI[5] = {1, 11, 21, 31, 42};
            std::function<void(double&)> dummy_prof = [](double&) {};

            if (!sg_entry.ready) {
                for (int seg = 0; seg < 5; ++seg) {
                    if (seg > 0 && has_engram) {
                        eng_wait(seg - 1);
                        const size_t rbytes = size_t(M) * geom::ENGRAM_HEADS * geom::ENGRAM_ROW_BYTES;
                        HY4_CK(cudaMemcpyAsync(eng_sc.raw_rows, eng_pinned[seg - 1], rbytes, cudaMemcpyHostToDevice, stream));
                    }
                    cudaGraph_t gr = nullptr;
                    HY4_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    int coll = 2 * SEG_LO[seg];
                    if (seg == 0) {
                        hy4::k_hy4_embed_expand<<<M * 12, 256, 0, stream>>>(d_tok, embed, hyper, M);
                    } else if (has_engram) {
                        hy4::engram_forward(blas, eng_w[size_t(seg - 1)], eng_sc, hyper, M, mx8_x8, stream);
                    }
                    for (int l = SEG_LO[seg]; l < SEG_HI[seg]; ++l) {
                        emit_layer_gpu(l, segs.data(), static_cast<int>(segs.size()), M, max_pos,
                                       coll, true, ib, snap_on, save_verify_idx_tk, dummy_prof);
                    }
                    if (seg == 4) {
                        hy4::k_hy4_hc_head_and_norm<<<M, 256, 0, stream>>>(
                            hyper, hc_head.fn, hc_head.base, hc_head.scale, final_norm, mixed_ln, M);
                        if (snap_on) {
                            k_hy4_scatter_hidden<<<M, 256, 0, stream>>>(mixed_ln, d_snap_dst, dfly_target_hidden, M);
                        }
                        hy4::linear_bf16_f32(blas, mixed_ln, lm_head[0], head_logits, M, v_rows[0], geom::HID, stream);
                        ink2::k_ink2_head_pack<<<M, 256, 0, stream>>>(head_logits, v_rows[0], v_lo[0], o.nc, head_dev_pkt, rb);
                    }
                    HY4_CK(cudaStreamEndCapture(stream, &gr));
                    HY4_CK(cudaGraphInstantiate(&sg_entry.exec[size_t(seg)], gr, nullptr, nullptr, 0));
                    HY4_CK(cudaGraphDestroy(gr));
                    HY4_CK(cudaGraphLaunch(sg_entry.exec[size_t(seg)], stream));
                }
                sg_entry.ready = true;
            } else {
                for (int seg = 0; seg < 5; ++seg) {
                    if (seg > 0 && has_engram) {
                        eng_wait(seg - 1);
                        const size_t rbytes = size_t(M) * geom::ENGRAM_HEADS * geom::ENGRAM_ROW_BYTES;
                        HY4_CK(cudaMemcpyAsync(eng_sc.raw_rows, eng_pinned[seg - 1], rbytes, cudaMemcpyHostToDevice, stream));
                    }
                    HY4_CK(cudaGraphLaunch(sg_entry.exec[size_t(seg)], stream));
                }
            }
            HY4_CK(cudaMemcpyAsync(head_host_parts + static_cast<size_t>(o.rank0) * M * rb,
                                   head_dev_pkt, size_t(M) * rb, cudaMemcpyDeviceToHost, stream));
            HY4_CK(cudaStreamSynchronize(stream));
            if (on_head_gather) on_head_gather(2 * geom::L_MAIN, M, head_host_parts, rb);
            if (on_step_end) on_step_end();
            out.resize(size_t(M));
            if (o.nr == 1) {
                for (int m = 0; m < M; ++m) {
                    out[size_t(m)] = head_merge(head_host_parts + size_t(m) * rb, rb, 1, o.nc);
                    out[size_t(m)].tok = sm ? sample_row(out[size_t(m)].top, sm[m])
                                            : (out[size_t(m)].top.empty() ? 0 : out[size_t(m)].top[0].id);
                }
            } else {
                std::vector<uint8_t> row_buf(size_t(o.nr) * rb);
                for (int m = 0; m < M; ++m) {
                    for (int r = 0; r < o.nr; ++r) {
                        std::memcpy(row_buf.data() + size_t(r) * rb,
                                    head_host_parts + (static_cast<size_t>(r) * M + m) * rb, rb);
                    }
                    out[size_t(m)] = head_merge(row_buf.data(), rb, o.nr, o.nc);
                    out[size_t(m)].tok = sm ? sample_row(out[size_t(m)].top, sm[m])
                                            : (out[size_t(m)].top.empty() ? 0 : out[size_t(m)].top[0].id);
                }
            }
            return;
        }

        // Eager path (prefill, TP>1, profile, dump, or GR8_HY4_GRAPHS=0)
        auto prof_mark = [&](double& acc) {
            if (!o.profile) return;
            HY4_CK(cudaEventRecord(ev_b, stream));
            HY4_CK(cudaEventSynchronize(ev_b));
            float ms = 0.f;
            HY4_CK(cudaEventElapsedTime(&ms, ev_a, ev_b));
            acc += ms;
            HY4_CK(cudaEventRecord(ev_a, stream));
        };
        if (o.profile) HY4_CK(cudaEventRecord(ev_a, stream));

        if (vis_n > 0 && vision.ready()) {
            hy4::k_hy4_embed_expand_vis<<<M * 12, 256, 0, stream>>>(
                d_tok, embed, vision.pool(), vision.row_map(), hyper, M);
        } else {
            hy4::k_hy4_embed_expand<<<M * 12, 256, 0, stream>>>(d_tok, embed, hyper, M);
        }
        if (!o.dump_dir.empty()) dump_tensor("embed.bf16", hyper, size_t(M) * geom::HC_D * 2);

        int coll = 0;
        for (int l = 0; l < geom::L_MAIN; ++l) {
            const Layer& y = L[size_t(l)];
            if (y.eng_slot >= 0 && has_engram) {
                const auto te0 = std::chrono::steady_clock::now();
                eng_wait(y.eng_slot);
                if (o.profile) {
                    prof.eng_io_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te0).count();
                    HY4_CK(cudaEventRecord(ev_a, stream));
                }
                const size_t rbytes = size_t(M) * geom::ENGRAM_HEADS * geom::ENGRAM_ROW_BYTES;
                HY4_CK(cudaMemcpyAsync(eng_sc.raw_rows, eng_pinned[y.eng_slot], rbytes, cudaMemcpyHostToDevice, stream));
                hy4::engram_forward(blas, eng_w[size_t(y.eng_slot)], eng_sc, hyper, M, mx8_x8, stream, eng_mask_ptr);
                prof_mark(prof.eng_gpu_ms);
                if (!o.dump_dir.empty()) {
                    char fn[64];
                    std::snprintf(fn, sizeof(fn), "L%03d_post_engram.bf16", l);
                    dump_tensor(fn, hyper, size_t(M) * geom::HC_D * 2);
                }
            }
            emit_layer_gpu(l, segs.data(), static_cast<int>(segs.size()), M, max_pos,
                           coll, false, 0, snap_on, save_verify_idx_tk, prof_mark, mrope_ptr);
        }

        if (skip_head) {
            if (on_step_end) on_step_end();
            HY4_CK(cudaStreamSynchronize(stream));
            if (o.profile) { prof.steps += 1; prof.tokens += M; }
            return;
        }

        hy4::k_hy4_hc_head_and_norm<<<M, 256, 0, stream>>>(
            hyper, hc_head.fn, hc_head.base, hc_head.scale, final_norm, mixed_ln, M);
        if (!o.dump_dir.empty()) dump_tensor("final_norm.bf16", mixed_ln, size_t(M) * geom::HID * 2);

        if (snap_on) {
            k_hy4_scatter_hidden<<<M, 256, 0, stream>>>(mixed_ln, d_snap_dst, dfly_target_hidden, M);
        }

        if (out_full_logits) out_full_logits->assign(size_t(M) * geom::VOCAB, 0.0f);

        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            const int vr = v_rows[size_t(i)];
            const int v0 = v_lo[size_t(i)];
            hy4::linear_bf16_f32(blas, mixed_ln, lm_head[size_t(i)], head_logits, M, vr, geom::HID, stream);
            if (out_full_logits) {
                HY4_CK(cudaStreamSynchronize(stream));
                std::vector<float> tmp(size_t(M) * vr);
                HY4_CK(cudaMemcpy(tmp.data(), head_logits, tmp.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int m = 0; m < M; ++m) {
                    std::memcpy(out_full_logits->data() + size_t(m) * geom::VOCAB + v0,
                                tmp.data() + size_t(m) * vr, size_t(vr) * sizeof(float));
                }
            }
            ink2::k_ink2_head_pack<<<M, 256, 0, stream>>>(head_logits, vr, v0, o.nc, head_dev_pkt, rb);
            HY4_CK(cudaMemcpyAsync(head_host_parts + static_cast<size_t>(r) * M * rb,
                                   head_dev_pkt, size_t(M) * rb, cudaMemcpyDeviceToHost, stream));
        }
        HY4_CK(cudaStreamSynchronize(stream));
        if (on_head_gather) on_head_gather(coll++, M, head_host_parts, rb);
        if (on_step_end) on_step_end();
        prof_mark(prof.head_ms);

        if (!o.dump_dir.empty() && out_full_logits) {
            std::ofstream f(o.dump_dir + "/logits.f32", std::ios::binary);
            f.write(reinterpret_cast<const char*>(out_full_logits->data()),
                    static_cast<std::streamsize>(out_full_logits->size() * sizeof(float)));
        }

        out.resize(size_t(M));
        std::vector<uint8_t> row_buf(size_t(o.nr) * rb);
        for (int m = 0; m < M; ++m) {
            for (int r = 0; r < o.nr; ++r) {
                std::memcpy(row_buf.data() + size_t(r) * rb,
                            head_host_parts + (static_cast<size_t>(r) * M + m) * rb, rb);
            }
            out[size_t(m)] = head_merge(row_buf.data(), rb, o.nr, o.nc);
            out[size_t(m)].tok = sm ? sample_row(out[size_t(m)].top, sm[m])
                                    : (out[size_t(m)].top.empty() ? 0 : out[size_t(m)].top[0].id);
        }
        if (o.profile) { prof.steps += 1; prof.tokens += M; }
    }

    // Propose K speculative draft tokens using built-in 0.85B DFly2 drafter!
    // `snap_off`: offset within slot's isolated snapshot slice [0..dfly_w_max-1] (0 after long prefill, kept-1 after verify).
    std::vector<int> dfly_propose(int slot, int bonus_tok, int anchor, int snap_off, int K,
                                  const RowSample* sm_base = nullptr,
                                  std::vector<RowOut>* out_draft_rows = nullptr) {
        std::vector<int> drafts;
        if (!has_dfly || K <= 0) return drafts;
        const int W = K + 1;
        if (W > dfly_sc.max_W) die("dfly_propose: W exceeds dfly_sc.max_W");
        if (slot < 0 || slot >= o.slots || snap_off < 0 || snap_off >= dfly_w_max)
            die("dfly_propose: slot/snap_off out of range");
        const int s_idx = slot * dfly_w_max + snap_off;

        const auto td0 = std::chrono::steady_clock::now();
        const int rdelta = (slot >= 0 && size_t(slot) < slot_mrope_delta.size())
            ? slot_mrope_delta[size_t(slot)] : 0;
        const bool fast_gpu_greedy = ((o.nr == 1 || (o.nr == 2 && static_cast<bool>(on_dfly_argmax_tp2))) &&
                                      o.nloc == 1 &&
                                      (!sm_base || !(sm_base->inv_temp > 0.f)) &&
                                      out_draft_rows == nullptr);
        const bool use_dfly_graph = fast_gpu_greedy && o.use_graphs && !o.profile;

        if (on_dfly_begin) on_dfly_begin(W, fast_gpu_greedy);

        // Gather target DSA KV pointers for layers [25, 29, 33, 37, 41]
        std::vector<const hy4::DsaLayerKv*> tkv(size_t(hy4::dfly::NUM_LAYERS * o.nloc));
        for (int d = 0; d < hy4::dfly::NUM_LAYERS; ++d) {
            const int tl = hy4::dfly::TARGET_DSA_LAYERS[d];
            for (int i = 0; i < o.nloc; ++i) tkv[size_t(d * o.nloc + i)] = &L[size_t(tl)].rk[size_t(i)].dsa_kv;
        }

        drafts.resize(size_t(K));

        if (fast_gpu_greedy) {
            // Zero-sync TP=1 & TP=2 greedy DFly2: entire 5-layer body + K hidden_correction + MXFP8 lm_head + argmax
            // (with 16B GPU-side RDMA argmax merge on TP=2) runs in ONE unbroken stream or ONE CUDA Graph!
            *h_dfly_meta_pinned = hy4::dfly::DflyGraphMeta{slot, bonus_tok, anchor, rdelta, s_idx, {0, 0, 0}};
            HY4_CK(cudaMemcpyAsync(dfly_sc.d_meta, h_dfly_meta_pinned, sizeof(hy4::dfly::DflyGraphMeta),
                                   cudaMemcpyHostToDevice, stream));

            std::array<const int*, hy4::dfly::NUM_LAYERS> sg_base{}, sc_base{}, sp_base{};
            for (int d = 0; d < hy4::dfly::NUM_LAYERS; ++d) {
                sg_base[size_t(d)] = dfly_snap_grp[size_t(d)];
                sc_base[size_t(d)] = dfly_snap_cnt[size_t(d)];
                sp_base[size_t(d)] = dfly_snap_pos[size_t(d)];
            }

            auto emit_dfly_greedy = [&]() {
                hy4::dfly::forward_body(
                    blas, dfly_w, dfly_sc, attn_sc, tkv.data(),
                    sg_base.data(), sc_base.data(), sp_base.data(),
                    embed, dfly_target_hidden, parts, d_page_table, mx8_x8,
                    o.nr, o.rank0, o.nloc, 0, 0, 0, W, max_pps, stream, on_dfly_ar_add, 0, dfly_sc.d_meta);
                int coll = 2 * hy4::dfly::NUM_LAYERS;
                for (int step = 0; step < K; ++step) {
                    const __nv_bfloat16* h_step = dfly_sc.h + size_t(step + 1) * geom::HID;
                    hy4::dfly::hidden_correction_step(
                        blas, dfly_w[0], dfly_sc, h_step, embed, 0, dfly_sc.d_draft_id + step,
                        mixed_ln, mx8_x8, stream);
                    hy4::mx8::linear_w_f32(blas, mixed_ln, dfly_lm_head[0], head_logits, 1, v_rows[0], geom::HID, mx8_x8, nullptr, stream);
                    hy4::dfly::k_dfly_argmax_p1<<<256, 256, 0, stream>>>(head_logits, dfly_sc.d_max_v, dfly_sc.d_max_i, v_rows[0]);
                    if (o.nr == 1) {
                        hy4::dfly::k_dfly_argmax_p2<<<1, 256, 0, stream>>>(dfly_sc.d_max_v, dfly_sc.d_max_i, dfly_sc.d_draft_id + step + 1, 256);
                    } else {
                        hy4::dfly::k_dfly_argmax_p2_pair<<<1, 256, 0, stream>>>(
                            dfly_sc.d_max_v, dfly_sc.d_max_i, v_lo[0], dfly_sc.d_arg_pair, 256);
                        on_dfly_argmax_tp2(coll++, dfly_sc.d_arg_pair, dfly_sc.d_draft_id + step + 1);
                    }
                }
            };

            if (use_dfly_graph) {
                auto it = dfly_graphs.find(W);
                if (it == dfly_graphs.end()) {
                    cudaGraph_t gr = nullptr;
                    cudaGraphExec_t ex = nullptr;
                    HY4_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    emit_dfly_greedy();
                    HY4_CK(cudaStreamEndCapture(stream, &gr));
                    HY4_CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
                    HY4_CK(cudaGraphDestroy(gr));
                    it = dfly_graphs.emplace(W, ex).first;
                }
                HY4_CK(cudaGraphLaunch(it->second, stream));
            } else {
                emit_dfly_greedy();
            }

            HY4_CK(cudaMemcpyAsync(drafts.data(), dfly_sc.d_draft_id + 1, size_t(K) * sizeof(int), cudaMemcpyDeviceToHost, stream));
            HY4_CK(cudaStreamSynchronize(stream));
            if (on_dfly_end) on_dfly_end();
        } else {
            std::array<const int*, hy4::dfly::NUM_LAYERS> sg_p{}, sc_p{}, sp_p{};
            constexpr int KSEL = geom::IDX_GROUP_TOPK - 1;
            for (int d = 0; d < hy4::dfly::NUM_LAYERS; ++d) {
                sg_p[size_t(d)] = dfly_snap_grp[size_t(d)] + size_t(s_idx) * KSEL;
                sc_p[size_t(d)] = dfly_snap_cnt[size_t(d)] + s_idx;
                sp_p[size_t(d)] = dfly_snap_pos[size_t(d)] + s_idx;
            }
            const __nv_bfloat16* tgt_h = dfly_target_hidden + size_t(s_idx) * geom::HID;
            hy4::dfly::forward_body(
                blas, dfly_w, dfly_sc, attn_sc, tkv.data(),
                sg_p.data(), sc_p.data(), sp_p.data(),
                embed, tgt_h, parts, d_page_table, mx8_x8,
                o.nr, o.rank0, o.nloc, slot, bonus_tok, anchor, W, max_pps, stream, on_dfly_ar_add, rdelta);
            if (out_draft_rows) out_draft_rows->resize(size_t(K));
            const size_t rb = head_row_bytes(o.nc);
            std::vector<uint8_t> row_buf(size_t(o.nr) * rb);
            int prev_tok = bonus_tok;
            int coll = 2 * hy4::dfly::NUM_LAYERS;
            for (int step = 0; step < K; ++step) {
                const __nv_bfloat16* h_step = dfly_sc.h + size_t(step + 1) * geom::HID;
                hy4::dfly::hidden_correction_step(
                    blas, dfly_w[0], dfly_sc, h_step, embed, prev_tok, nullptr,
                    mixed_ln, mx8_x8, stream);
                for (int i = 0; i < o.nloc; ++i) {
                    const int r = o.rank0 + i;
                    const int vr = v_rows[size_t(i)];
                    const int v0 = v_lo[size_t(i)];
                    hy4::mx8::linear_w_f32(blas, mixed_ln, dfly_lm_head[size_t(i)], head_logits, 1, vr, geom::HID, mx8_x8, nullptr, stream);
                    ink2::k_ink2_head_pack<<<1, 256, 0, stream>>>(head_logits, vr, v0, o.nc, head_dev_pkt, rb);
                    HY4_CK(cudaMemcpyAsync(head_host_parts + static_cast<size_t>(r) * rb, head_dev_pkt, rb,
                                           cudaMemcpyDeviceToHost, stream));
                }
                HY4_CK(cudaStreamSynchronize(stream));
                if (on_dfly_head_gather) on_dfly_head_gather(coll++, 1, head_host_parts, rb);
                for (int r = 0; r < o.nr; ++r) {
                    std::memcpy(row_buf.data() + size_t(r) * rb, head_host_parts + size_t(r) * rb, rb);
                }
                RowOut ro = head_merge(row_buf.data(), rb, o.nr, o.nc);
                RowSample s_step{};
                if (sm_base) {
                    s_step = *sm_base;
                    s_step.spos = static_cast<uint32_t>(anchor + step + 1);
                }
                ro.tok = sm_base ? sample_row(ro.top, s_step) : (ro.top.empty() ? 0 : ro.top[0].id);
                drafts[size_t(step)] = ro.tok;
                prev_tok = ro.tok;
                if (out_draft_rows) (*out_draft_rows)[size_t(step)] = std::move(ro);
            }
            if (on_dfly_end) on_dfly_end();
        }
        if (o.profile) {
            prof.dfly_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            prof.dfly_steps += 1;
        }
        return drafts;
    }
};

struct SpecStats {
    int rounds = 0;
    int total_drafted = 0;
    int total_accepted = 0;
    int total_emitted = 0;
};

// Single-session lossless DFly2 speculative generation loop (works on TP=1 and TP=2!).
inline void run_dfly_gen_s1(Engine& e, int slot, int start_pos, int first_tok, int first_snap_off,
                            int target_gen, float inv_temp, float top_p, uint32_t seed,
                            std::vector<int>& said, SpecStats& stats) {
    said.push_back(first_tok);
    int cur_pos = start_pos;
    int bonus_tok = first_tok;
    int snap_off = first_snap_off;
    const int K_cfg = e.o.dfly_tokens;
    std::vector<RowOut> out, draft_rows;

    while (static_cast<int>(said.size()) < target_gen) {
        const int rem = target_gen - static_cast<int>(said.size());
        const int K = std::min({K_cfg, rem, e.o.cap - cur_pos - 1});
        if (K <= 0) {
            std::vector<hy4::StepSeg> segs = {{slot, 0, cur_pos, 1}};
            RowSample sm{inv_temp, top_p, seed, uint32_t(cur_pos + 1)};
            e.forward(segs, &bonus_tok, &sm, out, false);
            bonus_tok = out[0].tok;
            said.push_back(bonus_tok);
            cur_pos += 1;
            snap_off = 0;
            break;
        }

        RowSample sm_base{inv_temp, top_p, seed, uint32_t(cur_pos + 1)};
        const bool need_draft_dist = (inv_temp > 0.f);
        std::vector<int> drafts = e.dfly_propose(
            slot, bonus_tok, cur_pos, snap_off, K,
            need_draft_dist ? &sm_base : nullptr,
            need_draft_dist ? &draft_rows : nullptr);

        const int W = K + 1;
        std::vector<int> vtoks(static_cast<size_t>(W), 0);
        vtoks[0] = bonus_tok;
        for (int i = 0; i < K; ++i) vtoks[size_t(i + 1)] = drafts[size_t(i)];
        std::vector<RowSample> sm(static_cast<size_t>(W));
        for (int i = 0; i < W; ++i) sm[size_t(i)] = RowSample{inv_temp, top_p, seed, uint32_t(cur_pos + i + 1)};

        const auto saved_eng = e.eng_ctx[size_t(slot)];
        e.dfly_save_idx_raw(slot);
        std::vector<hy4::StepSeg> segs = {{slot, 0, cur_pos, W}};
        e.forward(segs, vtoks.data(), sm.data(), out, false, nullptr, true);

        int n_acc = 0;
        int next_bonus = 0;
        bool all_ok = true;
        for (int i = 0; i < K; ++i) {
            int repl = 0;
            const bool ok = spec_verify_token(
                drafts[size_t(i)], out[size_t(i)],
                need_draft_dist ? &draft_rows[size_t(i)] : nullptr,
                sm[size_t(i)], repl);
            if (ok) {
                ++n_acc;
                said.push_back(drafts[size_t(i)]);
                if (static_cast<int>(said.size()) >= target_gen) {
                    next_bonus = drafts[size_t(i)];
                    all_ok = false;
                    break;
                }
            } else {
                next_bonus = repl;
                all_ok = false;
                break;
            }
        }
        const int kept_rows = n_acc + 1;
        if (all_ok) next_bonus = out[size_t(K)].tok;
        if (static_cast<int>(said.size()) < target_gen) said.push_back(next_bonus);
        if (kept_rows < W) {
            e.dfly_rollback_slot(slot, cur_pos, kept_rows, vtoks.data(), saved_eng);
        }
        stats.rounds += 1;
        stats.total_drafted += K;
        stats.total_accepted += n_acc;
        stats.total_emitted += (n_acc + 1);

        cur_pos += kept_rows;
        bonus_tok = next_bonus;
        snap_off = kept_rows - 1;
    }
}

} // namespace hy4rt
