// ERNIE-5.5-Flash-Preview (`Ernie5_5_MoeForCausalLM`) unified runtime engine (`ernie55rt::Engine`).
//
// Targets BOTH platforms from one codebase (controlled by `GR8_ERNIE55_HYBRID`):
//   1. 1x DGX Spark (TP=1, 100% GPU-resident, `GR8_ERNIE55_HYBRID=0`) + TP=2/4 RDMA fabric
//      - 2-stage Blackwell MXFP8 dense quantization (`GR8_ERNIE55_MX8_DENSE=1`, saves ~5.4 GiB VRAM + 2x dense BW)
//      - Full-step CUDA Graph capture & replay (`GR8_ERNIE55_GRAPHS=1`) for small-batch decode (M <= 12)
//   2. 1x RTX 5090 (32 GB VRAM) + 128 GB System RAM (`GR8_ERNIE55_HYBRID=1`, TP=1)
//      - Loads part=3 VRAM core (~3.11 GiB) + part=2 MXFP8 dense (~5.94 GiB) + part=4 pageable host backing
//      - Auto-carves remaining VRAM into STRP static core + adaptive dynamic hot-expert cache (`d_ptab[512][16][2]`)
//      - Double-buffered bounded pinned H2D staging for cold experts (bit-exact vs resident mode!)
//
// Architecture:
//   - 43 layers: 37x HCA C128 (YaRN RoPE theta=160k, C128 compressor + ape, SWA-128, F32 sink, VHA, inv-YaRN, grouped wo_a)
//                + 6x sparse MLA-DSA (L7,14,21,28,35,42; plain RoPE theta=8e6, Lightning Indexer top-2048 + SWA-128,
//                  BF16/FP8/INT4 paged KV, absorbed MLA, BF16->F32 sink, VHA, gate(q_c))
//   - 4-stream Sinkhorn-20 mHC (eps outside sqrt, boot-transposed F32 mapping weights) + mhc_contract head
//   - L0 dense SiTU-GLU MLP (I=14336); L1..L42 Latent MoE (FP32 split-feature top-10 router, 512 mul1 routed + 1 mul1 shared)
//   - Vocab 201216 + BF16 multimax LM-head adjustment
#pragma once
#include "../model_geom_ernie55.h"
#include "../knobs-rs/gr8_knobs.h"
#include "../kernels/ernie55_mhc.cuh"
#include "../kernels/ernie55_mxfp8.cuh"
#include "../kernels/ernie55_moe.cuh"
#include "../attn/ernie55_hca.cuh"
#include "../attn/ernie55_dsa.cuh"
#include "../kernels/ink2_head.cuh"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"
#include "ernie55_hybrid.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <tuple>
#include <unistd.h>
#include <vector>

namespace ernie55rt {

__global__ void k_ernie55_bench_seed_bf16(__nv_bfloat16* dst, size_t n, uint32_t seed) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
        const float v = static_cast<float>(int32_t(h & 0xffffu) - 32768) * (1.0f / 65536.0f);
        dst[i] = __float2bfloat16(v);
    }
}

__global__ void k_ernie55_bench_seed_u8(uint8_t* dst, size_t n, uint32_t seed) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352du;
        dst[i] = static_cast<uint8_t>(((h & 1u) << 7) | ((3u + ((h >> 1) & 3u)) << 3) | ((h >> 3) & 7u));
    }
}

__global__ void k_ernie55_bench_seed_half(__half* dst, size_t n, float val) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const __half hv = __float2half_rn(val);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = hv;
    }
}

[[noreturn]] inline void die(const std::string& m) {
    std::fprintf(stderr, "ernie55: %s\n", m.c_str());
    std::exit(2);
}

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}
#define ERNIE55_CK(x) ::ernie55rt::ck((x), #x)

template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    ERNIE55_CK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    ERNIE55_CK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
    return static_cast<T*>(p);
}

inline ernie55::KvFmt parse_kv_fmt(const std::string& s) {
    if (s == "bf16" || s == "BF16") return ernie55::KvFmt::BF16;
    if (s == "fp8"  || s == "FP8")  return ernie55::KvFmt::FP8;
    if (s == "int4" || s == "INT4" || s == "kv4") return ernie55::KvFmt::INT4;
    die("bad --kv-format / GR8_ERNIE55_KV_FORMAT '" + s + "' (want bf16|fp8|int4)");
}

inline const char* kv_fmt_name(ernie55::KvFmt f) {
    if (f == ernie55::KvFmt::FP8)  return "fp8";
    if (f == ernie55::KvFmt::INT4) return "int4";
    return "bf16";
}

// ------------------------------------------------------------------------------------------------ Head & Sampling
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
        float mx; int count; double sum;
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

// ------------------------------------------------------------------------------------------------ GPU Rank Image
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
        std::snprintf(mf, sizeof(mf), "/tmp/ernie55-mf-p%d-r%d-pt%u.json", static_cast<int>(getpid()), r, part);
        char err[512] = {};
        plan = g6l_ernie55_plan_part(release_dir.c_str(), widths_path.c_str(), static_cast<uint32_t>(nr),
                                     static_cast<uint32_t>(r), part, mf, err, sizeof(err));
        if (!plan) die("g6l_ernie55_plan_part(r" + std::to_string(r) + ", part=" + std::to_string(part) + "): " + err);

        std::string perr;
        const bool pok = pack.open_manifest(mf, perr);
        ::unlink(mf);
        if (!pok) die("open_manifest(r" + std::to_string(r) + "): " + perr);
        ERNIE55_CK(cudaMalloc(&dev, pack.bytes()));

        Sink sink;
        sink.dev = dev;
        const g6l_sink csink{&sink, Sink::alloc, Sink::release, Sink::copy};
        g6l_stats st{};
        const int rc = g6l_plan_load(plan, &csink, &st, err, sizeof(err));
        for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
        if (rc != 0) die("g6l_plan_load(r" + std::to_string(r) + "): " + err);
        g6l_plan_free(plan);
        plan = nullptr;
        static const char* pnames[5] = {"full", "perm", "dense-staging", "hybrid-vram-core", "hybrid-host"};
        std::fprintf(stderr, "# rank %d: ernie55[%s] loaded %.2f GiB in %.2f s (%u threads)\n",
                     r, pnames[part < 5 ? part : 0], st.image_bytes / double(1ull << 30), st.seconds, st.threads);
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
        ERNIE55_CK(cudaMemcpy(out.data(), dev + e.offset, e.nbytes, cudaMemcpyDeviceToHost));
    }
};

// ------------------------------------------------------------------------------------------------ Options & Profile
struct Options {
    int nr = 1;
    int rank0 = 0;
    int nloc = 1;                  // nr for m4sim, 1 for g6m4
    int slots = 1;
    int max_rows = 512;
    int cap = 16384;               // max sequence length per slot
    int pool_rows = 0;             // global dynamic DSA KV pool rows (0 -> auto = slots * cap)
    int nc = 256;
    std::string widths_path;
    bool profile = false;
    std::string dump_dir;

    ernie55::KvFmt kv_fmt = ernie55::KvFmt::FP8;
    bool kv_fmt_set = false;
    bool mx8_dense = true;
    bool mx8_set = false;
    bool mx8_head = false;
    bool mx8_head_set = false;
    bool use_graphs = true;
    bool graphs_set = false;
    bool use_hybrid = false;
    bool hybrid_set = false;
};

struct ProfileAccum {
    double mhc_ms = 0, hca_ms = 0, dsa_ms = 0, moe_ms = 0, ar_ms = 0, head_ms = 0;
    double head_gemv_ms = 0, head_pack_ms = 0;
    ernie55::HcaProfDetail hca_d{};
    ernie55::DsaProfDetail dsa_d{};
    ernie55::MoeProfDetail moe_d{};
    int steps = 0, tokens = 0;
    void reset() {
        mhc_ms = hca_ms = dsa_ms = moe_ms = ar_ms = head_ms = 0;
        head_gemv_ms = head_pack_ms = 0;
        hca_d.qkv_comp_ms = hca_d.prep_ms = hca_d.qup_gate_ms = hca_d.q_norm_rope_ms = 0;
        hca_d.attn_tc_ms = hca_d.vha_invrope_ms = hca_d.woa_gate_ms = hca_d.o_proj_ms = 0;
        dsa_d.qkv_idx_ms = dsa_d.prep_ms = dsa_d.qbg_ms = dsa_d.q_abs_ms = 0;
        dsa_d.idx_score_ms = dsa_d.mla_tc_ms = dsa_d.v_exp_ms = dsa_d.vha_gate_ms = dsa_d.o_proj_ms = 0;
        moe_d.router_ms = moe_d.fc1_lat_ms = moe_d.prerot_r_ms = moe_d.routed_fc1_ms = 0;
        moe_d.routed_fc2_ms = moe_d.shared_ms = moe_d.finish_ms = 0;
        steps = tokens = 0;
    }
    void print(int rank) const {
        if (!steps) return;
        const double tot = mhc_ms + hca_ms + dsa_ms + moe_ms + ar_ms + head_ms;
        std::fprintf(stderr,
                     "# ernie55 profile r%d (%d steps, %d tok, %.2f ms total, %.2f ms/step): "
                     "mhc=%.2f hca=%.2f dsa=%.2f moe=%.2f ar=%.2f head=%.2f (gemv=%.2f pack=%.2f)\n",
                     rank, steps, tokens, tot, tot / steps,
                     mhc_ms / steps, hca_ms / steps, dsa_ms / steps,
                     moe_ms / steps, ar_ms / steps, head_ms / steps,
                     head_gemv_ms / steps, head_pack_ms / steps);
        std::fprintf(stderr,
                     "# ernie55 hca_detail (ms/step, 37L): qkv_comp=%.2f prep=%.2f qup_gate=%.2f q_rope=%.2f "
                     "attn_tc=%.2f vha=%.2f woa_gate=%.2f o_proj=%.2f\n",
                     hca_d.qkv_comp_ms / steps, hca_d.prep_ms / steps, hca_d.qup_gate_ms / steps,
                     hca_d.q_norm_rope_ms / steps, hca_d.attn_tc_ms / steps, hca_d.vha_invrope_ms / steps,
                     hca_d.woa_gate_ms / steps, hca_d.o_proj_ms / steps);
        std::fprintf(stderr,
                     "# ernie55 dsa_detail (ms/step, 6L): qkv_idx=%.2f prep=%.2f qbg=%.2f q_abs=%.2f "
                     "idx=%.2f mla_tc=%.2f v_exp=%.2f vha=%.2f o_proj=%.2f\n",
                     dsa_d.qkv_idx_ms / steps, dsa_d.prep_ms / steps, dsa_d.qbg_ms / steps,
                     dsa_d.q_abs_ms / steps, dsa_d.idx_score_ms / steps, dsa_d.mla_tc_ms / steps,
                     dsa_d.v_exp_ms / steps, dsa_d.vha_gate_ms / steps, dsa_d.o_proj_ms / steps);
        std::fprintf(stderr,
                     "# ernie55 moe_detail (ms/step, 42L): router=%.2f fc1_lat=%.2f prerot=%.2f r_fc1=%.2f "
                     "r_fc2=%.2f shared=%.2f finish=%.2f\n",
                     moe_d.router_ms / steps, moe_d.fc1_lat_ms / steps, moe_d.prerot_r_ms / steps,
                     moe_d.routed_fc1_ms / steps, moe_d.routed_fc2_ms / steps, moe_d.shared_ms / steps,
                     moe_d.finish_ms / steps);
    }
};

// ------------------------------------------------------------------------------------------------ Engine
struct Engine {
    Options o;
    cublasHandle_t blas = nullptr;
    void* cublas_ws = nullptr;
    cudaStream_t stream = 0;
    cudaStream_t stream_sh = 0;
    cudaEvent_t  ev_moe_fork = nullptr, ev_moe_join = nullptr;
    std::vector<std::unique_ptr<Image>> img;
    std::vector<std::unique_ptr<Image>> dense_bf16_keep; // kept only if !mx8_dense && hybrid
    std::vector<ernie55::mx8::Mx8Arena> mx8_arena;

    // Hybrid 5090 offload state
    HybridHostBacking hybrid_host;
    HybridExpertCache hybrid_cache;

    int max_pps = 0;
    int pool_pages = 0;
    int max_comp_blks = 0;

    // Global weights & precomputed tables
    const __nv_bfloat16* embed = nullptr; // non-null in resident mode; null in hybrid mode (host_embed used)
    ernie55::HcHeadWeights hc_head;
    const __nv_bfloat16* final_norm = nullptr;
    const __nv_bfloat16* multimax_ranges = nullptr;
    const __nv_bfloat16* multimax_ts = nullptr;
    std::vector<const __nv_bfloat16*> lm_head;
    std::vector<ernie55::mx8::Mx8Arena> mx8_head_arena;
    std::vector<ernie55::mx8::Mx8Mat> lm_head_mx8;
    std::vector<int> v_lo, v_rows;

    float* d_yarn_inv_freq = nullptr; // [32] FP32
    float* d_vha_mix_pool = nullptr;  // [43 * 64 * 64] FP32
    float* d_dsa_sinks_f32 = nullptr; // [6 * nloc * nh_loc] FP32

    struct RankLayer {
        ernie55::HcaWeights      hca_w;
        ernie55::HcaLayerKv      hca_kv;
        ernie55::DsaWeights      dsa_w;
        ernie55::DsaLayerKv      dsa_kv;
        ernie55::DenseMlpWeights dense_w;
        ernie55::MoeWeights      moe_w;
    };
    struct Layer {
        int l = 0;
        bool is_dsa = false;
        int dsa_idx = -1; // 0..5
        int hca_idx = -1; // 0..36
        bool is_moe = false;
        int moe_idx = -1; // 0..41
        ernie55::HcWeights attn_hc;
        ernie55::HcWeights mlp_hc;
        const __nv_bfloat16* in_ln = nullptr;
        const __nv_bfloat16* post_attn_ln = nullptr;
        std::vector<RankLayer> rk;
    };
    std::vector<Layer> L;

    // Shared step scratch buffers
    __nv_bfloat16* hyper = nullptr;
    __nv_bfloat16* mixed_ln = nullptr;
    __nv_bfloat16* parts = nullptr;
    uint8_t* mx8_x8 = nullptr;
    ernie55::HcScratch       hc_sc{};
    ernie55::HcaScratch      hca_sc{};
    ernie55::DsaScratch      dsa_sc{};
    ernie55::DenseMlpScratch dense_sc{};
    ernie55::MoeScratch      moe_sc{};

    float*   head_logits = nullptr;
    float*   head_argmax_val = nullptr;
    int*     head_argmax_id = nullptr;
    uint8_t* head_dev_pkt = nullptr;
    uint8_t* head_host_parts = nullptr;
    int*     d_tok = nullptr;
    int*     d_slot = nullptr;
    int*     d_pos = nullptr;
    int*     h_meta_pinned = nullptr;
    int*     d_page_table = nullptr;
    std::vector<int> h_page_table;

    // CUDA Graphs for resident TP=1 decode M <= 12
    struct GraphKey {
        int  M = 0;
        bool big_kv = false;
        bool sparse = false;
        int  idx_bucket = 0;
        bool greedy_fast = false;
        bool operator<(const GraphKey& r) const {
            return std::tie(M, big_kv, sparse, idx_bucket, greedy_fast) <
                   std::tie(r.M, r.big_kv, r.sparse, r.idx_bucket, r.greedy_fast);
        }
    };
    std::map<GraphKey, cudaGraphExec_t> graphs;

    // Optional collective hooks (set by ernie55_m4.cu for TP=2/4 over RDMA)
    std::function<void(int M, bool skip_head)> on_step_begin;
    std::function<void(int c, int M)> on_ar_mhc_post;
    std::function<void(int c, int M, uint8_t* host_parts, size_t row_bytes)> on_head_gather;
    std::function<void()> on_step_end;

    ProfileAccum prof{};
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;

    ~Engine() {
        clear_graphs();
        if (head_host_parts) cudaFreeHost(head_host_parts);
        if (h_meta_pinned) cudaFreeHost(h_meta_pinned);
        if (head_argmax_val) cudaFree(head_argmax_val);
        if (head_argmax_id) cudaFree(head_argmax_id);
        if (d_yarn_inv_freq) cudaFree(d_yarn_inv_freq);
        if (d_vha_mix_pool) cudaFree(d_vha_mix_pool);
        if (d_dsa_sinks_f32) cudaFree(d_dsa_sinks_f32);
        if (ev_a) cudaEventDestroy(ev_a);
        if (ev_b) cudaEventDestroy(ev_b);
        if (ev_moe_fork) cudaEventDestroy(ev_moe_fork);
        if (ev_moe_join) cudaEventDestroy(ev_moe_join);
        if (prof.hca_d.ev_a) cudaEventDestroy(prof.hca_d.ev_a);
        if (prof.hca_d.ev_b) cudaEventDestroy(prof.hca_d.ev_b);
        if (prof.dsa_d.ev_a) cudaEventDestroy(prof.dsa_d.ev_a);
        if (prof.dsa_d.ev_b) cudaEventDestroy(prof.dsa_d.ev_b);
        if (prof.moe_d.ev_a) cudaEventDestroy(prof.moe_d.ev_a);
        if (prof.moe_d.ev_b) cudaEventDestroy(prof.moe_d.ev_b);
        if (blas) cublasDestroy(blas);
        if (cublas_ws) cudaFree(cublas_ws);
        if (stream_sh) cudaStreamDestroy(stream_sh);
        if (stream) cudaStreamDestroy(stream);
    }

    void clear_graphs() {
        for (auto& kv : graphs) {
            if (kv.second) cudaGraphExecDestroy(kv.second);
        }
        graphs.clear();
    }

    static std::string lname(int l, const std::string& suf) {
        char b[96];
        std::snprintf(b, sizeof(b), "L%03d.%s", l, suf.c_str());
        return b;
    }

    void reset_slot(int slot) {
        if (slot < 0 || slot >= o.slots) return;
        // Zero HCA partial compressor accumulators for clean slot reuse
        for (int l = 0; l < geom::L_MAIN; ++l) {
            if (L[size_t(l)].is_dsa) continue;
            for (int i = 0; i < o.nloc; ++i) {
                auto& hkv = L[size_t(l)].rk[size_t(i)].hca_kv;
                const size_t pbytes = static_cast<size_t>(geom::HCA_CR) * geom::HCA_HD * sizeof(float);
                ERNIE55_CK(cudaMemsetAsync(hkv.part_kv + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD, 0, pbytes, stream));
                ERNIE55_CK(cudaMemsetAsync(hkv.part_sc + static_cast<size_t>(slot) * geom::HCA_CR * geom::HCA_HD, 0, pbytes, stream));
            }
        }
    }

    void set_slot_pages(int slot, const std::vector<int>& phys_pages) {
        if (slot < 0 || slot >= o.slots) return;
        const int n = std::min(static_cast<int>(phys_pages.size()), max_pps);
        if (n <= 0) return;
        std::memcpy(h_page_table.data() + static_cast<size_t>(slot) * max_pps, phys_pages.data(),
                    static_cast<size_t>(n) * sizeof(int));
        ERNIE55_CK(cudaMemcpyAsync(d_page_table + static_cast<size_t>(slot) * max_pps,
                                   phys_pages.data(), static_cast<size_t>(n) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    }

    void seed_bench_kv(int fill_pos, int sessions) {
        if (fill_pos <= 0) return;
        if (fill_pos >= o.cap) die("seed_bench_kv: fill_pos >= cap");
        const int need_pages = (fill_pos + geom::KV_PAGE - 1) / geom::KV_PAGE;
        if (need_pages * sessions > pool_pages)
            die("seed_bench_kv: sessions * pages exceeds pool_pages");
        for (int l = 0; l < geom::L_MAIN; ++l) {
            for (int i = 0; i < o.nloc; ++i) {
                auto& rl = L[size_t(l)].rk[size_t(i)];
                if (L[size_t(l)].is_dsa) {
                    const size_t n_idx = size_t(pool_pages) * geom::KV_PAGE * geom::IDX_DIM;
                    if (o.kv_fmt == ernie55::KvFmt::BF16) {
                        k_ernie55_bench_seed_bf16<<<4096, 256, 0, stream>>>(rl.dsa_kv.idx_bf16, n_idx, 0x12345678u + uint32_t(l * 17 + i));
                        const size_t n_kv = size_t(pool_pages) * geom::KV_PAGE * geom::KVA;
                        k_ernie55_bench_seed_bf16<<<4096, 256, 0, stream>>>(
                            reinterpret_cast<__nv_bfloat16*>(rl.dsa_kv.kv_pool), n_kv, 0x9abcdef0u + uint32_t(l * 31 + i));
                    } else {
                        k_ernie55_bench_seed_u8<<<4096, 256, 0, stream>>>(rl.dsa_kv.idx_q, n_idx, 0x12345678u + uint32_t(l * 17 + i));
                        k_ernie55_bench_seed_half<<<1024, 256, 0, stream>>>(rl.dsa_kv.idx_sc, size_t(pool_pages) * geom::KV_PAGE * 4, 0.05f);
                        const size_t n_kv_b = size_t(pool_pages) * geom::KV_PAGE * ernie55::dsa_kv_tok_bytes(o.kv_fmt);
                        k_ernie55_bench_seed_u8<<<4096, 256, 0, stream>>>(rl.dsa_kv.kv_pool, n_kv_b, 0x9abcdef0u + uint32_t(l * 31 + i));
                    }
                } else {
                    const size_t n_sw = size_t(o.slots) * geom::HCA_SWA_RING * geom::HCA_HD;
                    k_ernie55_bench_seed_bf16<<<2048, 256, 0, stream>>>(rl.hca_kv.swa_ring, n_sw, 0x55aa33ccu + uint32_t(l * 13 + i));
                    const size_t n_cp = size_t(o.slots) * max_comp_blks * geom::HCA_HD;
                    k_ernie55_bench_seed_bf16<<<2048, 256, 0, stream>>>(rl.hca_kv.comp_kv, n_cp, 0x33cc55aau + uint32_t(l * 19 + i));
                }
            }
        }
        ERNIE55_CK(cudaStreamSynchronize(stream));
    }

    static std::string default_widths_path(const std::string& release_dir) {
        const std::array<const char*, 5> cands = {
            "/manifests/exlr8-manifest-k4.json",
            "/manifests/exlr8-manifest-k3.json",
            "/manifests/exlr8-manifest-k2.json",
            "/manifests/k4.json",
            "/manifests/k3.json",
        };
        for (const char* c : cands) {
            const std::string p = release_dir + c;
            if (std::filesystem::exists(p)) return p;
        }
        return release_dir + "/manifests/exlr8-manifest-k4.json";
    }

    void init(const std::string& release_dir, const Options& opts) {
        o = opts;
        if (!geom::rank_count_ok(o.nr)) die("unsupported TP nr=" + std::to_string(o.nr));
        if (o.widths_path.empty()) o.widths_path = default_widths_path(release_dir);

        // Apply typed knobs (CLI overrides when explicitly set!)
        if (!o.kv_fmt_set)   o.kv_fmt = parse_kv_fmt(knob::ernie55_kv_format());
        if (!o.mx8_set)      o.mx8_dense = knob::ernie55_mx8_dense();
        if (!o.mx8_head_set) o.mx8_head = knob::ernie55_mx8_head();
        if (!o.graphs_set)   o.use_graphs = knob::ernie55_graphs();
        HybridConfig hcfg = load_hybrid_config();
        if (o.hybrid_set) hcfg.enabled = o.use_hybrid;
        else              o.use_hybrid = hcfg.enabled;
        o.profile = o.profile || knob::ernie55_profile();

        if (o.use_hybrid) {
            if (o.nr != 1 || o.nloc != 1) die("hybrid mode requires TP=1 (nr=1, nloc=1)");
            o.use_graphs = false; // pageable offload runs eager per-layer staging
        }

        ERNIE55_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERNIE55_CK(cudaStreamCreateWithFlags(&stream_sh, cudaStreamNonBlocking));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) die("cublasCreate failed");
        cublasSetStream(blas, stream);
        ERNIE55_CK(cudaMalloc(&cublas_ws, 32ull << 20));
        cublasSetWorkspace(blas, cublas_ws, 32ull << 20);
        ERNIE55_CK(cudaEventCreate(&ev_a));
        ERNIE55_CK(cudaEventCreate(&ev_b));
        ERNIE55_CK(cudaEventCreateWithFlags(&ev_moe_fork, cudaEventDisableTiming));
        ERNIE55_CK(cudaEventCreateWithFlags(&ev_moe_join, cudaEventDisableTiming));
        ERNIE55_CK(cudaEventCreate(&prof.hca_d.ev_a));
        ERNIE55_CK(cudaEventCreate(&prof.hca_d.ev_b));
        ERNIE55_CK(cudaEventCreate(&prof.dsa_d.ev_a));
        ERNIE55_CK(cudaEventCreate(&prof.dsa_d.ev_b));
        ERNIE55_CK(cudaEventCreate(&prof.moe_d.ev_a));
        ERNIE55_CK(cudaEventCreate(&prof.moe_d.ev_b));

        // Pre-configure dynamic shared memory attributes BEFORE any graph capture
        ernie55::ensure_hca_tc_smem();
        ernie55::ensure_dsa_tc_smem();
        ernie55::mx8::ensure_large_smem_attr();

        // Precompute FP64->FP32 HCA YaRN inverse frequencies on GPU
        const std::vector<float> yarn_inv = ernie55::build_hca_yarn_inv_freq();
        d_yarn_inv_freq = dalloc<float>(yarn_inv.size());
        ERNIE55_CK(cudaMemcpy(d_yarn_inv_freq, yarn_inv.data(), yarn_inv.size() * sizeof(float), cudaMemcpyHostToDevice));

        // 1. Stage 1: Load permanent VRAM weights (part=3 hybrid, part=1 resident MXFP8, or part=0 resident BF16)
        const uint32_t perm_part = o.use_hybrid ? 3u : (o.mx8_dense ? 1u : 0u);
        img.resize(size_t(o.nloc));
        mx8_arena.resize(size_t(o.nloc));
        for (int i = 0; i < o.nloc; ++i) {
            img[size_t(i)] = std::make_unique<Image>();
            img[size_t(i)]->load_part(release_dir, o.widths_path, o.nr, o.rank0 + i, perm_part);
        }

        // 2. Bind global weights + boot-time F32 transpose of hc_head.fn [16384, 4] -> [4, 16384]
        const Image& im0 = *img[0];
        embed = o.use_hybrid ? nullptr : im0.get<__nv_bfloat16>("embed.weight");
        float* f32_tr_tmp = dalloc<float>(static_cast<size_t>(geom::HC_DIM) * geom::HC_MIX);

        hc_head.fn    = im0.get<float>("hc_head.fn");
        hc_head.base  = im0.get<float>("hc_head.base");
        hc_head.scale = im0.get<float>("hc_head.scale");
        ernie55::transpose_f32_inplace(im0.get_mut<float>("hc_head.fn"), f32_tr_tmp, geom::HC_DIM, geom::HC, stream);

        final_norm      = im0.get<__nv_bfloat16>("norm.weight");
        multimax_ranges = im0.get<__nv_bfloat16>("lm_head.multimax_ranges");
        multimax_ts     = im0.get<__nv_bfloat16>("lm_head.multimax_ts");

        lm_head.resize(size_t(o.nloc));
        mx8_head_arena.resize(size_t(o.nloc));
        lm_head_mx8.resize(size_t(o.nloc));
        v_lo.resize(size_t(o.nloc));
        v_rows.resize(size_t(o.nloc));
        int max_vr = 0;
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            lm_head[size_t(i)] = img[size_t(i)]->get<__nv_bfloat16>("lm_head.weight");
            v_lo[size_t(i)]    = geom::vocab_row0(o.nr, r);
            v_rows[size_t(i)]  = static_cast<int>(img[size_t(i)]->entry("lm_head.weight").shape[0]);
            max_vr = std::max(max_vr, v_rows[size_t(i)]);
            if (o.mx8_head) {
                auto& ha = mx8_head_arena[size_t(i)];
                ha.reserve(v_rows[size_t(i)], geom::HID);
                if (!ha.alloc()) die("mx8_head_arena cudaMalloc failed");
                lm_head_mx8[size_t(i)] = ha.add(lm_head[size_t(i)], v_rows[size_t(i)], geom::HID, stream);
            }
        }

        // 3. Bind permanent layer structures, boot-transpose mHC weights, precompute VHA I+V@U^T,
        //    convert DSA BF16 sinks -> F32, and widen VRAM-resident experts.
        const int nh_loc = geom::nh_local(o.nr);
        const int ng_loc = geom::hca_ogroups_local(o.nr);
        const int eb_loc = geom::exp_blocks_local(o.nr);
        const int dense_i_loc = geom::dense_i_local(o.nr);

        d_vha_mix_pool = dalloc<float>(static_cast<size_t>(geom::L_MAIN) * geom::HEADS * geom::HEADS);
        d_dsa_sinks_f32 = dalloc<float>(static_cast<size_t>(geom::N_DSA) * o.nloc * nh_loc);

        std::vector<std::vector<long long>> hyb_tabs(o.use_hybrid ? geom::N_MOE_LAYERS : 0);
        std::vector<std::vector<long long>> hyb_kmixes(o.use_hybrid ? geom::N_MOE_LAYERS : 0);
        const bool hyb_cpu_ov = o.use_hybrid && hcfg.cpu_overlap;
        std::vector<std::vector<uint16_t>> hyb_sgs(hyb_cpu_ov ? geom::N_MOE_LAYERS : 0);
        std::vector<std::vector<uint16_t>> hyb_v1bs(hyb_cpu_ov ? geom::N_MOE_LAYERS : 0);
        std::vector<std::vector<uint16_t>> hyb_v2s(hyb_cpu_ov ? geom::N_MOE_LAYERS : 0);
        std::vector<std::vector<uint16_t>> hyb_v3bs(hyb_cpu_ov ? geom::N_MOE_LAYERS : 0);

        L.resize(geom::L_MAIN);
        for (int l = 0; l < geom::L_MAIN; ++l) {
            Layer& y = L[size_t(l)];
            y.l       = l;
            y.is_dsa  = geom::is_dsa_layer(l);
            y.dsa_idx = geom::dsa_slot_of(l);
            y.hca_idx = geom::hca_slot_of(l);
            y.is_moe  = geom::is_moe(l);
            y.moe_idx = y.is_moe ? (l - geom::FIRST_MOE) : -1;

            y.attn_hc.fn    = im0.get<float>(lname(l, "attn_hc.fn"));
            y.attn_hc.base  = im0.get<float>(lname(l, "attn_hc.bias"));
            y.attn_hc.scale = im0.get<float>(lname(l, "attn_hc.scale"));
            ernie55::transpose_f32_inplace(im0.get_mut<float>(lname(l, "attn_hc.fn")), f32_tr_tmp, geom::HC_DIM, geom::HC_MIX, stream);

            y.mlp_hc.fn    = im0.get<float>(lname(l, "mlp_hc.fn"));
            y.mlp_hc.base  = im0.get<float>(lname(l, "mlp_hc.bias"));
            y.mlp_hc.scale = im0.get<float>(lname(l, "mlp_hc.scale"));
            ernie55::transpose_f32_inplace(im0.get_mut<float>(lname(l, "mlp_hc.fn")), f32_tr_tmp, geom::HC_DIM, geom::HC_MIX, stream);

            y.in_ln        = im0.get<__nv_bfloat16>(lname(l, "input_norm"));
            y.post_attn_ln = im0.get<__nv_bfloat16>(lname(l, "post_attn_norm"));

            float* vha_mix = d_vha_mix_pool + static_cast<size_t>(l) * geom::HEADS * geom::HEADS;
            ernie55::k_ernie55_build_vha_mix<<<(geom::HEADS * geom::HEADS + 255) / 256, 256, 0, stream>>>(
                im0.get<__nv_bfloat16>(lname(l, "attn.vha_u")),
                im0.get<__nv_bfloat16>(lname(l, "attn.vha_v")), vha_mix);

            y.rk.resize(size_t(o.nloc));
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                const Image& im = *img[size_t(i)];
                RankLayer& rl = y.rk[size_t(i)];
                if (y.is_dsa) {
                    rl.dsa_w.nh_loc       = nh_loc;
                    rl.dsa_w.h_off        = r * nh_loc;
                    rl.dsa_w.q_a_ln       = im.get<__nv_bfloat16>(lname(l, "attn.q_a_norm"));
                    rl.dsa_w.kv_a_ln      = im.get<__nv_bfloat16>(lname(l, "attn.kv_a_norm"));
                    rl.dsa_w.vha_mix      = vha_mix;
                    rl.dsa_w.idx_k_norm_w = im.get<__nv_bfloat16>(lname(l, "idx.k_norm_w"));
                    rl.dsa_w.idx_k_norm_b = im.get<__nv_bfloat16>(lname(l, "idx.k_norm_b"));
                    rl.dsa_w.idx_w_proj   = im.get<__nv_bfloat16>(lname(l, "idx.weights_proj"));

                    float* sink_dst = d_dsa_sinks_f32 + (static_cast<size_t>(y.dsa_idx) * o.nloc + i) * nh_loc;
                    ernie55::k_ernie55_bf16_to_f32_vec<<<(nh_loc + 63) / 64, 64, 0, stream>>>(
                        im.get<__nv_bfloat16>(lname(l, "attn.sinks")), sink_dst, nh_loc);
                    rl.dsa_w.sinks = sink_dst;
                } else {
                    rl.hca_w.nh_loc    = nh_loc;
                    rl.hca_w.ng_loc    = ng_loc;
                    rl.hca_w.h_off     = r * nh_loc;
                    rl.hca_w.q_norm    = im.get<__nv_bfloat16>(lname(l, "attn.q_norm"));
                    rl.hca_w.kv_norm   = im.get<__nv_bfloat16>(lname(l, "attn.kv_norm"));
                    rl.hca_w.comp_norm = im.get<__nv_bfloat16>(lname(l, "attn.comp_norm"));
                    rl.hca_w.comp_ape  = im.get<float>(lname(l, "attn.comp_ape"));
                    rl.hca_w.sinks     = im.get<float>(lname(l, "attn.sinks"));
                    rl.hca_w.vha_mix   = vha_mix;
                }

                if (!y.is_moe) {
                    rl.dense_w.dense_i_loc = dense_i_loc;
                } else {
                    rl.moe_w.router_w0   = im.get<float>(lname(l, "router.w0"));
                    rl.moe_w.router_w1   = im.get<float>(lname(l, "router.w1"));
                    rl.moe_w.router_bias = im.get<float>(lname(l, "router.bias"));
                    rl.moe_w.router_sc   = im.get<float>(lname(l, "router.scale"));
                    rl.moe_w.lat_norm    = im.get<__nv_bfloat16>(lname(l, "moe.latent_norm"));

                    rl.moe_w.routed.tab  = im.get<long long>(lname(l, "tr3.table"));
                    rl.moe_w.routed.kmix = im.opt<long long>(lname(l, "tr3.kmix"));
                    rl.moe_w.routed.sg   = im.get<__half>(lname(l, "tr3.sg"));
                    rl.moe_w.routed.v1b  = im.get<__half>(lname(l, "tr3.v1b"));
                    rl.moe_w.routed.v2   = im.get<__half>(lname(l, "tr3.v2"));
                    rl.moe_w.routed.v3b  = im.get<__half>(lname(l, "tr3.v3b"));
                    rl.moe_w.routed.ne   = geom::NEXP;
                    rl.moe_w.routed.eb   = eb_loc;

                    if (o.use_hybrid) {
                        im.read_host<long long>(lname(l, "tr3.table"), hyb_tabs[size_t(y.moe_idx)]);
                        if (im.find(lname(l, "tr3.kmix")))
                            im.read_host<long long>(lname(l, "tr3.kmix"), hyb_kmixes[size_t(y.moe_idx)]);
                        if (hyb_cpu_ov) {
                            im.read_host<uint16_t>(lname(l, "tr3.sg"),  hyb_sgs[size_t(y.moe_idx)]);
                            im.read_host<uint16_t>(lname(l, "tr3.v1b"), hyb_v1bs[size_t(y.moe_idx)]);
                            im.read_host<uint16_t>(lname(l, "tr3.v2"),  hyb_v2s[size_t(y.moe_idx)]);
                            im.read_host<uint16_t>(lname(l, "tr3.v3b"), hyb_v3bs[size_t(y.moe_idx)]);
                        }
                    } else {
                        rl.moe_w.routed.fc1 = im.get<uint8_t>(lname(l, "tr3.fc1"));
                        rl.moe_w.routed.fc2 = im.get<uint8_t>(lname(l, "tr3.fc2"));
                        ernie55::k_ernie55_tr3_widen_fc1<geom::EHB><<<geom::NEXP * 2 * eb_loc * geom::EHB, 256, 0, stream>>>(
                            im.get_mut<uint8_t>(lname(l, "tr3.fc1")), rl.moe_w.routed.tab, rl.moe_w.routed.kmix, geom::NEXP, eb_loc);
                        ernie55::k_ernie55_tr3_widen_fc2<geom::EHB><<<geom::NEXP * eb_loc * geom::EHB, 256, 0, stream>>>(
                            im.get_mut<uint8_t>(lname(l, "tr3.fc2")), rl.moe_w.routed.tab, geom::NEXP, eb_loc);
                    }

                    rl.moe_w.shared.fc1  = im.get<uint8_t>(lname(l, "sh_tr3.fc1"));
                    rl.moe_w.shared.fc2  = im.get<uint8_t>(lname(l, "sh_tr3.fc2"));
                    rl.moe_w.shared.tab  = im.get<long long>(lname(l, "sh_tr3.table"));
                    rl.moe_w.shared.kmix = im.opt<long long>(lname(l, "sh_tr3.kmix"));
                    rl.moe_w.shared.sg   = im.get<__half>(lname(l, "sh_tr3.sg"));
                    rl.moe_w.shared.v1b  = im.get<__half>(lname(l, "sh_tr3.v1b"));
                    rl.moe_w.shared.v2   = im.get<__half>(lname(l, "sh_tr3.v2"));
                    rl.moe_w.shared.v3b  = im.get<__half>(lname(l, "sh_tr3.v3b"));
                    rl.moe_w.shared.ne   = 1;
                    rl.moe_w.shared.eb   = eb_loc;
                    ernie55::k_ernie55_tr3_widen_fc1<geom::HB><<<1 * 2 * eb_loc * geom::HB, 256, 0, stream>>>(
                        im.get_mut<uint8_t>(lname(l, "sh_tr3.fc1")), rl.moe_w.shared.tab, rl.moe_w.shared.kmix, 1, eb_loc);
                    ernie55::k_ernie55_tr3_widen_fc2<geom::HB><<<1 * eb_loc * geom::HB, 256, 0, stream>>>(
                        im.get_mut<uint8_t>(lname(l, "sh_tr3.fc2")), rl.moe_w.shared.tab, 1, eb_loc);
                }
            }
        }
        ERNIE55_CK(cudaStreamSynchronize(stream));
        cudaFree(f32_tr_tmp);

        // 4. Stage 2: Load/quantize dense linears (freeing ~11.34 GiB BF16 staging BEFORE KV & hot-expert cache!)
        __nv_bfloat16* kv_b_tmp = dalloc<__nv_bfloat16>(static_cast<size_t>(nh_loc) * geom::KVB_HEAD * geom::KV_LORA);
        dense_bf16_keep.resize(size_t(o.nloc));
        const bool need_part2 = o.mx8_dense || o.use_hybrid;
        for (int i = 0; i < o.nloc; ++i) {
            std::unique_ptr<Image> dense_tmp;
            if (need_part2) {
                dense_tmp = std::make_unique<Image>();
                dense_tmp->load_part(release_dir, o.widths_path, o.nr, o.rank0 + i, 2u);
            }
            const Image& dim = need_part2 ? *dense_tmp : *img[size_t(i)];
            auto& ar = mx8_arena[size_t(i)];

            if (o.mx8_dense) {
                for (int l = 0; l < geom::L_MAIN; ++l) {
                    const Layer& y = L[size_t(l)];
                    if (y.is_dsa) {
                        ar.reserve(geom::QA, geom::HID);
                        ar.reserve(nh_loc * geom::QK_HEAD, geom::QA);
                        ar.reserve(geom::KVA, geom::HID);
                        ar.reserve(nh_loc * geom::KV_LORA, geom::QK_NOPE);
                        ar.reserve(nh_loc * geom::V_HEAD, geom::KV_LORA);
                        ar.reserve(nh_loc * geom::V_HEAD, geom::QA);
                        ar.reserve(geom::HID, nh_loc * geom::V_HEAD);
                        ar.reserve(geom::IDX_HEADS * geom::IDX_DIM, geom::QA);
                        ar.reserve(geom::IDX_DIM, geom::HID);
                    } else {
                        ar.reserve(geom::HCA_Q_LORA, geom::HID);
                        ar.reserve(nh_loc * geom::HCA_HD, geom::HCA_Q_LORA);
                        ar.reserve(geom::HCA_HD, geom::HID);
                        ar.reserve(geom::HCA_HD, geom::HID);
                        ar.reserve(geom::HCA_HD, geom::HID);
                        ar.reserve(ng_loc * geom::HCA_O_LORA, geom::HCA_HPG * geom::HCA_HD);
                        ar.reserve(ng_loc * geom::HCA_O_LORA, geom::HCA_Q_LORA);
                        ar.reserve(geom::HID, ng_loc * geom::HCA_O_LORA);
                    }
                    if (!y.is_moe) {
                        ar.reserve(dense_i_loc, geom::HID);
                        ar.reserve(dense_i_loc, geom::HID);
                        ar.reserve(geom::HID, dense_i_loc);
                    } else {
                        ar.reserve(geom::EH, geom::HID);
                        ar.reserve(geom::HID, geom::EH);
                    }
                }
                if (!ar.alloc()) die("mx8_arena cudaMalloc failed");
            }

            auto bind_w = [&](const __nv_bfloat16* src, int N, int K) -> ernie55::mx8::LinW {
                if (o.mx8_dense) return ernie55::mx8::LinW::from_mx8(ar.add(src, N, K, stream));
                return ernie55::mx8::LinW::from_bf16(src, N, K);
            };

            for (int l = 0; l < geom::L_MAIN; ++l) {
                Layer& y = L[size_t(l)];
                RankLayer& rl = y.rk[size_t(i)];
                if (y.is_dsa) {
                    rl.dsa_w.q_a_proj  = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.q_a_proj")),  geom::QA, geom::HID);
                    rl.dsa_w.q_b_proj  = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.q_b_proj")),  nh_loc * geom::QK_HEAD, geom::QA);
                    rl.dsa_w.kv_a_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.kv_a_proj")), geom::KVA, geom::HID);
                    rl.dsa_w.gate_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.gate_proj")), nh_loc * geom::V_HEAD, geom::QA);
                    rl.dsa_w.o_proj    = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.o_proj")),    geom::HID, nh_loc * geom::V_HEAD);
                    rl.dsa_w.idx_wq_b  = bind_w(dim.get<__nv_bfloat16>(lname(l, "idx.wq_b")),       geom::IDX_HEADS * geom::IDX_DIM, geom::QA);
                    rl.dsa_w.idx_wk    = bind_w(dim.get<__nv_bfloat16>(lname(l, "idx.wk")),         geom::IDX_DIM, geom::HID);

                    __nv_bfloat16* kv_b = dim.get_mut<__nv_bfloat16>(lname(l, "attn.kv_b_proj"));
                    const size_t kv_b_el = static_cast<size_t>(nh_loc) * geom::KVB_HEAD * geom::KV_LORA;
                    ERNIE55_CK(cudaMemcpyAsync(kv_b_tmp, kv_b, kv_b_el * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                    __nv_bfloat16* w_uk_t = kv_b;
                    __nv_bfloat16* w_uv   = kv_b + static_cast<size_t>(nh_loc) * geom::KV_LORA * geom::QK_NOPE;
                    ernie55::k_ernie55_unpack_kv_b<<<nh_loc * geom::KVB_HEAD, 256, 0, stream>>>(kv_b_tmp, w_uk_t, w_uv, nh_loc);
                    rl.dsa_w.w_uk_t = bind_w(w_uk_t, nh_loc * geom::KV_LORA, geom::QK_NOPE);
                    rl.dsa_w.w_uv   = bind_w(w_uv,   nh_loc * geom::V_HEAD,  geom::KV_LORA);
                } else {
                    rl.hca_w.q_down     = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.q_down")),     geom::HCA_Q_LORA, geom::HID);
                    rl.hca_w.q_up       = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.q_up")),       nh_loc * geom::HCA_HD, geom::HCA_Q_LORA);
                    rl.hca_w.kv_proj    = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.kv_proj")),    geom::HCA_HD, geom::HID);
                    rl.hca_w.comp_wkv   = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.comp_wkv")),   geom::HCA_HD, geom::HID);
                    rl.hca_w.comp_wgate = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.comp_wgate")), geom::HCA_HD, geom::HID);
                    const __nv_bfloat16* woa_all = dim.get<__nv_bfloat16>(lname(l, "attn.wo_a"));
                    rl.hca_w.wo_a_flat = bind_w(woa_all, ng_loc * geom::HCA_O_LORA, geom::HCA_HPG * geom::HCA_HD);
                    for (int g = 0; g < ng_loc; ++g) {
                        if (rl.hca_w.wo_a_flat.is_mx8()) {
                            ernie55::mx8::Mx8Mat sub{};
                            sub.N  = geom::HCA_O_LORA;
                            sub.K  = geom::HCA_HPG * geom::HCA_HD;
                            sub.q  = rl.hca_w.wo_a_flat.mx8.q  + static_cast<size_t>(g) * sub.N * sub.K;
                            sub.sc = rl.hca_w.wo_a_flat.mx8.sc + static_cast<size_t>(g) * sub.N * (sub.K >> 5);
                            rl.hca_w.wo_a[g] = ernie55::mx8::LinW::from_mx8(sub);
                        } else {
                            rl.hca_w.wo_a[g] = ernie55::mx8::LinW::from_bf16(
                                woa_all + static_cast<size_t>(g) * geom::HCA_O_LORA * (geom::HCA_HPG * geom::HCA_HD),
                                geom::HCA_O_LORA, geom::HCA_HPG * geom::HCA_HD);
                        }
                    }
                    rl.hca_w.gate_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.gate_proj")), ng_loc * geom::HCA_O_LORA, geom::HCA_Q_LORA);
                    rl.hca_w.o_proj    = bind_w(dim.get<__nv_bfloat16>(lname(l, "attn.o_proj")),    geom::HID, ng_loc * geom::HCA_O_LORA);
                }
                if (!y.is_moe) {
                    rl.dense_w.gate_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "mlp.gate_proj")), dense_i_loc, geom::HID);
                    rl.dense_w.up_proj   = bind_w(dim.get<__nv_bfloat16>(lname(l, "mlp.up_proj")),   dense_i_loc, geom::HID);
                    rl.dense_w.down_proj = bind_w(dim.get<__nv_bfloat16>(lname(l, "mlp.down_proj")), geom::HID, dense_i_loc);
                } else {
                    rl.moe_w.fc1_lat = bind_w(dim.get<__nv_bfloat16>(lname(l, "moe.fc1_lat")), geom::EH, geom::HID);
                    rl.moe_w.fc2_lat = bind_w(dim.get<__nv_bfloat16>(lname(l, "moe.fc2_lat")), geom::HID, geom::EH);
                }
            }
            ERNIE55_CK(cudaStreamSynchronize(stream));
            if (o.mx8_dense) {
                std::fprintf(stderr, "# rank %d: quantized dense linears to MXFP8 (%.2f GiB, freed %.2f GiB BF16 staging)\n",
                             o.rank0 + i, ar.total_bytes / double(1ull << 30), dense_tmp->pack.bytes() / double(1ull << 30));
            } else if (o.use_hybrid) {
                dense_bf16_keep[size_t(i)] = std::move(dense_tmp);
            }
        }
        cudaFree(kv_b_tmp);

        // 5. Stage 3 (Hybrid only): load part=4 pageable host backing & CPU-widen routed experts
        if (o.use_hybrid) {
            hybrid_host.load_part4(release_dir, o.widths_path, o.nr, o.rank0);
            hybrid_cache.init_tables(hcfg, hybrid_host, hyb_tabs, hyb_kmixes, o.max_rows, stream);
            if (hyb_cpu_ov) hybrid_cache.init_cpu_scales(hyb_sgs, hyb_v1bs, hyb_v2s, hyb_v3bs);
            for (int l = geom::FIRST_MOE; l < geom::L_MAIN; ++l) {
                L[size_t(l)].rk[0].moe_w.routed.ptab = hybrid_cache.layer_ptab(L[size_t(l)].moe_idx);
            }
        }

        // 6. Stage 4: Allocate HCA + DSA KV caches (AFTER dense BF16 staging is freed!)
        max_pps = (o.cap + geom::KV_PAGE - 1) / geom::KV_PAGE;
        max_comp_blks = std::max(1, (o.cap + geom::HCA_CR - 1) / geom::HCA_CR);
        const int want_rows = (o.pool_rows > 0) ? o.pool_rows : (o.slots * o.cap);
        pool_pages = std::max(max_pps, (want_rows + geom::KV_PAGE - 1) / geom::KV_PAGE);

        const size_t kv_tok_b = ernie55::dsa_kv_tok_bytes(o.kv_fmt);
        const size_t idx_tok_b = (o.kv_fmt == ernie55::KvFmt::BF16) ? (geom::IDX_DIM * 2) : (geom::IDX_DIM + 4 * 2);
        const size_t dsa_bytes = size_t(geom::N_DSA) * pool_pages * geom::KV_PAGE * (kv_tok_b + idx_tok_b) * size_t(o.nloc);
        const size_t hca_bytes = size_t(geom::N_HCA) * o.slots *
            (size_t(geom::HCA_SWA_RING + max_comp_blks) * geom::HCA_HD * 2 + 2ull * geom::HCA_CR * geom::HCA_HD * 4) * size_t(o.nloc);
        std::fprintf(stderr, "# ernie55 KV pool [%s]: cap=%d (max_pps=%d, comp_blks=%d), pool_pages=%d (%d tok), slots=%d | DSA=%.2f GiB + HCA=%.2f GiB\n",
                     kv_fmt_name(o.kv_fmt), o.cap, max_pps, max_comp_blks, pool_pages, pool_pages * geom::KV_PAGE, o.slots,
                     dsa_bytes / double(1ull << 30), hca_bytes / double(1ull << 30));

        for (int l = 0; l < geom::L_MAIN; ++l) {
            Layer& y = L[size_t(l)];
            for (int i = 0; i < o.nloc; ++i) {
                RankLayer& rl = y.rk[size_t(i)];
                if (y.is_dsa) {
                    rl.dsa_kv.fmt = o.kv_fmt;
                    rl.dsa_kv.kv_pool = dalloc<uint8_t>(size_t(pool_pages) * geom::KV_PAGE * kv_tok_b);
                    if (o.kv_fmt == ernie55::KvFmt::BF16) {
                        rl.dsa_kv.idx_bf16 = dalloc<__nv_bfloat16>(size_t(pool_pages) * geom::KV_PAGE * geom::IDX_DIM);
                    } else {
                        rl.dsa_kv.idx_q  = dalloc<uint8_t>(size_t(pool_pages) * geom::KV_PAGE * geom::IDX_DIM);
                        rl.dsa_kv.idx_sc = dalloc<__half>(size_t(pool_pages) * geom::KV_PAGE * 4);
                    }
                } else {
                    rl.hca_kv.max_comp_blks = max_comp_blks;
                    rl.hca_kv.swa_ring = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HCA_SWA_RING * geom::HCA_HD);
                    rl.hca_kv.comp_kv  = dalloc<__nv_bfloat16>(size_t(o.slots) * max_comp_blks * geom::HCA_HD);
                    rl.hca_kv.part_kv  = dalloc<float>(size_t(o.slots) * geom::HCA_CR * geom::HCA_HD);
                    rl.hca_kv.part_sc  = dalloc<float>(size_t(o.slots) * geom::HCA_CR * geom::HCA_HD);
                }
            }
        }

        // 7. Allocate shared step scratch buffers
        const int Mmax = std::max(o.max_rows, 16);
        const int max_k = std::max({dense_i_loc, nh_loc * geom::V_HEAD, ng_loc * geom::HCA_O_LORA, geom::HID});
        hyper    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HC_DIM);
        mixed_ln = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        parts    = dalloc<__nv_bfloat16>(size_t(o.nr) * Mmax * geom::HID);
        mx8_x8   = dalloc<uint8_t>(ernie55::mx8::x8_bytes(Mmax, max_k));

        hc_sc.flat_f32 = dalloc<float>(size_t(Mmax) * geom::HC_DIM);
        hc_sc.mixes    = dalloc<float>(size_t(Mmax) * geom::HC_MIX);
        hc_sc.post     = dalloc<float>(size_t(Mmax) * geom::HC);
        hc_sc.comb     = dalloc<float>(size_t(Mmax) * geom::HC * geom::HC);

        hca_sc.q_lora   = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HCA_Q_LORA);
        hca_sc.q_raw    = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::HCA_HD);
        hca_sc.kv_raw   = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HCA_HD);
        hca_sc.c_kv     = dalloc<float>(size_t(Mmax) * geom::HCA_HD);
        hca_sc.c_sc     = dalloc<float>(size_t(Mmax) * geom::HCA_HD);
        hca_sc.attn_o   = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::HCA_HD);
        hca_sc.post_o   = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::HCA_HD);
        hca_sc.o_grp_in = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HCA_HPG * geom::HCA_HD);
        hca_sc.o_grp_z  = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HCA_O_LORA);
        hca_sc.z_cat    = dalloc<__nv_bfloat16>(size_t(Mmax) * ng_loc * geom::HCA_O_LORA);
        hca_sc.gate     = dalloc<__nv_bfloat16>(size_t(Mmax) * ng_loc * geom::HCA_O_LORA);
        hca_sc.split_m  = dalloc<float>(size_t(16) * ernie55::HCA_MAX_SPLITS * nh_loc);
        hca_sc.split_l  = dalloc<float>(size_t(16) * ernie55::HCA_MAX_SPLITS * nh_loc);
        hca_sc.split_o  = dalloc<float>(size_t(16) * ernie55::HCA_MAX_SPLITS * nh_loc * geom::HCA_HD);

        dsa_sc.q_c      = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::QA);
        dsa_sc.q_raw    = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::QK_HEAD);
        dsa_sc.q_nope_h = dalloc<__nv_bfloat16>(size_t(nh_loc) * Mmax * geom::QK_NOPE);
        dsa_sc.q_abs_h  = dalloc<__nv_bfloat16>(size_t(nh_loc) * Mmax * geom::KV_LORA);
        dsa_sc.q_full   = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::KVA);
        dsa_sc.kv_raw   = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::KVA);
        dsa_sc.gate     = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::V_HEAD);
        dsa_sc.u_h      = dalloc<__nv_bfloat16>(size_t(nh_loc) * Mmax * geom::KV_LORA);
        dsa_sc.v_h      = dalloc<__nv_bfloat16>(size_t(nh_loc) * Mmax * geom::V_HEAD);
        dsa_sc.attn_out = dalloc<__nv_bfloat16>(size_t(Mmax) * nh_loc * geom::V_HEAD);
        dsa_sc.kv_b_tmp = dalloc<__nv_bfloat16>(size_t(nh_loc) * geom::KV_LORA * geom::V_HEAD);
        dsa_sc.split_m  = dalloc<float>(size_t(16) * ernie55::DSA_MAX_SPLITS * nh_loc);
        dsa_sc.split_l  = dalloc<float>(size_t(16) * ernie55::DSA_MAX_SPLITS * nh_loc);
        dsa_sc.split_o  = dalloc<float>(size_t(16) * ernie55::DSA_MAX_SPLITS * nh_loc * geom::KV_LORA);
        dsa_sc.idx_k    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::IDX_DIM);
        dsa_sc.idx_q    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::IDX_HEADS * geom::IDX_DIM);
        dsa_sc.idx_w    = dalloc<float>(size_t(Mmax) * geom::IDX_HEADS);
        dsa_sc.idx_sc   = dalloc<float>(size_t(ernie55::IDX_SCORE_BATCH) * o.cap);
        dsa_sc.sel_tok  = dalloc<int>(size_t(Mmax) * geom::IDX_TOPK);
        dsa_sc.sel_cnt  = dalloc<int>(size_t(Mmax));

        dense_sc.gate = dalloc<__nv_bfloat16>(size_t(Mmax) * dense_i_loc);
        dense_sc.up   = dalloc<__nv_bfloat16>(size_t(Mmax) * dense_i_loc);

        moe_sc.x_f32         = dalloc<float>(size_t(Mmax) * geom::HID);
        moe_sc.router_l0     = dalloc<float>(size_t(Mmax) * geom::NEXP);
        moe_sc.router_l1     = dalloc<float>(size_t(Mmax) * geom::NEXP);
        moe_sc.topk_idx      = dalloc<int>(size_t(Mmax) * geom::TOPK);
        moe_sc.topk_w        = dalloc<float>(size_t(Mmax) * geom::TOPK);
        moe_sc.routed_in     = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::EH);
        moe_sc.routed_lat    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::EH);
        moe_sc.routed_normed = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::EH);
        moe_sc.routed_up_out = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        moe_sc.shared_out    = dalloc<__nv_bfloat16>(size_t(Mmax) * geom::HID);
        moe_sc.d_n_groups    = dalloc<int>(1);
        moe_sc.grp_exp       = dalloc<int>(size_t(Mmax) * geom::TOPK + geom::NEXP);
        moe_sc.grp_cnt       = dalloc<int>(size_t(Mmax) * geom::TOPK + geom::NEXP);
        moe_sc.grp_routes    = dalloc<int>((size_t(Mmax) * geom::TOPK + geom::NEXP) * 16);
        moe_sc.x_had_r       = dalloc<__half>(size_t(Mmax) * geom::EHB * 128);
        moe_sc.x_had_s       = dalloc<__half>(size_t(Mmax) * geom::HB * 128);
        moe_sc.fc1_act_r     = dalloc<__half>(size_t(Mmax) * geom::TOPK * eb_loc * 128);
        moe_sc.fc1_act_s     = dalloc<__half>(size_t(Mmax) * eb_loc * 128);
        moe_sc.fc2_part_r    = dalloc<float>(size_t(Mmax) * geom::TOPK * geom::EH);
        moe_sc.sh_fc1_part   = dalloc<float>(size_t(16) * eb_loc * 4 * 2 * 128);
        moe_sc.sh_fc2_part   = dalloc<float>(size_t(16) * geom::HB * 2 * 128);

        head_logits     = dalloc<float>(size_t(Mmax) * max_vr);
        head_argmax_val = dalloc<float>(size_t(Mmax) * ink2::HEAD_ARGMAX_NG);
        head_argmax_id  = dalloc<int>(size_t(Mmax) * ink2::HEAD_ARGMAX_NG);
        head_dev_pkt    = dalloc<uint8_t>(size_t(Mmax) * head_row_bytes(o.nc));
        ERNIE55_CK(cudaHostAlloc(&head_host_parts, size_t(o.nr) * Mmax * head_row_bytes(o.nc), cudaHostAllocDefault));
        ERNIE55_CK(cudaHostAlloc(&h_meta_pinned, size_t(3) * Mmax * sizeof(int), cudaHostAllocDefault));
        d_tok  = dalloc<int>(size_t(Mmax));
        d_slot = dalloc<int>(size_t(Mmax));
        d_pos  = dalloc<int>(size_t(Mmax));

        h_page_table.assign(size_t(o.slots) * max_pps, 0);
        for (int s = 0; s < o.slots; ++s) {
            for (int p = 0; p < max_pps; ++p) h_page_table[size_t(s) * max_pps + p] = (s * max_pps + p) % pool_pages;
        }
        d_page_table = dalloc<int>(size_t(o.slots) * max_pps);
        ERNIE55_CK(cudaMemcpy(d_page_table, h_page_table.data(), h_page_table.size() * sizeof(int), cudaMemcpyHostToDevice));

        // 8. Stage 5 (Hybrid only): carve ALL remaining free VRAM into static + dynamic hot-expert cache!
        if (o.use_hybrid) {
            hybrid_cache.init_vram_pool(Mmax, stream);
        }

        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr, "# ernie55 ready: TP=%d rank0=%d kv=%s mx8=%d mx8_head=%d hybrid=%d graphs=%d VRAM used=%.2f / %.2f GiB (free=%.2f GiB)\n",
                     o.nr, o.rank0, kv_fmt_name(o.kv_fmt), o.mx8_dense ? 1 : 0, o.mx8_head ? 1 : 0, o.use_hybrid ? 1 : 0,
                     o.use_graphs ? 1 : 0, (total_b - free_b) / double(1ull << 30),
                     total_b / double(1ull << 30), free_b / double(1ull << 30));
    }

    void dump_tensor(const std::string& name, const void* dptr, size_t nbytes) {
        if (o.dump_dir.empty()) return;
        std::filesystem::create_directories(o.dump_dir);
        std::vector<uint8_t> buf(nbytes);
        ERNIE55_CK(cudaStreamSynchronize(stream));
        ERNIE55_CK(cudaMemcpy(buf.data(), dptr, nbytes, cudaMemcpyDeviceToHost));
        std::ofstream f(o.dump_dir + "/" + name, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(nbytes));
    }

    static int idx_cap_bucket(int max_pos, int max_cap) {
        if (max_pos + 1 - geom::SLIDING_WIN <= geom::IDX_TOPK) return 0;
        const int need = max_pos + 1 - geom::SLIDING_WIN;
        int b = 4096;
        while (b < need && b < max_cap) b <<= 1;
        return std::min(b, max_cap);
    }

    void emit_layer(int l, const ernie55::StepSeg* segs, int nsegs, int M, int max_pos,
                    int& coll, bool capture_mode, int graph_idx_cap,
                    const std::function<void(double&)>& prof_mark) {
        const Layer& y = L[size_t(l)];

        // 1. Attention mHC Pre + input_layernorm
        ernie55::mhc_pre_and_norm(blas, y.attn_hc, hc_sc, hyper, y.in_ln, mixed_ln, M, stream);
        if (!capture_mode) prof_mark(prof.mhc_ms);

        // 2. HCA C128 or Sparse MLA-DSA Attention
        const int hca_g_ns = capture_mode
            ? ((max_pos + 1 < 256) ? 1 : std::max(1, 4 / M))
            : 0;
        const int dsa_g_ns = capture_mode
            ? ((max_pos + 1 < 256) ? 1 : std::max(1, 12 / M))
            : 0;
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            RankLayer& rl = const_cast<RankLayer&>(y.rk[size_t(i)]);
            __nv_bfloat16* p_dst = parts + static_cast<size_t>(r) * M * geom::HID;
            if (y.is_dsa) {
                ernie55::dsa_forward(blas, rl.dsa_w, dsa_sc, rl.dsa_kv, mixed_ln, p_dst,
                                     segs, nsegs, d_slot, d_pos, d_page_table,
                                     M, max_pps, o.cap, max_pos, mx8_x8, stream, graph_idx_cap, dsa_g_ns,
                                     (!capture_mode && o.profile) ? &prof.dsa_d : nullptr);
            } else {
                ernie55::hca_forward(blas, rl.hca_w, hca_sc, rl.hca_kv, mixed_ln, p_dst,
                                     segs, nsegs, d_slot, d_pos, d_yarn_inv_freq,
                                     M, max_pos, mx8_x8, stream, hca_g_ns,
                                     (!capture_mode && o.profile) ? &prof.hca_d : nullptr);
            }
        }
        if (!capture_mode) prof_mark(y.is_dsa ? prof.dsa_ms : prof.hca_ms);

        // 3. Attention TP AllReduce + mHC Post Residual Combine
        if (on_ar_mhc_post) {
            on_ar_mhc_post(coll++, M);
        } else {
            ernie55::k_ernie55_sum_mhc_post<<<M * 16, 256, 0, stream>>>(
                parts, hc_sc.post, hc_sc.comb, hyper, o.nr, M, M);
        }
        if (!capture_mode) {
            prof_mark(prof.ar_ms);
            if (!o.dump_dir.empty()) {
                char fn[64]; std::snprintf(fn, sizeof(fn), "L%03d_post_attn.bf16", l);
                dump_tensor(fn, hyper, size_t(M) * geom::HC_DIM * 2);
            }
        }

        // 4. MLP / MoE mHC Pre + post_attention_layernorm
        ernie55::mhc_pre_and_norm(blas, y.mlp_hc, hc_sc, hyper, y.post_attn_ln, mixed_ln, M, stream);
        if (!capture_mode) prof_mark(prof.mhc_ms);

        // 5. L0 Dense SiTU MLP or L1..L42 Latent MoE (resident OR hybrid!)
        const bool fuse_tp1_moe = (y.is_moe && o.nr == 1 && !on_ar_mhc_post);
        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            const RankLayer& rl = y.rk[size_t(i)];
            __nv_bfloat16* p_dst = parts + static_cast<size_t>(r) * M * geom::HID;
            if (!y.is_moe) {
                ernie55::dense_mlp_forward(blas, rl.dense_w, dense_sc, mixed_ln, p_dst, M, mx8_x8, stream);
            } else if (o.use_hybrid) {
                ernie55::moe_route(blas, rl.moe_w, moe_sc, mixed_ln, M, hybrid_cache.layer_hits(y.moe_idx), stream);
                hybrid_cache.begin_routed_layer(moe_sc, M, stream);
                ernie55::mx8::linear_w(blas, mixed_ln, rl.moe_w.fc1_lat, moe_sc.routed_in, M, geom::EH, geom::HID, mx8_x8, nullptr, stream);
                ernie55::k_ernie55_prerot_x<geom::EHB><<<M * geom::EHB, 32, 0, stream>>>(moe_sc.routed_in, rl.moe_w.routed.sg, moe_sc.x_had_r, M);
                hybrid_cache.begin_routed_x(moe_sc, M, stream);
                ernie55::moe_shared(rl.moe_w, moe_sc, mixed_ln, M, stream);
                hybrid_cache.run_routed_layer(y.moe_idx, rl.moe_w, moe_sc, M, stream);
                ernie55::moe_routed_finish(blas, rl.moe_w, moe_sc, p_dst, M, mx8_x8, stream, fuse_tp1_moe);
            } else {
                const bool use_par_sh = (capture_mode || !o.profile);
                ernie55::moe_forward_resident(blas, rl.moe_w, moe_sc, mixed_ln, p_dst, M, mx8_x8, stream, fuse_tp1_moe,
                                              (!capture_mode && o.profile) ? &prof.moe_d : nullptr,
                                              use_par_sh ? stream_sh : nullptr,
                                              use_par_sh ? ev_moe_fork : nullptr,
                                              use_par_sh ? ev_moe_join : nullptr);
            }
        }
        if (!capture_mode) prof_mark(prof.moe_ms);

        // 6. MLP / MoE TP AllReduce + mHC Post Residual Combine
        if (fuse_tp1_moe) {
            ernie55::k_ernie55_add_mhc_post<<<M * 16, 256, 0, stream>>>(
                moe_sc.routed_up_out, moe_sc.shared_out, hc_sc.post, hc_sc.comb, hyper, M);
        } else if (on_ar_mhc_post) {
            on_ar_mhc_post(coll++, M);
        } else {
            ernie55::k_ernie55_sum_mhc_post<<<M * 16, 256, 0, stream>>>(
                parts, hc_sc.post, hc_sc.comb, hyper, o.nr, M, M);
        }
        if (!capture_mode) {
            prof_mark(prof.ar_ms);
            if (!o.dump_dir.empty()) {
                char fn[64]; std::snprintf(fn, sizeof(fn), "L%03d_out.bf16", l);
                dump_tensor(fn, hyper, size_t(M) * geom::HC_DIM * 2);
            }
        }
    }

    // Forward pass across all 43 layers + optional mhc_contract + final norm + lm_head + multimax + sampling.
    void forward(const std::vector<ernie55::StepSeg>& segs, const int* toks, const RowSample* sm,
                 std::vector<RowOut>& out, bool skip_head = false,
                 std::vector<float>* out_full_logits = nullptr, bool need_logprobs = false) {
        int M = 0, max_pos = 0;
        for (const auto& sg : segs) {
            M = std::max(M, sg.row0 + sg.len);
            if (sg.len > 0) max_pos = std::max(max_pos, sg.pos0 + sg.len - 1);
        }
        if (M <= 0 || M > o.max_rows) die("forward: bad M=" + std::to_string(M));
        if (max_pos >= o.cap) die("forward: pos " + std::to_string(max_pos) + " >= cap " + std::to_string(o.cap));

        // Upload token/slot/pos metadata
        int* h_tok  = h_meta_pinned;
        int* h_slot = h_meta_pinned + M;
        int* h_pos  = h_meta_pinned + 2 * M;
        for (const auto& sg : segs) {
            for (int t = 0; t < sg.len; ++t) {
                const int m = sg.row0 + t;
                h_tok[m]  = toks[m];
                h_slot[m] = sg.slot;
                h_pos[m]  = sg.pos0 + t;
            }
        }
        ERNIE55_CK(cudaMemcpyAsync(d_tok,  h_tok,  size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        ERNIE55_CK(cudaMemcpyAsync(d_slot, h_slot, size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));
        ERNIE55_CK(cudaMemcpyAsync(d_pos,  h_pos,  size_t(M) * sizeof(int), cudaMemcpyHostToDevice, stream));

        if (on_step_begin) on_step_begin(M, skip_head);

        bool all_greedy = true;
        if (sm != nullptr) {
            for (int m = 0; m < M; ++m) {
                if (sm[m].inv_temp > 0.f) { all_greedy = false; break; }
            }
        }
        const bool greedy_fast = (o.nr == 1 && o.nloc == 1 && all_greedy && !need_logprobs && !out_full_logits);

        const size_t rb = head_row_bytes(o.nc);
        const bool use_graph_now = o.use_graphs && !o.use_hybrid && o.nr == 1 && o.nloc == 1 &&
                                   !o.profile && o.dump_dir.empty() && !skip_head && !out_full_logits && M <= 12;

        if (use_graph_now) {
            const bool is_big_kv = (max_pos + 1 >= 256);
            const bool is_sparse = (max_pos + 1 - geom::SLIDING_WIN > geom::IDX_TOPK);
            const int  ib        = idx_cap_bucket(max_pos, o.cap);
            const GraphKey gk{M, is_big_kv, is_sparse, ib, greedy_fast};
            auto it = graphs.find(gk);
            if (it == graphs.end()) {
                cudaGraph_t gr = nullptr;
                ERNIE55_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                ernie55::k_ernie55_embed_expand<<<M * 16, 256, 0, stream>>>(d_tok, embed, hyper, M);
                int coll = 0;
                std::function<void(double&)> dummy = [](double&) {};
                for (int l = 0; l < geom::L_MAIN; ++l) {
                    emit_layer(l, segs.data(), static_cast<int>(segs.size()), M, max_pos, coll, true, ib, dummy);
                }
                ernie55::k_ernie55_hc_head_and_norm<<<M, 256, 0, stream>>>(
                    hyper, hc_head.fn, hc_head.base, hc_head.scale, final_norm, mixed_ln, M);
                if (o.mx8_head && lm_head_mx8[0].ready()) {
                    ernie55::mx8::linear_mx8_f32_multimax(blas, mixed_ln, lm_head_mx8[0], multimax_ranges, multimax_ts,
                                                          head_logits, M, mx8_x8, stream);
                } else {
                    ernie55::linear_bf16_f32_multimax(blas, mixed_ln, lm_head[0], multimax_ranges, multimax_ts,
                                                      head_logits, M, v_rows[0], geom::HID, stream);
                }
                if (greedy_fast) {
                    ink2::head_argmax_dev(head_logits, M, v_rows[0], v_lo[0],
                                          head_argmax_val, head_argmax_id, head_dev_pkt, rb, stream);
                } else {
                    ink2::k_ink2_head_pack<<<M, 256, 0, stream>>>(head_logits, v_rows[0], v_lo[0], o.nc, head_dev_pkt, rb);
                }
                ERNIE55_CK(cudaStreamEndCapture(stream, &gr));
                cudaGraphExec_t ex = nullptr;
                ERNIE55_CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
                ERNIE55_CK(cudaGraphDestroy(gr));
                it = graphs.emplace(gk, ex).first;
            }
            ERNIE55_CK(cudaGraphLaunch(it->second, stream));
            ERNIE55_CK(cudaMemcpyAsync(head_host_parts, head_dev_pkt, size_t(M) * rb, cudaMemcpyDeviceToHost, stream));
            ERNIE55_CK(cudaStreamSynchronize(stream));
            out.resize(size_t(M));
            for (int m = 0; m < M; ++m) {
                out[size_t(m)] = head_merge(head_host_parts + size_t(m) * rb, rb, 1, o.nc);
                out[size_t(m)].tok = sm ? sample_row(out[size_t(m)].top, sm[m])
                                        : (out[size_t(m)].top.empty() ? 0 : out[size_t(m)].top[0].id);
            }
            return;
        }

        // Eager path (prefill, hybrid 5090 offload, TP>1, profile, dump, or GR8_ERNIE55_GRAPHS=0)
        auto prof_mark = [&](double& acc) {
            if (!o.profile) return;
            ERNIE55_CK(cudaEventRecord(ev_b, stream));
            ERNIE55_CK(cudaEventSynchronize(ev_b));
            float ms = 0.f;
            ERNIE55_CK(cudaEventElapsedTime(&ms, ev_a, ev_b));
            acc += ms;
            ERNIE55_CK(cudaEventRecord(ev_a, stream));
        };
        if (o.profile) ERNIE55_CK(cudaEventRecord(ev_a, stream));

        if (o.use_hybrid) {
            hybrid_cache.embed_expand_host(h_tok, M, hyper, stream);
        } else {
            ernie55::k_ernie55_embed_expand<<<M * 16, 256, 0, stream>>>(d_tok, embed, hyper, M);
        }
        if (!o.dump_dir.empty()) dump_tensor("embed.bf16", hyper, size_t(M) * geom::HC_DIM * 2);

        int coll = 0;
        for (int l = 0; l < geom::L_MAIN; ++l) {
            emit_layer(l, segs.data(), static_cast<int>(segs.size()), M, max_pos, coll, false, 0, prof_mark);
        }

        if (skip_head) {
            if (on_step_end) on_step_end();
            ERNIE55_CK(cudaStreamSynchronize(stream));
            if (o.use_hybrid) hybrid_cache.maintain_after_step(stream);
            if (o.profile) { prof.steps += 1; prof.tokens += M; }
            return;
        }

        ernie55::k_ernie55_hc_head_and_norm<<<M, 256, 0, stream>>>(
            hyper, hc_head.fn, hc_head.base, hc_head.scale, final_norm, mixed_ln, M);
        if (!o.dump_dir.empty()) dump_tensor("final_norm.bf16", mixed_ln, size_t(M) * geom::HID * 2);

        if (out_full_logits) out_full_logits->assign(size_t(M) * geom::VOCAB, 0.0f);

        for (int i = 0; i < o.nloc; ++i) {
            const int r = o.rank0 + i;
            const int vr = v_rows[size_t(i)];
            const int v0 = v_lo[size_t(i)];
            if (o.mx8_head && lm_head_mx8[size_t(i)].ready()) {
                ernie55::mx8::linear_mx8_f32_multimax(blas, mixed_ln, lm_head_mx8[size_t(i)], multimax_ranges, multimax_ts,
                                                      head_logits, M, mx8_x8, stream);
            } else {
                ernie55::linear_bf16_f32_multimax(blas, mixed_ln, lm_head[size_t(i)], multimax_ranges, multimax_ts,
                                                  head_logits, M, vr, geom::HID, stream);
            }
            if (o.profile) {
                const double before = prof.head_gemv_ms;
                prof_mark(prof.head_gemv_ms);
                prof.head_ms += (prof.head_gemv_ms - before);
            }
            if (out_full_logits) {
                ERNIE55_CK(cudaStreamSynchronize(stream));
                std::vector<float> tmp(size_t(M) * vr);
                ERNIE55_CK(cudaMemcpy(tmp.data(), head_logits, tmp.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int m = 0; m < M; ++m) {
                    std::memcpy(out_full_logits->data() + size_t(m) * geom::VOCAB + v0,
                                tmp.data() + size_t(m) * vr, size_t(vr) * sizeof(float));
                }
            }
            if (greedy_fast) {
                ink2::head_argmax_dev(head_logits, M, vr, v0, head_argmax_val, head_argmax_id, head_dev_pkt, rb, stream);
            } else {
                ink2::k_ink2_head_pack<<<M, 256, 0, stream>>>(head_logits, vr, v0, o.nc, head_dev_pkt, rb);
            }
            ERNIE55_CK(cudaMemcpyAsync(head_host_parts + static_cast<size_t>(r) * M * rb,
                                       head_dev_pkt, size_t(M) * rb, cudaMemcpyDeviceToHost, stream));
        }
        ERNIE55_CK(cudaStreamSynchronize(stream));
        if (on_head_gather) on_head_gather(coll++, M, head_host_parts, rb);
        if (on_step_end) on_step_end();
        if (o.use_hybrid) hybrid_cache.maintain_after_step(stream);
        if (o.profile) {
            const double before = prof.head_pack_ms;
            prof_mark(prof.head_pack_ms);
            prof.head_ms += (prof.head_pack_ms - before);
        }

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
};

} // namespace ernie55rt
