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
#include "qw38_dflash.cuh"
#include "qw38_strata.cuh"

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

    // Strata-mode split placement: compact GPU VRAM + mapped pinned host RAM + temporary BF16 host staging
    bool strata_mode = false;
    uint8_t* pin_host = nullptr;
    uint8_t* pin_dev = nullptr;
    uint8_t* tmp_host = nullptr;
    uint8_t* tmp_dev = nullptr;
    enum RegionKind : uint8_t { REG_GPU = 0, REG_PIN = 1, REG_TMP = 2 };
    struct Placement {
        uint64_t pack_off = 0, nbytes = 0, dst_off = 0;
        RegionKind reg = REG_GPU;
    };
    std::vector<Placement> placements;
    std::unordered_map<uint64_t, size_t> off_to_place;

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

    struct SplitSink {
        uint8_t* dev_gpu = nullptr;
        uint8_t* pin_host = nullptr;
        uint8_t* tmp_host = nullptr;
        const std::vector<Placement>* placements = nullptr;
        std::mutex mu;
        std::vector<cudaStream_t> streams;
        static void* alloc(void*, size_t n) {
            void* p = nullptr;
            return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
        }
        static void release(void*, void* p) { cudaFreeHost(p); }
        static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n) {
            SplitSink& s = *static_cast<SplitSink*>(ctx);
            const auto& pl = *s.placements;
            // Binary search for first placement with pack_off + nbytes > off
            size_t lo_i = 0, hi_i = pl.size();
            while (lo_i < hi_i) {
                const size_t mid = lo_i + ((hi_i - lo_i) >> 1);
                if (pl[mid].pack_off + pl[mid].nbytes <= off) lo_i = mid + 1;
                else hi_i = mid;
            }
            cudaStream_t st = nullptr;
            auto get_stream = [&]() -> cudaStream_t {
                if (st) return st;
                std::lock_guard<std::mutex> lk(s.mu);
                while (static_cast<int>(s.streams.size()) <= w) {
                    cudaStream_t x;
                    if (cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
                    s.streams.push_back(x);
                }
                st = s.streams[static_cast<size_t>(w)];
                return st;
            };
            uint64_t cur = off;
            const uint64_t end = off + n;
            size_t idx = lo_i;
            while (cur < end && idx < pl.size()) {
                const auto& p = pl[idx];
                if (cur < p.pack_off) {
                    cur = std::min(end, p.pack_off);
                    if (cur >= end) break;
                }
                const uint64_t seg_end = std::min(end, p.pack_off + p.nbytes);
                if (seg_end > cur) {
                    const size_t len = static_cast<size_t>(seg_end - cur);
                    const uint8_t* sp = static_cast<const uint8_t*>(src) + (cur - off);
                    const uint64_t dst_off = p.dst_off + (cur - p.pack_off);
                    if (p.reg == REG_PIN) {
                        std::memcpy(s.pin_host + dst_off, sp, len);
                    } else if (p.reg == REG_TMP) {
                        std::memcpy(s.tmp_host + dst_off, sp, len);
                    } else {
                        cudaStream_t cs = get_stream();
                        if (!cs) return 2;
                        if (cudaMemcpyAsync(s.dev_gpu + dst_off, sp, len, cudaMemcpyHostToDevice, cs) != cudaSuccess) return 3;
                        if (cudaStreamSynchronize(cs) != cudaSuccess) return 4;
                    }
                    cur = seg_end;
                }
                if (cur >= p.pack_off + p.nbytes) ++idx;
            }
            return 0;
        }
    };

    ~Image() {
        if (dev) cudaFree(dev);
        if (pin_host) cudaFreeHost(pin_host);
        if (tmp_host) cudaFreeHost(tmp_host);
        if (plan) g6l_plan_free(plan);
    }

    void free_temp_host() {
        if (tmp_host) {
            cudaFreeHost(tmp_host);
            tmp_host = nullptr;
            tmp_dev = nullptr;
        }
    }

    static bool ends_with(const std::string& s, const char* suf) {
        const size_t sl = std::strlen(suf);
        return s.size() >= sl && s.compare(s.size() - sl, sl, suf) == 0;
    }

    static RegionKind classify_strata_tensor(const std::string& name, bool dense_fp8, bool pf_mxfp8, bool mtp_fast) {
        if (ends_with(name, ".tr3.fc1") || ends_with(name, ".tr3.fc2") || name == "embed.weight") {
            return REG_PIN;
        }
        // Move trunk BF16 HC down/up matrices (~1.27 GB) off GPU:
        // when pf_mxfp8 is active, prefill runs on VRAM-resident fp8_down (MXFP8 TC) + fp8_up (FP8->BF16 TC),
        // so BF16 originals are REG_TMP (freed right after quantization -> 0 PCIe traffic, -1.27 GB host RAM!).
        if (dense_fp8) {
            const RegionKind hc_rk = pf_mxfp8 ? REG_TMP : REG_PIN;
            if (name == "hc_final.down" || name == "hc_final.up") return hc_rk;
            const bool is_trunk_layer = (name.size() >= 5 && name[0] == 'L' && name[4] == '.' && name.compare(0, 5, "L048.") != 0);
            if (is_trunk_layer && (ends_with(name, ".attn_hc.down") || ends_with(name, ".attn_hc.up") ||
                                   ends_with(name, ".mlp_hc.down") || ends_with(name, ".mlp_hc.up"))) {
                return hc_rk;
            }
        }
        if (mtp_fast && (name == "L048.mtp_exp.gate_up" || name == "L048.mtp_exp.down")) {
            return REG_TMP;
        }
        if (dense_fp8 && pf_mxfp8) {
            if (name == "lm_head.weight") return REG_TMP;
            const bool is_trunk_layer = (name.size() >= 5 && name[0] == 'L' && name[4] == '.' && name.compare(0, 5, "L048.") != 0);
            if (is_trunk_layer) {
                if (ends_with(name, ".attn.q_proj") || ends_with(name, ".attn.k_proj") ||
                    ends_with(name, ".attn.v_proj") || ends_with(name, ".attn.o_proj") ||
                    ends_with(name, ".gdn.in_proj_qkv") || ends_with(name, ".gdn.in_proj_z") ||
                    ends_with(name, ".gdn.out_proj") || ends_with(name, ".shared.gate_proj") ||
                    ends_with(name, ".shared.up_proj") || ends_with(name, ".shared.down_proj")) {
                    return REG_TMP;
                }
            }
        }
        return REG_GPU;
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

    void whole_strata(bool dense_fp8, bool pf_mxfp8, bool mtp_fast) {
        strata_mode = true;
        auto al4096 = [](uint64_t n) -> uint64_t { return (n + 4095ull) & ~4095ull; };
        uint64_t gpu_cur = 0, pin_cur = 0, tmp_cur = 0;
        placements.clear();
        placements.reserve(pack.entries().size());
        off_to_place.clear();
        for (const auto& e : pack.entries()) {
            const RegionKind rk = classify_strata_tensor(e.name, dense_fp8, pf_mxfp8, mtp_fast);
            uint64_t dst = 0;
            if (rk == REG_GPU) { dst = gpu_cur; gpu_cur = al4096(gpu_cur + e.nbytes); }
            else if (rk == REG_PIN) { dst = pin_cur; pin_cur = al4096(pin_cur + e.nbytes); }
            else { dst = tmp_cur; tmp_cur = al4096(tmp_cur + e.nbytes); }
            off_to_place[e.offset] = placements.size();
            placements.push_back({e.offset, e.nbytes, dst, rk});
        }
        std::fprintf(stderr,
                     "# rank %d: qw38 strata split plan: GPU=%.2f GB, Pinned-UVA=%.2f GB, Temp-BF16-Host=%.2f GB (total %.2f GB)\n",
                     rank, gpu_cur / 1e9, pin_cur / 1e9, tmp_cur / 1e9, pack.bytes() / 1e9);

        void* d_gpu = nullptr;
        QW38_CK(cudaMalloc(&d_gpu, gpu_cur ? gpu_cur : 4096));
        dev = static_cast<uint8_t*>(d_gpu);
        if (pin_cur > 0) {
            void* p_h = nullptr;
            void* p_d = nullptr;
            QW38_CK(cudaHostAlloc(&p_h, pin_cur, cudaHostAllocMapped | cudaHostAllocPortable));
            QW38_CK(cudaHostGetDevicePointer(&p_d, p_h, 0));
            pin_host = static_cast<uint8_t*>(p_h);
            pin_dev = static_cast<uint8_t*>(p_d);
        }
        if (tmp_cur > 0) {
            void* t_h = nullptr;
            void* t_d = nullptr;
            QW38_CK(cudaHostAlloc(&t_h, tmp_cur, cudaHostAllocMapped | cudaHostAllocPortable));
            QW38_CK(cudaHostGetDevicePointer(&t_d, t_h, 0));
            tmp_host = static_cast<uint8_t*>(t_h);
            tmp_dev = static_cast<uint8_t*>(t_d);
        }

        SplitSink sink;
        sink.dev_gpu = dev;
        sink.pin_host = pin_host;
        sink.tmp_host = tmp_host;
        sink.placements = &placements;
        const g6l_sink s{&sink, &SplitSink::alloc, &SplitSink::release, &SplitSink::copy};
        g6l_stats st{};
        char err[1024] = "";
        if (g6l_plan_load(plan, &s, &st, err, sizeof err))
            die("rank " + std::to_string(rank) + ": strata split load: " + err);
        for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
        std::fprintf(stderr, "# rank %d: qw38 strata split loaded %.2f GB in %.2f s (%u threads)\n",
                     rank, st.image_bytes / 1e9, st.seconds, st.threads);
    }

    const g6::Entry* find(const std::string& name) const { return pack.find(name); }
    const g6::Entry& entry(const std::string& name) const {
        const g6::Entry* e = pack.find(name);
        if (!e) die("rank " + std::to_string(rank) + ": missing tensor " + name);
        return *e;
    }
    const uint8_t* dev_addr(const g6::Entry& e) const {
        if (!strata_mode) return dev + e.offset;
        const auto& p = placements[off_to_place.at(e.offset)];
        if (p.reg == REG_GPU) return dev + p.dst_off;
        if (p.reg == REG_PIN) return pin_dev + p.dst_off;
        return tmp_dev + p.dst_off;
    }
    uint8_t* host_addr_mut(const g6::Entry& e) const {
        if (!strata_mode) return nullptr;
        const auto& p = placements[off_to_place.at(e.offset)];
        if (p.reg == REG_PIN) return pin_host + p.dst_off;
        if (p.reg == REG_TMP) return tmp_host + p.dst_off;
        return nullptr;
    }
    template <class T> const T* get(const std::string& name) const {
        return reinterpret_cast<const T*>(dev_addr(entry(name)));
    }
    template <class T> const T* opt(const std::string& name) const {
        const g6::Entry* e = pack.find(name);
        return e ? reinterpret_cast<const T*>(dev_addr(*e)) : nullptr;
    }
    template <class T> T* host_mut(const std::string& name) const {
        return reinterpret_cast<T*>(host_addr_mut(entry(name)));
    }
    template <class T> void read_host(const std::string& name, std::vector<T>& out) const {
        const g6::Entry& e = entry(name);
        out.resize(e.nbytes / sizeof(T));
        if (uint8_t* hp = host_addr_mut(e)) {
            std::memcpy(out.data(), hp, e.nbytes);
        } else {
            QW38_CK(cudaMemcpy(out.data(), dev_addr(e), e.nbytes, cudaMemcpyDeviceToHost));
        }
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
    bool mtp_fast = true;
    bool trunk_head2 = true;
    bool pf_mxfp8 = true;
    bool ple_async = true;
    bool use_graph = true;
    bool profile = false;
    // QSpine-DF2 parallel block drafter + inline tap capture
    std::string dfl_pack_path;
    int dfl_k = 0;
    int dfl_dp = 2;
    bool no_mtp = false;
    std::string tap_dump_path;
    StrataConfig strata;
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

    // QSpine-DF2 parallel block drafter + inline tap capture state
    bool has_dfl = false;
    qw38::dfl::Drafter dfl;
    __half* dfl_step_taps = nullptr;
    int dfl_step_taps_cap = 0;
    int dfl_pf_lo = 0, dfl_pf_hi = 0;
    std::vector<uint8_t> dfl_cand_host;
    FILE* tap_dump_fp = nullptr;
    std::vector<uint16_t> h_tap_dump;
    int64_t tap_dump_cnt = 0, tap_dump_align_ok = 0;
    double dfl_draft_ms = 0.0, dfl_verify_ms = 0.0;
    int max_snap_k = 0;

    // Weight paging pool (GR8_PTAB=1|2) and on-load FP8 dense arena (GR8_QW38_DENSE_FP8=1)
    ptab::Pool ptab_pool;
    qw38::Fp8Arena fp8_arena;
    StrataExpertCache strata_cache;

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
    uint8_t* x8_scratch = nullptr;        // shared MXFP8 A-record scratch (GR8_QW38_PF_MXFP8=1)
    uint8_t* stage_slab = nullptr;        // Strata-mode aliased scratch slab for mutually exclusive gdn/attn/moe/ple/head
    size_t stage_slab_bytes = 0;
    size_t mtp_sc_cap = 0;
    float* head_logits = nullptr;         // [max_rows, max_v_rows] FP32
    uint8_t* head_dev_pkt = nullptr;      // [max_rows * head_row_bytes(nc)]
    uint8_t* head_host_parts = nullptr;   // [nr * max_rows * head_row_bytes(nc)] pinned host staging (graph-safe D2H!)
    int* h_meta_pinned[2] = {nullptr, nullptr};   // double-buffered [3 * max_rows] pinned staging for tok/slot/pos
    int3* h_rope_pinned[2] = {nullptr, nullptr};  // double-buffered [max_rows] pinned staging for 3D MRoPE (t, h, w)
    cudaEvent_t meta_h2d_done[2] = {nullptr, nullptr};
    bool meta_h2d_pending[2] = {false, false};
    int meta_buf = 0;
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
        // Engine::stream is cudaStreamNonBlocking, so drain prior async slot resets on stream
        // before the synchronous H2D updates below.
        QW38_CK(cudaStreamSynchronize(stream));
        for (auto& y : L) {
            if (!y.is_full) continue;
            for (int i = 0; i < o.nloc; ++i)
                QW38_CK(cudaMemcpy(y.rk[size_t(i)].kv.idx_raw_thw + slot, &thw, sizeof(int3), cudaMemcpyHostToDevice));
        }
        if (has_mtp) {
            for (int i = 0; i < o.nloc; ++i)
                QW38_CK(cudaMemcpy(mtp_w.kv[size_t(i)].idx_raw_thw + slot, &thw, sizeof(int3), cudaMemcpyHostToDevice));
        }
    }

    // Optional Layer-48 MTP drafter state (loaded when o.mtp > 0)
    bool has_mtp = false;
    qw38::MtpWeights mtp_w;
    qw38::MtpScratch mtp_sc;
    qw38::Int4Arena mtp_i4_arena;
    std::vector<qw38::MtpHead2> mtp_head2;
    qw38::MtpHead2Scratch mtp_h2_sc;
    qw38::HeadArgmax* d_mtp_argmax = nullptr;    // [max_rows]
    float* d_mtp_lp = nullptr;                   // [max_rows] top-64 softmax log-prob per step
    int* d_mtp_draft_out = nullptr;              // [max_rows * 8]
    int* h_mtp_draft_out = nullptr;              // [max_rows * 8] pinned host staging
    float* d_mtp_draft_lp = nullptr;             // [max_rows * 8]
    float* h_mtp_draft_lp = nullptr;             // [max_rows * 8] pinned host staging
    __nv_bfloat16* last_hyper = nullptr;         // [slots, 10240] BF16 (trunk final hyper state per slot)
    __nv_bfloat16* mtp_hid_in = nullptr;         // [slots, 10240] BF16 (gathered initial hyper for arbitrary S2 slot subsets)
    __nv_bfloat16* mtp_hid_cur = nullptr;        // [slots, 10240] BF16 (chained MTP draft hidden state)

    // Fabric callbacks (set by g6m4_qw38; nullptr on m4sim_qw38 where all ranks are already in `parts`)
    std::function<void(int M, bool skip_head, int n_layers_override, int head_M)> step_begin;
    std::function<void(int S, int K)> step_begin_mtp_k;
    std::function<void(int c, const qw38::HeadArgmax* local_arg, int* d_draft_out,
                       int* d_tok, int* d_pos, int3* d_rope, int S, int K, int k)> mtp_argmax_ar_chain;
    std::function<void(int c, __nv_bfloat16* parts, size_t bytes)> gather_dev;
    std::function<void(int c, __nv_bfloat16* hyper, const __nv_bfloat16* parts, const float* inj, int M)> gather_inject;
    std::function<void(int c, uint8_t* host_parts, size_t bytes)> gather_host;
    std::function<void(uint8_t* host_parts, size_t bytes)> gather_raw;
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
        int M, need_qsa, ns_attn, par, snap_k, V;
        bool operator<(const GraphKey& b) const {
            if (M != b.M) return M < b.M;
            if (need_qsa != b.need_qsa) return need_qsa < b.need_qsa;
            if (ns_attn != b.ns_attn) return ns_attn < b.ns_attn;
            if (par != b.par) return par < b.par;
            if (snap_k != b.snap_k) return snap_k < b.snap_k;
            return V < b.V;
        }
    };
    struct GraphPair {
        cudaGraphExec_t exec_a = nullptr; // embed + Layer 0 (overlapped with PLE NVMe fetch)
        cudaGraphExec_t exec_b = nullptr; // L1 PLE + Layers 1..47 + Final HC + lm_head + head_pack + D2H
    };
    std::map<GraphKey, GraphPair> graphs;

    struct MtpGraphKey {
        int M, need_qsa, ns_attn, par, hin_kind, argmax_only; // hin_kind: 0 = last_hyper, 1 = mtp_hid_cur
        bool operator<(const MtpGraphKey& b) const {
            if (M != b.M) return M < b.M;
            if (need_qsa != b.need_qsa) return need_qsa < b.need_qsa;
            if (ns_attn != b.ns_attn) return ns_attn < b.ns_attn;
            if (par != b.par) return par < b.par;
            if (hin_kind != b.hin_kind) return hin_kind < b.hin_kind;
            return argmax_only < b.argmax_only;
        }
    };
    std::map<MtpGraphKey, cudaGraphExec_t> mtp_graphs;

    struct MtpKGraphKey {
        int S, K, need_qsa, ns_attn, par, mode; // mode: 0=full, 1=lazy_head(defer idx restore), 2=lazy_tail(from mtp_hid_cur)
        bool operator<(const MtpKGraphKey& b) const {
            if (S != b.S) return S < b.S;
            if (K != b.K) return K < b.K;
            if (need_qsa != b.need_qsa) return need_qsa < b.need_qsa;
            if (ns_attn != b.ns_attn) return ns_attn < b.ns_attn;
            if (par != b.par) return par < b.par;
            return mode < b.mode;
        }
    };
    std::map<MtpKGraphKey, cudaGraphExec_t> mtp_k_graphs;

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
        for (auto& kv : mtp_k_graphs) {
            if (kv.second) cudaGraphExecDestroy(kv.second);
        }
        mtp_k_graphs.clear();
        if (h_mtp_draft_out) cudaFreeHost(h_mtp_draft_out);
        if (h_mtp_draft_lp) cudaFreeHost(h_mtp_draft_lp);
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
        if (tap_dump_fp) {
            std::fclose(tap_dump_fp);
            tap_dump_fp = nullptr;
            std::fprintf(stderr, "# qw38 tap_dump: wrote %lld rows (greedy top-1 alignment self-test: %lld/%lld)\n",
                         (long long)tap_dump_cnt, (long long)tap_dump_align_ok, (long long)tap_dump_cnt);
        }
        if (dfl_step_taps) cudaFree(dfl_step_taps);
        if (stage_slab) cudaFree(stage_slab);
        if (head_host_parts) cudaFreeHost(head_host_parts);
        for (int b = 0; b < 2; ++b) {
            if (meta_h2d_done[b]) cudaEventDestroy(meta_h2d_done[b]);
            if (h_meta_pinned[b]) cudaFreeHost(h_meta_pinned[b]);
            if (h_rope_pinned[b]) cudaFreeHost(h_rope_pinned[b]);
        }
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
        o.mtp_fast = knob::qw38_mtp_fast();
        o.trunk_head2 = knob::qw38_trunk_head2();
        o.pf_mxfp8 = knob::qw38_pf_mxfp8();
        o.ple_async = knob::qw38_ple_async();
        o.use_graph = knob::qw38_graph();
        o.profile = knob::qw38_profile();
        o.no_mtp = knob::qw38_no_mtp();
        o.dfl_dp = static_cast<int>(knob::qw38_dfl_dp());
        if (o.dfl_pack_path.empty() && knob::qw38_dfl() && knob::qw38_dfl_pack() && !knob::qw38_dfl_pack()->empty()) {
            o.dfl_pack_path = *knob::qw38_dfl_pack();
            o.dfl_k = std::max(1, std::min(7, static_cast<int>(knob::qw38_dfl_k())));
        }
        if (o.tap_dump_path.empty() && knob::qw38_tap_dump() && !knob::qw38_tap_dump()->empty()) {
            o.tap_dump_path = *knob::qw38_tap_dump();
        }
        max_snap_k = std::max(o.mtp, o.dfl_k);
        const bool want_mtp = (o.mtp > 0 && !o.no_mtp);
        if (!o.strata.enabled) {
            const StrataConfig env_s = load_strata_config_from_env();
            if (env_s.enabled) o.strata = env_s;
        }
        if (o.strata.enabled) {
            if (o.nloc != 1 || o.nr != 1) die("qw38 strata mode currently supports TP=1 (single GPU)");
            o.dense_fp8 = true;
            o.pf_mxfp8 = true;
            o.mtp_fast = true;
            o.wide = true;
            o.ptab_mode = ptab::OFF;
        }
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
            img[size_t(i)]->open(release_dir, o.widths_path, o.nr, r, o.layout_path, want_mtp);
            if (o.strata.enabled) img[size_t(i)]->whole_strata(o.dense_fp8, o.pf_mxfp8, o.mtp_fast);
            else img[size_t(i)]->whole();
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
                    if (max_snap_k > 0) {
                        rl.kv.idx_raw_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_snap_k + 1) * 4 * geom::IDX_DIM);
                        rl.kv.idx_raw_thw_snap = dalloc<int3>(size_t(o.slots) * size_t(max_snap_k + 1));
                        rl.kv.snap_stride = max_snap_k;
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
                    if (max_snap_k > 0) {
                        const int v_stride = max_snap_k + 1;
                        rl.gdn_st.v_stride = v_stride;
                        rl.gdn_st.snap_stride = max_snap_k;
                        rl.gdn_st.v_qkv = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(v_stride) * size_t(qkv_loc));
                        rl.gdn_st.v_a = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(v_stride) * size_t(nh_v));
                        rl.gdn_st.v_b = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(v_stride) * size_t(nh_v));
                        rl.gdn_st.v_pos0 = dalloc<int>(size_t(o.slots));
                        rl.gdn_st.conv_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_snap_k + 1) * geom::GDN_NHIST * size_t(qkv_loc));
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

                if (o.wide && !o.strata.enabled) {
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
                } else if (o.strata.enabled) {
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

        // 3a. Bind optional Layer-48 MTP drafter weights & KV cache (when --mtp > 0 && !GR8_QW38_NO_MTP)
        if (want_mtp && im0.find("MTP.fc_embedding") != nullptr) {
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
            mtp_w.i4_exp_gu.resize(size_t(o.nloc));
            mtp_w.i4_exp_dn.resize(size_t(o.nloc));
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
            mtp_sc_cap = o.strata.enabled ? std::min<size_t>(ms, std::max<size_t>(1024, size_t(o.slots))) : ms;
            const size_t mtp_moe_rows = o.strata.enabled ? std::max<size_t>(size_t(o.slots), 16) : ms;
            last_hyper = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HC_D);
            mtp_hid_in = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HC_D);
            mtp_hid_cur = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::HC_D);
            mtp_sc.emb_raw = dalloc<__nv_bfloat16>(mtp_sc_cap * geom::HID);
            mtp_sc.emb_norm = dalloc<__nv_bfloat16>(mtp_sc_cap * geom::HID);
            mtp_sc.emb_proj = dalloc<__nv_bfloat16>(mtp_sc_cap * geom::HID);
            mtp_sc.hid_norm = dalloc<__nv_bfloat16>(mtp_sc_cap * geom::HC_D);
            mtp_sc.hid_proj = dalloc<__nv_bfloat16>(mtp_sc_cap * geom::HC_D);
            mtp_sc.mtp_hyper = dalloc<__nv_bfloat16>(mtp_sc_cap * geom::HC_D);
            mtp_sc.exp_mid = dalloc<__nv_bfloat16>(mtp_moe_rows * geom::TOPK * geom::SHARED_I);
            mtp_sc.exp_dn_part = dalloc<float>(mtp_moe_rows * geom::TOPK * geom::HID);
            d_mtp_argmax = dalloc<qw38::HeadArgmax>(mtp_moe_rows);
            d_mtp_lp = dalloc<float>(mtp_moe_rows);
            d_mtp_draft_out = dalloc<int>(mtp_moe_rows * 8);
            d_mtp_draft_lp = dalloc<float>(mtp_moe_rows * 8);
            QW38_CK(cudaHostAlloc(&h_mtp_draft_out, mtp_moe_rows * 8 * sizeof(int), cudaHostAllocDefault));
            QW38_CK(cudaHostAlloc(&h_mtp_draft_lp, mtp_moe_rows * 8 * sizeof(float), cudaHostAllocDefault));
            std::fprintf(stderr, "# rank %d: qw38 MTP L048 drafter loaded (K=%d)\n", o.rank0, o.mtp);
        }

        // 3b. On-load block-32 FP8 E4M3 quantization of dense decode projections (GR8_QW38_DENSE_FP8=1)
        if (o.dense_fp8) {
            fp8_lm_head.resize(size_t(o.nloc));
            for (int i = 0; i < o.nloc; ++i) {
                fp8_arena.add(lm_head[size_t(i)], fp8_lm_head[size_t(i)], v_rows[size_t(i)], geom::HID, false, o.strata.enabled);
            }
            fp8_arena.add(final_hc.down, final_hc.fp8_down, geom::HC_RANK, geom::HC_D, false, o.strata.enabled);
            fp8_arena.add(final_hc.up, final_hc.fp8_up, geom::HC_D, geom::HC_RANK, true, o.strata.enabled);
            for (auto& y : L) {
                fp8_arena.add(y.attn_hc.down, y.attn_hc.fp8_down, geom::HC_RANK, geom::HC_D, false, o.strata.enabled);
                fp8_arena.add(y.attn_hc.up, y.attn_hc.fp8_up, geom::HC_D, geom::HC_RANK, true, o.strata.enabled);
                fp8_arena.add(y.mlp_hc.down, y.mlp_hc.fp8_down, geom::HC_RANK, geom::HC_D, false, o.strata.enabled);
                fp8_arena.add(y.mlp_hc.up, y.mlp_hc.fp8_up, geom::HC_D, geom::HC_RANK, true, o.strata.enabled);
                for (int i = 0; i < o.nloc; ++i) {
                    const int r = o.rank0 + i;
                    auto& rl = y.rk[size_t(i)];
                    if (y.is_full) {
                        const int nh_q = geom::nh_q(o.nr);
                        const int nh_kv = geom::nh_kv(o.nr, r);
                        fp8_arena.add(rl.attn_w.q_proj, rl.attn_w.fp8_q, nh_q * geom::HEAD_DIM * 2, geom::HID, false, true);
                        fp8_arena.add(rl.attn_w.k_proj, rl.attn_w.fp8_k, nh_kv * geom::HEAD_DIM, geom::HID, false, true);
                        fp8_arena.add(rl.attn_w.v_proj, rl.attn_w.fp8_v, nh_kv * geom::HEAD_DIM, geom::HID, false, true);
                        fp8_arena.add(rl.attn_w.idx_qk_proj, rl.attn_w.fp8_idx, geom::IDX_PROJ_OUT, geom::HID, false, false);
                        fp8_arena.add(rl.attn_w.o_proj, rl.attn_w.fp8_o, geom::HID, nh_q * geom::HEAD_DIM, false, true);
                    } else {
                        const int nh_v = geom::gdn_vh(o.nr);
                        const int qkv_loc = geom::gdn_qkv_local(o.nr, r);
                        fp8_arena.add(rl.gdn_w.in_proj_qkv, rl.gdn_w.fp8_qkv, qkv_loc, geom::HID, false, true);
                        fp8_arena.add(rl.gdn_w.in_proj_z, rl.gdn_w.fp8_z, nh_v * geom::GDN_DV, geom::HID, false, true);
                        fp8_arena.add(rl.gdn_w.out_proj, rl.gdn_w.fp8_out, geom::HID, nh_v * geom::GDN_DV, false, true);
                    }
                    fp8_arena.add(rl.moe_w.sh_gate, rl.moe_w.fp8_sh_gate, rl.moe_w.sh_i, geom::HID, false, true);
                    fp8_arena.add(rl.moe_w.sh_up, rl.moe_w.fp8_sh_up, rl.moe_w.sh_i, geom::HID, false, true);
                    fp8_arena.add(rl.moe_w.sh_down, rl.moe_w.fp8_sh_down, geom::HID, rl.moe_w.sh_i, false, true);
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
                    if (o.mtp_fast) {
                        // Quantize all 512 MTP BF16 routed experts to block-32 INT4 (2x faster & -1.25 GB VRAM on TP=1!)
                        mtp_i4_arena.add(mw.mtp_gate_up, mtp_w.i4_exp_gu[size_t(i)], geom::NEXP * 2 * mw.sh_i, geom::HID);
                        mtp_i4_arena.add(mw.mtp_down, mtp_w.i4_exp_dn[size_t(i)], geom::NEXP * geom::HID, mw.sh_i);
                    } else {
                        // Fallback: block-32 FP8 routed experts
                        fp8_arena.add(mw.mtp_gate_up, mtp_w.fp8_exp_gu[size_t(i)], geom::NEXP * 2 * mw.sh_i, geom::HID);
                        fp8_arena.add(mw.mtp_down, mtp_w.fp8_exp_dn[size_t(i)], geom::NEXP * geom::HID, mw.sh_i);
                    }
                }
            }
            const auto fp8_t0 = std::chrono::steady_clock::now();
            const size_t fp8_b = fp8_arena.build(stream, o.pf_mxfp8);
            if (o.pf_mxfp8) qw38::qw38_mxfp8_init_attrs();
            size_t i4_b = 0, h2_b = 0;
            const bool need_h2 = o.trunk_head2 || (has_mtp && o.mtp_fast) || (!o.dfl_pack_path.empty() && o.dfl_k > 0);
            if (has_mtp && o.mtp_fast) {
                i4_b = mtp_i4_arena.build(stream);
            }
            if (need_h2) {
                mtp_head2.resize(size_t(o.nloc));
                for (int i = 0; i < o.nloc; ++i) {
                    h2_b += qw38::build_mtp_head2(lm_head[size_t(i)], v_rows[size_t(i)], v_lo[size_t(i)],
                                                  mtp_head2[size_t(i)], stream);
                }
                const size_t h2_rows = std::max<size_t>(size_t(o.slots), 8);
                mtp_h2_sc.cand = dalloc<int>(h2_rows * qw38::MtpHead2::SL_TRUNK);
                mtp_h2_sc.cl = dalloc<float>(h2_rows * qw38::MtpHead2::SL_TRUNK);
                mtp_h2_sc.tail_dsum = dalloc<double>(h2_rows);
                mtp_h2_sc.screen_max = dalloc<float>(h2_rows);
            }
            QW38_CK(cudaStreamSynchronize(stream));
            if (o.strata.enabled) {
                for (int i = 0; i < o.nloc; ++i) {
                    img[size_t(i)]->free_temp_host();
                    lm_head[size_t(i)] = nullptr;
                }
                if (o.pf_mxfp8) {
                    final_hc.down = nullptr;
                    final_hc.up = nullptr;
                    for (auto& y : L) {
                        y.attn_hc.down = nullptr;
                        y.attn_hc.up = nullptr;
                        y.mlp_hc.down = nullptr;
                        y.mlp_hc.up = nullptr;
                    }
                }
            }
            const double fp8_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fp8_t0).count();
            if (has_mtp && o.mtp_fast) {
                std::fprintf(stderr,
                             "# rank %d: GR8_QW38_DENSE_FP8=1 + MTP_FAST=1: dense FP8 %.2f GB + MTP INT4 %.2f GB + head2 %.2f GB in %.2f s\n",
                             o.rank0, fp8_b / 1e9, i4_b / 1e9, h2_b / 1e9, fp8_s);
            } else if (need_h2) {
                std::fprintf(stderr, "# rank %d: GR8_QW38_DENSE_FP8=1 + TRUNK_HEAD2=1: dense FP8 %.2f GB + head2 %.2f GB in %.2f s\n",
                             o.rank0, fp8_b / 1e9, h2_b / 1e9, fp8_s);
            } else {
                std::fprintf(stderr, "# rank %d: GR8_QW38_DENSE_FP8=1: quantized dense decode weights (%.2f GB in %.2f s)\n",
                             o.rank0, fp8_b / 1e9, fp8_s);
            }
        }

        // 3c. Load QSpine-DF2 parallel block drafter (GR8_QW38_DFL_PACK) and/or inline tap capture (GR8_QW38_TAP_DUMP)
        if (!o.dfl_pack_path.empty() && o.dfl_k > 0) {
            std::string derr;
            if (!dfl.load_df2p(o.dfl_pack_path, true, derr)) die("QSpine-DF2 load failed: " + derr);
            has_dfl = true;
            dfl_cand_host.assign(size_t(o.nr) * qw38::dfl::K_MAX * qw38::dfl::TOPK * sizeof(qw38::Cand16), 0);
            std::fprintf(stderr, "# rank %d: QSpine-DF2 loaded %s (K=%d, dp=%d, head=%s, blob=%.1f MB, slot1_r=%d)\n",
                         o.rank0, o.dfl_pack_path.c_str(), o.dfl_k, o.dfl_dp,
                         (!mtp_head2.empty() && mtp_head2[0].ready()) ? "pr599-head2" : "fp8",
                         dfl.blob_bytes / 1e6, dfl.slot1_rank);
        }
        if (o.rank0 == 0 && !o.tap_dump_path.empty()) {
            tap_dump_fp = std::fopen(o.tap_dump_path.c_str(), "wb");
            if (!tap_dump_fp) die("cannot open GR8_QW38_TAP_DUMP: " + o.tap_dump_path);
            std::fprintf(stderr, "# rank 0: GR8_QW38_TAP_DUMP enabled -> %s\n", o.tap_dump_path.c_str());
        }
        if (has_dfl || tap_dump_fp != nullptr) {
            dfl_step_taps_cap = std::max(o.max_rows, 32);
            dfl_step_taps = dalloc<__half>(size_t(dfl_step_taps_cap) * qw38::dfl::NT * geom::HID);
            if (tap_dump_fp) h_tap_dump.resize(size_t(dfl_step_taps_cap) * qw38::dfl::NT * geom::HID);
        }

        // 3d. Initialize Layer-1 PLE store AFTER freeing temp BF16 host staging (-6.2 GiB peak host RAM in Strata mode)
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
            } else if (ple_store.is_gguf_iq4nl()) {
                std::fprintf(stderr, "# rank %d: qw38 PLE opened %s (GGUF IQ4_NL RAM-mmap, %llu rows, 90 B/row = %.2f GiB)\n",
                             o.rank0, ple_bin.c_str(), (unsigned long long)ple_store.total_rows(),
                             (double(ple_store.total_rows()) * 90.0) / 1073741824.0);
            } else {
                std::fprintf(stderr, "# rank %d: qw38 PLE opened %s (%llu rows, direct=%d, uring=%d, cache=%zu MiB)\n",
                             o.rank0, ple_bin.c_str(), (unsigned long long)ple_store.total_rows(),
                             int(ple_store.is_direct()), int(ple_store.has_uring()), o.ple_cache_mib);
            }

            ple_st.conv_hist = dalloc<__nv_bfloat16>(size_t(o.slots) * geom::PLE_NHIST * geom::HC_D);
            if (max_snap_k > 0) {
                ple_st.conv_hist_snap = dalloc<__nv_bfloat16>(size_t(o.slots) * size_t(max_snap_k + 1) * geom::PLE_NHIST * geom::HC_D);
                ple_st.snap_stride = max_snap_k;
            }
            if (!o.strata.enabled) {
                ple_sc.emb = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::PLE_DIM);
                ple_sc.key = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
                ple_sc.query = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
                ple_sc.val = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HID);
                ple_sc.gv = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
                ple_sc.gv_norm = dalloc<__nv_bfloat16>(size_t(o.max_rows) * geom::HC_D);
            }
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
        const size_t max_wy_chunks = (M + qw38::GDN_WY_CH - 1) / qw38::GDN_WY_CH;
        const int max_nh_q = geom::nh_q(o.nr);
        const int max_nh_kv = geom::nh_kv_max(o.nr);
        const size_t fc2_f32_n = o.strata.enabled
            ? std::max<size_t>(32 * geom::TOPK * geom::HID, (M * geom::TOPK * geom::HID + 1) / 2)
            : (M * geom::TOPK * geom::HID);
        constexpr size_t HEAD_CHUNK = 256;
        const size_t HM = std::min<size_t>(M, HEAD_CHUNK);

        if (o.strata.enabled) {
            // Strata-mode: alias mutually exclusive stage scratch (gdn / attn / moe / ple / head_logits)
            // into one shared VRAM slab (`max` instead of `sum`), reclaiming ~2.4 GB at M=8192 for ~1,360 extra experts!
            auto al256 = [](size_t n) -> size_t { return (n + 255ull) & ~255ull; };
            auto need_b = [&](size_t& cur, size_t elems, size_t elem_sz) {
                cur = al256(cur + std::max<size_t>(1, elems) * elem_sz);
            };
            size_t b_gdn = 0, b_attn = 0, b_moe = 0, b_ple = 0, b_head = 0;
            need_b(b_gdn, M * size_t(max_qkv), sizeof(__nv_bfloat16));
            need_b(b_gdn, M * size_t(max_qkv), sizeof(__nv_bfloat16));
            need_b(b_gdn, M * size_t(max_nh_v) * geom::GDN_DV, sizeof(__nv_bfloat16));
            need_b(b_gdn, M * size_t(max_nh_v), sizeof(__nv_bfloat16));
            need_b(b_gdn, M * size_t(max_nh_v), sizeof(__nv_bfloat16));
            need_b(b_gdn, M * size_t(max_nh_v) * geom::GDN_DV, sizeof(__nv_bfloat16));
            need_b(b_gdn, size_t(max_nh_v) * M * geom::GDN_DK, sizeof(uint16_t));
            need_b(b_gdn, size_t(max_nh_v) * M * geom::GDN_DK, sizeof(uint16_t));
            need_b(b_gdn, size_t(max_nh_v) * M * geom::GDN_DK, sizeof(uint16_t));
            need_b(b_gdn, size_t(max_nh_v) * M * geom::GDN_DV, sizeof(uint16_t));
            need_b(b_gdn, size_t(max_nh_v) * M * qw38::GDN_WY_CH, sizeof(uint16_t));
            need_b(b_gdn, size_t(max_nh_v) * max_wy_chunks, sizeof(float));

            need_b(b_attn, M * size_t(max_nh_q) * 512, sizeof(__nv_bfloat16));
            need_b(b_attn, M * size_t(max_nh_q) * 256, sizeof(__nv_bfloat16));
            need_b(b_attn, M * size_t(max_nh_q) * 256, sizeof(__nv_bfloat16));
            need_b(b_attn, M * size_t(max_nh_kv) * 256, sizeof(__nv_bfloat16));
            need_b(b_attn, M * size_t(max_nh_kv) * 256, sizeof(__nv_bfloat16));
            need_b(b_attn, M * geom::IDX_PROJ_OUT, sizeof(__nv_bfloat16));
            need_b(b_attn, M * geom::IDX_HEADS * geom::IDX_DIM, sizeof(__nv_bfloat16));
            need_b(b_attn, M * size_t(max_nh_q) * 256, sizeof(__nv_bfloat16));
            need_b(b_attn, M * size_t(max_blocks), sizeof(float));
            need_b(b_attn, M * geom::IDX_TOP_BLOCKS, sizeof(int));
            need_b(b_attn, M, sizeof(int));
            need_b(b_attn, 8 * size_t(max_nh_q) * 24, sizeof(float));
            need_b(b_attn, 8 * size_t(max_nh_q) * 24, sizeof(float));
            need_b(b_attn, 8 * size_t(max_nh_q) * 24 * 256, sizeof(float));

            need_b(b_moe, M * geom::NEXP, sizeof(float));
            need_b(b_moe, M * geom::TOPK, sizeof(int));
            need_b(b_moe, M * geom::TOPK, sizeof(float));
            need_b(b_moe, 1, sizeof(int));
            need_b(b_moe, M * geom::TOPK, sizeof(int));
            need_b(b_moe, M * geom::TOPK, sizeof(int));
            need_b(b_moe, M * geom::TOPK, sizeof(int));
            need_b(b_moe, M * geom::TOPK * 8, sizeof(int));
            need_b(b_moe, M * geom::HID, sizeof(__half));
            need_b(b_moe, 8 * geom::TOPK * size_t(g6nr::EB) * 2 * 2 * 128, sizeof(float));
            need_b(b_moe, M * geom::TOPK * size_t(g6nr::EB) * 128, sizeof(__half));
            need_b(b_moe, fc2_f32_n, sizeof(float));
            need_b(b_moe, M * geom::HID, sizeof(float));
            need_b(b_moe, M * geom::SHARED_I, sizeof(__nv_bfloat16));
            need_b(b_moe, M * geom::SHARED_I, sizeof(__nv_bfloat16));
            need_b(b_moe, M * geom::HID, sizeof(__nv_bfloat16));
            need_b(b_moe, M, sizeof(float));

            if (has_ple) {
                need_b(b_ple, M * geom::PLE_DIM, sizeof(__nv_bfloat16));
                need_b(b_ple, M * geom::HC_D, sizeof(__nv_bfloat16));
                need_b(b_ple, M * geom::HC_D, sizeof(__nv_bfloat16));
                need_b(b_ple, M * geom::HID, sizeof(__nv_bfloat16));
                need_b(b_ple, M * geom::HC_D, sizeof(__nv_bfloat16));
                need_b(b_ple, M * geom::HC_D, sizeof(__nv_bfloat16));
            }
            need_b(b_head, HM * size_t(max_vr), sizeof(float));

            stage_slab_bytes = std::max({b_gdn, b_attn, b_moe, b_ple, b_head, size_t(4096)});
            stage_slab = dalloc<uint8_t>(stage_slab_bytes);
            std::fprintf(stderr,
                         "# rank %d: qw38 strata aliased stage_slab %.1f MiB (saved %.1f MiB: gdn=%.1f attn=%.1f moe=%.1f ple=%.1f head=%.1f)\n",
                         o.rank0, stage_slab_bytes / 1048576.0,
                         (b_gdn + b_attn + b_moe + b_ple + b_head - stage_slab_bytes) / 1048576.0,
                         b_gdn / 1048576.0, b_attn / 1048576.0, b_moe / 1048576.0, b_ple / 1048576.0, b_head / 1048576.0);

            auto take = [&]<class T>(size_t& cur, size_t elems) -> T* {
                T* p = reinterpret_cast<T*>(stage_slab + cur);
                cur = al256(cur + std::max<size_t>(1, elems) * sizeof(T));
                return p;
            };

            size_t cur = 0;
            gdn_sc.qkv_raw = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_qkv));
            gdn_sc.qkv_conv = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_qkv));
            gdn_sc.z = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_v) * geom::GDN_DV);
            gdn_sc.a = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_v));
            gdn_sc.b = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_v));
            gdn_sc.core = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_v) * geom::GDN_DV);
            gdn_sc.wy_qe = take.template operator()<uint16_t>(cur, size_t(max_nh_v) * M * geom::GDN_DK);
            gdn_sc.wy_kd = take.template operator()<uint16_t>(cur, size_t(max_nh_v) * M * geom::GDN_DK);
            gdn_sc.wy_w = take.template operator()<uint16_t>(cur, size_t(max_nh_v) * M * geom::GDN_DK);
            gdn_sc.wy_u = take.template operator()<uint16_t>(cur, size_t(max_nh_v) * M * geom::GDN_DV);
            gdn_sc.wy_aq = take.template operator()<uint16_t>(cur, size_t(max_nh_v) * M * qw38::GDN_WY_CH);
            gdn_sc.wy_glast = take.template operator()<float>(cur, size_t(max_nh_v) * max_wy_chunks);

            cur = 0;
            attn_sc.q_gate = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_q) * 512);
            attn_sc.q = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_q) * 256);
            attn_sc.gate = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_q) * 256);
            attn_sc.k = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_kv) * 256);
            attn_sc.v = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_kv) * 256);
            attn_sc.idx_qk = take.template operator()<__nv_bfloat16>(cur, M * geom::IDX_PROJ_OUT);
            attn_sc.idx_q = take.template operator()<__nv_bfloat16>(cur, M * geom::IDX_HEADS * geom::IDX_DIM);
            attn_sc.attn_out = take.template operator()<__nv_bfloat16>(cur, M * size_t(max_nh_q) * 256);
            attn_sc.idx_scores = take.template operator()<float>(cur, M * size_t(max_blocks));
            attn_sc.sel_blocks = take.template operator()<int>(cur, M * geom::IDX_TOP_BLOCKS);
            attn_sc.sel_counts = take.template operator()<int>(cur, M);
            attn_sc.split_m = take.template operator()<float>(cur, 8 * size_t(max_nh_q) * 24);
            attn_sc.split_l = take.template operator()<float>(cur, 8 * size_t(max_nh_q) * 24);
            attn_sc.split_o = take.template operator()<float>(cur, 8 * size_t(max_nh_q) * 24 * 256);

            cur = 0;
            moe_sc.router_logits = take.template operator()<float>(cur, M * geom::NEXP);
            moe_sc.topk_idx = take.template operator()<int>(cur, M * geom::TOPK);
            moe_sc.topk_w = take.template operator()<float>(cur, M * geom::TOPK);
            moe_sc.d_n_groups = take.template operator()<int>(cur, 1);
            moe_sc.grp_exp = take.template operator()<int>(cur, M * geom::TOPK);
            moe_sc.grp_cnt = take.template operator()<int>(cur, M * geom::TOPK);
            moe_sc.grp_off = take.template operator()<int>(cur, M * geom::TOPK);
            moe_sc.grp_routes = take.template operator()<int>(cur, M * geom::TOPK * 8);
            moe_sc.x_had = take.template operator()<__half>(cur, M * geom::HID);
            moe_sc.fc1_part = take.template operator()<float>(cur, 8 * geom::TOPK * size_t(g6nr::EB) * 2 * 2 * 128);
            moe_sc.fc1_act = take.template operator()<__half>(cur, M * geom::TOPK * size_t(g6nr::EB) * 128);
            moe_sc.fc2_part = take.template operator()<float>(cur, fc2_f32_n);
            moe_sc.routed_acc = take.template operator()<float>(cur, M * geom::HID);
            moe_sc.sh_gate_buf = take.template operator()<__nv_bfloat16>(cur, M * geom::SHARED_I);
            moe_sc.sh_up_buf = take.template operator()<__nv_bfloat16>(cur, M * geom::SHARED_I);
            moe_sc.sh_out = take.template operator()<__nv_bfloat16>(cur, M * geom::HID);
            moe_sc.sh_scalar_gate = take.template operator()<float>(cur, M);

            if (has_ple) {
                cur = 0;
                ple_sc.emb = take.template operator()<__nv_bfloat16>(cur, M * geom::PLE_DIM);
                ple_sc.key = take.template operator()<__nv_bfloat16>(cur, M * geom::HC_D);
                ple_sc.query = take.template operator()<__nv_bfloat16>(cur, M * geom::HC_D);
                ple_sc.val = take.template operator()<__nv_bfloat16>(cur, M * geom::HID);
                ple_sc.gv = take.template operator()<__nv_bfloat16>(cur, M * geom::HC_D);
                ple_sc.gv_norm = take.template operator()<__nv_bfloat16>(cur, M * geom::HC_D);
            }

            cur = 0;
            head_logits = take.template operator()<float>(cur, HM * size_t(max_vr));
        } else {
            gdn_sc.qkv_raw = dalloc<__nv_bfloat16>(M * size_t(max_qkv));
            gdn_sc.qkv_conv = dalloc<__nv_bfloat16>(M * size_t(max_qkv));
            gdn_sc.z = dalloc<__nv_bfloat16>(M * size_t(max_nh_v) * geom::GDN_DV);
            gdn_sc.a = dalloc<__nv_bfloat16>(M * size_t(max_nh_v));
            gdn_sc.b = dalloc<__nv_bfloat16>(M * size_t(max_nh_v));
            gdn_sc.core = dalloc<__nv_bfloat16>(M * size_t(max_nh_v) * geom::GDN_DV);
            gdn_sc.wy_qe = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DK);
            gdn_sc.wy_kd = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DK);
            gdn_sc.wy_w = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DK);
            gdn_sc.wy_u = dalloc<uint16_t>(size_t(max_nh_v) * M * geom::GDN_DV);
            gdn_sc.wy_aq = dalloc<uint16_t>(size_t(max_nh_v) * M * qw38::GDN_WY_CH);
            gdn_sc.wy_glast = dalloc<float>(size_t(max_nh_v) * max_wy_chunks);

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
            moe_sc.fc2_part = dalloc<float>(fc2_f32_n);
            moe_sc.routed_acc = dalloc<float>(M * geom::HID);
            moe_sc.sh_gate_buf = dalloc<__nv_bfloat16>(M * geom::SHARED_I);
            moe_sc.sh_up_buf = dalloc<__nv_bfloat16>(M * geom::SHARED_I);
            moe_sc.sh_out = dalloc<__nv_bfloat16>(M * geom::HID);
            moe_sc.sh_scalar_gate = dalloc<float>(M);

            head_logits = dalloc<float>(HM * size_t(max_vr));
        }

        if (o.dense_fp8 && o.pf_mxfp8) {
            const size_t max_k_mx = std::max<size_t>(geom::HC_D, std::max<size_t>(size_t(max_nh_v) * geom::GDN_DV, size_t(max_nh_q) * geom::HEAD_DIM));
            const size_t x8_bytes = ((M + 15) / 16) * (max_k_mx / 32) * qw38::MX8_REC;
            x8_scratch = dalloc<uint8_t>(x8_bytes);
            hc_sc.x8 = x8_scratch;
            gdn_sc.x8 = x8_scratch;
            attn_sc.x8 = x8_scratch;
            moe_sc.x8 = x8_scratch;
        }

        head_sc.blk_max = dalloc<float>(HM * qw38::HEAD_BLKS);
        head_sc.blk_cnt = dalloc<int>(HM * qw38::HEAD_BLKS);
        head_sc.blk_argmax = dalloc<int>(HM * qw38::HEAD_BLKS);
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
        for (int b = 0; b < 2; ++b) {
            QW38_CK(cudaHostAlloc(&h_meta_pinned[b], 3 * M * sizeof(int), cudaHostAllocDefault));
            QW38_CK(cudaHostAlloc(&h_rope_pinned[b], M * sizeof(int3), cudaHostAllocDefault));
            QW38_CK(cudaEventCreateWithFlags(&meta_h2d_done[b], cudaEventDisableTiming));
            meta_h2d_pending[b] = false;
        }
        meta_buf = 0;

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
            if (lm_head[0]) {
                qw38::gemm_bf16_nt_f32(blas, warm_m, warm_v, geom::HID, mixed, lm_head[0], head_logits);
            }
            QW38_CK(cudaStreamSynchronize(stream));
        }

        // 5. Strata-mode: parallel CPU wide-reorder of pinned host routed experts + VRAM hot-expert cache init!
        if (o.strata.enabled && !L.empty()) {
            const Image& im = *img[0];
            const size_t nl = L.size();
            std::vector<const uint8_t*> h_fc1_v(nl), d_fc1_v(nl), h_fc2_v(nl), d_fc2_v(nl);
            std::vector<std::vector<long long>> h_tabs(nl), h_kmixs(nl);
            std::vector<int> rows_v(nl);
            for (size_t li = 0; li < nl; ++li) {
                const int l = L[li].l;
                const auto& rl = L[li].rk[0];
                h_fc1_v[li] = im.host_mut<uint8_t>(lname(l, "tr3.fc1"));
                d_fc1_v[li] = rl.moe_w.t_fc1;
                h_fc2_v[li] = im.host_mut<uint8_t>(lname(l, "tr3.fc2"));
                d_fc2_v[li] = rl.moe_w.t_fc2;
                rows_v[li] = rl.moe_w.t_rows;
                im.read_host(lname(l, "tr3.table"), h_tabs[li]);
                if (im.find(lname(l, "tr3.kmix")) != nullptr) {
                    im.read_host(lname(l, "tr3.kmix"), h_kmixs[li]);
                }
            }
            // Wide-reorder all routed experts in-place in pinned host RAM across CPU threads
            const auto wt0 = std::chrono::steady_clock::now();
            const unsigned nthr = std::max(1u, std::min<unsigned>(static_cast<unsigned>(nl), std::thread::hardware_concurrency()));
            std::atomic<size_t> next_li{0};
            std::vector<std::thread> wthrs;
            for (unsigned t = 0; t < nthr; ++t) {
                wthrs.emplace_back([&]() {
                    for (;;) {
                        const size_t li = next_li.fetch_add(1, std::memory_order_relaxed);
                        if (li >= nl) break;
                        widen_layer_experts_host(
                            const_cast<uint8_t*>(h_fc1_v[li]),
                            const_cast<uint8_t*>(h_fc2_v[li]),
                            h_tabs[li].data(),
                            h_kmixs[li].empty() ? nullptr : h_kmixs[li].data(),
                            rows_v[li]);
                    }
                });
            }
            for (auto& th : wthrs) th.join();
            const double ws = std::chrono::duration<double>(std::chrono::steady_clock::now() - wt0).count();
            std::fprintf(stderr, "# rank %d: qw38 strata CPU wide-reorder %zu layers in %.2f s (%u threads)\n",
                         o.rank0, nl, ws, nthr);

            strata_cache.init(o.strata, static_cast<int>(nl), g6nr::EB, o.max_rows, layer_ids,
                              h_fc1_v, d_fc1_v, h_fc2_v, d_fc2_v, h_tabs, h_kmixs, rows_v, stream);
            {
                std::vector<const __half*> d_sg_v(nl), d_v1b_v(nl), d_v2_v(nl), d_v3b_v(nl);
                std::vector<size_t> v1b_cnt(nl), v2_cnt(nl), v3b_cnt(nl);
                for (size_t li = 0; li < nl; ++li) {
                    const int l = L[li].l;
                    const auto& mw = L[li].rk[0].moe_w;
                    d_sg_v[li] = mw.t_sg;
                    d_v1b_v[li] = mw.t_v1b;
                    d_v2_v[li] = mw.t_v2;
                    d_v3b_v[li] = mw.t_v3b;
                    v1b_cnt[li] = im.entry(lname(l, "tr3.v1b")).nbytes / sizeof(uint16_t);
                    v2_cnt[li]  = im.entry(lname(l, "tr3.v2")).nbytes / sizeof(uint16_t);
                    v3b_cnt[li] = im.entry(lname(l, "tr3.v3b")).nbytes / sizeof(uint16_t);
                }
                strata_cache.init_cpu_pool(h_fc1_v, h_fc2_v, h_tabs, h_kmixs,
                                           d_sg_v, d_v1b_v, d_v2_v, d_v3b_v,
                                           v1b_cnt, v2_cnt, v3b_cnt);
            }
            for (size_t li = 0; li < nl; ++li) {
                auto& mw = L[li].rk[0].moe_w;
                mw.t_ptab = strata_cache.layer_ptab(static_cast<int>(li));
                mw.d_exp_hits = strata_cache.layer_hits(static_cast<int>(li));
                mw.d_is_cold = strata_cache.layer_is_cold(static_cast<int>(li));
                mw.cpu_bridge = strata_cache.cpu_bridge_dev();
                mw.cpu_frac = strata_cache.cpu_moe_frac();
                mw.strata_ctx = &strata_cache;
                mw.strata_li = static_cast<int>(li);
                mw.strata_pf_fn = &StrataExpertCache::prefill_moe_cb;
            }
            if (o.use_graph && o.nr == 1 && o.nloc == 1) {
                const auto t_gw0 = std::chrono::steady_clock::now();
                std::vector<int> dummy_toks(size_t(std::max(1, o.mtp + 1)), 0);
                std::vector<RowOut> dummy_out;
                int n_warm = 0;
                const int qsa_pos[2] = {0, 3000};
                for (int qi = 0; qi < 2; ++qi) {
                    std::vector<qw38::StepSeg> s1 = {{0, 0, qsa_pos[qi], 1}};
                    forward(s1, dummy_toks.data(), nullptr, dummy_out, false, 0, 0, nullptr, false, false, true);
                    ++n_warm;
                    if (o.mtp > 0) {
                        for (int v = 2; v <= o.mtp + 1; ++v) {
                            std::vector<qw38::StepSeg> sv = {{0, 0, qsa_pos[qi], v}};
                            forward(sv, dummy_toks.data(), nullptr, dummy_out, false, o.mtp, 0, nullptr, false, false, true);
                            ++n_warm;
                        }
                    }
                }
                const double gw_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gw0).count();
                std::fprintf(stderr, "# qw38 strata CUDA graphs pre-warmed: %d graphs in %.1f ms\n", n_warm, gw_ms);
            }
        }
    }

    // Copies the unfinished KV + QSA index page at `pos` of each full-attention layer (and MTP when enabled)
    // to `kv_staging`/`idx_staging` (before capture/promote) or back (after restore).
    void stage_partial_page(int slot, uint32_t pos, bool to_staging, int i = 0) {
        if (slot < 0 || slot >= o.slots) die("stage_partial_page: slot");
        const uint32_t rem = pos % static_cast<uint32_t>(geom::KV_PAGE);
        const uint32_t p = pos / static_cast<uint32_t>(geom::KV_PAGE);
        const int r = o.rank0 + i;
        const int nh_kv = geom::nh_kv(o.nr, r);
        const size_t kv_elems = static_cast<size_t>(2) * geom::KV_PAGE * static_cast<size_t>(nh_kv) * geom::HEAD_DIM;
        const size_t idx_elems = static_cast<size_t>(16) * geom::IDX_DIM;
        auto stage_one = [&](qw38::FullLayerKv& kv) {
            __nv_bfloat16* kv_stg = kv.kv_staging + static_cast<size_t>(slot) * kv_elems;
            __nv_bfloat16* idx_stg = kv.idx_staging + static_cast<size_t>(slot) * idx_elems;
            if (rem > 0 && p < static_cast<uint32_t>(max_pps)) {
                __nv_bfloat16* kv_pg = kv.kv_pages + (static_cast<size_t>(slot) * max_pps + p) * kv_elems;
                __nv_bfloat16* idx_pg = kv.idx_pages + (static_cast<size_t>(slot) * max_pps + p) * idx_elems;
                if (to_staging) {
                    QW38_CK(cudaMemcpyAsync(kv_stg, kv_pg, kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                    QW38_CK(cudaMemcpyAsync(idx_stg, idx_pg, idx_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                } else {
                    QW38_CK(cudaMemcpyAsync(kv_pg, kv_stg, kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                    QW38_CK(cudaMemcpyAsync(idx_pg, idx_stg, idx_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                }
            } else if (to_staging) {
                QW38_CK(cudaMemsetAsync(kv_stg, 0, kv_elems * sizeof(__nv_bfloat16), stream));
                QW38_CK(cudaMemsetAsync(idx_stg, 0, idx_elems * sizeof(__nv_bfloat16), stream));
            }
        };
        for (auto& y : L) {
            if (!y.is_full) continue;
            stage_one(y.rk[size_t(i)].kv);
        }
        if (has_mtp) {
            stage_one(mtp_w.kv[size_t(i)]);
        }
        QW38_CK(cudaStreamSynchronize(stream));
    }

    // Acquire an idle double-buffered pinned host staging slot (waits on prior H2D completion if still in flight).
    int meta_acquire() {
        const int b = meta_buf;
        meta_buf ^= 1;
        if (meta_h2d_pending[b]) {
            QW38_CK(cudaEventSynchronize(meta_h2d_done[b]));
            meta_h2d_pending[b] = false;
        }
        return b;
    }

    void meta_upload(int b, int count) {
        const int* h_tok_p = h_meta_pinned[b];
        const int* h_slot_p = h_meta_pinned[b] + o.max_rows;
        const int* h_pos_p = h_meta_pinned[b] + 2 * o.max_rows;
        QW38_CK(cudaMemcpyAsync(d_tok, h_tok_p, size_t(count) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_slot, h_slot_p, size_t(count) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_pos, h_pos_p, size_t(count) * 4, cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaMemcpyAsync(d_rope, h_rope_pinned[b], size_t(count) * sizeof(int3), cudaMemcpyHostToDevice, stream));
        QW38_CK(cudaEventRecord(meta_h2d_done[b], stream));
        meta_h2d_pending[b] = true;
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
    // When `snap_k > 0` (MTP verify step), saves pre-verify (index 0) and post-token-t (index t+1) snapshots for t in 0..snap_k-1!
    // When `vis_n > 0` (multimodal prefill), splices staged vision embeddings into HC streams and masks image tokens in PLE.
    // When `rope_thw != nullptr`, uses per-row 3D MRoPE (t, h, w) coords; else uses isotropic (pos + slot_mrope_delta[slot]).
    // When `head_last_only` is true (prefill final chunk), runs final HC + lm_head only on row `M - 1` into `out[M - 1]`.
    // When `exact_head` is true (logprobs/prompt_logprobs), bypasses trunk_head2 for exact full-vocab FP8 lm_head + LSE.
    void forward(const std::vector<qw38::StepSeg>& segs, const int* toks, const RowSample* samples,
                 std::vector<RowOut>& out, bool skip_head = false, int snap_k = 0,
                 int vis_n = 0, const int3* rope_thw = nullptr, bool head_last_only = false,
                 bool exact_head = false, bool capture_only = false) {
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
        // Commit any async expert promotions finished during the preceding MTP draft window (zero-delay 1-round adaptation!)
        if (!capture_only && o.strata.enabled && M < 32) strata_cache.commit_ready_promotions(stream, false);

        if (!capture_only) {
            const int mb = meta_acquire();
            int* h_tok_p = h_meta_pinned[mb];
            int* h_slot_p = h_meta_pinned[mb] + o.max_rows;
            int* h_pos_p = h_meta_pinned[mb] + 2 * o.max_rows;
            std::memcpy(h_tok_p, toks, size_t(M) * sizeof(int));
            for (const auto& g : segs) {
                const int dlt = slot_mrope_delta[size_t(g.slot)];
                for (int t = 0; t < g.len; ++t) {
                    h_slot_p[g.row0 + t] = g.slot;
                    h_pos_p[g.row0 + t] = g.pos0 + t;
                    if (!rope_thw) {
                        const int rp = g.pos0 + t + dlt;
                        h_rope_pinned[mb][g.row0 + t] = make_int3(rp, rp, rp);
                    }
                }
            }
            if (rope_thw) std::memcpy(h_rope_pinned[mb], rope_thw, size_t(M) * sizeof(int3));
            meta_upload(mb, M);
        }

        const bool use_direct_ar = (o.nr > 1 && o.nloc == 1 && direct_send_dst && direct_ar_wait && direct_hc_inject);
        const bool use_direct_send = (use_direct_ar && (M > 8 || !direct_copy_ring_wait));
        const bool use_sp_hc = (use_direct_ar && skip_head && !has_mtp && !has_dfl && tap_dump_fp == nullptr && M >= 64 && L.size() >= 2 &&
                                step_begin_sp && direct_hc_rs_inject_norm && direct_ag_assemble);
        const int sp_slice = use_sp_hc ? ((M + o.nr - 1) / o.nr) : M;
        const int sp_r0 = use_sp_hc ? std::min(o.rank0 * sp_slice, M) : 0;
        const int sp_m = use_sp_hc ? std::min(sp_slice, M - sp_r0) : M;
        const bool last_only = head_last_only && !skip_head && M > 1;
        const int head_M = last_only ? 1 : M;

        // 0a. Notify multi-collective fabric step start (single enqueue_step for all collectives on TP>=2)
        if (!capture_only) {
            if (use_sp_hc) step_begin_sp(M, int(L.size()), sp_slice);
            else if (step_begin) step_begin(M, skip_head, -1, head_M);
        }

        // Mask <|image_pad|> -> TOK_EOS for Layer-1 PLE n-gram hashing (matches Qwen4ExpTextModel conv_mask).
        std::vector<int> ple_tok_buf;
        const int* ple_toks = toks;
        if (!capture_only && has_ple && vis_n > 0) {
            ple_tok_buf.assign(toks, toks + M);
            for (int& tk : ple_tok_buf) if (tk == qw38v::TOK_IMAGE) tk = geom::TOK_EOS;
            ple_toks = ple_tok_buf.data();
        }

        // 0b. Consume cross-chunk PLE prefetch if it matches, else kick off async PLE NVMe fetch
        //     into the alternate double buffer BEFORE Layer 0 GPU compute!
        bool ple_in_flight = false;
        int ple_cur_buf = 0;
        if (!capture_only && has_ple && o.ple_async && ple_thr.joinable()) {
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
            if ((has_dfl || tap_dump_fp != nullptr) && y.l >= 5 && y.l <= 45 && ((y.l - 5) % 10 == 0)) {
                const int tap_idx = (y.l - 5) / 10;
                if (dfl_pf_hi > dfl_pf_lo && dfl.pf_feat) {
                    qw38::dfl::k_pf_tap_hc_mean<<<dim3(geom::HID / 256, M), 256, 0, stream>>>(
                        hyper, d_pos, dfl_pf_lo, dfl_pf_hi, dfl.pf_feat, tap_idx);
                }
                if (dfl_step_taps && M <= dfl_step_taps_cap) {
                    qw38::dfl::k_tap_hc_mean<<<dim3(geom::HID / 256, M), 256, 0, stream>>>(
                        hyper, dfl_step_taps, tap_idx);
                }
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
            const bool use_trunk_h2 = o.trunk_head2 && !exact_head && head_M <= 48 && o.nc <= qw38::MtpHead2::SL_TRUNK &&
                                      !mtp_head2.empty() && !fp8_lm_head.empty();
            const int head_chunk = use_trunk_h2 ? 8 : 256;
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                for (int r0 = 0; r0 < head_M; r0 += head_chunk) {
                    const int mc = std::min(head_M - r0, head_chunk);
                    const __nv_bfloat16* mx_ptr = mixed + size_t(r0) * geom::HID;
                    if (use_trunk_h2 && i < static_cast<int>(mtp_head2.size()) && mtp_head2[size_t(i)].ready() &&
                        i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q) {
                        qw38::trunk_head2_pack(mtp_head2[size_t(i)], fp8_lm_head[size_t(i)], mx_ptr,
                                               head_logits, mtp_h2_sc, o.nc,
                                               head_dev_pkt + size_t(r0) * hrb, hrb, mc, stream);
                    } else {
                        if (head_M >= 16 && x8_scratch && i < static_cast<int>(fp8_lm_head.size()) &&
                            fp8_lm_head[size_t(i)].mx_sc && (fp8_lm_head[size_t(i)].N & 15) == 0) {
                            qw38::qw38_bf16_to_mx8(mx_ptr, x8_scratch, mc, geom::HID, stream);
                            qw38::qw38_gemm_mx8_nt<true>(x8_scratch, fp8_lm_head[size_t(i)], head_logits, mc, stream);
                        } else if (head_M <= 20 && i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q) {
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
                }
                QW38_CK(cudaMemcpyAsync(head_host_parts + size_t(r) * rank_pkt_bytes,
                                        head_dev_pkt, rank_pkt_bytes, cudaMemcpyDeviceToHost, stream));
            }
        };

        // Fast CUDA Graph path for decode (M <= 16, all_len1) AND MTP verify (M <= 64, all_verify, including KStopB V < snap_k+1)!
        const int V_chk = (!segs.empty() && segs[0].len >= 2 && segs[0].len <= snap_k + 1) ? segs[0].len : (snap_k + 1);
        bool all_verify = (!all_len1 && snap_k > 0 && !segs.empty() && static_cast<int>(segs.size()) * V_chk == M);
        if (all_verify) {
            for (size_t s = 0; s < segs.size(); ++s) {
                if (segs[s].len != V_chk || segs[s].row0 != static_cast<int>(s) * V_chk) { all_verify = false; break; }
            }
        }
        const bool verify_graph_ok = all_verify && M <= 64;
        const bool graph_ok = o.use_graph && !o.profile && !skip_head && !last_only && !exact_head && vis_n == 0 && !rope_thw &&
                              !(dfl_pf_hi > dfl_pf_lo) &&
                              ((snap_k == 0 && M <= 16 && all_len1) || verify_graph_ok) &&
                              (o.nr == 1 || o.nloc == o.nr || static_cast<bool>(step_parity));
        if (graph_ok) {
            const int need_qsa = (((max_pos + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS) ? 1 : 0;
            const int ns_attn = qw38::gqa_split_count(M, geom::nh_kv(o.nr, o.rank0), max_pos, attn_sc.split_o != nullptr);
            const int par = step_parity ? step_parity() : 0;
            const GraphKey gk{M, need_qsa, ns_attn, par, snap_k, snap_k > 0 ? V_chk : 1};
            const bool split_l0_ple = (L.size() >= 2 && geom::is_ple(L[1].l) && has_ple);

            if (capture_only) {
                if (graphs.find(gk) != graphs.end()) return;
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

                    cudaGraph_t gb = nullptr;
                    QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                    run_ple_gpu();
                    for (size_t li = 1; li < L.size(); ++li) run_layer_gpu(li, coll);
                    run_head_gpu();
                    QW38_CK(cudaStreamEndCapture(stream, &gb));
                    QW38_CK(cudaGraphInstantiate(&gp.exec_b, gb, nullptr, nullptr, 0));
                    QW38_CK(cudaGraphDestroy(gb));
                } else {
                    cudaGraph_t gb = nullptr;
                    QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                    qw38::k_embed_hc4<<<M, 256, 0, stream>>>(d_tok, embed, hyper, M);
                    for (size_t li = 0; li < L.size(); ++li) {
                        if (geom::is_ple(L[li].l) && has_ple) run_ple_gpu();
                        run_layer_gpu(li, coll);
                    }
                    run_head_gpu();
                    QW38_CK(cudaStreamEndCapture(stream, &gb));
                    QW38_CK(cudaGraphInstantiate(&gp.exec_b, gb, nullptr, nullptr, 0));
                    QW38_CK(cudaGraphDestroy(gb));
                }
                graphs.emplace(gk, gp);
                return;
            }

            if (o.strata.enabled && M <= 8) strata_cache.cpu_pool.begin_step();
            const auto t_v0 = std::chrono::steady_clock::now();
            auto it = graphs.find(gk);
            const bool did_cap = (it == graphs.end());
            if (did_cap) {
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
            const auto t_v1 = std::chrono::steady_clock::now();
            if (o.strata.enabled && M <= 8) strata_cache.cpu_pool.end_step();
            if (o.strata.enabled) {
                const uint64_t dt_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t_v1 - t_v0).count());
                if (did_cap) { strata_cache.stat_cap_ns += dt_ns; ++strata_cache.stat_cap_n; }
                strata_cache.stat_verify_ns += dt_ns;
                strata_cache.stat_verify_toks += uint64_t(M);
                const auto t_m0 = std::chrono::steady_clock::now();
                strata_cache.maintain_after_step(stream);
                strata_cache.stat_maint_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_m0).count());
            }
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
        if (capture_only) return;
        const bool use_cpu_step = o.strata.enabled && (M <= 8 || (M >= 32 && M <= o.strata.pf_cpu_max_m));
        const auto t_eg0 = std::chrono::steady_clock::now();
        if (use_cpu_step) strata_cache.cpu_pool.begin_step(M >= 32);
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
                    upload_ple_to_dev();
                    run_ple_gpu();
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
            if ((has_dfl || tap_dump_fp != nullptr) && y.l >= 5 && y.l <= 45 && ((y.l - 5) % 10 == 0)) {
                const int tap_idx = (y.l - 5) / 10;
                if (dfl_pf_hi > dfl_pf_lo && dfl.pf_feat) {
                    qw38::dfl::k_pf_tap_hc_mean<<<dim3(geom::HID / 256, M), 256, 0, stream>>>(
                        hyper, d_pos, dfl_pf_lo, dfl_pf_hi, dfl.pf_feat, tap_idx);
                }
                if (dfl_step_taps && M <= dfl_step_taps_cap) {
                    qw38::dfl::k_tap_hc_mean<<<dim3(geom::HID / 256, M), 256, 0, stream>>>(
                        hyper, dfl_step_taps, tap_idx);
                }
            }
        }

        if (skip_head) {
            if (use_cpu_step) {
                QW38_CK(cudaStreamSynchronize(stream));
                strata_cache.cpu_pool.end_step();
            }
            if (step_end) step_end();
            if (o.profile) { ++prof.steps; prof.tokens += M; }
            return;
        }

        t_start();
        run_head_gpu(hn_ready);
        QW38_CK(cudaStreamSynchronize(stream));
        if (use_cpu_step) strata_cache.cpu_pool.end_step();
        if (o.strata.enabled) {
            if (M <= 8) {
                strata_cache.stat_verify_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_eg0).count());
                strata_cache.stat_verify_toks += uint64_t(M);
                ++strata_cache.stat_eager_n;
            }
            const auto t_m0 = std::chrono::steady_clock::now();
            strata_cache.maintain_after_step(stream);
            strata_cache.stat_maint_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_m0).count());
        }
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
        const int cap_chunk = mtp_sc_cap ? static_cast<int>(mtp_sc_cap) : len;
        if (len > cap_chunk) {
            for (int off = 0; off < len; off += cap_chunk) {
                mtp_sync_kv(slot, row0 + off, pos0 + off, std::min(cap_chunk, len - off), next_toks + off);
            }
            return;
        }
        const int mb = meta_acquire();
        int* h_tok_p = h_meta_pinned[mb];
        int* h_slot_p = h_meta_pinned[mb] + o.max_rows;
        int* h_pos_p = h_meta_pinned[mb] + 2 * o.max_rows;
        const int dlt = slot_mrope_delta[size_t(slot)];
        for (int t = 0; t < len; ++t) {
            h_tok_p[t] = next_toks[t];
            h_slot_p[t] = slot;
            h_pos_p[t] = pos0 + t;
            const int rp = pos0 + t + dlt;
            h_rope_pinned[mb][t] = make_int3(rp, rp, rp);
        }
        meta_upload(mb, len);

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
        const int mb = meta_acquire();
        int* h_tok_p = h_meta_pinned[mb];
        int* h_slot_p = h_meta_pinned[mb] + o.max_rows;
        int* h_pos_p = h_meta_pinned[mb] + 2 * o.max_rows;
        std::vector<qw38::StepSeg> segs(static_cast<size_t>(M));
        int max_pos = 0;
        for (int s = 0; s < M; ++s) {
            h_tok_p[s] = toks[s];
            h_slot_p[s] = slots[size_t(s)];
            h_pos_p[s] = pos_mtp[s];
            const int rp = pos_mtp[s] + slot_mrope_delta[size_t(slots[size_t(s)])];
            h_rope_pinned[mb][s] = make_int3(rp, rp, rp);
            if (pos_mtp[s] > max_pos) max_pos = pos_mtp[s];
            segs[size_t(s)] = {slots[size_t(s)], s, pos_mtp[s], 1};
        }
        meta_upload(mb, M);

        bool all_greedy = true;
        if (samples) {
            for (int m = 0; m < M; ++m) {
                if (samples[m].inv_temp > 0.f) { all_greedy = false; break; }
            }
        }

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

            // L048 MoE (GPU-native INT4/FP8/BF16!) + AR + Inject
            qw38::hc_prepare(blas, mtp_w.mlp_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                qw38::mtp_moe_forward(blas, mtp_w.moe[size_t(i)],
                                      mtp_w.i4_exp_gu[size_t(i)], mtp_w.i4_exp_dn[size_t(i)],
                                      mtp_w.fp8_exp_gu[size_t(i)], mtp_w.fp8_exp_dn[size_t(i)],
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

            // MTP Final HC -> 2-bit screen + top-64 exact rescore (or FP8/BF16 lm_head) -> head_pack -> D2H
            qw38::hc_prepare(blas, mtp_w.final_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
            constexpr int HEAD_CHUNK = 256;
            for (int i = 0; i < o.nloc; ++i) {
                const int r = o.rank0 + i;
                if (o.mtp_fast && M <= 8 && i < static_cast<int>(mtp_head2.size()) && mtp_head2[size_t(i)].ready() &&
                    i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q) {
                    qw38::mtp_head2_pack(mtp_head2[size_t(i)], fp8_lm_head[size_t(i)], mixed, head_logits,
                                         mtp_h2_sc, head_sc, o.nc, head_dev_pkt, hrb, M, all_greedy, stream);
                } else {
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
                                             head_dev_pkt + size_t(r0) * hrb, hrb, stream, all_greedy);
                    }
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
            const MtpGraphKey mk{M, need_qsa, ns_attn, par, hin_kind, all_greedy ? 1 : 0};
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

    // Roll back slot `slot`'s conv/idx_raw state to exact state after `kept` verify tokens
    // (`kept == 0` = pre-verify state before token 0; `1 <= kept <= max_snap_k` = post-token-(kept - 1)).
    void mtp_rollback_slot(int slot, int kept) {
        if (kept < 0 || kept > max_snap_k) die("mtp_rollback_slot: bad kept=" + std::to_string(kept));
        const size_t snap_stride = size_t(max_snap_k + 1);
        if (has_ple && ple_st.conv_hist_snap) {
            const size_t pb = size_t(geom::PLE_NHIST) * geom::HC_D * 2;
            QW38_CK(cudaMemcpyAsync(
                ple_st.conv_hist + size_t(slot) * geom::PLE_NHIST * geom::HC_D,
                ple_st.conv_hist_snap + (size_t(slot) * snap_stride + size_t(kept)) * geom::PLE_NHIST * geom::HC_D,
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
                            rl.kv.idx_raw_snap + (size_t(slot) * snap_stride + size_t(kept)) * 4 * geom::IDX_DIM,
                            ib, cudaMemcpyDeviceToDevice, stream));
                    }
                    if (rl.kv.idx_raw_thw_snap && rl.kv.idx_raw_thw) {
                        QW38_CK(cudaMemcpyAsync(
                            rl.kv.idx_raw_thw + size_t(slot),
                            rl.kv.idx_raw_thw_snap + size_t(slot) * snap_stride + size_t(kept),
                            sizeof(int3), cudaMemcpyDeviceToDevice, stream));
                    }
                } else {
                    const size_t qkv_l = size_t(geom::gdn_qkv_local(o.nr, r));
                    if (rl.gdn_st.conv_snap) {
                        const size_t cb = geom::GDN_NHIST * qkv_l * 2;
                        QW38_CK(cudaMemcpyAsync(
                            rl.gdn_st.conv + size_t(slot) * geom::GDN_NHIST * qkv_l,
                            rl.gdn_st.conv_snap + (size_t(slot) * snap_stride + size_t(kept)) * geom::GDN_NHIST * qkv_l,
                            cb, cudaMemcpyDeviceToDevice, stream));
                    }
                }
            }
        }
    }

    // Batched GDN state commit: replays exact `kept_c[i]` verified tokens into `gdn_st.S` ONCE for each slot!
    // Paired with read-only `k_gdn_rec_verify_batch`, eliminates `S_snap` (2.7 GB VRAM) and cuts GDN verify/rollback DRAM >2x.
    void mtp_commit_gdn_batch(const std::vector<int>& slots_c, const std::vector<int>& kept_c) {
        const int n_tot = static_cast<int>(slots_c.size());
        if (n_tot <= 0) return;
        for (int off = 0; off < n_tot; off += 16) {
            const int n = std::min(16, n_tot - off);
            qw38::GdnCommitArgs args{};
            int any = 0;
            for (int c = 0; c < n; ++c) {
                args.slot[c] = slots_c[size_t(off + c)];
                args.kept[c] = kept_c[size_t(off + c)];
                if (args.kept[c] > 0) ++any;
            }
            if (any == 0) continue;
            for (auto& y : L) {
                if (y.is_full) continue;
                for (int i = 0; i < o.nloc; ++i) {
                    const int r = o.rank0 + i;
                    auto& rl = y.rk[size_t(i)];
                    if (!rl.gdn_st.v_qkv || rl.gdn_st.v_stride <= 0) continue;
                    const int nh_k = geom::gdn_kh(o.nr, r);
                    const int nh_v = geom::gdn_vh(o.nr);
                    const int k0 = geom::gdn_k0(o.nr, r);
                    const int v0 = geom::gdn_v0(o.nr, r);
                    qw38::k_gdn_commit_batch<<<n * nh_v * 2, 128, 0, stream>>>(
                        rl.gdn_st.v_qkv, rl.gdn_st.v_a, rl.gdn_st.v_b, rl.gdn_st.v_pos0,
                        rl.gdn_w.A_log, rl.gdn_w.dt_bias, rl.gdn_st.S,
                        args, n, rl.gdn_st.v_stride, nh_k, nh_v, k0, v0);
                }
            }
        }
    }

    void mtp_restore_idx_raw(int slot) {
        if (!has_mtp) return;
        const size_t idx_slot_b = 4 * geom::IDX_DIM * sizeof(__nv_bfloat16);
        const size_t off = size_t(slot) * 4 * geom::IDX_DIM;
        for (int li = 0; li < o.nloc; ++li) {
            auto& kv = mtp_w.kv[size_t(li)];
            if (kv.idx_raw && kv.idx_raw_snap) {
                QW38_CK(cudaMemcpyAsync(kv.idx_raw + off, kv.idx_raw_snap + off, idx_slot_b, cudaMemcpyDeviceToDevice, stream));
            }
            if (kv.idx_raw_thw && kv.idx_raw_thw_snap) {
                QW38_CK(cudaMemcpyAsync(kv.idx_raw_thw + size_t(slot), kv.idx_raw_thw_snap + size_t(slot),
                                        sizeof(int3), cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    // Draft K tokens per session for an arbitrary subset of S2 `slots`, returning flat [slots.size() * K] token IDs.
    // `pos_in[i]` is the last committed trunk position q (whose hyper state is in `last_hyper[slots[i]]`),
    // and `tok_in[i]` is the golden token at position q + 1. Preserves L048 idx_raw across speculative k >= 1 steps.
    // `mode`: 0 = normal full K, 1 = stage-1 lazy head (keeps mtp_hid_cur & defers idx_raw restore),
    //         2 = stage-2 lazy tail (starts from mtp_hid_cur, restores idx_raw at end),
    //         3 = lazy mid-step (starts from mtp_hid_cur, keeps mtp_hid_cur & defers idx_raw restore).
    void mtp_draft_k(const std::vector<int>& slots, const std::vector<int>& pos_in, const std::vector<int>& tok_in,
                     const std::vector<RowSample>& base_sm, int K, std::vector<int>& out_flat,
                     std::vector<float>* out_lp_flat = nullptr, int mode = 0) {
        const size_t S = slots.size();
        out_flat.assign(S * size_t(K), 0);
        if (out_lp_flat) out_lp_flat->assign(S * size_t(K), 0.0f);
        if (S == 0 || K <= 0) return;

        bool all_greedy = true;
        for (size_t i = 0; i < S; ++i) {
            if (base_sm[i].inv_temp > 0.f) { all_greedy = false; break; }
        }
        // Under S2's exact verification (target_sample == draft), any draft token sequence preserves the
        // exact target output distribution (bit-exact vs MTP=0 at the same T/top_p/seed), while greedy
        // argmax drafts mathematically maximize P(target_sample == draft) and enable single-graph K-step drafting.
        const bool use_greedy_draft = all_greedy || o.mtp_fast;
        const bool fast_k_graph = use_greedy_draft && o.mtp_fast && o.use_graph && !o.profile &&
                                  S <= 16 && K <= 8 && o.nloc == 1 &&
                                  (o.nr == 1 || (mode == 0 && static_cast<bool>(step_begin_mtp_k) &&
                                                 static_cast<bool>(mtp_argmax_ar_chain) &&
                                                 static_cast<bool>(step_parity)));
        if (fast_k_graph) {
            const int M = static_cast<int>(S);
            const int mb = meta_acquire();
            int* h_tok_p = h_meta_pinned[mb];
            int* h_slot_p = h_meta_pinned[mb] + o.max_rows;
            int* h_pos_p = h_meta_pinned[mb] + 2 * o.max_rows;
            int max_pos = 0;
            for (int s = 0; s < M; ++s) {
                h_tok_p[s] = tok_in[size_t(s)];
                h_slot_p[s] = slots[size_t(s)];
                h_pos_p[s] = pos_in[size_t(s)];
                const int rp = pos_in[size_t(s)] + slot_mrope_delta[size_t(slots[size_t(s)])];
                h_rope_pinned[mb][s] = make_int3(rp, rp, rp);
                if (pos_in[size_t(s)] > max_pos) max_pos = pos_in[size_t(s)];
            }
            meta_upload(mb, M);

            if (o.nr > 1) step_begin_mtp_k(M, K);

            const int max_pos_k = max_pos + K - 1;
            const int need_qsa = (((max_pos_k + 1) / geom::IDX_COMPRESS) > geom::IDX_TOP_BLOCKS) ? 1 : 0;
            const int ns_attn = qw38::gqa_split_count(M, geom::nh_kv(o.nr, o.rank0), max_pos_k, attn_sc.split_o != nullptr);
            const int par = step_parity ? step_parity() : 0;
            const MtpKGraphKey kgk{M, K, need_qsa, ns_attn, par, mode};

            const auto t_mtp0 = std::chrono::steady_clock::now();
            auto it = mtp_k_graphs.find(kgk);
            const bool did_mtp_cap = (it == mtp_k_graphs.end());
            if (did_mtp_cap) {
                std::vector<qw38::StepSeg> segs_k(static_cast<size_t>(M));
                for (int s = 0; s < M; ++s) segs_k[size_t(s)] = {0, s, max_pos_k, 1};
                const size_t part_stride = size_t(M) * geom::HID;
                const bool use_direct_ar = (o.nr > 1 && direct_send_dst && direct_ar_wait && direct_hc_inject);
                const bool use_direct_send = (use_direct_ar && (M > 8 || !direct_copy_ring_wait));

                const bool is_cont = (mode == 2 || mode == 3);
                const bool keep_state = (mode == 1 || mode == 3);
                cudaGraph_t kg = nullptr;
                cudaGraphExec_t kexec = nullptr;
                QW38_CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                if (!is_cont) {
                    qw38::k_mtp_gather_hid<<<M, 256, 0, stream>>>(d_slot, last_hyper, mtp_hid_in, M);
                }

                for (int k = 0; k < K; ++k) {
                    const __nv_bfloat16* hin = (!is_cont && k == 0) ? mtp_hid_in : mtp_hid_cur;
                    qw38::k_gather_embed<<<M, 256, 0, stream>>>(d_tok, embed, mtp_sc.emb_raw, M);
                    qw38::k_rmsnorm_1pw<geom::HID><<<M, 256, 0, stream>>>(mtp_sc.emb_raw, mtp_w.pre_fc_norm_emb, mtp_sc.emb_norm, M, 1, geom::RMS_EPS);
                    qw38::k_rmsnorm_1pw<geom::HID><<<M * geom::HC, 256, 0, stream>>>(hin, mtp_w.pre_fc_norm_hid, mtp_sc.hid_norm, M, geom::HC, geom::RMS_EPS);

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

                    int coll = 3 * k;
                    // L048 Full Attention + AR + Inject
                    qw38::hc_prepare(blas, mtp_w.attn_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
                    {
                        const int r = o.rank0;
                        __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                        qw38::full_attn_forward(blas, mtp_w.attn[0], mtp_w.kv[0], attn_sc, mixed, p_dst,
                                                d_slot, d_pos, segs_k.data(), M, M, o.nr, r, max_pps, max_blocks, stream, 0, d_rope);
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
                        ++coll;
                        qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(mtp_sc.mtp_hyper, parts, hc_sc.inj, M, o.nr, part_stride);
                    }

                    // Save committed L048 idx_raw right after k=0 attention before speculative k>=1 steps touch it
                    if (!is_cont && k == 0 && (K > 1 || keep_state)) {
                        auto& kv = mtp_w.kv[0];
                        qw38::k_mtp_idx_snap<true><<<M, 64, 0, stream>>>(
                            d_slot, kv.idx_raw, kv.idx_raw_snap, kv.idx_raw_thw, kv.idx_raw_thw_snap, M);
                    }

                    // L048 MoE (INT4 routed experts!) + AR + Inject
                    qw38::hc_prepare(blas, mtp_w.mlp_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
                    {
                        const int r = o.rank0;
                        __nv_bfloat16* p_dst = use_direct_send ? direct_send_dst(coll) : (parts + size_t(r) * part_stride);
                        qw38::mtp_moe_forward(blas, mtp_w.moe[0],
                                              mtp_w.i4_exp_gu[0], mtp_w.i4_exp_dn[0],
                                              mtp_w.fp8_exp_gu[0], mtp_w.fp8_exp_dn[0],
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
                        ++coll;
                        qw38::k_hc_inject<<<M * 5, 64, 0, stream>>>(mtp_sc.mtp_hyper, parts, hc_sc.inj, M, o.nr, part_stride);
                    }

                    if (k + 1 < K || keep_state) {
                        QW38_CK(cudaMemcpyAsync(mtp_hid_cur, mtp_sc.mtp_hyper, size_t(M) * geom::HC_D * sizeof(__nv_bfloat16),
                                                cudaMemcpyDeviceToDevice, stream));
                    }

                    // MTP Final HC -> 2-bit screen + exact top-64 rescore -> on-GPU argmax chain!
                    qw38::hc_prepare(blas, mtp_w.final_hc, mtp_sc.mtp_hyper, mixed, hc_sc, M, stream);
                    if (M <= 8 && !mtp_head2.empty() && mtp_head2[0].ready() && !fp8_lm_head.empty() && fp8_lm_head[0].q) {
                        qw38::mtp_head2_argmax_pair(mtp_head2[0], fp8_lm_head[0], mixed, head_logits,
                                                    mtp_h2_sc, d_mtp_argmax, M, stream, d_mtp_lp);
                    } else {
                        if (!fp8_lm_head.empty() && fp8_lm_head[0].q) {
                            qw38::launch_fp8_gemv_smallm<true>(mixed, fp8_lm_head[0], head_logits, M, stream);
                        } else {
                            qw38::launch_bf16_gemv_smallm<true>(mixed, lm_head[0], head_logits, M, v_rows[0], geom::HID, stream);
                        }
                        qw38::k_qw38_head_pass1<<<M * qw38::HEAD_BLKS, 256, 0, stream>>>(
                            head_logits, v_rows[0], v_lo[0], head_sc.blk_max, head_sc.blk_cnt, head_sc.blk_argmax);
                        qw38::k_qw38_head_argmax_to_pair<<<M, qw38::HEAD_BLKS, 0, stream>>>(
                            head_sc.blk_max, head_sc.blk_argmax, d_mtp_argmax);
                    }

                    if (o.nr == 1) {
                        qw38::k_mtp_chain_tp1<<<1, 32, 0, stream>>>(
                            d_mtp_argmax, d_mtp_draft_out, d_tok, d_pos, d_rope, M, K, k,
                            d_mtp_lp, d_mtp_draft_lp);
                    } else {
                        mtp_argmax_ar_chain(coll, d_mtp_argmax, d_mtp_draft_out, d_tok, d_pos, d_rope, M, K, k);
                    }
                }

                if ((mode == 0 && K > 1) || mode == 2) {
                    auto& kv = mtp_w.kv[0];
                    qw38::k_mtp_idx_snap<false><<<M, 64, 0, stream>>>(
                        d_slot, kv.idx_raw, kv.idx_raw_snap, kv.idx_raw_thw, kv.idx_raw_thw_snap, M);
                }
                QW38_CK(cudaMemcpyAsync(h_mtp_draft_out, d_mtp_draft_out, size_t(M) * size_t(K) * sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
                if (o.nr == 1 && d_mtp_draft_lp && h_mtp_draft_lp) {
                    QW38_CK(cudaMemcpyAsync(h_mtp_draft_lp, d_mtp_draft_lp, size_t(M) * size_t(K) * sizeof(float),
                                            cudaMemcpyDeviceToHost, stream));
                }
                QW38_CK(cudaStreamEndCapture(stream, &kg));
                QW38_CK(cudaGraphInstantiate(&kexec, kg, nullptr, nullptr, 0));
                QW38_CK(cudaGraphDestroy(kg));
                mtp_k_graphs.emplace(kgk, kexec);
                QW38_CK(cudaGraphLaunch(kexec, stream));
            } else {
                QW38_CK(cudaGraphLaunch(it->second, stream));
            }

            QW38_CK(cudaStreamSynchronize(stream));
            if (o.strata.enabled) {
                const uint64_t dt_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_mtp0).count());
                if (did_mtp_cap) strata_cache.stat_mtp_cap_ns += dt_ns;
                strata_cache.stat_mtp_ns += dt_ns;
                ++strata_cache.stat_mtp_calls;
            }
            if (o.nr > 1 && step_end) step_end();
            std::memcpy(out_flat.data(), h_mtp_draft_out, size_t(M) * size_t(K) * sizeof(int));
            if (out_lp_flat && o.nr == 1 && h_mtp_draft_lp) {
                std::memcpy(out_lp_flat->data(), h_mtp_draft_lp, size_t(M) * size_t(K) * sizeof(float));
            }
            return;
        }

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
                if (use_greedy_draft) dsm[i].inv_temp = 0.0f;
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

        // 1. Draft K tokens per session using Layer-48 MTP (unified fast K-step graph path!)
        std::vector<int> d_pos_in(static_cast<size_t>(S));
        std::vector<RowSample> base_sm(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s) {
            d_pos_in[size_t(s)] = pos[size_t(s)] - 1;
            base_sm[size_t(s)] = RowSample{inv_temp, top_p, seed, 0u};
        }
        std::vector<int> drafts_flat;
        mtp_draft_k(slots, d_pos_in, cur_tok, base_sm, K, drafts_flat);
        std::vector<std::vector<int>> drafts(static_cast<size_t>(S), std::vector<int>(static_cast<size_t>(K), 0));
        for (int s = 0; s < S; ++s) {
            for (int k = 0; k < K; ++k) drafts[size_t(s)][size_t(k)] = drafts_flat[size_t(s) * size_t(K) + size_t(k)];
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

        // 3. Exact greedy verification + batched GDN state commit + lightweight rollback + MTP KV sync!
        std::vector<int> c_slots(static_cast<size_t>(S));
        std::vector<int> c_kept(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s) {
            int n_acc = 0;
            for (int k = 0; k < K; ++k) {
                if (vout[size_t(s * V + k)].tok == drafts[size_t(s)][size_t(k)]) ++n_acc;
                else break;
            }
            total_accepted += n_acc;
            c_slots[size_t(s)] = s;
            c_kept[size_t(s)] = n_acc + 1;

            // Emitted tokens: accepted drafts [0..n_acc-1] + trunk token at row n_acc
            for (int k = 0; k < n_acc; ++k) emitted[size_t(s)].push_back(drafts[size_t(s)][size_t(k)]);
            const int bonus = vout[size_t(s * V + n_acc)].tok;
            emitted[size_t(s)].push_back(bonus);

            // Roll back lightweight conv/idx_raw if partial accept (< K)
            if (n_acc < K) {
                mtp_rollback_slot(s, n_acc + 1);
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
        mtp_commit_gdn_batch(c_slots, c_kept);
    }

    // ---------------------------------------------------------------- QSpine-DF2 + Inline Tap Capture Methods
    void dfl_begin_prompt(int prompt_len) {
        if (!has_dfl) return;
        dfl_pf_hi = prompt_len;
        dfl_pf_lo = std::max(0, prompt_len - qw38::dfl::WIN);
        dfl.ctx_lo = dfl_pf_lo;
    }

    void dfl_begin_prompt_window(int lo, int hi) {
        if (!has_dfl) return;
        dfl_pf_lo = lo;
        dfl_pf_hi = hi;
        dfl.ctx_lo = lo;
    }

    void dfl_end_prompt_capture() {
        dfl_pf_lo = 0;
        dfl_pf_hi = 0;
    }

    void dfl_finish_prompt() {
        if (!has_dfl) return;
        const int n = dfl_pf_hi - dfl_pf_lo;
        const int lo = dfl_pf_lo;
        dfl_pf_lo = dfl_pf_hi = 0;
        if (n > 0) dfl.ingest_context(dfl.pf_feat, n, lo, stream);
    }

    void dfl_finish_prompt_range(int lo, int from, int hi) {
        if (!has_dfl) return;
        dfl_pf_lo = dfl_pf_hi = 0;
        const int n = hi - from;
        if (n > 0 && dfl.pf_feat) {
            const __half* src = dfl.pf_feat + size_t(from - lo) * qw38::dfl::NT * geom::HID;
            dfl.ingest_context(src, n, from, stream);
        }
    }

    // Helper: serialize one QTAP record (25880 B) into `dst` vector (or write directly to tap_dump_fp if dst==nullptr).
    void tap_emit_one(std::vector<uint8_t>* dst, int32_t pos, int32_t tin, const RowOut* ro,
                      int32_t l48_tok1, const uint16_t* tap_row) {
        constexpr size_t TAP_B = size_t(qw38::dfl::NT) * geom::HID * 2;
        constexpr size_t REC_B = 24 + 128 + 128 + TAP_B; // 25880 B
        uint8_t buf[280];
        int32_t hdr[6] = {0x50415451, pos, tin, -1, l48_tok1, 0};
        int32_t tids[32] = {};
        float tlg[32] = {};
        if (ro) {
            hdr[3] = ro->tok;
            hdr[5] = std::min(32, static_cast<int>(ro->top.size()));
            for (int k = 0; k < hdr[5]; ++k) {
                tids[k] = ro->top[size_t(k)].id;
                tlg[k] = ro->top[size_t(k)].logit;
            }
            ++tap_dump_cnt;
            if (hdr[5] > 0 && tids[0] == hdr[3]) ++tap_dump_align_ok;
        }
        std::memcpy(buf, hdr, 24);
        std::memcpy(buf + 24, tids, 128);
        std::memcpy(buf + 152, tlg, 128);
        if (dst) {
            const size_t off = dst->size();
            dst->resize(off + REC_B);
            std::memcpy(dst->data() + off, buf, 280);
            std::memcpy(dst->data() + off + 280, tap_row, TAP_B);
        } else if (tap_dump_fp) {
            std::fwrite(buf, 1, 280, tap_dump_fp);
            std::fwrite(tap_row, 1, TAP_B, tap_dump_fp);
        }
    }

    // Inline capture for contiguous rows [pos0..pos0+n) of one prompt (single-session or into per-slot buffer).
    void tap_dump_rows(int pos0, int n, const int* toks_in, const std::vector<RowOut>* outs,
                       int l48_tok1 = -1, std::vector<uint8_t>* dst = nullptr) {
        if ((!tap_dump_fp && !dst) || !dfl_step_taps || n <= 0 || n > dfl_step_taps_cap) return;
        const size_t n_h = size_t(n) * qw38::dfl::NT * geom::HID;
        QW38_CK(cudaMemcpyAsync(h_tap_dump.data(), dfl_step_taps, n_h * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        QW38_CK(cudaStreamSynchronize(stream));
        for (int m = 0; m < n; ++m) {
            const RowOut* ro = (outs && m < static_cast<int>(outs->size())) ? &(*outs)[size_t(m)] : nullptr;
            const int32_t l48 = (m == n - 1) ? l48_tok1 : -1;
            tap_emit_one(dst, pos0 + m, toks_in[m], ro, l48,
                         h_tap_dump.data() + size_t(m) * qw38::dfl::NT * geom::HID);
        }
    }

    // High-throughput batched S-session decode capture: dumps all B rows in ONE D2H copy into per-slot buffers!
    void tap_dump_batch_rows(int B, const int* pos_arr, const int* tok_arr, const std::vector<RowOut>& outs,
                             std::vector<std::vector<uint8_t>>& slot_bufs) {
        if (!dfl_step_taps || B <= 0 || B > dfl_step_taps_cap) return;
        const size_t n_h = size_t(B) * qw38::dfl::NT * geom::HID;
        QW38_CK(cudaMemcpyAsync(h_tap_dump.data(), dfl_step_taps, n_h * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        QW38_CK(cudaStreamSynchronize(stream));
        for (int s = 0; s < B; ++s) {
            const RowOut* ro = (s < static_cast<int>(outs.size())) ? &outs[size_t(s)] : nullptr;
            tap_emit_one(&slot_bufs[size_t(s)], pos_arr[s], tok_arr[s], ro, -1,
                         h_tap_dump.data() + size_t(s) * qw38::dfl::NT * geom::HID);
        }
    }

    // Run one QSpine-DF2 draft step for solo session (q0 = pos of anchor_tok, drafts K_eff in 1..K tokens).
    void dfl_draft_step(int anchor_tok, int q0, int K, qw38::dfl::DraftResult& res, int dp_override = -1) {
        if (o.profile) cudaStreamSynchronize(stream);
        const auto t0 = std::chrono::steady_clock::now();
        dfl.forward_block(embed, anchor_tok, q0, stream, o.use_graph);
        if (o.profile) cudaStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.nloc; ++i) {
            const qw38::MtpHead2* h2p = (i < static_cast<int>(mtp_head2.size()) && mtp_head2[size_t(i)].ready())
                                            ? &mtp_head2[size_t(i)] : nullptr;
            const qw38::Fp8Mat* fp8p = (i < static_cast<int>(fp8_lm_head.size()) && fp8_lm_head[size_t(i)].q)
                                           ? &fp8_lm_head[size_t(i)] : nullptr;
            dfl.extract_local_top16(i, K, o.trunk_head2 && h2p != nullptr, h2p, fp8p, lm_head[size_t(i)],
                                    v_lo[size_t(i)], v_rows[size_t(i)], head_logits, mtp_h2_sc, stream);
        }
        if (o.nr > 1 && o.nloc == 1 && gather_raw) {
            const size_t rk_b = size_t(qw38::dfl::K_MAX) * qw38::dfl::TOPK * sizeof(qw38::Cand16);
            QW38_CK(cudaMemcpyAsync(dfl_cand_host.data() + size_t(o.rank0) * rk_b,
                                    dfl.d_rank_cands, rk_b, cudaMemcpyDeviceToHost, stream));
            QW38_CK(cudaStreamSynchronize(stream));
            gather_raw(dfl_cand_host.data(), rk_b);
            QW38_CK(cudaMemcpyAsync(dfl.d_rank_cands, dfl_cand_host.data(), size_t(o.nr) * rk_b,
                                    cudaMemcpyHostToDevice, stream));
        }
        if (o.profile) cudaStreamSynchronize(stream);
        const auto t2 = std::chrono::steady_clock::now();
        const int dp_mode = (dp_override >= 0) ? dp_override : o.dfl_dp;
        dfl.select_chain(o.nr, K, anchor_tok, 1.0f, dp_mode, res, stream);
        if (o.profile) {
            const auto t3 = std::chrono::steady_clock::now();
            static int pcnt = 0;
            if (++pcnt <= 5) {
                std::fprintf(stderr, "# dfl_prof#%d: body=%.2fms head=%.2fms walk=%.2fms (Keff=%d)\n",
                             pcnt,
                             std::chrono::duration<double, std::milli>(t1 - t0).count(),
                             std::chrono::duration<double, std::milli>(t2 - t1).count(),
                             std::chrono::duration<double, std::milli>(t3 - t2).count(), res.K);
            }
        }
    }

    // Execute one full 100% lossless QSpine-DF2 speculative round (solo S = 1) using PR #599's read-only GDN commit replay!
    void dfl_round(int K, int& cur_tok, int& pos, float inv_temp, float top_p, uint32_t seed,
                   std::vector<int>& emitted, int& total_drafted, int& total_accepted,
                   std::vector<int>* acc_hist = nullptr) {
        const auto td0 = std::chrono::steady_clock::now();
        qw38::dfl::DraftResult dr{};
        dfl_draft_step(cur_tok, pos, K, dr);
        const auto td1 = std::chrono::steady_clock::now();
        dfl_draft_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();

        const int Keff = dr.K;
        total_drafted += Keff;

        const auto saved_ple_ctx = ple_ctx;
        const int V = Keff + 1;
        std::vector<qw38::StepSeg> segs = {{0, 0, pos, V}};
        std::vector<int> vtoks(static_cast<size_t>(V));
        std::vector<RowSample> vsm(static_cast<size_t>(V));
        vtoks[0] = cur_tok;
        vsm[0] = RowSample{inv_temp, top_p, seed, uint32_t(pos + 1)};
        for (int k = 0; k < Keff; ++k) {
            vtoks[size_t(1 + k)] = dr.tok[k];
            vsm[size_t(1 + k)] = RowSample{inv_temp, top_p, seed, uint32_t(pos + 2 + k)};
        }
        std::vector<RowOut> vout;
        forward(segs, vtoks.data(), vsm.data(), vout, false, Keff);
        const auto td2 = std::chrono::steady_clock::now();
        dfl_verify_ms += std::chrono::duration<double, std::milli>(td2 - td1).count();

        int n_acc = 0;
        for (int k = 0; k < Keff; ++k) {
            if (vout[size_t(k)].tok == dr.tok[k]) ++n_acc;
            else break;
        }
        total_accepted += n_acc;
        if (acc_hist && n_acc < static_cast<int>(acc_hist->size())) (*acc_hist)[size_t(n_acc)] += 1;

        for (int k = 0; k < n_acc; ++k) emitted.push_back(dr.tok[k]);
        const int bonus = vout[size_t(n_acc)].tok;
        emitted.push_back(bonus);

        // PR #599 read-only GDN verify -> ALWAYS commit exact (n_acc + 1) accepted tokens into gdn_st.S!
        if (n_acc < Keff) {
            mtp_rollback_slot(0, n_acc + 1);
            ple_ctx[0] = saved_ple_ctx[0];
            const int n_comm = n_acc + 1;
            if (n_comm >= 2) {
                ple_ctx[0][0] = vtoks[size_t(n_comm - 2)];
                ple_ctx[0][1] = vtoks[size_t(n_comm - 1)];
            } else {
                ple_ctx[0][0] = ple_ctx[0][1];
                ple_ctx[0][1] = vtoks[0];
            }
        }
        const auto tc0 = std::chrono::steady_clock::now();
        mtp_commit_gdn_batch({0}, {n_acc + 1});
        if (o.profile) cudaStreamSynchronize(stream);
        const auto tc1 = std::chrono::steady_clock::now();
        // Ingest ONLY the (n_acc + 1) newly committed target tap rows [pos .. pos + n_acc] into DF2 FP16 KV ring!
        dfl.ingest_context(dfl_step_taps, n_acc + 1, pos, stream);
        if (o.profile) {
            cudaStreamSynchronize(stream);
            const auto tc2 = std::chrono::steady_clock::now();
            static int ppost = 0;
            if (++ppost <= 5) {
                std::fprintf(stderr, "# dfl_post#%d: gdn_commit=%.2fms ingest=%.2fms (n_comm=%d)\n",
                             ppost,
                             std::chrono::duration<double, std::milli>(tc1 - tc0).count(),
                             std::chrono::duration<double, std::milli>(tc2 - tc1).count(), n_acc + 1);
            }
        }
        cur_tok = bonus;
        pos += n_acc + 1;
    }
};

} // namespace qw38rt
