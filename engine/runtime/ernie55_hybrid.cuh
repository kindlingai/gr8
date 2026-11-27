// Single-GPU Hybrid VRAM Hot-Expert Cache + Pageable Host RAM Offload + Bounded Pinned H2D Staging
// for ERNIE-5.5-Flash-Preview (`Ernie5_5_MoeForCausalLM`, TP=1, GR8_ERNIE55_HYBRID=1).
//
// Target: 1x RTX 5090 (32 GB VRAM) + 128 GB System RAM (and compatible with any single Blackwell GPU).
// Design:
//   - VRAM permanent core (part=3, ~3.11 GiB) + MXFP8 dense linears (from part=2, ~5.94 GiB) + KV cache live in VRAM.
//   - `embed.weight` (1.65 GiB) + routed `L001..L042.tr3.{fc1,fc2}` (part=4, ~33..63 GiB) live in pageable host RAM
//     (zero whole-model pinning; only small bounded H2D bounce buffers are `cudaHostAlloc`-pinned).
//   - Routed experts are wide-reordered in-place on CPU into exact K2/K3/K4 wide-load layout matching ernie55_moe.cuh.
//   - Every MoE layer owns a fixed-address device piece table `d_ptab[512][eb][2]`.
//   - Free VRAM (minus GR8_ERNIE55_VRAM_RESERVE_MIB) is carved into an optional STRP v1 tight-packed static core plus
//     a uniform dynamic VRAM hot-expert cache pool with ring slots for async H2D heat-based promotions.
//   - Each MoE layer splits routed groups into VRAM-hot groups (executed immediately on compute stream) + double-buffered
//     cold chunks (copied host -> pinned bounce -> VRAM staging on h2d_stream, then executed with exact same GPU kernels).
//     Because `fc2_part_r[route * EH]` is disjoint across routes, output is bit-exact vs 100% GPU-resident mode!
//   - Optional `GR8_ERNIE55_HYBRID_CPU_OVERLAP=1` evaluates overflow M=1 cold experts on host CPU while GPU runs hot experts.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "../model_geom_ernie55.h"
#include "../knobs-rs/gr8_knobs.h"
#include "../kernels/ernie55_moe.cuh"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"

namespace ernie55rt {

using ernie55::u32;
using ernie55::u64;

[[noreturn]] inline void hybrid_die(const std::string& m) {
    std::fprintf(stderr, "ernie55 hybrid: %s\n", m.c_str());
    std::exit(2);
}

inline void hybrid_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) hybrid_die(std::string(what) + ": " + cudaGetErrorString(e));
}

// =================================================================================================
// Host CPU Wide-Order Reorder (K2, K3, K4 — bit-exact match with k_ernie55_tr3_widen_fc1/fc2!)
// =================================================================================================
template <int HB>
inline void widen_fc1_mat_host(uint8_t* mat, int bits) {
    if (bits == 2) {
        constexpr int GRP_B = 2048 * 2;
        uint8_t tmp[GRP_B];
        for (int hb = 0; hb < HB; ++hb) {
            uint8_t* grp = mat + static_cast<size_t>(hb) * GRP_B;
            for (int j = 0; j < 8; ++j) {
                for (int lane = 0; lane < 32; ++lane) {
                    u64 v[3] = {0, 0, 0};
                    for (int p = 0; p < 8; ++p)
                        ernie55::trw_pack<2>(v, p, ernie55::trw_tile_chunk<2>(grp + (p * 8 + j) * 64, lane));
                    for (int n = 0; n < 2; ++n)
                        std::memcpy(tmp + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
                }
            }
            std::memcpy(grp, tmp, GRP_B);
        }
    } else if (bits == 3) {
        constexpr int GRP_B = 2048 * 3;
        uint8_t tmp[GRP_B];
        for (int hb = 0; hb < HB; ++hb) {
            uint8_t* grp = mat + static_cast<size_t>(hb) * GRP_B;
            for (int j = 0; j < 8; ++j) {
                for (int lane = 0; lane < 32; ++lane) {
                    u64 v[3] = {0, 0, 0};
                    for (int p = 0; p < 8; ++p)
                        ernie55::trw_pack<3>(v, p, ernie55::trw_tile_chunk<3>(grp + (p * 8 + j) * 96, lane));
                    for (int n = 0; n < 3; ++n)
                        std::memcpy(tmp + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
                }
            }
            std::memcpy(grp, tmp, GRP_B);
        }
    } else if (bits == 4) {
        constexpr int GRP_B = 2048 * 4;
        uint8_t tmp[GRP_B];
        for (int hb = 0; hb < HB; ++hb) {
            uint8_t* grp = mat + static_cast<size_t>(hb) * GRP_B;
            for (int j = 0; j < 8; ++j) {
                for (int lane = 0; lane < 32; ++lane) {
                    u32 w[8];
                    for (int p = 0; p < 8; ++p)
                        std::memcpy(&w[p], grp + (p * 8 + j) * 128 + lane * 4, 4);
                    std::memcpy(tmp + j * 1024 + lane * 16,       &w[0], 16);
                    std::memcpy(tmp + j * 1024 + 512 + lane * 16, &w[4], 16);
                }
            }
            std::memcpy(grp, tmp, GRP_B);
        }
    }
}

template <int HB>
inline void widen_fc2_blk_host(uint8_t* blk, int bits) {
    constexpr int N_TILES = HB * 8;
    if (bits == 2) {
        constexpr int NT_B = 256 * 2;
        uint8_t tmp[NT_B];
        for (int nt = 0; nt < N_TILES; ++nt) {
            uint8_t* dst = blk + static_cast<size_t>(nt) * NT_B;
            for (int lane = 0; lane < 32; ++lane) {
                u64 v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p)
                    ernie55::trw_pack<2>(v, p, ernie55::trw_tile_chunk<2>(dst + p * 64, lane));
                for (int n = 0; n < 2; ++n)
                    std::memcpy(tmp + n * 256 + lane * 8, &v[n], 8);
            }
            std::memcpy(dst, tmp, NT_B);
        }
    } else if (bits == 3) {
        constexpr int NT_B = 256 * 3;
        uint8_t tmp[NT_B];
        for (int nt = 0; nt < N_TILES; ++nt) {
            uint8_t* dst = blk + static_cast<size_t>(nt) * NT_B;
            for (int lane = 0; lane < 32; ++lane) {
                u64 v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p)
                    ernie55::trw_pack<3>(v, p, ernie55::trw_tile_chunk<3>(dst + p * 96, lane));
                for (int n = 0; n < 3; ++n)
                    std::memcpy(tmp + n * 256 + lane * 8, &v[n], 8);
            }
            std::memcpy(dst, tmp, NT_B);
        }
    } else if (bits == 4) {
        constexpr int NT_B = 256 * 4;
        uint8_t tmp[NT_B];
        for (int nt = 0; nt < N_TILES; ++nt) {
            uint8_t* dst = blk + static_cast<size_t>(nt) * NT_B;
            for (int lane = 0; lane < 32; ++lane) {
                u32 w[8];
                for (int p = 0; p < 8; ++p)
                    std::memcpy(&w[p], dst + p * 128 + lane * 4, 4);
                std::memcpy(tmp + lane * 16,       &w[0], 16);
                std::memcpy(tmp + 512 + lane * 16, &w[4], 16);
            }
            std::memcpy(dst, tmp, NT_B);
        }
    }
}

inline int fc1_unit_width_host(long long km, int dk, int unit) {
    if (km == 0LL) return dk;
    if (km < 0LL) return static_cast<int>((km >> (4 * (unit & 1))) & 0xfull);
    return static_cast<int>((km >> (4 * unit)) & 0xfull);
}

inline void widen_routed_layer_host(uint8_t* fc1, uint8_t* fc2, const long long* tab,
                                    const long long* kmix, int ne) {
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(geom::EHB) * 2048; // 32768 B
    for (int e = 0; e < ne; ++e) {
        const long long* tr = tab + static_cast<size_t>(e) * 6;
        const int nb = static_cast<int>(tr[3]);
        const int dk = static_cast<int>(tr[4]);
        const long long km = kmix ? kmix[e] : 0LL;
        size_t off1 = static_cast<size_t>(tr[0]);
        for (int u = 0; u < 2 * nb; ++u) {
            const int k = fc1_unit_width_host(km, dk, u);
            widen_fc1_mat_host<geom::EHB>(fc1 + off1, k);
            off1 += static_cast<size_t>(k) * UNIT_PER_BIT;
        }
        size_t off2 = static_cast<size_t>(tr[1]);
        for (int b = 0; b < nb; ++b) {
            widen_fc2_blk_host<geom::EHB>(fc2 + off2, dk);
            off2 += static_cast<size_t>(dk) * UNIT_PER_BIT;
        }
    }
}

// =================================================================================================
// Atomic device piece-table patch kernel
// =================================================================================================
struct PtabPatchEntry {
    const uint8_t** dst_slot;
    const uint8_t*  ptr;
};

__global__ void __launch_bounds__(256) k_ernie55_ptab_patch(
    const PtabPatchEntry* __restrict__ patches, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *patches[i].dst_slot = patches[i].ptr;
}

// =================================================================================================
// Optional STRP v1 routing-frequency profile reader
// =================================================================================================
inline bool read_strp_profile(const std::string& path, int n_layers, int n_experts,
                              std::vector<std::pair<int, int>>& ranked) {
    ranked.clear();
    if (path.empty()) return false;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4] = {};
    uint32_t hdr[5] = {};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0 || hdr[0] != 1 ||
        static_cast<int>(hdr[1]) != n_layers || static_cast<int>(hdr[2]) != n_experts) {
        std::fclose(f);
        return false;
    }
    const uint32_t n_ranked = hdr[4];
    std::vector<uint16_t> raw(static_cast<size_t>(n_ranked) * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, raw.size(), f) != raw.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    ranked.reserve(n_ranked);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int l = static_cast<int>(raw[static_cast<size_t>(i) * 2]);
        const int e = static_cast<int>(raw[static_cast<size_t>(i) * 2 + 1]);
        if (l >= 0 && l < n_layers && e >= 0 && e < n_experts) ranked.emplace_back(l, e);
    }
    return !ranked.empty();
}

// =================================================================================================
// Host Pageable Backing for part=4 (`embed.weight` + `L001..L042.tr3.{fc1,fc2}`)
// =================================================================================================
struct HybridHostBacking {
    g6::Pack pack;
    uint8_t* base = nullptr;
    size_t   bytes = 0;
    int      backing_fd = -1;

    struct Sink {
        uint8_t* base = nullptr;
        static void* alloc(void*, size_t n) { return std::malloc(n); }
        static void  release(void*, void* p) { std::free(p); }
        static int   copy(void* ctx, int /*w*/, uint64_t off, const void* src, size_t n) {
            auto* s = static_cast<Sink*>(ctx);
            std::memcpy(s->base + off, src, n);
            return 0;
        }
    };

    ~HybridHostBacking() { release(); }

    void release() {
        if (base) {
            munmap(base, bytes);
            base = nullptr;
            bytes = 0;
        }
        if (backing_fd >= 0) {
            close(backing_fd);
            backing_fd = -1;
        }
    }

    void load_part4(const std::string& release_dir, const std::string& widths_path, int nr, int r) {
        char mf[256];
        std::snprintf(mf, sizeof(mf), "/tmp/ernie55-p4-p%d-r%d.json", static_cast<int>(getpid()), r);
        char err[512] = {};
        void* plan = g6l_ernie55_plan_part(release_dir.c_str(), widths_path.c_str(),
                                           static_cast<uint32_t>(nr), static_cast<uint32_t>(r),
                                           4u, mf, err, sizeof(err));
        if (!plan) hybrid_die(std::string("g6l_ernie55_plan_part(part=4): ") + err);
        std::string perr;
        const bool ok = pack.open_manifest(mf, perr);
        ::unlink(mf);
        if (!ok) {
            g6l_plan_free(plan);
            hybrid_die("open_manifest(part=4): " + perr);
        }
        bytes = pack.bytes();

        // Allocate reclaimable pageable host RAM (zero mlock / zero cudaHostAlloc pinning!).
        void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem == MAP_FAILED) {
            g6l_plan_free(plan);
            hybrid_die("mmap pageable host RAM (" + std::to_string(bytes >> 20) + " MiB) failed");
        }
        base = static_cast<uint8_t*>(mem);
#ifdef MADV_HUGEPAGE
        madvise(base, bytes, MADV_HUGEPAGE);
#endif
        Sink sink{base};
        const g6l_sink csink{&sink, Sink::alloc, Sink::release, Sink::copy};
        g6l_stats st{};
        const int rc = g6l_plan_load(plan, &csink, &st, err, sizeof(err));
        g6l_plan_free(plan);
        if (rc != 0) hybrid_die(std::string("g6l_plan_load(part=4): ") + err);

        std::fprintf(stderr, "# ernie55 hybrid: loaded pageable host part=4 %.2f GiB in %.2f s (%u threads)\n",
                     st.image_bytes / double(1ull << 30), st.seconds, st.threads);
    }

    uint8_t* get_mut(const std::string& name) const {
        const g6::Entry* e = pack.find(name);
        if (!e) hybrid_die("part=4 missing tensor: " + name);
        return base + e->offset;
    }

    const uint8_t* get(const std::string& name) const { return get_mut(name); }
};

// =================================================================================================
// Hybrid Configuration
// =================================================================================================
struct HybridConfig {
    bool        enabled = false;
    int64_t     vram_reserve_mib = 1024;
    int64_t     cache_slots_override = -1;
    std::string profile_path;
    bool        adaptive = true;
    int         promote_max = 8;
    bool        cpu_overlap = false;
    float       promote_hyst = 1.15f;
    float       heat_decay = 0.97f;
    float       heat_cap = 16.0f;
    float       min_promote_heat = 2.5f;
    float       static_pin_frac = 0.45f;
    int         stage_slots = 16; // experts per cold staging chunk
};

inline HybridConfig load_hybrid_config() {
    HybridConfig c;
    c.enabled = knob::ernie55_hybrid();
    c.vram_reserve_mib = knob::ernie55_vram_reserve_mib();
    c.cache_slots_override = knob::ernie55_expert_cache_slots();
    if (const auto& p = knob::ernie55_expert_profile(); p && !p->empty()) c.profile_path = *p;
    c.adaptive = knob::ernie55_hybrid_adaptive();
    c.promote_max = std::max(0, static_cast<int>(knob::ernie55_hybrid_promote_max()));
    c.cpu_overlap = knob::ernie55_hybrid_cpu_overlap();
    return c;
}

// =================================================================================================
// Hybrid Expert Cache + Staged Cold Execution Engine
// =================================================================================================
struct HybridExpertCache {
    static constexpr int NE = geom::NEXP;      // 512
    static constexpr int EB = geom::EXP_BLOCKS; // 16 at TP=1
    static constexpr size_t ALIGN256 = 256;
    static size_t round256(size_t n) { return (n + ALIGN256 - 1) & ~(ALIGN256 - 1); }

    struct ExpMeta {
        size_t fc1_off = 0, fc1_bytes = 0;
        size_t fc2_off = 0, fc2_bytes = 0;
        int    nb = 0;
        size_t piece_fc1_off[EB] = {};
        size_t piece_fc2_off[EB] = {};
    };

    struct LayerData {
        int             layer_id = -1; // 1..42
        const uint8_t*  h_fc1 = nullptr;
        const uint8_t*  h_fc2 = nullptr;
        const uint8_t** d_ptab = nullptr; // [NE * EB * 2] in VRAM
        std::vector<const uint8_t*> h_ptab;
        std::vector<ExpMeta> exp;         // [NE]
        std::vector<long long> h_tab;     // [NE * 6]
        std::vector<long long> h_kmix;    // [NE] (empty if absent)
    };

    HybridConfig cfg;
    bool active = false;
    int  n_layers = 0; // 42 (L1..L42)
    std::vector<LayerData> layers;
    const __nv_bfloat16* host_embed = nullptr; // [VOCAB, HID] in pageable host RAM

    // Pinned host->device embedding row transfer scratch
    __nv_bfloat16* h_embed_rows = nullptr;
    __nv_bfloat16* d_embed_rows = nullptr;
    int max_embed_rows = 0;

    // VRAM hot-expert pools (static tight-packed + dynamic uniform slots)
    uint8_t* vram_static_pool = nullptr;
    size_t   vram_static_bytes = 0;
    int      pinned_count = 0;

    uint8_t* vram_dyn_pool = nullptr;
    size_t   dyn_slot_stride = 0;
    int      dyn_total_slots = 0;
    int      dyn_active_slots = 0;
    int      ring_slots = 0;

    // Double-buffered bounded staging for cold experts (decode & prefill)
    size_t   stg_slot_stride = 0;
    int      stg_slots = 16;
    uint8_t* stg_vram[2]    = {nullptr, nullptr}; // [stg_slots * stg_slot_stride] in VRAM
    uint8_t* stg_bounce[2]  = {nullptr, nullptr}; // [stg_slots * stg_slot_stride] pinned host
    const uint8_t** stg_d_ptab[2] = {nullptr, nullptr}; // [NE * EB * 2] in VRAM
    const uint8_t** stg_h_ptab[2] = {nullptr, nullptr}; // [NE * EB * 2] pinned host
    cudaEvent_t stg_h2d_ev[2]  = {nullptr, nullptr};
    cudaEvent_t stg_comp_ev[2] = {nullptr, nullptr};

    // Host/device group buffers for partitioned hot + cold chunk execution
    int* h_topk_idx = nullptr; // pinned [max_M * 10]
    int  max_routes_cap = 0;
    int* h_grp_n[2]      = {nullptr, nullptr};
    int* h_grp_exp[2]    = {nullptr, nullptr};
    int* h_grp_cnt[2]    = {nullptr, nullptr};
    int* h_grp_routes[2] = {nullptr, nullptr};
    int* d_grp_n[2]      = {nullptr, nullptr};
    int* d_grp_exp[2]    = {nullptr, nullptr};
    int* d_grp_cnt[2]    = {nullptr, nullptr};
    int* d_grp_routes[2] = {nullptr, nullptr};
    int  max_grp_cap = 0;

    // Residency, heat, and async promotion state [n_layers * NE]
    // exp_slot[idx]: -2 = static pinned, -1 = cold (host RAM), >= 0 = dynamic slot
    std::vector<int>     exp_slot;
    std::vector<int>     slot_owner;
    std::vector<uint8_t> exp_pending;
    std::vector<float>   heat;
    std::vector<int>     free_slots;

    uint32_t*       d_exp_hits = nullptr;
    uint32_t*       h_exp_hits = nullptr;
    uint8_t*        promo_bounce = nullptr; // pinned [promote_max * dyn_slot_stride]
    PtabPatchEntry* d_patches = nullptr;
    PtabPatchEntry* h_patches = nullptr;
    int             max_patches = 0;

    cudaStream_t h2d_stream = nullptr;
    cudaEvent_t  promo_ev = nullptr;
    cudaEvent_t  patch_ev = nullptr;
    bool         promo_in_flight = false;
    bool         patch_ev_recorded = false;

    struct PendingSwap { int new_idx, old_idx, stg_slot, old_slot; };
    std::vector<PendingSwap> pending_swaps;

    // Telemetry
    uint64_t stat_hits = 0;
    uint64_t stat_misses = 0;
    uint64_t stat_swaps = 0;
    uint64_t stat_cold_staged = 0;
    uint64_t stat_steps = 0;

    ~HybridExpertCache() {
        if (active && (stat_steps > 0 || stat_cold_staged > 0)) print_stats();
        release();
    }

    void release() {
        if (h2d_stream) { cudaStreamSynchronize(h2d_stream); cudaStreamDestroy(h2d_stream); h2d_stream = nullptr; }
        if (promo_ev) { cudaEventDestroy(promo_ev); promo_ev = nullptr; }
        if (patch_ev) { cudaEventDestroy(patch_ev); patch_ev = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (stg_h2d_ev[b])  { cudaEventDestroy(stg_h2d_ev[b]); stg_h2d_ev[b] = nullptr; }
            if (stg_comp_ev[b]) { cudaEventDestroy(stg_comp_ev[b]); stg_comp_ev[b] = nullptr; }
            if (stg_vram[b])    { cudaFree(stg_vram[b]); stg_vram[b] = nullptr; }
            if (stg_bounce[b])  { cudaFreeHost(stg_bounce[b]); stg_bounce[b] = nullptr; }
            if (stg_d_ptab[b])  { cudaFree(stg_d_ptab[b]); stg_d_ptab[b] = nullptr; }
            if (stg_h_ptab[b])  { cudaFreeHost(stg_h_ptab[b]); stg_h_ptab[b] = nullptr; }
            if (h_grp_n[b])     { cudaFreeHost(h_grp_n[b]); h_grp_n[b] = nullptr; }
            if (h_grp_exp[b])   { cudaFreeHost(h_grp_exp[b]); h_grp_exp[b] = nullptr; }
            if (h_grp_cnt[b])   { cudaFreeHost(h_grp_cnt[b]); h_grp_cnt[b] = nullptr; }
            if (h_grp_routes[b]){ cudaFreeHost(h_grp_routes[b]); h_grp_routes[b] = nullptr; }
            if (d_grp_n[b])     { cudaFree(d_grp_n[b]); d_grp_n[b] = nullptr; }
            if (d_grp_exp[b])   { cudaFree(d_grp_exp[b]); d_grp_exp[b] = nullptr; }
            if (d_grp_cnt[b])   { cudaFree(d_grp_cnt[b]); d_grp_cnt[b] = nullptr; }
            if (d_grp_routes[b]){ cudaFree(d_grp_routes[b]); d_grp_routes[b] = nullptr; }
        }
        if (h_topk_idx)   { cudaFreeHost(h_topk_idx); h_topk_idx = nullptr; }
        if (h_embed_rows) { cudaFreeHost(h_embed_rows); h_embed_rows = nullptr; }
        if (d_embed_rows) { cudaFree(d_embed_rows); d_embed_rows = nullptr; }
        if (vram_static_pool) { cudaFree(vram_static_pool); vram_static_pool = nullptr; }
        if (vram_dyn_pool)    { cudaFree(vram_dyn_pool); vram_dyn_pool = nullptr; }
        if (promo_bounce)     { cudaFreeHost(promo_bounce); promo_bounce = nullptr; }
        if (d_exp_hits)       { cudaFree(d_exp_hits); d_exp_hits = nullptr; }
        if (h_exp_hits)       { cudaFreeHost(h_exp_hits); h_exp_hits = nullptr; }
        if (d_patches)        { cudaFree(d_patches); d_patches = nullptr; }
        if (h_patches)        { cudaFreeHost(h_patches); h_patches = nullptr; }
        for (auto& ld : layers) {
            if (ld.d_ptab) { cudaFree(ld.d_ptab); ld.d_ptab = nullptr; }
        }
        layers.clear();
        active = false;
    }

    uint32_t* layer_hits(int moe_li) const {
        return (active && d_exp_hits && moe_li >= 0 && moe_li < n_layers)
            ? (d_exp_hits + static_cast<size_t>(moe_li) * NE) : nullptr;
    }

    const uint8_t* const* layer_ptab(int moe_li) const {
        return (active && moe_li >= 0 && moe_li < n_layers) ? layers[static_cast<size_t>(moe_li)].d_ptab : nullptr;
    }

    void set_expert_ptab_host(int li, int e, const uint8_t* fc1_base, const uint8_t* fc2_base) {
        auto& ld = layers[static_cast<size_t>(li)];
        const auto& em = ld.exp[static_cast<size_t>(e)];
        for (int b = 0; b < em.nb; ++b) {
            const int sl = (e * EB + b) * 2;
            ld.h_ptab[static_cast<size_t>(sl)]     = fc1_base + em.piece_fc1_off[b];
            ld.h_ptab[static_cast<size_t>(sl) + 1] = fc2_base + em.piece_fc2_off[b];
        }
    }

    // Gather `M` token embeddings from pageable host RAM into pinned buffer and async H2D -> expand 4 streams!
    void embed_expand_host(const int* h_toks, int M, __nv_bfloat16* d_hyper, cudaStream_t st) const {
        if (!host_embed || M > max_embed_rows) hybrid_die("embed_expand_host: bad M or missing host_embed");
        for (int m = 0; m < M; ++m) {
            const int tok = h_toks[m];
            if (tok < 0 || tok >= geom::VOCAB) hybrid_die("embed_expand_host: token out of range");
            std::memcpy(h_embed_rows + static_cast<size_t>(m) * geom::HID,
                        host_embed + static_cast<size_t>(tok) * geom::HID,
                        static_cast<size_t>(geom::HID) * sizeof(__nv_bfloat16));
        }
        hybrid_ck(cudaMemcpyAsync(d_embed_rows, h_embed_rows,
                                  static_cast<size_t>(M) * geom::HID * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice, st), "embed H2D");
        ernie55::k_ernie55_expand_from_rows<<<M * 16, 256, 0, st>>>(d_embed_rows, d_hyper, M);
    }

    // Phase 1 (before KV pool allocation): bind host backing, CPU-widen all 42 routed layers, allocate d_ptab & hits.
    void init_tables(const HybridConfig& c, const HybridHostBacking& hb,
                     const std::vector<std::vector<long long>>& tabs,
                     const std::vector<std::vector<long long>>& kmixes,
                     int max_M, cudaStream_t st) {
        cfg = c;
        active = true;
        n_layers = geom::N_MOE_LAYERS; // 42 (L1..L42)
        layers.resize(static_cast<size_t>(n_layers));
        host_embed = reinterpret_cast<const __nv_bfloat16*>(hb.get("embed.weight"));

        max_embed_rows = max_M;
        hybrid_ck(cudaHostAlloc(&h_embed_rows, static_cast<size_t>(max_M) * geom::HID * sizeof(__nv_bfloat16), cudaHostAllocDefault), "h_embed_rows");
        hybrid_ck(cudaMalloc(&d_embed_rows, static_cast<size_t>(max_M) * geom::HID * sizeof(__nv_bfloat16)), "d_embed_rows");

        stg_slot_stride = 0;
        constexpr size_t UNIT_PER_BIT = static_cast<size_t>(geom::EHB) * 2048;

        // Bind per-layer host pointers & metadata
        std::vector<uint8_t*> fc1_mut(static_cast<size_t>(n_layers), nullptr);
        std::vector<uint8_t*> fc2_mut(static_cast<size_t>(n_layers), nullptr);
        for (int li = 0; li < n_layers; ++li) {
            const int l = geom::FIRST_MOE + li;
            char nm1[64], nm2[64];
            std::snprintf(nm1, sizeof(nm1), "L%03d.tr3.fc1", l);
            std::snprintf(nm2, sizeof(nm2), "L%03d.tr3.fc2", l);
            auto& ld = layers[static_cast<size_t>(li)];
            ld.layer_id = l;
            fc1_mut[static_cast<size_t>(li)] = hb.get_mut(nm1);
            fc2_mut[static_cast<size_t>(li)] = hb.get_mut(nm2);
            ld.h_fc1  = fc1_mut[static_cast<size_t>(li)];
            ld.h_fc2  = fc2_mut[static_cast<size_t>(li)];
            ld.h_tab  = tabs[static_cast<size_t>(li)];
            ld.h_kmix = kmixes[static_cast<size_t>(li)];
            ld.h_ptab.assign(static_cast<size_t>(NE) * EB * 2, nullptr);
            ld.exp.resize(static_cast<size_t>(NE));

            for (int e = 0; e < NE; ++e) {
                const long long* tr = &ld.h_tab[static_cast<size_t>(e) * 6];
                const long long km = ld.h_kmix.empty() ? 0LL : ld.h_kmix[static_cast<size_t>(e)];
                auto& em = ld.exp[static_cast<size_t>(e)];
                em.nb = static_cast<int>(tr[3]);
                em.fc1_off = static_cast<size_t>(tr[0]);
                em.fc2_off = static_cast<size_t>(tr[1]);
                const int dk = static_cast<int>(tr[4]);
                size_t f1_cur = 0, f2_cur = 0;
                for (int b = 0; b < em.nb; ++b) {
                    em.piece_fc1_off[b] = f1_cur;
                    em.piece_fc2_off[b] = f2_cur;
                    const int kg = fc1_unit_width_host(km, dk, 2 * b + 0);
                    const int ku = fc1_unit_width_host(km, dk, 2 * b + 1);
                    f1_cur += static_cast<size_t>(kg + ku) * UNIT_PER_BIT;
                    f2_cur += static_cast<size_t>(dk) * UNIT_PER_BIT;
                }
                em.fc1_bytes = f1_cur;
                em.fc2_bytes = f2_cur;
                const size_t tot_al = round256(em.fc1_bytes) + round256(em.fc2_bytes);
                if (tot_al > stg_slot_stride) stg_slot_stride = tot_al;
            }
            hybrid_ck(cudaMalloc(&ld.d_ptab, ld.h_ptab.size() * sizeof(void*)), "ld.d_ptab");
            hybrid_ck(cudaMemsetAsync(ld.d_ptab, 0, ld.h_ptab.size() * sizeof(void*), st), "ld.d_ptab zero");
        }

        // Parallel CPU wide-reorder across all 42 MoE layers (K2, K3, AND K4!)
        const auto wt0 = std::chrono::steady_clock::now();
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned nt = std::min<unsigned>(std::min<unsigned>(hw, 16u), static_cast<unsigned>(n_layers));
        std::atomic<int> next_li{0};
        std::vector<std::thread> workers;
        workers.reserve(nt);
        for (unsigned t = 0; t < nt; ++t) {
            workers.emplace_back([&]() {
                for (;;) {
                    const int li = next_li.fetch_add(1, std::memory_order_relaxed);
                    if (li >= n_layers) break;
                    auto& ld = layers[static_cast<size_t>(li)];
                    widen_routed_layer_host(fc1_mut[static_cast<size_t>(li)], fc2_mut[static_cast<size_t>(li)],
                                            ld.h_tab.data(), ld.h_kmix.empty() ? nullptr : ld.h_kmix.data(), NE);
                }
            });
        }
        for (auto& th : workers) th.join();
        const double wide_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wt0).count();
        std::fprintf(stderr, "# ernie55 hybrid: CPU wide-reordered %d MoE layers (K2/K3/K4) in %.2f s (%u threads)\n",
                     n_layers, wide_s, nt);

        const size_t total_exps = static_cast<size_t>(n_layers) * NE;
        hybrid_ck(cudaMalloc(&d_exp_hits, total_exps * sizeof(uint32_t)), "d_exp_hits");
        hybrid_ck(cudaHostAlloc(&h_exp_hits, total_exps * sizeof(uint32_t), cudaHostAllocDefault), "h_exp_hits");
        hybrid_ck(cudaMemsetAsync(d_exp_hits, 0, total_exps * sizeof(uint32_t), st), "d_exp_hits zero");
        std::memset(h_exp_hits, 0, total_exps * sizeof(uint32_t));
        hybrid_ck(cudaStreamSynchronize(st), "hybrid init_tables sync");
    }

    // Phase 2 (AFTER KV + scratch allocations): allocate staging buffers and carve remaining VRAM into hot-expert pool!
    void init_vram_pool(int max_M, cudaStream_t main_st) {
        if (!active || n_layers == 0) return;

        hybrid_ck(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking), "h2d_stream");
        hybrid_ck(cudaEventCreateWithFlags(&promo_ev, cudaEventDisableTiming), "promo_ev");
        hybrid_ck(cudaEventCreateWithFlags(&patch_ev, cudaEventDisableTiming), "patch_ev");

        // Allocate double-buffered cold staging scratch + pinned bounce buffers BEFORE sizing hot pool
        stg_slots = std::max(4, cfg.stage_slots);
        const size_t stg_bytes = static_cast<size_t>(stg_slots) * stg_slot_stride;
        const size_t ptab_n = static_cast<size_t>(NE) * EB * 2;
        max_routes_cap = max_M * geom::TOPK;
        max_grp_cap = std::max(NE + max_routes_cap, 1024);

        hybrid_ck(cudaHostAlloc(&h_topk_idx, static_cast<size_t>(max_routes_cap) * sizeof(int), cudaHostAllocDefault), "h_topk_idx");
        for (int b = 0; b < 2; ++b) {
            hybrid_ck(cudaEventCreateWithFlags(&stg_h2d_ev[b], cudaEventDisableTiming), "stg_h2d_ev");
            hybrid_ck(cudaEventCreateWithFlags(&stg_comp_ev[b], cudaEventDisableTiming), "stg_comp_ev");
            hybrid_ck(cudaMalloc(&stg_vram[b], stg_bytes), "stg_vram");
            hybrid_ck(cudaHostAlloc(&stg_bounce[b], stg_bytes, cudaHostAllocDefault), "stg_bounce");
            hybrid_ck(cudaMalloc(&stg_d_ptab[b], ptab_n * sizeof(void*)), "stg_d_ptab");
            hybrid_ck(cudaHostAlloc(&stg_h_ptab[b], ptab_n * sizeof(void*), cudaHostAllocDefault), "stg_h_ptab");
            hybrid_ck(cudaHostAlloc(&h_grp_n[b], sizeof(int), cudaHostAllocDefault), "h_grp_n");
            hybrid_ck(cudaHostAlloc(&h_grp_exp[b], static_cast<size_t>(max_grp_cap) * sizeof(int), cudaHostAllocDefault), "h_grp_exp");
            hybrid_ck(cudaHostAlloc(&h_grp_cnt[b], static_cast<size_t>(max_grp_cap) * sizeof(int), cudaHostAllocDefault), "h_grp_cnt");
            hybrid_ck(cudaHostAlloc(&h_grp_routes[b], static_cast<size_t>(max_grp_cap) * 16 * sizeof(int), cudaHostAllocDefault), "h_grp_routes");
            hybrid_ck(cudaMalloc(&d_grp_n[b], sizeof(int)), "d_grp_n");
            hybrid_ck(cudaMalloc(&d_grp_exp[b], static_cast<size_t>(max_grp_cap) * sizeof(int)), "d_grp_exp");
            hybrid_ck(cudaMalloc(&d_grp_cnt[b], static_cast<size_t>(max_grp_cap) * sizeof(int)), "d_grp_cnt");
            hybrid_ck(cudaMalloc(&d_grp_routes[b], static_cast<size_t>(max_grp_cap) * 16 * sizeof(int)), "d_grp_routes");
        }

        const int max_swaps = std::max(cfg.promote_max, 16);
        max_patches = std::max(256, max_swaps * 2 * EB * 2 + 128);
        hybrid_ck(cudaMalloc(&d_patches, static_cast<size_t>(max_patches) * sizeof(PtabPatchEntry)), "d_patches");
        hybrid_ck(cudaHostAlloc(&h_patches, static_cast<size_t>(max_patches) * sizeof(PtabPatchEntry), cudaHostAllocDefault), "h_patches");

        // Query free VRAM after ALL permanent, MXFP8, KV, and staging allocations
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve_b = static_cast<size_t>(std::max<int64_t>(128, cfg.vram_reserve_mib)) << 20;
        const size_t budget_b = (free_b > reserve_b) ? (free_b - reserve_b) : 0;

        const size_t total_exps = static_cast<size_t>(n_layers) * NE;
        exp_slot.assign(total_exps, -1);
        exp_pending.assign(total_exps, 0);
        heat.assign(total_exps, 0.0f);

        std::vector<std::pair<int, int>> prof_order;
        const bool have_prof = read_strp_profile(cfg.profile_path, geom::L_MAIN, NE, prof_order);

        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(total_exps);
        std::vector<uint8_t> seen(total_exps, 0);
        if (have_prof) {
            for (const auto& le : prof_order) {
                const int li = le.first - geom::FIRST_MOE;
                if (li < 0 || li >= n_layers) continue;
                const size_t idx = static_cast<size_t>(li) * NE + static_cast<size_t>(le.second);
                if (!seen[idx]) { seen[idx] = 1; ranked.emplace_back(li, le.second); }
            }
        }
        for (int e = 0; e < NE; ++e) {
            for (int li = 0; li < n_layers; ++li) {
                const size_t idx = static_cast<size_t>(li) * NE + static_cast<size_t>(e);
                if (!seen[idx]) { seen[idx] = 1; ranked.emplace_back(li, e); }
            }
        }

        const int max_total_cap = (cfg.cache_slots_override >= 0)
            ? std::min<int>(static_cast<int>(cfg.cache_slots_override), static_cast<int>(total_exps))
            : static_cast<int>(total_exps);

        // 1. Tight-packed pinned static core from STRP profile
        std::vector<std::pair<int, int>> static_list;
        std::vector<size_t> static_offs;
        vram_static_bytes = 0;
        const float pin_frac = have_prof ? std::clamp(cfg.static_pin_frac, 0.0f, 0.90f) : 0.0f;
        const size_t static_budget = static_cast<size_t>(static_cast<double>(budget_b) * static_cast<double>(pin_frac));
        if (pin_frac > 0.0f && static_budget > 0 && max_total_cap > 0) {
            for (const auto& le : ranked) {
                if (static_cast<int>(static_list.size()) >= max_total_cap) break;
                const auto& em = layers[static_cast<size_t>(le.first)].exp[static_cast<size_t>(le.second)];
                const size_t sz = round256(em.fc1_bytes) + round256(em.fc2_bytes);
                if (vram_static_bytes + sz > static_budget) break;
                static_offs.push_back(vram_static_bytes);
                vram_static_bytes += sz;
                static_list.push_back(le);
            }
        }
        pinned_count = static_cast<int>(static_list.size());
        std::vector<uint8_t> is_pinned_exp(total_exps, 0);
        for (const auto& le : static_list)
            is_pinned_exp[static_cast<size_t>(le.first) * NE + static_cast<size_t>(le.second)] = 1;

        // 2. Dynamic uniform slot pool
        dyn_slot_stride = stg_slot_stride;
        const size_t dyn_budget = (budget_b > vram_static_bytes) ? (budget_b - vram_static_bytes) : 0;
        const int rem_cap = std::max(0, max_total_cap - pinned_count);
        const int rem_exps = static_cast<int>(total_exps) - pinned_count;
        int want_dyn = (dyn_slot_stride > 0) ? static_cast<int>(dyn_budget / dyn_slot_stride) : 0;
        want_dyn = std::min({want_dyn, rem_cap, rem_exps});
        dyn_total_slots = std::max(0, want_dyn);

        if (cfg.adaptive && cfg.promote_max > 0 && (pinned_count + dyn_total_slots) < static_cast<int>(total_exps) && dyn_total_slots >= 32) {
            ring_slots = std::min(cfg.promote_max, std::max(4, dyn_total_slots / 16));
        } else {
            ring_slots = 0;
        }
        dyn_active_slots = dyn_total_slots - ring_slots;

        if (vram_static_bytes > 0)
            hybrid_ck(cudaMalloc(&vram_static_pool, vram_static_bytes), "vram_static_pool");
        if (dyn_total_slots > 0)
            hybrid_ck(cudaMalloc(&vram_dyn_pool, static_cast<size_t>(dyn_total_slots) * dyn_slot_stride), "vram_dyn_pool");
        if (ring_slots > 0)
            hybrid_ck(cudaHostAlloc(&promo_bounce, static_cast<size_t>(ring_slots) * dyn_slot_stride, cudaHostAllocDefault), "promo_bounce");

        slot_owner.assign(static_cast<size_t>(dyn_total_slots), -1);
        free_slots.clear();
        for (int s = dyn_active_slots; s < dyn_total_slots; ++s) free_slots.push_back(s);

        // Pre-fill static core + initial dynamic pool through bounded staging bounce buffer
        const auto t0 = std::chrono::steady_clock::now();
        size_t copied_bytes = 0;
        auto copy_expert_sync = [&](int li, int e, uint8_t* dst) {
            const auto& ld = layers[static_cast<size_t>(li)];
            const auto& em = ld.exp[static_cast<size_t>(e)];
            const size_t fc1_al = round256(em.fc1_bytes);
            std::memcpy(stg_bounce[0], ld.h_fc1 + em.fc1_off, em.fc1_bytes);
            std::memcpy(stg_bounce[0] + fc1_al, ld.h_fc2 + em.fc2_off, em.fc2_bytes);
            hybrid_ck(cudaMemcpyAsync(dst, stg_bounce[0], fc1_al + em.fc2_bytes, cudaMemcpyHostToDevice, h2d_stream), "prefill H2D");
            hybrid_ck(cudaStreamSynchronize(h2d_stream), "prefill H2D sync");
            set_expert_ptab_host(li, e, dst, dst + fc1_al);
            copied_bytes += em.fc1_bytes + em.fc2_bytes;
        };

        for (int i = 0; i < pinned_count; ++i) {
            const int li = static_list[static_cast<size_t>(i)].first;
            const int e  = static_list[static_cast<size_t>(i)].second;
            const size_t idx = static_cast<size_t>(li) * NE + static_cast<size_t>(e);
            copy_expert_sync(li, e, vram_static_pool + static_offs[static_cast<size_t>(i)]);
            exp_slot[idx] = -2;
            heat[idx] = cfg.heat_cap;
        }
        int dyn_filled = 0;
        for (const auto& le : ranked) {
            if (dyn_filled >= dyn_active_slots) break;
            const size_t idx = static_cast<size_t>(le.first) * NE + static_cast<size_t>(le.second);
            if (is_pinned_exp[idx]) continue;
            const int s = dyn_filled++;
            copy_expert_sync(le.first, le.second, vram_dyn_pool + static_cast<size_t>(s) * dyn_slot_stride);
            exp_slot[idx] = s;
            slot_owner[static_cast<size_t>(s)] = static_cast<int>(idx);
            const float dfrac = static_cast<float>(s) / static_cast<float>(std::max(1, dyn_active_slots));
            heat[idx] = 1.20f - 1.00f * dfrac;
        }

        for (int li = 0; li < n_layers; ++li) {
            auto& ld = layers[static_cast<size_t>(li)];
            hybrid_ck(cudaMemcpyAsync(ld.d_ptab, ld.h_ptab.data(), ld.h_ptab.size() * sizeof(void*),
                                      cudaMemcpyHostToDevice, main_st), "d_ptab upload");
        }
        hybrid_ck(cudaStreamSynchronize(main_st), "d_ptab upload sync");

        size_t free_after = 0;
        cudaMemGetInfo(&free_after, &total_b);
        const int active_total = pinned_count + dyn_filled;
        const size_t pool_total_bytes = vram_static_bytes + static_cast<size_t>(dyn_total_slots) * dyn_slot_stride;
        std::fprintf(stderr,
                     "# ernie55 hybrid cache: %d/%zu experts in VRAM (%.1f%%; %d pinned %.2f GiB + %d dyn + %d ring "
                     "@ %.2f MiB/slot = %.2f GiB, stg=2x%.1f MiB, adaptive=%d, cpu_overlap=%d, profile=%s, "
                     "prefilled %.2f GB in %.2f s; %.0f MiB VRAM free)\n",
                     active_total, total_exps, 100.0 * active_total / static_cast<double>(total_exps),
                     pinned_count, vram_static_bytes / double(1ull << 30), dyn_filled, ring_slots,
                     dyn_slot_stride / double(1ull << 20), pool_total_bytes / double(1ull << 30),
                     stg_bytes / double(1ull << 20), cfg.adaptive ? 1 : 0, cfg.cpu_overlap ? 1 : 0,
                     have_prof ? cfg.profile_path.c_str() : "none",
                     copied_bytes / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                     free_after / double(1ull << 20));
    }

    // Run routed MoE experts for layer `moe_li` (0..41) using VRAM hot cache + double-buffered cold H2D staging.
    // Caller has already executed `moe_route`, `fc1_lat`, and `k_ernie55_prerot_x<EHB>`.
    // Output `sc.fc2_part_r[route * EH]` is 100% bit-exact vs resident GPU execution!
    void run_routed_layer(int moe_li, const ernie55::MoeWeights& w, ernie55::MoeScratch& sc,
                          int M, cudaStream_t st) {
        const auto& ld = layers[static_cast<size_t>(moe_li)];
        const int n_routes = M * geom::TOPK;
        hybrid_ck(cudaMemcpyAsync(h_topk_idx, sc.topk_idx, static_cast<size_t>(n_routes) * sizeof(int),
                                  cudaMemcpyDeviceToHost, st), "topk D2H");
        hybrid_ck(cudaStreamSynchronize(st), "topk D2H sync");

        const int GMAX = (M < 16) ? 8 : 16;
        // Group routes by expert id
        std::vector<int> exp_routes[NE];
        std::vector<int> hot_exps;
        std::vector<int> cold_exps;
        hot_exps.reserve(64);
        cold_exps.reserve(64);
        for (int r = 0; r < n_routes; ++r) {
            const int e = h_topk_idx[r];
            if (e < 0 || e >= NE) hybrid_die("bad routed expert id " + std::to_string(e));
            if (exp_routes[e].empty()) {
                if (exp_slot[static_cast<size_t>(moe_li) * NE + static_cast<size_t>(e)] != -1) hot_exps.push_back(e);
                else cold_exps.push_back(e);
            }
            exp_routes[e].push_back(r);
        }

        auto fill_groups = [&](int buf, const std::vector<int>& exps, size_t lo, size_t hi) -> int {
            int ng = 0;
            for (size_t idx = lo; idx < hi; ++idx) {
                const int e = exps[idx];
                const auto& rv = exp_routes[e];
                for (size_t p = 0; p < rv.size(); p += static_cast<size_t>(GMAX)) {
                    const int cnt = static_cast<int>(std::min<size_t>(static_cast<size_t>(GMAX), rv.size() - p));
                    h_grp_exp[buf][ng] = e;
                    h_grp_cnt[buf][ng] = cnt;
                    for (int k = 0; k < cnt; ++k) h_grp_routes[buf][ng * GMAX + k] = rv[p + static_cast<size_t>(k)];
                    ++ng;
                }
            }
            *h_grp_n[buf] = ng;
            return ng;
        };

        auto launch_groups = [&](int buf, int ng, const uint8_t* const* ptab, cudaStream_t stream) {
            if (ng <= 0) return;
            hybrid_ck(cudaMemcpyAsync(d_grp_n[buf], h_grp_n[buf], sizeof(int), cudaMemcpyHostToDevice, stream), "grp_n");
            hybrid_ck(cudaMemcpyAsync(d_grp_exp[buf], h_grp_exp[buf], static_cast<size_t>(ng) * sizeof(int), cudaMemcpyHostToDevice, stream), "grp_exp");
            hybrid_ck(cudaMemcpyAsync(d_grp_cnt[buf], h_grp_cnt[buf], static_cast<size_t>(ng) * sizeof(int), cudaMemcpyHostToDevice, stream), "grp_cnt");
            hybrid_ck(cudaMemcpyAsync(d_grp_routes[buf], h_grp_routes[buf], static_cast<size_t>(ng) * GMAX * sizeof(int), cudaMemcpyHostToDevice, stream), "grp_routes");
            if (GMAX == 8) {
                ernie55::k_ernie55_routed_fc1_grp<<<ng * w.routed.eb, 256, 0, stream>>>(
                    sc.x_had_r, d_grp_n[buf], d_grp_exp[buf], d_grp_cnt[buf], d_grp_routes[buf],
                    w.routed.tab, w.routed.kmix, nullptr, ptab, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
                ernie55::k_ernie55_routed_fc2_grp<<<ng * geom::EHB, 256, 0, stream>>>(
                    sc.fc1_act_r, d_grp_n[buf], d_grp_exp[buf], d_grp_cnt[buf], d_grp_routes[buf],
                    sc.topk_w, w.routed.tab, nullptr, ptab, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
            } else {
                ernie55::k_ernie55_routed_fc1_grp16<<<ng * w.routed.eb, 256, 0, stream>>>(
                    sc.x_had_r, d_grp_n[buf], d_grp_exp[buf], d_grp_cnt[buf], d_grp_routes[buf],
                    w.routed.tab, w.routed.kmix, nullptr, ptab, w.routed.v1b, w.routed.v2, sc.fc1_act_r, w.routed.eb);
                ernie55::k_ernie55_routed_fc2_grp16<<<ng * geom::EHB, 256, 0, stream>>>(
                    sc.fc1_act_r, d_grp_n[buf], d_grp_exp[buf], d_grp_cnt[buf], d_grp_routes[buf],
                    sc.topk_w, w.routed.tab, nullptr, ptab, w.routed.sg, w.routed.v3b, sc.fc2_part_r, w.routed.eb);
            }
        };

        auto stage_cold_chunk = [&](int buf, size_t lo, size_t hi) {
            const size_t ptab_n = static_cast<size_t>(NE) * EB * 2;
            std::memset(stg_h_ptab[buf], 0, ptab_n * sizeof(void*));
            size_t cur_off = 0;
            for (size_t idx = lo; idx < hi; ++idx) {
                const int e = cold_exps[idx];
                const auto& em = ld.exp[static_cast<size_t>(e)];
                const size_t fc1_al = round256(em.fc1_bytes);
                const size_t need = fc1_al + round256(em.fc2_bytes);
                uint8_t* bdst = stg_bounce[buf] + cur_off;
                uint8_t* vdst = stg_vram[buf] + cur_off;
                std::memcpy(bdst, ld.h_fc1 + em.fc1_off, em.fc1_bytes);
                std::memcpy(bdst + fc1_al, ld.h_fc2 + em.fc2_off, em.fc2_bytes);
                for (int b = 0; b < em.nb; ++b) {
                    const int sl = (e * EB + b) * 2;
                    stg_h_ptab[buf][static_cast<size_t>(sl)]     = vdst + em.piece_fc1_off[b];
                    stg_h_ptab[buf][static_cast<size_t>(sl) + 1] = vdst + fc1_al + em.piece_fc2_off[b];
                }
                cur_off += need;
            }
            if (cur_off > 0) {
                hybrid_ck(cudaMemcpyAsync(stg_vram[buf], stg_bounce[buf], cur_off, cudaMemcpyHostToDevice, h2d_stream), "cold stg H2D");
            }
            hybrid_ck(cudaMemcpyAsync(stg_d_ptab[buf], stg_h_ptab[buf], ptab_n * sizeof(void*), cudaMemcpyHostToDevice, h2d_stream), "cold ptab H2D");
            hybrid_ck(cudaEventRecord(stg_h2d_ev[buf], h2d_stream), "stg_h2d_ev");
            stat_cold_staged += (hi - lo);
        };

        // 1. Run all VRAM-hot groups immediately on `st` (buf 0) while CPU prepares cold chunk 0!
        if (!hot_exps.empty()) {
            const int ng_hot = fill_groups(0, hot_exps, 0, hot_exps.size());
            launch_groups(0, ng_hot, ld.d_ptab, st);
            hybrid_ck(cudaEventRecord(stg_comp_ev[0], st), "hot comp ev");
        }

        // 2. Process cold experts in double-buffered chunks of `stg_slots`
        const size_t n_cold = cold_exps.size();
        const size_t stride = static_cast<size_t>(stg_slots);
        if (n_cold > 0) {
            const size_t n_chunks = (n_cold + stride - 1) / stride;
            bool buf_in_flight[2] = {false, false};
            for (size_t ch = 0; ch < n_chunks; ++ch) {
                const int buf = static_cast<int>(ch & 1);
                const size_t lo = ch * stride;
                const size_t hi = std::min(n_cold, lo + stride);
                if (ch == 0 && !hot_exps.empty()) {
                    // Wait for hot group buffers on buf 0 to finish DMA before reusing h_grp_*[0]
                    hybrid_ck(cudaEventSynchronize(stg_comp_ev[0]), "hot grp sync");
                }
                if (buf_in_flight[buf]) {
                    hybrid_ck(cudaEventSynchronize(stg_comp_ev[buf]), "cold buf reuse sync");
                    buf_in_flight[buf] = false;
                }
                stage_cold_chunk(buf, lo, hi);
                const int ng_c = fill_groups(buf, cold_exps, lo, hi);
                hybrid_ck(cudaStreamWaitEvent(st, stg_h2d_ev[buf], 0), "wait cold H2D");
                launch_groups(buf, ng_c, stg_d_ptab[buf], st);
                hybrid_ck(cudaEventRecord(stg_comp_ev[buf], st), "cold comp ev");
                buf_in_flight[buf] = true;
            }
            for (int b = 0; b < 2; ++b) {
                if (buf_in_flight[b]) hybrid_ck(cudaEventSynchronize(stg_comp_ev[b]), "cold final sync");
            }
        }
    }

    // Commit completed async H2D promotions into fixed-address `d_ptab` on `st`.
    void commit_ready_promotions(cudaStream_t st, bool blocking) {
        if (!promo_in_flight) return;
        if (!blocking && cudaEventQuery(promo_ev) != cudaSuccess) return;
        if (blocking) cudaEventSynchronize(promo_ev);
        int np = 0;
        for (const auto& sw : pending_swaps) {
            if (sw.old_idx >= 0) {
                const int ol = sw.old_idx / NE, oe = sw.old_idx % NE;
                auto& old_ld = layers[static_cast<size_t>(ol)];
                const auto& old_em = old_ld.exp[static_cast<size_t>(oe)];
                for (int b = 0; b < old_em.nb; ++b) {
                    const int sl = (oe * EB + b) * 2;
                    h_patches[np++] = {old_ld.d_ptab + sl,     nullptr};
                    h_patches[np++] = {old_ld.d_ptab + sl + 1, nullptr};
                    old_ld.h_ptab[static_cast<size_t>(sl)]     = nullptr;
                    old_ld.h_ptab[static_cast<size_t>(sl) + 1] = nullptr;
                }
                exp_slot[static_cast<size_t>(sw.old_idx)] = -1;
                slot_owner[static_cast<size_t>(sw.old_slot)] = -1;
                free_slots.push_back(sw.old_slot);
            }
            {
                const int nl_i = sw.new_idx / NE, ne_i = sw.new_idx % NE;
                auto& new_ld = layers[static_cast<size_t>(nl_i)];
                const auto& new_em = new_ld.exp[static_cast<size_t>(ne_i)];
                const uint8_t* v1 = vram_dyn_pool + static_cast<size_t>(sw.stg_slot) * dyn_slot_stride;
                const uint8_t* v2 = v1 + round256(new_em.fc1_bytes);
                for (int b = 0; b < new_em.nb; ++b) {
                    const int sl = (ne_i * EB + b) * 2;
                    h_patches[np++] = {new_ld.d_ptab + sl,     v1 + new_em.piece_fc1_off[b]};
                    h_patches[np++] = {new_ld.d_ptab + sl + 1, v2 + new_em.piece_fc2_off[b]};
                    new_ld.h_ptab[static_cast<size_t>(sl)]     = v1 + new_em.piece_fc1_off[b];
                    new_ld.h_ptab[static_cast<size_t>(sl) + 1] = v2 + new_em.piece_fc2_off[b];
                }
                exp_slot[static_cast<size_t>(sw.new_idx)] = sw.stg_slot;
                slot_owner[static_cast<size_t>(sw.stg_slot)] = sw.new_idx;
                exp_pending[static_cast<size_t>(sw.new_idx)] = 0;
            }
            ++stat_swaps;
        }
        pending_swaps.clear();
        promo_in_flight = false;
        if (np > 0) {
            hybrid_ck(cudaMemcpyAsync(d_patches, h_patches, static_cast<size_t>(np) * sizeof(PtabPatchEntry),
                                      cudaMemcpyHostToDevice, st), "ptab patch H2D");
            k_ernie55_ptab_patch<<<(np + 255) / 256, 256, 0, st>>>(d_patches, np);
            hybrid_ck(cudaEventRecord(patch_ev, st), "patch_ev");
            patch_ev_recorded = true;
        }
    }

    // Call between steps AFTER main stream `st` has synchronized.
    void maintain_after_step(cudaStream_t st) {
        if (!active || n_layers == 0 || !d_exp_hits) return;
        const size_t total_exps = static_cast<size_t>(n_layers) * NE;

        commit_ready_promotions(st, false);

        hybrid_ck(cudaMemcpy(h_exp_hits, d_exp_hits, total_exps * sizeof(uint32_t), cudaMemcpyDeviceToHost), "hits D2H");
        hybrid_ck(cudaMemsetAsync(d_exp_hits, 0, total_exps * sizeof(uint32_t), st), "hits zero");

        uint64_t step_hits = 0, step_misses = 0;
        for (size_t i = 0; i < total_exps; ++i) {
            const uint32_t c = h_exp_hits[i];
            float h = heat[i] * cfg.heat_decay;
            if (c > 0) {
                h += static_cast<float>(c);
                if (exp_slot[i] != -1) step_hits += c;
                else step_misses += c;
            }
            if (h > cfg.heat_cap) h = cfg.heat_cap;
            heat[i] = h;
        }
        stat_hits += step_hits;
        stat_misses += step_misses;
        if (step_hits + step_misses > 0) {
            ++stat_steps;
            if ((stat_steps % 50) == 0) print_stats();
        }

        if (!cfg.adaptive || !promo_bounce || dyn_total_slots == 0 ||
            (pinned_count + dyn_total_slots) >= static_cast<int>(total_exps) ||
            promo_in_flight || free_slots.empty()) return;

        const int max_k = std::min({cfg.promote_max, ring_slots, static_cast<int>(free_slots.size())});
        if (max_k <= 0) return;

        std::vector<std::pair<float, int>> cold_cands;
        cold_cands.reserve(64);
        for (size_t i = 0; i < total_exps; ++i) {
            if (exp_slot[i] == -1 && !exp_pending[i] && heat[i] >= cfg.min_promote_heat)
                cold_cands.emplace_back(heat[i], static_cast<int>(i));
        }
        if (cold_cands.empty()) return;
        const int n_cold = std::min(max_k, static_cast<int>(cold_cands.size()));
        std::partial_sort(cold_cands.begin(), cold_cands.begin() + n_cold, cold_cands.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<std::pair<float, int>> hot_victims;
        hot_victims.reserve(static_cast<size_t>(dyn_active_slots));
        for (int s = 0; s < dyn_total_slots; ++s) {
            const int owner = slot_owner[static_cast<size_t>(s)];
            if (owner >= 0) hot_victims.emplace_back(heat[static_cast<size_t>(owner)], s);
        }
        if (hot_victims.empty()) return;
        const int n_vic = std::min(n_cold, static_cast<int>(hot_victims.size()));
        std::partial_sort(hot_victims.begin(), hot_victims.begin() + n_vic, hot_victims.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

        if (patch_ev_recorded) {
            cudaStreamWaitEvent(h2d_stream, patch_ev, 0);
            patch_ev_recorded = false;
        }

        pending_swaps.clear();
        for (int k = 0; k < n_vic; ++k) {
            const float c_heat = cold_cands[static_cast<size_t>(k)].first;
            const int   c_idx  = cold_cands[static_cast<size_t>(k)].second;
            const float v_heat = hot_victims[static_cast<size_t>(k)].first;
            const int   v_slot = hot_victims[static_cast<size_t>(k)].second;
            const int   v_idx  = slot_owner[static_cast<size_t>(v_slot)];
            if (c_heat <= v_heat * cfg.promote_hyst) break;
            if (free_slots.empty()) break;

            const int stg = free_slots.back();
            free_slots.pop_back();
            const int cli = c_idx / NE, ce = c_idx % NE;
            const auto& cld = layers[static_cast<size_t>(cli)];
            const auto& cem = cld.exp[static_cast<size_t>(ce)];
            const size_t fc1_al = round256(cem.fc1_bytes);
            uint8_t* pb = promo_bounce + static_cast<size_t>(k) * dyn_slot_stride;
            std::memcpy(pb, cld.h_fc1 + cem.fc1_off, cem.fc1_bytes);
            std::memcpy(pb + fc1_al, cld.h_fc2 + cem.fc2_off, cem.fc2_bytes);
            uint8_t* dst = vram_dyn_pool + static_cast<size_t>(stg) * dyn_slot_stride;
            hybrid_ck(cudaMemcpyAsync(dst, pb, fc1_al + cem.fc2_bytes, cudaMemcpyHostToDevice, h2d_stream), "promo H2D");
            exp_pending[static_cast<size_t>(c_idx)] = 1;
            pending_swaps.push_back({c_idx, v_idx, stg, v_slot});
        }
        if (!pending_swaps.empty()) {
            hybrid_ck(cudaEventRecord(promo_ev, h2d_stream), "promo_ev");
            promo_in_flight = true;
        }
    }

    void print_stats() const {
        const uint64_t tot = stat_hits + stat_misses;
        const double hit_pct = tot ? (100.0 * static_cast<double>(stat_hits) / static_cast<double>(tot)) : 0.0;
        std::fprintf(stderr,
                     "# ernie55 hybrid stats (%llu steps): hit=%.1f%% (%llu/%llu), swaps=%llu, cold_staged=%llu\n",
                     static_cast<unsigned long long>(stat_steps), hit_pct,
                     static_cast<unsigned long long>(stat_hits), static_cast<unsigned long long>(tot),
                     static_cast<unsigned long long>(stat_swaps), static_cast<unsigned long long>(stat_cold_staged));
    }
};

} // namespace ernie55rt
