// Qwen3.8-Flash-Next Strata-mode hot/cold VRAM expert cache + pinned host RAM backing + double-buffered H2D prefill.
//
// Design:
//   - Routed experts (L000..L047 tr3.fc1, tr3.fc2) live in mapped pinned host RAM (cudaHostAllocMapped), wide-reordered
//     in place on CPU at boot so every expert is ready for both zero-copy UVA reads (~26.5 GB/s PCIe Gen4 x16) and
//     async H2D DMA (~28.9 GB/s).
//   - Remaining free VRAM (minus vram_reserve_mib) is carved into a global pool of uniform expert slots (~9,000-10,000
//     K3 experts = ~36-41% of all 24,576 experts on RTX 5090), pre-filled from a Strata STRP v1 frequency profile.
//   - Every layer keeps a fixed-address device piece table `d_ptab[l]` (512 * EB * 2 pointers): hot experts point into
//     GDDR7 VRAM (`vram_pool`), cold experts point directly to mapped pinned host RAM UVA -> 100% CUDA Graph compatible
//     and 100% bit-exact vs VRAM-resident gr8!
//   - Adaptive tiering: `k_qw38_moe_build_groups` records per-expert GPU hit counts (`d_exp_hits`). Between decode steps,
//     exponentially decayed heat scores select hot cold experts and asynchronously H2D-copy them into spare ring slots
//     on `h2d_stream`, then atomically patch `d_ptab` via `k_strata_ptab_patch` (zero graph invalidation, zero stall).
//   - High-throughput prefill (M >= 32): partitions groups into VRAM-hot vs cold batches of up to 24 unique cold experts,
//     double-buffering H2D DMA into VRAM staging buffers so each cold expert crosses PCIe ONCE per prefill chunk.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "../model_geom_qw38.h"
#include "../kernels/tr3.cuh"
#include "../kernels/tr3_wide.cuh"
#include "../kernels/qw38_moe.cuh"

namespace qw38rt {

// ------------------------------------------------------------------------------------------------ Env helpers
inline bool strata_env_bool(const char* name, bool def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}
inline int64_t strata_env_i64(const char* name, int64_t def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    return std::atoll(v);
}
inline float strata_env_f32(const char* name, float def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    return static_cast<float>(std::atof(v));
}
inline std::string strata_env_str(const char* name, const std::string& def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

// ------------------------------------------------------------------------------------------------ Host wide-order reorder (K2/K3/K4)
template <int BITS>
inline void trw_widen_fc1_mat_host(uint8_t* mat) {
    constexpr int GRP_B = 2048 * BITS;
    uint8_t tmp[GRP_B];
    for (int g = 0; g < geom::KSH / 8; ++g) {
        uint8_t* grp = mat + static_cast<size_t>(g) * GRP_B;
        for (int j = 0; j < 8; ++j) {
            for (int lane = 0; lane < 32; ++lane) {
                u64 v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p)
                    trw_pack<BITS>(v, p, trw_tile_chunk<BITS>(grp + (p * 8 + j) * 32 * BITS, lane));
                for (int n = 0; n < BITS; ++n)
                    std::memcpy(tmp + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
            }
        }
        std::memcpy(grp, tmp, GRP_B);
    }
}

inline void tr4w_widen_fc1_mat_host(uint8_t* mat) {
    uint8_t tmp[TR4W_GROUP];
    for (int g = 0; g < geom::KSH / 8; ++g) {
        uint8_t* grp = mat + static_cast<size_t>(g) * TR4W_GROUP;
        for (int j = 0; j < 8; ++j) {
            for (int lane = 0; lane < 32; ++lane) {
                u32 w[8];
                for (int p = 0; p < 8; ++p)
                    std::memcpy(&w[p], grp + (p * 8 + j) * 128 + lane * 4, 4);
                std::memcpy(tmp + (j * 2 + 0) * 512 + lane * 16, &w[0], 16);
                std::memcpy(tmp + (j * 2 + 1) * 512 + lane * 16, &w[4], 16);
            }
        }
        std::memcpy(grp, tmp, TR4W_GROUP);
    }
}

template <int BITS>
inline void trw_widen_fc2_blk_host(uint8_t* mat) {
    constexpr int NT_B = 8 * 32 * BITS;
    uint8_t tmp[NT_B];
    for (int nt = 0; nt < geom::KSH; ++nt) {
        uint8_t* grp = mat + static_cast<size_t>(nt) * NT_B;
        for (int lane = 0; lane < 32; ++lane) {
            u64 v[3] = {0, 0, 0};
            for (int p = 0; p < 8; ++p)
                trw_pack<BITS>(v, p, trw_tile_chunk<BITS>(grp + p * 32 * BITS, lane));
            for (int n = 0; n < BITS; ++n)
                std::memcpy(tmp + n * 256 + lane * 8, &v[n], 8);
        }
        std::memcpy(grp, tmp, NT_B);
    }
}

inline void tr4w_widen_fc2_blk_host(uint8_t* mat) {
    uint8_t tmp[TR4W_NT];
    for (int nt = 0; nt < geom::KSH; ++nt) {
        uint8_t* grp = mat + static_cast<size_t>(nt) * TR4W_NT;
        for (int lane = 0; lane < 32; ++lane) {
            u32 w[8];
            for (int p = 0; p < 8; ++p)
                std::memcpy(&w[p], grp + p * 128 + lane * 4, 4);
            std::memcpy(tmp + 0 + lane * 16, &w[0], 16);
            std::memcpy(tmp + 512 + lane * 16, &w[4], 16);
        }
        std::memcpy(grp, tmp, TR4W_NT);
    }
}

inline void widen_layer_experts_host(uint8_t* fc1, uint8_t* fc2, const long long* tab,
                                     const long long* kmix, int rows) {
    constexpr size_t MB = static_cast<size_t>(geom::KSH) * 8 * 32; // 40960 B/bit
    for (int e = 0; e < rows; ++e) {
        const long long* tr = tab + 6 * e;
        const int nb = static_cast<int>(tr[3]);
        const long long km = kmix ? kmix[e] : 0;
        size_t off1 = static_cast<size_t>(tr[0]);
        for (int bm = 0; bm < 2 * nb; ++bm) {
            const int k = km ? static_cast<int>((km >> (4 * bm)) & 15) : static_cast<int>(tr[4]);
            if (k == 3) trw_widen_fc1_mat_host<3>(fc1 + off1);
            else if (k == 2) trw_widen_fc1_mat_host<2>(fc1 + off1);
            else if (k == 4) tr4w_widen_fc1_mat_host(fc1 + off1);
            off1 += static_cast<size_t>(k) * MB;
        }
        const int k2 = static_cast<int>(tr[4]);
        size_t off2 = static_cast<size_t>(tr[1]);
        for (int b = 0; b < nb; ++b) {
            if (k2 == 3) trw_widen_fc2_blk_host<3>(fc2 + off2);
            else if (k2 == 2) trw_widen_fc2_blk_host<2>(fc2 + off2);
            else if (k2 == 4) tr4w_widen_fc2_blk_host(fc2 + off2);
            off2 += static_cast<size_t>(k2) * MB;
        }
    }
}

// ------------------------------------------------------------------------------------------------ Device kernels for atomic ptab patching
struct PtabPatchEntry {
    const uint8_t** dst_slot;
    const uint8_t* ptr;
    uint8_t* cold_dst;   // optional: update d_is_cold[li*NE+e] atomically alongside ptab!
    uint8_t cold_val;
};

__global__ void k_strata_ptab_patch(const PtabPatchEntry* __restrict__ patches, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        *patches[i].dst_slot = patches[i].ptr;
        if (patches[i].cold_dst) *patches[i].cold_dst = patches[i].cold_val;
    }
}

// ------------------------------------------------------------------------------------------------ Strata STRP v1 profile reader
inline bool read_strata_expert_profile(const std::string& path, int n_layers, int n_experts,
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
        if (l >= 0 && l < n_layers && e >= 0 && e < n_experts) {
            ranked.emplace_back(l, e);
        }
    }
    return !ranked.empty();
}

// ------------------------------------------------------------------------------------------------ StrataConfig & StrataExpertCache
struct StrataConfig {
    bool enabled = false;
    int64_t vram_reserve_mib = 1280;
    int64_t cache_slots_override = -1; // -1 = auto from free VRAM
    std::string profile_path;
    bool adaptive = true;
    bool async_promote = true;
    int promote_max_experts = 12;      // max experts swapped per decode round (~22 MiB async on h2d_stream)
    float promote_hyst = 1.15f;        // cold heat must beat worst hot heat by 15%
    float heat_decay = 0.97f;          // fast enough to adapt across domain switches while retaining session experts
    float heat_cap = 16.0f;            // cap dynamic heat so stale session experts never lock out new sessions
    float min_promote_heat = 3.0f;     // require >= 3 hits before promoting (eliminates 1-2 hit PCIe thrashing)
    float static_pin_frac = 0.45f;     // pin top 45% of profile-ranked VRAM experts as universal static core
    int pf_stage_experts = 32;         // cold experts per double-buffered prefill H2D batch
    int pf_d2d_promote = 48;           // max cold experts per layer promoted via zero-PCIe GDDR7 D2D during prefill
    bool pf_warm_heat = true;          // gently warm session expert heat during prompt prefill
    bool cpu_moe = true;               // hybrid CPU AVX-512 BF16 K3 cold-expert offload (decode + small prefill)
    int cpu_moe_threads = 14;          // 14 unpinned work-stealing workers (leaves 2 hw threads for CUDA/OS)
    float cpu_moe_frac = 0.60f;        // fraction of decode cold experts dispatched to CPU in parallel with GPU PCIe
    int pf_cpu_max_m = 512;            // max prefill chunk M for hybrid CPU cold-expert offload
    float pf_cpu_frac = 0.50f;         // fraction of small-prefill cold experts (cnt <= 4) dispatched to CPU
};

inline StrataConfig load_strata_config_from_env() {
    StrataConfig c;
    c.enabled = strata_env_bool("GR8_QW38_STRATA", false);
    c.vram_reserve_mib = strata_env_i64("GR8_QW38_VRAM_RESERVE_MIB", 1920);
    c.cache_slots_override = strata_env_i64("GR8_QW38_EXPERT_CACHE_SLOTS", -1);
    c.profile_path = strata_env_str("GR8_QW38_EXPERT_PROFILE", "");
    c.adaptive = strata_env_bool("GR8_QW38_ADAPTIVE", true);
    c.async_promote = strata_env_bool("GR8_QW38_PROMOTE_ASYNC", true);
    c.promote_max_experts = static_cast<int>(strata_env_i64("GR8_QW38_PROMOTE_MAX", 32));
    c.promote_hyst = strata_env_f32("GR8_QW38_PROMOTE_HYST", 1.10f);
    c.heat_decay = strata_env_f32("GR8_QW38_HEAT_DECAY", 0.94f);
    c.heat_cap = strata_env_f32("GR8_QW38_HEAT_CAP", 16.0f);
    c.min_promote_heat = strata_env_f32("GR8_QW38_MIN_PROMOTE_HEAT", 1.5f);
    c.static_pin_frac = strata_env_f32("GR8_QW38_STATIC_PIN_FRAC", 0.20f);
    c.pf_stage_experts = static_cast<int>(strata_env_i64("GR8_QW38_PF_STAGE_EXPS", 32));
    c.pf_d2d_promote = static_cast<int>(strata_env_i64("GR8_QW38_PF_D2D_PROMOTE", 64));
    c.pf_warm_heat = strata_env_bool("GR8_QW38_PF_WARM_HEAT", true);
    c.cpu_moe = strata_env_bool("GR8_QW38_CPU_MOE", true);
    c.cpu_moe_threads = static_cast<int>(strata_env_i64("GR8_QW38_CPU_THREADS", 14));
    c.cpu_moe_frac = strata_env_f32("GR8_QW38_CPU_FRAC", 0.70f);
    c.pf_cpu_max_m = static_cast<int>(strata_env_i64("GR8_QW38_PF_CPU_MAX_M", 512));
    c.pf_cpu_frac = strata_env_f32("GR8_QW38_PF_CPU_FRAC", 0.55f);
    return c;
}

struct StrataExpertCache {
    static constexpr int NE = geom::NEXP; // 512
    static constexpr size_t ALIGN256 = 256;
    static size_t round256(size_t n) { return (n + ALIGN256 - 1) & ~(ALIGN256 - 1); }

    struct ExpMeta {
        size_t fc1_off = 0, fc1_bytes = 0;
        size_t fc2_off = 0, fc2_bytes = 0;
        int nb = 0;
        size_t piece_fc1_off[8] = {};
        size_t piece_fc2_off[8] = {};
    };

    struct LayerData {
        int layer_id = 0;
        int rows = 0;
        int eb = 0;
        const uint8_t* h_fc1 = nullptr;   // pinned host pointer
        const uint8_t* d_fc1 = nullptr;   // UVA device pointer
        const uint8_t* h_fc2 = nullptr;   // pinned host pointer
        const uint8_t* d_fc2 = nullptr;   // UVA device pointer
        const uint8_t** d_ptab = nullptr; // fixed-address device [rows * eb * 2]
        std::vector<const uint8_t*> h_ptab;
        std::vector<ExpMeta> exp;
    };

    StrataConfig cfg;
    int n_layers = 0;
    int eb = 5;
    size_t slot_stride = 0;
    size_t fc1_max_aligned = 0;

    uint8_t* vram_pool = nullptr;
    int total_slots = 0;
    int active_slots = 0;
    int ring_slots = 0;

    std::vector<LayerData> layers;
    // Global residency & heat tables [n_layers * NE]
    std::vector<int> exp_slot;          // -1 = cold (UVA), >= 0 = VRAM slot index
    std::vector<int> slot_owner;        // -1 = free/staging, else (li * NE + e)
    std::vector<uint8_t> slot_pinned;   // 1 = pinned static-profile core slot (never evicted)
    std::vector<uint8_t> exp_pending;   // 1 = currently being promoted
    std::vector<float> heat;
    std::vector<int> free_slots;        // spare staging slots not referenced by any d_ptab
    int pinned_slots = 0;

    // GPU hit counters, cold bitmap & async promotion machinery
    uint32_t* d_exp_hits = nullptr;     // [n_layers * NE]
    uint32_t* h_exp_hits = nullptr;     // pinned [n_layers * NE]
    uint8_t* d_is_cold = nullptr;       // [n_layers * NE] 1 = cold in host RAM, 0 = hot in VRAM
    CpuMoePool cpu_pool;                // hybrid AVX-512 K3 CPU worker pool for decode cold experts
    PtabPatchEntry* d_patches = nullptr;
    PtabPatchEntry* h_patches = nullptr;
    int max_patches = 0;

    cudaStream_t h2d_stream = nullptr;
    cudaEvent_t promo_ev = nullptr;
    bool promo_in_flight = false;
    struct PendingSwap {
        int new_idx;   // li * NE + e
        int old_idx;   // li * NE + e (-1 if free slot)
        int stg_slot;  // VRAM slot where new_idx was copied
        int old_slot;  // VRAM slot freed from old_idx (-1 if none)
    };
    std::vector<PendingSwap> pending_swaps;

    // Prefill double-buffered H2D staging (allocated before init_pool so VRAM budget accounts for it)
    uint8_t* pf_stg[2] = {nullptr, nullptr};
    const uint8_t** pf_ptab[2] = {nullptr, nullptr};
    int* pf_d_ng[2] = {nullptr, nullptr};
    int* pf_d_gexp[2] = {nullptr, nullptr};
    int* pf_d_gcnt[2] = {nullptr, nullptr};
    int* pf_d_goff[2] = {nullptr, nullptr};
    int* pf_h_gexp[2] = {nullptr, nullptr};
    int* pf_h_gcnt[2] = {nullptr, nullptr};
    int* pf_h_goff[2] = {nullptr, nullptr};
    int* pf_h_ro_ng = nullptr;
    int* pf_h_ro_gexp = nullptr;
    int* pf_h_ro_gcnt = nullptr;
    int* pf_h_ro_goff = nullptr;
    int* pf_h_hot_ng = nullptr;
    int* pf_h_hot_gexp = nullptr;
    int* pf_h_hot_gcnt = nullptr;
    int* pf_h_hot_goff = nullptr;
    int pf_max_groups = 0;
    cudaEvent_t pf_grp_ev = nullptr;
    cudaEvent_t pf_ro_ev = nullptr;
    cudaEvent_t pf_hot_h2d_ev = nullptr;
    bool pf_hot_h2d_pending = false;
    cudaEvent_t pf_patch_ev = nullptr;
    bool pf_patch_pending = false;
    cudaEvent_t pf_h2d_ev[2] = {nullptr, nullptr};
    cudaEvent_t pf_comp_ev[2] = {nullptr, nullptr};
    bool pf_h2d_pending[2] = {false, false};
    bool pf_comp_pending[2] = {false, false};
    const uint8_t** pf_h_ptab[2] = {nullptr, nullptr};
    int* pf_h_ng[2] = {nullptr, nullptr};
    size_t pf_ptab_n = 0;

    // Small/medium prefill (M <= 512) CPU hybrid cold-expert offload buffers (100% pinned host RAM, 0 GPU VRAM!)
    static constexpr int PF_CPU_MAX_ROUTES = 2048;
    uint16_t* pf_h_xhad = nullptr;           // [512 * 2560] FP16
    int* pf_h_ro_routes = nullptr;           // [512 * 10] int
    float* pf_h_ro_topkw = nullptr;          // [512 * 10] float
    uint16_t* pf_h_cpu_fc2[2] = {nullptr, nullptr}; // mapped pinned [2048 * 2560] FP16
    __half* pf_d_cpu_fc2[2] = {nullptr, nullptr};   // UVA device pointers to pf_h_cpu_fc2
    int* pf_h_cpu_ridx[2] = {nullptr, nullptr};     // mapped pinned [2048] int
    int* pf_d_cpu_ridx[2] = {nullptr, nullptr};     // UVA device pointers to pf_h_cpu_ridx
    cudaEvent_t pf_x_ev = nullptr;
    cudaEvent_t pf_xhad_ev = nullptr;
    cudaEvent_t pf_cpu_scat_ev[2] = {nullptr, nullptr};
    bool pf_cpu_scat_pending[2] = {false, false};
    int pf_cpu_buf_idx = 0;

    // Telemetry
    uint64_t stat_hits = 0;
    uint64_t stat_misses = 0;
    uint64_t stat_swaps = 0;
    uint64_t stat_pf_d2d_promotes = 0;
    uint64_t stat_steps = 0;
    uint64_t stat_pf_hot_groups = 0;
    uint64_t stat_pf_cold_experts = 0;
    uint64_t stat_verify_ns = 0;
    uint64_t stat_cap_ns = 0;
    uint64_t stat_cap_n = 0;
    uint64_t stat_mtp_ns = 0;
    uint64_t stat_mtp_cap_ns = 0;
    uint64_t stat_mtp_calls = 0;
    uint64_t stat_maint_ns = 0;
    uint64_t stat_verify_toks = 0;
    uint64_t stat_eager_n = 0;
    uint64_t last_pf_decay_step = UINT64_MAX;

    ~StrataExpertCache() { if (stat_steps > 0) print_stats(); release(); }

    void release() {
        if (h2d_stream) { cudaStreamSynchronize(h2d_stream); cudaStreamDestroy(h2d_stream); h2d_stream = nullptr; }
        if (promo_ev) { cudaEventDestroy(promo_ev); promo_ev = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (pf_h2d_ev[b]) { cudaEventDestroy(pf_h2d_ev[b]); pf_h2d_ev[b] = nullptr; }
            if (pf_comp_ev[b]) { cudaEventDestroy(pf_comp_ev[b]); pf_comp_ev[b] = nullptr; }
            if (pf_stg[b]) { cudaFree(pf_stg[b]); pf_stg[b] = nullptr; }
            if (pf_ptab[b]) { cudaFree(pf_ptab[b]); pf_ptab[b] = nullptr; }
            if (pf_d_ng[b]) { cudaFree(pf_d_ng[b]); pf_d_ng[b] = nullptr; }
            if (pf_d_gexp[b]) { cudaFree(pf_d_gexp[b]); pf_d_gexp[b] = nullptr; }
            if (pf_d_gcnt[b]) { cudaFree(pf_d_gcnt[b]); pf_d_gcnt[b] = nullptr; }
            if (pf_d_goff[b]) { cudaFree(pf_d_goff[b]); pf_d_goff[b] = nullptr; }
            if (pf_h_gexp[b]) { cudaFreeHost(pf_h_gexp[b]); pf_h_gexp[b] = nullptr; }
            if (pf_h_gcnt[b]) { cudaFreeHost(pf_h_gcnt[b]); pf_h_gcnt[b] = nullptr; }
            if (pf_h_goff[b]) { cudaFreeHost(pf_h_goff[b]); pf_h_goff[b] = nullptr; }
            if (pf_h_ptab[b]) { cudaFreeHost(pf_h_ptab[b]); pf_h_ptab[b] = nullptr; }
            if (pf_h_ng[b]) { cudaFreeHost(pf_h_ng[b]); pf_h_ng[b] = nullptr; }
        }
        if (pf_grp_ev) { cudaEventDestroy(pf_grp_ev); pf_grp_ev = nullptr; }
        if (pf_ro_ev) { cudaEventDestroy(pf_ro_ev); pf_ro_ev = nullptr; }
        if (pf_x_ev) { cudaEventDestroy(pf_x_ev); pf_x_ev = nullptr; }
        if (pf_xhad_ev) { cudaEventDestroy(pf_xhad_ev); pf_xhad_ev = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (pf_cpu_scat_ev[b]) { cudaEventDestroy(pf_cpu_scat_ev[b]); pf_cpu_scat_ev[b] = nullptr; }
            if (pf_h_cpu_fc2[b]) { cudaFreeHost(pf_h_cpu_fc2[b]); pf_h_cpu_fc2[b] = nullptr; pf_d_cpu_fc2[b] = nullptr; }
            if (pf_h_cpu_ridx[b]) { cudaFreeHost(pf_h_cpu_ridx[b]); pf_h_cpu_ridx[b] = nullptr; pf_d_cpu_ridx[b] = nullptr; }
        }
        if (pf_h_xhad) { cudaFreeHost(pf_h_xhad); pf_h_xhad = nullptr; }
        if (pf_h_ro_routes) { cudaFreeHost(pf_h_ro_routes); pf_h_ro_routes = nullptr; }
        if (pf_h_ro_topkw) { cudaFreeHost(pf_h_ro_topkw); pf_h_ro_topkw = nullptr; }
        if (pf_hot_h2d_ev) { cudaEventDestroy(pf_hot_h2d_ev); pf_hot_h2d_ev = nullptr; }
        if (pf_patch_ev) { cudaEventDestroy(pf_patch_ev); pf_patch_ev = nullptr; }
        if (pf_h_ro_ng) { cudaFreeHost(pf_h_ro_ng); pf_h_ro_ng = nullptr; }
        if (pf_h_ro_gexp) { cudaFreeHost(pf_h_ro_gexp); pf_h_ro_gexp = nullptr; }
        if (pf_h_ro_gcnt) { cudaFreeHost(pf_h_ro_gcnt); pf_h_ro_gcnt = nullptr; }
        if (pf_h_ro_goff) { cudaFreeHost(pf_h_ro_goff); pf_h_ro_goff = nullptr; }
        if (pf_h_hot_ng) { cudaFreeHost(pf_h_hot_ng); pf_h_hot_ng = nullptr; }
        if (pf_h_hot_gexp) { cudaFreeHost(pf_h_hot_gexp); pf_h_hot_gexp = nullptr; }
        if (pf_h_hot_gcnt) { cudaFreeHost(pf_h_hot_gcnt); pf_h_hot_gcnt = nullptr; }
        if (pf_h_hot_goff) { cudaFreeHost(pf_h_hot_goff); pf_h_hot_goff = nullptr; }
        for (auto& ld : layers) {
            if (ld.d_ptab) { cudaFree(ld.d_ptab); ld.d_ptab = nullptr; }
        }
        layers.clear();
        cpu_pool.shutdown();
        if (vram_pool) { cudaFree(vram_pool); vram_pool = nullptr; }
        if (d_is_cold) { cudaFree(d_is_cold); d_is_cold = nullptr; }
        if (d_exp_hits) { cudaFree(d_exp_hits); d_exp_hits = nullptr; }
        if (h_exp_hits) { cudaFreeHost(h_exp_hits); h_exp_hits = nullptr; }
        if (d_patches) { cudaFree(d_patches); d_patches = nullptr; }
        if (h_patches) { cudaFreeHost(h_patches); h_patches = nullptr; }
    }

    const uint8_t* const* layer_ptab(int li) const { return layers[size_t(li)].d_ptab; }
    uint32_t* layer_hits(int li) const { return d_exp_hits ? (d_exp_hits + size_t(li) * NE) : nullptr; }
    const uint8_t* layer_is_cold(int li) const { return d_is_cold ? (d_is_cold + size_t(li) * NE) : nullptr; }
    CpuMoeBridge* cpu_bridge_dev() const { return cpu_pool.enabled ? cpu_pool.d_bridge : nullptr; }
    float cpu_moe_frac() const { return cpu_pool.enabled ? cpu_pool.cpu_frac : 0.0f; }

    void init_cpu_pool(const std::vector<const uint8_t*>& h_fc1_vec,
                       const std::vector<const uint8_t*>& h_fc2_vec,
                       const std::vector<std::vector<long long>>& h_tabs,
                       const std::vector<std::vector<long long>>& h_kmixs,
                       const std::vector<const __half*>& d_sg_vec,
                       const std::vector<const __half*>& d_v1b_vec,
                       const std::vector<const __half*>& d_v2_vec,
                       const std::vector<const __half*>& d_v3b_vec,
                       const std::vector<size_t>& v1b_cnt,
                       const std::vector<size_t>& v2_cnt,
                       const std::vector<size_t>& v3b_cnt) {
        if (!cfg.cpu_moe || cfg.cpu_moe_threads <= 0 || cfg.cpu_moe_frac <= 0.0f) return;
        // Only enable for pure K3 experts (every expert's fc1/fc2 bit width == 3)
        bool all_k3 = true;
        for (int li = 0; li < n_layers && all_k3; ++li) {
            const auto& tab = h_tabs[size_t(li)];
            const auto& kmx = h_kmixs[size_t(li)];
            for (int e = 0; e < NE && all_k3; ++e) {
                if (tab[size_t(e) * 6 + 4] != 3) all_k3 = false;
                if (!kmx.empty() && kmx[size_t(e)] != 0) {
                    const int nb = int(tab[size_t(e) * 6 + 3]);
                    for (int u = 0; u < 2 * nb; ++u) {
                        if (((kmx[size_t(e)] >> (4 * u)) & 15) != 3) { all_k3 = false; break; }
                    }
                }
            }
        }
        if (!all_k3) {
            std::fprintf(stderr, "# qw38 strata CPU hybrid K3 MoE: skipped (non-K3 experts present)\n");
            return;
        }
        cpu_pool.init(n_layers, eb, h_fc1_vec, h_fc2_vec, h_tabs, h_kmixs,
                      d_sg_vec, d_v1b_vec, d_v2_vec, d_v3b_vec,
                      v1b_cnt, v2_cnt, v3b_cnt, cfg.cpu_moe_threads, cfg.cpu_moe_frac);
    }

    void set_expert_ptab_host(int li, int e, const uint8_t* fc1_base, const uint8_t* fc2_base) {
        auto& ld = layers[size_t(li)];
        const auto& em = ld.exp[size_t(e)];
        for (int b = 0; b < em.nb; ++b) {
            const int sl = (e * ld.eb + b) * 2;
            ld.h_ptab[size_t(sl)]     = fc1_base + em.piece_fc1_off[b];
            ld.h_ptab[size_t(sl) + 1] = fc2_base + em.piece_fc2_off[b];
        }
    }

    // Call once after all layers' tr3 metadata and host-pinned experts are loaded, and after all other GPU allocations!
    void init(const StrataConfig& c, int nl, int eb_in, int max_pf_rows,
              const std::vector<int>& layer_ids,
              const std::vector<const uint8_t*>& h_fc1_vec,
              const std::vector<const uint8_t*>& d_fc1_vec,
              const std::vector<const uint8_t*>& h_fc2_vec,
              const std::vector<const uint8_t*>& d_fc2_vec,
              const std::vector<std::vector<long long>>& h_tabs,
              const std::vector<std::vector<long long>>& h_kmixs,
              const std::vector<int>& rows_vec,
              cudaStream_t main_st) {
        cfg = c;
        n_layers = nl;
        eb = eb_in;
        layers.resize(size_t(nl));
        slot_stride = 0;
        fc1_max_aligned = 0;

        constexpr size_t MB = static_cast<size_t>(geom::KSH) * 8 * 32;
        for (int li = 0; li < nl; ++li) {
            auto& ld = layers[size_t(li)];
            ld.layer_id = layer_ids[size_t(li)];
            ld.rows = rows_vec[size_t(li)];
            ld.eb = eb;
            ld.h_fc1 = h_fc1_vec[size_t(li)];
            ld.d_fc1 = d_fc1_vec[size_t(li)];
            ld.h_fc2 = h_fc2_vec[size_t(li)];
            ld.d_fc2 = d_fc2_vec[size_t(li)];
            ld.h_ptab.assign(size_t(ld.rows) * size_t(eb) * 2, nullptr);
            ld.exp.resize(size_t(ld.rows));

            const auto& tab = h_tabs[size_t(li)];
            const auto& kmx = h_kmixs[size_t(li)];
            for (int e = 0; e < ld.rows; ++e) {
                const long long* tr = &tab[size_t(e) * 6];
                const long long km = kmx.empty() ? 0 : kmx[size_t(e)];
                auto& em = ld.exp[size_t(e)];
                em.nb = static_cast<int>(tr[3]);
                em.fc1_off = static_cast<size_t>(tr[0]);
                em.fc2_off = static_cast<size_t>(tr[1]);
                size_t f1_cur = 0, f2_cur = 0;
                const size_t f2_pb = tr3_piece_fc2_bytes(tr);
                for (int b = 0; b < em.nb; ++b) {
                    em.piece_fc1_off[b] = f1_cur;
                    em.piece_fc2_off[b] = f2_cur;
                    f1_cur += tr3_piece_fc1_bytes(km, tr, b);
                    f2_cur += f2_pb;
                }
                em.fc1_bytes = f1_cur;
                em.fc2_bytes = f2_cur;
                const size_t f1_al = round256(em.fc1_bytes);
                const size_t tot_al = f1_al + round256(em.fc2_bytes);
                if (e < NE) {
                    if (f1_al > fc1_max_aligned) fc1_max_aligned = f1_al;
                    if (tot_al > slot_stride) slot_stride = tot_al;
                }
                // Default: point to mapped pinned host RAM UVA
                set_expert_ptab_host(li, e, ld.d_fc1 + em.fc1_off, ld.d_fc2 + em.fc2_off);
            }
            if (cudaMalloc(&ld.d_ptab, ld.h_ptab.size() * sizeof(void*)) != cudaSuccess) {
                std::fprintf(stderr, "qw38 strata: cudaMalloc d_ptab failed\n");
                std::exit(2);
            }
        }
        (void)MB;

        // Allocate small control & prefill staging buffers BEFORE sizing vram_pool
        cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&promo_ev, cudaEventDisableTiming);

        const size_t total_exps = size_t(nl) * NE;
        cudaMalloc(&d_exp_hits, total_exps * sizeof(uint32_t));
        cudaMemset(d_exp_hits, 0, total_exps * sizeof(uint32_t));
        cudaMalloc(&d_is_cold, total_exps * sizeof(uint8_t));
        cudaMemset(d_is_cold, 1, total_exps * sizeof(uint8_t));
        cudaHostAlloc(&h_exp_hits, total_exps * sizeof(uint32_t), cudaHostAllocDefault);
        std::memset(h_exp_hits, 0, total_exps * sizeof(uint32_t));

        const int max_batch_swaps = std::max({cfg.promote_max_experts, cfg.pf_d2d_promote, 48});
        max_patches = std::max(64, max_batch_swaps * 2 * eb * 2 + 64);
        cudaMalloc(&d_patches, size_t(max_patches) * sizeof(PtabPatchEntry));
        cudaHostAlloc(&h_patches, size_t(max_patches) * sizeof(PtabPatchEntry), cudaHostAllocDefault);

        // Prefill double-buffered H2D staging buffers
        const int pf_se = std::max(8, cfg.pf_stage_experts);
        cfg.pf_stage_experts = pf_se;
        const size_t pf_stg_bytes = size_t(pf_se) * slot_stride;
        const int max_routes = max_pf_rows * geom::TOPK;
        pf_max_groups = std::max(1024, std::min(max_routes, NE + (max_routes + 15) / 16) + 64);
        pf_ptab_n = size_t(layers[0].rows) * size_t(eb) * 2;
        for (int b = 0; b < 2; ++b) {
            cudaEventCreateWithFlags(&pf_h2d_ev[b], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&pf_comp_ev[b], cudaEventDisableTiming);
            cudaMalloc(&pf_stg[b], pf_stg_bytes);
            cudaMalloc(&pf_ptab[b], pf_ptab_n * sizeof(void*));
            cudaMalloc(&pf_d_ng[b], sizeof(int));
            cudaMalloc(&pf_d_gexp[b], size_t(pf_max_groups) * sizeof(int));
            cudaMalloc(&pf_d_gcnt[b], size_t(pf_max_groups) * sizeof(int));
            cudaMalloc(&pf_d_goff[b], size_t(pf_max_groups) * sizeof(int));
            cudaHostAlloc(&pf_h_gexp[b], size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc(&pf_h_gcnt[b], size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc(&pf_h_goff[b], size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc(&pf_h_ptab[b], pf_ptab_n * sizeof(void*), cudaHostAllocDefault);
            cudaHostAlloc(&pf_h_ng[b], sizeof(int), cudaHostAllocDefault);
            pf_h2d_pending[b] = false;
            pf_comp_pending[b] = false;
        }
        cudaEventCreateWithFlags(&pf_grp_ev, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&pf_ro_ev, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&pf_x_ev, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&pf_xhad_ev, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&pf_hot_h2d_ev, cudaEventDisableTiming);
        pf_hot_h2d_pending = false;
        cudaEventCreateWithFlags(&pf_patch_ev, cudaEventDisableTiming);
        pf_patch_pending = false;
        cudaHostAlloc(&pf_h_ro_ng, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_ro_gexp, size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_ro_gcnt, size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_ro_goff, size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_hot_ng, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_hot_gexp, size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_hot_gcnt, size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_hot_goff, size_t(pf_max_groups) * sizeof(int), cudaHostAllocDefault);
        // Small/medium prefill CPU hybrid buffers (pinned host RAM only, 0 GPU VRAM)
        cudaHostAlloc(&pf_h_xhad, size_t(CpuMoePool::MAX_PACK_M) * geom::HID * sizeof(uint16_t), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_ro_routes, size_t(CpuMoePool::MAX_PACK_M) * geom::TOPK * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&pf_h_ro_topkw, size_t(CpuMoePool::MAX_PACK_M) * geom::TOPK * sizeof(float), cudaHostAllocDefault);
        for (int b = 0; b < 2; ++b) {
            cudaEventCreateWithFlags(&pf_cpu_scat_ev[b], cudaEventDisableTiming);
            pf_cpu_scat_pending[b] = false;
            cudaHostAlloc(&pf_h_cpu_fc2[b], size_t(PF_CPU_MAX_ROUTES) * geom::HID * sizeof(uint16_t), cudaHostAllocMapped);
            cudaHostGetDevicePointer(&pf_d_cpu_fc2[b], pf_h_cpu_fc2[b], 0);
            cudaHostAlloc(&pf_h_cpu_ridx[b], size_t(PF_CPU_MAX_ROUTES) * sizeof(int), cudaHostAllocMapped);
            cudaHostGetDevicePointer(&pf_d_cpu_ridx[b], pf_h_cpu_ridx[b], 0);
        }

        // Size VRAM hot-expert pool from remaining free GPU memory
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve_b = static_cast<size_t>(std::max<int64_t>(0, cfg.vram_reserve_mib)) << 20;
        size_t budget_b = (free_b > reserve_b) ? (free_b - reserve_b) : 0;
        int want_slots = (slot_stride > 0) ? static_cast<int>(budget_b / slot_stride) : 0;
        if (cfg.cache_slots_override >= 0) {
            want_slots = std::min(want_slots, static_cast<int>(cfg.cache_slots_override));
        }
        want_slots = std::min(want_slots, nl * NE);
        total_slots = std::max(0, want_slots);
        if (cfg.adaptive && total_slots < nl * NE && total_slots >= 64) {
            ring_slots = std::min(cfg.promote_max_experts, std::max(8, total_slots / 32));
        } else {
            ring_slots = 0;
        }
        active_slots = total_slots - ring_slots;

        if (total_slots > 0) {
            const size_t pool_bytes = size_t(total_slots) * slot_stride;
            if (cudaMalloc(&vram_pool, pool_bytes) != cudaSuccess) {
                std::fprintf(stderr, "qw38 strata: cudaMalloc vram_pool %.2f GiB failed!\n", pool_bytes / 1073741824.0);
                std::exit(2);
            }
        }

        exp_slot.assign(total_exps, -1);
        slot_owner.assign(size_t(total_slots), -1);
        slot_pinned.assign(size_t(total_slots), 0);
        exp_pending.assign(total_exps, 0);
        heat.assign(total_exps, 0.0f);
        free_slots.clear();
        for (int s = active_slots; s < total_slots; ++s) free_slots.push_back(s);
        pinned_slots = 0;

        // Build initial fill order: profile-ranked if available, else round-robin across layers
        std::vector<std::pair<int, int>> order;
        const bool have_prof = read_strata_expert_profile(cfg.profile_path, geom::L_MAIN, NE, order);
        std::vector<int> layer_to_li(geom::L_MAIN, -1);
        for (int li = 0; li < nl; ++li) {
            if (layer_ids[size_t(li)] >= 0 && layer_ids[size_t(li)] < geom::L_MAIN)
                layer_to_li[size_t(layer_ids[size_t(li)])] = li;
        }
        std::vector<std::pair<int, int>> fill;
        fill.reserve(size_t(active_slots));
        std::vector<uint8_t> seen(total_exps, 0);
        if (active_slots > 0 && have_prof) {
            for (const auto& le : order) {
                const int li = layer_to_li[size_t(le.first)];
                if (li < 0) continue;
                const int idx = li * NE + le.second;
                if (!seen[size_t(idx)]) {
                    seen[size_t(idx)] = 1;
                    fill.emplace_back(li, le.second);
                    if (static_cast<int>(fill.size()) >= active_slots) break;
                }
            }
        }
        // Top up any remaining active slots round-robin across layers
        for (int e = 0; e < NE && static_cast<int>(fill.size()) < active_slots; ++e) {
            for (int li = 0; li < nl && static_cast<int>(fill.size()) < active_slots; ++li) {
                const int idx = li * NE + e;
                if (!seen[size_t(idx)]) {
                    seen[size_t(idx)] = 1;
                    fill.emplace_back(li, e);
                }
            }
        }

        // Pre-fill VRAM hot-expert pool asynchronously on h2d_stream!
        const int pin_count = (have_prof && active_slots > 0)
            ? std::max(0, std::min(active_slots, static_cast<int>(std::lround(double(active_slots) * double(std::clamp(cfg.static_pin_frac, 0.0f, 0.95f))))))
            : 0;
        pinned_slots = pin_count;
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < static_cast<int>(fill.size()); ++s) {
            const int li = fill[size_t(s)].first;
            const int e = fill[size_t(s)].second;
            const int idx = li * NE + e;
            const auto& ld = layers[size_t(li)];
            const auto& em = ld.exp[size_t(e)];
            uint8_t* dst = vram_pool + size_t(s) * slot_stride;
            uint8_t* dst_fc2 = dst + round256(em.fc1_bytes);
            cudaMemcpyAsync(dst, ld.h_fc1 + em.fc1_off, em.fc1_bytes, cudaMemcpyHostToDevice, h2d_stream);
            cudaMemcpyAsync(dst_fc2, ld.h_fc2 + em.fc2_off, em.fc2_bytes, cudaMemcpyHostToDevice, h2d_stream);
            set_expert_ptab_host(li, e, dst, dst_fc2);
            exp_slot[size_t(idx)] = s;
            slot_owner[size_t(s)] = idx;
            if (s < pin_count) {
                slot_pinned[size_t(s)] = 1;
                heat[size_t(idx)] = cfg.heat_cap;
            } else {
                const int dyn_n = std::max(1, static_cast<int>(fill.size()) - pin_count);
                const float dfrac = static_cast<float>(s - pin_count) / static_cast<float>(dyn_n);
                heat[size_t(idx)] = 1.20f - 1.00f * dfrac;
            }
        }
        cudaStreamSynchronize(h2d_stream);
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // Upload all layers' initial fixed-address d_ptab tables and initial cold bitmap
        std::vector<uint8_t> h_cold(total_exps, 1);
        for (size_t i = 0; i < total_exps; ++i) h_cold[i] = (exp_slot[i] >= 0) ? 0 : 1;
        cudaMemcpyAsync(d_is_cold, h_cold.data(), total_exps * sizeof(uint8_t), cudaMemcpyHostToDevice, main_st);
        for (int li = 0; li < nl; ++li) {
            auto& ld = layers[size_t(li)];
            cudaMemcpyAsync(ld.d_ptab, ld.h_ptab.data(), ld.h_ptab.size() * sizeof(void*), cudaMemcpyHostToDevice, main_st);
        }
        cudaStreamSynchronize(main_st);

        size_t free_after = 0;
        cudaMemGetInfo(&free_after, &total_b);
        std::fprintf(stderr,
                     "# qw38 strata cache: %d/%d experts in VRAM (%.1f%%, %d active [%d pinned + %d dyn] + %d ring, %.2f GiB pool, "
                     "slot=%zu B, profile=%s, prefill-filled in %.2f s @ %.1f GB/s; %.0f MiB VRAM free)\n",
                     active_slots, nl * NE, 100.0 * active_slots / double(nl * NE),
                     active_slots, pinned_slots, active_slots - pinned_slots, ring_slots,
                     (double(total_slots) * slot_stride) / 1073741824.0,
                     slot_stride, have_prof ? cfg.profile_path.c_str() : "none",
                     fill_s, fill_s > 1e-6 ? ((double(fill.size()) * slot_stride / 1e9) / fill_s) : 0.0,
                     free_after / 1048576.0);
    }

    // Wait for any in-flight async promotion and commit its d_ptab patches on `st`.
    void commit_ready_promotions(cudaStream_t st, bool blocking) {
        if (!promo_in_flight) return;
        if (!blocking && cudaEventQuery(promo_ev) != cudaSuccess) return;
        if (blocking) cudaEventSynchronize(promo_ev);
        if (pf_patch_pending) {
            cudaEventSynchronize(pf_patch_ev);
            pf_patch_pending = false;
        }
        int np = 0;
        for (const auto& sw : pending_swaps) {
            // 1. Evict old_idx back to UVA pinned host RAM
            if (sw.old_idx >= 0) {
                const int ol = sw.old_idx / NE, oe = sw.old_idx % NE;
                auto& old_ld = layers[size_t(ol)];
                const auto& old_em = old_ld.exp[size_t(oe)];
                const uint8_t* uva1 = old_ld.d_fc1 + old_em.fc1_off;
                const uint8_t* uva2 = old_ld.d_fc2 + old_em.fc2_off;
                for (int b = 0; b < old_em.nb; ++b) {
                    const int sl = (oe * old_ld.eb + b) * 2;
                    h_patches[np++] = {old_ld.d_ptab + sl,     uva1 + old_em.piece_fc1_off[b],
                                       (b == 0) ? (d_is_cold + size_t(sw.old_idx)) : nullptr, 1};
                    h_patches[np++] = {old_ld.d_ptab + sl + 1, uva2 + old_em.piece_fc2_off[b], nullptr, 0};
                    old_ld.h_ptab[size_t(sl)]     = uva1 + old_em.piece_fc1_off[b];
                    old_ld.h_ptab[size_t(sl) + 1] = uva2 + old_em.piece_fc2_off[b];
                }
                exp_slot[size_t(sw.old_idx)] = -1;
                slot_owner[size_t(sw.old_slot)] = -1;
                free_slots.push_back(sw.old_slot);
            }
            // 2. Point new_idx to its ready VRAM slot
            {
                const int nl_i = sw.new_idx / NE, ne_i = sw.new_idx % NE;
                auto& new_ld = layers[size_t(nl_i)];
                const auto& new_em = new_ld.exp[size_t(ne_i)];
                const uint8_t* v1 = vram_pool + size_t(sw.stg_slot) * slot_stride;
                const uint8_t* v2 = v1 + round256(new_em.fc1_bytes);
                for (int b = 0; b < new_em.nb; ++b) {
                    const int sl = (ne_i * new_ld.eb + b) * 2;
                    h_patches[np++] = {new_ld.d_ptab + sl,     v1 + new_em.piece_fc1_off[b],
                                       (b == 0) ? (d_is_cold + size_t(sw.new_idx)) : nullptr, 0};
                    h_patches[np++] = {new_ld.d_ptab + sl + 1, v2 + new_em.piece_fc2_off[b], nullptr, 0};
                    new_ld.h_ptab[size_t(sl)]     = v1 + new_em.piece_fc1_off[b];
                    new_ld.h_ptab[size_t(sl) + 1] = v2 + new_em.piece_fc2_off[b];
                }
                exp_slot[size_t(sw.new_idx)] = sw.stg_slot;
                slot_owner[size_t(sw.stg_slot)] = sw.new_idx;
                exp_pending[size_t(sw.new_idx)] = 0;
            }
            ++stat_swaps;
        }
        pending_swaps.clear();
        promo_in_flight = false;
        if (np > 0) {
            cudaMemcpyAsync(d_patches, h_patches, size_t(np) * sizeof(PtabPatchEntry), cudaMemcpyHostToDevice, st);
            k_strata_ptab_patch<<<(np + 255) / 256, 256, 0, st>>>(d_patches, np);
        }
    }

    // Call between decode/verify steps AFTER main `st` has synchronized!
    void maintain_after_step(cudaStream_t st) {
        if (n_layers == 0 || !d_exp_hits) return;
        const size_t total_exps = size_t(n_layers) * NE;

        // 1. Commit any completed async promotions from previous step
        commit_ready_promotions(st, !cfg.async_promote);

        // 2. Read & reset GPU hit counters (~96 KB -> ~4 us!)
        cudaMemcpy(h_exp_hits, d_exp_hits, total_exps * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemsetAsync(d_exp_hits, 0, total_exps * sizeof(uint32_t), st);

        const float dec = cfg.heat_decay;
        const float cap = cfg.heat_cap;
        uint64_t step_hits = 0, step_misses = 0;
        for (size_t i = 0; i < total_exps; ++i) {
            const uint32_t c = h_exp_hits[i];
            float h = heat[i] * dec;
            if (c > 0) {
                h += static_cast<float>(c);
                if (exp_slot[i] >= 0) step_hits += c;
                else step_misses += c;
            }
            if (h > cap) h = cap;
            heat[i] = h;
        }
        stat_hits += step_hits;
        stat_misses += step_misses;
        if (step_hits + step_misses > 0) {
            ++stat_steps;
            if ((stat_steps % 50) == 0) {
                print_stats();
            }
        }

        if (!cfg.adaptive || total_slots == 0 || total_slots >= static_cast<int>(total_exps) || promo_in_flight || free_slots.empty()) {
            return;
        }

        // 3. Pick top cold candidates and coldest unpinned resident victims globally
        const int max_k = std::min(cfg.promote_max_experts, static_cast<int>(free_slots.size()));
        if (max_k <= 0) return;

        // Collect hot cold-candidates with heat >= min_promote_heat
        std::vector<std::pair<float, int>> cold_cands;
        cold_cands.reserve(64);
        for (size_t i = 0; i < total_exps; ++i) {
            if (exp_slot[i] < 0 && !exp_pending[i] && heat[i] >= cfg.min_promote_heat) {
                cold_cands.emplace_back(heat[i], static_cast<int>(i));
            }
        }
        if (cold_cands.empty()) return;
        const int n_cold = std::min(max_k, static_cast<int>(cold_cands.size()));
        std::partial_sort(cold_cands.begin(), cold_cands.begin() + n_cold, cold_cands.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Find the `n_cold` coldest currently-resident UNPINNED slots
        std::vector<std::pair<float, int>> hot_victims;
        hot_victims.reserve(size_t(std::max(0, active_slots - pinned_slots)));
        for (int s = 0; s < total_slots; ++s) {
            const int owner = slot_owner[size_t(s)];
            if (owner >= 0 && !slot_pinned[size_t(s)]) {
                hot_victims.emplace_back(heat[size_t(owner)], s);
            }
        }
        if (hot_victims.empty()) return;
        const int n_vic = std::min(n_cold, static_cast<int>(hot_victims.size()));
        std::partial_sort(hot_victims.begin(), hot_victims.begin() + n_vic, hot_victims.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

        // Pair best cold with coldest victim if cold_heat > victim_heat * promote_hyst
        pending_swaps.clear();
        for (int k = 0; k < n_vic; ++k) {
            const float c_heat = cold_cands[size_t(k)].first;
            const int c_idx = cold_cands[size_t(k)].second;
            const float v_heat = hot_victims[size_t(k)].first;
            const int v_slot = hot_victims[size_t(k)].second;
            const int v_idx = slot_owner[size_t(v_slot)];
            if (c_heat <= v_heat * cfg.promote_hyst) break;
            if (free_slots.empty()) break;

            const int stg = free_slots.back();
            free_slots.pop_back();
            const int cli = c_idx / NE, ce = c_idx % NE;
            const auto& cld = layers[size_t(cli)];
            const auto& cem = cld.exp[size_t(ce)];
            uint8_t* dst = vram_pool + size_t(stg) * slot_stride;
            uint8_t* dst2 = dst + round256(cem.fc1_bytes);
            cudaMemcpyAsync(dst, cld.h_fc1 + cem.fc1_off, cem.fc1_bytes, cudaMemcpyHostToDevice, h2d_stream);
            cudaMemcpyAsync(dst2, cld.h_fc2 + cem.fc2_off, cem.fc2_bytes, cudaMemcpyHostToDevice, h2d_stream);
            exp_pending[size_t(c_idx)] = 1;
            pending_swaps.push_back({c_idx, v_idx, stg, v_slot});
        }

        if (!pending_swaps.empty()) {
            cudaEventRecord(promo_ev, h2d_stream);
            promo_in_flight = true;
            if (!cfg.async_promote) {
                commit_ready_promotions(st, true);
            }
        }
    }

    // Double-buffered H2D DMA prefill for M >= 32 (+ parallel CPU AVX-512 cold offload for M <= 512).
    template <int PF_GRP, int HB_PER_CTA>
    void prefill_moe_layer_impl(int li, const qw38::MoeWeights& w, qw38::MoeScratch& sc,
                                const __nv_bfloat16* x, __nv_bfloat16* out_part, int M, cudaStream_t st) {
        commit_ready_promotions(st, true);
        // Domain-switch dynamic heat decay ONCE per new prompt prefill (never on subsequent chunks of the same prompt,
        // and never touching the pinned static core!)
        if (li == 0 && M >= 64) {
            if (stat_steps > 0) print_stats();
            if (cfg.adaptive && cfg.pf_warm_heat && stat_steps != last_pf_decay_step) {
                last_pf_decay_step = stat_steps;
                for (int s = 0; s < total_slots; ++s) {
                    const int owner = slot_owner[size_t(s)];
                    if (owner >= 0 && !slot_pinned[size_t(s)]) heat[size_t(owner)] *= 0.50f;
                }
            }
        }
        const int n_routes = M * geom::TOPK;
        const bool use_pf_cpu = (M <= cfg.pf_cpu_max_m && M <= CpuMoePool::MAX_PACK_M &&
                                 cpu_pool.enabled && cfg.pf_cpu_frac > 0.0f);

        // Launch group builder first so D2H metadata readout overlaps behind k_qw38_moe_prerot_x!
        qw38::k_qw38_moe_build_groups_pf<PF_GRP><<<1, geom::NEXP, 0, st>>>(
            sc.topk_idx, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_off, sc.grp_routes, n_routes);
        cudaEventRecord(pf_grp_ev, st);
        qw38::k_qw38_moe_prerot_x<<<M * geom::HB, 32, 0, st>>>(x, w.t_sg, sc.x_had, M);
        if (use_pf_cpu) cudaEventRecord(pf_x_ev, st);

        const int copy_g = std::min({pf_max_groups, n_routes, NE + (n_routes + PF_GRP - 1) / PF_GRP});
        cudaStreamWaitEvent(h2d_stream, pf_grp_ev, 0);
        cudaMemcpyAsync(pf_h_ro_ng, sc.d_n_groups, sizeof(int), cudaMemcpyDeviceToHost, h2d_stream);
        cudaMemcpyAsync(pf_h_ro_gexp, sc.grp_exp, size_t(copy_g) * sizeof(int), cudaMemcpyDeviceToHost, h2d_stream);
        cudaMemcpyAsync(pf_h_ro_gcnt, sc.grp_cnt, size_t(copy_g) * sizeof(int), cudaMemcpyDeviceToHost, h2d_stream);
        cudaMemcpyAsync(pf_h_ro_goff, sc.grp_off, size_t(copy_g) * sizeof(int), cudaMemcpyDeviceToHost, h2d_stream);
        if (use_pf_cpu) {
            cudaMemcpyAsync(pf_h_ro_routes, sc.grp_routes, size_t(n_routes) * sizeof(int), cudaMemcpyDeviceToHost, h2d_stream);
            cudaMemcpyAsync(pf_h_ro_topkw, sc.topk_w, size_t(n_routes) * sizeof(float), cudaMemcpyDeviceToHost, h2d_stream);
        }
        cudaEventRecord(pf_ro_ev, h2d_stream);
        if (use_pf_cpu) {
            cudaStreamWaitEvent(h2d_stream, pf_x_ev, 0);
            cudaMemcpyAsync(pf_h_xhad, sc.x_had, size_t(M) * geom::HID * sizeof(uint16_t), cudaMemcpyDeviceToHost, h2d_stream);
            cudaEventRecord(pf_xhad_ev, h2d_stream);
        }
        cudaEventSynchronize(pf_ro_ev);

        const int n_groups = *pf_h_ro_ng;
        if (n_groups <= 0 || n_groups > copy_g) {
            qw38::launch_moe_pf_wide<PF_GRP, HB_PER_CTA>(w, sc, x, out_part, M, st);
            return;
        }

        // First scan: accumulate per-expert token counts and separate hot vs cold groups
        std::vector<int> hot_grps;
        hot_grps.reserve(size_t(n_groups));
        std::vector<int> cold_grps;
        cold_grps.reserve(size_t(n_groups));
        int exp_tok_cnt[NE];
        int exp_first_grp[NE];
        std::memset(exp_tok_cnt, 0, sizeof(exp_tok_cnt));
        for (int e = 0; e < NE; ++e) exp_first_grp[e] = -1;
        int n_cold_unique = 0;
        for (int g = 0; g < n_groups; ++g) {
            const int e = pf_h_ro_gexp[g];
            if (e >= 0 && e < NE) {
                if (exp_tok_cnt[e] == 0 && exp_slot[size_t(li) * NE + size_t(e)] < 0) ++n_cold_unique;
                exp_tok_cnt[e] += pf_h_ro_gcnt[g];
                if (exp_first_grp[e] < 0) exp_first_grp[e] = g;
            }
            if (exp_slot[size_t(li) * NE + size_t(e)] >= 0) {
                hot_grps.push_back(g);
            } else {
                cold_grps.push_back(g);
            }
        }

        // Plan zero-PCIe GDDR7 D2D promotions for this layer's hottest prompt-routed cold experts
        // (each expert is already staged in pf_stg[buf] VRAM -> 1.2 us D2D copy, 0 PCIe bytes!)
        int d2d_slot[NE];
        int d2d_vidx[NE];
        float d2d_score[NE];
        for (int e = 0; e < NE; ++e) { d2d_slot[e] = -1; d2d_vidx[e] = -1; d2d_score[e] = 0.0f; }
        bool have_d2d = false;
        if (cfg.adaptive && cfg.pf_warm_heat && active_slots > pinned_slots && active_slots < n_layers * NE) {
            const float avg_cnt = std::max(1.0f, static_cast<float>(M * geom::TOPK) / static_cast<float>(NE));
            const int min_pf_cnt = std::max(2, static_cast<int>(std::ceil(avg_cnt * 1.15f)));
            std::vector<std::pair<float, int>> pf_cands;
            pf_cands.reserve(64);
            for (int e = 0; e < NE; ++e) {
                const int cnt = exp_tok_cnt[e];
                if (cnt <= 0) continue;
                const int idx = li * NE + e;
                const float ratio = static_cast<float>(cnt) / avg_cnt;
                const float score = std::min(cfg.heat_cap, 2.5f + 3.5f * std::log1pf(ratio));
                if (exp_slot[size_t(idx)] >= 0) {
                    if (score > heat[size_t(idx)]) heat[size_t(idx)] = score;
                } else if (cfg.pf_d2d_promote > 0 && cnt >= min_pf_cnt) {
                    pf_cands.emplace_back(score, e);
                }
            }
            if (!pf_cands.empty()) {
                const int max_p = std::min({cfg.pf_d2d_promote, static_cast<int>(pf_cands.size()), 64});
                std::partial_sort(pf_cands.begin(), pf_cands.begin() + max_p, pf_cands.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                std::vector<std::pair<float, int>> victims;
                victims.reserve(size_t(std::max(0, active_slots - pinned_slots)));
                for (int s = 0; s < total_slots; ++s) {
                    const int owner = slot_owner[size_t(s)];
                    if (owner >= 0 && !slot_pinned[size_t(s)]) {
                        victims.emplace_back(heat[size_t(owner)], s);
                    }
                }
                const int n_vic = std::min(max_p, static_cast<int>(victims.size()));
                if (n_vic > 0) {
                    std::partial_sort(victims.begin(), victims.begin() + n_vic, victims.end(),
                                      [](const auto& a, const auto& b) { return a.first < b.first; });
                    for (int k = 0; k < n_vic; ++k) {
                        const float c_score = pf_cands[size_t(k)].first;
                        const int ce = pf_cands[size_t(k)].second;
                        const float v_heat = victims[size_t(k)].first;
                        const int v_slot = victims[size_t(k)].second;
                        if (c_score <= v_heat * cfg.promote_hyst) break;
                        d2d_slot[ce] = v_slot;
                        d2d_vidx[ce] = slot_owner[size_t(v_slot)];
                        d2d_score[ce] = c_score;
                        have_d2d = true;
                    }
                }
            }
        }

        // For small/medium prompts (M <= 512), assign ~50% of smallest-cnt (<=4) non-D2D cold experts to CPU!
        uint8_t exp_to_cpu[NE];
        std::memset(exp_to_cpu, 0, sizeof(exp_to_cpu));
        int pf_cpu_exps[CpuMoePool::MAX_GROUPS];
        int pf_cpu_cnts[CpuMoePool::MAX_GROUPS];
        int pf_cpu_toks[CpuMoePool::MAX_GROUPS][8];
        int pf_cpu_out_rows[CpuMoePool::MAX_GROUPS][8];
        float pf_cpu_rws[CpuMoePool::MAX_GROUPS][8];
        int n_cpu_exps = 0;
        int n_cpu_routes = 0;
        const int cbuf = (pf_cpu_buf_idx++) & 1;

        if (use_pf_cpu && n_cold_unique >= 4) {
            std::vector<std::pair<int, int>> cpu_cands; // (cnt, expert)
            cpu_cands.reserve(size_t(n_cold_unique));
            for (int e = 0; e < NE; ++e) {
                if (exp_tok_cnt[e] > 0 && exp_tok_cnt[e] <= 4 &&
                    exp_slot[size_t(li) * NE + size_t(e)] < 0 && d2d_slot[e] < 0) {
                    cpu_cands.emplace_back(exp_tok_cnt[e], e);
                }
            }
            if (!cpu_cands.empty()) {
                const int want_cpu = std::min({static_cast<int>(cpu_cands.size()),
                                               CpuMoePool::MAX_GROUPS,
                                               static_cast<int>(std::lround(double(n_cold_unique) * double(cfg.pf_cpu_frac)))});
                if (want_cpu > 0) {
                    std::partial_sort(cpu_cands.begin(), cpu_cands.begin() + want_cpu, cpu_cands.end(),
                                      [](const auto& a, const auto& b) { return a.first < b.first; });
                    if (pf_cpu_scat_pending[cbuf]) {
                        cudaEventSynchronize(pf_cpu_scat_ev[cbuf]);
                        pf_cpu_scat_pending[cbuf] = false;
                    }
                    for (int k = 0; k < want_cpu; ++k) {
                        const int cnt = cpu_cands[size_t(k)].first;
                        const int e = cpu_cands[size_t(k)].second;
                        if (n_cpu_routes + cnt > PF_CPU_MAX_ROUTES) break;
                        const int g0 = exp_first_grp[e];
                        if (g0 < 0 || pf_h_ro_gcnt[g0] != cnt) continue;
                        const int off = pf_h_ro_goff[g0];
                        exp_to_cpu[e] = 1;
                        pf_cpu_exps[n_cpu_exps] = e;
                        pf_cpu_cnts[n_cpu_exps] = cnt;
                        for (int t = 0; t < cnt; ++t) {
                            const int r = pf_h_ro_routes[off + t];
                            pf_cpu_toks[n_cpu_exps][t] = r / geom::TOPK;
                            pf_cpu_out_rows[n_cpu_exps][t] = n_cpu_routes;
                            pf_cpu_rws[n_cpu_exps][t] = pf_h_ro_topkw[r];
                            pf_h_cpu_ridx[cbuf][n_cpu_routes] = r;
                            ++n_cpu_routes;
                        }
                        ++n_cpu_exps;
                    }
                }
            }
        }

        // Partition remaining GPU cold groups into double-buffered H2D batches
        struct ColdBatch {
            std::vector<int> exps;
            std::vector<int> grps;
        };
        std::vector<ColdBatch> batches;
        int cur_e = -1;
        for (int g : cold_grps) {
            const int e = pf_h_ro_gexp[g];
            if (exp_to_cpu[e]) continue;
            if (e != cur_e) {
                if (batches.empty() || static_cast<int>(batches.back().exps.size()) >= cfg.pf_stage_experts) {
                    batches.emplace_back();
                    batches.back().exps.reserve(size_t(cfg.pf_stage_experts));
                }
                batches.back().exps.push_back(e);
                cur_e = e;
            }
            batches.back().grps.push_back(g);
        }

        const auto& ld = layers[size_t(li)];
        auto stage_cold_batch = [&](size_t bi, int buf) {
            if (pf_h2d_pending[buf]) {
                cudaEventSynchronize(pf_h2d_ev[buf]);
                pf_h2d_pending[buf] = false;
            }
            if (pf_comp_pending[buf]) {
                cudaStreamWaitEvent(h2d_stream, pf_comp_ev[buf], 0);
                pf_comp_pending[buf] = false;
            }
            const auto& bt = batches[bi];
            std::memset(pf_h_ptab[buf], 0, pf_ptab_n * sizeof(void*));
            for (size_t k = 0; k < bt.exps.size(); ++k) {
                const int e = bt.exps[k];
                const auto& em = ld.exp[size_t(e)];
                uint8_t* dst = pf_stg[buf] + k * slot_stride;
                uint8_t* dst2 = dst + round256(em.fc1_bytes);
                cudaMemcpyAsync(dst, ld.h_fc1 + em.fc1_off, em.fc1_bytes, cudaMemcpyHostToDevice, h2d_stream);
                cudaMemcpyAsync(dst2, ld.h_fc2 + em.fc2_off, em.fc2_bytes, cudaMemcpyHostToDevice, h2d_stream);
                for (int b = 0; b < em.nb; ++b) {
                    const int sl = (e * ld.eb + b) * 2;
                    pf_h_ptab[buf][size_t(sl)]     = dst + em.piece_fc1_off[b];
                    pf_h_ptab[buf][size_t(sl) + 1] = dst2 + em.piece_fc2_off[b];
                }
            }
            const int ng = static_cast<int>(bt.grps.size());
            *pf_h_ng[buf] = ng;
            for (int i = 0; i < ng; ++i) {
                const int g = bt.grps[size_t(i)];
                pf_h_gexp[buf][i] = pf_h_ro_gexp[g];
                pf_h_gcnt[buf][i] = pf_h_ro_gcnt[g];
                pf_h_goff[buf][i] = pf_h_ro_goff[g];
            }
            cudaMemcpyAsync(pf_ptab[buf], pf_h_ptab[buf], pf_ptab_n * sizeof(void*), cudaMemcpyHostToDevice, h2d_stream);
            cudaMemcpyAsync(pf_d_ng[buf], pf_h_ng[buf], sizeof(int), cudaMemcpyHostToDevice, h2d_stream);
            cudaMemcpyAsync(pf_d_gexp[buf], pf_h_gexp[buf], size_t(ng) * sizeof(int), cudaMemcpyHostToDevice, h2d_stream);
            cudaMemcpyAsync(pf_d_gcnt[buf], pf_h_gcnt[buf], size_t(ng) * sizeof(int), cudaMemcpyHostToDevice, h2d_stream);
            cudaMemcpyAsync(pf_d_goff[buf], pf_h_goff[buf], size_t(ng) * sizeof(int), cudaMemcpyHostToDevice, h2d_stream);
            cudaEventRecord(pf_h2d_ev[buf], h2d_stream);
            pf_h2d_pending[buf] = true;
        };

        // 1. Kick off BOTH Cold Batch 0 (buf 0) AND Cold Batch 1 (buf 1) immediately so PCIe stays 100% saturated!
        if (!batches.empty()) stage_cold_batch(0, 0);
        if (batches.size() > 1) stage_cold_batch(1, 1);

        // 2. Run all VRAM-hot groups on `st` using dedicated host buffer + sc.grp_*
        if (!hot_grps.empty()) {
            if (pf_hot_h2d_pending) {
                cudaEventSynchronize(pf_hot_h2d_ev);
                pf_hot_h2d_pending = false;
            }
            const int ng = static_cast<int>(hot_grps.size());
            *pf_h_hot_ng = ng;
            for (int i = 0; i < ng; ++i) {
                const int g = hot_grps[size_t(i)];
                pf_h_hot_gexp[i] = pf_h_ro_gexp[g];
                pf_h_hot_gcnt[i] = pf_h_ro_gcnt[g];
                pf_h_hot_goff[i] = pf_h_ro_goff[g];
            }
            cudaMemcpyAsync(sc.d_n_groups, pf_h_hot_ng, sizeof(int), cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(sc.grp_exp, pf_h_hot_gexp, size_t(ng) * sizeof(int), cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(sc.grp_cnt, pf_h_hot_gcnt, size_t(ng) * sizeof(int), cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(sc.grp_off, pf_h_hot_goff, size_t(ng) * sizeof(int), cudaMemcpyHostToDevice, st);
            cudaEventRecord(pf_hot_h2d_ev, st);
            pf_hot_h2d_pending = true;
            qw38::launch_moe_pf_wide_batch<PF_GRP, HB_PER_CTA>(
                w, sc, sc.d_n_groups, sc.grp_exp, sc.grp_cnt, sc.grp_off, ld.d_ptab, ng, st);
            stat_pf_hot_groups += static_cast<uint64_t>(ng);
        }

        int np_pf = 0;
        if (have_d2d && pf_patch_pending) {
            cudaEventSynchronize(pf_patch_ev);
            pf_patch_pending = false;
        }

        auto launch_cold_batch = [&](size_t bi, int buf) {
            cudaStreamWaitEvent(st, pf_h2d_ev[buf], 0);
            const int ng = static_cast<int>(batches[bi].grps.size());
            qw38::launch_moe_pf_wide_batch<PF_GRP, HB_PER_CTA>(
                w, sc, pf_d_ng[buf], pf_d_gexp[buf], pf_d_gcnt[buf], pf_d_goff[buf], pf_ptab[buf], ng, st);
            if (have_d2d) {
                for (size_t k = 0; k < batches[bi].exps.size(); ++k) {
                    const int e = batches[bi].exps[k];
                    const int dst_slot = d2d_slot[e];
                    if (dst_slot < 0) continue;
                    const int v_idx = d2d_vidx[e];
                    const int c_idx = li * NE + e;
                    uint8_t* dst = vram_pool + size_t(dst_slot) * slot_stride;
                    const uint8_t* src = pf_stg[buf] + k * slot_stride;
                    cudaMemcpyAsync(dst, src, slot_stride, cudaMemcpyDeviceToDevice, st);
                    if (v_idx >= 0) {
                        const int ol = v_idx / NE, oe = v_idx % NE;
                        auto& old_ld = layers[size_t(ol)];
                        const auto& old_em = old_ld.exp[size_t(oe)];
                        const uint8_t* uva1 = old_ld.d_fc1 + old_em.fc1_off;
                        const uint8_t* uva2 = old_ld.d_fc2 + old_em.fc2_off;
                        for (int b = 0; b < old_em.nb; ++b) {
                            const int sl = (oe * old_ld.eb + b) * 2;
                            h_patches[np_pf++] = {old_ld.d_ptab + sl,     uva1 + old_em.piece_fc1_off[b],
                                                  (b == 0) ? (d_is_cold + size_t(v_idx)) : nullptr, 1};
                            h_patches[np_pf++] = {old_ld.d_ptab + sl + 1, uva2 + old_em.piece_fc2_off[b], nullptr, 0};
                            old_ld.h_ptab[size_t(sl)]     = uva1 + old_em.piece_fc1_off[b];
                            old_ld.h_ptab[size_t(sl) + 1] = uva2 + old_em.piece_fc2_off[b];
                        }
                        exp_slot[size_t(v_idx)] = -1;
                    }
                    {
                        auto& new_ld = layers[size_t(li)];
                        const auto& new_em = new_ld.exp[size_t(e)];
                        const uint8_t* v2 = dst + round256(new_em.fc1_bytes);
                        for (int b = 0; b < new_em.nb; ++b) {
                            const int sl = (e * new_ld.eb + b) * 2;
                            h_patches[np_pf++] = {new_ld.d_ptab + sl,     dst + new_em.piece_fc1_off[b],
                                                  (b == 0) ? (d_is_cold + size_t(c_idx)) : nullptr, 0};
                            h_patches[np_pf++] = {new_ld.d_ptab + sl + 1, v2 + new_em.piece_fc2_off[b], nullptr, 0};
                            new_ld.h_ptab[size_t(sl)]     = dst + new_em.piece_fc1_off[b];
                            new_ld.h_ptab[size_t(sl) + 1] = v2 + new_em.piece_fc2_off[b];
                        }
                        exp_slot[size_t(c_idx)] = dst_slot;
                        slot_owner[size_t(dst_slot)] = c_idx;
                        heat[size_t(c_idx)] = d2d_score[e];
                    }
                    ++stat_pf_d2d_promotes;
                }
            }
            cudaEventRecord(pf_comp_ev[buf], st);
            pf_comp_pending[buf] = true;
            stat_pf_cold_experts += static_cast<uint64_t>(batches[bi].exps.size());
        };

        // 3. Overlap CPU small-prefill cold experts 100% in parallel with GPU hot_grps + cold batches 0 & 1!
        if (n_cpu_exps > 0) {
            const size_t pre_q = std::min<size_t>(batches.size(), 2);
            for (size_t bi = 0; bi < pre_q; ++bi) {
                launch_cold_batch(bi, static_cast<int>(bi & 1));
            }
            // Wait for x_had D2H (queued before stage_cold_batch(0)) and execute CPU cold experts while GPU runs!
            cudaEventSynchronize(pf_xhad_ev);
            cpu_pool.run_prefill_sync(li, M, pf_h_xhad, n_cpu_exps, pf_cpu_exps, pf_cpu_cnts,
                                      pf_cpu_toks, pf_cpu_out_rows, pf_cpu_rws, pf_h_cpu_fc2[cbuf]);
            // Finish any remaining GPU cold batches (typically 0 or 1 for M <= 512)
            for (size_t bi = pre_q; bi < batches.size(); ++bi) {
                const int buf = static_cast<int>(bi & 1);
                stage_cold_batch(bi, buf);
                launch_cold_batch(bi, buf);
            }
        } else {
            // Pure GPU double-buffered streaming for large prefills (M > 512)
            for (size_t bi = 0; bi < batches.size(); ++bi) {
                const int buf = static_cast<int>(bi & 1);
                launch_cold_batch(bi, buf);
                if (bi + 2 < batches.size()) {
                    stage_cold_batch(bi + 2, buf);
                }
            }
        }

        if (np_pf > 0) {
            cudaMemcpyAsync(d_patches, h_patches, size_t(np_pf) * sizeof(PtabPatchEntry), cudaMemcpyHostToDevice, st);
            k_strata_ptab_patch<<<(np_pf + 255) / 256, 256, 0, st>>>(d_patches, np_pf);
            cudaEventRecord(pf_patch_ev, st);
            pf_patch_pending = true;
        }

        // 4. Scatter CPU-computed FP16 cold routes into sc.fc2_part and deterministically combine!
        __half* fc2_part_h = reinterpret_cast<__half*>(sc.fc2_part);
        if (n_cpu_routes > 0) {
            qw38::k_qw38_pf_scatter_cpu_fc2<<<n_cpu_routes, 320, 0, st>>>(
                pf_d_cpu_fc2[cbuf], pf_d_cpu_ridx[cbuf], fc2_part_h, n_cpu_routes);
            cudaEventRecord(pf_cpu_scat_ev[cbuf], st);
            pf_cpu_scat_pending[cbuf] = true;
        }
        qw38::k_qw38_moe_combine_det_h16<<<M, 256, 0, st>>>(fc2_part_h, sc.sh_out, sc.sh_scalar_gate, out_part, M);
    }

    static void prefill_moe_cb(void* ctx, int li, const qw38::MoeWeights& w, qw38::MoeScratch& sc,
                               const __nv_bfloat16* x, __nv_bfloat16* out_part, int M, cudaStream_t st) {
        auto* self = static_cast<StrataExpertCache*>(ctx);
        if (M >= 128 && w.t_eb <= 5) self->prefill_moe_layer_impl<32, 10>(li, w, sc, x, out_part, M, st);
        else self->prefill_moe_layer_impl<16, 10>(li, w, sc, x, out_part, M, st);
    }

    void reset_stats() {
        stat_hits = stat_misses = stat_swaps = stat_pf_d2d_promotes = stat_steps = 0;
        stat_pf_hot_groups = stat_pf_cold_experts = 0;
    }

    void print_stats() const {
        const uint64_t tot = stat_hits + stat_misses;
        const double hit_pct = tot ? (100.0 * stat_hits / double(tot)) : 0.0;
        const double miss_pct = tot ? (100.0 * stat_misses / double(tot)) : 0.0;
        const double ns_s = stat_steps ? double(stat_steps) : 1.0;
        const double cpu_us_exp = cpu_pool.stat_cpu_groups ? (double(cpu_pool.stat_cpu_busy_ns) / 1e3 / double(cpu_pool.stat_cpu_groups)) : 0.0;
        std::fprintf(stderr,
                     "# qw38 strata stats (%llu steps, M=%.2f, eager=%llu): hit=%.1f%% (%llu/%llu, %.1f%% cold, %llu cpu_grps [%.1fus/exp, %.2fms/st], "
                     "%llu dec_swaps, %llu pf_d2d; pf_hot=%llu pf_cold=%llu pf_cpu=%llu | "
                     "verify=%.2fms/st cap=%.1fms(%llu) mtp=%.2fms/st(%llu,cap=%.1fms) maint=%.2fms/st)\n",
                     (unsigned long long)stat_steps, double(stat_verify_toks) / ns_s, (unsigned long long)stat_eager_n, hit_pct,
                     (unsigned long long)stat_hits, (unsigned long long)tot,
                     miss_pct, (unsigned long long)cpu_pool.stat_cpu_groups, cpu_us_exp,
                     double(cpu_pool.stat_cpu_busy_ns) / 1e6 / ns_s,
                     (unsigned long long)stat_swaps, (unsigned long long)stat_pf_d2d_promotes,
                     (unsigned long long)stat_pf_hot_groups, (unsigned long long)stat_pf_cold_experts,
                     (unsigned long long)cpu_pool.stat_cpu_pf_groups,
                     double(stat_verify_ns) / 1e6 / ns_s, double(stat_cap_ns) / 1e6, (unsigned long long)stat_cap_n,
                     double(stat_mtp_ns) / 1e6 / ns_s,
                     (unsigned long long)stat_mtp_calls, double(stat_mtp_cap_ns) / 1e6,
                     double(stat_maint_ns) / 1e6 / ns_s);
    }
};

} // namespace qw38rt
