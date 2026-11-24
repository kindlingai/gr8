// In-process width-module lifecycle ABI and EL7 (#155) per-layout CUDA graph capture policy.
//
// In Phase B dynamic TP scaling (#147), one `g6m4` process owns the process-lifetime singletons (`G6HostRuntime`:
// `ptab::Pool`, VMM `kvchunks`, `RunPool`, `FabricPool`, S2 `Service`/shim connection) while each compiled TP width
// (`build_m4.sh` -> `g6w<N>_create`) instantiates an opaque device-side width module (`G6WidthModule`). During a
// live switch, the active width and a staged target width coexist briefly inside the same CUDA context:
//   - CaptureAhead (`MemAvailable > est_graph_bytes + mem_min_bytes`): target graphs capture while old width serves
//     (0 ms added to the cutover pause).
//   - CaptureAfter (tight headroom): old graphs release immediately at cutover before capturing target graphs.
//   - Rollback: any failure before cutover tears down staged graphs and module cleanly; active width never pauses.
//
// Pure host-only, zero CUDA dependency; tested under ASan+UBSan in engine/runtime/tests/width_module_test.cpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" {

// Process-lifetime singletons hoisted out of per-width `g6w<N>_main` and shared across width switches.
struct G6HostRuntime {
    void* ptab_pool;                          // ptab::Pool*
    void* kv_chunks;                          // G6KvChunks* (GLM / Flash VMM KV backing)
    void* run_pool;                           // g6s2::RunPool*
    void* fabric_pool;                        // g6::FabricPool*
    const char* s2_dir;                       // persistent shim/S2 directory
    uint64_t mem_min_bytes;                   // GR8_MEM_MIN floor in bytes
    uint64_t (*mem_avail_bytes)(void* ctx);   // returns current MemAvailable bytes
    void* mem_ctx;
};

// Target width/group specification passed to `g6w<N>_create`.
struct G6WidthSpec {
    int nr;                                   // target TP width (1, 2, 3, 4, 5, 6, 8)
    int rank;                                 // this box's rank inside target group
    uint64_t group_id;                        // FabricPool Group::id
    const char* layout_path;                  // optional g6layout-v1 JSON path (NULL = formula)
    uint64_t est_graph_bytes;                 // estimated graph memory Y for this width/model
};

// Opaque handle returned by a compiled width unit (`g6w<N>_create`).
struct G6WidthModule {
    int nr;
    void* ctx;
    // Captures this width's per-layout CUDA graph set against its target FabricPool group and ptab::Tables.
    // Returns 0 on success (filling graph_bytes_out & capture_ms_out when non-null), non-zero + err on failure.
    int (*capture_graphs)(void* ctx, int ahead, uint64_t* graph_bytes_out, double* capture_ms_out,
                          char* err, size_t errlen);
    // Releases this width's captured CUDA graph set immediately (freeing graph memory Y).
    void (*release_graphs)(void* ctx);
    // Tears down this width's device module without touching G6HostRuntime singletons.
    void (*destroy)(void* ctx);
};

typedef int (*G6WidthCreateFn)(const G6HostRuntime* host, const G6WidthSpec* spec,
                               G6WidthModule* out, char* err, size_t errlen);

} // extern "C"

namespace g6rt {

enum class CaptureMode {
    CaptureAhead = 0, // MemAvailable - Y > GR8_MEM_MIN: capture while old width serves (0 ms cutover pause)
    CaptureAfter = 1, // Tight memory: release old graphs at cutover, then capture target graphs
};

inline CaptureMode choose_capture_mode(uint64_t mem_avail_bytes, uint64_t est_graph_bytes, uint64_t mem_min_bytes) {
    if (est_graph_bytes + mem_min_bytes < est_graph_bytes) return CaptureMode::CaptureAfter; // overflow guard
    return mem_avail_bytes > est_graph_bytes + mem_min_bytes ? CaptureMode::CaptureAhead : CaptureMode::CaptureAfter;
}

// Registry of width factories linked into this binary (populated by widths.cpp from weak g6w<N>_create symbols).
class WidthRegistry {
    std::map<int, G6WidthCreateFn> factories_;
public:
    void add(int nr, G6WidthCreateFn fn) {
        if (nr > 0 && fn) factories_[nr] = fn;
    }
    bool has(int nr) const { return factories_.count(nr) != 0; }
    G6WidthCreateFn get(int nr) const {
        auto it = factories_.find(nr);
        return it != factories_.end() ? it->second : nullptr;
    }
    std::vector<int> widths() const {
        std::vector<int> out;
        out.reserve(factories_.size());
        for (const auto& [nr, fn] : factories_) out.push_back(nr);
        return out;
    }
};

struct CutoverReport {
    int old_nr = 0;
    int new_nr = 0;
    CaptureMode mode = CaptureMode::CaptureAhead;
    bool captured_ahead = false;
    bool captured_in_cutover = false;
    uint64_t target_graph_bytes = 0;
    double target_capture_ms = 0.0;
};

// Coordinates the in-process active and staged width modules across one live switch:
//   activate() -> stage_target() -> [maybe_capture_ahead()] -> cutover() / rollback_staged()
class WidthCoordinator {
    const WidthRegistry& reg_;
    const G6HostRuntime& host_;
    G6WidthModule active_{};
    bool has_active_ = false;
    G6WidthModule staged_{};
    G6WidthSpec staged_spec_{};
    bool has_staged_ = false;
    bool staged_captured_ = false;
    uint64_t staged_graph_bytes_ = 0;
    double staged_capture_ms_ = 0.0;

    static void destroy_mod(G6WidthModule& m) {
        if (m.release_graphs && m.ctx) m.release_graphs(m.ctx);
        if (m.destroy && m.ctx) m.destroy(m.ctx);
        m = G6WidthModule{};
    }

public:
    WidthCoordinator(const WidthRegistry& reg, const G6HostRuntime& host) : reg_(reg), host_(host) {}
    ~WidthCoordinator() {
        if (has_staged_) { destroy_mod(staged_); has_staged_ = false; }
        if (has_active_) { destroy_mod(active_); has_active_ = false; }
    }
    WidthCoordinator(const WidthCoordinator&) = delete;
    WidthCoordinator& operator=(const WidthCoordinator&) = delete;

    bool has_active() const { return has_active_; }
    bool has_staged() const { return has_staged_; }
    bool staged_captured() const { return staged_captured_; }
    int active_nr() const { return has_active_ ? active_.nr : 0; }
    int staged_nr() const { return has_staged_ ? staged_.nr : 0; }
    const G6WidthModule* active() const { return has_active_ ? &active_ : nullptr; }
    const G6WidthModule* staged() const { return has_staged_ ? &staged_ : nullptr; }

    // Boots the initial active width module and captures its graphs.
    bool activate(const G6WidthSpec& spec, std::string& err) {
        if (has_active_) { err = "width coordinator: active width already set"; return false; }
        G6WidthCreateFn fn = reg_.get(spec.nr);
        if (!fn) { err = "width coordinator: TP " + std::to_string(spec.nr) + " not linked in binary"; return false; }
        char ebuf[512] = "";
        G6WidthModule mod{};
        if (fn(&host_, &spec, &mod, ebuf, sizeof ebuf) != 0) {
            err = ebuf[0] ? std::string(ebuf) : "g6w_create failed";
            return false;
        }
        uint64_t gb = 0; double ms = 0;
        if (mod.capture_graphs && mod.capture_graphs(mod.ctx, /*ahead=*/0, &gb, &ms, ebuf, sizeof ebuf) != 0) {
            destroy_mod(mod);
            err = ebuf[0] ? std::string(ebuf) : "initial capture_graphs failed";
            return false;
        }
        active_ = mod;
        has_active_ = true;
        return true;
    }

    // Stages a target width module alongside the active width while the active width continues serving.
    // Captures target graphs ahead of cutover iff `MemAvailable - Y > GR8_MEM_MIN` (`CaptureAhead`).
    bool stage_target(const G6WidthSpec& spec, CaptureMode& chosen_mode, std::string& err) {
        if (!has_active_) { err = "width coordinator: cannot stage without active width"; return false; }
        if (has_staged_) { err = "width coordinator: target width already staged"; return false; }
        G6WidthCreateFn fn = reg_.get(spec.nr);
        if (!fn) { err = "width coordinator: target TP " + std::to_string(spec.nr) + " not linked in binary"; return false; }

        char ebuf[512] = "";
        G6WidthModule mod{};
        if (fn(&host_, &spec, &mod, ebuf, sizeof ebuf) != 0) {
            err = ebuf[0] ? std::string(ebuf) : "staged g6w_create failed";
            return false;
        }
        has_staged_ = true;
        staged_ = mod;
        staged_spec_ = spec;
        staged_captured_ = false;
        staged_graph_bytes_ = 0;
        staged_capture_ms_ = 0.0;

        const uint64_t avail = host_.mem_avail_bytes ? host_.mem_avail_bytes(host_.mem_ctx) : 0;
        chosen_mode = choose_capture_mode(avail, spec.est_graph_bytes, host_.mem_min_bytes);
        if (chosen_mode == CaptureMode::CaptureAhead && staged_.capture_graphs) {
            if (staged_.capture_graphs(staged_.ctx, /*ahead=*/1, &staged_graph_bytes_, &staged_capture_ms_,
                                       ebuf, sizeof ebuf) != 0) {
                rollback_staged();
                err = ebuf[0] ? std::string(ebuf) : "capture-ahead failed";
                return false;
            }
            staged_captured_ = true;
        }
        return true;
    }

    // Tears down a staged target width cleanly on mid-prefetch or mid-capture failure; active width never pauses.
    void rollback_staged() {
        if (!has_staged_) return;
        destroy_mod(staged_);
        has_staged_ = false;
        staged_captured_ = false;
        staged_graph_bytes_ = 0;
        staged_capture_ms_ = 0.0;
    }

    // Executes cutover at the S2 step fence:
    //  1. Releases old width's graph set immediately (freeing graph memory Y_old).
    //  2. If `CaptureAfter`, captures target graphs now (with Y_old already returned!).
    //  3. Swaps staged -> active and destroys the old width module.
    bool cutover(CutoverReport& rep, std::string& err) {
        if (!has_active_ || !has_staged_) {
            err = "width coordinator: cutover requires active and staged widths";
            return false;
        }
        rep = CutoverReport{};
        rep.old_nr = active_.nr;
        rep.new_nr = staged_.nr;
        rep.mode = staged_captured_ ? CaptureMode::CaptureAhead : CaptureMode::CaptureAfter;
        rep.captured_ahead = staged_captured_;

        // Release old graphs FIRST so CaptureAfter sees the freed graph memory immediately.
        if (active_.release_graphs && active_.ctx) active_.release_graphs(active_.ctx);

        if (!staged_captured_ && staged_.capture_graphs) {
            char ebuf[512] = "";
            if (staged_.capture_graphs(staged_.ctx, /*ahead=*/0, &staged_graph_bytes_, &staged_capture_ms_,
                                       ebuf, sizeof ebuf) != 0) {
                err = ebuf[0] ? std::string(ebuf) : "capture-after failed at cutover";
                rollback_staged();
                return false;
            }
            staged_captured_ = true;
            rep.captured_in_cutover = true;
        }
        rep.target_graph_bytes = staged_graph_bytes_;
        rep.target_capture_ms = staged_capture_ms_;

        G6WidthModule old = active_;
        active_ = staged_;
        staged_ = G6WidthModule{};
        has_staged_ = false;
        staged_captured_ = false;
        if (old.destroy && old.ctx) old.destroy(old.ctx);
        return true;
    }
};

} // namespace g6rt
