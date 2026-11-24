// In-process live width switch coordinator (EL8, #156).
//
// Drives the 5-step single-process live cutover inside g6m4 without restarting the engine or dropping the front-end
// Unix stream socket:
//   1. plan_and_announce: validate target width & group, verify width module exists, plan weight/KV deltas.
//   2. prefetch_while_serving: stage missing weight deltas into ptab::Pool, open the target FabricPool group, and
//      run background bulk KV migration rounds while the active width continues serving uninterrupted.
//   3. precapture_if_room: optionally pre-capture target-width CUDA graphs if free VRAM headroom permits; otherwise
//      defer capture to the cutover fence after releasing the old width's graphs.
//   4. cutover_at_fence: at the synchronized S2 EventKind::Switch boundary, run the final KV migration tail, release
//      old graphs (if not pre-captured) and capture target graphs (if deferred), flip the active WidthModule and
//      FabricPool group, and preserve every live session and the front-end connection.
//   5. release_and_defrag: release old-width weight chunks from ptab::Pool, tear down the old FabricPool group, adjust
//      VMM KV chunks, and arm online per-step weight defragmentation (#164).
//
// Any failure in Steps 1–3 aborts cleanly and rolls back all staged state (weight deltas, target fabric group,
// pre-captured graphs, in-flight migration state) with zero disruption to the active serving width.
//
// Host only, no CUDA; tested under ASan + UBSan in tests/live_switch_test.cpp.
#pragma once

#include "scheduler.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace g6s2 {

enum class SwitchPhase : uint32_t {
    Idle = 0,
    Planned = 1,
    Prefetching = 2,
    PreCaptured = 3,
    Cutover = 4,
    Completed = 5,
    Aborted = 6,
};

struct SwitchPlan {
    uint64_t group_id = 0;
    uint32_t old_nr = 0;
    uint32_t target_nr = 0;
    uint32_t flags = 0;
    uint64_t need_capture_bytes = 0;
    uint64_t avail_vram_bytes = 0;
    bool can_precapture() const { return need_capture_bytes == 0 || avail_vram_bytes >= need_capture_bytes; }
};

struct SwitchStats {
    SwitchPhase phase = SwitchPhase::Idle;
    bool precaptured = false;
    bool defrag_armed = false;
    bool rolled_back = false;
    uint32_t migrated_sessions = 0;
    double plan_ms = 0.0;
    double prefetch_ms = 0.0;
    double precapture_ms = 0.0;
    double cutover_ms = 0.0;
    double release_ms = 0.0;
    std::string abort_reason;
};

// Callbacks wired by g6m4 (or by host tests) for each step of the 5-step in-process live switch.
// Throwing from any Step 1-3 callback triggers automatic rollback of all prior staged resources.
struct LiveSwitchOps {
    // Step 1: verify target width is compiled in and plan weight/KV deltas.
    std::function<bool(uint32_t target_nr)> has_width;
    std::function<void(const SwitchPlan&)> announce_group;

    // Step 2 (while serving continues): stage weight deltas, open new fabric group, run bulk KV migration.
    std::function<void(const SwitchPlan&)> stage_weights;
    std::function<void(const SwitchPlan&)> open_fabric_group;
    std::function<uint32_t(const SwitchPlan&)> migrate_kv_bulk;

    // Step 3 (optional): pre-capture target graphs if VRAM headroom permits.
    std::function<void(const SwitchPlan&)> precapture_graphs;

    // Step 4 (at synchronized S2 Switch event fence): tail KV sync, graph swap, width/fabric cutover.
    std::function<void(const SwitchPlan&)> fence_tail_kv;
    std::function<void()> release_old_graphs;
    std::function<void(const SwitchPlan&)> capture_target_graphs;
    std::function<void(const SwitchPlan&)> swap_active_width;

    // Step 5 (post-cutover): release old weights & fabric group, resize KV chunks, arm online defrag (#164).
    std::function<void(const SwitchPlan&)> release_old_weights;
    std::function<void(const SwitchPlan&)> close_old_fabric_group;
    std::function<void(const SwitchPlan&)> arm_online_defrag;

    // Rollback hooks invoked in reverse order if Steps 1-3 fail before the cutover fence.
    std::function<void(const SwitchPlan&)> rollback_precaptured_graphs;
    std::function<void(const SwitchPlan&)> rollback_kv_migration;
    std::function<void(const SwitchPlan&)> rollback_fabric_group;
    std::function<void(const SwitchPlan&)> rollback_staged_weights;
};

class LiveSwitchCoordinator {
    using Clock = std::chrono::steady_clock;
    static double elapsed_ms(Clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    LiveSwitchOps ops_;
    SwitchPlan plan_{};
    SwitchStats stats_{};
    bool staged_weights_ = false;
    bool opened_fabric_ = false;
    bool started_migration_ = false;
    bool did_precapture_ = false;

    void rollback_staged(const std::string& reason) noexcept {
        stats_.abort_reason = reason;
        stats_.phase = SwitchPhase::Aborted;
        stats_.rolled_back = true;
        if (did_precapture_) {
            if (ops_.rollback_precaptured_graphs) try { ops_.rollback_precaptured_graphs(plan_); } catch (...) {}
            did_precapture_ = false;
        }
        if (started_migration_) {
            if (ops_.rollback_kv_migration) try { ops_.rollback_kv_migration(plan_); } catch (...) {}
            started_migration_ = false;
        }
        if (opened_fabric_) {
            if (ops_.rollback_fabric_group) try { ops_.rollback_fabric_group(plan_); } catch (...) {}
            opened_fabric_ = false;
        }
        if (staged_weights_) {
            if (ops_.rollback_staged_weights) try { ops_.rollback_staged_weights(plan_); } catch (...) {}
            staged_weights_ = false;
        }
    }

public:
    explicit LiveSwitchCoordinator(LiveSwitchOps ops) : ops_(std::move(ops)) {}

    const SwitchPlan& plan() const { return plan_; }
    const SwitchStats& stats() const { return stats_; }
    SwitchPhase phase() const { return stats_.phase; }

    // Steps 1-3: prepare everything while the old width continues serving uninterrupted.
    // Returns true when ready to announce the S2 EventKind::Switch fence; returns false (having rolled back
    // all staged resources) on any pre-cutover failure.
    bool prepare(const SwitchPlan& p) noexcept {
        plan_ = p;
        stats_ = SwitchStats{};
        staged_weights_ = opened_fabric_ = started_migration_ = did_precapture_ = false;
        try {
            // Step 1: plan & announce
            auto t0 = Clock::now();
            if (p.group_id == 0) throw std::runtime_error("live_switch: group_id must be nonzero");
            if (p.target_nr < 1 || p.target_nr > 8) throw std::runtime_error("live_switch: target_nr out of range");
            if (p.old_nr == p.target_nr) throw std::runtime_error("live_switch: target_nr equals current width");
            if (ops_.has_width && !ops_.has_width(p.target_nr))
                throw std::runtime_error("live_switch: target width module not compiled into binary");
            if (ops_.announce_group) ops_.announce_group(p);
            stats_.plan_ms = elapsed_ms(t0);
            stats_.phase = SwitchPhase::Planned;

            // Step 2: prefetch weights + open target fabric group + bulk KV migration while serving
            t0 = Clock::now();
            stats_.phase = SwitchPhase::Prefetching;
            if (ops_.stage_weights) {
                ops_.stage_weights(p);
                staged_weights_ = true;
            }
            if (ops_.open_fabric_group) {
                ops_.open_fabric_group(p);
                opened_fabric_ = true;
            }
            if (ops_.migrate_kv_bulk) {
                started_migration_ = true;
                stats_.migrated_sessions = ops_.migrate_kv_bulk(p);
            }
            stats_.prefetch_ms = elapsed_ms(t0);

            // Step 3: pre-capture target graphs if VRAM headroom permits
            t0 = Clock::now();
            if (p.can_precapture() && ops_.precapture_graphs) {
                ops_.precapture_graphs(p);
                did_precapture_ = true;
                stats_.precaptured = true;
            }
            stats_.precapture_ms = elapsed_ms(t0);
            stats_.phase = SwitchPhase::PreCaptured;
            return true;
        } catch (const std::exception& e) {
            rollback_staged(e.what());
            return false;
        } catch (...) {
            rollback_staged("live_switch: unknown pre-cutover failure");
            return false;
        }
    }

    // Step 4: invoked from DriverHooks::switch_width / ModuleOps::switch_width at the synchronized S2 fence.
    // Must match the prepared group_id and target_nr. Leaves live S2 sessions and front-end connection intact.
    void cutover_at_fence(const SwitchOp& op) {
        require(stats_.phase == SwitchPhase::PreCaptured, "live_switch: cutover before prepare completed");
        require(op.group_id == plan_.group_id && op.target_nr == plan_.target_nr,
                "live_switch: fence SwitchOp mismatch against prepared plan");
        const auto t0 = Clock::now();
        stats_.phase = SwitchPhase::Cutover;
        if (ops_.fence_tail_kv) ops_.fence_tail_kv(plan_);
        if (ops_.release_old_graphs) ops_.release_old_graphs();
        if (!did_precapture_ && ops_.capture_target_graphs) ops_.capture_target_graphs(plan_);
        if (ops_.swap_active_width) ops_.swap_active_width(plan_);
        stats_.cutover_ms = elapsed_ms(t0);
    }

    // Step 5: post-cutover cleanup & online defragmentation arming while the new width serves.
    void finish_and_defrag() {
        require(stats_.phase == SwitchPhase::Cutover, "live_switch: finish before cutover");
        const auto t0 = Clock::now();
        if (ops_.release_old_weights) ops_.release_old_weights(plan_);
        if (ops_.close_old_fabric_group) ops_.close_old_fabric_group(plan_);
        if (ops_.arm_online_defrag) {
            ops_.arm_online_defrag(plan_);
            stats_.defrag_armed = true;
        }
        staged_weights_ = opened_fabric_ = started_migration_ = did_precapture_ = false;
        stats_.release_ms = elapsed_ms(t0);
        stats_.phase = SwitchPhase::Completed;
    }

    // Explicit abort before cutover (e.g., if a peer failed prepare or the operator cancelled).
    void abort(const std::string& reason) noexcept {
        if (stats_.phase == SwitchPhase::Cutover || stats_.phase == SwitchPhase::Completed) return;
        rollback_staged(reason);
    }
};

} // namespace g6s2
