// Host-only online per-step weight defragmentation / compaction engine for EL3b (#164).
// After an in-process width switch, routed experts whose kept and newly-prefetched blocks live in separate slabs
// (or whose slabs pin sparsely occupied chunks) are compacted back into contiguous per-expert runs in `ptab::Pool`
// using a bounded per-step copy budget on the serving stream (automatically raised when idle; no "fast mode" flag).
//
// Safety invariants enforced here:
//   1. Event-fenced deferred retirement: old slabs are NEVER returned to `Pool::give` immediately after `ptab::set`.
//      They wait in `pending_` until the caller's stream fence confirms every prior launch / captured graph replay
//      has completed, after which `Pool::give` + `Pool::reclaim_empty()` frees them and returns empty VMM chunks.
//   2. Shared-expert refresh: moving shared-expert row `shared_row` invokes `refresh_shared(layer)` immediately after
//      `ptab::set` so host caches (`bind::Layer::shw_piece` for `warm_mlp` / `l2pf`) never point at retired slabs.
//   3. Headroom borrowing: if `Pool::take_run` fails under tight memory (e.g., right after narrowing 4->3), an
//      optional `borrow_headroom(need_bytes)` callback (EL5 KV shrink, #153) is invoked once before retrying.
//
// Pure host-only, zero CUDA dependency; tested under ASan+UBSan in engine/kernels/tests/ptab_compact_test.cpp.
#pragma once
#include "ptab_delta.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

namespace ptab
{

// Mutable view of one MoE layer's host piece table for compaction.
struct CompactLayerView
{
    int layer_id = 0;
    int rows = 0;
    int eb = 0;
    int shared_row = -1;                  // row NEXP (e.g., 256) if this layer holds a trellis shared expert; -1 = none
    std::vector<const uint8_t*>* host = nullptr;
    std::vector<size_t>* bytes = nullptr;

    int slot(int e, int b) const { return (e * eb + b) * 2; }
};

struct CompactConfig
{
    size_t active_experts_per_step = 2;   // copy budget per decode step while sessions are live
    size_t idle_experts_per_step = 16;    // automatically raised when scheduler has 0 live sessions
    double sparse_chunk_ratio = 0.70;     // evacuate experts in chunks with live_bytes / bytes < ratio (when >1 live chunk)
    size_t max_pending_batches = 64;      // backpressure bound on un-signaled fence batches
};

struct CompactStats
{
    size_t experts_moved = 0;
    size_t slabs_moved = 0;
    size_t bytes_copied = 0;
    size_t slabs_retired = 0;
    size_t chunks_reclaimed = 0;
    size_t bytes_reclaimed = 0;
    size_t shared_refreshes = 0;
    size_t headroom_borrows = 0;
    size_t split_remaining = 0;
    size_t sparse_remaining = 0;
};

struct CompactCandidate
{
    size_t layer_idx = 0;
    int row = 0;
    bool split_fc1 = false;
    bool split_fc2 = false;
    bool sparse_fc1 = false;
    bool sparse_fc2 = false;
    double min_chunk_fill = 1.0;

    bool is_split() const { return split_fc1 || split_fc2; }
    bool need_fc1() const { return split_fc1 || sparse_fc1; }
    bool need_fc2() const { return split_fc2 || sparse_fc2; }
};

// Inspects whether blocks 0..nb-1 of one matrix (mat_off = 0 for fc1, 1 for fc2) are contiguous back-to-back in memory.
template <class PoolT>
inline bool row_matrix_contiguous(const CompactLayerView& lv, int e, int mat_off, int& nb_out)
{
    nb_out = 0;
    const uint8_t* expect = nullptr;
    bool contig = true;
    for (int b = 0; b < lv.eb; ++b)
    {
        const size_t s = static_cast<size_t>(lv.slot(e, b) + mat_off);
        const size_t n = (*lv.bytes)[s];
        const uint8_t* p = (*lv.host)[s];
        if (!n || !p) break;
        ++nb_out;
        if (b > 0 && p != expect) contig = false;
        expect = p + PoolT::round(n);
    }
    return contig;
}

// Minimum chunk occupancy ratio among all slabs of (lv, e, mat_off); returns 1.0 if not in pool or only 1 live chunk.
template <class PoolT>
inline double row_matrix_min_fill(const CompactLayerView& lv, int e, int mat_off, int nb, const PoolT& pool)
{
    if (pool.live_chunks() <= 1) return 1.0;
    double worst = 1.0;
    for (int b = 0; b < nb; ++b)
    {
        const size_t s = static_cast<size_t>(lv.slot(e, b) + mat_off);
        const int ci = pool.chunk_of((*lv.host)[s]);
        if (ci < 0) continue;
        // Never treat the active bump chunk `pool.cur` as sparse (we are actively filling it!).
        if (ci == pool.cur) continue;
        const auto& c = pool.chunks[static_cast<size_t>(ci)];
        if (c.alive && c.bytes > 0)
        {
            const double fill = static_cast<double>(c.live_bytes) / static_cast<double>(c.bytes);
            if (fill < worst) worst = fill;
        }
    }
    return worst;
}

// Scans all MoE layers for experts needing compaction: split experts first (immediate decode locality win),
// then sparse-chunk evacuations ordered by lowest chunk occupancy (fastest chunk reclamation).
template <class PoolT>
inline std::vector<CompactCandidate> scan_compact_candidates(const std::vector<CompactLayerView>& layers,
                                                             const PoolT& pool, double sparse_ratio)
{
    std::vector<CompactCandidate> out;
    for (size_t li = 0; li < layers.size(); ++li)
    {
        const CompactLayerView& lv = layers[li];
        if (!lv.host || !lv.bytes || lv.rows <= 0 || lv.eb <= 0) continue;
        for (int e = 0; e < lv.rows; ++e)
        {
            int nb1 = 0, nb2 = 0;
            const bool c1 = row_matrix_contiguous<PoolT>(lv, e, 0, nb1);
            const bool c2 = row_matrix_contiguous<PoolT>(lv, e, 1, nb2);
            if (nb1 <= 0 && nb2 <= 0) continue;
            const double f1 = row_matrix_min_fill(lv, e, 0, nb1, pool);
            const double f2 = row_matrix_min_fill(lv, e, 1, nb2, pool);
            CompactCandidate cand;
            cand.layer_idx = li;
            cand.row = e;
            cand.split_fc1 = nb1 > 1 && !c1;
            cand.split_fc2 = nb2 > 1 && !c2;
            cand.sparse_fc1 = nb1 > 0 && f1 < sparse_ratio;
            cand.sparse_fc2 = nb2 > 0 && f2 < sparse_ratio;
            cand.min_chunk_fill = std::min(f1, f2);
            if (cand.need_fc1() || cand.need_fc2()) out.push_back(cand);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const CompactCandidate& a, const CompactCandidate& b) {
        if (a.is_split() != b.is_split()) return a.is_split() > b.is_split();
        if (a.min_chunk_fill != b.min_chunk_fill) return a.min_chunk_fill < b.min_chunk_fill;
        if (a.layer_idx != b.layer_idx) return a.layer_idx < b.layer_idx;
        return a.row < b.row;
    });
    return out;
}

template <class PoolT, class EventT>
struct CompactorOps
{
    // Copies `nbytes` from `src` to `dst` on the serving stream. Returns nullptr on success, else error string.
    std::function<const char*(uint8_t* dst, const uint8_t* src, size_t nbytes)> copy_d2d;
    // Updates piece (e, b) in layer `layer_idx` via ptab::set on the serving stream. Returns nullptr on success.
    std::function<const char*(size_t layer_idx, int e, int b, const uint8_t* fc1, size_t f1b,
                              const uint8_t* fc2, size_t f2b)> set_piece;
    // Called immediately after ptab::set updates shared-expert row `shared_row` (bind::refresh_shared_pieces).
    std::function<const char*(size_t layer_idx)> refresh_shared;
    // Records a fence event on the serving stream after a step's ptab::set batch.
    std::function<const char*(EventT& out_ev)> record_fence;
    // Non-blocking poll: true iff every launch ordered before `ev` on the serving stream has finished.
    std::function<bool(const EventT& ev)> fence_ready;
    // Releases resources associated with `ev`.
    std::function<void(EventT& ev)> destroy_fence;
    // Optional (EL5 #153): borrow `need_bytes` of copy-destination headroom by shrinking VMM KV pool.
    // Returns true if headroom was freed and take_run should be retried.
    std::function<bool(size_t need_bytes)> borrow_headroom;
};

template <class PoolT, class EventT>
class Compactor
{
    struct PendingFree
    {
        std::vector<SlabRef> slabs;
        EventT fence{};
    };

    CompactConfig cfg_;
    CompactorOps<PoolT, EventT> ops_;
    std::deque<PendingFree> pending_;
    CompactStats stats_;

    template <class TakeFn>
    bool take_with_borrow(size_t need_bytes, TakeFn&& take_fn, const char*& err)
    {
        if (take_fn(err)) return true;
        if (ops_.borrow_headroom && ops_.borrow_headroom(need_bytes))
        {
            ++stats_.headroom_borrows;
            if (take_fn(err)) return true;
        }
        return false;
    }

public:
    Compactor(CompactConfig cfg, CompactorOps<PoolT, EventT> ops) : cfg_(cfg), ops_(std::move(ops)) {}
    ~Compactor()
    {
        if (ops_.destroy_fence)
        {
            for (auto& pf : pending_) ops_.destroy_fence(pf.fence);
        }
        pending_.clear();
    }

    Compactor(const Compactor&) = delete;
    Compactor& operator=(const Compactor&) = delete;

    size_t pending_batches() const { return pending_.size(); }
    const CompactStats& stats() const { return stats_; }

    // Polls completed fence batches in FIFO order, returning old slabs to `pool` via Pool::give + reclaim_empty().
    size_t poll_retire(PoolT& pool)
    {
        size_t retired_now = 0;
        while (!pending_.empty() && ops_.fence_ready(pending_.front().fence))
        {
            PendingFree pf = std::move(pending_.front());
            pending_.pop_front();
            for (const SlabRef& s : pf.slabs)
            {
                pool.give(s.ptr, s.bytes);
                ++retired_now;
            }
            if (ops_.destroy_fence) ops_.destroy_fence(pf.fence);
        }
        if (retired_now)
        {
            stats_.slabs_retired += retired_now;
            size_t cf = 0;
            const size_t bf = pool.reclaim_empty(&cf);
            stats_.chunks_reclaimed += cf;
            stats_.bytes_reclaimed += bf;
        }
        return retired_now;
    }

    // Runs one compaction step bounded by `idle ? idle_experts_per_step : active_experts_per_step`.
    // Returns nullptr on success (including 0 work left), or an error string.
    const char* step(std::vector<CompactLayerView>& layers, PoolT& pool, bool idle)
    {
        poll_retire(pool);
        if (pending_.size() >= cfg_.max_pending_batches) return nullptr;

        const auto cands = scan_compact_candidates(layers, pool, cfg_.sparse_chunk_ratio);
        stats_.split_remaining = 0;
        stats_.sparse_remaining = 0;
        for (const auto& c : cands)
        {
            if (c.is_split()) ++stats_.split_remaining;
            else ++stats_.sparse_remaining;
        }
        if (cands.empty()) return nullptr;

        const size_t budget = idle ? cfg_.idle_experts_per_step : cfg_.active_experts_per_step;
        std::vector<SlabRef> batch_old;
        std::vector<size_t> f1_sz, f2_sz;
        std::vector<uint8_t*> f1_p, f2_p;
        size_t moved_this_step = 0;

        for (size_t ci = 0; ci < cands.size() && moved_this_step < budget; ++ci)
        {
            const CompactCandidate& c = cands[ci];
            CompactLayerView& lv = layers[c.layer_idx];
            int nb1 = 0, nb2 = 0;
            (void)row_matrix_contiguous<PoolT>(lv, c.row, 0, nb1);
            (void)row_matrix_contiguous<PoolT>(lv, c.row, 1, nb2);
            const int nb = std::max(nb1, nb2);
            if (nb <= 0) continue;

            const bool move_fc1 = c.need_fc1() && nb1 > 0;
            const bool move_fc2 = c.need_fc2() && nb2 > 0;
            if (!move_fc1 && !move_fc2) continue;

            f1_sz.assign(static_cast<size_t>(nb), 0);
            f2_sz.assign(static_cast<size_t>(nb), 0);
            f1_p.assign(static_cast<size_t>(nb), nullptr);
            f2_p.assign(static_cast<size_t>(nb), nullptr);

            size_t need1 = 0, need2 = 0;
            for (int b = 0; b < nb; ++b)
            {
                const size_t s = static_cast<size_t>(lv.slot(c.row, b));
                f1_sz[static_cast<size_t>(b)] = (*lv.bytes)[s];
                f2_sz[static_cast<size_t>(b)] = (*lv.bytes)[s + 1];
                if (move_fc1) need1 += PoolT::round(f1_sz[static_cast<size_t>(b)]);
                if (move_fc2) need2 += PoolT::round(f2_sz[static_cast<size_t>(b)]);
            }

            const char* err = nullptr;
            if (move_fc1)
            {
                if (!take_with_borrow(need1 + need2, [&](const char*& e) {
                        return pool.take_run(f1_sz.data(), nb1, f1_p.data(), e);
                    }, err))
                {
                    // Tight memory even after borrow: stop cleanly for this step rather than failing serving.
                    break;
                }
            }
            if (move_fc2)
            {
                if (!take_with_borrow(need2, [&](const char*& e) {
                        return pool.take_run(f2_sz.data(), nb2, f2_p.data(), e);
                    }, err))
                {
                    if (move_fc1)
                    {
                        for (int b = 0; b < nb1; ++b) pool.give(f1_p[static_cast<size_t>(b)], f1_sz[static_cast<size_t>(b)]);
                        pool.reclaim_empty();
                    }
                    break;
                }
            }

            // Copy bytes to the new contiguous slabs, then atomically re-point each block via ptab::set on `st`.
            for (int b = 0; b < nb; ++b)
            {
                const size_t s = static_cast<size_t>(lv.slot(c.row, b));
                const uint8_t* old_f1 = (*lv.host)[s];
                const uint8_t* old_f2 = (*lv.host)[s + 1];
                const uint8_t* new_f1 = (move_fc1 && b < nb1) ? f1_p[static_cast<size_t>(b)] : old_f1;
                const uint8_t* new_f2 = (move_fc2 && b < nb2) ? f2_p[static_cast<size_t>(b)] : old_f2;
                if (move_fc1 && b < nb1 && f1_sz[static_cast<size_t>(b)])
                {
                    if (const char* ce = ops_.copy_d2d(f1_p[static_cast<size_t>(b)], old_f1, f1_sz[static_cast<size_t>(b)]))
                        return ce;
                    stats_.bytes_copied += f1_sz[static_cast<size_t>(b)];
                    ++stats_.slabs_moved;
                }
                if (move_fc2 && b < nb2 && f2_sz[static_cast<size_t>(b)])
                {
                    if (const char* ce = ops_.copy_d2d(f2_p[static_cast<size_t>(b)], old_f2, f2_sz[static_cast<size_t>(b)]))
                        return ce;
                    stats_.bytes_copied += f2_sz[static_cast<size_t>(b)];
                    ++stats_.slabs_moved;
                }
                if (const char* se = ops_.set_piece(c.layer_idx, c.row, b, new_f1, f1_sz[static_cast<size_t>(b)],
                                                    new_f2, f2_sz[static_cast<size_t>(b)]))
                    return se;
                (*lv.host)[s] = new_f1;
                (*lv.host)[s + 1] = new_f2;
                if (move_fc1 && b < nb1 && old_f1) batch_old.push_back(SlabRef{const_cast<uint8_t*>(old_f1), f1_sz[static_cast<size_t>(b)]});
                if (move_fc2 && b < nb2 && old_f2) batch_old.push_back(SlabRef{const_cast<uint8_t*>(old_f2), f2_sz[static_cast<size_t>(b)]});
            }

            // Safety invariant #2: refresh host copies of shared-expert row NEXP after ptab::set.
            if (lv.shared_row >= 0 && c.row == lv.shared_row && ops_.refresh_shared)
            {
                if (const char* re = ops_.refresh_shared(c.layer_idx)) return re;
                ++stats_.shared_refreshes;
            }
            ++stats_.experts_moved;
            ++moved_this_step;
        }

        if (!batch_old.empty())
        {
            PendingFree pf;
            pf.slabs = std::move(batch_old);
            if (const char* fe = ops_.record_fence(pf.fence)) return fe;
            pending_.push_back(std::move(pf));
        }
        // Recompute remaining counts accurately if everything in cands was processed.
        const auto after = scan_compact_candidates(layers, pool, cfg_.sparse_chunk_ratio);
        stats_.split_remaining = 0;
        stats_.sparse_remaining = 0;
        for (const auto& x : after)
        {
            if (x.is_split()) ++stats_.split_remaining;
            else ++stats_.sparse_remaining;
        }
        return nullptr;
    }

    // Returns true when no fragmented/sparse experts remain and all fenced retirements have drained.
    bool done() const { return stats_.split_remaining == 0 && stats_.sparse_remaining == 0 && pending_.empty(); }
};

}   // namespace ptab
