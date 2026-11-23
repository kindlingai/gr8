// Pure host-only coordination between kvman-rs RunPool chunks (EL5 #153) and a VMM backing sync callback
// (kvchunks.h / flash_kv.h / glm_kv.h). No CUDA headers: unit-tested on CPU against libg6host.a.
//
// Invariant ordering (engine/attn/kvchunks.h):
//   Shrink: pool.shrink(delta) -> drain_io() (spill evicted runs first) -> sync_fn(new_mask) (unmap granules).
//           If pool.shrink returns Busy (live sessions hold pages in those chunks or an exclusive run is resident),
//           neither drain_io nor sync_fn runs and the pool is unchanged.
//   Grow:   compute next_mask matching pool.grow_pool(delta) (lowest offline chunks) -> sync_fn(next_mask) FIRST ->
//           on success, pool.grow_pool(delta). On sync_fn failure, roll back via sync_fn(cur_mask) so no unbacked
//           page is ever handed out by the pool.
#pragma once
#include "pool.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace g6s2 {

// Default resizable KV chunk size below place (256 pages = 16,384 tokens when override_pages == 0, clamped to place).
inline uint32_t kv_chunk_pages_default(uint32_t place, uint32_t override_pages = 0) {
    if (!place) return 0;
    const uint32_t want = override_pages ? override_pages : 256u;
    return std::min(want, place);
}

// Current per-chunk online mask [0..chunk_count) of a RunPool with chunks() enabled.
inline std::vector<uint8_t> kv_pool_mask(const RunPool& pool) {
    const uint32_t n = pool.chunk_count();
    std::vector<uint8_t> m(n, 0);
    for (uint32_t k = 0; k < n; ++k) m[k] = pool.chunk_online(k) ? 1 : 0;
    return m;
}

// Count of currently online chunks in `mask`.
inline uint32_t kv_mask_online(const std::vector<uint8_t>& mask) {
    uint32_t c = 0;
    for (uint8_t x : mask) c += x ? 1u : 0u;
    return c;
}

// Predict the mask after RunPool::grow_pool(add): lowest `add` offline chunks brought online.
// Returns std::nullopt when fewer than `add` chunks are offline.
inline std::optional<std::vector<uint8_t>> kv_grow_mask(const std::vector<uint8_t>& cur, uint32_t add,
                                                        std::vector<uint32_t>* picked = nullptr) {
    std::vector<uint8_t> next = cur;
    std::vector<uint32_t> chosen;
    chosen.reserve(add);
    for (uint32_t k = 0; k < (uint32_t)next.size() && chosen.size() < add; ++k) {
        if (!next[k]) {
            next[k] = 1;
            chosen.push_back(k);
        }
    }
    if (chosen.size() < add) return std::nullopt;
    if (picked) *picked = std::move(chosen);
    return next;
}

enum class KvResizeStatus { Ok, Noop, Busy, Invalid, SyncError };

struct KvResizeResult {
    KvResizeStatus status = KvResizeStatus::Ok;
    uint32_t before_online = 0;
    uint32_t after_online = 0;
    size_t mapped_bytes = 0;
    size_t released_bytes = 0;
    int sync_rc = 0;
    std::vector<uint32_t> changed_chunks;
};

// Take a chunked RunPool to `target_online` chunks online (clamped to chunk_count), running `drain_io` and `sync_fn`
// in the required EL5 order. `sync_fn(mask, online_count, &mapped, &released)` returns 0 on success or an error code.
inline KvResizeResult kv_pool_resize_to(
    RunPool& pool, uint32_t target_online, const std::function<void()>& drain_io,
    const std::function<int(const std::vector<uint8_t>& mask, uint32_t online_count, size_t* mapped, size_t* released)>& sync_fn) {
    KvResizeResult res{};
    const uint32_t total = pool.chunk_count();
    if (!total) {
        res.status = KvResizeStatus::Invalid;
        return res;
    }
    const std::vector<uint8_t> cur = kv_pool_mask(pool);
    const uint32_t cur_online = kv_mask_online(cur);
    res.before_online = res.after_online = cur_online;
    const uint32_t want = std::min(target_online, total);
    if (want == cur_online) {
        res.status = KvResizeStatus::Noop;
        return res;
    }
    if (want < cur_online) {
        const uint32_t delta = cur_online - want;
        std::optional<std::vector<uint32_t>> off;
        try {
            off = pool.shrink(delta);
        } catch (const std::exception&) {
            res.status = KvResizeStatus::Invalid;
            return res;
        }
        if (!off) {
            res.status = KvResizeStatus::Busy;
            return res;
        }
        res.changed_chunks = *off;
        res.after_online = want;
        if (drain_io) drain_io();
        const std::vector<uint8_t> next = kv_pool_mask(pool);
        res.sync_rc = sync_fn(next, want, &res.mapped_bytes, &res.released_bytes);
        res.status = res.sync_rc == 0 ? KvResizeStatus::Ok : KvResizeStatus::SyncError;
        return res;
    }
    const uint32_t delta = want - cur_online;
    std::vector<uint32_t> expected;
    const auto next = kv_grow_mask(cur, delta, &expected);
    if (!next) {
        res.status = KvResizeStatus::Invalid;
        return res;
    }
    res.sync_rc = sync_fn(*next, want, &res.mapped_bytes, &res.released_bytes);
    if (res.sync_rc != 0) {
        // Roll back mapped granules so the pool's view and backed memory stay aligned.
        size_t rb_m = 0, rb_r = 0;
        (void)sync_fn(cur, cur_online, &rb_m, &rb_r);
        res.status = KvResizeStatus::SyncError;
        return res;
    }
    res.changed_chunks = pool.grow_pool(delta);
    require(res.changed_chunks == expected, "kv_pool_resize_to: RunPool::grow_pool chunk order diverged from kv_grow_mask");
    res.after_online = want;
    res.status = KvResizeStatus::Ok;
    return res;
}

}   // namespace g6s2
