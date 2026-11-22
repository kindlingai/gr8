// Host-only tests for EL3b (#164) online per-step weight defragmentation / compaction:
//  1. Post-4->3 split experts (kept blocks + newly fetched blocks in separate slabs) compacted back into contiguous
//     take_run slabs with bounded per-step budget; every single byte stays bit-identical before, during, and after.
//  2. Safety Invariant #1 (event-fenced deferred retirement): while an in-flight GPU reader holds a fence, old slabs
//     are NEVER passed to Pool::give or overwritten; once the fence signals, slabs are retired & sparse chunks reclaimed.
//  3. Safety Invariant #2 (shared-expert refresh): moving shared-expert row NEXP triggers refresh_shared(layer)
//     immediately after ptab::set (and never on non-shared rows).
//  4. Safety Invariant #3 (EL5 KV-shrink headroom borrow): under a tight pool cap, take_run invokes borrow_headroom
//     and succeeds without failing serving.
//  5. Post-3->4 sparse chunks (< 70% full): evacuated cleanly so Pool::reclaim_empty() drops live_chunks, and
//     idle=true raises per-step budget automatically (no "fast mode" flag).
// Zero CUDA dependency; runs under ASan+UBSan in scripts/host_tests.sh.
#include "../ptab_compact.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {
int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

struct CappedBackend {
    static size_t max_live_bytes;
    static size_t cur_live_bytes;
    static bool alloc_chunk(size_t want_bytes, bool want_vmm, uint8_t*& out_p, size_t& out_bytes, bool& out_vmm,
                            int&, const char*& err) {
        out_bytes = (want_bytes + 4095u) & ~4095u;
        if (max_live_bytes > 0 && cur_live_bytes + out_bytes > max_live_bytes) {
            err = "capped backend: out of headroom";
            return false;
        }
        out_p = static_cast<uint8_t*>(std::calloc(1, out_bytes));
        if (!out_p) { err = "calloc failed"; return false; }
        cur_live_bytes += out_bytes;
        out_vmm = want_vmm;
        return true;
    }
    static void free_chunk(uint8_t* p, bool) {
        if (!p) return;
        std::free(p);
    }
};
size_t CappedBackend::max_live_bytes = 0;
size_t CappedBackend::cur_live_bytes = 0;

struct SimpleBackend {
    static bool alloc_chunk(size_t want_bytes, bool want_vmm, uint8_t*& out_p, size_t& out_bytes, bool& out_vmm,
                            int&, const char*& err) {
        out_bytes = (want_bytes + 4095u) & ~4095u;
        out_p = static_cast<uint8_t*>(std::calloc(1, out_bytes));
        if (!out_p) { err = "calloc failed"; return false; }
        out_vmm = want_vmm;
        return true;
    }
    static void free_chunk(uint8_t* p, bool) { std::free(p); }
};

using HostPool = ptab::SlabPool<SimpleBackend>;

struct FakeLayer {
    int id = 0, rows = 0, eb = 0, shared_row = -1;
    std::vector<const uint8_t*> host;
    std::vector<size_t> bytes;
    std::vector<const uint8_t*> dev_shadow; // verifies set_piece updates device table in lockstep!
};

void fill_pattern(uint8_t* p, size_t n, uint64_t seed) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ seed;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ (i + 1)) * 0xbf58476d1ce4e5b9ull;
        p[i] = uint8_t((h >> 32) ^ h);
    }
}

uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

void verify_all_bytes(const std::vector<FakeLayer>& ls, const std::vector<std::vector<uint64_t>>& want_sums) {
    for (size_t li = 0; li < ls.size(); ++li) {
        const auto& l = ls[li];
        for (size_t i = 0; i < l.host.size(); ++i) {
            CHECK(l.host[i] == l.dev_shadow[i]);
            if (l.bytes[i]) CHECK(fnv(l.host[i], l.bytes[i]) == want_sums[li][i]);
        }
    }
}

// Test 1 & 2 & 3: Post-switch split experts + fence-gated retirement + shared-expert refresh + idle boost.
void test_split_fence_and_shared_refresh() {
    HostPool pool;
    pool.chunk_bytes = 32768;

    constexpr int N_LAYERS = 2;
    constexpr int N_ROUTED = 12;
    constexpr int SHARED_ROW = N_ROUTED;
    constexpr int ROWS = N_ROUTED + 1;
    constexpr int EB = 5;

    std::vector<FakeLayer> layers(N_LAYERS);
    std::vector<std::vector<uint64_t>> want_sums(N_LAYERS);

    // Populate a post-4->3 fragmented state: first 4 blocks of each expert live in chunk 0/1,
    // 5th block (newly prefetched!) lives in a separate chunk -> every expert is split across non-adjacent slabs!
    const char* err = nullptr;
    for (int li = 0; li < N_LAYERS; ++li) {
        auto& l = layers[li];
        l.id = 3 + li; l.rows = ROWS; l.eb = EB; l.shared_row = SHARED_ROW;
        l.host.assign(size_t(ROWS * EB * 2), nullptr);
        l.bytes.assign(size_t(ROWS * EB * 2), 0);
        l.dev_shadow.assign(size_t(ROWS * EB * 2), nullptr);
        want_sums[li].assign(size_t(ROWS * EB * 2), 0);
        for (int e = 0; e < ROWS; ++e) {
            size_t f1_sz[4] = {1024, 1024, 1024, 1024};
            size_t f2_sz[4] = {512, 512, 512, 512};
            uint8_t* f1_p[4] = {};
            uint8_t* f2_p[4] = {};
            CHECK(pool.take_run(f1_sz, 4, f1_p, err));
            CHECK(pool.take_run(f2_sz, 4, f2_p, err));
            for (int b = 0; b < 4; ++b) {
                const size_t s = size_t((e * EB + b) * 2);
                fill_pattern(f1_p[b], f1_sz[b], uint64_t(li) * 10000 + e * 100 + b * 2);
                fill_pattern(f2_p[b], f2_sz[b], uint64_t(li) * 10000 + e * 100 + b * 2 + 1);
                l.host[s] = l.dev_shadow[s] = f1_p[b]; l.bytes[s] = f1_sz[b];
                l.host[s + 1] = l.dev_shadow[s + 1] = f2_p[b]; l.bytes[s + 1] = f2_sz[b];
                want_sums[li][s] = fnv(f1_p[b], f1_sz[b]);
                want_sums[li][s + 1] = fnv(f2_p[b], f2_sz[b]);
            }
        }
    }
    // Allocate the 5th block of every expert afterwards (in later chunks -> guaranteed non-contiguous with block 3!).
    for (int li = 0; li < N_LAYERS; ++li) {
        auto& l = layers[li];
        for (int e = 0; e < ROWS; ++e) {
            uint8_t* p1 = pool.take(1024, err, /*fresh=*/true);
            uint8_t* p2 = pool.take(512, err, /*fresh=*/true);
            CHECK(p1 && p2);
            const size_t s = size_t((e * EB + 4) * 2);
            fill_pattern(p1, 1024, uint64_t(li) * 10000 + e * 100 + 8);
            fill_pattern(p2, 512, uint64_t(li) * 10000 + e * 100 + 9);
            l.host[s] = l.dev_shadow[s] = p1; l.bytes[s] = 1024;
            l.host[s + 1] = l.dev_shadow[s + 1] = p2; l.bytes[s + 1] = 512;
            want_sums[li][s] = fnv(p1, 1024);
            want_sums[li][s + 1] = fnv(p2, 512);
        }
    }

    std::vector<ptab::CompactLayerView> views(N_LAYERS);
    for (int li = 0; li < N_LAYERS; ++li) {
        views[li] = {layers[li].id, layers[li].rows, layers[li].eb, layers[li].shared_row,
                     &layers[li].host, &layers[li].bytes};
    }

    // Fence simulation: `completed_fence` tracks the highest fence id the GPU has completed.
    uint64_t next_fence = 1, completed_fence = 0;
    std::vector<size_t> refreshed_layers;

    ptab::CompactConfig cfg;
    cfg.active_experts_per_step = 2;
    cfg.idle_experts_per_step = 10;
    cfg.sparse_chunk_ratio = 0.70;

    ptab::CompactorOps<HostPool, uint64_t> ops;
    ops.copy_d2d = [](uint8_t* dst, const uint8_t* src, size_t n) -> const char* {
        std::memcpy(dst, src, n);
        return nullptr;
    };
    ops.set_piece = [&](size_t li, int e, int b, const uint8_t* f1, size_t, const uint8_t* f2, size_t) -> const char* {
        const size_t s = size_t((e * layers[li].eb + b) * 2);
        layers[li].dev_shadow[s] = f1;
        layers[li].dev_shadow[s + 1] = f2;
        return nullptr;
    };
    ops.refresh_shared = [&](size_t li) -> const char* {
        refreshed_layers.push_back(li);
        return nullptr;
    };
    ops.record_fence = [&](uint64_t& ev) -> const char* {
        ev = next_fence++;
        return nullptr;
    };
    ops.fence_ready = [&](const uint64_t& ev) -> bool { return ev <= completed_fence; };
    ops.destroy_fence = [](uint64_t& ev) { ev = 0; };

    ptab::Compactor<HostPool, uint64_t> comp(cfg, std::move(ops));

    // Step 1 (active, fence NOT yet signaled!): moves exactly 2 experts, ZERO slabs freed while fence pending!
    const uint8_t* reader_pin = layers[0].host[0]; // simulated in-flight GPU reader holding old slab 0
    const uint64_t reader_pin_sum = want_sums[0][0];
    CHECK(comp.step(views, pool, /*idle=*/false) == nullptr);
    CHECK(comp.stats().experts_moved == 2);
    CHECK(comp.stats().slabs_retired == 0);
    CHECK(comp.pending_batches() == 1);
    // Old slab held by in-flight reader MUST still be live in pool and hold intact bytes!
    CHECK(pool.chunk_of(reader_pin) >= 0);
    CHECK(fnv(reader_pin, 1024) == reader_pin_sum);
    verify_all_bytes(layers, want_sums);

    // Step 2 (active, still before fence 1 completes): moves 2 more experts, still 0 retired.
    CHECK(comp.step(views, pool, /*idle=*/false) == nullptr);
    CHECK(comp.stats().experts_moved == 4);
    CHECK(comp.stats().slabs_retired == 0);
    CHECK(comp.pending_batches() == 2);
    CHECK(fnv(reader_pin, 1024) == reader_pin_sum);
    verify_all_bytes(layers, want_sums);

    // Signal fence 1 ONLY -> batch 1 retires (2 experts * 5 blocks * 2 matrices = 20 slabs), batch 2 stays pending!
    completed_fence = 1;
    CHECK(comp.poll_retire(pool) == 20);
    CHECK(comp.pending_batches() == 1);
    CHECK(comp.stats().slabs_retired == 20);

    // Run idle steps (budget = 10 experts/step!) with fences completing each step until all split + sparse work finishes.
    const size_t moved_before_idle = comp.stats().experts_moved;
    completed_fence = next_fence;
    CHECK(comp.step(views, pool, /*idle=*/true) == nullptr);
    CHECK(comp.stats().experts_moved - moved_before_idle == 10); // idle budget boost active!
    verify_all_bytes(layers, want_sums);

    for (int iter = 0; iter < 32 && !comp.done(); ++iter) {
        completed_fence = next_fence;
        CHECK(comp.step(views, pool, /*idle=*/true) == nullptr);
        verify_all_bytes(layers, want_sums);
    }
    completed_fence = next_fence;
    comp.poll_retire(pool);

    CHECK(comp.done());
    CHECK(comp.stats().split_remaining == 0);
    CHECK(comp.stats().sparse_remaining == 0);
    // Safety Invariant #2: shared row NEXP moved in BOTH layers -> refresh_shared fired for layer 0 and layer 1!
    CHECK(comp.stats().shared_refreshes >= 2);
    bool got_l0 = false, got_l1 = false;
    for (size_t li : refreshed_layers) { if (li == 0) got_l0 = true; if (li == 1) got_l1 = true; }
    CHECK(got_l0 && got_l1);
    // Every single expert in every layer is now 100% contiguous back-to-back in memory!
    for (int li = 0; li < N_LAYERS; ++li) {
        for (int e = 0; e < ROWS; ++e) {
            int nb1 = 0, nb2 = 0;
            CHECK(ptab::row_matrix_contiguous<HostPool>(views[li], e, 0, nb1) && nb1 == EB);
            CHECK(ptab::row_matrix_contiguous<HostPool>(views[li], e, 1, nb2) && nb2 == EB);
        }
    }
    // Old fragmented chunks MUST have been reclaimed!
    CHECK(comp.stats().chunks_reclaimed > 0);
    CHECK(comp.stats().bytes_reclaimed > 0);
    verify_all_bytes(layers, want_sums);
}

// Test 4: Safety Invariant #3 — EL5 KV-shrink headroom borrow under a tight pool cap.
void test_kv_headroom_borrow() {
    CappedBackend::cur_live_bytes = 0;
    CappedBackend::max_live_bytes = 0;
    using CapPool = ptab::SlabPool<CappedBackend>;
    CapPool pool;
    pool.chunk_bytes = 16384;

    FakeLayer l;
    l.id = 3; l.rows = 4; l.eb = 2; l.shared_row = -1;
    l.host.assign(4 * 2 * 2, nullptr);
    l.bytes.assign(4 * 2 * 2, 0);
    l.dev_shadow.assign(4 * 2 * 2, nullptr);
    const char* err = nullptr;
    // Place block 0 in chunk 0, block 1 in chunk 1 (split experts) filling both chunks tightly.
    for (int e = 0; e < 4; ++e) {
        uint8_t* p1 = pool.take(2048, err, true);
        uint8_t* p2 = pool.take(2048, err, true);
        CHECK(p1 && p2);
        fill_pattern(p1, 2048, 100 + e); fill_pattern(p2, 2048, 200 + e);
        const size_t s = size_t((e * 2 + 0) * 2);
        l.host[s] = l.dev_shadow[s] = p1; l.bytes[s] = 2048;
        l.host[s + 1] = l.dev_shadow[s + 1] = p2; l.bytes[s + 1] = 2048;
    }
    for (int e = 0; e < 4; ++e) {
        uint8_t* p1 = pool.take(2048, err, true);
        uint8_t* p2 = pool.take(2048, err, true);
        CHECK(p1 && p2);
        fill_pattern(p1, 2048, 300 + e); fill_pattern(p2, 2048, 400 + e);
        const size_t s = size_t((e * 2 + 1) * 2);
        l.host[s] = l.dev_shadow[s] = p1; l.bytes[s] = 2048;
        l.host[s + 1] = l.dev_shadow[s + 1] = p2; l.bytes[s + 1] = 2048;
    }
    // Cap pool right at current live bytes so take_run MUST borrow headroom from EL5 KV shrink to proceed!
    CappedBackend::max_live_bytes = CappedBackend::cur_live_bytes;

    std::vector<ptab::CompactLayerView> views = {{l.id, l.rows, l.eb, -1, &l.host, &l.bytes}};
    ptab::CompactConfig cfg;
    cfg.active_experts_per_step = 4;
    ptab::CompactorOps<CapPool, uint64_t> ops;
    ops.copy_d2d = [](uint8_t* dst, const uint8_t* src, size_t n) -> const char* { std::memcpy(dst, src, n); return nullptr; };
    ops.set_piece = [&](size_t, int e, int b, const uint8_t* f1, size_t, const uint8_t* f2, size_t) -> const char* {
        const size_t s = size_t((e * 2 + b) * 2);
        l.dev_shadow[s] = f1; l.dev_shadow[s + 1] = f2;
        return nullptr;
    };
    ops.record_fence = [](uint64_t& ev) -> const char* { ev = 1; return nullptr; };
    ops.fence_ready = [](const uint64_t&) -> bool { return true; };
    ops.destroy_fence = [](uint64_t&) {};
    size_t borrowed_bytes = 0;
    ops.borrow_headroom = [&](size_t need) -> bool {
        // Simulate EL5 KV shrink freeing 32 KiB of physical memory.
        const size_t grant = std::max<size_t>(need, 32768);
        CappedBackend::max_live_bytes += grant;
        borrowed_bytes += grant;
        return true;
    };

    ptab::Compactor<CapPool, uint64_t> comp(cfg, std::move(ops));
    CHECK(comp.step(views, pool, /*idle=*/false) == nullptr);
    CHECK(comp.stats().headroom_borrows >= 1);
    CHECK(borrowed_bytes >= 32768);
    CHECK(comp.stats().experts_moved == 4);
    comp.poll_retire(pool);
    CHECK(comp.stats().chunks_reclaimed > 0);
}
} // namespace

int main() {
    test_split_fence_and_shared_refresh();
    test_kv_headroom_borrow();
    std::printf(g_fails ? "ptab_compact_test: %d FAIL\n" : "ptab_compact_test: PASS\n", g_fails);
    return g_fails != 0;
}
