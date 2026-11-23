// CPU host test for EL5 (#153) kv_chunks_plan.h: verifies RunPool chunk resize coordination (shrink -> drain_io ->
// sync; sync -> grow_pool with OOM rollback; busy refusal on live chunks; non-prefix multi-session masks) against the
// real kvman-rs RunPool in libg6host.a.
#include "../kv_chunks_plan.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

int main() {
    using namespace g6s2;

    // 1. Default chunk page sizing
    check(kv_chunk_pages_default(0, 0) == 0, "default_chunk_pages(0, 0) == 0");
    check(kv_chunk_pages_default(1024, 0) == 256, "default_chunk_pages(1024, 0) == 256");
    check(kv_chunk_pages_default(100, 0) == 100, "default_chunk_pages(100, 0) clamped to place");
    check(kv_chunk_pages_default(1024, 128) == 128, "default_chunk_pages(1024, 128) uses override");
    check(kv_chunk_pages_default(64, 128) == 64, "default_chunk_pages(64, 128) clamped to place");

    // 2. Shrink order: pool.shrink -> drain_io -> sync_fn; Grow order: sync_fn -> pool.grow_pool
    {
        // 400 pages below place=400, R=64 pages (4096 rows), 4 chunks of 100 pages
        RunPool pool(400, 400, 64 * 64);
        g6rt::PrefixReuse rules{};
        rules.grid = 64;
        pool.keep(rules);
        pool.stores(2);
        pool.add_tier(512, 2, "", "");   // keep=all so evicted retained runs queue spills into pool.io()
        pool.chunks(100);
        check(pool.chunk_count() == 4, "4 chunks of 100 pages");

        // Place live session A in chunk 0 (pages 0..39), filler across chunk 1..229, live session B in chunk 2
        // (230..253), filler to 299, retained run in chunk 3 (300..339), then free fillers.
        std::vector<Run> fills;
        auto fill = [&](uint32_t pages) {
            for (uint32_t left = pages; left;) {
                uint32_t n = std::min(left, 64u);
                fills.push_back(pool.reserve(n * 64, n * 64));
                left -= n;
            }
        };
        const Run ra = pool.reserve(40 * 64, 64 * 64);
        fill(230 - 40);
        const Run rb = pool.reserve(24 * 64, 64 * 64);
        fill(300 - 254);
        const Run rk = pool.reserve(40 * 64, 40 * 64);
        g6rt::PrefixReuse st{};
        st.history.resize(40 * 64 - 1, 7);
        st.valid = 40 * 64 - 2;
        st.main_end = 40 * 64 - 2;
        st.written_end = 40 * 64 - 1;
        st.draft_end = 40 * 64 - 2;
        st.pf_end = 40 * 64 - 2;
        pool.retain(rk, st);
        check(pool.kept_runs() == 1, "retained run in chunk 3 is kept before shrink");
        (void)pool.io();
        (void)pool.io_done();
        for (const Run& f : fills) pool.release(f);

        // 2a. Shrinking 3 chunks while 2 chunks hold live runs must return Busy and call neither drain nor sync.
        bool drain_called = false, sync_called = false;
        auto r_busy = kv_pool_resize_to(
            pool, 1,
            [&] { drain_called = true; },
            [&](const std::vector<uint8_t>&, uint32_t, size_t*, size_t*) {
                sync_called = true;
                return 0;
            });
        check(r_busy.status == KvResizeStatus::Busy, "shrink into live chunks returns Busy");
        check(!drain_called && !sync_called, "Busy shrink invokes neither drain nor sync");
        check(kv_mask_online(kv_pool_mask(pool)) == 4, "Busy shrink leaves all 4 chunks online");

        // 2b. Shrinking to 2 chunks takes non-prefix chunks 1 and 3 offline, drains pending spill IO BEFORE sync.
        std::vector<std::string> order;
        size_t spilled_io = 0;
        std::vector<uint8_t> synced_mask;
        auto r_shrink = kv_pool_resize_to(
            pool, 2,
            [&] {
                order.push_back("drain");
                auto io = pool.io();
                spilled_io = io.size();
                (void)pool.io_done();
            },
            [&](const std::vector<uint8_t>& mask, uint32_t n, size_t* mp, size_t* rl) {
                order.push_back("sync");
                synced_mask = mask;
                check(n == 2, "shrink sync sees online_count == 2");
                check(pool.io_pending() == 0, "pool IO already drained before sync unmaps granules");
                *mp = 0;
                *rl = 8 * 1024 * 1024;
                return 0;
            });
        check(r_shrink.status == KvResizeStatus::Ok, "shrink to 2 chunks succeeds");
        check(r_shrink.before_online == 4 && r_shrink.after_online == 2, "shrink updates online counts 4 -> 2");
        check(r_shrink.changed_chunks == std::vector<uint32_t>({1, 3}), "shrink took non-prefix chunks 1 and 3 offline");
        check(synced_mask == std::vector<uint8_t>({1, 0, 1, 0}), "synced mask matches [1,0,1,0]");
        check(order == std::vector<std::string>({"drain", "sync"}), "drain ran strictly before sync");
        check(spilled_io > 0, "evicted retained run queued spill IO drained before unmap");
        check(r_shrink.released_bytes == 8 * 1024 * 1024, "released bytes reported");

        // 2c. Noop resize to 2 chunks does nothing.
        auto r_noop = kv_pool_resize_to(
            pool, 2,
            [&] { check(false, "noop must not drain"); },
            [&](const std::vector<uint8_t>&, uint32_t, size_t*, size_t*) {
                check(false, "noop must not sync");
                return 0;
            });
        check(r_noop.status == KvResizeStatus::Noop, "resize to current online count is Noop");

        // 2d. Grow with sync OOM failure must roll back sync to previous mask and NOT grow pool.
        std::vector<std::vector<uint8_t>> masks_seen;
        int call_idx = 0;
        auto r_oom = kv_pool_resize_to(
            pool, 4, nullptr,
            [&](const std::vector<uint8_t>& mask, uint32_t, size_t*, size_t*) {
                masks_seen.push_back(mask);
                // First call (grow attempt) sees pool still at 2 online chunks (sync happens BEFORE grow_pool!)
                if (call_idx++ == 0) {
                    check(kv_mask_online(kv_pool_mask(pool)) == 2, "grow sync runs BEFORE pool.grow_pool");
                    return 2;   // simulated OOM
                }
                return 0;       // rollback call
            });
        check(r_oom.status == KvResizeStatus::SyncError && r_oom.sync_rc == 2, "grow OOM reports SyncError");
        check(masks_seen.size() == 2 && masks_seen[0] == std::vector<uint8_t>({1, 1, 1, 1}) &&
                  masks_seen[1] == std::vector<uint8_t>({1, 0, 1, 0}),
              "grow OOM rolls back mask to [1,0,1,0]");
        check(kv_pool_mask(pool) == std::vector<uint8_t>({1, 0, 1, 0}), "pool unchanged after grow OOM");

        // 2e. Partial grow by 1 chunk brings lowest offline chunk (chunk 1) back first.
        std::vector<uint8_t> grow1_mask;
        auto r_grow1 = kv_pool_resize_to(
            pool, 3, nullptr,
            [&](const std::vector<uint8_t>& mask, uint32_t n, size_t* mp, size_t* rl) {
                grow1_mask = mask;
                check(n == 3, "grow1 sync sees online_count == 3");
                check(kv_mask_online(kv_pool_mask(pool)) == 2, "grow1 sync runs before pool.grow_pool");
                *mp = 4 * 1024 * 1024;
                *rl = 0;
                return 0;
            });
        check(r_grow1.status == KvResizeStatus::Ok, "grow to 3 chunks succeeds");
        check(r_grow1.changed_chunks == std::vector<uint32_t>({1}), "grow1 brought lowest offline chunk 1 back");
        check(grow1_mask == std::vector<uint8_t>({1, 1, 1, 0}), "grow1 synced mask [1,1,1,0]");
        check(kv_pool_mask(pool) == std::vector<uint8_t>({1, 1, 1, 0}), "pool matches [1,1,1,0]");

        // 2f. Grow remaining chunk 3 back.
        auto r_grow2 = kv_pool_resize_to(
            pool, 99, nullptr,   // clamped to 4
            [&](const std::vector<uint8_t>& mask, uint32_t n, size_t* mp, size_t* rl) {
                check(n == 4 && mask == std::vector<uint8_t>({1, 1, 1, 1}), "grow2 clamped to all 4 chunks");
                *mp = 4 * 1024 * 1024;
                *rl = 0;
                return 0;
            });
        check(r_grow2.status == KvResizeStatus::Ok && r_grow2.changed_chunks == std::vector<uint32_t>({3}),
              "grow2 brings chunk 3 back");
        check(kv_pool_mask(pool) == std::vector<uint8_t>({1, 1, 1, 1}), "all 4 chunks online again");

        pool.release(ra);
        pool.release(rb);
    }

    // 3. Multi-session Flash scenario (flat non-CP pool: below_rows == place * 64): shrink down to 1 chunk while
    //    session 0 stays in chunk 0, refuse shrinking to 0 chunks, then recall chunk 1 when session 1 arrives.
    {
        RunPool pool(1024, 1024, 1024 * 64);
        pool.chunks(256);   // 4 chunks of 256 pages (16,384 tokens each)
        auto r_zero = kv_pool_resize_to(pool, 0, nullptr,
                                        [](const std::vector<uint8_t>&, uint32_t, size_t* mp, size_t* rl) {
                                            *mp = 0; *rl = 0; return 0;
                                        });
        check(r_zero.status == KvResizeStatus::Invalid, "flat pool shrink to 0 chunks is rejected as Invalid");
        const Run s0 = pool.reserve(100 * 64, 200 * 64);   // fits in chunk 0
        auto r_sh = kv_pool_resize_to(pool, 1, nullptr,
                                      [](const std::vector<uint8_t>& mask, uint32_t n, size_t* mp, size_t* rl) {
                                          check(n == 1 && mask == std::vector<uint8_t>({1, 0, 0, 0}),
                                                "multi-session Flash shrink to 1 chunk leaves chunk 0 online");
                                          *mp = 0; *rl = 3; return 0;
                                      });
        check(r_sh.status == KvResizeStatus::Ok, "multi-session Flash shrink with 1 live session in chunk 0 succeeds");
        check(!pool.fits(200 * 64, 200 * 64), "2nd session cannot fit while only 1 chunk is online");
        auto r_gr = kv_pool_resize_to(pool, 2, nullptr,
                                      [](const std::vector<uint8_t>& mask, uint32_t n, size_t* mp, size_t* rl) {
                                          check(n == 2 && mask == std::vector<uint8_t>({1, 1, 0, 0}),
                                                "recall brings chunk 1 online");
                                          *mp = 1; *rl = 0; return 0;
                                      });
        check(r_gr.status == KvResizeStatus::Ok, "recall grow to 2 chunks succeeds");
        check(pool.fits(200 * 64, 200 * 64), "2nd session fits once chunk 1 is recalled");
        const Run s1 = pool.reserve(200 * 64, 200 * 64);
        for (int p : pool.list(s1)) check(pool.chunk_online(uint32_t(p) / 256), "every page of s1 is in an online chunk");
        pool.release(s1);
        pool.release(s0);
    }

    // 4. CP pool (below_rows / 64 < place) still enforces the R/64 floor.
    {
        RunPool pool(1024, 1024, 512 * 64);
        pool.chunks(256);
        auto r_low = kv_pool_resize_to(pool, 1, nullptr,
                                       [](const std::vector<uint8_t>&, uint32_t, size_t* mp, size_t* rl) {
                                           *mp = 0; *rl = 0; return 0;
                                       });
        check(r_low.status == KvResizeStatus::Invalid, "CP pool shrink below R/64 is rejected as Invalid");
    }

    std::printf("kv_chunks_plan_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
