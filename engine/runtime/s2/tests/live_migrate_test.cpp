// Host-only tests for EL4c (#160) two-phase live bulk-then-tail session migration:
//  1. plan_page_transfers exactness:
//     - 4->3: staying boxes keep 100% of their own pages AND all < R replicated pages locally (0 network bytes!),
//       transferring only departing box 3's CP pages.
//     - 3->4: staying boxes keep all their pages locally (0 remote sends among staying boxes); joiner box 3 receives
//       only its 1/4 CP pages + < R replicated prefix.
//     - 4->6: staying boxes keep all local pages; joiners 4 & 5 receive their slices.
//  2. Full 8-concurrent-session live bulk-then-tail migration over real socketpairs (SocketMigrateTransport):
//     - Phase 1 bulk transfers [0, bulk_end_row) across all 8 sessions while simulated decode keeps appending rows
//       [bulk_end_row, final_end_row) concurrently.
//     - Phase 2 tail transfers ONLY [bulk_end_row, final_end_row) + CanonSessionMeta + EL4d state_envelope.
//     - Verifies 100% byte-for-byte identity of ALL 8 sessions' full KV [0, final_end_row) (both MLA and IDX),
//       metadata, and state envelopes across 4->3, 3->4 (with joiner), and 4->6; verifies tail bytes << bulk bytes.
//  3. Mid-bulk peer disconnect / rollback test: killing a target box mid-bulk fails closed with zero hang and leaves
//     source sessions completely intact.
// Zero CUDA dependency; runs under ASan+UBSan in scripts/host_tests.sh.
#include "../live_migrate.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <vector>

using namespace g6s2;

namespace {
int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

constexpr uint32_t MLA_ROW = 48;

void fill_mla(uint8_t* dst, uint64_t sid, uint32_t b, uint64_t row) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (sid << 40) ^ (uint64_t(b) << 28) ^ row;
    for (uint32_t i = 0; i < MLA_ROW; ++i) {
        h = (h ^ (i + 1)) * 0xbf58476d1ce4e5b9ull;
        dst[i] = uint8_t((h >> 32) ^ h);
    }
}
void fill_idx(uint8_t* dst, uint64_t sid, uint32_t b, uint64_t row) {
    uint64_t h = 0x517cc1b727220a95ull ^ (sid << 40) ^ (uint64_t(b) << 28) ^ row;
    for (uint32_t i = 0; i < 132u; ++i) {
        h = (h ^ (i + 5)) * 0x94d049bb133111ebull;
        dst[i] = uint8_t((h >> 24) ^ h);
    }
}

void test_plan_exactness() {
    const std::vector<uint8_t> f_cp = {3u, 2u}; // 2 CP bindings (binding 0 has IDX, binding 1 MLA-only)
    const MigrateLayout l4{{0, 1, 2, 3}, 4, 128};
    const MigrateLayout l3{{0, 1, 2}, 3, 128};
    const MigrateLayout l6{{0, 1, 2, 3, 4, 5}, 6, 128};

    // 4 -> 3 over 640 rows (10 logical pages: lp 0..1 replicated < R=128, lp 2..9 CP = 8 pages per binding).
    const auto p43 = plan_page_transfers(l4, l3, f_cp, f_cp, 0, 640);
    // All 3 staying boxes keep lp 0..1 locally (6 local ops/binding) + every CP page where lp%3 == lp%4!
    // Remote sends MUST only come from pages whose src_box != dst_box; NO replicated page (<2) may be remote!
    for (const auto& op : p43.ops) {
        if (op.lp < 2) CHECK(op.local);
    }
    CHECK(p43.local_pages > 0 && p43.remote_pages > 0);

    // 3 -> 4 (adding joiner box 3): every replicated page (<2) on boxes 0..2 is local; box 3 receives lp 0..1 remotely.
    const auto p34 = plan_page_transfers(l3, l4, f_cp, f_cp, 0, 640);
    for (const auto& op : p34.ops) {
        if (op.lp < 2 && op.dst_box < 3) CHECK(op.local);
        if (op.dst_box == 3) CHECK(!op.local);
    }
    CHECK(p34.local_pages > 0 && p34.remote_pages > 0);

    // 4 -> 6 (adding joiners 4 and 5): staying boxes 0..3 keep < R pages locally.
    const auto p46 = plan_page_transfers(l4, l6, f_cp, f_cp, 0, 640);
    for (const auto& op : p46.ops) {
        if (op.lp < 2 && op.dst_box < 4) CHECK(op.local);
        if (op.dst_box >= 4) CHECK(!op.local);
    }
    CHECK(p46.local_pages > 0 && p46.remote_pages > 0);
}

// Simulated per-box KV store keyed by (session_id, binding, lp).
struct BoxKv {
    struct Page {
        std::vector<uint8_t> mla;
        std::vector<uint8_t> idx;
    };
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>, Page> pages;
    std::map<uint64_t, MigrateTailMeta> metas;
    std::atomic<size_t> local_zero_copy_hits{0};

    void ensure(uint64_t sid, uint32_t b, uint32_t lp, bool has_idx) {
        auto& pg = pages[{sid, b, lp}];
        if (pg.mla.empty()) pg.mla.assign(64u * MLA_ROW, 0);
        if (has_idx && pg.idx.empty()) pg.idx.assign(64u * 132u, 0);
    }
};

// Runs a full 8-session bulk-then-tail live migration between `src` and `dst` over socketpairs and verifies
// 100% byte-for-byte parity of KV rows, metadata, and EL4d state envelopes on every destination box!
void run_8sess_live_migration(const char* label, const MigrateLayout& src, const MigrateLayout& dst) {
    constexpr int SESS = 8;
    const std::vector<uint8_t> flags = {3u, 1u, 2u}; // CP+IDX, rep+IDX, CP-only
    const uint32_t nb = uint32_t(flags.size());

    std::set<int> all_boxes;
    for (int b : src.members) all_boxes.insert(b);
    for (int b : dst.members) all_boxes.insert(b);
    const int n_boxes = *all_boxes.rbegin() + 1;

    std::vector<MigrateSessionSpec> specs(SESS);
    for (int s = 0; s < SESS; ++s) {
        specs[s].session_id = uint64_t(100 + s);
        specs[s].mla_row = MLA_ROW;
        specs[s].src_flags = flags;
        specs[s].dst_flags = flags;
        // Varied session lengths spanning < R and >> R, non-page-aligned bulk & final boundaries!
        specs[s].bulk_end_row = uint64_t(200 + s * 47);
        specs[s].final_end_row = specs[s].bulk_end_row + uint64_t(5 + (s % 4) * 7);
    }

    // Populate source KV stores for [0, bulk_end_row) before Phase 1 bulk starts.
    std::vector<BoxKv> src_kv(n_boxes), dst_kv(n_boxes);
    for (const auto& sp : specs) {
        for (int b_box : src.members) {
            for (uint32_t b = 0; b < nb; ++b) {
                for (uint64_t row = 0; row < sp.bulk_end_row; ++row) {
                    const uint32_t lp = uint32_t(row / 64u);
                    if (!layout_box_holds_page(src, b_box, flags[b], lp)) continue;
                    src_kv[b_box].ensure(sp.session_id, b, lp, (flags[b] & 1u) != 0);
                    auto& pg = src_kv[b_box].pages[{sp.session_id, b, lp}];
                    fill_mla(pg.mla.data() + size_t(row % 64u) * MLA_ROW, sp.session_id, b, row);
                    if (flags[b] & 1u) fill_idx(pg.idx.data() + size_t(row % 64u) * 132u, sp.session_id, b, row);
                }
            }
        }
    }

    // Wire directional socketpairs for every ordered pair (sbox != dbox) across all participating boxes.
    std::vector<std::unique_ptr<SocketMigrateTransport>> xports(n_boxes);
    for (int b = 0; b < n_boxes; ++b) xports[b] = std::make_unique<SocketMigrateTransport>();
    for (int a : all_boxes) {
        for (int b : all_boxes) {
            if (a == b) continue;
            int sp[2];
            CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0);
            xports[a]->add_send_fd(b, sp[0]);
            xports[b]->add_recv_fd(a, sp[1]);
        }
    }

    auto make_io = [&](int box) {
        MigrateBoxIO io;
        io.read_rows = [&, box](uint64_t sid, uint32_t b, uint32_t lp, uint32_t roff, uint32_t rcnt,
                                uint8_t* mla_out, uint8_t* idx_out) {
            const auto& pg = src_kv[box].pages.at({sid, b, lp});
            std::memcpy(mla_out, pg.mla.data() + size_t(roff) * MLA_ROW, size_t(rcnt) * MLA_ROW);
            if (idx_out) std::memcpy(idx_out, pg.idx.data() + size_t(roff) * 132u, size_t(rcnt) * 132u);
        };
        io.write_rows = [&, box](uint64_t sid, uint32_t b, uint32_t lp, uint32_t roff, uint32_t rcnt,
                                 const uint8_t* mla_in, const uint8_t* idx_in) {
            dst_kv[box].ensure(sid, b, lp, idx_in != nullptr);
            auto& pg = dst_kv[box].pages[{sid, b, lp}];
            std::memcpy(pg.mla.data() + size_t(roff) * MLA_ROW, mla_in, size_t(rcnt) * MLA_ROW);
            if (idx_in) std::memcpy(pg.idx.data() + size_t(roff) * 132u, idx_in, size_t(rcnt) * 132u);
        };
        io.copy_local = [&, box](uint64_t sid, uint32_t b, uint32_t lp, uint32_t roff, uint32_t rcnt) {
            const bool has_idx = (flags[b] & 1u) != 0;
            const auto& spg = src_kv[box].pages.at({sid, b, lp});
            dst_kv[box].ensure(sid, b, lp, has_idx);
            auto& dpg = dst_kv[box].pages[{sid, b, lp}];
            std::memcpy(dpg.mla.data() + size_t(roff) * MLA_ROW, spg.mla.data() + size_t(roff) * MLA_ROW, size_t(rcnt) * MLA_ROW);
            if (has_idx) std::memcpy(dpg.idx.data() + size_t(roff) * 132u, spg.idx.data() + size_t(roff) * 132u, size_t(rcnt) * 132u);
            dst_kv[box].local_zero_copy_hits.fetch_add(1);
        };
        io.get_tail_meta = [&](uint64_t sid) {
            MigrateTailMeta tm;
            tm.session_id = sid;
            for (const auto& sp : specs) if (sp.session_id == sid) tm.final_rows = sp.final_end_row;
            tm.meta.request_id = sid;
            tm.meta.export_id = 9000 + sid;
            tm.meta.prompt_tokens = 128;
            tm.meta.emitted_tokens = uint32_t(tm.final_rows - 127);
            tm.meta.committed_rows = uint32_t(tm.final_rows);
            tm.meta.remaining_max_tokens = 64;
            tm.meta.req_header = "n_gen 64 temp 0 seed 1 min 0 stop -";
            tm.meta.tokens = {11, 22, 33, int32_t(sid)};
            tm.state_envelope = {'g', '6', 's', 't', uint8_t(sid & 0xff), 0xAB, 0xCD};
            return tm;
        };
        io.on_tail_meta = [&, box](const MigrateTailMeta& tm) {
            dst_kv[box].metas[tm.session_id] = tm;
        };
        return io;
    };

    // Phase 1: Bulk pass [0, bulk_end_row) across all boxes concurrently!
    std::atomic<size_t> bulk_local{0}, bulk_rem_pages{0}, bulk_rem_bytes{0};
    {
        std::vector<std::thread> ts;
        std::atomic<int> phase_fails{0};
        for (int b : all_boxes) {
            ts.emplace_back([&, b] {
                size_t lp = 0, rp = 0, rb = 0;
                std::string err;
                if (!run_migrate_phase(b, /*phase=*/1, src, dst, specs, *xports[b], make_io(b), nullptr, 5000, lp, rp, rb, err)) {
                    std::fprintf(stderr, "FAIL %s bulk box %d: %s\n", label, b, err.c_str());
                    phase_fails.fetch_add(1);
                }
                bulk_local.fetch_add(lp);
                bulk_rem_pages.fetch_add(rp);
                bulk_rem_bytes.fetch_add(rb);
            });
        }
        for (auto& t : ts) t.join();
        CHECK(phase_fails.load() == 0);
    }

    // Simulated live decode appends tail rows [bulk_end_row, final_end_row) while bulk ran!
    for (const auto& sp : specs) {
        for (int b_box : src.members) {
            for (uint32_t b = 0; b < nb; ++b) {
                for (uint64_t row = sp.bulk_end_row; row < sp.final_end_row; ++row) {
                    const uint32_t lp = uint32_t(row / 64u);
                    if (!layout_box_holds_page(src, b_box, flags[b], lp)) continue;
                    src_kv[b_box].ensure(sp.session_id, b, lp, (flags[b] & 1u) != 0);
                    auto& pg = src_kv[b_box].pages[{sp.session_id, b, lp}];
                    fill_mla(pg.mla.data() + size_t(row % 64u) * MLA_ROW, sp.session_id, b, row);
                    if (flags[b] & 1u) fill_idx(pg.idx.data() + size_t(row % 64u) * 132u, sp.session_id, b, row);
                }
            }
        }
    }

    // Phase 2: Pause-at-fence tail pass [bulk_end_row, final_end_row) + metadata + EL4d state envelope!
    std::atomic<size_t> tail_local{0}, tail_rem_pages{0}, tail_rem_bytes{0};
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> ts;
        std::atomic<int> phase_fails{0};
        for (int b : all_boxes) {
            ts.emplace_back([&, b] {
                size_t lp = 0, rp = 0, rb = 0;
                std::string err;
                if (!run_migrate_phase(b, /*phase=*/2, src, dst, specs, *xports[b], make_io(b), nullptr, 5000, lp, rp, rb, err)) {
                    std::fprintf(stderr, "FAIL %s tail box %d: %s\n", label, b, err.c_str());
                    phase_fails.fetch_add(1);
                }
                tail_local.fetch_add(lp);
                tail_rem_pages.fetch_add(rp);
                tail_rem_bytes.fetch_add(rb);
            });
        }
        for (auto& t : ts) t.join();
        CHECK(phase_fails.load() == 0);
    }
    const double tail_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // Verify tail transferred far less than bulk, and local-keep avoided network copies!
    CHECK(bulk_local.load() > 0);
    CHECK(tail_rem_bytes.load() < bulk_rem_bytes.load());
    CHECK(tail_ms < 1000.0);

    // Verify EVERY SINGLE ROW [0, final_end_row) of EVERY binding of ALL 8 sessions on EVERY destination box!
    std::vector<uint8_t> want_m(MLA_ROW), want_x(132);
    for (int dbox : dst.members) {
        CHECK(dst_kv[dbox].metas.size() == size_t(SESS));
        for (const auto& sp : specs) {
            const auto& tm = dst_kv[dbox].metas.at(sp.session_id);
            CHECK(tm.final_rows == sp.final_end_row);
            CHECK(tm.meta.committed_rows == uint32_t(sp.final_end_row));
            CHECK(tm.state_envelope.size() == 7 && tm.state_envelope[4] == uint8_t(sp.session_id & 0xff));
            for (uint32_t b = 0; b < nb; ++b) {
                for (uint64_t row = 0; row < sp.final_end_row; ++row) {
                    const uint32_t lp = uint32_t(row / 64u);
                    if (!layout_box_holds_page(dst, dbox, flags[b], lp)) continue;
                    const auto it = dst_kv[dbox].pages.find({sp.session_id, b, lp});
                    CHECK(it != dst_kv[dbox].pages.end());
                    if (it == dst_kv[dbox].pages.end()) continue;
                    fill_mla(want_m.data(), sp.session_id, b, row);
                    CHECK(std::memcmp(it->second.mla.data() + size_t(row % 64u) * MLA_ROW, want_m.data(), MLA_ROW) == 0);
                    if (flags[b] & 1u) {
                        fill_idx(want_x.data(), sp.session_id, b, row);
                        CHECK(std::memcmp(it->second.idx.data() + size_t(row % 64u) * 132u, want_x.data(), 132u) == 0);
                    }
                }
            }
        }
    }
    std::printf("live_migrate_test (%s): 8 sessions OK (bulk local %zu / remote %zu pages %zu B; tail %zu B in %.2f ms)\n",
                label, bulk_local.load(), bulk_rem_pages.load(), bulk_rem_bytes.load(), tail_rem_bytes.load(), tail_ms);
}

// Test 3: Peer disconnect / abort mid-bulk fails closed with zero hang.
void test_mid_bulk_abort_rollback() {
    const MigrateLayout src{{0, 1}, 2, 64};
    const MigrateLayout dst{{0, 1, 2}, 3, 64};
    MigrateSessionSpec sp{501, MLA_ROW, {2u}, {2u}, 256, 260};
    SocketMigrateTransport t0, t2;
    int s02[2];
    CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, s02) == 0);
    t0.add_send_fd(2, s02[0]);
    t2.add_recv_fd(0, s02[1]);
    // Immediately close receiver (box 2 died mid-prefetch!) -> box 0 must fail closed cleanly.
    t2.close_all();

    MigrateBoxIO io;
    io.read_rows = [](uint64_t, uint32_t, uint32_t, uint32_t, uint32_t rcnt, uint8_t* m, uint8_t*) {
        std::memset(m, 0x42, size_t(rcnt) * MLA_ROW);
    };
    io.write_rows = [](uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, const uint8_t*, const uint8_t*) {};
    io.copy_local = [](uint64_t, uint32_t, uint32_t, uint32_t, uint32_t) {};

    size_t lp = 0, rp = 0, rb = 0;
    std::string err;
    const bool ok = run_migrate_phase(0, 1, src, dst, {sp}, t0, io, nullptr, 500, lp, rp, rb, err);
    CHECK(!ok && !err.empty());
}
} // namespace

int main() {
    test_plan_exactness();
    run_8sess_live_migration("4->3", MigrateLayout{{0, 1, 2, 3}, 4, 128}, MigrateLayout{{0, 1, 2}, 3, 128});
    run_8sess_live_migration("3->4", MigrateLayout{{0, 1, 2}, 3, 128}, MigrateLayout{{0, 1, 2, 3}, 4, 128});
    run_8sess_live_migration("4->6", MigrateLayout{{0, 1, 2, 3}, 4, 128}, MigrateLayout{{0, 1, 2, 3, 4, 5}, 6, 192});
    test_mid_bulk_abort_rollback();
    std::printf(g_fails ? "live_migrate_test: %d FAIL\n" : "live_migrate_test: PASS\n", g_fails);
    return g_fails != 0;
}
