// End-to-end host tests for EL4b live session KV export & import (#159):
//  1. Compact owned-page planning (`canon_plan_owned`), shard encode/decode, multi-rank assembly
//     (`canon_assemble_shards`), and per-rank unpack (`canon_unpack_for_rank`) across sparse shuffled physical
//     page tables and cross-width CP/replicated transforms (TP4 CP -> TP3 CP, TP4 CP -> TP6 CP, TP3 CP -> TP4 rep,
//     TP4 rep -> TP3 CP, and partial tail range first_row > 0), plus check=true mismatch detection.
//  2. Full multi-rank `Service` loopback test:
//     - Cluster A (2 ranks, CP R=64) serves req 1 and req 2 concurrently -> live mid-decode `EXPORT` of req 1 while
//       both sessions keep decoding uninterrupted -> assembles shards into a `"g6canon1"` blob and reads `export.<id>.meta`.
//     - Cluster B (3 ranks, different width, CP R=128) is already actively decoding req 10 -> live `IMPORT` of req 20
//       (built via `canon_make_import_req(meta)` + the canonical blob) -> verifies `import_session` runs on all 3 ranks,
//       zero prefill calls run for req 20, req 20 emits the exact remaining tokens req 1 emitted on Cluster A, and
//       req 10 finishes unaffected.
//  3. Refusal & edge cases: `EXPORT` of unknown req, `IMPORT` with missing/corrupt/oversized canonical blob.
// Host only, no CUDA; runs under ASan + UBSan in scripts/host_tests.sh.
#include "../service.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;

namespace {
int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

std::string mktemp_dir(const char* prefix) {
    std::string tmpl = std::string("/tmp/") + prefix + "XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    require(::mkdtemp(buf.data()) != nullptr, "mkdtemp");
    return std::string(buf.data());
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}

// Deterministic synthetic bytes for logical (binding b, logical row r) so any 1-byte shift or corruption is caught.
void fill_mla_row(uint8_t* dst, uint32_t mla_row, uint32_t b, uint64_t row, uint32_t tag) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t(b) << 40) ^ (row << 12) ^ tag;
    for (uint32_t i = 0; i < mla_row; ++i) {
        h = (h ^ (i + 1)) * 0xbf58476d1ce4e5b9ull;
        dst[i] = uint8_t((h >> 32) ^ (h >> 8));
    }
}
void fill_idx_row(uint8_t* dst, uint32_t b, uint64_t row, uint32_t tag) {
    uint64_t h = 0x517cc1b727220a95ull ^ (uint64_t(b) << 40) ^ (row << 12) ^ tag;
    for (uint32_t i = 0; i < 132u; ++i) {
        h = (h ^ (i + 7)) * 0x94d049bb133111ebull;
        dst[i] = uint8_t((h >> 24) ^ h);
    }
}

// Resolve physical page id in `list` (with `place`) for binding `b` (CP if flags[b] & 2) at logical page `lp`.
// Returns -1 if this rank does not own `lp` on binding `b`.
int64_t owner_phys(const std::vector<uint32_t>& list, uint32_t place, uint8_t flag, uint32_t nr, uint32_t rank,
                   uint32_t below_rows, uint32_t lp) {
    const bool is_cp = (flag & 2u) != 0 && nr > 0;
    const uint32_t rp = below_rows / 64u;
    if (!is_cp || lp < rp) return int64_t(list[lp]);
    if ((lp % nr) != rank) return -1;
    uint32_t j = 0;
    for (uint32_t p = rp; p < lp; ++p) if ((p % nr) == rank) ++j;
    uint32_t seen_low = 0;
    for (uint32_t phys : list) {
        if (phys < place) {
            if (seen_low == rp + j) return int64_t(phys);
            ++seen_low;
        }
    }
    return -1;
}

// Helper to build a sparse physical page list of length `n_pages` with at least `need_low` pages `< place`
// and shuffled physical page numbers inside `[0, pool_pages)`.
std::vector<uint32_t> make_sparse_list(uint64_t seed, uint32_t pool_pages, uint32_t place, uint32_t n_pages, uint32_t need_low) {
    std::mt19937_64 g(seed);
    std::vector<uint32_t> lows, highs;
    for (uint32_t p = 0; p < place && p < pool_pages; ++p) lows.push_back(p);
    for (uint32_t p = place; p < pool_pages; ++p) highs.push_back(p);
    std::shuffle(lows.begin(), lows.end(), g);
    std::shuffle(highs.begin(), highs.end(), g);
    require(lows.size() >= need_low, "not enough low pages");
    std::vector<uint32_t> out;
    out.reserve(n_pages);
    for (uint32_t i = 0; i < need_low && i < n_pages; ++i) out.push_back(lows[i]);
    uint32_t li = need_low, hi = 0;
    while (out.size() < n_pages) {
        if (hi < highs.size() && (g() % 2 == 0 || li >= lows.size())) out.push_back(highs[hi++]);
        else out.push_back(lows[li++]);
    }
    std::shuffle(out.begin(), out.end(), g);
    return out;
}

// ---------- Test 1: Cross-width compact plan + shard assembly + unpack ----------
void run_transform_case(const char* name, uint32_t src_nr, uint32_t src_r, uint32_t dst_nr, uint32_t dst_r,
                        uint64_t first_row, uint64_t rows, uint64_t seed) {
    constexpr uint32_t MLA_ROW = 64;
    constexpr uint32_t POOL_PAGES = 128;
    constexpr uint32_t SRC_PLACE = 64;
    constexpr uint32_t DST_PLACE = 72;
    const std::vector<uint8_t> src_flags = {
        uint8_t(1u | (src_nr ? 2u : 0u)), // binding 0: MLA + IDX + (CP if src_nr > 0)
        uint8_t(1u),                       // binding 1: MLA + IDX (non-CP, replicated)
        uint8_t(src_nr ? 2u : 0u),         // binding 2: MLA + (CP if src_nr > 0), no IDX
    };
    const std::vector<uint8_t> dst_flags = {
        uint8_t(1u | (dst_nr ? 2u : 0u)),
        uint8_t(1u),
        uint8_t(dst_nr ? 2u : 0u),
    };
    const uint32_t nb = uint32_t(src_flags.size());
    const uint32_t end_page = uint32_t((first_row + rows + 63u) / 64u);
    const uint32_t list_len = end_page + 2;

    const auto src_list = make_sparse_list(seed, POOL_PAGES, SRC_PLACE, list_len, list_len);
    const auto dst_list = make_sparse_list(seed + 101, POOL_PAGES, DST_PLACE, list_len, list_len);

    const uint32_t s_ranks = src_nr ? src_nr : 1;
    const uint32_t d_ranks = dst_nr ? dst_nr : 1;

    // Populate simulated source pools for each rank.
    std::vector<std::vector<std::vector<uint8_t>>> src_mla(s_ranks, std::vector<std::vector<uint8_t>>(nb));
    std::vector<std::vector<std::vector<uint8_t>>> src_idx(s_ranks, std::vector<std::vector<uint8_t>>(nb));
    for (uint32_t r = 0; r < s_ranks; ++r) {
        for (uint32_t b = 0; b < nb; ++b) {
            src_mla[r][b].assign(size_t(POOL_PAGES) * 64u * MLA_ROW, 0xCC);
            if (src_flags[b] & 1u) src_idx[r][b].assign(size_t(POOL_PAGES) * 64u * 132u, 0xDD);
            for (uint64_t row = first_row; row < first_row + rows; ++row) {
                const uint32_t lp = uint32_t(row / 64u);
                const uint32_t off = uint32_t(row % 64u);
                const int64_t phys = owner_phys(src_list, SRC_PLACE, src_flags[b], src_nr, r, src_r, lp);
                if (phys < 0) continue;
                fill_mla_row(src_mla[r][b].data() + (size_t(phys) * 64u + off) * MLA_ROW, MLA_ROW, b, row, uint32_t(seed));
                if (src_flags[b] & 1u)
                    fill_idx_row(src_idx[r][b].data() + (size_t(phys) * 64u + off) * 132u, b, row, uint32_t(seed));
            }
        }
    }

    // Export each source rank using canon_plan_owned + shard binary round-trip.
    std::vector<CanonShard> shards;
    shards.reserve(s_ranks);
    for (uint32_t r = 0; r < s_ranks; ++r) {
        const CanonPlan plan = canon_plan_owned(src_flags.data(), nb, src_nr, src_r,
                                                src_list.data(), src_list.size(), SRC_PLACE, r, first_row, rows);
        CanonShard sh;
        sh.mla_row = MLA_ROW;
        sh.nb = nb;
        sh.nr = src_nr;
        sh.rank = r;
        sh.below_rows = src_r;
        sh.compact_place = plan.compact_place;
        sh.first_row = first_row;
        sh.rows = rows;
        sh.owned_pages = uint32_t(plan.phys_pages.size());
        sh.flags = src_flags;
        sh.compact_list = plan.compact_list;
        sh.mla.resize(nb);
        sh.idx.resize(nb);
        for (uint32_t b = 0; b < nb; ++b) {
            sh.mla[b].resize(size_t(sh.owned_pages) * 64u * MLA_ROW);
            if (src_flags[b] & 1u) sh.idx[b].resize(size_t(sh.owned_pages) * 64u * 132u);
            for (size_t k = 0; k < plan.phys_pages.size(); ++k) {
                const uint32_t phys = plan.phys_pages[k];
                std::memcpy(sh.mla[b].data() + k * 64u * MLA_ROW,
                            src_mla[r][b].data() + size_t(phys) * 64u * MLA_ROW, 64u * MLA_ROW);
                if (src_flags[b] & 1u) {
                    std::memcpy(sh.idx[b].data() + k * 64u * 132u,
                                src_idx[r][b].data() + size_t(phys) * 64u * 132u, 64u * 132u);
                }
            }
        }
        const auto wire = canon_shard_encode(sh);
        shards.push_back(canon_shard_decode(wire.data(), wire.size()));
    }

    const auto blob = canon_assemble_shards(shards, /*check=*/true);
    CanonBlobInfo info;
    std::string err;
    CHECK(canon_inspect_blob(blob.data(), blob.size(), info, err));
    CHECK(info.mla_row == MLA_ROW && info.nb == nb && info.first == first_row && info.rows == rows);

    // Negative check: corrupting a replicated byte on rank 1 must be caught when check=true and s_ranks > 1.
    if (s_ranks > 1 && first_row == 0 && rows > 0 && !shards[1].mla[1].empty()) {
        auto bad = shards;
        bad[1].mla[1][0] ^= 0x55;
        bool caught = false;
        try { (void)canon_assemble_shards(bad, /*check=*/true); } catch (const std::exception&) { caught = true; }
        CHECK(caught);
    }

    // Unpack onto destination cluster (different width, different R, different sparse physical pages!).
    std::vector<std::vector<std::vector<uint8_t>>> dst_mla(d_ranks, std::vector<std::vector<uint8_t>>(nb));
    std::vector<std::vector<std::vector<uint8_t>>> dst_idx(d_ranks, std::vector<std::vector<uint8_t>>(nb));
    for (uint32_t r = 0; r < d_ranks; ++r) {
        const CanonUnpacked un = canon_unpack_for_rank(blob.data(), blob.size(), MLA_ROW, nb, dst_flags.data(),
                                                       dst_nr, dst_r, dst_list.data(), dst_list.size(), DST_PLACE, r);
        for (uint32_t b = 0; b < nb; ++b) {
            dst_mla[r][b].assign(size_t(POOL_PAGES) * 64u * MLA_ROW, 0xEE);
            if (dst_flags[b] & 1u) dst_idx[r][b].assign(size_t(POOL_PAGES) * 64u * 132u, 0xFF);
            for (size_t k = 0; k < un.plan.phys_pages.size(); ++k) {
                const uint32_t phys = un.plan.phys_pages[k];
                std::memcpy(dst_mla[r][b].data() + size_t(phys) * 64u * MLA_ROW,
                            un.mla[b].data() + k * 64u * MLA_ROW, 64u * MLA_ROW);
                if (dst_flags[b] & 1u) {
                    std::memcpy(dst_idx[r][b].data() + size_t(phys) * 64u * 132u,
                                un.idx[b].data() + k * 64u * 132u, 64u * 132u);
                }
            }
        }
    }

    // Re-export from destination cluster and verify the canonical blob is byte-for-byte identical!
    std::vector<CanonShard> dst_shards;
    dst_shards.reserve(d_ranks);
    for (uint32_t r = 0; r < d_ranks; ++r) {
        const CanonPlan plan = canon_plan_owned(dst_flags.data(), nb, dst_nr, dst_r,
                                                dst_list.data(), dst_list.size(), DST_PLACE, r, first_row, rows);
        CanonShard sh;
        sh.mla_row = MLA_ROW;
        sh.nb = nb;
        sh.nr = dst_nr;
        sh.rank = r;
        sh.below_rows = dst_r;
        sh.compact_place = plan.compact_place;
        sh.first_row = first_row;
        sh.rows = rows;
        sh.owned_pages = uint32_t(plan.phys_pages.size());
        sh.flags = dst_flags;
        sh.compact_list = plan.compact_list;
        sh.mla.resize(nb);
        sh.idx.resize(nb);
        for (uint32_t b = 0; b < nb; ++b) {
            sh.mla[b].resize(size_t(sh.owned_pages) * 64u * MLA_ROW);
            if (dst_flags[b] & 1u) sh.idx[b].resize(size_t(sh.owned_pages) * 64u * 132u);
            for (size_t k = 0; k < plan.phys_pages.size(); ++k) {
                const uint32_t phys = plan.phys_pages[k];
                std::memcpy(sh.mla[b].data() + k * 64u * MLA_ROW,
                            dst_mla[r][b].data() + size_t(phys) * 64u * MLA_ROW, 64u * MLA_ROW);
                if (dst_flags[b] & 1u) {
                    std::memcpy(sh.idx[b].data() + k * 64u * 132u,
                                dst_idx[r][b].data() + size_t(phys) * 64u * 132u, 64u * 132u);
                }
            }
        }
        dst_shards.push_back(std::move(sh));
    }
    const auto blob2 = canon_assemble_shards(dst_shards, /*check=*/true);
    CHECK(blob == blob2);
    if (blob != blob2) std::fprintf(stderr, "FAIL transform case %s: blob mismatch (%zu vs %zu)\n", name, blob.size(), blob2.size());
}

void test_canon_transforms() {
    run_transform_case("tp4_cp_to_tp3_cp",   4, 128, 3, 128, 0,   400, 1001);
    run_transform_case("tp4_cp_to_tp6_cp",   4, 128, 6, 192, 0,   400, 2002);
    run_transform_case("tp3_cp_to_tp4_rep",  3,  64, 0, 128, 0,   300, 3003);
    run_transform_case("tp4_rep_to_tp3_cp",  0, 128, 3, 128, 0,   300, 4004);
    run_transform_case("tail_only_tp4_tp3",  4, 128, 3, 128, 128, 150, 5005);
    run_transform_case("tp6_cp_to_tp4_cp",   6, 192, 4, 128, 0,   520, 6006);
}

// ---------- Test 2 & 3: Multi-rank Service EXPORT mid-decode + cross-width IMPORT ----------
struct RankBarrier {
    std::mutex mu;
    std::condition_variable cv;
    unsigned total = 1, waiting = 0, gen = 0;
    explicit RankBarrier(unsigned n) : total(n) {}
    void arrive() {
        std::unique_lock<std::mutex> lk(mu);
        const unsigned g = gen;
        if (++waiting == total) {
            waiting = 0;
            ++gen;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return gen != g; });
        }
    }
};

struct MockCluster {
    static constexpr uint32_t MLA_ROW = 32;
    static constexpr uint32_t POOL_PAGES = 64;
    static constexpr uint32_t PLACE_PAGES = 64;

    unsigned nr = 0;
    uint32_t below_rows = 0;
    uint64_t epoch = 1;
    std::string dir;
    std::vector<BootstrapConfig> cfg;
    std::vector<ControlLinks> links;
    std::vector<std::unique_ptr<RunPool>> pools;
    std::vector<std::vector<uint8_t>> kv; // [rank][POOL_PAGES * 64 * MLA_ROW]
    std::vector<std::array<std::vector<uint32_t>, MAX_SESSIONS>> slot_pages;
    std::vector<std::unique_ptr<Service>> svcs;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errs;
    std::unique_ptr<RankBarrier> bar;

    std::atomic<unsigned> next_cmd{0};
    std::atomic<unsigned> export_hooks{0};
    std::atomic<unsigned> import_hooks{0};
    std::mutex mu;
    std::map<uint64_t, unsigned> prefill_calls;
    std::map<uint64_t, std::vector<int>> emitted_by_req;
    std::function<void(uint64_t req_id, size_t emitted_count)> on_emit_r0;

    static void write_row_pattern(uint8_t* dst, int tok, int pos) {
        uint64_t h = 0x123456789abcdef0ull ^ (uint64_t(uint32_t(tok)) << 32) ^ uint64_t(uint32_t(pos));
        for (uint32_t i = 0; i < MLA_ROW; ++i) {
            h = (h ^ (i + 3)) * 1099511628211ull;
            dst[i] = uint8_t((h >> 24) ^ h);
        }
    }

    int predict_from_kv(unsigned slot, int pos) const {
        // Hash every byte of logical rows [0 .. pos] read from whichever rank owns each logical page.
        uint64_t h = 14695981039346656037ull;
        const uint8_t flag = 2u; // CP binding (MLA only, no IDX)
        for (int p = 0; p <= pos; ++p) {
            const uint32_t lp = uint32_t(p / 64);
            const uint32_t off = uint32_t(p % 64);
            const uint32_t rp = below_rows / 64u;
            const uint32_t owner = (lp < rp) ? 0u : (lp % nr);
            const auto& list = slot_pages[owner][slot];
            const int64_t phys = owner_phys(list, PLACE_PAGES, flag, nr, owner, below_rows, lp);
            require(phys >= 0, "predict_from_kv: owner missing page");
            const uint8_t* row = kv[owner].data() + (size_t(phys) * 64u + off) * MLA_ROW;
            for (uint32_t i = 0; i < MLA_ROW; ++i) h = (h ^ row[i]) * 1099511628211ull;
            // For replicated pages (< rp), also verify rank 0..nr-1 hold identical bytes!
            if (lp < rp) {
                for (unsigned r = 1; r < nr; ++r) {
                    const int64_t pr = owner_phys(slot_pages[r][slot], PLACE_PAGES, flag, nr, r, below_rows, lp);
                    require(pr >= 0, "replicated page missing");
                    const uint8_t* row_r = kv[r].data() + (size_t(pr) * 64u + off) * MLA_ROW;
                    require(std::memcmp(row, row_r, MLA_ROW) == 0, "replicated KV page diverged across ranks");
                }
            }
        }
        return int(1000 + (h % 5000));
    }

    void start(unsigned n_ranks, uint32_t r_rows, uint64_t ep) {
        nr = n_ranks;
        below_rows = r_rows;
        epoch = ep;
        dir = mktemp_dir("el4b_svc_");
        bar = std::make_unique<RankBarrier>(nr);
        cfg.resize(nr);
        links.resize(nr);
        pools.resize(nr);
        kv.assign(nr, std::vector<uint8_t>(size_t(POOL_PAGES) * 64u * MLA_ROW, 0));
        slot_pages.resize(nr);
        svcs.resize(nr);
        errs.resize(nr);

        for (unsigned r = 1; r < nr; ++r) {
            int sp[2];
            require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0, "socketpair");
            links[0].links.push_back(std::make_unique<ControlIO>(sp[0]));
            links[r].links.push_back(std::make_unique<ControlIO>(sp[1]));
        }

        Config sc;
        sc.epoch = epoch;
        sc.pool_rows = POOL_PAGES * PAGE;
        sc.max_rows = 16;
        sc.max_sessions = 4;
        sc.mtp = 1;
        sc.prefill_rows = 64;
        sc.two_lanes = false;

        for (unsigned r = 0; r < nr; ++r) {
            cfg[r].rank = r;
            cfg[r].ranks = nr;
            cfg[r].head_ip = "127.0.0.1";
            cfg[r].port = 1;
            cfg[r].directory = (r == 0) ? dir : (dir + "/r" + std::to_string(r));
            if (r > 0) ::mkdir(cfg[r].directory.c_str(), 0700);
            cfg[r].runtime_signature = "el4b_test";
            cfg[r].scheduler = sc;

            pools[r] = std::make_unique<RunPool>(POOL_PAGES, PLACE_PAGES, below_rows);
            pools[r]->tail(nr, 0);

            NativeHooks h;
            h.pool.pool = pools[r].get();
            h.pool.grow = true;
            h.bind = [this, r](const Grant& g) {
                const auto v = pools[r]->list(g.run);
                slot_pages[r][g.key.slot].assign(v.begin(), v.end());
            };
            h.grown = [this, r](unsigned slot, const Run& run) {
                const auto v = pools[r]->list(run);
                slot_pages[r][slot].assign(v.begin(), v.end());
            };
            h.retire = [this, r](const Slot& s) {
                slot_pages[r][s.key.slot].clear();
            };
            h.fence = [] {};
            h.export_session = [this, r](const Slot& s, const Export& ex) {
                const uint8_t flags[1] = {2u}; // CP MLA-only binding
                const auto& list = slot_pages[r][s.key.slot];
                const CanonPlan plan = canon_plan_owned(flags, 1, nr, below_rows, list.data(), list.size(),
                                                        PLACE_PAGES, r, ex.first_row, ex.rows);
                CanonShard sh;
                sh.mla_row = MLA_ROW;
                sh.nb = 1;
                sh.nr = nr;
                sh.rank = r;
                sh.below_rows = below_rows;
                sh.compact_place = plan.compact_place;
                sh.first_row = ex.first_row;
                sh.rows = ex.rows;
                sh.owned_pages = uint32_t(plan.phys_pages.size());
                sh.flags = {2u};
                sh.compact_list = plan.compact_list;
                sh.mla.resize(1);
                sh.idx.resize(1);
                sh.mla[0].resize(size_t(sh.owned_pages) * 64u * MLA_ROW);
                for (size_t k = 0; k < plan.phys_pages.size(); ++k) {
                    const uint32_t phys = plan.phys_pages[k];
                    std::memcpy(sh.mla[0].data() + k * 64u * MLA_ROW,
                                kv[r].data() + size_t(phys) * 64u * MLA_ROW, 64u * MLA_ROW);
                }
                canon_shard_write(dir + "/export." + std::to_string(ex.export_id) + ".r" + std::to_string(r) + ".shard", sh);
                export_hooks.fetch_add(1);
            };
            h.import_session = [this, r](const Grant& g) {
                const std::string path = dir + "/import." + std::to_string(g.req.id) + ".canon";
                std::vector<uint8_t> blob;
                std::string err;
                require(canon_read_file_bytes(path, blob, err), err.c_str());
                const uint8_t flags[1] = {2u};
                const auto& list = slot_pages[r][g.key.slot];
                const CanonUnpacked un = canon_unpack_for_rank(blob.data(), blob.size(), MLA_ROW, 1, flags,
                                                               nr, below_rows, list.data(), list.size(), PLACE_PAGES, r);
                for (size_t k = 0; k < un.plan.phys_pages.size(); ++k) {
                    const uint32_t phys = un.plan.phys_pages[k];
                    std::memcpy(kv[r].data() + size_t(phys) * 64u * MLA_ROW,
                                un.mla[0].data() + k * 64u * MLA_ROW, 64u * MLA_ROW);
                }
                import_hooks.fetch_add(1);
            };
            h.dispatch.bucket = [](bool, int) { return 0; };
            h.dispatch.mode = [](int, int) {};
            h.dispatch.prefill = [this, r](const Slot& s, const Payload& pay, uint32_t first, uint32_t rows) {
                if (r == 0) {
                    std::lock_guard<std::mutex> lk(mu);
                    prefill_calls[s.req.id] += 1;
                }
                const auto& list = slot_pages[r][s.key.slot];
                for (uint32_t p = first; p < first + rows; ++p) {
                    const uint32_t lp = p / 64u;
                    const uint32_t off = p % 64u;
                    const int64_t phys = owner_phys(list, PLACE_PAGES, 2u, nr, r, below_rows, lp);
                    if (phys >= 0) {
                        write_row_pattern(kv[r].data() + (size_t(phys) * 64u + off) * MLA_ROW, pay.prompt[p], int(p));
                    }
                }
                bar->arrive();
            };
            h.dispatch.launch = [this, r](const Launch& l) {
                // Phase 1: each rank writes its own owned KV rows for this step (zero data race across ranks).
                for (const auto& row : l.step.r) {
                    const uint32_t lp = uint32_t(row.pos / 64);
                    const uint32_t off = uint32_t(row.pos % 64);
                    const auto& list = slot_pages[r][unsigned(row.sess)];
                    const int64_t phys = owner_phys(list, PLACE_PAGES, 2u, nr, r, below_rows, lp);
                    if (phys >= 0) {
                        write_row_pattern(kv[r].data() + (size_t(phys) * 64u + off) * MLA_ROW, row.tok, row.pos);
                    }
                }
                bar->arrive();
                // Phase 2: all ranks read across owning ranks' KV buffers to produce identical next tokens.
                std::vector<int> out(l.step.r.size(), 0);
                for (size_t i = 0; i < l.step.r.size(); ++i) {
                    out[i] = predict_from_kv(unsigned(l.step.r[i].sess), l.step.r[i].pos);
                }
                bar->arrive();
                if (r == 0 && l.step.kind == s2::Step::Kind::MAIN) {
                    // Track newly emitted tokens on rank 0 for live test triggers (read from resp later as truth).
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                return out;
            };
            svcs[r] = std::make_unique<Service>(cfg[r], links[r], std::move(h), 1);
        }

        for (unsigned r = 0; r < nr; ++r) {
            threads.emplace_back([this, r] {
                try { svcs[r]->run(); } catch (...) { errs[r] = std::current_exception(); }
            });
        }
    }

    void write_cmd(const std::string& line) {
        const unsigned idx = next_cmd.fetch_add(1);
        atomic_file(dir + "/cmd." + std::to_string(idx), line + "\n");
    }

    void submit_admit(uint64_t id, const std::string& req_bytes) {
        atomic_file(dir + "/req." + std::to_string(id), req_bytes);
        write_cmd("ADMIT " + std::to_string(epoch) + " " + std::to_string(id) + " " + std::to_string(payload_hash(req_bytes)));
    }

    void submit_export(uint64_t req_id, uint64_t export_id) {
        write_cmd("EXPORT " + std::to_string(epoch) + " " + std::to_string(req_id) + " " + std::to_string(export_id));
    }

    void submit_import(uint64_t id, const std::string& req_bytes) {
        atomic_file(dir + "/req." + std::to_string(id), req_bytes);
        write_cmd("IMPORT " + std::to_string(epoch) + " " + std::to_string(id) + " " + std::to_string(payload_hash(req_bytes)));
    }

    static std::vector<int> parse_resp_tokens(const std::string& resp) {
        std::vector<int> toks;
        std::istringstream is(resp);
        std::string line;
        while (std::getline(is, line)) {
            if (line.rfind("t ", 0) == 0) {
                std::istringstream ls(line.substr(2));
                int id = 0;
                while (ls >> id) toks.push_back(id);
            }
        }
        return toks;
    }

    bool wait_resp_done(uint64_t id, int timeout_ms = 10000) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            const std::string s = slurp(dir + "/resp." + std::to_string(id));
            if (s.find("\nend ") != std::string::npos || s.rfind("end ", 0) == 0) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    }

    bool wait_resp_tokens(uint64_t id, size_t min_toks, int timeout_ms = 10000) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (parse_resp_tokens(slurp(dir + "/resp." + std::to_string(id))).size() >= min_toks) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    }

    bool wait_file(const std::string& path, int timeout_ms = 10000) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (::access(path.c_str(), F_OK) == 0) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    }

    void stop() {
        atomic_file(dir + "/quit", "");
        for (auto& t : threads) if (t.joinable()) t.join();
        for (unsigned r = 0; r < nr; ++r) {
            if (errs[r]) {
                try { std::rethrow_exception(errs[r]); }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "FAIL cluster rank %u threw: %s\n", r, e.what());
                    ++g_fails;
                }
            }
        }
        svcs.clear();
        pools.clear();
        links.clear();
    }
};

std::string make_req_text(const std::vector<int>& prompt, int n_gen, uint64_t seed) {
    std::ostringstream o;
    o << "n_gen " << n_gen << " temp 0 seed " << seed << " min 0 stop -\n";
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (i) o << ',';
        o << prompt[i];
    }
    o << "\n";
    return o.str();
}

void test_live_service_export_import() {
    // 1. Start Cluster A (2 ranks, CP R=64) and admit req 1 (96 prompt tokens, 18 gen) + req 2 (80 prompt, 14 gen).
    MockCluster a;
    a.start(/*nr=*/2, /*below_rows=*/64, /*epoch=*/1);

    std::vector<int> p1(96), p2(80);
    for (size_t i = 0; i < p1.size(); ++i) p1[i] = int(200 + (i * 13) % 400);
    for (size_t i = 0; i < p2.size(); ++i) p2[i] = int(700 + (i * 17) % 400);

    a.submit_admit(1, make_req_text(p1, 18, 42));
    a.submit_admit(2, make_req_text(p2, 14, 99));

    // Wait until req 1 is actively mid-decode (at least 3 tokens emitted, still well below 18) and export it live!
    CHECK(a.wait_resp_tokens(1, 3));
    a.submit_export(/*req_id=*/1, /*export_id=*/701);

    CHECK(a.wait_file(a.dir + "/export.701.meta"));
    CHECK(a.wait_file(a.dir + "/export.701.r0.shard"));
    CHECK(a.wait_file(a.dir + "/export.701.r1.shard"));

    // Both req 1 and req 2 must finish uninterrupted on Cluster A!
    CHECK(a.wait_resp_done(1));
    CHECK(a.wait_resp_done(2));
    CHECK(a.export_hooks.load() == 2);

    const std::vector<int> toks_a1 = MockCluster::parse_resp_tokens(slurp(a.dir + "/resp.1"));
    const std::vector<int> toks_a2 = MockCluster::parse_resp_tokens(slurp(a.dir + "/resp.2"));
    CHECK(toks_a1.size() == 18);
    CHECK(toks_a2.size() == 14);

    // Read export metadata and assemble the 2 rank shards into one width-independent canonical blob.
    const CanonSessionMeta meta = canon_meta_read(a.dir + "/export.701.meta");
    CHECK(meta.request_id == 1 && meta.export_id == 701);
    CHECK(meta.prompt_tokens == 96);
    CHECK(meta.emitted_tokens >= 3 && meta.emitted_tokens < 18);
    CHECK(meta.committed_rows == meta.prompt_tokens + meta.emitted_tokens - 1);
    CHECK(meta.remaining_max_tokens == 18 - meta.emitted_tokens);
    CHECK(meta.tokens.size() == meta.prompt_tokens + meta.emitted_tokens);

    std::vector<CanonShard> shards;
    shards.push_back(canon_shard_read(a.dir + "/export.701.r0.shard"));
    shards.push_back(canon_shard_read(a.dir + "/export.701.r1.shard"));
    const auto blob = canon_assemble_shards(shards, /*check=*/true);

    a.stop();

    // 2. Start Cluster B (3 ranks — DIFFERENT width! — CP R=128) serving background req 10 while importing req 20!
    MockCluster b;
    b.start(/*nr=*/3, /*below_rows=*/128, /*epoch=*/2);

    std::vector<int> p10(72);
    for (size_t i = 0; i < p10.size(); ++i) p10[i] = int(300 + (i * 19) % 400);
    b.submit_admit(10, make_req_text(p10, 16, 77));
    CHECK(b.wait_resp_tokens(10, 2));

    // Stage import.20.canon and req.20 built directly from export.701.meta, then submit IMPORT mid-decode!
    canon_write_file_atomic(b.dir + "/import.20.canon", blob.data(), blob.size());
    const std::string req20 = canon_make_import_req(meta);
    b.submit_import(20, req20);

    CHECK(b.wait_resp_done(20));
    CHECK(b.wait_resp_done(10));
    CHECK(b.import_hooks.load() == 3); // all 3 ranks ran import_session!
    {
        std::lock_guard<std::mutex> lk(b.mu);
        CHECK(b.prefill_calls[20] == 0); // zero prefill quanta ran for imported session 20!
    }

    // Imported req 20 MUST have emitted the EXACT remaining tokens that req 1 produced on Cluster A!
    const std::vector<int> toks_b20 = MockCluster::parse_resp_tokens(slurp(b.dir + "/resp.20"));
    const std::vector<int> want_tail(toks_a1.begin() + ptrdiff_t(meta.emitted_tokens), toks_a1.end());
    CHECK(toks_b20 == want_tail);
    if (toks_b20 != want_tail) {
        std::fprintf(stderr, "FAIL imported tokens mismatch: got %zu vs want %zu\n", toks_b20.size(), want_tail.size());
    }
    const std::vector<int> toks_b10 = MockCluster::parse_resp_tokens(slurp(b.dir + "/resp.10"));
    CHECK(toks_b10.size() == 16);

    // 3. Refusal & edge cases while Cluster B stays healthy and serving:
    // (a) EXPORT unknown request id -> clean no-op, no meta written.
    b.submit_export(/*req_id=*/999, /*export_id=*/702);
    // (b) IMPORT missing canonical blob -> refused with end error.
    b.submit_import(21, req20);
    CHECK(b.wait_resp_done(21));
    CHECK(slurp(b.dir + "/resp.21").find("end error") != std::string::npos);
    // (c) IMPORT corrupt canonical blob -> refused with end error.
    const std::string junk = "not-a-valid-canon-blob";
    canon_write_file_atomic(b.dir + "/import.22.canon", junk.data(), junk.size());
    b.submit_import(22, req20);
    CHECK(b.wait_resp_done(22));
    CHECK(slurp(b.dir + "/resp.22").find("end error") != std::string::npos);
    // (d) IMPORT blob with rows >= prompt_tokens -> refused with end error.
    canon_write_file_atomic(b.dir + "/import.23.canon", blob.data(), blob.size());
    std::vector<int> short_prompt(16, 101);
    b.submit_import(23, make_req_text(short_prompt, 4, 1));
    CHECK(b.wait_resp_done(23));
    CHECK(slurp(b.dir + "/resp.23").find("end error") != std::string::npos);

    // (e) Cluster B is still healthy and serves a normal request after all refusals!
    b.submit_admit(24, make_req_text(p10, 4, 88));
    CHECK(b.wait_resp_done(24));
    CHECK(MockCluster::parse_resp_tokens(slurp(b.dir + "/resp.24")).size() == 4);
    CHECK(::access((b.dir + "/export.702.meta").c_str(), F_OK) != 0);

    b.stop();
}
} // namespace

int main() {
    test_canon_transforms();
    test_live_service_export_import();
    std::printf(g_fails ? "session_export_import_test: %d FAIL\n" : "session_export_import_test: PASS\n", g_fails);
    return g_fails != 0;
}
