// Host-only tests for EL3 (#151 Stage 3) in-process delta piece-prefetch:
//  - C FFI g6l_delta_new / g6l_delta_halves / g6l_delta_free (formula & planned layouts, joiner old_nr=0)
//  - ptab::plan_table_delta + SlabPool + RouteMap::extents():
//     * 4->3 switch on staying rank: keeps all 4 resident slabs untouched (identical pointer & byte checksum),
//       fetches only ~1/12th missing pieces into contiguous per-expert take_run slabs; rollback returns pool
//       used/held to exact pre-prefetch bytes without touching resident slabs.
//     * 3->4 switch on staying rank: fetches 0 expert bytes (added_pieces == 0), post-cutover retire_dropped_slabs
//       frees all dropped slabs and reclaims empty chunks.
//     * 3->4 joiner rank (old_nr == 0): loads all target pieces contiguously per expert via take_run.
//     * Staged target table bytes are byte-for-byte identical to a fresh full load of the target layout!
// Zero CUDA dependency; runs under ASan+UBSan in scripts/host_tests.sh.
#include "../ptab_delta.h"
#include "../../loader-rs/g6load.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

struct HostBackend {
    static bool alloc_chunk(size_t want_bytes, bool want_vmm, uint8_t*& out_p, size_t& out_bytes, bool& out_vmm,
                            int&, const char*& err) {
        out_bytes = (want_bytes + 4095u) & ~4095u;
        out_p = static_cast<uint8_t*>(std::calloc(1, out_bytes));
        if (!out_p) { err = "host calloc failed"; return false; }
        out_vmm = want_vmm;
        return true;
    }
    static void free_chunk(uint8_t* p, bool) { std::free(p); }
};
using HostPool = ptab::SlabPool<HostBackend>;

constexpr size_t TEST_MB = 256; // scaled-down MB per bit (matches 256 B slab alignment)
constexpr int N_EXPERTS = 16;

uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// Deterministic bytes for global (layer, expert, half_id, is_fc2) so any layout maps the exact same half bytes.
void fill_half_bytes(uint8_t* dst, size_t n, int layer, int expert, int half_id, bool is_fc2) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t(layer) << 40) ^ (uint64_t(expert) << 24) ^
                 (uint64_t(half_id) << 8) ^ (is_fc2 ? 0xFFu : 0u);
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ (i + 1)) * 0xbf58476d1ce4e5b9ull;
        dst[i] = uint8_t((h >> 32) ^ h);
    }
}

// Builds a synthetic g6layout-v1 JSON file with explicit per-expert ownership for layer 3.
std::string write_layout_json(const std::string& path, const std::vector<std::string>& members,
                              const std::vector<std::vector<int>>& owners_by_expert) {
    std::ofstream f(path);
    f << "{\"format\": \"g6layout-v1\", \"members\": [";
    for (size_t i = 0; i < members.size(); ++i) f << (i ? ", " : "") << "\"" << members[i] << "\"";
    f << "], \"groups\": {\n";
    for (size_t e = 0; e < owners_by_expert.size(); ++e) {
        if (e) f << ",\n";
        f << "\"halves 3 " << e << "\": [";
        for (size_t h = 0; h < owners_by_expert[e].size(); ++h) {
            if (h) f << ", ";
            f << owners_by_expert[e][h];
        }
        f << "]";
    }
    f << "\n}}\n";
    return path;
}

struct PackBuild {
    std::vector<long long> tr_tab; // [N_EXPERTS * 6]
    std::vector<long long> kmix;   // [N_EXPERTS]
    std::vector<uint8_t> image;
    uint64_t fc1_off = 0, fc2_off = 0;
};

// Materializes a rank's logical pack image (fc1 + fc2) for layer 3 given which global halves it owns per expert.
PackBuild build_rank_pack(int layer, const std::vector<std::vector<int>>& held_halves) {
    PackBuild pb;
    pb.tr_tab.assign(N_EXPERTS * 6, 0);
    pb.kmix.assign(N_EXPERTS, 0);
    std::vector<uint8_t> fc1, fc2;
    for (int e = 0; e < N_EXPERTS; ++e) {
        const auto& hs = held_halves[e];
        const int nb = int(hs.size());
        const long long bits = 2 + (e % 3); // K2..K4
        long long* tr = &pb.tr_tab[e * 6];
        tr[0] = static_cast<long long>(fc1.size());
        tr[1] = static_cast<long long>(fc2.size());
        tr[3] = nb;
        tr[4] = bits;
        for (int b = 0; b < nb; ++b) {
            const size_t f1b = ptab::host_piece_fc1_bytes(0, tr, b, TEST_MB);
            const size_t f2b = ptab::host_piece_fc2_bytes(tr, TEST_MB);
            const size_t o1 = fc1.size(), o2 = fc2.size();
            fc1.resize(o1 + f1b);
            fc2.resize(o2 + f2b);
            fill_half_bytes(fc1.data() + o1, f1b, layer, e, hs[b], false);
            fill_half_bytes(fc2.data() + o2, f2b, layer, e, hs[b], true);
        }
    }
    pb.fc1_off = 4096;
    pb.fc2_off = pb.fc1_off + ((fc1.size() + 4095u) & ~4095u);
    pb.image.assign(size_t(pb.fc2_off + fc2.size()), 0);
    std::memcpy(pb.image.data() + pb.fc1_off, fc1.data(), fc1.size());
    std::memcpy(pb.image.data() + pb.fc2_off, fc2.data(), fc2.size());
    return pb;
}

std::vector<ptab::ExpertDelta> query_deltas(const void* dh, int layer, int n_exp) {
    std::vector<ptab::ExpertDelta> out(n_exp);
    char err[256] = "";
    for (int e = 0; e < n_exp; ++e) {
        uint8_t ko[16], kn[16], ad[16], dr[16];
        uint32_t nk = 0, na = 0, nd = 0;
        const int rc = g6l_delta_halves(dh, uint32_t(layer), uint32_t(e), ko, kn, &nk, ad, &na, dr, &nd, err, sizeof err);
        CHECK(rc == 0);
        out[e].row = e;
        for (uint32_t i = 0; i < nk; ++i) out[e].kept.push_back({ko[i], kn[i]});
        for (uint32_t i = 0; i < na; ++i) out[e].added.push_back(ad[i]);
        for (uint32_t i = 0; i < nd; ++i) out[e].dropped.push_back(dr[i]);
    }
    return out;
}

// Loads a rank's initial table via plan_table_delta from empty (joiner) so all slabs live in `pool`.
ptab::TableDeltaResult load_fresh(HostPool& pool, const PackBuild& pb, int eb, const void* dh_join, int layer) {
    const auto deltas = query_deltas(dh_join, layer, N_EXPERTS);
    ptab::RouteMap route;
    ptab::TableDeltaResult res;
    std::vector<const uint8_t*> empty_h;
    std::vector<size_t> empty_b;
    const char* err = ptab::plan_table_delta(empty_h, empty_b, 0, 0, pb.tr_tab, pb.kmix, N_EXPERTS, eb,
                                             deltas, pool, pb.fc1_off, pb.fc2_off, TEST_MB, route, res);
    CHECK(err == nullptr);
    std::string rerr;
    CHECK(route.finalize(rerr));
    CHECK(route.dispatch(0, pb.image.data(), pb.image.size(), [](uint8_t* dst, const uint8_t* src, size_t n) {
        std::memcpy(dst, src, n);
        return 0;
    }) == 0);
    return res;
}
} // namespace

int main() {
    char tmpl[] = "/tmp/ptab_delta_XXXXXX";
    CHECK(::mkdtemp(tmpl) != nullptr);
    const std::string dir = tmpl;

    // Build explicit TP4 layout (4 halves/rank) and planned TP3 layout (keeping ALL 4 halves on staying ranks 0..2
    // and distributing rank 3's 4 halves across ranks 0..2: 1 or 2 added per expert = 1/12th of model per staying rank).
    std::vector<std::vector<int>> own4(N_EXPERTS, std::vector<int>(16));
    std::vector<std::vector<int>> own3(N_EXPERTS, std::vector<int>(16));
    for (int e = 0; e < N_EXPERTS; ++e) {
        int rr = e % 3;
        for (int h = 0; h < 16; ++h) {
            const int r4 = (16 * e + h + 3) % 4;
            own4[e][h] = r4;
            own3[e][h] = (r4 < 3) ? r4 : (rr++ % 3);
        }
    }
    const std::string f4 = write_layout_json(dir + "/l4.json", {"a", "b", "c", "d"}, own4);
    const std::string f3 = write_layout_json(dir + "/l3.json", {"a", "b", "c"}, own3);

    auto held_of = [](const std::vector<std::vector<int>>& own, int rank) {
        std::vector<std::vector<int>> out(N_EXPERTS);
        for (int e = 0; e < N_EXPERTS; ++e)
            for (int h = 0; h < 16; ++h)
                if (own[e][h] == rank) out[e].push_back(h);
        return out;
    };

    const PackBuild pk4_r0 = build_rank_pack(3, held_of(own4, 0));
    const PackBuild pk3_r0 = build_rank_pack(3, held_of(own3, 0));
    const PackBuild pk4_r3 = build_rank_pack(3, held_of(own4, 3));

    // 1. Fresh load of TP4 rank 0 into HostPool (using small 32 KiB chunks so chunk reclamation is exercised).
    HostPool pool;
    pool.chunk_bytes = 32768;
    char err[256] = "";
    void* dh_init4 = g6l_delta_new(nullptr, 0, 0, f4.c_str(), 4, 0, err, sizeof err);
    CHECK(dh_init4 != nullptr);
    const auto tab4 = load_fresh(pool, pk4_r0, /*eb=*/4, dh_init4, 3);
    g6l_delta_free(dh_init4);
    CHECK(tab4.kept_pieces == 0 && tab4.added_pieces == size_t(N_EXPERTS * 4));

    // Record checksums & addresses of every resident TP4 slab before prefetch.
    std::vector<const uint8_t*> orig_ptrs = tab4.host;
    std::vector<uint64_t> orig_sums(tab4.host.size(), 0);
    for (size_t i = 0; i < tab4.host.size(); ++i)
        if (tab4.host[i]) orig_sums[i] = fnv(tab4.host[i], tab4.bytes[i]);
    const size_t used_before = pool.used;
    const size_t held_before = pool.held;

    // 2. Delta prefetch 4 -> 3 on staying rank 0 + verify mid-prefetch rollback restores exact pool state!
    void* dh_4to3 = g6l_delta_new(f4.c_str(), 4, 0, f3.c_str(), 3, 0, err, sizeof err);
    CHECK(dh_4to3 != nullptr);
    const auto d_4to3 = query_deltas(dh_4to3, 3, N_EXPERTS);
    {
        ptab::RouteMap route_rb;
        ptab::TableDeltaResult rb;
        const char* pe = ptab::plan_table_delta(tab4.host, tab4.bytes, N_EXPERTS, 4, pk3_r0.tr_tab, pk3_r0.kmix,
                                                N_EXPERTS, 6, d_4to3, pool, pk3_r0.fc1_off, pk3_r0.fc2_off, TEST_MB,
                                                route_rb, rb);
        CHECK(pe == nullptr);
        CHECK(rb.kept_pieces == size_t(N_EXPERTS * 4));
        CHECK(rb.dropped_pieces == 0);
        CHECK(rb.added_pieces > 0 && rb.added_pieces < size_t(N_EXPERTS * 2));
        // Roll back mid-prefetch: pool.used must return to exact used_before, all resident slabs untouched!
        ptab::rollback_delta_slabs(pool, rb.added_slabs);
        CHECK(pool.used == used_before);
        for (size_t i = 0; i < orig_ptrs.size(); ++i)
            if (orig_ptrs[i]) CHECK(fnv(orig_ptrs[i], tab4.bytes[i]) == orig_sums[i]);
    }

    // 3. Real 4 -> 3 delta prefetch on staying rank 0: stream ONLY route_extents (missing pieces!) and verify
    //    every piece in staged target table is byte-for-byte identical to a fresh TP3 load!
    ptab::RouteMap route_3;
    ptab::TableDeltaResult tab3;
    CHECK(ptab::plan_table_delta(tab4.host, tab4.bytes, N_EXPERTS, 4, pk3_r0.tr_tab, pk3_r0.kmix,
                                 N_EXPERTS, 6, d_4to3, pool, pk3_r0.fc1_off, pk3_r0.fc2_off, TEST_MB,
                                 route_3, tab3) == nullptr);
    std::string rerr;
    CHECK(route_3.finalize(rerr));
    const auto exts = ptab::route_extents(route_3);
    uint64_t routed_bytes = 0;
    for (const auto& [off, nb] : exts) routed_bytes += nb;
    CHECK(routed_bytes == tab3.added_bytes);
    CHECK(routed_bytes < tab3.kept_bytes); // 4->3 fetches only ~1/4 of target (~1/12 of model), NOT kept slabs!
    // Feed ONLY the missing extents into route_3 (proving kept slabs are NEVER overwritten).
    for (const auto& [off, nb] : exts) {
        CHECK(route_3.dispatch(off, pk3_r0.image.data() + off, size_t(nb), [](uint8_t* dst, const uint8_t* src, size_t n) {
            std::memcpy(dst, src, n);
            return 0;
        }) == 0);
    }
    // Verify kept slabs are STILL at exact original addresses & checksums!
    for (int e = 0; e < N_EXPERTS; ++e) {
        for (const auto& pk : d_4to3[e].kept) {
            const size_t os = size_t(e * 4 + pk.old_b) * 2;
            const size_t ns = size_t(tab3.slot(e, pk.new_b));
            CHECK(tab3.host[ns] == orig_ptrs[os] && tab3.host[ns + 1] == orig_ptrs[os + 1]);
            CHECK(fnv(tab3.host[ns], tab3.bytes[ns]) == orig_sums[os]);
            CHECK(fnv(tab3.host[ns + 1], tab3.bytes[ns + 1]) == orig_sums[os + 1]);
        }
        // Verify added pieces of an expert with >=2 added pieces are contiguous (take_run)!
        if (d_4to3[e].added.size() >= 2) {
            const int b0 = d_4to3[e].added[0], b1 = d_4to3[e].added[1];
            const size_t s0 = size_t(tab3.slot(e, b0)), s1 = size_t(tab3.slot(e, b1));
            CHECK(tab3.host[s1] == tab3.host[s0] + HostPool::round(tab3.bytes[s0]));
            CHECK(tab3.host[s1 + 1] == tab3.host[s0 + 1] + HostPool::round(tab3.bytes[s0 + 1]));
        }
    }
    // Compare every single piece of tab3 against an independent fresh TP3 load!
    HostPool ref_pool;
    ref_pool.chunk_bytes = 32768;
    void* dh_ref3 = g6l_delta_new(nullptr, 0, 0, f3.c_str(), 3, 0, err, sizeof err);
    const auto ref3 = load_fresh(ref_pool, pk3_r0, /*eb=*/6, dh_ref3, 3);
    g6l_delta_free(dh_ref3);
    CHECK(tab3.host.size() == ref3.host.size());
    for (size_t i = 0; i < tab3.host.size(); ++i) {
        CHECK(tab3.bytes[i] == ref3.bytes[i]);
        if (tab3.bytes[i]) CHECK(std::memcmp(tab3.host[i], ref3.host[i], tab3.bytes[i]) == 0);
    }
    g6l_delta_free(dh_4to3);

    // 4. Reverse switch 3 -> 4 on staying rank 0: MUST fetch 0 expert bytes (added_pieces == 0),
    //    and post-cutover retire_dropped_slabs frees all dropped slabs & reclaims empty chunks!
    void* dh_3to4 = g6l_delta_new(f3.c_str(), 3, 0, f4.c_str(), 4, 0, err, sizeof err);
    CHECK(dh_3to4 != nullptr);
    const auto d_3to4 = query_deltas(dh_3to4, 3, N_EXPERTS);
    ptab::RouteMap route_back4;
    ptab::TableDeltaResult back4;
    CHECK(ptab::plan_table_delta(tab3.host, tab3.bytes, N_EXPERTS, 6, pk4_r0.tr_tab, pk4_r0.kmix,
                                 N_EXPERTS, 4, d_3to4, pool, pk4_r0.fc1_off, pk4_r0.fc2_off, TEST_MB,
                                 route_back4, back4) == nullptr);
    CHECK(back4.added_pieces == 0 && back4.added_bytes == 0);
    CHECK(back4.kept_pieces == size_t(N_EXPERTS * 4));
    CHECK(back4.dropped_pieces == tab3.added_pieces);
    CHECK(route_back4.segs.empty()); // zero bytes routed for staying rank on 3->4!
    size_t chunks_reclaimed = 0;
    ptab::retire_dropped_slabs(pool, back4.dropped_slabs, &chunks_reclaimed);
    CHECK(pool.used == used_before);
    CHECK(pool.held <= held_before + pool.chunk_bytes);
    // Verify back4 still matches original TP4 bytes bit-for-bit!
    for (size_t i = 0; i < back4.host.size(); ++i) {
        CHECK(back4.host[i] == orig_ptrs[i]);
        if (back4.host[i]) CHECK(fnv(back4.host[i], back4.bytes[i]) == orig_sums[i]);
    }
    g6l_delta_free(dh_3to4);

    // 5. Joiner rank 3 on 3 -> 4 (old_nr == 0): all 4 pieces/expert loaded contiguously via take_run!
    HostPool join_pool;
    join_pool.chunk_bytes = 65536;
    void* dh_join = g6l_delta_new(nullptr, 0, 0, f4.c_str(), 4, 3, err, sizeof err);
    CHECK(dh_join != nullptr);
    const auto tab_join = load_fresh(join_pool, pk4_r3, /*eb=*/4, dh_join, 3);
    g6l_delta_free(dh_join);
    CHECK(tab_join.kept_pieces == 0 && tab_join.added_pieces == size_t(N_EXPERTS * 4));
    for (int e = 0; e < N_EXPERTS; ++e) {
        for (int b = 1; b < 4; ++b) {
            const size_t prev = size_t(tab_join.slot(e, b - 1));
            const size_t cur = size_t(tab_join.slot(e, b));
            CHECK(tab_join.host[cur] == tab_join.host[prev] + HostPool::round(tab_join.bytes[prev]));
            CHECK(tab_join.host[cur + 1] == tab_join.host[prev + 1] + HostPool::round(tab_join.bytes[prev + 1]));
        }
    }

    ::unlink(f4.c_str());
    ::unlink(f3.c_str());
    ::rmdir(dir.c_str());
    std::printf(g_fails ? "ptab_delta_test: %d FAIL\n" : "ptab_delta_test: PASS\n", g_fails);
    return g_fails != 0;
}
