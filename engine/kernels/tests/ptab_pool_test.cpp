// CPU host test for EL3 (#151) ptab_pool.h: SlabPool per-chunk liveness tracking, contiguous-per-expert take_run,
// individual piece give() (width switch / partial drop), reclaim_empty() chunk release + free-list pruning, oversize
// slabs, and RouteMap direct-load byte-range routing across arbitrary chunk splits and padding gaps.
#include "../ptab_pool.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* what)
{
    if (ok) { ++g_pass; }
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", what); }
}

struct HostBackend
{
    static inline size_t alloc_calls = 0;
    static inline size_t free_calls = 0;
    static inline size_t live_bytes = 0;
    static inline std::map<uint8_t*, size_t> sizes;

    static void reset()
    {
        alloc_calls = free_calls = live_bytes = 0;
        sizes.clear();
    }

    static bool alloc_chunk(size_t want_bytes, bool want_vmm, uint8_t*& out_p, size_t& out_bytes, bool& out_vmm,
                            int&, const char*& err)
    {
        // Round VMM chunks to 4096 B to exercise granularity rounding.
        const size_t sz = want_vmm ? ((want_bytes + 4095) & ~size_t(4095)) : want_bytes;
        void* p = nullptr;
        if (posix_memalign(&p, ptab::SLAB_ALIGN, sz) != 0 || !p)
        {
            err = "HostBackend: alloc failed";
            return false;
        }
        std::memset(p, 0, sz);
        out_p = static_cast<uint8_t*>(p);
        out_bytes = sz;
        out_vmm = want_vmm;
        ++alloc_calls;
        live_bytes += sz;
        sizes[out_p] = sz;
        return true;
    }

    static void free_chunk(uint8_t* p, bool)
    {
        if (!p) return;
        auto it = sizes.find(p);
        check(it != sizes.end(), "HostBackend::free_chunk sees valid allocated pointer");
        if (it != sizes.end())
        {
            live_bytes -= it->second;
            sizes.erase(it);
        }
        ++free_calls;
        std::free(p);
    }
};

int main()
{
    // 1. parse_tr3_layer_name
    {
        int l = -1; bool fc2 = false;
        check(ptab::parse_tr3_layer_name("L003.tr3.fc1", l, fc2) && l == 3 && !fc2, "parse L003.tr3.fc1");
        check(ptab::parse_tr3_layer_name("L078.tr3.fc2", l, fc2) && l == 78 && fc2, "parse L078.tr3.fc2");
        check(!ptab::parse_tr3_layer_name("L003.tr3.table", l, fc2), "reject L003.tr3.table");
        check(!ptab::parse_tr3_layer_name("G.emb", l, fc2), "reject G.emb");
        check(!ptab::parse_tr3_layer_name("L00a.tr3.fc1", l, fc2), "reject non-numeric layer");
    }

    // 2. SlabPool chunk-level tracking + take_run contiguity + partial give + reclaim_empty
    {
        HostBackend::reset();
        ptab::SlabPool<HostBackend> pool;
        pool.chunk_bytes = 4096;
        pool.vmm = true;

        const char* err = nullptr;
        // Expert 0: 4 blocks of 512 B = 2048 B contiguous inside chunk 0
        const size_t sz4[4] = {512, 512, 512, 512};
        uint8_t* e0[4] = {};
        check(pool.take_run(sz4, 4, e0, err) && !err, "take_run expert 0 succeeds");
        check(pool.live_chunks() == 1 && pool.chunk_of(e0[0]) == 0, "expert 0 in chunk 0");
        for (int b = 1; b < 4; ++b)
            check(e0[b] == e0[b - 1] + 512, "expert 0 blocks are physically contiguous");
        check(pool.chunks[0].live_slabs == 4 && pool.chunks[0].live_bytes == 2048, "chunk 0 live counts = 4 slabs / 2048 B");

        // Expert 1: 4 blocks of 512 B = 2048 B -> fills chunk 0 completely (4096 B)
        uint8_t* e1[4] = {};
        check(pool.take_run(sz4, 4, e1, err) && !err, "take_run expert 1 fits chunk 0");
        check(pool.live_chunks() == 1 && pool.chunk_of(e1[0]) == 0, "expert 1 also in chunk 0");
        check(pool.chunks[0].live_slabs == 8 && pool.chunks[0].live_bytes == 4096, "chunk 0 full at 8 slabs / 4096 B");

        // Expert 2: 3 blocks of 1024 B = 3072 B -> triggers chunk 1
        const size_t sz3[3] = {1024, 1024, 1024};
        uint8_t* e2[3] = {};
        check(pool.take_run(sz3, 3, e2, err) && !err, "take_run expert 2 allocates chunk 1");
        check(pool.live_chunks() == 2 && pool.chunk_of(e2[0]) == 1, "expert 2 in chunk 1");
        check(e2[1] == e2[0] + 1024 && e2[2] == e2[1] + 1024, "expert 2 blocks contiguous");

        // Simulate 4->3 width switch: drop block 3 of expert 0 and expert 1 (chunk 0 still has 6 live slabs)
        pool.give(e0[3], 512);
        pool.give(e1[3], 512);
        check(pool.chunks[0].live_slabs == 6 && pool.chunks[0].live_bytes == 3072, "chunk 0 tracks partial drop (6 live)");
        size_t freed_chunks = 99;
        check(pool.reclaim_empty(&freed_chunks) == 0 && freed_chunks == 0, "reclaim_empty keeps non-empty chunks");
        check(pool.live_chunks() == 2, "still 2 live chunks");

        // Simulate EL3b compaction: move remaining 6 slabs out of chunk 0 and give() them
        for (int b = 0; b < 3; ++b)
        {
            pool.give(e0[b], 512);
            pool.give(e1[b], 512);
        }
        check(pool.chunks[0].live_slabs == 0 && pool.chunks[0].live_bytes == 0, "chunk 0 reaches 0 live slabs");
        const size_t rel = pool.reclaim_empty(&freed_chunks);
        check(rel == 4096 && freed_chunks == 1, "reclaim_empty frees empty chunk 0 (4096 B)");
        check(pool.live_chunks() == 1 && pool.held == 4096 && pool.freed == 0, "free_by_size pruned for reclaimed chunk 0");
        check(pool.chunk_of(e0[0]) == -1, "reclaimed chunk pointer no longer resolves in chunk_of");

        // Allocate a new 512 B slab: must come from chunk 1 (NOT stale free_by_size from reclaimed chunk 0!)
        uint8_t* pnew = pool.take(512, err);
        check(pnew != nullptr && pool.chunk_of(pnew) == 1, "new slab after reclaim comes from live chunk 1");
        pool.give(pnew, 512);

        // Oversize slab gets its own exact chunk and does not replace `cur`
        uint8_t* big = pool.take(8192, err);
        check(big != nullptr && pool.chunk_of(big) == 2 && pool.cur == 1, "oversize slab gets own chunk, preserves cur");
        pool.give(big, 8192);
        const size_t rel_big = pool.reclaim_empty(&freed_chunks);
        check(rel_big == 8192 && freed_chunks == 1, "reclaim_empty frees oversize chunk");

        // Free expert 2 -> chunk 1 (also cur) becomes empty and is reclaimed
        for (int b = 0; b < 3; ++b) pool.give(e2[b], 1024);
        const size_t rel_last = pool.reclaim_empty(&freed_chunks);
        check(rel_last == 4096 && freed_chunks == 1 && pool.live_chunks() == 0 && pool.cur == -1,
              "reclaim_empty frees last chunk and resets cur=-1");
        check(pool.used == 0 && pool.freed == 0 && pool.held == 0 && HostBackend::live_bytes == 0,
              "all memory cleanly accounted and freed");
    }

    // 3. RouteMap direct-load byte-range dispatch across arbitrary chunk boundaries and padding gaps
    {
        std::vector<uint8_t> logical(5000, 0);
        for (size_t i = 0; i < logical.size(); ++i) logical[i] = static_cast<uint8_t>((i * 131 + 17) & 0xff);

        // Dense tensor A at [0, 700), padding [700, 1024), expert slab 0 at [1024, 2100),
        // expert slab 1 at [2100, 3300), padding [3300, 4096), dense tensor B at [4096, 4800)
        std::vector<uint8_t> denseA(700, 0), slab0(1076, 0), slab1(1200, 0), denseB(704, 0);
        ptab::RouteMap route;
        // Add intentionally out of order to verify finalize() sorts properly.
        route.add(4096, denseB.size(), denseB.data());
        route.add(1024, slab0.size(), slab0.data());
        route.add(0, denseA.size(), denseA.data());
        route.add(2100, slab1.size(), slab1.data());
        std::string rerr;
        check(route.finalize(rerr), "RouteMap::finalize accepts non-overlapping segments");

        // Stream in odd-sized chunks (333 B) crossing every boundary and gap.
        for (size_t off = 0; off < logical.size(); off += 333)
        {
            const size_t len = std::min<size_t>(333, logical.size() - off);
            const int rc = route.dispatch(off, logical.data() + off, len, [](uint8_t* dst, const uint8_t* src, size_t n) {
                std::memcpy(dst, src, n);
                return 0;
            });
            check(rc == 0, "RouteMap::dispatch succeeds");
        }
        check(std::memcmp(denseA.data(), logical.data() + 0, denseA.size()) == 0, "denseA bytes match");
        check(std::memcmp(slab0.data(), logical.data() + 1024, slab0.size()) == 0, "slab0 bytes match");
        check(std::memcmp(slab1.data(), logical.data() + 2100, slab1.size()) == 0, "slab1 bytes match");
        check(std::memcmp(denseB.data(), logical.data() + 4096, denseB.size()) == 0, "denseB bytes match");

        // Overlap rejection
        ptab::RouteMap bad;
        bad.add(100, 50, denseA.data());
        bad.add(140, 50, denseB.data());
        check(!bad.finalize(rerr), "RouteMap::finalize rejects overlapping segments");
    }

    std::printf("ptab_pool_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
