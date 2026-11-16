// Real-file test of the Engram row store (host only). Writes (or reuses) two w8 files of known row content in a scratch dir, then
// checks fetch() against the content: owned/foreign split, cold + warm, duplicates, tail block of an unaligned file, eviction in a
// small cache, LRU order, warm-only calls, open() errors, and a thread sweep that reports the host NVMe random-read rate.
// The rate is a proxy for the NVMe of the machine that runs the test, not the serving machine's.
//   test_rowstore [scratch dir] [rows in table 0]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include "../rowstore.h"

using namespace ds41::engram;

// T10 test seam, in this binary only: a pread() that overlaps the file bytes [g_slow_lo, g_slow_hi) before g_slow_until_ns (steady
// clock) first sleeps until then, so one fetch of those rows lasts longer than the keepalive's 250 ms active window. The range is
// one 4 KiB block, so the keepalive's own strided reads almost never hit it. RowStore itself is unchanged.
static std::atomic<int64_t> g_slow_until_ns{0}, g_slow_lo{0}, g_slow_hi{0};
static int64_t steady_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
extern "C" ssize_t pread(int fd, void* buf, size_t n, off_t off)
{
    const int64_t until = g_slow_until_ns.load(std::memory_order_acquire);
    const int64_t now = steady_ns();
    if (until > now && off < g_slow_hi.load(std::memory_order_relaxed) && off + (off_t)n > g_slow_lo.load(std::memory_order_relaxed))
        std::this_thread::sleep_for(std::chrono::nanoseconds(until - now));
    return (ssize_t)syscall(SYS_pread64, fd, buf, n, off);
}

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static uint64_t splitmix(uint64_t& x)
{
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void row_bytes(int table, int64_t gid, uint8_t* out)   // the content of global row gid of a table
{
    uint64_t s = ((uint64_t)table << 40) ^ (uint64_t)gid ^ 0x5151ull;
    for (int j = 0; j < 256; j += 8) { uint64_t v = splitmix(s); memcpy(out + j, &v, 8); }
}

struct Spec { int table; int64_t first, valid, part; std::string path; };

static bool make_file(const Spec& sp)
{
    const uint64_t want = (uint64_t)sp.part * 256;
    struct stat st;
    if (stat(sp.path.c_str(), &st) == 0 && (uint64_t)st.st_size == want)
    {
        uint8_t a[256], b[256];
        FILE* f = fopen(sp.path.c_str(), "rb");
        bool ok = f && fread(a, 1, 256, f) == 256 && fseek(f, (long)(want - 256), SEEK_SET) == 0 && fread(b, 1, 256, f) == 256;
        if (f) fclose(f);
        uint8_t x[256], y[256];
        row_bytes(sp.table, sp.first, x);
        row_bytes(sp.table, sp.first + sp.part - 1, y);
        if (ok && !memcmp(a, x, 256) && !memcmp(b, y, 256)) return true;   // reuse
    }
    FILE* f = fopen(sp.path.c_str(), "wb");
    if (!f) return false;
    std::vector<uint8_t> buf(256 * 32768);
    for (int64_t r = 0; r < sp.part; r += 32768)
    {
        const int64_t n = std::min<int64_t>(32768, sp.part - r);
        for (int64_t i = 0; i < n; ++i) row_bytes(sp.table, sp.first + r + i, &buf[i * 256]);
        if (fwrite(buf.data(), 256, n, f) != (size_t)n) { fclose(f); return false; }
    }
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    return true;
}

static bool same(int table, uint32_t id, const uint8_t* got)
{
    uint8_t w[256];
    row_bytes(table, id, w);
    return !memcmp(w, got, 256);
}

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "/tmp/engram-rowstore-test";
    const int64_t rows0 = argc > 2 ? atoll(argv[2]) : (4ll << 20);
    mkdir(dir.c_str(), 0755);
    // table 0 plays rank 3 (valid < part: pad tail rows); part = 5 mod 16 so the file ends inside a 4 KiB block
    Spec s0{0, 1000, rows0, (rows0 / 16) * 16 + 5 + 16, dir + "/t0.w8"};
    s0.part = std::max<int64_t>(s0.part, s0.valid + 5);
    Spec s1{1, 0, 100005, 100005, dir + "/t1.w8"};   // no pad rows; 100005 % 16 = 5, so the file ends inside a block of owned rows
    if (!make_file(s0) || !make_file(s1)) { fprintf(stderr, "cannot write test files in %s\n", dir.c_str()); return 2; }
    std::vector<TableFile> files = {{s0.path, s0.first, s0.valid, {}}, {s1.path, s1.first, s1.valid, {}}};
    printf("files: t0 %lld valid rows (%lld in file, %llu B, size%%4096=%llu), t1 %lld rows\n", (long long)s0.valid, (long long)s0.part,
           (unsigned long long)(s0.part * 256), (unsigned long long)((s0.part * 256) % 4096), (long long)s1.valid);

    std::string err;
    // ---- open errors
    {
        RowStore bad;
        std::vector<TableFile> f2 = {{s1.path, 0, 1000000, {}}};   // asks for more rows than the file has
        CHECK(!bad.open(f2, StoreOpts{}, &err) && err.find("fewer") != std::string::npos, "short file must fail open: %s", err.c_str());
        RowStore nofile;
        std::vector<TableFile> f3 = {{dir + "/missing.w8", 0, 1, {}}};
        CHECK(!nofile.open(f3, StoreOpts{}, &err), "missing file must fail open");
    }

    StoreOpts o;
    o.io_uring = 0; // Keep the original pread-pool regression coverage.
    o.cache_bytes = 256ull << 20;   // 1M rows: holds everything the checks below touch
    o.io_threads = 96;
    RowStore rs;
    if (!rs.open(files, o, &err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 2; }
    printf("store: block %zu B, direct t0=%d t1=%d, cache %zu sets x %d ways\n", rs.block(), rs.direct(0), rs.direct(1), rs.cache().sets(), rs.cache().ways());
    CHECK(rs.block() >= 512 && (rs.block() & (rs.block() - 1)) == 0, "block size %zu", rs.block());

    // ---- T1: cold fetch: owned rows right, foreign rows skipped, duplicates read once, edges
    {
        const uint32_t last_valid = (uint32_t)(s0.first + s0.valid - 1);
        const uint32_t last_file = (uint32_t)(s0.first + s0.part - 1);   // a pad row in the file's last (short) block
        std::vector<uint32_t> ids = {(uint32_t)s0.first, last_valid, (uint32_t)s0.first - 1, last_valid + 1, 5, 4000000000u,
                                     (uint32_t)s0.first, (uint32_t)s0.first, last_valid, (uint32_t)s0.first + 1, (uint32_t)s0.first + 2};
        std::mt19937_64 rng(7);
        for (int i = 0; i < 2000; ++i) ids.push_back((uint32_t)(s0.first + rng() % s0.valid));
        std::vector<uint8_t> dst(ids.size() * 256, 0xEE);
        FetchStats st;
        CHECK(rs.fetch(0, ids.data(), ids.size(), dst.data(), &st) == 0, "fetch err %d", st.err);
        printf("  %s\n", RowStore::line(st).c_str());
        size_t owned = 0, bad = 0;
        for (size_t i = 0; i < ids.size(); ++i)
        {
            const bool own = ids[i] >= s0.first && ids[i] < s0.first + s0.valid;
            owned += own;
            if (own) bad += !same(0, ids[i], &dst[i * 256]);
            else for (int j = 0; j < 256; ++j) if (dst[i * 256 + j] != 0xEE) { ++bad; break; }
        }
        CHECK(bad == 0, "%zu wrong rows in cold fetch", bad);
        CHECK(st.foreign == ids.size() - owned, "foreign %u vs %zu", st.foreign, ids.size() - owned);
        CHECK(st.hits + st.miss_req == owned, "hits+miss %u vs owned %zu", st.hits + st.miss_req, owned);
        CHECK(st.hits == 0, "cold cache had %u hits", st.hits);
        CHECK(st.miss_uniq < st.miss_req, "duplicates must collapse: uniq %u req %u", st.miss_uniq, st.miss_req);
        CHECK(st.blocks <= st.miss_uniq, "blocks %u > uniq rows %u", st.blocks, st.miss_uniq);
        // the pad rows after the last valid row sit in the file but are not owned: never read for a request
        (void)last_file;
        // T2: the same request again: everything owned hits, no read
        std::vector<uint8_t> dst2(ids.size() * 256, 0xEE);
        FetchStats st2;
        CHECK(rs.fetch(0, ids.data(), ids.size(), dst2.data(), &st2) == 0, "warm fetch err");
        printf("  %s\n", RowStore::line(st2).c_str());
        CHECK(st2.hits == owned && st2.blocks == 0 && st2.bytes == 0, "warm: hits %u of %zu, reads %u", st2.hits, owned, st2.blocks);
        CHECK(!memcmp(dst.data(), dst2.data(), dst.size()), "warm fetch differs from cold fetch");
    }
    // ---- T3: warm-only call (dst null), then a fetch is all hits
    {
        std::vector<uint32_t> ids;
        std::mt19937_64 rng(11);
        for (int i = 0; i < 500; ++i) ids.push_back((uint32_t)(s0.first + rng() % s0.valid));
        FetchStats w, f;
        CHECK(rs.fetch(0, ids.data(), ids.size(), nullptr, &w) == 0, "warm-only err");
        std::vector<uint8_t> dst(ids.size() * 256);
        CHECK(rs.fetch(0, ids.data(), ids.size(), dst.data(), &f) == 0, "fetch err");
        CHECK(f.hits == ids.size() && f.blocks == 0, "after warm-only: hits %u of %zu reads %u", f.hits, ids.size(), f.blocks);
        for (size_t i = 0; i < ids.size(); ++i) CHECK(same(0, ids[i], &dst[i * 256]), "warm-only row %zu wrong", i);
    }
    // ---- T4: second table, other tag space: same ids as table 0 must not collide in the cache
    {
        std::vector<uint32_t> ids = {1005, 1006, 2000, 99999, 100000, 100004};   // the last two sit in t1's short final block
        std::vector<uint8_t> d0(ids.size() * 256), d1(ids.size() * 256, 0xEE);
        FetchStats a, b;
        CHECK(rs.fetch(0, ids.data(), ids.size(), d0.data(), &a) == 0 && rs.fetch(1, ids.data(), ids.size(), d1.data(), &b) == 0, "fetch err");
        for (size_t i = 0; i < ids.size(); ++i)
        {
            CHECK(same(0, ids[i], &d0[i * 256]), "t0 row %u wrong", ids[i]);
            CHECK(same(1, ids[i], &d1[i * 256]), "t1 row %u wrong", ids[i]);
        }
        CHECK(memcmp(&d0[0], &d1[0], 256) != 0, "tables must not alias");
    }
    // ---- T5: small cache, many ids: evictions, data always right, hit rate between 0 and 1
    {
        StoreOpts so = o;
        so.cache_bytes = 1ull << 20;   // 4096 rows
        so.io_threads = 8;
        RowStore sm;
        CHECK(sm.open(files, so, &err), "open small: %s", err.c_str());
        std::mt19937_64 rng(13);
        uint64_t req = 0, hit = 0;
        for (int round = 0; round < 30; ++round)
        {
            std::vector<uint32_t> ids;
            for (int i = 0; i < 3000; ++i) ids.push_back((uint32_t)(s0.first + (rng() % 6000)));   // 6000-row hot set > 4096 slots
            std::vector<uint8_t> dst(ids.size() * 256);
            FetchStats st;
            CHECK(sm.fetch(0, ids.data(), ids.size(), dst.data(), &st) == 0, "small fetch err");
            for (size_t i = 0; i < ids.size(); ++i) if (!same(0, ids[i], &dst[i * 256])) { CHECK(false, "small cache row %u wrong", ids[i]); break; }
            req += st.req; hit += st.hits;
        }
        printf("  small cache (4096 rows, hot set 6000): hit rate %.1f%% over %llu requests\n", 100.0 * hit / req, (unsigned long long)req);
        CHECK(hit > 0 && hit < req, "eviction test hit rate %llu/%llu", (unsigned long long)hit, (unsigned long long)req);
    }
    // ---- T6: LRU order inside one set (1 set, 4 ways)
    {
        RowCache c;
        CHECK(c.init(4 * 256, 4, &err), "init");
        CHECK(c.sets() == 1, "one set");
        uint8_t r[256] = {0};
        for (uint32_t k = 1; k <= 4; ++k) { r[0] = (uint8_t)k; c.insert(row_tag(0, k), r); }
        CHECK(c.find(row_tag(0, 1)) != nullptr, "A hit");                // recency now: A, D, C, B
        r[0] = 5;
        c.insert(row_tag(0, 5), r);                                      // evicts B (id 2)
        CHECK(c.find(row_tag(0, 2)) == nullptr, "B must be evicted");
        CHECK(c.find(row_tag(0, 1)) && c.find(row_tag(0, 3)) && c.find(row_tag(0, 4)) && c.find(row_tag(0, 5)), "A, C, D, E stay");
        CHECK(c.find(row_tag(0, 5))[0] == 5, "payload");
        CHECK(c.find(row_tag(1, 1)) == nullptr, "table 1 tag differs from table 0");
        RowCache w16;
        CHECK(w16.init(16 * 256, 16, &err) && w16.sets() == 1, "16 ways");
        for (uint32_t k = 1; k <= 17; ++k) { r[0] = (uint8_t)k; w16.insert(row_tag(0, k), r); }
        CHECK(w16.find(row_tag(0, 1)) == nullptr && w16.find(row_tag(0, 17)) != nullptr, "16-way: oldest evicted by the 17th");
        for (uint32_t k = 2; k <= 16; ++k) CHECK(w16.find(row_tag(0, k)) != nullptr, "16-way row %u kept", k);
        CHECK(!w16.init(0, 16, &err), "a cache with no set must fail");
    }
    // ---- T7: n = 0, all-foreign, and the file's short final block
    {
        FetchStats st;
        CHECK(rs.fetch(0, nullptr, 0, nullptr, &st) == 0 && st.req == 0 && st.blocks == 0, "n=0");
        uint32_t f[3] = {0, 5, (uint32_t)(s0.first + s0.valid)};   // below first, below first, one past the last valid row (a pad row of the file)
        std::vector<uint8_t> d(3 * 256, 0xEE);
        CHECK(rs.fetch(0, f, 3, d.data(), &st) == 0 && st.foreign == 3 && st.blocks == 0, "all foreign: foreign %u reads %u", st.foreign, st.blocks);
        CHECK(d[0] == 0xEE && d[3 * 256 - 1] == 0xEE, "foreign rows must stay untouched");
        CHECK((s1.part * 256) % 4096 != 0, "table 1 must end inside a block");
        uint32_t tail[2] = {100004, 100000};
        std::vector<uint8_t> t(2 * 256);
        CHECK(rs.fetch(1, tail, 2, t.data(), &st) == 0, "tail fetch err");
        CHECK(same(1, tail[0], &t[0]) && same(1, tail[1], &t[256]), "tail block rows wrong");
    }
    // ---- T8: thread sweep, cold cache each time: host NVMe 4 KiB random reads (proxy rate)
    printf("sweep (cold cache, O_DIRECT=%d, block %zu): unique random rows of t0\n", rs.direct(0), rs.block());
    for (int th : {1, 8, 32, 96})
    {
        StoreOpts so = o;
        so.io_threads = th;
        so.cache_bytes = 64ull << 20;
        RowStore sw;
        if (!sw.open(files, so, &err)) { fprintf(stderr, "open sweep: %s\n", err.c_str()); return 2; }
        std::mt19937_64 rng(99);
        std::set<uint32_t> u;
        const size_t want = th == 1 ? 2000 : 20000;
        while (u.size() < want) u.insert((uint32_t)(s0.first + rng() % s0.valid));
        std::vector<uint32_t> ids(u.begin(), u.end());
        std::shuffle(ids.begin(), ids.end(), rng);
        std::vector<uint8_t> dst(ids.size() * 256);
        FetchStats st;
        CHECK(sw.fetch(0, ids.data(), ids.size(), dst.data(), &st) == 0, "sweep fetch err");
        for (size_t i = 0; i < ids.size(); i += 97) CHECK(same(0, ids[i], &dst[i * 256]), "sweep row wrong");
        printf("  threads %3d: %u reads in %.0f us = %.0f reads/s (%.1f us per read at the pool)\n", th, st.blocks, st.io_us, st.blocks / (st.io_us * 1e-6), st.io_us / st.blocks);
    }
    // ---- one table in two files (the release's split files): t1's rows 0..49999 at byte 65536 of one file, the rest
    // from byte 0 of another; every row reads the same bytes as through t1.w8
    {
        std::vector<uint8_t> all((size_t)s1.valid * 256);
        FILE* f = fopen(s1.path.c_str(), "rb");
        CHECK(f && fread(all.data(), 1, all.size(), f) == all.size(), "read t1.w8 back");
        if (f) fclose(f);
        const int64_t cut = 50000;
        const std::string pa = dir + "/split.a", pb = dir + "/split.b";
        FILE* a = fopen(pa.c_str(), "wb");
        std::vector<uint8_t> pad(65536, 0);
        fwrite(pad.data(), 1, pad.size(), a);
        fwrite(all.data(), 256, cut, a);
        fclose(a);
        FILE* b = fopen(pb.c_str(), "wb");
        fwrite(all.data() + (size_t)cut * 256, 256, s1.valid - cut, b);
        fclose(b);
        std::vector<TableFile> runs = {{"", 0, s1.valid, {{pa, 0, cut, 65536}, {pb, cut, s1.valid - cut, 0}}}};
        RowStore rr;
        CHECK(rr.open(runs, o, &err), "open runs: %s", err.c_str());
        std::vector<uint32_t> ids;
        for (int64_t i = 0; i < 4096; ++i) ids.push_back((uint32_t)((i * 7919) % s1.valid));
        ids.push_back((uint32_t)(cut - 1)); ids.push_back((uint32_t)cut);
        std::vector<uint8_t> got(ids.size() * 256);
        CHECK(rr.fetch(0, ids.data(), ids.size(), got.data()) == 0, "fetch runs");
        bool same = true;
        for (size_t i = 0; i < ids.size(); ++i) same &= !memcmp(&got[i * 256], &all[(size_t)ids[i] * 256], 256);
        CHECK(same, "rows read through two runs differ from the single file");
        std::vector<TableFile> gap = {{"", 0, s1.valid, {{pa, 0, cut, 65536}, {pb, cut + 1, s1.valid - cut - 1, 0}}}};
        RowStore rg;
        CHECK(!rg.open(gap, o, &err) && err.find("consecutive") != std::string::npos, "a gap between runs must fail open: %s", err.c_str());
    }

    // ---- T9: keepalive (keepalive_ms = 10): reads only while fetches are active, none when idle, fetch results unchanged
    {
        StoreOpts off = o;
        off.cache_bytes = 64ull << 20;
        off.keepalive_ms = 0;
        StoreOpts on = off;
        on.keepalive_ms = 10;
        RowStore a;
        std::unique_ptr<RowStore> bp(new RowStore());
        RowStore& b = *bp;
        CHECK(a.open(files, off, &err), "open keepalive off: %s", err.c_str());
        CHECK(b.open(files, on, &err), "open keepalive on: %s", err.c_str());
        CHECK(!a.keepalive_running(), "keepalive_ms = 0 must not start a thread");
        CHECK(b.keepalive_running() == b.direct(0), "keepalive runs iff table 0 is O_DIRECT (direct %d running %d)", b.direct(0), b.keepalive_running());
        std::mt19937_64 rng(7);
        std::vector<uint32_t> ids(3000);
        for (auto& x : ids) x = (uint32_t)(s0.first + rng() % s0.valid);
        std::vector<uint8_t> da(ids.size() * 256), db(ids.size() * 256);
        auto ms = [](int n) { std::this_thread::sleep_for(std::chrono::milliseconds(n)); };
        ms(60);
        CHECK(b.keepalive_stats().reads == 0, "no reads before the first fetch: %llu", (unsigned long long)b.keepalive_stats().reads);
        FetchStats sa, sb;
        CHECK(a.fetch(0, ids.data(), ids.size(), da.data(), &sa) == 0 && b.fetch(0, ids.data(), ids.size(), db.data(), &sb) == 0, "keepalive fetch err");
        CHECK(!memcmp(da.data(), db.data(), da.size()), "keepalive changed fetched rows");
        CHECK(sa.blocks == sb.blocks && sa.bytes == sb.bytes && sa.miss_uniq == sb.miss_uniq && sa.hits == sb.hits,
              "keepalive changed fetch stats: blocks %u/%u bytes %llu/%llu", sa.blocks, sb.blocks, (unsigned long long)sa.bytes, (unsigned long long)sb.bytes);
        if (b.keepalive_running())
        {
            CHECK(b.keepalive_blocks() > 0, "no eligible keepalive blocks");
            for (int i = 0; i < 80; ++i)   // ~400 ms of active all-hit fetches (no drive reads): the keepalive must read
            {
                CHECK(b.fetch(0, ids.data(), 64, db.data(), &sb) == 0 && sb.blocks == 0, "active fetch err or read");
                ms(5);
            }
            const RowStore::KeepaliveStats k1 = b.keepalive_stats();
            // the two counters are read while a tick may be in flight: reads can lead ticks by up to one tick's 4 reads
            CHECK(k1.reads >= 8 && k1.reads >= 4 * k1.ticks && k1.reads <= 4 * k1.ticks + 4 && k1.errors == 0, "active: ticks %llu reads %llu err %llu",
                  (unsigned long long)k1.ticks, (unsigned long long)k1.reads, (unsigned long long)k1.errors);
            for (size_t i = 0; i < 64; ++i) CHECK(same(0, ids[i], &db[i * 256]), "row %zu wrong with keepalive", i);
            ms(300);   // > the 250 ms active window: the thread parks
            const uint64_t r1 = b.keepalive_stats().reads;
            ms(150);
            const RowStore::KeepaliveStats k_parked = b.keepalive_stats();
            CHECK(k_parked.reads == r1, "idle keepalive read: %llu -> %llu", (unsigned long long)r1, (unsigned long long)k_parked.reads);
            CHECK(k_parked.reads == 4 * k_parked.ticks, "parked: ticks %llu reads %llu", (unsigned long long)k_parked.ticks, (unsigned long long)k_parked.reads);
            printf("keepalive: ticks %llu reads %llu max_us %llu over %llu blocks\n", (unsigned long long)k1.ticks, (unsigned long long)k1.reads,
                   (unsigned long long)k1.max_us, (unsigned long long)b.keepalive_blocks());
            CHECK(b.open(files, on, &err) && b.keepalive_running(), "second open must restart the keepalive: %s", err.c_str());
            CHECK(b.fetch(0, ids.data(), 64, db.data(), &sb) == 0, "fetch after reopen");
            ms(40);
            CHECK(b.keepalive_stats().errors == 0, "keepalive error after reopen");
        }
        else printf("keepalive: table 0 is not O_DIRECT here; thread checks skipped\n");
        const auto t0 = std::chrono::steady_clock::now();
        bp.reset();   // shutdown while active: stop + join must be quick
        const double ms_down = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ms_down < 100.0, "keepalive shutdown took %.1f ms", ms_down);
    }

    // ---- T10: a fetch longer than the 250 ms active window must not leave the keepalive parked for the short-gap fetches after it
    {
        StoreOpts on = o;
        on.cache_bytes = 64ull << 20;
        on.keepalive_ms = 10;
        RowStore c;
        CHECK(c.open(files, on, &err), "T10 open: %s", err.c_str());
        if (c.keepalive_running())
        {
            auto ms = [](int n) { std::this_thread::sleep_for(std::chrono::milliseconds(n)); };
            std::vector<uint32_t> warm(64), cold(8);
            for (int i = 0; i < 64; ++i) warm[i] = (uint32_t)(s0.first + (int64_t)i * 37);   // rows 0..2331
            const int64_t c0 = (s0.valid / 2) & ~(int64_t)15;   // 8 rows inside one 4 KiB block of table 0, not in the warm set
            for (int i = 0; i < 8; ++i) cold[i] = (uint32_t)(s0.first + c0 + i);
            g_slow_lo.store(c0 * 256);
            g_slow_hi.store(c0 * 256 + 4096);
            std::vector<uint8_t> d(64 * 256);
            FetchStats st;
            CHECK(c.fetch(0, warm.data(), warm.size(), d.data(), &st) == 0, "T10 warm fetch");
            ms(20);
            g_slow_until_ns.store(steady_ns() + 320000000ll, std::memory_order_release);
            const auto t0 = std::chrono::steady_clock::now();
            CHECK(c.fetch(0, cold.data(), cold.size(), d.data(), &st) == 0 && st.blocks > 0, "T10 long fetch err or no drive read");
            const double long_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            g_slow_until_ns.store(0, std::memory_order_release);
            CHECK(long_ms > 260.0, "T10 long fetch took only %.1f ms (pread seam not active)", long_ms);
            const RowStore::KeepaliveStats k0 = c.keepalive_stats();
            for (int i = 0; i < 100; ++i)   // ~0.6 s of all-hit fetches 5 ms apart
            {
                CHECK(c.fetch(0, warm.data(), warm.size(), d.data(), &st) == 0 && st.blocks == 0, "T10 short fetch err or read");
                ms(5);
            }
            const RowStore::KeepaliveStats k1 = c.keepalive_stats();
            CHECK(k1.reads - k0.reads >= 8 && k1.errors == 0, "T10 keepalive parked after a long fetch: reads %llu -> %llu err %llu",
                  (unsigned long long)k0.reads, (unsigned long long)k1.reads, (unsigned long long)k1.errors);
            printf("keepalive T10: long fetch %.0f ms, then %llu keepalive reads in 100 short-gap fetches\n", long_ms, (unsigned long long)(k1.reads - k0.reads));
        }
        else printf("keepalive T10: table 0 is not O_DIRECT here; skipped\n");
    }

    if (g_fail) { fprintf(stderr, "FAILED %d checks\n", g_fail); return 1; }
    printf("PASS rowstore\n");
    return 0;
}
