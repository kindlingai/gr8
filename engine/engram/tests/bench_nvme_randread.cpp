// NVMe random-read rate for the Engram row store (host only: no CUDA, no GPU, no serve).
//   bench_nvme_randread <scratch dir> [--gib N] [--secs S] [--quick]
// It writes (or reuses) one real-data file <dir>/nvme_bench.w8 of N GiB (default 24, about one TP4 rank's layer table of 24.6 GB) with
// O_DIRECT, so no hole is left and nothing is served from the page cache. Every read is checked against the data it must hold.
//   A. RAW   O_DIRECT pread() of one block at a uniform random aligned offset, T blocking threads (the model the row store uses):
//            threads sweep at 4096 B, block-size sweep (512 / 4096 / 8192 / 16384) at 64 threads. IOPS, MiB/s, latency, CPU cores.
//   B. STORE the real RowStore (rowstore.h) with a 1 MiB cache (every row misses): fetch() latency for batches of B random rows, the
//            number the engine sees. B = 12..24 is a decode step (2 layers x 24 ids / 4 ranks per token), B in the thousands is a
//            prefill chunk. io_threads sweep at B = 24 and B = 4096, then a batch sweep at the default 96 threads.
// Output is plain text; the last line is "PASS bench_nvme_randread" when no read failed and no read returned wrong data.
// Nothing is deleted. The file stays; remove it by its literal path when done.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../rowstore.h"

using namespace ds41::engram;
using Clock = std::chrono::steady_clock;

static std::atomic<uint64_t> g_bad{0};      // reads that failed or returned data that is not what was written
static std::atomic<uint64_t> g_reads{0};

static inline uint64_t mix(uint64_t x)      // splitmix64 finalizer: word w of the file holds mix(kSeed + w)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static constexpr uint64_t kSeed = 0x6e766d6562656e63ull;
static inline uint64_t word_at(uint64_t byte_off) { return mix(kSeed + byte_off / 8); }

// log-linear latency histogram in ns: 32 sub-buckets per power of two (about 3 % wide)
struct Hist
{
    std::vector<uint64_t> c = std::vector<uint64_t>(2048, 0);
    uint64_t n = 0, sum = 0, mx = 0;
    static int idx(uint64_t ns)
    {
        if (ns < 32) return (int)ns;
        const int e = 63 - __builtin_clzll(ns);
        return ((e - 4) << 5) + (int)((ns >> (e - 5)) & 31);
    }
    static double mid(int i)
    {
        if (i < 32) return i;
        const int e = (i >> 5) + 4;
        return (double)((uint64_t)(32 + (i & 31)) << (e - 5)) + (double)(1ull << (e - 6));
    }
    void add(uint64_t ns) { ++c[idx(ns)]; ++n; sum += ns; if (ns > mx) mx = ns; }
    void merge(const Hist& o) { for (size_t i = 0; i < c.size(); ++i) c[i] += o.c[i]; n += o.n; sum += o.sum; mx = std::max(mx, o.mx); }
    double q_us(double q) const
    {
        if (!n) return 0;
        const uint64_t want = (uint64_t)(q * (double)(n - 1)) + 1;
        uint64_t acc = 0;
        for (size_t i = 0; i < c.size(); ++i) { acc += c[i]; if (acc >= want) return mid((int)i) / 1000.0; }
        return mx / 1000.0;
    }
};

static double cpu_s()
{
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double)r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + (double)r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
}

// ---- file
static bool prep_file(const std::string& path, uint64_t bytes)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && (uint64_t)st.st_size == bytes && (uint64_t)st.st_blocks * 512 >= bytes)
    {
        printf("file %s: %.1f GiB exists, reused (allocated %.1f GiB)\n", path.c_str(), bytes / 1073741824.0, st.st_blocks * 512 / 1073741824.0);
        return true;
    }
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { fprintf(stderr, "open %s for write: %s\n", path.c_str(), strerror(errno)); return false; }
    const size_t chunk = 8u << 20;
    const uint64_t nchunk = (bytes + chunk - 1) / chunk;
    const int nth = 8;
    std::atomic<uint64_t> next{0};
    std::atomic<int> werr{0};
    const auto t0 = Clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t)
        th.emplace_back([&] {
            void* raw = nullptr;
            if (posix_memalign(&raw, 4096, chunk) != 0) { werr = ENOMEM; return; }
            uint64_t* b = (uint64_t*)raw;
            for (;;)
            {
                const uint64_t ci = next++;
                if (ci >= nchunk || werr) break;
                const uint64_t off = ci * chunk;
                const size_t len = (size_t)std::min<uint64_t>(chunk, bytes - off);
                for (size_t w = 0; w < chunk / 8; ++w) b[w] = word_at(off + w * 8);
                size_t done = 0;
                while (done < len)
                {
                    const ssize_t r = pwrite(fd, (char*)b + done, ((len - done) + 4095) & ~(size_t)4095, (off_t)(off + done));
                    if (r < 0) { if (errno == EINTR) continue; werr = errno; break; }
                    done += (size_t)r;
                }
            }
            free(raw);
        });
    for (auto& t : th) t.join();
    if (!werr && ftruncate(fd, (off_t)bytes) != 0) werr = errno;
    if (!werr && fsync(fd) != 0) werr = errno;
    close(fd);
    if (werr) { fprintf(stderr, "write %s: %s\n", path.c_str(), strerror(werr)); return false; }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    stat(path.c_str(), &st);
    printf("file %s: wrote %.1f GiB in %.1f s = %.0f MiB/s (8 O_DIRECT writers, 8 MiB each); allocated %.1f GiB\n", path.c_str(),
           bytes / 1073741824.0, s, bytes / 1048576.0 / s, st.st_blocks * 512 / 1073741824.0);
    return true;
}

// ---- A. raw preads
struct RawRes { double iops, mibs, p50, p90, p99, p999, mx, cores, avg; uint64_t bad; };

static RawRes raw_point(const std::string& path, uint64_t fsize, size_t bs, int nth, double secs)
{
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    RawRes R{};
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path.c_str(), strerror(errno)); g_bad++; return R; }
    const uint64_t nblk = fsize / bs;
    std::atomic<bool> go{false}, stop{false};
    std::vector<Hist> hs(nth);
    std::vector<uint64_t> bad(nth, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t)
        th.emplace_back([&, t] {
            void* raw = nullptr;
            if (posix_memalign(&raw, 4096, std::max<size_t>(bs, 4096)) != 0) { bad[t]++; return; }
            uint64_t rng = 0x1234567ull * (uint64_t)(t + 1) + bs;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            while (!stop.load(std::memory_order_relaxed))
            {
                rng = mix(rng);
                const uint64_t off = (rng % nblk) * bs;
                const auto a = Clock::now();
                ssize_t r;
                do r = pread(fd, raw, bs, (off_t)off); while (r < 0 && errno == EINTR);
                const auto b = Clock::now();
                if (r != (ssize_t)bs || ((uint64_t*)raw)[0] != word_at(off) || ((uint64_t*)raw)[bs / 8 - 1] != word_at(off + bs - 8)) { bad[t]++; continue; }
                hs[t].add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
            }
            free(raw);
        });
    const double c0 = cpu_s();
    const auto t0 = Clock::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop.store(true);
    const auto t1 = Clock::now();
    for (auto& t : th) t.join();
    const double wall = std::chrono::duration<double>(t1 - t0).count(), c1 = cpu_s();
    close(fd);
    Hist all;
    for (int t = 0; t < nth; ++t) { all.merge(hs[t]); R.bad += bad[t]; }
    g_bad += R.bad;
    g_reads += all.n;
    R.iops = all.n / wall;
    R.mibs = all.n * (double)bs / 1048576.0 / wall;
    R.avg = all.n ? all.sum / 1000.0 / all.n : 0;
    R.p50 = all.q_us(0.5); R.p90 = all.q_us(0.9); R.p99 = all.q_us(0.99); R.p999 = all.q_us(0.999); R.mx = all.mx / 1000.0;
    R.cores = (c1 - c0) / wall;
    return R;
}

static void raw_line(const char* tag, size_t bs, int nth, const RawRes& r)
{
    printf("RAW   %-6s bs %5zu B  threads %4d | %9.0f IOPS %8.0f MiB/s | lat us avg %7.1f p50 %7.1f p90 %7.1f p99 %8.1f p99.9 %8.1f max %9.1f | cpu %.2f cores%s\n",
           tag, bs, nth, r.iops, r.mibs, r.avg, r.p50, r.p90, r.p99, r.p999, r.mx, r.cores, r.bad ? "  BAD READS" : "");
    fflush(stdout);
}

// ---- B. the row store
struct StoreRes { double fetches_s, rows_s, p50, p90, p99, mx, avg_reads, io_p50, cores; int reps; bool ok; };

static StoreRes store_point(const std::string& path, uint64_t fsize, int io_threads, size_t B, double secs, int max_reps)
{
    StoreRes R{};
    RowStore rs;
    StoreOpts o;
    o.cache_bytes = 1u << 20;    // 4096 rows against 100 M: every request misses
    o.io_threads = io_threads;
    std::string err;
    const uint64_t rows = fsize / kRowBytes;
    if (!rs.open({TableFile{path, 0, (int64_t)rows}}, o, &err)) { fprintf(stderr, "store open: %s\n", err.c_str()); g_bad++; return R; }
    if (!rs.direct(0)) { fprintf(stderr, "store: O_DIRECT refused by the filesystem; rate would be a page-cache rate\n"); g_bad++; return R; }
    std::vector<uint32_t> ids(B);
    std::vector<uint8_t> dst(B * kRowBytes);
    std::vector<double> tot, iot;
    double blocks = 0;
    uint64_t rng = 0xabcdef12345ull + B * 7 + (uint64_t)io_threads;
    // two untimed warm-up fetches (thread start, first page faults)
    for (int w = 0; w < 2; ++w)
    {
        for (size_t i = 0; i < B; ++i) { rng = mix(rng); ids[i] = (uint32_t)(rng % rows); }
        rs.fetch(0, ids.data(), B, dst.data());
    }
    const double c0 = cpu_s();
    const auto t0 = Clock::now();
    R.ok = true;
    while (R.reps < max_reps)
    {
        for (size_t i = 0; i < B; ++i) { rng = mix(rng); ids[i] = (uint32_t)(rng % rows); }
        FetchStats st;
        const int e = rs.fetch(0, ids.data(), B, dst.data(), &st);
        if (e) { R.ok = false; g_bad++; fprintf(stderr, "fetch error %d\n", e); break; }
        for (size_t i = 0; i < B; ++i)
        {
            uint64_t w0;
            memcpy(&w0, &dst[i * kRowBytes], 8);
            if (st.hits == 0 && w0 != word_at((uint64_t)ids[i] * kRowBytes)) { R.ok = false; g_bad++; }   // wrong data (a hit is a repeat id: also checked, same rule)
        }
        tot.push_back(st.total_us);
        iot.push_back(st.io_us);
        blocks += st.blocks;
        ++R.reps;
        if (std::chrono::duration<double>(Clock::now() - t0).count() >= secs && R.reps >= 20) break;
    }
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count(), c1 = cpu_s();
    g_reads += (uint64_t)blocks;
    std::sort(tot.begin(), tot.end());
    std::sort(iot.begin(), iot.end());
    auto q = [](const std::vector<double>& v, double p) { return v.empty() ? 0.0 : v[(size_t)(p * (double)(v.size() - 1))]; };
    R.fetches_s = R.reps / wall;
    R.rows_s = R.reps * (double)B / wall;
    R.p50 = q(tot, 0.5); R.p90 = q(tot, 0.9); R.p99 = q(tot, 0.99); R.mx = tot.empty() ? 0 : tot.back();
    R.io_p50 = q(iot, 0.5);
    R.avg_reads = R.reps ? blocks / R.reps : 0;
    R.cores = (c1 - c0) / wall;
    return R;
}

static void store_line(int th, size_t B, const StoreRes& r)
{
    printf("STORE io_threads %4d batch %5zu rows | %6d fetches | fetch us p50 %9.0f p90 %9.0f p99 %9.0f max %9.0f (io p50 %9.0f) | reads/fetch %7.1f | %9.0f rows/s %8.1f fetch/s | cpu %.2f cores%s\n",
           th, B, r.reps, r.p50, r.p90, r.p99, r.mx, r.io_p50, r.avg_reads, r.rows_s, r.fetches_s, r.cores, r.ok ? "" : "  FAILED");
    fflush(stdout);
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: bench_nvme_randread <scratch dir> [--gib N] [--secs S] [--quick]\n"); return 2; }
    std::string dir = argv[1];
    double gib = 24, secs = 3;
    bool quick = false;
    for (int i = 2; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--gib") && i + 1 < argc) gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quick")) quick = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (quick) { gib = std::min(gib, 1.0); secs = std::min(secs, 1.0); }
    const uint64_t fsize = ((uint64_t)(gib * 1073741824.0)) & ~(uint64_t)(16384 - 1);
    const std::string path = dir + "/nvme_bench.w8";
    printf("bench_nvme_randread: file %.2f GiB, %.1f s per point, %s\n", fsize / 1073741824.0, secs, quick ? "quick" : "full");
    if (!prep_file(path, fsize)) return 1;
    {   // let the drive settle after the write (garbage collection, SLC flush)
        const int settle = quick ? 1 : 5;
        printf("settle %d s\n", settle);
        std::this_thread::sleep_for(std::chrono::seconds(settle));
    }

    printf("\n== A. RAW O_DIRECT pread, uniform random aligned offsets over the whole file ==\n");
    const std::vector<int> tl = quick ? std::vector<int>{1, 16, 64, 96} : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 512};
    double best_iops = 0; int best_t = 0; RawRes r96{}, r1{};
    for (int t : tl)
    {
        const RawRes r = raw_point(path, fsize, 4096, t, secs);
        raw_line("thr", 4096, t, r);
        if (r.iops > best_iops) { best_iops = r.iops; best_t = t; }
        if (t == 96) r96 = r;
        if (t == 1) r1 = r;
    }
    for (size_t bs : std::vector<size_t>{512, 4096, 8192, 16384})
        raw_line("bs", bs, 64, raw_point(path, fsize, bs, 64, secs));

    printf("\n== B. RowStore.fetch(): 1 MiB cache (all misses), 4096-B blocks, uniform random row ids, 256-B rows ==\n");
    const int cap = quick ? 200 : 20000;
    for (int th : quick ? std::vector<int>{96} : std::vector<int>{8, 16, 32, 64, 96, 128, 192, 256})
        for (size_t B : std::vector<size_t>{24, 4096})
            store_line(th, B, store_point(path, fsize, th, B, secs, cap));
    StoreRes s24{}, s4096{};
    for (size_t B : std::vector<size_t>{1, 6, 12, 24, 48, 96, 256, 1024, 4096, 16384})
    {
        const StoreRes r = store_point(path, fsize, 96, B, secs, cap);
        store_line(96, B, r);
        if (B == 24) s24 = r;
        if (B == 4096) s4096 = r;
    }

    printf("\n== SUMMARY ==\n");
    printf("raw 4096 B: best %.0f IOPS (%.0f MiB/s) at %d threads; 1 thread %.0f IOPS (p50 %.0f us); 96 threads %.0f IOPS (p50 %.0f us, p99 %.0f us)\n",
           best_iops, best_iops * 4096 / 1048576.0, best_t, r1.iops, r1.p50, r96.iops, r96.p50, r96.p99);
    printf("store, 96 threads: batch 24 rows -> fetch p50 %.0f us p99 %.0f us; batch 4096 rows -> fetch p50 %.0f us = %.0f rows/s\n",
           s24.p50, s24.p99, s4096.p50, s4096.rows_s);
    printf("reads checked %llu, bad %llu\n", (unsigned long long)g_reads.load(), (unsigned long long)g_bad.load());
    if (g_bad) { printf("FAIL bench_nvme_randread\n"); return 1; }
    printf("PASS bench_nvme_randread\n");
    return 0;
}
