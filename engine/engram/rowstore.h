// Engram row store (host only, no CUDA): the 256-byte E4M3 rows of this rank's slice of a layer's table, read from NVMe with
// O_DIRECT behind a set-associative row cache. Contract: docs/DEEPSEEK-ENGRAM.md section 3 step 2 and section 4.
//
//   file      PREFIX.engram.L###.w8, u8 [part][256], rank r serves global rows [first, first+valid), file row = id - first.
//   request   O_DIRECT needs an aligned offset, length and buffer. The store asks statx for the file's dio_offset_align / dio_mem_align,
//             reads the aligned block (default 4096 B = 16 rows) that holds the row and copies the 256 B out of it. A row never
//             straddles a block. Rows of one fetch that share a block share one read; the same row asked twice is read once.
//   cache     16-way (knob) set-associative, true LRU per set, keeps 256-B rows (not blocks). Key = (table, id).
//   threads   a persistent pool of blocking pread() workers (default 32; 64 is fastest for prefill); decode needs about 20 reads a step, a prefill
//             chunk thousands, and the pool is sized for the latter.
//   fetch     one call per (table, step or chunk): ids[n] -> dst[n][256]. Rows this rank does not own are skipped (dst untouched, the
//             caller zeroes or ignores them). dst may be null: the call then only warms the cache (prefetch).
// A fetch is serialized by a mutex; the reads of one fetch run in parallel on the pool.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "hash.h"

namespace ds41 { namespace engram {

struct TableFile
{
    std::string path;       // PREFIX.engram.L###.w8
    int64_t first = 0;      // first global row this rank serves
    int64_t valid = 0;      // rows it serves (the file may hold more: pad rows up to `part`)
};

struct StoreOpts
{
    size_t cache_bytes = 4ull << 30;   // GR8_DS_ENGRAM_CACHE_MIB
    int ways = 16;                     // GR8_DS_ENGRAM_WAYS, 1..16
    int io_threads = 32;               // GR8_DS_ENGRAM_IO_THREADS (64 is fastest for prefill)
    size_t block = 4096;               // smallest read; raised to the file's dio_offset_align if that is larger
    bool direct = true;                // false: buffered pread (tmpfs, or a filesystem that refuses O_DIRECT)
};

struct FetchStats
{
    uint32_t req = 0, foreign = 0, hits = 0;   // requests, rows owned by another rank (skipped), rows served from the cache
    uint32_t miss_req = 0, miss_uniq = 0;      // requests that missed, distinct rows among them
    uint32_t blocks = 0;                       // reads issued
    uint64_t bytes = 0;                        // bytes read from the file
    double lookup_us = 0, io_us = 0, total_us = 0;
    int err = 0;                               // errno of the first failed read (0 = none)
    void add(const FetchStats& o)
    {
        req += o.req; foreign += o.foreign; hits += o.hits; miss_req += o.miss_req; miss_uniq += o.miss_uniq; blocks += o.blocks;
        bytes += o.bytes; lookup_us += o.lookup_us; io_us += o.io_us; total_us += o.total_us;
        if (!err) err = o.err;
    }
};

inline uint32_t row_tag(int table, uint32_t id) { return ((uint32_t)table << 30) | (id + 1); }   // 0 = empty slot; ids < 2^30 - 1

// Set-associative cache of 256-B rows, true LRU per set (order word: nibble p = the way at recency p, 0 = most recent).
class RowCache
{
public:
    ~RowCache() { if (slots_) munmap(slots_, (size_t)nsets_ * ways_ * kRowBytes); }
    bool init(size_t bytes, int ways, std::string* err)
    {
        if (ways < 1 || ways > 16) { *err = "cache ways must be 1..16"; return false; }
        const size_t nslots = bytes / kRowBytes;
        nsets_ = nslots / (size_t)ways;
        if (!nsets_) { *err = "cache smaller than one set"; return false; }
        ways_ = ways;
        void* m = mmap(nullptr, nsets_ * ways_ * kRowBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) { *err = std::string("cache mmap: ") + strerror(errno); nsets_ = 0; return false; }
        slots_ = (uint8_t*)m;
        tags_.assign(nsets_ * ways_, 0);
        uint64_t o = 0;
        for (int p = 0; p < 16; ++p) o |= (uint64_t)p << (4 * p);
        order_.assign(nsets_, o);
        return true;
    }
    size_t sets() const { return nsets_; }
    int ways() const { return ways_; }
    size_t rows() const { return nsets_ * (size_t)ways_; }
    // the cached row (and mark it most recent), or null
    const uint8_t* find(uint32_t tag)
    {
        const size_t s = set_of(tag);
        const uint32_t* t = &tags_[s * ways_];
        for (int w = 0; w < ways_; ++w)
            if (t[w] == tag) { touch(s, w); return slots_ + (s * ways_ + w) * kRowBytes; }
        return nullptr;
    }
    // put a row that is not in the cache: an empty way, else the least recent one
    void insert(uint32_t tag, const uint8_t* row)
    {
        const size_t s = set_of(tag);
        uint32_t* t = &tags_[s * ways_];
        int v = -1;
        for (int w = 0; w < ways_; ++w) if (t[w] == 0) { v = w; break; }
        if (v < 0) v = (int)((order_[s] >> (4 * (ways_ - 1))) & 15);
        t[v] = tag;
        memcpy(slots_ + (s * ways_ + v) * kRowBytes, row, kRowBytes);
        touch(s, v);
    }
private:
    size_t set_of(uint32_t tag) const { return (size_t)((((uint64_t)tag * 0x9E3779B97F4A7C15ull) >> 32) * nsets_ >> 32); }
    void touch(size_t s, int w)
    {
        const uint64_t o = order_[s];
        int p = 0;
        while (((o >> (4 * p)) & 15) != (uint64_t)w) ++p;
        const uint64_t low = p ? (o & ((1ull << (4 * p)) - 1)) : 0;
        const uint64_t high = p >= 15 ? 0 : ((o >> (4 * (p + 1))) << (4 * (p + 1)));
        order_[s] = high | (low << 4) | (uint64_t)w;
    }
    uint8_t* slots_ = nullptr;
    size_t nsets_ = 0;
    int ways_ = 0;
    std::vector<uint32_t> tags_;
    std::vector<uint64_t> order_;
};

// Persistent pool of blocking readers. Each worker owns one aligned buffer of `bufsz` bytes.
class IoPool
{
public:
    ~IoPool()
    {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    bool start(int n, size_t bufsz, size_t align)
    {
        n_ = n;
        for (int i = 0; i < n; ++i) th_.emplace_back([this, bufsz, align] { loop(bufsz, align); });
        return true;
    }
    // run fn(i, buf) for i in [0, n) on the pool and wait
    void run(int n, const std::function<void(int, uint8_t*)>& fn)
    {
        if (n <= 0) return;
        {
            std::lock_guard<std::mutex> g(mu_);
            fn_ = &fn;
            remaining_ = n;
            for (int i = 0; i < n; ++i) q_.push_back(i);
        }
        const int wake = std::min(n, n_);
        for (int i = 0; i < wake; ++i) cv_.notify_one();
        std::unique_lock<std::mutex> g(mu_);
        done_.wait(g, [this] { return remaining_ == 0; });
    }
private:
    void loop(size_t bufsz, size_t align)
    {
        void* raw = nullptr;
        if (posix_memalign(&raw, align, bufsz) != 0) raw = nullptr;
        uint8_t* buf = (uint8_t*)raw;
        for (;;)
        {
            int i;
            const std::function<void(int, uint8_t*)>* fn;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_.wait(g, [this] { return stop_ || !q_.empty(); });
                if (q_.empty()) break;   // stop_ and drained
                i = q_.front();
                q_.pop_front();
                fn = fn_;
            }
            (*fn)(i, buf);
            {
                std::lock_guard<std::mutex> g(mu_);
                if (--remaining_ == 0) done_.notify_all();
            }
        }
        free(buf);
    }
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_, done_;
    std::deque<int> q_;
    const std::function<void(int, uint8_t*)>* fn_ = nullptr;
    int remaining_ = 0, n_ = 0;
    bool stop_ = false;
};

class RowStore
{
public:
    ~RowStore()
    {
        pool_.reset();   // join the workers before the descriptors go
        for (auto& t : tab_) if (t.fd >= 0) close(t.fd);
    }

    bool open(const std::vector<TableFile>& files, const StoreOpts& o, std::string* err)
    {
        opts_ = o;
        size_t align = 4096;
        block_ = o.block;
        for (const TableFile& f : files)
        {
            Tab t;
            t.first = f.first;
            t.valid = f.valid;
            t.path = f.path;
            if (f.first < 0 || f.valid < 0 || f.first + f.valid >= (1ll << 30) - 1) { *err = f.path + ": row range outside the 30-bit tag space"; return false; }
            t.fd = ::open(f.path.c_str(), O_RDONLY | O_CLOEXEC | (o.direct ? O_DIRECT : 0));
            if (t.fd < 0 && o.direct && errno == EINVAL)
            {
                t.fd = ::open(f.path.c_str(), O_RDONLY | O_CLOEXEC);   // the filesystem refuses O_DIRECT: buffered, said in the stats
                t.direct = false;
            }
            else t.direct = o.direct;
            if (t.fd < 0) { *err = f.path + ": open: " + strerror(errno); return false; }
            struct stat st;
            if (fstat(t.fd, &st) != 0) { *err = f.path + ": fstat: " + strerror(errno); return false; }
            t.bytes = (uint64_t)st.st_size;
            if (t.bytes < (uint64_t)f.valid * kRowBytes) { *err = f.path + ": file holds fewer bytes than valid rows * 256"; return false; }
            if (t.direct)
            {
#ifdef STATX_DIOALIGN
                struct statx sx;
                if (statx(t.fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 && (sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align)
                {
                    block_ = std::max<size_t>(block_, sx.stx_dio_offset_align);
                    align = std::max<size_t>(align, std::max<size_t>(sx.stx_dio_mem_align, sx.stx_dio_offset_align));
                }
#endif
            }
            tab_.push_back(t);
        }
        size_t b = 256;
        while (b < block_) b <<= 1;   // power of two, multiple of 256: a row never straddles a block
        block_ = b;
        align = std::max(align, block_);
        if (!cache_.init(o.cache_bytes, o.ways, err)) return false;
        pool_.reset(new IoPool());
        pool_->start(std::max(1, o.io_threads), block_, align);
        return true;
    }

    size_t block() const { return block_; }
    bool direct(int table) const { return tab_[table].direct; }
    const RowCache& cache() const { return cache_; }
    const FetchStats& totals() const { return total_; }
    uint64_t fetches() const { return nfetch_; }

    // ids[n] -> dst[n][256] (null: warm the cache only). Returns 0, or the errno of a failed read (stats->err).
    int fetch(int table, const uint32_t* ids, size_t n, uint8_t* dst, FetchStats* stats = nullptr)
    {
        std::lock_guard<std::mutex> lk(mu_);
        FetchStats st;
        const auto t0 = Clock::now();
        const Tab& T = tab_[table];
        st.req = (uint32_t)n;

        struct Miss { uint32_t id, req; };
        std::vector<Miss> miss;
        for (size_t i = 0; i < n; ++i)
        {
            const int64_t id = ids[i];
            if (id < T.first || id >= T.first + T.valid) { ++st.foreign; continue; }
            const uint8_t* row = cache_.find(row_tag(table, (uint32_t)id));
            if (row)
            {
                ++st.hits;
                if (dst) memcpy(dst + i * kRowBytes, row, kRowBytes);
            }
            else miss.push_back({(uint32_t)id, (uint32_t)i});
        }
        st.miss_req = (uint32_t)miss.size();
        const auto t1 = Clock::now();

        if (!miss.empty())
        {
            std::sort(miss.begin(), miss.end(), [](const Miss& a, const Miss& b) { return a.id != b.id ? a.id < b.id : a.req < b.req; });
            // distinct rows (uniq) and, per row, its range in `miss`
            std::vector<uint32_t> uid, ubeg;
            for (size_t k = 0; k < miss.size(); ++k)
                if (k == 0 || miss[k].id != miss[k - 1].id) { uid.push_back(miss[k].id); ubeg.push_back((uint32_t)k); }
            ubeg.push_back((uint32_t)miss.size());
            const size_t nu = uid.size();
            st.miss_uniq = (uint32_t)nu;
            std::vector<uint8_t> urow(nu * kRowBytes);
            // reads: consecutive distinct rows in the same block share one
            struct Task { uint64_t off; uint32_t u0, u1; };
            std::vector<Task> tasks;
            for (size_t u = 0; u < nu; ++u)
            {
                const uint64_t boff = (uint64_t)(uid[u] - T.first) * kRowBytes;
                const uint64_t blk = boff & ~(uint64_t)(block_ - 1);
                if (tasks.empty() || tasks.back().off != blk) tasks.push_back({blk, (uint32_t)u, (uint32_t)u + 1});
                else tasks.back().u1 = (uint32_t)u + 1;
            }
            st.blocks = (uint32_t)tasks.size();
            std::atomic<int> err{0};
            std::atomic<uint64_t> bytes{0};
            const size_t blk_sz = block_;
            pool_->run((int)tasks.size(), [&](int ti, uint8_t* buf)
            {
                const Task& k = tasks[ti];
                ssize_t r;
                do r = pread(T.fd, buf, blk_sz, (off_t)k.off); while (r < 0 && errno == EINTR);
                if (r < 0) { int z = 0; err.compare_exchange_strong(z, errno); return; }
                bytes += (uint64_t)r;
                for (uint32_t u = k.u0; u < k.u1; ++u)
                {
                    const uint64_t in = (uint64_t)(uid[u] - T.first) * kRowBytes - k.off;
                    if ((uint64_t)r < in + kRowBytes) { int z = 0; err.compare_exchange_strong(z, EIO); return; }   // short read inside the row
                    memcpy(&urow[(size_t)u * kRowBytes], buf + in, kRowBytes);
                }
            });
            st.bytes = bytes.load();
            st.err = err.load();
            if (!st.err)
                for (size_t u = 0; u < nu; ++u)
                {
                    const uint8_t* row = &urow[u * kRowBytes];
                    if (dst) for (uint32_t k = ubeg[u]; k < ubeg[u + 1]; ++k) memcpy(dst + (size_t)miss[k].req * kRowBytes, row, kRowBytes);
                    cache_.insert(row_tag(table, uid[u]), row);
                }
        }
        const auto t2 = Clock::now();
        st.lookup_us = us(t0, t1);
        st.io_us = us(t1, t2);
        st.total_us = us(t0, t2);
        total_.add(st);
        ++nfetch_;
        if (stats) *stats = st;
        return st.err;
    }

    static std::string line(const FetchStats& s)
    {
        char b[256];
        snprintf(b, sizeof b, "engram fetch: req %u foreign %u hit %u miss %u (uniq %u) reads %u %.1f KiB; lookup %.0f us, io %.0f us, total %.0f us%s",
                 s.req, s.foreign, s.hits, s.miss_req, s.miss_uniq, s.blocks, s.bytes / 1024.0, s.lookup_us, s.io_us, s.total_us, s.err ? " ERROR" : "");
        return b;
    }

private:
    using Clock = std::chrono::steady_clock;
    static double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
    struct Tab
    {
        std::string path;
        int fd = -1;
        int64_t first = 0, valid = 0;
        uint64_t bytes = 0;
        bool direct = true;
    };
    StoreOpts opts_;
    size_t block_ = 4096;
    std::vector<Tab> tab_;
    RowCache cache_;
    std::unique_ptr<IoPool> pool_;
    std::mutex mu_;
    FetchStats total_;
    uint64_t nfetch_ = 0;
};

}}  // namespace ds41::engram
