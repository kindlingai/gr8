// gr8 device image implementation (see devimage.h). Depends on g6pack + the CUDA runtime only.
#include "devimage.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace g6
{

namespace
{
constexpr size_t kStagingMax = 256ull << 20;   // pinned host memory for the whole ring
constexpr int kBufs = 3;

std::string cuda_err(const char* what, cudaError_t e) { return std::string(what) + ": " + cudaGetErrorString(e); }

// Walks the stream in order: tensor bytes feed that tensor's sha256, every other byte must be zero.
struct Checker
{
    const std::vector<Entry>& es;
    size_t i = 0, nbad = 0;
    uint64_t pos = 0, nonzero = 0;
    Sha256 h;
    std::string bad;

    explicit Checker(const std::vector<Entry>& entries) : es(entries) {}
    void finish_tensor()
    {
        uint8_t d[32];
        h.final(d);
        if (memcmp(d, es[i].sha256, 32) && nbad++ < 8) bad += (bad.empty() ? "" : ", ") + es[i].name;
        h = Sha256();
        ++i;
    }
    void feed(const uint8_t* p, size_t n)
    {
        while (n)
        {
            if (i < es.size() && pos >= es[i].offset)
            {
                const uint64_t end = es[i].offset + es[i].nbytes;
                const size_t take = (size_t)std::min<uint64_t>(n, end - pos);
                h.update(p, take);
                p += take, n -= take, pos += take;
                if (pos == end) finish_tensor();
            }
            else
            {
                const size_t take = i < es.size() ? (size_t)std::min<uint64_t>(n, es[i].offset - pos) : n;
                for (size_t k = 0; k < take; ++k) nonzero += p[k] != 0;
                p += take, n -= take, pos += take;
            }
        }
    }
    bool complete()
    {
        while (i < es.size() && es[i].nbytes == 0 && pos >= es[i].offset) finish_tensor();
        return i == es.size();
    }
};
}  // namespace

DevImage::~DevImage() { reset(); }

void DevImage::adopt(const Pack& pack, uint8_t* dev, uint64_t bytes)
{
    reset();
    base_ = dev;
    bytes_ = bytes;
    pack_ = &pack;
}

void DevImage::reset()
{
    if (base_) cudaFree(base_);
    base_ = nullptr;
    bytes_ = 0;
    pack_ = nullptr;
}

bool DevImage::load(const Pack& pack, const std::string& path, const LoadOpts& opt, LoadStats& st, std::string& err)
{
    using clk = std::chrono::steady_clock;
    auto since = [](clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); };
    const auto t0 = clk::now();
    reset();
    st = LoadStats();
    const uint64_t total = pack.bytes();
    if (pack.rank() < 0 || total == 0) { err = "open_manifest first"; return false; }

    const bool is_stdin = path == "-";
    int fd = 0;
    bool regular = false;
    if (!is_stdin)
    {
        fd = opt.direct_io ? open(path.c_str(), O_RDONLY | O_DIRECT) : -1;
        if (fd >= 0) st.used_direct_io = true;
        else fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open " + path + ": " + strerror(errno); return false; }
        struct stat sb;
        regular = fstat(fd, &sb) == 0 && S_ISREG(sb.st_mode);
        if (regular && (uint64_t)sb.st_size != total)
        {
            close(fd);
            err = "pack size " + std::to_string(sb.st_size) + " != manifest pack_bytes " + std::to_string(total);
            return false;
        }
        if (!st.used_direct_io) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
    else
        fcntl(fd, F_SETPIPE_SZ, 1 << 20);   // fewer, larger pipe reads when streamed over the fabric
    auto close_fd = [&] { if (!is_stdin) close(fd); };

    const size_t chunk = (size_t)std::min<uint64_t>(std::clamp<size_t>(opt.chunk, 4096, kStagingMax / kBufs) / 4096 * 4096, total);
    uint8_t* stage = nullptr;
    void* dev = nullptr;
    cudaError_t ce = cudaHostAlloc((void**)&stage, chunk * kBufs, cudaHostAllocDefault);
    if (ce != cudaSuccess) { close_fd(); err = cuda_err("cudaHostAlloc (staging)", ce); return false; }
    ce = cudaMalloc(&dev, total);
    if (ce != cudaSuccess)
    {
        cudaFreeHost(stage);
        close_fd();
        err = cuda_err(("cudaMalloc of " + std::to_string(total) + " bytes").c_str(), ce);
        return false;
    }

    // staging ring: chunk k sits in buffer k % kBufs from the read until the worker has hashed it
    std::mutex mu;
    std::condition_variable cv;
    uint64_t queued = 0, hashed = 0;
    size_t len[kBufs] = {};
    bool stop = false, abort = false;
    const bool hashing = opt.hash_threads > 0;
    Checker chk(pack.entries());
    std::thread worker;
    if (hashing)
        worker = std::thread([&] {
            for (;;)
            {
                uint64_t k;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return hashed < queued || stop; });
                    if (abort || hashed == queued) return;
                    k = hashed;
                }
                chk.feed(stage + (k % kBufs) * chunk, len[k % kBufs]);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    ++hashed;
                }
                cv.notify_all();
            }
        });

    uint64_t pos = 0, since_drop = 0;
    bool ok = true;
    for (uint64_t k = 0; pos < total && ok; ++k)
    {
        if (hashing)
        {   // buffer k % kBufs is free once chunk k - kBufs is hashed
            const auto tw = clk::now();
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return k - hashed < (uint64_t)kBufs; });
            st.stall_seconds += since(tw);
        }
        uint8_t* buf = stage + (k % kBufs) * chunk;
        const size_t want = (size_t)std::min<uint64_t>(chunk, total - pos);
        size_t got = 0;
        while (got < want)
        {
            const ssize_t n = read(fd, buf + got, want - got);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            {   // a pipe someone else set O_NONBLOCK on (e.g. an ssh sharing the description): wait for data
                pollfd p{fd, POLLIN, 0};
                poll(&p, 1, -1);
                continue;
            }
            if (n < 0 && st.used_direct_io && errno == EINVAL)
            {   // filesystem refused O_DIRECT (or a short read left us unaligned): continue buffered from here
                const int fd2 = open(path.c_str(), O_RDONLY);
                if (fd2 < 0 || lseek(fd2, (off_t)(pos + got), SEEK_SET) != (off_t)(pos + got)) { ok = false; err = "O_DIRECT fallback failed"; break; }
                close(fd);
                fd = fd2;
                st.used_direct_io = false;
                continue;
            }
            if (n <= 0)
            {
                ok = false;
                err = n == 0 ? "stream ended at byte " + std::to_string(pos + got) + " of " + std::to_string(total)
                             : std::string("read error: ") + strerror(errno);
                break;
            }
            got += (size_t)n;
        }
        if (!ok) break;
        const auto tc = clk::now();
        ce = cudaMemcpy((uint8_t*)dev + pos, buf, want, cudaMemcpyHostToDevice);
        st.copy_seconds += since(tc);
        if (ce != cudaSuccess) { ok = false; err = cuda_err("cudaMemcpy to device", ce); break; }
        pos += want;
        since_drop += want;
        if (regular && !st.used_direct_io && since_drop >= (1ull << 30))
        {
            posix_fadvise(fd, 0, (off_t)pos, POSIX_FADV_DONTNEED);
            since_drop = 0;
        }
        if (hashing)
        {
            {
                std::lock_guard<std::mutex> lk(mu);
                len[k % kBufs] = want;
                queued = k + 1;
            }
            cv.notify_all();
        }
    }
    st.read_seconds = since(t0);
    if (ok && !regular)
    {   // the stream must end exactly at pack_bytes
        char extra;
        ssize_t n;
        while ((n = read(fd, &extra, 1)) < 0 && errno == EINTR) {}
        if (n != 0) { ok = false; err = "stream has bytes past pack_bytes"; }
    }
    if (regular && !st.used_direct_io) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close_fd();
    if (hashing)
    {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            abort = !ok;
        }
        cv.notify_all();
        worker.join();
    }
    cudaFreeHost(stage);

    if (ok && hashing)
    {
        const bool complete = chk.complete();
        if (!complete || chk.nbad || chk.nonzero)
        {
            ok = false;
            err = "checksum mismatch in " + std::to_string(chk.nbad) + " tensor(s)" + (chk.bad.empty() ? "" : " [" + chk.bad + "]") +
                  (chk.nonzero ? "; " + std::to_string(chk.nonzero) + " non-zero padding byte(s)" : "") +
                  (complete ? "" : "; tensor walk did not reach the last entry");
        }
    }
    st.bytes = pos;
    st.seconds = since(t0);
    st.gb_per_s = pos / 1e9 / st.seconds;
    if (!ok)
    {
        cudaFree(dev);
        return false;
    }
    base_ = (uint8_t*)dev;
    bytes_ = total;
    pack_ = &pack;
    return true;
}

}  // namespace g6
