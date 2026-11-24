// Host-only Engram block reader. Our raw-ring implementation; no liburing.
// One serialized fetch lane owns a ring and QD aligned buffers, reused across fetches.
// Linux UAPI layout; the gr8-team erb.c experiment established the raw syscall route.
#pragma once
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <linux/io_uring.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace ds41 { namespace engram {

static constexpr int kUringPadAuto = -1;

// Parse GR8_URING_PAD ("auto" -> kUringPadAuto, "0"/"1" -> 0, "2".."64" -> N).
inline int parse_uring_pad(const std::string& raw)
{
    if (raw == "auto") return kUringPadAuto;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(raw.c_str(), &end, 0);
    if (errno || !end || end == raw.c_str() || *end != '\0' || v < 0 || v > 64) return 0;
    return v <= 1 ? 0 : (int)v;
}

struct PadProbeResult
{
    bool ran = false;
    unsigned pad_mod = 0;
    double n1_us = 0.0;
    double n8_us = 0.0;
    double probe_ms = 0.0;
};

class UringReader
{
public:
    UringReader() = default;
    UringReader(const UringReader&) = delete;
    UringReader& operator=(const UringReader&) = delete;
    ~UringReader() { release(); }

    void set_pad_mod(unsigned m) { pad_mod_ = m; }
    unsigned pad_mod() const { return pad_mod_; }

    // Returns errno; partially initialized readers are safe to destroy.
    int start(unsigned qd, size_t block, size_t align, unsigned pad_mod = 0)
    {
        if (!qd || block > (size_t)std::numeric_limits<int>::max()) return EINVAL;
        pad_mod_ = pad_mod;
        io_uring_params p{};
        do fd_ = (int)syscall(__NR_io_uring_setup, qd, &p); while (fd_ < 0 && errno == EINTR);
        if (fd_ < 0) return errno;
        sq_bytes_ = p.sq_off.array + (size_t)p.sq_entries * sizeof(unsigned);
        cq_bytes_ = p.cq_off.cqes + (size_t)p.cq_entries * sizeof(io_uring_cqe);
        single_ = p.features & IORING_FEAT_SINGLE_MMAP;
        if (single_) sq_bytes_ = std::max(sq_bytes_, cq_bytes_);
        sq_ = map(sq_bytes_, IORING_OFF_SQ_RING);
        if (sq_ == MAP_FAILED) return errno;
        cq_ = single_ ? sq_ : map(cq_bytes_, IORING_OFF_CQ_RING);
        if (cq_ == MAP_FAILED) return errno;
        sqe_bytes_ = (size_t)p.sq_entries * sizeof(io_uring_sqe);
        sqes_ = (io_uring_sqe*)map(sqe_bytes_, IORING_OFF_SQES);
        if (sqes_ == MAP_FAILED) return errno;
        sq_head_ = word(sq_, p.sq_off.head); sq_tail_ = word(sq_, p.sq_off.tail);
        cq_head_ = word(cq_, p.cq_off.head); cq_tail_ = word(cq_, p.cq_off.tail);
        sq_mask_ = *word(sq_, p.sq_off.ring_mask); cq_mask_ = *word(cq_, p.cq_off.ring_mask);
        array_ = word(sq_, p.sq_off.array);
        cqes_ = (io_uring_cqe*)(cq_ + p.cq_off.cqes);
        qd_ = std::min(qd, p.sq_entries);
        block_ = block;
        pad_len_ = align ? std::min(block, align) : block;
        stride_ = ((block + align - 1) / align) * align;
        if (stride_ > SIZE_MAX / qd_) return ENOMEM;
        void* buf = nullptr;
        const int e = posix_memalign(&buf, align, stride_ * qd_);
        if (e) return e;
        buffers_ = (uint8_t*)buf;
        tasks_.resize(qd_);
        return 0;
    }

    // Boot-time <3 ms O_DIRECT probe on the actual file descriptor: warms an 8-block aligned window,
    // compares 8x n=1 vs 8x n=8 reads, and enables pad_mod=8 only when median(n=1) exceeds median(n=8)
    // by >50 us and >1.15x.
    unsigned probe_pad_mod(int fd, uint64_t base_off, uint64_t span_bytes, PadProbeResult* out = nullptr)
    {
        PadProbeResult res{};
        pad_mod_ = 0;
        if (broken_ || fd < 0 || qd_ < 8 || !pad_len_ || span_bytes < pad_len_)
        {
            if (out) *out = res;
            return 0;
        }
        const uint64_t lo = (base_off + pad_len_ - 1) & ~(uint64_t(pad_len_) - 1);
        const uint64_t end = base_off + span_bytes;
        if (end < lo + pad_len_)
        {
            if (out) *out = res;
            return 0;
        }
        using Clock = std::chrono::steady_clock;
        const auto t_probe0 = Clock::now();
        const uint64_t step = 2ULL * pad_len_; // non-adjacent offsets prevent Linux block-layer read merging
        const uint64_t n_slots = (end - lo - pad_len_) / step + 1ULL;
        const uint64_t win_blocks = std::min<uint64_t>(n_slots, 8ULL);
        auto run_batch = [&](size_t cnt, double* elapsed_us) -> bool {
            int io_err = 0;
            const auto a = Clock::now();
            const int rc = run(
                cnt,
                [&](size_t idx, io_uring_sqe& e) {
                    e.fd = fd;
                    e.off = lo + (uint64_t(idx) % win_blocks) * step;
                    e.len = (unsigned)pad_len_;
                },
                [&](size_t, const uint8_t*, int r) {
                    if (r < 0 && !io_err) io_err = -r;
                });
            const auto b = Clock::now();
            if (elapsed_us) *elapsed_us = std::chrono::duration<double, std::micro>(b - a).count();
            return rc == 0 && io_err == 0;
        };
        if (!run_batch(8, nullptr))
        {
            if (out) *out = res;
            return 0;
        }
        double t1[8]{}, t8[8]{};
        for (int i = 0; i < 8; ++i)
        {
            if (!run_batch(1, &t1[i]) || !run_batch(8, &t8[i]))
            {
                if (out) *out = res;
                return 0;
            }
        }
        const auto t_probe1 = Clock::now();
        std::sort(t1, t1 + 8);
        std::sort(t8, t8 + 8);
        const double med1 = 0.5 * (t1[3] + t1[4]);
        const double med8 = 0.5 * (t8[3] + t8[4]);
        const bool enable = (med1 - med8 > 50.0) && (med1 > med8 * 1.15);
        pad_mod_ = enable ? 8u : 0u;
        res.ran = true;
        res.pad_mod = pad_mod_;
        res.n1_us = med1;
        res.n8_us = med8;
        res.probe_ms = std::chrono::duration<double, std::milli>(t_probe1 - t_probe0).count();
        if (out) *out = res;
        return pad_mod_;
    }

    // prepare(task, sqe) sets fd/off. complete consumes a CQE before slot reuse.
    // Negative results are -errno. No pool threads or SQPOLL.
    template<class Prepare, class Complete>
    int run(size_t n, const Prepare& prepare, const Complete& complete)
    {
        if (broken_) return broken_;
        if (!n) return 0;
        static constexpr uint64_t kPadTag = 1ULL << 63;
        size_t next = 0, done = 0;
        const size_t pad_target = (pad_mod_ > 1 && qd_ >= pad_mod_ && (n % pad_mod_) != 0)
                                      ? (pad_mod_ - (n % pad_mod_))
                                      : 0;
        size_t pad_queued = 0, pad_done = 0;
        int pad_fd = -1;
        uint64_t pad_off = 0;
        unsigned tail = acquire(sq_tail_);
        auto queue = [&](unsigned slot, size_t task)
        {
            tasks_[slot] = task;
            const unsigned ix = tail++ & sq_mask_;
            io_uring_sqe& e = sqes_[ix];
            memset(&e, 0, sizeof e);
            e.opcode = IORING_OP_READ;
            e.addr = (uint64_t)(uintptr_t)(buffers_ + slot * stride_);
            e.len = (unsigned)block_;
            e.user_data = slot;
            prepare(task, e);
            if (pad_fd < 0 && e.fd >= 0) { pad_fd = e.fd; pad_off = e.off; }
            array_[ix] = ix;
        };
        auto queue_pad = [&](unsigned slot)
        {
            const unsigned ix = tail++ & sq_mask_;
            io_uring_sqe& e = sqes_[ix];
            memset(&e, 0, sizeof e);
            e.opcode = IORING_OP_READ;
            e.fd = pad_fd;
            e.off = pad_off;
            e.addr = (uint64_t)(uintptr_t)(buffers_ + slot * stride_);
            e.len = (unsigned)pad_len_;
            e.user_data = kPadTag | (uint64_t)slot;
            array_[ix] = ix;
            ++pad_queued;
        };
        unsigned used_slots = 0;
        for (; used_slots < qd_ && next < n; ++used_slots) queue(used_slots, next++);
        // Pad O_DIRECT submissions up to a multiple of pad_mod_ with dummy reads ignored on
        // completion, avoiding the ~200 us NVMe interrupt-coalescing cliff on non-multiple-of-8
        // batches (both small decode fetches and large prefill tails).
        for (; used_slots < qd_ && pad_fd >= 0 && pad_queued < pad_target; ++used_slots) queue_pad(used_slots);
        while (done < n || pad_done < pad_queued)
        {
            publish(sq_tail_, tail);
            // Re-read kernel head after EINTR or partial submission; never blindly
            // resubmit the original count (some SQEs may already be consumed).
            const unsigned pending = tail - acquire(sq_head_);
            const int rc = (int)syscall(__NR_io_uring_enter, fd_, pending, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
            if (rc < 0 && errno != EINTR)
            {
                broken_ = errno;
                // A failed enter may have consumed SQEs. close() alone is not a
                // completion fence on older kernels: drain every accepted read
                // before freeing buffers. No SQPOLL means the SQ head is stable
                // here. Leave the unaccepted SQEs unsubmitted; the ring is retired.
                size_t live = (next + pad_queued) - (done + pad_done) - (tail - acquire(sq_head_));
                while (live)
                {
                    unsigned h = acquire(cq_head_);
                    const unsigned t = acquire(cq_tail_);
                    for (; h != t; ++h)
                    {
                        const io_uring_cqe c = cqes_[h & cq_mask_];
                        if (c.user_data & kPadTag) { ++pad_done; }
                        else
                        {
                            const unsigned slot = (unsigned)c.user_data;
                            complete(tasks_[slot], buffers_ + slot * stride_, c.res);
                            ++done;
                        }
                        --live;
                    }
                    publish(cq_head_, h);
                    if (live)
                    {
                        // poll also works if enter itself is denied; interrupted
                        // waits simply check CQ again. No deferred-taskrun setup.
                        pollfd p{fd_, POLLIN, 0};
                        (void)poll(&p, 1, -1);
                    }
                }
                release();
                return broken_;
            }
            unsigned head = acquire(cq_head_);
            const unsigned end = acquire(cq_tail_);
            for (; head != end; ++head)
            {
                const io_uring_cqe c = cqes_[head & cq_mask_];
                const unsigned slot = (unsigned)c.user_data;
                if (c.user_data & kPadTag)
                {
                    ++pad_done;
                    if (next < n) queue(slot, next++);
                    else if (pad_fd >= 0 && pad_queued < pad_target) queue_pad(slot);
                    continue;
                }
                const size_t task = tasks_[slot];
                if (c.res == -EINTR) { queue(slot, task); continue; }
                complete(task, buffers_ + slot * stride_, c.res);
                ++done;
                if (next < n) queue(slot, next++);
                else if (pad_fd >= 0 && pad_queued < pad_target) queue_pad(slot);
            }
            publish(cq_head_, head);
        }
        return 0;
    }

private:
    static unsigned* word(uint8_t* base, unsigned off) { return (unsigned*)(base + off); }
    static unsigned acquire(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
    static void publish(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
    uint8_t* map(size_t n, uint64_t off)
    { return (uint8_t*)mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, (off_t)off); }
    void release()
    {
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
        if (sqes_ != MAP_FAILED) { munmap(sqes_, sqe_bytes_); sqes_ = (io_uring_sqe*)MAP_FAILED; }
        if (!single_ && cq_ != MAP_FAILED) munmap(cq_, cq_bytes_);
        if (sq_ != MAP_FAILED) munmap(sq_, sq_bytes_);
        sq_ = cq_ = (uint8_t*)MAP_FAILED;
        free(buffers_); buffers_ = nullptr;
    }
    int fd_ = -1, broken_ = 0;
    bool single_ = false;
    uint8_t *sq_ = (uint8_t*)MAP_FAILED, *cq_ = (uint8_t*)MAP_FAILED, *buffers_ = nullptr;
    io_uring_sqe* sqes_ = (io_uring_sqe*)MAP_FAILED;
    io_uring_cqe* cqes_ = nullptr;
    unsigned *sq_head_ = nullptr, *sq_tail_ = nullptr, *cq_head_ = nullptr, *cq_tail_ = nullptr, *array_ = nullptr;
    unsigned sq_mask_ = 0, cq_mask_ = 0, qd_ = 0, pad_mod_ = 0;
    size_t sq_bytes_ = 0, cq_bytes_ = 0, sqe_bytes_ = 0, block_ = 0, pad_len_ = 0, stride_ = 0;
    std::vector<size_t> tasks_;
};
}} // namespace ds41::engram
