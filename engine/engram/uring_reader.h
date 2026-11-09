// Host-only Engram block reader. Our raw-ring implementation; no liburing.
// One serialized fetch lane owns a ring and QD aligned buffers, reused across fetches.
// Linux UAPI layout; the gr8-team erb.c experiment established the raw syscall route.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <linux/io_uring.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace ds41 { namespace engram {
class UringReader
{
public:
    UringReader() = default;
    UringReader(const UringReader&) = delete;
    UringReader& operator=(const UringReader&) = delete;
    ~UringReader() { release(); }

    // Returns errno; partially initialized readers are safe to destroy.
    int start(unsigned qd, size_t block, size_t align)
    {
        if (!qd || block > (size_t)std::numeric_limits<int>::max()) return EINVAL;
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
        stride_ = ((block + align - 1) / align) * align;
        if (stride_ > SIZE_MAX / qd_) return ENOMEM;
        void* buf = nullptr;
        const int e = posix_memalign(&buf, align, stride_ * qd_);
        if (e) return e;
        buffers_ = (uint8_t*)buf;
        tasks_.resize(qd_);
        return 0;
    }

    // prepare(task, sqe) sets fd/off. complete consumes a CQE before slot reuse.
    // Negative results are -errno. No pool threads or SQPOLL.
    template<class Prepare, class Complete>
    int run(size_t n, const Prepare& prepare, const Complete& complete)
    {
        if (broken_) return broken_;
        if (!n) return 0;
        size_t next = 0, done = 0;
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
            array_[ix] = ix;
        };
        for (unsigned slot = 0; slot < qd_ && next < n; ++slot) queue(slot, next++);
        while (done < n)
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
                size_t live = next - done - (tail - acquire(sq_head_));
                while (live)
                {
                    unsigned h = acquire(cq_head_);
                    const unsigned t = acquire(cq_tail_);
                    for (; h != t; ++h)
                    {
                        const io_uring_cqe c = cqes_[h & cq_mask_];
                        complete(tasks_[(unsigned)c.user_data], buffers_ + (unsigned)c.user_data * stride_, c.res);
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
                const size_t task = tasks_[slot];
                if (c.res == -EINTR) { queue(slot, task); continue; }
                complete(task, buffers_ + slot * stride_, c.res);
                ++done;
                if (next < n) queue(slot, next++);
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
    unsigned sq_mask_ = 0, cq_mask_ = 0, qd_ = 0;
    size_t sq_bytes_ = 0, cq_bytes_ = 0, sqe_bytes_ = 0, block_ = 0, stride_ = 0;
    std::vector<size_t> tasks_;
};
}} // namespace ds41::engram
