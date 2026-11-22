// Qwen3.8-Flash-Next Layer-1 PLE NVMe Row Store + RAM Set-Associative LRU Cache (host only, no CUDA).
// Verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py (Qwen4ExpTextNGramEmbedding):
//   - Table file: <release>/ple/exlr8-ple-L001.bin (contiguous ~320M rows x 160 BF16 = 320 B/row, ~95.4 GiB)
//   - Metadata:   <release>/ple/exlr8-ple-L001.json (or tensor buffers layer_multipliers[3], ngram_heads_offsets[16],
//                 ngram_heads_vocab_sizes[16], eos_token_id=248044)
//   - N-gram hash per token t with 2-token prior context (initialized to eos_token_id = 248044):
//       _shift_right_ignore_eos: resets segment at any eos_token_id; out-of-segment shifts become eos_token_id.
//       ngram=2 (heads 0..7):  mixed = (s0 * mult[0]) ^ (s1 * mult[1])
//       ngram=3 (heads 8..15): mixed = (s0 * mult[0]) ^ (s1 * mult[1]) ^ (s2 * mult[2])
//       row_id[h] = ((mixed % vocab_size[h]) + vocab_size[h]) % vocab_size[h] + offset[h]   (signed Python/PyTorch remainder!)
//   - I/O: O_DIRECT + io_uring (reusing engram/uring_reader.h) with automatic buffered pread fallback on NFS/tmpfs,
//     handling 320-byte rows that straddle 4096-byte page boundaries cleanly, fronted by a 16-way set-associative
//     true-LRU RAM cache (default 4 GiB = ~13.4M hot rows).
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "uring_reader.h"
#include "../model_geom_qw38.h"

namespace qw38 {

constexpr size_t PLE_ROW_BYTES = size_t(geom::PLE_HEAD_DIM) * 2; // 320 bytes (160 BF16)

struct PleStats {
    uint64_t req = 0, hits = 0, misses = 0, reads = 0;
    double io_us = 0.0, total_us = 0.0;
};

struct PleStoreOpts {
    size_t cache_bytes = 4ull << 30;  // 4 GiB RAM LRU cache (~13.4M rows)
    int ways = 16;                    // 1..16
    int io_uring = 128;               // >0: io_uring QD; 0: pread
    bool direct = true;               // O_DIRECT when filesystem allows, else automatic fallback
};

// 16-way set-associative true-LRU cache for 320-byte PLE rows (64-bit row_id + 1 tags).
class PleRowCache {
public:
    ~PleRowCache() {
        if (slots_) ::munmap(slots_, nsets_ * size_t(ways_) * PLE_ROW_BYTES);
    }
    bool init(size_t bytes, int ways, std::string* err) {
        if (ways < 1 || ways > 16) { if (err) *err = "ple cache ways must be 1..16"; return false; }
        const size_t nslots = bytes / PLE_ROW_BYTES;
        nsets_ = nslots / size_t(ways);
        if (!nsets_) { if (err) *err = "ple cache smaller than one set"; return false; }
        ways_ = ways;
        void* m = ::mmap(nullptr, nsets_ * size_t(ways_) * PLE_ROW_BYTES,
                         PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) {
            if (err) *err = std::string("ple cache mmap: ") + std::strerror(errno);
            nsets_ = 0;
            return false;
        }
        slots_ = static_cast<uint8_t*>(m);
        tags_.assign(nsets_ * size_t(ways_), 0);
        uint64_t ord = 0;
        for (int p = 0; p < 16; ++p) ord |= uint64_t(p) << (4 * p);
        order_.assign(nsets_, ord);
        return true;
    }
    const uint8_t* find(uint64_t row_id) {
        if (!nsets_) return nullptr;
        const uint64_t tag = row_id + 1;
        const size_t s = set_of(tag);
        const uint64_t* t = &tags_[s * size_t(ways_)];
        for (int w = 0; w < ways_; ++w) {
            if (t[w] == tag) {
                touch(s, w);
                return slots_ + (s * size_t(ways_) + size_t(w)) * PLE_ROW_BYTES;
            }
        }
        return nullptr;
    }
    void insert(uint64_t row_id, const uint8_t* row) {
        if (!nsets_) return;
        const uint64_t tag = row_id + 1;
        const size_t s = set_of(tag);
        uint64_t* t = &tags_[s * size_t(ways_)];
        int v = -1;
        for (int w = 0; w < ways_; ++w) if (t[w] == 0) { v = w; break; }
        if (v < 0) v = int((order_[s] >> (4 * (ways_ - 1))) & 15);
        t[v] = tag;
        std::memcpy(slots_ + (s * size_t(ways_) + size_t(v)) * PLE_ROW_BYTES, row, PLE_ROW_BYTES);
        touch(s, v);
    }
    size_t capacity_rows() const { return nsets_ * size_t(ways_); }
private:
    size_t set_of(uint64_t tag) const {
        return size_t((((tag * 0x9E3779B97F4A7C15ull) >> 32) * nsets_) >> 32);
    }
    void touch(size_t s, int w) {
        const uint64_t o = order_[s];
        int p = 0;
        while (((o >> (4 * p)) & 15) != uint64_t(w)) ++p;
        const uint64_t low = p ? (o & ((1ull << (4 * p)) - 1)) : 0;
        const uint64_t high = p >= 15 ? 0 : ((o >> (4 * (p + 1))) << (4 * (p + 1)));
        order_[s] = high | (low << 4) | uint64_t(w);
    }
    uint8_t* slots_ = nullptr;
    size_t nsets_ = 0;
    int ways_ = 0;
    std::vector<uint64_t> tags_;
    std::vector<uint64_t> order_;
};

class PleStore {
public:
    ~PleStore() {
        uring_.reset();
        if (iobuf_) ::free(iobuf_);
        if (fd_ >= 0) ::close(fd_);
    }

    // Configure hash parameters from the checkpoint's int64 buffers.
    void set_hash_params(const int64_t mult[3], const int64_t offsets[16], const int64_t vocab_sizes[16], int32_t eos_tok = geom::TOK_EOS) {
        for (int i = 0; i < 3; ++i) mult_[i] = mult[i];
        for (int i = 0; i < 16; ++i) { offsets_[i] = offsets[i]; vsizes_[i] = vocab_sizes[i]; }
        eos_tok_ = eos_tok;
        has_hash_ = true;
    }

    bool open(const std::string& bin_path, const PleStoreOpts& opts, std::string* err) {
        opts_ = opts;
        int flags = O_RDONLY | O_CLOEXEC;
        bool direct_ok = false;
#ifdef O_DIRECT
        if (opts.direct) {
            fd_ = ::open(bin_path.c_str(), flags | O_DIRECT);
            if (fd_ >= 0) direct_ok = true;
        }
#endif
        if (fd_ < 0) fd_ = ::open(bin_path.c_str(), flags);
        if (fd_ < 0) {
            if (err) *err = "ple open " + bin_path + ": " + std::strerror(errno);
            return false;
        }
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            if (err) *err = "ple fstat " + bin_path + ": " + std::strerror(errno);
            return false;
        }
        file_bytes_ = uint64_t(st.st_size);
        total_rows_ = file_bytes_ / PLE_ROW_BYTES;
        direct_ = direct_ok;
        align_ = 4096;

        // Probe O_DIRECT alignment (some FSes accept open(O_DIRECT) but fail aligned pread if block > 4096)
        void* probe = nullptr;
        if (::posix_memalign(&probe, align_, 8192) != 0) {
            if (err) *err = "ple posix_memalign failed";
            return false;
        }
        if (direct_ && file_bytes_ >= align_) {
            if (::pread(fd_, probe, align_, 0) < 0) {
                ::close(fd_);
                fd_ = ::open(bin_path.c_str(), flags);
                direct_ = false;
            }
        }
        ::free(probe);

        if (opts.cache_bytes >= PLE_ROW_BYTES * size_t(opts.ways)) {
            if (!cache_.init(opts.cache_bytes, opts.ways, err)) return false;
        }

        if (opts.io_uring > 0) {
            auto u = std::make_unique<ds41::engram::UringReader>();
            if (u->start(unsigned(opts.io_uring), 8192, align_) == 0) {
                uring_ = std::move(u);
                iobuf_slots_ = opts.io_uring;
            }
        }
        return true;
    }

    bool is_open() const { return fd_ >= 0; }
    bool is_direct() const { return direct_; }
    bool has_uring() const { return uring_ != nullptr; }
    uint64_t total_rows() const { return total_rows_; }

    // Exact match to Qwen4ExpTextNGramEmbedding.forward() for one token sequence with 2-token prior context.
    // Updates `ctx[2]` in place when `update_ctx` is true. Writes `n * 16` global row IDs into `row_ids`.
    void compute_ids(const int* toks, int n, int32_t ctx[2], bool update_ctx, uint64_t* row_ids) const {
        // Build token_history = [ctx[0], ctx[1], toks[0..n-1]] (length n + 2)
        const size_t L = static_cast<size_t>(n + 2);
        std::vector<int64_t> thist(L);
        thist[0] = ctx[0];
        thist[1] = ctx[1];
        for (int i = 0; i < n; ++i) thist[static_cast<size_t>(i + 2)] = toks[i];

        // Track last EOS position strictly before index i (matching _shift_right_ignore_eos)
        std::vector<int> seg_start(L, 0);
        int last_eos = -1;
        for (int i = 0; i < static_cast<int>(L); ++i) {
            seg_start[size_t(i)] = last_eos + 1;
            if (thist[size_t(i)] == eos_tok_) last_eos = i;
        }

        auto shifted = [&](int i, int shift) -> int64_t {
            if (shift == 0) return thist[size_t(i)];
            const int pos_in_seg = i - seg_start[size_t(i)];
            const int src = i - shift;
            return (pos_in_seg >= shift && src >= 0) ? thist[size_t(src)] : int64_t(eos_tok_);
        };

        auto py_rem = [](int64_t a, int64_t m) -> int64_t {
            int64_t r = a % m;
            return r < 0 ? r + m : r;
        };

        for (int t = 0; t < n; ++t) {
            const int i = t + 2;
            const int64_t s0 = shifted(i, 0);
            const int64_t s1 = shifted(i, 1);
            const int64_t s2 = shifted(i, 2);
            const int64_t m2 = (s0 * mult_[0]) ^ (s1 * mult_[1]);
            const int64_t m3 = m2 ^ (s2 * mult_[2]);
            uint64_t* dst = row_ids + size_t(t) * geom::PLE_HEADS;
            for (int h = 0; h < 8; ++h) {
                dst[h] = uint64_t(py_rem(m2, vsizes_[h]) + offsets_[h]);
            }
            for (int h = 8; h < 16; ++h) {
                dst[h] = uint64_t(py_rem(m3, vsizes_[h]) + offsets_[h]);
            }
        }

        if (update_ctx) {
            if (n >= 2) {
                ctx[0] = toks[n - 2];
                ctx[1] = toks[n - 1];
            } else if (n == 1) {
                ctx[0] = ctx[1];
                ctx[1] = toks[0];
            }
        }
    }

    // Fetch `n_ids` 320-byte rows into contiguous `dst` (`n_ids * 320` bytes). Thread-safe.
    bool fetch_rows(const uint64_t* ids, size_t n_ids, uint8_t* dst, PleStats* st_out = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mu_);
        PleStats s{};
        s.req = n_ids;

        struct Miss { uint64_t id; size_t dst_idx; };
        std::vector<Miss> misses;
        misses.reserve(n_ids);

        for (size_t i = 0; i < n_ids; ++i) {
            const uint64_t id = ids[i];
            if (id >= total_rows_) {
                std::memset(dst + i * PLE_ROW_BYTES, 0, PLE_ROW_BYTES);
                continue;
            }
            if (const uint8_t* hit = cache_.find(id)) {
                std::memcpy(dst + i * PLE_ROW_BYTES, hit, PLE_ROW_BYTES);
                ++s.hits;
            } else {
                misses.push_back({id, i});
            }
        }
        s.misses = misses.size();

        const auto tio0 = std::chrono::steady_clock::now();
        if (!misses.empty()) {
            if (uring_) {
                int first_err = 0;
                const int rc = uring_->run(
                    misses.size(),
                    [&](size_t idx, io_uring_sqe& sqe) {
                        const uint64_t byte_off = misses[idx].id * PLE_ROW_BYTES;
                        const uint64_t aligned_off = byte_off & ~(uint64_t(align_) - 1);
                        const uint64_t end_off = (byte_off + PLE_ROW_BYTES + align_ - 1) & ~(uint64_t(align_) - 1);
                        sqe.fd = fd_;
                        sqe.off = aligned_off;
                        sqe.len = unsigned(std::min<uint64_t>(end_off - aligned_off, 8192));
                    },
                    [&](size_t idx, const uint8_t* buf, int res) {
                        if (res < int(PLE_ROW_BYTES)) {
                            if (!first_err) first_err = res < 0 ? -res : EIO;
                            return;
                        }
                        const uint64_t byte_off = misses[idx].id * PLE_ROW_BYTES;
                        const size_t in_buf = size_t(byte_off & (uint64_t(align_) - 1));
                        const uint8_t* row = buf + in_buf;
                        std::memcpy(dst + misses[idx].dst_idx * PLE_ROW_BYTES, row, PLE_ROW_BYTES);
                        cache_.insert(misses[idx].id, row);
                        ++s.reads;
                    });
                if (rc != 0 || first_err != 0) {
                    std::fprintf(stderr, "# qw38 PLE io_uring error rc=%d err=%d\n", rc, first_err);
                    return false;
                }
            } else {
                alignas(4096) uint8_t tmp[8192];
                for (const Miss& m : misses) {
                    const uint64_t byte_off = m.id * PLE_ROW_BYTES;
                    const uint64_t aligned_off = direct_ ? (byte_off & ~(uint64_t(align_) - 1)) : byte_off;
                    const size_t len = direct_
                        ? size_t(((byte_off + PLE_ROW_BYTES + align_ - 1) & ~(uint64_t(align_) - 1)) - aligned_off)
                        : PLE_ROW_BYTES;
                    const ssize_t got = ::pread(fd_, tmp, len, off_t(aligned_off));
                    if (got < ssize_t(PLE_ROW_BYTES)) return false;
                    const uint8_t* row = tmp + (byte_off - aligned_off);
                    std::memcpy(dst + m.dst_idx * PLE_ROW_BYTES, row, PLE_ROW_BYTES);
                    cache_.insert(m.id, row);
                    ++s.reads;
                }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        s.io_us = std::chrono::duration<double, std::micro>(t1 - tio0).count();
        s.total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (st_out) *st_out = s;
        return true;
    }

private:
    int fd_ = -1;
    bool direct_ = false;
    bool has_hash_ = false;
    size_t align_ = 4096;
    uint64_t file_bytes_ = 0;
    uint64_t total_rows_ = 0;
    PleStoreOpts opts_{};
    PleRowCache cache_;
    std::unique_ptr<ds41::engram::UringReader> uring_;
    void* iobuf_ = nullptr;
    int iobuf_slots_ = 0;
    int64_t mult_[3] = {};
    int64_t offsets_[16] = {};
    int64_t vsizes_[16] = {};
    int32_t eos_tok_ = geom::TOK_EOS;
    std::mutex mu_;
};

} // namespace qw38
