// Tencent Hy4-flash 4-Layer Engram NVMe Row Store + Set-Associative LRU RAM Cache (host only, zero CUDA deps).
// Verified verbatim against HF PR 50903 modeling_hy_v4_flash.py (HYV4FlashEngramLayer / HYV4FlashEngramEmbedding):
//   - 4 Engram layers: layer_idx in {1, 11, 21, 31} (1-indexed layer_id in {2, 12, 22, 32}).
//   - Files: <release>/engram/exlr8-engram-L{001,011,021,031}.bin
//   - On-disk row format (128 B/row = 32 rows / 4 KiB page, ZERO cross-page straddling!):
//       bytes [0..95]:  96 FP8-E4M3FN quantized values (exact match to PyTorch _fp8_round_trip)
//       bytes [96..97]: BF16 row scale (= clamp(max(|x|), 1e-12) / 448.0)
//       bytes [98..127]: zero padding
//   - Exact n-gram hash:
//       shifted = [tok(t), tok(t-1) if pos>=1 else PAD, tok(t-2) if pos>=2 else PAD]  (PAD = 120002)
//       compressed[k] = lookup_table[tok_k] if tok_k < lookup_len else lookup_table[PAD]
//       mixed[k] = int64(compressed[k]) * hash_multipliers[layer_slot][k]
//       order 2 (h = 0..7):  row = ((mixed[0] ^ mixed[1]) % prime[h]) + offset[h]
//       order 3 (h = 8..15): row = ((mixed[0] ^ mixed[1] ^ mixed[2]) % prime[h]) + offset[h]
//   - I/O: O_DIRECT + raw Linux io_uring (ds41::engram::UringReader) with automatic buffered pread fallback,
//     in-batch deduplication and LBA sorting, fronted by a 16-way set-associative true-LRU RAM cache
//     (or optional direct mmap when GR8_HY4_ENGRAM_MMAP=1 on discrete-GPU hosts with ample host RAM).
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
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "uring_reader.h"
#include "../model_geom_hy4.h"

namespace hy4 {

constexpr size_t ENGRAM_ROW_BYTES = geom::ENGRAM_ROW_BYTES; // 128 bytes
constexpr size_t ENGRAM_PAGE_BYTES = 4096;
constexpr size_t ENGRAM_ROWS_PER_PAGE = ENGRAM_PAGE_BYTES / ENGRAM_ROW_BYTES; // 32 exact

// Exact per-layer prime table sizes from Tencent Hy4-flash config.json (engram_layer_head_vocab_sizes).
// Slot 0 = L001 (id 2), Slot 1 = L011 (id 12), Slot 2 = L021 (id 22), Slot 3 = L031 (id 32).
constexpr uint64_t ENGRAM_PRIMES[4][16] = {
    { // L001 (layer_id = 2)
        10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147,
        10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259,
    },
    { // L011 (layer_id = 12)
        10312277, 10312283, 10312301, 10312343, 10312373, 10312397, 10312409, 10312453,
        10312457, 10312483, 10312507, 10312513, 10312531, 10312543, 10312567, 10312571,
    },
    { // L021 (layer_id = 22)
        10312613, 10312619, 10312633, 10312657, 10312681, 10312699, 10312723, 10312747,
        10312759, 10312789, 10312811, 10312823, 10312867, 10312873, 10312889, 10312903,
    },
    { // L031 (layer_id = 32)
        10312909, 10312919, 10312927, 10312957, 10312979, 10313011, 10313021, 10313057,
        10313063, 10313117, 10313123, 10313129, 10313161, 10313179, 10313189, 10313201,
    },
};

constexpr int ENGRAM_LAYER_IDX[4] = {1, 11, 21, 31};

// Exact NumPy default_rng(0 + 10007 * layer_id) odd multipliers for Hy4-flash (cv = 92620).
// Verified directly against numpy.random.default_rng on the real checkpoint lookup_tables.
constexpr int64_t ENGRAM_MULTS_CV92620[4][3] = {
    {81853575799317LL, 39690085502883LL, 24229915018165LL}, // L001 (id=2)
    {19457948045719LL, 44729556510419LL, 84779352922315LL}, // L011 (id=12)
    {72974230057307LL, 21530602121221LL, 55998057243097LL}, // L021 (id=22)
    {29128509562787LL, 16745155128523LL, 22474588302733LL}, // L031 (id=32)
};

// Exact NumPy SeedSequence + PCG64 (XSL-RR) matching numpy.random.default_rng for arbitrary cv.
struct NumpyPcg64 {
    unsigned __int128 state = 0;
    unsigned __int128 inc = 0;

    static inline uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
        value ^= hash_const;
        hash_const *= 0x931e8875u;
        value *= hash_const;
        value ^= value >> 16;
        return value;
    }
    static inline uint32_t mix(uint32_t x, uint32_t y) {
        uint32_t res = 0xca01f9ddu * x - 0x4973f715u * y;
        res ^= res >> 16;
        return res;
    }
    static void seed_seq_generate(uint64_t entropy, uint32_t out[8]) {
        uint32_t ent[2];
        int n_ent = 1;
        ent[0] = static_cast<uint32_t>(entropy & 0xffffffffull);
        if (entropy >> 32) {
            ent[1] = static_cast<uint32_t>(entropy >> 32);
            n_ent = 2;
        }
        uint32_t pool[4] = {0, 0, 0, 0};
        uint32_t hash_const = 0x43b0d7e5u;
        for (int i = 0; i < 4; ++i) pool[i] = hashmix(i < n_ent ? ent[i] : 0u, hash_const);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (i != j) pool[j] = mix(pool[j], hashmix(pool[i], hash_const));
            }
        }
        uint32_t hc2 = 0x8b51f9ddu;
        for (int i = 0; i < 8; ++i) {
            uint32_t v = pool[i % 4];
            v ^= hc2;
            hc2 *= 0x58f38dedu;
            v *= hc2;
            v ^= v >> 16;
            out[i] = v;
        }
    }
    void seed(uint64_t entropy) {
        uint32_t words[8];
        seed_seq_generate(entropy, words);
        uint64_t w64[4];
        for (int k = 0; k < 4; ++k) {
            w64[k] = static_cast<uint64_t>(words[2 * k]) | (static_cast<uint64_t>(words[2 * k + 1]) << 32);
        }
        const unsigned __int128 initstate = (static_cast<unsigned __int128>(w64[0]) << 64) | w64[1];
        const unsigned __int128 initseq   = (static_cast<unsigned __int128>(w64[2]) << 64) | w64[3];
        state = 0;
        inc = (initseq << 1) | 1;
        step();
        state += initstate;
        step();
    }
    inline void step() {
        constexpr unsigned __int128 MULT =
            (static_cast<unsigned __int128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
        state = state * MULT + inc;
    }
    inline uint64_t next_u64() {
        step();
        const uint64_t hi = static_cast<uint64_t>(state >> 64);
        const uint64_t lo = static_cast<uint64_t>(state);
        const uint64_t x = hi ^ lo;
        const unsigned rot = static_cast<unsigned>(state >> 122);
        return (x >> rot) | (x << ((-rot) & 63));
    }
    inline int64_t bounded_u64(uint64_t rng) {
        // NumPy random_bounded_uint64_lemire (rng = high - 1):
        unsigned __int128 m = static_cast<unsigned __int128>(next_u64()) * (static_cast<unsigned __int128>(rng) + 1);
        uint64_t l = static_cast<uint64_t>(m);
        if (l < rng) {
            const uint64_t t = (0ULL - (rng + 1)) % (rng + 1);
            while (l < t) {
                m = static_cast<unsigned __int128>(next_u64()) * (static_cast<unsigned __int128>(rng) + 1);
                l = static_cast<uint64_t>(m);
            }
        }
        return static_cast<int64_t>(m >> 64);
    }
};

inline std::array<int64_t, 3> hy4_draw_hash_multipliers(int layer_slot, int64_t compressed_vocab_size,
                                                        uint64_t base_seed = 0) {
    if (base_seed == 0 && compressed_vocab_size == 92620) {
        return {ENGRAM_MULTS_CV92620[layer_slot][0],
                ENGRAM_MULTS_CV92620[layer_slot][1],
                ENGRAM_MULTS_CV92620[layer_slot][2]};
    }
    const uint64_t layer_id = static_cast<uint64_t>(ENGRAM_LAYER_IDX[layer_slot] + 1);
    const int64_t half_bound = std::max<int64_t>(1, INT64_MAX / compressed_vocab_size / 2);
    NumpyPcg64 rng;
    rng.seed(base_seed + 10007ULL * layer_id);
    std::array<int64_t, 3> out{};
    for (int k = 0; k < 3; ++k) {
        out[k] = rng.bounded_u64(static_cast<uint64_t>(half_bound - 1)) * 2 + 1;
    }
    return out;
}

struct EngramStats {
    uint64_t req = 0, hits = 0, misses = 0, reads = 0;
    double io_us = 0.0, total_us = 0.0;
};

struct EngramStoreOpts {
    size_t cache_bytes = 2ull << 30; // 2 GiB unified LRU cache across all 4 layers (~16M cached rows)
    int ways = 16;
    int io_uring = 128;
    bool direct = true;
    bool use_mmap = false;
    std::string compact_dir;
};

class EngramStore {
public:
    EngramStore() = default;
    ~EngramStore() { close_all(); }
    EngramStore(const EngramStore&) = delete;
    EngramStore& operator=(const EngramStore&) = delete;

    bool open(const std::string& engram_dir,
              const std::array<std::vector<int32_t>, 4>& lookup_tables,
              const EngramStoreOpts& opts,
              std::string& err) {
        close_all();
        opts_ = opts;
        for (int s = 0; s < 4; ++s) {
            if (lookup_tables[s].empty()) {
                err = "hy4 engram: empty lookup_table for slot " + std::to_string(s);
                return false;
            }
            luts_[s] = lookup_tables[s];
            int32_t mx = 0;
            for (int32_t v : luts_[s]) if (v > mx) mx = v;
            cv_[s] = static_cast<int64_t>(mx) + 1;
            mults_[s] = hy4_draw_hash_multipliers(s, cv_[s], 0);
            uint64_t off = 0;
            for (int h = 0; h < 16; ++h) {
                offsets_[s][h] = off;
                off += ENGRAM_PRIMES[s][h];
            }
            nrows_[s] = off;

            char fname[64];
            std::snprintf(fname, sizeof(fname), "exlr8-engram-L%03d.bin", ENGRAM_LAYER_IDX[s]);
            const std::string path = engram_dir + "/" + fname;
            const uint64_t expect_bytes = nrows_[s] * ENGRAM_ROW_BYTES;

            bool is_direct = false;
            int fd = -1;
            if (!opts_.use_mmap && opts_.direct) {
                fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
                if (fd >= 0) is_direct = true;
                else if (errno == EINVAL) fd = ::open(path.c_str(), O_RDONLY);
            } else {
                fd = ::open(path.c_str(), O_RDONLY);
            }
            if (fd < 0) {
                err = "hy4 engram: open(" + path + "): " + std::strerror(errno);
                close_all();
                return false;
            }
            struct stat st{};
            if (fstat(fd, &st) != 0) {
                err = "hy4 engram: fstat(" + path + "): " + std::strerror(errno);
                ::close(fd);
                close_all();
                return false;
            }
            if (static_cast<uint64_t>(st.st_size) != expect_bytes) {
                err = "hy4 engram: " + path + " size " + std::to_string(st.st_size) +
                      " != expected " + std::to_string(expect_bytes);
                ::close(fd);
                close_all();
                return false;
            }
            fds_[s] = fd;
            direct_[s] = is_direct;
            file_bytes_[s] = static_cast<size_t>(st.st_size);

            if (opts_.use_mmap) {
                void* mp = mmap(nullptr, file_bytes_[s], PROT_READ, MAP_PRIVATE, fd, 0);
                if (mp == MAP_FAILED) {
                    err = "hy4 engram: mmap(" + path + "): " + std::strerror(errno);
                    close_all();
                    return false;
                }
                madvise(mp, file_bytes_[s], MADV_RANDOM);
                mmaps_[s] = static_cast<const uint8_t*>(mp);
            }
        }

        if (!opts_.use_mmap) {
            if (opts_.io_uring > 0) {
                const int rc = uring_.start(static_cast<unsigned>(opts_.io_uring), ENGRAM_PAGE_BYTES, ENGRAM_PAGE_BYTES);
                if (rc == 0) use_uring_ = true;
                else std::fprintf(stderr, "# hy4 engram: io_uring unavailable (errno=%d), falling back to pread\n", rc);
            }
            ways_ = std::max(1, std::min(16, opts_.ways));
            const size_t entry_bytes = ENGRAM_ROW_BYTES + sizeof(uint64_t) + 1;
            const size_t total_slots = opts_.cache_bytes / entry_bytes;
            const size_t nsets = total_slots / static_cast<size_t>(ways_);
            size_t p2 = 1;
            while ((p2 << 1) <= nsets) p2 <<= 1;
            nsets_ = (opts_.cache_bytes >= entry_bytes * static_cast<size_t>(ways_)) ? p2 : 0;
            if (nsets_ > 0) {
                const size_t n = nsets_ * static_cast<size_t>(ways_);
                cache_keys_.assign(n, EMPTY_KEY);
                cache_age_.assign(n, 0);
                cache_vals_.resize(n * ENGRAM_ROW_BYTES);
            }
            void* p = nullptr;
            if (posix_memalign(&p, ENGRAM_PAGE_BYTES, ENGRAM_PAGE_BYTES) != 0) {
                err = "hy4 engram: posix_memalign failed";
                close_all();
                return false;
            }
            pread_buf_ = static_cast<uint8_t*>(p);
        }

        if (!opts_.compact_dir.empty()) {
            const char* cdir = opts_.compact_dir.c_str();
            if (cdir[0]) {
                size_t tot_c = 0;
                for (int s = 0; s < 4; ++s) {
                    char f1[256], f2[256];
                    std::snprintf(f1, sizeof(f1), "%s/L%03d_ids.npy", cdir, ENGRAM_LAYER_IDX[s]);
                    std::snprintf(f2, sizeof(f2), "%s/L%03d_rows.bin", cdir, ENGRAM_LAYER_IDX[s]);
                    FILE* fp1 = std::fopen(f1, "rb");
                    FILE* fp2 = std::fopen(f2, "rb");
                    if (fp1 && fp2) {
                        uint8_t hdr[10];
                        if (std::fread(hdr, 1, 10, fp1) == 10 && hdr[0] == 0x93) {
                            uint32_t hl = (hdr[6] == 1) ? (uint32_t(hdr[8]) | (uint32_t(hdr[9]) << 8)) : 0;
                            if (hdr[6] >= 2) { std::fseek(fp1, 8, SEEK_SET); if (std::fread(&hl, 4, 1, fp1) != 1) hl = 0; }
                            std::fseek(fp1, (hdr[6] == 1 ? 10 : 12) + hl, SEEK_SET);
                            std::fseek(fp2, 0, SEEK_END);
                            const size_t rb = static_cast<size_t>(std::ftell(fp2));
                            std::fseek(fp2, 0, SEEK_SET);
                            const size_t n = rb / ENGRAM_ROW_BYTES;
                            compact_ids_[s].resize(n);
                            compact_rows_[s].resize(rb);
                            if (std::fread(compact_ids_[s].data(), 8, n, fp1) == n &&
                                std::fread(compact_rows_[s].data(), 1, rb, fp2) == rb) {
                                tot_c += n;
                            } else { compact_ids_[s].clear(); compact_rows_[s].clear(); }
                        }
                    }
                    if (fp1) std::fclose(fp1);
                    if (fp2) std::fclose(fp2);
                }
                if (tot_c > 0) std::fprintf(stderr, "# hy4 engram: loaded compact exact eval cache (%zu rows from %s)\n", tot_c, cdir);
            }
        }
        opened_ = true;
        std::fprintf(stderr, "# hy4 engram: opened 4 tables (%.2f GiB) mode=%s direct=%d uring=%d cache=%.1f MiB (%zu sets x %d ways)\n",
                     double(file_bytes_[0] + file_bytes_[1] + file_bytes_[2] + file_bytes_[3]) / double(1ull << 30),
                     opts_.use_mmap ? "mmap" : (use_uring_ ? "io_uring" : "pread"),
                     int(direct_[0]), int(use_uring_),
                     double(nsets_ * size_t(ways_) * ENGRAM_ROW_BYTES) / double(1ull << 20),
                     nsets_, ways_);
        return true;
    }

    void close_all() {
        for (int s = 0; s < 4; ++s) {
            if (mmaps_[s]) { munmap(const_cast<uint8_t*>(mmaps_[s]), file_bytes_[s]); mmaps_[s] = nullptr; }
            if (fds_[s] >= 0) { ::close(fds_[s]); fds_[s] = -1; }
            direct_[s] = false;
            file_bytes_[s] = 0;
        }
        if (pread_buf_) { std::free(pread_buf_); pread_buf_ = nullptr; }
        cache_keys_.clear();
        cache_age_.clear();
        cache_vals_.clear();
        nsets_ = 0;
        for (int s = 0; s < 4; ++s) { compact_ids_[s].clear(); compact_rows_[s].clear(); }
        use_uring_ = false;
        opened_ = false;
    }

    bool is_open() const { return opened_; }
    const EngramStats& stats() const { return stats_; }
    const std::array<int64_t, 3>& multipliers(int layer_slot) const { return mults_[layer_slot]; }

    inline void hash_token(int layer_slot, int32_t tok0, int32_t tok1, int32_t tok2, uint64_t out_rows[16]) const {
        const auto& lut = luts_[layer_slot];
        const int32_t tsz = static_cast<int32_t>(lut.size());
        const int32_t pad_c = lut[geom::ENGRAM_PAD_ID];
        auto comp = [&](int32_t t) -> int64_t {
            if (t >= 0 && t < tsz) return static_cast<int64_t>(lut[static_cast<size_t>(t)]);
            return static_cast<int64_t>(pad_c);
        };
        const auto& mu = mults_[layer_slot];
        const int64_t m0 = comp(tok0) * mu[0];
        const int64_t m1 = comp(tok1) * mu[1];
        const int64_t m2 = comp(tok2) * mu[2];
        const uint64_t x2 = static_cast<uint64_t>(m0 ^ m1);
        const uint64_t x3 = static_cast<uint64_t>(m0 ^ m1 ^ m2);
        const uint64_t* pr = ENGRAM_PRIMES[layer_slot];
        const uint64_t* of = offsets_[layer_slot].data();
        for (int h = 0; h < 8; ++h)  out_rows[h] = (x2 % pr[h]) + of[h];
        for (int h = 8; h < 16; ++h) out_rows[h] = (x3 % pr[h]) + of[h];
    }

    bool fetch_rows(int layer_slot, const uint64_t* row_ids, size_t n_rows, uint8_t* dst) {
        if (!opened_ || layer_slot < 0 || layer_slot >= 4) return false;
        if (n_rows == 0) return true;
        const auto t0 = std::chrono::steady_clock::now();
        stats_.req += n_rows;

        if (mmaps_[layer_slot]) {
            const uint8_t* base = mmaps_[layer_slot];
            for (size_t i = 0; i < n_rows; ++i) {
                std::memcpy(dst + i * ENGRAM_ROW_BYTES, base + row_ids[i] * ENGRAM_ROW_BYTES, ENGRAM_ROW_BYTES);
            }
            stats_.hits += n_rows;
            stats_.total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            return true;
        }

        const auto& cids = compact_ids_[layer_slot];
        const auto& crows = compact_rows_[layer_slot];
        misses_tmp_.clear();
        for (size_t i = 0; i < n_rows; ++i) {
            const uint64_t r = row_ids[i];
            if (!cids.empty()) {
                auto it = std::lower_bound(cids.begin(), cids.end(), r);
                if (it != cids.end() && *it == r) {
                    const size_t ci = static_cast<size_t>(it - cids.begin());
                    std::memcpy(dst + i * ENGRAM_ROW_BYTES, crows.data() + ci * ENGRAM_ROW_BYTES, ENGRAM_ROW_BYTES);
                    ++stats_.hits;
                    continue;
                }
            }
            const uint64_t key = make_key(layer_slot, r);
            if (nsets_ > 0 && cache_get(key, dst + i * ENGRAM_ROW_BYTES)) {
                ++stats_.hits;
            } else {
                ++stats_.misses;
                misses_tmp_.push_back({r, i, r / ENGRAM_ROWS_PER_PAGE});
            }
        }
        if (misses_tmp_.empty()) {
            stats_.total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            return true;
        }

        std::sort(misses_tmp_.begin(), misses_tmp_.end(), [](const Miss& a, const Miss& b) {
            return a.page < b.page || (a.page == b.page && a.row < b.row);
        });
        pages_tmp_.clear();
        for (size_t i = 0; i < misses_tmp_.size();) {
            size_t j = i + 1;
            while (j < misses_tmp_.size() && misses_tmp_[j].page == misses_tmp_[i].page) ++j;
            pages_tmp_.push_back({misses_tmp_[i].page, i, j});
            i = j;
        }

        const int fd = fds_[layer_slot];
        const size_t fbytes = file_bytes_[layer_slot];
        const auto tio0 = std::chrono::steady_clock::now();
        stats_.reads += pages_tmp_.size();

        auto handle_page = [&](size_t pi, const uint8_t* page_buf, ssize_t got) -> bool {
            const PageJob& pj = pages_tmp_[pi];
            const uint64_t page_off = pj.page * ENGRAM_PAGE_BYTES;
            const size_t valid = (page_off < fbytes) ? std::min<size_t>(ENGRAM_PAGE_BYTES, fbytes - page_off) : 0;
            if (got < static_cast<ssize_t>(valid)) return false;
            for (size_t k = pj.m0; k < pj.m1; ++k) {
                const uint64_t r = misses_tmp_[k].row;
                const size_t off_in_page = (r % ENGRAM_ROWS_PER_PAGE) * ENGRAM_ROW_BYTES;
                const uint8_t* src = page_buf + off_in_page;
                std::memcpy(dst + misses_tmp_[k].out_idx * ENGRAM_ROW_BYTES, src, ENGRAM_ROW_BYTES);
                if (nsets_ > 0 && (k == pj.m0 || r != misses_tmp_[k - 1].row)) {
                    cache_put(make_key(layer_slot, r), src);
                }
            }
            return true;
        };

        bool ok = true;
        if (use_uring_ && pages_tmp_.size() >= 2) {
            int io_err = 0;
            const int rc = uring_.run(
                pages_tmp_.size(),
                [&](size_t pi, io_uring_sqe& sqe) {
                    sqe.fd = fd;
                    sqe.off = pages_tmp_[pi].page * ENGRAM_PAGE_BYTES;
                },
                [&](size_t pi, const uint8_t* buf, int res) {
                    if (res < 0 || !handle_page(pi, buf, res)) io_err = (res < 0) ? -res : EIO;
                });
            if (rc != 0 || io_err != 0) ok = false;
        } else {
            for (size_t pi = 0; pi < pages_tmp_.size(); ++pi) {
                const uint64_t off = pages_tmp_[pi].page * ENGRAM_PAGE_BYTES;
                const ssize_t got = ::pread(fd, pread_buf_, ENGRAM_PAGE_BYTES, static_cast<off_t>(off));
                if (got < 0 || !handle_page(pi, pread_buf_, got)) { ok = false; break; }
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        stats_.io_us += std::chrono::duration<double, std::micro>(t1 - tio0).count();
        stats_.total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        return ok;
    }

private:
    static constexpr uint64_t EMPTY_KEY = ~0ull;
    static inline uint64_t make_key(int layer_slot, uint64_t row) {
        return (static_cast<uint64_t>(layer_slot) << 56) | row;
    }
    inline size_t set_index(uint64_t key) const {
        uint64_t h = key ^ (key >> 30);
        h *= 0xbf58476d1ce4e5b9ull;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebull;
        h ^= h >> 31;
        return static_cast<size_t>(h) & (nsets_ - 1);
    }
    inline bool cache_get(uint64_t key, uint8_t* dst) {
        const size_t base = set_index(key) * static_cast<size_t>(ways_);
        const uint64_t* ks = cache_keys_.data() + base;
        uint8_t* age = cache_age_.data() + base;
        for (int w = 0; w < ways_; ++w) {
            if (ks[w] == key) {
                std::memcpy(dst, cache_vals_.data() + (base + static_cast<size_t>(w)) * ENGRAM_ROW_BYTES, ENGRAM_ROW_BYTES);
                const uint8_t old_a = age[w];
                for (int j = 0; j < ways_; ++j) if (age[j] < old_a) ++age[j];
                age[w] = 0;
                return true;
            }
        }
        return false;
    }
    inline void cache_put(uint64_t key, const uint8_t* src) {
        const size_t base = set_index(key) * static_cast<size_t>(ways_);
        uint64_t* ks = cache_keys_.data() + base;
        uint8_t* age = cache_age_.data() + base;
        int victim = 0;
        uint8_t max_a = 0;
        for (int w = 0; w < ways_; ++w) {
            if (ks[w] == key || ks[w] == EMPTY_KEY) { victim = w; break; }
            if (age[w] >= max_a) { max_a = age[w]; victim = w; }
        }
        ks[victim] = key;
        std::memcpy(cache_vals_.data() + (base + static_cast<size_t>(victim)) * ENGRAM_ROW_BYTES, src, ENGRAM_ROW_BYTES);
        for (int j = 0; j < ways_; ++j) if (j != victim && age[j] < 254) ++age[j];
        age[victim] = 0;
    }

    EngramStoreOpts opts_{};
    bool opened_ = false;
    std::array<std::vector<int32_t>, 4> luts_;
    std::array<int64_t, 4> cv_{};
    std::array<std::array<int64_t, 3>, 4> mults_{};
    std::array<std::array<uint64_t, 16>, 4> offsets_{};
    std::array<uint64_t, 4> nrows_{};
    std::array<int, 4> fds_{-1, -1, -1, -1};
    std::array<bool, 4> direct_{false, false, false, false};
    std::array<size_t, 4> file_bytes_{0, 0, 0, 0};
    std::array<const uint8_t*, 4> mmaps_{nullptr, nullptr, nullptr, nullptr};
    std::array<std::vector<uint64_t>, 4> compact_ids_;
    std::array<std::vector<uint8_t>, 4> compact_rows_;

    ds41::engram::UringReader uring_;
    bool use_uring_ = false;
    uint8_t* pread_buf_ = nullptr;

    size_t nsets_ = 0;
    int ways_ = 16;
    std::vector<uint64_t> cache_keys_;
    std::vector<uint8_t> cache_age_;
    std::vector<uint8_t> cache_vals_;
    EngramStats stats_{};

    struct Miss { uint64_t row; size_t out_idx; uint64_t page; };
    struct PageJob { uint64_t page; size_t m0, m1; };
    std::vector<Miss> misses_tmp_;
    std::vector<PageJob> pages_tmp_;
};

} // namespace hy4
