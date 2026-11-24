// Qwen3.8-Flash-Next Layer-1 PLE NVMe Row Store + RAM Set-Associative LRU Cache (host only, no CUDA).
// Verified verbatim against transformers/models/qwen4_exp/modeling_qwen4_exp.py (Qwen4ExpTextNGramEmbedding)
// and kindlingai/exlr8 FORMAT v4 Record 9 (Row trellis):
//   - Raw BF16 table:    <release>/ple/exlr8-ple-L001.bin (contiguous ~320M rows x 160 BF16 = 320 B/row, ~95.4 GiB)
//   - Row-trellis table: <release>/ple/exlr8-ple-L001.{k5,k4p5,k4}.safetensors or rows.safetensors (exlr8 Record 9:
//                        `rows.pmcgx` or `rows.pmcgx-f16` U8 [R, (C/256)*32K], `suh` F16 [C], `svh` F16 [R],
//                        `record` I64 [8] = [9, R, C, K_rec, 1, 1, 0, 0]; K_rec = K or -2K for half widths)
//   - N-gram hash per token t with 2-token prior context (initialized to eos_token_id = 248044):
//       _shift_right_ignore_eos: resets segment at any eos_token_id; out-of-segment shifts become eos_token_id.
//       ngram=2 (heads 0..7):  mixed = (s0 * mult[0]) ^ (s1 * mult[1])
//       ngram=3 (heads 8..15): mixed = (s0 * mult[0]) ^ (s1 * mult[1]) ^ (s2 * mult[2])
//       row_id[h] = ((mixed % vocab_size[h]) + vocab_size[h]) % vocab_size[h] + offset[h]   (signed Python/PyTorch remainder!)
//   - I/O: O_DIRECT + io_uring (reusing engram/uring_reader.h) with automatic buffered pread fallback on NFS/tmpfs,
//     in-batch deduplication and LBA sorting, fronted by a 16-way set-associative true-LRU RAM cache (or direct mmap
//     when a row-trellis table fits within cache_bytes).
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
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
#include <thread>
#include <unistd.h>
#include <vector>
#include "uring_reader.h"
#include "../model_geom_qw38.h"
#include "../runtime/json.h"

namespace qw38 {

constexpr size_t PLE_ROW_BYTES = size_t(geom::PLE_HEAD_DIM) * 2; // 320 bytes (160 BF16)
constexpr int64_t PLE_RECORD_ROW_TRELLIS = 9;

struct PleStats {
    uint64_t req = 0, hits = 0, misses = 0, reads = 0;
    double io_us = 0.0, total_us = 0.0;
};

struct PleStoreOpts {
    size_t cache_bytes = 4ull << 30;  // 4 GiB RAM LRU cache (~13.4M raw rows or ~33M K4 row-trellis rows)
    int ways = 16;                    // 1..16
    int io_uring = 128;               // >0: io_uring QD; 0: pread
    bool direct = true;               // O_DIRECT when filesystem allows, else automatic fallback
    bool gguf_ram = false;            // copy GGUF IQ4_NL table into anonymous THP RAM instead of file mmap
    bool gguf_no_warm = false;        // skip 4-thread sequential warm-up pass on file-backed GGUF mmap
};

// Metadata for an exlr8 FORMAT v4 record-9 row-trellis lookup table.
struct PleRowTrellisInfo {
    uint64_t nr = 0;         // total rows
    int nc = 0;              // stored columns per row (multiple of 256)
    int n_tiles = 0;         // nc / 256
    int k_bits = 0;          // new stream bits per pair state (8=K4, 9=K4.5, 10=K5, 1..12)
    int tile_bytes = 0;      // 16 * k_bits (= 32 * K)
    size_t row_bytes = 0;    // n_tiles * tile_bytes
    bool e4m3 = true;        // true = pmcgx (E4M3 rounded), false = pmcgx-f16 (unrounded fp16)
    uint64_t rows_off = 0;   // byte offset of rows.<cb> tensor inside the safetensors file
    uint64_t rows_bytes = 0; // byte size of rows.<cb> tensor
    uint64_t suh_off = 0;    // byte offset of suh F16 [nc]
    uint64_t svh_off = 0;    // byte offset of svh F16 [nr]
    std::string codebook;    // "pmcgx" or "pmcgx-f16"
    std::string width_name;  // e.g. "k4", "k4p5", "k5"
};

// Host FP16 <-> FP32 helpers (no CUDA dependency).
inline float ple_fp16_to_f32(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            e = 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            f = s | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline uint16_t ple_f32_to_fp16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    const uint16_t sign = uint16_t((u >> 16) & 0x8000u);
    const uint32_t exp = (u >> 23) & 0xFFu;
    const uint32_t mant = u & 0x7FFFFFu;
    if (exp == 0xFFu) {
        return mant ? uint16_t(sign | 0x7E00u | (mant >> 13)) : uint16_t(sign | 0x7C00u);
    }
    const int e = int(exp) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return sign;
        const uint32_t m = mant | 0x800000u;
        const int shift = 14 - e;
        const uint32_t half_ulp = 1u << (shift - 1);
        const uint32_t r = (m >> shift) + (((m & ((1u << shift) - 1)) > half_ulp ||
                                            ((m & ((1u << shift) - 1)) == half_ulp && ((m >> shift) & 1u))) ? 1u : 0u);
        return uint16_t(sign | r);
    }
    const uint32_t r = (mant >> 13) + (((mant & 0x1FFFu) > 0x1000u ||
                                        ((mant & 0x1FFFu) == 0x1000u && ((mant >> 13) & 1u))) ? 1u : 0u);
    if (r >= 0x400u) return uint16_t(sign | ((e + 1) << 10));
    return uint16_t(sign | (e << 10) | r);
}

inline uint16_t ple_f32_to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    const uint32_t lsb = (u >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    return uint16_t((u + rounding_bias) >> 16);
}

// Saturating RN-even FP16 -> E4M3FN -> FP16 bit pattern (matches PTX cvt.rn.satfinite.e4m3x2.f16x2 + back).
inline uint16_t ple_round_fp16_e4m3(uint16_t h) {
    const float v = ple_fp16_to_f32(h);
    const uint16_t sign = h & 0x8000u;
    const float av = std::fabs(v);
    if (av >= 448.0f) return uint16_t(sign | 0x5F00u); // +/- 448.0
    if (av == 0.0f) return sign;
    float q;
    if (av < (1.0f / 64.0f)) {
        const float r = std::nearbyint(av * 512.0f);
        q = r / 512.0f;
    } else {
        int exp = 0;
        const float frac = std::frexp(av, &exp);
        const float m = std::nearbyint(frac * 16.0f);
        q = std::ldexp(m / 16.0f, exp);
        if (q > 448.0f) q = 448.0f;
    }
    return ple_f32_to_fp16(sign ? -q : q);
}

// Exact host pmcgx / pmcgx-f16 codebook hash for one 20-bit state -> pair of FP16 bit patterns (weight 2p, 2p+1).
inline void ple_pmcgx_cpu(uint32_t state, bool e4m3, uint16_t& h0, uint16_t& h1) {
    const uint32_t w = state & 0xFFFFFu;
    const uint32_t x1 = ((w * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const uint32_t x2 = ((w * 0xBCA14C51u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const float v0 = ple_fp16_to_f32(uint16_t(x1 & 0xFFFFu)) + ple_fp16_to_f32(uint16_t(x2 >> 16));
    const float v1 = ple_fp16_to_f32(uint16_t(x1 >> 16)) + ple_fp16_to_f32(uint16_t(x2 & 0xFFFFu));
    h0 = ple_f32_to_fp16(v0);
    h1 = ple_f32_to_fp16(v1);
    if (e4m3) {
        h0 = ple_round_fp16_e4m3(h0);
        h1 = ple_round_fp16_e4m3(h1);
    }
}

// Decode one 256-value pmcgx tile in stream order (0..255, no tile_perm) to float[256].
inline void ple_decode_tile_stream_cpu(const uint8_t* tile, int k_bits, bool e4m3, float out[256]) {
    const int nw32 = 4 * k_bits;
    const int nb = 128 * k_bits;
    const uint32_t* wptr = reinterpret_cast<const uint32_t*>(tile);
    for (int p = 0; p < 128; ++p) {
        int start = (p + 1) * k_bits - 20;
        if (start < 0) start += nb;
        const int w0 = start >> 5;
        const int w1 = (w0 + 1 < nw32) ? (w0 + 1) : 0;
        const int off = start & 31;
        const uint64_t m = (uint64_t(wptr[w0]) << 32) | uint64_t(wptr[w1]);
        const uint32_t st = uint32_t(m >> (44 - off)) & 0xFFFFFu;
        uint16_t h0, h1;
        ple_pmcgx_cpu(st, e4m3, h0, h1);
        out[2 * p] = ple_fp16_to_f32(h0);
        out[2 * p + 1] = ple_fp16_to_f32(h1);
    }
}

inline void ple_fwht128_cpu(float* x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += len * 2) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    constexpr float inv_sqrt128 = 0.08838834764831844f;
    for (int i = 0; i < 128; ++i) x[i] *= inv_sqrt128;
}

// Host reference decode for exlr8 Record 9 row-trellis: W[r, c] = suh[c] * (H128 v_r)[c] * svh[r] -> BF16 bits.
inline void ple_rowtrellis_decode_cpu(const uint8_t* codes, const uint16_t* svh, const uint16_t* suh,
                                      uint16_t* out_bf16, int n_rows, int nc, int out_cols, int k_bits, bool e4m3) {
    const int n_tiles = nc / 256;
    const int tile_bytes = 16 * k_bits;
    for (int r = 0; r < n_rows; ++r) {
        const float sv = ple_fp16_to_f32(svh[r]);
        for (int t = 0; t < n_tiles; ++t) {
            float v[256];
            ple_decode_tile_stream_cpu(codes + (size_t(r) * n_tiles + size_t(t)) * size_t(tile_bytes), k_bits, e4m3, v);
            ple_fwht128_cpu(v);
            ple_fwht128_cpu(v + 128);
            for (int c = 0; c < 256; ++c) {
                const int col = t * 256 + c;
                if (col < out_cols) {
                    out_bf16[size_t(r) * size_t(out_cols) + size_t(col)] =
                        ple_f32_to_bf16(v[c] * ple_fp16_to_f32(suh[col]) * sv);
                }
            }
        }
    }
}

// Resolve the PLE table path under `release_dir`:
//   1. If `override_path` is non-empty, use it (or `<release_dir>/ple/<override_path>` if relative and not found directly).
//   2. Else prefer `<release_dir>/ple/exlr8-ple-L001.bin` when present (backwards-compatible default).
//   3. Else auto-detect an exlr8 row-trellis safetensors under `<release_dir>/ple/`.
inline std::string resolve_ple_path(const std::string& release_dir, const std::string& override_path = "") {
    auto file_exists = [](const std::string& p) -> bool {
        struct stat st{};
        return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
    };
    if (!override_path.empty()) {
        if (file_exists(override_path) || (!override_path.empty() && override_path[0] == '/'))
            return override_path;
        const std::string under_ple = release_dir + "/ple/" + override_path;
        if (file_exists(under_ple)) return under_ple;
        return override_path;
    }
    const std::string raw_bin = release_dir + "/ple/exlr8-ple-L001.bin";
    if (file_exists(raw_bin)) return raw_bin;
    const char* cands[] = {
        "exlr8-ple-L001.k5.safetensors",
        "exlr8-ple-L001.k4p5.safetensors",
        "exlr8-ple-L001.k4.safetensors",
        "exlr8-ple-L001.safetensors",
        "rows.safetensors",
    };
    for (const char* name : cands) {
        const std::string p = release_dir + "/ple/" + name;
        if (file_exists(p)) return p;
    }
    return raw_bin;
}

// 16-way set-associative true-LRU cache for fixed-size PLE rows (320 B raw BF16 or 128/144/160 B row-trellis).
class PleRowCache {
public:
    ~PleRowCache() {
        if (slots_) ::munmap(slots_, nsets_ * size_t(ways_) * row_bytes_);
    }
    bool init(size_t bytes, int ways, size_t row_bytes, std::string* err) {
        if (ways < 1 || ways > 16) { if (err) *err = "ple cache ways must be 1..16"; return false; }
        if (row_bytes == 0) { if (err) *err = "ple cache row_bytes must be > 0"; return false; }
        const size_t nslots = bytes / row_bytes;
        nsets_ = nslots / size_t(ways);
        if (!nsets_) { if (err) *err = "ple cache smaller than one set"; return false; }
        ways_ = ways;
        row_bytes_ = row_bytes;
        void* m = ::mmap(nullptr, nsets_ * size_t(ways_) * row_bytes_,
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
    bool init(size_t bytes, int ways, std::string* err) {
        return init(bytes, ways, PLE_ROW_BYTES, err);
    }
    const uint8_t* find(uint64_t row_id) {
        if (!nsets_) return nullptr;
        const uint64_t tag = row_id + 1;
        const size_t s = set_of(tag);
        const uint64_t* t = &tags_[s * size_t(ways_)];
        for (int w = 0; w < ways_; ++w) {
            if (t[w] == tag) {
                touch(s, w);
                return slots_ + (s * size_t(ways_) + size_t(w)) * row_bytes_;
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
        std::memcpy(slots_ + (s * size_t(ways_) + size_t(v)) * row_bytes_, row, row_bytes_);
        touch(s, v);
    }
    size_t capacity_rows() const { return nsets_ * size_t(ways_); }
    size_t row_bytes() const { return row_bytes_; }
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
    size_t row_bytes_ = PLE_ROW_BYTES;
    int ways_ = 0;
    std::vector<uint64_t> tags_;
    std::vector<uint64_t> order_;
};

class PleStore {
public:
    ~PleStore() {
        stop_pool();
        uring_.reset();
        if (rows_mmap_) ::munmap(rows_mmap_, rows_mmap_len_);
        if (svh_mmap_) ::munmap(svh_mmap_, svh_mmap_len_);
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

    bool open(const std::string& path, const PleStoreOpts& opts, std::string* err) {
        opts_ = opts;
        const bool is_st = path.size() >= 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0;
        if (is_st) return open_row_trellis(path, opts, err);
        const bool is_gguf = path.size() >= 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
        if (is_gguf) return open_gguf_iq4nl(path, opts, err);
        return open_raw_bin(path, opts, err);
    }

    bool is_open() const { return fd_ >= 0; }
    bool is_direct() const { return direct_; }
    bool has_uring() const { return uring_ != nullptr; }
    bool is_row_trellis() const { return row_trellis_; }
    bool is_gguf_iq4nl() const { return gguf_iq4nl_; }
    bool is_resident() const { return resident_; }
    uint64_t total_rows() const { return total_rows_; }
    size_t row_bytes() const { return row_bytes_; }
    const PleRowTrellisInfo& trellis_info() const { return rt_info_; }
    const std::vector<uint16_t>& suh() const { return suh_; }

    // Exact match to Qwen4ExpTextNGramEmbedding.forward() for one token sequence with 2-token prior context.
    // Updates `ctx[2]` in place when `update_ctx` is true. Writes `n * 16` global row IDs into `row_ids`.
    void compute_ids(const int* toks, int n, int32_t ctx[2], bool update_ctx, uint64_t* row_ids) const {
        const size_t L = static_cast<size_t>(n + 2);
        std::vector<int64_t> thist(L);
        thist[0] = ctx[0];
        thist[1] = ctx[1];
        for (int i = 0; i < n; ++i) thist[static_cast<size_t>(i + 2)] = toks[i];

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

    // Fetch `n_ids` raw rows (320 B/row for .bin, or `row_bytes_` B/row for row-trellis) into contiguous `dst`. Thread-safe.
    bool fetch_rows(const uint64_t* ids, size_t n_ids, uint8_t* dst, PleStats* st_out = nullptr) {
        return fetch_rows_impl(ids, n_ids, dst, nullptr, st_out);
    }

    // Warm `PleRowCache` in RAM for `n_ids` rows without copying to a destination buffer. Thread-safe.
    bool prefetch_rows(const uint64_t* ids, size_t n_ids, PleStats* st_out = nullptr) {
        return fetch_rows_impl(ids, n_ids, nullptr, nullptr, st_out);
    }

    // Fetch `n_ids` compressed row-trellis rows into `dst_codes` (`n_ids * row_bytes_`) and their FP16 row scales into
    // `dst_svh` (`n_ids` uint16_t). Thread-safe.
    bool fetch_trellis_rows(const uint64_t* ids, size_t n_ids, uint8_t* dst_codes, uint16_t* dst_svh,
                            PleStats* st_out = nullptr) {
        if (!row_trellis_) return false;
        return fetch_rows_impl(ids, n_ids, dst_codes, dst_svh, st_out);
    }

private:
    static constexpr size_t PLE_SPAN_MAX = 65536;
    struct Miss { uint64_t id; size_t dst_idx; };
    struct UniqMiss { uint64_t id; size_t first; size_t count; };
    struct IoSpan { uint64_t off; size_t len; uint64_t need_end; size_t first_u; size_t count_u; };

    static bool has_ci_substr(const std::string& hay, const std::string& needle) {
        auto to_low = [](std::string s) {
            for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        return to_low(hay).find(to_low(needle)) != std::string::npos;
    }

    bool open_fd_and_io(const std::string& path, const PleStoreOpts& opts, bool need_cache, std::string* err) {
        int flags = O_RDONLY | O_CLOEXEC;
        bool direct_ok = false;
#ifdef O_DIRECT
        if (opts.direct && !resident_) {
            fd_ = ::open(path.c_str(), flags | O_DIRECT);
            if (fd_ >= 0) direct_ok = true;
        }
#endif
        if (fd_ < 0) fd_ = ::open(path.c_str(), flags);
        if (fd_ < 0) {
            if (err) *err = "ple open " + path + ": " + std::strerror(errno);
            return false;
        }
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            if (err) *err = "ple fstat " + path + ": " + std::strerror(errno);
            return false;
        }
        file_bytes_ = uint64_t(st.st_size);
        direct_ = direct_ok;
        align_ = 4096;

        if (!resident_) {
            void* probe = nullptr;
            if (::posix_memalign(&probe, align_, 8192) != 0) {
                if (err) *err = "ple posix_memalign failed";
                return false;
            }
            if (direct_ && file_bytes_ >= align_) {
                if (::pread(fd_, probe, align_, 0) < 0) {
                    ::close(fd_);
                    fd_ = ::open(path.c_str(), flags);
                    direct_ = false;
                }
            }
            ::free(probe);

            if (need_cache && opts.cache_bytes >= row_bytes_ * size_t(opts.ways)) {
                if (!cache_.init(opts.cache_bytes, opts.ways, row_bytes_, err)) return false;
            }
            const size_t io_slot_bytes = std::max<size_t>(8192, ((row_bytes_ + 2 * align_ - 1) / align_) * align_);
            if (opts.io_uring > 0) {
                auto u = std::make_unique<ds41::engram::UringReader>();
                if (u->start(unsigned(opts.io_uring), unsigned(io_slot_bytes), align_) == 0) {
                    uring_ = std::move(u);
                    iobuf_slots_ = opts.io_uring;
                }
            }
            if (!uring_) start_pool(64);
        }
        return true;
    }

    bool open_raw_bin(const std::string& bin_path, const PleStoreOpts& opts, std::string* err) {
        row_trellis_ = false;
        gguf_iq4nl_ = false;
        resident_ = false;
        row_bytes_ = PLE_ROW_BYTES;
        rows_base_off_ = 0;
        if (!open_fd_and_io(bin_path, opts, true, err)) return false;
        total_rows_ = file_bytes_ / PLE_ROW_BYTES;
        return true;
    }

    static inline void decode_iq4nl_row_bf16(const uint8_t* src90, uint16_t* dst160) {
        static constexpr int8_t KVAL[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };
        for (int b = 0; b < 5; ++b) {
            uint16_t dh = 0;
            std::memcpy(&dh, src90, 2);
            const float d = ple_fp16_to_f32(dh);
            const uint8_t* qs = src90 + 2;
            uint16_t* out = dst160 + b * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = qs[j];
                out[j] = ple_f32_to_bf16(d * float(KVAL[q & 0x0Fu]));
                out[j + 16] = ple_f32_to_bf16(d * float(KVAL[q >> 4]));
            }
            src90 += 18;
        }
    }

    bool open_gguf_iq4nl(const std::string& gguf_path, const PleStoreOpts& opts, std::string* err) {
        int bfd = ::open(gguf_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (bfd < 0) {
            if (err) *err = "ple gguf open " + gguf_path + ": " + std::strerror(errno);
            return false;
        }
        struct stat st{};
        if (::fstat(bfd, &st) != 0) {
            if (err) *err = "ple gguf fstat " + gguf_path + ": " + std::strerror(errno);
            ::close(bfd);
            return false;
        }
        const uint64_t fsz = uint64_t(st.st_size);
        if (fsz < 64) {
            if (err) *err = "ple gguf file too small: " + gguf_path;
            ::close(bfd);
            return false;
        }
        const size_t hdr_read = size_t(std::min<uint64_t>(fsz, 1ull << 20));
        std::vector<uint8_t> hdr(hdr_read);
        if (!pread_exact(bfd, hdr.data(), hdr_read, 0, err)) { ::close(bfd); return false; }

        size_t pos = 0;
        auto need = [&](size_t n) -> bool {
            if (pos + n > hdr.size()) {
                if (err) *err = "ple gguf header truncated";
                return false;
            }
            return true;
        };
        auto rd_u32 = [&](uint32_t& v) -> bool { if (!need(4)) return false; std::memcpy(&v, hdr.data() + pos, 4); pos += 4; return true; };
        auto rd_u64 = [&](uint64_t& v) -> bool { if (!need(8)) return false; std::memcpy(&v, hdr.data() + pos, 8); pos += 8; return true; };
        auto rd_str = [&](std::string& s) -> bool {
            uint64_t l = 0;
            if (!rd_u64(l) || !need(size_t(l))) return false;
            s.assign(reinterpret_cast<const char*>(hdr.data() + pos), size_t(l));
            pos += size_t(l);
            return true;
        };
        auto skip_val = [&](auto& self, uint32_t vt) -> bool {
            switch (vt) {
                case 0: case 1: case 7: return need(1) && ((pos += 1), true);
                case 2: case 3: return need(2) && ((pos += 2), true);
                case 4: case 5: case 6: return need(4) && ((pos += 4), true);
                case 10: case 11: case 12: return need(8) && ((pos += 8), true);
                case 8: { std::string tmp; return rd_str(tmp); }
                case 9: {
                    uint32_t et = 0; uint64_t cnt = 0;
                    if (!rd_u32(et) || !rd_u64(cnt)) return false;
                    for (uint64_t i = 0; i < cnt; ++i) if (!self(self, et)) return false;
                    return true;
                }
                default:
                    if (err) *err = "ple gguf unsupported kv type " + std::to_string(vt);
                    return false;
            }
        };

        if (std::memcmp(hdr.data(), "GGUF", 4) != 0) {
            if (err) *err = "ple gguf bad magic in " + gguf_path;
            ::close(bfd);
            return false;
        }
        pos = 4;
        uint32_t ver = 0;
        uint64_t n_tensors = 0, n_kv = 0;
        if (!rd_u32(ver) || !rd_u64(n_tensors) || !rd_u64(n_kv)) { ::close(bfd); return false; }
        uint64_t align = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key;
            uint32_t vt = 0;
            if (!rd_str(key) || !rd_u32(vt)) { ::close(bfd); return false; }
            if (key == "general.alignment" && vt == 4) {
                uint32_t a = 32;
                if (!rd_u32(a)) { ::close(bfd); return false; }
                if (a > 0) align = a;
            } else {
                if (!skip_val(skip_val, vt)) { ::close(bfd); return false; }
            }
        }

        bool found = false;
        uint64_t t_rows = 0, t_off = 0;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            std::string tname;
            uint32_t nd = 0, ttype = 0;
            uint64_t off = 0;
            if (!rd_str(tname) || !rd_u32(nd)) { ::close(bfd); return false; }
            std::vector<uint64_t> dims(nd);
            for (uint32_t d = 0; d < nd; ++d) if (!rd_u64(dims[d])) { ::close(bfd); return false; }
            if (!rd_u32(ttype) || !rd_u64(off)) { ::close(bfd); return false; }
            if (tname == "per_layer_token_embd.weight") {
                if (nd != 2 || dims[0] != geom::PLE_HEAD_DIM || ttype != 20) {
                    if (err) *err = "ple gguf unexpected per_layer_token_embd.weight shape/type (need [160, R] IQ4_NL=20)";
                    ::close(bfd);
                    return false;
                }
                found = true;
                t_rows = dims[1];
                t_off = off;
            }
        }
        if (!found) {
            if (err) *err = "ple gguf missing per_layer_token_embd.weight in " + gguf_path;
            ::close(bfd);
            return false;
        }
        const uint64_t data_base = ((uint64_t(pos) + align - 1) & ~(align - 1)) + t_off;
        const uint64_t need_bytes = t_rows * 90ull;
        if (data_base + need_bytes > fsz) {
            if (err) *err = "ple gguf tensor exceeds file size";
            ::close(bfd);
            return false;
        }

        if (opts.gguf_ram) {
            // Explicit anonymous THP copy (immune to page-cache reclaim).
            rows_mmap_len_ = size_t(need_bytes);
            void* rm = ::mmap(nullptr, rows_mmap_len_, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (rm == MAP_FAILED) {
                if (err) *err = std::string("ple gguf anon mmap: ") + std::strerror(errno);
                ::close(bfd);
                rows_mmap_len_ = 0;
                return false;
            }
#ifdef MADV_HUGEPAGE
            ::madvise(rm, rows_mmap_len_, MADV_HUGEPAGE);
#endif
            const unsigned nt = 4;
            const uint64_t per_t = (need_bytes + nt - 1) / nt;
            std::atomic<int> io_err{0};
            std::vector<std::thread> thrs;
            uint8_t* dst_base = static_cast<uint8_t*>(rm);
            for (unsigned t = 0; t < nt; ++t) {
                const uint64_t o0 = uint64_t(t) * per_t;
                const uint64_t o1 = std::min<uint64_t>(need_bytes, o0 + per_t);
                if (o0 >= o1) continue;
                thrs.emplace_back([&, o0, o1]() {
                    constexpr size_t CH = 8ull << 20;
                    uint64_t cur = o0;
                    while (cur < o1 && io_err.load(std::memory_order_relaxed) == 0) {
                        const size_t n = size_t(std::min<uint64_t>(CH, o1 - cur));
                        const ssize_t r = ::pread(bfd, dst_base + cur, n, off_t(data_base + cur));
                        if (r <= 0) { io_err.store(1, std::memory_order_relaxed); break; }
                        cur += uint64_t(r);
                    }
                });
            }
            for (auto& th : thrs) th.join();
            ::close(bfd);
            if (io_err.load() != 0) {
                ::munmap(rm, rows_mmap_len_);
                rows_mmap_len_ = 0;
                if (err) *err = "ple gguf pread failed for " + gguf_path;
                return false;
            }
            rows_mmap_ = rm;
            rows_ptr_ = static_cast<const uint8_t*>(rm);
        } else {
            // File-backed mmap + 4-thread sequential warm-up: stays warm in Linux page cache across runs,
            // 100% reclaimable under cgroup memory pressure (never OOMs), and zero-fault once warm.
            const size_t pg = size_t(::sysconf(_SC_PAGESIZE));
            const uint64_t map_off = data_base & ~(uint64_t(pg) - 1);
            const size_t delta = size_t(data_base - map_off);
            rows_mmap_len_ = delta + size_t(need_bytes);
            void* rm = ::mmap(nullptr, rows_mmap_len_, PROT_READ, MAP_PRIVATE, bfd, off_t(map_off));
            ::close(bfd);
            if (rm == MAP_FAILED) {
                if (err) *err = std::string("ple gguf mmap: ") + std::strerror(errno);
                rows_mmap_len_ = 0;
                return false;
            }
            if (!opts.gguf_no_warm) {
                ::madvise(rm, rows_mmap_len_, MADV_NORMAL);
                const size_t n_pages = (rows_mmap_len_ + pg - 1) / pg;
                const unsigned nt = 4;
                std::atomic<uint64_t> sink{0};
                std::vector<std::thread> thrs;
                const size_t chunk = (n_pages + nt - 1) / nt;
                const uint8_t* base_p = static_cast<const uint8_t*>(rm);
                for (unsigned t = 0; t < nt; ++t) {
                    const size_t p0 = size_t(t) * chunk;
                    const size_t p1 = std::min(n_pages, p0 + chunk);
                    if (p0 >= p1) continue;
                    thrs.emplace_back([&, p0, p1]() {
                        uint64_t acc = 0;
                        for (size_t p = p0; p < p1; ++p) acc += base_p[p * pg];
                        sink.fetch_add(acc, std::memory_order_relaxed);
                    });
                }
                for (auto& th : thrs) th.join();
                (void)sink.load(std::memory_order_relaxed);
            }
            ::madvise(rm, rows_mmap_len_, MADV_RANDOM);
            rows_mmap_ = rm;
            rows_ptr_ = static_cast<const uint8_t*>(rm) + delta;
        }

        row_trellis_ = false;
        gguf_iq4nl_ = true;
        resident_ = true;
        row_bytes_ = PLE_ROW_BYTES;
        rows_base_off_ = data_base;
        total_rows_ = t_rows;
        if (!open_fd_and_io(gguf_path, opts, false, err)) return false;
        return true;
    }

    bool pread_exact(int fd, void* buf, size_t n, uint64_t off, std::string* err) {
        uint8_t* p = static_cast<uint8_t*>(buf);
        size_t got = 0;
        while (got < n) {
            const ssize_t r = ::pread(fd, p + got, n - got, off_t(off + got));
            if (r <= 0) {
                if (err) *err = "ple pread failed: " + std::string(r < 0 ? std::strerror(errno) : "unexpected EOF");
                return false;
            }
            got += size_t(r);
        }
        return true;
    }

    bool open_row_trellis(const std::string& st_path, const PleStoreOpts& opts, std::string* err) {
        // 1. Open buffered temporarily to parse safetensors header and small metadata tensors
        int bfd = ::open(st_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (bfd < 0) {
            if (err) *err = "ple row-trellis open " + st_path + ": " + std::strerror(errno);
            return false;
        }
        struct stat st{};
        if (::fstat(bfd, &st) != 0) {
            if (err) *err = "ple row-trellis fstat " + st_path + ": " + std::strerror(errno);
            ::close(bfd);
            return false;
        }
        const uint64_t fsz = uint64_t(st.st_size);
        if (fsz < 16) {
            if (err) *err = "ple row-trellis file too small: " + st_path;
            ::close(bfd);
            return false;
        }
        uint64_t hdr_len = 0;
        if (!pread_exact(bfd, &hdr_len, 8, 0, err)) { ::close(bfd); return false; }
        if (hdr_len == 0 || hdr_len > (64ull << 20) || 8 + hdr_len > fsz) {
            if (err) *err = "ple row-trellis invalid safetensors header length";
            ::close(bfd);
            return false;
        }
        std::string hdr_json(size_t(hdr_len), '\0');
        if (!pread_exact(bfd, hdr_json.data(), size_t(hdr_len), 8, err)) { ::close(bfd); return false; }
        const uint64_t data_start = 8 + hdr_len;

        g6v::Json hdr;
        try {
            hdr = g6v::Json::parse(hdr_json);
        } catch (const std::exception& e) {
            if (err) *err = std::string("ple row-trellis header json: ") + e.what();
            ::close(bfd);
            return false;
        }
        if (hdr.t != g6v::Json::Obj) {
            if (err) *err = "ple row-trellis header is not a JSON object";
            ::close(bfd);
            return false;
        }

        // Locate record tensor: either bare "record" (exlr8enc rowtrellis) or "<prefix>.record" (exlr8-v4).
        std::string prefix;
        std::string rec_key;
        for (const auto& kv : hdr.o) {
            if (kv.first == "record") {
                prefix = "";
                rec_key = "record";
                break;
            }
            if (kv.first.size() > 7 && kv.first.compare(kv.first.size() - 7, 7, ".record") == 0) {
                prefix = kv.first.substr(0, kv.first.size() - 6); // includes trailing '.'
                rec_key = kv.first;
                break;
            }
        }
        if (rec_key.empty()) {
            if (err) *err = "ple row-trellis: missing 'record' tensor in " + st_path;
            ::close(bfd);
            return false;
        }

        auto parse_tensor = [&](const std::string& key, std::string& dtype, std::vector<int64_t>& shape,
                                uint64_t& off0, uint64_t& nbytes) -> bool {
            const g6v::Json* t = hdr.find(key);
            if (!t || t->t != g6v::Json::Obj) {
                if (err) *err = "ple row-trellis: missing tensor '" + key + "'";
                return false;
            }
            try {
                dtype = t->at("dtype").str();
                const g6v::Json& sh = t->at("shape");
                shape.clear();
                for (size_t i = 0; i < sh.size(); ++i) shape.push_back(sh.at(i).i64());
                const g6v::Json& offs = t->at("data_offsets");
                const int64_t a = offs.at(0).i64();
                const int64_t b = offs.at(1).i64();
                if (a < 0 || b < a || data_start + uint64_t(b) > fsz) {
                    if (err) *err = "ple row-trellis: invalid data_offsets for '" + key + "'";
                    return false;
                }
                off0 = data_start + uint64_t(a);
                nbytes = uint64_t(b - a);
            } catch (const std::exception& e) {
                if (err) *err = std::string("ple row-trellis tensor '") + key + "': " + e.what();
                return false;
            }
            return true;
        };

        // Read and validate I64[8] record
        std::string rec_dtype;
        std::vector<int64_t> rec_shape;
        uint64_t rec_off = 0, rec_bytes = 0;
        if (!parse_tensor(rec_key, rec_dtype, rec_shape, rec_off, rec_bytes)) { ::close(bfd); return false; }
        if (rec_dtype != "I64" || rec_shape.size() != 1 || rec_shape[0] != 8 || rec_bytes != 64) {
            if (err) *err = "ple row-trellis: '" + rec_key + "' must be I64[8]";
            ::close(bfd);
            return false;
        }
        int64_t rec[8] = {};
        if (!pread_exact(bfd, rec, 64, rec_off, err)) { ::close(bfd); return false; }
        if (rec[0] != PLE_RECORD_ROW_TRELLIS) {
            if (err) *err = "ple row-trellis: record format is " + std::to_string(rec[0]) + ", expected 9";
            ::close(bfd);
            return false;
        }
        const int64_t nr = rec[1];
        const int64_t nc = rec[2];
        const int64_t k_rec = rec[3];
        if (nr <= 0 || nc <= 0 || (nc % 256) != 0) {
            if (err) *err = "ple row-trellis: invalid shape (" + std::to_string(nr) + ", " + std::to_string(nc) + ")";
            ::close(bfd);
            return false;
        }

        // Find rows.<cb> tensor and validate codebook
        std::string rows_key;
        std::string cb;
        const std::string rows_pref = prefix + "rows.";
        for (const auto& kv : hdr.o) {
            if (kv.first.size() > rows_pref.size() && kv.first.compare(0, rows_pref.size(), rows_pref) == 0) {
                if (!rows_key.empty()) {
                    if (err) *err = "ple row-trellis: multiple rows.<codebook> tensors in " + st_path;
                    ::close(bfd);
                    return false;
                }
                rows_key = kv.first;
                cb = kv.first.substr(rows_pref.size());
            }
        }
        if (rows_key.empty()) {
            if (err) *err = "ple row-trellis: missing '" + rows_pref + "<codebook>' tensor";
            ::close(bfd);
            return false;
        }
        bool e4m3 = true;
        if (cb == "pmcgx") e4m3 = true;
        else if (cb == "pmcgx-f16") e4m3 = false;
        else {
            if (err) *err = "ple row-trellis: unsupported codebook '" + cb + "' (expected pmcgx or pmcgx-f16)";
            ::close(bfd);
            return false;
        }

        // If __metadata__.format is exlr8-v4, require `codebooks` metadata declaring cb and its multipliers.
        if (const g6v::Json* meta = hdr.find("__metadata__"); meta && meta->t == g6v::Json::Obj) {
            const g6v::Json* fmt_j = meta->find("format");
            if (fmt_j && fmt_j->t == g6v::Json::Str && fmt_j->s == "exlr8-v4") {
                const g6v::Json* cbs_j = meta->find("codebooks");
                if (!cbs_j || cbs_j->t != g6v::Json::Str ||
                    cbs_j->s.find(cb) == std::string::npos ||
                    !has_ci_substr(cbs_j->s, "0xCBAC1FED") ||
                    !has_ci_substr(cbs_j->s, "0xBCA14C51")) {
                    if (err) *err = "ple row-trellis: exlr8-v4 codebooks metadata missing " + cb + " multipliers";
                    ::close(bfd);
                    return false;
                }
            }
        }

        // Validate K: positive integer K -> 2*K pair bits; negative odd -2K -> -k_rec pair bits.
        int k_bits = 0;
        if (k_rec > 0) {
            k_bits = int(k_rec * 2);
        } else if (k_rec < 0 && ((-k_rec) & 1) == 1) {
            k_bits = int(-k_rec);
        } else {
            if (err) *err = "ple row-trellis: invalid K in record field 3: " + std::to_string(k_rec);
            ::close(bfd);
            return false;
        }
        const int max_k_bits = e4m3 ? 10 : 12; // pmcgx: up to K5 (10 bits); pmcgx-f16: up to K6 (12 bits)
        if (k_bits < 1 || k_bits > max_k_bits) {
            if (err) *err = "ple row-trellis: unsupported width for " + cb + " (pair bits=" + std::to_string(k_bits) + ")";
            ::close(bfd);
            return false;
        }

        const int n_tiles = int(nc / 256);
        const int tile_bytes = 16 * k_bits;
        const size_t rbytes = size_t(n_tiles) * size_t(tile_bytes);

        std::string r_dt, suh_dt, svh_dt;
        std::vector<int64_t> r_sh, suh_sh, svh_sh;
        uint64_t r_off = 0, r_nbytes = 0, suh_off = 0, suh_nbytes = 0, svh_off = 0, svh_nbytes = 0;
        if (!parse_tensor(rows_key, r_dt, r_sh, r_off, r_nbytes) ||
            !parse_tensor(prefix + "suh", suh_dt, suh_sh, suh_off, suh_nbytes) ||
            !parse_tensor(prefix + "svh", svh_dt, svh_sh, svh_off, svh_nbytes)) {
            ::close(bfd);
            return false;
        }
        if (r_dt != "U8" || r_sh.size() != 2 || r_sh[0] != nr || uint64_t(r_sh[1]) != uint64_t(rbytes) ||
            r_nbytes != uint64_t(nr) * uint64_t(rbytes)) {
            if (err) *err = "ple row-trellis: rows tensor dtype/shape mismatch";
            ::close(bfd);
            return false;
        }
        if (suh_dt != "F16" || suh_sh.size() != 1 || suh_sh[0] != nc || suh_nbytes != uint64_t(nc) * 2) {
            if (err) *err = "ple row-trellis: suh tensor must be F16[" + std::to_string(nc) + "]";
            ::close(bfd);
            return false;
        }
        if (svh_dt != "F16" || svh_sh.size() != 1 || svh_sh[0] != nr || svh_nbytes != uint64_t(nr) * 2) {
            if (err) *err = "ple row-trellis: svh tensor must be F16[" + std::to_string(nr) + "]";
            ::close(bfd);
            return false;
        }

        // Load small suh [nc] F16 into host vector for GPU upload.
        suh_.resize(size_t(nc));
        if (!pread_exact(bfd, suh_.data(), size_t(suh_nbytes), suh_off, err)) { ::close(bfd); return false; }

        // Map svh [nr] F16 read-only in host virtual memory (O(1) lookup, zero extra NVMe IOPS).
        const size_t pg = size_t(::sysconf(_SC_PAGESIZE));
        const uint64_t svh_map_off = svh_off & ~(uint64_t(pg) - 1);
        const size_t svh_delta = size_t(svh_off - svh_map_off);
        svh_mmap_len_ = svh_delta + size_t(svh_nbytes);
        void* sm = ::mmap(nullptr, svh_mmap_len_, PROT_READ, MAP_PRIVATE, bfd, off_t(svh_map_off));
        if (sm == MAP_FAILED) {
            if (err) *err = std::string("ple row-trellis mmap svh: ") + std::strerror(errno);
            ::close(bfd);
            return false;
        }
        ::madvise(sm, svh_mmap_len_, MADV_WILLNEED);
        svh_mmap_ = sm;
        svh_ptr_ = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(sm) + svh_delta);

        // If the entire compressed rows tensor fits in cache_bytes, mmap it directly (zero NVMe I/O!).
        resident_ = (opts.cache_bytes >= r_nbytes && r_nbytes > 0);
        if (resident_) {
            const uint64_t r_map_off = r_off & ~(uint64_t(pg) - 1);
            const size_t r_delta = size_t(r_off - r_map_off);
            rows_mmap_len_ = r_delta + size_t(r_nbytes);
            void* rm = ::mmap(nullptr, rows_mmap_len_, PROT_READ, MAP_PRIVATE, bfd, off_t(r_map_off));
            if (rm == MAP_FAILED) {
                resident_ = false;
                rows_mmap_len_ = 0;
            } else {
                ::madvise(rm, rows_mmap_len_, MADV_WILLNEED);
                rows_mmap_ = rm;
                rows_ptr_ = static_cast<const uint8_t*>(rm) + r_delta;
            }
        }
        ::close(bfd);

        rt_info_.nr = uint64_t(nr);
        rt_info_.nc = int(nc);
        rt_info_.n_tiles = n_tiles;
        rt_info_.k_bits = k_bits;
        rt_info_.tile_bytes = tile_bytes;
        rt_info_.row_bytes = rbytes;
        rt_info_.e4m3 = e4m3;
        rt_info_.rows_off = r_off;
        rt_info_.rows_bytes = r_nbytes;
        rt_info_.suh_off = suh_off;
        rt_info_.svh_off = svh_off;
        rt_info_.codebook = cb;
        rt_info_.width_name = (k_bits & 1) ? ("k" + std::to_string(k_bits / 2) + "p5")
                                           : ("k" + std::to_string(k_bits / 2));

        row_trellis_ = true;
        row_bytes_ = rbytes;
        rows_base_off_ = r_off;
        total_rows_ = uint64_t(nr);

        if (!open_fd_and_io(st_path, opts, !resident_, err)) return false;
        return true;
    }

    bool fetch_rows_impl(const uint64_t* ids, size_t n_ids, uint8_t* dst, uint16_t* dst_svh, PleStats* st_out) {
        const auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mu_);
        PleStats s{};
        s.req = n_ids;
        const size_t rb = row_bytes_;

        // Fast path: RAM-resident mmap for GGUF IQ4_NL PLE table (zero HDD IOPS, instant BF16 dequant!)
        if (gguf_iq4nl_ && resident_ && rows_ptr_) {
            if (dst) {
                auto work = [&](size_t i0, size_t i1) {
                    for (size_t i = i0; i < i1; ++i) {
                        const uint64_t id = ids[i];
                        uint16_t* out = reinterpret_cast<uint16_t*>(dst + i * PLE_ROW_BYTES);
                        if (id >= total_rows_) {
                            std::memset(out, 0, PLE_ROW_BYTES);
                        } else {
                            decode_iq4nl_row_bf16(rows_ptr_ + id * 90ull, out);
                        }
                    }
                };
                if (n_ids >= 4096) {
                    const unsigned nt = std::max(1u, std::min<unsigned>(8u, std::thread::hardware_concurrency()));
                    std::vector<std::thread> thrs;
                    thrs.reserve(nt - 1);
                    const size_t step = (n_ids + nt - 1) / nt;
                    for (unsigned t = 1; t < nt; ++t) {
                        const size_t a = size_t(t) * step;
                        const size_t b = std::min(n_ids, a + step);
                        if (a < b) thrs.emplace_back(work, a, b);
                    }
                    work(0, std::min(n_ids, step));
                    for (auto& th : thrs) th.join();
                } else {
                    work(0, n_ids);
                }
            }
            s.hits = n_ids;
            const auto t1 = std::chrono::steady_clock::now();
            s.total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (st_out) *st_out = s;
            return true;
        }

        // Fast path: RAM-resident mmap for row-trellis tables that fit in cache_bytes
        if (resident_ && rows_ptr_) {
            if (dst || dst_svh) {
                for (size_t i = 0; i < n_ids; ++i) {
                    const uint64_t id = ids[i];
                    if (id >= total_rows_) {
                        if (dst) std::memset(dst + i * rb, 0, rb);
                        if (dst_svh) dst_svh[i] = 0;
                        continue;
                    }
                    if (dst) std::memcpy(dst + i * rb, rows_ptr_ + id * rb, rb);
                    if (dst_svh) dst_svh[i] = svh_ptr_[id];
                    ++s.hits;
                }
            } else {
                s.hits = n_ids;
            }
            const auto t1 = std::chrono::steady_clock::now();
            s.total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (st_out) *st_out = s;
            return true;
        }

        misses_.clear();
        if (misses_.capacity() < n_ids) misses_.reserve(n_ids);

        for (size_t i = 0; i < n_ids; ++i) {
            const uint64_t id = ids[i];
            if (id >= total_rows_) {
                if (dst) std::memset(dst + i * rb, 0, rb);
                if (dst_svh) dst_svh[i] = 0;
                continue;
            }
            if (dst_svh) dst_svh[i] = svh_ptr_[id];
            if (const uint8_t* hit = cache_.find(id)) {
                if (dst) std::memcpy(dst + i * rb, hit, rb);
                ++s.hits;
            } else {
                misses_.push_back({id, i});
            }
        }
        s.misses = misses_.size();

        const auto tio0 = std::chrono::steady_clock::now();
        if (!misses_.empty()) {
            // 1. Sort misses by ascending row_id -> exact intra-chunk deduplication + LBA locality
            std::sort(misses_.begin(), misses_.end(), [](const Miss& a, const Miss& b) {
                return a.id < b.id;
            });
            uniq_.clear();
            uniq_.reserve(misses_.size());
            for (size_t i = 0; i < misses_.size();) {
                size_t j = i + 1;
                while (j < misses_.size() && misses_[j].id == misses_[i].id) ++j;
                uniq_.push_back({misses_[i].id, i, j - i});
                i = j;
            }

            // 2. Coalesce overlapping / adjacent 4K-aligned page spans
            const size_t io_slot_max = std::max<size_t>(8192, ((rb + 2 * align_ - 1) / align_) * align_);
            const size_t max_span = uring_ ? io_slot_max : std::max<size_t>(PLE_SPAN_MAX, io_slot_max);
            spans_.clear();
            spans_.reserve(uniq_.size());
            for (size_t ui = 0; ui < uniq_.size(); ++ui) {
                const uint64_t boff = rows_base_off_ + uniq_[ui].id * rb;
                const uint64_t bend = boff + rb;
                const uint64_t rlo = direct_ ? (boff & ~(uint64_t(align_) - 1)) : boff;
                const uint64_t rhi = direct_ ? ((bend + align_ - 1) & ~(uint64_t(align_) - 1)) : bend;
                if (!spans_.empty()) {
                    IoSpan& prev = spans_.back();
                    const uint64_t cur_end = prev.off + prev.len;
                    const uint64_t new_end = std::max(cur_end, rhi);
                    const uint64_t gap_ok = direct_ ? 0 : uint64_t(align_);
                    if (rlo <= cur_end + gap_ok && new_end - prev.off <= max_span) {
                        prev.len = size_t(new_end - prev.off);
                        if (bend > prev.need_end) prev.need_end = bend;
                        ++prev.count_u;
                        continue;
                    }
                }
                spans_.push_back({rlo, size_t(rhi - rlo), bend, ui, 1});
            }

            if (uring_) {
                int first_err = 0;
                // Pad O_DIRECT io_uring submissions to a multiple of 8 (dummy 4 KiB read at offset 0)
                // to avoid the ~200 us NVMe IRQ-coalescing cliff on small / non-mod-8 batches.
                const size_t n_real = spans_.size();
                const bool pad8 = direct_ && file_bytes_ >= align_ && iobuf_slots_ >= 8 && (n_real & 7u) != 0;
                const size_t n_sub = pad8 ? ((n_real + 7u) & ~size_t(7u)) : n_real;
                const int rc = uring_->run(
                    n_sub,
                    [&](size_t idx, io_uring_sqe& sqe) {
                        if (idx >= n_real) {
                            sqe.fd = fd_;
                            sqe.off = 0;
                            sqe.len = unsigned(align_);
                            return;
                        }
                        const IoSpan& sp = spans_[idx];
                        sqe.fd = fd_;
                        sqe.off = sp.off;
                        sqe.len = unsigned(sp.len);
                    },
                    [&](size_t idx, const uint8_t* buf, int res) {
                        if (idx >= n_real) return;
                        const IoSpan& sp = spans_[idx];
                        const int need = int(sp.need_end - sp.off);
                        if (res < need) {
                            if (!first_err) first_err = res < 0 ? -res : EIO;
                            return;
                        }
                        for (size_t k = 0; k < sp.count_u; ++k) {
                            const UniqMiss& u = uniq_[sp.first_u + k];
                            const uint8_t* row = buf + (rows_base_off_ + u.id * rb - sp.off);
                            if (dst) {
                                for (size_t d = 0; d < u.count; ++d) {
                                    std::memcpy(dst + misses_[u.first + d].dst_idx * rb, row, rb);
                                }
                            }
                            cache_.insert(u.id, row);
                        }
                        ++s.reads;
                    });
                if (rc != 0 || first_err != 0) {
                    std::fprintf(stderr, "# qw38 PLE io_uring error rc=%d err=%d\n", rc, first_err);
                    return false;
                }
            } else if (!pool_.empty() && spans_.size() >= 2) {
                uint8_t* pf_dst = nullptr;
                if (!dst) {
                    pf_buf_.resize(uniq_.size() * rb);
                    pf_dst = pf_buf_.data();
                }
                if (!run_pool_pread(spans_.data(), spans_.size(), uniq_.data(), misses_.data(), dst, pf_dst)) {
                    return false;
                }
                for (size_t ui = 0; ui < uniq_.size(); ++ui) {
                    const uint8_t* row = dst
                        ? (dst + misses_[uniq_[ui].first].dst_idx * rb)
                        : (pf_dst + ui * rb);
                    cache_.insert(uniq_[ui].id, row);
                }
                s.reads += spans_.size();
            } else {
                const size_t tmp_sz = std::max<size_t>(PLE_SPAN_MAX, ((rb + 2 * align_ - 1) / align_) * align_);
                void* raw = nullptr;
                if (::posix_memalign(&raw, align_, tmp_sz) != 0) return false;
                uint8_t* tmp = static_cast<uint8_t*>(raw);
                bool ok = true;
                for (const IoSpan& sp : spans_) {
                    const ssize_t got = ::pread(fd_, tmp, sp.len, off_t(sp.off));
                    if (got < ssize_t(sp.need_end - sp.off)) { ok = false; break; }
                    for (size_t k = 0; k < sp.count_u; ++k) {
                        const UniqMiss& u = uniq_[sp.first_u + k];
                        const uint8_t* row = tmp + (rows_base_off_ + u.id * rb - sp.off);
                        if (dst) {
                            for (size_t d = 0; d < u.count; ++d) {
                                std::memcpy(dst + misses_[u.first + d].dst_idx * rb, row, rb);
                            }
                        }
                        cache_.insert(u.id, row);
                    }
                    ++s.reads;
                }
                ::free(raw);
                if (!ok) return false;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        s.io_us = std::chrono::duration<double, std::micro>(t1 - tio0).count();
        s.total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (st_out) *st_out = s;
        return true;
    }

    void start_pool(int nthreads) {
        pool_stop_ = false;
        pool_gen_ = 0;
        pool_.reserve(size_t(nthreads));
        for (int i = 0; i < nthreads; ++i) {
            pool_.emplace_back([this]() { pool_worker(); });
        }
    }

    void stop_pool() {
        if (pool_.empty()) return;
        {
            std::lock_guard<std::mutex> lk(pool_mu_);
            pool_stop_ = true;
            ++pool_gen_;
        }
        pool_cv_.notify_all();
        for (auto& t : pool_) if (t.joinable()) t.join();
        pool_.clear();
    }

    void pool_worker() {
        const size_t tmp_sz = std::max<size_t>(PLE_SPAN_MAX, ((row_bytes_ + 2 * align_ - 1) / align_) * align_);
        void* raw = nullptr;
        if (::posix_memalign(&raw, align_, tmp_sz) != 0) raw = nullptr;
        uint8_t* tmp = static_cast<uint8_t*>(raw);
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pool_mu_);
                pool_cv_.wait(lk, [&]() { return pool_gen_ != seen; });
                seen = pool_gen_;
                if (pool_stop_) break;
            }
            const IoSpan* spans = pool_spans_;
            const size_t n = pool_n_;
            const UniqMiss* uniq = pool_uniq_;
            const Miss* misses = pool_misses_;
            uint8_t* dst = pool_dst_;
            uint8_t* pf_dst = pool_pf_dst_;
            const size_t rb = row_bytes_;
            for (;;) {
                const size_t idx = pool_next_.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n) break;
                if (!tmp) { pool_err_.store(1, std::memory_order_relaxed); continue; }
                const IoSpan& sp = spans[idx];
                const ssize_t got = ::pread(fd_, tmp, sp.len, off_t(sp.off));
                if (got < ssize_t(sp.need_end - sp.off)) {
                    pool_err_.store(1, std::memory_order_relaxed);
                } else {
                    for (size_t k = 0; k < sp.count_u; ++k) {
                        const size_t ui = sp.first_u + k;
                        const UniqMiss& u = uniq[ui];
                        const uint8_t* row = tmp + (rows_base_off_ + u.id * rb - sp.off);
                        if (dst) {
                            for (size_t d = 0; d < u.count; ++d) {
                                std::memcpy(dst + misses[u.first + d].dst_idx * rb, row, rb);
                            }
                        } else if (pf_dst) {
                            std::memcpy(pf_dst + ui * rb, row, rb);
                        }
                    }
                }
            }
            if (pool_rem_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(pool_mu_);
                pool_done_ = true;
                pool_done_cv_.notify_one();
            }
        }
        if (raw) ::free(raw);
    }

    bool run_pool_pread(const IoSpan* spans, size_t n, const UniqMiss* uniq, const Miss* misses,
                        uint8_t* dst, uint8_t* pf_dst) {
        pool_spans_ = spans;
        pool_n_ = n;
        pool_uniq_ = uniq;
        pool_misses_ = misses;
        pool_dst_ = dst;
        pool_pf_dst_ = pf_dst;
        pool_next_.store(0, std::memory_order_relaxed);
        pool_err_.store(0, std::memory_order_relaxed);
        pool_rem_.store(int(pool_.size()), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(pool_mu_);
            pool_done_ = false;
            ++pool_gen_;
        }
        pool_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(pool_mu_);
            pool_done_cv_.wait(lk, [&]() { return pool_done_; });
        }
        return pool_err_.load(std::memory_order_relaxed) == 0;
    }

    int fd_ = -1;
    bool direct_ = false;
    bool has_hash_ = false;
    bool row_trellis_ = false;
    bool gguf_iq4nl_ = false;
    bool resident_ = false;
    size_t align_ = 4096;
    size_t row_bytes_ = PLE_ROW_BYTES;
    uint64_t rows_base_off_ = 0;
    uint64_t file_bytes_ = 0;
    uint64_t total_rows_ = 0;
    PleStoreOpts opts_{};
    PleRowTrellisInfo rt_info_{};
    std::vector<uint16_t> suh_;
    void* svh_mmap_ = nullptr;
    size_t svh_mmap_len_ = 0;
    const uint16_t* svh_ptr_ = nullptr;
    void* rows_mmap_ = nullptr;
    size_t rows_mmap_len_ = 0;
    const uint8_t* rows_ptr_ = nullptr;
    PleRowCache cache_;
    std::unique_ptr<ds41::engram::UringReader> uring_;
    void* iobuf_ = nullptr;
    int iobuf_slots_ = 0;
    int64_t mult_[3] = {};
    int64_t offsets_[16] = {};
    int64_t vsizes_[16] = {};
    int32_t eos_tok_ = geom::TOK_EOS;
    std::mutex mu_;
    // Reusable scratch for dedup + 4K coalescing (guarded by mu_)
    std::vector<Miss> misses_;
    std::vector<UniqMiss> uniq_;
    std::vector<IoSpan> spans_;
    std::vector<uint8_t> pf_buf_;
    // Persistent pread worker pool (used when io_uring is unavailable)
    std::vector<std::thread> pool_;
    std::mutex pool_mu_;
    std::condition_variable pool_cv_, pool_done_cv_;
    uint64_t pool_gen_ = 0;
    bool pool_stop_ = false, pool_done_ = true;
    const IoSpan* pool_spans_ = nullptr;
    const UniqMiss* pool_uniq_ = nullptr;
    const Miss* pool_misses_ = nullptr;
    size_t pool_n_ = 0;
    uint8_t* pool_dst_ = nullptr;
    uint8_t* pool_pf_dst_ = nullptr;
    std::atomic<size_t> pool_next_{0};
    std::atomic<int> pool_rem_{0};
    std::atomic<int> pool_err_{0};
};

} // namespace qw38
