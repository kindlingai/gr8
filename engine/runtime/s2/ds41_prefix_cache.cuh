// DeepSeek-V4.1-Flash-1017 prefix state snapshot cache (B4 / B5 / B11 / B12).
// Stores prefix snapshots [0, p) (any p >= 1, even or odd) in lazily allocated device slabs
// bounded by GR8_DS_CACHE_MIB (default 4096 MiB per rank) with LRU eviction.
// A snapshot at position p consists of:
//   - Main model attention (40 layers): C.ring (ring_rows x 512 bf16) + C.comp + C.kvs/C.scs at ratio-2 sources 2,8,14
//   - Indexer / ratio-1 KV (Ds41IdxModel): ica[0..2] planar codes+sf for p/2 rows, ica[3] planar codes+sf for p rows,
//     ca[3] (p x 288 B), shadow (p x 512 bf16)
//   - Engram token ring (ehist): 64 int32 entries reconstructed from tokens[0..p)
//   - DSpark draft model (3 layers, when enabled): C.ring (ring_rows x 512 bf16)
// No device kernels are added or modified here; all transfers use cudaMemcpyAsync on the session stream.
#pragma once
#if G6_MODEL_DS41
#include "service.h"
#ifndef GR8_DS41_PREFIX_CACHE_STANDALONE
#include "ds41_dspark_round.cuh"
#endif
#include "../../attn/kvfmt.h"
#include "../../engram/engram_ring.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace g6s2 {

struct Ds41PrefixEntry {
    uint64_t generation = 0;
    uint32_t position = 0;
    size_t bytes = 0;
    std::vector<int> tokens;
    g6rt::CacheNamespace ns{};
    void* dev_buf = nullptr;
    int ehist[ds41::engram::kRing] = {};
};

struct Ds41PrefixHit {
    bool hit = false;
    uint64_t generation = 0;
    uint32_t position = 0;
    int lcp = 0;
    const char* why = "first";
};

class Ds41PrefixCache {
public:
    struct LastOp {
        size_t bytes = 0;
        double ms = 0.0;
    } last{};

    Ds41PrefixCache(g6rt::Model& main, Ds41Dspark* dsp, size_t budget_mib)
        : main_(main), dsp_(dsp), budget_bytes_(budget_mib << 20) { live() = this; }

    // The engine builds one cache per rank, inside serve_ds41's scope, and the stop-time report lives in main's scope.
    // A single self-registered pointer carries it there (PRL dev-proof condition 2: the hit counter must be readable).
    static Ds41PrefixCache*& live() { static Ds41PrefixCache* p = nullptr; return p; }
    // One line, the same fields every time: what a gate reads back after the prefix-cache requests.
    std::string stats_line() const {
        char b[256];
        snprintf(b, sizeof b, "entries %zu used_mib %zu captures %llu hits %llu evictions %llu restored_bytes %zu",
                 entries_.size(), used_bytes_ >> 20, (unsigned long long)captures_, (unsigned long long)hits_,
                 (unsigned long long)evictions_, restored_bytes_);
        return std::string(b);
    }

    Ds41PrefixCache(const Ds41PrefixCache&) = delete;
    Ds41PrefixCache& operator=(const Ds41PrefixCache&) = delete;

    ~Ds41PrefixCache() {
        for (auto& e : entries_) {
            if (e.dev_buf) cudaFree(e.dev_buf);
        }
    }

    size_t budget_bytes() const { return budget_bytes_; }
    size_t used_bytes() const { return used_bytes_; }
    size_t size() const { return entries_.size(); }
    uint64_t hits() const { return hits_; }
    uint64_t captures() const { return captures_; }
    uint64_t evictions() const { return evictions_; }

    uint64_t digest() const {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ uint64_t(entries_.size());
        for (const auto& e : entries_) {
            h ^= (e.generation + (uint64_t(e.position) << 32)) * 0xBF58476D1CE4E5B9ull;
            h = (h << 27) | (h >> 37);
        }
        return h;
    }

    size_t entry_bytes(uint32_t pos) const {
        using namespace g6rt;
        const size_t p = size_t(pos), p2 = p / 2;
        size_t total = 0;
        for (int i = 0; i < (int)main_.layers.size(); ++i) {
            const Ds41AttnLayer* a = ds41_attn_state(main_, i);
            if (!a) continue;
            total += size_t(a->C.ring_rows) * ds41a::HD * sizeof(__nv_bfloat16);
            if (a->C.comp) {
                total += p2 * ds41a::HD * sizeof(__nv_bfloat16);
                if (a->C.kvs) total += 2 * ds41a::HD * sizeof(float);
                if (a->C.scs) total += 2 * ds41a::HD * sizeof(float);
            }
        }
        if (const Ds41IdxModel* S = ds41_idx_detail::find(main_, false)) {
            for (int k = 0; k < 4; ++k) {
                const size_t nrows = (k < 3) ? p2 : p;
                if (S->ica[k]) total += nrows * DS41_ICA_ROW;
                if (S->ca[k]) total += nrows * DS41_CA_ROW;
            }
            if (S->shadow) total += p * 512 * sizeof(__nv_bfloat16);
        }
        if (dsp_) {
            for (int j = 0; j < (int)dsp_->draft.layers.size(); ++j) {
                const Ds41AttnLayer* da = ds41_attn_state(dsp_->draft, j);
                if (da && da->C.ring) total += size_t(da->C.ring_rows) * ds41a::HD * sizeof(__nv_bfloat16);
            }
        }
        return total;
    }

    const Ds41PrefixEntry* find(uint64_t gen) const {
        for (const auto& e : entries_) {
            if (e.generation == gen) return &e;
        }
        return nullptr;
    }

    // Longest stored prefix whose token IDs exactly equal the start of `prompt` (up to `limit` = prompt.size() - 1).
    // Any token mismatch inside [0, e.position) means lcp < e.position, so `e` is not matched.
    Ds41PrefixHit match(const std::vector<int>& prompt, int limit, const g6rt::CacheNamespace& ns) const {
        if (entries_.empty()) return Ds41PrefixHit{false, 0, 0, 0, "first"};
        if (limit <= 0) return Ds41PrefixHit{false, 0, 0, 0, "short"};
        int best_idx = -1;
        uint32_t best_pos = 0;
        uint64_t best_gen = 0;
        int max_lcp = 0;
        bool saw_ns = false;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& e = entries_[i];
            if (e.ns != ns) continue;
            saw_ns = true;
            const int lcp = g6rt::PrefixReuse::common_prefix(e.tokens, prompt);
            if (lcp > max_lcp) max_lcp = lcp;
            if (e.position <= uint32_t(limit) && lcp >= int(e.position)) {
                if (e.position > best_pos || (e.position == best_pos && e.generation > best_gen)) {
                    best_idx = int(i);
                    best_pos = e.position;
                    best_gen = e.generation;
                }
            }
        }
        if (best_idx >= 0) return Ds41PrefixHit{true, best_gen, best_pos, max_lcp, "hit"};
        return Ds41PrefixHit{false, 0, 0, max_lcp, !saw_ns ? "namespace" : (max_lcp > 0 ? "short" : "no-lcp")};
    }

    uint64_t capture(int slot, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns, cudaStream_t st) {
        if (pos == 0 || tokens.size() < size_t(pos)) return 0;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].position == pos && entries_[i].ns == ns &&
                std::equal(entries_[i].tokens.begin(), entries_[i].tokens.end(), tokens.begin())) {
                Ds41PrefixEntry hit = std::move(entries_[i]);
                entries_.erase(entries_.begin() + long(i));
                const uint64_t gen = hit.generation;
                last = {hit.bytes, 0.0};
                entries_.push_back(std::move(hit));
                return gen;
            }
        }
        const size_t need = entry_bytes(pos);
        if (need == 0 || need > budget_bytes_) return 0;
        while (used_bytes_ + need > budget_bytes_ && !entries_.empty()) {
            evict_oldest();
        }
        void* dev_buf = nullptr;
        while (cudaMalloc(&dev_buf, need) != cudaSuccess) {
            cudaGetLastError();
            if (entries_.empty()) return 0;
            evict_oldest();
        }
        const auto t0 = std::chrono::steady_clock::now();
        copy_state(slot, pos, dev_buf, true, st);
        G6RT_CK(cudaStreamSynchronize(st));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        Ds41PrefixEntry e;
        e.generation = next_gen_++;
        e.position = pos;
        e.bytes = need;
        e.tokens.assign(tokens.begin(), tokens.begin() + pos);
        e.ns = ns;
        e.dev_buf = dev_buf;
        std::fill_n(e.ehist, ds41::engram::kRing, -1);
        for (int t = std::max(0, int(pos) - ds41::engram::kRing); t < int(pos); ++t) {
            ds41::engram::ring_put(e.ehist, t, e.tokens[size_t(t)]);
        }
        const uint64_t gen = e.generation;
        used_bytes_ += need;
        captured_bytes_ += need;
        ++captures_;
        last = {need, ms};
        entries_.push_back(std::move(e));
        return gen;
    }

    void restore(uint64_t gen, int slot, cudaStream_t st) {
        int idx = -1;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].generation == gen) { idx = int(i); break; }
        }
        require(idx >= 0, "DS41 prefix cache restore: entry not found");
        const auto t0 = std::chrono::steady_clock::now();
        Ds41PrefixEntry& e = entries_[size_t(idx)];
        copy_state(slot, e.position, e.dev_buf, false, st);
        if (main_.ehist) {
            G6RT_CK(cudaMemcpyAsync(main_.ehist + size_t(slot) * g6rt::ENG_RING, e.ehist, sizeof(e.ehist),
                                    cudaMemcpyHostToDevice, st));
        }
        G6RT_CK(cudaStreamSynchronize(st));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        last = {e.bytes, ms};
        ++hits_;
        restored_bytes_ += e.bytes;
        // one line per hit: a gate reads the increment itself, not a value derived from the HTTP usage field
        fprintf(stderr, "# ds41 prefix cache hit gen %llu slot %d pos %u bytes %zu ms %.2f hits %llu restored_bytes %zu\n",
                (unsigned long long)gen, slot, e.position, e.bytes, ms, (unsigned long long)hits_, restored_bytes_);
        if (size_t(idx) + 1 < entries_.size()) {
            Ds41PrefixEntry moved = std::move(entries_[size_t(idx)]);
            entries_.erase(entries_.begin() + idx);
            entries_.push_back(std::move(moved));
        }
    }

    g6kv_tier_totals tier_stat() const {
        constexpr uint64_t BLOCK = 1ull << 20; // report in 1 MiB blocks
        g6kv_tier_totals t{};
        t.block_bytes = BLOCK;
        t.slots = budget_bytes_ / BLOCK;
        t.free_slots = (budget_bytes_ >= used_bytes_) ? (budget_bytes_ - used_bytes_) / BLOCK : 0;
        t.entries = uint32_t(entries_.size());
        t.hits = hits_;
        t.dropped = evictions_;
        t.spilled_pages = (captured_bytes_ + BLOCK - 1) / BLOCK;
        t.restored_pages = (restored_bytes_ + BLOCK - 1) / BLOCK;
        return t;
    }

private:
    void evict_oldest() {
        if (entries_.empty()) return;
        if (entries_.front().dev_buf) cudaFree(entries_.front().dev_buf);
        used_bytes_ -= entries_.front().bytes;
        entries_.erase(entries_.begin());
        ++evictions_;
    }

    void copy_state(int slot, uint32_t pos, void* buf, bool to_cache, cudaStream_t st) {
        using namespace g6rt;
        const size_t p = size_t(pos), p2 = p / 2, q = size_t(slot);
        uint8_t* cur = static_cast<uint8_t*>(buf);
        auto xfer = [&](void* dev_ptr, size_t nbytes) {
            if (!nbytes) return;
            if (to_cache) G6RT_CK(cudaMemcpyAsync(cur, dev_ptr, nbytes, cudaMemcpyDeviceToDevice, st));
            else G6RT_CK(cudaMemcpyAsync(dev_ptr, cur, nbytes, cudaMemcpyDeviceToDevice, st));
            cur += nbytes;
        };
        for (int i = 0; i < (int)main_.layers.size(); ++i) {
            Ds41AttnLayer* a = ds41_attn_state(main_, i);
            if (!a) continue;
            const size_t ring_elems = size_t(a->C.ring_rows) * ds41a::HD;
            xfer(a->C.ring + q * ring_elems, ring_elems * sizeof(__nv_bfloat16));
            if (a->C.comp) {
                xfer(a->C.comp + q * a->C.comp_ss, p2 * ds41a::HD * sizeof(__nv_bfloat16));
                if (a->C.kvs) xfer(a->C.kvs + q * 2 * ds41a::HD, 2 * ds41a::HD * sizeof(float));
                if (a->C.scs) xfer(a->C.scs + q * 2 * ds41a::HD, 2 * ds41a::HD * sizeof(float));
            }
        }
        if (Ds41IdxModel* S = ds41_idx_detail::find(main_, false)) {
            for (int k = 0; k < 4; ++k) {
                const size_t nrows = (k < 3) ? p2 : p;
                const size_t rpp = size_t(S->rows_kv[k]);
                if (S->ica[k]) {
                    static_assert(g6kv::ICA_CODES + g6kv::ICA_SF == DS41_ICA_ROW, "DS41_ICA_ROW layout");
                    uint8_t* page0 = S->ica[k] + q * rpp * DS41_ICA_ROW;
                    xfer(page0, nrows * size_t(g6kv::ICA_CODES));
                    xfer(page0 + rpp * size_t(g6kv::ICA_CODES), nrows * size_t(g6kv::ICA_SF));
                }
                if (S->ca[k]) {
                    uint8_t* page0 = S->ca[k] + q * rpp * DS41_CA_ROW;
                    xfer(page0, nrows * DS41_CA_ROW);
                }
            }
            if (S->shadow) {
                const size_t sh_stride = size_t(S->cap + S->rcap) * 512;
                xfer(S->shadow + q * sh_stride, p * 512 * sizeof(__nv_bfloat16));
            }
        }
        if (dsp_) {
            for (int j = 0; j < (int)dsp_->draft.layers.size(); ++j) {
                Ds41AttnLayer* da = ds41_attn_state(dsp_->draft, j);
                if (da && da->C.ring) {
                    const size_t ring_elems = size_t(da->C.ring_rows) * ds41a::HD;
                    xfer(da->C.ring + q * ring_elems, ring_elems * sizeof(__nv_bfloat16));
                }
            }
        }
    }

    g6rt::Model& main_;
    Ds41Dspark* dsp_ = nullptr;
    size_t budget_bytes_ = 0;
    size_t used_bytes_ = 0;
    size_t captured_bytes_ = 0;
    size_t restored_bytes_ = 0;
    uint64_t next_gen_ = 1;
    uint64_t hits_ = 0;
    uint64_t captures_ = 0;
    uint64_t evictions_ = 0;
    std::vector<Ds41PrefixEntry> entries_;
};

} // namespace g6s2
#endif
