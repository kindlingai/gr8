// DeepSeek-V4.1-Flash-1017 prefix state snapshot cache (attention, indexer, Engram and DSpark state).
// Stores prefix snapshots [0, p) (any p >= 1, even or odd) in lazily allocated device slabs
// bounded by GR8_DS_CACHE_MIB (default 4096 MiB per rank) with LRU eviction.
// A snapshot at position p consists of:
//   - Main model attention (40 layers): C.ring (ring_rows x 512 bf16) + C.comp (or C.compq, 288 B rows: GR8_DS_COMP_FP4) + C.kvs/C.scs at ratio-2 sources 2,8,14
//   - Indexer / ratio-1 KV (Ds41IdxModel): ica[0..2] planar codes+sf for p/2 rows, ica[3] planar codes+sf for p rows,
//     ca[3] (p x 288 B), shadow (p x 512 bf16; none with GR8_DS_COMP_FP4)
//   - Engram token ring (ehist): 64 int32 entries reconstructed from tokens[0..p)
//   - DSpark draft model (3 layers, when enabled): C.ring (ring_rows x 512 bf16)
// Device snapshots use cudaMemcpyAsync; tiered snapshots use the declarative state mover on the session stream.
#pragma once
#if G6_MODEL_DS41
#include "service.h"
#ifndef GR8_DS41_PREFIX_CACHE_STANDALONE
#include "ds41_dspark_round.cuh"
#endif
#include "../../attn/kvfmt.h"
#include "../../engram/engram_ring.h"
#include "ds41_engram_ring.h"
#include "kv_state_desc.h"
#if defined(__CUDACC__) || !defined(GR8_DS41_PREFIX_CACHE_STANDALONE)
#include "kv_state_mover.cuh"
#endif
#include <filesystem>
#include <memory>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace g6s2 {

// What a request captured an entry at (GR8_DS_CACHE_REPLACE): its prompt end or its request end.
enum class Ds41Snap : uint8_t { Other = 0, PromptEnd = 1, RequestEnd = 2 };

struct Ds41PrefixEntry {
    uint64_t generation = 0;
    Ds41Snap kind = Ds41Snap::Other;
    uint64_t sibling = 0;   // the other entry the same request captured (0: none); it may have been freed since
    uint32_t position = 0;
    size_t bytes = 0;
    std::vector<int> tokens;
    g6rt::CacheNamespace ns{};
    uint32_t continuations = 0, tier_pages = 0;
    uint64_t tier_id = 0;
    int pf_end = 0;
    int64_t system_span = -1;
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
    // A single self-registered pointer carries it there (the hit counter must be readable at stop time).
    static Ds41PrefixCache*& live() { static Ds41PrefixCache* p = nullptr; return p; }
    // One line, the same fields every time: what a gate reads back after the prefix-cache requests.
    std::string stats_line() const {
        char b[256];
        snprintf(b, sizeof b, "entries %zu used_mib %zu captures %llu hits %llu evictions %llu restored_bytes %zu replaced %llu siblings %llu",
                 entries_.size(), used_bytes_ >> 20, (unsigned long long)captures_, (unsigned long long)hits_,
                 (unsigned long long)evictions_, restored_bytes_, (unsigned long long)replaced_,
                 (unsigned long long)siblings_);
        return std::string(b);
    }

    Ds41PrefixCache(const Ds41PrefixCache&) = delete;
    Ds41PrefixCache& operator=(const Ds41PrefixCache&) = delete;

    ~Ds41PrefixCache() {
        tiers_.reset();
        if (d_ehist_snap_) cudaFree(d_ehist_snap_);
        if (live() == this) live() = nullptr;
        for (auto& e : entries_) {
            if (e.dev_buf) cudaFree(e.dev_buf);
        }
    }

    size_t budget_bytes() const { return budget_bytes_; }
    size_t used_bytes() const { return used_bytes_; }
    size_t size() const { return entries_.size(); }
    const std::vector<Ds41PrefixEntry>& entries() const { return entries_; }
    uint64_t hits() const { return hits_; }
    uint64_t captures() const { return captures_; }
    uint64_t deduped() const { return deduped_; }
    uint64_t replaced() const { return replaced_; }
    uint64_t siblings() const { return siblings_; }
    uint64_t evictions() const { return evictions_; }

    uint64_t digest() const {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ uint64_t(entries_.size());
        for (const auto& e : entries_) {
            h ^= (e.generation + (uint64_t(e.position) << 32)) * 0xBF58476D1CE4E5B9ull;
            h = (h << 27) | (h >> 37);
        }
        if (stash_) h ^= stash_->digest() * 0x94D049BB133111EBull;
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
            if (a->C.comp || a->C.compq) {   // GR8_DS_COMP_FP4: packed 288 B rows
                total += p2 * (a->C.compq ? size_t(ds41a::CQ_ROW) : ds41a::HD * sizeof(__nv_bfloat16));
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
        auto best = match_l1(prompt, limit, ns);
        if (stash_) {
            const auto h = stash_->match(prompt, limit, ns);
            if (h.stored && h.skip > int(best.position)) {
                require(h.age <= STORE_ID && h.tier < 128, "DS41 store handle overflow");
                best = {true, STORE_HIT | (uint64_t(h.tier) << 56) | h.age, uint32_t(h.skip), h.lcp, "stored"};
            }
        }
        return best;
    }

    Ds41PrefixHit match_l1(const std::vector<int>& prompt, int limit, const g6rt::CacheNamespace& ns) const {
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

    // GR8_DS_CACHE_REPLACE: `kind` is recorded on a new entry. `prompt_gen` is the entry this same request captured at its
    // prompt end (0: none), used when it is still stored, same namespace, shorter and a token prefix of the new entry:
    //   replace = true (the request restored a request-end entry, so its client sends replies back as generated): it is
    //     freed before the new buffer is allocated. match() needs an entry's tokens to be a prefix of the new prompt, so it
    //     is lost only to prompts that share its prefix and leave the reply (a retry, regenerate or edit: a re-prefill).
    //   replace = false: a new entry is linked to it as its sibling; the first restore of either frees the other (restore()).
    // keep (GR8_DS_PARK: a grant's own hit, restored after this capture) is never freed or evicted here: when only it stands
    // in the way the capture is skipped (returns 0). images: the request's image rows, -1 in the entry's Engram ring as in the
    // live ring (ds41_engram_ring.h)
    uint64_t capture(int slot, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns, cudaStream_t st,
                     Ds41Snap kind = Ds41Snap::Other, uint64_t prompt_gen = 0, bool replace = false, uint64_t keep = 0,
                     const Ds41ImageRows& images = {}) {
        const uint64_t replace_gen = replace && prompt_gen != keep ? prompt_gen : 0;
        if (pos == 0 || tokens.size() < size_t(pos)) return 0;
        capture_store(slot, pos, tokens, ns, st, kind, replace, keep, images);
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].position == pos && entries_[i].ns == ns &&
                std::equal(entries_[i].tokens.begin(), entries_[i].tokens.end(), tokens.begin())) {
                const long gone = drop_replaced(replace_gen, pos, tokens, ns);   // index erased, or -1
                if (gone >= 0 && gone < long(i)) --i;
                if (stash_) {
                    const auto& state = store_sessions_.at(slot);
                    entries_[i].continuations = state.hits;
                    entries_[i].tier_id = state.id;
                    entries_[i].tier_pages = state.pages;
                    entries_[i].pf_end = state.pf_end; entries_[i].system_span = state.sys;
                }
                Ds41PrefixEntry hit = std::move(entries_[i]);
                entries_.erase(entries_.begin() + long(i));
                const uint64_t gen = hit.generation;
                last = {hit.bytes, 0.0};
                entries_.push_back(std::move(hit));
                return gen;
            }
        }
        // Any stored same-namespace token prefix at most K tokens behind `pos` is reused instead of copying essentially the
        // same GPU state twice. K=1 by default; a larger value also skips short-turn snapshots.
        // K=0 with GR8_DS_CACHE_REPLACE=0 keeps every snapshot. Return 0 when no new generation was captured.
        if (const int K = dedup_k()) {
            for (size_t i = 0; i < entries_.size(); ++i) {
                const auto& e = entries_[i];
                if (e.ns == ns && e.position < pos && e.position + uint32_t(K) >= pos &&
                    std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin())) {
                    Ds41PrefixEntry keep = std::move(entries_[i]);
                    entries_.erase(entries_.begin() + long(i));
                    ++deduped_;
                    entries_.push_back(std::move(keep));
                    return 0;
                }
            }
        }
        const size_t need = entry_bytes(pos);
        if (need == 0 || need > budget_bytes_) return 0;
        drop_replaced(replace_gen, pos, tokens, ns);
        while (used_bytes_ + need > budget_bytes_ && !entries_.empty()) {
            if (!evict_oldest(keep)) return 0;
        }
        void* dev_buf = nullptr;
        while (cudaMalloc(&dev_buf, need) != cudaSuccess) {
            cudaGetLastError();
            if (!evict_oldest(keep)) return 0;
        }
        const auto t0 = std::chrono::steady_clock::now();
        copy_state(slot, pos, dev_buf, true, st);
        G6RT_CK(cudaStreamSynchronize(st));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        Ds41PrefixEntry e;
        e.generation = next_gen_++;
        if (stash_) {
            const auto& state = store_sessions_.at(slot);
            e.continuations = state.hits; e.tier_id = state.id; e.tier_pages = state.pages;
            e.pf_end = state.pf_end; e.system_span = state.sys;
        }
        e.kind = kind;
        if (!replace && replace_on())
            if (Ds41PrefixEntry* pe = prompt_entry(prompt_gen, pos, tokens, ns); pe && !pe->sibling) {
                pe->sibling = e.generation;
                e.sibling = prompt_gen;
            }
        e.position = pos;
        e.bytes = need;
        e.tokens.assign(tokens.begin(), tokens.begin() + pos);
        e.ns = ns;
        e.dev_buf = dev_buf;
        ds41_engram_ring(e.ehist, e.tokens, pos, images);
        const uint64_t gen = e.generation;
        used_bytes_ += need;
        captured_bytes_ += need;
        ++captures_;
        last = {need, ms};
        entries_.push_back(std::move(e));
        return gen;
    }

    // Returns the restored entry's kind. GR8_DS_CACHE_REPLACE: the follow-up shows which of a request's two entries its chat
    // continues from, so the other one (its sibling) is freed now, if it is still stored. A freed request-end sibling is lost
    // to a later prompt that keeps that reply after all (a tools turn after a thinking turn without them): a re-prefill.
    Ds41Snap restore(uint64_t gen, int slot, cudaStream_t st, uint32_t pos = 0) {
        if (stored_hit(gen)) {
            require(stash_ && pos, "DS41 tier restore needs a position");
            const auto meta = stash_->info(store_id(gen));
            auto r = stash_->restore(uint32_t((gen >> 56) & 127), store_id(gen),
                                     state_pages(slot, pos), pack_state_handle(slot, pos));
            require(r.valid == int(pos), "DS41 tier position differs from granted skip");
            drain_store(st);
            if (d_ehist_snap_) {
                G6RT_CK(cudaMemcpyAsync(main_.ehist + size_t(slot) * g6rt::ENG_RING,
                    d_ehist_snap_ + size_t(slot) * g6rt::ENG_RING, g6rt::ENG_RING * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, st));
                G6RT_CK(cudaStreamSynchronize(st));
            }
            store_sessions_.at(slot) = {r.hits + 1, uint32_t(r.full_pages), pos, store_id(gen), 0, meta.pf_end, -1};
            ++hits_;
            last = {(r.full_pages + tiers_->desc().state_blocks()) * tiers_->desc().slot_bytes(), 0};
            restored_bytes_ += last.bytes;
            return pos == uint32_t(meta.pf_end) ? Ds41Snap::PromptEnd : Ds41Snap::RequestEnd;
        }
        int idx = -1;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].generation == gen) { idx = int(i); break; }
        }
        require(idx >= 0, "DS41 prefix cache restore: entry not found");
        const auto t0 = std::chrono::steady_clock::now();
        Ds41PrefixEntry& e = entries_[size_t(idx)];
        if (stash_) store_sessions_.at(slot) =
            {e.continuations + 1, e.tier_pages, e.position, e.tier_id, 0, e.pf_end, e.system_span};
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
        const Ds41Snap kind = e.kind;
        const uint64_t sib = e.sibling;
        e.sibling = 0;
        if (size_t(idx) + 1 < entries_.size()) {
            Ds41PrefixEntry moved = std::move(entries_[size_t(idx)]);
            entries_.erase(entries_.begin() + idx);
            entries_.push_back(std::move(moved));
        }
        free_sibling(gen, sib);
        return kind;
    }

    // GR8_DS_PARK: a chat parked at [0, pos) keeps its request end in its slot; the capture waits for the bind that evicts
    // it. park_end does now the bookkeeping capture(RequestEnd, prompt_gen, replace) does before its copy, so the other
    // entries change when they would without park:
    //   an equal entry stored: with replace, the request's own prompt-end entry is freed; the equal entry is refreshed
    //   the dedup, pos 0 or over the budget: nothing
    //   else: with replace, the prompt-end entry is freed; without, it is linked to a placeholder for this park
    //     (park_mark(id)). A restore of the prompt-end entry then frees the placeholder, as it frees a captured request end
    //     (the eviction captures nothing)
    // Returns the park's id, unique (never a slot number: a later chat on the slot cannot alias it). Every park ends in
    // evict_done or resume_parked, which detach it.
    uint64_t park_end(uint64_t prompt_gen, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns,
                      bool replace) {
        const uint64_t id = ++next_park_;
        live_parks_.push_back(id);
        if (pos == 0 || tokens.size() < size_t(pos)) return id;
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].position == pos && entries_[i].ns == ns &&
                std::equal(entries_[i].tokens.begin(), entries_[i].tokens.end(), tokens.begin())) {   // capture's equal-entry path
                const long gone = drop_replaced(replace ? prompt_gen : 0, pos, tokens, ns);
                if (gone >= 0 && gone < long(i)) --i;
                Ds41PrefixEntry hit = std::move(entries_[i]);
                entries_.erase(entries_.begin() + long(i));
                entries_.push_back(std::move(hit));
                return id;
            }
        if (!new_entry(pos, tokens, ns)) return id;
        if (replace) drop_replaced(prompt_gen, pos, tokens, ns);
        else if (replace_on())
            if (Ds41PrefixEntry* pe = prompt_entry(prompt_gen, pos, tokens, ns); pe && !pe->sibling) pe->sibling = park_mark(id);
        return id;
    }
    // GR8_DS_PARK eviction of park id, before its capture: true when its request end was freed as a restored entry's sibling,
    // or will be now (keep, the entry the evicting grant restores next, is its linked prompt-end entry): nothing to capture.
    // Either way the park is detached (no entry links to it), so the capture links the prompt-end entry as retain's does
    bool evict_parked_freed(uint64_t id, uint64_t keep) {
        bool f = std::find(freed_parks_.begin(), freed_parks_.end(), id) != freed_parks_.end();
        if (!f && keep && replace_on())
            if (const Ds41PrefixEntry* e = find(keep); e && e->sibling == park_mark(id)) { ++siblings_; f = true; }
        detach(id);
        return f;
    }
    // GR8_DS_PARK resume of park id: the follow-up continues from the parked request end, as from a restore of it: its
    // sibling (the request's prompt-end entry, linked by park_end) is freed. A request end that would have been an equal
    // stored entry or folded into one by the dedup is that entry, restored. The park is detached on every path. Returns
    // the kind that restore would have returned.
    Ds41Snap resume_parked(uint64_t id, uint64_t prompt_gen, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns) {
        Ds41Snap kind = Ds41Snap::RequestEnd;
        if (Ds41PrefixEntry* e = stored_as(pos, tokens, ns)) {
            const uint64_t gen = e->generation, sib = e->sibling;
            kind = e->kind;
            e->sibling = 0;
            free_sibling(gen, sib);
        } else if (Ds41PrefixEntry* pe = prompt_entry(prompt_gen, pos, tokens, ns); pe && pe->sibling == park_mark(id)) {
            free_at(size_t(pe - entries_.data()));
            ++siblings_;
        }
        detach(id);
        return kind;
    }
    // GR8_DS_PARK only (the pool digest folds it in with park on): the state the next eviction or resume reads, beyond
    // digest(): each entry's kind and sibling, the live and freed parks
    uint64_t park_digest() const {
        uint64_t h = 0x84222325CBF29CE4ull ^ next_park_;
        auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001B3ull; h ^= h >> 29; };
        for (const auto& e : entries_) { mix(e.generation); mix(uint64_t(e.kind)); mix(e.sibling); }
        mix(0xA11CEull);
        for (uint64_t id : live_parks_) mix(id);
        mix(0xF4EEDull);
        for (uint64_t id : freed_parks_) mix(id);
        return h;
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

    struct StoreSession {
        uint32_t hits = 0, pages = 0, earned = 0;
        uint64_t id = 0, prompt_id = 0;
        int pf_end = 0;
        int64_t sys = -1;
    };
    static constexpr uint64_t STORE_HIT = 1ull << 63;
    static constexpr uint64_t STORE_ID = (1ull << 56) - 1;
    static bool stored_hit(uint64_t gen) { return (gen & STORE_HIT) != 0; }
    static uint64_t store_id(uint64_t gen) { return gen & STORE_ID; }

    KvStateDesc state_desc(uint32_t slots, uint32_t max_pages) {
        using namespace g6rt;
        KvStateDesc d;
        d.model_tag = "ds41";
        d.max_slots = slots;
        d.max_pages_per_slot = max_pages;
        auto tail = [&](void* p, size_t bytes) {
            if (p) d.tail_extents.push_back({static_cast<uint8_t*>(p), bytes, bytes});
        };
        for (int i = 0; i < int(main_.layers.size()); ++i) {
            auto* a = ds41_attn_state(main_, i);
            if (!a) continue;
            tail(a->C.ring, size_t(a->C.ring_rows) * 512 * sizeof(__nv_bfloat16));
            if (a->C.compq)
                d.page_planes.push_back({a->C.compq, size_t(a->C.comp_ss), 32, 288, 2});
            else if (a->C.comp)
                d.page_planes.push_back({reinterpret_cast<uint8_t*>(a->C.comp),
                    size_t(a->C.comp_ss) * sizeof(__nv_bfloat16), 32, 512 * sizeof(__nv_bfloat16), 2});
            if (a->C.compq || a->C.comp) {
                tail(a->C.kvs, 2 * 512 * sizeof(float));
                tail(a->C.scs, 2 * 512 * sizeof(float));
            }
        }
        if (auto* s = ds41_idx_detail::find(main_, false)) {
            for (int k = 0; k < 4; ++k) {
                const uint32_t ratio = k < 3 ? 2 : 1, rows = 64 / ratio;
                const size_t cap = size_t(s->rows_kv[k]);
                if (s->ica[k]) {
                    d.page_planes.push_back({s->ica[k], cap * 68, rows, 64, ratio});
                    d.page_planes.push_back({s->ica[k] + cap * 64, cap * 68, rows, 4, ratio});
                }
                if (s->ca[k]) d.page_planes.push_back({s->ca[k], cap * 288, rows, 288, ratio});
            }
            if (s->shadow) d.page_planes.push_back({reinterpret_cast<uint8_t*>(s->shadow),
                size_t(s->cap + s->rcap) * 1024, 64, 1024, 1});
        }
        if (dsp_) for (int i = 0; i < int(dsp_->draft.layers.size()); ++i) {
            auto* a = ds41_attn_state(dsp_->draft, i);
            if (a) tail(a->C.ring, size_t(a->C.ring_rows) * 1024);
        }
        tail(d_ehist_snap_, g6rt::ENG_RING * sizeof(int32_t));
        d.validate();
        return d;
    }

    void open_stores(const std::vector<std::string>& specs, bool old, const std::string& default_keep,
                     unsigned shared_after, double write_limit, const std::string& set_id,
                     unsigned nr, unsigned slots, unsigned max_pages) {
        if (specs.empty()) return;
        store_sessions_.resize(slots);
        if (main_.ehist)
            G6RT_CK(cudaMalloc(reinterpret_cast<void**>(&d_ehist_snap_), slots * g6rt::ENG_RING * sizeof(int32_t)));
        auto desc = state_desc(slots, max_pages);
        tiers_ = std::make_unique<KvStateTiers>(desc);
        stash_ = std::make_unique<StashPool>(shared_after, 64, desc.state_blocks(), false);
        stash_->set_rules(false, true, 0);
        const std::string exe = exe_stamp();
        for (size_t i = 0; i < specs.size(); ++i) {
            const auto t = TierSpec::parse(specs[i], old);
            std::string spec = t.store, index;
            if (t.ssd()) {
                const auto dir = t.store.substr(4) + "/r" + std::to_string(main_.rank);
                if (!t.persist && std::filesystem::exists(dir)) std::filesystem::remove_all(dir);
                std::filesystem::create_directories(dir);
                spec = "ssd:" + dir + "/blocks";
                if (t.persist && !set_id.empty()) index = dir;
            }
            if (t.spare_ram) spec = "ram:" + std::to_string(t.spare_ram_floor) + "+" + spec;
            const std::string name = "ds41-" + set_id + "-" + std::to_string(getpid()) +
                "-r" + std::to_string(main_.rank) + "-t" + std::to_string(i);
            const auto blocks = t.group.empty() ? tiers_->add(spec, name, t.bytes) :
                tiers_->add_group(spec, name, t.group, nr, main_.rank, t.bytes, t.cap, t.weight);
            const std::string keep = t.rule.empty() ? default_keep : t.rule;
            const bool whole = !index.empty() && keep != "system";
            store_shared_ |= keep == "system" || keep == "all";
            kv_index_ |= !index.empty();
            kv_whole_ |= whole;
            // Layout hash includes the compression and draft-ring extents, not process pointers.
            const std::string compat = "ds41 set " + set_id + " nr " + std::to_string(nr) +
                " slots " + std::to_string(slots) + " pages " + std::to_string(max_pages) +
                " dspark " + std::to_string(dsp_ != nullptr) +
                " format " + std::to_string(desc.format_hash()) + " state " + std::to_string(desc.state_blocks()) +
                " blocks " + std::to_string(blocks) +
                " engine " + exe;
            stash_->add_tier(blocks, (keep == "system" ? 1 : 2) | (whole ? 4 : 0),
                index.empty() ? nullptr : index.c_str(), compat.c_str());
            stash_->set_policy(i, 0, keep == "sessions" ? 1 : 0, desc.slot_bytes());
            fprintf(stderr, "# rank %d: s2 store %zu: %s, %llu blocks of %zu B, keep %s%s\n",
                main_.rank, i, tiers_->describe(i).c_str(), (unsigned long long)blocks, desc.slot_bytes(),
                keep.c_str(), index.empty() ? "" : ", indexed");
        }
        stash_->tiers_loaded();
        stash_->write_limit(uint64_t(std::llround(write_limit * 1000)));
    }
    bool has_stores() const { return bool(stash_); }
    bool persistent() const { return kv_index_; }
    bool whole_store() const { return kv_whole_; }
    std::string stored_summary() const { return stash_ ? stash_->stored_summary() : ""; }
    void keep_agreed(const std::string& s) { if (stash_) stash_->keep_agreed(s); }
    std::vector<g6kv_tier_totals> tier_stats() const {
        std::vector<g6kv_tier_totals> out{tier_stat()};
        if (stash_) for (size_t i = 0; i < tiers_->size(); ++i) out.push_back(stash_->tier_stats(i));
        return out;
    }
    bool available(uint64_t gen, uint32_t pos = 0) const {
        if (stored_hit(gen)) return stash_ && stash_->has(store_id(gen)) && (!pos || stash_->info(store_id(gen)).valid == int(pos));
        auto* e = find(gen);
        return e && (!pos || e->position == pos);
    }
    void reset_store_slot(int q) { if (stash_) store_sessions_.at(q) = StoreSession{}; }
    void resume_store_slot(int q) { if (stash_) ++store_sessions_.at(q).hits; }
    void admit_store(const g6rt::ServeReq& r, bool continuing) {
        if (!stash_) return;
        stash_->count(r.prompt, r.sys, g6rt::steering::cache_namespace(r.steering), continuing);
        stash_->io_done();
    }
    void store_span(int q, int64_t sys) { if (stash_) store_sessions_.at(q).sys = sys; }
    void unlimited_store() { if (kv_whole_) stash_->write_limit(0); }

    uint32_t system_boundary(const g6rt::ServeReq& r) const {
        if (!store_shared_ || r.prompt.empty() || r.steering.enabled) return 0;
        const uint64_t top = std::min<uint64_t>(r.prompt.size() - 1, r.sys < 0 ? r.prompt.size() - 1 : uint64_t(r.sys));
        return uint32_t(top / 64 * 64);
    }
    void capture_system(int q, uint32_t pos, const std::vector<int>& tokens,
                        cudaStream_t st, const Ds41ImageRows& images) {
        if (!stash_ || !store_shared_ || !pos) return;
        auto& s = store_sessions_.at(q);
        if (d_ehist_snap_) {
            int ring[g6rt::ENG_RING]; ds41_engram_ring(ring, tokens, pos, images);
            G6RT_CK(cudaMemcpyAsync(d_ehist_snap_ + size_t(q) * g6rt::ENG_RING, ring, sizeof(ring),
                                   cudaMemcpyHostToDevice, st));
            G6RT_CK(cudaStreamSynchronize(st));
        }
        if (pos > s.earned) {
            stash_->earn(pos / 64 - s.earned / 64 + tiers_->desc().state_blocks());
            s.earned = pos;
        }
        stash_->promote(tokens, pos, pos, pos, state_pages(q, pos), pack_state_handle(q, pos), pos, s.sys);
        drain_store(st);
    }

    void flush_store(cudaStream_t st) {
        if (!kv_whole_) return;
        stash_->write_limit(0);
        // Serving has stopped: slot zero is scratch, never a live session.
        for (const auto& e : entries_) {
            if (e.tier_id && stash_->has(e.tier_id) && stash_->info(e.tier_id).valid == int(e.position)) continue;
            copy_state(0, e.position, e.dev_buf, false, st);
            G6RT_CK(cudaStreamSynchronize(st));
            store_sessions_[0] = {e.continuations, e.tier_pages, e.tier_pages, e.tier_id, 0, e.pf_end, e.system_span};
            if (d_ehist_snap_)
                G6RT_CK(cudaMemcpyAsync(d_ehist_snap_, e.ehist, sizeof(e.ehist), cudaMemcpyHostToDevice, st));
            capture_store(0, e.position, e.tokens, e.ns, st, e.kind, false, 0, {}, true);
        }
        drain_store(st);
        uint64_t entries = 0, blocks = 0;
        for (size_t i = 0; i < tiers_->size(); ++i) {
            auto t = stash_->tier_stats(i); entries += t.entries; blocks += t.spilled_pages;
        }
        fprintf(stderr, "# rank %d: s2 stores: quit stored %llu entries, %llu blocks written\n",
            main_.rank, (unsigned long long)entries, (unsigned long long)blocks);
    }

private:
    std::unique_ptr<StashPool> stash_;
    std::unique_ptr<KvStateTiers> tiers_;
    std::vector<StoreSession> store_sessions_;
    int32_t* d_ehist_snap_ = nullptr;
    bool kv_index_ = false, kv_whole_ = false, store_shared_ = false;

    std::vector<uint32_t> state_pages(unsigned q, unsigned pos) const {
        std::vector<uint32_t> pages(pos / 64);
        for (unsigned i = 0; i < pages.size(); ++i)
            pages[i] = pack_page_handle(q, i, tiers_->desc().max_pages_per_slot);
        return pages;
    }
    void drain_store(cudaStream_t st) {
        if (!stash_) return;
        const auto io = stash_->io();
        // A checksum or I/O error is fatal to this serving instance: never use a partly restored slot.
        // serve_ds41 unwinds through links.shutdown(); pending uncommitted captures are not published.
        tiers_->run(io, st);
        stash_->io_done();
    }
    void capture_store(int q, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns,
                       cudaStream_t st, Ds41Snap kind, bool replace, uint64_t keep, const Ds41ImageRows& images,
                       bool ehist_ready = false) {
        if (!stash_ || !pos || tokens.size() < pos) return;
        auto& s = store_sessions_.at(q);
        if (d_ehist_snap_ && !ehist_ready) {
            int ring[g6rt::ENG_RING];
            ds41_engram_ring(ring, tokens, pos, images);
            // Never overwrite the active ring when capturing a prompt boundary.
            G6RT_CK(cudaMemcpyAsync(d_ehist_snap_ + size_t(q) * g6rt::ENG_RING, ring, sizeof(ring),
                                   cudaMemcpyHostToDevice, st));
            G6RT_CK(cudaStreamSynchronize(st));
        }
        const auto pages = state_pages(q, pos);
        // Full-page credit counts only newly computed rows. Each new boundary also earns its fixed state overhead.
        if (pos > s.earned) {
            stash_->earn(pos / 64 - s.earned / 64 + tiers_->desc().state_blocks());
            s.earned = pos;
        }
        if (kind == Ds41Snap::PromptEnd) s.pf_end = pos;
        uint64_t keep_id = stored_hit(keep) ? store_id(keep) : 0;
        if (keep && !keep_id) if (auto* e = find(keep)) keep_id = e->tier_id;
        const bool keep_from = kind == Ds41Snap::RequestEnd && !replace && s.prompt_id == s.id;
        const uint64_t id = stash_->capture(tokens, pos, pos, s.pf_end, pages, pack_state_handle(q, pos),
            s.hits, ns, s.id, s.pages, keep_from, keep_id);
        drain_store(st);
        if (id) {
            // A refused extension returns the old id; retain its old page count in that case.
            s.pages = uint32_t(stash_->info(id).pages);
            s.id = id;
            if (kind == Ds41Snap::PromptEnd) s.prompt_id = id;
        }

    }
    static int dedup_k() {
        static const int k = int(knob::ds_cache_dedup());
        return k;
    }
    static bool replace_on() {
        static const bool on = knob::ds_cache_replace();
        return on;
    }
    // Entry `gen` if it is still stored, same namespace, shorter than `pos` and its tokens start `tokens`; else null.
    // Only the generation a request captured itself is passed in, so another request's prefix entry is never touched.
    Ds41PrefixEntry* prompt_entry(uint64_t gen, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns) {
        if (!gen) return nullptr;
        for (auto& e : entries_)
            if (e.generation == gen)
                return (e.ns == ns && e.position < pos && std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin())) ? &e
                                                                                                                          : nullptr;
        return nullptr;
    }
    // GR8_DS_PARK: the sibling of a prompt-end entry whose request end is parked under park id (not a generation: next_gen_
    // is far below bit 63)
    static uint64_t park_mark(uint64_t id) { return (1ull << 63) | id; }
    // the end of park id (evicted or resumed): no entry links to it, it is neither live nor freed
    void detach(uint64_t id) {
        for (auto& e : entries_) if (e.sibling == park_mark(id)) e.sibling = 0;
        live_parks_.erase(std::remove(live_parks_.begin(), live_parks_.end(), id), live_parks_.end());
        freed_parks_.erase(std::remove(freed_parks_.begin(), freed_parks_.end(), id), freed_parks_.end());
    }
    // the entry a capture at [0, pos) would return instead of a new one: an equal stored entry, or (the dedup) a stored
    // same-namespace prefix at most K tokens behind
    Ds41PrefixEntry* stored_as(uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns) {
        for (auto& e : entries_)
            if (e.position == pos && e.ns == ns && std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin())) return &e;
        if (const int K = dedup_k())
            for (auto& e : entries_)
                if (e.ns == ns && e.position < pos && e.position + uint32_t(K) >= pos &&
                    std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin()))
                    return &e;
        return nullptr;
    }
    bool new_entry(uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns) {
        const size_t need = entry_bytes(pos);
        return !stored_as(pos, tokens, ns) && need && need <= budget_bytes_;
    }
    // restore of entry gen whose sibling was sib: the sibling is freed if it is still linked back (GR8_DS_CACHE_REPLACE)
    void free_sibling(uint64_t gen, uint64_t sib) {
        if (!sib || !replace_on()) return;
        if (sib >> 63) {   // a parked request end: nothing stored yet; only a live park (its owner still parked) is freed
            const uint64_t id = sib & ~(1ull << 63);
            if (std::find(live_parks_.begin(), live_parks_.end(), id) != live_parks_.end() &&
                std::find(freed_parks_.begin(), freed_parks_.end(), id) == freed_parks_.end()) {
                freed_parks_.push_back(id);   // its eviction captures nothing
                ++siblings_;
            }
            return;
        }
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].generation == sib) {
                if (entries_[i].sibling == gen) {
                    free_at(i);
                    ++siblings_;
                }
                break;
            }
    }
    // Frees prompt_entry(gen, ...) and returns its old index, or -1 when there is none.
    long drop_replaced(uint64_t gen, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns) {
        if (!replace_on()) return -1;
        const Ds41PrefixEntry* e = prompt_entry(gen, pos, tokens, ns);
        if (!e) return -1;
        const long i = long(e - entries_.data());
        free_at(size_t(i));
        ++replaced_;
        return i;
    }
    void free_at(size_t i) {
        if (entries_[i].dev_buf) cudaFree(entries_[i].dev_buf);
        used_bytes_ -= entries_[i].bytes;
        entries_.erase(entries_.begin() + long(i));
    }
    // the oldest entry other than keep; false when there is none
    bool evict_oldest(uint64_t keep = 0) {
        size_t i = 0;
        while (i < entries_.size() && keep && entries_[i].generation == keep) ++i;
        if (i == entries_.size()) return false;
        free_at(i);
        ++evictions_;
        return true;
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
            if (a->C.comp || a->C.compq) {
                if (a->C.compq) xfer(a->C.compq + q * a->C.comp_ss, p2 * size_t(ds41a::CQ_ROW));   // GR8_DS_COMP_FP4
                else xfer(a->C.comp + q * a->C.comp_ss, p2 * ds41a::HD * sizeof(__nv_bfloat16));
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
    uint64_t deduped_ = 0;
    uint64_t replaced_ = 0;
    uint64_t siblings_ = 0;
    uint64_t next_park_ = 0;                // GR8_DS_PARK: the last park id (park_end)
    std::vector<uint64_t> live_parks_;      // parks not yet evicted or resumed
    std::vector<uint64_t> freed_parks_;     // live parks whose request end was freed as a sibling (nothing to capture)
    std::vector<Ds41PrefixEntry> entries_;
};

// GR8_DS_CACHE_REPLACE state of each session slot from a request's bind to its retain: the entry the request captured at
// its prompt end, and whether it restored a request-end entry. serve_ds41 (ds41_native.cuh) calls these in this order:
// bind, restore (a prefix hit), capture_prompt_end, capture_request_end (retain, when there is a position), reset (retain).
// GR8_DS_PARK: a park keeps the slot's state; the next bind either resumes it (resume) or evicts it (capture_request_end, reset).
class Ds41SnapSlots {
public:
    explicit Ds41SnapSlots(int slots) : prompt_gen_(size_t(slots), 0), restored_end_(size_t(slots), false), park_id_(size_t(slots), 0) {}
    void reset(int q) { prompt_gen_[size_t(q)] = 0; restored_end_[size_t(q)] = false; park_id_[size_t(q)] = 0; }
    Ds41Snap restore(Ds41PrefixCache& ck, uint64_t gen, int q, cudaStream_t st, uint32_t pos = 0) {
        const Ds41Snap kind = ck.restore(gen, q, st, pos);
        restored_end_[size_t(q)] = kind == Ds41Snap::RequestEnd;
        return kind;
    }
    uint64_t capture_prompt_end(Ds41PrefixCache& ck, int q, uint32_t pos, const std::vector<int>& prompt,
                                const g6rt::CacheNamespace& ns, cudaStream_t st, const Ds41ImageRows& images = {}) {
        const uint64_t before = ck.captures();
        const uint64_t gen = ck.capture(q, pos, prompt, ns, st, Ds41Snap::PromptEnd, 0, false, 0, images);
        // only a new entry is this request's own: a refreshed one may belong to another request that still needs it
        if (ck.captures() != before) prompt_gen_[size_t(q)] = gen;
        return gen;
    }
    uint64_t capture_request_end(Ds41PrefixCache& ck, int q, uint32_t pos, const std::vector<int>& history,
                                 const g6rt::CacheNamespace& ns, cudaStream_t st, const Ds41ImageRows& images = {}) {
        return ck.capture(q, pos, history, ns, st, Ds41Snap::RequestEnd, prompt_gen_[size_t(q)], restored_end_[size_t(q)], 0, images);
    }
    // GR8_DS_PARK: a park keeps q's state for the bind that resumes or evicts it; the request-end bookkeeping happens now
    void park(Ds41PrefixCache& ck, int q, uint32_t pos, const std::vector<int>& history, const g6rt::CacheNamespace& ns) {
        park_id_[size_t(q)] = ck.park_end(prompt_gen_[size_t(q)], pos, history, ns, restored_end_[size_t(q)]);
        if (restored_end_[size_t(q)]) prompt_gen_[size_t(q)] = 0;   // replaced (freed) or not this request's to link
    }
    // GR8_DS_PARK eviction of the chat parked on q: its request-end capture, with the kind and sibling link of retain's
    // (keep = the new grant's claimed entry, never freed here; the capture is skipped when only it is in the way). Nothing
    // when a restore already freed it as a sibling. The park is detached and q's state reset on every path
    uint64_t evict_parked(Ds41PrefixCache& ck, int q, uint32_t pos, const std::vector<int>& history, const g6rt::CacheNamespace& ns,
                          cudaStream_t st, uint64_t keep, const Ds41ImageRows& images = {}) {
        const bool freed = ck.evict_parked_freed(park_id_[size_t(q)], keep);
        const uint64_t gen = freed ? 0 : ck.capture(q, pos, history, ns, st, Ds41Snap::RequestEnd, prompt_gen_[size_t(q)], false, keep, images);
        reset(q);
        return gen;
    }
    // GR8_DS_PARK: a follow-up resumes the chat parked on q, as if its request end had been captured and then restored on q
    void resume(Ds41PrefixCache& ck, int q, uint32_t pos, const std::vector<int>& history, const g6rt::CacheNamespace& ns) {
        const Ds41Snap kind = ck.resume_parked(park_id_[size_t(q)], prompt_gen_[size_t(q)], pos, history, ns);
        ck.resume_store_slot(q);
        reset(q);
        restored_end_[size_t(q)] = kind == Ds41Snap::RequestEnd;
    }
    // GR8_DS_PARK only (folded into the pool digest with park on): every slot's prompt-end entry, restored kind, park id
    uint64_t digest() const {
        uint64_t h = 0xCBF29CE484222325ull;
        for (size_t q = 0; q < prompt_gen_.size(); ++q) {
            h = (h ^ prompt_gen_[q]) * 0x100000001B3ull;
            h = (h ^ (uint64_t(restored_end_[q]) + (park_id_[q] << 1))) * 0x100000001B3ull;
        }
        return h;
    }

private:
    std::vector<uint64_t> prompt_gen_;
    std::vector<bool> restored_end_;
    std::vector<uint64_t> park_id_;   // GR8_DS_PARK: the park of the chat parked on the slot (0: none)
};

} // namespace g6s2
#endif
