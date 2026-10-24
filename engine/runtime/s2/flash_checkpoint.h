// Flash session checkpoints (G6_FLASH_CACHE), the host half: no CUDA. flash_checkpoint.cuh copies the bytes.
//
// GLM-5.3-Flash keeps two kinds of state for a sequence: the paged KV of its 12 attention bindings (11 DSA layers and
// the MTP layer, rows by position) and the KDA state of its 34 linear-attention layers (fp32 S and the conv history,
// one value after the last row consumed, the same size at any context). A page-only hit is not a hit: the KDA state
// after p rows cannot be rebuilt from the pages. So a checkpoint is one object: the pages [0, ceil(p / 64)), the KDA state
// after rows [0, p) and a manifest, stored in one contiguous block range of one store and dropped as one.
//
// aligned (the mode this file serves first): p is a point of the cold prefill grid, G = --prefill rows x lanes (the
// rows of one full prefill quantum, scheduler.h plan). A cold prompt is prefilled in quanta [0, G), [G, 2G), .. and a
// checkpoint is taken after the quantum that ends at the prompt's last grid point (aligned_point). A later prompt that
// starts with the checkpoint's tokens [0, p] skips [0, p) and is prefilled from p in the same quanta a cold prefill
// runs, on the same KDA state and pages: every row it computes is the cold row (bit-exact against no cache). Two
// conditions make that true: the KDA prefill is chunked by quantum (engine/kda/prefill.h), so p must be a quantum
// boundary; and the MTP layer's row p - 1 reads token p, so the prompt must match through token p (the MTP rule of
// prefix_reuse.h), which is why an entry keeps p + 1 tokens.
//
// terminal (G6_FLASH_CACHE=terminal, NOT cold-exact): aligned's checkpoints, and at the end of every request one more at
// p = the rows the request consumed (prompt + output, the last output token still pending), with that pending token as
// token p. Its pages may end mid-page (the whole last page is kept). The KDA state is the one decode left: decode
// advances it one row at a time (engine/kda/kda.cuh) and the prefill in chunks of 64 rows (engine/kda/kda_prefill.cuh),
// two orders of fp32 arithmetic, so it is not the state a cold prefill of the same tokens gives, and the prefill from p
// runs quanta a cold prefill does not. Its effect on the output is measured, not argued. The MTP layer's row p - 1 may
// be the one decode wrote before the last token or none: it changes only which drafts are accepted.
//
// Not the full DESIGN M5: checkpoints are not entries of the pool's block stash (kvman-rs stash.rs). Every rank keeps
// the same index (the same captures in the same order, the same sizes), and the index digest joins the pool digest
// every rank compares.
//
// A group lease (--kv-store ...,group=, s2/kv_share.h, D-331 M8 (c)): the store shares its KV node with other clusters
// (the full model's pool) and changes size by whole segments at a Resize event. The index reports what the pool stash
// reports: its counters as g6kv_tier_totals (share_want asks from them) and the ages of what a give would drop
// (coldest). A shrink drops the least recently used entries until the rest fit, then packs them below the new end:
// the lease gives its top segments back.
#pragma once
#include "../cache_namespace.h"
#include "../../kvman-rs/g6kvman.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6s2 {
// terminal: aligned, and also a checkpoint at the end of every request (below). Not cold-exact.
enum class FlashCache { Off, Aligned, Terminal };
inline FlashCache flash_cache_mode(const char* v) {
    const std::string s = v ? v : "off";
    if (s == "off" || s == "0") return FlashCache::Off;
    if (s == "aligned") return FlashCache::Aligned;
    if (s == "terminal") return FlashCache::Terminal;
    throw std::runtime_error("G6_FLASH_CACHE=" + s + ": off, aligned or terminal");
}
inline const char* flash_cache_name(FlashCache m) {
    return m == FlashCache::Terminal ? "terminal" : m == FlashCache::Aligned ? "aligned" : "off";
}

// the last cold prefill quantum boundary at or before the prompt's prefilled rows (prompt_tokens - 1: the last prompt
// token runs as the prompt-tail step). 0 = none. A quantum is full only while more than one lane's rows are left, so
// every quantum but the last starts and ends on the grid.
inline uint32_t aligned_point(uint32_t prompt_tokens, uint32_t grid) {
    if (!grid || prompt_tokens < 1) return 0;
    return (prompt_tokens - 1) / grid * grid;
}

// a 64-bit mix of one word at its index: the checksum of an extent is the wrapping sum of mix(word, index) over its
// 8-byte words, so a GPU reduction and this loop agree in any order. A word moved, lost or changed changes the sum.
inline uint64_t ckpt_mix(uint64_t w, uint64_t i) {
    uint64_t x = w ^ (i * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull);
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ull; x ^= x >> 33;
    return x;
}
inline uint64_t ckpt_sum_host(const void* p, size_t bytes, uint64_t first_word = 0) {
    const auto* w = static_cast<const uint64_t*>(p);
    uint64_t s = 0;
    for (size_t i = 0; i < bytes / 8; ++i) s += ckpt_mix(w[i], first_word + i);
    return s;
}
inline uint64_t fnv64(const void* p, size_t n, uint64_t h = 14695981039346656037ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// The first block of an entry, written after its data: a block whose manifest is missing, from another entry, or not
// complete is not a checkpoint (a torn write, a reused range).
struct FlashManifest {
    static constexpr uint64_t MAGIC = 0x31544b4348534c46ull;   // "FLSHCKT1"
    static constexpr uint32_t FORMAT = 1;                     // FLASH_RUN_FORMAT: bump when what an entry holds changes
    uint64_t magic = 0;
    uint32_t format = 0, complete = 0;
    uint64_t generation = 0;         // the entry's id, the same on every rank
    uint64_t compat = 0;             // fnv64 of the store's compat line (model set, TP, rank, layout, engine build)
    uint64_t position = 0;           // p: rows [0, p) are in the pages and the KDA state is after them
    uint64_t tokens = 0, token_hash = 0, ns_hash = 0;   // the p + 1 tokens and their namespace
    uint64_t pages = 0, page_bytes = 0, page_sum = 0;   // pages [0, pages) of every binding, their checksum
    uint64_t state_bytes = 0, state_sum = 0;            // the KDA state after rows [0, p), its checksum
    uint64_t self_sum = 0;           // fnv64 of every field above
    uint64_t own_sum() const { return fnv64(this, offsetof(FlashManifest, self_sum)); }
    void seal() { magic = MAGIC; format = FORMAT; complete = 1; self_sum = own_sum(); }
    // why this block is not the manifest of `want` (nullptr: it is)
    const char* differs(const FlashManifest& want) const {
        if (magic != MAGIC || format != FORMAT) return "no manifest";
        if (self_sum != own_sum()) return "manifest checksum";
        if (!complete) return "incomplete";
        if (generation != want.generation) return "another entry";
        if (compat != want.compat) return "another engine or layout";
        if (position != want.position || tokens != want.tokens || token_hash != want.token_hash || ns_hash != want.ns_hash)
            return "other tokens";
        if (pages != want.pages || page_bytes != want.page_bytes || state_bytes != want.state_bytes) return "other sizes";
        return nullptr;
    }
};
static_assert(sizeof(FlashManifest) % 8 == 0, "manifest words");

struct FlashEntry {
    uint64_t generation = 0, used = 0;   // used: the index clock at capture or the last hit (least recently used goes first)
    uint64_t used_ms = 0;                // FlashIndex::now_ms at the same moment: the ages a head reports (not digested)
    uint64_t base = 0, blocks = 0;       // the store's blocks [base, base + blocks): manifest, pages, state
    uint32_t position = 0;
    bool terminal = false;               // kept at the end of a request (G6_FLASH_CACHE=terminal), not on the grid
    std::vector<int> tokens;             // position + 1
    g6rt::CacheNamespace ns{};
    FlashManifest manifest;              // what the entry's first block must hold (sealed)
};
struct FlashHit { bool hit = false; uint32_t position = 0; uint64_t generation = 0; int lcp = 0; const char* why = "none"; };

// The entries of one store: contiguous block ranges, first fit, the least recently used dropped until a new entry fits.
class FlashIndex {
    uint64_t blocks_ = 0, clock_ = 0, next_ = 1;
    std::vector<FlashEntry> e_;   // by base
    static int lcp(const std::vector<int>& a, const std::vector<int>& b) {
        const size_t n = std::min(a.size(), b.size());
        return int(std::mismatch(a.begin(), a.begin() + n, b.begin()).first - a.begin());
    }
    bool gap(uint64_t need, uint64_t& at) const {
        uint64_t from = 0;
        for (const auto& x : e_) { if (x.base - from >= need) { at = from; return true; } from = x.base + x.blocks; }
        if (blocks_ - from >= need) { at = from; return true; }
        return false;
    }
    void drop_lru(std::vector<uint64_t>* gone) {
        auto lru = std::min_element(e_.begin(), e_.end(), [](const FlashEntry& a, const FlashEntry& b) { return a.used < b.used; });
        if (gone) gone->push_back(lru->generation);
        dropped_blocks += lru->blocks;
        e_.erase(lru); ++dropped;
    }
public:
    uint64_t dropped = 0, hits = 0, captured = 0;
    // for a group lease's report (totals): blocks dropped and captured, captures that had to drop (full), and entries
    // bigger than the whole store (oversize, the last one's blocks)
    uint64_t dropped_blocks = 0, captured_blocks = 0, full = 0, oversize = 0, oversize_last = 0;
    // the clock of used_ms (tests set their own)
    std::function<uint64_t()> now_ms = [] {
        return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    explicit FlashIndex(uint64_t blocks = 0) : blocks_(blocks) {}
    uint64_t blocks() const { return blocks_; }
    size_t size() const { return e_.size(); }
    const std::vector<FlashEntry>& entries() const { return e_; }
    const FlashEntry* find(uint64_t generation) const {
        for (const auto& x : e_) if (x.generation == generation) return &x;
        return nullptr;
    }
    // the longest entry whose tokens [0, p] start the prompt, with p <= limit (rows the prefill may skip) and the same
    // namespace. Ties cannot happen at one p with different tokens; at one p and the same tokens the newest wins.
    FlashHit match(const std::vector<int>& prompt, int limit, const g6rt::CacheNamespace& ns) const {
        FlashHit h;
        for (const auto& x : e_) {
            const int l = lcp(prompt, x.tokens);
            h.lcp = std::max(h.lcp, l);
            if (x.ns != ns || l < int(x.tokens.size()) || int(x.position) > limit) continue;
            if (!h.hit || x.position > h.position || (x.position == h.position && x.generation > h.generation))
                h = FlashHit{true, x.position, x.generation, h.lcp, "hit"};
        }
        if (!h.hit) h.why = e_.empty() ? "empty" : "no-match";
        return h;
    }
    // an entry with these tokens is already kept (the same prefix captured again): mark it used instead
    const FlashEntry* same(const std::vector<int>& tokens, const g6rt::CacheNamespace& ns) const {
        for (const auto& x : e_) if (x.ns == ns && x.tokens == tokens) return &x;
        return nullptr;
    }
    void touch(uint64_t generation) {
        for (auto& x : e_) if (x.generation == generation) { x.used = ++clock_; x.used_ms = now_ms(); ++hits; return; }
        throw std::runtime_error("Flash checkpoint: no entry " + std::to_string(generation));
    }
    // room for `need` blocks, dropping the least recently used entries; the dropped generations in `gone`. false: the
    // store is smaller than `need` (nothing dropped)
    bool reserve(uint64_t need, uint64_t& at, std::vector<uint64_t>* gone = nullptr) {
        if (need > blocks_) { ++oversize; oversize_last = need; }
        if (!need || need > blocks_) return false;
        bool dropped_any = false;
        while (!gap(need, at)) { drop_lru(gone); dropped_any = true; }
        if (dropped_any) ++full;
        return true;
    }
    uint64_t next_generation() const { return next_; }
    // publish an entry whose blocks are written (reserve gave `base`)
    void add(FlashEntry x) {
        if (x.generation != next_) throw std::runtime_error("Flash checkpoint: generation out of order");
        ++next_; x.used = ++clock_; x.used_ms = now_ms(); ++captured; captured_blocks += x.blocks;
        auto at = std::lower_bound(e_.begin(), e_.end(), x.base, [](const FlashEntry& a, uint64_t b) { return a.base < b; });
        if ((at != e_.end() && x.base + x.blocks > at->base) || (at != e_.begin() && std::prev(at)->base + std::prev(at)->blocks > x.base))
            throw std::runtime_error("Flash checkpoint: overlapping entries");
        e_.insert(at, std::move(x));
    }
    void drop(uint64_t generation) {
        for (auto it = e_.begin(); it != e_.end(); ++it)
            if (it->generation == generation) { dropped_blocks += it->blocks; e_.erase(it); ++dropped; return; }
    }
    uint64_t used_blocks() const { uint64_t n = 0; for (const auto& x : e_) n += x.blocks; return n; }
    // the counters a group lease's head reports from (s2/kv_share.h share_want), in the pool tier's form: a capture is
    // a spill, a dropped entry dropped pages, a capture that had to drop one a full tier
    g6kv_tier_totals totals() const {
        g6kv_tier_totals t{};
        t.slots = blocks_; t.free_slots = blocks_ - std::min(blocks_, used_blocks()); t.entries = (uint32_t)e_.size();
        t.spilled_pages = captured_blocks; t.dropped = dropped; t.dropped_pages = dropped_blocks; t.full = full;
        t.hits = hits; t.oversize = oversize; t.oversize_last = oversize_last;
        return t;
    }
    // what giving i + 1 segments of `per` blocks costs, at most `most` (kvman-rs remote coldest_ages, the rule the pool
    // stash reports by): UINT64_MAX while the free blocks cover it, then the age in ms of the most recently used entry a
    // give that size drops (least recently used first, as shrink drops them), so the ages never rise
    std::vector<uint64_t> coldest(uint64_t per, size_t most) const {
        per = std::max<uint64_t>(per, 1);
        const uint64_t free = blocks_ - std::min(blocks_, used_blocks()), now = now_ms();
        std::vector<uint64_t> out;
        while (out.size() < most && (out.size() + 1) * per <= free) out.push_back(std::numeric_limits<uint64_t>::max());
        std::vector<const FlashEntry*> order;
        for (const auto& x : e_) order.push_back(&x);
        std::sort(order.begin(), order.end(), [](const FlashEntry* a, const FlashEntry* b) { return a->used < b->used; });
        uint64_t seen = free, youngest = std::numeric_limits<uint64_t>::max();
        for (const FlashEntry* x : order) {
            const uint64_t end = seen + x->blocks;
            youngest = std::min(youngest, now - std::min(now, x->used_ms));
            while (out.size() < most && out.size() * per < end) out.push_back(youngest);
            seen = end;
        }
        return out;
    }
    // a group lease grew to `blocks`
    void grow(uint64_t blocks) {
        if (blocks < blocks_) throw std::runtime_error("Flash checkpoint: grow to fewer blocks");
        blocks_ = blocks;
    }
    // a group lease shrinks to `blocks`: drop the least recently used entries until the rest fit, then pack them
    // (in base order, each to the lowest free block) when one ends past the new end. The copies to run, in order, before
    // the lease gives its top blocks back (each moves its entry down, so copying a range from its first block on is safe)
    struct Move { uint64_t from = 0, to = 0, blocks = 0; };
    std::vector<Move> shrink(uint64_t blocks, std::vector<uint64_t>* gone = nullptr) {
        std::vector<Move> moves;
        if (blocks >= blocks_) { blocks_ = blocks; return moves; }
        while (used_blocks() > blocks) drop_lru(gone);
        const bool pack = std::any_of(e_.begin(), e_.end(), [&](const FlashEntry& x) { return x.base + x.blocks > blocks; });
        if (pack) {
            uint64_t at = 0;
            for (auto& x : e_) {
                if (x.base != at) { moves.push_back({x.base, at, x.blocks}); x.base = at; }
                at += x.blocks;
            }
        }
        blocks_ = blocks;
        return moves;
    }
    // every rank's index must agree: folded into the pool digest
    uint64_t digest() const {
        uint64_t h = fnv64(&next_, sizeof next_);
        h = fnv64(&clock_, sizeof clock_, h);
        h = fnv64(&blocks_, sizeof blocks_, h);   // a group lease's size: every rank applies the same Resize
        for (const auto& x : e_) {
            const uint64_t f[] = {x.generation, x.used, x.base, x.blocks, x.position, x.manifest.token_hash};
            h = fnv64(f, sizeof f, h);
        }
        return h;
    }
};
inline FlashManifest flash_manifest(uint64_t generation, uint64_t compat, uint32_t position, const std::vector<int>& tokens,
                                    const g6rt::CacheNamespace& ns, uint64_t pages, uint64_t page_bytes, uint64_t state_bytes) {
    FlashManifest m;
    m.generation = generation; m.compat = compat; m.position = position;
    m.tokens = tokens.size(); m.token_hash = fnv64(tokens.data(), tokens.size() * sizeof(int));
    m.ns_hash = fnv64(ns.data(), ns.size());
    m.pages = pages; m.page_bytes = page_bytes; m.state_bytes = state_bytes;
    return m;
}
} // namespace g6s2
