// Declarative model KV + boundary-state descriptor and C++ wrapper over `g6kv_stash` (engine/kvman-rs).
// Pure C++17 host header (no CUDA dependency) so CPU unit tests and tools share the exact layout and checksums
// that `kv_state_mover.cuh` runs on the device.
//
// Three model shapes fit one descriptor:
//   1. Pure paged attention: `page_planes` non-empty, `tail_extents` empty (`state_blocks() == 0` when grid-aligned).
//   2. Hybrid attention + rolling/recurrent state (DeepSeek-V4.1 MLA+DSA+Conv1d+Engram, Flash KDA+Conv, Qwen3-Next):
//      full 64-token pages `[0, pos / 64)` use `Part::Full`, while partial-page rows `[ (pos / 64) * 64, pos )` plus
//      `tail_extents` pack into `state_blocks()` self-checksummed `Part::State(0..K)` blocks.
//   3. Pure recurrent / SSM models: `page_planes` empty, `tail_extents` non-empty (`0` full pages + `State(0..K)`).
#pragma once

#include "../../kvman-rs/g6kvman.h"
#include "../prefix_reuse.h"
#include "kv_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6s2 {

inline constexpr uint32_t KV_STATE_PAGE_TOKENS = 64;
inline constexpr uint64_t KV_STATE_BLOCK_MAGIC = 0x3153564B38524747ull; // "GGR8KVS1"
inline constexpr uint32_t KV_STATE_BLOCK_VERSION = 1;

// Encodes `(slot, page_in_slot)` into `g6kv_io::page` for `Part::Full`.
inline constexpr uint32_t pack_page_handle(uint32_t slot, uint32_t page_in_slot, uint32_t max_pages_per_slot) {
    return slot * max_pages_per_slot + page_in_slot;
}

inline constexpr void unpack_page_handle(uint32_t handle, uint32_t max_pages_per_slot, uint32_t& slot, uint32_t& page_in_slot) {
    slot = max_pages_per_slot ? (handle / max_pages_per_slot) : 0;
    page_in_slot = max_pages_per_slot ? (handle % max_pages_per_slot) : handle;
}

// Encodes `(slot, valid_pos)` into `g6kv_io::page` for `Part::State(sub)` (up to 256 slots and 16M tokens).
inline uint32_t pack_state_handle(uint32_t slot, uint32_t pos) {
    if (slot >= 256u || pos > 0x00FF'FFFFu) {
        throw std::runtime_error("KV state handle out of range");
    }
    return (slot << 24) | (pos & 0x00FF'FFFFu);
}

inline constexpr void unpack_state_handle(uint32_t handle, uint32_t& slot, uint32_t& pos) {
    slot = handle >> 24;
    pos = handle & 0x00FF'FFFFu;
}

// One position-indexed plane per slot (planar sub-arrays such as `ica` codes and `ica` scales are simply two planes
// sharing `slot_stride` with different `base` and `row_bytes`).
struct KvPagePlane {
    uint8_t* base = nullptr;
    uint64_t slot_stride = 0;
    uint32_t rows_per_page = 64;
    uint32_t row_bytes = 0;
    uint32_t token_ratio = 1; // tokens per plane row (1 for 1:1, 2 for 2:1 compressed pairs)

    uint64_t page_bytes() const { return uint64_t(rows_per_page) * uint64_t(row_bytes); }
    uint32_t partial_rows(uint32_t pos) const {
        return (pos % KV_STATE_PAGE_TOKENS) / (token_ratio ? token_ratio : 1u);
    }
    uint64_t partial_bytes(uint32_t pos) const {
        return uint64_t(partial_rows(pos)) * uint64_t(row_bytes);
    }
    uint64_t max_partial_bytes() const {
        return partial_bytes(KV_STATE_PAGE_TOKENS - 1);
    }
};

// One fixed-size per-slot boundary buffer (Conv1d ring, compressor pending-half kvs/scs, Engram ring, KDA S, etc.).
struct KvTailExtent {
    uint8_t* base = nullptr;
    uint64_t slot_stride = 0;
    uint64_t bytes = 0;
};

// 32-byte header at the front of every `Part::State(sub)` block so each block is self-describing and checksummed
// independently across `kvt::next_batch` boundaries.
struct alignas(16) KvStateHeader32 {
    uint64_t magic = 0;
    uint64_t format_hash = 0;
    uint64_t payload_fnv1a = 0;
    uint32_t pos = 0; // bits [0..24): token pos, bits [24..32): version (KV_STATE_BLOCK_VERSION)
    uint16_t sub = 0;
    uint16_t slice_blocks_or_pad = 0;
};
static_assert(sizeof(KvStateHeader32) == 32, "KvStateHeader32 must be 32 bytes");

struct KvCopySpan {
    uint8_t* ptr = nullptr; // device (or simulated host) pointer
    uint32_t hslot = 0;     // staging slot index in batch
    uint32_t hoff = 0;      // byte offset within staging slot
    uint32_t len = 0;       // byte length
};

inline uint64_t kv_fnv1a_bytes(const void* data, size_t len, uint64_t h = 14695981039346656037ull) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h = (h ^ uint64_t(p[i])) * 1099511628211ull;
    }
    return h;
}

inline uint64_t kv_fnv1a_u64(uint64_t h, uint64_t v) {
    return kv_fnv1a_bytes(&v, sizeof(v), h);
}

struct KvStateDesc {
    std::string model_tag;
    uint32_t max_slots = 1;
    uint32_t max_pages_per_slot = 1;
    bool mtp_lookahead = false;
    bool allow_partial_pages = true; // true when boundary state captures `pos % 64 != 0` rows
    size_t min_block_bytes = 4096;   // fallback block size when `page_planes` is empty
    std::vector<KvPagePlane> page_planes;
    std::vector<KvTailExtent> tail_extents;

    void validate() const {
        if (max_slots == 0 || max_slots > 256) {
            throw std::runtime_error("KvStateDesc: max_slots must be in 1..256");
        }
        if (!max_pages_per_slot || uint64_t(max_slots) * max_pages_per_slot > uint64_t(UINT32_MAX) + 1 ||
            uint64_t(max_pages_per_slot) * KV_STATE_PAGE_TOKENS > 0x01000000u)
            throw std::runtime_error("KvStateDesc: page handle capacity out of range");
        // Arbitrary base/stride alignment is supported by the mover's scalar fallback.
        for (const auto& p : page_planes) {
            if (!p.base || p.row_bytes == 0 || p.token_ratio == 0 ||
                (KV_STATE_PAGE_TOKENS % p.token_ratio) != 0 ||
                p.rows_per_page != KV_STATE_PAGE_TOKENS / p.token_ratio) {
                throw std::runtime_error("KvStateDesc: invalid KvPagePlane geometry");
            }
        }
        for (const auto& t : tail_extents) {
            if (t.bytes > 0 && !t.base) {
                throw std::runtime_error("KvStateDesc: null KvTailExtent base");
            }
        }
        if (state_blocks() > UINT16_MAX || slot_bytes() > UINT32_MAX)
            throw std::runtime_error("KvStateDesc: state block geometry too large");
        if (page_planes.empty() && tail_extents.empty()) {
            throw std::runtime_error("KvStateDesc: empty descriptor");
        }
    }

    size_t page_block_bytes() const {
        size_t sum = 0;
        for (const auto& p : page_planes) sum += size_t(p.page_bytes());
        return sum;
    }

    size_t raw_block_bytes() const {
        const size_t pb = page_block_bytes();
        return pb ? pb : std::max<size_t>(min_block_bytes, 4096);
    }

    size_t slot_bytes() const {
        const size_t raw = raw_block_bytes();
        return (raw + 4095u) / 4096u * 4096u;
    }

    size_t state_payload_cap_per_block() const {
        return slot_bytes() - sizeof(KvStateHeader32);
    }

    size_t tail_bytes() const {
        size_t sum = 0;
        for (const auto& t : tail_extents) sum += size_t(t.bytes);
        return sum;
    }

    size_t partial_page_bytes(uint32_t pos) const {
        if (!allow_partial_pages) return 0;
        size_t sum = 0;
        for (const auto& p : page_planes) sum += size_t(p.partial_bytes(pos));
        return sum;
    }

    size_t state_payload_bytes(uint32_t pos) const {
        return tail_bytes() + partial_page_bytes(pos);
    }

    size_t max_state_payload_bytes() const {
        size_t sum = tail_bytes();
        if (allow_partial_pages) {
            for (const auto& p : page_planes) sum += size_t(p.max_partial_bytes());
        }
        return sum;
    }

    uint32_t state_blocks() const {
        const size_t max_p = max_state_payload_bytes();
        if (max_p == 0) return 0;
        const size_t cap = state_payload_cap_per_block();
        return uint32_t((max_p + cap - 1) / cap);
    }

    uint64_t format_hash() const {
        uint64_t h = kv_fnv1a_bytes(model_tag.data(), model_tag.size());
        h = kv_fnv1a_u64(h, KV_STATE_BLOCK_VERSION);
        h = kv_fnv1a_u64(h, slot_bytes());
        h = kv_fnv1a_u64(h, page_planes.size());
        h = kv_fnv1a_u64(h, tail_extents.size());
        h = kv_fnv1a_u64(h, mtp_lookahead ? 1 : 0);
        h = kv_fnv1a_u64(h, allow_partial_pages ? 1 : 0);
        for (const auto& p : page_planes) {
            h = kv_fnv1a_u64(h, p.rows_per_page);
            h = kv_fnv1a_u64(h, p.row_bytes);
            h = kv_fnv1a_u64(h, p.token_ratio);
        }
        for (const auto& t : tail_extents) {
            h = kv_fnv1a_u64(h, t.bytes);
        }
        return h;
    }

    // Appends the contiguous byte spans for `Part::State(sub)` at `(slot, pos)` into `out`, returning the slice byte
    // length (`0` when `sub * cap >= state_payload_bytes(pos)`).
    uint32_t append_state_spans(uint32_t slot, uint32_t pos, uint32_t sub, uint32_t hslot,
                                std::vector<KvCopySpan>& out) const {
        if (slot >= max_slots || sub >= state_blocks() || pos > 0x00FFFFFFu ||
            uint64_t(pos) > uint64_t(max_pages_per_slot) * KV_STATE_PAGE_TOKENS) {
            throw std::runtime_error("KvStateDesc: state span slot or sub out of range");
        }
        const size_t cap = state_payload_cap_per_block();
        const size_t total = state_payload_bytes(pos);
        const size_t lo = size_t(sub) * cap;
        if (lo >= total) return 0;
        const size_t hi = std::min(lo + cap, total);
        size_t cur = 0;
        uint32_t written = 0;

        auto emit = [&](uint8_t* seg_ptr, size_t seg_len) {
            if (!seg_len) return;
            const size_t seg_lo = cur;
            const size_t seg_hi = cur + seg_len;
            cur = seg_hi;
            if (seg_hi <= lo || seg_lo >= hi) return;
            const size_t take_lo = std::max(seg_lo, lo);
            const size_t take_hi = std::min(seg_hi, hi);
            const uint32_t n = uint32_t(take_hi - take_lo);
            out.push_back(KvCopySpan{
                seg_ptr + (take_lo - seg_lo),
                hslot,
                uint32_t(sizeof(KvStateHeader32) + (take_lo - lo)),
                n,
            });
            written += n;
        };

        for (const auto& t : tail_extents) {
            emit(t.base + uint64_t(slot) * t.slot_stride, size_t(t.bytes));
        }
        if (allow_partial_pages) {
            const uint64_t full_idx = pos / KV_STATE_PAGE_TOKENS;
            for (const auto& p : page_planes) {
                const size_t pbytes = size_t(p.partial_bytes(pos));
                emit(p.base + uint64_t(slot) * p.slot_stride + full_idx * p.page_bytes(), pbytes);
            }
        }
        return written;
    }

    void seal_state_block_header(uint8_t* block_buf, uint32_t pos, uint32_t sub, uint32_t slice_len) const {
        const size_t sb = slot_bytes();
        const size_t used = sizeof(KvStateHeader32) + size_t(slice_len);
        if (used < sb) {
            std::memset(block_buf + used, 0, sb - used);
        }
        KvStateHeader32 hdr{};
        hdr.magic = KV_STATE_BLOCK_MAGIC;
        hdr.format_hash = format_hash();
        hdr.payload_fnv1a = kv_fnv1a_bytes(block_buf + sizeof(KvStateHeader32), slice_len);
        hdr.pos = (pos & 0x00FF'FFFFu) | (KV_STATE_BLOCK_VERSION << 24);
        hdr.sub = uint16_t(sub);
        hdr.slice_blocks_or_pad = uint16_t(state_blocks());
        std::memcpy(block_buf, &hdr, sizeof(hdr));
    }

    void verify_state_block_header(const uint8_t* block_buf, uint32_t pos, uint32_t sub, uint32_t expected_slice_len) const {
        KvStateHeader32 hdr{};
        std::memcpy(&hdr, block_buf, sizeof(hdr));
        const uint32_t got_ver = hdr.pos >> 24;
        const uint32_t got_pos = hdr.pos & 0x00FF'FFFFu;
        if (hdr.magic != KV_STATE_BLOCK_MAGIC ||
            got_ver != KV_STATE_BLOCK_VERSION ||
            hdr.format_hash != format_hash() ||
            got_pos != (pos & 0x00FF'FFFFu) ||
            hdr.sub != uint16_t(sub) ||
            hdr.slice_blocks_or_pad != uint16_t(state_blocks())) {
            throw std::runtime_error("KV state block header mismatch or corrupt metadata");
        }
        const uint64_t sum = kv_fnv1a_bytes(block_buf + sizeof(KvStateHeader32), expected_slice_len);
        if (sum != hdr.payload_fnv1a) {
            throw std::runtime_error("KV state block payload checksum mismatch");
        }
    }

    // Pure-host pack/unpack helpers (used by host unit tests and CPU verification).
    void pack_page_host(uint32_t page_handle, uint8_t* dst) const {
        uint32_t slot = 0, p = 0;
        unpack_page_handle(page_handle, max_pages_per_slot, slot, p);
        if (slot >= max_slots || p >= max_pages_per_slot) {
            throw std::runtime_error("KvStateDesc::pack_page_host: handle out of range");
        }
        size_t off = 0;
        for (const auto& pl : page_planes) {
            const size_t n = size_t(pl.page_bytes());
            const uint8_t* src = pl.base + uint64_t(slot) * pl.slot_stride + uint64_t(p) * n;
            std::memcpy(dst + off, src, n);
            off += n;
        }
        const size_t sb = slot_bytes();
        if (off < sb) std::memset(dst + off, 0, sb - off);
    }

    void unpack_page_host(uint32_t page_handle, const uint8_t* src) const {
        uint32_t slot = 0, p = 0;
        unpack_page_handle(page_handle, max_pages_per_slot, slot, p);
        if (slot >= max_slots || p >= max_pages_per_slot) {
            throw std::runtime_error("KvStateDesc::unpack_page_host: handle out of range");
        }
        size_t off = 0;
        for (const auto& pl : page_planes) {
            const size_t n = size_t(pl.page_bytes());
            uint8_t* dst = pl.base + uint64_t(slot) * pl.slot_stride + uint64_t(p) * n;
            std::memcpy(dst, src + off, n);
            off += n;
        }
    }

    void pack_state_block_host(uint32_t state_handle, uint32_t sub, uint8_t* dst) const {
        uint32_t slot = 0, pos = 0;
        unpack_state_handle(state_handle, slot, pos);
        std::vector<KvCopySpan> spans;
        const uint32_t slice_len = append_state_spans(slot, pos, sub, 0, spans);
        for (const auto& sp : spans) {
            std::memcpy(dst + sp.hoff, sp.ptr, sp.len);
        }
        seal_state_block_header(dst, pos, sub, slice_len);
    }

    void unpack_state_block_host(uint32_t state_handle, uint32_t sub, const uint8_t* src) const {
        uint32_t slot = 0, pos = 0;
        unpack_state_handle(state_handle, slot, pos);
        std::vector<KvCopySpan> spans;
        const uint32_t slice_len = append_state_spans(slot, pos, sub, 0, spans);
        verify_state_block_header(src, pos, sub, slice_len);
        for (const auto& sp : spans) {
            std::memcpy(sp.ptr, src + sp.hoff, sp.len);
        }
    }
};

// RAII C++ wrapper over `g6kv_stash*` (mirrors `RunPool` in `pool.h` for slot-slab / boundary-state engines).
class StashPool {
    g6kv_stash* s_ = nullptr;

    static void ck(int rc) {
        if (rc) throw std::runtime_error(g6kv_error(rc));
    }

public:
    StashPool(uint32_t shared_after, uint64_t grid, uint32_t state_blocks, bool mtp) {
        ck(g6kv_stash_new(shared_after, grid, state_blocks, mtp ? 1 : 0, &s_));
    }
    ~StashPool() { g6kv_stash_free(s_); }
    StashPool(const StashPool&) = delete;
    StashPool& operator=(const StashPool&) = delete;

    void set_rules(bool strict, bool noend, int32_t grid) {
        g6kv_stash_rules(s_, strict ? 1 : 0, noend ? 1 : 0, grid);
    }

    void add_tier(uint64_t slots, int keep, const char* index_dir = nullptr, const char* compat = nullptr) {
        if (g6kv_stash_add_tier(s_, slots, keep, index_dir, compat)) {
            throw std::runtime_error(std::string("KV stash index: ") + g6kv_tier_error());
        }
    }

    uint64_t tiers_loaded() { return g6kv_stash_tiers_loaded(s_); }

    void set_policy(uint32_t t, uint64_t min_rows, uint32_t min_hits, uint64_t block_bytes) {
        ck(g6kv_stash_tier_policy(s_, t, min_rows, min_hits, block_bytes));
    }

    void write_limit(uint64_t milli) { g6kv_stash_write_limit(s_, milli); }
    void earn(uint64_t pages) { g6kv_stash_earn(s_, pages); }

    uint32_t count(const std::vector<int>& prompt, int64_t span, const g6rt::CacheNamespace& ns,
                   bool resident_continuing = false) {
        uint32_t out = 0;
        ck(g6kv_stash_count(s_, prompt.data(), prompt.size(), span, ns.data(), ns.size(),
                            resident_continuing ? 1 : 0, &out));
        return out;
    }

    g6kv_hit match(const std::vector<int>& prompt, int limit, const g6rt::CacheNamespace& ns) const {
        g6kv_hit h{};
        ck(g6kv_stash_match(s_, prompt.data(), prompt.size(), limit, ns.data(), ns.size(), &h));
        return h;
    }

    struct Info {
        uint32_t tier = 0, hits = 0;
        int32_t valid = 0, pf_end = 0;
        size_t pages = 0;
    };
    Info info(uint64_t id) const {
        Info x;
        ck(g6kv_stash_info(s_, id, &x.tier, &x.valid, &x.pf_end, &x.hits, &x.pages));
        return x;
    }

    struct Restored {
        int32_t valid = 0;
        uint32_t hits = 0;
        size_t full_pages = 0;
    };

    Restored restore(uint32_t tier, uint64_t id, const std::vector<uint32_t>& pages, uint32_t state_handle) {
        Restored r{};
        ck(g6kv_stash_restore(s_, tier, id, pages.data(), pages.size(), state_handle, &r.valid, &r.hits, &r.full_pages));
        return r;
    }

    uint64_t capture(const std::vector<int>& history, int32_t valid, int32_t written_end, int32_t pf_end,
                     const std::vector<uint32_t>& pages, uint32_t state_handle, uint32_t hits,
                     const g6rt::CacheNamespace& ns, uint64_t from_id = 0, size_t from_pages = 0,
                     bool keep_from = false, uint64_t keep_id = 0,
                     const std::vector<const std::vector<int>*>& resident = {}) {
        std::vector<const int32_t*> ptrs;
        std::vector<size_t> lens;
        ptrs.reserve(resident.size());
        lens.reserve(resident.size());
        for (const auto* r : resident) {
            if (r && !r->empty()) {
                ptrs.push_back(r->data());
                lens.push_back(r->size());
            }
        }
        uint64_t id = 0;
        ck(g6kv_stash_capture(s_, history.data(), history.size(), valid, written_end, pf_end, pages.data(), pages.size(),
                              state_handle, hits, ns.data(), ns.size(), from_id, from_pages,
                              keep_from ? 1 : 0, keep_id, ptrs.data(), lens.data(), ptrs.size(), &id));
        return id;
    }

    void promote(const std::vector<int>& history, int32_t valid, int32_t written_end, int32_t pf_end,
                 const std::vector<uint32_t>& pages, uint32_t state_handle, uint64_t rows_max, int64_t span = -1,
                 uint64_t keep_id = 0) {
        ck(g6kv_stash_promote(s_, history.data(), history.size(), valid, written_end, pf_end, pages.data(), pages.size(),
                              state_handle, rows_max, span, keep_id));
    }

    bool drop_entry(uint64_t id) {
        int dropped = 0;
        ck(g6kv_stash_drop(s_, id, &dropped));
        return dropped != 0;
    }

    bool has(uint64_t id) const { return g6kv_stash_has(s_, id) != 0; }

    std::vector<g6kv_io> io() {
        const size_t n = g6kv_stash_io_pending(s_);
        if (!n) return {};
        std::vector<g6kv_io> v(n);
        ck(g6kv_stash_io_take(s_, v.data(), n));
        return v;
    }

    void io_done() {
        if (g6kv_stash_io_done(s_)) {
            fprintf(stderr, "# s2 stash index write failed (continuing without those entries on next boot): %s\n",
                    g6kv_tier_error());
        }
    }

    g6kv_tier_totals tier_stats(uint32_t t) const {
        g6kv_tier_totals x{};
        ck(g6kv_stash_tier_stats(s_, t, &x));
        return x;
    }

    std::string stored_summary() const {
        const size_t n = g6kv_stash_stored_summary(s_, nullptr, 0);
        std::string s(n ? n - 1 : 0, '\0');
        if (n > 1) g6kv_stash_stored_summary(s_, &s[0], n);
        return s;
    }

    void keep_agreed(const std::string& agreed) { g6kv_stash_keep_agreed(s_, agreed.c_str()); }

    bool resize_tier(uint32_t t, uint64_t slots) {
        int32_t ok = 0;
        ck(g6kv_stash_tier_resize(s_, t, slots, &ok));
        return ok != 0;
    }

    std::vector<uint64_t> coldest_ages(uint32_t t, uint64_t per_seg, size_t most = 4096) {
        std::vector<uint64_t> a(most);
        a.resize(g6kv_stash_tier_coldest(s_, t, per_seg, a.data(), most));
        return a;
    }

    uint64_t digest() const { return g6kv_stash_digest(s_); }
};

// FNV-1a of this process's executable, hex: two builds that may compute different KV bits get different stamps.
inline std::string exe_stamp() {
    FILE* f = fopen("/proc/self/exe", "rb");
    if (!f) return "unknown";
    uint64_t h = 14695981039346656037ull;
    std::vector<unsigned char> buf(1 << 20);
    for (size_t n; (n = fread(buf.data(), 1, buf.size(), f)) > 0;)
        for (size_t i = 0; i < n; ++i) h = (h ^ buf[i]) * 1099511628211ull;
    fclose(f);
    char s[17];
    snprintf(s, sizeof s, "%016llx", (unsigned long long)h);
    return s;
}

// One --kv-store value: STORE,size=N(M|G|T)[,persist=on|off][,keep=system|sessions|all]
// [,spare_ram_cache=on|off[,spare_ram_floor=N(M|G)]][,group=NAME[,cap=N(M|G|T)][,weight=N]].
struct TierSpec {
    std::string store;
    uint64_t bytes = 0;
    int keep = 0;       // --kv-tier only: 0 evicted, 1 shared, 2 all
    std::string rule;   // --kv-store keep=: system, sessions or all; empty = --kv-keep
    bool persist = false;
    bool spare_ram = false;
    uint64_t spare_ram_floor = 2ull << 30;
    std::string group;
    uint64_t cap = 0;
    uint32_t weight = 0;
    static TierSpec parse(const std::string& v, bool old = false) {
        TierSpec t;
        size_t at = v.find(',');
        t.store = v.substr(0, at);
        while (at != std::string::npos) {
            const size_t next = v.find(',', at + 1);
            const std::string kv = v.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
            const size_t eq = kv.find('=');
            const std::string k = kv.substr(0, eq), x = eq == std::string::npos ? "" : kv.substr(eq + 1);
            auto bytes = [&] {
                char* end = nullptr;
                const double n = strtod(x.c_str(), &end);
                const std::string unit = end ? end : "";
                const double mul = unit == "T" ? 1099511627776.0 : unit == "G" ? 1073741824.0 : unit == "M" ? 1048576.0 : 0.0;
                if (!(n > 0) || !mul) throw std::runtime_error("--kv-store " + v + ": " + k + "=N with M, G or T");
                return uint64_t(n * mul);
            };
            if (k == "size") {
                t.bytes = bytes();
            } else if (k == "cap") {
                t.cap = bytes();
            } else if (k == "group") {
                if (x.empty()) throw std::runtime_error("--kv-store " + v + ": group=NAME");
                t.group = x;
            } else if (k == "weight") {
                char* end = nullptr;
                const long w = strtol(x.c_str(), &end, 10);
                if (x.empty() || *end || w < 1 || w > 1000000) throw std::runtime_error("--kv-store " + v + ": weight=N, 1000 = 1x");
                t.weight = uint32_t(w);
            } else if (k == "spare_ram_cache") {
                if (x != "on" && x != "off") throw std::runtime_error("--kv-store " + v + ": spare_ram_cache=on or off");
                t.spare_ram = x == "on";
            } else if (k == "spare_ram_floor") {
                t.spare_ram_floor = bytes();
            } else if (k == "persist") {
                if (x != "on" && x != "off") throw std::runtime_error("--kv-store " + v + ": persist=on or off");
                t.persist = x == "on";
            } else if (k == "keep" && !old) {
                if (x != "system" && x != "sessions" && x != "all")
                    throw std::runtime_error("--kv-store " + v + ": keep=system, sessions or all");
                t.rule = x;
            } else if (k == "keep") {
                if (x == "evicted") t.keep = 0; else if (x == "shared") t.keep = 1; else if (x == "all") t.keep = 2;
                else throw std::runtime_error("--kv-tier " + v + ": keep=evicted, shared or all");
            } else throw std::runtime_error("--kv-store " + v + ": unknown option " + k);
            at = next;
        }
        const bool ssd = t.store.rfind("ssd:", 0) == 0 && t.store.size() > 4;
        const bool remote = t.store.rfind("remote:", 0) == 0 && t.store.size() > 7 && t.store != "remote:auto";
        if (!ssd && !remote && t.store != "mem:")
            throw std::runtime_error("--kv-store " + v + ": the store is ssd:PATH or remote:HOST:PORT (remote:auto needs the launcher)");
        if (!t.bytes) throw std::runtime_error("--kv-store " + v + ": size=N(G|T) is needed");
        if (t.persist && !ssd) throw std::runtime_error("--kv-store " + v + ": persist=on is for ssd: stores");
        if (!t.group.empty() && !remote) throw std::runtime_error("--kv-store " + v + ": group= is for remote: stores");
        if (t.spare_ram && !t.group.empty())
            throw std::runtime_error("--kv-store " + v + ": spare_ram_cache=on needs a fixed lease (no group=)");
        if (t.group.empty() && (t.cap || t.weight)) throw std::runtime_error("--kv-store " + v + ": cap= and weight= need group=");
        if (t.cap && t.cap < t.bytes) throw std::runtime_error("--kv-store " + v + ": cap= is below size=");
        return t;
    }
    bool ssd() const { return store.rfind("ssd:", 0) == 0; }
};

} // namespace g6s2
