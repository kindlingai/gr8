// Pure host-only slab pool and byte-range router for EL3 (#151) piece tables.
// Zero CUDA dependency so engine/kernels/tests/ptab_pool_test.cpp can exhaustively exercise chunk-level accounting,
// contiguous-per-expert allocation (take_run), individual piece recycling (give), empty-chunk unmap/reclaim, and
// direct-load segment routing under ASan+UBSan on CPU.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ptab
{

constexpr size_t SLAB_ALIGN = 256;         // every slab starts on a 256 B boundary (the widest load is 16 B cp.async)

// Slab pool over chunked device memory (Backend allocates/frees chunks; CUDA uses g6vmm::alloc or cudaMalloc).
// Tracks live slab count & live bytes per chunk so reclaim_empty() can unmap and return empty chunks to unified RAM
// / the EL5 KV pool after a width switch or EL3b compaction pass.
// take_run() places all `count` slabs of one expert contiguously inside one chunk (matching mode-1 DRAM/L2 locality)
// while keeping every slab independently freeable via give(ptr, bytes).
template <class Backend>
struct SlabPool
{
    size_t chunk_bytes = size_t(256) << 20;   // default chunk; a slab or run larger than it gets a chunk of its own size
    bool vmm = false;                         // chunks from VMM backend instead of plain alloc
    struct Chunk
    {
        uint8_t* p = nullptr;
        size_t bytes = 0, used = 0;
        size_t live_bytes = 0, live_slabs = 0;
        bool vmm = false;
        bool alive = false;
    };
    std::vector<Chunk> chunks;
    std::map<uintptr_t, size_t> chunk_by_end; // (chunk_base + chunk_bytes) -> index in chunks; O(log C) chunk_of
    std::map<size_t, std::vector<uint8_t*>> free_by_size;
    size_t held = 0, used = 0, freed = 0;     // bytes: in alive chunks, in live slabs, in the free lists
    int cur = -1;                             // the chunk new slabs are carved from
    int vmm_fallbacks = 0;
    size_t chunks_reclaimed = 0;
    size_t bytes_reclaimed = 0;

    SlabPool() = default;
    SlabPool(const SlabPool&) = delete;
    SlabPool& operator=(const SlabPool&) = delete;
    ~SlabPool() { release(); }

    static size_t round(size_t n) { return (n + SLAB_ALIGN - 1) / SLAB_ALIGN * SLAB_ALIGN; }

    // Index of the alive chunk containing `p`, or -1 if not found.
    int chunk_of(const uint8_t* p) const
    {
        if (!p) return -1;
        const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        auto it = chunk_by_end.upper_bound(addr);
        if (it == chunk_by_end.end()) return -1;
        const size_t idx = it->second;
        const Chunk& c = chunks[idx];
        if (!c.alive) return -1;
        const uintptr_t base = reinterpret_cast<uintptr_t>(c.p);
        return (addr >= base && addr < base + c.bytes) ? static_cast<int>(idx) : -1;
    }

    size_t live_chunks() const
    {
        size_t n = 0;
        for (const Chunk& c : chunks) n += c.alive ? 1 : 0;
        return n;
    }

    // Allocates a chunk of at least `at_least` bytes; returns its index in chunks, or -1 + err.
    // An exact-size chunk for an oversize slab/run (at_least > chunk_bytes) is not made the current chunk.
    int grow(size_t at_least, const char*& err)
    {
        err = nullptr;
        const bool oversize = at_least > chunk_bytes;
        Chunk c{};
        c.bytes = oversize ? at_least : chunk_bytes;
        if (!Backend::alloc_chunk(c.bytes, vmm, c.p, c.bytes, c.vmm, vmm_fallbacks, err)) return -1;
        c.alive = true;
        const size_t idx = chunks.size();
        chunks.push_back(c);
        chunk_by_end[reinterpret_cast<uintptr_t>(c.p) + c.bytes] = idx;
        held += c.bytes;
        if (!oversize) cur = static_cast<int>(idx);
        return static_cast<int>(idx);
    }

    // A single slab of `n` bytes (rounded up to SLAB_ALIGN); nullptr + err on failure.
    // fresh: skip free_by_size and bump-allocate from the current chunk.
    uint8_t* take(size_t n, const char*& err, bool fresh = false)
    {
        err = nullptr;
        n = round(n);
        auto it = free_by_size.find(n);
        if (!fresh && it != free_by_size.end() && !it->second.empty())
        {
            uint8_t* p = it->second.back();
            it->second.pop_back();
            freed -= n;
            used += n;
            const int ci = chunk_of(p);
            if (ci >= 0) { ++chunks[ci].live_slabs; chunks[ci].live_bytes += n; }
            return p;
        }
        int i = cur;
        if (i < 0 || !chunks[i].alive || chunks[i].used + n > chunks[i].bytes)
            if ((i = grow(n, err)) < 0) return nullptr;
        Chunk& c = chunks[i];
        uint8_t* p = c.p + c.used;
        c.used += n;
        ++c.live_slabs;
        c.live_bytes += n;
        used += n;
        return p;
    }

    // Allocates `count` slabs contiguously inside one chunk (all blocks of one expert's fc1 or fc2 sit back-to-back
    // in memory for mode-1 locality), while tracking each slab individually so any piece can later be freed via give().
    // Zero-sized entries get nullptr. Returns true on success, false + err on failure.
    bool take_run(const size_t* sizes, int count, uint8_t** out_ptrs, const char*& err)
    {
        err = nullptr;
        if (count < 0 || (count > 0 && (!sizes || !out_ptrs))) { err = "ptab::Pool::take_run: invalid arguments"; return false; }
        size_t total = 0;
        int nonzero = 0;
        for (int b = 0; b < count; ++b)
        {
            if (sizes[b]) { total += round(sizes[b]); ++nonzero; }
        }
        if (!nonzero)
        {
            for (int b = 0; b < count; ++b) out_ptrs[b] = nullptr;
            return true;
        }
        int i = cur;
        if (i < 0 || !chunks[i].alive || chunks[i].used + total > chunks[i].bytes)
            if ((i = grow(total, err)) < 0) return false;
        Chunk& c = chunks[i];
        for (int b = 0; b < count; ++b)
        {
            if (!sizes[b]) { out_ptrs[b] = nullptr; continue; }
            const size_t rn = round(sizes[b]);
            out_ptrs[b] = c.p + c.used;
            c.used += rn;
            ++c.live_slabs;
            c.live_bytes += rn;
            used += rn;
        }
        return true;
    }

    // Returns a slab that take() or take_run() gave, with the byte count it was taken with.
    // Non-blocking: updates per-chunk live counts and puts the slab in free_by_size; call reclaim_empty() after a
    // batch of event-fenced give() calls to unmap and release any chunk whose live_slabs reached 0.
    void give(uint8_t* p, size_t n)
    {
        if (!p) return;
        n = round(n);
        const int ci = chunk_of(p);
        if (ci >= 0)
        {
            Chunk& c = chunks[ci];
            if (c.live_slabs > 0) --c.live_slabs;
            c.live_bytes = c.live_bytes >= n ? c.live_bytes - n : 0;
        }
        free_by_size[n].push_back(p);
        used = used >= n ? used - n : 0;
        freed += n;
    }

    // Releases every alive chunk with zero live slabs, pruning its free-list entries first.
    // Call only after the GPU stream has been synchronized past the last launch that read any slab in those chunks.
    // Returns bytes freed; optionally sets *freed_chunks_out to the number of chunks released.
    size_t reclaim_empty(size_t* freed_chunks_out = nullptr)
    {
        std::vector<bool> drop(chunks.size(), false);
        size_t ndrop = 0;
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            if (chunks[i].alive && chunks[i].live_slabs == 0)
            {
                drop[i] = true;
                ++ndrop;
            }
        }
        if (!ndrop)
        {
            if (freed_chunks_out) *freed_chunks_out = 0;
            return 0;
        }
        for (auto it = free_by_size.begin(); it != free_by_size.end(); )
        {
            const size_t sz = it->first;
            auto& vec = it->second;
            size_t write = 0;
            for (size_t r = 0; r < vec.size(); ++r)
            {
                const int ci = chunk_of(vec[r]);
                if (ci >= 0 && drop[static_cast<size_t>(ci)])
                {
                    freed = freed >= sz ? freed - sz : 0;
                }
                else
                {
                    vec[write++] = vec[r];
                }
            }
            vec.resize(write);
            if (vec.empty()) it = free_by_size.erase(it);
            else ++it;
        }
        size_t bytes_out = 0;
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            if (!drop[i]) continue;
            Chunk& c = chunks[i];
            chunk_by_end.erase(reinterpret_cast<uintptr_t>(c.p) + c.bytes);
            Backend::free_chunk(c.p, c.vmm);
            held = held >= c.bytes ? held - c.bytes : 0;
            bytes_out += c.bytes;
            c.p = nullptr;
            c.bytes = c.used = c.live_bytes = c.live_slabs = 0;
            c.alive = false;
            if (cur == static_cast<int>(i)) cur = -1;
        }
        chunks_reclaimed += ndrop;
        bytes_reclaimed += bytes_out;
        if (freed_chunks_out) *freed_chunks_out = ndrop;
        return bytes_out;
    }

    void release()
    {
        for (Chunk& c : chunks)
        {
            if (c.alive && c.p) Backend::free_chunk(c.p, c.vmm);
        }
        chunks.clear();
        chunk_by_end.clear();
        free_by_size.clear();
        held = used = freed = 0;
        cur = -1;
    }
};

// Recognizes routed-expert tensor names "L%03d.tr3.fc1" / "L%03d.tr3.fc2".
inline bool parse_tr3_layer_name(const std::string& name, int& layer, bool& is_fc2)
{
    if (name.size() < 12 || name[0] != 'L') return false;
    const size_t dot = name.find('.');
    if (dot == std::string::npos || dot < 2) return false;
    const std::string suffix = name.substr(dot + 1);
    if (suffix == "tr3.fc1") is_fc2 = false;
    else if (suffix == "tr3.fc2") is_fc2 = true;
    else return false;
    int val = 0;
    for (size_t i = 1; i < dot; ++i)
    {
        if (name[i] < '0' || name[i] > '9') return false;
        val = val * 10 + (name[i] - '0');
    }
    layer = val;
    return true;
}

// Sorted non-overlapping byte-range router for direct slab loading: maps logical pack offsets [pack_lo, pack_hi)
// directly to destination device pointers (either compact DevImage for dense/attention/vocab or ptab::Pool slabs for
// routed experts), skipping padding gaps.
struct RouteMap
{
    struct Seg
    {
        uint64_t pack_lo = 0, pack_hi = 0;
        uint8_t* dst = nullptr;
    };
    std::vector<Seg> segs;

    void add(uint64_t pack_lo, uint64_t nbytes, uint8_t* dst)
    {
        if (!nbytes) return;
        segs.push_back(Seg{pack_lo, pack_lo + nbytes, dst});
    }

    // Sorts segments and verifies they are strictly non-overlapping.
    bool finalize(std::string& err)
    {
        std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.pack_lo < b.pack_lo; });
        for (size_t i = 1; i < segs.size(); ++i)
        {
            if (segs[i].pack_lo < segs[i - 1].pack_hi)
            {
                err = "ptab::RouteMap: overlapping segments at pack offset " + std::to_string(segs[i].pack_lo);
                return false;
            }
        }
        return true;
    }

    // Routes chunk [off, off + n) from `src` to every overlapping segment by calling copy_fn(dst, src_slice, len).
    // Stops immediately and returns the non-zero error code if copy_fn returns non-zero.
    template <class CopyFn>
    int dispatch(uint64_t off, const uint8_t* src, size_t n, CopyFn&& copy_fn) const
    {
        if (!n || segs.empty()) return 0;
        const uint64_t end = off + n;
        // First segment with pack_hi > off
        auto it = std::upper_bound(segs.begin(), segs.end(), off,
                                   [](uint64_t x, const Seg& s) { return x < s.pack_hi; });
        for (; it != segs.end() && it->pack_lo < end; ++it)
        {
            const uint64_t lo = std::max(off, it->pack_lo);
            const uint64_t hi = std::min(end, it->pack_hi);
            if (hi <= lo) continue;
            const size_t len = static_cast<size_t>(hi - lo);
            const int rc = copy_fn(it->dst + (lo - it->pack_lo), src + (lo - off), len);
            if (rc) return rc;
        }
        return 0;
    }
};

}   // namespace ptab
