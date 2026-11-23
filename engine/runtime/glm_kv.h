// GLM-5.3 VMM-backed resizable KV cache regions (GR8_KV_CHUNKS=1, EL5 #153).
// Each layer binding's MLA cache (and index cache on full-index layers) sits in its own G6KvChunks reserved virtual
// region (engine/attn/kvchunks.h). Virtual addresses never move across shrinks/grows, so block tables, S2 batch views,
// PageBlocks (kv_tier.cuh), and captured CUDA graphs remain valid without recapture.
// Host only. Not thread safe.
#pragma once
#include "../attn/kvchunks.h"
#include "../attn/kvfmt.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace glmkv {

constexpr size_t MLA_PAGE = (size_t)64 * g6kv::ROW;
constexpr size_t IDX_PAGE = (size_t)64 * g6kv::IDX_ROW;

struct Map {
    struct Region {
        G6KvChunks* mla = nullptr;
        G6KvChunks* idx = nullptr;
        uint32_t pages = 0;
    };

    int device = 0;
    std::vector<Region> regions;
    std::string err;

    bool active() const { return !regions.empty(); }

    // Create fully backed VMM regions for one GLM binding (`pages` physical pages; `full_index` adds the index cache).
    int add(uint32_t pages, bool full_index, int dev, G6KvChunks** mla_out, G6KvChunks** idx_out) {
        if (!pages || !mla_out || !idx_out) return fail(1, "invalid arguments");
        device = dev;
        *mla_out = nullptr;
        *idx_out = nullptr;
        int e = g6_kvc_create(mla_out, (size_t)pages * MLA_PAGE, dev);
        if (!e && full_index) e = g6_kvc_create(idx_out, (size_t)pages * IDX_PAGE, dev);
        if (e) {
            g6_kvc_free(*mla_out);
            *mla_out = nullptr;
            g6_kvc_free(*idx_out);
            *idx_out = nullptr;
            return fail(e, "g6_kvc_create");
        }
        regions.push_back(Region{*mla_out, *idx_out, pages});
        return 0;
    }

    // Back/unmap granules across every binding's MLA and index regions to match `online[0..chunks)` below `place`.
    // Pages >= `place` stay backed (g6_kvc_sync's guarantee).
    int sync(uint32_t chunk_pages, uint32_t place, const std::vector<uint8_t>& online, void* stream,
             size_t* mapped = nullptr, size_t* released = nullptr) {
        size_t mp = 0, rl = 0;
        int e = 0;
        const uint32_t chunks = (uint32_t)online.size();
        for (const Region& r : regions) {
            size_t a = 0, b = 0;
            e = g6_kvc_sync(r.mla, MLA_PAGE, chunk_pages, place, online.data(), chunks, stream, &a, &b);
            mp += a;
            rl += b;
            if (e) break;
            if (r.idx) {
                a = b = 0;
                e = g6_kvc_sync(r.idx, IDX_PAGE, chunk_pages, place, online.data(), chunks, stream, &a, &b);
                mp += a;
                rl += b;
                if (e) break;
            }
        }
        if (mapped) *mapped = mp;
        if (released) *released = rl;
        return e ? fail(e, "g6_kvc_sync") : 0;
    }

    size_t backed() const {
        size_t total = 0;
        for (const Region& r : regions) {
            total += g6_kvc_backed(r.mla);
            if (r.idx) total += g6_kvc_backed(r.idx);
        }
        return total;
    }

    size_t reserved() const {
        size_t total = 0;
        for (const Region& r : regions) {
            total += g6_kvc_bytes(r.mla);
            if (r.idx) total += g6_kvc_bytes(r.idx);
        }
        return total;
    }

    void release() {
        for (const Region& r : regions) {
            g6_kvc_free(r.mla);
            if (r.idx) g6_kvc_free(r.idx);
        }
        regions.clear();
    }

private:
    int fail(int e, const std::string& what) {
        err = what + " (" + std::to_string(e) + ")";
        return e;
    }
};

}   // namespace glmkv
