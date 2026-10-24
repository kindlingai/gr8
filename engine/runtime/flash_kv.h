// G6_FLASH_KV_CHUNKS=1 (D-331 M7 part B; 0, the default, keeps one cudaMalloc per binding as before): a Flash rank's
// MLA and index caches, every DSA binding's, in kvchunks regions (attn/kvchunks.h) under one chunk map. Chunk k is
// pages [k * chunk_pages, (k + 1) * chunk_pages) of every binding (the last may be shorter), online in all of them or
// in none. The Flash session's block table is the identity (flash_bind.cu), so its context fills pages from 0 up and
// the online chunks are a prefix: the map is one count. chunk_pages is the fewest pages whose MLA and index bytes are
// both whole granules (g6_flash_kv_chunk_pages: 256 at 2 MiB), so an offline chunk gives back all of its bytes.
// resize(n) backs or releases granules to match n chunks online. No step may reach a token past tokens(): the caller
// admits nothing longer. With a ledger (the box's host ledger, src/ledger.rs), the bytes move as one transfer:
//   shrink: begin gpu -> node, sync (unmapped), released, commit: the node may lease them from then on
//   grow:   begin node -> gpu, wait until the node released them (its leases fit the rest of its credit), sync, commit
// A crash at any point leaves a book that recovers (kvman-ledger recover) and never counts a byte twice: bytes
// released and not committed go to free.
// Host only. Not thread safe. G6_MODEL_FLASH builds.
#pragma once
#include "../attn/bind.h"
#include "../attn/kvchunks.h"
#include "../attn/kvfmt.h"
#include "../kvman-rs/g6kvman.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace flashkv {

// G6_FLASH_KV_CHUNKS: 0 (default) or 1; anything else is an error (-1)
inline int env_mode()
{
    const char* e = getenv("G6_FLASH_KV_CHUNKS");
    if (!e || !strcmp(e, "0")) return 0;
    return strcmp(e, "1") ? -1 : 1;
}

// tests only: G6_FLASH_KV_CRASH_AT=begun|synced|released ends the process there (the ledger's crash points)
inline void crash_at(const char* point)
{
    const char* e = getenv("G6_FLASH_KV_CRASH_AT");
    if (e && !strcmp(e, point)) std::_Exit(3);
}

constexpr size_t MLA_PAGE = (size_t)g6kv::PAGE * g6kv::ROW, IDX_PAGE = g6kv::IDX_PAGE;

struct Map {
    uint32_t chunk_pages = 0, pages = 0, chunks = 0, online = 0;
    size_t gran = 0;
    int device = 0;
    std::vector<G6KvChunks*> mla, idx;   // one each per binding
    std::string ledger;                  // the host ledger's path, or empty: no ledger
    int wait_ms = 10000;                 // grow: how long the node may take to release
    std::string err;                     // the last failure

    // pages: the bindings' capacity in pages; online: chunks online at boot (clamped to all)
    int init(uint32_t pages_, uint32_t online_, int device_)
    {
        device = device_;
        gran = g6_kvc_device_granule(device);
        chunk_pages = g6_flash_kv_chunk_pages(gran);
        if (!pages_ || !chunk_pages) return fail(1, "no granule or chunk size on this device");
        pages = pages_;
        chunks = (pages + chunk_pages - 1) / chunk_pages;
        online = std::min(online_, chunks);
        return 0;
    }
    std::vector<uint8_t> mask(uint32_t n) const
    {
        std::vector<uint8_t> m(chunks, 0);
        std::fill(m.begin(), m.begin() + std::min(n, chunks), 1);
        return m;
    }
    // the two regions of one binding, backed for the chunks online now
    int add(G6KvChunks** m, G6KvChunks** i)
    {
        const std::vector<uint8_t> on = mask(online);
        *m = *i = nullptr;
        int e = g6_kvc_create_masked(m, pages * MLA_PAGE, device, MLA_PAGE, chunk_pages, pages, on.data(), chunks);
        if (!e) e = g6_kvc_create_masked(i, pages * IDX_PAGE, device, IDX_PAGE, chunk_pages, pages, on.data(), chunks);
        if (e) { g6_kvc_free(*m); *m = nullptr; return fail(e, "creating a binding's regions"); }
        mla.push_back(*m);
        idx.push_back(*i);
        return 0;
    }
    uint64_t tokens() const { return (uint64_t)std::min(online * chunk_pages, pages) * g6kv::PAGE; }
    // bytes every region backs with n chunks online
    uint64_t bytes_at(uint32_t n) const
    {
        const uint64_t p = std::min(std::min(n, chunks) * chunk_pages, pages);
        auto up = [&](uint64_t b) { return (b + gran - 1) / gran * gran; };
        return (uint64_t)mla.size() * (up(p * MLA_PAGE) + up(p * IDX_PAGE));
    }
    uint64_t backed() const
    {
        uint64_t b = 0;
        for (size_t k = 0; k < mla.size(); ++k) b += g6_kvc_backed(mla[k]) + g6_kvc_backed(idx[k]);
        return b;
    }
    // every region to n chunks online, no ledger. *mapped / *released: granule bytes changed, all regions
    int sync(uint32_t n, void* stream, uint64_t* mapped = nullptr, uint64_t* released = nullptr)
    {
        const std::vector<uint8_t> on = mask(n);
        uint64_t mp = 0, rl = 0;
        int e = 0;
        for (size_t k = 0; !e && k < mla.size(); ++k)
            for (int w = 0; !e && w < 2; ++w)
            {
                size_t a = 0, b = 0;
                e = g6_kvc_sync(w ? idx[k] : mla[k], w ? IDX_PAGE : MLA_PAGE, chunk_pages, pages, on.data(), chunks,
                                stream, &a, &b);
                mp += a;
                rl += b;
            }
        if (mapped) *mapped = mp;
        if (released) *released = rl;
        return e ? fail(e, "g6_kvc_sync") : 0;
    }
    // n chunks online; with a ledger, the bytes move GPU <-> node as one transfer (the header). 0, or an error with err
    int resize(uint32_t n, void* stream)
    {
        n = std::min(n, chunks);
        if (n == online) return 0;
        if (ledger.empty())
        {
            const int e = sync(n, stream);
            // a failed shrink may leave some regions backed past n: none of those pages is handed out again
            if (!e || n < online) online = n;
            return e;
        }
        return n < online ? shrink(n, stream) : grow(n, stream);
    }
    void release()
    {
        for (auto* c : mla) g6_kvc_free(c);
        for (auto* c : idx) g6_kvc_free(c);
        mla.clear();
        idx.clear();
    }

private:
    int fail(int e, const std::string& what)
    {
        err = what + " (" + std::to_string(e) + ")";
        return e;
    }
    int ledger_fail(const char* what)
    {
        err = std::string("the host ledger: ") + what + ": " + g6kv_tier_error();
        return -1;
    }
    int shrink(uint32_t n, void* stream)
    {
        const uint64_t want = bytes_at(online) - bytes_at(n);
        uint64_t gen = 0, released = 0;
        if (g6kv_ledger_begin(ledger.c_str(), 0, 1, want, &gen)) return ledger_fail("begin gpu -> node");
        crash_at("begun");
        const int e = sync(n, stream, nullptr, &released);
        online = n;   // as resize: even a failed shrink hands out no page past n
        crash_at("synced");
        if (g6kv_ledger_released(ledger.c_str(), gen, released)) return ledger_fail("released");
        crash_at("released");
        if (g6kv_ledger_commit(ledger.c_str(), gen, released)) return ledger_fail("commit");
        return e;
    }
    int grow(uint32_t n, void* stream)
    {
        const uint64_t want = bytes_at(n) - bytes_at(online), before = backed();
        uint64_t gen = 0, book[9] = {};
        if (g6kv_ledger_begin(ledger.c_str(), 1, 0, want, &gen)) return ledger_fail("begin node -> gpu");
        crash_at("begun");
        // the node releases once its leases fit the rest of its credit (kvman-remote --ledger)
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        for (;;)
        {
            if (g6kv_ledger_show(ledger.c_str(), book)) return ledger_fail("show");
            if (book[6] != gen) { err = "the host ledger: transfer " + std::to_string(gen) + " ended elsewhere"; return -1; }
            if (book[7] == 2) break;
            if (std::chrono::steady_clock::now() >= until)
            {
                g6kv_ledger_abort(ledger.c_str(), gen);
                err = "the KV node did not release " + std::to_string(want) + " bytes within " + std::to_string(wait_ms) +
                      " ms (its leases hold them)";
                return -1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        uint64_t mapped = 0;
        const int e = sync(n, stream, &mapped, nullptr);
        crash_at("synced");
        if (e)
        {
            // back to what was online; what the GPU still holds of the transfer is committed, the rest goes to free
            sync(online, stream);
            const uint64_t now = backed(), kept = now > before ? now - before : 0;
            if (g6kv_ledger_commit(ledger.c_str(), gen, std::min(kept, book[8]))) return ledger_fail("commit");
            return e;
        }
        online = n;
        if (g6kv_ledger_commit(ledger.c_str(), gen, std::min(mapped, book[8]))) return ledger_fail("commit");
        return 0;
    }
};

}   // namespace flashkv
