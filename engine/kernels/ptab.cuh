// EL3 (#151) stage 1: paged routed-expert weights. A piece is one 128-column block of one tr3.table row (its gate and up
// units of tr3.fc1, and its down block of tr3.fc2). Pieces live in slabs from a pool, and a per-layer device table at a
// fixed address maps piece (row e, block b) to its two slabs: ptab[(e * EB + b) * 2] = fc1 bytes (gate then up, each at
// its unit's width, the order tr3_unit gives them), ptab[... + 1] = fc2 bytes. The decode (k1.cuh, tr3.cuh) and prefill
// (bind_prefill.cuh) expert kernels read a piece through the table when K1Args / bindp::Expert carry one (t_ptab), one
// table read per CTA before the tile loops; with t_ptab == nullptr they read the pack image as before (tr3_unit).
//
//   ptab::Pool pool;                                   one per process (or per test), device memory in chunks
//   ptab::Table t;  err = ptab::build(t, T, mode, pool, st);   after bind::scan_widths and bind::widen_fc1 / widen_fc2
//   T.ptab = t.dev;                                     bind::make_k1 / bindp::make_layer hand it to the kernels
//   ptab::set(t, e, b, fc1, fc2, st);                   change two entries (the fixed address keeps captured graphs valid)
//
// Modes (G6_PTAB, docs/SETTINGS.md): IMAGE = every entry points into the rank's tr3.fc1 / tr3.fc2 (the same bytes at
// the same addresses as the pack layout, only read through the table); SCATTER = every piece copied into its own
// slab from the pool, in a scrambled order, so no two adjacent pieces of the image are adjacent in memory (the proof
// that the kernels need no contiguity; the image's expert bytes stay allocated in this stage). Pieces of one row in
// SCATTER mode are copied after the wide reorder, so a slab holds the order the kernels read (t_wide / t_wide2).
//
// Left for later stages: attention, dense and vocab pieces (they stay contiguous); the loader filling slabs straight
// from the files without the image; returning freed slabs to the KV pool (#153); the switch (#156).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "tr3.cuh"
#include "../vmm.h"

namespace ptab
{

constexpr size_t SLAB_ALIGN = 256;         // every slab starts on a 256 B boundary (the widest load is 16 B cp.async)

// Device memory in chunks (cudaMalloc, or cuMemCreate ranges under G6_IMG_VMM, engine/vmm.h); slabs are carved from the
// current chunk, and a freed slab of a size goes to that size's free list, where the next take of that size finds it.
// A slab never spans two chunks. Stage 1 keeps whole chunks until release(); a later stage hands empty chunks to the
// KV pool (#153).
struct Pool
{
    size_t chunk_bytes = size_t(256) << 20;   // default chunk; a slab larger than it gets a chunk of its own size
    bool vmm = false;                         // chunks from g6vmm::alloc (G6_IMG_VMM) instead of cudaMalloc
    struct Chunk { uint8_t* p; size_t bytes, used; bool vmm; };
    std::vector<Chunk> chunks;
    std::map<size_t, std::vector<uint8_t*>> free_by_size;
    size_t held = 0, used = 0, freed = 0;     // bytes: of chunks, in live slabs, in the free lists
    int cur = -1;

    Pool() = default;
    Pool(const Pool&) = delete;              // owns device memory: never copied
    Pool& operator=(const Pool&) = delete;
    ~Pool() { release(); }
    static size_t round(size_t n) { return (n + SLAB_ALIGN - 1) / SLAB_ALIGN * SLAB_ALIGN; }

    const char* grow(size_t at_least)
    {
        Chunk c{nullptr, at_least > chunk_bytes ? at_least : chunk_bytes, 0, false};
        if (vmm)
        {
            std::string why;
            c.p = static_cast<uint8_t*>(g6vmm::alloc(c.bytes, 0, why));
            c.vmm = c.p != nullptr;
        }
        if (!c.p && cudaMalloc(&c.p, c.bytes) != cudaSuccess) return "ptab::Pool: cudaMalloc of a chunk failed";
        chunks.push_back(c);
        held += c.bytes;
        cur = static_cast<int>(chunks.size()) - 1;
        return nullptr;
    }
    // a slab of n bytes (rounded up to SLAB_ALIGN); nullptr + err on failure
    uint8_t* take(size_t n, const char*& err)
    {
        err = nullptr;
        n = round(n);
        auto it = free_by_size.find(n);
        if (it != free_by_size.end() && !it->second.empty())
        {
            uint8_t* p = it->second.back();
            it->second.pop_back();
            freed -= n; used += n;
            return p;
        }
        if (cur < 0 || chunks[cur].used + n > chunks[cur].bytes)
            if ((err = grow(n))) return nullptr;
        Chunk& c = chunks[cur];
        uint8_t* p = c.p + c.used;
        c.used += n; used += n;
        return p;
    }
    void give(uint8_t* p, size_t n)
    {
        n = round(n);
        free_by_size[n].push_back(p);
        used -= n; freed += n;
    }
    void release()
    {
        for (Chunk& c : chunks)
        {
            if (c.vmm) g6vmm::release(c.p);
            else cudaFree(c.p);
        }
        chunks.clear();
        free_by_size.clear();
        held = used = freed = 0;
        cur = -1;
    }
};

enum Mode { OFF = 0, IMAGE = 1, SCATTER = 2 };

// One layer's piece table: the device array at a fixed address (dev), and its host twin with each entry's bytes.
struct Table
{
    const uint8_t** dev = nullptr;        // device [rows][EB][2]; nullptr until build
    std::vector<const uint8_t*> host;     // the same entries
    std::vector<size_t> bytes;            // per entry: the slab's bytes (0 = no piece)
    int rows = 0, eb = 0;
    Mode mode = OFF;
    size_t slab_bytes = 0;                // SCATTER: bytes taken from the pool

    Table() = default;
    Table(const Table&) = delete;         // owns the device array: never copied
    Table& operator=(const Table&) = delete;
    ~Table() { if (dev) cudaFree(dev); }
    int slot(int e, int b) const { return (e * eb + b) * 2; }
    size_t n() const { return static_cast<size_t>(rows) * eb * 2; }
};

// Host side of tr3_unit: byte offset of fc1 unit u of table row tr (kmix entry km) in tr3.fc1
inline size_t unit_off(long long km, const long long* tr, int u)
{
    constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;
    if (!km) return static_cast<size_t>(tr[0]) + static_cast<size_t>(u) * tr[4] * MB;
    size_t off = static_cast<size_t>(tr[0]);
    for (int i = 0; i < u; ++i) off += static_cast<size_t>((km >> (4 * i)) & 15) * MB;
    return off;
}

// Reads tr3.table (and tr3.kmix) back: rows x 6 and rows entries, or an error
inline const char* read_table(const void* tab, const void* kmix, int rows, std::vector<long long>& h, std::vector<long long>& k)
{
    h.assign(static_cast<size_t>(rows) * 6, 0);
    k.assign(static_cast<size_t>(rows), 0);
    if (cudaMemcpy(h.data(), tab, h.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return "ptab: cannot read tr3.table";
    if (kmix && cudaMemcpy(k.data(), kmix, k.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return "ptab: cannot read tr3.kmix";
    return nullptr;
}

// The entries of the pack layout: piece (e, b) at its tr3.table offsets inside fc1 / fc2
inline const char* image_entries(Table& t, const uint8_t* fc1, const uint8_t* fc2, const std::vector<long long>& h, const std::vector<long long>& k)
{
    t.host.assign(t.n(), nullptr);
    t.bytes.assign(t.n(), 0);
    for (int e = 0; e < t.rows; ++e)
    {
        const long long* tr = &h[static_cast<size_t>(e) * 6];
        if (tr[3] > t.eb) return "ptab: a row has more blocks than EB";
        for (int b = 0; b < tr[3]; ++b)
        {
            const size_t f1 = tr3_piece_fc1_bytes(k[e], tr, b), f2 = tr3_piece_fc2_bytes(tr);
            t.host[t.slot(e, b)] = fc1 + unit_off(k[e], tr, 2 * b);
            t.host[t.slot(e, b) + 1] = fc2 + static_cast<size_t>(tr[1]) + static_cast<size_t>(b) * f2;
            t.bytes[t.slot(e, b)] = f1;
            t.bytes[t.slot(e, b) + 1] = f2;
        }
    }
    return nullptr;
}

inline const char* upload(Table& t, cudaStream_t st)
{
    if (cudaMemcpyAsync(t.dev, t.host.data(), t.n() * sizeof(void*), cudaMemcpyHostToDevice, st) != cudaSuccess) return "ptab: table upload failed";
    return cudaStreamSynchronize(st) == cudaSuccess ? nullptr : "ptab: table upload failed";
}

// Builds the table of one layer. tab / kmix / fc1 / fc2: the layer's tr3.table, tr3.kmix (or nullptr), tr3.fc1, tr3.fc2
// device tensors, rows = tr3.table's rows (NEXP, or NEXP + 1 with the trellis shared expert). Call after
// bind::widen_fc1 / widen_fc2 (SCATTER copies the bytes as they are). Synchronous.
inline const char* build(Table& t, const void* tab, const void* kmix, const uint8_t* fc1, const uint8_t* fc2, int rows, int eb,
                         Mode mode, Pool& pool, cudaStream_t st)
{
    if (mode == OFF) return nullptr;
    if (!tab || !fc1 || !fc2 || rows <= 0 || eb <= 0) return "ptab: a MoE layer needs tr3.table, tr3.fc1 and tr3.fc2";
    t.rows = rows; t.eb = eb; t.mode = mode;
    std::vector<long long> h, k;
    if (const char* e = read_table(tab, kmix, rows, h, k)) return e;
    if (const char* e = image_entries(t, fc1, fc2, h, k)) return e;
    if (!t.dev && cudaMalloc(&t.dev, t.n() * sizeof(void*)) != cudaSuccess) return "ptab: cudaMalloc of the table failed";
    if (mode == SCATTER)
    {
        // every piece into a slab of its own, in a scrambled order (a fixed permutation of the entries), so a piece's
        // neighbours in the image are not its neighbours in memory
        std::vector<size_t> order;
        for (size_t i = 0; i < t.n(); ++i) if (t.bytes[i]) order.push_back(i);
        uint64_t s = 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(rows) << 32) ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fc1));
        for (size_t i = order.size(); i > 1; --i)
        {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const size_t j = static_cast<size_t>((s >> 33) % i);
            std::swap(order[i - 1], order[j]);
        }
        std::vector<const uint8_t*> src = t.host;
        for (size_t i : order)
        {
            const char* err;
            uint8_t* p = pool.take(t.bytes[i], err);
            if (!p) return err;
            if (cudaMemcpyAsync(p, src[i], t.bytes[i], cudaMemcpyDeviceToDevice, st) != cudaSuccess) return "ptab: piece copy failed";
            t.host[i] = p;
            t.slab_bytes += Pool::round(t.bytes[i]);
        }
    }
    return upload(t, st);
}

// Changes the two entries of piece (e, b) in place (host twin and device table): the fixed address keeps captured graphs
// valid; the next launch reads the new slabs
inline const char* set(Table& t, int e, int b, const uint8_t* fc1, const uint8_t* fc2, cudaStream_t st)
{
    if (!t.dev || e < 0 || e >= t.rows || b < 0 || b >= t.eb) return "ptab::set: no such piece";
    const int s = t.slot(e, b);
    t.host[s] = fc1; t.host[s + 1] = fc2;
    if (cudaMemcpyAsync(t.dev + s, t.host.data() + s, 2 * sizeof(void*), cudaMemcpyHostToDevice, st) != cudaSuccess) return "ptab::set: upload failed";
    return nullptr;
}

// G6_PTAB: 0 / unset = off (the pack layout), 1 = image, 2 = scatter; anything else refuses
inline const char* mode_env(Mode& m)
{
    m = OFF;
    const char* v = getenv("G6_PTAB");
    if (!v || !*v) return nullptr;
    if (!strcmp(v, "0")) { m = OFF; return nullptr; }
    if (!strcmp(v, "1")) { m = IMAGE; return nullptr; }
    if (!strcmp(v, "2")) { m = SCATTER; return nullptr; }
    return "G6_PTAB must be 0, 1 (table into the image) or 2 (pieces scattered into slabs)";
}

}   // namespace ptab
