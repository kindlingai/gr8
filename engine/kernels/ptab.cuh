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
//   old = ptab::set(t, e, b, fc1, f1b, fc2, f2b, st);   change two entries (the fixed address keeps captured graphs valid)
//
// Modes (GR8_PTAB, docs/SETTINGS.md): IMAGE = every entry points into the rank's tr3.fc1 / tr3.fc2 (the same bytes at
// the same addresses as the pack layout, only read through the table); SCATTER = every piece copied into its own
// slab from the pool, in a scrambled order (a Fisher-Yates shuffle of the pieces, reshuffled until no two pieces that
// are adjacent in the image are adjacent in memory: the proof that the kernels need no contiguity; the image's expert
// bytes stay allocated in this stage). Pieces of one row in SCATTER mode are copied after the wide reorder, so a slab
// holds the order the kernels read (t_wide / t_wide2).
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
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "ptab_pool.h"
#include "tr3.cuh"
#include "../vmm.h"
#include "../knobs-rs/gr8_knobs.h"

namespace ptab
{

// CUDA chunk backend for SlabPool (ptab_pool.h): allocates chunks via g6vmm::alloc (when vmm=true) or cudaMalloc,
// and releases empty chunks back to unified RAM / the EL5 KV pool via g6vmm::release or cudaFree.
struct CudaBackend
{
    static bool alloc_chunk(size_t want_bytes, bool want_vmm, uint8_t*& out_p, size_t& out_bytes, bool& out_vmm,
                            int& vmm_fallbacks, const char*& err)
    {
        out_p = nullptr;
        out_bytes = want_bytes;
        out_vmm = false;
        if (want_vmm)
        {
            std::string why;
            size_t gmin = 0, grec = 0;
            int dev = 0;
            cudaGetDevice(&dev);
            g6vmm::granularity(dev, gmin, grec);
            out_p = static_cast<uint8_t*>(g6vmm::alloc(out_bytes, 0, why));
            out_vmm = out_p != nullptr;
            if (out_vmm && grec) out_bytes = (out_bytes + grec - 1) / grec * grec;
            else if (!out_vmm)
            {
                fprintf(stderr, "# ptab::Pool: VMM chunk of %zu MB failed (%s); cudaMalloc instead\n", out_bytes >> 20, why.c_str());
                ++vmm_fallbacks;
            }
        }
        if (!out_p && cudaMalloc(&out_p, out_bytes) != cudaSuccess)
        {
            err = "ptab::Pool: cudaMalloc of a chunk failed";
            return false;
        }
        return true;
    }
    static void free_chunk(uint8_t* p, bool is_vmm)
    {
        if (!p) return;
        if (is_vmm) g6vmm::release(p);
        else cudaFree(p);
    }
};

using Pool = SlabPool<CudaBackend>;

enum Mode { OFF = 0, IMAGE = 1, SCATTER = 2, CONTIG = 3 };

// One layer's piece table: the device array at a fixed address (dev), and its host twin with each entry's bytes.
// Movable (a runtime Layer that holds one can live in a std::vector), never copied: the device array has one owner.
struct Table
{
    const uint8_t** dev = nullptr;        // device [rows][EB][2]; nullptr until build
    std::vector<const uint8_t*> host;     // the same entries
    std::vector<size_t> bytes;            // per entry: the slab's bytes (0 = no piece)
    int rows = 0, eb = 0;
    Mode mode = OFF;
    size_t slab_bytes = 0;                // SCATTER: bytes taken from the pool
    int reshuffles = 0;                   // SCATTER: shuffles redrawn because an image-adjacent pair would have stayed adjacent
    int relocated = 0;                    // SCATTER: pieces moved after the copy because a pair still was (chunk boundaries)

    Table() = default;
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    Table(Table&& o) noexcept { *this = std::move(o); }
    Table& operator=(Table&& o) noexcept
    {
        if (this == &o) return *this;
        if (dev) cudaFree(dev);
        dev = o.dev; o.dev = nullptr;
        host = std::move(o.host); bytes = std::move(o.bytes);
        rows = o.rows; eb = o.eb; mode = o.mode; slab_bytes = o.slab_bytes; reshuffles = o.reshuffles; relocated = o.relocated;
        o.rows = o.eb = 0; o.mode = OFF; o.slab_bytes = 0; o.reshuffles = o.relocated = 0;
        return *this;
    }
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
        if (tr[3] > 8) return "ptab: a row has more than 8 blocks (the tr3.kmix format's ceiling)";
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

// Uploads the whole host twin; synchronous
inline const char* upload(Table& t, cudaStream_t st)
{
    if (cudaMemcpyAsync(t.dev, t.host.data(), t.n() * sizeof(void*), cudaMemcpyHostToDevice, st) != cudaSuccess) return "ptab: table upload failed";
    return cudaStreamSynchronize(st) == cudaSuccess ? nullptr : "ptab: table upload failed";
}

// Pairs of entries that are adjacent in the image (entry i follows entry i - 2 of the same matrix) and whose slabs are
// adjacent in memory, under the slab order `at` (at[i] = the slab entry i would get). The test prints this too.
inline size_t adjacent_pairs(const Table& t, const std::vector<const uint8_t*>& at)
{
    size_t adj = 0;
    for (size_t i = 2; i < t.n(); ++i)
        if (t.bytes[i] && t.bytes[i - 2] && at[i] == at[i - 2] + Pool::round(t.bytes[i - 2])) ++adj;
    return adj;
}

// Builds the table of one layer. tab / kmix / fc1 / fc2: the layer's tr3.table, tr3.kmix (or nullptr), tr3.fc1, tr3.fc2
// device tensors, rows = tr3.table's rows (NEXP, or NEXP + 1 with the trellis shared expert). Call after
// bind::widen_fc1 / widen_fc2 (SCATTER copies the bytes as they are). Synchronous. SCATTER: the pieces are shuffled
// (Fisher-Yates from a seed of the layer's rows and fc1 address) and bump-allocated from the pool in that order; a
// shuffle that would leave an image-adjacent pair adjacent is redrawn before any slab is taken, and a pair that still
// is afterwards (a chunk boundary the dry run could not place) has its second piece moved to a fresh slab, so the built
// table has none, or build fails (Table::reshuffles / relocated count both).
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
        // every piece into a slab of its own, in a scrambled order (a permutation of the entries), so a piece's
        // neighbours in the image are not its neighbours in memory. The pool is a bump allocator (fresh slabs, no
        // free list in this stage), so the slab order is the shuffle order: a shuffle is checked for image-adjacent
        // pairs that stay adjacent, from the pool's next free address, before any slab is taken, and redrawn if one is.
        std::vector<size_t> order;
        for (size_t i = 0; i < t.n(); ++i) if (t.bytes[i]) order.push_back(i);
        uint64_t s = 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(rows) << 32) ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fc1));
        auto shuffle = [&]() {
            for (size_t i = order.size(); i > 1; --i)
            {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                const size_t j = static_cast<size_t>((s >> 33) % i);
                std::swap(order[i - 1], order[j]);
            }
        };
        // where the pool's bump allocation would put each entry (a slab never spans two chunks; the dry run follows take)
        std::vector<const uint8_t*> at(t.n(), nullptr);
        auto dry_run = [&]() {
            const int c = pool.cur;
            size_t used = c >= 0 ? pool.chunks[c].used : 0, cap = c >= 0 ? pool.chunks[c].bytes : 0;
            const uint8_t* base = c >= 0 ? pool.chunks[c].p : nullptr;
            uintptr_t fresh = 0;       // a chunk not yet allocated has no address yet: modelled as a range apart from everything
            for (size_t i : order)
            {
                const size_t n = Pool::round(t.bytes[i]);
                auto it = pool.free_by_size.find(n);
                if (it != pool.free_by_size.end() && !it->second.empty()) { at[i] = nullptr; continue; }   // a free-list slab: checked after the real take
                fresh += size_t(1) << 40;
                if (n > pool.chunk_bytes) { at[i] = reinterpret_cast<const uint8_t*>(fresh); continue; }   // a chunk of its own; cur stays
                if (!base || used + n > cap) { base = reinterpret_cast<const uint8_t*>(fresh); used = 0; cap = pool.chunk_bytes; }
                at[i] = base + used;
                used += n;
            }
        };
        size_t adj = 0;
        for (int tries = 0; tries < 256; ++tries)
        {
            shuffle();
            dry_run();
            adj = adjacent_pairs(t, at);
            if (!adj) break;
            ++t.reshuffles;
        }
        std::vector<const uint8_t*> src = t.host;
        auto place = [&](size_t i, bool fresh) -> const char* {
            const char* err;
            uint8_t* p = pool.take(t.bytes[i], err, fresh);
            if (!p) return err;
            if (cudaMemcpyAsync(p, src[i], t.bytes[i], cudaMemcpyDeviceToDevice, st) != cudaSuccess) return "ptab: piece copy failed";
            t.host[i] = p;
            t.slab_bytes += Pool::round(t.bytes[i]);
            return nullptr;
        };
        for (size_t i : order)
            if (const char* e = place(i, false)) return e;
        // The dry run cannot see where a chunk the pool had not allocated yet lands (or which free-list slab a take
        // returns), so a pair can still come out adjacent across a chunk boundary. Each such pair's second piece moves
        // to a fresh slab at the pool's bump pointer, which follows every slab taken so far and so is adjacent to none
        // of their image neighbours; the old slab goes to the free list (a later layer's build may take it).
        for (int pass = 0; pass < 1024; ++pass)
        {
            size_t bad = t.n();
            for (size_t i = 2; i < t.n(); ++i)
                if (t.bytes[i] && t.bytes[i - 2] && t.host[i] == t.host[i - 2] + Pool::round(t.bytes[i - 2])) { bad = i; break; }
            if (bad == t.n()) break;
            uint8_t* old = const_cast<uint8_t*>(t.host[bad]);
            t.slab_bytes -= Pool::round(t.bytes[bad]);
            if (const char* e = place(bad, true)) return e;
            pool.give(old, t.bytes[bad]);
            ++t.relocated;
        }
        if (adjacent_pairs(t, t.host)) return "ptab: an image-adjacent pair of pieces is adjacent in memory after the relocations";
    }
    else if (mode == CONTIG)
    {
        // EL3 (#151) serving layout: all fc1 blocks of each expert contiguous in one take_run, and all fc2 blocks
        // contiguous in one take_run (matching mode-1 DRAM/L2 locality while keeping every block individually freeable).
        std::vector<const uint8_t*> src = t.host;
        std::vector<size_t> f1_sz(eb), f2_sz(eb);
        std::vector<uint8_t*> f1_p(eb), f2_p(eb);
        for (int e = 0; e < t.rows; ++e)
        {
            const int nb = static_cast<int>(h[static_cast<size_t>(e) * 6 + 3]);
            if (nb <= 0) continue;
            for (int b = 0; b < nb; ++b)
            {
                const int s = t.slot(e, b);
                f1_sz[b] = t.bytes[s];
                f2_sz[b] = t.bytes[s + 1];
            }
            const char* err = nullptr;
            if (!pool.take_run(f1_sz.data(), nb, f1_p.data(), err)) return err;
            if (!pool.take_run(f2_sz.data(), nb, f2_p.data(), err)) return err;
            for (int b = 0; b < nb; ++b)
            {
                const int s = t.slot(e, b);
                if (cudaMemcpyAsync(f1_p[b], src[s], t.bytes[s], cudaMemcpyDeviceToDevice, st) != cudaSuccess)
                    return "ptab: contig fc1 piece copy failed";
                if (cudaMemcpyAsync(f2_p[b], src[s + 1], t.bytes[s + 1], cudaMemcpyDeviceToDevice, st) != cudaSuccess)
                    return "ptab: contig fc2 piece copy failed";
                t.host[s] = f1_p[b];
                t.host[s + 1] = f2_p[b];
                t.slab_bytes += Pool::round(t.bytes[s]) + Pool::round(t.bytes[s + 1]);
            }
        }
    }
    return upload(t, st);
}

// EL3 (#151) direct slab allocation: given host copies of tr3.table (`h`) and tr3.kmix (`k`), allocates every
// routed-expert piece contiguously per expert in `pool` (Mode::CONTIG), populates `t.dev`, and records every
// piece's logical pack byte range [pack_lo, pack_hi) -> slab pointer into `route` so the Rust loader can stream
// directly into slabs without allocating tr3.fc1 / tr3.fc2 in DevImage.
inline const char* alloc_direct(Table& t, const std::vector<long long>& h, const std::vector<long long>& k, int rows,
                                int eb, Pool& pool, uint64_t fc1_pack_off, uint64_t fc2_pack_off, RouteMap& route,
                                cudaStream_t st)
{
    if (rows <= 0 || eb <= 0 || h.size() < static_cast<size_t>(rows) * 6 || (!k.empty() && k.size() < static_cast<size_t>(rows)))
        return "ptab::alloc_direct: invalid tr3.table / tr3.kmix";
    t.rows = rows; t.eb = eb; t.mode = CONTIG;
    t.host.assign(t.n(), nullptr);
    t.bytes.assign(t.n(), 0);
    t.slab_bytes = 0;
    if (!t.dev && cudaMalloc(&t.dev, t.n() * sizeof(void*)) != cudaSuccess) return "ptab::alloc_direct: cudaMalloc of table failed";
    std::vector<size_t> f1_sz(eb), f2_sz(eb);
    std::vector<uint8_t*> f1_p(eb), f2_p(eb);
    for (int e = 0; e < rows; ++e)
    {
        const long long* tr = &h[static_cast<size_t>(e) * 6];
        const long long km = k.empty() ? 0 : k[e];
        const int nb = static_cast<int>(tr[3]);
        if (nb > eb) return "ptab::alloc_direct: a row has more blocks than EB";
        if (nb > 8) return "ptab::alloc_direct: a row has more than 8 blocks (tr3.kmix ceiling)";
        if (nb <= 0) continue;
        for (int b = 0; b < nb; ++b)
        {
            f1_sz[b] = tr3_piece_fc1_bytes(km, tr, b);
            f2_sz[b] = tr3_piece_fc2_bytes(tr);
        }
        const char* err = nullptr;
        if (!pool.take_run(f1_sz.data(), nb, f1_p.data(), err)) return err;
        if (!pool.take_run(f2_sz.data(), nb, f2_p.data(), err)) return err;
        for (int b = 0; b < nb; ++b)
        {
            const int s = t.slot(e, b);
            t.host[s] = f1_p[b];
            t.host[s + 1] = f2_p[b];
            t.bytes[s] = f1_sz[b];
            t.bytes[s + 1] = f2_sz[b];
            t.slab_bytes += Pool::round(f1_sz[b]) + Pool::round(f2_sz[b]);
            route.add(fc1_pack_off + unit_off(km, tr, 2 * b), f1_sz[b], f1_p[b]);
            route.add(fc2_pack_off + static_cast<uint64_t>(tr[1]) + static_cast<uint64_t>(b) * f2_sz[b], f2_sz[b], f2_p[b]);
        }
    }
    return upload(t, st);
}

// The slabs that held a piece before a set(): hand them to Pool::give (SCATTER slabs) once nothing reads them.
struct Old { const uint8_t* fc1 = nullptr; size_t fc1_bytes = 0; const uint8_t* fc2 = nullptr; size_t fc2_bytes = 0; };

// Changes the two entries of piece (e, b) in place: the host twin, its byte counts, and the device table, with one
// 16 B H2D copy on `st`. The table's fixed address keeps captured graphs valid; which launch first reads the new slabs
// is a stream question, and the contract is:
//   - call it on the stream the kernels that read the table run on (the serving stream), or on a stream that an
//     event orders before them: a launch ordered after the copy on that stream reads both new entries, one ordered
//     before it reads both old ones. A kernel in flight on another stream, or a graph replayed from one, is not
//     ordered against the copy and may see one new and one old slab;
//   - the old slabs are returned so the caller can Pool::give them with their sizes, after the last launch that read
//     them (an event recorded on `st` after the copy, waited on before give);
//   - the host copies some bindings took of row NEXP's slabs (bind::Layer::shw_piece for warm_mlp and l2pf) are
//     not refreshed here: rebind them (bind::refresh_shared_pieces) after a set() on that row.
// Returns nullptr and fills `old`, or an error (the table unchanged).
inline const char* set(Table& t, int e, int b, const uint8_t* fc1, size_t fc1_bytes, const uint8_t* fc2, size_t fc2_bytes, cudaStream_t st, Old* old = nullptr)
{
    if (!t.dev || e < 0 || e >= t.rows || b < 0 || b >= t.eb) return "ptab::set: no such piece";
    const int s = t.slot(e, b);
    Old o{t.host[s], t.bytes[s], t.host[s + 1], t.bytes[s + 1]};
    t.host[s] = fc1; t.host[s + 1] = fc2;
    t.bytes[s] = fc1_bytes; t.bytes[s + 1] = fc2_bytes;
    if (cudaMemcpyAsync(t.dev + s, t.host.data() + s, 2 * sizeof(void*), cudaMemcpyHostToDevice, st) != cudaSuccess)
    {
        t.host[s] = o.fc1; t.host[s + 1] = o.fc2; t.bytes[s] = o.fc1_bytes; t.bytes[s + 1] = o.fc2_bytes;
        return "ptab::set: upload failed";
    }
    if (old) *old = o;
    return nullptr;
}

// GR8_PTAB (knobs/decode.yaml): 0 / unset = off (the pack layout), 1 = image, 2 = scatter; the knob table refuses the rest
inline Mode mode_env() { return static_cast<Mode>(knob::ptab()); }

}   // namespace ptab
