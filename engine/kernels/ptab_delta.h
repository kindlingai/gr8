// Host-only delta piece-prefetch planner for EL3 (#151 Stage 3).
// Given a rank's current routed-expert piece table and a target layout delta (from g6l_delta_halves / layout.rs),
// keeps every resident piece untouched in ptab::Pool (zero bytes fetched, zero copy), allocates ONLY missing pieces
// contiguously per expert via Pool::take_run, records ONLY missing pack byte ranges into RouteMap so g6l_plan_load_ranges
// skips resident weights, and tracks both added_slabs (for clean mid-prefetch rollback) and dropped_slabs (for
// post-cutover Pool::give + reclaim_empty). Pure host-only, zero CUDA dependency.
#pragma once
#include "ptab_pool.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ptab
{

// Pure host arithmetic matching tr3_piece_fc1_bytes / tr3_piece_fc2_bytes / unit_off (tr3.cuh).
// mb_bytes = (hidden / 16) * 8 * 32 = hidden * 16 (98304 for GLM-5.3 hidden=6144).
inline size_t host_piece_fc1_bytes(long long km, const long long* tr, int b, size_t mb_bytes)
{
    if (!km) return 2 * static_cast<size_t>(tr[4]) * mb_bytes;
    return (static_cast<size_t>((km >> (8 * b)) & 15) + static_cast<size_t>((km >> (8 * b + 4)) & 15)) * mb_bytes;
}

inline size_t host_piece_fc2_bytes(const long long* tr, size_t mb_bytes)
{
    return static_cast<size_t>(tr[4]) * mb_bytes;
}

inline size_t host_unit_off(long long km, const long long* tr, int u, size_t mb_bytes)
{
    if (!km) return static_cast<size_t>(tr[0]) + static_cast<size_t>(u) * static_cast<size_t>(tr[4]) * mb_bytes;
    size_t off = static_cast<size_t>(tr[0]);
    for (int i = 0; i < u; ++i) off += static_cast<size_t>((km >> (4 * i)) & 15) * mb_bytes;
    return off;
}

struct PieceKeep
{
    int old_b = 0;
    int new_b = 0;
};

struct ExpertDelta
{
    int row = 0;
    std::vector<PieceKeep> kept;   // (old block index -> new block index) kept in place
    std::vector<int> added;        // new block indices missing on this rank
    std::vector<int> dropped;      // old block indices dropped after cutover
};

struct SlabRef
{
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
};

struct TableDeltaResult
{
    int rows = 0, eb = 0;
    std::vector<const uint8_t*> host;   // [rows * eb * 2] staged target table
    std::vector<size_t> bytes;          // [rows * eb * 2] byte counts
    std::vector<SlabRef> added_slabs;   // newly taken from pool; release on rollback
    std::vector<SlabRef> dropped_slabs; // old slabs no longer held in target; release after cutover
    size_t kept_pieces = 0, added_pieces = 0, dropped_pieces = 0;
    size_t kept_bytes = 0, added_bytes = 0, dropped_bytes = 0;
    size_t slab_bytes = 0;              // total rounded slab bytes in target table (kept + added)

    int slot(int e, int b) const { return (e * eb + b) * 2; }
};

// Extracts sorted (offset, nbytes) extents from a finalized RouteMap for g6l_plan_load_ranges.
inline std::vector<std::pair<uint64_t, uint64_t>> route_extents(const RouteMap& route)
{
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(route.segs.size());
    for (const auto& s : route.segs)
    {
        if (s.pack_hi > s.pack_lo) out.emplace_back(s.pack_lo, s.pack_hi - s.pack_lo);
    }
    return out;
}

// Rolls back any newly-allocated slabs from a staged delta (leaves every kept resident slab untouched!).
template <class PoolT>
inline size_t rollback_delta_slabs(PoolT& pool, std::vector<SlabRef>& added_slabs, size_t* chunks_freed = nullptr)
{
    for (const SlabRef& s : added_slabs) pool.give(s.ptr, s.bytes);
    added_slabs.clear();
    return pool.reclaim_empty(chunks_freed);
}

// Retires dropped old slabs after cutover (call ONLY after a GPU fence confirms the old width is done reading them).
template <class PoolT>
inline size_t retire_dropped_slabs(PoolT& pool, std::vector<SlabRef>& dropped_slabs, size_t* chunks_freed = nullptr)
{
    for (const SlabRef& s : dropped_slabs) pool.give(s.ptr, s.bytes);
    dropped_slabs.clear();
    return pool.reclaim_empty(chunks_freed);
}

// Plans one MoE layer's in-process delta piece-prefetch:
//  - old_host / old_bytes (may be empty if this box is a joiner with old_rows == 0)
//  - new_h / new_k: target rank's tr3.table [new_rows * 6] and optional tr3.kmix [new_rows]
//  - deltas: one ExpertDelta per row 0..new_rows-1
//  - pool: shared SlabPool (kept slabs stay untouched; added slabs per expert come from one take_run for fc1 and one for fc2)
//  - route: RouteMap that receives ONLY the added pieces' pack byte ranges (fc1_pack_off + ..., fc2_pack_off + ...)
template <class PoolT>
inline const char* plan_table_delta(const std::vector<const uint8_t*>& old_host, const std::vector<size_t>& old_bytes,
                                    int old_rows, int old_eb, const std::vector<long long>& new_h,
                                    const std::vector<long long>& new_k, int new_rows, int new_eb,
                                    const std::vector<ExpertDelta>& deltas, PoolT& pool,
                                    uint64_t fc1_pack_off, uint64_t fc2_pack_off, size_t mb_bytes,
                                    RouteMap& route, TableDeltaResult& out)
{
    out = TableDeltaResult{};
    if (new_rows <= 0 || new_eb <= 0 || new_h.size() < static_cast<size_t>(new_rows) * 6 ||
        (!new_k.empty() && new_k.size() < static_cast<size_t>(new_rows)) ||
        deltas.size() != static_cast<size_t>(new_rows))
    {
        return "ptab::plan_table_delta: invalid table or delta dimensions";
    }
    out.rows = new_rows;
    out.eb = new_eb;
    const size_t n = static_cast<size_t>(new_rows) * static_cast<size_t>(new_eb) * 2;
    out.host.assign(n, nullptr);
    out.bytes.assign(n, 0);

    std::vector<size_t> add_f1_sz, add_f2_sz;
    std::vector<uint8_t*> add_f1_p, add_f2_p;
    add_f1_sz.reserve(new_eb); add_f2_sz.reserve(new_eb);
    add_f1_p.reserve(new_eb); add_f2_p.reserve(new_eb);

    for (int e = 0; e < new_rows; ++e)
    {
        const ExpertDelta& d = deltas[static_cast<size_t>(e)];
        if (d.row != e) return "ptab::plan_table_delta: delta row index mismatch";
        const long long* tr = &new_h[static_cast<size_t>(e) * 6];
        const long long km = new_k.empty() ? 0 : new_k[static_cast<size_t>(e)];
        const int nb = static_cast<int>(tr[3]);
        if (nb < 0 || nb > new_eb || nb > 8) return "ptab::plan_table_delta: invalid block count in target tr3.table";
        if (static_cast<int>(d.kept.size() + d.added.size()) != nb)
            return "ptab::plan_table_delta: kept + added count does not match target tr3.table";

        // 1. Reuse every kept piece in place (zero allocation, zero copy, unchanged slab address).
        for (const PieceKeep& pk : d.kept)
        {
            if (e >= old_rows || pk.old_b < 0 || pk.old_b >= old_eb || pk.new_b < 0 || pk.new_b >= nb)
                return "ptab::plan_table_delta: kept block index out of range";
            const size_t os = static_cast<size_t>(e * old_eb + pk.old_b) * 2;
            const size_t ns = static_cast<size_t>(out.slot(e, pk.new_b));
            if (os + 1 >= old_host.size() || !old_host[os] || !old_host[os + 1])
                return "ptab::plan_table_delta: missing resident slab for kept piece";
            const size_t want_f1 = host_piece_fc1_bytes(km, tr, pk.new_b, mb_bytes);
            const size_t want_f2 = host_piece_fc2_bytes(tr, mb_bytes);
            if (old_bytes[os] != want_f1 || old_bytes[os + 1] != want_f2)
                return "ptab::plan_table_delta: kept piece byte size mismatch between old and target layout";
            out.host[ns] = old_host[os];
            out.host[ns + 1] = old_host[os + 1];
            out.bytes[ns] = want_f1;
            out.bytes[ns + 1] = want_f2;
            out.kept_pieces += 1;
            out.kept_bytes += want_f1 + want_f2;
            out.slab_bytes += PoolT::round(want_f1) + PoolT::round(want_f2);
        }

        // 2. Allocate ONLY added pieces contiguously per expert via Pool::take_run, and route ONLY their pack ranges.
        if (!d.added.empty())
        {
            const int na = static_cast<int>(d.added.size());
            add_f1_sz.resize(static_cast<size_t>(na));
            add_f2_sz.resize(static_cast<size_t>(na));
            add_f1_p.resize(static_cast<size_t>(na));
            add_f2_p.resize(static_cast<size_t>(na));
            for (int i = 0; i < na; ++i)
            {
                const int b = d.added[static_cast<size_t>(i)];
                if (b < 0 || b >= nb) return "ptab::plan_table_delta: added block index out of range";
                add_f1_sz[static_cast<size_t>(i)] = host_piece_fc1_bytes(km, tr, b, mb_bytes);
                add_f2_sz[static_cast<size_t>(i)] = host_piece_fc2_bytes(tr, mb_bytes);
            }
            const char* err = nullptr;
            if (!pool.take_run(add_f1_sz.data(), na, add_f1_p.data(), err))
            {
                rollback_delta_slabs(pool, out.added_slabs);
                return err;
            }
            if (!pool.take_run(add_f2_sz.data(), na, add_f2_p.data(), err))
            {
                for (int i = 0; i < na; ++i) pool.give(add_f1_p[static_cast<size_t>(i)], add_f1_sz[static_cast<size_t>(i)]);
                rollback_delta_slabs(pool, out.added_slabs);
                return err;
            }
            for (int i = 0; i < na; ++i)
            {
                const int b = d.added[static_cast<size_t>(i)];
                const size_t ns = static_cast<size_t>(out.slot(e, b));
                const size_t f1b = add_f1_sz[static_cast<size_t>(i)];
                const size_t f2b = add_f2_sz[static_cast<size_t>(i)];
                uint8_t* f1p = add_f1_p[static_cast<size_t>(i)];
                uint8_t* f2p = add_f2_p[static_cast<size_t>(i)];
                out.host[ns] = f1p;
                out.host[ns + 1] = f2p;
                out.bytes[ns] = f1b;
                out.bytes[ns + 1] = f2b;
                out.added_slabs.push_back(SlabRef{f1p, f1b});
                out.added_slabs.push_back(SlabRef{f2p, f2b});
                out.added_pieces += 1;
                out.added_bytes += f1b + f2b;
                out.slab_bytes += PoolT::round(f1b) + PoolT::round(f2b);
                route.add(fc1_pack_off + host_unit_off(km, tr, 2 * b, mb_bytes), f1b, f1p);
                route.add(fc2_pack_off + static_cast<uint64_t>(tr[1]) + static_cast<uint64_t>(b) * f2b, f2b, f2p);
            }
        }

        // 3. Record dropped old slabs to be retired via Pool::give ONLY after cutover fence.
        for (int ob : d.dropped)
        {
            if (e >= old_rows || ob < 0 || ob >= old_eb)
                return "ptab::plan_table_delta: dropped block index out of range";
            const size_t os = static_cast<size_t>(e * old_eb + ob) * 2;
            if (os + 1 < old_host.size() && old_host[os] && old_host[os + 1])
            {
                out.dropped_slabs.push_back(SlabRef{const_cast<uint8_t*>(old_host[os]), old_bytes[os]});
                out.dropped_slabs.push_back(SlabRef{const_cast<uint8_t*>(old_host[os + 1]), old_bytes[os + 1]});
                out.dropped_pieces += 1;
                out.dropped_bytes += old_bytes[os] + old_bytes[os + 1];
            }
        }
    }
    return nullptr;
}

}   // namespace ptab
