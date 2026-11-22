// CUDA GPU-side adapter for EL4c (#160) two-phase live bulk-then-tail session migration.
// Connects `g6s2::run_migrate_phase` (live_migrate.h) to GPU attention bindings (`G6AttnBinding`), using
// `canon_mover.cuh` / D2H / H2D row transfers for remote pages and direct GPU D2D copies (or zero-copy kvchunks
// page-table retention) for local-keep pages where `src_box == dst_box`.
#pragma once

#include "live_migrate.h"
#include "canon_mover.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace g6s2 {

struct GpuMigrateSession {
    uint64_t session_id = 0;
    std::vector<const G6AttnBinding*> src_bindings;
    std::vector<const G6AttnBinding*> dst_bindings;
    std::vector<uint32_t> src_list;
    std::vector<uint32_t> dst_list;
    uint32_t src_place = UINT32_MAX;
    uint32_t dst_place = UINT32_MAX;
};

// Resolves physical page id in `list` for logical page `lp` on rank `rank` of CP layout (`cp_nr`, `below_rows`).
inline int64_t migrate_resolve_phys(const std::vector<uint32_t>& list, uint32_t place, bool is_cp,
                                    uint32_t cp_nr, uint32_t rank, uint32_t below_rows, uint32_t lp) {
    const uint32_t rp = below_rows / 64u;
    if (!is_cp || lp < rp) return lp < list.size() ? int64_t(list[lp]) : -1;
    if ((lp % cp_nr) != rank) return -1;
    uint32_t j = 0;
    for (uint32_t p = rp; p < lp; ++p) if ((p % cp_nr) == rank) ++j;
    uint32_t low_seen = 0;
    for (uint32_t phys : list) {
        if (phys < place) {
            if (low_seen == rp + j) return int64_t(phys);
            ++low_seen;
        }
    }
    return -1;
}

// Builds a `MigrateBoxIO` backed by GPU `G6AttnBinding` MLA + Indexer caches on stream `st`.
// When `src_phys == dst_phys` and `src_b->args.mla_cache == dst_b->args.mla_cache` (same VMM `kvchunks` region!),
// `copy_local` is a true zero-copy no-op; otherwise it performs a single device-to-device copy on `st`.
inline MigrateBoxIO make_gpu_migrate_io(const MigrateLayout& src_lay, const MigrateLayout& dst_lay, int my_box,
                                        const std::map<uint64_t, GpuMigrateSession>& sess_map, cudaStream_t st) {
    const int src_rank = src_lay.rank_of_box(my_box);
    const int dst_rank = dst_lay.rank_of_box(my_box);
    MigrateBoxIO io;
    io.read_rows = [&, src_rank, st](uint64_t sid, uint32_t b, uint32_t lp, uint32_t roff, uint32_t rcnt,
                                     uint8_t* mla_out, uint8_t* idx_out) {
        const auto& s = sess_map.at(sid);
        const G6AttnBinding* ab = s.src_bindings.at(b);
        const bool is_cp = ab->args.cp_nr != 0;
        const int64_t phys = migrate_resolve_phys(s.src_list, s.src_place, is_cp, src_lay.cp_nr,
                                                  uint32_t(src_rank), src_lay.below_rows, lp);
        if (phys < 0) throw std::runtime_error("gpu migrate read_rows: missing physical page");
        const size_t row0 = size_t(phys) * 64u + roff;
        const uint8_t* mla_dev = static_cast<const uint8_t*>(ab->args.mla_cache) + row0 * g6kv::ROW;
        G6RT_CK(cudaMemcpyAsync(mla_out, mla_dev, size_t(rcnt) * g6kv::ROW, cudaMemcpyDeviceToHost, st));
        if (idx_out && ab->args.index_cache) {
            const uint8_t* idx_dev = static_cast<const uint8_t*>(ab->args.index_cache) + row0 * g6kv::IDX_ROW;
            G6RT_CK(cudaMemcpyAsync(idx_out, idx_dev, size_t(rcnt) * g6kv::IDX_ROW, cudaMemcpyDeviceToHost, st));
        }
        G6RT_CK(cudaStreamSynchronize(st));
    };
    io.write_rows = [&, dst_rank, st](uint64_t sid, uint32_t b, uint32_t lp, uint32_t roff, uint32_t rcnt,
                                      const uint8_t* mla_in, const uint8_t* idx_in) {
        const auto& s = sess_map.at(sid);
        const G6AttnBinding* ab = s.dst_bindings.at(b);
        const bool is_cp = ab->args.cp_nr != 0;
        const int64_t phys = migrate_resolve_phys(s.dst_list, s.dst_place, is_cp, dst_lay.cp_nr,
                                                  uint32_t(dst_rank), dst_lay.below_rows, lp);
        if (phys < 0) throw std::runtime_error("gpu migrate write_rows: missing physical page");
        const size_t row0 = size_t(phys) * 64u + roff;
        uint8_t* mla_dev = static_cast<uint8_t*>(ab->args.mla_cache) + row0 * g6kv::ROW;
        G6RT_CK(cudaMemcpyAsync(mla_dev, mla_in, size_t(rcnt) * g6kv::ROW, cudaMemcpyHostToDevice, st));
        if (idx_in && ab->args.index_cache) {
            uint8_t* idx_dev = static_cast<uint8_t*>(ab->args.index_cache) + row0 * g6kv::IDX_ROW;
            G6RT_CK(cudaMemcpyAsync(idx_dev, idx_in, size_t(rcnt) * g6kv::IDX_ROW, cudaMemcpyHostToDevice, st));
        }
        G6RT_CK(cudaStreamSynchronize(st));
    };
    io.copy_local = [&, src_rank, dst_rank, st](uint64_t sid, uint32_t b, uint32_t lp, uint32_t roff, uint32_t rcnt) {
        const auto& s = sess_map.at(sid);
        const G6AttnBinding* sb = s.src_bindings.at(b);
        const G6AttnBinding* db = s.dst_bindings.at(b);
        const int64_t sp = migrate_resolve_phys(s.src_list, s.src_place, sb->args.cp_nr != 0, src_lay.cp_nr,
                                                uint32_t(src_rank), src_lay.below_rows, lp);
        const int64_t dp = migrate_resolve_phys(s.dst_list, s.dst_place, db->args.cp_nr != 0, dst_lay.cp_nr,
                                                uint32_t(dst_rank), dst_lay.below_rows, lp);
        if (sp < 0 || dp < 0) throw std::runtime_error("gpu migrate copy_local: missing physical page");
        const size_t srow = size_t(sp) * 64u + roff;
        const size_t drow = size_t(dp) * 64u + roff;
        uint8_t* d_mla = static_cast<uint8_t*>(db->args.mla_cache) + drow * g6kv::ROW;
        const uint8_t* s_mla = static_cast<const uint8_t*>(sb->args.mla_cache) + srow * g6kv::ROW;
        if (d_mla != s_mla) {
            G6RT_CK(cudaMemcpyAsync(d_mla, s_mla, size_t(rcnt) * g6kv::ROW, cudaMemcpyDeviceToDevice, st));
        }
        if (db->args.index_cache && sb->args.index_cache) {
            uint8_t* d_idx = static_cast<uint8_t*>(db->args.index_cache) + drow * g6kv::IDX_ROW;
            const uint8_t* s_idx = static_cast<const uint8_t*>(sb->args.index_cache) + srow * g6kv::IDX_ROW;
            if (d_idx != s_idx) {
                G6RT_CK(cudaMemcpyAsync(d_idx, s_idx, size_t(rcnt) * g6kv::IDX_ROW, cudaMemcpyDeviceToDevice, st));
            }
        }
    };
    return io;
}

} // namespace g6s2
