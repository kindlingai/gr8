// CUDA device<->host page mover for canonical session KV export and import (#159).
// Uses compact owned-page planning (`canon_mover.h`) so only the pages this rank actually holds for the
// session's row range are transferred between GPU memory (`G6AttnBinding`) and pinned host memory, in bounded
// batches using 16-byte vectorized `uint4` transfers.
#pragma once

#include "canon_mover.h"
#include "../rank.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace g6s2 {
namespace canon_dev {

constexpr size_t MLA16 = 64 * g6kv::ROW / 16;
constexpr size_t IDX16 = 64 * g6kv::IDX_ROW / 16;

struct BindingPtr {
    uint4* mla;
    uint4* idx;
    unsigned long long mla_off16;
    unsigned long long idx_off16;
    int has_idx;
    int phys;
};

__device__ __forceinline__ void copy16(uint4* h, uint4* d, size_t n, int out) {
    for (size_t k = threadIdx.x; k < n; k += blockDim.x) {
        if (out) h[k] = d[k];
        else d[k] = h[k];
    }
}

// grid (n_pages, n_bindings): page i -> physical page phys_pages[i] in binding b.
// host layout per page slot: binding-major offsets (mla_off16, idx_off16) within one page stride `stride16`.
__global__ void k_canon_move_pages(const uint32_t* phys_pages, const BindingPtr* bindings, uint4* host,
                                   unsigned long long stride16, int out) {
    const BindingPtr bp = bindings[blockIdx.y];
    const uint32_t p = phys_pages[blockIdx.x];
    if (int(p) >= bp.phys) return;
    uint4* slot = host + size_t(blockIdx.x) * stride16;
    copy16(slot + bp.mla_off16, bp.mla + size_t(p) * MLA16, MLA16, out);
    if (bp.has_idx) copy16(slot + bp.idx_off16, bp.idx + size_t(p) * IDX16, IDX16, out);
}

inline std::vector<uint8_t> binding_flags(const std::vector<const G6AttnBinding*>& bindings) {
    std::vector<uint8_t> f(bindings.size(), 0);
    for (size_t i = 0; i < bindings.size(); ++i) {
        uint8_t v = 0;
        if (bindings[i]->args.index_cache) v |= 1u;
        if (bindings[i]->args.cp_nr != 0) v |= 2u;
        f[i] = v;
    }
    return f;
}

struct BatchBuffers {
    std::vector<BindingPtr> ptrs;
    BindingPtr* d_ptrs = nullptr;
    uint32_t* h_pages = nullptr;
    uint32_t* d_pages = nullptr;
    uint8_t* h_stage = nullptr;
    uint8_t* d_stage = nullptr;
    size_t stride = 0;
    size_t batch = 0;

    BatchBuffers(const std::vector<const G6AttnBinding*>& bindings, size_t max_batch) : batch(max_batch) {
        size_t off = 0;
        ptrs.reserve(bindings.size());
        for (const G6AttnBinding* b : bindings) {
            const size_t mla_off = off;
            off += 64u * g6kv::ROW;
            const size_t idx_off = off;
            const int has_idx = b->args.index_cache ? 1 : 0;
            if (has_idx) off += 64u * g6kv::IDX_ROW;
            ptrs.push_back({static_cast<uint4*>(b->args.mla_cache), static_cast<uint4*>(b->args.index_cache),
                            mla_off / 16, idx_off / 16, has_idx, b->args.page_count});
        }
        stride = (off + 15u) / 16u * 16u;
        G6RT_CK(cudaMalloc(&d_ptrs, ptrs.size() * sizeof(BindingPtr)));
        G6RT_CK(cudaMemcpy(d_ptrs, ptrs.data(), ptrs.size() * sizeof(BindingPtr), cudaMemcpyHostToDevice));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&h_pages), batch * sizeof(uint32_t), cudaHostAllocMapped));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_pages), h_pages, 0));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&h_stage), stride * batch, cudaHostAllocMapped));
        G6RT_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_stage), h_stage, 0));
    }
    ~BatchBuffers() {
        if (d_ptrs) cudaFree(d_ptrs);
        if (h_pages) cudaFreeHost(h_pages);
        if (h_stage) cudaFreeHost(h_stage);
    }
    BatchBuffers(const BatchBuffers&) = delete;
    BatchBuffers& operator=(const BatchBuffers&) = delete;
};

} // namespace canon_dev

// Downloads only this rank's owned pages of `bindings` for [first_row, first_row + rows) into a `CanonShard`.
inline CanonShard canon_export_bindings(const std::vector<const G6AttnBinding*>& bindings,
                                        const std::vector<int>& list_int, uint32_t place, uint32_t nr, uint32_t rank,
                                        uint32_t below_rows, uint64_t first_row, uint64_t rows, cudaStream_t st) {
    const uint32_t nb = uint32_t(bindings.size());
    const std::vector<uint8_t> flags = canon_dev::binding_flags(bindings);
    std::vector<uint32_t> list(list_int.size());
    for (size_t i = 0; i < list_int.size(); ++i) list[i] = uint32_t(list_int[i]);

    const CanonPlan plan = canon_plan_owned(flags.data(), nb, nr, below_rows, list.data(), list.size(),
                                            place, rank, first_row, rows);
    CanonShard shard;
    shard.mla_row = g6kv::ROW;
    shard.nb = nb;
    shard.nr = nr;
    shard.rank = rank;
    shard.below_rows = below_rows;
    shard.compact_place = plan.compact_place;
    shard.first_row = first_row;
    shard.rows = rows;
    shard.owned_pages = uint32_t(plan.phys_pages.size());
    shard.flags = flags;
    shard.compact_list = plan.compact_list;
    shard.mla.resize(nb);
    shard.idx.resize(nb);
    const size_t mla_page = 64u * g6kv::ROW;
    const size_t idx_page = 64u * g6kv::IDX_ROW;
    for (uint32_t b = 0; b < nb; ++b) {
        shard.mla[b].resize(size_t(shard.owned_pages) * mla_page);
        if (flags[b] & 1u) shard.idx[b].resize(size_t(shard.owned_pages) * idx_page);
    }
    if (plan.phys_pages.empty()) return shard;

    G6RT_CK(cudaStreamSynchronize(st));
    const size_t batch = std::min<size_t>(plan.phys_pages.size(), 32u);
    canon_dev::BatchBuffers buf(bindings, batch);
    for (size_t off = 0; off < plan.phys_pages.size(); off += batch) {
        const size_t n = std::min(batch, plan.phys_pages.size() - off);
        for (size_t i = 0; i < n; ++i) buf.h_pages[i] = plan.phys_pages[off + i];
        canon_dev::k_canon_move_pages<<<dim3(unsigned(n), unsigned(nb)), 256, 0, st>>>(
            buf.d_pages, buf.d_ptrs, reinterpret_cast<uint4*>(buf.d_stage), buf.stride / 16, 1);
        G6RT_CK(cudaGetLastError());
        G6RT_CK(cudaStreamSynchronize(st));
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* slot = buf.h_stage + i * buf.stride;
            for (uint32_t b = 0; b < nb; ++b) {
                const auto& bp = buf.ptrs[b];
                std::memcpy(shard.mla[b].data() + (off + i) * mla_page, slot + bp.mla_off16 * 16, mla_page);
                if (bp.has_idx) {
                    std::memcpy(shard.idx[b].data() + (off + i) * idx_page, slot + bp.idx_off16 * 16, idx_page);
                }
            }
        }
    }
    return shard;
}

// Unpacks a `"g6canon1"` blob for this rank and uploads its owned pages into `bindings` on `st`.
inline uint64_t canon_import_bindings(const std::vector<const G6AttnBinding*>& bindings,
                                      const std::vector<int>& list_int, uint32_t place, uint32_t nr, uint32_t rank,
                                      uint32_t below_rows, const uint8_t* blob, size_t blob_n, cudaStream_t st) {
    const uint32_t nb = uint32_t(bindings.size());
    const std::vector<uint8_t> flags = canon_dev::binding_flags(bindings);
    std::vector<uint32_t> list(list_int.size());
    for (size_t i = 0; i < list_int.size(); ++i) list[i] = uint32_t(list_int[i]);

    const CanonUnpacked u = canon_unpack_for_rank(blob, blob_n, g6kv::ROW, nb, flags.data(), nr, below_rows,
                                                  list.data(), list.size(), place, rank);
    if (u.plan.phys_pages.empty()) return u.written;

    G6RT_CK(cudaStreamSynchronize(st));
    const size_t mla_page = 64u * g6kv::ROW;
    const size_t idx_page = 64u * g6kv::IDX_ROW;
    const size_t batch = std::min<size_t>(u.plan.phys_pages.size(), 32u);
    canon_dev::BatchBuffers buf(bindings, batch);
    for (size_t off = 0; off < u.plan.phys_pages.size(); off += batch) {
        const size_t n = std::min(batch, u.plan.phys_pages.size() - off);
        for (size_t i = 0; i < n; ++i) {
            buf.h_pages[i] = u.plan.phys_pages[off + i];
            uint8_t* slot = buf.h_stage + i * buf.stride;
            for (uint32_t b = 0; b < nb; ++b) {
                const auto& bp = buf.ptrs[b];
                std::memcpy(slot + bp.mla_off16 * 16, u.mla[b].data() + (off + i) * mla_page, mla_page);
                if (bp.has_idx) {
                    std::memcpy(slot + bp.idx_off16 * 16, u.idx[b].data() + (off + i) * idx_page, idx_page);
                }
            }
        }
        canon_dev::k_canon_move_pages<<<dim3(unsigned(n), unsigned(nb)), 256, 0, st>>>(
            buf.d_pages, buf.d_ptrs, reinterpret_cast<uint4*>(buf.d_stage), buf.stride / 16, 0);
        G6RT_CK(cudaGetLastError());
        G6RT_CK(cudaStreamSynchronize(st));
    }
    return u.written;
}

} // namespace g6s2
