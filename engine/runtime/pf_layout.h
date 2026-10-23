// Host-safe prefill fabric layout math (no CUDA): owner rows, RS slice strides, slot sizes, the RS element->owner
// mapping and the Flash sequence-parallel (SP) five-collective plan. Shared by the device kernels (prefill.cuh,
// flash_prefill.cuh) and the CPU layout gates (engine/tests/test_flash_sp_layout.cpp), so both sides compute every
// offset, byte count, tail position and collective sequence from this one source. fabric.h is host-safe; coll_seq
// lives there. HID-parameterized _rb/_h forms let the CPU gates run the same arithmetic at a tiny row width.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
#include "../comms/fabric.h"

namespace g6rt
{
#ifdef __CUDACC__
#define PF_LAYOUT_HD __host__ __device__
#else
#define PF_LAYOUT_HD
#endif

PF_LAYOUT_HD inline int pf_r0(int M, int p, int nr) { return p * M / nr; }          // first row of rank p
PF_LAYOUT_HD inline int pf_rows(int M, int p, int nr) { return pf_r0(M, p + 1, nr) - pf_r0(M, p, nr); }
PF_LAYOUT_HD inline size_t pf_ss(const g6::FabricDev& f) { return (f.stride / f.nr) & ~size_t(255); }
// the owner of global row `row` of an M-row sub-batch: the largest p with pf_r0(M, p) <= row (k_put_rs's mapping)
PF_LAYOUT_HD inline int pf_owner_of(int M, int row, int nr) { return ((row + 1) * nr - 1) / M; }
// RS slice stride for row_byte-byte rows (BF16 partial: HID * 2), tail included, 256-B aligned
inline size_t pf_ss_min_rb(int Mmax, int nr, size_t row_bytes) { return ((size_t)(Mmax + nr - 1) / nr * row_bytes + 8 + 255) & ~size_t(255); }
// FabricConfig::slot_bytes of a prefill region over row_byte-byte rows
inline size_t pf_slot_bytes_rb(int Mmax, bool rs, int nr, size_t row_bytes)
{
    const size_t ar = (size_t)Mmax * row_bytes;
    return rs ? std::max(ar, nr * pf_ss_min_rb(Mmax, nr, row_bytes)) : ar;
}

// ---------------------------------------------------------- Flash sequence-parallel (SP) plan
// G6_PF_RS=2 (Flash only, default off): five collectives per layer-step. In time order c0 = RS of the BF16 attention
// partial, c1 = AG of the owner's FP32 normalized rows (xn2), c2 = RS of the BF16 MLP partial, c3 = AG of the FP32
// normalized rows (xn, or the final norm rows on the last layer), c4 = AG of the owner's FP32 residual rows, enqueued
// on the LAST layer-step only. Every rank, including an owner with zero rows, publishes the c1/c3 (and final c4)
// doorbells: the proxy waits for every enqueued collective of a step.
constexpr unsigned sp_rs_mask = 0x5;      // Fabric::Rs mask: c0 and c2 are reduce-scatter
constexpr int sp_ncoll = 5;
constexpr int sp_c_attn_rs = 0, sp_c_attn_ag = 1, sp_c_mlp_rs = 2, sp_c_mlp_ag = 3, sp_c_resid_ag = 4;
// RS: sender r packs its own BF16 partial bytes of owner p's rows, pf_rows(M, p) * HID * 2 of them, into slice p
inline uint32_t sp_rs_bytes(int M, int p, int nr, int hid) { return (uint32_t)((size_t)pf_rows(M, p, nr) * hid * 2); }
// AG: the owner's FP32 normalized / residual rows, pf_rows(M, p) * HID * 4 bytes, at offset 0 of its send slot
inline uint32_t sp_ag_bytes(int M, int p, int nr, int hid) { return (uint32_t)((size_t)pf_rows(M, p, nr) * hid * 4); }
// enqueue_step sizes of one SP layer-step: {0, mine4, 0, mine4} + mine4 on the last layer (c4). c0/c2 are Rs
// collectives (sizes ignored by the proxy); mine4 = this rank's FP32 AG bytes.
inline std::vector<uint32_t> sp_sizes(int M, int me, int nr, int hid, bool last)
{
    const uint32_t mine4 = sp_ag_bytes(M, me, nr, hid);
    std::vector<uint32_t> s{0u, mine4, 0u, mine4};
    if (last) s.push_back(mine4);
    return s;
}
}   // namespace g6rt
