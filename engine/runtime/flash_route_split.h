// G6_PF_ROUTER_SPLIT layout and row partition; shared by the CPU test and CUDA (flash_route_split.cuh).
// Each rank routes rows [r0(M, rank), r0(M, rank + 1)) and sends them as 64-byte records: the 8 tok_e ints (selection order,
// -1 = dropped route of a G6_MOE_TOPK cut) then the 8 tok_w floats, raw bits. All 8 slots travel, so a dropped slot reaches the
// peer as the -1 / 0 its owner's k_pf_route wrote and every rank ends with byte-equal tables.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef __CUDACC__
#define G6_RS_HD __host__ __device__
#else
#define G6_RS_HD
#endif
namespace flash_route_split {
constexpr int SLOTS = 8, ROW_BYTES = SLOTS * 8;                 // 8 x int32 + 8 x float32
G6_RS_HD inline int r0(int M, int p, int nr) { return p * M / nr; }          // = g6rt::pf_r0 (the GLM prefill all-gather split)
G6_RS_HD inline int rows(int M, int p, int nr) { return r0(M, p + 1, nr) - r0(M, p, nr); }
G6_RS_HD inline int padded_rows(int M, int nr) { return (M + nr - 1) / nr; }   // >= rows(M, p, nr) for every p
G6_RS_HD inline size_t payload(int M, int nr) { return size_t(padded_rows(M, nr)) * ROW_BYTES; }
// the split runs for a block of M rows on nr ranks only when every rank owns >= 2 rows; a smaller M takes the replicated path
// (every rank decides from M alone, so both ranks agree)
G6_RS_HD inline bool active(int M, int nr) { return nr >= 2 && M >= 2 * nr; }
// Host model of the kernels: rank `rank` packs its rows into `slot`; rank `rank` unpacks the slot of peer p into its tables.
inline void pack(const int32_t* tok_e, const float* tok_w, int M, int nr, int rank, uint8_t* slot) {
    const int lo = r0(M, rank, nr), n = rows(M, rank, nr);
    for (int i = 0; i < n; ++i) {
        std::memcpy(slot + size_t(i) * ROW_BYTES, tok_e + size_t(lo + i) * SLOTS, SLOTS * 4);
        std::memcpy(slot + size_t(i) * ROW_BYTES + SLOTS * 4, tok_w + size_t(lo + i) * SLOTS, SLOTS * 4);
    }
}
inline void unpack(const uint8_t* slot, int M, int nr, int p, int32_t* tok_e, float* tok_w) {
    const int lo = r0(M, p, nr), n = rows(M, p, nr);
    for (int i = 0; i < n; ++i) {
        std::memcpy(tok_e + size_t(lo + i) * SLOTS, slot + size_t(i) * ROW_BYTES, SLOTS * 4);
        std::memcpy(tok_w + size_t(lo + i) * SLOTS, slot + size_t(i) * ROW_BYTES + SLOTS * 4, SLOTS * 4);
    }
}
} // namespace flash_route_split
#undef G6_RS_HD
