// Engram token ring (DeepSeek-V4.1-Flash-1017): the n-gram window of a row, kept as one host-and-device code path so the CPU test
// (tests/test_ring.cpp) runs the same lines the GPU kernel runs (runtime/ds41_engram_ops.cuh k_engram_win).
//
// The official reference keeps a positional cache: engram.py NgramHashState.forward writes the step's token ids at cache[start_pos : start_pos + seqlen]
// and reads cache[pos - shift] for each row. The ring is that cache folded to kRing slots by absolute position. Valid for one sequence per step whose
// rows are consecutive positions (an offline sequence; one serving session; a DSpark verify chain): the step's rows are written first, then every row reads
// its three predecessors (rows of this step or entries an earlier step wrote). A rolled-back verify step needs no undo: the next step starts at
// the first rejected position and rewrites it before any row reads it.
#pragma once
#ifdef __CUDACC__
#define ENGRAM_RING_HD __host__ __device__
#else
#define ENGRAM_RING_HD
#endif

namespace ds41 { namespace engram {

constexpr int kRing = 64;   // >= MX (32 rows of a step) + 3 earlier positions; a power of two

// row at position pos with token tok
ENGRAM_RING_HD inline void ring_put(int* ring, int pos, int tok) { ring[pos & (kRing - 1)] = tok; }
// the token at position q of the current sequence, or -1 before its start (a blocked lookback; the hash maps it to the pad id)
ENGRAM_RING_HD inline int ring_get(const int* ring, int q) { return q < 0 ? -1 : ring[q & (kRing - 1)]; }

// ---- W-row chunks (prefill, rows > MX): the same window without the "rows + 3 <= kRing" limit (kernels k_win_wide + k_ring_put_tail,
// engram_dev.cuh; tests/test_engram_wide.cpp runs these lines). The chunk's rows are consecutive positions of one sequence, row r at
// pos[r] = pos[0] + r, so the token at position pos[r]-k is tok[r-k] while r-k >= 0 (this chunk, no ring involved) and else one of the
// 3 positions before the chunk, which an earlier step left in the ring. Two passes, in this order, because the second one overwrites
// ring slots the first one still reads (the rows 61..63 of a chunk of 64..127 rows share a slot with the positions pos[0]-3..-1):
//   1. every row r: win_wide_row   (reads tok, pos and the ring, writes the row's 4 window entries, writes nothing else)
//   2. every row r: ring_put_tail  (the last kRing rows of the chunk enter the ring, so the next step, a W-row chunk or a decode step
//                                    of 1..MX rows, finds its 3 earlier positions in the ring exactly as after a decode step)
// win_wide_row returns false when pos[r] != pos[0] + r (rows that are not consecutive positions): the caller counts that and the
// engine fails closed (status -2) instead of hashing a wrong window.
ENGRAM_RING_HD inline bool win_wide_row(const int* tok, const int* pos, const int* ring, int r, int* w)
{
    const int p0 = pos[0];
    for (int k = 0; k < 4; ++k) w[k] = r - k >= 0 ? tok[r - k] : ring_get(ring, p0 + r - k);
    return pos[r] == p0 + r;
}
ENGRAM_RING_HD inline void ring_put_tail(int* ring, const int* tok, const int* pos, int rows, int r)
{
    if (r >= rows - kRing) ring_put(ring, pos[r], tok[r]);
}

}}   // namespace ds41::engram
