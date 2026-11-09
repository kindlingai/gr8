#pragma once
// Which rows of one k_kv_prep launch write their window ring slot (pos % R). One rule, shared by the kernel
// (ds41_attn.cuh k_kv_prep) and the host test (tests/ds41_ring_keep_test.cpp).
#ifdef __CUDACC__
#define DS41_RING_KEEP_FN __host__ __device__ __forceinline__
#else
#define DS41_RING_KEEP_FN inline
#endif
namespace ds41a {
// No session map (one sequence): the last R rows of the launch. With a session map (SESS > 1) the rows of
// one slot are one contiguous run: a DS41 ragged-pack segment (ds41::pack: unique slots, first 0, at most 64 rows) or a
// uniform session block (row / rows_per_sess). Retention is per run: a row writes when it is among the last R rows of
// its own run. A run's positions are consecutive, so no two writing rows of one run share a ring slot. A rule over the
// whole packed launch (row >= n - R) would drop every row before n - R: at R 175 an 8 x 38 pack would write nothing
// for slots 0-2.
DS41_RING_KEEP_FN bool ring_keep(int row, int n, int R, const int* dsess)
{
    if (!dsess) return row >= n - R;
    const int s = dsess[row];
    int end = row + 1;
    while (end < n && end - row <= R && dsess[end] == s) ++end;   // past R later rows of its run the row is not kept
    return row >= end - R;
}
}
#undef DS41_RING_KEEP_FN
