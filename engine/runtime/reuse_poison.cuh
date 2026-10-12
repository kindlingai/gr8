// Prefix-reuse gate arm (G6_REUSE_POISON=1, g6m4 --reuse-prefix): on a hit, rows [r0, r1) of every layer's latent (MLA)
// cache and index (DSA) cache are set to NaN bytes before the suffix prefill. r0 = the retained rows, r1 = the old
// sequence's written_end: the stale rows past the new prefix. prefix_reuse.h assumes no stale value
// contributes to a score or an output, so a poisoned run must print the same lines as the same run without poison;
// a read of a poisoned row gives NaN scores (attention error word 4) or different lines. Gate only (default off).
#pragma once
#include <cstdint>
#include "rank.cuh"

namespace g6rt {
// grid r1 - r0, block 128: one row; the latent row and the index row where this rank stores them (the same page)
__global__ void __launch_bounds__(128) k_reuse_poison(G6Attn a, int r0)
{
    const int pos = r0 + blockIdx.x, lp = pos / a.page_size, off = pos % a.page_size;
    if (lp >= a.table_size) return;
    const int pg = a.block_table[lp];
    if (pg >= 0 && pg < a.page_count)
    {
        uint8_t* d = static_cast<uint8_t*>(a.mla_cache) + ((size_t)pg * a.page_size + off) * 656;
        for (int k = threadIdx.x; k < 656; k += blockDim.x) d[k] = 0xff;        // fp8 e4m3 NaN, NaN fp32 scales
    }
    const int ip = pg;                         // index keys sit on the latent's pages (attn.cu ipage)
    if (a.index_cache && ip >= 0 && ip < a.page_count)
    {
        uint8_t* base = static_cast<uint8_t*>(a.index_cache) + (size_t)ip * a.page_size * 132;
        base[off * 128 + threadIdx.x] = 0xff;
        if (threadIdx.x == 0) reinterpret_cast<float*>(base + a.page_size * 128)[off] = __int_as_float(0x7fc00000);
    }
}
inline void reuse_poison(Model& m, int r0, int r1, cudaStream_t st)
{
    if (r1 <= r0) return;
    for (Layer& y : m.layers) k_reuse_poison<<<r1 - r0, 128, 0, st>>>(y.ab.args, r0);
    G6RT_CK(cudaGetLastError());
}
}   // namespace g6rt
