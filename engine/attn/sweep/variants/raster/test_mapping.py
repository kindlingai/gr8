#!/usr/bin/env python3
"""CPU checks for a bijective raster and an unchanged arithmetic body."""
from pathlib import Path

here = Path(__file__).resolve().parent
base = (here.parents[2] / 'ds41_attn.cuh').read_text()
start = base.index('template <bool FL, bool CQ = false>\n__global__ void __launch_bounds__(AT_THREADS, 1) k_attn_cpa_t')
stop = base.index('\n#include "ds41_attn_pipe.cuh"', start)
original = base[start:stop]
private = (here / 'kernel.cuh').read_text()
private = private[private.index('template <bool FL, bool CQ = false>'):private.index('\n} // namespace sweep_raster')]
private = private.removesuffix('\n')
a = private.index('    // Heads run in the fast grid dimension;')
b = private.index('    const int sh = tid >> 4', a)
private = private[:a] + '    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;\n' + private[b:]
private = private.replace('raster_attn_t(AttnP p)', 'k_attn_cpa_t(AttnP p)')
assert private == original, 'unexpected arithmetic/body difference'
print('PASS: kernel body equals off after undoing symbol and CTA mapping only')

lengths = list(range(1, 258)) + [511, 512, 513, 8191, 8192, 8193]
for n in lengths:
    rows = []
    for row in range(n):
        tile, within = row & ~63, row & 63
        rows.append(tile + ((within & 7) << 3) + (within >> 3) if tile + 64 <= n else row)
    assert sorted(rows) == list(range(n)), n
    for groups in (1, 2, 4):
        # CUDA grid x is head group, y is the logical row.
        pairs = {(row, hg) for row in rows for hg in range(groups)}
        assert len(pairs) == n * groups
print(f'PASS: {len(lengths)} row lengths x 3 head-group counts; bijective, tails included')
