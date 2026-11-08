# Raster sanity variant

Only CTA scheduling coordinates change. The private kernel is our cp.async
baseline with a distinct symbol and row/head-group mapping; QK, online softmax,
PV, output rounding and all input addressing for each logical row are unchanged.

The launch grid is `(heads / 16, n)`: head groups occupy the fast dimension.
Within each complete 64-row tile, transpose its 8-by-8 row coordinates. A short
final tile keeps natural row order. This is a bijection, with no padded blocks,
additional kernel, scratch allocation or host synchronization.

Hypothesis: a different launch raster may improve inter-CTA key reuse in L2.
This is a cheap sanity arm, not a prediction of a win: at 16 local heads there
is only one head group, so only the row permutation can matter. Adjacent rows
already have substantial key locality in the baseline. GPU scheduling order is
not guaranteed by the grid, and a shared development GPU cannot decide a winner.

CPU structural check from the repository root:

```
python3 engine/attn/sweep/variants/raster/test_mapping.py
```

The local sweep `smoke` suite must report EXACT against `off`; every raster
specialization must compile with zero stack and spills. The full `all` suite
and performance selection belong to the separately scheduled native target-GPU
benchmark.
