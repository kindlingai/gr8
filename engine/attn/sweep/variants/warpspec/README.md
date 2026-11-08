# warpspec

Private variant of our `ds41a::k_attn_cpa_t`, not a served-engine change.

- Physical warps 0–1 gather window rows with `cp.async` and widen CQ rows
  into the original single bf16 K tile. Q also loads through these warps.
- Physical warps 2–7 are consumers. Their first logical roles are 0–5;
  physical warps 2–3 additionally execute logical roles 6–7. Each logical
  role has its own original eight-by-four PV accumulator and softmax state.
- Named barrier 1 synchronizes 64 producers; barriers 2 and 3 transfer
  tile ownership across all 256 threads; barrier 4 synchronizes only the
  192 consumers between QK, softmax and PV. No CTA `__syncthreads` occurs.
- After handing off a tile, producers construct the next slot list and
  prefetch its rows while consumers compute. Widening itself cannot overlap
  PV: the single K tile is not overwritten until barrier 3 completes.
- No pre-pass, scratch allocation, input mutation, host synchronization,
  or fallback. Window, packed CQ and floor-masked CQ use this kernel.

## Exactness argument

Only ownership/scheduling changes. Logical role `w` retains its original
QK column group, all 32 sequential K=16 MMAs, score scaling, 16-lane
softmax reduction tree, bf16 probability rounding, all four ordered PV
K=16 groups, rescaling, sink, and output conversion. Separate logical-role
state prevents a physical warp's second role from altering the first.
CQ widening uses our original `cq_dq4` and scale conversion.

All-null blocks are still skipped. The two producer warp votes determine
that condition before tile-ready; no producer writes that stage's list or
votes until consumers finish using them. Consumers finish reading K/P and
all role accumulators before tile-consumed. The final denominator barrier
is consumer-only; producers exit only after the last tile-consumed.

The uneven 8-to-6 role mapping and serial two-warp widening may offset
barrier savings. Local timing is not a performance decision; the sweep
owner measures this on the target GPU.

## Run

From the repository root:

```sh
python3 engine/attn/sweep/run.py --arch 120a --variants off warpspec \
  --suite smoke --out /tmp/warpspec-smoke
python3 engine/attn/sweep/run.py --arch 120a --variants off warpspec \
  --suite all --out /tmp/warpspec-all
```

Acceptance requires every case EXACT and the three selected `k_warpspec`
instantiations to have zero stack and zero spill stores/loads. Only our own
baseline code was used; no external implementation was read or copied.
