# splitbar: copy-arrival handoff and warp-owned K reuse

Private derivative of **our** `ds41_attn_pipe.cuh`, with the same Q-register
fragments, unswizzled shared layout, QK MMA order, online softmax and PV MMA
order. One kernel per launch, no prepass or scratch allocation. Window-only
and floor>0 dispatch the original off kernel, as the built-in pipe does.

## Synchronization proof

- Warp `w` consumes only K dimensions `[64*w, 64*w+64)` during PV.
  Both window copies and FP4 widening now write that same slice for all keys.
  After `__syncwarp()`, it may overwrite its slice for the next tile without
  waiting for other warps' PV. QK of the current tile has already completed
  for every warp at the score/probability readiness barriers.
- Each thread attaches its packed `cp.async` operations to a 256-arrival
  `mbarrier` using `cp.async.mbarrier.arrive.noinc`. Copy completion arrives;
  the thread does not wait there. The next iteration acquires completion with
  parity wait, after the current QK/softmax/PV work. Phase toggles only after
  the matching successful wait. Initially no packed operation is pending.
- K readiness still needs all warps before QK, and also protects the packed
  stage from overwrite while any warp is widening it. Score and probability
  readiness still need cross-warp barriers. They use named barrier 1;
  the existing all-masked-block reduction uses named barrier 2. These are
  still **full-CTA collective dependencies**, not claimed eliminated merely
  because they are no longer spelled `__syncthreads`.
- The end-of-PV CTA barrier and blocking packed-copy CTA barrier are actually
  removed. Faster warps can start next-tile widening while lagging warps
  finish current PV. Actual overlap/speed on GB10 remains to be measured.

Only launch output is written globally. Prologue and epilogue retain their
original CTA synchronization. Mixed WL=37/100 tiles use the same ownership
rule for both window and compressed rows.

## Checks

`python3 engine/attn/sweep/variants/splitbar/test_splitbar.py` exhaustively
checks each warp's write coverage and compares the three math blocks with
our original pipe source. GPU byte-EXACT smoke/all and resource checks use
the shared sweep runner, `--arch 120a --variants off splitbar`.

PTX semantics consulted: NVIDIA PTX ISA, sections `mbarrier.arrive`,
`cp.async.mbarrier.arrive`, and `mbarrier.test_wait / mbarrier.try_wait`:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
No external kernel implementation was copied.
