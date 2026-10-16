# Flash main-model decode loop

The Flash main model uses `runtime/flash_step.cuh` and `mhc/mhc.cuh`.
The `rank.cuh` entry points dispatch on the allocated `Model::resid4`, so the
multi-rank runtime and the staged m4sim loop call the same block functions.
The existing GLM path stays behind its original compile-time target.

## State and input contract

- `resid4`: FP32 `[MX,4,HID]`, all streams on every rank.
- `hc_post`, `hc_comb`: the pending block's FP32 output coefficients.
- `xn`, `xn2`: FP32 attention/MLP normalized input. `x`, `x2` are FP16 copies
  for reused code. FP16/BF16 fragment tables round directly from FP32.
- `row_slot`: device `[MX]`; step input sets all rows to slot zero for this
  first single-session decode path. Rows run in position order.
- `resid`: final stream mean `[MX,HID]`, retained for legacy diagnostics.
- Attention and MLP produce BF16 rank partials. Consume waits for the fabric,
  reads each rank's partial in ascending rank order, updates all four streams,
  and computes the next flat-RMS inverse in the same kernel.

The loop runs attention, consume/mHC, MLP, consume/mHC per layer.
`geom::is_kda` selects KDA or DSA; `geom::is_moe` selects dense or MoE.
Embedding initializes four equal streams. The final head consumes the
normalized stream mean and the loader's BF16 `G.lm_head.gv`, not MXFP8.

`build_flash` allocates all state and calls the block binders. `prepare_flash`
constructs row-count-specific block arguments before graph capture. Launch
functions do not allocate or copy host buffers. The final head retains the
existing sampling and head-merge code.

## Scope

Run this decode path with `G6_CP=0`, without `--prefill`, `--s2`, or `--mtp`.
The compiled CP helpers retain the GLM launch width; a wider BF16 row only
changes their compile-time copy width, not the unsupported Flash CP status.

This is the first decode path, not Flash prefill, S2 slot lifecycle, CP, or
speculative state rollback. The MTP layer-45 decode hooks do not use mHC;
see `../../docs/FLASH-MTP.md` for their per-op gate and remaining integration work. `G6_RESID_DUMP` in m4sim records width `4*HID` for Flash and `HID`
for GLM; the rest of the dump framing remains unchanged.

## Gates

The standalone mHC gates are in `mhc/README.md`. They include real-weight
per-op checks and a four-layer residual replay. The latter uses W0 block
outputs, not W3-W5 kernels. An integrated first-token or full-chain claim
requires the actual KDA, DSA and MLP blocks and their own gates.

The CPU dump-reader check is `python3 -m unittest discover -s engine/mhc -p test_dump.py`.
For a four-layer prompt run with the exact golden token IDs and `--gen 0`, use
`python3 engine/mhc/check_dump.py RESID_DUMP prompt-bf16.safetensors --layer3 layer3-K4.safetensors`.
The comparator checks all nine residual snapshots at the 2e-2 chain gate.
It does not certify the prompt IDs (which the dump format does not store), logits,
or first-token correctness.

### Integrated four-layer gate

The actual KDA, DSA and MLP kernels passed the layers 0-3 residual gate on
sm_121a at TP4 with the lossless dense export, K4 experts and BF16 KV.
The run used all 64 official prompt tokens, no generated-input steps, and
both M=1 and M=8 decode chunks. Each run passed all nine residual snapshots
against `prompt-bf16` (through L03.mid) and `layer3-K4` (L03.out).

| Metric | M=1 | M=8 | Gate |
|---|---:|---:|---:|
| Maximum snapshot rel-L2 | 0.00311898 | 0.00311898 | 0.02 |
| L03.out rel-L2 | 0.00130328 | 0.00130357 | 0.02 |
| Rank step-line differences | 0 / 64 steps | 0 / 8 steps | 0 |

M=8 used the final rebased runtime and partial-planner implementation. M=1
preceded the KDA helper extraction in the rebase. These are correctness
results, not performance measurements. They do not establish full-model,
logit, graph-replay, prefill, S2, CP or MTP correctness. The partial release
planner has a separate CPU C ABI gate; these GPU runs used native fixture
packs made by the same W1 selected planner.
