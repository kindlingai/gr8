# DS41 trellis experts

DeepSeek-V4.1-Flash layers can carry exlr8 trellis experts (kindlingai/deepseek-v4.1-flash-exlr8, manifest `pmcgx-k4`: layers
0-7 at K4 with the pmcgx codebook, routed and shared; the other layers keep the official MXFP4 / FP8 experts). This note
describes how the DS41 engine runs them.

## Layout

A DS41 expert is 2304 wide, 18 blocks of 128. The width files follow the GLM v2 layout (tools/convert/v2.py), with 18
blocks per expert instead of 16 and the shared expert at index 384. Rank r of TP n owns block h of expert e in layer l
when `(18 e + h + l) % n == r`, so a rank always holds whole 128-blocks and every Hadamard block stays on one rank. At
TP4 an expert puts 5 blocks on two ranks and 4 on the other two, and which two alternates with the expert's parity, so a
layer's 6,912 routed blocks split 1,728 per rank. Simulated with random routing, the busiest rank of a decode step does
3.4% more expert work than the mean at 1 row, 1.7% at 4 and 1.2% at 8.

## Engine

Trellis layers reuse the GLM K1 kernels (k1.cuh, tr3.cuh) inside the DS41 MoE ops (ds41_moe_ops.cuh):

- ST_ROUTER stays: DS41's gate, sqrtsoftplus, top-6 and slot plan. dsr::PlanT has GLM's RoutePlan fields in the same
  order, so K1Args points at it directly. The M 9..32 plan (dsr::PlanMT) gets a per-tile expert table first.
- ST_SHARED does nothing on a trellis layer: the shared expert runs as K1's trellis shared slot (SH, row 384).
- ST_EXPERTS launches K1's fc1, fc2 and finalize. The finalize writes the BF16 partial (routed with the plan's weights,
  which carry the 1.5 route scale, plus the shared expert at weight 1) to Model::sc.part[1], as the MXFP4 path does.
- fc2's prologue clamps before SwiGLU in a DS41 build (gate <= 10, up in [-10, 10]; model.py swiglu_limit).

The loader (loader-rs ds41.rs) builds each trellis layer's tr3.* tensors from the width files and the index slices
through v2plan.rs, and drops that layer's x.fc1 / x.fc2 / sh.* records.

## Not yet

Above 32 rows (prefill) the MoE op hands DS41's routes to the GLM prefill MoE (bindp::Expert with pre_routed): DS41's
plan of that size supplies each token's experts and weights, GLM's count and fill group the routes into segments of up
to 64 rows per expert, and fc1, mid and fc2 read each expert's units once per segment. The trellis shared expert runs as
its own pass first, and the routed pass's finalize adds it into the BF16 partial. The DSpark draft layers stay MXFP4.

## Measured

TP4, four GB10 boxes, the launcher defaults plus GR8_DS_MANIFEST=pmcgx-k4, RigMark 1.0.0 at reasoning effort low. The
boxes' GPU clocks follow a thermal cap (1500-2400 MHz by temperature; tools/trace/nicprof.py records them): these
runs sat at 2050-2120 MHz mean per box with the GPUs at 68-83 C.

| | native | pmcgx-k4 (layers 0-7) |
|---|---|---|
| decode, code / prose | 77.8 / 49.7 tok/s | 78.7 / 50.2 tok/s |
| 8K prefill, cold | 2,023 tok/s | 2,148 tok/s |
| aggregate, 1 / 2 / 4 streams | 64.8 / 52.8 / 75.0 tok/s | 64.7 / 51.4 / 72.1 tok/s |

## Where a decode step goes

`GR8_DS_SPAN=2` on the served engine (tools/trace/README.md), the same configuration, the step of one session with
its 5 DSpark drafts (M = 6 rows), rank 0, the mean of 39 prints of 16 sampled replays each (a profiled step runs
about 1.5% slower than an unprofiled one: RigMark code 77.6 against 78.7 tok/s):

| unit | per step | layers 0-7 (trellis) | layers 8-39 (native) |
|---|---|---|---|
| attention | 13.2 ms | 0.33 ms a layer | 0.33 ms a layer |
| consume (attention) | 1.6 ms | 0.04 ms a layer | 0.04 ms a layer |
| MoE | 22.0 ms | 0.55 ms a layer | 0.55 ms a layer |
| consume (MoE) | 4.2 ms | 0.20 ms a layer | 0.08 ms a layer |
| head | 1.4 ms | | |
| step | 42.5 ms | | |

The trellis layers' MoE costs the same as the native layers' at 6 rows (0.547 against 0.552 ms), and the consume
after a trellis layer waits 2.5 times longer for the other ranks (0.20 against 0.08 ms: about 1 ms a step). That is
the rank imbalance of the 5 / 4 block split described above, which the plan's slot order does not even out at 6 rows.

On one GPU (m4sim, layers 0-2, real text, eager) the trellis MoE is 16% faster than the native one at 1 row (0.19
against 0.23 ms a layer), equal at 24 rows (1.3 against 1.3 ms) and 12-17% faster at 512 rows (5.4 against 6.4 ms).
The K1 split knobs at 6 rows (GR8_K1_SPLIT 8x1, 2x4, /1, /4, /8 against the release 4x2/2) and GR8_K1_TLBW (64 and
2048 KB) all measured within 2% of the release shape, with the run-to-run spread of the same size.

