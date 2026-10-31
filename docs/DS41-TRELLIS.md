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

A prefill chunk above 32 rows runs on a trellis layer as 32-row K1 passes, which reads the routed experts once per pass.
A prefill kernel that reads each expert once (GLM's tr3_gemm.cuh route) comes after the decode path is measured. The
DSpark draft layers stay MXFP4.
