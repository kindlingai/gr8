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
The K1 split knobs at 6, 12 and 24 rows (GR8_K1_SPLIT 8x1, 2x4, /1, /4, /8 against the release 4x2/2) and GR8_K1_TLBW
(64 and 2048 KB) all measured within the run-to-run spread of the release shape (1-3%).

## Where a prefill chunk goes

The same configuration, RigMark's prefill prompts (8K, 32K and 128K), `GR8_DS_SPAN=2` on the served engine: one pass per
W-row chunk with a mark after every op. With the two prefill lanes on (the launcher default, GR8_DS_PF_LANES=2) the lanes
serialize their attention stages through stream waits and their marks interleave, so the per-op times of a two-lane run
are not the kernels' times; the table is the single-lane run (GR8_DS_PF_LANES=1, W = 1024 rows), rank 0, the mean of 495
passes of 1024 rows, 553 ms a pass. A chunk that is not a prompt's last runs layers 0-20 only (the bounded replay of
ds41_pf_split.h: layers 21-39 write no cache, so they run over the tail rows of the last chunk), 8 trellis and 13 native
layers, and the op costs are the same on both kinds:

| op | ms a pass | share | ms a layer |
|---|---|---|---|
| attn_proj (q_a, q_b, kv projections) | 119.5 | 22% | 5.7 |
| attn_out (o_proj) | 115.0 | 21% | 5.5 |
| consume_attn + consume_ffn (the peers' partials) | 93.8 | 17% | 2.2 + 2.2 |
| read-once MoE, native layers (gate, fc1, fc2, shared, combine) | 105.1 | 19% | 8.1 |
| trellis MoE, layers 0-7 (shared pass, route, fc1, mid, fc2, finalize) | 29.5 | 5% | 3.7 |
| index_topk (layers 2, 8, 14, 20) + ca_row + cand_blocks | 39.6 | 7% | 7 at a source layer |
| engram (layers 1, 14) | 26.5 | 5% | 13 |
| attn_core | 20.2 | 4% | 1.0 |

Two things set this profile. First, RigMark's prefill text routes a 1024-row chunk to only 58-115 of the 384 experts a
layer (the engine's `distinct` lines), so the expert reads that dominate a varied prompt (384 experts x 4.7 MB a rank a
layer = 1.8 GB, 7.7 ms at the memory bandwidth) are a third of that here, and the read-once kernels run at about 14
TFLOPS on 109 GFLOP a layer rather than at the bandwidth. Second, the attention projections run on the CUDA-core MXFP8
GEMM (k_gemm_mxfp8): about 32 GFLOP each for attn_proj and attn_out at 1024 rows, 5.7 and 5.5 ms a layer, under 6
TFLOPS. The block-scaled tensor-core GEMM exists behind GR8_DS_ATTN_TC=1 (default off); its effect is measured below.
The trellis layers' MoE (3.7 ms a layer) costs less than the native read-once path (8.1 ms) at these routes.

The context hardly matters: in a 128K prompt the 2048-row quantum of the two-lane run grows from 1020 ms at the start to
1171 ms at the end (+15%), so the prefill rate is set per row, and RigMark's 8K / 32K / 128K rates differ by that slope
(2,102 / 2,022 / 1,851 tok/s with two lanes; 1,946 / 1,940 / 1,806 with one: the second lane buys 8% at 8K and 2.5% at
128K). The fabric is lightly used throughout: 3.6 Gb/s mean on one rail and 1.8 on the other (12 Gb/s peaks), no
congestion marks, a few retransmits in a 25-minute run.

### Chunk width and lanes

RigMark cold prefill, tok/s, the same runs (`GR8_DS_SPAN=2` on in all of them; the box clocks at 1990-2200 MHz mean):

| | 8K | 32K | 128K | MemAvailable a box |
|---|---|---|---|---|
| W 1024, two lanes (the launcher default) | 2,102 | 2,022 | 1,851 | 9.3 GiB |
| W 1024, one lane | 1,946 | 1,940 | 1,806 | 12 GiB |
| W 4096, one lane | 1,978 | 1,890 | 1,725 | 0.3-1.1 GiB |

A 4096-row pass costs, per 1024 rows, the same as a 1024-row pass in every compute op (attn_proj 118 against 119.5 ms,
attn_out 116 against 115, read-once fc1 55 against 51: the chunk routes to 129-137 distinct experts instead of 58-115, so
the expert bytes a row fall by a third but the kernels are not bound by them), and more in the consumes (118 against 94
ms), with the memory gone. The width stays at 1024.

### The projections on the tensor cores

`GR8_DS_ATTN_TC=1` (default off: the five attention projections at 9 or more rows on the block-scaled tensor-core GEMM
k_gemm_mx8t, the same MXFP8 operands) against the same single-lane W 1024 run without it, `GR8_DS_SPAN=2` on in both:

| | 8K | 32K | 128K | a 1024-row pass | attn_proj | attn_out |
|---|---|---|---|---|---|---|
| CUDA-core GEMM (the default) | 1,946 | 1,940 | 1,806 tok/s | 553 ms | 119.5 ms | 115.0 ms |
| GR8_DS_ATTN_TC=1 | 2,998 | 3,057 | 2,828 tok/s | 351 ms | 15.5 ms | 19.1 ms |

The projections fall from 42% of the pass to 10%, the prefill rate rises 54-58% at every length, and the multi-agent
aggregates (short prompts, 256-token answers) go from 47 / 70 to 84.5 / 111.7 tok/s at 2 / 4 streams; decode is
unchanged (74.5 / 47.3 against 74.5 / 48.5 tok/s, within the spread). The two runs' prefill answers agree token for
token (RigMark samples, so the decode answers vary within a run either way). The remaining pass is the read-once MoE
(96 ms), the two consumes (90 ms), the indexer (32 ms), Engram (26 ms) and the attention core (20 ms).

### The fabric

The raw links, measured with ib_write_bw between every pair of boxes (64 KiB writes, 8 s, RoCE v2) with the engines
idle: 108.7-109.2 Gb/s on either PCIe root of every pair, 98.0 x 2 = 196 Gb/s with both roots of a pair at once, the
numbers a healthy ConnectX-7 pair gives on these boxes. The engine's prefill collectives are far from that: a layer's two
consumes take 2.2 ms each for a 1024 x 5120 BF16 partial (about 8 MB a rank each way), about 30 Gb/s effective, and the
rails average 3.6 and 1.8 Gb/s over a run. The wait is in the collective's shape and the proxy, not the wire.

