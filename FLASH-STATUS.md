# GLM 5.3 Flash on gr8: status (paused 2026-10-16)

This branch (`flash/current`) is one squashed snapshot of the best measured Flash source: the TP4/TP2 "speed
stack" (internal source commit 1d3e96f, on top of gr8 main d9188e3). Run evidence, logs and per-run results are
not in this branch. All other Flash experiments are on `flash/*` branches, one squashed snapshot each.

The work is paused. Short version: good at 8K, but well behind the gx10 Flash build on long prompts and at clean
(top-k 8) quality.

## Build (CPU, on a Spark, aarch64, CUDA sm_121a)

    MODEL=flash G6_NR=4 KV_FORMAT=fp8 scripts/build.sh bin4    # TP4 -> bin4/g6m4_flash, bin4/m4sim_flash
    MODEL=flash G6_NR=2 KV_FORMAT=fp8 scripts/build.sh bin2    # TP2 -> bin2/g6m4_flash

`MODEL=flash` adds `-DG6_MODEL_FLASH=1` (geometry in `engine/model_geom.h`, Flash attention in
`engine/attn/flash_attn.cu` + `flash_bind.cu`). fp8 KV only. Host tests: `scripts/host_tests.sh`.

## Boot (offline, one process per rank)

You need a Flash exlr8 release per rank (dense, experts, index-slices, manifests/exlr8-manifest-k4.json). It is
not in the repo. Run the same command on every rank with `--rank 0..N-1`. `<ips>` = the data-rail IPs of all
ranks in rank order, `<ips2>` = the second rail.

TP4 speed stack (the 3.35 s / 13.92 s rows):

    G6_CP=0 G6_PF_RS=0 G6_PF_OWN=0 G6_PF_POOL_OWNER=0 G6_PF_STRIPE=0 \
    G6_FPF_P1=0 G6_FPF_DSA_P1=0 G6_FPF_ATTN_P1=0 G6_PF_SPARSE_MLA_P1=0 G6_PF_SPARSE_MLA_HYBRID=0 \
    G6_PF_SPARSE_MLA=1 G6_PF_SPARSE_MLA_512=1 G6_PF_SPARSE_MLA_V2=1 G6_PF_SPARSE_MLA_H16C=1 \
    G6_PF_DTR8=1 G6_PF_SHFP8=1 G6_PF_FC2V2=1 \
    G9_KDA_SCAN=1 G9_KDA_VS=2 \
    G6_PF_KDA_WS=1 G6_PF_DSA_FULL_GV=1 G6_MOE_TOPK=4 G6_PF_ROUTER_SPLIT=1 G6_PF_DSA_F16GEMM=1 \
    G6_PF_PREWARM=1 G6_PF_STAGE_ASYNC=1 G6_PF_MHC_STAGE=1 G6_PF_MHC_FUSE=2 \
    bin4/g6m4_flash --rank R --ips <ips> --pf-ips2 <ips2> --proxy-cpu 19 --image \
      --v2-dense $REL/dense --v2 $REL/experts --v2-slices $REL/index-slices \
      --v2-widths $REL/manifests/exlr8-manifest-k4.json --v2-dense-form trellis \
      --layers all --prompt @prompt8192.ids --gen 8 --cap 8704 --chunk 1 \
      --prefill 4096 --pf-lanes 2 --graph

For 32K: `--prompt @prompt32768.ids --cap 33280`.

TP2 speed stack (the 16.82 s 32K row):

    G6_CP=0 G6_PF_RS=0 G6_PF_KDA_WS=1 G6_PF_SPARSE_MLA=1 G6_PF_SPARSE_MLA_512=1 G6_PF_SPARSE_MLA_V2=1 \
    G6_PF_SHFP8=1 G6_PF_DTR8=1 G6_PF_FC2V2=1 G6_PF_DSA_F16GEMM=1 G6_PF_ROUTER_SPLIT=1 \
    G9_KDA_SCAN=1 G9_KDA_VS=2 G6_MOE_TOPK=4 \
    bin2/g6m4_flash --rank R --ips <ip0>,<ip1> --pf-ips2 <ip0b>,<ip1b> --proxy-cpu 19 --image \
      --v2 $REL/experts --v2-slices $REL/index-slices \
      --v2-widths $REL/manifests/exlr8-manifest-k4.json --v2-dense $REL \
      --layers 0,1,...,44 --prompt @prompt32768.ids --gen 8 --temp 0 --cap 33280 \
      --prefill 8192 --pf-lanes 2 --graph

The TP2 row was measured on an earlier source (internal 8993df6), which this tree contains.

## Default flags

Source defaults are the stock engine: `G6_MOE_TOPK` 8, and router split, DSA f16 GEMM and stage async off at
TP4. The speed stack is turned on by the env above. Defaults that are already on: `G6_PF_KDA_WS=1`,
`G6_PF_DSA_FULL_GV=1`, `G6_PF_MHC_STAGE=1`, `G6_PF_MHC_FUSE=2`, `G6_PF_PREWARM=1` at TP2. `G6_MOE_TOPK=4` changes
the model output (see quality). It is not a safe default.

## Best measured numbers

Offline ready-to-first-token, full model (45 layers), K4 experts, GPU clock cap 2000 MHz, cold first request
unless marked. Not served TTFT.

| Setup | 8K | 32K |
|---|---:|---:|
| TP2 base engine | 6.10 s | 22.40 s |
| TP2 speed stack, top-k 4 | 4.48 s cold (4.32 s warm-up proxy) | 16.82 s (8,192-token chunks) |
| TP2 speed stack, top-k 8 | 5.11 s | 20.29 s (4,096-token chunks) |
| TP4 base engine | 6.70 s | 21.48 s |
| TP4 speed stack, top-k 4 | **3.35 s** | **13.92 s** (4,096-token chunks) |
| TP4 speed stack, top-k 8 | 3.73 s | not run |
| gx10 Flash build, TP2, warm | 4.23 s | not run |
| gx10 Flash build, TP4 | not run | ~6.64 s |

Quality (teacher-forced, structured prompt set, FP32 reference, TP2 stack): top-k 8 KL 0.0056, top-1 250/256;
top-k 6 KL 0.100, 241/256; top-k 4 KL 0.153, 234/256. Report only; no quality gate passed.

## Live vs killed

Live (in the stack): routed top-k cut 8 -> 4 (renormalized), FP16 DSA GEMM path, sparse MLA (512, v2, H16c),
router split, KDA scan kernel (G9), KDA workspace, the `G6_PF_SHFP8` / `G6_PF_DTR8` / `G6_PF_FC2V2` prefill
paths, mHC stage + fuse, stage async, 4,096-token prefill chunks (8,192 is faster at 32K on TP2).

Killed (tried, dropped): eager module loading, GPU clock lock, 2x DSA head split, FP8 reduce, FP8 projections,
cuBLASLt grouped GEMM, two-stream overlap, mHC reduce register tiling (exact, 141 ms slower), min-blocks hint on
the mHC kernel, boot-time warm-up hook (exact, 211 ms faster first request, +4.6 s boot).

## Open next steps

1. TP4 at 32K with 8,192-token prefill chunks (wins on TP2, not tested on TP4; memory not measured).
2. Split the mHC reduce across ranks on TP4 (each rank repeats the full reduce now). Source:
   `flash/G3-stack` (SP-mHC, `G6_PF_RS=2`); host tests pass, no GPU run.
3. FP4 MoE: 1.12-1.31x faster per layer at 32K in a synthetic one-layer test, slower at 8K; no real-weights run.
   Source: `flash/G5-tp4`.
4. A full four-rank 32K profile. Rank 0 only: MoE ~28%, communication + mHC reduce ~22%, DSA ~20%; no single
   stage explains the gap to gx10.
5. Decode speed was not measured on the speed stack.
