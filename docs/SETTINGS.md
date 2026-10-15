# Engine settings

`./gr8 glm53 up` builds every rank's command line from `scripts/engine_args.sh` and sets everything below. Every rank
gets the same environment and flags, except `--rank`. The values shown are what TP=6 serves with; KNOBS.md has each TP's
defaults.
The source comments in `engine/runtime/` (`g6m4.cu`, `prefill.cuh`, `rank.cuh`) are the full reference. This file is
the short version.

## Engine flags (`g6m4`)

| flag | served value | meaning |
|---|---|---|
| `--rank R` | 0..TP-1 | this process's rank; rank r loads `rank<r>.pack` |
| `--ips a,..` / `--ips2 a,..` | from `cluster.env` | RoCE v2 address of every rank on rail 1 / rail 2, in rank order (their count is the TP width) |
| `--proxy-cpu N` | 19 | CPU core for the decode fabric's proxy thread. Every other fabric's proxy has its own core below it: N-1 and N-2 the two prefill lanes, N-3 the CP fabric, N-4 CP staging, N-5 the QP roll, N-6 the MTP draft, N-7 DFlash; N-9 the replicated-GEMV gathers when `G6_QA_SPLIT`, `G6_IDXQ_SPLIT` or `G6_MTP_SPLIT` is on (`G6_QA_SPLIT_CPU` moves it) |
| `--manifest` / `--pack` | `packs/rank<r>.*` | this rank's pack; `--pack -` reads it from stdin |
| `--v2 DIR` / `--v2-slices DIR` / `--v2-widths FILE` | the release (`V2`, `V2_SLICES`, `--variant`) | the routed and shared experts from the exlr8 width files at the index slices' offsets, at the manifest's widths, in place of the pack's |
| `--v2-dense DIR` | the release (`V2_DENSE`) | with `--v2`: no `--manifest` / `--pack`. Every other weight comes from the release's `dense/` files, planned at load (`engine/loader-rs/src/release.rs`); rank 0 prints which form each matrix took. Under `G6_PF_OWN=3` the image also holds the full-head q_b / o_proj / UK / UV, so no QO pack |
| `--v2-dense-form trellis\|rounded` | trellis (`V2_DENSE_FORM`) | trellis: q_b, o_proj and the dense MLP stored as trellis stay the release's entries for the dense trellis kernels (`engine/kernels/dtr.cuh`). rounded, the fallback: they are rounded to MXFP8 and int4 at load (not the release's weights) |
| `--image` | on | load the whole pack into one device image (required for full packs) |
| `--layers all` | all | all 78 layers + the MTP layer |
| `--qo-manifest` / `--qo-pack` | `packs/qo.*` | full-head attention weights for `G6_PF_OWN=3` (identical file on every node) |
| `--qp-manifest` / `--qp-pack` | (TP4, not used) | resident pair weights for `G6_PF_OWN=4` (`tools/convert/pairpack.py`, one file per rank); the TP4 launcher uses `G6_PF_QP_ROLL` instead |
| `--dflash-manifest` / `--dflash-pack` | TP4 launcher default (`DFLASH=1`; `DFLASH=0` for MTP) | replaces MTP with the DFlash2 block drafter (`engine/runtime/dflash.cuh`), packs from `tools/convert/dflash2.py` (one per rank, about 0.86 GB). It drafts 7 tokens per round in one forward. Offline (`--prompt`) it replaces `--mtp`. Served, `--mtp` stays on: a request alone at its prompt tail drafts with DFlash2, the others with MTP. TP4 builds (the rank count must divide the drafter's 8 KV heads) |
| `--dflash-sel 0\|1` | 1 (`DFLASH_SEL`) | 1 = the DFlash2 selector walk over the top-16 candidates of each position, 0 = the drafter's argmax (sampled with the target's position keys) |
| `--dflash-k K` | 7 (`DFLASH_K`) | the drafts a round verifies (1..7); a verify step carries K + 1 rows. The drafter always drafts the whole block |
| `--spec-accept coupled\|rejection` | coupled (`SPEC_ACCEPT`) | how verify steps accept drafts at temperature > 0, for MTP and the DFlash2 drafter. coupled: drafts are sampled with the target's position-keyed noise and kept when they equal the target's sample, so a seed gives the same text with or without speculation. rejection: greedy drafts, each kept with probability p(draft) under the request's temperature and top_p (the engine has no top_k or min_p), else replaced by a sample of p without it. The output distribution is the target's in both modes, but rejection gives other text for a seed. Which mode keeps more drafts depends on how close the drafter's distribution is to the target's. Temperature 0 is the same in both. Needs a drafter |
| `--spec-adaptive` | TP4 launcher default with DFlash2 (`SPEC_ADAPT`, default 1 with DFlash2, 0 with MTP) | each round verifies the draft count with the most expected tokens per ms (`engine/runtime/spec_len.h`), from the drafter's own probability of each draft, recent acceptance per position and the verify cost by row count at 2200 MHz. MTP (`--mtp K`: up to K drafts) also stops its chain when one more draft does not pay. DFlash2 chooses among 1..`--dflash-k`. Every level gets a verify graph. Under S2 it is the policy of the DFlash2 sessions |
| `--spec-levels 7,5,3,2` | all of 1..K (`SPEC_LEVELS`) | `--spec-adaptive`: the draft counts it may choose, for example 7,5,3,2 with the drafter or 1,2,4 with `--mtp 4`. The MTP chain drafts up to the next allowed count, so gaps are fine |
| `--serve DIR` | `$G6_HOME/s2` | serve requests through S2: the shim drives DIR (`cmd.<n>` / `req.<n>` / `resp.<n>`), `DIR/ready.s2` appears once loaded. Without it, g6m4 runs its `--prompt` sequences once and exits |
| `--s2-port N` / `--s2-sessions S` / `--s2-epoch E` | 7341 (`S2PORT`) / 8 (`SESS`) / the boot epoch | S2: the head's control port, the sessions that decode at once (1-8), and the epoch every rank must share |
| `--cap N` | 614400 | KV pool in tokens (the longest context) |
| `--mtp K` | 2 | MTP speculative draft length |
| `--graph` | on | decode runs as captured CUDA graphs, one per row count |
| `--wide` | on | the experts' fc1 records are reordered into the wide order at load (`rank.cuh`; outputs bit-identical) |
| `--prefill MP` | 4096 | prompt tokens go through the prefill path in sub-batches of up to MP rows |
| `--pf-lanes 2` | 2 | two prefill sub-batches in flight as two lanes |
| `--pfv2` | on | the v2 prefill path (tensor-core attention). `G6_PF_OWN>=2` and `G6_DEC_SHARED_SCRATCH` need it |
| `--chunk 8` | 8 | `Plan::chunk` of the step plan (`rank.cuh`); 8 is what we serve with |
| `--temp 0.7` / `--top-p-cands 256` | | engine default temperature (each request carries its own through the shim) / top-p candidates per rank per row (256 = served) |
| `--no-hash` | off (`NOHASH=1`) | skip the sha256 check of every pack tensor at load |

## Engine environment

| variable | K1H2 | meaning |
|---|---|---|
| `G6_PF_TILE` | 64 | prefill attention query tile (64/128/256) |
| `G6_PF_DENSE` | 8192 | context threshold for the dense prefill attention path; past it the sparse (indexer-selected) path runs; 0 = always sparse |
| `G6_PF_RS` | 1 | prefill uses reduce-scatter + all-gather instead of all-reduce |
| `G6_TEST_ZERO_DELAY_MS` | unset | a test of the zeroing order (#49, #54): every `dmalloc` zeroing waits behind an N ms kernel on the legacy stream, so a write to new memory not ordered after it loses every time. Boots take about N ms longer per allocation |
| `G6_WEIGHT_CHECK` | 1 | sum each loaded image on the CPU and the GPU and stop the boot with exit 3 if they differ; seal the sums once ready to serve and answer `DIR/check.<n>` (`/health?deep`, engine/runtime/weight_check.cuh). 0 = off |
| `G6_WEIGHT_CHECK_FLIP` | unset | `RANK:OFFSET`, a test of the weight check: flip one byte of rank RANK's first image after the load, so the boot stops with exit 3 |
| `G6_PF_STREAMS` | 2 | the two prefill lanes run on two CUDA streams |
| `G6_PF_ATTN_ROWS` | 1024 | prefill attention runs in chunks of this many rows inside a sub-batch |
| `G6_PF_OWN` | 3 | row-owner prefill. 1 = the owner computes the next layer's projections. 2 = also the indexer. 3 = the owner runs the whole attention block for all 64 heads of its rows (QO; needs the QO pack, `G6_PF_RS=1`, `--pfv2`). 4 = TP4 only: pair-owner QO, OWN=3's attention block split over two head pairs ([DESIGN-QO-PAIR.md](DESIGN-QO-PAIR.md)); TP4 launcher default |
| `G6_PF_QP_ROLL` | TP4: 2 | `G6_PF_OWN=4`: pair weights in this many rolling slots (2..4), built per layer from the rank pack and the pair partner's slice over RDMA; 0 = resident `--qp-pack` |
| `G6_PF_QP_CTX` | TP4: the cap | `G6_PF_OWN=4` with `G6_PF_QP_ROLL`: QO key-table size in positions (required); a request whose prompt + gen is longer runs OWN=2 |
| `G6_PF_QP_LAYERS` | all | `G6_PF_OWN=4`: model layers (0-77, list and ranges) that run pair-owner attention; the others run OWN=2 |
| `G6_PF_QP_MIN` | 768 | `G6_PF_OWN=4`: a prompt shorter than this many tokens runs OWN=2 (OWN=4's fixed per-request cost loses below ~770 tokens); 0 = every request that fits `G6_PF_QP_CTX` runs OWN=4 |
| `G6_PF_QP_EARLY` | TP4: 1 | `G6_PF_OWN=4`: 1 = send the counterpart's partial before the own rows' o_proj; 2 = also split the QO call in two |
| `G6_PF_QP_WARM` | 0 | `G6_PF_OWN=4`: warm the pair cuBLAS and MXFP8 GEMM shapes at load |
| `G6_PF_QP_WIRE_E` | 1 | `G6_PF_OWN=4` with `G6_PF_QP_ROLL`: entries (1..4) in the RDMA ring of half images |
| `G6_PF_QP_ROLL_SER` | 0 | measurement: 1 = build rolling slots on the prefill stream (fully exposed) |
| `G6_PF_BLAS_ROWS` | 1024 | prefill cuBLAS calls in chunks of this many rows (multiple of 16) |
| `G6_PF_TIMES` | 1 | `m4sim --serve`: per-request timing lines on stderr, from the request file to the first token. g6m4 does not read it |
| `G6_PF_HASH_GPU` | 1 | the per-sub-batch residual hash (rank agreement check) is computed on the GPU |
| `G6_PF_RSDIRECT` | 1 | o_proj, dense down and the MoE final sum write straight into the reduce-scatter send slot |
| `G6_PF_AOP` | 1 (when built with `-DPF_AOP`) | fc1 A-operand share: `k_pf_aop` quantizes the fc1 A operand once per (segment, matrix, distinct prologue input) into global records that the fc1 block-CTAs stage by cp.async. Needs build flag `-DPF_AOP`; 0 disables; `G6_PF_AOP_MMIN` (default 3072) sets the M floor. Byte-exact |
| `G6_PF_F12` | 1 (when built with `-DPF_F12`) | fc1 runs as the fat-CTA kernel `k_pf_fc1f` (NE=2 expert blocks share one quantized A record). Needs build flag `-DPF_F12`, which `scripts/build.sh` sets at TP=4; 0 = the stock fc1. Byte-exact |
| `G6_PF_GEMMF32` | 0 | the row-owner QO o_proj (`G6_PF_OWN=3`; its FP32 output kept it on the guarded kernel) runs as `k_gemm_mx8f` with an FP32 epilogue instead of the guarded `k_gemm_mx8b`. Needs build flag `-DPF_GEMMF` (in the stock build); read once at the first launch, so set it for the boot. Byte-exact |
| `G6_DEC_SPLIT` | 3 | the decode step is captured as two graphs split at layer 3, so the second submission overlaps the first one's GPU time |
| `G6_DEC_SHARED_SCRATCH` | 1 | one decode attention scratch for all layers (frees memory for the big pool; needs `--pfv2`) |
| `G6_CP` | 1 | context-parallel KV on (build flag `-DG6_CP=1` too) |
| `G6_CP_R` | `SHARE_LIMIT` | the first R tokens of KV are replicated on every rank. Pages past R are spread across ranks. Under S2 it is also the largest request that shares decode steps. A multiple of 64; R is always a decode bucket boundary (a value off the 2048 doubling is added as a bucket of its own) |
| `G6_CP_S3` | 1 | CP combine order (1 = head-group overlap order, same arithmetic as 0) |
| `G6_CP_ONECHAIN` | 0 | 1 = past R, the other ranks' head groups run as one attention call of 48 heads instead of three of 16 (the scores, softmax, context and merge kernels once per layer, not three times; their partials are published after it, in the order `G6_CP_S3=0` uses). Byte-exact. TP=4 only, and not with `G6_ATTN_FLIP`; about 1.3 MB a row of the CP fabric's rows (`G6_CP_MMAX`) for the gathered queries and the call's scratch |
| `G6_CP_PF_VERIFY` | 1 | check every KV page staged to another rank against a hash |
| `G6_CP_PF_HASH2` | 1 | the fast HASH2 page hash for that check (build flag `-DG6_CP_PF_HASH2=1`). This is the "H2" in K1H2 |
| `G6_PROGRESS_ISOLATE` | 1 | just before the first fabric barrier, move every thread that is not a fabric proxy thread off the proxy CPUs (the cores of the proxies this rank started, see `--proxy-cpu`) so no worker thread can preempt a progress thread. 0 = off (the A/B switch). Anything but 0/1 is rejected at startup. A thread that cannot be moved is named in the log |
| `G6_XTRACE` | 0 | 1 = transport trace (measurement only): GPU globaltimer stamps per collective (publish, doorbell, each peer's tail, reduce end), proxy tick stamps per rail, per-QP WC retry/RNR counters. Dumped outside the timed loop, at proxy stop or when `G6_XTRACE_QUIT` appears. `tools/xtrace_driver.sh` wraps a run and samples the NIC counters per rail |
| `G6_XTRACE_OUT` | /tmp | directory for the dumps, `g6xtrace.r<rank>.f<n>.txt` |
| `G6_XTRACE_QUIT` | unset | path of a file; when it appears, each fabric dumps once |
| `G6_ATTN_HEADS` | 1 | decode attention scores and context with one key load for a group of heads; 0 = the per-head kernels |
| `G6_ATTN_SC2` | 0 | decode attention scores of the head-group kernel with S = 1, 2, 4 or 8 key slots per warp, the query loaded once for all S; 0 = off. FP8 KV only, same bits |
| `G6_ATTN_SC2_TR` | 0 | with `G6_ATTN_SC2`: 1 = the heads' sums reduced by one transpose across the warp, 2 = the same reduction eager; S = 8 ignores it. Same bits |
| `G6_ATTN_CTX2` | 0 | decode attention context: 1 / 2 = the head-group kernel with 32 / 64 gathered pairs per batch (16 when off), 3 / 4 = a packed, branch-free kernel with 16 / 32, two batches in flight. FP8 KV only, same bits |
| `G6_ATTN_CTX3` | 0 | decode attention context with one test per batch of slots instead of per slot: 1 / 2 = 16 / 32 pairs per batch over 256 dims per CTA, 3 / 4 = the same over 512. Overrides `G6_ATTN_CTX2`. FP8 KV only, same bits |
| `G6_ATTN_CTX3_HG` | 0 | with `G6_ATTN_CTX3`: heads per CTA, 2, 4 or 8; 0 = 4 up to 4 rows, then 8 |
| `G6_ATTN_SEL2` | 0 | decode index selection: the final sort as a rank by count over 8, 16, 32 or 64 CTAs per row. Same bits |
| `G6_IDX_GRP` | 0 | S2 batched decode steps: the indexer reads each key tile once for up to G = 1..64 rows of one session. Same bits |
| `G6_ATTN_QT` | 0 | 1 = decode absorb stores q and rope lane-transposed and the scores kernel reads them with 16 B loads. Needs `G6_ATTN_SC2` 2 or 4; not with `G6_AVP1`. FP8 KV only, same bits |
| `G6_ATTN_FRONT` | 0 | 1 = a decode step's first attention kernels (KV append, then index append and index query prep, or index reuse) as one launch, when the step's rows fit one query tile. Same bits |
| `G6_DTR` | 1 | the dense trellis matrices of the exlr8 release form (q_b, o_proj, the dense MLP of layers 0-2; `engine/kernels/dtr.cuh`) where a layer has them: decode on the trellis tiles, prefill on the o_proj tiles and on exact FP8 copies of q_b and the dense MLP (made at load). 0 = the int4 / MXFP8 records, which the layer must then have (A/B) |
| `G6_DTR_PF_COPY` | 0 | 1 = o_proj prefill on an FP8 copy of its tiles too, as q_b and the MLP (about 25 MB more per layer at TP4; A/B, same bits) |
| `G6_MEM_MIN` | 8 | GiB of MemAvailable the boot must keep after the weights and each layer of the KV pool; under it the engine stops with a message instead of meeting the OOM killer |
| `G6_PF_TR_BM` | 16 | row blocks per CTA of the prefill GEMM on trellis tiles (`k_gemm_tr`) for large launches: 16 (256 rows, 512 threads) or 8. Small launches take 4 |
| `G6_QA_SPLIT` | 2 when TP>1 (else 0) | decode: the GEMVs on the layer input, whose weights every rank holds in full, split by 16-output tiles across the ranks, then all-gathered per layer on a fabric of their own (port + 64) before the q_a norm. 1 = q_a, kv_a and the indexer's k and weights; 2 = q_a only, with kv_a and the indexer's GEMVs run in full while the gather is on the wire. Same bits as 0 (an output tile's arithmetic does not depend on which tiles a launch covers). Every rank must set the same value. Main model only (the MTP layer keeps the full GEMVs). `G6_QA_SPLIT_TPC` (1) = tiles per CTA of the split GEMVs, `G6_QA_SPLIT_CPU` (`--proxy-cpu` - 9) = the gather proxy's core |
| `G6_IDXQ_SPLIT` | 1 when TP>1 (else 0) | decode: the indexer's wq_b (idx_q, 4096 outputs, replicated) split by output tiles the same way on the same fabric, on the indexer layers: each rank's tiles are published, q_b runs while the gather is on the wire, then the gather. Same bits as 0. Independent of `G6_QA_SPLIT` |
| `G6_MTP_SPLIT` | 1 when TP>1 + --mtp (else 0) | MTP draft steps (needs `--mtp`): eh_proj (6144 outputs, 78 MB replicated), then layer 78's q_a / kv_a / indexer k and weights (as `G6_QA_SPLIT=1`) and idx_q (as `G6_IDXQ_SPLIT=1`), split by output tiles and gathered on a draft gather fabric (port + 72). Same bits as 0 |
| `G6_DRAFT_HEAD_I4` | 0 | MTP draft steps (needs `--mtp`): the draft's lm_head GEMV reads INT4 g32 records of the rank's vocab slice, made at boot on the GPU from its MXFP8 records (134 MB more per rank at TP4), instead of the MXFP8 records: 45% fewer head bytes per draft step. The target's head and the verify steps are unchanged, so the committed tokens are the target's; drafts (and their log-probabilities, which `--spec-adaptive` reads) can differ from the MXFP8 head's. Same on every rank. `G6_DRAFT_HEAD_KS` (32, 2..32) / `G6_DRAFT_HEAD_TPC` (1) = the GEMV's k parts per tile and tiles per CTA; every rank must set the same values |
| `G6_DFL_HEAD_I4` | 0 | DFlash2 drafter (needs `--dflash-manifest`): its lm_head GEMV reads the same INT4 g32 records as `G6_DRAFT_HEAD_I4` (the MTP draft's copy when that is on, else one made the same way; same `G6_DRAFT_HEAD_KS` / `_TPC`). In selector mode the head's raw logits choose each row's top-16 candidates and enter their walk scores, so INT4 can change both. The target's head and the verify steps are unchanged. Every rank must set the same value |
| `G6_DFL_BODY_I4` | 0 | DFlash2 drafter (needs `--dflash-manifest`): its body GEMVs (fc, and per layer kv, aconv, q, o, mconv, gu, down) read INT4 g32 copies of their MXFP8 records, made at boot on the GPU by the `G6_DRAFT_HEAD_I4` arithmetic, with the same splits: 382 MB instead of 701 MB per draft step's reads (the pack image keeps the MXFP8 records: +382 MB per rank). The selector's hproj and the head (`G6_DFL_HEAD_I4`) are not touched. Drafts only: the target and the verify steps are unchanged; drafts and their log-probabilities can differ (`engine/runtime/dflreplay.cu` compares both drafters on a `G6_DFL_DUMP`). The drafter choice's drafting cost (`G6_SPEC_DRAFT`) is not changed with it. Every rank must set the same value |
| `G6_ROUTE_OVL` | 0 | decode, trellis-shared MoE layers with separate fc1 / fc2 launches (2+ rows): the shared expert runs on a forked stream from before the router GEMV, so its weight reads overlap the router GEMV (8 of 48 SMs) and the single-CTA plan kernel. 1 = its fc1 and fc2 on the fork, joined before the finalize; 2 = its fc1 only, joined before fc2. Same bytes out (`k1_ovl_test`). Fused layers (1 row) and the MTP layer run as before. Read once |
| `G6_DFL_SEL_ALPHA` | 1 | DFlash2 selector walk (needs `--dflash-manifest` and the selector): each candidate scores `logit + a * correction` instead of `logit + correction`, and the draft's log-probability comes from those scores. Drafts only; the verify is unchanged. Every rank must set the same value |
| `G6_L2PF` | 0 | decode, main model: a paced L2 prefetch (`engine/kernels/l2pf.cuh`) of the weights the step reads next, in the windows where DRAM is idle. One warp on a high-priority side stream issues 8 KiB `cp.async.bulk.prefetch.L2` chunks from a token bucket; a collective window stops once its collective has arrived. Writes nothing: same bytes out. Window mask: 1 = the MLP all-reduce wait (the next layer's q_a / kv_a / idx tiles, q_b, UK, ...), 2 = the `G6_QA_SPLIT` / `G6_IDXQ_SPLIT` gather waits, 4 = the attention (UK, UV, o_proj, ...), 8 = the attention all-reduce wait (router, shared fc1); each window continues the layer's queue where the last ended. `G6_L2PF_T` (0xff) = targets (1 x_in, 2 idx_q, 4 q_b, 8 UK, 16 UV, 32 o_proj, 64 router, 128 shared / dense fc1), `G6_L2PF_GBPS` (175) = pace, `G6_L2PF_WAVE` (8) = chunks per poll at most, `G6_L2PF_US` (30,30,100,30) = budget per window in us at the pace, `G6_L2PF_MB` (10) = MiB per window at most, `G6_L2PF_OPMB` (8) = MB of o_proj's entries per layer (0 = all). Not on the CP attention path's attention window. `engine/kernels/tests/l2pf_bench.cu` measures it |
| `G6_GV_MULTI` | 1 | steps of up to 8 rows run the GEMVs on the layer input (q_a, kv_a, the indexer's k and weights) as one launch; 0 = one launch each (same bits) |
| `G6_S2_ADAPT` | 0 | S2: choose each round's MTP draft count (1..k) from the sessions' accept rates and the S2 step costs; 1 adds ~1 GiB of main-step graphs |
| `G6_S2_GROW` | 1 | S2: a session below R holds the KV pages it needs now and takes more as its output grows, admitted and grown only while every live session can still reach prompt + `max_tokens` + guard; a session that cannot grow sits out rounds. 0 = every run holds that worst case from admission |
| `G6_S2_DSEL` | 1 | S2 with DFlash2: a request that decodes alone drafts each round with DFlash2 or MTP, whichever is expected to commit more tokens per ms (prose +4.5% on RigMark, code level); 0 = DFlash2 every round. `G6_S2_DSEL_MARGIN` (0.15), `G6_S2_DSEL_DWELL` (8 rounds) and `G6_S2_DSEL_PROBE` (4) tune it; `G6_S2_DSEL_DCOST` (8.1), `G6_S2_DSEL_MCOST` (5.5) and `G6_S2_DSEL_MEACH` (2.1) override its drafting ms per round (DFlash2; MTP's first form and the drafter's context step; each MTP chain step) |
| `G6_S2_DECODE_PF` | 0 | S2: while any session decodes, a prefill quantum takes at most this many rows on one lane, so a decode round comes round sooner (each round waits out one quantum). 0 = the full `--prefill` x lanes quantum. A multiple of 64, at most `--prefill`. Part of the bootstrap agreement when set |
| `G6_S2_DECODE_ROUNDS` | 0 | S2: while a prefill waits and sessions decode, K decode rounds run per prefill quantum instead of one (0 or 1 = off; at most 64). Part of the bootstrap agreement when set |
| `G6_S2_ADMIT_HOLD` | 0 | S2, head only: 1 = an admission with another request queued behind it and a slot left holds its boundary, and the next admission is announced at that same boundary, so a burst of queued requests is granted at once instead of one per notice (the workers follow the commit frame's hold field) |
| `G6_SPEC_TABLE` | built-in measured table | Override verify costs for 1–8 rows with eight comma-separated finite positive times in ms. Applies to adaptive DFlash2 length and drafter choice; survives session resets. |
| `G6_SPEC_DRAFT` | built-in drafting costs | Override DFlash2 round, MTP round and MTP additional chain-step costs with three comma-separated finite nonnegative times in ms. |
| `G6_DFL_REUSE` | 0 | TP4 with DFlash2 under S2: 1 = a request whose prefix-reuse hit skipped into the drafter's 2,048-position window keeps the drafter instead of running MTP for its whole life. The drafter's context starts past the hit's rows, or, when the drafter's KV ring still holds the previous turn of the same conversation (the last DFlash2 session there had this prompt's tokens up to the hit, and its rows were not written over), at the window's start; with fewer than `G6_DFL_REUSE_MIN` (256, 0..2048) context rows the request runs MTP rounds that feed the drafter until it holds that many, then drafts with DFlash2. A hit with no prefill at all (a replay) is allowed too. 2 = also a request that runs alone on MTP (it was not alone at its prompt tail, or another request has since left) takes the drafter the same way, from its last token. Drafts only: the target decides every token. In the S2 runtime signature; every rank must set the same values. With it on, rank 0 prints one `# s2 dflash reuse slot` line per request (rounds on each drafter, where the context started, the warm-up, the promotion). KNOWN-ISSUES #1 |
| `G6_MTP_KSTOP` | 2 when --serve + DFlash2 + --mtp 2+ (else 0) | S2 drafter choice, MTP rounds of a request that decodes alone: 1 = verify draft j while the MTP head's p_1 x .. x p_j >= `G6_MTP_KSTOP_TAU` t_j (default `0,0.2,0.5`; at least one draft) and stop the chain before a draft that could not pass; 2 = choose the count from the drafter choice's MTP estimates (`G6_MTP_KSTOP_SEED3`, 0.55: their prior for a third draft). `G6_MTP_KSTOP_KMAX` (1..3, default `--mtp`) is the most drafts. Drafts only: the target decides every token, but the verify's row count (and so its k1 split class) varies. Every rank must set the same values |
| `G6_MTP_LPLOG` | 0 | S2: one `# kstop` line per round of a request that decodes alone (rank 0; 2 = every rank): drafter, verified k, drafted count, accepted count, the drafts' log-probabilities, MTP's first draft's log-probability and whether the target took it, and a hash of the decision's inputs. Changes nothing else |
| `G6_DRAFT_PAR` | 2 when --serve + DFlash2 + --mtp (else 0) | S2 with DFlash2, a round of a request that decodes alone: 1 = the drafter choice is made before MTP's first form (it reads nothing the first form writes), the DFlash2 drafter step runs on its own CUDA stream under the first form (and the MTP chain on MTP rounds), and the round waits for it at the end. The same graphs on the same inputs as serial, so every step's bytes are unchanged (`batch_draftpar_test`: the same tokens, steps and statistics as serial). 2 = also a drafting step goes on a high-priority stream. Shared rounds run as before. In the S2 runtime signature; every rank must set the same value. Needs `--dflash-manifest`, `--mtp` and `--serve` |
| `G6_S2_TIMELINE` | unset | measurement, S2: N = every N main steps each rank prints, per decode step kind (main, first-form draft, chain, DFlash2 context and query steps, and their `G6_DRAFT_PAR` forms), the mean GPU span (CUDA events around the graph launch), host input prep, launch call and fence wait, and per transition the main-stream gap between steps (the drafter's side stream may run inside it under `G6_DRAFT_PAR`) and the host time between them (`frontend/README.md`). Unset: no events, no stamps |
| `G6_S2_DSEL_MODEL` | 0 | S2 drafter choice: 1 = price both drafters from one frozen cost table, `G6_S2_DSEL_TABLE` (a `key:value` list or the path of a file holding one: verify ms by rows `v1`..`v8`, drafting ms `d` (DFlash2 round), `m` (MTP round), `e` (MTP chain step), `p` (a probe's extra), and optionally the MTP rounds' prior under `G6_MTP_KSTOP`: `g` drafts accepted after the first, `k` verified, `n` drafted, `w` the estimates' prior weight, and `r`: price DFlash2 from the request's DFlash2 rounds and scored probes as they ran, that evidence aged by `r` per MTP round, in place of its position rates), in place of `G6_S2_DSEL_DCOST`/`_MCOST`/`_MEACH` and the built-in verify table, and price MTP rounds as the confidence stop runs them (their mean table cost and accepted drafts in the request) instead of as `--mtp` drafts; 2 = the DFlash2 rounds' own draft count (`--spec-adaptive`) uses the table too. `tools/s2_dsel_table.py` writes the table from a `G6_S2_TIMELINE` + `G6_MTP_LPLOG` boot. The table is in the S2 runtime signature: every rank must read the same one |
| `G6_S2_DSEL_START` | dflash | S2 drafter choice: `mtp` = a DFlash2 request starts on MTP (the drafter's prompt-tail drafts are its first probe) and moves to DFlash2 when the evidence says so. Every rank must set the same value |
| `G6_S2_DSEL_POSCLK` | 0 | S2 drafter choice: 1 = its DFlash2 and MTP estimates decay a draft position only on rounds that observe it, so the DFlash2 rounds (which score MTP's first draft alone) no longer age MTP's later positions toward the seed. Every rank must set the same value |
| `G6_MTP_KSTOPB` | 0 | S2, shared rounds (two or more sessions): 1 = each session verifies its own count of the `--mtp` drafts, the most j with the MTP head's p_1 x .. x p_i >= t_i for every i <= j, at least `G6_MTP_KSTOPB_KMIN` (0, or 1). `G6_MTP_KSTOPB_TAU` (`0.6,0.88`) at every session count, `G6_MTP_KSTOPB_TAU_S<n>` at n sessions. The boot captures a main graph per width the rule reaches (about 115 MiB each over 8 buckets; `G6_MTP_KSTOPB_WIDTHS=M,..` limits them, a round then verifies the next most likely drafts up to a captured width). `G6_MTP_KSTOPB_EXPLORE=n` (measurement): every n-th round's shared sessions verify a hashed count, so the `G6_MTP_LPLOG` lines time every width. Drafts only: the target decides every token. Every rank must set the same values |
| `G6_DEC_EVPROF` | unset | measurement: N = per-stage decode step times from event nodes in the captured main-step graphs, printed every N main steps (rank logs) |
| `G6_DEC_PROF` | unset | measurement, plain launches only: the same stages with a synchronize at every mark |
| `G6_DFL_PFCTX` | 1 | `--dflash`: 0 = no prompt context from the prefill (measurement); the drafter then starts from the last prompt token |
| `G6_DFL_DUMP` | unset | `--dflash` measurement, rank 0: a file that gets every draft step's context features, drafts and the committed tokens (`engine/ref/dflash2_replay.py` replays it through the reference drafter) |

The decode attention variables from `G6_ATTN_SC2` to `G6_ATTN_FRONT` are read once, before graph capture, so set them
for the boot. Other values than the ones listed mean off. `G6_ATTN_P11=1`, and `G6_ATTN_FLIP` where it applies, take
precedence over `G6_ATTN_SC2`, `G6_ATTN_CTX2` and `G6_ATTN_CTX3` (and so over `G6_ATTN_QT`), and `G6_ATTN_HEADS=0` turns
those four off. The measured set is `G6_ATTN_SC2=4 G6_ATTN_SC2_TR=2 G6_ATTN_CTX2=4 G6_ATTN_CTX3=2 G6_ATTN_SEL2=16
G6_IDX_GRP=16 G6_ATTN_QT=1 G6_ATTN_FRONT=1` (CHANGELOG, 2026-10-16). "Same bits" is by construction for every value;
the attnk bench (`engine/attn/tests/attnk.cu`) checked the measured set, with and without `QT=1 FRONT=1`.

`G6ENV_X="NAME=value ..."` adds more variables on every rank. The engine rejects bad `G6_PF_*` values at startup
(`engine/runtime/envparse.h`).

## Launch knobs

| variable | default | meaning |
|---|---|---|
| `RANK_HEALTH` | 1 | probe every rank before any engine starts (`scripts/rank_health.sh`, `tools/health/`) and refuse to boot, exit 3, with one named reason per refused rank. 0 = skip. The result is written to `$G6_HOME/out/rank_health.json` |
| `PROBE_ARGS` | empty | thresholds for the probe, e.g. `--min-clock-mhz 1800 --min-link-gbps 90` (list in `tools/health/README.md`) |

The shim and the front end run under `taskset` on CPUs `0..PROXY_CPU-8`, away from the proxy CPUs. At TP=6 the shim is
`g6s2shim.py` with `--stall-timeout $STALL_TMO` (default 2400 s).

## Build flags (`scripts/build.sh`)

Each `-D` selects a kernel or host-path variant that won an A/B on our cluster. Without it the build uses the earlier
path. The `#if` blocks and their comments in the source are the reference:

| flag | what it selects (from the source comment at its `#if`) |
|---|---|
| `PF_FC2GL=6` | prefill expert fc2: one CTA loops over 6 consecutive 128-column groups (`kernels/bind_prefill.cuh`) |
| `PF_GEMMF` | MXFP8 prefill GEMM with an unguarded mma loop for full tiles |
| `PF_GEMMR`, `PF_GEMMRF` | GEMM launch order with the row block fastest, so CTAs sharing a weight tile run together (`GEMMRF`: owner-row launches) |
| `PF_XQ8C` | activation-quantize kernel variant: one CTA = one 16-row block x 16 k-groups |
| `PF_RNWIDE` | reduce-scatter add + RMSNorm with one 768-thread CTA per row instead of an 8-CTA cluster (`runtime/prefill.cuh`) |
| `PF_DRAFTKV` | in prefill, the MTP layer only fills its caches; its output is dropped |
| `PF_QOBH`, `PF_QOROPE`, `PF_ROPE2` | QO attention variants (`attn/prefill-qo-bind.cuh`, `attn/attn.cu`): RoPE from a precomputed table, bit-equal to the computed one |
| `PF_ISGRT` | the prefill index scorer re-tiled: a CTA keeps its 64-key tile and walks several rows (`attn/prefill.cuh`) |
| `DEC_SLOT=2` | decode partial producers write straight into the send slot and ring the doorbell themselves (`kernels/decslot.cuh`) |
| `DEC_CPB=1`, `DEC_CPB2=64` | decode attention prefetches the fp8 pairs of the next 64 slots before their FMAs (same output bytes) |
| `DEC_AVP=1` | decode absorb / value-project for small row counts in one launch |
| `DEC_I4PAIR=1` | decode int4 records (q_a, o_proj) repacked once at load into a paired layout |
| `SEL32` | the exact top-k selector is used past 32,768 tokens of context too |
| `DEC_IDX31=1` | decode index scorer reads each key once for all query rows and skips tiles another CP rank owns; select stops at the live end of the bucket (exact; needs `SEL32`) |
| `G6_CP=1`, `G6_CP_PF_HASH2=1` | context-parallel KV, and its HASH2 staged-page check |
| `PF_F12` (TP4 build only) | prefill expert fc1: one fat CTA runs `F12_NE` (default 2) adjacent expert blocks on one quantized A record (`kernels/bind_prefill.cuh`, `k_pf_fc1f`). `F12_NE` must divide the expert blocks per rank (4 at TP4, 3 at TP6). `G6_PF_F12=0` turns it off at run time |
| `G6_KV4=1` (a build flag; experimental) | the MLA latent cached as INT4 after a Hadamard rotation, 416 B a row instead of 656 B (`attn/kvfmt.h`, [KV4.md](KV4.md)). Changes the outputs. The image ships it as `bin/g6m4-int4`, which `--kv-format int4` (`KV_FORMAT`) runs; `KV_FORMAT=int4 scripts/build.sh` builds it |

## Front end and shim

The head runs a shim that hands requests to the engine: `frontend/g6s2shim.py --dir <the head's DIR> --sock <sock>`
(the head engine passes them to the other ranks over its control links). The API is the Rust front end,
`bin/g6front-rs --model-dir <tokenizer dir> --engine-sock <sock> --port 8100`. Its flags are in `frontend/README.md`.

## Local profile boot flags

`--steering-profile DIR` loads one verified local profile. Supply `--steering-model-revision`,
`--steering-tokenizer-revision`, `--steering-chat-template-sha256`, and the actual served file through
`--steering-chat-template FILE`. The supervisor derives/checks `--steering-weight-set`; a direct engine invocation
needs it and `--set-id` explicitly. Profile loading refuses disabled weight hash checks.
`--steering-force SCALE` enables explicit forced qualification mode. These are g6m4 flags; m4sim does not load profiles.
See [local steering](STEERING-LOCAL.md) for the complete contract, identity limitations and unit-test boundaries.
The launcher also accepts `--steering-profile hf:default` / `hf:NAME` with an HF model release and derives the
profile's identity assertions before validating the actual model. See [HF steering](STEERING-HF.md) for publication,
explicit selection and cached offline restart. The engine itself receives only the verified local directory.
