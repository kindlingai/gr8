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
| `--ips a,..` / `--ips2 a,..` | from `cluster.env` | RoCE v2 address of every rank on rail 1 / rail 2, in rank order (exactly `G6_NR` of them) |
| `--proxy-cpu N` | 19 | CPU core for the decode fabric's proxy thread. Every other fabric's proxy has its own core below it: N-1 and N-2 the two prefill lanes, N-3 the CP fabric, N-4 CP staging, N-5 the QP roll, N-6 the MTP draft, N-7 DFlash |
| `--manifest` / `--pack` | `packs/rank<r>.*` | this rank's pack; `--pack -` reads it from stdin |
| `--v2 DIR` / `--v2-slices DIR` / `--v2-widths FILE` | the release (`V2`, `V2_SLICES`, `--variant`) | the routed and shared experts from the exlr8 width files at the index slices' offsets, at the manifest's widths, in place of the pack's |
| `--v2-dense DIR` | the release (`V2_DENSE`) | with `--v2`: no `--manifest` / `--pack`. Every other weight comes from the release's `dense/` files, planned at load (`engine/loader-rs/src/release.rs`); rank 0 prints which form each matrix took. Under `G6_PF_OWN=3` the image also holds the full-head q_b / o_proj / UK / UV, so no QO pack |
| `--v2-dense-form trellis\|rounded` | trellis (`V2_DENSE_FORM`) | trellis: q_b, o_proj and the dense MLP stored as trellis stay the release's entries for the dense trellis kernels (`engine/kernels/dtr.cuh`). rounded, the fallback: they are rounded to MXFP8 and int4 at load (not the release's weights) |
| `--image` | on | load the whole pack into one device image (required for full packs) |
| `--layers all` | all | all 78 layers + the MTP layer |
| `--qo-manifest` / `--qo-pack` | `packs/qo.*` | full-head attention weights for `G6_PF_OWN=3` (identical file on every node) |
| `--qp-manifest` / `--qp-pack` | (TP4, not used) | resident pair weights for `G6_PF_OWN=4` (`tools/convert/pairpack.py`, one file per rank); the TP4 launcher uses `G6_PF_QP_ROLL` instead |
| `--dflash-manifest` / `--dflash-pack` | TP4 launcher default (`DFLASH=1`; `DFLASH=0` for MTP) | replaces MTP with the DFlash2 block drafter (`engine/runtime/dflash.cuh`), packs from `tools/convert/dflash2.py` (one per rank, about 0.86 GB). It drafts 7 tokens per round in one forward. One request at a time it replaces `--mtp`; under S2, `--mtp` stays on: a request alone at its prompt tail drafts with DFlash2, the others with MTP. TP4 builds (the rank count must divide the drafter's 8 KV heads) |
| `--dflash-sel 0\|1` | 1 (`DFLASH_SEL`) | 1 = the DFlash2 selector walk over the top-16 candidates of each position, 0 = the drafter's argmax (sampled with the target's position keys) |
| `--dflash-k K` | 7 (`DFLASH_K`) | the drafts a round verifies (1..7); a verify step carries K + 1 rows. The drafter always drafts the whole block |
| `--spec-accept coupled\|rejection` | coupled (`SPEC_ACCEPT`) | how verify steps accept drafts at temperature > 0, for MTP and the DFlash2 drafter. coupled: drafts are sampled with the target's position-keyed noise and kept when they equal the target's sample, so a seed gives the same text with or without speculation. rejection: greedy drafts, each kept with probability p(draft) under the request's temperature and top_p (the engine has no top_k or min_p), else replaced by a sample of p without it. The output distribution is the target's in both modes, but rejection gives other text for a seed. Which mode keeps more drafts depends on how close the drafter's distribution is to the target's. Temperature 0 is the same in both. Needs a drafter |
| `--spec-adaptive` | TP4 launcher default with DFlash2 (`SPEC_ADAPT`, default 1 with DFlash2, 0 with MTP) | each round verifies the draft count with the most expected tokens per ms (`engine/runtime/spec_len.h`), from the drafter's own probability of each draft, recent acceptance per position and the verify cost by row count at 2200 MHz. MTP (`--mtp K`: up to K drafts) also stops its chain when one more draft does not pay. DFlash2 chooses among 1..`--dflash-k`. Every level gets a verify graph. Under S2 it is the policy of the DFlash2 sessions |
| `--spec-levels 7,5,3,2` | all of 1..K (`SPEC_LEVELS`) | `--spec-adaptive`: the draft counts it may choose, for example 7,5,3,2 with the drafter or 1,2,4 with `--mtp 4`. The MTP chain drafts up to the next allowed count, so gaps are fine |
| `--serve DIR` | `$G6_HOME/s2` | serve requests. A DEC_S2 build (every TP) always serves through S2: the shim drives DIR (`cmd.<n>` / `req.<n>` / `resp.<n>`), `DIR/ready.s2` appears once loaded. A build without DEC_S2 serves one request at a time: `DIR/req.<n>` in, rank 0 writes `DIR/resp.<n>`, `DIR/quit` ends it, `DIR/ready` appears once loaded |
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
| `G6_PF_TIMES` | 1 | per-request prefill timing lines in rank 0's log |
| `G6_PF_HASH_GPU` | 1 | the per-sub-batch residual hash (rank agreement check) is computed on the GPU |
| `G6_PF_RSDIRECT` | 1 | o_proj, dense down and the MoE final sum write straight into the reduce-scatter send slot |
| `G6_PF_AOP` | 1 (when built with `-DPF_AOP`) | fc1 A-operand share: `k_pf_aop` quantizes the fc1 A operand once per (segment, matrix, distinct prologue input) into global records that the fc1 block-CTAs stage by cp.async. Needs build flag `-DPF_AOP`; 0 disables; `G6_PF_AOP_MMIN` (default 3072) sets the M floor. Byte-exact |
| `G6_PF_F12` | 1 (when built with `-DPF_F12`) | fc1 runs as the fat-CTA kernel `k_pf_fc1f` (NE=2 expert blocks share one quantized A record). Needs build flag `-DPF_F12`, which `G6_NR=4 scripts/build.sh` sets; 0 = the stock fc1. Byte-exact |
| `G6_PF_GEMMF32` | 0 | the row-owner QO o_proj (`G6_PF_OWN=3`; its FP32 output kept it on the guarded kernel) runs as `k_gemm_mx8f` with an FP32 epilogue instead of the guarded `k_gemm_mx8b`. Needs build flag `-DPF_GEMMF` (in the stock build); read once at the first launch, so set it for the boot. Byte-exact |
| `G6_DEC_SPLIT` | 3 | the decode step is captured as two graphs split at layer 3, so the second submission overlaps the first one's GPU time |
| `G6_DEC_SHARED_SCRATCH` | 1 | one decode attention scratch for all layers (frees memory for the big pool; needs `--pfv2`) |
| `G6_CP` | 1 | context-parallel KV on (build flag `-DG6_CP=1` too) |
| `G6_CP_R` | 32768 | the first R tokens of KV are replicated on every rank. Pages past R are spread across ranks (multiple of 64) |
| `G6_CP_S3` | 1 | CP combine order (1 = head-group overlap order, same arithmetic as 0) |
| `G6_CP_PF_VERIFY` | 1 | check every KV page staged to another rank against a hash |
| `G6_CP_PF_HASH2` | 1 | the fast HASH2 page hash for that check (build flag `-DG6_CP_PF_HASH2=1`). This is the "H2" in K1H2 |
| `G6_PROGRESS_ISOLATE` | 1 | just before the first fabric barrier, move every thread that is not a fabric proxy thread off the proxy CPUs (the cores of the proxies this rank started, see `--proxy-cpu`) so no worker thread can preempt a progress thread. 0 = off (the A/B switch). Anything but 0/1 is rejected at startup. A thread that cannot be moved is named in the log |
| `G6_XTRACE` | 0 | 1 = transport trace (measurement only): GPU globaltimer stamps per collective (publish, doorbell, each peer's tail, reduce end), proxy tick stamps per rail, per-QP WC retry/RNR counters. Dumped outside the timed loop, at proxy stop or when `G6_XTRACE_QUIT` appears. `tools/xtrace_driver.sh` wraps a run and samples the NIC counters per rail |
| `G6_XTRACE_OUT` | /tmp | directory for the dumps, `g6xtrace.r<rank>.f<n>.txt` |
| `G6_XTRACE_QUIT` | unset | path of a file; when it appears, each fabric dumps once |
| `G6_ATTN_HEADS` | 1 | decode attention scores and context with one key load for a group of heads; 0 = the per-head kernels |
| `G6_DTR` | 1 | the dense trellis matrices of the exlr8 release form (q_b, o_proj, the dense MLP of layers 0-2; `engine/kernels/dtr.cuh`) where a layer has them: decode on the trellis tiles, prefill on the o_proj tiles and on exact FP8 copies of q_b and the dense MLP (made at load). 0 = the int4 / MXFP8 records, which the layer must then have (A/B) |
| `G6_DTR_PF_COPY` | 0 | 1 = o_proj prefill on an FP8 copy of its tiles too, as q_b and the MLP (about 25 MB more per layer at TP4; A/B, same bits) |
| `G6_MEM_MIN` | 8 | GiB of MemAvailable the boot must keep after the weights and each layer of the KV pool; under it the engine stops with a message instead of meeting the OOM killer |
| `G6_PF_TR_BM` | 16 | row blocks per CTA of the prefill GEMM on trellis tiles (`k_gemm_tr`) for large launches: 16 (256 rows, 512 threads) or 8. Small launches take 4 |
| `G6_GV_MULTI` | 1 | steps of up to 8 rows run the GEMVs on the layer input (q_a, kv_a, the indexer's k and weights) as one launch; 0 = one launch each (same bits) |
| `G6_S2_ADAPT` | 0 | S2: choose each round's MTP draft count (1..k) from the sessions' accept rates and the S2 step costs; 1 adds ~1 GiB of main-step graphs |
| `G6_S2_DSEL` | 1 | S2 with DFlash2: a request that decodes alone drafts each round with DFlash2 or MTP, whichever is expected to commit more tokens per ms (prose +4.5% on RigMark, code level); 0 = DFlash2 every round. `G6_S2_DSEL_MARGIN` (0.15), `G6_S2_DSEL_DWELL` (8 rounds) and `G6_S2_DSEL_PROBE` (4) tune it |
| `G6_DEC_EVPROF` | unset | measurement: N = per-stage decode step times from event nodes in the captured main-step graphs, printed every N main steps (rank logs) |
| `G6_DEC_PROF` | unset | measurement, plain launches only: the same stages with a synchronize at every mark |
| `G6_DFL_PFCTX` | 1 | `--dflash`: 0 = no prompt context from the prefill (measurement); the drafter then starts from the last prompt token |
| `G6_DFL_DUMP` | unset | `--dflash` measurement, rank 0: a file that gets every draft step's context features, drafts and the committed tokens (`engine/ref/dflash2_replay.py` replays it through the reference drafter) |

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

## Front end and shim

The head runs a shim that hands requests to the engine: `frontend/g6s2shim.py --dir <the head's DIR> --sock <sock>`
(the head engine passes them to the other ranks over its control links). `frontend/g6shim.py --rank-dirs` drives an
older build without the multi-session path, writing to the other ranks over ssh. The API is the Rust front end,
`bin/g6front-rs --model-dir <tokenizer dir> --engine-sock <sock> --port 8100`. Its flags are in `frontend/README.md`.
