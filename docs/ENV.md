# Engine environment

Every setting the engine, its tests and its build read from the environment, from the YAML files in
[knobs/](../knobs/README.md).
`knobs gen` writes this page: change the table, not this page. [SETTINGS.md](SETTINGS.md) explains how the main ones
work together, and `./gr8 knobs` lists the launcher options that set them.

The knob `NAME` is the variable `GR8_NAME`. Its old name `G6_NAME` still works and prints a warning at boot. A name
ending in `<n>` stands for the name followed by any number. An empty value counts as unset. A value outside the
listed ones stops the engine at boot, and so does a `GR8_` or `G6_` name the engine does not know about.

Types: flag is 0 or 1, int is decimal or 0x hex, ints and floats are comma lists. Values: `lo..hi` is a range
(either end may be left out), `a|b` the only accepted values.

## Every rank the same

Every rank of one engine must run the same value. The set ones that differ from the default join the S2 runtime signature, so a rank whose value differs refuses to start.

| variable | type | unset | values | what it does |
|---|---|---|---|---|
| `GR8_ATTN_CTX2` | int | `0` | 0..4 | Decode attention context kernel: 1 or 2 = the head-group kernel with 32 or 64 gathered pairs per batch, 3 or 4 = a packed kernel with 16 or 32 and two batches in flight. Same bits, FP8 KV only. |
| `GR8_ATTN_CTX3` | int | `0` | 0..4 | Decode attention context with one test per batch of slots: 1 or 2 = 16 or 32 pairs per batch over 256 dims per CTA, 3 or 4 = the same over 512. Overrides GR8_ATTN_CTX2. Same bits, FP8 KV only. |
| `GR8_ATTN_CTX3_HG` | int | `0` | 0\|2\|4\|8 | With GR8_ATTN_CTX3, sets the heads per CTA of the context kernel. 0: 4 heads up to 4 rows, then 8. |
| `GR8_ATTN_CTX3_SP` | int | `0` | 0\|64\|128 | With GR8_ATTN_CTX3, splits the replicated decode context into partitions of 64 or 128 key slots instead of 256. Deterministic but a different fp32 summation order, FP8 KV only. 0: 256 slots. |
| `GR8_ATTN_CTX_TILE` | text | unset (a per-shape tile table) | 2x128\|4x128\|8x128\|16x128\|4x256\|8x256\|16x256\|4x64\|8x64 | With GR8_ATTN_P11, forces one decode context tile of HG heads by D dims for every shape, written as HGxD. |
| `GR8_ATTN_D256` | flag | `0` |  | With GR8_ATTN_FLIP, runs the flip context kernel over 256 dims per CTA (4 heads at one row, 8 above) with a dead-slot skip. |
| `GR8_ATTN_FLIP` | flag | `0` |  | Decode attention takes the head-inner flip kernels (decode-flip.cuh) for eligible calls at a context of at least GR8_ATTN_FLIP_MIN. Takes precedence over the head-batched kernels. |
| `GR8_ATTN_FLIP_MIN` | int | `2048` |  | With GR8_ATTN_FLIP, the smallest context in tokens at which a decode call takes the flip kernels. |
| `GR8_ATTN_FRONT` | flag | `0` | 0\|1 | Runs a decode step's first attention kernels (KV append, index append and index query prep, or index reuse) as one launch when the step's rows fit one query tile. Same bits. |
| `GR8_ATTN_HEADS` | flag | `1` |  | Decode attention scores and context load each key once for a group of heads. 0 = the per-head kernels and turns off GR8_ATTN_SC2, GR8_ATTN_CTX2 and GR8_ATTN_CTX3. |
| `GR8_ATTN_P11` | flag | `0` |  | Decode attention uses the P11 tile retune (per-shape context tiles) on the head-batched path, ahead of GR8_ATTN_SC2, GR8_ATTN_CTX2 and GR8_ATTN_CTX3. |
| `GR8_ATTN_POSIDX` | flag | `0` | 0\|1 | A full-index layer's decode attention in the buckets up to 2048 lists the keys in position order instead of running the indexer, and skips that layer's idx_q GEMV. Deterministic, a different fp32 order. |
| `GR8_ATTN_QT` | flag | `0` | 0\|1 | Decode absorb stores q and rope lane-transposed and the scores kernel reads them with 16 B loads. Needs GR8_ATTN_SC2 2 or 4, not with GR8_AVP1. Same bits, FP8 KV only. |
| `GR8_ATTN_SC2` | int | `0` | 0\|1\|2\|4\|8 | Decode attention scores in the head-group kernel take S = 1, 2, 4 or 8 key slots per warp with the query loaded once for all S. Same bits, FP8 KV only. |
| `GR8_ATTN_SC2_PART` | flag | `1` |  | With GR8_ATTN_SC2, a head count that is not a multiple of 16 (TP=3, TP=6) also takes the scores kernel with a partial last head group. 0 = those layers keep the 16-head kernel. Same bits. |
| `GR8_ATTN_SC2_TR` | int | `0` | 0\|1\|2 | With GR8_ATTN_SC2, 1 reduces the heads' sums by one transpose across the warp and 2 does the same reduction eagerly. S = 8 ignores it. Same bits. |
| `GR8_ATTN_SEL2` | int | `0` | 0\|8\|16\|32\|64 | Decode index selection runs its final sort as a rank by count over 8, 16, 32 or 64 CTAs per row. Same bits. |
| `GR8_ATTN_TILE_L` | int | `0` |  | With GR8_ATTN_P11, the context length in tokens used to pick the decode context tile, in place of the captured call's context. 0: the call's context window. |
| `GR8_AVP1` | flag | `0` |  | Decode absorb and value projection run as one launch for any row count, weights in registers, rows staged 8 at a time. |
| `GR8_CP` | flag | 1 in a CP build |  | Context-parallel KV: the first GR8_CP_R tokens are replicated and later pages are spread across ranks on a CP fabric of their own. Read only in builds with -DG6_CP=1. |
| `GR8_CP_MMAX` | int | the most rows of any step |  | Raises the row count the CP fabric is sized for. A value below the computed rows has no effect. |
| `GR8_CP_ONECHAIN` | flag | `0` | 0\|1 | Past R, the other ranks' head groups run as one attention call instead of one call per group, with partials published in the GR8_CP_S3=0 order. Byte-exact, TP=4 only, not with GR8_ATTN_FLIP. |
| `GR8_CP_PF_HASH2` | flag | `1` | 0\|1 | The CP prefill page check uses the block-reduced HASH2 page hash. 0 = the older k_cp_pf_hash. Read only in builds with -DG6_CP_PF_HASH2=1. |
| `GR8_CP_PF_VERIFY` | flag | `0` |  | Every KV page a rank stages to another rank in CP prefill is checked against the hash the sender computed. A mismatch sets the fabric error word. |
| `GR8_CP_R` | int | `0` |  | The first R tokens of KV are replicated on every rank and pages past R are spread across ranks. Under S2 it is also the largest request that shares decode steps. A multiple of 64. |
| `GR8_CP_R2` | int | none: every session takes GR8_CP_R |  | A smaller R a session may take (the head's policy names which): its first GR8_CP_R2 tokens are replicated and its pages past them are spread across ranks, so a run past it holds fewer pages below place and more sessions past R fit. Every step past GR8_CP_R2 runs the CP block. A multiple of 64 below GR8_CP_R and a context bucket boundary; needs GR8_RT_LOCAL. |
| `GR8_CP_R2_BIG` | int | `1` |  | With GR8_CP_R2: how many sessions past GR8_CP_R2 may run at GR8_CP_R at once. Later ones take GR8_CP_R2, and so does one the pool has no room for at GR8_CP_R. A turn that continues a retained run keeps that run's R. |
| `GR8_CP_S3` | flag | `1` |  | CP combine order: 1 = the head-group overlap order, 0 = the plain order. Same arithmetic. |
| `GR8_CP_STEP_ROWS` | int | `0` | 0..4096 | S2 only: a decode step past R carries at most this many rows and the CP fabric is sized for it instead of the widest S2 step. Sessions past R give the same tokens. |
| `GR8_CP_STEP_SEG` | flag | `0` | 0\|1 | With GR8_CP_STEP_ROWS, a round past R takes only sessions past R, so a short session never runs the CP path because a long one is resident. |
| `GR8_DEC_BK` | int | `2048` |  | The first decode context bucket boundary in tokens. Decode graphs are captured per bucket and GR8_RT_L must be a bucket boundary. |
| `GR8_DEC_SHARED_SCRATCH` | flag | `0` |  | One decode attention scratch serves every layer of the main model, which frees memory for the KV pool. Needs --pfv2. |
| `GR8_DEC_SPLIT` | int | `0` | 0.. | Captures the decode step as two graphs split at this layer, so the second submission overlaps the first one's GPU time. Needs --graph. 0: one graph. |
| `GR8_DFL_BODY_I4` | flag | `0` | 0\|1 | The DFlash2 drafter's body GEMVs read INT4 g32 copies of their MXFP8 records, made at boot. Drafts only, +382 MB per rank. Needs --dflash-manifest. |
| `GR8_DFL_HEAD_I4` | flag | `0` | 0\|1 | The DFlash2 drafter's lm_head GEMV reads INT4 g32 records, the same ones as GR8_DRAFT_HEAD_I4. Drafts only. Needs --dflash-manifest. |
| `GR8_DFL_PFCTX` | flag | `1` |  | With --dflash, 0 gives the drafter no prompt context from the prefill, so it starts from the last prompt token. Flash --prefill with --dflash needs 0. |
| `GR8_DFL_REUSE` | int | 2 when serving DFlash2 with MTP, else 0 | 0..2 | With DFlash2 under S2, 1 lets a request whose prefix-reuse hit landed in the drafter's window keep the drafter, and 2 also gives the drafter to a request alone on MTP. |
| `GR8_DFL_REUSE_MIN` | int | `256` | 0..2048 | With GR8_DFL_REUSE, the context rows the drafter must hold before it drafts. Until then the request runs MTP rounds that feed it. |
| `GR8_DFL_SEL_ALPHA` | float | `1` |  | The DFlash2 selector walk scores each candidate as logit + a * correction instead of logit + correction. Drafts only. Needs --dflash-manifest and the selector. |
| `GR8_DRAFT_HEAD2` | int | `0` | 0..2 | Experimental: the MTP draft head runs in two stages, a 2-bit screen then the exact MXFP8 logits of its best 64 rows. 2 also runs the MXFP8 GEMV and counts misses. Needs --mtp. |
| `GR8_DRAFT_HEAD_I4` | flag | `0` | 0\|1 | The MTP draft's lm_head GEMV reads INT4 g32 records made at boot from the MXFP8 records, +134 MB per rank at TP4. Drafts only. Needs --mtp. |
| `GR8_DRAFT_HEAD_KS` | int | `32` | 2\|4\|8\|16\|32 | With GR8_DRAFT_HEAD_I4 or GR8_DFL_HEAD_I4, the INT4 head GEMV's k parts per tile. Its product with GR8_DRAFT_HEAD_TPC must be at most 32. |
| `GR8_DRAFT_HEAD_TPC` | int | `1` | 1\|2\|4\|8\|16\|32 | With GR8_DRAFT_HEAD_I4 or GR8_DFL_HEAD_I4, the INT4 head GEMV's tiles per CTA. Its product with GR8_DRAFT_HEAD_KS must be at most 32. |
| `GR8_DRAFT_LOGIT_DUMP` | path | unset |  | Every rank writes its vocab slice of the MTP first draft pass logits to this path plus .rank<R>. Needs --mtp and no --serve. |
| `GR8_DRAFT_PAR` | int | 2 when serving DFlash2 with MTP, else 0 | 0..2 | S2 with DFlash2, a solo round runs the drafter step on its own CUDA stream under MTP's first form. 2 also puts drafting steps on a high-priority stream. Same bytes as serial. |
| `GR8_DS41_BORROW_TOPK` | path | unset: a reuse layer whose index source is outside the layer subset reads the last list written |  | A file of raw int32 [rows][512] top-k lists. A run over a layer subset reads it in place of an index source that the subset leaves out. |
| `GR8_DS41_IDX_DENSE_MAXK` | int | `0` | 0.. | Layers 24 to 36 run the dense index stage in place of the candidate-block path while a chunk's key bound is at most this many keys. 0 = always the candidate path. |
| `GR8_DS41_IDX_LOGITS` | text | `mma` | mma\|dp4a\|mmafast | Kernel of the dense index logits: mma = tensor cores, bit-identical to dp4a; dp4a = the reference kernel; mmafast = tensor cores with their own fp32 rounding. |
| `GR8_DS41_IDX_MAXK` | int | `0` | 0.. | Upper bound on the keys of the dense index grid, below the pool capacity. For bring-up. 0 = no bound. |
| `GR8_DS41_IDX_PIPE_B` | flag | `0` |  | 1 = the dense index logits of the source layers 2, 8, 14 and 20 run on the pipelined kernel (launch_logits_pipe). The candidate layers keep their path. |
| `GR8_DS41_IDX_PIPE_QR` | flag | `0` |  | 1 = the dense index logits run on the query-stationary cp.async kernel (launch_logits_pipe_qr). |
| `GR8_DS41_IDX_PW` | int | MX, the decode width | 0..65536 | Prefill chunk width of the indexer: every per-row buffer of the indexer holds max(MX, PW) rows. The step driver's per-row arrays must hold the same number of rows. |
| `GR8_DS41_IDX_PWK` | int | `0` | 0.. | Ratio-1 key rows the indexer scratch holds logits for, when that is below capacity plus rows. It bounds the largest position the indexer serves. 0 = capacity plus rows. |
| `GR8_DS41_IDX_Q_TRIG` | flag | `0` |  | 1 = the index query's rope sin and cos are computed once per row and cached. |
| `GR8_DS41_IDX_REUSE` | flag | `0` |  | 1 = layers 24 to 36 keep layer 20's top-512 list and run no wq_b, logits or top-k of their own; layer 20 then skips its candidate blocks. |
| `GR8_DS41_IDX_TOPK` | text | `mc` | mc\|1cta | Top-k kernel of the dense index stage: mc = several CTAs per long row, 1cta = the single-CTA reference kernel. Both give the same lists. |
| `GR8_DS41_IDX_WQB_TC` | int | `0` | 0.. | The indexer's wq_b runs on the attention block's tensor-core GEMM at this many rows or more. Not bit-exact against the scalar GEMM. 0 = off. |
| `GR8_DS_ATTN_CPA` | flag | `0` |  | 1 = the attention core gathers its K rows with cp.async (k_attn_cpa). The output is bit-identical to k_attn. |
| `GR8_DS_ATTN_GV` | flag | `1` |  | Decode projections of up to 8 rows run on the MXFP8 GEMV where the weight has .mx.gv records. 0 = the generic MXFP8 GEMM for every projection. |
| `GR8_DS_ATTN_GV_KS` | int | `0` | 0.. | k-parts per output tile of the attention GEMV. 0 = the built-in pick per shape (gv_pick_ks). |
| `GR8_DS_ATTN_GV_TPC` | int | `0` | 0.. | Output tiles per CTA of the attention GEMV. 0 keeps the built-in 1. |
| `GR8_DS_ATTN_PW` | int | `0` | 0.. | Prefill row capacity of the attention ops, apart from the decode width MX. A value above MX gives every layer a window ring of 128 + PW - 1 slots and one scratch that all layers share, and lets one op take PW rows. 0, or a value up to MX, keeps the per-layer scratch and the MX row limit. |
| `GR8_DS_ATTN_SPLIT` | flag | `0` |  | 1 = two CTAs per row and head group in the attention core, joined by an online-softmax merge. Needs GR8_DS_ATTN_CPA=1. |
| `GR8_DS_ATTN_TC` | flag | `0` |  | 1 = the five attention projections at 9 or more rows run on the block-scaled tensor-core GEMM (k_gemm_mx8t) where the shape allows it. |
| `GR8_DS_ATTN_TC_MF` | int | `0` | 0\|1\|2 | Grid order of the tensor-core GEMM: 0 = the N tile is the fastest grid dimension, 1 = the M tile is, 2 = the M tile only for a weight of 100 MiB or more (the Engram wkv). |
| `GR8_DS_BLOCK` | text | the checkpoint's DSpark block, 5 |  | DSpark draft block size; the draft step always holds 5 rows a session, so only 5 is served. |
| `GR8_DS_C8PACK` | flag | `0` | 0\|1 | Short fresh prompts (up to 64 rows each, up to 8 of them, 512 rows in all) prefill together in one ragged pass and share one prompt-tail step. Needs --serve, --s2-sessions >= 2 and GR8_DS_PREFILL_MIN >= 2. |
| `GR8_DS_COMP_FP8` | flag | `0` |  | 1 = the ratio-2 compressor weights are packed to E4M3 with UE8M0 scales at bind time and prefill steps of 9 or more rows use them. Lossy, and its effect on quality has not been measured. |
| `GR8_DS_DEC_STAGE` | flag | `0` | 0\|1 | Decode consumes of the main model (1..MX rows) read the peers' slots with 16-byte loads through shared memory. |
| `GR8_DS_DSP_GV` | flag | `0` |  | 1 = the DSpark drafter's projections run on the MXFP8 GEMV for 1 to 8 rows. A projection that cannot take the GEMV then stops the process. 0 = the generic MXFP8 GEMM. |
| `GR8_DS_DSP_TC` | flag | `0` |  | 1 = a DSpark prefill projection of more than 8 rows in one group runs on the attention block's tensor-core MXFP8 GEMM (ds41_gemm_mx_tc) when that entry takes the shape, else on the generic GEMM. |
| `GR8_DS_ENGRAM` | flag | `0` | 0\|1 | 1 runs Engram at layers 1 and 14: every step hashes the last four tokens of each row, reads this rank's table rows from NVMe and adds the gated result to the trunk. It needs GR8_DS_ENGRAM_PREFIX. 0 skips Engram, and the other GR8_DS_ENGRAM_ knobs are not read. |
| `GR8_DS_ENGRAM_MTP` | flag | `0` | 0\|1 | Allows GR8_DS_ENGRAM=1 together with GR8_DS_MTP. The verify rows' n-gram window comes from the host-built step input and a rejected draft's ring tokens are rewritten before they are read. Not yet compared against Engram at MTP 0. |
| `GR8_DS_GATE_RO` | flag | `0` |  | 1: the read-once prefill path computes the main layers' gate logits with k_gate_logits_ro (8 experts x 8 tokens per warp) instead of k_gate_logits_mt. Same fma chain per token and expert, so the logits are the same bytes. The 128-expert draft layers keep k_gate_logits_mt. |
| `GR8_DS_MHC_DECOMP` | flag | `0` | 0\|1 | Prefill chunks above MX rows run the mHC consume and prepare as separate decomposed kernels (ds_mhc_decomp.cuh). |
| `GR8_DS_MHC_FUSE` | flag | `1` | 0\|1 | The mHC consume of one sub-layer and the prepare of the next run as one trunk pass, except where Engram sits between them. 0 runs them apart. |
| `GR8_DS_MHC_ROW_SHARD` | flag | `0` | 0\|1 | W-row prefill: each rank runs the late layers' mHC for a quarter of the rows and all-gathers the trunk state. Needs W 4096, GR8_DS_PF_TAIL 128, TP4, GR8_DS_PF_RS=1 and GR8_DS_MHC_FUSE=1. |
| `GR8_DS_MHC_Y_MXFP8` | flag | `0` | 0\|1 | With GR8_DS_MHC_ROW_SHARD=1, the y all-gather carries MXFP8 instead of BF16. Lossy. |
| `GR8_DS_MOE_FC1BSTG` | flag | `0` |  | 1: the read-once fc1 expert loop copies its FP16 B fragments to a two-record cp.async shared ring before the same MMA. Every value is the same. |
| `GR8_DS_MOE_FC2STG` | flag | `0` |  | 1: the read-once fc2 expert loop loads a group's act tile into registers in one batch and then stores it to shared memory, instead of one load and one store per trip. Every address and value is the same. Decode, k_fc1_ro and the shared expert are not touched. |
| `GR8_DS_MOE_L2HINT` | int | `0` | 0..2 | An L2 evict_last hint on the routed expert weight loads of the read-once expert loop. 0: off. 1: k_fc1_ro only. 2: k_fc1_ro and k_fc2_ro. No activation load or store carries a hint, and decode and the shared expert are not touched. Every value is the same. |
| `GR8_DS_MOE_PDL` | flag | `0` |  | 1: the MoE kernels of a decode layer after build_xfrag launch with programmatic dependent launch. 0: plain launches. |
| `GR8_DS_MOE_PW_BOUND` | flag | `0` |  | 1: the chunked prefill path (33 to 256 rows) sizes its expert-slot grid from the routes the top-k cut keeps (rows x GR8_DS_MOE_TOPK) instead of rows x 6. The grid only shrinks when GR8_DS_MOE_TOPK is below 6. CTAs past the plan's slot count exit before any work, so the output is the same. |
| `GR8_DS_MOE_SH_1PASS` | flag | `0` |  | 1: the shared expert at 33 to 48 rows runs in one pass of 6 tiles, so it reads its weights once, instead of a loop over 32-row chunks. |
| `GR8_DS_MOE_TOPK` | int | `6` | 1..6 | Routed experts kept per token on the main layers (384 experts). The kept routes are the planner's first picks (best biased key first) and their weights renormalize over the kept. The 128-expert draft layers keep their top-3. 6 is the model's top-k. Quality below 6 is not measured. |
| `GR8_DS_MOE_TOPK_DECODE` | int | the main step's routed experts | 1..6 | Routed experts a decode step takes, below the prefill's (GR8_DS_MOE_TOPK). A screen arm, not the model. |
| `GR8_DS_MTP` | text | 0, no drafts |  | DSpark drafts verified a round, 1..5 (the checkpoint's three draft layers, block of 5). Served only. |
| `GR8_DS_PF_DIRECT` | flag | `0` | 0\|1 | On a W-row prefill chunk the attention output and the MoE combine write their collective's send slot in place; reduce-scatter slices go as flat slices. Needs GR8_DS_PREFILL_W above MX. |
| `GR8_DS_PF_GATHER` | flag | `0` | 0\|1 | Prefill consumes above MX rows read one device BF16 sum of the peers' partials instead of the peers' slots. |
| `GR8_DS_PF_LANES` | text | 1, one lane |  | 1 or 2. 2 runs W-row prefill quanta of up to 2W rows of one session on two lanes with their own streams and fabric. |
| `GR8_DS_PF_LATE_FINAL` | flag | `0` | 0\|1 | With GR8_DS_PF_TAIL, a chunk with at least tail + window rows after it skips the late layers' rows that no later reader needs. |
| `GR8_DS_PF_POOL` | text | off |  | 0, or 2..the run-0 collectives: the prefill fabric's layer exchanges and all-gathers reuse this many slots each (FabricConfig::pool). |
| `GR8_DS_PF_RS` | flag | `0` | 0\|1 | W-row prefill: the layer exchanges before the late split run as reduce-scatter plus all-gather. Needs GR8_DS_PF_TAIL and GR8_DS_PREFILL_W above MX. |
| `GR8_DS_PF_TAIL` | text | the model's tail rows |  | Rows of each prefill chunk that run the late layers (bounded replay): 0 (off) or the attention window..4096. |
| `GR8_DS_PREFILL_KEEP` | text | 0, every prompt row |  | 0, or 128 and more: only the last N prompt positions enter the DSpark draft's key rings. |
| `GR8_DS_PREFILL_MIN` | int | `0` | 0.. | Prompts of at least this many tokens prefill in W-row chunks (GR8_DS_PREFILL_W); shorter ones, and every prompt at 0, run as decode steps. Several sessions (--s2-sessions > 1) need 2 or more. |
| `GR8_DS_PREFILL_W` | text | MX: no W-row prefill |  | Rows of one prefill chunk, MX..4096. GR8_DS_ATTN_PW and GR8_DS41_IDX_PW must name the same width. |
| `GR8_DS_RELAXED_ACCEPT` | text | off: exact acceptance |  | DSpark verify accepts a greedy row's draft by a relaxed rule: ratio:R (0 < R <= 1) and / or topp:P (0 < P < 1), comma separated. Not exact. |
| `GR8_DS_RO_L2LA` | int | `0` | 0..1048576 | L2 lookahead of the read-once expert loop, in CTAs. While CTA b of k_fc1_ro runs, its warp 0 prefetches into L2 the weights of CTA b + n; k_fc2_ro (2 CTAs per SM) looks 2n ahead. A hint only: every value, order and write is the same. 0: off. |
| `GR8_DS_RO_ORDER` | flag | `0` |  | 1: the read-once expert loop starts its CTA groups in the order k_order_ro computes (heavy and light experts merged in proportion to their work, heaviest first) instead of expert-id order. Only the start order changes, so every output byte is the same. |
| `GR8_DS_SH_TC` | flag | `0` |  | 1: the shared expert of the read-once prefill path runs on the tensor-core MXFP8 GEMM (x and act quantized to E4M3 per 32 with E8M0 scales), and the combine reads its bf16 output. Not bit-identical to 0. Decode and the chunked path keep the GEMV. Costs a plain copy of the shared weights per MoE layer (9,461,760 bytes) plus scratch, and needs the attention object's ds41_gemm_mx_tc and ds41_sh_* entries linked. |
| `GR8_DS_SRC_FAST` | int | `0` | 0..3 | Bits of the source-layer fast paths: 1 = the ratio-2 compressor projections of up to 8 rows use k_f32gemv2 (bit-identical to k_f32gemm2), 2 = the indexer's wq_b runs on the MXFP8 GEMV (a different summation order). |
| `GR8_DS_STRIPE` | flag | `0` | 0\|1 | The main fabric stripes writes of at least GR8_DS_STRIPE_MIN bytes over both rails (needs --ips2). |
| `GR8_DS_STRIPE_MIN` | int | `1048576` | 8192.. | With GR8_DS_STRIPE=1, the smallest write the fabric stripes, in bytes. |
| `GR8_DS_WAIT_K` | int | `0` | 0..2 | Who waits for the peers' tails before a consume kernel: 0 every CTA, 1 one wait kernel ahead of it, 2 one CTA that then releases the others through a device word (main model only; others fall back to 1). |
| `GR8_DS_WOB_PERSIST` | flag | `0` |  | 1 = the coalesced-epilogue wo_b GEMM of a direct-send prefill chunk runs as a persistent kernel (k_gemm_mx8t_c2_persist). |
| `GR8_DTR` | flag | `1` |  | Uses the exlr8 release's dense trellis matrices (q_b, o_proj, the dense MLP) where a layer has them. 0 = the int4 or MXFP8 records, which the layer must then have. |
| `GR8_DTR_NBFAST` | flag | `1` |  | Single-matrix dense trellis GEMVs whose group stride exceeds their block stride assign CTAs output block first instead of k part first. CTA order only. |
| `GR8_DTR_NBFAST_OUT` | flag | `1` |  | As GR8_DTR_NBFAST, for single-matrix dense trellis GEMVs whose block stride is at least their group stride. |
| `GR8_DTR_PF_COPY` | flag | `0` |  | Prefill runs o_proj on an exact FP8 copy of its trellis tiles too, as q_b and the dense MLP do, about 25 MB more per layer at TP4. Same bits. |
| `GR8_DTR_PF_KDA_COPY` | flag | `1` |  | Flash prefill makes exact FP8 copies of the KDA q, k and v trellis matrices for the prefill GEMMs. 0 = no copies. |
| `GR8_DTR_PF_SHARE_SUH` | flag | `0` |  | Flash prefill dense MLP: up reuses gate's rotated input instead of rotating with its own suh vector. |
| `GR8_FPF_ATTN_P1` | flag | `0` |  | Flash prefill attention runs the P1 bodies instead of the GEMM bodies. |
| `GR8_FPF_DSA_P1` | flag | `0` |  | Flash prefill DSA layers run the P1 bodies instead of the GEMM bodies. |
| `GR8_FPF_P1` | flag | `0` |  | Flash prefill runs the P1 bodies everywhere, the reference the GEMM bodies are checked against. |
| `GR8_FUSE` | int | `1` | 0\|1\|2 | Decode MoE K1: 1 = one fused launch, 0 = separate launches, 2 = fused with a KS1=2 WG=2 split at 2 and 3 rows, which rounds differently. |
| `GR8_GEMM` | int | `0` | 0..3 | Prefill MXFP8 GEMM kernel: 1 = k_gemm_mx8 (v1), 2 = v2 with 128-row CTAs, 3 = v2 with 64-row CTAs, 0 = v2 with row blocks by grid size. |
| `GR8_GLM53F_CACHE` | text | `off` | off\|0\|aligned\|terminal | GLM-5.3-Flash --serve with --prefill: aligned keeps session checkpoints so a later prompt with the same start is prefilled from there. terminal also checkpoints each request's end and is not cold-exact. |
| `GR8_GLM53F_DEC_MHC_STAGE` | flag | `1` |  | Flash decode: the fused mHC consume stages the peers' partials in shared memory before summing, at 1 or 4 rows. Same rank-order summation. |
| `GR8_GLM53F_DRAFT_KV` | flag | `1` |  | Flash prefill fills the MTP draft's caches only. 0 = runs the full draft step over the prompt rows, which uses the draft fabric. |
| `GR8_GLM53F_DSA_IDXQ_TPC` | int | `1` |  | Flash decode: tiles per CTA of the indexer's wq_b GEMV. |
| `GR8_GLM53F_DSA_OPROJ_ROT` | flag | 1 for steps of more than one row, else 0 | 0\|1 | Flash decode: 1 = the o_proj trellis GEMV reads the attention output as rotated fragment tables, 0 = it rotates the attention output itself. |
| `GR8_GLM53F_DSA_OPROJ_SPLIT` | ints | 4 k parts, 1 sub-part |  | Dense trellis o_proj GEMV split as k1,s1[,k4,s4]: k parts and sub-parts at one row and at more rows. |
| `GR8_GLM53F_DSA_QB_FUSE` | int | `1` | 0\|1\|2 | Flash decode: q_b and the indexer's wq_b GEMV run as one 640-CTA kernel (k_dsa_qb_idxq) when their shapes match. 2 = the same with the indexer's CTAs first, 0 = two launches. |
| `GR8_GLM53F_DSA_QB_ROT` | flag | `1` |  | Flash decode: the q_a norm kernel writes q_a rotated as the fragment tables the q_b trellis GEMV reads. 0 = the q_b GEMV rotates the q_a rows itself. |
| `GR8_GLM53F_DSA_QB_SPLIT` | ints | QA/128 k parts (16 GLM, 12 Flash), 1 sub-part |  | Dense trellis q_b GEMV split as k1,s1[,k4,s4]: k parts and sub-parts at one row and at more rows. |
| `GR8_GLM53F_DSA_XIN_ORDER` | flag | `0` |  | Flash decode with GR8_GV_MULTI: a nonzero value launches the layer-input GEMVs in the order idx_k, q_a, idx_w, idx_g, kv_a instead of q_a, kv_a, idx_k, idx_w, idx_g. |
| `GR8_GLM53F_FPF_DENSE_P1` | flag | `0` |  | Flash prefill dense MLP layers run the P1 bodies instead of the GEMM bodies. |
| `GR8_GLM53F_FPF_KDA_P1` | flag | `0` |  | Flash prefill KDA q, k, v and o_proj run as the P1 projections instead of M-row GEMMs. |
| `GR8_GLM53F_KDA_MX` | flag | `1` | 0\|1 | KDA decode q / k / v read MXFP8 records built at bind with FP16 activations. 0 reads the BF16 weights. Not byte- exact, and the records are an extra copy outside the loader's accounting. |
| `GR8_GLM53F_KDA_OPROJ_SPLIT` | ints | ks 8 at 1 row, 4 at 2+ rows, sub 1 |  | Sets the k-split and sub-split of the KDA o_proj trellis GEMV as k1,s1,k4,s4 (1 row, then 2+ rows), in `kernels/dtr.cuh`. |
| `GR8_GLM53F_KDA_PF_IN1` | flag | `0` | 0\|1 | 1 runs the older v1 intra-chunk kernel in KDA prefill instead of k_kda_intra3. |
| `GR8_GLM53F_KDA_PF_P1` | flag | `0` | 0\|1 | 1 runs the older row-loop prep kernel in KDA prefill instead of launch_prep_pf. |
| `GR8_GLM53F_KDA_PF_ST1` | flag | `0` | 0\|1 | 1 runs the older v1 state kernel in KDA prefill instead of k_kda_state2. |
| `GR8_GLM53F_KDA_PROJ_KS` | int | `16` | 8\|16\|32 | Sets the k-split of the KDA trellis q / k / v projection GEMV. |
| `GR8_GLM53F_KDA_PROJ_MODE` | int | `0` | 0\|1\|2 | Picks how the KDA trellis q / k / v projection rotates its input: 0 inside the GEMV, 1 and 2 in a separate pre- rotate launch that also carries the f_a / g_a / b GEMVs (1) or not (2). |
| `GR8_GLM53F_KDA_SCAN` | int | `0` | 0\|1 | 1 runs the Flash KDA prefill scan as the CH16 bf16-MMA scan (`kda/kda_g9.cuh`), a precision arm that is not byte-exact. 0 runs the base scan. |
| `GR8_GLM53F_KDA_VS` | int | `2` | 1\|2\|4 | With GR8_KDA_SCAN=1, the v slices per head of the G9 KDA scan kernel: 1, 2 or 4. |
| `GR8_GLM53F_KV_CHUNKS` | flag | `0` | 0\|1 | Flash: the DSA MLA and index caches sit in kvchunks regions under one chunk map per model, so S2 Resize events can take chunks online or offline on every rank. |
| `GR8_GLM53F_KV_IDLE_CHUNKS` | int | `1` | 1.. | With GR8_FLASH_KV_LEND_MS, the chunks left online while the rest are lent. A request that fits them is admitted without a recall. |
| `GR8_GLM53F_KV_LEND_MS` | int | `0` | 0.. | With GR8_FLASH_KV_CHUNKS=1, once the serve is idle this many ms the head lends every chunk above GR8_FLASH_KV_IDLE_CHUNKS, and recalls them before a request that needs them. 0: never lend. |
| `GR8_GLM53F_MHC_FUSE` | int | `2` | 0\|1\|2 | Sets the decode mHC consume form for every row count unless GR8_MHC_FUSE_M1 / _M4 set it: 0 a separate peer-wait kernel, 1 the consume waits itself, 2 the fused consume. |
| `GR8_GLM53F_MHC_FUSE_M1` | int | MHC_FUSE, else 2 | 0\|1\|2 | Sets the decode mHC consume form for 1-row steps, as GR8_MHC_FUSE. |
| `GR8_GLM53F_MHC_FUSE_M4` | int | MHC_FUSE, else 2 | 0\|1\|2 | Sets the decode mHC consume form for every step of 2 or more rows, as GR8_MHC_FUSE. |
| `GR8_GLM53F_PF_BF16_EXACT` | flag | `0` | 0\|1 | The Flash BF16 dense MLP fallback in prefill runs plain kernels instead of cuBLAS. |
| `GR8_GLM53F_PF_DRAFT_FULL` | int | `1` | 0\|1\|2 | The Flash MTP draft cache fill runs over the lane's rows in 256-row passes instead of one draft step per MX rows, with the same bits. 2 also counts differing words against the old path. |
| `GR8_GLM53F_PF_DSA_F16GEMM` | flag | `0` | 0\|1 | Flash DSA prefill input projections run as FP16 cuBLAS GEMMs on weights decoded once at bind, instead of the MXFP8 GV kernel. Near-byte, not byte-equal. |
| `GR8_GLM53F_PF_DSA_FULL_GV` | flag | `1` | 0\|1 | Flash DSA prefill runs its GV projections as one launch over all rows. 0 restores the per-32-row launches. Byte- equal. |
| `GR8_GLM53F_PF_DSA_LT` | flag | `1` | 0\|1 | Flash DSA prefill q_a and idx_q run on a cublasLt algo with a bigger tile where the default is one k pass, with the same bits. 0 keeps cublasGemmEx. |
| `GR8_GLM53F_PF_DSA_PROJ_P1` | flag | `0` | 0\|1 | Flash DSA prefill q_b and o_proj trellis projections run as the decode GEMVs per MX rows instead of the prefill trellis GEMM. |
| `GR8_GLM53F_PF_KDA_GZF` | flag | `1` | 0\|1 | 0 makes the G9 KDA prefill scan write and re-read the G16 / Z16 planes instead of building those tiles in its K1 and output kernels (`kda/kda_g9.cuh`). Same bits. |
| `GR8_GLM53F_PF_KDA_TAIL` | int | `1` | 0\|1\|2 | Selects how Flash prefill runs the KDA f_a / g_a / b projections: 0 = the per-MX-row host loop, 1 = full-M 32-row tiles, 2 = full-M 64-row tiles. |
| `GR8_GLM53F_PF_KDA_WS` | flag | `1` | 0\|1 | 0 turns off the warp-specialized GEMM for the Flash KDA prefill projections (`kda/proj_in_prefill.cuh`). |
| `GR8_GLM53F_PF_KP_TC` | int | `2` | 0\|1\|2\|3 | Selects the Flash prefill kpool score kernel: 0 = `kp_scores_pf`, 1 = tensor-core with one stage, 2 = tensor- core with two stages, 3 = the streaming tensor-core form. |
| `GR8_GLM53F_PF_MHC_FUSE` | int | `2` | 0\|1\|2 | Sets how far Flash prefill fuses the mHC update with the fabric wait: 0 = separate passes, 1 = the mHC kernel reads the partials, 2 = it also waits for the payload. |
| `GR8_GLM53F_PF_MHC_PK` | flag | `1` | 0\|1 | 0 turns off the packed mHC fn weights for `k_mhc_out_inv_prepare_pk` in Flash prefill. |
| `GR8_GLM53F_PF_MHC_RR` | flag | `1` | 0\|1 | 0 runs `k_mhc_out_inv_prepare_pk` instead of the row-blocked `k_mhc_out_inv_prepare_rr` when the peer partial is staged in Flash prefill. |
| `GR8_GLM53F_PF_MHC_STAGE` | flag | `1` | 0\|1 | 0 stops Flash prefill at TP2 from copying the peer's partial into device memory with the copy engine before the mHC update. |
| `GR8_GLM53F_PF_MLA_PROJ2` | flag | `1` | 0\|1 | 0 makes Flash prefill attention use the older MLA projection kernels instead of `fa_pf::mla_proj2`. |
| `GR8_GLM53F_PF_NOWIDE` | flag | `0` | 0\|1 | Any value other than 0 stops Flash prefill from reordering the tr3 fc1 / fc2 records into the wide load order at bind time. |
| `GR8_GLM53F_PF_POOL_OWNER` | int | on for TP2 Flash prefill (not ring, no P1 flags), else off | 0\|1 | 1 runs the Flash prefill pool-ID exchange on its own fabric (TP2, not ring) and stops the boot if it cannot, 0 turns it off. |
| `GR8_GLM53F_PF_PUT_CTA` | int | `96` |  | Sets the CTA count of the Flash prefill partial publish copy (`k_put_publish_n`) outside the SP path. Copy only, same bytes. |
| `GR8_GLM53F_PF_ROUTER1` | flag | `0` | 0\|1 | 1 runs the Flash prefill router logits on `k_linear_mr` instead of the tiled `k_router_tile` (0 ULP apart). |
| `GR8_GLM53F_PF_SHFP8` | flag | the PF_MOE8 value | 0\|1 | 1 makes an FP8 copy of the Flash trellis shared expert for the prefill GEMMs, which quantizes its activations, and 0 keeps the fp16 trellis pass that matches decode. |
| `GR8_GLM53F_PF_SP` | flag | `0` | 0\|1 | 1 runs Flash prefill mHC sequence-parallel: partials go reduce-scatter, each owner runs mHC on its rows, then all-gathers the normalized rows. |
| `GR8_GLM53F_PF_SPARSE_MLA` | flag | `0` | 0\|1 | 1 runs Flash prefill attention as the fused sparse MLA kernel when a rank has at most 32 heads. |
| `GR8_GLM53F_PF_SPARSE_MLA_512` | flag | `0` | 0\|1 | With GR8_PF_SPARSE_MLA=1, 1 selects the `sparse_mla_pf_512` kernel and makes the V2 kernel the default. |
| `GR8_GLM53F_PF_SPARSE_MLA_H16C` | flag | `0` | 0\|1 | With the V2 sparse MLA kernel and at most 16 heads a rank, 1 runs its 16-head form. |
| `GR8_GLM53F_PF_SPARSE_MLA_V2` | flag | the PF_SPARSE_MLA_512 value | 0\|1 | With GR8_PF_SPARSE_MLA=1, 1 runs the V2 sparse MLA kernel and 0 the 512 or base kernel. |
| `GR8_GLM53F_PF_SP_AGD` | flag | `1` | 0\|1 | 0 stops the GR8_PF_SP owner pass from writing its rows straight into the all-gather send slot, so the publish copies them. Byte-equal. |
| `GR8_GLM53F_PF_SP_AGK` | flag | `1` | 0\|1 | 0 moves GR8_PF_SP all-gathers back to copy-engine copies on the lane's copy stream instead of one SM pass on the compute stream that also writes the FP16 copy. |
| `GR8_GLM53F_PF_SP_AGKA` | flag | `1` | 0\|1 | 0 limits the GR8_PF_SP_AGK compute-stream all-gather to lane B instead of both prefill lanes. |
| `GR8_GLM53F_PF_SP_ATTD` | flag | `1` | 0\|1 | 0 stops the GR8_PF_SP attention o_proj GEMM from writing each row into its owner's slice of the send slot. Byte- equal. |
| `GR8_GLM53F_PF_SP_FIND` | flag | `1` | 0\|1 | 0 stops the GR8_PF_SP MoE final sum from writing each row into its owner's slice of the send slot. Byte-equal. |
| `GR8_GLM53F_PF_SP_PUT_CTA` | int | `96` | 1..1024 | Sets the CTA count of the GR8_PF_SP send copies (`k_put_rs`, `k_put_publish_n`). Copy only, same bytes. |
| `GR8_GLM53F_PF_SP_X16` | flag | `0` | 0\|1 | With GR8_PF_SP and GR8_PF_ROUTER_SPLIT, an MoE layer's MLP input travels as FP16 rows instead of FP32 rows, halving that all-gather. Byte-equal. |
| `GR8_GLM53F_PF_SP_XFRAG` | flag | `1` | 0\|1 | 0 stops the GR8_PF_SP owner pass and gather from writing the next attention's fragment tables directly. Same tables either way. |
| `GR8_GLM53F_PF_STAGE_ASYNC` | flag | 1 at TP2 when the config allows, else 0 | 0\|1 | 1 waits for and copies the peers' partials on a per-lane side stream under the other lane's block, at any rank count, and stops the boot if the config cannot run it. |
| `GR8_GLM53F_PF_XFRAG` | flag | `0` | 0\|1 | 1 makes the Flash mHC kernel write the next attention's input as f16 and bf16 fragment tables instead of FP32 rows when that attention reads only fragments. Same bits. |
| `GR8_GLM53F_ROUTER_MF` | flag | `1` |  | Flash MoE router: 1 runs a 4-row step with the 4-row router kernel, 0 with the per-row kernel. Each warp sums its output in the same order either way. |
| `GR8_GLM53F_ROUTER_NW` | int | `2` | 2\|4\|8 | Flash MoE router: warps per CTA of the router GEMV (one output channel per warp). Each warp sums in the same order at any width. |
| `GR8_GLM53F_S2_COST` | flag | `0` |  | Flash solo serving prices S2 rounds with spec_len.h's Flash cost table for this build's width instead of the GLM step costs. |
| `GR8_GLM53F_STATE_BUDGET_MIB` | int | `8192` |  | With GR8_FLASH_CACHE and no --kv-store, the size in MiB of each rank's in-memory checkpoint store. |
| `GR8_GV_MULTI` | flag | `1` |  | Steps of up to 8 rows run the GEMVs on the layer input (q_a, kv_a, the indexer's k and weights) as one launch. 0 = one launch each. Same bits. |
| `GR8_IDXQ_SPLIT` | flag | g6m4: 1 when TP>1 on GLM, else 0; m4sim: 0 | 0\|1 | Splits the indexer's replicated wq_b GEMV by output tiles across the ranks and all-gathers it on the split-GEMV fabric, with the same bits as the full GEMV. |
| `GR8_IDX_GRP` | int | `0` | 0..64 | In S2 batched decode steps the per-row indexer reads each key tile once for up to this many rows of one session, with the same bits. 0 is off. |
| `GR8_IDX_VEC` | flag | `0` | 0\|1 | The per-row decode index_scores loads its key tile with 16 B vector loads (page size 32 or more), with the same bytes. Read once before graph capture. |
| `GR8_K1_FUSE_POLICY` | flag | `1` | 0\|1 | 0 turns off the timed table that makes exlr8 v2 MoE layers pick fused or separate fc1 / fc2 launches by row count, so GR8_FUSE applies as given. |
| `GR8_K1_SPLIT` | text | the release split per row count |  | Sets the decode MoE K1 split: a KS1,WG,NS2 triple for every row count, or a table of M:KS1xWG/NS2 entries that keeps the bytes. A bad split stops the bind. |
| `GR8_K1_SPLIT_M1` | ints | GR8_K1_SPLIT's KS1,WG,NS2 triple, else the release split (4,1,2) |  | Sets the decode MoE K1 split KS1,WG,NS2 for 1-row steps, over GR8_K1_SPLIT. Another fc1 partial count can change the sums. |
| `GR8_K1_SPLIT_M2` | ints | GR8_K1_SPLIT's KS1,WG,NS2 triple, else the release split (1,2,8) |  | Sets the decode MoE K1 split KS1,WG,NS2 for 2- and 3-row steps, over GR8_K1_SPLIT. Another fc1 partial count can change the sums. |
| `GR8_K1_SPLIT_M4` | ints | GR8_K1_SPLIT's KS1,WG,NS2 triple, else the release split (4,2,2) |  | Sets the decode MoE K1 split KS1,WG,NS2 for steps of 4 or more rows, over GR8_K1_SPLIT. Another fc1 partial count can change the sums. |
| `GR8_KVB` | text | `bf16` | bf16\|fp8 | fp8 loads every layer's kv_b (UK / UV) as the release's block FP8 bytes and scales instead of BF16, with the same output bytes. Needs --v2-dense. |
| `GR8_MHC_BF16` | int | `0` | 0\|1\|2 | Quality test: rounds mHC residual stream writes to BF16, 1 = after both blocks, 2 = only after the MLP block. |
| `GR8_MOE_TOPK` | int | `8` | 1..8 | Keeps only this many routed experts per token and renormalizes their weights. A model change for screens. |
| `GR8_MTP_KSTOP` | int | 2 with --serve, DFlash2 and --mtp 2+, else 0 | 0\|1\|2 | A solo session's MTP rounds verify only the drafts the MTP head is confident in: 1 by the GR8_MTP_KSTOP_TAU thresholds, 2 by the drafter choice's MTP estimates. |
| `GR8_MTP_KSTOPB` | flag | `0` | 0\|1 | In shared S2 rounds each session verifies its own count of the --mtp drafts, from the MTP head's cumulative draft probability against the GR8_MTP_KSTOPB_TAU thresholds. |
| `GR8_MTP_KSTOPB_EXPLORE` | int | `0` |  | Measurement: with GR8_MTP_KSTOPB, every Nth round's shared sessions verify a hashed draft count instead. 0 is off. |
| `GR8_MTP_KSTOPB_KMIN` | int | `0` | 0\|1 | Sets the fewest drafts a session verifies under GR8_MTP_KSTOPB. |
| `GR8_MTP_KSTOPB_TAU` | floats | `0.6,0.88,0.95` |  | Sets the GR8_MTP_KSTOPB thresholds t1,t2,... (up to 7) on each session's cumulative draft probability, at every session count. Past the last entry there is no stop. |
| `GR8_MTP_KSTOPB_TAU_S<n>` | floats | MTP_KSTOPB_TAU |  | Sets the GR8_MTP_KSTOPB thresholds t1,t2,... (up to 7) for rounds of n sessions, n 1..8. |
| `GR8_MTP_KSTOPB_WIDTHS` | ints | every width the rule reaches |  | Sets the shared-round main widths captured as graphs besides the multiples of --mtp + 1. A width outside them takes the next most likely drafts. |
| `GR8_MTP_KSTOP_KMAX` | int | --mtp | 1..3 | Sets the most drafts a solo MTP round chains under GR8_MTP_KSTOP. Without DFlash2 it must not exceed --mtp. |
| `GR8_MTP_KSTOP_SEED3` | float | `0.55` | 0..1 | With GR8_MTP_KSTOP=2, sets the prior acceptance of a third MTP draft in the drafter choice's estimates. |
| `GR8_MTP_KSTOP_TAU` | floats | `0,0.2,0.5` |  | With GR8_MTP_KSTOP=1, sets the thresholds t1,t2,t3: a solo round verifies draft j while the product of the draft probabilities up to j is at least t_j. |
| `GR8_MTP_LASTROW` | flag | `0` | 0\|1 | A first-form MTP draft step routes only each session's last row through the MTP layer's routed experts. Caches, drafts and log-probabilities keep their bytes. |
| `GR8_MTP_SPLIT` | flag | g6m4: 1 when TP>1 on GLM with --mtp, else 0; m4sim: 0 | 0\|1 | MTP draft steps split eh_proj and the draft layer's replicated input GEMVs by output tiles across the ranks, gathered on a draft fabric (port + 72). Same bits. |
| `GR8_NGRAM` | path | unset |  | A solo DFlash2 session also drafts from this n-gram table file and verifies its chain when it promises more tokens per verify ms. Needs --serve. |
| `GR8_NGRAM_KMAX` | int | --dflash-k | 1.. | Sets the longest GR8_NGRAM chain a round verifies. |
| `GR8_NGRAM_KMIN` | int | `2` | 1.. | Sets the shortest GR8_NGRAM chain worth a round. |
| `GR8_NGRAM_MARGIN` | float | `0.1` | 0..10 | Sets how many more tokens per verify ms a GR8_NGRAM chain must promise than the drafter's drafts. |
| `GR8_NOWIDE2` | flag | `0` |  | Keeps the MoE fc2 weights in the old order instead of the wide order. Outputs are bit-identical. |
| `GR8_PF_AOP` | flag | 1 when built with -DPF_AOP | 0\|1 | 0 turns off the prefill fc1 A-operand share, which quantizes the fc1 A operand once into records the fc1 CTAs stage. Byte-exact. Needs the -DPF_AOP build. |
| `GR8_PF_AOP_MMIN` | int | `3072` |  | Sets the fewest rows a prefill fc1 pass needs to use the A-operand share. Smaller passes run the stock fc1, with the same bytes. |
| `GR8_PF_ATTN_ROWS` | int | the whole sub-batch (the GLM launcher sets 1024) |  | Prefill attention runs in chunks of this many rows inside a sub-batch. The rest of the layer runs at the sub- batch size. |
| `GR8_PF_BLAS_ROWS` | int | the whole sub-batch (the GLM launcher sets 1024) | 16.. | Runs the prefill cuBLAS calls in chunks of this many rows, so a row's bits depend on its chunk and not on the sub-batch size. |
| `GR8_PF_DENSE` | int | `8192` | 0..2147483616 | Sets the context up to which prefill attention takes the dense path. Past it the sparse, indexer-selected path runs. 0 means always sparse. |
| `GR8_PF_DRAFT_SHARE` | int | on when the target has tensor-core prefill states | 0\|1 | The MTP draft prefill borrows the target prefill's tensor-core attention scratch instead of its own, about 2 GB a rank. 1 requires it, 0 gives the draft its own. |
| `GR8_PF_DTR8` | flag | `0` | 0\|1 | Flash prefill runs the dense trellis matrices from FP8 tile copies made once at load (`dtr_find.h`) instead of the fp16 trellis-tile GEMM. |
| `GR8_PF_F12` | flag | 1 when built with -DPF_F12 | 0\|1 | 0 runs the stock prefill fc1 instead of the fat-CTA `k_pf_fc1f`, where adjacent expert blocks share one quantized A record (byte-exact). |
| `GR8_PF_FC2ONE` | flag | `1` | 0\|1 | 0 makes the Flash fp16 MoE fc2 run one output Hadamard per block instead of one per 128-column group. |
| `GR8_PF_FC2V2` | flag | `1` | 0\|1 | 0 runs the Flash prefill fc2 as `k_pf_fc2g` instead of `k_pf_fc2v`, which stages A and weights in shared memory. |
| `GR8_PF_GEMMF32` | flag | `1` | 0\|1 | 0 keeps big FP32-output prefill GEMMs (the row-owner QO o_proj) on the guarded `k_gemm_mx8b` instead of the unguarded `k_gemm_mx8f`. Same bits either way. |
| `GR8_PF_GEMMG` | int | `8` |  | Sets how many row blocks `k_gemm_mx8f` full-M launches group in their raster order, and 0 or 1 gives a plain raster. Launch order only, same bits. |
| `GR8_PF_HASH_GPU` | flag | `0` | 0\|1 | The per-sub-batch residual hash is a splitmix64 sum computed on the GPU instead of an FNV of the rows on the host, which gives a different value. |
| `GR8_PF_HX` | flag | `1` | 0\|1 | 0 makes the Flash MoE fc1 rotate x per CTA instead of reading rows rotated and scaled once per token by `k_pf_hx`. |
| `GR8_PF_HXF` | flag | `1` | 0\|1 | 0 runs `k_pf_hx` and `k_pf_hxq` as two passes on GR8_PF_Q8 layers instead of the fused `k_pf_hxqf`. |
| `GR8_PF_ISGRT_RG` | int | `8` | 8.. | Sets the rows per CTA of the prefill index-score kernel, rounded down to a multiple of 8 (below 8 means 8). Same bits. |
| `GR8_PF_MOE8` | flag | `0` | 0\|1 | 1 runs the Flash prefill MoE on MXFP8 activations instead of the fp16 passes, and turns on the shared expert's FP8 copy unless GR8_PF_SHFP8 says otherwise. |
| `GR8_PF_OWN` | int | `0` | 0\|1\|2\|3\|4 | Sets row-owner prefill: 1 = owners compute the next layer's projections, 2 = also the indexer, 3 = owners run all heads of their rows (QO pack), 4 = TP4 pair-owner QO. |
| `GR8_PF_OWNPERM` | flag | `0` | 0\|1 | With GR8_PF_OWN=3, interleaves the QO owner rows in 16-row blocks instead of contiguous row ranges. |
| `GR8_PF_Q8` | flag | `0` | 0\|1 | 1 runs the Flash routed fc1 on MXFP8 activations quantized once per token and layer instead of fp16 (not byte- equal to the fp16 path). |
| `GR8_PF_QO` | text | `b` | b\|v1\|r1 | Selects the QO attention kernel for GR8_PF_OWN=3 and 4: b, v1 or r1. Other values make the QO state fail to create. |
| `GR8_PF_QP_CTX` | int | the KV cap | 1.. | With GR8_PF_OWN=4 and GR8_PF_QP_ROLL, sets the QO key-table size in positions, and a request whose prompt plus generation is longer runs GR8_PF_OWN=2. |
| `GR8_PF_QP_EARLY` | int | `0` | 0\|1\|2 | With GR8_PF_OWN=4: 1 sends the counterpart's partial before the own rows' o_proj, 2 also splits the QO call in two. |
| `GR8_PF_QP_LAYERS` | text | all |  | With GR8_PF_OWN=4, the comma list of model layers and ranges that run pair-owner attention, while the others run GR8_PF_OWN=2. |
| `GR8_PF_QP_MIN` | int | `768` |  | With GR8_PF_OWN=4, a prompt shorter than this many tokens runs GR8_PF_OWN=2, and 0 sends every request that fits GR8_PF_QP_CTX through OWN=4. |
| `GR8_PF_QP_ROLL` | int | `0` | 0\|2\|3\|4 | With GR8_PF_OWN=4, builds the pair weights per layer in this many rolling slots from the rank pack and the partner's half over RDMA. 0 uses a resident --qp-pack. |
| `GR8_PF_QP_WIRE_E` | int | 1 entry; set, it needs GR8_PF_QP_ROLL | 1..4 | With GR8_PF_OWN=4 and GR8_PF_QP_ROLL, sets the entries in the RDMA ring of pair-weight half images (`qp_wire.h`). |
| `GR8_PF_ROUTER_SPLIT` | flag | `0` | 0\|1 | 1 splits the Flash prefill MoE router by rows across the ranks and exchanges the route tables on a dedicated fabric per lane (`flash_route_split.cuh`). |
| `GR8_PF_RS` | flag | `0` | 0\|1 | 1 runs GLM prefill with reduce-scatter plus all-gather (five collectives) instead of all-reduce. |
| `GR8_PF_RSDIRECT` | flag | `0` | 0\|1 | With GR8_PF_RS=1, o_proj, the dense down GEMM and the MoE final sum write straight into the reduce-scatter send slot, so the put only rings the doorbell. |
| `GR8_PF_SELX_CTX` | int | `53248` | 32769..1048576 | Prefill index tiles of at least 32 rows at a context of at least this many tokens pick their top keys with SELX instead of SEL32, with the same IDs in the same order. |
| `GR8_PF_SHR` | flag | `0` | 0\|1 | 1 runs the Flash trellis shared expert inside the routed MoE pass as extra routes instead of its own pass. Same bytes. |
| `GR8_PF_STREAMS` | int | `1` | 1\|2 | 2 runs the two GLM prefill lanes on two CUDA streams, and any other value keeps one stream. |
| `GR8_PF_STRIPE` | flag | 1 with --pf-ips2 or on a three-rank ring, else 0 | 0\|1 | 1 splits large prefill payload writes over two rails (needs --pf-ips2 or --ips2, or a three-rank ring that uses the other PCIe root's device). |
| `GR8_PF_TILE` | int | `256` | 64\|128\|256 | Sets the prefill attention query tile in rows. Other values make the attention state fail to create. |
| `GR8_PF_TR_BM` | int | `16` | 8\|16 | Sets the row blocks per CTA of large prefill GEMMs on trellis tiles (`k_gemm_tr`): 8 or 16. Small launches take 4. |
| `GR8_PF_WM` | flag | `1` | 0\|1 | 0 runs the Flash routed fc1 on `k_pf_fc1` instead of the weight-major `k_pf_fc1ws` for passes of at least GR8_PF_WM_MIN rows. Byte-equal. |
| `GR8_PF_WM2` | flag | `0` | 0\|1 | 1 runs the Flash prefill fc2 weight-major (`k_pf_fc2w`) instead of `k_pf_fc2v` when the fc1 runs weight-major. Byte-equal. |
| `GR8_PF_WM_MIN` | int | `6144` |  | Sets the fewest rows a Flash MoE prefill pass needs before the routed fc1 runs weight-major. |
| `GR8_QA_SPLIT` | int | g6m4: 2 when TP>1 on GLM, else 0; m4sim: 0 | 0\|1\|2 | Splits the decode layer-input GEMVs across the ranks by 16-output tiles and all-gathers them on their own fabric. 1 = q_a, kv_a and the indexer's k and weights, 2 = q_a only. Same bits. |
| `GR8_QA_SPLIT_TPC` | int | `1` |  | Sets the output tiles per CTA of the GEMVs that GR8_QA_SPLIT, GR8_IDXQ_SPLIT and GR8_MTP_SPLIT narrow. It only lowers a GEMV's own count and keeps the output bits (rank.cuh qs_narrow). |
| `GR8_QO_NANINJ` | flag | `0` |  | Writes an FP16 NaN into q[0] of the prefill attention on every call, a fault injection for the error checks. |
| `GR8_REUSE_ALIGN` | flag | 1 with --serve, 0 offline | 0\|1 | With --reuse-prefix, a hit keeps only whole cold-prefill grid cells (--prefill rows), so the result equals a cold prefill bit for bit. 0 keeps the whole matched prefix. 1 needs --prefill. |
| `GR8_REUSE_NOEND` | flag | 1 with --serve, 0 offline | 0\|1 | With --reuse-prefix, a hit no longer needs the new prompt to reach the end of what the last sequence wrote, so a prompt can be replayed as it is. 0 restores that rule (prefix_reuse.h). |
| `GR8_REUSE_STRICT` | flag | `0` | 0\|1 | With --reuse-prefix, a hit needs the new prompt to extend the previous sequence exactly, the strict append rule (prefix_reuse.h). |
| `GR8_ROUTER_TPC` | int | `2` | 1\|2 | Sets the router GEMV's output tiles per CTA: 1 = 16 CTAs of 512 threads instead of 8 of 1024. Same tiles, k parts and reduction order, so the same bits (bind.cuh). |
| `GR8_ROUTE_OVL` | int | `0` | 0\|1\|2 | Decode MoE layers with a trellis shared expert and 2+ rows: runs the shared expert on a forked stream during the router GEMV and plan kernel. 1 = its fc1 and fc2, 2 = fc1 only. Same bytes out. |
| `GR8_RT_FMT` | text | `fp8` | fp8\|int4 | Sets the row format of the GR8_RT_STORE records: fp8 = the cache's own 656 B row, int4 = the 416 B INT4 row of --kv-format int4. |
| `GR8_RT_L` | int | CP_R |  | Sets the first position GR8_RT_STORE holds: a multiple of 64, at least GR8_CP_R, and a context bucket boundary. With GR8_RT_LOCAL it must equal GR8_CP_R. |
| `GR8_RT_LOCAL` | int | `0` |  | Releases each session's tail pages past GR8_RT_L to the store: they cycle through this many window pages per rank and are read back from slots or the store. Needs GR8_RT_STORE. |
| `GR8_RT_PLACE` | int | every tail page of --cap |  | With GR8_RT_LOCAL, sets the main model's physical pages per rank, so --cap becomes the logical length only. Must hold the GR8_CP_R prefix and one window. |
| `GR8_RT_POOL` | int | 4096, or with RT_LOCAL 2048 x rows a step x sessions past R |  | Sets GR8_RT_STORE's resident slots per indexer group, at least 2048. With GR8_RT_LOCAL a smaller pool lets fewer sessions past R in at once. |
| `GR8_RT_SIZE` | int | unset |  | Sets each rank's GR8_RT_STORE lease budget in bytes (the launcher's KV_ROWS size=). With GR8_RT_LOCAL it sizes the records, and the pool admits a tail past R only when its records fit. |
| `GR8_S2_ADAPT` | flag | `0` |  | S2: chooses each round's MTP draft count (1..k) from the sessions' accept rates and the S2 step costs. Adds about 1 GiB of main-step graphs. |
| `GR8_S2_ADMIT_HOLD` | flag | `0` | 0\|1 | S2: an admission with another request queued behind it and a slot left holds its boundary, so a burst of queued requests is admitted at one boundary instead of one per notice. |
| `GR8_S2_BELOW` | int | CP_R |  | S2: the context below which sessions share the pool's pages: 64..GR8_CP_R, a multiple of 64. |
| `GR8_S2_DECODE_PF` | int | `0` |  | S2: while any session decodes, a prefill quantum takes at most this many rows on one lane, so decode rounds come round sooner. A multiple of 64, at most --prefill. 0 = the full quantum. |
| `GR8_S2_DECODE_ROUNDS` | int | `0` | 0..64 | S2: while a prefill waits and sessions decode, runs this many decode rounds per prefill quantum instead of one. 0 or 1 = off. |
| `GR8_S2_DSEL` | flag | `1` |  | S2 with DFlash2: a session that decodes alone drafts each round with DFlash2 or MTP, whichever is expected to commit more tokens per ms. 0 = DFlash2 every round. |
| `GR8_S2_DSEL_DWELL` | int | `8` |  | S2 drafter choice: rounds a session stays on one drafter before it may switch. |
| `GR8_S2_DSEL_MARGIN` | float | `0.15` |  | S2 drafter choice: switches when the other drafter's expected tokens per ms is higher by this fraction. |
| `GR8_S2_DSEL_MODEL` | flag | `0` | 0\|1 | S2 drafter choice: 1 prices both drafters from the frozen cost table in GR8_S2_DSEL_TABLE. Needs --serve and --dflash-manifest. |
| `GR8_S2_DSEL_PROBE` | int | `4` |  | S2 drafter choice: every this-many MTP rounds the DFlash2 drafter also drafts, and its drafts are scored against the next tokens. |
| `GR8_S2_DSEL_TABLE` | text | unset |  | With GR8_S2_DSEL_MODEL=1, the drafter choice's cost table: a key:value list (v1..v8, d, m, e, p and optional keys) or the path of a file holding one. |
| `GR8_S2_GROW` | flag | `1` | 0\|1 | S2: a session holds the KV pages it needs now and takes more as its output grows, while every live session can still reach its worst case. 0 = every run holds its worst case from admission. |
| `GR8_S2_NOTICE_MAX` | int | `8` | 2..8 | S2: the latest boundary a notice may target, in boundaries after the one it was anchored at. Lower only from a measured margin: a notice that lands after its boundary stops the engine. |
| `GR8_S2_TAIL_FIRST` | flag | `0` | 0\|1 | S2: a session with only its prompt tail left is served before any prefill, and its one-row tail does not ask for a decode round, so a burst of short prompts reaches its first tokens one tail a step. |
| `GR8_SPEC_DRAFT` | floats | `8.1,5.5,2.1` |  | S2 with DFlash2: overrides the DFlash2 round, MTP round and each extra MTP chain step costs with three comma- separated nonnegative times in ms. |
| `GR8_SPEC_TABLE` | floats | built-in measured table |  | S2 with DFlash2: overrides the verify costs for 1..8 rows with eight comma-separated positive times in ms, for the DFlash2 draft count and the drafter choice. |
| `GR8_TR3_W2ALL` | flag | `0` |  | Uses the window-2 fc1 split (256-thread CTAs, KS1 x2, WG 1) at M > 1 on every MoE layer, not only layers with 2-bit units. Same k ranges and order, so the same bits (bind.cuh k1_split). |
| `GR8_V2_SHARED` | text | FP8 copy | trellis | trellis: prefill runs an exlr8 v2 layer's shared expert as a trellis pass instead of from an FP8 copy made at boot, which saves the copy (about 9.7 MB per layer at TP4). |
| `GR8_WARM` | int | `0` | 0\|1\|2 | Warms the L2 with the next weights while a decode collective waits: 1 = this layer's router and shared fc1 and the next layer's q_a, 2 also kv_a and q_b. Loads only, no output change. |

## Per rank

These may differ between ranks: paths, per-box resources, logging and timing.

| variable | type | unset | values | what it does |
|---|---|---|---|---|
| `GR8_DEC_EVPROF` | int | `0` |  | Prints per-stage decode step times from event nodes in the captured main-step graphs every N main steps, in the rank logs. 0: off. |
| `GR8_DEC_PROF` | flag | `0` |  | Measures decode stages with a synchronize at every mark and prints them per rank. Plain launches only, the engine stops with --graph. |
| `GR8_DFL_DUMP` | path | unset |  | With --dflash, rank 0 writes every draft step's context features, drafts and committed tokens to this file, for engine/ref/dflash2_replay.py. |
| `GR8_DRAFT_IN_DUMP` | path | unset |  | Rank 0 writes the inputs of every MTP first draft pass to this file. Needs --mtp and no --serve. |
| `GR8_DS_C2_TRACE` | flag | `0` |  | 1 = count every launch of the coalesced-epilogue wo_b GEMM (k_gemm_mx8t_c2), so a run can print ds41a::c2_dispatch_count() at stop. |
| `GR8_DS_CACHE_MIB` | int | the prefix cache's own size | 1.. | Size in MiB of the host prefix cache of DS41 session snapshots (--reuse-prefix). |
| `GR8_DS_ENGRAM_CACHE_MIB` | int | `4096` | 1..1048576 | Size in MiB of the host cache of Engram table rows, shared by both layers. It holds 256-byte rows, so 4096 MiB holds 16 M rows. |
| `GR8_DS_ENGRAM_IO_THREADS` | int | `32` | 1..1024 | Threads that read Engram table rows from NVMe. A decode step needs about 20 reads and a prefill chunk thousands; 64 reads a prefill chunk fastest. |
| `GR8_DS_ENGRAM_LOOKAHEAD` | flag | `0` | 0\|1 | 1 starts a host thread that, during a solo prefill, reads the next chunk's layer 1 rows into the row cache while the current chunk runs the later layers. It only warms the cache: the results do not change. |
| `GR8_DS_ENGRAM_PREFIX` | path | no Engram tables; GR8_DS_ENGRAM=1 then stops the boot |  | The path prefix of this rank's Engram files: PREFIX.engram.json (the manifest), PREFIX.engram.cmap (the token map), and PREFIX.engram.L001.w8 / .s8 and PREFIX.engram.L014.w8 / .s8 (the rows and scales of this rank's slice of each table). |
| `GR8_DS_ENGRAM_VERBOSE` | flag | `0` | 0\|1 | 1 prints one line of row store numbers per layer and step on stderr: requests, cache hits, reads, bytes and time. |
| `GR8_DS_ENGRAM_WAYS` | int | `16` | 1..16 | Ways of the set-associative Engram row cache. |
| `GR8_DS_FEED_DIR` | path | off |  | Debug: a directory of reference activations the step driver feeds in place of its own, layer by layer. |
| `GR8_DS_MOE_ELOG` | flag | `1` |  | 1: every prefill router call (more than 32 rows) writes the distinct experts of each layer to a device ring that the host reads after the chunk. 0: no record is written. |
| `GR8_DS_SPAN` | flag | `0` | 0\|1 | Prints the host time of each step's spans. |
| `GR8_DS_SUBDUMP` | path | off |  | Debug: a directory where each step's sub-layer partials are written. |
| `GR8_DS_XTRACE` | flag | `0` | 0\|1 | With GR8_XTRACE=1 and DSpark, also trace the DSpark draft fabric. |
| `GR8_FABRIC_CONNECT_TRIES` | int | `1200` |  | Fabric TCP connect attempts, 100 ms apart. Raise it when ranks load at different speeds. |
| `GR8_GLM53F_DRAFT_CACHE_DUMP` | path | unset |  | Flash with --mtp: each rank writes its draft cache pages before the first live draft to this path plus .s<seq>.r<rank>.bin. |
| `GR8_GLM53F_KDA_STATE_DUMP` | path | unset |  | After each offline sequence, each rank writes the KDA S state and conv state of every layer to PATH.rank<R>. |
| `GR8_GLM53F_KV_CRASH_AT` | text | unset | begun\|synced\|released | Ends the process at that ledger point of a chunk resize (flash_kv.h). |
| `GR8_GLM53F_KV_LEDGER` | path | unset |  | With GR8_FLASH_KV_CHUNKS=1, the box's host ledger that chunk resizes move bytes through. With GR8_FLASH_KV_LEND_MS the rank writes its lendable bytes to this path plus .gpu. |
| `GR8_GLM53F_PF_AMARK` | flag | `0` | 0\|1 | With GR8_PF_PROF, the Flash DSA prefill attention adds its own profile marks. |
| `GR8_GLM53F_PF_KDA_TAIL_CHECK` | flag | `0` | 0\|1 | Flash prefill also runs the KDA tail projections on the host-loop path, compares them byte for byte with the tiled result and stops on a difference. |
| `GR8_GLM53F_PF_POOL_CAPTURE` | path | unset |  | A non-empty value stops the boot when GR8_FPF_ATTN_P1=1, because pool capture needs the prefill attention path. |
| `GR8_GLM53F_PF_PREWARM` | flag | 1 at TP2, else 0 | 0\|1 | 1 creates the per-lane Flash DSA and MLP prefill states at boot instead of inside the first chunk, and any other value turns it off. |
| `GR8_GLM53F_PF_XTRACE` | flag | `0` | 0\|1 | Measurement: Flash prefill stamps GPU timestamps around each layer's fabric wait and copy for a trace. |
| `GR8_GLM53F_RESET_ALL` | flag | `0` |  | Flash serve clears every attention cache page per request instead of only the pages the last request wrote. |
| `GR8_GLM53F_RESET_CHECK` | flag | `0` |  | Flash serve stops if any cache page past the cleared ones is not zero. It reads every cache, so it is for tests. |
| `GR8_IMG_VMM` | int | `0` |  | Builds the device weight image from cuMemCreate chunks mapped into one range, 1 = the recommended granularity, N = N MB chunks. Same bytes. Falls back to cudaMalloc and says why. |
| `GR8_K1_TLBW` | int | `0` | 0..1048576 | At the start of each decode MoE fc1 launch, prefetches the routed slots' fc1 and fc2 weight translations into L2 at this granule in KB. No output change. |
| `GR8_KVMAN_TRANSPORT` | text | `auto` | tcp\|rdma\|auto | How this box reaches a KV node. A rank whose path to the node runs through another box sets up RDMA but cannot write, and auto does not fall back after setup. |
| `GR8_KV_DRAIN_DELAY` | text | unset |  | For tests, rank R sleeps the given ms inside every Nth KV tier drain that copies anything, set as <rank>:<ms>[:<every>]. |
| `GR8_KV_SHARE_WHEN` | text | `idle` | idle\|always | With a group KV lease, idle makes the head offer its blocks to other groups only while no session is resident or queued, always offers on every report. |
| `GR8_L2PF` | int | `0` | 0..31 | Window mask for a paced L2 prefetch of the weights the decode step reads next, on a side stream while DRAM is idle (`kernels/l2pf.cuh`). Writes nothing, so output bytes are unchanged. |
| `GR8_L2PF_GBPS` | float | `175` |  | Sets the pace of the GR8_L2PF prefetch in GB/s. |
| `GR8_L2PF_MB` | float | `10` |  | Sets the most MiB the GR8_L2PF prefetch issues in one window. |
| `GR8_L2PF_OPMB` | float | `8` |  | Sets how many MB of o_proj entries the GR8_L2PF prefetch takes per layer. 0 takes all. |
| `GR8_L2PF_T` | int | `0x3ff` |  | Bit mask of the GR8_L2PF prefetch targets: 1 x_in, 2 idx_q, 4 q_b, 8 UK, 16 UV, 32 o_proj, 64 router, 128 shared or dense fc1, 256 eh_proj, 512 head records. |
| `GR8_L2PF_US` | floats | `30,30,100,30` |  | Sets the GR8_L2PF budget per window in us at the pace, as four numbers: MLP all-reduce wait, gather waits, attention, attention all-reduce wait. |
| `GR8_L2PF_WAVE` | int | `8` | 1..32 | Sets the most 8 KiB GR8_L2PF prefetch chunks issued per poll. |
| `GR8_LOADER` | text | the Rust loader | cpp | cpp loads a pack through the C++ path instead of the Rust loader. |
| `GR8_LOAD_CHUNK_MB` | int | `32` | 1.. | Sets the chunk size in MB that the Rust loader reads and the weight check sums. |
| `GR8_LOAD_MODE` | text | `read` | read\|mmap | Picks how the Rust loader reads weight files: read or mmap. Other values fail the load. |
| `GR8_LOAD_THREADS` | int | `8` | 1.. | Sets the number of Rust loader worker threads. |
| `GR8_LOGIT_DUMP` | path | unset |  | Offline runs write each main step's head logits (each rank's vocab slice) to PATH.rank<R>, with a header. Not with --serve. |
| `GR8_LOGIT_DUMP_FROM` | int | `0` |  | Main-step dumps (GR8_LOGIT_DUMP, GR8_RESID_DUMP) start at this step. |
| `GR8_MEM_MIN` | float | `8` | 0.. | GiB of MemAvailable the boot must keep after the weights and each layer of the KV pool. Below it the engine stops with a message instead of meeting the OOM killer. |
| `GR8_MTP_L2PF` | int | `0` | 0..3 | With GR8_L2PF and --mtp, runs the paced L2 prefetch in MTP draft steps: 1 the draft step's windows, 2 the head waits prefetch the next draft step's front, 3 both. Same output bytes. |
| `GR8_MTP_L2PF_US` | float | `60` | 0..1000 | Sets the budget in us at the pace of each GR8_MTP_L2PF head-wait prefetch window. |
| `GR8_MTP_LPLOG` | int | `0` | 0\|1\|2 | Prints one # kstop line per solo S2 round with the drafter, verified and accepted counts and the drafts' log- probabilities: 1 on rank 0, 2 on every rank. |
| `GR8_NGRAM_LOG` | flag | `0` |  | Rank 0 prints one line per round that takes a GR8_NGRAM chain. |
| `GR8_PF_ADUMP` | path | unset |  | On the first prefill call of layer position 0, each rank writes diagnostic buffers to DIR/<name>.r<rank>.bin. Synchronizes. |
| `GR8_PF_EVENTS` | flag | `0` | 0\|1 | GLM prefill attention allocates and records CUDA timing events per query tile (`attn/prefill.h`). Other values make the attention state fail to create. |
| `GR8_PF_HASH` | flag | `0` | 0\|1 | Flash prefill prints a hash of each sub-batch's residual rows (stream sync, copy to host, FNV) on its `prefill rows` line. |
| `GR8_PF_HEAD` | flag | `0` | 0\|1 | With GR8_LOGIT_DUMP set, Flash prefill also runs the head on the prompt rows and writes their logits to the dump. |
| `GR8_PF_IDDUMP` | path | unset |  | Rank 0 writes the ordered index IDs and positions of the first, middle and last full-index layers of each prefill sub-batch to this directory. |
| `GR8_PF_PDUMP` | path | unset |  | Each rank writes the IDs, attention probabilities and index scores of selected prefill rows and layers to this directory (`pf_pdump`). |
| `GR8_PF_PDUMP_EVERY` | int | `32` |  | With GR8_PF_PDUMP, dumps the rows whose position is a multiple of this number. |
| `GR8_PF_PDUMP_LAYERS` | ints | `0,3,20,34,55,74` |  | With GR8_PF_PDUMP, the comma list of model layers to dump, plus their index-score source layers. |
| `GR8_PF_PROF` | flag | `0` |  | Prefill synchronizes after each stage and rank 0 prints a per-stage time profile (ms) to stderr. |
| `GR8_PF_QP_ROLL_SER` | flag | `0` | 0\|1 | Measurement: 1 builds the rolling pair-weight slots on the prefill lane stream instead of the stage stream, so the build time is fully exposed. |
| `GR8_PF_QP_WARM` | flag | `0` | 0\|1 | With GR8_PF_OWN=4, warms the pair cuBLAS and MXFP8 GEMM shapes at load so the first request does not pay for them. |
| `GR8_PF_RCDUMP` | path | unset |  | Each rank appends per-expert prefill route counts per MoE layer and sub-batch to `routes.r<rank>.jsonl` in this directory. |
| `GR8_PF_RSHASH` | flag | `0` | 0\|1 | Measurement: with GR8_PF_RS=1, syncs the stream after each put and prints an FNV hash of every owner slice of the send slot. |
| `GR8_PF_STAGGER` | int | `0` |  | Lane B's prefill stream first sleeps this many microseconds, so its first work starts after lane A's. Writes nothing. |
| `GR8_PF_STRIPE_MIN` | int | `1048576` | 8192.. | With GR8_PF_STRIPE=1, the smallest prefill payload write in bytes that goes out split over the two rails. |
| `GR8_POOL_LOG` | flag | `0` | 0\|1 | Prints one stderr line per S2 pool claim with its first page, page count and the retained runs it evicted. |
| `GR8_PROGRESS_ISOLATE` | flag | `1` | 0\|1 | Moves every thread that is not a fabric proxy off this rank's proxy CPUs before the first fabric barrier, so no worker thread preempts a progress thread. 0 turns it off. |
| `GR8_QA_SPLIT_CPU` | int | --proxy-cpu minus 9, or -1 when --proxy-cpu is 8 or less |  | Sets the CPU core of the replicated-GEMV gather fabric's proxy thread (GR8_QA_SPLIT, GR8_IDXQ_SPLIT, GR8_MTP_SPLIT). -1 leaves it unpinned. |
| `GR8_QO_CACHE` | path | unset |  | With GR8_PF_OWN=3, the first boot writes each layer's full-head q_b, o_proj, UK and UV tensors to DIR/<layer>.qo.bin and later boots read them instead of decoding them. Same bits. |
| `GR8_QO_DUMP` | flag | `0` |  | Prints an FNV-64 hash of the prefill attention q to stderr on every call (skipped inside a graph capture). |
| `GR8_QO_ROPECHK` | flag | `0` |  | Checks the prefill rope table against the decode rope function once at bind and prints the mismatch count to stderr. |
| `GR8_RESID_DUMP` | path | unset |  | Writes rank 0's residual stream after the embedding and after every attention and MLP sublayer of each offline main step to this file, in the m4sim dump format. |
| `GR8_REUSE_HDUMP` | path | unset |  | Writes every layer's KV cache bytes for the prefilled rows of each sequence GR8_REUSE_HDUMP_SEQ lists to DIR/hd.<q>.r<rank>.bin, a prefix-reuse diagnostic (reuse_hdump.cuh). |
| `GR8_REUSE_HDUMP_SEQ` | ints | unset | 0.. | Lists the sequence numbers (comma-separated, 0 and up) whose cache rows GR8_REUSE_HDUMP dumps. |
| `GR8_REUSE_POISON` | flag | `0` | 0\|1 | With --reuse-prefix, sets every stale cache row past the reused prefix to NaN on a hit, so a read of a stale row shows in the output. A test gate. |
| `GR8_REUSE_ZSCRATCH` | flag | `0` | 0\|1 | With --reuse-prefix, clears every layer's per-sequence attention scratch before the suffix prefill on a hit (indices to -1, output to 0), a prefix-reuse diagnostic. |
| `GR8_ROUTE_LOG` | path | unset |  | Writes one JSON line per MTP main step to this file with each MoE layer's distinct routed expert ids, from rank 0. |
| `GR8_RT_CHECK` | flag | `0` |  | With GR8_RT_FMT=int4, prints the latent SQNR and rope check of the FP8 to INT4 round trip. With GR8_RT_LOCAL also checks the record bookkeeping after every change and stops on an error. |
| `GR8_RT_PF_SLICE` | flag | `1` |  | With GR8_RT_LOCAL, a prefill stages each released page with one read of each row's slice for the layer. 0 reads the page's whole block for every layer of its group, about 3.9x the bytes. |
| `GR8_RT_SELSTAT` | flag | `0` | 0\|1 | With GR8_RT_LOCAL, counts what a prefill reading only the selected rows of released pages would read, at each full-index layer. Measurement only, it synchronizes the stream. |
| `GR8_RT_STORE` | text | unset |  | Sends a session's whole pages past GR8_RT_L to a per-rank lease, and CP decode reads them back. mem: or remote:HOST:PORT (a KV node). Needs --serve, GR8_CP and GR8_S2_BELOW = GR8_CP_R. Local: with remote:auto the launcher gives odd ranks the KV node's second rail. |
| `GR8_S2_BIND_DELAY` | text | unset |  | S2 test: RANK:MS makes that rank sleep MS ms after each request's bind, to show what a bind skew does to the next quantum on the other ranks. |
| `GR8_S2_EVENTLOG` | flag | `0` |  | S2: logs one line when the head anchors a notice, one when each rank applies the commit, and one per prefill quantum. Logs only. |
| `GR8_S2_ROUNDLOG` | flag | `0` |  | S2: rank 0 prints one '# s2_round' line per decode step, with the step's routed experts. |
| `GR8_S2_SHARE_MS` | int | `1000` |  | S2 with a KV node lease: how often in ms the head reports to the node and asks for segments. |
| `GR8_S2_TIMELINE` | int | `0` |  | S2: every N main steps each rank prints, per decode step kind, the mean GPU span, host input prep, launch call and fence wait. Measurement only. |
| `GR8_S2_TIMEOUT` | int | `120` | 1..86400 | S2: seconds the bootstrap waits on the control links before a rank gives up. |
| `GR8_S2_TRACE` | path | off |  | S2: writes a Chrome trace (chrome://tracing, ui.perfetto.dev) of this rank to PATH.r<rank>.json: a span per prefill quantum, prompt tail and decode step, and on the head each request's arrival, wait, admission, first token and finish, in wall-clock microseconds. tools/trace/merge.py joins the ranks and a client's request times. Host clock only. |
| `GR8_STEP_TMR` | flag | `0` |  | S2: prints a '# step_tmr' line per decode step with its launch and fence time in us, its graph key, rows and context. |
| `GR8_TEST_ZERO_DELAY_MS` | int | `0` |  | A test of the zeroing order: every dmalloc zeroing waits behind an N ms kernel on the legacy stream, so a write to new memory not ordered after it loses every time. |
| `TMPDIR` | path | `/tmp` |  | Where the loader writes its planned rank manifests. |
| `GR8_WEIGHT_CHECK` | flag | `1` |  | Sums each loaded image on the CPU and the GPU and stops the boot with exit 3 if they differ, then seals the sums for /health?deep (weight_check.cuh). 0 = off. |
| `GR8_WEIGHT_CHECK_FLIP` | text | unset |  | RANK:OFFSET flips one byte of rank RANK's first image after the load, a test of GR8_WEIGHT_CHECK that makes the boot stop with exit 3. |
| `GR8_XTRACE` | flag | `0` |  | 1 = transport trace: GPU timer stamps per collective, proxy tick stamps per rail and per-QP retry counters, dumped at proxy stop or when GR8_XTRACE_QUIT appears. Measurement only. |
| `GR8_XTRACE_OUT` | path | /tmp |  | Directory for the GR8_XTRACE dumps, g6xtrace.r<rank>.f<n>.txt. |
| `GR8_XTRACE_PF` | flag | `0` |  | With GR8_XTRACE=1, traces the two prefill fabrics instead of the decode fabric. |
| `GR8_XTRACE_QUIT` | path | unset |  | With GR8_XTRACE=1, each fabric dumps its trace once when this file appears. |

## Tests and benchmarks

Read by test programs, audits and benchmarks. `g6m4 --serve` refuses to start while one is set.

| variable | type | unset | values | what it does |
|---|---|---|---|---|
| `GR8_TEST_ATTNK_ARMS` | text | the built-in arms |  | attnk: the arms to run, as name or KEY=V,... specs joined by spaces. |
| `GR8_TEST_ATTNK_BASE` | text | the serving arm SC2=4,TR=2,CTX2=4,CTX3=2,SEL2=16,IDX=16 |  | attnk: the base arm the others compare with, as KEY=V,... over the stock arm (every flag 0), or stock. |
| `GR8_TEST_ATTNK_E1` | flag | `0` |  | attnk: 1 runs the round 10 arm set. |
| `GR8_TEST_ATTNK_H` | int | g6nr::HMAX, the larger head count of a rank in the build |  | attnk: the rank's head count, g6nr::HMIN or g6nr::HMAX of the build (a TP=6 build: 11 or 10). |
| `GR8_TEST_ATTNK_NANKEY` | int | `-1` |  | attnk: the index key position (session 0) whose scale is NaN, -1 none. |
| `GR8_TEST_ATTNK_QSCALE` | float | `1` |  | attnk: scales the test queries. |
| `GR8_TEST_ATTNK_SESS` | int | `1` | 1\|2 | attnk: 2 runs two S2 sessions when the build and shape allow it. |
| `GR8_TEST_ATTNK_TIES` | flag | `0` |  | attnk: 1 makes the test scores tie. |
| `GR8_TEST_CPG_ARMS` | text | the built-in arms |  | cpgroups: the arms to run, as KEY=V,... specs (SC2 TR CTX2 CTX3 HG) or stock, joined by spaces. |
| `GR8_TEST_CPG_LOCAL` | int | `0` |  | cpgroups: puts the N most recent positions first in each row's index list, as a local window. 0: uniform random positions. |
| `GR8_TEST_CP_PF_BREAK` | text | unset | gather\|back | Negative control for the CP prefill staging gate in m4sim: skips the gather or the back half of the staging so the gate must fail. |
| `GR8_TEST_CP_PF_NET` | flag | `0` |  | In m4sim, the CP prefill gather runs as g6m4's real fabric does it, each rank filling its own pool and sending tail pages to the others, instead of the loopback gather. |
| `GR8_TEST_DS41_KERNELS_DIR` | path | unset |  | engine/kernels, for the loader test that checks the Rust mx4g packer against the C++ one (mxfp4_pack.h); unset skips it. |
| `GR8_TEST_DS41_M4_PFW` | int | `0` | 0.. | m4sim: the prompt runs in W-row chunks of this many rows (it must equal GR8_DS_PREFILL_W and exceed MX); 0 runs it as decode steps. |
| `GR8_TEST_DS41_MGATE_DIAG` | flag | `0` |  | ds41_attn_test --mgate: 1 prints, for a kv compare that differs, up to 8 differing elements of the chunk and of the single-row calls, before and after kv prep. |
| `GR8_TEST_DS41_RO_CAP_MB` | float | `400` | 0.. | ds_ro_flags_test: the memory cap in MB on the test's device and host allocations. A mode that would allocate more stops before it runs. |
| `GR8_TEST_DS41_TOKENIZER_DIR` | path | unset |  | The DeepSeek-V4.1-Flash tokenizer directory (tokenizer.json, tokenizer_config.json) the front end's ds41 dialect test reads; unset skips it. |
| `GR8_TEST_FUZZ_ROUNDS` | int | `40` |  | Rounds of the front end tokenizer fuzz test. |
| `GR8_TEST_IDX_DUMP` | path | unset |  | m4sim writes rank 0's selected key ids of the first --layers layer to this file for each main step from GR8_LOGIT_DUMP_FROM on. |
| `GR8_TEST_KVMAN_LEDGER` | path | `kvman-ledger` |  | flash_kv_chunks_test: the kvman-ledger binary. |
| `GR8_TEST_KVSIM_ENTRIES` | flag | `0` |  | kvman-sim: 1 prints the stash entries at the end. |
| `GR8_TEST_KV_HASH` | flag | `0` |  | m4sim prints per layer an FNV hash of the stored latent records of the prompt positions, read from every rank that holds them, after the prefill. |
| `GR8_TEST_KV_RANKS_CMP` | flag | `0` |  | m4sim with --reuse-prefix prints whether every rank's KV page blocks match rank 0's byte for byte. |
| `GR8_TEST_KV_TIER_CHECK` | text | unset |  | m4sim with --reuse-prefix sends every rank's reused pages to this block store (mem:, ssd:PATH, remote:HOST:PORT), poisons them and reads them back. |
| `GR8_TEST_KV_TIER_CHECK_DROP` | int | none dropped | 0\|1\|2 | With GR8_KV_TIER_CHECK past GR8_CP_R, the copies of this part do not come back, so the check must see the poison. |
| `GR8_TEST_LEDGER_CRASH_AT` | text | unset | write\|written | Aborts the process in the host ledger at that point, before or after the rename, so tests can check crash recovery. Only test and debug builds read it. |
| `GR8_TEST_MTP_LASTROW_CHECK` | flag | `0` |  | m4sim prints a check line per draft step with the last row's token, log-probability and hashes of its hidden row, residual and draft-layer caches. |
| `GR8_TEST_PERTURB` | float | `0` |  | m4sim multiplies every rank's residual after layer 0's attention by 1 + eps * u, with u from a hash of row and element, to measure the noise floor. |
| `GR8_TEST_PF_DRAFTDUMP` | path | unset |  | m4sim with --mtp writes the MTP layer's KV and indexer caches after the draft prefill to DIR/r<rank>.s<seq>.{kv,idx}. |
| `GR8_TEST_PF_HASH_POKE` | ints | unset |  | m4sim test hook `rank,row`: adds one ulp to the first word of that lane-0 row on that rank before the cross-rank hash check, which must then report the rank differs. |
| `GR8_TEST_PF_REPLAY_VERBOSE` | flag | `0` |  | pool_conc_replay_test: 1 logs every replayed call. |
| `GR8_TEST_PF_RESDUMP` | path | unset |  | m4sim writes each prefill sub-batch's FP32 residual rows to `resid.s<seq>.p<pos>.f32` in this directory. |
| `GR8_TEST_PF_TIMES` | flag | `0` | 0\|1 | m4sim --serve prints per-request wall-clock timing laps on stderr, from the request file's mtime to the first token. |
| `GR8_TEST_ROW_LINES` | flag | `0` |  | m4sim: prints one line per target-step row (input, token, log-probabilities, residual and logit hashes) and flags any rank whose residual row differs from rank 0's. |
| `GR8_TEST_RT_ADOPT_SLOT` | flag | `0` | 0\|1 | m4sim with GR8_RT_LOCAL: 1 = the next sequence takes the released records over in another slot, as a g6m4 grant does, instead of in place. |
| `GR8_TEST_RT_POISON` | flag | `0` |  | m4sim only: overwrites the local copy of every row sent to GR8_RT_STORE, so a read of the local copy shows in the output. |
| `GR8_TEST_S2_ATTN_PERSESSION` | flag | `0` | 0\|1 | S2: runs attention once per session instead of one batched launch over the sessions. Refused with graphs and on Flash. |
| `GR8_TEST_S2_CLS` | int | `0` |  | m4sim --s2 probe: runs every step at this k1 row class. |
| `GR8_TEST_S2_GEN` | ints | --gen |  | m4sim --s2: per-session generation lengths (comma-separated), so sessions finish at different rounds. |
| `GR8_TEST_S2_GROW_POOL` | int | fixed runs (GR8_S2_RUNS, GR8_S2_PAGES or the default layout) | 1.. | m4sim --s2: the sessions' runs come from a pool of this many pages (s2/pool.h) and grow on demand: each starts with its prompt + 1 + guard rows, grows before each round, sits a round out when the pool would be unsafe, and gives its pages back when it finishes. At most the pool's pages. Not with GR8_S2_RUNS or GR8_S2_PAGES. |
| `GR8_TEST_S2_INSERT` | text | unset |  | m4sim --s2: q:after:unless:id,id,.. gives session q an insert rule (s2_batch.h), several rules joined by semicolons. unless -1 = none. |
| `GR8_TEST_S2_PAGES` | ints | unset |  | m4sim --s2: seed[,pages] gives each slot a scattered run from a seeded shuffle of the pool. Output must equal the default layout's. |
| `GR8_TEST_S2_PFBASE` | ints | unset |  | m4sim gate: B[,pages] runs every sequence as one session at base page B below R. Stdout must equal the run at B = 0. Needs GR8_CP, --prefill and no --s2. |
| `GR8_TEST_S2_PFNOFIX` | flag | `0` |  | m4sim negative control for GR8_S2_PFBASE: stages prefill index keys at logical pages from page 0, so the output must differ. |
| `GR8_TEST_S2_PFPOISON` | flag | `0` |  | m4sim gates GR8_S2_PFBASE and GR8_S2_XLIST: fills every cache with 0x3c bytes first, so a read outside the run changes the output. |
| `GR8_TEST_S2_POOL` | int | unset |  | m4sim --s2 without S2 below R: the pool's page count for variable runs. |
| `GR8_TEST_S2_RELOC` | flag | `0` |  | m4sim --s2: when one session is left, relocates its run to base page 0 instead of leaving it where it is. |
| `GR8_TEST_S2_RELOC_PAGES` | int | unset |  | m4sim --s2 with GR8_S2_RELOC: copies at most this many pages, to time the stall. |
| `GR8_TEST_S2_ROWHASH` | flag | `0` |  | m4sim --s2 probe: prints one line per target row with an FNV hash of its residual and its log-probability bits (rank 0). |
| `GR8_TEST_S2_RUNS` | text | unset |  | m4sim --s2: b:p,b:p,.. sets each slot's run (base page, pages) on every model. |
| `GR8_TEST_S2_SHRINK_AT` | int | unset |  | m4sim gate, with GR8_TEST_S2_XLIST, GR8_CP_R2 and GR8_RT_LOCAL: each sequence starts at GR8_CP_R and takes GR8_CP_R2 once its rows reach this position (past GR8_CP_R), as a session g6m4 shrinks. Stdout must equal the run at GR8_CP_R2 from the start. |
| `GR8_TEST_S2_SOLO1` | flag | `1` |  | m4sim --s2: a session left alone runs the single-session path. 0 keeps it on the S2 path. |
| `GR8_TEST_S2_XLIST` | ints | unset |  | m4sim gate: seed[,pages] runs every sequence as one session past R on a shuffled page list of the pool's two classes. Stdout must equal the run without it. |
| `GR8_TEST_SPARSE_MLA` | flag | `0` |  | flash_dsa_pf_test: compares the fused MLA with the current GEMM prefill instead of the older P1 math. |
| `GR8_TEST_TOKENIZER_DIR` | path | unset |  | The tokenizer directory the front end tests need, unset skips them. |

## Build and launcher

Read by the build or the launcher, never by the engine.

| variable | type | unset | values | what it does |
|---|---|---|---|---|
| `GR8_CACHE` | path | unset |  | docker/compose.yaml: a host directory for the cache volume, this box's part of the model. |
| `GR8_CLUSTER` | path | unset |  | The cluster.env file the launcher reads when the command line names none. |
| `GR8_CONTAINER` | flag | unset |  | The image sets 1, so the launcher knows it runs in a container. |
| `GR8_ENV` | path | unset |  | model-loader's compose project: the .env file of gr8's settings. |
| `GR8_FETCH_WORKERS` | int | unset |  | The launcher's Hugging Face fetch: parallel downloads, 64 when unset. |
| `GR8_HF_ENDPOINT` | text | unset |  | The hub the launcher fetches from, https://huggingface.co when unset. The tests point it at a fake one. |
| `GR8_HOME` | path | unset |  | docker/compose.yaml: a host directory for the home volume (logs and state). Not the bare-metal GR8_WORK_DIR. |
| `GR8_IMAGE` | text | unset |  | docker/compose.yaml: the gr8 image to run. |
| `GR8_KNOBS_LIB` | path | the crate's build, else bin/lib/libgr8knobs.so |  | The knob table library the launcher loads (tools/launch/knobs.py). |
| `GR8_MODEL` | text | unset |  | The model the launcher and the Helm chart serve: glm53 or glm53f. |
| `GR8_NR` | text | unset |  | The tensor-parallel widths scripts/build.sh builds, the same as GR8_WIDTHS. |
| `GR8_NVCC_CACHE` | flag | `1` | 0\|1 | 0 turns off the compile cache in front of nvcc and g++. |
| `GR8_NVCC_CACHE_DIR` | path | ~/.cache/gr8-nvcc |  | Where the compile cache keeps outputs. |
| `GR8_NVCC_CACHE_LOG` | path | unset |  | A file that gets one hit, miss or pass line per compile. |
| `GR8_PSEUDO_TOKENIZER` | path | unset |  | CI: a tokenizer directory that turns on the pseudo engine's smoke and API tests. |
| `GR8_REFUSAL_LINGER` | float | unset |  | Seconds the supervisor keeps a refused boot's status up, 15 when unset. |
| `GR8_TESTS` | text | `1` | 0\|1\|only | Whether scripts/build.sh builds the test binaries: 0 skips them, only builds them alone. |
| `GR8_WIDTHS` | text | unset |  | The tensor-parallel widths scripts/build.sh builds, "3 4" for example. |
| `GR8_WORK_DIR` | path | the repository |  | The working directory of a bare-metal install: bin/, out/, s2/, run/. |
| `GR8_XGRAMMAR` | path | unset |  | An XGrammar source tree to build request grammars into the engine. |
| `GR8_XGRAMMAR_LIB` | path | unset |  | A prebuilt XGrammar library in place of building one from GR8_XGRAMMAR. |
