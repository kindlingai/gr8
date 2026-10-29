// Every setting the engine, its tests and its build read from the environment. One entry:
//
//     /// What it does.
//     NAME: scope kind = "default", "values";
//     NAME: scope kind ? "what unset does", "values";
//
// NAME is the variable GR8_NAME, and its old name G6_NAME still works with a warning. NAME[n] is a family: NAME
// followed by any decimal number. scope: agree, local, test or outside (lib.rs Scope). kind: flag, int, float, text,
// path, ints or floats (lib.rs Kind). `= "default"` gives the value an unset variable has, and `knob::name()` returns
// it. `? "..."` has no default: the accessor returns an optional, and the text says what the code does when unset.
// values: "" any, "lo..hi" (an end may be left out) or "a|b|c". Optional after values: `env "NAME"` names the variable
// exactly (a system variable), `legacy "NAME"` gives an old name other than G6_NAME.
//
// After a change, `cargo run --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen` writes gr8_knobs.h and
// docs/ENV.md. Keep the entries sorted by name.

use crate::{Family, Kind, Knob, OptKnob, Scope, Spec};

macro_rules! rt {
    (flag) => { bool };
    (int) => { i64 };
    (float) => { f64 };
    (text) => { &'static str };
    (path) => { &'static str };
    (ints) => { &'static [i64] };
    (floats) => { &'static [f64] };
}

macro_rules! knob_type {
    ($k:ident = $d:literal) => { Knob<rt!($k)> };
    ($k:ident ? $u:literal) => { OptKnob<rt!($k)> };
    ($k:ident ? $u:literal [$n:ident]) => { Family<rt!($k)> };
}

macro_rules! opt {
    () => {
        None
    };
    ($v:literal) => {
        Some($v)
    };
}

macro_rules! family {
    () => {
        false
    };
    ($n:ident) => {
        true
    };
}

macro_rules! table {
    ($($(#[doc = $doc:literal])* $name:ident $([$n:ident])? : $scope:ident $kind:ident $(= $d:literal)? $(? $u:literal)?,
       $values:literal $(, env $env:literal)? $(, legacy $legacy:literal)?;)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[repr(usize)]
        enum Ix { $($name,)* }

        /// The table, sorted by name. A knob's index here is its id in the C API.
        pub static TABLE: &[Spec] = &[$(Spec {
            name: stringify!($name),
            family: family!($($n)?),
            scope: match stringify!($scope).as_bytes() {
                b"agree" => Scope::Agree,
                b"local" => Scope::Local,
                b"test" => Scope::Test,
                b"outside" => Scope::Outside,
                _ => panic!("scope"),
            },
            kind: match stringify!($kind).as_bytes() {
                b"flag" => Kind::Flag,
                b"int" => Kind::Int,
                b"float" => Kind::Float,
                b"text" => Kind::Text,
                b"path" => Kind::Path,
                b"ints" => Kind::Ints,
                b"floats" => Kind::Floats,
                _ => panic!("kind"),
            },
            default: opt!($($d)?),
            unset: concat!("" $(, $u)?),
            values: $values,
            env: opt!($($env)?),
            legacy: opt!($($legacy)?),
            doc: concat!($($doc, "\n",)*),
        },)*];

        $($(#[doc = $doc])* pub static $name: knob_type!($kind $(= $d)? $(? $u)? $([$n])?) =
            <knob_type!($kind $(= $d)? $(? $u)? $([$n])?)>::new(Ix::$name as usize);)*
    };
}

table! {
    /// attnk: the arms to run, as name or KEY=V,... specs joined by spaces.
    ATTNK_ARMS: test text ? "the built-in arms", "", legacy "ATTNK_ARMS";
    /// attnk: the base arm the others compare with, as KEY=V,... over the stock arm (every flag 0), or stock.
    ATTNK_BASE: test text ? "the serving arm SC2=4,TR=2,CTX2=4,CTX3=2,SEL2=16,IDX=16", "", legacy "ATTNK_BASE";
    /// attnk: 1 runs the round 10 arm set.
    ATTNK_E1: test flag = "0", "", legacy "ATTNK_E1";
    /// attnk: the rank's head count, g6nr::HMIN or g6nr::HMAX of the build (a TP=6 build: 11 or 10).
    ATTNK_H: test int ? "g6nr::HMAX, the larger head count of a rank in the build", "", legacy "ATTNK_H";
    /// attnk: the index key position (session 0) whose scale is NaN, -1 none.
    ATTNK_NANKEY: test int = "-1", "", legacy "ATTNK_NANKEY";
    /// attnk: scales the test queries.
    ATTNK_QSCALE: test float = "1", "", legacy "ATTNK_QSCALE";
    /// attnk: 2 runs two S2 sessions when the build and shape allow it.
    ATTNK_SESS: test int = "1", "1|2", legacy "ATTNK_SESS";
    /// attnk: 1 makes the test scores tie.
    ATTNK_TIES: test flag = "0", "", legacy "ATTNK_TIES";
    /// Decode attention context kernel: 1 or 2 = the head-group kernel with 32 or 64 gathered pairs per batch, 3 or 4 =
    /// a packed kernel with 16 or 32 and two batches in flight. Same bits, FP8 KV only.
    ATTN_CTX2: agree int = "0", "0..4";
    /// Decode attention context with one test per batch of slots: 1 or 2 = 16 or 32 pairs per batch over 256 dims per
    /// CTA, 3 or 4 = the same over 512. Overrides GR8_ATTN_CTX2. Same bits, FP8 KV only.
    ATTN_CTX3: agree int = "0", "0..4";
    /// With GR8_ATTN_CTX3, sets the heads per CTA of the context kernel. 0: 4 heads up to 4 rows, then 8.
    ATTN_CTX3_HG: agree int = "0", "0|2|4|8";
    /// With GR8_ATTN_CTX3, splits the replicated decode context into partitions of 64 or 128 key slots instead of 256.
    /// Deterministic but a different fp32 summation order, FP8 KV only. 0: 256 slots.
    ATTN_CTX3_SP: agree int = "0", "0|64|128";
    /// With GR8_ATTN_P11, forces one decode context tile of HG heads by D dims for every shape, written as HGxD.
    ATTN_CTX_TILE: agree text ? "unset (a per-shape tile table)", "2x128|4x128|8x128|16x128|4x256|8x256|16x256|4x64|8x64";
    /// With GR8_ATTN_FLIP, runs the flip context kernel over 256 dims per CTA (4 heads at one row, 8 above) with a
    /// dead-slot skip.
    ATTN_D256: agree flag = "0", "";
    /// Decode attention takes the head-inner flip kernels (decode-flip.cuh) for eligible calls at a context of at least
    /// GR8_ATTN_FLIP_MIN. Takes precedence over the head-batched kernels.
    ATTN_FLIP: agree flag = "0", "";
    /// With GR8_ATTN_FLIP, the smallest context in tokens at which a decode call takes the flip kernels.
    ATTN_FLIP_MIN: agree int = "2048", "";
    /// Runs a decode step's first attention kernels (KV append, index append and index query prep, or index reuse) as
    /// one launch when the step's rows fit one query tile. Same bits.
    ATTN_FRONT: agree flag = "0", "0|1";
    /// Decode attention scores and context load each key once for a group of heads. 0 = the per-head kernels and turns
    /// off GR8_ATTN_SC2, GR8_ATTN_CTX2 and GR8_ATTN_CTX3.
    ATTN_HEADS: agree flag = "1", "";
    /// Decode attention uses the P11 tile retune (per-shape context tiles) on the head-batched path, ahead of
    /// GR8_ATTN_SC2, GR8_ATTN_CTX2 and GR8_ATTN_CTX3.
    ATTN_P11: agree flag = "0", "";
    /// A full-index layer's decode attention in the buckets up to 2048 lists the keys in position order instead of
    /// running the indexer, and skips that layer's idx_q GEMV. Deterministic, a different fp32 order.
    ATTN_POSIDX: agree flag = "0", "0|1";
    /// Decode absorb stores q and rope lane-transposed and the scores kernel reads them with 16 B loads. Needs
    /// GR8_ATTN_SC2 2 or 4, not with GR8_AVP1. Same bits, FP8 KV only.
    ATTN_QT: agree flag = "0", "0|1";
    /// Decode attention scores in the head-group kernel take S = 1, 2, 4 or 8 key slots per warp with the query loaded
    /// once for all S. Same bits, FP8 KV only.
    ATTN_SC2: agree int = "0", "0|1|2|4|8";
    /// With GR8_ATTN_SC2, a head count that is not a multiple of 16 (TP=3, TP=6) also takes the scores kernel with a
    /// partial last head group. 0 = those layers keep the 16-head kernel. Same bits.
    ATTN_SC2_PART: agree flag = "1", "";
    /// With GR8_ATTN_SC2, 1 reduces the heads' sums by one transpose across the warp and 2 does the same reduction
    /// eagerly. S = 8 ignores it. Same bits.
    ATTN_SC2_TR: agree int = "0", "0|1|2";
    /// Decode index selection runs its final sort as a rank by count over 8, 16, 32 or 64 CTAs per row. Same bits.
    ATTN_SEL2: agree int = "0", "0|8|16|32|64";
    /// With GR8_ATTN_P11, the context length in tokens used to pick the decode context tile, in place of the captured
    /// call's context. 0: the call's context window.
    ATTN_TILE_L: agree int = "0", "";
    /// Decode absorb and value projection run as one launch for any row count, weights in registers, rows staged 8 at a
    /// time.
    AVP1: agree flag = "0", "";
    /// docker/compose.yaml: a host directory for the cache volume, this box's part of the model.
    CACHE: outside path ? "unset", "", env "GR8_CACHE";
    /// The cluster.env file the launcher reads when the command line names none.
    CLUSTER: outside path ? "unset", "";
    /// The image sets 1, so the launcher knows it runs in a container.
    CONTAINER: outside flag ? "unset", "", env "GR8_CONTAINER";
    /// Context-parallel KV: the first GR8_CP_R tokens are replicated and later pages are spread across ranks on a CP
    /// fabric of their own. Read only in builds with -DG6_CP=1.
    CP: agree flag ? "1 in a CP build", "";
    /// cpgroups: the arms to run, as KEY=V,... specs (SC2 TR CTX2 CTX3 HG) or stock, joined by spaces.
    CPG_ARMS: test text ? "the built-in arms", "", legacy "CPG_ARMS";
    /// cpgroups: puts the N most recent positions first in each row's index list, as a local window. 0: uniform random
    /// positions.
    CPG_LOCAL: test int = "0", "", legacy "CPG_LOCAL";
    /// Raises the row count the CP fabric is sized for. A value below the computed rows has no effect.
    CP_MMAX: agree int ? "the most rows of any step", "";
    /// Past R, the other ranks' head groups run as one attention call instead of one call per group, with partials
    /// published in the GR8_CP_S3=0 order. Byte-exact, TP=4 only, not with GR8_ATTN_FLIP.
    CP_ONECHAIN: agree flag = "0", "0|1";
    /// Negative control for the CP prefill staging gate in m4sim: skips the gather or the back half of the staging so
    /// the gate must fail.
    CP_PF_BREAK: test text ? "unset", "gather|back";
    /// The CP prefill page check uses the block-reduced HASH2 page hash. 0 = the older k_cp_pf_hash. Read only in
    /// builds with -DG6_CP_PF_HASH2=1.
    CP_PF_HASH2: agree flag = "1", "0|1";
    /// In m4sim, the CP prefill gather runs as g6m4's real fabric does it, each rank filling its own pool and sending
    /// tail pages to the others, instead of the loopback gather.
    CP_PF_NET: test flag = "0", "";
    /// Every KV page a rank stages to another rank in CP prefill is checked against the hash the sender computed. A
    /// mismatch sets the fabric error word.
    CP_PF_VERIFY: agree flag = "0", "";
    /// The first R tokens of KV are replicated on every rank and pages past R are spread across ranks. Under S2 it is
    /// also the largest request that shares decode steps. A multiple of 64.
    CP_R: agree int = "0", "";
    /// CP combine order: 1 = the head-group overlap order, 0 = the plain order. Same arithmetic.
    CP_S3: agree flag = "1", "";
    /// S2 only: a decode step past R carries at most this many rows and the CP fabric is sized for it instead of the
    /// widest S2 step. Sessions past R give the same tokens.
    CP_STEP_ROWS: agree int = "0", "0..4096";
    /// With GR8_CP_STEP_ROWS, a round past R takes only sessions past R, so a short session never runs the CP path
    /// because a long one is resident.
    CP_STEP_SEG: agree flag = "0", "0|1";
    /// The first decode context bucket boundary in tokens. Decode graphs are captured per bucket and GR8_RT_L must be a
    /// bucket boundary.
    DEC_BK: agree int = "2048", "";
    /// Prints per-stage decode step times from event nodes in the captured main-step graphs every N main steps, in the
    /// rank logs. 0: off.
    DEC_EVPROF: local int = "0", "";
    /// Flash decode: the fused mHC consume stages the peers' partials in shared memory before summing, at 1 or 4 rows.
    /// Same rank-order summation.
    DEC_MHC_STAGE: agree flag = "1", "";
    /// Measures decode stages with a synchronize at every mark and prints them per rank. Plain launches only, the
    /// engine stops with --graph.
    DEC_PROF: local flag = "0", "";
    /// One decode attention scratch serves every layer of the main model, which frees memory for the KV pool. Needs
    /// --pfv2.
    DEC_SHARED_SCRATCH: agree flag = "0", "";
    /// Captures the decode step as two graphs split at this layer, so the second submission overlaps the first one's
    /// GPU time. Needs --graph. 0: one graph.
    DEC_SPLIT: agree int = "0", "0..";
    /// The DFlash2 drafter's body GEMVs read INT4 g32 copies of their MXFP8 records, made at boot. Drafts only, +382 MB
    /// per rank. Needs --dflash-manifest.
    DFL_BODY_I4: agree flag = "0", "0|1";
    /// With --dflash, rank 0 writes every draft step's context features, drafts and committed tokens to this file, for
    /// engine/ref/dflash2_replay.py.
    DFL_DUMP: local path ? "unset", "";
    /// The DFlash2 drafter's lm_head GEMV reads INT4 g32 records, the same ones as GR8_DRAFT_HEAD_I4. Drafts only.
    /// Needs --dflash-manifest.
    DFL_HEAD_I4: agree flag = "0", "0|1";
    /// With --dflash, 0 gives the drafter no prompt context from the prefill, so it starts from the last prompt token.
    /// Flash --prefill with --dflash needs 0.
    DFL_PFCTX: agree flag = "1", "";
    /// With DFlash2 under S2, 1 lets a request whose prefix-reuse hit landed in the drafter's window keep the drafter,
    /// and 2 also gives the drafter to a request alone on MTP.
    DFL_REUSE: agree int ? "2 when serving DFlash2 with MTP, else 0", "0..2";
    /// With GR8_DFL_REUSE, the context rows the drafter must hold before it drafts. Until then the request runs MTP
    /// rounds that feed it.
    DFL_REUSE_MIN: agree int = "256", "0..2048";
    /// The DFlash2 selector walk scores each candidate as logit + a * correction instead of logit + correction. Drafts
    /// only. Needs --dflash-manifest and the selector.
    DFL_SEL_ALPHA: agree float = "1", "";
    /// Experimental: the MTP draft head runs in two stages, a 2-bit screen then the exact MXFP8 logits of its best 64
    /// rows. 2 also runs the MXFP8 GEMV and counts misses. Needs --mtp.
    DRAFT_HEAD2: agree int = "0", "0..2";
    /// The MTP draft's lm_head GEMV reads INT4 g32 records made at boot from the MXFP8 records, +134 MB per rank at
    /// TP4. Drafts only. Needs --mtp.
    DRAFT_HEAD_I4: agree flag = "0", "0|1";
    /// With GR8_DRAFT_HEAD_I4 or GR8_DFL_HEAD_I4, the INT4 head GEMV's k parts per tile. Its product with
    /// GR8_DRAFT_HEAD_TPC must be at most 32.
    DRAFT_HEAD_KS: agree int = "32", "2|4|8|16|32";
    /// With GR8_DRAFT_HEAD_I4 or GR8_DFL_HEAD_I4, the INT4 head GEMV's tiles per CTA. Its product with
    /// GR8_DRAFT_HEAD_KS must be at most 32.
    DRAFT_HEAD_TPC: agree int = "1", "1|2|4|8|16|32";
    /// Rank 0 writes the inputs of every MTP first draft pass to this file. Needs --mtp and no --serve.
    DRAFT_IN_DUMP: local path ? "unset", "";
    /// Every rank writes its vocab slice of the MTP first draft pass logits to this path plus .rank<R>. Needs --mtp and
    /// no --serve.
    DRAFT_LOGIT_DUMP: agree path ? "unset", "";
    /// S2 with DFlash2, a solo round runs the drafter step on its own CUDA stream under MTP's first form. 2 also puts
    /// drafting steps on a high-priority stream. Same bytes as serial.
    DRAFT_PAR: agree int ? "2 when serving DFlash2 with MTP, else 0", "0..2";
    /// Flash decode: tiles per CTA of the indexer's wq_b GEMV.
    DSA_IDXQ_TPC: agree int = "1", "";
    /// Flash decode: 1 = the o_proj trellis GEMV reads the attention output as rotated fragment tables, 0 = it rotates
    /// the attention output itself.
    DSA_OPROJ_ROT: agree flag ? "1 for steps of more than one row, else 0", "0|1";
    /// Dense trellis o_proj GEMV split as k1,s1[,k4,s4]: k parts and sub-parts at one row and at more rows.
    DSA_OPROJ_SPLIT: agree ints ? "4 k parts, 1 sub-part", "";
    /// Flash decode: q_b and the indexer's wq_b GEMV run as one 640-CTA kernel (k_dsa_qb_idxq) when their shapes match.
    /// 2 = the same with the indexer's CTAs first, 0 = two launches.
    DSA_QB_FUSE: agree int = "1", "0|1|2";
    /// Flash decode: the q_a norm kernel writes q_a rotated as the fragment tables the q_b trellis GEMV reads. 0 = the
    /// q_b GEMV rotates the q_a rows itself.
    DSA_QB_ROT: agree flag = "1", "";
    /// Dense trellis q_b GEMV split as k1,s1[,k4,s4]: k parts and sub-parts at one row and at more rows.
    DSA_QB_SPLIT: agree ints ? "QA/128 k parts (16 GLM, 12 Flash), 1 sub-part", "";
    /// Flash decode with GR8_GV_MULTI: a nonzero value launches the layer-input GEMVs in the order idx_k, q_a, idx_w,
    /// idx_g, kv_a instead of q_a, kv_a, idx_k, idx_w, idx_g.
    DSA_XIN_ORDER: agree flag = "0", "";
    /// Uses the exlr8 release's dense trellis matrices (q_b, o_proj, the dense MLP) where a layer has them. 0 = the
    /// int4 or MXFP8 records, which the layer must then have.
    DTR: agree flag = "1", "";
    /// Single-matrix dense trellis GEMVs whose group stride exceeds their block stride assign CTAs output block first
    /// instead of k part first. CTA order only.
    DTR_NBFAST: agree flag = "1", "";
    /// As GR8_DTR_NBFAST, for single-matrix dense trellis GEMVs whose block stride is at least their group stride.
    DTR_NBFAST_OUT: agree flag = "1", "";
    /// Prefill runs o_proj on an exact FP8 copy of its trellis tiles too, as q_b and the dense MLP do, about 25 MB more
    /// per layer at TP4. Same bits.
    DTR_PF_COPY: agree flag = "0", "";
    /// Flash prefill makes exact FP8 copies of the KDA q, k and v trellis matrices for the prefill GEMMs. 0 = no
    /// copies.
    DTR_PF_KDA_COPY: agree flag = "1", "";
    /// Flash prefill dense MLP: up reuses gate's rotated input instead of rotating with its own suh vector.
    DTR_PF_SHARE_SUH: agree flag = "0", "";
    /// model-loader's compose project: the .env file of gr8's settings.
    ENV: outside path ? "unset", "", env "GR8_ENV";
    /// Fabric TCP connect attempts, 100 ms apart. Raise it when ranks load at different speeds.
    FABRIC_CONNECT_TRIES: local int = "1200", "";
    /// The launcher's Hugging Face fetch: parallel downloads, 64 when unset.
    FETCH_WORKERS: outside int ? "unset", "", env "GR8_FETCH_WORKERS";
    /// GLM-5.3-Flash --serve with --prefill: aligned keeps session checkpoints so a later prompt with the same start is
    /// prefilled from there. terminal also checkpoints each request's end and is not cold-exact.
    FLASH_CACHE: agree text = "off", "off|0|aligned|terminal";
    /// Flash with --mtp: each rank writes its draft cache pages before the first live draft to this path plus
    /// .s<seq>.r<rank>.bin.
    FLASH_DRAFT_CACHE_DUMP: local path ? "unset", "";
    /// Flash prefill fills the MTP draft's caches only. 0 = runs the full draft step over the prompt rows, which uses
    /// the draft fabric.
    FLASH_DRAFT_KV: agree flag = "1", "";
    /// Flash: the DSA MLA and index caches sit in kvchunks regions under one chunk map per model, so S2 Resize events
    /// can take chunks online or offline on every rank.
    FLASH_KV_CHUNKS: agree flag = "0", "0|1";
    /// Ends the process at that ledger point of a chunk resize (flash_kv.h).
    FLASH_KV_CRASH_AT: local text ? "unset", "begun|synced|released";
    /// With GR8_FLASH_KV_LEND_MS, the chunks left online while the rest are lent. A request that fits them is admitted
    /// without a recall.
    FLASH_KV_IDLE_CHUNKS: agree int = "1", "1..";
    /// With GR8_FLASH_KV_CHUNKS=1, the box's host ledger that chunk resizes move bytes through. With
    /// GR8_FLASH_KV_LEND_MS the rank writes its lendable bytes to this path plus .gpu.
    FLASH_KV_LEDGER: local path ? "unset", "";
    /// With GR8_FLASH_KV_CHUNKS=1, once the serve is idle this many ms the head lends every chunk above
    /// GR8_FLASH_KV_IDLE_CHUNKS, and recalls them before a request that needs them. 0: never lend.
    FLASH_KV_LEND_MS: agree int = "0", "0..";
    /// Flash serve clears every attention cache page per request instead of only the pages the last request wrote.
    FLASH_RESET_ALL: local flag = "0", "";
    /// Flash serve stops if any cache page past the cleared ones is not zero. It reads every cache, so it is for tests.
    FLASH_RESET_CHECK: local flag = "0", "";
    /// Flash solo serving prices S2 rounds with spec_len.h's Flash cost table for this build's width instead of the GLM
    /// step costs.
    FLASH_S2_COST: agree flag = "0", "";
    /// With GR8_FLASH_CACHE and no --kv-store, the size in MiB of each rank's in-memory checkpoint store.
    FLASH_STATE_BUDGET_MIB: agree int = "8192", "";
    /// Flash prefill attention runs the P1 bodies instead of the GEMM bodies.
    FPF_ATTN_P1: agree flag = "0", "";
    /// Flash prefill dense MLP layers run the P1 bodies instead of the GEMM bodies.
    FPF_DENSE_P1: agree flag = "0", "";
    /// Flash prefill DSA layers run the P1 bodies instead of the GEMM bodies.
    FPF_DSA_P1: agree flag = "0", "";
    /// Flash prefill KDA q, k, v and o_proj run as the P1 projections instead of M-row GEMMs.
    FPF_KDA_P1: agree flag = "0", "";
    /// Flash prefill runs the P1 bodies everywhere, the reference the GEMM bodies are checked against.
    FPF_P1: agree flag = "0", "";
    /// Decode MoE K1: 1 = one fused launch, 0 = separate launches, 2 = fused with a KS1=2 WG=2 split at 2 and 3 rows,
    /// which rounds differently.
    FUSE: agree int = "1", "0|1|2";
    /// Rounds of the front end tokenizer fuzz test.
    FUZZ_ROUNDS: test int = "40", "";
    /// Prefill MXFP8 GEMM kernel: 1 = k_gemm_mx8 (v1), 2 = v2 with 128-row CTAs, 3 = v2 with 64-row CTAs, 0 = v2 with
    /// row blocks by grid size.
    GEMM: agree int = "0", "0..3";
    /// Steps of up to 8 rows run the GEMVs on the layer input (q_a, kv_a, the indexer's k and weights) as one launch. 0
    /// = one launch each. Same bits.
    GV_MULTI: agree flag = "1", "";
    /// The hub the launcher fetches from, https://huggingface.co when unset. The tests point it at a fake one.
    HF_ENDPOINT: outside text ? "unset", "", env "GR8_HF_ENDPOINT";
    /// docker/compose.yaml: a host directory for the home volume (logs and state). Not the bare-metal GR8_WORK_DIR.
    HOME: outside path ? "unset", "", env "GR8_HOME";
    /// Splits the indexer's replicated wq_b GEMV by output tiles across the ranks and all-gathers it on the split-GEMV
    /// fabric, with the same bits as the full GEMV.
    IDXQ_SPLIT: agree flag ? "g6m4: 1 when TP>1 on GLM, else 0; m4sim: 0", "0|1";
    /// m4sim writes rank 0's selected key ids of the first --layers layer to this file for each main step from
    /// GR8_LOGIT_DUMP_FROM on.
    IDX_DUMP: test path ? "unset", "";
    /// In S2 batched decode steps the per-row indexer reads each key tile once for up to this many rows of one session,
    /// with the same bits. 0 is off.
    IDX_GRP: agree int = "0", "0..64";
    /// The per-row decode index_scores loads its key tile with 16 B vector loads (page size 32 or more), with the same
    /// bytes. Read once before graph capture.
    IDX_VEC: agree flag = "0", "0|1";
    /// docker/compose.yaml: the gr8 image to run.
    IMAGE: outside text ? "unset", "", env "GR8_IMAGE";
    /// Builds the device weight image from cuMemCreate chunks mapped into one range, 1 = the recommended granularity, N
    /// = N MB chunks. Same bytes. Falls back to cudaMalloc and says why.
    IMG_VMM: local int = "0", "";
    /// 0 turns off the timed table that makes exlr8 v2 MoE layers pick fused or separate fc1 / fc2 launches by row
    /// count, so GR8_FUSE applies as given.
    K1_FUSE_POLICY: agree flag = "1", "0|1";
    /// Sets the decode MoE K1 split: a KS1,WG,NS2 triple for every row count, or a table of M:KS1xWG/NS2 entries that
    /// keeps the bytes. A bad split stops the bind.
    K1_SPLIT: agree text ? "the release split per row count", "";
    /// Sets the decode MoE K1 split KS1,WG,NS2 for 1-row steps, over GR8_K1_SPLIT. Another fc1 partial count can change
    /// the sums.
    K1_SPLIT_M1: agree ints ? "GR8_K1_SPLIT's KS1,WG,NS2 triple, else the release split (4,1,2)", "";
    /// Sets the decode MoE K1 split KS1,WG,NS2 for 2- and 3-row steps, over GR8_K1_SPLIT. Another fc1 partial count can
    /// change the sums.
    K1_SPLIT_M2: agree ints ? "GR8_K1_SPLIT's KS1,WG,NS2 triple, else the release split (1,2,8)", "";
    /// Sets the decode MoE K1 split KS1,WG,NS2 for steps of 4 or more rows, over GR8_K1_SPLIT. Another fc1 partial
    /// count can change the sums.
    K1_SPLIT_M4: agree ints ? "GR8_K1_SPLIT's KS1,WG,NS2 triple, else the release split (4,2,2)", "";
    /// At the start of each decode MoE fc1 launch, prefetches the routed slots' fc1 and fc2 weight translations into L2
    /// at this granule in KB. No output change.
    K1_TLBW: local int = "0", "0..1048576";
    /// KDA decode q / k / v read MXFP8 records built at bind with FP16 activations. 0 reads the BF16 weights. Not byte-
    /// exact, and the records are an extra copy outside the loader's accounting.
    KDA_MX: agree flag = "1", "0|1";
    /// Sets the k-split and sub-split of the KDA o_proj trellis GEMV as k1,s1,k4,s4 (1 row, then 2+ rows), in
    /// `kernels/dtr.cuh`.
    KDA_OPROJ_SPLIT: agree ints ? "ks 8 at 1 row, 4 at 2+ rows, sub 1", "";
    /// 1 runs the older v1 intra-chunk kernel in KDA prefill instead of k_kda_intra3.
    KDA_PF_IN1: agree flag = "0", "0|1";
    /// 1 runs the older row-loop prep kernel in KDA prefill instead of launch_prep_pf.
    KDA_PF_P1: agree flag = "0", "0|1";
    /// 1 runs the older v1 state kernel in KDA prefill instead of k_kda_state2.
    KDA_PF_ST1: agree flag = "0", "0|1";
    /// Sets the k-split of the KDA trellis q / k / v projection GEMV.
    KDA_PROJ_KS: agree int = "16", "8|16|32";
    /// Picks how the KDA trellis q / k / v projection rotates its input: 0 inside the GEMV, 1 and 2 in a separate pre-
    /// rotate launch that also carries the f_a / g_a / b GEMVs (1) or not (2).
    KDA_PROJ_MODE: agree int = "0", "0|1|2";
    /// 1 runs the Flash KDA prefill scan as the CH16 bf16-MMA scan (`kda/kda_g9.cuh`), a precision arm that is not
    /// byte-exact. 0 runs the base scan.
    KDA_SCAN: agree int = "0", "0|1", legacy "G9_KDA_SCAN";
    /// After each offline sequence, each rank writes the KDA S state and conv state of every layer to PATH.rank<R>.
    KDA_STATE_DUMP: local path ? "unset", "";
    /// With GR8_KDA_SCAN=1, the v slices per head of the G9 KDA scan kernel: 1, 2 or 4.
    KDA_VS: agree int = "2", "1|2|4", legacy "G9_KDA_VS";
    /// fp8 loads every layer's kv_b (UK / UV) as the release's block FP8 bytes and scales instead of BF16, with the
    /// same output bytes. Needs --v2-dense.
    KVB: agree text = "bf16", "bf16|fp8";
    /// flash_kv_chunks_test: the kvman-ledger binary.
    KVMAN_LEDGER: test path = "kvman-ledger", "", legacy "KVMAN_LEDGER";
    /// How this box reaches a KV node. A rank whose path to the node runs through another box sets up RDMA but cannot
    /// write, and auto does not fall back after setup.
    KVMAN_TRANSPORT: local text = "auto", "tcp|rdma|auto", legacy "KVMAN_TRANSPORT";
    /// kvman-sim: 1 prints the stash entries at the end.
    KVSIM_ENTRIES: test flag = "0", "", legacy "KVSIM_ENTRIES";
    /// For tests, rank R sleeps the given ms inside every Nth KV tier drain that copies anything, set as
    /// <rank>:<ms>[:<every>].
    KV_DRAIN_DELAY: local text ? "unset", "";
    /// m4sim prints per layer an FNV hash of the stored latent records of the prompt positions, read from every rank
    /// that holds them, after the prefill.
    KV_HASH: test flag = "0", "";
    /// m4sim with --reuse-prefix prints whether every rank's KV page blocks match rank 0's byte for byte.
    KV_RANKS_CMP: test flag = "0", "";
    /// With a group KV lease, idle makes the head offer its blocks to other groups only while no session is resident or
    /// queued, always offers on every report.
    KV_SHARE_WHEN: local text = "idle", "idle|always";
    /// m4sim with --reuse-prefix sends every rank's reused pages to this block store (mem:, ssd:PATH,
    /// remote:HOST:PORT), poisons them and reads them back.
    KV_TIER_CHECK: test text ? "unset", "";
    /// With GR8_KV_TIER_CHECK past GR8_CP_R, the copies of this part do not come back, so the check must see the
    /// poison.
    KV_TIER_CHECK_DROP: test int ? "none dropped", "0|1|2";
    /// Window mask for a paced L2 prefetch of the weights the decode step reads next, on a side stream while DRAM is
    /// idle (`kernels/l2pf.cuh`). Writes nothing, so output bytes are unchanged.
    L2PF: local int = "0", "0..31";
    /// Sets the pace of the GR8_L2PF prefetch in GB/s.
    L2PF_GBPS: local float = "175", "";
    /// Sets the most MiB the GR8_L2PF prefetch issues in one window.
    L2PF_MB: local float = "10", "";
    /// Sets how many MB of o_proj entries the GR8_L2PF prefetch takes per layer. 0 takes all.
    L2PF_OPMB: local float = "8", "";
    /// Bit mask of the GR8_L2PF prefetch targets: 1 x_in, 2 idx_q, 4 q_b, 8 UK, 16 UV, 32 o_proj, 64 router, 128 shared
    /// or dense fc1, 256 eh_proj, 512 head records.
    L2PF_T: local int = "0x3ff", "";
    /// Sets the GR8_L2PF budget per window in us at the pace, as four numbers: MLP all-reduce wait, gather waits,
    /// attention, attention all-reduce wait.
    L2PF_US: local floats = "30,30,100,30", "";
    /// Sets the most 8 KiB GR8_L2PF prefetch chunks issued per poll.
    L2PF_WAVE: local int = "8", "1..32";
    /// Aborts the process in the host ledger at that point, before or after the rename, so tests can check crash
    /// recovery. Only test and debug builds read it.
    LEDGER_CRASH_AT: test text ? "unset", "write|written";
    /// cpp loads a pack through the C++ path instead of the Rust loader.
    LOADER: local text ? "the Rust loader", "cpp";
    /// Sets the chunk size in MB that the Rust loader reads and the weight check sums.
    LOAD_CHUNK_MB: local int = "32", "1..";
    /// Picks how the Rust loader reads weight files: read or mmap. Other values fail the load.
    LOAD_MODE: local text = "read", "read|mmap";
    /// Sets the number of Rust loader worker threads.
    LOAD_THREADS: local int = "8", "1..";
    /// Offline runs write each main step's head logits (each rank's vocab slice) to PATH.rank<R>, with a header. Not
    /// with --serve.
    LOGIT_DUMP: local path ? "unset", "";
    /// Main-step dumps (GR8_LOGIT_DUMP, GR8_RESID_DUMP) start at this step.
    LOGIT_DUMP_FROM: local int = "0", "";
    /// GiB of MemAvailable the boot must keep after the weights and each layer of the KV pool. Below it the engine
    /// stops with a message instead of meeting the OOM killer.
    MEM_MIN: local float = "8", "0..";
    /// Quality test: rounds mHC residual stream writes to BF16, 1 = after both blocks, 2 = only after the MLP block.
    MHC_BF16: agree int = "0", "0|1|2";
    /// Sets the decode mHC consume form for every row count unless GR8_MHC_FUSE_M1 / _M4 set it: 0 a separate peer-wait
    /// kernel, 1 the consume waits itself, 2 the fused consume.
    MHC_FUSE: agree int = "2", "0|1|2";
    /// Sets the decode mHC consume form for 1-row steps, as GR8_MHC_FUSE.
    MHC_FUSE_M1: agree int ? "MHC_FUSE, else 2", "0|1|2";
    /// Sets the decode mHC consume form for every step of 2 or more rows, as GR8_MHC_FUSE.
    MHC_FUSE_M4: agree int ? "MHC_FUSE, else 2", "0|1|2";
    /// The model the launcher and the Helm chart serve: glm53 or glm53f.
    MODEL: outside text ? "unset", "", env "GR8_MODEL";
    /// Keeps only this many routed experts per token and renormalizes their weights. A model change for screens.
    MOE_TOPK: agree int = "8", "1..8";
    /// A solo session's MTP rounds verify only the drafts the MTP head is confident in: 1 by the GR8_MTP_KSTOP_TAU
    /// thresholds, 2 by the drafter choice's MTP estimates.
    MTP_KSTOP: agree int ? "2 with --serve, DFlash2 and --mtp 2+, else 0", "0|1|2";
    /// In shared S2 rounds each session verifies its own count of the --mtp drafts, from the MTP head's cumulative
    /// draft probability against the GR8_MTP_KSTOPB_TAU thresholds.
    MTP_KSTOPB: agree flag = "0", "0|1";
    /// Measurement: with GR8_MTP_KSTOPB, every Nth round's shared sessions verify a hashed draft count instead. 0 is
    /// off.
    MTP_KSTOPB_EXPLORE: agree int = "0", "";
    /// Sets the fewest drafts a session verifies under GR8_MTP_KSTOPB.
    MTP_KSTOPB_KMIN: agree int = "0", "0|1";
    /// Sets the GR8_MTP_KSTOPB thresholds t1,t2,... (up to 7) on each session's cumulative draft probability, at every
    /// session count. Past the last entry there is no stop.
    MTP_KSTOPB_TAU: agree floats = "0.6,0.88,0.95", "";
    /// Sets the GR8_MTP_KSTOPB thresholds t1,t2,... (up to 7) for rounds of n sessions, n 1..8.
    MTP_KSTOPB_TAU_S[n]: agree floats ? "MTP_KSTOPB_TAU", "";
    /// Sets the shared-round main widths captured as graphs besides the multiples of --mtp + 1. A width outside them
    /// takes the next most likely drafts.
    MTP_KSTOPB_WIDTHS: agree ints ? "every width the rule reaches", "";
    /// Sets the most drafts a solo MTP round chains under GR8_MTP_KSTOP. Without DFlash2 it must not exceed --mtp.
    MTP_KSTOP_KMAX: agree int ? "--mtp", "1..3";
    /// With GR8_MTP_KSTOP=2, sets the prior acceptance of a third MTP draft in the drafter choice's estimates.
    MTP_KSTOP_SEED3: agree float = "0.55", "0..1";
    /// With GR8_MTP_KSTOP=1, sets the thresholds t1,t2,t3: a solo round verifies draft j while the product of the draft
    /// probabilities up to j is at least t_j.
    MTP_KSTOP_TAU: agree floats = "0,0.2,0.5", "";
    /// With GR8_L2PF and --mtp, runs the paced L2 prefetch in MTP draft steps: 1 the draft step's windows, 2 the head
    /// waits prefetch the next draft step's front, 3 both. Same output bytes.
    MTP_L2PF: local int = "0", "0..3";
    /// Sets the budget in us at the pace of each GR8_MTP_L2PF head-wait prefetch window.
    MTP_L2PF_US: local float = "60", "0..1000";
    /// A first-form MTP draft step routes only each session's last row through the MTP layer's routed experts. Caches,
    /// drafts and log-probabilities keep their bytes.
    MTP_LASTROW: agree flag = "0", "0|1";
    /// m4sim prints a check line per draft step with the last row's token, log-probability and hashes of its hidden
    /// row, residual and draft-layer caches.
    MTP_LASTROW_CHECK: test flag = "0", "";
    /// Prints one # kstop line per solo S2 round with the drafter, verified and accepted counts and the drafts' log-
    /// probabilities: 1 on rank 0, 2 on every rank.
    MTP_LPLOG: local int = "0", "0|1|2";
    /// MTP draft steps split eh_proj and the draft layer's replicated input GEMVs by output tiles across the ranks,
    /// gathered on a draft fabric (port + 72). Same bits.
    MTP_SPLIT: agree flag ? "g6m4: 1 when TP>1 on GLM with --mtp, else 0; m4sim: 0", "0|1";
    /// A solo DFlash2 session also drafts from this n-gram table file and verifies its chain when it promises more
    /// tokens per verify ms. Needs --serve.
    NGRAM: agree path ? "unset", "";
    /// Sets the longest GR8_NGRAM chain a round verifies.
    NGRAM_KMAX: agree int ? "--dflash-k", "1..";
    /// Sets the shortest GR8_NGRAM chain worth a round.
    NGRAM_KMIN: agree int = "2", "1..";
    /// Rank 0 prints one line per round that takes a GR8_NGRAM chain.
    NGRAM_LOG: local flag = "0", "";
    /// Sets how many more tokens per verify ms a GR8_NGRAM chain must promise than the drafter's drafts.
    NGRAM_MARGIN: agree float = "0.1", "0..10";
    /// Keeps the MoE fc2 weights in the old order instead of the wide order. Outputs are bit-identical.
    NOWIDE2: agree flag = "0", "";
    /// The tensor-parallel widths scripts/build.sh builds, the same as GR8_WIDTHS.
    NR: outside text ? "unset", "";
    /// 0 turns off the compile cache in front of nvcc and g++.
    NVCC_CACHE: outside flag = "1", "0|1";
    /// Where the compile cache keeps outputs.
    NVCC_CACHE_DIR: outside path ? "~/.cache/gr8-nvcc", "";
    /// A file that gets one hit, miss or pass line per compile.
    NVCC_CACHE_LOG: outside path ? "unset", "";
    /// m4sim multiplies every rank's residual after layer 0's attention by 1 + eps * u, with u from a hash of row and
    /// element, to measure the noise floor.
    PERTURB: test float = "0", "";
    /// On the first prefill call of layer position 0, each rank writes diagnostic buffers to DIR/<name>.r<rank>.bin.
    /// Synchronizes.
    PF_ADUMP: local path ? "unset", "";
    /// With GR8_PF_PROF, the Flash DSA prefill attention adds its own profile marks.
    PF_AMARK: local flag = "0", "0|1";
    /// 0 turns off the prefill fc1 A-operand share, which quantizes the fc1 A operand once into records the fc1 CTAs
    /// stage. Byte-exact. Needs the -DPF_AOP build.
    PF_AOP: agree flag ? "1 when built with -DPF_AOP", "0|1";
    /// Sets the fewest rows a prefill fc1 pass needs to use the A-operand share. Smaller passes run the stock fc1, with
    /// the same bytes.
    PF_AOP_MMIN: agree int = "3072", "";
    /// Prefill attention runs in chunks of this many rows inside a sub-batch. The rest of the layer runs at the sub-
    /// batch size.
    PF_ATTN_ROWS: agree int ? "the whole sub-batch (the GLM launcher sets 1024)", "";
    /// The Flash BF16 dense MLP fallback in prefill runs plain kernels instead of cuBLAS.
    PF_BF16_EXACT: agree flag = "0", "0|1";
    /// Runs the prefill cuBLAS calls in chunks of this many rows, so a row's bits depend on its chunk and not on the
    /// sub-batch size.
    PF_BLAS_ROWS: agree int ? "the whole sub-batch (the GLM launcher sets 1024)", "16..";
    /// Sets the context up to which prefill attention takes the dense path. Past it the sparse, indexer-selected path
    /// runs. 0 means always sparse.
    PF_DENSE: agree int = "8192", "0..2147483616";
    /// m4sim with --mtp writes the MTP layer's KV and indexer caches after the draft prefill to
    /// DIR/r<rank>.s<seq>.{kv,idx}.
    PF_DRAFTDUMP: test path ? "unset", "";
    /// The Flash MTP draft cache fill runs over the lane's rows in 256-row passes instead of one draft step per MX
    /// rows, with the same bits. 2 also counts differing words against the old path.
    PF_DRAFT_FULL: agree int = "1", "0|1|2";
    /// The MTP draft prefill borrows the target prefill's tensor-core attention scratch instead of its own, about 2 GB
    /// a rank. 1 requires it, 0 gives the draft its own.
    PF_DRAFT_SHARE: agree int ? "on when the target has tensor-core prefill states", "0|1";
    /// Flash DSA prefill input projections run as FP16 cuBLAS GEMMs on weights decoded once at bind, instead of the
    /// MXFP8 GV kernel. Near-byte, not byte-equal.
    PF_DSA_F16GEMM: agree flag = "0", "0|1";
    /// Flash DSA prefill runs its GV projections as one launch over all rows. 0 restores the per-32-row launches. Byte-
    /// equal.
    PF_DSA_FULL_GV: agree flag = "1", "0|1";
    /// Flash DSA prefill q_a and idx_q run on a cublasLt algo with a bigger tile where the default is one k pass, with
    /// the same bits. 0 keeps cublasGemmEx.
    PF_DSA_LT: agree flag = "1", "0|1";
    /// Flash DSA prefill q_b and o_proj trellis projections run as the decode GEMVs per MX rows instead of the prefill
    /// trellis GEMM.
    PF_DSA_PROJ_P1: agree flag = "0", "0|1";
    /// Flash prefill runs the dense trellis matrices from FP8 tile copies made once at load (`dtr_find.h`) instead of
    /// the fp16 trellis-tile GEMM.
    PF_DTR8: agree flag = "0", "0|1";
    /// GLM prefill attention allocates and records CUDA timing events per query tile (`attn/prefill.h`). Other values
    /// make the attention state fail to create.
    PF_EVENTS: local flag = "0", "0|1";
    /// 0 runs the stock prefill fc1 instead of the fat-CTA `k_pf_fc1f`, where adjacent expert blocks share one
    /// quantized A record (byte-exact).
    PF_F12: agree flag ? "1 when built with -DPF_F12", "0|1";
    /// 0 makes the Flash fp16 MoE fc2 run one output Hadamard per block instead of one per 128-column group.
    PF_FC2ONE: agree flag = "1", "0|1";
    /// 0 runs the Flash prefill fc2 as `k_pf_fc2g` instead of `k_pf_fc2v`, which stages A and weights in shared memory.
    PF_FC2V2: agree flag = "1", "0|1";
    /// 0 keeps big FP32-output prefill GEMMs (the row-owner QO o_proj) on the guarded `k_gemm_mx8b` instead of the
    /// unguarded `k_gemm_mx8f`. Same bits either way.
    PF_GEMMF32: agree flag = "1", "0|1";
    /// Sets how many row blocks `k_gemm_mx8f` full-M launches group in their raster order, and 0 or 1 gives a plain
    /// raster. Launch order only, same bits.
    PF_GEMMG: agree int = "8", "";
    /// Flash prefill prints a hash of each sub-batch's residual rows (stream sync, copy to host, FNV) on its `prefill
    /// rows` line.
    PF_HASH: local flag = "0", "0|1";
    /// The per-sub-batch residual hash is a splitmix64 sum computed on the GPU instead of an FNV of the rows on the
    /// host, which gives a different value.
    PF_HASH_GPU: agree flag = "0", "0|1";
    /// m4sim test hook `rank,row`: adds one ulp to the first word of that lane-0 row on that rank before the cross-rank
    /// hash check, which must then report the rank differs.
    PF_HASH_POKE: test ints ? "unset", "";
    /// With GR8_LOGIT_DUMP set, Flash prefill also runs the head on the prompt rows and writes their logits to the
    /// dump.
    PF_HEAD: local flag = "0", "0|1";
    /// 0 makes the Flash MoE fc1 rotate x per CTA instead of reading rows rotated and scaled once per token by
    /// `k_pf_hx`.
    PF_HX: agree flag = "1", "0|1";
    /// 0 runs `k_pf_hx` and `k_pf_hxq` as two passes on GR8_PF_Q8 layers instead of the fused `k_pf_hxqf`.
    PF_HXF: agree flag = "1", "0|1";
    /// Rank 0 writes the ordered index IDs and positions of the first, middle and last full-index layers of each
    /// prefill sub-batch to this directory.
    PF_IDDUMP: local path ? "unset", "";
    /// Sets the rows per CTA of the prefill index-score kernel, rounded down to a multiple of 8 (below 8 means 8). Same
    /// bits.
    PF_ISGRT_RG: agree int = "8", "8..";
    /// 0 makes the G9 KDA prefill scan write and re-read the G16 / Z16 planes instead of building those tiles in its K1
    /// and output kernels (`kda/kda_g9.cuh`). Same bits.
    PF_KDA_GZF: agree flag = "1", "0|1";
    /// Selects how Flash prefill runs the KDA f_a / g_a / b projections: 0 = the per-MX-row host loop, 1 = full-M
    /// 32-row tiles, 2 = full-M 64-row tiles.
    PF_KDA_TAIL: agree int = "1", "0|1|2";
    /// Flash prefill also runs the KDA tail projections on the host-loop path, compares them byte for byte with the
    /// tiled result and stops on a difference.
    PF_KDA_TAIL_CHECK: local flag = "0", "0|1";
    /// 0 turns off the warp-specialized GEMM for the Flash KDA prefill projections (`kda/proj_in_prefill.cuh`).
    PF_KDA_WS: agree flag = "1", "0|1";
    /// Selects the Flash prefill kpool score kernel: 0 = `kp_scores_pf`, 1 = tensor-core with one stage, 2 = tensor-
    /// core with two stages, 3 = the streaming tensor-core form.
    PF_KP_TC: agree int = "2", "0|1|2|3";
    /// Sets how far Flash prefill fuses the mHC update with the fabric wait: 0 = separate passes, 1 = the mHC kernel
    /// reads the partials, 2 = it also waits for the payload.
    PF_MHC_FUSE: agree int = "2", "0|1|2";
    /// 0 turns off the packed mHC fn weights for `k_mhc_out_inv_prepare_pk` in Flash prefill.
    PF_MHC_PK: agree flag = "1", "0|1";
    /// 0 runs `k_mhc_out_inv_prepare_pk` instead of the row-blocked `k_mhc_out_inv_prepare_rr` when the peer partial is
    /// staged in Flash prefill.
    PF_MHC_RR: agree flag = "1", "0|1";
    /// 0 stops Flash prefill at TP2 from copying the peer's partial into device memory with the copy engine before the
    /// mHC update.
    PF_MHC_STAGE: agree flag = "1", "0|1";
    /// 0 makes Flash prefill attention use the older MLA projection kernels instead of `fa_pf::mla_proj2`.
    PF_MLA_PROJ2: agree flag = "1", "0|1";
    /// 1 runs the Flash prefill MoE on MXFP8 activations instead of the fp16 passes, and turns on the shared expert's
    /// FP8 copy unless GR8_PF_SHFP8 says otherwise.
    PF_MOE8: agree flag = "0", "0|1";
    /// Any value other than 0 stops Flash prefill from reordering the tr3 fc1 / fc2 records into the wide load order at
    /// bind time.
    PF_NOWIDE: agree flag = "0", "0|1";
    /// Sets row-owner prefill: 1 = owners compute the next layer's projections, 2 = also the indexer, 3 = owners run
    /// all heads of their rows (QO pack), 4 = TP4 pair-owner QO.
    PF_OWN: agree int = "0", "0|1|2|3|4";
    /// With GR8_PF_OWN=3, interleaves the QO owner rows in 16-row blocks instead of contiguous row ranges.
    PF_OWNPERM: agree flag = "0", "0|1";
    /// Each rank writes the IDs, attention probabilities and index scores of selected prefill rows and layers to this
    /// directory (`pf_pdump`).
    PF_PDUMP: local path ? "unset", "";
    /// With GR8_PF_PDUMP, dumps the rows whose position is a multiple of this number.
    PF_PDUMP_EVERY: local int = "32", "";
    /// With GR8_PF_PDUMP, the comma list of model layers to dump, plus their index-score source layers.
    PF_PDUMP_LAYERS: local ints = "0,3,20,34,55,74", "";
    /// A non-empty value stops the boot when GR8_FPF_ATTN_P1=1, because pool capture needs the prefill attention path.
    PF_POOL_CAPTURE: local path ? "unset", "";
    /// 1 runs the Flash prefill pool-ID exchange on its own fabric (TP2, not ring) and stops the boot if it cannot, 0
    /// turns it off.
    PF_POOL_OWNER: agree int ? "on for TP2 Flash prefill (not ring, no P1 flags), else off", "0|1";
    /// 1 creates the per-lane Flash DSA and MLP prefill states at boot instead of inside the first chunk, and any other
    /// value turns it off.
    PF_PREWARM: local flag ? "1 at TP2, else 0", "0|1";
    /// Prefill synchronizes after each stage and rank 0 prints a per-stage time profile (ms) to stderr.
    PF_PROF: local flag = "0", "";
    /// Sets the CTA count of the Flash prefill partial publish copy (`k_put_publish_n`) outside the SP path. Copy only,
    /// same bytes.
    PF_PUT_CTA: agree int = "96", "";
    /// 1 runs the Flash routed fc1 on MXFP8 activations quantized once per token and layer instead of fp16 (not byte-
    /// equal to the fp16 path).
    PF_Q8: agree flag = "0", "0|1";
    /// Selects the QO attention kernel for GR8_PF_OWN=3 and 4: b, v1 or r1. Other values make the QO state fail to
    /// create.
    PF_QO: agree text = "b", "b|v1|r1";
    /// With GR8_PF_OWN=4 and GR8_PF_QP_ROLL, sets the QO key-table size in positions, and a request whose prompt plus
    /// generation is longer runs GR8_PF_OWN=2.
    PF_QP_CTX: agree int ? "the KV cap", "1..";
    /// With GR8_PF_OWN=4: 1 sends the counterpart's partial before the own rows' o_proj, 2 also splits the QO call in
    /// two.
    PF_QP_EARLY: agree int = "0", "0|1|2";
    /// With GR8_PF_OWN=4, the comma list of model layers and ranges that run pair-owner attention, while the others run
    /// GR8_PF_OWN=2.
    PF_QP_LAYERS: agree text ? "all", "";
    /// With GR8_PF_OWN=4, a prompt shorter than this many tokens runs GR8_PF_OWN=2, and 0 sends every request that fits
    /// GR8_PF_QP_CTX through OWN=4.
    PF_QP_MIN: agree int = "768", "";
    /// With GR8_PF_OWN=4, builds the pair weights per layer in this many rolling slots from the rank pack and the
    /// partner's half over RDMA. 0 uses a resident --qp-pack.
    PF_QP_ROLL: agree int = "0", "0|2|3|4";
    /// Measurement: 1 builds the rolling pair-weight slots on the prefill lane stream instead of the stage stream, so
    /// the build time is fully exposed.
    PF_QP_ROLL_SER: local flag = "0", "0|1";
    /// With GR8_PF_OWN=4, warms the pair cuBLAS and MXFP8 GEMM shapes at load so the first request does not pay for
    /// them.
    PF_QP_WARM: local flag = "0", "0|1";
    /// With GR8_PF_OWN=4 and GR8_PF_QP_ROLL, sets the entries in the RDMA ring of pair-weight half images
    /// (`qp_wire.h`).
    PF_QP_WIRE_E: agree int ? "1 entry; set, it needs GR8_PF_QP_ROLL", "1..4";
    /// Each rank appends per-expert prefill route counts per MoE layer and sub-batch to `routes.r<rank>.jsonl` in this
    /// directory.
    PF_RCDUMP: local path ? "unset", "";
    /// pool_conc_replay_test: 1 logs every replayed call.
    PF_REPLAY_VERBOSE: test flag = "0", "", legacy "PF_REPLAY_VERBOSE";
    /// m4sim writes each prefill sub-batch's FP32 residual rows to `resid.s<seq>.p<pos>.f32` in this directory.
    PF_RESDUMP: test path ? "unset", "";
    /// 1 runs the Flash prefill router logits on `k_linear_mr` instead of the tiled `k_router_tile` (0 ULP apart).
    PF_ROUTER1: agree flag = "0", "0|1";
    /// 1 splits the Flash prefill MoE router by rows across the ranks and exchanges the route tables on a dedicated
    /// fabric per lane (`flash_route_split.cuh`).
    PF_ROUTER_SPLIT: agree flag = "0", "0|1";
    /// 1 runs GLM prefill with reduce-scatter plus all-gather (five collectives) instead of all-reduce.
    PF_RS: agree flag = "0", "0|1";
    /// With GR8_PF_RS=1, o_proj, the dense down GEMM and the MoE final sum write straight into the reduce-scatter send
    /// slot, so the put only rings the doorbell.
    PF_RSDIRECT: agree flag = "0", "0|1";
    /// Measurement: with GR8_PF_RS=1, syncs the stream after each put and prints an FNV hash of every owner slice of
    /// the send slot.
    PF_RSHASH: local flag = "0", "0|1";
    /// Prefill index tiles of at least 32 rows at a context of at least this many tokens pick their top keys with SELX
    /// instead of SEL32, with the same IDs in the same order.
    PF_SELX_CTX: agree int = "53248", "32769..1048576";
    /// 1 makes an FP8 copy of the Flash trellis shared expert for the prefill GEMMs, which quantizes its activations,
    /// and 0 keeps the fp16 trellis pass that matches decode.
    PF_SHFP8: agree flag ? "the PF_MOE8 value", "0|1";
    /// 1 runs the Flash trellis shared expert inside the routed MoE pass as extra routes instead of its own pass. Same
    /// bytes.
    PF_SHR: agree flag = "0", "0|1";
    /// 1 runs Flash prefill mHC sequence-parallel: partials go reduce-scatter, each owner runs mHC on its rows, then
    /// all-gathers the normalized rows.
    PF_SP: agree flag = "0", "0|1";
    /// 1 runs Flash prefill attention as the fused sparse MLA kernel when a rank has at most 32 heads.
    PF_SPARSE_MLA: agree flag = "0", "0|1";
    /// With GR8_PF_SPARSE_MLA=1, 1 selects the `sparse_mla_pf_512` kernel and makes the V2 kernel the default.
    PF_SPARSE_MLA_512: agree flag = "0", "0|1";
    /// With the V2 sparse MLA kernel and at most 16 heads a rank, 1 runs its 16-head form.
    PF_SPARSE_MLA_H16C: agree flag = "0", "0|1";
    /// With GR8_PF_SPARSE_MLA=1, 1 runs the V2 sparse MLA kernel and 0 the 512 or base kernel.
    PF_SPARSE_MLA_V2: agree flag ? "the PF_SPARSE_MLA_512 value", "0|1";
    /// 0 stops the GR8_PF_SP owner pass from writing its rows straight into the all-gather send slot, so the publish
    /// copies them. Byte-equal.
    PF_SP_AGD: agree flag = "1", "0|1";
    /// 0 moves GR8_PF_SP all-gathers back to copy-engine copies on the lane's copy stream instead of one SM pass on the
    /// compute stream that also writes the FP16 copy.
    PF_SP_AGK: agree flag = "1", "0|1";
    /// 0 limits the GR8_PF_SP_AGK compute-stream all-gather to lane B instead of both prefill lanes.
    PF_SP_AGKA: agree flag = "1", "0|1";
    /// 0 stops the GR8_PF_SP attention o_proj GEMM from writing each row into its owner's slice of the send slot. Byte-
    /// equal.
    PF_SP_ATTD: agree flag = "1", "0|1";
    /// 0 stops the GR8_PF_SP MoE final sum from writing each row into its owner's slice of the send slot. Byte-equal.
    PF_SP_FIND: agree flag = "1", "0|1";
    /// Sets the CTA count of the GR8_PF_SP send copies (`k_put_rs`, `k_put_publish_n`). Copy only, same bytes.
    PF_SP_PUT_CTA: agree int = "96", "1..1024";
    /// With GR8_PF_SP and GR8_PF_ROUTER_SPLIT, an MoE layer's MLP input travels as FP16 rows instead of FP32 rows,
    /// halving that all-gather. Byte-equal.
    PF_SP_X16: agree flag = "0", "0|1";
    /// 0 stops the GR8_PF_SP owner pass and gather from writing the next attention's fragment tables directly. Same
    /// tables either way.
    PF_SP_XFRAG: agree flag = "1", "0|1";
    /// 1 waits for and copies the peers' partials on a per-lane side stream under the other lane's block, at any rank
    /// count, and stops the boot if the config cannot run it.
    PF_STAGE_ASYNC: agree flag ? "1 at TP2 when the config allows, else 0", "0|1";
    /// Lane B's prefill stream first sleeps this many microseconds, so its first work starts after lane A's. Writes
    /// nothing.
    PF_STAGGER: local int = "0", "";
    /// 2 runs the two GLM prefill lanes on two CUDA streams, and any other value keeps one stream.
    PF_STREAMS: agree int = "1", "1|2";
    /// 1 splits large prefill payload writes over two rails (needs --pf-ips2 or --ips2, or a three-rank ring that uses
    /// the other PCIe root's device).
    PF_STRIPE: agree flag ? "1 with --pf-ips2 or on a three-rank ring, else 0", "0|1";
    /// With GR8_PF_STRIPE=1, the smallest prefill payload write in bytes that goes out split over the two rails.
    PF_STRIPE_MIN: local int = "1048576", "8192..";
    /// Sets the prefill attention query tile in rows. Other values make the attention state fail to create.
    PF_TILE: agree int = "256", "64|128|256";
    /// m4sim --serve prints per-request wall-clock timing laps on stderr, from the request file's mtime to the first
    /// token.
    PF_TIMES: test flag = "0", "0|1";
    /// Sets the row blocks per CTA of large prefill GEMMs on trellis tiles (`k_gemm_tr`): 8 or 16. Small launches take
    /// 4.
    PF_TR_BM: agree int = "16", "8|16";
    /// 0 runs the Flash routed fc1 on `k_pf_fc1` instead of the weight-major `k_pf_fc1ws` for passes of at least
    /// GR8_PF_WM_MIN rows. Byte-equal.
    PF_WM: agree flag = "1", "0|1";
    /// 1 runs the Flash prefill fc2 weight-major (`k_pf_fc2w`) instead of `k_pf_fc2v` when the fc1 runs weight-major.
    /// Byte-equal.
    PF_WM2: agree flag = "0", "0|1";
    /// Sets the fewest rows a Flash MoE prefill pass needs before the routed fc1 runs weight-major.
    PF_WM_MIN: agree int = "6144", "";
    /// 1 makes the Flash mHC kernel write the next attention's input as f16 and bf16 fragment tables instead of FP32
    /// rows when that attention reads only fragments. Same bits.
    PF_XFRAG: agree flag = "0", "0|1";
    /// Measurement: Flash prefill stamps GPU timestamps around each layer's fabric wait and copy for a trace.
    PF_XTRACE: local flag = "0", "0|1";
    /// Prints one stderr line per S2 pool claim with its first page, page count and the retained runs it evicted.
    POOL_LOG: local flag = "0", "0|1";
    /// Moves every thread that is not a fabric proxy off this rank's proxy CPUs before the first fabric barrier, so no
    /// worker thread preempts a progress thread. 0 turns it off.
    PROGRESS_ISOLATE: local flag = "1", "0|1";
    /// CI: a tokenizer directory that turns on the pseudo engine's smoke and API tests.
    PSEUDO_TOKENIZER: outside path ? "unset", "", env "GR8_PSEUDO_TOKENIZER";
    /// Splits the decode layer-input GEMVs across the ranks by 16-output tiles and all-gathers them on their own
    /// fabric. 1 = q_a, kv_a and the indexer's k and weights, 2 = q_a only. Same bits.
    QA_SPLIT: agree int ? "g6m4: 2 when TP>1 on GLM, else 0; m4sim: 0", "0|1|2";
    /// Sets the CPU core of the replicated-GEMV gather fabric's proxy thread (GR8_QA_SPLIT, GR8_IDXQ_SPLIT,
    /// GR8_MTP_SPLIT). -1 leaves it unpinned.
    QA_SPLIT_CPU: local int ? "--proxy-cpu minus 9, or -1 when --proxy-cpu is 8 or less", "";
    /// Sets the output tiles per CTA of the GEMVs that GR8_QA_SPLIT, GR8_IDXQ_SPLIT and GR8_MTP_SPLIT narrow. It only
    /// lowers a GEMV's own count and keeps the output bits (rank.cuh qs_narrow).
    QA_SPLIT_TPC: agree int = "1", "";
    /// With GR8_PF_OWN=3, the first boot writes each layer's full-head q_b, o_proj, UK and UV tensors to
    /// DIR/<layer>.qo.bin and later boots read them instead of decoding them. Same bits.
    QO_CACHE: local path ? "unset", "";
    /// Prints an FNV-64 hash of the prefill attention q to stderr on every call (skipped inside a graph capture).
    QO_DUMP: local flag = "0", "";
    /// Writes an FP16 NaN into q[0] of the prefill attention on every call, a fault injection for the error checks.
    QO_NANINJ: agree flag = "0", "";
    /// Checks the prefill rope table against the decode rope function once at bind and prints the mismatch count to
    /// stderr.
    QO_ROPECHK: local flag = "0", "";
    /// Seconds the supervisor keeps a refused boot's status up, 15 when unset.
    REFUSAL_LINGER: outside float ? "unset", "", env "GR8_REFUSAL_LINGER";
    /// Writes rank 0's residual stream after the embedding and after every attention and MLP sublayer of each offline
    /// main step to this file, in the m4sim dump format.
    RESID_DUMP: local path ? "unset", "";
    /// With --reuse-prefix, a hit keeps only whole cold-prefill grid cells (--prefill rows), so the result equals a
    /// cold prefill bit for bit. 0 keeps the whole matched prefix. 1 needs --prefill.
    REUSE_ALIGN: agree flag ? "1 with --serve, 0 offline", "0|1";
    /// Writes every layer's KV cache bytes for the prefilled rows of each sequence GR8_REUSE_HDUMP_SEQ lists to
    /// DIR/hd.<q>.r<rank>.bin, a prefix-reuse diagnostic (reuse_hdump.cuh).
    REUSE_HDUMP: local path ? "unset", "";
    /// Lists the sequence numbers (comma-separated, 0 and up) whose cache rows GR8_REUSE_HDUMP dumps.
    REUSE_HDUMP_SEQ: local ints ? "unset", "0..";
    /// With --reuse-prefix, a hit no longer needs the new prompt to reach the end of what the last sequence wrote, so a
    /// prompt can be replayed as it is. 0 restores that rule (prefix_reuse.h).
    REUSE_NOEND: agree flag ? "1 with --serve, 0 offline", "0|1";
    /// With --reuse-prefix, sets every stale cache row past the reused prefix to NaN on a hit, so a read of a stale row
    /// shows in the output. A test gate.
    REUSE_POISON: local flag = "0", "0|1";
    /// With --reuse-prefix, a hit needs the new prompt to extend the previous sequence exactly, the strict append rule
    /// (prefix_reuse.h).
    REUSE_STRICT: agree flag = "0", "0|1";
    /// With --reuse-prefix, clears every layer's per-sequence attention scratch before the suffix prefill on a hit
    /// (indices to -1, output to 0), a prefix-reuse diagnostic.
    REUSE_ZSCRATCH: local flag = "0", "0|1";
    /// Flash MoE router: 1 runs a 4-row step with the 4-row router kernel, 0 with the per-row kernel. Each warp sums
    /// its output in the same order either way.
    ROUTER_MF: agree flag = "1", "";
    /// Flash MoE router: warps per CTA of the router GEMV (one output channel per warp). Each warp sums in the same
    /// order at any width.
    ROUTER_NW: agree int = "2", "2|4|8";
    /// Sets the router GEMV's output tiles per CTA: 1 = 16 CTAs of 512 threads instead of 8 of 1024. Same tiles, k
    /// parts and reduction order, so the same bits (bind.cuh).
    ROUTER_TPC: agree int = "2", "1|2";
    /// Writes one JSON line per MTP main step to this file with each MoE layer's distinct routed expert ids, from rank
    /// 0.
    ROUTE_LOG: local path ? "unset", "";
    /// Decode MoE layers with a trellis shared expert and 2+ rows: runs the shared expert on a forked stream during the
    /// router GEMV and plan kernel. 1 = its fc1 and fc2, 2 = fc1 only. Same bytes out.
    ROUTE_OVL: agree int = "0", "0|1|2";
    /// m4sim: prints one line per target-step row (input, token, log-probabilities, residual and logit hashes) and
    /// flags any rank whose residual row differs from rank 0's.
    ROW_LINES: test flag = "0", "";
    /// m4sim with GR8_RT_LOCAL: 1 = the next sequence takes the released records over in another slot, as a g6m4 grant
    /// does, instead of in place.
    RT_ADOPT_SLOT: test flag = "0", "0|1";
    /// With GR8_RT_FMT=int4, prints the latent SQNR and rope check of the FP8 to INT4 round trip. With GR8_RT_LOCAL
    /// also checks the record bookkeeping after every change and stops on an error.
    RT_CHECK: local flag = "0", "";
    /// Sets the row format of the GR8_RT_STORE records: fp8 = the cache's own 656 B row, int4 = the 416 B INT4 row of
    /// --kv-format int4.
    RT_FMT: agree text = "fp8", "fp8|int4";
    /// Sets the first position GR8_RT_STORE holds: a multiple of 64, at least GR8_CP_R, and a context bucket boundary.
    /// With GR8_RT_LOCAL it must equal GR8_CP_R.
    RT_L: agree int ? "CP_R", "";
    /// Releases each session's tail pages past GR8_RT_L to the store: they cycle through this many window pages per
    /// rank and are read back from slots or the store. Needs GR8_RT_STORE.
    RT_LOCAL: agree int = "0", "";
    /// With GR8_RT_LOCAL, a prefill stages each released page with one read of each row's slice for the layer. 0 reads
    /// the page's whole block for every layer of its group, about 3.9x the bytes.
    RT_PF_SLICE: local flag = "1", "";
    /// With GR8_RT_LOCAL, sets the main model's physical pages per rank, so --cap becomes the logical length only. Must
    /// hold the GR8_CP_R prefix and one window.
    RT_PLACE: agree int ? "every tail page of --cap", "";
    /// m4sim only: overwrites the local copy of every row sent to GR8_RT_STORE, so a read of the local copy shows in
    /// the output.
    RT_POISON: test flag = "0", "";
    /// Sets GR8_RT_STORE's resident slots per indexer group, at least 2048. With GR8_RT_LOCAL a smaller pool lets fewer
    /// sessions past R in at once.
    RT_POOL: agree int ? "4096, or with RT_LOCAL 2048 x rows a step x sessions past R", "";
    /// With GR8_RT_LOCAL, counts what a prefill reading only the selected rows of released pages would read, at each
    /// full-index layer. Measurement only, it synchronizes the stream.
    RT_SELSTAT: local flag = "0", "0|1";
    /// Sets each rank's GR8_RT_STORE lease budget in bytes (the launcher's KV_ROWS size=). With GR8_RT_LOCAL it sizes
    /// the records, and the pool admits a tail past R only when its records fit.
    RT_SIZE: agree int ? "unset", "";
    /// Sends a session's whole pages past GR8_RT_L to a per-rank lease, and CP decode reads them back. mem: or
    /// remote:HOST:PORT (a KV node). Needs --serve, GR8_CP and GR8_S2_BELOW = GR8_CP_R. Local: with remote:auto the
    /// launcher gives odd ranks the KV node's second rail.
    RT_STORE: local text ? "unset", "";
    /// S2: chooses each round's MTP draft count (1..k) from the sessions' accept rates and the S2 step costs. Adds
    /// about 1 GiB of main-step graphs.
    S2_ADAPT: agree flag = "0", "";
    /// S2: an admission with another request queued behind it and a slot left holds its boundary, so a burst of queued
    /// requests is admitted at one boundary instead of one per notice.
    S2_ADMIT_HOLD: agree flag = "0", "0|1";
    /// S2: runs attention once per session instead of one batched launch over the sessions. Refused with graphs and on
    /// Flash.
    S2_ATTN_PERSESSION: test flag = "0", "0|1";
    /// S2: the context below which sessions share the pool's pages: 64..GR8_CP_R, a multiple of 64.
    S2_BELOW: agree int ? "CP_R", "";
    /// S2 test: RANK:MS makes that rank sleep MS ms after each request's bind, to show what a bind skew does to the
    /// next quantum on the other ranks.
    S2_BIND_DELAY: local text ? "unset", "";
    /// m4sim --s2 probe: runs every step at this k1 row class.
    S2_CLS: test int = "0", "";
    /// S2: while any session decodes, a prefill quantum takes at most this many rows on one lane, so decode rounds come
    /// round sooner. A multiple of 64, at most --prefill. 0 = the full quantum.
    S2_DECODE_PF: agree int = "0", "";
    /// S2: while a prefill waits and sessions decode, runs this many decode rounds per prefill quantum instead of one.
    /// 0 or 1 = off.
    S2_DECODE_ROUNDS: agree int = "0", "0..64";
    /// S2 with DFlash2: a session that decodes alone drafts each round with DFlash2 or MTP, whichever is expected to
    /// commit more tokens per ms. 0 = DFlash2 every round.
    S2_DSEL: agree flag = "1", "";
    /// S2 drafter choice: rounds a session stays on one drafter before it may switch.
    S2_DSEL_DWELL: agree int = "8", "";
    /// S2 drafter choice: switches when the other drafter's expected tokens per ms is higher by this fraction.
    S2_DSEL_MARGIN: agree float = "0.15", "";
    /// S2 drafter choice: 1 prices both drafters from the frozen cost table in GR8_S2_DSEL_TABLE. Needs --serve and
    /// --dflash-manifest.
    S2_DSEL_MODEL: agree flag = "0", "0|1";
    /// S2 drafter choice: every this-many MTP rounds the DFlash2 drafter also drafts, and its drafts are scored against
    /// the next tokens.
    S2_DSEL_PROBE: agree int = "4", "";
    /// With GR8_S2_DSEL_MODEL=1, the drafter choice's cost table: a key:value list (v1..v8, d, m, e, p and optional
    /// keys) or the path of a file holding one.
    S2_DSEL_TABLE: agree text ? "unset", "";
    /// S2: logs one line when the head anchors a notice, one when each rank applies the commit, and one per prefill
    /// quantum. Logs only.
    S2_EVENTLOG: local flag = "0", "";
    /// m4sim --s2: per-session generation lengths (comma-separated), so sessions finish at different rounds.
    S2_GEN: test ints ? "--gen", "";
    /// S2: a session holds the KV pages it needs now and takes more as its output grows, while every live session can
    /// still reach its worst case. 0 = every run holds its worst case from admission.
    S2_GROW: agree flag = "1", "0|1";
    /// m4sim --s2: the sessions' runs come from a pool of this many pages (s2/pool.h) and grow on demand: each starts
    /// with its prompt + 1 + guard rows, grows before each round, sits a round out when the pool would be unsafe, and
    /// gives its pages back when it finishes. At most the pool's pages. Not with GR8_S2_RUNS or GR8_S2_PAGES.
    S2_GROW_POOL: test int ? "fixed runs (GR8_S2_RUNS, GR8_S2_PAGES or the default layout)", "1..";
    /// m4sim --s2: q:after:unless:id,id,.. gives session q an insert rule (s2_batch.h), several rules joined by
    /// semicolons. unless -1 = none.
    S2_INSERT: test text ? "unset", "";
    /// m4sim --s2: seed[,pages] gives each slot a scattered run from a seeded shuffle of the pool. Output must equal
    /// the default layout's.
    S2_PAGES: test ints ? "unset", "";
    /// m4sim gate: B[,pages] runs every sequence as one session at base page B below R. Stdout must equal the run at B
    /// = 0. Needs GR8_CP, --prefill and no --s2.
    S2_PFBASE: test ints ? "unset", "";
    /// m4sim negative control for GR8_S2_PFBASE: stages prefill index keys at logical pages from page 0, so the output
    /// must differ.
    S2_PFNOFIX: test flag = "0", "";
    /// m4sim gates GR8_S2_PFBASE and GR8_S2_XLIST: fills every cache with 0x3c bytes first, so a read outside the run
    /// changes the output.
    S2_PFPOISON: test flag = "0", "";
    /// m4sim --s2 without S2 below R: the pool's page count for variable runs.
    S2_POOL: test int ? "unset", "";
    /// m4sim --s2: when one session is left, relocates its run to base page 0 instead of leaving it where it is.
    S2_RELOC: test flag = "0", "";
    /// m4sim --s2 with GR8_S2_RELOC: copies at most this many pages, to time the stall.
    S2_RELOC_PAGES: test int ? "unset", "";
    /// S2: rank 0 prints one '# s2_round' line per decode step, with the step's routed experts.
    S2_ROUNDLOG: local flag = "0", "";
    /// m4sim --s2 probe: prints one line per target row with an FNV hash of its residual and its log-probability bits
    /// (rank 0).
    S2_ROWHASH: test flag = "0", "";
    /// m4sim --s2: b:p,b:p,.. sets each slot's run (base page, pages) on every model.
    S2_RUNS: test text ? "unset", "";
    /// S2 with a KV node lease: how often in ms the head reports to the node and asks for segments.
    S2_SHARE_MS: local int = "1000", "";
    /// m4sim --s2: a session left alone runs the single-session path. 0 keeps it on the S2 path.
    S2_SOLO1: test flag = "1", "";
    /// S2: every N main steps each rank prints, per decode step kind, the mean GPU span, host input prep, launch call
    /// and fence wait. Measurement only.
    S2_TIMELINE: local int = "0", "";
    /// S2: seconds the bootstrap waits on the control links before a rank gives up.
    S2_TIMEOUT: local int = "120", "1..86400";
    /// m4sim gate: seed[,pages] runs every sequence as one session past R on a shuffled page list of the pool's two
    /// classes. Stdout must equal the run without it.
    S2_XLIST: test ints ? "unset", "";
    /// S2 with DFlash2: overrides the DFlash2 round, MTP round and each extra MTP chain step costs with three comma-
    /// separated nonnegative times in ms.
    SPEC_DRAFT: agree floats = "8.1,5.5,2.1", "";
    /// S2 with DFlash2: overrides the verify costs for 1..8 rows with eight comma-separated positive times in ms, for
    /// the DFlash2 draft count and the drafter choice.
    SPEC_TABLE: agree floats ? "built-in measured table", "";
    /// S2: prints a '# step_tmr' line per decode step with its launch and fence time in us, its graph key, rows and
    /// context.
    STEP_TMR: local flag = "0", "";
    /// Whether scripts/build.sh builds the test binaries: 0 skips them, only builds them alone.
    TESTS: outside text = "1", "0|1|only";
    /// flash_dsa_pf_test: compares the fused MLA with the current GEMM prefill instead of the older P1 math.
    TEST_SPARSE_MLA: test flag = "0", "";
    /// A test of the zeroing order: every dmalloc zeroing waits behind an N ms kernel on the legacy stream, so a write
    /// to new memory not ordered after it loses every time.
    TEST_ZERO_DELAY_MS: local int = "0", "";
    /// Where the loader writes its planned rank manifests.
    TMPDIR: local path = "/tmp", "", env "TMPDIR";
    /// The tokenizer directory the front end tests need, unset skips them.
    TOKENIZER_DIR: test path ? "unset", "";
    /// Uses the window-2 fc1 split (256-thread CTAs, KS1 x2, WG 1) at M > 1 on every MoE layer, not only layers with
    /// 2-bit units. Same k ranges and order, so the same bits (bind.cuh k1_split).
    TR3_W2ALL: agree flag = "0", "";
    /// trellis: prefill runs an exlr8 v2 layer's shared expert as a trellis pass instead of from an FP8 copy made at
    /// boot, which saves the copy (about 9.7 MB per layer at TP4).
    V2_SHARED: agree text ? "FP8 copy", "trellis";
    /// Warms the L2 with the next weights while a decode collective waits: 1 = this layer's router and shared fc1 and
    /// the next layer's q_a, 2 also kv_a and q_b. Loads only, no output change.
    WARM: agree int = "0", "0|1|2";
    /// Sums each loaded image on the CPU and the GPU and stops the boot with exit 3 if they differ, then seals the sums
    /// for /health?deep (weight_check.cuh). 0 = off.
    WEIGHT_CHECK: local flag = "1", "";
    /// RANK:OFFSET flips one byte of rank RANK's first image after the load, a test of GR8_WEIGHT_CHECK that makes the
    /// boot stop with exit 3.
    WEIGHT_CHECK_FLIP: local text ? "unset", "";
    /// The tensor-parallel widths scripts/build.sh builds, "3 4" for example.
    WIDTHS: outside text ? "unset", "";
    /// The working directory of a bare-metal install: bin/, out/, s2/, run/.
    WORK_DIR: outside path ? "the repository", "", legacy "G6_HOME";
    /// An XGrammar source tree to build request grammars into the engine.
    XGRAMMAR: outside path ? "unset", "";
    /// A prebuilt XGrammar library in place of building one from GR8_XGRAMMAR.
    XGRAMMAR_LIB: outside path ? "unset", "";
    /// 1 = transport trace: GPU timer stamps per collective, proxy tick stamps per rail and per-QP retry counters,
    /// dumped at proxy stop or when GR8_XTRACE_QUIT appears. Measurement only.
    XTRACE: local flag = "0", "";
    /// Directory for the GR8_XTRACE dumps, g6xtrace.r<rank>.f<n>.txt.
    XTRACE_OUT: local path ? "/tmp", "";
    /// With GR8_XTRACE=1, traces the two prefill fabrics instead of the decode fabric.
    XTRACE_PF: local flag = "0", "";
    /// With GR8_XTRACE=1, each fabric dumps its trace once when this file appears.
    XTRACE_QUIT: local path ? "unset", "";
}
