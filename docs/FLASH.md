# GLM-5.3-Flash

One to four DGX Spark (GB10) boxes (TP=1 to TP=4). Weights: the exlr8 release on Hugging Face,
[kindlingai/glm-5.3-flash-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4).

gr8 serves [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) from its exlr8 release
([kindlingai/glm-5.3-flash-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4)) at TP=4,
TP=3 (a three-box ring, variant k4), TP=2 (two boxes) or TP=1 (one box, variant k2, the one width whose weights fit). The engine is `bin/g6m4_flash`, a build of the same `g6m4` source with `-DG6_MODEL_FLASH=1`
(`engine/model_geom.h` holds the geometry).

## Results

On one and two boxes, from each recipe's `kindling.json` as of 2026-11-08 (the numbers on its card, with the image's
defaults):

| | one box (TP=1) | two boxes (TP=2) |
|---|---|---|
| manifest, KV, drafting | `bit201`, int4 KV, DFlash2 | `bit349`, int4 KV, DFlash2 |
| decode, code / prose / structured | 53.8 / 28.2 / 65.1 tok/s | 74.5 / 41.9 / 94.6 tok/s |
| prefill, 8K / 32K / 128K prompt | 995 / 1,024 / 916 tok/s | 1,675 / 1,676 / 1,354 tok/s |
| aggregate decode, 1 / 2 / 4 / 8 streams | 43.5 / 41.3 / 53.4 / 57.5 tok/s | 54.0 / 62.5 / 75.5 / 98.2 tok/s |
| KV pool | 700,032 tokens, up to 4 requests at once | 2,097,152 tokens, up to 8 requests at once |
| from | [glm-5.3-flash-exlr8-1x-spark `kindling.json`](https://github.com/kindlingai/glm-5.3-flash-exlr8-1x-spark/blob/3e9754c5/kindling.json) | [glm-5.3-flash-exlr8-2x-spark `kindling.json`](https://github.com/kindlingai/glm-5.3-flash-exlr8-2x-spark/blob/81c29ca3/kindling.json) |

The recipes use two other manifests of the same release. `bit201` (one box) is K2, with the 124 routed experts that
carry the most routing weight at tool-call positions at K3. `bit349` (two boxes) is K3, with K4 where it lowers KL most,
about 3.49 bits a routed weight.

At TP=3 and TP=4 the sections below have what is measured so far. Open bugs: [KNOWN-ISSUES.md](../KNOWN-ISSUES.md#glm-53-flash).

## Build and run

    ./gr8 glm53f build
    ./gr8 glm53f up --model https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4 --model-cache DIR --tp 4 ...
    ./gr8 glm53f smoke

With the box's model-loader fetching the weights ([DEPLOY.md](DEPLOY.md#with-model-loader)), give its store instead:
`--model-loader unix:/var/lib/model-loader/model-loader.sock --model-cache /var/lib/model-loader`, or in Docker add `-f docker/model-loader.yaml`. gr8 finds
the Flash release in model-loader's catalog by its repo. Its catalog needs an entry for the Flash release, with the width files in small
segments as for GLM-5.3. GLM-5.3 and Flash then share one store and its disk budget.

`./gr8 glm53f` takes the options of `./gr8 glm53` (KNOBS.md). Each box fetches its rank's part of the release and loads
it through the same release planner as GLM-5.3 (`engine/loader-rs`): `release::plan` reads the model from the release's
own metadata (`hidden_size` 4096) and maps Flash's tensors with `engine/loader-rs/src/flash.rs`. The width manifests are
`k2`, `k3` and `k4` (the default); the recipes use `bit201` (one box) and `bit349` (two boxes), see [Results](#results).

The front end serves the model as `glm-5.3-flash` (`/v1/models` lists it first) and also answers to `glm53f`. A request
that names `glm-5.3` gets a 404 (one that names no model is served), so a router that picks backends by model name
keeps Flash and GLM-5.3 apart (`served` in `tools/launch/models.py`). `./gr8 glm53f smoke` sends the Flash name.

## The model

- 45 layers plus the MTP layer 45. 34 layers use KDA linear attention (`engine/kda`). The others (3, 7, ..., 43) use
  DSA sparse MLA with an indexer (`engine/attn/flash_attn.cu`).
- Four mHC residual streams (`engine/mhc`).
- 288 routed experts, top-8, from layer 3 (the exlr8 trellis experts, as GLM-5.3).

## What works

- `./gr8 glm53f up` boots in about 16 s from a local cache and passes `./gr8 glm53f smoke`.
- TP=4 prefill of an 8K prompt: about 3.55 s at 2200 MHz.
- 8 sessions at TP=4 and 4 at TP=3 share decode steps (below), with MTP=1 (`flash_defaults` in `scripts/engine_args.sh`).
- At TP=4, TP=2 and TP=1 a session that runs alone drafts with the DFlash2 drafter when the boot has one: `--drafter https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2` (below).
- `response_format` `json_object` and `json_schema` replies are masked to their XGrammar grammar, as on GLM-5.3 (`GRAMMAR`).

## Image input (`GR8_GLM53F_VISION=1`, opt-in)

The engine takes images in chat requests with the vision tower loaded ([FLASH-VISION.md](FLASH-VISION.md)). Without
the knob it serves text, and the fetch skips `dense/vision-*`.

## Context

`CAP` defaults to 262208: a request of 262,144 tokens (prompt + `max_tokens`) plus a page for the guard rows the
scheduler keeps past it (2 at MTP=1), so the front end's `max_model_len` is 262,206 (`pool_rows - guard`).

`CAP` is the pool the sessions share, and `MAX_MODEL_LEN` (`--max-model-len`) is the longest single request: the
shim takes the smaller of it and `pool_rows - guard`, and the front end reports that. Unset, it is the model's
1,048,576 positions, so a `CAP` past 1,048,640 means more room for sessions at once, never a longer context. Admission
reserves each request's `prompt + max_tokens` pages from the pool, so with several sessions, for example,
`CAP=1048640 MAX_MODEL_LEN=262144` holds four full-length requests at once, or more shorter ones.

Only the attention layers keep KV: the 11 DSA layers and the MTP layer, 12 bindings a rank. KDA's state is the same size
at any context. A token costs each of them a 1,024 B bf16 latent row and 128 B of pooled indexer keys
(`engine/attn/kvfmt.h`; a build with `-DG6_FLASH_KV_FP8=1` makes the row 528 B, the int4 build below 288 B), 13,824 B
in all (4,992 B with int4). A token's raw indexer row (512 B) feeds only the pool of 4 it belongs to, so it is not in
the pages: the pool reads it from its own call, or from the binding's raw ring of the last 64 positions of each session
(256 KB a binding). Without CP (below) every rank holds the whole context, and the cost is the same at TP=4 and TP=2:

| CAP | pages of 64 | KV a rank | with int4 |
|---|---|---|---|
| 33,280 (the old default) | 520 | 0.46 GB | 0.17 GB |
| 262,208 | 4,097 | 3.62 GB | 1.31 GB |
| 1,048,640 | 16,385 | 14.50 GB | 5.23 GB |

The indexer's score scratch holds one key tile per row: a call whose context passes 16,384 pools (64K tokens) scores
the pools in tiles of that size and keeps a running top-512 (`kp_select_tile` in `engine/attn/kpool.cuh`, the same
lists as one pass; `engine/tests/test_flash_kpool.cu`), so decode and prefill scratch stop growing with `CAP` past 64K.
The model's config allows 1,048,576 positions, and `CAP` may go higher where memory allows. Each layer's bind checks
that `GR8_MEM_MIN` (8 GiB) stays free, so a `CAP` that leaves too little stops the boot; the boot log's `MemAvailable`
line shows what is left.

Nothing else in the engine limits the context to less than `CAP`:

- Flash has no RoPE (`g6_attn_rope_init` does nothing), so a position only picks a KV row.
- The indexer's top-2048 selection takes the general path past 8,192 visible pools of 4 tokens (`engine/attn/kpool.cuh`),
  with 64-bit offsets.
- Prefill runs in chunks of 4,096 rows a lane, two lanes, at any context. Decode captures nine buckets, 2048 to 262144
  by doubling, and `CAP`.

With one session (`SESS=1`, the default) `SHARE_LIMIT` (32768) shares nothing: it only sets the `max_tokens` a chat
request without one gets, `SHARE_LIMIT - 64 - prompt` but at least 8192, capped by the room left. With several (below)
it also decides which requests share decode steps, as on GLM-5.3: a longer one waits for the others and runs alone.

Prefill time grows faster than the prompt because the indexer scores every earlier key for each row. At TP=1 a
4,096-row quantum takes 3.2 s at the start of a prompt, 5.1 s at 240K and 8.4 s at 900K (90 C, clocks held at
1,781 MHz). The shim errors a request after 900 s with no output (`--stall-timeout`), but the engine reports each
quantum to it (an `f` line, `s2/sessions.h progress`), so the timeout bounds one quantum, not the whole prefill: a
923K-token prompt prefilled in 1,371 s with the default.

## Context-parallel KV (`GR8_CP=1`, experimental)

Off by default in the Flash build (`GR8_CP` unset is 0 here, 1 in GLM-5.3's). With `EXTRA_ENV=GR8_CP=1` (the launcher
sets `GR8_CP=0` and `GR8_CP_R=$SHARE_LIMIT` before it; add `GR8_CP_R=R` for another R), the 12 attention bindings keep the first `R` tokens on every
rank, and each later 64-row page on one rank only, `(pos / 64) % TP`, as GLM-5.3's CP does (#384). A page holds 16 whole
pools of the indexer, so a pool and its key live on one rank. Decode past R runs five stages per DSA layer
(`engine/attn/flash_attn.cu`, `engine/runtime/flash_dsa.cuh` `launch_dsa_cp`):

1. append the rows this rank stores (every rank writes the raw index ring, which stays replicated), absorb its heads;
2. score the pools this rank owns, the prefix's too, and keep a top 512 per row;
3. all-gather the candidates (512 x 8 B a row) and merge them into the replicated path's list, bit for bit;
4. all-gather the absorbed queries, and for every head group a partial attention over this rank's selected keys (with
   equal head groups, TP=2 and TP=4, all of them in one call), each sent to the group's rank;
5. merge the partials in rank order (log-sum-exp), then the value projection.

The lists and the cache bytes equal the replicated path's. The output does not: the merge sums in another order, as
GLM-5.3's does, so T=0 output with CP on is not the same bits as with it off. `engine/tests/test_flash_cp.cu`
(`TEST=test_flash_cp scripts/c1_tests.sh`) checks all three against the replicated bindings on one GPU.

Limits for now:

- A prefill quantum past R runs on GLM-5.3's staged view (`engine/runtime/prefill.cuh` `cp_pf_*`): before each DSA
  layer's attention, every page below the quantum is gathered from its owner into one of two pools of one layer at
  `CAP` (pinned, `2 x CAP x (row + 128 B)` a rank: 0.87 GB at `CAP` 1,048,640 with int4), the attention runs unchanged
  on it, and the rows this rank stores go back. So the prefill is the replicated path's bits; it sends every remote page
  below the quantum once per DSA layer and quantum. m4sim does not stage, and refuses a prompt past R with `--prefill`.
- With several sessions the pool takes GLM-5.3's page classes (S2 below R): a request whose `prompt + max_tokens`
  passes R gets its prefix and its share of the tail below `place`, the MTP layer's copy above it.
- The MTP layer keeps its whole KV (it is the draft model's binding), and the DFlash2 drafter its own window.
- With `GR8_GLM53F_CACHE`, each rank's checkpoint keeps the pages of the 11 DSA layers it holds (another rank's page
  is zeros in the entry) and the MTP layer's whole pages. Not with `GR8_GLM53F_KV_CHUNKS`.
- Not measured on a cluster yet.

## One box (TP=1)

`--tp 1` serves the whole model on one GB10 with the k2 width (about 93 GB of weights on the device). The ranks' fabric
has no peers then: `g6::Fabric` keeps its region for the kernels' own slots and skips the NIC, the bootstrap and the
proxy (`engine/comms/fabric.cpp`), and the S2 head agrees with itself (`s2/control_channel.h`). The launcher skips the
RDMA checks. The router is not split (no ranks to split it over), so the routed experts run on the expert tiles. A
session alone drafts with the DFlash2 drafter when the boot names one (`--drafter`): its body is whole on the one rank,
about 1.3 GB, and the supervisor converts it for the rank on the first boot.

Memory on one GB10 with the K2 release, int4 KV, `--sess 4`, the DFlash2 drafter and aligned checkpoints on an SSD
store with a spare-RAM cache (`--pf-lanes 1`), MemAvailable at boot and the lowest under a long prompt:

| CAP | boot | lowest | prompt |
|---|---|---|---|
| 262,208 | 6.8 GiB | 6.3 GiB | 240K |
| 700,032 | 4.8 GiB | 4.5 GiB | 640K: TTFT 863 s (760 tok/s) |
| 1,048,640 | 2.9 GiB | 2.3 GiB | 900K: TTFT 1,361 s (677 tok/s), then 14.5 tok/s decode; cached TTFT 10.5 s |

At 1,048,640 the boot's headroom checks after the prefill lane come to about 5.5 and 4.8 GiB, so it needs
`GR8_MEM_MIN=4` (the others pass at 6).

## int4 KV (`--kv-format int4`, experimental)

`bin/g6m4_flash-int4` (`KV_FORMAT=int4 MODEL=flash scripts/build.sh`, `-DG6_KV4=1`) stores the latent in 4 bits: 256 B
of codes and one FP16 scale per 32 dims (amax / 7), 288 B a row against bf16's 1,024. The pooled indexer keys are
unchanged.
There is no rotation: Flash's latent is NoPE, and every read dequantizes in place (`latent`, `lat4`, `stage_cvt8` in
`engine/attn/flash_attn.cu`). `--kv-format int4` makes the launcher run it and pass the flag, which the build requires.

## Several sessions

`--sess N` (`SESS`, 1 to 8; 8 at TP=4, 4 at TP=3, 1 at TP=2) serves up to N requests at once on S2, as GLM-5.3 does: their decode
rounds share main and draft steps, and their KV runs share the `CAP` pool (`--s2-sessions N` to `g6m4_flash`). Those per-TP
counts are the launcher's defaults (`scripts/engine_args.sh`); the two-box recipe overrides `SESS` to 4 at TP=2 and serves
four requests at once.

- KDA state: each session holds a region of `MTP + 1` slots (`FlashSlots` in `engine/runtime/s2/flash_ring.h`), reset
  when a request starts on it. With the drafter, region 0 is 8 slots wide (below). In a main step each session's first
  row loads its own committed slot (`StepIn::krun`, `Args::src` per row in `engine/kda`), and its verify rows write its
  own ring, so sessions sharing a step never read each other's state. The engine allocates `SESS x (MTP + 1)` slots, plus
  `8 - (MTP + 1)` with the drafter.
- Prefill runs a request alone on its own slots and KV run.
- Every main and draft width a round can take is captured as a graph per context bucket at boot
  (`flash_shapes`, `engine/runtime/s2/flash_native.cuh`); a step outside the set is an error, not a capture.
- A round of every session, `SESS x (MTP + 1)` rows, must fit a step's 32 (`G6_MX`); the launcher and the engine refuse
  more.
- With `SESS=1` the engine runs the one-session path as before.

The memory these take (slots and graphs), the speed of shared steps and the output of sessions sharing steps have not
been measured on Flash yet.

## DFlash2

At TP=4, TP=2 and TP=1 a session that runs alone drafts with the DFlash2 block drafter, as on GLM-5.3 (`s2_batch.h`, `Batch::dflash`).
When a second session arrives it moves to MTP, and the sessions share steps. `--spec-adaptive` (on with the drafter) picks
each round's draft count from the verify table in `spec_len.h` (`flash_cost`), and the drafter choice (`GR8_S2_DSEL`) runs
MTP rounds when they commit more tokens a millisecond, which they do on prose. A DFlash2 round writes up to 8 rows, so
one KDA region is 8 slots wide (`Model::kda_ring`; 153 MB a slot at TP=1) and the scheduler keeps 14 guard rows
(`Config::guard_tokens`). DFlash2 drafts for a session alone, so one wide region is enough: at its first DFlash2
round the session takes region 0 and the session that held it, retiring or free, takes the session's MTP region
(`FlashSlots::take_wide`, `widen` in `engine/runtime/s2/flash_native.cuh`, which copies the session's `MTP + 1` slots
across first). The prefill fills the drafter's prompt context. The batched graphs of several sessions leave out the
drafter taps.

A session that started next to others decodes on MTP. When the others finish, `DFL_REUSE=2` (the default with MTP and
the drafter) promotes it, as on GLM-5.3 (`Batch::promote`): MTP rounds feed the drafter the accepted rows as context
until it holds `DFL_REUSE_MIN` (256) rows, then the session drafts with DFlash2 and takes region 0 as above. Rank 0's
`# s2 dflash reuse slot` line for the request shows the position it was promoted at.

speedbench `run` at TP=4, 2200 MHz, one stream (tok/s): code 70.2 -> 88.9, structured 72.1 -> 97.2, agentic 67.5 -> 84.7;
prose 65.0 -> 59.0 before the drafter choice settles on MTP.

## Session checkpoints (`GR8_GLM53F_CACHE=aligned`, experimental)

Off by default. With `FLASH_CACHE=aligned` (`GR8_GLM53F_CACHE=aligned`), a prompt's prefill keeps a checkpoint at its last
cold prefill grid point, and a later prompt that starts with the same tokens is prefilled from that point, not from 0.
This is the Flash form of prefix reuse for an agent's next turn. The code is `engine/runtime/s2/flash_checkpoint.h` (the
index, the aligned point, the manifest) and `flash_checkpoint.cuh` (the store), hooked into `s2/flash_native.cuh`.

- The grid is the prefill quantum: `--prefill` rows times the lanes, 4,096 x 2 = 8,192 rows. A prompt of N tokens keeps
  a checkpoint at `p = floor((N - 1) / 8192) * 8192`, so prompts of 8,192 tokens or fewer keep none.
- A checkpoint holds rows `[0, p)` of the 12 attention bindings (the latent rows and the pooled indexer keys), the
  session's first KDA slot's state and conv history, its raw index rings, and the prompt's tokens `[0, p]`. A later
  prompt hits it when it starts with those `p + 1` tokens under the same cache namespace (the steering profile); the
  longest entry wins, then the newest.
- A hit restores the checkpoint at bind, after the cold reset, and prefills from `p`.
- With several sessions (`SESS` 2 to 8) each session keeps and restores its own: the rows are the session's run's
  pages in order (its logical pages, wherever the pool put them), and the KDA state is its first slot (`FlashSlots`).
  An entry is the same bytes whichever session or pages took it, so any session may restore it. Captures wait for the
  next boundary as with one session; no other session's work touches the run or the slot until then.
- With the DFlash2 drafter (`DFLASH=1`, below).

With the DFlash2 drafter (aligned checkpoints only: the engine refuses `GR8_GLM53F_CACHE=terminal` with it, since a
terminal checkpoint's KDA state comes from the verify ring and is not cold-exact): a checkpoint holds the target's state
but not the drafter's context, the last 2,048
positions of taps in its KV ring (84 MB a rank as prompt taps). Nothing more is stored. The prefill from `p` taps the
positions it runs, and the drafter takes its context from those (`engine/runtime/s2/flash_dfl_window.h`):

- A hit at least 2,048 rows before the prompt's end (an identical prompt sent again: `p` is the grid point at or below
  `N - 1`, so this holds whenever the last grid point is 2,048 rows or more back) leaves the drafter's whole window to
  the prefill. The drafter starts as on a cold prompt.
- A hit inside the window (an agent's next turn, a few hundred tokens past the last turn's grid point): the prefill taps
  only `[p, N - 1)`. Under `DFL_REUSE=1` (`GR8_DFL_REUSE`; on with `FLASH_CACHE`, and in mode 2) the session keeps the
  drafter, as a GLM-5.3 prefix-reuse hit does (`s2_batch.h` `Batch::reuse_start`). When the drafter's ring still holds
  the previous turn of the same conversation (the last DFlash2 session there had this prompt's tokens up to `p`, and its
  rows were not written over), the context starts at the window's start and the session drafts with DFlash2 from its
  first round. Else the context starts at `p`, and with fewer than `DFL_REUSE_MIN` (256) rows the session runs MTP rounds
  that feed the drafter until it holds that many, then DFlash2. Rank 0 prints `# s2 dflash reuse slot` lines with the
  rounds on each drafter and the warm-up.
- A hit that holds every prompt row but the last (a replay) has no prefill: the drafter starts at the prompt step, with
  the ring's rows when they are this conversation's.
- `DFL_REUSE=0`: a hit inside the window runs MTP for the whole request, as before. `DFL_REUSE=2` adds the promotion of
  a session alone on MTP (DFlash2, above).

The ring's rows below `p` came from the previous turn's prefill and decode taps, not from a cold prefill of this prompt,
so drafts may differ from a cold boot's. The target verifies every draft: the output is the same as without the drafter
(aligned is still bit-exact against no cache). Not measured on a cluster yet: smoke, bench's `TTFT cached` with the
drafter, and decode after a restore against a cold-prefilled request's are open on #340.

Why the output is bit-exact against no cache: a cold prefill runs its quanta as `[kG, (k+1)G)`, and the KDA prefill is
chunked by quantum, so a warm prefill from `p = kG` runs the same quanta on the same KDA state and the same pages. The
MTP layer's row `p - 1` reads token `p`, which is why the match runs through token `p`. The indexer's pools are
recomputed from the raw rows (`engine/attn/kpool.cuh`), so no other state is kept. Rows past `p` and the other KDA slots
are as a cold bind leaves them.

Why a checkpoint is not taken at the end of the prompt: decode advances the KDA state one row at a time
(`engine/kda/kda.cuh`), and the prefill runs it in chunks of 64 rows (`engine/kda/kda_prefill.cuh`). The two orders of
fp32 arithmetic give different state, so a state taken after decode is not the state a cold prefill of the same tokens
gives. A terminal checkpoint is not cold-exact by construction.

`FLASH_CACHE=terminal` (`GR8_GLM53F_CACHE=terminal`, experimental, **NOT cold-exact**) keeps aligned's checkpoints and
one more at the end of every request: `p` = the rows the request consumed (prompt and output; its last output token is
still pending and becomes token `p`), with the KDA state the verify ring holds for `p` and the pages through `p`, the
last one partial. A later prompt that starts with those `p + 1` tokens (an agent's next turn sent as raw tokens) is
prefilled from `p`. Its output differs from a cold prefill's: the KDA state comes from decode, and the prefill from `p`
runs quanta a cold prefill does not. How much is measured (token diff against no cache on the same trace, NLL delta,
needles), not argued. After a terminal hit the request keeps no aligned checkpoint (its rows are not cold rows), so
aligned entries stay exact. The MTP layer's row `p - 1` may not hold the last token's draft row; that changes only
which drafts are accepted. The boot and request log lines say `terminal`. Aligned stays the mode to use when the output
must match no cache.

The store: the one `KV_STORE` (`--kv-store`; `mem:`, `ssd:PATH` or `remote:HOST:PORT`), else this process's memory,
`FLASH_STATE_BUDGET_MIB` (8,192) a rank. An SSD folder starts empty at every boot unless `persist=on` (aligned only).
Then the index is written beside the blocks after every capture and resize (`flash-ckpt.index`), and each entry's tokens
once, in `flash-ckpt.tokens/<generation>`. Every file is written to a temporary name, synced, renamed and its folder
synced. A boot with the same compat line (model set, TP, rank, CAP, layout, engine build) and store size reads the
index, and a helper thread checks every entry's stored manifest, pages and state (the KDA state, then the raw index
rings) against the manifest's checksums. An entry sealed by another entry format (`FlashManifest::FORMAT`; 2 since the
raw index rows left the pages, 3 since a CP binding keeps only its own pages) is dropped.
The blocks of an entry are not synced before the index names it, so after a power loss the index can name an entry whose
blocks never reached the disk: the checks drop it, so the boot loses the entry, never serves its bytes.
On a GB10's SSD the checks run at about 6 GiB/s, as fast as reading the store alone, so a full 64 GiB store takes
some 11 s. The thread runs during the graph capture, and the serve waits for it. Before serving, the ranks agree: each sends one line an
entry, every rank keeps the entries every rank kept, and rewrites its index. So a rank whose index was lost or torn, or
whose check dropped an entry, makes the others drop it too, and the boot goes on. Persistence is for fixed-size `ssd:`
stores: after a group lease resizes, the next boot finds another store size and starts empty. Entries a capture drops
leave the index before their blocks are overwritten, and an index that cannot be written turns persistence off for that
boot (the file goes, so the next boot starts with no index). A spare RAM cache on the store (the launcher's
`cache:NAME,kind=spare_ram` in `KV_STORE`, passed to the engine as `spare_ram_cache=on`, and refused with `group=`)
keeps a copy of the blocks in memory the kernel may take back, as on GLM-5.3's stores. On a KV node the lease is
`g6m4f-ckpt-<host>-<epoch>-r<rank>`, apart from the pool stash's. Each checkpoint is one contiguous range of blocks with a
manifest and GPU checksums of every extent. A restore that does not match its checksums removes the persisted index and
stops the engine.
A full store drops the least recently used checkpoints. Every rank keeps the same index, and its digest joins the pool
digest the ranks compare.

`group=` on the store (`remote:HOST:PORT,size=N,group=G[,cap=,weight=]`, experimental, D-331 M8 (c)): the lease is a
member of group `G` on the KV node, the way a full model's `--kv-store group=` tier is, so the two models share the
node's memory. It starts at `size=`; the head reports to the node every `GR8_S2_SHARE_MS` and asks for segments when
captures drop entries, an entry does not fit at all, or the next capture would drop one (after a capture, or at any
report while a request is resident or queued; the node holds the room an ask made another group give and grants it at
the next report, so the grow lands before the capture that would drop), and offers what giving would
drop (the least recently used checkpoints' ages) only while idle (`KV_SHARE_WHEN=idle`, default; `always`). A grant
or give is a Resize event every rank applies at the same commit: a grow adds blocks; a shrink drops the least recently
used checkpoints until the rest fit, copies the ones above the new end down, then gives the top segments back. The log
prints `# rank R: flash ckpt share: now S segments, B blocks (N entries, D dropped) in T ms`. Not measured on a box.

Size the floor for the prompts that come back. The node rounds `size=` up to whole segments of whole blocks, and
a 1 GiB segment holds 840 blocks of Flash's TP=2 block (1,277,952 B), which is just under 1 GiB. So `size=1G` is 2
segments (2.00 GiB, 1,680 blocks) and `size=2G` is 3 (3.00 GiB, 2,520 blocks). One checkpoint at 16,384 rows is 317
blocks, so `size=1G` holds 5 and `size=2G` holds 7. A cycle of more prompts than the lease holds restores nothing: each
capture drops the least recently used, the one the cycle needs next. With no `cap=` the lease asks and grows past
the floor, as far as the node has room.

The boot log prints `# rank R: flash ckpt aligned: ...`, and each request prints `flash ckpt request ... kept ...` or
`... restored ...` with the time.

Measured on one RTX 5090 (Flash's TP=2 geometry, 100 pages, 161 blocks): a capture
(161 blocks) takes 26-50 ms and a restore 21-25 ms through memory, an SSD file and a loopback KV node. Not measured on GB10, and the
end-to-end bit-exact check against no cache on a cluster has not run yet.

Limits:

- Checkpoints are not entries of the pool's block stash (`engine/kvman-rs` `stash.rs`); a `group=` store shares and
  resizes through its own index (above), with the same report rule.
- Each capture writes the whole prefix `[0, p)`, not only the rows past the entry it started from.
- `terminal` is not cold-exact (above).
- `GR8_GLM53F_KV_CHUNKS` (borrow-idle) still needs `--s2-sessions 1`.
- Several sessions with checkpoints: not measured on a cluster yet (the capture and restore are the one-session copies
  through the run's page list).

## Not yet

- Prefix reuse: the engine refuses it under `--serve`, and CP is experimental (above). `GR8_GLM53F_CACHE=aligned` (above) is the
  experimental Flash form of prefix reuse, with or without DFlash2. The DFlash2 drafter's taps are the mean of the four
  mHC streams after each tapped layer, in decode steps and in the prefill, which keeps the last 2,048 prompt positions
  for the drafter's prompt context. DFlash2 at TP=3 runs offline only.
- A 262K prefill's time and memory headroom on a GB10: the `CAP` above is computed, not measured.
