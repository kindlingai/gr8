# Knobs

The options of `./gr8 glm53`, the launchers in `scripts/` and the build. Each knob has a name
(`CAP`) and a flag (`--cap`): the flag is the name in lower case with `-` for `_`, except `--s2-port` for `S2PORT` and
`--g6-home` for `G6_HOME`. The engine's own environment (`G6_*`) and flags are in [docs/SETTINGS.md](docs/SETTINGS.md).

Where a value comes from, strongest first:

1. the command line: `./gr8 glm53 up --cap 262144`;
2. one config file: `@FILE` on the command line, else `$G6_CLUSTER`, else `./cluster.env`, else `config/cluster.env`
   (`--no-config` reads none). `./gr8 glm53` parses the file and does not source it: `KEY=value` lines, `#` comments, optional `export`, quotes
   stripped, and no `$` expansion;
3. the environment: `CAP=262144 ./gr8 glm53 up`;
4. the default.

`./gr8 glm53 up` prints every value and where it came from, and `./gr8 glm53 show-config` prints them without booting.

The handshake before the engines start compares the cluster options on every box: the TP, the shape, the peer list or
mentat group, the ports, every knob in the launch table below that reaches the engine, and the engine environment and
arguments they produce. Per-box values, such as paths and the proxy CPU, do not count. Any difference stops every
box, with the option and each box's value.

The engine does not inherit `G6_*` variables from the environment, and `./gr8 glm53 up` refuses to start while one is
set. Pass engine variables with `--g6env-x "NAME=value ..."` so that every box compares them.

The defaults in the launch table come from `scripts/engine_args.sh`, which the supervisor reads. Every TP runs the multi-session engine (`--serve`). A dash means that TP does not read the knob. `V2`, `V2_SLICES`, `V2_DENSE` and
`V2_WIDTHS` are paths, so each box may have its own. The handshake compares the variant's name and the sha256 of its
manifest instead. `LAYOUT` is a path too: the handshake compares the sha256 of the file.

## Launch

| variable | TP=4 | TP=6 | TP=3 | meaning |
|---|---|---|---|---|
| `CAP` | 485376 | 614400 | 307136 | KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory. Unset, it follows `SHARE_LIMIT`: each rank keeps the same KV memory (2,280 pages of 64 tokens at TP=4, 2,026 at TP=6, 1,941 at TP=3) |
| `KV_FORMAT` | fp8 | fp8 | fp8 | experimental. The KV latent row: `fp8` (656 B) or `int4` (416 B, INT4 after a Hadamard rotation). `int4` runs `bin/g6m4-int4` (in the image; `KV_FORMAT=int4 ./gr8 glm53 build` builds it), and its default `CAP` keeps fp8's memory a rank: 415,232 at TP=4 (2,390 pages a rank, measured against fp8 at 1,920; the fp8 default is 2,280 now), 345,088 at TP=3 and 857,600 at TP=6 (estimates). At TP=4 it decodes about 3% slower and prefills about 6% faster, and finds the same needles. `docs/KV4.md` |
| `SHARE_LIMIT` | 32768 | 32768 | 32768 | requests up to this many tokens (prompt + `max_tokens`) share decode steps; a longer one waits for the others and runs alone. Every rank holds the first `SHARE_LIMIT` tokens of a session and a share of the rest, so a higher limit leaves a shorter longest context: at TP=4, 65536 gives `CAP` 387072, 98304 gives 288768. A multiple of 64 from 4096; a limit off the 2048 doubling is a decode bucket boundary of its own |
| `KV_FORMAT` | fp8 | fp8 | fp8 | experimental. The KV latent row: `fp8` (656 B) or `int4` (416 B, INT4 after a Hadamard rotation). `int4` runs `bin/g6m4-int4` (in the image; `KV_FORMAT=int4 ./gr8 glm53 build` builds it), and its default `CAP` keeps fp8's memory a rank: 415,232 at TP=4 (2,390 pages a rank, measured against fp8 at 1,920; the fp8 default is 1,981 now), 345,088 at TP=3 and 857,600 at TP=6 (estimates). At TP=4 it decodes about 3% slower and prefills about 6% faster, and finds the same needles. `docs/KV4.md` |
| `SESS` | 8 | 8 | 8 | requests decoding at once (1-8); every reachable step graph is captured at boot (about 2.5 GiB at 8) |
| `REUSE` | 1 | 1 | 1 | 1 = `--reuse-prefix`: a request that continues an earlier one re-prefills only the new part. 0 = off |
| `KV_STORE` | off | off | off | experimental. Block stores below the KV pool (`--kv-store`, given once per store, or stores separated by `;` in a file or the environment), tried in order: `STORE,size=N(M\|G\|T)[,persist=on\|off][,keep=system\|sessions\|all][,group=NAME[,cap=N(M\|G\|T)][,weight=N]]`. `group=` (on `remote:`) makes the stores of a cluster one group lease that grows to `cap=` and gives to other clusters on the same node (`KV_SHARE_WHEN`). `keep=` is that store's `KV_KEEP`: `ssd:PATH,size=4G,persist=on,keep=system;remote:auto,size=4G,keep=all` keeps system prompts on the SSD and every other evicted run (with its tail past `SHARE_LIMIT`) on the KV node. STORE is `ssd:PATH`, local SSD offload (a folder on each box's own SSD: rank r uses `PATH/r<r>`), or `remote:HOST:PORT` / `remote:auto`, a KV node on another machine (`auto`: the one the mentat group announces, `./gr8 kv up`). An SSD store is cleared at every boot. With `persist=on` its system prefixes and the counts survive restarts while the model, layout, block format and engine build stay the same; a boot that served wrong output would hand them on, so persist only a store you trust. Sizes are per rank, G = 2^30. Needs `REUSE`. `engine/kvman-rs/REMOTE-TIER.md` |
| `KV_KEEP` | system | system | system | what the stores keep. `system`: each system prompt with its tool definitions (the span the front end sends; without one, the prompt's grid cells), once, after `KV_SYSTEM_AFTER` conversations started with it. `sessions`: those, plus the runs the pool evicts that continued their conversation at least once, stored on the system prefix they start with, a run past `SHARE_LIMIT` with each rank's shard of its tail. `all`: every run the pool evicts, for benchmarks. `system` suits most; `sessions` suits agent fleets larger than the KV pool. On the cluster's 8-agent replay (TP=4) `sessions` cut TTFT p50 from 234 s to 126 s writing 98 GB/h a rank, `all` was about the same, and `system` wrote 1.2 GB/h (TTFT p50 213 s); `engine/kvman-rs/REMOTE-TIER.md` has the table. In kvman-sim on the opencode captures `system` writes under 4 GB/h per rank and `all` 90-135 |
| `KV_SYSTEM_AFTER` | 2 | 2 | 2 | the conversations a system prefix must start before the stores keep it. A request that continues a live, retained or stored run counts toward nothing |
| `KV_SHARE_WHEN` | idle | idle | idle | a `KV_STORE` `group=` lease (a KV node shared by clusters): `idle` = the engine offers its blocks to other groups only while it has no session, so a give's compaction never pauses a turn; `always` = whenever the node asks. A group starting below its floor is served either way (`G6_KV_SHARE_WHEN`, `engine/kvman-rs/REMOTE-TIER.md` Sharing a node) |
| `KV_STORE` `cap=` (with `group=`) | none | none | none | the most one rank's group lease may grow to. Omit `cap=` for no cap (the engine rejects `cap=0`; internally cap 0 = no cap): the group grows into the node's free room under demand (an entry that does not fit, or the tier dropping entries), and gives back to other groups down to its floor (`size=`). Set a cap to keep one cluster from taking a shared node's free room that another cluster on it boots later or wants back (a later group opening on a full node asks the others down to their floors, and waits up to `--stall-s`). Cost of no cap: more node memory held by the busiest cluster, and a give's compaction when the other group asks (`KV_SHARE_WHEN` idle keeps it out of a turn). Before the D-331 M8 fix the node read an absent cap as cap = floor, so a group never grew. Measured effect: not measured on a box. `engine/kvman-rs/REMOTE-TIER.md` Sharing a node |
| `KV_WRITE_LIMIT` | off | off | off | a cap on store writes: at most this many page blocks per page of KV the engine computes (prefill and decode), so `0.25` writes at most a quarter of what the engine fills. Counted the same on every rank, never by the clock; a store refuses what would go past it |
| `KV_ROWS` | off | off | off | experimental. Remote attention rows: each rank's attention rows past `SHARE_LIMIT` go to a rows store as their pages fill, and decode reads back the rows its selection picks (`--kv-rows`): `STORE,size=N(M\|G\|T)[,local=PAGES][,pool=SLOTS][,place=PAGES][,fmt=fp8\|int4]`. STORE is `remote:auto` (a KV node the mentat group announces, odd ranks on its second rail, as `KV_STORE`), `remote:HOST:PORT` or `mem:`. `size` is each rank's lease; with `local` a session waits until its tail's records fit it. `local=PAGES`: release, a session's pages past `SHARE_LIMIT` cycle through that many local pages a rank, so its tail no longer needs local memory; `place=PAGES` then caps the main model's pages a rank, so `CAP` may pass what they hold. `pool` is the resident slots per indexer group (with `local` it defaults to what the sessions past `SHARE_LIMIT` that `place` holds can select in a step, and a smaller value admits fewer of them at once), `fmt=int4` stores 416 B rows for 656. Sets the engine's `G6_RT_*` ([docs/SETTINGS.md](docs/SETTINGS.md)) |
| `KV_TIER`, `KV_SHARED_AFTER` | | | | deprecated, for one release: `KV_TIER` is `KV_STORE` with `keep=evicted\|shared\|all` per tier, and maps to `KV_KEEP=all` when any tier kept evicted runs (the old default), else `system`; `KV_SHARED_AFTER` is `KV_SYSTEM_AFTER`. A warning says so |
| `DFLASH` | 1 with a drafter, else 0 | 0 | 0 (fixed) | 1 = the DFlash2 drafter (`--drafter`, or `dflash.r<r>.*` in the model set); 0 = MTP only. TP=6: the drafter's body runs whole on every rank (its 8 KV heads do not split six ways; packs converted with `--nr 6`, about 2.6 GB a rank, and 1.3 GB more with `DFL_BODY_I4=1`), drafting what TP=4 drafts. TP=3 refuses 1 |
| `DFLASH_SEL` | engine default (1) | – | – | 1 = the DFlash2 selector walk, 0 = the drafter's argmax |
| `DFLASH_K` | engine default (7) | – | – | verify at most the first k of the 7 drafts |
| `SPEC_ADAPT` | `DFLASH` | – | – | 1 = choose each round's draft count from step cost and acceptance |
| `SPEC_LEVELS` | all | – | – | the draft counts `SPEC_ADAPT` may choose, e.g. `7,5,3,2` |
| `SPEC_ACCEPT` | coupled | coupled | coupled | `rejection` = rejection sampling of drafts at temperature > 0 |
| `MTP` | 2 | 2 | 2 | MTP draft count (TP=4: for sessions that share steps, or with `DFLASH=0`; TP=6 and TP=3: every round) |
| `PF_OWN` | 4 | 3 | 1 | prefill attention ownership (`G6_PF_OWN`); 3 also loads `packs/qo.pack`. TP=6 with `PF_OWN=1` needs no QO pack. TP=5: 3 |
| `QP_ROLL` | 2 | – | – | `PF_OWN=4`: rolling pair-weight slots (`G6_PF_QP_ROLL`) |
| `QP_CTX` | `CAP` | – | – | `PF_OWN=4`: QO key-table size; a longer request runs OWN=2 |
| `QP_EARLY` | 1 | – | – | `PF_OWN=4`: send the counterpart's partial before the own rows' o_proj |
| `V2`, `V2_SLICES` | unset | unset | unset | exlr8 v2 experts from their raw width files: the width files and the index slices |
| `VARIANT` (`--variant`) | `home` | `home` | `bit225` | the exlr8 v2 width manifest: `bit225`, `bit243`, `bit260`, `bit325`, `bit338` (routed experts at about 2.25-3.38 bpw, spent where they lower KL against bf16 the most), `home` (3.25 bpw), `down1` (about 2.25), `up1` (about 4.0), or a manifest path. A name is `exlr8-manifest-<name>.json` in the model set or in `V2` (directly or under `manifests/`). At TP=3, a release without `bit225` boots `down1`. Without `V2`, only `home` |
| `V2_WIDTHS` | unset | unset | unset | the width manifest as a path: the older spelling of `--variant <path>`. Giving both is refused |
| `DFL_REUSE` | 2 | – | – | 1 = a request whose prefix-reuse hit skipped into the DFlash2 drafter's window keeps the drafter; 2 = also a request alone on MTP takes it (`G6_DFL_REUSE`). Defaults to 2 when serving DFlash2 with MTP. Needs `DFLASH=1` |
| `DFL_REUSE_MIN` | unset (256) | – | – | `DFL_REUSE`: context rows the drafter must hold before it drafts; fewer = MTP warm-up rounds (`G6_DFL_REUSE_MIN`) |
| `MTP_KSTOP` | on (2) when serving w/ DFlash2 + MTP 2+ | – | – | a request's solo MTP rounds chain and verify the drafts the MTP head is confident in: 1 = thresholds (`MTP_KSTOP_TAU`), 2 = the drafter choice's MTP estimates (`G6_MTP_KSTOP`). Needs `MTP` 2+; without `DFLASH` (TP=6, TP=3): 1 only, for a request decoding alone, `MTP_KSTOP_KMAX` at most `MTP` |
| `MTP_KSTOP_KMAX` | unset (`MTP`) | – | – | `MTP_KSTOP`: the most drafts a solo MTP round chains, 1..3 (`G6_MTP_KSTOP_KMAX`) |
| `MTP_KSTOP_TAU` | unset (`0,0.2,0.5`) | – | – | `MTP_KSTOP=1`: thresholds t1,t2,t3 on the drafts' cumulative probability (`G6_MTP_KSTOP_TAU`) |
| `DRAFT_PAR` | on (2) when serving w/ DFlash2 + MTP | – | – | a solo DFlash2 round's drafter step on its own stream under MTP's first form, 1, or 2 = drafting steps at high priority (`G6_DRAFT_PAR`; bit-exact). Needs `DFLASH=1` |
| `QA_SPLIT` | on (2) when TP>1 | unset (0) | unset (0) | decode: the layer-input GEMVs split by output tiles across the ranks and all-gathered (`G6_QA_SPLIT`, 1 or 2; bit-exact). Measured at TP=4 only |
| `IDXQ_SPLIT` | on (1) when TP>1 | unset (0) | unset (0) | decode: the indexer's wq_b split the same way (`G6_IDXQ_SPLIT`; bit-exact) |
| `MTP_SPLIT` | on (1) when TP>1 + MTP | unset (0) | unset (0) | MTP draft steps: eh_proj and the draft layer's input GEMVs split the same way (`G6_MTP_SPLIT`; bit-exact) |
| `DRAFT_HEAD_I4` | 1 (with `MTP` 1+) | unset (0) | unset (0) | 1 = the MTP draft's lm_head on INT4 g32 copies made at boot (`G6_DRAFT_HEAD_I4`; drafts only, +134 MB a rank at TP=4) |
| `DFL_HEAD_I4` | unset (0) | – | – | 1 = the DFlash2 drafter's lm_head on the same INT4 records (`G6_DFL_HEAD_I4`; drafts only). Needs `DFLASH=1` |
| `DFL_BODY_I4` | unset (0) | – | – | 1 = the DFlash2 drafter's body GEMVs on INT4 g32 copies (`G6_DFL_BODY_I4`; drafts only, +382 MB a rank). Needs `DFLASH=1` |
| `ROUTE_OVL` | 1 | unset (0) | unset (0) | decode MoE layers of 2+ rows: the trellis shared expert on a forked stream during the router, 1 = fc1 + fc2, 2 = fc1 (`G6_ROUTE_OVL`; bit-exact). Measured at TP=4 only |
| `ATTN_SC2` | 4 | unset (0) | unset (0) | decode attention scores with S = 1, 2, 4 or 8 key slots per warp (`G6_ATTN_SC2`; bit-exact) |
| `ATTN_SC2_TR` | 2 | unset (0) | unset (0) | with `ATTN_SC2`: the transpose reduction, 1 or 2 = eager (`G6_ATTN_SC2_TR`; bit-exact) |
| `ATTN_CTX2` | 4 | unset (0) | unset (0) | decode attention context kernel variant 1..4 (`G6_ATTN_CTX2`; bit-exact) |
| `ATTN_CTX3` | 2 | unset (0) | unset (0) | decode attention context with one test per batch of slots, 1..4 (`G6_ATTN_CTX3`; bit-exact) |
| `ATTN_SEL2` | 16 | unset (0) | unset (0) | decode index selection's final sort as a rank by count over 8, 16, 32 or 64 CTAs per row (`G6_ATTN_SEL2`; bit-exact) |
| `IDX_GRP` | 16 | unset (0) | unset (0) | S2 batched decode: the indexer reads each key tile once for up to 1..64 rows of a session (`G6_IDX_GRP`; bit-exact) |
| `ATTN_QT` | 1 | unset (0) | unset (0) | 1 = q and rope stored lane-transposed for the scores kernel (`G6_ATTN_QT`; needs `ATTN_SC2` 2 or 4; bit-exact) |
| `ATTN_FRONT` | 1 | unset (0) | unset (0) | 1 = a decode step's first attention kernels in one launch (`G6_ATTN_FRONT`; bit-exact) |
| `IDX_VEC` | 1 | unset (0) | unset (0) | 1 = the decode indexer's vector loads (`G6_IDX_VEC`) |
| `DEC_BK` | 512 | unset (2048) | unset (2048) | decode context bucket in tokens: the step graphs are captured per bucket (`G6_DEC_BK`; 0 = the engine default 2048) |
| `S2_ADAPT` | 1 (with `MTP` 1+) | unset (0) | unset (0) | S2: each round's MTP draft count from the sessions' accept rates and step costs (`G6_S2_ADAPT`; drafts only; about 1 GiB more step graphs) |
| `S2_DECODE_PF` | 4096 | unset (0) | unset (0) | S2: while a session decodes, a prefill quantum takes at most this many rows on one lane (`G6_S2_DECODE_PF`; a multiple of 64, at most `--prefill`) |
| `S2_ADMIT_HOLD` | 1 | unset (0) | unset (0) | S2: a burst of queued requests is admitted at one boundary instead of one per notice (`G6_S2_ADMIT_HOLD`) |
| `PF_SELX_CTX` | 53248 | unset (0) | unset (0) | prefill index tiles at context >= this select with selx instead of SEL32, same ids in the same order (`G6_PF_SELX_CTX`; 0 = SEL32) |
| `ROUTE_OVL` | unset (0) | unset (0) | unset (0) | decode MoE layers of 2+ rows: the trellis shared expert on a forked stream during the router, 1 = fc1 + fc2, 2 = fc1 (`G6_ROUTE_OVL`; bit-exact). Measured at TP=4 only |
| `ATTN_SC2` | unset (0) | unset (0) | unset (0) | decode attention scores with S = 1, 2, 4 or 8 key slots per warp (`G6_ATTN_SC2`; bit-exact). At TP=6 / TP=3 also the partial head groups of 10 / 11 / 21 / 22 heads (`G6_ATTN_SC2_PART`, default 1) |
| `ATTN_SC2_TR` | unset (0) | unset (0) | unset (0) | with `ATTN_SC2`: the transpose reduction, 1 or 2 = eager (`G6_ATTN_SC2_TR`; bit-exact) |
| `ATTN_CTX2` | unset (0) | unset (0) | unset (0) | decode attention context kernel variant 1..4 (`G6_ATTN_CTX2`; bit-exact) |
| `ATTN_CTX3` | unset (0) | unset (0) | unset (0) | decode attention context with one test per batch of slots, 1..4 (`G6_ATTN_CTX3`; bit-exact) |
| `ATTN_SEL2` | unset (0) | unset (0) | unset (0) | decode index selection's final sort as a rank by count over 8, 16, 32 or 64 CTAs per row (`G6_ATTN_SEL2`; bit-exact) |
| `IDX_GRP` | unset (0) | unset (0) | unset (0) | S2 batched decode: the indexer reads each key tile once for up to 1..64 rows of a session (`G6_IDX_GRP`; bit-exact) |
| `ATTN_QT` | unset (0) | unset (0) | unset (0) | 1 = q and rope stored lane-transposed for the scores kernel (`G6_ATTN_QT`; needs `ATTN_SC2` 2 or 4; bit-exact) |
| `ATTN_FRONT` | unset (0) | unset (0) | unset (0) | 1 = a decode step's first attention kernels in one launch (`G6_ATTN_FRONT`; bit-exact) |
| `DSEL_MODEL` | unset (0) | – | – | 1 = the drafter choice prices both drafters from `DSEL_TABLE` and a request's MTP rounds as the confidence stop runs them (`G6_S2_DSEL_MODEL`; drafts only: the target decides every token). Needs `DFLASH=1` |
| `DSEL_TABLE` | unset | – | – | `DSEL_MODEL`: the cost table, a `key:value` list (`v1`..`v8`, `d`, `m`, `e`, `p`, optional `g`, `k`, `n`, `w`, `r`) or a file holding one, measured on the cluster with `tools/s2_dsel_table.py` (`G6_S2_DSEL_TABLE`) |
| `NGRAM` | unset | – | – | a request decoding alone with the DFlash2 drafter also drafts from this n-gram table file: a round verifies the table's chain instead of the drafter's drafts when its calibrated probabilities promise more tokens per verify ms (`G6_NGRAM`; drafts only: the target decides every token; #56). Write the file with `tools/ngram/ngram.py export`. Needs `DFLASH=1` |
| `NGRAM_KMAX` | unset (`--dflash-k`) | – | – | `NGRAM`: the longest chain a round verifies, 1..`--dflash-k` (`G6_NGRAM_KMAX`) |
| `NGRAM_KMIN` | unset (2) | – | – | `NGRAM`: the shortest chain worth a round, 1..`NGRAM_KMAX` (`G6_NGRAM_KMIN`) |
| `NGRAM_MARGIN` | unset (0.1) | – | – | `NGRAM`: a chain must promise this much more tokens per verify ms than the drafter's drafts (`G6_NGRAM_MARGIN`) |
| `MTP_KSTOPB` | unset (0) | unset (0) | unset (0) | 1 = in rounds that several sessions share, each session verifies its own count of the `MTP` drafts, from the MTP head's confidence (`G6_MTP_KSTOPB`; drafts only: the target decides every token). Captures a main graph per width the rule reaches |
| `MTP_KSTOPB_KMIN` | unset (0) | unset (0) | unset (0) | `MTP_KSTOPB`: the least drafts a session verifies, 0 or 1 (`G6_MTP_KSTOPB_KMIN`) |
| `MTP_KSTOPB_TAU` | unset (`0.6,0.88`) | same | same | `MTP_KSTOPB`: thresholds t1,t2,... (up to 7) on each session's cumulative draft probability, at every session count (`G6_MTP_KSTOPB_TAU`) |
| `MTP_KSTOPB_TAU_S` | unset | unset | unset | `MTP_KSTOPB`: thresholds per session count, `n:t1,t2,...` entries separated by `/`, e.g. `1:0.7,0.7/2:0.7,0.8` (`G6_MTP_KSTOPB_TAU_S<n>`) |
| `L2PF` | unset (0) | unset (0) | unset (0) | decode: a paced L2 prefetch of the weights the step reads next, on a side stream in the windows where DRAM is idle; window mask 1 = MLP all-reduce wait, 2 = gather waits, 4 = attention, 8 = attention all-reduce wait (`G6_L2PF`; bit-exact). Measured at TP=4 only, as 15 |
| `CP_STEP_ROWS` | unset (0) | unset (0) | unset (0) | S2: a decode step past `SHARE_LIMIT` carries at most this many rows, and the CP fabric is sized for it instead of for the widest S2 step (`SESS` x (`MTP` + 1) = 24 rows, 1.358 GB pinned per rank at TP=4; 8 rows: 0.453 GB). While a session past the limit decodes and the sessions' rows do not fit, they take turns, each once per cycle. At least the rows of a solo step (the boot names the minimum, 8 with DFlash2) (`G6_CP_STEP_ROWS`) |
| `CP_STEP_SEG` | unset (0) | unset (0) | unset (0) | with `CP_STEP_ROWS`: 1 = a round past `SHARE_LIMIT` takes only sessions past it, so a shorter session never runs the CP attention because a long one is resident (`G6_CP_STEP_SEG`) |
| `CP_ONECHAIN` | unset (0) | – | – | 1 = past `SHARE_LIMIT`, each rank runs the other ranks' head groups over its keys as one attention call instead of one per group (`G6_CP_ONECHAIN`; bit-exact). TP=4 only (the engine refuses it elsewhere), not with `G6_ATTN_FLIP` |
| `KVB` | unset (bf16) | unset (bf16) | unset (bf16) | `fp8` = kv_b (UK / UV) as the release's block FP8 bytes and scales instead of their BF16 (`G6_KVB`; half the bytes, bit-exact). Needs `V2_DENSE` (a release boot) |
| `ATTN_CTX3_SP` | unset (0) | unset (0) | unset (0) | with `ATTN_CTX3`: 64 or 128 key slots per context partition instead of 256 (`G6_ATTN_CTX3_SP`; deterministic, not bit-exact) |
| `ATTN_POSIDX` | unset (0) | unset (0) | unset (0) | 1 = decode steps in the buckets up to 2048 list the keys by position instead of running the indexer, which selects all of them there (`G6_ATTN_POSIDX`; deterministic, not bit-exact) |
| `PF_GEMMF32` | unset (0) | unset (0) | unset (0) | 1 = with `PF_OWN=3`, prefill's row-owner QO o_proj on the FP32-epilogue GEMM (`G6_PF_GEMMF32`; byte-exact). TP=6 K3: +1.0 % cold prefill (1693.6 -> 1710.4, one point, inside the 2 % band). Set for the boot |
| `MTP_LASTROW` | unset (0) | unset (0) | unset (0) | 1 = an MTP first-form draft step routes only each session's last row through layer 78's routed experts (`G6_MTP_LASTROW`; same caches, drafts and log-probabilities). Skips the routed experts of every row but the last; no end-to-end number. In the TP=6 K3 recipe |
| `DRAFT_HEAD2` | unset (0) | unset (0) | unset (0) | experimental: 1 = the MTP draft head in two stages, a 2-bit screen then the exact MXFP8 logits of its best 64 (`G6_DRAFT_HEAD2`; drafts only, +74.5 MB a rank at TP=4); 2 = also the MXFP8 check (measurement). No end-to-end number. In the TP=6 K3 recipe |
| `MTP_L2PF` | unset (0) | unset (0) | unset (0) | with `G6_L2PF`: the paced L2 prefetch in MTP draft steps, 1 = the draft step, 2 = the head waits, 3 = both (`G6_MTP_L2PF`; same bytes). TP=6: about 0.1 ms per ~39 ms round, below speedbench resolution. In the TP=6 K3 recipe (3) |
| `MTP_KSTOPB_WIDTHS` | unset | unset | unset | with `G6_MTP_KSTOPB`: the main widths captured besides the multiples of `MTP` + 1, `w1,w2,...` (`G6_MTP_KSTOPB_WIDTHS`); fewer widths = less graph memory (about 115 MiB a width a rank at TP=4) and a shorter boot. |
| `TR3_W2ALL` | 1 | unset (0) | unset (0) | 1 = the window-2 fc1 split (KS1x2, WG1) on every MoE layer at M > 1, not only the layers the k2 gate picks (`G6_TR3_W2ALL`; bit-exact). TP=4: prose +1.3 %, c4 +2.7 to +3.4 %, exact; not measured at TP=5 or TP=6 |
| `QO_CACHE` | unset | unset | unset | `PF_OWN=3`: a directory; the first boot writes each layer's full-head q_b / o_proj / UK / UV tensors to it and later boots read them (`G6_QO_CACHE`; same bits). TP=5: second boot /health at 46 s against 199 s without; 13 GB of disk a box. In the TP=5 T5X recipe |
| `PF_ISGRT_RG` | unset (8) | unset (8) | unset (8) | prefill index scores: rows per CTA, a multiple of 8 (`G6_PF_ISGRT_RG`; same bits). TP=5: 16 = 157K TTFT about -2 %; 32 with `PF_ATTN_ROWS=512` = 32K prefill +2.2 %, 157K TTFT -2.8 %. In the TP=5 T5X recipe (32) |
| `PF_ATTN_ROWS` | 1024 | 1024 | 1024 | prefill attention chunk rows inside a sub-batch (`G6_PF_ATTN_ROWS`). TP=5: 512 = about 0.5 GB more free memory. In the TP=5 T5X recipe (512) |
| `PF_F12` | unset (1 when built with `PF_F12`) | unset | unset | 0 = the stock prefill fc1 instead of the fat-CTA `k_pf_fc1f` (`G6_PF_F12`; byte-exact). Effect not measured on its own (TP=5 T5X added it with `PF_OWN=3` and `PF_GEMMF32=1` in one step) |
| `ATTN_P11` | unset (0) | unset (0) | unset (0) | 1 = the P11 decode attention tile retune; takes precedence over `ATTN_SC2`, `ATTN_CTX2`, `ATTN_CTX3` (`G6_ATTN_P11`; read before graph capture). Effect not measured |
| `DTR` | unset (1) | unset (1) | unset (1) | 0 = the int4 / MXFP8 records instead of the exlr8 release's dense trellis matrices, where the layer has both (`G6_DTR`; A/B). Effect not measured; inert on pack boots |
| `PF_AOP` | unset (1 when built with `PF_AOP`) | unset | unset | 0 = no fc1 A-operand share in prefill (`G6_PF_AOP`; byte-exact). Effect not measured |
| `NOHASH` | 0 | 0 | 0 | 1 = skip the sha256 check of every pack tensor at load (run once without it first) |
| `G6ENV_X` | empty | empty | empty | more engine environment for every rank (`"A=1 B=2"`) |
| `G6ARGS_X` | – | empty | empty | more engine arguments for every rank |
| `READY_TMO` | 1800 | 1800 | 1800 | seconds to wait for the ranks to load |
| `FRONT_BIN` | `bin/g6front-rs` | same | same | the Rust front end binary |
| `PY` | `$G6_HOME/venv/bin/python` | same | same | the python that runs the shim |
| `S2PORT` | 7341 | 7341 | 7341 | the multi-session control port on the head |

## `./gr8 glm53`

`./gr8 glm53f` takes the same options for GLM-5.3-Flash: TP 4 or 2, an fp8 KV build, no drafter, no prefix reuse, and
`CAP` 262208 at either TP, a 262,144-token request (`tools/launch/models.py`, `flash_defaults` in
`scripts/engine_args.sh`, [docs/FLASH.md](docs/FLASH.md)). `SESS` defaults to 1 there; 2 to 8 share decode steps and
the `CAP` pool, each session with its own KDA state slots, while `SESS` x (`MTP` + 1) is at most 32.

A Flash box can lend what its serve leaves to a KV node beside it (D-331 M7; `donate` in `tools/launch/supervisor.py`):

| name | flag | default | meaning |
|---|---|---|---|
| `FLASH_KV_DONATE_GIB` | `--flash-kv-donate-gib` | 0 | once the serve has booted, each Flash box starts `./gr8 kv up --gb N --lazy-lock --ledger out/kv/ledger` with N = this, or less when MemAvailable then - `FLASH_KV_GROWTH_GIB` - `KV_MIN_FREE_GB` is less (`flash_donor_budget`, whole GiB). No budget, or a node that fails, leaves the serve as it is; the serve's stop stops the node. 0 = no node. Measured effect: not measured |
| `FLASH_KV_GROWTH_GIB` | `--flash-kv-growth-gib` | none | with `FLASH_KV_DONATE_GIB`: the GiB the serve may still allocate after boot. Required: the Flash serve's growth after boot is not measured yet |
| `FLASH_KV_GROUP` | `--flash-kv-group` | none | with `FLASH_KV_DONATE_GIB`: the mentat group the KV node joins, the cluster that borrows from it (the GLM-5.3 serve's), not the Flash serve's. Required: a donation without it is refused at boot |
| `FLASH_CAP_POLICY` | `--flash-cap-policy` | `protected` | `protected`: the configured `CAP`'s memory is never lent; the host ledger counts only what is lent. `borrow-idle` (needs `FLASH_KV_DONATE_GIB`): the Flash KV pool in chunks (`G6_FLASH_KV_CHUNKS=1`); once the serve has been idle `FLASH_KV_IDLE_S`, every chunk but one (16,384 tokens) is lent to the KV node, and a request that needs more waits for their recall before its admission (its TTFT includes the recall). The ledger books the lendable chunks as the GPU's. Measured effect: not measured |
| `FLASH_KV_IDLE_S` | `--flash-kv-idle-s` | 60 | with `FLASH_CAP_POLICY=borrow-idle`: idle seconds before a lend (`G6_FLASH_KV_LEND_MS`). Not measured |

| name | flag | default | scope | meaning |
|---|---|---|---|---|
| `GR8_MODEL` | `--gr8-model` | `glm53` | cluster | the model: `glm53` or `glm53f`. `./gr8 <model>` sets it |
| `TP` | `--tp` | the number of peers | cluster | 4, 6 or 3 (glm53f: 4 or 2); needed with mentat |
| `FLASH_CACHE` | `--flash-cache` | `off` | cluster | glm53f, experimental: `terminal` = `aligned` plus a checkpoint at the end of every request, **NOT cold-exact** (docs/FLASH.md). `aligned` = session checkpoints, so a prompt that starts with an earlier prompt's tokens is prefilled from that prompt's last 8,192-row grid point, bit-exact (`G6_FLASH_CACHE`, [docs/FLASH.md](docs/FLASH.md)). The store is the one `KV_STORE` (no `persist=on`), else memory. On glm53f, `KV_STORE` needs it. The prefill checkpoint is taken at the boundary after the quantum that reaches the grid point, once that boundary's events have committed, so a store Resize that lands there comes first |
| `FLASH_STATE_BUDGET_MIB` | `--flash-state-budget-mib` | 8192 | cluster | `FLASH_CACHE` with no `KV_STORE`: the in-memory checkpoint store a rank, in MiB (`G6_FLASH_STATE_BUDGET_MIB`) |
| `SHAPE` | `--shape` | `mesh` | cluster | `mesh` (a switch), or `ring` (TP=3 or 4, no switch). With `--peers`, each box's f0 port is cabled to the next box's f1 port, in `--peers` order. With mentat (mentatd 0.24.0 or later on every box), the boxes claim a ring from mentat, which solves it once over the cables its probes find, on any cabling. They take the rank order and their `--ring-devs` from that one view, and refuse the boot, naming the pair, when the links form no ring. `--head` and the model set's offers turn the ring. Takes no `--ips` |
| `PEERS` | `--peers` | none | cluster | the boxes' LAN addresses in rank order, the same on every box. The first is the head. `HOSTS` in an old config file is read as `PEERS` |
| `MENTAT` | `--mentat` | `localhost` | box | find the boxes through the local mentat daemon (the default when `PEERS` is unset) |
| `GROUP` | `--group` | `gr8-<model>` | cluster | the mentat group the boxes join |
| `HEAD` | `--head` | the lowest address | cluster | mentat mode: the head's LAN address |
| `IPS`, `IPS2` | `--ips`, `--ips2` | discovered | cluster | each rank's rail-0 and rail-1 fabric address. Without them each box finds its own two RoCE v2 addresses |
| `MODEL` | `--model` | none | box | the exlr8 release: `https://huggingface.co/<org>/<repo>[@rev]` (each box fetches its part into `MODEL_CACHE`), or a local copy of it (`dense/`, `experts/`, `index-slices/`, `manifests/`, the tokenizer files) used in place, on a local disk or a shared mount such as NFS. The engine plans every rank's weights from the release's files (`V2_DENSE`) |
| `V2_DENSE` | `--v2-dense` | the release | box | the exlr8 release directory whose `dense/` files give every non-expert weight (`g6m4 --v2-dense`) |
| `V2_DENSE_FORM` | `--v2-dense-form` | `trellis` | cluster | with `V2_DENSE`: `trellis` keeps q_b, o_proj and the dense MLP as the release's trellis entries for the dense trellis kernels, so the weights are the release's own. `rounded` is the fallback: it rounds them to the MXFP8 and int4 forms of a converted pack |
| `LAYOUT` | `--layout` | the formulas | box | experimental. With `V2_DENSE`: a layout file that says which pieces each rank holds (`bin/g6layout`, `engine/loader-rs/src/layout.rs`). Rank r holds what the layout gives its r-th member, so list `PEERS` in the layout's member order. `bin/g6layout plan` makes one from the current layout, keeping each box's pieces where it can. A box reads the slice rows of halves the formulas give other ranks, so it needs every rank's index slices at the target TP: a local release (`--model DIR`) must hold them (`index-slices/tp<N>/k<K>/rank*.safetensors`), and with `--model https://...` the box's cache fetches them and the halves the layout gives its rank. Halves the cache already holds, from another TP or layout, are not fetched again. GLM 5.3 only: GLM-5.3-Flash plans the formulas |
| `DRAFTER` | `--drafter` | none | box | TP=4: the DFlash2 drafter's repo, `https://huggingface.co/incoai/GLM-5.3-DFlash2`. Each box fetches it and converts its rank into `MODEL_CACHE`. Its repo and commit go into the id the boxes compare |
| `DFLASH_DIR` | `--dflash-dir` | none | box | a directory holding `dflash.r<r>.*`, when it is not the model set's |
| `RANK` | `--rank` | none | run | `prepare`: the rank to fetch, when this box cannot tell (mentat mode) |
| `MODEL_CACHE` | `--model-cache` | none | box | with an https:// `MODEL`: this box's local directory for its rank's part of the release, and for the converted drafter |
| `MODEL_DIR` | `--model-dir` | none | box | the tokenizer files, when the release directory has none |
| `G6_HOME` | `--g6-home` | the repo checkout | box | working directory: `bin/`, `out/`, `s2/`, `run/` |
| `PROXY_CPU` | `--proxy-cpu` | 19 | box | CPU core of this rank's fabric proxy thread. The other proxies take the 7 cores below it, and the ring's relay thread the one below those |
| `RING_DEVS` | `--ring-devs` | from mentat's probes, or the GB10 CX-7 names | box | `--shape ring`: this box's RDMA devices as relay-out,next,relay-in,prev. With mentat, the devices behind its addresses toward the next and previous ranks (`/sys/class/infiniband`); with `--peers`, `rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`. An explicit list wins |
| `NODE_IP` | `--node-ip` | found | box | this box's LAN address, when several peers are addresses of this box |
| `CONTROL_PORT` | `--control-port` | 7340 | cluster | the supervisors' TCP port on the LAN (handshake, status, logs, down) |
| `PORT` | `--port` | 8100 | cluster | the OpenAI API on the head |
| `DEFAULT_MAX_TOKENS` | `--default-max-tokens` | 0 | box | the head's front end: `max_tokens` for a chat request that names none. 0 = what keeps the request within `SHARE_LIMIT`, but at least 8,192, and never more than the room the context leaves; another value is used as it is |
| `DEFAULT_REASONING_EFFORT` | `--default-reasoning-effort` | none | box | the head's front end: `reasoning_effort` for a chat request that names none: `minimal`, `low`, `medium`, `high`, `xhigh` or `max`, which the template hears as the nearest level it names (README, "Thinking"). Unset = the template's default (`max` on GLM 5.3). `none` is refused: GLM 5.3 cannot turn thinking off |
| `DEFAULT_THINKING_BUDGET` | `--default-thinking-budget` | 0 | box | the head's front end: `thinking_budget` for a chat request that names none, or names 0 or null (TensorFold's rule). The front end closes the think block at the reply's Nth token. 0 = no limit. A request with `logprobs` gets no default budget |
| `ALLOW_THINKING_OFF` | `--allow-thinking-off` | 0 | box | experimental. 1 = the head's front end takes `enable_thinking: false` and `reasoning_effort: "none"` instead of refusing them: the prompt drops its `Reasoning Effort` turn and closes the think block, so the reply starts on content. GLM 5.3's official template always opens `<think>`; a closed block gave degenerate output on an earlier serving stack, so it is off by default (`frontend/README.md`) |
| `GRAMMAR` | `--grammar` | 1 | cluster | 1 = the engine masks `response_format` `json_object` / `json_schema` replies with their XGrammar grammar from the think close on, so the reply's content is valid JSON of the schema (each rank masks its own vocabulary slice; GLM-5.3 only: a Flash engine gets no `--grammar-dir`, and an engine built without XGrammar serves without grammars. `scripts/build.sh` finds XGrammar through the front end's xgrammar crate). The front end checks the reply either way. PROTOCOL.md `grammar` |
| `GRAMMAR_DIR` | `--grammar-dir` | the model set's tokenizer | box | with `GRAMMAR=1`: the directory whose `tokenizer.json` and `generation_config.json` the grammars compile against |
| `JOIN_TIMEOUT` | `--join-timeout` | 600 | run | seconds for the mentat group to fill, for a worker to reach the head, and on the head from the first hello to the last (0 = forever) |
| `CHECK` | `--no-check` | 1 | run | 0 = skip this box's checks (the model files are checked anyway) |
| `G6_CLUSTER` | | none | | the config file when there is no `@FILE` |

The launcher does not run the rank health probe: run `scripts/rank_health.sh` with the same options before a boot if
you want it (`PROBE_ARGS` sets its thresholds, `tools/health/README.md`).

## `./gr8 kv`

A KV node: a box that locks some of its memory and serves it as KV page blocks to the cluster, over RDMA where both
ends have a RoCE v2 address and TCP otherwise (`engine/kvman-rs/REMOTE-TIER.md`). The cluster uses it through `KV_STORE` and `KV_ROWS` with `remote:auto`.
`./gr8 kv up --gb N` starts it on this box. The node joins the mentat group as an agent without GPUs, with the service
`gr8-kv`, and the model's election counts only agents with GPUs. `down`, `status` (`status all`: every node the group
announces) and `logs [-f]` work as for the model. The options come from the same places as above. A shared config
file is read for these keys only.

| name | flag | default | meaning |
|---|---|---|---|
| `KV_GB` | `--gb` | none | GiB of this box's memory to lock and serve. RLIMIT_MEMLOCK must allow it (`ulimit -l`; root, or `systemd-run -p LimitMEMLOCK=infinity`) |
| `KV_PORT` | `--port` | 7350 | the node's TCP port; RDMA connections are set up through it |
| `KV_BIND` | `--bind` | `0.0.0.0` | the address it listens on |
| `KV_MIN_FREE_GB` | `--min-free-gb` | 10 | refuse to start when locking `KV_GB` would leave less MemAvailable than this; with `KV_LAZY_LOCK=1`, refuse each lease that would |
| `KV_LAZY_LOCK` | `--lazy-lock` | 0 | 1 = lock a lease's memory only while the lease lives (given back to the box when it ends), and refuse leases while MemAvailable is under `KV_MIN_FREE_GB`: a node on a box that also serves. 0 = the whole `KV_GB` is locked at start. Measured effect: not measured |
| `KV_GUARD_EVERY_S` | `--guard-every-s` | 1 | with `KV_LAZY_LOCK=1`: how often the node reads MemAvailable (0 = at each lease only). Measured effect: not measured |
| `KV_LEDGER` | `--ledger` | none | with `KV_LAZY_LOCK=1`: the host ledger of a box shared with a Flash rank (`kvman-ledger`, `engine/kvman-rs/src/ledger.rs`). A lease is refused when the node would hold more than the ledger credits it. `FLASH_KV_DONATE_GIB` sets it |
| `KV_RDMA` | `--no-rdma` | 1 | 0 = TCP only |
| `KV_BIN` | `--bin` | `bin/kvman-remote` | the node binary (`./gr8 glm53 build` puts it in `bin/`) |
| `MENTAT` | `--mentat` | `localhost` | the daemon to announce through; `none` = do not announce |
| `GROUP` | `--group` | `gr8-glm53` | the mentat group it joins |
| `NODE_IP` | `--node-ip` | mentat's | the LAN address to announce |
| `G6_HOME` | `--g6-home` | the repo checkout | the node keeps `out/kv/` here: `kv.log`, `state.json`, `ready.json` |

## Build

| variable | script | meaning |
|---|---|---|
| `G6_WIDTHS` | `build.sh` | the TP widths g6m4 and m4sim run, default `1 2 3 4 6 8`; fewer is a shorter build |
| `G6_NR` | `build.sh`, `c1_tests.sh` | `build.sh`: one width, as `G6_WIDTHS=N`. `c1_tests.sh`: rank count of the tests, 4, 3, or unset for 6 |
| `JOBS` | `build.sh` | compiles at once, default the CPU count |
| `BUNDLE_LIBS` | `build.sh` | 1 = copy cuBLAS into `bin/lib` for boxes without the CUDA toolkit |
| `ARCH` | `build.sh`, `c1_tests.sh` | GPU architecture, default `121a` (GB10) |
| `EXTRA` | `build.sh`, `c1_tests.sh` | more `-D` flags |
| `KV_FORMAT` | `build.sh`, `./gr8 glm53 build` | `int4` = build `bin/g6m4-int4` (the 4-bit KV latent, `-DG6_KV4=1`) only. `./gr8 glm53 build` builds it after the fp8 engine when `KV_FORMAT=int4` |

## Local steering profile

Profile loading is explicit. See [local steering](docs/STEERING-LOCAL.md) for identity checks and validation limits.

| key | flag | default | scope | meaning |
|---|---|---|---|---|
| `STEERING_PROFILE` | `--steering-profile` | unset | box | verified schema-v1 profile directory, or `hf:default` / `hf:NAME` from the model release |
| `STEERING_FORCE` | `--steering-force` | unset | cluster | force qualification requests to this positive FP32 scale, at most 4 |
| `STEERING_MODEL_REVISION` | `--steering-model-revision` | unset | cluster | profile's exact model revision assertion; checked against config metadata where present |
| `STEERING_TOKENIZER_REVISION` | `--steering-tokenizer-revision` | unset | cluster | profile's exact tokenizer revision assertion; checked where present |
| `STEERING_CHAT_TEMPLATE_SHA256` | `--steering-chat-template-sha256` | unset | cluster | must match the actual served template bytes |
| `STEERING_WEIGHT_SET` | `--steering-weight-set` | resolved model | cluster | content identity; explicit mismatches fail |
| `STEERING_CHAT_TEMPLATE` | `--steering-chat-template` | resolved model | box | actual served template file, resolved by the supervisor |
