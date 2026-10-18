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
manifest instead.

## Launch

| variable | TP=4 | TP=6 | TP=3 | meaning |
|---|---|---|---|---|
| `CAP` | 485376 | 614400 | 307136 | KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory. Unset, it follows `SHARE_LIMIT`: each rank keeps the same KV memory (2,280 pages of 64 tokens at TP=4, 2,026 at TP=6, 1,941 at TP=3) |
| `KV_FORMAT` | fp8 | fp8 | fp8 | experimental. The KV latent row: `fp8` (656 B) or `int4` (416 B, INT4 after a Hadamard rotation). `int4` runs `bin/g6m4-int4` (in the image; `KV_FORMAT=int4 ./gr8 glm53 build` builds it), and its default `CAP` keeps fp8's memory a rank: 415,232 at TP=4 (2,390 pages a rank, measured against fp8 at 1,920; the fp8 default is 2,280 now), 345,088 at TP=3 and 857,600 at TP=6 (estimates). At TP=4 it decodes about 3% slower and prefills about 6% faster, and finds the same needles. `docs/KV4.md` |
| `SHARE_LIMIT` | 32768 | 32768 | 32768 | requests up to this many tokens (prompt + `max_tokens`) share decode steps; a longer one waits for the others and runs alone. Every rank holds the first `SHARE_LIMIT` tokens of a session and a share of the rest, so a higher limit leaves a shorter longest context: at TP=4, 65536 gives `CAP` 387072, 98304 gives 288768. A multiple of 64 from 4096; a limit off the 2048 doubling is a decode bucket boundary of its own |
| `KV_FORMAT` | fp8 | fp8 | fp8 | experimental. The KV latent row: `fp8` (656 B) or `int4` (416 B, INT4 after a Hadamard rotation). `int4` runs `bin/g6m4-int4` (in the image; `KV_FORMAT=int4 ./gr8 glm53 build` builds it), and its default `CAP` keeps fp8's memory a rank: 415,232 at TP=4 (2,390 pages a rank, measured against fp8 at 1,920; the fp8 default is 1,981 now), 345,088 at TP=3 and 857,600 at TP=6 (estimates). At TP=4 it decodes about 3% slower and prefills about 6% faster, and finds the same needles. `docs/KV4.md` |
| `SESS` | 8 | 8 | 8 | requests decoding at once (1-8); every reachable step graph is captured at boot (about 2.5 GiB at 8) |
| `REUSE` | 1 | 1 | 1 | 1 = `--reuse-prefix`: a request that continues an earlier one re-prefills only the new part. 0 = off |
| `KV_STORE` | off | off | off | experimental. Block stores below the KV pool (`--kv-store`, given once per store, or stores separated by `;` in a file or the environment), tried in order: `STORE,size=N(M\|G\|T)[,persist=on\|off][,keep=system\|sessions\|all]`. `keep=` is that store's `KV_KEEP`: `ssd:PATH,size=4G,persist=on,keep=system;remote:auto,size=4G,keep=all` keeps system prompts on the SSD and every other evicted run (with its tail past `SHARE_LIMIT`) on the KV node. STORE is `ssd:PATH` (a folder on each box: rank r uses `PATH/r<r>`), `remote:HOST:PORT` or `remote:auto` (a KV node the mentat group announces, `./gr8 kv up`). An SSD store is cleared at every boot. With `persist=on` its system prefixes and the counts survive restarts while the model, layout, block format and engine build stay the same; a boot that served wrong output would hand them on, so persist only a store you trust. Sizes are per rank, G = 2^30. Needs `REUSE`. `engine/kvman-rs/REMOTE-TIER.md` |
| `KV_KEEP` | system | system | system | what the stores keep. `system`: each system prompt with its tool definitions (the span the front end sends; without one, the prompt's grid cells), once, after `KV_SYSTEM_AFTER` conversations started with it. `sessions`: those, plus the runs the pool evicts that continued their conversation at least once, stored on the system prefix they start with, a run past `SHARE_LIMIT` with each rank's shard of its tail. `all`: every run the pool evicts, for benchmarks. `system` suits most; `sessions` suits agent fleets larger than the KV pool. On the cluster's 8-agent replay (TP=4) `sessions` cut TTFT p50 from 234 s to 126 s writing 98 GB/h a rank, `all` was about the same, and `system` wrote 1.2 GB/h (TTFT p50 213 s); `engine/kvman-rs/REMOTE-TIER.md` has the table. In kvman-sim on the opencode captures `system` writes under 4 GB/h per rank and `all` 90-135 |
| `KV_SYSTEM_AFTER` | 2 | 2 | 2 | the conversations a system prefix must start before the stores keep it. A request that continues a live, retained or stored run counts toward nothing |
| `KV_WRITE_LIMIT` | off | off | off | a cap on store writes: at most this many page blocks per page of KV the engine computes (prefill and decode), so `0.25` writes at most a quarter of what the engine fills. Counted the same on every rank, never by the clock; a store refuses what would go past it |
| `KV_ROWS` | off | off | off | experimental. Remote attention rows: each rank's attention rows past `SHARE_LIMIT` go to a rows store as their pages fill, and decode reads back the rows its selection picks (`--kv-rows`): `STORE,size=N(M\|G\|T)[,local=PAGES][,pool=SLOTS][,place=PAGES][,fmt=fp8\|int4]`. STORE is `remote:auto` (a KV node the mentat group announces, odd ranks on its second rail, as `KV_STORE`), `remote:HOST:PORT` or `mem:`. `size` is each rank's lease; with `local` a session waits until its tail's records fit it. `local=PAGES`: release, a session's pages past `SHARE_LIMIT` cycle through that many local pages a rank, so its tail no longer needs local memory; `place=PAGES` then caps the main model's pages a rank, so `CAP` may pass what they hold. `pool` is the resident slots per indexer group (with `local` it defaults to what the sessions past `SHARE_LIMIT` that `place` holds can select in a step, and a smaller value admits fewer of them at once), `fmt=int4` stores 416 B rows for 656. Sets the engine's `G6_RT_*` ([docs/SETTINGS.md](docs/SETTINGS.md)) |
| `KV_TIER`, `KV_SHARED_AFTER` | | | | deprecated, for one release: `KV_TIER` is `KV_STORE` with `keep=evicted\|shared\|all` per tier, and maps to `KV_KEEP=all` when any tier kept evicted runs (the old default), else `system`; `KV_SHARED_AFTER` is `KV_SYSTEM_AFTER`. A warning says so |
| `DFLASH` | 1 with a drafter, else 0 | 0 (fixed) | 0 (fixed) | 1 = the DFlash2 drafter (`--drafter`, or `dflash.r<r>.*` in the model set); 0 = MTP only. TP=3 and TP=6 refuse 1: the drafter's 8 KV heads do not split three or six ways |
| `DFLASH_SEL` | engine default (1) | – | – | 1 = the DFlash2 selector walk, 0 = the drafter's argmax |
| `DFLASH_K` | engine default (7) | – | – | verify at most the first k of the 7 drafts |
| `SPEC_ADAPT` | `DFLASH` | – | – | 1 = choose each round's draft count from step cost and acceptance |
| `SPEC_LEVELS` | all | – | – | the draft counts `SPEC_ADAPT` may choose, e.g. `7,5,3,2` |
| `SPEC_ACCEPT` | coupled | coupled | coupled | `rejection` = rejection sampling of drafts at temperature > 0 |
| `MTP` | 2 | 2 | 2 | MTP draft count (TP=4: for sessions that share steps, or with `DFLASH=0`; TP=6 and TP=3: every round) |
| `PF_OWN` | 4 | 3 | 1 | prefill attention ownership (`G6_PF_OWN`); 3 also loads `packs/qo.pack`. TP=6 with `PF_OWN=1` needs no QO pack |
| `QP_ROLL` | 2 | – | – | `PF_OWN=4`: rolling pair-weight slots (`G6_PF_QP_ROLL`) |
| `QP_CTX` | `CAP` | – | – | `PF_OWN=4`: QO key-table size; a longer request runs OWN=2 |
| `QP_EARLY` | 1 | – | – | `PF_OWN=4`: send the counterpart's partial before the own rows' o_proj |
| `V2`, `V2_SLICES` | unset | unset | unset | exlr8 v2 experts from their raw width files: the width files and the index slices |
| `VARIANT` (`--variant`) | `home` | `home` | `bit225` | the exlr8 v2 width manifest: `bit225`, `bit243`, `bit260`, `bit325`, `bit338` (routed experts at about 2.25-3.38 bpw, spent where they lower KL against bf16 the most), `home` (3.25 bpw), `down1` (about 2.25), `up1` (about 4.0), or a manifest path. A name is `exlr8-manifest-<name>.json` in the model set or in `V2` (directly or under `manifests/`). At TP=3, a release without `bit225` boots `down1`. Without `V2`, only `home` |
| `V2_WIDTHS` | unset | unset | unset | the width manifest as a path: the older spelling of `--variant <path>`. Giving both is refused |
| `DFL_REUSE` | 2 | – | – | 1 = a request whose prefix-reuse hit skipped into the DFlash2 drafter's window keeps the drafter; 2 = also a request alone on MTP takes it (`G6_DFL_REUSE`). Defaults to 2 when serving DFlash2 with MTP. Needs `DFLASH=1` |
| `DFL_REUSE_MIN` | unset (256) | – | – | `DFL_REUSE`: context rows the drafter must hold before it drafts; fewer = MTP warm-up rounds (`G6_DFL_REUSE_MIN`) |
| `MTP_KSTOP` | on (2) when serving w/ DFlash2 + MTP 2+ | – | – | a request's solo MTP rounds chain and verify the drafts the MTP head is confident in: 1 = thresholds (`MTP_KSTOP_TAU`), 2 = the drafter choice's MTP estimates (`G6_MTP_KSTOP`). Needs `DFLASH=1`, `MTP` 2+ |
| `MTP_KSTOP_KMAX` | unset (`MTP`) | – | – | `MTP_KSTOP`: the most drafts a solo MTP round chains, 1..3 (`G6_MTP_KSTOP_KMAX`) |
| `MTP_KSTOP_TAU` | unset (`0,0.2,0.5`) | – | – | `MTP_KSTOP=1`: thresholds t1,t2,t3 on the drafts' cumulative probability (`G6_MTP_KSTOP_TAU`) |
| `DRAFT_PAR` | on (2) when serving w/ DFlash2 + MTP | – | – | a solo DFlash2 round's drafter step on its own stream under MTP's first form, 1, or 2 = drafting steps at high priority (`G6_DRAFT_PAR`; bit-exact). Needs `DFLASH=1` |
| `QA_SPLIT` | on (2) when TP>1 | unset (0) | unset (0) | decode: the layer-input GEMVs split by output tiles across the ranks and all-gathered (`G6_QA_SPLIT`, 1 or 2; bit-exact). Measured at TP=4 only |
| `IDXQ_SPLIT` | on (1) when TP>1 | unset (0) | unset (0) | decode: the indexer's wq_b split the same way (`G6_IDXQ_SPLIT`; bit-exact) |
| `MTP_SPLIT` | on (1) when TP>1 + MTP | unset (0) | unset (0) | MTP draft steps: eh_proj and the draft layer's input GEMVs split the same way (`G6_MTP_SPLIT`; bit-exact) |
| `DRAFT_HEAD_I4` | unset (0) | unset (0) | unset (0) | 1 = the MTP draft's lm_head on INT4 g32 copies made at boot (`G6_DRAFT_HEAD_I4`; drafts only, +134 MB a rank at TP=4) |
| `DFL_HEAD_I4` | unset (0) | – | – | 1 = the DFlash2 drafter's lm_head on the same INT4 records (`G6_DFL_HEAD_I4`; drafts only). Needs `DFLASH=1` |
| `DFL_BODY_I4` | unset (0) | – | – | 1 = the DFlash2 drafter's body GEMVs on INT4 g32 copies (`G6_DFL_BODY_I4`; drafts only, +382 MB a rank). Needs `DFLASH=1` |
| `ROUTE_OVL` | unset (0) | unset (0) | unset (0) | decode MoE layers of 2+ rows: the trellis shared expert on a forked stream during the router, 1 = fc1 + fc2, 2 = fc1 (`G6_ROUTE_OVL`; bit-exact). Measured at TP=4 only |
| `ATTN_SC2` | unset (0) | unset (0) | unset (0) | decode attention scores with S = 1, 2, 4 or 8 key slots per warp (`G6_ATTN_SC2`; bit-exact) |
| `ATTN_SC2_TR` | unset (0) | unset (0) | unset (0) | with `ATTN_SC2`: the transpose reduction, 1 or 2 = eager (`G6_ATTN_SC2_TR`; bit-exact) |
| `ATTN_CTX2` | unset (0) | unset (0) | unset (0) | decode attention context kernel variant 1..4 (`G6_ATTN_CTX2`; bit-exact) |
| `ATTN_CTX3` | unset (0) | unset (0) | unset (0) | decode attention context with one test per batch of slots, 1..4 (`G6_ATTN_CTX3`; bit-exact) |
| `ATTN_SEL2` | unset (0) | unset (0) | unset (0) | decode index selection's final sort as a rank by count over 8, 16, 32 or 64 CTAs per row (`G6_ATTN_SEL2`; bit-exact) |
| `IDX_GRP` | unset (0) | unset (0) | unset (0) | S2 batched decode: the indexer reads each key tile once for up to 1..64 rows of a session (`G6_IDX_GRP`; bit-exact) |
| `ATTN_QT` | unset (0) | unset (0) | unset (0) | 1 = q and rope stored lane-transposed for the scores kernel (`G6_ATTN_QT`; needs `ATTN_SC2` 2 or 4; bit-exact) |
| `ATTN_FRONT` | unset (0) | unset (0) | unset (0) | 1 = a decode step's first attention kernels in one launch (`G6_ATTN_FRONT`; bit-exact) |
| `DSEL_MODEL` | unset (0) | – | – | 1 = the drafter choice prices both drafters from `DSEL_TABLE` and a request's MTP rounds as the confidence stop runs them (`G6_S2_DSEL_MODEL`; drafts only: the target decides every token). Needs `DFLASH=1` |
| `DSEL_TABLE` | unset | – | – | `DSEL_MODEL`: the cost table, a `key:value` list (`v1`..`v8`, `d`, `m`, `e`, `p`, optional `g`, `k`, `n`, `w`, `r`) or a file holding one, measured on the cluster with `tools/s2_dsel_table.py` (`G6_S2_DSEL_TABLE`) |
| `MTP_KSTOPB` | unset (0) | unset (0) | unset (0) | 1 = in rounds that several sessions share, each session verifies its own count of the `MTP` drafts, from the MTP head's confidence (`G6_MTP_KSTOPB`; drafts only: the target decides every token). Captures a main graph per width the rule reaches |
| `MTP_KSTOPB_KMIN` | unset (0) | unset (0) | unset (0) | `MTP_KSTOPB`: the least drafts a session verifies, 0 or 1 (`G6_MTP_KSTOPB_KMIN`) |
| `MTP_KSTOPB_TAU` | unset (`0.6,0.88`) | same | same | `MTP_KSTOPB`: thresholds t1,t2,... (up to 7) on each session's cumulative draft probability, at every session count (`G6_MTP_KSTOPB_TAU`) |
| `MTP_KSTOPB_TAU_S` | unset | unset | unset | `MTP_KSTOPB`: thresholds per session count, `n:t1,t2,...` entries separated by `/`, e.g. `1:0.7,0.7/2:0.7,0.8` (`G6_MTP_KSTOPB_TAU_S<n>`) |
| `L2PF` | unset (0) | unset (0) | unset (0) | decode: a paced L2 prefetch of the weights the step reads next, on a side stream in the windows where DRAM is idle; window mask 1 = MLP all-reduce wait, 2 = gather waits, 4 = attention, 8 = attention all-reduce wait (`G6_L2PF`; bit-exact). Measured at TP=4 only, as 15 |
| `CP_STEP_ROWS` | unset (0) | unset (0) | unset (0) | S2: a decode step past `SHARE_LIMIT` carries at most this many rows, and the CP fabric is sized for it instead of for the widest S2 step (`SESS` x (`MTP` + 1) = 24 rows, 1.358 GB pinned per rank at TP=4; 8 rows: 0.453 GB). While a session past the limit decodes and the sessions' rows do not fit, they take turns, each once per cycle. At least the rows of a solo step (the boot names the minimum, 8 with DFlash2) (`G6_CP_STEP_ROWS`) |
| `CP_STEP_SEG` | unset (0) | unset (0) | unset (0) | with `CP_STEP_ROWS`: 1 = a round past `SHARE_LIMIT` takes only sessions past it, so a shorter session never runs the CP attention because a long one is resident (`G6_CP_STEP_SEG`) |
| `NOHASH` | 0 | 0 | 0 | 1 = skip the sha256 check of every pack tensor at load (run once without it first) |
| `G6ENV_X` | empty | empty | empty | more engine environment for every rank (`"A=1 B=2"`) |
| `G6ARGS_X` | – | empty | empty | more engine arguments for every rank |
| `READY_TMO` | 1800 | 1800 | 1800 | seconds to wait for the ranks to load |
| `FRONT_BIN` | `bin/g6front-rs` | same | same | the Rust front end binary |
| `PY` | `$G6_HOME/venv/bin/python` | same | same | the python that runs the shim |
| `S2PORT` | 7341 | 7341 | 7341 | the multi-session control port on the head |

## `./gr8 glm53`

`./gr8 glm53f` takes the same options for GLM-5.3-Flash: TP 4 or 2, one session, an fp8 KV build, no drafter, and
`CAP` 262208 at either TP, a 262,144-token request (`tools/launch/models.py`, `flash_defaults` in
`scripts/engine_args.sh`, [docs/FLASH.md](docs/FLASH.md)).

| name | flag | default | scope | meaning |
|---|---|---|---|---|
| `GR8_MODEL` | `--gr8-model` | `glm53` | cluster | the model: `glm53` or `glm53f`. `./gr8 <model>` sets it |
| `TP` | `--tp` | the number of peers | cluster | 4, 6 or 3 (glm53f: 4 or 2); needed with mentat |
| `SHAPE` | `--shape` | `mesh` | cluster | `mesh` (a switch), or `ring` (TP=3 or 4, no switch): each box's f0 port cabled to the next box's f1 port, in `--peers` order. Needs `--peers`, takes no `--ips` |
| `PEERS` | `--peers` | none | cluster | the boxes' LAN addresses in rank order, the same on every box. The first is the head. `HOSTS` in an old config file is read as `PEERS` |
| `MENTAT` | `--mentat` | `localhost` | box | find the boxes through the local mentat daemon (the default when `PEERS` is unset) |
| `GROUP` | `--group` | `gr8-<model>` | cluster | the mentat group the boxes join |
| `HEAD` | `--head` | the lowest address | cluster | mentat mode: the head's LAN address |
| `IPS`, `IPS2` | `--ips`, `--ips2` | discovered | cluster | each rank's rail-0 and rail-1 fabric address. Without them each box finds its own two RoCE v2 addresses |
| `MODEL` | `--model` | none | box | the exlr8 release: `https://huggingface.co/<org>/<repo>[@rev]` (each box fetches its part into `MODEL_CACHE`), or a local copy of it (`dense/`, `experts/`, `index-slices/`, `manifests/`, the tokenizer files) used in place, on a local disk or a shared mount such as NFS. The engine plans every rank's weights from the release's files (`V2_DENSE`) |
| `V2_DENSE` | `--v2-dense` | the release | box | the exlr8 release directory whose `dense/` files give every non-expert weight (`g6m4 --v2-dense`) |
| `V2_DENSE_FORM` | `--v2-dense-form` | `trellis` | cluster | with `V2_DENSE`: `trellis` keeps q_b, o_proj and the dense MLP as the release's trellis entries for the dense trellis kernels, so the weights are the release's own. `rounded` is the fallback: it rounds them to the MXFP8 and int4 forms of a converted pack |
| `DRAFTER` | `--drafter` | none | box | TP=4: the DFlash2 drafter's repo, `https://huggingface.co/incoai/GLM-5.3-DFlash2`. Each box fetches it and converts its rank into `MODEL_CACHE`. Its repo and commit go into the id the boxes compare |
| `DFLASH_DIR` | `--dflash-dir` | none | box | a directory holding `dflash.r<r>.*`, when it is not the model set's |
| `RANK` | `--rank` | none | run | `prepare`: the rank to fetch, when this box cannot tell (mentat mode) |
| `MODEL_CACHE` | `--model-cache` | none | box | with an https:// `MODEL`: this box's local directory for its rank's part of the release, and for the converted drafter |
| `MODEL_DIR` | `--model-dir` | none | box | the tokenizer files, when the release directory has none |
| `G6_HOME` | `--g6-home` | the repo checkout | box | working directory: `bin/`, `out/`, `s2/`, `run/` |
| `PROXY_CPU` | `--proxy-cpu` | 19 | box | CPU core of this rank's fabric proxy thread. The other proxies take the 7 cores below it, and the ring's relay thread the one below those |
| `RING_DEVS` | `--ring-devs` | the GB10 CX-7 names | box | `--shape ring`: this box's RDMA devices as relay-out,next,relay-in,prev (`rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`) |
| `NODE_IP` | `--node-ip` | found | box | this box's LAN address, when several peers are addresses of this box |
| `CONTROL_PORT` | `--control-port` | 7340 | cluster | the supervisors' TCP port on the LAN (handshake, status, logs, down) |
| `PORT` | `--port` | 8100 | cluster | the OpenAI API on the head |
| `DEFAULT_MAX_TOKENS` | `--default-max-tokens` | 0 | box | the head's front end: `max_tokens` for a chat request that names none. 0 = what keeps the request within `SHARE_LIMIT`, but at least 8,192, and never more than the room the context leaves; another value is used as it is |
| `DEFAULT_REASONING_EFFORT` | `--default-reasoning-effort` | none | box | the head's front end: `reasoning_effort` for a chat request that names none: `minimal`, `low`, `medium`, `high`, `xhigh` or `max`, which the template hears as the nearest level it names (README, "Thinking"). Unset = the template's default (`max` on GLM 5.3). `none` is refused: GLM 5.3 cannot turn thinking off |
| `DEFAULT_THINKING_BUDGET` | `--default-thinking-budget` | 0 | box | the head's front end: `thinking_budget` for a chat request that names none, or names 0 or null (TensorFold's rule). The front end closes the think block at the reply's Nth token. 0 = no limit. A request with `logprobs` gets no default budget |
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
| `KV_MIN_FREE_GB` | `--min-free-gb` | 10 | refuse to start when locking `KV_GB` would leave less MemAvailable than this |
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
