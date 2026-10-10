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

This is the other way round from the ssh launchers (`./gr8 glm53 --ssh up`, deprecated), where every script sources the
config file, so a value there wins over the environment. `./gr8 glm53 up` prints every value and where it came from,
and `./gr8 glm53 show-config` prints them without booting.

The handshake before the engines start compares the cluster options on every box: the TP, the shape, the peer list or
mentat group, the ports, every knob in the launch table below that reaches the engine, and the engine environment and
arguments they produce. Per-box values, such as paths and the proxy CPU, do not count. Any difference stops every
box, with the option and each box's value.

The engine does not inherit `G6_*` variables from the environment, and `./gr8 glm53 up` refuses to start while one is
set. Pass engine variables with `--g6env-x "NAME=value ..."` so that every box compares them.

The defaults in the launch table come from `scripts/engine_args.sh`. The supervisor and the ssh launchers both read
it. "TP=4" is the multi-session engine (`--s2`). A dash means that TP does not read the knob. `V2`, `V2_SLICES` and
`V2_WIDTHS` are paths, so each box may have its own. The handshake compares the variant's name and the sha256 of its
manifest instead.

## Launch

| variable | TP=4 | TP=6 | TP=3 | meaning |
|---|---|---|---|---|
| `CAP` | 393216 | 614400 | 307200 | KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory |
| `SESS` | 8 | – | – | requests decoding at once (1-8); every reachable step graph is captured at boot (about 2.5 GiB at 8) |
| `REUSE` | 1 | – | – | 1 = `--reuse-prefix`: a request that continues an earlier one re-prefills only the new part. 0 = off |
| `DFLASH` | 1 with a drafter, else 0 | – | – | 1 = the DFlash2 drafter (`--drafter`, or `dflash.r<r>.*` in the model set); 0 = MTP only |
| `DFLASH_SEL` | engine default (1) | – | – | 1 = the DFlash2 selector walk, 0 = the drafter's argmax |
| `DFLASH_K` | engine default (7) | – | – | verify at most the first k of the 7 drafts |
| `SPEC_ADAPT` | `DFLASH` | – | – | 1 = choose each round's draft count from step cost and acceptance |
| `SPEC_LEVELS` | all | – | – | the draft counts `SPEC_ADAPT` may choose, e.g. `7,5,3,2` |
| `SPEC_ACCEPT` | coupled | – | – | `rejection` = rejection sampling of drafts at temperature > 0 |
| `MTP` | 2 | 2 | 2 | MTP draft count (TP=4: for sessions that share steps, or with `DFLASH=0`) |
| `PF_OWN` | 4 | 3 (fixed) | 1 | prefill attention ownership (`G6_PF_OWN`); 3 also loads `packs/qo.pack` |
| `QP_ROLL` | 2 | – | – | `PF_OWN=4`: rolling pair-weight slots (`G6_PF_QP_ROLL`) |
| `QP_CTX` | `CAP` | – | – | `PF_OWN=4`: QO key-table size; a longer request runs OWN=2 |
| `QP_EARLY` | 1 | – | – | `PF_OWN=4`: send the counterpart's partial before the own rows' o_proj |
| `PACKS` | `packs` | – | – | where the rank packs are, relative to `G6_HOME` or absolute |
| `V2`, `V2_SLICES` | unset | unset | unset | exlr8 v2 experts from their raw width files (`tools/tune/README.md`): the width files and the index slices |
| `VARIANT` (`--variant`) | `home` | `home` | `home` | the exlr8 v2 width manifest: `home` (3.25 bpw routed experts), `down1` (about 2.25), `up1` (about 4.0), or a manifest path. A name is `exlr8-manifest-<name>.json` in the model set or in `V2` (directly or under `manifests/`). Without `V2`, only `home` |
| `V2_WIDTHS` | unset | unset | unset | the width manifest as a path: the older spelling of `--variant <path>`. Giving both is refused |
| `NOHASH` | 0 | 0 | 0 | 1 = skip the sha256 check of every pack tensor at load (run once without it first) |
| `G6ENV_X` | empty | empty | empty | more engine environment for every rank (`"A=1 B=2"`) |
| `G6ARGS_X` | – | empty | empty | more engine arguments for every rank |
| `READY_TMO` | 1800 | 900 | 900 | seconds to wait for the ranks to load |
| `FRONT` | rs only | rs, else py | rs, else py | front end: `rs` = `bin/g6front-rs`, `py` = `frontend/g6front` |
| `FRONT_BIN` | `bin/g6front-rs` | same | same | the Rust front end binary |
| `PY` | `$G6_HOME/venv/bin/python` | same | same | the python that runs the shim (and the Python front end) |
| `S2PORT` | 7341 | – | – | the multi-session control port on the head |
| `RANK_HEALTH` | – | 1 | 1 | probe every rank before booting and refuse a bad one (`scripts/rank_health.sh`) |
| `PROBE_ARGS` | – | empty | empty | thresholds for that probe (`tools/health/README.md`) |

## `./gr8 glm53`

| name | flag | default | scope | meaning |
|---|---|---|---|---|
| `TP` | `--tp` | the number of peers | cluster | 4, 6 or 3; needed with mentat |
| `SHAPE` | `--shape` | `mesh` | cluster | `mesh` (a switch). `ring` is refused: the engine has no ring fabric yet |
| `PEERS` | `--peers` | none | cluster | the boxes' LAN addresses in rank order, the same on every box. The first is the head. `HOSTS` in an old config file is read as `PEERS` |
| `MENTAT` | `--mentat` | `localhost` | box | find the boxes through the local mentat daemon (the default when `PEERS` is unset) |
| `GROUP` | `--group` | `gr8-glm53` | cluster | the mentat group the boxes join |
| `HEAD` | `--head` | the lowest address | cluster | mentat mode: the head's LAN address |
| `IPS`, `IPS2` | `--ips`, `--ips2` | discovered | cluster | each rank's rail-0 and rail-1 fabric address. Without them each box finds its own two RoCE v2 addresses |
| `MODEL` | `--model` | none | box | the exlr8 release, `https://huggingface.co/<org>/<repo>[@rev]` (each box fetches its part into `MODEL_CACHE`), or a prepared pack directory with every rank's manifest. Without `MODEL_CACHE` the engine loads a pack directory in place |
| `BASE_PACKS` | `--base-packs` | none | box | with an https:// `MODEL`: the prepared pack directory the non-expert weights come from, until the engine reads the release's dense files (#116) |
| `DRAFTER` | `--drafter` | none | box | TP=4: the DFlash2 drafter's repo, `https://huggingface.co/incoai/GLM-5.3-DFlash2`. Each box fetches it and converts its rank into `MODEL_CACHE`. Its repo and commit go into the id the boxes compare |
| `DFLASH_DIR` | `--dflash-dir` | none | box | a directory holding `dflash.r<r>.*`, when it is not the model set's |
| `RANK` | `--rank` | none | run | `prepare`: the rank to fetch, when this box cannot tell (mentat mode) |
| `MODEL_CACHE` | `--model-cache` | none | box | this box's local directory for its shard of the model set. Alone, it boots from the one set it holds |
| `MODEL_DIR` | `--model-dir` | none | box | the tokenizer files, when the pack directory has no `tokenizer/` |
| `G6_HOME` | `--g6-home` | the repo checkout | box | working directory: `bin/`, `out/`, `s2/`, `serve/`, `run/` |
| `PROXY_CPU` | `--proxy-cpu` | 19 | box | CPU core of this rank's fabric proxy thread |
| `NODE_IP` | `--node-ip` | found | box | this box's LAN address, when several peers are addresses of this box |
| `CONTROL_PORT` | `--control-port` | 7340 | cluster | the supervisors' TCP port on the LAN (handshake, status, logs, down) |
| `PORT` | `--port` | 8100 | cluster | the OpenAI API on the head |
| `JOIN_TIMEOUT` | `--join-timeout` | 600 | run | seconds for the mentat group to fill, for a worker to reach the head, and on the head from the first hello to the last (0 = forever) |
| `CHECK` | `--no-check` | 1 | run | 0 = skip this box's checks (the model files are checked anyway) |
| `G6_CLUSTER` | | none | | the config file when there is no `@FILE` |
| `PACK_DIR` | | none | | `./gr8 glm53 stage` with no argument copies from here |
| `BIN_DIR` | | `bin` | | what `./gr8 glm53 stage` copies the engine from |

`PACKS`, `RANK_HEALTH` and `PROBE_ARGS` work with `--ssh` only. The symmetric launch takes the packs from `MODEL`, and it
does not run the rank health probe yet: run `scripts/rank_health.sh` first if you want it.

## Build and stage

| variable | script | meaning |
|---|---|---|
| `G6_NR` | `build.sh`, `c1_tests.sh` | rank count of the build: 4, 3, or unset for 6 (`./gr8 glm53 build` sets it from `TP`) |
| `BUNDLE_LIBS` | `build.sh` | 1 = copy cuBLAS into `bin/lib` for boxes without the CUDA toolkit |
| `ARCH` | `build.sh`, `c1_tests.sh` | GPU architecture, default `121a` (GB10) |
| `EXTRA` | `build.sh`, `c1_tests.sh` | more `-D` flags |
| `SHARED` | `stage.sh` | 1 = `PACK_DIR` is a shared mount at the same path on every box; each box copies its own files |
