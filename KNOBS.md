# Knobs

Environment variables the launch path reads: `./gr8 glm53`, the launchers in `scripts/` and the build. Set one on the
command line (`CAP=262144 ./gr8 glm53 up`) or in `config/cluster.env`; every script sources that file, so a value there
wins over the same variable on the command line. The engine's own environment (`G6_*`) and flags are in
[docs/SETTINGS.md](docs/SETTINGS.md), and each launcher's header comment is the full reference for its knobs.

"TP=4" is the served engine, `launch_tp4.sh` running `launch_s2.sh`. "TP=4 plain" is `PLAIN=1`, the one-request loop
being retired. A dash means the launcher does not read it.

## Launch

| variable | TP=4 | TP=4 plain | TP=6 | TP=3 | meaning |
|---|---|---|---|---|---|
| `CAP` | 393216 | 393216 | 614400 | 307200 | KV pool in tokens, and so the longest context. Lower it first when a box runs short of memory |
| `SESS` | 8 | – | – | – | requests decoding at once (1-8); every reachable step graph is captured at boot (about 2.5 GiB at 8) |
| `REUSE` | 0 | – | – | – | 1 = `--reuse-prefix`: a request that continues an earlier one re-prefills only the new part |
| `DFLASH` | 1 | 1, or 0 when `MTP` is set | – | – | 1 = the DFlash2 drafter (needs `packs/dflash.r<r>.*` on every box); 0 = MTP only |
| `DFLASH_SEL` | engine default (1) | same | – | – | 1 = the DFlash2 selector walk, 0 = the drafter's argmax |
| `DFLASH_K` | engine default (7) | same | – | – | verify at most the first k of the 7 drafts |
| `SPEC_ADAPT` | `DFLASH` | `DFLASH` | – | – | 1 = choose each round's draft count from step cost and acceptance |
| `SPEC_LEVELS` | all | all | – | – | the draft counts `SPEC_ADAPT` may choose, e.g. `7,5,3,2` |
| `SPEC_ACCEPT` | – | coupled | – | – | `rejection` = rejection sampling of drafts at temperature > 0 |
| `MTP` | 2 | 2 | 2 | 2 | MTP draft count (TP=4: for sessions that share steps, or with `DFLASH=0`) |
| `PF_OWN` | 4 | 4 | 3 (fixed) | 1 | prefill attention ownership (`G6_PF_OWN`); 3 also loads `packs/qo.pack` |
| `QP_ROLL` | 2 | 2 | – | – | `PF_OWN=4`: rolling pair-weight slots (`G6_PF_QP_ROLL`) |
| `QP_CTX` | `CAP` | `CAP` | – | – | `PF_OWN=4`: QO key-table size; a longer request runs OWN=2 |
| `QP_EARLY` | 1 | 1 | – | – | `PF_OWN=4`: send the counterpart's partial before the own rows' o_proj |
| `QP_PACK` | – | 0 | – | – | 1 = resident pair packs (`packs/qp.rank<r>.*`) instead of `QP_ROLL` |
| `PACKS` | `packs` | `packs` | – | – | where the rank packs are, relative to `G6_HOME` or absolute |
| `V2`, `V2_SLICES`, `V2_WIDTHS` | unset | unset | – | – | exlr8 v2 experts from their raw width files (`tools/tune/README.md`) |
| `NOHASH` | 0 | 0 | 0 | 0 | 1 = skip the sha256 check of every pack tensor at load (run once without it first) |
| `G6ENV_X` | empty | empty | empty | empty | more engine environment for every rank (`"A=1 B=2"`) |
| `G6ARGS_X` | – | – | empty | empty | more engine arguments for every rank |
| `READY_TMO` | 1800 | 900 | 900 | 900 | seconds to wait for the ranks to load |
| `FRONT` | rs only | rs, else py | rs, else py | rs, else py | front end: `rs` = `bin/g6front-rs`, `py` = `frontend/g6front` |
| `FRONT_BIN` | `bin/g6front-rs` | same | same | same | the Rust front end binary |
| `PY` | `$G6_HOME/venv/bin/python` | same | same | same | the python that runs the shim (and the Python front end) |
| `S2PORT` | 7341 | – | – | – | the multi-session control port on the head |
| `PLAIN` | 0 | 1 | – | – | 1 = the one-request loop at TP=4, from a `PLAIN=1` build |
| `RANK_HEALTH` | – | – | 1 | 1 | probe every rank before booting and refuse a bad one (`scripts/rank_health.sh`) |
| `PROBE_ARGS` | – | – | empty | empty | thresholds for that probe (`tools/health/README.md`) |

## `./gr8 glm53`

| variable | default | meaning |
|---|---|---|
| `G6_CLUSTER` | `config/cluster.env` | the cluster file, for every script |
| `TP` | the number of `HOSTS` | must agree with `HOSTS`; picks the launcher and the build |
| `CHECK` | 1 | 0 = `up` and `restart` skip `./gr8 glm53 check` (same as `--no-check`) |
| `PACK_DIR` | none | `./gr8 glm53 stage` with no argument copies from here |
| `BIN_DIR` | `bin` | what `./gr8 glm53 stage` copies the engine from |

## Build and stage

| variable | script | meaning |
|---|---|---|
| `G6_NR` | `build.sh`, `c1_tests.sh` | rank count of the build: 4, 3, or unset for 6 (`./gr8 glm53 build` sets it from `TP`) |
| `PLAIN` | `build.sh` | 1 = a TP=4 build without the multi-session path |
| `BUNDLE_LIBS` | `build.sh` | 1 = copy cuBLAS into `bin/lib` for boxes without the CUDA toolkit |
| `ARCH` | `build.sh`, `c1_tests.sh` | GPU architecture, default `121a` (GB10) |
| `EXTRA` | `build.sh`, `c1_tests.sh` | more `-D` flags |
| `SHARED` | `stage.sh` | 1 = `PACK_DIR` is a shared mount at the same path on every box; each box copies its own files |
