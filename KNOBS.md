# Knobs

The options of `./gr8 glm53`, the launchers in `scripts/` and the build. Each knob has a name
(`CAP`) and a flag (`--cap`): the flag is the name in lower case with `-` for `_`, except `--s2-port` for `S2PORT` and
`--work-dir` for `GR8_WORK_DIR`. The engine's own environment and flags are in [docs/SETTINGS.md](docs/SETTINGS.md), and
[docs/ENV.md](docs/ENV.md) lists every `GR8_*` engine variable.

The launcher prints its options:

- `./gr8 knobs` lists every option of `./gr8 <model>` and `./gr8 kv`: its flag, its scope, its default and what it does.
- `./gr8 glm53 knobs` (or `glm53f`) adds the value each TP runs with, from `scripts/engine_args.sh` and the options
  given: `./gr8 glm53 knobs @cluster.env --tp 4` shows what that config file boots at TP=4. A dash means unset, so the
  engine runs its own default, or the TP does not read the knob.
- Words after `knobs` keep the options whose name holds one of them: `./gr8 glm53 knobs ATTN KV_`.

Where a value comes from, strongest first:

1. the command line: `./gr8 glm53 up --cap 262144`;
2. one config file: `@FILE` on the command line, else `$GR8_CLUSTER`, else `./cluster.env`, else `config/cluster.env`
   (`--no-config` reads none). `./gr8 glm53` parses the file and does not source it: `KEY=value` lines, `#` comments, optional `export`, quotes
   stripped, and no `$` expansion;
3. the environment: `CAP=262144 ./gr8 glm53 up`;
4. the default.

`./gr8 glm53 up` prints every value and where it came from, and `./gr8 glm53 show-config` prints them without booting.

The handshake before the engines start compares the cluster options on every box: the TP, the shape, the peer list or
mentat group, the ports, every engine knob (`./gr8 knobs` lists them after `CAP`), and the engine environment and
arguments they produce. Per-box values, such as paths and the proxy CPU, do not count. Any difference stops every
box, with the option and each box's value.

The engine does not inherit `G6_*` variables from the environment, and `./gr8 glm53 up` refuses to start while one is
set. Pass engine variables with `--extra-env "NAME=value ..."` so that every box compares them.

The defaults come from `scripts/engine_args.sh`, which the supervisor reads. Every TP runs the multi-session engine
(`--serve`). `V2`, `V2_SLICES`, `V2_DENSE` and `V2_WIDTHS` are paths, so each box may have its own. The handshake
compares the variant's name and the sha256 of its manifest instead. `LAYOUT` is a path too: the handshake compares the
sha256 of the file.

## `./gr8 glm53f`

`./gr8 glm53f` takes the same options for GLM-5.3-Flash: TP 4 or 2, an fp8 KV build, no drafter, no prefix reuse, and
`CAP` 262208 at either TP, a 262,144-token request (`tools/launch/models.py`, `flash_defaults` in
`scripts/engine_args.sh`, [docs/FLASH.md](docs/FLASH.md)). `SESS` defaults to 1 there; 2 to 8 share decode steps and
the `CAP` pool, each session with its own KDA state slots, while `SESS` x (`MTP` + 1) is at most 32.

A Flash box can lend what its serve leaves to a KV node beside it (D-331 M7; `donate` in `tools/launch/supervisor.py`):
the `FLASH_KV_*` and `FLASH_CAP_POLICY` options (`./gr8 knobs FLASH_`).

## Rank health

The launcher does not run the rank health probe: run `scripts/rank_health.sh` with the same options before a boot if
you want it (`PROBE_ARGS` sets its thresholds, `tools/health/README.md`).

## `./gr8 kv`

A KV node: a box that locks some of its memory and serves it as KV page blocks to the cluster, over RDMA where both
ends have a RoCE v2 address and TCP otherwise (`engine/kvman-rs/REMOTE-TIER.md`). The cluster uses it through `KV_STORE` and `KV_ROWS` with `remote:auto`.
`./gr8 kv up --gb N` starts it on this box. The node joins the mentat group as an agent without GPUs, with the service
`gr8-kv`, and the model's election counts only agents with GPUs. `down`, `status` (`status all`: every node the group
announces), `health` (exit 0 when this box's node serves, else 1) and `logs [-f]` work as for the model. The options
come from the same places as above. A shared config file is read for these keys only. `./gr8 knobs` lists them
after the model's options.

## Build

| variable | script | meaning |
|---|---|---|
| `GR8_WIDTHS` | `build.sh` | the TP widths g6m4 and m4sim run, default `1 2 3 4 6 8`; fewer is a shorter build |
| `GR8_NR` | `build.sh`, `c1_tests.sh` | `build.sh`: one width, as `GR8_WIDTHS=N`. `c1_tests.sh`: rank count of the tests, 4, 3, or unset for 6 |
| `JOBS` | `build.sh` | compiles at once, default the CPU count |
| `BUNDLE_LIBS` | `build.sh` | 1 = copy cuBLAS into `bin/lib` for boxes without the CUDA toolkit |
| `ARCH` | `build.sh`, `c1_tests.sh` | GPU architecture, default `121a` (GB10) |
| `EXTRA` | `build.sh`, `c1_tests.sh` | more `-D` flags |
| `KV_FORMAT` | `build.sh`, `./gr8 glm53 build` | `int4` = build `bin/g6m4-int4` (the 4-bit KV latent, `-DG6_KV4=1`) only. `./gr8 glm53 build` builds it after the fp8 engine when `KV_FORMAT=int4` |

## Local steering profile

Profile loading is explicit: the `STEERING_*` options (`./gr8 knobs STEERING`). See
[local steering](docs/STEERING-LOCAL.md) for identity checks and validation limits.

