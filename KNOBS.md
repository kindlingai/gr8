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

GLM-5.3 at `TP=4` defaults to `KV_FORMAT=int4`: `bin/g6m4-int4`, `CAP=513536`
(2,390 pages per rank), the bit-exact INT4 fast paths, folded Hadamard weights,
tensor-core decode and `CP_ONECHAIN`. Roll back with `--kv-format fp8` or
`KV_FORMAT=fp8`: `bin/g6m4`, `CAP=485376`, and no implicit `GR8_KV4_*` or
`GR8_CP_ONECHAIN` settings. Explicit settings still take precedence.
GLM-5.3 at `TP=3/5/6` and GLM-5.3-Flash retain FP8 unless INT4 is requested;
DeepSeek and pseudo-engine defaults are unchanged.

`DISPRAM=1` (`--dispram 1`; unset or `0` is off by default so baseline solo decode speed is unchanged) opts
into Kindling OS `dispramd` for GLM-5.3 and non-chunked GLM-5.3-Flash KV caches, growing an unset GLM-5.3
`CAP` from the minimum free display memory across all ranks (for example `513536` to `611328` tokens, `+19.0%`,
on 4x Spark INT4 with about `4.5%` slower solo decode and flat 8-stream decode). If that memory is no longer
available at boot, the cluster retries once with baseline `CAP`; `DISPRAM_DISABLE=1` opts out on a box.

## `./gr8 glm53f`

`./gr8 glm53f` takes the same options for GLM-5.3-Flash: TP 4, 3, 2 or 1, an fp8 KV build, the DFlash2 drafter
at TP 4, 2 and 1 when `--drafter` (or `DFLASH_DIR`) is given, else MTP (MTP at TP 3), no prefix reuse, and `CAP` 262208 at any TP, a 262,144-token request
(`tools/launch/models.py`, `flash_defaults` in `scripts/engine_args.sh`, [docs/FLASH.md](docs/FLASH.md)). `SESS`
defaults to 8 at TP=4, 4 at TP=3 and TP=2 INT4, and 1 at TP=2 FP8 and TP=1; 2 to 8 share decode steps and the `CAP` pool, each session
with its own KDA state slots, while `SESS` x (`MTP` + 1) is at most 32.

At `TP=2` with `KV_FORMAT=int4`, ordinary launches use `SESS=4` (max concurrent 4); when `SESS >= 2`, defaults are `MTP=3`,
`S2_SHARED_WATERFILL=1` and `S2_SHARED_WF_MARGIN=0.02`. `--s2-shared-waterfill 0` rolls back to uniform shared
rows and, unless explicitly set, `MTP=1`. Pass `--sess 1` for the prior single-session launch.
Measured on TP=2 INT4 against `S2_SHARED_WATERFILL=0` at matched `SESS=4`, `MTP=3`: +10.45% pooled concurrent decode throughput
(+12.27% at 4 streams, +7.96% at 2 streams), 20/20 exact solo `T=0`, KL = 0.0 and qeval 73/75 vs 73/75.
Additional graph captures can increase boot time and memory.
Measured on two GB10s at `TP=2` with `KV_FORMAT=int4`, plain `./gr8 glm53f --tp 2 --kv-format int4`
(`SESS=4`, max concurrent 4, `MTP=3`, `S2_SHARED_WATERFILL=1`, `k4` manifest, `CAP=262208`) against the prior
`SESS=1`, `MTP=1` default: 51.1 vs 51.9 tok/s at 1 stream (-1.7%), 68.3 vs 61.5 tok/s at 2 streams (+11.0%),
79.2 vs 59.6 tok/s at 4 streams (+32.9%), 14/14 smoke, and 11.1-11.3 GiB post-capture `MemAvailable` a box
(11.5-12.2 GiB with `--s2-shared-waterfill 0`, 24 s warm boot; 12.6-12.9 GiB at `SESS=1`, 27 s warm boot).
With the two-box recipe's `bit349` manifest, `CAP=2097152`, vision tower and aligned SSD prompt cache, `SESS=4`
(`max_concurrent=4`) leaves 8.9-9.2 GiB post-capture `MemAvailable` (25 s warm boot; 55.1 / 70.3 / 82.9 tok/s
at 1 / 2 / 4 streams), whereas `SESS=8`, `MTP=3` without `GR8_CP=1` leaves 6.4-6.9 GiB (8.7-9.1 GiB with
`GR8_CP=1`, 29 s warm boot, 97.4 tok/s at 8 streams).
See [Shared waterfill](docs/SETTINGS.md#shared-waterfill-speculative-row-allocation).

A Flash box can lend what its serve leaves to a KV node beside it (`donate` in `tools/launch/supervisor.py`):
the `FLASH_KV_*` and `FLASH_CAP_POLICY` options (`./gr8 knobs FLASH_`).

## SSD session store (`KV_STORE`)

`KV_STORE=ssd:/data/gr8kv,size=200G,persist=on,keep=sessions` keeps committed session KV across restarts,
both stock and steered chats (also with `keep=all`), each isolated to its exact steering profile and scale.
Continuing sessions reuse their stored full page blocks and write only new pages when evicted or drained at quit;
with reuse alignment disabled, the changed partial page is rewritten too. A hit on its own resident run does not
spill it. Linux SSD stores fully allocate their `size=` budget (rounded down to whole blocks) before opening, and
fail at boot if the reserve cannot be allocated. No new knob is needed; see
[SSD session extensions and reserve](engine/kvman-rs/SSD-TIER.md) for behavior, measurements, and limits.
`./gr8 knobs KV_STORE` lists the existing fields.

DeepSeek uses the same stores and keep policies alongside `GR8_DS_CACHE_MIB` (set to `0` for tier-only snapshots).
Its immutable full pages are written only once per retained continuation; rolling state and partial-page rows are
written at each snapshot, so total bytes can exceed the full-page-only `1.00x` ratio. `keep=system` checkpoints
the exact 64-token boundary within the system span after `KV_SYSTEM_AFTER` conversations; a span shorter than
64 tokens has no shared system checkpoint. Sessions admission requires a continuation hit. See
[DS41 session stores](docs/DEEPSEEK.md#tiered-session-stores).

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
| `KV_FORMAT` | `build.sh`, `./gr8 glm53 build` | `int4` = build `bin/g6m4-int4` (the 4-bit KV latent, `-DG6_KV4=1`) only. `./gr8 glm53 build` builds it after the fp8 engine for TP=4 or an unconfigured clone; explicit `KV_FORMAT=fp8` or another configured TP skips it unless INT4 is requested |

## Local steering profile

Profile loading is explicit: the `STEERING_*` options (`./gr8 knobs STEERING`). See
[local steering](docs/STEERING-LOCAL.md) for identity checks and validation limits.

