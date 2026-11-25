# Switching models on the same boxes

A box runs one gr8 model at a time. To move four boxes between GLM-5.3 at TP=4 and DeepSeek-V4.1-Flash at TP=4
([the DeepSeek recipe](https://github.com/kindlingai/recipes/tree/main/deepseek-v4.1-flash/dgx-spark/4x/exlr8-gr8)), stop one model and start
the other on every box. A two-box pair can also switch with GLM-5.3-Flash at TP=2;
[the pair's numbers](#the-glm-53-flash-pair-tp2) are at the bottom of this page. Both models keep their files on
the boxes, so a switch fetches nothing. The settings below make the switch fast; each one keeps its data on the
box's local disk. [docker/.env.example](../docker/.env.example) has the GLM-5.3 ones on.

## One command: `./gr8 switch`

`./gr8 switch <model>` moves the boxes to one model: `glm53` (GLM-5.3), `glm53f` (GLM-5.3-Flash) or `ds41`
(DeepSeek-V4.1-Flash). Run it on any machine that can reach the boxes over SSH (keys and `known_hosts` must already
work without a prompt) and the new model's `/health` URL.

1. It runs `docker compose down` for every other model on that model's boxes, all boxes at once. On one box, its
   compose commands run one after another.
2. It runs `docker compose up -d` for the new model on its boxes.
3. It waits until the head's `/health` reports `"ready":true` (it polls every 0.1 s, for `TIMEOUT` seconds at most).

It prints the time of each step: `down: <s>`, `up: <s>`, `health: ready in <s>` and `total: <s>`. The command changes
no configuration on the boxes: set up each model's compose directory and its `.env` on its boxes first, with the
settings below. Only one switch may run at a time. A failed step stops the command, names the box, and exits non-zero;
nothing rolls back, so look at that box before you run the command again.

The config is a file of `KEY=value` lines on the machine that runs the command. Copy
[config/switch.env.example](../config/switch.env.example) to `switch.env` (the default) or pass `@FILE`:

| Key | Value |
|---|---|
| `BOXES` | every box, as SSH targets (`user@host`), comma-separated |
| `<MODEL>_BOXES` | that model's boxes in rank order, head first (`GLM53_BOXES`, `GLM53F_BOXES`, `DS41_BOXES`) |
| `<MODEL>_DIR` | that model's compose directory, the same absolute path on each of its boxes |
| `<MODEL>_HEALTH_URL` | that model's head `/health` URL |
| `TIMEOUT` | seconds to wait for ready (default 900; `--timeout` overrides it) |
| `SSH_TIMEOUT` | seconds for each SSH and compose step (default 300); more than the stop grace (`GR8_STOP_GRACE`) |

```bash
cp config/switch.env.example switch.env    # then set the boxes, directories and URLs
./gr8 switch ds41                          # GLM-5.3 (or GLM-5.3-Flash) to DeepSeek
./gr8 switch glm53 @/path/to/switch.env    # back to GLM-5.3, with another config file
```

Measured on 2026-11-09 (Pacific time) with the command at gr8 `d62cb6b4`, on four DGX Sparks: GLM-5.3 and
DeepSeek-V4.1-Flash at TP=4 on all four, GLM-5.3-Flash at TP=2 on two of them. Settings: GLM-5.3 with
`GR8_GEN_CACHE`, `GR8_QUIT_FAST=1` and compact expert copies (`GR8_LOAD_COMPACT`, `GR8_LOAD_MERGE_KB=1`) in
`EXTRA_ENV`; DeepSeek-V4.1-Flash from the recipe image `sha-7c2088a` with its stock `.env` (fast mode) plus
`GR8_GEN_CACHE`; GLM-5.3-Flash from its two-box recipe with the stock `.env` (no compact copies). Warm: each model booted once before the timed runs. Two switches each way (n=2), alternated. All 12
switches exited 0, and after each one the new model answered a chat. "Total" runs from the switch command to the
head's `/health` ready. The README's switch card shows the mean total of each row, rounded to whole seconds:

| Switch | down | up | health: ready in | total | mean |
|---|---|---|---|---|---|
| GLM-5.3 to DeepSeek-V4.1-Flash | 2.953 / 4.786 s | 0.775 / 0.662 s | 18.500 / 18.647 s | 22.227 / 24.095 s | 23.2 s |
| DeepSeek-V4.1-Flash to GLM-5.3 | 4.576 / 4.428 s | 0.686 / 0.634 s | 19.548 / 19.628 s | 24.810 / 24.690 s | 24.8 s |
| GLM-5.3 to GLM-5.3-Flash | 4.743 / 4.204 s | 0.668 / 0.656 s | 24.626 / 24.547 s | 30.037 / 29.408 s | 29.7 s |
| GLM-5.3-Flash to GLM-5.3 | 4.191 / 3.949 s | 0.651 / 0.610 s | 17.945 / 18.976 s | 22.786 / 23.535 s | 23.2 s |
| DeepSeek-V4.1-Flash to GLM-5.3-Flash | 4.465 / 4.300 s | 0.660 / 0.648 s | 24.811 / 24.650 s | 29.937 / 29.598 s | 29.8 s |
| GLM-5.3-Flash to DeepSeek-V4.1-Flash | 4.025 / 4.116 s | 0.645 / 0.605 s | 18.671 / 17.729 s | 23.341 / 22.449 s | 22.9 s |

Cold switches (empty page cache) were not measured.

Measured on 2026-11-08 (Pacific time) with the command at gr8 `a4cdfd10`, on four boxes (two DGX Sparks, two GB10
boxes with PCIe Gen4 NVMe), with each model's stock compose `.env` (GLM-5.3 at TP=4 with `GR8_GEN_CACHE` and
`GR8_QUIT_FAST=1`; DeepSeek-V4.1-Flash on all four boxes). Two switches each way (n=2), alternated, starting from
GLM-5.3. Every run exited 0, and each box's `.env` was byte-identical before and after:

| `./gr8 switch` | down | up | health: ready in | total |
|---|---|---|---|---|
| `ds41` (from GLM-5.3) | 2.715 / 4.260 s | 0.610 / 0.777 s | 35.564 / 35.202 s | 38.889 / 40.239 s |
| `glm53` (from DeepSeek) | 3.744 / 3.537 s | 0.593 / 0.574 s | 69.512 / 69.496 s | 73.848 / 73.607 s |

## The settings

**Both models: `GR8_GEN_CACHE`.** Put `GR8_GEN_CACHE=<dir>` in `EXTRA_ENV` on every box
([ENV.md](ENV.md), [FIRST-BOOT.md](FIRST-BOOT.md)). The loader keeps the tensors it computes at load and reads them
back at the next boot of the same engine build. The first boot of a build writes them. With `docker/compose.yaml`,
a directory in the cache volume works:

```bash
EXTRA_ENV=GR8_GEN_CACHE=/var/lib/gr8/cache/gen-cache
```

The DeepSeek recipe's `.env.example` has the same line for its home volume.

**GLM-5.3: a persistent SSD KV store.** In `docker/.env`:

```bash
KV_STORE=ssd:/var/lib/gr8/cache/kv,size=200G,persist=on,keep=sessions
GR8_STOP_GRACE=150s
```

At an orderly stop the engine writes the chats it still holds to the store. The next boot reads the store's index
back, so a chat's next turn restores its KV instead of a prefill. `GR8_STOP_GRACE` gives docker's stop the time the
engine needs for that write; without it, docker kills the engine mid-store
([docker/compose.yaml](../docker/compose.yaml)). `./gr8 knobs KV_STORE` describes every field.

**GLM-5.3: compact expert copies.** `GR8_LOAD_COMPACT=<dir> GR8_LOAD_MERGE_KB=1` in `EXTRA_ENV`, then
`docker compose run --rm gr8 prepare` once on every box to write the copies.
[The measured section](#compact-expert-copies-gr8_load_compact) has the numbers and the disk space they take.
GLM-5.3-Flash uses the same two knobs ([below](#compact-copies-on-the-pair)).

**GLM-5.3: `GR8_QUIT_FAST=1`.** In `EXTRA_ENV`. At an orderly stop, once the KV store has stored the pool's runs and
the links have closed, the engine ends right after the ranks meet at the quit barrier: the process exit frees its
memory, instead of the engine freeing the CUDA graphs, buffers and runtime one by one. GLM-5.3-Flash and DeepSeek
ignore it. [Measured below](#fast-quit-gr8_quit_fast). The default is 0 (off).

**DeepSeek: the release check.** The recipe checks its release at every start, but it hashes only the files that
changed since the last check, so a start after the first one spends about 1 s on it.

## One command each way

With GLM-5.3 in gr8's `docker/` directory and the DeepSeek recipe in its own, on every box:

```bash
# GLM-5.3 to DeepSeek
(cd gr8/docker && docker compose down) && (cd deepseek-v4.1-flash-exlr8-4x-spark && docker compose up -d)
# DeepSeek to GLM-5.3
(cd deepseek-v4.1-flash-exlr8-4x-spark && docker compose down) && (cd gr8/docker && docker compose up -d)
```

The head's `/health` reports `"ready":true` when the new model serves.

## Measured

2026-11-07, four GB10 boxes cabled in a ring (`SHAPE=ring`): two DGX Sparks and two GB10 boxes from another maker
whose NVMe is PCIe Gen4. GLM-5.3 from the image `ghcr.io/kindlingai/gr8:sha-4ed9e7b`, DeepSeek from the recipe's image
`ghcr.io/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark:sha-50a8773` (built on gr8 `4ed9e7b`). Warm (gen cache
written), two runs each way. The DeepSeek starts ran with `GR8_LOAD_PROF=1` (one start without it was not faster).
"Down" is the old model's `down` on all four boxes, until the last one has returned. "Wait after `up -d` returns"
runs from the moment the new model's `up -d` returns until `/health` reports ready. The total runs from the start of
`down` to ready, so it also holds the 3.2-3.4 s between `down` finishing and `up -d` returning, including
orchestration:

| Switch | Down | Wait after `up -d` returns | Total |
|---|---|---|---|
| GLM-5.3 to DeepSeek | 4.7-4.9 s | 35.9-36.8 s | 44.1-44.7 s |
| DeepSeek to GLM-5.3 | 2.8-3.5 s | 70.5-71.3 s | 77.1-77.3 s |

- The runs timed `down` on all four boxes together, not on each box. At the GLM-5.3 stop, each rank stored the chats
  it still held in its SSD KV store: in 45-194 ms on the DGX Sparks and 79-95 ms on the Gen4 boxes (both runs).
- Chats survive the switch. Four GLM-5.3 chats, resumed after GLM-5.3 to DeepSeek to GLM-5.3, answered their next
  turn in 0.16-0.21 s (time to first token, both runs). The resumed answers were the same in both runs. In an earlier
  run on the same images, the first turn of the same chats, a cold prefill, took about 3.8 s.
- The slowest boxes set each start. DeepSeek: the DGX Spark ranks finished their boot 13.8-15.1 s after their engine
  started, the Gen4 boxes 28.5-31.7 s, reading 79.6 GB of gen cache each. GLM-5.3: the DGX Sparks read their 78.3 GB
  of weights in 14.8-15.0 s, the Gen4 boxes in 59.2-60.9 s.
- `GR8_LOAD_MERGE_KB=1024` makes the loader read each chunk's file ranges in file order, with one read for ranges up
  to 1 MB apart. Each rank then read 223.4 GB of file for its 78.3 GB (about 2.9x). Measured on 2026-11-07, GLM-5.3 at
  TP=4, one build with the knob off and on, gen cache warm, two starts each way. On a DGX Spark's own NVMe, a rank's
  load went from 14.2-14.8 s to 11.8-11.9 s. On the PCIe Gen4 NVMe of the other two boxes, a rank's
  load went from 58.5-60.4 s to 57.3-58.9 s, 2-3%. Those boxes set each start: ready after `up -d` returned went
  from 70.1-70.5 s to 68.8-68.9 s. The default is 0 (off).
- First starts, which write the gen cache, from `up -d` returning to ready: DeepSeek 242 s, 79.6 GB on each box's
  disk; GLM-5.3 73 s, 4.8 GB a box. At TP=4 GLM-5.3 computes few tensors at load, so its gen cache saves little.
  The gen cache is keyed on the engine binary, so the first start after any image change also writes it. The warm
  numbers on this page are from the second start on.

### Compact expert copies: `GR8_LOAD_COMPACT`

The GLM-5.3 start is set by the slowest boxes reading their weights: at TP=4 each rank reads its experts as many small
ranges spread through the release files, and the Gen4 boxes read them at about 1.3 GB/s. `prepare` with
`GR8_LOAD_COMPACT=<dir>` writes each rank's expert ranges once into dense copies, and the loader then reads those
instead. The bytes loaded are the same ([COMPACT-WEIGHTS.md](COMPACT-WEIGHTS.md)). The copies take about 70 GiB on
each box, beside the release, which stays.

Measured on 2026-11-07 on the same four boxes, GLM-5.3 at TP=4, two runs each way. Compact: a local build of gr8 at
`a11ec5ee` (merged unchanged as `3269873`), copies written by `prepare` on every box, with `GR8_LOAD_MERGE_KB=1`
also set. Stock: the image `sha-4ed9e7b`, earlier the same day. Both set `GR8_GEN_CACHE`. These are two recorded
configurations, not an isolated compact-only A/B.

| Switch into GLM-5.3 | Stock total | With compact copies, total |
|---|---|---|
| from GLM-5.3-Flash (TP=2) | 76.2 / 76.3 s | 47.2 / 47.2 s |
| from DeepSeek-V4.1-Flash | 75.7 / 77.1 s | 46.8 / 47.2 s |

- The loader's time per rank (one boot each): on the DGX Sparks 18.9 s stock, 9.6-10.2 s with the copies; on the Gen4
  boxes 62.3-63.6 s stock, 23.5-30.8 s with the copies.
- Output: 8 fixed prompts at temperature 0 (`max_tokens=512`), replayed once after the four compact runs, gave the
  same text and total token counts as the stock image, 8 of 8.
- The copies match the release, rank, TP size and layout they were prepared for. A new release or layout needs a new
  `prepare` into a new directory.
- Prepare on a disk with room: a nearly full disk can fragment the copies. On one Gen4 box, the same build's 151 copy
  files lay in 24,032 pieces (extents) when `prepare` wrote them onto a disk that was 92% full (96% after), and in
  4,306 pieces when it wrote them again onto the same disk at 58% full. A second Gen4 box, 96% full before the same
  `prepare` (100% after), got 442 pieces.

### Fast quit: `GR8_QUIT_FAST`

Measured on 2026-11-07 (Pacific time) on the same four boxes, GLM-5.3 at TP=4, the stock SSD KV store
(`persist=on,keep=sessions`). One local build of gr8 (`632bf295` with the change merged as #440) in both arms; the
only difference was `GR8_QUIT_FAST=1` in `EXTRA_ENV`. Three stops with the flag off and three with it on, alternated.
Each stop was timed on every box from the start of its `docker compose down` until the engine process was gone. The
per-rank rows pool two ranks over the same three stops:

| Stop of GLM-5.3 | Flag off | Flag on |
|---|---|---|
| All four engines gone (n=3) | 4.41-5.83 s, mean 5.22 s | 2.43-3.20 s, mean 2.76 s |
| A DGX Spark rank (n=6) | 3.89-5.73 s, mean 5.02 s | 2.18-3.10 s, mean 2.53 s |
| A Gen4 rank (n=6) | 4.15-5.73 s, mean 5.05 s | 2.33-3.09 s, mean 2.67 s |

- Chats survive a fast quit. In a second run each arm first served six new two-turn chats of about 8,200 tokens,
  then stopped. Every stop stored all the chats the engine held (6, 12, 18 and 24 entries over four stops, the same
  on all four ranks), the next boot loaded the same count, and every chat's next turn was read from the store, 6 of
  6 each time. Those stops took 5.33 / 6.77 s with the flag off and 4.47 / 4.59 s with it on (all four engines
  gone, two stops each).
- Output after a fast quit: 8 fixed prompts at temperature 0 (`max_tokens=512`) on the next boot gave the same text and
  total token counts as before, 8 of 8.

## Changing width: `./gr8 <model> scale`

`./gr8 glm53 scale --tp N --peers LIST [--drain-timeout S]` (also `glm53f`) moves a serving cluster to another TP or
set of boxes without failing a request. Run it on any box of the cluster; the head (the first address) must stay first
in `LIST`. The boot's `--ips`/`--ips2` list the old members' rails and are dropped: give the new list's on the scale
command when its boxes need them (`--ips A1,B1 --ips2 A2,B2`; none at TP=1). It is an orderly restart of the engines,
not an in-process switch:

1. The head plans the new layout (GLM-5.3 with `bin/g6layout plan`, so a box keeps the expert halves it holds) and
   every box of the new list makes its rank's files ready and runs its checks while the cluster serves. A box new to
   the cluster joins by running `./gr8 <model> up --tp N --peers LIST` (with the cluster's other options) after the
   scale has started.
2. The head's front end holds new requests (`/internal/hold`, frontend/README.md) and waits for the running ones to
   finish, at most `--drain-timeout` seconds (300). A drain that runs out ends the scale with nothing changed.
3. Every old engine quits as for `down` (its KV store keeps the pool's runs), and each box serves its store folder to
   the new ranks through `bin/kvman-peer` on `CONTROL_PORT + 3` (7343 by default). The new member list boots with
   `GR8_KV_REWIDTH`, which rewrites the store for each new rank before it opens: GLM-5.3's persisted SSD store
   (`KV_STORE=ssd:...,persist=on`, sessions with `keep=sessions`) and GLM-5.3-Flash's aligned checkpoints
   (`FLASH_CACHE=aligned` with a persisted SSD store; not with `GR8_CP=1`). A chat's next turn then reads its run from
   the store instead of prefilling it again. Without such a store the scale works and sessions prefill again. The
   store must be on each box's local disk: the scale moves rank folders within it.
4. When the head's API answers, the hold ends and the held requests run on the new width. A box that left exits. The
   head logs, and the scale command prints, a timeline of the hold (`timeline: drained +0.1s, stopped +6.2s,
   handshake +1.4s, started +0.3s, ready +52.0s, serving +0.4s`: each part the seconds since the one before), and a
   scale's boot relays each rank's store rewrite, weight check, boot capture and load-cache lines to its supervisor log.

A scale also prepares what the new width loads, before the hold: with `GR8_LOAD_COMPACT=DIR` in the cluster's
`EXTRA_ENV` every box writes its new rank's compact copies into `DIR.tp<N>-<tag>` while the old width serves
([COMPACT-WEIGHTS.md](COMPACT-WEIGHTS.md)), and GLM-5.3's planned layout is kept as `out/scale/layout-<hash>.json`, so
a scale back to a layout seen before keeps its `GR8_GEN_CACHE` entries. The scale never turns compact copies on by
itself (about 75 GB of disk a box); name the knob on the `scale` command (`--extra-env`) to start using them.

If the new width fails to boot (a refused handshake, an engine that exits, `READY_TMO`), every box of the old list
puts its store back and boots the old list again, and the held requests run there. The new engines never served, so
they stop at once (SIGTERM, not the orderly quit), and a box of the old list that asks the head during its rollback
waits for it. The hold lasts the drain plus a boot (`READY_TMO` + 120 s, at least 900 s), and a rollback sets it again
for the old list's boot: clients need a request timeout at least that long. A supervisor that dies mid-scale is not
resumed: `down` and `up` as after any failure; a box that leaves stops after `JOIN_TIMEOUT` + `READY_TMO` without word
from the head.

With `POOL` (every box that may serve, head first, with `TP` the boxes that serve now; `deploy/k8s` sets both with
`tp`), `scale --tp N` needs no `--peers`: the new member list is the pool's first N. A box of the pool outside the
member list is a standby: it runs its supervisor only (state `standby`, `health` true), joins the scale that names
it, and a box that leaves stands by again instead of exiting. The head writes the last scale's options to
`out/scale.json` and boots that width at its next `up` while `TP` and `POOL` are the ones it was written under; the
other boxes take the head's member list when theirs differs. While the head's front end holds requests, the head's
`health` is true too (`the front end holds requests through a scale`), so a readiness probe keeps it in service.

What the scale keeps the same, so the store's blocks stay valid: the running variant (`--variant`), the KV format
(`--kv-format`, also when a TP default chose it, as int4 at GLM-5.3 TP=4), `SHARE_LIMIT` and `REUSE_GRID` are pinned.
It refuses `SHAPE=ring`, and a variant larger than the new width's default: a GLM-5.3 cluster that will scale to TP=3
boots TP=4 with `--variant bit260` (bit325, the TP=4 default, does not fit a rank at TP=3), and names no `--drafter`
(the drafter does not run at TP=3). It changes the running cluster only: it prints the option lines
(`TP=`, `PEERS=`, the pins) to put into every box's config for later boots, and a mentat cluster runs in peers mode
after its first scale. Output after a scale is that of a fresh boot at the new width given the same stored state; a
run that never scaled may give other tokens (another width computes other KV bits).

Measured on 2026-11-19 with GLM-5.3-Flash on two DGX Sparks (one cable, `--variant k2`, fp8 KV, `--cap 65536`,
`FLASH_CACHE=aligned` with a persisted `ssd:` store, the warm model on local disk), a request every 0.5 s through the
head's API the whole time, and three 26.8K-token prompts checkpointed at 24,576 rows before the first scale:

| scale | requests waited (hold set to cleared) | checkpoints carried | a stored prompt again |
|---|---|---|---|
| TP=2 -> 1 (`--extra-env GR8_MEM_MIN=6`) | 39.2 s | 3 of 3, verified | 3.1-3.3 s (27.7 s cold at TP=1) |
| TP=1 -> 2, the second box joining | 46.7 s | 3 of 3 on each rank | 2.1 s (15.8 s cold at TP=2) |

No request failed (0 of 53 not 200 across both scales); the drain took 0 s (the short requests had finished). Two attempts at TP=1
without `GR8_MEM_MIN=6` stopped at boot with 6.2 and 7.1 GiB left (k2's 93 GB device image, two prefill lanes) and
rolled back to TP=2: the requests waited 68 s and none failed, and the stored prompts hit again at TP=2. The checkpoint
carry needs the same `CAP`, prefill rows and lanes at both widths (they are in the store's compatibility line).

Measured on 2026-11-19 with GLM-5.3 on four DGX Sparks on a switch (`--variant bit260`, int4 KV, a persisted `ssd:`
store of 64G per rank with `keep=sessions`, the warm model on local disk), a request every 1 s through the head's API
during each scale (0.5 s during the rollback test), and three 32K-34K-token conversations stored before the first scale:

| scale | requests waited (hold set to cleared) | entries carried | a stored prompt again |
|---|---|---|---|
| TP=4 -> 3, one box leaving | 77.6 s | 14 on each rank | 0.21-0.22 s (52-57 s cold at TP=3) |
| TP=3 -> 4, the fourth box joining | 64.6 s | 17 on each rank | 0.16-0.27 s (20-22 s cold at TP=4) |

No request failed (0 of 376 not 200 across both scales); the drain took 0.0-0.5 s. A scale to a width whose rank files
are not on the boxes fetches them first, while the old width serves, within `JOIN_TIMEOUT`: the first TP=3 needed about
60 GB per rank. A scale forced to fail after the switch (`--ready-tmo 20`) rolled back to TP=4: the requests waited
201.0 s, most of it the new engines' 120 s orderly quit, and none failed (0 of 1305); the stored prompts hit again at
TP=4.

## The GLM-5.3-Flash pair (TP=2)

A two-box pair can also switch between GLM-5.3-Flash at TP=2
([the two-box recipe](https://github.com/kindlingai/recipes/tree/main/glm-5.3-flash/dgx-spark/2x/exlr8-gr8)) and the TP=4 models above. The
pair in the runs below was two DGX Sparks; the TP=4 models ran on their four-box set. Measured on 2026-11-07,
warm, two runs each way, with GLM-5.3-Flash on a local build of its recipe at `319ed60` with gr8 `4ed9e7b` (the
engine the recipe pins after its gr8 bump). Same columns as above; "Down" here is the parallel
`docker compose down` that stops and removes the old model's containers:

| Switch | Down | Wait after `up -d` returns | Total |
|---|---|---|---|
| GLM-5.3 to GLM-5.3-Flash (TP=2) | 4.9 s | 24.8 s | 31.3 s |
| GLM-5.3-Flash (TP=2) to GLM-5.3 | 3.4-4.3 s | 69.8-70.6 s | 75.8 s (both runs) |
| DeepSeek-V4.1-Flash to GLM-5.3-Flash (TP=2) | 3.2-4.2 s | 23.8-24.0 s | 28.5 / 29.3 s |
| GLM-5.3-Flash (TP=2) to DeepSeek-V4.1-Flash | 3.1-3.6 s | 34.0-35.4 s | 39.7 / 40.2 s |

- The pair's stop-and-remove on this engine build took 3.1-4.3 s. On the recipe's engine 32 commits back
  (gr8 `16351164`), the same phase took about 13-14 s, most of it the fixed quit wait that gr8 #429 removed.
- The pair's boot is loading its about 86 GB of weights (K4, the recipe's default at `319ed60`; the default since
  recipe `8f1992b` is bit349, about 76 GB) and starting the engines: the pack loads 21.2 s into the
  boot, the graphs capture at 22.1 s, and the engines report ready 23 s in. The first boot after the weights were
  in place and the warm restart both waited 24.8 s (rounded) after `up -d` returned; the first boot's whole-run
  total, with nothing to stop first, was 26.5 s. The runs kept the recipe's published setting, gen cache off.
- Chats survive the pair's round trip. Four Flash chats, their next turn replayed after a full round trip to the
  TP=4 model and back, answered in 3.1-5.9 s (total response time), all four with their context intact. These
  test prompts are short, and the prompt cache's checkpoints land at 4,096-token multiples, so the replays read
  their prompts fresh.

### Compact copies on the pair

Measured on 2026-11-07 (Pacific time) on the two DGX Sparks, GLM-5.3-Flash at TP=2, a local build of its recipe on
gr8 `d21ddaa`, copies written by `prepare` on both boxes (73 GiB a box: 43 copy files and their maps). On
GLM-5.3-Flash the copies pay only together with `GR8_LOAD_MERGE_KB=1`. Two experiments, each with its own stock runs,
each in the order stock, knob, knob, stock. The total is the whole switch from GLM-5.3 to GLM-5.3-Flash, until
`/health` reports ready; the read time is the loader's own for one rank:

| Switch GLM-5.3 to GLM-5.3-Flash | Stock, total (n=2) | With the knob, total (n=2) | A rank's read, stock / knob |
|---|---|---|---|
| `GR8_LOAD_COMPACT` alone | 34.2 / 32.8 s | 31.4 / 33.0 s | 19.6-19.9 s / 19.4-19.7 s |
| `GR8_LOAD_COMPACT` with `GR8_LOAD_MERGE_KB=1` | 33.5 / 33.5 s | 26.9 / 25.2 s | 19.5-19.9 s / 12.6-12.9 s |

- Without `GR8_LOAD_MERGE_KB`, each rank still read its about 298,000 ranges one read each; with it, ranges close
  together in one copy are read with one read.
- With both knobs, the switches into the pair (two runs each): from GLM-5.3 25.8 / 25.5 s, from DeepSeek-V4.1-Flash
  22.9 / 24.1 s. Earlier the same day, without the copies and on the recipe's previous build, the same switches took
  33.4 / 33.5 s and 30.7 / 29.5 s.
- Output: the same 8 fixed prompts gave the same text and token counts as the stock arm, 8 of 8.
