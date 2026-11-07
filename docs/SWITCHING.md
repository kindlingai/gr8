# Switching models on the same boxes

A box runs one gr8 model at a time. To move four boxes between GLM-5.3 at TP=4 and DeepSeek-V4.1-Flash at TP=4
([the DeepSeek recipe](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark)), stop one model and start
the other on every box. A two-box pair can also switch with GLM-5.3-Flash at TP=2;
[the pair's numbers](#the-glm-53-flash-pair-tp2) are at the bottom of this page. Both models keep their files on
the boxes, so a switch fetches nothing. Three settings make the switch fast. All three keep their data on each
box's local disk.

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

## The GLM-5.3-Flash pair (TP=2)

A two-box pair can also switch between GLM-5.3-Flash at TP=2
([the two-box recipe](https://github.com/kindlingai/glm-5.3-flash-exlr8-2x-spark)) and the TP=4 models above. The
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
- The pair's boot is loading its about 86 GB of weights and starting the engines: the pack loads 21.2 s into the
  boot, the graphs capture at 22.1 s, and the engines report ready 23 s in. The first boot after the weights were
  in place and the warm restart both waited 24.8 s (rounded) after `up -d` returned; the first boot's whole-run
  total, with nothing to stop first, was 26.5 s. The runs kept the recipe's published setting, gen cache off.
- Chats survive the pair's round trip. Four Flash chats, their next turn replayed after a full round trip to the
  TP=4 model and back, answered in 3.1-5.9 s (total response time), all four with their context intact. These
  test prompts are short, and the prompt cache's checkpoints land at 4,096-token multiples, so the replays read
  their prompts fresh.
