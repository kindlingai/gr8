# Switching models on the same boxes

A box runs one gr8 model at a time. To move four boxes between GLM-5.3 at TP=4 and DeepSeek-V4.1-Flash at TP=4
([the DeepSeek recipe](https://github.com/kindlingai/deepseek-v4.1-flash-exlr8-4x-spark)), stop one model and start
the other on every box. Both models keep their files on the boxes, so a switch fetches nothing. Three settings make
the switch fast. All three keep their data on each box's local disk.

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
