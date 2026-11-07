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
whose NVMe is PCIe Gen4. GLM-5.3 from the image `ghcr.io/kindlingai/gr8:sha-7962e74`, DeepSeek from the recipe built
on gr8 `7962e74`. Each time is the old model's `down` on all four plus the new model's `up -d` until `/health`
reports ready, warm (gen cache written), two runs each way:

| Switch | Down | Up to ready | Total |
|---|---|---|---|
| GLM-5.3 to DeepSeek | 19.3-21.1 s | 34.9-35.6 s | 58.0-59.3 s |
| DeepSeek to GLM-5.3 | 12.8-13.4 s | 71.0-71.2 s | 87.2-87.5 s |

- Chats survive the switch. Four GLM-5.3 chats of ~5,400 tokens each, resumed after GLM-5.3 to DeepSeek to GLM-5.3,
  answered their next turn in 0.16-0.20 s (time to first token, both runs). Their first turn, a cold prefill, took
  3.8-4.2 s. At the stop, GLM-5.3 stored 5 entries a rank in 169-390 ms. The resumed answers were the same in both
  runs.
- The slowest boxes set each start. DeepSeek: the DGX Spark ranks were ready 13-15 s after the start, the Gen4 boxes
  27-30 s, reading 79.6 GB of gen cache each. GLM-5.3: the DGX Sparks read their 78.3 GB of weights in 15 s, the Gen4
  boxes in 59-60 s. The loader, not the drive, sets that time: it reads at 31-46% of each drive's sequential speed.
- First starts, which write the gen cache: DeepSeek 226 s up to ready, 79.6 GB on each box's disk; GLM-5.3 73 s,
  4.8 GB a box. At TP=4 GLM-5.3 computes few tensors at load, so its gen cache saves little.
