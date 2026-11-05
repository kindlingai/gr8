# Booting DeepSeek-V4.1-Flash: the whole process

`ds41` serves [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four GB10
boxes. It takes the same two settings as every other model ([DEPLOY.md](DEPLOY.md)), and it needs two things GLM does
not: the release is read in place, and Engram needs a token map you write once. This page is the whole boot in order.
`./gr8 models` does not list `ds41` and the README's table does not hold it, so start here.

[FIRST-BOOT.md](FIRST-BOOT.md) is the companion: it asks what you will run on the model and gives the measured numbers
for each answer (how many chats, how long a context, speed against exactness). Read this page to get it serving, that
one to choose its settings.

## 1. What you need

- **Four GB10 boxes**, set up as the README's [What you need](../README.md#what-you-need) and
  [DEPLOY.md](DEPLOY.md#before-you-start) say: a RoCE v2 fabric with two rails (a switch, or four boxes cabled in a
  ring with `SHAPE=ring`), TCP ports 7340 and 7341 open between the boxes, and proactive memory compaction off.
  `ds41` serves at **TP=4** only.
- **The release**: `https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native`, the official weights in the
  exlr8 native layout. Unlike the GLM releases, each box does **not** fetch its own part into a model cache: the
  release is a local copy, read in place ([DEPLOY.md](DEPLOY.md#the-model)). Fetch it to local NVMe on every box
  before you boot.
- **Disk**: well over 100 GB a box. The copy on one of our boxes measures 134 GB, 78 GB of it routed experts and
  57 GB the dense files, which include the two Engram tables (12 files of about 4.4 GB each).
- **No drafter.** `ds41` drafts with the checkpoint's own MTP layers (DSpark), so leave `DRAFTER` out of the config.
  The check refuses a boot that names one.

## 2. Engram: the token map you write once

Engram is part of the model, not a trade-off: the engine reads n-gram table rows at layers 1 and 14 on every step.
Turn it on in `EXTRA_ENV` on every box, with the path prefix of this rank's Engram files:

    EXTRA_ENV="GR8_DS_ENGRAM=1 GR8_DS_ENGRAM_PREFIX=/opt/gr8/engram/ds41"

Two files sit at that prefix:

| file | who writes it |
|---|---|
| `PREFIX.engram.json` | the loader, at every boot: this rank's runs of the release's split table files (`g6engram-v2`) |
| `PREFIX.engram.cmap` | **you, once** — it is in neither the repo nor the release |

So the prefix only needs a writable directory: with `g6engram-v2` the table rows are read straight out of the
release's `dense/exlr8-split-001.engram.embed.p0NN.safetensors` and `exlr8-split-014.…` files, and no sidecar copy of
the tables is made.

The cmap is the compressed token map (516 KB). It is rank-independent: write it once and copy the same file to every
box. Its sha256 must equal `kCmapSha256` in `engine/engram/hash_consts.h`, so a wrong or stale file fails the boot
closed instead of serving wrong logits. Write it with the generator, from the release's tokenizer and the official
reference:

    python3 engine/engram/gen_engram.py \
      --ref <official DeepSeek-V4.1-Flash-1017 reference checkout> \
      --tok <release>/tokenizer.json \
      --out-h /tmp/hash_consts.h \
      --out-cmap /opt/gr8/engram/ds41.engram.cmap

It needs `torch` and `tokenizers`, and the reference's `inference/config.json` and `inference/engram.py`. Send
`--out-h` to a scratch path: `engine/engram/hash_consts.h` is checked in and the constants are never recomputed at run
time. If your scratch copy differs from the checked-in header, your reference is not the one the engine was built
against — stop there rather than boot.

Without `GR8_DS_ENGRAM_PREFIX`, `GR8_DS_ENGRAM=1` stops the boot. Engram together with `GR8_DS_MTP` additionally needs
`GR8_DS_ENGRAM_MTP=1` ([ENV.md](ENV.md)).

## 3. The config

`cluster.env` on every box, from [config/cluster.env.example](../config/cluster.env.example) with the GLM lines
replaced. The four addresses are in the same order on every box, and the first is the head: it serves the
OpenAI-compatible API on port 8100.

    PEERS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4
    MODEL=/opt/gr8/deepseek-v4.1-flash-exlr8-native
    EXTRA_ENV="GR8_DS_ENGRAM=1 GR8_DS_ENGRAM_PREFIX=/opt/gr8/engram/ds41"

No `DRAFTER` line and no `--variant`: the release has no width variants. `MODEL_CACHE` is where a box puts its part of
a Hugging Face `MODEL`; a local release is read where it lies. Boxes cabled without a switch add `SHAPE=ring`, in cable order. `./gr8 ds41 show-config` prints every value
and where it came from, and `./gr8 knobs` lists every option.

## 4. Boot and check

    ./gr8 ds41 check        # what would stop this box's boot
    ./gr8 ds41 up           # the same command on every box
    ./gr8 ds41 smoke        # smoketest/ds41.sh against this head's API
    ./gr8 ds41 status       # and health, logs, restart, down

`smoke` asks the questions a client relies on and whose answers can be checked: the served name
(`deepseek-v4.1-flash`), a one-prompt answer, the thinking switch, DSML tool calls and a forced `tool_choice`. A wrong
rank pack or a broken kernel can load, report healthy and serve fluent nonsense, so treat a `smoke` failure as a boot
failure.

Two served defaults differ from GLM's: a request that names no reasoning effort gets low (50), and
`DEFAULT_MAX_TOKENS` is 32768.

## 5. Then choose its settings

[FIRST-BOOT.md](FIRST-BOOT.md) has the measured numbers for the three questions that decide how the memory is split:
how many chats at once (`SESS`, with `GR8_DS_SESS_DRAFTS=1` for more than five), how long a context (`CAP`), and how
much speed you want for how much exactness. Every one of them is a config line or an `EXTRA_ENV` word; none of them
needs a rebuild.
