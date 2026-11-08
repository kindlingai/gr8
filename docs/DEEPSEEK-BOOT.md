# Booting DeepSeek-V4.1-Flash

`ds41` serves [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four GB10
boxes. It takes the same two settings as every other model ([DEPLOY.md](DEPLOY.md)), and it needs two things GLM does
not: the release is read in place, and Engram needs a token map you write once. This page is the whole boot in order.

[FIRST-BOOT.md](FIRST-BOOT.md) is the companion: it asks what you will run on the model and gives the measured numbers
for each answer (how many chats, how long a context, speed against exactness). Read this page to get it serving, that
one to choose its settings.

> **What this page is, and is not.** Every step and every number below was read out of this tree or measured on a
> four-box set. The `./gr8 ds41` route itself has been checked against the launcher source, not yet run end to end on
> four boxes from a clean machine — our own boots go through a separate test harness. So treat a surprise here as a
> gap in this page and say so, rather than assuming you did it wrong. The settings numbers in
> [FIRST-BOOT.md](FIRST-BOOT.md) come from that harness, not from this route.

## 1. What you need

- **Four GB10 boxes**, set up as the README's [What you need](../README.md#what-you-need) and
  [DEPLOY.md](DEPLOY.md#before-you-start) say: a RoCE v2 fabric with two rails (a switch, or four boxes cabled in a
  ring with `SHAPE=ring`), TCP ports 7340 and 7341 open between the boxes, and proactive memory compaction off.
  `ds41` serves at **TP=4** only.
- **The release**: `https://huggingface.co/kindlingai/deepseek-v4.1-flash-exlr8-native`, the official weights in the
  exlr8 native layout. Unlike the GLM releases, each box does **not** fetch its own part into a model cache: the
  release is a local copy, read in place ([DEPLOY.md](DEPLOY.md#the-model)). Fetch it to local NVMe on every box
  before you boot.
- **Disk: fetch the whole release, about 510.6 GB a box, and allow headroom.** That is the supported path, and the
  one to plan disk for. The published release is 510.60 GB over 604 files: 307.79 GB of weights and metadata, plus
  202.81 GB of Engram table parts.
- **Every Engram part file must be present.** The two tables are sharded into 23 parts each
  (`dense/exlr8-split-001.engram.embed.p000…p022.safetensors`, and the same for `-014`), 46 files in all — 4.43 GB a
  part, except `p022` of each table, which is 3.94 GB. The loader scans them from `p000` and **stops at the first one
  missing**, then adds up the row counts of every part it found to work out the table's size before it splits that by
  TP (`engine/loader-rs/src/ds41.rs:278-285`, `:930-948`). So a box missing `p000` finds no table at all, and a box
  holding only some of the parts computes the wrong shard and reads the wrong rows. **Do not delete, skip or truncate
  part files to save space.**

  With `GR8_DS_ENGRAM_COL_SHARD=0` (the legacy layout), each TP=4 rank only reads the rows in about 6 of the 23 parts, which is worth knowing when you are reading
  logs or sizing a cache — it is *not* a download list:

  | rank | parts holding its rows |
  |---|---|
  | 0 | `p000`-`p005` |
  | 1 | `p005`-`p011` |
  | 2 | `p011`-`p017` |
  | 3 | `p017`-`p022` |

  A boundary part (`p005`, `p011`, `p017`) holds rows for both of the ranks that share it, and each reads only its
  own. Fetching whole files for one rank's run plus the other files saves far less than it looks: 360.95 GB for
  rank 0, and more for ranks 1 and 2.

  Our own boxes measure 133.69 GiB (143.55 GB) each — 77.00 GiB routed experts and 56.66 GiB dense; the dense total
  already holds the 50.74 GB of Engram rows, which are not a third category. That is a **measurement of a
  sparse-staged copy**, not what downloading gives you: every one of the 46 part files is there with its real header
  and logical size, and only the byte ranges that rank reads were ever written, so `du` reads 143 GB where
  `du --apparent-size` reads 510 GB. We have no public tool that stages a copy that way, so treat 143 GB as our
  number and 510.6 GB as yours. **Those old contiguous-rank sparse copies are not compatible with the new
  column-sharded default.** Use a complete release copy, or explicitly keep `GR8_DS_ENGRAM_COL_SHARD=0` on all ranks.
- **No drafter.** `ds41` drafts with the checkpoint's own MTP layers (DSpark), so leave `DRAFTER` out of the config.
  The check refuses a boot that names one.
- **The DeepSeek binary, built on every box.** `up` does not build it for you:

      ./gr8 ds41 build        # this box's engine (bin/g6m4_ds41) and the front end (bin/g6front-rs)
      python3 -m venv venv && venv/bin/pip install -r frontend/requirements.txt   # the shim, rank 0 only

  `ds41` has its own engine binary, so a box built only for `glm53` is not ready for it — build `ds41` on all four.
  That one command also builds the front end, which only rank 0 runs. The shim venv is rank 0's too, and
  `frontend/requirements.txt` is empty by design: the venv itself is what the shim needs. `./gr8 ds41 check` names
  whichever piece is missing, and says which command would make it.

## 2. Engram: the token map you write once

Engram is part of the model, not a trade-off: the engine reads n-gram table rows at layers 1 and 14 on every step.
Turn it on in `EXTRA_ENV` on every box, with the path prefix of this rank's Engram files:

    EXTRA_ENV="GR8_DS_ENGRAM=1 GR8_DS_ENGRAM_PREFIX=/opt/gr8/engram/ds41"

Two files sit at that prefix:

| file | who writes it |
|---|---|
| `PREFIX.engram.json` | the loader, at every boot: this rank's runs of the release's split table files (`g6engram-v3-col` by default) |
| `PREFIX.engram.cmap` | **you, once** — it is in neither the repo nor the release |

So the prefix only needs a writable directory: with either layout the table rows are read straight out of the
release's `dense/exlr8-split-001.engram.embed.p0NN.safetensors` and `exlr8-split-014.…` files, and no sidecar copy of
the tables is made.

`GR8_DS_ENGRAM_COL_SHARD=1` (the default; all ranks must agree) assigns hash column `j` to rank `j % TP`
for both layers. Each rank must have the **complete** release table locally, not the old rank-only sparse ranges.
The loader then writes `g6engram-v3-col`: the same table fields, but `part = valid` is the sum of the owned column
lengths, `first` is the lowest owned global row, and `segs` tiles each owned column consecutively in ascending
column order (gaps between columns are intentional). Scales are concatenated in that order, with no padding;
row IDs and cache tags remain global. The engine checks this layout against its hash constants and refuses a
sidecar whose ownership mode disagrees with the knob. Non-divisors of 24 get unequal column counts, not dropped
columns. The loader's existing TP limits still apply. With 0, v1/v2 loading and emitted v2 JSON are unchanged.

On a complete release copy, the next boot regenerates the manifest in the new format; the cmap does not change.
Directly loaded legacy v1/v2 sidecars require `GR8_DS_ENGRAM_COL_SHARD=0`, or replanning from the complete release.
Replanning metadata does **not** fill missing bytes in an old sparse-staged copy: restore the complete release
before using the column-sharded default, or retain the explicit legacy setting. Boot checks owned weight and
scale ranges using the local filesystem's `SEEK_HOLE` support before reading tables; a reported hole, truncated
range or unsupported check stops boot with both fixes named. Holes outside this rank's owned ranges do not
trigger the check. This checks allocation coverage, not a content checksum, and never restages files automatically.

`GR8_DS_ENGRAM_IO_URING=128` is the default row reader queue depth. It uses aligned block reads on the fetching
thread without pread-pool workers. That default reader needs a seccomp profile that allows `io_uring_setup`,
`io_uring_enter` and `io_uring_register`. The stock Docker profile denies those calls with EPERM.
`GR8_DS_ENGRAM_IO_URING=0` selects the original pread pool. A denied setup logs the errno once and uses that
pread pool, sized by `GR8_DS_ENGRAM_IO_THREADS` (`engine/engram/rowstore.h`; tested with EPERM and ENOSYS). The
boot log identifies the selected reader. No container security change is required for the fallback.

The cmap is the compressed token map, 517,144 bytes. It is rank-independent: write it once and copy the same file to
every box. Its sha256 must equal `kCmapSha256` in `engine/engram/hash_consts.h`, so a wrong or stale file fails the
boot closed instead of serving wrong logits. That constant is checked in, so you can check your own file before you
boot — on a good copy:

    $ sha256sum /opt/gr8/engram/ds41.engram.cmap
    46ed86d2ee83d4b764e4390d38393a373886efdf2baeaa71a5370090d85b76de  …

which is the value in `engine/engram/hash_consts.h` and the same file on every rank.

Write it with the generator. It reads two files out of the official DeepSeek repo and the tokenizer out of the
release — not the original multi-hundred-GB checkpoint, so fetch just those two:

    pip install torch tokenizers numpy          # numpy is imported directly by the generator
    mkdir -p /opt/gr8/engram                    # nothing below creates this directory for you

    python3 engine/engram/gen_engram.py \
      --ref <reference dir holding inference/config.json and inference/engram.py> \
      --tok <release>/tokenizer.json \
      --out-h /tmp/hash_consts.h \
      --out-cmap /opt/gr8/engram/ds41.engram.cmap

The reference is the official `DeepSeek-V4.1-Flash-1017` repo; the exlr8 release records the revision it was built
from, `1bae0e1088993b9b82de08f59de5be5a34e42170`. The sha256 check below is what finally decides whether your
reference matches this engine. Send `--out-h` to a scratch path: `engine/engram/hash_consts.h` is checked in and the
constants are never recomputed at run time. If your scratch copy differs from the checked-in header, your reference is
not the one the engine was built against — stop there rather than boot.

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
a Hugging Face `MODEL`; a local release is read where it lies. Boxes cabled without a switch add `SHAPE=ring`, in
cable order. `./gr8 ds41 show-config` prints every value and where it came from, and `./gr8 knobs` lists every
option.

## 4. Boot and check

    ./gr8 ds41 check        # on every box: what would stop this box's boot
    ./gr8 ds41 up           # on every box, the same command
    ./gr8 ds41 smoke        # on rank 0: smoketest/ds41.sh against 127.0.0.1:8100
    ./gr8 ds41 status       # and health, logs, restart, down

`smoke` with no argument asks `http://127.0.0.1:<PORT>`, so run it on rank 0 — the head is the only box serving the
API. From any other box, give it the head: `./gr8 ds41 smoke http://<head>:8100`.

`check` is preflight, not proof: it reads the config, the binaries and this box's memory, and it never reads the cmap
or starts the engine. The cmap's sha256 is checked by the engine as Engram boots, and a cold graph capture happens
only on a real `up`. A clean `check` means nothing known will stop the boot, not that the boot works.

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
