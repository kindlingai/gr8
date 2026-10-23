# TP5 T5X at 617K context (5x DGX Spark, GLM-5.3 full)

This branch is `adapt-ai-systems/gr8-tp5` main `0574fe5`: upstream `3bdd2cb` plus 22 commits (TP5 admission for a
five-rank mesh, the QO cache, KVB fp8, K1 split, L2 prefetch, CP_ONECHAIN, CP_STEP_ROWS, KSTOPB, the DSel cost table,
the T5X example config and `docs/TP5-SERVE.md`); it includes the heads of gr8 PRs #106-#114, merged locally. `git cherry main` marks
all 22 as missing from `main`. This directory adds the exact serving files we ran on top of it.

## What "T5X" is
T5X = the TP4 FSM decode set, on from the first TP5 boot, plus five knobs measured at TP5: `K1_TLBW=2048`,
`G6_PF_GEMMF32=1`, `G6_ATTN_CTX3=4`, `G6_PF_ISGRT_RG=32`, `G6_PF_ATTN_ROWS=512`. Also `PF_OWN=3` with
`G6_QO_CACHE` (13 GB per box; boot ~47 s with it, ~190 s without, the first boot writes it), `KVB=fp8`, `MTP=2`,
`CAP=617536` (max_model_len 617532), `SESS=4`, `SHARE_LIMIT=4096`. Output is byte-identical to the first checked TP5
build (T5S2): prefill KL against it 0.0 (max).

## Files
| file | what |
|---|---|
| `T5X.env` | the launcher env file of the measured serve; values identical to the deployed file (md5 `139ef131...`) |
| `boot.sh` | boot on all five boxes with `./gr8 glm53 up @ENV` under systemd-run, wait for `/health` 200 and max_model_len 617532 |
| `down.sh` | `./gr8 glm53 down @ENV`, then wait until no g6m4 is left on each box |
| `T5XR.env` | OPTION ONLY, NEVER RUN AS A SERVE: T5X plus a remote KV store on a sixth Spark (see below) |

`boot.sh` and `down.sh` are cleaned-up copies of `serve-t5x.sh`, the script that runs the measured serve (fleet
addresses, host memory scrub and menu hook removed). They were syntax-checked, not re-run in this form.

## Build and boot
Build with the TP5 widths: `G6_WIDTHS="4 5" F12_WIDTHS="4 5"`, Rust 1.94.1. Measured engine `bin/g6m4` md5
`2fa1d7e78d6d758a81a8078499ba58a8`. `MODEL` / `V2_DENSE` = the exlr8 K2-K3-K4 release, `V2` / `V2_SLICES` = each rank's
TP5 expert and index slices. On the head box, as root, with the same checkout path `H` on all five boxes:
```
BOXES="<head> <box2> <box3> <box4> <box5>" H=/path/to/gr8 bash recipes/tp5-t5x/boot.sh /path/to/T5X.env
BOXES="..." H=/path/to/gr8 bash recipes/tp5-t5x/down.sh /path/to/T5X.env
```
Before boot our script also drops the page cache and compacts memory on every box (`memclean.sh`, site-only).

## Numbers (2026-10-19; GPU clock capped at 2000 MHz unless noted; same-env rerun noise about +-2%)
| measure | 2000 MHz cap (serve) | full clock (1 run) |
|---|---:|---:|
| boot to `/health` 200 (QO cache present) | 46-47 s | |
| 8K cold prefill, tok/s | 1572 (2-run mean) | 1731 |
| 32K cold prefill, tok/s | 1569 (2-run mean) | 1652 |
| prose decode, 1 chat, tok/s | 39.8 | 41.9 |
| code decode, 4 chats, total tok/s | 77.4 | 80.1 |
| prose decode, 4 chats, total tok/s | 64.6 | 66.8 |
| first token, 157K prompt | 115.8 s (2-run mean) | 107.7 s |
| decode at 157K context, tok/s | 24.5 | 28.8 |
| 90K chat decode: 1 chat / 4 chats total, tok/s | 25.8 / 44.1 (2-run means) | |
| 15 min mixed agent load | 107 requests, 0 errors | not run |
| 4 chats x 139K at once | all resident, phase-2 TTFT 5.0-5.6 s | not run |
| needles 128K and 437K | found (measured on T5S2, same numerics) | |
| host free memory, lowest box | 2.62 GB under mixed load | 2.69 GB in the bench |
| GPU temperature max | | 73 C |
| canary / smoke | same canary every row, smoke 14/14 | same |

Full clock changes no output (same canary); memory and heat under long load were not measured at full clock.

## Measured at TP5 and left off
- `G6_L2PF`: 0.4 GB less free memory, no mixed-load gain.
- `PF_OWNPERM=1`: output changes (prefill KL 0.0031 vs T5S2), 8K prefill -6.7%.
- `QA_SPLIT=2`: 32K prefill -5%.
- `SHARE_LIMIT=8192`: the 4th 128K chat is evicted, 90K solo decode -4%.
- `KV_FORMAT=int4` with CAP 810,496 (+31% context): -1 to -4% speed, KL 0.0169 against the 0.01 gate. Option only.

## Option, never run as a serve: T5XR (remote KV store on a sixth Spark)
`T5XR.env` = `T5X.env` with `G6_HOME` pointing at a build of our `rkv` branch and two lines:
`KV_STORE=remote:<node>:7350,size=21G` and `KV_KEEP=sessions`. The sixth Spark runs `./gr8 kv up --gb 106` and
keeps evicted chats' KV in its RAM; a chat that comes back reads its blocks over RDMA instead of a new prefill.
The `rkv` commits (stores for steered runs: 864c387, 5c9fdc4, 2c1a4a7; copy-batch fixes 5f5b862; drain trace c5f674d)
are NOT in this PR. With `STEERING_FORCE=1` and without them, the store holds nothing.

Measured on 2026-10-20 only as test arms, never as a serve:
- Evict/restore A/B (10 chats x ~22K on a reduced pool, 30 requests per arm): restore TTFT p50 2.03 s with the store
  against 14.24 s without (-86%); outputs equal 30/30. Local NVMe tier: 1.87 s, but ~12.4 GB written per box per run.
- On the full T5X config: 8K / 32K prefill and prose decode 1570 / 1545 / 39.5 (T5X 1572 / 1569 / 39.8, inside the
  noise); needle at 500,460 tokens found; lowest free memory 2.45 GB (the store costs ~0.2 GB per rank).
- 4 chats x ~97.5K, pushed out of the GPU pool by a 355K prompt, one chat at a time: all 4 restored from the node,
  first token 1.26-5.51 s against ~67 s for a full prefill.
- Limits: the node is a sixth box with 106 GiB of locked RAM; five ranks share its one 200G port; the store is empty
  after any restart.

## Sources
Every number above comes from our run logs and reports: `reports/tp5/SERVE.md` (T5X rows 11:13Z / 11:24Z, the
full-clock row 12:06Z, the mixed / coresident / decode-gate rows), `reports/tp5/t2-t5s2.md` (needles),
`reports/tp5/kv4.md` (int4), `reports/remote-kv-tp5/RESULTS.md` and `BOOT-OPTION.md` (T5XR). Raw logs are on the
head box: `/root/tp5/t5q2.log`, `arms.log`, `arms-fullclk.log`, `runs/1019T120604Z-T5Xfullclk`.
