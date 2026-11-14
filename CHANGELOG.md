# Changelog

Every numbered `v*` release of `gr8` is recorded in this file alongside its machine-readable version spec in
[`releases/<tag>.json`](releases/) and its matching [GitHub Release](https://github.com/kindlingai/gr8/releases).
Format and release rules live in [`RELEASING.md`](RELEASING.md).

---

## [Unreleased]

Changes merged on `main` since `v1.0.1` that are not yet cut in a numbered `v*` release tag.

---

## [v1.0.1] — 2026-11-12

- **Tag / commit:** [`v1.0.1`](https://github.com/kindlingai/gr8/releases/tag/v1.0.1) · `6333c78a4ab4cfedc706793d689f54858c576d84`
- **Container image:** `ghcr.io/kindlingai/gr8:v1.0.1@sha256:02e44b9fc971f2acffe117df50de1be829462453287aab2c1d61a31a55738c0b`
- **Version spec:** [`releases/v1.0.1.json`](releases/v1.0.1.json)
- **Validation gate:** Because `v1.0.1` contains zero CUDA kernel, model-default, or HTTP wire-protocol changes relative
  to `v1.0`, this patch was tagged from `main` after passing the full CPU test suite and a live `4x GB10` (`TP=4`)
  `GLM-5.3` SSD session-store gate covering both stock and steered chats rather than a full multi-model `stable`
  throughput re-sweep.

### Highlights and measured numbers

- **SSD KV session store (`KV_STORE=ssd:...`): delta-only session extension, steered-chat persistence, and boot-time space reservation (`#484`, `f56657b9`; `#498`, `6333c78a`):**
  - **Delta-only session extension (`#484`):** With `KV_STORE=ssd:PATH,...,keep=sessions` (or `keep=all`), a store hit
    retains its stored entry, tracks full-page provenance on the continuing run, and extends the entry in place on
    eviction or orderly shutdown (`down`) by writing **only newly added page blocks** instead of rewriting the entire
    conversation (rewriting the changed partial page as well when reuse-grid alignment is disabled). Continuing hits on
    a session's own resident pool run no longer spill redundant copies on intermediate turns.
  - **Steered-session persistence and namespace isolation (`#498`):** `KV_STORE=ssd:...,persist=on,keep=sessions` (and
    `keep=all`) now journals and restores steered sessions (including `STEERING_FORCE=1` deployments) across orderly
    restarts and extends continuing steered sessions in place at `1.00x` write amplification. Each stored entry carries
    its exact 70-byte steering profile/scale namespace and remains invisible to unsteered (`scale=0`) requests or other
    steering selections (`keep=system` prefix indexes stay stock-only).
  - **Boot-time `fallocate` reservation (`#484`):** On Linux, opening an SSD block file calls `fallocate` (mode 0) for
    the full `size=` capacity (rounded down to whole page blocks) before the cache opens, preventing other files from
    claiming the budget and turning insufficient disk space into an immediate boot/open error rather than a delayed
    mid-serve write failure. Reopening an existing store reserves its extent without altering stored blocks.
  - **CPU pool write-amplification verification (`engine/kvman-rs/tests/c21_write_amp.rs` and `tests/stash.rs`,
    `64`-row pages, `4,096`-token reuse grid, `3,506,176 B` page blocks, including orderly quit drain):**
    - `1` chat, `10` turns, evicted between turns: `3,519` written / `2,964` rewritten blocks (`6.34x`) -> `512 / 0` (**`1.00x`**).
    - `1` chat, `10` turns, no competing traffic: `1,983 / 1,428` (`3.57x`) -> `512 / 0` (**`1.00x`**).
    - `2` chats, `8` turns each, evicting each other: `6,678 / 5,478` (`5.57x`) -> `1,152 / 0` (**`1.00x`**).
    - `4` chats sharing a `10,000`-token system prompt: `2,776 / 1,824` (`2.92x`) -> `896 / 84` (**`1.10x`**; `1.00x` when prompt-aligned to `8,192`).
    - `6` chats exceeding store capacity (undersized store): `8,360 / 6,318` (`4.09x`) -> `6,400 / 4,736` (`3.85x`).
  - **Live `GLM-5.3` (`TP=4`, `4x GB10`, `KV_FORMAT=int4`, `KV_STORE=ssd:...,size=16G,persist=on,keep=sessions`) verification:**
    - Restart and resume (`19K`-token chat on a `16 GB` SSD store): first-reply latency (TTFT) drops from **`12.75 s`
      cold to `3.04 s` from SSD (`4.2x`)** on plain chats and **`12.99 s` cold to `2.99 s` from SSD (`4.3x`)** on
      steered chats, restoring **`16,384` cached tokens in `~56 ms`**.
    - Write-only-new extension (`1.00x`): **`0 B`** written to SSD mid-chat; orderly quit writes **`256` pages** on the
      first session drain and **only `128` new pages (`1.00x`)** after extending the chat across a second quit.
    - Exactness and isolation: replies are **byte-identical** warm from SSD vs cold prefill on both plain and steered
      chats, and steered and plain SSD caches remain isolated.
- **Loader `QO_CACHE` directory creation and write diagnostics (`#493`, `434ae699`):** `engine/loader-rs`
  (`qo_cache_write`) creates missing `QO_CACHE` parent directories (`create_dir_all`), ignores empty `QO_CACHE=""`
  settings, uses per-write atomic temporary filenames (`.tmp<pid>.<seq>`), and logs
  `# qo-cache: write <path>: <error> (not cached, continuing)` when a cache write fails instead of silently skipping
  `.qo.bin` persistence on first boot.
- **Standalone recipe repositories and release documentation (`#492`, `#494`, `#495`, `#496`, `#497`):** Moves in-tree
  `GLM-5.3` recipe directories (`recipes/tp5-t5x/`, `recipes/tp6-k3-613k/`) and root `kindling.json` into standalone
  repositories (`glm-5.3-exlr8-4x-spark`, `glm-5.3-exlr8-5x-spark`, `glm-5.3-exlr8-6x-spark`), adds [`recipes.json`](recipes.json)
  (`#494`), renames the 5-box (`737K context`) and 6-box (`1M context`) recipe labels (`#496`, `#497`), and adds
  [`RELEASING.md`](RELEASING.md) (`#492`), `README.md#building-on-gr8`, and the keep-current checklist in
  [`AGENTS.md`](AGENTS.md) (`#495`).

### Breaking changes

None. `./gr8` CLI commands and flags, port `8100` HTTP endpoints and payloads, Hugging Face `exlr8` weight and steering
profile schemas, and 128 GB GB10 runtime memory and numerical contracts are backward-compatible with `v1.0`.

Operational notes when upgrading:
- **Linux `KV_STORE=ssd:...` free-space check at boot (`#484`):** Because `KV_STORE=ssd:PATH,size=N` preallocates the
  full `size=` budget with `fallocate` at startup instead of creating a sparse file, a `size=` value larger than
  available filesystem space (or on a filesystem without `fallocate` support) fails immediately at boot rather than
  mid-serve. Lower `size=` to fit free disk space if startup reports `ENOSPC`.
- **In-tree recipe paths moved (`#494`):** Automation referencing `recipes/tp5-t5x/` or `recipes/tp6-k3-613k/` inside a
  `gr8` checkout should use `kindlingai/glm-5.3-exlr8-5x-spark` and `kindlingai/glm-5.3-exlr8-6x-spark`.

### Default or cache-key changes

- **Model and launch defaults:** None. All default `KV_FORMAT`, `CAP`, `SESS`, `MTP`, attention/prefill fast paths, and
  rollback flags are unchanged from `v1.0`.
- **Cache keys and on-disk formats:** Neither the `QO_CACHE` `.qo.bin` key/layout nor the `KV_STORE=ssd:...` persistent
  index format changed (whole-index steered entries use existing namespace fields when `keep=sessions` or `keep=all` is
  active). As on every engine binary upgrade, `KV_STORE`'s build-compatibility check starts the persistent session index
  empty on the first boot of `v1.0.1` so active sessions prefill once on the new binary while `fallocate` reserves the
  existing block file in place without manual cleanup.

### Toolchain / Dockerfile sync

None. `Dockerfile` (`CUDA_VERSION=13.0.1`, `UBUNTU=ubuntu24.04`, and runtime `apt` packages) and `rust-toolchain.toml`
(`1.94.1`) are unchanged from `v1.0`.

---

## [v1.0] — 2026-11-12

- **Tag / commit:** [`v1.0`](https://github.com/kindlingai/gr8/releases/tag/v1.0) · `db6f2c17c7b05303b69ce667e416d10340aeb9b7`
- **Container image:** `ghcr.io/kindlingai/gr8:v1.0@sha256:11ff2823184b0512ae62e0f1c281c46941cfb0fd81096e1dac4c313ad3fe9413`
- **Version spec:** [`releases/v1.0.json`](releases/v1.0.json)

### Highlights and measured numbers

`gr8 v1.0` promotes the fastest validated GB10 serving configurations for **DeepSeek-V4.1-Flash (`TP=4`)**,
**GLM-5.3 (`TP=4`)**, and **GLM-5.3-Flash (`TP=2`)** to out-of-the-box `./gr8` defaults with one-line CLI/env rollbacks
to the prior or exact reference paths, and refreshes the multi-node `README.md` benchmark cards (`d5543af0..db6f2c17`).

| Configuration | Decode (`code / prose / structured`, tok/s) | Cold Prefill (tok/s) | Concurrent Aggregate Decode (tok/s) | Measured 1-Chat `max_model_len` | Quality & Recall |
|---|---|---|---|---|---|
| **GLM-5.3 (`TP=4`, 4x GB10, `KV_FORMAT=int4`, `SESS=8`)** | `63.3 / 37.8 / 78.0` (`counting: 81.5`) | `32K: 1,540`; `128K: 1,361` | `1 / 2 / 4 / 8` streams: `47.7 / 59.8 / 69.5 / 85.8` | `513,522` tokens (`CAP=513536`, `+5.8%` vs `fp8` `485,362`) | `14/14` smoke, `73/75` check (`fp8: 74/75`), `128K` needle found |
| **GLM-5.3-Flash (`TP=2`, 2x GB10, `TUNE=speed`, `SESS=4`, `MTP=3`)** | `76.5 / 41.8 / 93.1` | `8K: 1,670`; `32K: 1,729`; `128K: 1,595` | `1 / 2 / 4` streams: `55.1 / 70.3 / 82.9` | `262,144` tokens (`CAP=2097152`) | `14/14` smoke, `128K` needle found |
| **DeepSeek-V4.1-Flash (`TP=4`, 4x GB10, `TUNE=speed`, `SESS=32`)** | `137.5 / 83.9 / 157.5` | `32K: 7,720`; `128K: 8,408`; `256K: 8,344` | `8 / 16 / 32` streams: `202.9 / 238.7 / 233.3` | `327,670` tokens (`32` x `327,680`-token slabs) | `11/11` smoke, `128K` needle found |

- **DeepSeek-V4.1-Flash (`TP=4`): 12,288-row wide v2 prefill, layer-20 tail hoist, and compact send + FFN unpack (`#486`, `d7dd706e`):**
  4-boot ABBA cold-prefill TTFT drops **`-8.25%` at `32K`** (`4.271 -> 3.918 s`, `7,637 -> 8,323 tok/s`),
  **`-10.68%` at `128K`** (`17.102 -> 15.276 s`, `7,645 -> 8,559 tok/s`), and **`-12.07%` at `256K`**
  (`35.828 -> 31.504 s`, `7,309 -> 8,313 tok/s`), with decode unchanged (`84.3` prose / `142.4` code tok/s), `6/6` cold
  `32K` + `128K` needles passed, and paired `75`-item `qeval` `72/75` vs `73/75` (`McNemar p = 1.000`).
- **GLM-5.3 (`TP=4`, `KV_FORMAT=int4`): bit-exact INT4 decode attention fast paths (`#487`, `49e23dcc`):**
  Zero-spill quantized-Q score tiles, compact INT4 context reduction, and fused post-attention inverse-Hadamard output
  rotation lift long-context CP decode geomean (`32K`/`64K`/`120K`) by **`+6.13%`** and short-context 1-stream decode
  geomean by **`+1.29%`** with literal **`mean_kl = 0.0`** (`0` differing bytes across `32,266` tensor checks).
- **GLM-5.3 (`TP=4`, `KV_FORMAT=int4`): folded-Hadamard weights, tensor-core INT4 decode, and `CP_ONECHAIN` (`#488`, `c2220f2e`):**
  Folds `128x128` Walsh-Hadamard blocks at `g6load` time (`GR8_KV4_FOLD=1`), enables BF16 `wmma` tensor-core INT4 decode
  attention (`GR8_KV4_TC=1`), and combines remote CP head groups in one chain (`GR8_CP_ONECHAIN=1`), lifting
  long-context CP decode geomean by **`+18.57%`** and short decode geomean by **`+3.69%`** against the bit-exact
  `KV_FORMAT=int4` baseline (`20/20` `32K`/`64K`/`120K` needles passed; paired `75`-item `qeval` `72/75` vs `71/75`).
- **GLM-5.3 `TP=4` default `KV_FORMAT=int4` and GLM-5.3-Flash `TP=2` INT4 shared waterfill (`#490`, `a555ee2a`):**
  `./gr8 glm53 up` at `TP=4` defaults to `KV_FORMAT=int4` (`bin/g6m4-int4`, `CAP=513536`, single-chat `max_model_len`
  `513,522` vs `485,362` on `fp8`, **`+5.8%`** capacity, `KL(fp8 || int4) = 0.01450`), and `./gr8 glm53f --tp 2 --kv-format int4`
  defaults to `SESS=4`, `MTP=3`, and shared verify-row waterfill (`68.3` vs `61.5 tok/s` at `2` streams, **`+11.0%`**;
  `79.2` vs `59.6 tok/s` at `4` streams, **`+32.9%`**, `mean_kl = 0.0`).
- **README model cards and per-chat KV limit schema (`#483`, `41b2d070`; `#491`, `db6f2c17`):** Adds the four-node
  `GLM-5.3` card, refreshes the `GLM-5.3`, `GLM-5.3-Flash`, and `DeepSeek-V4.1-Flash` cards with `v1.0` numbers, and
  adds `max_tokens_per_chat` / `max_tokens_per_chat_note` to `bench/gif.py`.

### Breaking changes

None. `v1.0` establishes the stable `./gr8` CLI (`glm53`, `glm53f`, `ds41`, `switch`, `kv`, `knobs`, `models`, `query`,
`perftest`, `test`), OpenAI-compatible HTTP API on port `8100` (`PROTOCOL.md`), cluster ports (`7340`, `7341`, `8100`),
and Hugging Face `exlr8` weight and steering profile schemas.

### Default or cache-key changes

- **DeepSeek-V4.1-Flash (`TP=4`) prefill default (`#486`):** `./gr8 ds41 up` enables `GR8_DS_PF_V2=1`,
  `GR8_DS_PF_V2_W16K=1`, `GR8_DS_PREFILL_W=12288`, `GR8_DS_PF_V2_L20_TAIL=1`, and `GR8_DS_PF_V2_SEND_FFN_OPT=3`
  (cold `240`-position teacher-forced `mean_kl = 0.002218`, `99.17%` top-1 agreement vs prior `8,192`-row path).
  Rollback to exact `4,096`-row reference: `--exact 1` (`EXACT=1`).
- **GLM-5.3 (`TP=4`) default `KV_FORMAT=int4` and INT4 fast stack (`#487`, `#488`, `#490`):** `./gr8 glm53 up` at `TP=4`
  defaults to `KV_FORMAT=int4` (`CAP=513536`, `GR8_KV4_SC=4`, `GR8_KV4_SC_TR=2`, `GR8_KV4_QT=1`, `GR8_KV4_CTX=2`,
  `GR8_KV4_FUSE_ROT=1`, `GR8_KV4_FOLD=1`, `GR8_KV4_TC=1`, `GR8_CP_ONECHAIN=1`; `KL(fp8 || int4) = 0.01450`, folded/TC vs
  bit-exact INT4 `mean_kl = 0.015803` under live adaptive serving). Rollback to `fp8`: `--kv-format fp8` (`CAP=485376`).
  Rollback to bit-exact `KV_FORMAT=int4` (`KL = 0.0`): `--kv4-tc 0 --kv4-fold 0 --cp-onechain 0`.
- **GLM-5.3-Flash (`TP=2`, `KV_FORMAT=int4`) multi-session waterfill (`#490`):** Defaults to `SESS=4`, `MTP=3`,
  `GR8_S2_SHARED_WATERFILL=1`, and `GR8_S2_SHARED_WF_MARGIN=0.02` (`mean_kl = 0.0`). Rollback: `--s2-shared-waterfill 0`
  (restores `MTP=1` when `--mtp` is unset) or `--sess 1`.
- **Cache keys:** No manual cache cleanup required (`g6load` folds Hadamard weights in memory at load time when
  `GR8_KV4_FOLD=1` without altering cached shard files).

### Toolchain / Dockerfile sync

Initial `v1.0` baseline:
- **CUDA / OS base:** `nvidia/cuda:13.0.1-devel-ubuntu24.04` and `nvidia/cuda:13.0.1-runtime-ubuntu24.04` (`linux/arm64`, `sm_121a`).
- **Rust toolchain:** `1.94.1` (`minimal` profile across `rust-toolchain.toml` and crate pins).
- **Runtime `apt` packages (`Dockerfile`):** `libibverbs1`, `ibverbs-providers`, `ibverbs-utils`, `python3`,
  `python3-yaml`, `curl`, `jq`.
