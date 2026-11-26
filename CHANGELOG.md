# Changelog

Releases of `gr8` are automated with `release-please` from Conventional Commits on `main`. See [`RELEASING.md`](RELEASING.md)
for versioning, release gates, and optional machine-readable specs in [`releases/`](releases/), and the
[GitHub Releases](https://github.com/kindlingai/gr8/releases) page for container digests and tester archives.

---

## [1.3.0](https://github.com/kindlingai/gr8/compare/v1.2.0...v1.3.0) (2026-11-26)


### Features

* **ci,docker:** build and publish linux/amd64 sm_120a container image ([#629](https://github.com/kindlingai/gr8/issues/629)) ([37ece27](https://github.com/kindlingai/gr8/commit/37ece27c067a6aea7d45fb019cabb27e24623157))
* **dispram:** support DISPRAM_KEY_FILE for spark-attic on stock DGX OS ([#614](https://github.com/kindlingai/gr8/issues/614)) ([4c82532](https://github.com/kindlingai/gr8/commit/4c8253270a3f98d2c130fd9b692eb2b5f48101f4))
* **ds41:** default DS41_MX=48 and first-boot engram cmap generation ([#630](https://github.com/kindlingai/gr8/issues/630)) ([9fb5370](https://github.com/kindlingai/gr8/commit/9fb5370beb8248316503928ab179c89b4260eb39))
* **engine:** add in-process delta piece-prefetch into ptab::Pool (EL3 Stage 3, [#151](https://github.com/kindlingai/gr8/issues/151)) ([#564](https://github.com/kindlingai/gr8/issues/564)) ([959df5a](https://github.com/kindlingai/gr8/commit/959df5aed648d4df1a006865d3c330186c54d553))
* **engine:** add in-process width-module lifecycle bridge and per-layout graph capture (EL7, [#155](https://github.com/kindlingai/gr8/issues/155)) ([#567](https://github.com/kindlingai/gr8/issues/567)) ([d13aefd](https://github.com/kindlingai/gr8/commit/d13aefdccb67fc298d114fb826cf2ef16de86d93))
* **engine:** add online per-step weight defragmentation for ptab::Pool (EL3b, [#164](https://github.com/kindlingai/gr8/issues/164)) ([#565](https://github.com/kindlingai/gr8/issues/565)) ([adf9c8e](https://github.com/kindlingai/gr8/commit/adf9c8e1354d3f6a79e4b1f36ccc5a1e237ab60e))
* **engine:** add ptab::Pool chunk release, contiguous-per-expert take_run (GR8_PTAB=3), and direct slab load (EL3, [#151](https://github.com/kindlingai/gr8/issues/151)) ([#561](https://github.com/kindlingai/gr8/issues/561)) ([e2caade](https://github.com/kindlingai/gr8/commit/e2caade62341b116bace33fabf3a9396bc603df2))
* **frontend,protocol:** add /v1/embeddings endpoint and embed wire frame ([#624](https://github.com/kindlingai/gr8/issues/624)) ([37ffb80](https://github.com/kindlingai/gr8/commit/37ffb80e2647e1f823f97ab9af30e6107ff00161))
* **glm53:** add TP=2 2-Spark support with native INT4 DFlash2 and adaptive drafting ([#617](https://github.com/kindlingai/gr8/issues/617)) ([8bbe6eb](https://github.com/kindlingai/gr8/commit/8bbe6ebf5b7edf4d648a8b39a0cbaa4d3d9abd6b))
* **ink2:** add Inkling-2-Mini looped engine, exlr8 MoE, and Blackwell speedups ([#611](https://github.com/kindlingai/gr8/issues/611)) ([bd1711c](https://github.com/kindlingai/gr8/commit/bd1711cc2945ee38a85f601053b0436bfe07a0b2))
* **ink2:** add melded DFlash2 drafter, Blackwell MXFP8 prefill, fast attention, and per-matrix kmix ([#615](https://github.com/kindlingai/gr8/issues/615)) ([6b87480](https://github.com/kindlingai/gr8/commit/6b8748038bc74cd746c2f83075b0fbb6ba5aa99e))
* **loader:** support multi-group exlr8 dense profiles ([#625](https://github.com/kindlingai/gr8/issues/625)) ([04790cc](https://github.com/kindlingai/gr8/commit/04790cc003bb79705417fc08b7a5528a1077ea6c))
* **qw38:** add QSpine-DF2 parallel block speculative drafter ([#609](https://github.com/kindlingai/gr8/issues/609)) ([903ad1b](https://github.com/kindlingai/gr8/commit/903ad1b1935532d7f1efbbd06544bf4bc575ac59))
* **qw38:** add single-GPU Strata-mode expert cache, AVX-512 K3 CPU MoE, and lazy MTP ([#610](https://github.com/kindlingai/gr8/issues/610)) ([5cedece](https://github.com/kindlingai/gr8/commit/5cedece3c4371e8dbf974078bdd22129c38e71c8))
* **qw38:** auto-fetch and auto-discover DFlash drafter pack and add 1x Spark recipe card ([#621](https://github.com/kindlingai/gr8/issues/621)) ([fa5d82d](https://github.com/kindlingai/gr8/commit/fa5d82d37751d8c71703e3d1e6f4c010427357d2))
* **s2:** add event-fence session KV export and cross-width canonical import (EL4b, [#159](https://github.com/kindlingai/gr8/issues/159)) ([#563](https://github.com/kindlingai/gr8/issues/563)) ([6e46b93](https://github.com/kindlingai/gr8/commit/6e46b93e8593d18edc372b9f4995bf6a4ed3dc26))
* **s2:** add in-process live width switch coordinator and --live scale (EL8, [#156](https://github.com/kindlingai/gr8/issues/156)) ([#568](https://github.com/kindlingai/gr8/issues/568)) ([12768a6](https://github.com/kindlingai/gr8/commit/12768a6a5df67b5a764898275925791a39269a3d))
* **s2:** add two-phase live bulk-then-tail multi-session KV migration (EL4c, [#160](https://github.com/kindlingai/gr8/issues/160)) ([#566](https://github.com/kindlingai/gr8/issues/566)) ([ea07f03](https://github.com/kindlingai/gr8/commit/ea07f039f3a796732e578bbbf53de5f2599643c0))


### Performance

* **kvman:** extend CP-tail stashes in place to avoid SSD write amplification ([#613](https://github.com/kindlingai/gr8/issues/613)) ([77e82fe](https://github.com/kindlingai/gr8/commit/77e82fef72ec1c3628338e08344fa98cc0da79e7))
* **runtime:** batch S2 page-table H2D uploads and O(N) CP page map ([#612](https://github.com/kindlingai/gr8/issues/612)) ([d4af92d](https://github.com/kindlingai/gr8/commit/d4af92d51d6f2313d7241cd3945b2ce5313a817d))

## [v1.2.0] — 2026-11-23

- **Tag / commit:** [`v1.2.0`](https://github.com/kindlingai/gr8/releases/tag/v1.2.0) · `ab2eb95f426ff908543d5d002218cf5a2a5c5be6`
- **Container image:** `ghcr.io/kindlingai/gr8:v1.2.0@sha256:3de05aab8cda21fd5d6b4af89ccdba6ff5abdbd704729601fd348fa638ade9fd`
- **Version spec:** [`releases/v1.2.0.json`](releases/v1.2.0.json)
- **Validation gate:** Full CI release build across all eight engine legs (including `Qwen3.8-Flash-Next` TP=1..4) and CPU test suite, plus live GB10 verification: `4x GB10` (`TP=4`) GLM-5.3 A/B gate (`#582`, 0.00000 KL on 6-commit 0-ULP + E4M3 DSA index MMA), `1x` & `2x GB10` (`TP=1`, `TP=2`) Qwen3.8-Flash-Next K3 (`bit300`) & K4 (`bit400`) live S2 `bench/bench.py` + 128K needle + MTP/prefix-cache/logprobs/forced-tool-choice suite (`#599`, `#600`), and `compute-sanitizer --tool memcheck` (`0 errors`) on `head_mask_test` (`NR=4/6`), `rstest` (`NR=4/6`), and `ptab_test_flash_tp4` (`#600`).

### Highlights and measured numbers

- **Qwen3.8-Flash-Next (`qw38`) full engine, vision, MTP, and Blackwell optimization pass (`#577`–`#581`, `#583`–`#594`, `#599`, `#600`):**
  - Adds `bin/g6m4_qw38` and `bin/m4sim_qw38` (`TP=1..4`) supporting 48-layer Gated DeltaNet + Full GQA/QSA + 4-stream hyper-connections + Layer-1 PLE NVMe/row-trellis embeddings (`#580`, `#589`, `#591`), Layer-48 MTP speculative decoding (`MTP=0..5`, `#585`, `#599`), opt-in 27-layer BF16 ViT vision with interleaved 3D MRoPE (`GR8_QW38_VISION=1`, `#578`), and `g6load`/`g6layout` dynamic `nh` planning (`#600`).
  - Optimization pass (`#599`) adds fast MTP drafting (`GR8_QW38_MTP_FAST=1`), trunk 2-bit head screen + exact top-256 FP8 rescore (`GR8_QW38_TRUNK_HEAD2=1`), Blackwell native MXFP8 Tensor Core dense GEMMs (`GR8_QW38_PF_MXFP8=1`), and read-only GDN verify with batched commit replay (`-2.69 GB` VRAM at `SESS=8`), freeing `~3.8 GB` VRAM per rank and unlocking full `CAP=139264` at K4 (`bit400`) on a single GB10 box.
  - **Live S2 `bench/bench.py` (`SESS=8`, `MTP=3`, `T=0.7`, `GR8_QW38_PLE_CACHE_MIB=4096`, 2026-11-23):**
    - **1× GB10 (`TP=1`) `bit300` (`CAP=139264`):** code **83.0 tok/s**, prose **61.5**, structured **87.4**, counting **88.8**; concurrency `c1` **72.6**, `c2` **102.1**, `c4` **121.7**, `c8` **155.2 tok/s**; cold prefill **2,258** (8K) / **2,269 tok/s** (32K); `8/8` smoke, `3/3` count-200, **128K needle PASS** (`131,572` tok). *(With `MTP=4`: code `89.5`, structured `97.2`, counting `99.1`.)*
    - **1× GB10 (`TP=1`) `bit400` (`CAP=139264`):** code **78.4 tok/s**, prose **55.1**, structured **81.2**, counting **82.0**; concurrency `c1` **65.5**, `c2` **93.9**, `c4` **106.7**, `c8` **140.6 tok/s**; cold prefill **2,215** (8K) / **2,200 tok/s** (32K); `8/8` smoke; `5.0 GiB` UMA headroom after graph capture.
    - **2× GB10 (`TP=2`, dual-rail RoCE) `bit300` (`CAP=262208`):** code **126.8 tok/s**, prose **89.2**, structured **135.1**, counting **135.8**; concurrency `c1` **104.1**, `c2` **151.2**, `c4` **183.5**, `c8` **254.5 tok/s**; cold prefill **3,027** (8K) / **3,142** (32K) / **2,973 tok/s** (128K); `8/8` smoke, `3/3` count-200, **128K needle PASS** (`131,069` tok).
    - **2× GB10 (`TP=2`, dual-rail RoCE) `bit400` (`CAP=262208`):** code **122.1 tok/s**, prose **86.9**, structured **126.7**, counting **128.6**; concurrency `c1` **101.7**, `c2` **145.7**, `c4` **169.6**, `c8` **234.1 tok/s**; cold prefill **2,997** (8K) / **3,069** (32K) / **2,915 tok/s** (128K); `8/8` smoke.
- **GLM-5.3 TP=4 throughput, concurrency, and dual-rail balance (`#582`):**
  - Zero-spill `M<=8` trellis decode, dedicated `_nt` weight dedup, unified capped `moe_rtr_fin_opt` (`1.27x–1.69x` faster on GB10), session-grouped S2 `idx31` scoring + 1-launch `decsel2`, fused `k_wait_peers`, symmetric 1-factorization non-striped dual-rail balance (`6/6` at TP=4, `28/28` at TP=8), and native E4M3×E4M3 `mma.m16n8k32` DSA prefill index scoring.
  - **Live `4x GB10` (`TP=4`, `MOE_UNION_CAP=48`, `KV_FORMAT=int4`, 2026-11-22):** `c1` **52.1 tok/s** (`+2.6%`), `c8` **124.2 tok/s** (`+9.1%`), post-32K decode **34.4 tok/s** (`+22.0%`), CP 4×37.4K warm decode **75.01 agg / 73.30 wall tok/s** (`+10.8%`), 32K prefill **1,678 tok/s** (`+4.6%`), 128K needle PASS, **`KL = 0.00000` / `100.0%` top-1**.
- **Pre-release bug sweep (`#600`, fixes `#576`):**
  - Fixes vocab-split head sampling and rejection sampling under grammar masking (`json_schema` / structured outputs) when a rank's entire vocab shard is masked to `-INFINITY` (eliminating `0 * inf = NaN`, out-of-range token ID `2147483647` / CUDA IMA, and out-of-nucleus leaks); verified with `head_mask_test` (`NR=4/6`, 10/10 modes) and `compute-sanitizer --tool memcheck` (`0 errors`).
  - Clamps `tr3_fc1_body_nt` tail tiles for non-divisible `steps % csteps` shapes (`ptab_test_flash_tp4` `158/158`, `memcheck` `0 errors`).
  - Fixes Qwen 3.8 unaligned prefix reuse (`stage_partial_page` + `idx_raw` state + auto-sized default `mem:` store up to `4 GiB`, cutting 97K warm TTFT from `41.1s -> 1.0s`), MTP verify rollback snapshots, concurrent prefill isolation, forced `tool_choice` with MTP, exact `logprobs` bypass when `logprobs` requested, streaming `logprobs` with thinking disabled in `g6front-rs`, and 96 MiB vision base64/RGB caps.
- **Engine architecture & CI infrastructure (`#489`, `#558`–`#560`, `#572`–`#575`, `#595`–`#597`):**
  - Data-driven `g6kv_layout_sizes` (`#489`), canonical DFlash2 ring / KDA session envelope (`#558`), `g6::FabricPool` (`GR8_FABRIC_POOL`, `#559`), VMM-backed resizable KV pool (`GR8_KV_CHUNKS`, `#560`), `glm-5.3-exlr8-3x-spark` recipe entry (`#572`), and faster parallel CI builds (`#573`, `#574`, `#596`, `#597`).

### Breaking changes

None. Existing `./gr8` CLI commands and flags, port `8100` HTTP endpoints and payloads, Hugging Face `exlr8` weight and steering profile schemas, and 128 GB GB10 runtime memory and numerical contracts remain backward-compatible with `v1.1.1`.

### Default or cache-key changes

- **New `qw38` default optimization knobs (`knobs/qw38.yaml`, `#599`):** `GR8_QW38_DENSE_FP8=1`, `GR8_QW38_MTP_FAST=1`, `GR8_QW38_TRUNK_HEAD2=1`, and `GR8_QW38_PF_MXFP8=1` are enabled by default on `qw38`. Roll back individually with `GR8_QW38_PF_MXFP8=0` (restores FP16-scaled FP8 decode + BF16 cuBLAS prefill; `mean KL = 0.000051`, `100.00%` top-1 vs full rollback), `GR8_QW38_TRUNK_HEAD2=0` (full-vocab FP8 `lm_head`), `GR8_QW38_MTP_FAST=0` (FP8 draft head + per-step host sync), or `GR8_QW38_DENSE_FP8=0` (pure BF16 dense decode).
- **Qwen 3.8 default fallback prefix store (`#600`):** Auto-sizes `mem:` store to `clamp((max_pps + 2*state_blocks + 16) * slot_bytes, 512M, 4096M)` when `--reuse-prefix` is enabled without an explicit `--kv-store` (previously fixed at `512M`). Override anytime with explicit `--kv-store`.
- **Launcher validation (`#600`):** `--shape ring` is rejected cleanly for `qw38` and `ink2` (mesh only), and `--flash-cache` is rejected for `qw38` (uses `--kv-store` directly).
- **Persistent `KV_STORE`:** As on any engine binary upgrade, the persistent session index resets once on first boot of `v1.2.0` while preserving the preallocated SSD block file.

### Toolchain / Dockerfile sync

Base CUDA (`13.0.1`), OS (`ubuntu24.04`), Rust (`1.94.1`), and runtime `apt` packages are unchanged from `v1.1.1`. `Dockerfile` adds the `engine-qw38` / `dist-qw38` stages shipping `bin/g6m4_qw38` and `bin/m4sim_qw38` (`#577`) and splits shipped test binaries (`dist-tests`, `#597`). Downstream GLM-5.3, Flash, and DeepSeek recipe Dockerfiles need no toolchain changes.

---

## [v1.1.1] — 2026-11-21

- **Tag / commit:** [`v1.1.1`](https://github.com/kindlingai/gr8/releases/tag/v1.1.1) · `a3f44a075854fc9af0ed56c6d43f7739705bc871`
- **Container image:** `ghcr.io/kindlingai/gr8:v1.1.1@sha256:fa5c9f66fbfa55d162782578b3262893b782e6365a76562bcab951b54acc6d96`
- **Version spec:** [`releases/v1.1.1.json`](releases/v1.1.1.json)
- **Validation gate:** Full CI build across all seven engine legs and CPU test suite plus live `4x GB10` (`TP=4`, 200G switch)
  long-context multi-session regression gate (`KV_STORE=ssd`, `MOE_UNION_CAP=48`, `GR8_KV_SHARE_PAGES=1`, `GR8_CP_PF_VERIFY=1`,
  `MTP=2`, 4 concurrent multi-turn sessions at 185K–195K across greedy `temp=0.0` and sampled `temp=0.7` runs).

### Highlights and measured numbers

- **Fix TP=4 150K+ multi-session corruption and empty-reply loops (`#557`):**
  - Orders `cp_pf_pre(i)` / `cp_gather(i)` ahead of `pf_ix(i - 1)` in `pf_step`, `pf_step2`, `pf_step2s`, and `m4sim`
    (`enqueue_pf` / `enqueue_pf2`) so staged peer index keys finish arriving over RDMA before `pf_ix` appends and scores
    layer `i` (while keeping `pf_ix` before `pf_set_step` so collective `c4` remains on layer-step `i - 1`). Adds CI
    regression gate `engine/tests/test_cp_pf_ix_order.py` and hardens direct RDMA receive-slot loads with `__ldcv`.
  - **Live `4x GB10` (`TP=4`, 200G switch, `bit325` v0.6, `int4` KV, full production defaults, 2026-11-21):**
    - Greedy (`temp=0.0`, exact seed that failed 8/12 on `v1.1.0`): **12/12 CLEAN** (0 empty replies, 0 corruption hits,
      0 `GR8_CP_PF_VERIFY` mismatches across all 4 ranks).
    - Sampled (`temp=0.7`, independent seed): **12/12 CLEAN** (**24/24 CLEAN** total, no prefill throughput regression).
- **TP=4/5/6 tuned launcher defaults (`#554`):** Extends the `v1.1.0` bit-exact tuned kernels, `DEC_BK_MAX=2048`,
  `L2PF=31`, `PF_GEMMF32=1`, MTP prefetch/lastrow/head2 tuning, `DFL_REUSE_MIN=0` (when `DFLASH=1`), and
  `MOE_UNION_CAP=48` to `TP=5` and `TP=6` alongside `TP=4`.
- **Dynamic capacity-scaled CP sharding in RAM (`#555`):** Adds a solo unsharded fast path and live `R1 -> R2` shrink
  as KV pool utilization increases.
- **Cluster scaling and standby pools (`#545`, `#547`, `#553`):**
  - `scale` prepares the new width's compact copies (`GR8_LOAD_COMPACT=DIR`) and content-addressed layout file before
    the hold (`#545`), keeping `GR8_GEN_CACHE` hits across repeated transitions.
  - Adds **`POOL`** (`--pool`) standby pool mode and Helm chart `tp` support (`#547`) so `gr8 <model> scale --tp N`
    changes serving width without pod restarts.
  - Ships `bin/g6layout` in the container image, `scripts/build.sh`, and release zip (`#553`).
- **Docs and runtime tests (`#546`, `#556`):** Documents how mixed-precision `int4` KV preserves quality and capacity
  (`#556`) and adds `img_handoff_test` (`#546`).

### Breaking changes

None. `./gr8` CLI commands and flags, port `8100` HTTP endpoints and payloads, Hugging Face `exlr8` weight and steering
profile schemas, and 128 GB GB10 runtime memory and numerical contracts remain backward-compatible with `v1.1.0`.

### Default or cache-key changes

- **GLM-5.3 TP=5 and TP=6 launcher defaults (`#554`):** Now enable the same bit-exact tuned kernels, `DEC_BK_MAX=2048`,
  `L2PF=31`, `PF_GEMMF32=1`, MTP prefetch/lastrow/head2 tuning, and `MOE_UNION_CAP=48` as TP=4 (and `DFL_REUSE_MIN=0`
  on TP=4 when `DFLASH=1`). Roll back MoE union capping with `MOE_UNION_CAP=0` (`--moe-union-cap 0`) or opt in to fast
  mode with `MOE_UNION_CAP=32`.
- **Persistent `KV_STORE`:** As on any engine binary upgrade, the persistent session index resets once on first boot of
  `v1.1.1` while preserving the preallocated SSD block file.

### Toolchain / Dockerfile sync

Base CUDA (`13.0.1`), OS (`ubuntu24.04`), Rust (`1.94.1`), and runtime `apt` packages are unchanged from `v1.1.0`.
`Dockerfile` adds `bin/g6layout` (`#553`) to `dist`. Downstream GLM-5.3, Flash, and DeepSeek recipe Dockerfiles need no
toolchain changes.

---

## [v1.1.0] — 2026-11-20

- **Tag / commit:** [`v1.1.0`](https://github.com/kindlingai/gr8/releases/tag/v1.1.0) · `cdd1e0931047f55bba8f682a0ae79c4198668ca3`
- **Container image:** `ghcr.io/kindlingai/gr8:v1.1.0@sha256:b22eedbd7902f65dbf2090897d75f5d92276cbf78a16bcde1f958aa56a306cf7`
- **Version spec:** [`releases/v1.1.0.json`](releases/v1.1.0.json)
- **Validation gate:** Full CI build across all seven engine legs plus live `4x GB10` (`TP=4`, 200G switch, uncapped clocks)
  `bench/bench.py` runs for GLM-5.3 `bit325` quality default (`MOE_UNION_CAP=48`), fast preset (`MOE_UNION_CAP=32`), and
  K4 `bit387` uncapped pure quality (`MOE_UNION_CAP=0`), alongside C=4 GSM8K-250, HumanEval-164, and sustained 4×153.6K
  long-context concurrency checks.

### Highlights and measured numbers

- **GLM-5.3 TP=4 default speedup, memory headroom, and quality/fast presets (`#532`–`#536`, `#539`, `#542`, `#548`, `#549`, `#551`):**
  - Enables all eight bit-exact decode, prefill, and DFlash2 kernel paths (`KL = 0.00000`) by default at TP=4 (`#542`, `#551`).
  - Defaults `DEC_BK_MAX=2048` (three decode graph buckets, `#549`) and shares scratch/workspace allocations (`#532`–`#536`),
    raising lowest `MemAvailable` after a full benchmark from **2.3 GB to 13.53 GiB** per box.
  - Adds batch-budgeted MoE expert union capping (`GR8_MOE_UNION_CAP` / `MOE_UNION_CAP`, `#548`), defaulting to **48** at TP=4
    (`#551`) for nearly lossless quality, with **32** as a fast preset and **0** for strict uncapped exactness (`KL = 0.00000`).
  - **Live `4x GB10` (`TP=4`, `bit325` v0.6 @ `3e388b45`, `int4` KV, DFlash2 + MTP, `bench/bench.py`, 2026-11-20):**
    - **Quality default (`MOE_UNION_CAP=48`):** code **67.1 tok/s** (+6.0% vs v1.0.1 `63.3`), prose **39.9** (+5.6%),
      structured **78.7**, counting **83.2**; cold prefill **1,457** (8K) / **1,605** (32K) / **1,470 tok/s** (128K, +8.0%);
      concurrency C=1 **50.6**, C=2 **62.1**, C=4 **83.3** (+19.9%), C=8 **120.5 tok/s** (+40.4% vs `85.8`);
      lowest `MemAvailable` **13.53 GiB** (`13.98–14.54 GiB` after boot); smoke `14/14`, 128K needle PASS (`130,813` tok),
      count `3/3` clean, cache `422/423`, `0` suspect chars; C=4 quality GSM8K-250 **246/250 (98.4%)**, HumanEval-164 **156/164 (95.1%)**.
    - **Fast preset (`MOE_UNION_CAP=32`, `--quick`):** code **73.2 tok/s**, prose **38.9**; concurrency C=4 **97.1** (+39.7%),
      C=8 **139.0 tok/s** (+62.0%); cold prefill **1,597 / 1,587 / 1,469 tok/s**; lowest `MemAvailable` **13.26 GiB**;
      `0` suspect chars; C=4 quality GSM8K-250 **242/250 (96.8%)**, HumanEval-164 **153/164 (93.3%)**, 8/8 needles at 4×153.6K.
    - **K4 (`bit387` @ `9e25c28a`, `MOE_UNION_CAP=0` uncapped, `DEC_BK=2048 DEC_BK_MAX=2048`, `CAP=442368`):**
      code **60.8 tok/s**, prose **36.2**, structured **72.9**, counting **76.5**; cold prefill **1,530 / 1,570 / 1,434 tok/s**;
      concurrency C=4 **62.3**, C=8 **82.0 tok/s**; lowest `MemAvailable` **3.78 GiB** (`4.32–4.72 GiB` after boot);
      smoke `14/14`, 128K needle PASS (`131,475` tok), count `3/3` clean, `0` suspect chars.
- **Live cluster width scaling (`./gr8 <model> scale --tp N --peers LIST`, `#538`, `#544`):** Orderly hold/drain/restart
  scaling for GLM-5.3 and GLM-5.3-Flash with automatic SSD KV store re-sharding (`GR8_KV_REWIDTH`, `bin/kvman-peer`),
  immediate rollback cleanup, hold timeline logging, and `Retry-After: 5` on front-end 503.
- **Inkling-2-Small (`ink2`, `#504`–`#507`, `#512`, `#517`, `#518`, `#520`):** Full text + vision (`GR8_INK2_VISION`) +
  MTP speculative decode on 4 and 6 Sparks (`bin/g6m4_ink2`, `bin/m4sim_ink2`).
- **DeepSeek-V4.1-Flash (`ds41`, `#500`, `#501`, `#510`, `#519`, `#523`, `#528`–`#531`):** V2 BF16 tensor-core projections on
  by default, multi-row decode GEMV and Engram optimizations, tool-turn cache fixes, and tiered session-store boundary state.
- **KV pool, prefix reuse, and front-end upgrades (`#502`, `#503`, `#514`, `#516`, `#521`, `#524`, `#526`, `#527`, `#540`, `#541`, `#543`, `#267`):**
  `REUSE_GRID=64` default at GLM-5.3 TP=4 int4, shared prefix pages and gated forks (`#540`), `GR8_CAP_TOTAL` (`#521`),
  `GR8_BUSY_*` 429 guard (`#516`), opt-in native `dispramd` KV backing (`#514`), literal control-token encoding (`#541`),
  GLM thinking-off control (`#526`), 1M+ token pool boot fix (`#543`), and device-table routed-expert stage 1 (`#267`).

### Breaking changes

None. `./gr8` CLI commands and flags, port `8100` HTTP endpoints and payloads, Hugging Face `exlr8` weight and steering
profile schemas, and 128 GB GB10 runtime memory and numerical contracts remain backward-compatible with `v1.0.1`.

### Default or cache-key changes

- **GLM-5.3 TP=4 defaults (`#524`, `#527`, `#539`, `#549`, `#551`):** Default width variant at TP=4/5/6 is `bit325` (`v0.6`;
  pass `VARIANT=home` for earlier `v0.5` checkpoints). `DEC_BK_MAX=2048`, `REUSE_GRID=64`, all 8 bit-exact `#542` kernels
  (`KL = 0.00000`), and `MOE_UNION_CAP=48` are enabled by default at TP=4. Roll back to strict uncapped exact routing with
  `MOE_UNION_CAP=0` (`--moe-union-cap 0`); opt in to fast mode with `MOE_UNION_CAP=32`.
- **DeepSeek-V4.1-Flash defaults (`#510`, `#519`):** V2 BF16 tensor-core projections on by default (`EXACT=1` restores exact
  path); forced tools default to native non-thinking mode (`--forced-tool-thinking 1` restores the old prompt).
- **Persistent `KV_STORE`:** As on any engine binary upgrade, the persistent session index resets once on first boot of
  `v1.1.0` while preserving the preallocated SSD block file.

### Toolchain / Dockerfile sync

Base CUDA (`13.0.1`), OS (`ubuntu24.04`), Rust (`1.94.1`), and runtime `apt` packages are unchanged from `v1.0.1`.
`Dockerfile` adds `bin/kvman-peer` (`#538`) and `bin/g6m4_ink2` / `bin/m4sim_ink2` (`#512`) to `dist`. Downstream GLM-5.3,
Flash, and DeepSeek recipe Dockerfiles need no toolchain changes.

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
