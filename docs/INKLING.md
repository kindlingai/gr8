# Inkling-2-Mini

Single-box NVIDIA DGX Spark (GB10, 128 GB) and single-GPU 32 GB Blackwell (RTX 5090) serving at `TP=1`. Weights: [kindlingai/inkling-2-mini-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/inkling-2-mini-exlr8-k2-k3-k4) (`exlr8-e4m3-pmcgx` routed experts at `K2`/`K3`/`K4`, unquantized BF16 non-expert weights, and a bundled 279 MB DFlash speculative drafter).

`gr8` serves Inkling-2-Mini (`Inkling2LoopForConditionalGeneration`) through `bin/g6m4_ink2`, built with `MODEL=ink2 GR8_WIDTHS=1` (`ARCH=121a` for GB10 / DGX Spark, `ARCH=120a` for RTX 5090). When `--mtp 7` is passed, `g6m4_ink2` automatically discovers the bundled DFlash drafter in the model directory with zero extra flags.

Three primary serving tiers share one weight pack (`--variant`):
- **Default — `bit325` (3.25-bit, `16.43 GB`):** In our 1,362-task evaluations, 3.25-bit performed nearly the same as 4-bit across coding and math except for graduate-level science (GPQA), while running faster and giving more KV headroom (`1.18M` KV pool at 8 streams on 1× Spark; `262K` at 4 streams / `303K` single-stream on 1× RTX 5090).
- **Higher quality — `k4` (4-bit, `18.72 GB`):** Recommended for maximum quality and graduate-level science (GPQA). Cuts KL by `-27.4%` vs Official NVFP4 and ties Official W4A4 on 1,362 tasks (`93.54%`, McNemar `p=0.6655`).
- **Fast version — `bit225` (2.25-bit, `13.37 GB`, `0` extra weight bytes):** `2,304` K2 + `768` loop-core K3 experts, **`0.041282` mean KL** (`-21.3%` lower KL than `k2`), hitting `133.5` Code / `228.2` `C=6` on 1× Spark and `354.4` Code / `768.3` `C=4` / `528.1` `C=6` Thinking-ON on 1× RTX 5090.

## Results (1× DGX Spark, `TP=1`)

Measured on 2026-11-25 on one NVIDIA DGX Spark (GB10, 128 GB) with local NVMe storage, `gr8` commit `6b874803` ([PR #615](https://github.com/kindlingai/gr8/pull/615)), bundled DFlash (`--mtp 7`), `GR8_INK2_PF_MX8_MOE=2`, `GR8_INK2_MX8_DENSE=2`, `GR8_INK2_MX8_LM_HEAD=0` (and `MX8_DENSE=1`, `MX8_LM_HEAD=1`, `GR8_S2_ADAPT=1` for `bit225` fast and `k2` fast).

### Single-stream 8-domain decode, `C=6` concurrency & 8K cold prefill (1× Spark)

| Preset (`--variant`) | Device image | Code | Math | JSON | HTML | Prose | Counting | `C=6` mixed | 8K prefill | 3-user 1K TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`bit325`** *(default)* | `16.43 GB` | `90.6` | `105.2` | `91.2` | `85.3` | `49.5` | `172.4` | `160.0` | **`3,484`** | `0.935 s` |
| **`k4`** *(higher quality)* | `18.72 GB` | `64.6` | `107.3` | `90.9` | `77.4` | `48.6` | `162.8` | `139.1` | `3,457` | `0.935 s` |
| `bit225` *(std)* | `13.37 GB` | `112.6` | `141.0` | `119.3` | `106.2` | `56.4` | `194.7` | `185.6` | `3,467` | `0.936 s` |
| **`bit225`** *(fast)* | `13.37 GB` | **`133.5`** | `160.2` | `144.4` | `112.7` | `75.5` | `211.7` | `228.2` | `3,474` | `0.845 s` |
| `k2` *(fast)* | `12.60 GB` | `100.5` | **`167.5`** | **`148.9`** | **`144.1`** | **`80.8`** | **`228.3`** | **`230.0`** | `3,475` | **`0.827 s`** |

All speeds are tokens/second (`C=6` homogeneous `6× Math` reaches `298.6 tok/s` on `bit325` and `479.5 tok/s` on `k2`).

### Cold prefill sweep (1× Spark, `bit325`, `CAP=139264`, `PREFILL=4096`)

| Prompt length | 4K | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Prefill speed (tok/s) | `3,436` (`1.19 s`) | `3,472` (`2.36 s`) | `3,150` (`10.40 s`) | `2,732` (`23.99 s`) | `2,162` (`60.63 s`) |

### Maximized concurrency & KV pools (1× Spark, `bit325`, live verified)

`g6m4_ink2` supports up to `8` concurrent sessions (`--s2-sessions 1..8`). Live verified on 2026-11-25 on 1× DGX Spark:

| Geometry (`CAP × SESSIONS`) | `PF_MX8_MOE` | Total KV pool | Boot time | Verified free OS memory | 37.8K prefill |
|---|---:|---:|---:|---:|---:|
| **`147,456 × 8`** *(default)* | `2` | **`1,179,648` (`1.18M`)** | `9.95 s` | **`9.52–9.69 GiB`** | `2,945 tok/s` |
| `163,840 × 8` | `2` | `1,310,720` (`1.31M`) | `9.92 s` | `5.45 GiB` | `2,876 tok/s` |
| `262,144 × 5` | `2` | `1,310,720` (`1.31M`) | `9.92 s` | `5.54 GiB` | `2,893 tok/s` |
| `196,608 × 8` | `1` | `1,572,864` (`1.57M`) | `9.72 s` | `9.12 GiB` | `2,589 tok/s` |
| `262,144 × 8` | `0` | **`2,097,152` (`2.10M`)** | `9.52 s` | `4.44 GiB` | `1,955 tok/s` |
| `1,048,576 × 1` *(full 1M)* | `2` | `1,048,576` (`1.05M`) | `10.5 s` | `7.8+ GiB` (`k4`) | — |

## Results (1× RTX 5090 32 GB, `TP=1`)

Measured live on 2026-11-25 on one NVIDIA GeForce RTX 5090 (32 GB, `sm_120a`), `gr8` commit `6b874803`, bundled DFlash (`--mtp 7`, **8/8 exact greedy parity** `1.90×–6.61×` speedup), `GR8_INK2_PF_MX8_MOE=0`, `GR8_INK2_MX8_DENSE=2`, `GR8_INK2_MX8_LM_HEAD=0` (plus `MX8_DENSE=1`, `MX8_LM_HEAD=1`, `GR8_S2_ADAPT=1` for `bit225-fast` and `k2-fast`):

### Single-stream 8-domain decode & concurrency (1× RTX 5090)

| Preset | Device image | Code | Math | JSON | HTML | Prose | Counting | `C=4` agg | `C=6` Think | NVMe boot | Dedicated 32 GB KV (`4×` / `1× long`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`bit325`** *(default)* | `16.43 GB` | `328.1` | `424.5` | `341.6` | `314.3` | `201.1` | `590.9` | `764.6` | `418.3` | `4.27 s` | **`262K`** (`64K×4`) / **`303K`** |
| **`k4`** *(higher quality)* | `18.72 GB` | `274.4` | **`438.5`** | `337.1` | `297.1` | `183.6` | `593.8` | `649.1` | — | `4.76 s` | `196K` (`48K×4`) / `245K` |
| `bit225` *(std)* | `13.37 GB` | `243.3` | `436.3` | **`384.1`** | **`343.4`** | `187.3` | **`604.8`** | `642.2` | `490.9` | `4.02 s` | `360K` (`88K×4`) / `385K` |
| **`bit225-fast`** | `13.37 GB` | **`354.4`** | `394.0` | `376.5` | `278.6` | `190.4` | `506.2` | **`768.3`** | **`528.1`** | **`3.76 s`** | `327K` (`80K×4`) / `385K` |
| `k2-fast` | `12.60 GB` | `244.5` | `418.7` | `360.6` | `336.9` | **`218.4`** | `510.8` | `608.4` | `516.5` | `3.76 s` | `360K` (`88K×4`) / `401K` |

### Cold prefill sweep (1× RTX 5090, `GR8_INK2_MX8_DENSE=2`, `bit325`)

With layer-wise streaming MXFP8 MoE prefill (`GR8_INK2_PF_MX8_MOE=3` 1-slot `0.98 GiB` scratch or `=4` 2-slot `1.96 GiB` ping-pong) and shared prefill buffers (`0.27 MiB/row` vs `5.06 MiB/row`), a 32 GB RTX 5090 runs `PREFILL=4096` alongside `262K` KV (`65536×4`) and bundled DFlash in `28.75 GiB` (`3.10 GiB` free at boot; leaves `~2.2 GiB` free even after `~0.9 GiB` lazy CUDA graph capture growth at long context — `0.96 GiB` less than v1.3.1 at `PREFILL=512`), doubling cold prefill across 4K–128K with `8/8` exact DFlash greedy parity (`C=4` `777.0 tok/s`):

| Mode / prefill chunk | 4K | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| **`PF_MX8_MOE=3`, `PREFILL=4096` (default, 0.98 GiB)** | **`10,076.1`** (`0.41 s`) | **`9,579.9`** (`0.86 s`) | **`8,728.5`** (`3.75 s`) | **`7,636.6`** (`8.58 s`) | **`6,056.6`** (`21.64 s`) |
| `PF_MX8_MOE=4`, `PREFILL=4096` (2-slot, 1.96 GiB) | **`10,395.5`** (`0.39 s`) | **`9,851.5`** (`0.83 s`) | **`9,010.7`** (`3.64 s`) | **`7,882.8`** (`8.31 s`) | `6,257.5`* (`20.95 s`) |
| `PF_MX8_MOE=3`, `PREFILL=2048` | `7,239.8` (`0.57 s`) | `6,964.6` (`1.18 s`) | `6,536.8` (`5.01 s`) | `5,927.3` (`11.06 s`) | `4,951.1` (`26.47 s`) |
| `PF_MX8_MOE=0`, `PREFILL=512` *(v1.3.1 baseline)* | `5,033.5` (`0.81 s`) | `4,990.9` (`1.64 s`) | `4,448.8` (`7.37 s`) | `3,851.5` (`17.02 s`) | `3,031.7` (`43.23 s`) |
| `PF_MX8_MOE=0`, `PREFILL=256` *(v1.3.1 `long`)* | `3,516.3` (`1.16 s`) | `3,426.8` (`2.39 s`) | `2,902.1` (`11.29 s`) | `2,407.5` (`27.22 s`) | `1,798.1` (`72.89 s`) |

*\*`PF_MX8_MOE=4` 128K measured on `k2` (`10,453.6` @ 4K, `6,257.5` @ 128K; `k2` `PF_MX8_MOE=3` matches `bit325` within 0.4%: `10,107.8` @ 4K, `6,074.2` @ 128K).*

## Quality & verification

- **40×2K (`81,920`-token) full-vocabulary `float64` KL vs BF16 (`PPL = 5.3840`):**
  - `k4` (`18.72 GB`, higher quality): **`0.010460`** mean / `0.001632` median (`96.91%` top-1, **`-27.4%` KL** vs Official NVFP4 `0.014407`).
  - `bit325` (`16.43 GB`, default): `0.015019` mean / `0.003027` median (`96.00%` top-1).
  - `bit250` (`14.13 GB`): `0.030018` mean / `0.007401` median (`93.93%` top-1).
  - `bit225` (`13.37 GB`, fast version, 0 extra weight bytes): **`0.041282`** mean / `0.011148` median (`92.70%` top-1, **`-21.3%` KL** vs `k2`).
  - `k2` (`12.60 GB`, E2E logit-KL scales): `0.052425` mean / `0.015119` median (`91.70%` top-1, **`-11.65%`** vs stock `k2`).
- **1,362-task controlled evaluation** (HumanEval+ 164, GPQA-Diamond 198, MATH-500 500, GSM8K-500 500, `temp=0.0`, `max_tokens=6144`, `reasoning_effort="low"`):
  - `exlr8 bit325` (default): **`92.80%`** (`1264/1362`) — nearly matches 4-bit on coding (`91.46%` HE+) and math (`97.80%` MATH-500), with the gap concentrated in graduate-level science (GPQA `72.22%`).
  - `exlr8 k4` (higher quality): **`93.54%`** (`1274/1362`) — statistical tie with Official NVFP4+MXFP8 W4A4 (`93.83%`, exact McNemar `p=0.6655`, exact tie on MATH-500 `98.00%`); recommended for maximum quality / GPQA.
- **DFlash exact greedy parity (`bit325`, `--mtp 7` vs `--mtp 0`):** **8/8 exact token-for-token** across all 8 domains on both 1× DGX Spark (`1.41×–5.18×`) and 1× RTX 5090 (`1.90×–6.61×`).
- **Live needle recall:** PASS on 1× DGX Spark (`bit325` `144K × 8` at `35.4K`; `k4` `256K × 4` at `36.6K` and `146.4K`) and on 1× RTX 5090 (`bit325` `PF_MX8_MOE=3` `PREFILL=4096` PASS at `35.4K` `8,328 tok/s`, `70.9K` `7,287 tok/s`, and `127.3K` `6,036 tok/s`; `k2` `262K` PASS at `243.7K` `3,851 tok/s`).

## Build and run

- **1× DGX Spark (Docker Compose, pinned `gr8` image, no build):** [inkling-2-mini-exlr8-1x-spark](https://github.com/kindlingai/recipes/tree/main/inkling-2-mini/dgx-spark/1x/exlr8-gr8)

```bash
npx @kindlingai/recipe inkling-2-mini-exlr8-1x-spark && cd inkling-2-mini-exlr8-1x-spark
hf download kindlingai/inkling-2-mini-exlr8-k2-k3-k4 \
  --local-dir ./models/inkling-2-mini-exlr8-k2-k3-k4
docker compose up -d
```

- **1× RTX 5090 (32 GB, native build from pinned commit):** [inkling-2-mini-exlr8-1x-rtx5090](https://github.com/kindlingai/recipes/tree/main/inkling-2-mini/rtx-5090/1x/exlr8-gr8) (standalone: [kindlingai/inkling-2-mini-exlr8-rtx5090](https://github.com/kindlingai/inkling-2-mini-exlr8-rtx5090))

```bash
npx @kindlingai/recipe inkling-2-mini-exlr8-1x-rtx5090 && cd inkling-2-mini-exlr8-1x-rtx5090
./build.sh
hf download kindlingai/inkling-2-mini-exlr8-k2-k3-k4 \
  --local-dir ./models/inkling-2-mini-exlr8-k2-k3-k4
./serve.sh                 # default: bit325 (64K x 4 = 262K KV pool)
# ./serve.sh k4            # higher quality / GPQA (48K x 4 = 196K KV pool)
# ./serve.sh bit225-fast   # fast 2.25-bit tier (80K x 4 = 327K KV pool)
# ./serve.sh long          # single-stream long context (303K on bit325)
```

To build `g6m4_ink2` from source directly:

```bash
# DGX Spark / GB10 (sm_121a):
ARCH=121a GR8_WIDTHS=1 MODEL=ink2 scripts/build.sh out/bin

# RTX 5090 / desktop Blackwell (sm_120a):
ARCH=120a GR8_WIDTHS=1 MODEL=ink2 scripts/build.sh out/bin
```

The OpenAI frontend runs with `--dialect inkling`, serves `inkling-2-mini`, and supports `--default-reasoning-effort none|low|medium|high` and `--allow-thinking-off`.

## The model

- **24 physical layers expanded to 48 virtual layers**: `prelude` layers `0–5`, `loop_core` layers `6–17` executed in `3` passes with rank-64 message-gate projections, and `coda` layers `18–23`.
- **Attention**: `24` Q-heads / `8` KV-heads (`head_dim=128`), sliding window `1024` on local layers and full attention on global layers `5, 11, 17, 23`, depthwise conv (`kernel=4`), and depth-attention residuals (`n_layers_per_attn_res_block=6`). Native context length `1,048,576`.
- **MoE**: `128` routed `exlr8-e4m3-pmcgx` experts per physical layer (`top-8`, hidden `2304`, intermediate `1152`, `nh=9`, `route_scale=8.0`).
- **DFlash speculative drafting**: Bundled `279 MB` Q4_0 DFlash drafter (`valid_vocab=200058`), auto-discovered when `--mtp 7` is set, with CUDA-graphed per-session-count drafting (`GR8_INK2_DF2_GRAPH=1`) and 2-bit shortlist head screening (`GR8_INK2_DF2_HEAD2=1`).

## Runtime knobs and rollback flags

Every `GR8_INK2_*` knob is registered in `knobs/ink2.yaml` (documented in [ENV.md](ENV.md)):

| Knob | Default | Purpose / rollback |
|---|---|---|
| `GR8_INK2_PF_MX8_MOE` | `2` | Blackwell grouped MXFP8 MoE prefill: `4` = layer-wise 2-slot double-buffered scratch (`~1.96 GiB`, overlapped with MoE GEMM & attention), `3` = layer-wise 1-slot streaming scratch (`~0.98 GiB`, overlapped with attention; ideal for 32 GB RTX 5090), `2` = cached MXFP8 on all 24 physical layers (`~23.5 GiB`, ideal for 128 GB Spark), `1` = cached looped core only (`1.57M` KV pool on Spark), `0` = off (`2.10M` KV pool on Spark). |
| `GR8_INK2_MX8_DENSE` | `2` | Blackwell MXFP8 dense projections: `2` = MXFP8 on prefill (`M>=64`) and BF16 on decode (`0.000000` decode KL delta), `1` = MXFP8 at all `M` (speed preset), `0` = BF16 always. |
| `GR8_INK2_MX8_LM_HEAD` | `0` | Blackwell MXFP8 LM head: `0` = BF16 always, `1` = MXFP8 at all `M` (speed preset), `2` = MXFP8 on prefill only. |
| `GR8_INK2_CUBLAS_PREFILL` | `0` | `1` = cuBLAS tensor-core GEMM for BF16 prompt projections (`M>=32`). Enabled in recipes. |
| `GR8_INK2_TC_PREFILL_ATTN` | `0` | `1` = tensor-core prefill attention (`M>=16`). Enabled in recipes. |
| `GR8_INK2_PF_FAST_ATTN` | `1` | `1` = 64-Q-row overlapped-SMEM prefill attention when `GR8_INK2_TC_PREFILL_ATTN=1`. `0` = 16-Q-row fallback. |
| `GR8_INK2_DEC_FAST_ATTN` | `1` | `1` = coalesced warp-per-head decode/verify attention kernel. `0` = stock decode attention. |
| `GR8_INK2_FAST_RES` | `1` | `1` = 1-pass dynamic-SMEM depth-attention kernels. `0` = stock multi-pass fallback. |
| `GR8_INK2_MSG_BF16` | `1` | `1` = BF16 looped message-gate projections for `M>64`. `0` = FP32. |
| `GR8_INK2_GPU_HEAD_PACK` | `0` | `1` = reduce head logits on GPU before transferring top candidates to host. Enabled in recipes. |
| `GR8_INK2_DF2_HEAD2` | `1` | `1` = 2-bit shortlist screen + exact BF16 top-64 rescore for DFlash draft LM head. `0` = full BF16 cuBLAS. |
| `GR8_INK2_DF2_GRAPH` | `1` | `1` = CUDA-graphed DFlash drafting per session count. `0` = eager launches. |
| `GR8_INK2_GRAPH` | `1` | `1` = CUDA-graphed trunk decode/verify steps on `TP=1`. `0` = eager launches. |
| `GR8_INK2_S2_COST` | *(auto)* | Optional 6-float override for `GR8_S2_ADAPT=1` adaptive speculative drafting cost model. |
