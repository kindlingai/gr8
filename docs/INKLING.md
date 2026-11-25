# Inkling-2-Mini

Single-box NVIDIA DGX Spark (GB10, 128 GB) and single-GPU 32 GB Blackwell (RTX 5090) serving at `TP=1`. Weights: [kindlingai/inkling-2-mini-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/inkling-2-mini-exlr8-k2-k3-k4) (`exlr8-e4m3-pmcgx` routed experts at `K2`/`K3`/`K4`, unquantized BF16 non-expert weights, and a bundled 279 MB DFlash speculative drafter).

`gr8` serves Inkling-2-Mini (`Inkling2LoopForConditionalGeneration`) through `bin/g6m4_ink2`, built with `MODEL=ink2 GR8_WIDTHS=1` (`ARCH=121a` for GB10 / DGX Spark, `ARCH=120a` for RTX 5090). When `--mtp 7` is passed, `g6m4_ink2` automatically discovers the bundled DFlash drafter in the model directory with zero extra flags.

## Results (1× DGX Spark, `TP=1`)

Measured on 2026-11-25 on one NVIDIA DGX Spark (GB10, 128 GB) with local NVMe storage, `gr8` commit `6b874803` ([PR #615](https://github.com/kindlingai/gr8/pull/615)), bundled DFlash (`--mtp 7`), `GR8_INK2_PF_MX8_MOE=2`, `GR8_INK2_MX8_DENSE=2`, `GR8_INK2_MX8_LM_HEAD=0` (and `MX8_DENSE=1`, `MX8_LM_HEAD=1`, `GR8_S2_ADAPT=1` for the `k2` speed preset).

### Single-stream 8-domain decode, C=6 concurrency & 8K cold prefill

| Preset (`--variant`) | Device image | Code | Math | JSON | HTML | Prose | Counting | `C=6` mixed | 8K prefill | 3-user 1K TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`k4`** *(default)* | `18.72 GB` | `64.6` | `107.3` | `90.9` | `77.4` | `48.6` | `162.8` | `139.1` | `3,457` | `0.935 s` |
| **`bit325`** | `16.43 GB` | `90.6` | `105.2` | `91.2` | `85.3` | `49.5` | `172.4` | `160.0` | **`3,484`** | `0.935 s` |
| **`bit250`** | `14.13 GB` | **`111.3`** | `133.4` | `108.3` | `100.0` | `55.3` | `186.3` | `167.6` | `3,480` | `0.942 s` |
| **`k2`** *(speed)* | `12.60 GB` | `100.5` | **`167.5`** | **`148.9`** | **`144.1`** | **`80.8`** | **`228.3`** | **`230.0`** | `3,475` | **`0.827 s`** |

All speeds are tokens/second (`C=6` homogeneous `6× Math` reaches `298.6 tok/s` on `bit325` and `479.5 tok/s` on `k2`).

### Cold prefill sweep (`bit325`, `CAP=139264`, `PREFILL=4096`)

| Prompt length | 4K | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Prefill speed (tok/s) | `3,436` (`1.19 s`) | `3,472` (`2.36 s`) | `3,150` (`10.40 s`) | `2,732` (`23.99 s`) | `2,162` (`60.63 s`) |

### Quality & verification

- **40×2K (`81,920`-token) full-vocabulary `float64` KL vs BF16 (`PPL = 5.3840`):**
  - `k4` (`18.72 GB`): **`0.010460`** mean / `0.001632` median (`96.91%` top-1, **`-27.4%` KL** vs Official NVFP4 `0.014407`).
  - `bit325` (`16.43 GB`): `0.015019` mean / `0.003027` median (`96.00%` top-1).
  - `bit250` (`14.13 GB`): `0.030018` mean / `0.007401` median (`93.93%` top-1).
  - `k2` (`12.60 GB`, E2E logit-KL scales): `0.052425` mean / `0.015119` median (`91.70%` top-1, **`-11.65%`** vs stock `k2`).
- **1,362-task controlled evaluation** (HumanEval+ 164, GPQA-Diamond 198, MATH-500 500, GSM8K-500 500, `temp=0.0`, `max_tokens=6144`, `reasoning_effort="low"`):
  - `exlr8 k4`: **`93.54%`** (`1274/1362`) — statistical tie with Official NVFP4+MXFP8 W4A4 (`93.83%`, exact McNemar `p=0.6655`, exact tie on MATH-500 `98.00%`).
  - `exlr8 bit325`: **`92.80%`** (`1264/1362`) — `91.46%` HumanEval+, `97.80%` MATH-500.
- **DFlash exact greedy parity (`bit325`, `--mtp 7` vs `--mtp 0`):** **8/8 exact token-for-token** across all 8 domains (`1.41×–5.18×` decode speedup).
- **1M KV pool on 1× DGX Spark (`k4`, Docker Compose):** `CAP=262144, SESSIONS=4` boots in `10.3 s`, passes `36.6K` (`3,068 tok/s`) and `146.4K` (`2,089 tok/s`) needle recall with `12.2 GiB` free OS memory; `131072 × 8` (`11.6 GiB` free) and full `1,048,576 × 1` (`10.5 s` boot, `7.8 GiB` free) also verified live.

## Build and run

For Docker Compose on 1× DGX Spark with the published `gr8` image (no compilation needed), use the recipe [inkling-2-mini-exlr8-1x-spark](https://github.com/kindlingai/recipes/tree/main/inkling-2-mini/dgx-spark/1x/exlr8-gr8):

```bash
npx @kindlingai/recipe inkling-2-mini-exlr8-1x-spark && cd inkling-2-mini-exlr8-1x-spark
hf download kindlingai/inkling-2-mini-exlr8-k2-k3-k4 \
  --local-dir ./models/inkling-2-mini-exlr8-k2-k3-k4
docker compose up -d
```

To build `g6m4_ink2` from source:

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
| `GR8_INK2_PF_MX8_MOE` | `2` | Blackwell cached MXFP8 MoE prefill: `2` = all 24 physical layers (`~23.5 GiB` extra memory, ideal for 128 GB Spark), `1` = looped core only, `0` = off (required on 32 GB GPUs to maximize VRAM for weights + KV). |
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
