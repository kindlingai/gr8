# Qwen3.8-Flash-Next

One to four DGX Spark (GB10) boxes (`TP=1` to `TP=4`). Weights: the exlr8 v2 release (`exlr8-e4m3-pmcgx` routed experts at `K2`/`K3`/`K4`, BF16 dense/vocab/MTP tensors, and the Layer-1 PLE NVMe rowstore).

`gr8` serves Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) through `bin/g6m4_qw38` (and the single-GPU multi-rank simulator `bin/m4sim_qw38`), built with `MODEL=qw38` (`-DG6_MODEL_QW38=1`, `engine/model_geom_qw38.h`). It supports the same weight paging (`GR8_PTAB=0|1|2`), minimal-move layout planning (`bin/g6layout`), and live width scaling (`./gr8 qw38 scale`, `GR8_KV_REWIDTH`, `kvman-peer`) as `glm53` and `glm53f`.

## Results

### 1× DGX Spark (`TP=1`) with auto-loaded DFlash + full native 256K context (2026-11-25)

Measured on 2026-11-25 on one GB10 (DGX Spark, 128 GB) with local NVMe release storage using the live S2 serving path (`bench/bench.py`, `CAP=262,208`, `MAX_MODEL_LEN=262,144`, auto-loaded **DFlash** + Layer-48 `MTP=3`, `REUSE=1`, `KV_STORE=ssd:/var/lib/gr8/cache/kv,size=32G,keep=all` `O_DIRECT`, `GR8_QW38_PLE_CACHE_MIB=1024`):

| Configuration | Manifest | Context × Slots (KV Pool) | Code | Prose | Structured | Counting | `c1` | `c2` | `c4` | Max Stream | Min Free RAM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 box (`TP=1`, Hybrid DFlash + `MTP=3`)** | `bit300` | `262K × 6` (`1.57M`) | **114.2** | **58.5** | **140.8** | **147.7** | **78.1** | **98.6** | **120.5** | **138.4** (`c6`) | `5.59 GiB` |
| **1 box (`TP=1`, Hybrid DFlash + `MTP=3`)** | `bit400` | `262K × 4` (`1.05M`) | **108.3** | **56.6** | **130.8** | **134.7** | **78.4** | **93.2** | **105.7** | **105.7** (`c4`) | `4.55 GiB` |
| 1 box (`TP=1`, DFlash-only `NO_MTP=1`) | `bit400` | `262K × 5` (`1.31M`) | **109.8** | **57.0** | — | — | **78.8** | 54.6 | 77.1 | 84.7 (`c5`) | `8.01 GiB` |

| Configuration | Manifest | 8K prefill (cold / cached) | 32K prefill (cold / cached) | 128K prefill (cold / cached) | Checks |
|---|---|---:|---:|---:|---|
| **1 box (`TP=1`, `6×262K`)** | `bit300` | **2,204** (`3.73 s` / **`0.27 s`**) | **2,315** (`14.20 s` / **`0.32 s`**) | **2,217** (`59.08 s` / **`0.56 s`**) | `8/8` smoke, **`8/8` exact greedy parity**, `3/3` count-200, **128K & 246K needle PASS** (`246,129` tok, `2,094 tok/s`), **RealMark RIPAL `1,780.7 A`** (`99.4%` cache hit across `11.74M` tok, `enable_thinking=false`) |
| **1 box (`TP=1`, `4×262K`)** | `bit400` | **2,160** (`3.80 s` / **`0.26 s`**) | **2,285** (`14.33 s` / **`0.32 s`**) | **2,170** (`60.31 s` / **`0.61 s`**) | `8/8` smoke, `3/3` count-200, **128K & 246K needle PASS** (`245,478` tok, `2,050 tok/s`) |

### Baseline `MTP=3` (pre-DFlash, 1× and 2× Spark, 2026-11-23)

Measured on 2026-11-23 on GB10 (DGX Spark) boxes with local NVMe release storage using the live S2 serving path (`bench/bench.py`, `SESS=8`, `MTP=3`, `GR8_QW38_PLE_CACHE_MIB=4096`, `bit300` / K3 and `bit400` / K4 manifests, and dual-rail RoCE `GR8_QW38_FAST_AR=1` on `TP=2`).

### Live S2 decode & concurrency (`SESS=8`, `MTP=3`, `T=0.7`)

| Configuration | Manifest | Context (`CAP`) | Code | Prose | Structured | Counting | `c1` | `c2` | `c4` | `c8` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 box (`TP=1`)** | `bit300` | `139,264` | **83.0** | **61.5** | **87.4** | **88.8** | **72.6** | **102.1** | **121.7** | **155.2** |
| **1 box (`TP=1`)** | `bit400` | `139,264` | **78.4** | **55.1** | **81.2** | **82.0** | **65.5** | **93.9** | **106.7** | **140.6** |
| **2 boxes (`TP=2`)** | `bit300` | `262,208` | **126.8** | **89.2** | **135.1** | **135.8** | **104.1** | **151.2** | **183.5** | **254.5** |
| **2 boxes (`TP=2`)** | `bit400` | `262,208` | **122.1** | **86.9** | **126.7** | **128.6** | **101.7** | **145.7** | **169.6** | **234.1** |

All values are tokens/second (`c1`–`c8` are aggregate throughput across mixed code/prose/structured prompts). On `TP=1 bit300`, `MTP=4` reaches `89.5` code / `97.2` structured / `99.1` counting (`57.7` prose, `c1` `73.1` / `c2` `103.3`).

### Cold prefill (`bench/bench.py`, salted prompts) & correctness

| Configuration | Manifest | 8K prefill (TTFT) | 32K prefill (TTFT) | 128K prefill (TTFT) | Checks |
|---|---|---:|---:|---:|---|
| **1 box (`TP=1`)** | `bit300` | **2,258 tok/s** (`3.64 s`) | **2,269 tok/s** (`14.44 s`) | — | `8/8` smoke, `3/3` count-200, **128K needle PASS** (`131,572` tok), `GR8_PTAB=0 == 2` exact |
| **1 box (`TP=1`)** | `bit400` | **2,215 tok/s** (`3.71 s`) | **2,200 tok/s** (`14.93 s`) | — | `8/8` smoke, full `CAP=139264` on 1 box |
| **2 boxes (`TP=2`)** | `bit300` | **3,027 tok/s** (`2.74 s`) | **3,142 tok/s** (`10.46 s`) | **2,973 tok/s** (`44.09 s`) | `8/8` smoke, `3/3` count-200, **128K needle PASS** (`131,069` tok) |
| **2 boxes (`TP=2`)** | `bit400` | **2,997 tok/s** (`2.74 s`) | **3,069 tok/s** (`10.71 s`) | **2,915 tok/s** (`45.04 s`) | `8/8` smoke |

## Build and run

```bash
./gr8 qw38 build
./gr8 qw38 up --model /path/to/qwen38-flash-next-exlr8 --tp 2 --peers HOST0,HOST1 --variant bit300
./gr8 qw38 smoke
```

For single-box (`TP=1`) serving, omit `--peers` (no fabric or LAN required):

```bash
./gr8 qw38 up --model /path/to/qwen38-flash-next-exlr8 --tp 1 --variant bit300
```

Live dynamic scaling (`TP=2 ↔ TP=1`, or across `TP=1..4`) uses the same supervisor workflow as `glm53` and `glm53f`:

```bash
./gr8 qw38 scale --tp 1 --peers HOST0
./gr8 qw38 scale --tp 2 --peers HOST0,HOST1
```

The front end runs with `--dialect qwen`, serves the model as `qwen3.8-flash-next` (alias `qw38`), handles Qwen3.8 XML tool calls (`<tool_call><function=...>`) and `<think>...</think>` reasoning (`reasoning_effort` / `enable_thinking`), and supports OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`.

## The model

- **48 text layers** (+ **Layer-48 MTP** 1-layer drafter):
  - **36 Gated DeltaNet (GDN)** linear-attention layers (layers `0,1,2, 4,5,6, ..., 44,45,46`), `16` K-heads / `32` V-heads × `128` dim, depthwise conv1d (`kernel=4`).
  - **12 Full GQA + QSA** layers (layers `3, 7, 11, ..., 47`), `24` Q-heads / `2` KV-heads × `256` dim (`GQA-12`), partial RoPE (`64` dims), block-4 QSA sparse indexer (`top-512` blocks = `2,048` tokens + tail, with automatic short-context bypass below `2,051` tokens).
  - **4-stream low-rank hyper-connections** (`Qwen4ExpTextGatedResidual`, `4 × 2560`, rank `320`) around every attention and MoE block plus final mixer.
  - **Layer-1 PLE (Per-Layer Embeddings)**: `320,001,536 × 160` BF16 NVMe rowstore (`~95.4 GiB`, 2/3-gram prime-hashed 16-head lookup summing to `2,560` dim), gated against the 4-stream hyper state + dilated-3 depthwise conv1d, fetched asynchronously behind Layer 0 over an 8-thread `O_DIRECT` `pread` pool with LRU host cache.
  - **MoE**: `512` routed `exlr8-e4m3-pmcgx` experts (`top-10`, intermediate `640`, `nh=5`) + BF16 gated shared expert (`intermediate 640`).
  - **Untied vocabulary**: `248,320 × 2560` BF16 `embed_tokens` and `lm_head`.

## Runtime knobs and rollback flags

Every `GR8_QW38_*` knob is registered in `knobs/qw38.yaml` (documented in [ENV.md](ENV.md)):

| Knob | Default | Purpose / rollback |
|---|---|---|
| `GR8_QW38_DENSE_FP8` | `1` | On-load GPU block-32 FP8 E4M3 quantization of dense decode weights & `lm_head`. Set `=0` for pure BF16 dense decode. |
| `GR8_QW38_MTP_FAST` | `1` | Fast MTP drafting: 2-bit block-32 draft head screen + exact top-64 rescore, INT4 L048 routed experts (`-1.05 GB` VRAM), and single-graph zero-host-sync K-step greedy argmax drafting. Set `=0` to roll back. |
| `GR8_QW38_TRUNK_HEAD2` | `1` | Trunk decode/verify `lm_head` (`M <= 48`) 2-bit screen + bit-exact FP8 top-256 rescore (reuses the MTP 2-bit table at zero extra VRAM when `MTP > 0`). Set `=0` for full-vocab FP8 `lm_head`. |
| `GR8_QW38_PF_MXFP8` | `1` | Blackwell native MXFP8 (`m16n8k32`, UE8M0 block-32 scales) Tensor Core GEMMs for trunk GDN, Full Attn (`Q/K/V/O`), and shared-MoE at `M >= 16` (`c4`/`c8` verify & prefill), keeping exact FP16 scales on HC, `lm_head`, QSA indexer, and MTP. Set `=0` for FP16-scaled FP8 decode + BF16 cuBLAS prefill. |
| `GR8_QW38_GRAPH` | `1` | Full 48-layer decode + MTP verify & draft CUDA Graph capture on `TP=1` and multi-box RDMA `TP>=2` (compatible with `GR8_PTAB=1\|2`). Set `=0` for eager launches. |
| `GR8_QW38_WIDE` | `1` | Wide-load K2/K3/K4 PMCGX MoE kernels (`tr3_wide.cuh`). Set `=0` for stock-order PMCGX decode. |
| `GR8_QW38_PLE_ASYNC` | `1` | Overlaps Layer-1 PLE NVMe `pread` pool behind Layer 0 compute. Set `=0` for synchronous PLE fetch. |
| `GR8_QW38_PLE_CACHE_MIB` | `1024` | Host LRU cache size (MiB) for PLE rows (`0` disables caching). |
| `GR8_QW38_WIDTHS` | *(unset)* | Optional path to a width manifest JSON when `--widths` is not passed on the command line. |
| `GR8_QW38_FAST_AR` | `1` | Single-step multi-collective device-driven RDMA all-reduce (`ncoll = 2*L+1 = 97` on `TP>=2`). Set `=0` for per-collective fallback. |
| `GR8_QW38_DFL` | `1` | Enables **DFlash** parallel block speculative drafting for single-stream (`c1`) decode when `drafter/dflash.q4_0.pack` is present (or set via `GR8_QW38_DFL_PACK`). Set `=0` to disable and use Layer-48 MTP only. |
| `GR8_QW38_DFL_PACK` | *(auto)* | Optional path to a Q4_0 DFlash drafter pack. Unset auto-discovers `<release>/drafter/dflash.q4_0.pack` (fallback `<release>/dflash.q4_0.pack`) when `GR8_QW38_DFL=1`. Set `0`, `off`, or `none` to disable auto-discovery. |
| `GR8_QW38_DFL_K` | `7` | Maximum draft prefix length verified per DFlash round (`1..7`). |
| `GR8_QW38_DFL_DP` | `2` | DFlash path selector (`0` = greedy, `1` = calibrated Viterbi DP, `2` = Viterbi DP + per-round dynamic `K*`). |
| `GR8_QW38_NO_MTP` | `0` | Set `=1` (with `MTP=0`) to omit loading Layer-48 MTP weights and KV pages (~5.2 GB VRAM saved for extra 262K session capacity in DFlash-only mode). |
| `GR8_QW38_VISION` | `0` | Opt-in multimodal image input (`--serve` only). Loads `dense/vision-bf16.safetensors` (27-layer BF16 ViT, ~898 MB per rank) with interleaved 3D MRoPE (`mrope_section=[11,11,10]`). Set `=0` (default) for text-only serving. |
| `GR8_PTAB` | `0` | Fixed-address device piece table (`0`=off, `1`=paged, `2`=paged+slab pool) for minimal-move live scaling. |
| `MTP` (`--mtp`) | `0` | Layer-48 MTP speculative draft depth (`0..5`). |

## Image input (vision)

Pass `GR8_QW38_VISION=1` at launch when the release directory includes `dense/vision-bf16.safetensors` (`v0.2` and later):

```bash
GR8_QW38_VISION=1 ./gr8 qw38 up --model /path/to/qwen38-flash-next-exlr8 --tp 2 --peers HOST0,HOST1 --variant bit300
GR8_QW38_VISION_SMOKE=1 ./gr8 qw38 smoke
```

- **Vision tower**: 27-layer BF16 ViT (`1152` hidden, `16 × 72` heads, axial 2D RoPE `theta=10000`, `48 × 48` bilinear `align_corners=True` position embeddings, `16 × 16` patches, `2 × 2` spatial merger to `2560`, `64..16384` tokens per image). Every TP rank runs the tower locally (~898 MB weights).
- **Text integration**: Interleaved 3D MRoPE (`mrope_section=[11, 11, 10]` across the 32 partial-RoPE pairs, including QSA block-4 keys) during prefill, and isotropic post-image `mrope_delta` shift during decode and MTP so CUDA-graphed decode and MTP stay active and bit-exact after image turns. Text-only requests are unaffected.
