# Qwen3.8-Flash-Next

One to four DGX Spark (GB10) boxes (`TP=1` to `TP=4`). Weights: the exlr8 v2 release (`exlr8-e4m3-pmcgx` routed experts at `K2`/`K3`/`K4`, BF16 dense/vocab/MTP tensors, and the Layer-1 PLE NVMe rowstore).

`gr8` serves Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) through `bin/g6m4_qw38` (and the single-GPU multi-rank simulator `bin/m4sim_qw38`), built with `MODEL=qw38` (`-DG6_MODEL_QW38=1`, `engine/model_geom_qw38.h`). It supports the same weight paging (`GR8_PTAB=0|1|2`), minimal-move layout planning (`bin/g6layout`), and live width scaling (`./gr8 qw38 scale`, `GR8_KV_REWIDTH`, `kvman-peer`) as `glm53` and `glm53f`.

## Results

Measured on 2026-11-22 on GB10 (DGX Spark) boxes with local NVMe release storage, `bit300` (`K3`) and `bit400` (`K4`) manifests, `GR8_QW38_DENSE_FP8=1`, `GR8_QW38_PLE_ASYNC=1`, `GR8_QW38_GRAPH=1`, and dual-rail RoCE `GR8_QW38_FAST_AR=1` on `TP=2`.

### Single-stream & 5-stream decode (English prose, `P=30`, `gen=96` for `c1`, `gen=64` for `c5`)

| Configuration | Manifest | `MTP=0` `c1` | `MTP=1` `c1` | `MTP=2` `c1` | `MTP=3` `c1` | `MTP=0` `c5` agg | `MTP=1` `c5` agg | `MTP=3` `c5` agg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 box (`TP=1`)** | `bit300` | 32.69 tok/s | 45.23 tok/s | 52.68 tok/s | **56.05 tok/s** | 135.62 tok/s | 145.68 tok/s | **165.87 tok/s** |
| **1 box (`TP=1`, `GR8_PTAB=2`)** | `bit300` | — | — | — | **55.15 tok/s** | — | — | — |
| **1 box (`TP=1`)** | `bit400` | — | — | — | **54.28 tok/s** | — | — | — |
| **2 boxes (`TP=2`)** | `bit300` | 48.55 tok/s | 68.79 tok/s | 80.17 tok/s | **81.90 tok/s** | 200.00 tok/s | 212.88 tok/s | **228.18 tok/s** |
| **2 boxes (`TP=2`)** | `bit400` | — | — | — | **78.20 tok/s** | — | — | **179.14 tok/s** |

- `MTP=3` achieves `71%–85%` draft acceptance (`3.13–3.50 tok/round`) on technical prose, with **100% byte-identical greedy token parity** across `MTP=0, 1, 2, 3` and `GR8_PTAB=0` vs `GR8_PTAB=2` on both `TP=1` and 2-box `TP=2`.

### Long-context prefill & solo decode (`P=128` and `P=2048`, `MTP=0`, `bit300`)

| Configuration | `P=128` solo decode | `P=2048` prefill | `P=2048` solo decode |
|---|---:|---:|---:|
| **1 box (`TP=1`)** | 34.41 tok/s (`bit400`: 32.88) | 1,018.4 tok/s | 32.12 tok/s |
| **2 boxes (`TP=2`)** | 51.52 tok/s | 1,176.6 tok/s | 46.82 tok/s |

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
  - **Layer-1 PLE (Per-Layer Embeddings)**: `320,001,536 × 2560` BF16 NVMe rowstore (`~95.4 GiB`, 2/3-gram prime-hashed 4-head lookup), gated against the 4-stream hyper state + dilated-3 depthwise conv1d, fetched asynchronously behind Layer 0 over an 8-thread `O_DIRECT` `pread` pool with LRU host cache.
  - **MoE**: `512` routed `exlr8-e4m3-pmcgx` experts (`top-10`, intermediate `640`, `nh=5`) + BF16 gated shared expert (`intermediate 640`).
  - **Untied vocabulary**: `248,320 × 2560` BF16 `embed_tokens` and `lm_head`.

## Runtime knobs and rollback flags

Every `GR8_QW38_*` knob is registered in `knobs/qw38.yaml` (documented in [ENV.md](ENV.md)):

| Knob | Default | Purpose / rollback |
|---|---|---|
| `GR8_QW38_DENSE_FP8` | `1` | On-load GPU block-32 FP8 E4M3 quantization of dense decode weights & `lm_head` (`M <= 12`). Set `=0` for pure BF16 dense decode. |
| `GR8_QW38_GRAPH` | `1` | Full 48-layer decode + MTP verify & draft CUDA Graph capture on `TP=1` and multi-box RDMA `TP>=2` (compatible with `GR8_PTAB=1\|2`). Set `=0` for eager launches. |
| `GR8_QW38_WIDE` | `1` | Wide-load K2/K3/K4 PMCGX MoE kernels (`tr3_wide.cuh`). Set `=0` for stock-order PMCGX decode. |
| `GR8_QW38_PLE_ASYNC` | `1` | Overlaps Layer-1 PLE NVMe `pread` pool behind Layer 0 compute. Set `=0` for synchronous PLE fetch. |
| `GR8_QW38_PLE_DIRECT` | `1` | `O_DIRECT` aligned reads on the PLE rowstore (`=0` uses OS page cache). |
| `GR8_QW38_PLE_CACHE_MIB` | `4096` | Host LRU cache size (MiB) for PLE rows. |
| `GR8_QW38_FAST_AR` | `1` | Single-step multi-collective device-driven RDMA all-reduce (`ncoll = 2*L+1 = 97` on `TP>=2`). Set `=0` for per-collective fallback. |
| `GR8_PTAB` | `0` | Fixed-address device piece table (`0`=off, `1`=paged, `2`=paged+slab pool) for minimal-move live scaling. |
| `MTP` (`--mtp`) | `0` | Layer-48 MTP speculative draft depth (`0..3`). |
