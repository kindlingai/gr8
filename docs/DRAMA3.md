# Drama 3 (`g6tts`)

One DGX Spark (GB10) box (`TP=1`). Native CUDA text-to-speech engine for `fishaudio/drama-3` (`FishOmniModel`, Built with Fish Audio), with zero PyTorch at serve time.

`gr8` serves Drama 3 through `bin/g6tts` (built with `MODEL=tts`, `engine/tts/build_tts.sh`) connected directly over a Unix domain socket (`run/engine.sock`, PROTOCOL v1) to `bin/g6front-rs`, exposing `/v1/audio/speech`, `/v1/audio/voices`, `/v1/models`, and `/health` on port `8100`.

## Architecture

- **Backbone**: 40 layers, hidden 2048 (`l % 4 == 3` full gated GQA attention with 16 Q / 2 KV heads × 256 dim and NeoX partial RoPE on 64 dims; remaining 30 layers Gated DeltaNet with 16 K / 32 V heads × 128 and depthwise conv k=4), every layer has a 256-expert top-8 MoE (intermediate 512) + gated shared expert.
- **Codebook-MTP tower**: 4 shared layers + 9 residual layers + 9 residual heads emitting 10 codebooks per frame (1 semantic code in `[0, 4096)` + 9 residual codes in `[0, 1024)`).
- **Audio codec**: native CUDA `modded_dac` (`exlr8-codec-v1`, 44.1 kHz mono, 2048 samples/frame = 21.533 Hz, zero lookahead) supporting both streaming decode (per-slot conv arenas, transposed-conv carry tails, 8-layer window-128 KV rings) and reference-voice encode (`encode_audio`).

## Variants

Release repository: `https://huggingface.co/kindlingai/drama-3-exlr8` (EXLR8 trellis routed experts at K2/K3/K4, K4 shared expert, EXLR8 dense/tower projections, FP8-row vocab and heads, and `codec/exlr8-codec.safetensors`).

| Variant | Manifest | Expert Width | Notes |
|---|---|---:|---|
| `bit200` | `exlr8-manifest-bit200.json` | 2.0 bpw (K2) | Smallest footprint / fastest expert bandwidth |
| `bit300` | `exlr8-manifest-bit300.json` | 3.0 bpw (K3) | Default balanced variant (updated after Gate Q) |
| `bit400` | `exlr8-manifest-bit400.json` | 4.0 bpw (K4) | Highest expert fidelity |

## Launch & Smoke Test

```bash
./gr8 tts up --tp 1 --peers 127.0.0.1 \
  --model https://huggingface.co/kindlingai/drama-3-exlr8 \
  --variant bit300 --model-cache /var/lib/gr8/cache

./gr8 tts smoke
```

Default launcher settings (`scripts/engine_args.sh`): `TP=1`, `CAP=4096`, `SESS=8`, `MAX_MODEL_LEN=CAP` (up to `8192`), `PORT=8100`, `READY_TMO=600`, served model name `drama-3`.

## Knobs (`GR8_TTS_*`)

Defined in `knobs/tts.yaml` and documented in [docs/ENV.md](ENV.md):

| Variable | Default | Description |
|---|---|---|
| `GR8_TTS_STREAMS` | `8` | Max concurrent streaming sessions when `--streams` is unset (`1..64`) |
| `GR8_TTS_GRAPH` | `1` | `1` = fused per-frame CUDA graph + codec decode graphs; `0` = eager launches |
| `GR8_TTS_VOC_STRIDES` | `1,1,2,4` | Per-stream codec decode stride schedule in frames (last entry repeats) |
| `GR8_TTS_CODEC_FP32` | `1` | `1` (default) = strict IEEE FP32 codec decoder GEMMs (streaming within 1 LSB of whole-sequence); `0` = FP16 GEMMs + FP32 norms/Snake |

## Hill-Climb Results

Measured on one DGX Spark (GB10) with `bench/tts_bench.py` (`--concurrency 1,2,4,8 --runs 4 --max-new-tokens 500 --temperature 1.0 --top-p 0.98 --top-k 30 --voice default`). Rows are appended as Wave 3.6 optimization passes land.

| Pass | Description | Exactness / Tolerance Check | TTFA@1 p50 (ms) | RTF@1 | audio-s/s@4 | audio-s/s@8 | Kept |
|---:|---|---|---:|---:|---:|---:|---|
| Baseline | Upstream SGLang-Omni (`drama-3`) | — | 112.0 | 0.51 | 4.2 | — | ref |

## License & Attribution

Built with Fish Audio. The `gr8` engine code is MIT licensed; Drama 3 model weights and quantized EXLR8 derivatives are subject to the Fish Audio Community License v1.0 (see `LICENSE` and `Notice` in the weights release).
