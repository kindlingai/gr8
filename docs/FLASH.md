# GLM-5.3-Flash

gr8 serves [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) from its exlr8 release
([kindlingai/glm-5.3-flash-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4)) at TP=4 or
TP=2. The engine is `bin/g6m4_flash`, a build of the same `g6m4` source with `-DG6_MODEL_FLASH=1`
(`engine/model_geom.h` holds the geometry).

## Build and run

    ./gr8 glm53f build
    ./gr8 glm53f up --model https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4 --model-cache DIR --tp 4 ...
    ./gr8 glm53f smoke

`./gr8 glm53f` takes the options of `./gr8 glm53` (KNOBS.md). Each box fetches its rank's part of the release and loads
it through the same release planner as GLM-5.3 (`engine/loader-rs`): `release::plan` reads the model from the release's
own metadata (`hidden_size` 4096) and maps Flash's tensors with `engine/loader-rs/src/flash.rs`. The width manifests are
`k2`, `k3` and `k4` (the default).

## The model

- 45 layers plus the MTP layer 45. 34 layers use KDA linear attention (`engine/kda`). The others (3, 7, ..., 43) use
  DSA sparse MLA with an indexer (`engine/attn/flash_attn.cu`).
- Four mHC residual streams (`engine/mhc`).
- 288 routed experts, top-8, from layer 3 (the exlr8 trellis experts, as GLM-5.3).

## What works

- `./gr8 glm53f up` boots in about 16 s from a local cache and passes `./gr8 glm53f smoke`.
- TP=4 prefill of an 8K prompt: about 3.55 s at 2200 MHz.
- One session at a time, with MTP=1 (`flash_defaults` in `scripts/engine_args.sh`).

## Not yet

- Several sessions, CP, prefix reuse and DFlash2: the engine refuses them under `--serve`.
- MTP drafts beyond the first: the KDA state does not roll back a rejected draft yet, so the launcher runs MTP=1.
- The int4 KV build.
- The vision tower: the engine serves text, and the fetch skips `dense/vision-*`.
- `CAP` defaults to a 32K context until the pool is measured on Flash.
