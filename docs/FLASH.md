# GLM-5.3-Flash

gr8 serves [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) from its exlr8 release
([kindlingai/glm-5.3-flash-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4)) at TP=4,
TP=3 (a three-box ring, variant k4) or TP=2. The engine is `bin/g6m4_flash`, a build of the same `g6m4` source with `-DG6_MODEL_FLASH=1`
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

## Context

`CAP` defaults to 262208: a request of 262,144 tokens (prompt + `max_tokens`) plus a page for the guard rows the
scheduler keeps past it (2 at MTP=1), so the front end's `max_model_len` is 262,206 (`pool_rows - guard`).

Only the attention layers keep KV: the 11 DSA layers and the MTP layer, 12 bindings a rank. KDA's state is the same size
at any context. A token costs each of them a 1,024 B bf16 latent row and 640 B of indexer keys (`engine/attn/kvfmt.h`;
`G6_FLASH_KV_FP8=1` makes the row 528 B), 19,968 B in all. Flash has no CP, so every rank holds the whole context and
the cost is the same at TP=4 and TP=2:

| CAP | pages of 64 | KV a rank |
|---|---|---|
| 33,280 (the old default) | 520 | 0.66 GB |
| 262,208 | 4,097 | 5.24 GB |
| 1,048,640 | 16,385 | 20.9 GB |

Decode and prefill scratch grow with a quarter of `CAP` and stay small by comparison. The model's config allows
1,048,576 positions, and `CAP` may go higher where memory allows. Each layer's bind checks that `G6_MEM_MIN` (8 GiB)
stays free, so a `CAP` that leaves too little stops the boot; the boot log's `MemAvailable` line shows what is left.

Nothing else in the engine limits the context to less than `CAP`:

- Flash has no RoPE (`g6_attn_rope_init` does nothing), so a position only picks a KV row.
- The indexer's top-2048 selection takes the general path past 8,192 visible pools of 4 tokens (`engine/attn/kpool.cuh`),
  with 64-bit offsets.
- Prefill runs in chunks of 4,096 rows a lane, two lanes, at any context. Decode captures nine buckets, 2048 to 262144
  by doubling, and `CAP`.

With one session (`SESS=1`, the default) `SHARE_LIMIT` (32768) shares nothing: it only sets the `max_tokens` a chat
request without one gets, `SHARE_LIMIT - 64 - prompt` but at least 8192, capped by the room left. With several (below)
it also decides which requests share decode steps, as on GLM-5.3: a longer one waits for the others and runs alone.

Prefill time grows faster than the prompt because the indexer scores every earlier key for each row. The time of a
262K prefill has not been measured. The shim errors a request after 900 s with no output (`--stall-timeout`), and a
prefill sends none, so a prefill longer than that fails.

## Several sessions

`--sess N` (`SESS`, 2 to 8; the default stays 1) serves up to N requests at once on S2, as GLM-5.3 does: their decode
rounds share main and draft steps, and their KV runs share the `CAP` pool (`--s2-sessions N` to `g6m4_flash`).

- KDA state: session q owns slots `q x (MTP + 1)` up to `(q + 1) x (MTP + 1)` (`FlashSlots` in
  `engine/runtime/s2/flash_ring.h`), reset when a request starts on q. In a main step each session's first row loads its
  own committed slot (`StepIn::krun`, `Args::src` per row in `engine/kda`), and its verify rows write its own ring, so
  sessions sharing a step never read each other's state. The engine allocates `SESS x (MTP + 1)` slots.
- Prefill runs a request alone on its own slots and KV run.
- Every main and draft width a round can take is captured as a graph per context bucket at boot
  (`flash_shapes`, `engine/runtime/s2/flash_native.cuh`); a step outside the set is an error, not a capture.
- A round of every session, `SESS x (MTP + 1)` rows, must fit a step's 32 (`G6_MX`); the launcher and the engine refuse
  more.
- With `SESS=1` the engine runs the one-session path as before.

The memory these take (slots and graphs), the speed of shared steps and the output of sessions sharing steps have not
been measured on Flash yet.

## Not yet

- CP, prefix reuse and DFlash2: the engine refuses them under `--serve`.
- MTP drafts beyond the first: the KDA state does not roll back a rejected draft yet, so the launcher runs MTP=1.
- The int4 KV build.
- The vision tower: the engine serves text, and the fetch skips `dense/vision-*`.
- A 262K prefill's time and memory headroom on a GB10: the `CAP` above is computed, not measured.
