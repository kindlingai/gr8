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

With one session `SHARE_LIMIT` (32768) shares nothing: it only sets the `max_tokens` a chat request without one gets,
`SHARE_LIMIT - 64 - prompt` but at least 8192, capped by the room left.

Prefill time grows faster than the prompt because the indexer scores every earlier key for each row. The time of a
262K prefill has not been measured. The shim errors a request after 900 s with no output (`--stall-timeout`), and a
prefill sends none, so a prefill longer than that fails.

## Session checkpoints (`G6_FLASH_CACHE=aligned`, experimental)

Off by default. With `FLASH_CACHE=aligned` (`G6_FLASH_CACHE=aligned`), a prompt's prefill keeps a checkpoint at its last
cold prefill grid point, and a later prompt that starts with the same tokens is prefilled from that point, not from 0.
This is the Flash form of prefix reuse for an agent's next turn. The code is `engine/runtime/s2/flash_checkpoint.h` (the
index, the aligned point, the manifest) and `flash_checkpoint.cuh` (the store), hooked into `s2/flash_native.cuh`.

- The grid is the prefill quantum: `--prefill` rows times the lanes, 4,096 x 2 = 8,192 rows. A prompt of N tokens keeps
  a checkpoint at `p = floor((N - 1) / 8192) * 8192`, so prompts of 8,192 tokens or fewer keep none.
- A checkpoint holds rows `[0, p)` of the 12 attention bindings (the latent rows and the indexer keys), KDA slot 0's
  state and conv history, and the prompt's tokens `[0, p]`. A later prompt hits it when it starts with those `p + 1`
  tokens under the same cache namespace (the steering profile); the longest entry wins, then the newest.
- A hit restores the checkpoint at bind, after the cold reset, and prefills from `p`.

Why the output is bit-exact against no cache: a cold prefill runs its quanta as `[kG, (k+1)G)`, and the KDA prefill is
chunked by quantum, so a warm prefill from `p = kG` runs the same quanta on the same KDA state and the same pages. The
MTP layer's row `p - 1` reads token `p`, which is why the match runs through token `p`. The indexer's pools are
recomputed from the raw rows (`engine/attn/kpool.cuh`), so no other state is kept. Rows past `p` and the other KDA slots
are as a cold bind leaves them.

Why a checkpoint is not taken at the end of the prompt: decode advances the KDA state one row at a time
(`engine/kda/kda.cuh`), and the prefill runs it in chunks of 64 rows (`engine/kda/kda_prefill.cuh`). The two orders of
fp32 arithmetic give different state, so a state taken after decode is not the state a cold prefill of the same tokens
gives. A terminal checkpoint is not cold-exact by construction.

`FLASH_CACHE=terminal` (`G6_FLASH_CACHE=terminal`, experimental, **NOT cold-exact**) keeps aligned's checkpoints and
one more at the end of every request: `p` = the rows the request consumed (prompt and output; its last output token is
still pending and becomes token `p`), with the KDA state the verify ring holds for `p` and the pages through `p`, the
last one partial. A later prompt that starts with those `p + 1` tokens (an agent's next turn sent as raw tokens) is
prefilled from `p`. Its output differs from a cold prefill's: the KDA state comes from decode, and the prefill from `p`
runs quanta a cold prefill does not. How much is measured (token diff against no cache on the same trace, NLL delta,
needles), not argued. After a terminal hit the request keeps no aligned checkpoint (its rows are not cold rows), so
aligned entries stay exact. The MTP layer's row `p - 1` may not hold the last token's draft row; that changes only
which drafts are accepted. The boot and request log lines say `terminal`. Aligned stays the mode to use when the output
must match no cache.

The store: the one `KV_STORE` (`--kv-store`; `mem:`, `ssd:PATH` or `remote:HOST:PORT`, `persist=on` refused), else
this process's memory, `FLASH_STATE_BUDGET_MIB` (8,192) a rank. An SSD folder starts empty at every boot. On a KV node
the lease is `g6m4f-ckpt-<host>-<epoch>-r<rank>`, never the pool stash's. Each checkpoint is one contiguous range of
blocks with a manifest and GPU checksums of every extent; a restore that does not match its checksums stops the engine.
A full store drops the least recently used checkpoints. Every rank keeps the same index, and its digest joins the pool
digest the ranks compare.

`group=` on the store (`remote:HOST:PORT,size=N,group=G[,cap=,weight=]`, experimental, D-331 M8 (c)): the lease is a
member of group `G` on the KV node, the way a full model's `--kv-store group=` tier is, so the two models share the
node's memory. It starts at `size=`; the head reports to the node every `G6_S2_SHARE_MS` and asks for segments when
captures drop entries, an entry does not fit at all, or the next capture would drop one, and offers what giving would
drop (the least recently used checkpoints' ages) only while idle (`KV_SHARE_WHEN=idle`, default; `always`). A grant
or give is a Resize event every rank applies at the same commit: a grow adds blocks; a shrink drops the least recently
used checkpoints until the rest fit, copies the ones above the new end down, then gives the top segments back. The log
prints `# rank R: flash ckpt share: now S segments, B blocks (N entries, D dropped) in T ms`. Not measured on a box.

The boot log prints `# rank R: flash ckpt aligned: ...`, and each request prints `flash ckpt request ... kept ...` or
`... restored ...` with the time.

Measured on one RTX 5090 (`engine/tests/test_flash_ckpt.cu`, Flash's TP=2 geometry, 100 pages, 161 blocks): a capture
(161 blocks) takes 26-50 ms and a restore 21-25 ms through memory, an SSD file and a loopback KV node. Not measured on GB10, and the
end-to-end bit-exact check against no cache on a cluster has not run yet.

Limits:

- Checkpoints are not entries of the pool's block stash (`engine/kvman-rs` `stash.rs`); a `group=` store shares and
  resizes through its own index (above), with the same report rule.
- Each capture writes the whole prefix `[0, p)`, not only the rows past the entry it started from.
- `terminal` is not cold-exact (above).

## Not yet

- Several sessions, CP, prefix reuse and DFlash2: the engine refuses them under `--serve`. `G6_FLASH_CACHE=aligned`
  (above) is the experimental Flash form of prefix reuse.
- MTP drafts beyond the first: the KDA state does not roll back a rejected draft yet, so the launcher runs MTP=1.
- The int4 KV build.
- The vision tower: the engine serves text, and the fetch skips `dense/vision-*`.
- A 262K prefill's time and memory headroom on a GB10: the `CAP` above is computed, not measured.
