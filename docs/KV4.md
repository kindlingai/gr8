# 4-bit MLA KV latent (`--kv-format int4`, default for GLM-5.3 TP=4)

## TP=4 INT4 folded decode default

GLM-5.3 at `TP=4` defaults to `KV_FORMAT=int4`: `bin/g6m4-int4`, `CAP=513536`
(2,390 pages per rank), the bit-exact INT4 fast paths, folded Hadamard weights,
tensor-core decode and `CP_ONECHAIN`. Roll back with `--kv-format fp8` or
`KV_FORMAT=fp8`: `bin/g6m4`, `CAP=485376`, and no implicit `GR8_KV4_*` or
`GR8_CP_ONECHAIN` settings. Explicit settings still take precedence.
GLM-5.3 at `TP=3/5/6` and GLM-5.3-Flash retain FP8 unless INT4 is requested;
DeepSeek and pseudo-engine defaults are unchanged.

The GLM-5.3 TP=4 INT4 default enables folded Hadamard `UK`/`UV`
weights (`GR8_KV4_FOLD=1`), 16-head WMMA tensor-core INT4 decode
(`GR8_KV4_TC=1`), and batched remote CP attention (`GR8_CP_ONECHAIN=1`). The
five bit-exact INT4 kernel settings remain enabled underneath this stack.
`GR8_KV4_FOLD=1` requires a `--v2-dense` release boot; the engine refuses
folded weights for a non-release boot. FP8 `kv_b` skips fold and its implicit
TC default, retaining the exact INT4 path.

Measured with live settings: long-context CP decode geomean improved 18.57%
across 32K/64K/120K; short decode geomean improved 3.69%. This is a numerical
quality trade, not bit-exact against the scalar reference: `mean_kl = 0.015803`,
`qeval` 72/75 versus 71/75, and needles 40/40. `kv4_test` same-pipeline relative
error was below `1.5e-3` and cosine above `0.999998`.

Use `--kv4-tc 0 --kv4-fold 0 --cp-onechain 0` for the measured bit-exact INT4
fast-path rollback. `CP_ONECHAIN=1` is documented bit-exact, but the exact INT4
fast paths plus `ONECHAIN=1` were not separately boot-measured. Single-flag
partial overrides (`--kv4-tc 0` or `--kv4-fold 0` alone), and explicit fold/TC
flags at TP=6/3, preserve single-rotation safety but are unmeasured. TP=6/3 and
FP8 do not acquire these defaults. `GR8_ATTN_FLIP=1` in `EXTRA_ENV` suppresses
the implicit `CP_ONECHAIN=1`; explicit settings retain precedence.


## Bit-exact decode fast paths

At TP=4, `--kv-format int4` enables multi-slot warp scores (`GR8_KV4_SC=4`,
`GR8_KV4_SC_TR=2`), lane-transposed query rotation (`GR8_KV4_QT=1`), compacted
double-buffered context partials (`GR8_KV4_CTX=2`), and fused output rotation
(`GR8_KV4_FUSE_ROT=1`). These kernels preserve identical output bits. The measured
decode gains were 6.13% for long-context CP and 1.29% for short decode. Other TP
widths and FP8 keep their existing defaults. Explicit `EXTRA_ENV` settings override
each default without duplicate keys.


The KV cache spends most of its bytes on the MLA latent: 512 dims per token per layer. `--kv-format int4`
(`KV_FORMAT=int4`) stores that latent in 4 bits, so the same memory holds more pages. The rope key stays BF16 and the
indexer keys stay FP8.

**Historical row-format trade at TP=4** (before the fast paths above; measured on the cluster, below): about 1.4x the tokens in the same memory (CAP 415,232, against 294,912 for
fp8 at the same 1,920-page memory; the fp8 default has since grown to 2,280 pages, CAP 387,072 at that share limit), decode about 3% slower, 8K prefill about 6% faster, speculative acceptance about 2% lower, and needles found
12/12 from 32K to 200K tokens (and at 400K with a larger CAP), as with fp8.

The row is fixed when the engine is compiled (`-DG6_KV4=1`, `attn/kvfmt.h`), so there are two engines: `bin/g6m4` (FP8)
and `bin/g6m4-int4` (the GLM-5.3 TP=4 default). The image ships both for every TP; `./gr8 glm53 build` builds both
at TP=4 and before a fresh clone has a TP configured. Explicit `KV_FORMAT=fp8` or another configured TP skips the
INT4 build unless INT4 is requested. `./gr8 glm53f build` adds `g6m4_flash-int4` only with `KV_FORMAT=int4`.
The INT4 default (or explicit `--kv-format int4`) runs `g6m4-int4` with its own CAP default
(Capacity, below) and pass the engine `--kv-format int4`. Each engine refuses the other's format, and `g6m4-int4`
refuses to start without the flag. A build without `G6_KV4` has the same row, the same kernels and the same SASS as
before.

## The row

| | FP8 row | INT4 row (`G6_KV4=1`, GLM-5.3 TP=4 default) |
|---|---|---|
| latent | 512 x E4M3, one FP32 scale per 128 dims (`amax / 448`) | `H x` (see below), 512 x INT4, one FP16 scale per 32 dims (`amax / 7`), codes -7..7 |
| bytes | 512 + 16 | 256 + 32 |
| rope key | 64 x BF16 | 64 x BF16 |
| row | 656 B | 416 B |

- `H` is the 512-point Sylvester Hadamard scaled by 1/sqrt(512). It is orthonormal and its own inverse.
- Byte `i` holds dims `2i` (low nibble) and `2i + 1` (high nibble), each as code `q + 8`.
- `engine/attn/kvfmt.h` holds the one definition. Every kernel, copy, store and dump that knows a row size reads it
  from there.

**Why INT4 with a rotation, not FP4.** The study used the real per-layer latent covariances (calibration Hessians of
all 79 layers) and kv_b. At every row size, INT4 after `H` has less error than NVFP4 or MXFP4:

| format | row | SQNR median |
|---|---|---|
| INT4 g32 + H | 416 B | 20.3 dB |
| NVFP4 | 432 B | 20.4 dB |
| MXFP4 | 400 B | 18.6 dB |
| FP8 today | 656 B | 31.8 dB |

The rotation spreads the outlier channels of layers 2 and 3, and adds 4-6 dB to their worst case. Decode never uses
an MMA on the latent, and prefill turns it into an FP16 table first, so FP4 tensor cores would not help.

**Where the rotation goes.** The cache holds `c' = H c`.
- Scores: `q . c = (H q) . c'`. Decode rotates the absorbed query (`fwht_rows` after absorb).
- Context: `UV c = UV H c'`. Decode rotates the context back before `value_project`. The CP path does both in
  `cp_front` and `cp_finish`.
- Prefill: `expand` (OWN=1/2) and the row-owner prepass (OWN=3/4) rotate each key back as they build their FP16 key
  table, so prefill's Q, its UV and every weight are unchanged.
- No weight is rewritten.

**Kernels.**
- Writes: `append_kv` (`engine/attn/attn.cu`).
- Decode reads: `kv4::scores_heads` and `kv4::context_parts_heads` (`engine/attn/kv4.cuh`, the default head-batched
  path), and the generic `mla_scores` / `context_parts` (`GR8_ATTN_HEADS=0`).
- Prefill reads: `expand` (`engine/attn/prefill.cuh`) and `gqkv4::prepass` (`engine/attn/prefill-qo-bind.cuh`).
- In a KV4 build, `GR8_ATTN_FLIP`, `GR8_ATTN_P11` and the per-head fixed-K kernels are ignored, and `GR8_PF_QO=v1|r1` is
  refused. They read the FP8 row directly.

**Stores.** Page blocks shrink from 3,502,592 B to 2,289,152 B (79 x 64 x 416 + 22 x 64 x 132; 559 x 4 KiB slots).
A store's index key names the row format (`kv int4g32h`) and the engine binary's stamp, so a `persist=on` store that
the other format wrote is cleared at boot (with a log line), never read. A KV node sizes each lease by the engine's
block, and the lease name ends in `-int4`. `kvman-sim --mla-row 416` models the smaller blocks.

## Build and test

```bash
KV_FORMAT=int4 GR8_NR=4 scripts/build.sh bin                        # bin/g6m4-int4: the TP4 engine with the INT4 row
EXTRA=-DG6_KV4=1 GR8_NR=4 scripts/build.sh bin4-kv4                 # the same as a full build (g6m4, m4sim) in its own dir
TEST=kv4_test GR8_NR=4 scripts/c1_tests.sh bin4                     # the FP8 row
TEST=kv4_test GR8_NR=4 EXTRA=-DG6_KV4=1 scripts/c1_tests.sh bin4-kv4 # the INT4 row
```

`kv4_test` checks two things on one GPU with synthetic weights:

- **Round trip.** It appends rows through `g6_attn_append`, decodes them on the host, and checks the latent SQNR, the
  codes and the rope keys.
- **Attention.** It runs decode, prefill and row-owner prefill over the cache, and compares each against a
  double-precision host reference. The kernel error is measured against the cache's own decoded latent, and must be
  below 1e-2. The cost of the row format is measured against the exact latent.

## Capacity

Every rank holds the first `SHARE_LIMIT` tokens and 1/TP of the rest: `CAP = SHARE_LIMIT + (P - SHARE_LIMIT/64) x 64 x TP`.
The int4 default keeps each rank's memory where fp8's default puts it. The model counts everything that grows with the
pool or with CAP:

- the pool: 78 MLA layers and 21 index caches on P pages a rank (the index keys sit on the latent's pages);
- the CP staging pools: 2 x (row + index key) a token;
- the MTP layer's latent and index key;
- five buffers sized by CAP, from the boot logs: the QO tables (1.41 KB a token), the tensor-core prefill states
  (2.95 KB), the shared decode scratch (0.77 KB), the MTP layer's scratch (1.53 KB) and the MTP layer's own tensor-core
  prefill state (2.95 KB).

Those buffers do not shrink with the row, which is why the gain is below 656/416. At TP=4 each added page (256 tokens)
costs 2.3 MB of int4 pool and 2.9 MB of them.

| TP (SHARE_LIMIT) | fp8 | int4 (GLM-5.3 TP=4 default) | gain | |
|---|---|---|---|---|
| TP=4 (65,536) | 1,981 pages a rank, CAP 310,528 | 2,390 pages, CAP 415,232 | 1.34x | measured |
| TP=3 (65,536) | 1,941, CAP 241,600 | 2,480, CAP 345,088 | 1.43x | estimate |
| TP=6 (32,768) | 2,026, CAP 614,400 | 2,440, CAP 773,120 | 1.26x | estimate |
| TP=6 (131,072, the default) | 3,307, CAP 614,400 | 3,940, CAP 857,600 | 1.40x | estimate |

Table figures are at the `SHARE_LIMIT` shown. The fp8 default is now 2,280 pages (CAP 387,072 at `SHARE_LIMIT` 65,536),
so the TP=4 gain is about 1.07x at that limit.

- **TP=4 rests on two boots.** At 1,920 pages the int4 row leaves 2.31 GiB more free than fp8 (the model says 2.34).
  At 2,876 pages it leaves 2.39 GiB less (model 2.23). Interpolating gives equal headroom at about 2,390 pages a rank.
  The model alone says 2,409. The fp8 default then moved from 1,920 to 1,981 pages a rank (the same model's
  equal-memory figure against the replicated index keys before #44), and int4 kept 2,390, so at TP=4 int4 now
  leaves about 0.3 GiB more free than fp8.
- **TP=3 and TP=6 are the model's figures, rounded down by the 0.8% it overshoots at TP=4.** They reuse TP=4's
  per-token buffer sizes, and the QO tables differ with `GR8_PF_OWN`, so they are not measured.
- `scripts/engine_args.sh` (`int4_pages`) holds these page counts. A `CAP` given by hand wins, as for fp8.
- An earlier version of this table (1.42-1.63x) counted only the pool, the staging pools and one prefill table, and
  allowed 2,876 pages a rank at TP=4. That boot ran with 4.7-5.4 GiB free against fp8's 7.2-8.0.

## Measured so far (one GB10, no cluster)

**Off is unchanged.** A build without the flag compiles every kernel to the same code as main: the cubins of
`attn.o`, `attn_bind.o` and `m4sim` match section for section, except the hash in internal symbol names. m4sim printed
the same stdout as main in three cases, all at TP=4 on the release's layers 0 and 3:

- prefill of 1,500 tokens plus 16 decode steps, with `GR8_TEST_KV_HASH` (R = 2048, every page replicated);
- the same with R = 1024, so pages past 1,024 are spread across ranks;
- three S2 sessions with MTP.

**Kernel test** (`kv4_test`, synthetic weights, 1,000 keys, score sd 3.2). Kernel error is measured against a host
reference on the cache's own decoded latent. Row cost is measured against the exact latent.

| | FP8 row | INT4 row |
|---|---|---|
| append round trip, latent SQNR | 33.9 dB | 20.4 dB |
| decode, kernel error | 2.1e-4 | 2.1e-4 |
| prefill (expand), kernel error | 5.7e-4 | 5.7e-4 |
| row-owner prefill, kernel error | 5.7e-4 | 5.7e-4 |
| attention output, row cost | 0.031 | 0.141 |

**Per-layer attention output** (host, each layer's real latent covariance and real kv_b; 2,048 keys; 64 heads;
random queries). The INT4 row has about 3.7x the FP8 row's error at every layer:

| score sd | FP8 row | INT4 row |
|---|---|---|
| 1 | 0.2-1.7% | 0.7-6.4% |
| 3 | 0.8-5.0% | 2.1-18% |

These queries are random, so the scores are not as peaked as real attention. Treat this as a ratio, not a forecast.

**m4sim, INT4 against FP8 on the release's weights** (TP=4, 1,500-token prompt). Relative L2 of the residual after
prefill:

| layers | residual difference |
|---|---|
| 0 | 0.69% |
| 0, 1 | 1.09% |
| 0, 1, 2 | 1.46% |
| 0, 1, 2, 3 | 1.49% |

- The difference is about the same in every quarter of the prompt.
- Decode chose the same 16 greedy tokens. The sampled tokens' logprobs moved by 0.002 on average (0.013 at most).
- With R = 1024 (CP decode and staging) the numbers match.
- In three S2 sessions with MTP, 168 of 209 steps match. The first difference is at step 15 of one session, on this
  two-layer model's flat distributions.
- No attention error word was set, and every rank stored the same bytes.

**Full-vocabulary KL** (2026-10-12). The served weights (exlr8 v2 `home`) run through a streamed BF16 forward, with the
cached latent (the `kv_a_layernorm` output, which feeds both K and V) replaced by its round trip through each row.
Setup: 100 windows of 2,048 tokens, microbatch 8 (the same rows and plan as the `v2-home` arm). Scored against the
BF16 model:

| arm | held-out KL mean / median | wikitext KL mean / median | top-1 held-out / wikitext |
|---|---|---|---|
| BF16 latent | 0.1488 / 0.0125 | 0.1036 / 0.0143 | 0.883 / 0.905 |
| FP8 row (today) | 0.1511 / 0.0127 | 0.1036 / 0.0140 | 0.882 / 0.905 |
| INT4 row | 0.1609 / 0.0145 | 0.1117 / 0.0155 | 0.877 / 0.900 |
| INT4 row, layers 0-3 on the FP8 row | 0.1617 / 0.0145 | 0.1112 / 0.0155 | 0.878 / 0.901 |

- **Cost of the INT4 row.** Over the FP8 row it adds 0.010 (held-out) and 0.008 (wikitext) KL. That is 6.5% and 7.8%
  of what the weight quantization costs.
- **Pairwise against the FP8 row, same weights:**
  - INT4 row: 0.1155 / 0.0058 held-out, 0.0256 / 0.0043 wikitext.
  - For scale, the FP8 row against a BF16 latent: 0.0947 / 0.0031 and 0.0134 / 0.0021.
  - The BF16 forward's own noise (microbatch 4 against 8): 0.112 / 0.0043.
  - On held-out text the INT4 row is at that noise floor. On wikitext it doubles the FP8 row's divergence.
- **Keeping layers 0-3 on the FP8 row changes nothing measurable.**
- **Per-layer attention output error** (the same input run with the exact and the quantized latent; real queries):
  - FP8 row: 0.2-2.9%, median 2.2%.
  - INT4 row: 0.3-11%, median 8.5%. It grows with depth: 1-5% in layers 1-9, 8-11% past layer 60.
- **Limits.** These windows are 2,048 tokens, so the indexer selects every key, and the MTP layer is not in this
  forward. Long-context needles and acceptance are on the cluster, below.

## On the cluster (TP=4, 2026-10-13)

Four GB10 boxes, the HF release with the DFlash2 drafter. The same session and front end for both. fp8 is main with
#44 and #43; int4 is the same tree built with `G6_KV4=1`, at 1,920 pages unless stated.

- **Gate.** Smoke 12/12 on both. The cold shared-prefix probe's first token stays 'The'; its logprob moves from
  -0.0770 to -0.1991, the same on two int4 boots.
- **Needles** (one fact in synthetic records at 10/50/90% depth, temperature 0, `reasoning_effort` low, every prompt
  cold): 12/12 found on both, with the same answers, at 32K, 64K, 128K and 200K tokens, and the same TTFT
  (24 / 48 / 105 / 181 s). The first token agrees on 11 of 12, logprobs within 0.2. At 128K and 10% depth fp8 closes
  the think block at once and int4 writes a few characters of reasoning first; both answer correctly. With CAP 539,648
  (2,876 pages) a 400K-token needle was found too (TTFT 461 s).
- **RigMark** (`reasoning_effort` low): code 49.1 tok/s against 50.8 (-3.3%), prose 29.9 against 30.6 (-2.3%), 8K
  prefill cold 1,358 against 1,284 (+5.8%).
- **Speculative acceptance**, one request at a time (DFlash2 at TP=4), 4 code and 4 prose prompts of 512 tokens, from
  the `vllm:spec_decode_*` counters: code 3.51 accepted per draft against 3.60 (-2.3%), prose 1.22 against 1.25
  (-2.1%). Decode 51.3 against 51.8 and 33.2 against 34.4 tok/s.

## Not measured yet

- TP=3 and TP=6 boots (their CAP defaults are estimates).
