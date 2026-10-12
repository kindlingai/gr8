# 4-bit MLA KV latent (`G6_KV4`, experimental, off by default)

The KV cache spends most of its bytes on the MLA latent: 512 dims per token per layer. `G6_KV4=1` stores that latent
in 4 bits, so the same memory holds more pages. The rope key stays BF16 and the indexer keys stay FP8.

The switch is at compile time. A build without it has the same row, the same kernels and the same SASS as before.

## The row

| | FP8 row (default) | INT4 row (`G6_KV4=1`) |
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
  path), and the generic `mla_scores` / `context_parts` (`G6_ATTN_HEADS=0`).
- Prefill reads: `expand` (`engine/attn/prefill.cuh`) and `gqkv4::prepass` (`engine/attn/prefill-qo-bind.cuh`).
- In a KV4 build, `G6_ATTN_FLIP`, `G6_ATTN_P11` and the per-head fixed-K kernels are ignored, and `G6_PF_QO=v1|r1` is
  refused. They read the FP8 row directly.

**Stores.** Page blocks shrink from 3,502,592 B to 2,289,152 B (79 x 64 x 416 + 22 x 64 x 132; 559 x 4 KiB slots).
A tier's compat key also names the row format (`kv int4g32h`), so a KV4 engine never reads blocks an FP8 engine
wrote. `kvman-sim --mla-row 416` models the smaller blocks.

## Build and test

```bash
EXTRA=-DG6_KV4=1 G6_NR=4 scripts/build.sh bin4-kv4                 # the TP4 engine with the INT4 row
TEST=kv4_test G6_NR=4 scripts/c1_tests.sh bin4                     # the FP8 row
TEST=kv4_test G6_NR=4 EXTRA=-DG6_KV4=1 scripts/c1_tests.sh bin4-kv4 # the INT4 row
```

`kv4_test` checks two things on one GPU with synthetic weights:

- **Round trip.** It appends rows through `g6_attn_append`, decodes them on the host, and checks the latent SQNR, the
  codes and the rope keys.
- **Attention.** It runs decode, prefill and row-owner prefill over the cache, and compares each against a
  double-precision host reference. The kernel error is measured against the cache's own decoded latent, and must be
  below 1e-2. The cost of the row format is measured against the exact latent.

`dec_heads_test` is skipped in a KV4 build, because it checks the FP8-row kernels.

## Capacity

Every rank holds the first `SHARE_LIMIT` tokens and 1/TP of the rest: `CAP = R + (P - R/64) * 64 * TP`. The table
below assumes the same per-rank memory for everything that grows with CAP:

- the MLA pages;
- the index keys (22 x 132 B per token, on every rank);
- the CP staging pools (2 x row per token);
- the FP16 prefill key table (about 1.4 KB per token).

Pages per rank:

| TP (R) | FP8 | INT4 |
|---|---|---|
| TP=4 (64K) | 1920 | about 2640 |
| TP=3 (64K) | 1941 | about 2740 |
| TP=6 (32K) | 2026 | about 2700 |

CAP:

| TP (R) | FP8 | INT4 | gain |
|---|---|---|---|
| TP=4 (64K) | 294,912 | 479,744 | 1.63x |
| TP=3 (64K) | 241,600 | 395,008 | 1.63x |
| TP=6 (32K) | 614,144 | 872,960 | 1.42x |

The launcher does not set these yet. Pass `CAP` by hand, and check the boot's memory line.

## Measured so far (one GB10, no cluster)

**Off is unchanged.** A build without the flag compiles every kernel to the same code as main: the cubins of
`attn.o`, `attn_bind.o` and `m4sim` match section for section, except the hash in internal symbol names. m4sim printed
the same stdout as main in three cases, all at TP=4 on the release's layers 0 and 3:

- prefill of 1,500 tokens plus 16 decode steps, with `G6_KV_HASH` (R = 2048, every page replicated);
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
  forward. Long-context needles and acceptance length are still open.

## Not measured yet

- Long-context needles on the cluster.
- Decode and prefill speed.
- MTP and DFlash2 acceptance. The MTP layer's cache is quantized too.
