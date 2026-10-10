# Changelog

## 2026-10-10: o_proj prefill reads its trellis tiles (#116)

- `bindp::k_gemm_tr` loads the raw K4 tiles of a dense trellis matrix (half the bytes of its FP8 copy) and decodes each
  stage into the bytes the copy would hold, so its output equals the copy path's to the bit. The decode of the next
  stage overlaps the current stage's mma.
- o_proj prefill uses it, including the pair-owner path (`G6_PF_OWN=4`), whose o_proj half now travels as raw tiles.
  The o_proj FP8 copies are gone: about 2.0 GB less per rank at TP4. `G6_DTR_PF_COPY=1` brings them back (A/B).
- Prefill time, tiles / copy, TP4 rank-0 shapes at 2200 MHz: o_proj 0.97-1.00 at 512 rows, 0.72 at 2048, 0.73-1.00 at
  4096 (the copy path's time varies between runs); the pair o_proj 0.69-0.95. q_b and the dense MLP keep their copies:
  on the tiles they measured 1.07-1.33.

## 2026-10-10: boot from the exlr8 release alone (#116, step 1, part 2)

- `g6m4 --v2 DIR --v2-slices DIR --v2-widths FILE --v2-dense DIR`, with no `--manifest` / `--pack`: the Rust loader
  plans each rank's whole image from the release (`engine/loader-rs/src/release.rs`). The experts come from the width
  files as `--v2` reads them. Every other tensor comes from `dense/`, copied as stored or computed at load (`Op::Gen`),
  under the names a converted pack uses. No TR3 checkpoint, GPTQ codes or packs are needed. Rank 0 prints which form
  each matrix took.
- `--v2-dense-form trellis` (the default) keeps q_b, o_proj and the dense MLP of layers 0-2 as the release's trellis
  entries for the dense trellis kernels, so every weight is the release's own: the MXFP8 matrices as stored, the int4
  o_proj / q_b records from the release's codes, kv_b and the embedding in BF16. Only lm_head is rounded to MXFP8.
  `--v2-dense-form rounded` is the fallback: it rounds those matrices to the MXFP8 and int4 forms of a converted pack.
- Under `G6_PF_OWN=3` the image also holds the full-head q_b / o_proj / UK / UV, so TP=6 needs no QO pack.
- The launcher: `--model` takes the release (https:// or a local copy) as the whole model, and `V2_DENSE` follows from
  it. `--base-packs` stays optional. The set id of a release is its weight set id with the tokenizer files.
- Non-expert weights per rank: TP=4 7.91 GB in the trellis form (11.21 rounded, about 11.6 from packs), TP=3 8.70
  (13.10, 13.80), TP=6 7.13 (9.33). The prefill FP8 copies of the trellis matrices add 2.81 GB per rank at TP=4.
- Checks: `tests/release_check.py` compares every planned non-expert tensor with an independent derivation
  (`loader/exlr8.py`, `g6ref.py` records, `plan.py` ownership), byte for byte: TP=4 trellis on all four ranks (876
  tensors, 11.85 GB), TP=4 rounded (302, 6.00 GB), TP=3 trellis on all three ranks (348, 8.30 GB), TP=6 rounded with the
  full-head tensors (72, 2.64 GB). On one GB10, m4sim from the release (layers 0-3, all ranks agree) matches an fp32
  PyTorch forward of the same weights (`engine/ref/release_forward.py`): per sublayer 1.2e-3 to 9.3e-3 rel L2 in the
  trellis form, logits KL 1.5e-5 nats per position, where a BF16 copy of that forward gives 9.3e-5. TP=3 and TP=4
  generate the same tokens. Prefill and decode-only runs generate the same tokens.

## 2026-10-10: the engine serves the dense trellis matrices (#116)

- Decode: `bind::make_layer` binds q_b, o_proj and the dense MLP of layers 0-2 from the release's trellis tensors
  (`LLLL.NAME.trellis / .suh / .svh`, `dtr::layer_mats`) when a layer has them: one launch each for q_b and o_proj,
  gate + up and down for the MLP. o_proj and down write the fabric send slot under `DEC_SLOT`. The MTP layer takes the
  same path. `q_b.gptq` (int4 g32, the layers where int4 won the encode) has a binding too. `G6_DTR=0` binds the int4 /
  MXFP8 records instead. Each rank logs how many layers run each form.
- Prefill: the matrices run as exact FP8 copies of their tiles on today's MXFP8 GEMMs. The quantize pass applies the
  input rotation (`Xq::suh`) and the GEMM epilogue the output rotation (`Gemm::osvh`). Prefill time stays at today's. `G6_PF_OWN=4` carries the copies and the pair's vectors over the wire. OWN=3 needs `--qo-pack`.
- `dtr_test --bind` runs one dense layer through both binders on one GPU and `engine/ref/dtr_ref.py` checks the results
  against `loader/exlr8.py` at TP=3, 4 and 6.

## 2026-10-10: `bin/g6load dflash2` converts the drafter without numpy

- `g6load dflash2 --src DIR --out DIR [--nr 4] [--ranks R] [--bf16 CLASSES]` (`engine/loader-rs/src/dflash2.rs`) writes
  the DFlash2 drafter's `dflash.r<r>.pack` and `.manifest.json`, byte for byte what `tools/convert/dflash2.py` writes:
  the same row and column slices, MXFP8 quantization (E8M0 scales, E4M3 round to nearest even) and decode record
  layouts, a sha256 per tensor and the manifest JSON as Python prints it. On `incoai/GLM-5.3-DFlash2` all four TP=4
  ranks match (pack sha256 and manifest bytes), in 7-9 s a rank against 14 s, and so does rank 1 with every class
  in BF16.
- `./gr8 glm53 up --drafter` converts this box's rank with it, so nothing on the boot path needs numpy now.
  `tools/convert/dflash2.py` stays for `./gr8 glm53 packs` and its `--head` test packs.
- `cargo test` compares both on a small synthetic drafter (`tests/data/dflash2`, from `tests/make_dflash2_fixture.py`)
  with zero groups, -0.0, BF16 subnormals and rounding ties.

## 2026-10-10: dense trellis kernels for the exlr8 dense weights (#116, not wired)

- `engine/kernels/dtr.cuh`: decode GEMV for the release's trellis K4 dense matrices (q_b, o_proj, the dense MLP of
  layers 0-2) on the release's entries as stored. One launch per matrix rotates its input in a prologue and reduces
  in fp32. The MLP's gate and up run as one launch whose epilogue writes down's rotated input. `dtr_prefill.cuh` runs
  prefill on an FP8 copy of the tiles (exact: the codebook is E4M3) with today's MXFP8 GEMMs inside the rotations.
- `engine/kernels/tests/dtr_test.cu` + `engine/ref/dtr_ref.py`: checks against `loader/exlr8.py` on the release's
  files and times decode against today's GEMVs. On one GB10 at 2200 MHz, TP4 rank shapes, M = 1/3/8: q_b 0.53-0.57x
  of today's MXFP8 time, o_proj 0.90-0.95x of int4 g32, the dense MLP 0.62-0.68x of the int4 K1 path.
- The engine does not call them yet: the loader's tensors (`LLLL.NAME.trellis/.suh/.svh`, `dtr::layer_mats`) come next.

## 2026-10-10: a decoder for the exlr8 dense and vocabulary files (#116, step 1, part 1)

- `engine/loader-rs/src/exlr8.rs` decodes the exlr8 v2 release's `dense/exlr8-dense-LLL` and `dense/exlr8-vocab`
  matrices in every format they use (trellis K4, int4 g32, MXFP8, FP8 block, FP8 row, BF16, FP32). The values equal the
  output of `loader/exlr8.py` bit for bit in fp32. The decoder takes any block of a matrix, so a rank reads only its own
  trellis entries. The two Hadamard products sum in input order with one fused multiply-add per term. NVPL and cuBLAS
  use that order for the reference's fp32 matmuls, and any other order changes the last bit of some results.
- `exlr8dump` writes a layer's decoded tensors under the reference's names. `tests/exlr8_compare.py` compares the dump
  with `loader/exlr8.py` on the release's files: all 80 files (549 matrices, 16.0 billion elements) are bit-identical in
  fp32 and in BF16, and so is every rank's block at TP=3, 4 and 6 for layers 0, 2, 3, 4, 6, 78 and the vocabulary.
- `cargo test` checks the decoder against the reference's output on a synthetic fixture of every format and side
  (`tests/data/exlr8-fixture.safetensors`, 194 KB), so CI needs neither torch nor the release.
- Nothing in the engine uses it yet: the boot from the release alone builds on it.

## 2026-10-09: DFlash2 under `--s2` (the S2 launcher default, `DFLASH=1`)

- `--s2 --dflash-manifest .. --spec-adaptive` (with `--mtp`): a session alone at its prompt tail drafts with DFlash2
  as the plain build's `dflash_sequence` does (prompt context from the prefill's taps, adaptive 1..7, one SpecLen per
  process). Each DFlash2 round also runs the MTP first-form pass over the accepted rows, so when a second session
  arrives the K-1 chain steps hand the session to the shared MTP rounds (`Batch::demote`). A demoted session stays on
  MTP. The batched graphs are captured without the drafter's residual taps; the solo graphs keep them. Guard rows are
  14 with the drafter loaded (a verify writes 7 rows past the last token).
- A prefix-reuse hit drafts with DFlash2 only when its re-prefill covers the drafter's 2,048-row window.
- Gate (TP4): 10 requests at temp 0 with logprobs, adaptive k, byte-identical to the plain build's DFlash2; 8 concurrent
  requests (the first starts on DFlash2 and demotes) byte-identical to MTP-only S2. RigMark 1.0.0 (knapcio's
  comparison id, `reasoning_effort` low): code 49.5, prose 28.4, structured 68.4 tok/s (plain DFlash2 49.2 / 29.1 /
  67.0; S2 MTP 40.9 / 31.3 / 43.2), C2 50.6, C4 63.6. Free memory at 8 sessions: 2.7-3.6 GB per node.

## 2026-10-09: prefix reuse under `--s2` (`REUSE=1 scripts/launch_s2.sh`)

- `--s2 --reuse-prefix`: a retired session's run stays in the pool with its prompt + emitted ids (`s2/pool.h`). At
  admission the head matches the new prompt against every retained run with the plain build's rules
  (`prefix_reuse.h`); a hit takes that run's base, extended in place, and its prefill starts at the retained row
  (`Grant::reuse`, sent to every rank, which re-derives it before claiming). Retained pages count as free: a claim
  that overlaps one evicts it, and a past-R run evicts all of them.
- On by default under `--s2`: the grid rule (`G6_REUSE_ALIGN`, whole 4,096-row prefill cells, so a hit is exact) and
  `G6_REUSE_NOEND` (the `prompt >= written_end` rule refused replaying a prompt as it was; attention reads only
  positions at or below a row's own).
- New runs go in the middle of the largest gap free of live and retained runs, so the conversation before it and this
  one both have room to grow on their next turn.
- Gate (TP4, 20,001-token prompts, temp 0, logprobs 5): every output identical to a cold boot of the same binary; a
  continued conversation 7.1 s vs 17.3 s cold, the same prompt again 4.6 s vs 17.3 s, two conversations continued at
  once next to a decoding session 16.1 / 15.4 s vs 28.8 / 28.1 s.

## 2026-10-09: fused K1 schedules only the plan's slots (#8)

- The fused K1 kernel handed out items for M x 8 slots (the capacity) and ran the empty ones (fence, barrier, counter)
  for slots past the plan's experts; it now hands out the plan's own slots (`n_slots_dev`). The last CTA zeroes the
  counters with all its threads instead of one. Same work per real item. Experts per step at one session: 14.49 ->
  14.21 ms at 1 row, 29.55 -> 29.22 ms at 3 rows; isolated K1 at 4-8 rows 220 -> 227 GB/s.


## 2026-10-09: multi-tile GEMVs at up to 128 registers (#75)

- The dense GEMVs above 8 rows (S2 batches) ran a 1024-thread-bound kernel, capped at 64 registers: the INT4 o_proj and
  q_a fell from ~190 to ~120 GB/s at 24 rows. Bindings with ks <= 16 now take a 512-thread-bound copy (fewer tiles per
  CTA, same bits). S2 at 24 rows: o_proj 9.2 -> 7.5 ms, q_a + kv_a 6.75 -> 6.3 ms per step; 8 sessions 75.5 -> 77.0 tok/s.


## 2026-10-09: separate K1 launches at 16+ rows (#8)

- `row_budget_test --time-v2`: K1 on an exlr8 v2 layer per row count, with the bytes read and GB/s (fresh routing and
  six layer copies per run, so every byte comes from DRAM). On GB10 at 2200 MHz the fused launch reaches 220-225 GB/s
  at 3-12 rows and falls to 203 at 24 and 193 at 32, where the separate fc1 / fc2 launches are 2-3.5% faster.
- v2 layers now take the separate launches when the step has 16+ rows (S2 batches; the fused / separate choice keeps
  every partial and its order). S2 at 8 sessions: 73.5 -> 76.5 tok/s (code); 4 sessions unchanged.


## 2026-10-09: short prompts skip OWN=4 (`G6_PF_QP_MIN`, default 768)

- OWN=4 pays a fixed ~0.13-0.35 s per request (the pair weights rolled through every layer), so it lost on short prompts:
  a ~50-token prompt's TTFT was ~0.69 s against ~0.34 s with OWN=2 (TP4 prefill tok/s OWN=4 / OWN=2: 256 tokens
  401 / 506, 1024 tokens 846 / 809, 8192 tokens 1,254 / 809). A prompt shorter than `G6_PF_QP_MIN` now runs OWN=2 in
  both the plain build and S2; 0 restores OWN=4 for every request that fits `G6_PF_QP_CTX`. Short prompts' output then
  equals an OWN=2 (and OWN=1) boot's. S2 with OWN=4 at 4 / 8 sessions back to OWN=1's decode aggregate (code 61.6 / 73.8).

## 2026-10-09: `G6_PF_OWN=4` prefill under `--s2`

- S2 prefill can run pair-owner QO attention (`G6_PF_OWN=4` with `G6_PF_QP_ROLL` / `G6_PF_QP_CTX`, as the TP4 launcher
  sets them); S2 prefill runs on the CP staging pool, indexed by position, so a session's base page never reaches the
  QO prepass. Each request takes OWN=4 or OWN=2 by the plain build's rule (prompt + gen against `G6_PF_QP_CTX`). Solo
  output equals the plain build's OWN=4 output byte for byte; 8K prefill 818 -> 1,260 tok/s.

## 2026-10-09: adaptive k verify costs re-measured (#75)

- `spec_len.h`'s verify-time table (ms by rows 1..8) now holds the current DFlash2 step times
  (43.0 / 52.4 / 60.4 / 68.4 / 76.1 / 83.7 / 91.1 / 97.4, was 44 ... 111): steps of 4-8 rows got 9-12% cheaper with
  #78-#80. Adaptive k now picks longer drafts. One session, DFlash2: after a 57.6K prompt 30.3 -> 34.1 tok/s (two
  runs each); short prompts code / prose / structured 38.9 / 35.0 / 41.3 -> 38.2 / 36.8 / 40.2.

## 2026-10-09: one-launch absorb / value project at any row count (#75)

- The plain build ran absorb and value project per row (weights reloaded per row) above 3 rows, so DFlash2 verify
  steps of 4-8 rows paid it. They now take the one-launch kernels at any row count. One session, DFlash2 (default):
  short prompts +2% (38.1 / 33.6 / 40.5 -> 38.7 / 34.5 / 41.3 code / prose / structured), 31.9 vs 28.0 tok/s after a
  57.6K prompt. MTP k=2 and prefill unchanged.

## 2026-10-09: head-batched attention for every decode call (#75)

- The head-batched scores and context kernels now serve every decode attention call, not only the fixed-K ones: verify
  rows of a fresh index and contexts past 2048 take them too (the generic `mla_scores` / `context_parts` do the same
  arithmetic; `dec_heads_test` checks both). One session, TP4: MTP k=2 34.2 -> 36.3 tok/s after a 13.5K prompt and
  28.0 -> 32.2 after 57.6K; DFlash2 32.8 -> 35.1 and 23.7 -> 28.0. Short prompts and prefill unchanged.

## 2026-10-09: decode step profile and attention kernels (#75, #76)

- `G6_DEC_EVPROF=N`: per-stage decode step times from event-record nodes in the captured main-step graphs (attention
  split into its kernels), printed every N main steps in each rank's log.
- Decode attention scores and context load each key once for a group of heads; absorb and value project run all rows of
  an S2 step in one launch with the weights loaded once; `k_ar_add_norm` loads all peers before adding. Bit-identical
  (`dec_heads_test` in `scripts/c1_tests.sh`). S2 at 24 rows: attention 51 -> 28 ms per step.
- Boots without QO attention (every `--s2` boot) now build the decode RoPE table; absorb computed RoPE in FP64 before.
- S2 aggregate decode (code, TP4, 2200 MHz): S4 55.6 -> 60.3 tok/s, S8 67.8 -> 74.0. One session and prefill
  unchanged.
- `--s2` + `G6_S2_ADAPT=1` (opt-in): each round's MTP draft count from the sessions' accept counts and the S2 step costs.

## 2026-10-08: prompt-prefix reuse, `--reuse-prefix` (ported from the K1/CP serve line)

- `g6m4 --reuse-prefix` keeps the attention caches of the longest common prefix of the new prompt and the previous
  sequence (its prompt + the ids it actually emitted) instead of resetting them, so an agentic conversation re-prefills
  only its suffix. `engine/runtime/prefix_reuse.h` holds the CPU state; the engine's cache path is unchanged.
- `G6_REUSE_ALIGN=1` makes a hit exact: it retains only whole cold prefill grid cells,
  `[0, floor(min(lcp, pf_end) / grid) * grid)` with `grid = --prefill MP`, and re-prefills from that grid point. Every
  retained cell was then computed with the cold chunk shape, so a hit's cache rows and output tokens are the cold run's
  bit for bit. A retained prefix below one grid cell is a miss (`why` = `grid`).
- Off unless given. Env arms: `G6_REUSE_STRICT=1` (strict append only), `G6_REUSE_NOEND=1` (drop the
  `prompt >= written_end` rule), and the gate/diag arms `G6_REUSE_POISON=1` (`reuse_poison.cuh`: a hit NaNs the stale
  rows first, so a read of one shows up), `G6_REUSE_HDUMP=<dir>` + `G6_REUSE_HDUMP_SEQ=<q,..>` and
  `G6_REUSE_ZSCRATCH=1` (`reuse_hdump.cuh`).
- Refused with `--dflash-manifest` (the drafter's prompt-context taps
  are written by the prefill rows a hit skips).
- `engine/tests/test_prefix_reuse.cpp` (CPU, in `scripts/build.sh`) and `engine/tests/test_prefix_reuse.py` (the same
  under ASan/UBSan, plus the `g6m4.cu` hook order) cover the state machine. Measured on the old line at `--prefill 4096
  --pf-lanes 2`: hit TTFT 53.8 s vs cold 155.6 s over 14 requests (-65%), hit output identical to cold. Not yet measured
  here.

## 2026-10-05: Rust front end, g6front-rs (#15)

- New `frontend/g6front-rs/`: the OpenAI/vLLM front end in Rust (hyper, HF `tokenizers` 0.23.2 as the Python
  front end uses, minijinja). It takes the same flags and gives byte-identical responses to `g6front`.
  `frontend/tests/diff_fronts.py` checks 127 cases, then replays the successful ones concurrently. Opt-in with
  `FRONT=rs` in `launch_tp6.sh`/`launch_tp4.sh` after `scripts/build_front.sh`; Python stays the default until the
  cluster gates run.
- In a linux/aarch64 container, against the Python front end: 64 vs 211 MB resident idle and 95 vs 362 MB under 16
  streams. TTFT added at a 225K-token prompt is 38 vs 316 ms (parallel encode at safe pre-tokenizer cut points). It
  carries about 3.4x the stream chunks per second at less than half the CPU per chunk. Details in
  `frontend/README.md`.
- `mock_engine.py --plain`: no per-request work that grows with the prompt, for front-end benchmarks.

## 2026-10-05: adaptive draft length (#12); TP4 drafts with DFlash2 by default

- `scripts/launch_tp4.sh` now boots the DFlash2 drafter with an adaptive draft count of 1..7 by default. Against MTP k=2
  at 2200 MHz: on answer text code +21%, prose level, structured +46%; on reasoning text about level (code -2.6%,
  within boot noise). It uses less memory, because the MTP layer is not built. `DFLASH=0` brings back MTP k=2 with a
  fixed draft count (setting `MTP=k` does the same). The launcher stops with instructions when a node has no drafter
  pack. TP6 (`launch_tp6.sh`) and multi-session (`--s2`) stay on MTP.
- `--spec-adaptive` chooses each round's draft count for MTP and DFlash2 (`engine/runtime/spec_len.h`, one host
  function per round). Expected tokens come from per-position acceptance, kept per bucket of the drafter's own
  probability of the draft, and shrunk toward the position's recent rate. Cost comes from the verify step time by row
  count at 2200 MHz. The decision reads only inputs that are the same bytes on every rank. MTP also stops its chain when
  one more draft does not pay. The engine flag is off unless given, and fixed k is unchanged. The TP4 launcher gives it
  with DFlash2.
- The DFlash2 selector walk reports each draft's probability among its 16 candidates (`DflOut::lp`, softmax of the walk
  scores). The drafts do not change.
- `engine/tests/test_spec_len.cpp` (CPU) and `engine/ref/dflash2_ab.py` (two feature sets for one token sequence
  through the reference drafter: feature distance and teacher-forced acceptance).

## 2026-10-05: exlr8-e4m3 expert packs, chosen per layer (#19)

- The converter reads TR3 and exlr8-e4m3 checkpoints. Both have the same tensor names and layout. The layer file's
  safetensors metadata `exlr8` tells the two formats apart, and the converter checks it against exlr8's shared input
  signs. An exlr8-e4m3 layer's `tr3.*` manifest entries carry `"expert_format": "exlr8-e4m3"`, and the manifest's top
  level maps those layers to their format (`expert_formats`). The converter writes TR3 packs and manifests byte-identical to before, so existing TP4 and
  TP6 packs stay valid. TP6 works from the same checkpoint: the per-128-block vectors take exlr8's 512-channel scale
  groups as they are.
- Decode rounds each codebook value to FP8 E4M3 (RN-even, saturating) on exlr8-e4m3 layers, as the encoder did and as
  prefill already does: `k1_tr3_fc1/fc2/fused<true>`, picked per layer at bind time. The TR3 kernels' SASS is unchanged
  (`G6_NR=6` and `G6_NR=4`). Unit test: `engine/kernels/tests/tr3_e4m3_test.cu`, built by `scripts/build.sh` and run
  there when a GPU is visible.
- exlr8-e4m3 packs need an engine at or after this change. An older engine ignores `expert_format` and runs them
  without the rounding, with no warning. At TP4 that raised KL against the TR3 reference from 0.049 to 0.053.
- At bind time each exlr8-e4m3 layer must have one hidden-side sign vector across its block rows (`tr3.v1`,
  `tr3.v3`), or the rank stops. Rank 0 logs which layers are exlr8-e4m3.
- TP4 at 2200 MHz, exlr8-e4m3 against TR3 on the same build: prefill, TTFT and decode within boot noise, 87K needle
  found, KL against the TR3 reference 0.049 (0.053 without the rounding), top-1 agreement 93.9%. Details on #19 / the PR.
- `tools/convert/experr.py`: one layer's routed-expert error against BF16 for TR3 and exlr8 checkpoints (CPU).

## 2026-10-05: rejection sampling for drafts (opt-in, `SPEC_ACCEPT=rejection`)

- `--spec-accept rejection` makes verify steps accept drafts by rejection sampling at temperature > 0, for MTP and the
  DFlash2 drafter. Drafts are greedy. The target keeps a draft d with probability p(d) under the request's temperature
  and top_p (its own uniform draw, keyed by seed and position), else it samples p without d. The output distribution is
  still the target's, but a seed no longer gives the same text as without speculation. The default stays `coupled`.
- New head kernels `k_head_local_rs`, `k_head_merge_rs` and `k_head_merge_tp_rs` (`engine/kernels/glue.cuh`), chosen by
  `Model::rs`. With the flag off the head runs the same kernels as before. Temperature 0 gives the same bits in both modes.
  `k_head_local`'s source and PTX are unchanged, but its SASS is not: the shared arrays of `head_local_topp` and
  `head_local_topk`, which now have two calling kernels, move to module scope, so its shared-memory offsets and schedule
  differ from before. A SASS diff against an older build shows this function even with the flag off.
- `engine/runtime/rstest.cu` (build_m4.sh `rs`): the head kernels of all ranks on one GPU, token frequencies against the
  target distribution by chi-square for both modes.

## 2026-10-05: DFlash2 block drafter (opt-in, `DFLASH=1`, TP4)

- `--dflash-manifest/--dflash-pack` (`engine/runtime/dflash.cuh`) replace MTP with the DFlash2 drafter of GLM 5.3. It
  drafts 7 tokens per round in one forward of its 6 layers, from the target's residual after layers 5, 19, 33, 47, 61
  and 75. Each TP4 rank holds 16 query heads, 2 KV heads and 3,072 MLP channels as MXFP8 records (0.70 GB). BF16
  records gave the same acceptance. The drafter runs on its own fabric region (14 collectives) as one captured graph,
  5.6 ms per step at 2200 MHz. Its KV is a 4,096-position ring per layer (25 MB per rank), and taps in the prefill give
  it the prompt's last 2,048 positions.
- `--dflash-sel 1` (default) runs the DFlash2 selector walk over each position's top 16. `--dflash-k K` verifies only
  the first K drafts. Committed tokens are always the target's own samples. At `--dflash-k 2` the output equals MTP
  k=2's byte for byte.
- Target taps: `Model::dfl_tap` and `Prefill::dfl_tap` hooks in host code, null without `--dflash`. The MTP path and the
  TP6 build stay as they were.
- Tools: `tools/convert/dflash2.py` (drafter safetensors -> rank packs, `--head` for lm_head test packs),
  `engine/ref/dflash2_ref.py` (NumPy reference), `engine/ref/dflash2_check.py` with `engine/runtime/dfltest.cu` (the
  step of all ranks on one GPU against the reference), `engine/ref/dflash2_replay.py` (replays a `G6_DFL_DUMP`).

## 2026-10-04: S2 round counters and the S2 shim (#28)

- **Round log.** `G6_S2_ROUNDLOG=1` (DEC_S2 build, rank 0) prints one `# s2_round` line per decode step: kind, round id, active sessions of the round, sessions and rows of the step, launch+fence ms, and on main steps the distinct experts per MoE layer. Off by default, no cost. `tools/s2_costfit.py` fits `step_ms = base + per_expert * experts(d*rows) + per_token * rows` per step kind.
- **`frontend/g6s2shim.py`.** PROTOCOL.md v1 on the engine socket over `g6m4 --s2 DIR`, so several requests run at once, with a real `CANCEL`. `tests/fake_s2_engine.py` and `tests/test_s2shim.py` run it without a GPU. See `frontend/README.md`.

## 2026-10-04: logprobs and prompt_logprobs (issue #23)

- The head returns the top 20 tokens of a row on request (`SampleRow::topk`): each rank selects its slice's best 20 and
  sends them after the normal head record; rank 0's merge turns them into ids, logprobs (log-sum-exp over the full
  vocabulary) and the ranks of the sampled and teacher-forced tokens. `engine/kernels/topk.cuh` holds the order, merge
  and rank rule. A step with no asking row sends the same fabric payload as before (boot gate: pbench with logprobs off vs main).
- `--serve` requests take `lp <k>` / `plp <k>`; `resp.<n>` gets `l` lines (before the `t` line) and `p` lines.
- `prompt_logprobs` runs the prompt through decode steps (teacher forced, 8 rows per step), not the prefill kernels, so
  the numbers are decode numerics and the prompt is limited to 4096 tokens by default. S2 sessions reject both keys.
- `frontend/g6shim.py` accepts `logprobs` / `prompt_logprobs` (max 20) and reports `max_logprobs: 20`.
- `engine/tests/test_topk.cpp`: CPU checks of the log-sum-exp, the merge and the ranks.

## 2026-10-04: rank health probe, boot admission, progress-thread isolation (#29)

- New `tools/health/`: `rank_probe.py` probes every rank before a boot (RDMA write rate per rail around a ring of the ranks, GPU clock and throttle reasons under load, a driver wedge check, device-memory read and host-to-device copy rates compared between ranks, proxy CPU state) and names the reason for each refusal. `gpuprobe.cu` is its GPU part (built by `scripts/build.sh`, copied by `scripts/stage.sh`). The probe only reads and measures, and refuses a node where a model is booted.
- `scripts/launch_tp6.sh` runs the probe first and refuses to boot (exit 3) when a rank fails; `RANK_HEALTH=0` skips it. The shim and front end now run off the proxy CPUs.
- Engine: `G6_PROGRESS_ISOLATE=1` (default) moves non-proxy threads off the proxy CPUs before the first barrier. The thresholds were checked on six healthy nodes (`tools/health/README.md`).
- `bench/gate/gate.py run` needs `--health FILE` (or `--no-health`) and stores the probe result in every run record; `baseline` and `check` refuse records without an admitted probe result. A missing or failed RDMA write result refuses the rank. Isolation exempts only the registered proxy thread ids.

## 2026-10-04: pair-owner QO prefill at TP4 (`G6_PF_OWN=4`)

- **`G6_PF_OWN=4`** (TP4 builds only; [docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)). OWN=3's attention block split
  over the head pairs {0,1} and {2,3}: each rank runs its pair's 32 heads on its own owner rows and its counterpart's
  (rank ^ 2), sends the counterpart's o_proj partial to that rank alone, and sums the two partials in a fixed order.
  The pair weights roll through two slots per layer, built from the rank pack and the pair partner's half over RDMA
  (`G6_PF_QP_ROLL`, a new bulk fabric on port + 56), so the cost is about 1.3-1.4 GB of device memory plus 84 MB pinned per rank instead of the 13 GB QO pack (about
  1.4 KB per position of it scales with `G6_PF_QP_CTX`).
  `tools/convert/pairpack.py` builds resident pair packs (`--qp-pack`) for smaller pools. Ported from adapt's TP4 tree.
- **`launch_tp4.sh` runs `G6_PF_OWN=4` by default** (`QP_ROLL=2`, `QP_CTX=$CAP`, `QP_EARLY=1`). At 2200 MHz: 8K/32K
  prefill 1,318/1,288 tok/s (OWN=1: 816/654), 87K TTFT 73.3 s (156.9 s), decode unchanged within noise, lowest
  MemAvailable 5.2 GiB at a 393,216 pool with a 250K prompt.
- Fabric: reduce-scatter collectives can go to one peer (`Fabric::Rs::p2p`), and bulk jobs can address one peer
  (`BulkJob::peer`). Defaults keep the old behaviour. The TP6 g6m4 and m4sim SASS is identical to main.

## 2026-10-04: TP4 (four Sparks) end to end, and pinned host-to-device copies

- **TP4.** The engine builds and runs at `G6_NR=4` (`G6_NR=4 scripts/build.sh`). Every per-rank width that was a TP6
  literal (attention head groups, vocab slice, routed-expert blocks and scratch, dense channels, buffer sizes, rank
  count checks) now comes from `engine/g6nr.h`; at NR=6 the values are the old ones and the TP6 GPU code is unchanged.
  The converter takes four GPTQ dirs (`tp4/r0..r3`) and writes four rank packs (`plan.set_nr`, `nr: 4` in the
  manifests); `qopack.py` takes four rank packs; `build_packs.sh` picks TP4 from the GPTQ dir. New `launch_tp4.sh` and
  `config/cluster.env.tp4.example`; `build.sh BUNDLE_LIBS=1` and `stage.sh SHARED=1` for nodes without the CUDA
  toolkit and packs on a shared mount. TP4 runs `G6_PF_OWN=1`: the 13 GB QO pack does not fit next to a 90 GB TP4
  rank pack. First numbers and the open QO question are on #2.
- **Pinned host-to-device copies** (`engine/h2d.h`, `g6h2d::copy`). Every copy from pageable host memory to the GPU
  goes through a per-thread pinned buffer, so a 64 KiB-page kernel cannot hang on it (#6). Host code only.

## 2026-10-04: gate harness (`bench/gate/`)

- New: `bench/gate/gate.py` measures a build (KL drift, needle, long prompt, pbench, workloads, drafter counters), appends one JSON line per run, builds a baseline per topology and checks a build against it. See `bench/gate/README.md`. No engine change.

## 2026-10-04: S2 multi-session snapshot (since b85a7a6, the K1H2 snapshot)

The default build (`scripts/build.sh`) is still K1H2. The new multi-session code is compiled in only with
`EXTRA=-DDEC_S2 scripts/build.sh`, and it is turned on at run time with `g6m4 --s2`. See KNOWN-ISSUES.md: S2 still has an open crash on long multi-chat decode.

- **S2 batched multi-session decode (opt-in, `-DDEC_S2`).** New files: `engine/runtime/s2.cuh`, `s2_batch.h`,
  `draft_shape.h`, `engine/runtime/s2/*` (scheduler, sessions, run pool, control channel, native dispatch).
  - Up to 8 sessions decode in one step. Sessions are grouped into MTP row classes (S = 3..8 at mtp 2 use ceil(S/2) groups per round).
  - Each session's KV is one contiguous run in a shared pool. A slot can move to a new base, and the 2->1 / S=1 transitions are handled.
  - Prefill can start at a base page. Below R the loop runs on the CP binding (`g6m4 --s2`).
  - The rank-0 shim talks to the native adapter over persistent control sockets.
- S2 slot views are sized for MX decode rows, not for the prefill chunk. This saves about 8.9 GB per rank at the 614K boot.
- A batched S2 chain draft step now has one row per session.
- If a step fails, S2 prints the fabric / attention / CP error words before the fail-closed check.
- New per-step timer `G6_STEP_TMR=1`, off by default. It prints a per-session request/event label for each step (TAIL / DECODE1 / ROUND / MIXED).
- Control-path disconnect logging: on a recv/send failure the engine prints ret, errno, remaining bytes and frame counts before it throws the epoch-fatal error.
- SELX (`engine/attn/selx.cuh`): our own exact top-2048 select for prefill index tiles. It runs on tiles with n >= 32 rows and context >= `G6_PF_SELX_CTX`. Its output is identical to SEL32's. Unset = SEL32 (unchanged behaviour).
- CP decode `-DCP_F1A`: the phase-2 per-partial wait + get is fused into one launch (off by default).
- The front end metrics have a hook for S2.
