# Flash MTP decode

Layer 45 uses the existing draft-step hooks, with Flash geometry and no mHC.
Its input is the main model's **final-normalized FP32 hidden**, not the raw
mean of the four residual streams. The embedding of the next token is zero
at position zero. The concatenation order is normalized embedding, then
normalized previous hidden. `L045.eh` is a replicated BF16 `[4096,8192]`
matrix; no MXFP8 `eh.mx.gv` record is required.

The layer has its own DSA cache and indexer. It adds the attention and MoE
partials to a single FP32 residual in rank order. The final norm is
`L045.shared_head_norm`; the vocabulary projection shares `G.lm_head.gv`.
A chained draft reads the preceding draft's final-normalized FP32 hidden.
The Flash main-model path and the non-Flash build retain their previous
behavior.

## Scope

This change implements the **decode layer**, not KDA rollback. Do not infer
correct speculative generation from its per-operation gate. Flash KDA
verification rollback, prefill, S2, full-model draft acceptance and speed
remain separate work. Runtime MoE binding uses the existing trellis expert
path. The first per-operation gate uses original BF16 expert matrices;
layer-45 K4 integration requires its encoded release.

## Per-operation GPU test

Generate a fixture from the first eight rows of the original prompt golden:

```sh
python engine/ref/flash/mtp_inputs.py \
  --golden "$GOLDEN/prompt-bf16.safetensors" --dense "$DENSE" \
  --loader "$LOADER" --index "$INDEX" --config "$CONFIG" \
  --rows 8 --out fixture
nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a \
  -DG6_MODEL_FLASH=1 -DG6_NR=4 -DDEC_S2 \
  engine/kernels/tests/flash_mtp_test.cu -o flash_mtp_test
nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a \
  -DG6_MODEL_FLASH=1 -DG6_NR=4 engine/attn/tests/flash_dsa_test.cu \
  engine/attn/flash_attn.cu engine/attn/flash_bind.cu -o flash_dsa_test
./flash_mtp_test fixture 1
./flash_mtp_test fixture 8
./flash_dsa_test fixture/dsa --layer 45 --m 1,8 --ctx 1,5,8 --replay
```

Each operation starts from identical golden inputs. The test covers the two
input norms and position-zero mask, eh projection, both residual additions,
attention/MLP/shared-head norms, router logits, exact routing from golden
logits, streamed BF16 routed/shared experts, and the production BF16 `.gv`
vocabulary kernel. FP32 gates use relative L2 <= 1e-4. The head uses relative
L2 <= 1e-2, top-1 agreement >= 99%, and mean KL <= 1e-3.

The DSA test runs the actual attention binding on all four head shards at
contexts 1, 5 and 8, for single-row and eight-row calls. It checks selection
and graph-replay equality, with output relative L2 <= 1e-2 against a
float64 reference of its FP16 inputs. It also reports error against the
unrounded FP32-input reference. Fixture construction checks that the latter
matches the original golden attention output to 1e-4. This short fixture
does not exercise indexer key exclusion at long context.

This is not an end-to-end runtime gate: DSA projections and trellis experts
retain their separate existing tests. Full runtime composition and the K4
MoE path require the layer-45 encoded release.

## Draft dumps (m4sim)

Both environment variables are opt-in and record **first draft passes only**,
not chain passes. Integers are signed 32-bit; floats are 32-bit. The first
pass counter starts at zero and is identical in both files.

- `G6_DRAFT_IN_DUMP=FILE`: for each pass, `(s,M)`, `pos[M]`, `tok[M]`, then
  `hidden[M,4096]`. The hidden is exactly the FP32 input consumed by MTP.
- `G6_DRAFT_LOGIT_DUMP=FILE`: `(s,M)`, then for each rank in vocabulary order,
  `vn` followed by `logits[M,vn]`. This matches the main logit-dump framing.

The reference comparator can replay the input dump through `FlashRef.mtp`
with a separate MTP state. These diagnostic copies run outside graph capture.

## Verified result

The eight-row BF16 gate passed at batch sizes 1 and 8. The input fixture
contains 56 distinct routed experts. Exact values and the source golden
SHA256 are in `flash-mtp-gate.json`.

| Check | Measured worst | Gate |
|---|---:|---:|
| FP32 MTP operations, relative L2 | 7.869e-7 | 1e-4 |
| Router selection mismatches | 0 | 0 |
| DSA attention, relative L2 (printed precision) | 2.20e-3 | 1e-2 |
| Shared vocabulary projection, relative L2 | 1.212e-3 | 1e-2 |
| Head top-1 agreement | 8/8 | >= 99% |
| Head mean KL | 4.657e-6 | 1e-3 |

DSA: 24/24 checks passed; all selections matched and replay was bit-identical.
The host suite and tiny reference suite exited zero. Both Flash driver
translation units compiled for the target architecture. In the GLM build,
both preprocessed driver translation units match the base apart from
changed diagnostic line numbers; this is not a GLM GPU regression run.
No speed measurement is claimed.
