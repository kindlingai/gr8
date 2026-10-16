# Flash first-token correctness: layers 0–5

The TP4 simulator passed the four `flash_forward.py` gates on one GPU,
using the 64-token golden prompt, lossless dense weights, and the same K4
routed/shared expert release as the CPU reference. Tested 2026-10-14.
This is a partial-model correctness result, not a speed or serving result.

Runtime and reference source: `20ea7da`. Expert checker: `0995c1a`.

## Results

| Check | Measured | Required | Result |
|---|---:|---:|---|
| Worst sublayer stream relative L2 | 0.0029685772 (layer 0 attention) | ≤ 0.02 | PASS |
| Chain top-1 agreement | 72/72 (100%) | ≥ 99% | PASS |
| Mean KL, reference versus simulator | 0.000020828628 | ≤ 0.001 | PASS |
| Greedy predictions | All 9 emitted IDs equal | Equal | PASS |
| Repeatability | 25 step/generated lines identical | Identical | PASS |

Chain logit relative L2: 0.0037678818. Maximum KL: 0.00089625389.
Head-only logit relative L2: 0.0015864940, top-1 72/72.
Embedding relative L2: zero. All 12 sublayers passed.

The command uses `--gen 8`. The current simulator emits the first prediction
from the last prompt position plus eight decode-step predictions: nine IDs.
The comparator checks all nine over 72 evaluated positions, without trimming:

```text
32327 38724 32327 48002 92257 55908 3295 39806 82
```

## Reproduce

Define `RELEASE`, `LOADER`, `CONFIG`, `PROMPT`, `OUT`, and `RUN` for the local
release, reference-loader checkout, source config, golden token IDs, binary
output, and result directory. Run from the repository root.

```sh
MODEL=flash G6_NR=4 KV_FORMAT=fp8 scripts/build.sh "$OUT"
G6_CP=0 G6_RESID_DUMP="$RUN/resid.bin" G6_LOGIT_DUMP="$RUN/logits.bin" \
  "$OUT/m4sim_flash" --v2 "$RELEASE/experts" \
  --v2-slices "$RELEASE/index-slices" \
  --v2-widths "$RELEASE/manifests/exlr8-manifest-k4.json" \
  --v2-dense "$RELEASE" --v2-partial --layers 0,1,2,3,4,5 \
  --prompt "@$PROMPT" --gen 8 --temp 0 --cap 128 \
  > "$RUN/steps.txt" 2> "$RUN/stderr.log"
python3 engine/ref/flash/flash_forward.py --release "$RELEASE" \
  --loader "$LOADER" --config "$CONFIG" --K 4 --layers 0,1,2,3,4,5 \
  --ids "$PROMPT" --m4 "$RUN/steps.txt" --resid "$RUN/resid.bin" \
  --logits "$RUN/logits.bin" --json "$RUN/report.json"
```

Both simulator runs and the CPU comparison returned exit 0. The run used
plain decode launches, no prefill, S2, MTP, or graph mode. The default prompt
chunk size was four. The GPU had concurrent unrelated work; timing is not
reported as performance evidence.

## Input and loader checks

- All 15 source-job SHA256 values matched the immutable queue records.
- Expert layers 3–5 were written with K2/K3/K4 and no error tables; execution
  selected uniform routed/shared K4.
- Four partial rank plans passed with layers 0–5 and vocabulary.
- Each expert layer passed 41,616 half-byte/side-vector comparisons and
  216 exact numerical reconstructions across all three widths and four ranks.
- All 19 transferred dense/K4/index file hashes matched before execution.
- Loader Rust tests, host tests, converter tests (15 tests, one skipped),
  and the synthetic forward positive/negative controls passed.

Peak sampled simulator RSS plus GPU allocation: 18,795,667,456 bytes.
CPU comparison maximum RSS: 6,220,328 KiB. No full-model, layers 6–7, MTP,
quality-versus-BF16, or throughput claim follows from this test.
