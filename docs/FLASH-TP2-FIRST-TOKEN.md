# Flash TP2: layers 0–5 first-token gate

The decode-only simulator at `G6_NR=2` passes all four first-token gates
on layers 0–5, with K4 routed/shared experts and BF16 dense weights.
This is a two-rank simulation on one GPU, not a two-device fabric test.

Runtime/reference baseline: `22689e5`. The reference used the same quantized
weights, with three diagnostic `max_abs` fields added to a private comparator
copy (sublayer, head, chain). The forward math and gate thresholds did not change.
No runtime kernel change was needed for TP2.

## Results

72 positions: the 64-token golden prompt plus 8 decoded input tokens.
Full measurements are in [the JSON report](FLASH-TP2-FIRST-TOKEN.json).

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| Worst sublayer stream rel-L2 | 0.002975225677 | <= 0.02 | PASS |
| Chain top-1 agreement | 72/72 (100%) | >= 99% | PASS |
| Mean KL(ref || engine) | 0.0000106692317047 | <= 0.001 | PASS |
| Greedy continuation | 9/9 exact | exact | PASS |

Additional diagnostics, not replacements for the four gates:

- Mean KL is also below `1e-4`; maximum positional KL is `0.000225074361683`.
  The maximum is **not** below `1e-4`.
- Chain logits: rel-L2 `0.003163230887`; maximum absolute difference
  `0.111966311932` across every position and vocabulary entry.
- Head on the engine's own final streams: top-1 72/72, rel-L2
  `0.001594841126`, maximum absolute difference `0.0174905061722`.
- Worst sublayer maximum absolute difference: `0.00618743896484` (layer 3 MLP).
- Embedding rel-L2: zero.
- Greedy IDs: `32327 38724 32327 48002 92257 55908 3295 39806 82`.
  As in the TP4 test, `--gen 8` emits the first prompt-tail prediction plus
  eight decode predictions; none were discarded.

Both simulator runs and the CPU comparator exit 0. Full simulator stdout,
residual dumps, and logit dumps match byte-for-byte between the two runs.
All 21 transferred weight/index-slice checksums pass. Sampled peak RSS plus
GPU memory is 17,027,084,288 and 17,027,088,384 bytes. The GPU runs execute
serially under a shared lock. The `NV_ERR_NO_MEMORY` counts before/after are
134/134 for each run: zero new entries in both intervals.

## Reproduction

Use the same release and prompt as [the TP4 first-token test](FLASH-FIRST-TOKEN.md),
but generate TP2 slices with `tools.convert.v2slices --tp 2 --widths 4`.
The new slices retain the release's existing `set_id` and verified index hashes;
the original release is not restamped or modified.

Build with the Flash flags from `scripts/build.sh`, `G6_NR=2`, and the simulator
target of `engine/runtime/build_m4.sh`. The tested invocation was
`MODEL=flash G6_NR=2 KV_FORMAT=fp8 scripts/build.sh OUT`: its simulator compiled
and linked, then its unrelated `g6m4` target rejected the missing `DEC_S2`
flag. The completed `OUT/flash/m4sim` binary was retained. This does not claim
that the full TP2 build wrapper or the TP2 distributed engine passed.

```sh
G6_CP=0 G6_RESID_DUMP=resid.bin G6_LOGIT_DUMP=logits.bin \
  m4sim_flash --v2 RELEASE/experts --v2-slices TP2_SLICES \
  --v2-widths RELEASE/manifests/exlr8-manifest-k4.json --v2-dense RELEASE \
  --v2-partial --layers 0,1,2,3,4,5 --prompt @prompt.ids \
  --gen 8 --temp 0 --cap 128 > steps.txt

python3 engine/ref/flash/flash_forward.py \
  --release RELEASE --loader EXLR8_LOADER --config config.json --K 4 \
  --layers 0,1,2,3,4,5 --ids prompt.ids --m4 steps.txt \
  --resid resid.bin --logits logits.bin --json report.json
```

The canonical comparator reproduces the four gates; the private diagnostic
copy also reports maximum absolute errors. Simulator binary SHA256:
`f4c05bf1c2bbe3be00b7f9cf947fa47da8c3100112d832f283a8d3c34ace18b4`.

No speed, full-model, MTP, service, BF16-quality, or two-device fabric claim.
