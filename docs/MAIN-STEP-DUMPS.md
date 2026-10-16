# Offline main-step dumps

`g6m4` supports the same diagnostic environment variables as `m4sim`:

- `G6_RESID_DUMP=FILE`: rank 0 writes the replicated residual after the
  embedding and after each attention/MLP consume. Flash writes all four
  streams. Other ranks do not open this file.
- `G6_LOGIT_DUMP=FILE`: each rank writes `FILE.rank<R>`, its contiguous
  vocabulary range. Collect every rank file, then merge:

  ```sh
  python3 tools/cp/merge_logits.py -o logits.bin logits.bin.rank*
  ```

- `G6_LOGIT_DUMP_FROM=N`: omit steps below N from both dumps (default 0).

Both variables are off by default. Dumps include offline main decode steps,
not tensor prefill, draft steps, or S2 service. `--serve` rejects them rather
than silently writing incomplete files. For `flash_forward.py`, use one
sequence, no tensor prefill, and dump from step 0. CUDA graphs, including
split graphs, capture the residual copies at the same sublayer boundaries.
The extra copies and host writes make these correctness runs, not speed runs.

## Formats

All integers are little-endian int32; payloads are float32, without conversion.

Residual (identical to `m4sim`), repeated per step:

```
s, M, slots, width
float[slots][M][width]
```

`slots = 2 * number_of_layers + 1`. Slot 0 is the embedding; slots `2*i+1`
and `2*i+2` are layer i's attention and MLP outputs. `width = HID` for the
single-stream model, `4*HID` for Flash, with stream-major `[M][4][HID]` rows.

Rank logits have one header:

```
8 bytes: G6LGR001
NR, rank, vocab, vlo, vn
```

`vlo` and `vn` describe a contiguous vocabulary range, as returned by
`bind::vocab_slice`. Each record then contains `s, M, float[M][vn]`.

Merged logits (identical to `m4sim`) have no file header. Each step contains
`s, M`, then `vn, float[M][vn]` for each part in vocabulary order. The merge
checks all ranks, complete non-overlapping vocabulary coverage, matching
step numbers/row counts, and complete payloads. It replaces the output only
after every input record passes. Step numbers can reset for a new sequence.

Host test: `python3 -m unittest tools.cp.test_merge_logits -v`.

## Build and host validation

Validated against base `22689e5`:

- Flash TP4: both `g6m4_flash` and `m4sim_flash` compile and link.
- GLM TP4/TP3/TP6: before and after `g6m4` compile and link. All device
  instruction bytes match: respectively 386/375/375 executable sections,
  totaling 13,228,032 / 11,524,096 / 11,467,264 bytes. Complete extracted
  cubins match after normalizing only compiler-generated internal symbol
  hashes in ELF string tables. No instruction or data bytes are normalized.
- Full host suite passes, including the merge tests. Synthetic TP2/3/4/6
  files merge byte-for-byte into the expected simulator format. Negative
  cases cover missing/duplicate ranks, bad headers, overlapping vocabulary,
  mismatched steps/row counts, truncation, empty dumps, and output aliases.
- The saved six-layer first-token logit dump also survives a four-rank
  split/merge byte-for-byte (24 steps, reversed input-file order).

The distributed dump path has not yet run on a live multi-rank fabric.
These checks establish build compatibility and the merge format, not a
multi-rank numerical or speed result.
