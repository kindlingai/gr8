# `--reuse-prefix`: the GPU check to run in a window

The port (`engine/runtime/prefix_reuse.h` + the `g6m4.cu` hooks) is desk-verified only: it compiles and its CPU state
machine passes `engine/tests/test_prefix_reuse.{cpp,py}`. Nothing here has run on a GPU in this lineage. This is the
W44-style check that proves the two claims the old line measured: **a hit's output is the cold run's bit for bit**, and
**a hit's TTFT is far below cold's**.

## One run, four prompts

Run `g6m4` on the six ranks with the TP6 engine line but without `--serve` (the offline loop runs each `--prompt` once,
in order, and exits), with these additions:

- the flag `--reuse-prefix`
- the env `G6_REUSE_ALIGN=1`
- `--gen 32 --temp 0` and four `--prompt @<file>` arguments, the prompts below in order

and keep `--prefill 4096 --pf-lanes 2 --mtp K --graph`. `--cap` must hold `20001 + 32 + 2K`. Rank 0 must print at start

```
# reuse grid 4096 rows (--prefill 4096, lanes 2)
```

Build one id corpus `P` of 20001 ids (any text; the old line used 8 seeded head ids + the KAT corpus, cycled). The four
prompts:

| n | arm | prompt ids | what it is |
|---|-----|-----------|------------|
| 0 | S seed | `P[:14001]` | fills the cache; prefills rows 0..14000 |
| 1 | H hit | `P` (all 20001) | shares 14001 ids with S, so it must be a hit |
| 2 | E evict | 9001 other ids whose FIRST id differs from `P[0]` | drops the cache (`no-lcp`) |
| 3 | C cold | `P` again (all 20001) | the cold reference for H |

## PASS criteria

Read rank 0's stderr for the four `# reuse seq ...` lines and the ranks' stdout for the `step` / `prefill rows` lines.

1. **H is a hit at the exact grid row count.** Rank 0 prints
   `# reuse seq 1 hit retained 12288 lcp 14001 history 14033 prompt 20001 ...` — `history` is 14033 (S's 14001
   prompt ids + its 32 emitted ids) and `lcp` is 14001 or a little more (if S's first generated id happens to equal
   `P[14001]`). `retained` must be exactly **12288**
   (`floor(14000 / 4096) * 4096`). 12288 is an ODD multiple of 4096, so this number alone proves the RGRID grid rule
   (`grid = MP`) is in the binary; a grid-8192 build would retain 8192 and fail.
   The second line for the same seq must read `# reuse seq 1 grid 4096 raw 14000 retained 12288`.
2. **C is cold.** `# reuse seq 3 no-lcp retained 0 ...` (and seq 2 likewise `no-lcp retained 0`).
3. **Bitwise-equal output.** seq 1's `generated` line equals seq 3's.
4. **Bitwise-equal rows.** On EVERY rank, seq 1's and seq 3's stdout must agree:
   - every `prefill rows a..b resid X` line with `a >= 12288` is identical between the two seqs, and
   - the two seqs have the same number of `step ...` lines (and `draft` / `chain` lines) and each one is identical.
   Seq 1 has `reuse rows 0..12287` and no prefill lines below 12288; that is the only allowed difference.
5. **TTFT.** H's time to first token must be far below C's. The old line measured -65% over 14 requests
   (53.8 s vs 155.6 s at `--prefill 4096 --pf-lanes 2`). Add up rank 0's `# seq 1 prefill ...` times and compare
   them with seq 3's.
6. **No fabric or attention error words** on any rank, and `rc 0` on all six.

Any line unreadable or missing = FAIL. The check is fail-closed: a boot that does not print the grid line is not
testing this feature.

## Deeper: every cache row

Criteria 3 and 4 are the output-level proof. The old line also compared every cache row of the hit against the cold
run with `G6_REUSE_HDUMP=<dir> G6_REUSE_HDUMP_SEQ=1` on a hit boot and `=3` on a cold boot (`reuse_hdump.cuh`, ported
here; the dump format is documented at the top of that header). The reader that merges the six ranks' dumps by global
row, `tools/reuse/hdump.py`, is NOT ported — it is at
the hdump reader on the old line.

The gate arm `G6_REUSE_POISON=1` (`reuse_poison.cuh`, ported) sets the stale rows to NaN on a hit, so a read of a row
the hit should not have retained shows up as an attention error word 4 or as different lines. A poisoned run must print
the same lines as the same run without poison.
