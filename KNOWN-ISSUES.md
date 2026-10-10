# Known issues

Bugs we are still working out. Fixed ones move to CHANGELOG.md.

## 1. TP=4: which requests get the DFlash2 drafter depends on timing - Open

- A request drafts with DFlash2 only if it is alone at its prompt tail and prefix reuse did not skip into the last 2,048
  prompt positions; otherwise it runs MTP-only for its whole life. With prefix reuse on, that depends on what the pool
  holds and on whether the previous request has finished.
- Effect: RigMark code on one TP=4 build ranged from 41.5 to 49.3 tok/s between boots. Prose and the multi-stream rows
  are not hurt.
- The drafter-choice log line only appears when a request switched drafters or ran MTP rounds, so a request that never
  had DFlash2 is hard to tell apart from one that ran it throughout.

## 2. Prefix-reuse miss near 786K tokens (TP=6, six Sparks) - Open

- A request that should reuse a ~786K-token resident prefix misses and does a full re-prefill.
- Next step: log a reason code for each miss and return the longest valid resident prefix, not all-or-nothing.

## 3. Aggregate decode stays flat past S=4 (TP=6, six Sparks) - Open

- Aggregate decode, tok/s: S1 39.7, S2 53.1, S4 52.1, S6 52.8 (2200 MHz cap).
- We do not know yet if the cause is the scheduler or weight bandwidth.

## 4. One first-request output mismatch after a boot - Open (not reproduced)

- **Symptom:** once, on a fresh TP6 boot (engine started with `--top 20`), the first-request check did not repeat itself.
  The check sends the same 8,203-row prompt twice and decodes 32 tokens greedily (A1, A2). A1 and A2 differed (text hash
  53b2416a vs 4e66e099), and the first prefill sub-batch already had a different residual hash (d54af8d1; every passing
  boot has 17435a0d, ffa6889a, 663a65a8 for the first three).
- **Build:** the logprobs branch, with logprobs off. The same branch's binary run with the same arguments, and main's binary, both
  passed afterwards.
- **Not reproduced:** 24 fresh boots in one run (12 of this branch, 12 of main, alternating, `--top 20`) and 5 more before it
  (4 bisect boots without `--top`, 1 clean rerun with 3 checks): every one gave A1 == A2, text hash 8b23c109, prefill
  hashes 17435a0d ffa6889a 663a65a8 and step-0 hash 129e55007adcb4ea, identical between the two binaries. With logprobs off, the
  branch's code is equal to main in every one of these.
- **Cause:** unknown. Suspected: a rare race on the first prefill after boot that also exists in main (the first
  sub-batch was already wrong, which points at something before or during the first prefill, not at decode). Not proven.
- **If it happens again:** keep the rank logs of that boot (each rank's first three `prefill rows` residual lines), note which binary and
  arguments ran, and repeat the check on a fresh boot of main.
