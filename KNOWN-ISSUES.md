# Known issues

Bugs we are still working out, by model. The PR that fixes one removes its entry.

## GLM-5.3

### 1. Prefix-reuse miss near 786K tokens (TP=6, six Sparks) - Open

- A request that should reuse a ~786K-token resident prefix misses and does a full re-prefill.
- Next step: log a reason code for each miss and return the longest valid resident prefix, not all-or-nothing.

### 2. Aggregate decode stays flat past S=4 (TP=6, six Sparks) - Open

- Aggregate decode, tok/s: S1 39.7, S2 53.1, S4 52.1, S6 52.8 (2200 MHz cap).
- We do not know yet if the cause is the scheduler or weight bandwidth.

### 3. One first-request output mismatch after a boot - Open (not reproduced)

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

## GLM-5.3-Flash

No open bugs recorded. [docs/FLASH.md](docs/FLASH.md#not-yet) lists what Flash does not do yet.

## DeepSeek-V4.1-Flash

### 1. The `./gr8 ds41` route has not been run end to end from a clean machine - Open

- [docs/DEEPSEEK-BOOT.md](docs/DEEPSEEK-BOOT.md) was checked against the launcher source. Our own boots, and every
  number in [docs/DEEPSEEK.md](docs/DEEPSEEK.md) and [docs/FIRST-BOOT.md](docs/FIRST-BOOT.md), went through a separate
  test harness. So no boot time through `./gr8 ds41 up` is measured yet.
- Setup has two manual steps GLM does not: the whole 510.6 GB release on every box, and the Engram token map.

### 2. No KV store below the device - Open

- `KV_STORE` (`--kv-store`) is refused for `ds41`: the launcher stops ("has no KV store") and so does the engine
  (`scripts/engine_args.sh`, `engine/runtime/g6m4.cu`). There is no SSD or KV-node tier for DeepSeek.
- Prefix reuse is the snapshot cache on the device (`GR8_DS_CACHE_MIB`, 4,096 MiB a box by default;
  [docs/FIRST-BOOT.md](docs/FIRST-BOOT.md#q2-how-long-can-one-chat-be)).

### 3. No JSON-schema masking - Open

- The `ds41` build has no XGrammar (`scripts/build.sh`), and the engine refuses `--grammar-dir`, so it cannot mask a
  reply to a JSON schema as it does on GLM-5.3.

### 4. Recall at 1M tokens is unreliable - Open

- 2 of 7 different 1M-token prompts found all 3 facts hidden in them. Up to 262,144 tokens, every run that completed
  found all 3 ([docs/FIRST-BOOT.md](docs/FIRST-BOOT.md#q2-how-long-can-one-chat-be)).

### 5. An image request holds the other chats - Open

- The vision tower runs on the step loop's thread, so a request with images holds every other chat's steps while its
  images are encoded ([docs/DS41-VISION.md](docs/DS41-VISION.md#costs-and-limits)). Image input is off by default.
