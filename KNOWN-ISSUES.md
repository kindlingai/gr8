# Known issues

Bugs we are still working out, by model. The PR that fixes one removes its entry.

## All models

### 1. `gr8 <model> bench` inside a recipe image stops: no `ssh` - Open ([#413](https://github.com/kindlingai/gr8/issues/413))

- In a recipe image with `PEERS` set, the bench stops before its first stage with `FileNotFoundError: ... 'ssh'`, and
  no report is written. The `gr8` wrapper passes `--hosts` with the `PEERS` list to `bench/bench.py`, and
  `bench/gr8bench/thermal.py` reaches every host but `local` through `ssh`, which the image does not have.
- Workaround: add `--hosts ''` after the command (`bench.py` keeps the last `--hosts`). The run then has no cooldown
  between stages and no heat samples.

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

### 4. A prefix-reuse hit can differ from a cold prefill at temperature 0 (TP=4, four Sparks) - Open

- **Symptom:** with the default reuse rules (`GR8_REUSE_ALIGN=1`), turn 2 of a chat that reuses turn 1's KV from the pool
  returns different text at temperature 0 than the same request on a freshly booted engine. In the report, 4 chats of two
  turns each (a system prompt of about 3,000 tokens each; turn 1 5,334 tokens, turn 2 5,408 tokens): turn 2 after a hit
  differed from a cold boot on 2 of 4 chats. Two cold boots agreed with each other on 4 of 4, so the cold side is
  deterministic.
- **Docs:** until this is fixed, read "a hit equals a cold prefill" in [docs/SETTINGS.md](docs/SETTINGS.md) and
  [docs/REUSE-GPU-TEST.md](docs/REUSE-GPU-TEST.md) as the intended behaviour, not a measured one.
- **Not narrowed down:** whether the difference comes from the retained rows [0, 4096) or from the re-prefilled suffix
  [4096, 5408) (for example the drafter state, or a suffix chunk shape that differs from the cold one).
- Tracking: [#412](https://github.com/kindlingai/gr8/issues/412).

## GLM-5.3-Flash

No open bugs recorded. [docs/FLASH.md](docs/FLASH.md#not-yet) lists what Flash does not do yet.

## DeepSeek-V4.1-Flash

### 1. The `./gr8 ds41` route has not been run end to end from a clean machine - Open

- [docs/DEEPSEEK-BOOT.md](docs/DEEPSEEK-BOOT.md) was checked against the launcher source. Our own boots, and every
  number in [docs/DEEPSEEK.md](docs/DEEPSEEK.md) and [docs/FIRST-BOOT.md](docs/FIRST-BOOT.md), went through a separate
  test harness. So no boot time through `./gr8 ds41 up` is measured yet.
- Setup has two manual steps GLM does not: the whole 510.6 GB release on every box, and the Engram token map.

### 2. KV store below the device - Implemented; live validation pending

- DS41 accepts `--kv-store` / `KV_STORE` for SSD, RAM and remote block stores, alongside the device snapshot cache.
  Full pages are shared across continuation snapshots; each boundary carries its rolling state and partial page.
- CPU tests cover admission, exact state copies, steering isolation, delta writes and persistent reload. CUDA object
  compilation is checked without executing a GPU. Cold/warm output equality and performance on the served topology
  still need a live gate; no speedup is claimed. See [session stores](docs/DEEPSEEK.md#tiered-session-stores).

### 3. No JSON-schema masking - Open

- The `ds41` build has no XGrammar (`scripts/build.sh`), and the engine refuses `--grammar-dir`, so it cannot mask a
  reply to a JSON schema as it does on GLM-5.3.

### 4. Recall at 1M tokens is unreliable - Open

- 2 of 7 different 1M-token prompts found all 3 facts hidden in them. Up to 262,144 tokens, every run that completed
  found all 3 ([docs/FIRST-BOOT.md](docs/FIRST-BOOT.md#q2-how-long-can-one-chat-be)).

### 5. An image request holds the other chats - Open

- The vision tower runs on the step loop's thread, so a request with images holds every other chat's steps while its
  images are encoded ([docs/DS41-VISION.md](docs/DS41-VISION.md#costs-and-limits)). Image input is off by default.
