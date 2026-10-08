# Gate harness (G0)

Measures any gr8 build the same way, and checks it against the baseline of its topology. Python 3 standard library
only; it talks to the front end over HTTP (`/tokenize`, `/v1/completions`, `/v1/chat/completions`, `/metrics`).

```bash
# 1. once per topology, on a build whose commit is recorded: the KL reference (16 sequences x 256 tokens, top 20)
python3 bench/gate/gate.py kl-ref --url http://HEAD:8100 --topology tp6 --commit <sha> --out bench/gate/refs/tp6.json
# 2. per boot: all measurements, twice (the spread between the two is printed); one JSON line per run
python3 bench/gate/gate.py run --url http://HEAD:8100 --topology tp6 --commit <sha> --boot b1 \
    --kl-ref bench/gate/refs/tp6.json --build-flags "G6_NR=6 ..." --engine-env "G6_CP=1 ..."
# 3. after 3 boots of main: the baseline file for the topology
python3 bench/gate/gate.py baseline --topology tp6 --commit <main sha> --out bench/gate/baselines/tp6.json
# 4. a PR build: gates table (markdown), exit 1 on any FAIL
python3 bench/gate/gate.py check --baseline bench/gate/baselines/tp6.json
```

`run` needs `--health FILE`, the rank health probe result of the boot (`scripts/rank_health.sh`, which `launch_tp6.sh` runs and
writes to `$G6_HOME/out/rank_health.json`). It is stored in every record of the run as `rank_health`. A result older than
`--health-max-age-min` (default 360) is refused, and `launch_tp6.sh` deletes the file when `RANK_HEALTH=0`, so a result
from an earlier boot is not reused. A boot the probe refused is not measured unless `--allow-unhealthy`. `--no-health` measures
without a result. `baseline` and `check` refuse any record that has no admitted probe result (`--no-health`,
`--allow-unhealthy` records, or records without the field). See `tools/health/README.md`.

`--only` / `--skip` pick parts: `kl needle long pbench workloads mixed4` (skip `mixed4` on a build without
multi-session). Run records go to `bench/gate/runs.jsonl` (git-ignored; `--runs` changes it).

## What it measures

| part | what | numbers |
|---|---|---|
| `kl` | 16 fixed prompts, greedy 256-token continuation on the reference build; each full sequence is re-scored on the build under test with `prompt_logprobs` (top 20) | KL(ref, new) per position with the tail lumped into one bucket (mean, p95, max), top-1 agreement |
| `needle` | passphrase line at 15% depth of a salted synthetic code document, ~87K and ~437K tokens (sized with the server's tokenizer); 400 answer tokens | pass if the answer contains it |
| `long` | one streamed request each at ~18K and ~87K tokens (cold: every prompt is salted) | TTFT, prefill tok/s, decode tok/s; prefix-cache hits (usage `cached_tokens` and `/metrics`) are recorded and noted |
| `pbench` | `bench/pbench.py`: 8K and 32K cold prefill, prose decode | its summary line |
| `workloads` | code, prose, structured output; 3 runs each of ~300 tokens | decode tok/s (median), drafter counters per workload |
| `mixed4` | four concurrent mixed streams (needs multi-session) | aggregate and per-stream tok/s |
| drafter | from `/metrics` deltas over the workloads | drafts, draft tokens, accepted, acceptance, tokens per step, survival per position |

A run record holds the build commit, topology, build flags, engine environment, all settings (including the seed),
every detailed result, and the flat `metrics` the gates compare.

## Prompt sets of another bench (cross-stack comparisons)

`run --prompt-set FILE` also runs another bench's prompts, given as exact token ids, so g6 and another stack can be
compared on the same text (the harness's own `workloads` use different prompts, and drafter acceptance depends on the
text). The ids go to `/v1/completions` as they are: the other bench's chat template, system message and thinking
setting are kept. One request per prompt, `--prompt-set-tokens` (default 512) at `--prompt-set-temp` (default 0).

```json
{"name": "glm53-bench-1", "prompts": [{"id": "code0", "kind": "code", "prompt_token_ids": [151331, ...]}, ...]}
```

Numbers are named `set.<name>.<kind>.decode_tok_s` and `set.<name>.<kind>.tokens_per_step` (drafter counters over that
kind only), so every number says which set it comes from; `work.*` and `spec.*` are always the harness's own workloads.
The record keeps the file's sha256. The part is off unless `--prompt-set` is given, and it is not in the gate set of
issue #1; a baseline built from runs with it gates these metrics like any other.

## Gates

- **Spread:** `run --repeat 2` (default) runs the suite twice against one boot and prints the spread per metric.
- **Baseline:** per topology, built from the runs of several boots (the repeats of a boot are reduced to their
  median). `bench/gate/baselines/<topology>.json` holds median, min, max and boot-to-boot spread per metric.
- **KL threshold:** null until three boots are in the baseline. Then it is `--kl-margin` (default 2) times the worst boot
  (never below 1e-6). Until then `check` reports `NO THRESHOLD` for `kl.mean` rather than passing or failing it.
  The margin is a default for people to confirm after they see the three-boot spread.
- **Check:** needle passes must be 1; `kl.mean` at or below the threshold; every other gated metric no worse than the
  baseline median by more than `max(--noise, the baseline's boot-to-boot spread)` (`--noise` defaults to 3%, g6's own
  single-boot decode noise). Shown, not gated: KL p95 and top-1, and the long-prompt decode tok/s (one request's decode
  varies with its text; the workloads gate decode).
- **Result line:** `RESULT: PASS` only when the whole gate set of issue #1 (default sizes) is in the baseline and in the run, nothing
  failed, and `kl.mean` had a threshold. Otherwise `FAIL` (a gate failed, or a metric is missing from the run) or
  `PARTIAL` (nothing failed, but part of the set is not in the baseline, or no KL threshold yet). `PARTIAL` exits 1
  unless `--allow-partial`; the table still says PARTIAL.
- **Failed streams:** a workload run or a mixed4 stream that errors or ends with anything but `stop`/`length` leaves no
  speed number for that measurement, so the check reports it `MISSING` (FAIL).
- **Provenance:** `baseline` takes `--commit` and uses only runs of that commit and topology; they must agree on build
  flags, engine environment, measurement settings (sizes, effort, temperature, token counts, pbench arguments) and KL reference, and all
  have the same metrics. `check` refuses (exit 1, "check refused") a run of another topology, other engine environment, other measurement
  settings, or a KL reference other than the baseline's. `run` refuses a `--kl-ref` taken on another topology.

## Notes

- The KL reference must be taken on the same topology, from gr8 `main` at a recorded commit (`--commit` is stored in
  the file). g6's front end needs an engine that returns `prompt_logprobs`; `frontend/g6shim.py` does not today.
- KL treatment of tokens missing from the new build's top 20: they take the new build's 20th logprob (an upper bound),
  so KL is a lower bound for those terms. The tail bucket is everything outside the reference's top 20.
- Tests (no GPU, no model): `python3 -m unittest discover bench/gate/tests`.
