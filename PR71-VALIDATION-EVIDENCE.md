# PR #71 validation evidence — response_format enforcement + opt-in thinking-off

Branch `missing-api-calls` @ `2bde6f6` (fork `unsaltedbutter-ai/gr8`), based on upstream `main` @ `1f8a84f`.
Validated 2026-10-16 on a 4-node DGX Spark (GB10) ring, TP=4, serving the full GLM-5.3 exlr8 bit338 release
(`f459f5d0272a529403dd9b3fd32f771f4250a0d4`) with the DFlash2 drafter (`425aa615ce320caac34400208b30808c8f14f76c`).

Commit ancestry of the validated branch (all tests below ran against the `e0d7f68` tree, whose PR
content is identical to the current tip `2bde6f6` — the only later change is the strict
count-200 comparison in `scripts/count200_probe.sh`, re-run 2026-10-17):

- `2bde6f6` evidence: eight raw count-200 replies + strict probe log (maintainer request)
- `7618dc5` Front end: forced-disabled thinking as opt-in --allow-thinking-off (amended; originally `e0d7f68`)
- `2d641a9` Front end: enforce response_format json_object / json_schema
- `1f8a84f` upstream `main`

Serving stack identity (recorded per the lab convention):
- engine: docker image `gr8:pr71-e0d7f68`, built from the branch tree (engine test_spec_len 44/44 checks,
  test_prefix_reuse PASS inside the build); engine identity `f250c380874328d7`
- front end: built from the branch tree, md5 `fe9b5c00f5a1e82bcc6c40d40daec7f9` (verified inside the serving container)
- "before" boot: docker image `gr8:main-1f8a84f` built from the `1f8a84f` tree; front md5 `a87331cc0c6589757f296ddc84e49c86`
- `NOHASH=1` was NOT set on any boot

## Phase 0 — checkout

- `missing-api-calls` HEAD at test time = `e0d7f68` (same tree as current tip `2bde6f6` minus the probe-script fix); parents `2d641a9` (response_format), `1f8a84f` (upstream main)
- `git diff --stat origin/main..missing-api-calls`: 14 files, all under `frontend/`, plus
  `scripts/count200_probe.sh` and `smoketest/run.sh`. No engine paths in the PR diff.

## Phase 1 — host-level tests (no GPU)

| suite | command | result |
|---|---|---|
| Rust unit tests | `cargo test --release --manifest-path frontend/g6front-rs/Cargo.toml` | `60 passed; 0 failed; 1 ignored` |
| xgrammar accept/reject (real tokenizer) | `G6_TOKENIZER_DIR=<release> cargo test ... schema_grammar -- --include-ignored` | `1 passed` — `{"a": 3}` accepted, `{"a": "x"}` rejected by the compiled grammar |
| S2 shim | `python3 -m unittest tests.test_s2shim -v` | `Ran 26 tests ... OK` |
| Mock-engine HTTP, flag off | `FRONT=<branch front> tests/serve_mock.sh start` + `URL=... python3 -m unittest tests.test_api` | `Ran 35 tests ... OK (skipped=4)` |
| Mock-engine HTTP, flag on | `FRONT=<branch front> FRONT_ARGS="--allow-thinking-off"` + `G6_ALLOW_THINKING_OFF=1 ...` | `Ran 35 tests ... OK (skipped=2)` — `test_enable_thinking_false_produces_no_reasoning` runs: 200, empty reasoning, real content |

(The two suites ran on the release's tokenizer files with the branch-built Linux front-end binary; the 4/2 skips
are the flag-on cases, matching the expected counts.)

## Phase 3 — boot A: branch build, front end WITH `--allow-thinking-off`

- `./gr8`-equivalent boot (docker compose lab, 4 nodes, ring l/z/m/r); head `/health` →
  `{"ready":true,"steering":{"enabled":false,"forced":false}}`; front-end launch line verified in-container:
  `g6front-rs.real --allow-thinking-off --share-tokens 65472 --model-dir /var/lib/gr8/cache/release/tokenizer ...`
- Smoketest, flag-on half (`OFF=1 ./gr8 glm53 smoke` equivalent): **14/14 passed**, including
  `thinking_off` asserting the 200 / empty `reasoning` / real content contract.
  Log: `/tmp/pr71-smoke-flagon.log` (contents below).

### 3.2 response_format acceptance evidence (real model, not the mock)

`json_object` request (thinking on by default, max_tokens 4096):

```
POST /v1/chat/completions {"model":"glm-5.3","max_tokens":4096,"messages":[{"role":"user",
  "content":"Return the weather in Paris as JSON with keys city and days."}],
  "response_format":{"type":"json_object"}}
finish_reason: stop
content parses as JSON: {"city": "Paris", "days": [], "note": "I don't have access to real-time weather data, ..."}
```

Uncompilable schema refused before the engine (request-time compile):

```
POST /v1/chat/completions {"response_format":{"type":"json_schema","json_schema":{"name":"x",
  "schema":{"type":"not-a-real-type"}}}}
-> 400 {"error":{"message":"response_format json_schema: the schema is not compilable: Unsupported type \"not-a-real-type\"",
        "type":"BadRequestError","param":"response_format","code":400}}
```

(The same json_object request with `max_tokens: 256` hits the runbook's noted trace-budget artifact:
thinking-on Max effort exhausts 256 tokens in reasoning, `finish: length`, content null. This is the
pre-existing thinking-budget behavior, not a response_format regression — with headroom the enforcement works.)

## Phase 4.2 — the maintainer's gate: count 1 to 200, temperature 0, eight runs

**Maintainer review round (2026-10-17).** The maintainer identified a false pass in the gate as
originally shipped: `scripts/count200_probe.sh` stripped non-digits and kept only the first 200
numbers, so a reply of 1..201, or 1..200 followed by prose, also reported PASS. The probe was fixed
in the branch (commit `7618dc5`, amending the original `e0d7f68`; the strict-run raw replies and log
are committed in `evidence/` by `2bde6f6`) to compare the RAW reply byte-exactly
against `seq 1 200` (only trailing whitespace tolerated; a 201st number, prose anywhere, bullets,
blank lines between numbers, CRLF and reordered numbers all fail — self-tested against each of
those shapes before deploying). `COUNT200_RAW_DIR` saves each run's raw reply.

Strict 8-run result, branch boot with `--allow-thinking-off` (fe9b5c00 binary verified in-container,
live cmdline carries the flag), gr8 GLM-5.3 TP=4 ring :8100:

```
run 1: PASS  (lines=200, digit-only=200/200)
run 2: PASS  (lines=200, digit-only=200/200)
run 3: PASS  (lines=200, digit-only=200/200)
run 4: PASS  (lines=200, digit-only=200/200)
run 5: PASS  (lines=200, digit-only=200/200)
run 6: PASS  (lines=200, digit-only=200/200)
run 7: PASS  (lines=200, digit-only=200/200)
run 8: PASS  (lines=200, digit-only=200/200)
8/8 clean
exit=0
```

Raw replies: all 8 runs byte-identical (md5 `dacc241c7db8e74a9ab3d063f3d139e0`, 691 bytes each =
`seq 1 200` minus the final newline). Archives: `evidence/count200-raw-8runs.tar` (all 8, tar md5
`cd8144d27a07a8e39edda5d687a5c1a4`), `evidence/count200-run1-raw.txt` (single reply), strict log
`evidence/pr71-count200-strict.log`.

> The earlier 8/8 PASS recorded 2026-10-16 was measured with the loose matcher and should be
> disregarded in favor of the strict result above. The loose matcher never had to fire: with
> max_tokens=1024 and temperature 0 the model produced the clean sequence on every run, but that is
> now demonstrated rather than assumed.

Historical (loose-matcher) run for completeness:

`scripts/count200_probe.sh http://127.0.0.1:8100 8 glm-5.3`:

```
run 1: PASS
run 2: PASS
run 3: PASS
run 4: PASS
run 5: PASS
run 6: PASS
run 7: PASS
run 8: PASS
8/8 clean
exit=0
```

Raw log: `/tmp/pr71-count200.log`.

## Phase 3.1 — boot C: branch build, front end WITHOUT the flag (flag-off contract)

Recreated the ring with the branch front (`fe9b5c00…`) and NO `--allow-thinking-off`:

```
smoketest glm-5.3 against http://127.0.0.1:8100
  ok   health
  ok   models_lists_served_name  (max_model_len 310514)
  ok   fact
  ok   thinking_on_by_default
  ok   thinking_off            <- asserts the 400 whose message names --allow-thinking-off
  ok   thinking_budget
  ok   stream
  ok   tool_call_parsed
  ok   tool_choice_required    (x2)
  ok   completion
  ok   logprobs
  ok   max_tokens_exact
  ok   context_limit_refused
14/14 passed
```

Log: `/tmp/pr71-smoke-flagoff.log`. Flag-off contract holds: nothing changed silently.

## Phase 5 — the "before" boot: main @ 1f8a84f, same boot conditions

Rebuilt the image from the `1f8a84f` tree (front md5 `a87331cc…`), same compose/boot procedure, no flag:

```
smoketest glm-5.3 against http://127.0.0.1:8100     (main's own smoketest/run.sh)
  ok   health
  ok   models_lists_served_name  (max_model_len 310514)
  ok   fact
  ok   thinking_on_by_default
  ok   thinking_off_refused       <- main's contract: the old "cannot be disabled" 400
  ok   thinking_budget
  ok   stream / tool_call_parsed / tool_choice_required x2 / completion / logprobs / max_tokens_exact / context_limit_refused
14/14 passed
```

Log: `/tmp/pr71-smoke-main.log`.

(First attempt ran the branch's smoketest file against main's binary: 13/14 — the only failure is the
renamed `t_thinking_off` case asserting the branch's 400 wording, which main's binary correctly answers with
the old "cannot be disabled" message. Re-run with main's own `smoketest/run.sh` = 14/14.)

### Gate runs (branch and main, same boot conditions)

`python3 bench/gate/gate.py run --url http://127.0.0.1:8100 --topology tp4 --commit <sha> --boot b1
 --kl-ref bench/gate/refs/tp4.json --health <rank_health.json> --allow-unhealthy --needle-sizes 87000`

Branch (`e0d7f68` tree; see ancestry note at top), 2 runs:

| metric | run 1 | run 2 | spread % |
|---|---|---|---|
| kl.mean | 0.0571117 | 0.0571117 | 0.0 |
| kl.p95 | 0.307589 | 0.307589 | 0.0 |
| kl.top1 | 0.936523 | 0.936523 | 0.0 |
| long.18k.decode_tok_s | 35.86 | 34.36 | 4.27 |
| long.18k.prefill_tok_s | 1336.1 | 1337 | 0.07 |
| long.18k.ttft_s | 13.481 | 13.494 | 0.1 |
| long.87k.decode_tok_s | 27.36 | 26.71 | 2.4 |
| long.87k.prefill_tok_s | 1256.8 | 1255.8 | 0.08 |
| long.87k.ttft_s | 69.197 | 69.306 | 0.16 |
| needle.87k.pass | 1 | 1 | 0.0 |
| pbench.prefill_32768 | 1370.9 | 1370.3 | 0.04 |
| pbench.prefill_8192 | 1358.3 | 1357.8 | 0.04 |
| pbench.prose_decode | 28.04 | 28.77 | 2.57 |
| spec.acceptance | 0.6509 | 0.6433 | 1.17 |
| spec.tokens_per_step | 2.514 | 2.504 | 0.4 |
| work.code.decode_tok_s | 35.9 | 37.59 | 4.6 |
| work.mixed4.agg_tok_s | 57.76 | 57.18 | 1.01 |
| work.mixed4.stream_tok_s | 16.83 | 16.85 | 0.12 |
| work.prose.decode_tok_s | 32.65 | 33.79 | 3.43 |
| work.structured.decode_tok_s | 39.39 | 37.47 | 5.0 |

Main (`1f8a84f`), 2 runs:

| metric | run 1 | run 2 | spread % |
|---|---|---|---|
| kl.mean | 0.0571117 | 0.0571117 | 0.0 |
| kl.p95 | 0.307589 | 0.307589 | 0.0 |
| kl.top1 | 0.936523 | 0.936523 | 0.0 |
| long.18k.decode_tok_s | 30.53 | 34.48 | 12.15 |
| long.18k.prefill_tok_s | 1340.3 | 1339.6 | 0.05 |
| long.18k.ttft_s | 13.47 | 13.387 | 0.62 |
| long.87k.decode_tok_s | 26.69 | 24.58 | 8.23 |
| long.87k.prefill_tok_s | 1258 | 1259.8 | 0.14 |
| long.87k.ttft_s | 69.259 | 69.215 | 0.06 |
| needle.87k.pass | 1 | 1 | 0.0 |
| pbench.prefill_32768 | 1371.6 | 1373.5 | 0.14 |
| pbench.prefill_8192 | 1359.9 | 1361.4 | 0.11 |
| pbench.prose_decode | 28.79 | 27.71 | 3.82 |
| spec.acceptance | 0.6459 | 0.6719 | 3.95 |
| spec.tokens_per_step | 2.473 | 2.561 | 3.5 |
| work.code.decode_tok_s | 37.32 | 37.91 | 1.57 |
| work.mixed4.agg_tok_s | 57.41 | 59.26 | 3.17 |
| work.mixed4.stream_tok_s | 16.57 | 16.57 | 0.0 |
| work.prose.decode_tok_s | 32.55 | 33.99 | 4.33 |
| work.structured.decode_tok_s | 36.85 | 42.7 | 14.71 |

Reading the table (branch vs main, boot-to-boot noise ~3-5% on decode):

- **kl identical to 7 digits** — the PR's decode path is untouched (the reference and the scored build agree
  on top-20 logprobs at the same rate: mean 0.0571, p95 0.308, top-1 0.9365 on both builds).
- **needle.87k.pass = 1 on both** — the 87K needle is found on both builds.
- **prefill (pbench 8K/32K, long 18K/87K) within 0.3%** — no prefill regression.
- **decode within noise** — pbench prose 28.0/28.8 vs 28.8/27.7 (spread 2.6-3.8%); workloads 35.9-39.4 vs
  36.9-42.7 (branch and main overlap within their own boot spreads); long-prompt decode is informational
  (single request, text-dependent — main's own two runs differ by 12%).

Notes on deviations from the runbook (docker lab + ring topology):

1. `gate.py check --baseline` refuses these records because the rank health probe result carries
   `admitted: false`. The refusal is entirely the probe's **RDMA write ring test**, which assumes a
   dual-rail-to-the-same-peer fabric (both rails of a rank face the same neighbor). This cluster's gr8 ring is
   cabled f0→next / f1→prev with per-cable /24s, so the probe's ring script cannot connect (its client targets
   `rails[r+1][0]` with the local device from `rails[r]` — no assignment satisfies both roles on this wiring).
   The probe's per-node measurements themselves are healthy and in-band on all four ranks of the branch boot:
   SM clocks under load 2177–2190 MHz (cap 2200), HBM read 233–238 GB/s, H2D 50.5–54.5 GB/s, proxy CPUs
   online and idle, no wedges. The main boot ran with the identical `--allow-unhealthy` flag, so the
   before/after comparison is apples-to-apples; `gate.py check` output is omitted for both boots for the same
   reason. Raw logs: `/tmp/pr71-gate-branch.log`, `/tmp/pr71-gate-main.log`;
   run records: `bench/gate/runs.jsonl` (2 branch + 2 main records, commit-tagged).
2. The first gate runs hit `--needle-sizes 87000,437000` (the default): the 437K needle sized to 437,675
   prompt tokens against a 310,514-token context cap → HTTP 400. Re-ran with `--needle-sizes 87000`
   (the size the baseline covers: the tp4 baseline records only `needle.87k.pass`). The baseline's
   `missing_from_gate_set` already lists the 437K needle for tp4.
3. `scripts/rank_health.sh` expects the ssh-launcher config (`config/cluster.env`); the lab runs the
   docker compose launcher, so the probe was invoked directly via
   `python3 tools/health/rank_probe.py cluster --hosts lspark,zspark,mspark,rspark --ips <ring rail0> --ips2 <ring rail1> --home <G6_HOME> --proxy-cpu 19 --out <G6_HOME>/out/rank_health.json`,
   and `bin/gpuprobe` was staged from the branch image (`scripts/build.sh` output) per boot.

## Statement for the PR comment

- The flag-off contract holds: with the front end started without `--allow-thinking-off`,
  `chat_template_kwargs.enable_thinking=false` → HTTP 400 whose message names the flag; with the flag,
  the same request → 200, empty `reasoning`/`reasoning_content`, real content (smoketest 14/14 both states).
- The maintainer's gate (count 1–200, temperature 0, eight runs): **8/8 clean**.
- response_format: `json_object` enforced (content is the extracted JSON object);
  `json_schema` compiled at request time — an uncompilable schema is a 400 before the engine.
- Default request paths are byte-identical to main: the Phase 1 suites assert the unchanged renders, and the
  gate's KL numbers are identical to 7 digits across the two builds. Decode path untouched, so no RigMark
  re-run (noted in the PR text per the runbook).

## Raw artifacts (this run)

All raw logs are inlined below (also kept as separate files on the head at `~/gr8-lab/pr71-*`).

### `/tmp/pr71-count200.log` — the maintainer's gate

```
run 1: PASS
run 2: PASS
run 3: PASS
run 4: PASS
run 5: PASS
run 6: PASS
run 7: PASS
run 8: PASS
8/8 clean
```

### `/tmp/pr71-smoke-flagon.log` — branch, `--allow-thinking-off` (flag-on half)

```
smoketest glm-5.3 against http://127.0.0.1:8100
  ok   health  (0.00s)
  ok   models_lists_served_name  (0.00s, max_model_len 310514)
  ok   fact  (2.32s, 72 tok)
  ok   thinking_on_by_default  (0.85s, 30 tok)
  ok   thinking_off  (0.52s)
  ok   thinking_budget  (0.79s, 27 reasoning tok)
  ok   stream  (2.00s)
  ok   tool_call_parsed  (1.32s)
  ok   tool_choice_required  (0.83s, "required")
  ok   tool_choice_required  (0.41s, {"type":"function","function":{"name":"get_weather"}})
  ok   completion  (0.52s)
  ok   logprobs  (0.47s)
  ok   max_tokens_exact  (0.42s)
  ok   context_limit_refused  (0.00s, limit 310514)
14/14 passed
```

### `/tmp/pr71-smoke-flagoff.log` — branch, no flag (flag-off contract: the 400 half)

```
smoketest glm-5.3 against http://127.0.0.1:8100
  ok   health  (0.00s)
  ok   models_lists_served_name  (0.00s, max_model_len 310514)
  ok   fact  (2.29s, 72 tok)
  ok   thinking_on_by_default  (0.83s, 30 tok)
  ok   thinking_off  (0.00s)
  ok   thinking_budget  (0.80s, 27 reasoning tok)
  ok   stream  (2.00s)
  ok   tool_call_parsed  (1.32s)
  ok   tool_choice_required  (0.84s, "required")
  ok   tool_choice_required  (0.41s, {"type":"function","function":{"name":"get_weather"}})
  ok   completion  (0.54s)
  ok   logprobs  (0.47s)
  ok   max_tokens_exact  (0.42s)
  ok   context_limit_refused  (0.00s, limit 310514)
14/14 passed
```

### `/tmp/pr71-smoke-main.log` — main @ 1f8a84f (main's own smoketest)

```
smoketest glm-5.3 against http://127.0.0.1:8100
  ok   health  (0.00s)
  ok   models_lists_served_name  (0.00s, max_model_len 310514)
  ok   fact  (2.26s, 82 tok)
  ok   thinking_on_by_default  (0.82s, 30 tok)
  ok   thinking_off_refused  (0.00s)
  ok   thinking_budget  (0.82s, 27 reasoning tok)
  ok   stream  (2.00s)
  ok   tool_call_parsed  (0.98s)
  ok   tool_choice_required  (0.40s, "required")
  ok   tool_choice_required  (0.40s, {"type":"function","function":{"name":"get_weather"}})
  ok   completion  (0.40s)
  ok   logprobs  (0.19s)
  ok   max_tokens_exact  (0.26s)
  ok   context_limit_refused  (0.00s, limit 310514)
14/14 passed
```

### `/tmp/pr71-rf-json_object.json` — the json_object response body (abridged)

```json
{"id":"chatcmpl-…","object":"chat.completion","model":"glm-5.3",
 "choices":[{"index":0,"message":{"role":"assistant","content":"{\n  \"city\": \"Paris\",\n  \"days\": [],\n  \"note\": \"I don't have access to real-time weather data, so no forecast can be provided.\"\n}"},"finish_reason":"stop"}]}
```
