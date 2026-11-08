# Paired sweep measurement order

The generic driver builds variants in parallel but launches benchmark processes
**serially**. `--rounds` (default 3; minimum 3) is rounded up to a complete cyclic
Latin block over candidates: each candidate occupies every position equally.
The next block reverses the candidate list. For two candidates the default is
four rounds: A B / B A / B A / A B. Every candidate has its own
`off-before / candidate / off-after` bracket; no controls are reused.
`off` here means the configured `--reference`, not a kernel-specific name.

## Records and uncertainty

* `results.jsonl` records **every attempt**, including retryable low-memory
  failures. `round_id`, `pair_id`, `run_id`, `role`, `attempt`, `final_attempt`
  identify runs. Per-attempt stdout/stderr never overwrite each other. Legacy
  `<variant>.<case>.tryN` files alias only the first occurrence for compatibility.
* Binary JSON is preserved in `binary_output`, including `samples_us`, input
  slots, correctness checks and in-process pair/null samples. Driver IDs cannot
  be overwritten by binary output.
* `pairs.jsonl` records each bracket. When available, `paired_delta_pct_mean`
  is the observation used for primary uncertainty: one **in-process paired mean
  per process/round**, NOT a ratio of separately run off means. `reference_mean_us`
  is required with this field. External brackets remain drift diagnostics.
  Legacy binaries instead get an explicitly labelled separate-process bracket
  fallback, using `(off_before + off_after) / 2`. Estimators never mix.
* Negative delta means faster. Summary `n_pairs` counts successful independent
  process-pair observations, not CUDA repetitions; `n_pairs_requested` exposes
  missing pairs. `paired_sd_pct` is between-pair sample SD; `paired_ci95_*_pct`
  is a two-sided Student-t 95% CI of the mean (conservative df=30 above 30 df).
  The CI assumes independent pair observations; serial autocorrelation can make
  it optimistic. `sd_us` remains mean within-process repetition SD for output
  compatibility, **not** between-process uncertainty.
* A failed final attempt, bad control, mixed regime, mixed estimator or invalid
  timing leaves the affected variant incomplete, with no valid prompt total.
  Failed low-memory attempts followed by success remain in raw records but are
  not observations. Mismatch timings in legacy `write_tables` inputs remain
  explicitly MISMATCH diagnostics, never EXACT totals.
* Totals are weighted kernel-only estimates, **not TTFT**, and their percentages
  are aggregate ratios, **not paired uncertainty or a model keep/drop gate**.

## GPU state and regimes

Before and after **each subprocess** the driver takes `nvidia-smi -q -x`
point snapshots: measured clocks, temperature, power, throttle/event reasons,
and competing compute/graphics processes. Raw XML, timestamps, errors and parsed
fields are stored in each record. This is outside CUDA timing and does not
monitor or control clocks during a timed interval. Unsupported fields and absent
telemetry are explicitly PARTIAL/UNAVAILABLE, never fabricated. All visible
physical GPUs are recorded alongside `CUDA_VISIBLE_DEVICES`; no guessed GPU
ordinal remapping is applied. Contention or telemetry gaps require inspection;
EXACT says output correctness, not an uncontended or representative measurement.

`--flush-mb` (0), `--pools` (3), and `--amortize` (1) are forwarded to the
benchmark binary. Actual `timing_regime` (or `regime`) from binary output wins;
otherwise `input_slots > 1` labels rotating-input-refresh with coldness and
predecessor unverified. Legacy missing slots labels warm/repeated-address with
cache residency unverified. Neither rotating addresses nor an L2 flush proves
an engine-like cache state. Captured/predecessor replay is **unavailable unless
explicitly reported by the adapter**; this driver does not implement one.
Synthetic rows are screening results, not evidence for locality-sensitive model
keep/drop decisions. No live GPU benchmark is needed for the CPU driver tests:

```
python3 -m unittest discover -s engine/attn/sweep -p 'test_*.py' -v
```
