#!/usr/bin/env python3
"""Reusable isolated CUDA variant builder/runner. All kernel knowledge is in adapters."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import statistics
import shlex
import shutil
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def ptxas_resources(log, key):
    """Never conflate helper-kernel resources with the timed main kernel."""
    entries = {}
    current = None
    for line in log.splitlines():
        match = re.search(r"Function properties for (\S+)", line)
        if match:
            current = match[1]
            entries[current] = {}
        if current:
            match = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if match:
                entries[current].update(zip(("stack_bytes", "spill_store_bytes", "spill_load_bytes"), map(int, match.groups())))
            match = re.search(r"Used (\d+) registers", line)
            if match:
                entries[current]["ptxas_regs"] = int(match[1])
    matches = [v for name, v in entries.items() if key and key in name]
    return matches[0] if len(matches) == 1 else {}


def validate_cases(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("case list must be a nonempty JSON array")
    seen = set()
    for case in rows:
        if not isinstance(case, dict) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case.get("id", "")) or case["id"] in seen:
            raise ValueError("case IDs must be unique safe names")
        seen.add(case["id"])
        if not isinstance(case.get("args"), list) or not all(isinstance(v, (str, int, float)) for v in case["args"]):
            raise ValueError("case args must be a list of scalar strings/numbers")
        if not isinstance(case.get("weights", {}), dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in case.get("weights", {}).values()):
            raise ValueError("case weights must be nonnegative numbers")
    return rows


def command(cmd, timeout):
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        def txt(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        return 124, txt(e.stdout), txt(e.stderr) + "\nTIMEOUT\n"
    except OSError as e:
        return 127, "", str(e)


def balanced_orders(variants, reference, rounds):
    """Complete cyclic Latin blocks: each candidate occupies every position equally."""
    candidates = [v for v in variants if v != reference] or [reference]
    size = len(candidates)
    count = ((rounds + size - 1) // size) * size
    for r in range(count):
        block = candidates if (r // size) % 2 == 0 else candidates[::-1]
        offset = r % size
        yield block[offset:] + block[:offset]


def telemetry_snapshot():
    """Point snapshots only, outside the child (and therefore its CUDA intervals).

    XML includes graphics as well as compute processes, unlike compute-apps queries.
    Preserve the raw response for driver-specific unsupported fields and audit.
    """
    started = time.time()
    cmd = ["nvidia-smi", "-q", "-x"]
    rc, stdout, stderr = command(cmd, 5)
    result = {"started_unix": started, "finished_unix": time.time(), "command": cmd,
              "returncode": rc, "stdout": stdout, "stderr": stderr, "status": "UNAVAILABLE",
              "scope": "all visible physical GPUs; point snapshots, not in-interval monitoring",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if rc:
        return result
    try:
        root = ET.fromstring(stdout)
        gpus = []
        for gpu in root.findall("gpu"):
            def section(*names):
                for name in names:
                    node = gpu.find(name)
                    if node is not None:
                        return {e.tag: e.text for e in node.iter() if len(e) == 0}
                return None
            processes = gpu.find("processes")
            gpus.append({"id": gpu.get("id"), "uuid": gpu.findtext("uuid"),
                         "clocks": section("clocks"), "temperature": section("temperature"),
                         "power": section("gpu_power_readings", "power_readings"),
                         "throttle": section("clocks_event_reasons", "clocks_throttle_reasons"),
                         "processes": None if processes is None or (processes.text or "").strip() == "N/A" else
                         [{e.tag: e.text for e in proc} for proc in processes.findall("process_info")]})
        result["gpus"] = gpus
        result["status"] = "AVAILABLE" if gpus and all(
            g[k] is not None and (k == "processes" or any(v and v not in ("N/A", "Not Supported") for v in g[k].values()))
            for g in gpus for k in ("clocks", "temperature", "power", "throttle", "processes")) else "PARTIAL"
    except ET.ParseError as e:
        result["parse_error"] = str(e)
    return result


def timing_regime(data):
    # Labels are observations/reports, not claims of model realism or cache residency.
    reported = data.get("timing_regime", data.get("regime"))
    if reported is not None:
        return str(reported)
    slots = data.get("input_slots", 1)
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        return "unavailable (invalid input_slots)"
    if slots > 1:
        return "rotating-input-refresh (coldness/predecessor unverified)"
    return "warm/repeated-address (cache residency unverified)"


def uncertainty(values):
    """Between-pair sample SD and two-sided Student-t 95% CI, not CUDA rep SD.

    For >30 df use the next lower tabulated df (conservative critical value).
    Serial drift/autocorrelation can invalidate the independent-pair CI assumption.
    """
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    critical = [0, 12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912,
                2.364624, 2.306005, 2.262157, 2.228139, 2.200986, 2.178813,
                2.160369, 2.144787, 2.131450, 2.119905, 2.109816, 2.100923,
                2.093024, 2.085963, 2.079614, 2.073873, 2.068658, 2.063899,
                2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042273]
    sd = statistics.stdev(values)
    half = critical[min(n - 1, 30)] * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean-half, "ci95_high": mean+half}


def valid_measurement(row):
    return (row.get("status") == "EXACT" and row.get("returncode", 0) == 0 and
            row.get("launch_valid", True) in (True, 1) and row.get("integrity_errors", 0) == 0 and
            row.get("mismatch_bytes", 0) == 0 and all(
        isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool) and
        math.isfinite(row[k]) and row[k] >= (0 if k == "sd_us" else 1e-300)
        for k in ("mean_us", "sd_us")))


def pair_record(before, candidate, after):
    record = {k: candidate[k] for k in ("kernel", "case", "variant", "round_id", "pair_id")}
    record.update(status="INVALID_PAIR", run_ids=[r["run_id"] for r in (before, candidate, after)])
    if not all(valid_measurement(r) for r in (before, candidate, after)):
        return record
    regimes = {timing_regime(r) for r in (before, candidate, after)}
    if len(regimes) != 1:
        record["reason"] = "regime mismatch"
        return record
    control = (before["mean_us"] + after["mean_us"]) / 2
    record.update(regime=regimes.pop(), bracket_reference_us=control,
                  bracket_delta_pct=100 * (candidate["mean_us"] / control - 1),
                  control_drift_pct=100 * (after["mean_us"] / before["mean_us"] - 1))
    if "paired_delta_pct_mean" in candidate:
        pct = candidate["paired_delta_pct_mean"]
        ref = candidate.get("reference_mean_us")
        if not isinstance(pct, (int, float)) or not math.isfinite(pct) or not isinstance(ref, (int, float)) or not math.isfinite(ref) or ref <= 0:
            record["reason"] = "invalid in-process paired fields"
            return record
        record.update(estimator="in-process paired mean; between-process uncertainty",
                      delta_pct=pct, delta_us=candidate["mean_us"] - ref, reference_us=ref)
    else:
        record.update(estimator="separate-process bracket fallback; between-pair uncertainty",
                      delta_pct=record["bracket_delta_pct"], delta_us=candidate["mean_us"] - control,
                      reference_us=control)
    record.update(status="EXACT", candidate_us=candidate["mean_us"])
    return record


def adaptive_decision(raw, pairs, case_id, variant, target):
    """Repeated-look heuristic, not a sequentially valid confidence guarantee.

    Any observed out-of-band null is sticky: it prevents early stopping even
    if later controls settle. Missing nulls or invalid pairs also fail closed.
    """
    ps = [p for p in pairs if p["case"] == case_id and p["variant"] == variant]
    used = len({p["round_id"] for p in ps})
    good = [p for p in ps if p["status"] == "EXACT"]
    stats = uncertainty([p["delta_pct"] for p in good])
    half = None if stats["ci95_high"] is None else (stats["ci95_high"] - stats["ci95_low"]) / 2
    ids = {rid for p in ps for rid in p.get("run_ids", [])}
    runs = [r for r in raw if r.get("final_attempt") and r.get("run_id") in ids]
    null_ok, outside = bool(runs), False
    for r in runs:
        value, band = r.get("null_delta_pct_mean"), r.get("null_band_pct")
        valid = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (value, band)) and band >= 0
        null_ok &= valid
        outside |= valid and abs(value) > band
    valid = len(good) == len(ps) and len({p.get("estimator") for p in good}) == 1 and len({p.get("regime") for p in good}) == 1
    precise = valid and used >= 3 and half is not None and half < target
    stop = precise and null_ok and not outside
    if stop:
        reason = "CI_REACHED" if used < 6 else "CI_REACHED_AT_CAP"
    elif used < 6:
        reason = "CONTINUE"
    elif outside:
        reason = "MAX_ROUNDS_NULL_OUTSIDE"
    elif not null_ok:
        reason = "MAX_ROUNDS_NULL_UNAVAILABLE"
    elif not valid:
        reason = "MAX_ROUNDS_INVALID"
    else:
        reason = "MAX_ROUNDS_CI_WIDE"
    return dict(rounds_used=used, final_ci95_half_width_pct=half,
                adaptive_stop_reason=reason, adaptive_ci_target_pct=target,
                adaptive_null_verdict="OUTSIDE_BAND" if outside else "WITHIN_BAND" if null_ok else "UNAVAILABLE",
                stop=stop or used >= 6)


def aggregate_runs(raw, pairs, cases, variants, reference, adaptive_ci=0):
    """Only complete requested runs yield valid totals; retain failed pairs separately."""
    final = [r for r in raw if r["final_attempt"]]
    summaries = []
    for case in cases:
        for v in variants:
            runs = [r for r in final if r["case"] == case["id"] and r["variant"] == v]
            if not runs:
                continue
            row = {"kernel": runs[0]["kernel"], "case": case["id"], "variant": v,
                   "status": "INCOMPLETE", "null_band_verdict": "INCOMPLETE", "n_runs": len(runs),
                   "n_valid_runs": sum(valid_measurement(r) for r in runs)}
            if all(r["status"] == "BUILD_FAILED" for r in runs):
                row["status"] = "BUILD_FAILED"
            ps = [r for r in pairs if r["case"] == case["id"] and r["variant"] == v]
            good = [r for r in ps if r["status"] == "EXACT"]
            row.update(n_pairs=len(good), n_pairs_requested=len(ps))
            regimes = {timing_regime(r) for r in runs}
            estimators = {r["estimator"] for r in good}
            if all(valid_measurement(r) for r in runs) and len(regimes) == 1 and (v == reference or len(good) == len(ps)) and len(estimators) <= 1:
                row.update(status="EXACT", mean_us=statistics.mean(r["mean_us"] for r in runs),
                           # Preserve sd_us as mean within-process rep SD, clearly labelled.
                           sd_us=statistics.mean(r["sd_us"] for r in runs), regime=regimes.pop(),
                           between_run_sd_us=statistics.stdev(r["mean_us"] for r in runs) if len(runs)>1 else None)
                for key in ("regs", "ptxas_regs", "stack_bytes", "spill_store_bytes", "spill_load_bytes", "dynamic_smem", "local_bytes", "mismatch_bytes"):
                    if key in runs[0] and all(r.get(key) == runs[0][key] for r in runs):
                        row[key] = runs[0][key]
                row["null_band_verdict"] = "UNAVAILABLE"
                nulls = [r.get("null_delta_pct_mean") for r in runs]
                if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in nulls):
                    row["null_delta_pct_mean"] = statistics.mean(nulls)
                    row["null_max_abs_delta_pct"] = max(map(abs, nulls))
                bands = [r.get("null_band_pct") for r in runs]
                if all(isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) and b >= 0 and b == bands[0] for b in bands):
                    row["null_band_pct"] = bands[0]
                    if "null_max_abs_delta_pct" in row:
                        # Compare EVERY process mean, not just their cancellable average.
                        row["null_band_verdict"] = "WITHIN_BAND" if row["null_max_abs_delta_pct"] <= bands[0] else "OUTSIDE_BAND"
                if good:
                    stats = uncertainty([r["delta_pct"] for r in good])
                    row.update(vs_reference_pct=stats["mean"], paired_delta_pct=stats["mean"],
                               paired_sd_pct=stats["sd"], paired_ci95_low_pct=stats["ci95_low"],
                               paired_ci95_high_pct=stats["ci95_high"], pairing_estimator=next(iter(estimators)),
                               paired_reference_us=statistics.mean(r["reference_us"] for r in good),
                               paired_delta_us=statistics.mean(r["delta_us"] for r in good))
            row["rounds_used"] = len({r.get("round_id") for r in runs})
            low, high = row.get("paired_ci95_low_pct"), row.get("paired_ci95_high_pct")
            row["final_ci95_half_width_pct"] = None if low is None or high is None else (high-low)/2
            if adaptive_ci and ps:
                decision = adaptive_decision(raw, pairs, case["id"], v, adaptive_ci)
                row.update({k: value for k, value in decision.items() if k != "stop"})
            else:
                row["adaptive_stop_reason"] = "REFERENCE_CONTROLS" if adaptive_ci else "DISABLED"
            summaries.append(row)
    return summaries


def reduced_coverage(cases):
    tagged = [c for c in cases if "reduced_set" in c]
    if not tagged:
        return None
    metadata = tagged[0]["reduced_set"]
    if len(tagged) != len(cases) or any(c["reduced_set"] != metadata for c in cases):
        raise ValueError("reduced cases must share one baseline coverage definition")
    totals = metadata.get("full_off_ms", {})
    if not totals or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in totals.values()):
        raise ValueError("invalid reduced full baseline denominator")
    if any(not isinstance(c.get("baseline_off_mean_us"), (float, int)) or not math.isfinite(c["baseline_off_mean_us"]) or c["baseline_off_mean_us"] <= 0 for c in cases):
        raise ValueError("missing reduced baseline case timing")
    return {label:100*sum(c["baseline_off_mean_us"]*c.get("weights", {}).get(label, 0)/1000 for c in cases)/full
            for label, full in totals.items()}


def write_tables(out, rows, cases, variants, reference, kernel):
    coverage = reduced_coverage(cases)
    baseline = {r["case"]: r for r in rows if r["variant"] == reference and "mean_us" in r}
    weights = {c["id"]: c.get("weights", {}) for c in cases}
    labels = sorted({k for c in cases for k in c.get("weights", {})}, key=lambda x: sum(c.get("weights", {}).get(x, 0) for c in cases))
    fields = ["kernel", "variant", "case", "status", "mean_us", "sd_us", "vs_reference_pct", "mismatch_bytes", "regs", "ptxas_regs",
              "stack_bytes", "spill_store_bytes", "spill_load_bytes", "dynamic_smem", "local_bytes",
              "regime", "n_runs", "n_valid_runs", "n_pairs", "n_pairs_requested", "between_run_sd_us",
              "pairing_estimator", "paired_delta_us", "paired_delta_pct", "paired_sd_pct",
              "paired_ci95_low_pct", "paired_ci95_high_pct", "null_delta_pct_mean",
              "null_max_abs_delta_pct", "null_band_pct", "null_band_verdict",
              "rounds_used", "final_ci95_half_width_pct", "adaptive_ci_target_pct", "adaptive_stop_reason", "adaptive_null_verdict", "scope"]
    for label in labels:
        fields.extend([f"calls_{label}", f"contribution_{label}_ms"])
        if coverage is not None:
            fields.append(f"off_weight_coverage_{label}_pct")
    for r in rows:
        base = baseline.get(r["case"])
        if "vs_reference_pct" not in r and "mean_us" in r and base and base["mean_us"] > 0:
            r["vs_reference_pct"] = 100 * (r["mean_us"] / base["mean_us"] - 1)
        for label in labels:
            w = weights[r["case"]].get(label, 0)
            r[f"calls_{label}"] = w
            if "mean_us" in r:
                r[f"contribution_{label}_ms"] = r["mean_us"] * w / 1000
    output_rows = list(rows)
    totals = {}
    for variant in variants:
        vr = {r["case"]: r for r in rows if r["variant"] == variant}
        for label in labels:
            needed = [c for c in cases if c.get("weights", {}).get(label, 0)]
            if not needed:
                continue
            total = {"kernel": kernel, "variant": variant, "case": f"TOTAL_{label}_" + ("reduced_set_" if coverage is not None else "") + "kernel_only",
                     "prompt_label": label, "scope": "reduced_set_kernel_only" if coverage is not None else "kernel_only",
                     "status": "INCOMPLETE", f"calls_{label}": sum(c["weights"][label] for c in needed)}
            if all(c["id"] in vr and "mean_us" in vr[c["id"]] for c in needed):
                total[f"contribution_{label}_ms"] = sum(vr[c["id"]]["mean_us"] * c["weights"][label] / 1000 for c in needed)
                total["status"] = "EXACT" if all(vr[c["id"]]["status"] == "EXACT" for c in needed) else "MISMATCH"
                totals[(variant, label)] = total[f"contribution_{label}_ms"]
            if coverage is not None:
                total[f"off_weight_coverage_{label}_pct"] = coverage.get(label)
            output_rows.append(total)
    for r in output_rows[len(rows):]:
        label = r["prompt_label"]
        if (r["variant"], label) in totals and (reference, label) in totals:
            r["vs_reference_pct"] = 100 * (totals[r["variant"], label] / totals[reference, label] - 1)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(output_rows)
    def cell(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v is not None else "UNKNOWN"
    md = [f"# {kernel} kernel sweep", "", f"Reference: `{reference}`. CUDA event batch means; sd_us is mean within-process repetition SD, NOT uncertainty.",
          "paired_sd_pct and paired_ci95_* are BETWEEN-process/pair SD and Student-t 95% CI; n_pairs is the independent observation count, not CUDA repetitions.",
          "null_band_verdict compares every process off/off null mean to the common reported band; a grand-mean cancellation cannot pass. Missing null data is UNAVAILABLE, failed rows INCOMPLETE; this is not a model keep/drop verdict.",
          "CI assumes independent pair observations; autocorrelation may understate uncertainty. See raw telemetry and controls for drift/competition.",
          "Adaptive stopping, when enabled, uses repeated nominal CI95 looks (min3/max6); not a sequentially valid coverage guarantee. Candidate dropout truncates position balance; fixed mode is unchanged.",
          "Prompt totals are weighted kernel-only estimates, NOT TTFT. No rank multiplier. Total percentage is an aggregate ratio, not paired uncertainty.",
          "Cold/rotating synthetic inputs are not an engine predecessor replay; captured regime is unavailable unless explicitly reported by the adapter.",
          "Resource columns describe the main kernel; timing includes every kernel in launch().", "",
          "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    if coverage is not None:
        note = "reduced set: " + " / ".join(f"{pct:.4f} % of {label}" for label, pct in coverage.items()) + " off weight"
        metadata = cases[0]["reduced_set"]
        md[2:2] = [note + ". Archived OFF-time contribution, NOT raw call fraction; NOT full prompt/TTFT.",
                   "Coverage provenance: " + json.dumps(metadata, sort_keys=True), ""]
    md.extend("| " + " | ".join(cell(r.get(k)) for k in fields) + " |" for r in output_rows)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return output_rows


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(paths):
    """Keep original absolute names: run-only never reattributes these to its checkout."""
    return {str(p.resolve()): sha256_file(p) if p.is_file() else None for p in sorted(set(paths))}


def build_dependencies(depfile):
    """Read nvcc's one-target make depfile (escaped spaces and line continuations)."""
    if not depfile.is_file():
        return []
    value = depfile.read_text().replace("\\\n", " ")
    if ":" not in value:
        return []
    return [Path(p) for p in shlex.split(value.split(":", 1)[1])]


def archive_run(out, destination):
    if destination is not None:
        # Never merge into an old archive, recurse into ourselves or remove files.
        target = destination.resolve()
        if target == out or target.is_relative_to(out):
            raise ValueError("archive destination must be outside the run directory")
        shutil.copytree(out, target)
        print(f"ARCHIVE {target}", flush=True)


def shard_cases(cases, specification):
    if specification is None:
        return cases, None
    match = re.fullmatch(r"(\d+)/(\d+)", specification)
    if not match:
        raise ValueError("shard must be zero-based i/N")
    index, count = map(int, match.groups())
    if not 0 <= index < count or count > len(cases):
        raise ValueError("shard needs 0<=i<N<=case count (no empty shards)")
    return cases[index::count], dict(index=index, count=count)


def cases_hash(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()


def merge_runs(directories, out):
    """Merge DISJOINT case shards; never average duplicate executions or hosts."""
    directories = [Path(d).resolve() for d in directories]
    out = Path(out).resolve()
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("merge needs distinct run directories")
    if out.exists() or any(out == d or out.is_relative_to(d) for d in directories):
        raise ValueError("merge output must be new and outside input directories")
    records = []
    compatible = ("kernel", "adapter", "reference", "variants", "arch", "seed", "warmup", "reps", "iters",
                  "rounds_requested", "orders", "flush_mb", "pools", "amortize", "adaptive_ci", "adaptive_max_rounds", "binary_sha256", "driver_sha256")
    for directory in directories:
        manifest = json.loads((directory/"run-manifest.json").read_text())
        if not manifest.get("finished_unix"):
            raise ValueError("cannot merge unfinished run: " + str(directory))
        if sha256_file(directory/"build-manifest.json") != manifest.get("build_manifest_sha256"):
            raise ValueError("shard build provenance hash mismatch")
        for v, expected_hash in manifest.get("binary_sha256", {}).items():
            if expected_hash is not None and (not (directory/v).is_file() or sha256_file(directory/v) != expected_hash):
                raise ValueError("shard archived binary hash mismatch")
        cases = validate_cases(json.loads((directory/"cases.json").read_text()))
        full = validate_cases(json.loads((directory/"all-cases.json").read_text()))
        shard = manifest.get("shard") or dict(index=0, count=1)
        expected, _ = shard_cases(full, f"{shard['index']}/{shard['count']}")
        if cases != expected or manifest.get("case_sha256") != cases_hash(cases) or manifest.get("all_cases_sha256") != cases_hash(full):
            raise ValueError("shard case identity mismatch")
        if records and (full != records[0][2] or any(manifest.get(k) != records[0][1].get(k) for k in compatible)):
            raise ValueError("incompatible shard settings/binaries/case list/driver")
        raw = [json.loads(line) for line in (directory/"results.jsonl").read_text().splitlines()]
        pairs = [json.loads(line) for line in (directory/"pairs.jsonl").read_text().splitlines()]
        ids = {c["id"] for c in cases}
        variants = manifest["variants"]
        candidates = [v for v in variants if v != manifest["reference"]] or [manifest["reference"]]
        if not raw or not pairs or any(r["case"] not in ids or r["variant"] not in variants for r in raw+pairs):
            raise ValueError("raw records outside shard or empty")
        if {(r["case"],r["variant"]) for r in pairs} != {(c,v) for c in ids for v in candidates}:
            raise ValueError("missing case/candidate pairs")
        for case_id in ids:
            for v in candidates:
                cohort = [p for p in pairs if p["case"] == case_id and p["variant"] == v]
                round_ids = sorted(p["round_id"] for p in cohort)
                if round_ids != list(range(1,len(cohort)+1)):
                    raise ValueError("missing/duplicate pair rounds")
                if manifest.get("adaptive_ci"):
                    if not 3 <= len(cohort) <= 6 or not adaptive_decision(raw,pairs,case_id,v,manifest["adaptive_ci"])["stop"]:
                        raise ValueError("incomplete adaptive pair cohort")
                elif len(cohort) != len(manifest["orders"]):
                    raise ValueError("incomplete fixed pair cohort")
        final_ids = {r["run_id"] for r in raw if r["final_attempt"]}
        if any(rid not in final_ids for pair in pairs for rid in pair["run_ids"]):
            raise ValueError("pair has missing final raw run")
        if len(final_ids) != sum(bool(r["final_attempt"]) for r in raw):
            raise ValueError("duplicate final raw run IDs")
        records.append((directory, manifest, full, shard, raw, pairs))
    counts = {r[3]["count"] for r in records}
    if len(counts) != 1 or len(records) != next(iter(counts)) or {r[3]["index"] for r in records} != set(range(len(records))):
        raise ValueError("need every shard index exactly once")
    records.sort(key=lambda r:r[3]["index"])
    manifest = records[0][1];cases = records[0][2]
    case_order = {c["id"]:i for i,c in enumerate(cases)}
    raw = [r for record in records for r in record[4]]
    pairs = [r for record in records for r in record[5]]
    # Stable case order makes the reconstructed table identical for identical observations.
    raw.sort(key=lambda r:case_order[r["case"]]);pairs.sort(key=lambda r:case_order[r["case"]])
    summary = aggregate_runs(raw,pairs,cases,manifest["variants"],manifest["reference"],manifest.get("adaptive_ci",0))
    out.mkdir(parents=True)
    sources=[]
    for directory, m, full, shard, _, _ in records:
        dest=out/"shards"/str(shard["index"])
        shutil.copytree(directory,dest)
        sources.append(dict(directory=str(directory), archived=str(dest.relative_to(out)), shard=shard,
                            hashes={name:sha256_file(directory/name) for name in
                                    ("run-manifest.json","build-manifest.json","cases.json","all-cases.json","results.jsonl","pairs.jsonl")}))
    for name, data in (("results.jsonl",raw),("pairs.jsonl",pairs)):
        (out/name).write_text("".join(json.dumps(r)+"\n" for r in data))
    (out/"cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    merged=dict(kind="case_shard_merge", sources=sources, all_cases_sha256=cases_hash(cases),
                settings={k:manifest.get(k) for k in compatible}, merged_unix=time.time(), merger_driver_sha256=sha256_file(HERE/"run.py"),
                note="Disjoint case observations from multiple hosts; not cross-host replicates or extra independent rounds. Raw file paths resolve within each archived source shard.")
    (out/"merge-manifest.json").write_text(json.dumps(merged,indent=2)+"\n")
    write_tables(out,summary,cases,manifest["variants"],manifest["reference"],manifest["kernel"])
    return int(any(r["status"] != "EXACT" for r in summary) or any(r["status"] != "EXACT" for r in pairs))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", type=Path, help="adapter JSON (default attention.json; run-only uses the build bundle)")
    p.add_argument("--kernel-name")
    p.add_argument("--cases", type=Path, help="opaque adapter case-list JSON")
    p.add_argument("--shard", help="zero-based case shard i/N; run stage only; build bundles retain the complete list")
    p.add_argument("--variants-dir", type=Path)
    p.add_argument("--reference", help="reference variant (default off, or bundle reference for run-only)")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--arch", choices=("120a", "121a"), required=True)
    p.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    p.add_argument("--ccbin", help="nvcc host compiler executable or directory")
    p.add_argument("--build-cmd", default="", help="shlex-split command prefix, e.g. a compiler container wrapper; never a shell")
    p.add_argument("--archive-to", type=Path, help="copy the COMPLETE bundle/run to a new directory, without deletion")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suite", default="all")
    p.add_argument("--late-final", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--rounds", type=int, default=3, help="minimum process rounds (>=3), rounded up to complete balanced blocks")
    p.add_argument("--adaptive-ci", type=float, default=0, help="CI95 half-width target in percentage points; default off; min3/max6 rounds")
    p.add_argument("--flush-mb", type=int, default=0)
    p.add_argument("--pools", type=int, default=3)
    p.add_argument("--amortize", type=int, default=1)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--retries", type=int, default=3, help="total low-memory attempts")
    p.add_argument("--retry-delay", type=float, default=5)
    p.add_argument("--timeout", type=float, default=180, help="per case seconds")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--run-only", action="store_true")
    a = p.parse_args(argv)
    if a.build_only and a.shard:
        p.error("shard is a run-stage option; build the full case bundle once")
    if a.build_only and a.run_only:
        p.error("build-only and run-only are mutually exclusive")
    if not math.isfinite(a.adaptive_ci) or a.adaptive_ci < 0:
        p.error("adaptive-ci must be finite and nonnegative")
    if a.rounds < 3 or a.flush_mb < 0 or min(a.pools, a.amortize, a.reps-1, a.iters, a.jobs, a.retries) < 1 or a.warmup < 0 or not math.isfinite(a.retry_delay) or a.retry_delay < 0 or not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error("invalid repetitions/jobs/retries/warmup/delay")
    out = a.out.resolve()
    if (out / "results.jsonl").exists() or (out / "run-manifest.json").exists():
        p.error("output already has measurements; choose a new directory")
    if a.archive_to and (a.archive_to.exists() or a.archive_to.resolve() == out or a.archive_to.resolve().is_relative_to(out)):
        p.error("archive destination must be new and outside the run directory")
    try:
        build_prefix = shlex.split(a.build_cmd)
    except ValueError as e:
        p.error(f"invalid build-cmd: {e}")
    prior = None
    if a.run_only:
        try:
            prior = json.loads((out / "build-manifest.json").read_text())
        except (OSError, ValueError) as e:
            p.error(f"run-only requires an immutable build-manifest.json: {e}")
    elif (out / "build-manifest.json").exists() or (out / "manifest.json").exists():
        p.error("output already has a build; use --run-only or choose a new directory")
    a.reference = a.reference or (prior["reference"] if prior else "off")
    if a.run_only and (a.variants_dir or a.ccbin or a.build_cmd):
        p.error("variants-dir/ccbin/build-cmd are build-only options; run-only uses the immutable bundle")
    kernel_path = a.kernel or HERE / "attention.json"
    config = json.loads(kernel_path.read_text()) if a.kernel or prior is None else prior["adapter"]
    base = kernel_path.resolve().parent
    name = a.kernel_name or config["name"]
    header = (base / config["header"]).resolve()
    variants_dir = (a.variants_dir or base / config["variants_dir"]).resolve()
    variants = a.variants or (prior["variants"] if prior else sorted(d.name for d in variants_dir.iterdir() if (d / "variant.cuh").is_file()))
    variants = list(dict.fromkeys([a.reference] + variants))
    if any(not VALID_NAME.fullmatch(v) for v in variants):
        p.error("unsafe variant name")
    if prior and (prior["arch"] != a.arch or prior["reference"] != a.reference or prior["adapter"] != config or
                  any(v not in prior["variants"] for v in variants)):
        p.error("build manifest differs from requested architecture/reference/adapter/variants")
    if a.cases:
        cases = validate_cases(json.loads(a.cases.read_text()))
    elif prior:
        cases = validate_cases(json.loads((out / "build-cases.json").read_text()))
    else:
        gen = [sys.executable, str(base / config["case_generator"]), "--suite", a.suite]
        if a.late_final:
            gen.append("--late-final")
        cases = validate_cases(json.loads(subprocess.check_output(gen, text=True)))
    all_cases = cases
    try:
        cases, shard = shard_cases(all_cases, a.shard)
        reduced_coverage(cases)
    except ValueError as e:
        p.error(str(e))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"kernel": name, "adapter": config, "reference": a.reference, "variants": variants,
                "argv": list(sys.argv if argv is None else [str(HERE / "run.py")] + argv), "arch": a.arch, "seed": a.seed, "warmup": a.warmup, "reps": a.reps, "iters": a.iters,
                "rounds_requested": a.rounds, "adaptive_ci": a.adaptive_ci,
                "adaptive_max_rounds": 6 if a.adaptive_ci else None,
                "orders": list(balanced_orders(variants, a.reference, 6 if a.adaptive_ci else a.rounds))[:6 if a.adaptive_ci else None],
                "flush_mb": a.flush_mb, "pools": a.pools, "amortize": a.amortize,
                "regime_note": "binary-reported; cold is not engine replay; captured/predecessor unavailable unless explicitly reported",
                "telemetry_note": "pre/post child-process snapshots only, no in-interval observation or clock control",
                "case_sha256": cases_hash(cases), "all_cases_sha256": cases_hash(all_cases), "shard": shard,
                "driver_sha256": sha256_file(HERE/"run.py"),
                "git_commit": command(["git", "-C", str(REPO), "rev-parse", "HEAD"], 10)[1].strip(),
                "git_status": command(["git", "-C", str(REPO), "status", "--short"], 10)[1],
                "started_unix": time.time()}
    builds = {}

    def build(v):
        cmd = build_prefix + [a.nvcc, "-std=c++17", "-O3", "-gencode", f"arch=compute_{a.arch},code=sm_{a.arch}",
               "-I", str(REPO / "engine"), "-Xptxas", "-v", f"-DSWEEP_SAME_VARIANT={int(v == a.reference)}"]
        if a.ccbin:
            cmd += ["-ccbin", a.ccbin]
        cmd += ["-MD", "-MF", str(out / f"{v}.deps.d")]
        cmd += ["-D" + d for d in config.get("defines", [])]
        cmd += [f'-DSWEEP_ADAPTER_HEADER="{header}"',
                f'-DSWEEP_REFERENCE_HEADER="{variants_dir / a.reference / "variant.cuh"}"',
                f'-DSWEEP_VARIANT_HEADER="{variants_dir / v / "variant.cuh"}"', str(HERE / "bench.cu"), "-o", str(out / v)]
        rc, stdout, stderr = command(cmd, 900)
        (out / f"{v}.build.log").write_text(stdout + stderr)
        print(f"BUILD {v}: {'OK' if rc == 0 else 'FAIL'} rc={rc}", flush=True)
        deps = build_dependencies(out / f"{v}.deps.d")
        explicit = [HERE / "bench.cu", header, variants_dir / a.reference / "variant.cuh", variants_dir / v / "variant.cuh"]
        return v, {"returncode": rc, "command": cmd,
                   "dependency_capture": "nvcc depfile" if deps else "UNAVAILABLE; explicit inputs only",
                   "source_sha256": source_hashes(explicit + deps),
                   "binary_sha256": sha256_file(out / v) if rc == 0 and (out / v).is_file() else None}

    if not a.run_only:
        compiler = command(build_prefix + [a.nvcc, "--version"], 10)
        manifest.update(nvcc_version=compiler[1], compiler_identity={"command": build_prefix + [a.nvcc, "--version"],
                        "returncode": compiler[0], "stdout": compiler[1], "stderr": compiler[2], "ccbin": a.ccbin})
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for v, data in pool.map(build, variants):
                builds[v] = data
        (out / "builds.json").write_text(json.dumps(builds, indent=2) + "\n")
        manifest.update(builds=builds, binary_sha256={v: data["binary_sha256"] for v, data in builds.items() if data["returncode"] == 0},
                        source_sha256=source_hashes([kernel_path, HERE / "run.py"]),
                        source_hash_note="explicit sources plus nvcc dependencies hashed after compile; null means unavailable",
                        build_finished_unix=time.time())
        with (out / "build-manifest.json").open("x") as f:
            f.write(json.dumps(manifest, indent=2) + "\n")
        (out / "build-cases.json").write_text(json.dumps(all_cases, indent=2) + "\n")
        # Portable bundle: no checkout, Python module imports, gcc or nvcc needed to run.
        shutil.copy2(HERE / "run.py", out / "run.py")
        if (HERE/"merge.py").is_file():
            shutil.copy2(HERE/"merge.py", out/"merge.py")
        prior = json.loads((out / "build-manifest.json").read_text())
    else:
        builds = prior["builds"]
    for v in variants:
        if builds.get(v, {}).get("returncode") == 0:
            expected = prior["binary_sha256"].get(v)
            if not expected or not (out / v).is_file() or sha256_file(out / v) != expected:
                p.error(f"binary hash mismatch or missing binary: {v}; refusing stale/tampered build")
    if a.build_only:
        (out / "manifest.json").write_text(json.dumps(prior, indent=2) + "\n")
        archive_run(out, a.archive_to)
        return int(any(x["returncode"] for x in builds.values()))
    # Build identity remains the immutable original, even from another checkout/host.
    run_git = {"commit": manifest["git_commit"], "status": manifest["git_status"]}
    manifest.update(git_commit=prior["git_commit"], git_status=prior["git_status"],
                    nvcc_version=prior["nvcc_version"], builds=builds,
                    binary_sha256={v: prior["binary_sha256"].get(v) for v in variants},
                    run_checkout=run_git, build_manifest="build-manifest.json",
                    build_manifest_sha256=sha256_file(out / "build-manifest.json"),
                    source_identity="original immutable build record; run checkout is NOT binary provenance")
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (out / "all-cases.json").write_text(json.dumps(all_cases, indent=2) + "\n")
    with (out / "run-manifest.json").open("x") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, pairs = [], []
    first_logs = set()

    def run_one(case, v, round_id, pair_id, role):
        run_id = f"{pair_id}.{role}"
        row = {"kernel": name, "variant": v, "case": case["id"], "status": "BUILD_FAILED",
               "round_id": round_id, "pair_id": pair_id, "run_id": run_id, "role": role}
        attempts = a.retries if builds.get(v, {}).get("returncode") == 0 else 1
        legacy_logs = (v, case["id"]) not in first_logs
        first_logs.add((v, case["id"]))
        for attempt in range(attempts):
            current = dict(row, attempt=attempt+1, final_attempt=True)
            if row["status"] == "BUILD_FAILED" and builds.get(v, {}).get("returncode") != 0:
                rows.append(current)
                with (out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(current) + "\n")
                return current
            cmd = [str(out / v), "--warmup", str(a.warmup), "--reps", str(a.reps), "--iters", str(a.iters),
                   "--flush-mb", str(a.flush_mb), "--pools", str(a.pools), "--amortize", str(a.amortize),
                   "--seed", str(a.seed), "--"] + list(map(str, case["args"]))
            current["telemetry_before"] = telemetry_snapshot()
            current["started_unix"] = time.time()
            rc, stdout, stderr = command(cmd, a.timeout)
            current["finished_unix"] = time.time()
            current["telemetry_after"] = telemetry_snapshot()
            prefix = out / f"{v}.{run_id}.try{attempt+1}"
            prefix.with_suffix(prefix.suffix + ".stdout").write_text(stdout)
            prefix.with_suffix(prefix.suffix + ".stderr").write_text(stderr)
            if legacy_logs:
                legacy = out / f"{v}.{case['id']}.try{attempt+1}"
                legacy.with_suffix(legacy.suffix + ".stdout").write_text(stdout)
                legacy.with_suffix(legacy.suffix + ".stderr").write_text(stderr)
            current.update(command=cmd, returncode=rc, stdout_file=prefix.name + ".stdout",
                           stderr_file=prefix.name + ".stderr",
                           status="LOW_MEMORY" if rc == 77 else "TIMEOUT" if rc == 124 else "RUN_FAILED")
            if rc in (0, 1) or (rc not in (77, 124) and stdout.strip().endswith("}")):
                try:
                    data = json.loads(stdout.strip().splitlines()[-1])
                    if not isinstance(data, dict) or data.get("status") not in ("EXACT", "MISMATCH", "MEMORY_CORRUPTION", "INVALID_LAUNCH"):
                        raise ValueError("bad status")
                    # Binary output cannot overwrite driver provenance, telemetry or pairing IDs.
                    current["binary_output"] = data
                    for key, value in data.items():
                        if key not in current or key == "status":
                            current[key] = value
                    if data["status"] == "EXACT" and not valid_measurement(current):
                        raise ValueError("invalid timing or inconsistent exit status")
                    current["regime"] = timing_regime(data)
                    current.update(ptxas_resources((out / f"{v}.build.log").read_text(), data.get("ptxas_key")))
                except (ValueError, IndexError, KeyError, TypeError):
                    current["status"] = "INVALID_OUTPUT"
            retry = rc == 77 and attempt+1 < attempts
            current["final_attempt"] = not retry
            rows.append(current)
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps(current) + "\n")
            print(f"RUN {v}/{case['id']} {run_id} try={attempt+1}: {current['status']} mean_us={current.get('mean_us', 'UNKNOWN')}", flush=True)
            if not retry:
                return current
            time.sleep(a.retry_delay)

    for case in cases:
        stopped = set()
        for r, order in enumerate(manifest["orders"]):
            for position, v in enumerate(order):
                if v in stopped:
                    continue
                pair_id = f"{case['id']}.r{r+1}.p{position+1}"
                before = run_one(case, a.reference, r+1, pair_id, "off_before")
                candidate = run_one(case, v, r+1, pair_id, "candidate")
                after = run_one(case, a.reference, r+1, pair_id, "off_after")
                pair = pair_record(before, candidate, after)
                pairs.append(pair)
                with (out / "pairs.jsonl").open("a") as f:
                    f.write(json.dumps(pair) + "\n")
                if a.adaptive_ci:
                    decision = adaptive_decision(rows, pairs, case["id"], v, a.adaptive_ci)
                    with (out / "adaptive-decisions.jsonl").open("a") as f:
                        f.write(json.dumps(dict(case=case["id"], variant=v, **decision)) + "\n")
                    if decision["stop"]:
                        stopped.add(v)
        summary = aggregate_runs(rows, pairs, cases, variants, a.reference, a.adaptive_ci)
        write_tables(out, summary, cases, variants, a.reference, name)
    manifest["finished_unix"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive_run(out, a.archive_to)
    print(f"TABLE {out / 'summary.md'}\nCSV {out / 'summary.csv'}", flush=True)
    return int(any(r["status"] != "EXACT" for r in summary) or any(r["status"] != "EXACT" for r in pairs))


if __name__ == "__main__":
    sys.exit(main())
