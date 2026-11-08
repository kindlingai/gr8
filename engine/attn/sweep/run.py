#!/usr/bin/env python3
"""Reusable isolated CUDA variant builder/runner. All kernel knowledge is in adapters."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
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
        if not isinstance(case.get("weights", {}), dict) or any(not isinstance(v, (int, float)) or v < 0 for v in case.get("weights", {}).values()):
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


def write_tables(out, rows, cases, variants, reference, kernel):
    baseline = {r["case"]: r for r in rows if r["variant"] == reference and "mean_us" in r}
    weights = {c["id"]: c.get("weights", {}) for c in cases}
    labels = sorted({k for c in cases for k in c.get("weights", {})}, key=lambda x: sum(c.get("weights", {}).get(x, 0) for c in cases))
    fields = ["kernel", "variant", "case", "status", "mean_us", "sd_us", "vs_reference_pct", "mismatch_bytes", "regs", "ptxas_regs",
              "stack_bytes", "spill_store_bytes", "spill_load_bytes", "dynamic_smem", "local_bytes"]
    for label in labels:
        fields.extend([f"calls_{label}", f"contribution_{label}_ms"])
    for r in rows:
        base = baseline.get(r["case"])
        if "mean_us" in r and base and base["mean_us"] > 0:
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
            total = {"kernel": kernel, "variant": variant, "case": f"TOTAL_{label}_attention_only",
                     "status": "INCOMPLETE", f"calls_{label}": sum(c["weights"][label] for c in needed)}
            if all(c["id"] in vr and "mean_us" in vr[c["id"]] for c in needed):
                total[f"contribution_{label}_ms"] = sum(vr[c["id"]]["mean_us"] * c["weights"][label] / 1000 for c in needed)
                total["status"] = "EXACT" if all(vr[c["id"]]["status"] == "EXACT" for c in needed) else "MISMATCH"
                totals[(variant, label)] = total[f"contribution_{label}_ms"]
            output_rows.append(total)
    for r in output_rows[len(rows):]:
        label = r["case"].removeprefix("TOTAL_").removesuffix("_attention_only")
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
    md = [f"# {kernel} kernel sweep", "", f"Reference: `{reference}`. CUDA event batch means; SD is sample SD across repetitions.",
          "Prompt totals are weighted attention-core estimates, NOT TTFT. No rank multiplier.",
          "Resource columns describe the main kernel; timing includes every kernel in launch().", "",
          "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    md.extend("| " + " | ".join(cell(r.get(k)) for k in fields) + " |" for r in output_rows)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return output_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", type=Path, default=HERE / "attention.json", help="adapter JSON")
    p.add_argument("--kernel-name")
    p.add_argument("--cases", type=Path, help="opaque adapter case-list JSON")
    p.add_argument("--variants-dir", type=Path)
    p.add_argument("--reference", default="off")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--arch", choices=("120a", "121a"), required=True)
    p.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suite", default="all")
    p.add_argument("--late-final", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--retries", type=int, default=3, help="total low-memory attempts")
    p.add_argument("--retry-delay", type=float, default=5)
    p.add_argument("--timeout", type=float, default=180, help="per case seconds")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--run-only", action="store_true")
    a = p.parse_args(argv)
    if a.build_only and a.run_only:
        p.error("build-only and run-only are mutually exclusive")
    if min(a.reps-1, a.iters, a.jobs, a.retries) < 1 or a.warmup < 0 or a.retry_delay < 0:
        p.error("invalid repetitions/jobs/retries/warmup/delay")
    config = json.loads(a.kernel.read_text())
    base = a.kernel.resolve().parent
    name = a.kernel_name or config["name"]
    header = (base / config["header"]).resolve()
    variants_dir = (a.variants_dir or base / config["variants_dir"]).resolve()
    variants = a.variants or sorted(d.name for d in variants_dir.iterdir() if (d / "variant.cuh").is_file())
    variants = list(dict.fromkeys([a.reference] + variants))
    if any(not VALID_NAME.fullmatch(v) for v in variants):
        p.error("unsafe variant name")
    if a.cases:
        cases = validate_cases(json.loads(a.cases.read_text()))
    else:
        gen = [sys.executable, str(base / config["case_generator"]), "--suite", a.suite]
        if a.late_final:
            gen.append("--late-final")
        cases = validate_cases(json.loads(subprocess.check_output(gen, text=True)))
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite an earlier measurement.
    if (out / "results.jsonl").exists():
        p.error("output already has measurements; choose a new directory")
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    manifest = {"kernel": name, "adapter": config, "reference": a.reference, "variants": variants,
                "argv": sys.argv, "arch": a.arch, "seed": a.seed, "warmup": a.warmup, "reps": a.reps, "iters": a.iters,
                "case_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                "git_commit": command(["git", "-C", str(REPO), "rev-parse", "HEAD"], 10)[1].strip(),
                "git_status": command(["git", "-C", str(REPO), "status", "--short"], 10)[1],
                "nvcc_version": command([a.nvcc, "--version"], 10)[1], "started_unix": time.time()}
    builds = {}

    def build(v):
        cmd = [a.nvcc, "-std=c++17", "-O3", "-gencode", f"arch=compute_{a.arch},code=sm_{a.arch}",
               "-I", str(REPO / "engine"), "-Xptxas", "-v"]
        cmd += ["-D" + d for d in config.get("defines", [])]
        cmd += [f'-DSWEEP_ADAPTER_HEADER="{header}"',
                f'-DSWEEP_REFERENCE_HEADER="{variants_dir / a.reference / "variant.cuh"}"',
                f'-DSWEEP_VARIANT_HEADER="{variants_dir / v / "variant.cuh"}"', str(HERE / "bench.cu"), "-o", str(out / v)]
        rc, stdout, stderr = command(cmd, 900)
        (out / f"{v}.build.log").write_text(stdout + stderr)
        print(f"BUILD {v}: {'OK' if rc == 0 else 'FAIL'} rc={rc}", flush=True)
        return v, {"returncode": rc, "command": cmd}

    if not a.run_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for v, data in pool.map(build, variants):
                builds[v] = data
        (out / "builds.json").write_text(json.dumps(builds, indent=2) + "\n")
    else:
        builds = json.loads((out / "builds.json").read_text())
        # Reject stale architecture/config when reusing native binaries.
        prior = json.loads((out / "manifest.json").read_text())
        if prior["arch"] != a.arch or prior["reference"] != a.reference or prior["adapter"] != config:
            p.error("build manifest differs from requested architecture/reference/adapter")
    manifest["builds"] = builds
    manifest["binary_sha256"] = {v: hashlib.sha256((out / v).read_bytes()).hexdigest()
                                  for v in variants if builds.get(v, {}).get("returncode") == 0}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if a.build_only:
        return int(any(x["returncode"] for x in builds.values()))
    rows = []
    for case in cases:
        for v in variants:
            row = {"kernel": name, "variant": v, "case": case["id"], "status": "BUILD_FAILED"}
            if builds.get(v, {}).get("returncode") == 0:
                cmd = [str(out / v), "--warmup", str(a.warmup), "--reps", str(a.reps), "--iters", str(a.iters),
                       "--seed", str(a.seed), "--"] + list(map(str, case["args"]))
                for attempt in range(a.retries):
                    rc, stdout, stderr = command(cmd, a.timeout)
                    prefix = out / f"{v}.{case['id']}.try{attempt+1}"
                    prefix.with_suffix(prefix.suffix + ".stdout").write_text(stdout)
                    prefix.with_suffix(prefix.suffix + ".stderr").write_text(stderr)
                    if rc != 77:
                        break
                    print(f"LOW_MEMORY {v}/{case['id']} attempt={attempt+1}/{a.retries}: {stderr.strip()}", flush=True)
                    if attempt + 1 < a.retries:
                        time.sleep(a.retry_delay)
                row["returncode"] = rc
                row["status"] = "LOW_MEMORY" if rc == 77 else "TIMEOUT" if rc == 124 else "RUN_FAILED"
                if rc in (0, 1):
                    try:
                        data = json.loads(stdout.strip().splitlines()[-1])
                        if data["status"] not in ("EXACT", "MISMATCH"):
                            raise ValueError("bad status")
                        row.update(data)
                        row.update(ptxas_resources((out / f"{v}.build.log").read_text(), data["ptxas_key"]))
                    except (ValueError, IndexError, KeyError):
                        row["status"] = "INVALID_OUTPUT"
            rows.append(row)
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"RUN {v}/{case['id']}: {row['status']} mean_us={row.get('mean_us', 'UNKNOWN')}", flush=True)
        write_tables(out, rows, cases, variants, a.reference, name)
    manifest["finished_unix"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"TABLE {out / 'summary.md'}\nCSV {out / 'summary.csv'}", flush=True)
    return int(any(r["status"] != "EXACT" for r in rows))


if __name__ == "__main__":
    sys.exit(main())
