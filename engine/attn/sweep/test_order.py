"""CPU-only deterministic measurement-order and telemetry regression tests."""
import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("order_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row(v="fast", mean=90, **kw):
    return dict(kernel="toy", case="x", variant=v, mean_us=mean, sd_us=1,
                status="EXACT", returncode=0, round_id=1, pair_id="x.r1.p1",
                run_id=v, final_attempt=True, **kw)


class OrderTests(unittest.TestCase):
    def test_complete_balanced_blocks(self):
        for size in (1, 2, 3, 5):
            names = [f"v{i}" for i in range(size)]
            orders = list(runner.balanced_orders(["off"] + names, "off", 3))
            self.assertGreaterEqual(len(orders), 3)
            for pos in range(size):
                counts = [sum(order[pos] == v for order in orders) for v in names]
                self.assertEqual(len(set(counts)), 1)
            self.assertEqual(orders, list(runner.balanced_orders(["off"]+names, "off", 3)))

    def test_t_ci_is_between_pairs_not_rep_sd(self):
        stats = runner.uncertainty([-12, -10, -8])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["mean"], -10)
        self.assertEqual(stats["sd"], 2)
        self.assertAlmostEqual(stats["ci95_high"], -10 + 4.302653*2/math.sqrt(3))
        self.assertIsNone(runner.uncertainty([1])["sd"])
        self.assertIsNone(runner.uncertainty([])["mean"])

    def test_in_process_estimator_wins_over_external_controls(self):
        before, after = row("off", 80), row("off", 120)
        candidate = row(paired_delta_pct_mean=-5, reference_mean_us=95)
        pair = runner.pair_record(before, candidate, after)
        self.assertEqual(pair["delta_pct"], -5)
        self.assertAlmostEqual(pair["bracket_delta_pct"], -10)
        self.assertEqual(pair["delta_us"], -5)
        self.assertIn("in-process", pair["estimator"])
        candidate.pop("paired_delta_pct_mean")
        self.assertIn("fallback", runner.pair_record(before, candidate, after)["estimator"])

    def test_failures_and_mixed_regimes_do_not_contribute_totals(self):
        for change in ({"status":"MISMATCH"}, {"status":"TIMEOUT"}, {"returncode":1},
                       {"mean_us":float("nan")}, {"mean_us":0}, {"launch_valid":False},
                       {"integrity_errors":1}, {"mismatch_bytes":1},
                       {"input_slots":3}, {"paired_delta_pct_mean":float("nan"), "reference_mean_us":100}):
            before, candidate, after = row("off",100), row(), row("off",100)
            candidate.update(change)
            pair = runner.pair_record(before,candidate,after)
            self.assertEqual(pair["status"],"INVALID_PAIR")
            cases=[dict(id="x",args=[],weights={"toy":2})]
            summary=runner.aggregate_runs([before,candidate,after],[pair],cases,["off","fast"],"off")
            with tempfile.TemporaryDirectory() as tmp:
                tables=runner.write_tables(Path(tmp),summary,cases,["off","fast"],"off","toy")
            total=next(r for r in tables if r["variant"]=="fast" and r["case"].startswith("TOTAL"))
            self.assertEqual(total["status"],"INCOMPLETE")
            self.assertNotIn("contribution_toy_ms",total)

    def test_partial_round_failure_and_mixed_estimators(self):
        cases=[dict(id="x",args=[],weights={})]
        before, candidate, after=row("off",100),row(),row("off",100)
        pair=runner.pair_record(before,candidate,after)
        failed=dict(pair,status="INVALID_PAIR")
        summary=runner.aggregate_runs([before,candidate,after],[pair,failed],cases,["fast"],"off")
        self.assertNotIn("mean_us",summary[0])
        self.assertEqual(summary[0]["n_pairs"],1)
        self.assertEqual(summary[0]["n_pairs_requested"],2)
        mixed=dict(pair,estimator="other")
        self.assertEqual(runner.aggregate_runs([candidate],[pair,mixed],cases,["fast"],"off")[0]["status"],"INCOMPLETE")

    def test_regime_labels_do_not_invent_cold_or_engine(self):
        self.assertIn("warm/repeated-address",runner.timing_regime({}))
        self.assertIn("unavailable",runner.timing_regime({"input_slots":"bad"}))
        self.assertIn("coldness/predecessor unverified",runner.timing_regime({"input_slots":3}))
        self.assertEqual(runner.timing_regime({"timing_regime":"flush-rotate"}),"flush-rotate")

    def test_telemetry_complete_partial_unavailable(self):
        xml='''<nvidia_smi_log><gpu id="1"><uuid>GPU-test</uuid>
          <clocks><graphics_clock>2600 MHz</graphics_clock></clocks>
          <temperature><gpu_temp>52 C</gpu_temp></temperature>
          <gpu_power_readings><power_draw>200 W</power_draw></gpu_power_readings>
          <clocks_event_reasons><clocks_event_reason_hw_thermal_slowdown>Not Active</clocks_event_reason_hw_thermal_slowdown></clocks_event_reasons>
          <processes><process_info><pid>123</pid><type>G</type><process_name>competing graphics</process_name></process_info></processes>
          </gpu></nvidia_smi_log>'''
        with patch.object(runner,"command",return_value=(0,xml,"")) as cmd:
            snap=runner.telemetry_snapshot()
        self.assertEqual(snap["status"],"AVAILABLE")
        self.assertEqual(snap["gpus"][0]["processes"][0]["pid"],"123")
        self.assertEqual(snap["gpus"][0]["clocks"]["graphics_clock"],"2600 MHz")
        self.assertEqual(snap["stdout"],xml)
        cmd.assert_called_once_with(["nvidia-smi","-q","-x"],5)
        for result, expected in [((127,"","missing"),"UNAVAILABLE"),((0,"bad xml",""),"UNAVAILABLE"),((0,"<root><gpu/></root>",""),"PARTIAL")]:
            with patch.object(runner,"command",return_value=result):
                self.assertEqual(runner.telemetry_snapshot()["status"],expected)

    def test_driver_serial_brackets_raw_attempts_and_telemetry_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for v in ("off","fast","other"):
                (root/"variants"/v).mkdir(parents=True)
                (root/"variants"/v/"variant.cuh").write_text("// test\n")
            (root/"fixture.h").write_text("// test\n")
            (root/"adapter.json").write_text(json.dumps(dict(name="toy",header="fixture.h",variants_dir="variants")))
            (root/"cases.json").write_text(json.dumps([dict(id="x",args=["opaque"],weights={"toy":2})]))
            # Actual executable CPU fake binaries; no CUDA compiler/device required.
            fake=root/"nvcc"
            fake.write_text('''#!/usr/bin/env python3
import sys,pathlib
if '--version' in sys.argv: print('test compiler'); sys.exit(0)
out=pathlib.Path(sys.argv[sys.argv.index('-o')+1])
out.write_text("""#!/usr/bin/env python3
import json,pathlib,sys
me=pathlib.Path(sys.argv[0]); marker=me.with_suffix('.attempt')
if me.name=='fast' and not marker.exists():
 marker.write_text('1'); sys.exit(77)
print(json.dumps(dict(status='EXACT',mean_us=100 if me.name=='off' else 90,sd_us=1,samples_us=[89,91],reference_mean_us=95,paired_delta_pct_mean=-5,paired_delta_pct_samples=[-4,-6],null_delta_pct_samples=[0,1],null_delta_pct_mean=.5,null_band_pct=3,input_slots=3,timed_checks=2,checks=4)))
""")
out.chmod(0o755)
''')
            fake.chmod(0o755)
            events=[]
            real_command=runner.command
            def command(cmd, timeout):
                if cmd[0] in [str(root/"out"/v) for v in ("off","fast","other")]:
                    events.append("run")
                return real_command(cmd,timeout)
            def telemetry():
                events.append("snapshot")
                return {"status":"AVAILABLE","sequence":len(events)}
            with patch.object(runner,"command",side_effect=command), patch.object(runner,"telemetry_snapshot",side_effect=telemetry), contextlib.redirect_stdout(io.StringIO()):
                result=runner.main(["--kernel",str(root/"adapter.json"),"--cases",str(root/"cases.json"),"--nvcc",str(fake),"--arch","120a","--out",str(root/"out"),"--retry-delay","0"])
            self.assertEqual(result,0)
            self.assertEqual(events,["snapshot","run","snapshot"]*25)
            raw=[json.loads(line) for line in (root/"out/results.jsonl").read_text().splitlines()]
            pairs=[json.loads(line) for line in (root/"out/pairs.jsonl").read_text().splitlines()]
            self.assertEqual(len(raw),25)  # 4 rounds x 2 candidates x 3 + retry
            self.assertEqual(len(pairs),8)
            self.assertEqual(raw[1]["status"],"LOW_MEMORY")
            self.assertFalse(raw[1]["final_attempt"])
            self.assertEqual(len({(r["run_id"],r["attempt"]) for r in raw}),25)
            final=[r for r in raw if r["final_attempt"]]
            for i in range(0,len(final),3):
                triplet=final[i:i+3]
                self.assertEqual([r["role"] for r in triplet],["off_before","candidate","off_after"])
                self.assertEqual(triplet[0]["variant"],"off")
                self.assertEqual(triplet[2]["variant"],"off")
                self.assertEqual(len({r["pair_id"] for r in triplet}),1)
            self.assertEqual(final[1]["samples_us"],[89,91])
            self.assertEqual(final[1]["checks"],4)
            self.assertEqual(final[1]["null_band_pct"],3)
            self.assertEqual(pairs[0]["delta_pct"],-5)
            self.assertTrue((root/"out/fast.x.try2.stdout").exists())
            summary=runner.aggregate_runs(raw,pairs,[dict(id="x")],["off","fast","other"],"off")
            fast=next(r for r in summary if r["variant"]=="fast")
            self.assertEqual(fast["n_pairs"],4)
            self.assertEqual(fast["paired_delta_pct"],-5)
            self.assertEqual(fast["paired_sd_pct"],0)


class ProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        for v in ("off", "fast"):
            (root / "variants" / v).mkdir(parents=True)
            (root / "variants" / v / "variant.cuh").write_text("// fixture " + v)
        (root / "fixture.h").write_text("// adapter header")
        (root / "extra header.h").write_text("// dependency with a space")
        (root / "adapter.json").write_text(json.dumps(dict(name="toy", header="fixture.h", variants_dir="variants")))
        (root / "cases.json").write_text(json.dumps([dict(id="x", args=["opaque"], weights={"toy":2})]))
        return ["--kernel", str(root / "adapter.json"), "--cases", str(root / "cases.json"),
                "--nvcc", "fake-nvcc", "--arch", "121a", "--out", str(root / "bundle")]

    def fake_build(self, root, commands, status="EXACT", returncode=0):
        original = runner.command
        def command(cmd, timeout):
            if "fake-nvcc" not in cmd:
                return original(cmd, timeout)
            commands.append(cmd)
            if "--version" in cmd:
                return 0, "immutable fake compiler version", ""
            out = Path(cmd[cmd.index("-o")+1])
            out.write_text(f"#!/usr/bin/env python3\nimport json,sys\nstatus='EXACT' if sys.argv[0].endswith('off') else {status!r}\nprint(json.dumps(dict(status=status, mean_us=10, sd_us=1, samples_us=[9,11], integrity_errors=0 if status=='EXACT' else 2, launch_valid=status=='EXACT')))\nsys.exit(0 if status=='EXACT' else {returncode})\n")
            out.chmod(0o755)
            dep = Path(cmd[cmd.index("-MF")+1])
            dep.write_text(str(out) + ": " + str(root / "extra header.h").replace(" ", "\\ ") + "\n")
            return 0, "", ""
        return command

    def test_immutable_build_portable_run_without_compiler_or_source_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            commands = []
            prefix = 'wrapper --label "a b; touch NEVER"'
            with patch.object(runner, "command", side_effect=self.fake_build(root, commands)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(args + ["--build-only", "--ccbin", "/compiler/g++", "--build-cmd", prefix,
                                                    "--archive-to", str(root / "build-copy")]), 0)
            bundle = root / "bundle"
            before = (bundle / "build-manifest.json").read_bytes()
            build = json.loads(before)
            self.assertEqual(build["nvcc_version"], "immutable fake compiler version")
            self.assertEqual(build["compiler_identity"]["ccbin"], "/compiler/g++")
            for cmd in commands:
                self.assertEqual(cmd[:3], ["wrapper", "--label", "a b; touch NEVER"])
            for v in ("off", "fast"):
                cmd = build["builds"][v]["command"]
                self.assertIn("-ccbin", cmd)
                self.assertEqual(cmd[cmd.index("-ccbin")+1], "/compiler/g++")
                self.assertEqual(build["builds"][v]["dependency_capture"], "nvcc depfile")
                self.assertEqual(build["builds"][v]["source_sha256"][str(root / "extra header.h")],
                                 runner.sha256_file(root / "extra header.h"))
            self.assertEqual((root / "build-copy/build-manifest.json").read_bytes(), before)
            # Move the standalone bundle; the source/compiler need not be available.
            moved = root / "portable"
            shutil.copytree(bundle, moved)
            (root / "fixture.h").write_text("// changed source must NOT become binary provenance")
            fake_smi = root / "nvidia-smi"
            fake_smi.write_text("#!/bin/sh\nprintf '<nvidia_smi_log/>'\n")
            fake_smi.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"])
            completed = subprocess.run([sys.executable, str(moved / "run.py"), "--run-only", "--arch", "121a",
                                        "--out", str(moved), "--nvcc", "/does/not/exist", "--archive-to", str(root / "archive")],
                                       env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual((moved / "build-manifest.json").read_bytes(), before)
            run = json.loads((moved / "run-manifest.json").read_text())
            self.assertEqual(run["git_commit"], build["git_commit"])
            self.assertEqual(run["nvcc_version"], build["nvcc_version"])
            self.assertEqual(run["build_manifest_sha256"], runner.sha256_file(moved / "build-manifest.json"))
            self.assertIn("run_checkout", run)
            original_files = {p.relative_to(moved):p.read_bytes() for p in moved.rglob("*") if p.is_file()}
            archived_files = {p.relative_to(root / "archive"):p.read_bytes() for p in (root / "archive").rglob("*") if p.is_file()}
            self.assertEqual(original_files, archived_files)
            self.assertIn(Path("results.jsonl"), archived_files)
            self.assertIn(Path("fast"), archived_files)
            self.assertIn(Path("fast.build.log"), archived_files)
            self.assertIn(Path("run-manifest.json"), archived_files)

    def test_tampered_missing_binary_mismatched_arch_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            with patch.object(runner, "command", side_effect=self.fake_build(root, [])), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(args + ["--build-only"]), 0)
            bundle = root / "bundle"
            before = (bundle / "build-manifest.json").read_bytes()
            # Existing build cannot be overwritten (including no compiler invocation).
            for more in (["--build-only"], ["--run-only", "--arch", "120a"],
                         ["--run-only", "--archive-to", str(bundle / "nested")],
                         ["--run-only", "--variants-dir", str(root / "variants")]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    runner.main(args + more)
                self.assertEqual(raised.exception.code, 2)
            (bundle / "fast").write_text("tampered")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                runner.main(args + ["--run-only"])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual((bundle / "build-manifest.json").read_bytes(), before)
            self.assertFalse((bundle / "results.jsonl").exists())
            self.assertFalse((bundle / "run-manifest.json").exists())
            # Missing binary is also a hash failure, not silently skipped.
            (bundle / "fast").rename(bundle / "fast.moved")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(args + ["--run-only"])

    def test_diagnostic_failure_statuses_preserve_numbers_but_not_valid_totals(self):
        for status, rc in [("MISMATCH",1), ("MEMORY_CORRUPTION",3), ("INVALID_LAUNCH",4)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                args=self.fixture(root)
                with patch.object(runner,"command",side_effect=self.fake_build(root,[],status,rc)), patch.object(runner,"telemetry_snapshot",return_value={"status":"UNAVAILABLE"}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(args+["--archive-to",str(root/"failed-archive")]),1)
                raw=[json.loads(line) for line in (root/"bundle/results.jsonl").read_text().splitlines()]
                candidates=[r for r in raw if r["variant"]=="fast"]
                self.assertEqual(len(candidates),3)
                for r in candidates:
                    self.assertEqual(r["status"],status)
                    self.assertEqual(r["mean_us"],10)
                    self.assertEqual(r["returncode"],rc)
                    self.assertEqual(r["binary_output"]["samples_us"],[9,11])
                pairs=[json.loads(line) for line in (root/"bundle/pairs.jsonl").read_text().splitlines()]
                self.assertTrue(all(r["status"]=="INVALID_PAIR" for r in pairs))
                self.assertTrue((root/"failed-archive/results.jsonl").exists())
                with (root/"bundle/summary.csv").open() as f:
                    tables=list(runner.csv.DictReader(f))
                total=next(r for r in tables if r["variant"]=="fast" and r["case"].startswith("TOTAL"))
                self.assertEqual(total["status"],"INCOMPLETE")
                self.assertEqual(total["contribution_toy_ms"],"")

    def test_depfile_escaped_spaces_and_continuations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deps.d"
            path.write_text("target: /one/file.h \\\n /two/with\\ space.h\n")
            self.assertEqual(runner.build_dependencies(path), [Path("/one/file.h"), Path("/two/with space.h")])


if __name__ == "__main__":
    unittest.main()
