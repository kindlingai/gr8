#!/usr/bin/env python3
"""Tests for the fc2 (down-projection) adapter of the sweep bench.

    python3 -m unittest discover -s engine/attn/sweep/fc2            # CPU + compile gates (+ GPU when 3 GB are free)
    FC2_SKIP_GPU=1 python3 -m unittest ...test_fc2_adapter           # CPU + compile gates only
    FC2_NVCC=/usr/local/cuda-13.0/bin/nvcc ...                       # pick another nvcc

What is checked
  CPU     case lists validate; adapter constants equal the engine's (ds41_v2_ffn.cuh, ds_ro_cfg.h, ds41_v2.h);
          variants/tmpl/kernel.cuh is a byte-for-byte copy of mxtc::k_fc2_tc (only the symbol name differs);
          the copy-me recipe in the README yields a valid, uniquely keyed variant.
  COMPILE off, tmpl and the four negative controls compile for sm_120a AND sm_121a (nvcc runs ptxas); the ptxas key of
          off/tmpl matches exactly one kernel entry, 0 stack, 0 spill; off and tmpl have the same register count.
  GPU     (small M, few experts, under the campaign flock) off and tmpl are EXACT on uniform, skew+holes and tiny shapes;
          each negative control must be caught: ulp_flip -> MISMATCH with exactly 1 differing byte,
          dead_row_write -> MISMATCH, compute_once -> timed_mismatch_bytes > 0, skip_group -> MISMATCH.
  E2E     run.py builds and runs the smoke suite for off + tmpl; summary.csv carries the new columns, populated.
"""
import concurrent.futures
import csv
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent            # engine/attn/sweep/fc2
SWEEP = HERE.parent                               # engine/attn/sweep
ENGINE = HERE.parents[2]                          # engine
NVCC = os.environ.get("FC2_NVCC", "/usr/local/cuda/bin/nvcc")
LOCK = "/tmp/attn-sweep-5090.lock"
NEEDED_FREE_MIB = 3000                            # smoke shapes need about 0.1 GB per fixture; leave room for retries
CONTROLS = ("ulp_flip", "dead_row_write", "compute_once", "skip_group")
STOCK = ("off", "tmpl")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


runner = load(SWEEP / "run.py", "sweep_run")
fc2_cases = load(HERE / "cases.py", "fc2_cases")


def variant_header(name):
    for base in (HERE / "variants" / name, HERE / "negative_controls" / name):
        if (base / "variant.cuh").is_file():
            return base / "variant.cuh"
    raise FileNotFoundError(name)


def nvcc_command(name, arch, out, compile_only):
    cmd = [NVCC, "-std=c++17", "-O3", "-gencode", f"arch=compute_{arch},code=sm_{arch}", "-I", str(ENGINE), "-Xptxas", "-v",
           f'-DSWEEP_ADAPTER_HEADER="{HERE / "adapter.cuh"}"',
           f'-DSWEEP_REFERENCE_HEADER="{HERE / "variants/off/variant.cuh"}"',
           f'-DSWEEP_VARIANT_HEADER="{variant_header(name)}"', f"-DSWEEP_SAME_VARIANT={int(name == 'off')}"]
    if compile_only:
        cmd.append("-c")
    return cmd + [str(SWEEP / "bench.cu"), "-o", str(out)]


def gpu_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.split()[0]) if out.returncode == 0 else 0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0


def build_all(arch, outdir, names, compile_only):
    """Compile every named variant in parallel; returns {name: (returncode, ptxas log)}."""
    outdir.mkdir(parents=True, exist_ok=True)

    def one(name):
        out = outdir / (name + (".o" if compile_only else ""))
        r = subprocess.run(nvcc_command(name, arch, out, compile_only), capture_output=True, text=True, timeout=1200)
        log = r.stdout + r.stderr
        (outdir / f"{name}.build.log").write_text(log)
        return name, (r.returncode, log)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(names))) as pool:
        return dict(pool.map(one, names))


# ------------------------------------------------------------------------------------------------------ CPU tests
class CpuTests(unittest.TestCase):
    def test_case_lists_validate(self):
        for suite in ("all", "repro", "smoke"):
            rows = runner.validate_cases(fc2_cases.cases(suite))
            self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual([r["id"] for r in fc2_cases.cases("repro")], ["fc2_m8192_uniform"])
        repro = fc2_cases.cases("repro")[0]
        self.assertEqual(repro["args"], [8192, "--pattern", "uniform"])        # full width: 384 experts, no holes
        self.assertEqual(repro["input_provenance"], "synthetic")
        smoke_args = [r["args"] for r in fc2_cases.cases("smoke")]
        self.assertTrue(all(isinstance(v, (int, str)) for a in smoke_args for v in a))
        self.assertTrue(any("--holes" in a for a in smoke_args) and any("skew" in a for a in smoke_args))

    def test_case_generator_cli_matches_run_py_contract(self):
        out = subprocess.check_output([sys.executable, str(HERE / "cases.py"), "--suite", "smoke", "--late-final"], text=True)
        self.assertEqual(json.loads(out), fc2_cases.cases("smoke"))

    def test_kernel_config(self):
        cfg = json.loads((HERE / "fc2.json").read_text())
        self.assertEqual((cfg["name"], cfg["header"], cfg["variants_dir"], cfg["case_generator"]),
                         ("fc2_down_tc", "adapter.cuh", "variants", "cases.py"))
        for v in (HERE / cfg["variants_dir"]).iterdir():
            self.assertRegex(v.name, runner.VALID_NAME.pattern)
            self.assertTrue((v / "variant.cuh").is_file(), v)
        self.assertTrue((HERE / cfg["variants_dir"] / "off" / "variant.cuh").is_file())

    def test_adapter_constants_equal_the_engine(self):
        ffn = (ENGINE / "runtime/ds41_v2_ffn.cuh").read_text()
        cfg = (ENGINE / "kernels/ds_ro_cfg.h").read_text()
        v2h = (ENGINE / "runtime/ds41_v2.h").read_text()
        adapter = (HERE / "adapter.cuh").read_text()

        def engine_int(text, pattern):
            m = re.search(pattern, text)
            self.assertIsNotNone(m, pattern)
            return int(m[1])

        def adapter_int(name):
            m = re.search(rf"\b{name}\s*=\s*(\d+)", adapter)
            self.assertIsNotNone(m, name)
            return int(m[1])

        want = {
            "kH": engine_int(ffn, r"constexpr int H = (\d+)"),
            "kI": engine_int(ffn, r"\bI = (\d+)"),
            "kNE": engine_int(ffn, r"\bNE = (\d+)"),
            "kTK": engine_int(ffn, r"\bTK = (\d+)"),
            "kMM": engine_int(v2h, r"MAX_ROWS = (\d+)"),
            "kColGroups": engine_int(v2h, r"COLUMN_GROUPS = (\d+)"),
            "kW2": engine_int(cfg, r"#define DS41_TC_W2 (\d+)"),
            "kTW": engine_int(cfg, r"#define DS41_TC_WT (\d+)"),
            "kTPC": engine_int(cfg, r"#define DS41_TC_TPC (\d+)"),
            "kST2": engine_int(cfg, r"#define DS41_TC_ST2 (\d+)"),
            "kGY2": engine_int(cfg, r"#define DS41_TC_GY2 (\d+)"),
            "kOS2": engine_int(ffn, r"a\.os2=(\d+)"),     # hard-coded in args(); the cfg default 16 is NOT what is served
        }
        for name, value in want.items():
            self.assertEqual(adapter_int(name), value, name)
        # the served call: k_fc2_tc<I,H,W2,WT,TPC,ST2,COLUMN_GROUPS> with those numbers, in the stock variant
        stock = (HERE / "variants/off/variant.cuh").read_text()
        spec = f"k_fc2_tc<{want['kI']}, {want['kH']}, {want['kW2']}, {want['kTW']}, {want['kTPC']}, {want['kST2']}, {want['kColGroups']}>"
        self.assertEqual(stock.count(spec), 2, spec)   # kernel() and launch()
        self.assertIn("k_fc2_tc<I,H,DS41_TC_W2,DS41_TC_WT,DS41_TC_TPC,DS41_TC_ST2,COLUMN_GROUPS>", ffn)

    @staticmethod
    def kernel_body(text, name):
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if f"k_{name}(Args A)" in l)
        end = next(i for i in range(start, len(lines)) if lines[i] == "}")
        return [re.sub(r"k_fc2_(tc|tmpl)", "k_fc2_X", l) for l in lines[start:end + 1]]

    def test_tmpl_kernel_is_a_verbatim_copy_of_the_engine_kernel(self):
        engine = self.kernel_body((ENGINE / "kernels/ds_moe_tc.cuh").read_text(), "fc2_tc")
        copy = self.kernel_body((HERE / "variants/tmpl/kernel.cuh").read_text(), "fc2_tmpl")
        self.assertGreater(len(engine), 120)
        self.assertEqual(copy, engine, "variants/tmpl/kernel.cuh drifted from mxtc::k_fc2_tc; re-copy it")

    def test_copy_recipe_yields_a_valid_keyed_variant(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "variants"
            shutil.copytree(HERE / "variants/tmpl", work / "myopt")
            for f in (work / "myopt").iterdir():
                f.write_text(f.read_text().replace("tmpl", "myopt"))
            self.assertRegex("myopt", runner.VALID_NAME.pattern)
            text = (work / "myopt/variant.cuh").read_text()
            self.assertNotIn("tmpl", text + (work / "myopt/kernel.cuh").read_text())
            self.assertIn("k_fc2_myoptILi576ELi5120ELi4ELi2ELi8ELi8ELi4E", text)     # key follows the renamed symbol
            self.assertIn("fc2_myopt::k_fc2_myopt<576, 5120, 4, 2, 8, 8, 4>", text)

    def test_stock_ptxas_key_matches_the_mangled_name(self):
        # itanium mangling of a template kernel in namespace mxtc: _ZN4mxtc7k_fc2_tcILi576E...EEvNS_4ArgsE
        log = ("ptxas info : Function properties for _ZN4mxtc7k_fc2_tcILi576ELi5120ELi4ELi2ELi8ELi8ELi4EEEvNS_4ArgsE\n"
               "  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
               "ptxas info : Used 251 registers\n"
               "ptxas info : Function properties for _ZN4dsr6k_scanILi384E\n  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
               "ptxas info : Used 32 registers\n")
        key = re.search(r'ptxas_key.*?return "([^"]+)"', (HERE / "variants/off/variant.cuh").read_text())[1]
        self.assertEqual(runner.ptxas_resources(log, key)["ptxas_regs"], 251)

    def test_every_control_says_it_must_never_be_scored(self):
        for c in CONTROLS:
            self.assertIn("NEVER be scored", (HERE / "negative_controls" / c / "variant.cuh").read_text(), c)


# ------------------------------------------------------------------------------------------------ compile gates
class CompileGates(unittest.TestCase):
    """nvcc compile-only (ptxas runs) for both architectures: sm_120a (RTX 5090) and sm_121a (GB10)."""
    names = STOCK + CONTROLS

    @classmethod
    def setUpClass(cls):
        if not Path(NVCC).exists():
            raise unittest.SkipTest(f"{NVCC} missing")
        cls.tmp = tempfile.TemporaryDirectory(prefix="fc2-compile-")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def gate(self, arch):
        res = build_all(arch, Path(self.tmp.name) / arch, self.names, compile_only=True)
        for name, (rc, log) in res.items():
            self.assertEqual(rc, 0, f"{name} sm_{arch}: {log[-2000:]}")
        regs = {}
        for name in STOCK:
            key = re.search(r'ptxas_key.*?return "([^"]+)"', variant_header(name).read_text())[1]
            info = runner.ptxas_resources(res[name][1], key)
            self.assertTrue(info, f"{name}: ptxas key {key!r} must match exactly one 'Function properties for' entry")
            self.assertEqual((info["stack_bytes"], info["spill_store_bytes"], info["spill_load_bytes"]), (0, 0, 0), name)
            regs[name] = info["ptxas_regs"]
        self.assertEqual(regs["off"], regs["tmpl"], "tmpl is the same code as off; register counts must agree")
        return regs

    def test_sm_120a(self):
        self.assertGreater(self.gate("120a")["off"], 0)

    def test_sm_121a(self):
        self.assertGreater(self.gate("121a")["off"], 0)


# ---------------------------------------------------------------------------------------------------- GPU tests
def gpu_run(binary, args, timeout=300):
    cmd = [str(binary), "--warmup", "1", "--reps", "3", "--iters", "3", "--pools", "3", "--", *map(str, args)]
    with open(LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    rows = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return r.returncode, (json.loads(rows[-1]) if rows else None), r.stdout + r.stderr


@unittest.skipIf(os.environ.get("FC2_SKIP_GPU"), "FC2_SKIP_GPU set")
class GpuTests(unittest.TestCase):
    UNIFORM = [256, "--experts", 64, "--pattern", "uniform"]
    SKEW_HOLES = [200, "--experts", 64, "--pattern", "skew", "--holes", 1]
    TINY = [8, "--experts", 16]

    @classmethod
    def setUpClass(cls):
        free = gpu_free_mib()
        if free < NEEDED_FREE_MIB:
            raise unittest.SkipTest(f"only {free} MiB GPU memory free; need {NEEDED_FREE_MIB}")
        if not Path(NVCC).exists():
            raise unittest.SkipTest(f"{NVCC} missing")
        cls.tmp = tempfile.TemporaryDirectory(prefix="fc2-gpu-")
        cls.bin = Path(cls.tmp.name)
        res = build_all("120a", cls.bin, STOCK + CONTROLS, compile_only=False)
        bad = {n: log[-1500:] for n, (rc, log) in res.items() if rc}
        if bad:
            raise RuntimeError(f"sm_120a builds failed: {bad}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def exact(self, name, args):
        rc, row, text = gpu_run(self.bin / name, args)
        self.assertIsNotNone(row, text)
        self.assertEqual((rc, row["status"]), (0, "EXACT"), text)
        self.assertEqual((row["mismatch_bytes"], row["timed_mismatch_bytes"], row["integrity_errors"]), (0, 0, 0), row)
        self.assertTrue(row["launch_valid"], row)
        self.assertEqual(row["kernel_nodes"], 4, "one k_fc2_tc launch per column group")
        self.assertEqual(row["compare_mode"], "live-rows-exact-dead-rows-pristine", row)
        self.assertGreater(row["timed_checks"], 0, row)
        return row

    def test_off_and_tmpl_exact_on_every_shape(self):
        regs = set()
        for name in STOCK:
            for args in (self.UNIFORM, self.SKEW_HOLES, self.TINY):
                row = self.exact(name, args)
                self.assertEqual(row["output_bytes"] % (5120 * 2), 0)
                self.assertGreater(row["occupancy_pct"], 0)
                self.assertEqual((row["block_threads"], row["dynamic_smem"]), (128, 43520), row)
                regs.add(row["regs"])
        self.assertEqual(len(regs), 1, "off and tmpl must report the same register count")

    def test_metadata_records_routing(self):
        _, row, _ = gpu_run(self.bin / "off", self.SKEW_HOLES)
        m = row["adapter_metadata"]
        self.assertEqual(m["input_provenance"], "synthetic")
        for k in ("n_slots", "n_tiles", "tile_bound", "live_routes", "distinct_experts"):
            self.assertIn(k, m)
        self.assertLessEqual(m["n_tiles"], m["tile_bound"])
        self.assertLess(m["live_routes"], 200 * 6, "holes=1 cuts routes")

    def control_caught(self, name, args):
        rc, row, text = gpu_run(self.bin / name, args)
        self.assertIsNotNone(row, text)
        self.assertEqual((rc, row["status"]), (1, "MISMATCH"), f"{name} was NOT caught: {row}")
        return row

    def test_ulp_flip_is_caught_with_exactly_one_byte(self):
        for args in (self.UNIFORM, self.SKEW_HOLES):
            row = self.control_caught("ulp_flip", args)
            self.assertEqual(row["mismatch_bytes"], 1, row)

    def test_dead_row_write_is_caught(self):
        for args in (self.UNIFORM, self.SKEW_HOLES):
            row = self.control_caught("dead_row_write", args)
            self.assertEqual(row["mismatch_bytes"], 2, row)   # one fp16 1.0 (0x3c00) over two poison bytes

    def test_compute_once_is_caught_by_fresh_inputs(self):
        row = self.control_caught("compute_once", self.UNIFORM)
        self.assertGreater(row["timed_mismatch_bytes"], 0, row)

    def test_skip_group_is_caught(self):
        row = self.control_caught("skip_group", self.UNIFORM)
        self.assertGreater(row["mismatch_bytes"], 10000, row)   # the last quarter of every live row

    def test_invalid_shape_is_refused(self):
        for bad in ([0], [8193], [64, "--experts", 5], [64, "--pattern", "zipf"], [64, "--bogus", 1]):
            rc, row, text = gpu_run(self.bin / "off", bad)
            self.assertEqual(rc, 2, (bad, text))
            self.assertIn("INVALID_CASE", text)


# -------------------------------------------------------------------------------------------------- run.py E2E
@unittest.skipIf(os.environ.get("FC2_SKIP_GPU"), "FC2_SKIP_GPU set")
class EndToEnd(unittest.TestCase):
    def test_run_py_smoke_suite_fills_the_new_columns(self):
        if gpu_free_mib() < NEEDED_FREE_MIB or not Path(NVCC).exists():
            self.skipTest("GPU memory or nvcc unavailable")
        with tempfile.TemporaryDirectory(prefix="fc2-e2e-") as d:
            out = Path(d) / "run"
            common = [sys.executable, str(SWEEP / "run.py"), "--arch", "120a", "--nvcc", NVCC, "--out", str(out)]
            b = subprocess.run(common + ["--kernel", str(HERE / "fc2.json"), "--suite", "smoke", "--variants", "off", "tmpl",
                                         "--build-only"], capture_output=True, text=True, timeout=1500)
            self.assertEqual(b.returncode, 0, b.stdout[-1500:] + b.stderr[-1500:])
            with open(LOCK, "w") as lock:        # builds above ran unlocked; only the measurement holds the GPU lock
                fcntl.flock(lock, fcntl.LOCK_EX)
                r = subprocess.run(common + ["--run-only", "--warmup", "1", "--reps", "2", "--iters", "2", "--pools", "3",
                                             "--timeout", "300"], capture_output=True, text=True, timeout=1800)
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            tables = sorted(out.rglob("summary.csv"))
            self.assertTrue(tables, sorted(p.name for p in out.iterdir()))
            with tables[0].open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["variant"] for r in rows}, {"off", "tmpl"})
            self.assertEqual({r["case"] for r in rows}, {c["id"] for c in fc2_cases.cases("smoke")})
            for r in rows:
                self.assertEqual(r["status"], "EXACT", r)
                for col in ("regs", "ptxas_regs", "stack_bytes", "spill_store_bytes", "dynamic_smem", "static_smem"):
                    self.assertNotEqual(r[col], "", (col, r))
                self.assertEqual(r["compare_mode"], "live-rows-exact-dead-rows-pristine", r)
                self.assertEqual(r["block_threads"], "128", r)
                self.assertGreater(float(r["occupancy_pct"]), 0, r)
                self.assertGreaterEqual(int(float(r["max_active_blocks_per_sm"])), 1, r)
                self.assertEqual(r["spill_store_bytes"], "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
