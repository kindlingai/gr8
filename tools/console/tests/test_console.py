import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.console.server.control import ComposeControl

from fastapi.testclient import TestClient

from tools.console.server import envfile, metrics, presets, weights
from tools.console.server.main import create_app
from tools.console.server.service import Console, fetch, rpc
from tools.console.tests.fakes import Fixture


class EnvironmentTests(unittest.TestCase):
    def test_patch_preserves_unrelated_lines_and_extra_env(self):
        text = "# note\nMODEL=unchanged # comment\nEXTRA_ENV='KEEP=42 GR8_QUIT_FAST=0'\nSESS=4\n"
        engine = {"GR8_QUIT_FAST"}
        changed = envfile.patch(text, {"GR8_QUIT_FAST": "1", "REUSE": "1", "SESS": None}, engine)
        self.assertIn("MODEL=unchanged # comment", changed)
        self.assertIn("KEEP=42", changed)
        self.assertNotIn("SESS=", changed)
        again = envfile.engine_env(envfile.parse(changed))
        self.assertEqual(again["KEEP"], "42")
        self.assertEqual(again["GR8_QUIT_FAST"], "1")
        self.assertEqual(envfile.read_values(changed, {"GR8_QUIT_FAST", "REUSE", "SESS"}, engine)["GR8_QUIT_FAST"], "1")

    def test_patch_strips_only_named_extra_env_words(self):
        text = "EXTRA_ENV='KEEP_ME=1 GR8_DS_ENGRAM=1 GR8_DS_ENGRAM_PREFIX=/opt/engram GR8_DS_MOE_TOPK_DECODE=4 GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1'\nEXACT=1\n"
        changed = envfile.patch(text, {"GR8_DS_MOE_TOPK_DECODE": None, "GR8_DS_RELAXED_ACCEPT": None, "EXACT": None, "TUNE": "quality"},
                                 {"GR8_DS_MOE_TOPK_DECODE", "GR8_DS_RELAXED_ACCEPT"})
        self.assertNotIn("EXACT=", changed)
        self.assertNotIn("MOE_TOPK_DECODE", changed)
        self.assertNotIn("RELAXED_ACCEPT", changed)
        self.assertIn("TUNE=quality", changed)
        extra = envfile.engine_env(envfile.parse(changed))
        self.assertEqual(extra, {"KEEP_ME": "1", "GR8_DS_ENGRAM": "1", "GR8_DS_ENGRAM_PREFIX": "/opt/engram"})
        untouched = "EXTRA_ENV='KEEP_ME=1'\n"
        self.assertEqual(envfile.patch(untouched, {"GR8_CP": None, "TUNE": "speed"}, {"GR8_CP"}), "EXTRA_ENV='KEEP_ME=1'\nTUNE=speed\n")

    def test_duplicate_and_ambiguous_rejected(self):
        with self.assertRaises(ValueError):
            envfile.parse("SESS=4\nSESS=8\n")
        with self.assertRaises(ValueError):
            envfile.engine_env({"EXTRA_ENV": "X=1 X=2"})

    def test_atomic_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SESS=4\n")
            path.chmod(0o640)
            old = envfile.transact(path, {"action": "read"})
            envfile.transact(path, {"action": "write", "expected": old["revision"], "text": "SESS=8\n"})
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            with self.assertRaisesRegex(ValueError, "changed"):
                envfile.transact(path, {"action": "write", "expected": old["revision"], "text": "oops"})
            self.assertEqual(path.read_text(), "SESS=8\n")
            path.write_bytes(b"SESS=4\r\n")
            self.assertEqual(envfile.transact(path, {"action": "read"})["text"], "SESS=4\r\n")

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "original"
            target.write_text("SESS=4")
            link = Path(tmp) / "link"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                envfile.transact(link, {"action": "read"})

    def test_preset_signatures_match_only_the_documented_keys(self):
        self.assertEqual(presets.classify("glm53", {}), "speed")
        self.assertEqual(presets.classify("glm53", {"VARIANT": "bit325"}), "quality")
        self.assertEqual(presets.classify("glm53", {"KV_FORMAT": "int4"}), "context")
        self.assertEqual(presets.classify("glm53", {"VARIANT": "bit325", "KV_FORMAT": "int4"}), "custom")
        self.assertEqual(presets.classify("glm53", {"TUNE": "speed"}), "custom")
        self.assertEqual(presets.classify("ds41", {}), "speed")
        self.assertEqual(presets.classify("glm53f", {}), "speed")
        self.assertEqual(presets.classify("ds41", {"TUNE": "context"}), "context")
        self.assertEqual(presets.classify("ds41", {"TUNE": "speed", "VARIANT": "home"}), "custom")
        self.assertEqual(presets.classify("glm53f", {"TUNE": "quality"}), "quality")
        self.assertEqual(presets.classify("ds41", {"SESS": "4"}), "custom")
        self.assertEqual(presets.classify("glm53f", {"CAP": "9"}), "speed")
        self.assertEqual(presets.classify("glm53", {"SESS": "4"}), "custom")
        self.assertEqual(presets.classify("glm53", {"MAX_MODEL_LEN": "100"}), "custom")
        overrides = {"TUNE": "speed", "SESS": "8", "CAP": "1", "MAX_MODEL_LEN": "9", "VARIANT": "home"}
        self.assertEqual(presets.classify("ds41", overrides), "custom")
        self.assertEqual(presets.classify("glm53f", overrides), "custom")
        self.assertIn("SESS=8", presets.describe("ds41", overrides, False, "custom"))
        self.assertEqual(presets.classify("glm53", {"KV_FORMAT": "int4", "CAP": "100"}), "context")
        self.assertEqual(presets.classify("glm53", {"CAP": "409600"}), "speed")
        self.assertEqual(presets.classify("glm53", {"VARIANT": "bit325", "CAP": "409600"}), "quality")
        self.assertEqual(presets.classify("ds41", {"TUNE": "context", "CAP": "409600"}), "context")
        self.assertEqual(presets.classify("ds41", {"TUNE": "context", "CAP": "409600", "SESS": "4"}), "custom")
        self.assertNotIn("CAP", presets.describe("glm53", {"KV_FORMAT": "int4", "CAP": "100", "SESS": "4"}, False, "custom"))
        self.assertEqual(presets.classify("glm53", {"VARIANT": "bit325", "KV_FORMAT": "int4", "CAP": "999", "TUNE": "speed"}), "custom")
        blank = {"TUNE": None, "VARIANT": None, "KV_FORMAT": None, "SESS": None, "MAX_MODEL_LEN": None}
        self.assertEqual(presets.changes_for("glm53", "speed"), blank)
        self.assertEqual(presets.changes_for("glm53", "quality"), {**blank, "VARIANT": "bit325"})
        self.assertEqual(presets.changes_for("glm53", "context"), {**blank, "KV_FORMAT": "int4"})
        quality = next(item for item in presets.CATALOG["glm53"] if item["id"] == "quality")
        self.assertIn("same 3.25 bpw size", quality["summary"])
        self.assertIn("bits placed differently", quality["summary"])
        self.assertEqual(quality["caption"], "8 slots · same size as home")
        speed = next(item for item in presets.CATALOG["glm53"] if item["id"] == "speed")
        self.assertIn("DFlash2 or MTP drafts", speed["summary"])
        self.assertNotIn("DFlash2 + MTP", speed["summary"])
        context = next(item for item in presets.CATALOG["glm53"] if item["id"] == "context")
        self.assertEqual(context["caption"], "8 slots · int4 KV (not re-measured)")
        for item in presets.CATALOG["glm53"]:
            self.assertNotIn("485,376", item["summary"] + item["caption"])
            self.assertNotIn("513,536", item["summary"] + item["caption"])
        self.assertEqual(presets.classify("ds41", {"GR8_DS_MOE_TOPK_DECODE": "4"}), "custom")
        self.assertEqual(presets.classify("ds41", {"EXACT": "1"}), "custom")
        self.assertEqual(presets.classify("ds41", {"GR8_DS_ENGRAM": "1"}), "speed")
        self.assertEqual(presets.classify("glm53f", {"GR8_CP": "0"}), "custom")
        self.assertEqual(presets.classify("glm53f", {"GR8_CP": "1"}), "custom")
        watched = {"GR8_DS_MOE_TOPK_DECODE": "4", "EXACT": "1", "GR8_CP": "1"}
        deep = {row["key"]: row for row in presets.diff_for("ds41", watched, (), "quality")}
        self.assertEqual(deep["GR8_DS_MOE_TOPK_DECODE"], {"key": "GR8_DS_MOE_TOPK_DECODE", "before": "4", "after": None})
        self.assertEqual(deep["EXACT"], {"key": "EXACT", "before": "1", "after": None})
        self.assertNotIn("GR8_CP", deep)
        self.assertNotIn("GR8_DS_ENGRAM", deep)
        flash = {row["key"]: row for row in presets.diff_for("glm53f", watched, (), "speed")}
        self.assertEqual(flash["GR8_CP"], {"key": "GR8_CP", "before": "1", "after": None})
        for model in presets.CATALOG:
            for name in ("speed", "quality", "context"):
                assign = presets.changes_for(model, name)
                for key in ("SESS", "MAX_MODEL_LEN"):
                    self.assertIsNone(assign[key])
                self.assertNotIn("CAP", assign)
                self.assertNotIn("CAP", presets.keys_for(model))
                for values in ({"CAP": "409600"}, {"CAP": "1", "SESS": "3"}):
                    self.assertNotIn("CAP", [row["key"] for row in presets.diff_for(model, values, ("CAP",), name)])
        cleared = ("SESS", "MAX_MODEL_LEN", "VARIANT", "KV_FORMAT")
        for model in ("ds41", "glm53f"):
            for name in ("speed", "quality", "context"):
                assign = presets.changes_for(model, name)
                self.assertEqual(assign["TUNE"], name)
                for key in cleared:
                    self.assertIsNone(assign[key])
        sess = next(row for row in presets.diff_for("ds41", {"TUNE": "speed", "SESS": "8"}, (), "quality") if row["key"] == "SESS")
        self.assertEqual(sess, {"key": "SESS", "before": "8", "after": None})
        mixed = next(row for row in presets.diff_for("ds41", {"SESS": "8"}, {"SESS"}, "quality") if row["key"] == "SESS")
        self.assertEqual(mixed["before"], "mixed")
        with self.assertRaises(ValueError):
            presets.changes_for("glm53", "fast")
        for model in presets.CATALOG:
            keys = set(presets.keys_for(model))
            for item in presets.CATALOG[model]:
                self.assertEqual(set(item["assign"]), keys)


class AdapterTests(unittest.TestCase):
    def test_switch_command_and_dotenv_rpc_are_reused_without_shell_input(self):
        calls = []
        def runner(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stdout='{"revision":"r","text":"SESS=4"}' if args[0] == "ssh" else "health: ready")
        cfg = SimpleNamespace(models={"glm53": SimpleNamespace(directory="/configured compose", boxes=("operator@box.example",))}, timeout=10, ssh_timeout=5)
        module = SimpleNamespace(read_config=lambda _: cfg, phase=lambda *args, **kwargs: calls.append(("phase", args, kwargs)))
        with patch("tools.console.server.control.load_switch", return_value=module), patch.object(Path, "read_bytes", return_value=b"config") as config_bytes:
            adapter = ComposeControl("switch.env", runner=runner)
            adapter.environment("glm53", "operator@box.example")
            args, kwargs = calls[-1]
            self.assertEqual(args[:3], ["ssh", "-o", "BatchMode=yes"])
            self.assertNotIn("shell", kwargs)
            data = json.loads(kwargs["input"])
            self.assertEqual(data["path"], "/configured compose/.env")
            self.assertEqual(data["action"], "read")
            adapter.stop("glm53", lambda _: None)
            self.assertEqual(calls[-1][0], "phase")
            adapter.activate("glm53", lambda _: None)
            args, _ = calls[-1]
            self.assertEqual(args[1:3], ["switch", "glm53"])
            self.assertTrue(args[3].startswith("@"))
            self.assertNotIn("--timeout", args)
            config_bytes.return_value = b"changed config"
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                adapter.activate("glm53", lambda _: None)

    def test_download_extends_the_switch_health_timeout(self):
        calls = []
        def runner(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stdout="health: ready")
        cfg = SimpleNamespace(models={"glm53": SimpleNamespace(directory="/configured compose", boxes=("operator@box.example",))}, timeout=900, ssh_timeout=5)
        module = SimpleNamespace(read_config=lambda _: cfg, phase=lambda *args, **kwargs: None)
        with patch("tools.console.server.control.load_switch", return_value=module), patch.object(Path, "read_bytes", return_value=b"config"):
            adapter = ComposeControl("switch.env", runner=runner)
            adapter.activate("glm53", lambda _: None, health_timeout=1961)
        args, kwargs = calls[-1]
        self.assertEqual(args[1:6], ["switch", "--timeout", "1961", "glm53", args[5]])
        self.assertTrue(args[5].startswith("@"))
        self.assertGreaterEqual(kwargs["timeout"], 1961)
        self.assertEqual(kwargs["timeout"], 1961 + 5 * 4 + 30)

    def test_probe_error_is_unknown_not_a_download(self):
        def runner(args, **kwargs):
            if args[0] != "ssh":
                return SimpleNamespace(returncode=0, stdout="ok")
            box = args[-2]
            if box.endswith("down.example"):
                raise OSError("ssh failed")
            if box.endswith("exit.example"):
                return SimpleNamespace(returncode=1, stdout="probe failed")
            return SimpleNamespace(returncode=0, stdout="not-json")

        boxes = ("op@down.example", "op@exit.example", "op@bad.example")
        cfg = SimpleNamespace(models={"glm53": SimpleNamespace(directory="/configured compose", boxes=boxes)}, timeout=900, ssh_timeout=5)
        module = SimpleNamespace(read_config=lambda _: cfg, phase=lambda *args, **kwargs: None)
        records = {box: {"text": "MODEL=https://huggingface.co/org/repo\nTP=4\n", "revision": "r"} for box in boxes}
        with patch("tools.console.server.control.load_switch", return_value=module), patch.object(Path, "read_bytes", return_value=b"config"):
            adapter = ComposeControl("switch.env", runner=runner)
            one = adapter.variant_cache("glm53", "bit325", {boxes[0]: records[boxes[0]]}, refresh=True)
            status = adapter.variant_cache("glm53", "bit325", records, refresh=True)
        self.assertEqual(one["state"], "unknown")
        self.assertEqual(one["unchecked"], [1])
        self.assertEqual(weights.preset_notice(one, 900), ("Cannot check the weights on box 1.", None))
        self.assertEqual(status["state"], "unknown")
        self.assertFalse(status["present"])
        self.assertEqual(status["unchecked"], [1, 2, 3])
        text, health = weights.preset_notice(status, 900)
        self.assertIsNone(health)
        self.assertEqual(text, "Cannot check the weights on box 1, box 2, and box 3.")
        self.assertNotIn("download", text)

    def test_nonzero_switch_is_failure(self):
        cfg = SimpleNamespace(timeout=10, ssh_timeout=5)
        with patch("tools.console.server.control.load_switch", return_value=SimpleNamespace(read_config=lambda _: cfg)), patch.object(Path, "read_bytes", return_value=b"config"):
            adapter = ComposeControl("switch.env", runner=lambda *a, **k: SimpleNamespace(returncode=7, stdout="not ready"))
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                adapter.activate("glm53", lambda _: None)


class MetricTests(unittest.TestCase):
    def test_exact_names_labels_and_denominators(self):
        values, info = metrics.parse('''# comment
vllm:generation_tokens_total{model_name="a"} 200
vllm:spec_decode_num_drafts_total 1
vllm:spec_decode_num_draft_tokens_total 10
vllm:spec_decode_num_accepted_tokens_total 7
vllm:request_success_total{finished_reason="stop"} 8
vllm:request_success_total{finished_reason="error"} 2
vllm:cache_config_info{kv_cache_max_concurrency="8"} 1
vllm:bad NaN
''')
        out = metrics.summarize(values, {"vllm:generation_tokens_total": 100}, 2)
        self.assertEqual(out["decode"], 50)
        self.assertEqual(out["draft_accept"], .7)
        self.assertEqual(out["finished"], 10)
        self.assertEqual(info["slots"], 8)
        self.assertNotIn("vllm:bad", values)
        self.assertIsNone(out["kv_used"])

    def test_reset_missing_and_zero_are_unknown(self):
        for prev in (None, {"vllm:generation_tokens_total": 200}):
            self.assertIsNone(metrics.summarize({"vllm:generation_tokens_total": 100}, prev, 2)["decode"])
        self.assertIsNone(metrics.summarize({"vllm:prefix_cache_queries_total": 0})["prefix_hits"])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.fake = Fixture()
        self.console = Console(self.fake.console_config, self.fake)

    def tearDown(self):
        self.console.close()
        self.fake.close()

    def done(self):
        # Queue a barrier behind the operation instead of polling or sleeping.
        self.console.pool.submit(lambda: None).result(timeout=5)
        return self.console.operation()

    def request(self, **extra):
        return dict(action="restart", model="glm53", confirm=True, expect_active="glm53", **extra)

    def test_real_http_and_line_json_protocol(self):
        snap = self.console.snapshot()
        self.assertEqual(snap["active"], "glm53")
        self.assertEqual(snap["stats"]["kv_size"], 65536)
        self.assertIsNone(snap["stats"]["decode"])
        self.assertEqual(len(snap["boxes"]), 4)
        self.assertIn("SYNTHETIC", snap["logs"])
        self.assertEqual([r["state"] for r in snap["models"]].count("ready"), 1)

    def test_read_presets_does_not_write_and_mixed_is_custom(self):
        self.fake.texts["glm53"]["box-1"] += "VARIANT=bit325\n"
        row = self.console.read_presets("glm53")
        self.assertEqual(row["current"], "custom")
        self.assertIn("VARIANT", row["mixed"])
        self.assertEqual(row["current_setting"], "mixed across boxes")
        self.assertEqual(self.fake.calls, [])
        self.fake.texts["ds41"]["box-0"] += "TUNE=quality\n"
        deep = self.console.read_presets("ds41")
        self.assertEqual(deep["current"], "custom")

    def test_unset_recipe_tune_reads_as_speed_until_sess_is_set(self):
        for model in ("ds41", "glm53f"):
            for box in self.fake.texts[model]:
                self.fake.texts[model][box] = self.fake.texts[model][box].replace("SESS=4\n", "")
            self.assertEqual(self.console.read_presets(model)["current"], "speed")
            for box in self.fake.texts[model]:
                self.fake.texts[model][box] += "SESS=4\n"
            self.assertEqual(self.console.read_presets(model)["current"], "custom")

    def test_recipe_preset_removes_explicit_overrides(self):
        extra = "TUNE=speed\nSESS=8\nCAP=409600\nMAX_MODEL_LEN=9\nVARIANT=home\n"
        for model in ("ds41", "glm53f"):
            for box in self.fake.texts[model]:
                text = self.fake.texts[model][box].replace("SESS=4\n", extra)
                self.assertNotEqual(text, self.fake.texts[model][box])
                self.fake.texts[model][box] = text
        row = self.console.read_presets("ds41")
        self.assertEqual(row["current"], "custom")
        self.assertIn("SESS=8", row["current_setting"])
        removal = next(item for item in next(p for p in row["presets"] if p["id"] == "quality")["diff"] if item["key"] == "SESS")
        self.assertEqual(removal, {"key": "SESS", "before": "8", "after": None})
        for item in row["presets"]:
            self.assertNotIn("CAP", [change["key"] for change in item["diff"]])
        self.assertNotIn("CAP", row["current_setting"])
        self.console.submit(dict(action="restart", model="ds41", confirm=True, expect_active="glm53", preset="quality", revision=row["revision"]))
        self.assertEqual(self.done()["status"], "succeeded")
        for text in self.fake.texts["ds41"].values():
            self.assertIn("TUNE=quality", text)
            self.assertIn("KEEP_ME=1", text)
            self.assertIn("CAP=409600", text)
            for key in ("SESS", "MAX_MODEL_LEN", "VARIANT"):
                self.assertNotIn(key + "=", text)
        self.assertEqual(self.console.read_presets("ds41")["current"], "quality")
        flash = self.console.read_presets("glm53f")
        self.assertEqual(flash["current"], "custom")
        self.console.submit(dict(action="restart", model="glm53f", confirm=True, expect_active="ds41", preset="speed", revision=flash["revision"]))
        self.assertEqual(self.done()["status"], "succeeded")
        for text in self.fake.texts["glm53f"].values():
            self.assertIn("TUNE=speed", text)
            self.assertIn("KEEP_ME=1", text)
            self.assertIn("CAP=409600", text)
            for key in ("SESS", "MAX_MODEL_LEN", "VARIANT"):
                self.assertNotIn(key + "=", text)
        self.assertEqual(self.console.read_presets("glm53f")["current"], "speed")

    def test_ds41_preset_strips_fast_words_and_exact(self):
        words = " ".join([
            "KEEP_ME=1", "GR8_DS_ENGRAM=1", "GR8_DS_ENGRAM_PREFIX=/opt/engram",
            "GR8_DS_RELAXED_ACCEPT=topp:0.9,ratio:0.1", "GR8_DS_MOE_TOPK_DECODE=4", "GR8_DS41_HEAD_MX=1",
            "GR8_DS_ATTN_TC_MF=2", "GR8_DS_ATTN_TC_TILE=1", "GR8_DS_MHC_ROW_SHARD=1", "GR8_DS_MHC_Y_MXFP8=1",
            "GR8_DS_CACHE_MIB=2048", "GR8_DS_SESS_DRAFTS=1",
        ])
        for box in self.fake.texts["ds41"]:
            text = self.fake.texts["ds41"][box].replace("EXTRA_ENV='KEEP_ME=1'", f"EXTRA_ENV='{words}'")
            self.assertNotEqual(text, self.fake.texts["ds41"][box])
            self.fake.texts["ds41"][box] = text + "EXACT=1\n"
        row = self.console.read_presets("ds41")
        self.assertEqual(row["current"], "custom")
        diff = {item["key"]: item for item in next(p for p in row["presets"] if p["id"] == "quality")["diff"]}
        self.assertEqual(diff["GR8_DS_MOE_TOPK_DECODE"], {"key": "GR8_DS_MOE_TOPK_DECODE", "before": "4", "after": None})
        self.assertEqual(diff["EXACT"], {"key": "EXACT", "before": "1", "after": None})
        self.assertEqual(diff["GR8_DS_RELAXED_ACCEPT"]["before"], "topp:0.9,ratio:0.1")
        self.assertNotIn("GR8_DS_ENGRAM", diff)
        self.console.submit(dict(action="restart", model="ds41", confirm=True, expect_active="glm53", preset="quality", revision=row["revision"]))
        self.assertEqual(self.done()["status"], "succeeded")
        for text in self.fake.texts["ds41"].values():
            self.assertIn("TUNE=quality", text)
            self.assertNotIn("EXACT=", text)
            extra = envfile.engine_env(envfile.parse(text))
            self.assertEqual(extra, {"KEEP_ME": "1", "GR8_DS_ENGRAM": "1", "GR8_DS_ENGRAM_PREFIX": "/opt/engram"})
        self.assertEqual(self.console.read_presets("ds41")["current"], "quality")

    def test_flash_preset_strips_gr8_cp(self):
        for box in self.fake.texts["glm53f"]:
            text = self.fake.texts["glm53f"][box].replace("EXTRA_ENV='KEEP_ME=1'", "EXTRA_ENV='KEEP_ME=1 GR8_CP=0'")
            self.fake.texts["glm53f"][box] = text.replace("SESS=4\n", "")
        row = self.console.read_presets("glm53f")
        self.assertEqual(row["current"], "custom")
        diff = {item["key"]: item for item in next(p for p in row["presets"] if p["id"] == "context")["diff"]}
        self.assertEqual(diff["GR8_CP"], {"key": "GR8_CP", "before": "0", "after": None})
        self.console.submit(dict(action="restart", model="glm53f", confirm=True, expect_active="glm53", preset="speed", revision=row["revision"]))
        self.assertEqual(self.done()["status"], "succeeded")
        for text in self.fake.texts["glm53f"].values():
            self.assertIn("TUNE=speed", text)
            extra = envfile.engine_env(envfile.parse(text))
            self.assertEqual(extra, {"KEEP_ME": "1"})
            self.assertNotIn("GR8_CP", text)
        self.assertEqual(self.console.read_presets("glm53f")["current"], "speed")

    def test_glm_presets_clear_the_other_format_and_keep_cap(self):
        for box in self.fake.texts["glm53"]:
            self.fake.texts["glm53"][box] += "TUNE=speed\nVARIANT=bit325\nKV_FORMAT=int4\nCAP=999\nMAX_MODEL_LEN=777\n"
        row = self.console.read_presets("glm53")
        self.assertEqual(row["current"], "custom")
        self.assertNotIn("CAP", row["current_setting"])
        for preset, present, absent in (
            ("speed", ("CAP=999",), ("TUNE=", "VARIANT=", "KV_FORMAT=", "SESS=", "MAX_MODEL_LEN=")),
            ("quality", ("VARIANT=bit325", "CAP=999"), ("KV_FORMAT=", "TUNE=", "SESS=", "MAX_MODEL_LEN=")),
            ("context", ("KV_FORMAT=int4", "CAP=999"), ("VARIANT=", "TUNE=", "SESS=", "MAX_MODEL_LEN=")),
        ):
            row = self.console.read_presets("glm53")
            self.console.submit(self.request(preset=preset, revision=row["revision"]))
            self.assertEqual(self.done()["status"], "succeeded")
            for text in self.fake.texts["glm53"].values():
                for needle in present:
                    self.assertIn(needle, text)
                for needle in absent:
                    self.assertNotIn(needle, text)
                self.assertIn("KEEP_ME=1", text)
            self.assertEqual(self.console.read_presets("glm53")["current"], preset)

    def test_confirmed_preset_writes_all_boxes_then_restart(self):
        row = self.console.read_presets("glm53")
        self.assertEqual(row["current"], "custom")
        removal = next(item for item in next(p for p in row["presets"] if p["id"] == "quality")["diff"] if item["key"] == "SESS")
        self.assertEqual(removal, {"key": "SESS", "before": "4", "after": None})
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        job = self.done()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual([c[0] for c in self.fake.calls], ["write"] * 4 + ["stop", "switch"])
        self.assertTrue(all("VARIANT=bit325" in t and "KEEP_ME=1" in t and "SESS=" not in t for t in self.fake.texts["glm53"].values()))
        self.assertEqual(self.console.read_presets("glm53")["current"], "quality")
        backups = list(self.console.state.glob("*.backup.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)

    def test_stale_revision_cannot_write_or_restart(self):
        row = self.console.read_presets("glm53")
        self.fake.texts["glm53"]["box-0"] += "VARIANT=home\n"
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        self.assertEqual(self.done()["status"], "failed")
        self.assertEqual(self.fake.calls, [])

    def test_partial_write_rolls_back_without_restart(self):
        old = copy.deepcopy(self.fake.texts)
        self.fake.fail_write = "box-2"
        row = self.console.read_presets("glm53")
        self.console.submit(self.request(preset="context", revision=row["revision"]))
        self.assertEqual(self.done()["status"], "failed")
        self.assertEqual(self.fake.texts, old)
        self.assertNotIn("stop", [x[0] for x in self.fake.calls])

    def test_raw_knob_map_rejected_before_writes(self):
        row = self.console.read_presets("glm53")
        with self.assertRaisesRegex(ValueError, "raw setting"):
            self.console.submit(self.request(changes={"KV_STORE": "ssd:/cache/kv,size=1G,persist=on,keep=sessions"}, revision=row["revision"]))
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            self.console.submit(self.request(preset="fast", revision=row["revision"]))
        self.assertEqual(self.fake.calls, [])

    def test_failed_boot_is_not_success(self):
        self.fake.fail_boot = True
        self.console.submit(self.request())
        job = self.done()
        self.assertEqual(job["status"], "failed")
        self.assertIn("timeout", job["error"])

    def test_controls_disabled_and_confirmation_required(self):
        with self.assertRaises(ValueError):
            self.console.submit(dict(action="stop", model="glm53"))
        self.console.config["allow_control"] = False
        with self.assertRaises(PermissionError):
            self.console.submit(self.request())
        self.assertEqual(self.fake.calls, [])

    def test_missing_variant_weights_are_confirmed_and_wait_past_900s(self):
        self.fake.missing_weights = {"variant": "bit325", "size_gb": 29.7, "rate_mb_s": 28}
        row = self.console.read_presets("glm53")
        by_id = {item["id"]: item for item in row["presets"]}
        sentence = weights.download_text(29.7, 28, self.console._health_budget() + weights.extra_seconds(29.7, 28))
        self.assertEqual(by_id["quality"]["download"], sentence)
        self.assertIn("The health wait is extended to", sentence)
        self.assertIsNone(by_id["speed"]["download"])
        self.assertIsNone(by_id["context"]["download"])
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        refused = self.done()
        self.assertEqual(refused["status"], "failed", refused)
        self.assertIn("confirm the download", refused["error"])
        self.assertEqual(self.fake.calls, [])
        self.console.submit(self.request(preset="quality", revision=row["revision"], accept_download=True))
        job = self.done()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertTrue(any(step.startswith("downloading " + sentence) for step in job["steps"]))
        switch = next(call for call in self.fake.calls if call[0] == "switch")
        self.assertEqual(switch[2], self.console._health_budget() + weights.extra_seconds(29.7, 28))
        self.assertGreater(switch[2], 900)
        self.assertEqual(self.console.read_presets("glm53")["current"], "quality")
        self.assertIsNone(next(item for item in self.console.read_presets("glm53")["presets"] if item["id"] == "quality")["download"])

    def test_unmeasured_rate_omits_minutes_and_still_extends_the_wait(self):
        self.fake.missing_weights = {"variant": "bit325", "size_gb": 29.7, "rate_mb_s": None}
        row = self.console.read_presets("glm53")
        quality = next(item for item in row["presets"] if item["id"] == "quality")
        health = self.console._health_budget() + weights.extra_seconds(29.7, None)
        self.assertEqual(quality["download"], weights.download_text(29.7, None, health))
        self.assertIn("hours", quality["download"])
        self.assertNotIn("measured rate", quality["download"])
        self.console.submit(self.request(preset="quality", revision=row["revision"], accept_download=True))
        job = self.done()
        self.assertEqual(job["status"], "succeeded", job)
        switch = next(call for call in self.fake.calls if call[0] == "switch")
        self.assertEqual(switch[2], 900 + weights.extra_seconds(29.7, None))
        self.assertGreater(switch[2], 900)

    def test_unknown_check_names_the_box_and_does_not_extend_the_wait(self):
        self.fake.missing_weights = {"unknown": True, "unchecked": [2]}
        row = self.console.read_presets("glm53")
        quality = next(item for item in row["presets"] if item["id"] == "quality")
        self.assertEqual(quality["download"], "Cannot check the weights on box 2.")
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        job = self.done()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertTrue(any(step == "Cannot check the weights on box 2." for step in job["steps"]))
        self.assertFalse(any("health wait extended" in step for step in job["steps"]))
        self.assertFalse(any("download" in step.lower() for step in job["steps"]))
        switch = next(call for call in self.fake.calls if call[0] == "switch")
        self.assertIsNone(switch[2])

    def test_dialog_shows_the_extended_health_wait(self):
        self.fake.missing_weights = {"variant": "bit325", "size_gb": 29.7, "rate_mb_s": None}
        row = self.console.read_presets("glm53")
        quality = next(item for item in row["presets"] if item["id"] == "quality")
        health = self.console._health_budget() + weights.extra_seconds(29.7, None)
        text = weights.download_text(29.7, None, health)
        self.assertEqual(quality["download"], text)
        self.assertIn("8 hours 30 minutes", text)
        self.assertNotIn("measured rate", text)
        self.console.submit(self.request(preset="quality", revision=row["revision"], accept_download=True))
        job = self.done()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertTrue(any("The health wait is extended to 8 hours 30 minutes." in step for step in job["steps"]))
        switch = next(call for call in self.fake.calls if call[0] == "switch")
        self.assertEqual(switch[2], health)

    def test_operation_serialization(self):
        release, started = threading.Event(), threading.Event()
        original = self.fake.stop
        def slow(*args):
            started.set()
            release.wait(5)
            original(*args)
        self.fake.stop = slow
        self.console.submit(self.request())
        self.assertTrue(started.wait(3))
        try:
            with self.assertRaises(RuntimeError):
                self.console.submit(self.request())
        finally:
            release.set()
        self.assertEqual(self.done()["status"], "succeeded")


    def test_changed_active_model_is_conflict_without_writes(self):
        self.fake.active = "ds41"
        with self.assertRaisesRegex(RuntimeError, "active model changed"):
            self.console.submit(self.request())
        self.assertEqual(self.fake.calls, [])

    def test_restore_after_failed_preset_restart(self):
        old = copy.deepcopy(self.fake.texts)
        row = self.console.read_presets("glm53")
        self.fake.fail_boot = True
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        job = self.done()
        self.assertEqual(job["status"], "failed")
        self.assertTrue(job["can_restore"])
        count = len(self.fake.calls)
        self.console.submit(dict(action="restore", model="glm53", confirm=True, expect_active=None, backup_id=job["backup_id"]))
        restored = self.done()
        self.assertEqual(restored["status"], "succeeded", restored)
        self.assertEqual(self.fake.texts, old)
        self.assertEqual([x[0] for x in self.fake.calls[count:]], ["write"] * 4)
        self.assertFalse(restored["can_restore"])

    def test_restore_refuses_outside_edit(self):
        row = self.console.read_presets("glm53")
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        job = self.done()
        self.fake.texts["glm53"]["box-2"] += "# outside edit\n"
        count = len(self.fake.calls)
        self.console.submit(dict(action="restore", model="glm53", confirm=True, expect_active="glm53", backup_id=job["backup_id"]))
        failed = self.done()
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["can_restore"])
        self.assertEqual(len(self.fake.calls), count)

    def test_api_csrf_and_errors(self):
        with TestClient(create_app(console=self.console), base_url="http://localhost") as client:
            self.assertEqual(client.get("/api/healthz").status_code, 200)
            self.assertEqual(client.get("/api/healthz", headers={"host": "attacker.invalid"}).status_code, 400)
            token = client.get("/api/snapshot").json()["value"]["csrf"]
            self.assertEqual(client.post("/api/operations", json=self.request()).status_code, 403)
            headers = {"x-console-token": token, "origin": "https://foreign.invalid"}
            self.assertEqual(client.post("/api/operations", headers=headers, json=self.request()).status_code, 403)
            headers.pop("origin")
            self.assertEqual(client.post("/api/operations", headers=headers, json=self.request()).status_code, 202)
            self.assertEqual(client.get("/api/not-found").status_code, 404)
            self.done()

    def test_supervisor_refusal_and_head_not_ready(self):
        self.fake.refuse_supervisor = True
        snap = self.console.snapshot(fresh=True)
        self.assertTrue(all(b["state"] == "unavailable" for b in snap["boxes"]))
        self.assertIn("not a member", snap["boxes"][0]["last_line"])
        self.assertIn("not a member", snap["logs"])
        self.fake.not_ready = True
        snap = self.console.snapshot(fresh=True)
        self.assertIsNone(snap["active"])
        self.assertEqual(snap["stats"], {})
        self.assertNotIn("ready", [m["state"] for m in snap["models"]])

    def test_no_stale_metrics_when_head_fails(self):
        self.console.snapshot()
        self.fake.active = None
        self.console.cache_at = 0
        snap = self.console.snapshot()
        self.assertIsNone(snap["active"])
        self.assertEqual(snap["stats"], {})


class WeightTests(unittest.TestCase):
    def test_sentence_uses_size_over_measured_rate(self):
        self.assertEqual(weights.download_text(29.7, 28),
                         "This preset downloads up to about 30 GB to each box. At the last measured rate this takes about 18 minutes.")
        self.assertEqual(weights.download_text(29.7, None), "This preset downloads up to about 30 GB to each box.")
        self.assertEqual(weights.download_text(None, None),
                         "This preset downloads the variant weights to each box that does not have them.")
        self.assertEqual(weights.health_sentence(1961), "The health wait is extended to 32 minutes 41 seconds.")
        self.assertIn("8 hours 30 minutes", weights.download_text(29.7, None, 900 + weights.extra_seconds(29.7, None)))
        self.assertEqual(weights.extra_seconds(29.7, 28), 1061)
        self.assertGreater(weights.extra_seconds(29.7, None), 900)
        self.assertIsNone(weights.changed_variant({}, {"VARIANT": None}))
        self.assertEqual(weights.changed_variant({}, {"VARIANT": "bit325"}), "bit325")
        self.assertEqual(weights.changed_variant({"VARIANT": "bit325"}, {"VARIANT": None}), "home")
        self.assertEqual(weights.changed_variant({"VARIANT": None}, {"VARIANT": "bit325"}, mixed=("VARIANT",)), "bit325")

    def test_complete_marker_is_weights_and_manifest_alone_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit = "a" * 40
            release = root / "cache" / "hf" / "org--repo" / commit
            (release / "manifests").mkdir(parents=True)
            (release / "complete").mkdir()
            manifest = {"hidden_size": 32, "layers": {"0": {"gate": [[2] * 16], "up": [[2] * 16], "down": [2], "shared": 0}}}
            (release / "manifests" / "exlr8-manifest-bit325.json").write_text(json.dumps(manifest))
            home = root / "home" / "out"
            home.mkdir(parents=True)
            (home / "fetch.log").write_text("fetching rank 0 of org/repo: 1.0 of 2.0 GB (28 MB/s)\n")
            request = {"cache": str(root / "cache"), "model": "https://huggingface.co/org/repo@" + commit,
                       "tp": 4, "rank": 0, "variant": "bit325", "home": str(root / "home"), "peers": []}
            missing = weights.probe(request)
            self.assertFalse(missing["present"])
            self.assertGreater(missing["size_gb"], 0)
            self.assertEqual(missing["rate_mb_s"], 28.0)
            blob = release / "experts" / "one.safetensors"
            blob.parent.mkdir()
            blob.write_bytes(b"abc")
            marker = release / "complete" / "tp4.rank0.bit325-0123456789ab.json"
            marker.write_text(json.dumps({"files": {"experts/one.safetensors": 3}}))
            self.assertTrue(weights.probe(request)["present"])
            request["rank"] = 1
            self.assertFalse(weights.probe(request)["present"])
            request["rank"] = 0
            blob.write_bytes(b"ab")
            self.assertFalse(weights.probe(request)["present"])

    def test_unset_cache_resolves_the_named_volume_or_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mount = root / "vol"
            commit = "b" * 40
            release = mount / "hf" / "org--repo" / commit
            (release / "manifests").mkdir(parents=True)
            (release / "complete").mkdir()
            manifest = {"hidden_size": 32, "layers": {"0": {"gate": [[2] * 16], "up": [[2] * 16], "down": [2], "shared": 0}}}
            (release / "manifests" / "exlr8-manifest-bit325.json").write_text(json.dumps(manifest))
            blob = release / "experts" / "one.safetensors"
            blob.parent.mkdir()
            blob.write_bytes(b"abc")
            for rank in range(4):
                (release / "complete" / f"tp4.rank{rank}.bit325-0123456789ab.json").write_text(
                    json.dumps({"files": {"experts/one.safetensors": 3}}))
            home = root / "home"
            (home / "out").mkdir(parents=True)
            request = {"cache": "", "model": "https://huggingface.co/org/repo@" + commit,
                       "tp": 4, "rank": 0, "variant": "bit325", "home": str(home), "peers": []}

            def inspect(name):
                called.append(name)
                return str(mount) if name == "gr8_cache" else None

            called = []
            with patch.object(weights, "volume_mountpoint", side_effect=inspect):
                found = weights.probe(request)
            self.assertEqual(called, ["gr8_cache"])
            self.assertEqual(found["state"], "present")
            rows = []
            for rank in range(4):
                request["rank"] = rank
                with patch.object(weights, "volume_mountpoint", return_value=str(mount)):
                    probed = weights.probe(request)
                probed["box"] = rank + 1
                rows.append(probed)
            self.assertEqual(weights.aggregate(rows)["state"], "present")
            self.assertEqual(weights.preset_notice(weights.aggregate(rows), 900), (None, None))
            (release / "complete" / "tp4.rank2.bit325-0123456789ab.json").unlink()
            rows = []
            for rank in range(4):
                request["rank"] = rank
                with patch.object(weights, "volume_mountpoint", return_value=str(mount)):
                    probed = weights.probe(request)
                probed["box"] = rank + 1
                rows.append(probed)
            status = weights.aggregate(rows)
            self.assertEqual([row["state"] for row in rows], ["present", "present", "missing", "present"])
            self.assertEqual(status["state"], "missing")
            text, health = weights.preset_notice(status, 900)
            self.assertIn("up to about", text)
            self.assertGreater(health, 900)
            request["rank"] = 0
            with patch.object(weights, "volume_mountpoint", return_value=None):
                unknown = weights.probe(request)
            self.assertEqual(unknown["state"], "unknown")
            unknown["box"] = 1
            text, health = weights.preset_notice(weights.aggregate([unknown]), 900)
            self.assertIsNone(health)
            self.assertEqual(text, "Cannot check the weights on box 1.")
            self.assertNotIn("GB", text)
            request["cache"] = "rel/cache"
            with patch.object(weights, "volume_mountpoint", side_effect=AssertionError("docker")):
                self.assertEqual(weights.probe(request)["state"], "unknown")
            request["cache"] = str(mount)
            request["home"] = ""
            with patch.object(weights, "volume_mountpoint", return_value=None) as mounted:
                row = weights.probe(request)
            mounted.assert_called_with("gr8_home")
            self.assertEqual(row["state"], "present")
            self.assertIsNone(row["rate_mb_s"])

    def test_unreadable_cache_is_unknown_not_missing(self):
        # A docker-group SSH user resolves the volume mountpoint under /var/lib/docker (710 root) but cannot read it.
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            release = cache / "hf" / "org--repo" / commit
            (release / "complete").mkdir(parents=True)
            request = {"cache": str(cache), "model": "https://huggingface.co/org/repo@" + commit,
                       "tp": 4, "rank": 0, "variant": "bit325", "home": "", "peers": []}
            self.assertEqual(weights.probe(request)["state"], "missing")
            with patch.object(weights.os, "access", return_value=False):
                denied = weights.probe(request)
            self.assertEqual(denied["state"], "unknown")
            self.assertIsNone(denied["size_gb"])
            with patch.object(weights.os, "listdir", side_effect=PermissionError("denied")):
                self.assertEqual(weights.probe(request)["state"], "unknown")
            request["model"] = "https://huggingface.co/org/repo"
            with patch("builtins.open", side_effect=PermissionError("denied")):
                self.assertEqual(weights.probe(request)["state"], "unknown")
            if os.geteuid() != 0:
                cache.chmod(0)
                try:
                    self.assertEqual(weights.probe(request)["state"], "unknown")
                    request["model"] = "https://huggingface.co/org/repo@" + commit
                    self.assertEqual(weights.probe(request)["state"], "unknown")
                finally:
                    cache.chmod(0o755)
            denied["box"] = 2
            text, health = weights.preset_notice(weights.aggregate([denied]), 900)
            self.assertEqual(text, "Cannot check the weights on box 2.")
            self.assertIsNone(health)

    def test_expert_gb_matches_the_launcher(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "launch"))
        import modelset
        manifest = {"hidden_size": 32, "layers": {"0": {"gate": [[2] * 16, [4] * 16], "up": [[3] * 16, [2] * 16], "down": [4, 3], "shared": 0},
                                                   "1": {"gate": [[2] * 16], "up": [[2] * 16], "down": [2], "shared": {"gate": [3] * 16, "up": [3] * 16, "down": 4}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exlr8-manifest-bit325.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(weights.expert_gb(manifest, 4, 32), modelset.expert_gb(str(path), 4))


if __name__ == "__main__":
    unittest.main()
