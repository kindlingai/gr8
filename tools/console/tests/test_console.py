import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.console.server.control import ComposeControl

from fastapi.testclient import TestClient

from tools.console.server import envfile, metrics, presets
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
        self.assertEqual(presets.classify("glm53f", {"CAP": "9"}), "custom")
        self.assertEqual(presets.classify("glm53", {"SESS": "4"}), "custom")
        self.assertEqual(presets.classify("glm53", {"MAX_MODEL_LEN": "100"}), "custom")
        overrides = {"TUNE": "speed", "SESS": "8", "CAP": "1", "MAX_MODEL_LEN": "9", "VARIANT": "home"}
        self.assertEqual(presets.classify("ds41", overrides), "custom")
        self.assertEqual(presets.classify("glm53f", overrides), "custom")
        self.assertIn("SESS=8", presets.describe("ds41", overrides, False, "custom"))
        self.assertEqual(presets.classify("glm53", {"KV_FORMAT": "int4", "CAP": "100"}), "custom")
        self.assertEqual(presets.classify("glm53", {"VARIANT": "bit325", "KV_FORMAT": "int4", "CAP": "999", "TUNE": "speed"}), "custom")
        blank = {"TUNE": None, "VARIANT": None, "KV_FORMAT": None, "SESS": None, "CAP": None, "MAX_MODEL_LEN": None}
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
        self.assertEqual(context["caption"], "8 slots · pool 513,536 (int4 not re-measured)")
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
                for key in ("SESS", "CAP", "MAX_MODEL_LEN"):
                    self.assertIsNone(assign[key])
        cleared = ("SESS", "CAP", "MAX_MODEL_LEN", "VARIANT", "KV_FORMAT")
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
            config_bytes.return_value = b"changed config"
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                adapter.activate("glm53", lambda _: None)

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
        extra = "TUNE=speed\nSESS=8\nCAP=100\nMAX_MODEL_LEN=9\nVARIANT=home\n"
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
        self.console.submit(dict(action="restart", model="ds41", confirm=True, expect_active="glm53", preset="quality", revision=row["revision"]))
        self.assertEqual(self.done()["status"], "succeeded")
        for text in self.fake.texts["ds41"].values():
            self.assertIn("TUNE=quality", text)
            self.assertIn("KEEP_ME=1", text)
            for key in ("SESS", "CAP", "MAX_MODEL_LEN", "VARIANT"):
                self.assertNotIn(key + "=", text)
        self.assertEqual(self.console.read_presets("ds41")["current"], "quality")
        flash = self.console.read_presets("glm53f")
        self.assertEqual(flash["current"], "custom")
        self.console.submit(dict(action="restart", model="glm53f", confirm=True, expect_active="ds41", preset="speed", revision=flash["revision"]))
        self.assertEqual(self.done()["status"], "succeeded")
        for text in self.fake.texts["glm53f"].values():
            self.assertIn("TUNE=speed", text)
            self.assertIn("KEEP_ME=1", text)
            for key in ("SESS", "CAP", "MAX_MODEL_LEN", "VARIANT"):
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

    def test_glm_presets_clear_the_other_format_and_cap(self):
        for box in self.fake.texts["glm53"]:
            self.fake.texts["glm53"][box] += "TUNE=speed\nVARIANT=bit325\nKV_FORMAT=int4\nCAP=999\nMAX_MODEL_LEN=777\n"
        row = self.console.read_presets("glm53")
        self.assertEqual(row["current"], "custom")
        self.assertIn("CAP=999", row["current_setting"])
        for preset, present, absent in (
            ("speed", (), ("TUNE=", "VARIANT=", "KV_FORMAT=", "CAP=", "SESS=", "MAX_MODEL_LEN=")),
            ("quality", ("VARIANT=bit325",), ("KV_FORMAT=", "CAP=", "TUNE=", "SESS=", "MAX_MODEL_LEN=")),
            ("context", ("KV_FORMAT=int4",), ("VARIANT=", "CAP=", "TUNE=", "SESS=", "MAX_MODEL_LEN=")),
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


if __name__ == "__main__":
    unittest.main()
