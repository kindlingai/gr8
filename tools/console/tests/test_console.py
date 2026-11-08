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
        self.assertEqual(envfile.read_values(changed, {"GR8_QUIT_FAST", "REUSE", "SESS"}, engine)["GR8_QUIT_FAST"], "1")

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
        self.assertEqual(presets.classify("ds41", {}), "custom")
        self.assertEqual(presets.classify("ds41", {"TUNE": "context"}), "context")
        self.assertEqual(presets.classify("ds41", {"TUNE": "speed", "VARIANT": "home"}), "custom")
        self.assertEqual(presets.classify("glm53f", {"TUNE": "quality"}), "quality")
        self.assertEqual(presets.changes_for("glm53", "quality"), {"TUNE": None, "VARIANT": "bit325", "KV_FORMAT": None})
        quality = next(item for item in presets.CATALOG["glm53"] if item["id"] == "quality")
        self.assertIn("same 3.25 bpw size", quality["summary"])
        self.assertIn("bits placed differently", quality["summary"])
        self.assertIsNone(presets.changes_for("glm53", "speed")["VARIANT"])
        self.assertEqual(presets.changes_for("ds41", "speed")["TUNE"], "speed")
        with self.assertRaises(ValueError):
            presets.changes_for("glm53", "fast")
        for model in presets.CATALOG:
            for item in presets.CATALOG[model]:
                self.assertEqual(set(item["assign"]), set(presets.KEYS))


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

    def test_confirmed_preset_writes_all_boxes_then_restart(self):
        row = self.console.read_presets("glm53")
        self.assertEqual(row["current"], "speed")
        self.console.submit(self.request(preset="quality", revision=row["revision"]))
        job = self.done()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual([c[0] for c in self.fake.calls], ["write"] * 4 + ["stop", "switch"])
        self.assertTrue(all("VARIANT=bit325" in t and "KEEP_ME=1" in t and "SESS=4" in t for t in self.fake.texts["glm53"].values()))
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
