import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("cluster_switch", ROOT / "tools/switch/switch.py")
sw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sw
SPEC.loader.exec_module(sw)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.config = sw.read_config(ROOT / "config/switch.env.example")

    def test_plan_all_targets(self):
        for target in sw.MODELS:
            with self.subTest(target=target):
                downs, ups = sw.plan(self.config, target)
                actual = {(box, model, action) for box, steps in downs.items() for model, action in steps}
                expected = {(box, model, "down") for model in sw.MODELS if model != target
                            for box in self.config.models[model].boxes}
                self.assertEqual(actual, expected)
                self.assertEqual(ups, {box: [(target, "up")] for box in self.config.models[target].boxes})
        downs, _ = sw.plan(self.config, "glm53f")
        self.assertEqual(set(downs), set(self.config.boxes))  # stop TP4 even outside pair

    def test_parallel_boxes_and_down_barrier(self):
        # All four workers must arrive; serial execution fails the barrier.
        barrier = threading.Barrier(4, timeout=2)
        calls = []
        lock = threading.Lock()
        def runner(box, command, timeout):
            if command.endswith("down") and "deepseek" not in command:
                barrier.wait()
            with lock:
                if command.endswith("up -d"):
                    self.assertEqual(sum(c.endswith("down") for _, c in calls), 8)
                calls.append((box, command))
        output = []
        sw.switch(self.config, "glm53f", runner, lambda *_: True, output.append)
        self.assertEqual(len(calls), 10)
        self.assertTrue(all("docker compose" in command for _, command in calls))
        self.assertTrue(any(line.startswith("total:") for line in output))
        self.assertTrue(any(line.startswith("health: ready") for line in output))

    def test_failed_box_prevents_up(self):
        box = self.config.boxes[-1]
        calls = []
        def runner(host, command, timeout):
            calls.append(command)
            if host == box:
                raise RuntimeError("exit 255: unreachable")
        with self.assertRaisesRegex(RuntimeError, box + ".*down failed"):
            sw.switch(self.config, "glm53f", runner, emit=lambda _: None)
        self.assertFalse(any("up -d" in command for command in calls))

    def test_up_failure_skips_health(self):
        checks = []
        def runner(host, command, timeout):
            if command.endswith("up -d"):
                raise subprocess.TimeoutExpired("ssh", timeout)
        with self.assertRaisesRegex(RuntimeError, "up failed"):
            sw.switch(self.config, "ds41", runner, lambda *_: checks.append(1), lambda _: None)
        self.assertEqual(checks, [])

    def test_health_retry_then_success(self):
        clock = Clock()
        calls = []
        def check(url, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise URLError("starting")
            return len(calls) == 3
        sw.wait_ready("http://192.0.2.1/health", 1, check, clock, clock.sleep)
        self.assertAlmostEqual(clock.now, 0.2)
        self.assertEqual(len(calls), 3)

    def test_health_timeout(self):
        clock = Clock()
        with self.assertRaisesRegex(RuntimeError, "health timeout after 0.25s"):
            sw.wait_ready("http://192.0.2.1/health", 0.25, lambda *_: False, clock, clock.sleep)
        self.assertAlmostEqual(clock.now, 0.25)

    def test_health_requires_json_boolean(self):
        for payload, expected in (({"ready": True}, True), ({"ready": False}, False),
                                  ({"ready": "true"}, False), ({"ready": 1}, False), ([], False)):
            with patch.object(sw, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                self.assertEqual(sw.probe("unused", 1), expected)
        with patch.object(sw, "urlopen", return_value=io.BytesIO(b"invalid")):
            with self.assertRaises(ValueError):
                sw.probe("unused", 1)

    def test_ssh_is_batch_and_timeout_bounded(self):
        with patch.object(sw.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            sw.ssh("operator@192.0.2.1", "command", 42)
            args, kwargs = run.call_args
            self.assertIn("BatchMode=yes", args[0])
            self.assertEqual(args[0][-2:], ["operator@192.0.2.1", "command"])
            self.assertEqual(kwargs["timeout"], 42)
        with patch.object(sw.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, "denied")):
            with self.assertRaisesRegex(RuntimeError, "255: denied"):
                sw.ssh("operator@192.0.2.1", "command", 42)

    def test_paths_are_quoted(self):
        model = sw.Model((self.config.boxes[0],), "/opt/a b;'quoted'", "http://192.0.2.1/health")
        config = sw.Config(self.config.boxes, {**self.config.models, "glm53": model})
        calls = []
        sw.phase(config, {model.boxes[0]: [("glm53", "down")]},
                 lambda box, command, timeout: calls.append(command), lambda _: None)
        self.assertEqual(calls, ["cd '/opt/a b;'\"'\"'quoted'\"'\"'' && docker compose down"])

    def test_bad_config_fails_before_ssh(self):
        original = (ROOT / "config/switch.env.example").read_text()
        for text in (original + "\nUNKNOWN=1", original + "\nTIMEOUT=2",
                     original.replace("TIMEOUT=900", "TIMEOUT=nan"),
                     original.replace("TIMEOUT=900", "TIMEOUT=0"),
                     original.replace("GLM53_DIR=/opt/gr8/docker", "GLM53_DIR=relative"),
                     original.replace("GLM53F_BOXES=operator@192.0.2.1,operator@192.0.2.2", "GLM53F_BOXES=-bad"),
                     original.replace("GLM53F_BOXES=operator@192.0.2.1,operator@192.0.2.2", "GLM53F_BOXES=unknown"),
                     original.replace("GLM53_DIR=/opt/gr8/docker\n", "")):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "switch.env"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    sw.read_config(path)

    def test_main_failure_nonzero(self):
        with patch.object(sw, "switch", side_effect=RuntimeError("box failed")), patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(sw.main(["ds41", "@" + str(ROOT / "config/switch.env.example")]), 1)
            self.assertIn("box failed", err.getvalue())

    def test_entrypoint(self):
        result = subprocess.run([str(ROOT / "gr8"), "switch", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("glm53f", result.stdout)
        result = subprocess.run([str(ROOT / "gr8"), "switch", "unknown"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
