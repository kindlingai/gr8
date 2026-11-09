"""Adapters for the existing switch CLI and configured Compose dotenv files."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path
import re
import shlex
import subprocess
import sys

from . import envfile, weights

REPO = Path(__file__).resolve().parents[3]


def load_switch(path=REPO / "tools/switch/switch.py"):
    if not path.is_file():
        raise ValueError("gr8 switch is not installed in this checkout; merge the switch command first")
    spec = importlib.util.spec_from_file_location("gr8_console_switch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ComposeControl:
    def __init__(self, switch_file, runner=subprocess.run):
        self.module = load_switch()
        self.file = Path(switch_file).resolve()
        self.config_bytes = self.file.read_bytes()
        self.config = self.module.read_config(self.file)
        self.ensure_config()
        self.runner = runner
        self.remote_source = Path(envfile.__file__).read_text()
        self.weight_source = Path(weights.__file__).read_text()
        self._variant_cache = {}

    def ensure_config(self):
        if self.file.read_bytes() != self.config_bytes:
            raise ValueError("switch configuration changed; restart the console and review its targets")

    def _run(self, args, timeout, input=None):
        result = self.runner(args, input=input, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"command exited {result.returncode}: {result.stdout[-8192:]}")
        return result.stdout

    def remote(self, box, command, data=None):
        return self._run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", box, command],
                         self.config.ssh_timeout, data)

    def environment(self, model, box, action="read", **kwargs):
        self.ensure_config()
        path = self.config.models[model].directory + "/.env"
        result = self.remote(box, "python3 -c " + shlex.quote(self.remote_source),
                             json.dumps(dict(path=path, action=action, **kwargs)))
        return json.loads(result)

    def read_all(self, model):
        boxes = self.config.models[model].boxes
        with ThreadPoolExecutor(max_workers=min(4, len(boxes))) as pool:
            return dict(zip(boxes, pool.map(lambda box: self.environment(model, box), boxes)))

    def variant_cache(self, model, variant, records, refresh=False):
        """On-disk check for one variant. present is true only when every box has it."""
        if not re.fullmatch(r"[A-Za-z0-9_]+", variant or ""):
            raise ValueError("unsupported variant")
        boxes = list(records)
        key = (model, variant, tuple((box, records[box].get("revision")) for box in boxes))
        now = time.monotonic()
        cached = self._variant_cache.get(key)
        if cached and not refresh and now - cached[0] < 5:
            return cached[1]
        width = len(boxes)

        def one(pair):
            index, box = pair
            values = envfile.parse(records[box]["text"])
            peers = [part.strip() for part in (values.get("PEERS") or "").split(",") if part.strip()]
            try:
                tp = int(values["TP"]) if values.get("TP") else (len(peers) or width)
            except ValueError:
                tp = len(peers) or width
            payload = {
                "cache": values.get("GR8_CACHE") or "",
                "model": values.get("MODEL") or "",
                "tp": tp,
                "rank": index,
                "variant": variant,
                "home": values.get("GR8_HOME") or "",
                "peers": peers,
                "node_ip": values.get("NODE_IP") or "",
            }
            try:
                raw = self.remote(box, "python3 -c " + shlex.quote(self.weight_source), json.dumps(payload))
                row = json.loads(raw)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                row = None
            # A failed probe is unknown. It is not evidence that the weights are missing.
            if not isinstance(row, dict):
                row = {"state": "unknown", "present": False, "size_gb": None, "rate_mb_s": None}
            elif row.get("state") not in ("present", "missing", "unknown"):
                row["state"] = "present" if row.get("present") is True else "missing"
            row["box"] = index + 1
            return row

        with ThreadPoolExecutor(max_workers=min(4, max(1, width))) as pool:
            status = weights.aggregate(list(pool.map(one, enumerate(boxes))))
        self._variant_cache[key] = (now, status)
        return status

    def stop(self, model, emit):
        self.ensure_config()
        # Use the existing switch command's phase runner, not a second switch.
        actions = {box: [(model, "down")] for box in self.config.models[model].boxes}
        self.module.phase(self.config, actions, emit=emit)

    @staticmethod
    def _end(process, reader, gently):
        """Stop the switch command and its children. gently: SIGTERM first, SIGKILL after 10 s."""
        for sig, wait in (((signal.SIGTERM, 10),) if gently else ()) + ((signal.SIGKILL, None),):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                pass
        reader.join(5)

    def activate(self, model, emit, health_timeout=None, abort=None):
        """Run gr8 switch. abort, a threading.Event, ends it early when set (the caller saw the engine fail)."""
        self.ensure_config()
        # Its exit status includes all-box start AND ready:true. Never claim ready from `up -d` alone.
        emit("gr8 switch: stop other deployments, start target, wait for health")
        args = [str(REPO / "gr8"), "switch", model, "@" + str(self.file)]
        health = float(self.config.timeout)
        if health_timeout is not None:
            # A confirmed variant download must outlive the stock health timeout.
            health = max(health, float(health_timeout))
            args = [str(REPO / "gr8"), "switch", "--timeout", str(int(math.ceil(health))), model, "@" + str(self.file)]
        timeout = health + self.config.ssh_timeout * 4 + 30
        if self.runner is not subprocess.run:  # injectable offline adapter tests
            emit(self._run(args, timeout)[-16384:])
            return
        tail = deque(maxlen=12)
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"}) as process:
            def read_lines():
                for line in iter(lambda: process.stdout.readline(16384), ""):
                    tail.append(line)
                    emit(line.rstrip())
            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            end = time.monotonic() + timeout
            while True:
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if abort is not None and abort.is_set():
                    self._end(process, reader, True)
                    raise RuntimeError("switch command stopped: the engine failed to boot")
                if time.monotonic() >= end:
                    self._end(process, reader, False)
                    raise RuntimeError("switch command timed out; inspect remote deployment state")
            reader.join(5)
            if code:
                raise RuntimeError(f"switch exited {code}: {''.join(tail)[-8192:]}")
