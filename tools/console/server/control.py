"""Adapters for the existing switch CLI and configured Compose dotenv files."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import signal
import threading
from collections import deque
from pathlib import Path
import shlex
import subprocess
import sys

from . import envfile

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

    def stop(self, model, emit):
        self.ensure_config()
        # Use the existing switch command's phase runner, not a second switch.
        actions = {box: [(model, "down")] for box in self.config.models[model].boxes}
        self.module.phase(self.config, actions, emit=emit)

    def activate(self, model, emit):
        self.ensure_config()
        # Its exit status includes all-box start AND ready:true. Never claim ready from `up -d` alone.
        emit("gr8 switch: stop other deployments, start target, wait for health")
        args = [str(REPO / "gr8"), "switch", model, "@" + str(self.file)]
        timeout = self.config.timeout + self.config.ssh_timeout * 4 + 30
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
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                reader.join(5)
                raise RuntimeError("switch command timed out; inspect remote deployment state")
            reader.join(5)
            if code:
                raise RuntimeError(f"switch exited {code}: {''.join(tail)[-8192:]}")
