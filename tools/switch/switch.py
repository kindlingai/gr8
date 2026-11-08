#!/usr/bin/env python3
"""Switch preconfigured Compose clusters over SSH; no remote configuration changes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from http.client import HTTPException
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

MODELS = ("glm53", "glm53f", "ds41")


@dataclass(frozen=True)
class Model:
    boxes: tuple
    directory: str
    health_url: str


@dataclass(frozen=True)
class Config:
    boxes: tuple
    models: dict
    timeout: float = 900
    ssh_timeout: float = 300


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("timeouts must be positive finite seconds")
    return number


def box_list(value):
    boxes = tuple(part.strip() for part in value.split(","))
    if not boxes or len(set(boxes)) != len(boxes) or any(
        not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:@\[\]-]*", box) for box in boxes
    ):
        raise ValueError("boxes must be unique comma-separated SSH targets")
    return boxes


def read_config(path):
    """Literal KEY=value, optional shell-style quotes/comments; never source a file."""
    allowed = {"BOXES", "TIMEOUT", "SSH_TIMEOUT"} | {
        f"{name.upper()}_{field}" for name in MODELS for field in ("BOXES", "DIR", "HEALTH_URL")
    }
    values = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in allowed or key in values:
            raise ValueError(f"{path}:{line_number}: unknown, duplicate or malformed key {key!r}")
        words = shlex.split(value, comments=True)
        if len(words) != 1:
            raise ValueError(f"{path}:{line_number}: expected one value (quote spaces)")
        values[key] = words[0]
    required = allowed - {"TIMEOUT", "SSH_TIMEOUT"}
    missing = required - values.keys()
    if missing:
        raise ValueError("missing config keys: " + ", ".join(sorted(missing)))
    boxes = box_list(values["BOXES"])
    models = {}
    for name in MODELS:
        prefix = name.upper()
        members = box_list(values[prefix + "_BOXES"])
        if not set(members) <= set(boxes):
            raise ValueError(f"{prefix}_BOXES must be a subset of BOXES")
        directory = values[prefix + "_DIR"]
        if not directory.startswith("/") or any(c in directory for c in "\n\r\x00"):
            raise ValueError(f"{prefix}_DIR must be an absolute remote directory")
        health_url = values[prefix + "_HEALTH_URL"]
        url = urlsplit(health_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.path != "/health" or url.fragment:
            raise ValueError(f"{prefix}_HEALTH_URL must be the head's http(s) /health URL")
        models[name] = Model(members, directory, health_url)
    return Config(boxes, models, positive(values.get("TIMEOUT", "900")),
                  positive(values.get("SSH_TIMEOUT", "300")))


def plan(config, target):
    """Down all non-target deployments, including boxes unused by the target."""
    downs = {box: [(name, "down") for name in MODELS
                   if name != target and box in config.models[name].boxes] for box in config.boxes}
    return {box: actions for box, actions in downs.items() if actions}, {
        box: [(target, "up")] for box in config.models[target].boxes
    }


def ssh(box, command, timeout):
    result = subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", box, command],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"ssh/compose exit {result.returncode}: {result.stdout.strip()}")


def phase(config, actions, runner=ssh, emit=print, clock=time.monotonic):
    # Serialize Compose operations on a box; separate boxes proceed concurrently.
    def on_box(box, steps):
        for model, action in steps:
            started = clock()
            command = f"cd {shlex.quote(config.models[model].directory)} && docker compose {action}"
            if action == "up":
                command += " -d"
            try:
                runner(box, command, config.ssh_timeout)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                raise RuntimeError(f"{box}: {model} {action} failed: {exc}") from exc
            emit(f"{box}: {model} {action} {clock() - started:.3f}s")

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, len(actions))) as pool:
        futures = [pool.submit(on_box, box, steps) for box, steps in actions.items()]
        for future in as_completed(futures):
            try:
                future.result()
            except RuntimeError as exc:
                failures.append(str(exc))
    if failures:
        raise RuntimeError("\n".join(failures))


def read_health(url, timeout):
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read(65536))
        return isinstance(payload, dict) and payload.get("ready") is True


def probe(url, timeout):
    # A socket timeout is per operation: headers or a body can trickle forever.
    # Bound the whole fetch (including DNS and TLS) in a disposable child. Unlike
    # a timed-out thread, the child is killed and reaped before another poll.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--health-probe", url, str(timeout)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=True,
    )
    payload = json.loads(result.stdout)
    if "error" in payload:
        raise HTTPException(payload["error"])
    return payload["ready"]


def health_worker(url, timeout):
    try:
        payload = {"ready": read_health(url, timeout)}
    except (OSError, URLError, HTTPException, ValueError) as exc:
        payload = {"error": str(exc)}
    print(json.dumps(payload))


def wait_ready(url, timeout, check=probe, clock=time.monotonic, pause=time.sleep):
    deadline = clock() + timeout
    last = "not ready"
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        try:
            ready = check(url, min(2.0, remaining))
            if clock() >= deadline:
                break
            if ready:
                return
            last = "ready is not true"
        except (OSError, URLError, HTTPException, ValueError, subprocess.TimeoutExpired) as exc:
            last = str(exc)
        remaining = deadline - clock()
        if remaining > 0:
            pause(min(0.1, remaining))
    raise RuntimeError(f"health timeout after {timeout:g}s: {url} ({last})")


def switch(config, target, runner=ssh, check=probe, emit=print, clock=time.monotonic):
    started = clock()
    try:
        downs, ups = plan(config, target)
        for name, actions in (("down", downs), ("up", ups)):
            step = clock()
            emit(f"{name}: starting")
            phase(config, actions, runner, emit, clock)
            emit(f"{name}: {clock() - step:.3f}s")
        step = clock()
        model = config.models[target]
        emit(f"health: waiting for {model.boxes[0]} at {model.health_url}")
        wait_ready(model.health_url, config.timeout, check, clock)
        emit(f"health: ready in {clock() - step:.3f}s")
    finally:
        emit(f"total: {clock() - started:.3f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Copy config/switch.env.example to switch.env or pass @FILE. Provision each model's "
        "Compose .env on its boxes first; run only one switch at a time. Failures do not roll back."))
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("config", nargs="?", default="@switch.env", metavar="@FILE")
    parser.add_argument("--timeout", type=positive, help="health wait seconds (overrides TIMEOUT)")
    args = parser.parse_args(argv)
    if not args.config.startswith("@") or len(args.config) < 2:
        parser.error("config must be @FILE")
    try:
        config = read_config(args.config[1:])
        if args.timeout is not None:
            config = Config(config.boxes, config.models, args.timeout, config.ssh_timeout)
        switch(config, args.model)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"gr8 switch: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--health-probe":
        health_worker(sys.argv[2], positive(sys.argv[3]))
    else:
        sys.exit(main())
