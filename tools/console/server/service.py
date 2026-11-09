"""Read-only polling and one serialized, explicitly confirmed control operation."""
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from urllib.request import urlopen
from urllib.parse import urlsplit

from . import envfile, metrics, presets, weights
from .control import ComposeControl

DEFAULT_IDS = {"ds41": "deepseek-v4.1-flash", "glm53": "glm-5.3", "glm53f": "glm-5.3-flash"}


def fetch(url):
    with urlopen(url, timeout=2) as response:
        raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError("head response too large")
        return raw.decode()


def rpc(host, port, message):
    with socket.create_connection((host, port), timeout=2) as conn:
        conn.sendall(json.dumps(message).encode() + b"\n")
        with conn.makefile("rb") as stream:
            raw = stream.readline(1048577)
        if len(raw) > 1048576:
            raise ValueError("supervisor response too large")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("t") == "error":
            raise ValueError("supervisor refused: " + str(value))
        return value


STARTED = re.compile(r"engine started \(epoch (\d+)\)")
# Supervisor log lines are "HH:MM:SS text". The head logs these when an engine, its front end, or its
# shim exits; a worker rank's exit reaches the head as "rank N (peer) failed: ...".
FAILED = re.compile(r"^\d\d:\d\d:\d\d (rank \d+ \([^)]*\) failed: .+|(?:engine|front end|shim) exited \(.+)$")
# supervisor.py logs this line when it resolves the model of a boot; status model_id starts with the same 16 characters.
MODEL_SET = re.compile(r"^\d\d:\d\d:\d\d model set \S+ from .*, variant .*, id (\S+)\s*$")
WATCH_WINDOW = 262144  # bytes read from the end of the head supervisor log
CAP_KEYS = ("KV_FORMAT", "CAP", "TP")  # read for the GLM-5.3 CAP rescale; CAP is not a preset key


def latest_epoch(text):
    """The epoch of the last "engine started" line in this log text, or None."""
    found = STARTED.findall(text)
    return int(found[-1]) if found else None


def since_model_set(text, model_id):
    """The log text after the last "model set ... id <model_id[:16]>" line, or None when that line is not in it.

    The head log holds lines of the deployment that was running before this operation. Those come before the
    line the target's supervisor writes when it resolves its model, so only the text after it is the target's."""
    lines = text.splitlines(keepends=True)
    for index in range(len(lines) - 1, -1, -1):
        found = MODEL_SET.match(lines[index])
        if found and found.group(1) == model_id[:16]:
            return "".join(lines[index + 1:])
    return None


def boot_failure(text, after):
    """The supervisor's first failure line of the latest boot in this log text, or None.

    A boot counts only when its "engine started (epoch E)" line has E > after, so a failure line left
    by a boot that began before this operation (or an old deployment that still answers) is not mistaken
    for this one."""
    epoch, reason = None, None
    for line in text.splitlines():
        started = STARTED.search(line)
        if started:
            epoch, reason = int(started.group(1)), None
            continue
        found = FAILED.match(line)
        if found and reason is None and epoch is not None and epoch > after:
            reason = found.group(1)[:500]
    return reason


def revision(records):
    return hashlib.sha256(json.dumps({b: r["revision"] for b, r in records.items()}, sort_keys=True).encode()).hexdigest()


class Console:
    def __init__(self, config, control=None, get=fetch, call=rpc):
        self.config, self.get, self.call = config, get, call
        self.state = Path(config.get("state_dir", Path(__file__).resolve().parents[1] / ".state"))
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state, 0o700)
        # A single worker owns polling and control. Fail clearly instead of split process state.
        self.instance = open(self.state / "console.lock", "a")
        try:
            fcntl.flock(self.instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance.close()
            raise ValueError("one console process per state directory; use one server worker")
        self.control, self.control_error = control, None
        if self.control is None and config.get("switch_file"):
            try:
                self.control = ComposeControl(config["switch_file"])
            except (ValueError, OSError) as exc:
                self.control_error = str(exc)
        self.models = copy.deepcopy(config.get("models", {}))
        if self.control:
            for name, model in self.control.config.models.items():
                row = self.models.setdefault(name, {})
                row.setdefault("head_url", model.health_url.removesuffix("/health"))
                row.setdefault("controls", [{"host": b.rsplit("@", 1)[-1].strip("[]"), "port": row.get("control_port", 7340)}
                                            for b in model.boxes])
                row["width"] = len(model.boxes)
        for name, row in self.models.items():
            if name not in presets.MODELS:
                raise ValueError("unknown configured model")
            url = urlsplit(row.get("head_url", ""))
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("head_url must be an HTTP base URL without credentials/query")
            row["head_url"] = row["head_url"].rstrip("/")
            row.setdefault("width", len(row.get("controls", [])))
            if not isinstance(row["width"], int) or not 0 <= row["width"] <= 8 or len(row.get("controls", [])) > 8:
                raise ValueError("invalid configured width or control count")
        self.csrf = secrets.token_urlsafe(32)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock, self.poll_lock, self.submit_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.job = None
        saved = self.state / "operation.json"
        if saved.exists():
            self.job = json.loads(saved.read_text())
            if self.job.get("status") == "running":
                self.job.update(status="interrupted", error="Console stopped mid-operation. Inspect every box and saved dotenv backups before retrying.")
                self._save_job()
        history = self.state / "switch-times.json"
        self.switch_times = json.loads(history.read_text()) if history.exists() else {}
        self.samples, self.cache, self.cache_at = {}, None, 0
        self.last_boot = None
        self.watch_interval = 5  # seconds between supervisor log reads while a switch waits for health

    def close(self):
        self.pool.shutdown(wait=True)
        self.instance.close()

    def can_write(self):
        return bool(self.config.get("allow_control", False) and self.control)

    def _save_job(self):
        path = self.state / "operation.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.job, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    def operation(self):
        with self.lock:
            job = copy.deepcopy(self.job)
        if job:
            job["elapsed"] = (job.get("ended") or time.time()) - job["started"]
        return job

    def snapshot(self, fresh=False):
        with self.poll_lock:
            now = time.monotonic()
            if not fresh and self.cache and now - self.cache_at < 1.5:
                return {**self.cache, "operation": self.operation()}
            models, observations, active = [], {}, []
            # Fetch each distinct head once: models can share a serving port.
            def probe(url):
                result = {}
                for route in ("/health", "/v1/models"):
                    try:
                        result[route] = json.loads(self.get(url + route))
                    except (OSError, ValueError) as exc:
                        result[route] = None
                        result["error"] = str(exc)
                return result
            urls = list(dict.fromkeys(m["head_url"] for m in self.models.values()))
            with ThreadPoolExecutor(max_workers=4) as pool:
                observations = dict(zip(urls, pool.map(probe, urls)))
            for name, title in presets.MODELS.items():
                config = self.models.get(name)
                row = dict(id=name, name=title, state="not configured", width=None, last_switch=self.switch_times.get(name))
                if config:
                    obs = observations[config["head_url"]]
                    health, model_list = obs.get("/health"), obs.get("/v1/models")
                    data = model_list.get("data", []) if isinstance(model_list, dict) else []
                    if not isinstance(data, list):
                        data = []
                    match = next((m for m in data if isinstance(m, dict) and
                                  str(m.get("id", "")).lower() == config.get("served_id", DEFAULT_IDS[name]).lower()), None)
                    ready = isinstance(health, dict) and health.get("ready") is True
                    row.update(state="ready" if ready and match else "other model" if ready and data else "unavailable",
                               width=config["width"], error=obs.get("error"))
                    if ready and match:
                        active.append(name)
                        row["context"] = match.get("max_model_len")
                        row["started"] = match.get("created")
                models.append(row)
            current = active[0] if len(active) == 1 else None
            summary, boxes, logs, error, info = {}, [], "", None, {}
            if current:
                config = self.models[current]
                try:
                    raw = self.get(config["head_url"] + "/metrics")
                    values, info = metrics.parse(raw)
                    prior, when = self.samples.get(current, ({}, now))
                    summary = metrics.summarize(values, prior, now - when)
                    self.samples = {current: (values, now)}
                except (OSError, ValueError) as exc:
                    self.samples.clear()
                    error = "Metrics unavailable: " + str(exc)
                controls = config.get("controls", [])
                def status(pair):
                    rank, target = pair
                    row = {"rank": rank, "state": "unavailable", "mem_gib": None, "last_line": ""}
                    try:
                        value = self.call(target["host"], int(target.get("port", 7340)), {"t": "status"})
                        row.update({k: value.get(k) for k in ("rank", "state", "mem_gib", "last_line", "engine_pid")})
                        mem = row["mem_gib"]
                        if not isinstance(mem, (float, int)) or not math.isfinite(mem):
                            row["mem_gib"] = None
                    except (OSError, ValueError) as exc:
                        row["last_line"] = str(exc)
                    return row
                with ThreadPoolExecutor(max_workers=4) as pool:
                    boxes = list(pool.map(status, enumerate(controls)))
                boot = (current, next(r.get("started") for r in models if r["id"] == current), tuple(b.get("engine_pid") for b in boxes))
                if self.last_boot is not None and boot != self.last_boot:
                    summary.update(decode=None, prefill=None)
                self.last_boot = boot
                if controls:
                    try:
                        target = controls[0]
                        logs = self.call(target["host"], int(target.get("port", 7340)),
                                         {"t": "logs", "which": "supervisor", "offset": -8192}).get("data", "")
                    except (OSError, ValueError) as exc:
                        logs = "Log unavailable: " + str(exc)
            else:
                self.samples.clear()
                if len(active) > 1:
                    error = "Multiple models report ready; inspect deployment configuration before control."
            self.cache = dict(models=models, active=current, stats={**summary, **info}, boxes=boxes,
                              logs=logs, error=error, observed_at=time.time(), can_control=self.can_write(),
                              control_error=self.control_error, csrf=self.csrf,
                              fixture=bool(self.config.get("fixture", False)))
            self.cache_at = now
            return {**self.cache, "operation": self.operation()}

    def read_presets(self, model):
        if model not in self.models:
            raise ValueError("model is not configured")
        records = self.control.read_all(model) if self.control else {}
        watched = presets.keys_for(model)
        rows = [envfile.read_values(r["text"], watched, presets.engine_keys(model)) for r in records.values()]
        values, mixed_keys = {}, []
        for key in watched:
            variants = {row[key] for row in rows}
            values[key] = next(iter(variants)) if len(variants) == 1 else None
            if len(variants) > 1:
                mixed_keys.append(key)
        current = presets.classify(model, values) if rows and not mixed_keys else ("custom" if rows else None)
        caps = {}
        if records:
            cap_rows = [envfile.read_values(r["text"], CAP_KEYS, frozenset()) for r in records.values()]
            for item in presets.CATALOG[model]:
                caps[item["id"]] = presets.cap_diff(model, item["id"], cap_rows)
        shown = presets.public(model, values, mixed_keys, caps)
        if records and self.control:
            seen = {}
            for item in shown:
                variant = weights.changed_variant(values, presets.changes_for(model, item["id"]), mixed_keys)
                item["download"] = None
                if not variant:
                    continue
                if variant not in seen:
                    seen[variant] = self.control.variant_cache(model, variant, records)
                item["download"] = weights.preset_notice(seen[variant], self._health_budget())[0]
        else:
            for item in shown:
                item["download"] = None
        return dict(source=presets.SOURCE[model], presets=shown, current=current,
                    current_setting=presets.describe(model, values, bool(mixed_keys), current),
                    values=values, mixed=mixed_keys, revision=revision(records) if records else None,
                    observed=bool(records))

    def submit(self, request):
        with self.submit_lock:
            if not self.can_write():
                raise PermissionError(self.control_error or "Control disabled; configure switch_file and allow_control")
            action, model = request.get("action"), request.get("model")
            if request.get("confirm") is not True:
                raise ValueError("explicit confirmation required")
            if action not in ("switch", "restart", "stop", "restore") or model not in self.models:
                raise ValueError("invalid action or model")
            if request.get("changes"):
                raise ValueError("raw setting maps are not accepted; choose a preset")
            preset = request.get("preset")
            changes = {}
            if preset is not None:
                if action != "restart" or not request.get("revision"):
                    raise ValueError("a preset requires a reviewed revision and restart")
                changes = presets.changes_for(model, preset)
            with self.lock:
                if self.job and self.job["status"] == "running":
                    raise RuntimeError("another console operation is running")
                prior = copy.deepcopy(self.job)
            if "expect_active" not in request:
                raise ValueError("reviewed active model is required")
            observed = self.snapshot(fresh=True)
            if observed["active"] != request["expect_active"] or observed.get("error", "") and "Multiple models" in observed["error"]:
                raise RuntimeError("active model changed since review; refresh and review again")
            backup_id = None
            if action == "restore":
                if not prior or not prior.get("can_restore") or prior["model"] != model or request.get("backup_id") != prior.get("backup_id"):
                    raise ValueError("no matching previous-settings backup for the last operation")
                backup_id = prior["backup_id"]
            with self.lock:
                self.job = dict(id=secrets.token_hex(8), action=action, model=model, status="running",
                                started=time.time(), steps=[], error=None, can_restore=bool(backup_id), backup_id=backup_id)
                self._save_job()
                self.pool.submit(self._operate, action, model, changes, request.get("revision"), backup_id,
                                 request.get("accept_download") is True, preset)
                return copy.deepcopy(self.job)

    def _restore(self, model, backup_id):
        saved = json.loads((self.state / (backup_id + ".backup.json")).read_text())
        if saved["model"] != model:
            raise ValueError("backup model differs")
        records = saved["records"]
        current = self.control.read_all(model)
        if current.keys() != records.keys():
            raise ValueError("backup membership differs; inspect configuration")
        for box, record in current.items():
            if record["revision"] not in (records[box]["revision"], saved["written"][box]):
                raise ValueError("dotenv edited outside this operation; refusing to overwrite it")
        self._emit("Restoring previous settings on every target box; no restart")
        for box, record in current.items():
            if record["revision"] != records[box]["revision"]:
                self.control.environment(model, box, "write", expected=record["revision"], text=records[box]["text"])
        self._emit("Previous settings restored. Restart or switch explicitly to apply them.")
        with self.lock:
            self.job["can_restore"] = False

    def _emit(self, message):
        with self.lock:
            self.job["steps"].append(str(message)[-16384:])
            self.job["steps"] = self.job["steps"][-80:]
            self._save_job()

    def _health_budget(self):
        config = getattr(self.control, "config", None)
        timeout = getattr(config, "timeout", None) if config is not None else None
        return float(timeout) if timeout else 900.0

    def _head_log(self, model):
        """The tail of the head supervisor log as text, or None when no supervisor answers."""
        controls = self.models.get(model, {}).get("controls") or []
        if not controls:
            return None
        target = controls[0]
        try:
            data = self.call(target["host"], int(target.get("port", 7340)),
                             {"t": "logs", "which": "supervisor", "offset": -WATCH_WINDOW}).get("data")
        except (OSError, ValueError, TypeError):
            return None
        return data if isinstance(data, str) else None

    def _head_model_id(self, model):
        """The model_id the head supervisor reports in its status, or None when none answers or it has none yet."""
        controls = self.models.get(model, {}).get("controls") or []
        if not controls:
            return None
        target = controls[0]
        try:
            value = self.call(target["host"], int(target.get("port", 7340)), {"t": "status"}).get("model_id")
        except (OSError, ValueError, TypeError):
            return None
        return value if isinstance(value, str) and value else None

    def _boot_baseline(self, model):
        """The epoch of the latest engine start already in the head log, read before this operation stops
        anything. Only a later epoch is a boot of this operation."""
        data = self._head_log(model)
        return latest_epoch(data) if data else None

    def _watch_boot(self, model, stop, abort, failure, after, download, old_id=None):
        """While a switch waits for health: log fetch-rate lines (a download), and stop the wait when the
        head supervisor reports that the engine of this boot exited. The boxes then keep restarting it, and
        the health wait would only run out its full timeout.

        The control port is shared by all models, so until gr8 switch downs the old deployment, the log and
        the status are its. A failure counts only when the status model_id is not the old deployment's (old_id,
        read before the switch) and the failing boot follows the target's own "model set" line in the log."""
        seen = set()
        while not stop.wait(self.watch_interval):
            current = self._head_model_id(model)
            data = self._head_log(model)
            if not data:
                continue
            mine = since_model_set(data, current) if current and current != old_id else None
            reason = boot_failure(mine, after) if mine else None
            if reason:
                failure.append(reason)
                self._emit("Boot failed: " + reason)
                abort.set()
                return
            for line in data.splitlines() if download else ():
                if "MB/s" not in line or line in seen:
                    continue
                seen.add(line)
                self._emit("downloading " + line[-500:])

    def _operate(self, action, model, changes, reviewed, backup_id=None, accept_download=False, preset=None):
        try:
            if action == "restore":
                self._restore(model, backup_id)
            download = None
            notice = None
            if changes:
                if set(changes) - set(presets.keys_for(model)):
                    raise ValueError("unsupported setting")
                records = self.control.read_all(model)
                if revision(records) != reviewed:
                    raise ValueError("dotenv changed since review; refresh presets")
                watched = presets.keys_for(model)
                rows = [envfile.read_values(record["text"], watched, presets.engine_keys(model)) for record in records.values()]
                merged, mixed = {}, []
                for key in watched:
                    variants = {row[key] for row in rows}
                    merged[key] = next(iter(variants)) if len(variants) == 1 else None
                    if len(variants) > 1:
                        mixed.append(key)
                variant = weights.changed_variant(merged, changes, mixed)
                if variant:
                    status = self.control.variant_cache(model, variant, records, refresh=True)
                    text, health = weights.preset_notice(status, self._health_budget())
                    if health is not None:
                        if not accept_download:
                            raise ValueError("variant weights are not on every box; confirm the download")
                        download = {"text": text, "health": health}
                    elif text:
                        notice = text
                # GLM-5.3 only: moving between the fp8 and int4 KV rows rescales an explicit CAP, per box.
                cap_envs = {b: envfile.read_values(r["text"], CAP_KEYS, frozenset()) for b, r in records.items()}
                new_caps = {b: presets.cap_for(model, preset, env, len(records)) for b, env in cap_envs.items()} if preset else {}
                updated = {b: envfile.patch(r["text"], {**changes, "CAP": new_caps[b]} if new_caps.get(b) else changes, presets.engine_keys(model))
                           for b, r in records.items()}
                backup = self.state / (self.job["id"] + ".backup.json")
                backup.write_text(json.dumps({"model": model, "records": records, "written": {b: envfile.digest(t) for b, t in updated.items()}}, indent=2))
                os.chmod(backup, 0o600)
                self._emit("Saved exact dotenv backups; writing reviewed overrides on every target box")
                if any(new_caps.values()):
                    self._emit("CAP: " + "; ".join(f"{cap_envs[b]['CAP']} to {cap}" for b, cap in new_caps.items() if cap)
                               + " tokens, the same KV memory")
                written = []
                try:
                    for box, old in records.items():
                        # Include the in-flight box: timeout may mean the write succeeded remotely.
                        written.append(box)
                        self.control.environment(model, box, "write", expected=old["revision"], text=updated[box])
                except Exception:
                    failures = []
                    for box in reversed(written):
                        try:
                            current = self.control.environment(model, box)
                            if current["revision"] == envfile.digest(updated[box]):
                                self.control.environment(model, box, "write", expected=current["revision"], text=records[box]["text"])
                            elif current["revision"] != records[box]["revision"]:
                                raise ValueError("concurrent external edit; not overwritten")
                        except Exception as exc:
                            failures.append(str(exc))
                    self._emit("Dotenv rollback: " + ("; ".join(failures) if failures else "originals restored; no restart attempted"))
                    raise
                with self.lock:
                    self.job.update(can_restore=True, backup_id=self.job["id"])
                    self._save_job()
            if action in ("stop", "restart"):
                self._emit("Stopping target on every configured box")
                self.control.stop(model, self._emit)
            if action in ("switch", "restart"):
                stop_watch, abort, failure = threading.Event(), threading.Event(), []
                # Read after the stop: an old deployment that crash-loops can start a boot between an earlier
                # read and its stop. The log still holds its boots; only a later epoch is a boot of this operation.
                # The status model_id is the deployment that answers now: a switch trusts the log only once another
                # one answers. Nothing answering (the usual case after a stop) gives no baseline and no old id.
                baseline = self._boot_baseline(model)
                old_id = self._head_model_id(model)
                # No baseline (no supervisor answered): count a boot only if it began in this second or later.
                after = baseline if baseline is not None else int(time.time()) - 1
                if download:
                    self._emit("downloading " + download["text"])
                    self._emit(f"health wait extended to {download['health']:.0f}s for the download")
                elif notice:
                    self._emit(notice)
                watcher = threading.Thread(target=self._watch_boot, args=(model, stop_watch, abort, failure, after, bool(download), old_id), daemon=True)
                watcher.start()
                try:
                    self.control.activate(model, self._emit, health_timeout=download["health"] if download else None, abort=abort)
                except Exception:
                    if failure:
                        raise RuntimeError(f"Boot failed: {failure[0]}. The boxes may keep restarting the engine; "
                                           "inspect them before you try again.") from None
                    raise
                finally:
                    stop_watch.set()
                    watcher.join(5)
                # CLI checks ready; also ensure the endpoint is serving the selected identity.
                row = self.models[model]
                models = json.loads(self.get(row["head_url"] + "/v1/models"))
                expected = row.get("served_id", DEFAULT_IDS[model]).lower()
                if not any(str(m.get("id", "")).lower() == expected for m in models.get("data", [])):
                    raise ValueError("health answered but the served model identity differs from the target")
                self._emit("Health ready; target model identity verified")
            elif action == "stop":
                self._emit("Compose down completed on every target box")
            with self.lock:
                self.job.update(status="succeeded", ended=time.time())
                if action == "switch":
                    self.switch_times[model] = self.job["ended"] - self.job["started"]
                    history = self.state / "switch-times.json"
                    tmp = history.with_suffix(".tmp")
                    tmp.write_text(json.dumps(self.switch_times))
                    tmp.replace(history)
        except Exception as exc:
            with self.lock:
                self.job.update(status="failed", error=str(exc), ended=time.time())
                self.job["steps"].append("Stopped at failure. Inspect boxes and saved backups; no automatic restart/rollback after a boot failure.")
        finally:
            with self.lock:
                self._save_job()
            self.cache_at = 0
            self.samples.clear()
