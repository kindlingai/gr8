"""Synthetic protocol-shaped fixtures, NOT recorded performance measurements.

Shapes follow frontend/g6front-rs/src/{api,metrics}.rs and tools/launch/supervisor.py.
No live head, GPU, SSH, Compose, or inference call is used.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socketserver
import tempfile
import threading
import time
from types import SimpleNamespace

from tools.console.server import envfile
from tools.console.server.service import DEFAULT_IDS


class Fixture:
    def __init__(self):
        self.active = "glm53"
        self.fail_write = None
        self.fail_boot = False
        self.missing_weights = None
        self.last_health_timeout = None
        self.refuse_supervisor = False
        self.not_ready = False
        self.supervisor_log = ""   # the head log while gr8 switch runs (the boot of this operation)
        self.old_log = ""          # the head log before it: the old deployment's boot
        self.switching = False
        self.new_up = True         # while switching: the target's supervisor answers (else the old deployment still does)
        self.old_phase = 0         # seconds the old deployment keeps the control port after gr8 switch starts
        self.race_log = ""         # lines the old deployment logs when gr8 switch starts (a crash-looping boot)
        self.on_stop = None        # called inside stop(): the old deployment logs something before it goes down
        self.live_id = None        # the model_id the head supervisor reports while switching
        self.activate_wait = 0
        self.calls = []
        self.texts = {m: {f"box-{i}": "# fixture dotenv\nSESS=4\nEXTRA_ENV='KEEP_ME=1'\n"
                          for i in range(2 if m == "glm53f" else 4)} for m in DEFAULT_IDS}
        self.config = SimpleNamespace(models={m: SimpleNamespace(boxes=tuple(boxes), directory="/fixture")
                                              for m, boxes in self.texts.items()})
        self.start = time.monotonic()
        self.tmp = tempfile.TemporaryDirectory()
        outer = self

        class Head(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    body = outer.get("http://fixture" + self.path)
                    self.send_response(503 if outer.not_ready and self.path == "/health" else 200)
                    self.end_headers()
                    self.wfile.write(body.encode())
                except OSError:
                    self.send_response(503)
                    self.end_headers()

            def log_message(self, *_):
                pass

        class Supervisor(socketserver.StreamRequestHandler):
            def handle(self):
                request = json.loads(self.rfile.readline())
                self.wfile.write(json.dumps(outer.call("fixture", 7340, request)).encode() + b"\n")

        self.head = ThreadingHTTPServer(("localhost", 0), Head)
        self.supervisor = socketserver.ThreadingTCPServer(("localhost", 0), Supervisor)
        for server in (self.head, self.supervisor):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        url = "http://localhost:" + str(self.head.server_port)
        self.console_config = {"fixture": True, "allow_control": True, "state_dir": self.tmp.name,
                               "models": {m: {"head_url": url,
                                              "controls": [{"host": "localhost", "port": self.supervisor.server_address[1]} for _ in boxes],
                                              "fast_values": {"GR8_QUIT_FAST": "1", "GR8_GEN_CACHE": "/fixture/gen-cache", "GR8_LOAD_COMPACT": "/fixture/compact", "GR8_LOAD_MERGE_KB": "1", "KV_STORE": "ssd:/fixture/kv,size=1G,persist=on,keep=sessions", "GR8_STOP_GRACE": "150s"} if m == "glm53" else {}}
                                          for m, boxes in self.texts.items()}}
        for m in self.config.models.values():
            m.health_url = url + "/health"

    def get(self, url):
        if self.active is None:
            raise OSError("fixture stopped")
        if url.endswith("/health"):
            return json.dumps({"ready": not self.not_ready, "steering": {"enabled": False, "forced": False}})
        if url.endswith("/v1/models"):
            return json.dumps({"object": "list", "data": [{"id": DEFAULT_IDS[self.active], "object": "model", "max_model_len": 32768}]})
        if url.endswith("/metrics"):
            seconds = time.monotonic() - self.start
            values = {"generation_tokens_total": 1000 + int(seconds * 12), "prompt_tokens_total": 8000 + int(seconds * 200),
                      "num_requests_running": 1, "num_requests_waiting": 0,
                      "kv_cache_size_tokens": 65536, "kv_active_tokens": 2048,
                      "time_to_first_token_seconds_sum": 2, "time_to_first_token_seconds_count": 10,
                      "prefix_cache_hits_total": 200, "prefix_cache_queries_total": 1000,
                      "spec_decode_num_draft_tokens_total": 10, "spec_decode_num_accepted_tokens_total": 7,
                      "request_success_total": 10}
            return "\n".join(f'vllm:{k}{{model_name="{DEFAULT_IDS[self.active]}",engine="0"}} {v}' for k, v in values.items()) + '\nvllm:cache_config_info{kv_cache_max_concurrency="4"} 1\n'
        raise OSError("unknown route")

    def call(self, host, port, message):
        if self.refuse_supervisor:
            return {"t": "error", "error": "caller is not a member of this cluster"}
        if message["t"] == "status":
            return {"t": "status", "rank": 0, "state": "serving" if self.active else "stopped",
                    "mem_gib": 20, "engine": bool(self.active), "model_id": self.live_id if self.switching else self.active}
        if message["t"] == "logs":
            # "@ID@" in a log is the model_id of the deployment that answers.
            text = self.supervisor_log if self.switching and self.new_up else self.old_log
            return {"t": "log", "data": "SYNTHETIC FIXTURE · not live measurements\nhealth ready · all ranks joined\n" + text.replace("@ID@", self.live_id or ""), "offset": 100}
        raise ValueError("unknown RPC")

    def read_all(self, model):
        return {b: {"text": t, "revision": envfile.digest(t)} for b, t in self.texts[model].items()}

    def environment(self, model, box, action="read", **kwargs):
        if action == "read":
            text = self.texts[model][box]
            return {"text": text, "revision": envfile.digest(text)}
        if self.fail_write == box:
            self.fail_write = None
            raise OSError("fixture write failure")
        if envfile.digest(self.texts[model][box]) != kwargs["expected"]:
            raise ValueError("stale dotenv")
        self.texts[model][box] = kwargs["text"]
        self.calls.append(("write", model, box))
        return {"revision": envfile.digest(kwargs["text"])}

    def stop(self, model, emit):
        self.calls.append(("stop", model))
        if self.on_stop:
            self.on_stop()
        self.active = None
        emit("fixture: down returned on all boxes")

    def variant_cache(self, model, variant, records, refresh=False):
        spec = self.missing_weights or {}
        if spec.get("unknown"):
            return {"state": "unknown", "present": False, "size_gb": None, "rate_mb_s": None,
                    "unchecked": list(spec.get("unchecked") or [1])}
        if spec.get("variant") != variant:
            return {"state": "present", "present": True, "size_gb": None, "rate_mb_s": None, "unchecked": []}
        return {"state": "missing", "present": False, "size_gb": spec.get("size_gb"),
                "rate_mb_s": spec.get("rate_mb_s"), "unchecked": []}

    def activate(self, model, emit, health_timeout=None, abort=None):
        self.calls.append(("switch", model, health_timeout))
        self.last_health_timeout = health_timeout
        self.switching = True
        self.live_id, self.new_up = self.active, not self.old_phase
        self.old_log += self.race_log
        try:
            if self.fail_boot:
                raise RuntimeError("fixture readiness timeout")
            if self.old_phase:
                if abort is not None and abort.wait(self.old_phase):
                    raise RuntimeError("switch command stopped: the engine failed to boot")
                self.live_id, self.new_up = model, True
            else:
                self.live_id = model
            # A switch that is still waiting for health; the console ends it when it sees the engine fail.
            if self.activate_wait and abort is not None and abort.wait(self.activate_wait):
                raise RuntimeError("switch command stopped: the engine failed to boot")
            self.active = model
            emit("fixture: gr8 switch completed; health ready")
        finally:
            self.switching = False

    def close(self):
        for server in (self.head, self.supervisor):
            server.shutdown()
            server.server_close()
        self.tmp.cleanup()
