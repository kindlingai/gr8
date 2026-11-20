#!/usr/bin/env python3
"""Greedy decoding corrupts tokens inside a tool call at long context.

From the glm-5.3-flash-gx10 recipe (dev/repro/toolcall_corruption.py); run() is the entry point.

Builds a 42272-token coding-agent transcript (47 tools, 24 tool-call/result
pairs, all generated from fixed seeds so every run sends identical bytes), then
sends it N times at temperature 0 with a fixed seed. The correct continuation is
a Bash call repeating a command from earlier in the conversation, so any
difference between runs is visible token by token.

Observed on 4x GB10 (sm121) at TP=4, GLM-5.3-Flash-NVFP4. One run of 40, all at
temperature 0 with the same seed, gave 5 distinct completions:

    30 runs  correct
     4 runs  pos 23  ' identical' for '>&'  -> pytest -q 2 identical 2>&1
     3 runs  pos 26  ' pytest'    for ' tail' -> ... 2>&1 | pytest -q 2>&1 | tail -40
     2 runs  pos 23  ' Bash'      for '>&'  -> pytest -q 2 Bash 2>&1
     1 run   pos 31  ' Read'      for 'description' -> <arg_key> Read more: https://...

A corrupted tool NAME is worse than a corrupted argument: the GLM parser runs
with validate_tool_names=True, so an unknown name emits zero deltas and the
request finishes `stop` with no content and no tool calls. Clients report that
as "completed response with no content".

Reducing the transcript weakens it. A short prompt (2k tokens) is bit-stable
over 16 runs, and a transcript whose padding repeats the command's own output
is stable over 40, because every token of the command is then high-margin.
"""
import collections
import json
import queue
import random
import threading
import time
import urllib.request

SYSTEM = """You are a coding agent operating inside a developer's repository. You help by reading files, searching the codebase, running commands, and editing code. Always investigate before you change anything. Prefer small, verifiable steps. When you need information from the repository, call a tool rather than guessing. When a task is complete, summarize what you did in plain language.

Guidelines:
- Never invent file contents. Read them.
- Use Grep to locate symbols and Glob to locate files by name.
- Use Bash only for commands that cannot be done with the other tools.
- Keep edits minimal and targeted; do not reformat unrelated code.
- If a command fails, read the error and decide the next step; do not repeat the same command blindly.
- Report absolute paths.
"""

def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

TOOLS = [
    tool("Read", "Read a file from the filesystem. Returns the content with line numbers.",
         {"file_path": {"type": "string", "description": "Absolute path"},
          "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["file_path"]),
    tool("Write", "Write a file, overwriting it.",
         {"file_path": {"type": "string"}, "content": {"type": "string"}}, ["file_path", "content"]),
    tool("Edit", "Exact string replacement in a file.",
         {"file_path": {"type": "string"}, "old_string": {"type": "string"},
          "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
         ["file_path", "old_string", "new_string"]),
    tool("Bash", "Run a shell command and return stdout/stderr.",
         {"command": {"type": "string"}, "timeout": {"type": "integer"},
          "description": {"type": "string"}}, ["command"]),
    tool("Grep", "Search file contents with a regex (ripgrep).",
         {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
          "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
          "-n": {"type": "boolean"}, "-i": {"type": "boolean"}}, ["pattern"]),
    tool("Glob", "Find files by glob pattern.",
         {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    tool("WebFetch", "Fetch a URL and return its text.",
         {"url": {"type": "string"}, "prompt": {"type": "string"}}, ["url", "prompt"]),
    tool("WebSearch", "Search the web.", {"query": {"type": "string"}}, ["query"]),
    tool("TodoWrite", "Replace the todo list.",
         {"todos": {"type": "array", "items": {"type": "object", "properties": {
             "content": {"type": "string"}, "status": {"type": "string",
             "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}},
             "required": ["content", "status", "activeForm"]}}}, ["todos"]),
    tool("Agent", "Launch a sub-agent for a multi-step task.",
         {"description": {"type": "string"}, "prompt": {"type": "string"},
          "subagent_type": {"type": "string"}}, ["description", "prompt"]),
    tool("NotebookEdit", "Edit a Jupyter notebook cell.",
         {"notebook_path": {"type": "string"}, "cell_id": {"type": "string"},
          "new_source": {"type": "string"}, "cell_type": {"type": "string"},
          "edit_mode": {"type": "string"}}, ["notebook_path", "new_source"]),
    tool("AskUserQuestion", "Ask the user a clarifying question.",
         {"questions": {"type": "array", "items": {"type": "object"}}}, ["questions"]),
    tool("KillShell", "Kill a background shell.", {"shell_id": {"type": "string"}}, ["shell_id"]),
    tool("TaskOutput", "Read output of a background task.",
         {"task_id": {"type": "string"}, "block": {"type": "boolean"}}, ["task_id"]),
    tool("EnterPlanMode", "Switch to planning mode.", {}, []),
    tool("ExitPlanMode", "Leave planning mode with a plan.", {"plan": {"type": "string"}}, ["plan"]),
    tool("LSP", "Language server query.",
         {"operation": {"type": "string"}, "filePath": {"type": "string"},
          "line": {"type": "integer"}, "character": {"type": "integer"}}, ["operation", "filePath"]),
    tool("SlashCommand", "Run a slash command.", {"command": {"type": "string"}}, ["command"]),
]

FILE_A = "\n".join(f"{i:>6}\t" + line for i, line in enumerate("""#!/usr/bin/env python3
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    command: list[str]
    retries: int = 0
    env: dict = field(default_factory=dict)


def load_jobs(path: Path) -> list[Job]:
    with path.open() as f:
        raw = json.load(f)
    jobs = []
    for item in raw["jobs"]:
        jobs.append(Job(name=item["name"], command=item["command"],
                        retries=item.get("retries", 0), env=item.get("env", {})))
    return jobs


def run_job(job: Job) -> int:
    import subprocess
    env = dict(os.environ)
    env.update(job.env)
    attempt = 0
    while True:
        started = time.monotonic()
        proc = subprocess.run(job.command, env=env, capture_output=True, text=True)
        elapsed = time.monotonic() - started
        if proc.returncode == 0:
            print(f"ok {job.name} in {elapsed:.1f}s")
            return 0
        attempt += 1
        print(f"fail {job.name} rc={proc.returncode} attempt={attempt}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        if attempt > job.retries:
            return proc.returncode


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: runner.py JOBS.json", file=sys.stderr)
        return 2
    jobs = load_jobs(Path(sys.argv[1]))
    worst = 0
    for job in jobs:
        rc = run_job(job)
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
""".splitlines(), 1))

TEST_OUT = """============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.2
collected 14 items

tests/test_runner.py ..........F...                                      [100%]

=================================== FAILURES ===================================
____________________________ test_retry_env_isolated ___________________________

    def test_retry_env_isolated(tmp_path, monkeypatch):
        monkeypatch.setenv("SHARED", "outer")
        job = Job(name="x", command=["sh", "-c", "test \\"$SHARED\\" = inner"], retries=1, env={"SHARED": "inner"})
>       assert run_job(job) == 0
E       assert 1 == 0

tests/test_runner.py:61: AssertionError
----------------------------- Captured stderr call -----------------------------
fail x rc=1 attempt=1
fail x rc=1 attempt=2
=========================== short test summary info ============================
FAILED tests/test_runner.py::test_retry_env_isolated - assert 1 == 0
========================= 1 failed, 13 passed in 0.41s =========================
"""

GREP_OUT = """/home/dev/proj/runner.py:32:    env = dict(os.environ)
/home/dev/proj/runner.py:33:    env.update(job.env)
/home/dev/proj/tests/test_runner.py:55:def test_retry_env_isolated(tmp_path, monkeypatch):
/home/dev/proj/tests/test_runner.py:57:    job = Job(name="x", command=["sh", "-c", "test \\"$SHARED\\" = inner"], retries=1, env={"SHARED": "inner"})
/home/dev/proj/docs/design.md:14:Each job's `env` overlays the process environment; keys set to null must unset."""

def tc(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

def extra_tools(n):
    out = []
    for i in range(n):
        out.append(tool(
            f"mcp__server{i%5}__op{i}",
            f"Operation {i} on server {i%5}: queries the {['inventory','billing','metrics','tickets','deploys'][i%5]} system and returns a JSON document. Use when the user asks about {['stock levels','invoices','latency','support cases','releases'][i%5]}.",
            {"query": {"type": "string", "description": "Free-text query"},
             "limit": {"type": "integer", "description": "Max rows", "default": 50},
             "since": {"type": "string", "description": "ISO-8601 timestamp"},
             "fields": {"type": "array", "items": {"type": "string"}}},
            ["query"]))
    return out

TOOLS47 = TOOLS + extra_tools(47 - len(TOOLS))

def big_file(seed, lines=220):
    rnd = random.Random(seed)
    words = "config loader parse validate schema emit retry backoff socket frame header payload checksum rotate flush queue worker shard replica leader follower epoch term".split()
    out = []
    for i in range(1, lines + 1):
        k = rnd.randint(2, 7)
        ident = "_".join(rnd.choice(words) for _ in range(2))
        out.append(f"{i:>6}\t" + ("    " * rnd.randint(0, 3)) + rnd.choice([
            f"def {ident}(self, {', '.join(rnd.choice(words) for _ in range(k))}):",
            f"{ident} = {rnd.choice(words)}.{rnd.choice(words)}({rnd.randint(0, 999)})",
            f"if {ident} is None or {rnd.choice(words)} > {rnd.randint(1, 64)}:",
            f"return {ident}",
            f"# {rnd.choice(words)} {rnd.choice(words)} {rnd.choice(words)} handled below",
            f"raise ValueError(f\"{ident} out of range: {{{rnd.choice(words)}}}\")",
            f"for {rnd.choice(words)} in {ident}:",
            f"self._{ident}[{rnd.choice(words)}] = {rnd.choice(words)}",
            "",
        ]))
    return "\n".join(out)

def build(nonce, pairs=24):
    msgs = [{"role": "system", "content": SYSTEM + f"\n\nSession: {nonce}\n"}]
    msgs.append({"role": "user", "content": "The nightly job runner in /home/dev/proj is flaky: some jobs report success but their env overrides are dropped on retry, and the metrics exporter shows gaps. Investigate the whole flow (runner.py, the exporter under svc/, and the tests), find the root causes, and fix them. Run the tests when you think you are done."})
    files = [f"/home/dev/proj/{p}" for p in ["runner.py", "svc/exporter.py", "svc/metrics.py", "svc/queue.py", "svc/config.py", "tests/test_runner.py", "tests/test_exporter.py", "svc/retry.py", "svc/shard.py", "svc/leader.py"]]
    rnd = random.Random(nonce)
    for i in range(pairs):
        cid = f"call_{i+1}"
        kind = i % 6
        if kind == 0:
            call = tc("Bash", {"command": "cd /home/dev/proj && python -m pytest -q 2>&1 | tail -40", "description": "Run tests"}, cid)
            result = TEST_OUT
        elif kind in (1, 2, 4):
            f = files[(i * 3) % len(files)]
            call = tc("Read", {"file_path": f}, cid)
            result = FILE_A if f.endswith("runner.py") else big_file(f + nonce[:2])
        elif kind == 3:
            call = tc("Grep", {"pattern": rnd.choice(["env", "retry", "flush", "shard", "epoch"]), "path": "/home/dev/proj", "output_mode": "content", "-n": True}, cid)
            result = GREP_OUT + "\n" + "\n".join(f"/home/dev/proj/svc/{rnd.choice(['exporter','queue','retry'])}.py:{rnd.randint(1,300)}:    {rnd.choice(['env','retry','flush'])} = {rnd.choice(['self','cfg','job'])}.{rnd.choice(['get','pop','update'])}(\"{rnd.choice(['SHARED','TIMEOUT','SHARD'])}\")" for _ in range(25))
        else:
            call = tc(f"mcp__server{i%5}__op{i%47}", {"query": "job runner gaps last 24h", "limit": 20}, cid)
            result = json.dumps({"rows": [{"ts": f"2026-09-0{1+(j%6)}T0{j%9}:1{j%5}:00Z", "job": f"nightly-{j}", "status": rnd.choice(["ok", "retry", "fail"]), "env_keys": rnd.randint(0, 5), "latency_ms": rnd.randint(50, 9000)} for j in range(20)], "total": 20})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
    return msgs

def post(base, path, payload, timeout=1800):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def run_once(base, model, messages, max_tokens):
    """Return the generated tokens, including the tool-call control tokens."""
    body = {"model": model, "messages": messages, "tools": TOOLS47,
            "temperature": 0, "seed": 1234, "max_tokens": max_tokens,
            "logprobs": True, "top_logprobs": 1}
    choice = post(base, "/v1/chat/completions", body)["choices"][0]
    return [t["token"] for t in (choice.get("logprobs") or {}).get("content") or []]


def run(base, model, runs, concurrency, max_tokens=40):
    """Send the 42K transcript `runs` times at temperature 0, `concurrency` at once. -> how many diverge from the
    majority completion, and where each minority first differs."""
    base = base.rstrip("/")
    messages = build("presstest")
    n_tokens = post(base, "/tokenize", {"model": model, "messages": messages, "tools": TOOLS47,
                                        "add_generation_prompt": True}, timeout=300)["count"]
    runs_out, lock, work = [], threading.Lock(), queue.Queue()
    for i in range(runs):
        work.put(i)

    def worker():
        while True:
            try:
                work.get_nowait()
            except queue.Empty:
                return
            try:
                tokens = run_once(base, model, messages, max_tokens)
            except Exception as exc:
                tokens = [f"<error {exc!r}>"]
            with lock:
                runs_out.append(tokens)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    texts = ["".join(t) for t in runs_out]
    groups = collections.Counter(texts)
    majority = groups.most_common(1)[0][0]
    reference = runs_out[texts.index(majority)]
    diverge = []
    for text, n in groups.most_common()[1:]:
        tokens = runs_out[texts.index(text)]
        pos = next((i for i in range(min(len(tokens), len(reference))) if tokens[i] != reference[i]),
                   min(len(tokens), len(reference)))
        diverge.append({"runs": n, "pos": pos, "got": tokens[pos] if pos < len(tokens) else "<end>",
                        "want": reference[pos] if pos < len(reference) else "<end>", "text": text[:200]})
    return {"prompt_tokens": n_tokens, "runs": runs, "concurrency": concurrency, "distinct": len(groups),
            "divergent": sum(d["runs"] for d in diverge), "majority": majority[:200], "diverge": diverge}
