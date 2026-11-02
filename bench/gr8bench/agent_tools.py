"""A small agent against a sandboxed file tree: can the model drive real tools?

Each task starts from a fresh in-memory tree, offers list_dir, read_file,
search_files, write_file and finish, runs the model's calls itself, and checks
the tree (or the finish answer) at the end. Every task runs streamed and not
streamed, because the two take different parser paths.

From the glm-5.3-flash-gx10 recipe (experimental/quality/agent_tools.py); run() is the entry point.

Prints one line per task and mode, then totals: tasks passed, turns, calls,
calls refused by the server (the fail-closed parser's sentinel), calls to tools
not offered, and calls whose arguments did not parse.
"""
import copy
import json
import re
import urllib.request

REFUSED = "_rejected_by_server"
MAX_TURNS = 12

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory's entries.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a whole file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_files", "description":
     "Search file contents under a directory with a regular expression; returns path:line:text matches.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or replace a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the task with a short answer.",
     "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]

TREE = {
    "/app/README.md": "# Inventory service\nRun `make serve`. Settings live in config/.\n",
    "/app/config/base.yaml": "service: inventory\nlog_level: info\n",
    "/app/config/prod.yaml": "listen_port: 8443\nreplicas: 3\ndatabse_url: postgres://db:5432/inv\n",
    "/app/config/dev.yaml": "listen_port: 8080\nreplicas: 1\ndatabse_url: postgres://localhost/inv\n",
    "/app/src/server.py": "# TODO: graceful shutdown\nimport config\n\ndef serve():\n    # TODO: TLS\n    pass\n",
    "/app/src/store.py": "def get(item):\n    return None  # TODO: cache\n",
    "/app/src/util.py": "def slug(s):\n    return s.lower().replace(' ', '-')\n",
}


def _norm(path: str) -> str:
    path = "/" + (path or "/").strip().lstrip("./")
    return re.sub(r"/+", "/", path).rstrip("/") or "/"


class Sandbox:
    def __init__(self) -> None:
        self.files = copy.deepcopy(TREE)

    def list_dir(self, path: str) -> str:
        path = _norm(path)
        entries = set()
        for f in self.files:
            if f.startswith(path + "/") or path == "/":
                rest = f[len(path):].lstrip("/")
                entries.add(rest.split("/")[0] + ("/" if "/" in rest else ""))
        return "\n".join(sorted(entries)) or f"error: no such directory {path}"

    def read_file(self, path: str) -> str:
        return self.files.get(_norm(path), f"error: no such file {_norm(path)}")

    def search_files(self, pattern: str, path: str = "/") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"error: bad pattern: {e}"
        root = _norm(path)
        hits = [f"{f}:{i}:{line}" for f, text in sorted(self.files.items()) if f.startswith(root)
                for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        return "\n".join(hits[:50]) or "no matches"

    def write_file(self, path: str, content: str) -> str:
        self.files[_norm(path)] = content
        return f"wrote {len(content)} bytes to {_norm(path)}"


# (name, prompt, check(sandbox, finish_answer) -> bool)
TASKS = [
    ("find-port", "Which port does the production config listen on? Write just the number to /app/answer.txt, "
     "then finish.", lambda s, a: s.files.get("/app/answer.txt", "").strip() == "8443"),
    ("fix-typo", "The key `databse_url` is misspelled in every config file under /app/config. Fix it to "
     "`database_url` without changing anything else, then finish.",
     lambda s, a: all(s.files[f] == TREE[f].replace("databse_url", "database_url")
                      for f in TREE if f.startswith("/app/config/"))),
    ("count-todos", "How many TODO comments are there in /app/src? Finish with the number.",
     lambda s, a: re.search(r"\b3\b", a or "") is not None),
    ("new-file", "Create /app/config/staging.yaml copying prod.yaml but with listen_port 9090 and 2 replicas. "
     "Then finish.", lambda s, a: "listen_port: 9090" in s.files.get("/app/config/staging.yaml", "")
     and "replicas: 2" in s.files.get("/app/config/staging.yaml", "")
     and "postgres://db:5432/inv" in s.files.get("/app/config/staging.yaml", "")),
    ("read-then-answer", "What does slug() in /app/src/util.py do to spaces? Finish with a one-line answer.",
     lambda s, a: "-" in (a or "") or "hyphen" in (a or "").lower() or "dash" in (a or "").lower()),
]


def chat(url, model, messages, stream):
    body = {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 2048,
            "stream": stream, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r)["choices"][0]["message"]
        content, calls = "", {}
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0].get("delta", {})
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"id": "", "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["function"]["name"] += fn.get("name") or ""
                c["function"]["arguments"] += fn.get("arguments") or ""
        return {"role": "assistant", "content": content or None,
                "tool_calls": [calls[i] for i in sorted(calls)] or None}


def run_task(url, model, task, stream, verbose):
    name, prompt, check = task
    box, answer = Sandbox(), None
    stats = {"turns": 0, "calls": 0, "refused": 0, "unknown": 0, "bad_args": 0}
    messages = [{"role": "system", "content": "You operate on a small repository through the tools given. "
                 "Use them; do not guess file contents."},
                {"role": "user", "content": prompt}]
    for _ in range(MAX_TURNS):
        stats["turns"] += 1
        msg = chat(url, model, messages, stream)
        calls = msg.get("tool_calls") or []
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v})
        if not calls:
            # A plain reply ends the task; it counts as the answer.
            answer = answer if answer is not None else (msg.get("content") or "")
            break
        for call in calls:
            stats["calls"] += 1
            fn = call["function"]
            try:
                args = json.loads(fn["arguments"] or "{}")
            except json.JSONDecodeError:
                stats["bad_args"] += 1
                result = "error: arguments are not valid JSON"
            else:
                if REFUSED in args:
                    stats["refused"] += 1
                    result = f"error: {args[REFUSED]}"
                elif fn["name"] == "finish":
                    answer = str(args.get("answer", ""))
                    result = "done"
                elif hasattr(Sandbox, fn["name"]) and not fn["name"].startswith("_"):
                    try:
                        result = getattr(box, fn["name"])(**args)
                    except TypeError as e:
                        stats["bad_args"] += 1
                        result = f"error: {e}"
                else:
                    stats["unknown"] += 1
                    result = f"error: no tool named {fn['name']}"
            if verbose:
                print(f"      {fn['name']}({fn['arguments'][:80]}) -> {str(result)[:80]!r}")
            messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": str(result)})
        if answer is not None:
            break
    return check(box, answer), stats


def run(url, model):
    """Every task, whole and streamed. -> passes and the call counts."""
    totals = {"passed": 0, "runs": 0, "turns": 0, "calls": 0, "refused": 0, "unknown": 0, "bad_args": 0}
    tasks = []
    for task in TASKS:
        for stream in (False, True):
            ok, s = run_task(url.rstrip("/"), model, task, stream, False)
            totals["passed"] += ok
            totals["runs"] += 1
            for k in s:
                totals[k] += s[k]
            tasks.append({"task": task[0], "stream": stream, "passed": ok, **s})
    return {**totals, "tasks": tasks}
