#!/usr/bin/env python3
"""Numeric session traces for kvman-sim from a capture proxy's JSON lines.

    sim/capture_trace.py CAPTURE.jsonl [CAPTURE2.jsonl ...] > TRACE.csv

Several captures make one trace that starts them all at once (an N-agent replay), sessions numbered across them.

Each capture line: {n, path, t_start, t_first, t_end, body: {messages, tools, max_tokens, ...}, usage: {prompt_tokens,
completion_tokens}}. Requests with tools are grouped into sessions by their first user message, one session per
group, numbered in the order they start. Requests without tools (title and summary calls) are left out. Per request:

    session, turn          the session's number and the request's place in it, by start time
    prompt_tokens          usage.prompt_tokens
    overlap_tokens         the longest common prefix of this request and the session's previous one, serialized
                           (tools, then system messages, then the other messages, as the GLM template renders them),
                           in characters, scaled to tokens by this request's prompt_tokens per character (at most
                           either request's prompt_tokens), or the previous request's prompt_tokens when this
                           one holds all of it
    completion_tokens      usage.completion_tokens
    think_s                seconds from the previous request's end to this one's start (0 when they overlap). For a
                           session's first request, seconds from the start of the first request in the capture
    max_tokens             body.max_tokens (0 when absent)
    ttft_s                 t_first - t_start as measured
    system_tokens          the session's system span: its tools and system messages, serialized, scaled to tokens as
                           overlap_tokens is (what the front end sends as the span, roughly)
    system_shared          the longest common prefix of that span with an earlier session's span, in tokens
    system_from            that earlier session (-1 when none shares any of it)

Only numbers go out: no prompt text.
"""
import json
import os
import sys


def serialize(body):
    msgs = body.get("messages") or []
    system = [m for m in msgs if m.get("role") == "system"]
    rest = [m for m in msgs if m.get("role") != "system"]
    parts = [json.dumps(body.get("tools") or [], sort_keys=True)]
    parts += [json.dumps(m, sort_keys=True) for m in system]
    parts += [json.dumps(m, sort_keys=True) for m in rest]
    return "".join(parts)


def first_user(body):
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            return json.dumps(m.get("content"), sort_keys=True)
    return ""


def system_span(body):
    msgs = body.get("messages") or []
    return json.dumps(body.get("tools") or [], sort_keys=True) + \
        "".join(json.dumps(m, sort_keys=True) for m in msgs if m.get("role") == "system")


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if (r.get("body") or {}).get("tools") and r.get("usage")]
    rows.sort(key=lambda r: r["t_start"])
    if not rows:
        sys.exit(f"{path}: no requests with tools")
    t0 = rows[0]["t_start"]
    sessions = {}
    for r in rows:
        sessions.setdefault(first_user(r["body"]), []).append(r)
    return t0, list(sessions.values())


def main(paths):
    groups = []
    for path in paths:
        t0, sessions = load(path)
        groups += [(t0, reqs) for reqs in sessions]
    out = ["session,turn,prompt_tokens,overlap_tokens,completion_tokens,think_s,max_tokens,ttft_s,"
           "system_tokens,system_shared,system_from"]
    spans = []
    for s, (t0, reqs) in enumerate(groups):
        first = reqs[0]
        span, text = system_span(first["body"]), serialize(first["body"])
        per = int(first["usage"]["prompt_tokens"]) / max(1, len(text))
        shared, src = 0, -1
        for i, other in enumerate(spans):
            c = common_prefix(other, span)
            if c > shared:
                shared, src = c, i
        spans.append(span)
        sys_cols = f"{round(len(span) * per)},{round(shared * per)},{src}"
        prev = None
        for turn, r in enumerate(reqs):
            text = serialize(r["body"])
            prompt = int(r["usage"]["prompt_tokens"])
            overlap = 0
            if prev is not None and text:
                old = serialize(prev["body"])
                chars = common_prefix(old, text)
                overlap = min(prompt, int(prev["usage"]["prompt_tokens"]), round(chars * prompt / len(text)))
                if chars == len(old):   # it holds the whole previous request: a continuation, exactly
                    overlap = min(prompt, int(prev["usage"]["prompt_tokens"]))
            think = r["t_start"] - t0 if prev is None else max(0.0, r["t_start"] - prev["t_end"])
            ttft = (r.get("t_first") or r["t_end"]) - r["t_start"]
            out.append(f"{s},{turn},{prompt},{overlap},{int(r['usage']['completion_tokens'])},{think:.3f},"
                       f"{int(r['body'].get('max_tokens') or 0)},{ttft:.3f},{sys_cols}")
            prev = r
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not all(os.path.exists(p) for p in sys.argv[1:]):
        sys.exit(__doc__)
    main(sys.argv[1:])
