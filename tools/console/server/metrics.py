"""Prometheus summaries: missing is null; rates are aggregate wall-time rates."""
import math
import re

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z_0-9:]*)(?:\{(.*?)\})?\s+(\S+)(?:\s+\S+)?$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text):
    out = {}
    info = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        name, labels, raw = match.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        out[name] = out.get(name, 0) + value
        if name == "vllm:cache_config_info":
            for key, raw_label in LABEL.findall(labels or ""):
                if key == "kv_cache_max_concurrency" and raw_label.isdecimal():
                    info["slots"] = int(raw_label)
    return out, info


def summarize(current, previous=None, elapsed=None):
    def get(name):
        return current.get("vllm:" + name)

    def ratio(a, b, scale=1):
        av, bv = get(a), get(b)
        return av / bv * scale if av is not None and bv is not None and bv > 0 else None

    def rate(name):
        key = "vllm:" + name
        a, b = current.get(key), (previous or {}).get(key)
        if a is None or b is None or not elapsed or elapsed <= 0 or a < b:
            return None
        return (a - b) / elapsed

    return {
        "decode": rate("generation_tokens_total"), "prefill": rate("prompt_tokens_total"),
        "ttft": ratio("time_to_first_token_seconds_sum", "time_to_first_token_seconds_count"),
        "running": get("num_requests_running"), "waiting": get("num_requests_waiting"),
        "kv_used": get("kv_active_tokens"), "kv_size": get("kv_cache_size_tokens"),
        "prefix_hits": ratio("prefix_cache_hits_total", "prefix_cache_queries_total", 100),
        "draft_accept": ratio("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"),
        "finished": get("request_success_total"),
    }
