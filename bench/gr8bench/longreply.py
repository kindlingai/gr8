"""Long replies at temperature 1.0: does decode drift away from prefill as a reply grows?

The NLL gate scores fixed texts, so it only sees the prefill side. This generates
REPLIES long replies (16,000 tokens, thinking high, temperature 1.0, unique
cache_salt each) with their decode-time logprobs, rescores the same token ids as
a prompt (prompt_logprobs), and compares the two per 2,000-token window.

On this stack decode and prefill differ by design (batch-size invariance traded
for speed), so the absolute gap is not the signal: about 0.05-0.31 nats mean
|diff| per window is normal here (first windows 0.13-0.30, usually falling as a
reply settles into code). The signal is growth: a reply that degrades as it gets
longer (a broken drafter, a cache or state bug that compounds) shows windows late
in the reply well above its first ones, or falls into a loop.

  FAIL  late windows (second half) average > 1.8x the first two windows, or any
        window > 0.40, or a loop: one block repeated back to back for 256 tokens
        or more (repetitive code, like test boilerplate with other names, is not
        back to back, so it does not count)
  PASS  otherwise (a reply that stops before 4,000 tokens is skipped; at least
        one must reach it, else INCONCLUSIVE)

Calibration, on per-window results saved from an equivalent checker (same
windows, decode vs prompt_logprobs): 22 16K replies from unaffected builds all
PASS (10 on this stack at 1173cc1..78b9540: late/first 0.92-1.37, max window
0.244; 12 on a vLLM-main build). On an 8-bit drafter that degraded long replies,
7 of 9 replies FAIL (late/first 2.31-5.59, or a window at 0.995); the 2 that
PASS show no growth within that reply. Loop rule: across 47 saved 16K replies
from this stack and earlier builds the longest back-to-back repeat in any window
was 13-50 tokens, so the 256 bar has about 5x headroom.

From the glm-5.3-flash-gx10 recipe (gate/longreply.py); run() is the entry point.
Each 16K reply took ~36 min here with ~30 other streams running; quiet is faster
(not measured on this stack).
"""
import concurrent.futures as cf, json, time, urllib.request

TOKENS, WIN, MIN_LEN = 16000, 2000, 4000
LOOP = 256  # tokens of back-to-back repetition that count as a loop
MESSAGES = [{"role": "user", "content":
    "Write a complete, production-quality Python implementation of a persistent key-value store with a write-ahead "
    "log, crash recovery, compaction, range scans and a small CLI. Explain each design decision as you go, then "
    "write thorough tests for every component, including crash-recovery tests that kill the process mid-write."}]


def post(base, path, body):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def generate(base, model, i):
    t = time.time()
    d = post(base, "/v1/chat/completions", {
        "model": model, "messages": MESSAGES, "temperature": 1.0, "max_tokens": TOKENS, "reasoning_effort": "high",
        "cache_salt": f"longreply-{time.time()}-{i}", "logprobs": True, "top_logprobs": 1,
        "return_tokens_as_token_ids": True})
    c = d["choices"][0]
    toks = [(int(x["token"].split(":", 1)[1]), x["logprob"]) for x in (c.get("logprobs") or {}).get("content") or []]
    return {"i": i, "secs": round(time.time() - t), "finish": c["finish_reason"], "tokens": toks}


def rescore(base, model, toks):
    pids = post(base, "/tokenize", {"model": model, "messages": MESSAGES, "add_generation_prompt": True})["tokens"]
    lp = post(base, "/v1/completions", {"model": model, "prompt": pids + [t for t, _ in toks], "max_tokens": 1,
                                  "temperature": 0, "prompt_logprobs": 0})["choices"][0]["prompt_logprobs"][len(pids):]
    return [next(iter(x.values()))["logprob"] if x else None for x in lp]


def longest_loop(ids, max_period=256):
    """Longest stretch in which every token equals the one p tokens earlier, for some period p <= max_period:
    a block repeated back to back. Similar blocks with different names (test boilerplate) break the stretch."""
    best = 0
    for p in range(1, min(max_period, len(ids) // 2) + 1):
        run = 0
        for j in range(p, len(ids)):
            run = run + 1 if ids[j] == ids[j - p] else 0
            best = max(best, run)
    return best


def windows(toks, res):
    """Per window of >= 500 tokens: (start, mean |diff|, per 1k > 1 nat, longest back-to-back repeat in tokens)."""
    out = []
    for k in range(0, len(toks), WIN):
        pairs = [(a[1], b) for a, b in zip(toks[k:k + WIN], res[k:k + WIN]) if b is not None]
        if len(pairs) < 500:
            continue
        diffs = [abs(a - b) for a, b in pairs]
        ids = [t for t, _ in toks[k:k + WIN]]
        out.append((k, sum(diffs) / len(diffs), 1000 * sum(x > 1 for x in diffs) / len(diffs), longest_loop(ids)))
    return out


def verdict(per_reply):
    """per_reply: list of window lists. Returns (verdict, reasons)."""
    scored = [w for w in per_reply if w and (w[-1][0] + WIN) >= MIN_LEN]
    if not scored:
        return "INCONCLUSIVE", [f"no reply reached {MIN_LEN} tokens"]
    bad = []
    for n, w in enumerate(scored):
        first = sum(x[1] for x in w[:2]) / len(w[:2])
        late = [x[1] for x in w[len(w) // 2:]]
        ratio = (sum(late) / len(late)) / first if first else 0
        if ratio > 1.8:
            bad.append(f"reply {n}: late windows {ratio:.2f}x the first two")
        if max(x[1] for x in w) > 0.40:
            bad.append(f"reply {n}: a window at {max(x[1] for x in w):.3f} mean |diff|")
        if max(x[3] for x in w) >= LOOP:
            bad.append(f"reply {n}: a block repeated back to back for {max(x[3] for x in w)} tokens (a loop)")
    return ("FAIL", bad) if bad else ("PASS", [])


def selftest():
    w = lambda *d: [(k * WIN, v, 0, 1) for k, v in enumerate(d)]
    clean = w(0.165, 0.150, 0.180, 0.190, 0.200, 0.190, 0.210, 0.180)        # this stack, flat
    grows = w(0.090, 0.085, 0.120, 0.180, 0.300, 0.350, 0.380, 0.390)        # collapses with length
    assert verdict([clean, clean]) == ("PASS", [])
    assert verdict([clean, grows])[0] == "FAIL"
    assert verdict([w(0.2, 0.5, 0.2, 0.2)])[0] == "FAIL"                     # one window past 0.40
    assert verdict([[(0, 0.1, 0, 300), (2000, 0.1, 0, 1)]])[0] == "FAIL"     # a loop
    assert verdict([w(0.1)])[0] == "INCONCLUSIVE"                            # 2,000 tokens only
    assert verdict([w(0.30, 0.29, 0.06, 0.05, 0.07, 0.12, 0.05, 0.06)]) == ("PASS", [])  # high first, falls (main)
    # Matt's false FAIL (2026-10-12): a 16-token run 9 times in a window, apart (code boilerplate): now passes.
    run, filler = list(range(1000, 1016)), lambda k: list(range(5000 + 150 * k, 5150 + 150 * k))
    ids = [t for k in range(9) for t in run + filler(k)] + list(range(9000, 9000 + 2000 - 9 * 166))
    toks = [(t, -0.5) for t in ids] * 2
    assert len(toks) == 4000 and windows(toks, [-0.6] * 4000)[0][3] < LOOP
    assert verdict([windows(toks, [-0.6] * 4000)]) == ("PASS", [])
    # A real loop: one 20-token block back to back 15 times (280 tokens) in otherwise varied text: fails.
    looped = list(range(20000, 21000)) + list(range(100, 120)) * 15 + list(range(30000, 30700))
    assert len(looped) == 2000 and longest_loop(looped) == 280
    toks = [(t, -0.5) for t in looped + list(range(40000, 42000))]
    assert verdict([windows(toks, [-0.6] * 4000)])[0] == "FAIL"
    print("selftest ok")


def run(base, model, replies=2):
    """`replies` long replies at once, each rescored as a prompt. -> the verdict and each reply's windows."""
    base = base.rstrip("/")
    with cf.ThreadPoolExecutor(replies) as ex:
        outs = list(ex.map(lambda i: generate(base, model, i), range(replies)))
    per_reply, rows = [], []
    for o in outs:
        w = windows(o["tokens"], rescore(base, model, o["tokens"]))
        per_reply.append(w)
        rows.append({"tokens": len(o["tokens"]), "secs": o["secs"], "finish": o["finish"],
                     "windows": [{"start": k, "mean_diff": round(d, 4), "over_1nat_per_1k": round(n, 1), "loop": rep}
                                 for k, d, n, rep in w]})
    v, why = verdict(per_reply)
    return {"verdict": v, "why": why, "replies": rows}
