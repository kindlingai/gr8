#!/usr/bin/env python3
"""S5 (#56) engine gate: one boot with NGRAM unset, one with it set, the same held-out prompts at temperature 0.
Stdlib only.

    # on each boot (NGRAM unset, then NGRAM=<table>), the same prompts and settings:
    python3 tools/ngram/ab.py run --url http://127.0.0.1:8100 --prompts tools/ngram/data/ab_prompts.json \
        --label off --out ab_off.json [--max-tokens 1024] [--effort low] [--traces traces.jsonl]
    python3 tools/ngram/ab.py run ... --label on --out ab_on.json
    # outputs byte for byte, decode tok/s and verify steps by workload
    python3 tools/ngram/ab.py compare ab_off.json ab_on.json

`run` templates each prompt on the server (/tokenize with messages and tools, as capture.py does), then sends
/v1/completions on those ids at temperature 0 with `return_token_ids` and `return_step_stats`, one request at a time
(the table only serves a request decoding alone). No log-probabilities are asked for, so decoding runs as it serves.
Per prompt it keeps the generated ids, decode tok/s ((tokens - 1) / step_stats.decode_s, the first token to the end on
the front end's clock; replies under 50 tokens are not timed) and the request's step counters. `--traces` refuses
the run, before any generation, when a prompt's ids are in a trace file: a table measured on its own prompts would
overstate the gain. The ids depend on `--effort`, so give `run` the effort the traces were captured with. A failed
request is kept as an error row and the run goes on.

`compare` checks that every prompt produced the same ids on both runs (the table drafts only, so any difference is a
bug), then prints, by workload, decode tok/s (median of the timed prompts, with their count) and decode tokens per
verify step on each side. It exits 1 on any output difference, any error row, or runs made with different settings,
prompt files or models. It cannot see whether the table served: check rank 0's `# s2 ngram slot` lines on the `on`
boot for nonzero table rounds (the table only serves a request decoding alone with DFlash2).
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture  # noqa: E402
from capture import Client, post  # noqa: E402

MIN_TIMED = 50      # bench/gate measure.rate(): shorter replies are not timed


def complete(c, ids, max_tokens):
    """/v1/completions on exact ids at temperature 0, not streamed (every generated id comes back in token_ids).
    -> generated ids, the request's step_stats (steps, draft and accepted tokens, decode_s on the shim's clock),
    finish reason, wall seconds."""
    body = {'model': c.model, 'prompt': ids, 'max_tokens': max_tokens, 'temperature': 0,
            'return_token_ids': True, 'return_step_stats': True}
    t0 = time.time()
    r = post(c, '/v1/completions', body)
    ch = r['choices'][0]
    return ch['token_ids'], (r.get('usage') or {}).get('step_stats') or {}, ch.get('finish_reason'), time.time() - t0


def trace_prompts(paths):
    out = set()
    for p in paths or ():
        with open(p) as f:
            for line in f:
                if line.strip():
                    out.add(tuple(json.loads(line)['prompt']))
    return out


def run(a):
    c = Client(a.url, model=a.model)
    with open(a.prompts) as f:
        items = json.load(f)
    seen = trace_prompts(a.traces)
    model = model_id(c)
    prompts = [capture.chat_ids(c, item, a.effort) for item in items]
    clash = [i for i, ids in enumerate(prompts) if tuple(ids) in seen]
    if clash:
        print(f'prompts {clash} are in the table\'s traces; hold them out', file=sys.stderr)
        return 2
    rows = []
    for i, (item, ids) in enumerate(zip(items, prompts)):
        row = {'i': i, 'workload': item.get('workload', 'other'), 'prompt_tokens': len(ids)}
        try:
            gen, st, finish, wall = complete(c, ids, a.max_tokens)
            if not isinstance(gen, list):
                raise ValueError(f'token_ids is {type(gen).__name__}, not a list')
        except (OSError, KeyError, IndexError, TypeError, ValueError) as e:
            row.update(gen=None, tokens=0, tok_s=None, steps=None, error=repr(e))
            rows.append(row)
            print(f'[{i}] {row["workload"]}: failed: {e}', file=sys.stderr)
            continue
        n = len(gen)
        dec_s = st.get('decode_s')
        tps = round((n - 1) / dec_s, 2) if n >= MIN_TIMED and dec_s else None
        row.update(gen=gen, tokens=n, decode_s=dec_s, wall_s=round(wall, 3), tok_s=tps, finish_reason=finish,
                   steps=st.get('steps'), draft_tokens=st.get('draft_tokens'),
                   accepted_tokens=st.get('accepted_tokens'))
        rows.append(row)
        print(f'[{i}] {row["workload"]}: {len(ids)} prompt, {n} generated, {tps} tok/s, {st.get("steps")} steps')
    out = {'label': a.label, 'model': model, 'prompts': os.path.basename(a.prompts),
           'prompts_sha': capture_sha(a.prompts),
           'settings': {'max_tokens': a.max_tokens, 'effort': a.effort or 'default', 'temperature': 0},
           'rows': rows}
    with open(a.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'{len(rows)} prompts -> {a.out}')
    return 0


def model_id(c):
    """The served model's id from /v1/models (None when the server has no such route)."""
    try:
        with urllib.request.urlopen(c.url + '/v1/models', timeout=60) as r:
            return [m.get('id') for m in json.load(r).get('data') or []]
    except (OSError, ValueError):
        return None


def capture_sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def per_step(rows):
    """Decode tokens (all but the first, which prefill samples) per verify step, over rows that report steps."""
    rs = [r for r in rows if r.get('steps')]
    return sum(r['tokens'] - 1 for r in rs) / sum(r['steps'] for r in rs) if rs else None


def compare(A, B):
    """-> (problems, markdown lines). A problem is a settings mismatch or a prompt whose ids differ."""
    probs = []
    for k in ('prompts_sha', 'settings', 'model'):
        if A.get(k) != B.get(k):
            probs.append(f'{k} differs: {A.get(k)} vs {B.get(k)}')
    ra, rb = {r['i']: r for r in A['rows']}, {r['i']: r for r in B['rows']}
    if set(ra) != set(rb):
        probs.append(f'prompt sets differ: {sorted(set(ra) ^ set(rb))}')
    same = 0
    for i in sorted(set(ra) & set(rb)):
        bad = [lab for lab, r in ((A.get('label', 'a'), ra[i]), (B.get('label', 'b'), rb[i])) if r.get('error')]
        if bad:
            probs.append(f'prompt {i} ({ra[i]["workload"]}): request failed on {", ".join(bad)}')
            continue
        x, y = ra[i]['gen'], rb[i]['gen']
        if x == y:
            same += 1
            continue
        d = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        probs.append(f'prompt {i} ({ra[i]["workload"]}): outputs differ at token {d} ({len(x)} vs {len(y)} tokens)')
    la, lb = A.get('label', 'a'), B.get('label', 'b')
    lines = [f'outputs: {same} of {len(set(ra) & set(rb))} prompts identical', '',
             f'| workload | prompts | timed {la} / {lb} | tok/s {la} | tok/s {lb} | ratio '
             f'| tokens / step {la} | tokens / step {lb} |',
             '|---|---|---|---|---|---|---|---|']
    ws = sorted({r['workload'] for r in A['rows']})
    for w in ws + ['all']:
        xa = [r for r in A['rows'] if w in ('all', r['workload'])]
        xb = [r for r in B['rows'] if w in ('all', r['workload'])]
        ta, tb = med([r['tok_s'] for r in xa]), med([r['tok_s'] for r in xb])
        sa, sb = per_step(xa), per_step(xb)
        f = (lambda v, p=2: '-' if v is None else f'{v:.{p}f}')
        ratio = f(tb / ta, 3) if ta and tb else '-'
        na, nb = (sum(r.get('tok_s') is not None for r in x) for x in (xa, xb))
        lines.append(f'| {w} | {len(xa)} | {na} / {nb} | {f(ta)} | {f(tb)} | {ratio} | {f(sa, 3)} | {f(sb, 3)} |')
    return probs, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run', help='one boot: every prompt, temperature 0, one at a time')
    p.add_argument('--url', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--label', required=True, help='e.g. off or on')
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--effort', default='', help="chat_template_kwargs reasoning_effort (default: the template's)")
    p.add_argument('--model', default='glm-5.3')
    p.add_argument('--traces', nargs='*', help="the table's trace files: refuse a prompt found in them")
    p = sub.add_parser('compare', help='two run files: identity, tok/s and tokens per step by workload')
    p.add_argument('a')
    p.add_argument('b')
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        return run(a)
    with open(a.a) as f:
        A = json.load(f)
    with open(a.b) as f:
        B = json.load(f)
    probs, lines = compare(A, B)
    print('\n'.join(lines))
    for p in probs:
        print('FAIL: ' + p)
    return 1 if probs else 0


if __name__ == '__main__':
    sys.exit(main())
