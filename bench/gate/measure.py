"""The measurements: needle, long prompt, pbench, workloads, mixed streams. Each returns a plain dict."""
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import client as C
import docs

PBENCH = Path(__file__).resolve().parent.parent / 'pbench.py'

WORKLOADS = {
    'code': [
        'Write a complete Python module that implements a thread-safe rate limiter with a token bucket, including '
        'docstrings and a small usage example.',
        'Write a Go program that reads lines from standard input and prints the ten most frequent words with counts. '
        'Include comments.',
        'Implement a binary heap in C++ as a class template with push, pop, and a heapify constructor. Add comments.',
    ],
    'prose': [
        'Write a vivid 600-word short story about a lighthouse keeper who finds a letter in a bottle. Plain prose, '
        'no headings.',
        'Write a 600-word essay on why small towns change slowly, in flowing paragraphs.',
        'Describe, in about 600 words of narrative prose, a thunderstorm rolling over a wheat farm at night.',
    ],
    'structured': [
        'Return only a JSON array of 12 objects describing fictional books, each with title, author, year, genre, '
        'pages, and a "reviews" list of two objects with reviewer and score.',
        'Return only a JSON object that models a small company: departments (list), each with name, budget, and '
        'employees (list of objects with name, role, salary, start_date).',
        'Return only a YAML document describing a CI pipeline with six jobs, each with name, image, needs, '
        'script (list) and artifacts.',
    ],
}
MIXED4 = ['code', 'prose', 'structured', 'prose']
OK_FINISH = ('stop', 'length')


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rate(r):
    """Decode tok/s of one streamed response (None when it is too short to time)."""
    return round((r['tokens'] - 1) / r['decode_s'], 2) if r['tokens'] >= 50 and r['decode_s'] > 0 else None


def needle(cl, sizes, seed, depth=0.15, max_tokens=400, effort='nothink'):
    """One needle request per size. Pass if the answer contains the passphrase."""
    import random
    out = {}
    for tokens in sizes:
        secret = docs.passphrase(random.Random(seed ^ tokens))
        messages, info = docs.needle_prompt(cl, tokens, depth, seed ^ tokens, secret)
        s0 = C.spec_snapshot(cl.metrics())
        r = cl.chat(messages, max_tokens, temperature=0.0, effort=effort)
        s1 = C.spec_snapshot(cl.metrics())
        norm = lambda t: ''.join(t.lower().split())
        in_content = norm(secret) in norm(r['content'])
        in_reasoning = norm(secret) in norm(r['reasoning'])
        out[f'{tokens // 1000}k'] = {
            'target_tokens': tokens, 'passed': in_content, 'found_in_reasoning_only': in_reasoning and not in_content,
            'prompt_tokens': r['prompt_tokens'], 'cached_tokens': r['cached_tokens'], 'ttft_s': round(r['ttft'], 3),
            'completion_tokens': r['tokens'], 'finish_reason': r['finish_reason'], 'answer_head': r['content'][:200],
            'prefix_cache_hit_tokens': s1['prefix_hits'] - s0['prefix_hits'], **info}
    return out


def long_prompt(cl, sizes, seed, decode_tokens=300, effort='nothink'):
    """One streamed request per size: TTFT, prefill tok/s, decode tok/s. Each prompt is salted, so it is cold unless
    the server reports prefix hits; those are recorded and the note says so."""
    ask = 'Summarize in plain prose what this code does, in about 250 words.'
    out = {}
    for tokens in sizes:
        messages = docs.filler_prompt(cl, tokens, seed ^ (tokens * 7), ask)
        s0 = C.spec_snapshot(cl.metrics())
        r = cl.chat(messages, decode_tokens, temperature=0.7, effort=effort)
        s1 = C.spec_snapshot(cl.metrics())
        hit = s1['prefix_hits'] - s0['prefix_hits']
        cached = r['cached_tokens'] or 0
        row = {'target_tokens': tokens, 'prompt_tokens': r['prompt_tokens'], 'ttft_s': round(r['ttft'], 3),
               'prefill_tok_s': round(r['prompt_tokens'] / r['ttft'], 1) if r['ttft'] > 0 else None,
               'decode_tokens': r['tokens'], 'decode_tok_s': rate(r), 'cached_tokens': r['cached_tokens'],
               'prefix_cache_hit_tokens': hit}
        if hit > 0 or cached > 0:
            row['note'] = (f'prefix cache hit: {max(hit, cached)} of {r["prompt_tokens"]} prompt tokens were reused, '
                           'so TTFT and prefill tok/s are not cold')
        out[f'{tokens // 1000}k'] = row
    return out


def pbench(url, sizes, reps, extra):
    """Run bench/pbench.py and return its summary line."""
    cmd = [sys.executable, str(PBENCH), '--url', url, '--tag', 'gate', '--reps', str(reps),
           '--sizes', ','.join(map(str, sizes))] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def _run_one(cl, kind, i, tokens, temp, effort):
    prompts = WORKLOADS[kind]
    msg = prompts[i % len(prompts)]
    r = cl.chat([{'role': 'user', 'content': msg}], tokens, temperature=temp, effort=effort)
    return {'tokens': r['tokens'], 'tok_s': rate(r), 'ttft_s': round(r['ttft'], 3), 'finish_reason': r['finish_reason'],
            'ok': r['finish_reason'] in OK_FINISH}


def workloads(cl, runs=3, tokens=300, temp=0.7, effort='nothink', kinds=('code', 'prose', 'structured')):
    """Decode tok/s per workload (a few runs of ~tokens each) with the drafter counters of that workload."""
    out = {}
    for kind in kinds:
        s0 = C.spec_snapshot(cl.metrics())
        rows = [_run_one(cl, kind, i, tokens, temp, effort) for i in range(runs)]
        s1 = C.spec_snapshot(cl.metrics())
        failed = sum(not r['ok'] for r in rows)
        # a workload with any failed run has no decode number: the gate then reports it missing
        out[kind] = {'runs': rows, 'failed_runs': failed,
                     'decode_tok_s': None if failed else med([r['tok_s'] for r in rows]),
                     'spec': C.spec_delta(s0, s1)}
    return out


def load_prompt_set(path):
    """A prompt set from another bench, as exact token ids: {"name": ..., "prompts": [{"id", "kind", "prompt_token_ids"}]}.
    -> (set, sha256 of the file). The name goes into the metric names, so every number says which set it comes from."""
    raw = Path(path).read_bytes()
    ps = json.loads(raw)
    if not isinstance(ps.get('name'), str) or not ps['name'] or '.' in ps['name']:
        raise ValueError(f'{path}: "name" must be a non-empty string without dots')
    for p in ps.get('prompts') or []:
        if not (p.get('id') and p.get('kind') and isinstance(p.get('prompt_token_ids'), list) and p['prompt_token_ids']):
            raise ValueError(f'{path}: every prompt needs id, kind and a non-empty prompt_token_ids list')
    if not ps.get('prompts'):
        raise ValueError(f'{path}: no prompts')
    return ps, hashlib.sha256(raw).hexdigest()


def prompt_set(cl, ps, sha, tokens=512, temp=0.0):
    """Decode tok/s and drafter counters per kind on a prompt set given as token ids (sent to /v1/completions as they
    are, so the other bench's chat template and system message are kept exactly)."""
    out = {'name': ps['name'], 'sha256': sha, 'kinds': {}}
    for kind in sorted({p['kind'] for p in ps['prompts']}):
        s0 = C.spec_snapshot(cl.metrics())
        rows = []
        for p in (q for q in ps['prompts'] if q['kind'] == kind):
            r = cl.complete_stream(p['prompt_token_ids'], tokens, temperature=temp)
            rows.append({'id': p['id'], 'tokens': r['tokens'], 'tok_s': rate(r), 'ttft_s': round(r['ttft'], 3),
                         'finish_reason': r['finish_reason'], 'ok': r['finish_reason'] in OK_FINISH})
        s1 = C.spec_snapshot(cl.metrics())
        failed = sum(not r['ok'] for r in rows)
        out['kinds'][kind] = {'runs': rows, 'failed_runs': failed,
                              'decode_tok_s': None if failed else med([r['tok_s'] for r in rows]),
                              'spec': C.spec_delta(s0, s1)}
    return out


def mixed4(cl, runs=3, tokens=300, temp=0.7, effort='nothink'):
    """Four concurrent mixed streams (needs multi-session). Aggregate tok/s = all tokens / wall time of the batch."""
    batches = []
    s0 = C.spec_snapshot(cl.metrics())
    for i in range(runs):
        res = [None] * len(MIXED4)

        def go(j):
            try:
                res[j] = _run_one(cl, MIXED4[j], i + j, tokens, temp, effort)
            except Exception as e:  # keep the other streams' numbers; the batch is marked failed below
                res[j] = {'tokens': 0, 'tok_s': None, 'ttft_s': None, 'finish_reason': None, 'ok': False,
                          'error': repr(e)}
        t0 = time.time()
        ths = [threading.Thread(target=go, args=(j,)) for j in range(len(MIXED4))]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.time() - t0
        batches.append({'streams': res, 'wall_s': round(wall, 2), 'ok': all(r['ok'] for r in res),
                        'agg_tok_s': round(sum(r['tokens'] for r in res) / wall, 2)})
    s1 = C.spec_snapshot(cl.metrics())
    failed = sum(not r['ok'] for b in batches for r in b['streams'])
    # one failed stream means four concurrent streams did not run: no speed numbers, the gate reports them missing
    return {'batches': batches, 'failed_streams': failed,
            'agg_tok_s': None if failed else med([b['agg_tok_s'] for b in batches]),
            'stream_tok_s': None if failed else med([r['tok_s'] for b in batches for r in b['streams']]),
            'spec': C.spec_delta(s0, s1)}
