"""The measurements. Each stage takes the run context and returns a plain dict for the report."""
import hashlib
import json
import random
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from . import agent_tools, client as C, docs, kl, longreply, prompts as P, quality, stats, toolcall

OK_FINISH = ('stop', 'length')
# Hangul and the replacement character: GLM-5.3 corruption has shown up as both
SUSPECT = re.compile('[가-힣�]')
DECODE_TOKENS = {'code': 512, 'prose': 512, 'structured': 512, 'counting': 2048}
REPO = Path(__file__).resolve().parents[2]


class Ctx:
    def __init__(self, cl, a, log):
        self.cl, self.a, self.log = cl, a, log
        self.requests = []     # single-stream requests in send order (route_bytes.py --drive reads them)
        self.suspect = []      # replies with suspect characters: (stage, what, excerpt)

    def chat(self, stage, kind, messages, max_tokens, **kw):
        r = self.cl.chat(messages, max_tokens, effort=kw.pop('effort', self.a.effort), **kw)
        text = r['content'] + r['reasoning']
        if SUSPECT.search(text):
            m = SUSPECT.search(text)
            self.suspect.append({'stage': stage, 'kind': kind, 'at': text[max(0, m.start() - 40):m.start() + 40]})
        self.requests.append({'stage': stage, 'kind': kind, 'prompt_tokens': r['prompt_tokens'], 'tokens': r['tokens'],
                              'finish': r['finish_reason']})
        return r


def rate(r):
    """Decode tok/s of one streamed reply: tokens after the first over the time after the first."""
    return (r['tokens'] - 1) / r['decode_s'] if r['tokens'] >= 2 and r['decode_s'] > 0 else None


def _problem(work, r):
    if r['finish_reason'] not in OK_FINISH:
        return f"finish {r['finish_reason']}"
    if SUSPECT.search(r['content'] + r['reasoning']):
        return 'suspect characters'
    if work == 'counting' and [x.strip() for x in r['content'].strip().split('\n')] != [str(k) for k in range(1, 201)]:
        return 'wrong count'
    return '' if (r['content'] + r['reasoning']).strip() else 'empty'


# ---------------------------------------------------------------- setup


def info(ctx):
    m = ctx.cl.models()[0]
    return {'model': m['id'], 'max_model_len': m.get('max_model_len'), 'url': ctx.cl.url}


def smoke(ctx):
    script = REPO / 'smoketest' / ('ds41.sh' if ctx.cl.model.startswith('deepseek') else 'run.sh')
    p = subprocess.run(['bash', str(script), ctx.cl.url, ctx.cl.model], capture_output=True, text=True, timeout=1800)
    lines = p.stdout.strip().splitlines()
    m = re.match(r'(\d+)/(\d+) passed', lines[-1]) if lines else None
    return {'passed': int(m.group(1)) if m else 0, 'cases': int(m.group(2)) if m else None,
            'failed': [x.strip()[5:] for x in lines if x.startswith('  FAIL ')], 'exit': p.returncode}


# ---------------------------------------------------------------- correctness


def t0(ctx):
    """Fixed prompts at temperature 0, one at a time: a hash of every reply, to compare builds byte for byte."""
    replies = []
    for p in P.T0:
        r = ctx.chat('correct', 't0', [{'role': 'user', 'content': p}], 256, temperature=0.0)
        replies.append({'prompt': p[:60], 'content': r['content'], 'reasoning': r['reasoning']})
    blob = json.dumps(replies, sort_keys=True).encode()
    return {'hash': hashlib.sha256(blob).hexdigest()[:12], 'replies': replies}


def count(ctx, n):
    runs = []
    for i in range(n):
        r = ctx.chat('correct', 'count', P.tagged(P.COUNT, P.run_tag(ctx.a.series, 'count', i)), 2048)
        runs.append({'clean': _problem('counting', r) == '', 'tokens': r['tokens'], 'finish': r['finish_reason']})
    return {'runs': n, 'clean': sum(r['clean'] for r in runs), 'detail': runs}


def cached(ctx):
    """The same salted ~400-token prompt twice: the second should report reused prompt tokens."""
    msgs = docs.filler_prompt(ctx.cl, 400, random.randrange(1 << 30), 'Name one function above. One line.')
    a = ctx.chat('correct', 'cached', msgs, 16)
    b = ctx.chat('correct', 'cached', msgs, 16)
    return {'prompt_tokens': b['prompt_tokens'], 'first': a['cached_tokens'], 'second': b['cached_tokens'],
            'ttft_first_s': round(a['ttft'], 3), 'ttft_second_s': round(b['ttft'], 3)}


def kl_check(ctx, ref_path):
    ref = json.loads(Path(ref_path).read_text())
    if ref.get('model') and ref['model'] != ctx.cl.model:
        return {'skipped': f"reference is for {ref['model']}"}
    out = kl.compare(ctx.cl, ref, ref.get('k', 20))
    out['reference'] = str(ref_path)
    return out


def needle(ctx, sizes, max_len):
    out = {}
    for tokens in sizes:
        if max_len and tokens + 1000 > max_len:
            out[f'{tokens // 1024}K'] = {'skipped': f'over max_model_len {max_len}'}
            continue
        seed = random.randrange(1 << 30)
        secret = docs.passphrase(random.Random(seed))
        messages, where = docs.needle_prompt(ctx.cl, tokens, 0.15, seed, secret)
        r = ctx.chat('correct', 'needle', messages, 400, temperature=0.0)
        norm = lambda t: ''.join(t.lower().split())
        out[f'{tokens // 1024}K'] = {'passed': norm(secret) in norm(r['content']),
                                     'in_reasoning_only': norm(secret) in norm(r['reasoning'])
                                     and norm(secret) not in norm(r['content']),
                                     'prompt_tokens': r['prompt_tokens'], 'ttft_s': round(r['ttft'], 2), **where}
    return out


def correctness(ctx, level, max_len):
    out = {'t0': t0(ctx), 'count': count(ctx, 5 if level == 'full' else 3), 'cached': cached(ctx)}
    if ctx.a.kl_ref:
        out['kl'] = kl_check(ctx, ctx.a.kl_ref)
    sizes = [32768, 131072, 262144, 491520] if level == 'full' else [131072]
    out['needle'] = needle(ctx, sizes, max_len)
    return out


# ---------------------------------------------------------------- decode


def decode_run(ctx, work, i, warm=False):
    tag = P.run_tag(ctx.a.series, work, f'warm{i}' if warm else i)
    s0 = C.spec_snapshot(ctx.cl.metrics())
    r = ctx.chat('decode', work, P.tagged(P.prompt_for(work, i), tag), DECODE_TOKENS[work])
    s1 = C.spec_snapshot(ctx.cl.metrics())
    d = C.spec_delta(s0, s1)
    steps = d['drafts'] or max(r['tokens'] - 1, 1)   # no drafter: one token a step
    return {'work': work, 'i': i, 'tag': tag, 'tokens': r['tokens'], 'ttft': r['ttft'], 'decode_s': r['decode_s'],
            'tps': rate(r) or 0.0, 'tok_per_step': d['tokens_per_step'] or 1.0,
            'ms_per_step': 1000 * r['decode_s'] / steps, 'accept': d['acceptance'], 'problem': _problem(work, r)}


def decode(ctx, works, runs, warmup):
    out = {}
    for w in works:
        for j in range(warmup):
            decode_run(ctx, w, j, warm=True)
        rs = [decode_run(ctx, w, i) for i in range(runs)]
        good = [r for r in rs if not r['problem']]
        out[w] = {'runs': rs, 'problems': [f"run {r['i']}: {r['problem']}" for r in rs if r['problem']],
                  'summary': stats.summarize(good) if good else None}
        s = out[w]['summary']
        ctx.log(f"  decode {w}: {s['tps']:.1f} tok/s" if s else f'  decode {w}: every run failed')
    return out


# ---------------------------------------------------------------- concurrency


def conc_batch(ctx, n, b):
    jobs = [(P.prompt_for('mixed', b * n + s), P.run_tag(ctx.a.series, f'c{n}', f'{b}.{s}')) for s in range(n)]
    res = [None] * n

    def go(s):
        t = time.time()
        try:
            r = ctx.cl.chat(P.tagged(*jobs[s]), 512, effort=ctx.a.effort)
            res[s] = {'sent': t, 'first': t + r['ttft'], 'end': t + r['total_s'], 'tokens': r['tokens'],
                      'tps': rate(r), 'ttft': r['ttft'], 'problem': _problem('mixed', r)}
        except Exception as e:   # the other streams keep their numbers; this batch is marked failed
            res[s] = {'sent': t, 'first': None, 'end': time.time(), 'tokens': 0, 'tps': None, 'ttft': None,
                      'problem': repr(e)}
    s0 = C.spec_snapshot(ctx.cl.metrics())
    ths = [threading.Thread(target=go, args=(s,)) for s in range(n)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    d = C.spec_delta(s0, C.spec_snapshot(ctx.cl.metrics()))
    ok = [x for x in res if not x['problem']]
    t0 = min((x['first'] for x in ok), default=None)
    t1 = max((x['end'] for x in ok), default=None)
    return {'i': b, 'streams': res, 'problems': [x['problem'] for x in res if x['problem']],
            # streams' tokens over the time from the first stream's first token to the last stream's end
            'tps': sum(x['tokens'] for x in ok) / (t1 - t0) if ok and t1 > t0 else None,
            'tokens': sum(x['tokens'] for x in ok), 'decode_s': (t1 - t0) if ok else None,
            'per_stream': statistics.median(x['tps'] for x in ok if x['tps']) if ok else None,
            'ttft': [x['ttft'] for x in ok], 'tok_per_step': d['tokens_per_step']}


def concurrency(ctx, ns, batches):
    out = {}
    for n in ns:
        bs = [conc_batch(ctx, n, b) for b in range(batches)]
        good = [b for b in bs if not b['problems']]
        ttft = sorted(t for b in good for t in b['ttft'])
        out[str(n)] = {'batches': bs, 'failed': sum(len(b['problems']) for b in bs),
                       'tps': sum(b['tokens'] for b in good) / sum(b['decode_s'] for b in good) if good else None,
                       'per_stream': statistics.median(b['per_stream'] for b in good) if good else None,
                       'ttft_p50': ttft[len(ttft) // 2] if ttft else None,
                       'ttft_p90': ttft[min(len(ttft) - 1, int(0.9 * len(ttft)))] if ttft else None,
                       'tok_per_step': statistics.median(b['tok_per_step'] for b in good if b['tok_per_step'])
                       if any(b['tok_per_step'] for b in good) else None}
        ctx.log(f"  {n} streams: {out[str(n)]['tps']:.1f} tok/s" if out[str(n)]['tps'] else f'  {n} streams: failed')
    return out


# ---------------------------------------------------------------- prefill


def prefill(ctx, sizes, reps, max_len):
    """Cold prefill: a salted code document of each size, then a short reply. Then the same prompt again (cached)."""
    out = {}
    ask = 'Summarize in plain prose what this code does, in about 200 words.'
    for tokens in sizes:
        key = f'{tokens // 1024}K'
        if max_len and tokens + 1000 > max_len:
            out[key] = {'skipped': f'over max_model_len {max_len}'}
            continue
        cold = []
        for _ in range(reps):
            msgs = docs.filler_prompt(ctx.cl, tokens, random.randrange(1 << 30), ask)
            r = ctx.chat('prefill', key, msgs, 256)
            cold.append({'prompt_tokens': r['prompt_tokens'], 'ttft': r['ttft'], 'cached_tokens': r['cached_tokens'],
                         'tok_s': r['prompt_tokens'] / r['ttft'] if r['ttft'] > 0 else None, 'decode_tps': rate(r)})
        warm = ctx.chat('prefill', key + '-cached', msgs, 16)
        med = lambda k: statistics.median(c[k] for c in cold if c[k] is not None)
        out[key] = {'prompt_tokens': cold[-1]['prompt_tokens'], 'ttft_s': med('ttft'), 'tok_s': med('tok_s'),
                    'decode_tps': med('decode_tps'), 'ttft_cached_s': warm['ttft'],
                    'cached_tokens': warm['cached_tokens'], 'cold': cold,
                    'not_cold': any(c['cached_tokens'] for c in cold)}
        ctx.log(f"  prefill {key}: {out[key]['tok_s']:.0f} tok/s, TTFT {out[key]['ttft_s']:.2f} s")
    return out


# ---------------------------------------------------------------- full only


def agentic(ctx):
    out = {}
    for name, fn in (('toolcall', lambda: toolcall.run(ctx.cl.url, ctx.cl.model, 40, 4)),
                     ('agent_tools', lambda: agent_tools.run(ctx.cl.url, ctx.cl.model)),
                     ('longreply', lambda: longreply.run(ctx.cl.url, ctx.cl.model, 2))):
        try:
            out[name] = fn()
        except Exception as e:   # a server that refuses a field the check needs: report it, keep going
            out[name] = {'error': repr(e)[:300]}
        ctx.log(f'  {name}: done')
    return out


def quality_stage(ctx):
    out = {}
    for name, fn in (('gsm8k', lambda: quality.gsm8k(ctx.cl, ctx.a.data_dir, 250, ctx.a.effort)),
                     ('humaneval', lambda: quality.humaneval(ctx.cl, ctx.a.data_dir, ctx.a.effort))):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {'error': repr(e)[:300]}
        ctx.log(f'  {name}: done')
    return out
