#!/usr/bin/env python3
"""Prose bench: cold prefill tok/s (TTFT proxy) at 8K/32K + prose decode tok/s.
Usage: pbench.py [--url http://127.0.0.1:8100] [--tag NAME] [--reps 3]
Appends one JSON line per run to results.jsonl next to this file."""
import argparse, json, random, time, uuid, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8100')
ap.add_argument('--tag', default='untagged')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--sizes', default='8192,32768')
ap.add_argument('--decode-tokens', type=int, default=1500)
ap.add_argument('--temp', type=float, default=0.7)
ap.add_argument('--min-p', type=float, default=0.0)
ap.add_argument('--effort', default='low', help='low|high|max, or nothink: no thinking controls (GLM 5.3 cannot turn '
                                                'thinking off, so the template default, Max)')
ap.add_argument('--no-prefill', action='store_true')
a = ap.parse_args()

WORDS = ('river stone morning harbor lantern quiet market winter orchard letter village '
         'engine garden thunder copper meadow window valley story candle bridge north '
         'silver forest ladder kitchen shadow evening mountain paper island').split()
PROSE = [
    'Write a vivid 600-word short story about a lighthouse keeper who finds a letter in a bottle. Plain prose, no headings.',
    'Write a 600-word essay on why small towns change slowly, in flowing paragraphs.',
    'Describe, in about 600 words of narrative prose, a thunderstorm rolling over a wheat farm at night.',
]

def req(payload):
    r = urllib.request.Request(a.url + '/v1/chat/completions', json.dumps(payload).encode(),
                               {'Content-Type': 'application/json'})
    return urllib.request.urlopen(r, timeout=1800)

TEXT = []; CONTENT = []; S = {}
def ntok(text):
    r = urllib.request.Request(a.url + '/tokenize', json.dumps({'model': 'glm-5.3', 'prompt': text}).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r))['count']
def stream(messages, max_tokens):
    TEXT.clear()
    p = {'model': 'glm-5.3', 'messages': messages, 'max_tokens': max_tokens, 'stream': True,
         'temperature': a.temp, 'min_p': a.min_p, 'stream_options': {'include_usage': True},
         'chat_template_kwargs': ({} if a.effort == 'nothink' else {'reasoning_effort': a.effort})}
    t0 = time.time(); tfirst = None; usage = None; S['tc'] = None; CONTENT.clear()
    with req(p) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            d = json.loads(line[6:])
            if d.get('usage'):
                usage = d['usage']
            ch = d.get('choices') or []
            dl = (ch[0].get('delta') or {}) if ch else {}
            txt = (dl.get('content') or '') + (dl.get('reasoning_content') or dl.get('reasoning') or '')
            if txt:
                if tfirst is None: tfirst = time.time()
                TEXT.append(txt)
            if dl.get('content'):
                if S['tc'] is None: S['tc'] = time.time()
                CONTENT.append(dl['content'])
    return t0, tfirst, time.time(), usage

def filler(n_tokens, rng):
    # ~1.075 tokens/word (measured) for this vocabulary; random order defeats prefix cache and MTP copy
    words = [rng.choice(WORDS) for _ in range(int(n_tokens / 1.075))]
    return ' '.join(words)

def metric():
    t = urllib.request.urlopen(a.url + '/metrics').read().decode()
    g = lambda k: sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(k))
    pos = [g('vllm:spec_decode_num_accepted_tokens_per_pos_total{' ) if False else sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total') and f'position="{i}"' in l) for i in range(8)]
    return g('vllm:spec_decode_num_draft_tokens_total'), g('vllm:spec_decode_num_accepted_tokens_total'), g('vllm:spec_decode_num_drafts_total'), pos
rng = random.Random()
out = {'tag': a.tag, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'prefill': {}, 'decode': []}
stream([{'role': 'user', 'content': 'hi'}], 8)  # warm
for size in ([] if a.no_prefill else map(int, a.sizes.split(','))):
    rows = []
    for _ in range(a.reps):
        msg = str(uuid.uuid4()) + ' ' + filler(size, rng) + '\nSummarize in one word.'
        t0, tf, t1, u = stream([{'role': 'user', 'content': msg}], 1)
        pt = u['prompt_tokens']; rows.append({'prompt_tokens': pt, 'ttft': round(t1 - t0, 3),
                                              'tok_s': round(pt / (t1 - t0), 1)})
    out['prefill'][size] = rows
m0 = metric()
for i in range(a.reps):
    t0, tf, t1, u = stream([{'role': 'user', 'content': PROSE[i % len(PROSE)] + ' ' + str(uuid.uuid4())[:8]}],
                           a.decode_tokens)
    n = u['completion_tokens']; c = ''.join(CONTENT); nc = ntok(c) if c else 0
    row = {'tokens': n, 'all_tok_s': round((n - 1) / (t1 - tf), 2), 'content_tokens': nc,
           'think_s': round((S['tc'] or t1) - tf, 2), 'text_head': c[:200]}
    row['tok_s'] = round((nc - 1) / (t1 - S['tc']), 2) if nc > 50 else None
    out['decode'].append(row)
m1 = metric(); out['accept'] = round((m1[1]-m0[1]) / max(1, m1[0]-m0[0]), 3); out['temp'] = a.temp; rounds = max(1, m1[2]-m0[2]); out['pos_survival'] = [round((b-c)/rounds, 3) for b, c in zip(m1[3], m0[3]) if b-c > 0]; out['tok_per_round'] = round(1 + (m1[1]-m0[1])/rounds, 2); out['min_p'] = a.min_p
def med(xs): xs = sorted(xs); return xs[len(xs) // 2]
out['summary'] = {f'prefill_{s}': med([r['tok_s'] for r in v]) for s, v in out['prefill'].items()}
out['summary']['accept'] = out['accept']; out['summary']['pos'] = out['pos_survival']; out['summary']['tok_per_round'] = out['tok_per_round']
ds = [r['tok_s'] for r in out['decode'] if r['tok_s']]
out['summary']['prose_decode'] = med(ds) if ds else None
out['summary']['all_decode'] = med([r['all_tok_s'] for r in out['decode']])
out['effort'] = a.effort
with open(Path(__file__).with_name('results.jsonl'), 'a') as f:
    f.write(json.dumps(out) + '\n')
print(json.dumps(out['summary']))
