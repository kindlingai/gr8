#!/usr/bin/env python3
"""S5 (#56) step 1: capture GLM-5.3 traces through the gr8 front end (or any OpenAI/vLLM server with the same
fields). Each prompt is chat-templated by the server (/tokenize with messages, and tools when the prompt has them),
then generated greedily on /v1/completions with the top-k log-probabilities of every generated token. Stdlib only.

    python3 tools/ngram/capture.py --url http://127.0.0.1:8100 --prompts tools/ngram/data/prompts.json \
        --out traces.jsonl [--max-tokens 1024] [--top 8] [--repeat 1]

The prompts file is a list of {"workload": ..., "messages": [...], "tools": [...] (optional)}. Workloads in the bundled
set: code (write code), edit (change given code), tool (tool-call turns), json, prose. The output is one trace per line
in ngram.py's format, with the server's model and the request settings. Traces from real traffic must stay local and
scrubbed; this tool only sends the prompts it is given.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bench', 'gate'))
from client import Client  # noqa: E402


def post(c, path, body):
    with c._post(path, body) as r:
        return json.load(r)


def chat_ids(c, item, effort):
    body = {'model': c.model, 'messages': item['messages'], 'add_generation_prompt': True}
    if item.get('tools'):
        body['tools'] = item['tools']
    if effort:
        body['chat_template_kwargs'] = {'reasoning_effort': effort}
    return post(c, '/tokenize', body)['tokens']


def top_of(entry):
    """{'token_id:123': lp, ...} -> [[123, lp], ...] by lp descending."""
    out = []
    for k, v in entry.items():
        out.append([int(str(k).split(':')[-1]), float(v)])
    out.sort(key=lambda x: -x[1])
    return out


def capture_one(c, item, max_tokens, top, effort):
    prompt = chat_ids(c, item, effort)
    r = post(c, '/v1/completions', {'model': c.model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0,
                                    'logprobs': top, 'return_token_ids': True, 'return_tokens_as_token_ids': True})
    ch = r['choices'][0]
    gen = ch['token_ids']
    tl = ch['logprobs']['top_logprobs']
    if len(tl) != len(gen):
        raise RuntimeError(f'{len(gen)} tokens but {len(tl)} top_logprobs entries')
    tops = []
    for tok, e in zip(gen, tl):
        t = top_of(e)
        # the sampled token comes first in the server's dict; keep only the top-k by value, with the sampled token
        # kept only if it is among them (greedy: it is the first)
        tops.append(t[:top])
    return {'workload': item.get('workload', 'other'), 'prompt': prompt, 'gen': gen, 'top': tops,
            'finish_reason': ch.get('finish_reason')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--url', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--top', type=int, default=8, help='top-k log-probabilities per position (the server allows 20)')
    ap.add_argument('--effort', default='', help='chat_template_kwargs reasoning_effort (default: the template\'s)')
    ap.add_argument('--model', default='glm-5.3')
    a = ap.parse_args(argv)
    c = Client(a.url, model=a.model)
    with open(a.prompts) as f:
        items = json.load(f)
    n = 0
    with open(a.out, 'a') as out:
        for i, item in enumerate(items):
            t0 = time.time()
            try:
                tr = capture_one(c, item, a.max_tokens, a.top, a.effort)
            except (OSError, RuntimeError, KeyError, ValueError, urllib.error.URLError) as e:
                print(f'[{i}] {item.get("workload")}: failed: {e}', file=sys.stderr)
                continue
            tr['settings'] = {'max_tokens': a.max_tokens, 'top': a.top, 'effort': a.effort or 'default',
                              'temperature': 0}
            out.write(json.dumps(tr, separators=(',', ':')) + '\n')
            out.flush()
            n += 1
            print(f'[{i}] {tr["workload"]}: {len(tr["prompt"])} prompt, {len(tr["gen"])} generated, '
                  f'{time.time() - t0:.1f} s')
    print(f'{n} of {len(items)} traces -> {a.out}')
    return 0 if n else 1


if __name__ == '__main__':
    sys.exit(main())
