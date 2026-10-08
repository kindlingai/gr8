"""Item 7 (#65): drive the gate's code / prose / structured workloads against a G6_ROUTE_LOG boot and record which
engine request each one was, so route_bytes.py can split the log by workload.

python3 bench/route_drive.py REPO OUT.json [--runs 6] [--tokens 300]
Requests go out one at a time in this order: one warm-up, then each kind's runs. OUT lists them in order with kind,
prompt_tokens, completion tokens and tok/s; the route log's q values follow the same order.
"""
import json
import sys
import time

repo, out = sys.argv[1], sys.argv[2]
runs = int(sys.argv[sys.argv.index('--runs') + 1]) if '--runs' in sys.argv else 6
tokens = int(sys.argv[sys.argv.index('--tokens') + 1]) if '--tokens' in sys.argv else 300
sys.path.insert(0, repo + '/bench/gate')
import client as C  # noqa: E402
import measure  # noqa: E402

cl = C.Client('http://127.0.0.1:8100', timeout=600)
reqs = []
r = cl.chat([{'role': 'user', 'content': 'hi'}], 8, temperature=0.7, effort='nothink')
reqs.append({'kind': 'warm', 'prompt_tokens': r['prompt_tokens'], 'tokens': r['tokens']})
for kind in ('code', 'prose', 'structured'):
    prompts = measure.WORKLOADS[kind]
    for i in range(runs):
        r = cl.chat([{'role': 'user', 'content': prompts[i % len(prompts)]}], tokens, temperature=0.7, effort='nothink')
        reqs.append({'kind': kind, 'i': i, 'prompt_tokens': r['prompt_tokens'], 'tokens': r['tokens'],
                     'tok_s': round((r['tokens'] - 1) / r['decode_s'], 2), 'finish': r['finish_reason']})
        print(kind, i, reqs[-1], flush=True)
json.dump({'runs': runs, 'tokens': tokens, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'requests': reqs},
          open(out, 'w'), indent=1)
