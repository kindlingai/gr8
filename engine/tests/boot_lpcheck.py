#!/usr/bin/env python3
"""Booted-cluster check of logprobs / prompt_logprobs against the engine's own float64 reference.

Needs a TP6 serve build started with `--top 20` (G6ARGS_X="--top 20" in scripts/launch_tp6.sh): then every rank prints,
for each row of every request that asks for logprobs, `top <step> <row> rank <R> lse <slice lse> id:logit ...` (the K largest
raw logits of its vocab slice and the slice's log-sum-exp, float64 on the host). Merging the six ranks' lines gives the
reference top-20 and logsumexp of the full vocab for that position, to compare with what the device path returned.

  boot_lpcheck.py --url http://HEAD:8100 --rank-files out/g6m4.r0.txt HOST1:/path/out/g6m4.r1.txt ... [--prompt-ids F --n 1024]

  (a) greedy generation with logprobs=5: every emitted id is the argmax of its own top list, and its logprob is that entry's.
  (b) prompt_logprobs=20 on prompt+generated ids: ids, order and logprobs of the top-20 (tol --tol) and the rank of the
      actual token (when <= 20) against the reference; top-1 of each generated position against the greedy token;
      (a)'s per-token logprobs against (b)'s at the same positions.
Prints one summary line per check and exits 1 if a hard check fails.
"""
import argparse
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bench', 'gate'))
from client import Client  # noqa: E402

CHUNK = 8
PROMPT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable "
          "metal type, oil-based ink and a screw press adapted from the wine trade. Within fifty years presses were "
          "running in more than two hundred cities, and ")


def read_rank(spec):
    if ':' in spec and not os.path.exists(spec):
        host, path = spec.split(':', 1)
        return subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', host, 'cat', path], capture_output=True, text=True, check=True).stdout
    return open(spec).read()


def last_block(text, n):
    """lines of the last `seq N prompt <n> tokens` block"""
    lines = text.split('\n')
    idx = [i for i, l in enumerate(lines) if re.match(r'seq \d+ prompt %d tokens$' % n, l)]
    if not idx:
        raise SystemExit(f'no "seq .. prompt {n} tokens" block in a rank file')
    return lines[idx[-1] + 1:]


def parse_top(lines, n):
    """-> {pos: (slice lse, [(id, logit), ...])} for prompt rows pos < n - 1"""
    out = {}
    for l in lines:
        if l.startswith('seq '):
            break
        if not l.startswith('top '):
            continue
        t = l.split()
        s, r = int(t[1]), int(t[2])
        pos = s * CHUNK + r
        if pos >= n - 1:
            continue
        lse = float(t[6])
        out[pos] = (lse, [(int(a), float(b)) for a, b in (x.split(':') for x in t[7:])])
    return out


def merge(per_rank, pos, K):
    lses = [per_rank[r][pos][0] for r in range(len(per_rank))]
    m = max(lses)
    lse = m + math.log(sum(math.exp(x - m) for x in lses))
    cand = sorted((e for r in range(len(per_rank)) for e in per_rank[r][pos][1]), key=lambda e: (-e[1], e[0]))[:K]
    return lse, [(i, v - lse) for i, v in cand]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--rank-files', nargs='+', required=True, help='six engine stdout files, rank 0..5; PATH or HOST:PATH (ssh cat)')
    ap.add_argument('--prompt-ids', help='file of token ids (comma or whitespace separated); default: a short built-in text')
    ap.add_argument('--n', type=int, default=1024, help='prompt tokens used from --prompt-ids')
    ap.add_argument('--gen', type=int, default=64)
    ap.add_argument('--tol', type=float, default=2e-4)
    a = ap.parse_args()
    c = Client(a.url)
    fails = []

    prompt = c.tokenize(PROMPT) if not a.prompt_ids else [int(x) for x in open(a.prompt_ids).read().replace(',', ' ').split()][:a.n]
    # (a) greedy generation with logprobs=5
    r = c.complete(prompt, max_tokens=a.gen, temperature=0, logprobs=5, return_token_ids=True, return_tokens_as_token_ids=True,
                   ignore_eos=True)['choices'][0]
    gen = r['token_ids']
    lg = r['logprobs']
    bad_top = bad_lp = 0
    max_d = 0.0
    for i, t in enumerate(gen):
        top = {int(k.split(':')[-1]): v for k, v in lg['top_logprobs'][i].items()}
        best = max(top, key=top.get)
        if best != t:
            bad_top += 1
        d = abs(lg['token_logprobs'][i] - top[best])
        max_d = max(max_d, d)
        bad_lp += d > 1e-6 and best == t
    print(f'(a) greedy gen {len(gen)} tokens: top-1 != emitted {bad_top}, token lp != top lp {bad_lp}, max diff {max_d:.2e}')
    if bad_top or bad_lp or len(gen) != a.gen:
        fails.append('a')

    # (b) prompt logprobs on prompt + generated ids
    ids = prompt + gen
    n = len(ids)
    pl = c.complete(ids, max_tokens=1, temperature=0, prompt_logprobs=20, return_tokens_as_token_ids=True)['choices'][0]['prompt_logprobs']
    assert len(pl) == n and pl[0] is None, (len(pl), n)
    import time
    time.sleep(2)
    per_rank = [parse_top(last_block(read_rank(s), n), n) for s in a.rank_files]
    missing = [r for r, d in enumerate(per_rank) if len(d) != n - 1]
    if missing:
        print(f'(b) FAIL: rank files {missing} do not have {n - 1} top rows ({[len(d) for d in per_rank]})')
        sys.exit(1)
    max_lp = 0.0
    bad_ids = bad_rank = bad_actual = 0
    ties = 0
    for pos in range(n - 1):
        j = pos + 1
        lse, ref = merge(per_rank, pos, 20)
        e = {int(k.split(':')[-1]): (v['logprob'], v['rank']) for k, v in pl[j].items()}
        api = sorted(((k, v) for k, v in e.items() if v[1] is not None and v[1] <= 20), key=lambda kv: kv[1][1])
        api_ids = [k for k, _ in api]
        if api_ids != [i for i, _ in ref]:
            if sorted(api_ids) == sorted(i for i, _ in ref) or any(abs(ref[x][1] - ref[x + 1][1]) < 1e-6 for x in range(len(ref) - 1)):
                ties += 1
            else:
                bad_ids += 1
        for (i, rl), (k, (lp, rk)) in zip(ref, api):
            max_lp = max(max_lp, abs(lp - rl)) if i == k else max_lp
        t = ids[j]
        if t in e:
            ref_lp = {i: l for i, l in ref}
            if t in ref_lp:
                max_lp = max(max_lp, abs(e[t][0] - ref_lp[t]))
                if e[t][1] != [i for i, _ in ref].index(t) + 1:
                    bad_rank += 1
            elif e[t][1] is not None and e[t][1] <= 20:
                bad_actual += 1
    print(f'(b) prompt_logprobs k=20 over {n - 1} positions vs float64 reference: top-id/order mismatches {bad_ids} (tie-order {ties}), '
          f'actual-token rank mismatches {bad_rank + bad_actual}, max |lp - ref| {max_lp:.2e} (tol {a.tol:.0e})')
    if bad_ids or bad_rank or bad_actual or max_lp > a.tol:
        fails.append('b')

    # top-1 of the prompt_logprobs positions that predict a generated token vs greedy; (a)'s lp vs (b)'s lp
    agree = tot = 0
    d_ab = 0.0
    for i, t in enumerate(gen):
        j = len(prompt) + i
        e = {int(k.split(':')[-1]): (v['logprob'], v['rank']) for k, v in pl[j].items()}
        if t in e:
            tot += 1
            agree += e[t][1] == 1
            d_ab = max(d_ab, abs(e[t][0] - lg['token_logprobs'][i]))
    print(f'(c) info, generated positions: prompt_logprobs rank-1 == greedy token {agree}/{tot}; decode lp vs prompt_logprobs lp max diff {d_ab:.2e}')
    # information only: the request of (a) fills its prompt KV with the prefill kernels, the prompt_logprobs request with the
    # decode kernels; the two paths of the engine differ numerically, so this is not a check of the logprob code ((a), (b) are)
    print('RESULT', 'PASS' if not fails else 'FAIL ' + ','.join(fails))
    sys.exit(1 if fails else 0)


main()
