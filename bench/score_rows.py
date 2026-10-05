"""Teacher-forced scoring of token rows on a running g6 (the student side of a KL against a teacher, #65 / SPEC A3).

python3 bench/score_rows.py ROWS OUT.jsonl [--url http://127.0.0.1:8100] [--k 20] [--rows 0:64] [--timeout 300]

ROWS: a .npy integer array [rows][len], a .json list of token-id lists, or a raw little-endian int32 file with
--len L. Each row is sent as a token-id prompt with max_tokens=1, temperature 0 and prompt_logprobs=k (the gate
harness's KL path, bench/gate/kl.py). OUT gets one JSON line per row: {"row", "n", "lp": [logprob of the actual token
at positions 1..n-1], "top": [[[id, logprob] x k] per position 1..n-1]}. The actual token's logprob is exact even
outside the top k. A request that takes longer than --timeout fails the run.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate'))
from client import Client  # noqa: E402


def load_rows(path, length):
    if path.endswith('.json'):
        return json.load(open(path))
    if path.endswith('.npy'):
        import numpy as np
        return np.load(path).astype('int64').tolist()
    import array
    a = array.array('i')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    if not length:
        sys.exit('--len is required for a raw int32 file')
    return [a[i:i + length].tolist() for i in range(0, len(a), length)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rows')
    ap.add_argument('out')
    ap.add_argument('--url', default='http://127.0.0.1:8100')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--rows', dest='sel', default=None, help='a:b slice of the rows')
    ap.add_argument('--len', type=int, default=0)
    ap.add_argument('--timeout', type=float, default=300)
    a = ap.parse_args()
    rows = load_rows(a.rows, a.len)
    if a.sel:
        lo, hi = (int(x) if x else None for x in a.sel.split(':'))
        idx = list(range(len(rows)))[lo:hi]
    else:
        idx = list(range(len(rows)))
    c = Client(a.url, timeout=a.timeout)
    t0 = time.time()
    with open(a.out, 'w') as f:
        for n, i in enumerate(idx):
            ids = [int(x) for x in rows[i]]
            r = c.complete(ids, max_tokens=1, temperature=0, prompt_logprobs=a.k)
            plp = r['choices'][0]['prompt_logprobs']
            if plp is None or len(plp) != len(ids):
                sys.exit(f'row {i}: {None if plp is None else len(plp)} prompt_logprobs for {len(ids)} tokens')
            lp, top = [], []
            for j in range(1, len(ids)):
                e = plp[j]
                lp.append(e[str(ids[j])]['logprob'])
                t = sorted(((int(t), v['logprob']) for t, v in e.items() if v.get('rank') is not None and v['rank'] <= a.k),
                           key=lambda x: -x[1])
                top.append([[t_, round(v, 6)] for t_, v in t])
            f.write(json.dumps({'row': i, 'n': len(ids), 'lp': lp, 'top': top}) + '\n')
            f.flush()
            print(f'row {i} ({n + 1}/{len(idx)}): {len(ids)} tokens, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
