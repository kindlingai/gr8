"""Teacher-forced scoring of token rows on a running gr8 (the student side of a KL against a teacher).

python3 bench/score_rows.py ROWS OUT.jsonl [--url http://127.0.0.1:8100] [--k 20] [--rows 977:1041] [--timeout 300]
        [--label L --note TEXT] [--raw]

ROWS: a .npy integer array [rows][len], a .json list of token-id lists, or a raw little-endian int32 file with
--len L. Each row is sent as a token-id prompt with max_tokens=1, temperature 0 and prompt_logprobs=k (the gate
harness's KL path, bench/gate/kl.py). OUT gets one JSON line per row: {"row", "n", "lp": [logprob of the actual token
at positions 1..n-1], "top": [[[id, logprob] x k] per position 1..n-1]}. The actual token's logprob is exact even
outside the top k. --raw writes {"row", "prompt_logprobs": the server's list unchanged} instead (entry 0 null, entry j
keyed by token id: the actual token plus ranks 1..k). OUT is written as OUT.tmp and renamed when every row is in. A
request that takes longer than --timeout fails the run. With --label, OUT.json next to OUT.jsonl
records the label, the note (commit, flags, packs), the rows file and its sha256, the selection and the server URL.
Stdlib only (the .npy reader handles little-endian integer arrays).
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate'))
from client import Client  # noqa: E402


def load_npy(path):
    """2-D little-endian int32 / int64 .npy (C order) -> list of lists, without numpy."""
    import array
    import ast
    with open(path, 'rb') as f:
        if f.read(6) != b'\x93NUMPY':
            sys.exit(f'{path}: not a .npy file')
        major = f.read(2)[0]
        hl = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
        hdr = ast.literal_eval(f.read(hl).decode('latin1'))
        code = {'<i4': 'i', '<i8': 'q'}.get(hdr['descr'])
        if code is None or hdr['fortran_order'] or len(hdr['shape']) != 2:
            sys.exit(f'{path}: want a 2-D C-order <i4 or <i8 array, got {hdr}')
        a = array.array(code)
        a.frombytes(f.read())
    rows, n = hdr['shape']
    return [a[i * n:(i + 1) * n].tolist() for i in range(rows)]


def load_rows(path, length):
    if path.endswith('.json'):
        return json.load(open(path))
    if path.endswith('.npy'):
        return load_npy(path)
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
    ap.add_argument('--label', default='')
    ap.add_argument('--note', default='')
    ap.add_argument('--raw', action='store_true', help="write the server's prompt_logprobs list unchanged")
    a = ap.parse_args()
    rows = load_rows(a.rows, a.len)
    if a.sel:
        lo, hi = (int(x) if x else None for x in a.sel.split(':'))
        idx = list(range(len(rows)))[lo:hi]
    else:
        idx = list(range(len(rows)))
    c = Client(a.url, timeout=a.timeout)
    if a.label:
        h = hashlib.sha256()
        with open(a.rows, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''):
                h.update(b)
        side = dict(label=a.label, note=a.note, rows_file=os.path.abspath(a.rows), rows_sha256=h.hexdigest(),
                    rows=a.sel or 'all', k=a.k, url=a.url, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with open(os.path.splitext(a.out)[0] + '.json', 'w') as f:
            json.dump(side, f, indent=1)
    t0 = time.time()
    with open(a.out + '.tmp', 'w') as f:
        for n, i in enumerate(idx):
            ids = [int(x) for x in rows[i]]
            r = c.complete(ids, max_tokens=1, temperature=0, prompt_logprobs=a.k)
            plp = r['choices'][0]['prompt_logprobs']
            if plp is None or len(plp) != len(ids):
                sys.exit(f'row {i}: {None if plp is None else len(plp)} prompt_logprobs for {len(ids)} tokens')
            if any(str(ids[j]) not in plp[j] for j in range(1, len(ids))):
                sys.exit(f'row {i}: a position lacks its actual token')
            if a.raw:
                f.write(json.dumps({'row': i, 'prompt_logprobs': plp}) + '\n')
                f.flush()
                print(f'row {i} ({n + 1}/{len(idx)}): {len(ids)} tokens, {time.time() - t0:.0f} s', flush=True)
                continue
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
    os.replace(a.out + '.tmp', a.out)


if __name__ == '__main__':
    main()
