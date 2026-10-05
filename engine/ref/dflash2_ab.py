"""Two sets of DFlash2 context features for the same token sequence, side by side (issue #12, the acceptance gap).

python3 engine/ref/dflash2_ab.py --src DRAFTER --glm GLM_DIR --tokens T --a A [--b B] --start N [--quant bf16] [--out F]

A and B are the six target taps (residual after layers 5, 19, 33, 47, 61, 75) at every position of the sequence:
  .npy / .safetensors   [N, 6, 6144] (or [N, 36864]) float32, float16 or bfloat16 (bfloat16 as uint16 in .npy)
  DUMP[:i]              a G6_DFL_DUMP, request i (default 0); a request ends at its token record
T is a .npy / .json list of token ids, or DUMP[:i] (that request's token record). --start: the first generated
position (the prompt length); the comparison covers positions from there on.

Printed:
  features     per tap layer, over the positions both sets have: cosine (mean, 5th percentile) and relative error
               |A - B| / |A| (median, 95th percentile), for the prompt and the generated part apart
  acceptance   every generated position b as a round start (bonus token = T[b], context = positions < b): the
               reference drafter's walk and argmax drafts, accepted against T[b + 1 ..] (leading matches). Tokens per
               round = 1 + mean accepted, and the share of rounds whose first n drafts were all accepted, for A and
               for B. Rounds whose drafts differ between A and B are counted.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash2_ref as R  # noqa: E402
from dflash2_replay import B, HID, NT, accept, ref_drafts  # noqa: E402


def dump_requests(path):
    """G6_DFL_DUMP -> [(feats {pos: float32 row}, tokens)] per request (the engine stores residual / 64 as fp16)."""
    raw = open(path, 'rb').read()
    off, out, feats = 0, [], {}
    while off < len(raw):
        h = np.frombuffer(raw, np.int32, 12, off)
        off += 48
        if h[0] == 2:
            out.append((feats, np.frombuffer(raw, np.int32, h[1], off).copy()))
            off += 4 * h[1]
            feats = {}
            continue
        pos0, rows = int(h[1]), int(h[2])
        f = np.frombuffer(raw, np.float16, rows * NT * HID, off).reshape(rows, NT * HID)
        off += f.nbytes
        for r in range(rows):
            feats[pos0 + r] = f[r].astype(np.float32) * 64
    return out


def split_ref(spec):
    path, _, i = spec.partition(':')
    return path, int(i) if i else 0


def load_feats(spec):
    """-> {pos: float32 [6 x 6144]}"""
    path, i = split_ref(spec)
    if path.endswith('.npy') or path.endswith('.safetensors'):
        if path.endswith('.npy'):
            x = np.load(path)
            if x.dtype == np.uint16:
                x = R.bf(x)
        else:
            from safetensors.numpy import load_file
            t = load_file(path)
            x = t[sorted(t)[0]] if len(t) == 1 else t['feat']
            if x.dtype == np.uint16:
                x = R.bf(x)
        x = np.asarray(x, np.float32).reshape(len(x), NT * HID)
        return {p: x[p] for p in range(len(x))}
    return dump_requests(path)[i][0]


def load_tokens(spec):
    path, i = split_ref(spec)
    if path.endswith('.npy'):
        return np.load(path).astype(np.int64).tolist()
    if path.endswith('.json'):
        return [int(t) for t in json.load(open(path))]
    return dump_requests(path)[i][1].tolist()


def compare(fa, fb, start):
    common = sorted(set(fa) & set(fb))
    print(f'features: {len(common)} positions in both ({len(fa)} in A, {len(fb)} in B)')
    for part, sel in (('prompt', [p for p in common if p < start]), ('generated', [p for p in common if p >= start])):
        if not sel:
            continue
        A = np.stack([fa[p] for p in sel]).reshape(len(sel), NT, HID)
        Bm = np.stack([fb[p] for p in sel]).reshape(len(sel), NT, HID)
        cos = (A * Bm).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(Bm, axis=-1) + 1e-30)
        rel = np.linalg.norm(A - Bm, axis=-1) / (np.linalg.norm(A, axis=-1) + 1e-30)
        for t in range(NT):
            print(f'  {part:9s} tap {t}: cosine mean {cos[:, t].mean():.5f} p5 {np.percentile(cos[:, t], 5):.5f}; '
                  f'relative error median {np.median(rel[:, t]):.4f} p95 {np.percentile(rel[:, t], 95):.4f} ({len(sel)} positions)')


def acceptance(D, glm, feats, tokens, start, label):
    """every generated position b with 7 known tokens after it as a round start -> per-round accepted counts"""
    starts = [b for b in range(start, len(tokens) - B + 1) if all(p in feats for p in range(max(0, b - D.win + 1), b))]
    pos = np.array(sorted(p for p in feats if p < len(tokens)))
    F = np.stack([feats[p] for p in pos])
    steps = [dict(q0=b, tok0=int(tokens[b])) for b in starts]
    walks, argmaxes = ref_drafts(D, glm, pos, F, steps)
    res = dict(walk=[], argmax=[])
    for i, b in enumerate(starts):
        actual = tokens[b + 1:b + B]
        res['walk'].append(accept(walks[i], actual))
        res['argmax'].append(accept(argmaxes[i], actual))
    for k, v in res.items():
        v = np.array(v)
        surv = [round(float((v >= n).mean()), 3) for n in range(1, B)]
        print(f'  {label} {k:6s}: {len(v)} rounds, tokens per round {1 + v.mean():.3f}, kept {surv}')
    return res, walks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--glm', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--a', required=True)
    ap.add_argument('--b')
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--quant', default='bf16')
    ap.add_argument('--out')
    a = ap.parse_args()
    tokens = load_tokens(a.tokens)
    fa = load_feats(a.a)
    fb = load_feats(a.b) if a.b else None
    print(f'{len(tokens)} tokens, generated from position {a.start}')
    if fb is not None:
        compare(fa, fb, a.start)
    D = R.Drafter(a.src, None if a.quant == 'bf16' else a.quant)
    print('acceptance (teacher-forced on the token sequence):')
    ra, wa = acceptance(D, a.glm, fa, tokens, a.start, 'A')
    out = dict(a=ra)
    if fb is not None:
        rb, wb = acceptance(D, a.glm, fb, tokens, a.start, 'B')
        out['b'] = rb
        diff = sum(x != y for x, y in zip(wa, wb))
        print(f'  walk drafts differ between A and B in {diff} of {len(wa)} rounds')
    if a.out:
        json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
