"""Expert bytes per decode step from a G6_ROUTE_LOG file, under exlr8 v2 width manifests and the half rule (#65).

python3 bench/route_bytes.py LOG.jsonl --manifest home=DIR/exlr8-manifest-home.json --manifest down1=... \
        [--cold-k2 0.5] [--tp 3,4,5,6,7] [--side full|block]

The log has one line per engine step (engine/runtime/rank.cuh route_log_line): the step's rows M and, per MoE layer,
the distinct routed experts. For each manifest and TP size the script sums, per step and rank, the trellis bytes of
the halves that rank owns ((16e + h + L) mod N) of each distinct expert, plus the shared expert (always read), plus
the hidden-side vectors (full: 36 KB per block row, as tr3.v1 + tr3.v3 today; block: 288 B). It prints the mean bytes
per step per rank (the mean over ranks, which sets the bandwidth cost when ranks are balanced) and the busiest rank's
mean (which sets the step time when they are not), by row count M, and the time at 220 GB/s.
--cold-k2 F adds a manifest "coldK2" built from the first manifest: the fraction F of each layer's experts least
routed in this log go to K2 (gate, up and down); the rest keep the first manifest's widths.
"""
import argparse
import collections
import json

import numpy as np

KSH, NH = 384, 16
HALF = {k: KSH * 8 * 32 * k for k in (2, 3, 4)}        # bytes of one half of one projection at width k
SIDE = {'full': (2 * 6144 + 6144) * 2 + 3 * 128 * 2, 'block': 3 * 48 * 2 + 3 * 128 * 2}


def load_manifest(path):
    m = json.load(open(path))
    out = {}
    for l, e in m['layers'].items():
        g, u, d = np.asarray(e['gate']), np.asarray(e['up']), np.asarray(e['down'])
        sh = e.get('shared', {'gate': [4] * 16, 'up': [4] * 16, 'down': 4})
        out[int(l)] = (g, u, d, np.asarray(sh['gate']), np.asarray(sh['up']), int(sh['down']))
    return out


def rank_bytes(widths, l, n, side):
    """[257][n] bytes per (expert, rank) of layer l at TP n (row 256 = shared)."""
    g, u, d, sg, su, sd = widths
    out = np.zeros((257, n))
    for e in range(257):
        gw, uw, dw = (sg, su, sd) if e == 256 else (g[e], u[e], d[e])
        for h in range(NH):
            r = (16 * e + h + l) % n
            out[e, r] += HALF[int(gw[h])] + HALF[int(uw[h])] + HALF[int(dw)] + SIDE[side]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--manifest', action='append', required=True, help='NAME=PATH')
    ap.add_argument('--cold-k2', type=float, default=0.0)
    ap.add_argument('--tp', default='4')
    ap.add_argument('--side', default='full', choices=('full', 'block'))
    a = ap.parse_args()
    steps = [json.loads(x) for x in open(a.log) if x.strip()]
    mans = {}
    for spec in a.manifest:
        name, path = spec.split('=', 1)
        mans[name] = load_manifest(path)
    if a.cold_k2 > 0:
        use = collections.defaultdict(lambda: np.zeros(256))
        for s in steps:
            for l, ex in s['L'].items():
                use[int(l)][ex] += 1
        base = mans[next(iter(mans))]
        cold = {}
        for l, (g, u, d, sg, su, sd) in base.items():
            idx = np.argsort(use[l], kind='stable')[:int(round(a.cold_k2 * 256))]
            g2, u2, d2 = g.copy(), u.copy(), d.copy()
            g2[idx] = 2; u2[idx] = 2; d2[idx] = 2
            cold[l] = (g2, u2, d2, sg, su, sd)
        mans['coldK2'] = cold
    tps = [int(x) for x in a.tp.split(',')]
    layers = sorted({int(l) for s in steps for l in s['L']})
    for n in tps:
        print(f'TP{n}, side vectors {a.side}, {len(steps)} steps, {len(layers)} MoE layers in the log')
        for name, w in mans.items():
            rb = {l: rank_bytes(w[l], l, n, a.side) for l in layers if l in w}
            by_m = collections.defaultdict(list)
            for s in steps:
                tot = np.zeros(n)
                for l, ex in s['L'].items():
                    l = int(l)
                    if l not in rb:
                        continue
                    tot += rb[l][ex].sum(0) + rb[l][256]
                by_m[s['M']].append(tot)
            for M in sorted(by_m):
                t = np.array(by_m[M])
                mean, busy = t.mean(), t.max(1).mean()
                print(f'  {name:8s} M={M:2d} ({len(t):5d} steps): {mean / 1e6:8.1f} MB/rank/step, busiest rank {busy / 1e6:8.1f} MB '
                      f'(+{100 * (busy / mean - 1):.1f}%), {busy / 220e6:6.2f} ms at 220 GB/s')


if __name__ == '__main__':
    main()
