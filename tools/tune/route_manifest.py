#!/usr/bin/env python3
"""Build an exlr8 v2 width manifest from OUR routing (G6_ROUTE_LOG) + Matt's per-unit error tables, at a byte budget.

python3 route_manifest.py --v2 DIR --base MANIFEST.json --route LOG[,LOG..] [--reqs REQS.json --split hist]
        [--freq route|matt] [--budget base|BYTES] [--alpha 200] --out OUT.json [--report REPORT.json]

Objective: minimise sum over layers l, experts e, units u of f[l][e] * err_u(width), subject to the routed bytes of
layers 3-77 <= budget. err_u is the index's err_gate[e][h] / err_up[e][h] / err_down[e] for the width: held-out error
vs BF16 with one unit swapped in, averaged over the tokens routed to that expert (a per-use error, so it is weighted
by OUR use frequency f here, not by Matt's). Units: gate half, up half (128 channels each, 16 per expert), down
(whole expert, 16 halves of bytes). Widths 2/3/4; bytes per half 384*8*32*k.
f[l][e]: share of decode steps whose routed set in layer l holds e (route-log lines with M <= --max-rows), smoothed
toward Matt's held-out token counts with --alpha pseudo-steps. --freq matt uses Matt's counts only (control arm).
Layer 78 (MTP, K3 only) and every layer's shared expert are copied from --base unchanged.
Greedy: start at K2, take the upgrade (k -> k+1 of one unit) with the best error drop per byte until the budget.
Ranks (--nr 6): half h of expert e in layer l lives on rank (16e + h + l) mod nr (v2 sharding rule; down halves too).
Per layer and rank: resident bytes, and traffic = sum f[e] * bytes of e's halves on the rank (expected bytes read per
decode step). --rank-cap base: an upgrade is refused if it lifts the layer's busiest-rank traffic above the base
manifest's busiest-rank traffic for that layer times (1 + --rank-slack). The report gives both maxima per layer.
Known blind spots: one-unit-at-a-time errors (no compounding), no per-layer sensitivity, distinct-id log (no weights).
"""
import argparse
import heapq
import json
import os
import struct
import sys

import numpy as np

NE, NH = 256, 16
HALF = 384 * 8 * 32          # bytes per 128-channel half per bit of width
LAYERS = range(3, 78)        # MoE layers with K2/K3/K4; 78 copied from base


def read_index(path, names):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name in names:
            t = h[name]
            a, b = t['data_offsets']
            f.seek(base + a)
            dt = {'F32': np.float32, 'I32': np.int32, 'F16': np.float16, 'I64': np.int64}[t['dtype']]
            out[name] = np.frombuffer(f.read(b - a), dt).reshape(t['shape']).astype(np.float64)
    return out


def load_errs(v2dir, l):
    names = ['heldout_tokens'] + [f'k{k}.err_{p}' for k in (2, 3, 4) for p in ('gate', 'up', 'down')]
    d = read_index(os.path.join(v2dir, f'exlr8-layer-{l:03d}.index.safetensors'), names)
    eg = np.stack([d[f'k{k}.err_gate'] for k in (2, 3, 4)])      # [3][256][16]
    eu = np.stack([d[f'k{k}.err_up'] for k in (2, 3, 4)])
    ed = np.stack([d[f'k{k}.err_down'] for k in (2, 3, 4)])      # [3][256]
    return eg, eu, ed, d['heldout_tokens']


def route_counts(paths, qs, max_rows):
    cnt = {l: np.zeros(NE) for l in LAYERS}
    steps = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('M', 1) > max_rows or (qs is not None and r.get('q') not in qs):
                    continue
                steps += 1
                for ls, ids in r['L'].items():
                    l = int(ls)
                    if l in cnt and ids:
                        cnt[l][np.asarray(ids, int)] += 1
    return cnt, steps


def manifest_bytes(m):
    tot = 0
    for l in LAYERS:
        e = m['layers'][str(l)]
        tot += sum(sum(g) + sum(u) for g, u in zip(e['gate'], e['up'])) * HALF + sum(e['down']) * NH * HALF
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--route', default='')
    ap.add_argument('--reqs', default=None, help='JSON list of {"q":..,"split":..}; keep only --split q values')
    ap.add_argument('--split', default='hist')
    ap.add_argument('--freq', choices=('route', 'matt'), default='route')
    ap.add_argument('--budget', default='base')
    ap.add_argument('--alpha', type=float, default=200.0)
    ap.add_argument('--max-rows', type=int, default=16)
    ap.add_argument('--name', default='route')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', default=None)
    ap.add_argument('--nr', type=int, default=6)
    ap.add_argument('--rank-cap', choices=('none', 'base'), default='none')
    ap.add_argument('--rank-slack', type=float, default=0.0)
    a = ap.parse_args()

    base = json.load(open(a.base))
    budget = manifest_bytes(base) if a.budget == 'base' else int(float(a.budget))
    qs = None
    if a.reqs:
        qs = {r['q'] for r in json.load(open(a.reqs)) if r.get('split') == a.split}
    cnt, steps = route_counts([p for p in a.route.split(',') if p], qs, a.max_rows) if a.freq == 'route' else ({}, 0)
    if a.freq == 'route' and steps == 0:
        sys.exit('no route-log steps matched')

    E, F, HT = {}, {}, {}
    for l in LAYERS:
        eg, eu, ed, ht = load_errs(a.v2, l)
        E[l] = (eg, eu, ed)
        prior = ht / ht.sum() * 8.0                      # Matt's routing share per step (top-8)
        HT[l] = ht
        F[l] = prior if a.freq == 'matt' else (cnt[l] + a.alpha * prior) / (steps + a.alpha)

    RK = {l: (16 * np.arange(NE)[:, None] + np.arange(NH)[None, :] + l) % a.nr for l in LAYERS}   # [e][h] -> rank

    def rank_load(l, gate, up, down):
        """resident and traffic bytes per rank for widths gate/up [e][h], down [e]"""
        hb = (np.asarray(gate) + np.asarray(up) + np.asarray(down)[:, None]) * HALF     # bytes of (e, h) on its rank
        res = np.bincount(RK[l].ravel(), hb.ravel(), a.nr)
        tra = np.bincount(RK[l].ravel(), (F[l][:, None] * hb).ravel(), a.nr)
        return res, tra

    T, CAP = {}, {}
    for l in LAYERS:
        T[l] = rank_load(l, np.full((NE, NH), 2), np.full((NE, NH), 2), np.full(NE, 2))[1]
        b = base['layers'][str(l)]
        CAP[l] = rank_load(l, b['gate'], b['up'], b['down'])[1].max() * (1 + a.rank_slack) if a.rank_cap == 'base' else np.inf
    refused = 0

    # units: (l, kind, e, h); width index wi = k-2; start at K2
    W = {}
    heap = []
    used = 0

    def push(u):
        l, kind, e, h = u
        wi = W[u]
        if wi >= 2:
            return
        eg, eu, ed = E[l]
        if kind == 'd':
            gain, db = F[l][e] * (ed[wi, e] - ed[wi + 1, e]), NH * HALF
        else:
            t = eg if kind == 'g' else eu
            gain, db = F[l][e] * (t[wi, e, h] - t[wi + 1, e, h]), HALF
        heapq.heappush(heap, (-gain / db, u, db))

    for l in LAYERS:
        for e in range(NE):
            for h in range(NH):
                for kind in ('g', 'u'):
                    W[(l, kind, e, h)] = 0
                    used += 2 * HALF
            W[(l, 'd', e, 0)] = 0
            used += 2 * NH * HALF
    for u in W:
        push(u)
    while heap:
        g, u, db = heapq.heappop(heap)
        if used + db > budget:
            continue                                     # a smaller (half) upgrade may still fit
        l, kind, e, h = u
        add = np.zeros(a.nr)
        if kind == 'd':
            np.add.at(add, RK[l][e], F[l][e] * HALF)
        else:
            add[RK[l][e, h]] = F[l][e] * HALF
        if (T[l] + add).max() > CAP[l] + 1e-9:
            refused += 1
            continue                                     # traffic only grows: this unit stays at its width
        T[l] += add
        W[u] += 1
        used += db
        push(u)

    out = {'format': 'exlr8-v2-manifest', 'name': f'{a.name} ({a.freq}, steps {steps}, alpha {a.alpha:g})',
           'rule': f'greedy f*err per byte at budget {budget} B (layers 3-77); L78 + shared copied from base',
           'mtp_layer': base.get('mtp_layer'), 'source': base.get('source'), 'layers': {}}
    rep = {'refused_rank_cap': refused, 'rank_cap': a.rank_cap, 'rank_slack': a.rank_slack, 'steps': steps, 'budget_bytes': budget, 'used_bytes': used, 'base_bytes': manifest_bytes(base), 'layers': {}}
    obj = {'custom': 0.0, 'base': 0.0}
    for l in range(3, 79):
        b = base['layers'][str(l)]
        if l not in LAYERS:
            out['layers'][str(l)] = b
            continue
        gate = [[W[(l, 'g', e, h)] + 2 for h in range(NH)] for e in range(NE)]
        up = [[W[(l, 'u', e, h)] + 2 for h in range(NH)] for e in range(NE)]
        down = [W[(l, 'd', e, 0)] + 2 for e in range(NE)]
        out['layers'][str(l)] = {'gate': gate, 'up': up, 'down': down, **({'shared': b['shared']} if 'shared' in b else {})}
        eg, eu, ed = E[l]
        f = F[l]
        for tag, gg, uu, dd in (('custom', gate, up, down), ('base', b['gate'], b['up'], b['down'])):
            gi, ui, di = np.asarray(gg) - 2, np.asarray(uu) - 2, np.asarray(dd) - 2
            ee = np.arange(NE)
            obj[tag] += float((f[:, None] * (np.take_along_axis(eg.transpose(1, 2, 0), gi[..., None], 2)[..., 0]
                                             + np.take_along_axis(eu.transpose(1, 2, 0), ui[..., None], 2)[..., 0])).sum()
                              + (f * ed[di, ee]).sum())
        hot = set(np.argsort(-f)[:64].tolist())
        matt_hot = set(np.argsort(-HT[l])[:64].tolist())
        base_k4 = {e for e in range(NE) if b['down'][e] == 4}
        cust_k4 = {e for e in range(NE) if down[e] == 4}
        rep['layers'][l] = {
            'hot64_vs_matt_hot64': len(hot & matt_hot), 'hot64_vs_base_k4': len(hot & base_k4),
            'hot64_vs_custom_k4': len(hot & cust_k4), 'mass_top64': float(np.sort(f)[::-1][:64].sum() / f.sum()),
            'mass_on_base_k4': float(f[list(base_k4)].sum() / f.sum()) if base_k4 else 0.0,
            'mass_on_custom_k4': float(f[list(cust_k4)].sum() / f.sum()) if cust_k4 else 0.0,
            'rank_res_max_GB_custom': float(rank_load(l, gate, up, down)[0].max() / 1e9),
            'rank_res_max_GB_base': float(rank_load(l, b['gate'], b['up'], b['down'])[0].max() / 1e9),
            'rank_traffic_max_MB_custom': float(rank_load(l, gate, up, down)[1].max() / 1e6),
            'rank_traffic_max_MB_base': float(rank_load(l, b['gate'], b['up'], b['down'])[1].max() / 1e6),
            'rank_traffic_imbalance_custom': float((lambda t: t.max() / t.mean())(rank_load(l, gate, up, down)[1])),
            'rank_traffic_imbalance_base': float((lambda t: t.max() / t.mean())(rank_load(l, b['gate'], b['up'], b['down'])[1])),
            'custom_mean_bits': float((np.mean(gate) + np.mean(up) + np.mean(down)) / 3)}
    rep['objective'] = obj
    L = rep['layers'].values()
    rep['summary'] = {k: float(np.mean([x[k] for x in L])) for k in next(iter(L))}
    json.dump(out, open(a.out, 'w'))
    if a.report:
        json.dump(rep, open(a.report, 'w'), indent=1)
    print(json.dumps({'refused_rank_cap': refused, 'steps': steps, 'used_GB': used / 1e9, 'budget_GB': budget / 1e9, 'base_GB': rep['base_bytes'] / 1e9,
                      'objective': obj, 'summary': rep['summary']}, indent=1))


if __name__ == '__main__':
    main()
