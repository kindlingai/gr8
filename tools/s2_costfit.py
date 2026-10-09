#!/usr/bin/env python3
"""Fit the S2 step cost model to the engine's round log (G6_S2_ROUNDLOG=1, DEC_S2 build, rank 0 stderr).

Log line, one per decode step (engine/runtime/s2/native.cuh, round_line):
  # s2_round step <n> <main|draft|chain> round <id> active <sessions in the round> S <sessions in this step> M <rows>
      ms <launch+fence ms> path <solo|reuse|capture>
      [ex_mean <mean distinct experts per MoE layer> ex_sum <sum over MoE layers> ex <per-layer counts, comma list>]
The ex fields are present on main steps only. Other lines of the log are ignored. A round with more sessions than fit one
step runs in groups, so S is the group's share; `active` is the whole round's session count and `round` its id (-1 and 0
for steps outside a round, e.g. the chains after an admit). Lines without round/active (older logs) still parse. The
report ends with a per-`active` summary of whole rounds (all steps of one round id added up).

Model, fitted per step kind:
  step_ms = base + per_expert * experts(d * rows) + per_token * rows,   experts(n) = 256 * (1 - (1 - 8/256)^n)
experts(n) is the expected distinct experts one MoE layer touches for n routed rows; d is the routing-diversity
factor (repetitive output touches fewer experts, d < 1). per_expert is ms per (distinct expert per layer), that is
per unit of experts(); with the layer count n_moe in the log, per_expert / n_moe is ms per expert per layer.

main steps:  d is fitted to the logged ex_mean (least squares of ex_mean against experts(d * rows)); then
  (a) "observed": step_ms = base + per_expert * ex_mean + per_token * rows        (what the step really touched)
  (b) "model":    step_ms = base + per_expert * experts(d * rows) + per_token * rows   (the published form)
draft / chain steps carry no expert counts: step_ms = base + per_token * rows (add --draft-experts to include the
experts(d * rows) term with d from the main fit; it is dropped when it is collinear with rows).

  s2_costfit.py LOG [LOG ...] [--include-capture] [--skip N] [--residuals] [--json]
  s2_costfit.py --selftest
"""
import argparse
import json
import math
import re
import sys
import unittest

KINDS = ('main', 'draft', 'chain')
LINE = re.compile(r'#\s*s2_round\b(.*)')


def experts(n, d=1.0):
    """Expected distinct experts one layer touches for d*n routed rows (8 of 256 routed per row)."""
    return 256.0 * (1.0 - (1.0 - 8.0 / 256.0) ** (d * n))


def parse_line(text):
    """-> dict(kind, step, S, M, ms, path[, ex_mean, ex_sum, ex]) or None if the line is not an s2_round line."""
    m = LINE.search(text)
    if not m:
        return None
    w = m.group(1).split()
    try:
        i = w.index('step')
        r = {'step': int(w[i + 1]), 'kind': w[i + 2]}
        kv = dict(zip(w[i + 3::2], w[i + 4::2]))
        r.update(S=int(kv['S']), M=int(kv['M']), ms=float(kv['ms']), path=kv.get('path', ''))
        if 'round' in kv:
            r['round'], r['active'] = int(kv['round']), int(kv.get('active', 0))
        if 'ex_mean' in kv:
            r['ex_mean'] = float(kv['ex_mean'])
            r['ex_sum'] = int(kv['ex_sum'])
            r['ex'] = [int(x) for x in kv['ex'].split(',')] if kv.get('ex') else []
    except (ValueError, KeyError, IndexError):
        return None
    return r if r['kind'] in KINDS else None


def read_logs(paths, include_capture=False, skip=0):
    rows = {k: [] for k in KINDS}
    raw = {}    # (log, round id) -> steps in the log, counted before any filtering
    for src, p in enumerate(paths):
        with open(p, errors='replace') as f:
            for ln in f:
                r = parse_line(ln)
                if r:
                    r['src'] = src
                    if r.get('round', -1) >= 0:
                        raw[(src, r['round'])] = raw.get((src, r['round']), 0) + 1
                    rows[r['kind']].append(r)
    for k in KINDS:
        for r in rows[k]:
            if r.get('round', -1) >= 0:
                r['raw_steps'] = raw[(r['src'], r['round'])]
    for k in KINDS:
        if not include_capture:
            rows[k] = [r for r in rows[k] if r['path'] != 'capture']
        rows[k] = rows[k][skip:]
    return rows


def round_summary(rows):
    """Whole rounds (every step with the same round id, in one log): -> {active: dict(n, steps, rows, ms)} of means.
    A round that lost any step to the capture filter or --skip (fewer steps left than the log had, r['raw_steps']) is
    dropped, so a partial round never shows low rows / ms. A round id of -1 is not a round."""
    rounds = {}
    for kind in KINDS:
        for r in rows[kind]:
            if r.get('round', -1) >= 0:
                x = rounds.setdefault((r['src'], r['round']), {'active': r['active'], 'steps': 0, 'rows': 0, 'ms': 0.0,
                                                               'kinds': set(), 'raw': r.get('raw_steps')})
                x['steps'] += 1
                x['ms'] += r['ms']
                x['kinds'].add(r['kind'])
                if r['kind'] == 'main':
                    x['rows'] += r['M']
    by = {}
    for x in rounds.values():
        if 'main' in x['kinds'] and x['raw'] in (None, x['steps']):
            by.setdefault(x['active'], []).append(x)
    return {a: {'n': len(v), 'steps': sum(x['steps'] for x in v) / len(v), 'rows': sum(x['rows'] for x in v) / len(v),
                'ms': sum(x['ms'] for x in v) / len(v)} for a, v in sorted(by.items())}


def lstsq(cols, y, tol=1e-9):
    """Least squares y ~ sum c_j cols[j] by modified Gram-Schmidt. A column that is (numerically) in the span of the
    ones before it is dropped (coefficient None). -> list of coefficients, one per column."""
    n = len(y)
    q, kept, rmat = [], [], []     # orthonormal basis, kept column indexes, R rows (kept x kept)
    for j, c in enumerate(cols):
        v = list(c)
        norm0 = math.sqrt(sum(x * x for x in v)) or 1.0
        r = []
        for u in q:
            t = sum(a * b for a, b in zip(u, v))
            r.append(t)
            v = [a - t * b for a, b in zip(v, u)]
        nv = math.sqrt(sum(x * x for x in v))
        if nv <= tol * norm0 or n <= len(q):
            continue
        q.append([x / nv for x in v])
        r.append(nv)
        kept.append(j)
        rmat.append((j, r))
    qty = [sum(a * b for a, b in zip(u, y)) for u in q]
    coef = [0.0] * len(q)
    for i in range(len(q) - 1, -1, -1):          # R is upper triangular in kept order
        j, r = rmat[i]
        s = qty[i] - sum(rmat_row_coef(rmat, k, i) * coef[k] for k in range(i + 1, len(q)))
        coef[i] = s / r[i]
    out = [None] * len(cols)
    for i, j in enumerate(kept):
        out[j] = coef[i]
    return out


def rmat_row_coef(rmat, k, i):
    """R[i][k] for k > i: the projection of kept column k onto basis vector i (stored in column k's r list)."""
    return rmat[k][1][i]


def fit_d(rows):
    """d minimizing sum (ex_mean - experts(d * M))^2 over rows with expert counts (golden section on [0.02, 6])."""
    pts = [(r['M'], r['ex_mean']) for r in rows if 'ex_mean' in r]
    if not pts:
        return None
    f = lambda d: sum((e - experts(m, d)) ** 2 for m, e in pts)
    a, b = 0.02, 6.0
    g = (math.sqrt(5) - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = f(c), f(e)
    for _ in range(80):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a)
            fe = f(e)
    return (a + b) / 2


def stats(y, pred):
    res = [a - b for a, b in zip(y, pred)]
    n = len(y)
    mean = sum(y) / n
    sst = sum((a - mean) ** 2 for a in y)
    sse = sum(r * r for r in res)
    return {'rms': math.sqrt(sse / n), 'max_abs': max(abs(r) for r in res),
            'r2': (1 - sse / sst) if sst > 0 else None}, res


def fit_kind(kind, rows, d_main=None, draft_experts=False):
    """-> dict with the fits of one step kind (see the module docstring), or None when there is nothing to fit."""
    if not rows:
        return None
    y = [r['ms'] for r in rows]
    ones = [1.0] * len(rows)
    toks = [float(r['M']) for r in rows]
    out = {'kind': kind, 'n': len(rows), 'fits': {}}
    if kind == 'main' and all('ex_mean' in r for r in rows):
        d = fit_d(rows)
        out['d'] = d
        out['n_moe_layers'] = len(rows[0]['ex']) or None
        out['fits']['observed'] = _one(rows, y, [ones, [r['ex_mean'] for r in rows], toks])
        out['fits']['model'] = _one(rows, y, [ones, [experts(r['M'], d) for r in rows], toks])
    else:
        d = d_main if (draft_experts and d_main is not None) else None
        cols = [ones, toks] if d is None else [ones, [experts(r['M'], d) for r in rows], toks]
        f = _one(rows, y, cols if d is not None else [ones, [0.0] * len(rows), toks])
        out['d'] = d
        out['fits']['model'] = f
    for f in out['fits'].values():
        npl = out.get('n_moe_layers')
        if npl and f['per_expert'] is not None:
            f['us_per_expert_per_layer'] = f['per_expert'] * 1000.0 / npl
    return out


def _one(rows, y, cols):
    c = lstsq(cols, y)
    base, pe, pt = (0.0 if v is None else v for v in c)
    pred = [base + pe * a + pt * b for a, b in zip(cols[1], cols[2])]
    st, res = stats(y, pred)
    st.update(base=base, per_expert=None if c[1] is None else pe, per_token=pt, dropped=[i for i, v in enumerate(c) if v is None],
              pred=pred, resid=res)
    return st


def by_group(rows, fit):
    g = {}
    for r, p in zip(rows, fit['pred']):
        g.setdefault((r['S'], r['M']), []).append((r['ms'], p))
    return [(s, m, len(v), sum(a for a, _ in v) / len(v), sum(p for _, p in v) / len(v)) for (s, m), v in sorted(g.items())]


def report(results, rows, residuals=False):
    lines = []
    for kind in KINDS:
        res = results.get(kind)
        if not res:
            continue
        lines.append(f"== {kind} steps: {res['n']} samples" + (f", d = {res['d']:.3f}" if res.get('d') is not None else '')
                     + (f", {res['n_moe_layers']} MoE layers" if res.get('n_moe_layers') else ''))
        for name, f in res['fits'].items():
            pe = 'n/a (dropped)' if f['per_expert'] is None else f"{f['per_expert']:.4f}"
            extra = f"  ({f['us_per_expert_per_layer']:.2f} us per expert per layer)" if 'us_per_expert_per_layer' in f else ''
            r2 = 'n/a' if f['r2'] is None else f"{f['r2']:.4f}"
            lines.append(f"  [{name}] base {f['base']:.3f} ms  per_expert {pe} ms{extra}  per_token {f['per_token']:.4f} ms"
                         f"  | rms resid {f['rms']:.3f} ms  max |resid| {f['max_abs']:.3f} ms  R2 {r2}")
        f = res['fits'].get('model')
        if f:
            lines.append('  residuals of [model] by (S, M):   S    M     n   mean_ms  pred_ms  resid_ms')
            for s, m, n, a, p in by_group(rows[kind], f):
                lines.append(f'                                 {s:3d} {m:4d} {n:5d} {a:9.3f} {p:8.3f} {a - p:9.3f}')
        if residuals and f:
            lines.append('  every sample: step S M ms pred resid')
            for r, p, e in zip(rows[kind], f['pred'], f['resid']):
                lines.append(f"    {r['step']} {r['S']} {r['M']} {r['ms']:.3f} {p:.3f} {e:.3f}")
    rs = round_summary(rows)
    if rs:
        lines.append('== whole rounds by active sessions (all steps of a round id added up)')
        lines.append('   active  rounds  steps/round  main_rows  ms/round')
        for a, v in rs.items():
            lines.append(f"   {a:6d} {v['n']:7d} {v['steps']:12.2f} {v['rows']:10.1f} {v['ms']:9.3f}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='*', help='engine rank-0 stderr log(s) with "# s2_round" lines')
    ap.add_argument('--include-capture', action='store_true', help='keep the first-use graph capture steps (dropped by default)')
    ap.add_argument('--skip', type=int, default=0, help='drop the first N samples of every kind (warmup)')
    ap.add_argument('--draft-experts', action='store_true', help='draft/chain: include the experts(d*rows) term (d from the main fit)')
    ap.add_argument('--residuals', action='store_true', help='print every sample with its residual')
    ap.add_argument('--json', action='store_true', help='print the fits as JSON instead of text')
    ap.add_argument('--selftest', action='store_true', help='run the unit tests on synthetic data')
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if unittest.main(argv=[sys.argv[0]], module=__name__, exit=False).result.wasSuccessful() else 1
    if not a.logs:
        ap.error('no log files')
    rows = read_logs(a.logs, a.include_capture, a.skip)
    if not any(rows.values()):
        print('no "# s2_round" lines found (engine needs a DEC_S2 build and G6_S2_ROUNDLOG=1 on rank 0)', file=sys.stderr)
        return 1
    results = {}
    results['main'] = fit_kind('main', rows['main'])
    d_main = results['main'].get('d') if results['main'] else None
    for k in ('draft', 'chain'):
        results[k] = fit_kind(k, rows[k], d_main, a.draft_experts)
    if a.json:
        slim = {'rounds': round_summary(rows), **{k: None if v is None else {**{x: v[x] for x in v if x != 'fits'},
                'fits': {n: {x: y for x, y in f.items() if x not in ('pred', 'resid')} for n, f in v['fits'].items()}}
                for k, v in results.items()}}
        print(json.dumps(slim, indent=1))
    else:
        print(report(results, rows, a.residuals))
    return 0


# ---- unit tests (synthetic data) ----------------------------------------------------------------------------------
def _synthetic(kind, base, pe, pt, d, sizes, reps, noise, seed, n_layers=75, with_ex=True):
    import random
    rnd = random.Random(seed)
    lines = []
    step = 0
    for _ in range(reps):
        for s, m in sizes:
            step += 1
            ex = experts(m, d)
            ms = base + pe * ex + pt * m + rnd.gauss(0, noise)
            ln = f'# s2_round step {step} {kind} round {step} active {s} S {s} M {m} ms {ms:.3f} path reuse'
            if with_ex:
                counts = [max(1, int(round(ex + rnd.gauss(0, 1.0)))) for _ in range(n_layers)]
                ln += f' ex_mean {sum(counts) / n_layers:.3f} ex_sum {sum(counts)} ex ' + ','.join(map(str, counts))
            lines.append(ln)
    return lines


class SyntheticTest(unittest.TestCase):
    SIZES = [(1, 3), (2, 6), (3, 9), (4, 12), (6, 18), (8, 24)]

    def rows(self, lines):
        return [parse_line(ln) for ln in lines]

    def test_parse(self):
        r = parse_line('x # s2_round step 7 main S 4 M 12 ms 81.250 path reuse ex_mean 60.500 ex_sum 4 ex 60,61,60,61')
        self.assertEqual((r['kind'], r['step'], r['S'], r['M'], r['path']), ('main', 7, 4, 12, 'reuse'))
        self.assertAlmostEqual(r['ms'], 81.25)
        self.assertEqual(r['ex'], [60, 61, 60, 61])
        self.assertIsNone(parse_line('# step_tmr rank 0 step 3 main ROUND_MAIN S2_REUSE 800 us'))
        self.assertIsNone(parse_line('# s2_round step 7 bogus S 4 M 12 ms 1 path reuse'))
        r = parse_line('# s2_round step 9 draft S 4 M 8 ms 20.1 path capture')
        self.assertEqual(r['kind'], 'draft')
        self.assertNotIn('ex_mean', r)

    def test_round_fields_and_summary(self):
        r = parse_line('# s2_round step 5 main round 12 active 4 S 2 M 6 ms 40.0 path reuse ex_mean 44.0 ex_sum 44 ex 44')
        self.assertEqual((r['S'], r['M'], r['round'], r['active']), (2, 6, 12, 4))
        self.assertNotIn('round', parse_line('# s2_round step 5 main S 2 M 6 ms 40.0 path reuse'))
        lines = ['# s2_round step 1 main round 0 active 4 S 2 M 6 ms 40.0 path reuse ex_mean 44.0 ex_sum 44 ex 44',
                 '# s2_round step 2 main round 0 active 4 S 2 M 6 ms 42.0 path reuse ex_mean 44.0 ex_sum 44 ex 44',
                 '# s2_round step 3 draft round 0 active 4 S 4 M 8 ms 10.0 path reuse',
                 '# s2_round step 4 main round 1 active 1 S 1 M 3 ms 30.0 path reuse ex_mean 20.0 ex_sum 20 ex 20',
                 '# s2_round step 5 chain round -1 active 0 S 1 M 1 ms 5.0 path reuse']
        rows = {k: [] for k in KINDS}
        for ln in lines:
            r = parse_line(ln)
            r['src'] = 0
            rows[r['kind']].append(r)
        rs = round_summary(rows)
        self.assertEqual(sorted(rs), [1, 4])
        self.assertEqual((rs[4]['n'], rs[4]['steps'], rs[4]['rows']), (1, 3, 12))
        self.assertAlmostEqual(rs[4]['ms'], 92.0)
        self.assertAlmostEqual(rs[1]['ms'], 30.0)

    def test_partial_round_excluded(self):
        import os
        import tempfile
        ln = '# s2_round step {} {} round {} active 4 S 2 M {} ms {} path {}'
        lines = []
        for rd in range(3):     # S4 rounds as two groups: each group = main + draft + chain
            for g in range(2):
                st = rd * 6 + g * 3
                lines += [ln.format(st + 1, 'main', rd, 6, 40.0, 'reuse') + ' ex_mean 44.0 ex_sum 44 ex 44',
                          ln.format(st + 2, 'draft', rd, 6, 10.0, 'reuse'), ln.format(st + 3, 'chain', rd, 2, 5.0, 'reuse')]
        lines[0] = lines[0].replace('path reuse', 'path capture')      # round 0 loses its first main step to the capture filter
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as t:
            t.write('\n'.join(lines) + '\n')
        try:
            rs = round_summary(read_logs([t.name]))
            self.assertEqual(rs[4]['n'], 2)                       # rounds 1 and 2 only
            self.assertEqual((rs[4]['rows'], rs[4]['steps'], rs[4]['ms']), (12, 6, 110.0))
            rs = round_summary(read_logs([t.name], include_capture=True))
            self.assertEqual(rs[4]['n'], 3)
            # --skip 1 drops the first main step of the log: round 0 again, nothing else
            rs = round_summary(read_logs([t.name], include_capture=True, skip=1))
            self.assertEqual(rs[4]['n'], 2)
            self.assertEqual(rs[4]['rows'], 12)
            # skipping past a later main step also drops its round
            self.assertEqual(round_summary(read_logs([t.name], skip=2))[4]['n'], 1)
        finally:
            os.unlink(t.name)

    def test_main_fit_recovers_parameters(self):
        rows = self.rows(_synthetic('main', 45.0, 0.40, 0.50, 0.7, self.SIZES, 40, 0.3, 1))
        res = fit_kind('main', rows)
        self.assertAlmostEqual(res['d'], 0.7, delta=0.05)
        f = res['fits']['model']
        self.assertAlmostEqual(f['base'], 45.0, delta=2.0)
        self.assertAlmostEqual(f['per_expert'], 0.40, delta=0.08)
        self.assertAlmostEqual(f['per_token'], 0.50, delta=0.25)
        self.assertLess(f['rms'], 1.0)
        self.assertGreater(f['r2'], 0.99)
        self.assertEqual(len(f['resid']), len(rows))
        self.assertAlmostEqual(f['us_per_expert_per_layer'], f['per_expert'] * 1000 / 75, places=6)
        self.assertLess(res['fits']['observed']['rms'], 1.0)

    def test_exact_data_zero_residual(self):
        rows = self.rows(_synthetic('main', 50.0, 0.30, 0.6, 1.0, self.SIZES, 3, 0.0, 2))
        f = fit_kind('main', rows)['fits']['model']
        self.assertLess(f['rms'], 0.5)   # ex counts are rounded to integers, the cost is exact in experts(d*M)

    def test_chain_fit_two_parameters(self):
        rows = self.rows(_synthetic('chain', 12.0, 0.0, 0.8, 1.0, [(s, s) for s in (1, 2, 4, 6, 8)], 30, 0.2, 3, with_ex=False))
        f = fit_kind('chain', rows)['fits']['model']
        self.assertAlmostEqual(f['base'], 12.0, delta=0.5)
        self.assertAlmostEqual(f['per_token'], 0.8, delta=0.15)
        self.assertIsNone(f['per_expert'])

    def test_collinear_column_dropped(self):
        c = lstsq([[1, 1, 1, 1], [2, 4, 6, 8], [1, 2, 3, 4]], [3, 5, 7, 9])
        self.assertIsNone(c[2])           # third column is half of the second
        self.assertAlmostEqual(c[0], 1.0, places=6)
        self.assertAlmostEqual(c[1], 1.0, places=6)

    def test_capture_rows_dropped_and_logs_read(self):
        import os
        import tempfile
        lines = _synthetic('main', 40.0, 0.5, 0.4, 1.0, self.SIZES, 5, 0.0, 4)
        lines.append('# s2_round step 99 main S 2 M 6 ms 5000.0 path capture ex_mean 30 ex_sum 30 ex 30')
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as t:
            t.write('\n'.join(['engine: unrelated line'] + lines) + '\n')
        try:
            rows = read_logs([t.name])
            self.assertEqual(len(rows['main']), 5 * len(self.SIZES))
            self.assertEqual(len(read_logs([t.name], include_capture=True)['main']), 5 * len(self.SIZES) + 1)
            self.assertEqual(len(read_logs([t.name], skip=4)['main']), 5 * len(self.SIZES) - 4)
            self.assertIn('== main steps', report({'main': fit_kind('main', rows['main'])}, rows))
        finally:
            os.unlink(t.name)

    def test_experts_formula(self):
        self.assertAlmostEqual(experts(0), 0.0)
        self.assertAlmostEqual(experts(1), 8.0)
        self.assertAlmostEqual(experts(6), 256 * (1 - (248 / 256) ** 6))
        self.assertEqual(round(experts(12)), 81)   # about 81 distinct experts at 12 rows, 44-45 at 6 rows (issue #28)
        self.assertIn(round(experts(6)), (44, 45))


if __name__ == '__main__':
    sys.exit(main())
