#!/usr/bin/env python3
"""Write the drafter choice's cost table (G6_S2_DSEL_TABLE, engine/runtime/s2_batch.h DSelTable) from one boot's logs.

Input: rank logs (stderr) of a boot with G6_S2_TIMELINE=N on every rank, and on rank 0 G6_MTP_LPLOG=1 (the "# kstop"
lines). Every "# s2_tl" block is cumulative, so the last one of each rank counts; the ranks are averaged. Lines used:
  # s2_tl rank R step <kind> n N rows X span S prep P launch L wait W ms      per step kind, mean GPU span
  # s2_tl rank R rows <kind> M <rows> n N span S ms                          per step kind and row count
  # s2_tl rank R trans <a>><b> n N idle I host H ms                          GPU idle between two steps
  # kstop round .. slot .. n .. <mtp|dfl> base B k K nd ND a A ...            one per solo round (rank 0)
Optionally (--roundlog) "# s2_round" lines (G6_S2_ROUNDLOG=1) give the verify ms by rows where no "rows main" line does.

A step costs its GPU span plus the mean GPU idle before a step of its kind (the host's accept, emit and planning):
  v<M> = main span at M rows + idle before main                 (rows with at least --min-n steps)
  f    = the first-form draft step; on DFlash2 and MTP rounds it takes a + 1 rows, so with "# kstop" lines it is the
         per-row span weighted by each drafter's accept counts, else the mean over all of them
  d    = f(DFlash2 rounds) + dfl_query          m = f(MTP rounds) + dfl_ctx          e = chain
  p    = dfl_query - dfl_ctx (a probe is a query in place of a context-only step)
  g k n from the MTP rounds (at least --min-n): drafts accepted after the first when the first is, verified, drafted.
The table goes to stdout (one line, for G6ENV_X), the derivation to stderr.

Measurement boot: the serving configuration being priced (its K-stop policy, DSel settings and kernels) with the model
(G6_S2_DSEL_MODEL, G6_S2_DSEL_TABLE) and G6_DRAFT_PAR removed, and G6_S2_TIMELINE=200 G6_MTP_LPLOG=1 appended to its
G6ENV_X (no G6_S2_ROUNDLOG: its route copies lengthen the main steps). Run
solo decode that visits both drafters (RigMark c1 prose, code and structured), then pass every rank's log. Check the
derivation: each v<M> the drafters reach (2..4 MTP, 2..8 DFlash2) from enough steps, and enough MTP rounds for g k n.

  s2_dsel_table.py LOG [LOG ...] [--min-n 30] [--roundlog] [--no-prior] [--w W] [--r R]
  s2_dsel_table.py --selftest
"""
import argparse
import re
import statistics
import sys
import unittest

KINDS = ('main', 'draft', 'chain', 'dfl_ctx', 'dfl_query')
VX = 8
STEP = re.compile(r'# s2_tl rank (\d+) step (\w+) n (\d+) rows ([\d.]+) span ([\d.]+) prep ([\d.]+) launch ([\d.]+) wait ([\d.]+) ms')
ROWS = re.compile(r'# s2_tl rank (\d+) rows (\w+) M (\d+) n (\d+) span ([\d.]+) ms')
TRANS = re.compile(r'# s2_tl rank (\d+) trans (\w+)>(\w+) n (\d+) idle ([\d.]+) host ([\d.]+) ms')
KSTOP = re.compile(r'# kstop round (-?\d+) slot \d+ n \d+ (mtp|dfl) base \d+ k (\d+) nd (\d+) a (\d+)')
RLOG = re.compile(r'# s2_round step \d+ main round (-?\d+) active \d+ S \d+ M (\d+) ms ([\d.]+) path solo')


def read(paths, roundlog=False):
    """-> dict: step[rank][kind] = (n, span), rows[rank][(kind, M)] = (n, span), trans[rank][(a, b)] = (n, idle),
    rounds = [(kind, k, nd, a)] (rank 0's kstop lines), rlog = {M: [ms]}. Later blocks of a rank replace earlier ones."""
    out = {'step': {}, 'rows': {}, 'trans': {}, 'rounds': [], 'rlog': {}}
    for p in paths:
        with open(p, errors='replace') as f:
            for ln in f:
                m = STEP.search(ln)
                if m:
                    out['step'].setdefault(int(m.group(1)), {})[m.group(2)] = (int(m.group(3)), float(m.group(5)))
                    continue
                m = ROWS.search(ln)
                if m:
                    out['rows'].setdefault(int(m.group(1)), {})[(m.group(2), int(m.group(3)))] = (int(m.group(4)), float(m.group(5)))
                    continue
                m = TRANS.search(ln)
                if m:
                    out['trans'].setdefault(int(m.group(1)), {})[(m.group(2), m.group(3))] = (int(m.group(4)), float(m.group(5)))
                    continue
                m = KSTOP.search(ln)
                if m:
                    out['rounds'].append((m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))))
                    continue
                m = RLOG.search(ln) if roundlog else None
                if m and int(m.group(1)) >= 0:
                    out['rlog'].setdefault(int(m.group(2)), []).append(float(m.group(3)))
    return out


def rank_mean(per_rank, key, min_n=1):
    """mean over the ranks that have key with at least min_n samples -> (n of rank 0 or the first, value) or None"""
    vals = [d[key] for d in per_rank.values() if key in d and d[key][0] >= min_n]
    if not vals:
        return None
    return vals[0][0], sum(v[1] for v in vals) / len(vals)


def idle_before(trans, kind):
    """mean GPU idle before a step of this kind, over every predecessor and rank"""
    n = s = 0.0
    for d in trans.values():
        for (a, b), (c, idle) in d.items():
            if b == kind:
                n += c
                s += c * idle
    return s / n if n else 0.0


def first_form(lg, kind_rounds, min_n):
    """the first-form step's cost on rounds of one drafter: the per-row spans weighted by that drafter's a + 1, or the
    mean over all first forms when there are no accept counts or a row count was never seen"""
    pooled = rank_mean(lg['step'], 'draft')
    if pooled is None:
        raise SystemExit('no "# s2_tl ... step draft" line: was G6_S2_TIMELINE set?')
    if not kind_rounds:
        return pooled[1], 'pooled'
    w = {}
    for _, _, _, a in kind_rounds:
        w[a + 1] = w.get(a + 1, 0) + 1
    s = 0.0
    for M, c in w.items():
        r = rank_mean(lg['rows'], ('draft', M), 1)
        if r is None:
            return pooled[1], 'pooled (no draft span at %d rows)' % M
        s += c * r[1]
    return s / sum(w.values()), 'by accepted rows'


def table(lg, min_n=30, prior=True, log=sys.stderr):
    """-> (canonical-order dict of the table's keys, list of derivation lines)"""
    t, why = {}, []
    if any(k.endswith('_par') for d in lg['step'].values() for k in d):
        raise SystemExit('"dfl_*_par" steps: under G6_DRAFT_PAR the drafter step overlaps the first form; measure with it off')
    im = idle_before(lg['trans'], 'main')
    for M in range(1, VX + 1):
        r = rank_mean(lg['rows'], ('main', M), min_n)
        if r is not None:
            t['v%d' % M] = r[1] + im
            why.append('v%d = main span %.3f + idle %.3f (n %d)' % (M, r[1], im, r[0]))
        elif M in lg['rlog'] and len(lg['rlog'][M]) >= min_n:
            t['v%d' % M] = statistics.median(lg['rlog'][M])
            why.append('v%d = median s2_round launch+fence %.3f (n %d)' % (M, t['v%d' % M], len(lg['rlog'][M])))
    if sum(1 for k in t if k[0] == 'v') < 2:
        raise SystemExit('fewer than two verify row counts with %d steps: lower --min-n or run more decode' % min_n)
    cost = {}
    for k in ('chain', 'dfl_ctx', 'dfl_query'):
        r = rank_mean(lg['step'], k)
        if r is None:
            raise SystemExit('no "# s2_tl ... step %s" line: a DFlash2 session with MTP rounds is needed' % k)
        cost[k] = r[1] + idle_before(lg['trans'], k)
        why.append('%s = span %.3f + idle %.3f (n %d)' % (k, r[1], cost[k] - r[1], r[0]))
    idf = idle_before(lg['trans'], 'draft')
    fd, hd = first_form(lg, [x for x in lg['rounds'] if x[0] == 'dfl'], min_n)
    fm, hm = first_form(lg, [x for x in lg['rounds'] if x[0] == 'mtp'], min_n)
    why.append('first form + idle %.3f: DFlash2 rounds %.3f, MTP rounds %.3f (%s / %s)' % (idf, fd, fm, hd, hm))
    t['d'] = fd + idf + cost['dfl_query']
    t['m'] = fm + idf + cost['dfl_ctx']
    t['e'] = cost['chain']
    t['p'] = max(0.0, cost['dfl_query'] - cost['dfl_ctx'])
    mtp = [x for x in lg['rounds'] if x[0] == 'mtp']
    if prior and len(mtp) >= min_n:
        acc = [x for x in mtp if x[3] >= 1]
        t['g'] = sum(x[3] - 1 for x in acc) / len(acc) if acc else 0.0
        t['k'] = sum(x[1] for x in mtp) / len(mtp)
        t['n'] = sum(x[2] for x in mtp) / len(mtp)
        why.append('MTP rounds %d (%d with d1 accepted): g %.4f, k %.4f, n %.4f; first-position rate %.4f' %
                   (len(mtp), len(acc), t['g'], t['k'], t['n'], len(acc) / len(mtp)))
    elif prior:
        why.append('MTP prior left out: %d MTP "# kstop" lines < %d (G6_MTP_LPLOG=1 on rank 0)' % (len(mtp), min_n))
    return t, why


def fmt(t):
    return ','.join('%s:%.4g' % (k, v) for k, v in t.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--min-n', type=int, default=30, help='steps (rounds) a cell needs')
    ap.add_argument('--roundlog', action='store_true', help='fill verify rows from "# s2_round" lines (launch+fence)')
    ap.add_argument('--no-prior', action='store_true', help='leave out g, k, n')
    ap.add_argument('--w', type=float, default=None, help="the acceptance estimates' prior weight in rounds (engine default 4)")
    ap.add_argument('--r', type=float, default=None, help="price DFlash2 from its rounds, aged by R per MTP round (e.g. 0.95)")
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful() else 1
    if not a.logs:
        ap.error('no log')
    t, why = table(read(a.logs, a.roundlog), a.min_n, not a.no_prior)
    if a.w is not None:
        t['w'] = a.w
    if a.r is not None:
        t['r'] = a.r
    for w in why:
        print('# ' + w, file=sys.stderr)
    print(fmt(t))
    return 0


class Test(unittest.TestCase):
    LOG = """\
# s2_tl rank 0 mains 100: per main step: GPU idle 0.5 ms
# s2_tl rank 0 step main n 100 rows 3.0 span 50.000 prep 0.1 launch 0.1 wait 50.0 ms
# s2_tl rank 0 step main n 400 rows 3.0 span 55.000 prep 0.1 launch 0.1 wait 55.0 ms
# s2_tl rank 0 step draft n 400 rows 2.0 span 1.500 prep 0.1 launch 0.1 wait 1.5 ms
# s2_tl rank 0 step chain n 300 rows 1.0 span 1.100 prep 0.1 launch 0.1 wait 1.1 ms
# s2_tl rank 0 step dfl_ctx n 250 rows 2.0 span 0.500 prep 0.1 launch 0.1 wait 0.5 ms
# s2_tl rank 0 step dfl_query n 150 rows 3.0 span 5.400 prep 0.1 launch 0.1 wait 5.4 ms
# s2_tl rank 0 rows main M 2 n 100 span 48.000 ms
# s2_tl rank 0 rows main M 3 n 200 span 56.000 ms
# s2_tl rank 0 rows main M 4 n 60 span 64.000 ms
# s2_tl rank 0 rows main M 8 n 5 span 100.000 ms
# s2_tl rank 0 rows draft M 1 n 100 span 1.200 ms
# s2_tl rank 0 rows draft M 2 n 200 span 1.400 ms
# s2_tl rank 0 rows draft M 3 n 100 span 1.600 ms
# s2_tl rank 0 trans main>draft n 400 idle 0.100 host 0.2 ms
# s2_tl rank 0 trans draft>dfl_ctx n 250 idle 0.050 host 0.1 ms
# s2_tl rank 0 trans draft>dfl_query n 150 idle 0.050 host 0.1 ms
# s2_tl rank 0 trans dfl_ctx>chain n 250 idle 0.020 host 0.1 ms
# s2_tl rank 0 trans chain>main n 300 idle 0.200 host 0.3 ms
# s2_tl rank 0 trans dfl_query>main n 100 idle 0.400 host 0.3 ms
# s2_tl rank 1 rows main M 3 n 200 span 58.000 ms
"""

    def lg(self, extra=''):
        import os
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as f:
            f.write(self.LOG + extra)
        try:
            return read([f.name])
        finally:
            os.unlink(f.name)

    def test_last_block_and_rank_mean(self):
        lg = self.lg()
        self.assertEqual(lg['step'][0]['main'], (400, 55.0))
        self.assertEqual(rank_mean(lg['rows'], ('main', 3)), (200, 57.0))   # ranks 0 and 1

    def test_table(self):
        t, _ = table(self.lg(), min_n=30)
        im = (300 * 0.2 + 100 * 0.4) / 400          # idle before main
        self.assertAlmostEqual(t['v2'], 48 + im)
        self.assertAlmostEqual(t['v3'], 57 + im)
        self.assertAlmostEqual(t['v4'], 64 + im)
        self.assertNotIn('v8', t)                   # 5 steps < min_n
        self.assertAlmostEqual(t['e'], 1.1 + 0.02)                 # chain span + idle after the context-only step
        self.assertAlmostEqual(t['d'], 1.5 + 0.1 + 5.4 + 0.05)
        self.assertAlmostEqual(t['m'], 1.5 + 0.1 + 0.5 + 0.05)
        self.assertAlmostEqual(t['p'], 4.9)
        self.assertNotIn('g', t)                    # no kstop lines

    def test_prior_and_first_form_by_rows(self):
        ks = ''.join('# kstop round %d slot 0 n %d mtp base 10 k %d nd %d a %d lp -0.1 m1 -0.1 m1ok 1 pe 0 hash 0\n' %
                     (i, i, 1 + i % 2, 2, i % 3) for i in range(60))
        ks += ''.join('# kstop round %d slot 0 n %d dfl base 10 k 7 nd 7 a 2 lp -0.1 m1 -0.1 m1ok 1 pe 0 hash 0\n' % (i, i) for i in range(40))
        t, _ = table(self.lg(ks), min_n=30)
        mtp = [(1 + i % 2, 2, i % 3) for i in range(60)]
        acc = [x for x in mtp if x[2] >= 1]
        self.assertAlmostEqual(t['g'], sum(x[2] - 1 for x in acc) / len(acc))
        self.assertAlmostEqual(t['k'], 1.5)
        self.assertAlmostEqual(t['n'], 2.0)
        self.assertAlmostEqual(t['d'], 1.6 + 0.1 + 5.4 + 0.05)          # DFlash2 rounds: a = 2, first form of 3 rows
        self.assertAlmostEqual(t['m'], (20 * 1.2 + 20 * 1.4 + 20 * 1.6) / 60 + 0.1 + 0.5 + 0.05)

    def test_format_parses_as_engine_list(self):
        s = fmt({'v2': 49.6, 'v3': 57.0, 'd': 6.9, 'm': 2.0, 'e': 1.17, 'p': 4.9})
        self.assertEqual(s, 'v2:49.6,v3:57,d:6.9,m:2,e:1.17,p:4.9')
        self.assertNotIn(' ', s)
        self.assertNotIn(';', s)                    # it travels through G6ENV_X and a shell command line


if __name__ == '__main__':
    sys.exit(main())
