#!/usr/bin/env python3
"""DSpark canary comparator (docs/DEEPSEEK-DSPARK.md section 7): the output tokens of GR8_DS_MTP=0 against MTP=1 and MTP=5 at temp 0,
plus the acceptance per class. It compares run files; whatever drives the booked box writes them (one JSONL per run):

  {"type":"hdr","mtp":5,"engram":1,"temp":0,"label":"real"}       label "diag" when engram=0
  {"type":"gen","id":"prose-1","class":"prose","tokens":[...]}    all emitted token ids, in order, cut at max_tokens
  {"type":"round","id":"prose-1","k":5,"acc":3}                   per verify round: drafts proposed k, accepted acc (MTP>0 runs only)
  {"type":"top2","id":"prose-1","i":17,"gap":0.31}                optional: top-2 logprob gap at token index i, for tie triage

Gates (a failure at a lower gate makes the higher ones meaningless):
  G0 MTP=0 vs MTP=0 (run-to-run determinism), G1 MTP=0 vs MTP=1, G2 MTP=0 vs MTP=5 (the done criterion).
Equality is the md5 of the token list as compact JSON. The comparator does not guess whether a divergence is a tie flip: it prints the
first divergent index, both tokens and the top2 gap when known. Any input run with engram=0 makes the whole report "diag".

  usage: ds41_dspark_canary.py --mtp0 RUN [--mtp0b RUN] [--mtp1 RUN] [--mtp5 RUN] [--fixture F] [--json OUT]
  exit:  0 every requested gate is EQUAL and G2 was requested, 1 a gate has a DIFF or G2 was not requested, 2 bad input
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_HIST = 5  # block is always 5 rows (D1), so a round accepts 0..5 drafts


class InputError(Exception):
    pass


def md5_tokens(tokens):
    return hashlib.md5(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def load_fixture(path):
    try:
        fx = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise InputError('fixture %s: %s' % (path, e))
    st = fx.get('settings', {})
    if st.get('temperature') != 0:
        raise InputError('fixture: settings.temperature must be 0')
    if not isinstance(st.get('max_tokens'), int) or st['max_tokens'] <= 0:
        raise InputError('fixture: settings.max_tokens must be a positive integer')
    classes = fx.get('classes') or []
    seen = set()
    for p in fx.get('prompts', []):
        if p.get('id') in seen:
            raise InputError('fixture: duplicate prompt id %r' % p.get('id'))
        seen.add(p.get('id'))
        if p.get('class') not in classes:
            raise InputError('fixture: prompt %r has class %r not in %s' % (p.get('id'), p.get('class'), classes))
        if not p.get('messages'):
            raise InputError('fixture: prompt %r has no messages' % p.get('id'))
    if not seen:
        raise InputError('fixture: no prompts')
    return fx


def load_run(path, fixture, expect_mtp):
    """Parse and validate one run file. expect_mtp is the MTP value the gate expects the header to carry."""
    run = {'path': str(path), 'hdr': None, 'gen': {}, 'cls': {}, 'rounds': {}, 'top2': {}}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as e:
        raise InputError('run %s: %s' % (path, e))
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = '%s:%d' % (path, n)
        try:
            rec = json.loads(line)
            t = rec['type']
        except (ValueError, KeyError, TypeError):
            raise InputError('%s: not a JSON object with a "type"' % where)
        if t == 'hdr':
            if run['hdr'] is not None:
                raise InputError('%s: second hdr' % where)
            run['hdr'] = rec
        elif t == 'gen':
            if rec['id'] in run['gen']:
                raise InputError('%s: duplicate gen for %r' % (where, rec['id']))
            if not all(isinstance(x, int) for x in rec['tokens']):
                raise InputError('%s: tokens must be integers' % where)
            run['gen'][rec['id']] = rec['tokens']
            run['cls'][rec['id']] = rec.get('class')
        elif t == 'round':
            k, acc = rec['k'], rec['acc']
            if not (isinstance(k, int) and isinstance(acc, int) and 0 <= acc <= k <= MAX_HIST):
                raise InputError('%s: round needs integers 0 <= acc <= k <= %d, got k=%r acc=%r' % (where, MAX_HIST, k, acc))
            run['rounds'].setdefault(rec['id'], []).append((k, acc))
        elif t == 'top2':
            run['top2'].setdefault(rec['id'], {})[rec['i']] = rec['gap']
        else:
            raise InputError('%s: unknown type %r' % (where, t))
    hdr = run['hdr']
    if hdr is None:
        raise InputError('%s: no hdr line' % path)
    if hdr.get('temp') != 0:
        raise InputError('%s: temp must be 0, got %r' % (path, hdr.get('temp')))
    if hdr.get('mtp') != expect_mtp:
        raise InputError('%s: expected mtp=%d for this gate, hdr says %r' % (path, expect_mtp, hdr.get('mtp')))
    if hdr.get('engram') not in (0, 1):
        raise InputError('%s: hdr.engram must be 0 or 1' % path)
    run['diag'] = hdr['engram'] == 0
    if run['diag'] != (hdr.get('label') == 'diag') or hdr.get('label') not in ('real', 'diag'):
        raise InputError('%s: hdr.label must be "diag" exactly when engram=0 (and "real" otherwise), got engram=%r label=%r'
                         % (path, hdr['engram'], hdr.get('label')))
    ids = {p['id']: p['class'] for p in fixture['prompts']}
    if set(run['gen']) != set(ids):
        raise InputError('%s: prompts differ from the fixture: missing %s, extra %s'
                         % (path, sorted(set(ids) - set(run['gen'])), sorted(set(run['gen']) - set(ids))))
    for i, c in run['cls'].items():
        if c != ids[i]:
            raise InputError('%s: prompt %r has class %r, fixture says %r' % (path, i, c, ids[i]))
    cap = fixture['settings']['max_tokens']
    for i, toks in run['gen'].items():
        if len(toks) > cap:
            raise InputError('%s: prompt %r emitted %d tokens > max_tokens %d (the harness must cut there)' % (path, i, len(toks), cap))
    if expect_mtp == 0 and run['rounds']:
        raise InputError('%s: an MTP=0 run must not carry round lines' % path)
    for i, rs in run['rounds'].items():
        if i not in ids:
            raise InputError('%s: round for unknown prompt %r' % (path, i))
        for k, _ in rs:
            if k > expect_mtp:
                raise InputError('%s: round k=%d exceeds mtp=%d' % (path, k, expect_mtp))
    return run


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def compare_runs(fixture, base, other):
    """Per prompt: md5 of both, EQUAL/DIFF, first divergent index with both tokens (None when one run ended) and the top2 gap."""
    out = []
    for p in fixture['prompts']:
        i = p['id']
        a, b = base['gen'][i], other['gen'][i]
        row = {'id': i, 'class': p['class'], 'md5_a': md5_tokens(a), 'md5_b': md5_tokens(b), 'len_a': len(a), 'len_b': len(b)}
        row['equal'] = row['md5_a'] == row['md5_b']
        row['div'] = None
        if not row['equal']:
            d = first_divergence(a, b)
            gap = base['top2'].get(i, {}).get(d)
            if gap is None:
                gap = other['top2'].get(i, {}).get(d)
            row['div'] = {'i': d, 'tok_a': a[d] if d < len(a) else None, 'tok_b': b[d] if d < len(b) else None, 'top2_gap': gap}
        out.append(row)
    return out


def _stats(rounds):
    hist = [0] * (MAX_HIST + 1)
    for _, acc in rounds:
        hist[acc] += 1
    n = len(rounds)
    total = sum(acc for _, acc in rounds)
    per_pos = []
    for r in range(MAX_HIST):
        reached = sum(1 for k, acc in rounds if k > r and acc >= r)  # draft r proposed and drafts 0..r-1 all accepted
        took = sum(1 for k, acc in rounds if acc > r)
        per_pos.append(took / reached if reached else None)
    return {'rounds': n, 'mean_a': total / n if n else None, 'tokens_per_round': 1 + total / n if n else None, 'hist': hist,
            'per_pos': per_pos}


def acceptance(fixture, run):
    """Acceptance per class and overall, from the round lines of an MTP>0 run."""
    out = {}
    pooled = []
    for c in fixture['classes']:
        rs = []
        for p in fixture['prompts']:
            if p['class'] == c:
                rs += run['rounds'].get(p['id'], [])
        out[c] = _stats(rs)
        pooled += rs
    out['all'] = _stats(pooled)
    return out


def build_report(fixture, runs):
    """runs: dict with 'mtp0' (required) and optional 'mtp0b', 'mtp1', 'mtp5' (parsed). Returns the result dict."""
    base = runs['mtp0']
    engram = {r['hdr']['engram'] for r in runs.values()}
    if len(engram) != 1:
        raise InputError('the runs disagree on engram (%s): their tokens cannot be compared' % sorted(engram))
    diag = base['diag']
    gates = []
    lower_failed = None
    for name, key, label in (('G0', 'mtp0b', 'MTP=0 vs MTP=0'), ('G1', 'mtp1', 'MTP=0 vs MTP=1'), ('G2', 'mtp5', 'MTP=0 vs MTP=5')):
        if key not in runs:
            continue
        rows = compare_runs(fixture, base, runs[key])
        equal = all(r['equal'] for r in rows)
        gates.append({'name': name, 'label': label, 'equal': equal, 'prompts': rows,
                      'meaningless': lower_failed is not None, 'after': lower_failed})
        if not equal and lower_failed is None:
            lower_failed = name
    acc = {}
    for key in ('mtp1', 'mtp5'):
        if key in runs:
            acc[key] = acceptance(fixture, runs[key])
    g2 = next((g for g in gates if g['name'] == 'G2'), None)
    if g2 is None:
        verdict = 'INCOMPLETE (G2, MTP=0 vs MTP=5, was not given)'
    elif lower_failed and lower_failed != 'G2':
        verdict = 'DIFF at %s (G2 meaningless)' % lower_failed
    elif not g2['equal']:
        verdict = 'DIFF at G2 (%d of %d prompts)' % (sum(not r['equal'] for r in g2['prompts']), len(g2['prompts']))
    else:
        verdict = 'EQUAL'
    if diag and verdict == 'EQUAL':
        verdict = 'EQUAL (diag: engine consistency only)'
    return {'label': 'diag' if diag else 'real', 'verdict': verdict, 'gates': gates, 'acceptance': acc,
            'pass': g2 is not None and verdict.startswith('EQUAL')}


def _fmt(x, digits=2):
    return '-' if x is None else '%.*f' % (digits, x)


def format_report(res):
    L = ['DSpark canary, label: %s%s' % (res['label'], ' (engram off: engine self-consistency only, not quality)' if res['label'] == 'diag' else '')]
    for g in res['gates']:
        n_eq = sum(r['equal'] for r in g['prompts'])
        head = '%s %s: %s (%d/%d prompts equal)' % (g['name'], g['label'], 'EQUAL' if g['equal'] else 'DIFF', n_eq, len(g['prompts']))
        if g['meaningless']:
            head += ' [meaningless: %s failed]' % g['after']
        L.append(head)
        for r in g['prompts']:
            if r['equal']:
                L.append('  %-9s EQUAL md5 %s len %d' % (r['id'], r['md5_a'], r['len_a']))
            else:
                d = r['div']
                gap = '' if d['top2_gap'] is None else ', top2 gap %.4g' % d['top2_gap']
                L.append('  %-9s DIFF  md5 %s vs %s, first divergence at token %d: %s vs %s%s (len %d vs %d)'
                         % (r['id'], r['md5_a'], r['md5_b'], d['i'], d['tok_a'], d['tok_b'], gap, r['len_a'], r['len_b']))
    for key, label in (('mtp1', 'MTP=1'), ('mtp5', 'MTP=5')):
        if key not in res['acceptance']:
            continue
        L.append('acceptance, %s run%s:' % (label, ' (diag: not valid for the shipped model)' if res['label'] == 'diag' else ''))
        L.append('  %-8s %6s %7s %10s  %-25s %s' % ('class', 'rounds', 'mean_a', 'tok/round', 'hist a=0..5', 'P(draft_r ok | 0..r-1 ok), r=0..4'))
        for c, s in res['acceptance'][key].items():
            L.append('  %-8s %6d %7s %10s  %-25s %s' % (c, s['rounds'], _fmt(s['mean_a']), _fmt(s['tokens_per_round']),
                                                         '/'.join(str(h) for h in s['hist']), ' '.join(_fmt(p) for p in s['per_pos'])))
    L.append('VERDICT: %s' % res['verdict'])
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fixture', default=str(HERE / 'ds41_dspark_canary.json'))
    ap.add_argument('--mtp0', required=True, help='baseline run, GR8_DS_MTP=0')
    ap.add_argument('--mtp0b', help='second MTP=0 run (G0, determinism)')
    ap.add_argument('--mtp1', help='MTP=1 run (G1)')
    ap.add_argument('--mtp5', help='MTP=5 run (G2, the done criterion)')
    ap.add_argument('--json', help='also write the result as JSON here')
    args = ap.parse_args(argv)
    try:
        fx = load_fixture(args.fixture)
        runs = {'mtp0': load_run(args.mtp0, fx, 0)}
        for key, mtp in (('mtp0b', 0), ('mtp1', 1), ('mtp5', 5)):
            path = getattr(args, key)
            if path:
                runs[key] = load_run(path, fx, mtp)
        res = build_report(fx, runs)
    except InputError as e:
        print('ds41_dspark_canary: bad input: %s' % e, file=sys.stderr)
        return 2
    print(format_report(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + '\n')
    return 0 if res['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
