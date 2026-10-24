"""Run records (one JSON line per run), spread between runs, baseline files and the check against a baseline."""
import json
import math
import statistics

KL_FLOOR = 1e-6  # a threshold never below this: boots that reproduce the reference exactly would otherwise fail on rounding
HIGHER = 'higher'
LOWER = 'lower'
INFO = 'info'  # reported, never gated

# The full gate set (default sizes). A check is PASS only when the baseline and the run both have all of it.
REQUIRED = ['kl.mean', 'needle.87k.pass', 'needle.437k.pass',
            'long.18k.ttft_s', 'long.18k.prefill_tok_s', 'long.87k.ttft_s', 'long.87k.prefill_tok_s',
            'pbench.prefill_8192', 'pbench.prefill_32768', 'pbench.prose_decode',
            'work.code.decode_tok_s', 'work.prose.decode_tok_s', 'work.structured.decode_tok_s',
            'work.mixed4.agg_tok_s', 'work.mixed4.stream_tok_s', 'spec.acceptance', 'spec.tokens_per_step']
# Settings that change what a number means: baseline and checked runs must agree on them.
MEASURE_KEYS = ['effort', 'temp', 'needle_sizes', 'needle_depth', 'needle_max_tokens', 'long_sizes', 'pbench_reps',
                'pbench_args', 'runs_per_workload', 'workload_tokens']


def flatten(results):
    """Detailed results -> {metric name: (value, better direction)}: the numbers that gates compare."""
    m = {}

    def put(name, v, d):
        if v is not None:
            m[name] = (float(v), d)
    kl = results.get('kl')
    if kl:
        put('kl.mean', kl['mean'], LOWER)
        put('kl.p95', kl['p95'], LOWER)
        put('kl.top1', kl['top1'], HIGHER)
    for size, r in (results.get('needle') or {}).items():
        put(f'needle.{size}.pass', 1.0 if r['passed'] else 0.0, HIGHER)
    for size, r in (results.get('long') or {}).items():
        put(f'long.{size}.ttft_s', r['ttft_s'], LOWER)
        put(f'long.{size}.prefill_tok_s', r['prefill_tok_s'], HIGHER)
        put(f'long.{size}.decode_tok_s', r['decode_tok_s'], INFO)  # one request's decode varies with its text
    pb = results.get('pbench') or {}
    for k in ('prefill_8192', 'prefill_32768', 'prose_decode'):
        put(f'pbench.{k}', pb.get(k), HIGHER)
    for kind, r in (results.get('workloads') or {}).items():
        put(f'work.{kind}.decode_tok_s', r['decode_tok_s'], HIGHER)
    mx = results.get('mixed4')
    if mx:
        put('work.mixed4.agg_tok_s', mx['agg_tok_s'], HIGHER)
        put('work.mixed4.stream_tok_s', mx['stream_tok_s'], HIGHER)
    ps = results.get('prompt_set')
    if ps:  # another bench's prompts: the set name is in every metric name
        for kind, r in ps['kinds'].items():
            put(f'set.{ps["name"]}.{kind}.decode_tok_s', r['decode_tok_s'], HIGHER)
            put(f'set.{ps["name"]}.{kind}.tokens_per_step', (r['spec'] or {}).get('tokens_per_step'), HIGHER)
    sp = results.get('spec')
    if sp:
        put('spec.acceptance', sp['acceptance'], HIGHER)
        put('spec.tokens_per_step', sp['tokens_per_step'], HIGHER)
    return m


def read_runs(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def append_run(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def select(runs, session=None):
    """The records of one session (default: the last one in the file)."""
    if session is None:
        session = runs[-1]['session']
    return [r for r in runs if r['session'] == session]


def spread(records):
    """Spread between runs against one boot: {metric: {values, min, max, spread_pct}} (spread = (max-min)/mean)."""
    out = {}
    names = sorted({k for r in records for k in r['metrics']})
    for k in names:
        vals = [r['metrics'][k]['value'] for r in records if k in r['metrics']]
        mean = statistics.fmean(vals)
        out[k] = {'values': vals, 'min': min(vals), 'max': max(vals),
                  'spread_pct': round(100 * (max(vals) - min(vals)) / abs(mean), 2) if mean else 0.0}
    return out


def spread_table(records):
    sp = spread(records)
    rows = ['| metric | ' + ' | '.join(f'run {i + 1}' for i in range(len(records))) + ' | spread % |',
            '|---|' + '---|' * (len(records) + 1)]
    for k, v in sp.items():
        rows.append(f'| {k} | ' + ' | '.join(f'{x:g}' for x in v['values']) + f' | {v["spread_pct"]} |')
    return '\n'.join(rows)


def norm_env(s):
    """Engine environment as a canonical string (order of NAME=value words does not matter)."""
    return ' '.join(sorted((s or '').split()))


def _same(runs, key, what):
    vals = {json.dumps(key(r), sort_keys=True) for r in runs}
    if len(vals) > 1:
        raise ValueError(f'runs differ in {what}: {sorted(vals)}')
    return json.loads(vals.pop())


def ineligible(rec):
    """Why a run record cannot count for a baseline or a check, or None. A run needs the rank health probe result of its boot,
    admitted: a run without one, or on ranks the probe refused (`run --allow-unhealthy`), says nothing about the build."""
    h = rec.get('rank_health')
    if not h:
        return 'it carries no rank health probe result (run with --health FILE, from scripts/rank_health.sh)'
    if not h.get('admitted', False):
        why = '; '.join('rank {} {}'.format(b['rank'], b['code']) for b in h.get('refusals', [])) or 'no reason recorded'
        return f'its rank health probe refused a rank ({why})'
    return None


def health_summary(runs):
    """Per boot, the time of the admitted rank health probe its runs carry (the baseline records it)."""
    return {r['boot']: r['rank_health'].get('ts') for r in runs}


def build_baseline(runs, topology, commit, kl_margin=2.0):
    """Baseline for a topology from the run records of several boots of one commit (records carry a `boot` label;
    the repeats of one boot are reduced to their median). Only runs of `commit` and `topology` are used. They must
    agree on build flags, measurement settings and KL reference, and every run must have the same metrics (a run
    that lost a metric, e.g. a failed stream, is an error, not a smaller sample). The KL threshold stays null until
    three boots have been measured."""
    runs = [r for r in runs if r['topology'] == topology and r['commit'] == commit]
    if not runs:
        raise ValueError(f'no runs of commit {commit} on {topology}')
    for r in runs:
        why = ineligible(r)
        if why:
            raise ValueError(f'boot {r["boot"]!r} cannot go into a baseline: {why}')
    flags = _same(runs, lambda r: r['build_flags'], 'build flags')
    env = _same(runs, lambda r: norm_env(r.get('engine_env')), 'engine environment')
    settings = _same(runs, lambda r: {k: r['settings'].get(k) for k in MEASURE_KEYS}, 'measurement settings')
    kl_ref = _same(runs, lambda r: {'commit': (r['results'].get('kl') or {}).get('reference_commit'),
                                    'topology': (r['results'].get('kl') or {}).get('reference_topology')},
                   'KL reference')
    names = {frozenset(r['metrics']) for r in runs}
    if len(names) > 1:
        raise ValueError('runs do not all have the same metrics; differing: '
                         f'{sorted(frozenset.union(*names) - frozenset.intersection(*names))}')
    if kl_ref['topology'] not in (None, topology):
        raise ValueError(f'KL reference is for {kl_ref["topology"]}, not {topology}')
    boots = {}
    for r in runs:
        boots.setdefault(r['boot'], []).append(r)
    per_boot = {b: {k: statistics.median(x['metrics'][k]['value'] for x in rs) for k in rs[0]['metrics']}
                for b, rs in boots.items()}
    direction = {k: v['better'] for k, v in runs[0]['metrics'].items()}
    metrics = {}
    for k in sorted(direction):
        vals = [pb[k] for pb in per_boot.values()]
        mean = statistics.fmean(vals)
        metrics[k] = {'better': direction[k], 'median': statistics.median(vals), 'min': min(vals), 'max': max(vals),
                      'spread_pct': round(100 * (max(vals) - min(vals)) / abs(mean), 2) if mean else 0.0,
                      'boots': len(vals)}
    kl_thr = None
    if len(boots) >= 3 and 'kl.mean' in metrics:
        kl_thr = max(metrics['kl.mean']['max'] * kl_margin, KL_FLOOR)
    return {'format': 1, 'topology': topology, 'commit': commit, 'build_flags': flags, 'engine_env': env, 'settings': settings,
            'kl_reference': kl_ref if kl_ref['commit'] else None,
            'missing_from_gate_set': [k for k in REQUIRED if k not in metrics],
            'rank_health': health_summary(runs),
            'boots': {b: {'runs': len(rs), 'metrics': per_boot[b]} for b, rs in boots.items()},
            'metrics': metrics,
            'thresholds': {'kl_mean': kl_thr, 'kl_margin': kl_margin,
                           'note': 'kl_mean is null until three boots are measured'}}


def validate(records, baseline):
    """Refuse to compare runs with a baseline they do not belong to."""
    for r in records:
        why = ineligible(r)
        if why:
            raise ValueError(f'boot {r["boot"]!r} cannot be checked: {why}')
        if r['topology'] != baseline['topology']:
            raise ValueError(f'run is {r["topology"]}, baseline is {baseline["topology"]}')
        if norm_env(r.get('engine_env')) != baseline.get('engine_env', ''):
            raise ValueError(f'engine environment {norm_env(r.get("engine_env"))!r} differs from the baseline\'s '
                             f'{baseline.get("engine_env", "")!r}')
        got = {k: r['settings'].get(k) for k in MEASURE_KEYS}
        if got != baseline['settings']:
            diff = sorted(k for k in MEASURE_KEYS if got[k] != baseline['settings'].get(k))
            raise ValueError(f'measurement settings differ from the baseline: {diff}')
        kl = r['results'].get('kl')
        ref = baseline.get('kl_reference')
        if kl and ref and (kl.get('reference_commit'), kl.get('reference_topology')) != (ref['commit'], ref['topology']):
            raise ValueError(f'KL reference ({kl.get("reference_commit")}, {kl.get("reference_topology")}) is not the '
                             f'baseline\'s ({ref["commit"]}, {ref["topology"]})')
        if kl and not ref:
            raise ValueError('run has KL numbers but the baseline was built without a KL reference')


def check(records, baseline, noise=0.03, required=REQUIRED):
    """Compare one session's records (median of its runs) with a baseline. -> (rows, status, reasons).
    status: FAIL (a gate failed or a metric is missing), PARTIAL (nothing failed, but part of the full gate set
    is absent from the baseline or has no KL threshold), PASS (the full gate set was checked and passed).
    Rules: needle passes must be 1; KL mean at or below the threshold; every other gated metric no worse than the
    baseline median by more than max(noise, the baseline's boot-to-boot spread). Raises ValueError when the runs do
    not belong to the baseline (topology, settings, KL reference)."""
    validate(records, baseline)
    cur = {k: statistics.median(r['metrics'][k]['value'] for r in records)
           for k in records[0]['metrics'] if all(k in r['metrics'] for r in records)}
    rows, failed, reasons = [], False, []
    for k in sorted(set(baseline['metrics']) | set(required or [])):
        b = baseline['metrics'].get(k)
        if b is None:
            rows.append((k, None, cur.get(k), None, 'NOT IN BASELINE'))
            reasons.append(f'{k}: not in the baseline')
            continue
        if k not in cur:
            if b['better'] == INFO:
                rows.append((k, b['median'], None, None, 'info'))
                continue
            rows.append((k, b['median'], None, None, 'MISSING'))
            failed = True
            continue
        v, base = cur[k], b['median']
        delta = 100 * (v - base) / abs(base) if base else None
        tol = max(noise, b['spread_pct'] / 100)
        if b['better'] == INFO or k in ('kl.p95', 'kl.top1'):
            verdict = 'info'
        elif k.endswith('.pass'):
            verdict = 'PASS' if v >= 1 else 'FAIL'
        elif k == 'kl.mean':
            thr = baseline['thresholds']['kl_mean']
            verdict = 'NO THRESHOLD' if thr is None else ('PASS' if v <= thr else 'FAIL')
            if thr is None:
                reasons.append('kl.mean: no threshold yet (needs three boots in the baseline)')
        else:
            worse = (base - v) / abs(base) if b['better'] == HIGHER else (v - base) / abs(base)
            verdict = 'FAIL' if worse > tol else ('better' if worse < -tol else 'PASS')
        failed |= verdict == 'FAIL'
        rows.append((k, base, v, delta, verdict))
    status = 'FAIL' if failed else ('PARTIAL' if reasons else 'PASS')
    return rows, status, reasons


def gates_table(rows):
    out = ['| metric | main (baseline) | this build | delta % | gate |', '|---|---|---|---|---|']
    for k, base, v, d, verdict in rows:
        out.append(f'| {k} | {"-" if base is None else format(base, "g")} | {"-" if v is None else format(v, "g")} | '
                   f'{"-" if d is None else format(d, "+.1f")} | {verdict} |')
    return '\n'.join(out)


# Two-sided 95% Student t quantiles by degrees of freedom (stdlib has no t distribution); past 30, the normal 1.96.
T975 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    return T975[df] if df < len(T975) else 1.96


def paired(runs, topology, a, b, noise=0.03):
    """Paired A/B of two commits (#32). Each pair is a `pair` label shared by one boot of `a` and one of `b`, measured
    back to back (a then b, or b then a), so slow drift between pairs (clock, thermals, a node that degrades) cancels in
    the difference. Per pair and metric: d = (b - a) / |a| on the medians of each side's repeats. Over the pairs: mean d,
    its 95% t interval, and how many pairs went each way. -> (rows, info). rows: (metric, better, n, mean %, lo %, hi %,
    b better, a better, verdict). verdict: 'worse' when the whole interval lies on the worse side of 0 and its mean is
    past `noise`, 'better' likewise, 'within noise' when the interval sits inside +-noise, else 'unresolved' (more pairs
    would settle it). One pair gives a delta and no interval. Pairs must agree on measurement settings and topology; build
    flags and engine environment may differ between a and b (that may be what is under test), not within a side.
    KL stays info here: `check` gates kl.mean on an absolute threshold with a floor, and a relative change of numbers
    at that floor means nothing. Only pairs with both sides count; their records must pass `ineligible`."""
    runs = [r for r in runs if r['topology'] == topology and r.get('pair') is not None and r['commit'] in (a, b)]
    if a == b:
        raise ValueError('paired needs two different commits')
    if not runs:
        raise ValueError(f'no paired runs of {a} and {b} on {topology} (run --pair ID)')
    _same(runs, lambda r: {k: r['settings'].get(k) for k in MEASURE_KEYS}, 'measurement settings')
    side = {}
    for c in (a, b):
        rs = [r for r in runs if r['commit'] == c]
        side[c] = {'build_flags': _same(rs, lambda r: r['build_flags'], f'build flags of {c}') if rs else None,
                   'engine_env': _same(rs, lambda r: norm_env(r.get('engine_env')), f'engine environment of {c}')
                   if rs else None}
    by = {}
    for r in runs:
        by.setdefault(r['pair'], {}).setdefault(r['commit'], []).append(r)
    whole = {p: v for p, v in by.items() if a in v and b in v}
    lone = sorted(set(by) - set(whole))
    if not whole:
        raise ValueError(f'no pair has runs of both {a} and {b}')
    for p, v in whole.items():
        boots = {c: {r['boot'] for r in v[c]} for c in (a, b)}
        if boots[a] & boots[b]:
            raise ValueError(f'pair {p!r}: {a} and {b} share boot {sorted(boots[a] & boots[b])}; each side is its own boot')
        for c in (a, b):
            if len(boots[c]) > 1:
                raise ValueError(f'pair {p!r} has {len(boots[c])} boots of {c} ({sorted(boots[c])}); a pair is one boot '
                                 'of each side')
        for r in v[a] + v[b]:
            why = ineligible(r)
            if why:
                raise ValueError(f'boot {r["boot"]!r} cannot be compared: {why}')
    runs = [r for v in whole.values() for r in v[a] + v[b]]

    def med(rs, k):
        vals = [r['metrics'][k]['value'] for r in rs if k in r['metrics']]
        return statistics.median(vals) if vals else None
    direction = {}
    for r in runs:
        for k, m in r['metrics'].items():
            direction.setdefault(k, m['better'])
    rows = []
    for k in sorted(direction):
        ds = []
        for p in sorted(whole):
            va, vb = med(whole[p][a], k), med(whole[p][b], k)
            if va is None or vb is None or va == 0:
                continue
            ds.append((vb - va) / abs(va))
        if not ds:
            continue
        d = direction[k]
        n = len(ds)
        mean = statistics.fmean(ds)
        if n > 1:
            h = t975(n - 1) * statistics.stdev(ds) / math.sqrt(n)
            lo, hi = mean - h, mean + h
        else:
            lo = hi = None
        sign = 1 if d == HIGHER else -1 if d == LOWER else 0
        b_better = sum(1 for x in ds if sign * x > 0)
        a_better = sum(1 for x in ds if sign * x < 0)
        if d == INFO or k.endswith('.pass') or k.startswith('kl.'):
            verdict = 'info'
        elif lo is None:
            verdict = 'one pair'
        else:
            wlo, whi = sorted((sign * lo, sign * hi))  # in "better" units: > 0 is b better
            if whi < 0 and -sign * mean > noise:
                verdict = 'worse'
            elif wlo > 0 and sign * mean > noise:
                verdict = 'better'
            elif wlo >= -noise and whi <= noise:
                verdict = 'within noise'
            else:
                verdict = 'unresolved'
        pct = lambda x: None if x is None else round(100 * x, 2)
        rows.append((k, d, n, pct(mean), pct(lo), pct(hi), b_better, a_better, verdict))
    info = {'a': a, 'b': b, 'topology': topology, 'pairs': sorted(whole), 'unpaired': lone, 'sides': side,
            'noise': noise}
    return rows, info


def paired_table(rows, info):
    a, b = info['a'], info['b']
    out = [f'| metric | pairs | {b} vs {a}, mean % | 95% interval % | {b} better | {a} better | verdict |',
           '|---|---|---|---|---|---|---|']
    for k, d, n, mean, lo, hi, nb, na, v in rows:
        ci = '-' if lo is None else f'{lo:+.2f} .. {hi:+.2f}'
        out.append(f'| {k} | {n} | {mean:+.2f} | {ci} | {nb} | {na} | {v} |')
    return '\n'.join(out)
