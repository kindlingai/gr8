#!/usr/bin/env python3
"""gr8 gate harness: measure any build the same way and compare it with the baseline of its topology.

  gate.py kl-ref   --url U --topology tp6 --commit SHA --out refs/tp6.json   # reference outputs of a recorded build
  gate.py run      --url U --topology tp6 --commit SHA [--kl-ref refs/tp6.json] [--repeat 2] [--boot ID]
                   --health FILE [--allow-unhealthy]   # FILE = the rank health probe of this boot (scripts/rank_health.sh)
  gate.py spread   [--runs runs.jsonl] [--session S]                         # spread between runs of one boot
  gate.py baseline --topology tp6 --commit SHA --out baselines/tp6.json [--runs runs.jsonl]  # runs of SHA, several boots
  gate.py check    --baseline baselines/tp6.json [--runs runs.jsonl] [--session S]   # gates table, exit 1 on FAIL

Each run appends one JSON line to --runs (default runs.jsonl next to this file). Stdlib only."""
import argparse
import calendar
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client as C  # noqa: E402
import kl  # noqa: E402
import measure  # noqa: E402
import record  # noqa: E402

PARTS = ['kl', 'needle', 'long', 'pbench', 'workloads', 'mixed4']


def git_rev():
    try:
        sha = subprocess.run(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(HERE), 'status', '--porcelain', '--', str(HERE)],
                               capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def parts_to_run(a):
    only = a.only.split(',') if a.only else PARTS
    skip = a.skip.split(',') if a.skip else []
    bad = [p for p in only + skip if p not in PARTS]
    if bad:
        sys.exit(f'unknown part(s) {bad}; choose from {PARTS}')
    return [p for p in only if p not in skip]


def one_run(cl, a, parts, seed):
    res, notes = {}, []
    notes.extend(f'{p} skipped by flag' for p in PARTS if p not in parts)
    if 'kl' in parts:
        if a.kl_ref:
            ref = json.loads(Path(a.kl_ref).read_text())
            res['kl'] = kl.compare(cl, ref, ref.get('k', 20))
        else:
            notes.append('kl skipped: no --kl-ref')
    if 'needle' in parts:
        res['needle'] = measure.needle(cl, [int(x) for x in a.needle_sizes.split(',')], seed, a.needle_depth,
                                       a.needle_max_tokens, a.effort)
    if 'long' in parts:
        res['long'] = measure.long_prompt(cl, [int(x) for x in a.long_sizes.split(',')], seed, effort=a.effort)
    if 'pbench' in parts:
        res['pbench'] = measure.pbench(a.url, [8192, 32768], a.pbench_reps, a.pbench_args.split())
    s_work = C.spec_snapshot(cl.metrics())
    if 'workloads' in parts:
        res['workloads'] = measure.workloads(cl, a.runs_per_workload, a.workload_tokens, a.temp, a.effort)
    if 'mixed4' in parts:
        res['mixed4'] = measure.mixed4(cl, a.runs_per_workload, a.workload_tokens, a.temp, a.effort)
    if 'workloads' in parts or 'mixed4' in parts:
        res['spec'] = C.spec_delta(s_work, C.spec_snapshot(cl.metrics()))
    if a.prompt_set:  # after the spec window above, so spec.* stays the workloads' own
        ps, sha = measure.load_prompt_set(a.prompt_set)
        res['prompt_set'] = measure.prompt_set(cl, ps, sha, a.prompt_set_tokens, a.prompt_set_temp)
    res['notes'] = notes
    return res


def load_health(a):
    """The rank health probe result of this boot (scripts/rank_health.sh writes it). Required: without --health the run is
    refused, unless --no-health, and then its records can not go into a baseline or a check. A result older than
    --health-max-age-min is refused (a file left from an earlier boot). A boot the probe refused is not measured unless
    --allow-unhealthy; its records then carry the refusal and no baseline or check takes them."""
    if not a.health:
        if a.no_health:
            return None
        sys.exit('run needs --health FILE (the rank health probe of this boot: scripts/rank_health.sh), or --no-health '
                 '(then no baseline or check accepts the records)')
    try:
        h = json.loads(Path(a.health).read_text())
        h['admitted'] = bool(h['admitted'])
        ts = calendar.timegm(time.strptime(h['ts'], '%Y-%m-%dT%H:%M:%SZ'))
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f'--health {a.health}: not a rank health result ({e!r})')
    age_min = (time.time() - ts) / 60
    if age_min > a.health_max_age_min:
        sys.exit(f'--health {a.health}: the probe is {age_min:.0f} min old (limit {a.health_max_age_min:g}); it may belong to an '
                 'earlier boot. Probe again before this boot (scripts/launch_tp6.sh does)')
    h['file'] = Path(a.health).name
    if not h['admitted']:
        why = '; '.join(f'rank {b["rank"]} {b["code"]}' for b in h.get('refusals', [])) or 'no reason recorded'
        if not a.allow_unhealthy:
            sys.exit(f'rank health probe refused this boot ({why}); fix the node, or --allow-unhealthy to measure anyway '
                     '(the record is then marked and no baseline or check accepts it)')
        print(f'warning: measuring a boot the rank health probe refused ({why})', file=sys.stderr)
    return h


def cmd_run(a):
    health = load_health(a)
    cl = C.Client(a.url, a.model)
    parts = parts_to_run(a)
    if a.kl_ref and 'kl' in parts:
        ref = json.loads(Path(a.kl_ref).read_text())
        if ref.get('topology') != a.topology:
            sys.exit(f'--kl-ref is a {ref.get("topology")} reference; this run is {a.topology}')
    session = a.session or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    recs = []
    for i in range(a.repeat):
        seed = a.seed if a.seed is not None else random.randrange(2 ** 32)
        seed = (seed + i * 7919) % 2 ** 32
        res = one_run(cl, a, parts, seed)
        rec = {'format': 1, 'session': session, 'repeat': i + 1, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'commit': a.commit, 'harness_commit': git_rev(), 'topology': a.topology, 'boot': a.boot or session,
               'build_flags': a.build_flags, 'engine_env': a.engine_env,
               'settings': {k: v for k, v in vars(a).items() if k not in ('func', 'cmd', 'health', 'no_health', 'health_max_age_min', 'allow_unhealthy')}
               | {'seed': seed},
               'results': res,
               'metrics': {k: {'value': v, 'better': d} for k, (v, d) in record.flatten(res).items()}}
        rec['rank_health'] = health
        record.append_run(a.runs, rec)
        recs.append(rec)
        print(f'run {i + 1}/{a.repeat} appended to {a.runs}', flush=True)
        for n in res['notes']:
            print('note:', n)
    if len(recs) > 1:
        print(record.spread_table(recs))


def cmd_klref(a):
    cl = C.Client(a.url, a.model)
    seqs = kl.make_reference(cl, kl.prompts(HERE / 'data' / 'kl_prompts.json')[:a.sequences], a.tokens, a.k)
    ref = {'format': 1, 'commit': a.commit, 'topology': a.topology, 'k': a.k, 'tokens': a.tokens,
           'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sequences': seqs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(ref, open(a.out, 'w'), separators=(',', ':'))
    print(f'wrote {a.out}: {len(seqs)} sequences, commit {a.commit}, {a.topology}')


def cmd_spread(a):
    recs = record.select(record.read_runs(a.runs), a.session)
    print(record.spread_table(recs))


def cmd_baseline(a):
    try:
        b = record.build_baseline(record.read_runs(a.runs), a.topology, a.commit, a.kl_margin)
    except ValueError as e:
        sys.exit(f'baseline refused: {e}')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(b, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out}: {len(b["boots"])} boot(s), {len(b["metrics"])} metrics, '
          f'kl threshold {b["thresholds"]["kl_mean"]}')
    if b['missing_from_gate_set']:
        print('baseline is missing parts of the gate set:', ', '.join(b['missing_from_gate_set']))


def cmd_check(a):
    recs = record.select(record.read_runs(a.runs), a.session)
    try:
        rows, status, reasons = record.check(recs, json.loads(Path(a.baseline).read_text()), a.noise)
    except ValueError as e:
        sys.exit(f'check refused: {e}')
    print(record.gates_table(rows))
    print(f'\nRESULT: {status}')
    for r in reasons:
        print('  -', r)
    if status == 'PARTIAL':
        print('  The full gate set was not checked; this is not a pass of issue #1\'s gates.')
    sys.exit(0 if status == 'PASS' or (status == 'PARTIAL' and a.allow_partial) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    runs_default = str(HERE / 'runs.jsonl')

    def common(p, build=True):
        p.add_argument('--url', default='http://127.0.0.1:8100')
        p.add_argument('--model', default=C.MODEL)
        p.add_argument('--topology', required=True, choices=['tp6', 'tp4'])
        if build:
            p.add_argument('--commit', required=True, help='gr8 commit of the build under test')

    p = sub.add_parser('kl-ref', help='write the KL reference from this build')
    common(p)
    p.add_argument('--out', required=True)
    p.add_argument('--sequences', type=int, default=16)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--k', type=int, default=20)
    p.set_defaults(func=cmd_klref)

    p = sub.add_parser('run', help='measure a build; append one record per run')
    common(p)
    p.add_argument('--build-flags', default='', help='build flags of this build, recorded as given')
    p.add_argument('--engine-env', default='', help='engine environment, recorded as given')
    p.add_argument('--boot', help='label of this boot (runs against one boot share it)')
    p.add_argument('--session')
    p.add_argument('--repeat', type=int, default=2, help='runs against this boot; the spread is reported')
    p.add_argument('--kl-ref')
    p.add_argument('--only', help=f'comma list of {PARTS}')
    p.add_argument('--skip', help='comma list, e.g. mixed4 when the build has no multi-session')
    p.add_argument('--runs', default=runs_default)
    p.add_argument('--seed', type=int)
    p.add_argument('--health', help='rank health probe result of this boot (JSON from scripts/rank_health.sh); '
                                    'it is stored in every record of the run. Required unless --no-health')
    p.add_argument('--no-health', action='store_true',
                   help='measure without a probe result; the records are marked and no baseline or check accepts them')
    p.add_argument('--health-max-age-min', type=float, default=360,
                   help='refuse a probe result older than this (a file left from an earlier boot)')
    p.add_argument('--allow-unhealthy', action='store_true',
                   help='measure a boot whose probe refused a rank (the record is marked; baseline and check refuse it)')
    p.add_argument('--effort', default='nothink', help='nothink|low|high|max')
    p.add_argument('--temp', type=float, default=0.7)
    p.add_argument('--needle-sizes', default='87000,437000')
    p.add_argument('--needle-depth', type=float, default=0.15)
    p.add_argument('--needle-max-tokens', type=int, default=400)
    p.add_argument('--long-sizes', default='18000,87000')
    p.add_argument('--pbench-reps', type=int, default=3)
    p.add_argument('--pbench-args', default='', help='extra arguments for bench/pbench.py')
    p.add_argument('--runs-per-workload', type=int, default=3)
    p.add_argument('--workload-tokens', type=int, default=300)
    p.add_argument('--prompt-set', help='JSON prompt set of another bench as token ids (see README); its numbers are '
                   'named set.<name>.<kind>.*')
    p.add_argument('--prompt-set-tokens', type=int, default=512)
    p.add_argument('--prompt-set-temp', type=float, default=0.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('spread', help='spread between the runs of a session')
    p.add_argument('--runs', default=runs_default)
    p.add_argument('--session')
    p.set_defaults(func=cmd_spread)

    p = sub.add_parser('baseline', help='baseline file from the run records of several boots')
    p.add_argument('--topology', required=True, choices=['tp6', 'tp4'])
    p.add_argument('--runs', default=runs_default)
    p.add_argument('--out', required=True)
    p.add_argument('--commit', required=True, help='only runs of this build commit go into the baseline')
    p.add_argument('--kl-margin', type=float, default=2.0, help='KL threshold = margin x worst boot (3+ boots)')
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser('check', help='gates table against a baseline; exit 1 on FAIL')
    p.add_argument('--baseline', required=True)
    p.add_argument('--runs', default=runs_default)
    p.add_argument('--session')
    p.add_argument('--allow-partial', action='store_true',
                   help='exit 0 on PARTIAL (nothing failed, part of the gate set not checked); the table still says PARTIAL')
    p.add_argument('--noise', type=float, default=0.03, help='allowed regression (fraction), at least the baseline spread')
    p.set_defaults(func=cmd_check)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
