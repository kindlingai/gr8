#!/usr/bin/env python3
"""gr8's benchmark and gate: speed and correctness of a serving cluster, measured the same way on every build.

  bench.py [--quick | --full] [--url U] [--hosts H1,H2,...] [--cap HOST=MHZ] [--baseline OLD.json] [--json OUT.json]
  bench.py --closed-loop [--streams N] [--warmup S] [--duration S]
  bench.py compare OLD.json NEW.json
  bench.py kl-ref --out FILE [--url U]
  bench.py kindling REPORT.json --out kindling.json --config-id ID [--label L] [--nodes N] [--default]

Levels (README.md has the table): --quick is prefill, code and prose decode, and concurrent streams; the default adds
the smoke test, the correctness checks and structured and counting decode, in about 20 minutes; --full adds more runs,
needles to 480K, the agentic checks, GSM8K and HumanEval.

The timed stages (decode, concurrency, prefill, closed loop) run uncapped, except a box with --cap, which stays at its cap. Before
each one every box is locked at --cool-mhz, and it starts once the server is idle and every box is at or under
--cool-temp or has stopped cooling. A box that reaches --guard-temp in a timed stage is held at --cool-mhz for the
rest of it, and the stage's numbers are marked. The other stages run at --cool-mhz with no wait. A service that moves the clocks
itself (--thermal-service, sparkos-thermal) is stopped for the run and started again at the end. --hosts names the boxes (ssh destinations, "local" for this one); without it nothing is cooled or sampled.

The report prints as ASCII tables and goes to --json (default /tmp/gr8bench-MODEL-LABEL-TIME.json) in full: every run,
every reply's numbers and every second of every box's clocks summed per stage. --baseline pairs this run with an
earlier report of the same --series run by run and gates the speed: see README.md."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gr8bench import client as C, kindling, kl, record, report, stages as S, thermal  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'
# the timed stages first: each starts from a cooled cluster, uncapped; the rest run capped and need no cooldown
LEVELS = {
    'quick': {'stages': ['info', 'decode', 'concurrency', 'prefill'], 'works': ['code', 'prose'], 'runs': 6,
              'warmup': 1, 'streams': [1, 2, 4, 8], 'batches': 2, 'prefill': [8192, 32768, 131072], 'reps': 1},
    'default': {'stages': ['info', 'smoke', 'decode', 'concurrency', 'prefill', 'correct'],
                'works': ['code', 'prose', 'structured', 'counting'], 'runs': 6, 'warmup': 1,
                'streams': [1, 2, 4, 8], 'batches': 2, 'prefill': [8192, 32768, 131072], 'reps': 1},
    'full': {'stages': ['info', 'smoke', 'decode', 'concurrency', 'prefill', 'correct', 'agentic', 'quality'],
             'works': ['code', 'prose', 'structured', 'counting'], 'runs': 16, 'warmup': 2,
             'streams': [1, 2, 4, 8, 16], 'batches': 4, 'prefill': [8192, 32768, 131072], 'reps': 2},
}
TIMED = {'decode', 'concurrency', 'prefill', 'closed'}


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', file=sys.stderr, flush=True)


def run(a):
    level = 'quick' if a.quick else 'full' if a.full else 'default'
    L = LEVELS[level]
    plan = a.stages.split(',') if a.stages else (['info'] if a.closed_loop else L['stages'])
    if a.closed_loop and 'closed' not in plan:
        plan = plan + ['closed']
    cl = C.Client(a.url, a.model, timeout=7200)
    ctx = S.Ctx(cl, a, log)
    caps = dict(c.split('=') for c in a.cap)
    caps = {h: int(v) for h, v in caps.items()}
    th = thermal.Thermal(a.hosts.split(',') if a.hosts else [], caps, a.cool_mhz, a.cool_temp, a.guard_temp,
                         max_wait=a.settle_max, log=log)
    base = json.loads(Path(a.baseline).read_text()) if a.baseline else None
    if base and base['meta'].get('series') != a.series:
        sys.exit(f'{a.baseline} is series {base["meta"].get("series")}, this run {a.series}: runs would not pair')
    t_start = time.time()
    label = a.label or time.strftime('%Y%m%d-%H%M%S')
    out = Path(a.json or f'/tmp/gr8bench-{cl.model}-{label}-{time.strftime("%Y%m%d-%H%M%S")}.json')
    rep = {'meta': {'model': cl.model, 'level': level, 'label': label, 'series': a.series, 'url': cl.url,
                    'hosts': [h.name for h in th.hosts], 'caps': caps, 'effort': a.effort,
                    'stage_clocks': 'timed stages uncapped' + ''.join(f', {h} at {m}' for h, m in caps.items()) +
                                    f'; the rest at {a.cool_mhz} MHz',
                    'cooldown': f'{a.cool_mhz} MHz until at most {a.cool_temp:g} C or level' if th.hosts else 'none',
                    'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'baseline': a.baseline, 'stages': plan},
           'heat': {}, 'requests': ctx.requests, 'suspect': ctx.suspect}
    if a.health:
        rep['meta']['rank_health'] = json.loads(Path(a.health).read_text())
    if a.record:
        ctx.recorder = record.Recorder(Path(a.record), record.context(cl, label, rep['meta']['hosts'], a.effort))
        rep['meta']['record'] = a.record

    def finish(*_):
        th.stop()
        rep['meta']['took'] = f'{(time.time() - t_start) / 60:.1f} min'
        rep['meta']['clock_control'] = th.clock_error or ('on' if th.hosts else 'off')
        cmp = report.compare(rep, base)
        checks = report.verdicts(rep, cmp, a.kl_max, base)
        rep['compare'], rep['checks'] = cmp, [{'check': c, 'verdict': v, 'detail': d} for c, v, d in checks]
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(report.render(rep, cmp, checks))
        print(f'JSON report: {out}')
        return 1 if rep['meta'].get('interrupted') or any(v == 'FAIL' for _, v, _ in checks) else 0
    def stopped(*_):
        rep['meta']['interrupted'] = True
        sys.exit(finish() or 1)
    signal.signal(signal.SIGTERM, stopped)

    try:
        th.start()
        if th.hosts and a.thermal_service:
            rep['meta']['services_stopped'] = th.pause_services(a.thermal_service.split(','), 4 * 3600)
        max_len = None
        for st in plan:
            cool = None
            if th.hosts and st in TIMED:
                log(f'cooling before {st}')
                cool = th.settle(lambda: (cl.running() or 0) > 0)
                th.stage_clocks()
                log(f'  settled in {cool["seconds"]:.0f} s at {cool["temps"]}')
            elif th.hosts:
                th.cool_clocks()
            log(f'stage {st}')
            t0 = time.time()
            if st == 'info':
                rep['info'] = S.info(ctx)
                max_len = rep['info']['max_model_len']
            elif st == 'smoke':
                rep['smoke'] = sm = S.smoke(ctx)
                if not sm['cases'] or sm['passed'] < sm['cases']:
                    log(f"smoke failed ({sm['passed']}/{sm['cases']}): stopping")
                    break
            elif st == 'correct':
                rep['correct'] = S.correctness(ctx, level, max_len)
            elif st == 'decode':
                rep['decode'] = S.decode(ctx, L['works'], a.runs or L['runs'], L['warmup'])
            elif st == 'concurrency':
                ns = [int(x) for x in a.streams.split(',')] if a.streams else L['streams']
                rep['concurrency'] = S.concurrency(ctx, ns, L['batches'])
            elif st == 'closed':
                ns = [int(x) for x in a.streams.split(',')] if a.streams else [16]
                rep['closed'] = S.closed_loop(ctx, ns, a.warmup, a.duration)
            elif st == 'prefill':
                sizes = [int(x) for x in a.prefill_sizes.split(',')] if a.prefill_sizes else L['prefill']
                rep['prefill'] = S.prefill(ctx, sizes, L['reps'], max_len)
            elif st == 'agentic':
                rep['agentic'] = S.agentic(ctx)
            elif st == 'quality':
                rep['quality'] = S.quality_stage(ctx)
            else:
                sys.exit(f'unknown stage {st}; stages: {", ".join(LEVELS["full"]["stages"] + ["closed"])}')
            if th.hosts and st != 'info':
                rep['heat'][st] = {'cooldown': cool, 'seconds': round(time.time() - t0, 1),
                                   'boxes': th.window_stats(t0, time.time()),
                                   'guard': [t for t in th.trips if t['time'] >= t0]}
    except KeyboardInterrupt:
        log('interrupted: reporting what ran')
        rep['meta']['interrupted'] = True
    return finish()


def kl_ref(a):
    cl = C.Client(a.url, a.model, timeout=7200)
    texts = json.loads((DATA / 'kl_prompts.json').read_text())
    ref = {'format': 1, 'model': cl.model, 'commit': a.commit, 'topology': a.topology, 'k': 20, 'tokens': 256,
           'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'sequences': kl.make_reference(cl, texts, 256, 20)}
    Path(a.out).write_text(json.dumps(ref))
    print(f'{a.out}: {len(ref["sequences"])} sequences of {cl.model}')


def compare_files(a):
    old, new = (json.loads(Path(p).read_text()) for p in (a.old, a.new))
    cmp = report.compare(new, old)
    checks = report.verdicts(new, cmp, None, old)
    print(report.render(new, cmp, checks))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8100')
    ap.add_argument('--model', help='served model name (default: the first /v1/models lists)')
    sub = ap.add_subparsers(dest='cmd')
    c = sub.add_parser('compare', help='an earlier report against a later one')
    c.add_argument('old')
    c.add_argument('new')
    k = sub.add_parser('kl-ref', help='record the KL reference of the build that serves')
    k.add_argument('--out', required=True)
    k.add_argument('--commit', default='')
    k.add_argument('--topology', default='')
    kj = sub.add_parser('kindling', help="create or update one config of a kindling.json from a report's numbers")
    kj.add_argument('report')
    kj.add_argument('--out', default='kindling.json')
    kj.add_argument('--config-id', required=True)
    kj.add_argument('--label', help='the config\'s label, e.g. "TP=1" (an existing one is kept without it)')
    kj.add_argument('--nodes', type=int)
    kj.add_argument('--default', action='store_true', help='make this config the default_config')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--quick', action='store_true', help='prefill, code and prose decode, concurrent streams')
    g.add_argument('--full', action='store_true', help='everything, with GSM8K and HumanEval')
    ap.add_argument('--stages', help='comma-separated stages to run instead of the level\'s')
    ap.add_argument('--runs', type=int, help="decode runs per workload (the level's by default)")
    ap.add_argument('--streams', help="concurrent stream counts, e.g. 1,4,8 (the level's by default)")
    ap.add_argument('--closed-loop', action='store_true',
                    help="add the closed-loop stage (with no --stages the plan is info,closed); --streams sets the "
                         "stream counts (16 by default), each run in turn")
    ap.add_argument('--warmup', type=float, default=60, help='closed loop: seconds before the measured window starts')
    ap.add_argument('--duration', type=float, default=300, help='closed loop: seconds of the measured window')
    ap.add_argument('--prefill-sizes', help="prompt sizes in tokens, e.g. 8192,32768 (the level's by default)")
    ap.add_argument('--hosts', default='', help='boxes to cool and sample: ssh destinations, or "local"')
    ap.add_argument('--cap', action='append', default=[], metavar='HOST=MHZ',
                    help='a box that stays capped during stages (repeat for more boxes)')
    ap.add_argument('--cool-mhz', type=int, default=1800)
    ap.add_argument('--cool-temp', type=float, default=55, help='a stage may start once every box is at or under this (C)')
    ap.add_argument('--guard-temp', type=float, default=90,
                    help='a box this hot in a timed stage is held at --cool-mhz for the rest of it (C)')
    ap.add_argument('--settle-max', type=int, default=300, help='longest cooldown before a stage starts anyway (s)')
    ap.add_argument('--thermal-service', default='sparkos-thermal',
                    help='services that set GPU clocks, stopped for the run where they run (comma-separated; "" for none)')
    ap.add_argument('--baseline', help='an earlier report of the same series to compare and gate against')
    ap.add_argument('--json', help='where the JSON report goes')
    ap.add_argument('--label', default='')
    ap.add_argument('--series', default='gr8bench-1', help='runs pair only within one series')
    ap.add_argument('--effort', default='low', help='reasoning_effort: low, high, max, or nothink')
    ap.add_argument('--needle-seed', type=int,
                    help="the needle stage's seed (the document and the secret; a failed needle's report has it)")
    ap.add_argument('--kl-ref', help='a KL reference (bench.py kl-ref) of this model and topology')
    ap.add_argument('--kl-max', type=float, help='largest mean KL that passes (shown, not gated, without it)')
    ap.add_argument('--record', metavar='DIR',
                    help='save every streamed delta of the concurrency and agentic stages, with arrival times, to DIR')
    ap.add_argument('--health', help="this boot's rank health probe result (scripts/rank_health.sh), kept in the report")
    ap.add_argument('--data-dir', default=str(Path.home() / '.cache' / 'gr8bench'), help='GSM8K and HumanEval data')
    a = ap.parse_args()
    if a.cmd == 'compare':
        return compare_files(a)
    if a.cmd == 'kl-ref':
        return kl_ref(a)
    if a.cmd == 'kindling':
        return kindling.run(a.report, a.out, a.config_id, a.label, a.nodes, a.default)
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
