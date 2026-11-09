#!/usr/bin/env python3
"""Joins S2 traces (GR8_S2_TRACE=PATH writes PATH.r<rank>.json on each rank) and a client's request times into one
Chrome trace, for chrome://tracing or ui.perfetto.dev.

    merge.py TRACE.r0.json [TRACE.r1.json ...] [--requests FILE.jsonl] [--clock RANK=BEFORE.json[,AFTER.json] ...]
             -o OUT.json

A requests file holds one JSON object per request: send, first and end in wall-clock seconds, and optionally agent,
turn, prompt and gen. Each agent gets a lane in a "client" process: "waiting" from send to the first token, then
"streaming" to the end. The engine's spans and the client's share one clock, so a slow first token lines up with what
the steps were doing.

Each machine stamps with its own wall clock. A rank on another machine is put on the client's clock with --clock:
clock_probe.py run from the client's machine before and after the run measures that rank's host; merge.py subtracts
the mean offset from the rank's timestamps and refuses a probe whose best round trip or before/after drift exceeds
--max-rtt-ms / --max-drift-ms. Without --clock the timestamps are joined as written."""
import argparse
import json
import sys
from fractions import Fraction

CLIENT_PID = 1000


def rank_events(path):
    """The events of one rank's trace. The file's array is never closed (trace.h), so this closes it."""
    text = open(path).read().rstrip().rstrip(',')
    if not text.endswith(']'):
        text += ']'
    return json.loads(text)


def clock_offset(spec, max_rtt_ms, max_drift_ms):
    """RANK=BEFORE.json[,AFTER.json] (clock_probe.py) -> (rank, offset in us, a summary). Per probe, the sample with the
    lowest round trip gives 2 x offset = 2 host - t0 - t1 in integer ns; the offset is the mean over the probes."""
    rank, sep, files = spec.partition('=')
    if not sep or not rank.isdigit() or not files:
        sys.exit(f'merge.py: --clock {spec!r} is not RANK=BEFORE.json[,AFTER.json]')
    twice, trips = [], []
    for path in files.split(','):
        with open(path) as f:
            samples = json.load(f)['samples']
        if not samples or any(len(s) != 3 or any(type(x) is not int for x in s) or s[2] < s[0] for s in samples):
            sys.exit(f'merge.py: {path}: samples must be [t0, host, t1] integer ns with t1 >= t0')
        t0, host, t1 = min(samples, key=lambda s: s[2] - s[0])
        twice.append(2 * host - t0 - t1)
        trips.append(t1 - t0)
    drift = Fraction(abs(twice[0] - twice[-1]), 2)
    line = (f'rank {rank}: offset {float(Fraction(sum(twice), 2 * len(twice))) / 1e6:.3f} ms, round trip '
            f'{"/".join(f"{t / 1e6:.3f}" for t in trips)} ms, drift {float(drift) / 1e6:.3f} ms')
    if max(trips) > Fraction(str(max_rtt_ms)) * 10**6:
        sys.exit(f'merge.py: {line}: round trip over --max-rtt-ms {max_rtt_ms}')
    if drift > Fraction(str(max_drift_ms)) * 10**6:
        sys.exit(f'merge.py: {line}: drift over --max-drift-ms {max_drift_ms}')
    return int(rank), round(Fraction(sum(twice), 2000 * len(twice))), line


def client_events(path):
    out = [{'name': 'process_name', 'ph': 'M', 'pid': CLIENT_PID, 'args': {'name': 'client'}}]
    agents = set()
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        agent = int(r.get('agent', 0))
        if agent not in agents:
            agents.add(agent)
            out.append({'name': 'thread_name', 'ph': 'M', 'pid': CLIENT_PID, 'tid': agent,
                        'args': {'name': f'agent {agent}'}})
        send, first, end = (int(r[k] * 1e6) for k in ('send', 'first', 'end'))
        args = {k: r[k] for k in ('turn', 'prompt', 'gen', 'max_tokens') if k in r}
        args['ttft_s'] = round((first - send) / 1e6, 3)
        out.append({'name': 'waiting', 'ph': 'X', 'pid': CLIENT_PID, 'tid': agent, 'ts': send, 'dur': first - send,
                    'args': args})
        out.append({'name': 'streaming', 'ph': 'X', 'pid': CLIENT_PID, 'tid': agent, 'ts': first, 'dur': end - first,
                    'args': args})
    return out


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('traces', nargs='+', help='PATH.r<rank>.json files')
    p.add_argument('--requests', help="the client's request times, one JSON object per line")
    p.add_argument('--clock', action='append', default=[], metavar='RANK=BEFORE.json[,AFTER.json]',
                   help="clock_probe.py files for a rank on another machine than the client's")
    p.add_argument('--max-rtt-ms', type=float, default=50.0)
    p.add_argument('--max-drift-ms', type=float, default=50.0)
    p.add_argument('-o', '--out', required=True)
    a = p.parse_args(argv)
    events = [e for t in a.traces for e in rank_events(t)]
    offsets = {}
    for spec in a.clock:
        rank, us, line = clock_offset(spec, a.max_rtt_ms, a.max_drift_ms)
        if rank in offsets or not any(e.get('pid') == rank for e in events):
            sys.exit(f'merge.py: --clock rank {rank} is given twice or has no trace')
        offsets[rank] = us
        print(line)
    for e in events:
        if 'ts' in e and e.get('pid') in offsets:
            e['ts'] -= offsets[e['pid']]
    if a.requests:
        events += client_events(a.requests)
    with open(a.out, 'w') as f:
        json.dump({'traceEvents': events, 'displayTimeUnit': 'ms'}, f)
    print(f'{a.out}: {len(events)} events from {len(a.traces)} rank traces'
          + (f' and {a.requests}' if a.requests else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
