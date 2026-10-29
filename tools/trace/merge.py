#!/usr/bin/env python3
"""Joins S2 traces (GR8_S2_TRACE=PATH writes PATH.r<rank>.json on each rank) and a client's request times into one
Chrome trace, for chrome://tracing or ui.perfetto.dev.

    merge.py TRACE.r0.json [TRACE.r1.json ...] [--requests FILE.jsonl] -o OUT.json

A requests file holds one JSON object per request: send, first and end in wall-clock seconds, and optionally agent,
turn, prompt and gen. Each agent gets a lane in a "client" process: "waiting" from send to the first token, then
"streaming" to the end. The engine's spans and the client's share one clock, so a slow first token lines up with what
the steps were doing."""
import argparse
import json
import sys

CLIENT_PID = 1000


def rank_events(path):
    """The events of one rank's trace. The file's array is never closed (trace.h), so this closes it."""
    text = open(path).read().rstrip().rstrip(',')
    if not text.endswith(']'):
        text += ']'
    return json.loads(text)


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
    p.add_argument('-o', '--out', required=True)
    a = p.parse_args(argv)
    events = [e for t in a.traces for e in rank_events(t)]
    if a.requests:
        events += client_events(a.requests)
    with open(a.out, 'w') as f:
        json.dump({'traceEvents': events, 'displayTimeUnit': 'ms'}, f)
    print(f'{a.out}: {len(events)} events from {len(a.traces)} rank traces'
          + (f' and {a.requests}' if a.requests else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
