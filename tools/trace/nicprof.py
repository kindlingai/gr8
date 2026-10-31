#!/usr/bin/env python3
"""Sample every ConnectX RDMA port of each rank's machine during a run: Chrome-trace counter tracks and a summary.

    tools/trace/nicprof.py --out /tmp/nic 0=HOST0 1=HOST1 2=HOST2 3=HOST3 --seconds 60

Each machine is read over ssh: every --interval seconds its /sys/class/infiniband/*/ports/1 counters, stamped with that
machine's wall clock in microseconds (the clock GR8_S2_TRACE uses). PATH.nic.r<rank>.json then holds the rank's
counter tracks: send and receive rate per port in Gb/s, and the per-interval increase of the counters that show trouble
(retransmits, sequence errors, congestion notifications, ECN marks, transmit waits). tools/trace/merge.py joins them
with the ranks' S2 traces (same pid, same --clock offsets). The summary prints, per rank and port, the mean and peak
rates while the port moved data and the totals of the trouble counters over the window.
"""

import argparse
import json
import shlex
import subprocess
import sys
import threading

# Counters read per port: (directory, name). port_*_data count units of 4 bytes.
RATE = [('counters', 'port_xmit_data'), ('counters', 'port_rcv_data')]
TROUBLE = [('hw_counters', n) for n in ('roce_adp_retrans', 'roce_adp_retrans_to', 'out_of_sequence', 'packet_seq_err',
                                         'local_ack_timeout_err', 'np_cnp_sent', 'rp_cnp_handled',
                                         'np_ecn_marked_roce_packets', 'out_of_buffer')] + [('counters', 'port_xmit_wait')]
COUNTERS = RATE + TROUBLE

# Runs on the machine: one JSON line per sample, {"ts": us, "ports": {dev: [counter values in COUNTERS order]}}, until
# stdin closes or the deadline passes. Missing counters read as -1.
SAMPLER = r'''
import json, os, select, sys, time
C = %s
root = "/sys/class/infiniband"
devs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "ports", "1")))
def read(d, sub, n):
    try:
        with open(os.path.join(root, d, "ports", "1", sub, n)) as f: return int(f.read())
    except (OSError, ValueError): return -1
end, step = time.time() + %f, %f
while time.time() < end:
    t = time.time()
    print(json.dumps({"ts": int(t * 1e6), "ports": {d: [read(d, s, n) for s, n in C] for d in devs}}), flush=True)
    if select.select([sys.stdin], [], [], max(0.0, step - (time.time() - t)))[0] and not sys.stdin.read(1): break
'''


def sample(host, seconds, interval, out):
    """Collect the sampler's lines from `host` into `out` (a list)."""
    code = SAMPLER % (json.dumps(COUNTERS), seconds, interval)
    p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', host, 'python3 -c ' + shlex.quote(code)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.strip():
            out.append(json.loads(line))
    p.wait()


def tracks(rank, samples):
    """Chrome-trace counter events of one rank from its samples: rates in Gb/s and trouble increases per interval."""
    events = []
    for a, b in zip(samples, samples[1:]):
        dt = (b['ts'] - a['ts']) / 1e6
        if dt <= 0:
            continue
        for dev, vb in b['ports'].items():
            va = a['ports'].get(dev)
            if not va:
                continue
            d = [y - x if x >= 0 and y >= 0 else 0 for x, y in zip(va, vb)]
            gbps = [d[i] * 4 * 8 / dt / 1e9 for i in range(len(RATE))]
            events.append({'name': f'{dev} Gb/s', 'ph': 'C', 'pid': rank, 'ts': b['ts'], 'args': {'tx': round(gbps[0], 3), 'rx': round(gbps[1], 3)}})
            bad = {n: d[len(RATE) + i] for i, (_, n) in enumerate(TROUBLE) if d[len(RATE) + i]}
            if bad:
                events.append({'name': f'{dev} trouble', 'ph': 'C', 'pid': rank, 'ts': b['ts'], 'args': bad})
    return events


def summary(rank, samples):
    """Lines of the rank's summary: per port that moved data, mean / peak tx and rx over the intervals it moved data,
    and the trouble counters' totals over the window."""
    lines = []
    if len(samples) < 2:
        return [f'rank {rank}: fewer than two samples']
    first, last = samples[0], samples[-1]
    for dev in sorted(last['ports']):
        rates = []
        for a, b in zip(samples, samples[1:]):
            dt = (b['ts'] - a['ts']) / 1e6
            va, vb = a['ports'].get(dev), b['ports'][dev]
            if va and dt > 0:
                rates.append([(vb[i] - va[i]) * 32 / dt / 1e9 for i in range(len(RATE))])
        busy = [r for r in rates if r[0] + r[1] > 0.01]
        if not busy:
            continue
        mean = [sum(r[i] for r in busy) / len(busy) for i in range(2)]
        peak = [max(r[i] for r in busy) for i in range(2)]
        tot = {n: last['ports'][dev][len(RATE) + i] - first['ports'][dev][len(RATE) + i] for i, (_, n) in enumerate(TROUBLE)
               if last['ports'][dev][len(RATE) + i] >= 0}
        bad = ' '.join(f'{n} {v}' for n, v in tot.items() if v)
        lines.append(f'rank {rank} {dev}: tx mean {mean[0]:.1f} peak {peak[0]:.1f} Gb/s, rx mean {mean[1]:.1f} peak {peak[1]:.1f} Gb/s '
                     f'over {len(busy)} busy of {len(rates)} intervals' + (f'; {bad}' if bad else ''))
    return lines or [f'rank {rank}: no port moved data']


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('ranks', nargs='+', metavar='RANK=HOST', help='each rank and the machine it runs on')
    p.add_argument('--out', required=True, help='writes PATH.nic.r<rank>.json')
    p.add_argument('--seconds', type=float, default=60.0, help='window length (default 60)')
    p.add_argument('--interval', type=float, default=0.1, help='seconds between samples (default 0.1)')
    a = p.parse_args(argv)
    hosts = {}
    for spec in a.ranks:
        r, _, h = spec.partition('=')
        if not r.isdigit() or not h:
            sys.exit(f'nicprof.py: {spec!r} is not RANK=HOST')
        hosts[int(r)] = h
    got = {r: [] for r in hosts}
    threads = [threading.Thread(target=sample, args=(h, a.seconds, a.interval, got[r])) for r, h in hosts.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in sorted(hosts):
        with open(f'{a.out}.nic.r{r}.json', 'w') as f:
            json.dump(tracks(r, got[r]), f)
        print('\n'.join(summary(r, got[r])))


if __name__ == '__main__':
    main(sys.argv[1:])
