#!/usr/bin/env python3
"""Sum up the GR8_DS_SPAN lines of DS41 engine logs.

    tools/trace/ds_span.py [--layers 0-7] LOG...

Each "# span ... rank R: unit t ms ..." line is one print: an eager pass ("# span step S") or the mean per step of a
captured graph's sampled replays ("# span M=6 per step over 16 sampled replays"). The table has one row per (print kind,
rank) with the mean over its prints of every unit's time and their sum. With GR8_DS_SPAN=2 the per-layer lines follow,
and --layers A-B splits each unit's per-layer sum into the layers A..B and the rest (the trellis layers of a release
against its native ones); without it the per-layer lines are ignored.
"""
import argparse
import re
import sys
from collections import defaultdict

HEAD = re.compile(r'^# span (.*?)(?: per step over \d+ sampled replays,)? rank (\d+)(?: per layer (\w+)| (ops)| layer (-?\d+) (ops))?:(.*)$')


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('logs', nargs='+')
    p.add_argument('--layers', metavar='A-B', help='the layers to sum apart in the per-layer lines (A..B inclusive)')
    a = p.parse_args(argv)
    lo = hi = None
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split('-'))
    tot = defaultdict(lambda: defaultdict(float))     # (kind, rank) -> unit -> ms summed over prints
    cnt = defaultdict(int)                            # (kind, rank) -> prints
    split = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))   # (kind, rank) -> unit -> [in A..B, other, prints]
    optot = defaultdict(lambda: defaultdict(float))   # (kind, rank) -> op label -> ms summed over prints
    opsplit = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))   # (kind, rank) -> op label -> [in A..B, other]
    rows = defaultdict(list)                          # (kind, rank) -> the rows of each prefill pass
    for path in a.logs:
        with open(path, errors='replace') as f:
            for line in f:
                m = HEAD.match(line.rstrip('\n'))
                if not m:
                    continue
                kind, rank, unit, ops, layer, lops, rest = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
                kind = re.sub(r'(step|pass) \d+', r'eager \1', kind)
                key = (kind, rank)
                if ops:                                   # the marks summed by label over the pass / step
                    for name, ms in re.findall(r' (\w+) ([\d.]+)', rest):
                        optot[key][name] += float(ms)
                elif lops:                                # one layer's marks
                    if lo is not None and int(layer) >= 0:
                        inside = lo <= int(layer) <= hi
                        for name, ms in re.findall(r' (\w+) ([\d.]+)', rest):
                            opsplit[key][name][0 if inside else 1] += float(ms)
                elif unit is None:
                    cnt[key] += 1
                    m2 = re.search(r' \((\d+) rows\)$', rest)
                    if m2:
                        rows[key].append(int(m2.group(1)))
                    for name, ms in re.findall(r' (\w+) ([\d.]+) ms', rest):
                        tot[key][name] += float(ms)
                elif lo is not None:
                    vals = [float(x) for x in rest.split()]
                    s = split[key][unit]
                    s[0] += sum(v for i, v in enumerate(vals) if lo <= i <= hi)
                    s[1] += sum(v for i, v in enumerate(vals) if not lo <= i <= hi)
                    s[2] += 1
    for key in sorted(tot, key=lambda k: (k[1], k[0])):
        kind, rank = key
        n = cnt[key]
        units = tot[key]
        total = sum(units.values()) / n
        what = f'{sum(rows[key]) / n:.0f} rows a pass, ' if rows[key] else ''
        print(f'{kind} rank {rank}: {n} prints, {what}{total:.3f} ms per step:',
              ' '.join(f'{u} {ms / n:.3f}' for u, ms in units.items()))
        if key in split:
            for unit, (inside, other, k) in split[key].items():
                print(f'    {unit}: layers {lo}-{hi} {inside / k:.3f} ms, the rest {other / k:.3f} ms')
        if key in optot:
            print('    ops:', ' '.join(f'{name} {ms / n:.3f}' for name, ms in optot[key].items()))
        if key in opsplit:
            print(f'    ops, layers {lo}-{hi}:', ' '.join(f'{name} {v[0] / n:.3f}' for name, v in opsplit[key].items() if v[0]))
            print(f'    ops, the rest:', ' '.join(f'{name} {v[1] / n:.3f}' for name, v in opsplit[key].items() if v[1]))


if __name__ == '__main__':
    main(sys.argv[1:])
