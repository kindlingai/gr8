#!/usr/bin/env python3
"""Compare operation counts in normalized CUDA traces from matched workload arms."""
import argparse
import collections
import json
import sys

if __package__:
    from .report_common import read_rows, write_report
else:
    from report_common import read_rows, write_report

KINDS = {'kernel', 'memcpy', 'allocation', 'synchronization', 'graph'}


def arm_name(value):
    if not isinstance(value, str) or not value or value != value.strip() or ',' in value:
        raise ValueError('arm names must be nonempty strings without surrounding whitespace or commas')
    return value


def load(path):
    rows, digest = read_rows(path)
    counts, meta = collections.Counter(), {}
    for row in rows:
        arm = arm_name(row.get('arm'))
        count = row.get('count', 1)
        if type(count) is not int or count < 0:
            raise ValueError(f'{arm}: count must be a nonnegative integer')
        if row.get('kind') not in KINDS:
            raise ValueError(f'{arm}: unknown trace kind')
        for key in ('path', 'name'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f'{arm}: missing trace {key}')
        if 'baseline' in row or 'candidate' in row:
            raise ValueError('baseline and candidate are reserved report fields')
        fields = {k: v for k, v in row.items() if k not in ('arm', 'count')}
        key = json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False)
        counts[(arm, key)] += count
        meta.setdefault(arm, set()).add(key)
    return counts, meta, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline', default='Z')
    p.add_argument('--arms', default='P,R,M')
    a = p.parse_args()
    baseline = arm_name(a.baseline)
    arms = [arm_name(arm) for arm in a.arms.split(',')]
    if len(set(arms)) != len(arms) or baseline in arms:
        p.error('--arms must be unique and exclude the baseline')
    counts, meta, digest = load(a.input)
    missing = [arm for arm in [baseline, *arms] if arm not in meta]
    if missing:
        p.error('missing trace arms: ' + ', '.join(missing))
    failures = []
    for arm in arms:
        for key in sorted(meta[baseline] | meta[arm]):
            z, v = counts[(baseline, key)], counts[(arm, key)]
            if z != v:
                failures.append({'arm': arm, 'baseline': z, 'candidate': v, **json.loads(key)})
    result = {'schema': 'g6-steering-trace-audit-v1', 'input_sha256': digest,
              'baseline': baseline, 'arms': arms, 'pass': not failures, 'differences': failures}
    write_report(a.output, result)
    print('PASS' if not failures else f'FAIL: {len(failures)} trace differences')
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, OverflowError) as e:
        print(f'trace_audit: {e}', file=sys.stderr)
        raise SystemExit(2)
