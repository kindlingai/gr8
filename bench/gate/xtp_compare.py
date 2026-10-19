#!/usr/bin/env python3
"""Cross-topology KL check: score a KL reference taken on ONE topology (gate.py kl-ref, e.g. TP4) on a build of ANOTHER
topology (e.g. TP5), and pass or fail it against a stated tolerance.

  xtp_compare.py --url U --ref refs/tp4-<commit>.json --topology tp5 --commit SHA [--out FILE]
                 [--max-mean 0.01] [--max-p95 0.05] [--min-top1 0.97]

gate.py run refuses a reference of another topology (a TP5 run is never recorded as TP4), and its records feed baselines;
this tool writes no run record. Both builds must serve the same weights, width manifest and steering: a different
topology changes only the reduction order, so the outputs differ by rounding, never by much.
Tolerance: different TP widths sum partial products in a different order, so bits differ and KL is not 0 (an A/A on
one topology is 0, see baselines/). The defaults are a starting bar, not a measured spread: tighten them once a
cross-topology pair of known-good builds (e.g. TP6 vs TP4 at one commit) has been measured.
Exit 0 = PASS, 1 = FAIL (or an error). Stdlib only."""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client as C  # noqa: E402
import kl  # noqa: E402


def verdict(res, max_mean, max_p95, min_top1):
    """-> (PASS|FAIL, [reasons]) for a kl.compare result."""
    why = []
    if res['mean'] > max_mean:
        why.append(f'kl mean {res["mean"]:.6f} > {max_mean}')
    if res['p95'] > max_p95:
        why.append(f'kl p95 {res["p95"]:.6f} > {max_p95}')
    if res['top1'] < min_top1:
        why.append(f'top1 {res["top1"]:.4f} < {min_top1}')
    return ('FAIL' if why else 'PASS'), why


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8100')
    ap.add_argument('--model', default=C.MODEL)
    ap.add_argument('--ref', required=True, help='gate.py kl-ref file of the reference topology')
    ap.add_argument('--topology', required=True, choices=['tp6', 'tp5', 'tp4', 'tp3'], help='topology under test')
    ap.add_argument('--commit', required=True, help='gr8 commit of the build under test')
    ap.add_argument('--max-mean', type=float, default=0.01)
    ap.add_argument('--max-p95', type=float, default=0.05)
    ap.add_argument('--min-top1', type=float, default=0.97)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    ref = json.loads(Path(a.ref).read_text())
    if 'sequences' not in ref or not ref['sequences'] or 'pos' not in ref['sequences'][0]:
        sys.exit(f'{a.ref} is not a gate.py kl-ref (no scored sequences)')
    if ref.get('topology') == a.topology:
        sys.exit(f'{a.ref} is already a {a.topology} reference: use gate.py run --kl-ref')
    res = kl.compare(C.Client(a.url, a.model), ref, ref.get('k', 20))
    v, why = verdict(res, a.max_mean, a.max_p95, a.min_top1)
    out = {'format': 1, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'verdict': v, 'reasons': why,
           'topology': a.topology, 'commit': a.commit, 'reference': {'topology': ref.get('topology'), 'commit': ref.get('commit'),
           'file': Path(a.ref).name}, 'tolerance': {'max_mean': a.max_mean, 'max_p95': a.max_p95, 'min_top1': a.min_top1},
           'kl': {k: res[k] for k in ('positions', 'sequences', 'mean', 'p95', 'max', 'top1')}, 'per_sequence': res['per_sequence']}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + '\n')
    print(f'{v}: {a.topology} {a.commit} vs {ref.get("topology")} {ref.get("commit")}: kl mean {res["mean"]:.6f} '
          f'p95 {res["p95"]:.6f} max {res["max"]:.6f} top1 {res["top1"]:.4f} over {res["positions"]} positions'
          + (f' ({"; ".join(why)})' if why else ''))
    return 0 if v == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
