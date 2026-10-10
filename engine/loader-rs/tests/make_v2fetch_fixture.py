#!/usr/bin/env python3
"""Writes tests/data/v2fetch/: index slices of two synthetic exlr8 v2 layers (tools/convert/v2synth.py with 2 experts,
then tools/convert/v2slices.py), a width manifest that mixes widths per unit, and the byte ranges rank 1 of TP=4 must
fetch, worked out from the layer index files with tools/convert/v2.py. The layers have no shared expert: its slice row
is 256 only when there are 256 routed experts. tests/v2fetch.rs checks src/fetch.rs against
them. Needs numpy. Run from the repo root:

    python3 engine/loader-rs/tests/make_v2fetch_fixture.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from tools.convert import v2, v2slices, v2synth  # noqa: E402

OUT = os.path.join(ROOT, 'engine', 'loader-rs', 'tests', 'data', 'v2fetch')
LAYERS, NE, TP, RANK = (3, 4), 2, 4, 1


def main():
    keep, v2.N_EXP = v2.N_EXP, NE
    tmp = tempfile.mkdtemp()
    try:
        for l in LAYERS:
            v2synth.make_layer(tmp, l, (2, 3, 4), NE, seed=l, shared=False)
        shutil.rmtree(OUT, ignore_errors=True)
        v2slices.main(['--v2', tmp, '--out', os.path.join(OUT, 'slices'), '--tp', str(TP), '--widths', '2,3,4'])
        for k in (2, 3, 4):                                  # keep only rank 1's slices: the test needs no more
            d = os.path.join(OUT, 'slices', f'tp{TP}', f'k{k}')
            for n in os.listdir(d):
                if n != f'rank{RANK}.safetensors':
                    os.unlink(os.path.join(d, n))
        man = {'format': 'exlr8-v2-manifest', 'name': 'fixture', 'layers': {}}
        for l in LAYERS:
            w = lambda e, h, p: (2, 3, 4)[(e + h + p + l) % 3]   # noqa: E731
            man['layers'][str(l)] = {'gate': [[w(e, h, 0) for h in range(16)] for e in range(NE)],
                                     'up': [[w(e, h, 1) for h in range(16)] for e in range(NE)],
                                     'down': [w(e, 0, 2) for e in range(NE)],
                                     'shared': 0}
        with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
            json.dump(man, f)
        want = {}
        for l in LAYERS:
            L = v2.V2Layer(tmp, l)
            m = man['layers'][str(l)]
            for e in range(NE):
                for h in v2.owned_halves(e, l, TP, RANK):
                    for p in v2.PROJ:
                        k = m['down'][e] if p == 'down' else m[p][e][h]
                        _, off, n = L.loc(e, p, h, k)
                        want.setdefault(v2.WIDTH_FILE.format(l=l, k=k), []).append([off, n])
            L.close()
        with open(os.path.join(OUT, 'plan.json'), 'w') as f:
            json.dump({name: sorted(r) for name, r in sorted(want.items())}, f, indent=0)
    finally:
        v2.N_EXP = keep
        shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
