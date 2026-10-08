"""Synthetic full + sparse packs for engine/tests/test_sources.cpp: one v2 layer (v2synth, 8 routed experts + the
trellis shared expert, mixed widths) converted twice for TP4, without and with refs.

python3 -m tools.convert.test_sources OUT     writes OUT/v2 (the v2 files), OUT/full, OUT/sparse
"""
import os
import sys

import numpy as np

from tools.convert import v2, v2synth
from tools.convert.pack import PackWriter

NE, L, NR = 8, 41, 4


def main(out):
    v2.N_EXP = NE
    src = os.path.join(out, 'v2')
    v2synth.make_layer(src, L, (2, 3, 4), NE, seed=3)
    rng = np.random.default_rng(1)
    wd = v2.Widths(rng.integers(2, 5, NE), rng.integers(2, 5, (NE, v2.NH)), rng.integers(2, 5, (NE, v2.NH)),
                   shared=dict(gate=rng.integers(3, 5, 16).tolist(), up=[4] * 16, down=4), name='mixed')
    for name, refs in (('full', None), ('sparse', (os.path.abspath(src), os.path.abspath(src)))):
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        ws = {r: PackWriter(os.path.join(d, f'rank{r}.pack'), r, False, {}, refs=refs) for r in range(NR)}
        lay = v2.V2Layer(src, L)
        v2.convert_experts_v2(L, lay, wd, ws, NR, refs=refs is not None)
        lay.close()
        for w in ws.values():
            w.close()
    print(f'{out}: full and sparse TP{NR} packs of layer {L}')


if __name__ == '__main__':
    main(sys.argv[1])
