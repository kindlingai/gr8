"""Desk tests for the DFlash2 converter's replicated layout (tools/convert/dflash2.py, TP=6): the small synthetic drafter
of engine/loader-rs/tests/data/dflash2/src converted at NR = 6 and at NR = 4. Every record the replicated body
(engine/runtime/dflash.cuh dfl::REP) reads must hold the TP4 ranks' records byte for byte, in the order the engine reads
them: then each GEMV output tile, and so every draft, is TP4's. No GPU. Needs numpy. Run from the repo root:

    python3 -m unittest tools.convert.test_dflash2
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tools.convert import dflash2
from tools.convert.st import ST

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'engine', 'loader-rs', 'tests', 'data', 'dflash2', 'src')
HID, HD, NH, NKV, INTER = 64, 16, 8, 4, 128       # make_dflash2_fixture.py's small drafter


def load(d, r):
    """dflash.r<r> of directory d -> (manifest, {name: (bytes, shape)})."""
    with open(os.path.join(d, f'dflash.r{r}.manifest.json')) as f:
        m = json.load(f)
    with open(os.path.join(d, f'dflash.r{r}.pack'), 'rb') as f:
        blob = f.read()
    def arr(t):
        a = np.frombuffer(blob[t['offset']:t['offset'] + t['nbytes']], np.uint8)
        return a.reshape(t['shape']) if t['dtype'] == 'u8' else a
    return m, {t['name']: (arr(t), t['shape']) for t in m['tensors']}


class Replicated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keep = dflash2.HID, dflash2.HD, dflash2.NH, dflash2.NKV, dflash2.INTER
        dflash2.HID, dflash2.HD, dflash2.NH, dflash2.NKV, dflash2.INTER = HID, HD, NH, NKV, INTER
        cls.tmp = tempfile.mkdtemp()
        with open(os.path.join(SRC, 'config.json')) as f:
            cls.cfg = json.load(f)
        st = ST(os.path.join(SRC, 'model.safetensors'))
        cls.d4, cls.d6 = os.path.join(cls.tmp, 'tp4'), os.path.join(cls.tmp, 'tp6')
        for d, nr in ((cls.d4, 4), (cls.d6, 6)):
            os.makedirs(d)
            for r in range(nr):
                dflash2.convert_rank(st, cls.cfg, nr, r, d, set(), False)
        cls.tp4 = [load(cls.d4, r)[1] for r in range(4)]
        cls.m6, cls.tp6 = load(cls.d6, 0)

    @classmethod
    def tearDownClass(cls):
        dflash2.HID, dflash2.HD, dflash2.NH, dflash2.NKV, dflash2.INTER = cls.keep
        shutil.rmtree(cls.tmp)

    def test_every_rank_has_the_same_pack(self):
        with open(os.path.join(self.d6, 'dflash.r0.pack'), 'rb') as f:
            p0 = f.read()
        for r in range(1, 6):
            m, _ = load(self.d6, r)
            with open(os.path.join(self.d6, f'dflash.r{r}.pack'), 'rb') as f:
                self.assertEqual(f.read(), p0, f'rank {r}')
            self.assertEqual(m['rank'], r)
            self.assertEqual({k: v for k, v in m.items() if k != 'rank'}, {k: v for k, v in self.m6.items() if k != 'rank'})

    def test_meta(self):
        meta = self.tp6['D.meta'][0].view(np.int32)
        self.assertEqual(int(meta[1]), 6)                 # nr: the engine checks it against its width
        self.assertEqual(self.m6['nr'], 6)
        self.assertEqual(self.m6['replicated_ks'], dflash2.REF_NR)
        self.assertNotIn('D0.o.gv', self.tp6)
        self.assertNotIn('replicated_ks', load(self.d4, 0)[0])

    def test_row_split_records_are_the_tp4_records_in_order(self):
        """fc and q: the full record is TP4's four row records concatenated (16-row tiles)."""
        for name in ['D.fc.gv'] + [f'D{j}.q.gv' for j in range(2)]:
            full = self.tp6[name][0]
            self.assertTrue(np.array_equal(full, np.concatenate([t[name][0] for t in self.tp4])), name)

    def test_kv_and_gu_hold_every_tp4_tile(self):
        """kv = [k of all heads; v of all heads], gu = [gate all; up all]; TP4 rank r's record is [k_r; v_r] / [gate_r; up_r]."""
        for j in range(2):
            for name, half in ((f'D{j}.kv.gv', NKV * HD), (f'D{j}.gu.gv', INTER)):
                full = self.tp6[name][0]
                t, per = half // 16, half // 16 // 4          # tiles of one half, of one rank's share of it
                for r in range(4):
                    want = self.tp4[r][name][0]
                    got = np.concatenate([full[r * per:(r + 1) * per], full[t + r * per:t + (r + 1) * per]])
                    self.assertTrue(np.array_equal(got, want), f'{name} rank {r}')

    def test_k_slices_are_the_tp4_o_and_down(self):
        for j in range(2):
            for v in range(4):
                self.assertTrue(np.array_equal(self.tp6[f'D{j}.o{v}.gv'][0], self.tp4[v][f'D{j}.o.gv'][0]), f'o{v}')
                self.assertTrue(np.array_equal(self.tp6[f'D{j}.down{v}.gv'][0], self.tp4[v][f'D{j}.down.gv'][0]), f'down{v}')

    def test_replicated_tensors_are_tp4s(self):
        for name, (a, _) in self.tp6.items():
            if name in self.tp4[0] and not name.endswith(('.q.gv', '.kv.gv', '.gu.gv', 'fc.gv', 'D.meta')):
                self.assertTrue(np.array_equal(a, self.tp4[0][name][0]), name)


if __name__ == '__main__':
    unittest.main()
