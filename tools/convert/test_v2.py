"""Desk tests for the v2 expert reader and writer (#65). No GPU. Needs numpy and libxxhash (or the xxhash package).
Run from the repo root: python3 -m unittest tools.convert.test_v2

Synthetic layers are written with 8 routed experts instead of 256 (v2.N_EXP is patched): the same layout per expert,
a fraction of the bytes.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from tools.convert import v2, v2synth
from tools.convert.pack import PackWriter
from tools.convert.st import ST

NE = 8
L = 41


def read_pack(d, r):
    with open(os.path.join(d, f'rank{r}.manifest.json')) as f:
        m = json.load(f)
    with open(os.path.join(d, f'rank{r}.pack'), 'rb') as f:
        fd = f.read()
    out = {}
    for t in m['tensors']:
        dt = {'u8': np.uint8, 'f16': np.float16, 'i64': np.int64}[t['dtype']]
        out[t['name'].split('.', 1)[1]] = np.frombuffer(fd[t['offset']:t['offset'] + t['nbytes']], dt).reshape(t['shape'])
    return out, m


class V2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ne = v2.N_EXP
        v2.N_EXP = NE
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, 'v2')
        v2synth.make_layer(cls.src, L, (2, 3, 4), NE, seed=3)
        cls.src_pad = os.path.join(cls.tmp.name, 'v2pad')
        v2synth.make_layer(cls.src_pad, L, (2, 3), NE, seed=4, shared=False, pad_halves=True)

    @classmethod
    def tearDownClass(cls):
        v2.N_EXP = cls._ne
        cls.tmp.cleanup()

    def mixed_widths(self, seed=0):
        rng = np.random.default_rng(seed)
        down = rng.integers(2, 5, NE)
        gate = rng.integers(2, 5, (NE, v2.NH))
        up = rng.integers(2, 5, (NE, v2.NH))
        gate[0] = up[0] = down[0] = 3          # expert 0: one width everywhere (kmix 0)
        gate[1], up[1] = 4, 4                  # expert 1: gate/up one width, down another
        down[1] = 3
        sh = dict(gate=rng.integers(3, 5, 16).tolist(), up=[4] * 16, down=4)   # shared: per-half gate widths too
        return v2.Widths(down, gate, up, shared=sh, name='mixed')

    def convert(self, src, wd, nr, ranks=None, shared=True):
        out = tempfile.mkdtemp(dir=self.tmp.name)
        ws = {r: PackWriter(os.path.join(out, f'rank{r}.pack'), r, False, {}) for r in (ranks or range(nr))}
        lay = v2.V2Layer(src, L)
        ex = v2.convert_experts_v2(L, lay, wd, ws, nr, shared=shared)
        lay.close()
        for w in ws.values():
            w.close()
        return out, ex

    def test_alignment_and_order_metadata(self):
        for name in [v2.WIDTH_FILE.format(l=L, k=k) for k in (2, 3, 4)] + [v2.INDEX_FILE.format(l=L)]:
            s = ST(os.path.join(self.src, name))
            self.assertEqual(s.base % (64 << 10), 0)
            for n, t in s.h.items():
                if not n.startswith('__pad.'):
                    self.assertEqual((s.base + t['data_offsets'][0]) % (64 << 10), 0, n)
            self.assertEqual(s.meta['down_order'], 'item128')
            s.close()
        lay = v2.V2Layer(self.src, L)
        self.assertEqual((lay.widths, lay.shared_widths), ([2, 3, 4], [3, 4]))
        # without pad_halves only fragments (even halves) start on 64 KiB. With it every half does.
        self.assertEqual(int(lay._t(3, 3, 'index')[1, 2, 0]) % (64 << 10), 0)
        self.assertNotEqual(int(lay._t(3, 3, 'index')[1, 3, 0]) % (64 << 10), 0)
        lay.close()
        lay = v2.V2Layer(self.src_pad, L)
        self.assertEqual(int(lay._t(3, 3, 'index')[1, 3, 0]) % (64 << 10), 0)
        self.assertEqual(int(lay._t(3, 3, 'index')[1, 3, 1]), v2.half_bytes(3))
        lay.close()

    def test_half_rule_balance(self):
        for n in (3, 4, 5, 6, 7):
            for l in (3, 40, 78):
                cnt = np.zeros(n, int)
                for e in range(257):
                    for h in range(16):
                        cnt[v2.owner(e, h, l, n)] += 1
                self.assertLessEqual(cnt.max() - cnt.min(), 1, (n, l))
                per = [len(v2.owned_halves(e, l, n, r)) for e in range(257) for r in range(n)]
                self.assertEqual(max(per), -(-16 // n))

    def test_down_permutation_roundtrip(self):
        for k in (2, 3, 4):
            a = np.random.default_rng(k).integers(0, 256, v2.half_bytes(k), dtype=np.uint8)
            self.assertTrue(np.array_equal(v2.down_from_g6(v2.down_to_g6(a, k), k), a))
            # item128 tile (c8, k16, c1) lands at g6 tile (c8 * 8 + c1, k16)
            tb = 32 * k
            g = v2.down_to_g6(a, k)
            c8, k16, c1 = 5, 3, 6
            src = a[((c8 * 8 + k16) * 8 + c1) * tb:((c8 * 8 + k16) * 8 + c1 + 1) * tb]
            self.assertTrue(np.array_equal(g[c8 * 8 + c1, k16], src))

    def check_reassembly(self, src, wd, nr, ex):
        """Every half of every expert, rebuilt from all ranks' packs, equals the source bytes at its chosen width."""
        out, _ = self.convert(src, wd, nr)
        lay = v2.V2Layer(src, L)
        packs = [read_pack(out, r)[0] for r in range(nr)]
        seen = set()
        for r, p in enumerate(packs):
            tab, km = p['tr3.table'], p['tr3.kmix']
            self.assertEqual(tab.shape, (len(ex), 6))
            fc1, fc2 = p['tr3.fc1'], p['tr3.fc2']
            for i, e in enumerate(ex):
                gw, uw, dk = wd.of(e)
                hs = v2.owned_halves(e, L, nr, r)
                o1, o2, row, n, bits, h0 = (int(x) for x in tab[i])
                self.assertEqual((n, bits), (len(hs), dk))
                if hs:
                    self.assertEqual(h0, hs[0])
                uniform = all(gw[h] == dk and uw[h] == dk for h in hs)
                self.assertEqual(km[i] == 0, uniform)
                for b, h in enumerate(hs):
                    if km[i]:
                        self.assertEqual((int(km[i]) >> (8 * b)) & 15, gw[h])
                        self.assertEqual((int(km[i]) >> (8 * b + 4)) & 15, uw[h])
                    g = fc1[o1:o1 + v2.half_bytes(gw[h])]
                    o1 += v2.half_bytes(gw[h])
                    u = fc1[o1:o1 + v2.half_bytes(uw[h])]
                    o1 += v2.half_bytes(uw[h])
                    self.assertTrue(np.array_equal(g, lay.half(e, 'gate', h, gw[h])))
                    self.assertTrue(np.array_equal(u, lay.half(e, 'up', h, uw[h])))
                    d = fc2[o2 + b * v2.half_bytes(dk):o2 + (b + 1) * v2.half_bytes(dk)]
                    self.assertTrue(np.array_equal(v2.down_from_g6(d.reshape(384, 8, -1), dk), lay.half(e, 'down', h, dk)))
                    self.assertTrue(np.array_equal(p['tr3.v1'][row + b, 0], lay.hidden(e, 'gate', gw[h])[0]))
                    self.assertTrue(np.array_equal(p['tr3.v1'][row + b, 1], lay.hidden(e, 'up', uw[h])[0]))
                    self.assertTrue(np.array_equal(p['tr3.v2'][row + b, 0], lay.inter(e, 'gate', gw[h])[h * 128:h * 128 + 128]))
                    self.assertTrue(np.array_equal(p['tr3.v2'][row + b, 1], lay.inter(e, 'up', uw[h])[h * 128:h * 128 + 128]))
                    self.assertTrue(np.array_equal(p['tr3.v2'][row + b, 2], lay.inter(e, 'down', dk)[h * 128:h * 128 + 128]))
                    self.assertTrue(np.array_equal(p['tr3.v3'][row + b], lay.hidden(e, 'down', dk)[0]))
                    seen.add((e, h))
                self.assertEqual(o1 - int(tab[i][0]), sum(v2.half_bytes(gw[h]) + v2.half_bytes(uw[h]) for h in hs))
        self.assertEqual(seen, {(e, h) for e in ex for h in range(16)})
        lay.close()

    def test_reassembly_mixed_tp4_5_6(self):
        wd = self.mixed_widths()
        for nr in (4, 5, 6):
            self.check_reassembly(self.src, wd, nr, list(range(NE)) + [256])

    def test_reassembly_uniform_k2_tp3_7(self):
        for nr in (3, 7):
            wd, _ = v2.load_widths('k2', L, shared=0)
            self.check_reassembly(self.src, wd, nr, list(range(NE)))

    def test_pad_halves_and_hidden_vectors(self):
        rng = np.random.default_rng(9)
        wd = v2.Widths(rng.integers(2, 4, NE), rng.integers(2, 4, (NE, 16)), rng.integers(2, 4, (NE, 16)), 0, 't')
        self.check_reassembly(self.src_pad, wd, 4, list(range(NE)))
        lay = v2.V2Layer(self.src_pad, L)
        for p, sg in (('gate', 'signs.in'), ('up', 'signs.in'), ('down', 'signs.out')):
            v, sc = lay.hidden(3, p, 2)
            self.assertEqual(sc.shape, (48,))
            want = lay.idx.get(sg).astype(np.float32) * np.repeat(sc.astype(np.float32), 128)
            self.assertTrue(np.array_equal(v.astype(np.float32), want))
        lay.close()

    def test_one_width_layer_has_zero_kmix(self):
        wd, _ = v2.load_widths('home', L, v1_bits=[3, 4] * (NE // 2), shared=4)
        out, ex = self.convert(self.src, wd, 4)
        for r in range(4):
            self.assertFalse(read_pack(out, r)[0]['tr3.kmix'].any())

    def test_only_owned_halves_read(self):
        wd, _ = v2.load_widths('k3', L, shared=4)
        lay = v2.V2Layer(self.src, L)
        got = []
        orig = lay.half
        lay.half = lambda e, p, h, k: (got.append((e, p, h)), orig(e, p, h, k))[1]
        out = tempfile.mkdtemp(dir=self.tmp.name)
        ws = {1: PackWriter(os.path.join(out, 'rank1.pack'), 1, False, {})}
        v2.convert_experts_v2(L, lay, wd, ws, 6)
        ws[1].close()
        lay.close()
        self.assertEqual(sorted(got), sorted((e, p, h) for e in list(range(NE)) + [256] for p in v2.PROJ
                                             for h in v2.owned_halves(e, L, 6, 1)))

    def test_corrupt_half_refused(self):
        d = tempfile.mkdtemp(dir=self.tmp.name)
        v2synth.make_layer(d, L, (3,), NE, seed=5, shared=False)
        p = os.path.join(d, v2.WIDTH_FILE.format(l=L, k=3))
        s = ST(p)
        lo = s.h['trellis.up']['data_offsets'][0] + s.base + (5 * 16 + 9) * v2.half_bytes(3) + 1234
        s.close()
        with open(p, 'r+b') as f:
            f.seek(lo)
            b = f.read(1)
            f.seek(lo)
            f.write(bytes([b[0] ^ 0x10]))
        lay = v2.V2Layer(d, L)
        lay.half(5, 'up', 8, 3)
        with self.assertRaisesRegex(ValueError, 'expert 5 up half 9'):
            lay.half(5, 'up', 9, 3)
        lay.close()

    def test_manifest_file(self):
        p = os.path.join(self.tmp.name, 'm.json')
        with open(p, 'w') as f:
            json.dump({'format': 'exlr8-v2-manifest', 'name': 'pilot', 'layers': {str(L): {
                'down': [3] * NE, 'gate': [[2] * 8 + [3] * 8] * NE, 'up': [[4] * 16] * NE,
                'shared': {'gate': [3] * 16, 'up': [4] * 8 + [3] * 8, 'down': 4}}}}, f)
        wd, prov = v2.load_widths(p, L)
        self.assertEqual(wd.of(2), ([2] * 8 + [3] * 8, [4] * 16, 3))
        self.assertEqual(wd.of(256), ([3] * 16, [4] * 8 + [3] * 8, 4))
        self.assertTrue(prov.startswith('pilot sha256:'))
        self.assertEqual(v2.load_widths('down1', L, v1_bits=[3, 4] * (NE // 2))[0].of(256), ([3] * 16, [3] * 16, 3))

    def test_parse_experts(self):
        self.assertEqual(v2.parse_experts('0-63 (partial: a test set, not a whole layer)'), list(range(64)))
        self.assertEqual(v2.parse_experts('0-2,7,9-10'), [0, 1, 2, 7, 9, 10])


if __name__ == '__main__':
    unittest.main()
