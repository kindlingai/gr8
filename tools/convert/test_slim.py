"""Desk tests for slim packs (tools/convert/slim.py, slimpack.py, convert.py --slim-qo). No GPU. Needs numpy.
Run from the repo root: python3 -m unittest tools.convert.test_slim
"""
import json
import os
import tempfile
import unittest

import numpy as np

from tools.convert import checkpack, slim, slimpack
from tools.convert.pack import PackWriter

# (name suffix, dtype, shape): a miniature layer holding every tensor the slimming touches, plus others that must stay
LAYER = [('attn_norm', 'bf16', (32,)), ('q_a.gptq', 'u8', (50,)), ('q_b.mx.codes', 'u8', (16, 64)), ('q_b.mx.scales', 'u8', (16, 2)),
         ('q_b.mx.gv', 'u8', (33,)), ('kv_b.uk', 'bf16', (2, 4, 8)), ('kv_b.uv', 'bf16', (2, 5, 8)), ('kv_b.uk.gv', 'u8', (40,)),
         ('kv_b.uv.gv', 'u8', (41,)), ('o_proj.gptq', 'u8', (60,)), ('o_proj.mx.codes', 'u8', (8, 64)), ('o_proj.mx.scales', 'u8', (8, 2)),
         ('router.w.gv', 'u8', (70,))]
QO_T = ('q_b.mx.codes', 'q_b.mx.scales', 'o_proj.mx.codes', 'o_proj.mx.scales')


def tensor(rng, dtype, shape):
    return rng.integers(0, 256, shape, dtype=np.uint8) if dtype == 'u8' else rng.integers(0, 60000, shape, dtype=np.uint16)


def write(path, layers, rank=0, writer=PackWriter, **kw):
    w = writer(path, rank, False, dict(converter='test', layers=sorted(layers)), **kw)
    want = {}
    for l in layers:
        for n, dt, sh in LAYER:
            a = tensor(np.random.default_rng(l * 1000 + sum(map(ord, n))), dt, sh)
            w.put(f'L{l:03d}.{n}', a, dt, layout='x', heads=[0, 2])
            want[f'L{l:03d}.{n}'] = a
    w.close()
    return want


def qo_manifest(path, layers):
    with open(path, 'w') as f:
        json.dump({'tensors': [{'name': f'L{l:03d}.qo.{t}'} for l in layers for t in QO_T]}, f)


def rd(d, f):
    with open(os.path.join(d, f), 'rb') as fh:
        return fh.read()


def load(d, r=0):
    m = json.loads(rd(d, f'rank{r}.manifest.json'))
    raw = rd(d, f'rank{r}.pack')
    return m, {e['name']: raw[e['offset']:e['offset'] + e['nbytes']] for e in m['tensors']}


class Slim(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        d = self.t.name
        self.full, self.out, self.out2 = (os.path.join(d, x) for x in ('full', 'out', 'out2'))
        os.mkdir(self.full)
        self.layers = [0, 1, 78]
        self.want = write(os.path.join(self.full, 'rank0.pack'), self.layers)
        self.qo = os.path.join(d, 'qo.manifest.json')
        qo_manifest(self.qo, [0, 1])                  # 78 (MTP) is not in the QO pack; layer 1 is covered too

    def tearDown(self):
        self.t.cleanup()

    def expected_gone(self):
        gone = set()
        for l in self.layers:
            gone |= {f'L{l:03d}.kv_b.uk.gv', f'L{l:03d}.kv_b.uv.gv'}
            if l != 78:
                gone |= {f'L{l:03d}.{t}' for t in QO_T}
        return gone

    def test_qo_layers(self):
        self.assertEqual(slim.qo_layers(self.qo), {0, 1})
        with open(self.qo) as f:
            m = json.load(f)
        m['tensors'] = [e for e in m['tensors'] if e['name'] != 'L001.qo.o_proj.mx.scales']   # one tensor missing: not covered
        self.assertEqual(slim.qo_layers(m), {0})

    def test_slimpack(self):
        info = slimpack.slimpack(os.path.join(self.full, 'rank0.manifest.json'), self.out, slim.qo_layers(self.qo), False)
        m, got = load(self.out)
        gone = self.expected_gone()
        self.assertEqual(set(got), set(self.want) - gone)
        self.assertEqual(info['dropped'], len(gone))
        for n, b in got.items():                                       # kept bytes identical
            self.assertEqual(b, self.want[n].tobytes(), n)
        for n in ('L078.q_b.mx.codes', 'L078.o_proj.mx.codes', 'L001.q_b.mx.gv', 'L001.kv_b.uk', 'L000.o_proj.gptq'):
            self.assertIn(n, got)                                      # MTP layer copies, decode streams and rowmajor kv_b stay
        self.assertTrue(m['slim_qo'])
        self.assertEqual(m['layers'], [0, 1, 78])
        self.assertEqual({e['name']: e.get('heads') for e in m['tensors']}['L000.q_a.gptq'], [0, 2])   # info fields survive
        self.assertLess(m['pack_bytes'], os.path.getsize(os.path.join(self.full, 'rank0.pack')))
        self.assertTrue(checkpack.check(os.path.join(self.out, 'rank0.manifest.json'))['ok'])

    def test_idempotent(self):
        qo = slim.qo_layers(self.qo)
        slimpack.slimpack(os.path.join(self.full, 'rank0.manifest.json'), self.out, qo, False)
        slimpack.slimpack(os.path.join(self.out, 'rank0.manifest.json'), self.out2, qo, False)
        for f in ('rank0.pack', 'rank0.manifest.json'):
            self.assertEqual(rd(self.out, f), rd(self.out2, f), f)

    def test_corrupt_source_aborts(self):
        p = os.path.join(self.full, 'rank0.pack')
        m = json.loads(rd(self.full, 'rank0.manifest.json'))
        e = next(e for e in m['tensors'] if e['name'] == 'L000.q_a.gptq')
        with open(p, 'r+b') as f:
            f.seek(e['offset'] + 3)
            f.write(b'\xff\xfe')
        with self.assertRaises(SystemExit):
            slimpack.slimpack(os.path.join(self.full, 'rank0.manifest.json'), self.out, slim.qo_layers(self.qo), False)

    def test_writer_matches_slimpack(self):
        """convert.py --slim-qo (SlimWriter) writes the same bytes as slimming the full pack afterwards"""
        d = os.path.join(self.t.name, 'sw')
        os.mkdir(d)
        write(os.path.join(d, 'rank0.pack'), self.layers, writer=slim.SlimWriter, qo=slim.qo_layers(self.qo))
        slimpack.slimpack(os.path.join(self.full, 'rank0.manifest.json'), self.out, slim.qo_layers(self.qo), False)
        for f in ('rank0.pack', 'rank0.manifest.json'):
            self.assertEqual(rd(d, f), rd(self.out, f), f)


if __name__ == '__main__':
    unittest.main()
