import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from steering import contrast, release

try:
    import numpy as np
except ImportError:
    np = None

TOOL = os.path.join(os.path.dirname(HERE), 'contrast.py')
HID, NL = 4, 3


class ArgvTests(unittest.TestCase):
    def test_serve_flags_dropped(self):
        line = ('2026-10-25 supervisor: engine: G6_CP=1 G6_RT_X=1 ./bin/g6m4 --rank 0 --serve 127.0.0.1:9 '
                '--mtp 2 --graph --v2 /m --prefill 4096 --chunk 8 --steering-profile /p --cap 500032')
        env, argv = contrast.offline_argv(line, ['/a.ids', '/b.ids'], 128)
        self.assertEqual(env, ['G6_CP=1', 'LD_LIBRARY_PATH=bin/lib'])
        self.assertEqual(argv, ['./bin/g6m4', '--rank', '0', '--v2', '/m', '--chunk', '8', '--cap', '128',
                                '--gen', str(contrast.GEN), '--temp', '0', '--prompt', '@/a.ids', '--prompt',
                                '@/b.ids'])


@unittest.skipIf(np is None, 'numpy is not installed')
class ReduceFitTests(unittest.TestCase):
    def test_direction_and_alpha(self):
        # positive answers write +1 along axis 0 in every sublayer, negative ones -1: the fit finds axis 0 with
        # c+ = 2, c- = -2 (attention + MLP) and alpha 1 - c+ / c- = 2
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            seqs = [{'file': 'x', 'label': 1, 'tokens': 5, 'start': 2, 'end': 4},
                    {'file': 'y', 'label': 0, 'tokens': 6, 'start': 2, 'end': 5}]
            with open(os.path.join(d, 'seqs.json'), 'w') as f:
                json.dump({'pairs_sha256': '1' * 64, 'seqs': seqs}, f)
            rows = []
            for s in seqs:
                for i in range(s['tokens'] + contrast.GEN):
                    sign = (1 if s['label'] else -1) if s['start'] <= i < s['end'] else 0
                    w = 0.01 * rng.standard_normal((2 * NL, HID))
                    w[:, 0] += sign
                    rows.append(np.concatenate([np.zeros((1, HID)), np.cumsum(w, 0)]))
            rows = np.asarray(rows, '<f4')           # [row][slot][hid]
            dump = b''
            for k in range(0, len(rows), 3):         # steps of up to 3 rows
                x = rows[k:k + 3].transpose(1, 0, 2)
                dump += struct.pack('<4i', k, x.shape[1], 2 * NL + 1, HID) + x.tobytes()
            subprocess.run([sys.executable, TOOL, 'reduce', d], input=dump, check=True, capture_output=True)
            base = {'model': {'name': 'm', 'revision': 'r', 'weight_set': 'a' * 64, 'tokenizer_revision': 't',
                              'chat_template_sha256': 'b' * 64, 'hidden_size': HID, 'layer_count': NL}}
            with open(os.path.join(d, 'base.json'), 'w') as f:
                json.dump(base, f)
            out = os.path.join(d, 'profile')
            subprocess.run([sys.executable, TOOL, 'fit', d, out, '--base', os.path.join(d, 'base.json'),
                            '--name', 'test', '--layers', '0-2'], check=True, capture_output=True)
            m = release.validate_directory(out, expected=base['model'])
            self.assertEqual(m['layers'], [0, 1, 2])
            for a in m['alphas']:
                self.assertAlmostEqual(a, 2.0, delta=0.05)
            dirs = np.fromfile(os.path.join(out, 'directions.f32'), '<f4').reshape(NL, HID)
            self.assertTrue((dirs[:, 0] > 0.99).all())


if __name__ == '__main__':
    unittest.main()
