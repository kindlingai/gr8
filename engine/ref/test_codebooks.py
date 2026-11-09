"""The Python reference's trellis codebooks (g6ref.py) against the constants the GPU test and the Rust loader check,
over every state; and its tile decode against a per-weight decode of random tiles.

    python3 -m unittest engine.ref.test_codebooks        (from the repo root; needs numpy)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g6ref  # noqa: E402

# engine/kernels/tests/tr3_e4m3_test.cu WANT, engine/loader-rs/src/codebook.rs HASHES
HASHES = {'mcg': (0x7bb36b05a3b885a7, 0xd6995a3ae102700a),
          'mul1': (0x55e1c748ed9f0cd7, 0xf4f1a8cc7cda90bf),
          'pmcgx': (0x6f7539ea76219e77, 0xc8d37146b1cb8465)}


class Codebooks(unittest.TestCase):
    def test_every_state(self):
        for cb in g6ref.CODEBOOKS:
            self.assertEqual(g6ref.codebook_hashes(cb), HASHES[cb], cb)

    def test_tiles(self):
        rng = np.random.default_rng(7)
        for cb in g6ref.CODEBOOKS:
            for k in (2, 3, 4):
                tiles = rng.integers(-32768, 32768, (3, 16 * k), dtype=np.int64).astype(np.int16)
                got = g6ref.tr3_values(tiles, cb)
                u32 = tiles.view(np.uint16).astype(np.uint32)
                words = u32[:, 0::2] | (u32[:, 1::2] << 16)
                bits = np.unpackbits(words.astype('>u4').view(np.uint8).reshape(3, -1), axis=-1)
                L = g6ref.STATE_BITS[cb]
                for i in range(3):
                    for t in range(256):
                        end = (((t | 1) if cb == 'pmcgx' else t) + 1) * k
                        s = 0
                        for q in range(L):
                            s = (s << 1) | int(bits[i, (end - L + q) % (256 * k)])
                        v = g6ref.tr3_cb(np.array([s], np.uint64), cb).reshape(-1)
                        want = v[t & 1] if cb == 'pmcgx' else v[0]
                        self.assertEqual(got[i, t].view(np.uint16), want.view(np.uint16), (cb, k, i, t))


if __name__ == '__main__':
    unittest.main()
