#!/usr/bin/env python3
"""CPU ownership and unchanged-math checks; GPU EXACT/racecheck remain required."""
from collections import Counter
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


class SplitBarrierOwnership(unittest.TestCase):
    def test_widen_matches_pv_ownership(self):
        for warp in range(8):
            writes = Counter(
                ((lane >> 2) + k * 8, (warp * 4 + (lane & 3)) * 16 + d)
                for lane in range(32) for k in range(8) for d in range(16)
            )
            expected = Counter((row, dim) for row in range(64)
                               for dim in range(warp * 64, warp * 64 + 64))
            self.assertEqual(writes, expected)

    def test_window_copy_matches_pv_ownership(self):
        for warp in range(8):
            writes = Counter(
                ((lane >> 3) + i * 4, warp * 64 + (lane & 7) * 8 + d)
                for lane in range(32) for i in range(16) for d in range(8)
            )
            expected = Counter((row, dim) for row in range(64)
                               for dim in range(warp * 64, warp * 64 + 64))
            self.assertEqual(writes, expected)

    def test_math_blocks_unchanged_from_our_pipe(self):
        original = (HERE.parents[2] / 'ds41_attn_pipe.cuh').read_text()
        candidate = (HERE / 'kernel.cuh').read_text()
        for marker in ('{   // S[16 heads]', '{   // online softmax',
                       '{   // acc[16 heads]'):
            def block(text):
                start = text.index(marker)
                # All three blocks end at this indentation in our kernel.
                end = text.index('\n        }', start) + len('\n        }')
                return text[start:end]
            self.assertEqual(block(original), block(candidate))
        loop = candidate[candidate.index('    while (cblk < nblk)\n'):
                         candidate.index("    cp_async_wait0();   // Q's copies")]
        self.assertNotIn('__syncthreads', loop)
        self.assertIn('__syncwarp();', loop)


if __name__ == '__main__':
    unittest.main()
