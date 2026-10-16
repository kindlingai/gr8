"""Host checks of the residual dump reader and numerical gate (no GPU)."""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from check_dump import read_dump, check


class DumpTest(unittest.TestCase):
    def test_records(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'dump'
            x = np.ones((9, 2, 4, 4096), np.float32)
            p.write_bytes(struct.pack('<4i', 0, 2, 9, 16384) + x.tobytes()
                          + struct.pack('<4i', 1, 1, 9, 16384) + (2*x[:, :1]).tobytes())
            y = read_dump(p)
            self.assertEqual(y.shape, (9, 3, 4, 4096))
            np.testing.assert_array_equal(y[:, :2], x)
            self.assertTrue(check(y[0], y[0])[0])
            self.assertFalse(check(y[0] * 1.1, y[0])[0])
            with self.assertRaises(ValueError):
                check(y[0], x[0])

    def test_bad_records(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'dump'
            cases = [b'', b'abc', struct.pack('<4i', 0, 1, 9, 16384),
                     struct.pack('<4i', 1, 1, 9, 16384), struct.pack('<4i', 0, 1, 9, 4096),
                     struct.pack('<4i', 0, 1, 9, 16384) + np.full((9, 1, 4, 4096), np.nan, np.float32).tobytes()]
            for raw in cases:
                p.write_bytes(raw)
                with self.assertRaises(ValueError):
                    read_dump(p)


if __name__ == '__main__':
    unittest.main()
