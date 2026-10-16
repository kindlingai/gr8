import math
from pathlib import Path
import struct
import tempfile
import unittest
from head_capture import Capture, compare_ranks, HEADER, CALL, ROW, DONE


def fixture(rank=0):
    h = HEADER.pack(b'G6DHD001', 2, rank, 4096, 154880, 8, 3, 0, 1)
    for call in range(3):
        h += b'CALL' + CALL.pack(call, 0, call + 1, call, 1, int(call > 0), 11 + call)
        h += ROW.pack(10 + call, 20 + call, 21 + call, 12 + call, 42, 0, 0, 0, 0, -1)
        h += struct.pack('<4096f', *([0.125] * 4096))
    return h + b'DONE' + DONE.pack(3, 3)


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = [Path(self.tmp.name) / f'rank{r}' for r in range(2)]
        for r, p in enumerate(self.paths):
            p.write_bytes(fixture(r))

    def tearDown(self):
        self.tmp.cleanup()

    def edit(self, offset, fmt, *values):
        p = self.paths[0]
        b = bytearray(p.read_bytes())
        struct.pack_into(fmt, b, offset, *values)
        p.write_bytes(b)

    def bad(self):
        with self.assertRaises(ValueError):
            compare_ranks(self.paths)

    def test_valid(self):
        self.assertEqual(compare_ranks(self.paths)['calls'], 3)

    def test_partial_consumption_not_complete(self):
        c = Capture(self.paths[0]); it = c.calls(); next(it)
        self.assertFalse(c.complete); it.close()

    def test_truncated(self):
        data = self.paths[0].read_bytes()
        for cut in (1, 19, 20, 21, 8192, len(data) - 1):
            self.paths[0].write_bytes(data[:-cut]); self.bad()

    def test_footer_totals(self):
        self.edit(len(fixture()) - 16, '<Q', 4); self.bad()

    def test_trailing_bytes(self):
        with self.paths[0].open('ab') as f: f.write(b'x')
        self.bad()

    def test_wrong_hidden_shape(self):
        self.edit(16, '<I', 4097); self.bad()

    def test_wrong_dtype(self):
        self.edit(36, '<I', 2); self.bad()

    def test_duplicate_rank(self):
        with self.assertRaises(ValueError): compare_ranks([self.paths[0], self.paths[0]])

    def test_missing_rank(self):
        with self.assertRaises(ValueError): compare_ranks(self.paths[:1])

    def test_bad_call_id(self):
        self.edit(HEADER.size + 4, '<Q', 2); self.bad()

    def test_bad_shape(self):
        self.edit(HEADER.size + 4 + 28, '<i', 99); self.bad()

    def test_bad_chain(self):
        self.edit(HEADER.size + 4 + 32, '<i', 1); self.bad()

    def test_bad_position(self):
        self.edit(HEADER.size + 4 + CALL.size + 12, '<i', 999); self.bad()

    def test_bad_token(self):
        self.edit(HEADER.size + 4 + CALL.size + 4, '<i', -1); self.bad()

    def test_lp_bits_differ(self):
        self.edit(HEADER.size + 4 + CALL.size + 36, '<f', -1.0000001); self.bad()

    def test_hidden_nan(self):
        self.edit(HEADER.size + 4 + CALL.size + ROW.size, '<f', math.nan); self.bad()

    def test_hidden_bits_differ(self):
        self.edit(HEADER.size + 4 + CALL.size + ROW.size, '<f', 0.12500001); self.bad()

    def test_broken_chain_input(self):
        self.edit(HEADER.size + 4 + CALL.size + ROW.size + 4096*4 + 4 + CALL.size + 4, '<i', 999); self.bad()


if __name__ == '__main__':
    unittest.main()
