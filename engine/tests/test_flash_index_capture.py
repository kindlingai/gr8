import struct
import tempfile
from pathlib import Path
import unittest
from check_flash_index_capture import compare, HEADER, MAGIC, WORDS


def block(layer, begin, end):
    payload = bytearray()
    for pos in range(begin, end):
        visible = (pos + 1) // 4
        ids = list(range(max(0, visible - 512), visible))
        ids += [-1] * (512 - len(ids))
        payload += struct.pack('<513i', pos, *ids)
    return HEADER.pack(MAGIC, layer, end, end-begin, 512) + payload


class CaptureTest(unittest.TestCase):
    def check(self, a, b=None, rows=9):
        with tempfile.TemporaryDirectory() as tmp:
            x, y = Path(tmp)/'a', Path(tmp)/'b'
            x.write_bytes(a)
            y.write_bytes(a if b is None else b)
            return compare(x, y, [3, 7], rows)

    def good(self):
        return b''.join(block(l, a, b) for a,b in [(0,4),(4,9)] for l in [3,7])

    def test_complete(self):
        self.assertEqual(self.check(self.good())['selected_pool_rows'],18)

    def test_sparse_boundary(self):
        data = block(3,0,2053)+block(7,0,2053)
        self.assertEqual(self.check(data,rows=2053)['status'],'PASS')

    def test_empty(self):
        with self.assertRaises(ValueError): self.check(b'')

    def test_missing_layer(self):
        with self.assertRaises(ValueError): self.check(block(3,0,9))

    def test_truncated(self):
        with self.assertRaises(ValueError): self.check(self.good()[:-1])

    def test_missing_rank_block(self):
        with self.assertRaises(ValueError): self.check(self.good(), self.good()[:24])

    def test_duplicate(self):
        with self.assertRaises(ValueError): self.check(self.good()+block(3,4,9))

    def test_wrong_position_both_ranks(self):
        x=bytearray(self.good());struct.pack_into('<i',x,24,1)
        with self.assertRaises(ValueError): self.check(x)

    def test_wrong_padding_both_ranks(self):
        x=bytearray(self.good());struct.pack_into('<i',x,28,0)
        with self.assertRaises(ValueError): self.check(x)

    def test_different_pool_ids(self):
        x=bytearray(self.good());struct.pack_into('<i',x,24+3*WORDS*4+4,1)
        with self.assertRaisesRegex(ValueError,'pool-ID mismatch'): self.check(self.good(),x)

    def test_bad_magic(self):
        with self.assertRaises(ValueError): self.check(b'BROKEN!!'+self.good()[8:])

    def test_gap(self):
        with self.assertRaises(ValueError): self.check(block(3,1,9)+block(7,0,9))


if __name__ == '__main__': unittest.main()
