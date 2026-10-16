import io
from pathlib import Path
import struct
import tempfile
import unittest
from tools.cp.merge_logits import MAGIC, HEADER, STEP, merge


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'merged'

    def fixture(self, nr=4):
        vocab = 13
        files, streams = [], []
        expected = io.BytesIO()
        lo = 0
        widths = [vocab // nr + (r < vocab % nr) for r in range(nr)]
        for r, vn in enumerate(widths):
            f = self.root / f'rank{r}'
            stream = io.BytesIO(MAGIC + HEADER.pack(nr, r, vocab, lo, vn))
            stream.seek(0, 2)
            streams.append(stream)
            files.append(f)
            lo += vn
        for step, rows in ((3, 4), (4, 1), (0, 2)):  # new sequence resets the step
            hdr = STEP.pack(step, rows)
            expected.write(hdr)
            for r, (vn, stream) in enumerate(zip(widths, streams)):
                data = b''.join(struct.pack('<f', step * 100 + r * 10 + i / 8) for i in range(rows * vn))
                stream.write(hdr + data)
                expected.write(struct.pack('<i', vn) + data)
        for f, stream in zip(files, streams):
            f.write_bytes(stream.getvalue())
        return files, expected.getvalue()

    def test_byte_exact_all_topologies(self):
        for nr in (2, 3, 4, 6):
            with self.subTest(nr=nr):
                files, expected = self.fixture(nr)
                self.assertEqual(merge(files[::-1], self.out), 3)
                self.assertEqual(self.out.read_bytes(), expected)

    def reject(self, change):
        files, _ = self.fixture()
        change(files)
        self.out.write_bytes(b'previous good output')
        with self.assertRaises(ValueError):
            merge(files, self.out)
        self.assertEqual(self.out.read_bytes(), b'previous good output')
        self.assertEqual(list(self.root.glob('merged.*')), [])

    def mutate(self, files, offset, data):
        b = bytearray(files[1].read_bytes())
        b[offset:offset + len(data)] = data
        files[1].write_bytes(b)

    def test_invalid_headers_and_records(self):
        for offset, data in ((0, b'X'), (8, struct.pack('<i', -1)), (12, struct.pack('<i', 0)),
                             (16, struct.pack('<i', 14)), (20, struct.pack('<i', 0)),
                             (28, struct.pack('<i', 9)), (32, struct.pack('<i', 9))):
            with self.subTest(offset=offset):
                self.reject(lambda fs: self.mutate(fs, offset, data))

    def test_missing_rank(self):
        self.reject(lambda fs: fs.pop())

    def test_truncation(self):
        for cut in (10, 28, 30, -1):
            with self.subTest(cut=cut):
                self.reject(lambda fs: fs[1].write_bytes(fs[1].read_bytes()[:cut]))

    def test_empty_dump(self):
        self.reject(lambda fs: [f.write_bytes(f.read_bytes()[:28]) for f in fs])

    def test_output_alias(self):
        files, _ = self.fixture()
        before = files[0].read_bytes()
        with self.assertRaises(ValueError):
            merge(files, files[0])
        self.assertEqual(files[0].read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
