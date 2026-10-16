"""CPU-only checks for the lane-aware staged-receive trace reader."""
import tempfile
import unittest
from pathlib import Path
from pf_overlap_trace import load, summary


class TraceTest(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.log'
            p.write_text(text)
            return load(p)

    def test_two_lanes_keep_repeated_steps(self):
        comm,pxt=self.parse('PXT 1 0 8192 1 2 8 9 9 10\nPFCOMM 1 0 3 5 7\n'
                            '# PXT second lane\nPXT 1 0 8191 10 20 80 90 90 100\n'
                            'PFCOMM 1 0 30 50 70\n')
        self.assertEqual(set(comm),{(0,1,0),(1,1,0)})
        result=summary(comm,pxt)
        self.assertAlmostEqual(result['exposed_wait_s'],22e-9,places=15)
        self.assertAlmostEqual(result['copy_s'],22e-9,places=15)

    def test_missing_wait_rejected(self):
        with self.assertRaises(AssertionError):
            self.parse('PXT 1 0 8192 1 2 8 9 9 10\n')

    def test_duplicate_rejected(self):
        with self.assertRaises(AssertionError):
            self.parse('PFCOMM 1 0 3 5 7\nPFCOMM 1 0 3 5 7\n')

    def test_reversed_timestamp_rejected(self):
        with self.assertRaises(AssertionError):
            self.parse('PXT 1 0 8192 1 2 8 9 9 10\nPFCOMM 1 0 5 3 7\n')


if __name__=='__main__': unittest.main()
