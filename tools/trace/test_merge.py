import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from tools.trace.merge import main

MS = 1000          # trace timestamps are microseconds
NS_MS = 10**6


def span(rank, ts):
    return {'name': 'prefill', 'ph': 'X', 'pid': rank, 'tid': 0, 'ts': ts, 'dur': 1500, 'args': {'requests': [7]}}


class ClockTest(unittest.TestCase):
    """--clock puts a rank written on another machine's clock onto the client's clock (clock_probe.py samples)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.true_us = 1_800_000_000 * 10**6           # the same moment, on the client's clock

    def trace(self, rank, skew_ms):
        """a rank file as trace.h writes it (array never closed), stamped skew_ms ahead of the client's clock."""
        p = self.root / f's2.r{rank}.json'
        meta = {'name': 'process_name', 'ph': 'M', 'pid': rank, 'args': {'name': f'rank {rank}'}}
        p.write_text('[\n' + json.dumps(meta) + ',\n' + json.dumps(span(rank, self.true_us + skew_ms * MS)) + ',\n')
        return str(p)

    def probe(self, name, offset_ms, rtt_ms, worse=True):
        """clock_probe.py output: a best sample with this offset and round trip, plus (worse) a slower sample whose
        offset is 100 ms wrong, so only the best-round-trip choice gives the right offset."""
        t0 = 1_800_000_000 * 10**9
        rtt = int(rtt_ms * NS_MS)
        samples = [[t0, t0 + rtt // 2 + int(offset_ms * NS_MS), t0 + rtt]]
        if worse:
            samples.insert(0, [t0 - 10**9, t0 - 10**9 + 15 * NS_MS + int((offset_ms + 100) * NS_MS), t0 - 10**9 + 30 * NS_MS])
        p = self.root / name
        p.write_text(json.dumps({'host': 'h', 'samples': samples}))
        return str(p)

    def merge(self, *args):
        out = self.root / 'merged.json'
        if out.exists():
            out.unlink()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            main([*args, '-o', str(out)])
        with open(out) as f:
            ev = json.load(f)['traceEvents']
        return {e['pid']: e['ts'] for e in ev if e.get('ph') == 'X'}, log.getvalue()

    def test_no_clock_joins_as_written(self):
        ts, _ = self.merge(self.trace(0, 0), self.trace(1, 37))
        self.assertEqual(ts, {0: self.true_us, 1: self.true_us + 37 * MS})

    def test_offset_mean_of_before_and_after(self):
        before = self.probe('before.json', 36, 0.4)
        after = self.probe('after.json', 38, 0.6)
        ts, log = self.merge(self.trace(0, 0), self.trace(1, 37), '--clock', f'1={before},{after}')
        self.assertEqual(ts, {0: self.true_us, 1: self.true_us})
        self.assertIn('rank 1: offset 37.000 ms, round trip 0.400/0.600 ms, drift 2.000 ms', log)

    def test_one_probe(self):
        ts, _ = self.merge(self.trace(0, 0), self.trace(1, -12), '--clock', f'1={self.probe("p.json", -12, 1)}')
        self.assertEqual(ts[1], self.true_us)

    def test_refuses_round_trip_over_limit(self):
        with self.assertRaises(SystemExit) as e:
            self.merge(self.trace(1, 0), '--clock', f'1={self.probe("p.json", 0, 50.001, worse=False)}')
        self.assertIn('round trip over --max-rtt-ms 50', str(e.exception))

    def test_drift_limit_is_inclusive(self):
        t = self.trace(1, 0)
        self.merge(t, '--clock', f'1={self.probe("a.json", 0, 1)},{self.probe("b.json", 50, 1)}')
        with self.assertRaises(SystemExit) as e:
            self.merge(t, '--clock', f'1={self.probe("c.json", 0, 1)},{self.probe("d.json", 50.001, 1)}')
        self.assertIn('drift over --max-drift-ms 50', str(e.exception))

    def test_refuses_a_rank_with_no_trace(self):
        with self.assertRaises(SystemExit) as e:
            self.merge(self.trace(0, 0), '--clock', f'3={self.probe("p.json", 0, 1)}')
        self.assertIn('has no trace', str(e.exception))

    def test_refuses_bad_samples(self):
        p = self.root / 'bad.json'
        p.write_text(json.dumps({'samples': [[1.5, 2, 3]]}))
        with self.assertRaises(SystemExit) as e:
            self.merge(self.trace(1, 0), '--clock', f'1={p}')
        self.assertIn('integer ns', str(e.exception))


if __name__ == '__main__':
    unittest.main()
