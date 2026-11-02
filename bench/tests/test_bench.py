"""bench.py against the fake server: python3 -m unittest discover -s bench/tests (no GPU, no model)."""
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import fake_server  # noqa: E402
from gr8bench import client as C, docs, kl, report, stages as S, stats, thermal  # noqa: E402

BENCH = HERE.parent / 'bench.py'


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = fake_server.start()
        cls.cl = C.Client(cls.url)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.srv.st.drift, self.srv.st.retrieve, self.srv.st.cache_hit = 0.0, True, 0
        self.srv.st.fail_nth = self.srv.st.chat_calls = 0

    def bench(self, *args):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'r.json'
            p = subprocess.run([sys.executable, str(BENCH), '--url', self.url, '--json', str(out), *args],
                               capture_output=True, text=True, timeout=300)
            return p, (json.loads(out.read_text()) if out.exists() else None)


class Stats(unittest.TestCase):
    def test_wilcoxon_exact(self):
        self.assertAlmostEqual(stats.wilcoxon([1, 2, 3, 4, 5]), 2 / 32)    # all positive: 1 of 32 patterns, two-sided
        self.assertEqual(stats.wilcoxon([0, 0]), 1.0)
        self.assertEqual(stats.wilcoxon([1, -1]), 1.0)

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_paired_ratio(self):
        a = [{'i': i, 'tps': 100.0 + i, 'tok_per_step': 3.0, 'ms_per_step': 40.0} for i in range(8)]
        b = [{'i': i, 'tps': 1.1 * (100.0 + i), 'tok_per_step': 3.3, 'ms_per_step': 40.0} for i in range(8)]
        r = stats.paired(a, b)
        self.assertAlmostEqual(r['ratio'], 1.1)
        self.assertAlmostEqual(r['tok_per_step'], 1.1)
        self.assertAlmostEqual(r['ms_per_step'], 1.0)
        self.assertLess(r['p'], 0.01)
        self.assertIsNone(stats.paired(a[:1], b[:1]))

    def test_paired_skips_runs_with_problems(self):
        a = [{'i': i, 'tps': 100.0} for i in range(4)]
        b = [{'i': i, 'tps': 50.0 if i == 0 else 100.0, 'problem': 'wrong count' if i == 0 else ''} for i in range(4)]
        self.assertAlmostEqual(stats.paired(a, b)['ratio'], 1.0)


class KLMath(unittest.TestCase):
    def test_identical_is_zero(self):
        top = [(1, math.log(0.5)), (2, math.log(0.3)), (3, math.log(0.1))]
        self.assertAlmostEqual(kl.kl_position(top, top), 0.0, places=9)

    def test_known_value(self):
        # ref: a=0.6 b=0.3, tail 0.1; new: a=0.3 b=0.6, tail 0.1 -> 0.6 ln2 + 0.3 ln(1/2) = 0.3 ln 2
        ref = [(1, math.log(0.6)), (2, math.log(0.3))]
        new = [(2, math.log(0.6)), (1, math.log(0.3))]
        self.assertAlmostEqual(kl.kl_position(ref, new), 0.3 * math.log(2), places=9)

    def test_missing_token_uses_floor_and_tail_is_lumped(self):
        ref = [(1, math.log(0.5)), (2, math.log(0.4))]
        new = [(1, math.log(0.7)), (3, math.log(0.2))]  # id 2 missing from new's top-K: floor 0.2
        want = 0.5 * math.log(0.5 / 0.7) + 0.4 * math.log(0.4 / 0.2) + 0.1 * math.log(0.1 / (1 - 0.9))
        self.assertAlmostEqual(kl.kl_position(ref, new), want, places=9)


class KLRun(Base):
    def test_self_comparison_and_drift(self):
        ref = {'sequences': kl.make_reference(self.cl, ['alpha beta gamma', 'delta epsilon'], n_tokens=12, k=5)}
        self.assertAlmostEqual(kl.compare(self.cl, ref, 5)['mean'], 0.0, places=9)
        self.srv.st.drift = 2.0
        d = kl.compare(self.cl, ref, 5)
        self.assertGreater(d['mean'], 0.01)
        self.assertLess(d['top1'], 1.0)


class Docs(Base):
    def test_size_and_depth(self):
        msgs, info = docs.needle_prompt(self.cl, 5000, 0.15, 7, 'amber-cedar-ember-1234')
        self.assertLess(abs(info['doc_tokens'] - 5000) / 5000, 0.03)
        self.assertLess(abs(info['needle_depth'] - 0.15), 0.03)


class Heat(unittest.TestCase):
    def thermal_with(self, temps, util=0):
        th = thermal.Thermal(['local'], window=60, min_wait=0, max_wait=5, log=lambda m: None)
        th.clock_control = False
        now = time.time()
        for k, t in enumerate(temps):   # one sample a second, ending now
            th.hosts[0].samples.append((now - len(temps) + k, t, 2000, 50, util, 0))
        return th, now

    def test_falling_is_not_flat(self):
        th, now = self.thermal_with([70 - k / 10 for k in range(90)])   # 6 C a minute
        self.assertLess(th._slope(th.hosts[0], now), -5)

    def test_level_is_flat(self):
        th, now = self.thermal_with([50] * 90)
        self.assertAlmostEqual(th._slope(th.hosts[0], now), 0.0)

    def test_short_window_has_no_slope(self):
        th, now = self.thermal_with([50] * 20)
        self.assertIsNone(th._slope(th.hosts[0], now))

    def test_window_stats_and_throttle(self):
        th, now = self.thermal_with([60, 70, 80])
        th.hosts[0].samples.append((now, 81, 1500, 60, 90, 0x40))
        w = th.window_stats(now - 10, now + 1)['local']
        self.assertEqual((w['temp_start'], w['temp_max'], w['sm_min']), (60, 81, 1500))
        self.assertEqual(w['throttle'], ['hw thermal'])

    def test_settle_starts_at_once_under_the_cool_temperature(self):
        th, _ = self.thermal_with([52] * 5)
        r = th.settle(lambda: False)
        self.assertFalse(r['timed_out'])
        self.assertLess(r['seconds'], 4)

    def test_settle_waits_while_a_hot_box_still_cools(self):
        th, _ = self.thermal_with([70 - k / 10 for k in range(90)])
        self.assertTrue(th.settle(lambda: False)['timed_out'])

    def test_guard_holds_a_hot_box_once(self):
        th = thermal.Thermal(['local'], log=lambda m: None)
        th.clock_control = False
        held = []
        th._clock = lambda h, mhz: held.append((h.name, mhz))
        th.guarding = True

        class Proc:
            stdout = iter(['88, 2400, 60, 99, 0x0', '90, 2400, 60, 99, 0x0', '91, 2400, 60, 99, 0x0'])
        th.hosts[0].proc = Proc()
        th._read(th.hosts[0])
        time.sleep(0.1)
        self.assertEqual([t['temp'] for t in th.trips], [90])
        self.assertEqual(held, [('local', 1800)])

    def test_settle_gives_up_on_a_busy_server(self):
        th, _ = self.thermal_with([50] * 90)
        r = th.settle(lambda: True)
        self.assertTrue(r['timed_out'])


class Verdicts(unittest.TestCase):
    def rep(self, tps, problems=()):
        runs = [{'i': i, 'tps': tps * (1 + i / 100), 'tokens': 500, 'decode_s': 5, 'tok_per_step': 3, 'ms_per_step': 40}
                for i in range(8)]
        return {'decode': {'code': {'runs': runs, 'problems': list(problems)}},
                'prefill': {'8K': {'tok_s': tps * 30}}, 'concurrency': {}}

    def test_slower_decode_and_prefill_fail(self):
        base, cur = self.rep(100), self.rep(90)
        checks = {c: v for c, v, _ in report.verdicts(cur, report.compare(cur, base), None, base)}
        self.assertEqual(checks['decode code vs baseline'], 'FAIL')
        self.assertEqual(checks['prefill 8K vs baseline'], 'FAIL')

    def test_within_noise_passes(self):
        base, cur = self.rep(100), self.rep(99)
        checks = {c: v for c, v, _ in report.verdicts(cur, report.compare(cur, base), None, base)}
        self.assertEqual(checks['decode code vs baseline'], 'PASS')
        self.assertEqual(checks['prefill 8K vs baseline'], 'PASS')

    def test_failed_runs_fail(self):
        cur = self.rep(100, ['run 3: wrong count'])
        checks = {c: v for c, v, _ in report.verdicts(cur, report.compare(cur, None), None)}
        self.assertEqual(checks['decode code'], 'FAIL')


class Stages(Base):
    def ctx(self):
        class A:
            series, effort, kl_ref = 'test', 'low', None
        return S.Ctx(self.cl, A(), lambda m: None)

    def test_needle_pass_and_fail(self):
        self.assertTrue(S.needle(self.ctx(), [3072], None)['3K']['passed'])
        self.srv.st.retrieve = False
        self.assertFalse(S.needle(self.ctx(), [3072], None)['3K']['passed'])

    def test_needle_over_the_context_is_skipped(self):
        self.assertIn('skipped', S.needle(self.ctx(), [8192], 4096)['8K'])

    def test_decode_pairs_by_run_tag(self):
        a = S.decode(self.ctx(), ['code'], 3, 0)['code']
        b = S.decode(self.ctx(), ['code'], 3, 0)['code']
        self.assertEqual([r['tag'] for r in a['runs']], [r['tag'] for r in b['runs']])
        self.assertEqual(a['summary']['runs'], 3)
        self.assertGreater(a['summary']['tok_per_step'], 1)

    def test_failed_stream_is_counted(self):
        self.srv.st.fail_nth = 2
        r = S.concurrency(self.ctx(), [2], 1)['2']
        self.assertEqual(r['failed'], 1)
        self.assertIsNone(r['tps'])

    def test_cached_tokens(self):
        self.srv.st.cache_hit = 100
        self.assertEqual(S.cached(self.ctx())['second'], 100)


class EndToEnd(Base):
    def test_quick_then_compare(self):
        p, a = self.bench('--quick', '--runs', '3', '--streams', '1,2', '--prefill-sizes', '2048')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('RESULT: PASS', p.stdout)
        self.assertEqual(set(a['decode']), {'code', 'prose'})
        self.assertEqual([r['kind'] for r in a['requests'][:2]], ['code', 'code'])   # route_bytes.py --drive order
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(a, f)
        p, b = self.bench('--quick', '--runs', '3', '--streams', '1,2', '--prefill-sizes', '2048', '--baseline', f.name)
        self.assertIn('decode code vs baseline', p.stdout)
        self.assertIn('code', b['compare']['decode'])

    def test_other_series_is_refused(self):
        p, a = self.bench('--quick', '--runs', '2', '--streams', '1', '--prefill-sizes', '2048')
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(a, f)
        p, _ = self.bench('--quick', '--series', 'other', '--baseline', f.name)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('would not pair', p.stderr)


if __name__ == '__main__':
    unittest.main()
