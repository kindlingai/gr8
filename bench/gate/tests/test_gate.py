"""Harness tests against the fake server: python -m unittest discover bench/gate/tests (no GPU, no model)."""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import client as C  # noqa: E402
import docs  # noqa: E402
import fake_server  # noqa: E402
import kl  # noqa: E402
import measure  # noqa: E402
import record  # noqa: E402

GATE = HERE.parent / 'gate.py'


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
        seqs = kl.make_reference(self.cl, ['alpha beta gamma', 'delta epsilon'], n_tokens=12, k=5)
        ref = {'sequences': seqs, 'commit': 'abc', 'topology': 'tp6'}
        self.assertEqual([len(s['pos']) for s in seqs], [12, 12])
        r = kl.compare(self.cl, ref, 5)
        self.assertAlmostEqual(r['mean'], 0.0, places=9)
        self.assertEqual(r['top1'], 1.0)
        self.assertEqual(r['positions'], 24)
        self.srv.st.drift = 2.0
        d = kl.compare(self.cl, ref, 5)
        self.assertGreater(d['mean'], 0.01)
        self.assertLess(d['top1'], 1.0)
        self.assertGreaterEqual(d['max'], d['p95'])

    def test_server_without_prompt_logprobs_is_an_error(self):
        class NoLp:
            def complete(self, *a, **k):
                return {'choices': [{'prompt_logprobs': None}]}
        with self.assertRaises(RuntimeError):
            kl.score(NoLp(), [1, 2, 3], 20)


class Docs(Base):
    def test_size_and_depth(self):
        msgs, info = docs.needle_prompt(self.cl, 5000, 0.15, 7, 'amber-cedar-ember-1234')
        self.assertLess(abs(info['doc_tokens'] - 5000) / 5000, 0.03)
        self.assertLess(abs(info['needle_depth'] - 0.15), 0.03)
        lines = msgs[0]['content'].splitlines()
        self.assertEqual(len([ln for ln in lines if ln.startswith('# PASSPHRASE:')]), 1)

    def test_salt_differs_per_seed(self):
        a = docs.filler_prompt(self.cl, 500, 1, 'x')[0]['content']
        b = docs.filler_prompt(self.cl, 500, 2, 'x')[0]['content']
        self.assertNotEqual(a.splitlines()[0], b.splitlines()[0])


class Measurements(Base):
    def test_needle_pass_and_fail(self):
        r = measure.needle(self.cl, [3000], 5)['3k']
        self.assertTrue(r['passed'])
        self.srv.st.retrieve = False
        self.assertFalse(measure.needle(self.cl, [3000], 5)['3k']['passed'])

    def test_long_prompt_fields_and_prefix_note(self):
        r = measure.long_prompt(self.cl, [2000], 3, decode_tokens=120)['2k']
        self.assertGreater(r['prefill_tok_s'], 0)
        self.assertGreater(r['decode_tok_s'], 0)
        self.assertNotIn('note', r)
        self.srv.st.cache_hit = 1500
        r = measure.long_prompt(self.cl, [2000], 3, decode_tokens=120)['2k']
        self.assertIn('prefix cache hit', r['note'])

    def test_workloads_and_spec(self):
        w = measure.workloads(self.cl, runs=2, tokens=100)
        self.assertEqual(set(w), {'code', 'prose', 'structured'})
        sp = w['code']['spec']
        self.assertEqual(sp['drafts'], 2 * (100 // 3))
        self.assertAlmostEqual(sp['acceptance'], 50 / 100, places=3)
        self.assertAlmostEqual(sp['tokens_per_step'], 1 + 50 / 33, places=3)
        self.assertEqual(len(sp['per_pos_survival']), 2)

    def test_mixed4(self):
        m = measure.mixed4(self.cl, runs=2, tokens=100)
        self.assertEqual(len(m['batches']), 2)
        self.assertEqual(len(m['batches'][0]['streams']), 4)
        self.assertGreater(m['agg_tok_s'], m['stream_tok_s'])

    def test_failed_mixed4_stream_leaves_no_numbers(self):
        self.srv.st.fail_nth = 2
        m = measure.mixed4(self.cl, runs=2, tokens=100)
        self.assertEqual(m['failed_streams'], 1)
        self.assertIsNone(m['agg_tok_s'])
        self.assertIsNone(m['stream_tok_s'])
        self.assertFalse(m['batches'][0]['ok'] and m['batches'][1]['ok'])
        flat = record.flatten({'mixed4': m})
        self.assertNotIn('work.mixed4.agg_tok_s', flat)
        self.assertNotIn('work.mixed4.stream_tok_s', flat)

    def test_failed_workload_run_has_no_decode_number(self):
        self.srv.st.fail_nth = 2
        w = measure.workloads(self.cl, runs=2, tokens=100)
        self.assertEqual(w['code']['failed_runs'], 1)
        self.assertIsNone(w['code']['decode_tok_s'])
        self.assertIsNotNone(w['prose']['decode_tok_s'])

    def test_pbench_runs(self):
        results = measure.PBENCH.with_name('results.jsonl')
        size = results.stat().st_size if results.exists() else None
        try:
            s = measure.pbench(self.url, [200, 400], 1, ['--decode-tokens', '60'])
        finally:  # pbench appends to results.jsonl next to itself; leave it as we found it
            if size is None:
                results.unlink(missing_ok=True)
            else:
                os.truncate(results, size)
        self.assertIn('prefill_200', s)
        self.assertIn('prose_decode', s)


SETTINGS = {k: 1 for k in record.MEASURE_KEYS}
HEALTHY = {'format': 1, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'admitted': True, 'elapsed_s': 31.2, 'refusals': [], 'ranks': [{'rank': 0}]}
SICK = {'format': 1, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'admitted': False, 'elapsed_s': 30.0, 'ranks': [{'rank': 3}],
        'refusals': [{'rank': 3, 'host': 'n3', 'code': 'LINK_SLOW', 'detail': 'RDMA write rail 0: 13 Gb/s'}]}




def rec(boot, metrics, topo='tp6', session=None, commit='c1', flags='G6_NR=6', kl_ref=None, settings=None, env='',
        health=HEALTHY):
    results = {'kl': {'reference_commit': kl_ref[0], 'reference_topology': kl_ref[1]}} if kl_ref else {}
    return {'session': session or boot, 'boot': boot, 'topology': topo, 'commit': commit, 'build_flags': flags, 'engine_env': env,
            'settings': settings or SETTINGS, 'results': results, 'rank_health': health,
            'metrics': {k: {'value': v, 'better': record.LOWER if k.endswith(('ttft_s', 'kl.mean', 'kl.p95'))
                            else record.INFO if k.startswith('long.') and k.endswith('decode_tok_s')
                            else record.HIGHER} for k, v in metrics.items()}}


def baseline(runs, **kw):
    return record.build_baseline(runs, kw.pop('topo', 'tp6'), kw.pop('commit', 'c1'), **kw)


class Records(unittest.TestCase):
    def test_spread(self):
        sp = record.spread([rec('a', {'x': 100.0}), rec('a', {'x': 103.0})])
        self.assertEqual(sp['x']['min'], 100.0)
        self.assertAlmostEqual(sp['x']['spread_pct'], 2.96, places=2)

    def test_baseline_kl_threshold_needs_three_boots(self):
        two = [rec('b1', {'kl.mean': 0.004, 'a.decode_tok_s': 40.0}), rec('b2', {'kl.mean': 0.0, 'a.decode_tok_s': 41.0})]
        self.assertIsNone(baseline(two)['thresholds']['kl_mean'])
        three = two + [rec('b3', {'kl.mean': 0.016, 'a.decode_tok_s': 39.0})]
        b = baseline(three, kl_margin=2.0)
        self.assertAlmostEqual(b['thresholds']['kl_mean'], 0.032)
        self.assertEqual(b['metrics']['a.decode_tok_s']['median'], 40.0)
        self.assertEqual(b['metrics']['a.decode_tok_s']['boots'], 3)

    def test_kl_threshold_has_a_floor(self):
        runs = [rec(b, {'kl.mean': 0.0}) for b in ('b1', 'b2', 'b3')]
        self.assertEqual(baseline(runs)['thresholds']['kl_mean'], record.KL_FLOOR)

    def test_baseline_takes_median_of_repeats_per_boot(self):
        runs = [rec('b1', {'m': 10.0}), rec('b1', {'m': 12.0}), rec('b1', {'m': 11.0}), rec('b2', {'m': 20.0})]
        self.assertEqual(baseline(runs)['boots']['b1']['metrics']['m'], 11.0)

    def test_check_rules(self):
        base = baseline([rec('b1', {'d.decode_tok_s': 40.0, 'x.ttft_s': 2.0, 'needle.87k.pass': 1.0,
                                                 'kl.mean': 0.004})])
        ok_rows, st, _ = record.check([rec('n', {'d.decode_tok_s': 39.5, 'x.ttft_s': 2.03, 'needle.87k.pass': 1.0,
                                              'kl.mean': 0.5})], base, required=None)
        self.assertEqual(st, 'PARTIAL')  # inside 3% noise; KL has no threshold yet
        self.assertIn('NO THRESHOLD', [r[4] for r in ok_rows])
        _, st, _ = record.check([rec('n', {'d.decode_tok_s': 38.0, 'x.ttft_s': 2.0, 'needle.87k.pass': 1.0,
                                        'kl.mean': 0.0})], base, required=None)
        self.assertEqual(st, 'FAIL')  # decode 5% down
        _, st, _ = record.check([rec('n', {'d.decode_tok_s': 40.0, 'x.ttft_s': 2.2, 'needle.87k.pass': 1.0,
                                        'kl.mean': 0.0})], base, required=None)
        self.assertEqual(st, 'FAIL')  # ttft 10% up
        _, st, _ = record.check([rec('n', {'d.decode_tok_s': 40.0, 'x.ttft_s': 2.0, 'needle.87k.pass': 0.0,
                                        'kl.mean': 0.0})], base, required=None)
        self.assertEqual(st, 'FAIL')  # needle lost
        _, st, _ = record.check([rec('n', {'d.decode_tok_s': 40.0, 'needle.87k.pass': 1.0, 'kl.mean': 0.0})], base, required=None)
        self.assertEqual(st, 'FAIL')  # metric missing

    def test_long_prompt_decode_is_reported_not_gated(self):
        base = baseline([rec('b1', {'long.87k.decode_tok_s': 100.0, 'work.code.decode_tok_s': 100.0})])
        rows, st, _ = record.check([rec('n', {'long.87k.decode_tok_s': 90.0, 'work.code.decode_tok_s': 100.0})], base,
                                   required=None)
        self.assertEqual(st, 'PASS')
        self.assertEqual({r[0]: r[4] for r in rows}['long.87k.decode_tok_s'], 'info')
        _, st, _ = record.check([rec('n', {'long.87k.decode_tok_s': 100.0, 'work.code.decode_tok_s': 90.0})], base,
                                required=None)
        self.assertEqual(st, 'FAIL')  # the workload decode still gates

    def test_baseline_uses_only_the_named_commit(self):
        runs = [rec('b1', {'m': 10.0}, commit='main1'), rec('b2', {'m': 11.0}, commit='main1'),
                rec('c1', {'m': 1.0}, commit='candidate')]
        b = baseline(runs, commit='main1')
        self.assertEqual(b['metrics']['m']['min'], 10.0)
        self.assertEqual(set(b['boots']), {'b1', 'b2'})

    def test_baseline_refuses_mixed_configuration(self):
        with self.assertRaisesRegex(ValueError, 'build flags'):
            baseline([rec('b1', {'m': 1.0}, flags='G6_NR=6'), rec('b2', {'m': 1.0}, flags='G6_NR=6 X=1')])
        with self.assertRaisesRegex(ValueError, 'measurement settings'):
            baseline([rec('b1', {'m': 1.0}), rec('b2', {'m': 1.0}, settings=dict(SETTINGS, long_sizes=2))])
        with self.assertRaisesRegex(ValueError, 'KL reference'):
            baseline([rec('b1', {'m': 1.0}, kl_ref=('r1', 'tp6')), rec('b2', {'m': 1.0}, kl_ref=('r2', 'tp6'))])
        with self.assertRaisesRegex(ValueError, 'KL reference is for tp4'):
            baseline([rec('b1', {'m': 1.0}, kl_ref=('r1', 'tp4'))])
        with self.assertRaisesRegex(ValueError, 'same metrics'):
            baseline([rec('b1', {'m': 1.0, 'n': 2.0}), rec('b2', {'m': 1.0})])
        with self.assertRaisesRegex(ValueError, 'no runs of commit'):
            baseline([rec('b1', {'m': 1.0})], commit='other')

    def test_baseline_refuses_mixed_engine_env_and_stores_it(self):
        with self.assertRaisesRegex(ValueError, 'engine environment'):
            baseline([rec('b1', {'m': 100.0}, env='G6_CP=0'), rec('b2', {'m': 50.0}, env='G6_CP=1')])
        b = baseline([rec('b1', {'m': 1.0}, env='G6_CP=1  G6_PF_OWN=3'), rec('b2', {'m': 1.0}, env='G6_PF_OWN=3 G6_CP=1')])
        self.assertEqual(b['engine_env'], 'G6_CP=1 G6_PF_OWN=3')  # order and spacing do not matter

    def test_check_refuses_other_engine_env(self):
        base = baseline([rec('b1', {'m': 1.0}, env='G6_CP=1')])
        with self.assertRaisesRegex(ValueError, 'engine environment'):
            record.check([rec('n', {'m': 1.0}, env='G6_CP=0')], base, required=None)
        with self.assertRaisesRegex(ValueError, 'engine environment'):
            record.check([rec('n', {'m': 1.0})], base, required=None)
        record.check([rec('n', {'m': 1.0}, env='G6_CP=1')], base, required=None)

    def test_check_refuses_runs_that_do_not_belong(self):
        base = baseline([rec('b1', {'m': 1.0}, kl_ref=('r1', 'tp6'))])
        with self.assertRaisesRegex(ValueError, 'run is tp4, baseline is tp6'):
            record.check([rec('n', {'m': 1.0}, topo='tp4')], base, required=None)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            record.check([rec('n', {'m': 1.0}, settings=dict(SETTINGS, effort=2))], base, required=None)
        with self.assertRaisesRegex(ValueError, 'KL reference'):
            record.check([rec('n', {'m': 1.0}, kl_ref=('other', 'tp6'))], base, required=None)
        with self.assertRaisesRegex(ValueError, 'without a KL reference'):
            record.check([rec('n', {'m': 1.0}, kl_ref=('r1', 'tp6'))], baseline([rec('b1', {'m': 1.0})]),
                         required=None)
        record.check([rec('n', {'m': 1.0}, kl_ref=('r1', 'tp6'))], base, required=None)  # the right one is fine

    def test_incomplete_baseline_is_partial_never_pass(self):
        base = baseline([rec('b1', {'needle.87k.pass': 1.0})])
        _, st, reasons = record.check([rec('n', {'needle.87k.pass': 1.0})], base)  # default required set
        self.assertEqual(st, 'PARTIAL')
        self.assertTrue(any('pbench.prose_decode' in r for r in reasons))
        self.assertIn('pbench.prose_decode', base['missing_from_gate_set'])

    def test_full_gate_set_passes(self):
        full = {k: 10.0 for k in record.REQUIRED}
        full['kl.mean'] = 0.001
        for k in full:
            if k.endswith('.pass'):
                full[k] = 1.0
        boots = [rec(b, dict(full, **{'kl.mean': kl}), kl_ref=('r1', 'tp6'))
                 for b, kl in (('b1', 0.0), ('b2', 0.004), ('b3', 0.016))]
        base = baseline(boots)
        self.assertEqual(base['missing_from_gate_set'], [])
        _, st, _ = record.check([rec('n', dict(full, **{'kl.mean': 0.01}), kl_ref=('r1', 'tp6'))], base)
        self.assertEqual(st, 'PASS')
        _, st, _ = record.check([rec('n', dict(full, **{'kl.mean': 0.5}), kl_ref=('r1', 'tp6'))], base)
        self.assertEqual(st, 'FAIL')
        two = baseline(boots[:2])  # two boots: no KL threshold, so no pass
        _, st, reasons = record.check([rec('n', full, kl_ref=('r1', 'tp6'))], two)
        self.assertEqual(st, 'PARTIAL')
        self.assertTrue(any('no threshold' in r for r in reasons))

    def test_check_uses_baseline_spread_when_wider_than_noise(self):
        base = baseline([rec('b1', {'d': 40.0}), rec('b2', {'d': 44.0})])  # ~9.5% spread
        _, st, _ = record.check([rec('n', {'d': 38.5})], base, required=None)
        self.assertEqual(st, 'PASS')  # 42 median, -8.3% is inside the 9.5% spread


class RankHealthRecords(unittest.TestCase):
    def test_baseline_refuses_boots_the_probe_refused(self):
        runs = [rec('b1', {'m': 10.0}), rec('b2', {'m': 11.0}, health=SICK)]
        with self.assertRaises(ValueError) as cm:
            baseline(runs)
        self.assertIn("'b2'", str(cm.exception))
        self.assertIn('rank 3 LINK_SLOW', str(cm.exception))

    def test_baseline_refuses_an_unprobed_boot(self):
        runs = [rec('b1', {'m': 10.0}), rec('b2', {'m': 11.0}, health=None)]
        with self.assertRaises(ValueError) as cm:
            baseline(runs)
        self.assertIn('no rank health probe', str(cm.exception))
        r = rec('b3', {'m': 1.0})
        del r['rank_health']                              # a record written before the probe existed
        with self.assertRaises(ValueError):
            baseline([r])

    def test_baseline_records_the_probe_time_per_boot(self):
        b = baseline([rec('b1', {'m': 10.0}), rec('b2', {'m': 11.0})])
        self.assertEqual(b['rank_health'], {'b1': HEALTHY['ts'], 'b2': HEALTHY['ts']})

    def test_check_refuses_unhealthy_and_unprobed_runs(self):
        base = baseline([rec('b1', {'m': 10.0})])
        record.check([rec('n', {'m': 10.0})], base, required=None)
        with self.assertRaises(ValueError) as cm:
            record.check([rec('n', {'m': 10.0}, health=SICK)], base, required=None)
        self.assertIn('rank 3 LINK_SLOW', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            record.check([rec('n', {'m': 10.0}, health=None)], base, required=None)
        self.assertIn('no rank health probe', str(cm.exception))


class CLI(Base):
    def run_gate(self, *args):
        if args[0] == 'run' and '--health' not in args and '--no-health' not in args:
            args = args + ('--health', self.health_file())
        return subprocess.run([sys.executable, str(GATE), *args], capture_output=True, text=True)

    def health_file(self, doc=None):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', d]))
        Path(f'{d}/rank_health.json').write_text(json.dumps(doc or HEALTHY))
        return f'{d}/rank_health.json'

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            runs, ref, base = f'{d}/runs.jsonl', f'{d}/ref.json', f'{d}/tp6.json'
            p = self.run_gate('kl-ref', '--url', self.url, '--topology', 'tp6', '--commit', 'abc1234', '--out', ref,
                              '--sequences', '3', '--tokens', '20')
            self.assertEqual(p.returncode, 0, p.stderr)
            for boot in ('b1', 'b2', 'b3'):
                p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'abc1234', '--boot', boot,
                                  '--kl-ref', ref, '--runs', runs, '--repeat', '2', '--skip', 'pbench',
                                  '--needle-sizes', '3000', '--long-sizes', '1500', '--runs-per-workload', '1',
                                  '--workload-tokens', '80', '--build-flags', 'G6_NR=6')
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn('spread %', p.stdout)
            with open(runs) as f:
                lines = [json.loads(ln) for ln in f]
            self.assertEqual(len(lines), 6)
            r = lines[0]
            for key in ('commit', 'topology', 'build_flags', 'settings', 'results', 'metrics', 'harness_commit'):
                self.assertIn(key, r)
            self.assertEqual(r['commit'], 'abc1234')
            self.assertAlmostEqual(r['metrics']['kl.mean']['value'], 0.0, places=9)
            self.assertEqual(r['metrics']['needle.3k.pass']['value'], 1.0)
            p = self.run_gate('baseline', '--topology', 'tp6', '--commit', 'zzz', '--runs', runs, '--out', base)
            self.assertNotEqual(p.returncode, 0)  # no runs of that commit
            self.assertIn('no runs of commit zzz', p.stderr)
            p = self.run_gate('baseline', '--topology', 'tp6', '--commit', 'abc1234', '--runs', runs, '--out', base)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('missing parts of the gate set', p.stdout)
            self.assertIsNotNone(json.loads(Path(base).read_text())['thresholds']['kl_mean'])
            p = self.run_gate('check', '--baseline', base, '--runs', runs, '--noise', '10')
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)  # small sizes: not the full gate set
            self.assertIn('RESULT: PARTIAL', p.stdout)
            self.assertIn('NOT IN BASELINE', p.stdout)
            p = self.run_gate('check', '--baseline', base, '--runs', runs, '--noise', '10', '--allow-partial')
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn('| metric | main (baseline)', p.stdout)
            self.assertIn('RESULT: PARTIAL', p.stdout)  # still labelled partial
            self.srv.st.retrieve = False
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'bad', '--boot', 'x',
                              '--runs', runs, '--repeat', '1', '--only', 'needle', '--needle-sizes', '3000',
                              '--long-sizes', '1500', '--runs-per-workload', '1', '--workload-tokens', '80')
            p = self.run_gate('check', '--baseline', base, '--runs', runs, '--allow-partial')
            self.assertEqual(p.returncode, 1)
            self.assertIn('RESULT: FAIL', p.stdout)

    def test_kl_ref_of_another_topology_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ref = f'{d}/ref.json'
            self.assertEqual(self.run_gate('kl-ref', '--url', self.url, '--topology', 'tp4', '--commit', 'a',
                                           '--out', ref, '--sequences', '1', '--tokens', '8').returncode, 0)
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'a', '--kl-ref', ref,
                              '--runs', f'{d}/r.jsonl', '--only', 'kl')
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('tp4 reference', p.stderr)
            self.assertFalse(Path(f'{d}/r.jsonl').exists())

    def test_check_refuses_other_topology(self):
        with tempfile.TemporaryDirectory() as d:
            runs, base = f'{d}/runs.jsonl', f'{d}/b.json'
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', runs,
                              '--repeat', '1', '--only', 'needle', '--needle-sizes', '2000')
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(self.run_gate('baseline', '--topology', 'tp6', '--commit', 'c', '--runs', runs,
                                           '--out', base).returncode, 0)
            self.run_gate('run', '--url', self.url, '--topology', 'tp4', '--commit', 'd', '--runs', runs,
                          '--repeat', '1', '--only', 'needle', '--needle-sizes', '2000')
            p = self.run_gate('check', '--baseline', base, '--runs', runs)  # last session is the tp4 run
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('check refused: run is tp4, baseline is tp6', p.stderr)

    def test_health_result_goes_into_every_record(self):
        with tempfile.TemporaryDirectory() as d:
            runs, hf = f'{d}/r.jsonl', f'{d}/rank_health.json'
            Path(hf).write_text(json.dumps(HEALTHY))
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', runs, '--repeat', '2',
                              '--only', 'needle', '--needle-sizes', '2000', '--health', hf)
            self.assertEqual(p.returncode, 0, p.stderr)
            lines = [json.loads(ln) for ln in Path(runs).read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            for r in lines:
                self.assertTrue(r['rank_health']['admitted'])
                self.assertEqual(r['rank_health']['file'], 'rank_health.json')
                self.assertNotIn('health', r['settings'])

    def test_refused_boot_is_not_measured_unless_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            runs, hf, base = f'{d}/r.jsonl', f'{d}/h.json', f'{d}/b.json'
            Path(hf).write_text(json.dumps(SICK))
            args = ['run', '--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', runs, '--repeat', '1',
                    '--only', 'needle', '--needle-sizes', '2000', '--health', hf]
            p = self.run_gate(*args)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('LINK_SLOW', p.stderr)
            self.assertFalse(Path(runs).exists())
            p = self.run_gate(*args, '--allow-unhealthy')
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertFalse(json.loads(Path(runs).read_text().splitlines()[0])['rank_health']['admitted'])
            p = self.run_gate('baseline', '--topology', 'tp6', '--commit', 'c', '--runs', runs, '--out', base)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('rank health probe refused', p.stderr)

    def test_health_file_must_be_a_probe_result(self):
        with tempfile.TemporaryDirectory() as d:
            Path(f'{d}/h.json').write_text('{"nope": 1}')
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', f'{d}/r.jsonl',
                              '--only', 'needle', '--health', f'{d}/h.json')
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('not a rank health result', p.stderr)

    def test_run_without_health_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            runs = f'{d}/r.jsonl'
            p = subprocess.run([sys.executable, str(GATE), 'run', '--url', self.url, '--topology', 'tp6', '--commit', 'c',
                                '--runs', runs, '--only', 'needle'], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('--health', p.stderr)
            self.assertFalse(Path(runs).exists())

    def test_no_health_runs_cannot_make_a_baseline_or_a_check(self):
        with tempfile.TemporaryDirectory() as d:
            runs, base = f'{d}/r.jsonl', f'{d}/b.json'
            args = ['--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', runs, '--repeat', '1', '--only', 'needle',
                    '--needle-sizes', '2000']
            p = self.run_gate('run', *args, '--no-health')
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIsNone(json.loads(Path(runs).read_text().splitlines()[0])['rank_health'])
            p = self.run_gate('baseline', '--topology', 'tp6', '--commit', 'c', '--runs', runs, '--out', base)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('no rank health probe', p.stderr)
            self.assertEqual(self.run_gate('run', *args).returncode, 0)          # a probed run: baseline works
            os.truncate(runs, 0)
            self.assertEqual(self.run_gate('run', *args).returncode, 0)
            self.assertEqual(self.run_gate('baseline', '--topology', 'tp6', '--commit', 'c', '--runs', runs,
                                           '--out', base).returncode, 0)
            self.assertEqual(self.run_gate('run', *args, '--no-health', '--boot', 'x').returncode, 0)
            p = self.run_gate('check', '--baseline', base, '--runs', runs, '--allow-partial')
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('check refused', p.stderr)
            self.assertIn('no rank health probe', p.stderr)

    def test_stale_health_result_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            old = dict(HEALTHY, ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 3 * 86400)))
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', f'{d}/r.jsonl',
                              '--only', 'needle', '--health', self.health_file(old))
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('min old', p.stderr)
            self.assertFalse(Path(f'{d}/r.jsonl').exists())
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'c', '--runs', f'{d}/r.jsonl',
                              '--only', 'needle', '--needle-sizes', '2000', '--repeat', '1', '--health-max-age-min', '100000',
                              '--health', self.health_file(old))
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_unknown_part_is_rejected(self):
        p = self.run_gate('run', '--url', self.url, '--topology', 'tp6', '--commit', 'x', '--only', 'bogus')
        self.assertNotEqual(p.returncode, 0)

    def test_kl_skipped_without_reference(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_gate('run', '--url', self.url, '--topology', 'tp4', '--commit', 'x', '--runs', f'{d}/r.jsonl',
                              '--repeat', '1', '--only', 'kl')
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('kl skipped', p.stdout)


if __name__ == '__main__':
    unittest.main()
