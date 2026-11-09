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
from gr8bench import agent_tools, client as C, docs, kindling, kl, report, stages as S, stats, thermal  # noqa: E402

BENCH = HERE.parent / 'bench.py'
GIF = HERE.parent / 'gif.py'


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
        self.srv.st.needle_cutoff = False

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

    def test_needle_answered_and_cut_off(self):
        ok = S.needle(self.ctx(), [3072], None)['3K']
        self.assertEqual((ok['passed'], ok['cut_off'], ok['finish']), (True, False, 'stop'))
        self.assertLess(ok['tokens'], S.NEEDLE_TOKENS)
        self.assertIn(ok['secret'], ok['content'])
        self.srv.st.needle_cutoff = True
        ctx = self.ctx()
        ctx.a.needle_seed = ok['seed']        # the same document and secret again
        cut = S.needle(ctx, [3072], None)['3K']
        self.assertEqual(cut['secret'], ok['secret'])
        self.assertEqual((cut['passed'], cut['cut_off'], cut['in_reasoning_only']), (False, True, True))
        self.assertEqual((cut['finish'], cut['tokens'], cut['content']), ('length', S.NEEDLE_TOKENS, ''))
        self.assertIn(cut['secret'], cut['reasoning'])
        self.assertLess(len(cut['reasoning']), 2 * S.KEEP_CHARS + 100)
        rep = {'correct': {'needle': {'3K': cut}}}
        checks = {c: (v, d) for c, v, d in report.verdicts(rep, report.compare(rep, None), None)}
        verdict, detail = checks['needle 3K']
        self.assertEqual(verdict, 'FAIL')
        self.assertIn(f'found in the reasoning, answer cut off at {S.NEEDLE_TOKENS} tokens', detail)
        self.assertIn(f"seed {cut['seed']}", detail)
        q, _ = kindling.fields({'correct': {'needle': {'3K': cut, '8K': ok}}})
        self.assertEqual(q['quality'], 'found at 8K; found at 3K (answer cut off)')

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


class Record(Base):
    def test_concurrency_streams_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            p, rep = self.bench('--stages', 'info,concurrency', '--streams', '1,2', '--label', 'rec', '--record', d)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ['concurrency-c1.json', 'concurrency-c2.json'])
            r = json.loads((Path(d) / 'concurrency-c2.json').read_text())
        self.assertEqual(r['format'], 'gr8bench-record-1')
        m = r['meta']
        self.assertEqual((m['model'], m['label'], m['health']), ('fake-model', 'rec', {'ready': True}))
        self.assertEqual(m['models'], ['fake-model'])
        self.assertEqual(r['streams'], 2)
        self.assertEqual(len(r['batches']), 2)          # the quick and default levels run two batches a level
        streams = r['batches'][0]['streams']
        self.assertEqual([s['i'] for s in streams], [0, 1])
        for s in streams:
            self.assertGreaterEqual(s['t0'], 0)
            self.assertEqual(s['messages'][-1]['role'], 'user')
            kinds = [e[1] for e in s['events']]
            self.assertEqual(kinds[-2:], ['finish', 'usage'])
            text = ''.join(e[2] for e in s['events'] if e[1] == 'answer')
            self.assertEqual(len(text.split()), 512)
            times = [e[0] for e in s['events']]
            self.assertEqual(times, sorted(times))
            self.assertEqual(s['events'][-1][2]['completion_tokens'], s['tokens'])
        self.assertNotIn('recorded', rep['concurrency']['2']['batches'][0])   # the report itself is unchanged

    def test_no_record_no_files(self):
        b = S.conc_batch(Stages.ctx(self), 1, 0)
        self.assertNotIn('recorded', b)

    def test_agent_task_requests_in_order(self):
        reqs = []
        ok, st = agent_tools.run_task(self.url, 'fake-model', agent_tools.TASKS[2], True, False, reqs)
        self.assertTrue(ok)                              # the fake finishes with "3", the TODO count
        self.assertEqual([r['i'] for r in reqs], [0, 1])
        self.assertLess(reqs[0]['t0'], reqs[1]['t0'])
        self.assertEqual(reqs[1]['messages'][-1]['role'], 'tool')
        tools = [e[2] for e in reqs[0]['events'] if e[1] == 'tool']
        self.assertEqual(tools[0], {'index': 0, 'id': 'call_1', 'name': 'list_dir'})
        self.assertEqual(''.join(t.get('arguments', '') for t in tools), '{"path": "/app"}')
        self.assertEqual([e[1] for e in reqs[0]['events']][0], 'think')
        self.assertEqual([e[2] for e in reqs[0]['events'] if e[1] == 'finish'], ['tool_calls'])

    def test_agentic_stage_writes_one_file_per_task(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = Stages.ctx(self)
            from gr8bench import record
            ctx.recorder = record.Recorder(Path(d), {'model': 'fake-model'})
            agent_tools.run(self.url, 'fake-model', lambda task, reqs, ok: ctx.recorder.save(
                f'agentic-{task}', {'stage': 'agentic', 'task': task, 'passed': ok, 'requests': reqs}))
            names = sorted(x.name for x in Path(d).iterdir())
            self.assertEqual(names, sorted(f'agentic-{t[0]}.json' for t in agent_tools.TASKS))
            r = json.loads((Path(d) / 'agentic-count-todos.json').read_text())
        self.assertEqual((r['task'], len(r['requests'])), ('count-todos', 2))


class Gif(unittest.TestCase):
    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest('Pillow is not installed')

    def render(self, rec, *args):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / 'rec.json', Path(d) / 'rec.gif'
            src.write_text(json.dumps(rec))
            p = subprocess.run([sys.executable, str(GIF), str(src), '--out', str(out), '--fps', '5', '--hold', '1',
                                *args], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            from PIL import Image
            with Image.open(out) as im:
                return p.stdout, im.size, im.n_frames

    def test_concurrency_recording(self):
        def stream(i, t0):
            ev = [[0.3 + k * 0.05, 'think', 'hmm '] for k in range(5)]
            ev += [[0.6 + k * 0.05, 'answer', f'word{k} '] for k in range(30)]
            return {'i': i, 't0': t0, 'messages': [], 'tokens': 35,
                    'events': ev + [[2.2, 'finish', 'length'], [2.2, 'usage', {'completion_tokens': 35}]]}
        rec = {'format': 'gr8bench-record-1', 'meta': {'model': 'fake'}, 'stage': 'concurrency', 'streams': 4,
               'batches': [{'i': 0, 'start': 0, 'streams': [stream(i, 0.1 * i) for i in range(4)]}]}
        out, size, frames = self.render(rec, '--cols', '20', '--rows', '5', '--title', 'test')
        self.assertIn('4 panes in 2x2', out)
        self.assertGreater(frames, 5)

    def test_query_recording(self):
        rec = {'prompt': 'hi', 'events': [[0.1, 'think', 'Thinking about it. '], [0.5, 'answer', 'Hello there.']]}
        out, size, frames = self.render(rec)
        self.assertIn('1 pane in 1x1', out)

    def test_agentic_recording_with_tools(self):
        ev = [[0.1, 'think', 'Look first. '], [0.2, 'tool', {'index': 0, 'id': 'c', 'name': 'list_dir'}],
              [0.3, 'tool', {'index': 0, 'arguments': '{"path": "/app"}'}], [0.3, 'finish', 'tool_calls'],
              [0.3, 'usage', {'completion_tokens': 9}]]
        rec = {'format': 'gr8bench-record-1', 'stage': 'agentic', 'task': 'find-port',
               'requests': [{'i': 0, 't0': 0, 'messages': [], 'events': ev},
                            {'i': 1, 't0': 0.5, 'messages': [], 'events': ev}]}
        out, size, frames = self.render(rec, '--grid', '1x1')
        self.assertIn('1 pane in 1x1', out)

    def test_card_from_a_generated_kindling_json(self):
        doc, _, _ = kindling.update({'version': 1, 'title': 'Model', 'stack': ['gr8'], 'configs': []},
                                    Kindling.REPORT, 'tp1', 'TP=1', 1, True)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'kindling.json'
            src.write_text(json.dumps(doc))
            from PIL import Image
            for out, frames in (('card.png', 1), ('card.gif', 2)):
                p = subprocess.run([sys.executable, str(GIF), 'card', str(src), '--out', str(Path(d) / out),
                                    '--light'], capture_output=True, text=True, timeout=120)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                with Image.open(Path(d) / out) as im:
                    self.assertEqual(im.size[0], 800)
                    self.assertGreaterEqual(getattr(im, 'n_frames', 1), frames)
            p = subprocess.run([sys.executable, str(GIF), 'card', str(src), '--config', 'tp9'],
                               capture_output=True, text=True, timeout=120)
            self.assertIn("no config 'tp9'", p.stderr)

    def test_card_rows_only_hold_what_the_config_has(self):
        sys.path.insert(0, str(GIF.parent))
        import gif
        rows = gif.card_rows({'title': 'M'}, {'id': 'x', 'decode': {'code': 50.0}})
        self.assertEqual([r[0] for r in rows], ['title', 'section', 'bar'])
        self.assertEqual(rows[2][1:5], ('code', 50.0, 50.0, '50.0'))

    def test_card_rows_show_first_token_and_prompt_lengths(self):
        sys.path.insert(0, str(GIF.parent))
        import gif
        rows = gif.card_rows({'title': 'M'}, {'id': 'x', 'decode': {'code': 50.0}, 'decode_prompt_tokens': [65, 72],
                                              'concurrency': [{'streams': 16, 'aggregate_tps': 249.2,
                                                               'ttft_p50_s': 2.48}],
                                              'concurrency_prompt_tokens': [62, 76]})
        self.assertEqual(rows[1], ('section', 'Decode · tok/s, one stream, prompts 65-72 tokens'))
        self.assertEqual(rows[3], ('section', 'Concurrency · total tok/s, prompts 62-76 tokens'))
        self.assertEqual(rows[4][1:5], ('16 streams', 249.2, 249.2, '249.2  first token 2.48 s'))


class Kindling(unittest.TestCase):
    REPORT = {'info': {'model': 'm', 'max_model_len': 262144},
              'decode': {'code': {'summary': {'tps': 61.234}}, 'prose': {'summary': {'tps': 40.06}},
                         'structured': {'summary': None}},
              'prefill': {'8K': {'tok_s': 1234.6}, '32K': {'tok_s': 1100.2}, '128K': {'skipped': 'over'}},
              'concurrency': {'1': {'tps': 60.04}, '4': {'tps': 120.56}, '8': {'tps': None}},
              'correct': {'needle': {'128K': {'passed': True}, '32K': {'passed': True}}}}

    def test_new_file(self):
        doc, filled, skipped = kindling.update({'version': 1, 'configs': []}, self.REPORT, 'tp1', 'TP=1', 1, True)
        self.assertEqual(list(doc), ['version', 'default_config', 'configs'])
        e = doc['configs'][0]
        self.assertEqual(list(e), ['id', 'label', 'nodes', 'decode', 'prefill', 'concurrency', 'max_context',
                                   'quality'])
        self.assertEqual(e['decode'], {'code': 61.2, 'prose': 40.1})
        self.assertEqual(e['prefill'], [{'context': 8192, 'tps': 1235}, {'context': 32768, 'tps': 1100}])
        self.assertEqual(e['concurrency'], [{'streams': 1, 'aggregate_tps': 60.0},
                                            {'streams': 4, 'aggregate_tps': 120.6}])
        self.assertEqual(e['quality'], [{'name': 'Needle recall', 'value': 'found at 32K, 128K'}])
        self.assertIn('decode structured', skipped)

    def test_update_keeps_everything_else(self):
        old = {'version': 1, 'title': 'T', 'summary': 'S', 'default_config': 'tp4', 'configs': [
            {'id': 'tp4', 'label': 'TP=4', 'nodes': 4},
            {'id': 'tp1', 'label': 'one', 'nodes': 1, 'decode': {'code': 1.0, 'structured': 9.9},
             'kv_pool_tokens': 5, 'quality': [{'name': 'GSM8K', 'value': '90%'}], 'notes': 'kept'}]}
        before = json.dumps(old)
        rep = {k: v for k, v in self.REPORT.items() if k != 'prefill'}
        doc, _, skipped = kindling.update(json.loads(before), rep, 'tp1')
        self.assertEqual(doc['configs'][0], json.loads(before)['configs'][0])
        self.assertEqual(doc['default_config'], 'tp4')
        e = doc['configs'][1]
        self.assertEqual(list(e), ['id', 'label', 'nodes', 'decode', 'concurrency', 'max_context', 'kv_pool_tokens',
                                   'quality', 'notes'])
        self.assertEqual(e['label'], 'one')
        self.assertEqual(e['decode'], {'code': 61.2, 'structured': 9.9, 'prose': 40.1})
        self.assertEqual(e['quality'], [{'name': 'GSM8K', 'value': '90%'},
                                        {'name': 'Needle recall', 'value': 'found at 32K, 128K'}])
        self.assertEqual((e['kv_pool_tokens'], e['notes']), (5, 'kept'))
        self.assertIn('prefill', skipped)

    def test_command_round_trips_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            rep, out = Path(d) / 'r.json', Path(d) / 'kindling.json'
            rep.write_text(json.dumps(self.REPORT))
            cmd = [sys.executable, str(BENCH), 'kindling', str(rep), '--out', str(out), '--config-id', 'tp1',
                   '--label', 'TP=1', '--nodes', '1', '--default']
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('not in the report', p.stdout)
            first = out.read_text()
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(out.read_text(), first)          # a second run changes nothing
            self.assertEqual(json.loads(first)['configs'][0]['max_context'], 262144)
            self.assertTrue(first.startswith('{\n  "version": 1,'))


if __name__ == '__main__':
    unittest.main()
