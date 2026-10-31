"""Unit test of the DSpark canary comparator (ds41_dspark_canary.py) on synthetic runs: no GPU, no model.
  usage: cd engine/tests && python3 -m unittest -q test_ds41_dspark_canary"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import ds41_dspark_canary as C

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'ds41_dspark_canary.json'


def tokens_for(pid, n=40):
    base = sum(map(ord, pid))
    return [(base * 31 + 7 * i) % 100000 for i in range(n)]


def write_run(path, fx, mtp, engram=1, label=None, edit=None, rounds=None, top2=None):
    """edit(pid, tokens) -> tokens lets a test change one prompt; rounds: {pid: [(k, acc)]}; top2: {pid: [(i, gap)]}."""
    lines = [{'type': 'hdr', 'mtp': mtp, 'engram': engram, 'temp': 0, 'label': label or ('real' if engram else 'diag')}]
    for p in fx['prompts']:
        toks = tokens_for(p['id'])
        if edit:
            toks = edit(p['id'], toks)
        lines.append({'type': 'gen', 'id': p['id'], 'class': p['class'], 'tokens': toks})
    for pid, rs in (rounds or {}).items():
        lines += [{'type': 'round', 'id': pid, 'k': k, 'acc': a} for k, a in rs]
    for pid, ts in (top2 or {}).items():
        lines += [{'type': 'top2', 'id': pid, 'i': i, 'gap': g} for i, g in ts]
    Path(path).write_text('\n'.join(json.dumps(x) for x in lines) + '\n')


class CanaryTest(unittest.TestCase):
    def setUp(self):
        self.fx = C.load_fixture(FIXTURE)
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_path(self, name):
        return self.d / (name + '.jsonl')

    def load(self, name, mtp):
        return C.load_run(self.run_path(name), self.fx, mtp)

    def report(self, runs):
        return C.build_report(self.fx, runs)

    # ---- fixture
    def test_fixture_shape(self):
        self.assertEqual(self.fx['classes'], ['prose', 'code', 'json', 'arith'])
        self.assertEqual(len(self.fx['prompts']), 12)
        for c in self.fx['classes']:
            self.assertEqual(sum(p['class'] == c for p in self.fx['prompts']), 3, c)
        self.assertEqual(self.fx['settings'], {'max_tokens': 256, 'temperature': 0, 'thinking': False})

    def test_fixture_rejects_bad(self):
        for edit, why in ((lambda f: f['settings'].update(temperature=0.7), 'temperature'),
                          (lambda f: f['prompts'][1].update(id=f['prompts'][0]['id']), 'duplicate'),
                          (lambda f: f['prompts'][0].update({'class': 'poetry'}), 'class')):
            fx = json.loads(FIXTURE.read_text())
            edit(fx)
            p = self.d / 'bad.json'
            p.write_text(json.dumps(fx))
            with self.assertRaisesRegex(C.InputError, why):
                C.load_fixture(p)

    # ---- gates
    def test_all_equal(self):
        for name, mtp in (('a', 0), ('b', 0), ('c', 1), ('d', 5)):
            write_run(self.run_path(name), self.fx, mtp, rounds=None if mtp == 0 else {'prose-1': [(mtp, 1)]})
        res = self.report({'mtp0': self.load('a', 0), 'mtp0b': self.load('b', 0), 'mtp1': self.load('c', 1), 'mtp5': self.load('d', 5)})
        self.assertEqual([g['name'] for g in res['gates']], ['G0', 'G1', 'G2'])
        self.assertTrue(all(g['equal'] for g in res['gates']))
        self.assertEqual(res['verdict'], 'EQUAL')
        self.assertTrue(res['pass'])
        self.assertEqual(res['label'], 'real')

    def test_diag_label(self):
        for name, mtp in (('a', 0), ('d', 5)):
            write_run(self.run_path(name), self.fx, mtp, engram=0)
        res = self.report({'mtp0': self.load('a', 0), 'mtp5': self.load('d', 5)})
        self.assertEqual(res['label'], 'diag')
        self.assertEqual(res['verdict'], 'EQUAL (diag: engine consistency only)')
        self.assertTrue(res['pass'])
        self.assertIn('label: diag', C.format_report(res))

    def test_divergence_reported_with_gap(self):
        write_run(self.run_path('a'), self.fx, 0, top2={'code-2': [(17, 0.0031)]})
        write_run(self.run_path('d'), self.fx, 5, edit=lambda pid, t: t[:17] + [t[17] + 1] + t[18:] if pid == 'code-2' else t)
        res = self.report({'mtp0': self.load('a', 0), 'mtp5': self.load('d', 5)})
        g2 = res['gates'][0]
        self.assertFalse(g2['equal'])
        self.assertEqual(res['verdict'], 'DIFF at G2 (1 of 12 prompts)')
        self.assertFalse(res['pass'])
        bad = [r for r in g2['prompts'] if not r['equal']]
        self.assertEqual([r['id'] for r in bad], ['code-2'])
        t = tokens_for('code-2')
        self.assertEqual(bad[0]['div'], {'i': 17, 'tok_a': t[17], 'tok_b': t[17] + 1, 'top2_gap': 0.0031})
        text = C.format_report(res)
        self.assertIn('first divergence at token 17: %d vs %d, top2 gap 0.0031' % (t[17], t[17] + 1), text)

    def test_length_difference_is_a_divergence(self):
        write_run(self.run_path('a'), self.fx, 0)
        write_run(self.run_path('d'), self.fx, 5, edit=lambda pid, t: t[:30] if pid == 'arith-3' else t)
        res = self.report({'mtp0': self.load('a', 0), 'mtp5': self.load('d', 5)})
        row = [r for r in res['gates'][0]['prompts'] if r['id'] == 'arith-3'][0]
        self.assertEqual(row['div']['i'], 30)
        self.assertEqual(row['div']['tok_b'], None)
        self.assertEqual(row['div']['tok_a'], tokens_for('arith-3')[30])

    def test_lower_gate_failure_makes_higher_meaningless(self):
        write_run(self.run_path('a'), self.fx, 0)
        write_run(self.run_path('b'), self.fx, 0, edit=lambda pid, t: t[:5] + [1] + t[6:] if pid == 'prose-1' else t)
        write_run(self.run_path('d'), self.fx, 5)
        res = self.report({'mtp0': self.load('a', 0), 'mtp0b': self.load('b', 0), 'mtp5': self.load('d', 5)})
        self.assertEqual([g['meaningless'] for g in res['gates']], [False, True])
        self.assertEqual(res['verdict'], 'DIFF at G0 (G2 meaningless)')
        self.assertIn('[meaningless: G0 failed]', C.format_report(res))

    def test_g2_missing_is_incomplete(self):
        write_run(self.run_path('a'), self.fx, 0)
        write_run(self.run_path('b'), self.fx, 0)
        res = self.report({'mtp0': self.load('a', 0), 'mtp0b': self.load('b', 0)})
        self.assertTrue(res['verdict'].startswith('INCOMPLETE'))
        self.assertFalse(res['pass'])

    # ---- acceptance
    def test_acceptance_numbers(self):
        # prose class gets rounds (k, acc): hand-computed below. P_r = #(acc>r) / #(k>r and acc>=r).
        rounds = {'prose-1': [(5, 0), (5, 3), (5, 5), (5, 2)], 'prose-2': [(5, 5), (5, 0)], 'code-1': [(5, 1)], 'json-1': [(3, 3), (3, 1)]}
        write_run(self.run_path('a'), self.fx, 0)
        write_run(self.run_path('d'), self.fx, 5, rounds=rounds)
        res = self.report({'mtp0': self.load('a', 0), 'mtp5': self.load('d', 5)})
        a = res['acceptance']['mtp5']
        prose = a['prose']
        self.assertEqual(prose['rounds'], 6)
        self.assertEqual(prose['hist'], [2, 0, 1, 1, 0, 2])
        self.assertAlmostEqual(prose['mean_a'], 15 / 6)
        self.assertAlmostEqual(prose['tokens_per_round'], 1 + 15 / 6)
        # acc values 0,3,5,2,5,0. r=0: 6 reached, 4 took (acc>0); r=1: reached acc>=1 = 4, took acc>1 = 4
        self.assertAlmostEqual(prose['per_pos'][0], 4 / 6)
        self.assertAlmostEqual(prose['per_pos'][1], 4 / 4)
        self.assertAlmostEqual(prose['per_pos'][2], 3 / 4)   # acc>=2: 3,5,2,5 -> 4 reached; acc>2: 3,5,5 -> 3 took
        self.assertAlmostEqual(prose['per_pos'][3], 2 / 3)   # reached acc>=3: 3,5,5; took acc>3: 5,5 -> 2/3
        self.assertAlmostEqual(prose['per_pos'][4], 2 / 2)   # reached acc>=4: 5,5; took acc>4: 5,5
        # json class has k=3 rounds: draft positions 3 and 4 were never proposed there, so they do not count as reached
        js = a['json']
        self.assertEqual(js['rounds'], 2)
        self.assertEqual(js['hist'], [0, 1, 0, 1, 0, 0])
        self.assertAlmostEqual(js['per_pos'][0], 2 / 2)
        self.assertAlmostEqual(js['per_pos'][2], 1 / 1)
        self.assertIsNone(js['per_pos'][3])
        # a class without rounds has no mean
        self.assertEqual(a['arith']['rounds'], 0)
        self.assertIsNone(a['arith']['mean_a'])
        # 'all' pools the classes
        self.assertEqual(a['all']['rounds'], 6 + 1 + 2)
        text = C.format_report(res)
        self.assertIn('acceptance, MTP=5 run:', text)
        self.assertIn('2/0/1/1/0/2', text)

    # ---- bad input
    def test_bad_inputs(self):
        write_run(self.run_path('a'), self.fx, 0)
        with self.assertRaisesRegex(C.InputError, 'expected mtp=5'):
            C.load_run(self.run_path('a'), self.fx, 5)
        write_run(self.run_path('t'), self.fx, 0)
        lines = self.run_path('t').read_text().replace('"temp": 0', '"temp": 0.7')
        self.run_path('t').write_text(lines)
        with self.assertRaisesRegex(C.InputError, 'temp must be 0'):
            self.load('t', 0)
        write_run(self.run_path('l'), self.fx, 0, engram=0, label='real')
        with self.assertRaisesRegex(C.InputError, 'label must be'):
            self.load('l', 0)
        write_run(self.run_path('r'), self.fx, 0, rounds={'prose-1': [(5, 1)]})
        with self.assertRaisesRegex(C.InputError, 'MTP=0 run must not carry round'):
            self.load('r', 0)
        write_run(self.run_path('k'), self.fx, 1, rounds={'prose-1': [(5, 1)]})
        with self.assertRaisesRegex(C.InputError, 'exceeds mtp=1'):
            self.load('k', 1)
        write_run(self.run_path('ov'), self.fx, 5, rounds={'prose-1': [(3, 4)]})
        with self.assertRaisesRegex(C.InputError, 'acc <= k'):
            self.load('ov', 5)
        write_run(self.run_path('cap'), self.fx, 0, edit=lambda pid, t: t + t * 6 if pid == 'code-3' else t)
        with self.assertRaisesRegex(C.InputError, 'max_tokens'):
            self.load('cap', 0)
        lines = self.run_path('a').read_text().splitlines()
        self.run_path('m').write_text('\n'.join(lines[:-1]) + '\n')
        with self.assertRaisesRegex(C.InputError, 'missing'):
            self.load('m', 0)
        (self.run_path('j')).write_text('not json\n')
        with self.assertRaisesRegex(C.InputError, 'not a JSON object'):
            self.load('j', 0)

    def test_engram_mismatch_refused(self):
        write_run(self.run_path('a'), self.fx, 0, engram=1)
        write_run(self.run_path('d'), self.fx, 5, engram=0)
        with self.assertRaisesRegex(C.InputError, 'disagree on engram'):
            self.report({'mtp0': self.load('a', 0), 'mtp5': self.load('d', 5)})

    # ---- command line
    def cli(self, *args):
        return subprocess.run([sys.executable, '-I', str(HERE / 'ds41_dspark_canary.py'), *args], capture_output=True, text=True)

    def test_cli_exit_codes_and_json(self):
        write_run(self.run_path('a'), self.fx, 0)
        write_run(self.run_path('d'), self.fx, 5, rounds={'prose-1': [(5, 2)]})
        out = self.d / 'res.json'
        r = self.cli('--mtp0', str(self.run_path('a')), '--mtp5', str(self.run_path('d')), '--json', str(out))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('VERDICT: EQUAL', r.stdout)
        self.assertEqual(json.loads(out.read_text())['verdict'], 'EQUAL')
        write_run(self.run_path('x'), self.fx, 5, edit=lambda pid, t: t[:3] + [0] + t[4:] if pid == 'json-1' else t)
        r = self.cli('--mtp0', str(self.run_path('a')), '--mtp5', str(self.run_path('x')))
        self.assertEqual(r.returncode, 1)
        self.assertIn('DIFF at G2', r.stdout)
        r = self.cli('--mtp0', str(self.run_path('a')), '--mtp5', str(self.d / 'nope.jsonl'))
        self.assertEqual(r.returncode, 2)
        self.assertIn('bad input', r.stderr)
        r = self.cli('--mtp0', str(self.run_path('a')))
        self.assertEqual(r.returncode, 1)
        self.assertIn('INCOMPLETE', r.stdout)


if __name__ == '__main__':
    unittest.main()
