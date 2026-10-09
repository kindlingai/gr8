"""K1REUSE CPU gate (port of the old-line tools/reuse/test_prefix.py, without its reuse_sim.cpp simulator).

test_unit: engine/tests/test_prefix_reuse.cpp under ASan/UBSan (LCP / strict / NOEND / miss reasons / the W44 and
  RGRID grid rules / prefill_rows replay).
test_integration_order: the g6m4.cu hooks are where prefix_reuse.h's contract needs them (the simulator in the old
  line assumed this order; here it is asserted directly).
"""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAN = ['g++', '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']


class PrefixReuseTests(unittest.TestCase):
    def test_unit(self):
        with tempfile.TemporaryDirectory() as d:
            exe = pathlib.Path(d) / 'test_prefix_reuse'
            subprocess.run(SAN + ['-I', str(ROOT / 'engine/runtime'),
                                  str(ROOT / 'engine/tests/test_prefix_reuse.cpp'), '-o', str(exe)], check=True)
            res = subprocess.run([str(exe)], text=True, capture_output=True)
            self.assertEqual(res.returncode, 0, res.stderr[-2000:] + res.stdout[-2000:])
            self.assertTrue(res.stdout.splitlines()[-1].startswith('PASS'), res.stdout)
            print(res.stdout.splitlines()[-1])

    def test_integration_order(self):
        s = (ROOT / 'engine/runtime/g6m4.cu').read_text()
        self.assertIn('reuse_prefix = false;', s)
        self.assertLess(s.index('reuse.begin(sp.prompt'), s.index('const int PS = prompt_steps(p)'))
        self.assertIn('reuse.begin(sp.prompt, MP ? sp.first : (int)sp.prompt.size() - 1, mtp > 0)', s)
        self.assertIn('if (!serve.empty() || s2mode || reuse_prefix)', s)   # prompt-step row counts 1..chunk
        self.assertIn('for (int a = skip; a < p.first; a += MP)', s)
        # RGRID: the reuse grid is ONE sub-batch (MP = --prefill = 4096 served), not 2 * MP (W44)
        self.assertIn('reuse.grid = al_ == 1 ? MP : 0;', s)
        self.assertNotIn('2 * MP : MP', s)
        self.assertEqual(s.count('reuse.prefill_rows('), 2)       # both prefill branches (two lanes, one sub-batch)
        for branch in ['reuse.prefill_rows(a, MA + MB, mtp > 0);', 'reuse.prefill_rows(a, M, mtp > 0);']:
            pos = s.index(branch)
            self.assertLess(s.rindex('pf_draft(', 0, pos), pos)   # after the sub-batch's MTP rows
        self.assertIn('reuse.main_rows(si.pos[0], M);', s)
        self.assertIn('reuse.main_rows(in->pos[0], M);', s)
        self.assertIn('reuse.draft_rows(di.pos[0], M, chain)', s)
        self.assertIn('if (!skip)\n        {\n            reset(m, st);\n            if (mtp) reset(D.d, st);', s)
        # gate arm: poison only on a hit (the else of if (!skip)), only behind G6_REUSE_POISON=1, before the prefill loop
        pos = s.index('else if (reuse_poison_on && old_dirty > skip)')
        self.assertIn('reuse_poison(m, skip, old_dirty, st);', s)
        self.assertIn('const int old_end = reuse.written_end, old_dirty = reuse.dirty_end;', s)
        self.assertLess(s.index('old_dirty = reuse.dirty_end;'), s.index('reuse.begin(sp.prompt'))   # read before begin() resets it
        self.assertLess(s.index('if (!skip)\n        {\n            reset(m, st);'), pos)
        self.assertLess(pos, s.index('for (int a = skip; a < p.first; a += MP)'))
        self.assertIn('reuse_poison_on = po_ == 1;', s)
        for mtp in [True, False]:
            call = 'reuse.finish(p.prompt, gen, em.on ? em.emitted : (int)gen.size(), ' + str(mtp).lower() + ');'
            pos = s.index(call)
            self.assertLess(pos, s.index('em.end(', pos))
        # the one-request loop refuses DFlash2 with reuse (the drafter's prompt taps come from the prefill a hit
        # skips); --s2 keeps retired runs in its pool (s2/pool.h) and gives DFlash2 only to a session that owns them
        self.assertIn('if (reuse_prefix && dfl && !s2mode) die(', s)
        self.assertNotIn('if (reuse_prefix && s2mode) die(', s)


if __name__ == '__main__':
    unittest.main()
