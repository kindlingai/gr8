#!/usr/bin/env python3
"""GR8_MODEL: GLM-5.3-Flash's options, engine plan, variants and release check. No network, no GPU.
    python3 -m unittest discover -s tools/launch"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelset as M  # noqa: E402
import models  # noqa: E402
import options as opt  # noqa: E402
import supervisor as S  # noqa: E402
from test_variant import manifest  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'


def flash(*argv, peers=PEERS):
    o = opt.parse(['--no-config', '--peers', peers, *argv], {'GR8_MODEL': 'glm53f'})
    opt.validate(o)
    return o


class OptionTests(unittest.TestCase):
    def test_defaults(self):
        o = flash()
        self.assertEqual((models.of(o), o.get('GROUP'), S.engine_bin(o)), (models.GLM53F, 'gr8-glm53f', 'g6m4_flash'))
        self.assertEqual(flash('--group', 'mine').get('GROUP'), 'mine')
        glm = opt.parse(['--no-config', '--peers', PEERS], {})
        self.assertEqual((models.of(glm), glm.get('GROUP')), (models.GLM53, 'gr8-glm53'))

    def test_refusals(self):
        for argv, want in ((['--tp', '3'], 'want 4 or 2 (glm53f)'), (['--kv-format', 'int4'], 'fp8 KV build only'),
                           (['--dflash', '1'], 'no DFlash2 drafter'),
                           (['--drafter', 'https://huggingface.co/a/b'], 'no DFlash2 drafter')):
            with self.assertRaises(opt.OptionError) as e:
                flash(*argv)
            self.assertIn(want, str(e.exception))
        with self.assertRaises(opt.OptionError) as e:
            opt.parse(['--no-config', '--gr8-model', 'glm9'], {})
        self.assertIn('want glm53 or glm53f', str(e.exception))
        for knob in ('--reuse', '--sess'):
            with self.assertRaises(opt.OptionError) as e:
                opt.engine_plan(flash(knob, '2' if knob == '--sess' else '1'))
            self.assertIn('GLM-5.3-Flash', str(e.exception))

    def test_engine_plan(self):
        for tp, peers in (('4', PEERS), ('2', '10.0.0.1,10.0.0.2')):
            plan = opt.engine_plan(flash(peers=peers))
            self.assertEqual(plan['knobs']['SESS'], '1')
            self.assertIn('G6_CP=0', plan['env'])
            self.assertIn('G6_PF_RS=0', plan['env'])
            self.assertIn('G6_PF_STAGE_ASYNC=1' if tp == '4' else 'G9_KDA_SCAN=1', plan['env'])
            for a in ('--reuse-prefix', '--wide', '--pfv2', '--dflash-manifest', '--qo-manifest'):
                self.assertNotIn(a, plan['args'])
            self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], '1')
            self.assertEqual(plan['args'][plan['args'].index('--chunk') + 1], '1')
            self.assertIn('G6_PF_DSA_F16GEMM=1', plan['env'])

    def test_model_counts_in_the_canonical_options(self):
        glm = opt.parse(['--no-config', '--peers', PEERS], {})
        opt.validate(glm)
        c, _, _ = opt.canonical(glm, opt.engine_plan(glm), opt.split_list(PEERS))
        self.assertNotIn('model', c)
        c, _, _ = opt.canonical(flash(), opt.engine_plan(flash()), opt.split_list(PEERS))
        self.assertEqual(c['model'], 'glm53f')


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()

    def test_k4_is_the_default_variant(self):
        v2 = os.path.join(self.t, 'v2')
        for k in (2, 3, 4):
            manifest(os.path.join(v2, 'manifests', f'exlr8-manifest-k{k}.json'), f'k{k}', k)
        self.assertEqual([M.resolve_variant(None, v2, [], tp, models.GLM53F).name for tp in (2, 4)], ['k4', 'k4'])
        self.assertEqual(M.resolve_variant('k2', v2, [], 4, models.GLM53F).name, 'k2')
        with self.assertRaises(M.ModelError):
            M.resolve_variant('home', v2, [], 4, models.GLM53F)

    def test_config_names_the_architecture(self):
        models.check_config(models.GLM53F, None)
        models.check_config(models.GLM53F, self.t)                     # no config.json
        with open(os.path.join(self.t, 'config.json'), 'w') as f:
            json.dump({'architectures': ['Glm5NextForConditionalGeneration']}, f)
        models.check_config(models.GLM53F, self.t)
        with self.assertRaises(models.ModelMismatch) as e:
            models.check_config(models.GLM53, self.t)
        self.assertIn('./gr8 glm53f serves it', str(e.exception))


if __name__ == '__main__':
    unittest.main()
