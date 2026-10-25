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
        for argv, want in ((['--tp', '6'], 'want 4, 3 or 2 (glm53f)'), (['--kv-format', 'int4'], 'fp8 KV build only'),
                           (['--dflash', '1'], 'no DFlash2 drafter'),
                           (['--drafter', 'https://huggingface.co/a/b'], 'no DFlash2 drafter')):
            with self.assertRaises(opt.OptionError) as e:
                flash(*argv)
            self.assertIn(want, str(e.exception))
        with self.assertRaises(opt.OptionError) as e:
            opt.parse(['--no-config', '--gr8-model', 'glm9'], {})
        self.assertIn('want glm53, glm53f, pseudo', str(e.exception))
        for argv, want in ((['--reuse', '1'], 'GLM-5.3-Flash has no REUSE'), (['--sess', '9'], 'serves 1 to 8 sessions'),
                           (['--sess', '0'], 'serves 1 to 8 sessions'),
                           (['--sess', '8', '--mtp', '4'], 'a round of 40 rows')):
            with self.assertRaises(opt.OptionError) as e:
                opt.engine_plan(flash(*argv))
            self.assertIn(want, str(e.exception))

    def test_engine_plan(self):
        for tp, peers in (('4', PEERS), ('2', '10.0.0.1,10.0.0.2')):
            plan = opt.engine_plan(flash(peers=peers))
            self.assertEqual(plan['knobs']['SESS'], '1')
            self.assertIn('G6_CP=0', plan['env'])
            self.assertIn('G6_PF_RS=0', plan['env'])
            self.assertIn('G6_PF_STAGE_ASYNC=1', plan['env'])
            (self.assertNotIn if tp == '4' else self.assertIn)('G9_KDA_SCAN=1', plan['env'])
            for a in ('--reuse-prefix', '--wide', '--pfv2', '--dflash-manifest', '--qo-manifest'):
                self.assertNotIn(a, plan['args'])
            self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], '1')
            self.assertEqual(plan['args'][plan['args'].index('--chunk') + 1], '1')
            self.assertIn('G6_PF_DSA_F16GEMM=1', plan['env'])

    def test_flash_cache(self):
        plan = opt.engine_plan(flash())
        self.assertFalse([e for e in plan['env'] if e.startswith('G6_FLASH_')])
        self.assertNotIn('--kv-store', plan['args'])
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--flash-state-budget-mib', '4096'))
        self.assertIn('G6_FLASH_CACHE=aligned', plan['env'])
        self.assertIn('G6_FLASH_STATE_BUDGET_MIB=4096', plan['env'])
        self.assertNotIn('--kv-store', plan['args'])
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', 'ssd:/tmp/fc,size=64G'))
        self.assertEqual(plan['args'][plan['args'].index('--kv-store') + 1], 'ssd:/tmp/fc,size=64G')
        plan = opt.engine_plan(flash('--flash-cache', 'terminal'))
        self.assertIn('G6_FLASH_CACHE=terminal', plan['env'])
        # the checkpoint store on a group lease (D-331 M8 (c)): the store as given, KV_SHARE_WHEN reaches the engine
        g = 'remote:192.0.2.5:7350,size=1G,group=flash'
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', g, '--kv-share-when', 'always'))
        self.assertEqual(plan['args'][plan['args'].index('--kv-store') + 1], g)
        self.assertIn('G6_KV_SHARE_WHEN=always', plan['env'])
        self.assertNotIn('G6_KV_SHARE_WHEN', ' '.join(opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', g))['env']))
        for argv, want in ((['--flash-cache', 'sessions'], 'want off, aligned or terminal'),
                           (['--flash-state-budget-mib', '0'], 'whole number of MiB'),
                           (['--kv-store', 'ssd:/tmp/fc,size=64G'], 'needs --flash-cache aligned or terminal'),
                           (['--flash-cache', 'aligned', '--kv-store', 'ssd:/tmp/a,size=1G', '--kv-store',
                             'ssd:/tmp/b,size=1G'], 'give one --kv-store'),
                           (['--flash-cache', 'aligned', '--kv-store', 'ssd:/tmp/a,size=1G,persist=on'],
                            'do not persist')):
            with self.assertRaises(opt.OptionError) as e:
                flash(*argv)
            self.assertIn(want, str(e.exception))
        glm = opt.parse(['--no-config', '--peers', PEERS, '--flash-cache', 'aligned'], {})
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(glm)
        self.assertIn('glm53f only', str(e.exception))

    def test_cap(self):
        """CAP: a 262,144-token request and a page for the guard rows (2 at MTP=1) at either TP, as no CP leaves every
        rank the whole context; a set CAP wins."""
        for peers in (PEERS, '10.0.0.1,10.0.0.2'):
            plan = opt.engine_plan(flash(peers=peers))
            cap = int(plan['knobs']['CAP'])
            self.assertEqual(cap, 262208)
            self.assertEqual(cap % 64, 0)   # the engine's pool_rows: CAP rounded down to the page
            self.assertGreaterEqual(cap - 2 * int(plan['knobs']['MTP']), 262144)   # the shim's max_model_len
            self.assertEqual(plan['args'][plan['args'].index('--cap') + 1], '262208')
            self.assertEqual(plan['knobs']['SHARE_LIMIT'], '32768')
            plan = opt.engine_plan(flash('--cap', '131072', peers=peers))
            self.assertEqual(plan['args'][plan['args'].index('--cap') + 1], '131072')

    def test_sessions(self):
        # SESS 2..8 share decode steps (each session its own KDA slots); a round of every session fits 32 rows
        for argv, sess in ((['--sess', '4'], '4'), (['--sess', '8'], '8'), (['--sess', '8', '--mtp', '3'], '8'),
                           (['--sess', '1', '--mtp', '7'], '1')):
            plan = opt.engine_plan(flash(*argv))
            self.assertEqual(plan['knobs']['SESS'], sess)
            self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], sess)
            self.assertEqual(plan['args'][plan['args'].index('--chunk') + 1], '1')
            self.assertNotIn('--reuse-prefix', plan['args'])

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



class PseudoTests(unittest.TestCase):
    """GR8_MODEL=pseudo: the pseudo engine's command line and its release, a directory with responses.toml."""

    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.rel = os.path.join(self.t, 'rel')
        os.makedirs(self.rel)
        for n in (*M.TOKENIZER, 'responses.toml'):
            with open(os.path.join(self.rel, n), 'w') as f:
                f.write(n)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.t)

    def pseudo(self, *argv):
        o = opt.parse(['--no-config', '--peers', '127.0.0.1', '--model', self.rel, *argv], {'GR8_MODEL': 'pseudo'})
        opt.validate(o)
        return o

    def test_engine_plan(self):
        o = self.pseudo()
        self.assertEqual((o.get('TP'), o.get('GROUP'), S.engine_bin(o)), ('1', 'gr8-pseudo', 'g6pseudo'))
        ms, variant, err = S.resolve_model(o)
        self.assertEqual((err, variant.name, ms.set['source']['kind']), ('', 'home', 'pseudo'))
        plan = opt.engine_plan(o, rank=0, home='/h', epoch=3, set_id=ms.id)
        self.assertEqual(plan['env'], [])
        self.assertEqual(plan['args'], ['--model', self.rel, '--cap', '202752', '--serve', '/h/s2', '--s2-sessions', '8',
                                        '--s2-epoch', '3', '--set-id', ms.id])

    def test_the_rules_are_the_weights(self):
        first = M.pseudo_set(self.rel, 1).id
        with open(os.path.join(self.rel, 'responses.toml'), 'a') as f:
            f.write('\n')
        self.assertNotEqual(M.pseudo_set(self.rel, 1).id, first)
        os.unlink(os.path.join(self.rel, 'responses.toml'))
        self.assertIn('want a pseudo engine release', S.resolve_model(self.pseudo())[2])


if __name__ == '__main__':
    unittest.main()
