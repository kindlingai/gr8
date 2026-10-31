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
from test_modelset import make_release  # noqa: E402
from test_variant import manifest  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'
GLM = {'GR8_MODEL': 'glm53'}   # the model: a boot names it (no default)


def flash(*argv, peers=PEERS):
    o = opt.parse(['--no-config', '--peers', peers, *argv], {'GR8_MODEL': 'glm53f'})
    opt.validate(o)
    return o


class OptionTests(unittest.TestCase):
    def test_defaults(self):
        o = flash()
        self.assertEqual((models.of(o), o.get('GROUP'), S.engine_bin(o)), (models.GLM53F, 'gr8-glm53f', 'g6m4_flash'))
        self.assertEqual(flash('--group', 'mine').get('GROUP'), 'mine')
        glm = opt.parse(['--no-config', '--peers', PEERS], GLM)
        self.assertEqual((models.of(glm), glm.get('GROUP')), (models.GLM53, 'gr8-glm53'))

    def test_refusals(self):
        for argv, want in ((['--tp', '6'], 'want 4, 3 or 2 (glm53f)'), (['--kv-format', 'int4'], 'fp8 KV build only'),
                           (['--dflash', '1'], 'no DFlash2 drafter'),
                           (['--drafter', 'https://huggingface.co/a/b'], 'no DFlash2 drafter')):
            with self.assertRaises(opt.OptionError) as e:
                flash(*argv)
            self.assertIn(want, str(e.exception))
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
            self.assertIn('GR8_CP=0', plan['env'])
            self.assertIn('GR8_PF_RS=0', plan['env'])
            self.assertIn('GR8_GLM53F_PF_STAGE_ASYNC=1', plan['env'])
            (self.assertNotIn if tp == '4' else self.assertIn)('GR8_GLM53F_KDA_SCAN=1', plan['env'])
            for a in ('--reuse-prefix', '--wide', '--pfv2', '--dflash-manifest', '--qo-manifest'):
                self.assertNotIn(a, plan['args'])
            self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], '1')
            self.assertEqual(plan['args'][plan['args'].index('--chunk') + 1], '1')
            self.assertIn('GR8_GLM53F_PF_DSA_F16GEMM=1', plan['env'])

    def test_flash_cache(self):
        plan = opt.engine_plan(flash())
        self.assertFalse([e for e in plan['env'] if e.startswith(('GR8_GLM53F_CACHE=', 'GR8_GLM53F_STATE_BUDGET_MIB='))])
        self.assertNotIn('--kv-store', plan['args'])
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--flash-state-budget-mib', '4096'))
        self.assertIn('GR8_GLM53F_CACHE=aligned', plan['env'])
        self.assertIn('GR8_GLM53F_STATE_BUDGET_MIB=4096', plan['env'])
        self.assertNotIn('--kv-store', plan['args'])
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', 'ssd:/tmp/fc,size=64G'))
        self.assertEqual(plan['args'][plan['args'].index('--kv-store') + 1], 'ssd:/tmp/fc,size=64G')
        plan = opt.engine_plan(flash('--flash-cache', 'terminal'))
        self.assertIn('GR8_GLM53F_CACHE=terminal', plan['env'])
        # the checkpoint store on a group lease (D-331 M8 (c)): the store as given, KV_SHARE_WHEN reaches the engine
        g = 'remote:192.0.2.5:7350,size=1G,group=flash'
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', g, '--kv-share-when', 'always'))
        self.assertEqual(plan['args'][plan['args'].index('--kv-store') + 1], g)
        self.assertIn('GR8_KV_SHARE_WHEN=always', plan['env'])
        self.assertNotIn('GR8_KV_SHARE_WHEN', ' '.join(opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', g))['env']))
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
        glm = opt.parse(['--no-config', '--peers', PEERS, '--flash-cache', 'aligned'], GLM)
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
        glm = opt.parse(['--no-config', '--peers', PEERS], GLM)
        opt.validate(glm)
        c, _, _ = opt.canonical(glm, opt.engine_plan(glm), opt.split_list(PEERS))
        self.assertNotIn('model', c)
        c, _, _ = opt.canonical(flash(), opt.engine_plan(flash()), opt.split_list(PEERS))
        self.assertEqual(c['model'], 'glm53f')


def ds41(*argv, peers=PEERS):
    o = opt.parse(['--no-config', '--peers', peers, *argv], {'GR8_MODEL': 'ds41'})
    opt.validate(o)
    return o


class Ds41Tests(unittest.TestCase):
    """DeepSeek-V4.1-Flash-1017 scaffold (docs/DEEPSEEK.md): TP=4, the engine bin, the refusals."""
    def test_defaults(self):
        o = ds41()
        self.assertEqual((models.of(o), o.get('GROUP'), S.engine_bin(o)), (models.DS41, 'gr8-ds41', 'g6m4_ds41'))
        self.assertEqual(models.DS41.architecture, 'DeepseekV41ForCausalLM')

    def test_refusals(self):
        for argv, want in ((['--tp', '3'], 'want 4 (ds41)'), (['--dflash', '1'], 'no DFlash2 drafter'),
                           (['--kv-format', 'int4'], 'fp8 KV build only')):
            with self.assertRaises(opt.OptionError) as e:
                ds41(*argv)
            self.assertIn(want, str(e.exception))
        # DS41 prefix snapshots are now opt-in via REUSE=1; the other refusals remain.
        opt.engine_plan(ds41('--reuse', '1'))
        for argv, want in ((['--sess', '2', '--extra-env', 'GR8_DS_PREFILL_MIN=0'], 'serves several sessions only with GR8_DS_PREFILL_MIN>=2'),
                           (['--mtp', '6'], 'drafts 0 to 5'), (['--sess', '8', '--mtp', '4'], 'a round of 40 rows')):
            with self.assertRaises(opt.OptionError) as e:
                opt.engine_plan(ds41(*argv))
            self.assertIn(want, str(e.exception))

    def test_front_end_defaults_are_the_models_and_a_setting_wins(self):
        o = ds41()
        self.assertEqual((o.get('DEFAULT_MAX_TOKENS'), o.get('DEFAULT_REASONING_EFFORT')), ('32768', 'low'))
        self.assertEqual((o.origin['DEFAULT_MAX_TOKENS'], o.origin['DEFAULT_REASONING_EFFORT']),
                         ('ds41 default', 'ds41 default'))
        o = ds41('--default-max-tokens', '0', '--default-reasoning-effort', 'max')
        self.assertEqual((o.get('DEFAULT_MAX_TOKENS'), o.get('DEFAULT_REASONING_EFFORT')), ('0', 'max'))
        self.assertEqual(o.origin['DEFAULT_MAX_TOKENS'], 'command line')
        glm = opt.parse(['--no-config', '--peers', PEERS], {})     # the GLM table defaults stay
        self.assertEqual((glm.get('DEFAULT_MAX_TOKENS'), glm.get('DEFAULT_REASONING_EFFORT')), ('0', None))

    def test_the_reasoning_effort_is_a_name_or_a_whole_number_to_100(self):
        for e in ('none', 'minimal', 'high', 'xhigh', 'max', '1', '50', '75', '100'):
            self.assertEqual(ds41('--default-reasoning-effort', e).get('DEFAULT_REASONING_EFFORT'), e)
        for e in ('0', '101', '-5', '7.5', 'auto', 'default'):
            with self.assertRaises(opt.OptionError) as err:
                ds41('--default-reasoning-effort', e)
            self.assertIn('want a whole number 1 to 100', str(err.exception))
        # GLM keeps its list: no number, no none
        for e in ('50', 'none'):
            with self.assertRaises(opt.OptionError):
                opt.validate(opt.parse(['--no-config', '--peers', PEERS, '--default-reasoning-effort', e], {}))

    def test_ds_variables_belong_to_ds41(self):
        ds41('--extra-env', 'GR8_DS_ENGRAM=0 GR8_DS41_IDX_REUSE=1')
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(opt.parse(['--no-config', '--peers', PEERS, '--extra-env', 'GR8_DS_ENGRAM=0'], GLM))
        self.assertIn('DeepSeek-V4.1 engine variables', str(e.exception))

    def test_engine_plan(self):
        plan = opt.engine_plan(ds41())
        self.assertEqual((plan['knobs']['MTP'], plan['knobs']['SESS'], plan['knobs']['CAP']), ('0', '5', '262144'))
        for e in ('GR8_CP=0', 'GR8_DEC_BK=16384', 'GR8_DS_MTP=5', 'GR8_DS_PREFILL_W=1024', 'GR8_DS_ATTN_PW=1024', 'GR8_DS_PF_LANES=2'):
            self.assertIn(e, plan['env'])
        self.assertFalse([e for e in plan['env'] if 'PF_STAGE_ASYNC' in e])   # none of the GLM or Flash prefill stack
        self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], '5')
        self.assertEqual(plan['args'][plan['args'].index('--mtp') + 1], '0')
        self.assertIn('--reuse-prefix', plan['args'])
        for a in ('--wide', '--pfv2', '--dflash-manifest', '--qo-manifest'):
            self.assertNotIn(a, plan['args'])
        # a variable EXTRA_ENV names replaces the tuned one, and the attention and indexer widths follow the prefill width
        plan = opt.engine_plan(ds41('--extra-env', 'GR8_DS_ENGRAM=0 GR8_DEC_SPLIT=0 GR8_DS_MTP=0 GR8_DS_PREFILL_W=4096'))
        self.assertEqual(sorted(e for e in plan['env'] if e.startswith(('GR8_DS_MTP=', 'GR8_DS_ATTN_PW=', 'GR8_DS41_IDX_PW='))),
                         ['GR8_DS41_IDX_PW=4096', 'GR8_DS_ATTN_PW=4096', 'GR8_DS_MTP=0'])
        self.assertIn('GR8_DS_ENGRAM=0', plan['env'])


class Ds41ReleaseTests(unittest.TestCase):
    """--model is a local copy of the exlr8 native release: dense/, manifests/ with a set id, two tokenizer files."""

    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.t)
        os.makedirs(os.path.join(self.t, 'dense'))
        os.makedirs(os.path.join(self.t, 'manifests'))
        for n in ('config.json', 'dense/exlr8-vocab.safetensors', *models.TOKENIZER_DS):
            with open(os.path.join(self.t, n), 'w') as f:
                f.write('{}')       # config.json as DeepSeek's: no `architectures`
        self.set_id('a' * 64)

    def set_id(self, sid):
        with open(os.path.join(self.t, 'manifests', 'exlr8-manifest-native.json'), 'w') as f:
            json.dump({'set_id': sid}, f)

    def ds41(self, *argv):
        return ds41('--model', self.t, *argv)

    def test_the_set_needs_no_chat_template_or_generation_config(self):
        ms, variant, err = S.resolve_model(self.ds41())
        self.assertEqual((err, variant.name, ms.set['source']['kind'], ms.tokenizer), ('', 'home', 'exlr8-release', self.t))
        self.assertEqual(sorted(ms.set['tokenizer']), sorted(models.TOKENIZER_DS))
        self.assertEqual(M.check_tokenizer(ms), [])
        o = self.ds41()
        S.resolve_model(o)
        self.assertEqual(o.get('V2_DENSE'), self.t)                   # the engine reads the release in place
        self.assertFalse(o.get('V2'))

    def test_the_weights_are_named_by_the_release_set_id(self):
        first = M.ds_set(self.t, 4).id
        self.assertEqual(M.ds_set(self.t, 4).id, first)
        self.set_id('b' * 64)
        self.assertNotEqual(M.ds_set(self.t, 4).id, first)

    def test_the_other_directories_are_refused_with_what_is_wanted(self):
        os.unlink(os.path.join(self.t, 'dense', 'exlr8-vocab.safetensors'))
        self.assertIn('want a local copy of the DeepSeek-V4.1-Flash exlr8 native release', S.resolve_model(self.ds41())[2])
        self.assertIn('no model: pass --model', S.resolve_model(ds41())[2])
        with open(os.path.join(self.t, 'dense', 'exlr8-vocab.safetensors'), 'w') as f:
            f.write('{}')
        os.unlink(os.path.join(self.t, 'tokenizer.json'))
        self.assertIn('no tokenizer files (tokenizer.json, tokenizer_config.json)', S.resolve_model(self.ds41())[2])


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

class RequiredTests(unittest.TestCase):
    """The runner names the model: no model, an unknown name or a release of the other model stops the boot before any
    download, in one line naming what is wrong and the models (pseudo, the test engine, only when asked for)."""

    def test_no_model(self):
        for env in ({}, {'GR8_MODEL': ''}):
            o = opt.parse(['--no-config', '--peers', PEERS], env)            # down, status and logs need none
            with self.assertRaises(opt.OptionError) as e:
                opt.validate(o)                                              # up, check, prepare, show-config
            self.assertIn('no model: set GR8_MODEL to glm53 (GLM-5.3) or glm53f (GLM-5.3-Flash)', str(e.exception))
            self.assertNotIn('pseudo', str(e.exception))

    def test_an_unknown_name(self):
        for argv, env, src in ((['--gr8-model', 'glm9'], {}, 'command line'),
                               ([], {'GR8_MODEL': 'glm53x'}, 'environment')):
            with self.assertRaises(opt.OptionError) as e:
                opt.parse(['--no-config', *argv], env)
            self.assertIn(f'from the {src}: not a model gr8 serves. Want glm53 (GLM-5.3) or glm53f (GLM-5.3-Flash)',
                          str(e.exception))
            self.assertNotIn('pseudo', str(e.exception))
        self.assertIs(models.of(opt.parse(['--no-config'], {'GR8_MODEL': 'pseudo'})), models.PSEUDO)   # asked for

    def test_a_release_of_the_other_model_is_named_before_the_variant(self):
        t = tempfile.mkdtemp()
        # TP=3 GLM-5.3 wants bit225 or down1, and Flash k4: neither release has the other's, so the variant step would
        # fail first, on a missing manifest, if the architecture were not checked before it
        for name, arch, peers, want in (
                ('glm53', 'Glm5NextForConditionalGeneration', '10.0.0.1,10.0.0.2,10.0.0.3', './gr8 glm53f serves it'),
                ('glm53f', 'GlmMoeDsaForCausalLM', PEERS, './gr8 glm53 serves it')):
            rel = make_release(os.path.join(t, name), variants=('k4',) if name == 'glm53' else ('home',))
            with open(os.path.join(rel, 'config.json'), 'w') as f:
                json.dump({'architectures': [arch]}, f)
            o = opt.parse(['--no-config', '--peers', peers, '--model', rel], {'GR8_MODEL': name})
            opt.validate(o)
            ms, variant, err = S.resolve_model(o)
            self.assertIn(f'is {arch}, and ./gr8 {name} serves', err)
            self.assertIn(want, err)
            self.assertIsNone(variant)
            self.assertIsNone(o.get('V2_DENSE'))

    def test_the_listing(self):
        glm, flash = models.describe(models.GLM53), models.describe(models.GLM53F)
        self.assertIn('--tp       4, 6, 3 or 5', glm)
        self.assertIn('TP=3 bit225 else down1', glm)
        self.assertIn(models.GLM53.release, glm)
        self.assertIn('--tp       4, 3 or 2', flash)
        self.assertNotIn('--drafter', flash)
        self.assertEqual(models.SERVED, (models.GLM53, models.GLM53F))


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
