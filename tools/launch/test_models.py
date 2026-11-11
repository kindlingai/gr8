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


class FlashWaterfillOptionsTests(unittest.TestCase):
    PEERS2 = '10.0.0.1,10.0.0.2'

    def plan(self, *argv):
        return opt.engine_plan(flash('--tp', '2', '--kv-format', 'int4', *argv, peers=self.PEERS2))

    def word(self, plan, key, value):
        self.assertEqual([w for w in plan['env'] if w.startswith(key + '=')],
                         [] if value is None else [key + '=' + value])

    def test_defaults_and_eight_streams(self):
        for args, sessions in (((), '4'), (('--sess', '8'), '8')):
            p = self.plan(*args)
            self.assertEqual((p['knobs']['SESS'], p['knobs']['MTP']), (sessions, '3'))
            self.word(p, 'GR8_S2_SHARED_WATERFILL', '1')
            self.word(p, 'GR8_S2_SHARED_WF_MARGIN', '0.02')
            for flag, value in (('--mtp', '3'), ('--s2-sessions', sessions)):
                self.assertEqual(p['args'][p['args'].index(flag) + 1], value)

    def test_rollbacks_and_overrides(self):
        for args, mtp, wf, margin in (
            (('--s2-shared-waterfill', '0'), '1', '0', None),
            (('--extra-env', 'GR8_S2_SHARED_WATERFILL=0'), '1', '0', None),
            (('--s2-shared-waterfill', '0', '--mtp', '3'), '3', '0', None),
            (('--mtp', '1'), '1', '1', '0.02'),
            (('--sess', '1'), '1', None, None),
        ):
            with self.subTest(args=args):
                p = self.plan(*args)
                self.assertEqual(p['knobs']['MTP'], mtp)
                self.word(p, 'GR8_S2_SHARED_WATERFILL', wf)
                self.word(p, 'GR8_S2_SHARED_WF_MARGIN', margin)
                if args == ('--sess', '1'):
                    self.assertEqual(p['knobs']['SESS'], '1')

    def test_other_shapes_untouched(self):
        cases = [flash('--tp', str(tp), *args, peers=','.join('10.0.0.' + str(i + 1) for i in range(tp)))
                 for tp, args in ((1, ('--kv-format', 'int4', '--sess', '4')), (2, ()), (3, ()),
                                  (4, ('--kv-format', 'int4')))]
        cases += [opt.parse(['--no-config', '--peers', PEERS], GLM), ds41()]
        for o in cases:
            opt.validate(o)
            p = opt.engine_plan(o)
            self.word(p, 'GR8_S2_SHARED_WATERFILL', None)
            self.word(p, 'GR8_S2_SHARED_WF_MARGIN', None)

    def test_refusals(self):
        for args in (('--s2-shared-waterfill', '2'), ('--s2-shared-wf-margin', '-1'),
                     ('--s2-shared-wf-margin', 'nan'), ('--s2-shared-wf-margin', 'inf')):
            with self.subTest(args=args), self.assertRaises(opt.OptionError):
                self.plan(*args)
        with self.assertRaises(opt.OptionError):
            ds41('--s2-shared-waterfill', '1')


class Kv4ExactOptionsTests(unittest.TestCase):
    WORDS = ('GR8_KV4_SC=4', 'GR8_KV4_SC_TR=2', 'GR8_KV4_QT=1',
             'GR8_KV4_CTX=2', 'GR8_KV4_FUSE_ROT=1')

    @staticmethod
    def plan(*argv, tp=4):
        peers = ','.join('192.0.2.' + str(i + 1) for i in range(tp))
        o = opt.parse(['--no-config', '--peers', peers, '--tp', str(tp), *argv], GLM)
        opt.validate(o)
        return opt.engine_plan(o)

    def test_int4_defaults(self):
        env = self.plan('--kv-format', 'int4')['env']
        for word in self.WORDS:
            self.assertEqual([w for w in env if w.startswith(word.split('=')[0] + '=')], [word])

    def test_other_shapes_unchanged(self):
        for tp, args in ((6, ('--kv-format', 'int4')), (3, ('--kv-format', 'int4')), (4, ())):
            env = self.plan(*args, tp=tp)['env']
            self.assertFalse([w for w in env if w.startswith('GR8_KV4_')])

    def test_extra_env_wins_once(self):
        keys = ('GR8_KV4_SC', 'GR8_KV4_CTX', 'GR8_KV4_QT', 'GR8_KV4_FUSE_ROT')
        env = self.plan('--kv-format', 'int4', '--extra-env', ' '.join(k + '=0' for k in keys))['env']
        for key in keys:
            self.assertEqual([w for w in env if w.startswith(key + '=')], [key + '=0'])


class Kv4OptionsTests(unittest.TestCase):
    WORDS = Kv4ExactOptionsTests.WORDS + ('GR8_KV4_FOLD=1', 'GR8_KV4_TC=1', 'GR8_CP_ONECHAIN=1')

    @staticmethod
    def plan(*argv, tp=4):
        plan = Kv4ExactOptionsTests.plan('--share-limit', '32768', *argv, tp=tp)
        if plan['knobs']['SHARE_LIMIT'] != '32768':
            raise AssertionError('tests require the stock shared prefix')
        return plan

    def assert_words_once(self, env, words):
        for word in words:
            self.assertEqual([w for w in env if w.startswith(word.split('=')[0] + '=')], [word])

    def test_default_stack(self):
        self.assert_words_once(self.plan('--kv-format', 'int4')['env'], self.WORDS)

    def test_exact_rollback(self):
        env = self.plan('--kv-format', 'int4', '--kv4-tc', '0', '--kv4-fold', '0', '--cp-onechain', '0')['env']
        self.assert_words_once(env, Kv4ExactOptionsTests.WORDS +
                               ('GR8_KV4_TC=0', 'GR8_KV4_FOLD=0', 'GR8_CP_ONECHAIN=0'))

    def test_fp8_weights_keep_exact_path(self):
        for args in (('--kvb', 'fp8'), ('--extra-env', 'GR8_KVB=fp8')):
            env = self.plan('--kv-format', 'int4', '--v2-dense', '/release', *args)['env']
            self.assertNotIn('GR8_KV4_FOLD=1', env)
            self.assertNotIn('GR8_KV4_TC=1', env)
            self.assert_words_once(env, Kv4ExactOptionsTests.WORDS)

    def test_fold_override_disables_implicit_tc(self):
        env = self.plan('--kv-format', 'int4', '--extra-env', 'GR8_KV4_FOLD=0')['env']
        self.assert_words_once(env, ('GR8_KV4_FOLD=0',))
        self.assertNotIn('GR8_KV4_FOLD=1', env)
        self.assertNotIn('GR8_KV4_TC=1', env)

    def test_flip_disables_implicit_onechain(self):
        env = self.plan('--kv-format', 'int4', '--extra-env', 'GR8_ATTN_FLIP=1')['env']
        self.assertNotIn('GR8_CP_ONECHAIN=1', env)

    def test_no_defaults_for_other_shapes(self):
        for tp, args in ((6, ('--kv-format', 'int4')), (3, ('--kv-format', 'int4')), (4, ())):
            env = self.plan(*args, tp=tp)['env']
            self.assertFalse([w for w in env if w.startswith(('GR8_KV4_', 'GR8_CP_ONECHAIN='))])

    def test_refusals(self):
        for args in (('--kv4-fold', '2'), ('--kv4-tc', 'x'), ('--kv4-fold', '1'),
                     ('--kv-format', 'int4', '--kv4-fold', '1', '--kvb', 'fp8', '--v2-dense', '/release')):
            with self.assertRaises(opt.OptionError):
                self.plan(*args)
        with self.assertRaises(opt.OptionError):
            flash('--kv-format', 'int4', '--kv4-fold', '1')


class OptionTests(unittest.TestCase):
    def test_defaults(self):
        o = flash()
        self.assertEqual((models.of(o), o.get('GROUP'), S.engine_bin(o)), (models.GLM53F, 'gr8-glm53f', 'g6m4_flash'))
        self.assertEqual(flash('--group', 'mine').get('GROUP'), 'mine')
        # --kv-format int4 runs g6m4_flash-int4 and passes the flag, which that build requires
        o = flash('--kv-format', 'int4')
        self.assertEqual(S.engine_bin(o), 'g6m4_flash-int4')
        self.assertEqual(opt.engine_plan(o)['args'][-2:], ['--kv-format', 'int4'])
        glm = opt.parse(['--no-config', '--peers', PEERS], GLM)
        self.assertEqual((models.of(glm), glm.get('GROUP')), (models.GLM53, 'gr8-glm53'))

    def test_refusals(self):
        for argv, want in ((['--tp', '6'], 'want 4, 3, 2 or 1 (glm53f)'),
                           (['--tp', '3', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3', '--drafter', 'https://huggingface.co/a/b'],
                            'runs at TP=4, TP=2 and TP=1 (split) and TP=6')):
            with self.assertRaises(opt.OptionError) as e:
                flash(*argv)
            self.assertIn(want, str(e.exception))
        for argv, want in ((['--reuse', '1'], 'GLM-5.3-Flash has no REUSE'), (['--sess', '9'], 'serves 1 to 8 sessions'),
                           (['--tp', '3', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3', '--dflash', '1'], 'drafter at TP=4'),
                           (['--sess', '0'], 'serves 1 to 8 sessions'),
                           (['--sess', '8', '--mtp', '4'], 'a round of 40 rows')):
            with self.assertRaises(opt.OptionError) as e:
                opt.engine_plan(flash(*argv))
            self.assertIn(want, str(e.exception))

    def test_engine_plan(self):
        for tp, peers in (('4', PEERS), ('3', '10.0.0.1,10.0.0.2,10.0.0.3'), ('2', '10.0.0.1,10.0.0.2')):
            plan = opt.engine_plan(flash(peers=peers))
            sess = {'4': '8', '3': '4', '2': '1'}[tp]
            self.assertEqual(plan['knobs']['SESS'], sess)
            self.assertIn('GR8_CP=0', plan['env'])
            self.assertIn('GR8_PF_RS=0', plan['env'])
            # TP=4: sequence-parallel mHC in place of the peer-partial staging
            self.assertIn('GR8_GLM53F_PF_STAGE_ASYNC=' + ('0' if tp == '4' else '1'), plan['env'])
            self.assertIn('GR8_GLM53F_PF_SP=' + ('1' if tp == '4' else '0'), plan['env'])
            self.assertIn('GR8_GLM53F_KDA_SCAN=1', plan['env'])
            self.assertIn('GR8_PF_ROUTER_SPLIT=' + ('0' if tp == '4' else '1'), plan['env'])
            for a in ('--reuse-prefix', '--wide', '--pfv2', '--qo-manifest'):
                self.assertNotIn(a, plan['args'])
            (self.assertNotIn if tp == '3' else self.assertIn)('--dflash-manifest', plan['args'])
            self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], sess)
            self.assertEqual(plan['args'][plan['args'].index('--chunk') + 1], '1')
            self.assertIn('GR8_GLM53F_PF_DSA_F16GEMM=1', plan['env'])

    def test_one_box(self):
        # TP=1: one box, K2 (the width whose experts fit), one session, the drafter on as at TP=4 (the supervisor turns
        # it off when the boot names none)
        o = flash('--tp', '1', peers='10.0.0.1')
        plan = opt.engine_plan(o)
        self.assertEqual((plan['knobs']['SESS'], plan['knobs']['DFLASH']), ('1', '1'))
        self.assertIn('--dflash-manifest', plan['args'])
        flash('--tp', '1', '--drafter', 'https://huggingface.co/a/b', peers='10.0.0.1')
        self.assertIn('GR8_PF_ROUTER_SPLIT=0', plan['env'])
        self.assertEqual(models.GLM53F.default_variant[1], ('k2',))
        self.assertEqual((S.fabric_needed(o), S.fabric_needed(flash())), (False, True))
        with self.assertRaises(opt.OptionError) as e:
            flash('--tp', '1', '--ips', '10.1.0.1', '--ips2', '10.2.0.1', peers='10.0.0.1')
        self.assertIn('--tp 1 takes no --ips', str(e.exception))

    def test_max_model_len(self):
        # #339: the longest request, apart from the pool; a cluster option, so the boxes compare it
        o = flash('--cap', '1048640', '--max-model-len', '262144')
        c, _, _ = opt.canonical(o, opt.engine_plan(o), opt.split_list(PEERS))
        self.assertEqual(c['MAX_MODEL_LEN'], '262144')
        self.assertNotIn('MAX_MODEL_LEN', opt.canonical(flash(), opt.engine_plan(flash()), opt.split_list(PEERS))[0])
        for argv, want in ((['--max-model-len', '0'], 'want 1 to 1048576'), (['--max-model-len', '2000000'], 'want 1 to 1048576'),
                           (['--cap', '131136', '--max-model-len', '262144'], 'more than CAP 131136'),
                           (['--max-model-len', 'x'], 'want a whole number')):
            with self.assertRaises(opt.OptionError) as e:
                flash(*argv)
            self.assertIn(want, str(e.exception))

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
        # persist=on keeps aligned checkpoints and their index across boots
        plan = opt.engine_plan(flash('--flash-cache', 'aligned', '--kv-store', 'ssd:/tmp/fc,size=64G,persist=on'))
        self.assertEqual(plan['args'][plan['args'].index('--kv-store') + 1], 'ssd:/tmp/fc,size=64G,persist=on')
        plan = opt.engine_plan(flash('--flash-cache', 'terminal'))
        self.assertIn('GR8_GLM53F_CACHE=terminal', plan['env'])
        # the checkpoint store on a group lease: the store as given, KV_SHARE_WHEN reaches the engine
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
                           (['--flash-cache', 'terminal', '--kv-store', 'ssd:/tmp/a,size=1G,persist=on'],
                            'cold-exact (aligned) checkpoints only')):
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

    def test_dflash(self):
        # DFLASH=1 at TP=4 and TP=2 gives the drafter packs and the adaptive draft count, and DFLASH=0 at TP=3. The
        # supervisor turns DFLASH off when there is no drafter.
        self.assertEqual(opt.engine_plan(flash())['knobs']['DFLASH'], '1')
        self.assertEqual(opt.engine_plan(flash(peers='10.0.0.1,10.0.0.2'))['knobs']['DFLASH'], '1')
        self.assertEqual(opt.engine_plan(flash(peers='10.0.0.1,10.0.0.2,10.0.0.3'))['knobs']['DFLASH'], '0')
        plan = opt.engine_plan(flash('--dflash', '1', '--dflash-dir', '/d'))
        a = ' '.join(plan['args'])
        self.assertIn('--dflash-manifest @DFLASH_DIR/dflash.r@R.manifest.json --dflash-pack @DFLASH_DIR/dflash.r@R.pack', a)
        self.assertIn('--spec-adaptive', a)

    def test_flash_cache_with_the_drafter(self):
        # FLASH_CACHE and DFLASH=1 together (#340): both reach the engine; DFL_REUSE and DFL_REUSE_MIN pass through
        # when set, else the engine's default holds
        plan = opt.engine_plan(flash('--dflash', '1', '--dflash-dir', '/d', '--flash-cache', 'aligned'))
        self.assertIn('GR8_GLM53F_CACHE=aligned', plan['env'])
        self.assertIn('--dflash-manifest', ' '.join(plan['args']))
        self.assertFalse([e for e in plan['env'] if e.startswith('GR8_DFL_REUSE')])
        plan = opt.engine_plan(flash('--dflash', '1', '--dflash-dir', '/d', '--flash-cache', 'aligned',
                                     '--dfl-reuse', '0', '--dfl-reuse-min', '64'))
        self.assertIn('GR8_DFL_REUSE=0', plan['env'])
        self.assertIn('GR8_DFL_REUSE_MIN=64', plan['env'])
        plan = opt.engine_plan(flash('--dflash', '1', '--dflash-dir', '/d', '--dfl-reuse', '2'))
        self.assertIn('GR8_DFL_REUSE=2', plan['env'])
        # terminal checkpoints are not cold-exact: the drafter takes aligned only (the engine refuses it too)
        with self.assertRaises(opt.OptionError) as e:
            flash('--dflash', '1', '--dflash-dir', '/d', '--flash-cache', 'terminal')
        self.assertIn('aligned checkpoints only', str(e.exception))

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

    # the release stack's prefill words (docs/DEEPSEEK.md): the plain default; the fast decode words stay out
    V2 = ('GR8_DS_PF_V2_L20_TAIL=1', 'GR8_DS_PF_V2_SEND_FFN_OPT=3', 'GR8_DS_PF_V2_W16K=1', 'GR8_DS_PF_V2=1', 'GR8_DS_PF_LANES=1', 'GR8_DS_PF_TAIL=128', 'GR8_DS_SH_TC=1', 'GR8_DS_PREFILL_W=12288', 'GR8_DS_ATTN_PW=12288',
          'GR8_DS41_IDX_PW=12288', 'GR8_DS_PF_V2_GATE_RO=1', 'GR8_DS_PF_V2_STRIPE_MIN=1048576', 'GR8_DS_PF_V2_ATTN_PIPE=1',
          'GR8_DS_ENGRAM_IO_THREADS=64', 'GR8_DS_ATTN_TC_TILE=2', 'GR8_S2_ADMIT_HOLD=1', 'GR8_S2_TAIL_FIRST=1', 'GR8_S2_DECODE_PF=1024')
    FAST = ('GR8_DS_RELAXED_ACCEPT', 'GR8_DS_MOE_TOPK_DECODE', 'GR8_DS41_HEAD_MX', 'GR8_DS_ATTN_TC_MF')

    def test_the_default_prefill_is_the_release_stack(self):
        env = opt.engine_plan(ds41())['env']
        for e in self.V2:
            self.assertEqual([w for w in env if w.startswith(e.split('=')[0] + '=')], [e])
        self.assertFalse([w for w in env if w.startswith(self.FAST)])
        self.assertFalse([w for w in env if w.startswith(('GR8_DS_MHC_ROW_SHARD=', 'GR8_DS_MHC_Y_MXFP8='))])   # engine defaults: 0

    def test_v2_width_overrides(self):
        for override in ('GR8_DS_PF_V2_W16K=0', 'GR8_DS_PREFILL_W=8192'):
            env = opt.engine_plan(ds41('--extra-env', override))['env']
            for key in ('GR8_DS_PREFILL_W', 'GR8_DS_ATTN_PW', 'GR8_DS41_IDX_PW'):
                self.assertEqual([w for w in env if w.startswith(key + '=')], [key + '=8192'])
            self.assertNotIn('GR8_DS_PF_V2_W16K=1', env)
            if override.endswith('W16K=0'):
                self.assertEqual(env.count(override), 1)

    def test_exact_output_is_the_old_prefill(self):
        # EXACT=1, EXTRA_ENV GR8_DS_PF_V2=0 and the 1M-context row split: today's stack, two lanes, W 4096, no v2 word
        for argv in (('--exact', '1'), ('--extra-env', 'GR8_DS_PF_V2=0'), ('--extra-env', 'GR8_DS_MHC_ROW_SHARD=1')):
            env = opt.engine_plan(ds41(*argv))['env']
            for e in ('GR8_DS_PF_LANES=2', 'GR8_DS_PREFILL_W=4096', 'GR8_DS_ATTN_PW=4096', 'GR8_DS41_IDX_PW=4096', 'GR8_S2_DECODE_PF=1024'):
                self.assertIn(e, env)
            for w in ('GR8_DS_PF_V2=1', 'GR8_DS_SH_TC', 'GR8_DS_PF_TAIL', 'GR8_DS_PF_V2_', 'GR8_S2_TAIL_FIRST', 'GR8_S2_ADMIT_HOLD',
                      'GR8_DS_ATTN_TC_TILE', 'GR8_DS_ENGRAM_IO_THREADS'):
                self.assertFalse([e for e in env if e.startswith(w)], (argv, w))
        # the exact list is the default list less the release words, with two lanes and W 4096
        exact = [e for e in opt.engine_plan(ds41('--exact', '1'))['env']]
        default = [e for e in opt.engine_plan(ds41())['env'] if not e.startswith(tuple(v.split('=')[0] + '=' for v in self.V2))]
        self.assertEqual([e for e in exact if not e.startswith(('GR8_DS_PF_LANES=', 'GR8_DS_PREFILL_W=', 'GR8_DS_ATTN_PW=',
                                                                'GR8_DS41_IDX_PW=', 'GR8_S2_DECODE_PF='))], default)
        # an EXTRA_ENV word keeps its value; W follows it on either stack
        env = opt.engine_plan(ds41('--extra-env', 'GR8_DS_PREFILL_W=4096'))['env']
        self.assertIn('GR8_DS_PF_V2=1', env)
        self.assertEqual(sorted(e for e in env if e.startswith(('GR8_DS_PREFILL_W=', 'GR8_DS_ATTN_PW='))),
                         ['GR8_DS_ATTN_PW=4096', 'GR8_DS_PREFILL_W=4096'])
        env = opt.engine_plan(ds41('--extra-env', 'GR8_DS_SH_TC=0'))['env']
        self.assertEqual([e for e in env if e.startswith('GR8_DS_SH_TC=')], ['GR8_DS_SH_TC=0'])   # the engine falls back, one line
        self.assertIn('GR8_S2_ADMIT_HOLD=0', opt.engine_plan(ds41('--s2-admit-hold', '0'))['env'])

    def test_exact_is_ds41_only_and_0_or_1(self):
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(opt.parse(['--no-config', '--peers', PEERS, '--exact', '1'], GLM))
        self.assertIn('ds41 only', str(e.exception))
        with self.assertRaises(opt.OptionError) as e:
            ds41('--exact', '2')
        self.assertIn('want 0', str(e.exception))

    def test_engine_plan(self):
        plan = opt.engine_plan(ds41())
        self.assertEqual((plan['knobs']['MTP'], plan['knobs']['SESS'], plan['knobs']['CAP']), ('0', '5', '262144'))
        for e in ('GR8_CP=0', 'GR8_DEC_BK=262144', 'GR8_DS_MTP=5', 'GR8_DS_PREFILL_W=12288', 'GR8_DS_ATTN_PW=12288', 'GR8_DS_PF_LANES=1', 'GR8_DS_ATTN_TC=1', 'GR8_DS_MOE_TC=1',
                  'GR8_DS_PF_POOL=4', 'GR8_DS_SRC_FAST=3', 'GR8_DS_MOE_PDL=1', 'GR8_DS41_ADMIT_KEEP_SCRATCH=1', 'GR8_DS41_PUB16=1',
                  'GR8_DS41_GV_QUAD=1', 'GR8_S2_DECODE_PF=1024', 'GR8_DS_COMP_FP4=1'):
            self.assertIn(e, plan['env'])
        self.assertFalse([e for e in plan['env'] if 'PF_STAGE_ASYNC' in e])   # none of the GLM or Flash prefill stack
        self.assertEqual(plan['args'][plan['args'].index('--s2-sessions') + 1], '5')
        # one decode graph bucket: DEC_BK follows CAP unless set
        self.assertIn('GR8_DEC_BK=1048576', opt.engine_plan(ds41('--cap', '1048576'))['env'])
        self.assertIn('GR8_DEC_BK=16384', opt.engine_plan(ds41('--dec-bk', '16384'))['env'])   # an explicit value still wins
        # packed FP4 compressed rows by default; EXTRA_ENV GR8_DS_COMP_FP4=0 opts out (one copy, its value)
        off = opt.engine_plan(ds41('--extra-env', 'GR8_DS_COMP_FP4=0'))['env']
        self.assertEqual([e for e in off if e.startswith('GR8_DS_COMP_FP4=')], ['GR8_DS_COMP_FP4=0'])
        self.assertEqual(plan['args'][plan['args'].index('--mtp') + 1], '0')
        self.assertIn('--reuse-prefix', plan['args'])
        for a in ('--wide', '--pfv2', '--dflash-manifest', '--qo-manifest'):
            self.assertNotIn(a, plan['args'])
        # a variable EXTRA_ENV names replaces the tuned one, and the attention and indexer widths follow the prefill width
        plan = opt.engine_plan(ds41('--extra-env', 'GR8_DS_ENGRAM=0 GR8_DEC_SPLIT=0 GR8_DS_MTP=0 GR8_DS_PREFILL_W=1024'))
        self.assertEqual(sorted(e for e in plan['env'] if e.startswith(('GR8_DS_MTP=', 'GR8_DS_ATTN_PW=', 'GR8_DS41_IDX_PW='))),
                         ['GR8_DS41_IDX_PW=1024', 'GR8_DS_ATTN_PW=1024', 'GR8_DS_MTP=0'])
        self.assertIn('GR8_DS_ENGRAM=0', plan['env'])
        self.assertEqual([e for e in plan['env'] if e.startswith('GR8_S2_DECODE_PF=')], ['GR8_S2_DECODE_PF=1024'])
        plan = opt.engine_plan(ds41('--extra-env', 'GR8_DS_PREFILL_W=512'))
        self.assertIn('GR8_S2_DECODE_PF=512', plan['env'])
        plan = opt.engine_plan(ds41('--extra-env', 'GR8_S2_DECODE_PF=0'))
        self.assertEqual([e for e in plan['env'] if e.startswith('GR8_S2_DECODE_PF=')], ['GR8_S2_DECODE_PF=0'])
        plan = opt.engine_plan(ds41('--s2-decode-pf', '512'))
        self.assertIn('GR8_S2_DECODE_PF=512', plan['env'])
        plan = opt.engine_plan(ds41('--sess', '1', '--extra-env', 'GR8_DS_PREFILL_MIN=0'))
        self.assertIn('GR8_S2_DECODE_PF=0', plan['env'])


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
            self.assertIn('no model: set GR8_MODEL to glm53 (GLM-5.3), glm53f (GLM-5.3-Flash) or ds41 (DeepSeek-V4.1-Flash)', str(e.exception))
            self.assertNotIn('pseudo', str(e.exception))

    def test_an_unknown_name(self):
        for argv, env, src in ((['--gr8-model', 'glm9'], {}, 'command line'),
                               ([], {'GR8_MODEL': 'glm53x'}, 'environment')):
            with self.assertRaises(opt.OptionError) as e:
                opt.parse(['--no-config', *argv], env)
            self.assertIn(f'from the {src}: not a model gr8 serves. Want glm53 (GLM-5.3), glm53f (GLM-5.3-Flash) or ds41 (DeepSeek-V4.1-Flash)',
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
        self.assertIn('--tp       4, 3, 2 or 1', flash)
        self.assertIn('TP=1 k2', flash)
        self.assertIn('--drafter URL (DFlash2), --kv-format int4', flash)
        self.assertEqual(models.SERVED, (models.GLM53, models.GLM53F, models.DS41))
        ds = models.describe(models.DS41)
        self.assertIn('a local copy of an exlr8 release of DeepseekV41ForCausalLM', ds)
        self.assertIn('--variant  none: the release is read in place', ds)
        self.assertNotIn('by default', ds)
        self.assertIn('ds41 (DeepSeek-V4.1-Flash)', models.CHOICES)


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
