#!/usr/bin/env python3
"""The option parser, the config file, precedence and the canonical option set. No network, no GPU.
    python3 -m unittest discover -s tools/launch"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402
import options as opt  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'
GLM = {'GR8_MODEL': 'glm53'}   # the model: a boot names it (no default)


def write(d, name, text):
    p = os.path.join(d, name)
    with open(p, 'w') as f:
        f.write(text)
    return p


class FileTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_parse(self):
        p = write(self.d, 'c.env', '# comment\n\nexport CAP=262144\nEXTRA_ENV="A=1 B=2"\nSESS=4   # sessions\n'
                                    "MODEL='/data/x y'\nHOSTS=\"n1 n2\"\n")
        self.assertEqual(opt.parse_file(p), {'CAP': '262144', 'EXTRA_ENV': 'A=1 B=2', 'SESS': '4', 'MODEL': '/data/x y',
                                             'HOSTS': 'n1 n2'})

    def test_refusals_name_the_line(self):
        for text, want in (('CAP=1\nnot a line\n', ':2: not KEY=value'), ('QP_CTX=$CAP\n', ':1: QP_CTX: the file is parsed'),
                           ('A="x\n', ':1: unterminated quote'), ('A="x" y\n', 'text after the closing quote')):
            p = write(self.d, 'c.env', text)
            with self.assertRaises(opt.OptionError) as e:
                opt.parse_file(p)
            self.assertIn(want, str(e.exception))


class PrecedenceTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.f = write(self.d, 'c.env', f'PEERS={PEERS}\nCAP=1000\nSESS=4\nMTP=3\nGR8_WORK_DIR=/from/file\nFOO=1\n')

    def test_command_line_then_file_then_environment_then_default(self):
        env = {'CAP': '9', 'SESS': '9', 'MTP': '9', 'REUSE': '1'}
        o = opt.parse(['--cap', '2000', '@' + self.f, '--sess=2'], env)
        self.assertEqual((o.get('CAP'), o.origin['CAP']), ('2000', 'command line'))
        self.assertEqual((o.get('SESS'), o.origin['SESS']), ('2', 'command line'))
        self.assertEqual((o.get('MTP'), o.origin['MTP']), ('3', '@file'))
        self.assertEqual((o.get('REUSE'), o.origin['REUSE']), ('1', 'environment'))
        self.assertEqual((o.get('PORT'), o.origin['PORT']), ('8100', 'default'))
        self.assertIn('unknown key FOO', o.warnings[0])

    def test_old_names_still_work_with_a_warning(self):
        f = write(self.d, 'old.env', f'PEERS={PEERS}\nG6_HOME=/from/file\nG6ENV_X="G6_QA_SPLIT=1"\n')
        o = opt.parse(['@' + f], {})
        self.assertEqual((o.get('GR8_WORK_DIR'), o.get('EXTRA_ENV')), ('/from/file', 'G6_QA_SPLIT=1'))
        self.assertEqual(len([w for w in o.warnings if 'is the old name of' in w]), 2)
        o = opt.parse(['--no-config', '--g6-home', '/a', 'G6ENV_X=X=1'], {'G6_CLUSTER': f})
        self.assertEqual((o.get('GR8_WORK_DIR'), o.get('EXTRA_ENV')), ('/a', 'X=1'))
        o = opt.parse(['--no-config'], {'G6_HOME': '/env', 'GR8_WORK_DIR': '/new'})
        self.assertEqual(o.get('GR8_WORK_DIR'), '/new')     # the new name wins
        self.assertEqual(opt.parse([], {'G6_CLUSTER': f}).file, f)

    def test_old_engine_names_in_extra_env_become_one_variable(self):
        o = opt.parse(['--no-config', '--peers', PEERS, '--tp', '4', '--extra-env', 'G6_QA_SPLIT=1 GR8_IDXQ_SPLIT=0'], GLM)
        env = opt.engine_plan(o)['env']
        self.assertIn('GR8_QA_SPLIT=1', env)
        self.assertEqual([e for e in env if e.startswith(('G6_', 'G9_'))], [])

    def test_test_knobs_are_refused(self):
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(opt.parse(['--no-config', '--peers', PEERS, '--extra-env', 'GR8_TEST_S2_RUNS=0:4'], GLM))
        self.assertIn('GR8_TEST_S2_RUNS: a test knob', str(e.exception))
        for old, why in (('G6_S2_RUNS', 'a test knob'), ('GR8_NO_SUCH', 'not an engine variable')):
            with self.assertRaises(opt.OptionError) as e:
                opt.validate(opt.parse(['--no-config', '--peers', PEERS, '--extra-env', f'{old}=1'], GLM))
            self.assertIn(f'{old}: {why}', str(e.exception))

    def test_default_file_and_no_config(self):
        o = opt.parse([], {'GR8_CLUSTER': self.f})
        self.assertEqual((o.file, o.get('CAP')), (self.f, '1000'))
        o = opt.parse(['--no-config'], {'GR8_CLUSTER': self.f})
        self.assertEqual((o.file, o.get('CAP')), (None, None))
        here = os.getcwd()
        try:
            os.chdir(self.d)
            write(self.d, 'cluster.env', 'CAP=77\n')
            self.assertEqual(opt.parse([], {}).get('CAP'), '77')
        finally:
            os.chdir(here)

    def test_hosts_are_peers(self):
        p = write(self.d, 'old.env', 'HOSTS="10.0.0.1 10.0.0.2 10.0.0.3"\nIPS=a,b,c\nIPS2=d,e,f\n')
        o = opt.parse(['@' + p], GLM)
        self.assertEqual((opt.mode(o), o.get('PEERS')), ('peers', '10.0.0.1,10.0.0.2,10.0.0.3'))
        opt.validate(o)
        self.assertEqual(o.get('TP'), '3')

    def test_mode(self):
        self.assertEqual(opt.mode(opt.parse(['--no-config'], {})), 'mentat')
        o = opt.parse(['@' + self.f, '--mentat', 'localhost'], {})
        self.assertEqual((opt.mode(o), o.get('PEERS')), ('mentat', None))
        o = opt.parse(['--no-config'], {'PEERS': PEERS})
        self.assertEqual(opt.mode(o), 'peers')
        with self.assertRaises(opt.OptionError):
            opt.parse(['--peers', PEERS, '--mentat', 'localhost'], {})

    def test_key_value_words(self):
        o = opt.parse(['--no-config', 'V2=/v2', 'CAP=1000', '--sess', '2'], {})
        self.assertEqual((o.get('V2'), o.get('CAP'), o.origin['CAP'], o.args), ('/v2', '1000', 'command line', []))
        for bad in ('V2_WIDTH=/w.json', 'PACKS=/p'):
            with self.assertRaises(opt.OptionError) as e:
                opt.parse(['--no-config', bad], {})
            self.assertIn(f'unknown option {bad.split("=")[0]}', str(e.exception))
        self.assertEqual(opt.parse(['--no-config', '3', 'shim'], {}).args, ['3', 'shim'])

    def test_flags(self):
        with self.assertRaises(opt.OptionError):
            opt.parse(['--nope', '1'], {})
        with self.assertRaises(opt.OptionError):
            opt.parse(['--cap'], {})
        with self.assertRaises(opt.OptionError):
            opt.parse(['@a', '@b'], {})
        o = opt.parse(['--no-config', '--no-check', '--work-dir', '/h', '--s2-port', '9'], {})
        self.assertEqual((o.get('CHECK'), o.get('GR8_WORK_DIR'), o.get('S2PORT')), ('0', '/h', '9'))


class ValidateTests(unittest.TestCase):
    def check(self, argv, want, env=None):
        o = opt.parse(['--no-config', *argv], {**GLM, **(env or {})})
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(o)
        self.assertIn(want, str(e.exception))

    def test_a_ring_with_mentat(self):
        o = opt.parse(['--no-config', '--mentat', 'localhost', '--tp', '3', '--shape', 'ring'], GLM)
        opt.validate(o)
        self.assertEqual((opt.mode(o), o.get('SHAPE')), ('mentat', 'ring'))

    def test_refusals(self):
        self.check(['--peers', PEERS, '--shape', 'star'], 'want mesh or ring')
        self.check(['--peers', ','.join(f'10.0.0.{i}' for i in range(1, 7)), '--shape', 'ring'], 'TP=3 or 4')
        self.check(['--mentat', 'localhost', '--tp', '6', '--shape', 'ring'], 'TP=3 or 4')
        self.check(['--mentat', 'localhost', '--tp', '4', '--shape', 'ring', '--ips', 'a,b,c,d', '--ips2', 'a,b,c,d'],
                   'no --ips')
        self.check(['--peers', PEERS, '--shape', 'ring', '--ips', 'a,b,c,d', '--ips2', 'a,b,c,d'], 'no --ips')
        self.check(['--peers', PEERS, '--shape', 'ring', '--ring-devs', 'a,b'], 'want four devices')
        self.check(['--peers', PEERS, '--tp', '7'], 'want 4, 6, 3 or 5')
        self.check(['--peers', PEERS, '--tp', '6'], 'names 4 boxes for --tp 6')
        self.check(['--mentat', 'localhost'], '--tp is needed')
        self.check(['--peers', PEERS], 'PLAIN=1', {'PLAIN': '1'})
        self.check(['--peers', PEERS, '--ips', 'a,b,c,d'], 'IPS and IPS2 go together')
        self.check(['--peers', PEERS, '--ips', 'a,b', '--ips2', 'a,b'], 'IPS has 2 addresses')
        self.check(['--peers', PEERS, '--default-max-tokens', '4k'], 'want a whole number')
        self.check(['--peers', PEERS, '--default-reasoning-effort', 'none'], 'cannot turn thinking off')
        self.check(['--peers', PEERS, '--default-reasoning-effort', 'auto'], 'want minimal, low, medium')
        self.check(['--peers', PEERS, '--default-thinking-budget', '-1'], 'want a whole number')
        o = opt.parse(['--no-config', '--peers', PEERS, '--default-reasoning-effort', 'medium',
                       '--default-thinking-budget', '4096'], GLM)
        opt.validate(o)
        self.assertEqual((o.get('DEFAULT_REASONING_EFFORT'), o.get('DEFAULT_THINKING_BUDGET')), ('medium', '4096'))
        self.check(['--peers', PEERS, '--allow-thinking-off', 'yes'], 'want 0 or 1')
        self.check(['--peers', PEERS, '--undeclared-tool-calls', 'on'], 'want text or emit')
        self.assertEqual(opt.parse(['--no-config', '--peers', PEERS], {}).get('UNDECLARED_TOOL_CALLS'), 'text')
        self.check(['--peers', PEERS, '--grammar', '2'], 'want 0 or 1')
        self.assertEqual(opt.parse(['--no-config', '--peers', PEERS], {}).get('GRAMMAR'), '1')
        # a grammar directory reaches the engine as --grammar-dir, a path per box
        o = opt.parse(['--no-config', '--peers', PEERS, '--grammar', '1', '--grammar-dir', '/t'], GLM)
        opt.validate(o)
        self.assertIn('--grammar-dir @GRAMMAR_DIR', ' '.join(opt.engine_plan(o)['args']))
        o = opt.parse(['--no-config', '--peers', PEERS, '--allow-thinking-off', '1'], GLM)
        opt.validate(o)
        self.assertEqual(o.get('ALLOW_THINKING_OFF'), '1')

    def test_peers_resolve(self):
        o = opt.parse(['--no-config', '--peers', '127.0.0.1,127.0.0.2'], {})
        self.assertEqual(opt.resolve_peers(o), ['127.0.0.1', '127.0.0.2'])
        o = opt.parse(['--no-config', '--peers', '127.0.0.1,127.0.0.1'], {})
        with self.assertRaises(opt.OptionError):
            opt.resolve_peers(o)
        o = opt.parse(['--no-config', '--peers', 'no-such-box.invalid,127.0.0.1'], {})
        with self.assertRaises(opt.OptionError) as e:
            opt.resolve_peers(o)
        self.assertIn('not ssh aliases', str(e.exception))


class GetTests(unittest.TestCase):
    """launch.py get: TP for ./gr8 glm53 build, PORT for smoke and bench, on a box whose options need not boot."""
    def get(self, *argv):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'launch.py'), 'get',
                            *argv], env={**os.environ, **GLM}, capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip(), r.stderr

    def test_port_needs_no_bootable_options(self):
        d = tempfile.mkdtemp()
        self.assertEqual(self.get('PORT', '--no-config')[:2], (0, '8100'))    # up's options came on its command line
        self.assertEqual(self.get('PORT', '@' + write(d, 'c.env', 'PORT=8200\n'))[:2], (0, '8200'))
        rc, _, err = self.get('PORT', '@' + write(d, 'bad.env', 'PORT=$X\n'))
        self.assertEqual(rc, 2)
        self.assertIn('the file is parsed', err)

    def test_tp_is_still_validated(self):
        self.assertEqual(self.get('TP', '--no-config', '--peers', PEERS)[:2], (0, '4'))
        rc, _, err = self.get('TP', '--no-config')
        self.assertEqual(rc, 2)
        self.assertIn('--tp is needed', err)


class KnobsTests(unittest.TestCase):
    """launch.py knobs, which ./gr8 knobs and ./gr8 <model> knobs run."""
    def knobs(self, model, *argv):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'launch.py'),
                            'knobs', '--no-config', *argv], env={**os.environ, 'GR8_MODEL': model},
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def heads(self, text):
        return [line.split()[0] for line in text.splitlines() if line and not line.startswith((' ', '-')) and
                line.split()[0].isupper() and len(line.split()) > 1 and line.split()[1].startswith('--')]

    def test_every_option_is_listed(self):
        import kvnode
        self.assertEqual(self.heads(self.knobs('')), list(opt.KEYS) + list(kvnode.KEYS))

    def test_a_model_shows_each_tp(self):
        out = self.knobs('glm53', 'CAP')
        self.assertEqual(self.heads(out), [k for k in opt.KEYS if 'CAP' in k])
        lines = out.splitlines()
        values = lines[lines.index(next(x for x in lines if x.startswith('CAP '))) + 1]
        self.assertEqual(re.findall(r'TP=(\d+) \d+', values), list(models.GLM53.tp))

    def test_tp_and_options_shape_the_values(self):
        out = self.knobs('glm53', '--tp', '6', '--cap', '1000', 'CAP')
        self.assertIn('    TP=6 1000 (set by the command line)\n', out)
        # an engine knob set through EXTRA_ENV, under its old name, shows the value the engine gets
        self.assertIn('    TP=4 1\n', self.knobs('glm53', '--tp', '4', '--extra-env', 'G6_QA_SPLIT=1', 'QA_SPLIT'))


class RestartTests(unittest.TestCase):
    def test_restart_is_refused_inside_the_container(self):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'launch.py'),
                            'restart', '--no-config'], env={**os.environ, 'GR8_CONTAINER': '1'},
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn('run down, then `docker compose up -d` on every box', r.stderr)
        self.assertEqual(r.stdout, '')           # down never ran


class PathTests(unittest.TestCase):
    def test_local_paths_become_absolute(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, 'glm', 'packs'))
        here = os.getcwd()
        try:
            os.chdir(d)
            o = opt.parse(['--no-config', '--model', 'glm/packs', '--model-cache', 'cache', '--work-dir', '.'], {})
            self.assertEqual(o.get('MODEL'), os.path.join(os.path.realpath(d), 'glm', 'packs'))
            self.assertEqual(o.get('MODEL_CACHE'), os.path.join(os.path.realpath(d), 'cache'))
            self.assertEqual(o.get('GR8_WORK_DIR'), os.path.realpath(d))
            remote = 'https://huggingface.co/org/repo@main'
            self.assertEqual(opt.parse(['--no-config', '--model', remote], {}).get('MODEL'), remote)
            # a bare org/repo is a local path, and only an https:// URL names a hub repo
            self.assertEqual(opt.parse(['--no-config', '--model', 'zai-org/GLM-5.3'], {}).get('MODEL'),
                             os.path.join(os.path.realpath(d), 'zai-org', 'GLM-5.3'))
        finally:
            os.chdir(here)


class CanonicalTests(unittest.TestCase):
    def digest(self, *argv, env=None):
        o = opt.parse(['--no-config', '--peers', PEERS, *argv], {**GLM, **(env or {})})
        opt.validate(o)
        c, text, d = opt.canonical(o, opt.engine_plan(o), opt.split_list(PEERS))
        return c, text, d

    def test_one_key_differs(self):
        a, _, da = self.digest('--cap', '262144')
        b, _, db = self.digest('--cap', '131072')
        self.assertNotEqual(da, db)
        self.assertEqual(sorted(k for k in a if a[k] != b[k]), ['CAP', 'engine.args', 'engine.env'])

    def test_defaults_resolved_and_equal_to_explicit(self):
        a, text, da = self.digest()
        self.assertIn('CAP=485376\n', text)
        self.assertIn('--s2-sessions 8 ', text)
        self.assertIn(' --reuse-prefix', text)            # prefix reuse is on by default at TP=4
        self.assertEqual(da, self.digest('--cap', '485376', '--share-limit', '32768', '--sess', '8', '--dflash', '1')[2])

    def test_l2pf_reaches_every_rank(self):
        _, text, _ = self.digest()
        self.assertNotIn('GR8_L2PF', text)
        _, text, _ = self.digest('--l2pf', '15')
        self.assertIn(' GR8_L2PF=15', text)

    def test_cp_step_rows_reaches_every_rank(self):
        _, text, d0 = self.digest()
        self.assertNotIn('G6_CP_STEP_', text)
        _, text, d8 = self.digest('--cp-step-rows', '8')
        self.assertIn(' GR8_CP_STEP_ROWS=8', text)
        self.assertNotIn('GR8_CP_STEP_SEG', text)
        self.assertNotEqual(d0, d8)
        _, text, _ = self.digest('--cp-step-rows', '8', '--cp-step-seg', '1')
        self.assertIn(' GR8_CP_STEP_ROWS=8', text)
        self.assertIn(' GR8_CP_STEP_SEG=1', text)

    def test_the_share_limit_sets_r_and_an_unset_cap_follows_it(self):
        _, text, _ = self.digest('--share-limit', '32768')
        self.assertIn('CAP=485376\n', text)              # 2,280 physical pages per rank at TP=4, as at 65536
        self.assertIn('GR8_CP_R=32768 ', text)
        _, text, _ = self.digest('--share-limit', '32768', '--cap', '200000')
        self.assertIn('CAP=200000\n', text)
        _, text, _ = self.digest('--share-limit', '98304')
        self.assertIn('CAP=288768\n', text)              # 98304 + (2280 - 1536) x 64 x 4
        self.assertIn('GR8_CP_R=98304 ', text)
        for bad in ('3000', '4095', '4100', '0', 'x'):
            with self.assertRaises(opt.OptionError):
                self.digest('--share-limit', bad)

    def test_a_busy_share_limit_sets_r2_with_the_rows_window(self):
        rows = ('--kv-rows', 'mem:,size=1G,local=40')
        _, text, d0 = self.digest(*rows)
        self.assertNotIn('GR8_CP_R2', text)
        _, text, d1 = self.digest(*rows, '--share-limit-busy', '16384')
        self.assertIn(' GR8_CP_R2=16384 ', text)
        self.assertIn(' GR8_CP_R=32768 ', text)
        self.assertNotEqual(d0, d1)
        _, text, _ = self.digest(*rows, '--share-limit-busy', '16384', '--share-big', '2')
        self.assertIn(' GR8_CP_R2_BIG=2 ', text)
        for bad in (('--share-limit-busy', '16384'), (*rows, '--share-limit-busy', '32768'),
                    (*rows, '--share-limit-busy', '16100'), ('--share-big', '2')):
            with self.assertRaises(opt.OptionError):
                self.digest(*bad)

    def test_per_box_options_do_not_count(self):
        base = self.digest()[2]
        self.assertEqual(base, self.digest('--work-dir', '/elsewhere', '--proxy-cpu', '15', '--model', '/m',
                                           '--model-cache', '/c', '--join-timeout', '5', '--node-ip', '10.0.0.2')[2])
        self.assertNotEqual(base, self.digest('--nohash', '1')[2])    # it reaches every rank's engine
        self.assertNotEqual(base, self.digest('--extra-env', 'GR8_DEC_BK=1024')[2])
        self.assertNotEqual(base, self.digest('--port', '8200')[2])
        # a knob set to its default, or one this TP does not read, changes nothing
        self.assertEqual(base, self.digest('--qp-roll', '2', '--qp-ctx', '485376', '--reuse', '1', '--extra-args', '--top 5')[2])

    def test_cp_onechain_reaches_every_rank(self):
        _, text, _ = self.digest()
        self.assertNotIn('GR8_CP_ONECHAIN', text)
        _, text, _ = self.digest('--cp-onechain', '1')
        self.assertIn(' GR8_CP_ONECHAIN=1', text)

    def test_template_has_no_paths(self):
        _, text, _ = self.digest('--work-dir', '/very/own/home', '--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json')
        self.assertNotIn('/very/own/home', text)
        self.assertIn('--serve @HOME/s2', text)
        self.assertIn('--manifest @RANKDIR/rank@R.manifest.json', text)
        self.assertIn('--v2 @V2', text)
        self.assertIn('--v2-widths @V2_WIDTHS', text)

    def test_dsel_model_reaches_every_rank(self):
        _, text, _ = self.digest()
        self.assertNotIn('GR8_S2_DSEL_MODEL', text)
        tab = 'v2:46.63,v3:54.71,v8:96.82,d:7.285,m:2.039,e:1.155,p:5.026,w:1,r:0.95'
        _, text, _ = self.digest('--dsel-model', '1', '--dsel-table', tab)
        self.assertIn(' GR8_S2_DSEL_MODEL=1 GR8_S2_DSEL_TABLE=' + tab, text)

    def test_ngram_reaches_every_rank(self):
        _, text, _ = self.digest()
        self.assertNotIn('GR8_NGRAM', text)
        _, text, _ = self.digest('--ngram', '/tables/ngram.tab', '--ngram-kmax', '5', '--ngram-kmin', '3',
                                 '--ngram-margin', '0.2')
        self.assertIn(' GR8_NGRAM=/tables/ngram.tab GR8_NGRAM_KMAX=5 GR8_NGRAM_KMIN=3 GR8_NGRAM_MARGIN=0.2', text)

    def test_release_alone_has_no_rank_pack(self):
        _, text, _ = self.digest('--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel')
        self.assertIn('--v2-dense @V2_DENSE', text)
        self.assertNotIn('--manifest', text)
        self.assertNotIn('--v2-dense-form', text)
        _, text, _ = self.digest('--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel',
                                 '--v2-dense-form', 'rounded')
        self.assertIn('--v2-dense @V2_DENSE --v2-dense-form rounded', text)
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5,10.0.0.6', '--v2', '/v2',
                       '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel'], GLM)
        opt.validate(o)
        p = opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')
        self.assertIn('GR8_PF_OWN=3', p['env'])
        self.assertNotIn('--qo-manifest', p['args'])          # PF_OWN=3's full-head weights come from the release
        with self.assertRaises(opt.OptionError):
            self.digest('--v2-dense-form', 'exact')

    def test_mtp_kstopb_reaches_every_rank(self):
        _, text, _ = self.digest()
        self.assertNotIn('GR8_MTP_KSTOPB', text)
        _, text, _ = self.digest('--mtp-kstopb', '1', '--mtp-kstopb-tau', '0.6,0.88',
                                 '--mtp-kstopb-tau-s', '1:0.7,0.7/2:0.7,0.8/4:0.7,0.75')
        self.assertIn(' GR8_MTP_KSTOPB=1 GR8_MTP_KSTOPB_TAU=0.6,0.88 GR8_MTP_KSTOPB_TAU_S1=0.7,0.7 GR8_MTP_KSTOPB_TAU_S2=0.7,0.8'
                      ' GR8_MTP_KSTOPB_TAU_S4=0.7,0.75', text)
        for bad in ('0.7,0.7', '9:0.7', '1:', '1:0.7/x'):
            with self.assertRaises(opt.OptionError):
                self.digest('--mtp-kstopb', '1', '--mtp-kstopb-tau-s', bad)
    def test_kvb_fp8_needs_a_release(self):
        rel = ['--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel']
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'] + rel + ['--kvb', 'fp8'], GLM)
        opt.validate(o)
        p = opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')
        self.assertIn('GR8_KVB=fp8', p['env'])
        self.assertNotEqual(self.digest(*rel)[2], self.digest(*rel, '--kvb', 'fp8')[2])   # a cluster option
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'] + rel, GLM)
        opt.validate(o)
        self.assertNotIn('GR8_KVB', opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')['env'])
        with self.assertRaises(opt.OptionError):
            self.digest('--kvb', 'fp8')                        # no model at all: rank packs hold kv_b in BF16
        self.digest('--model', '/some/model', '--kvb', 'fp8')   # a --model may be a release (the engine checks)
        with self.assertRaises(opt.OptionError):
            self.digest(*rel, '--kvb', 'int4')

    def test_a_layout_counts_by_its_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (os.path.join(d, n) for n in ('a.json', 'b.json', 'c.json'))
            for p, t in ((a, '{"x": 1}'), (b, '{"x": 1}'), (c, '{"x": 2}')):
                with open(p, 'w') as f:
                    f.write(t)
            rel = ('--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel')
            _, text, da = self.digest(*rel, '--layout', a)
            self.assertIn('--v2-dense @V2_DENSE --layout @LAYOUT', text)
            self.assertNotIn(d, text)
            self.assertEqual(da, self.digest(*rel, '--layout', b)[2])      # another path, the same file
            self.assertNotEqual(da, self.digest(*rel, '--layout', c)[2])
            self.assertNotEqual(da, self.digest(*rel)[2])
            with self.assertRaises(opt.OptionError):
                self.digest(*rel, '--layout', os.path.join(d, 'none.json'))
            # a box's cache fetches the layout's pieces (hfsource.Release)
            self.digest('--model', 'https://huggingface.co/org/repo', '--layout', a)

    def test_tp3_is_s2_without_drafter(self):
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3'], GLM)
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertIn('GR8_PF_OWN=1', p['env'])
        self.assertEqual(p['args'][:2], ['--manifest', '/r/rank2.manifest.json'])
        self.assertEqual(p['args'][-2:], ['--set-id', 'ab'])
        self.assertIn('/h/s2', p['args'])
        self.assertIn('--reuse-prefix', p['args'])
        self.assertFalse(any(a.startswith('--dflash') for a in p['args']))

    def test_tp4_serve_options_default_on_and_override(self):
        on = ['GR8_ROUTE_OVL=1', 'GR8_DRAFT_HEAD_I4=1', 'GR8_ATTN_SC2=4', 'GR8_ATTN_SC2_TR=2', 'GR8_ATTN_CTX2=4', 'GR8_ATTN_CTX3=2',
              'GR8_ATTN_SEL2=16', 'GR8_IDX_GRP=16', 'GR8_ATTN_QT=1', 'GR8_ATTN_FRONT=1', 'GR8_IDX_VEC=1', 'GR8_DEC_BK=512',
              'GR8_S2_ADAPT=1', 'GR8_S2_DECODE_PF=4096', 'GR8_S2_ADMIT_HOLD=1', 'GR8_PF_SELX_CTX=53248']
        def env(peers, *argv):
            o = opt.parse(['--no-config', '--peers', peers, *argv], GLM)
            opt.validate(o)
            return opt.engine_plan(o, rank=0, rank_dir='/r', aux_dir='/a', home='/h')['env']
        e = env(PEERS)
        for v in on:
            self.assertIn(v, e)
        e = env(PEERS, '--route-ovl', '0', '--attn-sc2', '0', '--draft-head-i4', '0')
        self.assertIn('GR8_ROUTE_OVL=0', e)
        self.assertIn('GR8_ATTN_SC2=0', e)
        self.assertIn('GR8_DRAFT_HEAD_I4=0', e)
        self.assertFalse(any(a.startswith('GR8_DRAFT_HEAD_I4') for a in env(PEERS, '--mtp', '0', '--dflash', '0')))
        e = env(PEERS, '--dec-bk', '0', '--pf-selx-ctx', '0')   # 0 = the engine default: not passed
        self.assertFalse(any(a.startswith(('GR8_DEC_BK=', 'GR8_PF_SELX_CTX=')) for a in e))
        e = env(PEERS, '--extra-env', 'GR8_ROUTE_OVL=0 GR8_DEC_BK=2048')   # EXTRA_ENV keeps its own value, no second copy
        self.assertEqual([a for a in e if a.startswith(('GR8_ROUTE_OVL=', 'GR8_DEC_BK='))], ['GR8_ROUTE_OVL=0', 'GR8_DEC_BK=2048'])
        e = env('10.0.0.1,10.0.0.2,10.0.0.3')   # TP=3 unchanged
        self.assertFalse(any(a.split('=')[0] in [v.split('=')[0] for v in on] for a in e))
    def test_tp5_prefills_with_pf_own_3(self):
        def env(*argv):
            o = opt.parse(['--no-config', '--peers', ','.join(f'10.0.0.{i}' for i in range(1, 6)), *argv], GLM)
            opt.validate(o)
            return opt.engine_plan(o, rank=4, rank_dir='/r', aux_dir='/a', home='/h')['env']
        self.assertIn('GR8_PF_OWN=3', env())
        self.assertIn('GR8_PF_OWN=1', env('--pf-own', '1'))

    def test_tp4_tr3_w2all_default_on_and_override(self):
        def env(peers, *argv):
            o = opt.parse(['--no-config', '--peers', peers, *argv], GLM)
            opt.validate(o)
            return [a for a in opt.engine_plan(o, rank=0, rank_dir='/r', aux_dir='/a', home='/h')['env'] if a.startswith('GR8_TR3_W2ALL=')]
        self.assertEqual(env(PEERS), ['GR8_TR3_W2ALL=1'])
        self.assertEqual(env(PEERS, '--tr3-w2all', '0'), ['GR8_TR3_W2ALL=0'])
        self.assertEqual(env(PEERS, '--extra-env', 'GR8_TR3_W2ALL=0'), ['GR8_TR3_W2ALL=0'])   # EXTRA_ENV keeps its value, no second copy
        self.assertEqual(env('10.0.0.1,10.0.0.2,10.0.0.3'), [])   # TP=3 unchanged

    def test_tp6_is_s2_without_drafter(self):
        peers = ','.join(f'10.0.0.{i}' for i in range(1, 7))
        o = opt.parse(['--no-config', '--peers', peers], GLM)
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertIn('GR8_PF_OWN=3', p['env'])
        self.assertIn('/a/qo.pack', p['args'])
        self.assertIn('/h/s2', p['args'])
        self.assertIn('--reuse-prefix', p['args'])
        self.assertIn('--s2-sessions', p['args'])
        self.assertNotIn('--spec-adaptive', p['args'])
        self.assertFalse(any(a.startswith('--dflash') for a in p['args']))
        # --dflash 1: the drafter (its body replicated on every rank), adaptive draft count as at TP=4
        o = opt.parse(['--no-config', '--peers', peers, '--dflash', '1'], GLM)
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertIn('/a/dflash.r2.manifest.json', p['args'])
        self.assertIn('/a/dflash.r2.pack', p['args'])
        self.assertIn('--spec-adaptive', p['args'])
        # PF_OWN=1 is the fallback: no QO pack
        o = opt.parse(['--no-config', '--peers', peers, '--pf-own', '1'], GLM)
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertIn('GR8_PF_OWN=1', p['env'])
        self.assertNotIn('/a/qo.pack', p['args'])


class KvStoreTests(unittest.TestCase):
    def plan(self, *argv, kv_store=None, env=None):
        o = opt.parse(['--no-config', '--peers', PEERS, *argv], {**GLM, **(env or {})})
        opt.validate(o)
        return o, opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h', kv_store=kv_store)

    def stores(self, p):
        return [p['args'][i + 1] for i, a in enumerate(p['args']) if a == '--kv-store']

    def arg(self, p, name):
        return p['args'][p['args'].index(name) + 1] if name in p['args'] else None

    def test_off_by_default(self):
        o, p = self.plan()
        self.assertEqual(opt.kv_stores(o), [])
        for a in ('--kv-store', '--kv-keep', '--kv-system-after', '--kv-write-limit', '--kv-tier'):
            self.assertNotIn(a, p['args'])
        self.assertNotIn('--kv-store', self.plan('--kv-store', 'off')[1]['args'])

    def test_a_group_lease_gives_when_idle_or_always(self):
        b = 'remote:192.0.2.5:7350,size=8G,group=glm,cap=32G,weight=2000'
        o, p = self.plan('--kv-store', b, '--kv-share-when', 'always')
        self.assertEqual(self.stores(p), [b])
        self.assertIn('GR8_KV_SHARE_WHEN=always', p['env'])
        self.assertNotIn('GR8_KV_SHARE_WHEN', ' '.join(self.plan('--kv-store', b)[1]['env']))
        with self.assertRaises(opt.OptionError):
            self.plan('--kv-store', b, '--kv-share-when', 'busy')

    def test_stores_in_flag_order_and_what_they_keep(self):
        a, b = 'ssd:/nvme/gr8kv,size=200G', 'remote:192.0.2.5:7350,size=8G'
        o, p = self.plan('--kv-store', a, '--kv-store', b, '--kv-keep', 'sessions', '--kv-system-after', '3',
                         '--kv-write-limit', '0.25')
        self.assertEqual(self.stores(p), [a, b])
        self.assertEqual((self.arg(p, '--kv-keep'), self.arg(p, '--kv-system-after'), self.arg(p, '--kv-write-limit')),
                         ('sessions', '3', '0.25'))
        t = opt.kv_stores(o)
        self.assertEqual([(x['kind'], x['path'], x['host'], x['port'], x['size']) for x in t],
                         [('ssd', '/nvme/gr8kv', None, None, '200G'), ('remote', None, '192.0.2.5', '7350', '8G')])
        self.assertEqual([x['persist'] for x in t], [False, False], 'an SSD store starts empty unless persist=on')
        self.assertTrue(opt.kv_stores(self.plan('--kv-store', a + ',persist=on')[0])[0]['persist'])
        # a store's own keep= reaches the engine as written; the others take --kv-keep
        o, p = self.plan('--kv-store', a + ',persist=on,keep=system', '--kv-store', b + ',keep=all')
        self.assertEqual(self.stores(p), [a + ',persist=on,keep=system', b + ',keep=all'])
        self.assertEqual([x['keep'] for x in opt.kv_stores(o)], ['system', 'all'])
        # a named cache reaches the engine as the store's spare_ram_cache=on
        o, p = self.plan('--kv-store', 'cache:s,kind=spare_ram,floor=3G', '--kv-store', b + ',name=far,cache=s')
        self.assertEqual(self.stores(p), [b + ',spare_ram_cache=on,spare_ram_floor=3G'])
        self.assertEqual([(t['name'], t['cache']) for t in opt.kv_stores(o)], [('far', 's')])
        # the same from the environment, stores separated by ;
        self.assertEqual(self.stores(self.plan(env={'KV_STORE': f'{a};{b}'})[1]), [a, b])

    def test_the_deprecated_tier_options_map_to_the_nearest_policy(self):
        o, p = self.plan('--kv-tier', 'ssd:/x,size=200G,keep=shared', '--kv-shared-after', '5')
        self.assertEqual((self.stores(p), self.arg(p, '--kv-keep'), self.arg(p, '--kv-system-after')),
                         (['ssd:/x,size=200G'], 'system', '5'))
        self.assertTrue(any('deprecated' in w for w in o.warnings))
        self.assertNotIn('--kv-tier', p['args'])
        _, p = self.plan('--kv-tier', 'ssd:/x,size=200G,keep=shared', '--kv-tier', 'remote:192.0.2.5:7350,size=8G')
        self.assertEqual(self.arg(p, '--kv-keep'), 'all', 'a tier that kept evicted runs (the default) maps to all')
        _, p = self.plan(env={'KV_TIER': 'ssd:/x,size=8G,keep=evicted', 'KV_KEEP': 'sessions'})
        self.assertEqual(self.arg(p, '--kv-keep'), 'sessions', 'an explicit KV_KEEP wins')

    def test_remote_auto_is_resolved_per_box(self):
        o = opt.parse(['--no-config', '--mentat', 'localhost', '--tp', '4', '--kv-store', 'remote:auto,size=8G'], GLM)
        opt.validate(o)
        self.assertEqual(opt.kv_stores(o)[0]['host'], 'auto')
        p = opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h', kv_store='remote:192.0.2.5:7350,size=8G')
        self.assertEqual(self.stores(p), ['remote:192.0.2.5:7350,size=8G'])
        self.assertIn('--kv-store remote:auto,size=8G ', opt.canonical(o, opt.engine_plan(o), None)[1])

    def test_refusals(self):
        v = ValidateTests()
        for argv, want in ((['--kv-store', 'nvme:/x,size=8G'], 'the store is'), (['--kv-store', 'ssd:/x'], 'size=N'),
                           (['--kv-store', 'ssd:/x,size=8'], 'want size='), (['--kv-store', 'ssd:/x,size=8G,keep=most'], 'keep=system, sessions or all'),
                           (['--kv-store', 'remote:192.0.2.5:7350,size=8G,persist=on'], 'for ssd:'),
                           (['--kv-store', 'ssd:/x,size=8G,persist=yes'], 'persist=on|off'),
                           (['--kv-store', 'ssd:/x,size=8G,spare_ram_cache=on'], "is the engine's"),
                           (['--kv-store', 'cache:s,kind=spare_ram', '--kv-store',
                             'remote:192.0.2.5:7350,size=8G,group=g,cache=s'], 'fixed lease'),
                           (['--kv-store', 'remote:auto,size=8G'], 'name it'), (['--kv-store', 'ssd:/x,size=8G', '--reuse', '0'], 'needs REUSE'),
                           (['--kv-store', 'ssd:/x,size=8G', '--kv-system-after', '0'], 'from 1'),
                           (['--kv-store', 'ssd:/x,size=8G', '--kv-keep', 'most'], 'system, sessions or all'),
                           (['--kv-store', 'ssd:/x,size=8G', '--kv-write-limit', 'lots'], 'per page computed')):
            v.check(['--peers', PEERS, *argv], want)


class KvStoreFileTests(unittest.TestCase):
    """KV_STORE=@FILE: tiers in a YAML file, lowered to the one-line form the engine and the logs take."""
    FILE = """
caches:
  spare: {kind: spare_ram, floor: 2G}
tiers:                       # tried in order
  - keep: system
    cache: spare
    store:
      ssd: {path: /nvme/gr8kv, size_per_rank: 4G, persist: true}
  - keep: all
    cache: spare
    store:
      remote: {address: auto, size_per_rank: 8G}
"""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def lower(self, text):
        return opt.kv_store_file(write(self.d, 'kv.yaml', text))

    def test_lowers_in_cascade_order(self):
        self.assertEqual(self.lower(self.FILE), 'cache:spare,kind=spare_ram,floor=2G;ssd:/nvme/gr8kv,size=4G,persist=on,'
                                                'keep=system,cache=spare;remote:auto,size=8G,keep=all,cache=spare')
        self.assertEqual(self.lower('tiers:\n- store:\n    remote: {address: "192.0.2.5:7350", size_per_rank: 8G, '
                                    'group: g, cap_per_rank: 32G, weight: 2000}\n'),
                         'remote:192.0.2.5:7350,size=8G,group=g,cap=32G,weight=2000')
        # two tiers of one kind need names
        two = 'tiers:\n- {name: a, store: {ssd: {path: /a, size_per_rank: 4G}}}\n' \
              '- {name: b, store: {ssd: {path: /b, size_per_rank: 4G}}}\n'
        self.assertEqual(self.lower(two), 'ssd:/a,size=4G,name=a;ssd:/b,size=4G,name=b')
        # unnamed stores of one kind are named kind-N
        self.assertEqual([t['name'] for t in opt.kv_parse('ssd:/a,size=4G;remote:auto,size=4G;ssd:/b,size=4G')[1]],
                         ['ssd-1', 'remote', 'ssd-2'])

    def test_at_file_beside_the_config_file(self):
        write(self.d, 'kv.yaml', 'tiers:\n- store: {ssd: {path: /x, size_per_rank: 4G}}\n')
        cfg = write(self.d, 'c.env', f'PEERS={PEERS}\nKV_STORE=@kv.yaml\n')
        o = opt.parse(['@' + cfg], {})
        self.assertEqual(o.get('KV_STORE'), 'ssd:/x,size=4G')
        self.assertIn('via ' + os.path.join(self.d, 'kv.yaml'), opt.summary(o))
        # an @FILE among one-line stores, from the command line: relative to the working directory
        o = opt.parse(['--no-config', '--kv-store', 'remote:192.0.2.5:7350,size=8G', '--kv-store',
                       '@' + os.path.join(self.d, 'kv.yaml')], {})
        self.assertEqual(o.get('KV_STORE'), 'remote:192.0.2.5:7350,size=8G;ssd:/x,size=4G')

    def test_the_log_shows_both_forms(self):
        o = opt.parse(['--no-config', '--kv-store', '@' + write(self.d, 'kv.yaml', self.FILE)], {})
        lines = opt.kv_describe(o)
        self.assertEqual(lines[0], 'KV_STORE=' + self.lower(self.FILE))
        self.assertEqual(lines[1], 'KV_STORE as YAML:')
        self.assertEqual(opt.kv_store_text('\n'.join(x[2:] for x in lines[2:])), self.lower(self.FILE))
        self.assertEqual(opt.kv_describe(opt.parse(['--no-config'], {})), [])

    def test_refusals_name_the_key(self):
        ssd = '{ssd: {path: /x, size_per_rank: 4G}}'
        for text, want in [
            ('', 'the file: want a mapping'),
            ('caches: {}\n', 'tiers: no tiers'),
            ('tiers: []\n', 'tiers: no tiers'),
            ('runs: []\ntiers: []\n', 'the file.runs: not built yet'),
            ('tier: []\n', 'the file.tier: want caches, tiers'),
            (f'tiers:\n- {{keep: all}}\n', 'tiers[0]: no store'),
            ('tiers:\n- store: {nvme: {path: /x}}\n', 'tiers[0].store: want one of ssd: or remote:'),
            ('tiers:\n- store: {ssd: {size_per_rank: 4G}}\n', 'tiers[0].store.ssd: no path'),
            ('tiers:\n- store: {ssd: {path: /x, size: 4G}}\n', 'tiers[0].store.ssd.size: want path, size_per_rank, persist'),
            ('tiers:\n- store: {ssd: {path: /x, size_per_rank: 4G, persist: on}}\n', None),
            ('tiers:\n- store: {ssd: {path: /x, size_per_rank: 4G, persist: maybe}}\n',
             "tiers[0].store.ssd.persist: want true or false, got 'maybe'"),
            ('tiers:\n- store: {ssd: {path: /x, size_per_rank: 4}}\n', 'tiers[0].store.ssd.size_per_rank: want a string'),
            ('tiers:\n- store: {ssd: {path: /x}}\n', 'tiers[0]: --kv-store ssd:/x: size=N(G|T) is needed'),
            (f'tiers:\n- {{name: ssd-2, store: {ssd}}}\n- store: {ssd}\n- store: {ssd}\n',
             'tiers[2]: --kv-store ssd:/x,size=4G: the name ssd-2 is also store 1\'s'),
            (f'tiers:\n- {{store: {ssd}, transforms: []}}\n', 'tiers[0].transforms: not built yet'),
            (f'tiers:\n- {{store: {ssd}, cache: spar}}\n', 'tiers[0]: --kv-store ssd:/x,size=4G,cache=spar: no cache spar (caches: none)'),
            (f'caches: {{spare: {{kind: fixed_ram}}}}\ntiers:\n- store: {ssd}\n', 'caches.spare: --kv-store cache:spare,kind=fixed_ram: kind=spare_ram'),
            (f'caches: {{spare: {{kind: spare_ram, takes: [system]}}}}\ntiers:\n- store: {ssd}\n',
             'caches.spare.takes: not built yet'),
            (f'tiers:\n- {{keep: most, store: {ssd}}}\n', 'keep=system, sessions or all'),
            ('caches: {s: {kind: spare_ram}}\ntiers:\n- cache: s\n  store: {remote: {address: "192.0.2.5:7350", '
             'size_per_rank: 8G, group: g}}\n', 'fixed lease'),
            ('tiers: [\n', 'KV_STORE=@'),
            ('tiers:\n- store: {remote: {address: auto, size_per_rank: 8G, group: g, weight: 02000}}\n', 'plain decimal'),
            ('tiers:\n- store: {remote: {address: 7350:1, size_per_rank: 8G}}\n', 'plain decimal'),
            ('a: 1\na: 2\n', 'a given twice'),
        ]:
            with self.subTest(text=text):
                if want is None:   # YAML 1.1: on is true
                    self.assertEqual(self.lower(text), 'ssd:/x,size=4G,persist=on')
                    continue
                with self.assertRaises(opt.OptionError) as e:
                    self.lower(text)
                self.assertIn(want, str(e.exception))
        with self.assertRaises(opt.OptionError) as e:
            opt.kv_store_file(os.path.join(self.d, 'missing.yaml'))
        self.assertIn('No such file', str(e.exception))


class KvFuzzTests(unittest.TestCase):
    """The one-line form and the YAML store file accept and refuse the same stores, and mean the same thing."""
    N = 3000

    @staticmethod
    def setup(rng):
        """Random caches and tiers, valid or not, as ([(name, kind, floor)], [(kind, address, {YAML key: value})])."""
        pick = rng.choice
        def some(good, bad):   # mostly good values, so both kinds of outcome get tested
            return pick(good) if rng.random() < 0.93 else pick(bad)

        caches = [(some(['spare', 's2'], ['bad name', 'a;b', 'a=b']), some(['spare_ram'], ['fixed_ram', None]),
                   some([None, '2G', '512M', '1.5G'], ['0G', '4T', 'x'])) for _ in range(pick([0, 0, 1, 1, 2]))]
        names = [c[0] for c in caches] + ['nope']
        tiers = []
        for _ in range(pick([0, 1, 1, 2, 3])):
            kind = pick(['ssd', 'remote'])
            address = some(['/nvme/a', '/b'], ['/x y', '/a,b', '']) if kind == 'ssd' else \
                some(['auto', '192.0.2.5:7350', 'kv:7350'], ['auto:7350', '', 'a;b'])
            t = {}
            if rng.random() < 0.97:
                t['size_per_rank'] = some(['4G', '1.5T', '512M'], ['4', '0G', 'x'])
            for key, good, bad, p in (('persist', [True, False], [], 0.4 if kind == 'ssd' else 0.03),
                                      ('keep', ['system', 'all', 'sessions'], ['most'], 0.5),
                                      ('name', ['a', 'b', 'c', 'd'], ['ssd', 'remote', 'x y'], 0.4),
                                      ('cache', names[:-1] or ['nope'], ['nope'], 0.4),
                                      ('group', ['g', 'h'], ['g;h'], 0.15 if kind == 'remote' else 0.02),
                                      ('cap_per_rank', ['32G'], ['x'], 0.1), ('weight', [2000, 1], [0, 'abc'], 0.1)):
                if rng.random() < p:
                    t[key] = some(good, bad or good)
            tiers.append((kind, address, t))
        return caches, tiers

    @staticmethod
    def line(caches, tiers, rng):
        one = {'size_per_rank': 'size', 'cap_per_rank': 'cap'}
        entries = [f'cache:{n}' + (f',kind={k}' if k else '') + (f',floor={f}' if f else '') for n, k, f in caches]
        for kind, address, t in tiers:
            settings = [f'{one.get(k, k)}={("on" if x else "off") if isinstance(x, bool) else x}' for k, x in t.items()]
            rng.shuffle(settings)
            entries.append(','.join([f'{kind}:{address}'] + settings))
        stores = iter([e for e in entries if not e.startswith('cache:')])
        rng.shuffle(entries)   # caches may come anywhere; the stores keep their order
        return ';'.join(e if e.startswith('cache:') else next(stores) for e in entries)

    @staticmethod
    def yaml_text(caches, tiers):
        import yaml
        doc = {}
        if caches:
            doc['caches'] = {n: {k: x for k, x in (('kind', kind), ('floor', f)) if x} for n, kind, f in caches}
        doc['tiers'] = []
        for kind, address, t in tiers:
            on_tier = {k: x for k, x in t.items() if k in ('keep', 'name', 'cache')}
            store = {('path' if kind == 'ssd' else 'address'): address}
            store.update({k: x for k, x in t.items() if k not in on_tier})
            doc['tiers'].append({**on_tier, 'store': {kind: store}})
        return yaml.safe_dump(doc, sort_keys=False)

    @staticmethod
    def outcome(f, *a):
        try:
            return f(*a)
        except opt.OptionError as e:
            return e

    def test_both_forms_agree(self):
        import random
        rng, agreed = random.Random(239), {True: 0, False: 0}
        for i in range(self.N):
            caches, tiers = self.setup(rng)
            if len({c[0] for c in caches}) < len(caches) or not (caches or tiers):
                continue   # a YAML mapping has no second key of one name; an empty one-line form is off
            line, text = self.line(caches, tiers, rng), self.yaml_text(caches, tiers)
            a, b = self.outcome(opt.kv_parse, line), self.outcome(opt.kv_store_text, text)
            if not isinstance(a, Exception) and not a[1]:
                a = opt.OptionError('no stores')   # caches alone: kv_parse refuses an unused cache first
            with self.subTest(i=i, line=line, yaml=text):
                self.assertEqual(isinstance(a, Exception), isinstance(b, Exception), f'{a!r} / {b!r}')
                agreed[not isinstance(a, Exception)] += 1
                if isinstance(a, Exception):
                    continue
                self.assertEqual(opt.kv_line(*a), b)
                self.assertEqual(opt.kv_engine(line), opt.kv_engine(b))
                self.assertEqual(opt.kv_store_text(opt.kv_yaml(*a)), b, 'the logged YAML reads back')
        self.assertGreater(min(agreed.values()), self.N // 10, f'too few of one kind: {agreed}')

    def test_mutations_only_refuse(self):
        import random
        rng = random.Random(7)
        line = ('cache:spare,kind=spare_ram,floor=2G;ssd:/nvme/gr8kv,size=4G,persist=on,keep=system,cache=spare;'
                'remote:192.0.2.5:7350,size=8G,keep=all,name=far,group=g,cap=32G,weight=2000')
        texts = [(opt.kv_parse, line), (opt.kv_store_text, opt.kv_yaml(*opt.kv_parse(line)))]
        alphabet = ',;=:{}[]-#&*!|>\'"\n\t ' + 'abcz019GT._/'
        for i in range(self.N):
            f, text = texts[i % 2]
            chars = list(text)
            for _ in range(rng.randint(1, 4)):
                at = rng.randrange(len(chars) + 1)
                if chars and rng.random() < 0.5:
                    del chars[min(at, len(chars) - 1)]
                else:
                    chars.insert(at, rng.choice(alphabet))
            got = ''.join(chars)
            with self.subTest(i=i, text=got):
                r = self.outcome(f, got)   # anything but an OptionError fails the test
                if isinstance(r, Exception):
                    continue
                caches, stores = opt.kv_parse(opt.kv_line(*r)) if f is opt.kv_parse else opt.kv_parse(r)
                self.assertEqual(opt.kv_store_text(opt.kv_yaml(caches, stores)), opt.kv_line(caches, stores))


class KvRowsTests(unittest.TestCase):
    def plan(self, *argv, kv_rows=None, env=None):
        o = opt.parse(['--no-config', '--peers', PEERS, *argv], {**GLM, **(env or {})})
        opt.validate(o)
        return o, opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h', kv_rows=kv_rows)

    def rt(self, p):
        return {k: v for k, _, v in (e.partition('=') for e in p['env']) if k.startswith('GR8_RT_')}

    def test_off_by_default(self):
        o, p = self.plan()
        self.assertIsNone(opt.kv_rows(o))
        self.assertEqual(self.rt(p), {})
        self.assertEqual(self.rt(self.plan('--kv-rows', 'off')[1]), {})

    def test_every_part_reaches_the_engine(self):
        o, p = self.plan('--kv-rows', 'remote:192.0.2.5:7350,size=12G,local=40,pool=49152,place=900,fmt=int4')
        self.assertEqual(self.rt(p), {'GR8_RT_STORE': 'remote:192.0.2.5:7350', 'GR8_RT_SIZE': str(12 << 30), 'GR8_RT_LOCAL': '40',
                                      'GR8_RT_POOL': '49152', 'GR8_RT_PLACE': '900', 'GR8_RT_FMT': 'int4'})
        t = opt.kv_rows(o)
        self.assertEqual((t['kind'], t['host'], t['port'], t['size'], t['local']), ('remote', '192.0.2.5', '7350', '12G', 40))
        # mem: (one box's memory, a test) and a fractional size
        self.assertEqual(self.rt(self.plan('--kv-rows', 'mem:,size=1.5M')[1]), {'GR8_RT_STORE': 'mem:', 'GR8_RT_SIZE': str(3 << 19)})
        # the environment form
        self.assertEqual(self.rt(self.plan(env={'KV_ROWS': 'mem:,size=1G,local=8'})[1]),
                         {'GR8_RT_STORE': 'mem:', 'GR8_RT_SIZE': str(1 << 30), 'GR8_RT_LOCAL': '8'})

    def test_remote_auto_is_resolved_per_box(self):
        o = opt.parse(['--no-config', '--mentat', 'localhost', '--tp', '4', '--kv-rows', 'remote:auto,size=8G,local=40'], GLM)
        opt.validate(o)
        self.assertEqual(opt.kv_rows(o)['host'], 'auto')
        rows = opt.kv_rows_with(o, '10.1.0.20:7351')
        self.assertEqual(rows, 'remote:10.1.0.20:7351,size=8G,local=40')
        p = opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h', kv_rows=rows)
        self.assertEqual(self.rt(p)['GR8_RT_STORE'], 'remote:10.1.0.20:7351')
        self.assertIn('GR8_RT_STORE=remote:auto', opt.canonical(o, opt.engine_plan(o), None)[1])

    def test_refusals(self):
        v = ValidateTests()
        for argv, want in ((['--kv-rows', 'ssd:/x,size=8G'], 'the store is'), (['--kv-rows', 'remote:auto:7350,size=8G'], 'the store is'),
                           (['--kv-rows', 'mem:'], 'size=N'), (['--kv-rows', 'mem:,size=8'], 'want size='),
                           (['--kv-rows', 'mem:,size=8G,local=0'], 'want size='), (['--kv-rows', 'mem:,size=8G,fmt=fp4'], 'fmt=fp8|int4'),
                           (['--kv-rows', 'mem:,size=8G,place=900'], 'place= needs local='),
                           (['--kv-rows', 'remote:192.0.2.5,size=8G'], 'names the port'),
                           (['--kv-rows', 'remote:auto,size=8G'], 'name it'),
                           (['--kv-rows', 'mem:,size=8G,fmt=int4', '--kv-format', 'int4'], 'INT4 already')):
            v.check(['--peers', PEERS, *argv], want)


class KvFormatTests(unittest.TestCase):
    def plan(self, *argv, tp='4'):
        o = opt.parse(['--no-config', '--peers', ','.join(PEERS.split(',')[:int(tp)]) if tp != '6' else
                       ','.join(f'10.0.0.{i}' for i in range(1, 7)), *argv], GLM)
        opt.validate(o)
        return o, opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')

    def test_fp8_is_the_default_and_unchanged(self):
        o, p = self.plan()
        _, p8 = self.plan('--kv-format', 'fp8')
        self.assertNotIn('--kv-format', p['args'])
        self.assertEqual((p['args'], p['knobs']['CAP']), (p8['args'], '485376'))

    def test_int4_runs_its_own_engine_with_its_own_pool(self):
        o, p = self.plan('--kv-format', 'int4')
        self.assertEqual(p['args'][-2:], ['--kv-format', 'int4'])
        self.assertEqual(p['knobs']['CAP'], str(32768 + (2390 - 512) * 64 * 4), '2,390 pages a rank at TP=4')
        self.assertEqual(self.plan('--kv-format', 'int4', '--cap', '300000')[1]['knobs']['CAP'], '300000', 'CAP wins')
        self.assertEqual(self.plan('--kv-format', 'int4', tp='3')[1]['knobs']['CAP'], str(32768 + (2480 - 512) * 64 * 3))
        self.assertEqual(self.plan('--kv-format', 'int4', tp='6')[1]['knobs']['CAP'], str(131072 + (3940 - 2048) * 64 * 6))

    def test_refusal(self):
        ValidateTests().check(['--peers', PEERS, '--kv-format', 'int8'], 'fp8 or int4')


if __name__ == '__main__':
    unittest.main()
