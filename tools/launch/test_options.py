#!/usr/bin/env python3
"""The option parser, the config file, precedence and the canonical option set. No network, no GPU.
    python3 -m unittest discover -s tools/launch"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options as opt  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'


def write(d, name, text):
    p = os.path.join(d, name)
    with open(p, 'w') as f:
        f.write(text)
    return p


class FileTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_parse(self):
        p = write(self.d, 'c.env', '# comment\n\nexport CAP=262144\nG6ENV_X="A=1 B=2"\nSESS=4   # sessions\n'
                                    "MODEL='/data/x y'\nHOSTS=\"n1 n2\"\n")
        self.assertEqual(opt.parse_file(p), {'CAP': '262144', 'G6ENV_X': 'A=1 B=2', 'SESS': '4', 'MODEL': '/data/x y',
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
        self.f = write(self.d, 'c.env', f'PEERS={PEERS}\nCAP=1000\nSESS=4\nMTP=3\nG6_HOME=/from/file\nFOO=1\n')

    def test_command_line_then_file_then_environment_then_default(self):
        env = {'CAP': '9', 'SESS': '9', 'MTP': '9', 'REUSE': '1'}
        o = opt.parse(['--cap', '2000', '@' + self.f, '--sess=2'], env)
        self.assertEqual((o.get('CAP'), o.origin['CAP']), ('2000', 'command line'))
        self.assertEqual((o.get('SESS'), o.origin['SESS']), ('2', 'command line'))
        self.assertEqual((o.get('MTP'), o.origin['MTP']), ('3', '@file'))
        self.assertEqual((o.get('REUSE'), o.origin['REUSE']), ('1', 'environment'))
        self.assertEqual((o.get('PORT'), o.origin['PORT']), ('8100', 'default'))
        self.assertIn('unknown key FOO', o.warnings[0])

    def test_default_file_and_no_config(self):
        o = opt.parse([], {'G6_CLUSTER': self.f})
        self.assertEqual((o.file, o.get('CAP')), (self.f, '1000'))
        o = opt.parse(['--no-config'], {'G6_CLUSTER': self.f})
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
        o = opt.parse(['@' + p], {})
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
        o = opt.parse(['--no-config', '--no-check', '--g6-home', '/h', '--s2-port', '9'], {})
        self.assertEqual((o.get('CHECK'), o.get('G6_HOME'), o.get('S2PORT')), ('0', '/h', '9'))


class ValidateTests(unittest.TestCase):
    def check(self, argv, want, env=None):
        o = opt.parse(['--no-config', *argv], env or {})
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(o)
        self.assertIn(want, str(e.exception))

    def test_refusals(self):
        self.check(['--peers', PEERS, '--shape', 'ring'], 'ring is not supported yet')
        self.check(['--peers', PEERS, '--tp', '5'], 'want 4, 6 or 3')
        self.check(['--peers', PEERS, '--tp', '6'], 'names 4 boxes for --tp 6')
        self.check(['--mentat', 'localhost'], '--tp is needed')
        self.check(['--peers', PEERS], 'PLAIN=1', {'PLAIN': '1'})
        self.check(['--peers', PEERS, '--ips', 'a,b,c,d'], 'IPS and IPS2 go together')
        self.check(['--peers', PEERS, '--ips', 'a,b', '--ips2', 'a,b'], 'IPS has 2 addresses')

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


class PathTests(unittest.TestCase):
    def test_local_paths_become_absolute(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, 'glm', 'packs'))
        here = os.getcwd()
        try:
            os.chdir(d)
            o = opt.parse(['--no-config', '--model', 'glm/packs', '--model-cache', 'cache', '--g6-home', '.'], {})
            self.assertEqual(o.get('MODEL'), os.path.join(os.path.realpath(d), 'glm', 'packs'))
            self.assertEqual(o.get('MODEL_CACHE'), os.path.join(os.path.realpath(d), 'cache'))
            self.assertEqual(o.get('G6_HOME'), os.path.realpath(d))
            remote = 'https://huggingface.co/org/repo@main'
            self.assertEqual(opt.parse(['--no-config', '--model', remote], {}).get('MODEL'), remote)
            # a bare org/repo is a local path, and only an https:// URL names a hub repo
            self.assertEqual(opt.parse(['--no-config', '--model', 'zai-org/GLM-5.3'], {}).get('MODEL'),
                             os.path.join(os.path.realpath(d), 'zai-org', 'GLM-5.3'))
        finally:
            os.chdir(here)


class CanonicalTests(unittest.TestCase):
    def digest(self, *argv, env=None):
        o = opt.parse(['--no-config', '--peers', PEERS, *argv], env or {})
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
        self.assertIn('CAP=393216\n', text)
        self.assertIn('--s2-sessions 8 ', text)
        self.assertIn(' --reuse-prefix', text)            # prefix reuse is on by default at TP=4
        self.assertEqual(da, self.digest('--cap', '393216', '--sess', '8', '--dflash', '1')[2])

    def test_per_box_options_do_not_count(self):
        base = self.digest()[2]
        self.assertEqual(base, self.digest('--g6-home', '/elsewhere', '--proxy-cpu', '15', '--model', '/m',
                                           '--model-cache', '/c', '--join-timeout', '5', '--node-ip', '10.0.0.2')[2])
        self.assertNotEqual(base, self.digest('--nohash', '1')[2])    # it reaches every rank's engine
        self.assertNotEqual(base, self.digest('--g6env-x', 'G6_X=1')[2])
        self.assertNotEqual(base, self.digest('--port', '8200')[2])
        # a knob set to its default, or one this TP does not read, changes nothing
        self.assertEqual(base, self.digest('--qp-roll', '2', '--qp-ctx', '393216', '--reuse', '1', '--g6args-x', '--top 5')[2])

    def test_template_has_no_paths(self):
        _, text, _ = self.digest('--g6-home', '/very/own/home', '--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json')
        self.assertNotIn('/very/own/home', text)
        self.assertIn('--s2 @HOME/s2', text)
        self.assertIn('--manifest @RANKDIR/rank@R.manifest.json', text)
        self.assertIn('--v2 @V2', text)
        self.assertIn('--v2-widths @V2_WIDTHS', text)

    def test_release_alone_has_no_rank_pack(self):
        _, text, _ = self.digest('--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel')
        self.assertIn('--v2-dense @V2_DENSE', text)
        self.assertNotIn('--manifest', text)
        self.assertNotIn('--v2-dense-form', text)
        _, text, _ = self.digest('--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel',
                                 '--v2-dense-form', 'rounded')
        self.assertIn('--v2-dense @V2_DENSE --v2-dense-form rounded', text)
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5,10.0.0.6', '--v2', '/v2',
                       '--v2-slices', '/s', '--v2-widths', '/w.json', '--v2-dense', '/rel'], {})
        opt.validate(o)
        p = opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')
        self.assertIn('G6_PF_OWN=3', p['env'])
        self.assertNotIn('--qo-manifest', p['args'])          # PF_OWN=3's full-head weights come from the release
        with self.assertRaises(opt.OptionError):
            self.digest('--v2-dense-form', 'exact')

    def test_tp3_is_s2_without_drafter(self):
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3'], {})
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertTrue(p['s2'])
        self.assertIn('G6_PF_OWN=1', p['env'])
        self.assertEqual(p['args'][:2], ['--manifest', '/r/rank2.manifest.json'])
        self.assertEqual(p['args'][-2:], ['--set-id', 'ab'])
        self.assertIn('/h/s2', p['args'])
        self.assertIn('--reuse-prefix', p['args'])
        self.assertFalse(any(a.startswith('--dflash') for a in p['args']))

    def test_tp6_serves_one_request(self):
        o = opt.parse(['--no-config', '--peers', ','.join(f'10.0.0.{i}' for i in range(1, 7))], {})
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertFalse(p['s2'])
        self.assertIn('/h/serve', p['args'])


if __name__ == '__main__':
    unittest.main()
