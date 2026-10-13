#!/usr/bin/env python3
"""The option parser, the config file, precedence and the canonical option set. No network, no GPU.
    python3 -m unittest discover -s tools/launch"""
import os
import subprocess
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
        self.check(['--peers', PEERS, '--shape', 'star'], 'want mesh or ring')
        self.check(['--peers', ','.join(f'10.0.0.{i}' for i in range(1, 7)), '--shape', 'ring'], 'TP=3 or 4')
        self.check(['--mentat', 'localhost', '--tp', '4', '--shape', 'ring'], 'needs --peers')
        self.check(['--peers', PEERS, '--shape', 'ring', '--ips', 'a,b,c,d', '--ips2', 'a,b,c,d'], 'no --ips')
        self.check(['--peers', PEERS, '--shape', 'ring', '--ring-devs', 'a,b'], 'want four devices')
        self.check(['--peers', PEERS, '--tp', '5'], 'want 4, 6 or 3')
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
                       '--default-thinking-budget', '4096'], {})
        opt.validate(o)
        self.assertEqual((o.get('DEFAULT_REASONING_EFFORT'), o.get('DEFAULT_THINKING_BUDGET')), ('medium', '4096'))

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
                            *argv], capture_output=True, text=True, timeout=60)
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
        self.assertIn('CAP=310528\n', text)
        self.assertIn('--s2-sessions 8 ', text)
        self.assertIn(' --reuse-prefix', text)            # prefix reuse is on by default at TP=4
        self.assertEqual(da, self.digest('--cap', '310528', '--share-limit', '65536', '--sess', '8', '--dflash', '1')[2])

    def test_the_share_limit_sets_r_and_an_unset_cap_follows_it(self):
        _, text, _ = self.digest('--share-limit', '32768')
        self.assertIn('CAP=408832\n', text)              # 1,981 physical pages per rank at TP=4, as at 65536
        self.assertIn('G6_CP_R=32768 ', text)
        _, text, _ = self.digest('--share-limit', '32768', '--cap', '200000')
        self.assertIn('CAP=200000\n', text)
        for bad in ('3000', '4095', '0', 'x'):
            with self.assertRaises(opt.OptionError):
                self.digest('--share-limit', bad)

    def test_per_box_options_do_not_count(self):
        base = self.digest()[2]
        self.assertEqual(base, self.digest('--g6-home', '/elsewhere', '--proxy-cpu', '15', '--model', '/m',
                                           '--model-cache', '/c', '--join-timeout', '5', '--node-ip', '10.0.0.2')[2])
        self.assertNotEqual(base, self.digest('--nohash', '1')[2])    # it reaches every rank's engine
        self.assertNotEqual(base, self.digest('--g6env-x', 'G6_X=1')[2])
        self.assertNotEqual(base, self.digest('--port', '8200')[2])
        # a knob set to its default, or one this TP does not read, changes nothing
        self.assertEqual(base, self.digest('--qp-roll', '2', '--qp-ctx', '310528', '--reuse', '1', '--g6args-x', '--top 5')[2])

    def test_template_has_no_paths(self):
        _, text, _ = self.digest('--g6-home', '/very/own/home', '--v2', '/v2', '--v2-slices', '/s', '--v2-widths', '/w.json')
        self.assertNotIn('/very/own/home', text)
        self.assertIn('--serve @HOME/s2', text)
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
        self.assertIn('G6_PF_OWN=1', p['env'])
        self.assertEqual(p['args'][:2], ['--manifest', '/r/rank2.manifest.json'])
        self.assertEqual(p['args'][-2:], ['--set-id', 'ab'])
        self.assertIn('/h/s2', p['args'])
        self.assertIn('--reuse-prefix', p['args'])
        self.assertFalse(any(a.startswith('--dflash') for a in p['args']))

    def test_tp6_is_s2_without_drafter(self):
        peers = ','.join(f'10.0.0.{i}' for i in range(1, 7))
        o = opt.parse(['--no-config', '--peers', peers], {})
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertIn('G6_PF_OWN=3', p['env'])
        self.assertIn('/a/qo.pack', p['args'])
        self.assertIn('/h/s2', p['args'])
        self.assertIn('--reuse-prefix', p['args'])
        self.assertIn('--s2-sessions', p['args'])
        self.assertNotIn('--spec-adaptive', p['args'])
        self.assertFalse(any(a.startswith('--dflash') for a in p['args']))
        # PF_OWN=1 is the fallback: no QO pack
        o = opt.parse(['--no-config', '--peers', peers, '--pf-own', '1'], {})
        opt.validate(o)
        p = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=5, set_id='ab')
        self.assertIn('G6_PF_OWN=1', p['env'])
        self.assertNotIn('/a/qo.pack', p['args'])


class KvStoreTests(unittest.TestCase):
    def plan(self, *argv, kv_store=None, env=None):
        o = opt.parse(['--no-config', '--peers', PEERS, *argv], env or {})
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
        o = opt.parse(['--no-config', '--mentat', 'localhost', '--tp', '4', '--kv-store', 'remote:auto,size=8G'], {})
        opt.validate(o)
        self.assertEqual(opt.kv_stores(o)[0]['host'], 'auto')
        p = opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h', kv_store='remote:192.0.2.5:7350,size=8G')
        self.assertEqual(self.stores(p), ['remote:192.0.2.5:7350,size=8G'])
        self.assertIn('--kv-store remote:auto,size=8G ', opt.canonical(o, opt.engine_plan(o), None)[1])

    def test_refusals(self):
        v = ValidateTests()
        for argv, want in ((['--kv-store', 'nvme:/x,size=8G'], 'the store is'), (['--kv-store', 'ssd:/x'], 'size=N'),
                           (['--kv-store', 'ssd:/x,size=8'], 'want size='), (['--kv-store', 'ssd:/x,size=8G,keep=all'], '--kv-keep'),
                           (['--kv-store', 'remote:192.0.2.5:7350,size=8G,persist=on'], 'for ssd:'),
                           (['--kv-store', 'ssd:/x,size=8G,persist=yes'], 'persist=on|off'),
                           (['--kv-store', 'remote:auto,size=8G'], 'name it'), (['--kv-store', 'ssd:/x,size=8G', '--reuse', '0'], 'needs REUSE'),
                           (['--kv-store', 'ssd:/x,size=8G', '--kv-system-after', '0'], 'from 1'),
                           (['--kv-store', 'ssd:/x,size=8G', '--kv-keep', 'most'], 'system, sessions or all'),
                           (['--kv-store', 'ssd:/x,size=8G', '--kv-write-limit', 'lots'], 'per page computed')):
            v.check(['--peers', PEERS, *argv], want)


class KvFormatTests(unittest.TestCase):
    def plan(self, *argv, tp='4'):
        o = opt.parse(['--no-config', '--peers', ','.join(PEERS.split(',')[:int(tp)]) if tp != '6' else
                       ','.join(f'10.0.0.{i}' for i in range(1, 7)), *argv], {})
        opt.validate(o)
        return o, opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')

    def test_fp8_is_the_default_and_unchanged(self):
        o, p = self.plan()
        _, p8 = self.plan('--kv-format', 'fp8')
        self.assertNotIn('--kv-format', p['args'])
        self.assertEqual((p['args'], p['knobs']['CAP']), (p8['args'], '310528'))

    def test_int4_runs_its_own_engine_with_its_own_pool(self):
        o, p = self.plan('--kv-format', 'int4')
        self.assertEqual(p['args'][-2:], ['--kv-format', 'int4'])
        self.assertEqual(p['knobs']['CAP'], str(65536 + (2390 - 1024) * 64 * 4), '2,390 pages a rank at TP=4')
        self.assertEqual(self.plan('--kv-format', 'int4', '--cap', '300000')[1]['knobs']['CAP'], '300000', 'CAP wins')
        self.assertEqual(self.plan('--kv-format', 'int4', tp='3')[1]['knobs']['CAP'], str(65536 + (2480 - 1024) * 64 * 3))
        self.assertEqual(self.plan('--kv-format', 'int4', tp='6')[1]['knobs']['CAP'], str(131072 + (3940 - 2048) * 64 * 6))

    def test_refusal(self):
        ValidateTests().check(['--peers', PEERS, '--kv-format', 'int8'], 'fp8 or int4')


if __name__ == '__main__':
    unittest.main()
