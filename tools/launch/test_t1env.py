"""envs/t1env.py: a saved TP4 mesh env becomes per-rank hybrid envs. Synthetic addresses only."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'envs'))
import options  # noqa: E402
import routes  # noqa: E402
import t1env  # noqa: E402

C = '''# serve C (synthetic)
TP=4
SHAPE=mesh
PROXY_CPU=19
PORT=8000
S2PORT=7341
CAP=408832
G6_HOME=/srv/serve/gr8
PEERS=127.0.0.11,127.0.0.12,127.0.0.13,127.0.0.14
IPS=192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4
IPS2=198.51.100.1,198.51.100.2,198.51.100.3,198.51.100.4
MODEL=/srv/models/rel
G6ENV_X="G6_DEC_BK=512 G6_DTR=1"
STEERING_FORCE=1
'''


class T1Env(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, 'C.env')
        with open(self.src, 'w') as f:
            f.write(C)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(HERE, 'envs', 't1env.py'), self.src, *args],
                              capture_output=True, text=True)

    def test_files_change_only_the_t1_keys(self):
        out = os.path.join(self.d, 'out')
        p = self.run_cli('--home', '/srv/test/gr8', '--control-port', '7350', '--out', out)
        self.assertEqual(p.returncode, 0, p.stderr)
        c = options.parse_file(self.src)
        for r in range(4):
            v = options.parse_file(os.path.join(out, f'cluster-t1.env.r{r}'))
            self.assertEqual(v['SHAPE'], 'hybrid')
            self.assertEqual(v['G6_HOME'], '/srv/test/gr8')
            self.assertEqual(v['CONTROL_PORT'], '7350')
            self.assertEqual(v['FAB_PORTS'], f'192.0.2.{r + 1}/24,198.51.100.{r + 1}/24')
            self.assertNotIn('IPS', v)
            self.assertNotIn('IPS2', v)
            self.assertNotIn('FAB_DENY', v)
            for k in set(c) - {'SHAPE', 'G6_HOME', 'IPS', 'IPS2'}:
                self.assertEqual(v[k], c[k], k)
        sha = routes.compute([f'192.0.2.{r}/24,198.51.100.{r}/24' for r in range(1, 5)]).sha
        self.assertIn(f'routes sha {sha}', p.stdout)

    def test_ips_first_keeps_c_rail_order(self):
        texts, _ = t1env.generate(self.src, '/srv/test/gr8', 7350)
        self.assertIn('FAB_PORTS=192.0.2.1/24,198.51.100.1/24', texts[0])

    def test_deny_is_written_and_changes_the_sha(self):
        a, sha0 = t1env.generate(self.src, '/srv/test/gr8', 7350)
        b, sha1 = t1env.generate(self.src, '/srv/test/gr8', 7350, deny='0-2,1-3')
        self.assertNotEqual(sha0, sha1)
        self.assertIn('FAB_DENY=0-2,1-3', b[3])

    def test_refuses_the_serve_control_ports(self):
        for port in (7340,):
            with self.assertRaisesRegex(ValueError, 'control port'):
                t1env.generate(self.src, '/srv/test/gr8', port)
        with open(self.src, 'a') as f:
            f.write('CONTROL_PORT=7360\n')
        with self.assertRaisesRegex(ValueError, 'control port'):
            t1env.generate(self.src, '/srv/test/gr8', 7360)

    def test_cli_refuses_a_port_the_launcher_refuses(self):
        p = self.run_cli('--home', '/srv/test/gr8', '--control-port', '18600', '--out', os.path.join(self.d, 'o'))
        self.assertEqual(p.returncode, 2)
        self.assertIn('fabric port block', p.stderr)

    def test_refuses_mismatched_lists_and_relative_home(self):
        with open(self.src, 'a') as f:
            f.write('IPS2=198.51.100.1,198.51.100.2\n')
        with self.assertRaisesRegex(ValueError, 'differ in length'):
            t1env.generate(self.src, '/srv/test/gr8', 7350)
        p = self.run_cli('--home', 'rel/gr8', '--control-port', '7350', '--out', os.path.join(self.d, 'o'))
        self.assertEqual(p.returncode, 2)

    def test_unroutable_deny_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'routes'):
            t1env.generate(self.src, '/srv/test/gr8', 7350, deny='0-1,0-2,0-3')


if __name__ == '__main__':
    unittest.main()
