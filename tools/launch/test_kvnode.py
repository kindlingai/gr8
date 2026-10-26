#!/usr/bin/env python3
"""./gr8 kv: options, the node's STATS answer, and the supervisor against a stub node (stubs/kvman-remote) and a fake
mentat daemon: it starts the node, announces it without GPUs, and stops both.
    python3 -m unittest discover -s tools/launch"""
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvnode as K  # noqa: E402
import mentat as M  # noqa: E402
import options as opt  # noqa: E402
from test_mentat import FakeDaemon, agent, snapshot  # noqa: E402

STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stubs', 'kvman-remote')
GLM = {'GR8_MODEL': 'glm53'}   # the model: a boot names it (no default)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def argv(home, *more, port=None):
    return ['--no-config', '--gb', '0.5', '--port', str(port or free_port()), '--bind', '127.0.0.1', '--bin', STUB,
            '--g6-home', home, *more]


class OptionTests(unittest.TestCase):
    def test_defaults_and_sources(self):
        o = K.parse(['--no-config', '--gb', '8'], {'KV_PORT': '7400', 'GROUP': 'g'})
        self.assertEqual((o.get('KV_GB'), o.get('KV_PORT'), o.get('GROUP'), o.get('MENTAT'), o.get('KV_RDMA')),
                         ('8', '7400', 'g', 'localhost', '1'))
        self.assertEqual(o.origin['KV_PORT'], 'environment')
        o = K.parse(['--no-config', 'KV_GB=2', '--no-rdma', '--port=7360'], {})
        self.assertEqual((o.get('KV_GB'), o.get('KV_RDMA'), o.get('KV_PORT')), ('2', '0', '7360'))

    def test_a_shared_config_file_gives_only_kv_keys(self):
        with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False) as f:
            f.write('TP=4\nPEERS=192.0.2.1,192.0.2.2\nGROUP=gr8-test\nKV_GB=3\n')
        o = K.parse([f'@{f.name}'], {})
        self.assertEqual((o.get('GROUP'), o.get('KV_GB')), ('gr8-test', '3'))
        os.unlink(f.name)

    def test_lazy_lock_and_its_guard(self):
        o = K.parse(['--no-config', '--gb', '8'], {})
        K.validate(o)
        self.assertNotIn('--lazy-lock', K.node_argv(o, 'kvman-remote', 'r.json'), 'off by default: the node as before')
        for a in (['--lazy-lock'], ['KV_LAZY_LOCK=1', '--guard-every-s', '0.5']):
            o = K.parse(['--no-config', '--gb', '8', *a], {})
            K.validate(o)
            v = K.node_argv(o, 'kvman-remote', 'r.json')
            self.assertEqual(v[v.index('--lazy-lock'):], ['--lazy-lock', '--guard-every-s', '0.5' if len(a) > 1 else '1'])
        for bad in (['KV_LAZY_LOCK=yes'], ['--guard-every-s', '-1'], ['--guard-every-s', 'often']):
            with self.assertRaises(opt.OptionError):
                K.validate(K.parse(['--no-config', '--gb', '8', *bad], {}))

    def test_up_needs_a_size(self):
        for a in ([], ['--gb', '0'], ['--gb', 'lots']):
            with self.assertRaises(opt.OptionError):
                K.validate(K.parse(['--no-config', *a], {}))
        with self.assertRaises(opt.OptionError):
            K.parse(['--no-config', '--tp', '4'], {})


class MentatTests(unittest.TestCase):
    def test_kv_nodes_are_live_agents_with_the_service(self):
        kv = dict(agent('10.0.0.9', 'n9', gpus=0), services={'gr8-kv': {'host': '', 'port': 7350}},
                  meta={'gr8.kv.rdma': '10.1.0.9'})
        dead = dict(agent('10.0.0.3', 'n3', gpus=0, alive=False), services={'gr8-kv': {'host': '', 'port': 7350}})
        snap = snapshot({'kv': kv, 'dead': dead, 'gpu': agent('10.0.0.2', 'n2'),
                         'kv2': dict(agent('10.0.0.10', 'n10', gpus=0), services={'gr8-kv': {'host': '', 'port': 7351}})})
        self.assertEqual([(a, p) for a, p, _ in M.kv_nodes(snap, 'gr8-glm53')], [('10.0.0.9', 7350), ('10.0.0.10', 7351)])
        self.assertEqual(M.kv_nodes(snap, 'gr8-glm53')[0][2]['gr8.kv.rdma'], '10.1.0.9')
        # a KV node never counts as a member in the election
        self.assertEqual([ip for ip, _, _ in M.members(snap, 'gr8-glm53')], ['10.0.0.2'])


class SupervisorTests(unittest.TestCase):
    def wait(self, cond, what, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return
            time.sleep(0.05)
        self.fail(f'timed out waiting for {what}')

    @mock.patch.object(K, 'log')
    def test_announces_without_gpus_and_stops_the_node(self, _):
        home = tempfile.mkdtemp()
        fake = FakeDaemon(snapshot({}))
        port = free_port()
        o = K.parse(argv(home, '--mentat', f'127.0.0.1:{fake.port}', '--group', 'gr8-test', port=port), {})
        s = K.Supervisor(o)
        s.node_out = subprocess.DEVNULL
        t = threading.Thread(target=s.run)
        t.start()
        self.wait(lambda: K.read_json(os.path.join(home, 'out/kv/state.json')).get('state') == 'serving', 'serving')
        self.wait(lambda: fake.registrations, 'a registration')
        r = fake.registrations[-1]
        self.assertEqual(r['agent_id'], 'gr8-test@gr8-kv@10.0.0.1')
        self.assertEqual(r['machine']['gpus'], [])
        self.assertEqual(r['services'], {'gr8-kv': {'host': '', 'port': port, 'path': '', 'provider': 'gr8'}})
        self.assertEqual((r['meta']['gr8.kv.rdma'], r['meta']['gr8.kv.port']), ('192.0.2.77', str(port)))
        st = K.stats('127.0.0.1', port)
        self.assertEqual(st['bytes'], 1 << 29)
        self.assertIn('--min-free-gb', st['argv'])
        node = s.node
        s.stop_ev.set()
        t.join(10)
        s.shutdown(0, '')
        self.assertIsNotNone(node.poll())
        self.assertEqual(K.read_json(os.path.join(home, 'out/kv/state.json'))['state'], 'stopped')
        fake.close()

    @mock.patch.object(K, 'log')
    def test_a_node_that_fails_fails_the_supervisor(self, logged):
        home = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {'STUB_KV_FAIL': 'RLIMIT_MEMLOCK is 8388608 bytes'}):
            code = K.supervise(K.parse(argv(home, '--mentat', 'none'), {}), node_out=subprocess.DEVNULL)
        self.assertEqual(code, 1)
        st = K.read_json(os.path.join(home, 'out/kv/state.json'))
        self.assertTrue(st['state'].startswith('failed: kvman-remote exited (1)'), st)

    def test_up_status_down_end_to_end(self):
        home = tempfile.mkdtemp()
        port = free_port()
        cli = [sys.executable, K.HERE]
        health = [*cli, 'health', '--no-config', '--g6-home', home]
        r = subprocess.run(health, capture_output=True, text=True, timeout=30)
        self.assertEqual((r.returncode, r.stdout), (1, 'KV node: not serving, no supervisor runs (last state none)\n'))
        self.assertFalse(os.path.exists(os.path.join(home, 'out/kv/kv.lock')), 'health takes no lock')
        r = subprocess.run([*cli, 'up', *argv(home, '--mentat', 'none', port=port)], capture_output=True, text=True,
                           timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('serving 0.50 GiB on port', r.stdout)
        r = subprocess.run(health, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(r.stdout.startswith(f'KV node: serving on port {port}, supervisor pid '), r.stdout)
        r2 = subprocess.run([*cli, 'up', *argv(home, '--mentat', 'none', port=port)], capture_output=True, text=True,
                            timeout=30)
        self.assertEqual(r2.returncode, 1)
        self.assertIn('already runs', r2.stderr)
        r = subprocess.run([*cli, 'status', '--no-config', '--g6-home', home], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('0.50 GiB locked, 0.50 GiB free', r.stdout)
        node = K.read_json(os.path.join(home, 'out/kv/state.json'))['node_pid']
        r = subprocess.run([*cli, 'down', '--no-config', '--g6-home', home], capture_output=True, text=True, timeout=30)
        self.assertIn('KV node stopped', r.stdout)
        with self.assertRaises(ProcessLookupError):
            for _ in range(50):
                os.kill(node, 0)
                time.sleep(0.1)
        r = subprocess.run([*cli, 'status', '--no-config', '--g6-home', home], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 3)
        self.assertIn('last state: stopped', r.stdout)
        r = subprocess.run(health, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, 'KV node: not serving, no supervisor runs (last state stopped)\n')


class HealthTests(unittest.TestCase):
    """./gr8 kv health: exit 0 only while this box's KV supervisor reports serving and it and its node still run."""
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.out = os.path.join(self.home, 'out', 'kv')
        os.makedirs(self.out)

    def process(self, *words):
        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', *words])
        self.addCleanup(p.wait)
        self.addCleanup(p.kill)
        return p.pid

    def write(self, state, pid, node=None):
        with open(os.path.join(self.out, 'state.json'), 'w') as f:
            json.dump({'state': state, 'pid': pid, 'time': time.time(), 'port': 7350, 'node_pid': node}, f)

    def health(self, *more):
        with mock.patch('sys.stdout', new_callable=io.StringIO) as so:
            rc = K.main(['health', '--no-config', '--g6-home', self.home, *more])
        return rc, so.getvalue()

    def test_states(self):
        sup = self.process(os.path.join(self.home, 'kvnode.py'), 'supervise')
        node = self.process('kvman-remote')
        self.write('starting', sup)
        self.assertEqual(self.health(), (1, f'KV node: starting, supervisor pid {sup}\n'))
        self.write('serving', sup, node)
        self.assertEqual(self.health(), (0, f'KV node: serving on port 7350, supervisor pid {sup}, kvman-remote pid '
                                            f'{node}\n'))
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        self.write('serving', sup, p.pid)                           # the node died and the supervisor has not seen it
        rc, line = self.health()
        self.assertEqual(rc, 1)
        self.assertIn(f'kvman-remote (pid {p.pid}) is gone', line)
        self.write('serving', os.getpid(), node)                    # a live pid that is not a KV supervisor
        self.assertEqual(self.health(), (1, 'KV node: not serving, no supervisor runs (last state serving)\n'))
        self.write('serving', self.process(os.path.join(self.home, 'launch.py'), 'supervise'), node)   # the model's
        self.assertEqual(self.health()[0], 1)

    def test_bad_options_exit_1_not_docker_reserved_2(self):
        rc, line = self.health('--no-such-option', 'x')
        self.assertEqual(rc, 1)
        self.assertIn('unknown option --no-such-option', line)


if __name__ == '__main__':
    unittest.main()


class KvRowsTargetTests(unittest.TestCase):
    """KV_ROWS remote:auto takes KV_STORE's node rule: an RDMA address on one of the box's rails, odd ranks on the
    second rail, so the ranks split over the node's two PCIe roots."""
    def test_ranks_split_over_the_rails(self):
        o = opt.parse(['--no-config', '--mentat', 'localhost', '--tp', '4', '--kv-rows', 'remote:auto,size=8G,local=40'], GLM)
        opt.validate(o)
        nodes = [('192.0.2.20', 7351, {'gr8.kv.rdma': '10.2.0.20,10.1.0.20'})]
        rows = [opt.kv_rows_with(o, M.kv_target(nodes, ('10.1.0.3', '10.2.0.3'), r)) for r in range(4)]
        self.assertEqual(rows, ['remote:10.1.0.20:7351,size=8G,local=40', 'remote:10.2.0.20:7351,size=8G,local=40'] * 2)
        # a named node stays as written
        o = opt.parse(['--no-config', '--mentat', 'localhost', '--tp', '4', '--kv-rows', 'remote:192.0.2.9:7350,size=8G'], GLM)
        self.assertEqual(opt.kv_rows_with(o, '10.1.0.20:7351'), 'remote:192.0.2.9:7350,size=8G')
