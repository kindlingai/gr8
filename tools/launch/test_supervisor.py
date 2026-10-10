#!/usr/bin/env python3
"""The supervisor's control port, a head that fails before its checks finish, the engine's environment and build id.
    python3 -m unittest discover -s tools/launch"""
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handshake as H  # noqa: E402
import options as opt  # noqa: E402
import supervisor as S  # noqa: E402
from test_handshake import hello  # noqa: E402

MEMBERS = ['192.0.2.1', '192.0.2.2', '192.0.2.3', '192.0.2.4']    # TEST-NET: never this box's address


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def supervisor(*argv, rank=0, members=MEMBERS):
    home = tempfile.mkdtemp()
    o = opt.parse(['--no-config', '--peers', ','.join(members), '--g6-home', home, '--control-port', str(free_port()),
                   *argv], {})
    opt.validate(o)
    s = S.Supervisor(o)
    s.rank, s.members, s.lan, s.rails = rank, members, members[rank], ('192.0.2.101', '192.0.2.201')
    os.makedirs(s.out)
    return s


class ControlPortTests(unittest.TestCase):
    @mock.patch.object(S, 'log')
    def test_requests_from_non_members_are_ignored(self, logged):
        s = supervisor(rank=1)
        for t in ('down', 'abort', 'ready', 'serving', 'hello', 'verdict', 'status', 'logs'):
            r = s.on_rpc({'t': t}, '192.0.2.99')
            self.assertEqual(r['t'], 'error', t)
            self.assertIn('not a member', r['error'])
        self.assertFalse(s.stop_ev.is_set())
        self.assertIn("ignored 'down' from 192.0.2.99", logged.call_args_list[0][0][0])
        self.assertEqual(s.on_rpc({'t': 'ping'}, '192.0.2.99')['t'], 'pong')      # the head's rail check
        self.assertEqual(s.on_rpc({'t': 'down'}, '192.0.2.4'), {'t': 'ok'})
        self.assertTrue(s.stop_ev.is_set())

    def test_a_workers_and_a_heads_own_rails_count(self):
        s = supervisor()
        s.head = H.Head(hello(0, members=MEMBERS, lan=MEMBERS[0]))
        s.head.have[2] = hello(2, rails=['192.0.2.103', '192.0.2.203'])
        self.assertTrue(s.allowed('192.0.2.203'))
        self.assertTrue(s.allowed('127.0.0.1'))       # this box: the command line on the same box
        self.assertFalse(s.allowed('192.0.2.77'))

    def test_hello_before_the_head_is_ready(self):
        s = supervisor()
        s.state = 'preparing'
        self.assertEqual(s.on_rpc({'t': 'hello'}, MEMBERS[2]),
                         {'t': 'retry', 'detail': 'the head has not finished its checks'})
        w = supervisor(rank=2)
        r = w.on_rpc({'t': 'hello'}, MEMBERS[1])
        self.assertEqual((r['t'], r['detail']), ('refused', '192.0.2.3 is rank 2, not the head'))

    def test_a_head_that_fails_before_its_checks_finish_tells_waiting_workers(self):
        s = supervisor(members=['127.0.0.1', '127.0.0.2', '127.0.0.3', '127.0.0.4'])
        s.locked = True
        s.state = 'preparing'
        s.serve(['127.0.0.1'])
        got = {}
        h = hello(3, lan='127.0.0.1')
        t = threading.Thread(target=lambda: got.update(v=H.send_hello('127.0.0.1', s.port, h, None, 20,
                                                                        lambda m: None, poll_s=0.05)))
        t.start()
        time.sleep(0.5)                       # the worker hears `retry` meanwhile
        self.assertNotIn('v', got)
        with mock.patch.dict(os.environ, {'GR8_REFUSAL_LINGER': '3'}):
            threading.Thread(target=s.shutdown, args=(1, 'model: /m: no such directory')).start()
            t.join(20)
        self.assertEqual(got['v']['field'], 'head')
        self.assertEqual(got['v']['detail'], 'the head stopped: model: /m: no such directory')


class EngineTests(unittest.TestCase):
    def test_the_engine_gets_the_plan_and_an_allowlist(self):
        s = supervisor()
        plan = {'env': ['G6_PF_TILE=64', 'G6_X=a=b'], 'args': []}
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'NVIDIA_TF32_OVERRIDE': '0', 'SECRET_TOKEN': 'x',
                                          'PYTHONPATH': '/p', 'LD_LIBRARY_PATH': '/usr/lib'}):
            env = s.engine_env(plan)
        self.assertEqual((env['G6_PF_TILE'], env['G6_X'], env['CUDA_VISIBLE_DEVICES']), ('64', 'a=b', '0'))
        self.assertEqual(env['LD_LIBRARY_PATH'], '/usr/lib')                               # no bin/lib: the box's own
        self.assertNotIn('SECRET_TOKEN', env)
        self.assertNotIn('PYTHONPATH', env)

    def test_a_bundled_cublas_is_used_at_every_tp(self):
        for tp in (3, 4):
            s = supervisor('--tp', str(tp), members=MEMBERS[:tp])
            os.makedirs(os.path.join(s.home, 'bin', 'lib'))
            with mock.patch.dict(os.environ, {'LD_LIBRARY_PATH': '/usr/lib'}):
                self.assertEqual(s.engine_env({'env': [], 'args': []})['LD_LIBRARY_PATH'], os.path.join(s.home, 'bin', 'lib'))

    def test_g6_variables_in_the_environment_are_refused_by_name(self):
        o = opt.parse(['--no-config', '--peers', ','.join(MEMBERS)], {'G6_S2_TIMEOUT': '300', 'G6_HOME': '/h',
                                                                      'G6_CLUSTER': '/c', 'G6_NR': '4'})
        with self.assertRaises(opt.OptionError) as e:
            opt.validate(o)
        self.assertIn('G6_S2_TIMEOUT set in the environment', str(e.exception))
        self.assertNotIn('G6_HOME', str(e.exception))

    def test_the_build_id_covers_bin_lib(self):
        home = tempfile.mkdtemp()
        self.assertEqual(S.engine_build(home), 'missing')
        os.makedirs(os.path.join(home, 'bin', 'lib'))
        with open(os.path.join(home, 'bin', 'g6m4'), 'w') as f:
            f.write('engine')
        a = S.engine_build(home)
        with open(os.path.join(home, 'bin', 'lib', 'libcublas.so.13'), 'w') as f:
            f.write('one')
        b = S.engine_build(home)
        with open(os.path.join(home, 'bin', 'lib', 'libcublas.so.13'), 'w') as f:
            f.write('two')
        self.assertEqual(len({a, b, S.engine_build(home)}), 3)


if __name__ == '__main__':
    unittest.main()
