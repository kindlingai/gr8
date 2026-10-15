#!/usr/bin/env python3
"""The supervisor's control port, a head that fails before its checks finish, the engine's environment and build id.
    python3 -m unittest discover -s tools/launch"""
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


class RingTests(unittest.TestCase):
    @mock.patch.object(S, 'log')
    def test_a_ring_uses_the_lan_for_its_rails(self, _):
        s = supervisor('--shape', 'ring', '--node-ip', MEMBERS[2])
        with mock.patch.object(S.fabric, 'rails', side_effect=AssertionError('no fabric discovery on a ring')):
            s.rendezvous(4)
        self.assertEqual((s.rank, s.rails), (2, (MEMBERS[2], MEMBERS[2])))


class ControlPortTests(unittest.TestCase):
    @mock.patch.object(S, 'log')
    def test_requests_from_non_members_are_ignored(self, logged):
        s = supervisor(rank=1)
        for t in ('down', 'abort', 'serving', 'hello', 'verdict', 'status', 'logs'):
            r = s.on_rpc({'t': t}, '192.0.2.99')
            self.assertEqual(r['t'], 'error', t)
            self.assertIn('not a member', r['error'])
        self.assertFalse(s.stop_ev.is_set())
        self.assertIn("ignored 'down' from 192.0.2.99", logged.call_args_list[0][0][0])
        self.assertEqual(s.on_rpc({'t': 'ping'}, '192.0.2.99')['t'], 'pong')      # the head's rail check
        self.assertEqual(s.on_rpc({'t': 'down'}, '192.0.2.4'), {'t': 'ok'})
        self.assertTrue(s.stop_ev.is_set())

    def test_a_stopping_head_asks_a_new_hello_to_wait(self):
        s = supervisor()
        s.head = H.Head(hello(0, members=MEMBERS, lan=MEMBERS[0]))
        s.head.decide({'t': 'ok', 'epoch': 7})
        s.head_ready.set()
        h = hello(2, members=MEMBERS, lan=MEMBERS[2])
        self.assertEqual(s.on_rpc(h, MEMBERS[2])['field'], 'booted')      # serving: refused
        s.request_stop('rank 1 failed', all_boxes=True)
        self.assertEqual(s.on_rpc(h, MEMBERS[2])['t'], 'retry')           # stopping: wait for the next boot

    def test_a_down_for_another_boot_is_ignored(self):
        s = supervisor(rank=1)
        s.hs = {'epoch': 8}
        self.assertEqual(s.on_rpc({'t': 'down', 'epoch': 7, 'failed': True}, MEMBERS[0])['detail'], 'not this boot')
        self.assertFalse(s.stop_ev.is_set())
        s.on_rpc({'t': 'down', 'epoch': 8, 'failed': True}, MEMBERS[0])
        self.assertTrue(s.stop_ev.is_set() and s.stop_failed)

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


class FrontTests(unittest.TestCase):
    def front_argv(self, *argv):
        s = supervisor(*argv)
        s.ms = mock.Mock(tokenizer='/t')
        with mock.patch.object(s, 'spawn') as spawn, mock.patch.object(S.os, 'access', return_value=True):
            s.start_front({'knobs': {'SHARE_LIMIT': '65536'}})
        return spawn.call_args[0][1]

    def test_the_front_end_gets_the_default_max_tokens_or_else_the_share_limit(self):
        a = self.front_argv()
        self.assertNotIn('--default-max-tokens', a)
        self.assertEqual(a[a.index('--share-tokens') + 1], '65472')     # a page short of the limit, for the guard rows
        a = self.front_argv('--default-max-tokens', '16384')
        self.assertEqual(a[a.index('--default-max-tokens') + 1], '16384')
        self.assertNotIn('--share-tokens', a)

    def test_the_thinking_defaults_reach_the_front_end_only_when_set(self):
        a = self.front_argv()
        self.assertNotIn('--default-reasoning-effort', a)
        self.assertNotIn('--default-thinking-budget', a)
        a = self.front_argv('--default-reasoning-effort', 'low', '--default-thinking-budget', '2048')
        self.assertEqual(a[a.index('--default-reasoning-effort') + 1], 'low')
        self.assertEqual(a[a.index('--default-thinking-budget') + 1], '2048')

    def test_the_front_end_finds_the_supervisor_for_deep_health(self):
        a = self.front_argv()
        host, port = a[a.index('--supervisor') + 1].rsplit(':', 1)
        self.assertEqual(host, MEMBERS[0])                      # the head's LAN address, where its supervisor listens
        self.assertTrue(port.isdigit())


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

    def test_the_kv_format_picks_the_engine(self):
        home = tempfile.mkdtemp()
        os.makedirs(os.path.join(home, 'bin'))
        with open(os.path.join(home, 'bin', 'g6m4'), 'w') as f:
            f.write('fp8')
        o8 = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'], {})
        o4 = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4', '--kv-format', 'int4'], {})
        self.assertEqual((S.engine_bin(o8), S.engine_bin(o4)), ('g6m4', 'g6m4-int4'))
        self.assertEqual(S.engine_build(home, S.engine_bin(o4)), 'missing', 'an fp8-only bin/ cannot run int4')
        with open(os.path.join(home, 'bin', 'g6m4-int4'), 'w') as f:
            f.write('int4')
        self.assertNotEqual(S.engine_build(home, 'g6m4-int4'), S.engine_build(home, 'g6m4'))


class PidFileTests(unittest.TestCase):
    """out/g6m4.pid, shim.pid and front.pid outlive their processes; a number taken later is not ours."""
    def pid_file(self, pid, age=0):
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'g6m4.pid')
        with open(p, 'w') as f:
            f.write(f'{pid}\n')
        t = time.time() - age
        os.utime(p, (t, t))
        return p

    def fake_proc(self, pid, started):
        """A /proc where pid started at `started` (seconds since the epoch): btime 1000, 100 ticks a second."""
        proc = tempfile.mkdtemp()
        os.makedirs(os.path.join(proc, str(pid)))
        with open(os.path.join(proc, 'stat'), 'w') as f:
            f.write('cpu  1 2 3\nbtime 1000\nprocesses 5\n')
        ticks = int((started - 1000) * os.sysconf('SC_CLK_TCK'))
        with open(os.path.join(proc, str(pid), 'stat'), 'w') as f:    # a comm with spaces and a ')' in it
            f.write(f'{pid} (a b) c) S ' + ' '.join(['0'] * 18) + f' {ticks} 0 0\n')
        return proc

    def test_started_at_reads_field_22(self):
        proc = self.fake_proc(4242, 5000.5)
        self.assertAlmostEqual(S.started_at(4242, proc), 5000.5, places=1)
        self.assertIsNone(S.started_at(4243, proc))
        self.assertIsNone(S.started_at(4242, os.path.join(proc, 'none')))

    def test_a_process_that_started_after_its_pid_file_is_not_ours(self):
        me = os.getpid()                                      # alive, so only the start time decides
        p = self.pid_file(me, age=3600)
        self.assertIsNone(S.recorded(p, self.fake_proc(me, time.time() - 60)))     # started an hour after the file
        self.assertEqual(S.recorded(p, self.fake_proc(me, time.time() - 7200)), me)  # started before it
        self.assertEqual(S.recorded(p, os.path.join(tempfile.mkdtemp(), 'none')), me)  # no /proc: as before

    def test_a_dead_pid_is_not_recorded(self):
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        self.assertIsNone(S.recorded(self.pid_file(p.pid)))

    @unittest.skipUnless(os.path.isfile('/proc/self/stat'), 'needs /proc')
    def test_down_leaves_a_process_that_took_a_stale_number_alone(self):
        home = tempfile.mkdtemp()
        os.makedirs(os.path.join(home, 'out'))
        other = subprocess.Popen(['sleep', '60'])               # not ours: it started after the files below
        self.addCleanup(other.wait)
        self.addCleanup(other.kill)
        for name in ('g6m4', 'shim', 'front'):
            p = os.path.join(home, 'out', f'{name}.pid')
            with open(p, 'w') as f:
                f.write(f'{other.pid}\n')
            os.utime(p, (time.time() - 3600,) * 2)          # written by a boot before a reboot
        self.assertEqual(S.recorded(os.path.join(home, 'out', 'g6m4.pid')), None)
        self.assertFalse(S.stop_local(home, log=lambda s: None))
        time.sleep(0.2)
        self.assertIsNone(other.poll(), 'stop_local signalled a process that was not ours')

    @unittest.skipUnless(os.path.isfile('/proc/self/stat'), 'needs /proc')
    def test_down_still_stops_our_own_processes(self):
        home = tempfile.mkdtemp()
        os.makedirs(os.path.join(home, 'out'))
        front = subprocess.Popen(['sleep', '60'])
        self.addCleanup(front.kill)
        time.sleep(0.05)
        with open(os.path.join(home, 'out', 'front.pid'), 'w') as f:    # written after it started, as spawn() does
            f.write(f'{front.pid}\n')
        S.stop_local(home, log=lambda s: None)
        self.assertEqual(front.wait(timeout=5), -15)

    def test_another_accounts_process_is_reported_not_raised(self):
        said = []
        with mock.patch.object(S.os, 'kill', side_effect=PermissionError):
            S.signal_pid(4242, 15, 'front', said.append)
        self.assertEqual(said, ['front (pid 4242) belongs to another account: not stopped'])


class EngineLogTests(unittest.TestCase):
    """A restart keeps the earlier boots' engine logs, and an exit names the line that says why."""
    def write(self, d, name, text):
        with open(os.path.join(d, name), 'w') as f:
            f.write(text)

    def test_each_boot_shifts_the_last_three(self):
        out = tempfile.mkdtemp()
        names = ['g6m4.r0.txt', 'g6m4.r0.err']
        S.rotate_logs(out, names)
        self.assertFalse(os.path.exists(os.path.join(out, 'prev1')))    # nothing to keep on a first boot
        for boot in range(5):
            self.write(out, 'g6m4.r0.err', f'boot {boot}\n')
            S.rotate_logs(out, names)
        kept = {}
        for d in sorted(os.listdir(out)):
            with open(os.path.join(out, d, 'g6m4.r0.err')) as f:
                kept[d] = f.read().strip()
        self.assertEqual(kept, {'prev1': 'boot 4', 'prev2': 'boot 3', 'prev3': 'boot 2'})

    def test_the_reason_is_found_behind_later_lines(self):
        d = tempfile.mkdtemp()
        self.write(d, 'e', '# rank 1: exlr8 release 78.49 GB -> device image\n'
                   'g6: G6_PF_OWN needs the reduce-scatter form (G6_PF_RS=1)\n' + '# loader thread done\n' * 30)
        reason = S.exit_reason([os.path.join(d, 'e')])
        self.assertEqual(reason, 'g6: G6_PF_OWN needs the reduce-scatter form (G6_PF_RS=1)')
        self.assertNotIn(reason, S.tail(os.path.join(d, 'e'), 20))

    def test_stdout_is_read_when_stderr_has_no_reason(self):
        d = tempfile.mkdtemp()
        self.write(d, 'e', '# loading\n')
        self.write(d, 't', "thread 'main' panicked at src/lib.rs:9:5:\nshort read\n")
        self.assertEqual(S.exit_reason([os.path.join(d, 'e'), os.path.join(d, 't'), os.path.join(d, 'none')]),
                         "thread 'main' panicked at src/lib.rs:9:5:")
        self.assertEqual(S.exit_reason([os.path.join(d, 'e')]), '')

    def test_a_failed_weight_check_stops_with_exit_3(self):
        s = supervisor()
        s.engine = mock.Mock(poll=mock.Mock(return_value=3), returncode=3)
        with open(os.path.join(s.out, 'g6m4.r0.err'), 'w') as f:
            f.write('g6: weight check: rank 0: 1 of 2400 chunks of the exlr8 release image differ from the bytes read\n')
        with self.assertRaises(S.Stop) as e:
            s.watch()
        self.assertEqual(e.exception.code, 3)
        with open(os.path.join(s.out, 'g6m4.r0.err'), 'w') as f:
            f.write('g6: G6_PF_OWN needs the reduce-scatter form (G6_PF_RS=1)\n')
        with self.assertRaises(S.Stop) as e:
            s.watch()
        self.assertEqual(e.exception.code, 1)

    def test_a_signal_is_named(self):
        self.assertEqual(S.exit_status(2), '2')
        self.assertEqual(S.exit_status(-6), '-6, SIGABRT')
        self.assertEqual(S.exit_status(-9), '-9, SIGKILL: the OOM killer? see dmesg')


class SelfCheckTests(unittest.TestCase):
    """/health?deep: each rank asks its engine for check.<n>.json; the head gathers every rank."""
    def serving(self, rank=0):
        s = supervisor(rank=rank)
        os.makedirs(os.path.join(s.home, 's2'))
        s.state, s.engine = 'serving', mock.Mock(poll=mock.Mock(return_value=None))
        return s

    def answer(self, s, n, body):
        """The engine's side: wait for DIR/check.<n>, then answer it."""
        def run():
            req = os.path.join(s.home, 's2', f'check.{n}')
            while not os.path.exists(req):
                time.sleep(0.01)
            with open(req + '.json', 'w') as f:
                f.write(body)
        threading.Thread(target=run, daemon=True).start()

    def test_a_rank_passes_its_engines_answer_on_in_order(self):
        s = self.serving()
        self.answer(s, 0, '{"rank": 0, "ok": true, "images": []}')
        r = s.local_check(timeout=5)
        self.assertTrue(r['ok'])
        self.assertEqual(r['weights']['images'], [])
        self.answer(s, 1, '{"rank": 0, "ok": false, "images": [{"changed": 3}]}')
        r = s.local_check(timeout=5)
        self.assertFalse(r['ok'])
        self.assertEqual(r['weights']['images'][0]['changed'], 3)

    def test_an_engine_that_is_not_serving_or_silent_fails(self):
        s = supervisor()
        self.assertIn('not serving', s.local_check(timeout=1)['error'])
        s = self.serving()
        self.assertIn('did not answer check.0', s.local_check(timeout=0.3)['error'])

    def test_the_head_gathers_every_rank(self):
        s = self.serving()
        self.answer(s, 0, '{"rank": 0, "ok": true}')
        answers = {1: {'t': 'check', 'rank': 1, 'ok': True}, 2: None, 3: {'t': 'check', 'rank': 3, 'ok': True}}
        with mock.patch.object(s, 'tell', side_effect=lambda r, m, timeout: answers[r]):
            a = s.check_all()
        self.assertFalse(a['ok'])                               # rank 2 did not answer
        self.assertEqual([x.get('ok') for x in a['ranks']], [True, True, False, True])
        self.assertIn('error', supervisor(rank=1).check_all())  # only the head gathers


if __name__ == '__main__':
    unittest.main()
