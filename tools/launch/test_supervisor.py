#!/usr/bin/env python3
"""The supervisor's control port, a head that fails before its checks finish, the engine's environment and build id.
    python3 -m unittest discover -s tools/launch"""
import fcntl
import http.server
import io
import json
import os
import shutil
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
from test_mentat import FakeDaemon, flipped, ring, solved  # noqa: E402
import mentat as M  # noqa: E402

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


class MentatRingTests(unittest.TestCase):
    """--shape ring with mentat: every box claims the ring, gets the one view the head solved, and derives the same
    ring and its own --ring-devs from it, whatever the cabling."""

    def boot(self, n, cables, broken=(), extra=()):
        view, ids, roots, addr = solved(n, cables, broken=broken)
        agents, lock, holders = {}, threading.Lock(), {}

        def group(_):
            with lock:
                return {'node_id': 'n1', 'peers': {}, 'groups': {'gr8-glm53': {'agents': {
                    k: {f: x for f, x in v.items() if f != 'link'} for k, v in agents.items()}}}}

        def on_link(h, alive, link):
            with lock:
                if alive:
                    agents[h['agent_id']] = {'node_ip': h['node_ip'], 'node_id': f'n{h["node_ip"].rsplit(".", 1)[1]}',
                                             'alive': True, 'machine': h['machine'], 'meta': h['meta'], 'link': link}
                elif agents[h['agent_id']]['link'] is link:
                    agents[h['agent_id']]['alive'] = False

        def claim(h, link):
            # the head: a claim solves once, and every later holder gets that view
            with lock:
                if h['shape']['sets'][0]['layout'] == 'any':
                    return {'t': 'claim_ok', 'view': {'sets': {'ring': []}, 'links': view['links']}}
                if broken:
                    return {'t': 'err', 'code': 'unsolvable_claim', 'error': 'set "ring" wants a ring of 3 nodes'}
                holders.setdefault(h['name'], []).append(link)
                return {'t': 'claim_ok', 'name': h['name'], 'generation': 1, 'head_node_id': 'n1', 'view': view}
        daemons = [FakeDaemon(group=group, local=lambda i=i: {'node_ip': f'10.0.0.{i + 1}', 'addr_tags': {}},
                              on_link=on_link, claim=claim) for i in range(n)]
        # the devices: port p of box i at root k is f{p}r{k}
        devs = [{a: (f'f{k[1]}r{k[2]}', k[2]) for k, a in addr.items() if k[0] == i} for i in range(n)]
        sups, res = [], [None] * n
        elect = M.elect
        try:
            with mock.patch.object(S, 'log'), \
                    mock.patch.object(S.fabric, 'ring_ports', lambda: devs[int(threading.current_thread().name)]), \
                    mock.patch.object(S.mentat, 'elect', lambda *a, **k: elect(*a, **{**k, 'hold_s': 0.2,
                                                                                    'ring_settle': 1.0})):
                for i in range(n):
                    o = opt.parse(['--no-config', '--mentat', f'127.0.0.1:{daemons[i].port}', '--tp', str(n),
                                   '--shape', 'ring', '--g6-home', tempfile.mkdtemp(), '--join-timeout', '30',
                                   *extra], {})
                    opt.validate(o)
                    s = S.Supervisor(o)
                    sups.append(s)

                def run(i):
                    try:
                        sups[i].rendezvous(n)
                        res[i] = 'ok'
                    except S.Stop as e:
                        res[i] = e
                ts = [threading.Thread(target=run, args=(i,), name=str(i)) for i in range(n)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join(60)
                self.holding = [s.claim_link.sock is not None for s in sups]
                for s in sups:
                    s.release_claim()
                self.released = [s.claim_link.sock is None for s in sups]
        finally:
            for s in sups:
                if s.agent:
                    s.agent.stop()
            for d in daemons:
                d.close()
        return sups, res

    def test_three_and_four_boxes_on_any_cabling(self):
        for n, order in ((3, [0, 2, 1]), (4, [0, 2, 1, 3])):
            for cables in (ring(order), flipped(order)):
                with self.subTest(n=n, cables=cables):
                    sups, res = self.boot(n, cables)
                    self.assertEqual(res, ['ok'] * n)
                    self.assertEqual(self.holding, [True] * n)      # each box holds the claim until the handshake
                    self.assertEqual(self.released, [True] * n)
                    self.assertEqual(len({tuple(s.members) for s in sups}), 1)
                    self.assertEqual(len({s.ring for s in sups}), 1)     # the handshake compares it
                    self.assertIn(f' claim=gr8-glm53-ring{n}@n1:1', sups[0].ring)
                    self.assertEqual(sups[0].members[0], '10.0.0.1')
                    by_rank = sorted(sups, key=lambda s: s.rank)
                    for s in by_rank:
                        mine = s.ring_devs.split(',')
                        self.assertEqual(s.rails, (s.lan, s.lan))
                        # direct-next and direct-prev are root 1 of two different ports; the relay (TP=4) root 0 of
                        # the same ports, and at TP=3, which opens no relay, the direct devices again
                        self.assertEqual((mine[1][-2:], mine[3][-2:]), ('r1', 'r1'))
                        self.assertNotEqual(mine[1][:2], mine[3][:2])
                        relay = ('r0', 'r0') if n == 4 else ('r1', 'r1')
                        self.assertEqual((mine[0][-2:], mine[2][-2:]), relay)
                        self.assertEqual((mine[0][:2], mine[2][:2]), (mine[1][:2], mine[3][:2]))
                        c, _, _ = opt.canonical(s.o, opt.engine_plan(s.o), None, ring=s.ring)
                        self.assertEqual(c['ring'], by_rank[0].ring)

    def test_the_engine_gets_the_found_devices_and_an_explicit_list_wins(self):
        sups, res = self.boot(3, flipped([0, 1, 2]))
        self.assertEqual(res, ['ok'] * 3)
        s = sups[0]
        self.assertEqual(s.ring_devs, 'f1r1,f1r1,f0r1,f0r1')       # its next box is on its f1 cable; TP=3: no relay
        self.assertEqual(s.ring_args(), ['--ring', '--ring-devs', 'f1r1,f1r1,f0r1,f0r1'])
        sups, res = self.boot(4, flipped([0, 1, 2, 3]))
        self.assertEqual(res, ['ok'] * 4)
        self.assertEqual(sups[0].ring_devs, 'f1r0,f1r1,f0r0,f0r1')
        sups, res = self.boot(3, flipped([0, 1, 2]), extra=('--ring-devs', 'a,b,c,d'))
        self.assertEqual(res, ['ok'] * 3)
        self.assertIsNone(sups[0].ring_devs)
        self.assertEqual(sups[0].ring_args(), ['--ring', '--ring-devs', 'a,b,c,d'])
        # --peers: the engine's default devices, as before
        self.assertEqual(supervisor('--shape', 'ring').ring_args(), ['--ring'])
        self.assertEqual(supervisor().ring_args(), [])

    def test_the_handshake_lets_the_claim_go_whatever_its_verdict(self):
        for verdict in ({'t': 'ok'}, RuntimeError('head gone')):
            for rank in (0, 1):
                with self.subTest(verdict=verdict, rank=rank):
                    s = supervisor('--shape', 'ring', rank=rank)
                    s.claim_link = mock.Mock()
                    which = 'handshake_head' if rank == 0 else 'handshake_worker'
                    effect = {'side_effect': verdict} if isinstance(verdict, Exception) else {'return_value': verdict}
                    with mock.patch.object(s, which, **effect):
                        if isinstance(verdict, Exception):
                            with self.assertRaises(RuntimeError):
                                s.handshake({})
                        else:
                            self.assertEqual(s.handshake({}), verdict)
                    s.claim_link.close.assert_called_once_with()

    def test_a_broken_cable_stops_every_box(self):
        sups, res = self.boot(3, ring([0, 1, 2]), broken={2})
        for e in res:
            self.assertIsInstance(e, S.Stop)
            self.assertEqual(e.code, 4)
            self.assertIn('no cable works between 10.0.0.1 and 10.0.0.3', str(e))


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


STUB_KV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stubs', 'kvman-remote')
ENGINE_ARGS = os.path.join(opt.REPO, 'scripts', 'engine_args.sh')
GIB_KIB = 1 << 20


def donor_budget(kib, **env):
    r = subprocess.run(['bash', '-c', '. "$0"; flash_donor_budget "$1"', ENGINE_ARGS, str(kib)],
                       env={'PATH': os.environ['PATH'], **env}, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else 'error: ' + r.stderr.strip()


class DonorTests(unittest.TestCase):
    """D-331 M7: a Flash box lends what its serve leaves to a KV node beside it."""

    def test_the_budget_is_memavailable_less_growth_and_reserve_capped_by_the_ask(self):
        w = {'FLASH_KV_GROWTH_GIB': '8', 'KV_MIN_FREE_GB': '10'}
        self.assertEqual(donor_budget(60 * GIB_KIB, FLASH_KV_DONATE_GIB='30', **w), '30')
        self.assertEqual(donor_budget(60 * GIB_KIB, FLASH_KV_DONATE_GIB='50', **w), '42')
        self.assertEqual(donor_budget(int(40.9 * GIB_KIB), FLASH_KV_DONATE_GIB='50', **w), '22')   # whole GiB, down
        self.assertEqual(donor_budget(15 * GIB_KIB, FLASH_KV_DONATE_GIB='50', **w), '0')
        self.assertEqual(donor_budget(60 * GIB_KIB, FLASH_KV_DONATE_GIB='0', **w), '0')
        # the node's floor, 10, when KV_MIN_FREE_GB is not set
        self.assertEqual(donor_budget(51 * GIB_KIB, FLASH_KV_DONATE_GIB='40', FLASH_KV_GROWTH_GIB='8'), '33')
        self.assertTrue(donor_budget('lots', FLASH_KV_DONATE_GIB='30', **w).startswith('error'))

    def flash_plan(self, *argv):
        o = opt.parse(['--no-config', '--gr8-model', 'glm53f', '--tp', '2', *argv], {})
        return opt.engine_plan(o)['knobs']

    def test_the_knobs_are_checked_at_boot(self):
        k = self.flash_plan()
        self.assertEqual((k['FLASH_KV_DONATE_GIB'], k['FLASH_CAP_POLICY']), ('0', 'protected'))
        with self.assertRaisesRegex(opt.OptionError, 'FLASH_KV_GROWTH_GIB'):
            self.flash_plan('--flash-kv-donate-gib', '20', '--flash-kv-group', 'gr8-glm53')
        # the group is the borrowing cluster's, never this serve's by default
        with self.assertRaisesRegex(opt.OptionError, 'FLASH_KV_GROUP'):
            self.flash_plan('--flash-kv-donate-gib', '20', '--flash-kv-growth-gib', '6')
        with self.assertRaisesRegex(opt.OptionError, 'borrow-idle needs FLASH_KV_DONATE_GIB'):
            self.flash_plan('--flash-cap-policy', 'borrow-idle')
        with self.assertRaisesRegex(opt.OptionError, 'want protected'):
            self.flash_plan('--flash-cap-policy', 'lend')
        k = self.flash_plan('--flash-kv-donate-gib', '20', '--flash-kv-growth-gib', '6',
                            '--flash-kv-group', 'gr8-glm53')
        self.assertEqual((k['FLASH_KV_DONATE_GIB'], k['FLASH_KV_GROWTH_GIB'], k['FLASH_KV_GROUP']),
                         ('20', '6', 'gr8-glm53'))

    def test_borrow_idle_puts_the_pool_in_chunks_and_lends_after_the_idle_time(self):
        give = ('--flash-kv-donate-gib', '20', '--flash-kv-growth-gib', '6', '--flash-kv-group', 'gr8-glm53')
        o = opt.parse(['--no-config', '--gr8-model', 'glm53f', '--tp', '2', *give], {})
        self.assertNotIn('G6_FLASH_KV_CHUNKS=1', opt.engine_plan(o)['env'])   # protected: the pool as before
        for idle, ms in ((None, '60000'), ('5', '5000')):
            argv = ['--flash-cap-policy', 'borrow-idle'] + (['--flash-kv-idle-s', idle] if idle else [])
            o = opt.parse(['--no-config', '--gr8-model', 'glm53f', '--tp', '2', *give, *argv], {})
            env = opt.engine_plan(o)['env']
            self.assertIn('G6_FLASH_KV_CHUNKS=1', env)
            self.assertIn('G6_FLASH_KV_LEND_MS=' + ms, env)
        with self.assertRaisesRegex(opt.OptionError, 'FLASH_KV_IDLE_S=0'):
            self.flash_plan(*give, '--flash-cap-policy', 'borrow-idle', '--flash-kv-idle-s', '0')
        # the engine gets this box's ledger
        s = supervisor('--gr8-model', 'glm53f', '--tp', '2', members=MEMBERS[:2])
        env = s.engine_env({'env': [], 'knobs': {'FLASH_CAP_POLICY': 'borrow-idle'}})
        self.assertEqual(env['G6_FLASH_KV_LEDGER'], os.path.join(s.home, 'out', 'kv', 'ledger'))
        self.assertNotIn('G6_FLASH_KV_LEDGER', s.engine_env({'env': [], 'knobs': {'FLASH_CAP_POLICY': 'protected'}}))

    def donor(self, knobs, model='glm53f', gpu=None):
        flash = model == 'glm53f'
        s = supervisor('--gr8-model', model, '--tp', '2' if flash else '4', members=MEMBERS[:2] if flash else MEMBERS)
        cfg = os.path.join(s.home, 'empty.env')
        open(cfg, 'w').close()
        # no MENTAT here: a PEERS serve's donor must not need a mentat daemon
        env = {'G6_CLUSTER': cfg, 'KV_BIN': STUB_KV, 'KV_PORT': str(free_port()),
               'KV_BIND': '127.0.0.1', 'KV_MIN_FREE_GB': '0'}
        s.kv_port = int(env['KV_PORT'])
        if gpu is not None:   # what the engine wrote at boot under borrow-idle
            os.makedirs(os.path.join(s.home, 'out', 'kv'), exist_ok=True)
            with open(os.path.join(s.home, 'out', 'kv', 'ledger.gpu'), 'w') as f:
                f.write(f'{gpu}\n')
        with mock.patch.dict(os.environ, env), mock.patch.object(S, 'log') as logged:
            os.environ.pop('MENTAT', None)
            s.donate(knobs)
        return s, ' '.join(str(c.args[0]) for c in logged.call_args_list)

    @unittest.skipUnless(S.ledger_bin(opt.REPO), 'kvman-ledger is not built (cargo build --release in engine/kvman-rs)')
    def test_a_flash_box_starts_a_lazy_node_with_a_ledger_and_stops_it(self):
        s, said = self.donor({'FLASH_KV_DONATE_GIB': '1', 'FLASH_KV_GROWTH_GIB': '0', 'FLASH_KV_GROUP': 'gr8-full'})
        try:
            self.assertTrue(s.donor, said)
            st = S.kvnode.read_json(os.path.join(s.home, 'out', 'kv', 'state.json'))
            self.assertEqual(st.get('state'), 'serving', st)
            ledger = os.path.join(s.home, 'out', 'kv', 'ledger')
            argv = S.kvnode.stats('127.0.0.1', s.kv_port)['argv']
            self.assertEqual(argv[argv.index('--gb') + 1], '1')
            self.assertEqual(argv[argv.index('--ledger') + 1], ledger)
            self.assertIn('--lazy-lock', argv)
            # the node joins the borrowing cluster's group, whatever this serve's config says
            with open(f'/proc/{st["pid"]}/cmdline', 'rb') as f:
                sup = f.read().decode().split('\0')
            self.assertEqual(sup[sup.index('--group') + 1], 'gr8-full')
            # this serve finds its boxes by PEERS: the node announces nowhere
            self.assertEqual(sup[sup.index('--mentat') + 1], 'none')
            book = subprocess.run([S.ledger_bin(opt.REPO), ledger, 'show'], capture_output=True, text=True).stdout
            self.assertIn(f'"total":{1 << 30},"gpu":0,"node":{1 << 30}', book)
        finally:
            s.locked = True
            s.shutdown(0, '')
        self.assertFalse(s.donor)
        self.assertFalse(S.kvnode.running(os.path.join(s.home, 'out', 'kv')))

    @unittest.skipUnless(S.ledger_bin(opt.REPO), 'kvman-ledger is not built (cargo build --release in engine/kvman-rs)')
    def test_borrow_idle_books_the_gpu_chunks_in_the_ledger(self):
        g = 5 * (1 << 30) + 12345
        s, said = self.donor({'FLASH_KV_DONATE_GIB': '1', 'FLASH_KV_GROWTH_GIB': '0', 'FLASH_KV_GROUP': 'gr8-full',
                              'FLASH_CAP_POLICY': 'borrow-idle'}, gpu=g)
        try:
            self.assertTrue(s.donor, said)
            ledger = os.path.join(s.home, 'out', 'kv', 'ledger')
            book = subprocess.run([S.ledger_bin(opt.REPO), ledger, 'show'], capture_output=True, text=True).stdout
            self.assertIn(f'"total":{(1 << 30) + g},"gpu":{g},"node":{1 << 30}', book)
            argv = S.kvnode.stats('127.0.0.1', s.kv_port)['argv']
            self.assertEqual(argv[argv.index('--gb') + 1], '6')   # the node may grow into the lent chunks
            self.assertIn(f'{g} bytes of GPU KV chunks', said)
        finally:
            s.locked = True
            s.shutdown(0, '')

    def test_no_budget_or_another_model_starts_nothing(self):
        s, said = self.donor({'FLASH_KV_DONATE_GIB': '1', 'FLASH_KV_GROWTH_GIB': '100000'})
        self.assertFalse(s.donor)
        self.assertIn('KV donation: none', said)
        s, said = self.donor({'FLASH_KV_DONATE_GIB': '1', 'FLASH_KV_GROWTH_GIB': '0'}, model='glm53')
        self.assertFalse(s.donor)
        self.assertEqual(said, '')


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


class HealthTests(unittest.TestCase):
    """./gr8 <model> health: this box alone, exit 0 only while its own supervisor reports serving (and on the head,
    while the API's /health answers 200), so compose and a Kubernetes exec probe can run it on every rank."""
    def setUp(self):
        import launch
        self.L = launch
        self.home = tempfile.mkdtemp()
        self.out = os.path.join(self.home, 'out')
        os.makedirs(self.out)
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)

    def fake_supervisor(self):
        """A process whose command line is a supervisor's (`launch.py supervise`), as health() checks in /proc."""
        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                              os.path.join(self.home, 'launch.py'), 'supervise'])
        self.addCleanup(p.wait)
        self.addCleanup(p.kill)
        return p.pid

    def write(self, state, rank, pid=None, supervisor=True):
        """gr8.state as set_state writes it, and supervisor.pid as run() does."""
        if pid is None:
            pid = self.sup = getattr(self, 'sup', None) or self.fake_supervisor()
        with open(os.path.join(self.out, 'gr8.state'), 'w') as f:
            json.dump({'state': state, 'rank': rank, 'pid': pid, 'time': time.time()}, f)
        if supervisor:
            with open(os.path.join(self.out, 'supervisor.pid'), 'w') as f:
                f.write(f'{pid}\n')

    def api(self, code):
        """A front end whose /health answers `code`; returns its port."""
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if code is None:                       # an answer that is not HTTP
                    self.wfile.write(b'garbage\r\n\r\n')
                    return
                self.send_response(code if self.path == '/health' else 404)
                self.end_headers()

            def log_message(self, *_):
                pass
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def run_health(self, *argv):
        """cmd_health through launch.main, as ./gr8 runs it: (exit status, the one line it printed)."""
        with mock.patch('sys.stdout', new_callable=io.StringIO) as so:
            rc = self.L.main(['health', '--no-config', '--g6-home', self.home, *argv])
        return rc, so.getvalue()

    def test_every_state_before_serving_is_not_ready(self):
        for state in ('starting', 'preparing', 'handshake', 'joining', 'loading', 'stopping'):
            self.write(state, 1 if state != 'starting' else None)
            rc, line = self.run_health()
            self.assertEqual(rc, 1, (state, line))
            self.assertIn(state, line)
            self.assertEqual(line.count('\n'), 1, line)

    def test_a_worker_that_serves_is_ready(self):
        self.write('serving', 2)
        rc, line = self.run_health('--port', str(free_port()))     # a worker does not ask any API
        self.assertEqual((rc, line), (0, f'rank 2: serving, supervisor pid {self.sup}\n'))

    def test_the_head_serves_only_while_its_api_answers(self):
        self.write('serving', 0)
        rc, line = self.run_health('--port', str(self.api(200)))
        self.assertEqual(rc, 0, line)
        self.assertIn('API /health 200', line)
        rc, line = self.run_health('--port', str(self.api(503)))
        self.assertEqual(rc, 1, line)
        self.assertIn('answers 503', line)
        port = free_port()
        rc, line = self.run_health('--port', str(port))
        self.assertEqual(rc, 1, line)
        self.assertIn(f'on port {port} answers nothing', line)

    def test_the_head_asks_the_port_the_boot_agreed(self):
        self.write('serving', 0)
        with open(os.path.join(self.out, 'handshake.json'), 'w') as f:
            json.dump({'t': 'ok', 'port': str(self.api(200)), 'rank': 0}, f)
        self.assertEqual(self.run_health('--port', str(free_port()))[0], 0)

    def test_a_failed_or_stopped_boot_is_not_ready(self):
        for state in ('failed: engine exited (-9)', 'stopped'):
            self.write(state, 1)
            rc, line = self.run_health()
            self.assertEqual(rc, 1, line)
            self.assertIn(state, line)

    def test_serving_written_by_a_supervisor_that_is_gone_is_not_ready(self):
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        self.write('serving', 1, pid=p.pid)                           # kill -9: no shutdown() rewrote the state
        rc, line = self.run_health()
        self.assertEqual(rc, 1, line)
        self.assertIn('no supervisor runs (last state serving)', line)
        other = self.fake_supervisor()
        self.write('serving', 1, pid=other, supervisor=False)
        with open(os.path.join(self.out, 'supervisor.pid'), 'w') as f:
            f.write(f'{self.fake_supervisor()}\n')                 # another supervisor than the one that wrote it
        self.assertEqual(self.run_health()[0], 1)

    def test_a_live_pid_that_is_not_a_supervisor_is_not_ready(self):
        self.write('serving', 1, pid=os.getpid())                     # pid reuse: the number now runs something else
        rc, line = self.run_health()
        self.assertEqual(rc, 1, line)
        self.assertIn('no supervisor runs', line)

    def test_without_proc_ps_tells_a_supervisor_from_another_process(self):
        """macOS has no /proc: the command line comes from ps."""
        no_proc = os.path.join(self.home, 'no-proc')
        self.assertTrue(self.L.is_supervisor(self.fake_supervisor(), no_proc))
        self.assertFalse(self.L.is_supervisor(os.getpid(), no_proc))

    def test_a_clock_step_after_the_boot_does_not_make_it_unready(self):
        self.write('serving', 1)
        p = os.path.join(self.out, 'supervisor.pid')
        os.utime(p, (time.time() - 3600,) * 2)    # NTP stepped back an hour: the file looks older than the process
        self.assertEqual(self.run_health()[0], 0)

    def test_an_http_proxy_in_the_environment_is_not_used(self):
        """As docker runs it: a new process, with an HTTP_PROXY (docker's daemon proxies) that answers nothing."""
        self.write('serving', 0)
        env = {**os.environ, 'http_proxy': 'http://127.0.0.1:9', 'HTTP_PROXY': 'http://127.0.0.1:9', 'no_proxy': '',
               'NO_PROXY': ''}
        r = subprocess.run([sys.executable, self.L.HERE, 'health', '--no-config', '--g6-home', self.home, '--port',
                            str(self.api(200))], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_an_answer_that_is_not_http_or_a_bad_port_is_one_line_not_a_traceback(self):
        self.write('serving', 0)
        rc, line = self.run_health('--port', str(self.api(None)))
        self.assertEqual((rc, line.count('\n')), (1, 1), line)
        self.assertIn('failed: ', line)
        rc, line = self.run_health('--port', 'eighty')
        self.assertEqual((rc, line.count('\n')), (1, 1), line)

    def test_no_state_is_not_ready(self):
        rc, line = self.run_health()
        self.assertEqual((rc, line), (1, 'this box: not serving, no supervisor runs (last state none)\n'))

    def test_bad_options_exit_1_not_docker_reserved_2(self):
        rc, line = self.run_health('--no-such-option', 'x')
        self.assertEqual(rc, 1)
        self.assertIn('unknown option --no-such-option', line)

    def test_it_takes_no_supervisor_lock(self):
        self.write('serving', 1)
        with open(os.path.join(self.out, 'supervisor.lock'), 'w') as lk:   # the supervisor holds it while it runs
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(fcntl, 'flock', side_effect=AssertionError('took the lock')):
                self.assertEqual(self.run_health()[0], 0)
        os.unlink(os.path.join(self.out, 'supervisor.lock'))
        self.assertEqual(self.run_health()[0], 0)
        self.assertFalse(os.path.exists(os.path.join(self.out, 'supervisor.lock')), 'health created the lock file')

if __name__ == '__main__':
    unittest.main()
