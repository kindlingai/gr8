#!/usr/bin/env python3
"""The mentat side, against a fake daemon that speaks the framing of mentat's PROTOCOL.md: membership, the election,
the rails, and the agent link with meta updates. No mentatd needed.
    python3 -m unittest discover -s tools/launch"""
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mentat as M  # noqa: E402

GROUP = 'gr8-glm53'


def agent(ip, node, alive=True, gpus=1, offers='any'):
    meta = {} if offers is None else {'gr8.offers': offers}
    return {'node_ip': ip, 'node_id': node, 'alive': alive, 'machine': {'gpus': [{'index': 0}] * gpus}, 'meta': meta}


def snapshot(agents, self_node='n1', self_ip='10.0.0.1'):
    peers = {f'n{i}': {'node_ip': f'10.0.0.{i}', 'addr_tags': {f'10.0.0.{i}': ['lan'], f'10.1.0.{i}': ['connectx', 'rdma'],
                                                               f'10.2.0.{i}': ['connectx', 'rdma']}}
             for i in range(2, 6)}
    return {'proto': '0.99', 'node_id': self_node, 'node_ip': self_ip, 'addrs': [self_ip, '10.1.0.1', '10.2.0.1'],
            'addr_tags': {self_ip: ['lan'], '10.2.0.1': ['rdma'], '10.1.0.1': ['rdma'], 'fe80::1': ['rdma']},
            'peers': peers, 'groups': {GROUP: {'agents': agents}}}


class FakeDaemon:
    """Control port: hello/status (client link) and agent_register/ping (agent link). HTTP on port + 1: /status.
    By default every answer is self.snap. A test of several daemons passes group(group) for the status answer,
    local() for /status, and gets on_link(header, alive, link) for each registration and its link's end; like mentat,
    a link's end matters only while that link holds the registration."""

    def __init__(self, snap=None, group=None, local=None, on_link=None, host='127.0.0.1'):
        self.snap, self.registrations, self.links = snap, [], 0
        fake = self
        group = group or (lambda g: fake.snap)
        local = local or (lambda: fake.snap)

        class Ctl(socketserver.BaseRequestHandler):
            def handle(self):
                s, reg = self.request, None
                try:
                    while True:
                        h = M.read_frame(s)
                        t = h.get('t')
                        if t == 'hello':
                            s.sendall(M.frame({'t': 'hello_ok', 'req': h['req'], 'proto': '0.99'}))
                        elif t == 'status':
                            s.sendall(M.frame({'t': 'status_ok', 'req': h['req'], 'snapshot': group(h.get('group'))}))
                        elif t == 'agent_register':
                            fake.registrations.append(h)
                            fake.links += 1
                            reg = h
                            if on_link:
                                on_link(h, True, self)
                            s.sendall(M.frame({'t': 'agent_register_ok', 'req': 1, 'proto': '1.0', 'node_id': 'n1'}))
                        elif t == 'ping':
                            s.sendall(M.frame({'t': 'pong', 'req': 0}))
                except (ConnectionError, OSError):
                    pass
                finally:
                    if reg and on_link:
                        on_link(reg, False, self)

        class Http(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(local()).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass
        for _ in range(20):
            with socket.socket() as s:
                s.bind((host, 0))
                port = s.getsockname()[1]
            try:
                self.ctl = socketserver.ThreadingTCPServer((host, port), Ctl)
                self.http = http.server.ThreadingHTTPServer((host, port + 1), Http)
                break
            except OSError:
                continue
        self.ctl.daemon_threads = True
        self.port = port
        for srv in (self.ctl, self.http):
            threading.Thread(target=srv.serve_forever, daemon=True).start()

    def close(self):
        for srv in (self.ctl, self.http):
            srv.shutdown()
            srv.server_close()


class MembershipTests(unittest.TestCase):
    def test_members_skip_dead_gpuless_and_duplicates(self):
        snap = snapshot({'a': agent('10.0.0.3', 'n3'), 'b': agent('10.0.0.1', 'n1'), 'c': agent('10.0.0.2', 'n2', alive=False),
                         'd': agent('10.0.0.4', 'n4', gpus=0), 'e': agent('10.0.0.3', 'n3'), 'f': agent('10.0.0.10', 'n10')})
        self.assertEqual(M.members(snap, GROUP), [('10.0.0.1', 'n1', 'any'), ('10.0.0.3', 'n3', 'any'),
                                                  ('10.0.0.10', 'n10', 'any')])

    def test_assignment(self):
        ips = ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4']
        any4 = dict.fromkeys(ips)
        self.assertEqual(M.assign_ranks(ips, any4, 4), ips)                          # address order
        self.assertEqual(M.assign_ranks(ips, any4, 4, head='10.0.0.3'),
                         ['10.0.0.3', '10.0.0.1', '10.0.0.2', '10.0.0.4'])
        # the boxes hold one rank each, out of address order, and one box can serve any rank
        held = {'10.0.0.1': [2], '10.0.0.2': None, '10.0.0.3': [0], '10.0.0.4': [3]}
        self.assertEqual(M.assign_ranks(ips, held, 4), ['10.0.0.3', '10.0.0.2', '10.0.0.1', '10.0.0.4'])
        # fewest choices first: 10.0.0.2 can take 1 or 2, 10.0.0.1 only 2
        self.assertEqual(M.assign_ranks(ips, {'10.0.0.1': [2], '10.0.0.2': [1, 2], '10.0.0.3': None, '10.0.0.4': None}, 4),
                         ['10.0.0.3', '10.0.0.2', '10.0.0.1', '10.0.0.4'])
        for offers, head, want in (({**any4, '10.0.0.1': [1], '10.0.0.2': [1]}, None, 'no rank left for 10.0.0.2'),
                                   ({**any4, '10.0.0.4': [2]}, '10.0.0.4', '--head 10.0.0.4 must be rank 0'),
                                   ({**any4, '10.0.0.2': []}, None, 'no rank left for 10.0.0.2')):
            with self.assertRaises(M.NoAssignment) as e:
                M.assign_ranks(ips, offers, 4, head)
            self.assertIn(want, str(e.exception))
            self.assertIn('The model set on each box offers: 10.0.0.1 ', str(e.exception))
        self.assertEqual((M.format_offers(None), M.format_offers([1, 3]), M.parse_offers('1,3'), M.parse_offers('any')),
                         ('any', '1,3', [1, 3], None))

    def test_rails(self):
        snap = snapshot({})
        self.assertEqual(M.rails_of(snap, 'n1'), ['10.1.0.1', '10.2.0.1'])     # own row, IPv6 left out
        self.assertEqual(M.rails_of(snap, 'n3'), ['10.1.0.3', '10.2.0.3'])
        self.assertEqual(M.node_view(snap), ('10.0.0.1', '10.0.0.1', ['10.1.0.1', '10.2.0.1']))

    def test_probe_problems(self):
        local = {'peers': {'n2': {'probes': {'10.1.0.1': {'10.1.0.2': {'ok': False}, '10.2.0.2': {'ok': True}},
                                             '10.2.0.1': {'10.2.0.2': {'ok': True}}}}}}
        ranks = [('10.0.0.1', 'n1', ('10.1.0.1', '10.2.0.1')), ('10.0.0.2', 'n2', ('10.1.0.2', '10.2.0.2'))]
        p = M.probe_problems(local, ranks, '10.0.0.1', ('10.1.0.1', '10.2.0.1'))
        self.assertEqual(len(p), 1)
        self.assertIn('number their PCIe roots differently', p[0])

    def test_proto(self):
        self.assertTrue(M.major_ok('0.99') and M.major_ok('1.0') and M.major_ok('00.5'))
        self.assertFalse(M.major_ok('2.0') or M.major_ok('x') or M.major_ok('1'))


class DaemonTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeDaemon(snapshot({}))
        self.d = M.Daemon(f'localhost:{self.fake.port}')

    def tearDown(self):
        self.fake.close()

    def test_election_waits_for_a_stable_full_group(self):
        reads = []
        full = {f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in (4, 1, 3, 2)}

        def sleep(_):
            reads.append(1)
            # grows to four members after two reads, then holds
            self.fake.snap = snapshot({k: v for k, v in list(full.items())[:2 + len(reads)]})
        ranks, _ = M.elect(self.d, GROUP, 4, sleep=sleep, log=lambda s: None)
        self.assertEqual([r[0] for r in ranks], ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4'])
        self.assertEqual(ranks[2][2], ('10.1.0.3', '10.2.0.3'))
        self.assertGreaterEqual(len(reads), 3)    # it saw four twice before answering

    def test_election_assigns_held_ranks_out_of_address_order(self):
        """Per-box sets: each box holds one rank, and the addresses sort differently from the ranks."""
        held = {1: '2', 2: '0', 3: '3', 4: '1'}
        self.fake.snap = snapshot({f'a{i}': agent(f'10.0.0.{i}', f'n{i}', offers=held[i]) for i in (1, 2, 3, 4)})
        ranks, _ = M.elect(self.d, GROUP, 4, sleep=lambda _: None, log=lambda s: None)
        self.assertEqual([r[0] for r in ranks], ['10.0.0.2', '10.0.0.4', '10.0.0.1', '10.0.0.3'])
        self.assertEqual(ranks[0][2], ('10.1.0.2', '10.2.0.2'))

    def test_election_waits_for_every_offer(self):
        rows = {f'a{i}': agent(f'10.0.0.{i}', f'n{i}', offers=None if i == 3 else 'any') for i in (1, 2, 3, 4)}
        self.fake.snap = snapshot(rows)
        reads = []

        def sleep(_):
            reads.append(1)
            if len(reads) == 2:      # box 3 publishes its offer late
                rows['a3']['meta'] = {'gr8.offers': 'any'}
        ranks, _ = M.elect(self.d, GROUP, 4, sleep=sleep, log=lambda s: None)
        self.assertEqual(len(ranks), 4)
        self.assertGreaterEqual(len(reads), 3)

    def test_election_refuses_offers_that_do_not_fit(self):
        self.fake.snap = snapshot({f'a{i}': agent(f'10.0.0.{i}', f'n{i}', offers='1' if i < 3 else 'any') for i in (1, 2, 3, 4)})
        with self.assertRaises(M.NoAssignment) as e:
            M.elect(self.d, GROUP, 4, sleep=lambda _: None, log=lambda s: None)
        self.assertIn('10.0.0.1 1, 10.0.0.2 1, 10.0.0.3 any, 10.0.0.4 any', str(e.exception))

    def test_election_refusals(self):
        self.fake.snap = snapshot({f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in range(1, 6)})
        with self.assertRaises(RuntimeError) as e:
            M.elect(self.d, GROUP, 4, sleep=lambda _: None, log=lambda s: None)
        self.assertIn('has 5 members for --tp 4', str(e.exception))
        snap = snapshot({f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in range(1, 5)})
        snap['peers']['n3']['addr_tags'] = {'10.1.0.3': ['rdma']}
        self.fake.snap = snap
        with self.assertRaises(RuntimeError) as e:
            M.elect(self.d, GROUP, 4, sleep=lambda _: None, log=lambda s: None)
        self.assertIn('10.0.0.3 has 1 rdma-tagged', str(e.exception))
        self.fake.snap = snapshot({})
        t = [0.0]

        def clock():
            t[0] += 100
            return t[0]
        with self.assertRaises(RuntimeError) as e:
            M.elect(self.d, GROUP, 4, timeout=300, sleep=lambda _: None, log=lambda s: None, clock=clock)
        self.assertIn('have 0', str(e.exception))

    def test_agent_registers_and_updates_meta(self):
        self.assertEqual(self.d.local_status()['node_ip'], '10.0.0.1')
        a = M.Agent(self.d, GROUP, f'{GROUP}@gr8@10.0.0.1', '10.0.0.1', log=lambda s: None)
        a.set_meta(**{'gr8.state': 'joining'})
        a.start()
        try:
            for _ in range(100):
                if self.fake.registrations:
                    break
                time.sleep(0.05)
            r = self.fake.registrations[0]
            self.assertEqual((r['agent_id'], r['group'], r['node_ip'], r['proto']), (f'{GROUP}@gr8@10.0.0.1', GROUP,
                                                                                     '10.0.0.1', '0.99'))
            self.assertEqual(r['machine']['gpus'][0]['uma'], True)
            self.assertEqual(r['meta'], {'gr8.state': 'joining'})
            a.set_meta(**{'gr8.state': 'loading', 'gr8.rank': 2})
            for _ in range(100):
                if len(self.fake.registrations) > 1:
                    break
                time.sleep(0.05)
            self.assertEqual(self.fake.registrations[-1]['meta'], {'gr8.state': 'loading', 'gr8.rank': '2'})
        finally:
            a.stop()


if __name__ == '__main__':
    unittest.main()
