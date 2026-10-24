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


def cabled(n, cables, broken=()):
    """Local /status snapshots of n switchless boxes. Box i's LAN address is 10.0.0.<i+1>. cables: [(box, port, box,
    port)] with port 0 (f0) or 1 (f1). Each port shows two RoCE addresses, one per PCIe root, and the two ends of a
    cable share a /24 per root: root 0 on 10.<10+c>.0.x, root 1 on 10.<20+c>.0.x for cable c. A probe is bound to its
    address, so it reaches only the far end of its own cable on its own root (mentat's PROTOCOL.md, "Probes").
    broken: cable indices that answer no probe. Returns (snaps, roots): snaps[i] is box i's local status, roots[i]
    its {address: root}, as fabric.ring_ports gives them."""
    addr = {}      # (box, port, root) -> address
    for c, (a, pa, b, pb) in enumerate(cables):
        for root in (0, 1):
            net = 10 + c if root == 0 else 20 + c
            addr[(a, pa, root)] = f'10.{net}.0.{a + 1}'
            addr[(b, pb, root)] = f'10.{net}.0.{b + 1}'
    snaps, roots = [], []
    for i in range(n):
        mine = {a: k[2] for k, a in addr.items() if k[0] == i}
        tags = {f'10.0.0.{i + 1}': ['lan'], **{a: ['connectx', 'rdma'] for a in mine}}
        peers = {}
        for j in range(n):
            if j == i:
                continue
            theirs = {a for k, a in addr.items() if k[0] == j}
            probes = {}
            for x in mine:
                row = {}
                for y in theirs:
                    if x.rsplit('.', 1)[0] != y.rsplit('.', 1)[0]:
                        continue             # a bound probe stays in its own subnet
                    cable = int(x.split('.')[1]) % 10
                    row[y] = {'ok': cable not in broken, 'rtt_ms': 0, 'direct': True if cable not in broken else None}
                if row:
                    probes[x] = row
            peers[f'n{j + 1}'] = {'node_ip': f'10.0.0.{j + 1}', 'probes': probes}
        snaps.append({'node_id': f'n{i + 1}', 'node_ip': f'10.0.0.{i + 1}', 'addr_tags': tags, 'peers': peers})
        roots.append(mine)
    return snaps, roots, addr


def ring_meta(snaps, roots):
    return {s['node_ip']: M.own_links(s, r) for s, r in zip(snaps, roots)}


def ring(order):
    """The cables of a ring in this order, each box's f0 to the next box's f1 (the --peers convention)."""
    return [(a, 0, b, 1) for a, b in zip(order, order[1:] + order[:1])]


def flipped(order):
    """The other direction: each box's f1 to the next box's f0."""
    return [(a, 1, b, 0) for a, b in zip(order, order[1:] + order[:1])]


class RingPlanTests(unittest.TestCase):
    def engine_view(self, order, ports, addr, tp):
        """Check what engine/comms/fabric.cpp's ring needs of --ring-devs: rank r's direct-next device is cabled to
        rank r+1's direct-prev one, and (TP=4) its relay-out to rank r+1's relay-in, on other ports of the same cable."""
        where = {a: k for k, a in addr.items()}
        for r in range(tp):
            ro, dn, ri, dp = ports[r]
            nro, ndn, nri, ndp = ports[(r + 1) % tp]
            me, nxt = int(order[r].rsplit('.', 1)[1]) - 1, int(order[(r + 1) % tp].rsplit('.', 1)[1]) - 1
            self.assertEqual((where[dn][0], where[ndp][0]), (me, nxt))
            self.assertEqual(dn.rsplit('.', 1)[0], ndp.rsplit('.', 1)[0], f'rank {r}: next is not on its cable')
            self.assertNotEqual(where[dn][1], where[dp][1], f'rank {r}: next and prev on one port')
            if tp == 4:
                self.assertEqual(ro.rsplit('.', 1)[0], nri.rsplit('.', 1)[0], f'rank {r}: relay is not on its cable')
                self.assertNotEqual(ro, dn)
                self.assertEqual(where[ro][1], where[dn][1], f'rank {r}: relay-out on another port than next')
                self.assertEqual(where[ri][1], where[dp][1], f'rank {r}: relay-in on another port than prev')
                # the direct writes take root 1 and the relay root 0, as the GB10 default --ring-devs does
                self.assertEqual((where[dn][2], where[ro][2]), (1, 0))

    def plan(self, n, cables, offers=None, head=None, broken=()):
        snaps, roots, addr = cabled(n, cables, broken)
        meta = ring_meta(snaps, roots)
        order, ports = M.ring_plan(meta, n, offers or dict.fromkeys(meta), head)
        return order, ports, addr

    def test_a_partial_probe_table_gives_the_same_tp3_plan(self):
        """mentat fills its table a round at a time. At TP=3 the relay pair is unused, so a box whose root-0 rows are
        not in yet derives the same plan (the handshake compares it) as one with the full table."""
        snaps, roots, addr = cabled(3, ring([0, 2, 1]))
        full = M.ring_text(*M.ring_plan(ring_meta(snaps, roots), 3, dict.fromkeys(f'10.0.0.{i}' for i in (1, 2, 3))))
        for s in snaps:
            for p in s['peers'].values():
                for x in [x for x in p['probes'] if x.split('.')[1].startswith('1')]:
                    del p['probes'][x]          # root 0 (10.1x.*) not probed yet
        order, ports = M.ring_plan(ring_meta(snaps, roots), 3, dict.fromkeys(f'10.0.0.{i}' for i in (1, 2, 3)))
        self.assertEqual(M.ring_text(order, ports), full)
        for ro, dn, ri, dp in ports:
            self.assertEqual((ro, ri), (dn, dp))
            self.assertTrue(dn.startswith('10.2') and dp.startswith('10.2'))     # root 1

    def test_fewer_boxes_or_a_stranger_head(self):
        snaps, roots, _ = cabled(3, ring([0, 1, 2]))
        meta = ring_meta(snaps, roots)
        with self.assertRaises(M.NoAssignment) as e:
            M.ring_plan(meta, 4, dict.fromkeys(meta))
        self.assertIn('links of 3 boxes form no ring of 4', str(e.exception))
        with self.assertRaises(M.NoAssignment) as e:
            M.ring_plan(meta, 3, dict.fromkeys(meta), head='10.0.0.9')
        self.assertIn('--head 10.0.0.9 is not a member', str(e.exception))

    def test_three_and_four_box_rings_in_both_directions(self):
        for n, cable_order in ((3, [0, 1, 2]), (3, [0, 2, 1]), (4, [0, 1, 2, 3]), (4, [0, 2, 1, 3]), (4, [0, 3, 1, 2])):
            for cables in (ring(cable_order), flipped(cable_order)):
                with self.subTest(n=n, cables=cables):
                    order, ports, addr = self.plan(n, cables)
                    lan = [f'10.0.0.{i + 1}' for i in range(n)]
                    want = [lan[i] for i in cable_order]
                    self.assertEqual(order[0], '10.0.0.1')                           # the lowest address leads
                    self.assertIn(order, (want, [want[0]] + want[1:][::-1]))          # along the cables
                    self.assertLess(M.ip_key(order[1]), M.ip_key(order[-1]))         # toward the lower neighbour
                    self.engine_view(order, ports, addr, n)

    def test_four_boxes_out_of_address_order(self):
        """LeaderWorkerSet places the pods anywhere: box 3 sits between 1 and 2 on the cables."""
        order, ports, addr = self.plan(4, ring([0, 2, 1, 3]))
        self.assertEqual(order, ['10.0.0.1', '10.0.0.3', '10.0.0.2', '10.0.0.4'])
        # rank 0's next cable is its f0 here: relay-out root 0, direct root 1
        self.assertEqual(ports[0], (addr[(0, 0, 0)], addr[(0, 0, 1)], addr[(0, 1, 0)], addr[(0, 1, 1)]))
        order, ports, addr = self.plan(4, flipped([0, 2, 1, 3]))
        self.assertEqual(order, ['10.0.0.1', '10.0.0.3', '10.0.0.2', '10.0.0.4'])
        self.assertEqual(ports[0], (addr[(0, 1, 0)], addr[(0, 1, 1)], addr[(0, 0, 0)], addr[(0, 0, 1)]))

    def test_every_box_derives_the_same_plan(self):
        snaps, roots, _ = cabled(4, ring([0, 3, 1, 2]))
        meta = ring_meta(snaps, roots)
        plans = {M.ring_text(*M.ring_plan(dict(reversed(list(meta.items()))), 4, dict.fromkeys(meta)))}
        plans.add(M.ring_text(*M.ring_plan(meta, 4, dict.fromkeys(meta))))
        self.assertEqual(len(plans), 1)

    def test_head_and_offers_turn_the_ring(self):
        order, _, _ = self.plan(4, ring([0, 1, 2, 3]), head='10.0.0.3')
        self.assertEqual(order, ['10.0.0.3', '10.0.0.2', '10.0.0.1', '10.0.0.4'])
        held = {'10.0.0.1': [2], '10.0.0.2': None, '10.0.0.3': [0], '10.0.0.4': None}
        order, ports, addr = self.plan(4, ring([0, 1, 2, 3]), offers=held)
        self.assertEqual(order, ['10.0.0.3', '10.0.0.2', '10.0.0.1', '10.0.0.4'])
        self.engine_view(order, ports, addr, 4)
        with self.assertRaises(M.NoAssignment) as e:     # rank 0 and rank 1 cannot both sit on 10.0.0.1's cables
            self.plan(4, ring([0, 1, 2, 3]), offers={'10.0.0.1': [0], '10.0.0.2': None, '10.0.0.3': [1],
                                                     '10.0.0.4': None})
        self.assertIn('no rank order along the ring (10.0.0.1 - 10.0.0.2 - 10.0.0.3 - 10.0.0.4)', str(e.exception))
        self.assertIn('10.0.0.3 1', str(e.exception))

    def test_a_broken_cable_is_refused_naming_the_pair(self):
        for n, cables in ((4, ring([0, 2, 1, 3])), (3, ring([0, 1, 2]))):
            with self.subTest(n=n):
                with self.assertRaises(M.NoAssignment) as e:
                    self.plan(n, cables, broken={1})
                a, b = cables[1][0] + 1, cables[1][2] + 1
                self.assertIn(f'no cable works between 10.0.0.{min(a, b)} and 10.0.0.{max(a, b)}', str(e.exception))
                self.assertIn('Ports that reach no box: ', str(e.exception))
                self.assertIn(f'10.11.0.{a} on 10.0.0.{a}', str(e.exception))

    def test_a_port_that_reaches_no_box_and_a_missing_box(self):
        # box 4 is cabled to nothing: a ring of 3 and an idle box
        snaps, roots, _ = cabled(4, ring([0, 1, 2]))
        with self.assertRaises(M.NoAssignment) as e:
            M.ring_plan(ring_meta(snaps, roots), 4, dict.fromkeys(f'10.0.0.{i}' for i in range(1, 5)))
        self.assertIn('10.0.0.4 reaches no box', str(e.exception))

    def test_four_boxes_need_both_roots_on_a_cable(self):
        snaps, roots, addr = cabled(4, ring([0, 1, 2, 3]))
        for s, r in zip(snaps, roots):                      # root 0 of every port unaddressed
            for a in [a for a, k in r.items() if k == 0]:
                del r[a]
        with self.assertRaises(M.NoAssignment) as e:
            M.ring_plan(ring_meta(snaps, roots), 4, dict.fromkeys(f'10.0.0.{i}' for i in range(1, 5)))
        self.assertIn('link on one port pair only', str(e.exception))
        # three boxes write their neighbours directly and need no relay pair
        snaps, roots, addr = cabled(3, ring([0, 1, 2]))
        for r in roots:
            for a in [a for a, k in r.items() if k == 0]:
                del r[a]
        order, ports = M.ring_plan(ring_meta(snaps, roots), 3, dict.fromkeys(f'10.0.0.{i}' for i in range(1, 4)))
        self.assertEqual(order, ['10.0.0.1', '10.0.0.2', '10.0.0.3'])
        self.assertEqual(ports[0], (addr[(0, 0, 1)], addr[(0, 0, 1)], addr[(0, 1, 1)], addr[(0, 1, 1)]))
        self.engine_view(order, ports, addr, 3)

    def test_forwarded_and_failed_probes_are_not_links(self):
        snaps, roots, _ = cabled(3, ring([0, 1, 2]))
        for p in snaps[0]['peers'].values():
            for row in p['probes'].values():
                for y in row:
                    row[y]['direct'] = False
        roots_, links = M.parse_links(M.own_links(snaps[0], roots[0]))
        self.assertEqual((sorted(roots_), links), (sorted(roots[0]), set()))
        self.assertEqual(M.parse_links(M.own_links({'peers': {}}, {})), ({}, set()))
        # an IPv6 far end, a null direct (kept: the MAC is unknown) and an address this box cannot use
        x = sorted(roots[0])[0]
        local = {'addr_tags': {x: ['rdma'], '10.99.0.1': ['rdma']}, 'peers': {'n2': {'probes': {
            x: {'fe80::2': {'ok': True, 'direct': True}, '10.11.0.2': {'ok': True, 'direct': None}},
            '10.99.0.1': {'10.99.0.2': {'ok': True, 'direct': True}}}}}}
        self.assertEqual(M.parse_links(M.own_links(local, {x: 0}))[1], {(x, '10.11.0.2')})

    def test_meta_fits_mentat(self):
        snaps, roots, _ = cabled(4, ring([0, 1, 2, 3]))
        self.assertLess(len(M.own_links(snaps[0], roots[0])), M.META_MAX)
        big = {f'10.{i}.{j}.1': 0 for i in range(12) for j in range(12)}
        with self.assertRaises(RuntimeError):
            M.own_links({'addr_tags': {a: ['rdma'] for a in big}, 'peers': {}}, big)


class KvTargetTests(unittest.TestCase):
    def test_rdma_on_a_rail_subnet_first_else_the_lan(self):
        nodes = [('192.0.2.10', 7350, {'gr8.kv.rdma': '10.9.0.10'}), ('192.0.2.20', 7351, {'gr8.kv.rdma': '10.2.0.20,10.1.0.20'})]
        self.assertEqual(M.kv_target(nodes, ('10.1.0.3', '10.2.0.3')), '10.1.0.20:7351')
        # odd ranks start on the second rail, so the node's two PCIe roots each serve half the ranks
        self.assertEqual([M.kv_target(nodes, ('10.1.0.3', '10.2.0.3'), r) for r in range(4)],
                         ['10.1.0.20:7351', '10.2.0.20:7351', '10.1.0.20:7351', '10.2.0.20:7351'])
        self.assertEqual(M.kv_target(nodes, ('10.1.0.3', ''), 1), '10.1.0.20:7351')
        self.assertEqual(M.kv_target(nodes, ('10.5.0.3', '10.6.0.3')), '192.0.2.10:7350')
        self.assertEqual(M.kv_target([('192.0.2.10', 7350, {})], ('', '')), '192.0.2.10:7350')
        self.assertIsNone(M.kv_target([], ('10.1.0.3', '10.2.0.3')))


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

    def test_a_ring_election_waits_for_links_that_hold(self):
        rows = {f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in (1, 2, 3)}
        snaps, roots, _ = cabled(3, ring([0, 2, 1]))
        meta = ring_meta(snaps, roots)
        partial = {ip: t.split(';')[0] + ';' for ip, t in meta.items()}   # no links probed yet
        for i in (1, 2, 3):
            rows[f'a{i}']['meta'][M.LINKS_META] = partial[f'10.0.0.{i}']
        self.fake.snap = snapshot(rows)
        t, reads = [0.0], []

        def sleep(s):
            t[0] += s
            reads.append(t[0])
            if len(reads) == 3:      # the probes come in
                for i in (1, 2, 3):
                    rows[f'a{i}']['meta'][M.LINKS_META] = meta[f'10.0.0.{i}']
        ranks, _ = M.elect(self.d, GROUP, 3, sleep=sleep, log=lambda s: None, clock=lambda: t[0], ring=lambda: None,
                           ring_stable=20, ring_settle=45)
        self.assertEqual([r[0] for r in ranks], ['10.0.0.1', '10.0.0.2', '10.0.0.3'])
        self.assertGreaterEqual(t[0] - reads[2], 20)        # the full links held a probe round first

    def test_a_ring_election_whose_links_keep_changing_times_out(self):
        rows = {f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in (1, 2, 3)}
        snaps, roots, _ = cabled(3, ring([0, 1, 2]))
        meta = ring_meta(snaps, roots)
        self.fake.snap = snapshot(rows)
        t, n = [0.0], [0]

        def flap():
            n[0] += 1
            for i in (1, 2, 3):
                rows[f'a{i}']['meta'][M.LINKS_META] = meta[f'10.0.0.{i}'] if n[0] % 2 else ';'

        def sleep(s):
            t[0] += s
        with self.assertRaises(RuntimeError) as e:
            M.elect(self.d, GROUP, 3, timeout=60, sleep=sleep, log=lambda s: None, clock=lambda: t[0], ring=flap)
        self.assertIn('ring links kept changing', str(e.exception))

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
