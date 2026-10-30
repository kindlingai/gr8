#!/usr/bin/env python3
"""The mentat side, against a fake daemon that speaks the framing of mentat's PROTOCOL.md: membership, the election,
the rails, and the agent link with meta updates. No mentatd needed.
    python3 -m unittest discover -s tools/launch"""
import ipaddress
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
import unittest
from unittest import mock

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
    a link's end matters only while that link holds the registration. claim(header, link) answers a claim."""

    def __init__(self, snap=None, group=None, local=None, on_link=None, host='127.0.0.1', claim=None):
        self.snap, self.registrations, self.links, self.claims = snap, [], 0, []
        self.claim_fn, self.hellos = None, []
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
                            fake.hellos.append(h.get('client_id'))
                            s.sendall(M.frame({'t': 'hello_ok', 'req': h['req'], 'proto': '0.99'}))
                        elif t == 'status':
                            s.sendall(M.frame({'t': 'status_ok', 'req': h['req'], 'snapshot': group(h.get('group'))}))
                        elif t == 'claim':
                            fake.claims.append(h)
                            fn = claim or fake.claim_fn
                            reply = fn(h, self) if fn else {'t': 'err', 'code': 'not_head', 'error': 'no'}
                            s.sendall(M.frame({**reply, 'req': h['req']}))
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

    def test_member_rails_take_the_fabric_every_member_shares(self):
        snap = snapshot({})
        # n1 and n3 share a direct cable (10.110/10.111) beside their switch ports (10.1/10.2)
        snap['addr_tags'].update({'10.110.0.0': ['rdma'], '10.111.0.0': ['rdma']})
        snap['addr_ifaces'] = {'10.1.0.1': 'enp1s0f1np1', '10.2.0.1': 'enP2p1s0f1np1',
                               '10.110.0.0': 'enp1s0f0np0', '10.111.0.0': 'enP2p1s0f0np0'}
        snap['peers']['n3']['addr_tags'].update({'10.110.0.1': ['rdma'], '10.111.0.1': ['rdma']})
        ok = {'ok': True}
        snap['peers']['n2']['probes'] = {'10.1.0.1': {'10.1.0.2': ok}, '10.2.0.1': {'10.2.0.2': ok}}
        snap['peers']['n3']['probes'] = {'10.1.0.1': {'10.1.0.3': ok}, '10.2.0.1': {'10.2.0.3': ok},
                                         '10.110.0.0': {'10.110.0.1': ok}, '10.111.0.0': {'10.111.0.1': ok}}
        self.assertEqual(M.member_rails(snap, ['n1', 'n2', 'n3']),
                         {'n1': ('10.1.0.1', '10.2.0.1'), 'n2': ('10.1.0.2', '10.2.0.2'), 'n3': ('10.1.0.3', '10.2.0.3')})
        # two boxes that share the switch and a direct cable: both ports qualify, the lowest address wins
        self.assertEqual(M.member_rails(snap, ['n1', 'n3']),
                         {'n1': ('10.1.0.1', '10.2.0.1'), 'n3': ('10.1.0.3', '10.2.0.3')})
        del snap['peers']['n2']['probes']['10.2.0.1']       # not probed yet: wait
        self.assertIsNone(M.member_rails(snap, ['n1', 'n2', 'n3']))
        self.assertEqual(M.reach_text(snap, ['n1', 'n2', 'n3']),
                         '10.1.0.1 reaches 2 of 2 members; 10.2.0.1 reaches 1 of 2 members; '
                         '10.110.0.0 reaches 1 of 2 members; 10.111.0.0 reaches 1 of 2 members')
        # a pair is one physical port: half-probed, the switch port is out and the cable is the only pair
        del snap['peers']['n3']['probes']['10.2.0.1']
        self.assertEqual(M.member_rails(snap, ['n1', 'n3'])['n3'], ('10.110.0.1', '10.111.0.1'))
        # every member with two: those two, probes or not
        self.assertEqual(M.member_rails(snapshot({}), ['n1', 'n2'])['n2'], ('10.1.0.2', '10.2.0.2'))
        snap['peers']['n2']['addr_tags'] = {'10.0.0.2': ['lan'], '10.1.0.2': ['rdma']}
        with self.assertRaises(RuntimeError) as e:
            M.member_rails(snap, ['n1', 'n2'], {'n2': '10.0.0.2'})
        self.assertIn('10.0.0.2 has 1 rdma-tagged addresses', str(e.exception))

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


def cabled(n, cables, broken=(), skip=()):
    """What mentat knows of n switchless boxes. Box i's LAN address is 10.0.0.<i+1> and its node id n<i+1>. cables:
    [(box, port, box, port)] with port 0 (f0) or 1 (f1). Each port has two RoCE addresses, one per PCIe root, and the
    two ends of a cable share a /24 per root: root 0 on 10.<10+c>.0.x, root 1 on 10.<20+c>.0.x for cable c, so a bound
    probe reaches only the far end of its own cable on its own root. broken: cables that answer no probe. skip:
    addresses whose probes are not in. Returns (links, roots, addr): the view's links[set] as mentat lists them, each
    box's {address: root} (gr8.roots), and {(box, port, root): address}."""
    addr = {}
    for c, (a, pa, b, pb) in enumerate(cables):
        for root in (0, 1):
            net = 10 + c if root == 0 else 20 + c
            addr[(a, pa, root)] = f'10.{net}.0.{a + 1}'
            addr[(b, pb, root)] = f'10.{net}.0.{b + 1}'
    links = []
    for c, (a, pa, b, pb) in enumerate(cables):
        if c in broken:
            continue
        for root in (0, 1):
            x, y = addr[(a, pa, root)], addr[(b, pb, root)]
            if x in skip or y in skip:
                continue
            ends = sorted([(f'n{a + 1}', x), (f'n{b + 1}', y)])
            links.append({'a': {'node': ends[0][0], 'addr': ends[0][1], 'port': ends[0][1], 'iface': None},
                          'b': {'node': ends[1][0], 'addr': ends[1][1], 'port': ends[1][1], 'iface': None},
                          'rtt_ms': 0})
    roots = [{a: k[2] for k, a in addr.items() if k[0] == i} for i in range(n)]
    return links, roots, addr


def solved(n, cables, broken=(), skip=(), cycle=None):
    """A claim_ok view of a ring over the cables, as mentat gives it: its members from the lowest node id along the
    cables (cycle overrides), and links[set]. Returns (view, ids, roots, addr) with ids and roots keyed by LAN."""
    links, roots, addr = cabled(n, cables, broken, skip)
    if cycle is None:
        nxt = {}
        for a, _, b, _ in cables:
            nxt.setdefault(a, []).append(b)
            nxt.setdefault(b, []).append(a)
        cycle, prev = [0], None
        while len(cycle) < n:
            step = min(x for x in nxt[cycle[-1]] if x != prev and x not in cycle)
            prev = cycle[-1]
            cycle.append(step)
    view = {'sets': {'ring': [{'node': f'n{i + 1}'} for i in cycle]}, 'links': {'ring': links}, 'paths': {'ring': {}}}
    lan = [f'10.0.0.{i + 1}' for i in range(n)]
    return view, {lan[i]: f'n{i + 1}' for i in range(n)}, {lan[i]: roots[i] for i in range(n)}, addr


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
            # the direct writes take root 1, as the GB10 default --ring-devs does
            self.assertEqual(where[dn][2], 1)
            if tp == 4:
                self.assertEqual(ro.rsplit('.', 1)[0], nri.rsplit('.', 1)[0], f'rank {r}: relay is not on its cable')
                self.assertEqual(where[ro][1:], (where[dn][1], 0), f'rank {r}: relay-out not root 0 of next\'s port')
                self.assertEqual(where[ri][1:], (where[dp][1], 0), f'rank {r}: relay-in not root 0 of prev\'s port')
            else:
                self.assertEqual((ro, ri), (dn, dp))      # TP=3 opens no relay: the fields repeat the direct pair

    def plan(self, n, cables, offers=None, head=None, **kw):
        view, ids, roots, addr = solved(n, cables, **kw)
        order, ports = M.ring_plan(view, ids, roots, n, offers or dict.fromkeys(ids), head)
        return order, ports, addr

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

    def test_the_plan_does_not_depend_on_mentats_rotation_or_direction(self):
        """mentat starts a ring at its lowest node id, and either way round is its answer: gr8's rule turns it."""
        n, cables = 4, ring([0, 3, 1, 2])
        plans = set()
        for cycle in ([0, 3, 1, 2], [3, 1, 2, 0], [0, 2, 1, 3], [2, 1, 3, 0]):
            view, ids, roots, _ = solved(n, cables, cycle=cycle)
            plans.add(M.ring_text(*M.ring_plan(view, ids, roots, n, dict.fromkeys(ids))))
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
        with self.assertRaises(M.NoAssignment) as e:
            self.plan(3, ring([0, 1, 2]), head='10.0.0.9')
        self.assertIn('--head 10.0.0.9 is not a member', str(e.exception))

    def test_a_view_of_other_nodes_is_stale(self):
        view, ids, roots, _ = solved(3, ring([0, 1, 2]))
        view['sets']['ring'][2]['node'] = 'n9'                # an earlier boot's box still holds the name
        with self.assertRaises(M.Stale) as e:
            M.ring_plan(view, ids, roots, 3, dict.fromkeys(ids))
        self.assertIn("is not the group's 3 members", str(e.exception))
        with self.assertRaises(M.Stale):                       # a ring of another size
            M.ring_plan(solved(3, ring([0, 1, 2]))[0], {**ids, '10.0.0.4': 'n4'}, roots, 4, dict.fromkeys(ids))

    def test_four_boxes_need_both_roots_on_a_cable(self):
        view, ids, roots, addr = solved(4, ring([0, 1, 2, 3]))
        for r in roots.values():                     # root 0 of every port has no RoCE device
            for a in [a for a, k in r.items() if k == 0]:
                del r[a]
        with self.assertRaises(M.Incomplete) as e:
            M.ring_plan(view, ids, roots, 4, dict.fromkeys(ids))
        self.assertIn('link on one port pair only', str(e.exception))
        # a root-0 probe not yet in: the same, and claim_ring claims again
        skip = {a for k, a in solved(4, ring([0, 1, 2, 3]))[3].items() if k == (1, 1, 0)}
        with self.assertRaises(M.Incomplete):
            self.plan(4, ring([0, 1, 2, 3]), skip=skip)
        # three boxes write their neighbours directly and need no relay pair
        view, ids, roots, addr = solved(3, ring([0, 1, 2]))
        for r in roots.values():
            for a in [a for a, k in r.items() if k == 0]:
                del r[a]
        order, ports = M.ring_plan(view, ids, roots, 3, dict.fromkeys(ids))
        self.assertEqual(order, ['10.0.0.1', '10.0.0.2', '10.0.0.3'])
        self.assertEqual(ports[0], (addr[(0, 0, 1)], addr[(0, 0, 1)], addr[(0, 1, 1)], addr[(0, 1, 1)]))
        self.engine_view(order, ports, addr, 3)

    def test_a_partial_view_gives_the_same_tp3_plan(self):
        """At TP=3 the relay pair is unused, so a view without the root-0 links plans the same ring."""
        full = M.ring_text(*self.plan(3, ring([0, 2, 1]))[:2])
        skip = {a for k, a in solved(3, ring([0, 2, 1]))[3].items() if k[2] == 0}
        self.assertEqual(M.ring_text(*self.plan(3, ring([0, 2, 1]), skip=skip)[:2]), full)

    def test_links_off_roce_addresses_do_not_count(self):
        """A link between addresses with no ACTIVE RoCE v2 device (not in gr8.roots) cannot carry the engine's ring."""
        view, ids, roots, addr = solved(3, ring([0, 1, 2]))
        for a in [a for k, a in addr.items() if k[:2] == (0, 0)]:
            del roots['10.0.0.1'][a]                       # box 1's f0 port is down on both roots
        with self.assertRaises(M.NoAssignment) as e:
            M.ring_plan(view, ids, roots, 3, dict.fromkeys(ids))
        self.assertIn('no link between them joins two RoCE v2 addresses', str(e.exception))

    def test_why_names_the_broken_pair(self):
        for n, cables in ((4, ring([0, 2, 1, 3])), (3, ring([0, 1, 2]))):
            with self.subTest(n=n):
                links, roots, addr = cabled(n, cables, broken={1})
                ids = {f'10.0.0.{i + 1}': f'n{i + 1}' for i in range(n)}
                pairs = M.view_pairs({'links': {'ring': links}}, {v: k for k, v in ids.items()})
                why = M.ring_why(pairs, {f'10.0.0.{i + 1}': roots[i] for i in range(n)}, n, 'set "ring" wants a ring')
                a, b = cables[1][0] + 1, cables[1][2] + 1
                self.assertIn(f'no cable works between 10.0.0.{min(a, b)} and 10.0.0.{max(a, b)}', why)
                self.assertIn('Ports that reach no box: ', why)
                self.assertIn(f'10.11.0.{a} on 10.0.0.{a}', why)
                self.assertIn('mentat: set "ring" wants a ring', why)
        links, roots, _ = cabled(4, ring([0, 1, 2]))           # box 4 is cabled to nothing
        roots = {f'10.0.0.{i + 1}': r for i, r in enumerate(roots)}
        why = M.ring_why(M.view_pairs({'links': {'ring': links}}, {f'n{i}': f'10.0.0.{i}' for i in range(1, 5)}),
                         roots, 4, '')
        self.assertIn('10.0.0.4 reaches no box', why)

    def test_roots_meta(self):
        roots = {'10.21.0.1': 1, '10.11.0.1': 0, '10.3.0.1': 0}
        self.assertEqual(M.format_roots(roots), '10.3.0.1=0,10.11.0.1=0,10.21.0.1=1')
        self.assertEqual(M.parse_roots(M.format_roots(roots)), roots)
        self.assertEqual(M.parse_roots(None), {})
        # 16 RoCE addresses still fit mentat's 1024-byte meta value
        self.assertLess(len(M.format_roots({f'10.{i}.{j}.100': 1 for i in range(200, 204) for j in range(200, 204)})),
                        1024)


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
        # rails on /16 subnets: the KV node is on the rail's subnet, though not on its /24
        wide = [('192.0.2.10', 7350, {'gr8.kv.rdma': '10.101.200.3,10.100.200.3'})]
        nets = {'10.100.1.2': ipaddress.ip_network('10.100.0.0/16'),
                '10.101.1.2': ipaddress.ip_network('10.101.0.0/16')}
        self.assertEqual(M.kv_target(wide, ('10.100.1.2', '10.101.1.2'), 1, nets), '10.101.200.3:7350')
        self.assertEqual(M.kv_target(wide, ('10.100.1.2', '10.101.1.2')), '192.0.2.10:7350')
        # a malformed announcement is skipped, as the string compare did
        bad = [('192.0.2.10', 7350, {'gr8.kv.rdma': 'not-an-address,10.100.200.3'})]
        self.assertEqual(M.kv_target(bad, ('10.100.1.2', '10.101.1.2'), 0, nets), '10.100.200.3:7350')


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

    def test_a_head_name_waits_until_it_resolves(self):
        """HEAD=$(LWS_LEADER_ADDRESS) under a LeaderWorkerSet: the leader pod's DNS name, which a worker can read before
        the leader's pod has an address."""
        self.fake.snap = snapshot({f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in (1, 2, 3, 4)})
        calls = []

        def resolve(name):
            calls.append(name)
            if len(calls) <= 2:
                raise socket.gaierror(-2, 'Name or service not known')
            return '10.0.0.3'
        with mock.patch.object(M.socket, 'gethostbyname', resolve):
            ranks, _ = M.elect(self.d, GROUP, 4, 'gr8-0.gr8.ns', sleep=lambda _: None, log=lambda s: None)
        self.assertEqual(set(calls), {'gr8-0.gr8.ns'})
        self.assertGreaterEqual(len(calls), 4)    # two failures, then two equal reads
        self.assertEqual([r[0] for r in ranks], ['10.0.0.3', '10.0.0.1', '10.0.0.2', '10.0.0.4'])
        t = [0.0]

        def clock():
            t[0] += 100
            return t[0]
        with mock.patch.object(M.socket, 'gethostbyname', side_effect=socket.gaierror(-2, 'Name or service not known')):
            with self.assertRaises(RuntimeError) as e:
                M.elect(self.d, GROUP, 4, 'gr8-0.gr8.ns', timeout=300, sleep=lambda _: None, log=lambda s: None,
                        clock=clock)
        self.assertIn('have 4 (10.0.0.1, 10.0.0.2, 10.0.0.3, 10.0.0.4): --head gr8-0.gr8.ns does not resolve',
                      str(e.exception))

    def test_election_refuses_offers_that_do_not_fit(self):
        self.fake.snap = snapshot({f'a{i}': agent(f'10.0.0.{i}', f'n{i}', offers='1' if i < 3 else 'any') for i in (1, 2, 3, 4)})
        with self.assertRaises(M.NoAssignment) as e:
            M.elect(self.d, GROUP, 4, sleep=lambda _: None, log=lambda s: None)
        self.assertIn('10.0.0.1 1, 10.0.0.2 1, 10.0.0.3 any, 10.0.0.4 any', str(e.exception))

    def ring_rows(self, n, cables, **kw):
        view, ids, roots, addr = solved(n, cables, **kw)
        rows = {f'a{i}': agent(f'10.0.0.{i}', f'n{i}') for i in range(1, n + 1)}
        for i in range(1, n + 1):
            rows[f'a{i}']['meta'][M.ROOTS_META] = M.format_roots(roots[f'10.0.0.{i}'])
        self.fake.snap = snapshot(rows)
        return view, rows

    def ring_elect(self, n, link=None, **kw):
        t, mine = [0.0], link is None
        link = link or M.ClaimLink(self.d, GROUP, 'test')

        def sleep(s):
            t[0] += s
        try:
            ranks, _ = M.elect(self.d, GROUP, n, sleep=sleep, log=lambda s: None, clock=lambda: t[0], ring=link, **kw)
        finally:
            if mine:
                link.close()
        return ranks, t[0]

    def test_a_ring_election_claims_until_mentat_finds_the_ring(self):
        view, rows = self.ring_rows(3, ring([0, 2, 1]))
        answers = [{'t': 'err', 'code': 'unsolvable_claim', 'error': 'set "ring" wants a ring'}] * 2 + \
            [{'t': 'claim_ok', 'name': 'x', 'generation': 7, 'head_node_id': 'n2', 'view': view}]
        self.fake.claim_fn = lambda h, link: answers.pop(0) if len(answers) > 1 else answers[0]
        link = M.ClaimLink(self.d, GROUP, 'test')
        ranks, _ = self.ring_elect(3, link)
        self.assertEqual([r[0] for r in ranks], ['10.0.0.1', '10.0.0.2', '10.0.0.3'])
        self.assertEqual(link.tag, f'{GROUP}-ring3@n2:7')
        c = self.fake.claims
        self.assertEqual(len(c), 3)
        self.assertEqual({h['name'] for h in c}, {f'{GROUP}-ring3'})
        self.assertEqual(c[0]['shape'], {'sets': [{'name': 'ring', 'bundles': 3, 'link': 'rdma', 'layout': 'ring'}],
                                         'driver': 'ignore'})
        self.assertIsNotNone(link.sock)       # the link holds the claim until the supervisor closes it
        link.close()

    def test_a_ring_claims_only_after_two_equal_reads(self):
        view, rows = self.ring_rows(3, ring([0, 1, 2]))
        sleeps, at_claim = [0], []

        def claim(h, link):
            at_claim.append(sleeps[0])
            return {'t': 'claim_ok', 'view': view}
        self.fake.claim_fn = claim

        def sleep(_):
            sleeps[0] += 1
            if sleeps[0] == 1:       # box 3 republishes its roots: the first read does not count
                rows['a3']['meta'][M.ROOTS_META] += ',10.99.0.3=0'
        link = M.ClaimLink(self.d, GROUP, 'test')
        try:
            M.elect(self.d, GROUP, 3, sleep=sleep, log=lambda s: None, ring=link)
        finally:
            link.close()
        self.assertEqual(at_claim, [2])

    def test_a_tp4_retry_stops_at_the_join_timeout(self):
        skip = {a for k, a in solved(4, ring([0, 1, 2, 3]))[3].items() if k == (1, 1, 0)}
        partial, _ = self.ring_rows(4, ring([0, 1, 2, 3]), skip=skip)
        self.fake.claim_fn = lambda h, link: {'t': 'claim_ok', 'view': partial}
        with self.assertRaises(RuntimeError) as e:
            self.ring_elect(4, timeout=40, ring_retry=15, ring_attempts=100)
        self.assertIn('timed out: ', str(e.exception))
        self.assertIn('link on one port pair only', str(e.exception))

    def test_a_ring_mentat_never_finds_is_refused_naming_the_pair(self):
        links, roots, _ = cabled(3, ring([0, 1, 2]), broken={2})
        self.ring_rows(3, ring([0, 1, 2]))

        def claim(h, link):
            if h['shape']['sets'][0]['layout'] == 'any':      # the why: every link between the members
                return {'t': 'claim_ok', 'view': {'sets': {'ring': []}, 'links': {'ring': links}}}
            return {'t': 'err', 'code': 'unsolvable_claim', 'error': 'set "ring" wants a ring of 3 nodes'}
        self.fake.claim_fn = claim
        with self.assertRaises(M.NoAssignment) as e:
            self.ring_elect(3, ring_settle=45)
        self.assertIn('no cable works between 10.0.0.1 and 10.0.0.3', str(e.exception))
        self.assertIn('mentat: set "ring" wants a ring of 3 nodes', str(e.exception))

    def test_a_tp4_view_without_relay_pairs_claims_a_fresh_name(self):
        skip = {a for k, a in solved(4, ring([0, 1, 2, 3]))[3].items() if k == (1, 1, 0)}
        partial, _ = self.ring_rows(4, ring([0, 1, 2, 3]), skip=skip)
        full, _ = self.ring_rows(4, ring([0, 1, 2, 3]))
        self.fake.claim_fn = lambda h, link: {'t': 'claim_ok', 'generation': 1, 'head_node_id': 'n1',
                                              'view': full if h['name'].endswith('-2') else partial}
        link = M.ClaimLink(self.d, GROUP, 'test')
        ranks, waited = self.ring_elect(4, link, ring_retry=15)
        self.assertEqual(len(ranks), 4)
        self.assertEqual([h['name'] for h in self.fake.claims], [f'{GROUP}-ring4', f'{GROUP}-ring4-1',
                                                                  f'{GROUP}-ring4-2'])
        self.assertEqual(link.tag, f'{GROUP}-ring4-2@n1:1')
        link.close()
        self.fake.claims.clear()
        self.fake.claim_fn = lambda h, link: {'t': 'claim_ok', 'view': partial}
        with self.assertRaises(M.NoAssignment) as e:
            self.ring_elect(4, ring_attempts=2)
        self.assertIn('link on one port pair only', str(e.exception))
        self.assertIn('2 claims', str(e.exception))

    def test_a_stale_view_is_let_go_and_claimed_again(self):
        view, _ = self.ring_rows(3, ring([0, 1, 2]))
        stale = json.loads(json.dumps(view))
        stale['sets']['ring'][0]['node'] = 'old'
        answers = [stale, stale, view]
        self.fake.claim_fn = lambda h, link: {'t': 'claim_ok', 'view': answers.pop(0)}
        ranks, _ = self.ring_elect(3)
        self.assertEqual(len(ranks), 3)
        self.assertEqual(self.fake.hellos.count('test'), 3)    # a new link each time: the old one let the claim go

    def test_a_ring_election_waits_for_every_roots_meta_and_refuses_old_mentat(self):
        view, rows = self.ring_rows(3, ring([0, 1, 2]))
        del rows['a2']['meta'][M.ROOTS_META]
        self.fake.claim_fn = lambda h, link: {'t': 'claim_ok', 'view': {'sets': view['sets']}}
        n = [0]

        def sleep(_):
            n[0] += 1
            if n[0] == 3:
                rows['a2']['meta'][M.ROOTS_META] = M.format_roots({'10.10.0.2': 0})
        link = M.ClaimLink(self.d, GROUP, 't')
        with self.assertRaises(M.NoAssignment) as e:
            M.elect(self.d, GROUP, 3, sleep=sleep, log=lambda s: None, ring=link)
        link.close()
        self.assertGreaterEqual(n[0], 3)
        self.assertEqual(len(self.fake.claims), 1)            # no claim before every box published its roots
        self.assertIn('needs mentatd 0.24.0 or later', str(e.exception))

    def test_a_refused_claim_stops_and_a_missing_head_times_out(self):
        self.ring_rows(3, ring([0, 1, 2]))
        self.fake.claim_fn = lambda h, link: {'t': 'err', 'code': 'shape_conflict', 'error': 'held for another shape'}
        with self.assertRaises(M.NoAssignment) as e:
            self.ring_elect(3)
        self.assertIn('held for another shape (shape_conflict)', str(e.exception))
        self.fake.claim_fn = lambda h, link: {'t': 'err', 'code': 'not_head', 'error': 'send claims to n1'}
        with self.assertRaises(RuntimeError) as e:
            self.ring_elect(3, timeout=60)
        self.assertIn('timed out: send claims to n1 (not_head)', str(e.exception))

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
