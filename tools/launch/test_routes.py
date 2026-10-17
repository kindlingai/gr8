"""routes.py: the hybrid route table on a synthetic 8-box topology shaped like a two-rail switch with cabled extras.
    python3 -m unittest discover -s tools/launch -p 'test_routes.py'

Ranks (all addresses are documentation-style synthetic ones):
  0      switch box with a cable to rank 7      (cable spark B)
  1-4    switch-only boxes; rank 1 also holds an undeclared stray alias in the 0-7 cable subnet
  5      switch box with a cable to rank 6      (cable spark A)
  6      cable box, cabled to ranks 5 and 7     (box A)
  7      cable box, cabled to ranks 0 and 6     (box B)
Each port has two roots, each its own /24: switch 172.16.1/172.16.2, 5-6 172.16.11/12, 0-7 172.16.21/22,
6-7 172.16.31/32.
"""
import collections
import io
import itertools
import os
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routes  # noqa: E402

SW = ('172.16.1', '172.16.2')
CAB_A = ('172.16.11', '172.16.12')    # rank 5 <-> rank 6
CAB_B = ('172.16.21', '172.16.22')    # rank 0 <-> rank 7
CAB_AB = ('172.16.31', '172.16.32')   # rank 6 <-> rank 7
STRAY = '172.16.21.200/24'            # rank 1's undeclared alias, in the rank 0 <-> rank 7 subnet


def port(subnets, host):
    return [f'{s}.{host}/24' for s in subnets]


def fixture():
    p = [port(SW, 10 + r) for r in range(6)]
    p[0] += port(CAB_B, 1)
    p[5] += port(CAB_A, 1)
    p.append(port(CAB_AB, 6) + port(CAB_A, 6))   # rank 6: f0 to rank 7, f1 to rank 5
    p.append(port(CAB_B, 7) + port(CAB_AB, 7))   # rank 7: f0 to rank 0, f1 to rank 6
    return p


def subnet(ip):
    return ip.rsplit('.', 1)[0]


def hops(t, s, d):
    r = t.route(s, d)
    if r[0] == 'direct':
        return [(s, d, r[1], r[2])]
    via, a, vi, vo, b = r[1:]
    return [(s, via, a, vi), (via, d, vo, b)]


class Routes(unittest.TestCase):
    def setUp(self):
        self.t = routes.compute(fixture())

    def test_56_directed_routes_at_most_one_relay_hop(self):
        t = self.t
        self.assertEqual(len(t.direct), 18)
        self.assertEqual(len(t.relay), 20)
        self.assertEqual(2 * len(t.direct) + len(t.relay), 56)
        for s, d in itertools.permutations(range(8), 2):
            self.assertIn(len(hops(t, s, d)), (1, 2), (s, d))

    def test_rule_a_vias(self):
        t = self.t
        far6 = sorted(d for (s, d) in t.relay if s == 6)
        far7 = sorted(d for (s, d) in t.relay if s == 7)
        self.assertEqual(far6, [0, 1, 2, 3, 4])
        self.assertEqual(far7, [1, 2, 3, 4, 5])
        for (s, d), r in t.relay.items():
            far = 6 if 6 in (s, d) else 7
            self.assertEqual(r[0], 5 if far == 6 else 0, (s, d))   # 6 via its cable box 5; 7 via 0
            self.assertEqual(t.relay[(d, s)][0], r[0])               # the reverse goes through the same via
        # the two pairs with two candidate vias take the lowest rank
        self.assertEqual(t.relay[(0, 6)][0], 5)
        self.assertEqual(t.relay[(5, 7)][0], 0)
        self.assertEqual(collections.Counter(r[0] for r in t.relay.values()), {5: 10, 0: 10})

    def test_every_hop_on_one_subnet_and_the_endpoints_own_it(self):
        t, decl = self.t, fixture()
        for s, d in itertools.permutations(range(8), 2):
            for x, y, ix, iy in hops(t, s, d):
                self.assertEqual(subnet(ix), subnet(iy), (s, d))
                self.assertIn(f'{ix}/24', decl[x])
                self.assertIn(f'{iy}/24', decl[y])
                switch_pair = x < 6 and y < 6
                self.assertEqual(subnet(ix) in SW, switch_pair, (s, d, ix))   # switch pairs on the switch only

    def test_switch_pairs_use_mesh_rails(self):
        for (a, b), (ia, _) in self.t.direct.items():
            if a < 6 and b < 6:
                self.assertEqual(subnet(ia), SW[(a + b) % 2], (a, b))

    def test_canonical_switch_order_follows_the_declarations(self):
        p = [list(reversed(x[:2])) + x[2:] for x in fixture()[:6]] + fixture()[6:]
        t = routes.compute(p)
        self.assertEqual([str(n) for n in t.switch_order], ['172.16.2.0/24', '172.16.1.0/24'])
        self.assertEqual(subnet(t.direct[(0, 2)][0]), '172.16.2')   # (0+2)%2 = rail 0 of the order (IPS in a mesh env)
        self.assertEqual(subnet(t.direct[(0, 1)][0]), '172.16.1')   # (0+1)%2 = rail 1 (IPS2)
        p[3] = fixture()[3]
        with self.assertRaisesRegex(routes.RouteError, 'rank 3 lists the switch rails .* in another order'):
            routes.compute(p)

    def test_rail_balance_within_one(self):
        """Directed flows sent and received per (rank, rail) on the two roots of each port differ by at most one, on
        every port of every rank (the relays' ports are the hot links)."""
        sent, got = collections.Counter(), collections.Counter()
        for s, d in itertools.permutations(range(8), 2):
            for x, y, ix, iy in hops(self.t, s, d):
                sent[(x, subnet(ix))] += 1
                got[(y, subnet(iy))] += 1
        for r, p in enumerate(fixture()):
            nets = [subnet(i) for i in p]
            for pair in (nets[0:2], nets[2:4]):
                for load in (sent, got) if pair else ():
                    a, b = load[(r, pair[0])], load[(r, pair[1])]
                    self.assertLessEqual(abs(a - b), 1, (r, pair, a, b))
        self.assertEqual(sent[(5, '172.16.1')] + sent[(5, '172.16.2')], 11)   # the via's switch port: 11 flows out

    def test_stray_alias_makes_no_adjacency(self):
        t = self.t
        self.assertNotIn((1, 7), t.direct)
        self.assertEqual(t.relay[(1, 7)][0], 0)
        self.assertNotIn(STRAY.split('/')[0], t.text)
        # declared, the alias would make a 3-member subnet: refused as a pair sharing switch and cable rails
        p = fixture()
        p[1] = p[1] + [STRAY]
        with self.assertRaisesRegex(routes.RouteError, 'share switch rails and cable rails'):
            routes.compute(p)

    def test_byte_stable(self):
        a = routes.compute(fixture())
        b = routes.compute([','.join(x) for x in fixture()])
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.sha, b.sha)
        self.assertEqual(len(a.sha), 64)
        lines = a.text.splitlines()
        self.assertEqual([x.split()[0] for x in lines], ['P'] * 8 + ['D'] * 18 + ['R'] * 20)
        self.assertEqual(lines[0], 'P 0 172.16.1.10/24 172.16.2.10/24 172.16.21.1/24 172.16.22.1/24')
        self.assertTrue(a.text.endswith('\n'))
        # the sha pins the whole layout: change it on purpose, and the test says so
        self.assertEqual(a.sha, GOLDEN_SHA, a.text)

    def test_deny_forces_relays(self):
        # T1b on four switch boxes: the diagonals relay through the lowest common neighbour
        sw4 = [port(SW, 10 + r) for r in range(4)]
        t = routes.compute(sw4, routes.parse_deny('0-2,1-3', 4))
        self.assertEqual(sorted(t.direct), [(0, 1), (0, 3), (1, 2), (2, 3)])
        self.assertEqual({k: v[0] for k, v in t.relay.items()}, {(0, 2): 1, (2, 0): 1, (1, 3): 0, (3, 1): 0})
        # T1c: rank 3 is like a cable box; rank 2 relays 4 routes
        t = routes.compute(sw4, routes.parse_deny('3-0,3-1', 4))
        self.assertEqual(collections.Counter(v[0] for v in t.relay.values()), {2: 4})
        for (s, d), v in t.relay.items():
            self.assertEqual(subnet(v[1]), subnet(v[2]))
            self.assertEqual(subnet(v[3]), subnet(v[4]))
        # no deny on a switch: exactly mesh, no relays
        t = routes.compute(sw4)
        self.assertEqual((len(t.direct), t.relay), (6, {}))

    def test_unroutable_is_refused_with_the_pair(self):
        p = fixture()
        p[5] = p[5][:2]                       # rank 6 loses its cable to rank 5: its switch peers have no common neighbour
        with self.assertRaisesRegex(routes.RouteError, r'no route between rank 1 \(b\) and rank 6 \(g\)'):
            routes.compute(p, names='abcdefgh')
        with self.assertRaisesRegex(routes.RouteError, r'no route between rank 0 and rank 1: denied by FAB_DENY'):
            routes.compute([port(SW, 1), port(SW, 2)], {(0, 1)})

    def test_declaration_errors(self):
        for bad, why in ((['172.16.1.1'], 'with its prefix'), (['172.16.1.0/24'], 'network or broadcast'),
                         (['172.16.1.1/24', '172.16.1.2/24'], 'overlap'), (['a/24'], 'with its prefix'),
                         ([port(SW, 1)[0]] * 5, '1 to 4')):
            with self.assertRaisesRegex(routes.RouteError, why):
                routes.parse_ports(','.join(bad))
        with self.assertRaisesRegex(routes.RouteError, 'both declare 172.16.1.1'):
            routes.compute([port(SW, 1), port(SW, 1)])
        with self.assertRaisesRegex(routes.RouteError, 'overlaps'):
            routes.compute([['172.16.1.1/24'], ['172.16.1.2/16']])
        for bad in ('0-0', '0-8', '1', 'a-b'):
            with self.assertRaises(routes.RouteError):
                routes.parse_deny(bad, 8)
        self.assertEqual(routes.parse_deny('3-0, 1-2', 4), {(0, 3), (1, 2)})

    def test_dry_run_cli(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = routes.main([','.join(x) for x in fixture()])
        self.assertEqual(rc, 0)
        self.assertTrue(out.getvalue().endswith(f'routes sha {self.t.sha}\n'))
        with redirect_stdout(out), redirect_stderr(err):
            rc = routes.main([port(SW, 1)[0], port(SW, 2)[0], '--deny', '0-1'])
        self.assertEqual(rc, 2)
        self.assertIn('no route between rank 0 and rank 1', err.getvalue())


GOLDEN_SHA = 'ce9bfe95f1a18cefc8fa1e036cd02a1b4a34ec3b7e4c9252594cf33a903091a8'

if __name__ == '__main__':
    unittest.main()
