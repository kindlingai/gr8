#!/usr/bin/env python3
"""Fabric discovery from a fake /sys/class/infiniband, and finding this box in the peer list.
    python3 -m unittest discover -s tools/launch"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabric as F  # noqa: E402


def gid(ip):
    a, b, c, d = (int(x) for x in ip.split('.'))
    return f'0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{d:02x}'


def make_sys(root, ports):
    """ports: {(dev, port): (state, [(gid text, type)])}"""
    for (dev, port), (state, gids) in ports.items():
        base = os.path.join(root, dev, 'ports', port)
        os.makedirs(os.path.join(base, 'gids'))
        os.makedirs(os.path.join(base, 'gid_attrs', 'types'))
        with open(os.path.join(base, 'state'), 'w') as f:
            f.write(state + '\n')
        for i, (g, typ) in enumerate(gids):
            with open(os.path.join(base, 'gids', str(i)), 'w') as f:
                f.write(g + '\n')
            with open(os.path.join(base, 'gid_attrs', 'types', str(i)), 'w') as f:
                f.write(typ + '\n')
    return root


V1, V2 = 'IB/RoCE v1', 'RoCE v2'
LINKLOCAL = 'fe80:0000:0000:0000:0000:0000:0000:0001'


class RailTests(unittest.TestCase):
    def test_two_active_rails_lowest_first(self):
        root = make_sys(tempfile.mkdtemp(), {
            ('rocep1s0f0', '1'): ('4: ACTIVE', [(LINKLOCAL, V2), (gid('10.2.0.3'), V1), (gid('10.2.0.3'), V2)]),
            ('roceP2p1s0f0', '1'): ('4: ACTIVE', [(gid('10.1.0.3'), V2)]),
            ('rocep1s0f1', '1'): ('1: DOWN', [(gid('10.3.0.3'), V2)])})
        self.assertEqual(F.rails(root), ('10.1.0.3', '10.2.0.3'))

    def test_one_rail_names_the_down_port(self):
        root = make_sys(tempfile.mkdtemp(), {
            ('a', '1'): ('4: ACTIVE', [(gid('10.1.0.3'), V2)]),
            ('b', '1'): ('1: DOWN', [(gid('10.2.0.3'), V2)])})
        with self.assertRaises(RuntimeError) as e:
            F.rails(root)
        self.assertIn('1 active rail address (10.1.0.3)', str(e.exception))
        self.assertIn('10.2.0.3 (b port 1: 1: DOWN)', str(e.exception))

    def test_none_and_too_many(self):
        with self.assertRaises(RuntimeError) as e:
            F.rails(tempfile.mkdtemp())
        self.assertIn('no RoCE v2 address', str(e.exception))
        root = make_sys(tempfile.mkdtemp(), {('a', '1'): ('4: ACTIVE', [(gid(f'10.{i + 1}.0.3'), V2) for i in range(3)])})
        with self.assertRaises(RuntimeError) as e:
            F.rails(root)
        self.assertIn('pass --ips and --ips2', str(e.exception))

    def test_gid_ipv4(self):
        self.assertEqual(F.gid_ipv4(gid('10.1.0.1')), '10.1.0.1')
        self.assertIsNone(F.gid_ipv4(LINKLOCAL))
        self.assertIsNone(F.gid_ipv4('nonsense'))


class OwnIndexTests(unittest.TestCase):
    def test_own_index(self):
        self.assertEqual(F.own_index(['192.0.2.1', '127.0.0.1']), 1)       # 192.0.2.1 is TEST-NET and is on no box
        self.assertEqual(F.own_index(['192.0.2.1', '192.0.2.2'], node_ip='192.0.2.2'), 1)
        with self.assertRaises(RuntimeError) as e:
            F.own_index(['192.0.2.1', '192.0.2.2'])
        self.assertIn('none of 192.0.2.1,192.0.2.2 is an address of this box', str(e.exception))
        with self.assertRaises(RuntimeError):
            F.own_index(['192.0.2.1'], node_ip='192.0.2.9')


if __name__ == '__main__':
    unittest.main()
