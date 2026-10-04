#!/usr/bin/env python3
"""Desk tests for rank_probe.py: no GPU, no network. Run: python3 -m unittest discover tools/health"""
import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_probe as rp

TH = rp.Thresholds()
IPS = ['10.0.0.1', '10.0.0.2']


class FakeHost(rp.Host):
    """files: {path: text}; dirs derived from files; smi: dict answering nvidia-smi."""
    def __init__(self, files=None, smi=None, list_out='GPU 0: NVIDIA GB10 (UUID: GPU-x)\n', list_rc=0):
        super().__init__('/')
        self.files = dict(files or {})
        self.smi = smi if smi is not None else {'clocks.sm': '2200', 'clocks.max.sm': '3003', 'utilization.gpu': '0',
                                                'power.draw': '12.5', 'temperature.gpu': '45',
                                                'clocks_throttle_reasons.active': '0x0000000000000001'}
        self.list_out, self.list_rc = list_out, list_rc
        self.calls = []
        self.slept = 0.0

    def read(self, p):
        return self.files.get(p)

    def listdir(self, p):
        p = p.rstrip('/') + '/'
        names = {f[len(p):].split('/')[0] for f in self.files if f.startswith(p)}
        return sorted(names)

    def run(self, argv, timeout=10.0):
        self.calls.append(argv)
        if argv[:2] == ['nvidia-smi', '-L']:
            return self.list_rc, self.list_out, ''
        if argv[0] == 'nvidia-smi':
            fields = argv[1].split('=')[1].split(',')
            if self.smi.get('hang'):
                return None, '', ''
            missing = [f for f in fields if f not in self.smi]
            if missing:
                return 2, '', f'Field "{missing[0]}" is not a valid field to query.'
            return 0, ', '.join(self.smi[f] for f in fields) + '\n', ''
        return 127, '', 'no such tool'

    def sleep(self, s):
        self.slept += s


def rails_files(ip, ib='roceP2p1s0f0', idx=3, state='4: ACTIVE', pcie=('32.0 GT/s PCIe', '4', '32.0 GT/s PCIe', '4')):
    gid = rp.roce_gid(ip)
    b = f'/sys/class/infiniband/{ib}/ports/1'
    f = {f'{b}/gids/0': 'fe80:0000:0000:0000:0000:0000:0000:0001', f'{b}/gid_attrs/types/0': 'IB/RoCE v1',
         f'{b}/gids/{idx}': gid, f'{b}/gid_attrs/types/{idx}': 'RoCE v2', f'{b}/gid_attrs/ndevs/{idx}': 'enp1s0f0np0',
         f'{b}/state': state, f'{b}/phys_state': '5: LinkUp', f'{b}/rate': '200 Gb/sec (4X HDR)'}
    d = f'/sys/class/infiniband/{ib}/device'
    for k, v in zip(('current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width'), pcie):
        f[f'{d}/{k}'] = v
    return f


def idle_files(extra=None):
    f = {'/proc/sys/kernel/hostname': 'node0', '/proc/meminfo': 'MemTotal: 125000000 kB\nMemAvailable: 118000000 kB\n',
         '/proc/1/comm': 'systemd', '/proc/1/stat': '1 (systemd) S 0', '/proc/2/comm': 'bash', '/proc/2/stat': '2 (bash) S 1',
         '/sys/devices/system/cpu/online': '0-19', '/proc/self/status': 'Name:\tpython3\nCpus_allowed_list:\t0-19\n',
         '/proc/stat': 'cpu  1 1 1 1 1 1 1 1\ncpu0 1 0 1 100 0 0 0 0\ncpu19 1 0 1 100 0 0 0 0\n'}
    f.update(rails_files(IPS[0]))
    f.update(extra or {})
    return f


class Parsers(unittest.TestCase):
    def test_cpu_list(self):
        self.assertEqual(rp.parse_cpu_list('0-3,8,10-11'), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(rp.parse_cpu_list(''), [])
        self.assertEqual(rp.parse_cpu_list(None), [])

    def test_throttle(self):
        self.assertEqual(rp.parse_throttle('0x0000000000000000'), (0, []))
        mask, names = rp.parse_throttle('0x0000000000000048')
        self.assertEqual(mask, 0x48)
        self.assertEqual(names, ['hw_slowdown', 'hw_thermal'])
        self.assertEqual(rp.parse_throttle('4')[1], ['sw_power_cap'])
        self.assertEqual(rp.parse_throttle('[N/A]'), (None, []))
        self.assertEqual(rp.parse_throttle(None), (None, []))

    def test_roce_gid(self):
        self.assertEqual(rp.roce_gid('192.168.1.7'), '0000:0000:0000:0000:0000:ffff:c0a8:0107')
        self.assertIsNone(rp.roce_gid('not an ip'))

    def test_ib_bw(self):
        out = ('---------------------------------------------------------------------------------------\n'
               ' #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n'
               ' 1048576    24000          112.31             111.87              0.013337\n'
               '---------------------------------------------------------------------------------------\n')
        self.assertEqual(rp.parse_ib_bw(out), 111.87)
        self.assertIsNone(rp.parse_ib_bw('Couldn\'t connect to 10.0.0.2:18700\n'))

    def test_gpuprobe_output(self):
        res, seen = rp.parse_gpuprobe('LOAD\nDONE\n{"device":"x","hbm_gbs":210.5,"h2d_gbs":22.1}\n')
        self.assertTrue(seen)
        self.assertEqual(res['hbm_gbs'], 210.5)
        self.assertEqual(rp.parse_gpuprobe('garbage\n'), (None, False))

    def test_ring_script(self):
        rails = [[{'ibdev': 'a0', 'gid_index': 3, 'ip': '10.0.0.1'}, {'ibdev': 'b0', 'gid_index': 2, 'ip': '10.1.0.1'}],
                 [{'ibdev': 'a1', 'gid_index': 3, 'ip': '10.0.0.2'}, {'ibdev': 'b1', 'gid_index': 2, 'ip': '10.1.0.2'}],
                 [{'ibdev': 'a2', 'gid_index': 3, 'ip': '10.0.0.3'}, {'ibdev': 'b2', 'gid_index': 2, 'ip': '10.1.0.3'}]]
        s = rp.ring_script(2, 3, rails)
        self.assertIn('-d a2 -x 3', s)
        self.assertIn('-d b2 -x 2', s)
        self.assertIn('10.0.0.1', s)               # rank 2 writes to rank 0 (wrap around)
        self.assertNotIn('10.1.0.1', s)            # the TCP control connection uses the first address on every rail (host firewalls)
        self.assertNotIn('10.0.0.2', s)
        self.assertIn('$T/0', s)                   # each rail's block goes to its own file, printed after `wait` (no interleaving)
        self.assertIn('-p 18700', s)
        self.assertIn('-p 18701', s)
        r = subprocess.run(['bash', '-n'], input=s, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_ring_output(self):
        txt = ('@@CLIENT rail=0 to=1 rc=0\n 1048576 100 110.0 109.5 0.01\n'
               '@@CLIENT rail=1 to=1 rc=1\nCouldn\'t connect\n')
        out, err = rp.parse_ring_output(txt)
        self.assertIsNone(err)
        self.assertEqual(out[0]['gbps'], 109.5)
        self.assertEqual(out[0]['tail'], '')
        self.assertIsNone(out[1]['gbps'])
        self.assertIn('connect', out[1]['tail'])
        self.assertEqual(rp.parse_ring_output('')[0], {})
        self.assertIn('no client output', rp.parse_ring_output('')[1])
        self.assertIn('twice', rp.parse_ring_output('@@CLIENT rail=0 to=1 rc=0\n x\n@@CLIENT rail=0 to=1 rc=0\n y\n')[1])
        out, _ = rp.parse_ring_output('@@CLIENT rail=0 to=1 rc=124\n 1048576 100 110.0 109.5 0.01\n')
        self.assertEqual((out[0]['rc'], out[0]['gbps']), (124, 109.5))
        self.assertEqual(rp.parse_ring_output('@@MISSING ib_write_bw\n')[0], {})
        self.assertIn('perftest', rp.parse_ring_output('@@MISSING ib_write_bw\n')[1])


class OneNode(unittest.TestCase):
    def test_smi_falls_back_to_event_reasons(self):
        h = FakeHost(smi={'clocks.sm': '2200', 'clocks.max.sm': '3000', 'utilization.gpu': '0', 'power.draw': '9',
                          'temperature.gpu': '40', 'clocks_event_reasons.active': '0x0000000000000004'})
        st = rp.smi_state(h)
        self.assertEqual(st['sm_mhz'], 2200.0)
        self.assertEqual(st['throttle'], ['sw_power_cap'])

    def test_smi_without_reasons_field(self):
        h = FakeHost(smi={'clocks.sm': '2200', 'clocks.max.sm': '3000', 'utilization.gpu': '0', 'power.draw': '[N/A]',
                          'temperature.gpu': '40'})
        st = rp.smi_state(h)
        self.assertEqual(st['sm_mhz'], 2200.0)
        self.assertIsNone(st['power_w'])
        self.assertEqual(st['throttle'], [])
        self.assertIsNone(st['throttle_mask'])

    def test_smi_timeout_is_one_call(self):
        h = FakeHost(smi={'hang': True})
        st = rp.smi_state(h)
        self.assertIn('timed out', st['error'])
        self.assertEqual(len(h.calls), 1)

    def test_idle_node_is_not_busy(self):
        why, _ = rp.busy_reasons(FakeHost(idle_files()), TH)
        self.assertEqual(why, [])

    def test_busy_by_process(self):
        why, _ = rp.busy_reasons(FakeHost(idle_files({'/proc/4242/comm': 'g6m4'})), TH)
        self.assertEqual(len(why), 1)
        self.assertIn('4242', why[0])

    def test_busy_by_memory_and_gpu(self):
        f = idle_files({'/proc/meminfo': 'MemAvailable: 4000000 kB\n'})
        h = FakeHost(f, smi={'clocks.sm': '2200', 'clocks.max.sm': '3000', 'utilization.gpu': '85', 'power.draw': '60',
                             'temperature.gpu': '60', 'clocks_throttle_reasons.active': '0x0'})
        why, _ = rp.busy_reasons(h, TH)
        self.assertEqual(len(why), 2)

    def test_wedge_timeout(self):
        w = rp.wedge_check(FakeHost(idle_files({'/proc/77/stat': '77 (nvidia-smi) D 1', '/proc/78/stat': '78 (apport) S 1'}),
                                    list_rc=None, list_out=''), TH)
        self.assertTrue(w['wedged'])
        self.assertIn('timed out', w['detail'])
        self.assertEqual(w['d_state'], ['77:nvidia-smi'])
        self.assertEqual(w['coredump_procs'], ['78:apport'])

    def test_wedge_ok_and_empty(self):
        self.assertFalse(rp.wedge_check(FakeHost(idle_files()), TH)['wedged'])
        self.assertTrue(rp.wedge_check(FakeHost(idle_files(), list_out='No devices were found\n'), TH)['wedged'])
        self.assertTrue(rp.wedge_check(FakeHost(idle_files(), list_rc=9, list_out=''), TH)['wedged'])

    def test_rails_state(self):
        r = rp.rails_state(FakeHost(idle_files()), IPS[:1])[0]
        self.assertEqual((r['ibdev'], r['gid_index'], r['netdev']), ('roceP2p1s0f0', 3, 'enp1s0f0np0'))
        self.assertIn('ACTIVE', r['state'])
        self.assertEqual(r['pcie']['cur_gbps'], r['pcie']['max_gbps'])
        self.assertEqual(r['pcie']['cur_width'], 4)

    def test_rails_unknown_ip_and_blank(self):
        rs = rp.rails_state(FakeHost(idle_files()), ['10.9.9.9', '', IPS[0]])
        self.assertEqual(len(rs), 2)               # the blank address is skipped
        self.assertIsNone(rs[0]['ibdev'])
        self.assertEqual(rs[1]['ibdev'], 'roceP2p1s0f0')
        self.assertEqual(rs[1]['rail'], 2)

    def test_rails_pcie_degraded(self):
        f = idle_files(rails_files(IPS[0], pcie=('8.0 GT/s PCIe', '4', '32.0 GT/s PCIe', '4')))
        r = rp.rails_state(FakeHost(f), IPS[:1])[0]
        self.assertLess(r['pcie']['cur_gbps'], r['pcie']['max_gbps'])

    def test_proxy_cpu_state(self):
        class H(FakeHost):
            def sleep(self, s):                  # CPU 19 is 90% busy during the sample, CPU 0 idle
                self.files['/proc/stat'] = 'cpu  1 1 1 1 1 1 1 1\ncpu0 1 0 1 200 0 0 0 0\ncpu19 91 0 1 110 0 0 0 0\n'
        st = rp.proxy_cpu_state(H(idle_files()), [19, 18, 0], 1.0)
        self.assertEqual(st['busy_pct'][19], 90.0)
        self.assertEqual(st['busy_pct'][0], 0.0)
        self.assertNotIn(18, st['busy_pct'])
        self.assertEqual(st['offline'], [])
        self.assertEqual(st['allowed_other'], [c for c in range(20) if c not in (19, 18, 0)])

    def test_proxy_cpu_offline(self):
        h = FakeHost(idle_files({'/sys/devices/system/cpu/online': '0-17',
                                 '/proc/self/status': 'Cpus_allowed_list:\t0-17\n'}))
        st = rp.proxy_cpu_state(h, [19, 18, 5], 0.1)
        self.assertEqual(st['offline'], [19, 18])
        self.assertEqual(st['not_allowed'], [19, 18])

    def test_local_probe_busy_measures_nothing(self):
        res = rp.local_probe(FakeHost(idle_files({'/proc/9/comm': 'g6m4'})), TH, 19, '/x/gpuprobe', IPS[:1])
        self.assertTrue(res['busy'])
        self.assertNotIn('gpu', res)
        self.assertNotIn('rails', res)

    def test_local_probe_wedged_has_no_gpu_section(self):
        h = FakeHost(idle_files(), list_rc=None, list_out='')
        res = rp.local_probe(h, TH, 19, '/x/gpuprobe', IPS[:1])
        self.assertTrue(res['wedge']['wedged'])
        self.assertNotIn('gpu', res)

    def test_local_probe_missing_tool(self):
        res = rp.local_probe(FakeHost(idle_files()), TH, 19, '/does/not/exist/gpuprobe', IPS[:1])
        self.assertIn('gpuprobe not found', res['gpu']['error'])
        self.assertIn('elapsed_s', res)


class GpuLoad(unittest.TestCase):
    """gpu_load against a stand-in gpuprobe script."""
    def tool(self, body):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', d]))
        p = os.path.join(d, 'gpuprobe')
        with open(p, 'w') as f:
            f.write('#!/bin/sh\n' + body)
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        return p

    class H(rp.Host):
        def __init__(self, smi_mhz, mask):
            super().__init__('/')
            self.mhz, self.mask = smi_mhz, mask

        def run(self, argv, timeout=10.0):
            return 0, f'{self.mhz}, 3000, 99, 40, 58, {self.mask}\n', ''

        def sleep(self, s):
            pass

    def test_healthy(self):
        t = self.tool('echo LOAD; sleep 0.3; echo DONE; echo \'{"device":"d","hbm_gbs":210.5,"h2d_gbs":22.0}\'\n')
        th = rp.Thresholds(load_seconds=0.2)
        g = rp.gpu_load(self.H(2200, '0x0'), th, t, None)
        self.assertEqual(g['hbm_gbs'], 210.5)
        self.assertEqual(g['clock_load_mhz'], 2200.0)
        self.assertEqual(g['throttle_mask'], 0)

    def test_throttled_clock_is_recorded(self):
        t = self.tool('echo LOAD; sleep 0.3; echo DONE; echo \'{"device":"d","hbm_gbs":100,"h2d_gbs":10}\'\n')
        g = rp.gpu_load(self.H(600, '0x40'), rp.Thresholds(load_seconds=0.2), t, None)
        self.assertEqual(g['clock_load_mhz'], 600.0)
        self.assertEqual(g['throttle'], ['hw_thermal'])

    def test_tool_error(self):
        t = self.tool('echo \'{"error":"cudaMalloc: out of memory"}\'; exit 2\n')
        g = rp.gpu_load(self.H(2200, '0x0'), rp.Thresholds(load_seconds=0.2), t, None)
        self.assertIn('out of memory', g['error'])


def good_rank(r, clock=2200.0, hbm=210.0, h2d=22.0, link=(111.0, 110.0), pcie_ok=True, **kw):
    rails = [{'rail': k, 'ip': f'10.{k}.0.{r}', 'ibdev': f'ib{k}', 'gid_index': 3, 'state': '4: ACTIVE', 'rate': '200',
              'pcie': {'cur_speed': '32.0', 'cur_width': 4 if pcie_ok else 1, 'max_speed': '32.0', 'max_width': 4,
                       'cur_gbps': 126.0 if pcie_ok else 31.5, 'max_gbps': 126.0}} for k in range(len(link))]
    loc = {'host': f'n{r}', 'wedge': {'wedged': False, 'detail': None, 'd_state': [], 'coredump_procs': []}, 'rails': rails,
           'proxy_cpu': {'cpus': [19], 'offline': [], 'not_allowed': [], 'busy_pct': {'19': 2.0}, 'allowed_other': [0, 1]},
           'gpu': {'hbm_gbs': hbm, 'h2d_gbs': h2d, 'clock_load_mhz': clock, 'throttle_mask': 0, 'throttle': []}}
    loc.update(kw)
    return {'rank': r, 'host': f'n{r}', 'local': loc,
            'link': {k: {'to': (r + 1) % 6, 'rc': 0, 'gbps': g, 'tail': ''} for k, g in enumerate(link)}}


def healthy_cluster(n=6):
    return [good_rank(r) for r in range(n)]


def codes(bad):
    return sorted((b['rank'], b['code']) for b in bad)


class Judge(unittest.TestCase):
    def test_healthy_cluster_is_admitted(self):
        self.assertEqual(rp.judge(healthy_cluster(), TH), [])

    def test_small_spread_is_fine(self):
        c = healthy_cluster()
        c[3] = good_rank(3, hbm=203.0, h2d=21.0, link=(104.0, 101.0))
        self.assertEqual(rp.judge(c, TH), [])

    def test_clock_cap_lowered(self):
        c = healthy_cluster()
        c[2] = good_rank(2, clock=600.0)
        c[2]['local']['gpu'].update(throttle_mask=0x4, throttle=['sw_power_cap'])
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(2, 'GPU_CLOCK_LOW')])
        self.assertIn('600', bad[0]['detail'])
        self.assertIn('sw_power_cap', bad[0]['detail'])

    def test_hw_throttle(self):
        c = healthy_cluster()
        c[1]['local']['gpu'].update(throttle_mask=0x80, throttle=['hw_power_brake'])
        self.assertEqual(codes(rp.judge(c, TH)), [(1, 'GPU_HW_THROTTLE')])

    def test_link_forced_down_to_13(self):
        c = healthy_cluster()
        c[4] = good_rank(4, link=(13.0, 110.0))
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(4, 'LINK_SLOW')])
        self.assertIn('rail 0', bad[0]['detail'])
        self.assertIn('13', bad[0]['detail'])

    def test_link_below_peers_but_above_floor(self):
        c = healthy_cluster()
        c[0] = good_rank(0, link=(85.0, 110.0))        # above 80 Gb/s floor, 23% under the median
        self.assertEqual(codes(rp.judge(c, TH)), [(0, 'LINK_SLOW')])

    def test_whole_cluster_slow_hits_the_floor(self):
        c = [good_rank(r, link=(13.0, 13.0)) for r in range(6)]
        self.assertEqual(len(rp.judge(c, TH)), 12)

    def test_port_down(self):
        c = healthy_cluster()
        c[5]['local']['rails'][1]['state'] = '1: DOWN'
        self.assertIn((5, 'LINK_DOWN'), codes(rp.judge(c, TH)))

    def test_pcie_degraded(self):
        c = healthy_cluster()
        c[0] = good_rank(0, pcie_ok=False)
        self.assertEqual({code for _, code in codes(rp.judge(c, TH))}, {'LINK_PCIE_DEGRADED'})

    def test_hbm_and_h2d_slow(self):
        c = healthy_cluster()
        c[3] = good_rank(3, hbm=150.0, h2d=12.0)
        self.assertEqual(codes(rp.judge(c, TH)), [(3, 'H2D_SLOW'), (3, 'HBM_SLOW')])

    def test_peer_rules_need_three_ranks(self):
        c = [good_rank(0), good_rank(1, hbm=150.0, link=(100.0, 100.0))]
        self.assertEqual(rp.judge(c, TH), [])           # two ranks: no median to compare with, floor still applies

    def test_wedged(self):
        c = healthy_cluster()
        c[1]['local']['wedge'] = {'wedged': True, 'detail': 'nvidia-smi -L timed out after 15 s', 'd_state': ['9:nvidia-smi'],
                                  'coredump_procs': []}
        del c[1]['local']['gpu']
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(1, 'GPU_WEDGED')])
        self.assertIn('power cycle', bad[0]['detail'])

    def test_busy_node_is_named_and_not_measured(self):
        c = healthy_cluster()
        c[2] = {'rank': 2, 'host': 'n2', 'local': {'busy': ['g6m4 process 5 is running'], 'host': 'n2'}}
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(2, 'NODE_BUSY')])

    def test_unreachable_node(self):
        c = healthy_cluster()
        c[0] = {'rank': 0, 'host': 'n0', 'error': 'probe did not run on n0: timed out'}
        self.assertEqual(codes(rp.judge(c, TH)), [(0, 'PROBE_FAILED')])

    def test_proxy_cpu(self):
        c = healthy_cluster()
        c[0]['local']['proxy_cpu']['offline'] = [19]
        c[1]['local']['proxy_cpu']['busy_pct'] = {'19': 70.0}
        self.assertEqual(codes(rp.judge(c, TH)), [(0, 'PROXY_CPU_OFFLINE'), (1, 'PROXY_CPU_BUSY')])

    def test_gpu_error_and_missing_clock(self):
        c = healthy_cluster()
        c[0]['local']['gpu'] = {'error': 'gpuprobe not found at /x'}
        c[1]['local']['gpu']['clock_load_mhz'] = None
        self.assertEqual(codes(rp.judge(c, TH)), [(0, 'PROBE_FAILED'), (1, 'PROBE_FAILED')])

    def test_link_without_result(self):
        c = healthy_cluster()
        c[2]['link'][1] = {'to': 3, 'rc': 1, 'gbps': None, 'tail': "Couldn't connect"}
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(2, 'PROBE_FAILED')])
        self.assertIn('connect', bad[0]['detail'])

    def test_unknown_rdma_device(self):
        c = healthy_cluster()
        c[2]['local']['rails'][0].update(ibdev=None, gid_index=None)
        self.assertEqual(codes(rp.judge(c, TH)), [(2, 'PROBE_FAILED')])

    def test_blank_link_result_refuses_every_rank(self):
        # review repro: a rank whose RDMA client printed nothing used to be admitted
        c = healthy_cluster()
        c[0]['link'] = {}
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(0, 'PROBE_FAILED'), (0, 'PROBE_FAILED')])     # one per rail
        self.assertIn('no result for rail 0', bad[0]['detail'])
        c = healthy_cluster()
        del c[3]['link']
        self.assertEqual({r for r, _ in codes(rp.judge(c, TH))}, {3})

    def test_one_rail_missing_refuses(self):
        c = healthy_cluster()
        del c[2]['link'][1]
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(2, 'PROBE_FAILED')])
        self.assertIn('rail 1', bad[0]['detail'])

    def test_nonzero_client_rc_refuses_even_with_a_bandwidth(self):
        c = healthy_cluster()
        c[1]['link'][0]['rc'] = 124                      # timeout killed the client after it printed a row
        bad = rp.judge(c, TH)
        self.assertEqual(codes(bad), [(1, 'PROBE_FAILED')])
        self.assertIn('rc 124', bad[0]['detail'])

    def test_link_rules_wait_for_the_ring_only_when_it_ran(self):
        c = healthy_cluster()
        c[4] = {'rank': 4, 'host': 'n4', 'local': {'busy': ['g6m4 process 5 is running']}}
        for rk in c:
            rk.pop('link', None)                          # the ring was skipped: no link noise on the other ranks
        self.assertEqual(codes(rp.judge(c, TH)), [(4, 'NODE_BUSY')])

    def test_string_rail_keys_from_a_saved_result_are_read(self):
        c = healthy_cluster()
        for rk in c:
            rk['link'] = {str(k): v for k, v in rk['link'].items()}
        self.assertEqual(rp.judge(c, TH), [])

    def test_tool_missing_link_error(self):
        c = healthy_cluster()
        for rk in c:
            rk['link'] = {}
            rk['link_error'] = 'ib_write_bw is not installed (package perftest)'
        self.assertEqual(len(rp.judge(c, TH)), 6)

    def test_several_faults_each_named(self):
        c = healthy_cluster()
        c[0] = good_rank(0, clock=500.0, link=(12.0, 110.0))
        self.assertEqual(codes(rp.judge(c, TH)), [(0, 'GPU_CLOCK_LOW'), (0, 'LINK_SLOW')])


class FakeSsh:
    def __init__(self, ranks, ring_gbps=None, hosts=None, local_fail=None):
        self.hosts = hosts or [f'n{r}' for r in range(len(ranks))]
        self.ranks, self.ring, self.local_fail = ranks, ring_gbps or {}, local_fail or {}
        self.scripts = {}

    def run(self, rank, argv, stdin=None, timeout=60):
        if argv[:2] == ['python3', '-']:
            if rank in self.local_fail:
                return self.local_fail[rank]
            return 0, 'noise\n' + json.dumps(self.ranks[rank]['local']) + '\n', ''
        self.scripts[rank] = stdin
        g = self.ring.get(rank, (111.0, 110.0))
        out = ''.join(f'@@CLIENT rail={k} to={(rank + 1) % len(self.ranks)} rc=0\n 1048576 100 {v + 1} {v} 0.01\n'
                      for k, v in enumerate(g))
        return 0, out, ''


def args(**kw):
    return argparse.Namespace(proxy_cpu=19, home='/opt/g6', **kw)


class Cluster(unittest.TestCase):
    def rail_ips(self, n=6):
        return [[f'10.0.0.{r}' for r in range(n)], [f'10.1.0.{r}' for r in range(n)]]

    def test_admitted(self):
        c = healthy_cluster()
        ranks = rp.probe_cluster(FakeSsh(c), 6, 'script', args(), TH, self.rail_ips())
        self.assertEqual(rp.judge(ranks, TH), [])
        self.assertEqual(len(ranks[0]['link']), 2)
        self.assertEqual(ranks[0]['link'][0]['gbps'], 111.0)

    def test_link_fault_in_ring_found(self):
        ssh = FakeSsh(healthy_cluster(), ring_gbps={3: (13.0, 110.0)})
        ranks = rp.probe_cluster(ssh, 6, 'script', args(), TH, self.rail_ips())
        self.assertEqual(codes(rp.judge(ranks, TH)), [(3, 'LINK_SLOW')])
        self.assertIn('10.0.0.4', ssh.scripts[3])                           # the ring script of rank 3 targets rank 4's rail IP

    def test_ring_skipped_when_a_node_is_busy(self):
        c = healthy_cluster()
        c[1] = {'rank': 1, 'host': 'n1', 'local': {'busy': ['g6m4 process 5 is running']}}
        ssh = FakeSsh(c)
        ranks = rp.probe_cluster(ssh, 6, 'script', args(), TH, self.rail_ips())
        self.assertEqual(ssh.scripts, {})
        self.assertEqual(codes(rp.judge(ranks, TH)), [(1, 'NODE_BUSY')])

    def test_blank_ring_output_end_to_end(self):
        class Blank(FakeSsh):
            def run(self, rank, argv, stdin=None, timeout=60):
                if argv[:2] == ['python3', '-']:
                    return super().run(rank, argv, stdin, timeout)
                return 0, '' if rank == 0 else super().run(rank, argv, stdin, timeout)[1], ''
        ranks = rp.probe_cluster(Blank(healthy_cluster()), 6, 'script', args(), TH, self.rail_ips())
        self.assertEqual({r for r, _ in codes(rp.judge(ranks, TH))}, {0})
        self.assertIn('no client output', ranks[0]['link_error'])

    def test_node_that_does_not_answer(self):
        ssh = FakeSsh(healthy_cluster(), local_fail={5: (None, '', 'timed out')})
        ranks = rp.probe_cluster(ssh, 6, 'script', args(), TH, self.rail_ips())
        self.assertEqual(codes(rp.judge(ranks, TH)), [(5, 'PROBE_FAILED')])

    def test_unreadable_output(self):
        ssh = FakeSsh(healthy_cluster(), local_fail={2: (0, 'Traceback...\n', '')})
        ranks = rp.probe_cluster(ssh, 6, 'script', args(), TH, self.rail_ips())
        self.assertEqual(codes(rp.judge(ranks, TH)), [(2, 'PROBE_FAILED')])

    def test_render(self):
        ranks = rp.probe_cluster(FakeSsh(healthy_cluster()), 6, 'script', args(), TH, self.rail_ips())
        bad = rp.judge(ranks, TH)
        res = {'admitted': not bad, 'elapsed_s': 1.0, 'ranks': ranks, 'refusals': bad}
        self.assertIn('ADMITTED', rp.render(res))
        ranks[0]['local']['gpu']['clock_load_mhz'] = 400.0
        bad = rp.judge(ranks, TH)
        res = {'admitted': False, 'elapsed_s': 1.0, 'ranks': ranks, 'refusals': bad}
        self.assertIn('rank 0 (n0): GPU_CLOCK_LOW', rp.render(res))


class Cli(unittest.TestCase):
    def test_help_and_threshold_flags(self):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(rp.__file__), 'rank_probe.py'), 'cluster', '--help'],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn('--min-clock-mhz', r.stdout)

    def test_cluster_address_count_checked(self):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(rp.__file__), 'rank_probe.py'), 'cluster', '--hosts',
                            'a,b', '--ips', '1.1.1.1', '--home', '/x', '--proxy-cpu', '19'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('one address per host', r.stderr)


if __name__ == '__main__':
    unittest.main()
