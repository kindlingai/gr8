#!/usr/bin/env python3
"""End to end: four supervisors on this machine, each `./gr8 glm53 up` with its own G6_HOME and model cache, the
stub engine (stubs/g6m4, which plays each rank with frontend/tests/fake_s2_engine.py), the real shim and a stub front
end. The boxes are 127.0.0.1-4 on the LAN and 127.0.1.x / 127.0.2.x on the rails, so it needs
Linux (every 127/8 address is local there).
    python3 -m unittest discover -s tools/launch -p 'test_e2e.py'"""
import asyncio
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'frontend'))
sys.path.insert(0, os.path.join(REPO, 'frontend', 'tests'))
from test_mentat import FakeDaemon  # noqa: E402
from test_modelset import make_packs  # noqa: E402
import modelset  # noqa: E402

TP = 4
LAN = [f'127.0.0.{i + 1}' for i in range(TP)]


def loopback_aliases():
    try:
        with socket.socket() as s:
            s.bind(('127.0.0.2', 0))
        return True
    except OSError:
        return False


def read(path):
    with open(path) as f:
        return f.read()


def running(pid):
    """Alive and not a zombie."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@unittest.skipUnless(loopback_aliases(), 'needs 127.0.0.2..4 (Linux)')
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix='gr8e2e-')
        self.packs = make_packs(os.path.join(self.t, 'packs'), tp=TP)
        self.port, self.api = free_port(), free_port()
        self.homes = []
        for i in range(TP):
            h = os.path.join(self.t, f'box{i}')
            os.makedirs(os.path.join(h, 'bin'))
            os.symlink(os.path.join(HERE, 'stubs', 'g6m4'), os.path.join(h, 'bin', 'g6m4'))
            self.homes.append(h)
        self.env = {**os.environ, 'PATH': os.path.join(HERE, 'stubs') + os.pathsep + os.environ['PATH'],
                    'GR8_REFUSAL_LINGER': '3'}   # three polls' worth
        for k in ('CAP', 'SESS', 'PEERS', 'G6_CLUSTER'):
            self.env.pop(k, None)

    def tearDown(self):
        for i in range(TP):
            self.gr8(i, 'down', timeout=60)
        shutil.rmtree(self.t, ignore_errors=True)

    def argv(self, i, *extra):
        where = ['--tp', str(TP), '--mentat', f'127.0.0.1:{self.daemons[i].port}'] if getattr(self, 'daemons', None) else \
            ['--peers', ','.join(LAN), '--node-ip', LAN[i], '--ips', ','.join(f'127.0.1.{r + 1}' for r in range(TP)),
             '--ips2', ','.join(f'127.0.2.{r + 1}' for r in range(TP))]
        return ['--no-config', '--no-check', *where,
                *(['--model', self.models[i]] if getattr(self, 'models', None) else
                  ['--model', self.packs, '--model-cache', os.path.join(self.homes[i], 'cache')]), '--g6-home', self.homes[i],
                '--control-port', str(self.port), '--port', str(self.api), '--py', sys.executable,
                '--front-bin', os.path.join(HERE, 'stubs', 'front'), '--join-timeout', '60', '--ready-tmo', '60', *extra]

    def gr8(self, i, cmd, *extra, timeout=120, env=None):
        args = self.argv(i, *extra) if cmd in ('up', 'show-config') else ['--g6-home', self.homes[i], *extra]
        return subprocess.run([sys.executable, os.path.join(HERE, 'launch.py'), cmd, *args], env=env or self.env,
                              capture_output=True, text=True, timeout=timeout)

    def up_all(self, extra=None, env=None):
        procs = [subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'up', *self.argv(i, *(extra or {}).get(i, []))],
                                  env=env or self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for i in range(TP)]
        out = []
        for p in procs:
            with p.stdout:
                out.append((p.wait(timeout=180), p.stdout.read()))
        return out

    def held_sets(self, held):
        """One in-place model set per box, holding only rank held[i]'s packs, with every rank's manifest and set.json."""
        setj, _ = modelset.describe(self.packs, TP)
        with open(os.path.join(self.packs, 'set.json'), 'w') as f:
            json.dump(setj, f)
        self.models = []
        for i in range(TP):
            d = shutil.copytree(self.packs, os.path.join(self.t, f'model{i}'))
            for r in range(TP):
                if r != held[i]:
                    os.unlink(os.path.join(d, f'rank{r}.pack'))
                    os.unlink(os.path.join(d, f'dflash.r{r}.pack'))
            self.models.append(d)

    def stub(self, i):
        with open(os.path.join(self.homes[i], 'out', 'stub.json')) as f:
            return json.load(f)

    def test_boot_serve_status_down(self):
        res = self.up_all()
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 0, f'box {i}:\n{out}')
        self.assertIn('serving: http://127.0.0.1', res[0][1])
        hs = [json.loads(read(os.path.join(h, 'out', 'handshake.json'))) for h in self.homes]
        self.assertEqual(len({x['epoch'] for x in hs}), 1)
        self.assertEqual(hs[2]['ips'], [f'127.0.1.{r + 1}' for r in range(TP)])
        set_id = hs[0]['set_id']
        for i in range(TP):
            a = self.stub(i)['argv']
            val = lambda f: a[a.index(f) + 1]   # noqa: E731
            rank_dir = os.path.join(self.homes[i], 'cache', 'sets', set_id, f'rank{i}')
            self.assertEqual((val('--rank'), val('--s2-epoch'), val('--set-id')), (str(i), str(hs[0]['epoch']), set_id))
            self.assertEqual(val('--manifest'), os.path.join(rank_dir, f'rank{i}.manifest.json'))
            self.assertEqual(val('--dflash-pack'), os.path.join(rank_dir, f'dflash.r{i}.pack'))
            self.assertEqual(val('--serve'), os.path.join(self.homes[i], 's2'))
            self.assertEqual(self.stub(i)['env']['G6_PF_OWN'], '4')
            self.assertEqual(os.listdir(os.path.join(self.homes[i], 'cache', 'sets', set_id)).count(f'rank{i}'), 1)
            self.assertEqual(sorted(n for n in os.listdir(os.path.join(self.homes[i], 'cache', 'sets', set_id))
                                    if n.startswith('rank')), [f'rank{i}'])
        # a request through the shim: req.<id> lands in the head's directory only (the head engine sends the bytes on)
        self.assertEqual(self.request(), 6)
        self.assertTrue(any(n.startswith('req.') for n in os.listdir(os.path.join(self.homes[0], 's2'))))
        for i in range(1, TP):
            self.assertFalse(any(n.startswith('req.') for n in os.listdir(os.path.join(self.homes[i], 's2'))), i)
        st = self.gr8(2, 'status')
        self.assertEqual(st.stdout.count(': serving, engine running'), TP, st.stdout)
        self.assertIn('answers (/health 200)', st.stdout)
        lg = self.gr8(1, 'logs', 'supervisor')
        self.assertIn('the head serves', lg.stdout)
        lg = self.gr8(1, 'logs', '0')
        self.assertEqual(lg.returncode, 0)
        lg = self.gr8(2, 'logs', 'shim')          # the shim runs on the head: a worker asks the head
        self.assertIn('g6s2shim on', lg.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.homes[2], 'out', 'shim.log')))
        lg = self.gr8(2, 'logs', 'all')
        self.assertIn('== shim (127.0.0.1)', lg.stdout)
        self.assertIn('== rank 3 (127.0.0.4)', lg.stdout)
        sc = self.gr8(1, 'show-config')
        self.assertIn(f'model set {json.loads(read(os.path.join(self.homes[0], "out", "handshake.json")))["set_id"]}', sc.stdout)
        self.assertIn('this box  rank 1 of 127.0.0.1', sc.stdout)
        dn = self.gr8(3, 'down', timeout=60)
        self.assertIn('rank 0 (127.0.0.1): stopping', dn.stdout)
        for i in range(TP):
            pid = int(read(os.path.join(self.homes[i], 'out', 'g6m4.pid')))
            for _ in range(200):
                if not running(pid):
                    break
                time.sleep(0.1)
            self.assertFalse(running(pid), f'box {i} engine {pid} runs')
        st = self.gr8(0, 'status')
        self.assertIn('no supervisor answers', st.stdout)

    def request(self):
        from fake_s2_engine import token_at
        from test_s2shim import Client, gen

        async def go():
            c = await Client(os.path.join(self.homes[0], 'run', 'engine.sock')).open()
            c.send(gen(1, prompt=[5, 6, 7], max_tokens=6))
            ids, last, _ = await c.wait(1)
            c.close()
            self.assertEqual(ids, [token_at([5, 6, 7], k) for k in range(1, 7)])
            return len(ids)
        return asyncio.run(go())

    def test_mismatch_stops_every_box(self):
        res = self.up_all(extra={2: ['--cap', '262144']})
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 4, f'box {i}:\n{out}')
            self.assertIn('handshake refused: option CAP', out)
            self.assertIn('rank 2 (127.0.0.3): 262144', out)
        for h in self.homes:
            self.assertFalse(os.path.exists(os.path.join(h, 'out', 'stub.json')))   # no engine started

    def v2(self):
        """A stand-in exlr8 v2 directory: the stub engine never reads it, so the manifests are enough."""
        from test_variant import manifest
        d = os.path.join(self.t, 'v2')
        for name, w in (('home', 3), ('down1', 2), ('up1', 4)):
            manifest(os.path.join(d, 'manifests', f'exlr8-manifest-{name}.json'), name, w)
        return ['--v2', d, '--v2-slices', d]

    def test_a_variant_mismatch_stops_every_box(self):
        v2 = self.v2()
        res = self.up_all(extra={i: v2 + (['--variant', 'down1'] if i == 2 else []) for i in range(TP)})
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 4, f'box {i}:\n{out}')
            self.assertIn('handshake refused: model set id', out)
            self.assertIn('variant down1 (', out)
            self.assertIn('variant home (', out)
        self.assertIn('about 0.6 GB of experts on the busiest rank at TP=4', res[2][1])

    def test_a_variant_boots_with_its_widths(self):
        v2 = self.v2()
        res = self.up_all(extra={i: v2 + ['--variant', 'up1'] for i in range(TP)})
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 0, f'box {i}:\n{out}')
        a = self.stub(3)['argv']
        self.assertEqual(a[a.index('--v2-widths') + 1], os.path.join(self.t, 'v2', 'manifests', 'exlr8-manifest-up1.json'))
        st = self.gr8(1, 'status')
        self.assertEqual(st.stdout.count('variant up1'), TP, st.stdout)

    def release(self):
        """A fake hub serving a synthetic exlr8 release at two commits (test_hf.py)."""
        import test_hf
        if not test_hf.HAVE:
            self.skipTest('needs numpy')
        root = os.path.join(self.t, 'release')
        test_hf.make_release(root)
        hub = test_hf.FakeHub({test_hf.RELEASE: {test_hf.C1: root, test_hf.C2: root}},
                              {test_hf.RELEASE: {'main': test_hf.C1, 'v0.4': test_hf.C2}})
        self.addCleanup(hub.close)
        env = {**self.env, 'GR8_HF_ENDPOINT': f'http://127.0.0.1:{hub.port}'}
        return f'https://huggingface.co/{test_hf.RELEASE}', env, test_hf

    def test_boot_from_a_release(self):
        url, env, T = self.release()
        res = self.up_all(extra={i: ['--model', url, '--base-packs', self.packs] for i in range(TP)}, env=env)
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 0, f'box {i}:\n{out}')
        rel = os.path.join(self.homes[2], 'cache', 'hf', 'fakeorg--glm-exlr8', T.C1)
        a = self.stub(2)['argv']
        val = lambda f: a[a.index(f) + 1]   # noqa: E731
        self.assertEqual((val('--v2'), val('--v2-slices'), val('--v2-widths')),
                         (os.path.join(rel, 'experts'), os.path.join(rel, 'index-slices'),
                          os.path.join(rel, 'manifests', 'exlr8-manifest-home.json')))
        self.assertEqual(os.listdir(os.path.join(rel, 'index-slices', 'tp4', 'k3')), ['rank2.safetensors'])
        self.assertEqual(len({self.stub(i)['argv'][self.stub(i)['argv'].index('--set-id') + 1] for i in range(TP)}), 1)
        self.assertIn(f'{T.RELEASE}@{T.C1[:12]}', res[0][1])

    def test_boxes_on_different_commits_refuse(self):
        url, env, T = self.release()
        res = self.up_all(extra={i: ['--model', url + ('@v0.4' if i == 2 else ''), '--base-packs', self.packs]
                                 for i in range(TP)}, env=env)
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 4, f'box {i}:\n{out}')
            self.assertIn('handshake refused: model set id', out)
            self.assertIn(f'{T.RELEASE}@{T.C2[:12]}', out)

    def test_a_missing_shard_fails_the_handshake(self):
        os.unlink(os.path.join(self.packs, 'dflash.r3.pack'))
        res = self.up_all()
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 4, f'box {i}:\n{out}')
            self.assertIn('1 of 4 boxes failed their checks', out)
            self.assertIn('rank 3 (127.0.0.4): model: ', out)

    def test_peers_must_list_each_box_at_the_rank_it_holds(self):
        self.held_sets([2, 0, 1, 3])
        res = self.up_all()
        for i, (rc, out) in enumerate(res):
            self.assertEqual(rc, 4, f'box {i}:\n{out}')
            self.assertIn('handshake refused: checks: 3 of 4 boxes failed their checks', out)
            self.assertIn('rank 0 (127.0.0.1): this box holds rank 2 of the model set, but it is rank 0 (127.0.0.1) in the '
                          'member list: list it at position 3 of --peers', out)
            self.assertNotIn('rank 3 (127.0.0.4):', out)

    def test_a_rank_that_dies_stops_the_cluster(self):
        res = self.up_all()
        self.assertTrue(all(rc == 0 for rc, _ in res), res)
        os.kill(int(read(os.path.join(self.homes[2], 'out', 'g6m4.pid'))), 9)
        pids = [int(read(os.path.join(h, 'out', 'g6m4.pid'))) for h in self.homes]
        for _ in range(300):   # a failed box says so before it has stopped its engine
            st = [json.loads(read(os.path.join(h, 'out', 'gr8.state')))['state'] for h in self.homes]
            if all(s == 'stopped' or s.startswith('failed') for s in st) and not any(running(p) for p in pids):
                break
            time.sleep(0.1)
        self.assertTrue(st[2].startswith('failed: engine exited'), st)
        self.assertTrue(st[0].startswith('failed: rank 2 (127.0.0.3) failed'), st)
        for i in (1, 3):    # the other workers fail too, so a container that restarts on failure rejoins
            self.assertTrue(st[i].startswith('failed: the head stopped the cluster: rank 2'), st)
        for h in self.homes:
            self.assertFalse(running(int(read(os.path.join(h, 'out', 'g6m4.pid')))), h)

    def test_mentat_mode(self):
        """Four fake mentat daemons, one per box, sharing one agent table: each supervisor registers, the boxes elect
        the lowest address, take their rails from the rdma-tagged addresses, and the head announces the API."""
        agents, lock = {}, threading.Lock()

        def tags(i):
            return {LAN[i]: ['lan'], f'127.0.1.{i + 1}': ['connectx', 'rdma'], f'127.0.2.{i + 1}': ['connectx', 'rdma']}

        def local(i):
            return {'node_id': f'n{i}', 'node_ip': LAN[i], 'addrs': list(tags(i)), 'addr_tags': tags(i), 'peers': {}}

        def group(_):
            with lock:
                rows = {k: {f: x for f, x in v.items() if f != 'link'} for k, v in agents.items()}
            return {**local(0), 'peers': {f'n{i}': {'node_ip': LAN[i], 'addr_tags': tags(i)} for i in range(1, TP)},
                    'groups': {'gr8-glm53': {'agents': rows}}}

        def on_link(h, alive, link):
            with lock:
                if not alive:
                    if agents[h['agent_id']]['link'] is link:     # mentat ignores the end of a replaced link
                        agents[h['agent_id']]['alive'] = False
                    return
                agents[h['agent_id']] = {'node_ip': h['node_ip'], 'node_id': f'n{LAN.index(h["node_ip"])}', 'alive': True,
                                         'machine': h['machine'], 'meta': h['meta'], 'services': h['services'], 'link': link}
        self.daemons = [FakeDaemon(group=group, local=lambda i=i: local(i), on_link=on_link) for i in range(TP)]
        # each box holds one rank, out of address order: box i (127.0.0.i+1) holds rank HELD[i]
        self.held_sets([2, 0, 3, 1])
        order = ['127.0.0.2', '127.0.0.4', '127.0.0.1', '127.0.0.3']
        try:
            res = self.up_all()
            for i, (rc, out) in enumerate(res):
                self.assertEqual(rc, 0, f'box {i}:\n{out}')
            self.assertIn(f'mentat group gr8-glm53: {",".join(order)}; this box is rank 2', res[0][1])
            hs = json.loads(read(os.path.join(self.homes[3], 'out', 'handshake.json')))
            self.assertEqual((hs['members'], hs['ips2'][2]), (order, '127.0.2.1'))
            a = self.stub(0)['argv']
            self.assertEqual((a[a.index('--rank') + 1], a[a.index('--manifest') + 1]),
                             ('2', os.path.join(self.models[0], 'rank2.manifest.json')))
            for _ in range(100):
                with lock:
                    states = [agents.get(f'gr8-glm53@gr8@{ip}', {}).get('meta', {}).get('gr8.state') for ip in LAN]
                if states == ['serving'] * TP:
                    break
                time.sleep(0.1)
            self.assertEqual(states, ['serving'] * TP)
            with lock:
                api = agents['gr8-glm53@gr8-api@127.0.0.2']
                me = agents['gr8-glm53@gr8@127.0.0.1']
            self.assertEqual((api['services']['openai']['port'], api['machine']['gpus']), (self.api, []))
            self.assertEqual((me['meta']['gr8.rank'], me['meta']['gr8.offers'], me['meta']['gr8.set']),
                             ('2', '2', hs['set_id']))
            self.gr8(1, 'down', timeout=60)
            for _ in range(100):
                with lock:
                    if not any(a['alive'] for a in agents.values()):
                        break
                time.sleep(0.1)
            with lock:
                self.assertFalse([k for k, a in agents.items() if a['alive']])
        finally:
            for d in self.daemons:
                d.close()


if __name__ == '__main__':
    unittest.main()
