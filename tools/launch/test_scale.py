#!/usr/bin/env python3
"""./gr8 glm53 scale end to end, as test_e2e.py runs the launch: four supervisors on this machine (127.0.0.1-4 on the
LAN, 127.0.1.x / 127.0.2.x on the rails), the stub engine (stubs/g6m4 with frontend/tests/fake_s2_engine.py), the real
shim and the stub front end, which holds requests as the real one does (/internal/hold).
    python3 -m unittest discover -s tools/launch -p 'test_scale.py'"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import supervisor as sv  # noqa: E402
from test_e2e import LAN, free_port, loopback_aliases, read, running  # noqa: E402
from test_modelset import make_release  # noqa: E402
from test_variant import manifest  # noqa: E402


def ips(n, rail):
    return ','.join(f'127.0.{rail}.{r + 1}' for r in range(n))


@unittest.skipUnless(loopback_aliases(), 'needs 127.0.0.1-4, 127.0.1.1-4 and 127.0.2.1-4 (macOS: see test_e2e.py)')
class Scale(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix='gr8scale-')
        self.rel = make_release(os.path.join(self.t, 'rel'))
        # the release's width profiles as test_e2e.py makes them, and bit260, the TP=3 default (models.py GLM53)
        for name, w in (('bit325', 3), ('bit260', 3), ('bit200', 2), ('bit390', 4)):
            p = manifest(os.path.join(self.rel, 'manifests', f'exlr8-manifest-{name}.json'), name, w)
            with open(p) as f:
                m = json.load(f)
            with open(p, 'w') as f:
                json.dump({**m, 'set_id': 'ab' * 32}, f)
        self.port, self.api = free_port(), free_port()
        self.homes = []
        for i in range(4):
            h = os.path.join(self.t, f'box{i}')
            os.makedirs(os.path.join(h, 'bin'))
            for binary in ('g6m4', 'g6m4-int4'):
                os.symlink(os.path.join(HERE, 'stubs', 'g6m4'), os.path.join(h, 'bin', binary))
            self.homes.append(h)
        self.env = {**os.environ, 'PATH': os.path.join(HERE, 'stubs') + os.pathsep + os.environ['PATH'],
                    'GR8_REFUSAL_LINGER': '3', 'GR8_MODEL': 'glm53'}
        for k in ('CAP', 'SESS', 'PEERS', 'GR8_CLUSTER'):
            self.env.pop(k, None)

    def tearDown(self):
        for i in range(4):
            self.gr8(i, 'down', timeout=60)
        shutil.rmtree(self.t, ignore_errors=True)

    extra = ()

    def argv(self, i, n):
        """Box i's up options as one of the first n boxes."""
        return ['--no-config', '--no-check', '--tp', str(n), '--peers', ','.join(LAN[:n]), '--node-ip', LAN[i],
                '--ips', ips(n, 1), '--ips2', ips(n, 2), '--model', self.rel, '--work-dir', self.homes[i],
                '--control-port', str(self.port), '--port', str(self.api), '--py', sys.executable,
                '--front-bin', os.path.join(HERE, 'stubs', 'front'), '--join-timeout', '60', '--ready-tmo', '60',
                *self.extra]

    def gr8(self, i, cmd, *extra, timeout=120):
        args = extra if cmd in ('up', 'scale') else ['--work-dir', self.homes[i], *extra]
        return subprocess.run([sys.executable, os.path.join(HERE, 'launch.py'), cmd, *args], env=self.env,
                              capture_output=True, text=True, timeout=timeout)

    def up(self, n):
        procs = [subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'up', *self.argv(i, n)],
                                  env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for i in range(n)]
        for i, p in enumerate(procs):
            with p.stdout:
                rc, out = p.wait(timeout=180), p.stdout.read()
            self.assertEqual(rc, 0, f'box {i}:\n{out}')

    def scale(self, i, n, *extra):
        """./gr8 glm53 scale from box i to the first n boxes, in the background: the Popen."""
        return subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'scale', '--work-dir', self.homes[i],
                                 '--tp', str(n), '--peers', ','.join(LAN[:n]), '--ips', ips(n, 1), '--ips2', ips(n, 2),
                                 *extra], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def finish(self, p, timeout=180):
        with p.stdout:
            return p.wait(timeout=timeout), p.stdout.read()

    def request(self, out, max_tokens=8):
        body = json.dumps({'prompt': [5, 6, 7], 'max_tokens': max_tokens}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{self.api}/v1/completions', body,
                                     {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as f:
                out.append((f.status, len(json.load(f)['ids'])))
        except urllib.error.HTTPError as e:
            out.append((e.code, 0))

    def holds(self):
        try:
            return [int(x) for x in read(os.path.join(self.homes[0], 'out', 'hold.log')).split()]
        except FileNotFoundError:
            return []

    def stub_tp(self, i):
        with open(os.path.join(self.homes[i], 'out', 'stub.json')) as f:
            a = json.load(f)['argv']
        return len(a[a.index('--ips') + 1].split(','))

    def wait_for(self, cond, what, s=60):
        for _ in range(int(s * 10)):
            if cond():
                return
            time.sleep(0.1)
        self.fail(f'no {what} within {s}s')

    def test_4_to_3_holds_a_request_through_the_switch(self):
        self.up(4)
        a, b = [], []
        ta = threading.Thread(target=self.request, args=(a, 300))   # runs a few seconds: the drain waits for it
        ta.start()
        self.wait_for(lambda: json.loads(urllib.request.urlopen(f'http://127.0.0.1:{self.api}/internal/hold').read())
                      ['running'] == 1, 'running request')
        p = self.scale(1, 3)
        self.wait_for(lambda: self.holds() and self.holds()[-1] > 0, 'hold')
        tb = threading.Thread(target=self.request, args=(b,))     # sent during the hold: it waits for the new engine
        tb.start()
        rc, out = self.finish(p)
        self.assertEqual(rc, 0, out)
        ta.join(60)
        tb.join(60)
        self.assertEqual((a, b), ([(200, 300)], [(200, 8)]))
        self.assertIn('serving TP=3 on 127.0.0.1,127.0.0.2,127.0.0.3', out)
        self.assertIn('SHARE_LIMIT=', out)
        self.assertEqual(self.holds()[-1], 0)
        self.assertEqual([self.stub_tp(i) for i in range(3)], [3, 3, 3])
        st = self.gr8(0, 'status')
        self.assertEqual(st.stdout.count(': serving, engine running'), 3, st.stdout)
        self.assertEqual(json.loads(read(os.path.join(self.homes[3], 'out', 'gr8.state')))['state'], 'stopped')
        sup = int(read(os.path.join(self.homes[3], 'out', 'supervisor.pid')))
        self.wait_for(lambda: not running(sup), 'the box that left to exit')
        hs = json.loads(read(os.path.join(self.homes[2], 'out', 'handshake.json')))
        self.assertEqual(hs['members'], LAN[:3])

    def test_3_to_4_with_a_box_that_joins(self):
        self.up(3)
        p = self.scale(0, 4)
        self.wait_for(lambda: os.path.exists(os.path.join(self.homes[0], 'out', 'scale', 'token')), 'the scale to start')
        j = subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'up', *self.argv(3, 4)], env=self.env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc, out = self.finish(p)
        jrc, jout = self.finish(j)
        self.assertEqual((rc, jrc), (0, 0), f'{out}\n---- joiner:\n{jout}')
        self.assertEqual([self.stub_tp(i) for i in range(4)], [4, 4, 4, 4])
        st = self.gr8(3, 'status')
        self.assertEqual(st.stdout.count(': serving, engine running'), 4, st.stdout)

    def test_a_drain_that_runs_out_changes_nothing(self):
        self.up(4)
        a = []
        ta = threading.Thread(target=self.request, args=(a, 1000))
        ta.start()
        self.wait_for(lambda: json.loads(urllib.request.urlopen(f'http://127.0.0.1:{self.api}/internal/hold').read())
                      ['running'] == 1, 'running request')
        rc, out = self.finish(self.scale(2, 3, '--drain-timeout', '1'))
        self.assertEqual(rc, 1, out)
        self.assertIn('1 requests still running', out)
        self.assertEqual(self.holds()[-1], 0)
        self.assertEqual([self.stub_tp(i) for i in range(4)], [4, 4, 4, 4])
        st = self.gr8(0, 'status')
        self.assertEqual(st.stdout.count(': serving, engine running'), 4, st.stdout)
        ta.join(120)
        self.assertEqual(a, [(200, 1000)])

    def test_a_new_width_that_fails_rolls_back(self):
        self.up(4)
        open(os.path.join(self.homes[0], 'fail.tp3'), 'w').close()   # the head's engine cannot boot TP=3
        b = []
        rc, out = self.finish(self.scale(1, 3))
        self.assertEqual(rc, 1, out)
        self.assertIn('scale failed', out)
        self.wait_for(lambda: self.gr8(0, 'status').stdout.count(': serving, engine running') == 4, 'TP=4 again', 120)
        self.assertEqual([self.stub_tp(i) for i in range(4)], [4, 4, 4, 4])
        self.assertEqual(self.holds()[-1], 0)
        self.request(b)
        self.assertEqual(b, [(200, 8)])

    def test_a_box_outside_the_cluster_is_refused(self):
        """With no scale running, a box that names itself in a longer member list is refused as before (here the boxes
        share loopback, which the head answers; on a LAN the head answers members only: refused 'member list')."""
        self.up(3)
        r = self.gr8(3, 'up', *self.argv(3, 4), timeout=120)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn('handshake refused: booted', r.stdout, r.stdout[-2000:])

    def test_4_to_3_hands_the_kv_store_to_the_new_ranks(self):
        """A persisted store: the boxes that stay boot with GR8_KV_REWIDTH naming the old ranks' kvman-peer, box 3's
        store folder is served until the scale is done, and the old width's index.prev goes then."""
        self.extra = ('--kv-store', 'ssd:store,size=1G,persist=on,keep=sessions')
        peer = os.environ.get('KVMAN_PEER')
        for h in self.homes:
            if peer:
                os.symlink(peer, os.path.join(h, 'bin', 'kvman-peer'))
        self.up(4)
        with open(os.path.join(self.homes[0], 'ready.delay'), 'w') as f:   # every worker's new engine starts first
            f.write('4')
        for i, h in enumerate(self.homes):   # what the old engines left at their quit
            os.makedirs(os.path.join(h, 'store', f'r{i}'), exist_ok=True)
            with open(os.path.join(h, 'store', f'r{i}', 'index'), 'w') as f:
                f.write(f'old index of rank {i}')
        rc, out = self.finish(self.scale(1, 3))
        self.assertEqual(rc, 0, out)
        token = os.path.join(self.homes[2], 'out', 'scale', 'token')
        with open(os.path.join(self.homes[2], 'out', 'stub.json')) as f:
            env = json.load(f)['env']
        self.assertEqual(env.get('GR8_KV_REWIDTH'), f'4,2,{token},' + ','.join(f'{a}:{self.port + 3}' for a in LAN),
                         read(os.path.join(self.homes[2], 'out', 'supervisor.log'))[-3000:])
        for i in range(3):
            self.assertEqual(sorted(os.listdir(os.path.join(self.homes[i], 'store', f'r{i}'))), ['index'])
        log3 = read(os.path.join(self.homes[3], 'out', 'supervisor.log'))
        self.assertIn('this box leaves', log3)
        self.assertIn('served to the new ranks' if peer else 'kvman-peer: the new ranks start', log3)
        self.assertEqual(sorted(os.listdir(os.path.join(self.homes[3], 'store', 'r3'))), ['index'])


class StorePrep(unittest.TestCase):
    """The KV store folders a scale hands the new engine (supervisor.py store_prep / store_undo / store_commit)."""

    def setUp(self):
        self.t = tempfile.mkdtemp(prefix='gr8store-')
        self.addCleanup(shutil.rmtree, self.t, True)

    def tree(self):
        return sorted(os.path.relpath(os.path.join(d, f), self.t) for d, _, fs in os.walk(self.t) for f in fs)

    def write(self, rel, text):
        os.makedirs(os.path.dirname(os.path.join(self.t, rel)), exist_ok=True)
        with open(os.path.join(self.t, rel), 'w') as f:
            f.write(text)

    def test_glm_rank_change_prep_and_undo(self):
        self.write('r2/index', 'i')
        self.write('r2/blocks', 'b')
        self.write('r1/index', 'stale')
        d = sv.store_prep(self.t, 2, 1, False)
        self.assertEqual(d, os.path.join(self.t, 'r1'))
        self.assertEqual(self.tree(), ['r1/blocks', 'r1/index', 'r1/index.prev'])
        self.write('r1/index', 'rewritten')
        sv.store_undo(self.t, 2, 1, False)
        self.assertEqual(self.tree(), ['r2/blocks', 'r2/index'])
        self.assertEqual(read(os.path.join(self.t, 'r2/index')), 'i')

    def test_flash_prep_and_commit(self):
        self.write('r0/flash-ckpt', 'b')
        self.write('r0/flash-ckpt.index', 'i')
        self.write('r0/flash-ckpt.tokens/7', 't')
        d = sv.store_prep(self.t, 0, 0, True)
        self.assertEqual(self.tree(), ['r0/flash-ckpt.index.prev', 'r0/flash-ckpt.prev', 'r0/flash-ckpt.tokens.prev/7'])
        self.write('r0/flash-ckpt', 'new')
        sv.store_commit(d, True)
        self.assertEqual(self.tree(), ['r0/flash-ckpt'])

    def test_a_box_that_leaves_keeps_its_folder(self):
        self.write('r3/flash-ckpt', 'b')
        d = sv.store_prep(self.t, 3, None, True)
        self.assertEqual((d, self.tree()), (os.path.join(self.t, 'r3'), ['r3/flash-ckpt.prev']))
        sv.store_undo(self.t, 3, None, True)
        self.assertEqual(self.tree(), ['r3/flash-ckpt'])

    def test_scale_argv_drops_mentat(self):
        self.assertEqual(sv.scale_argv(['--mentat', 'h:1', '--tp', '4', '--mentat=x', 'MENTAT=y', '--port', '1']),
                         ['--tp', '4', '--port', '1'])


if __name__ == '__main__':
    unittest.main()
