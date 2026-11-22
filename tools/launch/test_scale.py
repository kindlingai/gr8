#!/usr/bin/env python3
"""./gr8 glm53 scale end to end, as test_e2e.py runs the launch: four supervisors on this machine (127.0.0.1-4 on the
LAN, 127.0.1.x / 127.0.2.x on the rails), the stub engine (stubs/g6m4 with frontend/tests/fake_s2_engine.py), the real
shim and the stub front end, which holds requests as the real one does (/internal/hold).
    python3 -m unittest discover -s tools/launch -p 'test_scale.py'"""
import json
import os
import re
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
        procs = []
        for i in range(4):
            out = os.path.join(self.homes[i], 'out')
            if os.path.exists(os.path.join(out, 'supervisor.lock')) or os.path.exists(os.path.join(out, 'supervisor.pid')):
                procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'down',
                                               '--work-dir', self.homes[i]], env=self.env,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait(timeout=60)
        shutil.rmtree(self.t, ignore_errors=True)

    extra = ()

    def argv(self, i, n):
        """Box i's up options as one of the first n boxes."""
        return ['--no-config', '--no-check', '--tp', str(n), '--peers', ','.join(LAN[:n]), '--node-ip', LAN[i],
                '--ips', ips(n, 1), '--ips2', ips(n, 2), '--model', self.rel, '--work-dir', self.homes[i],
                '--control-port', str(self.port), '--port', str(self.api), '--py', sys.executable,
                '--front-bin', os.path.join(HERE, 'stubs', 'front'), '--join-timeout', '60', '--ready-tmo', '60',
                '--variant', 'bit260', *self.extra]   # bit260: the TP=4 default bit325 does not fit TP=3 (scale_fit)

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
        self.assertRegex(out, r'timeline: drained \+')
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

    def test_a_box_that_joins_later_boots_with_the_earlier_scales_options(self):
        """4 -> 3 with --sess 4, then 3 -> 4: box 3's own up names no --sess, and the scale that adds it hands it the
        first scale's (the handshake compares the engine's arguments)."""
        self.up(4)
        rc, out = self.finish(self.scale(1, 3, '--sess', '4'))
        self.assertEqual(rc, 0, out)
        self.wait_for(lambda: not running(int(read(os.path.join(self.homes[3], 'out', 'supervisor.pid')))),
                      'the box that left to exit')
        p = self.scale(0, 4)
        self.wait_for(lambda: f'waiting for {LAN[3]}' in read(os.path.join(self.homes[0], 'out', 'supervisor.log')),
                      'the second scale to start')
        j = subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'up', *self.argv(3, 4)], env=self.env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc, out = self.finish(p)
        jrc, jout = self.finish(j)
        self.assertEqual((rc, jrc), (0, 0), f'{out}\n---- joiner:\n{jout}')
        for i in range(4):
            with open(os.path.join(self.homes[i], 'out', 'stub.json')) as f:
                a = json.load(f)['argv']
            self.assertEqual(a[a.index('--s2-sessions') + 1], '4', i)

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

    def test_a_rollback_stops_engines_that_never_served_at_once(self):
        """With a store that keeps sessions an orderly quit gets 120 s, but a new engine that never served never reads
        s2/quit: the rollback stops it at once. The box that left waits for the head's rollback (retry), not refused."""
        self.extra = ('--kv-store', 'ssd:store,size=1G,persist=on,keep=sessions')
        self.up(4)
        open(os.path.join(self.homes[2], 'fail.tp3'), 'w').close()       # rank 2's new engine fails...
        for i in (0, 1):                                                  # ...while the others still load
            open(os.path.join(self.homes[i], 'ignore.quit.tp3'), 'w').close()
        sup3 = int(read(os.path.join(self.homes[3], 'out', 'supervisor.pid')))
        rc, out = self.finish(self.scale(1, 3))
        self.assertEqual(rc, 1, out)
        self.wait_for(lambda: self.gr8(0, 'status').stdout.count(': serving, engine running') == 4, 'TP=4 again', 60)
        b = []
        self.request(b)
        self.assertEqual(b, [(200, 8)])
        self.assertNotIn('handshake refused', read(os.path.join(self.homes[3], 'out', 'supervisor.log')))
        self.assertEqual(int(read(os.path.join(self.homes[3], 'out', 'supervisor.pid'))), sup3)
        self.assertTrue(running(sup3))

    def test_a_box_outside_the_cluster_is_refused(self):
        """With no scale running, a box that names itself in a longer member list is refused as before (here the boxes
        share loopback, which the head answers; on a LAN the head answers members only: refused 'member list')."""
        self.up(3)
        r = self.gr8(3, 'up', *self.argv(3, 4), timeout=120)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn('handshake refused: booted', r.stdout, r.stdout[-2000:])

    def test_4_to_3_hands_the_kv_store_to_the_new_ranks(self):
        """A persisted store: the boxes that stay boot with GR8_KV_REWIDTH naming the old ranks' kvman-peer, box 3's
        store folder is served until the scale is done, and the old width's index.prev goes then. GR8_LOAD_COMPACT
        becomes the new width's directory on every box; a box that cannot write the copies says so and boots anyway."""
        self.extra = ('--kv-store', 'ssd:store,size=1G,persist=on,keep=sessions',
                      '--extra-env', f'GR8_LOAD_COMPACT={self.t}/compact')
        peer = os.environ.get('KVMAN_PEER')
        for h in self.homes:
            if peer:
                os.symlink(peer, os.path.join(h, 'bin', 'kvman-peer'))
            with open(os.path.join(h, 'bin', 'g6load'), 'w') as f:   # a CI checkout builds the real one
                f.write('#!/bin/sh\necho "stub g6load: no compact copies here" >&2\nexit 1\n')
            os.chmod(os.path.join(h, 'bin', 'g6load'), 0o755)
        self.up(4)
        with open(os.path.join(self.homes[0], 'ready.delay'), 'w') as f:   # every worker's new engine starts first
            f.write('4')
        for i, h in enumerate(self.homes):   # what the old engines left at their quit
            os.makedirs(os.path.join(h, 'store', f'r{i}'), exist_ok=True)
            with open(os.path.join(h, 'store', f'r{i}', 'index'), 'w') as f:
                f.write(f'old index of rank {i}')
        def set_id():
            with open(os.path.join(self.homes[2], 'out', 'stub.json')) as f:
                a = json.load(f)['argv']
            return a[a.index('--set-id') + 1]
        old_set = set_id()
        rc, out = self.finish(self.scale(1, 3))
        self.assertEqual(rc, 0, out)
        self.assertNotEqual(set_id(), old_set)   # the set id names the width: the engine checks the old store by old_set
        token = os.path.join(self.homes[2], 'out', 'scale', 'token')
        with open(os.path.join(self.homes[2], 'out', 'stub.json')) as f:
            env = json.load(f)['env']
        self.assertEqual(env.get('GR8_KV_REWIDTH'),
                         f'4,2,{token},{old_set},' + ','.join(f'{a}:{self.port + 3}' for a in LAN),
                         read(os.path.join(self.homes[2], 'out', 'supervisor.log'))[-3000:])
        for i in range(3):
            self.assertEqual(sorted(os.listdir(os.path.join(self.homes[i], 'store', f'r{i}'))), ['index'])
        log3 = read(os.path.join(self.homes[3], 'out', 'supervisor.log'))
        # the scale's boot relays the engine's store rewrite to the supervisor log, whatever the model's stage_relay
        self.assertRegex(read(os.path.join(self.homes[2], 'out', 'supervisor.log')),
                         r'\d+s rank 2: flash ckpt rewidth carried 0 entries from TP=4')
        self.assertIn('this box leaves', log3)
        self.assertIn('served to the new ranks' if peer else 'kvman-peer: the new ranks start', log3)
        self.assertEqual(sorted(os.listdir(os.path.join(self.homes[3], 'store', 'r3'))), ['index'])
        for i in range(3):
            with open(os.path.join(self.homes[i], 'out', 'stub.json')) as f:
                self.assertRegex(json.load(f)['env'].get('GR8_LOAD_COMPACT', ''),
                                 '^' + re.escape(f'{self.t}/compact.tp3-') + '[0-9a-f]{8}$')
            self.assertIn('warning: no compact copies for the new width (g6load compact: stub g6load',
                          read(os.path.join(self.homes[i], 'out', 'supervisor.log')))

    def pool_argv(self, i):
        """Box i's up options in a pool of the four boxes, three of them serving."""
        a = self.argv(i, 3)
        k = a.index('--peers')
        return a[:k] + a[k + 2:] + ['--pool', ','.join(LAN)]

    def pool_up(self):
        procs = [subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'up', *self.pool_argv(i)],
                                  env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for i in range(4)]
        for i, p in enumerate(procs):
            rc, out = self.finish(p)
            self.assertEqual(rc, 0, f'box {i}:\n{out}')

    def pool_scale(self, n):
        p = subprocess.Popen([sys.executable, os.path.join(HERE, 'launch.py'), 'scale', '--work-dir', self.homes[0],
                              '--tp', str(n), '--ips', ips(n, 1), '--ips2', ips(n, 2)],
                             env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc, out = self.finish(p)
        self.assertEqual(rc, 0, out)

    def state(self, i):
        return json.loads(read(os.path.join(self.homes[i], 'out', 'gr8.state')))['state']

    def test_a_pool_scales_without_a_new_up_and_keeps_the_width(self):
        """--pool of four with --tp 3: box 3 stands by (healthy), a scale to 4 takes it in and one back to 3 leaves it
        standing by, the same supervisor throughout. The head keeps the last width for its next up."""
        self.pool_up()
        self.assertEqual(self.state(3), 'standby')
        self.assertEqual([self.stub_tp(i) for i in range(3)], [3, 3, 3])
        self.assertEqual(self.gr8(3, 'health').returncode, 0)
        sup3 = int(read(os.path.join(self.homes[3], 'out', 'supervisor.pid')))
        self.pool_scale(4)
        self.wait_for(lambda: self.gr8(0, 'status').stdout.count(': serving, engine running') == 4, 'TP=4')
        self.assertEqual([self.stub_tp(i) for i in range(4)], [4, 4, 4, 4])
        self.pool_scale(3)
        self.wait_for(lambda: self.state(3) == 'standby', 'box 3 to stand by again')
        self.assertEqual(int(read(os.path.join(self.homes[3], 'out', 'supervisor.pid'))), sup3)
        self.assertTrue(running(sup3))
        self.assertEqual([self.stub_tp(i) for i in range(3)], [3, 3, 3])
        self.pool_scale(4)
        self.wait_for(lambda: self.gr8(0, 'status').stdout.count(': serving, engine running') == 4, 'TP=4 again')
        for i in range(4):
            self.gr8(i, 'down', timeout=180)
        self.assertTrue(os.path.exists(os.path.join(self.homes[0], 'out', 'scale.json')))
        os.unlink(os.path.join(self.homes[3], 'out', 'stub.json'))
        self.pool_up()   # the configured --tp 3; the head boots the kept TP=4 and the others take it
        self.wait_for(lambda: self.gr8(0, 'status').stdout.count(': serving, engine running') == 4, 'TP=4 after up')
        self.assertEqual([self.stub_tp(i) for i in range(4)], [4, 4, 4, 4])
        b = []
        self.request(b)
        self.assertEqual(b, [(200, 8)])

    def test_the_head_is_healthy_while_it_holds_requests(self):
        self.up(4)
        with open(os.path.join(self.homes[0], 'ready.delay'), 'w') as f:   # the new width loads for 4 s
            f.write('4')
        p = self.scale(1, 3)
        self.wait_for(lambda: self.holds() and self.holds()[-1] > 0 and self.state(0) != 'serving', 'the switch')
        r = self.gr8(0, 'health')
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn('holds requests through a scale', r.stdout)
        rc, out = self.finish(p)
        self.assertEqual(rc, 0, out)


GLM = {'GR8_MODEL': 'glm53'}


class ScalePlan(unittest.TestCase):
    """What scale_begin checks and pins before a scale starts (supervisor.py scale_pins, scale_fit, on_rpc)."""

    def boot(self, n, *extra):
        return sv.opt.parse(['--no-config', '--tp', str(n), '--peers', ','.join(LAN[:n]), *extra], GLM)

    def rescale(self, o, n, variant):
        return sv.opt.parse(sv.scale_argv(o.argv) + ['--tp', str(n), '--peers', ','.join(LAN[:n])] +
                            sv.scale_pins(o, variant, {}), GLM)

    def test_the_kv_format_of_a_tp_default_is_kept(self):
        """GLM-5.3 TP=4 resolves int4 KV by default (models.py tp_defaults), TP=3 fp8: the scale keeps int4 both ways,
        so the engine and its KV row do not change and the store carries."""
        v = sv.modelset.Variant('bit260')
        o4 = self.boot(4, '--variant', 'bit260')
        self.assertEqual((o4.get('KV_FORMAT'), sv.engine_bin(o4)), ('int4', 'g6m4-int4'))
        o3 = self.rescale(o4, 3, v)
        sv.opt.validate(o3)
        self.assertEqual((o3.get('KV_FORMAT'), sv.engine_bin(o3)), ('int4', 'g6m4-int4'))
        back = self.rescale(o3, 4, v)
        self.assertEqual((back.get('KV_FORMAT'), sv.engine_bin(back)), ('int4', 'g6m4-int4'))
        o4 = self.rescale(self.boot(3), 4, v)   # an fp8 TP=3 boot stays fp8 at TP=4, where the default is int4
        self.assertEqual((o4.get('KV_FORMAT'), sv.engine_bin(o4)), ('fp8', 'g6m4'))

    def test_a_variant_that_does_not_fit_the_new_width_is_refused(self):
        V = sv.modelset.Variant
        why = sv.scale_fit(sv.models.GLM53, V('bit325'), '3')
        self.assertIn('--variant bit325 does not fit a rank at TP=3', why)
        self.assertIn('--variant bit260', why)
        self.assertEqual(sv.scale_fit(sv.models.GLM53, V('bit260'), '3'), '')
        self.assertEqual(sv.scale_fit(sv.models.GLM53, V('bit260'), '4'), '')
        self.assertEqual(sv.scale_fit(sv.models.GLM53F, V('k2'), '1'), '')
        self.assertEqual(sv.scale_fit(sv.models.GLM53F, V('k2'), '2'), '')
        self.assertIn('--variant k2', sv.scale_fit(sv.models.GLM53F, V('k4'), '1'))
        t = tempfile.mkdtemp(prefix='gr8fit-')
        self.addCleanup(shutil.rmtree, t, True)
        for name, w in (('bit260', 3), ('bit225', 2), ('mine', 4)):   # a manifest path: sized against the defaults
            manifest(os.path.join(t, f'exlr8-manifest-{name}.json'), name, w)
        path = lambda n: os.path.join(t, f'exlr8-manifest-{n}.json')   # noqa: E731
        self.assertIn('GB of experts', sv.scale_fit(sv.models.GLM53, V('mine', path('mine')), '3'))
        self.assertEqual(sv.scale_fit(sv.models.GLM53, V('b', path('bit225')), '3'), '')

    def test_after_a_scale_another_member_list_is_refused_not_retried(self):
        """A box that hellos with another member list waits (retry) only while a scale runs; after it, the handshake
        refuses it, so a box with restart: on-failure does not loop."""
        class Head:
            def rpc(self, m, peer):
                return {'t': 'refused', 'field': 'members', 'values': {}, 'exit': 4, 'detail': 'member list'}
        sup = sv.Supervisor.__new__(sv.Supervisor)
        sup.allowed = lambda peer: True
        sup.boot_attempt, sup.rank, sup.members, sup.head, sup.state = 0, 0, LAN[:3], Head(), 'serving'
        sup.pool = []
        sup.head_ready, sup.stop_ev = threading.Event(), threading.Event()
        sup.head_ready.set()
        hello = {'t': 'hello', 'members': LAN[:4], 'attempt': 0}
        for phase, closed, want in (('switch', False, 'retry'), ('done', True, 'refused'), ('failed', True, 'refused'),
                                    ('failed', False, 'retry')):   # failed, not closed: the head rolls back
            sup.scale = {'phase': phase, 'members': LAN[:3], 'old_members': LAN[:4], 'msg': None, 'ready': {},
                         **({'closed': True} if closed else {})}
            self.assertEqual(sup.on_rpc(dict(hello), LAN[3])['t'], want, phase)
        sup.pool, sup.scale_words, sup.layout_text, sup.scale = LAN[:4], ['--tp', '3'], None, None
        self.assertEqual(sup.on_rpc(dict(hello), LAN[3]),   # a pool: a box that counted another list takes this one
                         {'t': 'adopt', 'argv': ['--tp', '3'], 'members': LAN[:3], 'layout': None})
        self.assertEqual(sup.on_rpc({'t': 'pool'}, LAN[3])['t'], 'adopt')


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

    def test_scale_argv_drops_mentat_and_the_old_rails(self):
        self.assertEqual(sv.scale_argv(['--mentat', 'h:1', '--tp', '4', '--mentat=x', 'MENTAT=y', '--port', '1',
                                        '--ips', 'a,b', '--ips2=c,d', 'IPS=e', '--node-ip', '10.0.0.1']),
                         ['--tp', '4', '--port', '1', '--node-ip', '10.0.0.1'])


if __name__ == '__main__':
    unittest.main()
