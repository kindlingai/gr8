#!/usr/bin/env python3
"""--fetch model-loader against a fake model-loader (fake_model_loader.py) holding the synthetic exlr8 release of
test_hf.py, and, when MODEL_LOADER_BIN names a model-loader binary, against the real one with a fake hub as its
origin. Needs numpy (to write the release) and bin/g6load, as test_hf.py does.
    python3 -m unittest discover -s tools/launch"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake_model_loader  # noqa: E402
import mlsource  # noqa: E402
import modelset  # noqa: E402
import options as opt  # noqa: E402
import supervisor as S  # noqa: E402
import test_hf as T  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'
NAME = T.RELEASE.split('/')[1]          # the catalog name the launcher takes by default
URL = f'https://huggingface.co/{T.RELEASE}'


@unittest.skipUnless(T.HAVE, 'needs numpy to write the synthetic release')
class ModelLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.mkdtemp(prefix='gr8ml-')
        cls.rel = os.path.join(cls.t, 'release')
        T.make_release(cls.rel)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.t, ignore_errors=True)

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='gr8mls-', dir=self.t)
        self.store, self.home = os.path.join(self.d, 'store'), os.path.join(self.d, 'home')
        self.ml = fake_model_loader.FakeModelLoader(self.store, {NAME: {'main': (self.rel, T.C1)}})
        self.addCleanup(self.ml.close)
        self.logs = []
        # no hub at all: any HF fetch would fail
        self.env = mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': 'http://127.0.0.1:9'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def opts(self, *extra, tp=4):
        o = opt.parse(['--no-config', '--peers', ','.join(PEERS.split(',')[:tp]), '--model', URL, '--model-cache',
                       self.store, '--fetch', 'model-loader', '--g6-home', self.home, '--dflash', '0', *extra], {})
        opt.validate(o)
        return o

    def resolve(self, *extra, tp=4):
        o = self.opts(*extra, tp=tp)
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertEqual(err, '')
        return o, ms, v

    def test_options(self):
        self.assertEqual(opt.parse(['--no-config'], {}).get('FETCH'), 'legacy')
        with self.assertRaises(opt.OptionError):
            opt.validate(opt.parse(['--no-config', '--peers', PEERS, '--fetch', 'nas'], {}))
        for bad in (['--drafter', 'https://huggingface.co/incoai/GLM-5.3-DFlash2'], ['--steering-profile', 'hf:default']):
            with self.assertRaises(opt.OptionError) as e:
                self.opts(*bad)
            self.assertIn('--fetch legacy', str(e.exception))

    def test_one_rank_through_model_loader(self):
        o, ms, v = self.resolve()
        rel = ms.release
        self.assertIsInstance(rel, mlsource.Release)
        self.assertEqual((rel.commit, rel.id), (T.C1, f'{T.RELEASE}@{T.C1}'))
        self.assertEqual(rel.dir, os.path.join(self.store, 'models', NAME, 'main'))
        self.assertEqual((o.get('V2_DENSE'), o.get('V2'), o.get('V2_SLICES')), (rel.dir, rel.v2, rel.slices))
        self.assertEqual(ms.tokenizer, rel.dir)
        self.assertEqual(v.path, os.path.join(rel.dir, 'manifests', 'exlr8-manifest-home.json'))
        self.assertTrue(rel.check(4, 1, v))
        rel.fetch_rank(4, 1, v)
        self.assertEqual(rel.check(4, 1, v), [])
        name, body = self.ml.puts[-1]
        self.assertEqual((name, body['model']), ('gr8-glm53', NAME))
        self.assertNotIn('revision', self.ml.puts[0][1])          # the URL names none: the catalog's only one
        self.assertEqual(body['revision'], 'main')                # then the one model-loader resolved it to
        inc = set(body['include'])
        self.assertIn('index-slices/tp4/k3/rank1.safetensors', inc)
        self.assertNotIn('index-slices/tp4/k3/rank0.safetensors', inc)
        self.assertIn('dense/exlr8-vocab.safetensors', inc)
        self.assertEqual({r['path'] for r in body['ranges']},
                         {f'experts/exlr8-layer-00{l}.k3.safetensors' for l in T.LAYERS})   # home: K3 only
        T.same_halves(self.rel, rel.v2, 3, 4, 1, lambda p: 3)
        self.assertFalse(os.path.exists(os.path.join(rel.v2, 'exlr8-layer-003.k2.safetensors')))
        self.assertTrue(any('verified the' in m for m in self.logs), self.logs)
        # the store is model-loader's: gr8 wrote nothing in it but what the fake wrote
        self.assertFalse([p for p in self.ml.files(self.store) if '.fetch' in p or 'complete' in p])

    def test_a_second_boot_reads_the_store_and_fetches_nothing(self):
        o, ms, v = self.resolve()
        ms.release.fetch_rank(4, 2, v)
        n = self.ml.copied
        o, ms, v = self.resolve()
        puts, said = len(self.ml.puts), len(self.logs)
        ms.release.fetch_rank(4, 2, v)
        self.assertEqual(self.ml.copied, n)                       # nothing new
        self.assertGreater(len(self.ml.puts), puts)               # the request is put again: it pins the pieces
        self.assertFalse(any('verified the' in m for m in self.logs[said:]))   # the receipt stands

    def test_a_restart_keeps_the_last_pieces_pinned(self):
        o, ms, v = self.resolve()
        ms.release.fetch_rank(4, 0, v)
        last = self.ml.puts[-1][1]
        self.resolve()
        first = self.ml.puts[-1][1]                               # the meta request of the next boot
        self.assertEqual({r['path'] for r in first['ranges']}, {r['path'] for r in last['ranges']})
        self.assertTrue(set(last['include']) <= set(first['include']))

    def test_a_state_from_an_older_model_loader(self):
        """segments_done, without held: the state a model-loader from before small segments writes."""
        self.ml.legacy_state = True
        o, ms, v = self.resolve()
        ms.release.fetch_rank(4, 1, v)
        with open(os.path.join(self.store, 'state', NAME, 'main.json')) as f:
            self.assertTrue(all('held' not in x for x in json.load(f)['files']))
        self.assertEqual(ms.release.check(4, 1, v), [])
        self.ml.drop(NAME, 'main', 'experts/exlr8-layer-003.k3.safetensors')
        self.assertTrue(ms.release.check(4, 1, v))

    def test_held_runs(self):
        fm = {'size': 10 * 64, 'segment_size': 64}
        V = mlsource.Store.verified
        st = {'complete': False, 'held': [[0, 2], [2, 4], [7, 9]]}
        self.assertTrue(V(fm, st, 0, 256))                        # runs that touch join
        self.assertTrue(V(fm, st, 7 * 64 + 3, 100))
        self.assertFalse(V(fm, st, 3 * 64, 65))                   # crosses into segment 4
        self.assertFalse(V(fm, st, 9 * 64, 1))
        self.assertFalse(V(fm, st, 0, 0))
        self.assertFalse(V(fm, {'complete': False, 'held': [[0, 10]]}, 0, 64))   # all held: whole check pending
        self.assertTrue(V(fm, {'complete': True, 'held': [[0, 10]]}, 0, 64))
        self.assertFalse(V(fm, {'complete': True, 'whole_failed': True, 'held': [[0, 10]]}, 0, 64))
        self.assertTrue(V(fm, {'complete': False, 'segments_done': [True, True] + [False] * 8}, 64, 64))

    def test_the_store_losing_a_piece_fails_the_check(self):
        o, ms, v = self.resolve()
        ms.release.fetch_rank(4, 3, v)
        self.ml.drop(NAME, 'main', 'experts/exlr8-layer-004.k3.safetensors')
        bad = ms.release.check(4, 3, v)
        self.assertEqual(len(bad), 1)
        self.assertIn('exlr8-layer-004.k3', bad[0])

    def test_a_bad_half_fails_the_boot(self):
        o, ms, v = self.resolve()
        rel = ms.release
        rel.fetch_rank(4, 1, v)
        halves, _ = rel.plan(4, 1, v)
        path, rs = sorted(halves.items())[0]
        os.unlink(rel.receipt(4, 1, v))
        self.ml.corrupt(NAME, 'main', path, rs[0][0] + 5)
        with self.assertRaises(modelset.ModelError) as e:
            rel.fetch_rank(4, 1, v)
        self.assertIn('fail their xxhash64', str(e.exception))
        self.assertTrue(rel.check(4, 1, v))

    def test_a_revision_and_a_catalog_name(self):
        self.ml.catalog['glm-x'] = {'v0.4': (self.rel, T.C2), 'main': (self.rel, T.C1)}
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', URL + '@v0.4', '--model-cache', self.store,
                       '--fetch', 'model-loader', '--model-loader-model', 'glm-x', '--g6-home', self.home,
                       '--dflash', '0'], {})
        opt.validate(o)
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertEqual(err, '')
        self.assertEqual((ms.release.revision, ms.release.commit), ('v0.4', T.C2))
        self.assertEqual(self.ml.puts[-1][1]['revision'], 'v0.4')
        o.values['MODEL'] = URL + '/tree/' + T.C1                   # a commit model-loader's revision is not
        self.ml.catalog['glm-x'][T.C1] = (self.rel, T.C2)
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertIn(f'is commit {T.C2}', err)

    def test_model_loader_down_is_named(self):
        self.ml.close()
        os.unlink(self.ml.sock)
        o = self.opts()
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertIn('model-loader at unix:', err)
        self.assertIn('is it running?', err)

    def test_the_local_api_token(self):
        self.ml.token = 's3cret'
        ms, v, err = S.resolve_model(self.opts(), log=self.logs.append)
        self.assertIn('MODEL_LOADER_LOCAL_TOKEN is unset', err)
        with mock.patch.dict(os.environ, {'MODEL_LOADER_LOCAL_TOKEN': 'wrong'}):
            ms, v, err = S.resolve_model(self.opts(), log=self.logs.append)
        self.assertIn('MODEL_LOADER_LOCAL_TOKEN is wrong', err)
        with mock.patch.dict(os.environ, {'MODEL_LOADER_LOCAL_TOKEN': 's3cret'}):
            self.resolve()
        # TCP only with a token: model-loader serves no TCP local API without one
        ms, v, err = S.resolve_model(self.opts('--model-loader', '127.0.0.1:9'), log=self.logs.append)
        self.assertIn('a TCP local API needs MODEL_LOADER_LOCAL_TOKEN', err)
        with mock.patch.dict(os.environ, {'MODEL_LOADER_LOCAL_TOKEN': 's3cret'}):
            ms, v, err = S.resolve_model(self.opts('--model-loader', '127.0.0.1:9'), log=self.logs.append)
        self.assertIn('model-loader at 127.0.0.1:9', err)                  # it tried: nothing listens there

    def test_a_revision_outside_the_store_is_refused(self):
        for bad in ('../../etc', '/abs', 'a//b', ''):
            with mock.patch.object(mlsource.Client, 'put', return_value={'state': 'ready', 'revision': bad}):
                ms, v, err = S.resolve_model(self.opts(), log=self.logs.append)
            self.assertIn(f'revision {bad!r}', err)
        ms, v, err = S.resolve_model(self.opts('--model-loader-model', '..'), log=self.logs.append)
        self.assertIn("MODEL_LOADER_MODEL '..'", err)

    def test_an_unknown_catalog_name_is_named(self):
        o = self.opts('--model-loader-model', 'nope')
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertIn('no catalog model nope', err)

    def test_it_waits_for_downloading(self):
        self.ml.slow = True
        o, ms, v = self.resolve()
        ms.release.fetch_rank(4, 1, v)
        self.assertEqual(ms.release.check(4, 1, v), [])
        self.assertTrue(any('downloading' in m for m in self.logs), self.logs)

    def test_a_layout_requests_its_halves(self):
        o, ms, v = self.resolve()
        g6layout = os.path.join(os.path.dirname(T.hfsource.fetch_ranks().g6load()), 'g6layout')
        t4, p3 = os.path.join(self.d, 't4.json'), os.path.join(self.d, 'p3.json')
        for argv in (['formula', '--members', 'a,b,c,d', '--out', t4],
                     ['plan', '--from', t4, '--members', 'a,b,d', '--out', p3]):
            subprocess.run([g6layout] + argv + ['--widths', v.path], check=True, capture_output=True)
        o, ms, v = self.resolve('--layout', p3, '--variant', 'home', tp=3)
        ms.release.fetch_rank(3, 1, v)
        self.assertEqual(ms.release.check(3, 1, v), [])
        inc = set(self.ml.puts[-1][1]['include'])
        self.assertTrue({f'index-slices/tp3/k3/rank{r}.safetensors' for r in range(3)} <= inc)
        for l in T.LAYERS:
            want = T.half_spans(l, lambda p: 3, T.layout_halves(p3, 1, l))
            for name, off, nb in want:
                with open(os.path.join(self.rel, 'experts', name), 'rb') as a, \
                        open(os.path.join(ms.release.v2, name), 'rb') as b:
                    a.seek(off)
                    b.seek(off)
                    self.assertEqual(a.read(nb), b.read(nb), (name, off))

    def test_prepare(self):
        import launch
        argv = ['--no-config', '--peers', PEERS, '--model', URL, '--model-cache', self.store, '--fetch',
                'model-loader', '--g6-home', self.home, '--dflash', '0', '--rank', '2']
        out = subprocess.run([sys.executable, launch.__file__, 'prepare', *argv], capture_output=True, text=True,
                             env={**os.environ, 'G6LOAD': T.hfsource.fetch_ranks().g6load()})
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn('rank 2 of model', out.stdout)


def free_port():
    import socket
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(T.HAVE and os.access(os.environ.get('MODEL_LOADER_BIN', ''), os.X_OK),
                     'set MODEL_LOADER_BIN to a model-loader binary (github.com/kindlingai/model-loader)')
class RealModelLoaderTests(unittest.TestCase):
    """The real model-loader, -mode=both, its catalog's origin a plain HTTP source on localhost serving the release (a
    Hugging Face source always goes to huggingface.co)."""

    def test_boot_from_a_real_model_loader(self):
        t = tempfile.mkdtemp(prefix='gr8mlr-')
        self.addCleanup(shutil.rmtree, t, True)
        rel = os.path.join(t, 'release')
        T.make_release(rel)
        files = []
        for d, _, names in os.walk(rel):
            for n in names:
                p = os.path.join(d, n)
                files.append({'path': os.path.relpath(p, rel), 'size': os.path.getsize(p),
                              'sha256': modelset.sha256_file(p)})
        with open(os.path.join(rel, 'manifest.json'), 'w') as f:
            json.dump({'files': files}, f)
        import http.server
        import functools
        import threading
        log = []

        class H(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                log.append(self.path)

            def send_head(self):
                rng = self.headers.get('Range')
                if not rng:
                    return super().send_head()
                p = self.translate_path(self.path)
                a, b = (int(x) if x else None for x in rng.split('=')[1].split('-'))
                size = os.path.getsize(p)
                b = size - 1 if b is None else min(b, size - 1)
                f = open(p, 'rb')
                f.seek(a)
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
                self.send_header('Content-Length', str(b - a + 1))
                self.end_headers()
                data = f.read(b - a + 1)
                f.close()
                self.wfile.write(data)
                return None

        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(H, directory=rel))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()) if srv.socket.fileno() >= 0 else None)
        store = os.path.join(t, 'store')
        with open(os.path.join(t, 'catalog.yaml'), 'w') as f:
            f.write(f'models:\n  - name: {NAME}\n    source:\n      type: http\n      url: '
                    f'http://127.0.0.1:{srv.server_address[1]}/\n    revision: main\n'
                    '    segment_sizes:\n      - {path: "experts/*.k?.safetensors", size: 65536}\n')
        sel = os.path.join(t, 'selection.yaml')
        p = subprocess.Popen([os.environ['MODEL_LOADER_BIN'], '-mode=both', f'-catalog={t}/catalog.yaml',
                              f'-selection={sel}', f'-store={store}', f'-listen=127.0.0.1:{free_port()}',
                              '-reconcile-interval=1s'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(p.wait)
        self.addCleanup(p.terminate)
        sock = os.path.join(store, 'model-loader.sock')
        for _ in range(100):
            if os.path.exists(sock):
                break
            time.sleep(0.1)
        home = os.path.join(t, 'home')
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', URL, '--model-cache', store, '--fetch',
                       'model-loader', '--g6-home', home, '--dflash', '0'], {})
        opt.validate(o)
        logs = []
        ms, v, err = S.resolve_model(o, log=logs.append)
        self.assertEqual(err, '', logs)
        ms.release.fetch_rank(4, 1, v)
        self.assertEqual(ms.release.check(4, 1, v), [])
        T.same_halves(rel, ms.release.v2, 3, 4, 1, lambda p: 3)
        self.assertFalse([x for x in log if 'k2.safetensors' in x])       # home: no K2 bytes from the origin
        # 64 KiB segments: the store holds about the rank's quarter of each width file, not all of it
        k3 = os.path.join(ms.release.v2, 'exlr8-layer-003.k3.safetensors')
        held = os.stat(k3).st_blocks * 512
        self.assertLess(held, 0.6 * os.path.getsize(k3), (held, os.path.getsize(k3)))
        # a second boot with the origin gone reads the store alone
        srv.shutdown()
        srv.server_close()
        ms, v, err = S.resolve_model(o, log=logs.append)
        self.assertEqual(err, '', logs)
        ms.release.fetch_rank(4, 1, v)
        self.assertEqual(ms.release.check(4, 1, v), [])


if __name__ == '__main__':
    unittest.main()
