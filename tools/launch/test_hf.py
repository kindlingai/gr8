#!/usr/bin/env python3
"""The Hugging Face source against a fake hub on localhost: commit resolution, range GETs of one rank's expert halves,
the hashes, resume, the token rule. The fake release holds synthetic exlr8 v2 layers (tools/convert/v2synth.py), so
fetch_ranks.py and bin/g6load run for real. Needs numpy to write the synthetic release (the fetch itself runs
without it) and bin/g6load (cargo build --release --manifest-path engine/loader-rs/Cargo.toml).
    python3 -m unittest discover -s tools/launch"""
import contextlib
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import hfsource  # noqa: E402
import modelset  # noqa: E402

try:
    import numpy  # noqa: F401
    from tools.convert import v2, v2slices, v2synth
    HAVE = True
except ImportError:
    HAVE = False

RELEASE = 'fakeorg/glm-exlr8'
LAYERS = (3, 4)
NE = 2                  # experts per layer in the synthetic files: small files, the same layout
LFS_MIN = 4096          # files at least this big are served as LFS files: a redirect to the "CDN"


def git_blob(b):
    return hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest()


def manifest(width_of):
    return {'format': 'exlr8-v2-manifest', 'name': 'x', 'set_id': '5e' * 32, 'layers': {
        str(l): {'gate': [[width_of('gate', e, h) for h in range(16)] for e in range(NE)],
                 'up': [[width_of('up', e, h) for h in range(16)] for e in range(NE)],
                 'down': [width_of('down', e, 0) for e in range(NE)], 'shared': 0} for l in LAYERS}}


def half_spans(l, width, owned):
    """{(width file, offset, nbytes)} of layer l's halves owned(e) gives. width(proj) -> k."""
    keep, v2.N_EXP = v2.N_EXP, NE
    try:
        with tempfile.TemporaryDirectory() as d:     # the release's index file says where each half sits
            v2synth.make_layer(d, l, (2, 3), NE, seed=l, shared=False)
            L = v2.V2Layer(d, l)
            out = {(v2.WIDTH_FILE.format(l=l, k=width(p)),) + tuple(L.loc(e, p, h, width(p))[1:])
                   for e in range(NE) for h in owned(e) for p in v2.PROJ}
            L.close()
            return out
    finally:
        v2.N_EXP = keep


def layout_halves(path, rank, l):
    """owned(e) for a layout file: the halves of layer l's expert e it gives rank."""
    with open(path) as f:
        g = json.load(f)['groups']
    return lambda e: [h for h, r in enumerate(g[f'halves {l} {e}']) if r == rank]


def same_halves(release, v2_dir, l, tp, rank, width):
    """Rank `rank`'s halves of layer l in the fetched sparse files equal the release's bytes. width(proj) -> k."""
    keep, v2.N_EXP = v2.N_EXP, NE
    try:
        with tempfile.TemporaryDirectory() as d:     # the release's index file says where each half sits
            v2synth.make_layer(d, l, (2, 3), NE, seed=l, shared=False)
            L = v2.V2Layer(d, l)
            for e in range(NE):
                for h in v2.owned_halves(e, l, tp, rank):
                    for p in v2.PROJ:
                        _, off, n = L.loc(e, p, h, width(p))
                        name = v2.WIDTH_FILE.format(l=l, k=width(p))
                        with open(os.path.join(release, 'experts', name), 'rb') as a, \
                                open(os.path.join(v2_dir, name), 'rb') as b:
                            a.seek(off)
                            b.seek(off)
                            assert a.read(n) == b.read(n), (name, e, h, p)
            L.close()
    finally:
        v2.N_EXP = keep


def make_release(root):
    """A release laid out like the real one: experts/, index-slices/, dense/, manifests/, hashes/, the tokenizer."""
    ex = os.path.join(root, 'experts')
    keep, v2.N_EXP = v2.N_EXP, NE
    try:
        for l in LAYERS:
            v2synth.make_layer(ex, l, (2, 3), NE, seed=l, shared=False)
        with contextlib.redirect_stdout(io.StringIO()):
            v2slices.main(['--v2', ex, '--out', os.path.join(root, 'index-slices'), '--tp', '2-4', '--widths', '2,3'])
    finally:
        v2.N_EXP = keep
    for n in os.listdir(ex):          # the release ships the index files, and the fetch has no use for them
        if '.index.' in n:
            os.unlink(os.path.join(ex, n))

    def put(rel, data):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(data)
    for name, w in (('home', lambda p, e, h: 3), ('down1', lambda p, e, h: 2),
                    ('up1', lambda p, e, h: 3 if p != 'up' else 2)):
        put(f'manifests/exlr8-manifest-{name}.json', json.dumps(manifest(w)).encode())
    rnd = numpy.random.default_rng(7)
    for name in ('exlr8-dense-003', 'exlr8-dense-004', 'exlr8-vocab'):
        put(f'dense/{name}.safetensors', rnd.bytes(300_000))
    for name in hfsource.META_FILES:
        put(name, f'{{"{name}": 1}}'.encode())
    put('README.md', b'fake release\n')
    for d, _, files in os.walk(root):
        for n in files:
            p = os.path.join(d, n)
            rel = os.path.relpath(p, root)
            if rel.startswith(('hashes/', 'manifests/', 'README')) or n in hfsource.META_FILES:
                continue
            with open(p, 'rb') as f:
                b = f.read()
            hp = f'hashes/{n}.sha256' if rel.startswith('experts/') else f'hashes/{rel}.sha256'
            put(hp, f'{hashlib.sha256(b).hexdigest()} {len(b)}\n'.encode())


class FakeHub:
    """The parts of the hub API and file serving the fetch uses. repos: {repo: {commit: root dir}}, refs: {repo: {rev:
    commit}}. log: (method, path, range, authorization) per request. corrupt: {repo path: n} flips the first byte of a
    response from the file's start n times, corrupt_halves the same for a range past the header. page: tree entries per
    page."""

    def __init__(self, repos, refs, page=20):
        self.repos, self.refs, self.page, self.log, self.corrupt, self.corrupt_halves = repos, refs, page, [], {}, {}
        hub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_HEAD(self):
                self.serve(head=True)

            def do_GET(self):
                self.serve(head=False)

            def send(self, code, body=b'', headers=(), head=False):
                self.send_response(code)
                for k, v in headers:
                    self.send_header(k, v)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)

            def serve(self, head):
                u = urllib.parse.urlparse(self.path)
                hub.log.append((self.command, u.path, self.headers.get('Range'), self.headers.get('Authorization')))
                m = re.fullmatch(r'/api/models/([^/]+/[^/]+)/revision/([^/]+)', u.path)
                if m:
                    c = hub.commit(m.group(1), m.group(2))
                    return self.send(200 if c else 404, json.dumps({'sha': c}).encode())
                m = re.fullmatch(r'/api/models/([^/]+/[^/]+)/tree/([^/]+)', u.path)
                if m:
                    c = hub.commit(m.group(1), m.group(2))
                    entries = hub.tree(m.group(1), c)
                    start = int(urllib.parse.parse_qs(u.query).get('cursor', ['0'])[0])
                    page = entries[start:start + hub.page]
                    hdr = []
                    if start + hub.page < len(entries):
                        nxt = f'http://127.0.0.1:{hub.port}{u.path}?recursive=true&cursor={start + hub.page}'
                        hdr.append(('Link', f'<{nxt}>; rel="next"'))
                    return self.send(200, json.dumps(page).encode(), hdr)
                m = re.fullmatch(r'/([^/]+/[^/]+)/resolve/([^/]+)/(.+)', u.path) or \
                    re.fullmatch(r'/cdn/([^/]+/[^/]+)/([0-9a-f]{40})/(.+)', u.path)
                if not m:
                    return self.send(404)
                repo, c, path = m.group(1), hub.commit(m.group(1), m.group(2)), m.group(3)
                local = os.path.join(hub.repos[repo][c], path)
                if not os.path.isfile(local):
                    return self.send(404)
                if u.path.startswith('/' + repo + '/resolve/') and os.path.getsize(local) >= LFS_MIN:
                    # another host name, as the real CDN is: the token must not follow the redirect
                    cdn = f'http://localhost:{hub.port}/cdn/{repo}/{c}/{path}'
                    return self.send(302, headers=[('Location', cdn)], head=head)
                with open(local, 'rb') as f:
                    data = f.read()
                rng = self.headers.get('Range')
                a, b = (int(x) for x in rng.split('=')[1].split('-')) if rng else (0, len(data) - 1)
                body = data[a:b + 1]
                for table, hit in ((hub.corrupt, a == 0), (hub.corrupt_halves, a >= 65536)):
                    if hit and table.get(path) and body and not head:
                        table[path] -= 1
                        body = bytes([body[0] ^ 1]) + body[1:]
                return self.send(206 if rng else 200, body, head=head)
        self.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def commit(self, repo, rev):
        return rev if rev in self.repos.get(repo, {}) else self.refs.get(repo, {}).get(rev)

    def tree(self, repo, commit):
        root, out = self.repos[repo][commit], []
        for d, _, files in sorted(os.walk(root)):
            for n in sorted(files):
                p = os.path.join(d, n)
                with open(p, 'rb') as f:
                    b = f.read()
                e = {'type': 'file', 'path': os.path.relpath(p, root), 'size': len(b), 'oid': git_blob(b)}
                if len(b) >= LFS_MIN:
                    e['lfs'] = {'oid': hashlib.sha256(b).hexdigest(), 'size': len(b)}
                out.append(e)
        return out

    def requests(self, pattern, method='GET'):
        return [x for x in self.log if x[0] == method and re.search(pattern, x[1])]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


C1, C2 = 'a' * 40, 'b' * 40


@unittest.skipUnless(HAVE, 'needs numpy (the fetch reads the v2 index files with it)')
class HFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.mkdtemp(prefix='gr8hf-')
        cls.r1 = os.path.join(cls.t, 'release1')
        make_release(cls.r1)
        cls.r2 = shutil.copytree(cls.r1, os.path.join(cls.t, 'release2'))     # a second commit of the same files

    def setUp(self):
        self.hub = FakeHub({RELEASE: {C1: self.r1, C2: self.r2}}, {RELEASE: {'main': C1, 'v0.4': C2}})
        self.env = mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': f'http://127.0.0.1:{self.hub.port}'})
        self.env.start()
        os.environ.pop('HF_TOKEN', None)
        self.cache = tempfile.mkdtemp(prefix='gr8hfc-')
        self.logs = []

    def tearDown(self):
        self.env.stop()
        self.hub.close()

    def release(self, rev='main', refresh=False):
        r = hfsource.Release(self.cache, RELEASE, rev, refresh=refresh, log=self.logs.append)
        r.fetch_meta()
        return r

    def variant(self, r, name='home'):
        return modelset.resolve_variant(name, r.v2, [r.dir])

    def test_parse_url(self):
        P = hfsource.parse_url
        self.assertEqual(P('https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4'), ('kindlingai/glm-5.3-exlr8-k2-k3-k4', 'main'))
        self.assertEqual(P('https://huggingface.co/a/b@v0.4'), ('a/b', 'v0.4'))
        self.assertEqual(P('https://huggingface.co/a/b/tree/' + C1), ('a/b', C1))
        self.assertIsNone(P('kindlingai/glm-5.3-exlr8-k2-k3-k4'))     # a bare org/repo is a local path
        self.assertIsNone(P('/data/packs'))
        with self.assertRaises(modelset.ModelError):
            P('https://example.com/a/b')

    def test_one_rank_fetched_and_checked(self):
        r = self.release()
        self.assertEqual(r.commit, C1)
        self.assertEqual(sorted(os.listdir(r.tokenizer)), sorted(hfsource.META_FILES))
        v = self.variant(r)
        self.assertEqual(r.check(4, 1, v), [f'{RELEASE}@{C1[:12]} rank 1 of TP=4, variant home, is not in the cache yet'])
        r.fetch_rank(4, 1, v)
        self.assertEqual(r.check(4, 1, v), [])
        self.assertEqual(sorted(os.listdir(os.path.join(r.slices, 'tp4', 'k3'))), ['rank1.safetensors'])
        self.assertFalse(os.path.exists(os.path.join(r.slices, 'tp2')))
        self.assertEqual(len(os.listdir(os.path.join(r.dir, 'dense'))), 3)
        # rank 1 owns a quarter of the halves: they hold the release's bytes, and the k2 file was not fetched
        same_halves(self.r1, r.v2, 3, 4, 1, lambda p: 3)
        got = sum(int(x[2].split('=')[1].split('-')[1]) - int(x[2].split('=')[1].split('-')[0]) + 1
                  for x in self.hub.requests(r'exlr8-layer-003\.k3') if x[2])
        size = os.path.getsize(os.path.join(self.r1, 'experts', 'exlr8-layer-003.k3.safetensors'))
        self.assertLess(got, size * 0.4)
        self.assertEqual(self.hub.requests(r'exlr8-layer-00\d\.k2'), [])            # home uses K3 only
        self.assertTrue(any('GB to fetch' in m for m in self.logs))

    def test_a_rerun_fetches_nothing_and_a_variant_adds_its_widths(self):
        r = self.release()
        r.fetch_rank(4, 2, self.variant(r))
        os.unlink(r.marker(4, 2, self.variant(r)))
        n = len(self.hub.log)
        r.fetch_rank(4, 2, self.variant(r))
        later = self.hub.log[n:]
        self.assertFalse([x for x in later if re.search(r'\.k3\.safetensors|dense/', x[1]) and x[2] and
                          not x[2].startswith('bytes=0-7')], later)                 # only the 8-byte header lengths
        r.fetch_rank(4, 2, self.variant(r, 'down1'))
        self.assertTrue(self.hub.requests(r'exlr8-layer-003\.k2'))
        self.assertEqual(r.check(4, 2, self.variant(r)), [])

    def test_a_new_commit_links_what_it_shares_with_the_last(self):
        r1 = self.release()
        r1.fetch_rank(4, 1, self.variant(r1))
        os.unlink(os.path.join(r1.dir, hfsource.TREE))               # a cache written before the file list existed
        c3 = 'c' * 40                                                  # main moves to a commit with one dense file changed
        r3 = shutil.copytree(self.r1, os.path.join(tempfile.mkdtemp(), 'release3'))
        dense = sorted(os.listdir(os.path.join(r3, 'dense')))[0]
        with open(os.path.join(r3, 'dense', dense), 'r+b') as f:
            f.write(b'\xff')
        for hp in (os.path.join(r3, 'hashes', 'dense', dense + '.sha256'), os.path.join(r3, 'hashes', dense + '.sha256')):
            if os.path.isfile(hp):                                     # the release's own hash of the new bytes
                with open(hp, 'w') as f:
                    f.write(f"{modelset.sha256_file(os.path.join(r3, 'dense', dense))} "
                            f"{os.path.getsize(os.path.join(r3, 'dense', dense))}\n")
        self.hub.repos[RELEASE][c3] = r3
        self.hub.refs[RELEASE]['main'] = c3
        r = self.release(refresh=True)
        n = len(self.hub.log)
        r.fetch_rank(4, 1, self.variant(r))
        later = [x for x in self.hub.log[n:] if x[2] and not x[2].startswith('bytes=0-7')]
        self.assertEqual({re.sub(r'.*/', '', x[1]) for x in later}, {dense})      # only the changed file
        self.assertEqual(r.check(4, 1, self.variant(r)), [])
        k3 = os.path.join('experts', 'exlr8-layer-003.k3.safetensors')
        self.assertTrue(os.path.samefile(os.path.join(r.dir, k3), os.path.join(r1.dir, k3)))
        self.assertFalse(os.path.samefile(os.path.join(r.dir, 'dense', dense), os.path.join(r1.dir, 'dense', dense)))
        same_halves(r3, r.v2, 3, 4, 1, lambda p: 3)

    def test_a_layout_fetches_only_the_halves_the_box_lacks(self):
        # a box that holds rank 1 of a TP=4 formula cache (b of a, b, c, d) goes to TP=3 with a planned layout (a, b, d)
        r = self.release()
        v = self.variant(r)
        r.fetch_rank(4, 1, v)
        g6layout = os.path.join(os.path.dirname(hfsource.fetch_ranks().g6load()), 'g6layout')
        t4, p3 = os.path.join(self.cache, 't4.json'), os.path.join(self.cache, 'p3.json')
        for argv in (['formula', '--members', 'a,b,c,d', '--out', t4],
                     ['plan', '--from', t4, '--members', 'a,b,d', '--out', p3]):
            subprocess.run([g6layout] + argv + ['--widths', v.path], check=True, capture_output=True)
        lr = hfsource.Release(self.cache, RELEASE, 'main', log=self.logs.append, layout=p3)
        self.assertEqual(lr.check(3, 1, v), [f'{RELEASE}@{C1[:12]} rank 1 of TP=3, variant home, layout '
                                             f'{modelset.sha256_file(p3)[:12]}, is not in the cache yet'])
        n = len(self.hub.log)
        lr.fetch_rank(3, 1, v)
        self.assertEqual(lr.check(3, 1, v), [])
        self.assertEqual(r.check(4, 1, v), [])                    # the formula's marker stands
        self.assertIn('.layout-', os.path.basename(lr.marker(3, 1, v)))
        self.assertEqual(sorted(os.listdir(os.path.join(r.slices, 'tp3', 'k3'))),
                         [f'rank{i}.safetensors' for i in range(3)])   # every rank's slices list the layout's halves
        got = [(re.sub(r'.*/', '', x[1]), *map(int, x[2].split('=')[1].split('-')))
               for x in self.hub.log[n:] if x[2] and 'exlr8-layer-' in x[1] and not x[2].startswith('bytes=0-7')]
        for l in LAYERS:
            held = half_spans(l, lambda p: 3, lambda e: v2.owned_halves(e, l, 4, 1))
            want = half_spans(l, lambda p: 3, layout_halves(p3, 1, l))
            self.assertTrue(want - held, f'layer {l}: the layout gives the box nothing new')
            # the new halves came from the hub, and none it held already
            for name, off, nb in want - held:
                self.assertTrue(any(f == name and a <= off and off + nb - 1 <= b for f, a, b in got), (name, off))
            for name, off, nb in held:
                self.assertFalse(any(f == name and a <= off and off + nb - 1 <= b for f, a, b in got), (name, off))
            for name, off, nb in want:
                with open(os.path.join(self.r1, 'experts', name), 'rb') as a, open(os.path.join(r.v2, name), 'rb') as b:
                    a.seek(off)
                    b.seek(off)
                    self.assertEqual(a.read(nb), b.read(nb), (name, off))

    def test_the_release_alone_is_the_model(self):
        import options as opt
        import supervisor as S
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4', '--model',
                       f'https://huggingface.co/{RELEASE}', '--model-cache', self.cache, '--dflash', '0'], {})
        opt.validate(o)
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertEqual(err, '')
        self.assertEqual((o.get('V2_DENSE'), o.get('V2')), (ms.release.dir, ms.release.v2))
        self.assertEqual(ms.set['source'], {'kind': 'exlr8-release', 'weights': '5e' * 32})
        self.assertEqual(ms.tokenizer, ms.release.tokenizer)
        plan = opt.engine_plan(o, rank=2, rank_dir=ms.dir, aux_dir=ms.dir, home='/h')
        self.assertIn(ms.release.dir, plan['args'][plan['args'].index('--v2-dense') + 1])
        self.assertNotIn('--manifest', plan['args'])
        self.assertEqual(str(ms.release), ms.source)

    def test_kvb_fp8_with_the_release_as_the_model(self):
        import options as opt
        import supervisor as S
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4', '--model',
                       f'https://huggingface.co/{RELEASE}', '--model-cache', self.cache, '--dflash', '0', '--kvb', 'fp8'], {})
        opt.validate(o)                                          # V2_DENSE is set later, by resolve_model
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertEqual(err, '')
        plan = opt.engine_plan(o, rank=2, rank_dir=ms.dir, aux_dir=ms.dir, home='/h')
        self.assertIn('G6_KVB=fp8', plan['env'])
        self.assertIn('--v2-dense', plan['args'])

    def test_commit_is_recorded_and_refreshed(self):
        r = self.release()
        with open(os.path.join(r.repo.root, 'refs', 'main')) as f:
            self.assertEqual(f.read().strip(), C1)
        self.hub.refs[RELEASE]['main'] = C2                                         # main moves on the hub
        self.assertEqual(self.release().commit, C1)                                 # up: the commit recorded before
        self.assertEqual(self.release(refresh=True).commit, C2)                     # prepare: asks again
        self.assertEqual(self.release('v0.4').commit, C2)
        self.assertEqual(self.release(C1).commit, C1)
        v, ids = modelset.Variant('home'), []
        for c in (C1, C2, None):
            ms = modelset.ModelSet({'set_id': 'f' * 64}, '', '', lambda r: '')
            ms.release = c and hfsource.Release(self.cache, RELEASE, c, log=print)
            ids.append(modelset.model_id(ms, v))
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids[2], 'f' * 64)

    def test_whole_file_resumes_and_hashes_are_checked(self):
        r = self.release()
        src = os.path.join(self.r1, 'dense', 'exlr8-vocab.safetensors')
        dest = os.path.join(r.dir, 'dense', 'exlr8-vocab.safetensors')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(src, 'rb') as f, open(dest + '.part', 'wb') as g:
            g.write(f.read(100_000))
        r.repo.fetch('dense/exlr8-vocab.safetensors', dest)
        self.assertEqual(self.hub.requests(r'^/cdn/.*dense/exlr8-vocab')[-1][2], 'bytes=100000-299999')
        self.assertEqual(modelset.sha256_file(dest), modelset.sha256_file(src))
        self.hub.corrupt['dense/exlr8-dense-003.safetensors'] = 1
        bad = os.path.join(r.dir, 'dense', 'exlr8-dense-003.safetensors')
        with self.assertRaises(modelset.ModelError) as e:
            r.repo.fetch('dense/exlr8-dense-003.safetensors', bad)
        self.assertIn('does not match its hash', str(e.exception))
        self.assertFalse(os.path.exists(bad) or os.path.exists(bad + '.part'))
        r.repo.fetch('dense/exlr8-dense-003.safetensors', bad)                    # the next try is clean
        self.assertTrue(os.path.isfile(bad))

    def test_token_only_when_set_and_never_to_the_cdn(self):
        self.release()
        self.assertFalse([x for x in self.hub.log if x[3]])
        with mock.patch.dict(os.environ, {'HF_TOKEN': 'hf_test'}):
            r = self.release(refresh=True)
            r.repo.fetch('dense/exlr8-vocab.safetensors', os.path.join(self.cache, 'v.safetensors'))
        authed = [x for x in self.hub.log if x[3]]
        self.assertTrue(authed)
        self.assertTrue(all(x[3] == 'Bearer hf_test' and not x[1].startswith('/cdn/') for x in authed))
        self.assertTrue(self.hub.requests(r'^/cdn/.*exlr8-vocab'))

    def test_a_missing_tokenizer_file_is_named(self):
        r3 = shutil.copytree(self.r1, os.path.join(self.cache, 'release3'))
        os.unlink(os.path.join(r3, 'chat_template.jinja'))
        self.hub.repos[RELEASE]['c' * 40] = r3
        with self.assertRaises(modelset.ModelError) as e:
            self.release('c' * 40)
        self.assertIn('lacks chat_template.jinja at its root', str(e.exception))



NO_NUMPY_FETCH = """import json, os, sys
sys.modules['numpy'] = None                     # a box without numpy: any import of it fails
sys.path[:0] = [sys.argv[1], os.path.dirname(os.path.dirname(sys.argv[1]))]
import hfsource, modelset
r = hfsource.Release(sys.argv[2], sys.argv[3], 'main', log=lambda m: None)
r.fetch_meta()
v = modelset.resolve_variant('up1', r.v2, [r.dir])
r.fetch_rank(4, 3, v)
print(json.dumps(r.check(4, 3, v)))
"""


@unittest.skipUnless(HAVE, 'needs numpy to write the synthetic release')
class NoNumpyFetchTests(unittest.TestCase):
    """The fetch on a box without numpy: the HTTP in Python, the planning and checking in bin/g6load."""

    def setUp(self):
        self.t = tempfile.mkdtemp(prefix='gr8np-')
        self.root = os.path.join(self.t, 'release')
        make_release(self.root)

    def test_the_fetch_runs_without_numpy(self):
        hub = FakeHub({RELEASE: {C1: self.root}}, {RELEASE: {'main': C1}})
        try:
            cache = os.path.join(self.t, 'cache')
            out = subprocess.run([sys.executable, '-c', NO_NUMPY_FETCH, HERE, cache, RELEASE], capture_output=True,
                                 text=True, env={**os.environ, 'GR8_HF_ENDPOINT': f'http://127.0.0.1:{hub.port}'})
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(json.loads(out.stdout), [])
        finally:
            hub.close()
        same_halves(self.root, os.path.join(cache, 'hf', 'fakeorg--glm-exlr8', C1, 'experts'), 4, 4, 3,
                    lambda p: 2 if p == 'up' else 3)

    def test_a_bad_half_is_fetched_again(self):
        hub = FakeHub({RELEASE: {C1: self.root}}, {RELEASE: {'main': C1}})
        hub.corrupt_halves['experts/exlr8-layer-003.k3.safetensors'] = 1     # one range of halves comes back bad
        try:
            with mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': f'http://127.0.0.1:{hub.port}'}):
                logs = []
                r = hfsource.Release(os.path.join(self.t, 'c2'), RELEASE, 'main', log=logs.append)
                r.fetch_meta()
                r.fetch_rank(4, 0, modelset.resolve_variant('home', r.v2, [r.dir]))
        finally:
            hub.close()
        self.assertTrue(any('failed their xxhash64: fetching them again' in m for m in logs), logs)
        same_halves(self.root, r.v2, 3, 4, 0, lambda p: 3)


class KeepAliveTests(unittest.TestCase):
    """Hub.get's range GETs reuse kept-alive connections (#147): one connection for many ranges, a redirect to another
    host followed on that host's connection, an idle connection the server closed (in clear or over TLS) replaced
    without a retry, and the token sent only to the hub's own scheme and host."""

    def setUp(self):
        self.data, self.conns, self.close_next, self.auth = bytes(range(256)) * 64, [], [False], []
        t = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def setup(self):
                super().setup()
                t.conns.append(self.client_address)

            def do_GET(self):
                t.auth.append(self.headers.get('Authorization'))
                if self.path.startswith('/hop/'):
                    self.send_response(302)
                    self.send_header('Location', f'http://localhost:{t.port}/f/{self.path[5:]}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                a, b = (int(x) for x in self.headers['Range'].split('=')[1].split('-'))
                body = t.data[a:b + 1]
                self.send_response(206)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if t.close_next[0]:
                    t.close_next[0], self.close_connection = False, True

        self.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.hub = hfsource.fetch_ranks().Hub('o/r', base=f'http://127.0.0.1:{self.port}')
        self.hub.proxies = {}

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_one_connection_for_many_ranges(self):
        self.hub.urls['x'] = f'http://127.0.0.1:{self.port}/f/x'
        for i in range(50):
            self.assertEqual(self.hub.get('x', i * 100, 100), self.data[i * 100:i * 100 + 100])
        self.assertEqual(len(self.conns), 1)

    def test_a_redirect_and_a_closed_idle_connection(self):
        self.hub.urls['x'] = f'http://127.0.0.1:{self.port}/hop/x'
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            for i in range(10):
                self.close_next[0] = i == 4      # the server drops the connection after the fifth range
                self.assertEqual(self.hub.get('x', i * 7, 7), self.data[i * 7:i * 7 + 7])
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(len(self.conns), 3)    # 127.0.0.1 and localhost, then localhost again after the close

    def test_the_token_goes_to_the_hub_host_only(self):
        here = f'http://127.0.0.1:{self.port}'
        with mock.patch.dict(os.environ, {'HF_TOKEN': 'tok'}):
            lookalike = hfsource.fetch_ranks().Hub('o/r', base=here[:-1])     # the same URL prefix, another port
            hub = hfsource.fetch_ranks().Hub('o/r', base=here)
        for h in (lookalike, hub):
            h.proxies, h.urls['x'] = {}, f'{here}/f/x'
            h.get('x', 0, 10)
        self.assertEqual(self.auth, [None, 'Bearer tok'])

    @unittest.skipUnless(shutil.which('openssl'), 'needs openssl to make a test certificate')
    def test_a_tls_connection_dropped_without_close_notify(self):
        d = tempfile.mkdtemp(prefix='gr8tls-')
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', f'{d}/k.pem', '-out',
                        f'{d}/c.pem', '-days', '1', '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
                       check=True, capture_output=True)
        sc = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        sc.load_cert_chain(f'{d}/c.pem', f'{d}/k.pem')
        ls, conns = socket.create_server(('127.0.0.1', 0)), []
        self.addCleanup(lambda: [c.close() for c in [ls] + conns])

        def one(c, drop):
            buf = b''
            try:
                while True:
                    while b'\r\n\r\n' not in buf:
                        chunk = c.recv(65536)
                        if not chunk:
                            return
                        buf += chunk
                    head, buf = buf.split(b'\r\n\r\n', 1)
                    if drop is None:                # FIN without close_notify after the first range, no RST
                        continue
                    a, b = (int(x) for x in re.search(rb'bytes=(\d+)-(\d+)', head).groups())
                    body = self.data[a:b + 1]
                    c.sendall(b'HTTP/1.1 206 Partial Content\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
                    if drop:
                        socket.socket.shutdown(c, socket.SHUT_WR)
                        drop = None
            except OSError:
                return

        def serve():
            while True:
                try:
                    s, _ = ls.accept()
                    c = sc.wrap_socket(s, server_side=True)
                except OSError:
                    return
                conns.append(c)
                threading.Thread(target=one, args=(c, len(conns) == 1), daemon=True).start()
        threading.Thread(target=serve, daemon=True).start()
        hub = hfsource.fetch_ranks().Hub('o/r', base=f'https://localhost:{ls.getsockname()[1]}')
        hub.proxies, hub.urls['x'] = {}, f'https://localhost:{ls.getsockname()[1]}/f/x'
        err = io.StringIO()
        with mock.patch('ssl._create_default_https_context', lambda: ssl.create_default_context(cafile=f'{d}/c.pem')), \
                contextlib.redirect_stderr(err):
            for i in range(4):
                self.assertEqual(hub.get('x', i * 9, 9), self.data[i * 9:i * 9 + 9])
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(len(conns), 2)


STUB_CONVERT = """import json, os, sys
a = sys.argv[1:]
out, r = a[a.index('--out') + 1], int(a[a.index('--ranks') + 1])
if os.environ.get('STUB_CONVERT_FAIL'):
    sys.exit('config hidden_size = 1, this converter is written for 6144')
if a[a.index('--nr') + 1] != os.environ.get('STUB_WANT_NR', '4'):
    sys.exit('stub: --nr ' + a[a.index('--nr') + 1])
open(os.path.join(out, 'args.json'), 'w').write(json.dumps(a))
open(os.path.join(out, f'dflash.r{r}.pack'), 'wb').write(bytes(1000 + r))
json.dump({'pack_bytes': 1000 + r}, open(os.path.join(out, f'dflash.r{r}.manifest.json'), 'w'))
"""


@unittest.skipUnless(HAVE, 'needs numpy')
class DrafterTests(unittest.TestCase):
    """The drafter's own repo on the fake hub, with a stub in place of bin/g6load dflash2 (the real drafter is 4.9 GB,
    and the converter checks its full dimensions). engine/loader-rs/tests/dflash2.rs checks the converter itself."""

    def setUp(self):
        self.t = tempfile.mkdtemp(prefix='gr8dr-')
        src = os.path.join(self.t, 'drafter')
        os.makedirs(src)
        with open(os.path.join(src, 'config.json'), 'w') as f:
            f.write('{"hidden_size": 6144}')
        with open(os.path.join(src, 'model.safetensors'), 'wb') as f:
            f.write(os.urandom(50_000))
        self.hub = FakeHub({'fakeorg/drafter': {C1: src}}, {'fakeorg/drafter': {'main': C1}})
        stub = os.path.join(self.t, 'convert.py')
        with open(stub, 'w') as f:
            f.write(STUB_CONVERT)
        self.patches = [mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': f'http://127.0.0.1:{self.hub.port}'}),
                        mock.patch.object(hfsource, 'converter', lambda: [sys.executable, stub])]
        for p in self.patches:
            p.start()
        self.cache = os.path.join(self.t, 'cache')

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.hub.close()

    def test_fetch_and_convert_one_rank(self):
        d = hfsource.Drafter(self.cache, 'fakeorg/drafter', 'main', log=lambda m: None)
        self.assertEqual(d.check(2), [f'the drafter fakeorg/drafter@{C1[:12]} is not converted for rank 2 yet'])
        d.fetch_rank(2)
        self.assertEqual(d.check(2), [])
        self.assertEqual(sorted(os.listdir(d.out)), ['dflash.r2.manifest.json', 'dflash.r2.pack', 'rank2.COMPLETE'])
        self.assertEqual(modelset.sha256_file(os.path.join(d.dir, 'model.safetensors')),
                         modelset.sha256_file(os.path.join(self.t, 'drafter', 'model.safetensors')))
        self.assertFalse([n for n in os.listdir(d.dir) if n.startswith('tmp.')])

    def test_tp6_converts_the_replicated_layout(self):
        d = hfsource.Drafter(self.cache, 'fakeorg/drafter', 'main', log=lambda m: None, tp=6)
        self.assertEqual(d.out, os.path.join(d.dir, 'tp6'))
        with mock.patch.dict(os.environ, {'STUB_WANT_NR': '6'}):
            d.fetch_rank(5)
        self.assertEqual(d.check(5), [])
        self.assertEqual(sorted(os.listdir(d.out)), ['dflash.r5.manifest.json', 'dflash.r5.pack', 'rank5.COMPLETE'])

    def test_the_real_converter_runs(self):
        """bin/g6load dflash2 itself, on the small test drafter: it refuses the dimensions, and the error comes back."""
        src = os.path.join(REPO, 'engine', 'loader-rs', 'tests', 'data', 'dflash2', 'src')
        hub = FakeHub({'fakeorg/small': {C1: src}}, {'fakeorg/small': {'main': C1}})
        self.patches[1].stop()
        try:
            with mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': f'http://127.0.0.1:{hub.port}'}):
                try:
                    hfsource.converter()
                except modelset.ModelError:
                    self.skipTest('no bin/g6load: cargo build --release --manifest-path engine/loader-rs/Cargo.toml')
                d = hfsource.Drafter(self.cache, 'fakeorg/small', 'main', log=lambda m: None)
                with self.assertRaises(modelset.ModelError) as e:
                    d.fetch_rank(3)
        finally:
            self.patches[1].start()
            hub.close()
        self.assertIn('hidden_size = 64, this converter is written for 6144', str(e.exception))
        self.assertFalse(os.path.exists(d.out))

    def test_a_failed_conversion_leaves_nothing(self):
        d = hfsource.Drafter(self.cache, 'fakeorg/drafter', 'main', log=lambda m: None)
        with mock.patch.dict(os.environ, {'STUB_CONVERT_FAIL': '1'}):
            with self.assertRaises(modelset.ModelError) as e:
                d.fetch_rank(1)
        self.assertIn('this converter is written for 6144', str(e.exception))
        self.assertFalse(os.path.exists(d.out))
        self.assertFalse([n for n in os.listdir(d.dir) if n.startswith('tmp.')])

    def test_the_boot_without_a_drafter_and_with_one(self):
        import options as opt
        import supervisor as S
        from test_modelset import make_release as local_release
        rel = local_release(os.path.join(self.t, 'local'))
        base = ['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4', '--model', rel]
        logs = []

        def boot(*argv):
            o = opt.parse([*base, *argv], {})
            opt.validate(o)
            return o, S.resolve_model(o, log=logs.append)
        o, (ms, v, err) = boot()
        self.assertEqual((err, o.get('DFLASH')), ('', '0'))                    # MTP only, with the flag to add
        self.assertIn(f'no DFlash2 drafter: MTP only. For DFlash2, add --drafter {hfsource.DRAFTER_URL}', logs[-1])
        o, (ms, v, err) = boot('--dflash', '1')
        self.assertIn(f'add --drafter {hfsource.DRAFTER_URL}', err)
        o, (ms, v, err) = boot('--drafter', 'https://huggingface.co/fakeorg/drafter', '--model-cache', self.cache)
        self.assertEqual(err, '')
        self.assertEqual((o.get('DFLASH_DIR'), str(ms.drafter)), (ms.drafter.out, f'fakeorg/drafter@{C1[:12]}'))
        plan = opt.engine_plan(o, rank=3, rank_dir='/r', aux_dir='/a', home='/h', epoch=1)
        self.assertIn(os.path.join(ms.drafter.out, 'dflash.r3.pack'), plan['args'])
        self.assertIn('--dflash-pack @DFLASH_DIR/dflash.r@R.pack', ' '.join(opt.engine_plan(o)['args']))
        self.assertNotEqual(modelset.model_id(ms, v), ms.id)
        with self.assertRaises(opt.OptionError):
            opt.validate(opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3', '--drafter',
                                    'https://huggingface.co/fakeorg/drafter'], {}))

    def test_the_tp6_boot(self):
        """TP=6: MTP only unless --dflash 1; with it, the drafter is converted in the replicated layout (--nr 6)."""
        import options as opt
        import supervisor as S
        from test_modelset import make_release as local_release
        rel = local_release(os.path.join(self.t, 'local6'))
        base = ['--no-config', '--peers', ','.join(f'10.0.0.{i}' for i in range(1, 7)), '--model', rel]
        logs = []

        def boot(*argv):
            o = opt.parse([*base, *argv], {})
            opt.validate(o)
            return o, S.resolve_model(o, log=logs.append)
        o, (ms, v, err) = boot()
        self.assertEqual((err, ms.drafter), ('', None))
        self.assertFalse(any('DFlash2' in m for m in logs), logs)                # the TP=6 default boot says nothing new
        self.assertFalse(any(a.startswith('--dflash') for a in opt.engine_plan(o, rank=1, rank_dir='/r', aux_dir='/a', home='/h')['args']))
        o, (ms, v, err) = boot('--dflash', '1')
        self.assertIn(f'add --drafter {hfsource.DRAFTER_URL}', err)
        o, (ms, v, err) = boot('--drafter', 'https://huggingface.co/fakeorg/drafter', '--model-cache', self.cache)
        self.assertEqual((err, ms.drafter), ('', None))
        self.assertIn('--drafter at TP=6 is used with --dflash 1 only: MTP only', logs[-1])
        o, (ms, v, err) = boot('--drafter', 'https://huggingface.co/fakeorg/drafter', '--model-cache', self.cache, '--dflash', '1')
        self.assertEqual(err, '')
        self.assertEqual(ms.drafter.out, os.path.join(ms.drafter.dir, 'tp6'))
        plan = opt.engine_plan(o, rank=5, rank_dir='/r', aux_dir='/a', home='/h', epoch=1)
        self.assertIn(os.path.join(ms.drafter.out, 'dflash.r5.pack'), plan['args'])
        self.assertIn('--spec-adaptive', plan['args'])


if __name__ == '__main__':
    unittest.main()
