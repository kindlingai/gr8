"""A fake model-loader for the tests: the local request API (docs/protocol.md in kindlingai/model-loader) on a unix
socket, and a store laid out as its store contract says (docs/design.md), filled from a release directory.

It holds whole segments, as model-loader does, and counts the bytes it copies. A PUT is served at once (the request is
ready when the PUT returns) unless `slow` is set: then a request is downloading for the first wait, and ready on the
next.
"""
import fnmatch
import hashlib
import http.server
import json
import os
import re
import socketserver
import threading


def glob_match(pattern, path):
    """model-loader's globs: * inside one path segment, ** over any number."""
    rx = ''
    for part in re.split(r'(\*\*/?|\*|\?)', pattern):
        if part in ('**', '**/'):
            rx += '.*'
        elif part == '*':
            rx += '[^/]*'
        elif part == '?':
            rx += '[^/]'
        else:
            rx += re.escape(part)
    return re.fullmatch(rx, path) is not None


class FakeModelLoader:
    def __init__(self, store, catalog, segment=64 << 10, slow=False):
        """catalog: {name: {revision: (release dir, commit)}}."""
        self.store, self.catalog, self.segment, self.slow = store, catalog, segment, slow
        self.copied, self.requests, self.puts, self.lock = 0, {}, [], threading.Lock()
        self.waits = {}
        os.makedirs(store, exist_ok=True)
        self.sock = os.path.join(store, 'model-loader.sock')
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def address_string(self):
                return 'unix'

            def reply(self, code, obj=None, text=None):
                body = json.dumps(obj).encode() if obj is not None else (text or '').encode()
                self.send_response(code)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_PUT(self):
                m = re.fullmatch(r'/requests/([A-Za-z0-9][A-Za-z0-9._-]{0,127})', self.path)
                if not m:
                    return self.reply(400, text='bad name')
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                code, out = fake.put(m.group(1), body)
                self.reply(code, out if code == 200 else None, None if code == 200 else out)

            def do_GET(self):
                m = re.fullmatch(r'/requests/([^?]+)(\?wait=.*)?', self.path)
                if self.path == '/requests':
                    return self.reply(200, list(fake.requests.values()))
                if not m or m.group(1) not in fake.requests:
                    return self.reply(404, text='404 page not found')
                st = fake.requests[m.group(1)]
                if m.group(2) and fake.slow and st['state'] == 'downloading':
                    st['state'] = 'ready'
                    return self.reply(504, dict(st, state='downloading'))
                code = {'ready': 200, 'invalid': 422}.get(st['state'], 504) if m.group(2) else 200
                self.reply(code, st)

            def do_DELETE(self):
                name = self.path.rsplit('/', 1)[-1]
                self.reply(204 if fake.requests.pop(name, None) else 404)

        class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.srv = S(self.sock, H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    @property
    def addr(self):
        return f'unix:{self.sock}'

    def files(self, root):
        out = {}
        for d, _, names in os.walk(root):
            for n in names:
                p = os.path.join(d, n)
                out[os.path.relpath(p, root)] = os.path.getsize(p)
        return out

    def put(self, name, body):
        with self.lock:
            self.puts.append((name, body))
            model, rev = body.get('model'), body.get('revision', '')
            revs = self.catalog.get(model)
            if not revs:
                return 400, f'invalid request: no catalog model {model}'
            if not rev:
                if len(revs) != 1:
                    return 400, f'invalid request: model {model} has several revisions; name one'
                rev = next(iter(revs))
            if rev not in revs:
                return 400, f'invalid request: no revision {rev} of {model}'
            src, commit = revs[rev]
            files = self.files(src)
            self.manifest(model, rev, commit, files)
            want = {p: None for p in files if any(glob_match(g, p) for g in body.get('include') or [])}
            err, total = '', 0
            for fr in body.get('ranges') or []:
                if fr['path'] not in files:
                    err = f'{fr["path"]}: not in the manifest'
                    continue
                if want.get(fr['path'], ()) is None:
                    continue
                for r in fr['ranges']:
                    if r['offset'] + r['length'] > files[fr['path']]:
                        err = f'{fr["path"]}: range past the end'
                    want.setdefault(fr['path'], []).append((r['offset'], r['length']))
            state = self.load_state(model, rev)
            for p, rs in want.items():
                total += self.hold(src, model, rev, p, files[p], rs, state)
            self.save_state(model, rev, state)
            st = {'name': name, 'model': model, 'revision': rev, 'include': body.get('include') or [],
                  'ranges': body.get('ranges') or [], 'priority': body.get('priority', 0),
                  'state': 'invalid' if err else ('downloading' if self.slow else 'ready'),
                  'bytes_total': total, 'bytes_done': 0 if self.slow else total,
                  'dir': os.path.join(self.store, 'models', model, rev), 'error': err}
            self.requests[name] = st
            return 200, st

    def manifest(self, model, rev, commit, files):
        p = os.path.join(self.store, 'manifests', model, rev + '.json')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        m = {'model': model, 'revision': rev, 'commit': commit,
             'files': [{'path': f, 'size': n, 'sha256': '', 'segment_size': self.segment} for f, n in sorted(files.items())]}
        with open(p + '.tmp', 'w') as f:
            json.dump(m, f)
        os.replace(p + '.tmp', p)

    def state_path(self, model, rev):
        return os.path.join(self.store, 'state', model, rev + '.json')

    def load_state(self, model, rev):
        try:
            with open(self.state_path(model, rev)) as f:
                return {x['path']: x for x in json.load(f)['files']}
        except (OSError, ValueError):
            return {}

    def save_state(self, model, rev, state):
        p = self.state_path(model, rev)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + '.tmp', 'w') as f:
            json.dump({'model': model, 'revision': rev, 'files': list(state.values())}, f)
        os.replace(p + '.tmp', p)

    def hold(self, src, model, rev, path, size, ranges, state):
        """Copy the segments ranges touch (every one when ranges is None) into the sparse store file; their bytes."""
        nseg = max(1, -(-size // self.segment))
        fs = state.setdefault(path, {'path': path, 'complete': False, 'segments_done': [False] * nseg})
        segs = range(nseg) if ranges is None else sorted({i for a, n in ranges
                                                          for i in range(a // self.segment, (a + n - 1) // self.segment + 1)})
        dst = os.path.join(self.store, 'models', model, rev, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size != size:
                os.ftruncate(fd, size)
            with open(os.path.join(src, path), 'rb') as f:
                for i in segs:
                    if fs['segments_done'][i]:
                        continue
                    f.seek(i * self.segment)
                    b = f.read(self.segment)
                    os.pwrite(fd, b, i * self.segment)
                    self.copied += len(b)
                    fs['segments_done'][i] = True
        finally:
            os.close(fd)
        fs['complete'] = all(fs['segments_done'])
        return len(segs) * self.segment

    def corrupt(self, model, rev, path, offset):
        """Flip one byte of a held file, as a disk fault would, leaving the state as it is."""
        p = os.path.join(self.store, 'models', model, rev, path)
        with open(p, 'r+b') as f:
            f.seek(offset)
            b = f.read(1)
            f.seek(offset)
            f.write(bytes([b[0] ^ 1]))

    def drop(self, model, rev, path):
        """Mark every segment of a file not held (an eviction)."""
        state = self.load_state(model, rev)
        if path in state:
            state[path]['segments_done'] = [False] * len(state[path]['segments_done'])
            state[path]['complete'] = False
            self.save_state(model, rev, state)
