"""The model-loader source: ./gr8 glm53 up --model https://huggingface.co/<org>/<repo>[@rev] --fetch model-loader.

The box's model-loader (github.com/kindlingai/model-loader) fetches the weights; gr8 only says which pieces this rank
needs and reads them in place from the model-loader store, which it may mount read-only:

    MODEL_CACHE                      the store (docs/design.md, "Store contract for consumers")
        models/<model>/<revision>/   the release's files at their manifest paths, a piece at its original offset
        manifests/<model>/<revision>.json, state/<model>/<revision>.json

<model> is the catalog name of the release (--model-loader-model, by default the repo's name after the org), and
<revision> its catalog revision: @rev of --model when the URL names one, else the catalog's only revision of it.

gr8 puts one named request (PUT /requests/gr8-<model>, docs/protocol.md) on MODEL_LOADER, the local API, and replaces it
as it learns more: first the manifests and the tokenizer and config files, then this rank's index slices and the dense
files, then the byte ranges bin/g6load fetch-plan gives (with --layout, the halves the layout gives the rank, and every
rank's slices at the TP). It waits until the request is ready. The request pins its pieces, so model-loader does not
evict them while gr8 runs. It is put at every boot, so a boot from a store that holds the pieces already fetches nothing
and needs no network beyond the local socket.

model-loader verifies every segment it holds (sha256 against the manifest or the origin). gr8 still checks the xxhash64
of every expert half before boot (bin/g6load verify), once per rank, variant and layout; the receipt of that check is
kept under G6_HOME, since the store is model-loader's alone. gr8's checksums cover each half's per-channel scales as
well as its bytes, so they cannot go into the request's xxh64 fields.

model-loader holds whole 64 MiB segments. A rank's expert halves are spread over every segment of a width file (the half
rule interleaves the ranks), so the store holds every expert file the rank's widths use, in full.
"""
import http.client
import json
import os
import re
import socket
import time
import urllib.parse

import modelset

META_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
WAIT = 60                  # seconds per GET ?wait=: one progress line each
STATE_LAG = 5              # seconds the store's state may lag a request model-loader reports ready (Store.settled)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__('localhost', timeout=timeout)
        self.path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.path)
        except OSError:
            s.close()
            raise
        self.sock = s


class Client:
    """The local request API of model-loader. addr: unix:PATH, or host:port (TCP). token, when set, is sent as
    Authorization: Bearer; TCP needs one, as model-loader serves TCP only with a token (MODEL_LOADER_LOCAL_TOKEN)."""

    def __init__(self, addr, token=None):
        self.addr, self.token = addr, token
        if not addr.startswith('unix:') and not token:
            raise modelset.ModelError(f'MODEL_LOADER {addr}: a TCP local API needs MODEL_LOADER_LOCAL_TOKEN, the token '
                                      'model-loader serves it with (or use unix:PATH)')

    def conn(self, timeout):
        if self.addr.startswith('unix:'):
            return UnixHTTPConnection(self.addr[5:], timeout)
        host = self.addr[7:] if self.addr.startswith('http://') else self.addr
        return http.client.HTTPConnection(host.rstrip('/'), timeout=timeout)

    def call(self, method, path, body=None, timeout=30):
        c = self.conn(timeout)
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        try:
            c.request(method, path, body=None if body is None else json.dumps(body).encode(), headers=headers)
            r = c.getresponse()
            data = r.read()
        except OSError as e:
            raise modelset.ModelError(f'model-loader at {self.addr}: {e} (is it running? MODEL_LOADER names its local '
                                      'API, by default the socket model-loader.sock in the store)') from None
        finally:
            c.close()
        if r.status == 401:
            raise modelset.ModelError(f'model-loader at {self.addr}: 401: MODEL_LOADER_LOCAL_TOKEN is '
                                      f'{"wrong" if self.token else "unset, and model-loader wants one"}')
        try:
            out = json.loads(data) if data and data.lstrip()[:1] in (b'{', b'[') else None
        except ValueError:
            out = None
        return r.status, out, data.decode(errors='replace').strip()

    def put(self, name, body):
        code, st, text = self.call('PUT', f'/requests/{urllib.parse.quote(name)}', body)
        if code != 200 or not isinstance(st, dict):
            raise modelset.ModelError(f'model-loader refused request {name}: HTTP {code}: {text[:500]}')
        return st

    def get(self, name):
        """The request's status, or None when there is none."""
        code, st, text = self.call('GET', f'/requests/{urllib.parse.quote(name)}')
        if code == 404:
            return None
        if code != 200 or not isinstance(st, dict):
            raise modelset.ModelError(f'model-loader: request {name}: HTTP {code}: {text[:500]}')
        return st

    def catalog(self):
        """The catalog's models (GET /catalog: name, revision, source), or None from a model-loader without it."""
        code, out, text = self.call('GET', '/catalog')
        if code == 404:
            return None
        if code != 200 or not isinstance(out, dict) or not isinstance(out.get('models'), list):
            raise modelset.ModelError(f'model-loader at {self.addr}: GET /catalog: HTTP {code}: {text[:500]}')
        return [m for m in out['models'] if isinstance(m, dict)]

    def wait(self, name, seconds):
        code, st, text = self.call('GET', f'/requests/{urllib.parse.quote(name)}?wait={int(seconds)}s',
                                   timeout=seconds + 30)
        if code not in (200, 422, 504) or not isinstance(st, dict):
            raise modelset.ModelError(f'model-loader: request {name}: HTTP {code}: {text[:500]}')
        return st


def safe_path(s):
    """Is s a relative path that stays where it is joined: a model name or revision under the store?"""
    return (isinstance(s, str) and '\\' not in s and '\0' not in s
            and all(c not in ('', '.', '..') for c in s.split('/')))


def merge(ranges, seg):
    """Sorted, merged [offset, length] of ranges, joining two whose gap is under one segment: model-loader holds whole
    segments, so the join adds no segment to what it holds, and the request stays small (a rank's halves of one file
    are some thousands of ranges)."""
    out = []
    for a, n in sorted((int(a), int(n)) for a, n in ranges):
        if out and a - (out[-1][0] + out[-1][1]) < seg:
            out[-1][1] = max(out[-1][1], a + n - out[-1][0])
        else:
            out.append([a, n])
    return [tuple(x) for x in out]


class Store:
    """The read side of the store contract: manifest, state, and whether a range is verified on disk."""

    def __init__(self, root, model, revision):
        self.root, self.model, self.revision = root, model, revision
        self.dir = os.path.join(root, 'models', model, revision)

    def _json(self, kind):
        p = os.path.join(self.root, kind, self.model, self.revision + '.json')
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def manifest(self):
        m = self._json('manifests')
        return {f['path']: f for f in m['files']} if m else None, (m or {}).get('commit') or ''

    def held(self):
        """{path: file state} of what the store holds now."""
        s = self._json('state')
        return {f['path']: f for f in (s or {}).get('files', [])}

    @staticmethod
    def verified(fm, fs, a, n):
        """Is [a, a + n) of the file verified on disk (the contract's rule)?"""
        if not fs or fs.get('whole_failed'):
            return False
        if fs.get('complete'):
            return True
        seg = int(fm.get('segment_size') or 0)
        if not seg or n <= 0 or a < 0 or a + n > fm['size']:
            return False
        nseg = -(-fm['size'] // seg)
        runs = Store.runs(fs, nseg)
        if runs == [[0, nseg]]:                 # every segment held, not complete: not used until its whole check
            return False
        first, last = a // seg, (a + n - 1) // seg
        return any(lo <= first and last < hi for lo, hi in runs)

    @staticmethod
    def runs(fs, nseg):
        """The held segments as sorted, joined half-open runs: `held` (the contract's field), else `segments_done` (a
        state an older model-loader wrote)."""
        if 'held' in fs:
            out = []
            for lo, hi in sorted((int(x[0]), int(x[1])) for x in fs['held'] or []):
                lo, hi = max(lo, 0), min(hi, nseg)
                if lo >= hi:
                    continue
                if out and lo <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], hi)
                else:
                    out.append([lo, hi])
            return out
        out, done = [], fs.get('segments_done') or []
        for i, d in enumerate(done[:nseg]):
            if d and out and out[-1][1] == i:
                out[-1][1] = i + 1
            elif d:
                out.append([i, i + 1])
        return out

    def settled(self, files, ranges, wait=STATE_LAG):
        """missing(), after model-loader said the request is ready: it saves a revision's state at most once a second
        while small segments land (docs/design.md, "Small segments"), so the state on disk may lag the answer by
        that. Read again until it holds every piece or the wait is over."""
        end = time.monotonic() + wait
        while True:
            bad = self.missing(files, ranges)
            if not bad or time.monotonic() >= end:
                return bad
            time.sleep(0.2)

    def missing(self, files, ranges):
        """The pieces the store does not hold verified: whole files, and {path: [[a, n]]}."""
        man, _ = self.manifest()
        if man is None:
            return ['no manifest yet']
        held, out = self.held(), []
        for p in files:
            if p not in man or not (held.get(p) or {}).get('complete') or held[p].get('whole_failed'):
                out.append(p)
        for p, rs in ranges.items():
            if p not in man or not all(self.verified(man[p], held.get(p), a, n) for a, n in rs):
                out.append(p)
        return out


class Release:
    """The exlr8 release at one revision, as the box's model-loader holds it. The same interface as hfsource.Release."""

    def __init__(self, store, addr, model, repo, rev, work, name, log=print, layout=None, token=None):
        if not safe_path(model):
            raise modelset.ModelError(f'MODEL_LOADER_MODEL {model!r}: not a catalog name')
        self.client, self.store_root, self.model, self.hf_repo = Client(addr, token), store, model, repo
        self.want_rev, self.work, self.request_name, self.log = rev, work, name, log
        self.layout = layout
        self.layout_sha256 = modelset.sha256_file(layout) if layout else None
        self.steering_profile = None
        self.include, self.ranges = list(META_FILES) + ['manifests/*.json'], {}
        self.priority = 10
        self.revision = rev
        old = self.client.get(name)
        if old and old.get('model') == model and (not rev or old.get('revision') == rev):
            # the last boot's pieces stay pinned until fetch_rank puts this boot's: a restart evicts nothing it needs
            self.revision = old.get('revision') or rev
            self.include += old.get('include') or []
            self.ranges = {fr['path']: [(r['offset'], r['length']) for r in fr['ranges']]
                           for fr in old.get('ranges') or []}
        st = self.request()                              # the manifests and the tokenizer: the model set needs them
        if (rev and st.get('revision') != rev) or not safe_path(st.get('revision')):
            want = rev or "the catalog's only revision"         # the revision is joined into store paths
            raise modelset.ModelError(f'model-loader answered {model} revision {st.get("revision")!r} for {want}')
        self.revision = st['revision']
        self.store = Store(store, model, self.revision)
        self.dir = self.store.dir
        self.v2 = os.path.join(self.dir, 'experts')
        self.slices = os.path.join(self.dir, 'index-slices')
        self.tokenizer = self.dir
        self.manifests = os.path.join(self.dir, 'manifests')
        man, self.commit = self.store.manifest()
        if man is None:
            raise modelset.ModelError(f'{self.where()}: ready, but the store has no manifest '
                                      f'({os.path.join(store, "manifests", model)}): is MODEL_CACHE the store?')
        if re.fullmatch(r'[0-9a-f]{40}', rev or '') and self.commit and self.commit != rev:
            raise modelset.ModelError(f'--model names commit {rev}, and model-loader\'s {model}@{self.revision} is '
                                      f'commit {self.commit}')
        self.commit = self.commit or self.revision
        self.files = man

    def where(self):
        return f'model-loader {self.model}' + (f'@{self.revision}' if self.revision else '')

    @property
    def label(self):
        return self.hf_repo

    def __str__(self):
        return f'{self.hf_repo}@{self.commit[:12]}'

    @property
    def id(self):
        """As hfsource.Release's: boxes that fetch the same commit either way agree."""
        return f'{self.hf_repo}@{self.commit}'

    def body(self):
        b = {'model': self.model, 'include': sorted(set(self.include)), 'priority': self.priority}
        if self.revision:
            b['revision'] = self.revision
        if self.ranges:
            b['ranges'] = [{'path': p, 'ranges': [{'offset': a, 'length': n} for a, n in rs]}
                           for p, rs in sorted(self.ranges.items())]
        return b

    def request(self, what='the release\'s manifests and tokenizer'):
        """Put the request with everything this boot needs so far, and wait until model-loader holds it."""
        st = self.client.put(self.request_name, self.body())
        t0, said = time.monotonic(), None
        while st.get('state') not in ('ready', 'invalid'):
            line = (f'{self.where()}: {what}: {st.get("state")}, {st.get("bytes_done", 0) / 1e9:.1f} of '
                    f'{st.get("bytes_total", 0) / 1e9:.1f} GB' + (f' ({st["error"]})' if st.get('error') else ''))
            if line != said:
                self.log(line)
                said = line
            st = self.client.wait(self.request_name, WAIT)
        if st['state'] == 'invalid':
            raise modelset.ModelError(f'{self.where()}: request {self.request_name} is invalid: {st.get("error")}')
        if time.monotonic() - t0 > 1:
            self.log(f'{self.where()}: {what}: held, {st.get("bytes_total", 0) / 1e9:.1f} GB')
        return st

    def fetch_meta(self, steering=None):
        if steering:
            raise modelset.ModelError('--steering-profile hf:... with --fetch model-loader: not yet (#179)')
        missing = [n for n in META_FILES if n not in self.files]
        if missing:
            raise modelset.ModelError(f'{self} lacks {", ".join(missing)} at its root')

    def whole_files(self, tp, rank):
        """The rank's index slices (every rank's under a layout), and the dense and vocab files without the vision
        tower, as hfsource.Release.whole_files."""
        r = r'\d+' if self.layout else str(rank)
        slices = sorted(p for p in self.files if re.fullmatch(rf'index-slices/tp{tp}/k\d/rank{r}\.safetensors', p))
        if not slices:
            raise modelset.ModelError(f'{self} has no index slices for TP={tp}')
        dense = sorted(p for p in self.files if p.startswith('dense/') and p.endswith('.safetensors')
                       and not p.startswith('dense/vision-'))
        return slices + dense

    def key(self, tp, rank, variant):
        lay = f'.layout-{self.layout_sha256[:12]}' if self.layout else ''
        return f'tp{tp}.rank{rank}.{variant.name}-{variant.sha256[:12]}{lay}'

    def receipt(self, tp, rank, variant):
        safe = re.sub(r'[^A-Za-z0-9._-]', '_', f'{self.model}@{self.revision}')
        return os.path.join(self.work, safe, self.key(tp, rank, variant) + '.json')

    @staticmethod
    def layers(variant):
        with open(variant.path) as f:
            return sorted(int(x) for x in json.load(f)['layers'])

    def plan(self, tp, rank, variant):
        """{width file path: [[offset, n]]} of this rank's expert halves (bin/g6load fetch-plan), without headers."""
        import fetch_ranks as fr
        layers = self.layers(variant)
        try:
            got = fr.g6load_run('fetch-plan', self.slices, tp, rank, variant.path, None, layers, layout=self.layout)
        except (RuntimeError, ValueError) as e:
            raise modelset.ModelError(f'{self}: {e}') from None
        where = {os.path.basename(p): p for p in self.files if p.startswith('experts/')}
        out = {}
        for f in got['files']:
            if f['name'] not in where:
                raise modelset.ModelError(f'{f["name"]}: not in {self}')
            out[where[f['name']]] = [tuple(x) for x in f['ranges']]
        return out, got['bytes']

    pins = True        # supervisor: put the request at every boot, so model-loader keeps the pieces while gr8 runs

    def check(self, tp, rank, variant):
        """Problems with this rank's part, from the store alone (no model-loader call): [] once a fetch_rank verified
        it and the store still holds every piece. The cheapest check: sizes and segment state, not the bytes."""
        try:
            with open(self.receipt(tp, rank, variant)) as f:
                rec = json.load(f)
            if rec.get('commit') != self.commit:
                raise ValueError
        except (OSError, ValueError):
            lay = f', layout {self.layout_sha256[:12]}' if self.layout else ''
            return [f'{self} rank {rank} of TP={tp}, variant {variant.name}{lay}, is not in the model-loader store yet']
        ranges = {p: [tuple(r) for r in rs] for p, rs in rec['ranges'].items()}
        bad = [f'{p}: not held in {self.dir}' for p in self.store.missing(rec['whole'], ranges)]
        return bad

    def fetch_rank(self, tp, rank, variant, workers=None):
        """Put this rank's pieces as the request, wait until model-loader holds them, then check every expert half's
        xxhash64 (once per rank, variant and layout: a receipt under G6_HOME records it)."""
        import fetch_ranks as fr
        whole = self.whole_files(tp, rank)
        self.include, self.ranges = list(META_FILES) + ['manifests/*.json'] + whole, {}   # this boot's pieces only
        self.request(f'rank {rank} of TP={tp}: the index slices and dense files')
        halves, nbytes = self.plan(tp, rank, variant)
        self.ranges = {p: merge(rs + [(0, 8)], int(self.files[p].get('segment_size') or 1)) for p, rs in halves.items()}
        st = self.request(f'rank {rank} of TP={tp}, variant {variant.name}: {nbytes / 1e9:.1f} GB of expert halves')
        for p, rs in list(self.ranges.items()):           # each width file's safetensors header, now that it is held
            with open(os.path.join(self.dir, p), 'rb') as f:
                hn = int.from_bytes(f.read(8), 'little')
            self.ranges[p] = merge(rs + [(0, 8 + hn)], int(self.files[p].get('segment_size') or 1))
        st = self.request(f'rank {rank} of TP={tp}: the width files\' headers')
        self.log(f'{self}: rank {rank} of TP={tp}, variant {variant.name}: {nbytes / 1e9:.1f} GB of expert halves, '
                 f'{st.get("bytes_total", 0) / 1e9:.1f} GB held for this request in {self.dir}')
        mine = {'commit': self.commit, 'whole': whole,
                'ranges': {p: [list(x) for x in rs] for p, rs in sorted(self.ranges.items())}}
        rec = self.receipt(tp, rank, variant)
        try:
            with open(rec) as f:
                same = json.load(f) == mine
        except (OSError, ValueError):
            same = False
        if not same:
            try:
                r = fr.g6load_run('verify', self.slices, tp, rank, variant.path, None, self.layers(variant),
                                  v2_dir=self.v2, layout=self.layout)
            except (RuntimeError, ValueError) as e:
                raise modelset.ModelError(f'{self}: {e}') from None
            if r['bad']:
                b = r['bad'][0]
                raise modelset.ModelError(f'{self}: {len(r["bad"])} of {r["checked"]} expert halves in {self.v2} fail '
                                          f'their xxhash64, e.g. {b["file"]} at {b["offset"]} (layer {b["layer"]}, '
                                          f'expert {b["expert"]}): model-loader -verify-on-start re-checks what it '
                                          'holds')
            self.log(f'{self}: verified the {r["checked"]} expert halves of rank {rank}')
            os.makedirs(os.path.dirname(rec), exist_ok=True)
            with open(rec + '.tmp', 'w') as f:
                json.dump(mine, f, indent=1, sort_keys=True)
            os.replace(rec + '.tmp', rec)
        bad = self.store.settled(whole, self.ranges)
        if bad:
            raise modelset.ModelError(f'{self}: model-loader reports the request ready, and the store does not hold '
                                      f'{", ".join(bad[:3])}')


def catalog_name(client, repo, rev, key, log=print):
    """The name model-loader's catalog gives the Hugging Face repo, looked up in GET /catalog, so no box has to be told
    it. rev: the revision the URL names, or '' for none. Several entries for the repo (different revisions) are told
    apart by rev; with none named they must all have one name. A model-loader without GET /catalog: the repo's name
    after the org, as before it."""
    models = client.catalog()
    if models is None:
        name = repo.split('/')[-1]
        log(f'model-loader at {client.addr} lists no catalog (GET /catalog): taking {repo} to be {name!r} in it; '
            f'{key} names it otherwise')
        return name
    want = repo.lower()
    hits = [m for m in models if (m.get('source') or {}).get('type') == 'huggingface'
            and str((m.get('source') or {}).get('repo') or '').lower() == want and isinstance(m.get('name'), str)]
    if rev and any(m.get('revision') == rev for m in hits):
        hits = [m for m in hits if m.get('revision') == rev]
    names = sorted({m['name'] for m in hits})
    if len(names) == 1:
        return names[0]
    if not names:
        listed = ', '.join(sorted(str((m.get('source') or {}).get('repo') or m.get('name')) for m in models)) or 'nothing'
        raise modelset.ModelError(f'model-loader\'s catalog has no entry for https://huggingface.co/{repo} (it lists '
                                  f'{listed}): add one to its catalog.yaml, or set {key} to the catalog name')
    raise modelset.ModelError(f'model-loader\'s catalog names https://huggingface.co/{repo} as {", ".join(names)}: name '
                              f'the revision in the URL (@rev), or set {key}')


def client_of(o):
    store = o.get('MODEL_CACHE')
    addr = o.get('MODEL_LOADER') or f'unix:{os.path.join(store, "model-loader.sock")}'
    return Client(addr, os.environ.get('MODEL_LOADER_LOCAL_TOKEN') or None)


def of(o, repo, rev, log=print):
    """The model-loader Release for --fetch model-loader and --model https://huggingface.co/<repo>[@rev]."""
    store = o.get('MODEL_CACHE')
    url = o.get('MODEL') or ''
    named = bool(re.search(r'(@|/tree/)[^/]+/?$', url))            # parse_url says main when the URL names none
    client = client_of(o)
    model = o.get('MODEL_LOADER_MODEL') or catalog_name(client, repo, rev if named else '', 'MODEL_LOADER_MODEL', log)
    work = os.path.join(os.path.abspath(o.get('G6_HOME')), 'run', 'model-loader')
    return Release(store, client.addr, model, repo, rev if named else '', work, f'gr8-{o.get("GR8_MODEL") or "glm53"}',
                   log=log, layout=o.get('LAYOUT'), token=client.token)
