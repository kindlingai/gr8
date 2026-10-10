#!/usr/bin/env python3
"""Fetch only the exlr8 v2 expert halves that some TP ranks own, into sparse files at the original offsets.

The engine (engine/loader-rs/src/v2plan.rs) and the v2 reader (tools/convert/v2.py) read each owned half at the
absolute offset the index gives, so a file that holds only those byte ranges (plus the safetensors header) loads the
same as the full file.

python3 fetch_ranks.py --out DIR --nr 6 --ranks 0,1,2 --widths MANIFEST.json|all|k2,k3 \
        [--layers all|3-40,78] [--workers 16] [--verify] [--repo REPO] [--revision REV]

- The manifests, this TP's index slices for the ranks (index-slices/tp<N>/k<K>/rank<r>.safetensors) and the
  safetensors header of every width file are fetched whole. bin/g6load fetch-plan (engine/loader-rs) works out each
  file's ranges from the slices, so neither numpy nor the layer index files are needed.
- Progress is kept in DIR/.fetch/<file>.json (done ranges); a rerun skips them.
- --verify checks the xxhash64 of every wanted half once all are fetched (bin/g6load verify), and fetches the bad ones
  again once.
- Each file's wanted ranges are allocated in offset order before the parallel fetch writes them, so a rank's halves
  sit on disk in file order. Written in arrival order they scatter, and a loader reading them in order gets about
  3.3 GB/s from a GB10 node's NVMe instead of about 4.1.
- HF_TOKEN is sent when set (a private repo needs it); the published repo is public.

fetch_layers() is the same fetch as a function: tools/launch (./gr8 glm53 up --model <hf url>) calls it.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = 'kindlingai/glm-5.3-exlr8-k2-k3-k4'
HUB = 'https://huggingface.co'
MERGE_GAP = 64 << 10         # fetch the bytes between two wanted ranges when the gap is under 64 KiB
CHUNK = 64 << 20             # split a merged range into requests of at most 64 MiB


def token():
    return os.environ.get('HF_TOKEN') or None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class Hub:
    """Resolves a repo file to its CDN URL once (no token sent to the CDN) and refreshes it on 403."""

    def __init__(self, repo, revision='main', base=HUB):
        self.repo, self.rev, self.tok, self.base = repo, revision, token(), base
        self.urls, self.lock = {}, threading.Lock()
        self.noredir = urllib.request.build_opener(NoRedirect)
        self.entries = {}                   # path -> the tree entry (size, oid, lfs), filled by tree()

    def tree(self):
        out, url = {}, f'{self.base}/api/models/{self.repo}/tree/{self.rev}?recursive=true'
        while url:
            req = urllib.request.Request(url, headers=self.auth())
            with urllib.request.urlopen(req, timeout=60) as r:
                for x in json.load(r):
                    if x['type'] == 'file':
                        out[x['path']] = x['size']
                        self.entries[x['path']] = x
                link = r.headers.get('Link', '')
            url = link.split(';')[0].strip('<> ') if 'rel="next"' in link else None
        return out

    def auth(self):
        return {'Authorization': f'Bearer {self.tok}'} if self.tok else {}

    def url(self, path, fresh=False):
        with self.lock:
            if fresh or path not in self.urls:
                req = urllib.request.Request(f'{self.base}/{self.repo}/resolve/{self.rev}/{path}', method='HEAD',
                                             headers=self.auth())
                try:
                    self.noredir.open(req, timeout=60)
                    loc = f'{self.base}/{self.repo}/resolve/{self.rev}/{path}'   # small non-LFS file: served directly
                except urllib.error.HTTPError as e:
                    if e.code not in (301, 302, 303, 307, 308):
                        raise
                    loc = e.headers['Location']
                    if loc.startswith('/'):
                        loc = self.base + loc
                self.urls[path] = loc
            return self.urls[path]

    def get(self, path, lo, n):
        """Bytes [lo, lo + n) of a repo file, with retries."""
        for attempt in range(8):
            try:
                u = self.url(path, fresh=attempt > 0)
                h = {'Range': f'bytes={lo}-{lo + n - 1}'}
                if u.startswith(self.base):
                    h.update(self.auth())                          # direct hub URL (non-LFS): the token, if any
                req = urllib.request.Request(u, headers=h)
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206 and not (lo == 0 and r.status == 200):
                        raise RuntimeError(f'{path}: HTTP {r.status} for a range')
                    b = r.read()
                if len(b) != n:
                    raise RuntimeError(f'{path} [{lo}, +{n}): got {len(b)} bytes')
                return b
            except Exception as e:  # noqa: BLE001 - network: retry, then raise
                if attempt == 7:
                    raise
                print(f'  retry {attempt + 1} {path} @{lo}: {e}', file=sys.stderr, flush=True)
                time.sleep(2 ** attempt)


def merge(ranges):
    ranges = sorted(ranges)
    out = []
    for a, n in ranges:
        if out and a - (out[-1][0] + out[-1][1]) < MERGE_GAP:
            out[-1][1] = max(out[-1][1], a + n - out[-1][0])
        else:
            out.append([a, n])
    split = []
    for a, n in out:
        for o in range(0, n, CHUNK):
            split.append((a + o, min(CHUNK, n - o)))
    return split


class Progress:
    def __init__(self, out, name):
        self.p = os.path.join(out, '.fetch', name + '.json')
        os.makedirs(os.path.dirname(self.p), exist_ok=True)
        self.done = set()
        if os.path.exists(self.p):
            with open(self.p) as f:
                self.done = set(map(tuple, json.load(f)))
        self.lock = threading.Lock()

    def add(self, r):
        with self.lock:
            self.done.add(tuple(r))
            tmp = self.p + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(sorted(self.done), f)
            os.replace(tmp, self.p)


def fetch_ranges(hub, out, path, size, ranges, workers, on_bytes=None):
    """Write the ranges of repo file `path` into the sparse local file of the same name. Returns bytes fetched.
    on_bytes(n) is called after each range lands."""
    local = os.path.join(out, os.path.basename(path))
    fd = os.open(local, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)
    prog = Progress(out, os.path.basename(path))
    todo = [r for r in ranges if tuple(r) not in prog.done]
    got = 0
    for r in sorted(todo):              # blocks in file order (see the module docstring)
        os.posix_fallocate(fd, r[0], r[1])

    def one(r):
        b = hub.get(path, r[0], r[1])
        os.pwrite(fd, b, r[0])
        prog.add(r)
        if on_bytes:
            on_bytes(len(b))
        return len(b)

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, r) for r in todo]):
            got += f.result()
    os.fsync(fd)
    os.close(fd)
    return got


CHECKOUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def g6load():
    """bin/g6load (engine/loader-rs), which plans and checks the fetch: $G6LOAD, else the checkout's bin/, else cargo's
    target/release/."""
    for p in (os.environ.get('G6LOAD'), os.path.join(CHECKOUT, 'bin', 'g6load'),
              os.path.join(CHECKOUT, 'engine', 'loader-rs', 'target', 'release', 'g6load')):
        if p and os.access(p, os.X_OK):
            return p
    raise RuntimeError('no bin/g6load, which plans and checks the fetch: ./gr8 glm53 build, or cargo build --release '
                       '--manifest-path engine/loader-rs/Cargo.toml')


def g6load_run(cmd, slices, nr, rank, man=None, allk=None, layers=None, v2_dir=None):
    """g6load's JSON for one rank: fetch-plan, or verify of the halves already in v2_dir."""
    argv = [g6load(), cmd, '--slices', slices, '--tp', str(nr), '--rank', str(rank)]
    argv += ['--manifest', man] if man else ['--widths', ','.join(map(str, allk))]
    if layers:
        argv += ['--layers', ','.join(map(str, layers))]
    if v2_dir:
        argv += ['--v2', v2_dir]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode not in (0, 3):
        raise RuntimeError(f'g6load {cmd}: {r.stderr.strip() or r.returncode}')
    return json.loads(r.stdout)


def plan_files(hub, nr, ranks, slices, man=None, allk=None, layers=None):
    """[(repo path, size, merged ranges)] of the width files ranks own: each file's safetensors header and the halves
    g6load fetch-plan names (from the ranks' index slices, at the manifest's widths or every unit at allk)."""
    tree = hub.tree()
    where = {os.path.basename(p): p for p in tree}     # layer files live under experts/ on the hub
    want = {}
    for r in ranks:
        for f in g6load_run('fetch-plan', slices, nr, r, man, allk, layers)['files']:
            want.setdefault(f['name'], []).extend(tuple(x) for x in f['ranges'])
    jobs = []
    for name, rng in sorted(want.items()):
        if name not in where:
            raise RuntimeError(f'{name}: not in {hub.repo}@{hub.rev}')
        hn = int.from_bytes(hub.get(where[name], 0, 8), 'little')
        jobs.append((where[name], tree[where[name]], merge([(0, 8 + hn)] + rng)))
    return jobs


def todo_bytes(out, jobs):
    """Bytes of the jobs' ranges that are not fetched yet."""
    n = 0
    for path, _, ranges in jobs:
        done = Progress(out, os.path.basename(path)).done
        n += sum(r[1] for r in ranges if tuple(r) not in done)
    return n


def forget(out, jobs, bad):
    """Drop the journal entries of the ranges that hold a bad half, so the next fetch_ranges takes them again."""
    for path, _, ranges in jobs:
        name = os.path.basename(path)
        hit = [b for b in bad if b['file'] == name]
        if not hit:
            continue
        prog = Progress(out, name)
        for r in ranges:
            if any(r[0] < b['offset'] + b['nbytes'] and b['offset'] < r[0] + r[1] for b in hit):
                prog.done.discard(tuple(r))
        with open(prog.p, 'w') as f:
            json.dump(sorted(prog.done), f)


def fetch_layers(hub, out, nr, ranks, slices, man=None, allk=None, layers=None, workers=16, verify=False, log=print,
                 on_bytes=None, plan_only=False):
    """Fetch the expert halves `ranks` own (of nr) into out, as sparse files at their offsets. slices: the index slices
    directory (tp<N>/k<K>/rank<r>.safetensors) for those ranks. man: a width manifest file, or allk: widths to fetch
    every unit at. verify checks every half's xxhash64 (g6load verify), fetches the bad ones again once, and raises if
    one is still bad. Returns the bytes fetched, or with plan_only the bytes still to fetch."""
    jobs = plan_files(hub, nr, ranks, slices, man, allk, layers)
    if plan_only:
        return todo_bytes(out, jobs)
    t0, total = time.time(), 0
    for attempt in range(2):
        for wp, size, rng in jobs:
            total += fetch_ranges(hub, out, wp, size, rng, workers, on_bytes)
        if not verify:
            break
        bad = [b for r in ranks for b in g6load_run('verify', slices, nr, r, man, allk, layers, v2_dir=out)['bad']]
        if not bad:
            log(f'verified the halves of rank{"s" if len(ranks) > 1 else ""} {",".join(map(str, sorted(ranks)))}')
            break
        if attempt:
            raise RuntimeError(f'{len(bad)} halves still fail their xxhash64 after a second fetch, e.g. {bad[0]}')
        log(f'{len(bad)} halves failed their xxhash64: fetching them again')
        forget(out, jobs, bad)
    log(f'DONE {total / 1e9:.1f} GB in {time.time() - t0:.0f} s')
    return total


def parse_layers(spec):
    if spec == 'all':
        return list(range(3, 79))
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--nr', type=int, default=6)
    ap.add_argument('--ranks', required=True)
    ap.add_argument('--widths', required=True, help='a v2 manifest file (fetch its widths), all, or e.g. k2,k3')
    ap.add_argument('--layers', default='all')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--revision', default='main', help='branch, tag (v0.2) or commit to fetch from')
    a = ap.parse_args()
    ranks = {int(r) for r in a.ranks.split(',')}
    os.makedirs(a.out, exist_ok=True)
    hub = Hub(a.repo, a.revision)
    tree = hub.tree()
    for p, n in tree.items():                      # manifests and this TP's index slices for the ranks: whole
        m = re.fullmatch(rf'index-slices/tp{a.nr}/k\d/rank(\d+)\.safetensors', p)
        if p.startswith('manifests/') or (m and int(m.group(1)) in ranks):
            local = os.path.join(a.out, p if m else os.path.basename(p))
            if not os.path.exists(local) or os.path.getsize(local) != n:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                with open(local, 'wb') as f:
                    f.write(hub.get(p, 0, n))
    man = a.widths if a.widths.endswith('.json') else None
    allk = None if man else [2, 3, 4] if a.widths == 'all' else [int(x.strip('k')) for x in a.widths.split(',')]
    fetch_layers(hub, a.out, a.nr, ranks, os.path.join(a.out, 'index-slices'), man=man, allk=allk,
                 layers=None if a.layers == 'all' else parse_layers(a.layers), workers=a.workers, verify=a.verify,
                 log=lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()
