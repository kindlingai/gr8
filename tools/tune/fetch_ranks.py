#!/usr/bin/env python3
"""Fetch only the exlr8 v2 expert halves that some TP ranks own, into sparse files at the original offsets.

The gr8 v2 reader (tools/convert/v2.py) preads each owned half at the absolute offset its index gives, so a file
that holds only those byte ranges (plus the safetensors header) converts the same as the full file.

python3 fetch_ranks.py --gr8 GR8_DIR --out DIR --nr 6 --ranks 0,1,2 --widths MANIFEST.json|all|k2,k3 \
        [--layers all|3-40,78] [--workers 16] [--verify]

- Index files, the manifests and the safetensors header of every width file are fetched whole.
- Progress is kept in DIR/.fetch/<file>.json (done ranges); a rerun skips them. Layers whose files are not on the
  hub yet are skipped and listed.
- --verify checks the xxhash64 of every wanted half after its layer is fetched (the converter checks them again).
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = 'mmastrac/glm-5.3-exlr8-k2-k3-k4'
HUB = 'https://huggingface.co'
MERGE_GAP = 64 << 10         # fetch the bytes between two wanted ranges when the gap is under 64 KiB
CHUNK = 64 << 20             # split a merged range into requests of at most 64 MiB


def token():
    t = os.environ.get('HF_TOKEN')
    if not t:
        sys.exit('set HF_TOKEN (read access to the exlr8 repo)')
    return t


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class Hub:
    """Resolves a repo file to its CDN URL once (no token sent to the CDN) and refreshes it on 403."""

    def __init__(self, repo):
        self.repo, self.tok = repo, token()
        self.urls, self.lock = {}, threading.Lock()
        self.noredir = urllib.request.build_opener(NoRedirect)

    def tree(self):
        out, url = {}, f'{HUB}/api/models/{self.repo}/tree/main?recursive=true'
        while url:
            req = urllib.request.Request(url, headers={'Authorization': f'Bearer {self.tok}'})
            with urllib.request.urlopen(req, timeout=60) as r:
                for x in json.load(r):
                    if x['type'] == 'file':
                        out[x['path']] = x['size']
                link = r.headers.get('Link', '')
            url = link.split(';')[0].strip('<> ') if 'rel="next"' in link else None
        return out

    def url(self, path, fresh=False):
        with self.lock:
            if fresh or path not in self.urls:
                req = urllib.request.Request(f'{HUB}/{self.repo}/resolve/main/{path}', method='HEAD',
                                             headers={'Authorization': f'Bearer {self.tok}'})
                try:
                    self.noredir.open(req, timeout=60)
                    loc = f'{HUB}/{self.repo}/resolve/main/{path}'      # small non-LFS file: served directly (200)
                except urllib.error.HTTPError as e:
                    if e.code not in (301, 302, 303, 307, 308):
                        raise
                    loc = e.headers['Location']
                    if loc.startswith('/'):
                        loc = HUB + loc
                self.urls[path] = loc
            return self.urls[path]

    def get(self, path, lo, n):
        """Bytes [lo, lo + n) of a repo file, with retries."""
        for attempt in range(8):
            try:
                u = self.url(path, fresh=attempt > 0)
                h = {'Range': f'bytes={lo}-{lo + n - 1}'}
                if u.startswith(HUB):
                    h['Authorization'] = f'Bearer {self.tok}'       # direct hub URL (non-LFS): needs the token
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
        self.done = set(map(tuple, json.load(open(self.p)))) if os.path.exists(self.p) else set()
        self.lock = threading.Lock()

    def add(self, r):
        with self.lock:
            self.done.add(tuple(r))
            tmp = self.p + '.tmp'
            json.dump(sorted(self.done), open(tmp, 'w'))
            os.replace(tmp, self.p)


def fetch_ranges(hub, out, path, size, ranges, workers):
    """Write the ranges of repo file `path` into the sparse local file of the same name. Returns bytes fetched."""
    local = os.path.join(out, os.path.basename(path))
    fd = os.open(local, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)
    prog = Progress(out, os.path.basename(path))
    todo = [r for r in ranges if tuple(r) not in prog.done]
    got = 0

    def one(r):
        b = hub.get(path, r[0], r[1])
        os.pwrite(fd, b, r[0])
        prog.add(r)
        return len(b)

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, r) for r in todo]):
            got += f.result()
    os.fsync(fd)
    os.close(fd)
    return got


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
    ap.add_argument('--gr8', required=True, help='gr8 checkout on graft/65-v2-loader (for tools.convert.v2)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nr', type=int, default=6)
    ap.add_argument('--ranks', required=True)
    ap.add_argument('--widths', required=True, help='a v2 manifest file (fetch its widths), all, or e.g. k2,k3')
    ap.add_argument('--layers', default='all')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--repo', default=REPO)
    a = ap.parse_args()
    sys.path.insert(0, a.gr8)
    from tools.convert import v2  # noqa: E402

    ranks = {int(r) for r in a.ranks.split(',')}
    os.makedirs(a.out, exist_ok=True)
    hub = Hub(a.repo)
    tree = hub.tree()
    where = {os.path.basename(p): p for p in tree}     # layer files live under experts/ on the hub
    for p, n in tree.items():                      # manifests: small, whole
        if p.startswith('manifests/') and not os.path.exists(os.path.join(a.out, os.path.basename(p))):
            open(os.path.join(a.out, os.path.basename(p)), 'wb').write(hub.get(p, 0, n))
    man = None
    if a.widths.endswith('.json'):
        man = json.load(open(a.widths))
        allk = None
    else:
        allk = [2, 3, 4] if a.widths == 'all' else [int(x.strip('k')) for x in a.widths.split(',')]

    skipped, total, t0 = [], 0, time.time()
    for l in parse_layers(a.layers):
        ip = f'exlr8-layer-{l:03d}.index.safetensors'
        if ip not in where:
            skipped.append((l, 'no index file'))
            continue
        local_ip = os.path.join(a.out, ip)
        if not os.path.exists(local_ip) or os.path.getsize(local_ip) != tree[where[ip]]:
            open(local_ip + '.part', 'wb').write(hub.get(where[ip], 0, tree[where[ip]]))
            os.replace(local_ip + '.part', local_ip)
        L = v2.V2Layer(a.out, l, verify=False)
        want = {}                                   # k -> list of (e, p, h)
        for e in L.experts + ([v2.SHARED] if L.shared_widths else []):
            if man is not None:
                if e == v2.SHARED:
                    s = man['layers'][str(l)].get('shared') or {}
                    if not s:
                        continue
                    gw, uw, dw = s['gate'], s['up'], s['down']
                else:
                    m = man['layers'][str(l)]
                    gw, uw, dw = m['gate'][e], m['up'][e], m['down'][e]
            for h in range(v2.NH):
                if v2.owner(e, h, l, a.nr) not in ranks:
                    continue
                for pi, p in enumerate(v2.PROJ):
                    if man is not None:
                        ks = [(gw[h] if p == 'gate' else uw[h] if p == 'up' else dw)]
                    else:
                        ks = allk
                    for k in ks:
                        if (e == v2.SHARED and k in L.shared_widths) or (e != v2.SHARED and k in L.widths):
                            want.setdefault(k, []).append((e, p, h))
        layer_bytes = 0
        for k, units in sorted(want.items()):
            wp = f'exlr8-layer-{l:03d}.k{k}.safetensors'
            if wp not in where:
                skipped.append((l, f'no k{k} file'))
                continue
            wp = where[wp]
            hdr = hub.get(wp, 0, 8)
            hn = int.from_bytes(hdr, 'little')
            rng = [(0, 8 + hn)]
            for e, p, h in units:
                off, n = (int(x) for x in L._t(e, k, 'index')[v2.PROJ.index(p), h])
                rng.append((off, n))
            layer_bytes += fetch_ranges(hub, a.out, wp, tree[wp], merge(rng), a.workers)
        if a.verify:
            L2 = v2.V2Layer(a.out, l, verify=True)
            for k, units in want.items():
                for e, p, h in units:
                    L2.half(e, p, h, k)
            L2.close()
        L.close()
        total += layer_bytes
        dt = time.time() - t0
        print(f'L{l:03d}: {layer_bytes / 1e9:.2f} GB, widths {sorted(want)}, units '
              f'{sum(len(u) for u in want.values())}{" verified" if a.verify else ""}; total {total / 1e9:.1f} GB '
              f'{total / max(dt, 1e-9) / 1e6:.0f} MB/s', flush=True)
    for l, why in skipped:
        print(f'SKIPPED L{l:03d}: {why}', flush=True)
    print(f'DONE {total / 1e9:.1f} GB in {time.time() - t0:.0f} s, {len(skipped)} skipped', flush=True)


if __name__ == '__main__':
    main()
