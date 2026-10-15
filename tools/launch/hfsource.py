"""The Hugging Face model source: ./gr8 glm53 up --model https://huggingface.co/<org>/<repo>[@rev | /tree/<rev>].

Each box fetches only its part of the exlr8 release into --model-cache, and nothing else:

    <cache>/hf/<org>--<repo>/refs/<rev>        the commit a revision resolved to (prepare resolves it again)
    <cache>/hf/<org>--<repo>/<commit>/
        manifests/                             the width manifests (home, down1, up1, bit*)
        tokenizer/                             config, generation config, tokenizer and chat template
        experts/                               this rank's expert halves at their file offsets (sparse files), and
                                               every layer's index file (tools/tune/fetch_ranks.py)
        index-slices/tp<N>/k<K>/rank<r>.safetensors
        dense/                                 the dense and vocab files
        complete/tp<N>.rank<r>.<variant>.json  written last, once everything above is verified

Every whole file is checked before it is renamed into place: against the release's hashes/<path>.sha256 (for an
experts/ file, hashes/<name>.sha256), else the tree's LFS sha256 or git blob id. Every expert half is checked against its xxhash64 in the index file. A
whole file resumes from its .part, and the expert halves resume from fetch_ranks.py's journal. HF_TOKEN is sent only
when it is set. GR8_HF_ENDPOINT points the fetches at another hub (the tests' fake one).

The tokenizer and config files come from the release too, so a boot touches one repo.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import modelset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The DFlash2 drafter for GLM 5.3. The drafter's licence (CC BY-NC-ND 4.0) forbids sharing a converted copy, so each box
# fetches the drafter from its own repo and converts its rank with bin/g6load dflash2 (the same bytes as
# tools/convert/dflash2.py, without numpy).
DRAFTER_URL = 'https://huggingface.co/incoai/GLM-5.3-DFlash2'
META_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
CHUNK = 64 << 20


def endpoint():
    return os.environ.get('GR8_HF_ENDPOINT', 'https://huggingface.co').rstrip('/')


def parse_url(model):
    """(repo, revision) for an https:// model source, else None (a local path)."""
    if not (model or '').startswith('https://'):
        return None
    m = re.fullmatch(r'https://huggingface\.co/([\w.-]+/[\w.-]+?)(?:@([\w.-]+)|/tree/([\w.-]+))?/?', model)
    if not m:
        raise modelset.ModelError(f'--model {model}: want https://huggingface.co/<org>/<repo>, optionally with @<rev> '
                                  'or /tree/<rev>')
    return m.group(1), m.group(2) or m.group(3) or 'main'


def converter():
    """The drafter converter's command: bin/g6load dflash2."""
    try:
        return [fetch_ranks().g6load(), 'dflash2']
    except RuntimeError as e:
        raise modelset.ModelError(f'converting the DFlash2 drafter: {e}') from None


def fetch_ranks():
    """tools/tune/fetch_ranks.py: the range GETs, with bin/g6load planning and checking them."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from tools.tune import fetch_ranks as fr
    return fr


def git_blob_id(path):
    h = hashlib.sha1(f'blob {os.path.getsize(path)}\0'.encode())
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


TREE = '.tree.json'        # in a commit's cache directory: {path: [hash, size]} of the commit's files, for reuse()


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_json(path, obj):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f)
    os.replace(path + '.tmp', path)


class Meter:
    """GB done of the total and the rate, one line every 30 s."""

    def __init__(self, total, log, what):
        self.total, self.log, self.what = total, log, what
        self.done, self.t0, self.said, self.lock = 0, time.monotonic(), time.monotonic(), threading.Lock()

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.monotonic()
            if now - self.said >= 30:
                self.said = now
                self.line()

    def line(self):
        rate = self.done / max(time.monotonic() - self.t0, 1e-9) / 1e6
        self.log(f'{self.what}: {self.done / 1e9:.1f} of {self.total / 1e9:.1f} GB ({rate:.0f} MB/s)')


class Repo:
    """One repo at one commit on the hub."""

    def __init__(self, cache, repo, rev, refresh, log):
        self.repo, self.rev, self.log = repo, rev, log
        self.root = os.path.join(cache, 'hf', repo.replace('/', '--'))
        self.commit = self.resolve(refresh)
        self.hub = fetch_ranks().Hub(repo, self.commit, base=endpoint())
        self._tree = None

    def resolve(self, refresh):
        """The commit rev names: a full commit as it is; otherwise the one recorded at the last resolve, which
        prepare (refresh) asks the hub for again."""
        if re.fullmatch(r'[0-9a-f]{40}', self.rev):
            return self.rev
        ref = os.path.join(self.root, 'refs', self.rev)
        if os.path.isfile(ref) and not refresh:
            with open(ref) as f:
                return f.read().strip()
        tok = os.environ.get('HF_TOKEN')
        req = urllib.request.Request(f'{endpoint()}/api/models/{self.repo}/revision/{self.rev}',
                                     headers={'Authorization': f'Bearer {tok}'} if tok else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                commit = json.load(r)['sha']
        except (OSError, ValueError, KeyError) as e:
            if os.path.isfile(ref):
                with open(ref) as f:
                    commit = f.read().strip()
                self.log(f'{self.repo}@{self.rev}: the hub did not answer ({e}); using the recorded commit {commit[:12]}')
                return commit
            raise modelset.ModelError(f'cannot resolve {self.repo}@{self.rev} on {endpoint()}: {e}')
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        with open(ref + '.tmp', 'w') as f:
            f.write(commit + '\n')
        os.replace(ref + '.tmp', ref)
        self.log(f'{self.repo}@{self.rev} is commit {commit[:12]}')
        return commit

    def tree(self):
        if self._tree is None:
            self._tree = self.hub.tree()
        return self._tree

    def ids(self):
        """{path: [hash, size]} of this commit's files, as the hub lists them: the LFS sha256, else the git blob id."""
        t = self.tree()
        return {p: [(e.get('lfs') or {}).get('oid') or e['oid'], e['size']] for p, e in self.hub.entries.items() if p in t}

    def expected(self, path):
        """(kind, value, size) a file must match: the release's hashes/<path>.sha256, else the LFS sha256, else the
        git blob id."""
        t = self.tree()
        for hp in (f'hashes/{path}.sha256', f'hashes/{os.path.basename(path)}.sha256'):   # experts/ files: by name
            if hp in t:
                sha, size = self.hub.get(hp, 0, t[hp]).decode().split()[:2]
                return 'sha256', sha, int(size)
        e = self.hub.entries[path]
        if e.get('lfs'):
            return 'sha256', e['lfs']['oid'], e['lfs']['size']
        return 'git', e['oid'], e['size']

    def matches(self, path, local):
        kind, want, size = self.expected(path)
        if os.path.getsize(local) != size:
            return False
        return (modelset.sha256_file(local) if kind == 'sha256' else git_blob_id(local)) == want

    def fetch(self, path, dest, meter=None):
        """A whole file into dest, through dest.part (resumed from its size), checked before the rename."""
        if os.path.isfile(dest):
            return
        size = self.tree()[path]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + '.part'
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.unlink(part)
            have = 0
        with open(part, 'ab') as f:
            while have < size:
                n = min(CHUNK, size - have)
                f.write(self.hub.get(path, have, n))
                have += n
                if meter:
                    meter.add(n)
        if not self.matches(path, part):
            os.unlink(part)
            raise modelset.ModelError(f'{self.repo}@{self.commit[:12]} {path}: the fetched file does not match its hash')
        os.replace(part, dest)


class Release:
    """The exlr8 release at one commit, in this box's cache."""

    def __init__(self, cache, repo, rev, refresh=False, log=print):
        self.repo = Repo(cache, repo, rev, refresh, log)
        self.log = log
        self.dir = os.path.join(self.repo.root, self.repo.commit)
        self.v2 = os.path.join(self.dir, 'experts')
        self.slices = os.path.join(self.dir, 'index-slices')
        self.tokenizer = os.path.join(self.dir, 'tokenizer')
        self.manifests = os.path.join(self.dir, 'manifests')

    @property
    def commit(self):
        return self.repo.commit

    def __str__(self):
        return f'{self.repo.repo}@{self.commit[:12]}'

    @property
    def id(self):
        return f'{self.repo.repo}@{self.commit}'

    def fetch_meta(self):
        """The manifests and the tokenizer and config files. Once there, a boot needs no network for them."""
        done = os.path.join(self.dir, 'meta.json')
        if os.path.isfile(done):
            return
        t = self.repo.tree()
        for p in sorted(t):
            if p.startswith('manifests/'):
                self.repo.fetch(p, os.path.join(self.dir, p))
        missing = [n for n in META_FILES if n not in t]
        if missing:
            raise modelset.ModelError(f'{self} lacks {", ".join(missing)} at its root')
        for name in META_FILES:
            self.repo.fetch(name, os.path.join(self.tokenizer, name))
        with open(done + '.tmp', 'w') as f:
            json.dump({'release': f'{self.repo.repo}@{self.commit}', 'files': list(META_FILES)}, f, indent=1)
        os.replace(done + '.tmp', done)

    def whole_files(self, tp, rank):
        """The release files this rank loads whole: its index slices, and the dense and vocab files."""
        t = self.repo.tree()
        slices = sorted(p for p in t if re.fullmatch(rf'index-slices/tp{tp}/k\d/rank{rank}\.safetensors', p))
        if not slices:
            raise modelset.ModelError(f'{self} has no index slices for TP={tp}')
        # TODO: every rank fetches every dense and vocab file (about 11 GB). Once the dense loader says which
        # sharding units a rank reads, fetch only those byte ranges here.
        dense = sorted(p for p in t if p.startswith('dense/') and p.endswith('.safetensors'))
        return slices + dense

    def reuse(self):
        """Hard-link the files an older commit of this release left in the cache, where the hub gives the file the same
        hash and size at both commits, so a new commit (a manifest added, say) does not fetch the rest again. An
        expert file takes its fetch journal along, and its halves are still checked against the index file. A cache
        directory written before this records no file list: it is asked from the hub once."""
        mine = self.repo.ids()
        os.makedirs(self.dir, exist_ok=True)
        save_json(os.path.join(self.dir, TREE), mine)
        n = 0
        for c in sorted(os.listdir(self.repo.root)):
            old_dir = os.path.join(self.repo.root, c)
            if c == self.commit or not re.fullmatch(r'[0-9a-f]{40}', c) or not os.path.isdir(old_dir):
                continue
            old = load_json(os.path.join(old_dir, TREE))
            if old is None:
                try:
                    hub = fetch_ranks().Hub(self.repo.repo, c, base=endpoint())
                    files = hub.tree()
                except (OSError, ValueError) as e:
                    self.log(f'{self}: cannot list commit {c[:12]} on the hub ({e}): none of its files reused')
                    continue
                old = {p: [(e.get('lfs') or {}).get('oid') or e['oid'], e['size']] for p, e in hub.entries.items()
                       if p in files}
                save_json(os.path.join(old_dir, TREE), old)
            for p, ident in mine.items():
                src, dst = os.path.join(old_dir, p), os.path.join(self.dir, p)
                if old.get(p) != ident or not os.path.isfile(src) or os.path.exists(dst):
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    os.link(src, dst)
                except OSError:
                    continue
                if p.startswith('experts/'):
                    j = os.path.join('.fetch', os.path.basename(p) + '.json')
                    if os.path.isfile(os.path.join(old_dir, 'experts', j)):
                        os.makedirs(os.path.join(self.v2, '.fetch'), exist_ok=True)
                        shutil.copyfile(os.path.join(old_dir, 'experts', j), os.path.join(self.v2, j))
                n += 1
        if n:
            self.log(f'{self}: {n} files unchanged since an earlier commit, linked from the cache')

    def marker(self, tp, rank, variant):
        return os.path.join(self.dir, 'complete', f'tp{tp}.rank{rank}.{variant.name}-{variant.sha256[:12]}.json')

    def check(self, tp, rank, variant):
        """Problems with this rank's part for this variant, cheapest checks only. [] = ready."""
        try:
            with open(self.marker(tp, rank, variant)) as f:
                c = json.load(f)
        except (OSError, ValueError):
            return [f'{self} rank {rank} of TP={tp}, variant {variant.name}, is not in the cache yet']
        bad = [f'{p}: missing or not {n} bytes' for p, n in sorted(c['files'].items())
               if not os.path.isfile(os.path.join(self.dir, p)) or os.path.getsize(os.path.join(self.dir, p)) != n]
        return bad

    def fetch_rank(self, tp, rank, variant, workers=None):
        """Fetch and check this rank's part, then write its COMPLETE marker. A rank's expert halves are small ranges
        (0.3-0.8 MB) a few apart, so the fetch is bound by requests in flight: 16 held a box near 10 MB/s from the hub.
        GR8_FETCH_WORKERS overrides the default 64."""
        workers = workers or int(os.environ.get('GR8_FETCH_WORKERS', '64'))
        fr = fetch_ranks()
        with open(variant.path) as f:
            layers = sorted(int(x) for x in json.load(f)['layers'])
        t = self.repo.tree()
        self.reuse()
        whole = self.whole_files(tp, rank)
        slices = [p for p in whole if p.startswith('index-slices/')]
        for p in slices:                    # first: bin/g6load plans this rank's halves from them
            self.repo.fetch(p, os.path.join(self.dir, p))
        os.makedirs(self.v2, exist_ok=True)
        try:
            experts_left = fr.fetch_layers(self.repo.hub, self.v2, tp, {rank}, self.slices, man=variant.path,
                                           layers=layers, plan_only=True)
        except RuntimeError as e:
            raise modelset.ModelError(f'{self}: {e}')
        rest = [p for p in whole if p not in slices]
        whole_left = sum(t[p] - (os.path.getsize(os.path.join(self.dir, p) + '.part')
                                 if os.path.exists(os.path.join(self.dir, p) + '.part') else 0)
                         for p in rest if not os.path.isfile(os.path.join(self.dir, p)))
        meter = Meter(experts_left + whole_left, self.log, f'fetching rank {rank} of {self}')
        self.log(f'{self}: rank {rank} of TP={tp}, variant {variant.name}: {meter.total / 1e9:.1f} GB to fetch')
        with ThreadPoolExecutor(4) as ex:
            for f in [ex.submit(self.repo.fetch, p, os.path.join(self.dir, p), meter) for p in rest]:
                f.result()
        try:
            fr.fetch_layers(self.repo.hub, self.v2, tp, {rank}, self.slices, man=variant.path, layers=layers,
                            workers=workers, verify=True, on_bytes=meter.add, log=self.log)
        except RuntimeError as e:
            raise modelset.ModelError(f'{self}: {e}')
        meter.line()
        files = {p: t[p] for p in whole}
        files.update({os.path.relpath(os.path.join(self.v2, n), self.dir): os.path.getsize(os.path.join(self.v2, n))
                      for n in os.listdir(self.v2) if n.endswith('.safetensors')})
        m = self.marker(tp, rank, variant)
        os.makedirs(os.path.dirname(m), exist_ok=True)
        with open(m + '.tmp', 'w') as f:
            json.dump({'release': f'{self.repo.repo}@{self.commit}', 'tp': tp, 'rank': rank, 'variant': variant.name,
                       'variant_sha256': variant.sha256, 'files': files}, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(m + '.tmp', m)
        self.log(f'{self}: rank {rank} of TP={tp}, variant {variant.name}, is complete in {self.dir}')


class Drafter:
    """The DFlash2 drafter at one commit, and this box's converted rank at TP=4:

        <cache>/hf/<org>--<repo>/<commit>/config.json, model.safetensors
        <cache>/hf/<org>--<repo>/<commit>/tp4/dflash.r<r>.{manifest.json,pack}, rank<r>.COMPLETE (written last)
    """

    def __init__(self, cache, repo, rev, refresh=False, log=print):
        self.repo = Repo(cache, repo, rev, refresh, log)
        self.log = log
        self.dir = os.path.join(self.repo.root, self.repo.commit)
        self.out = os.path.join(self.dir, 'tp4')

    @property
    def commit(self):
        return self.repo.commit

    @property
    def id(self):
        return f'{self.repo.repo}@{self.commit}'

    def __str__(self):
        return f'{self.repo.repo}@{self.commit[:12]}'

    def check(self, rank):
        """Problems with this rank's converted drafter. [] = ready."""
        try:
            with open(os.path.join(self.out, f'rank{rank}.COMPLETE')) as f:
                c = json.load(f)
        except (OSError, ValueError):
            return [f'the drafter {self} is not converted for rank {rank} yet']
        return [f'{p}: missing or not {n} bytes' for p, n in sorted(c['files'].items())
                if not os.path.isfile(os.path.join(self.out, p)) or os.path.getsize(os.path.join(self.out, p)) != n]

    def fetch_rank(self, rank):
        """Fetch the drafter (checked against its hashes), convert this rank, check the pack, then write COMPLETE."""
        convert = converter()                 # before the 4.9 GB download: is there a converter at all
        t = self.repo.tree()
        left = sum(t[n] for n in ('config.json', 'model.safetensors') if not os.path.isfile(os.path.join(self.dir, n)))
        meter = Meter(left, self.log, f'fetching the drafter {self}')
        for n in ('config.json', 'model.safetensors'):
            if n not in t:
                raise modelset.ModelError(f'the drafter {self} has no {n}')
            self.repo.fetch(n, os.path.join(self.dir, n), meter)
        tmp = os.path.join(self.dir, f'tmp.rank{rank}.{os.getpid()}')
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        self.log(f'converting the drafter {self} for rank {rank} of TP=4')
        r = subprocess.run(convert + ['--src', self.dir, '--out', tmp, '--nr', '4', '--ranks', str(rank)], cwd=REPO,
                           capture_output=True, text=True)
        names = [f'dflash.r{rank}.manifest.json', f'dflash.r{rank}.pack']
        if r.returncode or not all(os.path.isfile(os.path.join(tmp, n)) for n in names):
            shutil.rmtree(tmp, ignore_errors=True)
            raise modelset.ModelError(f'converting the drafter {self} for rank {rank} failed: '
                                      + (r.stderr or r.stdout).strip()[-500:])
        with open(os.path.join(tmp, names[0])) as f:
            want = int(json.load(f)['pack_bytes'])
        if os.path.getsize(os.path.join(tmp, names[1])) != want:
            shutil.rmtree(tmp, ignore_errors=True)
            raise modelset.ModelError(f'the converted drafter pack for rank {rank} is not the {want} bytes its manifest says')
        os.makedirs(self.out, exist_ok=True)
        files = {}
        for n in names:
            files[n] = os.path.getsize(os.path.join(tmp, n))
            os.replace(os.path.join(tmp, n), os.path.join(self.out, n))
        shutil.rmtree(tmp, ignore_errors=True)
        m = os.path.join(self.out, f'rank{rank}.COMPLETE')
        with open(m + '.tmp', 'w') as f:
            json.dump({'drafter': self.id, 'rank': rank, 'files': files}, f, indent=1, sort_keys=True)
        os.replace(m + '.tmp', m)
        self.log(f'the drafter {self} is converted for rank {rank} in {self.out}')
