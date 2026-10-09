"""Model sets: what a rank loads, named by a content id, and the local cache that holds one box's shard of it.

A model set is a prepared pack directory at one TP. Its description, set.json, lists for every rank (not only this
box's) the sha256 of each pack manifest and the pack size, the same for the drafter and QO packs when the
directory has them, and the sha256 of the four tokenizer files. A manifest carries a sha256 per tensor, so its hash
covers the pack bytes, and the engine checks those at load unless NOHASH=1. The set id is the sha256 of set.json
without the set_id field, written canonically (sorted keys, no spaces). Paths are not in it, so boxes that hold the
same set at different paths agree.

--model DIR uses a prepared directory in place (it needs every rank's manifest; the pack bytes only for the ranks
that load from it). --model-cache DIR is this box's local directory: a boot copies only this rank's files into

    <cache>/sets/<set_id>/set.json, tokenizer/, rank<r>/{rank<r>.*, dflash.r<r>.*, qo.*, COMPLETE}

through <cache>/tmp/ and a rename, with COMPLETE written last. --model-cache alone boots from the one complete set in
the cache for this TP.
"""
import fcntl
import hashlib
import json
import os
import re
import shutil
import time

FORMAT = 'gr8-set-1'
TOKENIZER = ('chat_template.jinja', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json')


class ModelError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compute_id(setj):
    body = {k: v for k, v in setj.items() if k != 'set_id'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def is_remote(model):
    """An https:// URL, or org/repo with no local path of that name. options.parse makes every local path absolute."""
    return bool(re.match(r'https?://', model or '')) or bool(re.fullmatch(r'[\w.-]+/[\w.-]+(@[\w.-]+)?', model or '')
                                                                 and not os.path.exists(model))


def _pack(d, stem):
    """{'manifest': sha256, 'pack_bytes': n} for <stem>.manifest.json, or None when the directory has no such pack."""
    man = os.path.join(d, stem + '.manifest.json')
    if not os.path.isfile(man):
        return None
    try:
        with open(man) as f:
            pack_bytes = int(json.load(f)['pack_bytes'])
    except (OSError, ValueError, KeyError) as e:
        raise ModelError(f'{man}: not a pack manifest ({e})')
    return {'manifest': sha256_file(man), 'pack_bytes': pack_bytes}


def tokenizer_dir(model, model_dir):
    """Where the four tokenizer files are: <model>/tokenizer, <model>, or MODEL_DIR."""
    for d in (os.path.join(model, 'tokenizer'), model, model_dir):
        if d and all(os.path.isfile(os.path.join(d, f)) for f in TOKENIZER):
            return d
    raise ModelError(f'no tokenizer files ({", ".join(TOKENIZER)}) in {model}/tokenizer, {model}'
                     + (f' or {model_dir}' if model_dir else ' (MODEL_DIR names a directory that has them)'))


def describe(model, tp, model_dir=None):
    """set.json for a prepared pack directory at this TP."""
    if not os.path.isdir(model):
        raise ModelError(f'--model {model}: no such directory')
    ranks = [_pack(model, f'rank{r}') for r in range(tp)]
    if any(x is None for x in ranks):
        if any(n.endswith('.safetensors') for n in os.listdir(model)):
            raise ModelError(f'{model} is a raw checkpoint, not packs. Preparing one is not in this version: build the '
                             'packs with ./gr8 glm53 packs, then --model <the pack dir>')
        missing = [f'rank{r}.manifest.json' for r in range(tp) if ranks[r] is None]
        raise ModelError(f'{model} lacks {", ".join(missing)}: a model set needs every rank\'s manifest at TP={tp}')
    if _pack(model, f'rank{tp}'):
        raise ModelError(f'{model} has a rank{tp}.manifest.json: these packs are for a TP above {tp}')
    tok = tokenizer_dir(model, model_dir)
    s = {'format': FORMAT, 'tp': tp, 'ranks': ranks,
         'dflash': [_pack(model, f'dflash.r{r}') for r in range(tp)],
         'qo': _pack(model, 'qo'),
         'tokenizer': {f: sha256_file(os.path.join(tok, f)) for f in TOKENIZER},
         'source': {'kind': 'packs'}}
    for k in ('dflash',):
        if not any(s[k]):
            s[k] = []
        elif not all(s[k]):
            raise ModelError(f'{model} has {k} packs for some ranks only')
    s['set_id'] = compute_id(s)
    return s, tok


def load_set(path):
    with open(path) as f:
        s = json.load(f)
    if s.get('format') != FORMAT:
        raise ModelError(f'{path}: format {s.get("format")!r}, want {FORMAT}')
    if s.get('set_id') != compute_id(s):
        raise ModelError(f'{path}: set_id does not match its contents (edited, or partly written)')
    return s


def needed(setj, rank, plan):
    """{file name: (kind, record)} this rank loads, from the resolved knobs."""
    k, tp = plan['knobs'], setj['tp']
    out = {f'rank{rank}.pack': ('pack', setj['ranks'][rank]), f'rank{rank}.manifest.json': ('manifest', setj['ranks'][rank])}

    def add(stem, rec, why):
        if not rec:
            raise ModelError(f'{why}, and the model set has no {stem}.manifest.json')
        out[f'{stem}.pack'] = ('pack', rec)
        out[f'{stem}.manifest.json'] = ('manifest', rec)
    if tp == 4 and k.get('DFLASH') == '1':
        add(f'dflash.r{rank}', (setj['dflash'] or [None] * tp)[rank], 'DFLASH=1 loads the DFlash2 drafter (DFLASH=0 for MTP only)')
    if k.get('PF_OWN') == '3':
        add('qo', setj['qo'], 'PF_OWN=3 loads the QO pack')
    return out


def check_files(d, files):
    """Problems with the files a rank loads from directory d, cheapest checks only. [] = fine."""
    bad = []
    for name, (kind, rec) in sorted(files.items()):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            bad.append(f'missing {p}')
        elif kind == 'pack' and os.path.getsize(p) != rec['pack_bytes']:
            bad.append(f'{p} is {os.path.getsize(p)} bytes, its manifest says {rec["pack_bytes"]} (partly copied?)')
        elif kind == 'manifest' and sha256_file(p) != rec['manifest']:
            bad.append(f'{p} differs from the model set\'s manifest')
    return bad


class ModelSet:
    """A resolved model set: set.json, and where each rank's files and the tokenizer are on this box."""

    def __init__(self, setj, source, tok, rank_dir_fn, cache=None, src_dir=None):
        self.set, self.source, self.tokenizer = setj, source, tok
        self.rank_dir, self.cache, self.src_dir = rank_dir_fn, cache, src_dir

    @property
    def id(self):
        return self.set['set_id']


def resolve(model, cache, tp, model_dir=None):
    """The model set for this boot. Nothing is copied here; prepare() does that once the rank is known."""
    if model and is_remote(model):
        raise ModelError(f'--model {model}: downloading from Hugging Face is not in this version. Download it, build the '
                         'packs (./gr8 glm53 packs), and pass --model <the pack dir>')
    if model:
        if os.path.isfile(os.path.join(model, 'set.json')):
            setj = load_set(os.path.join(model, 'set.json'))
            if setj['tp'] != tp:
                raise ModelError(f'{model}/set.json is a TP={setj["tp"]} set, this boot is --tp {tp}')
            tok = tokenizer_dir(model, model_dir)
        else:
            setj, tok = describe(model, tp, model_dir)
        if cache:
            return ModelSet(setj, f'{model}, cached in {cache}', tok,
                            lambda r: os.path.join(cache, 'sets', setj['set_id'], f'rank{r}'), cache, model)
        return ModelSet(setj, f'{model} (in place)', tok, lambda r: model)
    if not cache:
        raise ModelError('no model: pass --model <prepared pack dir> (and --model-cache <local dir> to copy this '
                         "box's shard there), or --model-cache alone once it holds a set")
    sets = []
    for sid in sorted(os.listdir(os.path.join(cache, 'sets'))) if os.path.isdir(os.path.join(cache, 'sets')) else []:
        p = os.path.join(cache, 'sets', sid, 'set.json')
        if os.path.isfile(p):
            s = load_set(p)
            if s['tp'] == tp:
                sets.append(s)
    if not sets:
        raise ModelError(f'{cache} holds no model set for TP={tp}: boot once with --model <pack dir> --model-cache {cache}')
    if len(sets) > 1:
        raise ModelError(f'{cache} holds {len(sets)} sets for TP={tp} ({", ".join(s["set_id"][:12] for s in sets)}): '
                         'pass --model to pick one, or remove the old one')
    setj = sets[0]
    d = os.path.join(cache, 'sets', setj['set_id'])
    return ModelSet(setj, f'{cache} (cache only)', os.path.join(d, 'tokenizer'),
                    lambda r: os.path.join(d, f'rank{r}'), cache, None)


def offers(ms, plan):
    """The ranks this box's model set can serve: None for any (every rank's files are here, or a cache can copy them
    from --model), else the ranks whose files are here."""
    if ms.cache and ms.src_dir:
        return None
    tp = ms.set['tp']
    ok = [r for r in range(tp) if not validate_rank(ms, r, plan)]
    return None if len(ok) == tp else ok


def read_complete(d):
    try:
        with open(os.path.join(d, 'COMPLETE')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def validate_rank(ms, rank, plan):
    """Problems with this rank's files for this boot. [] = ready."""
    files = needed(ms.set, rank, plan)
    d = ms.rank_dir(rank)
    bad = []
    if ms.cache:
        c = read_complete(d)
        if c is None:
            return [f'{d} has no COMPLETE']
        if c.get('set_id') != ms.id or c.get('rank') != rank:
            return [f'{d}/COMPLETE is for set {str(c.get("set_id"))[:12]} rank {c.get("rank")}']
        for name, rec in sorted((c.get('files') or {}).items()):
            p = os.path.join(d, name)
            if not os.path.isfile(p) or os.path.getsize(p) != rec.get('bytes'):
                bad.append(f'{p}: missing or not {rec.get("bytes")} bytes')
        bad += [f'{d} lacks {n}' for n in sorted(files) if n not in (c.get('files') or {})]
        if bad:
            return bad
    return check_files(d, files)


def _copy(src, dst, log, label):
    """Copy one file, with a progress line every 30 s, and fsync it."""
    total, done, t0, said = os.path.getsize(src), 0, time.monotonic(), time.monotonic()
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        while True:
            b = fi.read(64 << 20)
            if not b:
                break
            fo.write(b)
            done += len(b)
            if hasattr(os, 'posix_fadvise'):   # a 90 GB copy must stay out of the page cache: on GB10 it is GPU memory
                fo.flush()
                os.posix_fadvise(fi.fileno(), 0, done, os.POSIX_FADV_DONTNEED)
            if time.monotonic() - said >= 30:
                said = time.monotonic()
                log(f'{label}: {done / 1e9:.1f} of {total / 1e9:.1f} GB ({done / 1e6 / (said - t0):.0f} MB/s)')
        fo.flush()
        os.fsync(fo.fileno())
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def prepare(ms, rank, plan, log=print):
    """Make the cache hold this rank's files for this boot. Copies only what is missing or wrong, from --model."""
    if not ms.cache:
        return
    if not validate_rank(ms, rank, plan):
        _use_cached_tokenizer(ms)
        return
    if not ms.src_dir:
        raise ModelError(f'the cache cannot boot rank {rank} of set {ms.id[:12]}: '
                         + '; '.join(validate_rank(ms, rank, plan)) + '. Pass --model <pack dir> to copy it')
    sdir = os.path.join(ms.cache, 'sets', ms.id)
    tmp_root = os.path.join(ms.cache, 'tmp')
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)
    with open(os.path.join(sdir, '.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not validate_rank(ms, rank, plan):    # another gr8 on this box finished it while we waited
            return
        for n in os.listdir(tmp_root):            # leftovers of a copy whose process is gone
            m = re.fullmatch(r'.*\.(\d+)', n)
            if m and not _pid_alive(int(m.group(1))):
                shutil.rmtree(os.path.join(tmp_root, n), ignore_errors=True)
        _write_set(ms, sdir)
        files = needed(ms.set, rank, plan)
        final = ms.rank_dir(rank)
        old = read_complete(final) if os.path.isdir(final) else None
        keep = (old or {}).get('files', {}) if (old or {}).get('set_id') == ms.id and (old or {}).get('rank') == rank else {}
        names = sorted(set(files) | set(keep))
        tmp = os.path.join(tmp_root, f'rank{rank}.{os.getpid()}')
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for n in names:
            src_old = os.path.join(final, n)
            rec = files.get(n)
            if n in keep and os.path.isfile(src_old) and not (rec and check_files(final, {n: rec})):
                os.link(src_old, os.path.join(tmp, n))       # already here and correct
                continue
            src = os.path.join(ms.src_dir, n)
            if not os.path.isfile(src):
                raise ModelError(f'{ms.src_dir} has no {n}')
            log(f'copying {n} ({os.path.getsize(src) / 1e9:.1f} GB) into {final}')
            _copy(src, os.path.join(tmp, n), log, f'copying {n}')
        bad = check_files(tmp, {n: r for n, r in files.items()})
        if bad:
            raise ModelError('after the copy: ' + '; '.join(bad))
        complete = {'set_id': ms.id, 'rank': rank,
                    'files': {n: {'bytes': os.path.getsize(os.path.join(tmp, n))} for n in names}}
        with open(os.path.join(tmp, 'COMPLETE'), 'w') as f:
            json.dump(complete, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(tmp)
        if os.path.isdir(final):
            gone = os.path.join(tmp_root, f'old.rank{rank}.{os.getpid()}')
            os.rename(final, gone)
            os.rename(tmp, final)
            shutil.rmtree(gone, ignore_errors=True)
        else:
            os.rename(tmp, final)
        _fsync_dir(sdir)
        log(f'rank {rank} of set {ms.id[:12]} is complete in {final}')


def _write_set(ms, sdir):
    """set.json and tokenizer/ in the cache's set directory, each through a temporary name."""
    p = os.path.join(sdir, 'set.json')
    if not os.path.isfile(p):
        with open(p + '.tmp', 'w') as f:
            json.dump(ms.set, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.rename(p + '.tmp', p)
    tdir = os.path.join(sdir, 'tokenizer')
    if not all(os.path.isfile(os.path.join(tdir, n)) and sha256_file(os.path.join(tdir, n)) == h
               for n, h in ms.set['tokenizer'].items()):
        tmp = tdir + f'.{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for n in ms.set['tokenizer']:
            shutil.copyfile(os.path.join(ms.tokenizer, n), os.path.join(tmp, n))
        shutil.rmtree(tdir, ignore_errors=True)
        os.rename(tmp, tdir)
    ms.tokenizer = tdir


def _use_cached_tokenizer(ms):
    tdir = os.path.join(ms.cache, 'sets', ms.id, 'tokenizer')
    if os.path.isdir(tdir) and not check_tokenizer(ms, tdir):
        ms.tokenizer = tdir


def check_tokenizer(ms, d=None):
    d = d or ms.tokenizer
    bad = [n for n, h in ms.set['tokenizer'].items()
           if not os.path.isfile(os.path.join(d, n)) or sha256_file(os.path.join(d, n)) != h]
    return [f'{d}: {n} missing or not the model set\'s' for n in bad]
