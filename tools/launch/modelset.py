"""Model sets: what the ranks load, named by a content id.

A model set is an exlr8 release: a directory with dense/, the width files, index-slices/, manifests/ and the tokenizer
files, either a local copy used in place or a Hugging Face release that hfsource.py fetches rank by rank into
--model-cache. Every rank plans its image from the release's files (g6m4 --v2-dense). The set's description names the
release's weight set id and the sha256 of the four tokenizer files. The set id is the sha256 of that description
without the set_id field, written canonically (sorted keys, no spaces). Paths are not in it, so boxes that hold the
same release at different paths agree.
"""
import hashlib
import json
import os

import models

FORMAT = 'gr8-set-1'
TOKENIZER = models.TOKENIZER_GLM


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


def tokenizer_dir(model, model_dir, files=TOKENIZER):
    """Where the tokenizer files are (the four of GLM's, or files): <model>/tokenizer, <model>, or MODEL_DIR."""
    for d in (os.path.join(model, 'tokenizer'), model, model_dir):
        if d and all(os.path.isfile(os.path.join(d, f)) for f in files):
            return d
    raise ModelError(f'no tokenizer files ({", ".join(files)}) in {model}/tokenizer, {model}'
                     + (f' or {model_dir}' if model_dir else ' (MODEL_DIR names a directory that has them)'))


def is_release(d):
    """A directory holding the exlr8 release's dense files (dense/ under it)."""
    return bool(d) and os.path.isfile(os.path.join(d, 'dense', 'exlr8-vocab.safetensors'))


def describe_release(d, tp, tok, files=TOKENIZER):
    """set.json for an exlr8 release at this TP: no rank packs. The weights are named by the release's weight set id
    (every width manifest carries it), else by the sha256 of its dense hashes."""
    weights = None
    for p in sorted(os.listdir(os.path.join(d, 'manifests'))) if os.path.isdir(os.path.join(d, 'manifests')) else []:
        try:
            with open(os.path.join(d, 'manifests', p)) as f:
                weights = json.load(f).get('set_id') or weights
        except (OSError, ValueError):
            continue
    if not weights:
        raise ModelError(f'{d}: no width manifest with a set_id under manifests/ (release v0.3 or later)')
    s = {'format': FORMAT, 'tp': tp, 'ranks': [], 'dflash': [], 'qo': None,
         'tokenizer': {f: sha256_file(os.path.join(tok, f)) for f in files},
         'source': {'kind': 'exlr8-release', 'weights': weights}}
    s['set_id'] = compute_id(s)
    return s


def is_pseudo(d):
    """A pseudo engine release: responses.toml, the rules the engine answers from (frontend/pseudo-engine)."""
    return bool(d) and os.path.isfile(os.path.join(d, 'responses.toml'))


def pseudo_set(d, tp, model_dir=None):
    """The model set of a pseudo engine release, used in place: its weights are named by the sha256 of responses.toml.
    The engine reads the tokenizer from the release alone, so MODEL_DIR is refused."""
    tok = tokenizer_dir(d, None)
    if model_dir and os.path.abspath(model_dir) != os.path.abspath(tok):
        raise ModelError(f'--model-dir {model_dir}: g6pseudo reads the tokenizer files from the release ({tok})')
    s = {'format': FORMAT, 'tp': tp, 'ranks': [], 'dflash': [], 'qo': None,
         'tokenizer': {f: sha256_file(os.path.join(tok, f)) for f in TOKENIZER},
         'source': {'kind': 'pseudo', 'weights': sha256_file(os.path.join(d, 'responses.toml'))}}
    s['set_id'] = compute_id(s)
    return ModelSet(s, f'{d} (pseudo engine rules)', tok, d)


def ds_set(d, tp, model_dir=None):
    """The model set of the DeepSeek-V4.1-Flash exlr8 native release, used in place: no width variants, and the
    tokenizer is two files."""
    tok = tokenizer_dir(d, model_dir, models.TOKENIZER_DS)
    return ModelSet(describe_release(d, tp, tok, models.TOKENIZER_DS), f'{d} (exlr8 native release)', tok, d)


def release_set(d, tp, model_dir=None, label=None):
    """The model set of an exlr8 release directory, used in place."""
    tok = tokenizer_dir(d, model_dir)
    return ModelSet(describe_release(d, tp, tok), label or f'{d} (exlr8 release)', tok, d)


class ModelSet:
    """A resolved model set: its description, the release directory and the tokenizer on this box. release: the
    Hugging Face release (hfsource.Release) when it was fetched, and drafter the DFlash2 drafter fetched and converted
    on this box (hfsource.Drafter), or None."""
    release = drafter = None

    def __init__(self, setj, source, tok, d):
        self.set, self.source, self.tokenizer, self.dir = setj, source, tok, d

    @property
    def id(self):
        return self.set['set_id']


# GLM-5.3's exlr8 v2 variants (models.py has each model's)
VARIANTS = models.GLM53.variants
DEFAULT_VARIANT = models.GLM53.default_variant
HALF_BYTES = {k: 384 * 8 * 32 * k for k in (2, 3, 4)}     # one 128-channel half of one projection (tools/convert/v2.py)


class Variant:
    """The exlr8 v2 width manifest a boot loads. path and sha256 are None for packs without v2 widths."""

    def __init__(self, name, path=None):
        self.name, self.path = name, path
        self.sha256 = sha256_file(path) if path else None

    def __str__(self):
        return self.name + (f' ({self.path}, sha256 {self.sha256[:16]})' if self.path else '')


def resolve_variant(spec, v2_dir, search, tp=None, model=models.GLM53):
    """spec: a name in model.variants or a manifest path (None: the first of model.default_variant[tp] the model set
    has when there are v2 widths, else home). v2_dir: the V2 width files, or None for packs without them. search: the directories
    that hold the model set; exlr8-manifest-<name>.json is looked for in each of them, then in v2_dir, directly or
    under manifests/."""
    def found(n):
        return any(os.path.isfile(os.path.join(d, sub, f'exlr8-manifest-{n}.json'))
                   for d in [*search, v2_dir] for sub in ('', 'manifests'))
    defaults = model.default_variant.get(tp, ()) if v2_dir else ()
    name = spec or next((n for n in defaults if found(n)), defaults[-1] if defaults else 'home')
    if not v2_dir:
        if name == 'home':
            return Variant('home')
        raise ModelError(f'--variant {name} needs the exlr8 v2 width files (--v2 DIR --v2-slices DIR): these packs '
                         'carry one set of expert widths, home')
    if name not in model.variants:
        if not os.path.isfile(name):
            raise ModelError(f'--variant {name}: not one of {", ".join(model.variants)}, and no such manifest file')
        with open(name) as f:
            label = json.load(f).get('name') or os.path.basename(name)
        return Variant(label, name)
    file = f'exlr8-manifest-{name}.json'
    tried = []
    for d in [*search, v2_dir]:
        for p in (os.path.join(d, file), os.path.join(d, 'manifests', file)):
            tried.append(p)
            if os.path.isfile(p):
                return Variant(name, p)
    raise ModelError(f'--variant {name}: no {file} in ' + ', '.join(dict.fromkeys(os.path.dirname(t) for t in tried)))


def model_id(ms, variant):
    """The id the boxes compare and the engine checks: the set id, joined with the variant's manifest when there is one,
    and with the repo and commit of each Hugging Face source (ms.release, ms.drafter). Two variants or two commits of
    one set never share an id. The cache keeps the set id: the base packs are the same."""
    src = ''.join(f' {s.id}' for s in (ms.release, ms.drafter) if s)
    if not variant.sha256 and not src:
        return ms.id
    return hashlib.sha256(f'{ms.id} {variant.name} {variant.sha256}{src}'.encode()).hexdigest()


def expert_gb(path, tp):
    """GB of routed and shared expert weights on the busiest rank at this TP, under the manifest's widths (half h of
    expert e in layer l belongs to rank (16e + h + l) mod tp)."""
    with open(path) as f:
        manifest = json.load(f)
        layers = manifest['layers']
    hidden = int(manifest.get('hidden_size', 6144))
    half_bytes = {k: hidden * 16 * k for k in (2, 3, 4)}
    per = [0] * tp
    for key, e in layers.items():
        l = int(key)
        sh = e.get('shared') or 0
        if isinstance(sh, int):
            sh = {'gate': [sh] * 16, 'up': [sh] * 16, 'down': sh}
        rows = [(x, e['gate'][x], e['up'][x], e['down'][x]) for x in range(len(e['down']))]
        if sh['down']:
            rows.append((len(e['down']), sh['gate'], sh['up'], sh['down']))      # the shared expert is e = 256 in the half rule
        for x, g, u, d in rows:
            for h in range(16):
                per[(16 * x + h + l) % tp] += half_bytes[g[h]] + half_bytes[u[h]] + half_bytes[d]
    return max(per) / 1e9


def check_tokenizer(ms, d=None):
    d = d or ms.tokenizer
    bad = [n for n, h in ms.set['tokenizer'].items()
           if not os.path.isfile(os.path.join(d, n)) or sha256_file(os.path.join(d, n)) != h]
    return [f'{d}: {n} missing or not the model set\'s' for n in bad]
