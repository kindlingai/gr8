"""Weight sources for flash_ref.FlashRef. Each has get(name) -> tensor and expert(L, e) -> (gate, up, down), with
checkpoint names relative to `model.language_model.` (plus `lm_head.weight`).

DictWeights   tensors in memory (tests).
SafeWeights   a directory of HF safetensors shards with model.safetensors.index.json (BF16 source).
Exlr8Dense    the non-expert tensors of an exlr8 release (dense/ files, e.g. the lossless BF16 pass-through export).
Exlr8Experts  an exlr8 release's routed + shared experts decoded at one width.
HFWeights     the same files read tensor by tensor from the Hugging Face hub with HTTP range requests, cached on
              disk: real layers without the 600 GB download.
Overlay       the first source that has a name wins; use it to put exlr8 experts (exlr8_experts.py) over BF16.
"""
import json
import os
import struct
import urllib.request

import torch

PREFIX = 'model.language_model.'
DT = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16, 'I64': torch.int64, 'I32': torch.int32}


def full_name(name):
    return name if name.startswith('lm_head.') else PREFIX + name


class DictWeights:
    def __init__(self, d):
        self.d = d

    def has(self, name):
        return name in self.d

    def get(self, name):
        return self.d[name]

    def expert(self, L, e):
        p = f'layers.{L}.mlp.experts.{e}.'
        return tuple(self.get(p + n + '.weight') for n in ('gate_proj', 'up_proj', 'down_proj'))


class _Shards(DictWeights):
    """Reads tensors by name from safetensors shards through read(file, start, end) -> bytes."""

    def __init__(self, index):
        self.map = index['weight_map']
        self.headers = {}
        self.cache = {}

    def has(self, name):
        return full_name(name) in self.map

    def header(self, f):
        if f not in self.headers:
            n = struct.unpack('<Q', self.read(f, 0, 8))[0]
            self.headers[f] = (8 + n, json.loads(self.read(f, 8, 8 + n)))
        return self.headers[f]

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        fn = full_name(name)
        f = self.map[fn]
        base, h = self.header(f)
        m = h[fn]
        a, b = m['data_offsets']
        t = torch.frombuffer(bytearray(self.read(f, base + a, base + b)), dtype=DT[m['dtype']]).reshape(m['shape'])
        if not name.split('.mlp.experts.')[1:]:
            self.cache[name] = t          # dense tensors are reused across calls; experts are read when picked
        return t


class SafeWeights(_Shards):
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, 'model.safetensors.index.json')) as f:
            super().__init__(json.load(f))

    def read(self, f, a, b):
        with open(os.path.join(self.path, f), 'rb') as fh:
            fh.seek(a)
            return fh.read(b - a)


class HFWeights(_Shards):
    """Range reads from https://huggingface.co/<repo>/resolve/<rev>/<shard>, one cached file per tensor."""

    def __init__(self, index_path, repo='zai-org/GLM-5.3-Flash-BF16', rev='a5b45eb', cache_dir='/tmp/fe/hfcache'):
        self.url = f'https://huggingface.co/{repo}/resolve/{rev}/'
        self.dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        with open(index_path) as f:
            super().__init__(json.load(f))

    def read(self, f, a, b):
        key = os.path.join(self.dir, f'{f}.{a}-{b}')
        if os.path.exists(key) and os.path.getsize(key) == b - a:
            with open(key, 'rb') as fh:
                return fh.read()
        req = urllib.request.Request(self.url + f, headers={'Range': f'bytes={a}-{b - 1}'})
        tok = os.environ.get('HF_TOKEN')
        if tok:
            req.add_header('Authorization', f'Bearer {tok}')
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if len(data) != b - a:
            raise IOError(f'{f} [{a},{b}): got {len(data)} bytes')
        with open(key + '.part', 'wb') as fh:
            fh.write(data)
        os.replace(key + '.part', key)
        return data

    def prefetch(self, names, threads=8):
        """Read these tensors into the disk cache in parallel (no-op for cached ones)."""
        from concurrent.futures import ThreadPoolExecutor

        def one(name):
            fn = full_name(name)
            f = self.map[fn]
            base, h = self.header(f)
            a, b = h[fn]['data_offsets']
            self.read(f, base + a, base + b)
        for f in {self.map[full_name(n)] for n in names}:
            self.header(f)
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, names))


class Overlay(DictWeights):
    def __init__(self, *sources):
        self.sources = sources

    def has(self, name):
        return any(s.has(name) for s in self.sources)

    def get(self, name):
        for s in self.sources:
            if s.has(name):
                return s.get(name)
        raise KeyError(name)

    def expert(self, L, e):
        for s in self.sources:
            if s.has(f'layers.{L}.mlp.experts.{e}.gate_proj.weight'):
                return s.expert(L, e)
        raise KeyError(f'expert {L}.{e}')


class Exlr8Experts(DictWeights):
    """Routed and shared experts of the MoE layers in an exlr8 v2 release, decoded at one width per layer (K) by the
    reference loader (exlr8-loader loader/exlr8.py; pass its repo root), when read."""

    def __init__(self, release, loader_repo, K, shared_K=None, layers=None):
        import sys
        if loader_repo not in sys.path:
            sys.path.insert(0, loader_repo)
        from loader import exlr8
        self.exlr8, self.release, self.K, self.shared_K = exlr8, release, K, shared_K or K
        self.layers = {}
        self.only = layers

    def layer(self, L):
        if L not in self.layers:
            self.layers[L] = self.exlr8.ExpertLayer(self.release, L)
        return self.layers[L]

    def _parse(self, name):
        parts = name.split('.')
        if len(parts) < 5 or parts[0] != 'layers' or parts[2] != 'mlp' or parts[-1] != 'weight':
            return None
        L = int(parts[1])
        if self.only is not None and L not in self.only:
            return None
        if parts[3] == 'experts' and len(parts) == 7:
            return L, int(parts[4]), parts[5]
        if parts[3] == 'shared_experts' and len(parts) == 6:
            return L, 'shared', parts[4]
        return None

    def has(self, name):
        p = self._parse(name)
        if p is None:
            return False
        try:
            lay = self.layer(p[0])
        except FileNotFoundError:
            return False
        return p[1] == 'shared' or p[1] < lay.experts

    def get(self, name):
        # not cached: a decoded expert is 100 MB fp32 and the reference reads each picked expert once per call
        L, e, proj = self._parse(name)
        lay = self.layer(L)
        K = self.shared_K if e == 'shared' else self.K
        return lay.projection(lay.shared_index if e == 'shared' else e, proj.split('_')[0], K)


class Exlr8Dense(DictWeights):
    """Non-expert tensors (layers, vocabulary, final norm) of an exlr8 release's dense/ files, read by the reference
    loader's load_layer / load_vocab one layer at a time. Lossless pass-through records keep the source dtype."""

    def __init__(self, release, loader_repo):
        import sys
        if loader_repo not in sys.path:
            sys.path.insert(0, loader_repo)
        from loader import exlr8
        self.exlr8, self.release = exlr8, release
        self.layers, self.vocab = {}, None

    def _get(self, name):
        if name.startswith('layers.'):
            L = int(name.split('.')[1])
            if L not in self.layers:
                self.layers[L] = self.exlr8.load_layer(self.release, {'layers': {}}, L)
            return self.layers[L].get(full_name(name))
        if self.vocab is None:
            self.vocab = self.exlr8.load_vocab(self.release)
        return self.vocab.get(full_name(name))

    def has(self, name):
        return '.mlp.experts.' not in name and self._get(name) is not None

    def get(self, name):
        t = self._get(name)
        if t is None:
            raise KeyError(name)
        return t
