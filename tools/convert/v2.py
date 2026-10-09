"""exlr8 v2 expert files -> g6 rank pack tensors (#65). Format: exlr8quant SPEC.md sections 5.2, 6.1-6.5, 7.1, 8.

v2 stores every routed expert of a layer at K2, K3 and K4 (one trellis file per width) and the shared expert at K3 and
K4. A deployment picks a width per unit from a manifest: gate and up per (expert, 128-channel half), down per expert.
Half h of expert e in layer l belongs to rank (16e + h + l) mod N; the shared expert is e = 256. This module reads the
layer's index file, then only the halves the requested ranks own, checks each half's xxhash64, and writes the same tr3.*
tensors convert.py writes for TR3 / exlr8 v1 layers:

  tr3.fc1    [blocks][2][384][8][TB]  gate then up of each owned half; TB = 32 * the width of that (half, matrix)
  tr3.fc2    [blocks][384][8][TB]     down of each owned half, item128 order permuted to g6's [col16][k16]
  tr3.v2     [blk][3][128]            gate / up output-side (svh) and down input-side (suh) channels of the half
  tr3.v1     [blk][2][6144]           gate / up input-side vector of the encode each half came from (signs.in x su_block)
  tr3.v3     [blk][6144]              down output-side vector (signs.out x sv_block, the expert's down width)
  or, with the trellis shared expert, the same vectors as signs and block scales (the kernels form sign x scale):
  tr3.sg     [2][6144]                signs.in, signs.out as fp16 +-1
  tr3.v1b    [blk][2][48]             gate / up su_block of each half's encode
  tr3.v3b    [blk][48]                down sv_block
  tr3.table  [E][6]                   fc1 offset, fc2 offset, first block row, block count, down width, first half
  tr3.kmix   [E]                      0: every fc1 (block, matrix) has width table[4]. Else 4 bits per (block b,
                                      matrix m) at bit 4 * (2b + m): that unit's width (the kernels then take fc1
                                      offsets from these widths)
E is 256, or 257 when the shared expert is stored as trellis (row 256, always routed with weight 1).

A rank's halves of one expert are not contiguous under the half rule (at TP4 they are h, h + 4, h + 8, h + 12). The
kernels index blocks by row only, so they do not care; table[5] (the first half) is informational.

python3 -m tools.convert.v2 --v2 DIR --layer 40 --nr 5 --out DIR [--widths home|down1|up1|k2|k3|k4|MANIFEST]
        [--v1 DIR] [--ranks 0,1] [--shared trellis|off] [--routed all|files|0-63]
    writes expert-only rank packs (rank<r>.pack + manifest: the layer's tr3.* tensors) for kernel and TP tests.
python3 -m tools.convert.v2 --v2 DIR --layer 40 --check
    reads and verifies every half the layer's files hold.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from tools.convert.st import ST
from tools.convert.xxh import xxh64

H = 6144
I = 2048
NH = 16                 # 128-channel halves per expert
KSH = 384               # k16 steps of the hidden side
N_EXP = 256
SHARED = 256            # the shared expert's index under the half rule
WIDTHS = (2, 3, 4)
PROJ = ('gate', 'up', 'down')

# SPEC.md 6.3 / 6.4 names. In the index file, per width K: f'{s}k{K}.index' etc., s = '' or 'shared.'.
INDEX_FILE = 'exlr8-layer-{l:03d}.index.safetensors'
WIDTH_FILE = 'exlr8-layer-{l:03d}.k{k}.safetensors'
SIGNS = {'gate': 'signs.in', 'up': 'signs.in', 'down': 'signs.out'}          # int8 [6144], one per layer and side
BLOCK = {'gate': 'gate.su_block', 'up': 'up.su_block', 'down': 'down.sv_block'}   # fp16 [E, 48] hidden-side scales
CHAN = {'gate': 'gate.svh', 'up': 'up.svh', 'down': 'down.suh'}              # fp16 [E, 2048] per channel, sign included
GATE_UP_ORDER = 'grouped'
DOWN_ORDER = 'item128'


def half_bytes(k):
    """B: bytes of one 128-channel half of one projection at width k: 384 x 8 tiles of 32k bytes."""
    return KSH * 8 * 32 * k


def owner(e, h, l, n):
    """Rank of half h of expert e (256 = shared) in layer l at TP = n."""
    return (16 * e + h + l) % n


def owned_halves(e, l, n, r):
    return [h for h in range(NH) if owner(e, h, l, n) == r]


def parse_experts(spec):
    """'0-255' or '0-63,128' -> sorted list of expert ids (index metadata `experts`; text after the first space, such as
    the writer's "(partial: a test set ...)" note, is ignored)."""
    out = set()
    for part in filter(None, spec.split(' ')[0].split(',')):
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def half_digest(trellis, chan, block):
    """SPEC.md 6.4: xxhash64 (seed 0) of the half's B trellis bytes, its 128 per-channel fp16 values, then the
    projection's 48 fp16 hidden-side block scales."""
    return xxh64(trellis, chan, block)


def down_to_g6(half, k):
    """down half in item128 order [col16 / 8][k16][col16 % 8][tile] -> g6 fc2 order [col16][k16][tile]."""
    tb = 32 * k
    return np.ascontiguousarray(half.reshape(48, 8, 8, tb).transpose(0, 2, 1, 3).reshape(KSH, 8, tb))


def down_from_g6(rec, k):
    """Inverse of down_to_g6 (tests)."""
    tb = 32 * k
    return np.ascontiguousarray(rec.reshape(48, 8, 8, tb).transpose(0, 2, 1, 3).reshape(-1))


# ---------------------------------------------------------------------------------------------- reading
class V2Layer:
    """The v2 files of one layer: the index file (read whole, cached) and the trellis files (opened on first use)."""

    def __init__(self, d, l, verify=True):
        self.d, self.l, self.verify = d, l, verify
        self.files, self.cache = {}, {}
        self.idx = ST(os.path.join(d, INDEX_FILE.format(l=l)))
        m = self.idx.meta
        if m.get('format') != 'exlr8-v2' or m.get('contents', 'index') != 'index':
            raise ValueError(f'L{l}: index file format {m.get("format")!r} / {m.get("contents")!r}')
        if m.get('gate_up_order', GATE_UP_ORDER) != GATE_UP_ORDER or m.get('down_order', DOWN_ORDER) != DOWN_ORDER:
            raise ValueError(f'L{l}: tile orders {m.get("gate_up_order")}/{m.get("down_order")}, want '
                             f'{GATE_UP_ORDER}/{DOWN_ORDER}')
        self.experts = parse_experts(m.get('experts', f'0-{N_EXP - 1}'))      # routed experts the files hold
        self.row = {e: i for i, e in enumerate(self.experts)}                 # their rows in the routed index tensors
        self.widths = [k for k in WIDTHS if f'k{k}.index' in self.idx]
        self.shared_widths = [k for k in WIDTHS if f'shared.k{k}.index' in self.idx]
        for key, have in (('widths', self.widths), ('shared_widths', self.shared_widths)):
            if key in m and sorted(int(x) for x in m[key].split(',') if x) != have:
                raise ValueError(f'L{l}: metadata {key}={m[key]!r} but index tensors for {have}')

    def file(self, k):
        if k not in self.files:
            f = ST(os.path.join(self.d, WIDTH_FILE.format(l=self.l, k=k)))
            if f.meta.get('format') != 'exlr8-v2' or int(f.meta.get('k', k)) != k:
                raise ValueError(f'{f.path}: format {f.meta.get("format")!r} k {f.meta.get("k")!r}')
            self.files[k] = f
        return self.files[k]

    def close(self):
        for f in list(self.files.values()) + [self.idx]:
            f.close()
        self.files = {}

    def has_shared(self, k):
        return k in self.shared_widths

    def _get(self, name):
        if name not in self.cache:
            self.cache[name] = self.idx.get(name)
        return self.cache[name]

    def _t(self, e, k, what):
        """Row of expert e (256 = shared, row 0 of its E = 1 tensors) of width k's index tensor `what`."""
        if e == SHARED:
            if k not in self.shared_widths:
                raise FileNotFoundError(f'L{self.l}: no shared expert at k{k}')
            return self._get(f'shared.k{k}.{what}')[0]
        if k not in self.widths:
            raise FileNotFoundError(f'L{self.l}: no routed experts at k{k}')
        if e not in self.row:
            raise KeyError(f'L{self.l}: expert {e} is not in the v2 files (they hold {len(self.experts)})')
        return self._get(f'k{k}.{what}')[self.row[e]]

    def hidden(self, e, p, k):
        """(fp16 [6144] hidden-side vector the kernel applies, the 48 block scales the checksum covers):
        gate / up input = signs.in x su_block, down output = signs.out x sv_block (SPEC.md 5.2)."""
        sc = self._t(e, k, BLOCK[p])
        sign = self._get(SIGNS[p])
        if sc.shape != (H // 128,) or sc.dtype != np.float16 or sign.shape != (H,):
            raise ValueError(f'L{self.l} k{k} {p}: block scales {sc.shape} {sc.dtype} / signs {sign.shape}')
        if not np.isin(sign, (-1, 1)).all():
            raise ValueError(f'L{self.l}: {SIGNS[p]} holds values other than +1 and -1')
        v = sign.astype(np.float32) * np.repeat(sc.astype(np.float32), 128)
        return v.astype(np.float16), sc

    def inter(self, e, p, k):
        v = self._t(e, k, CHAN[p])
        if v.shape != (I,) or v.dtype != np.float16:
            raise ValueError(f'L{self.l} k{k} {p} per-channel vector: {v.shape} {v.dtype}')
        return v

    def loc(self, e, p, h, k):
        """(path, absolute file offset, length) of half h of projection p of expert e at width k."""
        off, n = (int(x) for x in self._t(e, k, 'index')[PROJ.index(p), h])
        if n != half_bytes(k):
            raise ValueError(f'L{self.l} k{k} expert {e} {p} half {h}: length {n}, want {half_bytes(k)}')
        return self.file(k).path, off, n

    def half(self, e, p, h, k):
        """Trellis bytes of half h of projection p of expert e at width k, in the stored order, checksum verified."""
        pi = PROJ.index(p)
        _, off, n = self.loc(e, p, h, k)
        b = self.file(k).read_abs(off, n)
        if self.verify:
            want = int(self._t(e, k, 'checksum')[pi, h])
            got = half_digest(b, self.inter(e, p, k)[h * 128:(h + 1) * 128], self.hidden(e, p, k)[1])
            if got != want:
                raise ValueError(f'L{self.l} k{k} expert {e} {p} half {h}: xxhash64 {got:#018x} != stored {want:#018x}')
        return b


# ---------------------------------------------------------------------------------------------- widths
class Widths:
    """Chosen widths of one layer: down[e], gate[e][h], up[e][h] for e = 0..255, and the shared expert's
    sgate[h], sup[h], sdown (sdown = 0: the shared expert is not taken from the v2 files)."""

    def __init__(self, down, gate, up, shared=0, name=''):
        self.down = np.asarray(down, np.int64).reshape(N_EXP)
        self.gate = np.broadcast_to(np.asarray(gate, np.int64).reshape(N_EXP, -1), (N_EXP, NH)).copy()
        self.up = np.broadcast_to(np.asarray(up, np.int64).reshape(N_EXP, -1), (N_EXP, NH)).copy()
        if isinstance(shared, dict):
            self.sgate = np.broadcast_to(np.asarray(shared['gate'], np.int64), (NH,)).copy()
            self.sup = np.broadcast_to(np.asarray(shared['up'], np.int64), (NH,)).copy()
            self.sdown = int(shared['down'])
        else:
            self.sgate = np.full(NH, int(shared), np.int64)
            self.sup = self.sgate.copy()
            self.sdown = int(shared)
        self.name = name
        for a in (self.down, self.gate, self.up):
            if not np.isin(a, WIDTHS).all():
                raise ValueError(f'widths {name}: values outside {WIDTHS}')
        if self.sdown and not (np.isin(self.sgate, (3, 4)).all() and np.isin(self.sup, (3, 4)).all() and self.sdown in (3, 4)):
            raise ValueError(f'widths {name}: the shared expert is stored at K3 and K4 only')

    @property
    def shared(self):
        return self.sdown

    def of(self, e):
        """(gate[16], up[16], down) of expert e (256 = shared)."""
        if e == SHARED:
            return list(self.sgate), list(self.sup), self.sdown
        return list(self.gate[e]), list(self.up[e]), int(self.down[e])


def load_widths(spec, l, v1_bits=None, shared=None):
    """spec: a manifest file (SPEC.md 6.5: {"format": "exlr8-v2-manifest", "name", "layers": {"40": {"gate": [256][16],
    "up": [256][16], "down": [256], "shared": {"gate": [16], "up": [16], "down": K}}}}); home / down1 / up1 rebuilt
    from the v1 per-expert bits (home = those bits, down1 = one less, up1 = all K4; shared K4 / K3 / K4 as the shipped
    manifests); or k2 / k3 / k4 (every routed unit at that width, shared at K3 for k2 and k3, K4 for k4; tests).
    shared overrides the shared expert's width (0 = do not take it from the v2 files).
    Returns (Widths, provenance string for the pack manifest)."""
    if spec in ('k2', 'k3', 'k4'):
        k = int(spec[1])
        return Widths([k] * N_EXP, [k] * N_EXP, [k] * N_EXP, max(k, 3) if shared is None else shared, spec), spec
    if spec in ('home', 'down1', 'up1'):
        if v1_bits is None:
            raise ValueError(f'widths {spec}: needs the v1 per-expert bits (--v1 / --tr3) or a manifest file')
        b = np.asarray(v1_bits, np.int64)
        b = b if spec == 'home' else b - 1 if spec == 'down1' else np.full(N_EXP, 4)
        sw = (3 if spec == 'down1' else 4) if shared is None else shared
        return Widths(b, b, b, sw, spec), f'{spec} (from v1 bits)'
    with open(spec, 'rb') as f:
        raw = f.read()
    m = json.loads(raw)
    if m.get('format') != 'exlr8-v2-manifest':
        raise ValueError(f'{spec}: format {m.get("format")!r}, want exlr8-v2-manifest')
    e = m['layers'][str(l)]
    sh = e.get('shared', 0) if shared is None else shared
    return (Widths(e['down'], e['gate'], e['up'], sh, m.get('name', spec)),
            f'{m.get("name", "")} sha256:{hashlib.sha256(raw).hexdigest()[:16]}')


def v1_bits_of(src, l):
    """Per-expert bits of a TR3 / exlr8 v1 layer file (convert.py's rule: trellis last dim / 16)."""
    p = f'model.layers.{l}.mlp.experts.'
    return [src.shape(f'{p}{e}.gate_proj.rank0.trellis')[-1] // 16 for e in range(N_EXP)]


# ---------------------------------------------------------------------------------------------- writing
def kmix_word(gw, uw, dk, hs):
    """tr3.kmix entry of one expert on one rank: 0 when every owned fc1 unit has width dk."""
    if all(gw[h] == dk and uw[h] == dk for h in hs):
        return 0
    v = 0
    for b, h in enumerate(hs):
        v |= int(gw[h]) << (8 * b) | int(uw[h]) << (8 * b + 4)
    return v


def convert_experts_v2(l, v2, widths, writers, nr, P=None, info=None, shared=True, routed=None):
    """Write layer l's tr3.* tensors from v2 files to every writer (rank r -> PackWriter). nr = the TP size.
    shared: also store the shared expert as trellis row 256 when widths.shared and the files have it.
    routed: the routed experts to take (tests on a partial layer); the others get empty table rows (no blocks), which
    the kernels skip. Default: all 256, which the files must hold."""
    P = P or f'L{l:03d}'
    xf = dict(info or {})
    routed = set(range(N_EXP) if routed is None else routed)
    if not routed <= set(v2.experts):
        raise ValueError(f'{P}: the v2 files hold experts {v2.experts[0]}..{v2.experts[-1]} ({len(v2.experts)}), '
                         f'not all {len(routed)} requested')
    experts = list(range(N_EXP))
    if shared and widths.shared:
        need = set(widths.of(SHARED)[0]) | set(widths.of(SHARED)[1]) | {widths.shared}
        if not all(v2.has_shared(k) for k in need):
            raise ValueError(f'{P}: shared expert at widths {sorted(need)} requested, the v2 files have {v2.shared_widths}')
        experts.append(SHARED)
    E = len(experts)
    own = {r: {e: owned_halves(e, l, nr, r) if e == SHARED or e in routed else [] for e in experts} for r in writers}
    W = {e: widths.of(e) for e in experts}
    table = {r: np.zeros((E, 6), np.int64) for r in writers}
    kmix = {r: np.zeros(E, np.int64) for r in writers}
    nblk = {r: sum(len(own[r][e]) for e in experts) for r in writers}
    bs = SHARED in experts                 # block-scale hidden-side vectors go with the trellis shared expert
    NB = H // 128
    v1 = {r: np.empty((nblk[r], 2, NB if bs else H), np.float16) for r in writers}
    v2v = {r: np.empty((nblk[r], 3, 128), np.float16) for r in writers}
    v3 = {r: np.empty((nblk[r], NB if bs else H), np.float16) for r in writers}
    hid = (lambda e, p, k: v2.hidden(e, p, k)[1]) if bs else (lambda e, p, k: v2.hidden(e, p, k)[0])
    fc1_bytes = {r: sum(half_bytes(W[e][0][h]) + half_bytes(W[e][1][h]) for e in experts for h in own[r][e]) for r in writers}
    fc2_bytes = {r: sum(half_bytes(W[e][2]) * len(own[r][e]) for e in experts) for r in writers}

    # pass 1: fc1 (gate and up halves, whose grouped order is g6's fc1 block order), v1, v2[0:2]
    for r, w in writers.items():
        w.begin(f'{P}.tr3.fc1', 'u8', (fc1_bytes[r],),
                layout='tr3_fc1 per expert per block [2][384][8][TB], TB = 32 * width of (block, matrix) (tr3.kmix)', **xf)
    off = {r: 0 for r in writers}
    row = {r: 0 for r in writers}
    for i, e in enumerate(experts):
        gw, uw, dk = W[e]
        need = sorted(set().union(*(own[r][e] for r in writers)))
        gu = {h: (v2.half(e, 'gate', h, gw[h]), v2.half(e, 'up', h, uw[h])) for h in need}
        for r, w in writers.items():
            hs = own[r][e]
            table[r][i] = (off[r], 0, row[r], len(hs), dk, hs[0] if hs else 0)
            kmix[r][i] = kmix_word(gw, uw, dk, hs)
            for h in hs:
                w.write(gu[h][0])
                w.write(gu[h][1])
                v1[r][row[r], 0] = hid(e, 'gate', gw[h])
                v1[r][row[r], 1] = hid(e, 'up', uw[h])
                v2v[r][row[r], 0] = v2.inter(e, 'gate', gw[h])[h * 128:(h + 1) * 128]
                v2v[r][row[r], 1] = v2.inter(e, 'up', uw[h])[h * 128:(h + 1) * 128]
                off[r] += half_bytes(gw[h]) + half_bytes(uw[h])
                row[r] += 1
    for w in writers.values():
        w.end()

    # pass 2: fc2 (down halves, item128 -> [col16][k16]), v2[2], v3
    for r, w in writers.items():
        w.begin(f'{P}.tr3.fc2', 'u8', (fc2_bytes[r],), layout='tr3_fc2 per expert per block [384][8][TB]', **xf)
    off = {r: 0 for r in writers}
    row = {r: 0 for r in writers}
    for i, e in enumerate(experts):
        dk = W[e][2]
        need = sorted(set().union(*(own[r][e] for r in writers)))
        dn = {h: v2.half(e, 'down', h, dk) for h in need}
        svd = hid(e, 'down', dk) if need else None
        sud = v2.inter(e, 'down', dk) if need else None
        for r, w in writers.items():
            table[r][i][1] = off[r]
            for h in own[r][e]:
                w.write(down_to_g6(dn[h], dk))
                v2v[r][row[r], 2] = sud[h * 128:(h + 1) * 128]
                v3[r][row[r]] = svd
                off[r] += half_bytes(dk)
                row[r] += 1
    for r, w in writers.items():
        w.end()
        if bs:
            sg = np.stack([v2._get('signs.in'), v2._get('signs.out')]).astype(np.float16)
            w.put(f'{P}.tr3.sg', sg, 'f16', layout='[2][6144] signs.in, signs.out (+-1)', **xf)
            w.put(f'{P}.tr3.v1b', v1[r], 'f16', layout='[blk][2][48] su_block gate, up (encode of the half)', **xf)
        else:
            w.put(f'{P}.tr3.v1', v1[r], 'f16', layout='[blk][2][6144] suh gate, suh up (encode of the half)', **xf)
        w.put(f'{P}.tr3.v2', v2v[r], 'f16', layout='[blk][3][128] svh gate, svh up, suh down (half slice)', **xf)
        if bs:
            w.put(f'{P}.tr3.v3b', v3[r], 'f16', layout='[blk][48] sv_block down (expert)', **xf)
        else:
            w.put(f'{P}.tr3.v3', v3[r], 'f16', layout='[blk][6144] svh down (expert)', **xf)
        w.put(f'{P}.tr3.table', table[r], 'i64',
              layout=f'[{E}][6] fc1_byte_off, fc2_byte_off, first_blk_row, n_blocks, bits (down), first half', **xf)
        w.put(f'{P}.tr3.kmix', kmix[r], 'i64', layout=f'[{E}] fc1 width per (block b, matrix m) at bit 8b + 4m; 0 = all bits', **xf)
    return experts


def check_layer(v2):
    """Read and verify every half of every expert and width the layer's files hold. Returns a summary dict."""
    n = 0
    for k in v2.widths:
        for e in v2.experts:
            for p in PROJ:
                for h in range(NH):
                    v2.half(e, p, h, k)
                    n += 1
    for k in v2.shared_widths:
        for p in PROJ:
            for h in range(NH):
                v2.half(SHARED, p, h, k)
                n += 1
    return dict(layer=v2.l, widths=v2.widths, shared_widths=v2.shared_widths, experts=len(v2.experts), halves_verified=n)


# ---------------------------------------------------------------------------------------------- CLI: expert-only packs
def main(argv=None):
    from tools.convert.pack import PackWriter
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--check', action='store_true', help='verify every half the files hold, write nothing')
    ap.add_argument('--nr', type=int, help='TP size (3..7)')
    ap.add_argument('--out')
    ap.add_argument('--widths', default='home')
    ap.add_argument('--v1', help='TR3 / exlr8 v1 checkpoint dir, for the home / down1 / up1 widths')
    ap.add_argument('--ranks', default='all')
    ap.add_argument('--shared', default='trellis', choices=('trellis', 'off'))
    ap.add_argument('--shared-width', type=int, default=None, help='override the manifest\'s shared expert width')
    ap.add_argument('--routed', default='all', help="routed experts to take, e.g. 0-63 (tests on a partial layer); "
                    "'files' = the experts the index file holds")
    a = ap.parse_args(argv)
    if a.check:
        lay = V2Layer(a.v2, a.layer)
        print(json.dumps(check_layer(lay)))
        lay.close()
        return
    if not a.nr or not a.out:
        ap.error('--nr and --out are required unless --check')
    bits = None
    if a.v1:
        s = ST(os.path.join(a.v1, f'model-layer-{a.layer:03d}.safetensors'))
        bits = v1_bits_of(s, a.layer)
        s.close()
    wd, prov = load_widths(a.widths, a.layer, bits, a.shared_width if a.shared == 'trellis' else 0)
    ranks = list(range(a.nr)) if a.ranks == 'all' else [int(x) for x in a.ranks.split(',')]
    os.makedirs(a.out, exist_ok=True)
    meta = dict(converter='g6convert-v2experts', layers=[a.layer], nr=a.nr, v2_widths=prov,
                expert_formats={str(a.layer): 'exlr8-e4m3'})
    writers = {r: PackWriter(os.path.join(a.out, f'rank{r}.pack'), r, False, meta) for r in ranks}
    v2 = V2Layer(a.v2, a.layer)
    routed = None if a.routed == 'all' else v2.experts if a.routed == 'files' else parse_experts(a.routed)
    ex = convert_experts_v2(a.layer, v2, wd, writers, a.nr, info=dict(expert_format='exlr8-e4m3', expert_source='exlr8-v2'),
                            shared=a.shared == 'trellis', routed=routed)
    v2.close()
    for r, w in writers.items():
        m = w.close()
        print(f'rank {r}: {m["pack_bytes"]} bytes, {len(ex)} experts', flush=True)


if __name__ == '__main__':
    main()
