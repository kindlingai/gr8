"""Synthetic exlr8 v2 layer files for tests (#65), in the layout of exlr8quant SPEC.md 6.3-6.4: one index file
(layer signs; per width index, checksums, block scales, per-channel vectors, error tables; heldout_tokens) and one
trellis file per width. Random trellis bytes (every bit pattern is a valid tail-biting stream), random signs, positive
block scales, random per-channel vectors. 64 KiB layout: header padded with spaces, every tensor on a 64 KiB offset,
`__pad.N` tensors in the gaps.

python3 -m tools.convert.v2synth OUT_DIR --layer 40 [--widths 2,3,4] [--experts 256] [--seed 1] [--no-shared]
        [--pad-halves]
--experts N < 256 fills only the first N experts with random bytes (the rest are zero bytes, still checksummed):
faster to write, same layout.
"""
import argparse
import json
import os
import struct

import numpy as np

from tools.convert import v2
from tools.convert.xxh import xxh64

ALIGN = 64 << 10
NP2ST = {np.dtype(np.uint8): 'U8', np.dtype(np.float16): 'F16', np.dtype(np.int64): 'I64',
         np.dtype(np.uint64): 'U64', np.dtype(np.float32): 'F32', np.dtype(np.int32): 'I32', np.dtype(np.int8): 'I8'}


def write_st(path, tensors, meta):
    """tensors: list of (name, array) in file order. Every tensor starts on a 64 KiB offset of the file."""
    # the header size depends on the pad tensors, which depend on the header size: iterate to a fixed point
    hsize = ALIGN
    while True:
        h, items, off = {'__metadata__': meta}, [], 0
        npad = 0
        for name, a in tensors:
            a = np.ascontiguousarray(a)
            gap = (-(8 + hsize + off)) % ALIGN
            if gap:
                pn = f'__pad.{npad}'
                npad += 1
                h[pn] = dict(dtype='U8', shape=[gap], data_offsets=[off, off + gap])
                items.append(np.zeros(gap, np.uint8))
                off += gap
            h[name] = dict(dtype=NP2ST[a.dtype], shape=list(a.shape), data_offsets=[off, off + a.nbytes])
            items.append(a)
            off += a.nbytes
        hj = json.dumps(h, separators=(',', ':')).encode()
        need = len(hj) + (-(8 + len(hj))) % ALIGN
        if need == hsize:
            break
        hsize = need
    hj = hj + b' ' * (hsize - len(hj))
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', hsize))
        f.write(hj)
        for a in items:
            f.write(memoryview(a).cast('B'))


def tensor_offsets(path):
    """name -> absolute file offset of each tensor's data."""
    with open(path, 'rb') as f:
        n, = struct.unpack('<Q', f.read(8))
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return {k: 8 + n + v['data_offsets'][0] for k, v in h.items()}


def make_layer(out, l, widths=(2, 3, 4), n_fill=256, seed=1, shared=True, pad_halves=False, shared_widths=(3, 4)):
    """Write the layer's index file and trellis files. Read them back with v2.V2Layer."""
    rng = np.random.default_rng(seed + 1000 * l)
    os.makedirs(out, exist_ok=True)
    signs = {'in': np.where(rng.random(v2.H) < 0.5, -1, 1).astype(np.int8),
             'out': np.where(rng.random(v2.H) < 0.5, -1, 1).astype(np.int8)}
    index = [('signs.in', signs['in']), ('signs.out', signs['out'])]
    heldout = []
    groups = [('', v2.N_EXP, widths)] + ([('shared.', 1, shared_widths)] if shared else [])
    allw = sorted(set(widths) | (set(shared_widths) if shared else set()))
    for k in allw:
        hb = v2.half_bytes(k)
        S = -(-hb // (64 << 10)) * (64 << 10) if pad_halves else hb
        trel, side = [], []
        for s, ne, ws in groups:
            if k not in ws:
                continue
            blk = {p: (rng.random((ne, v2.H // 128)) * 0.02 + 0.005).astype(np.float16) for p in v2.PROJ}
            chan = {p: (rng.standard_normal((ne, v2.I)) * 0.5 + np.sign(rng.standard_normal((ne, v2.I)))).astype(np.float16)
                    for p in v2.PROJ}
            ck = np.zeros((ne, 3, v2.NH), np.uint64)
            for pi, p in enumerate(v2.PROJ):
                t = np.zeros((ne, v2.NH, S), np.uint8)
                nf = ne if s else min(ne, n_fill)
                t[:nf, :, :hb] = rng.integers(0, 256, (nf, v2.NH, hb), dtype=np.uint8)
                for e in range(ne):
                    for h in range(v2.NH):
                        ck[e, pi, h] = xxh64(t[e, h, :hb], chan[p][e, h * 128:(h + 1) * 128], blk[p][e])
                trel.append((f'{s}trellis.{p}', t))
            side.append((s, ne, ck, blk, chan))
        meta = {'format': 'exlr8-v2', 'contents': 'trellis', 'scheme': 'exlr8-e4m3-v2', 'layer': str(l), 'k': str(k),
                'hessian': 'synthetic', 'gate_up_order': v2.GATE_UP_ORDER, 'down_order': v2.DOWN_ORDER,
                'pad_halves': 'true' if pad_halves else 'false', 'index_file': v2.INDEX_FILE.format(l=l)}
        path = os.path.join(out, v2.WIDTH_FILE.format(l=l, k=k))
        write_st(path, trel, meta)
        offs = tensor_offsets(path)
        for s, ne, ck, blk, chan in side:
            idx = np.zeros((ne, 3, v2.NH, 2), np.int64)
            for pi, p in enumerate(v2.PROJ):
                base = offs[f'{s}trellis.{p}']
                for e in range(ne):
                    for h in range(v2.NH):
                        idx[e, pi, h] = (base + (e * v2.NH + h) * S, hb)
            pre = f'{s}k{k}.'
            index += [(pre + 'index', idx), (pre + 'checksum', ck)]
            index += [(pre + v2.BLOCK[p], blk[p]) for p in v2.PROJ]
            index += [(pre + v2.CHAN[p], chan[p]) for p in v2.PROJ]
            index += [(pre + 'err_gate', rng.random((ne, 16), np.float32)), (pre + 'err_up', rng.random((ne, 16), np.float32)),
                      (pre + 'err_down', rng.random(ne, np.float32))]
            if s and not heldout:
                heldout.append(('shared.heldout_tokens', np.array([131072], np.int32)))
    index.append(('heldout_tokens', rng.integers(900, 60000, v2.N_EXP).astype(np.int32)))
    index += heldout
    write_st(os.path.join(out, v2.INDEX_FILE.format(l=l)), index,
             {'format': 'exlr8-v2', 'contents': 'index', 'scheme': 'exlr8-e4m3-v2', 'layer': str(l),
              'experts': f'0-{v2.N_EXP - 1}', 'widths': ','.join(map(str, widths)),
              'shared_widths': ','.join(map(str, shared_widths)) if shared else '', 'hessian': 'synthetic',
              'gate_up_order': v2.GATE_UP_ORDER, 'down_order': v2.DOWN_ORDER,
              'pad_halves': 'true' if pad_halves else 'false'})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--layer', type=int, default=40)
    ap.add_argument('--widths', default='2,3,4')
    ap.add_argument('--experts', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no-shared', action='store_true')
    ap.add_argument('--pad-halves', action='store_true')
    a = ap.parse_args(argv)
    make_layer(a.out, a.layer, tuple(int(x) for x in a.widths.split(',')), a.experts, a.seed, not a.no_shared,
               a.pad_halves)


if __name__ == '__main__':
    main()
