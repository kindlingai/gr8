"""Per-rank index slices of the exlr8 v2 files: what one TP rank needs to fetch and load its halves at one width,
without reading the layer index files.

For every TP size N, rank r and width K, one safetensors file tp<N>/k<K>/rank<r>.safetensors holds, per layer L that
has width K (L<LLL>. prefix):

  units    I16 [M, 2]       (expert, half) of every unit rank r owns: (16 e + h + L) mod N == r, e = 256 is the shared
                            expert (when stored at K), sorted by expert then half
  offset   I64 [M, 3]       absolute byte offset of the unit's gate, up and down half in exlr8-layer-LLL.k<K>.safetensors;
                            each half is 98304 * K bytes (metadata half_bytes)
  checksum U64 [M, 3]       the index file's xxhash64 of each half (SPEC.md 6.4: trellis bytes, its 128 per-channel
                            values, then the projection's 48 block scales)
  chan     F16 [M, 3, 128]  gate svh, up svh and down suh of the half's 128 channels
  block    F16 [E, 3, 48]   gate su_block, up su_block and down sv_block of every expert (E = 256, or 257 with the
                            shared expert as row 256); a rank owns halves of every expert when N <= 16
  signs    I8  [2, 6144]    the layer's signs.in and signs.out

Metadata: format, version, tp, rank, k, layers, half_bytes, rule, files, source, and, with --set-id, set_id (the weight
set these slices were cut from, see SET_ID_RULE) and index_sha256 (JSON: layer -> sha256 of its index file).

The offsets follow 65536 + ((p * 256 + e) * 16 + h) * half_bytes in every published file (p = 0 gate, 1 up, 2 down;
the shared expert's halves follow the routed ones); they are copied from the index files, and the generator checks them
against that rule.

python3 -m tools.convert.v2slices --v2 DIR --out DIR [--tp 3-7] [--widths 2,3,4] [--layers auto] [--set-id FILE]
    FILE: JSON {"set_id": ..., "index_sha256": {"exlr8-layer-LLL.index.safetensors": sha256, ...}}
"""
import argparse
import json
import os
import struct

import numpy as np

from tools.convert import v2
from tools.convert.st import ST

ST_DTYPE = {np.dtype(np.int8): 'I8', np.dtype(np.int16): 'I16', np.dtype(np.int64): 'I64',
            np.dtype(np.uint64): 'U64', np.dtype(np.float16): 'F16'}
FORMAT = 'exlr8-v2-index-slice'
SET_ID_RULE = ('sha256 of the lines "<path> <size> <sha256>" (one per weight file: experts/*.safetensors and '
               'dense/*.safetensors, sorted by path, each ending in a newline)')
PROJ = ('gate', 'up', 'down')
CHAN = ('gate.svh', 'up.svh', 'down.suh')
BLOCK = ('gate.su_block', 'up.su_block', 'down.sv_block')


def write_safetensors(path, tensors, meta):
    """tensors: {name: numpy array}. Data is packed in order of decreasing item size, so every tensor is aligned."""
    order = sorted(tensors, key=lambda n: (-tensors[n].dtype.itemsize, n))
    head, off = {}, 0
    for n in order:
        a = tensors[n]
        head[n] = {'dtype': ST_DTYPE[a.dtype], 'shape': list(a.shape), 'data_offsets': [off, off + a.nbytes]}
        off += a.nbytes
    head['__metadata__'] = {k: str(v) for k, v in meta.items()}
    hb = json.dumps(head, separators=(',', ':')).encode()
    hb += b' ' * (-len(hb) % 8)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<Q', len(hb)))
        f.write(hb)
        for n in order:
            f.write(np.ascontiguousarray(tensors[n]).tobytes())
    os.replace(tmp, path)


def layer_tensors(idx, L, k, n):
    """{rank: {name: array}} for layer L at width k, TP n (idx: the layer's index file, an ST)."""
    shared = f'shared.k{k}.index' in idx
    E = v2.N_EXP + (1 if shared else 0)
    get = lambda name, sh: idx.get(f'shared.k{k}.{name}' if sh else f'k{k}.{name}')
    index = np.concatenate([get('index', False)] + ([get('index', True)] if shared else []))       # [E, 3, 16, 2]
    check = np.concatenate([get('checksum', False)] + ([get('checksum', True)] if shared else []))  # [E, 3, 16]
    chan = np.stack([np.concatenate([get(c, False)] + ([get(c, True)] if shared else [])) for c in CHAN], 1)   # [E,3,2048]
    block = np.stack([np.concatenate([get(b, False)] + ([get(b, True)] if shared else [])) for b in BLOCK], 1)  # [E,3,48]
    signs = np.stack([idx.get('signs.in'), idx.get('signs.out')]).astype(np.int8)
    hb = v2.half_bytes(k)
    e_, p_, h_ = np.meshgrid(np.arange(E), np.arange(3), np.arange(16), indexing='ij')
    want = np.where(e_ < v2.N_EXP, 65536 + ((p_ * v2.N_EXP + e_) * 16 + h_) * hb,     # routed: [proj][expert][half]
                    65536 + (3 * v2.N_EXP * 16 + p_ * 16 + h_) * hb)                    # shared: after all routed halves
    if not (index[..., 0] == want).all() or not (index[..., 1] == hb).all():
        raise ValueError(f'L{L} k{k}: half offsets do not follow the published layout')
    owner = (16 * np.arange(E)[:, None] + np.arange(16)[None, :] + L) % n                            # [E, 16]
    out = {}
    for r in range(n):
        e, h = np.nonzero(owner == r)                                                             # sorted by e, then h
        P = f'L{L:03d}.'
        out[r] = {P + 'units': np.stack([e, h], 1).astype(np.int16),
                  P + 'offset': index[e, :, h, 0].astype(np.int64),
                  P + 'checksum': check[e, :, h].astype(np.uint64),
                  P + 'chan': np.stack([chan[e[:, None], p, h[:, None] * 128 + np.arange(128)] for p in range(3)], 1),
                  P + 'block': block,
                  P + 'signs': signs}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tp', default='3-7')
    ap.add_argument('--widths', default='2,3,4')
    ap.add_argument('--layers', default='auto')
    ap.add_argument('--source', default='', help='provenance for the metadata (repo@revision)')
    ap.add_argument('--set-id', help='JSON file with set_id and index_sha256 (file name -> sha256)')
    a = ap.parse_args(argv)
    lo, _, hi = a.tp.partition('-')
    tps = list(range(int(lo), int(hi or lo) + 1))
    widths = [int(x) for x in a.widths.split(',')]
    layers = [l for l in range(79) if os.path.exists(os.path.join(a.v2, v2.INDEX_FILE.format(l=l)))] \
        if a.layers == 'auto' else [int(x) for x in a.layers.split(',')]
    sid = json.load(open(a.set_id)) if a.set_id else None
    acc = {(n, k, r): {} for n in tps for k in widths for r in range(n)}
    have = {k: [] for k in widths}
    for L in layers:
        idx = ST(os.path.join(a.v2, v2.INDEX_FILE.format(l=L)))
        for k in widths:
            if f'k{k}.index' not in idx:
                continue
            have[k].append(L)
            for n in tps:
                for r, t in layer_tensors(idx, L, k, n).items():
                    acc[(n, k, r)].update(t)
        idx.close()
        print(f'layer {L} done', flush=True)
    for (n, k, r), t in acc.items():
        d = os.path.join(a.out, f'tp{n}', f'k{k}')
        os.makedirs(d, exist_ok=True)
        meta = dict(format=FORMAT, version=1, tp=n, rank=r, k=k, layers=','.join(map(str, have[k])),
                    half_bytes=v2.half_bytes(k), rule='(16 * expert + half + layer) mod tp',
                    files=f'experts/exlr8-layer-LLL.k{k}.safetensors', source=a.source)
        if sid:
            meta.update(set_id=sid['set_id'], set_id_rule=SET_ID_RULE, index_sha256=json.dumps(
                {str(L): sid['index_sha256'][v2.INDEX_FILE.format(l=L)] for L in have[k]}, separators=(',', ':')))
        write_safetensors(os.path.join(d, f'rank{r}.safetensors'), t, meta)
    print(f'wrote {len(acc)} slices for tp {tps}, widths {widths}')


if __name__ == '__main__':
    main()
