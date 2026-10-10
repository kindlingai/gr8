"""NumPy reference of one rank's routed-expert partial, against a k1_v2_test --dump.

python3 engine/ref/k1_v2_ref.py PACKDIR/rank0 LAYER DUMP [--tol 0.02]

The rank partial of token m is sum_k w[m][k] * (the same for row 256, weight 1, when the dump says the trellis shared
expert is in the output) + sum over the expert's blocks on this rank of
    svh_d[row] * H128(z),  z = f16(H128(f16(silu(g) * u) * suh_d) @ Wd_block)          (fp16 operands, fp32 sums)
    g, u = f16(f16(H128(f16(f16(H128(x * suh)) @ W_block))) * svh)                       (per matrix)
with the pack's tr3.table / tr3.kmix / tr3.v1 / v2 / v3 and tile layouts (convert.py, tools/convert/v2.py), the
codebook rounded to E4M3 for exlr8 layers. It follows the kernel's rounding points; sums differ in order, so the
check is relative: max |out - ref| / max |ref| per token.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g6ref  # noqa: E402

H = 6144
TB1 = 384 * 8 * 32      # bytes of one fc1 matrix or fc2 block per bit of width


def e4m3_round(v):
    """fp16 -> E4M3 (round to nearest even, saturating finite) -> fp16, vectorized."""
    f = v.astype(np.float32)
    a = np.abs(f)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    e = np.maximum(e, -6.0)                       # subnormal step 2^-9 below 2^-6
    q = 2.0 ** (e - 3)
    r = np.round(a / q) * q                       # np.round is round-half-even
    r = np.minimum(r, 448.0)
    return (np.sign(f) * r).astype(np.float16)


def unpack(tiles, k, e4):
    """[kt][nt][32k] uint8 -> W [kt*16][nt*16] fp32."""
    t16 = np.ascontiguousarray(tiles).view(np.int16)
    w = g6ref.tr3_unpack(t16)
    if e4:
        w = e4m3_round(w)
    return w.astype(np.float32)


def load(base, layer):
    m = json.load(open(base + '.manifest.json'))
    raw = np.memmap(base + '.pack', np.uint8, 'r')
    T, fmt = {}, None
    for t in m['tensors']:
        n = t['name']
        if not n.startswith(f'L{layer:03d}.tr3.'):
            continue
        dt = {'u8': np.uint8, 'f16': np.float16, 'i64': np.int64}[t['dtype']]
        T[n.split('.tr3.', 1)[1]] = np.frombuffer(raw[t['offset']:t['offset'] + t['nbytes']], dt).reshape(t['shape'])
        fmt = t.get('expert_format', fmt)
    return T, fmt == 'exlr8-e4m3'


def expert_partial(T, e, x, e4):
    tr = T['table'][e]
    o1, o2, row, nb, dk = (int(v) for v in tr[:5])
    km = int(T['kmix'][e]) if 'kmix' in T else 0
    y = np.zeros((x.shape[0], H), np.float32)
    off = o1
    for b in range(nb):
        r = row + b
        gu = []
        for mat in range(2):
            k = (km >> (4 * (2 * b + mat))) & 15 if km else dk
            w = unpack(T['fc1'][off:off + k * TB1].reshape(384, 8, 32 * k), k, e4)      # [6144][128]
            off += k * TB1
            suh = (T['sg'][0].astype(np.float32) * np.repeat(T['v1b'][r, mat].astype(np.float32), 128) if 'v1b' in T
                   else T['v1'][r, mat].astype(np.float32))
            xr = g6ref.f16(g6ref.had128(x.astype(np.float32) * suh)).astype(np.float32)
            acc = g6ref.f16(xr @ w).astype(np.float32)
            gu.append(g6ref.f16(g6ref.had128(acc) * T['v2'][r, mat].astype(np.float32)).astype(np.float32))
        act = g6ref.f16(g6ref.silu(gu[0]) * gu[1]).astype(np.float32) * T['v2'][r, 2].astype(np.float32)
        h2 = g6ref.f16(g6ref.had128(act)).astype(np.float32)
        d = T['fc2'][o2 + b * dk * TB1:o2 + (b + 1) * dk * TB1].reshape(384, 8, 32 * dk).transpose(1, 0, 2)
        wd = unpack(d, dk, e4)                                                              # [128][6144]
        z = g6ref.f16(h2 @ wd).astype(np.float32)
        svd = (T['sg'][1].astype(np.float32) * np.repeat(T['v3b'][r].astype(np.float32), 128) if 'v3b' in T
               else T['v3'][r].astype(np.float32))
        y += g6ref.had128(z) * svd
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack')
    ap.add_argument('layer', type=int)
    ap.add_argument('dump')
    ap.add_argument('--tol', type=float, default=0.02)
    a = ap.parse_args()
    T, e4 = load(a.pack, a.layer)
    raw = open(a.dump, 'rb').read()
    M, flags = np.frombuffer(raw[:8], np.int32)
    o = 8
    x = np.frombuffer(raw[o:o + M * H * 2], np.float16).reshape(M, H); o += M * H * 2
    te = np.frombuffer(raw[o:o + M * 32], np.int32).reshape(M, 8); o += M * 32
    tw = np.frombuffer(raw[o:o + M * 32], np.float32).reshape(M, 8); o += M * 32
    out = np.frombuffer(raw[o:o + M * H * 2], np.float16).reshape(M, H).astype(np.float32)
    ref = np.zeros((M, H), np.float32)
    cache = {}
    for m in range(M):
        for k in range(8):
            e = int(te[m, k])
            if (e, m) not in cache:
                cache[(e, m)] = expert_partial(T, e, x[m:m + 1], e4)[0]
            ref[m] += tw[m, k] * cache[(e, m)]
        if flags & 2:                         # the trellis shared expert, row 256, weight 1
            ref[m] += expert_partial(T, 256, x[m:m + 1], e4)[0]
    rel = np.abs(out - ref).max(1) / np.maximum(np.abs(ref).max(1), 1e-30)
    cos = (out * ref).sum(1) / np.maximum(np.linalg.norm(out, axis=1) * np.linalg.norm(ref, axis=1), 1e-30)
    ok = bool((rel < a.tol).all())
    print(json.dumps(dict(M=int(M), shared=bool(flags & 2), max_rel=float(rel.max()), min_cos=float(cos.min()), ref_absmax=float(np.abs(ref).max()),
                          e4m3=e4, kmix=bool('kmix' in T and T['kmix'].any()), ok=ok)))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
