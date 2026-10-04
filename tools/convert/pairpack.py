#!/usr/bin/env python3
"""Pair-owner QO prefill pack (docs/DESIGN-QO-PAIR.md section 1, G6_PF_OWN=4 with --qp-pack, TP4 only): per rank r,
layers 0-77, the 32 heads its head pair {r & ~1, r | 1} holds, assembled from the four rank packs' head slices. Stdlib
only (the Sparks have no numpy). The served TP4 default builds the same bytes on the fly instead (G6_PF_QP_ROLL,
engine/runtime/qo_weights.cuh), because ~6 GB of resident pair weights per rank do not fit next to a TP4 rank pack.

Layer l: pair member m holds head group g(m) = (m - l) mod 4, heads 16 g .. 16 g + 16 (g6nr::start(64, g), NR=4). The pair's
groups in LOGICAL HEAD ORDER (lower group first): ga = min(g(a), g(b)), gb = max. Output tensors, named L%03d.qp.<t>:
  q_b.mx.codes [8192][2048] u8, q_b.mx.scales [8192][64] u8        rows: group ga's 4,096 rows, then gb's   } byte-identical
  o_proj.mx.codes [6144][8192] u8, o_proj.mx.scales [6144][256] u8  each row: ga's columns, then gb's        } on a and b
  kv_b.uk [16][192][512] bf16, kv_b.uv [16][256][512] bf16          the PARTNER's (r ^ 1) own slices, bytes unchanged
(the rank's own kv_b stays in its rank pack).
Per rank and layer 76,546,048 B; 78 layers = 5.97 GB (+ alignment).

python3 pairpack.py OUTDIR SRC0 SRC1 SRC2 SRC3 [--layers 0-77] [--ranks 0,1,2,3]
  SRCr as qopack.py (local rank manifest, or HOST:PATH streamed with qopack.py --extract). Every source tensor's sha256 and
  head range are checked against its manifest. Writes OUTDIR/qp.rank<r>.pack + qp.rank<r>.manifest.json per requested rank
  (format g6pack-v1, 4096-byte alignment, deterministic).
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qopack as Q

NR, HG = 4, 16
OUT = {'q_b.mx.codes': ([8192, 2048], 'u8'), 'q_b.mx.scales': ([8192, 64], 'u8'),
       'o_proj.mx.codes': ([6144, 8192], 'u8'), 'o_proj.mx.scales': ([6144, 256], 'u8'),
       'kv_b.uk': ([16, 192, 512], 'bf16'), 'kv_b.uv': ([16, 256, 512], 'bf16')}


def group(m, l):
    return (m - l) % NR


def pair_order(r, l):
    """(ga, member a), (gb, member b): the pair's two groups, lower group first"""
    a, b = r & ~1, r | 1
    return sorted(((group(a, l), a), (group(b, l), b)))


class Out:
    def __init__(self, path):
        self.path, self.fd, self.off, self.ents = path, os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), 0, []

    def raw(self, b):
        mv, n = memoryview(b), 0
        while n < len(mv):
            n += os.write(self.fd, mv[n:])
        self.off += len(mv)

    def tensor(self, name, parts, shape, dt, heads, layout):
        """parts: a list of byte strings written back to back as one tensor"""
        self.raw(bytes((-self.off) % Q.ALIGN))
        hs, o0 = hashlib.sha256(), self.off
        for b in parts:
            hs.update(b); self.raw(b)
        n = 1
        for s in shape:
            n *= s
        if self.off - o0 != n * (2 if dt == 'bf16' else 1):
            sys.exit(f'{name}: wrote {self.off - o0} bytes, expected {n * (2 if dt == "bf16" else 1)}')
        self.ents.append(dict(name=name, dtype=dt, shape=shape, offset=o0, nbytes=self.off - o0, sha256=hs.hexdigest(),
                              layout=layout, heads=heads))


def main():
    args = sys.argv[1:]
    lspec, rspec = '0-77', '0,1,2,3'
    for k in ('--layers', '--ranks'):
        if k in args:
            i = args.index(k)
            if k == '--layers':
                lspec = args[i + 1]
            else:
                rspec = args[i + 1]
            del args[i:i + 2]
    if len(args) != 1 + NR:
        sys.exit('usage: pairpack.py OUTDIR SRC0 SRC1 SRC2 SRC3 [--layers 0-77] [--ranks 0,1,2,3]')
    outdir, srcs = args[0], args[1:]
    layers, ranks = Q.parse_layers(lspec), sorted(set(int(x) for x in rspec.split(',')))
    if not ranks or any(r < 0 or r >= NR for r in ranks) or any(l < 0 or l > 77 for l in layers):
        sys.exit(f'ranks {rspec} / layers {lspec}: ranks 0-3, layers 0-77 (the MTP layer 78 keeps the head-split path)')
    E = Q.qo_edges(NR)
    R = [Q.Remote(s, layers, lspec) if ':' in s else Q.Local(s, layers) for s in srcs]
    for r, s in enumerate(R):
        if s.man['rank'] != r:
            sys.exit(f'source {r} is rank {s.man["rank"]}')
    os.makedirs(outdir, exist_ok=True)
    O = {r: Out(f'{outdir}/qp.rank{r}.pack') for r in ranks}
    for l in layers:
        got = {}   # (member, t) -> (entry, bytes)
        for m in range(NR):
            g = group(m, l)
            for t in Q.SRC:
                e, b = R[m].next()
                if e['name'] != f'L{l:03d}.{t}':
                    sys.exit(f'stream order: got {e["name"]}')
                if hashlib.sha256(b).hexdigest() != e['sha256']:
                    sys.exit(f'rank {m} {e["name"]}: sha256 mismatch')
                if e.get('heads') != [E[g], E[g + 1]]:
                    sys.exit(f'rank {m} {e["name"]}: heads {e.get("heads")} are not group {g}')
                got[m, t] = (e, b)
        for r in ranks:
            (ga, a), (gb, b) = pair_order(r, l)
            hd = [[E[ga], E[ga + 1]], [E[gb], E[gb + 1]]]
            o, lay = O[r], lambda t: got[a, t][0]['layout']
            for t in ('q_b.mx.codes', 'q_b.mx.scales'):
                o.tensor(f'L{l:03d}.qp.{t}', [got[a, t][1], got[b, t][1]], *OUT[t], hd, lay(t))
            for t in ('o_proj.mx.codes', 'o_proj.mx.scales'):
                wa, wb = got[a, t][0]['shape'][1], got[b, t][0]['shape'][1]
                if wa + wb != OUT[t][0][1]:
                    sys.exit(f'L{l:03d}.{t}: widths {wa} + {wb}')
                ca, cb = got[a, t][1], got[b, t][1]
                o.tensor(f'L{l:03d}.qp.{t}', [ca[w * wa:(w + 1) * wa] + cb[w * wb:(w + 1) * wb] for w in range(6144)],
                         *OUT[t], hd, lay(t))
            p = r ^ 1
            for t in ('kv_b.uk', 'kv_b.uv'):
                e, bb = got[p, t]
                if e['shape'] != OUT[t][0]:
                    sys.exit(f'rank {p} {e["name"]}: shape {e["shape"]}')
                o.tensor(f'L{l:03d}.qp.{t}', [bb], *OUT[t], e['heads'], e['layout'])
        for o in O.values():
            os.fdatasync(o.fd)
            os.posix_fadvise(o.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        print(f'L{l:03d} done, {sum(o.off for o in O.values()) / 1e9:.2f} GB', flush=True)
    for s in R:
        if isinstance(s, Q.Remote) and s.p.wait() != 0:
            sys.exit('remote extract failed')
    for r, o in O.items():
        o.raw(bytes((-o.off) % Q.ALIGN))
        os.fsync(o.fd); os.close(o.fd)
        man = dict(format='g6pack-v1', rank=r, align=Q.ALIGN, pack_bytes=o.off, tensors=o.ents, qp=True, pair=[r & ~1, r | 1],
                   layers=layers, converter='pairpack-v1', sources=[s.man.get('converter') for s in R])
        mp = f'{outdir}/qp.rank{r}.manifest.json'
        with open(mp + '.tmp', 'w') as f:
            json.dump(man, f, indent=1, sort_keys=True); f.write('\n')
        os.replace(mp + '.tmp', mp)
        print(f'wrote {o.path} ({o.off} bytes, {len(o.ents)} tensors) + {mp}')


if __name__ == '__main__':
    main()
