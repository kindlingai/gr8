"""Build v2 rank packs from an existing rank pack and the exlr8 v2 files, without converting (#65).

Everything outside the v2 experts already sits in a converted pack (a TR3 / exlr8 v1 pack from convert.py), and the
expert halves already sit in the v2 files. This writes, per rank, a manifest plus a sparse pack and rank<r>.sources
(PackWriter refs, engine/loader g6::Sources): every base tensor is a reference into the base pack, under its own
sha256, and every v2 layer's tr3.* tensors are references into the v2 files (tools/convert/v2.py, refs=True). Only the
small v2 side tensors (vectors, table, kmix) are written into the new pack. In each v2 layer the base pack's tr3.* and
shared-expert records (L<l>.shared.*) are dropped, as convert.py --v2 does with the trellis shared expert, and the v2
tensors take their place. The result is what convert.py --v2 --v2-refs gives for the same base pack, minutes instead of
hours: the v2 halves are only read, checked (xxhash64) and hashed.

python3 -m tools.convert.v2pack --base PACKDIR --v2 V2DIR --widths MANIFEST --out OUTDIR --ranks 0 \
        [--layers auto|3,4,..] [--root DIR] [--src DIR]
    --root: the directory the engine resolves the sources against (default: --src). --src: a common ancestor of
    PACKDIR and V2DIR as this process sees them (default: their common path). Paths in rank<r>.sources are relative
    to it, so run with the files at the same paths as on the engine host, or pass --root.
"""
import argparse
import json
import os

from tools.convert import v2
from tools.convert.pack import PackWriter

SKIP = ('tr3.', 'shared.')         # per v2 layer: the base pack's routed experts and shared-expert records


def v2_layers(v2dir, want):
    have = [l for l in range(79) if os.path.exists(os.path.join(v2dir, v2.INDEX_FILE.format(l=l)))]
    if want == 'auto':
        return have
    ls = [int(x) for x in want.split(',')]
    if not set(ls) <= set(have):
        raise SystemExit(f'--layers: {sorted(set(ls) - set(have))} have no v2 files in {v2dir}')
    return ls


def build_rank(r, base, v2dir, widths, out, layers, src, root, log=print):
    with open(os.path.join(base, f'rank{r}.manifest.json')) as f:
        bm = json.load(f)
    nr = bm.get('nr', 6)
    base_pack = os.path.join(base, f'rank{r}.pack')
    meta = {k: v for k, v in bm.items() if k not in ('format', 'rank', 'align', 'pack_bytes', 'tensors', 'sources')}
    meta['v2_base'] = os.path.basename(os.path.abspath(base))
    w = PackWriter(os.path.join(out, f'rank{r}.pack'), r, False, meta, refs=(src, root))
    done, prefixes = set(), {l: f'L{l:03d}.' for l in layers}
    for e in sorted(bm['tensors'], key=lambda t: t['offset']):
        l = next((l for l, p in prefixes.items() if e['name'].startswith(p)), None)
        if l is not None and e['name'][len(prefixes[l]):].startswith(SKIP):
            if l not in done:              # the v2 experts go where the base layer's experts / records began
                wd, prov = v2.load_widths(widths, l)
                lay = v2.V2Layer(v2dir, l)
                v2.convert_experts_v2(l, lay, wd, {r: w}, nr, P=f'L{l:03d}', refs=True,
                                      info=dict(expert_format='exlr8-e4m3', expert_source='exlr8-v2'))
                lay.close()
                w.meta.setdefault('expert_formats', {})[str(l)] = 'exlr8-e4m3'
                w.meta.setdefault('v2_layers', {})[str(l)] = prov
                done.add(l)
            continue
        w.ref_entry(e, base_pack, e['offset'])
    missing = set(layers) - done
    if missing:
        raise SystemExit(f'rank {r}: the base pack has no tr3 / shared tensors for v2 layers {sorted(missing)}')
    m = w.close()
    filled = sum(s[1] for s in w.segments)
    log(f'rank {r}: {m["pack_bytes"]} bytes, {len(m["tensors"])} tensors, {len(w.segments)} source ranges '
        f'({filled / 1e9:.2f} GB), {os.stat(w.path).st_blocks * 512 / 1e9:.2f} GB on disk')
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='dir with the base rank<r>.manifest.json / rank<r>.pack')
    ap.add_argument('--v2', required=True, help='exlr8 v2 files dir (all of them, or each rank its own halves)')
    ap.add_argument('--widths', required=True, help='a width manifest file (exlr8-manifest-*.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ranks', required=True, help='comma list')
    ap.add_argument('--layers', default='auto', help='comma list, or auto = every layer with v2 files')
    ap.add_argument('--src', help='common ancestor of --base and --v2 (default: their common path)')
    ap.add_argument('--root', help='--src as the engine host sees it (default: --src)')
    a = ap.parse_args(argv)
    base, v2dir = os.path.abspath(a.base), os.path.abspath(a.v2)
    src = os.path.abspath(a.src) if a.src else os.path.commonpath([base, v2dir])
    os.makedirs(a.out, exist_ok=True)
    layers = v2_layers(v2dir, a.layers)
    for r in [int(x) for x in a.ranks.split(',')]:
        build_rank(r, base, v2dir, a.widths, a.out, layers, src, a.root or src)


if __name__ == '__main__':
    main()
