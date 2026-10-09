"""Write a slim copy of a rank pack: the same tensors minus slim.dropped(), 4096-aligned, deterministic (PackWriter).

python3 -m tools.convert.slimpack --qo QO.manifest.json --out DIR  rank{r}.manifest.json [...]
  -> DIR/rank{r}.pack + DIR/rank{r}.manifest.json. Each kept tensor is read from the old pack, checked against its sha256 in
  the old manifest and written byte for byte; a mismatch aborts (DIR gets no manifest). Needs free space for the whole slim
  pack beside the old one (the old pack is not touched). Run it ON the node that holds the pack: nothing is shipped.
  Idempotent: a slim pack slimmed again gives the same bytes. Run checkpack on the result, then mv over the old pack.
  --qo may be given as `all` to use layers 0-77 (when the QO pack manifest is not at hand; the QO pack must cover them).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from tools.convert import slim
from tools.convert.pack import DTYPES, PackWriter

STD = ('name', 'dtype', 'shape', 'offset', 'nbytes', 'sha256')
TOP = ('format', 'rank', 'align', 'pack_bytes', 'tensors')
CHUNK = 64 << 20


def slimpack(man_path, out_dir, qo, drop_cache=True):
    t0 = time.time()
    with open(man_path) as f:
        m = json.load(f)
    pack = man_path.rsplit('.manifest.json', 1)[0] + '.pack'
    size = os.path.getsize(pack)
    if size != m['pack_bytes']:
        raise SystemExit(f'{pack}: {size} bytes but the manifest says {m["pack_bytes"]}')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'rank{m["rank"]}.pack')
    if os.path.realpath(out) == os.path.realpath(pack):
        raise SystemExit('--out must be another directory than the pack\'s own')
    meta = {k: v for k, v in m.items() if k not in TOP}
    meta['slim_qo'] = True
    w = PackWriter(out, m['rank'], drop_cache, meta)
    fd = os.open(pack, os.O_RDONLY)
    kept = skipped = skipped_bytes = 0
    for e in sorted(m['tensors'], key=lambda e: e['offset']):
        if slim.dropped(e['name'], qo):
            skipped += 1
            skipped_bytes += e['nbytes']
            continue
        info = {k: v for k, v in e.items() if k not in STD}
        w.begin(e['name'], e['dtype'], e['shape'], **info)
        h, pos, end = hashlib.sha256(), e['offset'], e['offset'] + e['nbytes']
        while pos < end:
            b = os.pread(fd, min(CHUNK, end - pos), pos)
            if not b:
                raise SystemExit(f'{pack}: short read in {e["name"]}')
            h.update(b)
            w.write(np.frombuffer(b, np.uint8))
            pos += len(b)
        if h.hexdigest() != e['sha256']:
            raise SystemExit(f'{pack}: {e["name"]} fails its sha256, slim copy aborted')
        w.end()
        kept += 1
        if drop_cache:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    new = w.close()
    return dict(pack=out, rank=m['rank'], kept=kept, dropped=skipped, dropped_bytes=skipped_bytes, bytes_before=size,
                bytes=new['pack_bytes'], seconds=round(time.time() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--qo', required=True, help='the QO pack manifest (qo.manifest.json), or `all`')
    ap.add_argument('--out', required=True)
    ap.add_argument('manifests', nargs='+')
    a = ap.parse_args(argv)
    qo = set(range(78)) if a.qo == 'all' else slim.qo_layers(a.qo)
    if not qo:
        sys.exit(f'{a.qo}: no layer with all of the QO tensors')
    for p in a.manifests:
        print(json.dumps(slimpack(p, a.out, qo)), flush=True)


if __name__ == '__main__':
    main()
