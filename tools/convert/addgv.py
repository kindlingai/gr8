"""Append the decode GEMV record tensors (g6ref.mxfp8_record / bf16_record) to an already written pack, in place.

Same tensors, names, shapes, sha256 and manifest fields as `convert.py --gv-records`; only their position differs
(here they sit at the end of the pack, in the order of their row-major twins). Records the pack already has are
skipped, so a pack from an older addgv gets only the missing ones. Reads the row-major tensors from the
pack itself, so it needs no TR3/D8 sources. The pack bytes below the old pack_bytes are not touched; the manifest is
replaced last, so an interrupted run leaves the old manifest plus trailing bytes, which this tool refuses.

python3 -m tools.convert.addgv DIR/rank{r}.manifest.json [...]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine', 'ref'))
import g6ref  # noqa: E402

from tools.convert.pack import DTYPES, PackWriter  # noqa: E402

MX = ('kv_a', 'q_b', 'idx.wq_b', 'lm_head', 'eh')                       # .mx.codes/.mx.scales -> .mx.gv
# ... and these only in a layer that lacks the int4 twin named here (the MTP layer: its decode path reads MXFP8)
MX_NO_INT4 = {'q_a': 'q_a.gptq', 'o_proj': 'o_proj.gptq', 'shared.gate': 'shared.gptq.fc1',
              'shared.up': 'shared.gptq.fc1', 'shared.down': 'shared.gptq.fc1'}
BF = ('router.w', 'kv_b.uk', 'kv_b.uv', 'idx.wk', 'idx.weights_proj')   # bf16 row-major -> .gv (rows as stored)
STD = ('name', 'dtype', 'shape', 'offset', 'nbytes', 'sha256', 'layout')
TOP = ('format', 'rank', 'align', 'pack_bytes', 'tensors')


def addgv(man_path):
    t0 = time.time()
    m = json.load(open(man_path))
    pack = man_path.rsplit('.manifest.json', 1)[0] + '.pack'
    size = os.path.getsize(pack)
    if size != m['pack_bytes']:
        raise SystemExit(f'{pack}: {size} bytes but the manifest says {m["pack_bytes"]} (interrupted addgv? check, then '
                         f'`truncate -s {m["pack_bytes"]} {pack}`)')
    mm = np.memmap(pack, dtype=np.uint8, mode='r')
    t = {e['name']: e for e in m['tensors']}
    arr = lambda e: mm[e['offset']:e['offset'] + e['nbytes']].view(DTYPES[e['dtype']][1]).reshape(e['shape'])
    todo = []   # (row-major twin, record name) still missing; a pack from an older addgv gets only the new ones
    for e in m['tensors']:
        layer, key = e['name'].split('.', 1)
        base = key[:-9] if key.endswith('.mx.codes') else None
        if base in MX or (base in MX_NO_INT4 and f'{layer}.{MX_NO_INT4[base]}' not in t):
            todo.append((e, e['name'][:-6] + '.gv'))
        elif key in BF and not (m.get('slim_qo') and key.startswith('kv_b.')):   # slim pack (tools/convert/slim.py): the kv_b .gv stay out
            todo.append((e, e['name'] + '.gv'))
    todo = [(e, g) for e, g in todo if g not in t]
    if not todo:
        return dict(pack=pack, rank=m['rank'], added=0, note='nothing to add')
    w = PackWriter(pack, m['rank'], True, dict({k: v for k, v in m.items() if k not in TOP}, gv_records=True), append=m)
    for e, gv in todo:
        info = {k: v for k, v in e.items() if k not in STD}
        if gv.endswith('.mx.gv'):
            c = arr(e)
            w.put(gv, g6ref.mxfp8_record(c, arr(t[gv[:-3] + '.scales'])), 'u8',
                  layout='mxfp8_gv_record[O/16][K/32][528]', rows=int(c.shape[0]), cols=int(c.shape[1]), **info)
        else:
            a = arr(e).reshape(-1, e['shape'][-1])
            w.put(gv, g6ref.bf16_record(a), 'u8', layout='bf16_gv_record[O/16][K/16][512]',
                  rows=int(a.shape[0]), cols=int(a.shape[1]), **info)
    n = len(todo)
    out = w.close()
    return dict(pack=pack, rank=m['rank'], added=n, bytes_before=size, bytes=out['pack_bytes'], seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(addgv(p)), flush=True)
