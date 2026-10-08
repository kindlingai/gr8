"""Replace one layer of a rank pack in place (#65): the v2 boot tests need a different layer 40 in 90 GB packs, and the
nodes have no room for a second copy.

python3 tools/convert/pack_splice.py splice PACKDIR/rank<r> NEWDIR/rank<r> --layer 40 --backup DIR
python3 tools/convert/pack_splice.py restore PACKDIR/rank<r> --backup DIR

The converter writes a layer's tensors as one contiguous byte range. `splice` checks that, checks the new layer's
tensors (NEWDIR, a pack converted with --layers L only) fit in that range, saves the range and the manifest to
--backup, writes the new tensors at the range's start in the new pack's order (4096-aligned), zeros the rest of the
range, and rewrites the manifest: the layer's old entries go, the new ones (with their sha256) come in, and the new
pack's top-level `expert_formats` / `v2_layers` entries for the layer are merged. The pack's size does not change.
`restore` writes the saved range and manifest back. Both leave a marker file PACKDIR/rank<r>.state.
"""
import argparse
import json
import os
import sys

ALIGN = 4096
CHUNK = 64 << 20


def load(p):
    with open(p + '.manifest.json') as f:
        return json.load(f)


def save_manifest(p, m):
    t = p + '.manifest.json.tmp'
    with open(t, 'w') as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(t, p + '.manifest.json')


def layer_range(m, prefix):
    ents = [e for e in m['tensors'] if e['name'].startswith(prefix)]
    if not ents:
        sys.exit(f'no {prefix}* tensors')
    a = min(e['offset'] for e in ents)
    b = max(e['offset'] + e['nbytes'] for e in ents)
    inside = [e for e in m['tensors'] if e['offset'] < b and e['offset'] + e['nbytes'] > a and not e['name'].startswith(prefix)]
    if inside:
        sys.exit(f'{prefix} is not one contiguous range: {inside[0]["name"]} lies inside it')
    end = min([e['offset'] for e in m['tensors'] if e['offset'] >= b] + [m['pack_bytes']])
    return ents, a, end      # the range runs to the next tensor (padding included)


def copy_range(src_fd, src_off, dst_fd, dst_off, n):
    while n:
        k = min(n, CHUNK)
        buf = os.pread(src_fd, k, src_off)
        if len(buf) != k:
            sys.exit('short read')
        os.pwrite(dst_fd, buf, dst_off)
        src_off += k
        dst_off += k
        n -= k


def splice(a):
    old, new = load(a.pack), load(a.new)
    prefix = f'L{a.layer:03d}.'
    ents, lo, hi = layer_range(old, prefix)
    newents = sorted((e for e in new['tensors'] if e['name'].startswith(prefix)), key=lambda e: e['offset'])
    if old.get('nr') != new.get('nr') or old['rank'] != new['rank']:
        sys.exit(f'rank / nr differ: old {old["rank"]}/{old.get("nr")}, new {new["rank"]}/{new.get("nr")}')
    need, o = 0, 0
    place = []
    for e in newents:
        o = (o + ALIGN - 1) // ALIGN * ALIGN
        place.append((e, o))
        o += e['nbytes']
    need = o
    if need > hi - lo:
        sys.exit(f'new layer needs {need} bytes, the old range has {hi - lo}')
    os.makedirs(a.backup, exist_ok=True)
    bk = os.path.join(a.backup, os.path.basename(a.pack))
    if os.path.exists(bk + '.range'):
        sys.exit(f'{bk}.range exists: restore first, or pick another --backup')
    fd = os.open(a.pack + '.pack', os.O_RDWR)
    with open(bk + '.range', 'wb') as f:
        copy_range(fd, lo, f.fileno(), 0, hi - lo)
    with open(bk + '.manifest.json', 'w') as f:
        json.dump(old, f, indent=1, sort_keys=True)
    json.dump({'lo': lo, 'hi': hi, 'layer': a.layer}, open(bk + '.where.json', 'w'))
    nfd = os.open(a.new + '.pack', os.O_RDONLY)
    zero = bytes(CHUNK)
    pos = lo
    out = []
    for e, rel in place:
        if lo + rel > pos:
            os.pwrite(fd, zero[:lo + rel - pos], pos)
        copy_range(nfd, e['offset'], fd, lo + rel, e['nbytes'])
        out.append(dict(e, offset=lo + rel))
        pos = lo + rel + e['nbytes']
    while pos < hi:
        k = min(CHUNK, hi - pos)
        os.pwrite(fd, zero[:k], pos)
        pos += k
    os.fsync(fd)
    os.close(fd)
    os.close(nfd)
    m = dict(old)
    m['tensors'] = sorted([e for e in old['tensors'] if not e['name'].startswith(prefix)] + out, key=lambda e: e['offset'])
    for key in ('expert_formats', 'v2_layers'):
        if str(a.layer) in new.get(key, {}):
            m.setdefault(key, {})[str(a.layer)] = new[key][str(a.layer)]
    m['spliced'] = {str(a.layer): os.path.abspath(a.new)}
    save_manifest(a.pack, m)
    with open(a.pack + '.state', 'w') as f:
        f.write(f'layer {a.layer} spliced from {os.path.abspath(a.new)}; restore: python3 tools/convert/pack_splice.py restore '
                f'{os.path.abspath(a.pack)} --backup {os.path.abspath(a.backup)}\n')
    print(f'{a.pack}: layer {a.layer} spliced, {need} of {hi - lo} bytes; backup in {a.backup}')


def restore(a):
    bk = os.path.join(a.backup, os.path.basename(a.pack))
    w = json.load(open(bk + '.where.json'))
    fd = os.open(a.pack + '.pack', os.O_RDWR)
    with open(bk + '.range', 'rb') as f:
        copy_range(f.fileno(), 0, fd, w['lo'], w['hi'] - w['lo'])
    os.fsync(fd)
    os.close(fd)
    m = json.load(open(bk + '.manifest.json'))
    save_manifest(a.pack, m)
    for suffix in ('.range', '.manifest.json', '.where.json'):
        os.remove(bk + suffix)
    with open(a.pack + '.state', 'w') as f:
        f.write('as built (layer %d restored)\n' % w['layer'])
    print(f'{a.pack}: layer {w["layer"]} restored')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('splice')
    s.add_argument('pack')
    s.add_argument('new')
    s.add_argument('--layer', type=int, required=True)
    s.add_argument('--backup', required=True)
    r = sub.add_parser('restore')
    r.add_argument('pack')
    r.add_argument('--backup', required=True)
    a = ap.parse_args()
    splice(a) if a.cmd == 'splice' else restore(a)


if __name__ == '__main__':
    main()
