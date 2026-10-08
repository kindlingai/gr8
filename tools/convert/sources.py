"""rank<r>.sources (pack.py PackWriter refs, convert.py --v2-refs): read the list and rebuild pack bytes on the host.

The engine does the same fill in C++ (engine/loader/g6pack.cpp, class Sources) while it streams a sparse pack into
the device image. This module is the reference for it, and its check command proves a sparse pack + sources give
exactly the bytes of a full pack, without a GPU:

python3 -m tools.convert.sources DIR --rank R [--root ROOT] [--ref FULL.pack]
    streams DIR/rank<R>.pack, fills every source range, and checks each tensor's sha256 against the manifest and
    every padding byte for zero. --ref also compares every byte with a full pack (the converter's output without
    --v2-refs); the first difference is reported with its tensor.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from tools.convert.pack import SOURCES_FORMAT
from tools.convert.v2 import down_to_g6

CHUNK = 256 << 20


def read_sources(path, root=None):
    """(root, [(pack_off, nbytes, op, file_off, file_nbytes, abs_path)]), sorted and checked for overlap."""
    with open(path) as f:
        head = f.readline().split(None, 1)
        if not head or head[0] != SOURCES_FORMAT:
            raise ValueError(f'{path}: not a {SOURCES_FORMAT} file')
        root = root or head[1].strip()
        segs = []
        for line in f:
            po, n, op, fo, fn, rel = line.split()
            segs.append((int(po), int(n), op, int(fo), int(fn), os.path.join(root, rel)))
    end = 0
    for s in segs:
        if s[0] < end:
            raise ValueError(f'{path}: segment at {s[0]} overlaps the previous one')
        if s[2] == 'item128' and (s[1] != s[4] or s[1] % (3072 * 32)):
            raise ValueError(f'{path}: item128 segment at {s[0]} is not one whole down half')
        if s[2] == 'copy' and s[1] != s[4]:
            raise ValueError(f'{path}: copy segment at {s[0]} has {s[1]} pack bytes for {s[4]} file bytes')
        end = s[0] + s[1]
    return root, segs


class Filler:
    """Fills source ranges into consecutive pack chunks (the engine's Sources::fill)."""

    def __init__(self, segs):
        self.segs, self.cur, self.fds, self.half = segs, 0, {}, (None, None)

    def _fd(self, p):
        if p not in self.fds:
            self.fds[p] = os.open(p, os.O_RDONLY)
        return self.fds[p]

    def _pread(self, p, off, n):
        b = os.pread(self._fd(p), n, off)
        if len(b) != n:
            raise IOError(f'short read {p} at {off}: {len(b)} of {n}')
        return b

    def fill(self, buf, pos):
        """buf: writable uint8 array holding pack bytes pos..pos+len(buf)-1."""
        end = pos + len(buf)
        while self.cur < len(self.segs) and self.segs[self.cur][0] + self.segs[self.cur][1] <= pos:
            self.cur += 1
        i = self.cur
        while i < len(self.segs) and self.segs[i][0] < end:
            po, n, op, fo, fn, p = self.segs[i]
            a, b = max(pos, po), min(end, po + n)
            if op == 'copy':
                buf[a - pos:b - pos] = np.frombuffer(self._pread(p, fo + a - po, b - a), np.uint8)
            else:
                if self.half[0] != i:
                    raw = np.frombuffer(self._pread(p, fo, fn), np.uint8)
                    self.half = (i, down_to_g6(raw, fn // (3072 * 32)).reshape(-1))
                buf[a - pos:b - pos] = self.half[1][a - po:b - po]
            i += 1

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def check(d, rank, root=None, ref=None, log=print):
    """Rebuild rank<rank>.pack of d from its sources; returns a summary dict. Raises on any mismatch."""
    with open(os.path.join(d, f'rank{rank}.manifest.json')) as f:
        man = json.load(f)
    _, segs = read_sources(os.path.join(d, f'rank{rank}.sources'), root)
    total = man['pack_bytes']
    ents = sorted(man['tensors'], key=lambda t: t['offset'])
    fill = Filler(segs)
    fd = os.open(os.path.join(d, f'rank{rank}.pack'), os.O_RDONLY)
    rfd = os.open(ref, os.O_RDONLY) if ref else None
    if os.fstat(fd).st_size != total:
        raise ValueError(f'rank{rank}.pack is {os.fstat(fd).st_size} bytes, manifest says {total}')
    ti, h, pos, bad = 0, hashlib.sha256(), 0, []
    try:
        while pos < total:
            n = min(CHUNK, total - pos)
            buf = np.frombuffer(bytearray(os.pread(fd, n, pos)), np.uint8)
            fill.fill(buf, pos)
            if rfd is not None:
                want = np.frombuffer(os.pread(rfd, n, pos), np.uint8)
                if len(want) != n or not np.array_equal(buf, want):
                    k = int(np.argmax(buf != want)) if len(want) == n else len(want)
                    at = pos + k
                    t = next((e['name'] for e in ents if e['offset'] <= at < e['offset'] + e['nbytes']), 'padding')
                    raise ValueError(f'byte {at} ({t}) differs from {ref}')
            p, q = pos, pos + n                 # walk the chunk: tensor bytes feed sha256, the rest must be zero
            while p < q:
                if ti < len(ents) and p >= ents[ti]['offset']:
                    e = ents[ti]
                    take = min(q, e['offset'] + e['nbytes']) - p
                    h.update(buf[p - pos:p - pos + take])
                    p += take
                    if p == e['offset'] + e['nbytes']:
                        if h.hexdigest() != e['sha256']:
                            bad.append(e['name'])
                        h, ti = hashlib.sha256(), ti + 1
                else:
                    take = (min(q, ents[ti]['offset']) if ti < len(ents) else q) - p
                    if buf[p - pos:p - pos + take].any():
                        raise ValueError(f'non-zero padding at bytes {p}..{p + take - 1}')
                    p += take
            pos += n
    finally:
        os.close(fd)
        if rfd is not None:
            os.close(rfd)
        fill.close()
    while ti < len(ents) and ents[ti]['nbytes'] == 0:
        ti += 1
    if bad or ti != len(ents):
        raise ValueError(f'sha256 mismatch in {len(bad)} tensor(s): {bad[:8]}' if bad else 'tensor walk incomplete')
    filled = sum(s[1] for s in segs)
    log(f'rank {rank}: {total} bytes, {len(ents)} tensors sha256 OK, {len(segs)} source ranges ({filled} bytes) filled'
        + (f', identical to {ref}' if ref else ''))
    return dict(rank=rank, pack_bytes=total, tensors=len(ents), segments=len(segs), filled=filled, ref=ref)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--root', help='source root (default: the one rank<r>.sources names)')
    ap.add_argument('--ref', help='a full pack to compare every byte with')
    a = ap.parse_args(argv)
    check(a.dir, a.rank, a.root, a.ref)


if __name__ == '__main__':
    main()
