"""g6 pack writer: one sequential file per rank + manifest JSON (name, dtype, shape, offset, nbytes, sha256).

Tensors start on 4096-byte boundaries (zero padding). Everything is deterministic: no timestamps, fixed order,
sorted JSON keys. Every data byte is covered by its tensor's sha256; padding bytes must be zero.

With refs=(src_dir, root), ref() accounts for bytes the engine reads from other files at load time (exlr8 v2 slices,
#65): they feed the tensor's sha256 like write(), but the pack keeps a hole there, and close() lists every such range
in rank<r>.sources (format: engine/loader/g6pack.h, class Sources). The manifest is the one write() would have given.
"""
import hashlib
import json
import os

import numpy as np

ALIGN = 4096
FORMAT = 'g6pack-v1'
_ZERO = bytes(ALIGN)
DROP_EVERY = 256 << 20
# manifest dtype labels -> (itemsize, numpy storage dtype); bf16 is stored as its uint16 bit pattern
DTYPES = {'u8': (1, np.uint8), 'i16': (2, np.int16), 'f16': (2, np.float16), 'bf16': (2, np.uint16),
          'f32': (4, np.float32), 'i32': (4, np.int32), 'i64': (8, np.int64)}


SOURCES_FORMAT = 'g6sources-v1'
REF_OPS = ('copy', 'item128')


class PackWriter:
    def __init__(self, path, rank, drop_cache=False, meta=None, append=None, refs=None):
        """append = the pack's current manifest dict: keep the file and its entries, continue at pack_bytes.
        refs = (src_dir, root): enable ref(); source paths are stored relative to src_dir, and root is the directory
        the engine resolves them against (the same files as the engine's host sees them)."""
        self.path, self.rank, self.drop = str(path), rank, drop_cache
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | (0 if append else os.O_TRUNC), 0o644)
        self.off, self.cur, self.entries = 0, None, []
        if append:
            if refs:
                raise ValueError('refs cannot extend an existing pack')
            self.off = os.lseek(self.fd, append['pack_bytes'], os.SEEK_SET)
            self.entries = list(append['tensors'])
        self.meta = dict(meta or {})
        self.since_drop = 0
        self.refs, self.segments = refs, []

    def _raw(self, b):
        mv = memoryview(b).cast('B')
        n = 0
        while n < len(mv):
            n += os.write(self.fd, mv[n:])
        self.off += len(mv)
        self.since_drop += len(mv)
        if self.drop and self.since_drop >= DROP_EVERY:
            os.fdatasync(self.fd)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
            self.since_drop = 0

    def begin(self, name, dtype, shape, **info):
        if self.cur is not None:
            raise RuntimeError(f'begin {name} while {self.cur["name"]} open')
        pad = (-self.off) % ALIGN
        if pad:
            self._raw(_ZERO[:pad])
        nbytes = int(np.prod(shape, dtype=np.int64)) * DTYPES[dtype][0]
        self.cur = dict(name=name, dtype=dtype, shape=[int(s) for s in shape], offset=self.off,
                        nbytes=nbytes, **info)
        self._h, self._n = hashlib.sha256(), 0

    def write(self, arr):
        a = np.ascontiguousarray(arr)
        self._h.update(memoryview(a).cast('B'))
        self._n += a.nbytes
        self._raw(a)

    def ref(self, arr, path, file_off, file_nbytes, op='copy'):
        """Like write(arr), but the bytes stay in `path`: the pack gets a hole, rank<r>.sources the range.
        op copy: arr is file bytes file_off.. as they are; op item128: arr is one v2 down half after down_to_g6."""
        if not self.refs:
            raise RuntimeError('ref() needs PackWriter(refs=...)')
        if op not in REF_OPS:
            raise ValueError(f'ref op {op!r}')
        a = np.ascontiguousarray(arr)
        rel = os.path.relpath(path, self.refs[0])
        if any(c.isspace() for c in rel) or rel.startswith('..'):
            raise ValueError(f'ref path {path!r} is not a plain path under {self.refs[0]}')
        if a.nbytes != file_nbytes:
            raise ValueError(f'ref {rel}: {a.nbytes} bytes for a {file_nbytes}-byte source range')
        self._h.update(memoryview(a).cast('B'))
        self._n += a.nbytes
        self._hole(rel, file_off, a.nbytes, op)

    def ref_entry(self, e, path, file_off):
        """Another pack's tensor, unchanged: its manifest entry e (sha256 included) and its bytes at path:file_off.
        Nothing is read; the engine copies the range at load."""
        if not self.refs:
            raise RuntimeError('ref_entry() needs PackWriter(refs=...)')
        rel = os.path.relpath(path, self.refs[0])
        if any(c.isspace() for c in rel) or rel.startswith('..'):
            raise ValueError(f'ref path {path!r} is not a plain path under {self.refs[0]}')
        info = {k: v for k, v in e.items() if k not in ('name', 'dtype', 'shape', 'offset', 'nbytes', 'sha256')}
        self.begin(e['name'], e['dtype'], e['shape'], **info)
        if self.cur['nbytes'] != e['nbytes']:
            raise ValueError(f'{e["name"]}: nbytes {e["nbytes"]} does not match its dtype and shape')
        if e['nbytes']:
            self._hole(rel, file_off, e['nbytes'], 'copy')
        self._n = e['nbytes']
        self.cur['sha256'] = e['sha256']
        self.entries.append(self.cur)
        self.cur = None

    def _hole(self, rel, file_off, n, op):
        last = self.segments[-1] if self.segments else None
        if (op == 'copy' and last and last[2] == 'copy' and last[5] == rel and last[0] + last[1] == self.off
                and last[3] + last[4] == file_off):
            self.segments[-1] = (last[0], last[1] + n, 'copy', last[3], last[4] + n, rel)
        else:
            self.segments.append((self.off, n, op, int(file_off), n, rel))
        self.off = os.lseek(self.fd, n, os.SEEK_CUR)

    def end(self):
        c = self.cur
        if self._n != c['nbytes']:
            raise RuntimeError(f'{c["name"]}: wrote {self._n} bytes, expected {c["nbytes"]}')
        c['sha256'] = self._h.hexdigest()
        self.entries.append(c)
        self.cur = None

    def put(self, name, arr, dtype, **info):
        a = np.asarray(arr)
        if a.dtype != DTYPES[dtype][1]:
            raise ValueError(f'{name}: array dtype {a.dtype} does not store label {dtype}')
        self.begin(name, dtype, a.shape, **info)
        self.write(a)
        self.end()

    def close(self):
        pad = (-self.off) % ALIGN
        if pad:
            self._raw(_ZERO[:pad])
        os.ftruncate(self.fd, self.off)        # a hole at the end of the pack still counts toward its size
        os.fsync(self.fd)
        if self.drop:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(self.fd)
        stem = self.path.rsplit('.pack', 1)[0]
        if self.segments:
            sp = stem + '.sources'
            with open(sp + '.tmp', 'w') as f:
                f.write(f'{SOURCES_FORMAT} {self.refs[1]}\n')
                for s in self.segments:
                    f.write(' '.join(str(x) for x in s) + '\n')
            os.replace(sp + '.tmp', sp)
            self.meta['sources'] = os.path.basename(sp)
        man = dict(format=FORMAT, rank=self.rank, align=ALIGN, pack_bytes=self.off,
                   tensors=self.entries, **self.meta)
        mp = stem + '.manifest.json'
        with open(mp + '.tmp', 'w') as f:
            json.dump(man, f, indent=1, sort_keys=True)
            f.write('\n')
        os.replace(mp + '.tmp', mp)
        return man
