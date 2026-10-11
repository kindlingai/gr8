"""gr8 pack writer: one sequential file per rank + manifest JSON (name, dtype, shape, offset, nbytes, sha256).

Tensors start on 4096-byte boundaries (zero padding). Everything is deterministic: no timestamps, fixed order,
sorted JSON keys. Every data byte is covered by its tensor's sha256; padding bytes must be zero.
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



class PackWriter:
    def __init__(self, path, rank, drop_cache=False, meta=None, append=None):
        """append = the pack's current manifest dict: keep the file and its entries, continue at pack_bytes."""
        self.path, self.rank, self.drop = str(path), rank, drop_cache
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | (0 if append else os.O_TRUNC), 0o644)
        self.off, self.cur, self.entries = 0, None, []
        if append:
            self.off = os.lseek(self.fd, append['pack_bytes'], os.SEEK_SET)
            self.entries = list(append['tensors'])
        self.meta = dict(meta or {})
        self.since_drop = 0

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
        os.ftruncate(self.fd, self.off)
        os.fsync(self.fd)
        if self.drop:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(self.fd)
        stem = self.path.rsplit('.pack', 1)[0]
        man = dict(format=FORMAT, rank=self.rank, align=ALIGN, pack_bytes=self.off,
                   tensors=self.entries, **self.meta)
        mp = stem + '.manifest.json'
        with open(mp + '.tmp', 'w') as f:
            json.dump(man, f, indent=1, sort_keys=True)
            f.write('\n')
        os.replace(mp + '.tmp', mp)
        return man
