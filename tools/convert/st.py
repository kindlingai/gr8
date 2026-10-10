"""Minimal safetensors reader: header parse + pread of one tensor (no mmap, so RSS stays at what we hold)."""
import json
import os
import struct

import numpy as np

DT = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32, 'I16': np.int16, 'I32': np.int32,
      'U8': np.uint8, 'I8': np.int8, 'U16': np.uint16, 'I64': np.int64, 'U64': np.uint64}


class ST:
    def __init__(self, path):
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        n, = struct.unpack('<Q', os.pread(self.fd, 8, 0))
        h = json.loads(os.pread(self.fd, n, 8))
        self.meta = h.pop('__metadata__', None) or {}
        self.h, self.base = h, 8 + n

    def __contains__(self, name):
        return name in self.h

    def dtype(self, name):
        return self.h[name]['dtype']

    def shape(self, name):
        return tuple(self.h[name]['shape'])

    def get(self, name, rows=None):
        """Tensor as a numpy array (BF16 as uint16 bits). rows=(a, b) reads only rows a..b-1 of dim 0."""
        t = self.h[name]
        lo, hi = t['data_offsets']
        dt = np.dtype(DT[t['dtype']])
        shape = list(t['shape'])
        if rows is not None:
            row = (hi - lo) // shape[0]
            lo, hi = lo + rows[0] * row, lo + rows[1] * row
            shape[0] = rows[1] - rows[0]
        buf = bytearray(hi - lo)
        mv, off = memoryview(buf), 0
        while off < len(buf):
            k = os.preadv(self.fd, [mv[off:]], self.base + lo + off)
            if k <= 0:
                raise IOError(f'short read {self.path}:{name}')
            off += k
        return np.frombuffer(buf, dtype=dt).reshape(shape)

    def read(self, name, lo, n):
        """Bytes lo..lo+n-1 of a tensor's data as uint8 (one half of a v2 trellis tensor)."""
        a, b = self.h[name]['data_offsets']
        if lo < 0 or lo + n > b - a:
            raise ValueError(f'{self.path}:{name}: range {lo}+{n} outside {b - a} bytes')
        buf = bytearray(n)
        mv, off = memoryview(buf), 0
        while off < n:
            k = os.preadv(self.fd, [mv[off:]], self.base + a + lo + off)
            if k <= 0:
                raise IOError(f'short read {self.path}:{name}')
            off += k
        return np.frombuffer(buf, np.uint8)

    def read_abs(self, off, n):
        """Bytes off..off+n-1 of the file as uint8 (absolute file offsets, as the v2 index gives them)."""
        buf = bytearray(n)
        mv, o = memoryview(buf), 0
        while o < n:
            k = os.preadv(self.fd, [mv[o:]], off + o)
            if k <= 0:
                raise IOError(f'short read {self.path} at {off + o}')
            o += k
        return np.frombuffer(buf, np.uint8)

    def drop_cache(self):
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def close(self):
        os.close(self.fd)
