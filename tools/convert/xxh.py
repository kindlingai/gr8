"""xxhash64 (seed 0 by default) for the exlr8 v2 half checksums (#65).

Uses the system libxxhash (Ubuntu's libxxhash0, which rsync depends on, so it is on DGX OS and in the CUDA images),
else the `xxhash` Python package. Raises if neither is present.
"""
import ctypes
import ctypes.util

import numpy as np

_lib = None
_mod = None
for name in (ctypes.util.find_library('xxhash'), 'libxxhash.so.0', 'libxxhash.so', 'libxxhash.dylib'):
    if not name:
        continue
    try:
        _lib = ctypes.CDLL(name)
        _lib.XXH64.restype = ctypes.c_uint64
        _lib.XXH64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
        _lib.XXH64_createState.restype = ctypes.c_void_p
        _lib.XXH64_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        _lib.XXH64_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.XXH64_digest.restype = ctypes.c_uint64
        _lib.XXH64_digest.argtypes = [ctypes.c_void_p]
        _lib.XXH64_freeState.argtypes = [ctypes.c_void_p]
        break
    except (OSError, AttributeError):
        _lib = None
if _lib is None:
    try:
        import xxhash as _mod
    except ImportError:
        _mod = None


def _buf(a):
    a = np.ascontiguousarray(a)
    return a, a.ctypes.data, a.nbytes


def xxh64(*parts, seed=0):
    """xxhash64 of the parts' bytes as one stream (numpy arrays or bytes, in order)."""
    if _lib is not None:
        if len(parts) == 1:
            a, p, n = _buf(np.frombuffer(parts[0], np.uint8) if isinstance(parts[0], (bytes, bytearray)) else parts[0])
            return int(_lib.XXH64(p, n, seed))
        st = _lib.XXH64_createState()
        try:
            _lib.XXH64_reset(st, seed)
            for x in parts:
                a, p, n = _buf(np.frombuffer(x, np.uint8) if isinstance(x, (bytes, bytearray)) else x)
                _lib.XXH64_update(st, p, n)
            return int(_lib.XXH64_digest(st))
        finally:
            _lib.XXH64_freeState(st)
    if _mod is not None:
        h = _mod.xxh64(seed=seed)
        for x in parts:
            h.update(x if isinstance(x, (bytes, bytearray)) else np.ascontiguousarray(x).tobytes())
        return h.intdigest()
    raise RuntimeError('xxhash64: neither libxxhash nor the xxhash package is available')
