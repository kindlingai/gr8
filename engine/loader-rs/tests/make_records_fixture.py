"""Writes tests/data/records-fixture.safetensors: small random inputs and what the Python converter makes of them
(engine/ref/g6ref.py's decode records, tools/convert/mxfp8.py's quantizer), for tests/records.rs. NumPy only.

    python3 engine/loader-rs/tests/make_records_fixture.py OUT.safetensors        (from the repo root)
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine', 'ref')]
import g6ref  # noqa: E402
from tools.convert import mxfp8  # noqa: E402

rng = np.random.default_rng(0)
T = {}
codes = rng.integers(0, 256, (32, 64), dtype=np.uint8)
codes[codes & 0x7f == 0x7f] ^= 1
scales = rng.integers(110, 140, (32, 2), dtype=np.uint8)
T['mx.codes'], T['mx.scales'] = codes, scales
T['mx.record'] = g6ref.mxfp8_record(codes, scales)
w = (rng.standard_normal((32, 48)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
T['bf.w'], T['bf.record'] = w, g6ref.bf16_record(w)
rt = g6ref.RT('int4')
q, s = rt.rand(rng, 32, 64)
T['rt.q'], T['rt.s'] = q, s
T['rt.record'] = g6ref.rt_record(rt, q, s)
gq, gs = rt.rand(rng, 32, 64)
uq, us = rt.rand(rng, 32, 64)
dq, ds = rt.rand(rng, 64, 32)
T.update({'ex.gq': gq, 'ex.gs': gs, 'ex.uq': uq, 'ex.us': us, 'ex.dq': dq, 'ex.ds': ds})
T['ex.fc1'], T['ex.fc2'] = g6ref.rt_expert_record(rt, gq, gs, uq, us, dq, ds)
x = (rng.standard_normal((16, 96)) * np.exp(rng.uniform(-12, 4, (16, 1)))).astype(np.float32)
x[3, 32:64] = 0                      # an all-zero group
x[5, :32] = 448 * 2.0 ** -3          # a power-of-two amax
xb = (x.view(np.uint32) >> 16).astype(np.uint16)       # truncate to BF16: mxfp8.quantize takes BF16 bits
T['q.w'] = xb
T['q.codes'], T['q.scales'] = mxfp8.quantize(xb)

DT = {np.uint8: 'U8', np.uint16: 'U16', np.float16: 'F16'}
head, off, blobs = {}, 0, []
for k, v in T.items():
    v = np.ascontiguousarray(v)
    b = v.tobytes()
    head[k] = {'dtype': DT[v.dtype.type], 'shape': list(v.shape), 'data_offsets': [off, off + len(b)]}
    off += len(b)
    blobs.append(b)
h = json.dumps(head, separators=(',', ':')).encode()
h += b' ' * (-len(h) % 8)
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<Q', len(h)) + h + b''.join(blobs))
print(f'wrote {len(T)} tensors to {sys.argv[1]}')
