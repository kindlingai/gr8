"""Writes tests/data/exlr8-fixture.safetensors: small synthetic matrices in every exlr8 dense format and on both
sharded sides, with the reference loader's fp32 decode of each (the HF weight (out, in)) as a 64-bit FNV-1a hash of its little-endian
bytes (`<name>.want_fnv`), and in full (`<name>.want`) where it is small.
tests/exlr8_fixture.rs checks gr8's decoder against it, so CI needs neither torch nor the release.

    python3 make_exlr8_fixture.py EXLR8_REPO OUT.safetensors

EXLR8_REPO is a kindlingai/exlr8 checkout (loader/exlr8.py). Needs torch. The output is deterministic (seed 0).
"""
import sys

import torch

sys.path.insert(0, sys.argv[1])
from loader import exlr8 as X  # noqa: E402

g = torch.Generator().manual_seed(0)


def u8(*shape):
    return torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)


def fp8(*shape):
    b = u8(*shape)
    return torch.where((b & 0x7f) == 0x7f, b ^ 1, b)       # no NaN codes


def f16(n, scale):
    return (torch.randn(n, generator=g) * scale).to(torch.float16)


def rec(fmt, out, inp, k, side, unit):
    return torch.tensor([fmt, out, inp, k, side, unit, 0, 1], dtype=torch.int64)


T = {}


def trellis(name, out, inp, k, side):
    n_ent, per = (inp // 128, out * 128 // 256) if side == X.IN else (out // 128, inp * 128 // 256)
    T[name + '.record'] = rec(X.TRELLIS, out, inp, k, side, 128)
    T[name + '.trellis'] = u8(n_ent, per * 32 * k)
    T[name + '.suh'] = f16(inp, 0.3)
    T[name + '.svh'] = f16(out, 0.05)


trellis('t_out_k4', 256, 256, 4, 1)      # entries on the output side, two Hadamard blocks on the input side
trellis('t_in_k4', 256, 256, 4, 2)       # entries on the input side, two blocks on the output side
trellis('t_out_k3', 128, 128, 3, 1)
trellis('t_in_k2', 128, 128, 2, 2)
T['i_out.record'] = rec(X.INT4, 64, 256, 0, 1, 32)
T['i_out.q4'] = u8(64, 128)
T['i_out.s4'] = f16(64 * 8, 0.01).reshape(64, 8)
T['i_in.record'] = rec(X.INT4, 64, 256, 0, 2, 128)
T['i_in.q4'] = u8(2, 64, 64)
T['i_in.s4'] = f16(2 * 64 * 4, 0.01).reshape(2, 64, 4)
T['mx.record'] = rec(X.MXFP8, 48, 128, 0, 0, 0)
T['mx.fp8'] = fp8(48, 128)
T['mx.e8m0'] = torch.randint(100, 135, (48, 4), generator=g, dtype=torch.int32).to(torch.uint8)
T['fb.record'] = rec(X.FP8_BLOCK, 200, 160, 0, 1, 100)
T['fb.fp8'] = fp8(200, 160)
T['fb.scale'] = torch.rand(2, 2, generator=g) * 1e-3
T['fr.record'] = rec(X.FP8_ROW, 40, 64, 0, 1, 8)
T['fr.fp8'] = fp8(40, 64)
T['fr.scale'] = torch.rand(40, generator=g) * 1e-3
T['bf.record'] = rec(X.BF16, 32, 64, 0, 0, 0)
T['bf.bf16'] = torch.randn(32, 64, generator=g).to(torch.bfloat16)
T['f32.record'] = rec(X.FP32, 16, 32, 0, 0, 0)
T['f32.fp32'] = torch.randn(16, 32, generator=g)


def fnv(t):
    h = 0xcbf29ce484222325
    for b in t.contiguous().view(torch.uint8).reshape(-1).tolist():
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


for m in [k[:-len('.record')] for k in list(T) if k.endswith('.record')]:
    r = T[m + '.record']
    want = X.decode_dense(r, {s: T[f'{m}.{s}'] for s in X.DATA[int(r[0])]}).contiguous()
    h = fnv(want)
    T[m + '.want_fnv'] = torch.tensor([h - (1 << 64) if h >> 63 else h], dtype=torch.int64)    # the u64 bits
    if want.numel() <= 8192:
        T[m + '.want'] = want
X.save_safetensors(T, sys.argv[2])
print(f'wrote {len(T)} tensors to {sys.argv[2]}')
