"""Routed-expert error of an EXL3 checkpoint (TR3 or exlr8-e4m3) against the BF16 originals, one layer, CPU only.

python3 -m tools.convert.experr --bf16 DIR --ckpt NAME=DIR [--ckpt NAME=DIR ...] [--layer 40] [--experts 0,17,...]
        [--rows 64] [--seed 0]

Per sampled expert: y = down(silu(gate x) * up x) for `rows` random normal inputs x, with the BF16 weights (fp32 math)
and with each checkpoint's decoded weights, summed over its TP4 rank slices (each slice has its own sign/scale vectors).
Decoding follows engine/ref/g6ref.py (the trellis codebook, Hadamard-128 rotations, fp16 at the kernel's rounding
points). An exlr8-e4m3 layer is also run with the codebook rounded to E4M3 (`NAME+e4m3`), as the engine decodes it;
`NAME` alone is the unrounded decode. Prints the relative output error ||y_q - y|| / ||y|| over all sampled experts and
rows, and its mean per expert. --e4m3-all adds the rounded arm for TR3 files too (what g6's prefill does to TR3
weights). BF16 dir: a Hugging Face checkpoint with model.safetensors.index.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'engine', 'ref'))
import g6ref  # noqa: E402

from tools.convert.convert import EXPERT_FORMATS  # noqa: E402
from tools.convert.st import ST  # noqa: E402


def e4m3_round(h):
    """fp16 array -> the nearest FP8 E4M3 values (round to nearest even, saturating at +-448), as fp16."""
    v = h.astype(np.float64)
    e = np.maximum(np.floor(np.log2(np.maximum(np.abs(v), 2.0 ** -10))), -6.0)   # subnormals share the 2^-6 binade's step
    q = 2.0 ** (e - 3)
    return np.clip(np.round(v / q) * q, -448.0, 448.0).astype(np.float16)


def bf16_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


class Bf16:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, 'model.safetensors.index.json')))['weight_map']
        self.files = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = ST(os.path.join(self.d, f))
        return bf16_f32(self.files[f].get(name))


def linear(x, t, suh, svh, rnd):
    """EXL3 linear of one slice: had(f16(f16(had(x*suh)) @ W)) * svh, W [in, out] decoded from trellis [kt][nt][16K]."""
    w = g6ref.tr3_unpack(t)
    if rnd:
        w = e4m3_round(w)
    xr = g6ref.f16(g6ref.had128(x * suh.astype(np.float32))).astype(np.float32)
    z = g6ref.f16(xr @ w.astype(np.float32)).astype(np.float32)
    return g6ref.had128(z) * svh.astype(np.float32)


def expert_q(src, l, e, x, rnd):
    p = f'model.layers.{l}.mlp.experts.{e}.'
    y = np.zeros((x.shape[0], g6ref.H), np.float32)
    nr = sum(1 for k in src.h if k.startswith(p + 'gate_proj.rank') and k.endswith('.trellis'))
    for r in range(nr):
        g = lambda m, k: src.get(f'{p}{m}_proj.rank{r}.{k}')
        gate = g6ref.f16(linear(x, g('gate', 'trellis'), g('gate', 'suh'), g('gate', 'svh'), rnd)).astype(np.float32)
        up = g6ref.f16(linear(x, g('up', 'trellis'), g('up', 'suh'), g('up', 'svh'), rnd)).astype(np.float32)
        act = g6ref.f16(g6ref.silu(gate) * up).astype(np.float32)
        # down trellis [32][384][16K]: k = this slice's 512 intermediate channels, n = 6144 hidden
        y += linear(act, g('down', 'trellis'), g('down', 'suh'), g('down', 'svh'), rnd)
    return y


def expert_ref(bf, l, e, x):
    p = f'model.layers.{l}.mlp.experts.{e}.'
    gate = x @ bf.get(p + 'gate_proj.weight').T
    up = x @ bf.get(p + 'up_proj.weight').T
    return (g6ref.silu(gate) * up) @ bf.get(p + 'down_proj.weight').T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bf16', required=True)
    ap.add_argument('--ckpt', action='append', required=True, help='NAME=DIR of a TR3 or exlr8-e4m3 checkpoint')
    ap.add_argument('--layer', type=int, default=40)
    ap.add_argument('--experts', default='0,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255')
    ap.add_argument('--rows', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--e4m3-all', action='store_true', help='also decode TR3 files with E4M3 rounding')
    a = ap.parse_args(argv)
    l = a.layer
    experts = [int(e) for e in a.experts.split(',')]
    x = g6ref.f16(np.random.default_rng(a.seed).standard_normal((a.rows, g6ref.H), np.float32)).astype(np.float32)
    bf = Bf16(a.bf16)
    ref = {e: expert_ref(bf, l, e, x) for e in experts}
    for spec in a.ckpt:
        name, d = spec.split('=', 1)
        src = ST(os.path.join(d, f'model-layer-{l:03d}.safetensors'))
        fmt = EXPERT_FORMATS[src.meta.get('exlr8')]
        arms = [(name, False)] + ([(name + '+e4m3', True)] if fmt == 'exlr8-e4m3' or a.e4m3_all else [])
        for arm, rnd in arms:
            num = den = 0.0
            per = []
            bits = []
            for e in experts:
                y = expert_q(src, l, e, x, rnd)
                d2 = float(((y - ref[e]) ** 2).sum())
                r2 = float((ref[e] ** 2).sum())
                num += d2
                den += r2
                per.append((d2 / r2) ** 0.5)
                bits.append(src.shape(f'model.layers.{l}.mlp.experts.{e}.gate_proj.rank0.trellis')[-1] // 16)
            print(json.dumps(dict(arm=arm, format=fmt, layer=l, experts=len(experts), rows=a.rows,
                                  k3=bits.count(3), k4=bits.count(4), rel_err=round((num / den) ** 0.5, 5),
                                  mean_rel_err_per_expert=round(float(np.mean(per)), 5))), flush=True)
        src.close()


if __name__ == '__main__':
    main()
