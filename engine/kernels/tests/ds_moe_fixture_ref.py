#!/usr/bin/env python3
"""Double-precision reference for `ds_shared_test fixture`: same checkpoint weights, same ffn_in, same routes as the engine run,
no activation quantisation and no intermediate rounding (W4A16 / W8A16 done exactly in float64). The engine dumps should sit near this
(fp16 activations and fp16 route-weighted partials are the only roundings); the oracle's own arms differ by their act-quant policy.

  ds_moe_fixture_ref.py FIX_DIR ENGINE_DIR LAYER      FIX_DIR = the dir ds_moe_fixture_prep.py wrote, ENGINE_DIR = fixture output
"""
import os
import sys

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def rd(path, dt, shape):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt).reshape(shape)


def e8(b):
    return torch.exp2(b.to(torch.float64) - 127.0)


def fp4(packed_i8, scale_u8):                       # packed [R, K/2] bytes (low nibble = even k), scale [R, K/32] -> [R, K] float64
    b = packed_i8.view(torch.uint8)
    lo, hi = b & 15, b >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(b.shape[0], -1)
    v = E2M1[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    return v * e8(scale_u8).repeat_interleave(32, dim=1)


def fp8(codes_u8, scale_rows_u8):                   # codes [R, K] E4M3, scale expanded to [R, K/32]
    v = codes_u8.view(torch.float8_e4m3fn).to(torch.float64)
    return v * e8(scale_rows_u8).repeat_interleave(32, dim=1)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    fix, eng, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
    x = rd(f"{fix}/x.bf16", torch.bfloat16, (-1, 5120)).to(torch.float64)
    S = x.shape[0]
    ids = [int(t) for t in open(f"{fix}/experts.txt").read().split()]
    idx = rd(f"{eng}/L{L}.router_idx.bin", torch.int32, (S, 6))
    w = rd(f"{eng}/L{L}.router_w.bin", torch.float32, (S, 6)).to(torch.float64)
    lim = 10.0

    def swiglu(g, u):
        g = g.clamp(max=lim)
        u = u.clamp(-lim, lim)
        return g * torch.sigmoid(g) * u

    # shared
    sw = {}
    for n, (r, k) in (("w1", (2304, 5120)), ("w3", (2304, 5120)), ("w2", (5120, 2304))):
        sw[n] = fp8(rd(f"{fix}/sh_{n}.c", torch.uint8, (r, k)), rd(f"{fix}/sh_{n}.s", torch.uint8, (r, k // 32)))
    shared = swiglu(x @ sw["w1"].T, x @ sw["w3"].T) @ sw["w2"].T
    del sw
    # routed, expert by expert
    routed = torch.zeros(S, 5120, dtype=torch.float64)
    EXB = 3 * (5898240 + 368640)
    blob = open(f"{fix}/experts.bin", "rb")
    for j, e in enumerate(ids):
        blob.seek(j * EXB)
        raw = torch.frombuffer(bytearray(blob.read(EXB)), dtype=torch.uint8)
        o = 0
        m = {}
        for n, (r, k) in (("w1", (2304, 2560)), ("w3", (2304, 2560)), ("w2", (5120, 1152))):
            p = raw[o:o + r * k].reshape(r, k)
            o += r * k
            s = raw[o:o + r * (k // 16)].reshape(r, k // 16)
            o += r * (k // 16)
            m[n] = fp4(p, s)
        rows, slot = torch.where(idx == e)
        if len(rows) == 0:
            continue
        xe = x[rows]
        y = swiglu(xe @ m["w1"].T, xe @ m["w3"].T) @ m["w2"].T
        routed[rows] += w[rows, slot, None] * y
    ffn = (routed + shared)
    mine_r = rd(f"{eng}/L{L}.moe_routed.bin", torch.float32, (S, 5120)).to(torch.float64)
    mine_s = rd(f"{eng}/L{L}.shared_out.bin", torch.float32, (S, 5120)).to(torch.float64)
    mine_f = rd(f"{eng}/L{L}.ffn_out.bin", torch.bfloat16, (S, 5120)).to(torch.float64)
    print(f"L{L}: vs exact float64 W4A16/W8A16 reference: moe_routed rel_L2 {rel(mine_r, routed):.3e}   shared_out {rel(mine_s, shared):.3e} "
          f"(bf16 output rounding alone ~{rel(shared.to(torch.bfloat16).double(), shared):.1e})   ffn_out {rel(mine_f, ffn):.3e} "
          f"(bf16 rounding alone ~{rel(ffn.to(torch.bfloat16).double(), ffn):.1e})")


if __name__ == "__main__":
    main()
