#!/usr/bin/env python3
"""Where does the routed weight go: BEFORE w2 (official model.py Expert.forward: x = weights * (silu(gate) * up); w2(x.to(dtype))) or AFTER
w2 (engine: ypart = fp16(w * (down @ act)))?  Exact real arithmetic is linear, so the two are the same function; this script measures what the
roundings do to each, on the real checkpoint weights, real ffn_in and real routes of the oracle's p6-faithful dump.

  ds_moe_placement_ref.py FIX_DIR ORACLE_DUMP_DIR LAYER      FIX_DIR = what ds_moe_fixture_prep.py wrote

All matmuls are float64 on the exactly dequantised W4 weights and bf16 x (no activation quantisation: the same W4A16 contract as the engine).
Variants (each differs from E only by the rounding it emulates):
  E     exact float64, no rounding                              reference for everything below
  P16   engine: act = fp16(h); ypart = fp16(w * (act @ w2^T)), TP4 slices of 576 (each slice rounds its own ypart), summed
  Q16   same as P16 but weight BEFORE w2: act = fp16(w * h); ypart = fp16(act @ w2^T), TP4 slices
  S16   P16 without the TP4 slicing (one rounding per expert)
  R16   Q16 without the TP4 slicing
  R     official placement and dtype: act = bf16(w * h) (fp32 product), y = bf16(act @ w2^T), no slicing
  S     the same with the weight AFTER w2: act = bf16(h), y = bf16(w * (act @ w2^T))
h = clamp(gate, max=10) * sigmoid(.) * clamp(up, +-10) in float64 (not rounded). Reported: rel_L2 of the routed sum (and of routed + exact shared, rounded
to bf16 like ffn_out) against E, and the distance between the placements.
"""
import sys

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
D = torch.float64


def rd(path, dt, shape):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt).reshape(shape)


def e8(b):
    return torch.exp2(b.to(D) - 127.0)


def fp4(packed_i8, scale_u8):
    b = packed_i8.view(torch.uint8)
    lo, hi = b & 15, b >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(b.shape[0], -1)
    v = E2M1[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    return v * e8(scale_u8).repeat_interleave(32, dim=1)


def fp8(codes_u8, scale_rows_u8):
    return codes_u8.view(torch.float8_e4m3fn).to(D) * e8(scale_rows_u8).repeat_interleave(32, dim=1)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def r16(t):
    return t.to(torch.float16).to(D)


def rb16(t):
    return t.to(torch.bfloat16).to(D)


def swiglu(g, u, lim=10.0):
    g = g.clamp(max=lim)
    u = u.clamp(-lim, lim)
    return g * torch.sigmoid(g) * u


def main():
    fix, odir, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
    x = rd(f"{fix}/x.bf16", torch.bfloat16, (-1, 5120)).to(D)
    S = x.shape[0]
    ids = [int(t) for t in open(f"{fix}/experts.txt").read().split()]
    idx = rd(f"{odir}/L{L}.router_idx.bin", torch.int32, (S, 6))
    w = rd(f"{odir}/L{L}.router_w.bin", torch.float32, (S, 6)).to(D)
    IW, NS = 576, 4

    sw = {}
    for n, (r, k) in (("w1", (2304, 5120)), ("w3", (2304, 5120)), ("w2", (5120, 2304))):
        sw[n] = fp8(rd(f"{fix}/sh_{n}.c", torch.uint8, (r, k)), rd(f"{fix}/sh_{n}.s", torch.uint8, (r, k // 32)))
    shared = swiglu(x @ sw["w1"].T, x @ sw["w3"].T) @ sw["w2"].T
    del sw

    names = ["E", "P16", "Q16", "S16", "R16", "R", "S"]
    acc = {n: torch.zeros(S, 5120, dtype=D) for n in names}
    EXB = 3 * (5898240 + 368640)
    blob = open(f"{fix}/experts.bin", "rb")
    for j, e in enumerate(ids):
        rows, slot = torch.where(idx == e)
        if len(rows) == 0:
            continue
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
        xe = x[rows]
        wt = w[rows, slot, None]
        h = swiglu(xe @ m["w1"].T, xe @ m["w3"].T)
        w2 = m["w2"]
        acc["E"][rows] += wt * (h @ w2.T)

        def sliced(a):                                # sum over the 4 TP slices of the contraction a[:, slice] @ w2[:, slice]^T
            return [a[:, i * IW:(i + 1) * IW] @ w2[:, i * IW:(i + 1) * IW].T for i in range(NS)]

        a16 = r16(h)
        acc["P16"][rows] += sum(r16(wt * y) for y in sliced(a16))
        aw16 = r16(wt * h)
        acc["Q16"][rows] += sum(r16(y) for y in sliced(aw16))
        acc["S16"][rows] += r16(wt * (a16 @ w2.T))
        acc["R16"][rows] += r16(aw16 @ w2.T)
        acc["R"][rows] += rb16(rb16(wt.float().to(D) * h) @ w2.T)
        acc["S"][rows] += rb16(wt * (rb16(h) @ w2.T))
        del m, w2
    E = acc["E"]
    print(f"L{L}: S={S} experts {len(ids)}  |routed|_rms {E.pow(2).mean().sqrt().item():.3e}  (rel_L2 vs E, routed sum)")
    for n in names[1:]:
        print(f"   {n:4s} {rel(acc[n], E):.3e}")
    print(f"   placement distance  P16-vs-Q16 {rel(acc['P16'], acc['Q16']):.3e}   S16-vs-R16 {rel(acc['S16'], acc['R16']):.3e}   S-vs-R {rel(acc['S'], acc['R']):.3e}"
          f"   P16-vs-R {rel(acc['P16'], acc['R']):.3e}   Q16-vs-R {rel(acc['Q16'], acc['R']):.3e}")
    fe = rb16(E + shared)
    print("   ffn_out = bf16(routed + exact shared), rel_L2 vs bf16(E+shared): " +
          "  ".join(f"{n} {rel(rb16(acc[n] + shared), fe):.3e}" for n in names[1:]) +
          f"   (bf16 rounding alone vs unrounded {rel(fe, E + shared):.1e})")


if __name__ == "__main__":
    main()
