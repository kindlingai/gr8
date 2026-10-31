"""CPU emulation of the decode GEMV's arithmetic (engine/kernels/densev.cuh gv_gemv_body<Mxfp8,1> behind attn k_xfrag_q + k_gv_mx),
run through the torch twin (attn_ref.py) and compared with the generic-GEMM twin and the oracle's dumps (ds41_oracle.py). No GPU.

  python3 gv_emul.py --layer 0 --shard .../model-00003-of-00048.safetensors --dumps .../out/p128-faithful --n 128
  python3 gv_emul.py --layer 2 --shard .../model-00005-of-00048.safetensors --dumps .../out/p6-faithful --n 6

GEMV arithmetic (one projection, x bf16 [n][K], weight fp8 [N][K] + ue8m0 block scale [N/32][K/32]):
  xd   = act_quant dequantized (the official numerics: e4m3 code * 2^e), clamped to +-65504 and ROUNDED TO FP16 (the mma B operand)
  part = for each 32-group g of K: two mma k16 steps, products fp16 x e4m3 exact, accumulated in fp32 (emulated: fp64 sum, one fp32 rounding
         per k16 step, the first step's result is the second step's C input)
  acc  = fmaf(row_scale[o][g], part, acc) in group order inside a k-part (power-of-two scale: the product is exact)
  ks   = k-parts (default: the largest divisor of K/32 that is <= 16); each part starts at 0; parts are summed in order in fp32
  out  = bf16 round-to-nearest-even
The generic GEMM (k_gemm_mxfp8) differs only in fp32 summation order, plus the fp16 step above (|xd| < 2^-24 flushes, > 65504 clamps).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ref as A  # noqa: E402
import oracle_cmp as OC  # noqa: E402

STATS = []   # one dict per projection call: K, N, n, inexact fraction of the fp16 cast, overflow / underflow counts


def pick_ks(K):
    for k in range(16, 1, -1):
        if (K // 32) % k == 0:
            return k
    return 1


def emul(x, w8, sc, ks=None, row_chunk=2048):
    """x [n, K] bf16, w8 [N, K] fp8, sc [N/32, K/32] -> bf16 [n, N] with the GEMV's arithmetic."""
    n, K = x.shape
    N = w8.shape[0]
    G = K // 32
    ks = ks or pick_ks(K)
    xd = A.act_quant_deq(x).float()                         # official activation values (exact in fp32)
    xh = xd.clamp(-65504.0, 65504.0).half().float()         # the fp16 B operand
    ad = xd.abs()
    STATS.append(dict(K=K, N=N, n=n, inexact=(xh != xd).float().mean().item(), over=int((ad > 65504.0).sum()),
                      under=int(((ad > 0) & (ad < 2.0 ** -14)).sum()), flushed=int(((xd != 0) & (xh == 0)).sum())))
    xg = xh.double().reshape(n, G, 2, 16)
    wf = w8.float().reshape(N, G, 2, 16).double()
    rs = sc.float().repeat_interleave(32, 0)                # [N, G] row scale (power of two)
    out = torch.empty(n, N, dtype=torch.bfloat16)
    npart = G // ks
    for r0 in range(0, N, row_chunk):
        r1 = min(r0 + row_chunk, N)
        w = wf[r0:r1]
        s1 = torch.einsum("ngk,ogk->nog", xg[:, :, 0], w[:, :, 0]).float()                 # k16 step 1 -> fp32
        s2 = (torch.einsum("ngk,ogk->nog", xg[:, :, 1], w[:, :, 1]).double() + s1.double()).float()   # step 2 with C = step 1
        prod = s2 * rs[r0:r1].unsqueeze(0)                                                 # exact (power-of-two scale)
        parts = []
        for p in range(ks):
            acc = torch.zeros(n, r1 - r0, dtype=torch.float32)
            for g in range(p * npart, (p + 1) * npart):
                acc = acc + prod[:, :, g]                                                  # fmaf with an exact product == one add
            parts.append(acc)
        tot = parts[0]
        for q in range(1, ks):
            tot = tot + parts[q]
        out[:, r0:r1] = tot.to(torch.bfloat16)
    return out


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a @ b) / (a.norm() * b.norm())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ks", type=int, default=0, help="k-parts (default: the GEMV's pick)")
    a = ap.parse_args()
    l, n, D = a.layer, a.n, a.dumps

    def f(s):
        return os.path.join(D, f"L{l}.{s}.bin")

    y = OC.rd(f("attn_in"), "bf16", (n, A.HID))
    w = A.load_layer_weights(a.shard, l)
    lay = A.Layer(l, w)
    dg = {}
    og = lay.forward(y, 0, dump=dg)
    A.GV_EMUL = lambda x, w8, sc: emul(x, w8, sc, a.ks or None)
    STATS.clear()
    dv = {}
    try:
        ov = lay.forward(y, 0, dump=dv)
    finally:
        A.GV_EMUL = None
    st = list(STATS)

    O = {}
    for nm, shape in (("qr", (n, A.QA)), ("q", (n, A.HEADS, A.HEAD_DIM)), ("kv", (n, A.HEAD_DIM)), ("o_sparse", (n, A.HEADS, A.HEAD_DIM)),
                      ("wo_a_out", (n, A.O_GROUPS, A.O_LORA)), ("attn_out", (n, A.HID))):
        if os.path.exists(f(nm)):
            O[nm] = OC.rd(f(nm), "bf16", shape)
    pairs = (("qr", "qr"), ("q", "q"), ("kv", "kv"), ("o_sparse", "o_pre"), ("wo_a_out", "oa"))
    print(f"layer {l} n {n} ks {'pick' if not a.ks else a.ks}: GEMV-twin vs generic-twin vs oracle ({D})")
    print(f"  {'stage':10s} {'gemv~gen':>10s} {'gen~oracle':>11s} {'gemv~oracle':>12s}  cos(gemv,oracle)")
    res = {}
    for nm, key in pairs:
        if nm not in O:
            continue
        r = (rel(dv[key], dg[key]), rel(dg[key], O[nm]), rel(dv[key], O[nm]), cos(dv[key], O[nm]))
        res[nm] = r
        print(f"  {nm:10s} {r[0]:10.3e} {r[1]:11.3e} {r[2]:12.3e}  {r[3]:.8f}")
    r = (rel(ov, og), rel(og, O["attn_out"]), rel(ov, O["attn_out"]), cos(ov, O["attn_out"]))
    res["attn_out"] = r
    print(f"  {'attn_out':10s} {r[0]:10.3e} {r[1]:11.3e} {r[2]:12.3e}  {r[3]:.8f}")
    ok = r[2] <= 2e-3 and r[3] >= 0.99999
    print(f"  attn_out tolerance (rel_l2 <= 2e-3, cos >= 0.99999): {'PASS' if ok else 'FAIL'}")
    print("fp16 cast of the dequantized activation, per projection call (K = input width, N = output rows):")
    for s in st:
        print(f"  K {s['K']:5d} N {s['N']:6d}  inexact {s['inexact']:.3e}  overflow {s['over']}  below-2^-14 {s['under']}  flushed-to-0 {s['flushed']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
