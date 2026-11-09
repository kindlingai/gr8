#!/usr/bin/env python3
"""Driver for attn_ref.py (attention A): self-tests and CUDA-test fixtures.

  attn_ref_run.py selftest
  attn_ref_run.py fixture --layer 0|2 --out DIR [--shard FILE] [--n 128] [--seed 1]
  attn_ref_run.py decode  --layer 0|2 --out DIR [--shard FILE] [--n0 131] [--steps 6] [--seed 1]
    (the fixture of the prefill of n0 rows, plus `--steps` single-row decode steps from the twin: d.* tensors)

A fixture directory holds manifest.json + one raw .bin per tensor, written exactly as the checkpoint stores weights
(fp8 codes and e8m0 scales as raw bytes) so the CUDA test reads them without conversion.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ref as A  # noqa: E402


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    return ok


def selftest():
    ok = True
    g = torch.Generator().manual_seed(0)
    # act_quant: quantize+dequantize is idempotent and the error is within one fp8 step
    x = (torch.randn(64, 512, generator=g) * 3).to(torch.bfloat16)
    d1 = A.act_quant_deq(x)
    d2 = A.act_quant_deq(d1)
    ok &= check("act_quant_deq idempotent", torch.equal(d1, d2))
    rel = ((d1.float() - x.float()).abs() / x.float().abs().clamp_min(1e-3)).max().item()
    ok &= check("act_quant_deq rel err < 0.07", rel < 0.07, f"max rel {rel:.4f}")
    q, e = A.act_quant_codes(x)
    amax = x.float().reshape(64, 16, 32).abs().amax(-1)
    ok &= check("act_quant scale covers amax", bool((amax / A._pow2(e).reshape(64, 16) <= 448.0).all()))
    # e2m1 rounding vs brute-force nearest with ties to the even code
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    v = torch.linspace(-7, 7, 56001)
    got = A.e2m1_round(v)
    a = v.abs().clamp(max=6.0)
    dist = (a.unsqueeze(1) - grid.unsqueeze(0)).abs()
    best = dist.min(1).values
    near = dist <= best.unsqueeze(1) + 0.0
    pick = torch.zeros_like(a, dtype=torch.long)
    for i in range(len(a)):
        c = near[i].nonzero().flatten().tolist()
        pick[i] = c[0] if len(c) == 1 else [k for k in c if k % 2 == 0][0]
    want = torch.copysign(grid[pick], v)
    ok &= check("e2m1_round == nearest/even-tie brute force", torch.equal(got, want), f"{(got != want).sum().item()} diffs")
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    ok &= check("e2m1 ties", A.e2m1_round(ties).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], str(A.e2m1_round(ties).tolist()))
    # rope: inverse(rope(x)) ~ x, rope leaves the first 448 dims alone
    inv = A.inv_freqs(160000.0, 65536)
    inv0 = A.inv_freqs(10000.0, 0)
    ok &= check("inv_freqs[0] == 1", abs(inv0[0].item() - 1.0) < 1e-7 and abs(inv[0].item() - 1.0) < 1e-7)
    ok &= check("yarn divides lowest freq by 16", abs(inv[-1].item() * 16 / (160000.0 ** (-62 / 64)) - 1) < 1e-4,
                f"{inv[-1].item():.3e} vs {160000.0 ** (-62 / 64) / 16:.3e}")
    xr = torch.randn(40, 8, 512, generator=g).to(torch.bfloat16)
    pos = torch.arange(100, 140, dtype=torch.float32)
    r = A.rope_apply(xr, pos, inv)
    b = A.rope_apply(r, pos, inv, inverse=True)
    ok &= check("rope leaves nope dims", torch.equal(r[..., :448], xr[..., :448]))
    err = (b.float() - xr.float()).abs().max().item()
    ok &= check("rope inverse ~ identity", err < 0.05, f"max abs {err:.4f}")
    # attention core vs dense float64 masked softmax with sink
    n, h, d = 200, 4, 512
    q = torch.randn(n, h, d, generator=g).to(torch.bfloat16)
    kv = A.act_quant_deq(torch.randn(n, d, generator=g).to(torch.bfloat16))
    sink = torch.randn(h, generator=g)
    end = torch.arange(n).unsqueeze(1)
    ids = (end - 127).clamp(min=0) + torch.arange(128)
    ids = torch.where(ids > end, torch.full_like(ids, -1), ids).int()
    o = A.sparse_attn_ref(q, kv, sink, ids)
    qd, kd = q.double(), kv.double()
    want = torch.zeros(n, h, d, dtype=torch.float64)
    for t in range(n):
        lo = max(0, t - 127)
        s = (qd[t] @ kd[lo:t + 1].T) * A.SCALE
        m = torch.maximum(s.amax(-1), sink.double())
        p = torch.exp(s - m.unsqueeze(-1))
        den = p.sum(-1) + torch.exp(sink.double() - m)
        want[t] = (p @ kd[lo:t + 1]) / den.unsqueeze(-1)
    err = (o.double() - want).abs().max().item()
    scale = want.abs().max().item()
    ok &= check("sparse_attn_ref ~ dense softmax with sink", err < 0.02 * scale, f"max abs {err:.4e} (|want| max {scale:.3f})")
    return ok


def to_bytes(t: torch.Tensor):
    t = t.contiguous()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return t.view(torch.uint8).numpy().tobytes(), "u8:" + ("e4m3" if t.dtype == torch.float8_e4m3fn else "e8m0")
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes(), "bf16"
    if t.dtype == torch.float32:
        return t.numpy().tobytes(), "f32"
    if t.dtype == torch.int32:
        return t.numpy().tobytes(), "i32"
    raise ValueError(t.dtype)


def fixture(args):
    l = args.layer
    if args.shard:
        w = A.load_layer_weights(args.shard, l)
        src = args.shard
    else:
        w = A.synth_layer_weights(l, args.seed)
        src = f"synth(seed={args.seed})"
    g = torch.Generator().manual_seed(args.seed + 100)
    y = torch.randn(args.n, A.HID, generator=g).to(torch.bfloat16)
    lay = A.Layer(l, w)
    dump = {}
    out = lay.forward(y, 0, dump=dump)
    os.makedirs(args.out, exist_ok=True)
    man = {"layer": l, "n": args.n, "source": src, "tensors": {}}

    def put(name, t):
        if t is None:
            return
        b, kind = to_bytes(t)
        with open(os.path.join(args.out, name + ".bin"), "wb") as f:
            f.write(b)
        man["tensors"][name] = {"dtype": kind, "shape": list(t.shape)}

    put("y", y)
    put("inv", lay.inv)
    for k, v in w.items():
        put("w." + k, v)
    for k, v in dump.items():
        put("exp." + k, v)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    for k, v in dump.items():
        if v is not None and v.dtype != torch.int32:
            vf = v.float()
            print(f"  {k:6s} {tuple(v.shape)} finite={bool(torch.isfinite(vf).all())} absmax={vf.abs().max().item():.4g} mean|x|={vf.abs().mean().item():.4g}")
    print("fixture written:", args.out, "tensors:", len(man["tensors"]))


def decode_fixture(args):
    """Prefill n0 rows, then `steps` one-row decode steps (Layer.forward with start > 0). Writes everything `fixture` writes for the
    prefill (n = n0) and d.y [D][5120], d.q [D][64][512], d.kv [D][512], d.o [D][64][512], d.out [D][5120], d.ids int32
    [D][128 + TKmax] (ring slots then compressed ids offset by 128; -1 pads), d.comp (all compressed rows after the last step),
    d.cnorm (post-norm pre-rope rows completed by the decode steps, in order)."""
    l, n0, D = args.layer, args.n0, args.steps
    if args.shard:
        w = A.load_layer_weights(args.shard, l)
        src = args.shard
    else:
        w = A.synth_layer_weights(l, args.seed)
        src = f"synth(seed={args.seed})"
    g = torch.Generator().manual_seed(args.seed + 100)
    y = torch.randn(n0, A.HID, generator=g).to(torch.bfloat16)
    ys = torch.randn(D, A.HID, generator=g).to(torch.bfloat16)
    lay = A.Layer(l, w)
    dump = {}
    lay.forward(y, 0, dump=dump)
    steps = []
    cn = []
    for k in range(D):
        dk = {}
        lay.forward(ys[k:k + 1], n0 + k, dump=dk)
        steps.append(dk)
        if (n0 + k) % 2 == 1 and lay.ratio == 2 and l in (2, 8, 14):
            cn.append(lay.dbg["cnorm"])
    tkmax = max(s["ids"].shape[1] for s in steps) - A.WINDOW
    ids = torch.full((D, A.WINDOW + tkmax), -1, dtype=torch.int32)
    for k, s in enumerate(steps):
        ids[k, :s["ids"].shape[1]] = s["ids"][0]
    os.makedirs(args.out, exist_ok=True)
    man = {"layer": l, "n": n0, "D": D, "source": src, "tensors": {}}

    def put(name, t):
        if t is None:
            return
        b, kind = to_bytes(t)
        with open(os.path.join(args.out, name + ".bin"), "wb") as f:
            f.write(b)
        man["tensors"][name] = {"dtype": kind, "shape": list(t.shape)}

    put("y", y)
    put("inv", lay.inv)
    for k, v in w.items():
        put("w." + k, v)
    for k, v in dump.items():
        put("exp." + k, v)
    put("d.y", ys)
    put("d.q", torch.cat([s["q"] for s in steps]))
    put("d.kv", torch.cat([s["kv"] for s in steps]))
    put("d.o", torch.cat([s["o_pre"] for s in steps]))
    put("d.out", torch.cat([s["out"] for s in steps]))
    put("d.ids", ids)
    if lay.ratio == 2:
        put("d.comp", lay.comp[:(n0 + D) // 2].contiguous())
        if cn:
            put("d.cnorm", torch.cat(cn))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("decode fixture written:", args.out, "n0", n0, "steps", D, "TKmax", tkmax, "tensors:", len(man["tensors"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    f = sub.add_parser("fixture")
    f.add_argument("--layer", type=int, required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--shard")
    f.add_argument("--n", type=int, default=128)
    f.add_argument("--seed", type=int, default=1)
    d = sub.add_parser("decode")
    d.add_argument("--layer", type=int, required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--shard")
    d.add_argument("--n0", type=int, default=131)
    d.add_argument("--steps", type=int, default=6)
    d.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "decode":
        decode_fixture(a)
    else:
        fixture(a)
