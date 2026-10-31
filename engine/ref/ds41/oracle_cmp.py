"""Compare the torch twin (attn_ref.py) and, optionally, the engine's dumped output against the oracle's dumps (ds41_oracle.py; raw .bin, bf16).

  python3 oracle_cmp.py --layer 0 --shard model-00003-of-00048.safetensors --dumps .../out/p128-faithful [--engine-out out.bin]
  python3 oracle_cmp.py --layer 2 --shard model-00005-of-00048.safetensors --dumps .../out/p128-faithful --fixture DIR

The input is the oracle's own L<l>.attn_in (bf16 [128][5120]), so every stage sees the oracle's activations. With --fixture the
fixture dir for the engine test (ds41_attn_test) is written with y = attn_in, so the GPU run is on the same input.
Metrics per stage: bit-equal fraction, max |a-b|, rel L2, and for bf16 the max ULP distance.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_ref as A  # noqa: E402
import attn_ref_run as R  # noqa: E402


def rd(path, dtype, shape=None):
    raw = np.fromfile(path, dtype=np.int16 if dtype == "bf16" else (np.int32 if dtype == "i32" else np.float32))
    t = torch.from_numpy(raw.copy())
    if dtype == "bf16":
        t = t.view(torch.bfloat16)
    return t.reshape(shape) if shape else t


def ulp_bf16(a, b):
    """max distance in bf16 steps between two bf16 tensors (sign-magnitude to ordered ints)."""
    def key(x):
        i = x.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)
    return (key(a) - key(b)).abs().max().item()


def stat(name, mine, ref):
    mine = mine.reshape(ref.shape)
    eq = (mine.view(torch.int16) == ref.view(torch.int16)).float().mean().item() if mine.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16 \
        else (mine == ref).float().mean().item()
    a, b = mine.float(), ref.float()
    den = b.norm().item()
    rl2 = (a - b).norm().item() / den if den else float("nan")
    ulp = ulp_bf16(mine, ref) if mine.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16 else float("nan")
    print(f"  {name:22s} bit_eq {eq:.6f}  max|d| {(a - b).abs().max().item():.4g}  rel_l2 {rl2:.3e}  max_ulp {ulp}")
    return eq, rl2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--engine-out", help="raw bf16 [n][5120] the engine produced for y = attn_in (optional)")
    ap.add_argument("--fixture", help="also write an engine-test fixture dir with y = attn_in")
    a = ap.parse_args()
    l, n = a.layer, a.n
    D = a.dumps

    def f(s):
        return os.path.join(D, f"L{l}.{s}.bin")

    y = rd(f("attn_in"), "bf16", (n, A.HID))
    w = A.load_layer_weights(a.shard, l)
    lay = A.Layer(l, w)
    dump = {}
    out = lay.forward(y, 0, dump=dump)

    print(f"layer {l} n {n}: torch twin vs oracle ({D})")
    stat("qr (post q_norm)", dump["qr"], rd(f("qr"), "bf16", (n, 1280)))
    for nm, key in (("q (post rope)", "q"),):
        stat(nm, dump[key], rd(f("q"), "bf16", (n, A.HEADS, A.HEAD_DIM)))
    stat("kv (post quant)", dump["kv"], rd(f("kv"), "bf16", (n, A.HEAD_DIM)))
    stat("o_sparse", dump["o_pre"], rd(f("o_sparse"), "bf16", (n, A.HEADS, A.HEAD_DIM)))
    stat("o_rot", dump["o_rot"], rd(f("o_rot"), "bf16", (n, A.HEADS, A.HEAD_DIM)))
    stat("wo_a_out", dump["oa"], rd(f("wo_a_out"), "bf16", (n, A.O_GROUPS, A.O_LORA)))
    if l in (2, 8, 14) and dump.get("comp") is not None and os.path.exists(f("lat")):
        stat("lat (compressed rows)", dump["comp"], rd(f("lat"), "bf16", (n // 2, A.HEAD_DIM)))
    if os.path.exists(f("topk")):
        tk = rd(f("topk"), "i32").reshape(n, -1)
        ids = dump["ids"][:, A.WINDOW:]
        print(f"  topk (ids incl. window offset) oracle {tuple(tk.shape)} twin {tuple(ids.shape)}; identical:",
              bool(tk.shape == ids.shape and torch.equal(ids.int(), tk.int())))
    eq, rl2 = stat("attn_out (twin)", out, rd(f("attn_out"), "bf16", (n, A.HID)))
    if a.engine_out:
        e = rd(a.engine_out, "bf16", (n, A.HID))
        print("engine:")
        stat("attn_out (engine)", e, rd(f("attn_out"), "bf16", (n, A.HID)))
        stat("engine vs twin out", e, out)
    if a.fixture:
        class NS:
            pass
        ns = NS()
        ns.layer, ns.out, ns.shard, ns.n, ns.seed = l, a.fixture, a.shard, n, 1
        # fixture() draws y from a seed; replace it with attn_in by monkeypatching torch.randn for the one call
        orig = torch.randn
        torch.randn = lambda *s, **k: y.float().reshape(*s) if tuple(s) == (n, A.HID) else orig(*s, **k)
        try:
            R.fixture(ns)
        finally:
            torch.randn = orig
        print("fixture with y = oracle attn_in written:", a.fixture)


if __name__ == "__main__":
    main()
