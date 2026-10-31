#!/usr/bin/env python3
"""Extract one layer's MoE weights from the DeepSeek-V4.1-Flash-1017 checkpoint for `ds_shared_test fixture`.

  ds_moe_fixture_prep.py WEIGHTS_DIR ORACLE_DUMP_DIR LAYER OUT_DIR

Writes (raw little-endian, no headers) into OUT_DIR:
  x.bf16        the oracle's ffn_in for LAYER, [S][5120]
  gate.bf16     ffn.gate.weight [384][5120]            bias.f32  ffn.gate.bias [384]
  sh_w1.c/.s    shared expert w1 codes [2304][5120] E4M3 and the E8M0 scale EXPANDED to one per row [2304][160]
  sh_w3.c/.s    same for w3;  sh_w2.c/.s  w2 codes [5120][2304], scale [5120][72]
  experts.txt   the expert ids in experts.bin, one per line (the union of the oracle's router_idx)
  experts.bin   per expert: w1 packed [2304][2560], w1 scale [2304][160], w3 ..., w2 packed [5120][1152], w2 scale [5120][72]
Only the experts the oracle routed to are written (a layer's full set is 7 GB); the C++ side remaps expert ids to the compact table.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open


def main():
    wdir, odir, layer, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    wm = json.load(open(os.path.join(wdir, "model.safetensors.index.json")))["weight_map"]
    opened = {}

    def get(name):
        f = wm[name]
        if f not in opened:
            opened[f] = safe_open(os.path.join(wdir, f), "pt")
        return opened[f].get_tensor(name)

    def raw(t):
        return t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.contiguous().numpy().tobytes()

    summ = json.load(open(os.path.join(odir, "summary.json")))["tensors"]
    P = f"layers.{layer}.ffn."
    x = open(os.path.join(odir, f"L{layer}.ffn_in.bin"), "rb").read()
    assert summ[f"L{layer}.ffn_in"]["dtype"] == "bfloat16"
    open(os.path.join(out, "x.bf16"), "wb").write(x)
    idx = torch.frombuffer(bytearray(open(os.path.join(odir, f"L{layer}.router_idx.bin"), "rb").read()), dtype=torch.int32)
    used = sorted(set(idx.tolist()))
    open(os.path.join(out, "experts.txt"), "w").write("\n".join(map(str, used)) + "\n")

    g = get(P + "gate.weight")
    assert g.dtype == torch.bfloat16 and tuple(g.shape) == (384, 5120), (g.dtype, g.shape)
    open(os.path.join(out, "gate.bf16"), "wb").write(g.view(torch.int16).numpy().tobytes())
    b = get(P + "gate.bias")
    assert b.dtype == torch.float32 and tuple(b.shape) == (384,)
    open(os.path.join(out, "bias.f32"), "wb").write(b.numpy().tobytes())

    for w, (r, k) in (("w1", (2304, 5120)), ("w3", (2304, 5120)), ("w2", (5120, 2304))):
        c = get(P + f"shared_experts.{w}.weight")
        s = get(P + f"shared_experts.{w}.scale")
        assert c.dtype == torch.float8_e4m3fn and tuple(c.shape) == (r, k), (w, c.dtype, c.shape)
        assert s.dtype == torch.float8_e8m0fnu and tuple(s.shape) == (r // 32, k // 32), (w, s.dtype, s.shape)
        se = s.view(torch.uint8).repeat_interleave(32, dim=0)           # one scale per row and 32 k
        open(os.path.join(out, f"sh_{w}.c"), "wb").write(raw(c))
        open(os.path.join(out, f"sh_{w}.s"), "wb").write(raw(se))

    with open(os.path.join(out, "experts.bin"), "wb") as f:
        for e in used:
            for w, (r, k) in (("w1", (2304, 2560)), ("w3", (2304, 2560)), ("w2", (5120, 1152))):
                c = get(P + f"experts.{e}.{w}.weight")
                s = get(P + f"experts.{e}.{w}.scale")
                assert c.dtype == torch.int8 and tuple(c.shape) == (r, k), (e, w, c.dtype, c.shape)
                assert s.dtype == torch.float8_e8m0fnu and tuple(s.shape) == (r, k // 16), (e, w, s.dtype, s.shape)
                f.write(raw(c))
                f.write(raw(s))
    print(f"L{layer}: S={len(x) // (5120 * 2)} experts {len(used)} -> {out}")


if __name__ == "__main__":
    main()
