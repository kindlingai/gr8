#!/usr/bin/env python3
"""ds41_xcheck.py - cross-check the oracle against the checkpoint's own official model.py (dev tool, CPU).

The official Block runs unchanged on CPU torch, with `kernel` replaced by ds41_kernels (exact fp8/fp4 GEMMs, quant,
sparse_attn, sinkhorn) and `engram` / `image_processor` / `vision` stubbed.  Each layer gets the oracle's own x_in and
pre_mix (teacher forcing), so the stage-by-stage differences below are one layer's worth of independent
implementation noise: what is bit-equal is the same arithmetic, what differs is torch's fp32 summation order vs
the oracle's fp64-exact contractions.  Output: a table + JSON; the numbers calibrate the per-op tolerances.

python3 ds41_xcheck.py --weights W --dump OUT_OF_ORACLE_RUN --layers 0,2 --official DIR [--threads N] [--json F] [--chain] [--wait-weights SEC]
Layers must be a sub-sequence of the oracle run's --layers (pre_mix of layer k = ffn_pre of the layer before it).
--chain: no teacher forcing.  Layer k's input is the official Block's own output of layer k-1 (layers consecutive in the dump), so the
table shows the drift an independent faithful implementation accumulates over depth; the extra row `x_in` is the official layer input
vs the oracle's.  This is the number the chained tolerance regime (manifest `<key>@chained`) must clear."""
import argparse
import json
import os
import sys
import time
import types

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds41_kernels as K  # noqa: E402
import ds41_oracle as O  # noqa: E402
import ds41_compare as C  # noqa: E402

def install_shims(rec):
    """kernel -> ds41_kernels (sparse_attn wrapped to record), plus stubs for engram / image_processor / vision."""
    k = types.ModuleType("kernel")
    for n in ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn"):
        setattr(k, n, getattr(K, n))

    def sparse_attn(q, kv, sink, idxs, scale):
        o = K.sparse_attn(q, kv, sink, idxs, scale)
        rec["o_sparse"] = o.clone()
        rec["o_q"] = o  # the official forward rotates, then act-quants, this very tensor in place: final state = oracle o_q
        return o
    k.sparse_attn = sparse_attn
    sys.modules["kernel"] = k
    e = types.ModuleType("engram")
    e.EngramLayout = type("EngramLayout", (), {})
    e.NgramHashState = type("NgramHashState", (nn.Module,), {})
    sys.modules["engram"] = e
    ip = types.ModuleType("image_processor")
    ip.IMAGE = ip.IMAGE_END = ip.IMAGE_NEW_LINE = ip.IMAGE_START = ""
    sys.modules["image_processor"] = ip
    v = types.ModuleType("vision")
    v.Aligner = v.ViT = type("Stub", (nn.Module,), {})
    sys.modules["vision"] = v


def build_block(model, args, L, ck):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        blk = model.Block(L, args)
    finally:
        torch.set_default_dtype(prev)
    blk.eval()
    n_loaded = 0
    for pname, p in list(blk.named_parameters()):
        cname = f"layers.{L}.{pname}"
        if pname == "attn.wo_a.weight":  # convert.py dequantizes wo_a to bf16 (fp8 * pow2 scale: exact)
            t = O.deq_fp8(ck(cname), ck(f"layers.{L}.attn.wo_a.scale")).to(torch.bfloat16)
        elif not ck.has(cname):
            raise KeyError(cname)
        else:
            t = ck(cname)
            if p.dtype == torch.float4_e2m1fn_x2:
                t = t.view(p.dtype)
            elif t.dtype != p.dtype:
                t = t.to(p.dtype)
        assert tuple(t.shape) == tuple(p.shape), (cname, tuple(t.shape), tuple(p.shape))
        p.data = t
        n_loaded += 1
    # Linear keeps `scale` as a Parameter AND attaches it to `.weight.scale` (linear() reads weight.scale)
    for m in blk.modules():
        if isinstance(m, model.Linear) and getattr(m, "scale", None) is not None:
            m.weight.scale = m.scale
    return blk, n_loaded


def attach_hooks(blk, rec):
    def out(name):
        def h(_m, _i, o):
            rec[name] = o
        return h
    A = blk.attn
    blk.attn_norm.register_forward_hook(out("attn_in"))
    blk.ffn_norm.register_forward_hook(out("ffn_in"))
    A.q_norm.register_forward_hook(out("qr"))
    A.wq_b.register_forward_hook(out("q"))        # the official forward rotates this tensor in place
    A.kv_norm.register_forward_hook(out("kv"))    # ... rope + fp8 round trip in place
    if A.compressor is not None:
        def h_comp(_m, _i, o):  # hooks must return None: a returned value replaces the module output
            rec["lat_pre"], rec["lat"] = o.clone(), o
        A.compressor.register_forward_hook(h_comp)
    def h_wob(_m, i):
        rec["wo_a_out"] = i[0]
    A.wo_b.register_forward_pre_hook(h_wob)
    A.register_forward_hook(out("attn_out"))
    g = blk.ffn.gate
    def h_gate(_m, _i, o):
        rec["router_w"], rec["router_idx"] = o[0], o[1]
    g.register_forward_hook(h_gate)
    blk.ffn.shared_experts.register_forward_hook(out("shared_out"))
    blk.ffn.register_forward_hook(out("ffn_out"))
    calls = {"mix": 0, "post": 0}
    orig_mix, orig_post = blk.hc_mixes, blk.hc_post

    def hc_mixes(*a, **k):
        r = orig_mix(*a, **k)
        w = "attn" if calls["mix"] == 0 else "ffn"
        calls["mix"] += 1
        rec[f"{w}_pre"], rec[f"{w}_post"], rec[f"{w}_comb"] = r
        return r

    def hc_post(*a, **k):
        r = orig_post(*a, **k)
        rec["resid_attn" if calls["post"] == 0 else "resid"] = r
        calls["post"] += 1
        return r
    blk.hc_mixes, blk.hc_post = hc_mixes, hc_post


def as_oracle(t, meta):
    dt = C.DT[meta["dtype"]]
    t = t.detach().reshape(meta["shape"])
    return t.to(dt) if t.dtype != dt else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dump", required=True, help="oracle output dir (summary.json + .bin)")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--official", required=True, help="the release's inference/ directory (model.py, kernel.py)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--json", default=None)
    ap.add_argument("--chain", action="store_true", help="feed each layer the previous layer's own official output (no teacher forcing)")
    ap.add_argument("--wait-weights", type=int, default=0, metavar="SEC", help="wait up to SEC s per layer for its shard to exist (fetch running)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    rec = {}
    install_shims(rec)
    sys.path.insert(0, a.official)
    import model  # the official file, unmodified
    cfg = json.load(open(os.path.join(a.official, "config.json")))
    fields = set(model.ModelArgs.__dataclass_fields__)
    kw = {k: v for k, v in cfg.items() if k in fields}
    kw.update(max_batch_size=1, max_seq_len=a.max_seq, vision_n_layers=0)
    args = model.ModelArgs(**kw)
    model.default_dtype = torch.float8_e4m3fn
    torch.set_default_dtype(torch.bfloat16)
    ck = O.Ckpt(a.weights)
    meta = json.load(open(os.path.join(a.dump, "summary.json")))
    tm, run_layers = meta["tensors"], meta["layers"]
    result = {}
    carry = None
    for L in [int(x) for x in a.layers.split(",")]:
        rec.clear()
        waited = 0
        while not ck.shard_ok(L) and waited < a.wait_weights:
            if waited % 300 == 0:
                print(f"layer {L}: waiting for its shard {waited}s", flush=True)
            time.sleep(20)
            waited += 20
        while True:  # the oracle run may still be writing: reload summary.json until this layer's final tensor is listed
            tm = json.load(open(os.path.join(a.dump, "summary.json")))["tensors"]
            if f"L{L}.resid" in tm or waited >= a.wait_weights:
                break
            if waited % 300 == 0:
                print(f"layer {L}: waiting for the oracle dump {waited}s", flush=True)
            time.sleep(20)
            waited += 20
        blk, n = build_block(model, args, L, ck)
        x = C.load(a.dump, f"L{L}.x_in", tm)
        pos = run_layers.index(L)
        if pos == 0:
            pre_mix = torch.zeros(1, x.size(1), args.hc_mult, dtype=torch.float32)
            pre_mix[:, :, 0] = 1.0
        else:
            pre_mix = C.load(a.dump, f"L{run_layers[pos - 1]}.ffn_pre", tm)
        if a.chain and carry is not None:
            assert run_layers[pos - 1] == L - 1 and carry[0] == L - 1, "--chain needs consecutive layers"
            x, pre_mix = carry[1], carry[2]
        rec["x_in"] = x
        attach_hooks(blk, rec)
        y, ffn_pre = blk(x, 0, pre_mix, None)
        carry = (L, y, ffn_pre)
        if model.shared_attn.topk_idxs is not None and blk.attn.compress_ratio:
            rec["topk"] = model.shared_attn.topk_idxs
        rec["resid"] = y
        print(f"== layer {L}: {n} params loaded; official Block vs oracle dump {a.dump}")
        res = {}
        for name in sorted(rec, key=lambda s: list(tm).index(f"L{L}.{s}") if f"L{L}.{s}" in tm else 999):
            full = f"L{L}.{name}"
            if full not in tm:
                continue
            ref = C.load(a.dump, full, tm)
            st = C.stats(as_oracle(rec[name], tm[full]), ref)
            res[full] = st
            print("%-22s %-9s eq=%-8s rel_l2=%-10s cos=%-12s max=%-10s ulp_max=%-6s %s" % (
                full, st["dtype"], "%.5f" % st["bit_equal_frac"] if "bit_equal_frac" in st else "-",
                "%.3e" % st["rel_l2"] if "rel_l2" in st else "-", "%.9f" % st["cos"] if "cos" in st else "-",
                "%.3e" % st["max_abs"] if "max_abs" in st else "-", st.get("ulp_max", "-"),
                "set_eq=%.4f" % st["row_set_equal_frac"] if "row_set_equal_frac" in st else st.get("error", "")), flush=True)
        result[f"L{L}"] = res
        del blk
        if a.json:  # rewritten after every layer: a long --chain run can be read mid-way
            json.dump(result, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
