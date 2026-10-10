"""Reference check of the dense trellis kernels (engine/kernels/dtr.cuh) on real exlr8 release matrices.

    python3 engine/ref/dtr_ref.py prepare DENSE_DIR LAYER OUT [--tp 4] [--ranks 0,1,2,3] [--rows 64]
    engine/kernels/tests/dtr_test OUT
    python3 engine/ref/dtr_ref.py check DENSE_DIR LAYER OUT --exlr8 LOADER_DIR [--device cuda]

prepare writes, per trellis matrix of dense/exlr8-dense-LLL.safetensors and rank, what the loader hands g6: the rank's
entries of M.trellis as stored, its slices of M.suh / M.svh, plus random fp16 input rows. Rank r holds chunk
(r - layer) mod TP (tools/convert/plan.py): q_b and o_proj g6's head split (64 heads, the larger slices first), the dense
MLP 12288 / TP contiguous channels. OUT/meta.txt names them for dtr_test.

check compares dtr_test's outputs with loader/exlr8.py of kindlingai/exlr8 (pure PyTorch, value for value equal to
exllamav3), passed as --exlr8 (the directory holding exlr8.py):
  weights   the FP8 copy (u8 [N][K]) against exlr8.py's decoded tiles T of the rank's entries: bit for bit; and
            exlr8.py's whole matrix W (decode_dense) sliced to the rank against reconstruct(T, suh, svh) of the slice
  decode    y = x @ W_rank^T in fp64 with W from decode_dense: max over rows of max |y - ref| / max |ref|
  prefill   the same for the MXFP8 GEMM path; gated on the row rms error (max over rows of ||y - ref|| / ||ref||, the
            max-abs ratio of 64 rows is too noisy for FP8 activations) against TOL_PREFILL times today's prefill on the
            same data (MXFP8 activations of the unrotated x times the exact W): both are FP8-activation GEMMs
  ranks     the rank partials summed (o_proj, down) or joined (q_b, gate, up) against the whole matrix
  glu       the MLP chain silu(x Wg^T) * (x Wu^T) @ Wd_rank^T against the kernels' gate+up launch and down
  bind      (dtr_test --bind outputs) rank 0's q_b, o_proj and MLP through bind::make_layer / bindp::make_layer: decode
            against the same references (BF16 outputs at TOL_BF16), prefill as above
For context it also prints the error of the routed experts' rounding points (k1_v2_ref.py: f16 sum before the output
Hadamard) on the same data.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

TOL_DECODE = 4e-3        # vs exact W: fp16 rotated activations (rel 2^-11 each) are the only rounding before the output
TOL_PREFILL = 1.5        # prefill, row rms error: at most this times today's MXFP8 prefill (k_xq8 activations, exact W)
TOL_PREFILL_GLU = 6e-2   # the MLP chain, whose today-equivalent is not emulated: the class of the single matrices
TOL_BF16 = 6e-3          # decode with BF16 output (o_proj, MLP partials): the BF16 rounding (rel 2^-9) on top of TOL_DECODE's
MATS = [("q_b", "self_attn.q_b_proj"), ("o_proj", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
        ("down", "mlp.down_proj")]


def load_exlr8(path):
    spec = importlib.util.spec_from_file_location("exlr8", os.path.join(path, "exlr8.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_st(path):
    """Minimal safetensors reader (numpy): {name: array}, metadata. Stdlib + numpy only."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", {})
    raw = np.memmap(path, np.uint8, "r")
    dt = {"U8": np.uint8, "F16": np.float16, "I64": np.int64, "U64": np.uint64, "F32": np.float32, "BF16": np.uint16}
    out = {}
    for k, v in h.items():
        if k.startswith("__pad."):
            continue
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(raw[8 + n + a:8 + n + b], dt[v["dtype"]]).reshape(v["shape"])
    return out, meta


def chunk(n, tp, j):
    """(start, width) of chunk j of n units over tp ranks, the larger chunks first (g6nr.h)"""
    w = n // tp + (1 if j < n % tp else 0)
    return j * (n // tp) + min(j, n % tp), w


def rank_channels(short, tp, r, layer):
    """the rank's channels on the sharded side: rank r holds chunk (r - layer) mod tp (tools/convert/plan.py heads and
    dense_groups), heads of 256 channels or the dense MLP's 12288 channels"""
    j = (r - layer) % tp
    if short in ("q_b", "o_proj"):
        s, w = chunk(64, tp, j)
        return 256 * s, 256 * (s + w)
    s, w = chunk(96, tp, j)
    return 128 * s, 128 * (s + w)


def trellis_mats(T):
    for short, name in MATS:
        if name + ".record" in T and int(T[name + ".record"][0]) == 1:
            yield short, name, [int(v) for v in T[name + ".record"]]


def prepare(a):
    T, _ = read_st(os.path.join(a.dense, f"exlr8-dense-{a.layer:03d}.safetensors"))
    os.makedirs(a.out, exist_ok=True)
    ranks = [int(v) for v in a.ranks.split(",")]
    rng = np.random.default_rng(a.layer * 101 + a.tp)
    lines, xs, have = [], {}, set()
    for short, name, rec in trellis_mats(T):
        fmt, out, inp, K, side = rec[:5]
        if K != 4:
            sys.exit(f"{name}: K{K}, the kernels take K4 only")
        have.add(short)
        # one input per input space: q_b reads q_a_norm, o_proj the attention output, gate/up the mlp input, down the MLP activation
        space = {"gate": "mlp_in", "up": "mlp_in"}.get(short, short)
        if space not in xs:
            xs[space] = rng.standard_normal((a.rows, inp)).astype(np.float16)
        for r in ranks:
            c0, c1 = rank_channels(short, a.tp, r, a.layer)
            ent = T[name + ".trellis"][c0 // 128:c1 // 128]
            if side == 1:
                suh, svh, x = T[name + ".suh"], T[name + ".svh"][c0:c1], xs[space]
            else:
                suh, svh, x = T[name + ".suh"][c0:c1], T[name + ".svh"], xs[space][:, c0:c1]
            nm = f"{short}.r{r}"
            np.ascontiguousarray(ent).tofile(os.path.join(a.out, nm + ".codes.bin"))
            np.ascontiguousarray(suh).tofile(os.path.join(a.out, nm + ".suh.bin"))
            np.ascontiguousarray(svh).tofile(os.path.join(a.out, nm + ".svh.bin"))
            np.ascontiguousarray(x).tofile(os.path.join(a.out, nm + ".x.bin"))
            lines.append(f"mat {nm} {side} {suh.shape[0]} {svh.shape[0]} {a.rows}")
    for space, x in xs.items():
        x.tofile(os.path.join(a.out, f"xfull.{space}.bin"))
    if {"gate", "up", "down"} <= have:
        lines.append(f"glu gate.r{ranks[0]} up.r{ranks[0]} down.r{ranks[0]}")
    with open(os.path.join(a.out, "meta.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    json.dump(dict(layer=a.layer, tp=a.tp, ranks=ranks, rows=a.rows), open(os.path.join(a.out, "prep.json"), "w"))
    print(f"prepared {len(lines)} lines in {a.out}")


def check(a):
    import torch
    X = load_exlr8(a.exlr8)
    dev = a.device
    P = json.load(open(os.path.join(a.out, "prep.json")))
    ranks, R = P["ranks"], P["rows"]
    f = X.SafeFile(os.path.join(a.dense, f"exlr8-dense-{a.layer:03d}.safetensors"))
    meta = [l.split() for l in open(os.path.join(a.out, "meta.txt")) if l.strip()]
    mats = {l[1]: (int(l[3]), int(l[4]), int(l[5])) for l in meta if l[0] == "mat"}
    had = X.hadamard(128, dev).double()
    res, ok = {}, True

    def rel(y, ref):
        """max over rows of max |y - ref| / max |ref|"""
        d = (y - ref).abs().amax(1) / ref.abs().amax(1).clamp_min(1e-30)
        return float(d.max())

    def rms(y, ref):
        """max over rows of ||y - ref|| / ||ref||"""
        return float(((y - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)).max())

    def h128(v):
        return (v.reshape(v.shape[0], -1, 128) @ had).reshape(v.shape)

    def f16(v):
        return v.half().double()

    def mx8(v):
        """MXFP8 activations as the prefill GEMMs take them (bind_prefill.cuh k_xq8): per row and 32 k, E8M0 scale
        2^ceil(log2(amax / 448)), codes E4M3 round to nearest even"""
        g = v.reshape(v.shape[0], -1, 32).float()
        am = g.abs().amax(-1, keepdim=True)
        e = torch.ceil(torch.log2((am * (1 / 448)).clamp_min(2.0 ** -126)))
        e = torch.where(am == 0, torch.zeros_like(e), e)
        sc = torch.pow(2.0, e)
        q = (g / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
        return q.reshape(v.shape).double()

    full, refs, today = {}, {}, {}
    for short, name in MATS:
        if not any(k.startswith(short + ".r") for k in mats):
            continue
        rec = f.get(name + ".record")
        t = {s: f.get(f"{name}.{s}") for s in X.DATA[int(rec[0])]}
        W = X.decode_dense(rec, t, dev).double()                    # (out, in) fp64 of exlr8.py's fp32 values
        side = int(rec[4])
        space = {"gate": "mlp_in", "up": "mlp_in"}.get(short, short)
        xfull = torch.from_numpy(np.fromfile(os.path.join(a.out, f"xfull.{space}.bin"), np.float16).reshape(R, -1)).to(dev).double()
        full[short] = (W, side, xfull)
        parts = []
        for r in ranks:
            nm = f"{short}.r{r}"
            K, N, _ = mats[nm]
            c0, c1 = rank_channels(short, P["tp"], r, a.layer)
            E = (N if side == 1 else K) // 128
            ent = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".codes.bin"), np.uint8).reshape(E, -1)).to(dev)
            suh = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".suh.bin"), np.float16)).to(dev)
            svh = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".svh.bin"), np.float16)).to(dev)
            x = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".x.bin"), np.float16).reshape(R, K)).to(dev).double()
            # weights: the rank's tiles, exlr8.py's decode
            if side == 2:
                Tr = torch.cat([X.tiles_in_major(ent[g], 4, N) for g in range(ent.shape[0])], 0)
                Wr = W[:, c0:c1]
            else:
                Tr = torch.cat([X.tiles_out_major(ent[g], 4, K, 128) for g in range(ent.shape[0])], 1)
                Wr = W[c0:c1, :]
            c8 = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".fp8.bin"), np.uint8).reshape(N, K)).to(dev)
            fp8_exact = bool(torch.equal(c8.view(torch.float8_e4m3fn).float(), Tr.T.float()))
            rec_d = float((X.reconstruct(Tr, suh, svh).T.double() - Wr).abs().max())
            ref = x @ Wr.T
            refs[nm] = ref
            dec = {}
            for fn in sorted(os.listdir(a.out)):
                if fn.startswith(nm + ".gemv.M"):
                    M = int(fn[len(nm) + 7:-4])
                    y = torch.from_numpy(np.fromfile(os.path.join(a.out, fn), np.float32).reshape(M, N)).to(dev).double()
                    dec[M] = rel(y, ref[:M])
            ypf = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".pf.bin"), np.float16).reshape(R, N)).to(dev).double()
            pf, pf_rms = rel(ypf, ref), rms(ypf, ref)
            # the routed experts' rounding points on the same data, for context (k1_v2_ref.py: f16 xa, f16 sum, f16 out)
            Td = Tr.double()
            xa = f16(h128(x * suh.double()))
            yexp = f16(h128(f16(xa @ Td)) * svh.double())
            ours = h128(xa @ Td) * svh.double()
            # today's prefill on the same data: MXFP8 activations of x (unrotated) times exact weights
            today_pf, today_rms = rel(mx8(x) @ Wr.T, ref), rms(mx8(x) @ Wr.T, ref)
            today[nm] = today_rms
            r_ = dict(fp8_exact=fp8_exact, reconstruct_vs_decode_dense=rec_d, decode_max_rel=max(dec.values()),
                      decode_by_M={str(k): round(v, 6) for k, v in sorted(dec.items())}, prefill_max_rel=pf, today_mxfp8_prefill_max_rel=today_pf,
                      prefill_rms_rel=pf_rms, today_mxfp8_prefill_rms_rel=today_rms,
                      ours_arith_max_rel=rel(ours, ref), expert_rounding_max_rel=rel(yexp, ref))
            good = fp8_exact and rec_d == 0.0 and r_["decode_max_rel"] < TOL_DECODE and pf_rms < TOL_PREFILL * today_rms
            r_["ok"] = good
            ok &= good
            res[nm] = r_
            print(json.dumps({nm: r_}))
            yfull = torch.from_numpy(np.fromfile(os.path.join(a.out, nm + ".gemv.M8.bin"), np.float32).reshape(8, N)).to(dev).double()
            parts.append((c0, yfull))
        if len(ranks) > 1:
            ref = xfull[:8] @ W.T
            y = sum(p[1] for p in parts) if side == 2 else torch.cat([p[1] for p in sorted(parts, key=lambda p: p[0])], 1)
            if side == 1:
                lo = min(rank_channels(short, P["tp"], r, a.layer)[0] for r in ranks)
                hi = max(rank_channels(short, P["tp"], r, a.layer)[1] for r in ranks)
                ref = ref[:, lo:hi]
            r_ = dict(ranks=ranks, combined_max_rel=rel(y, ref))
            r_["ok"] = r_["combined_max_rel"] < TOL_DECODE
            ok &= r_["ok"]
            res[short + ".ranks"] = r_
            print(json.dumps({short + ".ranks": r_}))
    glu = [l for l in meta if l[0] == "glu"]
    if glu:
        g_, u_, d_ = glu[0][1:]
        r = int(g_.split(".r")[1])
        Wg, _, xg = full["gate"]
        Wu, _, _ = full["up"]
        Wd, _, _ = full["down"]
        c0, c1 = rank_channels("gate", P["tp"], r, a.layer)
        hg, hu = xg @ Wg[c0:c1].T, xg @ Wu[c0:c1].T
        act = hg / (1 + torch.exp(-hg)) * hu
        ref = act @ Wd[:, c0:c1].T
        dec = {}
        for fn in sorted(os.listdir(a.out)):
            if fn.startswith("glu.gemv.M"):
                M = int(fn[10:-4])
                y = torch.from_numpy(np.fromfile(os.path.join(a.out, fn), np.float32).reshape(M, -1)).to(dev).double()
                dec[M] = rel(y, ref[:M])
        ypf = torch.from_numpy(np.fromfile(os.path.join(a.out, "glu.pf.bin"), np.float16).reshape(R, -1)).to(dev).double()
        r_ = dict(decode_max_rel=max(dec.values()), decode_by_M={str(k): round(v, 6) for k, v in sorted(dec.items())},
                  prefill_max_rel=rel(ypf, ref))
        r_["ok"] = r_["decode_max_rel"] < TOL_DECODE and r_["prefill_max_rel"] < TOL_PREFILL_GLU
        ok &= r_["ok"]
        res["glu"] = r_
        refs["glu"] = ref
        print(json.dumps({"glu": r_}))
    if os.path.exists(os.path.join(a.out, "bind.q_b.M1.bin")):
        def load(fn, rows, dt):
            v = np.fromfile(os.path.join(a.out, fn), np.uint16 if dt == "bf16" else np.float16)
            f = (v.astype(np.uint32) << 16).view(np.float32) if dt == "bf16" else v.astype(np.float32)
            return torch.from_numpy(f).reshape(rows, -1).to(dev).double()
        r_, good = {}, True
        for short, ref_name, dt, tol in (("q_b", "q_b.r0", "f16", TOL_DECODE), ("o_proj", "o_proj.r0", "bf16", TOL_BF16),
                                         ("mlp", "glu", "bf16", TOL_BF16)):
            ref = refs[ref_name]
            dec = {}
            for fn in sorted(os.listdir(a.out)):
                if fn.startswith(f"bind.{short}.M"):
                    M = int(fn[len(short) + 7:-4])
                    dec[M] = rel(load(fn, M, dt), ref[:M])
            pf = rms(load(f"bind.pf.{short}.bin", R, dt), ref)
            pf_tol = TOL_PREFILL_GLU if short == "mlp" else TOL_PREFILL * today[ref_name]
            r_[short] = dict(decode_max_rel=max(dec.values()), prefill_rms_rel=pf, ok=max(dec.values()) < tol and pf < pf_tol)
            good &= r_[short]["ok"]
        r_["ok"] = good
        ok &= good
        res["bind"] = r_
        print(json.dumps({"bind": r_}))
    json.dump(res, open(os.path.join(a.out, "check.json"), "w"), indent=1)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "check"))
    ap.add_argument("dense")
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--exlr8", help="directory holding kindlingai/exlr8 loader/exlr8.py (check)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else check(a)


if __name__ == "__main__":
    main()
