#!/usr/bin/env python3
"""ds41_compare.py A B [--names n1,n2] [--tol FILE] [--json OUT] : compare oracle dumps (or engine dumps in the same layout).

A and B are dump directories (raw .bin per tensor + summary.json, written by ds41_oracle.py).  B is the reference
for the relative figures.  An engine dump dir only needs the same <name>.bin files; its dtype/shape come from
--meta-from (default: the other directory's summary.json).  Per tensor it prints
  shape, dtype, bit-equal fraction, max|a-b|, rel_L2 = ||a-b||/||b||, cos, bf16 ulp stats (bf16 / bf16-valued fp32),
  and for int tensors the fraction of equal rows and, for [S,k] expert ids, the fraction of rows with an equal SET.
With --tol FILE (fixtures manifest.json) each tensor with a stated tolerance gets PASS / FAIL.
Exit status 1 when any gated tensor fails."""
import argparse
import json
import os
import sys

import torch

DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "int32": torch.int32,
      "int64": torch.int64, "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool,
      "float8_e4m3fn": torch.float8_e4m3fn, "float8_e8m0fnu": torch.float8_e8m0fnu, "float64": torch.float64}


def load(d, name, meta):
    m = meta[name]
    dt = DT[m["dtype"]]
    raw = torch.frombuffer(bytearray(open(os.path.join(d, name + ".bin"), "rb").read()), dtype=torch.uint8)
    t = raw.view(torch.uint8 if dt == torch.bool else dt).reshape(m["shape"])
    return t.to(torch.bool) if dt == torch.bool else t


def bf16_ulps(a, b):
    """distance in bf16 ulps between a and b after rounding both to bf16 (monotone integer mapping)."""
    def key(x):
        i = x.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (key(a) - key(b)).abs()


def stats(a, b):
    out = {"shape": list(a.shape), "dtype": str(a.dtype).replace("torch.", "")}
    if a.shape != b.shape:
        out["error"] = "shape mismatch %s vs %s" % (list(a.shape), list(b.shape))
        return out
    if a.dtype in (torch.int32, torch.int64, torch.bool, torch.int8, torch.uint8):
        eq = (a == b)
        out["bit_equal_frac"] = float(eq.float().mean())
        if a.dim() >= 2 and a.dtype in (torch.int32, torch.int64):
            a2, b2 = a.reshape(-1, a.size(-1)), b.reshape(-1, b.size(-1))
            sa, sb = a2.sort(-1).values, b2.sort(-1).values
            out["row_equal_frac"] = float((a2 == b2).all(-1).float().mean())
            out["row_set_equal_frac"] = float((sa == sb).all(-1).float().mean())
        return out
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        a, b = a.float(), b.float()
    a64, b64 = a.double(), b.double()
    d = a64 - b64
    nb = b64.norm()
    out["bit_equal_frac"] = float((a.contiguous().view(torch.uint8) == b.contiguous().view(torch.uint8)).all(-1).float().mean()) \
        if a.dtype in (torch.float32, torch.bfloat16) and a.dim() == 1 else float((a64 == b64).float().mean())
    out["max_abs"] = float(d.abs().max())
    out["rel_l2"] = float(d.norm() / nb) if nb > 0 else float(d.norm())
    na = a64.norm()
    out["cos"] = float((a64 * b64).sum() / (na * nb)) if na > 0 and nb > 0 else 1.0 if na == nb else 0.0
    out["finite"] = bool(torch.isfinite(a64).all() and torch.isfinite(b64).all())
    if a.dtype == torch.bfloat16 or (a.dtype == torch.float32 and bool((a.to(torch.bfloat16).float() == a).all())):
        u = bf16_ulps(a, b).float()
        out["ulp_max"] = float(u.max())
        out["ulp_mean"] = float(u.mean())
        out["ulp_ge2_frac"] = float((u >= 2).float().mean())
    return out


def gate(st, tol):
    """tol: {"rel_l2": x, "cos": y, "bit_equal": true, "row_set_equal_frac": z, "ulp_max": u, "max_abs": m}"""
    if "error" in st:
        return False, [st["error"]]
    bad = []
    if "rel_l2" in tol and st.get("rel_l2", 1e9) > tol["rel_l2"]:
        bad.append("rel_l2 %.3g > %.3g" % (st["rel_l2"], tol["rel_l2"]))
    if "cos" in tol and st.get("cos", 0) < tol["cos"]:
        bad.append("cos %.8f < %.8f" % (st["cos"], tol["cos"]))
    if "max_abs" in tol and st.get("max_abs", 1e9) > tol["max_abs"]:
        bad.append("max_abs %.3g > %.3g" % (st["max_abs"], tol["max_abs"]))
    if "ulp_max" in tol and st.get("ulp_max", 1e9) > tol["ulp_max"]:
        bad.append("ulp_max %g > %g" % (st["ulp_max"], tol["ulp_max"]))
    if "row_set_equal_frac" in tol and st.get("row_set_equal_frac", 0) < tol["row_set_equal_frac"]:
        bad.append("row_set_equal_frac %.4f < %.4f" % (st["row_set_equal_frac"], tol["row_set_equal_frac"]))
    if tol.get("bit_equal") and st.get("bit_equal_frac", 0) != 1.0:
        bad.append("bit_equal_frac %.6f != 1" % st.get("bit_equal_frac", 0))
    if st.get("finite") is False:
        bad.append("non-finite values")
    return not bad, bad


# tensors that do not depend on the MoE arm: under --arm they are gated against the plain (faithful) reference tensor
ARM_INDEPENDENT = ("router_idx", "router_w")


def base_key(n):
    return n.split(".", 1)[1] if n[:1] == "L" and "." in n and n.split(".")[0][1:].isdigit() else n


def expected_names(mb, tols, names_arg, arm=None):
    """what must be compared: --names, else every tensor the tolerance manifest gates (its key, per layer present in
    the reference), else every tensor in the reference.  Never derived from what A happens to contain."""
    if names_arg:
        return names_arg.split(",")
    if arm:  # engine names (no suffix) for every <key>@arm the manifest gates and the reference has
        suf = "@" + arm
        keys = [k[: -len(suf)] for k in tols if k.endswith(suf)]
        out = [n[: -len(suf)] for n in mb if n.endswith(suf) and (n[: -len(suf)].split(".", 1)[-1] in keys or n[: -len(suf)] in keys)]
        return out + [n for n in mb if base_key(n) in ARM_INDEPENDENT and base_key(n) in tols]
    if not tols:
        return list(mb)
    layers = sorted({n.split(".")[0] for n in mb if n[:1] == "L" and "." in n and n.split(".")[0][1:].isdigit()})
    out = []
    for k in tols:
        if "@" in k:  # arm / regime variants (<key>@none, @w4a16, @chained, @tf) are selected by --arm / --chained / --regime, never expected by default
            continue
        if k in mb:
            out.append(k)
        out += [f"{l}.{k}" for l in layers if f"{l}.{k}" in mb]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", default=None, help="comma list; default: tolerance keys (with --tol) or every tensor of B")
    ap.add_argument("--meta-from", default=None, help="dir whose summary.json gives dtype/shape for A (default A, else B)")
    ap.add_argument("--tol", default=None, help="fixtures manifest.json with per-tensor tolerances")
    ap.add_argument("--arm", default=None,
                    help="gate A's <name> against the reference's <name>@ARM (e.g. --arm w4a16 for a W4A16 MoE) with the tolerance key <key>@ARM")
    ap.add_argument("--chained", action="store_true",
                    help="gate with the chained tolerance regime: tolerance key <key>@chained when the manifest has one (an engine layer fed its own previous output, "
                         "not the oracle's input); keys without a chained variant keep their per-op tolerance")
    ap.add_argument("--regime", default=None,
                    help="gate with the tolerance variant <key>@REGIME when the manifest has one (e.g. --regime tf for an engine layer that saw the SAME input as "
                         "the oracle layer: reference = a run made with --engine-trunk); keys without that variant keep their plain tolerance")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.chained:
        a.regime = a.regime or "chained"
    mb = json.load(open(os.path.join(a.b, "summary.json")))["tensors"]
    mp = os.path.join(a.meta_from or a.a, "summary.json")
    ma = json.load(open(mp))["tensors"] if os.path.exists(mp) else mb
    tols = json.load(open(a.tol))["tolerances"] if a.tol else {}
    names = expected_names(mb, tols, a.names, a.arm)
    res, fails = {}, 0
    for n in names:
        rn = n + "@" + a.arm if a.arm and base_key(n) not in ARM_INDEPENDENT else n   # reference tensor (and tolerance key suffix)
        miss = [w for w, d, nm in (("A", a.a, n), ("B", a.b, rn)) if not os.path.exists(os.path.join(d, nm + ".bin"))]
        if miss or n not in ma or rn not in mb:
            st = {"shape": [], "dtype": "?", "error": "MISSING in " + ",".join(miss or ["summary"]), "gate": "FAIL: missing tensor"}
            fails += 1
            res[n] = st
            print("%-26s %s" % (n, st["gate"]))
            continue
        st = stats(load(a.a, n, ma), load(a.b, rn, mb))
        key = base_key(rn)
        t = tols.get(rn, tols.get(key))
        if a.regime:
            t = tols.get(rn + "@" + a.regime, tols.get(key + "@" + a.regime, t))
        if t is not None:
            ok, why = gate(st, t["gate"] if "gate" in t else t)
            st["gate"] = "PASS" if ok else "FAIL: " + "; ".join(why)
            fails += (not ok)
        elif "error" in st:
            st["gate"] = "FAIL: " + st["error"]
            fails += 1
        res[n] = st
        print("%-26s %-9s %-18s eq=%-8s rel_l2=%-10s cos=%-12s max=%-10s ulp_max=%-6s %s" % (
            n, st.get("dtype", "?"), "x".join(map(str, st["shape"])), "%.5f" % st["bit_equal_frac"] if "bit_equal_frac" in st else "-",
            "%.3e" % st["rel_l2"] if "rel_l2" in st else "-", "%.9f" % st["cos"] if "cos" in st else "-",
            "%.3e" % st["max_abs"] if "max_abs" in st else "-", st.get("ulp_max", "-"), st.get("gate", st.get("error", ""))))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if not res:
        print("FAIL: nothing was compared (no expected tensors)")
        sys.exit(2)
    print("%d tensors compared, %d failed" % (len(res), fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
