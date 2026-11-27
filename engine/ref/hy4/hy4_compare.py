#!/usr/bin/env python3
"""Compare m4sim_hy4 per-layer dumps against hy4_oracle.py BF16 reference dumps."""
import argparse, os, sys
import numpy as np


def load_bf16(path: str) -> np.ndarray:
    u16 = np.fromfile(path, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def cos_and_rel(a: np.ndarray, b: np.ndarray):
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.linalg.norm(a64)
    nb = np.linalg.norm(b64)
    cos = float(np.dot(a64, b64) / (na * nb + 1e-30))
    rel = float(np.linalg.norm(a64 - b64) / (na + 1e-30))
    mx = float(np.max(np.abs(a64 - b64)))
    return cos, rel, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("sim_dir")
    ap.add_argument("--min-cos", type=float, default=0.985)
    ap.add_argument("--max-kld", type=float, default=0.15)
    args = ap.parse_args()

    files = ["embed.bf16"]
    for l in range(42):
        if l in (1, 11, 21, 31):
            files.append(f"L{l:03d}_post_engram.bf16")
        files.append(f"L{l:03d}_post_attn.bf16")
        files.append(f"L{l:03d}_out.bf16")
    files.append("final_norm.bf16")

    fails = 0
    print(f"{'tensor':<26} {'cos':>10} {'rel_l2':>10} {'max_abs':>10} {'status':>8}")
    for fn in files:
        rp = os.path.join(args.ref_dir, fn)
        sp = os.path.join(args.sim_dir, fn)
        if not os.path.exists(rp) or not os.path.exists(sp):
            continue
        a, b = load_bf16(rp), load_bf16(sp)
        if a.shape != b.shape:
            print(f"{fn:<26} SHAPE MISMATCH {a.shape} vs {b.shape}")
            fails += 1
            continue
        cos, rel, mx = cos_and_rel(a, b)
        thr = 0.999 if (fn == "embed.bf16" or fn.startswith("L000_")) else args.min_cos
        ok = (cos >= thr) and np.isfinite(cos)
        if not ok:
            fails += 1
        print(f"{fn:<26} {cos:10.6f} {rel:10.6f} {mx:10.5f} {'PASS' if ok else 'FAIL':>8}")

    rlp = os.path.join(args.ref_dir, "logits.f32")
    slp = os.path.join(args.sim_dir, "logits.f32")
    if os.path.exists(rlp) and os.path.exists(slp):
        ra = np.fromfile(rlp, dtype=np.float32).reshape(-1, 120832)
        sb = np.fromfile(slp, dtype=np.float32).reshape(-1, 120832)
        cos, rel, mx = cos_and_rel(ra.ravel(), sb.ravel())
        klds = []
        top1_eq = 0
        for i in range(ra.shape[0]):
            r = ra[i].astype(np.float64)
            q = sb[i].astype(np.float64)
            r -= r.max(); pr = np.exp(r); pr /= pr.sum(); lpr = np.log(pr + 1e-30)
            q -= q.max(); lpq = q - np.log(np.exp(q).sum())
            klds.append(float(np.sum(pr * (lpr - lpq))))
            if int(ra[i].argmax()) == int(sb[i].argmax()):
                top1_eq += 1
        mkld = float(np.mean(klds))
        ok = (mkld <= args.max_kld) and np.isfinite(mkld)
        if not ok:
            fails += 1
        print(f"{'logits.f32':<26} {cos:10.6f} {rel:10.6f} {mx:10.5f} {'PASS' if ok else 'FAIL':>8}")
        print(f"LOGITS_PARITY: rows={ra.shape[0]} mean_kld={mkld:.6f} top1_match={top1_eq}/{ra.shape[0]} last_ref={int(ra[-1].argmax())} last_sim={int(sb[-1].argmax())}")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
