#!/usr/bin/env python3
"""idx_oracle_check.py ORACLE_DIR --weights DIR [--layers 2,20] : run the CPU index reference (idx_ref.py) on the oracle's inputs
(attn_in, qr, lat_pre) with the real checkpoint weights and compare the oracle's own index stages:
  L<l>.idx_k (fp4 e8m0/32 fake-quant of the index key), L<l>.idx_q, L<l>.idx_w, L<l>.idx_score, L<l>.topk (+offset), L<l>.cand,
  L<l>.lat (the fp4 e4m3/16 fake-quant of the compressed KV row, ratio-1 layer 20).
The GEMM outputs (wk, wq_b fp8, weights_proj) use the oracle's own K.* kernels; everything downstream is idx_ref (the code the CUDA
kernels were verified against). Exit 1 when a stage mismatches by more than its bound (printed)."""
import argparse
import os
import sys
import json

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idx_ref as R                       # noqa: E402
import ds41_oracle as O                   # noqa: E402

K = O.K
BF16 = torch.bfloat16
DT = {"bf16": torch.bfloat16, "i32": torch.int32, "bool": torch.bool, "f32": torch.float32}


def load(d, name, dt, shape):
    raw = torch.frombuffer(bytearray(open(os.path.join(d, name + ".bin"), "rb").read()), dtype=torch.uint8)
    t = raw.view(torch.uint8 if dt == "bool" else DT[dt])
    t = t.reshape(shape)
    return t.to(torch.bool) if dt == "bool" else t


def cmp(name, a, b, bound=None):
    a64, b64 = a.double(), b.double()
    fin = torch.isfinite(a64) & torch.isfinite(b64)
    same_inf = (torch.isinf(a64) == torch.isinf(b64)).all() and (a64[torch.isinf(a64)] == b64[torch.isinf(b64)]).all()
    d = (a64[fin] - b64[fin])
    rel = float(d.norm() / b64[fin].norm().clamp_min(1e-30)) if fin.any() else 0.0
    eq = float((a64[fin] == b64[fin]).double().mean()) if fin.any() else 1.0
    ok = bool(same_inf) and (bound is None or rel <= bound)
    print("%-4s %-18s rel_L2=%.3e bit_equal=%.5f max|d|=%.3e inf_pattern_equal=%s%s" % (
        "PASS" if ok else "FAIL", name, rel, eq, float(d.abs().max()) if d.numel() else 0.0, bool(same_inf),
        "" if bound is None else "  (bound %.0e)" % bound))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("--layers", default="2,20")
    ap.add_argument("--weights", required=True, help="the checkpoint's safetensors directory")
    ap.add_argument("--seq", type=int, default=0, help="S; default read from summary.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = O.Cfg()
    ck = O.Ckpt(a.weights)
    S = a.seq
    if S == 0:
        S = json.load(open(os.path.join(a.oracle, "summary.json")))["S"]
    table = R.rope_table(S + 8)
    ok = True
    for L in [int(x) for x in a.layers.split(",")]:
        ratio = cfg.ratios[L]
        n = S // ratio
        p = "layers.%d.attn." % L
        print("== layer %d ratio %d S=%d (compressed rows %d)" % (L, ratio, S, n))
        x = load(a.oracle, "L%d.attn_in" % L, "bf16", (1, S, cfg.dim))
        qr = load(a.oracle, "L%d.qr" % L, "bf16", (1, S, cfg.q_lora))
        lat_pre = load(a.oracle, "L%d.lat_pre" % L, "bf16", (1, n, cfg.head_dim)) if L in cfg.kv_sources else None
        # ---- idx_k
        if L in cfg.kv_sources:
            wk = ck(p + "indexer.wk.weight")
            kn = ck(p + "indexer.k_norm.weight")
            pos = torch.arange(n) * ratio
            k, kpack, kexp = R.build_ica_row(lat_pre[0], wk, kn, cfg.eps, pos, table)
            kdq = R.deq_fp4_ue8m0(kpack, kexp, 32)
            ref_k = load(a.oracle, "L%d.idx_k" % L, "bf16", (1, n, cfg.idx_dim))[0]
            ok &= cmp("idx_k", kdq.to(BF16), ref_k, 2e-3)
        # ---- idx_q (oracle fp8 gemm of qr; weights_proj; both oracle kernels)
        wq = O.QW("fp8", ck(p + "indexer.wq_b.weight"), ck(p + "indexer.wq_b.scale"))
        xq, xs = K.act_quant(qr, 32, "ue8m0", O.E8)
        qb = K.fp8_gemm(xq, xs, wq.w, wq.s, block_size=32, out_dtype=BF16)[0]       # [S, 4096]
        qpack, qexp = R.build_index_q(qb, torch.arange(S), table)
        qdq = R.deq_fp4_ue8m0(qpack, qexp, 32)
        ref_q = load(a.oracle, "L%d.idx_q" % L, "bf16", (1, S, cfg.idx_heads, cfg.idx_dim))[0]
        ok &= cmp("idx_q", qdq.to(BF16), ref_q, 2e-3)
        wp = K.mm_exact(x, ck(p + "indexer.weights_proj.weight"), BF16)[0]
        w = R.index_weights(wp)
        ref_w = load(a.oracle, "L%d.idx_w" % L, "bf16", (1, S, cfg.idx_heads))[0]
        ok &= cmp("idx_w", w, ref_w, 1e-3)
        # ---- score from the oracle's own quantized q/k (isolates the MQA-logit formula + masking)
        ref_k_all = load(a.oracle, "L%d.idx_k" % L, "bf16", (1, n, cfg.idx_dim))[0] if L in cfg.kv_sources else None
        klen = R.klen_of(torch.arange(S), ratio)
        if ref_k_all is not None:
            dots = torch.einsum("rhd,nd->rhn", ref_q.float(), ref_k_all.float()).to(BF16)
            prod = (torch.relu(dots).float() * ref_w.float().unsqueeze(-1)).to(BF16)
            sc = prod.float().sum(dim=1).to(BF16)
            sc = sc.masked_fill(torch.arange(n).unsqueeze(0) >= klen.long().unsqueeze(1), float("-inf"))
            ref_s = load(a.oracle, "L%d.idx_score" % L, "bf16", (1, S, n))[0]
            ok &= cmp("idx_score(formula)", sc, ref_s, None)
            # and with MY packed q/k pipeline end to end
            sc2 = R.mqa_psum(qpack, qexp, w, kpack, kexp, klen)
            ok &= cmp("idx_score(idx_ref e2e)", sc2, ref_s, 2e-2)
            # ---- topk against the oracle's list (offset S for the compressed part); my list from the oracle's own score
            tk = R.topk_sorted(ref_s, klen, min(cfg.idx_topk, n))
            ref_t = load(a.oracle, "L%d.topk" % L, "i32", (1, S, min(cfg.idx_topk, n)))[0]
            mine = torch.where(tk >= 0, tk + S, tk)
            if L == cfg.cand_layer:
                cand_ref = load(a.oracle, "L%d.cand" % L, "bool", (1, S, n))[0]
                cb = R.candidate_blocks(ref_s, klen)           # [S, 2048] block ids, -1 pad
                cand = torch.zeros(S, n, dtype=torch.bool)
                for r in range(S):
                    for b in cb[r]:
                        b = int(b)
                        if b >= 0:
                            cand[r, b * 8: min(b * 8 + 8, n)] = True
                eqc = bool((cand == cand_ref).all())
                print("%-4s %-18s equal=%s" % ("PASS" if eqc else "FAIL", "cand", eqc))
                ok &= eqc
            eqt = float((mine == ref_t).all(-1).float().mean())
            print("%-4s %-18s row_equal=%.4f" % ("PASS" if eqt == 1.0 else "FAIL", "topk(from oracle score)", eqt))
            ok &= eqt == 1.0
        # ---- ca row (ratio-1): fp4 e4m3/16 of the roped latent
        if ratio == 1:
            roped = lat_pre[0].clone()
            roped[..., -64:] = R.rope_interleaved(lat_pre[0][..., -64:], torch.arange(S), table)
            cpack, csf = R.cast_fp4_e4m3(roped, 16)
            cdq = R.deq_fp4_e4m3(cpack, csf, 16)
            ref_lat = load(a.oracle, "L%d.lat" % L, "bf16", (1, n, cfg.head_dim))[0]
            ok &= cmp("lat(ca row dq)", cdq.to(BF16), ref_lat, 2e-3)
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
