#!/usr/bin/env python3
"""CPU replay of engine/runtime/ds41_vision.cuh's call sequence on flat buffers, against the golden data.

    python3 engine/ref/ds41/vision_layout_check.py --weights FILE [--fixtures engine/ref/ds41/fixtures/vision] [--image square_300x300]

Every cuBLAS call of Tower::linear / Tower::attention is replayed with its exact (transa, transb, m, n, k, pointer offset,
lda, ldb, ldc, strides) on flat column-major buffers, and every kernel's index math is written the way the kernel
writes it, so a layout mistake in the CUDA file shows here without a GPU. Arithmetic is float32 with the kernels'
bf16 rounding points, so the error against the fp32 golden rows should sit near the official bf16 run's.
"""
import argparse
import json
import os

import numpy as np
import torch

PATCH, DOWN, VD, VL, VH, VHD, VI, RD, OD = 14, 3, 1024, 32, 16, 64, 2816, 32, 5120
PK, AIN, QT, RCH = 3 * 14 * 14, 1024 * 9, 256, 2048


def bf(x):
    return torch.as_tensor(x, dtype=torch.float32).to(torch.bfloat16).float()


def gemm(transa, transb, m, n, k, A, a0, lda, B, b0, ldb, C, c0, ldc):
    """C[c0 + i + j ldc] = sum_p op(A)[i, p] op(B)[p, j], column-major, on flat float32 tensors."""
    i = torch.arange(m).view(m, 1)
    j = torch.arange(n).view(1, n)
    p = torch.arange(k)
    # op(A) [m, k]
    Ai = (a0 + (i * lda + p.view(1, k)) if transa == "T" else a0 + (i + p.view(1, k) * lda))
    Bi = (b0 + (p.view(k, 1) + j * ldb) if transb == "N" else b0 + (p.view(k, 1) * ldb + j))
    out = A[Ai] @ B[Bi]
    C[(c0 + i + j * ldc).reshape(-1)] = out.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fixtures", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "vision"))
    ap.add_argument("--image", default="square_300x300")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from safetensors.torch import load_file

    W = {k: v.float().reshape(-1) for k, v in load_file(a.weights).items() if not k.startswith("__pad")}
    meta = json.load(open(os.path.join(a.fixtures, a.image + ".json")))
    vh, vw, lh, lw = meta["n_vit_h"], meta["n_vit_w"], meta["n_llm_h"], meta["n_llm_w"]
    with open(os.path.join(a.fixtures, a.image + ".ppm"), "rb") as f:
        raw = f.read()
    hdr_end = raw.index(b"255\n") + 4
    px = torch.frombuffer(bytearray(raw[hdr_end:]), dtype=torch.uint8).float()
    n = vh * vw

    # k_patchify
    nn_, ii = torch.meshgrid(torch.arange(n), torch.arange(PK), indexing="ij")
    ph, pw = nn_ // vw, nn_ % vw
    c, py, pxx = ii // (PATCH * PATCH), (ii // PATCH) % PATCH, ii % PATCH
    Wpx = vw * PATCH
    u = px[(((ph * PATCH + py) * Wpx + pw * PATCH + pxx) * 3 + c).reshape(-1)]
    pat = bf(((u / 255.0) - 0.5) / 0.5)  # float32 ops, as torch does them

    def linear(x, ldx, wname, bname, M, N, K, mode, y=None):
        out_n = N // 2 if mode == "swiglu" else N
        if y is None:
            y = torch.zeros(M * out_n)
        Wt = W[wname]
        for r0 in range(0, M, RCH):
            rows = min(RCH, M - r0)
            acc = torch.zeros(rows * N)
            gemm("T", "N", N, rows, K, Wt, 0, K, x, r0 * ldx, ldx, acc, 0, N)
            acc = acc.view(rows, N)
            ys = slice(r0 * out_n, (r0 + rows) * out_n)
            if mode == "swiglu":
                g, uu = bf(acc[:, :out_n]), bf(acc[:, out_n:])
                y[ys] = bf(bf(g * torch.sigmoid(g)) * uu).reshape(-1)
                continue
            v = acc + (W[bname].view(1, N) if bname else 0)
            if mode == "bias":
                y[ys] = bf(v).reshape(-1)
            elif mode == "bias_gelu":
                y[ys] = bf(torch.nn.functional.gelu(bf(v))).reshape(-1)
            else:  # residual
                y[ys] = bf(y[ys].view(rows, N) + bf(v)).reshape(-1)
        return y

    def rmsnorm(x, wname):
        v = x.view(-1, VD)
        rs = torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
        return bf(W[wname].view(1, VD) * (v * rs)).reshape(-1)

    x = linear(pat.reshape(-1), PK, "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", n, VD, PK, "bias")
    # k_rope tables
    j = torch.arange(RD)
    fi = j % (RD // 2)
    inv = 1.0 / torch.pow(torch.tensor(10000.0), (2 * fi).float() / RD)
    rows_ph = (torch.arange(n) // vw).float().view(n, 1)
    rows_pw = (torch.arange(n) % vw).float().view(n, 1)
    ang = torch.where(j.view(1, RD) < RD // 2, rows_ph, rows_pw) * inv.view(1, RD)
    cs, sn = torch.cos(ang), torch.sin(ang)
    for l in range(VL):
        p = f"vision.blocks.{l}."
        h = rmsnorm(x, p + "norm1.weight")
        qkv = linear(h, VD, p + "attn.wqkv.weight", p + "attn.wqkv.bias", n, 3 * VD, VD, "bias")
        q3 = qkv.view(n, 3 * VD)
        for hh in range(2 * VH):
            base = (hh if hh < VH else VD // VHD + (hh - VH)) * VHD
            x1, x2 = q3[:, base:base + RD].clone(), q3[:, base + RD:base + 2 * RD].clone()
            q3[:, base:base + RD] = bf(x1 * cs - x2 * sn)
            q3[:, base + RD:base + 2 * RD] = bf(x2 * cs + x1 * sn)
        qkv = q3.reshape(-1)
        o = torch.zeros(n * VD)
        ld = 3 * VD
        for q0 in range(0, n, QT):
            tq = min(QT, n - q0)
            S = torch.zeros(VH * tq * n)
            for hd in range(VH):  # strided batched: strideA = strideB = 64, strideC = tq n
                gemm("T", "N", n, tq, VHD, qkv, VD + hd * VHD, ld, qkv, q0 * ld + hd * VHD, ld, S, hd * tq * n, n)
            Sv = S.view(VH * tq, n) * 0.125
            P = bf(torch.softmax(Sv, dim=-1)).reshape(-1)
            for hd in range(VH):  # strideA = 64, strideB = tq n, strideC = 64
                tmp = torch.zeros(o.numel())
                gemm("N", "N", VHD, tq, n, qkv, 2 * VD + hd * VHD, ld, P, hd * tq * n, n, tmp, q0 * VD + hd * VHD, VD)
                idx = (q0 * VD + hd * VHD + torch.arange(VHD).view(VHD, 1) + torch.arange(tq).view(1, tq) * VD).reshape(-1)
                o[idx] = bf(tmp[idx])
        x = linear(o, VD, p + "attn.wo.weight", p + "attn.wo.bias", n, VD, VD, "res", y=x)
        h = rmsnorm(x, p + "norm2.weight")
        m = linear(h, VD, p + "mlp.w1.weight", None, n, 2 * VI, VD, "swiglu")
        x = linear(m, VI, p + "mlp.w2.weight", None, n, VD, VI, "res", y=x)
        if l % 8 == 7:
            print("block", l, flush=True)
    h = rmsnorm(x, "vision.norm.weight")
    L = lh * lw
    # k_unfold
    t, col = torch.meshgrid(torch.arange(L), torch.arange(AIN), indexing="ij")
    bh, bw = t // lw, t % lw
    cc, kk = col // 9, col % 9
    r, q = 3 * bh + kk // 3, 3 * bw + kk % 3
    valid = (r < vh) & (q < vw)
    src = ((r.clamp(max=vh - 1) * vw + q.clamp(max=vw - 1)) * VD + cc)
    U = torch.where(valid, h[src.reshape(-1)].view(L, AIN), torch.zeros(()))
    a1 = linear(U.reshape(-1), AIN, "aligner.w1.weight", "aligner.w1.bias", L, OD, AIN, "bias_gelu")
    a2 = linear(a1, OD, "aligner.w2.weight", "aligner.w2.bias", L, OD, OD, "bias").view(L, OD)

    rows = meta["rows"]
    ref = np.fromfile(os.path.join(a.fixtures, a.image + ".ref.bin"), dtype=np.float32).reshape(len(rows), OD)
    got = a2[rows].numpy()
    e_rows = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    import importlib.util
    spec = importlib.util.spec_from_file_location("vg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_golden.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    R = vg.proj_matrix(OD, meta["nproj"])
    pref = np.fromfile(os.path.join(a.fixtures, a.image + ".proj.bin"), dtype=np.float32).reshape(L, -1)
    e_proj = float(np.linalg.norm(a2.numpy() @ R - pref) / np.linalg.norm(pref))
    tol = 1.5 * meta["bf16_vs_f32"]["rel_fro"]
    ok = e_rows <= tol and e_proj <= tol
    print(json.dumps({"image": a.image, "rows_rel": e_rows, "proj_rel": e_proj, "official_bf16_rel": meta["bf16_vs_f32"]["rel_fro"],
                      "tol": tol, "pass": ok}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
