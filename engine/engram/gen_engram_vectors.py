#!/usr/bin/env python3
"""Oracle vectors for the Engram element maths: dequant of table rows and the gate, from the official reference (MIT).

  gen_engram_vectors.py --ref <ref-official> --out <vectors_math.bin>

Dequant: official ParallelEngramEmbedding.forward on random E4M3 bytes and E8M0 scales (world_size 1, every row owned).
Gate:    official Engram.forward with wkv/embed replaced by stubs (wkv returns the kv given here, bf16 [rows][25600]).
Only the GPU-only `kernel` module is stubbed; none of its functions run. File layout (little endian):
  u32 N, u32 R, then
  deq_w u8[N][256]  deq_s u8[N][8]  deq_out u16[N][256] (bf16 bits)
  gate_h u16[R][4][5120]  gate_kv u16[R][25600]  gate_q f32[4][5120]  gate_k f32[4][5120]  gate_ref f32[R][4]  gate_out u16[R][4][5120]
"""
import argparse, os, struct, sys, types

import numpy as np

N, R, DIM, HC = 512, 8, 5120, 4


def load_reference(ref):
    k = types.ModuleType("kernel")
    for n in ("act_quant", "fp4_gemm", "fp8_gemm", "sparse_attn", "hc_split_sinkhorn", "fp8_index", "fp4_act_quant"):
        setattr(k, n, lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("GPU kernel stub called")))
    sys.modules["kernel"] = k
    sys.path.insert(0, os.path.join(ref, "inference"))
    import model  # noqa: E402
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch as _t; _t.set_grad_enabled(False)
    import torch
    import torch.nn as nn
    model = load_reference(a.ref)
    rng = np.random.default_rng(20251025)

    # ---- dequant -------------------------------------------------------------------------------------------------------
    w = rng.integers(0, 256, size=(N, 256), dtype=np.uint8)
    w[(w == 0x7F) | (w == 0xFF)] = 0x38          # NaN encodings never occur in a table; replace by 1.0
    w[0, :] = np.arange(256, dtype=np.uint8) % 255            # row 0: every finite byte value once (0x7F -> wraps below)
    w[0][w[0] == 0x7F] = 0x7E
    w[0][w[0] == 0xFF] = 0xFE
    pool = np.array([0, 1, 2, 3, 60, 100, 118, 126, 127, 128, 130, 150, 200, 250, 253, 254], dtype=np.uint8)
    s = np.where(rng.random((N, 8)) < 0.6, rng.integers(100, 151, size=(N, 8)), rng.choice(pool, size=(N, 8))).astype(np.uint8)
    emb = model.ParallelEngramEmbedding(N, 256)
    emb.weight.data = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn)
    emb.scale.data = torch.from_numpy(s.copy()).view(torch.float8_e8m0fnu)
    ids = torch.arange(N, dtype=torch.long)
    out = emb(ids).contiguous()
    assert out.dtype == torch.bfloat16 and out.shape == (N, 256)
    deq_out = out.view(torch.int16).numpy().view(np.uint16)

    # ---- gate ----------------------------------------------------------------------------------------------------------
    q = rng.standard_normal((HC, DIM)).astype(np.float32)
    k = rng.standard_normal((HC, DIM)).astype(np.float32)
    wgt = q * k
    h = rng.standard_normal((R, HC, DIM)).astype(np.float32)
    key = rng.standard_normal((R, HC, DIM)).astype(np.float32)
    val = rng.standard_normal((R, DIM)).astype(np.float32)
    key[1] = h[1] * np.sign(wgt) + 0.1 * rng.standard_normal((HC, DIM)).astype(np.float32)     # large positive dot
    key[2] = -h[2] * np.sign(wgt) + 0.1 * rng.standard_normal((HC, DIM)).astype(np.float32)    # large negative dot
    h[3] = 0.0                                   # zero stream: rsqrt(eps)
    key[4] = 0.0                                 # zero key: dot = 0, clamp
    h[5] *= 1e-9; key[5] *= 1e-12                # eps in play
    h[6] *= 300.0; key[6] *= 30.0                # large
    for j in range(HC):                          # streams of very different size in one row
        h[7, j] *= 10.0 ** (j - 1); key[7, j] *= 10.0 ** (1 - j)
    hb = torch.from_numpy(h).to(torch.bfloat16)
    kvb = torch.cat([torch.from_numpy(key).to(torch.bfloat16).flatten(1), torch.from_numpy(val).to(torch.bfloat16)], dim=1)
    assert kvb.shape == (R, DIM * (HC + 1))

    class Stub(nn.Module):
        pass
    eng = Stub()
    eng.dim, eng.hc_mult, eng.clamp_value, eng.eps = DIM, HC, 1e-6, 1e-20
    eng.q_weight = nn.Parameter(torch.from_numpy(q))
    eng.k_weight = nn.Parameter(torch.from_numpy(k))
    eng.embed = lambda ids: torch.zeros(1, 1, 1, 1)           # forward() only passes it to wkv
    eng.wkv = lambda x: kvb.reshape(1, R, -1)
    x = hb.reshape(1, R, HC, DIM)
    res = model.Engram.forward(eng, x, torch.zeros(1, R, 1, dtype=torch.long)).reshape(R, HC, DIM)
    assert res.dtype == torch.bfloat16

    # the gate itself, the same torch ops as Engram.forward (the reference returns only h + gate * value)
    keyf = kvb[:, :HC * DIM].float().unflatten(-1, (HC, DIM))
    weight = eng.q_weight.float() * eng.k_weight.float()
    hf = hb.float()
    rstd = torch.rsqrt(hf.square().mean(-1) + 1e-20) * torch.rsqrt(keyf.square().mean(-1) + 1e-20)
    dot = (hf * weight * keyf).sum(-1) * rstd * DIM ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    chk = (hf + gate.unsqueeze(-1) * kvb[:, HC * DIM:].float().unsqueeze(-2)).to(torch.bfloat16)
    assert torch.equal(chk.view(torch.int16), res.view(torch.int16)), "gate replica differs from Engram.forward"

    with open(a.out, "wb") as f:
        f.write(struct.pack("<II", N, R))
        for arr in (w, s, deq_out,
                    hb.view(torch.int16).numpy().view(np.uint16), kvb.view(torch.int16).numpy().view(np.uint16),
                    q, k, gate.numpy().astype(np.float32), res.view(torch.int16).numpy().view(np.uint16)):
            f.write(np.ascontiguousarray(arr).tobytes())
    print("vectors: N=%d dequant rows, R=%d gate rows -> %s (%d B); gate range %.4g..%.4g" %
          (N, R, a.out, os.path.getsize(a.out), float(gate.min()), float(gate.max())))


if __name__ == "__main__":
    main()
