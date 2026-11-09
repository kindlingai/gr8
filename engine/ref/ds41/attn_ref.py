#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 attention block, torch reference (attention A).

Semantics are the release's own inference code (inference/model.py: Attention, Compressor, sparse_attn,
act_quant, fp4_act_quant, fp8_gemm). Written from that reading, not copied: every quantize/round step the GPU kernels
perform is an explicit step here so the CUDA kernels in engine/attn/ds41_attn.cuh can be compared stage by stage.

Stages of one layer (n rows, positions start_pos .. start_pos+n-1, y = the bf16 block input after attn_norm):
  qa   = wq_a(y)                 act_quant(y) -> fp8 gemm -> bf16
  qr   = q_norm(qa)              RMSNorm(weight, eps 1e-20), fp32 math, bf16 out
  q    = wq_b(qr)                act_quant(qr) -> fp8 gemm -> bf16 [n, heads, 512]; no per-head norm; rope on the last 64
  kv   = kv_norm(wkv(y))         rope on the last 64, then act_quant(32, ue8m0) over all 512 (window KV is fp8-valued)
  comp = compressor(y)           layers 2, 8, 14 only (ratio 2): fp32 wkv/wgate, softmax over the pair, norm, rope at the
                                 group start position, fp4 (block 16, e4m3 scale) quantize
  o    = sparse_attn(q, [window | compressed], sink)   blocks of 64 slots, bf16 P, fp32 accumulate
  o    = conj-rope(o tail); act_quant(32) ; wo_a grouped (8 groups x 4096 -> 1024) ; wo_b (8192 -> 5120)
"""
import math
import sys
import numpy as np
import torch

HEADS, HEAD_DIM, ROPE_DIM = 64, 512, 64
QA, O_LORA, O_GROUPS, HID, WINDOW = 1280, 1024, 8, 5120, 128
EPS = 1e-20
SCALE = HEAD_DIM ** -0.5
FP8_MAX_INV = np.float32(1.0 / 448.0)


# ------------------------------------------------------------------ quantizers (the kernels' arithmetic)
def _pow2_ceil_exp(v: torch.Tensor) -> torch.Tensor:
    """ceil(log2(v)) for positive fp32 v by exponent bits: the kernels' fast_log2_ceil."""
    bits = v.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _pow2(e: torch.Tensor) -> torch.Tensor:
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def act_quant_codes(x: torch.Tensor):
    """x [..., N] -> (fp8 codes as float8_e4m3fn [..., N], exponent e [..., N/32] int32 with scale = 2**e)."""
    N = x.shape[-1]
    xf = x.float().reshape(-1, N // 32, 32)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    e = _pow2_ceil_exp(amax * torch.tensor(FP8_MAX_INV))
    s = _pow2(e)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), e.reshape(*x.shape[:-1], N // 32)


def act_quant_deq(x: torch.Tensor) -> torch.Tensor:
    """quantize + dequantize back to x.dtype (the kernels' inplace=True form)."""
    q, e = act_quant_codes(x)
    N = x.shape[-1]
    s = _pow2(e).unsqueeze(-1)
    y = (q.float().reshape(-1, N // 32, 32) * s.reshape(-1, N // 32, 1)).reshape(x.shape)
    return y.to(x.dtype)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """round-to-nearest-even onto the e2m1 grid, saturating at 6 (sign kept).
    Grid codes 0..7 = 0 .5 1 1.5 2 3 4 6; a tie goes to the even code: 0.25->0 0.75->1 1.25->1 1.75->2 2.5->2 3.5->4 5->4."""
    a = v.abs().clamp(max=6.0)
    out = torch.zeros_like(a)
    for thr, val, up_on_tie in ((0.25, 0.5, False), (0.75, 1.0, True), (1.25, 1.5, False), (1.75, 2.0, True),
                                (2.5, 3.0, False), (3.5, 4.0, True), (5.0, 6.0, False)):
        out = torch.where((a >= thr) if up_on_tie else (a > thr), torch.full_like(a, val), out)
    return torch.copysign(out, v)


def fp4_quant_deq_e4m3(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """fp4_act_quant(block 16, inplace, e4m3 scale): scale = e4m3(amax/6), values e2m1(x/scale) * scale."""
    N = x.shape[-1]
    xf = x.float().reshape(-1, N // block, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(6.0 * 2.0 ** -9)
    s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    y = e2m1_round((xf / s).clamp(-6.0, 6.0)) * s
    return y.reshape(x.shape).to(x.dtype)


# ------------------------------------------------------------------ rope
def inv_freqs(base: float, orig_len: int, factor: float = 16.0, beta_fast: float = 32.0, beta_slow: float = 1.0):
    """fp32 inverse frequencies [32] in the release's op order (YaRN when orig_len > 0)."""
    dim = ROPE_DIM
    f = 1.0 / (torch.tensor(float(base), dtype=torch.float32) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if orig_len > 0:
        def cd(rot):
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(cd(beta_fast)), 0)
        high = min(math.ceil(cd(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        f = f / factor * (1 - smooth) + f * smooth
    return f


def rope_apply(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """rotate adjacent pairs of the last 64 dims of x [n, ..., 512] (bf16) at integer positions pos [n] (fp32 math)."""
    ang = torch.outer(pos.float(), inv)               # fp32 angle = pos * inv_freq
    c, s = torch.cos(ang), torch.sin(ang)
    if inverse:
        s = -s
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [ROPE_DIM // 2]
    c, s = c.reshape(shape), s.reshape(shape)
    t = x[..., -ROPE_DIM:].float().unflatten(-1, (-1, 2))
    a, b = t[..., 0], t[..., 1]
    r = torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)
    out = x.clone()
    out[..., -ROPE_DIM:] = r.to(x.dtype)
    return out


# ------------------------------------------------------------------ layers of math
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return (w.float() * (xf * torch.rsqrt(var + eps))).to(x.dtype)


def deq_weight(w8: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """fp8 [N, K] + ue8m0 scale [N/32, K/32] -> float32 [N, K]."""
    s = sc.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w8.float() * s


GV_EMUL = None   # gv_emul.py sets a callable (x, w8, sc) -> bf16 to run the projections with the decode GEMV's arithmetic


def fp8_linear(x: torch.Tensor, w8: torch.Tensor, sc: torch.Tensor, wdeq=None) -> torch.Tensor:
    """act_quant(x) then block-scaled fp8 gemm -> bf16. x [n, K] bf16."""
    if GV_EMUL is not None:
        return GV_EMUL(x, w8, sc)
    xd = act_quant_deq(x).float()
    wd = wdeq if wdeq is not None else deq_weight(w8, sc)
    return (xd @ wd.T).to(torch.bfloat16)


def sparse_attn_ref(q, kv, sink, idx):
    """q [n, h, 512] bf16, kv [R, 512] bf16, sink [h] fp32, idx [n, L] int32 (-1 = empty slot) -> o [n, h, 512] bf16.
    Blocks of 64 slots in list order, running max, bf16 P for the PV product, sum over fp32 P, sink in the denominator."""
    n, h, d = q.shape
    out = torch.zeros(n, h, d, dtype=torch.bfloat16)
    qf, kvf = q.float(), kv.float()
    for t in range(n):
        ids = idx[t]
        L = ids.shape[0]
        m = torch.full((h,), -1e30)
        sumexp = torch.zeros(h)
        acc = torch.zeros(h, d)
        for b0 in range(0, L, 64):
            blk = ids[b0:b0 + 64]
            valid = blk >= 0
            rows = kvf[blk.clamp(min=0).long()] * valid.unsqueeze(-1)
            s = qf[t] @ rows.T                                       # [h, 64]
            s = torch.where(valid.unsqueeze(0), s * SCALE, torch.full_like(s, -float("inf")))
            mp = m
            m = torch.maximum(mp, s.amax(-1))
            sc = torch.exp(mp - m)
            p = torch.exp(s - m.unsqueeze(-1))
            sumexp = sumexp * sc + p.sum(-1)
            acc = acc * sc.unsqueeze(-1) + p.to(torch.bfloat16).float() @ rows
        sumexp = sumexp + torch.exp(sink - m)
        out[t] = (acc / sumexp.unsqueeze(-1)).to(torch.bfloat16)
    return out


class Layer:
    """One attention layer's weights (TP=1: 64 heads, 8 groups) and the per-layer caches."""

    def __init__(self, l: int, w: dict):
        self.l = l
        self.ratio = 0 if l < 2 else (2 if l < 20 else 1)
        self.w = w
        self.wd = {k: deq_weight(w[k + ".weight"], w[k + ".scale"]) for k in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}
        if self.ratio:
            self.inv = inv_freqs(160000.0, 65536)
        else:
            self.inv = inv_freqs(10000.0, 0)
        self.win = torch.zeros(WINDOW, HEAD_DIM, dtype=torch.bfloat16)        # ring cache
        self.comp = None                                                        # compressed cache [R, 512]
        self.kv_state = torch.zeros(2, HEAD_DIM)
        self.score_state = torch.full((2, HEAD_DIM), -float("inf"))

    # --- compressor (ratio 2, source layers only): official Compressor.forward semantics for start_pos == 0 / one token
    def compress(self, y: torch.Tensor, start: int):
        n = y.shape[0]
        wkv = self.w["compressor.wkv.weight"].float()
        wgate = self.w["compressor.wgate.weight"].float()
        x = y.float()
        kv, score = x @ wkv.T, x @ wgate.T
        ratio = 2
        if start == 0:
            rem = n % ratio
            cut = n - rem
            if rem:
                self.kv_state[:rem], self.score_state[:rem] = kv[cut:], score[cut:]
                kv, score = kv[:cut], score[:cut]
            kv = kv.unflatten(0, (-1, ratio))
            score = score.unflatten(0, (-1, ratio))
            self.dbg = dict(ckv=kv.flatten(0, 1), csc=score.flatten(0, 1))
            lat = (kv * score.softmax(dim=1)).sum(dim=1)
            first = 0
        else:
            assert n == 1
            slot = start % ratio
            self.kv_state[slot], self.score_state[slot] = kv[0], score[0]
            if (start + 1) % ratio:
                return None
            lat = (self.kv_state * self.score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
            first = start // ratio
        self.dbg = getattr(self, "dbg", {})
        self.dbg["cpool"] = lat
        lat = rmsnorm(lat.to(torch.bfloat16), self.w["compressor.norm.weight"])
        self.dbg["cnorm"] = lat
        pos = torch.arange(lat.shape[0], dtype=torch.float32) * ratio + (first * ratio)
        lat = rope_apply(lat, pos, self.inv)
        self.dbg["crope"] = lat
        lat = fp4_quant_deq_e4m3(lat, 16)
        return lat, first

    def forward(self, y: torch.Tensor, start: int, comp_src: "Layer" = None, topk: torch.Tensor = None, dump: dict = None):
        n = y.shape[0]
        w = self.w
        pos = torch.arange(start, start + n, dtype=torch.float32)
        qa = fp8_linear(y, w["wq_a.weight"], w["wq_a.scale"], self.wd["wq_a"])
        qr = rmsnorm(qa, w["q_norm.weight"])
        q0 = fp8_linear(qr, w["wq_b.weight"], w["wq_b.scale"], self.wd["wq_b"]).view(n, HEADS, HEAD_DIM)
        q = rope_apply(q0, pos, self.inv)
        kv0 = fp8_linear(y, w["wkv.weight"], w["wkv.scale"], self.wd["wkv"])
        kvn = rmsnorm(kv0, w["kv_norm.weight"])
        kvr = rope_apply(kvn, pos, self.inv)
        kv = act_quant_deq(kvr)
        # window: list of kv rows + slot ids exactly like get_window_topk_idxs
        if start == 0:
            kvw = kv
            end = torch.arange(n).unsqueeze(1)
            ids = (end - WINDOW + 1).clamp(min=0) + torch.arange(min(n, WINDOW))
            ids = torch.where(ids > end, torch.full_like(ids, -1), ids).int()
            if n <= WINDOW:
                self.win[:n] = kv
            else:
                cut = n % WINDOW
                self.win[cut:WINDOW], self.win[:cut] = kv[-WINDOW:].split([WINDOW - cut, cut], dim=0)
        else:
            assert n == 1
            self.win[start % WINDOW] = kv[0]
            kvw = self.win
            oldest = start % WINDOW + 1
            ids = torch.cat([torch.arange(oldest, WINDOW), torch.arange(oldest)])
            ids = torch.where(ids > start, torch.full_like(ids, -1), ids).int().unsqueeze(0)
        allkv, allids = kvw, ids
        comp_lat = None
        if self.ratio == 2:
            if self.l in (2, 8, 14):
                r = self.compress(y, start)
                if r is not None:
                    lat, first = r
                    if self.comp is None:
                        self.comp = torch.zeros(4096, HEAD_DIM, dtype=torch.bfloat16)
                    self.comp[first:first + lat.shape[0]] = lat
                    comp_lat = lat
                comp_src = self
            cl = (start + n) // 2
            ck = comp_src.comp[:cl]
            if topk is None:
                tk = min(512, cl)
                j = torch.arange(tk)
                end_pos = torch.arange(start, start + n).unsqueeze(1)
                lens = (end_pos + 1) // 2
                topk = torch.where(j.unsqueeze(0) < lens, j.unsqueeze(0), torch.full_like(j, -1).unsqueeze(0)).int()
            off = kvw.shape[0]
            tki = torch.where(topk >= 0, topk + off, torch.full_like(topk, -1)).int()
            allkv = torch.cat([kvw, ck], dim=0)
            allids = torch.cat([ids.expand(n, -1) if ids.shape[0] == 1 and n > 1 else ids, tki], dim=-1)
        o = sparse_attn_ref(q, allkv, w["attn_sink"], allids)
        o_pre = o
        o = rope_apply(o, pos, self.inv, inverse=True)
        o_rot = o
        oq_codes, oq_e = act_quant_codes(o.reshape(n, O_GROUPS, -1))
        if GV_EMUL is not None:
            ow, osc = w["wo_a.weight"].view(O_GROUPS, O_LORA, -1), w["wo_a.scale"].view(O_GROUPS, O_LORA // 32, -1)
            og = o.reshape(n, O_GROUPS, -1)
            oa = torch.stack([GV_EMUL(og[:, g], ow[g], osc[g]) for g in range(O_GROUPS)], dim=1)
        else:
            od = act_quant_deq(o.reshape(n, O_GROUPS, -1)).float()
            wa = self.wd["wo_a"].view(O_GROUPS, O_LORA, -1)
            oa = torch.einsum("ngd,grd->ngr", od, wa).to(torch.bfloat16)
        out = fp8_linear(oa.reshape(n, -1), w["wo_b.weight"], w["wo_b.scale"], self.wd["wo_b"])
        if dump is not None:
            dump.update(dict(qa=qa, qr=qr, q0=q0, q=q, kv0=kv0, kvn=kvn, kvr=kvr, kv=kv, comp=comp_lat, o_pre=o_pre,
                             o_rot=o_rot, oa=oa, out=out, ids=allids))
            if self.ratio == 2 and self.l in (2, 8, 14) and start == 0:
                dump.update(self.dbg)
        return out


# ------------------------------------------------------------------ weights
def load_layer_weights(shard_path: str, l: int) -> dict:
    from safetensors import safe_open
    w = {}
    pre = f"layers.{l}.attn."
    with safe_open(shard_path, "pt") as f:
        for k in f.keys():
            if k.startswith(pre):
                w[k[len(pre):]] = f.get_tensor(k)
    return w


def synth_layer_weights(l: int, seed: int = 0) -> dict:
    """random fp8/ue8m0 weights in the checkpoint's shapes, scales ~ realistic, for runs without the shards"""
    g = torch.Generator().manual_seed(seed + l)
    def mat(n, k, std):
        w = (torch.randn(n, k, generator=g) * std * 8).clamp(-440, 440).to(torch.float8_e4m3fn)
        e = torch.randint(-12, -8, (n // 32, k // 32), generator=g)
        return w, (2.0 ** e.float()).to(torch.float8_e8m0fnu)
    w = {}
    for name, (n, k) in dict(wq_a=(QA, HID), wq_b=(HEADS * HEAD_DIM, QA), wkv=(HEAD_DIM, HID),
                             wo_a=(O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS), wo_b=(HID, O_GROUPS * O_LORA)).items():
        w[name + ".weight"], w[name + ".scale"] = mat(n, k, 1.0)
    w["q_norm.weight"] = (1 + 0.1 * torch.randn(QA, generator=g)).to(torch.bfloat16)
    w["kv_norm.weight"] = (1 + 0.1 * torch.randn(HEAD_DIM, generator=g)).to(torch.bfloat16)
    w["attn_sink"] = torch.randn(HEADS, generator=g)
    if l in (2, 8, 14):
        w["compressor.wkv.weight"] = (torch.randn(HEAD_DIM, HID, generator=g) * 0.02).to(torch.bfloat16)
        w["compressor.wgate.weight"] = (torch.randn(HEAD_DIM, HID, generator=g) * 0.02).to(torch.bfloat16)
        w["compressor.norm.weight"] = (1 + 0.1 * torch.randn(HEAD_DIM, generator=g)).to(torch.bfloat16)
    return w


if __name__ == "__main__":
    print(__doc__)
