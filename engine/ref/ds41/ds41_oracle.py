#!/usr/bin/env python3
"""ds41_oracle.py - CPU reference forward of DeepSeek-V4.1-Flash-1017 (prefill) with per-stage dumps.

Pure torch, no GPU.  Written from the checkpoint's own inference graph; every contraction goes through
ds41_kernels (fp64-exact products, one rounding at the same format boundary the graph rounds at), so the
result does not depend on thread count.  Engram layers (1, 14) are not implemented: running them is a
"diag" row.  Layers 0, 2, 20, 39 have no Engram.

  python3 ds41_oracle.py --weights W --prompt prompts/p128.json --layers 0,2,20,39 --out OUT [--quant faithful|none]

--quant faithful  official recipe: GEMM operands (Linear inputs and `o` before wo_a) are fp8-e4m3 quantized
                  per 32 with a pow2 scale; weights fp8 32x32-block / fp4 1x32-block, dequantized exactly.
--quant none      same graph, GEMM operands stay bf16 (a W4A16-style engine).  The KV-cache format
                  quantizations (window kv fp8, compressed kv fp4, indexer q/k fp4) stay on in both arms.

SUBSET RULE (SYNTHETIC): a layer that is not an index source (e.g. L39) reuses the last list and the last
compressed kv published by an earlier layer that ran.  With --layers 0,2,20,39 that is L20's; the true L39
input is the full 0..39 prefix, where it reads L36's list.
Tie rule (oracle only): top-k on equal keys takes the lowest index first (stable sort); the number of rows
with a tie at the k-th boundary is reported.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds41_kernels as K  # noqa: E402

BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
E8 = torch.float8_e8m0fnu
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------------- config
class Cfg:
    def __init__(self, path=os.path.join(HERE, "config.json")):
        t = json.load(open(path))["text_config"]
        self.dim, self.inter, self.n_layers = t["hidden_size"], t["moe_intermediate_size"], t["num_hidden_layers"]
        self.n_heads, self.head_dim, self.rd = t["num_attention_heads"], t["head_dim"], t["qk_rope_head_dim"]
        self.q_lora, self.o_groups, self.o_lora = t["q_lora_rank"], t["o_groups"], t["o_lora_rank"]
        self.window, self.eps, self.hc_eps = t["sliding_window"], t["rms_norm_eps"], t["hc_eps"]
        self.hc, self.hc_iters = t["hc_mult"], t["hc_sinkhorn_iters"]
        self.n_experts, self.topk_experts = t["n_routed_experts"], t["num_experts_per_tok"]
        self.route_scale, self.swiglu_limit = t["routed_scaling_factor"], t["swiglu_limit"]
        self.ratios = t["compress_ratios"]
        self.rope_theta, self.compress_theta = t["rope_theta"], t["compress_rope_theta"]
        ys = t["rope_scaling"]
        self.yarn = (ys["factor"], ys["beta_fast"], ys["beta_slow"], ys["original_max_position_embeddings"])
        self.idx_heads, self.idx_dim, self.idx_topk = t["index_n_heads"], t["index_head_dim"], t["index_topk"]
        self.cand_layer, self.cand_blocks, self.cand_bs = (
            t["candidate_source_layer_id"], t["candidate_topk_blocks"], t["candidate_block_size"])
        self.kv_sources = set(t["kv_source_layer_ids"])
        self.index_sources = set(t["index_source_layer_ids"])
        self.engram_layers = set(t["engram_layer_ids"])


# ------------------------------------------------------------------------------------------ checkpoint
class Ckpt:
    def __init__(self, wdir, index=None):
        self.dir = wdir
        self.map = json.load(open(index or os.path.join(wdir, "model.safetensors.index.json")))["weight_map"]
        self._h = {}

    def __call__(self, name):
        f = self.map[name]
        if f not in self._h:
            self._h[f] = safe_open(os.path.join(self.dir, f), "pt")
        return self._h[f].get_tensor(name)

    def has(self, name):
        return name in self.map

    def shard_ok(self, layer):
        n = self.map[f"layers.{layer}.attn.wq_a.weight"]
        return os.path.exists(os.path.join(self.dir, n))


def deq_fp8(w, s):
    """fp8-e4m3 [N,K] with E8M0 scale [ceil(N/32), ceil(K/32)] -> fp64, exact."""
    n, k = w.shape
    sf = s.double().repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]
    return w.double() * sf


def deq_fp4(w, s):
    """packed fp4 [N,K/2] with E8M0 scale [N,K/32] -> fp64 [N,K], exact."""
    return K.decode_fp4(w) * s.double().repeat_interleave(32, 1)


class QW:
    """a quantized (or plain) weight: kind in fp8 / fp4 / bf16 / f32."""

    def __init__(self, kind, w, s=None):
        self.kind, self.w, self.s = kind, w, s

    def dense(self):
        return deq_fp8(self.w, self.s) if self.kind == "fp8" else deq_fp4(self.w, self.s) if self.kind == "fp4" else self.w.double()


# --------------------------------------------------------------------------------------------- rope
def make_freqs(rd, S, orig, base, factor, beta_fast, beta_slow):
    """complex64 [S, rd/2].  freqs / ramp as the graph; cos and sin of the fp32 angle are taken in fp64 and
    rounded to fp32 (correctly rounded, independent of the libm)."""
    freqs = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=F32) / rd))
    if orig > 0:
        def cdim(r):
            return rd * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(cdim(beta_fast)), 0)
        high = min(math.ceil(cdim(beta_slow)), rd - 1)
        ramp = ((torch.arange(rd // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = torch.outer(torch.arange(S), freqs)
    a64 = ang.double()
    return torch.complex(a64.cos().float(), a64.sin().float()), freqs


def rope_(x, fc, inverse=False):
    """rotate adjacent pairs of x's last dim in place; x [b,s,d] or [b,s,h,d]; fc complex64 [s, d/2].
    Real arithmetic, one fp32 rounding per op (no fused multiply-add)."""
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    c, s = fc.real, fc.imag
    if inverse:
        s = -s
    shape = (1, fc.size(0)) + (1,) * (a.ndim - 3) + (fc.size(1),)
    c, s = c.reshape(shape), s.reshape(shape)
    re = a * c - b * s
    im = a * s + b * c
    x.copy_(torch.stack([re, im], dim=-1).flatten(-2))
    return x


# -------------------------------------------------------------------------------------- misc helpers
def sha(t):
    t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.to(torch.uint8).numpy().tobytes()).hexdigest()


def stable_topk(x, k):
    """indices of the k largest along the last dim, descending, equal keys -> lowest index first."""
    v, i = torch.sort(x, dim=-1, descending=True, stable=True)
    return v[..., :k], i[..., :k]


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Level one of the two-level top-k.  logits [..., n] (unreachable positions already -inf); a block scores
    its best position; the block holding the query's newest reachable position is pinned in; keep the
    topk_blocks best blocks (ties -> lowest block); blocks scoring -inf are dropped.  Bool mask like logits."""
    n = logits.size(-1)
    pad = (-n) % block_size
    sc = F.pad(logits, (0, pad), value=float("-inf")).unflatten(-1, (-1, block_size)).amax(-1)
    nb = sc.size(-1)
    last = ((compress_lens - 1) // block_size)
    pin = torch.arange(nb) == last
    sc = sc.masked_fill(pin, float("inf"))
    kk = min(topk_blocks, nb)
    vals, idx = stable_topk(sc, kk)
    keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, idx, vals > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :n]


# ------------------------------------------------------------------------------------------ dumping
class Dumper:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.meta = {}

    def put(self, name, t):
        t = t.detach().contiguous()
        dt = str(t.dtype).replace("torch.", "")
        raw = t.to(torch.uint8) if t.dtype == torch.bool else t
        raw.view(torch.uint8).numpy().tofile(os.path.join(self.out, name + ".bin"))
        if name.endswith((".router_idx", ".router_w")):  # also as .pt for the expert-count tools
            torch.save(t, os.path.join(self.out, name + ".pt"))
        self.meta[name] = {"dtype": dt, "shape": list(t.shape), "sha256": sha(t)}
        return self.meta[name]["sha256"]

    def save(self, extra=None):
        d = {"tensors": self.meta}
        d.update(extra or {})
        json.dump(d, open(os.path.join(self.out, "summary.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------- oracle
class Shared:
    compress_kv = None
    topk_idxs = None
    index_k = None
    candidates = None


class Oracle:
    def __init__(self, ck, cfg, quant="faithful", dump=None, shadow=False, w4a16=False, main_w4a16=False):
        assert quant in ("faithful", "none")
        self.ck, self.c, self.quant, self.d = ck, cfg, quant == "faithful", dump
        self.main_w4a16 = main_w4a16  # the MAIN chain carries the W4A16 MoE (an engine run chained against it)
        self.w4a16 = w4a16 or main_w4a16  # also dump the MoE as the engine's W4A16 kernels compute it (fp16 acts), stages named <stage>@w4a16
        self._w4 = None
        self.shadow = shadow  # also dump <stage>@<other arm>: the stage recomputed with the other arm on the same inputs
        self.sh = Shared()
        self.stats = {}
        self._fc = {}

    # ---- weights
    def t(self, L, name):
        return self.ck(f"layers.{L}.{name}")

    def qw(self, L, name, kind):
        n = f"layers.{L}.{name}"
        if kind in ("fp8", "fp4"):
            return QW(kind, self.ck(n + ".weight"), self.ck(n + ".scale"))
        return QW(kind, self.ck(n + ".weight"))

    # ---- linear
    def other(self, dump, name, fn):
        """teacher-forced other arm: run fn() with the quant flag flipped (inputs are the main flow's), dump name@arm."""
        if not self.shadow:
            return
        self.quant = not self.quant
        try:
            t = fn()
        finally:
            self.quant = not self.quant
        dump(name + ("@none" if self.quant else "@faithful"), t)

    def aq(self, x):
        return K.act_quant(x, 32, "ue8m0", E8) if self.quant else None

    def lin(self, x, q, aq=None, out=BF16):
        """x [..., K] bf16 -> [..., N].  Quantized weights take the fp8 operand path when --quant faithful."""
        if q.kind == "bf16":
            return K.mm_exact(x, q.w, out)
        if q.kind == "f32":
            return K.mm_exact(x.float(), q.w.float(), F32)
        if not self.quant:
            return K.mm_exact(x, q.dense(), out)
        xq, xs = aq if aq is not None else K.act_quant(x, 32, "ue8m0", E8)
        if q.kind == "fp8":
            return K.fp8_gemm(xq, xs, q.w, q.s, block_size=32, out_dtype=out)
        return K.fp4_gemm(xq, xs, q.w, q.s, act_block_size=32, out_dtype=out)

    def norm(self, x, w):
        return K.rmsnorm(x, w, self.c.eps)

    def freqs(self, L, S):
        c = self.c
        key = (c.ratios[L] > 0, S)
        if key not in self._fc:
            if c.ratios[L] > 0:
                f, raw = make_freqs(c.rd, S, c.yarn[3], c.compress_theta, *c.yarn[:3])
            else:
                f, raw = make_freqs(c.rd, S, 0, c.rope_theta, 1, 1, 1)
            self._fc[key] = f
        return self._fc[key]

    # ---- hyper connections
    def hc_mixes(self, L, x, which):
        c = self.c
        fn, scale, base = self.t(L, f"hc_{which}_fn"), self.t(L, f"hc_{which}_scale"), self.t(L, f"hc_{which}_base")
        xf = x.flatten(2).float()
        ms = xf.double().square().mean(-1, keepdim=True).float()
        rs = torch.rsqrt(ms + c.eps)
        mixes = K.mm_exact(xf, fn, F32) * rs
        return K.hc_split_sinkhorn(mixes, scale, base, c.hc, c.hc_iters, c.hc_eps)

    @staticmethod
    def hc_pre(x, pre):
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    @staticmethod
    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    # ---- attention
    def compressor(self, L, x):
        c, p = self.c, f"layers.{L}.attn.compressor."
        ratio = c.ratios[L]
        nw = self.ck(p + "norm.weight")
        if ratio == 1:
            return self.norm(self.lin(x, QW("bf16", self.ck(p + "wkv.weight"))), nw)
        xf = x.float()
        kv = self.lin(xf, QW("f32", self.ck(p + "wkv.weight")))
        score = self.lin(xf, QW("f32", self.ck(p + "wgate.weight")))
        cut = x.size(1) - x.size(1) % ratio
        kv = kv[:, :cut].unflatten(1, (-1, ratio))
        score = score[:, :cut].unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(BF16), nw)

    def indexer(self, L, x, qr, lat_pre, offset, fc, dump):
        c, p = self.c, f"layers.{L}.attn.indexer."
        S, ratio, rd = x.size(1), c.ratios[L], c.rd
        if L in c.kv_sources:
            k = self.norm(self.lin(lat_pre, QW("bf16", self.ck(p + "wk.weight"))), self.ck(p + "k_norm.weight"))
            rope_(k[..., -rd:], fc[: S - S % ratio: ratio])
            K.fp4_act_quant(k, 32, True)
            self.sh.index_k = k
            dump("idx_k", k)
        def f_iq():
            t = self.lin(qr, self.qw(L, "attn.indexer.wq_b", "fp8")).unflatten(-1, (c.idx_heads, c.idx_dim))
            rope_(t[..., -rd:], fc[:S])
            K.fp4_act_quant(t, 32, True)
            return t
        q = f_iq()
        dump("idx_q", q)
        self.other(dump, "idx_q", f_iq)
        ik = self.sh.index_k[:, : S // ratio]
        wp = self.lin(x, QW("bf16", self.ck(p + "weights_proj.weight")))
        weights = wp * (c.idx_dim ** -0.5 * c.idx_heads ** -0.5)
        dump("idx_w", weights)
        raw = torch.einsum("bshd,btd->bsht", q.double(), ik.double()).float().to(BF16)
        score = (raw.relu() * weights.unsqueeze(-1)).double().sum(dim=2).to(BF16)
        clen = (torch.arange(1, S + 1) // ratio).unsqueeze(-1)
        score = score.masked_fill(torch.arange(S // ratio) >= clen, float("-inf"))
        dump("idx_score", score)
        if L == c.cand_layer:
            self.sh.candidates = select_candidate_blocks(score, clen, c.cand_blocks, c.cand_bs)
            dump("cand", self.sh.candidates)
        elif 0 <= c.cand_layer < L:
            score = score.masked_fill(~self.sh.candidates, float("-inf"))
        topk = min(c.idx_topk, S // ratio)
        vals, idx = stable_topk(score, topk)
        kth = vals[..., topk - 1: topk]
        ties = ((score == kth).sum(-1) > (topk - (score > kth).sum(-1))) & torch.isfinite(kth.squeeze(-1))
        self.stats[f"L{L}.index_tie_rows"] = [int(ties.sum()), int(ties.numel())]
        idx = idx.sort(dim=-1).values
        return torch.where(idx < clen, idx + offset, -1).int()

    def attention(self, L, x, dump):
        c = self.c
        S, H, HD, rd, ratio = x.size(1), c.n_heads, c.head_dim, c.rd, c.ratios[L]
        fc = self.freqs(L, S)
        aq = self.aq(x)
        def f_qr():
            return self.norm(self.lin(x, self.qw(L, "attn.wq_a", "fp8"), self.aq(x)), self.t(L, "attn.q_norm.weight"))

        def f_q():
            t = self.lin(qr, self.qw(L, "attn.wq_b", "fp8")).unflatten(-1, (H, HD))
            rope_(t[..., -rd:], fc[:S])
            return t

        def f_kv():
            t = self.norm(self.lin(x, self.qw(L, "attn.wkv", "fp8"), self.aq(x)), self.t(L, "attn.kv_norm.weight"))
            rope_(t[..., -rd:], fc[:S])
            K.act_quant(t, 32, "ue8m0", E8, True)
            return t
        qr = self.norm(self.lin(x, self.qw(L, "attn.wq_a", "fp8"), aq), self.t(L, "attn.q_norm.weight"))
        dump("qr", qr)
        self.other(dump, "qr", f_qr)
        q = f_q()
        dump("q", q)
        self.other(dump, "q", f_q)
        kv = f_kv()
        dump("kv", kv)
        self.other(dump, "kv", f_kv)
        i = torch.arange(S).unsqueeze(1)
        win = (i - c.window + 1).clamp(0) + torch.arange(min(S, c.window))
        idxs = torch.where(win > i, -1, win).int().unsqueeze(0)
        kvall = kv
        if ratio:
            lat_pre = None
            if L in c.kv_sources:
                lat_pre = self.compressor(L, x)
                dump("lat_pre", lat_pre)
            if L in c.index_sources:
                cidx = self.indexer(L, x, qr, lat_pre, kv.size(1), fc, dump)
            else:
                cidx = self.sh.topk_idxs
            self.sh.topk_idxs = cidx
            dump("topk", cidx)
            if lat_pre is not None:
                lat = lat_pre.clone()
                rope_(lat[..., -rd:], fc[: S - S % ratio: ratio])
                K.fp4_act_quant(lat, 16, True, scale_dtype=torch.float8_e4m3fn)
                self.sh.compress_kv = lat
                dump("lat", lat)
            ckv = self.sh.compress_kv[:, : S // ratio]
            kvall = torch.cat([kv, ckv], dim=1)
            idxs = torch.cat([idxs, cidx], dim=-1)
        o = K.sparse_attn(q, kvall, self.t(L, "attn.attn_sink"), idxs, HD ** -0.5)
        dump("o_sparse", o)
        rope_(o[..., -rd:], fc[:S], inverse=True)
        dump("o_rot", o)
        o = o.view(1, S, c.o_groups, -1)
        wo_a64 = deq_fp8(self.t(L, "attn.wo_a.weight"), self.t(L, "attn.wo_a.scale"))
        wo_a = wo_a64.to(BF16).view(c.o_groups, c.o_lora, -1)
        assert bool((wo_a.double().flatten() == wo_a64.flatten()).all()), "wo_a bf16 dequant is not exact"
        del wo_a64

        def f_woa():
            oo = o.clone()
            if self.quant:
                K.act_quant(oo, 32, "ue8m0", E8, True)
            if self.quant is self_quant0:
                dump("o_q", oo)
            yy = torch.empty(1, S, c.o_groups, c.o_lora, dtype=BF16)
            for g in range(c.o_groups):
                yy[:, :, g] = (oo[0, :, g].double() @ wo_a[g].double().T).float().to(BF16)
            return yy
        self_quant0 = self.quant
        y = f_woa()
        dump("wo_a_out", y)
        self.other(dump, "wo_a_out", f_woa)
        self.other(dump, "attn_out", lambda: self.lin(y.flatten(2), self.qw(L, "attn.wo_b", "fp8")))
        return self.lin(y.flatten(2), self.qw(L, "attn.wo_b", "fp8"))

    # ---- moe
    def expert(self, x, L, name, kind, weights=None):
        q1, q2, q3 = (self.qw(L, f"{name}.{w}", kind) for w in ("w1", "w2", "w3"))
        aq = self.aq(x)
        gate = self.lin(x, q1, aq).float()
        up = self.lin(x, q3, aq).float()
        lim = self.c.swiglu_limit
        up = torch.clamp(up, -lim, lim)
        gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.lin(h.to(BF16), q2)

    def expert_w4a16(self, x16, L, name, kind):
        """The engine's W4A16 expert: fp16 activations, weights exact (fp4/fp8 x ue8m0), fp32 accumulate; swiglu result stored fp16.
        Returns (fc2 output fp32, before any rounding); the route weight is applied by the caller after fc2."""
        q1, q2, q3 = (self.qw(L, f"{name}.{w}", kind) for w in ("w1", "w2", "w3"))
        gate = K.mm_exact(x16, q1.dense(), F32)
        up = K.mm_exact(x16, q3.dense(), F32)
        lim = self.c.swiglu_limit
        h = (F.silu(torch.clamp(gate, max=lim)) * torch.clamp(up, -lim, lim)).to(torch.float16)
        return K.mm_exact(h, q2.dense(), F32)

    def moe_w4a16(self, L, xs, idx, w, dump, shape):
        x16 = xs.to(torch.float16)
        yy = torch.zeros(xs.size(0), self.c.dim, dtype=F32)
        for e in range(self.c.n_experts):
            rows, slot = torch.where(idx == e)
            if rows.numel() == 0:
                continue
            yp = self.expert_w4a16(x16[rows], L, f"ffn.experts.{e}", "fp4").to(torch.float16).float()  # ypart is fp16 in the engine
            yy[rows] += w[rows, slot, None] * yp   # the GPU combine adds in slot order; the fp32 adds here are in expert order
        sh = self.expert_w4a16(x16, L, "ffn.shared_experts", "fp8")  # sh_out stays fp32 in the engine
        dump("moe_routed@w4a16", yy)
        dump("shared_out@w4a16", sh)
        out = (yy + sh).to(BF16).view(shape)
        dump("ffn_out@w4a16", out)
        return out

    def moe(self, L, x, dump):
        c = self.c
        xs = x.reshape(-1, c.dim)
        gw, gb = self.t(L, "ffn.gate.weight"), self.t(L, "ffn.gate.bias")
        scores = self.lin(xs.float(), QW("f32", gw))
        scores = F.softplus(scores).sqrt()
        keys = scores + gb
        sv, si = torch.sort(keys, dim=-1, descending=True, stable=True)
        idx = si[:, : c.topk_experts]
        gap = sv[:, c.topk_experts - 1] - sv[:, c.topk_experts]
        self.stats[f"L{L}.router_min_gap"] = float(gap.min())
        w = scores.gather(1, idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        w = w * c.route_scale
        dump("router_idx", idx.int())
        dump("router_w", w)
        def f_routed():
            yy = torch.zeros(xs.size(0), c.dim, dtype=F32)
            cnt = torch.bincount(idx.flatten(), minlength=c.n_experts).tolist()
            for e in range(c.n_experts):
                if cnt[e] == 0:
                    continue
                rows, slot = torch.where(idx == e)
                yy[rows] += self.expert(xs[rows], L, f"ffn.experts.{e}", "fp4", w[rows, slot, None])
            return yy
        y = f_routed()
        dump("moe_routed", y)
        self.other(dump, "moe_routed", f_routed)
        sh = self.expert(xs, L, "ffn.shared_experts", "fp8")
        dump("shared_out", sh)
        self.other(dump, "shared_out", lambda: self.expert(xs, L, "ffn.shared_experts", "fp8"))
        y += sh
        if self.w4a16:
            self._w4 = self.moe_w4a16(L, xs, idx, w, dump, x.shape)
            if self.main_w4a16:
                return self._w4
        return y.to(BF16).view(x.shape)

    # ---- block
    def layer(self, L, h, pre_mix):
        d = self.d

        def dump(name, t):
            if d is not None:
                d.put(f"L{L}.{name}", t)

        dump("x_in", h)
        residual = h
        ap, apo, ac = self.hc_mixes(L, h, "attn")
        dump("attn_pre", ap)
        dump("attn_post", apo)
        dump("attn_comb", ac)
        x = self.hc_pre(h, pre_mix)
        x = self.norm(x, self.t(L, "attn_norm.weight"))
        dump("attn_in", x)
        a = self.attention(L, x, dump)
        dump("attn_out", a)
        h = self.hc_post(a, residual, apo, ac)
        dump("resid_attn", h)
        resid_attn = h
        fp, fpo, fc = self.hc_mixes(L, h, "ffn")
        dump("ffn_pre", fp)
        dump("ffn_post", fpo)
        dump("ffn_comb", fc)
        x = self.hc_pre(h, ap)
        x = self.norm(x, self.t(L, "ffn_norm.weight"))
        dump("ffn_in", x)
        f = self.moe(L, x, dump)
        dump("ffn_out", f)
        h = self.hc_post(f, resid_attn, fpo, fc)
        dump("resid", h)
        if self.w4a16:
            dump("resid@w4a16", self.hc_post(self._w4, resid_attn, fpo, fc))
        return h, fp, a, resid_attn


def t_hash(h):
    """The step driver's trunk T: fp32 [rows,4,5120], bf16-valued."""
    return sha(h[0].float())[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--index", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", default="0,2,20,39")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="faithful", choices=["faithful", "none"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--wait-weights", type=int, default=0, metavar="SEC",
                    help="before each layer wait (up to SEC s total per layer) for its shard file to exist (fetch still running)")
    ap.add_argument("--main-w4a16", action="store_true", help="the main chain itself uses the W4A16 MoE (implies --w4a16): the reference for an engine that runs W4A16 end to end, layer outputs chained")
    ap.add_argument("--engine-trunk", default=None, metavar="DIR",
                    help="teacher-force the chain to an ENGINE run: after layer L, the trunk handed to the next layer is DIR/L<L>.resid.bin and "
                         "its mHC pre-mix is rebuilt from DIR/L<L>.resid_attn.bin (bf16 [1,S,4,5120]). Every layer then sees the engine's own "
                         "input, so its outputs are same-input per-layer references for the engine's slots (the chained drift is cut out)")
    ap.add_argument("--w4a16", action="store_true", help="also dump the MoE the way the engine's W4A16 kernels compute it (<stage>@w4a16)")
    ap.add_argument("--shadow", action="store_true", help="also dump <stage>@<other arm> (same inputs, quant flag flipped)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    cfg = Cfg()
    ck = Ckpt(a.weights, a.index)
    ids = torch.tensor([json.load(open(a.prompt))["ids"]], dtype=torch.long)
    S = ids.size(1)
    dump = Dumper(a.out)
    orc = Oracle(ck, cfg, a.quant, dump, a.shadow, a.w4a16, a.main_w4a16)
    layers = [int(x) for x in a.layers.split(",")]
    t0 = time.time()
    h = F.embedding(ids, ck("embed.weight"))
    h = h.unsqueeze(2).repeat(1, 1, cfg.hc, 1)
    pre_mix = torch.zeros(1, S, cfg.hc, dtype=F32)
    pre_mix[:, :, 0] = 1.0
    dump.put("embed_T", h)
    rc = [{"rc": 0, "name": "embed", "T_fp32_sha": t_hash(h)}]
    def summary(done):
        return {"prompt": a.prompt, "S": S, "quant": a.quant, "layers": layers, "rc": rc, "stats": orc.stats,
                "threads": a.threads, "seconds": time.time() - t0, "complete": done, "engine_trunk": a.engine_trunk,
                "engram": "off (checkpoint Engram layers 1,14 not implemented; shards 47/48 not fetched): rows touching a layer > 1 "
                          "are DIAG, not a quality oracle" if max(layers) >= 1 else "n/a (layer 0 only; Engram layers are 1,14)",
                "diag": max(layers) >= 1,
                "topology": "TRUE (every kv/index source layer ran)" if set(range(max(layers) + 1)) <= set(layers) and max(layers) >= 39
                else "SYNTHETIC: layers that are not index/kv sources reuse the last published list/compress_kv of a layer that ran"}

    print(f"prompt={os.path.basename(a.prompt)} S={S} quant={a.quant} layers={layers}")
    print(f"embed  T={t_hash(h)}", flush=True)
    for pos, L in enumerate(layers):
        tl = time.time()
        h_in = h
        waited = 0
        while not ck.shard_ok(L) and waited < a.wait_weights:
            if waited % 300 == 0:
                print(f"layer {L}: waiting for its shard (fetch running) {waited}s", flush=True)
            time.sleep(20)
            waited += 20
        h, pre_mix, attn_out, resid_attn = orc.layer(L, h, pre_mix)
        rc.append({"rc": 2 * pos + 1, "layer": L, "name": "consume_attn", "T_fp32_sha": t_hash(resid_attn)})
        rc.append({"rc": 2 * pos + 2, "layer": L, "name": "consume_ffn", "T_fp32_sha": t_hash(h)})
        print(f"layer {L} attn={t_hash(resid_attn)} resid={t_hash(h)} attn_out={sha(attn_out)[:12]} ({time.time()-tl:.0f}s)", flush=True)
        if a.engine_trunk:
            rf, raf = (os.path.join(a.engine_trunk, f"L{L}.{n}.bin") for n in ("resid", "resid_attn"))
            if os.path.exists(rf) and os.path.exists(raf):
                ld = lambda f: torch.frombuffer(bytearray(open(f, "rb").read()), dtype=BF16).reshape(1, S, cfg.hc, cfg.dim)
                h = ld(rf)
                pre_mix = orc.hc_mixes(L, ld(raf), "ffn")[0]
                print(f"  trunk after layer {L} replaced by the engine's ({rf})", flush=True)
        dump.save(summary(False))  # peers can read finished layers while the run continues
    if not a.no_final:
        hp = Oracle.hc_pre(h, pre_mix)
        dump.put("final.h_pre", hp)
        xn = K.rmsnorm(hp, ck("norm.weight"), cfg.eps)
        dump.put("final.norm", xn)
        last = xn[:, -1].float()
        head = ck("head.weight")
        logits = torch.cat([K.mm_exact(last, head[i:i + 8192].float(), F32) for i in range(0, head.size(0), 8192)], dim=-1)
        dump.put("final.logits", logits)
        top = logits[0].topk(5)
        print("final  logits=%s top5=%s" % (sha(logits)[:12], top.indices.tolist()))
    dump.save(summary(True))
    print("stats", json.dumps(orc.stats))
    print("done in %.0fs -> %s" % (time.time() - t0, a.out))


if __name__ == "__main__":
    main()
