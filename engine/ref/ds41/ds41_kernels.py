"""DS41 oracle numerics: the leaf operations of the DeepSeek-V4.1-Flash-1017 forward on CPU.

Written from the format/semantics of the checkpoint's own inference graph (kernel.py signatures kept so
ds41_xcheck.py can drop this module in for the GPU `kernel` module).  Rule for every contraction: the
products of the quantized operands are exact in fp64, so a sum over a K-block is order independent and
the result does not depend on the thread count; it is then rounded ONCE at the same format boundary the
graph rounds at (fp32 partial per 32-wide K block, fp32 sequential accumulate over blocks, bf16 out).

Functions
  act_quant(x, block, scale_fmt, scale_dtype, inplace)    fp8-e4m3 per-block activation quant (ue8m0 pow2 scale)
  fp4_act_quant(x, block, inplace, scale_dtype)           fp4-e2m1 fake quant, in place (indexer q/k, compressed KV)
  fp8_gemm / fp4_gemm                                     blocked-scale GEMMs (weights fp8 32x32 / mxfp4 1x32)
  sparse_attn                                             window + compressed top-k attention with attn_sink
  hc_split_sinkhorn                                       hyper-connection pre/post/comb from the mixes
  mm_exact / rmsnorm                                      fp64-accumulated linear and the RMSNorm used everywhere
"""
import torch

FP8_MAX = 448.0
_INV448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
_INV6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)
_SIX = torch.tensor(6.0, dtype=torch.float32)
_FP4_VAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_LUT = torch.tensor(_FP4_VAL, dtype=torch.float32)
_FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
_FP4_TIE_UP = torch.tensor([0.75, 1.75, 3.5], dtype=torch.float32)  # ties that round UP (to the even code)


def _byte_table() -> torch.Tensor:
    """byte -> (even-k value, odd-k value); low nibble is the even k; nibble bit 3 = sign."""
    t = torch.zeros(256, 2, dtype=torch.float64)
    for b in range(256):
        for j, nib in enumerate((b & 0xF, b >> 4)):
            v = _FP4_VAL[nib & 7]
            t[b, j] = -v if nib & 8 else v
    return t


_BYTE_TAB = _byte_table()


def pow2ceil(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(v)) for positive normal fp32 v, by the exponent-field trick of the GPU kernel."""
    bits = v.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


# --------------------------------------------------------------------------------------- quantizers
def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Per-block fp8-e4m3 quantization along the last dim.  scale_fmt set -> scale rounded up to 2^k.
    inplace=True writes quantize-dequantize back into x (in x's dtype) and returns x."""
    n = x.size(-1)
    assert n % block_size == 0
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    s = pow2ceil(amax * _INV448) if scale_fmt is not None else amax * _INV448
    q = torch.div(xf, s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).to(x.dtype).reshape(x.shape))
        return x
    return q.reshape(x.shape), s.reshape(*x.shape[:-1], n // block_size).to(scale_dtype)


def _fp4_round(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> nearest fp4-e2m1 value, ties to even code; v already clamped to [-6, 6]."""
    a = v.abs()
    code = torch.bucketize(a, _FP4_MID) + torch.isin(a, _FP4_TIE_UP).to(torch.int64)
    return torch.copysign(_FP4_LUT[code], v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """fp4-e2m1 fake quantization (write back dequantized values).  scale_dtype e8m0: pow2 scale
    2^ceil(log2(max(amax, 6*2^-126)/6)) (indexer q/k).  scale_dtype e4m3: scale = e4m3(max(amax, 6*2^-9)/6)
    (compressed KV, block 16)."""
    if not inplace:
        raise NotImplementedError("oracle only needs the in-place fake-quant form")
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6.0 * 2.0 ** -9)
        s = torch.div(amax, _SIX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6.0 * 2.0 ** -126)
        s = pow2ceil(amax * _INV6)
    v = torch.div(xf, s.unsqueeze(-1)).clamp(-6.0, 6.0)
    x.copy_((_fp4_round(v) * s.unsqueeze(-1)).to(x.dtype).reshape(x.shape))
    return x


# ------------------------------------------------------------------------------------------- gemms
def _blocked_gemm(a_vals, a_s, w_vals, w_s, bs=32, budget=1 << 28):
    """out[m,n] = sum_kb ((fp32(sum_{k in kb} a[m,k]*w[n,k]) * a_s[m,kb]) * w_s[n,kb]), fp32 sequential over kb.
    a_vals [M,K] / w_vals [N,K] fp64 (quantized values, scale not applied); a_s [M,KB], w_s [N,KB] fp32.
    The scales are powers of two, so the two multiplies are exact; the only roundings are the per-block
    fp32 partial and the fp32 running sum, exactly as in the GPU kernels."""
    m, k = a_vals.shape
    n = w_vals.shape[0]
    kb = k // bs
    a3 = a_vals.reshape(m, kb, bs).transpose(0, 1)  # [KB, M, bs]
    out = torch.empty(m, n, dtype=torch.float32)
    nc = max(64, min(n, budget // max(1, kb * m * 8)))
    for n0 in range(0, n, nc):
        n1 = min(n, n0 + nc)
        w3 = w_vals[n0:n1].reshape(n1 - n0, kb, bs).permute(1, 2, 0)  # [KB, bs, nc]
        part = torch.bmm(a3, w3)  # [KB, M, nc] fp64, exact
        acc = torch.zeros(m, n1 - n0, dtype=torch.float32)
        ws = w_s[n0:n1]
        for b in range(kb):
            acc += (part[b].float() * a_s[:, b].unsqueeze(1)) * ws[:, b].unsqueeze(0)
        out[:, n0:n1] = acc
    return out


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128, out_dtype=None):
    """C = A_fp8[M,K] @ B_fp8[N,K]^T with A scale [M,K/bs], B scale [ceil(N/bs), K/bs] (32x32 weight blocks)."""
    k = a.size(-1)
    lead = a.shape[:-1]
    a2 = a.reshape(-1, k).double()
    sa = a_s.reshape(a2.shape[0], -1).float()
    sb = b_s.float().repeat_interleave(block_size, 0)[: b.size(0)]
    out = _blocked_gemm(a2, sa, b.double(), sb, block_size)
    return out.to(out_dtype or torch.get_default_dtype()).reshape(*lead, b.size(0))


def decode_fp4(b: torch.Tensor) -> torch.Tensor:
    """packed fp4 [N, K/2] (float4_e2m1fn_x2 or uint8/int8) -> fp64 values [N, K]; low nibble = even k."""
    bb = b.view(torch.uint8) if b.dtype != torch.uint8 else b
    return _BYTE_TAB[bb.reshape(-1).long()].reshape(bb.shape[0], bb.shape[1] * 2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128, out_dtype=None):
    """C = A_fp8[M,K] @ B_fp4[N,K]^T; A scale per act_block_size, B scale [N, K/32] (mxfp4)."""
    k = a.size(-1)
    lead = a.shape[:-1]
    a2 = a.reshape(-1, k).double()
    sa = a_s.reshape(a2.shape[0], -1).float()
    if act_block_size != 32:
        sa = sa.repeat_interleave(act_block_size // 32, 1)
    out = _blocked_gemm(a2, sa, decode_fp4(b), b_s.float(), 32)
    return out.to(out_dtype or torch.get_default_dtype()).reshape(*lead, b.size(0))


def mm_exact(x: torch.Tensor, w: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """x @ w.T with exact (fp64) accumulation, rounded once to fp32 then to out_dtype (default x dtype)."""
    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    y = (x.reshape(-1, x.size(-1)).double() @ w.double().T).float()
    return y.to(out_dtype).reshape(*lead, w.size(0))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm as the graph does it (fp32 math, one final round to x's dtype); the mean of squares is
    accumulated in fp64 so it does not depend on the reduction order."""
    xf = x.float()
    var = xf.double().square().mean(-1, keepdim=True).float()
    xf = xf * torch.rsqrt(var + eps)
    return (weight.float() * xf).to(x.dtype)


# ------------------------------------------------------------------------------------ sparse attention
def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, qchunk=64):
    """q [b,s,h,d] bf16, kv [b,n,d] bf16 (K == V), topk_idxs [b,s,t] int32 (-1 = empty slot).
    Slots are consumed in blocks of 64 with a running max (initial -1e30); P is rounded to bf16 before the
    P@V product; the sink adds exp(sink - max) to the denominator only.  Returns bf16 [b,s,h,d]."""
    b, s, h, d = q.shape
    out = torch.empty_like(q)
    scale = torch.tensor(softmax_scale, dtype=torch.float32)
    sink = attn_sink.float()
    n_slots = topk_idxs.size(-1)
    for bi in range(b):
        kvb = kv[bi]
        for q0 in range(0, s, qchunk):
            q1 = min(s, q0 + qchunk)
            c = q1 - q0
            qs = q[bi, q0:q1].double()
            idx = topk_idxs[bi, q0:q1]
            m = torch.full((c, h), -1e30, dtype=torch.float32)
            sum_exp = torch.zeros(c, h, dtype=torch.float32)
            acc = torch.zeros(c, h, d, dtype=torch.float32)
            for t0 in range(0, n_slots, 64):
                ids = idx[:, t0 : t0 + 64]
                valid = ids != -1
                kvg = kvb[ids.clamp_min(0).long()].double() * valid.unsqueeze(-1)  # [c, B, d]
                sc = torch.bmm(qs, kvg.transpose(1, 2)).float() * scale  # [c, h, B]
                sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
                m_prev = m
                m = torch.maximum(m_prev, sc.amax(-1))
                rescale = torch.exp(m_prev - m)
                p = torch.exp(sc - m.unsqueeze(-1))
                sum_exp = sum_exp * rescale + p.double().sum(-1).float()
                pv = torch.bmm(p.to(torch.bfloat16).double(), kvg)  # exact partial
                acc = ((acc * rescale.unsqueeze(-1)).double() + pv).float()
            sum_exp = sum_exp + torch.exp(sink.unsqueeze(0) - m)
            out[bi, q0:q1] = (acc / sum_exp.unsqueeze(-1)).to(q.dtype)
    return out


# ----------------------------------------------------------------------------------------- hyper-conn
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """mixes [b,s,(2+hc)*hc] fp32 -> pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc] (doubly stochastic)."""
    b, s, _ = mixes.shape
    hc = hc_mult
    m = mixes.float()
    sc = hc_scale.float()
    bs_ = hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + bs_[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * sc[1] + bs_[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * sc[2] + bs_[2 * hc :]).reshape(b, s, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
