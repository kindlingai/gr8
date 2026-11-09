"""DS41 attention B (indexer) reference (torch CPU, no GPU): the compressed-row producers, the indexer logits, the dense top-512,
the candidate-block stage (L20 source, 2048 blocks x 8) and the candidate consumers (L24/28/32/36).

The numerics follow the official DeepSeek inference code (the release's inference/model.py Compressor / Indexer /
select_candidate_blocks / RMSNorm / precompute_freqs_cis, kernel.py fp4_act_quant); the topology and the visible-row counts were
first read from DeepInfer (Apache-2.0, learn-only) and agree with it. Nothing is copied. The facts this file fixes (each one is a
test in engine/tests/test_ds41_idx_ref.py, and test_official_transcription runs the official Indexer decode path line by line):
  * a compressed row exists for a token when (pos + 1) % ratio == 0; its index is pos // ratio and its rope position is
    (pos // ratio) * ratio. A row's visible count for a query at pos is klen = (pos + 1) // ratio, so ratio 1 includes the
    row of the query's own token (written first, in the same step).
  * ratio 1 has no softmax-compress state: the row is wkv(x) (bf16) -> RMSNorm(w, eps) -> rope(trailing 64) -> FP4 e2m1 with an
    E4M3 scale per 16 over all 512 dims (the rope tail included): 256 B codes + 32 B scales = 288 B. RMSNorm rounds ONCE:
    bf16(w * (x * rsqrt(mean(x^2) + eps))) with fp32 inside.
  * the index-K row is k_proj(post-norm PRE-rope ca row) -> RMSNorm -> rope -> FP4 e2m1 (ue8m0 per 32), 68 B.
  * index q: q_b(q_a) (32 x 128, bf16) -> rope(trailing 64) -> FP4 e2m1 (ue8m0 per 32). No norm, no hadamard.
  * index weights: weights_proj(x) (bf16) * 2^-6 (= index_head_dim^-0.5 * index_n_heads^-0.5 = 1/sqrt(4096), exact in bf16).
  * p_sum[q][j] in the official bf16 steps: dot_h = bf16(q_h . k_j) (fp32 accumulate), r_h = bf16(relu(dot_h) * w_h),
    p = bf16(sum_h r_h) (fp32 sum).
  * topk outputs are sorted ASCENDING by index and padded with -1 at the tail (official: topk(sorted=False).indices.sort();
    DeepInfer deep_select sorted_index=True, idx_oob_fill_value=-1). Ties at the cut are unspecified in torch.topk and deep_select; this file takes the LOWER index first.
  * candidate blocks (L20): block score = max of p_sum over the block's valid tokens, the last valid block
    gets +inf, top-2048 blocks sorted ascending, -1 pad. Consumers score only the tokens of those blocks in a compact row
    (slot * 8 + offset), top-512 over compact_lens valid columns (a prefix, because the only partial block is the last
    valid one), then restore compact -> block_id * 8 + offset.
"""
import math

import torch

F4_MAX = 6.0
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ----------------------------------------------------------------------------------------------------------------------
# topology: the cross-layer maps from the release config (text_config) and the checks DeepInfer enforces at load
# ----------------------------------------------------------------------------------------------------------------------
def source_map(ids, n):
    """layer -> nearest source id <= layer, for layers >= ids[0]; -1 before the first source."""
    ids = sorted(ids)
    out = []
    for l in range(n):
        s = -1
        for i in ids:
            if i <= l:
                s = i
        out.append(s)
    return out


class Topo:
    def __init__(self, text, n_layers=None):
        n = n_layers or text['num_hidden_layers']
        self.n = n
        self.ratio = list(text['compress_ratios'][:n])
        kv_ids = list(text['kv_source_layer_ids'])
        idxk_ids = list(text.get('index_k_source_layer_ids', kv_ids))
        idx_ids = list(text['index_source_layer_ids'])
        self.kv_ids, self.idxk_ids, self.idx_ids = kv_ids, idxk_ids, idx_ids
        self.kv_src = source_map(kv_ids, n)
        self.idxk_src = source_map(idxk_ids, n)
        self.idx_src = source_map(idx_ids, n)
        self.cand_src_id = text['candidate_source_layer_id']
        self.cand_blocks = text['candidate_topk_blocks']
        self.cand_block = text['candidate_block_size']
        self.index_topk = text['index_topk']
        self.window = text['sliding_window']
        self.cand_src = [self.cand_src_id if l >= self.cand_src_id else -1 for l in range(n)]
        self.validate()

    def validate(self):
        for ids in (self.kv_ids, self.idxk_ids, self.idx_ids):
            m = source_map(ids, self.n)
            assert all(m[i] == i for i in ids), 'a source maps to itself'
        assert set(self.kv_ids) <= set(self.idxk_ids) <= set(self.idx_ids), 'index segments refine index-K refine kv'
        for l in range(self.n):
            if self.kv_src[l] >= 0:
                assert self.ratio[l] == self.ratio[self.kv_src[l]] > 0, f'layer {l}: consumer ratio == source ratio'
                assert self.idxk_src[l] == self.kv_src[l] or self.idxk_src[l] > self.kv_src[l]
            if self.idx_src[l] >= 0:
                assert self.idx_src[l] >= self.idxk_src[l] >= self.kv_src[l]
        c = self.cand_src_id
        assert c in self.idx_ids and self.ratio[c] > 0, 'candidate source is an index source with a compressor'
        assert self.cand_blocks % 8 == 0 and 0 < self.cand_block <= 16
        assert self.cand_blocks * self.cand_block >= self.index_topk, 'candidate capacity >= index_topk'
        for l in range(c, self.n):
            assert self.ratio[l] == self.ratio[c], 'candidate consumers share the source ratio'

    def is_kv_source(self, l): return l in self.kv_ids
    def is_idxk_source(self, l): return l in self.idxk_ids
    def is_index_source(self, l): return l in self.idx_ids
    def is_candidate_source(self, l): return l == self.cand_src_id
    def is_candidate_consumer(self, l): return self.is_index_source(l) and l > self.cand_src_id

    def role(self, l):
        """What one layer's attention-B step runs: 'none' (window only), 'reuse' (reads another layer's topk),
        'dense' (indexer over all rows), 'dense+cand' (dense + candidate blocks, L20), 'cand' (candidate consumer)."""
        if self.ratio[l] == 0 or self.idx_src[l] < 0:
            return 'none'
        if not self.is_index_source(l):
            return 'reuse'
        if self.is_candidate_source(l):
            return 'dense+cand'
        return 'cand' if self.is_candidate_consumer(l) else 'dense'


# ----------------------------------------------------------------------------------------------------------------------
# casts: kernel.py fp4_act_quant (official). scale_dtype E8M0 for the indexer (groups of 32), E4M3 for the compressed KV (groups of 16)
# ----------------------------------------------------------------------------------------------------------------------
F4_MAX_INV = torch.tensor(1.0 / 6.0, dtype=torch.float32)   # the float32 constant the kernel multiplies by


def _log2_ceil(x):
    """kernel.py fast_log2_ceil: ceil(log2(x)) from the float32 bits (exact for normal x)."""
    bits = x.float().contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _pow2(e):
    """kernel.py fast_pow2: 2^e as float32 for integer e."""
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def quant_e2m1(x):
    """float -> 4-bit code (sign bit 3, magnitude index 0..7 of FP4_VALUES), round to nearest, ties to the even code,
    saturating at 6."""
    ax = x.abs()
    code = torch.zeros(x.shape, dtype=torch.uint8)
    for boundary, up in zip((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), (False, True, False, True, False, True, False)):
        code += ((ax >= boundary) if up else (ax > boundary)).to(torch.uint8)
    neg = (x < 0) & (code != 0)
    return code | (neg.to(torch.uint8) << 3)


def deq_e2m1(code):
    table = torch.tensor(FP4_VALUES, dtype=torch.float32)
    val = table[(code & 7).long()]
    return torch.where(((code & 8) != 0) & ((code & 7) != 0), -val, val)


def pack_nibbles(code):
    """[..., n] codes -> [..., n/2] bytes: element 2i in the low nibble, 2i+1 in the high nibble."""
    c = code.reshape(*code.shape[:-1], code.shape[-1] // 2, 2)
    return (c[..., 0] & 0xF) | ((c[..., 1] & 0xF) << 4)


def unpack_nibbles(packed):
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def cast_fp4_ue8m0(x, g=32):
    """The index rows and index q: x [..., n] -> (packed uint8 [..., n/2], E8M0 bytes [..., n/g]).
    scale = 2^ceil(log2(max(amax, 6*2^-126) * float32(1/6))); code = e2m1(clamp(x / scale, -6, 6))."""
    n = x.shape[-1]
    v = x.float().reshape(*x.shape[:-1], n // g, g)
    amax = v.abs().amax(dim=-1).clamp_min(6.0 * 2.0 ** -126)
    e = _log2_ceil(amax * F4_MAX_INV)
    sf = _pow2(e)
    code = quant_e2m1((v / sf.unsqueeze(-1)).clamp(-F4_MAX, F4_MAX).reshape(x.shape))
    return pack_nibbles(code), (e + 127).clamp(0, 255).to(torch.uint8)


def deq_fp4_ue8m0(packed, exp, g=32):
    v = deq_e2m1(unpack_nibbles(packed))
    n = v.shape[-1]
    sf = (exp.to(torch.int32) << 23).view(torch.float32)
    return (v.reshape(*v.shape[:-1], n // g, g) * sf.unsqueeze(-1)).reshape(v.shape)


def cast_fp4_e4m3(x, g=16):
    """The compressed-KV (ca) row: x [..., n] -> (packed uint8 [..., n/2], float8_e4m3fn scales [..., n/g]).
    scale = e4m3(max(amax, 6*2^-9) / 6) (a true division); code = e2m1(clamp(x / float32(scale), -6, 6))."""
    n = x.shape[-1]
    v = x.float().reshape(*x.shape[:-1], n // g, g)
    amax = v.abs().amax(dim=-1).clamp_min(6.0 * 2.0 ** -9)
    sf = (amax / F4_MAX).to(torch.float8_e4m3fn)
    code = quant_e2m1((v / sf.float().unsqueeze(-1)).clamp(-F4_MAX, F4_MAX).reshape(x.shape))
    return pack_nibbles(code), sf


def deq_fp4_e4m3(packed, sf, g=16):
    v = deq_e2m1(unpack_nibbles(packed))
    n = v.shape[-1]
    return (v.reshape(*v.shape[:-1], n // g, g) * sf.float().unsqueeze(-1)).reshape(v.shape)


# ----------------------------------------------------------------------------------------------------------------------
# rmsnorm and rope (model.py RMSNorm, precompute_freqs_cis, apply_rotary_emb)
# ----------------------------------------------------------------------------------------------------------------------
def rms_norm(x, w, eps):
    """bf16 in, bf16 out: fp32 statistics, one rounding: bf16(w * (x * rsqrt(mean(x^2) + eps)))."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (w.float() * y).to(torch.bfloat16)


def rope_freqs(dim, base, yarn):
    """float32 [dim/2] inverse frequencies; yarn = (factor, original positions, beta_fast, beta_slow) or None."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if yarn is not None:
        factor, orig_pos, beta_fast, beta_slow = yarn

        def corr_dim(rot):
            return dim * math.log(orig_pos / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_table(n_pos, dim=64, base=160000.0, yarn=(16, 65536, 32, 1)):
    """[n_pos, dim] fp32: cos then sin (dim/2 each); the angle is the float32 product position * freq (torch.outer).
    The engine must build the same table on the GPU with powf/cosf/sinf (no fast-math)."""
    freqs = torch.outer(torch.arange(n_pos, dtype=torch.float32), rope_freqs(dim, base, yarn))
    return torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)


def rope_interleaved(x, positions, table, conjugate=False):
    """x [T, ..., 64] bf16, dim 0 the token axis; pairs (2i, 2i+1) rotate by angle table[pos, i] as a complex product in fp32,
    rounded to bf16 (apply_rotary_emb)."""
    half = table.shape[-1] // 2
    cos = table[positions, :half]
    sin = table[positions, half:]
    if conjugate:
        sin = -sin
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    xf = x.float()
    a, b = xf[..., 0::2], xf[..., 1::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out.to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------------------------------
# row producers
# ----------------------------------------------------------------------------------------------------------------------
def row_positions(pos, ratio):
    """For tokens at pos: (is_boundary, compressed row index, rope position) as DeepInfer's prepare_v4_decode_metadata."""
    pos = torch.as_tensor(pos, dtype=torch.int64)
    boundary = ((pos + 1) % ratio) == 0
    idx = pos // ratio
    return boundary, idx, torch.where(boundary, idx * ratio, torch.zeros_like(pos))


def klen_of(pos, ratio):
    return ((torch.as_tensor(pos, dtype=torch.int64) + 1) // ratio).to(torch.int32)


def build_ca_row_r1(wkv_out, norm_w, eps, positions, table):
    """Ratio-1 ca row. wkv_out [T,512] bf16 (the bf16 projection output). Returns pre_rope (post-norm, bf16), roped (bf16)
    and the record (packed [T,256] uint8, scales e4m3 [T,32])."""
    pre = rms_norm(wkv_out, norm_w, eps)
    roped = pre.clone()
    roped[..., -64:] = rope_interleaved(pre[..., -64:], positions, table)
    return pre, roped, cast_fp4_e4m3(roped, 16)


def build_ica_row(pre_rope, kproj_w, knorm_w, eps, positions, table):
    """Index-K row from the post-norm pre-rope ca row. kproj_w [128,512] bf16. Returns (k roped bf16 [T,128], packed [T,64],
    exps [T,4])."""
    k = (pre_rope.float() @ kproj_w.float().t()).to(torch.bfloat16)
    k = rms_norm(k, knorm_w, eps)
    k[..., -64:] = rope_interleaved(k[..., -64:], positions, table)
    packed, exp = cast_fp4_ue8m0(k, 32)
    return k, packed, exp


def build_index_q(q_b, positions, table, n_heads=32, head_dim=128):
    """q_b [R, 32*128] bf16 -> (packed [R,32,64], exps [R,32,4])."""
    q = q_b.reshape(q_b.shape[0], n_heads, head_dim).clone()
    q[..., -64:] = rope_interleaved(q[..., -64:], positions, table)
    return cast_fp4_ue8m0(q, 32)


INDEX_W_SCALE = 2.0 ** -6   # index_head_dim^-0.5 * index_n_heads^-0.5 = 1/sqrt(128 * 32)


def index_weights(weights_proj_out):
    """weights_proj(x) [R,32] bf16 -> the bf16 per-head weights of the logits (the multiply by 2^-6 is exact)."""
    return (weights_proj_out.float() * INDEX_W_SCALE).to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------------------------------
# indexer logits, selection, candidate blocks
# ----------------------------------------------------------------------------------------------------------------------
def mqa_psum(q_pack, q_exp, w, k_pack, k_exp, klen, cols=None):
    """p_sum [R, N] bf16, -inf at columns >= klen[r]. q_*: from build_index_q; w [R,32] bf16 (index_weights); k_*: [N,64] / [N,4].
    The official bf16 steps: bf16(dot) per head, bf16(relu * w) per head, bf16(sum over heads)."""
    q = deq_fp4_ue8m0(q_pack, q_exp, 32)                    # [R,32,128]
    k = deq_fp4_ue8m0(k_pack, k_exp, 32)                    # [N,128]
    dots = torch.einsum('rhd,nd->rhn', q, k).to(torch.bfloat16)
    prod = (torch.relu(dots).float() * w.float().unsqueeze(-1)).to(torch.bfloat16)
    s = prod.float().sum(dim=1).to(torch.bfloat16)
    n = s.shape[1]
    col = torch.arange(n).unsqueeze(0)
    return s.masked_fill(col >= klen.long().unsqueeze(1), float('-inf'))


def topk_sorted(p, end, k, fill=-1):
    """Per row: the k largest p[r, :end[r]] (ties: lower index first), returned as int32 indices ascending, `fill` after."""
    rows = p.shape[0]
    out = torch.full((rows, k), fill, dtype=torch.int32)
    for r in range(rows):
        e = int(end[r])
        if e <= 0:
            continue
        v = p[r, :e].float()
        order = torch.argsort(-v, stable=True)[:k]
        out[r, :order.numel()] = torch.sort(order.to(torch.int32)).values
    return out


def block_scores(p, klen, bs=8):
    """Block max with the last valid block forced to +inf. Returns (scores bf16 [R, nb], valid_blocks int32 [R])."""
    rows, n = p.shape
    nb = (n + bs - 1) // bs
    pad = torch.full((rows, nb * bs), float('-inf'), dtype=torch.bfloat16)
    pad[:, :n] = p
    col = torch.arange(nb * bs).unsqueeze(0)
    pad = pad.masked_fill(col >= klen.long().unsqueeze(1), float('-inf'))
    sc = pad.view(rows, nb, bs).amax(dim=-1)
    vb = ((klen.long() + bs - 1) // bs).clamp_max(nb).to(torch.int32)
    last = (vb.long() - 1).clamp_min(0)
    for r in range(rows):
        if int(vb[r]) > 0:
            sc[r, last[r]] = float('inf')
    return sc, vb


def candidate_blocks(p, klen, n_blocks=2048, bs=8):
    sc, vb = block_scores(p, klen, bs)
    return topk_sorted(sc, vb, n_blocks)


def compact_lens(cand, klen, bs=8):
    start = cand.long() * bs
    n = (klen.long().unsqueeze(1) - start).clamp(0, bs)
    n = n.masked_fill(cand < 0, 0)
    return n.sum(dim=1).to(torch.int32)


def compact_psum(p_full, cand, klen, bs=8):
    """The consumer's p_sum over the candidate tokens only: column slot * bs + o is token cand[slot] * bs + o.
    Columns of -1 slots and tokens >= klen are -inf (the engine never reads them: top-k uses end=compact_lens)."""
    rows, nslot = cand.shape
    out = torch.full((rows, nslot * bs), float('-inf'), dtype=torch.bfloat16)
    for r in range(rows):
        for s in range(nslot):
            b = int(cand[r, s])
            if b < 0:
                continue
            for o in range(bs):
                t = b * bs + o
                if t < int(klen[r]):
                    out[r, s * bs + o] = p_full[r, t]
    return out


def restore(compact_idx, cand, bs=8):
    cap = cand.shape[1] * bs
    valid = (compact_idx >= 0) & (compact_idx < cap)
    safe = compact_idx.clamp(0, cap - 1).long()
    blk = cand.long().gather(1, safe // bs)
    res = blk * bs + safe % bs
    return torch.where(valid & (blk >= 0), res, torch.full_like(res, -1)).to(torch.int32)


def indexer_dense(p, klen, topk=512):
    return topk_sorted(p, klen, topk)


def indexer_candidate(p_cons_full, cand, klen, topk=512, bs=8):
    """Candidate consumer: full-row scores are the reference stand-in for what the kernel computes only over the candidate
    tokens (compact_psum reads just those columns)."""
    cp = compact_psum(p_cons_full, cand, klen, bs)
    cl = compact_lens(cand, klen, bs)
    ci = topk_sorted(cp, cl, topk)
    return restore(ci, cand, bs)
