"""NumPy reference of the DFlash2 drafter (engine/runtime/dflash.cuh), fp32 math, for the drafter step test.

Reads the drafter's own safetensors (BF16) or, with quant='mxfp8', the same weights rounded the way the converter
stores them (tools/convert/dflash2.py), so a test can separate kernel error from quantization error.

The drafter, as its config and the public DFlash2 description define it:
  context  ctx = hidden_norm(fc(concat of the target residual after the target layers)); per drafter layer j the
           context key k_norm_j(k_proj_j(ctx)) with RoPE at the row's position, and the value v_proj_j(ctx)
  block    8 query rows at positions q0..q0+7: the bonus token, then 7 mask tokens (target embedding)
  layer j  h += conv_finish(attn(conv_prepare(in_norm(h)))); h += conv_finish(mlp(conv_prepare(post_norm(h))))
           conv: per row r of the block, two taps over rows r and r-1 (r = 0 has no r-1), per channel
           (base[tap][c] + delta_r[tap][c // group]); delta from kernel_projection of the prepare input (side 0 for the
           prepare conv, side 1 for the finish conv)
  attn     64 q heads over 8 KV heads (GQA), q_norm / k_norm per head, NeoX RoPE; a query at position q sees every key k
           of the block and every context key with q - k < window; not causal inside the block
  head     norm, the target's lm_head; row r (1..7) predicts position q0 + r
  selector top-16 per row, then a walk: score(c) = logit(c) + <pred[prev] * hproj(h_r), succ[c]>, prev starts at the bonus
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.convert import mxfp8  # noqa: E402
from tools.convert.st import ST  # noqa: E402

HID, HD, NH, NKV, INTER = 6144, 128, 64, 8, 12288


def bf(u16):
    return mxfp8.bf16_to_f32(u16)


def mx_round(w, chunk=128):
    """BF16 bits [O, K] -> fp32 of the MXFP8 codes the converter stores, in row chunks (bounded memory)."""
    out = np.empty(w.shape, np.float32)
    for a in range(0, w.shape[0], chunk):
        c, s = mxfp8.quantize(w[a:a + chunk])
        out[a:a + chunk] = mxfp8.dequantize(c, s)
    return out


def rms(x, w, eps):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope_neox(x, pos, theta):
    """x [..., n, 128] (rows on the second-to-last axis broadcast against pos [n])."""
    inv = theta ** (-np.arange(0, HD, 2, dtype=np.float64) / HD)            # [64]
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]               # [n, 64]
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


class Drafter:
    def __init__(self, src, quant=None):
        self.cfg = json.load(open(os.path.join(src, 'config.json')))
        self.d = self.cfg['dflash_config']
        self.st = ST(os.path.join(src, 'model.safetensors'))
        self.quant = quant
        self.eps = self.cfg['rms_norm_eps']
        self.theta = self.cfg['rope_parameters']['rope_theta']
        self.win = self.cfg.get('sliding_window') or 1 << 30
        self.G = self.d['conv_group_size']
        self.L = self.cfg['num_hidden_layers']
        self._w = {}

    def w(self, name):
        """fp32 weight; matrices go through MXFP8 when quant == 'mxfp8' (vectors and codebooks stay BF16). Only vectors
        are kept: a matrix is read again at each use, so the whole drafter never sits in memory as fp32."""
        if name in self._w:
            return self._w[name]
        a = self.st.get(name)
        if self.quant == 'mxfp8' and a.ndim == 2 and 'codebook' not in name:
            v = mx_round(a)
        else:
            v = bf(a)
        if a.ndim == 1 or a.size <= 4 * HID:
            self._w[name] = v
        return v

    def lin(self, x, name):
        return x.astype(np.float32) @ self.w(name).T

    # ---------------------------------------------------------------- context
    def context_kv(self, feat, pos):
        """feat [n, 6 * 6144] (target residual rows), pos [n] -> per layer (K [n, 8, 128], V [n, 8, 128]), and ctx."""
        ctx = rms(self.lin(feat, 'fc.weight'), self.w('hidden_norm.weight'), self.eps)
        out = []
        for j in range(self.L):
            p = f'layers.{j}.self_attn.'
            k = self.lin(ctx, p + 'k_proj.weight').reshape(-1, NKV, HD)
            v = self.lin(ctx, p + 'v_proj.weight').reshape(-1, NKV, HD)
            k = rms(k, self.w(p + 'k_norm.weight'), self.eps)
            k = rope_neox(k.transpose(1, 0, 2), pos, self.theta).transpose(1, 0, 2)
            out.append((k, v))
        return out, ctx

    # ---------------------------------------------------------------- block
    def conv(self, x, delta, base):
        """x [8, 6144], delta [8, 2 taps, 384 groups], base [2 taps, 6144] -> two-tap grouped conv over the block rows."""
        g = np.repeat(delta, self.G, axis=-1)                               # [8, 2, 6144]
        y = (base[0] + g[:, 0]) * x
        y[1:] += (base[1] + g[1:, 1]) * x[:-1]
        return y

    def conv_prepare(self, x, j, which):
        p = f'layers.{j}.{which}.'
        dl = self.lin(x, p + 'kernel_projection.weight').reshape(-1, 2, 2, HID // self.G)
        base = self.w(p + 'base_kernel')
        return self.conv(x, dl[:, 0], base[0]), (dl[:, 1], base[1])

    def attn(self, x, j, qpos, cache):
        """cache: list of (positions [n], K [n, 8, 128], V [n, 8, 128]) for layer j, block keys appended here."""
        p = f'layers.{j}.self_attn.'
        n = x.shape[0]
        q = rms(self.lin(x, p + 'q_proj.weight').reshape(n, NH, HD), self.w(p + 'q_norm.weight'), self.eps)
        k = rms(self.lin(x, p + 'k_proj.weight').reshape(n, NKV, HD), self.w(p + 'k_norm.weight'), self.eps)
        v = self.lin(x, p + 'v_proj.weight').reshape(n, NKV, HD)
        q = rope_neox(q.transpose(1, 0, 2), qpos, self.theta).transpose(1, 0, 2)
        k = rope_neox(k.transpose(1, 0, 2), qpos, self.theta).transpose(1, 0, 2)
        cpos, ck, cv = cache
        keep = cpos < qpos[0]                                               # context keys before the block
        kp = np.concatenate([cpos[keep], qpos])
        K = np.concatenate([ck[keep], k])
        V = np.concatenate([cv[keep], v])
        out = np.zeros((n, NH, HD), np.float32)
        for r in range(n):
            vis = (qpos[r] - kp < self.win) & (kp - qpos[r] < self.win)
            for h in range(NH):
                s = (K[vis, h // (NH // NKV)] @ q[r, h]) / np.sqrt(HD)
                e = np.exp(s - s.max())
                out[r, h] = (e / e.sum()) @ V[vis, h // (NH // NKV)]
        return self.lin(out.reshape(n, NH * HD), p + 'o_proj.weight'), (k, v)

    def mlp(self, x, j):
        p = f'layers.{j}.mlp.'
        g, u = self.lin(x, p + 'gate_proj.weight'), self.lin(x, p + 'up_proj.weight')
        return self.lin(g / (1 + np.exp(-g)) * u, p + 'down_proj.weight')

    def block(self, emb, qpos, caches, trace=None):
        """emb [8, 6144] target embedding rows of [bonus, mask x 7]; caches[j] = (pos, K, V) -> normed hidden [8, 6144]."""
        h = emb.astype(np.float32).copy()
        for j in range(self.L):
            x = rms(h, self.w(f'layers.{j}.input_layernorm.weight'), self.eps)
            xc, fin = self.conv_prepare(x, j, 'attention_conv')
            a, _ = self.attn(xc, j, qpos, caches[j])
            h = h + self.conv(a, *fin)
            x2 = rms(h, self.w(f'layers.{j}.post_attention_layernorm.weight'), self.eps)
            xc2, fin2 = self.conv_prepare(x2, j, 'mlp_conv')
            h = h + self.conv(self.mlp(xc2, j), *fin2)
            if trace is not None:
                trace.append(h.copy())
        return rms(h, self.w('norm.weight'), self.eps)

    # ---------------------------------------------------------------- head
    def top(self, hidden, lm_head, k=16, chunk=16384):
        """hidden [n, 6144] -> top-k (logit desc, id asc) of hidden @ lm_head^T: (ids [n, k], logits [n, k]).
        lm_head: an ST holding lm_head.weight (BF16); MXFP8-rounded when quant == 'mxfp8', as the engine's records."""
        V = lm_head.shape('lm_head.weight')[0]
        best_l = np.full((len(hidden), 0), -np.inf, np.float32)
        best_i = np.zeros((len(hidden), 0), np.int64)
        for a in range(0, V, chunk):
            w = lm_head.get('lm_head.weight', rows=(a, min(V, a + chunk)))
            w = mx_round(w) if self.quant == 'mxfp8' else bf(w)
            lg = hidden.astype(np.float32) @ w.T
            ids = np.arange(a, a + lg.shape[1])[None, :].repeat(len(hidden), 0)
            L, I = np.concatenate([best_l, lg], 1), np.concatenate([best_i, ids], 1)
            o = np.lexsort((I, -L), axis=-1)[:, :k]
            best_l, best_i = np.take_along_axis(L, o, -1), np.take_along_axis(I, o, -1)
        return best_i, best_l

    def select(self, hidden, cand, logit, bonus):
        """hidden [7, 6144] (rows 1..7), the top-k ids / logits per row -> the walk's tokens [7] and scores [7]."""
        hp = self.lin(hidden, 'candidate_selector.hidden_projection.weight')
        pred, succ = self.st.get('candidate_selector.predecessor_codebook'), self.st.get('candidate_selector.successor_codebook')
        prev, path, scores = bonus, [], []
        for s in range(len(hidden)):
            sc = logit[s] + (bf(succ[cand[s]]) * (bf(pred[prev]) * hp[s])).sum(-1)
            j = int(np.argmax(sc))
            prev = int(cand[s][j])
            path.append(prev)
            scores.append(float(sc[j]))
        return np.array(path), np.array(scores)
