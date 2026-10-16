"""GLM-5.3-Flash reference forward, op by op, in fp32 (or another dtype), for one sequence. The golden source for
the gr8 Flash kernels (W0 in ENGINE-PLAN.md).

It follows transformers 5.19.0 models/glm5_next/modeling_glm5_next.py for layers 0-44 and vLLM's glm5_next MTP (as
exlr8quant requant/calib_forward.py documents it) for layer 45. It is written the way the engine computes, not the way
transformers does: decode is token by token with a fixed KDA state per layer, MLA attention is absorbed (q @ UK
against the 512-d latent, UV after), and the indexer works on a growing cache. test_flash_ref.py proves it equals the
transformers model on a small random config.

Weights are read by checkpoint name relative to `model.language_model.` (`layers.3.self_attn.q_a_proj.weight`,
`embed_tokens.weight`, `norm.weight`) plus `lm_head.weight`; see weights.py. Routed experts are read only when a
token picks one, so a real MoE layer runs in a few GB.

`rec(name, tensor)` sees every op's output (`L03.attn.q_abs`, ...); golden.py stores them.
"""
import json
import math

import torch
import torch.nn.functional as F


class Cfg:
    """The geometry of one Flash model, from the HF config.json (whole file or its text_config)."""

    def __init__(self, d):
        d = d.get('text_config', d)
        self.d = d
        self.H = d['hidden_size']
        self.vocab = d['vocab_size']
        self.n_layers = d['num_hidden_layers']
        self.mtp = self.n_layers if d.get('num_nextn_predict_layers', 0) else None
        self.eps = d['rms_norm_eps']
        lt = d.get('layer_types') or ['linear_attention' if i % 4 != 3 else 'indexed_attention'
                                       for i in range(self.n_layers)]
        self.kda = [t == 'linear_attention' for t in lt]
        mt = d.get('mlp_layer_types') or ['dense'] * min(3, self.n_layers) + ['sparse'] * (self.n_layers - 3)
        self.moe = [t == 'sparse' for t in mt]
        self.hc = d.get('hc_mult', 4)
        self.hc_iters = d.get('hc_sinkhorn_iters', 20)
        self.hc_eps = d.get('hc_eps', 1e-6)
        la = d.get('linear_attn_config') or {}
        self.kh = la.get('num_heads', d.get('linear_num_heads', 64))
        self.kd = la.get('head_dim', d.get('linear_head_dim', 128))
        self.conv = la.get('short_conv_kernel_size', d.get('linear_conv_kernel_dim', 4))
        self.lower = la.get('gate_lower_bound', d.get('linear_lower_bound', -5.0))
        self.heads = d['num_attention_heads']
        self.q_lora = d['q_lora_rank']
        self.kv_lora = d['kv_lora_rank']
        self.nope = d['qk_nope_head_dim']
        self.v = d['v_head_dim']
        assert d.get('qk_rope_head_dim', 0) == 0, 'Flash attention has no rope'
        self.ih = d['index_n_heads']
        self.idim = d['index_head_dim']
        self.itopk = d['index_topk']
        self.kpool = d.get('index_kpool', 4)
        self.tail = d.get('index_kpool_always_select_tail', True)
        self.E = d['n_routed_experts']
        self.topk = d['num_experts_per_tok']
        self.I = d['moe_intermediate_size']
        self.I_dense = d['intermediate_size']
        self.route_scale = d['routed_scaling_factor']
        self.norm_topk = d.get('norm_topk_prob', True)
        assert d.get('n_group', 1) == 1, 'grouped routing not implemented'
        self.limit = d.get('swiglu_limit')

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


def rms(x, w=None, eps=1e-6):
    """RMSNorm computed in fp32, result in x's dtype, then times w"""
    y = x.float()
    y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return y if w is None else y * w


def swiglu(gate, up, limit):
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return F.silu(gate) * up


def l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


class SeqState:
    """Everything one sequence carries between steps: per KDA layer the conv inputs of the last conv-1 tokens and the
    fp32 recurrent state [heads, k, v]; per DSA layer (and MTP) the latent rows and the indexer's key and pool-gate
    rows."""

    def __init__(self):
        self.pos = 0
        self.kda = {}
        self.dsa = {}
        self.mtp_pos = 0


class FlashRef:
    def __init__(self, cfg, weights, dtype=torch.float32, rec=None, dev='cpu'):
        self.c = cfg
        self.w = weights
        self.dt = dtype
        self.rec = rec
        self.dev = dev          # where activations and states live; weights.get must return tensors there
        self.qz = None          # optional rounding hook qz(name, x) -> x at the engine's rounding points (see z)

    def g(self, name):
        return self.w.get(name).to(self.dt)

    def r(self, name, x):
        if self.rec is not None:
            self.rec(name, x)
        return x

    def z(self, name, x):
        """an engine rounding point: block inputs ('in'), block outputs ('out'), the DSA latent row ('kv')"""
        return x if self.qz is None else self.qz(name, x)

    def lin(self, x, name):
        w = self.w.get(name)
        if w.numel() <= 1 << 26:
            return x @ w.to(self.dt).T
        # the vocabulary matrices: convert 16384 rows at a time, not a 2.5 GB fp32 copy (fits an 8 GiB memory cap)
        return torch.cat([x @ w[i:i + 16384].to(self.dt).T for i in range(0, w.shape[0], 16384)], -1)

    def emb(self, tokens):
        return self.w.get('embed_tokens.weight')[torch.as_tensor(tokens, dtype=torch.long, device=self.dev)].to(self.dt)

    # ---- mHC ----
    def mhc(self, streams, p, tag):
        """streams [T, hc, H] -> (block input [T, H], post [T, hc], comb [T, hc, hc]); p = 'layers.L.hc_attn_'"""
        c, n = self.c, self.c.hc
        flat = rms(streams.reshape(streams.shape[0], -1).float(), eps=c.eps)
        mix = flat @ self.w.get(p + 'fn').float().T
        pre_w, post_w, comb_w = mix.split([n, n, n * n], -1)
        base = self.w.get(p + 'base').float()
        pre_b, post_b, comb_b = base.split([n, n, n * n])
        s = self.w.get(p + 'scale').float()
        pre = torch.sigmoid(pre_w * s[0] + pre_b) + c.hc_eps
        post = 2 * torch.sigmoid(post_w * s[1] + post_b)
        comb = torch.softmax((comb_w * s[2]).view(-1, n, n) + comb_b.view(n, n), -1) + c.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        for _ in range(c.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + c.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        x = (pre.unsqueeze(-1) * streams.float()).sum(1).to(self.dt)
        self.r(tag + '.pre', pre)
        self.r(tag + '.post', post)
        self.r(tag + '.comb', comb)
        self.r(tag + '.x', x)
        return x, post.to(self.dt), comb.to(self.dt)

    @staticmethod
    def mhc_out(block_out, post, comb, residual):
        """stream i = post_i * block_out + sum_j comb[j, i] * residual_j"""
        return post.unsqueeze(-1) * block_out.unsqueeze(1) + torch.einsum('tji,tjh->tih', comb, residual)

    # ---- KDA ----
    def kda_inputs(self, x, L, st):
        c, p = self.c, f'layers.{L}.self_attn.'
        T, D = x.shape[0], c.kh * c.kd
        mixed = torch.cat([self.lin(x, p + 'q_proj.weight'), self.lin(x, p + 'k_proj.weight'),
                           self.lin(x, p + 'v_proj.weight')], -1)                       # [T, 3D]
        w = torch.cat([self.g(p + f'{n}_conv1d.weight').reshape(D, c.conv) for n in 'qkv'])  # [3D, K]
        s = st.kda.get(L)
        prev = s['conv'] if s else torch.zeros(c.conv - 1, 3 * D, dtype=self.dt, device=self.dev)
        full = torch.cat([prev, mixed])                                                  # [K-1+T, 3D]
        conv = sum(full[j:j + T] * w[:, j] for j in range(c.conv))
        conv = F.silu(conv)
        q, k, v = conv.split(D, -1)
        q = q.reshape(T, c.kh, c.kd).float()
        k = k.reshape(T, c.kh, c.kd).float()
        v = v.reshape(T, c.kh, c.kd).float()
        q = l2norm(q) * c.kd ** -0.5
        k = l2norm(k)
        fg = self.lin(self.lin(x, p + 'f_a_proj.weight'), p + 'f_b_proj.weight').float()
        fg = (fg + self.w.get(p + 'dt_bias').float()).view(T, c.kh, c.kd)
        A = torch.exp(self.w.get(p + 'A_log').float()).view(1, c.kh, 1)
        gk = c.lower * torch.sigmoid(A * fg)                                             # log decay <= 0
        beta = torch.sigmoid(self.lin(x, p + 'b_proj.weight').float())                  # [T, heads]
        tag = f'L{L:02d}.kda'
        for n, t in (('mixed', mixed), ('q', q), ('k', k), ('v', v), ('g', gk), ('beta', beta)):
            self.r(f'{tag}.{n}', t)
        return full[-(c.conv - 1):].clone(), q, k, v, gk, beta

    @staticmethod
    def kda_recurrent(q, k, v, g, beta, S):
        """q,k,v,g [T, h, d] fp32, beta [T, h], S [h, dk, dv] -> o [T, h, dv], S"""
        out = []
        for t in range(q.shape[0]):
            S = S * g[t].exp().unsqueeze(-1)
            delta = (v[t] - torch.einsum('hkv,hk->hv', S, k[t])) * beta[t].unsqueeze(-1)
            S = S + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
            out.append(torch.einsum('hkv,hk->hv', S, q[t]))
        return torch.stack(out), S

    @staticmethod
    def kda_chunk(q, k, v, g, beta, S, chunk=64):
        """The chunked form transformers' chunk_kimi_delta_attention uses (q already scaled). Same result as
        kda_recurrent to fp32 rounding."""
        T = q.shape[0]
        pad = (chunk - T % chunk) % chunk
        q, k, v, g = (F.pad(t.transpose(0, 1), (0, 0, 0, pad)) for t in (q, k, v, g))      # [h, T', d]
        beta = F.pad(beta.transpose(0, 1), (0, pad))
        h, d = q.shape[0], q.shape[-1]
        n = q.shape[1] // chunk
        q, k, v, g = (t.reshape(h, n, chunk, -1) for t in (q, k, v, g))
        beta = beta.reshape(h, n, chunk)
        kb, vb = k * beta.unsqueeze(-1), v * beta.unsqueeze(-1)
        g = g.cumsum(-2)
        incl = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), 0)
        strict = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), 1)
        o = torch.zeros_like(v)
        for i in range(n):                      # one chunk at a time: the pairwise decay is [h, C, C, d] per chunk
            gi, ki, kbi = g[:, i], k[:, i], kb[:, i]
            decay = (gi.unsqueeze(-2) - gi.unsqueeze(-3)).masked_fill(strict[..., None], float('-inf')).exp()
            A = -(kbi.unsqueeze(-2) * ki.unsqueeze(-3) * decay).sum(-1).masked_fill(incl, 0)
            for r in range(1, chunk):
                A[..., r, :r] = A[..., r, :r].clone() + (A[..., r, :r].clone().unsqueeze(-1) * A[..., :r, :r].clone()).sum(-2)
            A = A + torch.eye(chunk)
            u = A @ vb[:, i]
            w = A @ (kbi * gi.exp())
            inter = (q[:, i] * gi.exp()) @ S
            intra = (q[:, i].unsqueeze(-2) * ki.unsqueeze(-3) * decay).sum(-1).masked_fill(strict, 0)
            vn = u - w @ S
            o[:, i] = inter + intra @ vn
            S = S * gi[:, -1].exp().unsqueeze(-1) + (ki * (gi[:, -1:] - gi).exp()).transpose(-1, -2) @ vn
        return o.reshape(h, -1, d)[:, :T].transpose(0, 1), S

    def kda(self, x, L, st, chunked=False):
        c, p = self.c, f'layers.{L}.self_attn.'
        T = x.shape[0]
        conv_state, q, k, v, gk, beta = self.kda_inputs(x, L, st)
        s = st.kda.get(L)
        S = s['S'] if s else torch.zeros(c.kh, c.kd, c.kd, device=self.dev)
        o, S = (self.kda_chunk if chunked else self.kda_recurrent)(q, k, v, gk, beta, S)
        st.kda[L] = {'conv': conv_state, 'S': S}
        gate = self.lin(self.lin(x, p + 'g_a_proj.weight'), p + 'g_b_proj.weight').float().view(T, c.kh, c.kd)
        o_n = rms(o, self.w.get(p + 'o_norm.weight').float(), c.eps) * torch.sigmoid(gate)
        tag = f'L{L:02d}.kda'
        self.r(tag + '.o', o)
        self.r(tag + '.S', S)
        self.r(tag + '.o_gated', o_n)
        return self.r(tag + '.out', self.lin(o_n.reshape(T, -1).to(self.dt), p + 'o_proj.weight'))

    # ---- DSA: indexer + NoPE MLA ----
    def indexer(self, x, qr, p, cache, pos0, tag):
        """Selected keys per query: bool [T, N] over every cached token."""
        c = self.c
        T = x.shape[0]
        iq = self.lin(qr, p + 'wq_b.weight').view(T, c.ih, c.idim).float()
        ik = F.layer_norm(self.lin(x, p + 'wk.weight').float(), (c.idim,), self.w.get(p + 'k_norm.weight').float(),
                          self.w.get(p + 'k_norm.bias').float(), 1e-6)
        ig = (x @ self.g(p + 'index_kpool_compress_gate').T).float()
        cache['ik'] = torch.cat([cache['ik'], ik]) if 'ik' in cache else ik
        cache['ig'] = torch.cat([cache['ig'], ig]) if 'ig' in cache else ig
        N, P = cache['ik'].shape[0], c.kpool
        n_pools = (N + P - 1) // P
        pad = n_pools * P - N
        keys = F.pad(cache['ik'], (0, 0, 0, pad)).view(n_pools, P, c.idim)
        logits = F.pad(cache['ig'], (0, 0, 0, pad)).view(n_pools, P, c.idim) + self.w.get(p + 'index_kpool_compress_ape').float()
        exists = (torch.arange(n_pools * P, device=self.dev) < N).view(n_pools, P)
        logits = logits.masked_fill(~exists[..., None], float('-inf'))
        prob = torch.nan_to_num(logits.softmax(1))
        pool_k = (prob * keys).sum(1)                                                   # [pools, idim]
        wts = self.lin(x, p + 'weights_proj.weight').float() * c.ih ** -0.5
        score = torch.cat([torch.einsum('th,thp->tp', wts[a:a + 1024],               # 1024 queries at a time:
                                        F.relu(torch.einsum('thd,pd->thp', iq[a:a + 1024], pool_k) * c.idim ** -0.5))
                           for a in range(0, T, 1024)])                             # [T, heads, pools] is GBs at 9k
        qpos = pos0 + torch.arange(T, device=self.dev)
        pool_end = torch.arange(n_pools, device=self.dev) * P + P - 1
        valid = (pool_end[None, :] <= qpos[:, None]) & exists.all(1)[None, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        k_sel = min(c.itopk // P, n_pools)
        # ties (e.g. pools where every head's relu is 0) go to the lower pool index; transformers' topk leaves the
        # order of ties unspecified
        sel = score.sort(dim=-1, descending=True, stable=True).indices[:, :k_sel]
        pools = torch.zeros(T, n_pools, dtype=torch.bool, device=self.dev)
        pools.scatter_(1, sel, valid.gather(1, sel))                  # each row's pools are distinct
        mask = pools.repeat_interleave(P, 1)[:, :N]
        if c.tail:                                                    # the open pool: keys cnt - cnt % P .. cnt - 1
            cnt = (qpos + 1)[:, None]
            j = torch.arange(N, device=self.dev)[None, :]
            mask |= (j >= cnt - cnt % P) & (j < cnt)
        self.r(tag + '.idx_q', iq)
        self.r(tag + '.idx_k', ik)
        self.r(tag + '.idx_gate', ig)
        self.r(tag + '.pool_k', pool_k)
        self.r(tag + '.idx_score', score)
        self.r(tag + '.idx_mask', mask)
        return mask

    def dsa(self, x, L, st, key=None):
        c, p = self.c, f'layers.{L}.self_attn.'
        key = L if key is None else key
        tag = f'L{L:02d}.attn'
        T = x.shape[0]
        cache = st.dsa.setdefault(key, {})
        pos0 = cache['lat'].shape[0] if 'lat' in cache else 0
        qr = rms(self.lin(x, p + 'q_a_proj.weight'), self.g(p + 'q_a_layernorm.weight'), c.eps)
        q = self.lin(qr, p + 'q_b_proj.weight').view(T, c.heads, c.nope).float()
        lat = rms(self.lin(x, p + 'kv_a_proj_with_mqa.weight'), self.g(p + 'kv_a_layernorm.weight'), c.eps).float()
        lat = self.z('kv', lat)
        cache['lat'] = torch.cat([cache['lat'], lat]) if 'lat' in cache else lat
        kvb = self.w.get(p + 'kv_b_proj.weight').float().view(c.heads, c.nope + c.v, c.kv_lora)
        uk, uv = kvb[:, :c.nope], kvb[:, c.nope:]
        q_abs = torch.einsum('thn,hnl->thl', q, uk)
        mask = self.indexer(x, qr, p + 'indexer.', cache, pos0, tag)
        ctx = torch.empty_like(q_abs)
        for t0 in range(0, T, 512):                     # 512 queries at a time: long prompts fit an 8 GiB memory cap
            t1 = min(T, t0 + 512)
            s = torch.einsum('thl,nl->thn', q_abs[t0:t1], cache['lat']) * c.nope ** -0.5
            prob = s.masked_fill(~mask[t0:t1, None, :], float('-inf')).softmax(-1)
            ctx[t0:t1] = torch.einsum('thn,nl->thl', prob, cache['lat'])
        o = torch.einsum('thl,hvl->thv', ctx, uv)
        self.r(tag + '.q_resid', qr)
        self.r(tag + '.q', q)
        self.r(tag + '.latent', lat)
        self.r(tag + '.q_abs', q_abs)
        self.r(tag + '.ctx', ctx)
        self.r(tag + '.o', o)
        return self.r(tag + '.out', self.lin(o.reshape(T, -1).to(self.dt), p + 'o_proj.weight'))

    # ---- MLP / MoE ----
    def mlp(self, x, p, tag):
        y = self.lin(swiglu(self.lin(x, p + 'gate_proj.weight'), self.lin(x, p + 'up_proj.weight'), self.c.limit),
                     p + 'down_proj.weight')
        return self.r(tag, y)

    def moe(self, x, L):
        c, p = self.c, f'layers.{L}.mlp.'
        tag = f'L{L:02d}.moe'
        logits = x.float() @ self.w.get(p + 'gate.weight').float().T
        scores = logits.sigmoid()
        choice = scores + self.w.get(p + 'gate.e_score_correction_bias').float()
        idx = choice.topk(c.topk, -1).indices
        wts = scores.gather(1, idx)
        if c.norm_topk:
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        wts = wts * c.route_scale
        self.r(tag + '.router_logits', logits)
        self.r(tag + '.topk_idx', idx)
        self.r(tag + '.topk_w', wts)
        y = torch.zeros_like(x)
        for e in idx.unique().tolist():
            rows, slot = (idx == e).nonzero(as_tuple=True)
            gate, up, down = (t.to(self.dt) for t in self.w.expert(L, e))
            h = swiglu(x[rows] @ gate.T, x[rows] @ up.T, c.limit) @ down.T
            y.index_add_(0, rows, h * wts[rows, slot, None].to(self.dt))
        self.r(tag + '.routed', y)
        sh = self.mlp(x, p + 'shared_experts.', tag + '.shared')
        return self.r(tag + '.out', y + sh)

    # ---- layers ----
    def attn_block(self, streams, L, st, chunked=False):
        """streams [T, hc, H] -> streams after layer L's attention and its mHC (the engine's consume 2L)"""
        c, p = self.c, f'layers.{L}.'
        tag = f'L{L:02d}'
        x, post, comb = self.mhc(streams, p + 'hc_attn_', tag + '.hc_attn')
        h = self.r(tag + '.attn_in', self.z('in', rms(x, self.g(p + 'input_layernorm.weight'), c.eps)))
        a = self.z('out', self.kda(h, L, st, chunked) if c.kda[L] else self.dsa(h, L, st))
        return self.r(tag + '.mid', self.mhc_out(a, post, comb, streams))

    def mlp_block(self, streams, L):
        """streams after layer L's attention -> after its MLP / MoE and its mHC (consume 2L + 1)"""
        c, p = self.c, f'layers.{L}.'
        tag = f'L{L:02d}'
        x, post, comb = self.mhc(streams, p + 'hc_ffn_', tag + '.hc_ffn')
        h = self.r(tag + '.mlp_in', self.z('in', rms(x, self.g(p + 'post_attention_layernorm.weight'), c.eps)))
        m = self.z('out', self.moe(h, L) if c.moe[L] else self.mlp(h, p + 'mlp.', tag + '.mlp.out'))
        return self.r(tag + '.out', self.mhc_out(m, post, comb, streams))

    def layer(self, streams, L, st, chunked=False):
        self.r(f'L{L:02d}.in', streams)
        return self.mlp_block(self.attn_block(streams, L, st, chunked), L)

    def final_hidden(self, streams):
        """final streams [T, hc, H] -> the final-normed hidden [T, H] (the lm_head input)"""
        return rms(streams.mean(1), self.g('norm.weight'), self.c.eps)

    def head(self, streams):
        """final streams [T, hc, H] -> (final-normed hidden [T, H], logits [T, V] fp32)"""
        hidden = self.r('final.hidden', self.final_hidden(streams))
        return hidden, self.r('final.logits', self.lin(hidden, 'lm_head.weight').float())

    def forward(self, tokens, st, layers=None, chunked=False, head=True):
        """Run len(tokens) new tokens of one sequence. layers: the decoder layers to run (default all, in order);
        a cut model still applies the final norm and head to its last layer's streams.
        Returns {'hidden': final-normed hidden [T, H], 'logits': [T, V] or None}."""
        c = self.c
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.dev)
        x = self.emb(tokens)
        streams = x.unsqueeze(1).expand(-1, c.hc, -1).contiguous()
        for L in (range(c.n_layers) if layers is None else layers):
            streams = self.layer(streams, L, st, chunked)
        st.pos += len(tokens)
        if head:
            hidden, logits = self.head(streams)
        else:
            hidden, logits = self.r('final.hidden', rms(streams.mean(1), self.g('norm.weight'), c.eps)), None
        return {'hidden': hidden, 'logits': logits, 'streams': streams}

    def mtp(self, prev_hidden, next_tokens, st):
        """vLLM's glm5_next MTP layer. Row i: prev_hidden = the main model's final-normed hidden at position
        st.mtp_pos + i, next_tokens[i] = the token at the next position; returns logits for the position after it.
        The embedding is zeroed at position 0 (vLLM's MTP input rule). Plain pre-norm block, no mHC, own DSA cache."""
        c, L = self.c, self.c.mtp
        p = f'layers.{L}.'
        T = prev_hidden.shape[0]
        e = self.emb(next_tokens)
        pos = st.mtp_pos + torch.arange(T, device=self.dev)
        e = e.masked_fill((pos == 0)[:, None], 0)
        cat = torch.cat([rms(e.float(), self.w.get(p + 'enorm.weight').float(), c.eps),
                         rms(prev_hidden.float(), self.w.get(p + 'hnorm.weight').float(), c.eps)], -1).to(self.dt)
        h = self.r('MTP.eh', self.lin(cat, p + 'eh_proj.weight'))
        h = h + self.dsa(self.r('MTP.attn_in', rms(h, self.g(p + 'input_layernorm.weight'), c.eps)), L, st, key='mtp')
        h = self.r('MTP.mid', h)
        h = h + self.moe(self.r('MTP.mlp_in', rms(h, self.g(p + 'post_attention_layernorm.weight'), c.eps)), L)
        h = self.r('MTP.out', h)
        st.mtp_pos += T
        return self.r('MTP.logits', self.lin(rms(h, self.g(p + 'shared_head.norm.weight'), c.eps),
                                             'lm_head.weight').float())
