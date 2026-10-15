"""m4sim from the exlr8 release (--v2-dense) against a PyTorch forward of the same weights: Hugging Face's GlmMoeDsa
model, cut to the first --layers layers, in fp32, with every matrix set to the values the engine's decode path reads.

    G6_CP=0 G6_RESID_DUMP=resid.bin G6_LOGIT_DUMP=logits.bin m4sim --v2 .. --v2-dense REL --layers 0,1,2,3 --gen 12 \
        --temp 0 > out.txt
    python3 engine/ref/release_forward.py --release REL --widths FILE --ref EXLR8_REPO --layers 4 --form rounded \
        --m4 out.txt --resid resid.bin --logits logits.bin

The weights: the release's MXFP8 matrices as stored, kv_b and the embedding in BF16, int4 q_b / o_proj as stored,
and the trellis matrices as --form says: `rounded` takes the MXFP8 / int4 values src/release.rs rounds them to for decode
(engine/loader-rs/tests/release_check.py proves the planner's bytes equal these), `trellis` the release's decoded
values. The experts come from loader/exlr8.py at the width manifest's widths. Each attention and MLP runs on gr8's own
input to it (the residual dump), over every position, and its output is compared with gr8's residual delta. Then the
whole cut model's logits are compared with gr8's. The same in BF16 gives the size of a BF16 model's own rounding.
Routed experts are decoded when a token first picks one, so the cut model needs about 15 GB. Needs torch and
transformers with glm_moe_dsa.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine', 'loader-rs', 'tests')]
from release_check import int4_quantize, mx_quantize  # noqa: E402
from tools.convert import mxfp8  # noqa: E402

H = 6144


def by_rows(f, w, rows=1024):
    """f on row blocks of w (both roundings work per row), to keep the float64 temporaries small"""
    return torch.cat([f(w[a:a + rows]) for a in range(0, w.shape[0], rows)])


def mx_values(w):
    return by_rows(lambda x: torch.from_numpy(mxfp8.dequantize(*mx_quantize(x)).astype(np.float32)), w)


def int4_values(w):
    def one(x):
        q, s = int4_quantize(x)
        n = np.stack([q & 15, q >> 4], -1).reshape(x.shape).astype(np.float32) - 8
        return torch.from_numpy(n * np.repeat(s.astype(np.float32), 32, 1))
    return by_rows(one, w)


def read_steps(path, per_rank=False):
    """G6_RESID_DUMP / G6_LOGIT_DUMP records, rows of every step in order."""
    out = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            s, M = np.frombuffer(h, np.int32)
            if per_rank:
                parts = []
                while sum(p.shape[1] for p in parts) < 154880:
                    vn = int(np.frombuffer(f.read(4), np.int32)[0])
                    parts.append(np.frombuffer(f.read(4 * M * vn), np.float32).reshape(M, vn))
                out.append(np.concatenate(parts, 1))
            else:
                slots, hid = np.frombuffer(f.read(8), np.int32)
                out.append(np.frombuffer(f.read(4 * slots * M * hid), np.float32).reshape(slots, M, hid))
    return out


def tokens_of(m4_out):
    prompt, gen = None, None
    for line in open(m4_out):
        if line.startswith('step 0 '):
            prompt_n = int(line.split()[3])
        if line.startswith('generated '):
            gen = [int(x) for x in line.split()[1:]]
    prompt = [154822, 154824, 154826, 25062][:prompt_n]
    return prompt + gen[:-1]


class LazyExperts(torch.nn.Module):
    """GlmMoeDsaExperts' forward, with each routed expert decoded by loader/exlr8.py when a token first picks it, in
    place of the 256 experts' parameters (39 GB per layer in fp32)."""

    def __init__(self, X, release, l, widths):
        super().__init__()
        self.x, self.w, self.cache = X.ExpertLayer(release, l), widths, {}
        self.dtype = torch.float32

    def _apply(self, fn, recurse=True):
        self.dtype = fn(torch.zeros(1)).dtype          # model.to(dtype) reaches here, and the decoded experts follow it
        self.cache = {}
        return self

    def weights(self, e, dev):
        if e not in self.cache:
            self.cache[e] = [self.x.projection(e, p, self.w[p][e]).to(dev, self.dtype) for p in ('gate', 'up', 'down')]
        return self.cache[e]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in sorted(set(top_k_index.flatten().tolist())):
            tok, k = torch.where(top_k_index == e)
            g, u, d = self.weights(e, hidden_states.device)
            x = hidden_states[tok]
            y = torch.nn.functional.linear(torch.nn.functional.silu(x @ g.T) * (x @ u.T), d)
            out.index_add_(0, tok, (y * top_k_weights[tok, k, None]).to(out.dtype))
        return out


def build(a, X, L, dev):
    from transformers import AutoConfig, GlmMoeDsaForCausalLM
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    cfg = AutoConfig.from_pretrained(a.config)
    cfg.num_hidden_layers = L
    cfg.mlp_layer_types = cfg.mlp_layer_types[:L]
    cfg.indexer_types = cfg.indexer_types[:L]
    cfg.num_nextn_predict_layers = 0
    cfg._attn_implementation = 'eager'
    man = json.load(open(a.widths))
    with torch.device('meta'):
        model = GlmMoeDsaForCausalLM(cfg)
    for l, layer in enumerate(model.model.layers):
        if cfg.mlp_layer_types[l] == 'sparse':
            layer.mlp.experts = LazyExperts(X, a.release, l, man['layers'][str(l)])
    model.to_empty(device=dev)
    model.model.rotary_emb = M.GlmMoeDsaRotaryEmbedding(cfg, device=dev)
    sd = model.state_dict()

    def put(name, t):
        p = sd[name]
        if tuple(p.shape) != tuple(t.shape):
            raise ValueError(f'{name}: {tuple(t.shape)} for {tuple(p.shape)}')
        p.copy_(t.to(p.dtype))

    bf = lambda w: torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(torch.bfloat16).float()
    for l in range(L):
        f = X.SafeFile(os.path.join(a.release, 'dense', f'exlr8-dense-{l:03d}.safetensors'))
        rec = {m[:-7]: f.get(m) for m in f.names() if m.endswith('.record')}
        dec = lambda m: X.decode_dense(rec[m], {s: f.get(f'{m}.{s}') for s in X.DATA[int(rec[m][0])]})
        p = f'model.layers.{l}.'
        owned = {f'{m}.{t}' for m in rec for t in ('record', 'checksum') + X.DATA[int(rec[m][0])]}
        for n in f.names():
            if n not in owned:                                   # norms and biases, e.g. mlp.gate.e_score_correction_bias
                put(p + n, f.get(n))
        for m in rec:
            fmt, w = int(rec[m][0]), dec(m)
            tr = fmt == X.TRELLIS and a.form == 'trellis'
            if m == 'self_attn.kv_b_proj':
                w = bf(w.numpy())
            elif m == 'self_attn.q_b_proj' and fmt == X.TRELLIS and not tr:
                w = mx_values(w.numpy())                          # decode reads the MXFP8 record
            elif m == 'self_attn.o_proj' and fmt == X.TRELLIS and not tr:
                w = int4_values(w.numpy())                        # decode reads the int4 record
            elif m.startswith('mlp.') and m.endswith('_proj') and not tr:
                w = int4_values(w.numpy())                        # the dense MLP's int4 records
            elif m == 'self_attn.indexer.wk':
                w = bf(w.numpy())
            if m == 'mlp.gate':
                put(p + 'mlp.gate.weight', w)
            else:
                put(p + m + '.weight', w)
        if cfg.mlp_layer_types[l] == 'sparse':
            x, wl = X.ExpertLayer(a.release, l), man['layers'][str(l)]
            for n in ('gate', 'up', 'down'):
                put(p + f'mlp.shared_experts.{n}_proj.weight', x.projection('shared', n, wl['shared'][n]))
    f = X.SafeFile(os.path.join(a.release, 'dense', 'exlr8-vocab.safetensors'))
    for m, name, form in (('embed_tokens', 'model.embed_tokens.weight', bf),
                          ('lm_head', 'lm_head.weight', mx_values)):            # decode reads lm_head's MXFP8 record
        rec, t = f.get(f'{m}.record'), {s: f.get(f'{m}.{s}') for s in ('fp8', 'scale')}
        for r0 in range(0, int(rec[1]), 8192):                                    # FP8 per row: row blocks decode alone
            blk = {'fp8': t['fp8'][r0:r0 + 8192], 'scale': t['scale'][r0:r0 + 8192]}
            r = rec.clone()
            r[1] = blk['fp8'].shape[0]
            sd[name][r0:r0 + r[1]].copy_(form(X.decode_dense(r, blk).numpy()))
    put('model.norm.weight', f.get('norm.weight'))
    return model.eval()


def sublayers(model, g6, dev, dtype):
    """Each sublayer's output on gr8's own input to it: the attention of layer i on gr8's residual before it, the MLP
    on gr8's residual after that attention. [(attention, MLP)] per layer, [T, H] fp32."""
    T = g6.shape[1]
    pos = torch.arange(T, device=dev)[None]
    out, topk = [], None
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            x = torch.from_numpy(g6[2 * i].copy()).to(dev, dtype)[None]
            pe = model.model.rotary_emb(x, pos)
            a, _, topk = layer.self_attn(hidden_states=layer.input_layernorm(x), position_embeddings=pe, attention_mask=None,
                                         position_ids=pos, prev_topk_indices=topk)
            x2 = torch.from_numpy(g6[2 * i + 1].copy()).to(dev, dtype)[None]
            m = layer.mlp(layer.post_attention_layernorm(x2))
            out.append((a[0].float().cpu(), m[0].float().cpu()))
    return out


def logits_of(model, ids, dev):
    with torch.no_grad():
        return model(torch.tensor([ids], device=dev), use_cache=False).logits[0].float().cpu()


def rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--config', required=True, help='a directory with the release\'s config.json')
    ap.add_argument('--widths', required=True)
    ap.add_argument('--ref', required=True, help='a kindlingai/exlr8 checkout')
    ap.add_argument('--layers', type=int, required=True, help='m4sim ran layers 0 .. N-1')
    ap.add_argument('--form', choices=('rounded', 'trellis'), required=True)
    ap.add_argument('--m4', required=True)
    ap.add_argument('--resid', required=True)
    ap.add_argument('--logits', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.ref)
    from loader import exlr8 as X
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    ids = tokens_of(a.m4)
    g6 = np.concatenate(read_steps(a.resid), 1)              # [slots][T][H]
    g6l = torch.from_numpy(np.concatenate(read_steps(a.logits, True), 0))
    L = a.layers
    model = build(a, X, L, dev)
    ref, ref_logits = sublayers(model, g6, dev, torch.float32), logits_of(model, ids, dev)
    bias = {i: l.mlp.gate.e_score_correction_bias.clone() for i, l in enumerate(model.model.layers) if hasattr(l.mlp, 'gate')}
    model = model.to(torch.bfloat16)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.float()
    for i, b in bias.items():                      # the routing bias stays fp32, as a BF16 checkpoint ships it
        model.model.layers[i].mlp.gate.e_score_correction_bias = b
    bfr, bf_logits = sublayers(model, g6, dev, torch.bfloat16), logits_of(model, ids, dev)
    print(f'{len(ids)} positions, tokens {ids}')
    print('| sublayer, on gr8\'s input | gr8 vs fp32 reference (rel L2) | BF16 reference vs fp32 |\n|---|---:|---:|')
    for i in range(L):
        for k, what in ((0, 'attention'), (1, 'MLP')):
            d_g6 = torch.from_numpy(g6[2 * i + k + 1] - g6[2 * i + k])
            print(f'| layer {i} {what} | {rel(d_g6, ref[i][k]):.2e} | {rel(bfr[i][k], ref[i][k]):.2e} |')
    top_g6, top_ref, top_bf = g6l.argmax(-1), ref_logits.argmax(-1), bf_logits.argmax(-1)
    print(f'logits: rel L2 gr8 {rel(g6l, ref_logits):.2e}, BF16 {rel(bf_logits, ref_logits):.2e}; argmax agrees with '
          f'fp32 at {(top_g6 == top_ref).sum().item()}/{len(ids)} positions (BF16 {(top_bf == top_ref).sum().item()})')
    lp = lambda x: torch.log_softmax(x.double(), -1)
    kl = lambda p, q: (lp(p).exp() * (lp(p) - lp(q))).sum(-1).mean().item()
    print(f'KL(fp32 || gr8) {kl(ref_logits, g6l):.2e}, KL(fp32 || BF16) {kl(ref_logits, bf_logits):.2e} nats per position')


if __name__ == '__main__':
    main()
