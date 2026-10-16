"""flash_ref.FlashRef against transformers' Glm5NextTextModel on a small random config, in fp32, layer by layer.

    python3 engine/ref/flash/test_flash_ref.py      # needs transformers >= 5.19 (glm5_next); CPU, ~1 min

Checks, each a rel-L2 bound:
1. every layer from transformers' input to it == transformers' output (all positions in one call);
2. the whole model in one call, recurrent and chunked KDA == transformers' last layer and final hidden;
3. every layer decoded one token at a time (KDA state, growing caches) from transformers' input == its output.
   (A whole-model decode chain is not a test: a 1e-5 difference flips near-tied router choices, see ENGINE-PLAN.md);
4. KDA chunk == recurrent on a longer run with a nonzero start state;
5. the indexer really drops keys at this length (the test is not just dense attention);
6. MTP: one call over all rows == one row at a time.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_ref import Cfg, FlashRef, SeqState  # noqa: E402
from weights import DictWeights  # noqa: E402

TOL = 1e-5      # one layer from identical inputs
CHAIN = 1e-4    # 8 layers chained: fp32 differences grow ~1.5x per layer

TINY = dict(hidden_size=64, num_hidden_layers=8, vocab_size=128, intermediate_size=48, moe_intermediate_size=16,
            n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, routed_scaling_factor=2.5,
            num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=24, qk_nope_head_dim=16,
            qk_rope_head_dim=0, qk_head_dim=16, v_head_dim=16, index_n_heads=16, index_head_dim=16, index_topk=8,
            index_kpool=4, index_kpool_always_select_tail=True, linear_num_heads=4, linear_head_dim=16,
            linear_conv_kernel_dim=4, linear_lower_bound=-5.0, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
            rms_norm_eps=1e-5, swiglu_limit=10.0, n_group=1, topk_group=1, norm_topk_prob=True,
            num_nextn_predict_layers=1, pad_token_id=0, head_dim=0)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def hf_model():
    from transformers.models.glm5_next import modeling_glm5_next as M
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    cfg = Glm5NextTextConfig(**TINY)
    cfg._attn_implementation = 'eager'
    torch.manual_seed(0)
    m = M.Glm5NextTextModel(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith('norm.weight') or n.endswith('layernorm.weight'):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif 'conv1d' in n:
                p.copy_(0.5 * torch.randn_like(p))
            elif n.endswith('A_log'):
                p.copy_(torch.randn_like(p))
            elif n.endswith('.scale'):
                p.copy_(1 + 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else p.shape[-1] ** -0.5))
        for mod in m.modules():
            if hasattr(mod, 'e_score_correction_bias'):
                mod.e_score_correction_bias.copy_(0.1 * torch.randn_like(mod.e_score_correction_bias))
    return cfg, m


def checkpoint_dict(cfg, m):
    """transformers module state -> checkpoint names (the reverse of transformers' glm5_next conversion)."""
    d = {}
    for k, v in m.state_dict().items():
        k2 = k
        for a in ('attn', 'ffn'):
            for f in ('fn', 'base', 'scale'):
                k2 = k2.replace(f'{a}_hc.{f}', f'hc_{a}_{f}')
        k2 = k2.replace('self_attn.forget_gate.', 'self_attn.')
        if k2.endswith('self_attn.conv1d.weight'):
            for n, part in zip('qkv', v.chunk(3)):
                d[k2.replace('conv1d', f'{n}_conv1d')] = part.clone()
        elif k2.endswith('mlp.experts.gate_up_proj'):
            base = k2[:-len('gate_up_proj')]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2)
                d[f'{base}{e}.gate_proj.weight'] = g.clone()
                d[f'{base}{e}.up_proj.weight'] = u.clone()
        elif k2.endswith('mlp.experts.down_proj'):
            base = k2[:-len('down_proj')]
            for e in range(v.shape[0]):
                d[f'{base}{e}.down_proj.weight'] = v[e].clone()
        else:
            d[k2] = v.clone()
    torch.manual_seed(1)
    H, V = cfg.hidden_size, cfg.vocab_size
    d['lm_head.weight'] = torch.randn(V, H) * H ** -0.5
    # an MTP layer (transformers has none): a DSA + MoE layer's tensors under index n_layers, no mHC
    L, src = cfg.num_hidden_layers, f'layers.{cfg.num_hidden_layers - 1}.'
    for k in list(d):
        if k.startswith(src) and '.hc_' not in k:
            d[f'layers.{L}.' + k[len(src):]] = (d[k] + 0.01 * torch.randn_like(d[k])) if d[k].is_floating_point() else d[k]
    d[f'layers.{L}.eh_proj.weight'] = torch.randn(H, 2 * H) * (2 * H) ** -0.5
    for n in ('enorm', 'hnorm', 'shared_head.norm'):
        d[f'layers.{L}.{n}.weight'] = 1 + 0.1 * torch.randn(H)
    return d


def main():
    cfg, m = hf_model()
    c = Cfg(dict(cfg.to_dict(), num_nextn_predict_layers=1))
    assert c.kda == [True, True, True, False] * 2 and c.moe == [False] * 3 + [True] * 5
    ref = FlashRef(c, DictWeights(checkpoint_dict(cfg, m)))
    T = 37
    torch.manual_seed(2)
    tok = torch.randint(0, cfg.vocab_size, (T,))
    fails = []

    def check(name, err, tol=TOL):
        ok = err <= tol
        print(f"{'ok  ' if ok else 'FAIL'} {name}: rel {err:.2e} (<= {tol:.0e})")
        if not ok:
            fails.append(name)

    # transformers, layer by layer
    with torch.no_grad():
        hs = m.embed_tokens(tok[None]).unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
        mask = torch.ones(1, T, dtype=torch.bool)
        hf_layers, hf_in, topk = [], [], None
        for i, layer in enumerate(m.layers):
            hf_in.append(hs[0])
            hs, topk = layer(hs, attention_mask=mask, position_ids=torch.arange(T)[None], past_key_values=None,
                             prev_topk_indices=topk)
            hf_layers.append(hs[0])
        hf_final = m.norm(m.hc_head(hs))[0]

    outs = {}
    ref.rec = lambda n, t: outs.__setitem__(n, t) if n.endswith('.out') and n.count('.') == 1 else None
    for i in range(cfg.num_hidden_layers):
        o = ref.layer(hf_in[i], i, SeqState())
        check(f'layer {i} from the same input', rel(o, hf_layers[i]))
    for chunked in (False, True):
        outs.clear()
        r = ref.forward(tok, SeqState(), chunked=chunked)
        last = cfg.num_hidden_layers - 1
        check(f"whole model{' chunked KDA' if chunked else ''}, layer {last}", rel(outs[f'L{last:02d}.out'],
                                                                                 hf_layers[last]), CHAIN)
        check(f"whole model{' chunked KDA' if chunked else ''}, final hidden", rel(r['hidden'], hf_final), CHAIN)

    ref.rec = None
    for i in range(cfg.num_hidden_layers):
        st = SeqState()
        dec = torch.cat([ref.layer(hf_in[i][t:t + 1], i, st) for t in range(T)])
        check(f'layer {i} decoded one token at a time from the same input', rel(dec, hf_layers[i]))

    torch.manual_seed(3)
    h, d, n = 3, 16, 150
    q, k = (torch.nn.functional.normalize(torch.randn(n, h, d), dim=-1) for _ in range(2))
    v = torch.randn(n, h, d)
    g = -5 * torch.sigmoid(torch.randn(n, h, d))
    beta = torch.sigmoid(torch.randn(n, h))
    S0 = torch.randn(h, d, d) * 0.1
    o_r, S_r = FlashRef.kda_recurrent(q * d ** -0.5, k, v, g, beta, S0)
    o_c, S_c = FlashRef.kda_chunk(q * d ** -0.5, k, v, g, beta, S0)
    check('KDA chunk vs recurrent output (T=150, start state)', rel(o_c, o_r))
    check('KDA chunk vs recurrent final state', rel(S_c, S_r))

    masks = {}
    ref.rec = lambda n, t: masks.__setitem__(n, t) if n.endswith('idx_mask') else None
    ref.forward(tok, SeqState())
    mk = masks['L03.attn.idx_mask']
    dropped = int((torch.tril(torch.ones(T, T, dtype=torch.bool)) & ~mk).sum())
    print(f"{'ok  ' if dropped else 'FAIL'} indexer drops {dropped} causal keys at T={T}, top {c.itopk} keys")
    if not dropped:
        fails.append('indexer sparsity')

    ref.rec = None
    st = SeqState()
    hid = ref.forward(tok, st)['hidden']
    st1, st2 = SeqState(), SeqState()
    nxt = torch.roll(tok, -1)
    all_rows = ref.mtp(hid, nxt, st1)
    rows = torch.cat([ref.mtp(hid[i:i + 1], nxt[i:i + 1], st2) for i in range(T)])
    check('MTP batched vs one row at a time', rel(rows, all_rows))

    print('PASS' if not fails else f'FAILED: {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
