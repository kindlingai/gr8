"""Cross-check golden files against transformers' own glm5_next modules on the REAL weights, in fp32, from the same
inputs: the whole decoder layer 0 (mHC, KDA, dense MLP) and the attention block of layer 3 (DSA: NoPE MLA + kpool
indexer). test_flash_ref.py proves the code on a random config; this proves the checkpoint names, shapes and the
real config are read the same way.

    python3 engine/ref/flash/check_real.py --index FLASH/fetch-evidence/model.safetensors.index.json \
        --golden /tmp/fe/golden/prompt-bf16.safetensors --dense FLASH/full-dense --loader FLASH/src/exlr8-loader
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import Exlr8Dense, HFWeights  # noqa: E402

TOL = 1e-5


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def module_state(w, L, names):
    """checkpoint tensors of layer L -> transformers module names (fp32)"""
    sd = {}
    for n in names:
        t = w.get(f'layers.{L}.{n}').float()
        k = n
        for a in ('attn', 'ffn'):
            for f in ('fn', 'base', 'scale'):
                k = k.replace(f'hc_{a}_{f}', f'{a}_hc.{f}')
        for f in ('f_a_proj.', 'f_b_proj.', 'dt_bias', 'A_log'):
            k = k.replace(f'self_attn.{f}', f'self_attn.forget_gate.{f}')
        sd[k] = t
    conv = [sd.pop(f'self_attn.{c}_conv1d.weight') for c in 'qkv' if f'self_attn.{c}_conv1d.weight' in sd]
    if conv:
        sd['self_attn.conv1d.weight'] = torch.cat(conv)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--golden', required=True)
    ap.add_argument('--dense', help='exlr8 release with the lossless BF16 dense/ export (else HF range reads)')
    ap.add_argument('--loader', help='exlr8-loader repo root (with --dense)')
    a = ap.parse_args()
    from transformers.models.glm5_next import modeling_glm5_next as M
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    cfg = Glm5NextConfig.from_pretrained(os.path.dirname(a.index)).text_config
    cfg._attn_implementation = 'eager'
    w = Exlr8Dense(a.dense, a.loader) if a.dense else HFWeights(a.index)
    g = load_file(a.golden)
    T = g['tokens'].shape[0]
    fails = []

    def check(name, err):
        print(f"{'ok  ' if err <= TOL else 'FAIL'} {name}: rel {err:.2e} (<= {TOL:.0e})")
        if err > TOL:
            fails.append(name)

    mask = torch.ones(1, T, dtype=torch.bool)
    import json
    with open(a.index) as f:
        wmap = json.load(f)['weight_map']
    names0 = [k[len('model.language_model.layers.0.'):] for k in wmap if k.startswith('model.language_model.layers.0.')]
    with torch.device('meta'):
        layer0 = M.Glm5NextTextDecoderLayer(cfg, 0)
    layer0.load_state_dict(module_state(w, 0, names0), strict=True, assign=True)
    layer0 = layer0.float().eval()
    with torch.no_grad():
        out, _ = layer0(g['L00.in'][None], attention_mask=mask, position_ids=torch.arange(T)[None])
    check('layer 0 (mHC + KDA + dense MLP) vs golden L00.out', rel(g['L00.out'], out[0]))

    names3 = [k[len('model.language_model.layers.3.'):] for k in wmap
              if k.startswith('model.language_model.layers.3.self_attn.')]
    with torch.device('meta'):
        attn = M.Glm5NextTextAttention(cfg, 3)
    attn.load_state_dict({k[len('self_attn.'):]: v for k, v in module_state(w, 3, names3).items()}, strict=True,
                         assign=True)
    attn = attn.float().eval()
    x3 = g['L03.hc_attn.x']
    norm = w.get('layers.3.input_layernorm.weight').float()
    h = x3 * torch.rsqrt(x3.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm
    with torch.no_grad():
        o = attn(h[None], attention_mask=mask, past_key_values=None)[0][0]
    check('layer 3 attention (NoPE MLA + kpool indexer) vs golden L03.attn.out', rel(g['L03.attn.out'], o))
    print('PASS' if not fails else f'FAILED: {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
