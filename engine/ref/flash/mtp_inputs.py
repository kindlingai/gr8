"""Prepare independent-op Flash MTP fixtures from a BF16 golden file and lossless weights.

No full model load: selected routed experts are read and written one at a time.
The DSA fixture uses the production attention test, with layer 45 head ownership.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from flash_ref import Cfg, FlashRef, rms
from weights import Exlr8Dense, HFWeights, Overlay
from dsa_inputs import projections, reference


def write(path, x):
    x = x.detach().cpu().contiguous() if isinstance(x, torch.Tensor) else torch.as_tensor(x).contiguous()
    if x.dtype == torch.bfloat16:
        x.view(torch.uint16).numpy().tofile(path)
    else:
        x.numpy().tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('golden', 'dense', 'loader', 'index', 'config', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--rows',type=int,default=8,help='prefix rows of the golden prompt (5..63)')
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    w = Overlay(Exlr8Dense(a.dense, a.loader), HFWeights(a.index))
    c = Cfg.load(a.config); ref = FlashRef(c, w)
    with safe_open(a.golden, framework='pt') as f:
        g = {k: f.get_tensor(k) for k in f.keys() if k.startswith(('MTP.', 'L45.'))}
    total = g['MTP.prev_hidden'].shape[0]
    assert 5 <= a.rows <= total
    g={k:(v[:a.rows] if v.ndim and v.shape[0]==total else v) for k,v in g.items()}
    n = a.rows
    assert n <= 64 and n > 1
    pos = torch.arange(n, dtype=torch.int32)
    # Compact embedding table: remap only the IDs for this kernel test.
    emb = w.get('embed_tokens.weight')[g['MTP.next_tokens']]
    write(d/'emb.bf16', emb)
    write(d/'tok.i32', pos); write(d/'pos.i32', pos)
    for name, value in g.items():
        if value.dtype.is_floating_point:
            write(d/(name+'.f32'), value.float())
    p = 'layers.45.'
    names = {'enorm': 'enorm.weight', 'hnorm': 'hnorm.weight', 'eh': 'eh_proj.weight',
             'attn_norm': 'input_layernorm.weight', 'mlp_norm': 'post_attention_layernorm.weight',
             'head_norm': 'shared_head.norm.weight', 'router': 'mlp.gate.weight'}
    for name, key in names.items():
        write(d/(name+'.bf16'), w.get(p+key).to(torch.bfloat16))
    write(d/'router_bias.f32', w.get(p+'mlp.gate.e_score_correction_bias').float())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import g6ref
    head=w.get('lm_head.weight').to(torch.bfloat16).view(torch.uint16).numpy()
    tiles=c.vocab//128
    for rank in range(4):
        lo=(tiles//4*rank+min(rank,tiles%4))*128
        width=(tiles//4+(rank<tiles%4))*128
        g6ref.bf16_record(head[lo:lo+width]).tofile(d/f'head.r{rank}.gv')
    e = emb.float().clone(); e[0] = 0
    cat = torch.cat([rms(e, w.get(p+'enorm.weight').float(), c.eps),
                     rms(g['MTP.prev_hidden'], w.get(p+'hnorm.weight').float(), c.eps)], -1)
    write(d/'cat.f32', cat)
    write(d/'head_in.f32', rms(g['MTP.out'], w.get(p+'shared_head.norm.weight').float(), c.eps))
    write(d/'topk.i32', g['L45.moe.topk_idx'].int())
    ids = sorted(g['L45.moe.topk_idx'].unique().tolist())
    for e in ids + ['shared']:
        pre = p + ('mlp.shared_experts.' if e == 'shared' else f'mlp.experts.{e}.')
        for proj in ('gate', 'up', 'down'):
            write(d/f'e{e}.{proj}.bf16', w.get(pre+proj+'_proj.weight').to(torch.bfloat16))
        print('expert', e, flush=True)
    (d/'meta.txt').write_text(f'{n} {len(ids)}\n'+' '.join(map(str,ids))+'\n')
    ds = d/'dsa'; ds.mkdir(exist_ok=True)
    inp = projections(ref,g['MTP.attn_in'],45)
    at = p+'self_attn.'
    kv = w.get(at+'kv_b_proj.weight').reshape(c.heads,c.nope+c.v,c.kv_lora)
    wt = {'uk':kv[:,:c.nope], 'uv':kv[:,c.nope:], 'kv_norm':w.get(at+'kv_a_layernorm.weight'),
          'k_norm_w':w.get(at+'indexer.k_norm.weight'), 'k_norm_b':w.get(at+'indexer.k_norm.bias'),
          'ape':w.get(at+'indexer.index_kpool_compress_ape')}
    for k,v in inp.items(): write(ds/(k+'.f16'),v.half())
    for k,v in wt.items(): write(ds/(k+'.bf16'),v.to(torch.bfloat16))
    for C in (1,5,n):
        for name,values in [('refa',{k:v.half().float().numpy() for k,v in inp.items()}),
                            ('refb',{k:v.float().numpy() for k,v in inp.items()})]:
            r=reference(values,{k:v.float().numpy() for k,v in wt.items()},C,c.eps)
            if name=='refb':
                expected=g['L45.attn.o'][C-1].numpy()
                rel=float(np.linalg.norm(r['o']-expected)/np.linalg.norm(expected))
                assert np.isfinite(rel) and rel<=1e-4, ('DSA reference vs golden',C,rel)
                print('DSA FP32 reference vs golden',C,rel,flush=True)
            for k,v in r.items(): write(ds/f'{name}_{C}_{k}.{ "i32" if k=="sel" else "f32"}',v)
    (d/'meta.json').write_text(json.dumps({'rows':n,'experts':ids,'golden':'prompt-bf16','layer':45},indent=2))
    print('fixture ready',n,flush=True)

if __name__ == '__main__':
    main()
