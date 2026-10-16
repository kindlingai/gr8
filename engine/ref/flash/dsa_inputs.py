"""Kernel inputs and references for the W5 DSA attention test (engine/attn/tests/flash_dsa_test.cu).

    python3 engine/ref/flash/dsa_inputs.py --index FLASH/fetch-evidence/model.safetensors.index.json \
        --dense FLASH/full-dense --loader FLASH/src/exlr8-loader --text 'docs/*.md' --n 9000 --out DIR

Tokens: the --text files concatenated, the first --n tokens. Layers 0-2 run on them (chunked KDA, segments of --seg
tokens), then the layer-3 mHC collapse + input_layernorm gives the attention input x. From x, in fp32, every projection
the attention binding consumes:
  q_b [N][64][256]   q_b(q_a_layernorm(q_a(x)))          kv_a [N][512]  kv_a_proj_with_mqa(x) (before kv_a_layernorm)
  idx_q [N][32][128] indexer.wq_b(q_resid)               idx_k [N][128] indexer.wk(x) (before k_norm)
  idx_w [N][32]      indexer.weights_proj(x)              idx_g [N][128] x @ index_kpool_compress_gate^T
written as fp16 (what the binding reads), plus the bf16 weights the binding binds (uk, uv [64][256][512], kv_norm,
k_norm w/b, ape [4][128]).

References at each checkpoint context C (--ctx, default 1,5,2049,9000; the query is the token at position C-1, the
cache holds tokens 0..C-1), from the same math as flash_ref.FlashRef.indexer/dsa (checked against it on --check
tokens):
  ref_a: float64 from the fp16-rounded inputs (the kernel's own inputs: the gate)
  ref_b: float64 from the fp32 inputs (how far the fp16 projections move the result: reported)
Each: sel (int32 selected positions, ascending), score (pool scores, -inf = invalid), o [64][256] float32.

Output: DIR/meta.json + one raw little-endian file per array (name.f16 / .bf16 / .f32 / .i32).
"""
import argparse
import glob
import json
import math
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_ref import Cfg, FlashRef, SeqState, rms  # noqa: E402
from weights import Exlr8Dense, HFWeights, Overlay  # noqa: E402


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def projections(ref, x, L):
    p = f'layers.{L}.self_attn.'
    qa = ref.lin(x, p + 'q_a_proj.weight')
    qr = rms(qa, ref.g(p + 'q_a_layernorm.weight'), ref.c.eps)
    return {
        'q_b': ref.lin(qr, p + 'q_b_proj.weight').view(-1, ref.c.heads, ref.c.nope),
        'kv_a': ref.lin(x, p + 'kv_a_proj_with_mqa.weight'),
        'idx_q': ref.lin(qr, p + 'indexer.wq_b.weight').view(-1, ref.c.ih, ref.c.idim),
        'idx_k': ref.lin(x, p + 'indexer.wk.weight'),
        'idx_w': ref.lin(x, p + 'indexer.weights_proj.weight'),
        'idx_g': x @ ref.g(p + 'indexer.index_kpool_compress_gate').T,
    }


def reference(inp, wt, C, eps, P=4, topk=2048):
    """float64 decode of the query at position C-1 over tokens 0..C-1 (flash_ref math, one query row)"""
    d = {k: torch.as_tensor(v[:C]).double() for k, v in inp.items()}
    w = {k: torch.as_tensor(v).double() for k, v in wt.items()}
    lat = d['kv_a']
    lat = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + eps) * w['kv_norm']
    ik = F.layer_norm(d['idx_k'], (128,), w['k_norm_w'], w['k_norm_b'], 1e-6)
    n_pools = (C + P - 1) // P
    pad = n_pools * P - C
    keys = F.pad(ik, (0, 0, 0, pad)).view(n_pools, P, 128)
    logits = F.pad(d['idx_g'], (0, 0, 0, pad)).view(n_pools, P, 128) + w['ape']
    exists = (torch.arange(n_pools * P) < C).view(n_pools, P)
    logits = logits.masked_fill(~exists[..., None], float('-inf'))
    pool_k = (torch.nan_to_num(logits.softmax(1)) * keys).sum(1)
    q = d['idx_q'][C - 1]                                                       # [32,128]
    s = F.relu(q @ pool_k.T * 128 ** -0.5)                                      # [32,pools]
    score = (d['idx_w'][C - 1] * 32 ** -0.5) @ s
    qpos = C - 1
    valid = (torch.arange(n_pools) * P + P - 1 <= qpos) & exists.all(1)
    score = score.masked_fill(~valid, float('-inf'))
    k_sel = min(topk // P, n_pools)
    order = score.sort(descending=True, stable=True).indices[:k_sel]
    pools = sorted(int(p) for p in order if valid[p])
    sel = [t for p in pools for t in range(p * P, p * P + P)]
    cnt = C
    sel += list(range(cnt - cnt % P, cnt))
    sel = torch.tensor(sel, dtype=torch.long)
    kvb_uk, kvb_uv = w['uk'], w['uv']                                            # [64,256,512]
    q_abs = torch.einsum('hn,hnl->hl', d['q_b'][C - 1], kvb_uk)
    sc = (q_abs @ lat[sel].T) * 256 ** -0.5
    ctx = sc.softmax(-1) @ lat[sel]
    o = torch.einsum('hl,hvl->hv', ctx, kvb_uv)
    return {'sel': sel.numpy().astype(np.int32), 'score': score.float().numpy(), 'o': o.float().numpy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', required=True)
    ap.add_argument('--config')
    ap.add_argument('--dense', required=True)
    ap.add_argument('--loader', required=True)
    ap.add_argument('--text', required=True, help='comma-separated globs of text files')
    ap.add_argument('--n', type=int, default=9000)
    ap.add_argument('--seg', type=int, default=512)
    ap.add_argument('--ctx', default='1,5,2049,9000')
    ap.add_argument('--check', type=int, default=300, help='tokens for the flash_ref.dsa equality check (0 = skip)')
    ap.add_argument('--layer', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.time()
    cfg = Cfg.load(a.config or os.path.join(os.path.dirname(a.index), 'config.json'))
    hf = HFWeights(a.index)
    w = Overlay(Exlr8Dense(a.dense, a.loader), hf)
    ref = FlashRef(cfg, w)
    from tokenizers import Tokenizer
    text = ''.join(open(f, errors='replace').read() for g in a.text.split(',') for f in sorted(glob.glob(g)))
    tokens = Tokenizer.from_file(os.path.join(hf.dir, 'tokenizer.json')).encode(text).ids
    assert len(tokens) >= a.n, f'only {len(tokens)} tokens in {a.text}'
    tokens = tokens[:a.n]
    L = a.layer
    st = SeqState()
    xs = []
    for s0 in range(0, a.n, a.seg):
        tk = torch.tensor(tokens[s0:s0 + a.seg])
        s = w.get('embed_tokens.weight')[tk].float().unsqueeze(1).expand(-1, cfg.hc, -1).contiguous()
        for l in range(L):
            s = ref.layer(s, l, st, chunked=True)
        x, _, _ = ref.mhc(s, f'layers.{L}.hc_attn_', 'x')
        xs.append(rms(x, ref.g(f'layers.{L}.input_layernorm.weight'), cfg.eps))
        print(f'tokens {s0 + len(tk)}/{a.n} {time.time() - t0:.0f}s peak {rss_gb():.1f} GB', flush=True)
    x = torch.cat(xs)
    dense = w.sources[0]                  # only layer L's weights are needed from here (8 GB memory cap)
    dense.vocab, dense.layers = None, {L: dense.layers[L]}
    hf.cache.clear()
    del st, s, xs
    import gc
    gc.collect()
    inp = {k: v.float().contiguous() for k, v in projections(ref, x, L).items()}
    p = f'layers.{L}.self_attn.'
    kvb = w.get(p + 'kv_b_proj.weight').view(cfg.heads, cfg.nope + cfg.v, cfg.kv_lora)
    wt = {'uk': kvb[:, :cfg.nope].contiguous(), 'uv': kvb[:, cfg.nope:].contiguous(),
          'kv_norm': w.get(p + 'kv_a_layernorm.weight'),
          'k_norm_w': w.get(p + 'indexer.k_norm.weight'), 'k_norm_b': w.get(p + 'indexer.k_norm.bias'),
          'ape': w.get(p + 'indexer.index_kpool_compress_ape')}
    wt = {k: v.to(torch.bfloat16).contiguous() for k, v in wt.items()}

    if a.check:
        # the reference function == flash_ref.dsa on the first --check tokens (fp32 inputs, every query row)
        T = a.check
        rec = {}
        ref.rec = lambda n, t: rec.__setitem__(n, t)
        ref.dsa(x[:T], L, SeqState())
        n_sel = int((rec['L%02d.attn.idx_mask' % L][-1]).sum())
        ref.rec = None
        worst_o, sel_ok = 0.0, True
        for C in [c for c in range(1, T + 1) if c <= 8 or c % 37 == 0 or c > T - 40]:
            r = reference({k: v.numpy() for k, v in inp.items()}, {k: v.float().numpy() for k, v in wt.items()},
                          C, cfg.eps)
            m = rec['L%02d.attn.idx_mask' % L][C - 1, :C].nonzero().flatten().numpy()
            sel_ok &= np.array_equal(np.sort(m), r['sel'])
            o = rec['L%02d.attn.o' % L][C - 1].numpy()
            worst_o = max(worst_o, float(np.linalg.norm(o - r['o']) / np.linalg.norm(o)))
        print(f'check vs flash_ref.dsa on {T} rows (last row attends {n_sel} of {T}): selection equal {sel_ok}, worst o rel-L2 {worst_o:.2e}', flush=True)
        assert sel_ok and worst_o < 1e-5

    os.makedirs(a.out, exist_ok=True)
    meta = {'n': a.n, 'layer': L, 'ctx': [int(c) for c in a.ctx.split(',')], 'eps': cfg.eps, 'arrays': {},
            'source': 'zai-org/GLM-5.3-Flash-BF16@a5b45eb', 'text': a.text}

    def put(name, arr, ext):
        arr = np.ascontiguousarray(arr)
        arr.tofile(os.path.join(a.out, f'{name}.{ext}'))
        meta['arrays'][name] = {'file': f'{name}.{ext}', 'shape': list(arr.shape)}

    inp16 = {k: v.half() for k, v in inp.items()}
    for k, v in inp16.items():
        put(k, v.numpy(), 'f16')
    for k, v in wt.items():
        put(k, v.view(torch.int16).numpy(), 'bf16')
    put('tokens', np.array(tokens, dtype=np.int32), 'i32')
    wt32 = {k: v.float().numpy() for k, v in wt.items()}
    for C in meta['ctx']:
        ra = reference({k: v.float().numpy() for k, v in inp16.items()}, wt32, C, cfg.eps)
        rb = reference({k: v.numpy() for k, v in inp.items()}, wt32, C, cfg.eps)
        for tag, r in (('a', ra), ('b', rb)):
            for k, v in r.items():
                put(f'ref{tag}_{C}_{k}', v, 'i32' if k == 'sel' else 'f32')
        jac = len(set(ra['sel']) & set(rb['sel'])) / max(1, len(set(ra['sel']) | set(rb['sel'])))
        print(f'C={C}: selected {len(ra["sel"])}, fp16-vs-fp32 inputs: selection Jaccard {jac:.4f}, o rel-L2 '
              f'{np.linalg.norm(ra["o"] - rb["o"]) / np.linalg.norm(rb["o"]):.2e}', flush=True)
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {a.out} {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
