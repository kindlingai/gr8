"""DFlash2 drafter (one safetensors file + config.json) -> one gr8 pack per rank for the drafter step
(engine/runtime/dflash.cuh). CPU only, numpy.

python3 -m tools.convert.dflash2 --src DIR --out DIR --nr 4 [--ranks 0,1,2,3] [--bf16 conv,sel] [--drop-cache]
python3 -m tools.convert.dflash2 --head GLM_DIR --out DIR --nr 4    (test only: head.r<r>.pack with the target's
        G.lm_head.mx.gv vocab slices, for engine/runtime/dfltest; the engine itself uses the rank packs' lm_head)

Tensor parallel split (NR must divide the 8 KV heads; TP4 = 16 q heads, 2 KV heads, 3,072 MLP channels per rank):
  D.fc.gv               fc rows of this rank's output slice [6144/NR][36864] (the context features -> 6144)
  D<j>.q.gv             q_proj rows of this rank's heads [64/NR * 128][6144]
  D<j>.kv.gv            k_proj then v_proj rows of this rank's KV heads [2 * 8/NR * 128][6144]
  D<j>.o.gv             o_proj columns of this rank's heads [6144][64/NR * 128]          (partial, all-reduced)
  D<j>.gu.gv            gate_proj then up_proj rows of this rank's channels [2 * 12288/NR][6144]
  D<j>.down.gv          down_proj columns of this rank's channels [6144][12288/NR]       (partial, all-reduced)
  D<j>.aconv.gv, D<j>.mconv.gv   the two convs' kernel_projection [1536][6144], every rank (replicated)
  D.sel.hproj.gv        candidate_selector.hidden_projection [256][6144], every rank
Weight records are MXFP8 decode records (g6ref.mxfp8_record, 528 B per group: the kv_a / q_b codec) unless their class
is listed in --bf16 (g6ref.bf16_record, 512 B per group): classes fc, attn (q, kv, o), mlp (gu, down), conv, sel.
The engine reads the codec from the record size.
BF16 as stored: D.hidden_norm, D.norm [6144]; D<j>.in_norm, D<j>.post_norm [6144]; D<j>.q_norm, D<j>.k_norm [128];
D<j>.aconv.base, D<j>.mconv.base [2 sides][2 taps][6144]; D.sel.pred, D.sel.succ [154880][256] (codebooks).
The manifest carries the drafter config (block size, mask id, target layers, window, rope theta, eps, selector).

`bin/g6load dflash2` (engine/loader-rs/src/dflash2.rs) writes the same packs and manifests byte for byte without
numpy; ./gr8 glm53 up --drafter uses it. Change both together: engine/loader-rs/tests/make_dflash2_fixture.py
rebuilds the test fixture from this file, and the cargo tests compare.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'engine', 'ref'))
import g6ref  # noqa: E402

from tools.convert import mxfp8, plan  # noqa: E402
from tools.convert.pack import PackWriter  # noqa: E402
from tools.convert.st import ST  # noqa: E402

VERSION = 'g6dflash2-v1'
HID, HD, NH, NKV, INTER = 6144, 128, 64, 8, 12288
CLASSES = ('fc', 'attn', 'mlp', 'conv', 'sel')


def check_config(cfg):
    d = cfg['dflash_config']
    want = dict(hidden_size=HID, head_dim=HD, num_attention_heads=NH, num_key_value_heads=NKV, intermediate_size=INTER)
    for k, v in want.items():
        if cfg[k] != v:
            raise SystemExit(f'config {k} = {cfg[k]}, this converter is written for {v}')
    if d['conv_kernel_size'] != 2 or HID % d['conv_group_size']:
        raise SystemExit('conv_kernel_size must be 2 and conv_group_size must divide hidden_size')
    if cfg.get('is_causal', False):
        raise SystemExit('a causal drafter is not supported (the block is attended to as a whole)')
    if cfg.get('attention_bias', False):
        raise SystemExit('attention bias is not supported')
    unsupported = dict(input_embedding_scale=d.get('input_embedding_scale', 1.0) != 1.0,
                       output_multiplier=d.get('output_multiplier', 1.0) != 1.0,
                       final_logit_softcapping=bool(d.get('final_logit_softcapping')),
                       attention_value_scale=d.get('attention_value_scale') is not None,
                       attention_sink_bias=bool(d.get('attention_sink_bias')),
                       sample_from_anchor=bool(d.get('sample_from_anchor')))
    for k, bad in unsupported.items():
        if bad:
            raise SystemExit(f'dflash_config {k} = {d[k]} is not supported')
    if cfg['vocab_size'] != 154880:
        raise SystemExit('vocab must be GLM 5.3 (154,880)')


def record(w_bf16, bf16):
    """[O, K] BF16 bits -> decode GEMV record stream: BF16 [O/16][K/16][512] or MXFP8 [O/16][K/32][528]."""
    if bf16:
        return g6ref.bf16_record(np.ascontiguousarray(w_bf16)), 'bf16_gv_record[O/16][K/16][512]'
    codes, scales = mxfp8.quantize(np.ascontiguousarray(w_bf16))
    return g6ref.mxfp8_record(codes, scales), 'mxfp8_gv_record[O/16][K/32][528]'


META = ('mask_id', 'nr', 'block', 'layers', 'window', 'tl0', 'tl1', 'tl2', 'tl3', 'tl4', 'tl5', 'theta', 'eps_bits',
        'sel_rank', 'sel_topk', 'conv_group', 'conv_kernel')


def put_meta(w, cfg, nr):
    """D.meta: int32 [17] in META order (the engine reads it instead of the manifest JSON)."""
    d = cfg['dflash_config']
    tl = list(d['target_layer_ids'])
    if len(tl) != 6:
        raise SystemExit('the engine taps six target layers')
    v = [d['mask_token_id'], nr, d['block_size'], cfg['num_hidden_layers'],
         cfg.get('sliding_window') if cfg.get('use_sliding_window', True) else 0, *tl,
         int(cfg['rope_parameters']['rope_theta']), int(np.float32(cfg['rms_norm_eps']).view(np.int32)),
         d.get('selector_rank') or 0, d.get('selector_top_k') or 0, d['conv_group_size'], d['conv_kernel_size']]
    w.put('D.meta', np.array(v, np.int32), 'i32', fields=list(META))


def convert_rank(st, cfg, nr, r, out, bf16, drop):
    d = cfg['dflash_config']
    nh, nkv, ni, nf = NH // nr, NKV // nr, INTER // nr, HID // nr
    meta = dict(format_dflash=VERSION, nr=nr, block_size=d['block_size'], mask_token_id=d['mask_token_id'],
                target_layer_ids=d['target_layer_ids'], num_layers=cfg['num_hidden_layers'],
                sliding_window=cfg.get('sliding_window') if cfg.get('use_sliding_window', True) else None,
                rope_theta=cfg['rope_parameters']['rope_theta'], rms_norm_eps=cfg['rms_norm_eps'],
                conv_group_size=d['conv_group_size'], conv_kernel_size=d['conv_kernel_size'],
                selector_rank=d.get('selector_rank'), selector_top_k=d.get('selector_top_k'), bf16_classes=sorted(bf16))
    w = PackWriter(os.path.join(out, f'dflash.r{r}.pack'), r, drop, meta)

    def put_rec(name, a, cls):
        rec, layout = record(a, cls in bf16)
        w.put(name, rec, 'u8', layout=layout, rows=int(a.shape[0]), cols=int(a.shape[1]))

    def put_bf(name, a):
        w.put(name, np.ascontiguousarray(a), 'bf16')

    put_meta(w, cfg, nr)
    put_rec('D.fc.gv', st.get('fc.weight', rows=(r * nf, (r + 1) * nf)), 'fc')
    put_bf('D.hidden_norm', st.get('hidden_norm.weight'))
    put_bf('D.norm', st.get('norm.weight'))
    for j in range(cfg['num_hidden_layers']):
        p = f'layers.{j}.'
        put_bf(f'D{j}.in_norm', st.get(p + 'input_layernorm.weight'))
        put_bf(f'D{j}.post_norm', st.get(p + 'post_attention_layernorm.weight'))
        put_bf(f'D{j}.q_norm', st.get(p + 'self_attn.q_norm.weight'))
        put_bf(f'D{j}.k_norm', st.get(p + 'self_attn.k_norm.weight'))
        put_rec(f'D{j}.q.gv', st.get(p + 'self_attn.q_proj.weight', rows=(r * nh * HD, (r + 1) * nh * HD)), 'attn')
        kv = np.concatenate([st.get(p + 'self_attn.k_proj.weight', rows=(r * nkv * HD, (r + 1) * nkv * HD)),
                             st.get(p + 'self_attn.v_proj.weight', rows=(r * nkv * HD, (r + 1) * nkv * HD))])
        put_rec(f'D{j}.kv.gv', kv, 'attn')
        o = st.get(p + 'self_attn.o_proj.weight')
        put_rec(f'D{j}.o.gv', o[:, r * nh * HD:(r + 1) * nh * HD], 'attn')
        del o
        gu = np.concatenate([st.get(p + 'mlp.gate_proj.weight', rows=(r * ni, (r + 1) * ni)),
                             st.get(p + 'mlp.up_proj.weight', rows=(r * ni, (r + 1) * ni))])
        put_rec(f'D{j}.gu.gv', gu, 'mlp')
        del gu
        dn = st.get(p + 'mlp.down_proj.weight')
        put_rec(f'D{j}.down.gv', dn[:, r * ni:(r + 1) * ni], 'mlp')
        del dn
        for c, s in (('aconv', 'attention_conv'), ('mconv', 'mlp_conv')):
            put_rec(f'D{j}.{c}.gv', st.get(p + f'{s}.kernel_projection.weight'), 'conv')
            put_bf(f'D{j}.{c}.base', st.get(p + f'{s}.base_kernel'))
        if drop:
            st.drop_cache()
    if 'candidate_selector.hidden_projection.weight' in st:
        put_rec('D.sel.hproj.gv', st.get('candidate_selector.hidden_projection.weight'), 'sel')
        put_bf('D.sel.pred', st.get('candidate_selector.predecessor_codebook'))
        put_bf('D.sel.succ', st.get('candidate_selector.successor_codebook'))
    return w.close()


def head_rank(src, nr, r, out):
    """The target lm_head rows of rank r's vocab slice as MXFP8 decode records, as convert.py --gv-records writes them."""
    plan.set_nr(nr)
    a, n = plan.vocab_rows(r)
    st = ST(os.path.join(src, 'model-head.safetensors'))
    w = PackWriter(os.path.join(out, f'head.r{r}.pack'), r, False, dict(format_dflash=VERSION, test_head=True))
    cs = mxfp8.quantize(st.get('lm_head.weight', rows=(a, a + n)))
    w.put('G.lm_head.mx.gv', g6ref.mxfp8_record(*cs), 'u8', layout='mxfp8_gv_record[O/16][K/32][528]', rows=n, cols=HID,
          vocab_rows=[a, a + n])
    return w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', help='drafter dir: config.json + model.safetensors')
    ap.add_argument('--head', help='test only: GLM 5.3 dir with model-head.safetensors -> head.r<r>.pack')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nr', type=int, default=4)
    ap.add_argument('--ranks', default=None, help='comma list (default all)')
    ap.add_argument('--bf16', default='', help=f'comma list of record classes kept in BF16: {",".join(CLASSES)}')
    ap.add_argument('--drop-cache', action='store_true')
    a = ap.parse_args()
    if NKV % a.nr or a.nr < 1:
        raise SystemExit('--nr must divide the 8 KV heads (1, 2, 4, 8)')
    ranks = [int(x) for x in a.ranks.split(',')] if a.ranks else list(range(a.nr))
    os.makedirs(a.out, exist_ok=True)
    if a.head:
        for r in ranks:
            m = head_rank(a.head, a.nr, r, a.out)
            print(json.dumps(dict(rank=r, head_bytes=m['pack_bytes'])), flush=True)
        return
    if not a.src:
        raise SystemExit('--src or --head')
    bf16 = set(x for x in a.bf16.split(',') if x)
    if bf16 - set(CLASSES):
        raise SystemExit(f'--bf16: unknown classes {sorted(bf16 - set(CLASSES))}')
    cfg = json.load(open(os.path.join(a.src, 'config.json')))
    check_config(cfg)
    st = ST(os.path.join(a.src, 'model.safetensors'))
    for r in ranks:
        t0 = time.time()
        m = convert_rank(st, cfg, a.nr, r, a.out, bf16, a.drop_cache)
        print(json.dumps(dict(rank=r, tensors=len(m['tensors']), bytes=m['pack_bytes'], seconds=round(time.time() - t0, 1))),
              flush=True)


if __name__ == '__main__':
    main()
