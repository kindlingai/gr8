#!/usr/bin/env python3
"""Gate the mHC host reference (ds_ref.py: embed / prepare / consume / final, the math ds_mhc.cuh implements) against the
reference-model dumps (engine/ref/ds41/fixtures, official model.py flow) with the real checkpoint weights. CPU only.

  python3 ds_oracle_gate.py ORACLE_DIR WEIGHTS_DIR [--layer 0] [--tol manifest.json]

Stages per layer l (inputs are the oracle's own dumps, so one op is tested alone):
  attn: prepare(l.x_in, pre_prev) -> attn_pre/attn_post/attn_comb, attn_in;  consume(l.attn_out) -> resid_attn
  ffn : prepare(l.resid_attn, attn_pre) -> ffn_pre/ffn_post/ffn_comb, ffn_in; consume(l.ffn_out) -> resid
  embed_T (token ids from the prompt), final.norm from final.h_pre, final.logits = final.norm[-1] @ head^T
Exit 1 when a stage is over its tolerance. Weight shards needed: embed (shard of embed.weight), the layer's, head/norm.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds_ref as R  # noqa: E402

EPS = 1e-20


def load(d, name, meta):
    m = meta[name]
    raw = np.fromfile(os.path.join(d, name + '.bin'), dtype=np.uint8)
    if m['dtype'] == 'bfloat16':
        t = torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16).float()
    elif m['dtype'] == 'float32':
        t = torch.from_numpy(raw.view(np.float32).copy())
    else:
        t = torch.from_numpy(raw.view(np.int32).copy())
    return t.reshape(m['shape'])


class Weights:
    def __init__(self, wdir):
        self.wdir = wdir
        self.map = json.load(open(os.path.join(wdir, 'model.safetensors.index.json')))['weight_map']

    def get(self, name):
        with safe_open(os.path.join(self.wdir, self.map[name]), 'pt') as f:
            return f.get_tensor(name)


def stat(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    d = (a - b).norm() / max(b.norm(), 1e-30)
    cos = (a @ b) / max(a.norm() * b.norm(), 1e-30)
    return float(d), float(cos), float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('oracle')
    ap.add_argument('weights')
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--tol', default=None)
    ap.add_argument('--no-head', action='store_true')
    ap.add_argument('--prev', type=int, default=None, help='layer whose ffn_pre is the carried pre (default: the dump layer before --layer)')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.oracle, 'summary.json')))['tensors']
    tol = json.load(open(a.tol))['tolerances'] if a.tol else {}
    W = Weights(a.weights)
    fails = 0

    def gate(label, got, ref, key):
        nonlocal fails
        rel, cos, mx = stat(got, ref)
        t = tol.get(key, {})
        lim = t.get('rel_l2', 1e-3)
        ok = rel <= lim and not torch.isnan(got).any()
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {label:22s} rel_l2 {rel:.3e} (limit {lim:g})  cos {cos:.7f}  max_abs {mx:.3e}")

    l = a.layer
    P = f'L{l}.'
    ld = lambda n: load(a.oracle, n, meta)
    # embed: the oracle's x_in at layer 0 is embed_T; token ids come from the prompt file
    if l == 0:
        prompt = json.load(open(json.load(open(os.path.join(a.oracle, 'summary.json')))['prompt']))
        ids = prompt['ids'] if isinstance(prompt, dict) and 'ids' in prompt else prompt
        emb = W.get('embed.weight').float()
        tok = torch.tensor(ids if not isinstance(ids[0], list) else ids[0])
        T, _ = R.embed(tok, emb)
        et = ld('embed_T')[0]
        print(f"{'PASS' if torch.equal(T, et) else 'FAIL'} embed_T bit-equal ({tuple(T.shape)})")
        fails += not torch.equal(T, et)
        pre_prev = torch.zeros(T.shape[0], 4)
        pre_prev[:, 0] = 1.0
    else:
        layers = json.load(open(os.path.join(a.oracle, 'summary.json')))['layers']
        prev = a.prev if a.prev is not None else layers[layers.index(l) - 1]
        pre_prev = ld(f'L{prev}.ffn_pre')[0]
        print(f'carried pre = L{prev}.ffn_pre (subset dump: layers {layers})')
    x_in = ld(P + 'x_in')[0]
    if l == 0:
        gate('x_in vs embed', x_in, T, 'embed_T')
    T = x_in
    if 'final.h_pre' in meta and l != 0:
        a.no_head = True    # the final stages hang off the LAST layer of the chain; gate them once, with --layer 0
    S = T.shape[0]
    hw = lambda s, n: W.get(f'layers.{l}.{n}')
    # attention sub-layer
    fn, base, scale = hw(0, 'hc_attn_fn').float(), hw(0, 'hc_attn_base').float(), hw(0, 'hc_attn_scale').float()
    nw = hw(0, 'attn_norm.weight').float()
    pre_n, post, comb, y = R.prepare(T, pre_prev, fn, scale, base, nw, EPS)
    gate(P + 'attn_pre', pre_n, ld(P + 'attn_pre')[0], 'attn_pre')
    gate(P + 'attn_post', post, ld(P + 'attn_post')[0], 'attn_post')
    gate(P + 'attn_comb', comb, ld(P + 'attn_comb')[0], 'attn_comb')
    gate(P + 'attn_in', y, ld(P + 'attn_in')[0], 'attn_in')
    T2 = R.consume(T, ld(P + 'attn_out')[0], post, comb)
    gate(P + 'resid_attn', T2, ld(P + 'resid_attn')[0], 'resid_attn')
    # FFN sub-layer, from the oracle's own resid_attn and its attn_pre
    fn, base, scale = hw(0, 'hc_ffn_fn').float(), hw(0, 'hc_ffn_base').float(), hw(0, 'hc_ffn_scale').float()
    nw = hw(0, 'ffn_norm.weight').float()
    Tr = ld(P + 'resid_attn')[0]
    pre_n2, post2, comb2, y2 = R.prepare(Tr, ld(P + 'attn_pre')[0], fn, scale, base, nw, EPS)
    gate(P + 'ffn_pre', pre_n2, ld(P + 'ffn_pre')[0], 'ffn_pre')
    gate(P + 'ffn_post', post2, ld(P + 'ffn_post')[0], 'ffn_post')
    gate(P + 'ffn_comb', comb2, ld(P + 'ffn_comb')[0], 'ffn_comb')
    gate(P + 'ffn_in', y2, ld(P + 'ffn_in')[0], 'ffn_in')
    T3 = R.consume(Tr, ld(P + 'ffn_out')[0], post2, comb2)
    gate(P + 'resid', T3, ld(P + 'resid')[0], 'resid')
    # end of model: collapse with the last sub-layer's pre, norm, head
    if 'final.h_pre' in meta and not a.no_head:
        hp = ld('final.h_pre')[0]
        norm_w = W.get('norm.weight').float()
        fnorm = R.rmsnorm_bf16(hp, norm_w, EPS)
        gate('final.norm', fnorm, ld('final.norm')[0], 'final.norm')
        head = W.get('head.weight')
        ref = ld('final.logits')[0]
        h = ld('final.norm')[0][-1]
        lg = torch.empty(head.shape[0])
        for i in range(0, head.shape[0], 8192):
            lg[i:i + 8192] = head[i:i + 8192].float() @ h
        gate('final.logits', lg, ref, 'final.logits')
        print(f"argmax engine-side {int(lg.argmax())} oracle {int(ref.argmax())}")
        fails += int(lg.argmax()) != int(ref.argmax())
    print('FAIL' if fails else 'PASS', f'({fails} stage(s) over tolerance)')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
