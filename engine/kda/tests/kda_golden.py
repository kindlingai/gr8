"""KDA layer goldens for the W4 kernels, from flash_ref.FlashRef.kda (fp32) on one real layer's lossless BF16 weights.

    python3 engine/kda/tests/kda_golden.py --dense FLASH/full-dense --config FLASH/fetch-evidence/config.json \
        --layer 0 --tokens 4096 --out FLASH/engine/golden/kda-L00-4096.safetensors

Input x[t] = rms(embed[tok_t]) * input_layernorm (real embedding rows of seeded random token ids); the layer runs
recurrent (token-by-token math) over all tokens. One safetensors file, keys as golden.py / INTERFACES.md 4.2:

  Lxx.attn_in                  [T, H] fp32 layer input
  Lxx.kda.{mixed,q,k,v,g,beta,o,o_gated,out}   [T, ...] every op of FlashRef.kda for every token (fp32)
  Lxx.S.<t> Lxx.conv.<t>       state after t tokens, t in 1, 63, 64, T-1, T (S [heads, k, v]; conv [3, 3D])
  Lxx.chunk.o Lxx.chunk.S   FlashRef.kda_chunk (block 64, 128-token calls carrying S) on the same q/k/v/g/beta from zero state
"""
import argparse
import json
import os
import struct
import sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ref', 'flash'))
from flash_ref import Cfg, FlashRef, SeqState, rms  # noqa: E402
from weights import DictWeights  # noqa: E402

DT = {'BF16': torch.bfloat16, 'F32': torch.float32}


def read_safetensors(path, want=None):
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        for k, v in h.items():
            if k == '__metadata__' or v['dtype'] not in DT or (want and not want(k)):
                continue
            a, b = v['data_offsets']
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            out[k] = torch.frombuffer(buf, dtype=DT[v['dtype']]).reshape(v['shape'])
    return out, h.get('__metadata__', {})


def layer_weights(dense, L):
    t, meta = read_safetensors(os.path.join(dense, 'dense', f'exlr8-dense-{L:03d}.safetensors'),
                               lambda k: k.startswith(('self_attn.', 'input_layernorm')))
    names = json.loads(meta['source_names'])
    shapes = json.loads(meta['source_shapes'])
    d = {}
    for k, v in t.items():
        local = k[:-5] if k.endswith('.bf16') else k
        full = names[local]
        d[full[len('model.language_model.'):]] = v.reshape(shapes.get(local, list(v.shape)))
    return d


def embed_rows(dd, ids, H):
    out = torch.empty(len(ids), H, dtype=torch.bfloat16)
    for p in sorted(os.listdir(dd)):
        if not p.startswith('exlr8-split-model.language_model.embed_tokens.weight.p'):
            continue
        with open(os.path.join(dd, p), 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
            r0, r1 = (int(v) for v in h['__metadata__']['rows'].split('-'))
            (k, v), = [(k, v) for k, v in h.items() if k.endswith('.bf16')]
            assert v['shape'] == [r1 - r0, H]
            for i, t in enumerate(ids):
                if r0 <= t < r1:
                    f.seek(8 + n + v['data_offsets'][0] + (t - r0) * H * 2)
                    out[i] = torch.frombuffer(bytearray(f.read(H * 2)), dtype=torch.bfloat16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    cfg = Cfg.load(a.config)
    L, T = a.layer, a.tokens
    assert cfg.kda[L], f'layer {L} is not KDA'
    w = layer_weights(a.dense, L)
    # embedding rows of the token ids, read row by row from the split vocab files (the whole table is 1.27 GB)
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, cfg.vocab, (T,), generator=g)
    rows = embed_rows(os.path.join(a.dense, 'dense'), ids.tolist(), cfg.H)
    p = f'layers.{L}.'
    x = rms(rows.float(), w[p + 'input_layernorm.weight'].float(), cfg.eps)

    rec = {}

    def r(name, t):
        rec.setdefault(name, []).append(t.detach().clone())

    ref = FlashRef(cfg, DictWeights(w), torch.float32, r)
    st = SeqState()
    marks = sorted({1, 63, 64, T - 1, T} - {0})
    states = {}
    t0 = 0
    for m in sorted(set(marks) | set(range(512, T, 512))):   # calls of <= 512 tokens keep the peak under 8 GB
        ref.kda(x[t0:m], L, st)
        if m in marks:
            states[m] = (st.kda[L]['S'].clone(), st.kda[L]['conv'].clone())
        t0 = m
    tag = f'L{L:02d}.kda.'
    seq = {}
    for n in ('mixed', 'q', 'k', 'v', 'g', 'beta', 'o', 'o_gated', 'out'):
        seq[n] = torch.cat(rec.pop(tag + n))
    rec.clear()
    D = cfg.kh * cfg.kd
    S0 = torch.zeros(cfg.kh, cfg.kd, cfg.kd)
    cos, cS = [], S0
    for a0 in range(0, T, 128):      # segments carry the state; one 4096-token call needs ~9 GB of decay tensors
        sl = slice(a0, min(T, a0 + 128))
        o_, cS = FlashRef.kda_chunk(seq['q'][sl], seq['k'][sl], seq['v'][sl], seq['g'][sl], seq['beta'][sl], cS, 64)
        cos.append(o_)
    co = torch.cat(cos)

    tag = f'L{L:02d}.'
    out = {tag + 'attn_in': x}
    for n, t in seq.items():
        out[tag + 'kda.' + n] = t.float()
    for m, (S, conv) in states.items():
        out[f'{tag}S.{m}'] = S
        out[f'{tag}conv.{m}'] = conv.float()
    out[tag + 'chunk.o'] = co
    out[tag + 'chunk.S'] = cS
    rel = lambda a_, b_: ((a_ - b_).norm() / b_.norm()).item()
    meta = {'layer': L, 'tokens': T, 'seed': a.seed, 'eps': cfg.eps, 'marks': marks,
            'chunk_vs_recurrent_o': rel(co, seq['o']), 'chunk_vs_recurrent_S': rel(cS, states[T][0])}
    save_file({k: v.contiguous() for k, v in out.items()}, a.out, metadata={k: json.dumps(v) for k, v in meta.items()})
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
