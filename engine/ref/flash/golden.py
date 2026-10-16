"""Golden vectors for the gr8 Flash kernels from flash_ref.FlashRef on the real GLM-5.3-Flash weights (W0b).

    python3 engine/ref/flash/golden.py --index FLASH/fetch-evidence/model.safetensors.index.json \
        --local-bf16 FLASH/fetch-evidence/L003-e000-071.bf16 --release FLASH/loader-evidence/real-pilot/release \
        --loader FLASH/src/exlr8-loader --out /tmp/fe/golden

Cases (one safetensors file each, every op output flash_ref records, plus the inputs a kernel test needs):
  prompt-bf16.safetensors   a 64-token prompt through layers 0-3 (KDA+dense x3, DSA+MoE) in one call, BF16
                            weights; final norm + lm_head of the cut model; MTP layer 45 on the cut model's hidden
  step-bf16.safetensors     token 65 decoded one step after that, every layer: state before the step
                            (`L00.state_in.S`, `.conv`, `L03.state_in.lat`, ...) and every op of the step
  layer3-K{2,3,4}.safetensors  layer 3 from the same input streams (L03.in of the prompt case) with the release's
                            exlr8 experts at that width (routed and shared), all ops
  long-L03.safetensors      layer 3 alone on --long tokens (default 4500 > 2048: the indexer selects); inputs are
                            the embeddings of a repeated prompt through layers 0-2 (chunked KDA)
Non-expert tensors come from --dense (the lossless exlr8 BF16 export); BF16 experts from --local-bf16, the rest from
the HF hub by range read (weights.HFWeights, cached on disk).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_ref import Cfg, FlashRef, SeqState  # noqa: E402
from weights import Exlr8Dense, Exlr8Experts, HFWeights, Overlay, SafeWeights  # noqa: E402

PROMPT = ("You are a careful senior engineer. Explain, step by step, why a recurrent state of fixed size lets a "
          "linear attention layer decode in constant time per token, and what that costs for long-range recall. "
          "Then write a short Python function that checks whether two strings are anagrams, with a docstring and two "
          "doctest examples, and say what its time complexity is.")


class Recorder:
    def __init__(self):
        self.t = {}

    def __call__(self, name, x):
        assert name not in self.t, f'{name} recorded twice'
        self.t[name] = x.detach().clone()

    def save(self, path, meta):
        out = {k: (v.to(torch.uint8) if v.dtype == torch.bool else v).contiguous() for k, v in self.t.items()}
        save_file(out, path, metadata={k: json.dumps(v) for k, v in meta.items()})
        print(f'wrote {path}: {len(out)} tensors, {os.path.getsize(path) / 1e6:.1f} MB', flush=True)


def tokenize(text, tok_path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tok_path).encode(text).ids


def state_tensors(st, prefix='state_in'):
    out = {}
    for L, s in st.kda.items():
        out[f'L{L:02d}.{prefix}.S'] = s['S'].clone()
        out[f'L{L:02d}.{prefix}.conv'] = s['conv'].clone()
    for key, c in st.dsa.items():
        tag = 'MTP' if key == 'mtp' else f'L{key:02d}'
        for n in ('lat', 'ik', 'ig'):
            out[f'{tag}.{prefix}.{n}'] = c[n].clone()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', required=True)
    ap.add_argument('--config', help='config.json (default: next to --index)')
    ap.add_argument('--dense', help='exlr8 release with the lossless BF16 dense/ export (used first)')
    ap.add_argument('--local-bf16', help='a local safetensors dir with some BF16 tensors (used next)')
    ap.add_argument('--release', help='exlr8 release root with experts/ for the layer-3 width cases')
    ap.add_argument('--loader', help='exlr8-loader repo root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--long', type=int, default=4500)
    ap.add_argument('--cases', default='prompt,step,widths,long')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = Cfg.load(a.config or os.path.join(os.path.dirname(a.index), 'config.json'))
    t0 = time.time()
    hf = HFWeights(a.index)
    srcs = [Exlr8Dense(a.dense, a.loader)] if a.dense else []
    srcs += [SafeWeights(a.local_bf16)] if a.local_bf16 else []
    w = Overlay(*srcs, hf)
    missing = [f'layers.3.mlp.experts.{e}.{p}.weight' for e in range(cfg.E) for p in ('gate_proj', 'up_proj', 'down_proj')
               if not any(s.has(f'layers.3.mlp.experts.{e}.{p}.weight') for s in srcs)]
    if missing:
        t = time.time()
        hf.prefetch(missing)
        print(f'prefetched {len(missing)} BF16 expert tensors of layer 3 from HF in {time.time() - t:.0f}s', flush=True)
    tok_path = os.path.join(hf.dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        import urllib.request
        urllib.request.urlretrieve(hf.url + 'tokenizer.json', tok_path)
    tokens = tokenize(PROMPT, tok_path)
    assert len(tokens) >= a.tokens + 1, f'prompt has {len(tokens)} tokens'
    tokens = tokens[:a.tokens + 1]
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    meta = {'tokens': tokens[:a.tokens], 'ref_commit': head, 'source': 'zai-org/GLM-5.3-Flash-BF16@a5b45eb',
            'dtype': 'float32', 'layers': [0, 1, 2, 3]}
    cases = a.cases.split(',')
    layers = [0, 1, 2, 3]

    rec = Recorder()
    ref = FlashRef(cfg, w, rec=rec)
    st = SeqState()
    with torch.no_grad():
        r = ref.forward(tokens[:a.tokens], st, layers=layers)
        rec.t['tokens'] = torch.tensor(tokens[:a.tokens])
        mtp_st = SeqState()
        ref.mtp(r['hidden'][:-1], torch.tensor(tokens[1:a.tokens]), mtp_st)
        rec.t['MTP.prev_hidden'] = r['hidden'][:-1].clone()
        rec.t['MTP.next_tokens'] = torch.tensor(tokens[1:a.tokens])
    prompt = rec.t
    if 'prompt' in cases:
        rec.save(os.path.join(a.out, 'prompt-bf16.safetensors'), dict(meta, note='layers 0-3 cut model; MTP on the '
                 'cut model hidden (not layer 44): an op-level golden, not a model-quality number'))
    print(f'prompt case {time.time() - t0:.0f}s', flush=True)

    if 'step' in cases:
        rec = Recorder()
        rec.t.update(state_tensors(st))
        rec.t.update(state_tensors(mtp_st))
        ref.rec = rec
        with torch.no_grad():
            r2 = ref.forward(tokens[a.tokens:a.tokens + 1], st, layers=layers)
            ref.mtp(r['hidden'][-1:], torch.tensor(tokens[a.tokens:a.tokens + 1]), mtp_st)
        rec.t['tokens'] = torch.tensor(tokens[a.tokens:a.tokens + 1])
        rec.t['MTP.prev_hidden'] = r['hidden'][-1:].clone()
        rec.save(os.path.join(a.out, 'step-bf16.safetensors'), dict(meta, position=a.tokens,
                 note='one decode step at the given position after the prompt case; *.state_in.* = state before it'))
        print(f'step case {time.time() - t0:.0f}s', flush=True)

    if 'widths' in cases and a.release:
        x3 = prompt['L03.in']
        for K in (4, 3, 2):
            rec = Recorder()
            qw = Overlay(Exlr8Experts(a.release, a.loader, K, layers=[3]), w)
            refq = FlashRef(cfg, qw, rec=rec)
            with torch.no_grad():
                refq.layer(x3, 3, SeqState())
            rec.save(os.path.join(a.out, f'layer3-K{K}.safetensors'),
                     dict(meta, experts=f'exlr8 K{K} routed + shared from {a.release}', layers=[3]))
            print(f'K{K} case {time.time() - t0:.0f}s', flush=True)

    if 'long' in cases:
        n = a.long
        long_tok = (tokens[:a.tokens] * (n // a.tokens + 1))[:n]
        lst = SeqState()
        ref.rec = None
        with torch.no_grad():
            x = ref.emb(long_tok)
            s = x.unsqueeze(1).expand(-1, cfg.hc, -1).contiguous()
            drop = [d for d in srcs if isinstance(d, Exlr8Dense)]
            for d in drop:
                d.vocab = None                          # free each cached weight once it is used (8 GiB cap)
            for L in (0, 1, 2):
                s = ref.layer(s, L, lst, chunked=True)
                for d in drop:
                    d.layers.pop(L, None)
                lst.kda.pop(L, None)
                print(f'long: layer {L} {time.time() - t0:.0f}s', flush=True)
            rec = Recorder()
            ref.rec = rec
            ref.layer(s, 3, SeqState())
        rec.t['tokens'] = torch.tensor(long_tok)
        rec.save(os.path.join(a.out, 'long-L03.safetensors'), dict(meta, layers=[3], long=n,
                 note='layer 3 on a repeated prompt; inputs from layers 0-2 (chunked KDA)'))
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
