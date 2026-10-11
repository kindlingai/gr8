"""Exact comparison of gr8's exlr8 dense decoder (src/exlr8.rs, through the exlr8dump binary) against the reference
loader `loader/exlr8.py` of kindlingai/exlr8, on the release's own files.

    python3 exlr8_compare.py --release DIR --ref EXLR8_REPO --dump target/release/exlr8dump \
        --layers 0,3,4,6,78,vocab [--shards 3,4] [--work DIR]

For each layer (or vocab) it decodes every matrix with both, in fp32 and in BF16, and requires the bits to be equal
(not just the values: -0 and +0 count as different). With --shards N, it also decodes every rank's block at TP=N
(the sharded side split into whole units, larger runs first) and compares it with the same block of the reference's
full matrix. Norms and biases must come back byte for byte. Needs torch (the reference's only dependency).
"""
import argparse
import collections
import os
import subprocess
import sys
import time

import torch


def bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True, help='a copy of the release (dense/ under it, or the dense files)')
    ap.add_argument('--ref', required=True, help='a kindlingai/exlr8 checkout (loader/exlr8.py)')
    ap.add_argument('--dump', required=True, help='the exlr8dump binary')
    ap.add_argument('--layers', required=True, help='comma list of layer numbers and/or vocab')
    ap.add_argument('--shards', default='', help='comma list of TP sizes whose rank blocks are compared too')
    ap.add_argument('--work', default='/tmp/exlr8-compare')
    a = ap.parse_args()
    sys.path.insert(0, a.ref)
    from loader import exlr8 as X
    os.makedirs(a.work, exist_ok=True)
    tally = collections.defaultdict(lambda: [0, 0])     # what -> [tensors, elements]
    fails = []

    def dump(which, dtype, shard=None):
        out = os.path.join(a.work, f'{which}.{dtype}' + (f'.tp{shard[0]}r{shard[1]}' if shard else '') + '.safetensors')
        cmd = [a.dump, a.release, which, out, '--dtype', dtype] + (['--shard', str(shard[0]), str(shard[1])] if shard else [])
        subprocess.run(cmd, check=True)
        f = X.SafeFile(out)
        got = {n: f.get(n) for n in f.names()}
        os.unlink(out)
        return got, f.meta

    def same(what, name, want, got):
        if want.shape != got.shape or want.dtype != got.dtype or not torch.equal(bits(want), bits(got)):
            n = (bits(want) != bits(got)).sum().item() if want.shape == got.shape and want.dtype == got.dtype else None
            fails.append((what, name, list(want.shape), list(got.shape), n))
            print(f'  MISMATCH {what} {name}: want {list(want.shape)} {want.dtype}, got {list(got.shape)} {got.dtype}, '
                  f'{n} elements differ', flush=True)
            return
        tally[what][0] += 1
        tally[what][1] += want.numel()

    for which in a.layers.split(','):
        t0 = time.time()
        if which == 'vocab':
            ref32 = X.load_vocab(a.release, torch.float32)
        else:
            L = int(which)
            f = X.SafeFile(os.path.join(a.release, 'dense', f'exlr8-dense-{L:03d}.safetensors'))
            ref32 = X.load_matrices(f, lambda n: f'model.layers.{L}.' + n, torch.float32, 'cpu', False)
        print(f'{which}: reference decoded in {time.time() - t0:.1f} s', flush=True)
        mats = {k for k, v in ref32.items() if k.endswith('.weight') and v.dim() == 2}
        for dtype in ('f32', 'bf16'):
            got, _ = dump(which, dtype)
            if set(got) != set(ref32):
                fails.append((which, 'names', sorted(set(ref32) ^ set(got)), None, None))
                print(f'  MISMATCH names: {sorted(set(ref32) ^ set(got))}')
            for k in sorted(set(got) & set(ref32)):
                if k in mats:
                    want = ref32[k] if dtype == 'f32' else ref32[k].to(torch.bfloat16)
                    same(f'matrix {dtype}', k, want, got[k])
                elif dtype == 'f32':
                    same('vector as stored', k, ref32[k], got[k])
        for n in filter(None, a.shards.split(',')):
            for r in range(int(n)):
                got, meta = dump(which, 'f32', (int(n), r))
                for k in sorted(mats):
                    if k not in meta:
                        continue                      # replicated: the full matrix, compared above
                    rr, cc = meta[k].split()[1], meta[k].split()[3]
                    (r0, r1), (c0, c1) = (map(int, rr.split(':')), map(int, cc.split(':')))
                    same(f'TP{n} rank block f32', f'{k}[r{r}]', ref32[k][r0:r1, c0:c1].contiguous(), got[k])
        print(f'{which}: done in {time.time() - t0:.1f} s', flush=True)
        del ref32

    print('\n| compared | tensors | elements |\n|---|---:|---:|')
    for what, (n, e) in sorted(tally.items()):
        print(f'| {what} | {n} | {e:,} |')
    if fails:
        print(f'\nFAILED: {len(fails)} mismatches')
        sys.exit(1)
    print('\nall equal, bit for bit')


if __name__ == '__main__':
    main()
