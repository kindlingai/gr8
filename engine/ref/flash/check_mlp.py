"""Prepare W1 rank images and compare the standalone W3 GPU test to W0 real-weight goldens.

prepare runs on CPU, one rank at a time. run is on the assigned Spark. compare runs on CPU.
No source weights, goldens, or model releases are modified. See engine/kernels/FLASH-MLP.md.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def compact(source, dest, layer):
    """Copy only block weights, streaming bytes; never dequantize or round them."""
    base, h = header(source)
    prefix = f'L{layer:03d}.'
    names = [n for n in h if n.startswith(prefix) and
             (n[len(prefix):].startswith('tr3.') or n[len(prefix):] in
              ('router.w', 'router.bias', 'dense.gate', 'dense.up', 'dense.down'))]
    if not names:
        raise ValueError('rank image contains no MLP tensors')
    out, off = {}, 0
    for n in names:
        v = h[n]; size = v['data_offsets'][1] - v['data_offsets'][0]
        out[n] = dict(dtype=v['dtype'], shape=v['shape'], data_offsets=[off, off + size])
        off += size
    raw = json.dumps(out, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with open(source, 'rb') as src, open(dest, 'wb') as dst:
        dst.write(struct.pack('<Q', len(raw))); dst.write(raw)
        for n in names:
            lo, hi = h[n]['data_offsets']; src.seek(base + lo)
            remain = hi - lo
            while remain:
                part = src.read(min(8 << 20, remain))
                if not part:
                    raise EOFError(n)
                dst.write(part); remain -= len(part)
    with open(str(dest) + '.meta', 'w') as f:
        for n, v in out.items():
            lo, hi = v['data_offsets']
            f.write(' '.join(map(str, [n, v['dtype'], len(v['shape']), *v['shape'], 8 + len(raw) + lo, hi - lo])) + '\n')


def prepare(a):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    key = f'L{a.layer:02d}.mlp_in'
    with safe_open(a.golden, framework='pt', device='cpu') as f:
        x = f.get_tensor(key)[a.offset:a.offset + a.rows].float().contiguous()
        if x.shape != (a.rows, 4096):
            raise ValueError(f'{key}: input shape {tuple(x.shape)} does not match rows')
        x.numpy().tofile(out / 'input.f32')
        if a.layer >= 3:
            f.get_tensor(f'L{a.layer:02d}.moe.router_logits')[a.offset:a.offset+a.rows].float().contiguous().numpy().tofile(out/'input.f32.router.f32')
    release = Path(a.release)
    for r in range(4):
        dest = out / f'rank{r}.safetensors'
        temp = out / f'dump-rank{r}.safetensors'
        cmd = [a.dump, str(release), str(release), str(release / 'index-slices'), a.widths,
               '4', str(r), str(temp), '--layers', str(a.layer), '--partial', '--experts']
        with open(out / f'prepare-r{r}.log', 'w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=log)
        compact(temp, dest, a.layer)
        temp.unlink()
        print(f'rank {r}: {dest.stat().st_size} bytes of exact W1 MLP weights', flush=True)
    meta = dict(golden=str(Path(a.golden).resolve()), release=str(release.resolve()),
                widths=str(Path(a.widths).resolve()), layer=a.layer, offset=a.offset, rows=a.rows,
                input_key=key, tp=4)
    (out / 'case.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('Prepared CPU fixture; no GPU result claimed.')


def run(a):
    p = Path(a.case); meta = json.loads((p/'case.json').read_text())
    for rank in range(4):
        subprocess.run([a.binary, str(p/f'rank{rank}.safetensors'), str(p/'input.f32'),
                        str(meta['layer']), str(meta['rows']), str(p/f'rank{rank}.f{a.fuse}'), str(a.fuse)], check=True)


def compare(a):
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    p = Path(a.case); meta = json.loads((p/'case.json').read_text())
    layer, off = meta['layer'], meta['offset']
    tag = f'L{layer:02d}.moe' if layer >= 3 else f'L{layer:02d}.mlp'
    with safe_open(a.golden or meta['golden'], framework='pt', device='cpu') as f:
        target = f.get_tensor(tag+'.out')[off:off+meta['rows']]
        logits = f.get_tensor(tag+'.router_logits')[off:off+meta['rows']] if layer >= 3 else None
    def read(path, dtype, shape):
        return torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape).copy())
    def rel(x, y):
        return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))
    def gate(name, x, y, tol):
        err = rel(x, y)
        print(f'{name}: rel-L2={err:.8g} gate={tol:g} '+('PASS' if err <= tol else 'FAIL'), flush=True)
        if not np.isfinite(err) or err > tol:
            raise AssertionError(name)
    # Bias is needed for tie-aware, order-independent router checks.
    if layer >= 3:
        with safe_open(p/'rank0.safetensors', framework='pt', device='cpu') as f:
            bias = f.get_tensor(f'L{layer:03d}.router.bias')
    for M in (1, 3, 8, 9, 16, 32):
        if M > meta['rows']:
            continue
        sums = {}; full32 = torch.zeros(M,4096)
        for rank in range(4):
            pre = p / f'rank{rank}.f{a.fuse}.m{M}'
            kinds = ['out'] + (['fixed_route.out'] if layer >= 3 else [])
            for kind in kinds:
                partial = read(str(pre)+f'.{kind}.bf16',np.uint16,(M,4096)).view(torch.bfloat16).float()
                sums[kind] = sums.get(kind, torch.zeros_like(partial)) + partial
            if layer < 3:
                full32 += read(str(pre)+'.out.f32',np.float32,(M,4096))
                continue
            got = read(str(pre)+'.logits.f32',np.float32,(M,288))
            gate(f'M={M} rank={rank} router logits',got,logits[:M],1e-4)
            idx = read(str(pre)+'.route_input.topk.i32',np.int32,(M,8)).long()
            wts = read(str(pre)+'.route_input.weights.f32',np.float32,(M,8))
            scores = logits[:M].sigmoid(); choice = scores+bias
            selected = choice.gather(1,idx)
            # Equal choice scores permit different expert ids; all distinct choices must match exactly.
            torch.testing.assert_close(selected.sort(-1).values,choice.topk(8,-1).values.sort(-1).values,rtol=0,atol=0)
            if any(len(set(row)) != 8 for row in idx.tolist()):
                raise AssertionError('duplicate router expert')
            expected = scores.gather(1,idx); expected=expected/(expected.sum(-1,keepdim=True)+1e-20)*2.5
            gate(f'M={M} rank={rank} router weights (identical logits)',wts,expected,1e-4)
        if layer < 3:
            gate(f'M={M} TP4 dense FP32 before partial rounding',full32,target[:M],1e-4)
        for kind, value in sums.items():
            if layer >= 3 and kind == 'out':
                print(f'M={M} TP4 chained-router MoE rel-L2={rel(value,target[:M]):.8g} REPORT (not the per-op gate)')
            else:
                gate(f'M={M} TP4 {kind} BF16 partials',value,target[:M],1e-2)
    print('W3 real-weight per-op gate PASS')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('release','widths','dump','golden','out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--layer', type=int, choices=(0,1,2,3), required=True)
    p.add_argument('--rows',type=int,default=32); p.add_argument('--offset',type=int,default=0)
    p.set_defaults(fn=prepare)
    p = sub.add_parser('run'); p.add_argument('--case', required=True); p.add_argument('--binary',required=True)
    p.add_argument('--fuse',type=int,choices=(0,1,2),default=0);p.set_defaults(fn=run)
    p = sub.add_parser('compare'); p.add_argument('--case',required=True);p.add_argument('--golden')
    p.add_argument('--fuse',type=int,choices=(0,1,2),default=0);p.set_defaults(fn=compare)
    a = ap.parse_args();a.fn(a)


if __name__ == '__main__':
    main()
