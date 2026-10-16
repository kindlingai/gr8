"""CPU-only W3 fixture checks. Does not execute or certify CUDA kernels."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from check_mlp import compact, header


class Fixture(unittest.TestCase):
    def test_compact_preserves_names_dtypes_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'src.st', Path(d)/'dst.st'
            values = {'L003.router.w': torch.arange(64).reshape(8,8).bfloat16(),
                      'L003.router.bias': torch.arange(8).float()/16,
                      'L003.tr3.table': torch.arange(18).reshape(3,6),
                      'L003.tr3.fc1': torch.arange(128, dtype=torch.uint8),
                      'L003.other': torch.ones(9), 'L000.dense.gate': torch.ones(8,8)}
            save_file(values,str(src));compact(src,dst,3)
            with safe_open(dst,framework='pt',device='cpu') as f:
                self.assertEqual(set(f.keys()),set(values)-{'L003.other','L000.dense.gate'})
                for n in f.keys():
                    self.assertEqual(f.get_tensor(n).dtype,values[n].dtype)
                    self.assertTrue(torch.equal(f.get_tensor(n),values[n]))
            base,head=header(dst)
            for line in Path(str(dst)+'.meta').read_text().splitlines():
                parts=line.split();name,dt,nd=parts[:3];nd=int(nd)
                shape=list(map(int,parts[3:3+nd]));off,n=map(int,parts[3+nd:])
                self.assertEqual(dt,head[name]['dtype']);self.assertEqual(shape,head[name]['shape'])
                lo,hi=head[name]['data_offsets'];self.assertEqual(off,base+lo);self.assertEqual(n,hi-lo)

    def test_compact_rejects_wrong_layer(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'src.st';save_file({'L000.other':torch.ones(1)},str(src))
            with self.assertRaises(ValueError):compact(src,Path(d)/'dst.st',3)


def dense_cpu_check(case):
    """Verify actual W1 TP4 dense shard names/layouts against the W0 golden on CPU."""
    from flash_ref import swiglu
    torch.set_num_threads(1)
    p=Path(case);meta=json.loads((p/'case.json').read_text());M=meta['rows'];l=meta['layer'];off=meta['offset']
    x=torch.from_numpy(np.fromfile(p/'input.f32',np.float32).reshape(M,4096).copy())
    with safe_open(meta['golden'],framework='pt',device='cpu') as f:
        expected=f.get_tensor(f'L{l:02d}.mlp.out')[off:off+M]
    y=torch.zeros_like(x);ybf=torch.zeros_like(x)
    for rank in range(4):
        with safe_open(p/f'rank{rank}.safetensors',framework='pt',device='cpu') as f:
            g,u,d=[f.get_tensor(f'L{l:03d}.dense.{n}').float() for n in ('gate','up','down')]
        part=swiglu(x@g.T,x@u.T,10.)@d.T
        y+=part;ybf+=part.bfloat16().float()
    for name,got,tol in [('fp32',y,1e-4),('bf16 rank partials',ybf,1e-2)]:
        rel=float((got.double()-expected.double()).norm()/expected.double().norm())
        if rel>tol:raise AssertionError((name,rel))
        print(f'CPU ONLY layer {l}, TP4 {name}, M={M}: W0 rel-L2={rel:.8g} PASS (not a GPU test)')


if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--dense-case':dense_cpu_check(sys.argv[2])
    else:unittest.main()
