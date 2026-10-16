"""CPU pilot gate: every rank-image expert byte and side vector, plus sampled numerical reconstruction.
Run from repo root with --release (real layer-3 pilot plus TP4 slices/manifests), --ref EXLR8_REPO.
No CUDA. Each rank dump is removed after verification. This is not a full-model inference test.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.convert.st import ST


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layer', type=int, default=3, help='expert layer to check (default: 3)')
    ap.add_argument('--release', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--dump', default='engine/loader-rs/target/release/g6plandump')
    ap.add_argument('--fetch', default='engine/loader-rs/target/release/g6load')
    ap.add_argument('--work', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.ref)
    from loader import exlr8 as X
    torch.set_num_threads(1)
    root, work = Path(a.release), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    src = X.ExpertLayer(str(root), a.layer)
    counts = dict(halves=0, decoded=0, bytes=0, checksums=0)
    for k in (2, 3, 4):
        manifest = root / f'manifests/exlr8-manifest-k{k}.json'
        for rank in range(4):
            check = subprocess.check_output([a.fetch, 'verify', '--slices', str(root/'index-slices'),
                '--v2',str(root/'experts'),'--tp','4','--rank',str(rank),'--manifest',str(manifest)], text=True)
            checked = json.loads(check)
            assert not checked['bad'], checked
            counts['checksums'] += checked['checked']
            out = work/f'k{k}.r{rank}.safetensors'
            subprocess.run([a.dump,str(root),str(root),str(root/'index-slices'),str(manifest),'4',str(rank),str(out),
                '--layers',str(a.layer),'--experts-only'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f = ST(out)
            p = f'L{a.layer:03d}.tr3.'
            table, kmix = f.get(p+'table'), f.get(p+'kmix')
            sg, v1, v2, v3 = [f.get(p+n) for n in ('sg','v1b','v2','v3b')]
            assert table.shape == (289,6) and not kmix.any()
            assert np.array_equal(sg, np.stack([src.t['signs.in'].numpy(),src.t['signs.out'].numpy()]))
            hb = 65536*k
            off1 = off2 = row = 0
            for e in range(289):
                hs = [h for h in range(16) if (16*e+h+a.layer)%4 == rank]
                assert table[e].tolist() == [off1,off2,row,len(hs),k,hs[0]]
                pre, j = ('shared.',0) if e == 288 else ('',e)
                key = f'{pre}k{k}.'
                for h in hs:
                    for pi, proj in enumerate(('gate','up','down')):
                        ix = src.t[key+'index'][j,pi,h]
                        raw = src.trellis_file(k).read(int(ix[0]), int(ix[1])).numpy()
                        got = f.read(p+('fc2' if pi==2 else 'fc1'), off2 if pi==2 else off1+pi*hb,hb)
                        expected = raw if pi<2 else raw.reshape(32,8,8,32*k).transpose(0,2,1,3).copy().reshape(-1)
                        assert np.array_equal(got,expected), (k,rank,e,h,proj,'trellis')
                        chan = src.t[key+('down.suh' if pi==2 else proj+'.svh')][j,h*128:(h+1)*128].numpy()
                        block = src.t[key+('down.sv_block' if pi==2 else proj+'.su_block')][j].numpy()
                        assert np.array_equal(v2[row,pi],chan)
                        assert np.array_equal(v3[row] if pi==2 else v1[row,pi],block)
                        counts['halves'] += 1
                        counts['bytes'] += hb
                        if e in (0,287,288) and h in (hs[0],hs[-1]):
                            # Decode the planned bytes and planned vectors, not the original source bytes.
                            b = got if pi<2 else got.reshape(32,8,8,32*k).transpose(0,2,1,3).copy().reshape(-1)
                            t = torch.from_numpy(b.copy())
                            c = torch.from_numpy(v2[row,pi].copy())
                            bs = torch.from_numpy((v3[row] if pi==2 else v1[row,pi]).copy()).float()
                            signs = torch.from_numpy(sg[1 if pi==2 else 0].copy()).float()
                            hid = signs * bs.repeat_interleave(128)
                            decoded = (X.reconstruct(X.tiles_in_major(t,k,4096),c,hid) if pi==2 else
                                       X.reconstruct(X.tiles_out_major(t,k,4096,128),hid,c))
                            reference = src.half(e,proj,h,k,verify=True)
                            assert torch.equal(decoded,reference), (k,rank,e,h,proj,'dequant')
                            counts['decoded'] += 1
                    off1 += 2*hb
                    off2 += hb
                    row += 1
            f.close()
            out.unlink()
            print(json.dumps(dict(k=k,rank=rank,**counts)),flush=True)
    print('PASS '+json.dumps(counts),flush=True)

if __name__ == '__main__':
    main()
