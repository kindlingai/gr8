"""CPU gate for real Flash quantized dense layers (no experts or vocabulary).

Compare every TP4 planned tensor to an independent derivation from stored
records, then compare Rust FP32 decodes with the Python dense codec bytewise.
Use --layers to test a partial release. This is not a GPU or model-quality gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engine/ref'), str(Path(__file__).parent)]
import g6ref
from release_check import FlashExpect
from tools.convert import plan


def expected_layer(f, layer, rank, D):
    names = json.loads(f.metadata()['source_names'])
    expected = {}
    h0, nh = plan.heads(layer)[rank]
    g0, ng = plan.dense_groups(layer)[rank]
    for local, full in names.items():
        hf = full.split(f'layers.{layer}.', 1)[1]
        label = f'L{layer:03}.' + FlashExpect.ALIAS.get(hf, hf)
        if local+'.record' not in f.keys():
            expected[label] = f.get_tensor(local)
            continue
        rec = f.get_tensor(local+'.record').numpy()
        data = {k: f.get_tensor(local+'.'+k) for k in D.data_names(int(rec[0]))}
        if rec[0] == D.TRELLIS:
            if local.startswith('mlp.'):
                ch = slice(g0*32, (g0+ng)*32)
            else:
                ch = slice(h0*int(rec[5]), (h0+nh)*int(rec[5]))
            expected[label+'.trellis'] = data['trellis'][ch.start//128:ch.stop//128]
            expected[label+'.suh'] = data['suh'][ch] if rec[4] == D.IN else data['suh']
            expected[label+'.svh'] = data['svh'] if rec[4] == D.IN else data['svh'][ch]
        elif rec[0] == D.MXFP8:
            expected[label+'.mx.codes'] = data['fp8']
            expected[label+'.mx.scales'] = data['e8m0']
            expected[label+'.mx.gv'] = torch.from_numpy(g6ref.mxfp8_record(data['fp8'].numpy(), data['e8m0'].numpy()))
        elif rec[0] == D.FP8_BLOCK:
            assert local == 'self_attn.kv_b_proj'
            w = D.dequant(rec, data)[0].bfloat16().view(64,512,512)[h0:h0+nh]
            expected[f'L{layer:03}.kv_b.uk'] = w[:,:256]
            expected[f'L{layer:03}.kv_b.uv'] = w[:,256:]
        elif rec[0] in (D.BF16, D.FP32):
            w = data[D.data_names(int(rec[0]))[0]]
            expected[label] = w
            if w.dtype == torch.bfloat16 and w.ndim == 2 and all(d%16 == 0 for d in w.shape):
                expected[label+'.gv'] = torch.from_numpy(g6ref.bf16_record(w.view(torch.int16).numpy().view(np.uint16)))
        else:
            raise AssertionError((local, rec.tolist()))
    return expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--release', required=True)
    ap.add_argument('--layers', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.encoder)/'requant/dense'))
    import dense_format as D
    torch.set_num_threads(2)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    plan.set_model('flash'); plan.set_nr(4)
    release, work = Path(a.release), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(',')]
    with safe_open(str(release/f'dense/exlr8-dense-{layers[0]:03}.safetensors'), 'pt') as f:
        meta = {k:f.metadata()[k] for k in D.MODEL_KEYS}
    man = work/'widths.json'
    man.write_text(json.dumps(dict(meta, format='exlr8-v2-manifest', layers={})))
    for rank in range(4):
        path = work/f'index-slices/tp4/k2/rank{rank}.safetensors'
        path.parent.mkdir(parents=True, exist_ok=True); save_file({}, str(path))
    h = torch.ones(1,1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
    h = h/(128**.5)
    class Q:
        had_k = had_n = 128
        @staticmethod
        def preapply_had_l(w,n): return (h@w.reshape(-1,128,w.shape[1])).reshape(w.shape)
        @staticmethod
        def preapply_had_r(w,n): return (w.reshape(w.shape[0],-1,128)@h).reshape(w.shape)
    dump = ROOT/'engine/loader-rs/target/release/g6plandump'
    decode = ROOT/'engine/loader-rs/target/release/exlr8dump'
    receipts = []
    for layer in layers:
        tensors = matrices = 0
        with safe_open(str(release/f'dense/exlr8-dense-{layer:03}.safetensors'), 'pt') as f:
            assert {k:f.metadata()[k] for k in D.MODEL_KEYS} == meta
            for rank in range(4):
                out = work/'image.safetensors'
                cmd = [str(dump), str(release), str(release), str(work/'index-slices'), str(man),
                       '4', str(rank), str(out), '--partial', '--layers', str(layer)]
                subprocess.run(cmd, check=True)
                expected = expected_layer(f, layer, rank, D)
                with safe_open(str(out), 'pt') as got:
                    assert set(got.keys()) == set(expected), (layer,rank,set(got.keys())^set(expected))
                    for name, w in expected.items():
                        assert got.get_tensor(name).contiguous().view(torch.uint8).equal(w.contiguous().view(torch.uint8)), (layer,rank,name)
                        tensors += 1
                del expected
                out.unlink()
                print(f'PASS image layer={layer} rank={rank}', flush=True)
            out = work/'decoded.safetensors'
            subprocess.run([str(decode),str(release),str(layer),str(out),'--threads','2'], check=True)
            with safe_open(str(out),'pt') as got:
                for key in f.keys():
                    if not key.endswith('.record'): continue
                    local = key[:-7]
                    rec = f.get_tensor(key).numpy()
                    data = {k:f.get_tensor(local+'.'+k) for k in D.data_names(int(rec[0]))}
                    want = D.dequant(rec, data, Q)[0]
                    actual = got.get_tensor(f'model.layers.{layer}.{local}.weight')
                    assert actual.view(torch.int32).equal(want.view(torch.int32)), (layer,local,'FP32 decode bytes')
                    matrices += 1
                    del actual, want, data
            out.unlink()
        receipts.append(dict(layer=layer, tensors=tensors, matrices=matrices, status='PASS'))
        Path(a.report).write_text(json.dumps(dict(status='INCOMPLETE', layers=receipts), indent=2)+'\n')
        print(f'PASS layer={layer} tensors={tensors} FP32_matrices={matrices}', flush=True)
    result = dict(status='PASS', layers=receipts, tp=4, ranks=[0,1,2,3],
                  planned_tensors=sum(r['tensors'] for r in receipts),
                  fp32_matrices=sum(r['matrices'] for r in receipts), globals=False, experts=False)
    Path(a.report).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
