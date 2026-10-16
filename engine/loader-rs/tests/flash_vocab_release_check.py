"""CPU gate for real Flash vocabulary: TP4 global images and FP32 decodes.

Read real FP8-row records in small chunks. Check all four ranks' replicated
BF16 embeddings, sharded MXFP8 heads and final norms against Python bytes.
This does not test GPU kernels, experts or model quality.
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

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engine/ref'),str(Path(__file__).parent)]
import g6ref
from release_check import mx_quantize
from tools.convert import plan


def decoded(f, name, lo, hi, D):
    return D.dequant_fp8_row(f.get_slice(name+'.fp8')[lo:hi],f.get_slice(name+'.scale')[lo:hi])


def check_rank(release,work,man,rank,D):
    path=release/'dense/exlr8-vocab.safetensors'
    out=work/'vocab-image.safetensors'
    dump=ROOT/'engine/loader-rs/target/release/g6plandump'
    subprocess.run([str(dump),str(release),str(release),str(work/'index-slices'),str(man),
                    '4',str(rank),str(out),'--partial','--globals'],check=True)
    with safe_open(str(path),'pt') as f, safe_open(str(out),'pt') as got:
        assert set(got.keys())=={'G.embed','G.final_norm','G.lm_head.r8.gv','G.lm_head.scale'}
        rows,cols=json.loads(f.metadata()['source_shapes'])['embed_tokens']
        assert (rows,cols)==(154880,4096)
        assert got.get_slice('G.embed').get_shape()==[rows,cols]
        for a in range(0,rows,1024):
            want=decoded(f,'embed_tokens',a,min(a+1024,rows),D).bfloat16()
            assert got.get_slice('G.embed')[a:a+1024].view(torch.uint8).equal(want.view(torch.uint8)),(rank,a,'embedding')
        assert got.get_tensor('G.final_norm').view(torch.uint8).equal(f.get_tensor('norm.weight').view(torch.uint8))
        start,n=plan.vocab_rows(rank)
        assert got.get_tensor('G.lm_head.scale').view(torch.uint8).equal(f.get_slice('lm_head.scale')[start:start+n].view(torch.uint8))
        assert got.get_slice('G.lm_head.r8.gv').get_shape()==[n//16,128,512]
        for a in range(0,n,1024):
            end=min(a+1024,n)
            codes=f.get_slice('lm_head.fp8')[start+a:start+end].numpy()
            expected=g6ref.fp8_record(codes)
            assert np.array_equal(got.get_slice('G.lm_head.r8.gv')[a//16:end//16].numpy(),expected),(rank,a,'head FP8-row')
    out.unlink()
    out=work/'vocab-decoded.safetensors'
    decoder=ROOT/'engine/loader-rs/target/release/exlr8dump'
    subprocess.run([str(decoder),str(release),'vocab',str(out),'--shard','4',str(rank),'--threads','2'],check=True)
    with safe_open(str(path),'pt') as f, safe_open(str(out),'pt') as got:
        # exlr8dump uses standard HF aliases; the model planner uses Flash names.
        assert set(got.keys()) == {'model.embed_tokens.weight','lm_head.weight','model.norm.weight'}
        assert got.get_tensor('model.norm.weight').view(torch.uint8).equal(f.get_tensor('norm.weight').view(torch.uint8))
        for name,hf in [('embed_tokens','model.embed_tokens.weight'),('lm_head','lm_head.weight')]:
            start,n=plan.vocab_rows(rank)
            assert got.get_slice(hf).get_shape()==[n,4096]
            for a in range(0,n,1024):
                w=decoded(f,name,start+a,min(start+a+1024,start+n),D)
                assert got.get_slice(hf)[a:a+1024].view(torch.int32).equal(w.view(torch.int32)),(rank,name,a,'FP32')
    out.unlink()
    return dict(rank=rank,status='PASS',planned_tensors=4,fp32_matrix_shards=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('encoder','release','work','report'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(Path(a.encoder)/'requant/dense'));import dense_format as D
    torch.set_num_threads(2);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    plan.set_model('flash');plan.set_nr(4)
    release,work=Path(a.release),Path(a.work);work.mkdir(parents=True,exist_ok=True)
    with safe_open(str(release/'dense/exlr8-vocab.safetensors'),'pt') as f:
        meta={k:f.metadata()[k] for k in D.MODEL_KEYS}
    man=work/'widths.json';man.write_text(json.dumps(dict(meta,format='exlr8-v2-manifest',layers={})))
    for rank in range(4):
        path=work/f'index-slices/tp4/k2/rank{rank}.safetensors';path.parent.mkdir(parents=True,exist_ok=True)
        save_file({},str(path))
    results=[]
    for rank in range(4):
        results.append(check_rank(release,work,man,rank,D))
        Path(a.report).write_text(json.dumps(dict(status='INCOMPLETE',ranks=results),indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
    result=dict(status='PASS',tp=4,ranks=results,planned_tensors=16,fp32_matrix_shards=8,
                unique_matrices=2,globals=True,experts=False)
    Path(a.report).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
