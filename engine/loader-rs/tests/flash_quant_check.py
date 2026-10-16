"""CPU W8 gate: independently derive quantized Flash rank images and decode bytes.

python3 engine/loader-rs/tests/flash_quant_check.py --encoder EXLR8QUANT --work DIR
Synthetic matrix inputs are narrow to keep the test small; sharded axes retain
all 64 heads / 12288 dense channels. kv_b retains its full Flash geometry.
No expert or GPU work. --vocab also tests full-size FP8-row vocab at rank 0.
"""
import argparse
import json
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
from tools.convert.test_flash_v2 import META


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--encoder',required=True)
    ap.add_argument('--work',required=True)
    ap.add_argument('--vocab',action='store_true')
    a=ap.parse_args()
    sys.path.insert(0,str(Path(a.encoder)/'requant/dense'))
    import dense_format as D
    root=Path(a.work); (root/'dense').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(82)
    man=root/'widths.json';man.write_text(json.dumps(dict(META,format='exlr8-v2-manifest',layers={})))
    for rank in range(4):
        p=root/f'index-slices/tp4/k2/rank{rank}.safetensors';p.parent.mkdir(parents=True,exist_ok=True)
        save_file({},str(p))
    plan.set_model('flash');plan.set_nr(4)
    sources={}
    h=torch.ones(1,1)
    while h.shape[0]<128:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
    h=h/(128**0.5)
    class Q:
        had_k=had_n=128
        @staticmethod
        def preapply_had_l(w,n):return (h@w.reshape(-1,128,w.shape[1])).reshape(w.shape)
        @staticmethod
        def preapply_had_r(w,n):return (w.reshape(w.shape[0],-1,128)@h).reshape(w.shape)

    def matrix(t,names,shapes,l,name,shape,fmt,side=0,unit=0,rot=0):
        s=D.Spec(name,None,fmt,side,unit,rot=rot)
        rec=D.record(fmt,*shape,4,s,1 if fmt in (1,3) else 0)
        if fmt==1:
            tiles=torch.randint(-32768,32767,(shape[1]//16,shape[0]//16,64),dtype=torch.int16)
            data={'trellis':D.trellis_entries(tiles,side),'suh':torch.ones(shape[1]).half()*0.25,
                  'svh':torch.ones(shape[0]).half()*0.5}
            if rot: data['suh'].view(-1,128)[1::2]*=-1
        elif fmt==3:
            data={'fp8':torch.randint(0,120,shape,dtype=torch.uint8),
                  'e8m0':torch.randint(118,128,(shape[0],shape[1]//32),dtype=torch.uint8)}
        elif fmt==4:
            data={'fp8':torch.randint(0,120,shape,dtype=torch.uint8),
                  'scale':torch.rand((shape[0]+127)//128,(shape[1]+127)//128)*0.05+0.001}
        elif fmt in (6,7):
            data={D.data_names(fmt)[0]:torch.randn(shape).to(torch.bfloat16 if fmt==6 else torch.float32)}
        else:raise AssertionError(fmt)
        t[name+'.record']=torch.from_numpy(rec)
        t.update({name+'.'+k:v for k,v in data.items()})
        raw={k:(v.view(torch.int16).numpy().view(np.uint16) if v.dtype==torch.bfloat16 else v.numpy()) for k,v in data.items()}
        t[name+'.checksum']=torch.from_numpy(D.checksums(rec,raw))
        full=name if name.startswith('hc_') else name+'.weight'
        names[name]=f'model.language_model.layers.{l}.'+full;shapes[name]=list(shape)
        return rec,data

    for l in (0,3,45):
        t,names,shapes={},{},{}
        specs=[]
        if l==0:
            specs += [(f'self_attn.{n}_proj',(8192,128),1,1,128,1) for n in ('q','k','v')]
            specs += [('self_attn.o_proj',(128,8192),1,2,128,0)]
            specs += [(f'self_attn.{n}_a_proj',(128,128),3,0,0,0) for n in ('f','g')]
            specs += [(f'mlp.{n}_proj',(12288,128),1,1,128,3) for n in ('gate','up')]
            specs += [('mlp.down_proj',(128,12288),1,2,128,0)]
        else:
            specs += [('self_attn.q_b_proj',(16384,128),1,1,256,2),('self_attn.o_proj',(128,16384),1,2,256,0),
                      ('self_attn.kv_b_proj',(32768,512),4,1,512,0)]
            specs += [(n,(128,128),3,0,0,0) for n in ('self_attn.q_a_proj','self_attn.kv_a_proj_with_mqa',
                      'self_attn.indexer.wq_b','self_attn.indexer.wk')]
            if l==45:specs += [('eh_proj',(128,256),3,0,0,0)]
        specs += [('hc_attn_fn',(24,128),7,0,0,0),('self_attn.b_proj',(64,128),6,0,0,0)]
        records={}
        for args in specs: records[args[0]]=matrix(t,names,shapes,l,*args)
        raw='self_attn.q_conv1d.weight' if l==0 else 'self_attn.indexer.index_kpool_compress_ape'
        t[raw]=torch.randn(128,1,4).bfloat16();names[raw]=f'model.language_model.layers.{l}.'+raw;shapes[raw]=[128,1,4]
        meta=dict(META,dense_model='flash',layer=str(l),attention='kda' if l==0 else 'dsa',
                  source_names=json.dumps(names),source_shapes=json.dumps(shapes))
        path=root/f'dense/exlr8-dense-{l:03}.safetensors';save_file(t,str(path),metadata=meta)
        sources[l]=(records,t,names)
    dump=ROOT/'engine/loader-rs/target/release/g6plandump'
    decode=ROOT/'engine/loader-rs/target/release/exlr8dump'
    from release_check import FlashExpect
    tensors=0;matrices=0
    for l,(records,t,names) in sources.items():
        for rank in range(4):
            out=root/'image.safetensors'
            cmd=[str(dump),str(root),str(root),str(root/'index-slices'),str(man),'4',str(rank),str(out),'--partial','--layers',str(l)]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            expected={};h0,nh=plan.heads(l)[rank];g0,ng=plan.dense_groups(l)[rank]
            for n,(rec,data) in records.items():
                hf=names[n].split(f'layers.{l}.')[1]
                alias=FlashExpect.ALIAS.get(hf,hf);label=f'L{l:03}.'+alias
                if rec[0]==1:
                    if n.startswith('mlp.'):ch=slice(g0*32,(g0+ng)*32)
                    else:ch=slice(h0*int(rec[5]),(h0+nh)*int(rec[5]))
                    expected[label+'.trellis']=data['trellis'][ch.start//128:ch.stop//128]
                    expected[label+'.suh']=data['suh'][ch] if rec[4]==2 else data['suh']
                    expected[label+'.svh']=data['svh'] if rec[4]==2 else data['svh'][ch]
                elif rec[0]==3:
                    expected[label+'.mx.codes']=data['fp8'];expected[label+'.mx.scales']=data['e8m0']
                    expected[label+'.mx.gv']=torch.from_numpy(g6ref.mxfp8_record(data['fp8'].numpy(),data['e8m0'].numpy()))
                elif rec[0]==4:
                    w=D.dequant(rec,data)[0].bfloat16().view(64,512,512)[h0:h0+nh]
                    expected[f'L{l:03}.kv_b.uk']=w[:,:256];expected[f'L{l:03}.kv_b.uv']=w[:,256:]
                else:
                    w=data[D.data_names(int(rec[0]))[0]];expected[label]=w
                    if w.dtype==torch.bfloat16 and all(d%16==0 for d in w.shape):
                        expected[label+'.gv']=torch.from_numpy(g6ref.bf16_record(w.view(torch.int16).numpy().view(np.uint16)))
            for n in t:
                if n in names and n not in records:expected[f'L{l:03}.'+n]=t[n]
            with safe_open(str(out),'pt') as f:
                assert set(f.keys())==set(expected),(l,rank,set(f.keys())^set(expected))
                for n,w in expected.items():
                    assert f.get_tensor(n).contiguous().view(torch.uint8).equal(w.contiguous().view(torch.uint8)),(l,rank,n)
                    tensors+=1
            out.unlink()
            print(f'PASS image layer={l} rank={rank} tensors={len(expected)}',flush=True)
        # Compare the decoder on narrow synthetic matrices; all record formats included.
        out=root/'decoded.safetensors'
        subprocess.run([str(decode),str(root),str(l),str(out),'--threads','2'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with safe_open(str(out),'pt') as f:
            for n,(rec,data) in records.items():
                want=D.dequant(rec,data,Q)[0]
                got=f.get_tensor(f'model.layers.{l}.{n}.weight')
                assert got.view(torch.int32).equal(want.view(torch.int32)),(l,n,'decoded bytes',float((got-want).abs().max()))
                matrices+=1
        out.unlink()
    # Invalid identity and unsupported rounded path must fail before materialization.
    cmd=[str(dump),str(root),str(root),str(root/'index-slices'),str(man),'4','0',str(root/'bad'),'--partial','--layers','0']
    r=subprocess.run(cmd+['--rounded'],capture_output=True,text=True)
    assert r.returncode and 'requires --trellis' in r.stderr
    bad=dict(META,format='exlr8-v2-manifest',layers={},model='wrong');man.write_text(json.dumps(bad))
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode and 'identity mismatch' in r.stderr
    man.write_text(json.dumps(dict(META,format='exlr8-v2-manifest',layers={})))
    print(f'PASS: {tensors} planned tensors byte-exact; {matrices} decoded matrices byte-exact; 2 rejection checks',flush=True)
    if a.vocab:
        vocab(root,dump,man,D)


def vocab(root,dump,man,D):
    # Build source bytes sequentially, not two vocabulary-sized torch tensors.
    import struct
    rows,cols=154880,4096
    shapes={'embed_tokens':[rows,cols],'lm_head':[rows,cols],'norm.weight':[cols]}
    names={'embed_tokens':'model.language_model.embed_tokens.weight','lm_head':'lm_head.weight','norm.weight':'model.language_model.norm.weight'}
    meta=dict(META,dense_model='flash',source_names=json.dumps(names),source_shapes=json.dumps(shapes))
    tensors=[]
    for n in ('embed_tokens','lm_head'):
        tensors += [(n+'.record','I64',[8],np.array([5,rows,cols,0,1,128,0,0],np.int64).tobytes()),
                    (n+'.fp8','U8',[rows,cols],None),(n+'.scale','F32',[rows],np.full(rows,0.1,np.float32).tobytes())]
    tensors += [('norm.weight','BF16',[cols],np.full(cols,0x3f80,np.uint16).tobytes())]
    head={'__metadata__':meta};off=0
    for n,dt,shape,b in tensors:
        size=len(b) if b is not None else rows*cols
        head[n]={'dtype':dt,'shape':shape,'data_offsets':[off,off+size]};off+=size
    h=json.dumps(head).encode();h+=b' '*(-len(h)%8)
    with (root/'dense/exlr8-vocab.safetensors').open('wb') as f:
        f.write(struct.pack('<Q',len(h)));f.write(h)
        for n,dt,shape,b in tensors:
            if b is not None:f.write(b)
            else:
                # E4M3 value 1.5, then scaled by 0.1. Nonzero tests the rounding.
                chunk=bytes([60])*(1<<20)
                for start in range(0,rows*cols,len(chunk)):f.write(chunk[:min(len(chunk),rows*cols-start)])
    out=root/'vocab-image.safetensors'
    subprocess.run([str(dump),str(root),str(root),str(root/'index-slices'),str(man),'4','0',str(out),'--partial','--globals'],check=True)
    with safe_open(str(out),'pt') as f:
        assert set(f.keys())=={'G.embed','G.final_norm','G.lm_head.r8.gv','G.lm_head.scale'}
        val=torch.tensor(1.5,dtype=torch.float32)*torch.tensor(0.1,dtype=torch.float32)
        for a in range(0,rows,4096):assert (f.get_slice('G.embed')[a:a+4096]==val.bfloat16()).all()
        assert (f.get_tensor('G.final_norm')==1).all()
        n=plan.vocab_rows(0)[1]
        assert (f.get_tensor('G.lm_head.scale')==torch.tensor(0.1,dtype=torch.float32)).all()
        expected=g6ref.fp8_record(np.full((16,cols),60,np.uint8))
        for a in range(0,n//16,128):
            got=f.get_slice('G.lm_head.r8.gv')[a:a+128].numpy()
            assert np.array_equal(got,np.broadcast_to(expected,got.shape))
    out.unlink()
    print('PASS: full-size FP8-row vocabulary, BF16 embedding and MXFP8 head bytes',flush=True)

if __name__=='__main__':main()
