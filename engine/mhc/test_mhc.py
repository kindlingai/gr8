"""CPU C++ mHC vs W0 on identical inputs; optional CUDA binary uses the same cases.

    python engine/mhc/test_mhc.py [--gpu /path/to/mhc_test]

No CUDA execution occurs without --gpu. The CPU test is a math gate, not a GPU gate.
"""
import argparse
import ctypes as C
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ref' / 'flash'))
from flash_ref import Cfg, FlashRef, SeqState, rms
from weights import DictWeights
from test_flash_ref import TINY, hf_model, checkpoint_dict, rel

P = C.POINTER(C.c_float)

def ptr(t):
    assert t.dtype == torch.float32 and t.is_contiguous()
    return C.cast(t.data_ptr(), P)

class Host:
    def __init__(self, lib):
        self.lib = C.CDLL(str(lib))
        self.lib.mhc_prepare.argtypes = [P]*5 + [C.c_int,C.c_int,C.c_float] + [P]*5
        self.lib.mhc_output.argtypes = [P]*4 + [C.c_int,C.c_int]
        self.lib.mhc_final.argtypes = [P,P,C.c_int,C.c_int,C.c_float,P]
    def prepare(self,r,fn,base,scale,norm,eps):
        m,_,h=r.shape
        pre,post,comb,x,n=(torch.empty(s) for s in [(m,4),(m,4),(m,4,4),(m,h),(m,h)])
        self.lib.mhc_prepare(*(ptr(t) for t in [r,fn,base,scale,norm]),m,h,eps,
                             *(ptr(t) for t in [pre,post,comb,x,n]))
        return pre,post,comb,x,n
    def output(self,r,y,post,comb):
        r=r.clone().contiguous()
        self.lib.mhc_output(*(ptr(t) for t in [r,y,post,comb]),r.shape[0],r.shape[2])
        return r
    def final(self,r,norm,eps):
        o=torch.empty(r.shape[0],r.shape[2])
        self.lib.mhc_final(ptr(r),ptr(norm),r.shape[0],r.shape[2],eps,ptr(o))
        return o

def check(name,actual,expected,tol=1e-4):
    err=rel(actual,expected)
    assert torch.isfinite(actual).all() and err <= tol, (name,err,tol)
    print(f'{name}: rel-L2={err:.3g} max={float((actual-expected).abs().max()):.3g} PASS')


def gpu_case(binary,tmp,case,r,fn,base,scale,norm,y,eps,expected):
    m,_,h=r.shape
    inp,out=tmp/f'{case}.in',tmp/f'{case}.out'
    with inp.open('wb') as f:
        f.write(struct.pack('<iif',m,h,eps))
        for t in [r,fn,base,scale,norm,y]: f.write(t.numpy().tobytes())
    subprocess.run([binary,str(inp),str(out)],check=True)
    values=torch.frombuffer(bytearray(out.read_bytes()),dtype=torch.float32)
    offset=0
    result={}
    for name,ref in expected:
        got=values[offset:offset+ref.numel()].reshape(ref.shape)
        check(f'GPU {case}/{name}',got,ref)
        result[name]=got.clone()
        offset+=ref.numel()
    assert offset == values.numel()
    return result


def per_op(host,gpu,tmp):
    torch.manual_seed(621)
    for case,h,m,amplitude in [('tiny',64,7,1.),('production',4096,3,1.),
                               ('zero',4096,1,0.),('saturated',64,2,1e4)]:
        r=(torch.randn(m,4,h)*amplitude).contiguous()
        fn=(torch.randn(24,4*h)/(4*h)**.5).bfloat16().float()
        base=torch.randn(24)*(100 if case=='saturated' else .5)
        scale=torch.tensor([.8,1.2,.7])
        # Loader norms are BF16, so both paths use identical rounded weights.
        norm=(1+torch.randn(h)*.1).bfloat16().float()
        y=torch.randn(m,h)
        cfg=Cfg(dict(TINY,hidden_size=h)); eps=cfg.eps
        rec={}
        ref=FlashRef(cfg,DictWeights({'fn':fn,'base':base,'scale':scale}),
                     rec=lambda n,t:rec.__setitem__(n,t))
        x,post,comb=ref.mhc(r,'','hc')
        normalized=rms(x,norm,eps)
        streams=ref.mhc_out(y,post,comb,r)
        final=rms(streams.mean(1),norm,eps)
        got=host.prepare(r,fn,base,scale,norm,eps)
        expected=list(zip(['pre','post','comb','collapse','normalized'],
                          [rec['hc.pre'],post,comb,x,normalized]))
        for (name,target),actual in zip(expected,got): check(f'CPU {case}/{name}',actual,target)
        check(f'CPU {case}/out',host.output(r,y,post,comb),streams)
        check(f'CPU {case}/final',host.final(streams,norm,eps),final)
        # A deliberately non-symmetric matrix catches an accidental comb transpose.
        c=torch.rand(m,4,4)
        check(f'CPU {case}/orientation',host.output(r,y,post,c),ref.mhc_out(y,post,c,r))
        assert torch.allclose(comb.sum(-2),torch.ones(m,4),atol=3e-6)
        if gpu: gpu_case(gpu,tmp,case,r,fn,base,scale,norm,y,eps,
                         expected+[('out',streams),('final',final)])


def chain(host):
    cfg,model=hf_model()
    weights=DictWeights(checkpoint_dict(cfg,model))
    c=Cfg(cfg.to_dict()); ref=FlashRef(c,weights)
    tokens=[1,5,9,11,4,6,8]
    streams=weights.get('embed_tokens.weight')[tokens,None,:].expand(-1,4,-1).clone()
    expected=streams.clone(); st,rs=SeqState(),SeqState()
    # CPU loop independently executes mHC; attention and MLP stay W0 calls.
    # This tests stream orchestration, NOT the W3/W4/W5 CUDA blocks.
    for L in range(4):
        p=f'layers.{L}.'
        for block in ['attn','ffn']:
            prefix=p+f'hc_{block}_'
            norm=weights.get(p+('input_layernorm.weight' if block=='attn' else 'post_attention_layernorm.weight'))
            _,post,comb,_,x=host.prepare(streams,*(weights.get(prefix+n) for n in ['fn','base','scale']),norm,c.eps)
            if block=='attn': y=ref.kda(x,L,st) if c.kda[L] else ref.dsa(x,L,st)
            else: y=ref.moe(x,L) if c.moe[L] else ref.mlp(x,p+'mlp.','mlp')
            streams=host.output(streams,y,post,comb)
        expected=ref.layer(expected,L,rs)
        check(f'CPU chain layer {L}',streams,expected,2e-2)
    check('CPU chain final',host.final(streams,weights.get('norm.weight'),c.eps),
          rms(expected.mean(1),weights.get('norm.weight'),c.eps),2e-2)


def real_cases(host,gpu,tmp,dense,golden):
    from safetensors import safe_open
    dense=Path(dense)
    if (dense/'dense').is_dir(): dense=dense/'dense'
    with safe_open(golden,framework='pt',device='cpu') as g:
        for layer in range(4):
            tag=f'L{layer:02d}'
            with safe_open(str(dense/f'exlr8-dense-{layer:03d}.safetensors'),framework='pt',device='cpu') as w:
                for block in ['attn','ffn']:
                    prefix=f'hc_{block}_'
                    fn=w.get_tensor(prefix+'fn.bf16').float()
                    base=w.get_tensor(prefix+'base').float()
                    scale=w.get_tensor(prefix+'scale').float()
                    norm=w.get_tensor('input_layernorm.weight' if block=='attn' else 'post_attention_layernorm.weight').float()
                    all_r=g.get_tensor(tag+('.in' if block=='attn' else '.mid'))
                    y_key=tag+('.kda.out' if layer!=3 else '.attn.out') if block=='attn' else tag+('.mlp.out' if layer<3 else '.moe.out')
                    # Test M=1 and a full runtime row tile from the same golden positions.
                    for rows in sorted(set([1,min(8,len(all_r))])):
                        r=all_r[:rows].contiguous(); y=g.get_tensor(y_key)[:rows].contiguous()
                        hc=tag+'.'+prefix[:-1]
                        expected=[(n,g.get_tensor(hc+'.'+key)[:rows].contiguous()) for n,key in
                                  [('pre','pre'),('post','post'),('comb','comb'),('collapse','x')]]
                        expected += [('normalized',rms(expected[-1][1],norm,1e-5))]
                        got=host.prepare(r,fn,base,scale,norm,1e-5)
                        name=f'{Path(golden).stem}/{tag}/{block}/M{rows}'
                        for (n,e),a in zip(expected,got): check('CPU real '+name+'/'+n,a,e)
                        target=g.get_tensor(tag+('.mid' if block=='attn' else '.out'))[:rows].contiguous()
                        check('CPU real '+name+'/out',host.output(r,y,got[1],got[2]),target)
                        final=rms(target.mean(1),norm,1e-5)
                        if gpu: gpu_case(gpu,tmp,name.replace('/','_'),r,fn,base,scale,norm,y,1e-5,
                                         expected+[('out',target),('final',final)])


def replay_chain(host,gpu,tmp,dense,golden):
    """Eight consecutive mHC blocks with W0 block-output replay, not W3-W5 kernels."""
    from safetensors import safe_open
    dense=Path(dense)
    if (dense/'dense').is_dir(): dense=dense/'dense'
    with safe_open(golden,framework='pt',device='cpu') as g:
        r=g.get_tensor('L00.in')[:min(8,g.get_tensor('L00.in').shape[0])].contiguous()
        m,_,h=r.shape
        for layer in range(4):
            tag=f'L{layer:02d}'
            with safe_open(str(dense/f'exlr8-dense-{layer:03d}.safetensors'),framework='pt',device='cpu') as w:
                for block in ['attn','ffn']:
                    prefix=f'hc_{block}_'
                    fn=w.get_tensor(prefix+'fn.bf16').float()
                    base=w.get_tensor(prefix+'base').float(); scale=w.get_tensor(prefix+'scale').float()
                    norm=w.get_tensor('input_layernorm.weight' if block=='attn' else 'post_attention_layernorm.weight').float()
                    key=tag+('.kda.out' if layer!=3 else '.attn.out') if block=='attn' else tag+('.mlp.out' if layer<3 else '.moe.out')
                    y=g.get_tensor(key)[:m].contiguous()
                    rec={}; ref=FlashRef(Cfg(dict(TINY,hidden_size=h)),DictWeights({'fn':fn,'base':base,'scale':scale}),
                                        rec=lambda n,t:rec.__setitem__(n,t))
                    x,post,comb=ref.mhc(r,'','hc'); updated=ref.mhc_out(y,post,comb,r)
                    name=f'replay_{Path(golden).stem}_{tag}_{block}'
                    expected=list(zip(['pre','post','comb','collapse','normalized','out','final'],
                                      [rec['hc.pre'],post,comb,x,rms(x,norm,1e-5),updated,rms(updated.mean(1),norm,1e-5)]))
                    if gpu: r=gpu_case(gpu,tmp,name,r,fn,base,scale,norm,y,1e-5,expected)['out']
                    else:
                        _,p,c,_,_=host.prepare(r,fn,base,scale,norm,1e-5)
                        r=host.output(r,y,p,c)
                    check(name+' residual vs W0',r,g.get_tensor(tag+('.mid' if block=='attn' else '.out'))[:m],2e-2)
    print('PASS: 4-layer mHC residual replay (W0 block outputs; NOT integrated W3-W5 chain)')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu')
    parser.add_argument('--replay-only',action='store_true')
    parser.add_argument('--dense',help='lossless dense export (needed with --golden)')
    parser.add_argument('--golden',action='append',default=[])
    args=parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); lib=tmp/'mhc.so'
        subprocess.run(['g++','-O2','-std=c++17','-DG6_MODEL_FLASH=1','-shared','-fPIC',str(HERE/'test_bridge.cpp'),'-o',str(lib)],check=True)
        host=Host(lib)
        with torch.no_grad():
            if not args.replay_only:
                per_op(host,args.gpu,tmp)
                chain(host)
            if args.golden and not args.dense: parser.error('--golden requires --dense')
            for golden in args.golden:
                if args.replay_only: replay_chain(host,args.gpu,tmp,args.dense,golden)
                else: real_cases(host,args.gpu,tmp,args.dense,golden)
    print('PASS: selected mHC cases; '+('GPU cases passed' if args.gpu else 'GPU not run'))

if __name__=='__main__': main()
