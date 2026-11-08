#!/usr/bin/env python3
"""Opt-in integrated adapter GPU gate using freshly built off/pipe/swz binaries.
No performance verdict: tests capture replay geometry, sessions and phase dispatch.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
SOURCE=r'''
#include "capture_io.h"
#include <fstream>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 sweep_capture::Record r;auto& m=r.metadata;
 m.n=4;m.prompt_tokens=32768;m.chunk_position=24576;m.layer=8;m.index_source=8;m.ratio=2;
 m.ring_rows=8319;m.key_rows=16386;m.ck_stride_bytes=16386*288;m.slot_count=3;
 m.request_id="fixture";m.dispatch_phase="prefill_v2";m.format="fp4_e2m1_e4m3_288";
 m.session_form="per_row_slots";m.engine_revision="test-only";m.knob_manifest="{\"test\":true}";m.origin="synthetic_test";
 r.positions={24576,24579,24577,24578};r.sessions={2,1,2,1};
 r.topk=sweep_topk::make(r.positions,2,m.key_rows,"shared",42);
 auto data=sweep_capture::encode(r);std::ofstream o(argv[1],std::ios::binary);o<<data;
 return !o;
}
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 (a.out/'capture_fixture.cpp').write_text(SOURCE)
 subprocess.run(['g++','-std=c++17','-O2','-I',str(HERE),str(a.out/'capture_fixture.cpp'),'-o',str(a.out/'capture_fixture')],check=True)
 capture=a.out/'synthetic.asw';subprocess.run([str(a.out/'capture_fixture'),str(capture)],check=True)
 cases={
  'capture':[4,24576,2,0,128,16386,'--capture',str(capture)],
  'main':[1,32767,2,0,128,32768,'--phase','main','--ring-rows',175,'--slots',2,'--slot',1],
  'legacy':[4,24576,2,0,128,32768,'--phase','prefill_legacy'],
  'local':[4,24576,2,0,128,32768,'--topk-pattern','local'],
 }
 count=0
 with open('/tmp/attn-sweep-5090.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for name,args in cases.items():
   for variant in ('off','pipe','swz'):
    exe=a.bundle/variant
    cmd=[str(exe),'--warmup','0','--reps','2','--iters','2','--',*map(str,args)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    (a.out/f'{name}.{variant}.stdout').write_text(r.stdout);(a.out/f'{name}.{variant}.stderr').write_text(r.stderr)
    assert r.returncode==0,(name,variant,r.stdout,r.stderr)
    data=json.loads(r.stdout);assert data['status']=='EXACT' and not data['integrity_errors'],data
    m=data['adapter_metadata'];assert m['within_row_unique_fraction']==1,m
    if name=='capture':assert m['slot_count']==3 and m['input_provenance']=='synthetic_capture_test' and m['capture_sha256'],m
    if name in ('main','legacy'):assert data['ptxas_key']=='k_attn_cpa_tILb0ELb1EE',data
    if name=='main':assert m['ring_rows']==175 and m['slot_count']==2,m
    print(name,variant,'EXACT',m['dispatch_phase'],data['ptxas_key'],flush=True);count+=1
  bad=list(cases['capture']);bad[0]=5
  r=subprocess.run([str(a.bundle/'off'),'--',*map(str,bad)],capture_output=True,text=True,timeout=120)
  assert r.returncode==2 and 'capture shape does not match' in r.stderr,r
  (a.out/'invalid.stderr').write_text(r.stderr)
 print(f'ADAPTER GPU PASS {count} cases; invalid capture refused')
if __name__=='__main__':main()
