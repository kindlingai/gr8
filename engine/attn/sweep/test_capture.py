import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = r'''
#include "capture_hook.h"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace sweep_capture;
Record record() {
    Record r;auto& m=r.metadata;
    m.n=4;m.prompt_tokens=32768;m.chunk_position=24576;m.layer=8;m.index_source=8;m.ratio=2;
    m.ring_rows=8319;m.key_rows=16386;m.ck_stride_bytes=16386*288;m.slot_count=1;
    m.request_id="7";m.dispatch_phase="prefill_v2";m.format="fp4_e2m1_e4m3_288";
    m.session_form="implicit_single";m.engine_revision="test-engine";m.knob_manifest="{\"GR8_DS_COMP_FP4\":1}";
    m.origin="synthetic_test";r.positions={24576,24577,24578,24579};r.sessions={0,0,0,0};
    r.topk=sweep_topk::make(r.positions,2,m.key_rows,"uniform",42);return r;
}
template<class F>void refuses(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught);}
int main(int argc,char** argv) {
    assert(argc==2);std::string dir=argv[1];
    assert(sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(sha256(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    auto r=record(),round=decode(encode(r));
    assert(!round.is_captured && round.topk==r.topk && round.positions==r.positions && round.sessions==r.sessions);
    assert(round.metrics.within_row_unique_fraction==1 && round.source_hash.size()==64);
    auto file=encode(r);auto corrupt=file;corrupt.back()^=1;refuses([&]{decode(corrupt);});
    refuses([&]{decode(file.substr(0,file.size()-1));});refuses([&]{decode(file+"x");});
    for(auto mutate:{0,1,2,3,4,5,6,7,8,9}) {
        auto bad=r;
        if(mutate==0)bad.topk[1]=bad.topk[0];
        if(mutate==1)bad.topk[0]=16385;
        if(mutate==2)bad.topk[0]=-2;
        if(mutate==3)bad.topk[0]=-1;
        if(mutate==4)bad.sessions[0]=1;
        if(mutate==5)bad.positions[0]=32768;
        if(mutate==6)bad.metadata.ck_stride_bytes=7;
        if(mutate==7)bad.metadata.dispatch_phase="guessed";
        if(mutate==8)bad.metadata.knob_manifest="{}";
        if(mutate==9)bad.topk.pop_back();
        refuses([&]{encode(bad);});
    }
    // Short causal rows retain exact -1 tails; window-only captures have no key list.
    auto small=r;small.positions={0,1,2,3};small.metadata.chunk_position=0;
    small.topk=sweep_topk::make(small.positions,2,small.metadata.key_rows,"uniform",7);
    assert(decode(encode(small)).topk==small.topk);
    auto window=small;window.metadata.ratio=0;window.metadata.format="window_only";window.metadata.index_source=-1;
    window.metadata.key_rows=0;window.metadata.ck_stride_bytes=0;window.topk.assign(4*512,-1);
    assert(decode(encode(window)).topk==window.topk);
    auto slots=r;slots.metadata.slot_count=3;slots.metadata.session_form="per_row_slots";slots.sessions={2,1,2,1};
    assert(decode(encode(slots)).sessions==slots.sessions);
    // Callback counters prove off and budget/filter skips cause no sync/copy.
    int copies=0,syncs=0;
    auto copy=[&](void* d,const void* s,size_t n){++copies;std::memcpy(d,s,n);};
    auto sync=[&](){++syncs;};
    Input in;in.metadata=r.metadata;in.topk=r.topk.data();in.positions=r.positions.data();
    Context ctx{32768,24576,0,0,"7","prefill_v2"};
    assert(config_from_file("").empty());
    refuses([&]{config_from_file(dir+"/missing-plan.json");});
    Hook off;assert(!off.enabled()&&!off.capture(ctx,in,copy,sync)&&copies==0&&syncs==0);
    std::string config="{\"schema\":1,\"output_dir\":"+quote(dir)+",\"engine_revision\":\"test\",\"knob_manifest\":{\"x\":1},"
        "\"layers\":[8],\"chunk_positions\":[24576],\"prompt_lengths\":[32768],\"ranks\":[0],\"max_records\":1,\"max_bytes\":100000}";
    {std::ofstream plan(dir+"/plan.json");plan<<config;}
    assert(config_from_file(dir+"/plan.json")==config);
    Hook on(config);assert(on.enabled());
    auto skip=ctx;skip.rank=1;assert(!on.capture(skip,in,copy,sync)&&copies==0&&syncs==0);
    skip=ctx;skip.phase="";assert(!on.capture(skip,in,copy,sync)&&copies==0&&syncs==0);
    assert(on.capture(ctx,in,copy,sync)&&copies==2&&syncs==1);
    auto captured=load(dir+"/r0_q7_c24576_l8_prefill_v2.asw");
    assert(captured.is_captured&&captured.topk==r.topk&&captured.positions==r.positions);
    ctx.request_id="8";assert(!on.capture(ctx,in,copy,sync)&&copies==2&&syncs==1);
    Hook collision(config);ctx.request_id="7";refuses([&]{collision.capture(ctx,in,copy,sync);});
    auto low=config;auto at=low.find("100000");low.replace(at,6,"1");Hook budget(low);
    copies=syncs=0;assert(!budget.capture(ctx,in,copy,sync)&&copies==0&&syncs==0);
    // A noncompact device ldtk is copied row-wise, preserving exact row correlation.
    Hook strided(config);ctx.request_id="9";in.metadata.ldtk=520;
    std::vector<int32_t> padded(4*520,999999);
    for(int i=0;i<4;++i)std::copy_n(r.topk.data()+i*512,512,padded.data()+i*520);
    in.topk=padded.data();assert(strided.capture(ctx,in,copy,sync));
    assert(load(dir+"/r0_q9_c24576_l8_prefill_v2.asw").topk==r.topk);
    // Per-row session and position maps must not be normalized or sorted.
    Hook mapped(config);ctx.request_id="10";in.metadata=r.metadata;
    in.metadata.slot_count=3;in.metadata.session_form="per_row_slots";
    std::vector<int32_t> mapping={2,1,2,1},positions={24579,1000,24576,1003};
    auto lists=sweep_topk::make(positions,2,r.metadata.key_rows,"uniform",3);
    in.topk=lists.data();in.positions=positions.data();in.sessions=mapping.data();
    assert(mapped.capture(ctx,in,copy,sync));
    auto got=load(dir+"/r0_q10_c24576_l8_prefill_v2.asw");
    assert(got.sessions==mapping&&got.positions==positions&&got.topk==lists);
    std::cout<<"capture: SHA256 vectors, roundtrip, corruption, bounds, uniqueness, padding, slots, default-off, filters, budgets, exclusive-write, stride PASS\n";
}
'''


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        (cls.root/'test.cpp').write_text(SOURCE)
        for name,src in [('test',cls.root/'test.cpp'),('tool',HERE/'capture_tool.cpp')]:
            subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pthread','-I',str(HERE),str(src),'-o',str(cls.root/name)],check=True)
        p=subprocess.run([str(cls.root/'test'),str(cls.root)],capture_output=True,text=True)
        if p.returncode:raise AssertionError(p.stdout+p.stderr)
        print(p.stdout,end='')
        cls.path=cls.root/'r0_q7_c24576_l8_prefill_v2.asw'

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_reader_and_independent_hash(self):
        data=self.path.read_bytes()
        self.assertEqual(data[:8],b'ASWEEP1\n')
        ml,pl=struct.unpack_from('<II',data,8)
        self.assertEqual(len(data),80+ml+pl)
        self.assertEqual(data[16:80].decode(),hashlib.sha256(data[80:]).hexdigest())
        result=subprocess.run([str(self.root/'tool'),'inspect',str(self.path)],capture_output=True,text=True,check=True)
        obj=json.loads(result.stdout)
        self.assertTrue(obj['is_captured'])
        self.assertEqual(obj['source_hash'],hashlib.sha256(data).hexdigest())
        self.assertEqual(obj['within_row_unique_fraction'],1)
        self.assertEqual(obj['metadata']['ring_rows'],8319)
        self.assertEqual(obj['metadata']['operands']['q'],'unavailable')
        self.assertFalse(obj['metadata']['timing_data'])
        cases=json.loads(subprocess.check_output([str(self.root/'tool'),'case',str(self.path)]))
        self.assertEqual(cases[0]['args'][-2:],['--capture',str(self.path)])
        self.assertFalse(cases[0]['decision_rows'])
        mapped=self.root/'r0_q10_c24576_l8_prefill_v2.asw'
        cases=json.loads(subprocess.check_output([str(self.root/'tool'),'case',str(mapped)]))
        self.assertEqual(cases[0]['args'][:6],[4,24576,2,0,128,16386])


    def test_rehashed_invalid_metadata_and_indices(self):
        data=self.path.read_bytes();ml,pl=struct.unpack_from('<II',data,8)
        meta=json.loads(data[80:80+ml]);payload=data[80+ml:]
        for field,value in [('ring_rows',0),('n',8193),('slot_count',0),('schema',2),('origin','realish'),('format','wrong'),('ck_stride_bytes',-1)]:
            altered=dict(meta);altered[field]=value
            m=json.dumps(altered).encode();body=m+payload
            bad=b'ASWEEP1\n'+struct.pack('<II',len(m),pl)+hashlib.sha256(body).hexdigest().encode()+body
            path=self.root/'bad.asw';path.write_bytes(bad)
            p=subprocess.run([str(self.root/'tool'),'inspect',str(path)],capture_output=True)
            self.assertNotEqual(p.returncode,0,field)
        bad_payload=bytearray(payload);bad_payload[4:8]=bad_payload[0:4]
        body=data[80:80+ml]+bad_payload
        (self.root/'bad.asw').write_bytes(data[:16]+hashlib.sha256(body).hexdigest().encode()+body)
        self.assertNotEqual(subprocess.run([str(self.root/'tool'),'inspect',str(self.root/'bad.asw')],capture_output=True).returncode,0)

