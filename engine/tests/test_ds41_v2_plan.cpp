#include "../runtime/fabric_plan.h"
#include <cassert>
#include <cstring>
#include <iostream>
int main() {
    using namespace g6rt;
    assert(ds41_prefill_rowcap("8192",32,true)==8192);
    assert(ds41_prefill_rowcap("8192",32)==0);
    for(bool v2:{false,true})for(int w:{256,4096,8192}) {
        EngineOptions o; o.s2mode=true; o.NL=40; o.NX=2; o.NC=83;
        for(int i=0;i<40;++i)o.layers.push_back(i);
        o.cfg.ips={"1","2","3","4"}; o.cfg.rank=0; o.cfg.proxy_cpu=13; o.cfg.port=9000;
        PlanInputs in; in.trunk_bytes=2; in.head_payload=512;
        in.k.ds_pf_v2=v2; in.k.ds_prefill_w=std::to_string(w); in.k.ds_pf_tail="128";
        auto p=fabric_plan(o,in);
        if(w==8192&&!v2) {assert(!p.refusal.empty());continue;}
        if(!p.refusal.empty())std::cerr<<p.refusal<<'\n';
        assert(p.refusal.empty());
        const auto* f=p.find(FabKind::Ds41V2);
        assert(bool(f)==v2);
        if(f) {
            assert(f->port_off==104 && f->cfg.port==9104 && f->cfg.proxy_cpu==11);
            assert(f->cfg.ncoll==12 && f->cfg.compact && f->cfg.pool==0);
            assert(f->cfg.slot_bytes>=128*(4*5120*2+48*4+5120*2));
            assert(f->cfg.slot_bytes1>=4*ds41::v2::aligned(size_t((w+3)/4)*6352+8,256)-8);
        }
        in.k.ds_pf_lanes="2";
        auto two=fabric_plan(o,in);
        if(v2)assert(!two.refusal.empty()); else assert(two.refusal.empty());
    }
    // Actual six-segment tail layout: stage each owner's slice then reconstruct
    // only the final rows. Includes every owner-empty and boundary-crossing case.
    for(int M:{1,2,3,5,17,127,128,129,257,511,4093,4096,8191,8192}) {
        const size_t rb[]={40960,96,16,16,64,10240};
        for(int k=0;k<6;++k) {
            std::vector<unsigned char> dst(size_t(std::min(M,128))*rb[k],0);
            for(int p=0;p<4;++p) {
                const auto g=ds41::v2::geometry(M,4,p); const auto t=ds41::v2::tail_slice(g);
                std::vector<unsigned char> own(size_t(g.owned)*rb[k]);
                for(int r=0;r<g.owned;++r)std::memset(own.data()+size_t(r)*rb[k],(g.row0+r+k)%251,rb[k]);
                std::vector<unsigned char> send(128*rb[k],0);
                if(t.count) {
                    std::memcpy(send.data(),own.data()+size_t(t.local)*rb[k],size_t(t.count)*rb[k]);
                    std::memcpy(dst.data()+size_t(t.destination)*rb[k],send.data(),size_t(t.count)*rb[k]);
                }
            }
            for(size_t i=0;i<dst.size();++i)assert(dst[i]==(std::max(0,M-128)+int(i/rb[k])+k)%251);
        }
    }
    std::cout<<"v2 plan: stock cap4096, v2 cap8192, dedicated fabric and six-segment ragged tail PASS\n";
}
