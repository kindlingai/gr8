#include "../runtime/fabric_plan.h"
#include "../runtime/s2/ds41_prefill.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
static void wide_plan_checks() {
    using namespace g6rt;
    assert(ds41_prefill_rowcap("8193",32,true,false)==0);
    assert(ds41_prefill_rowcap("16385",32,true,true)==0);
    assert(ds41_prefill_rowcap("12288",32,false,true)==0);
    std::vector<int> layers; for(int i=0;i<61;++i)layers.push_back(i);
    auto make=[&](int width,bool wide) {
        EngineOptions o; o.s2mode=true; o.NL=61; o.NX=2; o.NC=125;
        o.layers=layers; o.cfg.rank=0; o.cfg.ips={"1","2","3","4"};
        PlanInputs in; in.trunk_bytes=2; in.head_payload=512;
        in.k.ds_pf_v2=true; in.k.ds_pf_v2_w16k=wide;
        in.k.ds_prefill_w=std::to_string(width); in.k.ds_pf_tail="128";
        in.k.ds_sh_tc=true; in.k.ds_prefill_min=2;
        return fabric_plan(o,in);
    };
    const auto baseline=make(4096,false); assert(baseline.refusal.empty());
    const auto& base=baseline.find(FabKind::Main)->cfg;
    for(int width : {12288,16384}) {
        assert(ds41_prefill_rowcap(std::to_string(width).c_str(),32,true,true)==width);
        assert(ds41_prefill_main_fabric_rows(width,true,true)==4096);
        assert(ds41_prefill_main_fabric_rows(width,true,false)==width);
        assert(!make(width,false).refusal.empty());
        auto p=make(width,true); assert(p.refusal.empty() && p.ds_pf_v2);
        assert(p.find(FabKind::Ds41V2));
        const auto& cfg=p.find(FabKind::Main)->cfg;
        assert(cfg.slot_bytes==base.slot_bytes && cfg.slot_bytes1==base.slot_bytes1 && cfg.slot_bytes2==base.slot_bytes2);
        int messages=0;
        for(const auto& line:p.log)if(line.find("# ds41_v2_w16k ")==0) {
            assert(line=="# ds41_v2_w16k on=1 pf_rows="+std::to_string(width)+" main_pf_rows=4096\n"); ++messages;
        }
        assert(messages==1);
        for(const auto& line:baseline.log)assert(line.find("# ds41_v2_w16k ")!=0);
        g6::FabricDev f{};assert(g6::fab_slot_layout(cfg,cfg.ncoll,&f));
        for(int late : {20,21})for(bool skip : {false,true}) {
            const auto raw=g6s2::ds41_chunk_collectives(layers,2,width,5120*2,6144*2,late,128,skip);
            const auto main=g6s2::ds41_main_chunk_collectives(raw,true,skip,late,61);
            const auto stock=g6s2::ds41_main_chunk_collectives(raw,false,skip,late,61);
            assert(stock.sizes==raw.sizes && stock.order==raw.order);
            if(skip)assert(main.sizes.empty() && main.order.empty());
            else {
                assert(main.order.size()==size_t(122-2*late));
                for(size_t j=0;j<main.order.size();++j)assert(main.order[j]==2*late+int(j));
                for(size_t c=0;c<main.sizes.size();++c) {
                    if(c<size_t(2*late) || c>=122)assert(main.sizes[c]==0);
                    else assert(main.sizes[c]==128*5120*2);
                    assert(main.sizes[c]+8<=g6::fab_slot_stride(f,int(c)));
                }
                // Without filtering, full-width Range A exceeds the capped main slots.
                assert(raw.sizes[0]+8>g6::fab_slot_stride(f,0));
            }
        }
    }
    std::cout<<"wide plan: main rows4096, late20/21 filtering, default unchanged PASS\n";
}
int main() {
    wide_plan_checks();
    using namespace g6rt;
    assert(ds41_prefill_rowcap("8192",32,true)==8192);
    assert(ds41_prefill_rowcap("8192",32)==0);
    for(bool v2:{false,true})for(bool xt:{false,true})for(bool pf:{false,true})for(int w:{256,4096,8192}) {
        EngineOptions o; o.s2mode=true; o.NL=40; o.NX=2; o.NC=83;
        for(int i=0;i<40;++i)o.layers.push_back(i);
        o.cfg.ips={"1","2","3","4"}; o.cfg.rank=0; o.cfg.proxy_cpu=13; o.cfg.port=9000;
        PlanInputs in; in.trunk_bytes=2; in.head_payload=512;
        in.k.ds_pf_v2=v2; in.k.ds_prefill_min=2; in.k.ds_sh_tc=true; in.k.ds_pf_v2_xtrace=xt; in.k.xtrace_pf=pf; in.k.ds_prefill_w=std::to_string(w); in.k.ds_pf_tail="128";
        auto p=fabric_plan(o,in);
        if(w==8192&&!v2) {assert(!p.refusal.empty());continue;}
        if(!p.refusal.empty())std::cerr<<p.refusal<<'\n';
        assert(p.refusal.empty());
        const auto* f=p.find(FabKind::Ds41V2);
        assert(bool(f)==v2);
        assert(p.find(FabKind::Main)->cfg.xtrace==(pf?0:1));
        if(f) {
            assert(f->cfg.xtrace==(xt?1:0));
            assert(f->port_off==104 && f->cfg.port==9104 && f->cfg.proxy_cpu==11);
            assert(f->cfg.ncoll==12 && f->cfg.compact && f->cfg.pool==0);
            assert(f->cfg.slot_bytes>=128*(4*5120*2+48*4+5120*2));
            assert(f->cfg.slot_bytes1>=4*ds41::v2::aligned(size_t((w+3)/4)*6352+8,256)-8);
            g6::FabricDev wire{}; assert(g6::fab_slot_layout(f->cfg,12,&wire));
            const size_t a2a=4*ds41::v2::aligned(size_t((w+3)/4)*6352+8,256);
            assert(a2a<=g6::fab_slot_stride(wire,int(ds41::v2::Coll::Engram)));
            if(w>=4096)assert(a2a>wire.stride); // enqueue must check the actual run, not run0

        }
        in.k.ds_pf_lanes="2";
        // two lanes with v2 asked: no refusal, no v2 fabric, the reason for the boot's one line
        auto two=fabric_plan(o,in);
        assert(two.refusal.empty() && !two.find(FabKind::Ds41V2) && !two.ds_pf_v2);
        assert(v2 ? two.v2_fallback && std::string(two.v2_fallback)=="GR8_DS_PF_LANES is not 1" : !two.v2_fallback);
    }
    // GR8_DS_PF_V2=1 never refuses a plan for its own conditions: each failed one plans no v2 fabric and names itself
    // (the boot logs it once and runs the old range A). A width the old path cannot take either stays a refusal.
    {
        auto plan=[](auto edit) {
            EngineOptions o; o.s2mode=true; o.NL=40; o.NX=2; o.NC=83;
            for(int i=0;i<40;++i)o.layers.push_back(i);
            o.cfg.ips={"1","2","3","4"}; o.cfg.rank=0; o.cfg.proxy_cpu=13; o.cfg.port=9000;
            PlanInputs in; in.trunk_bytes=2; in.head_payload=512;
            in.k.ds_pf_v2=true; in.k.ds_prefill_min=2; in.k.ds_sh_tc=true; in.k.ds_prefill_w="8192"; in.k.ds_pf_tail="128";
            edit(o,in);
            return fabric_plan(o,in);
        };
        auto off=[&](auto edit,const char* why) {
            const auto p=plan(edit);
            if(!p.refusal.empty())std::cerr<<p.refusal<<'\n';
            assert(p.refusal.empty() && !p.ds_pf_v2 && !p.find(FabKind::Ds41V2));
            assert(p.v2_fallback && std::string(p.v2_fallback)==why);
        };
        const auto on=plan([](EngineOptions&,PlanInputs&){});
        assert(on.refusal.empty() && on.ds_pf_v2 && on.find(FabKind::Ds41V2) && !on.v2_fallback && on.pf_rows==8192);
        off([](EngineOptions& o,PlanInputs&){o.s2mode=false;},"not a serve run (offline or simulator)");
        off([](EngineOptions&,PlanInputs& in){in.k.ds_prefill_min=0;},"no W-row prefill (GR8_DS_PREFILL_MIN=0)");
        off([](EngineOptions&,PlanInputs& in){in.k.ds_sh_tc=false;},"GR8_DS_SH_TC=0 (exact output)");
        off([](EngineOptions&,PlanInputs& in){in.k.ds_pf_lanes="2";},"GR8_DS_PF_LANES is not 1");
        off([](EngineOptions&,PlanInputs& in){in.k.ds_pf_tail="256";},"GR8_DS_PF_TAIL is not 128");
        off([](EngineOptions&,PlanInputs& in){in.k.ds_pf_l20_tail=true;},"GR8_DS_PF_L20_TAIL=1 moves the late range to layer 20");
        off([](EngineOptions&,PlanInputs& in){in.k.ds_prefill_w="4000";},"GR8_DS_PREFILL_W is not 256..8192 in steps of 256");
        // the 1M-context setup: W 4096, the late mHC row split
        off([](EngineOptions&,PlanInputs& in){in.k.ds_prefill_w="4096"; in.k.ds_pf_rs=true; in.k.ds_mhc_fuse=true; in.k.ds_mhc_row_shard=true;},
            "GR8_DS_MHC_ROW_SHARD=1 (the 1M-context setup)");
        // the exact-output launcher stack with v2 asked anyway: two lanes and SH_TC=0 at W 4096
        off([](EngineOptions&,PlanInputs& in){in.k.ds_prefill_w="4096"; in.k.ds_pf_lanes="2"; in.k.ds_sh_tc=false;},"GR8_DS_PF_LANES is not 1");
        // genuine errors stay errors
        assert(!plan([](EngineOptions&,PlanInputs& in){in.k.ds_prefill_w="9000";}).refusal.empty());
        assert(!plan([](EngineOptions&,PlanInputs& in){in.k.ds_pf_v2=false;}).refusal.empty());   // W 8192 without v2
        assert(!plan([](EngineOptions&,PlanInputs& in){in.k.ds_pf_lanes="3";}).refusal.empty());
        std::cout<<"v2 fallback plan: serve, prefill, SH_TC, lanes, tail, L20, width, row shard fall back with their reason; bad values still refuse PASS\n";
    }
    // Striping is host-config only. Zero retains inherited DS_STRIPE exactly;
    // a positive override touches only the dedicated v2 fabric, never main.
    for(bool v2:{false,true})for(bool dual:{false,true})for(bool inherited:{false,true})
    for(int64_t threshold:{int64_t(0),int64_t(1),int64_t(8192),int64_t(1048576),INT64_MAX}) {
        if(inherited&&!dual)continue; // DS_STRIPE's existing two-rail guard
        EngineOptions o; o.s2mode=true; o.NL=40; o.NX=2; o.NC=83;
        for(int i=0;i<40;++i)o.layers.push_back(i);
        o.cfg.ips={"1","2","3","4"}; o.cfg.rank=0;
        if(dual)o.cfg.ips2={"5","6","7","8"};
        PlanInputs in; in.trunk_bytes=2; in.head_payload=512;
        in.k.ds_pf_v2=v2; in.k.ds_prefill_min=2; in.k.ds_sh_tc=true; in.k.ds_prefill_w="4096"; in.k.ds_pf_tail="128";
        in.k.ds_stripe=inherited; in.k.ds_stripe_min=2097152;
        const auto baseline=fabric_plan(o,in); assert(baseline.refusal.empty());
        in.k.ds_pf_v2_stripe_min=threshold;
        const auto p=fabric_plan(o,in); assert(p.refusal.empty());
        assert(p.fabrics.size()==baseline.fabrics.size());
        for(size_t i=0;i<p.fabrics.size();++i) {
            const auto& f=p.fabrics[i]; const auto& b=baseline.fabrics[i];
            assert(f.kind==b.kind);
            auto cfg=f.cfg;
            assert(cfg.stripe_min==(f.kind==FabKind::Ds41V2&&threshold>0
                ? size_t(threshold) : b.cfg.stripe_min));
            cfg.stripe_min=b.cfg.stripe_min;
            // describe() serializes every FabricConfig field, including slots.
            assert(describe_fabric("test",cfg)==describe_fabric("test",b.cfg));
        }
        int logs=0;
        for(const auto& line:p.log)if(line.find("# ds41_v2_stripe ")==0) {
            ++logs;
            const size_t min=threshold>0 ? size_t(threshold) : inherited ? 2097152 : 0;
            assert(line==fplan::fmt("# ds41_v2_stripe min=%zu rails=%d effective_min=%zu\n",
                min,dual?2:1,min&&dual?std::max(min,size_t(8192)):0));
        }
        assert(logs==(v2?1:0));
    }
    // TP4 ring (cfg.ring): the dedicated fabric never stripes. The plan logs rails=3 and
    // effective_min=0, keeps the ring layout, and passes a positive threshold through
    // unchanged on the dedicated fabric only (the proxy ignores it at four ranks).
    for(bool dual:{false,true})
    for(int64_t threshold:{int64_t(0),int64_t(1),int64_t(8192),int64_t(1048576),INT64_MAX}) {
        EngineOptions o; o.s2mode=true; o.NL=40; o.NX=2; o.NC=83;
        for(int i=0;i<40;++i)o.layers.push_back(i);
        o.cfg.ips={"1","2","3","4"}; o.cfg.rank=0; o.cfg.ring=true;
        if(dual)o.cfg.ips2={"5","6","7","8"};
        PlanInputs in; in.trunk_bytes=2; in.head_payload=512;
        in.k.ds_pf_v2=true; in.k.ds_prefill_min=2; in.k.ds_sh_tc=true; in.k.ds_prefill_w="4096"; in.k.ds_pf_tail="128";
        const auto baseline=fabric_plan(o,in); assert(baseline.refusal.empty());
        in.k.ds_pf_v2_stripe_min=threshold;
        const auto p=fabric_plan(o,in); assert(p.refusal.empty());
        assert(p.fabrics.size()==baseline.fabrics.size());
        for(size_t i=0;i<p.fabrics.size();++i) {
            const auto& f=p.fabrics[i]; const auto& b=baseline.fabrics[i];
            assert(f.kind==b.kind);
            assert(f.cfg.ring==b.cfg.ring);
            if(f.kind==FabKind::Ds41V2)assert(f.cfg.ring);
            auto cfg=f.cfg;
            assert(cfg.stripe_min==(f.kind==FabKind::Ds41V2&&threshold>0
                ? size_t(threshold) : b.cfg.stripe_min));
            cfg.stripe_min=b.cfg.stripe_min;
            assert(describe_fabric("test",cfg)==describe_fabric("test",b.cfg));
        }
        int logs=0;
        for(const auto& line:p.log)if(line.find("# ds41_v2_stripe ")==0) {
            ++logs;
            assert(line==fplan::fmt("# ds41_v2_stripe min=%zu rails=3 effective_min=0\n",
                threshold>0 ? size_t(threshold) : size_t(0)));
        }
        assert(logs==1);
    }
    std::cout<<"v2 stripe plan: zero/inherited, thresholds 1/8192/1MiB/INT64_MAX, one/two rails, TP4 ring (rails=3, never striped), v2 off, isolated layout and once-per-plan log PASS\n";
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
    std::cout<<"v2 plan: stock cap4096, v2 cap8192, isolated xtrace/PF gates, dedicated fabric and six-segment ragged tail PASS\n";
}
