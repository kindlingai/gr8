#include "../ds41_prefill_fabric.h"
#include "../s2/ds41_prefill.h"
#include "../ds41_rowshard.h"
#include <cassert>
#include <cstdio>
#include <numeric>
#include <vector>
int main() {
    constexpr int W=4096,TAIL=128,NL=40,NX=2,MX=48,H=5120,NR=4,NC=83,K=42;
    std::vector<int> ids(NL); std::iota(ids.begin(),ids.end(),0);
    const int il=ds41::br::late_start(ids); assert(il==21);
    const auto off=ds41::pf::fabric_runs(W,TAIL,il,NL,NX,MX,H,NR,MX*2240,true,false,false);
    const auto on=ds41::pf::fabric_runs(W,TAIL,il,NL,NX,MX,H,NR,MX*2240,true,false,true);
    assert(off.cfg.ncoll==NC+K && on.cfg.ncoll==NC+K+1 && on.cfg.run3==NC);
    const auto state=(uint32_t)ds41::rowshard::state_bytes(TAIL,H,2);
    assert(state==6578176 && state+8<=on.cfg.slot_bytes3+256);
    g6::FabricDev f{}; f.nr=NR; f.compact=1;
    assert(g6::fab_slot_layout(on.cfg,on.cfg.ncoll,&f));
    assert(state+8<=g6::fab_slot_stride(f,NC+K));
    const auto coll=g6s2::ds41_chunk_collectives(ids,NX,W,H*2,6144*2,il,TAIL,false);
    assert(ds41::rs::job_error(coll.sizes,NC,K,W,NR,H*2)==nullptr);
    const auto j=ds41::rs::job(coll.sizes,coll.order,NC,K,W,NR,H*2,state,true);
    assert(j.sizes.size()==NC+K+1 && j.sizes[NC+K]==state);
    bool state_after_last_ag=false;
    for(size_t i=1;i<j.order.size();++i) if(j.order[i]==NC+K) state_after_last_ag=j.order[i-1]==NC+K-1;
    assert(state_after_last_ag);
    bool engram_order0=false,engram_order1=false;
    for(size_t i=0;i+3<j.order.size();++i) {
        if(j.order[i]==1) engram_order0=j.order[i+1]==81 && j.order[i+2]==NC+1 && j.order[i+3]==2;
        if(j.order[i]==27) engram_order1=j.order[i+1]==82 && j.order[i+2]==NC+27 && j.order[i+3]==28;
    }
    assert(engram_order0 && engram_order1);
    // Boot contract: the real fabric sizer/layout and build_ds41's validator
    // must agree before weights/graphs. Engram ON (NX=2), PF_RS ON (K=42),
    // ROW_SHARD 0/1; lossy MXFP8 is legal only with ROW_SHARD and adds no c.
    int valid=0, invalid=0, old_count_red=0;
    for (int rowshard : {0,1}) for (int y_mxfp8 : {0,1}) {
        if (y_mxfp8 && !rowshard) { ++invalid; continue; } // g6m4 knob parser refuses this pair
        auto boot = ds41::pf::fabric_runs(W,TAIL,il,NL,NX,MX,H,NR,MX*2240,true,false,rowshard).cfg;
        boot.pool=4; // the pooled all-levers route
        g6::FabricDev dev{}; dev.nr=NR; dev.compact=1;
        assert(g6::fab_slot_layout(boot,boot.ncoll,&dev));
        assert(ds41::rs::fabric_ncoll_error(dev.ncoll,NC,K,false,rowshard)==nullptr);
        assert(dev.ncoll==NC+K+rowshard);
        const int old_count=ds41::rs::main_ncoll(NC,K); // old g6m4 fabric sizer omitted state
        const bool old_rejected=ds41::rs::fabric_ncoll_error(old_count,NC,K,false,rowshard)!=nullptr;
        assert(old_rejected==bool(rowshard));
        old_count_red+=old_rejected;
        ++valid;
    }
    assert(valid==3 && invalid==1 && old_count_red==2);
    std::printf("ROW_SHARD_BOOT_CONTRACT_PASS engram=1 pf_rs=1 valid=3 invalid_y_without_row=1 old_count_RED=2\n");
    // F14 pooled run 3 has K AGs plus this one state collective. Check the
    // actual RS/AG/state proxy walk, not only the configuration parser.
    for (int nx : {0, NX}) for (int pool : {0, 2, 3, 4, 8, K}) {
        const int nc = 2*NL+1+nx, state_c = nc+K;
        auto cfg = ds41::pf::fabric_runs(W,TAIL,il,NL,nx,MX,H,NR,MX*2240,true,false,true).cfg;
        cfg.pool = pool;
        g6::FabricDev d{}; d.nr=NR; d.compact=1;
        assert(g6::fab_slot_layout(cfg,cfg.ncoll,&d));
        assert(d.pool==pool && d.ncoll==state_c+1);
        const auto chunk = g6s2::ds41_chunk_collectives(ids,nx,W,H*2,6144*2,il,TAIL,false);
        const auto job = ds41::rs::job(chunk.sizes,chunk.order,nc,K,W,NR,H*2,state,true);
        assert(job.sizes.size()==(size_t)cfg.ncoll && job.sizes[state_c]==state);
        std::vector<int> credit;
        assert(g6::fab_pool_credits(d,job.order,&credit));
        const size_t half = g6::fab_slot_base(d,d.ncoll);
        for (int c=0;c<d.ncoll;++c) {
            assert(g6::fab_slot_base(d,c)+g6::fab_slot_stride(d,c)<=half);
            assert(g6::fab_slot_fits(d,c,job.sizes[c]));
        }
        if (pool) {
            const int previous=state_c-pool;
            std::vector<int> pos((size_t)d.ncoll,-1);
            for (size_t k=0;k<job.order.size();++k) pos[job.order[k]]=(int)k;
            assert(g6::fab_pool_slot(d,state_c)==g6::fab_pool_slot(d,previous));
            assert(g6::fab_slot_base(d,state_c)==g6::fab_slot_base(d,previous));
            assert(credit[state_c]>=0 && pos[previous]<pos[credit[state_c]] && pos[credit[state_c]]<pos[state_c]);
        } else assert(credit[state_c]==-1);
        for (int rank=0;rank<NR;++rank) for (int peer=0;peer<NR;++peer) if (rank!=peer) {
            g6::FabricDev sender=d, receiver=d; sender.rank=rank; receiver.rank=peer;
            assert(g6::fab_recv_off_at(sender,0,state_c,peer,rank)==g6::fab_recv_off(receiver,0,state_c,rank));
            assert(g6::fab_recv_off(receiver,0,state_c,rank)+g6::fab_slot_stride(d,state_c)<=g6::fab_recv_bytes(d));
        }
    }
    auto bad = on.cfg; bad.pool=2; bad.ncoll++;
    g6::FabricDev bad_dev{};
    assert(!g6::fab_slot_layout(bad,bad.ncoll,&bad_dev)); // a second extra run-3 collective remains unsupported
    std::printf("ROW_SHARD_POOL_PASS configurations=12 pooled=10 unpooled=2 state_credit=10 invalid_extra=1\n");
    std::printf("ROW_SHARD_FABRIC_PASS ncoll_off=%d ncoll_on=%d state_bytes=%u state_stride=%zu order_after_last_ag=1\n",
                off.cfg.ncoll,on.cfg.ncoll,state,g6::fab_slot_stride(f,NC+K));
}
