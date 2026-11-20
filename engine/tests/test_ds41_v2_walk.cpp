#include "../runtime/ds41_v2_walk.h"
#include "../runtime/ds41_pf_split.h"
#include "../runtime/ds41_dspark_prefill.h"
#include <cassert>
#include <iostream>
using namespace ds41;
int main() {
    std::vector<int> ids; for(int l=0;l<40;++l) ids.push_back(l);
    int cases=0;
    for(bool eng:{false,true}) for(bool dsp:{false,true}) for(bool skip:{false,true}) {
        auto nf=dsp::pf::nonfinal_plan(plan(ids,0,40,true,true,eng,dsp),dsp);
        auto split=br::split_plan(nf,ids);
        assert(split.on() && br::check_split(nf,split,ids).empty());
        const size_t end=skip?split.b-1:split.b;
        size_t next=0; int compact=0,candidates=0;
        auto stop=v2::walk_stock(split.ops,end,[&](const v2::Run& r) {
            assert(r.begin==next && r.end>r.begin && r.end<=end);
            for(size_t i=r.begin;i<r.end;++i) {
                assert(split.ops[i].unit==r.unit && split.ops[i].ui==r.layer);
                compact+=split.ops[i].stage==ST_PF_COMPACT;
                candidates+=split.ops[i].stage==ST_CAND_BLOCKS;
            }
            next=r.end; return Stop{};
        });
        assert(!stop.stopped && next==end && compact==!skip && candidates==1);
        int calls=0;
        auto failed=v2::walk_stock(split.ops,end,[&](const v2::Run&) {
            Stop s; s.stopped=(++calls==2); return s;
        });
        assert(failed.stopped && calls==2);
        ++cases;
    }
    std::cout<<"ds41_v2_walk: "<<cases<<" Engram/DSpark/tail combinations preserve every stock op; stop propagation PASS\n";
}
