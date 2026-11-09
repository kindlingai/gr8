#include "../runtime/ds41_v2.h"
#include <cassert>
#include <iostream>
#include <string>
using namespace ds41::v2;
int main() {
    assert(boot_ok(4096,128,4,1,21));
    assert(boot_ok(8192,128,4,1,21));
    assert(!boot_ok(4096,128,4,2,21));
    assert(!boot_ok(4096,128,3,1,21));
    assert(!boot_ok(4096,128,4,1,20));
    // GR8_DS_PF_V2=1: which settings run v2 and which fall back to the old range A (one reason each, first failure wins)
    {
        BootIn ok; ok.serve=true; ok.prefill=true; ok.width=8192; ok.tail=128; ok.nr=4; ok.lanes=1; ok.sh_tc=true;
        assert(!fallback_reason(ok));
        ok.width=4096; assert(!fallback_reason(ok));
        ok.width=256; assert(!fallback_reason(ok));
        auto why=[&](auto edit) { BootIn b=ok; edit(b); const char* r=fallback_reason(b); assert(r); return std::string(r); };
        assert(why([](BootIn& b){b.serve=false;})=="not a serve run (offline or simulator)");
        assert(why([](BootIn& b){b.prefill=false;})=="no W-row prefill (GR8_DS_PREFILL_MIN=0)");
        assert(why([](BootIn& b){b.nr=3;})=="TP is not 4");
        for(int w:{0,32,255,4000,8448})assert(why([w](BootIn& b){b.width=w;})=="GR8_DS_PREFILL_W is not 256..8192 in steps of 256");
        for(int t:{0,64,256})assert(why([t](BootIn& b){b.tail=t;})=="GR8_DS_PF_TAIL is not 128");
        assert(why([](BootIn& b){b.lanes=2;})=="GR8_DS_PF_LANES is not 1");
        assert(why([](BootIn& b){b.late_layer=20;})=="GR8_DS_PF_L20_TAIL=1 moves the late range to layer 20");
        assert(why([](BootIn& b){b.sh_tc=false;})=="GR8_DS_SH_TC=0 (exact output)");
        assert(why([](BootIn& b){b.row_shard=true;})=="GR8_DS_MHC_ROW_SHARD=1 (the 1M-context setup)");
        // the old launcher stack (exact output): two lanes and SH_TC=0; the first failed condition names it
        assert(why([](BootIn& b){b.width=4096; b.lanes=2; b.sh_tc=false;})=="GR8_DS_PF_LANES is not 1");
        // a fallback from a v2 width runs the old path's chunks
        assert(fallback_width(8192)==4096 && fallback_width(4096)==4096 && fallback_width(1024)==1024);
    }
    for (int M = 1; M <= MAX_ROWS; ++M) {
        int rows = 0, tails = 0, next_tail = 0;
        for (int r = 0; r < 4; ++r) {
            auto g = geometry(M,4,r);
            assert(g.owned >= 0 && g.owned <= g.Q);
            assert(g.row0 == r*g.Q);
            rows += g.owned;
            auto t = tail_slice(g);
            assert(t.local >= 0 && t.local + t.count <= g.owned);
            if (t.count) { assert(t.destination == next_tail); next_tail += t.count; }
            tails += t.count;
        }
        assert(rows == M && tails == std::min(M,128));
    }
    Segment segments[] = {{nullptr,nullptr,5120},{nullptr,nullptr,160},
                          {nullptr,nullptr,24},{nullptr,nullptr,24}};
    for (int Q : {1,17,1024,2048}) {
        auto w = wire_layout(Q,segments,4);
        assert(w.count == 4 && w.payload % 16 == 0 && w.stride % 256 == 0);
        assert(w.stride >= w.payload + 8);
        for (int s = 1; s < 4; ++s)
            assert(w.offsets[s] >= w.offsets[s-1] + Q*segments[s-1].row_bytes);
    }
    assert(wire_layout(0,segments,4).count == 0);
    assert(wire_layout(1,segments,9).count == 0);
    for (int flags = 0; flags < 16; ++flags) {
        Order o{layer_order(flags&1,flags&2,flags&4,flags&8)};
        auto expected=o.expected;
        assert(!o.complete()); assert(!o.take(Coll::Boot));
        for (int c : expected) { assert(c<int(Coll::Count)); assert(o.take(Coll(c))); }
        assert(o.complete()); assert(!o.take(Coll::Ffn0));
    }
    std::cout << "ds41_v2: geometry (M=1..8192), ragged tail, wire layout, collective order PASS\n";
}
