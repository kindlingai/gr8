#include "../runtime/ds41_v2.h"
#include <cassert>
#include <iostream>
using namespace ds41::v2;
int main() {
    assert(boot_ok(4096,128,4,1,21));
    assert(boot_ok(8192,128,4,1,21));
    assert(!boot_ok(4096,128,4,2,21));
    assert(!boot_ok(4096,128,3,1,21));
    assert(!boot_ok(4096,128,4,1,20));
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
