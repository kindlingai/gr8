#include "../runtime/ds41_v2_width.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
using ds41::v2::CommWidths;
int main() {
    CommWidths w;
    assert(CommWidths::decode({0},w.rs) && w.rs.empty());
    for(const auto& bad:std::vector<std::vector<int64_t>>{{},{-1},{0,8},{8,0},{int64_t(INT_MAX)+1}})
        assert(!CommWidths::decode(bad,w.rs));
    assert(CommWidths::decode({8,48,192,1024},w.rs));
    assert(CommWidths::decode({48,192},w.pack));
    FILE* log=std::tmpfile(); assert(log);
    for(uint64_t r=0;r<12;++r) {
        w.select(r,0,3,log);
        assert(w.rs_blocks==w.rs[r%4] && w.pack_blocks==w.pack[r%2]);
        long pos=std::ftell(log); w.select(r,0,3,log); assert(std::ftell(log)==pos);
    }
    w.select(12,1,3,log); w.select(11,0,3,log);
    long pos=std::ftell(log); w.select(12,1,3,log); assert(std::ftell(log)==pos);
    assert(w.rs_blocks==8 && w.pack_blocks==48 && w.seen.size()==2);
    w.select(std::numeric_limits<uint64_t>::max(),1,3,log);
    assert(w.rs_blocks==1024 && w.pack_blocks==192 && w.seen.size()==2);
    std::rewind(log); int lines=0; char buf[256];
    while(std::fgets(buf,sizeof(buf),log)) { assert(std::string(buf).find("rank=3")!=std::string::npos); ++lines; }
    assert(lines==28); std::fclose(log);
    CommWidths off; log=std::tmpfile(); assert(log); off.select(0,0,0,log);
    assert(off.rs_blocks==0 && off.pack_blocks==8 && off.seen.empty() && std::ftell(log)==0); std::fclose(log);
    std::cout<<"ds41_v2_width: defaults, validation, request modulo, independent lists, interleaved slots and bounded once-per-request logs PASS\n";
}
