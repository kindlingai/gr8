// Host-only request selection for the two independent v2 wire width sweeps.
#pragma once
#include <climits>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

namespace ds41::v2 {
struct CommWidths {
    std::vector<int> rs, pack;
    int rs_blocks=0, pack_blocks=8;
    // At most one entry per scheduler slot, not one per historical request.
    // This also handles interleaved chunks and a prefix-hit first chunk != 0.
    std::map<unsigned,uint64_t> seen;
    static bool decode(const std::vector<int64_t>& values,std::vector<int>& out) {
        out.clear();
        if(values.size()==1 && values[0]==0)return true;
        if(values.empty())return false;
        for(auto v:values)if(v<=0 || v>INT_MAX)return false;
        for(auto v:values)out.push_back(int(v));
        return true;
    }
    void select(uint64_t request,unsigned slot,int rank,FILE* log=stderr) {
        if(rs.empty() && pack.empty())return;
        rs_blocks=rs.empty()?0:rs[request%rs.size()];
        pack_blocks=pack.empty()?8:pack[request%pack.size()];
        auto it=seen.find(slot);
        if(it!=seen.end() && it->second==request)return;
        seen[slot]=request;
        if(!rs.empty())std::fprintf(log,"# ds41_v2_rs_blocks request=%llu blocks=%d rank=%d\n",
            (unsigned long long)request,rs_blocks,rank);
        if(!pack.empty())std::fprintf(log,"# ds41_v2_pack_blocks request=%llu blocks=%d rank=%d\n",
            (unsigned long long)request,pack_blocks,rank);
    }
};
} // namespace ds41::v2
