#include "../ds41_rowshard.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int NR=4, M=4096, Q=1024, LATE=128;
struct State {
    std::array<uint32_t,4> trunk{},pre{},post{};
    std::array<uint32_t,16> comb{};
    std::array<uint32_t,24> mix{};
    uint32_t y=0;
    bool operator==(const State& o) const {
        return trunk==o.trunk && pre==o.pre && post==o.post && comb==o.comb && mix==o.mix && y==o.y;
    }
};
uint32_t rot(uint32_t x,unsigned k) { return (x<<k)|(x>>(32-k)); }
State seed(int sess,int row) {
    State s; uint32_t x=uint32_t(sess*1000003+row*977+11);
    for(auto* a:{&s.trunk,&s.pre,&s.post}) for(auto& v:*a) { x=rot(x^0x9e3779b9u,7)+17; v=x; }
    for(auto& v:s.comb) { x=rot(x^0x7f4a7c15u,11)+23; v=x; }
    for(auto& v:s.mix) { x=rot(x^0x6a09e667u,13)+29; v=x; }
    s.y=x; return s;
}
void step(State& s,uint32_t sum,int layer) {
    for(int j=0;j<4;++j) { s.trunk[j]=rot(s.trunk[j]+sum+uint32_t(layer*17+j),7); s.pre[j]^=s.trunk[j]; s.post[j]+=s.pre[j]; }
    for(int j=0;j<16;++j) s.comb[j]=rot(s.comb[j]^s.trunk[j&3],9);
    for(int j=0;j<24;++j) s.mix[j]+=s.comb[j&15]+uint32_t(j);
    s.y=s.trunk[0]^s.pre[1]^s.mix[2];
}
bool run_sessions(int sessions,const char* mutant) {
    for(int sess=0;sess<sessions;++sess) {
        std::vector<State> base(M); std::array<std::vector<State>,NR> rank;
        for(auto& v:rank) v.resize(M);
        for(int m=0;m<M;++m) { base[m]=seed(sess,m); for(auto& v:rank) v[m]=base[m]; }
        for(int c=0;c<42;++c) {
            for(int m=0;m<M;++m) {
                uint32_t sum=0; for(int p=0;p<NR;++p) sum+=uint32_t(sess*131+c*19+m*7+p*3); // rank-order reduction
                step(base[m],sum,c);
                const int owner=m/Q;
                step(rank[owner][m],sum,c); // only this rank runs its Q-row k_step
            }
            for(int m=0;m<M;++m) { // AG(y), not AG(block sum)
                const uint32_t y=rank[m/Q][m].y;
                if(std::strcmp(mutant,"no-y-ag")!=0) for(auto& v:rank) v[m].y=y;
                assert(y==base[m].y);
            }
            for(const auto& v:rank) for(int m=0;m<M;++m) if(v[m].y!=base[m].y) return false;
        }
        for(int m=0;m<M;++m) assert(rank[m/Q][m]==base[m]);
        // One-time boundary transfer before compact: every rank needs all late-row state, not just y.
        for(int m=M-LATE;m<M;++m) {
            if(std::strcmp(mutant,"drop-tail")!=0) {
                const State owner=rank[std::strcmp(mutant,"wrong-owner")==0 ? 0 : NR-1][m];
                for(auto& v:rank) v[m]=owner;
            }
        }
        for(auto& v:rank) for(int m=M-LATE;m<M;++m) if(!(v[m]==base[m])) return false;
        // Drop-y-AG mutant: an unowned early row is stale in its peer before the boundary transfer.
        assert(rank[0][Q+1].trunk!=base[Q+1].trunk);
    }
    std::printf("ROW_SHARD_CPU_PASS sessions=%d Q=%d late=%d tail_state_all_ranks=1\n",sessions,Q,LATE);
    return true;
}
}
int main(int argc,char** argv) {
    const char* mutant=argc>1 ? argv[1] : "none";
    using namespace ds41::rowshard;
    assert(boot_ok(M,LATE,NR,true,true));
    assert(!boot_ok(2048,LATE,NR,true,true) && !boot_ok(M,LATE,NR,false,true));
    assert(chunk_on(true,M,LATE,NR) && !chunk_on(true,2048,LATE,NR));
    assert(row0(M,NR,3)==3072 && state_bytes(LATE,5120,2)==6578176);
    const int nc=85,k=42; std::vector<uint32_t> sizes(nc,uint32_t(M*5120*2));
    auto j=ds41::rs::job(sizes,{},nc,k,M,NR,5120*2,uint32_t(state_bytes(LATE,5120,2)));
    assert(j.sizes.size()==size_t(nc+k+1) && j.sizes[nc+k]==6578176);
    bool order_ok=false; for(size_t i=1;i<j.order.size();++i) if(j.order[i]==nc+k) order_ok=j.order[i-1]==nc+k-1;
    assert(order_ok);
    bool red=false;
    for(int sessions:{8,32}) if(!run_sessions(sessions,mutant)) {
        std::printf("ROW_SHARD_CPU_RED sessions=%d mutant=%s\n",sessions,mutant);
        red=true;
    }
    return red ? 2 : 0;
}
