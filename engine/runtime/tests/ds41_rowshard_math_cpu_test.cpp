// Exact host-oracle per-row check: ds_mhc::consume_cpu + prepare_cpu on 4096 rows
// versus the same functions on four disjoint 1024-row slices, then AG(y).
#include "../../mhc/ds_cpu.h"
#include "../ds41_rowshard.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
namespace {
constexpr int M=4096,H=8,Q=1024,NR=4,LATE=128;
struct A { std::vector<float> stream,pre,post,comb,y;
    A():stream(M*4*H),pre(M*4),post(M*4),comb(M*16),y(M*H) {} };
bool eq(const float* a,const float* b,size_t n) { return std::memcmp(a,b,n*sizeof(float))==0; }
void check_owned(const A& a,const A& b,int r) {
    const int o=r*Q;
    assert(eq(a.stream.data()+o*4*H,b.stream.data()+o*4*H,Q*4*H));
    assert(eq(a.pre.data()+o*4,b.pre.data()+o*4,Q*4));
    assert(eq(a.post.data()+o*4,b.post.data()+o*4,Q*4));
    assert(eq(a.comb.data()+o*16,b.comb.data()+o*16,Q*16));
    assert(eq(a.y.data()+o*H,b.y.data()+o*H,Q*H));
}
void run(int sessions) {
    std::vector<float> fn(24*4*H),base(24),scale(3),norm(H),block(M*H);
    for(size_t i=0;i<fn.size();++i) fn[i]=(int(i%19)-9)*0.001f;
    for(int i=0;i<24;++i) base[i]=(int(i%7)-3)*0.01f;
    scale={0.8f,0.9f,1.1f};
    for(int h=0;h<H;++h) norm[h]=1.f+0.01f*h;
    for(int sess=0;sess<sessions;++sess) {
        A ref; std::array<A,NR> shard;
        for(size_t i=0;i<ref.stream.size();++i) ref.stream[i]=ds_mhc::bf16r((int((i+sess*7)%23)-11)*0.03f);
        for(int m=0;m<M;++m) for(int j=0;j<4;++j) ref.pre[m*4+j]=j==0?1.f:0.f;
        ds_mhc::prepare_cpu(ref.stream.data(),ref.pre.data(),fn.data(),base.data(),scale.data(),norm.data(),M,H,1e-6f,ref.post.data(),ref.comb.data(),ref.y.data());
        for(auto& s:shard) s=ref;
        for(int layer=0;layer<2;++layer) {
            for(int m=0;m<M;++m) for(int h=0;h<H;++h) {
                float sum=0.f;
                for(int p=0;p<NR;++p) sum+=ds_mhc::bf16r((int((m+h+p+sess+layer)%17)-8)*0.02f);
                block[m*H+h]=ds_mhc::bf16r(sum);
            }
            ds_mhc::consume_cpu(ref.stream.data(),block.data(),ref.post.data(),ref.comb.data(),M,H);
            ds_mhc::prepare_cpu(ref.stream.data(),ref.pre.data(),fn.data(),base.data(),scale.data(),norm.data(),M,H,1e-6f,ref.post.data(),ref.comb.data(),ref.y.data());
            for(int r=0;r<NR;++r) {
                auto& s=shard[r]; const int o=r*Q;
                ds_mhc::consume_cpu(s.stream.data()+o*4*H,block.data()+o*H,s.post.data()+o*4,s.comb.data()+o*16,Q,H);
                ds_mhc::prepare_cpu(s.stream.data()+o*4*H,s.pre.data()+o*4,fn.data(),base.data(),scale.data(),norm.data(),Q,H,1e-6f,
                                    s.post.data()+o*4,s.comb.data()+o*16,s.y.data()+o*H);
                check_owned(s,ref,r);
            }
            // AG(y): one owner has each row's exact result.
            for(int r=0;r<NR;++r) for(int o=0;o<NR;++o) if(r!=o)
                std::memcpy(shard[r].y.data()+o*Q*H,shard[o].y.data()+o*Q*H,Q*H*sizeof(float));
            for(const auto& s:shard) assert(eq(s.y.data(),ref.y.data(),M*H));
        }
        const int o=M-LATE; const auto owner=shard[NR-1];
        for(auto& s:shard) {
            std::memcpy(s.stream.data()+o*4*H,owner.stream.data()+o*4*H,LATE*4*H*sizeof(float));
            std::memcpy(s.pre.data()+o*4,owner.pre.data()+o*4,LATE*4*sizeof(float));
            std::memcpy(s.post.data()+o*4,owner.post.data()+o*4,LATE*4*sizeof(float));
            std::memcpy(s.comb.data()+o*16,owner.comb.data()+o*16,LATE*16*sizeof(float));
            std::memcpy(s.y.data()+o*H,owner.y.data()+o*H,LATE*H*sizeof(float));
            assert(eq(s.stream.data()+o*4*H,ref.stream.data()+o*4*H,LATE*4*H));
            assert(eq(s.pre.data()+o*4,ref.pre.data()+o*4,LATE*4));
            assert(eq(s.post.data()+o*4,ref.post.data()+o*4,LATE*4));
            assert(eq(s.comb.data()+o*16,ref.comb.data()+o*16,LATE*16));
            assert(eq(s.y.data()+o*H,ref.y.data()+o*H,LATE*H));
        }
    }
    std::printf("MHC_CPU_BIT_EQUAL sessions=%d rows=%d Q=%d layers=2 tail=%d state_all_ranks=1\n",sessions,M,Q,LATE);
}
}
int main() { assert(ds41::rowshard::chunk_on(true,M,LATE,NR)); run(8); run(32); }
