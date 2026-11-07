// Reuse gr8's mHC fixture; test row OWNER slices instead of single-row launches.
// No model-weight or quality claim. Timing is not measured by this fixture.
#define main ds41_mhc_fixture_main
#include "ds_mhc_mtest.cu"
#undef main
#include "../ds41_v2.h"
#include <cassert>
template<class TT> void owner_case(int M) {
    constexpr int H=5120;
    std::vector<int> tok(M); for(int i=0;i<M;++i)tok[i]=i%16;
    std::vector<__nv_bfloat16> emb(16*H),norm(H);
    std::vector<float> fn(24*4*H),base(24),scale(3,0.1f),block(size_t(M)*H);
    for(size_t i=0;i<emb.size();++i)emb[i]=__float2bfloat16_rn(float(int(i%97)-48)/50);
    for(int i=0;i<H;++i)norm[i]=__float2bfloat16_rn(1.f+float(i%3)/32);
    for(size_t i=0;i<fn.size();++i)fn[i]=float(int(i%13)-6)/1000;
    for(int i=0;i<24;++i)base[i]=float(i%7)/10;
    for(size_t i=0;i<block.size();++i)block[i]=float(int(i%83)-41)/50;
    auto* dt=up(tok); auto* de=up(emb); auto* dn=up(norm);
    auto* df=up(fn); auto* db=up(base); auto* ds=up(scale); auto* dout=up(block);
    const ds_mhc::Weights<float> w{df,db,ds,dn};
    SetT<TT> full(H,M,false);
    ds_mhc::embed(full.b,dt,de,M,H,0);
    ds_mhc::prepare<float>(full.b,w,M,H,1.e-6f,full.y,0);
    ds_mhc::consume_prepare<float>(full.b,flash_mhc::Fp32Block{dout},w,M,H,1.e-6f,full.y,0);
    ds_mhc::consume(full.b,flash_mhc::Fp32Block{dout},M,H,0);
    ds_mhc::prepare<float>(full.b,w,M,H,1.e-6f,full.y,0); // Engram boundary's separate prepare
    CK(cudaDeviceSynchronize());
    const auto fy=down(full.y,size_t(M)*H); const auto ft=down_wide(full.b.streams,size_t(M)*4*H);
    size_t bad=0;
    for(int p=0;p<4;++p) {
        const auto g=ds41::v2::geometry(M,4,p); SetT<TT> own(H,g.Q,false);
        if(!g.owned)continue;
        ds_mhc::embed(own.b,dt+g.row0,de,g.owned,H,0);
        ds_mhc::prepare<float>(own.b,w,g.owned,H,1.e-6f,own.y,0);
        const flash_mhc::Fp32Block out{dout+size_t(g.row0)*H};
        ds_mhc::consume_prepare<float>(own.b,out,w,g.owned,H,1.e-6f,own.y,0);
        ds_mhc::consume(own.b,out,g.owned,H,0);
        ds_mhc::prepare<float>(own.b,w,g.owned,H,1.e-6f,own.y,0);
        CK(cudaDeviceSynchronize());
        const auto oy=down(own.y,size_t(g.owned)*H),ot=down_wide(own.b.streams,size_t(g.owned)*4*H);
        for(size_t j=0;j<oy.size();++j)bad+=std::memcmp(&oy[j],&fy[size_t(g.row0)*H+j],2)!=0;
        for(size_t j=0;j<ot.size();++j)bad+=ot[j]!=ft[size_t(g.row0)*4*H+j];
        const auto done=down(own.b.done,ds_mhc::done_words(g.Q)); for(auto x:done)assert(x==0);
    }
    printf("v2 mHC ownership M=%d trunk_bytes=%zu differences=%zu\n",M,sizeof(TT),bad);
    assert(bad==0);
    cudaFree(dt);cudaFree(de);cudaFree(dn);cudaFree(df);cudaFree(db);cudaFree(ds);cudaFree(dout);
}
int main() {
    for(int M:{1,3,5,17,129,257,4096,8192}) {owner_case<float>(M);owner_case<__nv_bfloat16>(M);}
    puts("v2 mHC owned slices: embed, fused and separate boundaries PASS");
}
