// Verify query-row slicing and stock candidate handoff with the real indexer
// kernels. No model/fabric needed. Build with nvcc -O2 -std=c++17 -arch=sm_120a.
#include "../ds41_idx.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#define CK(x) do { auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
using namespace ds41i;
struct Arena {
    std::vector<void*> pointers;
    template<class T> T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));pointers.push_back(p);CK(cudaMemset(p,0,n*sizeof(T)));return p;}
    ~Arena(){for(auto p:pointers)CK(cudaFree(p));}
};
unsigned seed=18;
unsigned rand32(){return seed=seed*1664525+1013904223;}
void same(const char* what,const int32_t* a,const int32_t* b,size_t n){
    std::vector<int32_t> ah(n),bh(n);CK(cudaMemcpy(ah.data(),a,n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(bh.data(),b,n*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<n;++i)if(ah[i]!=bh[i]){fprintf(stderr,"FAIL %s at %zu: %d != %d\n",what,i,ah[i],bh[i]);exit(1);}
    printf("%s: %zu list entries PASS\n",what,n);
}
void run(int M,int keys){
    Arena a;const int nr=4,Q=(M+nr-1)/nr,slot=2,rpp=(keys+M+3)&~3;const size_t page=size_t(rpp)*ICA_ROW;
    auto* pool=a.alloc<uint8_t>(3*page);std::vector<uint8_t> ph(3*page);
    for(int s=0;s<3;++s){for(size_t k=0;k<size_t(rpp)*IK_CODES;++k)ph[size_t(s)*page+k]=rand32()>>24;
        std::fill(ph.begin()+size_t(s)*page+size_t(rpp)*IK_CODES,ph.begin()+(s+1)*page,126);}
    CK(cudaMemcpy(pool,ph.data(),ph.size(),cudaMemcpyHostToDevice));auto* pages=a.alloc<int32_t>(1);CK(cudaMemcpy(pages,&slot,4,cudaMemcpyHostToDevice));
    auto* qb=a.alloc<bf16>(size_t(M)*IH*ID);auto* wp=a.alloc<bf16>(size_t(M)*IH);
    std::vector<bf16> qh(size_t(M)*IH*ID),wh(size_t(M)*IH);
    for(auto&v:qh)v=bf16(float(int(rand32()%2001)-1000)/500);for(auto&v:wh)v=bf16(float(rand32()%1000)/1000);
    CK(cudaMemcpy(qb,qh.data(),qh.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(wp,wh.data(),wh.size()*2,cudaMemcpyHostToDevice));
    auto* pos=a.alloc<int>(M);auto* len=a.alloc<int32_t>(M);std::vector<int> pp(M),ll(M);
    for(int r=0;r<M;++r){pp[r]=keys+r;ll[r]=keys+r+1;}
    CK(cudaMemcpy(pos,pp.data(),M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(len,ll.data(),M*4,cudaMemcpyHostToDevice));
    void* sfmem=a.alloc<char>(Scratch::bytes(M,rpp));void* somem=a.alloc<char>(Scratch::bytes(M,rpp));
    auto sf=Scratch::carve(sfmem,M,rpp,M);auto* full=a.alloc<int32_t>(size_t(M)*TOPK);auto* own=a.alloc<int32_t>(size_t(Q)*TOPK);
    auto* gathered=a.alloc<int32_t>(size_t(M)*TOPK);auto* candg=a.alloc<int32_t>(size_t(M)*CAND_BLOCKS);
    IndexIn in{};in.q_b=qb;in.wproj=wp;in.inv=rope_inv_ds41();in.R=M;in.klen=len;in.max_klen=keys+M;
    in.k=ica_map(pool,pages,rpp,page);in.dpos=pos;in.logits_mma=1;in.topk_mc=1;
    stage_index_dense(in,sf,full,TOPK,0,true);stage_cand_blocks(in,sf,0,true);
    for(int rank=0;rank<nr;++rank){int r0=rank*Q,n=std::max(0,std::min(Q,M-r0));if(!n)continue;
        auto row=in;row.q_b+=size_t(r0)*IH*ID;row.wproj+=size_t(r0)*IH;row.klen+=r0;row.dpos+=r0;row.R=n;
        auto so=Scratch::carve(somem,M,rpp,n);stage_index_dense(row,so,own,TOPK,0,true);stage_cand_blocks(row,so,0,true);
        CK(cudaMemcpyAsync(gathered+size_t(r0)*TOPK,own,size_t(n)*TOPK*4,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(candg+size_t(r0)*CAND_BLOCKS,so.cand,size_t(n)*CAND_BLOCKS*4,cudaMemcpyDeviceToDevice));
    }
    same("owned query top-k gather vs full",gathered,full,size_t(M)*TOPK);
    same("owned candidate gather vs full",candg,sf.cand,size_t(M)*CAND_BLOCKS);
    // Late stock consumer after handoff: final rows, candidate selection from
    // the gathered array, no special own-row knowledge in the consumer.
    const int tail=std::min(128,M),r0=M-tail;auto late=in;late.q_b+=size_t(r0)*IH*ID;late.wproj+=size_t(r0)*IH;late.klen+=r0;late.dpos+=r0;late.R=tail;
    auto sl=Scratch::carve(somem,M,rpp,tail);auto* lf=a.alloc<int32_t>(size_t(tail)*TOPK);auto* lg=a.alloc<int32_t>(size_t(tail)*TOPK);
    stage_index_cand(late,sl,sf.cand+size_t(r0)*CAND_BLOCKS,lf,TOPK,0);
    stage_index_cand(late,sl,candg+size_t(r0)*CAND_BLOCKS,lg,TOPK,0);
    same("stock late consumer after handoff",lg,lf,size_t(tail)*TOPK);
    printf("M=%d Q=%d keys=%d nonzero-slot=%d PASS\n",M,Q,keys,slot);
}
int main(){run(129,512);run(65,20032);run(8192,64);CK(cudaDeviceSynchronize());puts("PASS v2 owned index and candidate handoff; fabric/model eval NOT measured");}
