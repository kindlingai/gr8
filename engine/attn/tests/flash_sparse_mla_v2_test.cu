// #10 gate: sparse_mla_pf_v2 must be BYTE-IDENTICAL to sparse_mla_pf_512 (memcmp of ctxh, equal error flags), plus A/B timing.
// nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=2 \
//   engine/attn/tests/flash_sparse_mla_v2_test.cu engine/attn/flash_bind.cu -o /tmp/flash_sparse_mla_v2_test
//   (same + -DG6_FLASH_KV_FP8=1: the cache holds 528-B FP8 rows encoded by kvrow_host.h from the same random values)
#include "../flash_attn.cu"
#include "kvrow_host.h"
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CK(x) do { cudaError_t e=(cudaError_t)(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
template<class T> T* alloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> T* upload(const std::vector<T>& v){T* p=alloc<T>(v.size());CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char** argv){
    int device=0;cudaDeviceProp prop{};CK(cudaGetDevice(&device));CK(cudaGetDeviceProperties(&prop,device));
    printf("GPU=%s SM=%d%d optin_smem=%zu v2_smem=%d kv=%s row=%d\n",prop.name,prop.major,prop.minor,prop.sharedMemPerBlockOptin,fa_pf::MLA2_SMEM,g6kv::NAME,g6kv::ROW);
    CK(cudaFuncSetAttribute(fa_pf::sparse_mla_pf_512<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM));
    CK(cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM));
    const bool quick=argc>1&&!strcmp(argv[1],"quick");   // small M, no 32K context, no timing repeats
    std::mt19937 rng(42);std::uniform_real_distribution<float> rnd(-1.f,1.f);
    const int M=quick?512:2048,R=256,D=fa::LAT,K=fa::TOP;
    int fails=0;
    for(int heads:{16,22,32,-32})for(int context:{2600,9216,32768}){
        const bool hole=heads<0;if(hole)heads=-heads;   // -32: same as 32 but page-table entry 0 = -1 (error bit 2)
        if(hole&&context!=9216)continue;
        if(quick&&context==32768)continue;
        std::vector<__half> q((size_t)M*heads*D);
        for(size_t i=0;i<q.size();i++)q[i]=__float2half_rn(rnd(rng)*((i/D)%3==0?4.f:0.5f));
        const int pages=(context+g6kv::PAGE-1)/g6kv::PAGE;
        std::vector<uint8_t> cache((size_t)pages*g6kv::PAGE*g6kv::ROW);
        std::vector<float> lat(D);
        for(size_t i=0;i<(size_t)pages*g6kv::PAGE;i++){for(auto& x:lat)x=rnd(rng);kvrow::encode(lat.data(),&cache[i*g6kv::ROW]);}
        std::vector<int> table(pages),pos(M),idx((size_t)M*K,-1);
        for(int j=0;j<pages;j++)table[j]=pages-1-j; // non-identity page table
        if(hole)table[3]=-1;
        for(int r=0;r<M;r++){
            pos[r]=context-M+r;
            if(pos[r]<0)pos[r]=r;
            // ~85% valid slots (valid first, -1 after, like kp_select); short rows when pos is small
            const int nvalid=std::min(K-256,pos[r]+1);
            for(int j=0;j<nvalid;j++)idx[(size_t)r*K+j]=rng()%(pos[r]+1);
        }
        // Edge rows: empty query, wholly invalid tiles, holes, future IDs, single visible row,
        // tail of 1..3 valid ids (rows 3..5), valid ids only in the last tile, only in the first tile, every tile live.
        for(int j=0;j<K;j++)idx[j]=-1;
        for(int j=0;j<K;j++)idx[K+j]=(j==37)?0:context-1;
        for(int j=0;j<K;j+=3)idx[2*K+j]=-1;
        for(int rr=3;rr<=5;rr++)for(int j=0;j<K;j++)idx[(size_t)rr*K+j]=(j<rr-2)?(int)(rng()%(pos[rr]+1)):-1;
        for(int j=0;j<K;j++)idx[(size_t)6*K+j]=(j>=K-5)?(int)(rng()%(pos[6]+1)):-1;
        for(int j=0;j<K;j++)idx[(size_t)7*K+j]=(j<2)?(int)(rng()%(pos[7]+1)):-1;
        for(int j=0;j<K;j++)idx[(size_t)8*K+j]=(int)(rng()%(pos[8]+1));
        // error bit 1 (id beyond live context): row 9 has pos past the context and one such id; bit 2 case below (table hole)
        pos[9]=context+10;for(int j=0;j<K;j++)idx[(size_t)9*K+j]=(j==0)?context+5:(j<40?(int)(rng()%context):-1);
        G6Attn a{};a.rows=M;a.heads=heads;a.context=context;a.page_size=g6kv::PAGE;a.page_count=pages;a.table_size=pages;
        a.positions=upload(pos);a.block_table=upload(table);a.indices=upload(idx);a.mla_cache=upload(cache);a.error=alloc<unsigned>(1);CK(cudaMemset(a.error,0,4));
        __half* dq=upload(q);__half* old=alloc<__half>(q.size());__half* v2=alloc<__half>(q.size());
        CK(cudaMemset(old,0xCD,q.size()*2));CK(cudaMemset(v2,0xAB,q.size()*2));   // unwritten rows would show
        auto run=[&](bool use_v2){
            for(int start=0;start<M;start+=R){
                const int n=std::min(R,M-start);
                const __half* qi=dq+(size_t)start*heads*D;
                __half* out=(use_v2?v2:old)+(size_t)start*heads*D;
                if(use_v2)fa_pf::sparse_mla_pf_v2<<<n,512,fa_pf::MLA2_SMEM>>>(a,start,qi,out);
                else fa_pf::sparse_mla_pf_512<<<n,512,fa_pf::MLA_SMEM>>>(a,start,qi,out);
            }
            CK(cudaGetLastError());
        };
        run(false);CK(cudaDeviceSynchronize());
        unsigned olderr;CK(cudaMemcpy(&olderr,a.error,4,cudaMemcpyDeviceToHost));CK(cudaMemset(a.error,0,4));
        run(true);CK(cudaDeviceSynchronize());
        unsigned newerr;CK(cudaMemcpy(&newerr,a.error,4,cudaMemcpyDeviceToHost));
        std::vector<__half> ref(q.size()),got(q.size());CK(cudaMemcpy(ref.data(),old,ref.size()*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),v2,got.size()*2,cudaMemcpyDeviceToHost));
        size_t bad=0,first=(size_t)-1;
        for(size_t i=0;i<ref.size();i++)if(memcmp(&ref[i],&got[i],2)){if(!bad)first=i;bad++;}
        float ms[2]={0,0};
        if(!quick){
            cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
            for(int rep=0;rep<4;rep++)for(int arm=0;arm<2;arm++){
                const int use=(rep+arm)%2;CK(cudaEventRecord(e0));run(use);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));float t;CK(cudaEventElapsedTime(&t,e0,e1));ms[use]+=t/4;
            }
            CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));
        }
        printf("%sM=%d H=%d ctx=%d err512=%u errv2=%u mismatched_halves=%zu/%zu %s 512=%.3f ms v2=%.3f ms speedup=%.3fx\n",hole?"[hole] ":"",M,heads,context,olderr,newerr,bad,ref.size(),
               bad||olderr!=newerr?"FAIL":"BYTE-EQUAL",ms[0],ms[1],ms[1]>0?ms[0]/ms[1]:0.f);
        if(bad){const size_t row=first/((size_t)heads*D);printf("  first mismatch: row %zu head %zu d %zu  512=%04x v2=%04x\n",row,first/D%heads,first%D,*(uint16_t*)&ref[first],*(uint16_t*)&got[first]);}
        fflush(stdout);
        if(bad||olderr!=newerr)fails++;
        for(const void* p:{(const void*)a.positions,(const void*)a.block_table,(const void*)a.indices,(const void*)a.mla_cache,(const void*)a.error,(const void*)dq,(const void*)old,(const void*)v2})CK(cudaFree((void*)p));
    }
    puts(fails?"FAIL":"ALL BYTE-EQUAL");
    return fails?1:0;
}
