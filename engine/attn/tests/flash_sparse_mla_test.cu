// nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=2 \
//   engine/attn/tests/flash_sparse_mla_test.cu engine/attn/flash_bind.cu -o /tmp/flash_sparse_mla_test
// Same-input numerical/performance gate against the current QK + softmax + PV prefill path.
#include "../flash_attn.cu"
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CK(x) do { cudaError_t e=(cudaError_t)(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
template<class T> T* alloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> T* upload(const std::vector<T>& v){T* p=alloc<T>(v.size());CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
    int device=0;cudaDeviceProp prop{};CK(cudaGetDevice(&device));CK(cudaGetDeviceProperties(&prop,device));
    printf("GPU=%s SM=%d%d optin_smem=%zu kernel_smem=%d\n",prop.name,prop.major,prop.minor,prop.sharedMemPerBlockOptin,fa_pf::MLA_SMEM);
    CK(cudaFuncSetAttribute(fa_pf::sparse_mla_pf<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM));
    std::mt19937 rng(42);std::uniform_real_distribution<float> rnd(-1.f,1.f);
    constexpr int M=2048,R=256,D=fa::LAT,K=fa::TOP;
    for(int heads:{16,22,32})for(int context:{9216,32768}){
        std::vector<__half> q((size_t)M*heads*D);
        for(size_t i=0;i<q.size();i++)q[i]=__float2half_rn(rnd(rng)*((i/D)%3==0?4.f:0.5f));
        const int pages=(context+g6kv::PAGE-1)/g6kv::PAGE;
        std::vector<__nv_bfloat16> cache((size_t)pages*g6kv::PAGE*D);
        for(auto& x:cache)x=__float2bfloat16_rn(rnd(rng));
        std::vector<int> table(pages),pos(M),idx((size_t)M*K,-1);
        for(int j=0;j<pages;j++)table[j]=pages-1-j; // non-identity page table
        for(int r=0;r<M;r++){
            pos[r]=context-M+r;
            for(int j=0;j<K-256;j++)idx[(size_t)r*K+j]=rng()%(pos[r]+1);
        }
        // Empty query, wholly invalid tiles, holes, future IDs and a single visible row.
        for(int j=0;j<K;j++)idx[j]=-1;
        for(int j=0;j<K;j++)idx[K+j]=(j==37)?0:context-1;
        for(int j=0;j<K;j+=3)idx[2*K+j]=-1;
        G6Attn a{};a.rows=M;a.heads=heads;a.context=context;a.page_size=g6kv::PAGE;a.page_count=pages;a.table_size=pages;
        a.positions=upload(pos);a.block_table=upload(table);a.indices=upload(idx);a.mla_cache=upload(cache);a.error=alloc<unsigned>(1);CK(cudaMemset(a.error,0,4));
        __half* dq=upload(q);__half* old=alloc<__half>(q.size());__half* fused=alloc<__half>(q.size());
        float* prob=alloc<float>((size_t)R*heads*K);
        auto run=[&](bool use_fused){
            for(int start=0;start<M;start+=R){
                const __half* qi=dq+(size_t)start*heads*D;
                __half* out=(use_fused?fused:old)+(size_t)start*heads*D;
                if(use_fused)fa_pf::sparse_mla_pf<<<R,256,fa_pf::MLA_SMEM>>>(a,start,qi,out);
                else{
                    if(heads>16)fa_pf::sparse_qk_pf<2><<<dim3(K/64,R),128>>>(a,start,qi,prob);
                    else fa_pf::sparse_qk_pf<1><<<dim3(K/64,R),128>>>(a,start,qi,prob);
                    fa::normalize<<<R*heads,256>>>(a,prob);
                    if(heads>16)fa_pf::sparse_pv_pf<2><<<dim3(D/64,R),128>>>(a,start,prob,out);
                    else fa_pf::sparse_pv_pf<1><<<dim3(D/64,R),128>>>(a,start,prob,out);
                }
            }
            CK(cudaGetLastError());
        };
        run(false);CK(cudaDeviceSynchronize());
        unsigned olderr;CK(cudaMemcpy(&olderr,a.error,4,cudaMemcpyDeviceToHost));CK(cudaMemset(a.error,0,4));
        run(true);CK(cudaDeviceSynchronize());
        unsigned newerr;CK(cudaMemcpy(&newerr,a.error,4,cudaMemcpyDeviceToHost));
        if(olderr!=16||newerr!=olderr){fprintf(stderr,"error flag mismatch old=%u new=%u expected=16\n",olderr,newerr);return 1;}
        std::vector<__half> ref(q.size()),got(q.size());CK(cudaMemcpy(ref.data(),old,ref.size()*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),fused,got.size()*2,cudaMemcpyDeviceToHost));
        double diff2=0,ref2=0,maxabs=0;size_t same=0;bool finite=true;
        for(size_t i=0;i<ref.size();i++){
            const double x=__half2float(ref[i]),y=__half2float(got[i]),d=x-y;
            finite=finite&&std::isfinite(x)&&std::isfinite(y);diff2+=d*d;ref2+=x*x;maxabs=fmax(maxabs,fabs(d));same+=x==y;
        }
        const double rel=sqrt(diff2/fmax(ref2,1e-30));
        cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));float times[2]={};
        for(int rep=0;rep<4;rep++)for(int arm=0;arm<2;arm++){
            int use=(rep+arm)%2;CK(cudaEventRecord(e0));run(use);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));float ms;CK(cudaEventElapsedTime(&ms,e0,e1));times[use]+=ms/4;
        }
        unsigned err;CK(cudaMemcpy(&err,a.error,4,cudaMemcpyDeviceToHost));
        printf("M=%d H=%d ctx=%d current=%.3f ms fused=%.3f ms speedup=%.3fx rel_l2=%.6g max_abs=%.6g exact=%.2f%% finite=%d err=%u\n",M,heads,context,times[0],times[1],times[0]/times[1],rel,maxabs,100.*same/ref.size(),finite,err);fflush(stdout);
        // Same 3e-3 relative L2 ceiling as the DSA block gate. Max-abs also guards outliers.
        if(!finite||err!=16||rel>3e-3||maxabs>0.01)return 1;
        CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));
        for(const void* p:{(const void*)a.positions,(const void*)a.block_table,(const void*)a.indices,(const void*)a.mla_cache,(const void*)a.error,(const void*)dq,(const void*)old,(const void*)fused,(const void*)prob})CK(cudaFree((void*)p));
    }
    puts("ALL PASS");
}
