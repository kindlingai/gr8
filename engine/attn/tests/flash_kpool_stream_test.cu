// Exact and speed screen for the streaming Flash pool scorer against the existing hi/lo scorer.
// From the repo root:
// nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=2 \
//   engine/attn/tests/flash_kpool_stream_test.cu engine/attn/flash_bind.cu -o flash_kpool_stream_test
// Run: flash_kpool_stream_test [reps=5]. Includes nonzero row offsets, partial tiles, causal masks,
// random signed weights, zero/tied scores, exact score bytes and selected IDs. No full-chain gate.
#include "../flash_attn.cu"
#include <vector>
#include <random>
#include <numeric>
#include <cstdio>
#include <cstring>
#define CK(x) do{cudaError_t ck_e=(cudaError_t)(x);if(ck_e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(ck_e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>T* upload(const std::vector<T>&v){T*p=alloc<T>(v.size());CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char**argv){
 const int reps=argc>1?atoi(argv[1]):5;
 cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));
 printf("GPU=%s SM=%d%d shared_per_sm=%zu regs_per_sm=%d reps=%d\n",prop.name,prop.major,prop.minor,prop.sharedMemPerMultiprocessor,prop.regsPerMultiprocessor,reps);
 cudaFuncAttributes attr[2];CK(cudaFuncGetAttributes(&attr[0],fa_pf::kp_scores_tc<2,G6Attn>));CK(cudaFuncGetAttributes(&attr[1],fa_pf::kp_scores_tc_stream<2,G6Attn>));
 for(int k=0;k<2;k++)printf("RESOURCE arm=%d regs=%d shared=%zu local=%zu\n",k,attr[k].numRegs,attr[k].sharedSizeBytes,attr[k].localSizeBytes);
 std::mt19937 rng(1701);std::uniform_real_distribution<float>d(-2,2);int cases=0;
 for(int context:{2051,9000,32772,131072,300000})for(int M:{1,4,65,256})for(int kind=0;kind<2;kind++){
  const int start=3,pages=(context+63)/64,pools=context/4,stride=fa::pool_stride(context);
  std::vector<unsigned char>cache((size_t)pages*g6kv::IDX_PAGE,0);
  for(int p=0;p<pools;p++)for(int c=0;c<128;c++){
   const float v=kind?0.f:d(rng);
   memcpy(cache.data()+(size_t)(p/16)*g6kv::IDX_PAGE+g6kv::IDX_POOL_OFF+(p%16*128+c)*4,&v,4);
  }
  std::vector<int>tab(pages);std::iota(tab.begin(),tab.end(),0);
  std::vector<int>pos(start+M,0);for(int r=0;r<M;r++)pos[start+r]=context-M+r;
  std::vector<__half>q((size_t)(start+M)*4096),weight((size_t)(start+M)*32);
  for(auto&v:q)v=__float2half(d(rng));for(auto&v:weight)v=__float2half(d(rng));
  G6Attn a{};a.rows=start+M;a.context=context;a.page_size=64;a.page_count=pages;a.table_size=pages;
  a.block_table=upload(tab);a.positions=upload(pos);a.index_cache=upload(cache);a.index_q=upload(q);a.index_weight=upload(weight);a.error=upload(std::vector<unsigned>{0});
  a.indices=upload(std::vector<int>((size_t)(start+M)*fa::TOP,-123456));
  float*scores[2]={alloc<float>((size_t)M*stride),alloc<float>((size_t)M*stride)};
  std::vector<float>expected((size_t)M*stride),actual(expected.size());
  std::vector<int>ids[2]={std::vector<int>((size_t)(start+M)*fa::TOP),std::vector<int>((size_t)(start+M)*fa::TOP)};
  auto run=[&](int arm){if(arm)fa_pf::kp_scores_tc_stream<2><<<dim3(stride/32,(M+63)/64),128>>>(a,start,M,scores[arm],stride);
   else fa_pf::kp_scores_tc<2><<<dim3(stride/32,(M+63)/64),128>>>(a,start,M,scores[arm],stride);CK(cudaGetLastError());};
  for(int arm=0;arm<2;arm++){
   CK(cudaMemset(scores[arm],0xa5,expected.size()*4));run(arm);CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(arm?actual.data():expected.data(),scores[arm],expected.size()*4,cudaMemcpyDeviceToHost));
   fa::kp_select<<<M,fa::SELT>>>(a,start,scores[arm],stride);CK(cudaGetLastError());
   CK(cudaMemcpy(ids[arm].data(),a.indices,ids[arm].size()*4,cudaMemcpyDeviceToHost));
  }
  unsigned err;CK(cudaMemcpy(&err,a.error,4,cudaMemcpyDeviceToHost));
  if(err||memcmp(expected.data(),actual.data(),expected.size()*4)||ids[0]!=ids[1]){
   fprintf(stderr,"FAIL ctx=%d rows=%d kind=%d error=%u\n",context,M,kind,err);
   for(size_t j=0;j<expected.size();j++)if(memcmp(&expected[j],&actual[j],4)){fprintf(stderr,"score %zu ref=%a got=%a\n",j,expected[j],actual[j]);break;}return 1;
  }
  float sums[2]={};cudaEvent_t b,e;CK(cudaEventCreate(&b));CK(cudaEventCreate(&e));
  for(int rep=0;rep<reps;rep++)for(int v=0;v<2;v++){const int arm=(rep+v)%2;CK(cudaEventRecord(b));run(arm);CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,b,e));sums[arm]+=ms/reps;}
  printf("PASS ctx=%d rows=%d kind=%d base_ms=%.6f stream_ms=%.6f speedup=%.3f score_bytes=%zu ids=%zu\n",context,M,kind,sums[0],sums[1],sums[0]/sums[1],expected.size()*4,ids[0].size());fflush(stdout);
  CK(cudaEventDestroy(b));CK(cudaEventDestroy(e));
  for(const void*p:{(const void*)a.block_table,(const void*)a.positions,(const void*)a.index_cache,(const void*)a.index_q,(const void*)a.index_weight,(const void*)a.error,(const void*)a.indices,(const void*)scores[0],(const void*)scores[1]})CK(cudaFree((void*)p));cases++;
 }
 printf("ALL PASS cases=%d: every FP32 score byte, selected ID, and untouched score sentinel agrees\n",cases);
}
