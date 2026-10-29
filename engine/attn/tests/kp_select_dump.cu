// kp_select exact + speed screen: random / tied / zero pool scores at several contexts and row positions.
// Dumps every selected ID to OUT so two builds (old and new kpool.cuh) can be compared byte for byte.
// From the repo root:
// nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=2 [-DKP_SELT=256] \
//   engine/attn/tests/kp_select_dump.cu engine/attn/flash_bind.cu -o kp_select_dump
// Run: kp_select_dump OUT [reps=20]
#include "../flash_attn.cu"
#include <vector>
#include <random>
#include <numeric>
#include <cstdio>
#define CK(x) do{cudaError_t ck_e=(cudaError_t)(x);if(ck_e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(ck_e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>T* upload(const std::vector<T>&v){T*p=alloc<T>(v.size());CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char**argv){
 if(argc<2){fprintf(stderr,"usage: kp_select_dump OUT [reps]\n");return 2;}
 FILE* out=fopen(argv[1],"wb");const int reps=argc>2?atoi(argv[2]):20;
 printf("SELT=%d\n",fa::SELT);
 std::mt19937 rng(4242);std::uniform_real_distribution<float>d(-2,2);
 for(int context:{2051,8192,9000,32772,131072})for(int spread=0;spread<2;spread++)for(int kind=0;kind<3;kind++){
  const int M=256,start=3,pages=(context+63)/64,stride=fa::pool_stride(context);
  std::vector<int>tab(pages);std::iota(tab.begin(),tab.end(),0);
  std::vector<int>pos(start+M,0);   // spread=0: the last M positions (decode-like); 1: spread over the context (prefill chunk)
  for(int r=0;r<M;r++)pos[start+r]=spread?(int)((long long)(r+1)*context/M)-1:context-M+r;
  std::vector<float>sc((size_t)M*stride);
  for(auto&v:sc){float x=d(rng);v=kind==0?x:kind==1?roundf(x*4)/4:0.f;}   // 1: heavy ties; 2: all zero
  G6Attn a{};a.rows=start+M;a.context=context;a.page_size=64;a.page_count=pages;a.table_size=pages;
  a.block_table=upload(tab);a.positions=upload(pos);a.error=upload(std::vector<unsigned>{0});
  a.indices=upload(std::vector<int>((size_t)(start+M)*fa::TOP,-123456));
  float* s=upload(sc);
  fa::kp_select<<<M,fa::SELT>>>(a,start,s,stride);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  std::vector<int>ids((size_t)(start+M)*fa::TOP);CK(cudaMemcpy(ids.data(),a.indices,ids.size()*4,cudaMemcpyDeviceToHost));
  fwrite(ids.data(),4,ids.size(),out);
  cudaEvent_t b,e;CK(cudaEventCreate(&b));CK(cudaEventCreate(&e));CK(cudaEventRecord(b));
  for(int rep=0;rep<reps;rep++)fa::kp_select<<<M,fa::SELT>>>(a,start,s,stride);
  CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,b,e));
  printf("ctx=%d spread=%d kind=%d us/call=%.1f\n",context,spread,kind,1000*ms/reps);fflush(stdout);
  for(const void*p:{(const void*)a.block_table,(const void*)a.positions,(const void*)a.error,(const void*)a.indices,(const void*)s})CK(cudaFree((void*)p));
 }
 fclose(out);
}
