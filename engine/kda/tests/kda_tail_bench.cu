// Packed f_a/g_a/b full-M gate vs the original MX-row projection loop.
// nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DDEC_S2 kda_tail_bench.cu -o kda_tail_bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include "../proj_prefill.cuh"
#include "../../mhc/mhc.cuh"
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
using namespace kda;
template<class T> T* alloc(size_t n) {T* p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> T* upload(const std::vector<T>& v) {T* p=alloc<T>(v.size());CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main() {
 constexpr int K=4096,LD=336;
 std::mt19937 rng(5021);std::normal_distribution<float> nd(0.f,0.2f);
 auto weights=[&](int n,bool mx) {
  const size_t bytes=mx?(size_t)(n/16)*(K/32)*528:(size_t)n*K*2;
  std::vector<uint8_t> w(bytes);
  if(mx) {
   for(size_t o=0;o<bytes;o+=528) {
    for(int j=0;j<512;++j) {unsigned c=rng()%127;w[o+j]=(uint8_t)(c|((rng()&1)<<7));}
    for(int j=512;j<528;++j) w[o+j]=(uint8_t)(119+rng()%13);
   }
  } else {
   for(size_t i=0;i<bytes;i+=2) {const auto x=__float2bfloat16_rn(nd(rng));std::memcpy(w.data()+i,&x,2);}
  }
  return upload(w);
 };
 uint8_t *f[2]={weights(128,false),weights(128,true)},*g[2]={weights(128,false),weights(128,true)},*b=weights(64,false);
 unsigned tests=0;
 for(int M: {1,7,8,9,17,31,32,33,63,64,65,127,300,2048,8192,32768}) {
  const size_t np=(size_t)((M+7)/8)*8*K/2;
  std::vector<float> hx((size_t)M*K);for(auto& x:hx)x=nd(rng);
  float* x=upload(hx),*y=alloc<float>((size_t)M*LD);
  unsigned *xf=alloc<unsigned>(np),*xb=alloc<unsigned>(np),*df=alloc<unsigned>(g6mx::MX*K/2),*db=alloc<unsigned>(g6mx::MX*K/2);
  auto args=[&](int rows,int offset,int cfg,const unsigned* af,const unsigned* ab) {
   ProjArgs a;a.xfrag=ab;a.M=rows;a.ld=LD;a.odt=gv::O_F32;
   a.add(f[cfg&1],y+(size_t)offset*LD,K,128,1,32,(cfg&1)?af:ab,(cfg&1)?1:2);
   a.add(g[cfg>>1],y+(size_t)offset*LD+128,K,128,1,32,(cfg>>1)?af:ab,(cfg>>1)?1:2);
   a.add(b,y+(size_t)offset*LD+256,K,64,1,32);return a;
  };
  for(int cfg=0;cfg<4;++cfg) {
   auto old=[&]{for(int off=0;off<M;off+=g6mx::MX){int rows=std::min(g6mx::MX,M-off);flash_mhc::fragments(x+(size_t)off*K,rows,K,df,db,0);launch_proj(args(rows,off,cfg,df,db),0);}};
   auto cur=[&](int tile){flash_mhc::fragments(x,M,K,xf,xb,0);launch_tail_gemm(args(M,0,cfg,xf,xb),tile,0);};
   CK(cudaMemset(y,0x3f,(size_t)M*LD*4));old();CK(cudaGetLastError());CK(cudaDeviceSynchronize());
   std::vector<float> ref((size_t)M*LD),got(ref.size());CK(cudaMemcpy(ref.data(),y,ref.size()*4,cudaMemcpyDeviceToHost));
   for(int tile:{32,64}) {
    CK(cudaMemset(y,0x3f,(size_t)M*LD*4));cur(tile);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(got.data(),y,got.size()*4,cudaMemcpyDeviceToHost));
    for(int p=0;p<3;++p){int lo=p*128,n=p==2?64:128;size_t diff=0;double mx=0,ss=0,rr=0;
     for(int m=0;m<M;++m)for(int j=lo;j<lo+n;++j){size_t i=(size_t)m*LD+j;float a=ref[i],v=got[i];diff+=memcmp(&a,&v,4)!=0;double d=(double)a-v;mx=std::max(mx,fabs(d));ss+=d*d;rr+=(double)a*a;}
     if(diff){fprintf(stderr,"FAIL M%d cfg%d tile%d mat%d differing%zu max%.9g rel%.9g\n",M,cfg,tile,p,diff,mx,sqrt(ss/(rr?rr:1)));return 3;}}
    if(memcmp(got.data(),ref.data(),got.size()*4)){fprintf(stderr,"padding modified\n");return 4;}
    ++tests;
   }
   if(M>=2048 && cfg==3) {
    auto time=[&](auto fn){cudaEvent_t a,z;CK(cudaEventCreate(&a));CK(cudaEventCreate(&z));fn();CK(cudaEventRecord(a));for(int rep=0;rep<5;++rep)fn();CK(cudaEventRecord(z));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,a,z));CK(cudaEventDestroy(a));CK(cudaEventDestroy(z));return ms/5;};
    printf("TIMING M%d MX%d old_ms%.6f tile32_ms%.6f tile64_ms%.6f\n",M,g6mx::MX,time(old),time([&]{cur(32);}),time([&]{cur(64);}));
   }
  }
  printf("PASS M%d all4 mixed/BF16/MXFP8 configs both tiles f_a/g_a/b and padding exact\n",M);fflush(stdout);
  CK(cudaFree(x));CK(cudaFree(y));CK(cudaFree(xf));CK(cudaFree(xb));CK(cudaFree(df));CK(cudaFree(db));
 }
 printf("PASS %u configurations; every matrix bit-identical (max/RMS error 0)\n",tests);
 for(auto p:f)CK(cudaFree(p));for(auto p:g)CK(cudaFree(p));CK(cudaFree(b));
}
