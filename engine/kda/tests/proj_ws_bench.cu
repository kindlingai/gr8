#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../proj_in_prefill.cuh"
#define CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
__device__ unsigned mix(unsigned x) {x^=x>>16;x*=0x85ebca6bu;x^=x>>13;x*=0xc2b2ae35u;return x^(x>>16);}
__global__ void fill_a(uint8_t* p,int M,int K) {
 int lane=threadIdx.x&31,warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
 for(int z=warp;z<M*(K/128);z+=gridDim.x*blockDim.x/32) {
  int row=z/(K/128),b=z%(K/128);float v[4];
  for(int j=0;j<4;++j)v[j]=float(int(mix(z*128+lane*4+j)&65535)-32768)/65536.f;
  bindp::shx_store_h16(v,lane,row,M,K/16,b*8,p);
 }
}
__global__ void fill_w(unsigned* p,size_t n) {
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=mix(unsigned(i)^23457u);
}
__global__ void compare(const float* a,const float* b,size_t n,unsigned long long* out) {
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
  if(__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(out,1ull);
  if(!isfinite(a[i])||!isfinite(b[i]))atomicAdd(out+1,1ull);
 }
}
template<class F>float timed(F f,int reps) {
 cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
 CK(cudaEventRecord(a));for(int i=0;i<reps;++i)f();CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
 float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms/reps;
}
int main(int argc,char**argv) {
 using namespace bindp;
 int M=argc>1?atoi(argv[1]):8192,N=argc>2?atoi(argv[2]):4096,K=argc>3?atoi(argv[3]):4096,reps=argc>4?atoi(argv[4]):3;
 if(M<1||N<128||N%128||K<128||K%128||reps<1)return 3;
 size_t na=size_t((M+15)/16)*(K/16)*512,nw=size_t(N/128)*(K/16)*8*128,ny=size_t(M)*(N+16)+64;
 auto a=alloc<uint8_t>(na),w=alloc<uint8_t>(nw);
 auto yo=alloc<float>(ny),yn=alloc<float>(ny);auto sv=alloc<__half>(N);auto diff=alloc<unsigned long long>(2);
 std::vector<__half> hsv(N);for(int i=0;i<N;++i)hsv[i]=__float2half((i%2?-1.f:1.f)*(0.5f+(i%13)/32.f));
 CK(cudaMemcpy(sv,hsv.data(),N*2,cudaMemcpyHostToDevice));
 CK(cudaMemset(a,0,na));fill_a<<<512,256>>>(a,M,K);fill_w<<<512,256>>>(reinterpret_cast<unsigned*>(w),nw/4);
 Gemm g;g.M=M;g.N=N;g.K=K;g.x8=a;g.tw=w;g.tsG=8192;g.tsB=(K/128)*8192;g.ldy=N+16;g.bf16=2;g.h16=1;
 CK(cudaFuncSetAttribute(kda_proj_ws::kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kda_proj_ws::SMEM));
 CK(cudaFuncSetAttribute(k_gemm_tr_h16<16,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,GTH16<16>::SMEM));
 cudaFuncAttributes ao,an;CK(cudaFuncGetAttributes(&ao,k_gemm_tr_h16<16,false,true>));CK(cudaFuncGetAttributes(&an,kda_proj_ws::kernel));
 printf("RESOURCE oldregs%d newregs%d oldlocal%zu newlocal%zu oldsmem%d newsmem%d\n",ao.numRegs,an.numRegs,ao.localSizeBytes,an.localSizeBytes,GTH16<16>::SMEM,kda_proj_ws::SMEM);
 auto run=[&](int mode) {
  Gemm v=g;v.y=mode?yn:yo;
  dim3 grid(N/128,(M+255)/256);
  if(mode)kda_proj_ws::kernel<<<grid,kda_proj_ws::THREADS,kda_proj_ws::SMEM>>>(v);
  else k_gemm_tr_h16<16,false,true><<<grid,512,GTH16<16>::SMEM>>>(v);
  CK(cudaGetLastError());
 };
 for(int rot=0;rot<2;++rot) {
  g.osvh=rot?sv:nullptr;
  CK(cudaMemset(yo,0x5a,ny*4));CK(cudaMemset(yn,0x5a,ny*4));CK(cudaMemset(diff,0,16));
  run(0);run(1);CK(cudaDeviceSynchronize());
  compare<<<512,256>>>(yo,yn,ny,diff);unsigned long long h[2];CK(cudaMemcpy(h,diff,16,cudaMemcpyDeviceToHost));
  printf("EXACT M%d N%d K%d rot%d different%llu nonfinite%llu padding_included\n",M,N,K,rot,h[0],h[1]);fflush(stdout);
  if(h[0]||h[1])return 1;
  // Five alternating pairs, with warmed kernels. No profile or compare in the timed interval.
  for(int i=0;i<5;++i) {
   float old,newt;
   if(i&1){newt=timed([&]{run(1);},reps);old=timed([&]{run(0);},reps);}
   else{old=timed([&]{run(0);},reps);newt=timed([&]{run(1);},reps);}
   printf("TIME M%d N%d K%d rot%d pair%d old%.6f new%.6f ms\n",M,N,K,rot,i,old,newt);fflush(stdout);
  }
 }
 puts("PROJ_WS_PASS");
}
