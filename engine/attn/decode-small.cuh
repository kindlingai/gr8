// Exact small-decode kernels; included inside ga after Work is defined.
// No cache, arithmetic boundary, selection-order or scratch-layout changes.
#define G6_SMALL_DECODE_KERNELS 1
namespace small_decode {
using ga::half;using ga::bf16;
__device__ __forceinline__ float2 fp8_pair(uint16_t u){__half2_raw raw=__nv_cvt_fp8x2_to_halfraw2(u,__NV_E4M3);return __half22float2(__half2(raw));}
// No CTA-level query staging: cached read-only Q loads and fully unrolled
// lane-strided accumulation retain the old 16 latent +2 RoPE FMA order.
template<class Input>
__global__ void scores_direct(Input input,int start,const float* q,const float* rope,float* out){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h=blockIdx.y,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;float sum=0;
 if(v){float scale[4];
 #pragma unroll
 for(int g=0;g<4;g++)scale[g]=((const float*)(v+512))[g];
 #pragma unroll
 for(int k=0;k<16;k++){int d=lane+k*32;sum=fmaf(__ldg(q+((int64_t)row*a.heads+h)*512+d),ga::e4(v[d])*scale[k/4],sum);}
 #pragma unroll
 for(int k=0;k<2;k++){int d=lane+k*32;sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+d),__bfloat162float(((const bf16*)(v+528))[d]),sum);}}
 sum=ga::warp_sum(sum);if(lane==0)out[((int64_t)row*a.heads+h)*ga::TOP+j]=v?sum*(1.f/16):-INFINITY;
}
template<class Input>
__global__ void context_parts(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128;
 int row=blockIdx.z,r=start+row,h=blockIdx.y,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 __shared__ uintptr_t rows[256];__shared__ float probability[256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];float p=probs[((int64_t)row*a.heads+h)*ga::TOP+slot];const uint8_t* src=nullptr;if(id>=0&&id<=a.positions[r]&&p!=0)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;probability[j]=p;active|=src!=nullptr;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];}
 bool any=__syncthreads_or(active);float lo=0,hi=0;int d=d0+2*t;
 if(any){
 #pragma unroll 4
 for(int j=0;j<256;j++){const uint8_t* src=(const uint8_t*)rows[j];if(!src)continue;float2 x=fp8_pair(*(const uint16_t*)(src+d));float s=scale[j*GROUPS+(2*t)/128],p=probability[j];lo=fmaf(p,x.x*s,lo);hi=fmaf(p,x.y*s,hi);}}
 float* dst=parts+(((int64_t)row*a.heads+h)*ga::PART+part)*512;dst[d]=lo;dst[d+1]=hi;
}
// DEC_CPB (bar RESULTS-M20 "DEC_CPB"): context_parts with the fp8 pairs of the next CPB slots loaded before their FMAs, so
// the gathered loads overlap (ncu: context_parts is long_scoreboard-bound). Same FMAs in the same slot order, same skip
// of dead / p == 0 slots, so every output byte matches context_parts.
#ifndef DEC_CPB2   // DEC_CPB2 (bar RESULTS-M20 "DEC_CPB2"): batch 8 -> 64 slots; default 8 = DEC_CPB as before
#define DEC_CPB2 8
#endif
constexpr int CPB=DEC_CPB2;
template<class Input>
__global__ void context_parts_cpb(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128;
 int row=blockIdx.z,r=start+row,h=blockIdx.y,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 __shared__ uintptr_t rows[256];__shared__ float probability[256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];float p=probs[((int64_t)row*a.heads+h)*ga::TOP+slot];const uint8_t* src=nullptr;if(id>=0&&id<=a.positions[r]&&p!=0)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;probability[j]=p;active|=src!=nullptr;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];}
 bool any=__syncthreads_or(active);float lo=0,hi=0;int d=d0+2*t;
 if(any){
 for(int j0=0;j0<256;j0+=CPB){uint16_t raw[CPB];
  #pragma unroll
  for(int u=0;u<CPB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}
  #pragma unroll
  for(int u=0;u<CPB;u++){const int j=j0+u;if(!rows[j])continue;float2 x=fp8_pair(raw[u]);float s=scale[j*GROUPS+(2*t)/128],p=probability[j];lo=fmaf(p,x.x*s,lo);hi=fmaf(p,x.y*s,hi);}}}
 float* dst=parts+(((int64_t)row*a.heads+h)*ga::PART+part)*512;dst[d]=lo;dst[d+1]=hi;
}
// Head-batched forms of scores_direct and context_parts_cpb (gr8 #75): one key's bytes are loaded once for HG heads
// instead of once per head. Every (row, head, slot) output keeps the per-head kernels' arithmetic: the same products, the
// same fmaf order, the same warp_sum, the same skip of dead slots and of p == 0, so the outputs are bit-identical.
// scores_heads: warp per (slot, row), heads [y*HG, y*HG + HG); lane l stores head y*HG + l.
template<int HG,class Input>
__global__ void __launch_bounds__(256) scores_heads(Input input,int start,const float* q,const float* rope,float* out){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int WARPS=8;int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,j=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x&31;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 int id=a.indices[(int64_t)r*ga::TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?ga::kv_at(a,id):nullptr;
 float mine=-INFINITY;
 if(v){float scale[4],kv[16],rv[2];
  #pragma unroll
  for(int g=0;g<4;g++)scale[g]=((const float*)(v+512))[g];
  #pragma unroll
  for(int k=0;k<16;k++)kv[k]=ga::e4(v[lane+k*32])*scale[k/4];
  #pragma unroll
  for(int k=0;k<2;k++)rv[k]=__bfloat162float(((const bf16*)(v+528))[lane+k*32]);
  for(int hh=0;hh<nh;hh++){const int h=h0+hh;float sum=0;
   #pragma unroll
   for(int k=0;k<16;k++)sum=fmaf(__ldg(q+((int64_t)row*a.heads+h)*512+lane+k*32),kv[k],sum);
   #pragma unroll
   for(int k=0;k<2;k++)sum=fmaf(__ldg(rope+((int64_t)row*a.heads+h)*64+lane+k*32),rv[k],sum);
   sum=ga::warp_sum(sum);if(lane==hh)mine=sum*(1.f/16);}}
 if(lane<nh)out[((int64_t)row*a.heads+h0+lane)*ga::TOP+j]=mine;
}
// context_parts_heads: block per (part, half of the 512 dims, head group, row); thread t owns dims d, d+1 for HG heads
template<int HG,class Input>
__global__ void __launch_bounds__(128) context_parts_heads(Input input,int start,const float* probs,float* parts){
 auto a=ga::resolve(input);if(!a.active)return;constexpr int D=256,THREADS=D/2,GROUPS=(D+127)/128,PB=16;
 int row=blockIdx.z,r=start+row,h0=blockIdx.y*HG,part=blockIdx.x/(512/D),d0=(blockIdx.x%(512/D))*D,t=threadIdx.x;FOR_ROW(a,input,r);
 const int nh=min(HG,a.heads-h0);
 __shared__ uintptr_t rows[256];__shared__ float probability[HG][256],scale[256*GROUPS];bool active=false;
 for(int j=t;j<256;j+=THREADS){int slot=part*256+j,id=a.indices[(int64_t)r*ga::TOP+slot];const uint8_t* src=nullptr;
  const bool idok=id>=0&&id<=a.positions[r];   // dead slots read no probabilities
  bool live=false;for(int hh=0;hh<HG;hh++){float p=idok&&hh<nh?probs[((int64_t)row*a.heads+h0+hh)*ga::TOP+slot]:0.f;probability[hh][j]=p;live|=p!=0;}
  if(live)src=ga::kv_at(a,id);
  rows[j]=(uintptr_t)src;if(src)for(int g=0;g<GROUPS;g++)scale[j*GROUPS+g]=((const float*)(src+512))[d0/128+g];active|=src!=nullptr;}
 const bool any=__syncthreads_or(active);float lo[HG],hi[HG];
 #pragma unroll
 for(int hh=0;hh<HG;hh++){lo[hh]=0;hi[hh]=0;}
 int d=d0+2*t;
 if(any)for(int j0=0;j0<256;j0+=PB){uint16_t raw[PB];
  #pragma unroll
  for(int u=0;u<PB;u++){const uint8_t* src=(const uint8_t*)rows[j0+u];raw[u]=src?*(const uint16_t*)(src+d):(uint16_t)0;}
  #pragma unroll
  for(int u=0;u<PB;u++){const int j=j0+u;if(!rows[j])continue;float2 x=fp8_pair(raw[u]);float s=scale[j*GROUPS+(2*t)/128];
   #pragma unroll
   for(int hh=0;hh<HG;hh++){float p=probability[hh][j];if(p==0)continue;lo[hh]=fmaf(p,x.x*s,lo[hh]);hi[hh]=fmaf(p,x.y*s,hi[hh]);}}}
 #pragma unroll
 for(int hh=0;hh<HG;hh++)if(hh<nh){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*ga::PART+part)*512;dst[d]=lo[hh];dst[d+1]=hi[hh];}
}
template<class Input>
__global__ void index_scores(Input input,int start,const half* q,const float* weights,float* scores,int* ids){
 auto a=ga::resolve(input);if(!a.active)return;using namespace nvcuda;int row=blockIdx.y,r=start+row,t0=blockIdx.x*32,t=threadIdx.x,warp=t/32;FOR_ROW(a,input,r);
 __shared__ __align__(32) half key[32*128];__shared__ __align__(32) float products[32*32];__shared__ float scales[32];__shared__ uintptr_t pointers[32];
 if(t0>a.positions[r]){if(t<32&&t0+t<a.context){scores[(int64_t)row*a.context+t0+t]=-INFINITY;ids[(int64_t)row*a.context+t0+t]=t0+t;}return;}
 if(t<32){float scale=0;const uint8_t* p=t0+t<a.live_context&&CP_MINE(a,t0+t)?ga::ik_at(a,t0+t,scale):nullptr;pointers[t]=(uintptr_t)p;scales[t]=scale;}__syncthreads();
 for(int i=t;i<32*128/2;i+=128){int col=i/64,d=(i%64)*2;const uint8_t* p=(const uint8_t*)pointers[col];uint16_t code=p?*(const uint16_t*)(p+d):0;((__half2*)key)[i]=__half2(__nv_cvt_fp8x2_to_halfraw2(code,__NV_E4M3));}
 __syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>B;wmma::fragment<wmma::accumulator,16,16,16,float>C;wmma::fill_fragment(C,0.f);int hs=(warp/2)*16,ts=(warp%2)*16;
 for(int k=0;k<128;k+=16){wmma::load_matrix_sync(A,q+((int64_t)row*32+hs)*128+k,128);wmma::load_matrix_sync(B,key+ts*128+k,128);wmma::mma_sync(C,A,B,C);}
 wmma::store_matrix_sync(products+hs*32+ts,C,32,wmma::mem_row_major);__syncthreads();
 if(t<32&&t0+t<a.context){float score=0;for(int h=0;h<32;h++)score+=fmaxf(products[h*32+t],0.f)*weights[row*32+h];score*=scales[t];if(t0+t>a.positions[r]||!CP_MINE(a,t0+t))score=-INFINITY;else if(!isfinite(score)){ga::error(a,4);score=-INFINITY;}scores[(int64_t)row*a.context+t0+t]=score;ids[(int64_t)row*a.context+t0+t]=t0+t;}
}
}
static bool use_small_decode(const G6Attn& a){return a.rows<=3&&a.context<=2048;}
// Sparse MLA consumes exactly TOP selected logical IDs, not a.context keys.
// Keep fresh-index dispatch unchanged; widen only decode/verify reuse rows.
static bool use_fixed_k_decode(const G6Attn& a){
 #ifdef DEC_S2
 if(a.s2_batched&&a.rows<=G6_MX)return true;   // batched S2 step: rows of several sessions, up to the row budget
 #endif
 return use_small_decode(a)||(a.reuse_indices&&a.rows<=8);
}
template<class Input>
static void launch_index_scores(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 if(use_small_decode(a))small_decode::index_scores<<<dim3((a.context+31)/32,n),128,0,st>>>(input,start,w.iq,w.weights,w.si,w.ii);
 else index_scores<<<dim3((a.context+31)/32,n),128,0,st>>>(input,start,w.iq,w.weights,w.si,w.ii);
}
// G6_ATTN_HEADS=0 (read at launch / capture): the per-head scores_direct / context_parts kernels (A/B and tests)
static bool attn_heads(){static const bool on=!getenv("G6_ATTN_HEADS")||atoi(getenv("G6_ATTN_HEADS"))!=0;return on;}
template<class Input>
static void launch_mla_scores(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 if(use_fixed_k_decode(a)&&attn_heads()){constexpr int HG=16;small_decode::scores_heads<HG><<<dim3(TOP/8,(a.heads+HG-1)/HG,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);}
 else if(use_fixed_k_decode(a))small_decode::scores_direct<<<dim3(TOP/8,a.heads,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
 else mla_scores<<<dim3(TOP/8,a.heads,n),256,0,st>>>(input,start,w.q,w.rope,w.prob);
}
template<class Input>
static void launch_context_parts(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 if(use_fixed_k_decode(a)&&attn_heads()){
  // head group by rows (dec_heads_test --bench on GB10, contexts 500 and 2048): 4 heads per block up to 4 rows, then 8
  if(n>=5){constexpr int HG=8;small_decode::context_parts_heads<HG><<<dim3(PART*2,(a.heads+HG-1)/HG,n),128,0,st>>>(input,start,w.prob,w.parts);}
  else{constexpr int HG=4;small_decode::context_parts_heads<HG><<<dim3(PART*2,(a.heads+HG-1)/HG,n),128,0,st>>>(input,start,w.prob,w.parts);}
  return;}
 #if defined(DEC_CPB) && DEC_CPB
 if(use_fixed_k_decode(a))small_decode::context_parts_cpb<<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);
 #else
 if(use_fixed_k_decode(a))small_decode::context_parts<<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);
 #endif
 else context_parts<<<dim3(PART,a.heads,n),256,0,st>>>(input,start,w.prob,w.parts);
}
template<class Input>   // tests: the DEC_CPB fixed-K launch regardless of the build flag
static void launch_context_parts_cpb(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
 small_decode::context_parts_cpb<<<dim3(PART*2,a.heads,n),128,0,st>>>(input,start,w.prob,w.parts);
}
