// Independent tensor-core prefill implementation; included after attn.cu.
#include "prefill.h"
#include <cublas_v2.h>
#include <cstdio>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include "prefill-index-team-profile.cuh"
#include "selx.cuh"
#ifndef G6_PF_MARK
#define G6_PF_MARK(stage, stream) ((void)0)
#endif
struct G6AttnPrefillScratch {
    void* data=nullptr;size_t bytes=0;int H=0,cap=0,maxrows=0,R=0,dense_limit=0,device=0;bool full_index=false;
    cublasHandle_t blas=nullptr;std::atomic<unsigned> refs{1};
};
struct G6AttnPrefill {
    gp::PrefillProfile profile;
    G6AttnBinding* owner=nullptr;int H=0,cap=0,maxrows=0,R=0,C=0,dense_limit=8192;bool full_index=false;
    // storage owns ONLY this layer's persistent converted UK/UV weights.
    // Scratch is separately allocated and may be retained by other states.
    void* storage=nullptr;size_t bytes=0,weight_bytes=0;void* indexwork=nullptr;
    G6AttnPrefillScratch* workspace=nullptr;
    ga::half *uk=nullptr,*uv=nullptr,*cache=nullptr,*ikey=nullptr,*q=nullptr,*ctxh=nullptr,*densep=nullptr;
    int* next=nullptr;float *kscale=nullptr,*qf=nullptr,*prob=nullptr,*ctx=nullptr,*dense=nullptr;
    cublasHandle_t blas=nullptr;
};
namespace gp {
using ga::half;using ga::bf16;constexpr int TOP=ga::TOP;
// At modest context, shared dense tensor GEMMs win despite extra masked FLOPs.
// Selection and softmax still use precisely the supplied TOP list.
constexpr int DEFAULT_DENSE_LIMIT=8192;
// Bound all prefill scratch and GEMM widths at creation, independent of tails.
// Larger batches reduce launch overhead and improve selector/WMMA occupancy.
constexpr int DEFAULT_QUERY_TILE=256;
static int blas_status(cublasStatus_t s){if(s==CUBLAS_STATUS_SUCCESS)return 0;fprintf(stderr,"g6 prefill cuBLAS status %d\n",int(s));return int(cudaErrorUnknown);}
__global__ void weight_half(const void* src,half* dst,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=__float2half_rn(ga::bfloat(src,i));}
__global__ void expand(G6Attn fixed,half* cache,half* ikey,float* scales,int count){
    auto a=ga::resolve(fixed);int pos=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;if(pos>=count)return;
    const uint8_t* v=pos<a.context?ga::kv_at(a,pos):nullptr;
    for(int d=lane;d<576;d+=32){float x=v?(d<512?ga::latent(v,d):__bfloat162float(((const bf16*)(v+528))[d-512])):0;
        half h=__float2half_rn(x);if(!isfinite(__half2float(h)))ga::error(a,4);cache[(int64_t)pos*576+d]=h;}
    if(!a.reuse_indices){float s=0;const uint8_t* k=pos<a.context?ga::ik_at(a,pos,s):nullptr;
        for(int d=lane;d<128;d+=32)ikey[(int64_t)pos*128+d]=__float2half_rn(k?ga::e4(k[d]):0);if(lane==0)scales[pos]=s;}
}
__global__ void index_scores(G6Attn a,int start,const half* q,const float* weight,const half* keys,const float* scales,float* scores,int* ids){
    using namespace nvcuda;int row=blockIdx.y,r=start+row,t0=blockIdx.x*32,t=threadIdx.x,warp=t/32;
    __shared__ __align__(32) float prod[32*32];
    if(t0>a.positions[r]){if(t<32&&t0+t<a.context){scores[(int64_t)row*a.context+t0+t]=-INFINITY;ids[(int64_t)row*a.context+t0+t]=t0+t;}return;}
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>B;
    wmma::fragment<wmma::accumulator,16,16,16,float>C;wmma::fill_fragment(C,0.f);
    int hs=warp/2*16,ts=warp%2*16;
    for(int k=0;k<128;k+=16){wmma::load_matrix_sync(A,q+((int64_t)row*32+hs)*128+k,128);wmma::load_matrix_sync(B,keys+(int64_t)(t0+ts)*128+k,128);wmma::mma_sync(C,A,B,C);}
    wmma::store_matrix_sync(prod+hs*32+ts,C,32,wmma::mem_row_major);__syncthreads();
    if(t<32&&t0+t<a.context){int pos=t0+t;float x=0;for(int h=0;h<32;h++)x+=fmaxf(prod[h*32+t],0.f)*weight[row*32+h];x*=scales[pos];
        if(pos>a.positions[r])x=-INFINITY;else if(!isfinite(x)){ga::error(a,4);x=-INFINITY;}scores[(int64_t)row*a.context+pos]=x;ids[(int64_t)row*a.context+pos]=pos;}
}
// Four queries share a 64-key tile. FP8 code products retain the original
// K=16 WMMA accumulation and 32-head weighted-ReLU order. Vector loads
// share one arena between cache codes and products, separated by a CTA barrier.
__global__ void index_scores_grouped(G6Attn a,int start,int n,const half* q,const float* weight,const half* keys,const float* scales,float* scores,int* ids){
 constexpr int Q=4,N=64;
 using namespace nvcuda;constexpr int THREADS=Q*64;constexpr int BYTES=(Q*32*N*4>N*128*2)?Q*32*N*4:N*128*2;
 int row0=blockIdx.y*Q,t0=blockIdx.x*N,t=threadIdx.x,w=t/32;
 __shared__ __align__(32) unsigned char arena[BYTES];half* B=(half*)arena;float* O=(float*)arena;
 for(int i=t;i<N*128/8;i+=THREADS){int key=t0+i/(128/8);((int4*)B)[i]=key<a.context?*(const int4*)(keys+(int64_t)key*128+(i%(128/8))*8):make_int4(0,0,0,0);}
 __syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>Bfrag;
 wmma::fragment<wmma::accumulator,16,16,16,float>C[N/16];
 #pragma unroll
 for(int j=0;j<N/16;j++)wmma::fill_fragment(C[j],0.f);
 int qr=row0+w/2,hs=(w%2)*16;
 for(int k=0;k<128;k+=16){if(qr<n)wmma::load_matrix_sync(A,q+((int64_t)qr*32+hs)*128+k,128);else wmma::fill_fragment(A,__float2half(0));
  #pragma unroll
  for(int j=0;j<N/16;j++){wmma::load_matrix_sync(Bfrag,B+j*16*128+k,128);wmma::mma_sync(C[j],A,Bfrag,C[j]);}}
 __syncthreads(); // B storage becomes O only after all warp loads have finished.
 #pragma unroll
 for(int j=0;j<N/16;j++)wmma::store_matrix_sync(O+w*16*N+j*16,C[j],N,wmma::mem_row_major);
 __syncthreads();
 for(int i=t;i<Q*N;i+=THREADS){int row=row0+i/N,key=t0+i%N;if(row<n&&key<a.context){float x=0;for(int h=0;h<32;h++)x+=fmaxf(O[((i/N)*32+h)*N+i%N],0.f)*weight[row*32+h];x*=scales[key];if(key>a.positions[start+row])x=-INFINITY;else if(!isfinite(x)){ga::error(a,4);x=-INFINITY;}scores[(int64_t)row*a.context+key]=x;ids[(int64_t)row*a.context+key]=key;}}
}
#if defined(PF_ISGRT) && PF_ISGRT
// PF_ISGRT: index_scores_grouped re-tiled. A CTA keeps its 64-key tile (row stride BL) and walks RG rows; fragments,
// mma k order and the 32-head epilogue order are unchanged per score. A row group whose positions all precede the tile
// skips the MMA: every score is -INF and no finiteness check applies.
// Eight rows per pass: warp w holds rows row0+w/2 and row0+4+w/2 (heads hs..hs+15) so each B fragment load feeds two
// MMAs; every accumulator still sees k ascending. O is filled and reduced one four-row half at a time. BL/OL are the
// shared row strides of B (halves) and O (floats); padding past 128/64 moves rows off a common bank and changes no value.
template<int BL,int OL>
__global__ void __launch_bounds__(256,2) index_scores_rg8(G6Attn a,int start,int n,int RG,const half* q,const float* weight,const half* keys,const float* scales,float* scores,int* ids){
 constexpr int Q=4,N=64;
 using namespace nvcuda;constexpr int THREADS=Q*64;
 int rbeg=blockIdx.y*RG,rend=min(n,rbeg+RG),t0=blockIdx.x*N,t=threadIdx.x,w=t/32;
 extern __shared__ __align__(32) unsigned char isg_smem[];half* B=(half*)isg_smem;float* O=(float*)(isg_smem+N*BL*2);
 for(int i=t;i<N*128/8;i+=THREADS){int key=t0+i/(128/8);*(int4*)(B+(i/(128/8))*BL+(i%(128/8))*8)=key<a.context?*(const int4*)(keys+(int64_t)key*128+(i%(128/8))*8):make_int4(0,0,0,0);}
 __syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A0,A1;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>Bfrag;
 wmma::fragment<wmma::accumulator,16,16,16,float>C0[N/16],C1[N/16];
 for(int row0=rbeg;row0<rend;row0+=2*Q){
  int pmax=a.positions[start+row0];for(int r=1;r<2*Q;r++)if(row0+r<n)pmax=max(pmax,a.positions[start+row0+r]);
  if(t0>pmax){for(int i=t;i<2*Q*N;i+=THREADS){int row=row0+i/N,key=t0+i%N;if(row<n&&key<a.context){scores[(int64_t)row*a.context+key]=-INFINITY;ids[(int64_t)row*a.context+key]=key;}}continue;}
  #pragma unroll
  for(int j=0;j<N/16;j++){wmma::fill_fragment(C0[j],0.f);wmma::fill_fragment(C1[j],0.f);}
  int qr=row0+w/2,qr1=qr+Q,hs=(w%2)*16;
  #pragma unroll 1
  for(int k=0;k<128;k+=16){if(qr<n)wmma::load_matrix_sync(A0,q+((int64_t)qr*32+hs)*128+k,128);else wmma::fill_fragment(A0,__float2half(0));
   if(qr1<n)wmma::load_matrix_sync(A1,q+((int64_t)qr1*32+hs)*128+k,128);else wmma::fill_fragment(A1,__float2half(0));
   #pragma unroll
   for(int j=0;j<N/16;j++){wmma::load_matrix_sync(Bfrag,B+j*16*BL+k,BL);wmma::mma_sync(C0[j],A0,Bfrag,C0[j]);wmma::mma_sync(C1[j],A1,Bfrag,C1[j]);}}
  #pragma unroll
  for(int j=0;j<N/16;j++)wmma::store_matrix_sync(O+w*16*OL+j*16,C0[j],OL,wmma::mem_row_major);
  __syncthreads();
  for(int i=t;i<Q*N;i+=THREADS){int row=row0+i/N,key=t0+i%N;if(row<n&&key<a.context){float x=0;for(int h=0;h<32;h++)x+=fmaxf(O[((i/N)*32+h)*OL+i%N],0.f)*weight[row*32+h];x*=scales[key];if(key>a.positions[start+row])x=-INFINITY;else if(!isfinite(x)){ga::error(a,4);x=-INFINITY;}scores[(int64_t)row*a.context+key]=x;ids[(int64_t)row*a.context+key]=key;}}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<N/16;j++)wmma::store_matrix_sync(O+w*16*OL+j*16,C1[j],OL,wmma::mem_row_major);
  __syncthreads();
  for(int i=t;i<Q*N;i+=THREADS){int row=row0+Q+i/N,key=t0+i%N;if(row<n&&key<a.context){float x=0;for(int h=0;h<32;h++)x+=fmaxf(O[((i/N)*32+h)*OL+i%N],0.f)*weight[row*32+h];x*=scales[key];if(key>a.positions[start+row])x=-INFINITY;else if(!isfinite(x)){ga::error(a,4);x=-INFINITY;}scores[(int64_t)row*a.context+key]=x;ids[(int64_t)row*a.context+key]=key;}}
  __syncthreads();
 }
}
template<int BL,int OL> constexpr int isg_rg8_smem(){return 64*BL*2+4*32*OL*4;}
// Rows per CTA: G6_PF_ISGRT_RG (multiple of 8) overrides the default 8.
static inline int isgrt_rg(){static const int env=[]{const char* s=getenv("G6_PF_ISGRT_RG");return s?atoi(s)&~7:0;}();return env>=8?env:8;}
static inline void launch_index_scores_rg(G6Attn a,int start,int n,const half* q,const float* weight,const half* keys,const float* scales,float* scores,int* ids,cudaStream_t st){
 constexpr int SMEM=isg_rg8_smem<136,64>();static const bool once=cudaFuncSetAttribute(index_scores_rg8<136,64>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM)==cudaSuccess;(void)once;
 int RG=isgrt_rg();index_scores_rg8<136,64><<<dim3((a.context+63)/64,(n+RG-1)/RG),256,SMEM,st>>>(a,start,n,RG,q,weight,keys,scales,scores,ids);
}
#endif
// Fixed-size cuBLAS shapes even when the final query tile is short.
// Reuse q scratch for raw UK input, then absorbed Q, then UV output.
__global__ void pack_uk_input(G6Attn a,int start,int n,int R,half* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=a.heads*R*192)return;int d=i%192,row=(i/192)%R,h=i/(192*R);out[((int64_t)h*R+row)*576+d]=row<n?((const half*)a.q_b)[((int64_t)(start+row)*a.heads+h)*256+d]:__float2half(0);
}
__global__ void unpack_uv(G6Attn a,int start,int n,int R,const half* src){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*a.heads*256)return;int d=i%256,h=(i/256)%a.heads,row=i/(a.heads*256);((half*)a.output)[((int64_t)start*a.heads)*256+i]=src[((int64_t)h*R+row)*256+d];
}
__global__ void pack_q(G6Attn a,int start,int n,int R,const float* qf,half* q){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R*a.heads*576)return;int d=i%576,h=(i/576)%a.heads,row=i/(576*a.heads);float x;if(row>=n){q[((int64_t)h*R+row)*576+d]=__float2half(0);return;}
    if(d<512)x=qf[((int64_t)h*R+row)*512+d];else{int j=(d-512)/2;float c,s;ga::cs(a.positions[start+row],j,c,s);
        float u=ga::hfloat(a.q_b,((int64_t)(start+row)*a.heads+h)*256+192+2*j),v=ga::hfloat(a.q_b,((int64_t)(start+row)*a.heads+h)*256+193+2*j);x=(d&1)?v*c+u*s:u*c-v*s;}
    half z=__float2half_rn(x);if(!isfinite(__half2float(z)))ga::error(a,4);q[((int64_t)h*R+row)*576+d]=z;
}
// A CTA evaluates all native heads against 64 selected keys. Internal WMMA
// fringe lanes are zero-filled; no padded heads exist in the ABI or output.
// Copy eight contiguous halves per load; all internal row strides are 16-byte
// aligned. Reuse operand storage for output only after every warp is finished.
template<int MT> __global__ void sparse_qk(G6Attn a,int start,int R,const half* q,const half* cache,float* scores){
 // MT head tiles of 16 share one gather of the selected cache rows (TP3: 22/21 heads -> MT=2)
 constexpr int N=64,K=64;
 using namespace nvcuda;constexpr int THREADS=N/16*32;int row=blockIdx.y,r=start+row,k0=blockIdx.x*N,t=threadIdx.x,w=t/32;
 const int hn=min(16*MT,a.heads);
 __shared__ __align__(32) half arena[(16*MT+N)*K];half* A=arena;half* B=A+16*MT*K;float* O=(float*)arena;
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>aa;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>bb;
 wmma::fragment<wmma::accumulator,16,16,16,float>cc[MT];
#pragma unroll
 for(int m=0;m<MT;m++)wmma::fill_fragment(cc[m],0.f);
 for(int base=0;base<576;base+=K){
  for(int i=t;i<16*MT*K/8;i+=THREADS){int h=i/(K/8),d=base+(i%(K/8))*8;((int4*)A)[i]=(h<a.heads&&d<576)?*(const int4*)(q+((int64_t)h*R+row)*576+d):make_int4(0,0,0,0);}
  for(int i=t;i<N*K/8;i+=THREADS){int j=i/(K/8),d=base+(i%(K/8))*8,id=a.indices[(int64_t)r*ga::TOP+k0+j];bool ok=id>=0&&id<=a.positions[r]&&id<a.context&&d<576;((int4*)B)[i]=ok?*(const int4*)(cache+(int64_t)id*576+d):make_int4(0,0,0,0);}
  __syncthreads();for(int k=0;k<K;k+=16){wmma::load_matrix_sync(bb,B+w*16*K+k,K);
#pragma unroll
   for(int m=0;m<MT;m++){wmma::load_matrix_sync(aa,A+m*16*K+k,K);wmma::mma_sync(cc[m],aa,bb,cc[m]);}}__syncthreads();
 }
#pragma unroll
 for(int m=0;m<MT;m++)wmma::store_matrix_sync(O+m*16*N+w*16,cc[m],N,wmma::mem_row_major);__syncthreads();
 for(int i=t;i<hn*N;i+=THREADS){int h=i/N,j=i%N,id=a.indices[(int64_t)r*ga::TOP+k0+j];bool ok=id>=0&&id<=a.positions[r]&&id<a.context;scores[((int64_t)row*a.heads+h)*ga::TOP+k0+j]=ok?O[i]*(1.f/16):-INFINITY;}
}
// K=128 staging amortizes gathers/barriers; MMA accumulation remains K=16
// in selected-list order, with one RN half conversion per probability.
template<int MT> __global__ void sparse_pv(G6Attn a,int start,int R,const float* p,const half* cache,float* context){
 constexpr int N=64,K=128;
 using namespace nvcuda;constexpr int THREADS=N/16*32;int row=blockIdx.y,r=start+row,d0=blockIdx.x*N,t=threadIdx.x,w=t/32;
 const int hn=min(16*MT,a.heads);
 __shared__ __align__(32) half arena[(16*MT+N)*K];half* A=arena;half* B=A+16*MT*K;float* O=(float*)arena;
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>aa;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major>bb;
 wmma::fragment<wmma::accumulator,16,16,16,float>cc[MT];
#pragma unroll
 for(int m=0;m<MT;m++)wmma::fill_fragment(cc[m],0.f);
 for(int base=0;base<ga::TOP;base+=K){
  for(int i=t;i<16*MT*K/2;i+=THREADS){int h=i/(K/2),j=base+(i%(K/2))*2;float2 v=h<a.heads?*(const float2*)(p+((int64_t)row*a.heads+h)*ga::TOP+j):make_float2(0,0);((half2*)A)[i]=__floats2half2_rn(v.x,v.y);}
  for(int i=t;i<K*N/8;i+=THREADS){int j=base+i/(N/8),d=d0+(i%(N/8))*8,id=a.indices[(int64_t)r*ga::TOP+j];bool ok=id>=0&&id<=a.positions[r]&&id<a.context;((int4*)B)[i]=ok?*(const int4*)(cache+(int64_t)id*576+d):make_int4(0,0,0,0);}
  __syncthreads();for(int k=0;k<K;k+=16){wmma::load_matrix_sync(bb,B+k*N+w*16,N);
#pragma unroll
   for(int m=0;m<MT;m++){wmma::load_matrix_sync(aa,A+m*16*K+k,K);wmma::mma_sync(cc[m],aa,bb,cc[m]);}}__syncthreads();
 }
#pragma unroll
 for(int m=0;m<MT;m++)wmma::store_matrix_sync(O+m*16*N+w*16,cc[m],N,wmma::mem_row_major);__syncthreads();for(int i=t;i<hn*N;i+=THREADS){int h=i/N,d=i%N;context[((int64_t)h*R+row)*512+d0+d]=O[i];}
}
__global__ void dense_gather(G6Attn a,int start,int n,int R,const float* dense,float* score){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*a.heads*TOP)return;int j=i%TOP,h=(i/TOP)%a.heads,row=i/(TOP*a.heads),r=start+row,id=a.indices[(int64_t)r*TOP+j];bool ok=id>=0&&id<=a.positions[r]&&id<a.context;
    score[i]=ok?dense[((int64_t)h*R+row)*((a.context+31)&~31)+id]:-INFINITY;
}
__global__ void dense_scatter(G6Attn a,int start,int n,int R,const float* p,float* dense){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*a.heads*TOP)return;int j=i%TOP,h=(i/TOP)%a.heads,row=i/(TOP*a.heads),r=start+row,id=a.indices[(int64_t)r*TOP+j];if(id>=0&&id<=a.positions[r]&&id<a.context&&p[i]!=0)atomicAdd(dense+((int64_t)h*R+row)*a.context+id,p[i]);
}
// Reuse the consumed dense QK buffer as a per-row token->selected-entry map.
// Linked entries retain duplicate-list semantics: merge probabilities in FP32
// before the sole FP16 cast, without an H-fold FP32 scatter/memset/readback.
__global__ void dense_links(G6Attn a,int start,int n,int* head,int* next){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*TOP)return;int row=i/TOP,j=i%TOP,id=a.indices[(int64_t)(start+row)*TOP+j];
    next[i]=-1;if(id>=0&&id<=a.positions[start+row]&&id<a.context)next[i]=atomicExch(head+(int64_t)row*a.context+id,j);
}
__global__ void dense_probabilities(G6Attn a,int n,int R,int D,const float* p,const int* head,const int* next,half* out){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(int64_t)a.heads*R*D)return;int id=i%D,row=(i/D)%R,h=i/(D*R);float sum=0;
    if(row<n&&id<a.context)for(int j=head[(int64_t)row*a.context+id];j>=0;j=next[(int64_t)row*TOP+j])sum+=p[((int64_t)row*a.heads+h)*TOP+j];out[i]=__float2half_rn(sum);
}
__global__ void cast_half(const float* src,half* dst,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=__float2half_rn(src[i]);}
__global__ void trace_hmajor(const float* src,float* dst,int start,int n,int R,int H,int D){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*H*D){int d=i%D,h=(i/D)%H,row=i/(D*H);dst[(int64_t)start*H*D+i]=src[((int64_t)h*R+row)*D+d];}}
static void trace_copy(float* dst,const float* src,int start,int n,int H,int D,cudaStream_t st){if(dst)cudaMemcpyAsync(dst+(int64_t)start*H*D,src,(size_t)n*H*D*4,cudaMemcpyDeviceToDevice,st);}
static void trace_h(float* dst,const float* src,int start,int n,int R,int H,int D,cudaStream_t st){if(dst)trace_hmajor<<<(n*H*D+255)/256,256,0,st>>>(src,dst,start,n,R,H,D);}
static ga::Work index_workspace(const G6Attn& a,void* base,bool full){
    ga::Work w{};if(!full)return w;size_t used=0,R=std::min(a.rows,a.query_tile),T=a.context;
    auto take=[&](size_t n){size_t at=used;used=(used+n+255)&~size_t(255);return (void*)((uintptr_t)base+at);};
    w.iq=(half*)take(R*32*128*2);w.weights=(float*)take(R*32*4);
    w.si=(float*)take(R*T*4);w.ii=(int*)take(R*T*4);
    w.selected=(uint64_t*)take(R*TOP*8);w.selection=(ga::SelectionState*)take(R*sizeof(ga::SelectionState));
    w.histogram=(int*)take(R*((std::min(T,size_t(32768))+ga::RADIX_CHUNK-1)/ga::RADIX_CHUNK)*256*4);
    if(T>32768){w.so=(float*)take(R*T*4);w.io=(int*)take(R*T*4);w.offsets=(int*)take((R+1)*4);
        size_t bytes=0,single=0;auto e=cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,bytes,w.si,w.so,w.ii,w.io,(int)(R*T),(int)R,w.offsets,w.offsets+1);if(e!=cudaSuccess)return ga::Work{};
        e=cub::DeviceRadixSort::SortPairsDescending(nullptr,single,w.si,w.so,w.ii,w.io,(int)T);if(e!=cudaSuccess)return ga::Work{};
        w.temp_bytes=std::max(bytes,single);w.temp=take(w.temp_bytes);}
    w.total=used;return w;
}
// G6_PF_SELX_CTX (selx.cuh = tp6-spare 446fa62, shared with TP4 verbatim): prefill index tiles with n >= SELX_MINN rows and context
// >= this select with selx (one CTA per row, sample bound + filter + radix + sort) instead of SEL32: the same ids in the same
// order (score desc, then id asc, -1 fill). Unset = SEL32 for every tile (today). Strict: an integer 32769..1048576, else exit 2.
// cand scratch = w.so .. w.io (2 * R * T * 4 B, present for T > 32768, used by SEL32 only as its histogram) -> min(CAPG, T) keys/row.
static constexpr int SELX_MINN=32;   // our gate: selx 1.6-2.0x SEL32 from 53K at n >= 33
static int selx_ctx(){
    static const int v=[]{const char* s=getenv("G6_PF_SELX_CTX");if(!s)return 0;
        char* e=nullptr;errno=0;const long x=strtol(s,&e,10);
        if(!*s||*e||errno||x<=32768||x>1048576){fprintf(stderr,"G6_PF_SELX_CTX='%s': need an integer 32769..1048576 (unset = SEL32)\n",s);exit(2);}
        fprintf(stderr,"# G6_PF_SELX_CTX %ld: prefill index tiles with n >= %d and context >= %ld select with selx\n",x,SELX_MINN,x);return (int)x;}();
    return v;
}
static cudaError_t select_prefill(const G6Attn& a,ga::Work& w,int start,int n,cudaStream_t st){
    static_assert(selx::TOP==TOP,"selx TOP");
#ifdef SEL32
    const int c=selx_ctx();
    if(c&&n>=SELX_MINN&&a.context>=c&&a.context>32768&&w.so)
        return selx::select(w.si,a.context,a.context,nullptr,n,a.indices+(int64_t)start*TOP,(uint64_t*)w.so,std::min(selx::CAPG,a.context),nullptr,st);
#endif
    return ga::select_index(a,w,start,n,st);
}
static size_t weight_layout(G6AttnPrefill& p,void* base){
    const size_t uk_bytes=(size_t)p.H*512*192*2,uv_bytes=(size_t)p.H*256*512*2;
    p.uk=(half*)base;p.uv=(half*)((uintptr_t)base+uk_bytes);return uk_bytes+uv_bytes;
}
static size_t scratch_layout(G6AttnPrefill& p,G6Attn a,void* base){size_t used=0;auto take=[&](size_t n){size_t at=used;used=(used+n+255)&~size_t(255);return (void*)((uintptr_t)base+at);};size_t H=p.H,R=p.R,C=p.C;
    p.cache=(half*)take(C*576*2);p.ikey=(half*)take(p.full_index?C*128*2:0);p.kscale=(float*)take(p.full_index?C*4:0);
    p.qf=(float*)take(H*R*512*4);p.q=(half*)take(H*R*576*2);p.prob=(float*)take(H*R*TOP*4);p.ctx=(float*)take(H*R*512*4);p.ctxh=(half*)take(H*R*512*2);p.next=(int*)take(R*TOP*4);
    size_t T=std::min(C,size_t((p.dense_limit+31)&~31));p.dense=(float*)take(H*R*T*4);p.densep=(half*)take(H*R*T*2);a.rows=p.maxrows;a.query_tile=p.R;a.context=p.cap;p.indexwork=take(index_workspace(a,nullptr,p.full_index).total);return used;
}
}
#include "prefill-map-fused.cuh"
namespace gp {
static cudaError_t release_scratch(G6AttnPrefillScratch* w){
    if(!w||w->refs.fetch_sub(1,std::memory_order_acq_rel)!=1)return cudaSuccess;
    auto bs=cublasDestroy(w->blas);auto e=cudaFree(w->data);delete w;return e!=cudaSuccess?e:cudaError_t(blas_status(bs));
}
static bool env_int(const char* name,int fallback,int& out){
    const char* v=std::getenv(name);if(!v||!*v){out=fallback;return true;}
    char* end=nullptr;errno=0;long x=std::strtol(v,&end,10);
    if(errno||end==v||*end||x<0||x>INT_MAX-31)return false;out=int(x);return true;
}
static int create_state(G6AttnPrefill** out,G6AttnBinding* b,void* s,G6AttnPrefill* owner,bool shared){
    if(!out||*out||!b||!b->storage)return int(cudaErrorInvalidValue);
    int tile=0,dense=0,events=0;
    if(!env_int("G6_PF_TILE",DEFAULT_QUERY_TILE,tile)||(tile!=64&&tile!=128&&tile!=256)||!env_int("G6_PF_DENSE",DEFAULT_DENSE_LIMIT,dense)||!env_int("G6_PF_EVENTS",0,events)||events>1)return int(cudaErrorInvalidValue);
    auto p=new G6AttnPrefill;p->dense_limit=dense;p->owner=b;p->full_index=b->full_index;p->H=b->args.heads;p->cap=b->capacity;p->maxrows=b->max_rows;p->R=std::min(tile,p->maxrows);p->C=(p->cap+31)&~31;
    const size_t need=scratch_layout(*p,b->args,nullptr);p->weight_bytes=weight_layout(*p,nullptr);
    int device=0;auto e=cudaGetDevice(&device);if(e!=cudaSuccess){delete p;return int(e);}
    if(owner){
        auto w=owner->workspace;
        if(!w||w->device!=device||p->cap>w->cap||p->maxrows>w->maxrows||p->R>w->R||p->dense_limit>w->dense_limit||p->H>w->H||(p->full_index&&!w->full_index)||need>w->bytes){delete p;return int(cudaErrorInvalidValue);}
        p->workspace=w;w->refs.fetch_add(1,std::memory_order_relaxed);
    }else{
        auto w=new G6AttnPrefillScratch;p->workspace=w;
        // A shared root is max HMAX/full (H11 at TP6) even if its first layer is HMIN/reuse.
        G6AttnPrefill plan=*p;if(shared){plan.H=g6nr::HMAX;plan.full_index=true;}
        w->bytes=scratch_layout(plan,b->args,nullptr);w->H=plan.H;w->full_index=plan.full_index;w->cap=p->cap;w->maxrows=p->maxrows;w->R=p->R;w->dense_limit=p->dense_limit;w->device=device;
        e=cudaMalloc(&w->data,w->bytes);if(e!=cudaSuccess){delete w;delete p;return int(e);}
        int rc=blas_status(cublasCreate(&w->blas));if(!rc)rc=blas_status(cublasSetMathMode(w->blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        if(rc){if(w->blas)cublasDestroy(w->blas);cudaFree(w->data);delete w;delete p;return rc;}
    }
    // Charge the shared allocation exactly once to its original root for
    // existing runtime sum(bytes) reporting. Refcounts, not this accounting,
    // control lifetime if that root is destroyed before its borrowers.
    p->bytes=p->weight_bytes+(owner?0:p->workspace->bytes);
    e=cudaMalloc(&p->storage,p->weight_bytes);if(e!=cudaSuccess){release_scratch(p->workspace);delete p;return int(e);}
    e=profile_create(p->profile,events,p->maxrows,p->R);
    if(e!=cudaSuccess){cudaFree(p->storage);release_scratch(p->workspace);delete p;return int(e);}
    weight_layout(*p,p->storage);scratch_layout(*p,b->args,p->workspace->data);
    // One handle follows the shared workspace lifetime. All calls are serialized
    // and use the same math mode; no per-layer cuBLAS state is required.
    p->blas=p->workspace->blas;
    cudaStream_t st=(cudaStream_t)s;weight_half<<<(p->H*512*192+255)/256,256,0,st>>>(b->args.uk,p->uk,p->H*512*192);weight_half<<<(p->H*256*512+255)/256,256,0,st>>>(b->args.uv,p->uv,p->H*256*512);e=cudaGetLastError();
    if(e!=cudaSuccess){profile_release(p->profile);cudaFree(p->storage);release_scratch(p->workspace);delete p;return int(e);}*out=p;return 0;
}
}
extern "C" int g6_attn_bind_prefill_create(G6AttnPrefill** out,G6AttnBinding* b,void* s){return gp::create_state(out,b,s,nullptr,false);}
extern "C" int g6_attn_bind_prefill_create_shared(G6AttnPrefill** out,G6AttnBinding* b,void* s,G6AttnPrefill* owner){return gp::create_state(out,b,s,owner,true);}
extern "C" int g6_attn_bind_prefill_release(G6AttnPrefill* p){
    if(!p)return 0;auto pe=gp::profile_release(p->profile);auto e=cudaFree(p->storage);auto w=gp::release_scratch(p->workspace);delete p;
    return pe!=cudaSuccess?int(pe):(e!=cudaSuccess?int(e):int(w));
}
extern "C" int g6_attn_bind_prefill_get_config(const G6AttnPrefill* p,int* tile,int* dense_limit){
    if(!p)return int(cudaErrorInvalidValue);if(tile)*tile=p->R;if(dense_limit)*dense_limit=p->dense_limit;return 0;
}
extern "C" int g6_attn_bind_prefill_profile_read(G6AttnPrefill* p,G6AttnPrefillProfile* out){
    if(!p||!out)return int(cudaErrorInvalidValue);return int(gp::profile_read(p->profile,*out));
}
extern "C" size_t g6_attn_bind_prefill_bytes(const G6AttnPrefill* p){return p?p->bytes:0;}
extern "C" int g6_attn_bind_prefill_trace(G6AttnPrefill* p,G6AttnBinding* b,const G6AttnStep* step,const G6AttnTrace* trace,void* s){
    if(!p)return int(cudaErrorInvalidValue);gp::profile_reset(p->profile);
    if(p->owner!=b||!step)return int(cudaErrorInvalidValue);int rc=g6_attn_bind_prepare(b,step,s);if(rc)return rc;G6Attn a=b->args;if(a.rows>p->maxrows||a.context>p->cap)return int(cudaErrorInvalidValue);
    cudaStream_t st=(cudaStream_t)s;rc=gp::blas_status(cublasSetStream(p->blas,st));if(rc)return rc;
    bool profiling=false;rc=int(gp::profile_begin(p->profile,st,a.rows,a.context,a.heads,p->R,!a.reuse_indices,trace!=nullptr,profiling));if(rc)return rc;
#define GP_PROFILE() do{if(profiling){rc=int(gp::profile_mark(p->profile,st));if(rc)return rc;}}while(0)
    G6_PF_MARK(0,st);rc=g6_attn_append(&a,s);if(rc)return rc;
    // Append validates the public <=64 decode scratch tile. Only the private
    // prefill workspace below uses the larger tile; no decode ABI is widened.
    a.query_tile=p->R;int expanded=(a.context+31)&~31;
    G6_PF_MARK(1,st);gp::expand<<<(expanded+7)/8,256,0,st>>>(a,p->cache,p->ikey,p->kscale,expanded);auto w=gp::index_workspace(a,p->indexwork,p->full_index);float one=1,zero=0,scale=1.f/16;G6AttnTrace tr=trace?*trace:G6AttnTrace{};
#define GP_BLAS(call) do{rc=gp::blas_status(call);if(rc)return rc;}while(0)
    for(int start=0;start<a.rows;start+=p->R){int n=std::min(p->R,a.rows-start),R=p->R,H=a.heads;
        GP_PROFILE();G6_PF_MARK(2,st);if(a.reuse_indices)ga::reuse<<<n,256,0,st>>>(a,start);else{
            ga::prep_index<<<dim3(n,32),128,0,st>>>(a,start,w.iq,w.weights);
#if defined(PF_ISGRT) && PF_ISGRT
            G6_PF_MARK(3,st);gp::launch_index_scores_rg(a,start,n,w.iq,w.weights,p->ikey,p->kscale,w.si,w.ii,st);
#else
            G6_PF_MARK(3,st);gp::index_scores_grouped<<<dim3((a.context+63)/64,(n+3)/4),256,0,st>>>(a,start,n,w.iq,w.weights,p->ikey,p->kscale,w.si,w.ii);
#endif
            if(tr.index_scores)cudaMemcpyAsync(tr.index_scores+(int64_t)start*a.context,w.si,(size_t)n*a.context*4,cudaMemcpyDeviceToDevice,st);
            G6_PF_MARK(4,st);auto e=gp::select_prefill(a,w,start,n,st);if(e!=cudaSuccess)return int(e);
        }
        GP_PROFILE();G6_PF_MARK(5,st);gp::pack_uk_input<<<(H*R*192+255)/256,256,0,st>>>(a,start,n,R,p->q);GP_BLAS(cublasGemmStridedBatchedEx(p->blas,CUBLAS_OP_T,CUBLAS_OP_N,512,R,192,&one,p->uk,CUDA_R_16F,192,512*192,
            p->q,CUDA_R_16F,576,(long long)R*576,&zero,p->qf,CUDA_R_32F,512,(long long)R*512,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        gp::trace_h(tr.q,p->qf,start,n,R,H,512,st);gp::pack_q<<<(R*H*576+255)/256,256,0,st>>>(a,start,n,R,p->qf,p->q);
        GP_PROFILE();G6_PF_MARK(6,st);const bool dense=a.context<=p->dense_limit;const int D=expanded;
        const bool mapped=dense&&a.context>0&&D<=8192&&H*ga::TOP>=a.context;
        if(dense){GP_BLAS(cublasGemmEx(p->blas,CUBLAS_OP_T,CUBLAS_OP_N,D,H*R,576,&scale,p->cache,CUDA_R_16F,576,p->q,CUDA_R_16F,576,&zero,p->dense,CUDA_R_32F,D,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            if(!mapped)gp::dense_gather<<<(n*H*ga::TOP+255)/256,256,0,st>>>(a,start,n,R,p->dense,p->prob);
        }else {if(a.heads>16)gp::sparse_qk<2><<<dim3(ga::TOP/64,n),128,0,st>>>(a,start,R,p->q,p->cache,p->prob);else gp::sparse_qk<1><<<dim3(ga::TOP/64,n),128,0,st>>>(a,start,R,p->q,p->cache,p->prob);}
        G6_PF_MARK(7,st);if(mapped){
            cudaMemsetAsync(p->prob,0xff,(size_t)R*a.context*4,st);
            gp::dense_links<<<(n*ga::TOP+255)/256,256,0,st>>>(a,start,n,(int*)p->prob,p->next);
            gpmap::normalize_expand<<<R*H,256,std::max(D,ga::TOP)*4,st>>>(a,start,n,R,D,p->dense,(const int*)p->prob,p->next,p->densep,tr.scores,tr.probs);
        }else{gp::trace_copy(tr.scores,p->prob,start,n,H,ga::TOP,st);ga::normalize<<<n*H,256,0,st>>>(a,p->prob);gp::trace_copy(tr.probs,p->prob,start,n,H,ga::TOP,st);}
        G6_PF_MARK(8,st);if(dense){if(!mapped){cudaMemsetAsync(p->dense,0xff,(size_t)R*a.context*4,st);gp::dense_links<<<(n*ga::TOP+255)/256,256,0,st>>>(a,start,n,(int*)p->dense,p->next);gp::dense_probabilities<<<(H*R*D+255)/256,256,0,st>>>(a,n,R,D,p->prob,(const int*)p->dense,p->next,p->densep);}
            GP_BLAS(cublasGemmEx(p->blas,CUBLAS_OP_N,CUBLAS_OP_N,512,H*R,D,&one,p->cache,CUDA_R_16F,576,p->densep,CUDA_R_16F,D,&zero,p->ctx,CUDA_R_32F,512,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }else {if(a.heads>16)gp::sparse_pv<2><<<dim3(8,n),128,0,st>>>(a,start,R,p->prob,p->cache,p->ctx);else gp::sparse_pv<1><<<dim3(8,n),128,0,st>>>(a,start,R,p->prob,p->cache,p->ctx);}
        GP_PROFILE();G6_PF_MARK(9,st);gp::trace_h(tr.context,p->ctx,start,n,R,H,512,st);gp::cast_half<<<(H*R*512+255)/256,256,0,st>>>(p->ctx,p->ctxh,H*R*512);
        GP_BLAS(cublasGemmStridedBatchedEx(p->blas,CUBLAS_OP_T,CUBLAS_OP_N,256,R,512,&one,p->uv,CUDA_R_16F,512,256*512,p->ctxh,CUDA_R_16F,512,(long long)R*512,&zero,p->q,CUDA_R_16F,256,(long long)R*256,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        gp::unpack_uv<<<(n*H*256+255)/256,256,0,st>>>(a,start,n,R,p->q);GP_PROFILE();
    }
    G6_PF_MARK(10,st);
#undef GP_BLAS
#undef GP_PROFILE
    rc=int(cudaGetLastError());if(!rc&&profiling)p->profile.report.status=G6_PF_PROFILE_READY;return rc;
}
extern "C" int g6_attn_bind_prefill_run(G6AttnPrefill* p,G6AttnBinding* b,const G6AttnStep* step,void* s){return g6_attn_bind_prefill_trace(p,b,step,nullptr,s);}
extern "C" int g6_attn_bind_baseline_trace(G6AttnBinding* b,const G6AttnStep* step,const G6AttnTrace* trace,void* s){int rc=g6_attn_bind_prepare(b,step,s);if(rc)return rc;auto a=b->args;auto st=(cudaStream_t)s;auto w=ga::workspace(a,b->scratch);rc=g6_attn_append(&a,s);if(rc)return rc;G6AttnTrace tr=trace?*trace:G6AttnTrace{};
    for(int start=0;start<a.rows;start+=a.query_tile){int n=std::min(a.rows-start,a.query_tile),H=a.heads;
        if(a.reuse_indices)ga::reuse<<<n,256,0,st>>>(a,start);else{ga::prep_index<<<dim3(n,32),128,0,st>>>(a,start,w.iq,w.weights);ga::launch_index_scores(a,a,w,start,n,st);if(tr.index_scores)cudaMemcpyAsync(tr.index_scores+(int64_t)start*a.context,w.si,(size_t)n*a.context*4,cudaMemcpyDeviceToDevice,st);auto e=ga::select_index(a,w,start,n,st);if(e!=cudaSuccess)return int(e);}
        ga::absorb<<<dim3(64,H,n),256,0,st>>>(a,start,w.q,w.rope);gp::trace_copy(tr.q,w.q,start,n,H,512,st);ga::launch_mla_scores(a,a,w,start,n,st);gp::trace_copy(tr.scores,w.prob,start,n,H,ga::TOP,st);
        ga::normalize<<<n*H,256,0,st>>>(a,w.prob);gp::trace_copy(tr.probs,w.prob,start,n,H,ga::TOP,st);ga::launch_context_parts(a,a,w,start,n,st);ga::merge_parts<<<n*H,256,0,st>>>(a,w.parts,w.context);gp::trace_copy(tr.context,w.context,start,n,H,512,st);ga::value_project<<<dim3(32,H,n),256,0,st>>>(a,start,w.context);
    }return int(cudaGetLastError());
}

#include "prefill-index-team-rows.cuh"
