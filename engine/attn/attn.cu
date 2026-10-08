// Independently written g6 paged sparse MLA + indexer. No inference-engine code.
#include "attn.h"
#include "replay.h"
#include "../g6nr.h"
#include "../h2d.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <float.h>
#include <limits.h>
#include <algorithm>
#include <cmath>
namespace ga {
constexpr int TOP=2048, PART=8;
constexpr int KV_BYTES=656, IDX_BYTES=132;
using half=__half; using bf16=__nv_bfloat16;
// The old C ABI remains unchanged. Static launches fold this resolver away;
// replay launches retain capacity strides but read live bounds from device RAM.
struct ReplayInput { G6Attn fixed; const G6AttnStepCell* cell; };
struct DeviceAttn : G6Attn { int live_context; bool active; };
__device__ __forceinline__ DeviceAttn resolve(G6Attn fixed) {
    DeviceAttn a; static_cast<G6Attn&>(a)=fixed;
    a.live_context=fixed.context; a.active=true; return a;
}
__device__ __forceinline__ DeviceAttn resolve(ReplayInput input) {
    DeviceAttn a=resolve(input.fixed);
#if defined(G6_CP) && G6_CP
    a.live_context=input.cell->context; if(!a.cp_nr)a.page_count=input.cell->page_count;   // CP: physical pages are fixed
#else
    a.live_context=input.cell->context; a.page_count=input.cell->page_count;
#endif
    a.table_size=input.cell->table_size; a.reuse_rows=input.cell->reuse_rows;
    a.active=input.cell->valid!=0; return a;
}
__global__ void validate_replay(G6Attn fixed,G6AttnStepCell* cell) {
    const int t=threadIdx.x;
    if(t==0)*fixed.error=0;
    __syncthreads();
#if defined(G6_CP) && G6_CP
    const bool bounds=cell->context>0&&cell->context<=fixed.context&&
        (fixed.cp_nr||(cell->page_count>0&&cell->page_count<=fixed.page_count))&&
#else
    const bool bounds=cell->context>0&&cell->context<=fixed.context&&
        cell->page_count>0&&cell->page_count<=fixed.page_count&&
#endif
        cell->table_size>0&&cell->table_size<=fixed.table_size&&
        ((int64_t)cell->context+fixed.page_size-1)/fixed.page_size<=cell->table_size&&
        (fixed.reuse_indices?(cell->reuse_rows>0&&cell->reuse_rows<=fixed.reuse_rows&&
         (fixed.reuse_row_map||cell->reuse_rows>=fixed.rows)):cell->reuse_rows==0);
    if(!bounds){if(t==0)*fixed.error=32;}
    else for(int row=t;row<fixed.rows;row+=blockDim.x){
        const int pos=fixed.positions[row];
        if(pos<0||pos>=cell->context)atomicOr(fixed.error,1u);
        else{
            const int pg=fixed.block_table[pos/fixed.page_size];
#if defined(G6_CP) && G6_CP
            if(!(fixed.cp_nr&&pg==G6_CP_REMOTE)&&(pg<0||pg>=(fixed.cp_nr?fixed.page_count:cell->page_count)))atomicOr(fixed.error,2u);   // CP: physical bound
#else
            if(pg<0||pg>=cell->page_count)atomicOr(fixed.error,2u);
#endif
        }
        for(int j=0;j<row;j++)if(pos==fixed.positions[j])atomicOr(fixed.error,1u);
        if(fixed.reuse_indices&&fixed.reuse_row_map){
            const int src=fixed.reuse_row_map[row];
            if(src<0||src>=cell->reuse_rows)atomicOr(fixed.error,8u);
        }
    }
    __syncthreads();if(t==0)cell->valid=*fixed.error==0;
}

__device__ float bfloat(const void* p,int64_t i){return __bfloat162float(((const bf16*)p)[i]);}
__device__ float hfloat(const void* p,int64_t i){return __half2float(((const half*)p)[i]);}
__device__ float e4(uint8_t x){__nv_fp8_e4m3 a;a.__x=x;return (float)a;}
__device__ uint8_t q4(float x){return __nv_fp8_e4m3(x).__x;}
__device__ void error(const G6Attn& a,unsigned code){atomicOr(a.error,code);}
#if defined(G6_CP) && G6_CP
// CP: the rank that scores key pos (indexer and split-K partial); every key is scored by exactly one rank
__device__ __forceinline__ bool cp_mine(const G6Attn& a,int pos){return !a.cp_nr||(pos/a.page_size)%a.cp_nr==a.cp_rank;}
// CP: rows whose page is stored elsewhere append nothing here (its owner appends it)
__device__ __forceinline__ bool cp_remote(const G6Attn& a,int pos){return a.cp_nr&&pos>=0&&pos/a.page_size<a.table_size&&a.block_table[pos/a.page_size]==G6_CP_REMOTE;}
#define CP_MINE(a,pos) ga::cp_mine(a,pos)
#define CP_REMOTE_ROW(a,pos) if(ga::cp_remote(a,pos))return;
#else
#define CP_MINE(a,pos) true
#define CP_REMOTE_ROW(a,pos)
#endif
__device__ void cs(int pos,int j,float& c,float& s){
    double sd,cd;sincos((double)pos*exp((-2.*j/64)*log(8000000.)),&sd,&cd);c=(float)cd;s=(float)sd;
}
#if PF_ROPE2
// PF_ROPE2: {cos,sin} from the PF_QOROPE table (bit-equal to cs for pos < cap), set at QO create; cs beyond cap or before
__device__ const float2* g_rope=nullptr;__device__ int g_rope_cap=0;
__device__ __forceinline__ void cs_t(int pos,int j,float& c,float& s){
    const float2* t=g_rope;if(t&&pos>=0&&pos<g_rope_cap){const float2 v=t[pos*32+j];c=v.x;s=v.y;}else cs(pos,j,c,s);
}
#define RCS cs_t
#else
#define RCS cs
#endif
__device__ float warp_sum(float x){for(int o=16;o;o>>=1)x+=__shfl_xor_sync(0xffffffff,x,o);return x;}
__device__ float warp_max(float x){for(int o=16;o;o>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,o));return x;}
__device__ int page(const DeviceAttn& a,int pos){
    if(pos<0 || pos>=a.live_context || pos/a.page_size>=a.table_size){error(a,1);return -1;}
    int p=a.block_table[pos/a.page_size];if(p<0||p>=a.page_count){error(a,2);return -1;}return p;
}
__device__ const uint8_t* kv_at(const DeviceAttn& a,int pos){
    int p=page(a,pos);return p<0?nullptr:(const uint8_t*)a.mla_cache+((int64_t)p*a.page_size+pos%a.page_size)*KV_BYTES;
}
#if defined(G6_CP) && G6_CP
// CP: index keys are REPLICATED (BUILD-CPE step 3b): the index cache is logical-page addressed (table_size pages) on every
// rank; below R the replicated path's identity prefix of the block table gives the same address
__device__ __forceinline__ int ipage(const DeviceAttn& a,int pos){
    if(a.cp_nr){if(pos<0||pos>=a.live_context||pos/a.page_size>=a.table_size){error(a,1);return -1;}return pos/a.page_size;}
    return page(a,pos);
}
#else
__device__ __forceinline__ int ipage(const DeviceAttn& a,int pos){return page(a,pos);}
#endif
__device__ const uint8_t* ik_at(const DeviceAttn& a,int pos,float& scale){
    int p=ipage(a,pos);if(p<0){scale=0;return nullptr;}
    const uint8_t* base=(const uint8_t*)a.index_cache+(int64_t)p*a.page_size*IDX_BYTES;
    scale=((const float*)(base+a.page_size*128))[pos%a.page_size];return base+(pos%a.page_size)*128;
}
__device__ float latent(const uint8_t* v,int d){return e4(v[d])*((const float*)(v+512))[d/128];}

template<class Input>
__global__ void append_kv(Input input){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.x,t=threadIdx.x,pos=a.positions[row];CP_REMOTE_ROW(a,pos)int pg=page(a,pos);if(pg<0)return;
    __shared__ float v[512],sums[8],sc[4];
    float ss=0;
    for(int d=t;d<512;d+=256){float x=hfloat(a.kv_a,(int64_t)row*576+d);v[d]=x;ss=fmaf(x,x,ss);}
    ss=warp_sum(ss);if((t&31)==0)sums[t/32]=ss;__syncthreads();
    float total=0;for(int j=0;j<8;j++)total+=sums[j];float inv=rsqrtf(total/512+1e-5f);
    for(int d=t;d<512;d+=256)v[d]*=inv*bfloat(a.kv_norm,d);__syncthreads();
    if(t<4){float mx=0;for(int d=t*128;d<(t+1)*128;d++)mx=fmaxf(mx,fabsf(v[d]));sc[t]=fmaxf(mx/448.f,FLT_MIN);}
    __syncthreads();
    uint8_t* dst=(uint8_t*)a.mla_cache+((int64_t)pg*a.page_size+pos%a.page_size)*KV_BYTES;
    for(int d=t;d<512;d+=256){if(!isfinite(v[d]))error(a,4);dst[d]=q4(v[d]/sc[d/128]);}
    if(t<4)((float*)(dst+512))[t]=sc[t];
    if(t<32){float c,s;RCS(pos,t,c,s);float x=hfloat(a.kv_a,(int64_t)row*576+512+2*t),y=hfloat(a.kv_a,(int64_t)row*576+513+2*t);
        ((bf16*)(dst+528))[2*t]=__float2bfloat16_rn(x*c-y*s);((bf16*)(dst+528))[2*t+1]=__float2bfloat16_rn(y*c+x*s);}
}
template<class Input>
__global__ void append_index(Input input){
    auto a=resolve(input);if(!a.active)return;
    int r=blockIdx.x,t=threadIdx.x,pos=a.positions[r];int pg=ipage(a,pos);if(pg<0)return;   // CP: every rank appends every key
    __shared__ float v[128],sum[4],scale;
    float x=hfloat(a.index_k,(int64_t)r*128+t),s=warp_sum(x);if((t&31)==0)sum[t/32]=s;__syncthreads();
    float mean=(sum[0]+sum[1]+sum[2]+sum[3])/128;
    float dx=x-mean;float ss=warp_sum(dx*dx);__syncthreads();if((t&31)==0)sum[t/32]=ss;__syncthreads();
    v[t]=dx*rsqrtf((sum[0]+sum[1]+sum[2]+sum[3])/128+1e-6f)*bfloat(a.index_norm,t)+bfloat(a.index_bias,t);__syncthreads();
    if(t<32){float c,s;RCS(pos,t,c,s);float a=v[2*t],b=v[2*t+1];v[2*t]=a*c-b*s;v[2*t+1]=b*c+a*s;}__syncthreads();
    float mx=warp_max(fabsf(v[t]));if((t&31)==0)sum[t/32]=mx;__syncthreads();
    if(t==0)scale=exp2f(ceilf(log2f(fmaxf(fmaxf(fmaxf(sum[0],sum[1]),fmaxf(sum[2],sum[3])),1e-4f)/448.f)));__syncthreads();
    uint8_t* base=(uint8_t*)a.index_cache+(int64_t)pg*a.page_size*IDX_BYTES;
    if(!isfinite(v[t]))error(a,4);base[(pos%a.page_size)*128+t]=q4(v[t]/scale);
    if(t==0)((float*)(base+a.page_size*128))[pos%a.page_size]=scale;
}
template<class Input>
__global__ void prep_index(Input input,int start,half* q,float* w){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.x,r=start+row,h=blockIdx.y,t=threadIdx.x;__shared__ float v[128],maxima[4],scale;
    v[t]=hfloat(a.index_q,((int64_t)r*32+h)*128+t);__syncthreads();
    if(t<32){float c,s;RCS(a.positions[r],t,c,s);float x=v[2*t],y=v[2*t+1];v[2*t]=x*c-y*s;v[2*t+1]=y*c+x*s;}__syncthreads();
    float mx=warp_max(fabsf(v[t]));if((t&31)==0)maxima[t/32]=mx;__syncthreads();
    if(t==0){float am=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));scale=exp2f(ceilf(log2f(fmaxf(am,1e-10f)/448.f)));
        w[row*32+h]=hfloat(a.index_weight,(int64_t)r*32+h)*scale*(1.f/64.f);}
    __syncthreads();q[((int64_t)row*32+h)*128+t]=__float2half_rn(e4(q4(v[t]/scale)));
}
// Four warps: two 16-head tiles x two 16-key tiles; no approximate candidate reduction.
template<class Input>
__global__ void index_scores(Input input,int start,const half* q,const float* weights,float* scores,int* ids){
    auto a=resolve(input);if(!a.active)return;
    using namespace nvcuda;
    int row=blockIdx.y,r=start+row,t0=blockIdx.x*32,t=threadIdx.x,warp=t/32;
    __shared__ __align__(32) half key[32*128];
    __shared__ __align__(32) float products[32*32];
    __shared__ float scales[32];
    if(t0>a.positions[r]){if(t<32&&t0+t<a.context){scores[(int64_t)row*a.context+t0+t]=-INFINITY;ids[(int64_t)row*a.context+t0+t]=t0+t;}return;}
    for(int i=t;i<32*128;i+=128){int col=i/128,dim=i%128,pos=t0+col;float scale=0;const uint8_t* src=pos<a.live_context&&CP_MINE(a,pos)?ik_at(a,pos,scale):nullptr;
        key[i]=__float2half_rn(src?e4(src[dim]):0);if(dim==0)scales[col]=scale;}
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> A;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> B;
    wmma::fragment<wmma::accumulator,16,16,16,float> C;wmma::fill_fragment(C,0.f);
    int hs=(warp/2)*16,ts=(warp%2)*16;
    for(int k=0;k<128;k+=16){wmma::load_matrix_sync(A,q+((int64_t)row*32+hs)*128+k,128);wmma::load_matrix_sync(B,key+ts*128+k,128);wmma::mma_sync(C,A,B,C);}
    wmma::store_matrix_sync(products+hs*32+ts,C,32,wmma::mem_row_major);__syncthreads();
    if(t<32&&t0+t<a.context){float score=0;for(int h=0;h<32;h++)score+=fmaxf(products[h*32+t],0.f)*weights[row*32+h];score*=scales[t];
        int pos=t0+t;if(pos>a.positions[r]||!CP_MINE(a,pos))score=-INFINITY;else if(!isfinite(score)){error(a,4);score=-INFINITY;}
        scores[(int64_t)row*a.context+pos]=score;ids[(int64_t)row*a.context+pos]=pos;}
}
__global__ void offsets(int* p,int count,int context){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<=count)p[i]=i*context;}
template<class Input>
__global__ void top_from_sort(Input input,int start,const float* scores,const int* ids){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.x,r=start+row;for(int j=threadIdx.x;j<TOP;j+=256){int v=-1;
        if(j<a.context){int id=ids[(int64_t)row*a.context+j];float score=scores[(int64_t)row*a.context+j];if(isfinite(score)&&id>=0&&id<=a.positions[r])v=id;}
        a.indices[(int64_t)r*TOP+j]=v;}
}
#include "select.cuh"
template<class Input>
__global__ void reuse(Input input,int start){
    auto a=resolve(input);if(!a.active)return;int r=start+blockIdx.x;int src=a.reuse_row_map?a.reuse_row_map[r]:r;
    if(src<0||src>=a.reuse_rows){error(a,8);for(int j=threadIdx.x;j<TOP;j+=256)a.indices[(int64_t)r*TOP+j]=-1;return;}
    for(int j=threadIdx.x;j<TOP;j+=256)a.indices[(int64_t)r*TOP+j]=a.reuse_indices[(int64_t)src*TOP+j];
}
template<class Input>
__global__ void absorb(Input input,int start,float* q,float* rope){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.z,r=start+row,h=blockIdx.y,d=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(d<512){float sum=0;for(int k=lane;k<192;k+=32)sum=fmaf(hfloat(a.q_b,((int64_t)r*a.heads+h)*256+k),bfloat(a.uk,((int64_t)h*512+d)*192+k),sum);
        sum=warp_sum(sum);if(lane==0)q[((int64_t)row*a.heads+h)*512+d]=sum;}
    if(blockIdx.x==0&&threadIdx.x<32){float c,s;RCS(a.positions[r],threadIdx.x,c,s);int j=threadIdx.x;
        float x=hfloat(a.q_b,((int64_t)r*a.heads+h)*256+192+2*j),y=hfloat(a.q_b,((int64_t)r*a.heads+h)*256+193+2*j);
        rope[((int64_t)row*a.heads+h)*64+2*j]=x*c-y*s;rope[((int64_t)row*a.heads+h)*64+2*j+1]=y*c+x*s;}
}
template<class Input>
__global__ void mla_scores(Input input,int start,const float* q,const float* rope,float* score){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.z,r=start+row,h=blockIdx.y,j=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,t=threadIdx.x;
    __shared__ float qr[512],rp[64];for(int d=t;d<512;d+=256)qr[d]=q[((int64_t)row*a.heads+h)*512+d];if(t<64)rp[t]=rope[((int64_t)row*a.heads+h)*64+t];__syncthreads();
    int id=a.indices[(int64_t)r*TOP+j];const uint8_t* v=(id>=0&&id<=a.positions[r]&&CP_MINE(a,id))?kv_at(a,id):nullptr;float sum=0;
    if(v){for(int d=lane;d<512;d+=32)sum=fmaf(qr[d],latent(v,d),sum);for(int d=lane;d<64;d+=32)sum=fmaf(rp[d],__bfloat162float(((const bf16*)(v+528))[d]),sum);}
    sum=warp_sum(sum);if(lane==0)score[((int64_t)row*a.heads+h)*TOP+j]=v?sum*(1.f/16):-INFINITY;
}
template<class Input>
__global__ void normalize(Input input,float* scores){
    auto a=resolve(input);if(!a.active)return;
    int i=blockIdx.x,t=threadIdx.x;float* p=scores+(int64_t)i*TOP;__shared__ float tmp[8],mx,den;
    float m=-INFINITY;for(int j=t;j<TOP;j+=256)m=fmaxf(m,p[j]);m=warp_max(m);if((t&31)==0)tmp[t/32]=m;__syncthreads();
    if(t==0){mx=-INFINITY;for(int j=0;j<8;j++)mx=fmaxf(mx,tmp[j]);}__syncthreads();
    float s=0;for(int j=t;j<TOP;j+=256){float v=isfinite(mx)?expf(p[j]-mx):0;p[j]=v;s+=v;}s=warp_sum(s);if((t&31)==0)tmp[t/32]=s;__syncthreads();
    if(t==0){den=0;for(int j=0;j<8;j++)den+=tmp[j];if(den==0)error(a,16);}__syncthreads();
    for(int j=t;j<TOP;j+=256)p[j]=den>0?p[j]/den:0;
}
template<class Input>
__global__ void context_parts(Input input,int start,const float* probs,float* parts){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.z,r=start+row,h=blockIdx.y,part=blockIdx.x,t=threadIdx.x;float lo=0,hi=0;
    const float* p=probs+((int64_t)row*a.heads+h)*TOP;
    // One lookup per selected position/CTA instead of per output thread/position.
    // Keep the original 8 partials and each partial's 256-FMA order unchanged.
    __shared__ uintptr_t rows[TOP/PART];
    __shared__ float probability[TOP/PART],scales[(TOP/PART)*4];
    const int slot=part*(TOP/PART)+t,id=a.indices[(int64_t)r*TOP+slot];
    const float prob=p[slot];const uint8_t* src=nullptr;
    if(id>=0&&id<=a.positions[r]&&prob!=0)src=kv_at(a,id);
    rows[t]=(uintptr_t)src;probability[t]=prob;
    if(src)for(int g=0;g<4;g++)scales[t*4+g]=((const float*)(src+512))[g];
    __syncthreads();
    #pragma unroll 4
    for(int j=0;j<TOP/PART;j++){
        const uint8_t* v=(const uint8_t*)rows[j];if(!v)continue;
        float weight=probability[j];
        lo=fmaf(weight,e4(v[t])*scales[j*4+(t>>7)],lo);
        hi=fmaf(weight,e4(v[t+256])*scales[j*4+2+(t>>7)],hi);
    }
    float* dst=parts+(((int64_t)row*a.heads+h)*PART+part)*512;dst[t]=lo;dst[t+256]=hi;
}
template<class Input>
__global__ void merge_parts(Input input,const float* parts,float* out){
    auto a=resolve(input);if(!a.active)return;
    int rowhead=blockIdx.x,t=threadIdx.x;for(int d=t;d<512;d+=256){float s=0;for(int part=0;part<PART;part++)s+=parts[((int64_t)rowhead*PART+part)*512+d];out[(int64_t)rowhead*512+d]=s;}
}
template<class Input>
__global__ void value_project(Input input,int start,const float* context){
    auto a=resolve(input);if(!a.active)return;
    int row=blockIdx.z,r=start+row,h=blockIdx.y,d=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;float sum=0;
    for(int k=lane;k<512;k+=32)sum=fmaf(context[((int64_t)row*a.heads+h)*512+k],bfloat(a.uv,((int64_t)h*256+d)*512+k),sum);
    sum=warp_sum(sum);if(lane==0)((half*)a.output)[((int64_t)r*a.heads+h)*256+d]=__float2half_rn(sum);
}
// DEC_AVP (bar RESULTS-M20 "DEC_AVP"): absorb / value_project for n <= AVP_ROWS rows in one launch. Grid has no row axis:
// each warp loads its whole lane-strided weight row (k = lane + 32m) into registers once, then per row runs the same
// fmaf chain over the same k order, the same warp_sum, and lane 0 stores, so every q / rope / output byte matches the
// per-row kernels. The inputs of the rows (q_b, context) are staged in shared memory as bit copies.
constexpr int AVP_ROWS=3;
template<class Input>
__global__ void absorb_m(Input input,int start,int n,float* q,float* rope){
    auto a=resolve(input);if(!a.active)return;
    int h=blockIdx.y,d=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,t=threadIdx.x;
    __shared__ float qb[AVP_ROWS][192];
    for(int i=t;i<n*192;i+=256){int row=i/192,k=i%192;qb[row][k]=hfloat(a.q_b,((int64_t)(start+row)*a.heads+h)*256+k);}
    float u[6];
    #pragma unroll
    for(int m=0;m<6;m++)u[m]=bfloat(a.uk,((int64_t)h*512+d)*192+lane+32*m);
    __syncthreads();
    for(int row=0;row<n;row++){float sum=0;
        #pragma unroll
        for(int m=0;m<6;m++)sum=fmaf(qb[row][lane+32*m],u[m],sum);
        sum=warp_sum(sum);if(lane==0)q[((int64_t)row*a.heads+h)*512+d]=sum;}
    if(blockIdx.x==0&&threadIdx.x<32)for(int row=0;row<n;row++){int r=start+row;float c,s;RCS(a.positions[r],threadIdx.x,c,s);int j=threadIdx.x;
        float x=hfloat(a.q_b,((int64_t)r*a.heads+h)*256+192+2*j),y=hfloat(a.q_b,((int64_t)r*a.heads+h)*256+193+2*j);
        rope[((int64_t)row*a.heads+h)*64+2*j]=x*c-y*s;rope[((int64_t)row*a.heads+h)*64+2*j+1]=y*c+x*s;}
}
template<class Input>
__global__ void value_project_m(Input input,int start,int n,const float* context){
    auto a=resolve(input);if(!a.active)return;
    int h=blockIdx.y,d=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,t=threadIdx.x;
    __shared__ float ctx[AVP_ROWS][512];
    for(int i=t;i<n*512;i+=256){int row=i/512,k=i%512;ctx[row][k]=context[((int64_t)row*a.heads+h)*512+k];}
    float u[16];
    #pragma unroll
    for(int m=0;m<16;m++)u[m]=bfloat(a.uv,((int64_t)h*256+d)*512+lane+32*m);
    __syncthreads();
    for(int row=0;row<n;row++){float sum=0;
        #pragma unroll
        for(int m=0;m<16;m++)sum=fmaf(ctx[row][lane+32*m],u[m],sum);
        sum=warp_sum(sum);if(lane==0)((half*)a.output)[((int64_t)(start+row)*a.heads+h)*256+d]=__float2half_rn(sum);}
}
struct Work {
    float *q,*rope,*weights,*si,*so,*prob,*parts,*context;
    half* iq;int *ii,*io,*offsets;uint64_t* selected;SelectionState* selection;int* histogram;void* temp;size_t temp_bytes,total;
};
static bool valid(const G6Attn& a){return a.rows>0&&(a.heads==g6nr::HMIN||a.heads==g6nr::HMAX)&&a.context>0&&a.page_size>0&&!(a.page_size&(a.page_size-1))&&a.page_count>0&&a.table_size>0&&a.query_tile>0&&a.query_tile<=64&&a.context<=INT_MAX/a.query_tile;}
static Work workspace(const G6Attn& a,void* base){
    Work w{};size_t n=0;const size_t R=std::min(a.rows,a.query_tile),H=a.heads,T=a.context;
    auto alloc=[&](size_t bytes){size_t old=n;n=(n+bytes+255)&~size_t(255);return (void*)((uintptr_t)base+old);};
    w.q=(float*)alloc(R*H*512*4);w.rope=(float*)alloc(R*H*64*4);w.iq=(half*)alloc(R*32*128*2);w.weights=(float*)alloc(R*32*4);
    w.si=(float*)alloc(R*T*4);w.so=(float*)alloc(R*T*4);w.ii=(int*)alloc(R*T*4);w.io=(int*)alloc(R*T*4);w.offsets=(int*)alloc((R+1)*4);
    w.selected=(uint64_t*)alloc(R*TOP*8);w.selection=(SelectionState*)alloc(R*sizeof(SelectionState));
    w.histogram=(int*)alloc(R*((std::min(T,size_t(32768))+RADIX_CHUNK-1)/RADIX_CHUNK)*256*4);
    w.prob=(float*)alloc(R*H*TOP*4);w.parts=(float*)alloc(R*H*PART*512*4);w.context=(float*)alloc(R*H*512*4);
    size_t b=0;cudaError_t e=cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,b,w.si,w.so,w.ii,w.io,(int)(R*T),(int)R,w.offsets,w.offsets+1);
    if(e!=cudaSuccess)return Work{};
    // The last query tile can contain one row even when the workspace has R>1.
    size_t single=0;e=cub::DeviceRadixSort::SortPairsDescending(nullptr,single,w.si,w.so,w.ii,w.io,(int)T);
    if(e!=cudaSuccess)return Work{};b=std::max(b,single);w.temp_bytes=b;w.temp=alloc(b);w.total=n;return w;
}
#include "decode-small.cuh"
template<class Input>
static void threshold_compact(const G6Attn& a,Input input,Work& w,int count,cudaStream_t stream){
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;
    for(int pass=0;pass<6;pass++){
        radix_histogram<<<dim3(chunks,count),256,0,stream>>>(input,w.si,w.selection,w.histogram,pass);
        radix_update<<<count,256,0,stream>>>(input,w.selection,w.histogram,w.selected,pass);
    }
    radix_compact<<<dim3(chunks,count),256,0,stream>>>(input,w.si,w.selection,w.selected);
}
#ifdef SEL32
template<class Input>   // context > 32768: the histogram lives in w.so (R*T*4 B >= R*chunks*1 KB), unused without the full sort
static void threshold_compact32(const G6Attn& a,Input input,Work& w,int count,cudaStream_t stream){
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;int* bins=(int*)w.so;
    for(int pass=0;pass<8;pass++){
        radix_histogram32<<<dim3(chunks,count),256,0,stream>>>(input,w.si,w.selection,bins,pass);
        radix_update32<<<count,256,0,stream>>>(input,w.selection,bins,w.selected,pass);
    }
    radix_compact32<<<dim3(chunks,count),256,0,stream>>>(input,w.si,w.selection,w.selected);
}
#endif
static void threshold_compact(const G6Attn& a,Work& w,int count,cudaStream_t stream){threshold_compact(a,a,w,count,stream);}
template<class Input>
static cudaError_t select_index(const G6Attn& a,Input input,Work& w,int start,int count,cudaStream_t stream){
    if(a.context<=TOP){sort_selected<false><<<count,256,0,stream>>>(input,start,w.si,w.selected);return cudaGetLastError();}
    if(a.context<=32768){
        threshold_compact(a,input,w,count,stream);
        sort_selected<true><<<count,256,0,stream>>>(input,start,w.si,w.selected);
        return cudaGetLastError();
    }
#ifdef SEL32
    threshold_compact32(a,input,w,count,stream);
    sort_selected32<<<count,256,0,stream>>>(input,start,w.selected);
    return cudaGetLastError();
#endif
    cudaError_t e;
    // Both CUB sorts are stable: equal scores retain increasing input token ID.
    if(count==1)e=cub::DeviceRadixSort::SortPairsDescending(w.temp,w.temp_bytes,w.si,w.so,w.ii,w.io,a.context,0,32,stream);
    else{
        offsets<<<(count+1+127)/128,128,0,stream>>>(w.offsets,count,a.context);
        e=cub::DeviceSegmentedRadixSort::SortPairsDescending(w.temp,w.temp_bytes,w.si,w.so,w.ii,w.io,count*a.context,count,w.offsets,w.offsets+1,0,32,stream);
    }
    if(e!=cudaSuccess)return e;top_from_sort<<<count,256,0,stream>>>(input,start,w.so,w.io);return cudaGetLastError();
}
static cudaError_t select_index(const G6Attn& a,Work& w,int start,int count,cudaStream_t stream){return select_index(a,a,w,start,count,stream);}
#include "idx31.cuh"
#if defined(DEC_IDX31) && DEC_IDX31
#ifndef SEL32
#error "DEC_IDX31 needs SEL32 (index_scores_m writes no ids; only the full sort reads them)"
#endif
// #31: decode-sized query tiles (<= IDX31_ROWS rows) score with shared keys and select with the live-context bound
constexpr int IDX31_ROWS=8;
template<class Input>
static void launch_index_scores31(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
    if(n<=IDX31_ROWS)launch_index_scores_m(a,input,w,start,n,st);else launch_index_scores(a,input,w,start,n,st);
}
template<class Input>
static cudaError_t select_index_dec(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
    return n<=IDX31_ROWS?select_index31(a,input,w,start,n,st):select_index(a,input,w,start,n,st);
}
#define LAUNCH_INDEX_SCORES launch_index_scores31
#define SELECT_INDEX select_index_dec
#else
#define LAUNCH_INDEX_SCORES launch_index_scores
#define SELECT_INDEX select_index
#endif
template<class Input>
static cudaError_t make_index(const G6Attn& a,Input input,Work& w,int start,int count,cudaStream_t stream,float* diagnostic){
    prep_index<<<dim3(count,32),128,0,stream>>>(input,start,w.iq,w.weights);
    LAUNCH_INDEX_SCORES(a,input,w,start,count,stream);
    if(diagnostic)return cudaMemcpyAsync(diagnostic+(int64_t)start*a.context,w.si,(size_t)count*a.context*4,cudaMemcpyDeviceToDevice,stream);
    return SELECT_INDEX(a,input,w,start,count,stream);
}
static cudaError_t make_index(const G6Attn& a,Work& w,int start,int count,cudaStream_t stream,float* diagnostic){return make_index(a,a,w,start,count,stream,diagnostic);}
template<class Input>   // DEC_AVP launches (tests call these regardless of the build flag)
static void launch_absorb_m(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){absorb_m<<<dim3(64,a.heads),256,0,st>>>(input,start,n,w.q,w.rope);}
template<class Input>
static void launch_value_project_m(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){value_project_m<<<dim3(32,a.heads),256,0,st>>>(input,start,n,w.context);}
template<class Input>
static cudaError_t run_body(const G6Attn& a,Input input,Work& w,cudaStream_t stream){
    cudaError_t e;
    for(int start=0;start<a.rows;start+=a.query_tile){int n=std::min(a.rows-start,a.query_tile);
        if(a.reuse_indices)ga::reuse<<<n,256,0,stream>>>(input,start);else{e=ga::make_index(a,input,w,start,n,stream,nullptr);if(e!=cudaSuccess)return e;}
#if defined(DEC_AVP) && DEC_AVP
        if(n<=AVP_ROWS)launch_absorb_m(a,input,w,start,n,stream);else
#endif
        ga::absorb<<<dim3(64,a.heads,n),256,0,stream>>>(input,start,w.q,w.rope);
        ga::launch_mla_scores(a,input,w,start,n,stream);
        ga::normalize<<<n*a.heads,256,0,stream>>>(input,w.prob);
        ga::launch_context_parts(a,input,w,start,n,stream);
        ga::merge_parts<<<n*a.heads,256,0,stream>>>(input,w.parts,w.context);
#if defined(DEC_AVP) && DEC_AVP
        if(n<=AVP_ROWS)launch_value_project_m(a,input,w,start,n,stream);else
#endif
        ga::value_project<<<dim3(32,a.heads,n),256,0,stream>>>(input,start,w.context);
    }return cudaGetLastError();
}
#if defined(G6_CP) && G6_CP
// ---- J4 context-parallel decode (BUILD-CPE.md, DESIGN-CP §2-3). Candidate key = the selector's own order:
// ordered_score (select.cuh: CUB float order, -0 == +0, nonfinite -> 0) << 32 | (0xffffffff - pos); 0 = no candidate.
// Keys are unique (every key is scored by exactly one rank), so descending key order is a total order.
__device__ __forceinline__ uint64_t cp_key(float score,int pos){
    const uint32_t k=ordered_score(score);return k?((uint64_t)k<<32)|(0xffffffffu-(uint32_t)pos):0;
}
// my local top-2048 (a.indices of rows start.., descending) + the scores the indexer wrote (w.si) -> my candidate list
template<class Input>
__global__ void cp_cand_pack(Input input,int start,const float* scores,uint64_t* cand){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,r=start+row;
    for(int j=threadIdx.x;j<TOP;j+=256){const int id=a.indices[(int64_t)r*TOP+j];
        cand[(int64_t)r*TOP+j]=id>=0?cp_key(scores[(int64_t)row*a.context+id],id):0;}
}
// exact merge of cp_nr descending candidate lists [s][rows][TOP]: an entry's output rank = its index in its own list +
// the number of larger keys in every other list (binary search). Ranks < TOP are written; the rest of the row is -1.
template<class Input>
__global__ void cp_merge(Input input,int start,const uint64_t* cand){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,r=start+row,t=threadIdx.x,nr=a.cp_nr;
    int* out=a.indices+(int64_t)r*TOP;
    for(int j=t;j<TOP;j+=256)out[j]=-1;
    __syncthreads();
    for(int s=0;s<nr;s++){const uint64_t* L=cand+((int64_t)s*a.rows+r)*TOP;
        for(int j=t;j<TOP;j+=256){const uint64_t k=L[j];if(!k)continue;int rank=j;
            for(int o=0;o<nr&&rank<TOP;o++){if(o==s)continue;const uint64_t* O=cand+((int64_t)o*a.rows+r)*TOP;
                int lo=0,hi=TOP;while(lo<hi){const int mid=(lo+hi)>>1;if(O[mid]>k)lo=mid+1;else hi=mid;}   // count of keys > k
                rank+=lo;}
            if(rank<TOP)out[rank]=int(0xffffffffu-uint32_t(k));}}
}
// unnormalized softmax of one head group's scores over my keys: p = exp(s - m) in place, m, l per (row, head).
// Same reduction structure as normalize; a row-head with no key of mine gives m = -inf, p = 0, l = 0 (no error).
template<class Input>
__global__ void cp_softmax(Input input,float* scores,float* m_out,float* l_out){
    auto a=resolve(input);if(!a.active)return;
    int i=blockIdx.x,t=threadIdx.x;float* p=scores+(int64_t)i*TOP;__shared__ float tmp[8],mx,den;
    float m=-INFINITY;for(int j=t;j<TOP;j+=256)m=fmaxf(m,p[j]);m=warp_max(m);if((t&31)==0)tmp[t/32]=m;__syncthreads();
    if(t==0){mx=-INFINITY;for(int j=0;j<8;j++)mx=fmaxf(mx,tmp[j]);}__syncthreads();
    float s=0;for(int j=t;j<TOP;j+=256){float v=isfinite(mx)?expf(p[j]-mx):0;p[j]=v;s+=v;}s=warp_sum(s);if((t&31)==0)tmp[t/32]=s;__syncthreads();
    if(t==0){den=0;for(int j=0;j<8;j++)den+=tmp[j];m_out[i]=mx;l_out[i]=den;}
}
// LSE merge of my head group's partials in shard order 0..cp_nr-1: M = max m_s, w_s = exp(m_s - M), L = sum w_s l_s,
// ctx = (sum w_s c_s) / L. part_all: [s] blocks of c [n][H][512], m [n][H], l [n][H] (n = a.rows); block stride rounded
// up to 4 floats (16 B; NR=6 head groups of 10/11 heads make n*H*514 odd x 2)
template<class Input>
__global__ void cp_lse(Input input,const float* part_all,float* context){
    auto a=resolve(input);if(!a.active)return;
    const int i=blockIdx.x,t=threadIdx.x,nh=a.rows*a.heads;const int64_t blk=((int64_t)nh*514+3)&~(int64_t)3;
    float M=-INFINITY;for(int sh=0;sh<a.cp_nr;sh++)M=fmaxf(M,part_all[sh*blk+(int64_t)nh*512+i]);
    float w[8],L=0;
    for(int sh=0;sh<a.cp_nr;sh++){const float m=part_all[sh*blk+(int64_t)nh*512+i];w[sh]=isfinite(m)?expf(m-M):0.f;
        L=fmaf(w[sh],part_all[sh*blk+(int64_t)nh*513+i],L);}
    if(t==0&&!(L>0))error(a,16);
    for(int d=t;d<512;d+=256){float c=0;for(int sh=0;sh<a.cp_nr;sh++)c=fmaf(w[sh],part_all[sh*blk+(int64_t)i*512+d],c);
        context[(int64_t)i*512+d]=L>0?c/L:0.f;}
}
template<class Input>
static cudaError_t cp_front_body(const G6Attn& a,Input input,Work& w,cudaStream_t stream){
    const int n=a.rows;
#if defined(DEC_AVP) && DEC_AVP
    if(n<=AVP_ROWS)launch_absorb_m(a,input,w,0,n,stream);else
#endif
    ga::absorb<<<dim3(64,a.heads,n),256,0,stream>>>(input,0,w.q,w.rope);
    return cudaGetLastError();
}
template<class Input>
static cudaError_t cp_index_body(const G6Attn& a,Input input,Work& w,uint64_t* cand,cudaStream_t stream){
    const int n=a.rows;
    if(a.reuse_indices){ga::reuse<<<n,256,0,stream>>>(input,0);return cudaGetLastError();}
    uint64_t* mine=cand+(int64_t)a.cp_rank*a.rows*TOP;
#if defined(DEC_CHUNKSEL) && DEC_CHUNKSEL
    // J5 select paths (make_index): <= 32768 scores in w.si; > 32768 one row: scores in w.s1 (1 x T); > 32768 several
    // rows: chunked, the last chunk_merge leaves the merged keys (chunk_key == cp_key) in w.carry
    cudaError_t e=make_index(a,input,w,0,n,stream,nullptr);if(e!=cudaSuccess)return e;
    if(a.context>32768&&n>1){cudaError_t c=cudaMemcpyAsync(mine,w.carry,(size_t)n*TOP*8,cudaMemcpyDeviceToDevice,stream);return c!=cudaSuccess?c:cudaGetLastError();}
    cp_cand_pack<<<n,256,0,stream>>>(input,0,a.context>32768?w.s1:w.si,mine);
#else
    prep_index<<<dim3(n,32),128,0,stream>>>(input,0,w.iq,w.weights);
    LAUNCH_INDEX_SCORES(a,input,w,0,n,stream);
    cudaError_t e=SELECT_INDEX(a,input,w,0,n,stream);if(e!=cudaSuccess)return e;
    cp_cand_pack<<<n,256,0,stream>>>(input,0,w.si,mine);
#endif
    return cudaGetLastError();
}
template<class Input>
static cudaError_t cp_group_body(const G6Attn& a,Input input,Work w,const float* q,const float* rope,float* part,cudaStream_t stream){
    const int n=a.rows,nh=n*a.heads;
    w.q=const_cast<float*>(q);w.rope=const_cast<float*>(rope);
    launch_mla_scores(a,input,w,0,n,stream);
    cp_softmax<<<nh,256,0,stream>>>(input,w.prob,part+(int64_t)nh*512,part+(int64_t)nh*513);
    launch_context_parts(a,input,w,0,n,stream);
    merge_parts<<<nh,256,0,stream>>>(input,w.parts,part);
    return cudaGetLastError();
}
template<class Input>
static cudaError_t cp_finish_body(const G6Attn& a,Input input,Work& w,const float* part_all,cudaStream_t stream){
    const int n=a.rows;
    cp_lse<<<n*a.heads,256,0,stream>>>(input,part_all,w.context);
#if defined(DEC_AVP) && DEC_AVP
    if(n<=AVP_ROWS)launch_value_project_m(a,input,w,0,n,stream);else
#endif
    ga::value_project<<<dim3(32,a.heads,n),256,0,stream>>>(input,0,w.context);
    return cudaGetLastError();
}
static bool cp_ok(const G6Attn* a){return a&&valid(*a)&&a->cp_nr>=2&&a->cp_nr<=8&&a->cp_rank>=0&&a->cp_rank<a->cp_nr&&a->rows<=a->query_tile;}
#endif
}
extern "C" size_t g6_attn_scratch_bytes(const G6Attn* a){if(!a||!ga::valid(*a))return 0;return ga::workspace(*a,nullptr).total;}
extern "C" int g6_attn_append(const G6Attn* a,void* s){
    if(!a||!ga::valid(*a)||!a->kv_a||!a->kv_norm||!a->mla_cache||!a->positions||!a->block_table||!a->error)return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;ga::append_kv<<<a->rows,256,0,stream>>>(*a);
    if(!a->reuse_indices){if(!a->index_k||!a->index_norm||!a->index_bias||!a->index_cache)return (int)cudaErrorInvalidValue;ga::append_index<<<a->rows,128,0,stream>>>(*a);}
    return (int)cudaGetLastError();
}
extern "C" int g6_attn_index_scores(const G6Attn* a,void* scratch,size_t bytes,float* scores,void* s){
    if(!a||!ga::valid(*a)||!scratch||!scores)return (int)cudaErrorInvalidValue;auto w=ga::workspace(*a,scratch);if(!w.total||bytes<w.total)return (int)cudaErrorInvalidValue;
    for(int start=0;start<a->rows;start+=a->query_tile){auto e=ga::make_index(*a,w,start,std::min(a->rows-start,a->query_tile),(cudaStream_t)s,scores);if(e!=cudaSuccess)return (int)e;}return (int)cudaGetLastError();
}
extern "C" int g6_attn_run(const G6Attn* a_in,void* scratch,size_t bytes,void* s){
#if defined(G6_CP) && G6_CP
    // a CP binding at context <= R (every page replicated, physical == logical on the prefix): today's replicated path
    G6Attn rep;if(a_in){rep=*a_in;rep.cp_nr=0;rep.cp_rank=0;}const G6Attn* a=a_in?&rep:nullptr;
#else
    const G6Attn* a=a_in;
#endif
    if(!a||!ga::valid(*a)||!scratch||!a->q_b||!a->uk||!a->uv||!a->indices||!a->output)return (int)cudaErrorInvalidValue;
    if(!a->reuse_indices&&(!a->index_q||!a->index_weight))return (int)cudaErrorInvalidValue;
    if(a->reuse_row_map&&a->reuse_indices==a->indices)return (int)cudaErrorInvalidValue;
    auto w=ga::workspace(*a,scratch);if(!w.total||bytes<w.total)return (int)cudaErrorInvalidValue;auto e=(cudaError_t)g6_attn_append(a,s);if(e!=cudaSuccess)return (int)e;
    return (int)ga::run_body(*a,*a,w,(cudaStream_t)s);
}
extern "C" int g6_attn_replay_run(const G6AttnReplay* r,void* s){
    if(!r||!r->cell)return (int)cudaErrorInvalidValue;
#if defined(G6_CP) && G6_CP
    G6Attn rep=r->fixed;rep.cp_nr=0;rep.cp_rank=0;const G6Attn& a=rep;   // CP binding below R: replicated, as g6_attn_run
#else
    const G6Attn& a=r->fixed;
#endif
    if(!ga::valid(a)||!r->scratch||!a.q_b||!a.uk||!a.uv||!a.indices||!a.output||
       !a.kv_a||!a.kv_norm||!a.mla_cache||!a.positions||!a.block_table||!a.error)
        return (int)cudaErrorInvalidValue;
    if(!a.reuse_indices&&(!a.index_q||!a.index_weight||!a.index_k||!a.index_norm||!a.index_bias||!a.index_cache))return (int)cudaErrorInvalidValue;
    if(a.reuse_row_map&&a.reuse_indices==a.indices)return (int)cudaErrorInvalidValue;
    auto w=ga::workspace(a,r->scratch);
    if(!w.total||r->scratch_bytes<w.total)return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;
    ga::validate_replay<<<1,256,0,stream>>>(a,r->cell);
    ga::ReplayInput input{a,r->cell};
    ga::append_kv<<<a.rows,256,0,stream>>>(input);
    if(!a.reuse_indices)ga::append_index<<<a.rows,128,0,stream>>>(input);
    return (int)ga::run_body(a,input,w,stream);
}

#include "prefill.cuh"
#include "prefill-qo-bind.cuh"

#if defined(G6_CP) && G6_CP
// ---- J4 CP entry points (attn.h). Graph form (cell) mirrors g6_attn_replay_run: validation + appends in the front stage.
template<class F>
static int cp_call(const G6Attn* a,const G6AttnStepCell* cell,void* scratch,size_t bytes,F body){
    if(!ga::cp_ok(a)||!scratch)return (int)cudaErrorInvalidValue;
    auto w=ga::workspace(*a,scratch);if(!w.total||bytes<w.total)return (int)cudaErrorInvalidValue;
    if(cell){ga::ReplayInput in{*a,cell};return (int)body(in,w);}
    return (int)body(*a,w);
}
extern "C" int g6_attn_cp_front(const G6Attn* a,const G6AttnStepCell* cell,void* scratch,size_t bytes,void* s){
    if(!a||!a->q_b||!a->uk||!a->kv_a||!a->kv_norm||!a->mla_cache||!a->positions||!a->block_table||!a->error||!a->indices)return (int)cudaErrorInvalidValue;
    if(!a->reuse_indices&&(!a->index_q||!a->index_weight||!a->index_k||!a->index_norm||!a->index_bias||!a->index_cache))return (int)cudaErrorInvalidValue;
    if(a->reuse_row_map&&a->reuse_indices==a->indices)return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;
    return cp_call(a,cell,scratch,bytes,[&](auto in,ga::Work& w){
        if(cell)ga::validate_replay<<<1,256,0,stream>>>(*a,const_cast<G6AttnStepCell*>(cell));
        ga::append_kv<<<a->rows,256,0,stream>>>(in);
        if(!a->reuse_indices)ga::append_index<<<a->rows,128,0,stream>>>(in);
        return ga::cp_front_body(*a,in,w,stream);});
}
extern "C" int g6_attn_cp_index(const G6Attn* a,const G6AttnStepCell* cell,void* scratch,size_t bytes,uint64_t* cand_all,void* s){
    if(!a||(!a->reuse_indices&&!cand_all))return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;
    return cp_call(a,cell,scratch,bytes,[&](auto in,ga::Work& w){return ga::cp_index_body(*a,in,w,cand_all,stream);});
}
extern "C" int g6_attn_cp_merge_index(const G6Attn* a,const G6AttnStepCell* cell,const uint64_t* cand_all,void* s){
    if(!ga::cp_ok(a)||!cand_all||a->reuse_indices)return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;
    if(cell)ga::cp_merge<<<a->rows,256,0,stream>>>(ga::ReplayInput{*a,cell},0,cand_all);
    else ga::cp_merge<<<a->rows,256,0,stream>>>(*a,0,cand_all);
    return (int)cudaGetLastError();
}
extern "C" int g6_attn_cp_group(const G6Attn* a,const G6AttnStepCell* cell,void* scratch,size_t bytes,const float* q,const float* rope,float* part,void* s){
    if(!a||!a->indices||!a->mla_cache||!a->block_table||!q||!rope||!part)return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;
    return cp_call(a,cell,scratch,bytes,[&](auto in,ga::Work& w){return ga::cp_group_body(*a,in,w,q,rope,part,stream);});
}
extern "C" int g6_attn_cp_finish(const G6Attn* a,const G6AttnStepCell* cell,void* scratch,size_t bytes,const float* part_all,void* s){
    if(!a||!a->uv||!a->output||!part_all)return (int)cudaErrorInvalidValue;
    cudaStream_t stream=(cudaStream_t)s;
    return cp_call(a,cell,scratch,bytes,[&](auto in,ga::Work& w){return ga::cp_finish_body(*a,in,w,part_all,stream);});
}
extern "C" int g6_attn_cp_qabs(const G6Attn* a,void* scratch,const float** q,const float** rope){
    if(!ga::cp_ok(a)||!scratch||!q||!rope)return (int)cudaErrorInvalidValue;
    auto w=ga::workspace(*a,scratch);if(!w.total)return (int)cudaErrorInvalidValue;
    *q=w.q;*rope=w.rope;return 0;
}
#endif
