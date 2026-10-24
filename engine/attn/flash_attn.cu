// GLM 5.3 Flash DSA attention: paged NoPE MLA over the 512-d latent + the kpool indexer (kpool.cuh). Replaces attn.cu in
// the G6_MODEL_FLASH build (runtime/build_m4.sh) behind the same C ABI (attn.h, bind.h, replay.h), so the rank driver
// binds, runs and replays it as the GLM attention.
// Per row r at position pos (decode or verify rows; rows of one call may hold several positions of one sequence):
//   append   latent[pos] = RMSNorm(kv_a, kv_a_norm) (eps 1e-5) as the kvfmt.h row; index raw slot; pool key at pos%4==3
//   select   indices[r] = kpool selection (kpool.cuh), or a copy of reuse_indices
//   attend   q_abs[h] = q_b[h] (256) @ UK[h] (256 x 512); s_j = q_abs . latent[id_j] / 16; p = softmax(s);
//            ctx = sum_j p_j latent[id_j]; output[h] = UV[h] (256 x 512) @ ctx, FP16 [rows][heads][256]
// Not in the Flash build: context-parallel decode (the cp entry points return cudaErrorNotSupported), the QO pair
// prefill and the prefill traces. g6_attn_bind_prefill_run runs the decode path over the rows (same results, query tiles).
#include "attn.h"
#include "bind.h"
#include "replay.h"
#include "prefill.h"
#include "flash_pool_owner.h"
#include "prefill-qo-bind.h"
#include "../g6nr.h"
#include "../model_geom.h"
#include "kvfmt.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <mma.h>
#include <atomic>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <algorithm>
#if !G6_MODEL_FLASH
#error "flash_attn.cu is the G6_MODEL_FLASH attention; the GLM build compiles attn.cu"
#endif
extern "C" { void (*g6_attn_mark)(int stage,void* stream)=nullptr; }
#define AMARK(k) do{if(g6_attn_mark)g6_attn_mark((k),(void*)stream);}while(0)
namespace fa {
constexpr int TOP=g6kv::SEL, PART=TOP/256, LAT=geom::KV_LORA, NOPE=geom::QK_NOPE, VD=geom::V_HEAD;
constexpr int IHEADS=g6kv::IDX_HEADS, IDIM=g6kv::IDX_DIM, KP=g6kv::KPOOL;
constexpr int HC=16;   // heads per CTA in mla_scores / context_parts
static_assert(NOPE==256&&VD==256&&LAT==512&&TOP%256==0&&geom::QK_ROPE==0,"Flash MLA geometry");
static_assert(IDIM==128&&IHEADS==32&&KP==4&&g6kv::PAGE%KP==0,"kpool geometry");
using half=__half; using bf16=__nv_bfloat16;
struct ReplayInput { G6Attn fixed; const G6AttnStepCell* cell; };
struct DeviceAttn : G6Attn { int live_context; bool active; };
__device__ __forceinline__ DeviceAttn resolve(G6Attn fixed){
    DeviceAttn a;static_cast<G6Attn&>(a)=fixed;a.live_context=fixed.context;a.active=true;return a;
}
__device__ __forceinline__ DeviceAttn resolve(ReplayInput input){
    DeviceAttn a=resolve(input.fixed);
    a.live_context=input.cell->context;a.page_count=input.cell->page_count;
    a.table_size=input.cell->table_size;a.reuse_rows=input.cell->reuse_rows;
    a.active=input.cell->valid!=0;return a;
}
#ifdef DEC_S2
// batched step (S2): the row's own session table and bounds (replay.h G6AttnRow)
__device__ __forceinline__ void for_row(DeviceAttn&,const G6Attn&,int){}
__device__ __forceinline__ void for_row(DeviceAttn& a,const ReplayInput& in,int r){
    const G6AttnRow* rows=in.cell->rows;if(!rows)return;
    const G6AttnRow x=rows[r];
    a.block_table=x.table;a.live_context=x.context;a.table_size=x.table_size;a.page_count=x.page_count;
}
#define FOR_ROW(a,in,r) fa::for_row(a,in,r)
#else
#define FOR_ROW(a,in,r)
#endif
// the GLM validate_replay without the CP terms (same error bits: 32 bounds, 1 position, 2 page, 8 reuse row)
__global__ void validate_replay(G6Attn fixed,G6AttnStepCell* cell){
    const int t=threadIdx.x;
    if(t==0)*fixed.error=0;
    __syncthreads();
    const bool bounds=cell->context>0&&cell->context<=fixed.context&&
        cell->page_count>0&&cell->page_count<=fixed.page_count&&
        cell->table_size>0&&cell->table_size<=fixed.table_size&&
        ((int64_t)cell->context+fixed.page_size-1)/fixed.page_size<=cell->table_size&&
        (fixed.reuse_indices?(cell->reuse_rows>0&&cell->reuse_rows<=fixed.reuse_rows&&
         (fixed.reuse_row_map||cell->reuse_rows>=fixed.rows)):cell->reuse_rows==0);
    if(!bounds){if(t==0)*fixed.error=32;}
    else for(int row=t;row<fixed.rows;row+=blockDim.x){
        const int pos=fixed.positions[row];
        int ctx=cell->context,pages=cell->page_count;const int* table=fixed.block_table;
#ifdef DEC_S2
        int sess=0;
        if(cell->rows){
            const G6AttnRow x=cell->rows[row];
            const bool ok=x.table&&x.context>0&&x.context<=fixed.context&&x.page_count>0&&x.page_count<=fixed.page_count&&
                x.table_size>0&&x.table_size<=fixed.table_size&&((int64_t)x.context+fixed.page_size-1)/fixed.page_size<=x.table_size;
            if(!ok){atomicOr(fixed.error,32u);continue;}
            ctx=x.context;pages=x.page_count;sess=x.sess;table=x.table;
        }
#endif
        if(pos<0||pos>=ctx)atomicOr(fixed.error,1u);
        else{const int pg=table[pos/fixed.page_size];if(pg<0||pg>=pages)atomicOr(fixed.error,2u);}
#ifdef DEC_S2
        if(cell->rows){for(int j=0;j<row;j++)if(pos==fixed.positions[j]&&cell->rows[j].sess==sess)atomicOr(fixed.error,1u);}
        else
#endif
        for(int j=0;j<row;j++)if(pos==fixed.positions[j])atomicOr(fixed.error,1u);
        if(fixed.reuse_indices&&fixed.reuse_row_map){
            const int src=fixed.reuse_row_map[row];
            if(src<0||src>=cell->reuse_rows)atomicOr(fixed.error,8u);
        }
    }
    __syncthreads();if(t==0)cell->valid=*fixed.error==0;
}
__device__ __forceinline__ float bfloat(const void* p,int64_t i){return __bfloat162float(((const bf16*)p)[i]);}
__device__ __forceinline__ float hfloat(const void* p,int64_t i){return __half2float(((const half*)p)[i]);}
__device__ __forceinline__ float e4(uint8_t x){__nv_fp8_e4m3 a;a.__x=x;return (float)a;}
__device__ __forceinline__ uint8_t q4(float x){return __nv_fp8_e4m3(x).__x;}
__device__ __forceinline__ void error(const G6Attn& a,unsigned code){atomicOr(a.error,code);}
__device__ __forceinline__ float warp_sum(float x){for(int o=16;o;o>>=1)x+=__shfl_xor_sync(0xffffffff,x,o);return x;}
__device__ __forceinline__ float warp_max(float x){for(int o=16;o;o>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,o));return x;}
__device__ int page(const DeviceAttn& a,int pos){
    if(pos<0||pos>=a.live_context||pos/a.page_size>=a.table_size){error(a,1);return -1;}
    const int p=a.block_table[pos/a.page_size];if(p<0||p>=a.page_count){error(a,2);return -1;}return p;
}
__device__ __forceinline__ const uint8_t* kv_at(const DeviceAttn& a,int pos){
    const int p=page(a,pos);return p<0?nullptr:(const uint8_t*)a.mla_cache+((int64_t)p*a.page_size+pos%a.page_size)*g6kv::ROW;
}
// latent element d of a cached row; lat4: elements d..d+3 (d % 4 == 0)
#if G6_FLASH_KV_FP8
__device__ __forceinline__ float latent(const uint8_t* v,int d){return e4(__ldg(v+d))*__ldg(((const float*)(v+g6kv::SCALE_OFF))+(d>>7));}
__device__ __forceinline__ void lat4(const uint8_t* v,int d,float* x){
    const uint32_t c=__ldg((const uint32_t*)(v+d));const float s=__ldg(((const float*)(v+g6kv::SCALE_OFF))+(d>>7));
    #pragma unroll
    for(int i=0;i<4;i++)x[i]=e4((c>>(8*i))&255)*s;
}
#else
__device__ __forceinline__ float latent(const uint8_t* v,int d){return __bfloat162float(__ldg(((const bf16*)v)+d));}
__device__ __forceinline__ void lat4(const uint8_t* v,int d,float* x){
    const uint2 c=__ldg((const uint2*)(v+2*d));const bf16* b=(const bf16*)&c;
    #pragma unroll
    for(int i=0;i<4;i++)x[i]=__bfloat162float(b[i]);
}
#endif
#include "kpool.cuh"

// 256 threads: dims t and t + 256
template<class Input>
__global__ void append_kv(Input input){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,pos=a.positions[row],w=t>>5,lane=t&31;FOR_ROW(a,input,row);
    const int pg=page(a,pos);if(pg<0)return;
    __shared__ float red[8][2];
    const float x0=hfloat(a.kv_a,(int64_t)row*LAT+t),x1=hfloat(a.kv_a,(int64_t)row*LAT+t+256);
    const float ss=warp_sum(fmaf(x0,x0,x1*x1));if(lane==0)red[w][0]=ss;
    __syncthreads();
    float total=0;for(int j=0;j<8;j++)total+=red[j][0];
    const float inv=rsqrtf(total/LAT+1e-5f);
    const float v0=x0*inv*bfloat(a.kv_norm,t),v1=x1*inv*bfloat(a.kv_norm,t+256);
    if(!isfinite(v0)||!isfinite(v1))error(a,4);
    uint8_t* dst=(uint8_t*)a.mla_cache+((int64_t)pg*a.page_size+pos%a.page_size)*g6kv::ROW;
#if G6_FLASH_KV_FP8
    // E4M3 per 128 dims, scale amax / 448: warps 0-3 hold dims 0-127 (v0) and 256-383 (v1), warps 4-7 the others
    __syncthreads();
    const float m0=warp_max(fabsf(v0)),m1=warp_max(fabsf(v1));if(lane==0){red[w][0]=m0;red[w][1]=m1;}
    __syncthreads();
    const int g=w>>2;   // 0 or 1
    const float a0=fmaxf(fmaxf(red[4*g][0],red[4*g+1][0]),fmaxf(red[4*g+2][0],red[4*g+3][0]));
    const float a1=fmaxf(fmaxf(red[4*g][1],red[4*g+1][1]),fmaxf(red[4*g+2][1],red[4*g+3][1]));
    const float s0=fmaxf(a0/448.f,FLT_MIN),s1=fmaxf(a1/448.f,FLT_MIN);
    dst[t]=q4(v0/s0);dst[t+256]=q4(v1/s1);
    if((t&127)==0){((float*)(dst+g6kv::SCALE_OFF))[g]=s0;((float*)(dst+g6kv::SCALE_OFF))[2+g]=s1;}
#else
    ((bf16*)dst)[t]=__float2bfloat16_rn(v0);((bf16*)dst)[t+256]=__float2bfloat16_rn(v1);
#endif
}
template<class Input>
__global__ void reuse(Input input,int start){
    auto a=resolve(input);if(!a.active)return;const int r=start+blockIdx.x;const int src=a.reuse_row_map?a.reuse_row_map[r]:r;
    if(src<0||src>=a.reuse_rows){error(a,8);for(int j=threadIdx.x;j<TOP;j+=256)a.indices[(int64_t)r*TOP+j]=-1;return;}
    for(int j=threadIdx.x;j<TOP;j+=256)a.indices[(int64_t)r*TOP+j]=a.reuse_indices[(int64_t)src*TOP+j];
}
// q[row][h][l] = sum_n q_b[r][h][n] UK[h][n][l]. grid (heads, LAT / 64), 64 threads: l = blockIdx.y * 64 + t
// (128B cache line per CTA per k, 128 CTAs on TP2); rows in chunks of 8 sharing each UK load
template<class Input>
__global__ void absorb(Input input,int start,int n,float* q){
    auto a=resolve(input);if(!a.active)return;
    const int h=blockIdx.x,t=threadIdx.x,l=blockIdx.y*64+t;
    const bf16* __restrict__ uk=(const bf16*)a.uk+(int64_t)h*NOPE*LAT+l;
    __shared__ float qb[8][NOPE];
    for(int c=0;c<n;c+=8){const int m=min(8,n-c);
        if(c)__syncthreads();
        for(int rr=0;rr<m;rr++){
            const half* __restrict__ qbp=(const half*)a.q_b+((int64_t)(start+c+rr)*a.heads+h)*NOPE;
            #pragma unroll
            for(int j=0;j<4;j++)qb[rr][t+j*64]=__half2float(qbp[t+j*64]);
        }
        __syncthreads();
        if(m==1){
            float acc0=0.f;
            #pragma unroll 16
            for(int k=0;k<NOPE;k++){
                const float u=__bfloat162float(__ldg(uk+(int64_t)k*LAT));
                acc0=fmaf(qb[0][k],u,acc0);
            }
            q[((int64_t)c*a.heads+h)*LAT+l]=acc0;
        }else if(m==4){
            float acc0=0.f,acc1=0.f,acc2=0.f,acc3=0.f;
            #pragma unroll 8
            for(int k=0;k<NOPE;k++){
                const float u=__bfloat162float(__ldg(uk+(int64_t)k*LAT));
                acc0=fmaf(qb[0][k],u,acc0);
                acc1=fmaf(qb[1][k],u,acc1);
                acc2=fmaf(qb[2][k],u,acc2);
                acc3=fmaf(qb[3][k],u,acc3);
            }
            q[((int64_t)(c+0)*a.heads+h)*LAT+l]=acc0;
            q[((int64_t)(c+1)*a.heads+h)*LAT+l]=acc1;
            q[((int64_t)(c+2)*a.heads+h)*LAT+l]=acc2;
            q[((int64_t)(c+3)*a.heads+h)*LAT+l]=acc3;
        }else{
            float acc[8];
            #pragma unroll
            for(int rr=0;rr<8;rr++)acc[rr]=0.f;
            #pragma unroll 8
            for(int k=0;k<NOPE;k++){const float u=__bfloat162float(__ldg(uk+(int64_t)k*LAT));
                #pragma unroll
                for(int rr=0;rr<8;rr++)acc[rr]=fmaf(qb[rr][k],u,acc[rr]);}
            for(int rr=0;rr<m;rr++)q[((int64_t)(c+rr)*a.heads+h)*LAT+l]=acc[rr];
        }
    }
}
// grid (TOP / 32, ceil(heads / HC), rows), 256 threads: warp w scores slots 4w..4w+3 of the CTA's 32 for HC heads; lane
// holds dims 4 lane + 128 k, k = 0..3
template<class Input>
__global__ void mla_scores(Input input,int start,const float* q,float* score){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.z,r=start+row,h0=blockIdx.y*HC,nh=min(HC,a.heads-h0),t=threadIdx.x,w=t>>5,lane=t&31;FOR_ROW(a,input,r);
    __shared__ const uint8_t* sh_v[32];
    __shared__ unsigned sh_mask;
    if(w==0){
        const int j=blockIdx.x*32+lane,id=a.indices[(int64_t)r*TOP+j];
        const uint8_t* v=(id>=0&&id<=a.positions[r])?kv_at(a,id):nullptr;
        sh_v[lane]=v;
        const unsigned m=__ballot_sync(0xffffffff,v!=nullptr);
        if(lane==0)sh_mask=m;
    }
    __syncthreads();
    if(!sh_mask){
        for(int i=t;i<nh*32;i+=256){
            const int hh=i>>5,s=i&31;
            score[((int64_t)row*a.heads+h0+hh)*TOP+blockIdx.x*32+s]=-INFINITY;
        }
        return;
    }
    __shared__ __align__(16) float qs[HC][LAT];
    for(int i=t;i<nh*LAT;i+=256)qs[i/LAT][i%LAT]=__ldg(q+((int64_t)row*a.heads+h0+i/LAT)*LAT+i%LAT);
    __syncthreads();
    for(int s=0;s<4;s++){
        const int slot=w*4+s,j=blockIdx.x*32+slot;
        const uint8_t* v=sh_v[slot];
        if(!v){
            if(lane<nh)score[((int64_t)row*a.heads+h0+lane)*TOP+j]=-INFINITY;
            continue;
        }
        float x[16];
        #pragma unroll
        for(int k=0;k<4;k++)lat4(v,4*lane+128*k,x+4*k);
        for(int hh=0;hh<nh;hh++){
            float sum=0;
            #pragma unroll
            for(int k=0;k<4;k++){const float4 qq=*(const float4*)&qs[hh][4*lane+128*k];
                sum=fmaf(qq.x,x[4*k],sum);sum=fmaf(qq.y,x[4*k+1],sum);sum=fmaf(qq.z,x[4*k+2],sum);sum=fmaf(qq.w,x[4*k+3],sum);}
            sum=warp_sum(sum);
            if(lane==0)score[((int64_t)row*a.heads+h0+hh)*TOP+j]=sum*(1.f/16);
        }
    }
}
template<class Input>
__global__ void normalize(Input input,float* scores){
    auto a=resolve(input);if(!a.active)return;
    const int i=blockIdx.x,t=threadIdx.x;float* p=scores+(int64_t)i*TOP;__shared__ float tmp[8],mx,den;
    float m=-INFINITY;for(int j=t;j<TOP;j+=256)m=fmaxf(m,p[j]);m=warp_max(m);if((t&31)==0)tmp[t/32]=m;__syncthreads();
    if(t==0){mx=-INFINITY;for(int j=0;j<8;j++)mx=fmaxf(mx,tmp[j]);}__syncthreads();
    float s=0;for(int j=t;j<TOP;j+=256){const float v=isfinite(mx)?expf(p[j]-mx):0;p[j]=v;s+=v;}s=warp_sum(s);if((t&31)==0)tmp[t/32]=s;__syncthreads();
    if(t==0){den=0;for(int j=0;j<8;j++)den+=tmp[j];if(den==0)error(a,16);}__syncthreads();
    for(int j=t;j<TOP;j+=256)p[j]=den>0?p[j]/den:0;
}
// grid (PART, ceil(heads / HC), rows), 256 threads: dims t, t + 256 over the part's 256 slots for HC heads; every
// cached row is read once per HC heads
template<class Input>
__global__ void context_parts(Input input,int start,const float* probs,float* parts){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.z,r=start+row,h0=blockIdx.y*HC,nh=min(HC,a.heads-h0),part=blockIdx.x,t=threadIdx.x,w=t>>5,lane=t&31;FOR_ROW(a,input,r);
    __shared__ uintptr_t rows[256];
    __shared__ float pr[HC][256];
    __shared__ unsigned warp_mask[8];
    const int pos=a.positions[r],slot=part*256+t,id=a.indices[(int64_t)r*TOP+slot];
    const uint8_t* v_ptr=nullptr;
    if(id>=0&&id<=pos){
        bool any=false;
        #pragma unroll
        for(int hh=0;hh<HC;hh++){const float p=hh<nh?__ldg(probs+((int64_t)row*a.heads+h0+hh)*TOP+slot):0.f;pr[hh][t]=p;any|=p!=0.f;}
        if(any)v_ptr=kv_at(a,id);
    }
    rows[t]=(uintptr_t)v_ptr;
    const unsigned m=__ballot_sync(0xffffffff,v_ptr!=nullptr);
    if(lane==0)warp_mask[w]=m;
    __syncthreads();
    if((warp_mask[0]|warp_mask[1]|warp_mask[2]|warp_mask[3]|warp_mask[4]|warp_mask[5]|warp_mask[6]|warp_mask[7])==0u){
        for(int hh=0;hh<nh;hh++){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*PART+part)*LAT;dst[t]=0.f;dst[t+256]=0.f;}
        return;
    }
    float lo[HC],hi[HC];
    #pragma unroll
    for(int hh=0;hh<HC;hh++){lo[hh]=0.f;hi[hh]=0.f;}
    for(int ww=0;ww<8;ww++){
        unsigned wm=warp_mask[ww];
        if(!wm)continue;
        if(wm==0xffffffffu){
            #pragma unroll 4
            for(int s=0;s<32;s++){
                const int j=ww*32+s;
                const uint8_t* v=(const uint8_t*)rows[j];
                const float l0=latent(v,t),l1=latent(v,t+256);
                #pragma unroll
                for(int hh=0;hh<HC;hh++){const float p=pr[hh][j];lo[hh]=fmaf(p,l0,lo[hh]);hi[hh]=fmaf(p,l1,hi[hh]);}
            }
        }else{
            while(wm){
                const int j=ww*32+(__ffs(wm)-1);
                wm&=wm-1;
                const uint8_t* v=(const uint8_t*)rows[j];
                const float l0=latent(v,t),l1=latent(v,t+256);
                #pragma unroll
                for(int hh=0;hh<HC;hh++){const float p=pr[hh][j];lo[hh]=fmaf(p,l0,lo[hh]);hi[hh]=fmaf(p,l1,hi[hh]);}
            }
        }
    }
    for(int hh=0;hh<nh;hh++){float* dst=parts+(((int64_t)row*a.heads+h0+hh)*PART+part)*LAT;dst[t]=lo[hh];dst[t+256]=hi[hh];}
}
template<class Input>
__global__ void merge_parts(Input input,const float* parts,float* out){
    auto a=resolve(input);if(!a.active)return;
    const int rh=blockIdx.x,t=threadIdx.x;
    #pragma unroll
    for(int d=t;d<LAT;d+=256){
        float s=0;
        #pragma unroll
        for(int p=0;p<PART;p++)s+=__ldg(parts+((int64_t)rh*PART+p)*LAT+d);
        out[(int64_t)rh*LAT+d]=s;
    }
}
// output[r][h][d] = UV[h][d] . ctx[row][h]; grid (VD / 8, heads, ceil(rows / 4)), warp per d, lane over the 512 latent dims
template<class Input>
__global__ void value_project(Input input,int start,int n,const float* context){
    auto a=resolve(input);if(!a.active)return;
    const int c=blockIdx.z*4,m=min(4,n-c),h=blockIdx.y,t=threadIdx.x,d=blockIdx.x*8+(t>>5),lane=t&31;
    const bf16* __restrict__ uvp=(const bf16*)a.uv+((int64_t)h*VD+d)*LAT+lane;
    float uv_reg[16];
    #pragma unroll
    for(int i=0;i<16;i++)uv_reg[i]=__bfloat162float(__ldg(uvp+i*32));
    __shared__ float sh_ctx[4][LAT];
    for(int rr=0;rr<m;rr++){
        const float* __restrict__ cp=context+((int64_t)(c+rr)*a.heads+h)*LAT;
        sh_ctx[rr][t]=__ldg(cp+t);
        sh_ctx[rr][t+256]=__ldg(cp+t+256);
    }
    __syncthreads();
    if(m==1){
        float sum0=0.f;
        #pragma unroll
        for(int i=0;i<16;i++)sum0=fmaf(sh_ctx[0][lane+i*32],uv_reg[i],sum0);
        sum0=warp_sum(sum0);
        if(lane==0)((half*)a.output)[((int64_t)(start+c)*a.heads+h)*VD+d]=__float2half_rn(sum0);
    }else{
        float sum[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int i=0;i<16;i++){
            const float u=uv_reg[i];
            #pragma unroll
            for(int rr=0;rr<4;rr++)if(rr<m)sum[rr]=fmaf(sh_ctx[rr][lane+i*32],u,sum[rr]);
        }
        #pragma unroll
        for(int rr=0;rr<4;rr++)if(rr<m){
            const float s=warp_sum(sum[rr]);
            if(lane==0)((half*)a.output)[((int64_t)(start+c+rr)*a.heads+h)*VD+d]=__float2half_rn(s);
        }
    }
}

struct Work { float *q,*ps,*prob,*parts,*ctx; int pstride; size_t total; };
static int pool_stride(int context){return ((context+KP-1)/KP+31)/32*32;}
static bool valid(const G6Attn& a){
#if defined(G6_CP) && G6_CP
    if(a.cp_nr)return false;
#endif
    return a.rows>0&&(a.heads==g6nr::HMIN||a.heads==g6nr::HMAX)&&a.context>0&&a.page_size==g6kv::PAGE&&a.page_count>0&&
        a.table_size>0&&a.query_tile>0&&a.query_tile<=64&&a.context<=INT_MAX/a.query_tile;
}
static Work workspace(const G6Attn& a,void* base){
    Work w{};size_t n=0;const size_t R=std::min(a.rows,a.query_tile),H=a.heads;
    auto alloc=[&](size_t bytes){size_t old=n;n=(n+bytes+255)&~size_t(255);return (void*)((uintptr_t)base+old);};
    w.pstride=pool_stride(a.context);
    w.q=(float*)alloc(R*H*LAT*4);w.ps=(float*)alloc(R*w.pstride*4);w.prob=(float*)alloc(R*H*TOP*4);
    w.parts=(float*)alloc(R*H*PART*LAT*4);w.ctx=(float*)alloc(R*H*LAT*4);w.total=n;return w;
}
template<class Input>
static void append(const G6Attn& a,Input input,cudaStream_t stream){
    append_kv<<<a.rows,256,0,stream>>>(input);
    if(!a.reuse_indices){kp_append<<<a.rows,2*IDIM,0,stream>>>(input);kp_pool<<<a.rows,IDIM,0,stream>>>(input);}
}
template<class Input>
static void make_index(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t stream){
    kp_scores<<<dim3(w.pstride/32,n),256,0,stream>>>(input,start,w.ps,w.pstride);
    kp_select<<<n,SELT,0,stream>>>(input,start,w.ps,w.pstride);
}
template<class Input>
static cudaError_t run_body(const G6Attn& a,Input input,Work& w,cudaStream_t stream){
    const int hch=(a.heads+HC-1)/HC;
    for(int start=0;start<a.rows;start+=a.query_tile){const int n=std::min(a.rows-start,a.query_tile);
        if(a.reuse_indices){AMARK(1);reuse<<<n,256,0,stream>>>(input,start);}
        else{make_index(a,input,w,start,n,stream);AMARK(1);}
        AMARK(2);
        absorb<<<dim3(a.heads,LAT/64),64,0,stream>>>(input,start,n,w.q);
        AMARK(3);
        mla_scores<<<dim3(TOP/32,hch,n),256,0,stream>>>(input,start,w.q,w.prob);
        AMARK(4);
        normalize<<<n*a.heads,256,0,stream>>>(input,w.prob);
        AMARK(5);
        context_parts<<<dim3(PART,hch,n),256,0,stream>>>(input,start,w.prob,w.parts);
        AMARK(6);
        merge_parts<<<n*a.heads,256,0,stream>>>(input,w.parts,w.ctx);
        AMARK(7);
        value_project<<<dim3(VD/8,a.heads,(n+3)/4),256,0,stream>>>(input,start,n,w.ctx);
        AMARK(8);
    }
    return cudaGetLastError();
}
static bool append_ok(const G6Attn& a){
    return a.kv_a&&a.kv_norm&&a.mla_cache&&a.positions&&a.block_table&&a.error&&
        (a.reuse_indices||(a.index_k&&a.index_gate&&a.index_ape&&a.index_norm&&a.index_bias&&a.index_cache));
}
static bool run_ok(const G6Attn& a){
    return a.q_b&&a.uk&&a.uv&&a.indices&&a.output&&(a.reuse_indices||(a.index_q&&a.index_weight))&&
        !(a.reuse_row_map&&a.reuse_indices==a.indices);
}
}   // namespace fa

constexpr int NOTSUP=(int)cudaErrorNotSupported, INVAL=(int)cudaErrorInvalidValue;
extern "C" size_t g6_attn_scratch_bytes(const G6Attn* a){if(!a||!fa::valid(*a))return 0;return fa::workspace(*a,nullptr).total;}
extern "C" int g6_attn_append(const G6Attn* a,void* s){
    if(!a||!fa::valid(*a)||!fa::append_ok(*a))return INVAL;
    fa::append(*a,*a,(cudaStream_t)s);return (int)cudaGetLastError();
}
// tests: the pool scores (kpool.cuh kp_scores), FP32 [rows][pool_stride(context)] (pools not visible to a row: untouched)
extern "C" int g6_attn_index_scores(const G6Attn* a,void* scratch,size_t bytes,float* scores,void* s){
    if(!a||!fa::valid(*a)||!scratch||!scores||a->reuse_indices||!a->index_q||!a->index_weight)return INVAL;
    auto w=fa::workspace(*a,scratch);if(!w.total||bytes<w.total)return INVAL;
    fa::kp_scores<<<dim3(w.pstride/32,a->rows),256,0,(cudaStream_t)s>>>(*a,0,scores,w.pstride);
    return (int)cudaGetLastError();
}
extern "C" int g6_attn_run(const G6Attn* a,void* scratch,size_t bytes,void* s){
    if(!a||!fa::valid(*a)||!scratch||!fa::append_ok(*a)||!fa::run_ok(*a))return INVAL;
    auto w=fa::workspace(*a,scratch);if(!w.total||bytes<w.total)return INVAL;
    cudaStream_t stream=(cudaStream_t)s;
    fa::append(*a,*a,stream);
    return (int)fa::run_body(*a,*a,w,stream);
}
extern "C" int g6_attn_replay_run(const G6AttnReplay* r,void* s){
    if(!r||!r->cell)return INVAL;
    const G6Attn& a=r->fixed;
    if(!fa::valid(a)||!r->scratch||!fa::append_ok(a)||!fa::run_ok(a))return INVAL;
    auto w=fa::workspace(a,r->scratch);
    if(!w.total||r->scratch_bytes<w.total)return INVAL;
    cudaStream_t stream=(cudaStream_t)s;
    fa::validate_replay<<<1,256,0,stream>>>(a,r->cell);
    fa::ReplayInput input{a,r->cell};
    fa::append(a,input,stream);
    AMARK(0);
    return (int)fa::run_body(a,input,w,stream);
}
extern "C" int g6_attn_rope_init(int,void*){return 0;}   // no RoPE in Flash

#if defined(G6_CP) && G6_CP
extern "C" int g6_attn_cp_front(const G6Attn*,const G6AttnStepCell*,void*,size_t,void*){return NOTSUP;}
extern "C" int g6_attn_cp_index(const G6Attn*,const G6AttnStepCell*,void*,size_t,uint64_t*,void*){return NOTSUP;}
extern "C" int g6_attn_cp_merge_index(const G6Attn*,const G6AttnStepCell*,const uint64_t*,void*){return NOTSUP;}
extern "C" int g6_attn_cp_group(const G6Attn*,const G6AttnStepCell*,void*,size_t,const float*,const float*,float*,void*){return NOTSUP;}
extern "C" int g6_attn_cp_finish(const G6Attn*,const G6AttnStepCell*,void*,size_t,const float*,void*){return NOTSUP;}
extern "C" int g6_attn_cp_qabs(const G6Attn*,void*,const float**,const float**){return NOTSUP;}
// G6_CP_ONECHAIN (#111): Flash has no CP, so the one-call head groups are never ok
extern "C" int g6_attn_cp_groups_ok(int){return 0;}
extern "C" size_t g6_attn_cp_groups_bytes(int,int){return 0;}
extern "C" int g6_attn_cp_groups(const G6Attn*,const G6AttnStepCell*,void*,size_t,const float*,const float*,float*,int,int64_t,void*){return NOTSUP;}
#endif

// ---- prefill: Tensor-Core WMMA attention (absorb_tc, sparse_qk_pf, sparse_pv_pf, value_project_tc) + multi-row kpool
// scorer (kp_scores_pf). Shared workspace across layers via g6_attn_bind_prefill_create_shared; also supports steps with
// rows > b->max_rows (when b was bound at decode MX) via the shared workspace's indices / output buffers.
namespace fa_pf {
using namespace fa;

__device__ __forceinline__ int4 lat8_half(const uint8_t* v,int d){
    half h[8];
#if G6_FLASH_KV_FP8
    float x[8];
    lat4(v,d,x);lat4(v,d+4,x+4);
    #pragma unroll
    for(int i=0;i<8;i++)h[i]=__float2half_rn(x[i]);
#else
    const uint4 c=__ldg((const uint4*)(v+2*d));const bf16* b=(const bf16*)&c;
    #pragma unroll
    for(int i=0;i<8;i++)h[i]=__float2half_rn(__bfloat162float(b[i]));
#endif
    return *(const int4*)h;
}

// Multi-row kpool scorer: 16 query rows (4 passes of 4) share each 32-pool tile load (ks[32][128]).
// 0-ULP bit-identical to kp_scores: exact same d=0..127 FMA order, k=0..3 head loop, and xor-1/2/4 warp reduction.
template<class Input>
__global__ void __launch_bounds__(256) kp_scores_pf(Input input,int start,int n,float* scores,int pstride){
    auto a=resolve(input);if(!a.active)return;
    const int row0=blockIdx.y*16,m16=min(16,n-row0),p0=blockIdx.x*32,t=threadIdx.x;
    int max_nvis=0;
    for(int rr=0;rr<m16;rr++){
        DeviceAttn ar=a;FOR_ROW(ar,input,start+row0+rr);
        const int nv=kp_visible(ar,start+row0+rr);
        if(nv>g6kv::TOP_POOLS)max_nvis=max(max_nvis,nv);
    }
    if(p0>=max_nvis)return;
    const int npools=min(32,max_nvis-p0);
    __shared__ __align__(16) float ks[32][IDIM+4];
    __shared__ __align__(16) half qs[4][IHEADS][IDIM+8];
    __shared__ float ws[4][IHEADS];
    __shared__ int nvis_s[4];
    for(int i=t;i<npools*IDIM;i+=256){
        const int pp=i/IDIM,d=i%IDIM,p=p0+pp;float v=0.f;
        const uint8_t* b=ix_page(a,p*KP);
        if(b)v=__ldg(((const float*)(b+g6kv::IDX_POOL_OFF))+((p%g6kv::PAGE_POOLS)*IDIM+d));
        ks[pp][d]=v;
    }
    const int pp=t>>3,hl=t&7;
    for(int pass=0;pass<m16;pass+=4){
        const int mp=min(4,m16-pass);
        __syncthreads();
        if(t<4){
            int nv=0;
            if(t<mp){
                DeviceAttn ar=a;FOR_ROW(ar,input,start+row0+pass+t);
                nv=kp_visible(ar,start+row0+pass+t);
                if(nv<=g6kv::TOP_POOLS)nv=0;
            }
            nvis_s[t]=nv;
        }
        for(int i=t;i<mp*(IHEADS*(IDIM/8));i+=256){
            const int rr=i/(IHEADS*(IDIM/8)),rem=i%(IHEADS*(IDIM/8)),h=rem/(IDIM/8),d8=(rem%(IDIM/8))*8;
            const int r=start+row0+pass+rr;
            *(uint4*)&qs[rr][h][d8]=__ldg((const uint4*)((const half*)a.index_q+(int64_t)r*IHEADS*IDIM+h*IDIM+d8));
        }
        if(t<mp*IHEADS){
            const int rr=t/IHEADS,h=t%IHEADS;
            ws[rr][h]=hfloat(a.index_weight,(int64_t)(start+row0+pass+rr)*IHEADS+h);
        }
        __syncthreads();
        const int pass_nvis=max(max(nvis_s[0],nvis_s[1]),max(nvis_s[2],nvis_s[3]));
        if(p0>=pass_nvis||(pp&~3)>=min(32,pass_nvis-p0))continue;
        float sum[4]={0.f,0.f,0.f,0.f};
        if(pp<npools){
            const bool v0=(p0+pp<nvis_s[0]),v1=(p0+pp<nvis_s[1]),v2=(p0+pp<nvis_s[2]),v3=(p0+pp<nvis_s[3]);
            float dot[4][4]={};
            #pragma unroll 8
            for(int d=0;d<IDIM;d+=4){
                const float4 kv=*(const float4*)&ks[pp][d];
                #pragma unroll
                for(int k=0;k<IHEADS/8;k++){
                    const int h=hl+8*k;
                    if(v0){
                        const uint2 qh=*(const uint2*)&qs[0][h][d];const half* qp=(const half*)&qh;
                        dot[0][k]=fmaf(__half2float(qp[0]),kv.x,dot[0][k]);dot[0][k]=fmaf(__half2float(qp[1]),kv.y,dot[0][k]);
                        dot[0][k]=fmaf(__half2float(qp[2]),kv.z,dot[0][k]);dot[0][k]=fmaf(__half2float(qp[3]),kv.w,dot[0][k]);
                    }
                    if(v1){
                        const uint2 qh=*(const uint2*)&qs[1][h][d];const half* qp=(const half*)&qh;
                        dot[1][k]=fmaf(__half2float(qp[0]),kv.x,dot[1][k]);dot[1][k]=fmaf(__half2float(qp[1]),kv.y,dot[1][k]);
                        dot[1][k]=fmaf(__half2float(qp[2]),kv.z,dot[1][k]);dot[1][k]=fmaf(__half2float(qp[3]),kv.w,dot[1][k]);
                    }
                    if(v2){
                        const uint2 qh=*(const uint2*)&qs[2][h][d];const half* qp=(const half*)&qh;
                        dot[2][k]=fmaf(__half2float(qp[0]),kv.x,dot[2][k]);dot[2][k]=fmaf(__half2float(qp[1]),kv.y,dot[2][k]);
                        dot[2][k]=fmaf(__half2float(qp[2]),kv.z,dot[2][k]);dot[2][k]=fmaf(__half2float(qp[3]),kv.w,dot[2][k]);
                    }
                    if(v3){
                        const uint2 qh=*(const uint2*)&qs[3][h][d];const half* qp=(const half*)&qh;
                        dot[3][k]=fmaf(__half2float(qp[0]),kv.x,dot[3][k]);dot[3][k]=fmaf(__half2float(qp[1]),kv.y,dot[3][k]);
                        dot[3][k]=fmaf(__half2float(qp[2]),kv.z,dot[3][k]);dot[3][k]=fmaf(__half2float(qp[3]),kv.w,dot[3][k]);
                    }
                }
            }
            #pragma unroll
            for(int k=0;k<IHEADS/8;k++){
                const int h=hl+8*k;
                sum[0]=fmaf(fmaxf(dot[0][k],0.f),ws[0][h],sum[0]);
                sum[1]=fmaf(fmaxf(dot[1][k],0.f),ws[1][h],sum[1]);
                sum[2]=fmaf(fmaxf(dot[2][k],0.f),ws[2][h],sum[2]);
                sum[3]=fmaf(fmaxf(dot[3][k],0.f),ws[3][h],sum[3]);
            }
        }
        #pragma unroll
        for(int rr=0;rr<4;rr++){
            if(rr>=mp)break;
            float s=sum[rr];
            s+=__shfl_xor_sync(0xffffffff,s,1);s+=__shfl_xor_sync(0xffffffff,s,2);s+=__shfl_xor_sync(0xffffffff,s,4);
            if(hl==0&&p0+pp<nvis_s[rr]){
                s=s*(1.f/64)+0.f;
                if(!isfinite(s)){error(a,4);s=-INFINITY;}
                scores[(int64_t)(row0+pass+rr)*pstride+p0+pp]=s;
            }
        }
    }
}

// Tensor-Core prefill indexer scores: 64 query rows per CTA share one load of 32 pooled keys.
// Each of the 4 warps (128 threads) holds the 32x128 pooled keys in WMMA B-fragments and processes
// one query row [IHEADS=32, IDIM=128] per iteration via wmma::mma_sync (16x16x16 FP16 -> FP32).
// TC_TERMS=1: single FP16 pooled key; TC_TERMS=2: 22-bit hi+lo FP16 pooled key pair.
// Grid: (pstride / 32, (n + 63) / 64), 128 threads (4 warps)
template<int TC_TERMS,class Input>
__global__ void __launch_bounds__(128) kp_scores_tc(Input input,int start,int n,float* scores,int pstride){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    const int p0=blockIdx.x*32,row0=blockIdx.y*64,t=threadIdx.x,w=t>>5,lane=t&31;
    const int m64=min(64,n-row0);
    if(m64<=0)return;
    const int last_r=start+row0+m64-1;
    DeviceAttn a_last=a;FOR_ROW(a_last,input,last_r);
    const int max_nvis=kp_visible(a_last,last_r);
    if(max_nvis<=g6kv::TOP_POOLS||p0>=max_nvis)return;
    const int npools=min(32,max_nvis-p0);
    constexpr int K_STRIDE=136,O_STRIDE=36;
    __shared__ __align__(32) half ks_hi[32][K_STRIDE];
    __shared__ __align__(32) half ks_lo[TC_TERMS>1?32:1][TC_TERMS>1?K_STRIDE:1];
    __shared__ __align__(32) float out_w[4][IHEADS][O_STRIDE];
    __shared__ float ws_w[4][IHEADS];
    for(int i=t;i<32*IDIM;i+=128){
        const int pp=i/IDIM,d=i%IDIM,p=p0+pp;
        float v=0.f;
        if(pp<npools){
            const uint8_t* b=ix_page(a,p*KP);
            if(b)v=__ldg(((const float*)(b+g6kv::IDX_POOL_OFF))+((p%g6kv::PAGE_POOLS)*IDIM+d));
        }
        const half hi=__float2half_rn(v);
        ks_hi[pp][d]=hi;
        if constexpr(TC_TERMS>1)ks_lo[pp][d]=__float2half_rn((v-__half2float(hi))*2048.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>bf_hi[IDIM/16][2];
    #pragma unroll
    for(int k=0;k<IDIM/16;k++){
        wmma::load_matrix_sync(bf_hi[k][0],&ks_hi[0][k*16],K_STRIDE);
        wmma::load_matrix_sync(bf_hi[k][1],&ks_hi[16][k*16],K_STRIDE);
    }
    for(int rr=w;rr<m64;rr+=4){
        const int r=start+row0+rr;
        DeviceAttn ar=a;FOR_ROW(ar,input,r);
        const int nvis=kp_visible(ar,r);
        if(nvis<=g6kv::TOP_POOLS||p0>=nvis)continue;
        ws_w[w][lane]=hfloat(a.index_weight,(int64_t)r*IHEADS+lane);
        const half* q_row=(const half*)a.index_q+(int64_t)r*(IHEADS*IDIM);
        wmma::fragment<wmma::accumulator,16,16,16,float>cf[2][2],cf_lo[2][2];
        #pragma unroll
        for(int hm=0;hm<2;hm++)
            #pragma unroll
            for(int pn=0;pn<2;pn++){
                wmma::fill_fragment(cf[hm][pn],0.f);
                if constexpr(TC_TERMS>1)wmma::fill_fragment(cf_lo[hm][pn],0.f);
            }
        #pragma unroll
        for(int k=0;k<IDIM/16;k++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>af0,af1;
            wmma::load_matrix_sync(af0,q_row+0*16*IDIM+k*16,IDIM);
            wmma::load_matrix_sync(af1,q_row+1*16*IDIM+k*16,IDIM);
            wmma::mma_sync(cf[0][0],af0,bf_hi[k][0],cf[0][0]);
            wmma::mma_sync(cf[0][1],af0,bf_hi[k][1],cf[0][1]);
            wmma::mma_sync(cf[1][0],af1,bf_hi[k][0],cf[1][0]);
            wmma::mma_sync(cf[1][1],af1,bf_hi[k][1],cf[1][1]);
            if constexpr(TC_TERMS>1){
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>bflo0,bflo1;
                wmma::load_matrix_sync(bflo0,&ks_lo[0][k*16],K_STRIDE);
                wmma::load_matrix_sync(bflo1,&ks_lo[16][k*16],K_STRIDE);
                wmma::mma_sync(cf_lo[0][0],af0,bflo0,cf_lo[0][0]);
                wmma::mma_sync(cf_lo[0][1],af0,bflo1,cf_lo[0][1]);
                wmma::mma_sync(cf_lo[1][0],af1,bflo0,cf_lo[1][0]);
                wmma::mma_sync(cf_lo[1][1],af1,bflo1,cf_lo[1][1]);
            }
        }
        if constexpr(TC_TERMS>1){
            #pragma unroll
            for(int hm=0;hm<2;hm++)
                #pragma unroll
                for(int pn=0;pn<2;pn++)
                    #pragma unroll
                    for(int e=0;e<cf[hm][pn].num_elements;e++)
                        cf[hm][pn].x[e]=fmaf(cf_lo[hm][pn].x[e],1.f/2048.f,cf[hm][pn].x[e]);
        }
        #pragma unroll
        for(int hm=0;hm<2;hm++)
            #pragma unroll
            for(int pn=0;pn<2;pn++)
                wmma::store_matrix_sync(&out_w[w][hm*16][pn*16],cf[hm][pn],O_STRIDE,wmma::mem_row_major);
        __syncwarp();
        if(p0+lane<nvis){
            float sum=0.f;
            #pragma unroll
            for(int h=0;h<IHEADS;h++)sum=fmaf(fmaxf(out_w[w][h][lane],0.f),ws_w[w][h],sum);
            float s=sum*(1.f/64.f)+0.f;
            if(!isfinite(s)){error(a,4);s=-INFINITY;}
            scores[(int64_t)(row0+rr)*pstride+p0+lane]=s;
        }
        __syncwarp();
    }
}

// Register-light exact alternative to kp_scores_tc: stream key fragments from shared memory and
// finish each 16-head half before the next. Every dot keeps its MMA accumulation order, and
// the head sum still visits heads 0..31. Half-sized output staging reduces shared memory too.
// G6_PF_KP_TC=3 selects this candidate; the default (=2) remains unchanged.
template<int TC_TERMS,class Input>
__global__ void __launch_bounds__(128) kp_scores_tc_stream(Input input,int start,int n,float* scores,int pstride){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    const int p0=blockIdx.x*32,row0=blockIdx.y*64,t=threadIdx.x,w=t>>5,lane=t&31;
    const int m64=min(64,n-row0);
    if(m64<=0)return;
    const int last_r=start+row0+m64-1;
    DeviceAttn a_last=a;FOR_ROW(a_last,input,last_r);
    const int max_nvis=kp_visible(a_last,last_r);
    if(max_nvis<=g6kv::TOP_POOLS||p0>=max_nvis)return;
    const int npools=min(32,max_nvis-p0);
    constexpr int K_STRIDE=136,O_STRIDE=36;
    __shared__ __align__(32) half ks_hi[32][K_STRIDE];
    __shared__ __align__(32) half ks_lo[TC_TERMS>1?32:1][TC_TERMS>1?K_STRIDE:1];
    __shared__ __align__(32) float out_w[4][16][O_STRIDE];
    __shared__ float ws_w[4][IHEADS];
    for(int i=t;i<32*IDIM;i+=128){
        const int pp=i/IDIM,d=i%IDIM,p=p0+pp;
        float v=0.f;
        if(pp<npools){
            const uint8_t* b=ix_page(a,p*KP);
            if(b)v=__ldg(((const float*)(b+g6kv::IDX_POOL_OFF))+((p%g6kv::PAGE_POOLS)*IDIM+d));
        }
        const half hi=__float2half_rn(v);
        ks_hi[pp][d]=hi;
        if constexpr(TC_TERMS>1)ks_lo[pp][d]=__float2half_rn((v-__half2float(hi))*2048.f);
    }
    __syncthreads();
    for(int rr=w;rr<m64;rr+=4){
        const int r=start+row0+rr;
        DeviceAttn ar=a;FOR_ROW(ar,input,r);
        const int nvis=kp_visible(ar,r);
        if(nvis<=g6kv::TOP_POOLS||p0>=nvis)continue;
        ws_w[w][lane]=hfloat(a.index_weight,(int64_t)r*IHEADS+lane);
        const half* q_row=(const half*)a.index_q+(int64_t)r*(IHEADS*IDIM);
        float sum=0.f;
        // One head half at a time; each output keeps the original MMA and head-sum order.
        #pragma unroll 1
        for(int hm=0;hm<2;hm++){
            wmma::fragment<wmma::accumulator,16,16,16,float> cf[2],cf_lo[2];
            #pragma unroll
            for(int pn=0;pn<2;pn++){
                wmma::fill_fragment(cf[pn],0.f);
                if constexpr(TC_TERMS>1)wmma::fill_fragment(cf_lo[pn],0.f);
            }
            #pragma unroll
            for(int k=0;k<IDIM/16;k++){
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
                wmma::load_matrix_sync(af,q_row+hm*16*IDIM+k*16,IDIM);
                #pragma unroll
                for(int pn=0;pn<2;pn++){
                    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
                    wmma::load_matrix_sync(bf,&ks_hi[pn*16][k*16],K_STRIDE);
                    wmma::mma_sync(cf[pn],af,bf,cf[pn]);
                    if constexpr(TC_TERMS>1){
                        wmma::load_matrix_sync(bf,&ks_lo[pn*16][k*16],K_STRIDE);
                        wmma::mma_sync(cf_lo[pn],af,bf,cf_lo[pn]);
                    }
                }
            }
            #pragma unroll
            for(int pn=0;pn<2;pn++){
                if constexpr(TC_TERMS>1){
                    #pragma unroll
                    for(int e=0;e<cf[pn].num_elements;e++)
                        cf[pn].x[e]=fmaf(cf_lo[pn].x[e],1.f/2048.f,cf[pn].x[e]);
                }
                wmma::store_matrix_sync(&out_w[w][0][pn*16],cf[pn],O_STRIDE,wmma::mem_row_major);
            }
            __syncwarp();
            if(p0+lane<nvis){
                #pragma unroll
                for(int h=0;h<16;h++)sum=fmaf(fmaxf(out_w[w][h][lane],0.f),ws_w[w][hm*16+h],sum);
            }
            __syncwarp();
        }
        if(p0+lane<nvis){
            float s=sum*(1.f/64.f)+0.f;
            if(!isfinite(s)){error(a,4);s=-INFINITY;}
            scores[(int64_t)(row0+rr)*pstride+p0+lane]=s;
        }
        __syncwarp();
    }
}

// Tensor-Core absorb: qh[row][h][l] = sum_k q_b[start+row][h][k] * UK[h][k][l]
// Grid: ((n + 63)/64, LAT / 64, heads), 128 threads (4 warps)
template<class Input>
__global__ void __launch_bounds__(128) absorb_tc(Input input,int start,int n,half* qh){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int BM=64,BN=64,BK=64,THREADS=128;
    const int r0=blockIdx.x*BM,l0=blockIdx.y*BN,h=blockIdx.z,t=threadIdx.x,w=t>>5;
    __shared__ __align__(32) half arena[BM*BK+BK*BN];
    half* A=arena;half* B=arena+BM*BK;float* O=(float*)arena;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>af;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major>bf;
    wmma::fragment<wmma::accumulator,16,16,16,float>cf[BN/16];
    #pragma unroll
    for(int j=0;j<BN/16;j++)wmma::fill_fragment(cf[j],0.f);
    const bf16* __restrict__ uk_h=(const bf16*)a.uk+(int64_t)h*NOPE*LAT;
    for(int k0=0;k0<NOPE;k0+=BK){
        for(int i=t;i<BM*BK/8;i+=THREADS){
            const int rr=i/(BK/8),kk=(i%(BK/8))*8;
            ((int4*)A)[i]=(r0+rr<n)
                ?__ldg((const int4*)((const half*)a.q_b+((int64_t)(start+r0+rr)*a.heads+h)*NOPE+k0+kk))
                :make_int4(0,0,0,0);
        }
        for(int i=t;i<BK*BN/8;i+=THREADS){
            const int kk=i/(BN/8),ll=(i%(BN/8))*8;
            const uint4 u=__ldg((const uint4*)(uk_h+(int64_t)(k0+kk)*LAT+l0+ll));
            const bf16* b=(const bf16*)&u;half h8[8];
            #pragma unroll
            for(int m=0;m<8;m++)h8[m]=__float2half_rn(__bfloat162float(b[m]));
            ((int4*)B)[i]=*(const int4*)h8;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k+=16){
            wmma::load_matrix_sync(af,A+(w*16)*BK+k,BK);
            #pragma unroll
            for(int j=0;j<BN/16;j++){
                wmma::load_matrix_sync(bf,B+k*BN+j*16,BN);
                wmma::mma_sync(cf[j],af,bf,cf[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<BN/16;j++)wmma::store_matrix_sync(O+(w*16)*BN+j*16,cf[j],BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=t;i<BM*BN;i+=THREADS){
        const int rr=i/BN,ll=i%BN;
        if(r0+rr<n){
            const half z=__float2half_rn(O[i]);
            if(!isfinite(__half2float(z)))error(a,4);
            qh[((int64_t)(r0+rr)*a.heads+h)*LAT+l0+ll]=z;
        }
    }
}

// Tensor-Core sparse QK: all native heads (MT*16) share one gather of 64 selected latent rows.
// Grid: (TOP / 64, n), 128 threads (4 warps)
template<int MT,class Input>
__global__ void __launch_bounds__(128) sparse_qk_pf(Input input,int start,const half* qh,float* scores){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int N=64,K=64,THREADS=128;
    const int row=blockIdx.y,r=start+row,k0=blockIdx.x*N,t=threadIdx.x,w=t>>5;FOR_ROW(a,input,r);
    const int hn=min(16*MT,a.heads);
    __shared__ const uint8_t* sh_v[N];
    __shared__ unsigned sh_any;
    if(t==0)sh_any=0;
    __syncthreads();
    if(t<N){
        const int id=a.indices[(int64_t)r*TOP+k0+t];
        const uint8_t* v=(id>=0&&id<=a.positions[r])?kv_at(a,id):nullptr;
        sh_v[t]=v;
        if(v)atomicOr(&sh_any,1u);
    }
    __syncthreads();
    if(!sh_any){
        for(int i=t;i<hn*N;i+=THREADS){
            const int h=i/N,j=i%N;
            scores[((int64_t)row*a.heads+h)*TOP+k0+j]=-INFINITY;
        }
        return;
    }
    __shared__ __align__(32) half arena[(16*MT+N)*K];
    half* A=arena;half* B=A+16*MT*K;float* O=(float*)arena;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>aa;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>bb;
    wmma::fragment<wmma::accumulator,16,16,16,float>cc[MT];
    #pragma unroll
    for(int m=0;m<MT;m++)wmma::fill_fragment(cc[m],0.f);
    for(int base=0;base<LAT;base+=K){
        for(int i=t;i<16*MT*K/8;i+=THREADS){
            const int h=i/(K/8),d=base+(i%(K/8))*8;
            ((int4*)A)[i]=(h<a.heads)?__ldg((const int4*)(qh+((int64_t)row*a.heads+h)*LAT+d)):make_int4(0,0,0,0);
        }
        for(int i=t;i<N*K/8;i+=THREADS){
            const int j=i/(K/8),d=base+(i%(K/8))*8;
            const uint8_t* v=sh_v[j];
            ((int4*)B)[i]=v?lat8_half(v,d):make_int4(0,0,0,0);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K;k+=16){
            wmma::load_matrix_sync(bb,B+w*16*K+k,K);
            #pragma unroll
            for(int m=0;m<MT;m++){
                wmma::load_matrix_sync(aa,A+m*16*K+k,K);
                wmma::mma_sync(cc[m],aa,bb,cc[m]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<MT;m++)wmma::store_matrix_sync(O+m*16*N+w*16,cc[m],N,wmma::mem_row_major);
    __syncthreads();
    for(int i=t;i<hn*N;i+=THREADS){
        const int h=i/N,j=i%N;
        scores[((int64_t)row*a.heads+h)*TOP+k0+j]=sh_v[j]?O[i]*(1.f/16):-INFINITY;
    }
}

// Tensor-Core sparse PV: all native heads (MT*16) share one gather of 128 selected latent rows per step, writing FP16 ctxh.
// Grid: (LAT / 64, n), 128 threads (4 warps)
template<int MT,class Input>
__global__ void __launch_bounds__(128) sparse_pv_pf(Input input,int start,const float* p,half* ctxh){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int N=64,K=128,THREADS=128;
    const int row=blockIdx.y,r=start+row,d0=blockIdx.x*N,t=threadIdx.x,w=t>>5;FOR_ROW(a,input,r);
    const int hn=min(16*MT,a.heads),pos=a.positions[r];
    __shared__ const uint8_t* sh_v[K];
    __shared__ unsigned sh_any;
    __shared__ __align__(32) half arena[(16*MT+N)*K];
    half* A=arena;half* B=A+16*MT*K;float* O=(float*)arena;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>aa;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major>bb;
    wmma::fragment<wmma::accumulator,16,16,16,float>cc[MT];
    #pragma unroll
    for(int m=0;m<MT;m++)wmma::fill_fragment(cc[m],0.f);
    for(int base=0;base<TOP;base+=K){
        if(t==0)sh_any=0;
        __syncthreads();
        const int id=a.indices[(int64_t)r*TOP+base+t];
        const uint8_t* v=(id>=0&&id<=pos)?kv_at(a,id):nullptr;
        sh_v[t]=v;
        if(v)atomicOr(&sh_any,1u);
        __syncthreads();
        if(!sh_any)continue;
        for(int i=t;i<16*MT*K/2;i+=THREADS){
            const int h=i/(K/2),j=base+(i%(K/2))*2;
            const float2 pv=(h<a.heads)?*(const float2*)(p+((int64_t)row*a.heads+h)*TOP+j):make_float2(0.f,0.f);
            ((half2*)A)[i]=__floats2half2_rn(pv.x,pv.y);
        }
        for(int i=t;i<K*N/8;i+=THREADS){
            const int j=i/(N/8),d=d0+(i%(N/8))*8;
            const uint8_t* vj=sh_v[j];
            ((int4*)B)[i]=vj?lat8_half(vj,d):make_int4(0,0,0,0);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<K;k+=16){
            wmma::load_matrix_sync(bb,B+k*N+w*16,N);
            #pragma unroll
            for(int m=0;m<MT;m++){
                wmma::load_matrix_sync(aa,A+m*16*K+k,K);
                wmma::mma_sync(cc[m],aa,bb,cc[m]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<MT;m++)wmma::store_matrix_sync(O+m*16*N+w*16,cc[m],N,wmma::mem_row_major);
    __syncthreads();
    for(int i=t;i<hn*N;i+=THREADS){
        const int h=i/N,d=i%N;
        ctxh[((int64_t)row*a.heads+h)*LAT+d0+d]=__float2half_rn(O[i]);
    }
}

// One CTA per query. The selected latent tile serves both QK and PV; no global scores/probabilities.
// A WMMA-loaded row-scale fragment rescales the running accumulator without assuming its opaque lane layout.
// P_hi + P_lo (scaled by 2048) retain the probability residual with FP32 accumulators.
constexpr int MLA_SMEM = (32*(LAT+8)+32*(LAT+8)+2*32*32)*sizeof(half)+(32*32+32*16+2*32)*sizeof(float);
template<class Input>
__global__ void __launch_bounds__(256) sparse_mla_pf(Input input,int start,const half* qh,half* ctxh){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int H=32,N=32,T=256,LD=LAT+8;
    const int row=blockIdx.x,r=start+row,t=threadIdx.x,w=t>>5,lane=t&31;
    FOR_ROW(a,input,r);
    extern __shared__ __align__(32) unsigned char sm[];
    half* Q=(half*)sm;half* V=Q+H*LD;half* P=V+N*LD;
    float* S=(float*)(P+H*N);float* A=S+H*N;float* M=A+H*16;float* L=M+H;
    half* Plo=(half*)(L+H);
    __shared__ const uint8_t* vp[N];
    __shared__ unsigned any;
    for(int i=t;i<H*LAT/8;i+=T){
        const int h=i/(LAT/8),d=(i%(LAT/8))*8;
        *(int4*)(Q+h*LD+d)=h<a.heads?__ldg((const int4*)(qh+((int64_t)row*a.heads+h)*LAT+d)):make_int4(0,0,0,0);
    }
    if(t<H){M[t]=-INFINITY;L[t]=0.f;}
    wmma::fragment<wmma::accumulator,16,16,16,float> out[8],outlo[8];
    #pragma unroll
    for(int j=0;j<8;j++){wmma::fill_fragment(out[j],0.f);wmma::fill_fragment(outlo[j],0.f);}
    __syncthreads();
    for(int base=0;base<TOP;base+=N){
        if(t==0)any=0;
        __syncthreads();
        if(t<N){
            const int id=a.indices[(int64_t)r*TOP+base+t];
            vp[t]=(id>=0&&id<=a.positions[r])?kv_at(a,id):nullptr;
            if(vp[t])atomicOr(&any,1u);
        }
        __syncthreads();
        if(!any)continue;
        for(int i=t;i<N*LAT/8;i+=T){
            const int j=i/(LAT/8),d=(i%(LAT/8))*8;
            *(int4*)(V+j*LD+d)=vp[j]?lat8_half(vp[j],d):make_int4(0,0,0,0);
        }
        __syncthreads();
        if(w<4){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> kb;
            wmma::fragment<wmma::accumulator,16,16,16,float> s;
            wmma::fill_fragment(s,0.f);
            #pragma unroll
            for(int k=0;k<LAT;k+=16){
                wmma::load_matrix_sync(qa,Q+(w/2)*16*LD+k,LD);
                wmma::load_matrix_sync(kb,V+(w%2)*16*LD+k,LD);
                wmma::mma_sync(s,qa,kb,s);
            }
            wmma::store_matrix_sync(S+(w/2)*16*N+(w%2)*16,s,N,wmma::mem_row_major);
        }
        __syncthreads();
        // Each warp computes four complete softmax rows. Invalid/padded selections contribute zero.
        #pragma unroll
        for(int h=w;h<H;h+=8){
            float s=vp[lane]?S[h*N+lane]*(1.f/16.f):-INFINITY;
            float mx=s;
            #pragma unroll
            for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
            const float m=fmaxf(M[h],mx),alpha=expf(M[h]-m);
            const float p=vp[lane]?expf(s-m):0.f;
            P[h*N+lane]=__float2half_rn(p);
            Plo[h*N+lane]=__float2half_rn((p-__half2float(P[h*N+lane]))*2048.f);
            float sum=p;
            #pragma unroll
            for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,d);
            if(lane<16)A[h*16+lane]=alpha;
            if(lane==0){M[h]=m;L[h]=L[h]*alpha+sum;}
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> scale;
        wmma::load_matrix_sync(scale,A+(w/4)*16*16,16,wmma::mem_row_major);
        #pragma unroll
        for(int j=0;j<8;j++){
            #pragma unroll
            for(int x=0;x<out[j].num_elements;x++){out[j].x[x]*=scale.x[x];outlo[j].x[x]*=scale.x[x];}
        }
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> pa,palo;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> vb;
        #pragma unroll
        for(int k=0;k<N;k+=16){
            wmma::load_matrix_sync(pa,P+(w/4)*16*N+k,N);
            wmma::load_matrix_sync(palo,Plo+(w/4)*16*N+k,N);
            #pragma unroll
            for(int j=0;j<8;j++){
                wmma::load_matrix_sync(vb,V+k*LD+(w%4)*128+j*16,LD);
                wmma::mma_sync(out[j],pa,vb,out[j]);
                wmma::mma_sync(outlo[j],palo,vb,outlo[j]);
            }
        }
        __syncthreads();
    }
    if(t<a.heads&&L[t]==0)error(a,16);
    for(int j=0;j<8;j++)for(int x=0;x<out[j].num_elements;x++)out[j].x[x]=fmaf(outlo[j].x[x],1.f/2048.f,out[j].x[x]);
    // Q and V are dead: reuse their combined 64 KiB for the FP32 output tile.
    float* O=(float*)Q;
    #pragma unroll
    for(int j=0;j<8;j++)wmma::store_matrix_sync(O+(w/4)*16*LAT+(w%4)*128+j*16,out[j],LAT,wmma::mem_row_major);
    __syncthreads();
    for(int i=t;i<a.heads*LAT;i+=T){
        const int h=i/LAT;
        ctxh[(int64_t)row*a.heads*LAT+i]=__float2half_rn(L[h]>0?O[i]/L[h]:0.f);
    }
}

// Opt-in 512-thread P1 variant: identical per-output math, four PV fragments/warp.
template<class Input>
__global__ void __launch_bounds__(512) sparse_mla_pf_512(Input input,int start,const half* qh,half* ctxh){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int H=32,N=32,T=512,LD=LAT+8;
    const int row=blockIdx.x,r=start+row,t=threadIdx.x,w=t>>5,lane=t&31;
    FOR_ROW(a,input,r);
    extern __shared__ __align__(32) unsigned char sm[];
    half* Q=(half*)sm;half* V=Q+H*LD;half* P=V+N*LD;
    float* S=(float*)(P+H*N);float* A=S+H*N;float* M=A+H*16;float* L=M+H;
    half* Plo=(half*)(L+H);
    __shared__ const uint8_t* vp[N];
    __shared__ unsigned any;
    for(int i=t;i<H*LAT/8;i+=T){
        const int h=i/(LAT/8),d=(i%(LAT/8))*8;
        *(int4*)(Q+h*LD+d)=h<a.heads?__ldg((const int4*)(qh+((int64_t)row*a.heads+h)*LAT+d)):make_int4(0,0,0,0);
    }
    if(t<H){M[t]=-INFINITY;L[t]=0.f;}
    wmma::fragment<wmma::accumulator,16,16,16,float> out[4],outlo[4];
    #pragma unroll
    for(int j=0;j<4;j++){wmma::fill_fragment(out[j],0.f);wmma::fill_fragment(outlo[j],0.f);}
    __syncthreads();
    for(int base=0;base<TOP;base+=N){
        if(t==0)any=0;
        __syncthreads();
        if(t<N){
            const int id=a.indices[(int64_t)r*TOP+base+t];
            vp[t]=(id>=0&&id<=a.positions[r])?kv_at(a,id):nullptr;
            if(vp[t])atomicOr(&any,1u);
        }
        __syncthreads();
        if(!any)continue;
        for(int i=t;i<N*LAT/8;i+=T){
            const int j=i/(LAT/8),d=(i%(LAT/8))*8;
            *(int4*)(V+j*LD+d)=vp[j]?lat8_half(vp[j],d):make_int4(0,0,0,0);
        }
        __syncthreads();
        if(w<4){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> kb;
            wmma::fragment<wmma::accumulator,16,16,16,float> s;
            wmma::fill_fragment(s,0.f);
            #pragma unroll
            for(int k=0;k<LAT;k+=16){
                wmma::load_matrix_sync(qa,Q+(w/2)*16*LD+k,LD);
                wmma::load_matrix_sync(kb,V+(w%2)*16*LD+k,LD);
                wmma::mma_sync(s,qa,kb,s);
            }
            wmma::store_matrix_sync(S+(w/2)*16*N+(w%2)*16,s,N,wmma::mem_row_major);
        }
        __syncthreads();
        // Each warp computes four complete softmax rows. Invalid/padded selections contribute zero.
        #pragma unroll
        for(int h=w;h<H;h+=16){
            float s=vp[lane]?S[h*N+lane]*(1.f/16.f):-INFINITY;
            float mx=s;
            #pragma unroll
            for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
            const float m=fmaxf(M[h],mx),alpha=expf(M[h]-m);
            const float p=vp[lane]?expf(s-m):0.f;
            P[h*N+lane]=__float2half_rn(p);
            Plo[h*N+lane]=__float2half_rn((p-__half2float(P[h*N+lane]))*2048.f);
            float sum=p;
            #pragma unroll
            for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,d);
            if(lane<16)A[h*16+lane]=alpha;
            if(lane==0){M[h]=m;L[h]=L[h]*alpha+sum;}
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> scale;
        wmma::load_matrix_sync(scale,A+(w/8)*16*16,16,wmma::mem_row_major);
        #pragma unroll
        for(int j=0;j<4;j++){
            #pragma unroll
            for(int x=0;x<out[j].num_elements;x++){out[j].x[x]*=scale.x[x];outlo[j].x[x]*=scale.x[x];}
        }
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> pa,palo;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> vb;
        #pragma unroll
        for(int k=0;k<N;k+=16){
            wmma::load_matrix_sync(pa,P+(w/8)*16*N+k,N);
            wmma::load_matrix_sync(palo,Plo+(w/8)*16*N+k,N);
            #pragma unroll
            for(int j=0;j<4;j++){
                wmma::load_matrix_sync(vb,V+k*LD+(w%8)*64+j*16,LD);
                wmma::mma_sync(out[j],pa,vb,out[j]);
                wmma::mma_sync(outlo[j],palo,vb,outlo[j]);
            }
        }
        __syncthreads();
    }
    if(t<a.heads&&L[t]==0)error(a,16);
    for(int j=0;j<4;j++)for(int x=0;x<out[j].num_elements;x++)out[j].x[x]=fmaf(outlo[j].x[x],1.f/2048.f,out[j].x[x]);
    // Q and V are dead: reuse their combined 64 KiB for the FP32 output tile.
    float* O=(float*)Q;
    #pragma unroll
    for(int j=0;j<4;j++)wmma::store_matrix_sync(O+(w/8)*16*LAT+(w%8)*64+j*16,out[j],LAT,wmma::mem_row_major);
    __syncthreads();
    for(int i=t;i<a.heads*LAT;i+=T){
        const int h=i/LAT;
        ctxh[(int64_t)row*a.heads*LAT+i]=__float2half_rn(L[h]>0?O[i]/L[h]:0.f);
    }
}

// Tensor-Core value_project: output[start+row][h][d] = sum_l ctxh[row][h][l] * UV[h][d][l]
// Grid: ((n + 63)/64, VD / 64, heads), 128 threads (4 warps)
template<class Input>
__global__ void __launch_bounds__(128) value_project_tc(Input input,int start,int n,const half* ctxh){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int BM=64,BN=64,BK=64,THREADS=128;
    const int r0=blockIdx.x*BM,d0=blockIdx.y*BN,h=blockIdx.z,t=threadIdx.x,w=t>>5;
    __shared__ __align__(32) half arena[BM*BK+BN*BK];
    half* A=arena;half* B=arena+BM*BK;float* O=(float*)arena;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>af;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>bf;
    wmma::fragment<wmma::accumulator,16,16,16,float>cf[BN/16];
    #pragma unroll
    for(int j=0;j<BN/16;j++)wmma::fill_fragment(cf[j],0.f);
    const bf16* __restrict__ uv_h=(const bf16*)a.uv+(int64_t)h*VD*LAT;
    for(int k0=0;k0<LAT;k0+=BK){
        for(int i=t;i<BM*BK/8;i+=THREADS){
            const int rr=i/(BK/8),kk=(i%(BK/8))*8;
            ((int4*)A)[i]=(r0+rr<n)
                ?__ldg((const int4*)(ctxh+((int64_t)(r0+rr)*a.heads+h)*LAT+k0+kk))
                :make_int4(0,0,0,0);
        }
        for(int i=t;i<BN*BK/8;i+=THREADS){
            const int dd=i/(BK/8),kk=(i%(BK/8))*8;
            const uint4 u=__ldg((const uint4*)(uv_h+(int64_t)(d0+dd)*LAT+k0+kk));
            const bf16* b=(const bf16*)&u;half h8[8];
            #pragma unroll
            for(int m=0;m<8;m++)h8[m]=__float2half_rn(__bfloat162float(b[m]));
            ((int4*)B)[i]=*(const int4*)h8;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k+=16){
            wmma::load_matrix_sync(af,A+(w*16)*BK+k,BK);
            #pragma unroll
            for(int j=0;j<BN/16;j++){
                wmma::load_matrix_sync(bf,B+(j*16)*BK+k,BK);
                wmma::mma_sync(cf[j],af,bf,cf[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<BN/16;j++)wmma::store_matrix_sync(O+(w*16)*BN+j*16,cf[j],BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=t;i<BM*BN;i+=THREADS){
        const int rr=i/BN,dd=i%BN;
        if(r0+rr<n)((half*)a.output)[((int64_t)(start+r0+rr)*a.heads+h)*VD+d0+dd]=__float2half_rn(O[i]);
    }
}

constexpr int PF_DEFAULT_TILE=256;
constexpr int PF_MAX_ROWS_DEFAULT=4096;

// ==== BEGIN #10 v2 (sparse_mla_pf_v2) ====
// ---- #10 v2 of sparse_mla_pf_512 (G6_PF_SPARSE_MLA_V2=1): byte-identical output, restructured data flow.
// Same per-output math as sparse_mla_pf_512: QK k-chain of 32 HMMA.16816 (k ascending, zero start), N=32 key tiles in order,
// same softmax code, same alpha multiply, same P_hi/P_lo PV chains (k 0..15 then 16..31), same epilogue ops.
// Changes: indices -> KV row offsets once (smem table), all-pad tiles skipped without syncs, KV tile prefetched into registers
// while QK/softmax/PV of the previous tile run, raw mma.sync m16n8k16 + ldmatrix, QK on 8 warps (one m16n8 chain each),
// alpha rescale and epilogue in registers, 4 barriers per live tile.
constexpr int MLA2_NT=TOP/32;
constexpr int MLA2_SMEM=(32*(LAT+8)*2+2*32*32)*sizeof(half)+(32*32+32*3)*sizeof(float)+TOP*sizeof(int)+((MLA2_NT+15)/16)*16;
constexpr int MLA2_SMEM_H16=(16*(LAT+8)+32*(LAT+8)+2*16*32)*sizeof(half)+
    (16*32+16*3)*sizeof(float)+TOP*sizeof(int)+((MLA2_NT+15)/16)*16;
static_assert(MLA2_SMEM_H16==63504,"H16 sparse MLA shared memory layout");
__device__ __forceinline__ void ldsm4(uint32_t addr,uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(addr):"memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t addr,uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(addr):"memory");
}
__device__ __forceinline__ void mma16816(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// Raw (unconverted) 8 latent elements of one cached row: BF16 = 16 B; FP8 = 2 x 4 B + the 128-element group scale.
__device__ __forceinline__ uint4 stage_ld8(const uint8_t* v,int d){
#if G6_FLASH_KV_FP8
    return make_uint4(__ldg((const uint32_t*)(v+d)),__ldg((const uint32_t*)(v+d+4)),
        __float_as_uint(__ldg(((const float*)(v+g6kv::SCALE_OFF))+(d>>7))),0u);
#else
    return __ldg((const uint4*)(v+2*d));
#endif
}
// Same conversion as lat8_half applied to staged raw data.
__device__ __forceinline__ int4 stage_cvt8(const uint4 c){
    half h[8];
#if G6_FLASH_KV_FP8
    const float s=__uint_as_float(c.z);
    #pragma unroll
    for(int i=0;i<4;i++)h[i]=__float2half_rn(e4((c.x>>(8*i))&255)*s);
    #pragma unroll
    for(int i=0;i<4;i++)h[4+i]=__float2half_rn(e4((c.y>>(8*i))&255)*s);
#else
    const bf16* b=(const bf16*)&c;
    #pragma unroll
    for(int i=0;i<8;i++)h[i]=__float2half_rn(__bfloat162float(b[i]));
#endif
    return *(const int4*)h;
}
template<class Input,int H=32>
__global__ void __launch_bounds__(512) sparse_mla_pf_v2(Input input,int start,const half* qh,half* ctxh){
    using namespace nvcuda;
    auto a=resolve(input);if(!a.active)return;
    constexpr int N=32,T=512,LD=LAT+8,NT=MLA2_NT;
    const int row=blockIdx.x,r=start+row,t=threadIdx.x,w=t>>5,lane=t&31,g=lane>>2,tq=lane&3;
    FOR_ROW(a,input,r);
    extern __shared__ __align__(32) unsigned char sm[];
    half* Q=(half*)sm;half* V=Q+H*LD;half* P=V+N*LD;half* Plo=P+H*N;
    float* S=(float*)(Plo+H*N);float* AL=S+H*N;float* M=AL+H;float* L=M+H;
    int* tab=(int*)(L+H);unsigned char* alive=(unsigned char*)(tab+TOP);
    const uint32_t q_s=(uint32_t)__cvta_generic_to_shared(Q),v_s=(uint32_t)__cvta_generic_to_shared(V),
        p_s=(uint32_t)__cvta_generic_to_shared(P),pl_s=(uint32_t)__cvta_generic_to_shared(Plo);
    for(int i=t;i<H*LAT/8;i+=T){
        const int h=i/(LAT/8),d=(i%(LAT/8))*8;
        *(int4*)(Q+h*LD+d)=h<a.heads?__ldg((const int4*)(qh+((int64_t)row*a.heads+h)*LAT+d)):make_int4(0,0,0,0);
    }
    if(t<H){M[t]=-INFINITY;L[t]=0.f;}
    // idx -> KV row index (or -1), once. Same validity test and page lookup as kv_at.
    {
        const int pos=a.positions[r];
        for(int j=t;j<TOP;j+=T){
            const int id=a.indices[(int64_t)r*TOP+j];
            int off=-1;
            if(id>=0&&id<=pos){const int p=page(a,id);if(p>=0)off=p*a.page_size+id%a.page_size;}
            tab[j]=off;
        }
    }
    __syncthreads();
    for(int ti=w;ti<NT;ti+=16){
        const unsigned b=__ballot_sync(0xffffffff,tab[ti*N+lane]>=0);
        if(lane==0)alive[ti]=b!=0;
    }
    float out[8][4],outlo[8][4];
    #pragma unroll
    for(int j=0;j<8;j++)
        #pragma unroll
        for(int x=0;x<4;x++){out[j][x]=0.f;outlo[j][x]=0.f;}
    __syncthreads();
    // thread t stages rows (t>>6)+8c, columns (t&63)*8.. of each tile (the sparse_mla_pf_512 gather mapping)
    const int gj=t>>6,gd=(t&63)*8;
    uint4 st[4];unsigned vmask=0;
    auto stage=[&](int ti){
        vmask=0;
        #pragma unroll
        for(int c=0;c<4;c++){
            const int off=tab[ti*N+gj+8*c];
            if(off>=0){vmask|=1u<<c;st[c]=stage_ld8((const uint8_t*)a.mla_cache+(int64_t)off*g6kv::ROW,gd);}
        }
    };
    int cur=0;while(cur<NT&&!alive[cur])cur++;
    if(cur<NT)stage(cur);
    while(cur<NT){
        // V free (previous PV done): convert + store tile cur, then start the next tile's global loads
        #pragma unroll
        for(int c=0;c<4;c++)
            *(int4*)(V+(gj+8*c)*LD+gd)=(vmask>>c&1)?stage_cvt8(st[c]):make_int4(0,0,0,0);
        int nxt=cur+1;while(nxt<NT&&!alive[nxt])nxt++;
        if(nxt<NT)stage(nxt);
        __syncthreads();                                  // KV ready
        if(w<H/4){                                        // QK: m16n8 chain per warp, 32 k-steps
            const int mb=w>>2,nb=w&3;
            float s[4]={0.f,0.f,0.f,0.f};
            const uint32_t qa=q_s+2*((mb*16+(lane&15))*LD+(lane>>4)*8);
            const uint32_t kb=v_s+2*((nb*8+(lane&7))*LD+(lane>>3)*8);
            #pragma unroll
            for(int k=0;k<LAT;k+=32){
                uint32_t a0,a1,a2,a3,c0,c1,c2,c3,b0,b1,b2,b3;
                ldsm4(qa+2*k,a0,a1,a2,a3);
                ldsm4(qa+2*(k+16),c0,c1,c2,c3);
                ldsm4(kb+2*k,b0,b1,b2,b3);
                mma16816(s,a0,a1,a2,a3,b0,b1);
                mma16816(s,c0,c1,c2,c3,b2,b3);
            }
            *(float2*)(S+(mb*16+g)*N+nb*8+2*tq)=make_float2(s[0],s[1]);
            *(float2*)(S+(mb*16+g+8)*N+nb*8+2*tq)=make_float2(s[2],s[3]);
        }
        __syncthreads();                                  // S ready
        const bool valid=tab[cur*N+lane]>=0;
        #pragma unroll
        for(int h=w;h<H;h+=16){
            float s=valid?S[h*N+lane]*(1.f/16.f):-INFINITY;
            float mx=s;
            #pragma unroll
            for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
            const float m=fmaxf(M[h],mx),alpha=expf(M[h]-m);
            const float p=valid?expf(s-m):0.f;
            P[h*N+lane]=__float2half_rn(p);
            Plo[h*N+lane]=__float2half_rn((p-__half2float(P[h*N+lane]))*2048.f);
            float sum=p;
            #pragma unroll
            for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,d);
            if(lane==0){AL[h]=alpha;M[h]=m;L[h]=L[h]*alpha+sum;}
        }
        __syncthreads();                                  // P ready
        if(H==32 || w<H/2){
            const int mb=w>>3,dw=(w&7)*64;
            const float al0=AL[mb*16+g],al1=AL[mb*16+g+8];
            #pragma unroll
            for(int j=0;j<8;j++){
                out[j][0]*=al0;out[j][1]*=al0;out[j][2]*=al1;out[j][3]*=al1;
                outlo[j][0]*=al0;outlo[j][1]*=al0;outlo[j][2]*=al1;outlo[j][3]*=al1;
            }
            #pragma unroll
            for(int k=0;k<N;k+=16){
                uint32_t p0,p1,p2,p3,l0,l1,l2,l3;
                ldsm4(p_s+2*((mb*16+(lane&15))*N+k+(lane>>4)*8),p0,p1,p2,p3);
                ldsm4(pl_s+2*((mb*16+(lane&15))*N+k+(lane>>4)*8),l0,l1,l2,l3);
                #pragma unroll
                for(int j=0;j<4;j++){
                    uint32_t b0,b1,b2,b3;
                    ldsm4t(v_s+2*((k+(lane&7)+((lane>>3)&1)*8)*LD+dw+j*16+(lane>>4)*8),b0,b1,b2,b3);
                    mma16816(out[2*j],p0,p1,p2,p3,b0,b1);mma16816(outlo[2*j],l0,l1,l2,l3,b0,b1);
                    mma16816(out[2*j+1],p0,p1,p2,p3,b2,b3);mma16816(outlo[2*j+1],l0,l1,l2,l3,b2,b3);
                }
            }
        }
        __syncthreads();                                  // V, P free
        cur=nxt;
    }
    if(t<a.heads&&L[t]==0)error(a,16);
    if(H==32 || w<H/2){
        const int mb=w>>3,dw=(w&7)*64,h0=mb*16+g,h1=h0+8;
        const float L0=L[h0],L1=L[h1];
        #pragma unroll
        for(int j=0;j<8;j++){
            const int d=dw+j*8+2*tq;
            #pragma unroll
            for(int x=0;x<4;x++)out[j][x]=fmaf(outlo[j][x],1.f/2048.f,out[j][x]);
            if(h0<a.heads)*(half2*)(ctxh+((int64_t)row*a.heads+h0)*LAT+d)=
                __floats2half2_rn(L0>0?out[j][0]/L0:0.f,L0>0?out[j][1]/L0:0.f);
            if(h1<a.heads)*(half2*)(ctxh+((int64_t)row*a.heads+h1)*LAT+d)=
                __floats2half2_rn(L1>0?out[j][2]/L1:0.f,L1>0?out[j][3]/L1:0.f);
        }
    }
}
// ==== END #10 v2 ====

}   // namespace fa_pf

struct G6AttnPrefillScratch {
    void* data=nullptr;
    size_t bytes=0;
    int H=0,cap=0,maxrows=0,R=0,pstride=0,device=0;
    fa::half *qh=nullptr,*ctxh=nullptr,*output=nullptr;
    float *ps=nullptr,*prob=nullptr;
    int* indices=nullptr;
    std::atomic<unsigned> refs{1};
};

struct G6AttnPrefill {
    G6AttnBinding* owner=nullptr;
    G6AttnPrefillScratch* workspace=nullptr;
    int H=0,cap=0,maxrows=0,R=0;
    size_t bytes=0;
};

namespace fa_pf {
static size_t scratch_layout(G6AttnPrefillScratch& w,void* base){
    size_t used=0;
    auto take=[&](size_t n){size_t at=used;used=(used+n+255)&~size_t(255);return (void*)((uintptr_t)base+at);};
    const size_t R=w.R,H=w.H,M=w.maxrows;
    w.pstride=pool_stride(w.cap);
    w.qh=(half*)take(R*H*LAT*2);
    w.ctxh=(half*)take(R*H*LAT*2);
    w.ps=(float*)take(R*w.pstride*4);
    w.prob=(float*)take(R*H*TOP*4);
    w.indices=(int*)take(M*TOP*4);
    w.output=(half*)take(M*H*VD*2);
    return used;
}
static cudaError_t release_scratch(G6AttnPrefillScratch* w){
    if(!w||w->refs.fetch_sub(1,std::memory_order_acq_rel)!=1)return cudaSuccess;
    cudaError_t e=w->data?cudaFree(w->data):cudaSuccess;
    delete w;
    return e;
}
static int create_prefill(G6AttnPrefill** out,G6AttnBinding* b,G6AttnPrefill* owner,bool shared){
    if(!out||*out||!b||!b->storage)return INVAL;
    int tile=PF_DEFAULT_TILE;
    if(const char* s=getenv("G6_PF_TILE")){
        tile=atoi(s);
        if(tile!=64&&tile!=128&&tile!=256)return INVAL;
    }
    const int maxrows=std::max(b->max_rows,PF_MAX_ROWS_DEFAULT);
    const int R=std::min(tile,maxrows);
    int device=0;
    cudaError_t e=cudaGetDevice(&device);
    if(e!=cudaSuccess)return (int)e;
    auto* p=new G6AttnPrefill{};
    p->owner=b;p->H=b->args.heads;p->cap=b->capacity;p->maxrows=maxrows;p->R=R;
    if(owner){
        auto* w=owner->workspace;
        if(!w||w->device!=device||p->cap>w->cap||p->maxrows>w->maxrows||p->R>w->R||p->H>w->H){delete p;return INVAL;}
        p->workspace=w;
        w->refs.fetch_add(1,std::memory_order_relaxed);
        p->bytes=0;
    }else{
        auto* w=new G6AttnPrefillScratch{};
        w->H=shared?g6nr::HMAX:p->H;
        w->cap=p->cap;w->maxrows=p->maxrows;w->R=p->R;w->device=device;
        w->bytes=scratch_layout(*w,nullptr);
        e=cudaMalloc(&w->data,w->bytes);
        if(e!=cudaSuccess){delete w;delete p;return (int)e;}
        scratch_layout(*w,w->data);
        p->workspace=w;
        p->bytes=w->bytes;
    }
    *out=p;
    return 0;
}
}   // namespace fa_pf

extern "C" int g6_attn_bind_prefill_create(G6AttnPrefill** out,G6AttnBinding* b,void*){
    return fa_pf::create_prefill(out,b,nullptr,false);
}
extern "C" int g6_attn_bind_prefill_create_shared(G6AttnPrefill** out,G6AttnBinding* b,void*,G6AttnPrefill* owner){
    return fa_pf::create_prefill(out,b,owner,true);
}
extern "C" int g6_attn_bind_prefill_release(G6AttnPrefill* p){
    if(!p)return 0;
    cudaError_t e=fa_pf::release_scratch(p->workspace);
    delete p;
    return (int)e;
}
extern "C" size_t g6_attn_bind_prefill_bytes(const G6AttnPrefill* p){return p?p->bytes:0;}
extern "C" int g6_attn_bind_prefill_get_config(const G6AttnPrefill* p,int* query_tile,int* dense_limit){
    if(!p)return INVAL;if(query_tile)*query_tile=p->R;if(dense_limit)*dense_limit=0;return 0;
}
extern "C" void* g6_attn_bind_prefill_output(const G6AttnPrefill* p,const G6AttnBinding* b,int rows){
    if(!p||!b)return nullptr;
    return (rows<=b->max_rows)?b->args.output:(void*)p->workspace->output;
}
extern "C" int* g6_attn_bind_prefill_indices(const G6AttnPrefill* p,const G6AttnBinding* b,int rows){
    if(!p||!b)return nullptr;
    return (rows<=b->max_rows)?b->args.indices:p->workspace->indices;
}
static int flash_prefill_run(G6AttnPrefill* p,G6AttnBinding* b,const G6AttnStep* step,void* s,
                             const G6FlashPoolOwner* owner){
    if(!p||!b||!b->storage||p->owner!=b||!step)return INVAL;
    static const bool force_p1=getenv("G6_FPF_ATTN_P1")&&atoi(getenv("G6_FPF_ATTN_P1"))==1;
    if(owner && (g6nr::NR!=2 || owner->rank!=b->rank || !owner->opaque || !owner->publish ||
                 !owner->receive || step->reuse_indices || force_p1)) return INVAL;
    if(force_p1&&step->rows<=b->max_rows)return g6_attn_bind_run(b,step,s);
    if(step->rows<=0||step->rows>p->maxrows||step->context<=0||step->context>b->capacity||
       step->rows>step->context||!step->q_b||!step->kv_a||!step->positions)return INVAL;
    if(step->reuse_indices){
        if(step->reuse_rows<=0||(!step->reuse_row_map&&step->reuse_rows<step->rows))return INVAL;
    }else if(!step->index_q||!step->index_k||!step->index_weight||!step->index_gate||
             step->reuse_row_map||step->reuse_rows)return INVAL;
    cudaStream_t stream=(cudaStream_t)s;
    G6Attn a=b->args;
    a.rows=step->rows;a.context=step->context;
    a.q_b=step->q_b;a.kv_a=step->kv_a;a.positions=step->positions;
    a.index_q=step->index_q;a.index_k=step->index_k;a.index_weight=step->index_weight;
    a.index_gate=step->index_gate;
    a.reuse_indices=step->reuse_indices;a.reuse_row_map=step->reuse_row_map;a.reuse_rows=step->reuse_rows;
    auto* w=p->workspace;
    if(step->rows>b->max_rows){
        a.indices=w->indices;
        a.output=w->output;
    }
    if(step->reuse_indices&&step->reuse_row_map&&step->reuse_indices==a.indices)return INVAL;
    cudaError_t ce=cudaMemsetAsync(a.error,0,sizeof(unsigned),stream);
    if(ce!=cudaSuccess)return (int)ce;
    fa::append(a,a,stream);
    const int pstride=fa::pool_stride(a.context);
    static const int kp_tc=getenv("G6_PF_KP_TC")?atoi(getenv("G6_PF_KP_TC")):2;
    const char* mla_env=getenv("G6_PF_SPARSE_MLA");
    const bool fused_mla=mla_env&&atoi(mla_env)==1&&a.heads<=32;
    const char* mla512_env=getenv("G6_PF_SPARSE_MLA_512");
    const bool fused_mla512=fused_mla&&mla512_env&&atoi(mla512_env)==1;
    // V2 defaults on only with sparse MLA + _512 and heads<=32. Explicit _V2=0 restores _512;
    // explicit _V2=1 retains the original opt-in outside _512 (still within sparse MLA).
    const char* mla_v2_env=getenv("G6_PF_SPARSE_MLA_V2");
    const bool fused_mla_v2=fused_mla&&(mla_v2_env?atoi(mla_v2_env)==1:fused_mla512);
    // Separate exact H16c arm. Unset keeps the existing H32 V2 instruction path.
    const char* h16c_env=getenv("G6_PF_SPARSE_MLA_H16C");
    const bool fused_mla_h16c=fused_mla_v2&&a.heads<=16&&h16c_env&&atoi(h16c_env)==1;
    if(fused_mla){
        if(fused_mla_v2)
            ce=fused_mla_h16c
                ?cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn,16>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM_H16)
                :cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM);
        else if(fused_mla512)
            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf_512<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM);
        else
            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM);
        if(ce!=cudaSuccess)return (int)ce;
    }
    auto select = [&](int start, int n) {
            if(kp_tc==3)fa_pf::kp_scores_tc_stream<2><<<dim3(pstride/32,(n+63)/64),128,0,stream>>>(a,start,n,w->ps,pstride);
            else if(kp_tc>=2)fa_pf::kp_scores_tc<2><<<dim3(pstride/32,(n+63)/64),128,0,stream>>>(a,start,n,w->ps,pstride);
            else if(kp_tc==1)fa_pf::kp_scores_tc<1><<<dim3(pstride/32,(n+63)/64),128,0,stream>>>(a,start,n,w->ps,pstride);
            else fa_pf::kp_scores_pf<<<dim3(pstride/32,(n+15)/16),256,0,stream>>>(a,start,n,w->ps,pstride);
            fa::kp_select<<<n,fa::SELT,0,stream>>>(a,start,w->ps,pstride);
    };
    if(owner){
        const int lo=flash_pool_owner::begin(a.rows,owner->rank),hi=flash_pool_owner::end(a.rows,owner->rank);
        for(int start=lo;start<hi;start+=p->R) select(start,std::min(hi-start,p->R));
        const int e=owner->publish(owner->opaque,a.indices,a.positions,a.rows,s);
        if(e)return e;
        AMARK(1);
    }
    for(int start=0;start<a.rows;start+=p->R){
        const int n=std::min(a.rows-start,p->R);
        if(owner){ /* owned selection completed above; the peer arrives before QK */ }
        else if(a.reuse_indices){
            AMARK(1);
            fa::reuse<<<n,256,0,stream>>>(a,start);
        }else{
            select(start,n);
            AMARK(1);
        }
        AMARK(2);
        fa_pf::absorb_tc<<<dim3((n+63)/64,fa::LAT/64,a.heads),128,0,stream>>>(a,start,n,w->qh);
        AMARK(3);
        // absorb_tc reads queries/UK only. Complete IDs before either MLA path.
        if(owner && start==0){
            const int e=owner->receive(owner->opaque,a.indices,a.positions,a.rows,s);
            if(e)return e;
        }
        if(fused_mla){
            if(fused_mla_v2){
                if(fused_mla_h16c)fa_pf::sparse_mla_pf_v2<G6Attn,16><<<n,512,fa_pf::MLA2_SMEM_H16,stream>>>(a,start,w->qh,w->ctxh);
                else fa_pf::sparse_mla_pf_v2<<<n,512,fa_pf::MLA2_SMEM,stream>>>(a,start,w->qh,w->ctxh);
            }
            else if(fused_mla512)fa_pf::sparse_mla_pf_512<<<n,512,fa_pf::MLA_SMEM,stream>>>(a,start,w->qh,w->ctxh);
            else fa_pf::sparse_mla_pf<<<n,256,fa_pf::MLA_SMEM,stream>>>(a,start,w->qh,w->ctxh);
            AMARK(4);
            AMARK(5);
        }else{
            if(a.heads>16)fa_pf::sparse_qk_pf<2><<<dim3(fa::TOP/64,n),128,0,stream>>>(a,start,w->qh,w->prob);
            else fa_pf::sparse_qk_pf<1><<<dim3(fa::TOP/64,n),128,0,stream>>>(a,start,w->qh,w->prob);
            AMARK(4);
            fa::normalize<<<n*a.heads,256,0,stream>>>(a,w->prob);
            AMARK(5);
            if(a.heads>16)fa_pf::sparse_pv_pf<2><<<dim3(fa::LAT/64,n),128,0,stream>>>(a,start,w->prob,w->ctxh);
            else fa_pf::sparse_pv_pf<1><<<dim3(fa::LAT/64,n),128,0,stream>>>(a,start,w->prob,w->ctxh);
        }
        AMARK(6);
        AMARK(7);
        fa_pf::value_project_tc<<<dim3((n+63)/64,fa::VD/64,a.heads),128,0,stream>>>(a,start,n,w->ctxh);
        AMARK(8);
    }
    return (int)cudaGetLastError();
}
extern "C" int g6_attn_bind_prefill_run(G6AttnPrefill* p,G6AttnBinding* b,const G6AttnStep* step,void* s){
    return flash_prefill_run(p,b,step,s,nullptr);
}
extern "C" int g6_attn_bind_prefill_run_owned(G6AttnPrefill* p,G6AttnBinding* b,const G6AttnStep* step,void* s,
                                           const G6FlashPoolOwner* owner){
    return owner ? flash_prefill_run(p,b,step,s,owner) : INVAL;
}
extern "C" int g6_attn_bind_prefill_trace(G6AttnPrefill*,G6AttnBinding*,const G6AttnStep*,const G6AttnTrace*,void*){return NOTSUP;}
extern "C" int g6_attn_bind_baseline_trace(G6AttnBinding*,const G6AttnStep*,const G6AttnTrace*,void*){return NOTSUP;}
extern "C" int g6_attn_bind_prefill_profile_read(G6AttnPrefill*,G6AttnPrefillProfile* out){
    if(out){*out=G6AttnPrefillProfile{};out->status=G6_PF_PROFILE_OFF;}return 0;
}
// append only (latent + index keys) of a full-index step
extern "C" int g6_attn_bind_index_append(G6AttnBinding* b,const G6AttnStep* step,void* s){
    int e=g6_attn_bind_prepare(b,step,s);return e?e:g6_attn_append(&b->args,s);
}
extern "C" int g6_attn_bind_prefill_index_rows(G6AttnPrefill*,G6AttnBinding*,const G6AttnStep*,int*,float*,void*){return NOTSUP;}
extern "C" int g6_attn_qo_create(G6AttnQo**,const G6AttnBinding*,const G6AttnTensor*,const G6AttnTensor*,int,void*,G6AttnQo*){return NOTSUP;}
extern "C" int g6_attn_qo_run(G6AttnQo*,G6AttnBinding*,const G6AttnQoStep*,void*){return NOTSUP;}
extern "C" size_t g6_attn_qo_bytes(const G6AttnQo*){return 0;}
extern "C" void g6_attn_qo_prof(int,double*){}
extern "C" int g6_attn_qo_warm(G6AttnQo*,int,void*){return NOTSUP;}
extern "C" int g6_attn_qo_release(G6AttnQo*){return 0;}
extern "C" int g6_attn_qo_load(G6AttnQo*,const G6AttnTensor*,const G6AttnTensor*,void*){return NOTSUP;}
extern "C" void g6_attn_qo_cap(int){}
// the remote rows tier (runtime/rt_rows.cuh) moves GLM's MLA latent rows, not Flash's: unsupported, as in a G6_KV4 build
extern "C" int g6_rt_pack_int4(const void*,size_t,void*,size_t,int,void*){return NOTSUP;}
extern "C" int g6_rt_unpack_int4(const void*,size_t,void*,size_t,int,void*){return NOTSUP;}
