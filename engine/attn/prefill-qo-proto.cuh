#pragma once
// Standalone query-owned sparse MLA prototype. No production caller.
// One CTA=(row, up to32 heads), eight warps. Q is already absorbed/rotated FP16.
// Actual656B cache format: 512 E4M3FN + four FP32 scales +64 BF16 RoPE values.
// Latent dequant uses FP32 code*scale then RN16, matching v5's expanded operand.
// Ordered duplicate IDs remain separate softmax entries. Invalid/future IDs mask.
// Online P is RN16(exp(score-new_max)); denominator stays unrounded FP32.
// Core emits FP32 latent context. UV/output rounding belongs to an untimed adapter.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>
namespace gqo {
constexpr int HEADS=32, KEYS=32, WIDTH=576, VALUE=512, TOP=2048, THREADS=256;
constexpr int STRIDE=584, PSTRIDE=40;
struct Args {
    const half* q;                // [M,H,576], half absorbed Q plus rotated RoPE
    const uint8_t* records;       // [T,656], logical contiguous records, read-only
    const int* ids;               // [M,2048], ordered list, -1 padding
    const int* positions;         // [M]
    float* context;               // [M,H,512]
    unsigned* error;              // bit4 nonfinite, bit16 no valid selected key
    int rows,heads,tokens;
};
struct alignas(16) Shared {
    half q[HEADS*STRIDE], key[KEYS*STRIDE], p[HEADS*PSTRIDE];
    float score[HEADS*KEYS], alpha[HEADS], denominator[HEADS];
    int ids[KEYS], last[8], end;
};
static_assert(sizeof(Shared)<=101376,"one-CTA shared memory limit");
__device__ __forceinline__ uint32_t word(const half* p){return *reinterpret_cast<const uint32_t*>(p);}
__device__ __forceinline__ uint32_t pair(half a,half b){return uint32_t(__half_as_ushort(a))|(uint32_t(__half_as_ushort(b))<<16);}
__device__ __forceinline__ void mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    gr8_sm70::mma16816_f16(*reinterpret_cast<float (*)[4]>(c), a0, a1, a2, a3, b0, b1);
}
__global__ __launch_bounds__(THREADS) void core(Args a){
    extern __shared__ __align__(16) unsigned char storage[];
    Shared& s=*reinterpret_cast<Shared*>(storage);
    const int groups=(a.heads+31)/32,row=blockIdx.x/groups,h0=(blockIdx.x%groups)*32;
    const int t=threadIdx.x,w=t/32,l=t%32,g=l/4,c=l%4,hm=(w/4)*16,kb=(w%4)*8;
    const int position=a.positions[row];
    for(int i=t;i<HEADS*WIDTH;i+=THREADS){
        int h=i/WIDTH,d=i%WIDTH;half q=h0+h<a.heads?a.q[((int64_t)row*a.heads+h0+h)*WIDTH+d]:__float2half(0.f);
        s.q[h*STRIDE+d]=q;if(!isfinite(__half2float(q)))atomicOr(a.error,4u);
    }
    // Only skip trailing invalid slots, never collapse or reorder selected IDs.
    int end=0;
    for(int j=t;j<TOP;j+=THREADS){int id=a.ids[(int64_t)row*TOP+j];if(id>=0&&id<a.tokens&&id<=position)end=max(end,j+1);}
    for(int d=16;d;d>>=1)end=max(end,__shfl_down_sync(0xffffffff,end,d));
    if(l==0)s.last[w]=end;__syncthreads();
    if(t==0){s.end=0;for(int j=0;j<8;j++)s.end=max(s.end,s.last[j]);if(!s.end)atomicOr(a.error,16u);}
    __syncthreads();
    float acc[16][4];
    #pragma unroll
    for(int n=0;n<16;n++)for(int z=0;z<4;z++)acc[n][z]=0.f;
    float running_max=-INFINITY,running_sum=0.f;
    for(int base=0;base<s.end;base+=KEYS){
        if(t<KEYS)s.ids[t]=a.ids[(int64_t)row*TOP+base+t];
        __syncthreads();
        for(int i=t;i<KEYS*WIDTH;i+=THREADS){
            int j=i/WIDTH,d=i%WIDTH,id=s.ids[j];float x=0.f;
            if(id>=0&&id<a.tokens&&id<=position){
                const uint8_t* p=a.records+(int64_t)id*656;
                if(d<512){__nv_fp8_e4m3 v;v.__x=p[d];x=float(v)*reinterpret_cast<const float*>(p+512)[d/128];}
                else x=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p+528)[d-512]);
            }
            half v=__float2half_rn(x);s.key[j*STRIDE+d]=v;
            if(!isfinite(__half2float(v)))atomicOr(a.error,4u);
        }
        __syncthreads();
        float score[4]={0,0,0,0};
        #pragma unroll
        for(int k=0;k<WIDTH;k+=16){
            const half* q=s.q+(hm+g)*STRIDE+k+2*c;
            const half* key=s.key+(kb+g)*STRIDE+k+2*c;
            mma(score,word(q),word(q+8*STRIDE),word(q+8),word(q+8*STRIDE+8),word(key),word(key+8));
        }
        *reinterpret_cast<float2*>(s.score+(hm+g)*KEYS+kb+2*c)=make_float2(score[0]/16.f,score[1]/16.f);
        *reinterpret_cast<float2*>(s.score+(hm+g+8)*KEYS+kb+2*c)=make_float2(score[2]/16.f,score[3]/16.f);
        __syncthreads();
        // Eight consecutive threads per head, four contiguous selected keys each.
        int sh=t/8,j0=(t%8)*4;float x[4],local_max=-INFINITY;
        #pragma unroll
        for(int j=0;j<4;j++){
            int id=s.ids[j0+j];x[j]=(id>=0&&id<a.tokens&&id<=position)?s.score[sh*KEYS+j0+j]:-INFINITY;
            if(id>=0&&id<a.tokens&&id<=position&&!isfinite(x[j]))atomicOr(a.error,4u);
            local_max=fmaxf(local_max,x[j]);
        }
        for(int d=4;d;d>>=1)local_max=fmaxf(local_max,__shfl_xor_sync(0xffffffff,local_max,d,8));
        float next_max=fmaxf(running_max,local_max);
        float alpha=isfinite(running_max)?expf(running_max-next_max):0.f,partial=0.f;
        #pragma unroll
        for(int j=0;j<4;j++){
            float e=isfinite(x[j])?expf(x[j]-next_max):0.f;
            partial+=e;s.p[sh*PSTRIDE+j0+j]=__float2half_rn(e);
        }
        for(int d=4;d;d>>=1)partial+=__shfl_xor_sync(0xffffffff,partial,d,8);
        running_sum=running_sum*alpha+partial;running_max=next_max;
        if(t%8==0){s.alpha[sh]=alpha;s.denominator[sh]=running_sum;}
        __syncthreads();
        const float al=s.alpha[hm+g],ah=s.alpha[hm+g+8];
        #pragma unroll
        for(int n=0;n<16;n++){
            acc[n][0]*=al;acc[n][1]*=al;acc[n][2]*=ah;acc[n][3]*=ah;
        }
        #pragma unroll
        for(int k=0;k<KEYS;k+=16){
            const half* p=s.p+(hm+g)*PSTRIDE+k+2*c;
            const uint32_t a0=word(p),a1=word(p+8*PSTRIDE),a2=word(p+8),a3=word(p+8*PSTRIDE+8);
            #pragma unroll
            for(int n=0;n<16;n++){
                int d=(w%4)*128+n*8+g;const half* v=s.key+(k+2*c)*STRIDE+d;
                mma(acc[n],a0,a1,a2,a3,pair(v[0],v[STRIDE]),pair(v[8*STRIDE],v[9*STRIDE]));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int n=0;n<16;n++){
        #pragma unroll
        for(int z=0;z<4;z++){
            int h=h0+hm+g+(z/2)*8,d=(w%4)*128+n*8+2*c+z%2;
            float den=s.end?s.denominator[hm+g+(z/2)*8]:0.f;
            float value=den>0?acc[n][z]/den:0.f;
            if(h<a.heads){if(!isfinite(value))atomicOr(a.error,4u);a.context[((int64_t)row*a.heads+h)*VALUE+d]=value;}
        }
    }
}
inline cudaError_t configure(){return cudaFuncSetAttribute(core,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));}
inline cudaError_t run(const Args& a,cudaStream_t stream){
    if(a.rows<=0||a.heads<=0||a.heads>64||a.tokens<=0||!a.q||!a.records||!a.ids||!a.positions||!a.context||!a.error)return cudaErrorInvalidValue;
    core<<<a.rows*((a.heads+31)/32),THREADS,sizeof(Shared),stream>>>(a);
    return cudaGetLastError();
}
} // namespace gqo
