#pragma once
// Standalone R1 only: async 16B record gather and packed exact dequantization.
// Math/chunk/softmax/MMA order remains the frozen v1 implementation.
#include "prefill-qo-proto.cuh"
#include "../kernels/sm70_compat.cuh"
#include <cstddef>
namespace gqo_r1 {
using gqo::Args;
using gqo::HEADS; using gqo::KEYS; using gqo::WIDTH; using gqo::VALUE;
using gqo::TOP; using gqo::THREADS; using gqo::STRIDE; using gqo::PSTRIDE;
using gqo::word; using gqo::pair; using gqo::mma;
struct alignas(16) Shared {
    half q[HEADS*STRIDE],key[KEYS*STRIDE];
    union Work {
        float score[HEADS*KEYS];
        struct PV {half p[HEADS*PSTRIDE];float alpha[HEADS],denominator[HEADS];} pv;
    } work;
    int ids[KEYS],last[8],end;
    alignas(16) uint8_t raw[KEYS*656];
};
static_assert(offsetof(Shared,raw)==79024,"approved common layout");
static_assert(sizeof(Shared)==100016,"approved R1 shared-memory size");

__device__ __forceinline__ void gather(uint8_t* raw,const uint8_t* __restrict__ records,
        const int* __restrict__ ids,int base,int tokens,int position){
    const int t=threadIdx.x,record=t/8,lane=t%8;
    const int id=ids[base+record];
    const bool valid=id>=0&&id<tokens&&id<=position;
    // Even a zero-fill instruction receives a valid, aligned source address.
    const uint8_t* src=records+(int64_t)(valid?id:0)*656;
    #pragma unroll
    for(int n=0;n<6;n++){
        const int byte=(n*8+lane)*16;
        if(byte<656){
            gr8_sm70::cp_async_cg_16_addr(unsigned(__cvta_generic_to_shared(raw+record*656+byte)), src+byte, valid?16:0);
        }
    }
    gr8_sm70::cp_async_commit();
}
__device__ __forceinline__ void store_pair(half* dst,half2 value,unsigned& errors){
    const __half2_raw raw=value;
    const unsigned u=unsigned(raw.x)|(unsigned(raw.y)<<16);
    *reinterpret_cast<unsigned*>(dst)=u;
    if((u&0x7c00u)==0x7c00u||(u&0x7c000000u)==0x7c000000u)errors|=4u;
}
__device__ __forceinline__ void dequant(Shared& s,unsigned& errors){
    const int record=threadIdx.x/8,lane=threadIdx.x%8;
    const uint8_t* raw=s.raw+record*656;
    // Exactly four FP32 scale loads per record; all eight lanes participate.
    const float own_scale=lane<4?reinterpret_cast<const float*>(raw+512)[lane]:0.f;
    #pragma unroll
    for(int block=0;block<4;block++){
        const float scale=__shfl_sync(0xffffffff,own_scale,block,8);
        #pragma unroll
        for(int n=0;n<8;n++){
            const int d=block*128+n*16+lane*2;
            const unsigned short bits=*reinterpret_cast<const unsigned short*>(raw+d);
            // E4M3 -> half is exact. Widen BEFORE multiplying by the FP32 scale.
            const half2 values=__nv_cvt_fp8x2_to_halfraw2(bits,__NV_E4M3);
            const float2 f=__half22float2(values);
            const half2 v=__floats2half2_rn(f.x*scale,f.y*scale);
            store_pair(s.key+record*STRIDE+d,v,errors);
        }
    }
    #pragma unroll
    for(int n=0;n<4;n++){
        const int d=n*16+lane*2;
        const __nv_bfloat162 b=*reinterpret_cast<const __nv_bfloat162*>(raw+528+2*d);
        const float2 f=__bfloat1622float2(b);
        store_pair(s.key+record*STRIDE+512+d,__floats2half2_rn(f.x,f.y),errors);
    }
}
__global__ __launch_bounds__(THREADS) void core(Args a){
    extern __shared__ __align__(16) unsigned char storage[];
    Shared& s=*reinterpret_cast<Shared*>(storage);
    const int groups=(a.heads+31)/32,row=blockIdx.x/groups,h0=(blockIdx.x%groups)*32;
    const int t=threadIdx.x,w=t/32,l=t%32,g=l/4,c=l%4,hm=(w/4)*16,kb=(w%4)*8;
    const half* __restrict__ query=a.q;
    const uint8_t* __restrict__ records=a.records;
    const int* __restrict__ selected=a.ids;
    float* __restrict__ output=a.context;
    const int position=a.positions[row];
    unsigned errors=0;
    for(int i=t;i<HEADS*WIDTH;i+=THREADS){
        int h=i/WIDTH,d=i%WIDTH;half q=h0+h<a.heads?query[((int64_t)row*a.heads+h0+h)*WIDTH+d]:__float2half(0.f);
        s.q[h*STRIDE+d]=q;if(!isfinite(__half2float(q)))errors|=4u;
    }
    // Only skip trailing invalid slots, never collapse or reorder selected IDs.
    int end=0;
    for(int j=t;j<TOP;j+=THREADS){int id=selected[(int64_t)row*TOP+j];if(id>=0&&id<a.tokens&&id<=position)end=max(end,j+1);}
    for(int d=16;d;d>>=1)end=max(end,__shfl_down_sync(0xffffffff,end,d));
    if(l==0)s.last[w]=end;__syncthreads();
    if(t==0){s.end=0;for(int j=0;j<8;j++)s.end=max(s.end,s.last[j]);if(!s.end)errors|=16u;}
    __syncthreads();
    float acc[16][4];
    #pragma unroll
    for(int n=0;n<16;n++)for(int z=0;z<4;z++)acc[n][z]=0.f;
    float running_max=-INFINITY,running_sum=0.f;
    if(s.end)gather(s.raw,records,selected+(int64_t)row*TOP,0,a.tokens,position);
    for(int base=0;base<s.end;base+=KEYS){
        // CURRENT IDs stay live until softmax. NEXT IDs are only gather registers.
        if(t<KEYS)s.ids[t]=selected[(int64_t)row*TOP+base+t];
        gr8_sm70::cp_async_wait<0>();
        __syncthreads();
        dequant(s,errors);
        // All raw reads complete before the next gather can overwrite the stage.
        __syncthreads();
        if(base+KEYS<s.end)gather(s.raw,records,selected+(int64_t)row*TOP,base+KEYS,a.tokens,position);
        float score[4]={0,0,0,0};
        #pragma unroll
        for(int k=0;k<WIDTH;k+=16){
            const half* q=s.q+(hm+g)*STRIDE+k+2*c;
            const half* key=s.key+(kb+g)*STRIDE+k+2*c;
            mma(score,word(q),word(q+8*STRIDE),word(q+8),word(q+8*STRIDE+8),word(key),word(key+8));
        }
        *reinterpret_cast<float2*>(s.work.score+(hm+g)*KEYS+kb+2*c)=make_float2(score[0]/16.f,score[1]/16.f);
        *reinterpret_cast<float2*>(s.work.score+(hm+g+8)*KEYS+kb+2*c)=make_float2(score[2]/16.f,score[3]/16.f);
        __syncthreads();
        // Eight consecutive threads per head, four contiguous selected keys each.
        int sh=t/8,j0=(t%8)*4;float x[4],local_max=-INFINITY;
        #pragma unroll
        for(int j=0;j<4;j++){
            int id=s.ids[j0+j];x[j]=(id>=0&&id<a.tokens&&id<=position)?s.work.score[sh*KEYS+j0+j]:-INFINITY;
            if(id>=0&&id<a.tokens&&id<=position&&!isfinite(x[j]))errors|=4u;
            local_max=fmaxf(local_max,x[j]);
        }
        // Score and P/alpha/den alias. Every thread must finish its four reads
        // before ANY thread stores P (P stride40, score stride32).
        __syncthreads();
        for(int d=4;d;d>>=1)local_max=fmaxf(local_max,__shfl_xor_sync(0xffffffff,local_max,d,8));
        float next_max=fmaxf(running_max,local_max);
        float alpha=isfinite(running_max)?expf(running_max-next_max):0.f,partial=0.f;
        #pragma unroll
        for(int j=0;j<4;j++){
            float e=isfinite(x[j])?expf(x[j]-next_max):0.f;
            partial+=e;s.work.pv.p[sh*PSTRIDE+j0+j]=__float2half_rn(e);
        }
        for(int d=4;d;d>>=1)partial+=__shfl_xor_sync(0xffffffff,partial,d,8);
        running_sum=running_sum*alpha+partial;running_max=next_max;
        if(t%8==0){s.work.pv.alpha[sh]=alpha;s.work.pv.denominator[sh]=running_sum;}
        __syncthreads();
        const float al=s.work.pv.alpha[hm+g],ah=s.work.pv.alpha[hm+g+8];
        #pragma unroll
        for(int n=0;n<16;n++){
            acc[n][0]*=al;acc[n][1]*=al;acc[n][2]*=ah;acc[n][3]*=ah;
        }
        #pragma unroll
        for(int k=0;k<KEYS;k+=16){
            const half* p=s.work.pv.p+(hm+g)*PSTRIDE+k+2*c;
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
            float den=s.end?s.work.pv.denominator[hm+g+(z/2)*8]:0.f;
            float value=den>0?acc[n][z]/den:0.f;
            if(h<a.heads){if(!isfinite(value))errors|=4u;output[((int64_t)row*a.heads+h)*VALUE+d]=value;}
        }
    }
    // Error tracking is thread-local in the hot loops; one CTA-end atomic only.
    errors=gr8_sm70::reduce_or_sync(0xffffffff,errors);
    if(l==0)s.last[w]=int(errors);
    __syncthreads();
    if(t==0){unsigned all=0;for(int i=0;i<8;i++)all|=unsigned(s.last[i]);if(all)atomicOr(a.error,all);}

}
inline cudaError_t configure(){return cudaFuncSetAttribute(core,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));}
inline cudaError_t run(const Args& a,cudaStream_t stream){
    if(a.rows<=0||a.heads<=0||a.heads>64||a.tokens<=0||!a.q||!a.records||!a.ids||!a.positions||!a.context||!a.error)return cudaErrorInvalidValue;
    gqo_r1::core<<<a.rows*((a.heads+31)/32),THREADS,sizeof(Shared),stream>>>(a);
    return cudaGetLastError();
}
} // namespace gqo_r1
