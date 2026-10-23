// Flash MTP: FP32 input norms and residual; BF16 weights. No mHC state.
#pragma once
#include "flash_mlp.cuh"
#if G6_MODEL_FLASH
namespace flash_mtp {
constexpr int H=geom::HID;
__device__ inline float sum256(float v,float* scratch) {
    for(int d=16;d;d>>=1) v+=__shfl_xor_sync(0xffffffffu,v,d);
    if(!(threadIdx.x&31)) scratch[threadIdx.x/32]=v;
    __syncthreads();
    v=0; for(int i=0;i<8;++i) v+=scratch[i];
    return v;
}
__global__ void k_input(const int* tok,const int* pos,const __nv_bfloat16* emb,
                         const __nv_bfloat16* enorm,const __nv_bfloat16* hnorm,
                         const float* hidden,float* cat) {
    __shared__ float scratch[8];
    const int row=blockIdx.y,half=blockIdx.x;
    float ss=0;
    for(int i=threadIdx.x;i<H;i+=256) {
        float v=half?hidden[(size_t)row*H+i]:__bfloat162float(emb[(size_t)tok[row]*H+i]);
        ss=fmaf(v,v,ss);
    }
    const float inv=rsqrtf(sum256(ss,scratch)/H+geom::RMS_EPS);
    for(int i=threadIdx.x;i<H;i+=256) {
        float v=half?hidden[(size_t)row*H+i]:__bfloat162float(emb[(size_t)tok[row]*H+i]);
        cat[((size_t)row*2+half)*H+i]=!half && pos[row]==0?0.f:v*inv*__bfloat162float((half?hnorm:enorm)[i]);
    }
}
__global__ void k_norm(const float* x,const __nv_bfloat16* w,float* y,__half* yh) {
    __shared__ float scratch[8];
    const size_t off=(size_t)blockIdx.x*H;
    float ss=0;
    for(int i=threadIdx.x;i<H;i+=256) ss=fmaf(x[off+i],x[off+i],ss);
    float inv=rsqrtf(sum256(ss,scratch)/H+geom::RMS_EPS);
    for(int i=threadIdx.x;i<H;i+=256) {
        const float v=x[off+i]*inv*__bfloat162float(w[i]);
        y[off+i]=v; if(yh) yh[off+i]=__float2half_rn(v);
    }
}
template<class Block> __global__ void k_add(float* resid,Block block,int n) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n) { block.init(); resid[i]+=block(i); }
}
inline void input(const int* tok,const int* pos,const __nv_bfloat16* emb,const __nv_bfloat16* enorm,
                  const __nv_bfloat16* hnorm,const float* hidden,float* cat,int M,cudaStream_t st) {
    k_input<<<dim3(2,M),256,0,st>>>(tok,pos,emb,enorm,hnorm,hidden,cat);
}
inline void norm(const float* x,const __nv_bfloat16* w,float* y,__half* yh,int M,cudaStream_t st) {
    k_norm<<<M,256,0,st>>>(x,w,y,yh);
}
}
#endif
