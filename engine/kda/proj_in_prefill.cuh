// KDA prefill q/k/v projections: shared decoded B ring with dense K4 addressing.
// 4 producer warps decode each 16x16 B tile once; 16 consumer warps retain baseline
// 4 M groups x 4 N groups and the original K16 accumulation order.
#pragma once
#include "../kernels/bind_prefill.cuh"
namespace kda_proj_ws {
using namespace bindp;
constexpr int BM=16, NA=4, NB=2, NW=4, LA=2, LW=2;
constexpr int SA=2*BM*512, SW=16*128, SB=16*512;
constexpr int SMEM=NA*SA+NW*SW+NB*SB, THREADS=640;
__device__ __forceinline__ void sync(int id) {
 asm volatile("bar.sync %0, %1;"::"r"(id),"r"(THREADS):"memory");
}
__device__ __forceinline__ void arrive(int id) {
 asm volatile("bar.arrive %0, %1;"::"r"(id),"r"(THREADS):"memory");
}
__global__ void __launch_bounds__(THREADS,1) kernel(Gemm g) {
 extern __shared__ __align__(16) unsigned char sm[];
 unsigned char *sA=sm, *sW=sA+NA*SA, *sB=sW+NW*SW;
 const int t=threadIdx.x,lane=t&31,warp=t>>5;
 const int NK=g.K/32,K16=g.K/16,RB=(g.M+15)/16;
 const int c0=blockIdx.x*128,rb0=blockIdx.y*BM,nb=min(BM,RB-rb0);
 const int cw=warp-4,wm=warp<4?-1:cw/4,wn=warp<4?0:cw%4,colw=wn*32;
 float acc[4][4][4]={};
 if(warp<4) {
  auto stage_a=[&](int k) {
   unsigned char* a=sA+(k%NA)*SA;
   #pragma unroll
   for(int i=0;i<8;++i) {
    const int c=t+128*i,j=c/(BM*32),rem=c%(BM*32),mb=rem>>5,ch=rem&31;
    if(mb<nb) pf::cp16(a+c*16,g.x8+(static_cast<size_t>(rb0+mb)*K16+2*k+j)*512+ch*16);
   }
  };
  auto stage_w=[&](int k) {
   // Each warp owns four whole 128-byte tiles. All lanes issue a 16-byte copy.
   const int v=warp*4+(lane>>3),ks=2*k+(v>>3),nt=v&7;
   pf::cp16(sW+(k%NW)*SW+v*128+(lane&7)*16,
    g.tw+blockIdx.x*g.tsB+(ks>>3)*g.tsG+((ks&7)*8+nt)*128+(lane&7)*16);
  };
  Tr3Lane<4,true> tl;tl.init(lane);
  #pragma unroll
  for(int k=0;k<LA;++k) {
   if(k<NK) stage_a(k);
   if(k==0) {
    #pragma unroll
    for(int i=0;i<LW;++i) if(i<NK) stage_w(i);
   }
   pf::cp_commit();
  }
  for(int k=0;k<NK;++k) {
   if(k>=NB) sync(1+NB+k%NB);
   if(k+LA<NK) stage_a(k+LA);
   if(k+LW<NK) stage_w(k+LW);
   pf::cp_commit();
   pf::cp_wait<LA>();
   __syncwarp();
   #pragma unroll
   for(int j=0;j<4;++j) {
    const int tile=warp*4+j;
    const u32* raw=reinterpret_cast<const u32*>(sW+(k%NW)*SW+tile*128);
    u32 f0[2],f1[2];tl.decode(raw[tl.i0],raw[tl.i1],f0,f1);
    *reinterpret_cast<uint4*>(sB+(k%NB)*SB+tile*512+lane*16)=make_uint4(f0[0],f0[1],f1[0],f1[1]);
   }
   arrive(1+k%NB);
  }
  pf::cp_wait<0>();
 } else {
  const bool full=nb==BM;
  for(int k=0;k<NK;++k) {
   sync(1+k%NB);
   const unsigned char *a=sA+(k%NA)*SA,*b=sB+(k%NB)*SB;
   #pragma unroll
   for(int j=0;j<2;++j) {
    const uint4 f0=*reinterpret_cast<const uint4*>(b+(j*8+wn*2)*512+lane*16);
    const uint4 f1=*reinterpret_cast<const uint4*>(b+(j*8+wn*2+1)*512+lane*16);
    #pragma unroll
    for(int mb=0;mb<4;++mb) {
     const int m=wm*4+mb;if(!full&&m>=nb)continue;
     const uint4 fa=*reinterpret_cast<const uint4*>(a+(j*BM+m)*512+lane*16);
     mma16816(acc[mb][0],fa.x,fa.y,fa.z,fa.w,f0.x,f0.y);
     mma16816(acc[mb][1],fa.x,fa.y,fa.z,fa.w,f0.z,f0.w);
     mma16816(acc[mb][2],fa.x,fa.y,fa.z,fa.w,f1.x,f1.y);
     mma16816(acc[mb][3],fa.x,fa.y,fa.z,fa.w,f1.z,f1.w);
    }
   }
   if(k+NB<NK) arrive(1+NB+k%NB);
  }
 }
 // Producers must join the baseline rotation's CTA barriers. All 20 warps share
 // its output row work; wm=-1 prevents producers from publishing accumulators.
 if(g.osvh) {
  gemm_rot_store_f32<BM,4,20>(acc,sm,g.osvh,c0,rb0,nb,g.M,wm,colw,
   [&](int row,int col,const float (&v)[4]) {
    *reinterpret_cast<float4*>(static_cast<float*>(g.y)+static_cast<size_t>(row)*g.ldy+col)=make_float4(v[0],v[1],v[2],v[3]);
   });
  return;
 }
 if(warp<4)return;
 const int c=lane&3,r=lane>>2;
 #pragma unroll
 for(int mb=0;mb<4;++mb) {
  const int m=wm*4+mb;if(m>=nb)continue;
  #pragma unroll
  for(int nt=0;nt<4;++nt) {
   const int col=c0+colw+nt*8+2*c,row=(rb0+m)*16+r;
   float* y=static_cast<float*>(g.y);
   if(row<g.M)*reinterpret_cast<float2*>(y+static_cast<size_t>(row)*g.ldy+col)=make_float2(acc[mb][nt][0],acc[mb][nt][1]);
   if(row+8<g.M)*reinterpret_cast<float2*>(y+static_cast<size_t>(row+8)*g.ldy+col)=make_float2(acc[mb][nt][2],acc[mb][nt][3]);
  }
 }
}

// Default on for supported FP32 matrices. Explicit 0 selects the original path;
// small or unsupported matrices continue to use it automatically.
inline bool enabled() {
 static const bool on = [] { const char* v = getenv("G6_PF_KDA_WS"); return !v || atoi(v) == 1; }();
 return on;
}
inline bool supported(const Gemm& g) {
 return g.on && g.M >= 256 && g.h16 == 1 && g.tw && g.bf16 == 2 &&
        g.N > 0 && g.N % 128 == 0 && g.K > 0 && g.K % 128 == 0 && !g.w2;
}
inline cudaError_t launch(const Gemm& g, cudaStream_t st) {
 static const cudaError_t attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
 if (attr != cudaSuccess) return attr;
 bindp::launch(g.pre, st);
 kernel<<<dim3(g.N / 128, (g.M + 255) / 256), THREADS, SMEM, st>>>(g);
 return cudaGetLastError();
}
} // namespace kda_proj_ws
