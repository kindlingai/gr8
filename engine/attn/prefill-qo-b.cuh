#pragma once
// QO variant B (synthesis R6): standalone, no production caller.
// prepass_b: once per layer, every cached 656 B record -> FP16 row [576] in a [T][576] table, v1's exact math
//   (FP32 E4M3 code x FP32 scale -> RN16; BF16 RoPE -> float -> RN16) plus a per-key nonfinite byte.
// core_b: v1's math, chunking and rounding (32-key online-softmax chunks, same QK/PV MMA chains and order,
//   FP32 max/sum, RN16 P) with no in-kernel dequant. Keys arrive by 16-B cp.async gathers of FP16 rows into a
//   3-slot ring of 16-key stages (96,496 B smem: leaves room for co-resident blocks), so the next chunk's first half loads during the current chunk's compute.
//   Operands come from ldmatrix (bit-identical fragments to v1's scalar loads). Error bits stay in registers,
//   one atomic per CTA at the end; v1's error semantics are kept (only selected valid keys are flagged).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../kernels/sm70_compat.cuh"
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>
#ifndef PF_QOBH
#define PF_QOBH 0
#endif
namespace gqob {
constexpr int HEADS=32, KEYS=32, HALF=16, WIDTH=576, VALUE=512, TOP=2048, THREADS=256, CHUNKS16=WIDTH/8;
constexpr int ROW=WIDTH, PSTRIDE=40, SLOTS=3, PER_HALF=HALF*CHUNKS16; // 1152 16-B copies per 16-key stage
struct Args {
    const half* __restrict__ q;          // [M,H,576]
    const half* __restrict__ table;      // [T,576] FP16 key view from prepass_b
    const uint8_t* __restrict__ bad;     // [T] 1 = row has a nonfinite RN16 value
    const int* __restrict__ ids;         // [M,2048]
    const int* __restrict__ positions;   // [M]
    float* __restrict__ context;         // [M,H,512]
    unsigned* error;                     // bit4 nonfinite, bit16 no valid selected key
    int rows,heads,tokens;
};
#if PF_QOBH
// Opt-in served output boundary; the existing FP32 Args/API stays unchanged.
struct HalfArgs {
    const half* __restrict__ q;
    const half* __restrict__ table;
    const uint8_t* __restrict__ bad;
    const int* __restrict__ ids;
    const int* __restrict__ positions;
    half* __restrict__ context;
    unsigned* error;
    int rows,heads,tokens;
};
#endif
struct alignas(16) Shared {
    half q[HEADS*ROW], key[SLOTS*HALF*ROW]; // q/key rows unpadded, 16-B chunks XOR-swizzled
    // score is dead once softmax has it in registers (barrier), then the same bytes hold P/alpha/denominator.
    union { float score[HEADS*KEYS]; struct { half p[HEADS*PSTRIDE]; float alpha[HEADS], denominator[HEADS]; } pv; } u;
    int kid[SLOTS*HALF], last[8], end;
};
static_assert(sizeof(Shared)<=101376,"one-CTA shared memory limit");
__device__ __forceinline__ uint32_t saddr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
// Byte offset of 16-B chunk ch of row r: chunk ch^(r&7), so 8 consecutive rows hit 8 distinct bank groups.
__device__ __forceinline__ uint32_t sw(int r,int ch){return uint32_t(r*CHUNKS16+(ch^(r&7)))*16u;}
__device__ __forceinline__ void mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    gr8_sm70::mma16816_f16(*reinterpret_cast<float (*)[4]>(c), a0, a1, a2, a3, b0, b1);
}
__device__ __forceinline__ void ldsm4(uint32_t* r,uint32_t a){
    gr8_sm70::ldmatrix_x4_addr(r[0],r[1],r[2],r[3],a);
}
__device__ __forceinline__ void ldsm4t(uint32_t* r,uint32_t a){
    gr8_sm70::ldmatrix_x4_trans_addr(r[0],r[1],r[2],r[3],a);
}
__device__ __forceinline__ void cp16(uint32_t dst,const void* src,int bytes){
    gr8_sm70::cp_async_cg_16_addr(dst,src,bytes);
}
__device__ __forceinline__ void commit(){gr8_sm70::cp_async_commit();}
__device__ __forceinline__ void wait1(){gr8_sm70::cp_async_wait<1>();}

// 288 threads = 4 keys x 72 16-B output chunks.
__global__ __launch_bounds__(288) void prepass_b(const uint8_t* __restrict__ records,half* __restrict__ table,uint8_t* __restrict__ bad,int tokens){
    __shared__ int flag[4];
    const int t=threadIdx.x,kk=t/CHUNKS16,ch=t%CHUNKS16,key=blockIdx.x*4+kk;
    if(t<4)flag[t]=0;
    __syncthreads();
    if(key<tokens){
        const uint8_t* p=records+(int64_t)key*656;
        half v[8];
        if(ch<64){
            const uint2 raw=*reinterpret_cast<const uint2*>(p+ch*8);
            const float scale=reinterpret_cast<const float*>(p+512)[ch/16];
            const uint8_t* b=reinterpret_cast<const uint8_t*>(&raw);
            #pragma unroll
            for(int j=0;j<8;j++){__nv_fp8_e4m3 e;e.__x=b[j];v[j]=__float2half_rn(float(e)*scale);}
        }else{
            const uint4 raw=*reinterpret_cast<const uint4*>(p+528+(ch-64)*16);
            const __nv_bfloat16* b=reinterpret_cast<const __nv_bfloat16*>(&raw);
            #pragma unroll
            for(int j=0;j<8;j++)v[j]=__float2half_rn(__bfloat162float(b[j]));
        }
        bool nf=false;
        #pragma unroll
        for(int j=0;j<8;j++)nf|=!isfinite(__half2float(v[j]));
        if(nf)flag[kk]=1;
        *reinterpret_cast<uint4*>(table+(int64_t)key*WIDTH+ch*8)=*reinterpret_cast<const uint4*>(v);
    }
    __syncthreads();
    if(t<4&&blockIdx.x*4+t<tokens)bad[blockIdx.x*4+t]=(uint8_t)flag[t];
}

template<bool SPLIT>
__global__ __launch_bounds__(THREADS,2) void core_b(Args a){ // minBlocks 2 caps regs at 128 (room for co-resident blocks)
    extern __shared__ __align__(16) unsigned char storage[];
    Shared& s=*reinterpret_cast<Shared*>(storage);
    const int groups=(a.heads+31)/32,row=blockIdx.x/groups,h0=(blockIdx.x%groups)*32;
    const int t=threadIdx.x,w=t/32,l=t%32,g=l/4,c=l%4,hm=(w/4)*16,kb=(w%4)*8;
    const int position=a.positions[row];
    unsigned err=0;
    for(int i=t;i<HEADS*CHUNKS16;i+=THREADS){
        const int h=i/CHUNKS16,d=(i%CHUNKS16)*8;
        uint4 v=make_uint4(0,0,0,0);
        if(h0+h<a.heads)v=*reinterpret_cast<const uint4*>(a.q+((int64_t)row*a.heads+h0+h)*WIDTH+d);
        const half* e=reinterpret_cast<const half*>(&v);
        #pragma unroll
        for(int j=0;j<8;j++)if(!isfinite(__half2float(e[j])))err|=4u;
        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(s.q)+sw(h,d/8))=v;
    }
    int end=0;
    for(int j=t;j<TOP;j+=THREADS){int id=a.ids[(int64_t)row*TOP+j];if(id>=0&&id<a.tokens&&id<=position)end=max(end,j+1);}
    for(int d=16;d;d>>=1)end=max(end,__shfl_down_sync(0xffffffff,end,d));
    if(l==0)s.last[w]=end;__syncthreads();
    if(t==0){s.end=0;for(int j=0;j<8;j++)s.end=max(s.end,s.last[j]);if(!s.end)err|=16u;}
    __syncthreads();
    const int chunks=(s.end+KEYS-1)/KEYS,halves=2*chunks;
    const int* rid=a.ids+(int64_t)row*TOP;
    // Copy i = t + 256r (r < 4, or r == 4 for t < 128) owns stage key i/72, 16-B column i%72.
    int nid[5];uint8_t pend[5];
    #pragma unroll
    for(int r=0;r<5;r++){pend[r]=0;const int i=t+THREADS*r;nid[r]=(i<PER_HALF&&halves>0)?rid[i/CHUNKS16]:-1;}
    auto issue=[&](int hidx){
        if(hidx<halves){
            const int slot=hidx%SLOTS;
            #pragma unroll
            for(int r=0;r<5;r++){
                const int i=t+THREADS*r;
                if(i<PER_HALF){
                    const int key=i/CHUNKS16,ch=i%CHUNKS16,id=nid[r];
                    const bool ok=id>=0&&id<a.tokens&&id<=position;
                    cp16(saddr(s.key)+sw(slot*HALF+key,ch),a.table+(int64_t)(ok?id:0)*WIDTH+ch*8,ok?16:0);
                    if(ch==0){s.kid[slot*HALF+key]=id;err|=pend[r];pend[r]=ok?a.bad[id]:0;}
                }
            }
            const int nh=hidx+1;
            #pragma unroll
            for(int r=0;r<5;r++){const int i=t+THREADS*r;nid[r]=(i<PER_HALF&&nh<halves)?rid[nh*HALF+i/CHUNKS16]:-1;}
        }
        commit();
    };
    issue(0);issue(1);issue(2);
    float acc[16][4];
    #pragma unroll
    for(int n=0;n<16;n++)for(int z=0;z<4;z++)acc[n][z]=0.f;
    float running_max=-INFINITY,running_sum=0.f;
    // ldmatrix lane addresses: A 16x16 (rows l%8 + 8*((l/8)%2), col 8*(l/16)); B non-trans keys l%8, col 8*(l/8).
    const int arow=(l%8)+((l/8)%2)*8,acol=(l/16)*8;
    // Swizzled byte offset of 16-B chunk ch in a row r with r&7 == l%8: (16ch)^(16(l%8)). Split into a constant
    // (folded into the ldmatrix immediate) plus one of a few per-thread XOR residues, so no per-load address math.
    const uint32_t m=l%8,c0=(l/8)^m,c1=(l/16)^m;
    const uint32_t kx0=16*c0,kx1=64^(16*c0);
    const uint32_t qx[4]={16*c1,32^(16*c1),64^(16*c1),96^(16*c1)};
    const uint32_t qs=saddr(s.q)+(hm+arow)*(WIDTH*2),ks=saddr(s.key),pa=saddr(s.u.pv.p+(hm+arow)*PSTRIDE+acol);
    for(int chunk=0;chunk<chunks;chunk++){
        const int sa=(2*chunk)%SLOTS,sb=(2*chunk+1)%SLOTS;
        wait1();
        __syncthreads();
        float score[4]={0,0,0,0};
        {
            const uint32_t kr=ks+((kb<16?sa:sb)*HALF+(kb%16)+(l%8))*(WIDTH*2);
            #pragma unroll
            for(int kk=0;kk<WIDTH/32;kk++){
                uint32_t b[4],q0[4],q1[4];
                ldsm4(b,kr+128*(kk>>1)+((kk&1)?kx1:kx0));
                ldsm4(q0,qs+128*((2*kk)>>2)+qx[(2*kk)&3]);ldsm4(q1,qs+128*((2*kk+1)>>2)+qx[(2*kk+1)&3]);
                mma(score,q0[0],q0[1],q0[2],q0[3],b[0],b[1]);
                mma(score,q1[0],q1[1],q1[2],q1[3],b[2],b[3]);
            }
        }
        *reinterpret_cast<float2*>(s.u.score+(hm+g)*KEYS+kb+2*c)=make_float2(score[0]/16.f,score[1]/16.f);
        *reinterpret_cast<float2*>(s.u.score+(hm+g+8)*KEYS+kb+2*c)=make_float2(score[2]/16.f,score[3]/16.f);
        __syncthreads();
        int sh=t/8,j0=(t%8)*4;float x[4],local_max=-INFINITY;
        #pragma unroll
        for(int j=0;j<4;j++){
            const int jj=j0+j,id=s.kid[(jj<16?sa:sb)*HALF+(jj%16)];
            const bool ok=id>=0&&id<a.tokens&&id<=position;
            x[j]=ok?s.u.score[sh*KEYS+jj]:-INFINITY;
            if(ok&&!isfinite(x[j]))err|=4u;
            local_max=fmaxf(local_max,x[j]);
        }
        __syncthreads();
        for(int d=4;d;d>>=1)local_max=fmaxf(local_max,__shfl_xor_sync(0xffffffff,local_max,d,8));
        float next_max=fmaxf(running_max,local_max);
        float alpha=isfinite(running_max)?expf(running_max-next_max):0.f,partial=0.f;
        #pragma unroll
        for(int j=0;j<4;j++){
            float e=isfinite(x[j])?expf(x[j]-next_max):0.f;
            partial+=e;s.u.pv.p[sh*PSTRIDE+j0+j]=__float2half_rn(e);
        }
        for(int d=4;d;d>>=1)partial+=__shfl_xor_sync(0xffffffff,partial,d,8);
        running_sum=running_sum*alpha+partial;running_max=next_max;
        if(t%8==0){s.u.pv.alpha[sh]=alpha;s.u.pv.denominator[sh]=running_sum;}
        __syncthreads();
        const float al=s.u.pv.alpha[hm+g],ah=s.u.pv.alpha[hm+g+8];
        #pragma unroll
        for(int n=0;n<16;n++){
            acc[n][0]*=al;acc[n][1]*=al;acc[n][2]*=ah;acc[n][3]*=ah;
        }
        #pragma unroll
        for(int k=0;k<KEYS;k+=16){
            uint32_t pf[4];ldsm4(pf,pa+k*2);
            const uint32_t vr=ks+((k?sb:sa)*HALF+arow)*(WIDTH*2)+(w%4)*256;
            #pragma unroll
            for(int n=0;n<16;n+=2){
                uint32_t b[4];ldsm4t(b,vr+128*(n>>3)+qx[(n/2)&3]);
                mma(acc[n],pf[0],pf[1],pf[2],pf[3],b[0],b[1]);
                mma(acc[n+1],pf[0],pf[1],pf[2],pf[3],b[2],b[3]);
            }
            if(SPLIT&&!k){__syncthreads();issue(2*chunk+3);}
        }
        __syncthreads();
        if(!SPLIT)issue(2*chunk+3);
        issue(2*chunk+4);
    }
    #pragma unroll
    for(int r=0;r<5;r++)err|=pend[r];
    #pragma unroll
    for(int n=0;n<16;n++){
        #pragma unroll
        for(int z=0;z<4;z++){
            int h=h0+hm+g+(z/2)*8,d=(w%4)*128+n*8+2*c+z%2;
            float den=s.end?s.u.pv.denominator[hm+g+(z/2)*8]:0.f;
            float value=den>0?acc[n][z]/den:0.f;
            if(h<a.heads){if(!isfinite(value))err|=4u;a.context[((int64_t)row*a.heads+h)*VALUE+d]=value;}
        }
    }
    for(int d=16;d;d>>=1)err|=__shfl_xor_sync(0xffffffff,err,d);
    __syncthreads();
    if(l==0)s.last[w]=(int)err;
    __syncthreads();
    if(t==0){unsigned e=0;for(int j=0;j<8;j++)e|=(unsigned)s.last[j];if(e)atomicOr(a.error,e);}
}
#if PF_QOBH
// Deliberately leave the FP32 kernel above intact, including flag-on builds.
// This candidate mirrors its body with only the final RN16 store changed.
// The FP32 value/nonfinite check precedes conversion: the old cast_half did
// not flag finite FP32 values that overflow only when converted to half.
template<bool SPLIT>
__global__ __launch_bounds__(THREADS,2) void core_b_half(HalfArgs a){ // minBlocks 2 caps regs at 128 (room for co-resident blocks)
    extern __shared__ __align__(16) unsigned char storage[];
    Shared& s=*reinterpret_cast<Shared*>(storage);
    const int groups=(a.heads+31)/32,row=blockIdx.x/groups,h0=(blockIdx.x%groups)*32;
    const int t=threadIdx.x,w=t/32,l=t%32,g=l/4,c=l%4,hm=(w/4)*16,kb=(w%4)*8;
    const int position=a.positions[row];
    unsigned err=0;
    for(int i=t;i<HEADS*CHUNKS16;i+=THREADS){
        const int h=i/CHUNKS16,d=(i%CHUNKS16)*8;
        uint4 v=make_uint4(0,0,0,0);
        if(h0+h<a.heads)v=*reinterpret_cast<const uint4*>(a.q+((int64_t)row*a.heads+h0+h)*WIDTH+d);
        const half* e=reinterpret_cast<const half*>(&v);
        #pragma unroll
        for(int j=0;j<8;j++)if(!isfinite(__half2float(e[j])))err|=4u;
        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(s.q)+sw(h,d/8))=v;
    }
    int end=0;
    for(int j=t;j<TOP;j+=THREADS){int id=a.ids[(int64_t)row*TOP+j];if(id>=0&&id<a.tokens&&id<=position)end=max(end,j+1);}
    for(int d=16;d;d>>=1)end=max(end,__shfl_down_sync(0xffffffff,end,d));
    if(l==0)s.last[w]=end;__syncthreads();
    if(t==0){s.end=0;for(int j=0;j<8;j++)s.end=max(s.end,s.last[j]);if(!s.end)err|=16u;}
    __syncthreads();
    const int chunks=(s.end+KEYS-1)/KEYS,halves=2*chunks;
    const int* rid=a.ids+(int64_t)row*TOP;
    // Copy i = t + 256r (r < 4, or r == 4 for t < 128) owns stage key i/72, 16-B column i%72.
    int nid[5];uint8_t pend[5];
    #pragma unroll
    for(int r=0;r<5;r++){pend[r]=0;const int i=t+THREADS*r;nid[r]=(i<PER_HALF&&halves>0)?rid[i/CHUNKS16]:-1;}
    auto issue=[&](int hidx){
        if(hidx<halves){
            const int slot=hidx%SLOTS;
            #pragma unroll
            for(int r=0;r<5;r++){
                const int i=t+THREADS*r;
                if(i<PER_HALF){
                    const int key=i/CHUNKS16,ch=i%CHUNKS16,id=nid[r];
                    const bool ok=id>=0&&id<a.tokens&&id<=position;
                    cp16(saddr(s.key)+sw(slot*HALF+key,ch),a.table+(int64_t)(ok?id:0)*WIDTH+ch*8,ok?16:0);
                    if(ch==0){s.kid[slot*HALF+key]=id;err|=pend[r];pend[r]=ok?a.bad[id]:0;}
                }
            }
            const int nh=hidx+1;
            #pragma unroll
            for(int r=0;r<5;r++){const int i=t+THREADS*r;nid[r]=(i<PER_HALF&&nh<halves)?rid[nh*HALF+i/CHUNKS16]:-1;}
        }
        commit();
    };
    issue(0);issue(1);issue(2);
    float acc[16][4];
    #pragma unroll
    for(int n=0;n<16;n++)for(int z=0;z<4;z++)acc[n][z]=0.f;
    float running_max=-INFINITY,running_sum=0.f;
    // ldmatrix lane addresses: A 16x16 (rows l%8 + 8*((l/8)%2), col 8*(l/16)); B non-trans keys l%8, col 8*(l/8).
    const int arow=(l%8)+((l/8)%2)*8,acol=(l/16)*8;
    // Swizzled byte offset of 16-B chunk ch in a row r with r&7 == l%8: (16ch)^(16(l%8)). Split into a constant
    // (folded into the ldmatrix immediate) plus one of a few per-thread XOR residues, so no per-load address math.
    const uint32_t m=l%8,c0=(l/8)^m,c1=(l/16)^m;
    const uint32_t kx0=16*c0,kx1=64^(16*c0);
    const uint32_t qx[4]={16*c1,32^(16*c1),64^(16*c1),96^(16*c1)};
    const uint32_t qs=saddr(s.q)+(hm+arow)*(WIDTH*2),ks=saddr(s.key),pa=saddr(s.u.pv.p+(hm+arow)*PSTRIDE+acol);
    for(int chunk=0;chunk<chunks;chunk++){
        const int sa=(2*chunk)%SLOTS,sb=(2*chunk+1)%SLOTS;
        wait1();
        __syncthreads();
        float score[4]={0,0,0,0};
        {
            const uint32_t kr=ks+((kb<16?sa:sb)*HALF+(kb%16)+(l%8))*(WIDTH*2);
            #pragma unroll
            for(int kk=0;kk<WIDTH/32;kk++){
                uint32_t b[4],q0[4],q1[4];
                ldsm4(b,kr+128*(kk>>1)+((kk&1)?kx1:kx0));
                ldsm4(q0,qs+128*((2*kk)>>2)+qx[(2*kk)&3]);ldsm4(q1,qs+128*((2*kk+1)>>2)+qx[(2*kk+1)&3]);
                mma(score,q0[0],q0[1],q0[2],q0[3],b[0],b[1]);
                mma(score,q1[0],q1[1],q1[2],q1[3],b[2],b[3]);
            }
        }
        *reinterpret_cast<float2*>(s.u.score+(hm+g)*KEYS+kb+2*c)=make_float2(score[0]/16.f,score[1]/16.f);
        *reinterpret_cast<float2*>(s.u.score+(hm+g+8)*KEYS+kb+2*c)=make_float2(score[2]/16.f,score[3]/16.f);
        __syncthreads();
        int sh=t/8,j0=(t%8)*4;float x[4],local_max=-INFINITY;
        #pragma unroll
        for(int j=0;j<4;j++){
            const int jj=j0+j,id=s.kid[(jj<16?sa:sb)*HALF+(jj%16)];
            const bool ok=id>=0&&id<a.tokens&&id<=position;
            x[j]=ok?s.u.score[sh*KEYS+jj]:-INFINITY;
            if(ok&&!isfinite(x[j]))err|=4u;
            local_max=fmaxf(local_max,x[j]);
        }
        __syncthreads();
        for(int d=4;d;d>>=1)local_max=fmaxf(local_max,__shfl_xor_sync(0xffffffff,local_max,d,8));
        float next_max=fmaxf(running_max,local_max);
        float alpha=isfinite(running_max)?expf(running_max-next_max):0.f,partial=0.f;
        #pragma unroll
        for(int j=0;j<4;j++){
            float e=isfinite(x[j])?expf(x[j]-next_max):0.f;
            partial+=e;s.u.pv.p[sh*PSTRIDE+j0+j]=__float2half_rn(e);
        }
        for(int d=4;d;d>>=1)partial+=__shfl_xor_sync(0xffffffff,partial,d,8);
        running_sum=running_sum*alpha+partial;running_max=next_max;
        if(t%8==0){s.u.pv.alpha[sh]=alpha;s.u.pv.denominator[sh]=running_sum;}
        __syncthreads();
        const float al=s.u.pv.alpha[hm+g],ah=s.u.pv.alpha[hm+g+8];
        #pragma unroll
        for(int n=0;n<16;n++){
            acc[n][0]*=al;acc[n][1]*=al;acc[n][2]*=ah;acc[n][3]*=ah;
        }
        #pragma unroll
        for(int k=0;k<KEYS;k+=16){
            uint32_t pf[4];ldsm4(pf,pa+k*2);
            const uint32_t vr=ks+((k?sb:sa)*HALF+arow)*(WIDTH*2)+(w%4)*256;
            #pragma unroll
            for(int n=0;n<16;n+=2){
                uint32_t b[4];ldsm4t(b,vr+128*(n>>3)+qx[(n/2)&3]);
                mma(acc[n],pf[0],pf[1],pf[2],pf[3],b[0],b[1]);
                mma(acc[n+1],pf[0],pf[1],pf[2],pf[3],b[2],b[3]);
            }
            if(SPLIT&&!k){__syncthreads();issue(2*chunk+3);}
        }
        __syncthreads();
        if(!SPLIT)issue(2*chunk+3);
        issue(2*chunk+4);
    }
    #pragma unroll
    for(int r=0;r<5;r++)err|=pend[r];
    #pragma unroll
    for(int n=0;n<16;n++){
        #pragma unroll
        for(int z=0;z<4;z++){
            int h=h0+hm+g+(z/2)*8,d=(w%4)*128+n*8+2*c+z%2;
            float den=s.end?s.u.pv.denominator[hm+g+(z/2)*8]:0.f;
            float value=den>0?acc[n][z]/den:0.f;
            if(h<a.heads){if(!isfinite(value))err|=4u;a.context[((int64_t)row*a.heads+h)*VALUE+d]=__float2half_rn(value);}
        }
    }
    for(int d=16;d;d>>=1)err|=__shfl_xor_sync(0xffffffff,err,d);
    __syncthreads();
    if(l==0)s.last[w]=(int)err;
    __syncthreads();
    if(t==0){unsigned e=0;for(int j=0;j<8;j++)e|=(unsigned)s.last[j];if(e)atomicOr(a.error,e);}
}
template<bool SPLIT>inline cudaError_t configure_half(){
    return cudaFuncSetAttribute(core_b_half<SPLIT>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));
}
template<bool SPLIT>inline cudaError_t run_half(const HalfArgs& a,cudaStream_t stream){
    if(a.rows<=0||a.heads<=0||a.heads>64||a.tokens<=0||!a.q||!a.table||!a.bad||!a.ids||!a.positions||!a.context||!a.error)return cudaErrorInvalidValue;
    core_b_half<SPLIT><<<a.rows*((a.heads+31)/32),THREADS,sizeof(Shared),stream>>>(a);
    return cudaGetLastError();
}
#endif
template<bool SPLIT>inline cudaError_t configure(){return cudaFuncSetAttribute(core_b<SPLIT>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));}
inline cudaError_t prepass(const uint8_t* records,half* table,uint8_t* bad,int tokens,cudaStream_t stream){
    prepass_b<<<(tokens+3)/4,288,0,stream>>>(records,table,bad,tokens);
    return cudaGetLastError();
}
template<bool SPLIT>inline cudaError_t run(const Args& a,cudaStream_t stream){
    if(a.rows<=0||a.heads<=0||a.heads>64||a.tokens<=0||!a.q||!a.table||!a.bad||!a.ids||!a.positions||!a.context||!a.error)return cudaErrorInvalidValue;
    core_b<SPLIT><<<a.rows*((a.heads+31)/32),THREADS,sizeof(Shared),stream>>>(a);
    return cudaGetLastError();
}
} // namespace gqob
