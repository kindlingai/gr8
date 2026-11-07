// Standalone CUDA implementation of the v2 fabric contract. Include globally.
#pragma once
#include "ds41_v2.h"
#include "ds41_v2_timing.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ds41::v2 {
inline void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr,"DS41 v2: %s\n",why); std::abort(); }
}
inline void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr,"DS41 v2 CUDA: %s\n",cudaGetErrorString(e)); std::abort(); }
}
namespace detail {
using Seq = unsigned long long;
struct Segments { Segment s[MAX_SEGMENTS]; WireLayout w; };
__device__ inline char* send(g6::FabricDev f, Seq epoch, int c) {
    return f.send + g6::fab_send_off(f,epoch&1,c);
}
__device__ inline const char* peer(g6::FabricDev f, Seq epoch, int c, int rank) {
    return rank == f.rank ? send(f,epoch,c) : f.recv + g6::fab_recv_off(f,epoch&1,c,rank);
}
__device__ inline Seq acquire(const Seq* p) {
    Seq x; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(x) : "l"(p) : "memory"); return x;
}
__global__ void publish(g6::FabricDev f, Seq epoch, int c) {
    // Separate stream-ordered launch after every pack CTA has exited.
    if (!threadIdx.x) {
        __threadfence_system();
        Seq* p = f.doorbell+c; const Seq v=g6::coll_seq(epoch,c,f.ncoll);
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
    }
}
__global__ void wait(g6::FabricDev f, Seq epoch, int c, size_t bytes) {
    const int p=threadIdx.x;
    if (p >= f.nr || p == f.rank) return;
    const Seq seq=g6::coll_seq(epoch,c,f.ncoll);
    const auto* tail=reinterpret_cast<const Seq*>(peer(f,epoch,c,p)+bytes);
    const long long start=clock64();
    unsigned it=0;
    while (acquire(tail)!=seq) {
        if ((++it&1023u)==0 && (*reinterpret_cast<volatile unsigned*>(f.error) || clock64()-start>20000000000LL)) {
            atomicCAS(f.error,0u,0x79000000u | (unsigned(c)<<8) | unsigned(p)); return;
        }
    }
}
// Copies one 16-byte wire unit, zero filling the part outside the valid rows.
// Irregular rows (e.g. Q odd with 24-byte routing entries) use the byte path
// only when the segment slice base is unaligned; common widths vectorize.
__device__ inline void put16(char* dst,const char* src,size_t i,size_t valid) {
    if (i+16<=valid && ((reinterpret_cast<uintptr_t>(src+i)|reinterpret_cast<uintptr_t>(dst))&15)==0) {
        *reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(src+i); return;
    }
    for (int b=0;b<16;++b) dst[b]=i+b<valid ? src[i+b] : 0;
}
__device__ inline void get16(char* dst,const char* src,size_t i,size_t valid) {
    if (i>=valid) return;
    if (i+16<=valid && ((reinterpret_cast<uintptr_t>(src)|reinterpret_cast<uintptr_t>(dst+i))&15)==0) {
        *reinterpret_cast<uint4*>(dst+i)=__ldcv(reinterpret_cast<const uint4*>(src)); return;
    }
    for (int b=0;b<16 && i+b<valid;++b) dst[i+b]=src[b];
}
__global__ void pack(g6::FabricDev f,Seq epoch,int c,Geometry g,Segments seg,bool scatter) {
    const size_t first=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;
    const size_t step=size_t(gridDim.x)*blockDim.x*16;
    for (int p=0;p<(scatter?g.nr:1);++p) {
        const int row0=scatter?p*g.Q:0;
        const int valid_rows=scatter?max(0,min(g.Q,g.M-row0)):g.owned;
        char* target=send(f,epoch,c)+(scatter?p*seg.w.stride:0);
        for(int s=0;s<seg.w.count;++s) {
            const auto& v=seg.s[s];
            const size_t n=size_t(g.Q)*v.row_bytes, valid=size_t(valid_rows)*v.row_bytes;
            const char* src=static_cast<const char*>(v.src)+size_t(row0)*v.row_bytes;
            for(size_t i=first;i<n;i+=step) put16(target+seg.w.offsets[s]+i,src,i,valid);
        }
    }
}
__global__ void unpack(g6::FabricDev f,Seq epoch,int c,Geometry g,Segments seg,bool scatter) {
    const size_t first=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;
    const size_t step=size_t(gridDim.x)*blockDim.x*16;
    for(int p=0;p<g.nr;++p) {
        const char* source=peer(f,epoch,c,p)+(scatter && p==g.rank ? size_t(g.rank)*seg.w.stride:0);
        const int valid_rows=scatter?g.owned:max(0,min(g.Q,g.M-p*g.Q));
        for(int s=0;s<seg.w.count;++s) {
            const auto& v=seg.s[s];
            const size_t n=size_t(valid_rows)*v.row_bytes;
            char* dst=static_cast<char*>(v.dst)+size_t(p)*g.Q*v.row_bytes;
            for(size_t i=first;i<n;i+=step) get16(dst,source+seg.w.offsets[s]+i,i,n);
        }
    }
}
__global__ void reduce(g6::FabricDev f,Seq epoch,int c,Geometry g,size_t stride,int cols,__nv_bfloat16* out) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(g.owned)*cols;i+=size_t(gridDim.x)*blockDim.x) {
        float sum=0.f;
        for(int p=0;p<g.nr;++p) {
            const char* base=peer(f,epoch,c,p)+(p==g.rank?size_t(g.rank)*stride:0);
            sum+=__bfloat162float(__ldcv(reinterpret_cast<const __nv_bfloat16*>(base)+i));
        }
        out[i]=__float2bfloat16_rn(sum);
    }
}
inline Segments segments(const Geometry& g,const Segment* src,int count) {
    Segments p{}; p.w=wire_layout(g.Q,src,count);
    require(p.w.count>0,"invalid segment layout");
    for(int i=0;i<count;++i) { require(src[i].src && src[i].dst,"null segment pointer"); p.s[i]=src[i]; }
    return p;
}
inline void enqueue(State& s,Coll coll,const WireLayout& w,bool scatter) {
    const int c=int(coll);
    require(s.live && s.order.take(coll),"collective out of layer order");
    const size_t capacity=scatter?size_t(s.g.nr)*w.stride:w.payload+8;
    require(capacity<=g6::fab_slot_stride(s.wire,c),"collective payload exceeds dedicated slot");
    std::vector<uint32_t> bytes(size_t(s.wire.ncoll),0);
    bytes[size_t(c)]=uint32_t(w.payload);
    g6::Fabric::Rs rs; rs.order={c};
    if(scatter) { rs.mask.set(c); rs.ss=w.stride; rs.bytes.assign(size_t(s.g.nr),uint32_t(w.payload)); }
    s.fabric->enqueue_step(s.epoch,bytes,rs);
}
} // namespace detail

inline void init(State& s,g6::Fabric& f,cudaStream_t st) {
    require(!s.live,"state initialized twice");
    s.fabric=&f; s.wire=f.dev(); s.st=st;
    require(s.wire.ncoll==int(Coll::Count),"v2 fabric collective count");
    int lo=0,hi=0; check(cudaDeviceGetStreamPriorityRange(&lo,&hi));
    check(cudaStreamCreateWithPriority(&s.cs,cudaStreamNonBlocking,hi));
    for(int c=0;c<int(Coll::Count);++c) {
        check(cudaEventCreateWithFlags(&s.produced[c],cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&s.received[c],cudaEventDisableTiming));
    }
    s.live=true;
    timing_init(s,knob::ds_pf_v2() && knob::ds_pf_v2_timing());
}
inline void release(State& s) {
    if(!s.live) return;
    check(cudaStreamSynchronize(s.st)); check(cudaStreamSynchronize(s.cs));
    require(s.fabric->wait_proxy_idle(),"v2 proxy did not drain");
    for(int c=0;c<int(Coll::Count);++c) { check(cudaEventDestroy(s.produced[c])); check(cudaEventDestroy(s.received[c])); }
    timing_release(s);
    check(cudaStreamDestroy(s.cs)); s.live=false;
}
inline void begin_layer(State& s,Geometry g,bool engram,bool index,bool tail,bool candidates) {
    require(s.live && s.order.complete(),"previous layer has unissued collectives");
    require(g.M>0 && g.nr==s.wire.nr && g.rank==s.wire.rank,"geometry does not match fabric");
    s.g=g; ++s.epoch; s.order={layer_order(engram,index,tail,candidates),0};
}
inline void end_layer(State& s) { require(s.order.complete(),"layer missing a collective"); }
inline void fork(State& s,Coll c) {
    check(cudaEventRecord(s.produced[int(c)],s.st)); check(cudaStreamWaitEvent(s.cs,s.produced[int(c)],0));
}
inline void record_received(State& s,Coll c) { check(cudaEventRecord(s.received[int(c)],s.cs)); }
inline void join(State& s,Coll c) {
    // group names the joined Coll id (AttnOut=7, Engram=6, Ffn3=11).
    TimingSpan timer(s,TimingStage::Join,s.st,int(c));
    check(cudaStreamWaitEvent(s.st,s.received[int(c)],0));
}
inline void all_gather(State& s,Coll c,const Segment* segments,int count,cudaStream_t st) {
    auto p=detail::segments(s.g,segments,count); detail::enqueue(s,c,p.w,false);
    const auto stage=c==Coll::AttnIn?TimingStage::AttnInAg:c==Coll::Index?TimingStage::IndexAg:
        c==Coll::Candidates?TimingStage::CandidatesAg:c==Coll::FfnIn?TimingStage::FfnAg:TimingStage::TailAg;
    TimingSpan timer(s,stage,st); // boot gathers have no active timing chunk
    detail::pack<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,false);
    detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
    detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
    detail::unpack<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,false);
    check(cudaPeekAtLastError());
}
inline void all_to_all(State& s,Coll c,const Segment* segments,int count,cudaStream_t st) {
    auto p=detail::segments(s.g,segments,count); detail::enqueue(s,c,p.w,true);
    TimingSpan timer(s,c==Coll::AttnOut?TimingStage::AttnA2A:TimingStage::EngramA2A,st);
    detail::pack<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
    detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
    detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
    detail::unpack<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
    check(cudaPeekAtLastError());
}
inline void reduce_scatter(State& s,int group,const __nv_bfloat16* all,__nv_bfloat16* own,int cols,cudaStream_t st) {
    require(group>=0 && group<COLUMN_GROUPS && cols>0,"invalid reduce-scatter group");
    Segment seg{all,own,size_t(cols)*2}; auto p=detail::segments(s.g,&seg,1); const Coll c=ffn_group(group);
    detail::enqueue(s,c,p.w,true);
    TimingSpan timer(s,TimingStage::Rs,st,group);
    detail::pack<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
    detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
    detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
    detail::reduce<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p.w.stride,cols,own);
    check(cudaPeekAtLastError());
}
inline void gather_weight(State& s,const void* source,size_t bytes,void* output) {
    require(s.live && bytes && source && output && s.order.complete(),"invalid boot weight gather");
    const size_t block=std::min(size_t(1)<<20,g6::fab_slot_stride(s.wire,int(Coll::Boot))-256);
    require(block>0,"no boot gather slot capacity");
    char* tmp=nullptr; check(cudaMalloc(&tmp,block*s.wire.nr));
    for(size_t off=0;off<bytes;off+=block) {
        const size_t n=std::min(block,bytes-off);
        s.g=geometry(s.wire.nr,s.wire.nr,s.wire.rank); ++s.epoch;
        s.order={{int(Coll::Boot)},0};
        Segment seg{static_cast<const char*>(source)+off,tmp,n};
        all_gather(s,Coll::Boot,&seg,1,s.st);
        check(cudaMemcpy2DAsync(static_cast<char*>(output)+off,bytes,tmp,n,n,s.wire.nr,cudaMemcpyDeviceToDevice,s.st));
        // A boot-only fence protects temporary storage and outstanding local RDMA
        // reads. Request-time collectives have no host fence.
        check(cudaStreamSynchronize(s.st));
        require(s.fabric->wait_proxy_idle(),"weight gather proxy drain");
    }
    check(cudaFree(tmp));
}
} // namespace ds41::v2
