// Standalone CUDA implementation of the v2 fabric contract. Include globally.
#pragma once
#include "ds41_v2.h"
#include "ds41_v2_timing.cuh"
#include "../comms/xtrace.h"
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
// Only the traced specialization reads clocks or writes trace storage. The
// default specialization retains the original kernels; no device-side branch.
__device__ inline Seq trace_now() {
    Seq t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ inline Seq* trace_record(g6::FabricDev f,Seq epoch,int c) {
    return f.xt+((epoch%g6::XT_STEPS)*f.ncoll+c)*g6::XT_W;
}
__global__ void trace_produced(g6::FabricDev f,Seq epoch,int c) {
    if(threadIdx.x)return;
    auto* r=trace_record(f,epoch,c);
    for(int i=0;i<g6::XT_W;++i)r[i]=0; // including stale optional peers/joins on wrap
    r[g6::XT_PRODUCED]=trace_now();
}
__global__ void trace_stamp(g6::FabricDev f,Seq epoch,int c,int stage) {
    if(!threadIdx.x)trace_record(f,epoch,c)[stage]=trace_now();
}
__device__ inline void trace_done(g6::FabricDev f,Seq epoch,int c) {
    __syncthreads(); // traced only: last writer in every CTA, not only thread 0
    if(!threadIdx.x)atomicMax(trace_record(f,epoch,c)+g6::XT_RED,trace_now());
}
template<bool Trace=false>
__global__ void publish(g6::FabricDev f, Seq epoch, int c) {
    // Separate stream-ordered launch after every pack CTA has exited.
    if (!threadIdx.x) {
        if constexpr(Trace)trace_record(f,epoch,c)[g6::XT_PUB]=trace_now();
        __threadfence_system();
        Seq* p = f.doorbell+c; const Seq v=g6::coll_seq(epoch,c,f.ncoll);
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
        if constexpr(Trace) {
            auto* r=trace_record(f,epoch,c); r[g6::XT_DB]=trace_now(); r[g6::XT_SEQ]=v;
        }
    }
}
template<bool Trace=false>
__global__ void wait(g6::FabricDev f, Seq epoch, int c, size_t bytes) {
    if constexpr(Trace) {
        if(!threadIdx.x)trace_record(f,epoch,c)[g6::XT_WAIT_BEGIN]=trace_now();
    }
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
    if constexpr(Trace)trace_record(f,epoch,c)[g6::XT_TAIL+p]=trace_now();
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
template<bool Trace=false>
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
    if constexpr(Trace)trace_done(f,epoch,c);
}
template<bool Trace=false>
__global__ void reduce(g6::FabricDev f,Seq epoch,int c,Geometry g,size_t stride,int cols,__nv_bfloat16* out) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(g.owned)*cols;i+=size_t(gridDim.x)*blockDim.x) {
        float sum=0.f;
        for(int p=0;p<g.nr;++p) {
            const char* base=peer(f,epoch,c,p)+(p==g.rank?size_t(g.rank)*stride:0);
            sum+=__bfloat162float(__ldcv(reinterpret_cast<const __nv_bfloat16*>(base)+i));
        }
        out[i]=__float2bfloat16_rn(sum);
    }
    if constexpr(Trace)trace_done(f,epoch,c);
}
// The opt-in path retains reduce<> above verbatim. __ldcv(uint4*) emits
// ld.global.cv.v4.u32, the same cache policy as the scalar bf16 __ldcv:
// peer/fabric data is not read through a stale cached global-load path.
// Publication/wait ordering remains the original separate kernel chain.
template<bool Trace=false>
__global__ void reduce_vector(g6::FabricDev f,Seq epoch,int c,Geometry g,size_t stride,int cols,__nv_bfloat16* out) {
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t step=size_t(gridDim.x)*blockDim.x, n=size_t(g.owned)*cols;
    uintptr_t alignment=reinterpret_cast<uintptr_t>(out);
    for(int p=0;p<g.nr;++p)
        alignment|=reinterpret_cast<uintptr_t>(peer(f,epoch,c,p)+(p==g.rank?size_t(g.rank)*stride:0));
    const size_t full=(alignment&15)?0:n/8*8;
    for(size_t i=tid*8;i<full;i+=step*8) {
        float sums[8]={};
        for(int p=0;p<g.nr;++p) {
            const char* base=peer(f,epoch,c,p)+(p==g.rank?size_t(g.rank)*stride:0);
            const uint4 v=__ldcv(reinterpret_cast<const uint4*>(base+i*2));
            const unsigned words[4]={v.x,v.y,v.z,v.w};
            #pragma unroll
            for(int j=0;j<8;++j)
                sums[j]+=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(words[j/2]>>(16*(j%2)))));
        }
        unsigned words[4];
        #pragma unroll
        for(int j=0;j<4;++j)
            words[j]=unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(sums[2*j]))) |
                (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(sums[2*j+1])))<<16);
        *reinterpret_cast<uint4*>(out+i)=make_uint4(words[0],words[1],words[2],words[3]);
    }
    // Usually fewer than eight elements; also the safe unaligned fallback.
    for(size_t i=full+tid;i<n;i+=step) {
        float sum=0.f;
        for(int p=0;p<g.nr;++p) {
            const char* base=peer(f,epoch,c,p)+(p==g.rank?size_t(g.rank)*stride:0);
            sum+=__bfloat162float(__ldcv(reinterpret_cast<const __nv_bfloat16*>(base)+i));
        }
        out[i]=__float2bfloat16_rn(sum);
    }
    if constexpr(Trace)trace_done(f,epoch,c);
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
    s.widths=CommWidths{};
    require(CommWidths::decode(knob::ds_pf_v2_rs_blocks(),s.widths.rs),
        "GR8_DS_PF_V2_RS_BLOCKS needs 0 or positive int block counts");
    require(CommWidths::decode(knob::ds_pf_v2_pack_blocks(),s.widths.pack),
        "GR8_DS_PF_V2_PACK_BLOCKS needs 0 or positive int block counts");
    require(s.wire.ncoll==int(Coll::Count),"v2 fabric collective count");
    int lo=0,hi=0; check(cudaDeviceGetStreamPriorityRange(&lo,&hi));
    check(cudaStreamCreateWithPriority(&s.cs,cudaStreamNonBlocking,hi));
    for(int c=0;c<int(Coll::Count);++c) {
        check(cudaEventCreateWithFlags(&s.produced[c],cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&s.received[c],cudaEventDisableTiming));
    }
    s.live=true;
    timing_init(s,knob::ds_pf_v2() && (knob::ds_pf_v2_timing() || s.wire.xt));
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
    if(s.wire.xt && s.timing)
        std::fprintf(stderr,"# ds41_v2_xtrace rank=%d epoch=%llu request=%llu chunk=%u rows=%d layer=%d\n",
            s.wire.rank,(unsigned long long)s.epoch,(unsigned long long)s.timing->request,
            s.timing->chunk,g.M,s.timing->layer);
}
inline void end_layer(State& s) { require(s.order.complete(),"layer missing a collective"); }
inline void fork(State& s,Coll c) {
    if(s.wire.xt)detail::trace_produced<<<1,32,0,s.st>>>(s.wire,s.epoch,int(c));
    check(cudaEventRecord(s.produced[int(c)],s.st)); check(cudaStreamWaitEvent(s.cs,s.produced[int(c)],0));
}
inline void record_received(State& s,Coll c) { check(cudaEventRecord(s.received[int(c)],s.cs)); }
inline void join(State& s,Coll c) {
    // group names the joined Coll id (AttnOut=7, Engram=6, Ffn3=11).
    TimingSpan timer(s,TimingStage::Join,s.st,int(c));
    if(s.wire.xt)detail::trace_stamp<<<1,32,0,s.st>>>(s.wire,s.epoch,int(c),g6::XT_JOIN_BEGIN);
    check(cudaStreamWaitEvent(s.st,s.received[int(c)],0));
    if(s.wire.xt)detail::trace_stamp<<<1,32,0,s.st>>>(s.wire,s.epoch,int(c),g6::XT_JOIN_END);
}
inline void all_gather(State& s,Coll c,const Segment* segments,int count,cudaStream_t st) {
    auto p=detail::segments(s.g,segments,count); detail::enqueue(s,c,p.w,false);
    const auto stage=c==Coll::AttnIn?TimingStage::AttnInAg:c==Coll::Index?TimingStage::IndexAg:
        c==Coll::Candidates?TimingStage::CandidatesAg:c==Coll::FfnIn?TimingStage::FfnAg:TimingStage::TailAg;
    TimingSpan timer(s,stage,st); // boot gathers have no active timing chunk
    if(s.wire.xt) {
        detail::trace_produced<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        TimingDetailSpan packed(s,TimingStage::PackPublish,st,int(c));
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,false);
        detail::publish<true><<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        packed.finish();
        TimingDetailSpan waited(s,TimingStage::Wait,st,int(c));
        detail::wait<true><<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
        waited.finish();
        TimingDetailSpan consumed(s,TimingStage::Unpack,st,int(c));
        detail::unpack<true><<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,false);
    } else {
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,false);
        detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
        detail::unpack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,false);
    }
    check(cudaPeekAtLastError());
}
inline void all_to_all(State& s,Coll c,const Segment* segments,int count,cudaStream_t st) {
    auto p=detail::segments(s.g,segments,count); detail::enqueue(s,c,p.w,true);
    TimingSpan timer(s,c==Coll::AttnOut?TimingStage::AttnA2A:TimingStage::EngramA2A,st);
    if(s.wire.xt) {
        TimingDetailSpan packed(s,TimingStage::PackPublish,st,int(c));
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
        detail::publish<true><<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        packed.finish();
        TimingDetailSpan waited(s,TimingStage::Wait,st,int(c));
        detail::wait<true><<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
        waited.finish();
        TimingDetailSpan consumed(s,TimingStage::Unpack,st,int(c));
        detail::unpack<true><<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
    } else {
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
        detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
        detail::unpack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
    }
    check(cudaPeekAtLastError());
}
inline void reduce_scatter(State& s,int group,const __nv_bfloat16* all,__nv_bfloat16* own,int cols,cudaStream_t st) {
    require(group>=0 && group<COLUMN_GROUPS && cols>0,"invalid reduce-scatter group");
    Segment seg{all,own,size_t(cols)*2}; auto p=detail::segments(s.g,&seg,1); const Coll c=ffn_group(group);
    detail::enqueue(s,c,p.w,true);
    TimingSpan timer(s,TimingStage::Rs,st,group);
    if(s.wire.xt) {
        TimingDetailSpan packed(s,TimingStage::PackPublish,st,int(c));
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
        detail::publish<true><<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        packed.finish();
        TimingDetailSpan waited(s,TimingStage::Wait,st,int(c));
        detail::wait<true><<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
        waited.finish();
        TimingDetailSpan consumed(s,TimingStage::Reduce,st,int(c));
        if(s.widths.rs_blocks)
            detail::reduce_vector<true><<<s.widths.rs_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p.w.stride,cols,own);
        else
            detail::reduce<true><<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p.w.stride,cols,own);
    } else {
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p,true);
        detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,int(c));
        detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,int(c),p.w.payload);
        if(s.widths.rs_blocks)
            detail::reduce_vector<<<s.widths.rs_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p.w.stride,cols,own);
        else
            detail::reduce<<<COMM_BLOCKS,COMM_THREADS,0,st>>>(s.wire,s.epoch,int(c),s.g,p.w.stride,cols,own);
    }
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
