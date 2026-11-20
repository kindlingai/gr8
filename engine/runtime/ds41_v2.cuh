// Standalone CUDA implementation of the v2 fabric contract. Include globally.
#pragma once
#include "ds41_v2.h"
#include "ds41_v2_timing.cuh"
#include "../comms/xtrace.h"
#include "../kernels/glue.cuh"
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
// The slot and doorbell helpers are the shared fabric ones (kernels/glue.cuh).
__device__ inline char* send(g6::FabricDev f, Seq epoch, int c) { return glue::send_slot(f,epoch,c); }
__device__ inline const char* peer(g6::FabricDev f, Seq epoch, int c, int rank) {
    return glue::peer_slot(f,epoch,c,rank);
}
// Trace stamps: the shared xtrace record of (epoch, c) (glue::xt_rec). Only the
// traced specializations stamp from here; the shared publish/wait stamp
// wherever the fabric has xtrace storage (f.xt), which is the traced case.
__device__ inline Seq trace_now() { return glue::xt_now(); }
__device__ inline Seq* trace_record(g6::FabricDev f,Seq epoch,int c) { return reinterpret_cast<Seq*>(glue::xt_rec(f,epoch,c)); }
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
// Separate stream-ordered launches: publish after every pack CTA has exited,
// wait on one CTA. Trace=true runs where the fabric was built with xtrace
// (f.xt set), which is where the shared helpers stamp.
template<bool Trace=false>
__global__ void publish(g6::FabricDev f, Seq epoch, int c) {
    glue::publish_after(f,c,g6::coll_seq(epoch,c,f.ncoll));
}
template<bool Trace=false>
__global__ void wait(g6::FabricDev f, Seq epoch, int c, size_t bytes) {
    if constexpr(Trace) {
        if(!threadIdx.x)trace_record(f,epoch,c)[g6::XT_WAIT_BEGIN]=trace_now();
    }
    glue::wait_peers(f,epoch,c,g6::coll_seq(epoch,c,f.ncoll),bytes,0x79000000u);
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
// Direct remote-quarter packing; the local projection reads codes/scales directly.
__global__ void __launch_bounds__(COMM_THREADS) pack_attn_out_remote(
    g6::FabricDev f, Seq epoch, int c, Geometry g,
    const uint8_t* __restrict__ oaq, const uint8_t* __restrict__ oas, int K, size_t w_stride) {
    const int ks=K/32, row_bytes=K+ks;
    const size_t n=size_t(g.Q)*row_bytes;
    const size_t first=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;
    const size_t step=size_t(gridDim.x)*blockDim.x*16;
    const bool aligned16=((K|ks)&15)==0 &&
        ((reinterpret_cast<uintptr_t>(oaq)|reinterpret_cast<uintptr_t>(oas))&15)==0;
    for(int p=0;p<g.nr;++p) {
        if(p==g.rank)continue;
        const int row0=p*g.Q, valid_rows=max(0,min(g.Q,g.M-row0));
        char* target=send(f,epoch,c)+size_t(p)*w_stride;
        if(aligned16 && (reinterpret_cast<uintptr_t>(target)&15)==0) {
            const size_t row_u4=size_t(row_bytes)/16;
            for(size_t i=first;i<n;i+=step) {
                const size_t u=i/16, row=u/row_u4, col=(u%row_u4)*16;
                uint4 value=make_uint4(0,0,0,0);
                if(row<size_t(valid_rows)) {
                    const uint8_t* source=col<size_t(K)
                        ? oaq+size_t(row0+int(row))*K+col
                        : oas+size_t(row0+int(row))*ks+col-size_t(K);
                    value=*reinterpret_cast<const uint4*>(source);
                }
                *reinterpret_cast<uint4*>(target+i)=value;
            }
        } else {
            for(size_t i=first;i<n;i+=step)for(int b=0;b<16 && i+b<n;++b) {
                const size_t offset=i+b,row=offset/size_t(row_bytes),col=offset%size_t(row_bytes);
                target[offset]=row<size_t(valid_rows)
                    ? char(col<size_t(K) ? oaq[size_t(row0+int(row))*K+col]
                                         : oas[size_t(row0+int(row))*ks+col-size_t(K)]) : char(0);
            }
        }
    }
}
template<bool Trace=false>
__global__ void __launch_bounds__(COMM_THREADS) unpack_remote(
    g6::FabricDev f, Seq epoch, int c, Geometry g, Segments seg) {
    const size_t first=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;
    const size_t step=size_t(gridDim.x)*blockDim.x*16;
    for(int p=0;p<g.nr;++p) {
        if(p==g.rank)continue;
        const char* source=peer(f,epoch,c,p);
        for(int k=0;k<seg.w.count;++k) {
            const auto& v=seg.s[k];
            const size_t n=size_t(g.owned)*v.row_bytes;
            char* dst=static_cast<char*>(v.dst)+size_t(p)*g.Q*v.row_bytes;
            for(size_t i=first;i<n;i+=step)get16(dst,source+seg.w.offsets[k]+i,i,n);
        }
    }
    if constexpr(Trace)trace_done(f,epoch,c);
}
template<bool Trace=false>
__global__ void __launch_bounds__(COMM_THREADS) unpack_ffn_in(
    g6::FabricDev f, Seq epoch, int c, Geometry g, Segments seg,
    uint8_t* __restrict__ record_q, int H) {
    const int lane_c = threadIdx.x & 3;
    const size_t quad_id = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 2;
    const size_t quad_step = (size_t(gridDim.x) * blockDim.x) >> 2;
    const size_t first = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    const size_t step = size_t(gridDim.x) * blockDim.x * 16;
    const size_t off0 = seg.w.offsets[0];
    char* const base_all = static_cast<char*>(seg.s[0].dst);
    for (int p = 0; p < g.nr; ++p) {
        const char* source = peer(f, epoch, c, p);
        const int valid_rows = max(0, min(g.Q, g.M - p * g.Q));
        {
            const size_t peer_off = size_t(p) * size_t(g.Q) * size_t(H);
            const char* src0 = source + off0;
            char* dst_all = base_all + peer_off;
            uint8_t* dst_rec = record_q + peer_off;
            const size_t blocks64 = size_t(valid_rows) * size_t(H) / 64;
            for (size_t block = quad_id; block < blocks64; block += quad_step) {
                const uint32_t quad_mask = 0xfu << (threadIdx.x & 28);
                const size_t off = block * 64 + size_t(lane_c) * 16;
                const uint4 value = __ldcv(reinterpret_cast<const uint4*>(src0 + off));
                *reinterpret_cast<uint4*>(dst_all + off) = value;
                const uint32_t p0 = (value.x & 0xffffu) | ((value.z & 0xffffu) << 16);
                const uint32_t p1 = (value.x >> 16)     | (value.z & 0xffff0000u);
                const uint32_t p2 = (value.y & 0xffffu) | ((value.w & 0xffffu) << 16);
                const uint32_t p3 = (value.y >> 16)     | (value.w & 0xffff0000u);
                const uint32_t u0 = lane_c == 0 ? p0 : lane_c == 1 ? p1 : lane_c == 2 ? p2 : p3;
                const uint32_t s1 = lane_c == 0 ? p1 : lane_c == 1 ? p0 : lane_c == 2 ? p3 : p2;
                const uint32_t s2 = lane_c == 0 ? p2 : lane_c == 1 ? p3 : lane_c == 2 ? p0 : p1;
                const uint32_t s3 = lane_c == 0 ? p3 : lane_c == 1 ? p2 : lane_c == 2 ? p1 : p0;
                const uint32_t u1 = __shfl_xor_sync(quad_mask, s1, 1);
                const uint32_t u2 = __shfl_xor_sync(quad_mask, s2, 2);
                const uint32_t u3 = __shfl_xor_sync(quad_mask, s3, 3);
                const uint32_t w0 = lane_c == 0 ? u0 : lane_c == 1 ? u1 : lane_c == 2 ? u2 : u3;
                const uint32_t w1 = lane_c == 0 ? u1 : lane_c == 1 ? u0 : lane_c == 2 ? u3 : u2;
                const uint32_t w2 = lane_c == 0 ? u2 : lane_c == 1 ? u3 : lane_c == 2 ? u0 : u1;
                const uint32_t w3 = lane_c == 0 ? u3 : lane_c == 1 ? u2 : lane_c == 2 ? u1 : u0;
                *reinterpret_cast<uint4*>(dst_rec + off) = make_uint4(w0, w1, w2, w3);
            }
        }
        for (int k = 1; k < seg.w.count; ++k) {
            const auto& v = seg.s[k];
            const size_t n = size_t(valid_rows) * v.row_bytes;
            char* dst = static_cast<char*>(v.dst) + size_t(p) * g.Q * v.row_bytes;
            for (size_t i = first; i < n; i += step) get16(dst, source + seg.w.offsets[k] + i, i, n);
        }
    }
    if constexpr (Trace) trace_done(f, epoch, c);
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
    s.l20_tail=L20Tail{};
    require(s.l20_tail.init(knob::ds_pf_v2_l20_tail()), "GR8_DS_PF_V2_L20_TAIL needs 0 or 1 entries");
    s.send_ffn_opt=SendFfnOpt{};
    require(s.send_ffn_opt.init(knob::ds_pf_v2_send_ffn_opt()), "GR8_DS_PF_V2_SEND_FFN_OPT needs modes 0..3");
    s.bf16_tc = Bf16Tc{};
    require(s.bf16_tc.init(knob::ds_pf_v2_bf16_tc()), "GR8_DS_PF_V2_BF16_TC needs masks 0..7");
    s.attn_pipe=AttnPipe{};
    s.attn_pipe.on=knob::ds_pf_v2_attn_pipe();
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
inline void begin_l20_tail(State& s,Geometry g,bool tail) {
    require(s.live && s.order.complete(),"previous layer has unissued collectives");
    require(g.M>0 && g.nr==s.wire.nr && g.rank==s.wire.rank,"geometry does not match fabric");
    s.g=g; ++s.epoch; s.order={l20_tail_order(tail),0};
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
inline void all_gather_ffn_in(State& s,const Segment* input,int count,uint8_t* record_q,int H,cudaStream_t st) {
    require(input && record_q && count>=1 && H>0 && H%64==0 && input[0].row_bytes==size_t(H),
            "invalid all_gather_ffn_in args");
    const uintptr_t alignment=reinterpret_cast<uintptr_t>(input[0].dst)|reinterpret_cast<uintptr_t>(record_q)|
        reinterpret_cast<uintptr_t>(s.wire.send)|reinterpret_cast<uintptr_t>(s.wire.recv);
    require((alignment&15)==0,"unaligned all_gather_ffn_in buffers");
    auto p=detail::segments(s.g,input,count);
    detail::enqueue(s,Coll::FfnIn,p.w,false);
    const int c=int(Coll::FfnIn), blocks=std::max(s.widths.pack_blocks,32);
    TimingSpan timer(s,TimingStage::FfnAg,st);
    if(s.wire.xt) {
        detail::trace_produced<<<1,32,0,st>>>(s.wire,s.epoch,c);
        TimingDetailSpan packed(s,TimingStage::PackPublish,st,c);
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,c,s.g,p,false);
        detail::publish<true><<<1,32,0,st>>>(s.wire,s.epoch,c);
        packed.finish();
        TimingDetailSpan waited(s,TimingStage::Wait,st,c);
        detail::wait<true><<<1,32,0,st>>>(s.wire,s.epoch,c,p.w.payload);
        waited.finish();
        TimingDetailSpan consumed(s,TimingStage::Unpack,st,c);
        detail::unpack_ffn_in<true><<<blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,c,s.g,p,record_q,H);
    } else {
        detail::pack<<<s.widths.pack_blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,c,s.g,p,false);
        detail::publish<<<1,32,0,st>>>(s.wire,s.epoch,c);
        detail::wait<<<1,32,0,st>>>(s.wire,s.epoch,c,p.w.payload);
        detail::unpack_ffn_in<<<blocks,COMM_THREADS,0,st>>>(s.wire,s.epoch,c,s.g,p,record_q,H);
    }
    check(cudaPeekAtLastError());
}
inline void attn_out_a2a_opt(State& s,const uint8_t* oaq,const uint8_t* oas,int K,void* output_recv) {
    require(oaq && oas && output_recv && K>0 && K%32==0,"invalid attn_out_a2a_opt args");
    Segment seg{oaq,output_recv,size_t(K)+size_t(K/32)};
    auto p=detail::segments(s.g,&seg,1);
    detail::enqueue(s,Coll::AttnOut,p.w,true);
    const int c=int(Coll::AttnOut), blocks=std::max(s.widths.pack_blocks,32);
    if(s.wire.xt) {
        detail::trace_produced<<<1,32,0,s.st>>>(s.wire,s.epoch,c);
        TimingDetailSpan packed(s,TimingStage::PackPublish,s.st,c);
        detail::pack_attn_out_remote<<<blocks,COMM_THREADS,0,s.st>>>(s.wire,s.epoch,c,s.g,oaq,oas,K,p.w.stride);
        detail::publish<true><<<1,32,0,s.st>>>(s.wire,s.epoch,c);
    } else {
        detail::pack_attn_out_remote<<<blocks,COMM_THREADS,0,s.st>>>(s.wire,s.epoch,c,s.g,oaq,oas,K,p.w.stride);
        detail::publish<<<1,32,0,s.st>>>(s.wire,s.epoch,c);
    }
    check(cudaEventRecord(s.produced[c],s.st));
    check(cudaStreamWaitEvent(s.cs,s.produced[c],0));
    {
        TimingSpan timer(s,TimingStage::AttnA2A,s.cs);
        if(s.wire.xt) {
            TimingDetailSpan waited(s,TimingStage::Wait,s.cs,c);
            detail::wait<true><<<1,32,0,s.cs>>>(s.wire,s.epoch,c,p.w.payload);
            waited.finish();
            TimingDetailSpan consumed(s,TimingStage::Unpack,s.cs,c);
            detail::unpack_remote<true><<<blocks,COMM_THREADS,0,s.cs>>>(s.wire,s.epoch,c,s.g,p);
        } else {
            detail::wait<<<1,32,0,s.cs>>>(s.wire,s.epoch,c,p.w.payload);
            detail::unpack_remote<<<blocks,COMM_THREADS,0,s.cs>>>(s.wire,s.epoch,c,s.g,p);
        }
    }
    record_received(s,Coll::AttnOut);
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
