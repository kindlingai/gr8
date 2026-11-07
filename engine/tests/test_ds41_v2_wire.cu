// No model, network or verbs needed: emulate only the proxy copies between
// mapped slots, then test the actual v2 GPU pack/unpack/reduce kernels.
#include "../runtime/ds41_v2.cuh"
#include <cassert>
#include <iostream>
#include <vector>
using namespace ds41::v2;
struct Region {
    g6::FabricDev f{};
    Region(int rank,size_t stride,bool trace=false) {
        f.rank=rank; f.nr=4; f.ncoll=1; f.stride=stride; f.compact=0;
        check(cudaHostAlloc(&f.send,2*stride,cudaHostAllocMapped));
        check(cudaHostAlloc(&f.recv,8*stride,cudaHostAllocMapped));
        check(cudaHostAlloc(&f.error,4,cudaHostAllocMapped)); *f.error=0;
        check(cudaHostAlloc(&f.doorbell,8,cudaHostAllocMapped)); *f.doorbell=0;
        if(trace) { check(cudaMalloc(&f.xt,g6::XT_STEPS*g6::XT_W*sizeof(*f.xt)));
            check(cudaMemset(f.xt,0xff,g6::XT_STEPS*g6::XT_W*sizeof(*f.xt))); }
        std::memset(f.send,0,2*stride); std::memset(f.recv,0,8*stride);
    }
    ~Region() { if(f.xt)check(cudaFree(f.xt)); check(cudaFreeHost(f.send));check(cudaFreeHost(f.recv));check(cudaFreeHost(f.error));check(cudaFreeHost(f.doorbell)); }
};
struct Data {
    std::vector<unsigned char> input,output;
    unsigned char *src=nullptr,*dst=nullptr;
    Data(size_t in,size_t out):input(in),output(out,0xcd) { check(cudaMalloc(&src,in));check(cudaMalloc(&dst,out)); }
    ~Data(){check(cudaFree(src));check(cudaFree(dst));}
};
static unsigned char pattern(int rank,int row,int col) { return (rank*47+row*13+col*3)%251; }
static void proxy(Region* const (&r)[4],uint64_t epoch,const WireLayout& w,bool scatter) {
    for(int from=0;from<4;++from) for(int to=0;to<4;++to) if(from!=to) {
        const size_t src=g6::fab_send_off(r[from]->f,epoch&1,0)+(scatter?size_t(to)*w.stride:0);
        const size_t dst=g6::fab_recv_off(r[to]->f,epoch&1,0,from);
        std::memcpy(r[to]->f.recv+dst,r[from]->f.send+src,w.payload);
        *reinterpret_cast<unsigned long long*>(r[to]->f.recv+dst+w.payload)=g6::coll_seq(epoch,0,1);
    }
}
// Start from dirty records: every optional field must be cleared on reuse.
static void validate_trace(g6::FabricDev f,uint64_t epoch) {
    unsigned long long rec[g6::XT_W];
    const auto* src=f.xt+(epoch%g6::XT_STEPS)*g6::XT_W;
    check(cudaMemcpy(rec,src,sizeof(rec),cudaMemcpyDeviceToHost));
    assert(rec[g6::XT_SEQ]==g6::coll_seq(epoch,0,1));
    assert(rec[g6::XT_PRODUCED] && rec[g6::XT_PUB]>=rec[g6::XT_PRODUCED]);
    assert(rec[g6::XT_DB]>=rec[g6::XT_PUB]);
    for(int p=0;p<8;++p) {
        if(p==f.rank || p>=f.nr)assert(rec[g6::XT_TAIL+p]==0);
        else assert(rec[g6::XT_TAIL+p]>=rec[g6::XT_DB] && rec[g6::XT_TAIL+p]>=rec[g6::XT_WAIT_BEGIN]);
    }
    assert(rec[g6::XT_WAIT_BEGIN]>=rec[g6::XT_DB]);
    for(int p=0;p<f.nr;++p)assert(rec[g6::XT_RED]>=rec[g6::XT_TAIL+p]);
    assert(rec[g6::XT_JOIN_BEGIN]==0 && rec[g6::XT_JOIN_END]==0);
    detail::trace_stamp<<<1,32>>>(f,epoch,0,g6::XT_JOIN_BEGIN);
    detail::trace_stamp<<<1,32>>>(f,epoch,0,g6::XT_JOIN_END);
    check(cudaMemcpy(rec,src,sizeof(rec),cudaMemcpyDeviceToHost));
    assert(rec[g6::XT_JOIN_BEGIN]>=rec[g6::XT_RED] && rec[g6::XT_JOIN_END]>=rec[g6::XT_JOIN_BEGIN]);
    // Reuse precisely the same ring entry, without publishing a new seq yet.
    detail::trace_produced<<<1,32>>>(f,epoch+g6::XT_STEPS,0);
    check(cudaMemcpy(rec,src,sizeof(rec),cudaMemcpyDeviceToHost));
    for(int i=0;i<g6::XT_W;++i)assert(i==g6::XT_PRODUCED ? rec[i]>0 : rec[i]==0);
}
template<bool Trace=false>
static void test(int M,bool scatter) {
    const int Q=(M+3)/4;
    const size_t widths[]={24,17,160}; // includes unaligned row/slice boundaries
    Segment shape[3]; for(int s=0;s<3;++s)shape[s].row_bytes=widths[s];
    auto w=wire_layout(Q,shape,3);
    const size_t stride=4*w.stride;
    Region* net[4]; Data* data[4][3]; detail::Segments packed[4]{};
    for(int r=0;r<4;++r) {
        net[r]=new Region(r,stride,Trace); auto g=geometry(M,4,r); packed[r].w=w;
        for(int s=0;s<3;++s) {
            auto& d=data[r][s];
            d=new Data(size_t(scatter?M:std::max(1,g.owned))*widths[s],size_t(4*Q)*widths[s]);
            for(int m=0;m<(scatter?M:g.owned);++m)for(size_t j=0;j<widths[s];++j)
                d->input[size_t(m)*widths[s]+j]=pattern(r,m,int(j));
            check(cudaMemcpy(d->src,d->input.data(),d->input.size(),cudaMemcpyHostToDevice));
            check(cudaMemcpy(d->dst,d->output.data(),d->output.size(),cudaMemcpyHostToDevice));
            packed[r].s[s]={d->src,d->dst,widths[s]};
        }
        if constexpr(Trace)detail::trace_produced<<<1,32>>>(net[r]->f,1,0);
        detail::pack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,1,0,g,packed[r],scatter);
        detail::publish<Trace><<<1,32>>>(net[r]->f,1,0);
    }
    check(cudaDeviceSynchronize());
    proxy(net,1,w,scatter);
    for(int r=0;r<4;++r) {
        auto g=geometry(M,4,r);
        detail::wait<Trace><<<1,32>>>(net[r]->f,1,0,w.payload);
        detail::unpack<Trace><<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,1,0,g,packed[r],scatter);
        check(cudaDeviceSynchronize()); assert(*net[r]->f.error==0);
        if constexpr(Trace)validate_trace(net[r]->f,1);
        for(int s=0;s<3;++s) {
            auto& d=data[r][s];check(cudaMemcpy(d->output.data(),d->dst,d->output.size(),cudaMemcpyDeviceToHost));
            for(int p=0;p<4;++p)for(int m=0;m<Q;++m)for(size_t j=0;j<widths[s];++j) {
                const bool valid=scatter?m<g.owned:p*Q+m<M;
                const auto want=valid?pattern(p,scatter?r*Q+m:m,int(j)):0xcd;
                assert(d->output[(size_t(p)*Q+m)*widths[s]+j]==want);
            }
        }
    }
    for(int r=0;r<4;++r) { for(int s=0;s<3;++s)delete data[r][s]; delete net[r]; }
}
template<bool Trace=false>
static void reduction(int M) {
    const int Q=(M+3)/4,cols=24;
    Segment shape{nullptr,nullptr,size_t(cols)*2};auto w=wire_layout(Q,&shape,1);
    Region* net[4]; __nv_bfloat16 *src[4],*dst[4]; std::vector<__nv_bfloat16> h(size_t(M)*cols),out(size_t(Q)*cols);
    for(int r=0;r<4;++r) {
        net[r]=new Region(r,4*w.stride,Trace);check(cudaMalloc(&src[r],h.size()*2));check(cudaMalloc(&dst[r],out.size()*2));
        for(int m=0;m<M;++m)for(int c=0;c<cols;++c)h[size_t(m)*cols+c]=__float2bfloat16_rn(float(r+m%8+c%4));
        check(cudaMemcpy(src[r],h.data(),h.size()*2,cudaMemcpyHostToDevice));
        detail::Segments seg{};seg.w=w;seg.s[0]={src[r],dst[r],size_t(cols)*2};
        if constexpr(Trace)detail::trace_produced<<<1,32>>>(net[r]->f,2,0);
        detail::pack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,2,0,geometry(M,4,r),seg,true);
        if constexpr(Trace)detail::publish<true><<<1,32>>>(net[r]->f,2,0);
    }
    check(cudaDeviceSynchronize());proxy(net,2,w,true);
    for(int r=0;r<4;++r) {
        auto g=geometry(M,4,r);
        if constexpr(Trace) {
            detail::wait<true><<<1,32>>>(net[r]->f,2,0,w.payload);
        }
        detail::reduce<Trace><<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,2,0,g,w.stride,cols,dst[r]);
        check(cudaMemcpy(out.data(),dst[r],out.size()*2,cudaMemcpyDeviceToHost));
        for(int m=0;m<g.owned;++m)for(int c=0;c<cols;++c)assert(__bfloat162float(out[size_t(m)*cols+c])==6+4*((g.row0+m)%8+c%4));
        if constexpr(Trace)validate_trace(net[r]->f,2);
        check(cudaFree(src[r]));check(cudaFree(dst[r]));delete net[r];
    }
}
static void tail_gather(int M) {
    const size_t widths[]={40960,96,16,16,64,10240};
    Segment shape[6]; for(int k=0;k<6;++k)shape[k].row_bytes=widths[k];
    const auto w=wire_layout(128,shape,6);
    Region* net[4]; Data* data[4][6]; detail::Segments seg[4]{};
    for(int r=0;r<4;++r) {
        net[r]=new Region(r,w.stride); seg[r].w=w;
        const auto g=geometry(M,4,r); const auto t=tail_slice(g);
        for(int k=0;k<6;++k) {
            auto* d=data[r][k]=new Data(128*widths[k],512*widths[k]);
            for(int m=0;m<t.count;++m)for(size_t j=0;j<widths[k];++j)
                d->input[size_t(m)*widths[k]+j]=pattern(k,g.row0+t.local+m,int(j));
            check(cudaMemcpy(d->src,d->input.data(),d->input.size(),cudaMemcpyHostToDevice));
            seg[r].s[k]={d->src,d->dst,widths[k]};
        }
        detail::pack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,1,0,geometry(512,4,r),seg[r],false);
    }
    check(cudaDeviceSynchronize());proxy(net,1,w,false);
    for(int r=0;r<4;++r) {
        detail::unpack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,1,0,geometry(512,4,r),seg[r],false);
        for(int k=0;k<6;++k) {
            auto* d=data[r][k];check(cudaMemcpy(d->output.data(),d->dst,d->output.size(),cudaMemcpyDeviceToHost));
            for(int p=0;p<4;++p) {
                const auto g=geometry(M,4,p);const auto t=tail_slice(g);
                for(int m=0;m<128;++m)for(size_t j=0;j<widths[k];++j) {
                    const auto want=m<t.count?pattern(k,g.row0+t.local+m,int(j)):0;
                    assert(d->output[(size_t(p)*128+m)*widths[k]+j]==want);
                }
            }
        }
    }
    for(int r=0;r<4;++r) {for(int k=0;k<6;++k)delete data[r][k];delete net[r];}
}
int main() {
    for(int M:{1,3,5,17,127,128,129,257,511,4093,4096,8191,8192})tail_gather(M);
    std::cout<<"ds41_v2_wire: six-array tail gather, 13 ragged chunks PASS (proxy-copy emulation)\n";
    for(int M:{1,3,5,17,257,4096,8191}) {test(M,false);test(M,true);reduction(M);}
    for(int M:{1,3,5,17,257,4096,8191}) {test<true>(M,false);test<true>(M,true);reduction<true>(M);}
    std::cout<<"ds41_v2_xtrace: AG/A2A/RS timestamps, optional fields, joins, ring reuse and unchanged output PASS (proxy-copy emulation)\n";
    std::cout<<"ds41_v2_wire: 7 row shapes x AG/A2A/RS, ragged padding and unaligned segments PASS (proxy-copy emulation, not RDMA)\n";
}
