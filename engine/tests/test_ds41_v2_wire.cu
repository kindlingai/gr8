// No model, network or verbs needed: emulate only the proxy copies between
// mapped slots, then test the actual v2 GPU pack/unpack/reduce kernels.
#include "../runtime/ds41_v2.cuh"
#include <cassert>
#include <iostream>
#include <vector>
using namespace ds41::v2;
struct Region {
    g6::FabricDev f{};
    Region(int rank,size_t stride) {
        f.rank=rank; f.nr=4; f.ncoll=1; f.stride=stride; f.compact=0;
        check(cudaHostAlloc(&f.send,2*stride,cudaHostAllocMapped));
        check(cudaHostAlloc(&f.recv,8*stride,cudaHostAllocMapped));
        check(cudaHostAlloc(&f.error,4,cudaHostAllocMapped)); *f.error=0;
        check(cudaHostAlloc(&f.doorbell,8,cudaHostAllocMapped)); *f.doorbell=0;
        std::memset(f.send,0,2*stride); std::memset(f.recv,0,8*stride);
    }
    ~Region() { check(cudaFreeHost(f.send));check(cudaFreeHost(f.recv));check(cudaFreeHost(f.error));check(cudaFreeHost(f.doorbell)); }
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
static void test(int M,bool scatter) {
    const int Q=(M+3)/4;
    const size_t widths[]={24,17,160}; // includes unaligned row/slice boundaries
    Segment shape[3]; for(int s=0;s<3;++s)shape[s].row_bytes=widths[s];
    auto w=wire_layout(Q,shape,3);
    const size_t stride=4*w.stride;
    Region* net[4]; Data* data[4][3]; detail::Segments packed[4]{};
    for(int r=0;r<4;++r) {
        net[r]=new Region(r,stride); auto g=geometry(M,4,r); packed[r].w=w;
        for(int s=0;s<3;++s) {
            auto& d=data[r][s];
            d=new Data(size_t(scatter?M:std::max(1,g.owned))*widths[s],size_t(4*Q)*widths[s]);
            for(int m=0;m<(scatter?M:g.owned);++m)for(size_t j=0;j<widths[s];++j)
                d->input[size_t(m)*widths[s]+j]=pattern(r,m,int(j));
            check(cudaMemcpy(d->src,d->input.data(),d->input.size(),cudaMemcpyHostToDevice));
            check(cudaMemcpy(d->dst,d->output.data(),d->output.size(),cudaMemcpyHostToDevice));
            packed[r].s[s]={d->src,d->dst,widths[s]};
        }
        detail::pack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,1,0,g,packed[r],scatter);
        detail::publish<<<1,32>>>(net[r]->f,1,0);
    }
    check(cudaDeviceSynchronize());
    proxy(net,1,w,scatter);
    for(int r=0;r<4;++r) {
        auto g=geometry(M,4,r);
        detail::wait<<<1,32>>>(net[r]->f,1,0,w.payload);
        detail::unpack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,1,0,g,packed[r],scatter);
        check(cudaDeviceSynchronize()); assert(*net[r]->f.error==0);
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
static void reduction(int M) {
    const int Q=(M+3)/4,cols=24;
    Segment shape{nullptr,nullptr,size_t(cols)*2};auto w=wire_layout(Q,&shape,1);
    Region* net[4]; __nv_bfloat16 *src[4],*dst[4]; std::vector<__nv_bfloat16> h(size_t(M)*cols),out(size_t(Q)*cols);
    for(int r=0;r<4;++r) {
        net[r]=new Region(r,4*w.stride);check(cudaMalloc(&src[r],h.size()*2));check(cudaMalloc(&dst[r],out.size()*2));
        for(int m=0;m<M;++m)for(int c=0;c<cols;++c)h[size_t(m)*cols+c]=__float2bfloat16_rn(float(r+m%8+c%4));
        check(cudaMemcpy(src[r],h.data(),h.size()*2,cudaMemcpyHostToDevice));
        detail::Segments seg{};seg.w=w;seg.s[0]={src[r],dst[r],size_t(cols)*2};
        detail::pack<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,2,0,geometry(M,4,r),seg,true);
    }
    check(cudaDeviceSynchronize());proxy(net,2,w,true);
    for(int r=0;r<4;++r) {
        auto g=geometry(M,4,r);
        detail::reduce<<<COMM_BLOCKS,COMM_THREADS>>>(net[r]->f,2,0,g,w.stride,cols,dst[r]);
        check(cudaMemcpy(out.data(),dst[r],out.size()*2,cudaMemcpyDeviceToHost));
        for(int m=0;m<g.owned;++m)for(int c=0;c<cols;++c)assert(__bfloat162float(out[size_t(m)*cols+c])==6+4*((g.row0+m)%8+c%4));
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
    std::cout<<"ds41_v2_wire: 7 row shapes x AG/A2A/RS, ragged padding and unaligned segments PASS (proxy-copy emulation, not RDMA)\n";
}
