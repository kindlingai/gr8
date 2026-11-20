// Single-GPU byte-exactness check; mapped peer slots emulate fabric writes.
// No model, network or proxy is started. Run separately from performance work.
#include "../runtime/ds41_v2.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace ds41::v2;
struct Region {
    g6::FabricDev f{};
    explicit Region(int rank,size_t stride,bool trace=false) {
        f.rank=rank; f.nr=4; f.ncoll=1; f.stride=stride;
        check(cudaHostAlloc(&f.send,2*stride,cudaHostAllocMapped));
        check(cudaHostAlloc(&f.recv,8*stride,cudaHostAllocMapped));
        std::memset(f.send,0xa5,2*stride); std::memset(f.recv,0xa5,8*stride);
        if(trace) { check(cudaMalloc(&f.xt,g6::XT_STEPS*g6::XT_W*sizeof(*f.xt)));
            check(cudaMemset(f.xt,0,g6::XT_STEPS*g6::XT_W*sizeof(*f.xt))); }
    }
    ~Region() { if(f.xt)check(cudaFree(f.xt)); check(cudaFreeHost(f.send)); check(cudaFreeHost(f.recv)); }
};
struct Buffer {
    unsigned char* p=nullptr; size_t size;
    explicit Buffer(size_t n):size(n) { check(cudaMalloc(&p,size)); }
    ~Buffer() { check(cudaFree(p)); }
    std::vector<unsigned char> read() const {
        std::vector<unsigned char> v(size); check(cudaMemcpy(v.data(),p,size,cudaMemcpyDeviceToHost)); return v;
    }
    void dirty() { check(cudaMemset(p,0xcd,size)); }
};
static void equal(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b) {
    assert(a.size()==b.size()); assert(std::memcmp(a.data(),b.data(),a.size())==0);
}
template<bool Trace>
static void reduction(int M,int cols,int rank,bool unaligned) {
    auto g=geometry(M,4,rank); Segment shape{nullptr,nullptr,size_t(cols)*2};
    auto w=wire_layout(g.Q,&shape,1); Region net(rank,4*w.stride,Trace);
    // Offset destination by one bf16 to test the alignment fallback too.
    Buffer scalar(size_t(g.Q)*cols*2+32),vector(scalar.size);
    auto* a=reinterpret_cast<__nv_bfloat16*>(scalar.p)+(unaligned?1:0);
    auto* b=reinterpret_cast<__nv_bfloat16*>(vector.p)+(unaligned?1:0);
    std::mt19937 rng(1234+M+cols+rank);
    for(unsigned epoch:{0,1,2}) { // reuse parity 0 with different peer values
        for(int p=0;p<4;++p) {
            char* base=p==rank?net.f.send+g6::fab_send_off(net.f,epoch&1,0)+rank*w.stride:
                net.f.recv+g6::fab_recv_off(net.f,epoch&1,0,p);
            auto* h=reinterpret_cast<__nv_bfloat16*>(base);
            for(int i=0;i<g.Q*cols;++i) {
                const float v=float(int(rng()%65537)-32768)/128.f;
                h[i]=__float2bfloat16_rn(std::ldexp(v,int(rng()%21)-10));
            }
        }
        scalar.dirty();
        detail::reduce<Trace><<<8,256>>>(net.f,epoch,0,g,w.stride,cols,a);
        auto expected=scalar.read();
        for(int width:{8,48,192,1024}) {
            vector.dirty();
            detail::reduce_vector<Trace><<<width,256>>>(net.f,epoch,0,g,w.stride,cols,b);
            equal(expected,vector.read());
        }
    }
}
static void copies(int M,int rank,bool scatter,bool unaligned) {
    auto g=geometry(M,4,rank);
    constexpr size_t widths[]={24,17,160};
    Segment shapes[3]; for(int j=0;j<3;++j)shapes[j].row_bytes=widths[j];
    auto w=wire_layout(g.Q,shapes,3); Region net(rank,4*w.stride,true);
    detail::Segments seg{}; seg.w=w;
    Buffer *src[3],*dst[3]; const size_t offset=unaligned?1:0;
    for(int j=0;j<3;++j) {
        src[j]=new Buffer(size_t(M)*widths[j]+32); dst[j]=new Buffer(size_t(4*g.Q)*widths[j]+32);
        std::vector<unsigned char> h(src[j]->size);
        for(size_t i=0;i<h.size();++i)h[i]=(i*13+j*29+rank*7)%251;
        check(cudaMemcpy(src[j]->p,h.data(),h.size(),cudaMemcpyHostToDevice));
        seg.s[j]={src[j]->p+offset,dst[j]->p+offset,widths[j]};
    }
    for(unsigned epoch:{0,1,2}) {
        // Compare every byte, including zero-filled wire padding and guards.
        std::memset(net.f.send,0xa5,2*net.f.stride);
        detail::pack<<<8,256>>>(net.f,epoch,0,g,seg,scatter); check(cudaDeviceSynchronize());
        std::vector<unsigned char> packed(net.f.send,net.f.send+2*net.f.stride);
        for(int width:{8,48,192,1024}) {
            std::memset(net.f.send,0xa5,2*net.f.stride);
            detail::pack<<<width,256>>>(net.f,epoch,0,g,seg,scatter); check(cudaDeviceSynchronize());
            equal(packed,std::vector<unsigned char>(net.f.send,net.f.send+2*net.f.stride));
        }
        for(size_t i=0;i<8*net.f.stride;++i)net.f.recv[i]=(i*19+epoch*23)%251;
        for(auto* p:dst)p->dirty();
        detail::unpack<<<8,256>>>(net.f,epoch,0,g,seg,scatter);
        std::vector<unsigned char> expected[3]; for(int j=0;j<3;++j)expected[j]=dst[j]->read();
        for(int width:{8,48,192,1024})for(bool trace:{false,true}) {
            for(auto* p:dst)p->dirty();
            if(trace)detail::unpack<true><<<width,256>>>(net.f,epoch,0,g,seg,scatter);
            else detail::unpack<<<width,256>>>(net.f,epoch,0,g,seg,scatter);
            for(int j=0;j<3;++j)equal(expected[j],dst[j]->read());
        }
    }
    for(int j=0;j<3;++j) { delete src[j]; delete dst[j]; }
}
int main() {
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
    size_t free=0,total=0; check(cudaMemGetInfo(&free,&total));
    std::cout<<"GPU="<<prop.name<<" free_bytes="<<free<<" total_bytes="<<total<<std::endl;
    // These fixed cases peak below 11 MiB cudaMalloc + 201 MiB mapped slots.
    // Leave the resident serving process alone if the device lacks headroom.
    if(free<size_t(1536)*1024*1024) { std::cout<<"SKIP: less than 1.5 GiB free\n"; return 77; }
    for(int M:{1,3,5,17,257,4093,8191})for(int rank=0;rank<4;++rank) {
        for(int cols:{1,7,24,127,1280})for(bool unaligned:{false,true}) {
            reduction<false>(M,cols,rank,unaligned); reduction<true>(M,cols,rank,unaligned);
        }
        for(bool scatter:{false,true})for(bool unaligned:{false,true})copies(M,rank,scatter,unaligned);
    }
    check(cudaDeviceSynchronize());
    std::cout<<"ds41_v2_width CUDA: widths 8/48/192/1024 byte-equal to scalar reduce / eight-block copies; 7 ragged M x 4 ranks, 5 cols, aligned/unaligned, trace on/off, dirty parity reuse and guards PASS (mapped-slot emulation, not RDMA)\n";
}
