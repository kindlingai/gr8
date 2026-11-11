// Bounded synthetic-test utilities. No model or network is opened by these tests.
#pragma once
#include "../runtime/ds41_v2.cuh"
#include <array>
#include <vector>
#include <cstring>
namespace v2test {
using namespace ds41::v2;
inline void insist(bool value,const char* label) {
    if(!value) { fprintf(stderr,"FAIL: %s\n",label); std::abort(); }
}
constexpr size_t budget=384ull<<20, red=256;
inline size_t live=0,peak=0;
using Bytes=std::vector<uint8_t>;
inline Bytes download(const void* p,size_t n) {
    Bytes b(n); if(n)check(cudaMemcpy(b.data(),p,n,cudaMemcpyDeviceToHost)); return b;
}
inline void upload(void* p,const Bytes& b) {
    if(!b.empty())check(cudaMemcpy(p,b.data(),b.size(),cudaMemcpyHostToDevice));
}
inline Bytes data(size_t n,unsigned seed) {
    Bytes b(n); uint32_t x=seed+1;
    for(auto& v:b) { x^=x<<13; x^=x>>17; x^=x<<5; v=uint8_t(x); } return b;
}
inline uint64_t hash(const Bytes& b) {
    uint64_t h=0xcbf29ce484222325ull;
    for(auto v:b)h=(h^v)*0x100000001b3ull; return h;
}
struct Arena {
    struct Block { uint8_t* p; size_t n; bool host; };
    std::vector<Block> blocks;
    template<typename T> T* make(size_t count,bool host=false) {
        const size_t n=std::max(size_t(1),count*sizeof(T)), total=n+2*red;
        insist(total<budget && live<budget-total,"device + mapped live bytes below 384 MiB");
        uint8_t* p=nullptr;
        if(host) { check(cudaHostAlloc(&p,total,cudaHostAllocMapped)); memset(p,0xd3,total); }
        else { check(cudaMalloc(&p,total)); check(cudaMemset(p,0xd3,total)); }
        live+=total; peak=std::max(peak,live); blocks.push_back({p,n,host});
        return reinterpret_cast<T*>(p+red);
    }
    void verify() {
        check(cudaDeviceSynchronize());
        for(auto b:blocks)for(size_t offset:{size_t(0),red+b.n}) {
            Bytes z=b.host?Bytes(b.p+offset,b.p+offset+red):download(b.p+offset,red);
            for(auto v:z)insist(v==0xd3,"allocation red zones");
        }
    }
    ~Arena() {
        verify();
        for(auto b:blocks) { if(b.host)check(cudaFreeHost(b.p)); else check(cudaFree(b.p)); live-=b.n+2*red; }
    }
};
struct Inputs {
    struct Entry { const void* p; size_t n; uint64_t digest; };
    std::vector<Entry> entries;
    void add(const void* p,size_t n) { entries.push_back({p,n,hash(download(p,n))}); }
    void verify() { for(auto e:entries)insist(hash(download(e.p,e.n))==e.digest,"immutable input hash"); }
};
inline void equal(const void* p,const Bytes& b,const char* name) {
    auto got=download(p,b.size()); insist(memcmp(got.data(),b.data(),b.size())==0,name);
}
inline void fp8(void* p,size_t n,unsigned seed,bool scale=false) {
    auto b=data(n,seed); for(auto& v:b)v=scale?uint8_t(120+v%4):uint8_t(0x20+(v%24)+(v&0x80)); upload(p,b);
}
inline void bf16(void* p,size_t n,unsigned seed) {
    auto raw=data(n*2,seed); std::vector<__nv_bfloat16> b(n);
    for(size_t i=0;i<n;++i)b[i]=__float2bfloat16_rn((int(raw[i])-128)/128.f);
    check(cudaMemcpy(p,b.data(),n*2,cudaMemcpyHostToDevice));
}
// One mapped fabric, reused sequentially. Wire snapshots are pageable CPU memory.
struct Wire {
    g6::FabricDev f{};
    Wire(Arena& a,size_t stride,bool trace=false) {
        f.nr=4; f.ncoll=1; f.stride=aligned(stride,256); f.compact=1;
        f.send=a.make<char>(2*f.stride,true); f.recv=a.make<char>(g6::fab_recv_bytes(f),true);
        f.error=a.make<unsigned>(1,true); f.doorbell=a.make<unsigned long long>(1,true);
        if(trace)f.xt=a.make<unsigned long long>(g6::XT_STEPS*g6::XT_W);
        clear(0);
    }
    char* slot() { return f.send+g6::fab_send_off(f,1,0); }
    void clear(int rank) {
        check(cudaDeviceSynchronize()); f.rank=rank;
        memset(f.send,0xa7,2*f.stride); memset(f.recv,0,g6::fab_recv_bytes(f)); *f.error=0; *f.doorbell=0;
        if(f.xt)check(cudaMemset(f.xt,0,g6::XT_STEPS*g6::XT_W*8));
    }
    Bytes snapshot() {
        check(cudaDeviceSynchronize()); insist(*f.doorbell==g6::coll_seq(1,0,1),"published sequence");
        return Bytes(slot(),slot()+f.stride);
    }
    void deliver(const std::array<Bytes,4>& sent,const WireLayout& w,bool scatter) {
        memcpy(slot(),sent[f.rank].data(),f.stride);
        for(int p=0;p<4;++p)if(p!=f.rank) {
            char* dest=f.recv+g6::fab_recv_off(f,1,0,p);
            memcpy(dest,sent[p].data()+(scatter?size_t(f.rank)*w.stride:0),w.payload);
            *reinterpret_cast<unsigned long long*>(dest+w.payload)=g6::coll_seq(1,0,1);
        }
    }
};
inline void summary(const char* name) {
    insist(live==0 && peak<budget,"bounded allocations released");
    printf("PASS %s SYNTHETIC correctness only; peak_bytes=%zu peak_MiB=%.3f live_bytes=%zu guards/input_hashes/negative_controls PASS\n",name,peak,peak/1048576.,live);
}
} // namespace v2test
