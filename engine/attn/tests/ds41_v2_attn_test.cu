// Synthetic native-wire/stage regression; mapped fabric, no model/network.
// Device + mapped allocations bounded below 1 GiB; GPU execution is a separate gate.
// nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
//   engine/attn/tests/ds41_v2_attn_test.cu engine/knobs-rs/target/release/libgr8knobs.a \
//   -lpthread -ldl -lm -o ds41_v2_attn_test
#include "../ds41_attn_v2.cuh"
#include "../ds41_attn_gv.cuh"
#include "../ds41_idx.cuh"
#include "../../kernels/ds41_v2_bf16_tc.cuh"
#include "../../tests/ds41_v2_test_support.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>
namespace test {
using namespace ds41a;
#define CK(call) do { auto e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2); } } while(0)
void require(bool ok, const char* what) { if (!ok) { fprintf(stderr,"FAIL %s\n",what); std::exit(1); } }
struct Arena {
    struct Block { uint8_t* base; size_t bytes; };
    std::vector<Block> blocks;
    size_t live=0;
    template<class T> T* alloc(size_t n) {
        const size_t bytes=std::max(size_t(1),n*sizeof(T));
        require(live+bytes+512 < (768ull<<20),"test arena below 768 MiB");
        uint8_t* base; CK(cudaMalloc(&base,bytes+512));
        CK(cudaMemset(base,0xd3,bytes+512)); CK(cudaMemset(base+256,0,bytes));
        blocks.push_back({base,bytes}); live+=bytes+512;
        return reinterpret_cast<T*>(base+256);
    }
    ~Arena() {
        CK(cudaDeviceSynchronize());
        for(auto b:blocks) {
            uint8_t guard[256];
            for(size_t offset:{size_t(0),256+b.bytes}) {
                CK(cudaMemcpy(guard,b.base+offset,256,cudaMemcpyDeviceToHost));
                for(auto x:guard)require(x==0xd3,"allocation red zones");
            }
            CK(cudaFree(b.base));
        }
    }
};
unsigned rng=1017;
float rnd() { rng=1664525*rng+1013904223; return float(int((rng>>8)%2001)-1000)/1000.f; }
template<class T> float val(T a) { return float(a); }
template<> float val(bf16 a) { return __bfloat162float(a); }
template<class T> std::vector<T> host(const T* p,size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
template<class T> std::vector<T> fill(T* p,size_t n,float scale=1) {
    std::vector<T> v(n); for(auto& x:v) x=T(rnd()*scale);
    CK(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); return v;
}
template<class A,class B> void close(const char* name,const std::vector<A>& a,const std::vector<B>& b,float tol) {
    require(a.size()==b.size(),"size"); double e=0,d=0; float mx=0;
    for(size_t i=0;i<a.size();++i) { float av=val(a[i]),bv=val(b[i]); require(std::isfinite(av)&&std::isfinite(bv),"finite"); double t=av-bv;e+=t*t;d+=bv*bv;mx=std::max(mx,float(fabs(t))); }
    require(a.empty() || d>0,"nontrivial reference output");
    double rel=std::sqrt(e/std::max(d,1e-30)); printf("%-36s rel_rms=%.7g max_abs=%.7g tolerance=%.4g %s\n",name,rel,mx,tol,rel<=tol?"PASS":"FAIL"); require(rel<=tol,name);
}
v2::InputBuf input(Arena& a,int rows,int q) { v2::InputBuf b; b.carve(a.alloc<char>(v2::InputBuf::bytes(rows,q)),rows,q); return b; }
Scratch scratch(Arena& a,int rows,int heads) { Scratch s{};s.carve(a.alloc<char>(Scratch::bytes(rows,heads)),rows,heads);return s; }
Fp8Mat weight(Arena& a,int N,int K,int groups=1) {
    auto* raw=a.alloc<bf16>(size_t(groups)*N*K); fill(raw,size_t(groups)*N*K,.04f);
    auto* codes=a.alloc<uint8_t>(size_t(groups)*N*K); auto* scales=a.alloc<uint8_t>(size_t(groups)*N*K/32);
    launch_act_quant(raw,K,codes,K,scales,K/32,groups*N,K,0);
    return {codes,scales,N,K};
}
#include "ds41_v2_attn_native.inc"
void output_test(int M) {
    Arena a;const int nr=4,Q=(M+nr-1)/nr,heads=16,K=v2::output_width(heads),stride=v2::output_stride(heads);
    std::vector<Weights> w(nr);std::vector<Scratch>s;std::vector<uint8_t*> sends;std::vector<bf16*> ref;
    auto* pos=a.alloc<int>(M);std::vector<int> positions(M);for(int i=0;i<M;++i)positions[i]=77+i;CK(cudaMemcpy(pos,positions.data(),M*4,cudaMemcpyHostToDevice));
    for(int p=0;p<nr;++p) {
        w[p].wo_a=weight(a,OLORA,GROUP_IN,heads/HPG);w[p].wo_b=weight(a,ds41a::HID,K);w[p].inv=a.alloc<float>(32);
        s.push_back(scratch(a,M,heads));fill(s[p].o,size_t(M)*heads*HD,.3f);
        sends.push_back(a.alloc<uint8_t>(size_t(nr)*Q*stride));ref.push_back(a.alloc<bf16>(size_t(M)*ds41a::HID));
        attn_out(w[p],s[p],M,0,heads,ref[p],0,pos);
        v2::output_a(w[p],s[p],M,heads,0,pos);v2::pack_output(s[p],sends[p],M,nr*Q,heads,0);
    }
    auto* recv=a.alloc<uint8_t>(size_t(nr)*Q*stride);auto* parts=a.alloc<bf16>(size_t(nr)*Q*ds41a::HID);auto* sum=a.alloc<bf16>(size_t(Q)*ds41a::HID);
    std::vector<std::vector<bf16>> refh;for(int p=0;p<nr;++p)refh.push_back(host(ref[p],size_t(M)*ds41a::HID));
    wire::Segment segment{sends[0],recv,size_t(stride)};
    auto layout=wire::wire_layout(Q,&segment,1);
    v2test::Arena mapped;v2test::Wire net(mapped,4*layout.stride);
    std::array<Bytes,4> opt;
    for(int p=0;p<nr;++p) {
        net.clear(p);auto g=wire::geometry(M,4,p);
        wire::detail::pack_attn_out_remote<<<32,256>>>(net.f,1,0,g,s[p].oaq,s[p].oas,K,layout.stride);
        wire::detail::publish<><<<1,32>>>(net.f,1,0);opt[p]=net.snapshot();
        auto reference=v2test::download(sends[p],size_t(nr)*Q*stride);
        for(int r=0;r<4;++r)if(r!=p)
            require(!memcmp(opt[p].data()+size_t(r)*layout.stride,reference.data()+size_t(r)*Q*stride,layout.payload),"optimized remote wire equality");
    }
    for(int rank=0;rank<nr;++rank) {
        int r0=rank*Q,n=std::max(0,std::min(Q,M-r0));
        for(int p=0;p<nr;++p)CK(cudaMemcpyAsync(recv+size_t(p)*Q*stride,sends[p]+size_t(r0)*stride,size_t(Q)*stride,cudaMemcpyDeviceToDevice));
        // Own slice is consumed from send while remote slices use receive.
        v2::output_b(w[rank].wo_b,sends[rank]+size_t(r0)*stride,parts+size_t(rank)*Q*ds41a::HID,n,s[rank].xf_oa,0);
        for(int p=0;p<nr;++p)if(p!=rank)v2::output_b(w[p].wo_b,recv+size_t(p)*Q*stride,parts+size_t(p)*Q*ds41a::HID,n,s[rank].xf_oa,0);
        v2::sum_output(parts,sum,n,Q,nr,0);CK(cudaDeviceSynchronize());
        std::vector<bf16> expected(size_t(n)*ds41a::HID);
        for(size_t j=0;j<expected.size();++j){float v=0;for(int p=0;p<nr;++p)v+=val(refh[p][size_t(r0)*ds41a::HID+j]);expected[j]=bf16(v);}
        close("wo_a a2a + wo_b vs stock sum",host(sum,size_t(n)*ds41a::HID),expected,.004f);
        net.clear(rank);net.deliver(opt,layout,true);
        CK(cudaMemset(recv,0xa7,size_t(nr)*Q*stride));
        auto args=wire::detail::segments(wire::geometry(M,4,rank),&segment,1);
        wire::detail::wait<><<<1,32>>>(net.f,1,0,layout.payload);
        wire::detail::unpack_remote<><<<32,256>>>(net.f,1,0,wire::geometry(M,4,rank),args);
        v2::output_b_split(w[rank].wo_b,s[rank].oaq+size_t(r0)*K,s[rank].oas+size_t(r0)*(K/32),
                          parts+size_t(rank)*Q*ds41a::HID,n,s[rank].xf_oa,0);
        for(int p=0;p<nr;++p)if(p!=rank)v2::output_b(w[p].wo_b,recv+size_t(p)*Q*stride,
                          parts+size_t(p)*Q*ds41a::HID,n,s[rank].xf_oa,0);
        CK(cudaMemset(sum,0xff,size_t(Q)*ds41a::HID*2));v2::sum_output(parts,sum,n,Q,nr,0);
        close("optimized A2A split wo_b stock sum",host(sum,size_t(n)*ds41a::HID),expected,.004f);
        printf("OUTPUT_PASS M=%d rank=%d Q=%d owned=%d remote_wire_bit_mismatches=0 tolerance=0.004\n",M,rank,Q,n);
    }
}
}
int main() {
    CK(ds41v2tc::prepare());
    for(int M:{2,5,65,129,8192,12288})for(int mode=0;mode<3;++mode)test::wire_test(M,mode);
    for(int start:{7,128})for(int M:{5,17,33,64,65,97,129,256,512,513})
        for(bool tc:{false,true})for(int mask:{0,7})for(int mode=0;mode<3;++mode)
            test::stage_test(M,start,mode,tc,mask);
    ds41a::g_use_gv=false; ds41a::g_use_tc=true;
    test::output_test(65);test::output_test(128);test::output_test(2);
    CK(cudaGetLastError());
    puts("PASS v2 native attention wire/stage suite; model quality and fabric speed NOT measured");
}
