// Synthetic v2 math/layout test; no model files, no fabric, <1 GiB device memory.
// nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
//   engine/attn/tests/ds41_v2_attn_test.cu engine/knobs-rs/target/release/libgr8knobs.a \
//   -lpthread -ldl -lm -o ds41_v2_attn_test
#include "../ds41_attn_v2.cuh"
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
    std::vector<void*> p;
    template<class T> T* alloc(size_t n) { T* a; CK(cudaMalloc(&a,n*sizeof(T))); p.push_back(a); CK(cudaMemset(a,0,n*sizeof(T))); return a; }
    ~Arena() { for(void* a:p) CK(cudaFree(a)); }
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
void wire_test() {
    Arena a; const int n=5,q=7; auto in=input(a,q,q),out=input(a,q,q);
    auto qa=fill(in.qa,size_t(n)*QA,3),kv=fill(in.kv0,size_t(n)*HD,2);
    auto c1=fill(in.ckv1,size_t(n)*HD,5); auto ck=fill(in.ckv,size_t(n)*HD,7),cs=fill(in.csc,size_t(n)*HD,9);
    auto* w=a.alloc<uint8_t>(size_t(q)*v2::input_stride(2));
    for(int mode=0;mode<=2;++mode) {
        v2::pack_input(in,w,n,q,mode,0);v2::unpack_input(w,out,q,mode,0);CK(cudaDeviceSynchronize());
        close("input FP8 qa wire",host(out.qa,size_t(n)*QA),qa,.04f);
        close("input FP8 kv wire",host(out.kv0,size_t(n)*HD),kv,.04f);
        if(mode==1)close("input FP8 ratio1 wire",host(out.ckv1,size_t(n)*HD),c1,.04f);
        if(mode==2) {close("input FP8 ckv wire",host(out.ckv,size_t(n)*HD),ck,.04f);close("input FP8 csc wire",host(out.csc,size_t(n)*HD),cs,.04f);}
        const auto padded=host(out.qa+size_t(n)*QA,size_t(q-n)*QA);
        for(auto v:padded)require(val(v)==0,"input padding");
        const auto bytes=host(w,size_t(q)*v2::input_stride(mode));
        for(int r=0;r<q;++r)for(int k=v2::input_width(mode)*33/32;k<v2::input_stride(mode);++k)require(bytes[size_t(r)*v2::input_stride(mode)+k]==0,"wire alignment initialized");
        printf("input mode=%d stride=%d ragged/padding PASS\n",mode,v2::input_stride(mode));
    }
}
void projection_test() {
    Arena a; const int M=65,Q=17; Weights w{};w.wq_a=weight(a,QA,HID);w.wkv=weight(a,HD,HID);
    auto* ckv=a.alloc<bf16>(size_t(HD)*HID);auto* csc=a.alloc<bf16>(size_t(HD)*HID);
    fill(ckv,size_t(HD)*HID,.02f);fill(csc,size_t(HD)*HID,.02f);w.c_wkv=ckv;w.c_wgate=csc;
    auto* y=a.alloc<bf16>(size_t(M)*HID);fill(y,size_t(M)*HID);
    auto full=input(a,M,M), own=input(a,M,Q);
    v2::attn_proj_own(w,full,y,nullptr,nullptr,0,M,true,0);
    for(int r=0;r<4;++r) { int r0=r*Q,n=std::max(0,std::min(Q,M-r0));if(n)v2::attn_proj_own(w,own,y+size_t(r0)*HID,nullptr,nullptr,r0,n,true,0); }
    CK(cudaDeviceSynchronize());
    close("own-row wq_a vs full",host(own.qa,size_t(M)*QA),host(full.qa,size_t(M)*QA),.003f);
    close("own-row wkv vs full",host(own.kv0,size_t(M)*HD),host(full.kv0,size_t(M)*HD),.003f);
    close("own-row compressor vs full",host(own.ckv,size_t(M)*HD),host(full.ckv,size_t(M)*HD),.003f);
    close("own-row gate vs full",host(own.csc,size_t(M)*HD),host(full.csc,size_t(M)*HD),.003f);
    // Gather projected FP8 rows in rank order, including a short final owner.
    auto* send=a.alloc<uint8_t>(size_t(Q)*v2::input_stride(2));auto* all=a.alloc<uint8_t>(size_t(4)*Q*v2::input_stride(2));
    for(int r=0;r<4;++r){int r0=r*Q,n=std::max(0,std::min(Q,M-r0));auto src=full;
        src.qa+=size_t(r0)*QA;src.kv0+=size_t(r0)*HD;src.ckv+=size_t(r0)*HD;src.csc+=size_t(r0)*HD;
        v2::pack_input(src,send,n,Q,2,0);CK(cudaMemcpyAsync(all+size_t(r)*Q*v2::input_stride(2),send,size_t(Q)*v2::input_stride(2),cudaMemcpyDeviceToDevice));}
    v2::unpack_input(all,own,M,2,0);CK(cudaDeviceSynchronize());
    close("gathered qa FP8 tolerance",host(own.qa,size_t(M)*QA),host(full.qa,size_t(M)*QA),.04f);
    // Pooling must see global order, including odd starts and cross-owner pairs.
    auto sf=scratch(a,M,16),so=scratch(a,M,16);Cache cf{},co{};const int start=7,cap=128;
    cf.comp=a.alloc<bf16>(cap*HD);co.comp=a.alloc<bf16>(cap*HD);cf.kvs=a.alloc<float>(2*HD);co.kvs=a.alloc<float>(2*HD);
    cf.scs=a.alloc<float>(2*HD);co.scs=a.alloc<float>(2*HD);auto* norm=a.alloc<bf16>(HD);std::vector<bf16> ones(HD,bf16(1.f));CK(cudaMemcpy(norm,ones.data(),HD*2,cudaMemcpyHostToDevice));w.c_norm=norm;
    auto* inv=a.alloc<float>(32); w.inv=inv;auto* pos=a.alloc<int>(M);std::vector<int> pv(M);for(int r=0;r<M;++r)pv[r]=start+r;CK(cudaMemcpy(pos,pv.data(),M*4,cudaMemcpyHostToDevice));
    // Independent stock compressor computes its own projections; v2 consumes
    // the gathered pre-wire projections. Lossy wire quality is the model gate.
    comp_row(w,cf,sf,y,M,0,0,pos);
    v2::comp_row_rest(w,co,so,full,M,0,0,pos);CK(cudaDeviceSynchronize());
    close("ratio2 odd-start pool vs stock",host(co.comp,cap*HD),host(cf.comp,cap*HD),.003f);
    close("ratio2 carry vs stock",host(co.kvs,2*HD),host(cf.kvs,2*HD),.003f);

}
void output_test(int M) {
    Arena a;const int nr=4,Q=(M+nr-1)/nr,heads=16,K=v2::output_width(heads),stride=v2::output_stride(heads);
    std::vector<Weights> w(nr);std::vector<Scratch>s;std::vector<uint8_t*> wire;std::vector<bf16*> ref;
    auto* pos=a.alloc<int>(M);std::vector<int> positions(M);for(int i=0;i<M;++i)positions[i]=77+i;CK(cudaMemcpy(pos,positions.data(),M*4,cudaMemcpyHostToDevice));
    for(int p=0;p<nr;++p) {
        w[p].wo_a=weight(a,OLORA,GROUP_IN,heads/HPG);w[p].wo_b=weight(a,HID,K);w[p].inv=a.alloc<float>(32);
        s.push_back(scratch(a,M,heads));fill(s[p].o,size_t(M)*heads*HD,.3f);
        wire.push_back(a.alloc<uint8_t>(size_t(nr)*Q*stride));ref.push_back(a.alloc<bf16>(size_t(M)*HID));
        attn_out(w[p],s[p],M,0,heads,ref[p],0,pos);
        v2::output_a(w[p],s[p],M,heads,0,pos);v2::pack_output(s[p],wire[p],M,nr*Q,heads,0);
    }
    auto* recv=a.alloc<uint8_t>(size_t(nr)*Q*stride);auto* parts=a.alloc<bf16>(size_t(nr)*Q*HID);auto* sum=a.alloc<bf16>(size_t(Q)*HID);
    std::vector<std::vector<bf16>> refh;for(int p=0;p<nr;++p)refh.push_back(host(ref[p],size_t(M)*HID));
    for(int rank=0;rank<nr;++rank) {
        int r0=rank*Q,n=std::max(0,std::min(Q,M-r0));
        for(int p=0;p<nr;++p)CK(cudaMemcpyAsync(recv+size_t(p)*Q*stride,wire[p]+size_t(r0)*stride,size_t(Q)*stride,cudaMemcpyDeviceToDevice));
        // Own slice is consumed from send while remote slices use receive.
        v2::output_b(w[rank].wo_b,wire[rank]+size_t(r0)*stride,parts+size_t(rank)*Q*HID,n,s[rank].xf_oa,0);
        for(int p=0;p<nr;++p)if(p!=rank)v2::output_b(w[p].wo_b,recv+size_t(p)*Q*stride,parts+size_t(p)*Q*HID,n,s[rank].xf_oa,0);
        v2::sum_output(parts,sum,n,Q,nr,0);CK(cudaDeviceSynchronize());
        std::vector<bf16> expected(size_t(n)*HID);
        for(size_t j=0;j<expected.size();++j){float v=0;for(int p=0;p<nr;++p)v+=val(refh[p][size_t(r0)*HID+j]);expected[j]=bf16(v);}
        close("wo_a a2a + wo_b vs stock sum",host(sum,size_t(n)*HID),expected,.004f);
        printf("output M=%d rank=%d Q=%d owned=%d PASS\n",M,rank,Q,n);
    }
}
}
int main(){ ds41a::g_use_tc=true;test::wire_test();test::projection_test();test::output_test(65);test::output_test(128);test::output_test(2);CK(cudaGetLastError());puts("PASS v2 attention synthetic tolerance/layout suite; model quality and fabric speed NOT measured"); }
