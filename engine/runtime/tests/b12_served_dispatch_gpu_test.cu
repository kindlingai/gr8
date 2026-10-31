// DEV-only test of the actual ds41_step.cuh dispatch wrappers, not a serving or quality test.
// Build with the DS41 TP4/BF16-trunk flags; run only on an authorized DEV GPU.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>  // ds41_step.cuh is included inside g6rt; this STL header must already be global
#include <memory>
#include <random>
#include <vector>
#include "runtime/rank.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__); std::exit(3); } } while(0)

template<class T> static T* alloc(size_t n) { T* p=nullptr; CK(cudaMalloc(&p,n*sizeof(T))); return p; }
template<class T> static T* up(const std::vector<T>& h) { T* p=alloc<T>(h.size()); CK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
template<class T> static std::vector<T> down(const T* p,size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return h; }

struct Input {
    int M,H=5120;
    std::vector<__nv_bfloat16> streams, sum, norm;
    std::vector<float> mix,pre,post,comb,fn,base,scale;
    explicit Input(int rows):M(rows) {
        std::mt19937 rng(20261028u); std::normal_distribution<float> n(0.f,1.f);
        auto bf=[](float x){return __float2bfloat16_rn(x);};
        streams.resize((size_t)M*4*H); for(auto& x:streams)x=bf(n(rng));
        sum.resize((size_t)M*H); for(auto& x:sum)x=bf(0.5f*n(rng));
        norm.resize(H); for(auto& x:norm)x=bf(1.f+0.1f*n(rng));
        mix.resize((size_t)M*24); for(auto& x:mix)x=0.01f*n(rng);
        pre.resize((size_t)M*4); for(auto& x:pre)x=0.25f+0.05f*n(rng);
        post.resize((size_t)M*4); for(auto& x:post)x=1.f+0.1f*n(rng);
        comb.resize((size_t)M*16); for(auto& x:comb)x=0.25f+0.02f*n(rng);
        fn.resize((size_t)24*4*H); for(auto& x:fn)x=0.01f*n(rng);
        base.resize(24); for(auto& x:base)x=n(rng);
        scale={0.25f,0.35f,0.45f};
    }
};

struct WeightDev {
    float *fn,*base,*scale; __nv_bfloat16* norm;
    explicit WeightDev(const Input& h):fn(up(h.fn)),base(up(h.base)),scale(up(h.scale)),norm(up(h.norm)){}
    ~WeightDev(){cudaFree(fn);cudaFree(base);cudaFree(scale);cudaFree(norm);}
};

struct Side {
    std::unique_ptr<g6rt::Model> m{new g6rt::Model{}};
    explicit Side(const Input& h,const WeightDev& w,bool candidate) {
        auto& x=*m; const size_t R=(size_t)h.M;
        x.ds41=true; x.ds41_draft=false; x.ds41_b12_mhc=candidate; x.hc_rows=h.M;
        x.resid4=up(h.streams); x.hc_mix=up(h.mix); x.hc_pre=up(h.pre);
        x.hc_post=up(h.post); x.hc_comb=up(h.comb);
        x.hc_part=alloc<float>(R*26*(h.H/64));
        x.hc_done=alloc<unsigned>(ds_mhc::done_words(h.M)); CK(cudaMemset(x.hc_done,0,ds_mhc::done_words(h.M)*sizeof(unsigned)));
        x.hc_y=alloc<__nv_bfloat16>(R*h.H);
        x.layers.resize(1);
        x.layers[0].hc[0]={w.fn,true,w.base,w.scale,w.norm};
        x.layers[0].hc[1]=x.layers[0].hc[0];
    }
    ~Side(){auto& x=*m;cudaFree(x.resid4);cudaFree(x.hc_mix);cudaFree(x.hc_pre);cudaFree(x.hc_post);cudaFree(x.hc_comb);cudaFree(x.hc_part);cudaFree(x.hc_done);cudaFree(x.hc_y);}
};

static unsigned bf_ulp(__nv_bfloat16 a,__nv_bfloat16 b) {
    const unsigned x=__bfloat16_as_ushort(a),y=__bfloat16_as_ushort(b);
    if(x==y)return 0;
    if((x>>15)!=(y>>15))return 65535;
    return x>y?x-y:y-x;
}
struct Stats { size_t stream_mism=0,state_mism=0,y_mism=0,nonfinite=0; unsigned state_ulp=0,y_ulp=0; float state_abs=0.f; };
static Stats compare(const Side& old,const Side& cand,int M,int H) {
    Stats z{}; const auto& a=*old.m; const auto& b=*cand.m;
    auto ta=down(a.resid4,(size_t)M*4*H),tb=down(b.resid4,(size_t)M*4*H);
    for(size_t i=0;i<ta.size();++i) {
        if(__bfloat16_as_ushort(ta[i])!=__bfloat16_as_ushort(tb[i])) ++z.stream_mism;
        if(!std::isfinite(__bfloat162float(ta[i])) || !std::isfinite(__bfloat162float(tb[i])))++z.nonfinite;
    }
    auto state=[&](const float* p,const float* q,size_t n) {
        auto v=down(p,n),w=down(q,n);
        for(size_t i=0;i<n;++i) {
            if(!std::isfinite(v[i]) || !std::isfinite(w[i])){++z.nonfinite;continue;}
            const float d=std::fabs(v[i]-w[i]);
            if(d>0)++z.state_mism;
            if(d>z.state_abs)z.state_abs=d;
            const unsigned u=(unsigned)std::ceil(d*8388608.f-1e-4f);
            if(u>z.state_ulp)z.state_ulp=u;
        }
    };
    state(a.hc_mix,b.hc_mix,(size_t)M*24);
    state(a.hc_pre,b.hc_pre,(size_t)M*4);
    state(a.hc_post,b.hc_post,(size_t)M*4);
    state(a.hc_comb,b.hc_comb,(size_t)M*16);
    auto ya=down(a.hc_y,(size_t)M*H),yb=down(b.hc_y,(size_t)M*H);
    for(size_t i=0;i<ya.size();++i) {
        if(!std::isfinite(__bfloat162float(ya[i])) || !std::isfinite(__bfloat162float(yb[i])))++z.nonfinite;
        unsigned u=bf_ulp(ya[i],yb[i]); if(u)++z.y_mism; if(u>z.y_ulp)z.y_ulp=u;
    }
    return z;
}

static bool one(int M,bool fused) {
    Input h(M); WeightDev w(h); Side old(h,w,false),cand(h,w,true);
    auto sum=up(h.sum); auto stepp=alloc<u64>(1); CK(cudaMemset(stepp,0,sizeof(u64)));
    if(fused) {
        g6rt::ds41_hc_consume_prepare_b(*old.m,g6rt::Ds41DevBlock{sum,stepp},0,false,M,0);
        g6rt::ds41_hc_consume_prepare_b(*cand.m,g6rt::Ds41DevBlock{sum,stepp},0,false,M,0);
    } else {
        g6rt::ds41_hc_prepare(*old.m,0,false,M,0);
        g6rt::ds41_hc_prepare(*cand.m,0,false,M,0);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const Stats s=compare(old,cand,M,h.H);
    const bool route=g6rt::ds41_b12_on(*cand.m,M);
    const unsigned long long expected=route?2ull*((M+ds_mhc::DECOMP_STRIP_M-1)/ds_mhc::DECOMP_STRIP_M):0ull;
    const bool pass=s.nonfinite==0 && s.stream_mism==0 && s.state_abs<=1e-5f && s.state_ulp<=16 && s.y_ulp<=1
        && old.m->ds41_b12_kernel_launches==0 && cand.m->ds41_b12_kernel_launches==expected;
    std::printf("B12_SERVED_DEV_PARITY op=%s M=%d route=%d old_launches=%llu cand_launches=%llu expected_launches=%llu nonfinite=%zu streams_mism=%zu state_mism=%zu state_abs=%.3e state_eps_ulp=%u y_mism=%zu y_bf16_ulp=%u verdict=%s\n",
        fused?"consume_prepare":"prepare",M,route?1:0,old.m->ds41_b12_kernel_launches,cand.m->ds41_b12_kernel_launches,expected,
        s.nonfinite,s.stream_mism,s.state_mism,s.state_abs,s.state_ulp,s.y_mism,s.y_ulp,pass?"PASS":"FAIL");
    cudaFree(sum);cudaFree(stepp);
    return pass;
}

int main(){
    bool ok=true;
    for(int M:{48,49,128,512,4046,4096}) for(bool fused:{false,true}) ok=one(M,fused)&&ok;
    std::puts(ok?"B12_SERVED_DEV_GATE_PASS":"B12_SERVED_DEV_GATE_FAIL");
    return ok?0:1;
}
