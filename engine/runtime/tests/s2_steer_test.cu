// S2 batched decode steers each row as solo (K1) decode does. One GPU, synthetic inputs, no model or network.
// Needs -DDEC_S2 (the S2 path); without it the test prints SKIP (EXTRA=-DDEC_S2 scripts/c1_tests.sh <bin_dir>).
//
// Part A (host): the production binding steps. base_bind (layer_base) and s2_bind (s2_layer) run on fresh K1Args (as
// make_k1 leaves them) for dense / steered MoE / unlisted MoE / absent layers, without a profile, in shared and full
// scope: the S2 binding's steering must equal today's and the model's; without a profile both keep make_k1's bytes
// (all of K1Args before the DEC_SLOT slot). A regression that drops k1_steer from s2_bind dies in s2_bind's own check
// (exit != 0) or fails these checks.
// Part B (shared-expert site, k1_tr3_fin): the steering fields come from those bindings (S2 launch: the s2_bind binding;
// K1 launches: the base_bind binding; "old": a binding without k1_steer, i.e. make_k1's zeros, as s2_layer bound it before), and the per-row
// scales reach the model's device scale vector through the production k_step_in from a StepIn. One S2-shaped launch of
// M rows from several sessions against a K1-shaped launch of each session alone and of every row alone (rows renumbered
// from 0, the same routed / shared partials, weights, scale): identical bytes per row. Scale-0 rows equal unsteered
// bytes, steered rows differ from them, and the old binding misses every steered row.
// Part C (attention site, k_ar_add_norm): the same S2-vs-K1 row comparison for the consume (residual, normed output and,
// under DEC_SLOT, the row's fragment-table words).
// The comparisons establish row invariance given identical pre-steering data, not end-to-end solo/batch identity, and
// read device memory after synchronization (no CPU / NIC publication check; steering_shared_test has that).
// Pass: last line "s2_steer_test: N checks, 0 failures  ALL IDENTICAL".
#ifndef DEC_S2
#include <cstdio>
int main()
{
    std::printf("SKIP s2_steer_test: build with -DDEC_S2 (EXTRA=-DDEC_S2 scripts/c1_tests.sh <bin_dir>)\n");
    return 0;
}
#else
#include "../prefill.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__); std::exit(2); } } while(0)
using namespace g6rt;
static int checks=0, failures=0;
static void check(bool ok,const char* what,const char* ctx="") { ++checks; if(!ok) { ++failures; std::printf("FAIL %s %s\n",what,ctx); } }
struct Arena {
    std::vector<void*> owned;
    template<class T> T* alloc(size_t n) { T* p; CK(cudaMalloc(&p,std::max<size_t>(n,1)*sizeof(T))); owned.push_back(p); CK(cudaMemset(p,0,std::max<size_t>(n,1)*sizeof(T))); return p; }
    ~Arena() { for(void* p:owned) cudaFree(p); }
};
template<class T> static void put(T* p,const std::vector<T>& v) { CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> static std::vector<T> get(const T* p,size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }

// ------------------------------------------------------------------------------------------------- Part A
static std::unique_ptr<bind::Layer> fresh_layer() {
    std::unique_ptr<bind::Layer> L(new bind::Layer());
    std::memset(&L->k1.a,0,sizeof L->k1.a);                  // make_k1: memset(&a, 0, sizeof a)
    return L;
}
#ifdef DEC_SLOT
constexpr size_t K1_HEAD=offsetof(K1Args,slot);              // the DEC_SLOT slot is K1Args' last member
#else
constexpr size_t K1_HEAD=sizeof(K1Args);
#endif
static bool steer_is(const K1Args& a,const float* dir,const float* scale,float alpha,const float* part,const unsigned* count) {
    return a.steer_dir==dir && a.steer_scale==scale && a.steer_alpha==alpha && a.steer_part==part && a.steer_count==count;
}
static void wiring() {
    Model m;
    static unsigned tickets[2*5+1];                     // base_bind's DEC_SLOT slots point into it (never dereferenced here)
    m.tickets=tickets;
    const float* d0=reinterpret_cast<const float*>(uintptr_t(0x1000));
    const float* d1=reinterpret_cast<const float*>(uintptr_t(0x2000));
    float* scale=reinterpret_cast<float*>(uintptr_t(0x3000));
    float* part=reinterpret_cast<float*>(uintptr_t(0x4000));
    unsigned* count=reinterpret_cast<unsigned*>(uintptr_t(0x5000));
    m.layers.resize(5);
    const bool moe[5]={false,true,true,true,true};
    for(int i=0;i<5;++i) m.layers[i].moe=moe[i];
    const auto zero=fresh_layer();
    // no profile (steering off): both bindings keep make_k1's bytes before the slot
    for(int i=0;i<5;++i) {
        auto base=fresh_layer(), s2=fresh_layer();
        base_bind(m,i,*base); s2_bind(m,i,*base,*s2);
        check(std::memcmp(&base->k1.a,&zero->k1.a,K1_HEAD)==0 && std::memcmp(&s2->k1.a,&zero->k1.a,K1_HEAD)==0,
              "steering off: base_bind / s2_bind leave make_k1's K1Args bytes");
#ifdef DEC_SLOT
        check(std::memcmp(&s2->k1.a.slot,&base->k1.a.slot,sizeof(dslot::Slot))==0,"s2_bind takes today's K1 slot");
#endif
    }
    m.steering_scale=scale; m.steering_part=part; m.steering_count=count;
    m.steering_dir={nullptr,d0,nullptr,d1};             // layer 2: MoE outside the profile; layer 4: past the vector
    m.steering_alpha={0.f,3.f,0.f,2.5f};
    m.steering_scope=steering::Scope::Shared;
    for(int i=0;i<5;++i) {
        auto base=fresh_layer(), s2=fresh_layer(), old=fresh_layer();
        base_bind(m,i,*base); s2_bind(m,i,*base,*s2);
        const bool want=moe[i] && i<4;
        if(want) {
            check(steer_is(base->k1.a,m.steering_dir[i],scale,m.steering_alpha[i],part,count),"shared scope: today's MoE layer takes the model's steering");
            check(steer_is(s2->k1.a,m.steering_dir[i],scale,m.steering_alpha[i],part,count),"shared scope: the S2 layer takes the same steering");
        } else {
            check(std::memcmp(&base->k1.a,&zero->k1.a,K1_HEAD)==0 && std::memcmp(&s2->k1.a,&zero->k1.a,K1_HEAD)==0,
                  "shared scope: dense / absent layer keeps make_k1's bytes in both bindings");
        }
        check(s2_steer_same(s2->k1,base->k1),"S2 and today's binding agree");
        if(m.steering_dir.size()>size_t(i) && m.steering_dir[i]) check(!s2_steer_same(old->k1,base->k1),"the old S2 binding (zero fields) disagrees");
    }
    m.steering_scope=steering::Scope::Full;
    for(int i=0;i<5;++i) {
        auto base=fresh_layer(), s2=fresh_layer();
        base_bind(m,i,*base); s2_bind(m,i,*base,*s2);
        check(std::memcmp(&base->k1.a,&zero->k1.a,K1_HEAD)==0 && std::memcmp(&s2->k1.a,&zero->k1.a,K1_HEAD)==0,
              "full scope: finalize unsteered in both bindings (the MLP consume steers)");
    }
}

// the model the kernel launches take their steering from: one steered MoE layer, the profile's alpha, device direction,
// scale vector (k_step_in), dot scratch; its today / S2 / old (no k1_steer) bindings
struct Env {
    Arena mem;
    Model m;
    std::unique_ptr<bind::Layer> base, s2, old;
    uint32_t* in=nullptr; u64* stepp=nullptr; int *tok=nullptr,*pos=nullptr,*tgt=nullptr; glue::SampleRow* srow=nullptr;
};
static std::vector<float> direction();
static Env* env_make() {
    Env* v=new Env;
    Model& m=v->m;
    m.layers.resize(1); m.layers[0].moe=true;
    m.tickets=v->mem.alloc<unsigned>(3);                  // [2*nl+1], as the model allocates them
    float* dd=v->mem.alloc<float>(HID); put(dd,direction());
    m.steering_dir={dd}; m.steering_alpha={3.f};          // ablit-behavior-v1 embeds alpha 3.0
    m.steering_scope=steering::Scope::Shared;
    m.steering_scale=v->mem.alloc<float>(MX); m.steering_slot=v->mem.alloc<int>(MX); m.steering_part=v->mem.alloc<float>(size_t(MX)*48); m.steering_count=v->mem.alloc<unsigned>(MX);
    v->base=fresh_layer(); v->s2=fresh_layer(); v->old=fresh_layer();
    base_bind(m,0,*v->base); s2_bind(m,0,*v->base,*v->s2);
    v->in=v->mem.alloc<uint32_t>(sizeof(StepIn)/4+1); v->stepp=v->mem.alloc<u64>(1);
    v->tok=v->mem.alloc<int>(MX); v->pos=v->mem.alloc<int>(MX); v->tgt=v->mem.alloc<int>(MX); v->srow=v->mem.alloc<glue::SampleRow>(MX);
    return v;
}
// the step's per-row scales -> the model's device scale vector, through StepIn and the production k_step_in
static void transport(Env& v,const std::vector<float>& scales) {
    StepIn si; std::memset(&si,0,sizeof si);
    si.step=1; si.M=int(scales.size()); si.context=1;
    for(size_t r=0;r<scales.size();++r) si.steering_scale[r]=scales[r];
    CK(cudaMemcpy(v.in,&si,sizeof si,cudaMemcpyHostToDevice));
    k_step_in<<<1,128>>>(StepCtl{v.in,v.stepp,v.tok,v.pos,v.tgt,v.m.steering_scale,v.m.steering_slot,v.srow,nullptr,nullptr,0});
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const auto got=get(v.m.steering_scale,MX);
    bool ok=true; for(int r=0;r<MX;++r) ok &= got[r]==(size_t(r)<scales.size() ? scales[r]:0.f);
    check(ok,"k_step_in: per-row scales, padding +0");
}
static Env* ENV=nullptr;

// ------------------------------------------------------------------------------------------------- Part B
// Synthetic step of M rows, plan 0. Routed slots as the production layout of the fixture (two slots per 8-row tile,
// column = row % 8; some routes absent), the shared expert at slot n_slots + m / 8, column m % 8.
struct FinStep {
    int M=0;
    int nslots=0, nsh=0;
    std::vector<int> experts, slots, cols; std::vector<float> weights;
    std::vector<__half> z, ys; std::vector<float> scales;
};
constexpr int FB=2, FE=3;
static FinStep make_step(int M,int seed) {
    FinStep s; s.M=M; s.nsh=(M+7)/8; s.nslots=2*s.nsh;
    s.experts.assign(SMAX,0); s.slots.assign(G6_MX*8,-1); s.cols.assign(G6_MX*8,0); s.weights.assign(G6_MX*8,0.f);
    for(int q=0;q<s.nslots;++q) s.experts[q]=q%2;
    for(int m=0;m<M;++m) for(int k=0;k<8;++k) {
        const int t=m*8+k;
        if((k+m+seed)%5!=0) { s.slots[t]=2*(m/8)+k%2; s.cols[t]=m%8; s.weights[t]=float(1+(k+m+seed)%3)/16.f; }
    }
    s.z.resize(size_t(s.nslots+s.nsh)*FB*8*HID);
    for(int q=0;q<s.nslots+s.nsh;++q) for(int b=0;b<FB;++b) for(int m=0;m<8;++m) for(int c=0;c<HID;++c) {
        const float v=c%128==0 ? (c/128%2 ? -1.f:1.f)*(1.f+float((q+b+m+seed)%5)/8.f) : float((c*7+q*3+b+m*5+seed)%13-6)/512.f;
        s.z[((size_t(q)*FB+b)*8+m)*HID+c]=__float2half_rn(v);
    }
    s.ys.resize(size_t(M+1)*HID);
    for(int m=0;m<M;++m) for(int c=0;c<HID;++c) s.ys[size_t(m+1)*HID+c]=__float2half_rn(float((c*3+m*7+seed)%17-8)/64.f);
    s.scales.assign(M,0.f);
    return s;
}
// rows [r0, r0 + n) of s as their own step (K1: one session's rows; n = 1: one row), rows renumbered from 0. Each
// (row, slot parity) keeps its partials: z of (slot 2 (r / 8) + u, column r % 8) -> (slot 2 (j / 8) + u, column j % 8),
// same expert (parity), same routes, weights and scale.
static FinStep sub_step(const FinStep& s,int r0,int n) {
    FinStep t; t.M=n; t.nsh=(n+7)/8; t.nslots=2*t.nsh;
    t.experts.assign(SMAX,0); t.slots.assign(G6_MX*8,-1); t.cols.assign(G6_MX*8,0); t.weights.assign(G6_MX*8,0.f);
    for(int q=0;q<t.nslots;++q) t.experts[q]=q%2;
    t.z.assign(size_t(t.nslots+t.nsh)*FB*8*HID,__float2half_rn(0.f));
    t.ys.assign(size_t(n+1)*HID,__float2half_rn(0.f));
    auto copyz=[&](int qs,int cs,int qd,int cd) { for(int b=0;b<FB;++b) std::memcpy(&t.z[((size_t(qd)*FB+b)*8+cd)*HID],&s.z[((size_t(qs)*FB+b)*8+cs)*HID],HID*2); };
    for(int j=0;j<n;++j) {
        const int r=r0+j;
        for(int k=0;k<8;++k) {
            const int ts=r*8+k, td=j*8+k;
            if(s.slots[ts]<0) continue;
            t.slots[td]=2*(j/8)+k%2; t.cols[td]=j%8; t.weights[td]=s.weights[ts];
            if(s.experts[s.slots[ts]]!=t.experts[t.slots[td]]) { std::fprintf(stderr,"sub_step: expert remap\n"); std::exit(2); }
        }
        for(int u=0;u<2;++u) copyz(2*(r/8)+u,r%8,2*(j/8)+u,j%8);
        copyz(s.nslots+r/8,r%8,t.nslots+j/8,j%8);
        std::memcpy(&t.ys[size_t(j+1)*HID],&s.ys[size_t(r+1)*HID],HID*2);
    }
    t.scales.assign(s.scales.begin()+r0,s.scales.begin()+r0+n);
    return t;
}
// one k1_tr3_fin launch of step s with the steering of binding `bound` (its fields, the model's scale vector written by
// k_step_in from s.scales); returns the M x HID output words (out, or the device send slot under slot)
static std::vector<unsigned short> run_fin(const FinStep& s,bool sh,bool bf,bool slot,const K1Args& bound) {
    Arena mem;
    const int M=s.M;
    std::vector<long long> tab(FE*6,0);
    for(int e=0;e<FE;++e) { tab[e*6+2]=e*FB; tab[e*6+3]=e==1 ? 1:FB; }
    std::vector<__half> sg(2*HID), v3(FE*FB*HID), vb(FE*FB*geom::HB);
    for(int c=0;c<2*HID;++c) sg[c]=__float2half_rn(c%3==0 ? -1.f:1.f);
    for(int r=0;r<FE*FB;++r) for(int c=0;c<HID;++c) v3[size_t(r)*HID+c]=__float2half_rn((c%3==0 ? -1.f:1.f)*float(2+(r+c/128)%3)/8.f);
    for(int r=0;r<FE*FB;++r) for(int b=0;b<geom::HB;++b) vb[r*geom::HB+b]=__float2half_rn(float(2+(r+b)%3)/8.f);
    K1Args a{}; a.M=M; a.n_slots=s.nslots; a.n_plans=1; a.B=FB; a.sh_e=2; a.out_bf16=bf; a.ysh=1;
    a.ctl=mem.alloc<int>(3);
    a.slot_expert=mem.alloc<int>(s.experts.size()); put(const_cast<int*>(a.slot_expert),s.experts);
    a.tok_slot=mem.alloc<int>(s.slots.size()); put(const_cast<int*>(a.tok_slot),s.slots);
    a.tok_col=mem.alloc<int>(s.cols.size()); put(const_cast<int*>(a.tok_col),s.cols);
    a.tok_w=mem.alloc<float>(s.weights.size()); put(const_cast<float*>(a.tok_w),s.weights);
    a.t_tab=mem.alloc<long long>(tab.size()); put(const_cast<long long*>(a.t_tab),tab);
    a.t_sg=mem.alloc<__half>(sg.size()); put(const_cast<__half*>(a.t_sg),sg);
    a.t_v3=mem.alloc<__half>(v3.size()); put(const_cast<__half*>(a.t_v3),v3);
    a.t_v3b=mem.alloc<__half>(vb.size()); put(const_cast<__half*>(a.t_v3b),vb);
    a.z=mem.alloc<__half>(s.z.size()); put(a.z,s.z);
    a.ypart=mem.alloc<__half>(s.ys.size()); put(a.ypart,s.ys);
    const size_t N=size_t(M)*HID;
    a.out=mem.alloc<__half>(N+HID); CK(cudaMemset(a.out,0x5a,(N+HID)*2));
    transport(*ENV,s.scales);
    a.steer_dir=bound.steer_dir; a.steer_scale=bound.steer_scale; a.steer_alpha=bound.steer_alpha;
    a.steer_part=bound.steer_part; a.steer_count=bound.steer_count;
    const unsigned short* target=reinterpret_cast<const unsigned short*>(a.out);
#ifdef DEC_SLOT
    if(slot) {
        const size_t stride=(N+HID)*2;
        char* send=mem.alloc<char>(4*stride); CK(cudaMemset(send,0x5a,4*stride));
        auto* bell=mem.alloc<unsigned long long>(2); auto* ticket=mem.alloc<unsigned>(1); auto* step=mem.alloc<u64>(1);
        const u64 one=1; CK(cudaMemcpy(step,&one,sizeof one,cudaMemcpyHostToDevice));
        a.slot={send,bell,step,2*stride,stride,2,1,ticket,nullptr};
        target=reinterpret_cast<const unsigned short*>(send+((1&1)*2+1)*stride);
    }
#else
    if(slot) { std::fprintf(stderr,"slot case without DEC_SLOT\n"); std::exit(2); }
#endif
    // as bind.cuh make_k1 launches it: one CTA per (row, 128-column block), 8 routed warps + warp 8 for SH
    if(sh) k1_tr3_fin<true><<<M*geom::HB,288>>>(a); else k1_tr3_fin<false><<<M*geom::HB,256>>>(a);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const auto cnt=get(ENV->m.steering_count,MX);
    check(std::all_of(cnt.begin(),cnt.end(),[](unsigned x){return x==0u;}),"finalize: arrival counters reset");
    return get(target,N);
}
static bool row_eq(const std::vector<unsigned short>& a,int ra,const std::vector<unsigned short>& b,int rb) {
    return std::memcmp(a.data()+size_t(ra)*HID,b.data()+size_t(rb)*HID,HID*2)==0;
}
static std::vector<float> direction() {
    std::vector<float> d(HID); double n=0;
    for(int c=0;c<HID;++c) { d[c]=(c/128%2 ? -1.f:1.f)*(1.f+float(c%11)/16.f); n+=double(d[c])*d[c]; }
    for(int c=0;c<HID;++c) d[c]=float(d[c]/std::sqrt(n));
    return d;
}
// the scale of session j under pattern p: 0 forced 1.0 (the served --steering-force 1.0), 1 per-session mixed, 2 all zero
static float pattern_scale(int p,int j) { static const float mix[4]={1.f,0.f,.5f,2.f}; return p==0 ? 1.f : p==1 ? mix[j%4] : 0.f; }
static void fin_layout(bool sh,bool bf,bool slot,const std::vector<int>& sess,int pat) {
    int M=0; for(int n:sess) M+=n;
    FinStep s=make_step(M,M*3+int(sess.size()));
    { int r=0; for(size_t j=0;j<sess.size();++j) for(int k=0;k<sess[j];++k) s.scales[r++]=pattern_scale(pat,int(j)); }
    char ctx[160]; std::snprintf(ctx,sizeof ctx,"(SH=%d bf16=%d slot=%d M=%d S=%zu pat=%d)",sh,bf,slot,M,sess.size(),pat);
    const K1Args& S2=ENV->s2->k1.a; const K1Args& K1=ENV->base->k1.a;
    const auto s2=run_fin(s,sh,bf,slot,S2);                      // the S2 step, s2_bind's binding
    const auto pre=run_fin(s,sh,bf,slot,ENV->old->k1.a);         // the S2 step with the old binding = unsteered bytes
    int r0=0, steered=0, live=0, missed=0;
    bool sess_ok=true, row_ok=true, off_ok=true;
    for(size_t j=0;j<sess.size();++j) {
        const int n=sess[j];
        const auto k1=run_fin(sub_step(s,r0,n),sh,bf,slot,K1);                  // the session alone (K1, base_bind)
        for(int r=0;r<n;++r) {
            sess_ok &= row_eq(s2,r0+r,k1,r);
            const auto one=run_fin(sub_step(s,r0+r,1),sh,bf,slot,K1);              // the row alone (K1, M = 1)
            row_ok &= row_eq(s2,r0+r,one,0);
            if(s.scales[r0+r]!=0.f) { ++steered; live+=!row_eq(s2,r0+r,pre,r0+r); missed+=!row_eq(pre,r0+r,k1,r); }
            else off_ok &= row_eq(s2,r0+r,pre,r0+r);
        }
        r0+=n;
    }
    check(sess_ok,"shared site: S2 row bytes == the session's own K1 launch",ctx);
    check(row_ok,"shared site: S2 row bytes == the row's own M=1 K1 launch",ctx);
    check(off_ok,"shared site: scale-0 rows == unsteered bytes",ctx);
    check(live==steered,"shared site: every steered row differs from unsteered (steering live)",ctx);
    check(missed==steered,"shared site: the old S2 binding misses every steered row (bug reproduced)",ctx);
    std::printf("fin %s rows %d steered %d differ-from-unsteered %d old-binding-mismatch %d\n",ctx,M,steered,live,missed);
}

// ------------------------------------------------------------------------------------------------- Part C
static float cpartial(int p,int m,int c,int seed) { return float((c*13+m*7+p*3+seed)%97-48)/64.f; }
struct ConsOut { std::vector<float> resid; std::vector<__half> out; std::vector<u32> xf, xb; };
// one k_ar_add_norm launch (collective 0, step 1) over rows g0..g0+M-1 of the global synthetic data
static ConsOut run_cons(int M,int g0,const std::vector<float>& scales_global,const std::vector<float>& d,float alpha,int seed) {
    Arena a;
    const int nr=g6nr::NR, rank=0;
    const size_t stride=size_t(M)*HID*2+256;
    g6::FabricDev f{}; f.nr=nr; f.rank=rank; f.ncoll=2; f.stride=stride;
    std::vector<char> send(4*stride,0), recv(4*nr*stride,0);
    for(int p=0;p<nr;++p) for(int j=0;j<M;++j) for(int c=0;c<HID;++c) {
        const auto v=__float2bfloat16(cpartial(p,g0+j,c,seed));
        const size_t off=size_t(j)*HID*2+c*2;
        char* dst=p==rank ? send.data()+2*stride+off : recv.data()+(2*nr+p)*stride+off;
        std::memcpy(dst,&v,sizeof v);
    }
    f.send=a.alloc<char>(send.size()); f.recv=a.alloc<char>(recv.size()); put(f.send,send); put(f.recv,recv);
    f.doorbell=a.alloc<unsigned long long>(2); f.error=a.alloc<unsigned>(1);
    auto* step=a.alloc<u64>(1); const u64 one=1; CK(cudaMemcpy(step,&one,sizeof one,cudaMemcpyHostToDevice));
    std::vector<float> resid(size_t(M)*HID); std::vector<__nv_bfloat16> w(HID);
    for(int j=0;j<M;++j) for(int c=0;c<HID;++c) resid[size_t(j)*HID+c]=float(((g0+j)*31+c)%53-26)/128.f;
    for(int c=0;c<HID;++c) w[c]=__float2bfloat16(float(3+c%5)/4.f);
    auto* dr=a.alloc<float>(resid.size()); put(dr,resid);
    auto* dw=a.alloc<__nv_bfloat16>(HID); put(dw,w);
    auto* dd=a.alloc<float>(HID); put(dd,d);
    std::vector<float> sc(scales_global.begin()+g0,scales_global.begin()+g0+M);
    auto* ds=a.alloc<float>(M); put(ds,sc);
    auto* out=a.alloc<__half>(size_t(M)*HID);
    glue::ArNormArgs args{f,step,0,M,dr,dw,out,glue::RMS_EPS};
    args.steer={dd,ds,0.f,alpha};
#ifdef DEC_SLOT
    const size_t fw=size_t((M+7)/8)*KSH*16*4;
    auto* xf=a.alloc<u32>(fw); auto* xb=a.alloc<u32>(fw);
    args.xf16=xf; args.xbf16=xb;
#endif
    glue::k_ar_add_norm<<<dim3(glue::RN_CL,M),glue::RN_T>>>(args);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ConsOut o; o.resid=get(dr,resid.size()); o.out=get(out,size_t(M)*HID);
#ifdef DEC_SLOT
    o.xf=get(xf,fw); o.xb=get(xb,fw);
#endif
    return o;
}
static bool cons_row_eq(const ConsOut& a,int ra,const ConsOut& b,int rb) {
    bool ok=std::memcmp(a.resid.data()+size_t(ra)*HID,b.resid.data()+size_t(rb)*HID,HID*4)==0 &&
            std::memcmp(a.out.data()+size_t(ra)*HID,b.out.data()+size_t(rb)*HID,HID*2)==0;
#ifdef DEC_SLOT
    // the row's fragment-table words: uint4 index tile * KSH * 16 + ks * 16 + (m % 8) * 2 + half (glue.cuh frag_store)
    const uint4* xa=reinterpret_cast<const uint4*>(a.xf.data()); const uint4* xb=reinterpret_cast<const uint4*>(b.xf.data());
    const uint4* ya=reinterpret_cast<const uint4*>(a.xb.data()); const uint4* yb=reinterpret_cast<const uint4*>(b.xb.data());
    for(int ks=0;ks<KSH;++ks) for(int h=0;h<2;++h) {
        const size_t ia=size_t(ra>>3)*KSH*16+size_t(ks)*16+(ra&7)*2+h, ib=size_t(rb>>3)*KSH*16+size_t(ks)*16+(rb&7)*2+h;
        ok &= std::memcmp(&xa[ia],&xb[ib],16)==0 && std::memcmp(&ya[ia],&yb[ib],16)==0;
    }
#endif
    return ok;
}
static void cons_layout(const std::vector<int>& sess,int pat) {
    int M=0; for(int n:sess) M+=n;
    std::vector<float> sc(M), zero(M,0.f);
    { int r=0; for(size_t j=0;j<sess.size();++j) for(int k=0;k<sess[j];++k) sc[r++]=pattern_scale(pat,int(j)); }
    const auto d=direction(); const float alpha=3.f; const int seed=M+7*int(sess.size());
    char ctx[120]; std::snprintf(ctx,sizeof ctx,"(M=%d S=%zu pat=%d nr=%d)",M,sess.size(),pat,g6nr::NR);
    const auto s2=run_cons(M,0,sc,d,alpha,seed);
    const auto un=run_cons(M,0,zero,d,alpha,seed);
    bool sess_ok=true, row_ok=true, off_ok=true; int steered=0, live=0, r0=0;
    for(size_t j=0;j<sess.size();++j) {
        const int n=sess[j];
        const auto k1=run_cons(n,r0,sc,d,alpha,seed);
        for(int r=0;r<n;++r) {
            sess_ok &= cons_row_eq(s2,r0+r,k1,r);
            row_ok &= cons_row_eq(s2,r0+r,run_cons(1,r0+r,sc,d,alpha,seed),0);
            if(sc[r0+r]!=0.f) { ++steered; live+=!cons_row_eq(s2,r0+r,un,r0+r); } else off_ok &= cons_row_eq(s2,r0+r,un,r0+r);
        }
        r0+=n;
    }
    check(sess_ok,"attention site: S2 row bytes == the session's own K1 launch",ctx);
    check(row_ok,"attention site: S2 row bytes == the row's own M=1 K1 launch",ctx);
    check(off_ok,"attention site: scale-0 rows == unsteered bytes",ctx);
    check(live==steered,"attention site: every steered row differs from unsteered",ctx);
    std::printf("consume %s rows %d steered %d differ-from-unsteered %d\n",ctx,M,steered,live);
}

int main() {
    wiring();
    ENV=env_make();
    // S2 step layouts (rows per session, session-major): MTP 2 / 3 rows per session at C2 .. C8, uneven sessions, an 8-row
    // tile boundary inside a session, the widest step; {n} alone = a solo step of n rows.
    const std::vector<std::vector<int>> layouts={{1,1},{3,3},{2,2,2,2},{3,3,3,3},{1,8},{3,4},{5,6,5},{8,8},{3,3,3,3,3,3,3,3},
                                                 {8,8,8,8},{4},{9}};
    std::vector<bool> slots={false};
#ifdef DEC_SLOT
    slots.push_back(true);
#endif
    for(bool sh:{false,true}) for(bool bf:{false,true}) for(bool slot:slots) for(const auto& L:layouts) for(int pat=0;pat<3;++pat) {
        if(G6_MX<32) { int M=0; for(int n:L) M+=n; if(M>G6_MX) continue; }
        // keep the run short: every layout at the served form (SH, BF16, slot when built); the rest on a subset
        const bool served=sh && bf && slot==(slots.size()>1);
        if(!served && L.size()>4) continue;
        fin_layout(sh,bf,slot,L,pat);
    }
    for(const auto& L:layouts) for(int pat=0;pat<3;++pat) {
        if(G6_MX<32) { int M=0; for(int n:L) M+=n; if(M>G6_MX) continue; }
        cons_layout(L,pat);
    }
    std::printf("s2_steer_test: %d checks, %d failures%s\n",checks,failures,failures ? "" : "  ALL IDENTICAL");
    return failures ? 1 : 0;
}
#endif
