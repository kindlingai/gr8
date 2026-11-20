// SYNTHETIC correctness only: bounded projections and four-rank FFN, no network.
#include "ds41_v2_test_support.cuh"
// Keep bounded test allocation ownership out of the production scratch API.
namespace tc_test_alloc {
v2test::Arena* arena = nullptr;
template<class T> cudaError_t allocate(T** p, size_t bytes) {
    v2test::insist(arena != nullptr, "FFN test arena active");
    *p = reinterpret_cast<T*>(arena->make<uint8_t>(bytes));
    return cudaSuccess;
}
inline cudaError_t release(void*) { return cudaSuccess; }
}
#define cudaMalloc tc_test_alloc::allocate
#define cudaFree tc_test_alloc::release
#include "../runtime/ds41_v2_ffn.cuh"
#undef cudaFree
#undef cudaMalloc
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <string>
using namespace v2test;
namespace ffn = ds41::v2::ffn;
namespace tc = ds41v2tc;
extern "C" void ds41_attn_set_tc(int);

// The test's public Fabric implementation uses mapped, arena-accounted slots.
namespace g6 {
struct Fabric::Impl {
    Arena memory;
    struct Transfer { unsigned long long epoch; std::vector<uint32_t> sizes; Rs rs; };
    std::deque<Transfer> pending;
};
}
namespace proxy {
std::mutex mutex;
std::condition_variable cv;
g6::Fabric* fabrics[4]{};
g6::Fabric::Impl* impls[4]{};
bool quit = false;
std::thread thread;
void start() {
    quit = false;
    thread = std::thread([] {
        std::unique_lock<std::mutex> lock(mutex);
        while (!quit) {
            bool moved = false;
            for (int r = 0; r < 4; ++r) {
                auto* p = impls[r];
                if (!p || p->pending.empty()) continue;
                const auto& task = p->pending.front();
                const int c = task.rs.order.at(0);
                const auto src = fabrics[r]->dev();
                const auto seq = g6::coll_seq(task.epoch, c, src.ncoll);
                const auto ready = __atomic_load_n(src.doorbell + c, __ATOMIC_ACQUIRE);
                if (ready < seq) continue;
                insist(ready == seq, "proxy observes each epoch");
                const char* send = src.send + g6::fab_send_off(src, int(task.epoch & 1), c);
                for (int peer = 0; peer < 4; ++peer) if (peer != r) {
                    const auto dst = fabrics[peer]->dev();
                    const bool rs = task.rs.mask.bit(c);
                    const size_t bytes = rs ? task.rs.bytes.at(peer) : task.sizes.at(c);
                    char* recv = dst.recv + g6::fab_recv_off(dst, int(task.epoch & 1), c, r);
                    std::memcpy(recv, send + (rs ? size_t(peer) * task.rs.ss : 0), bytes);
                    __atomic_store_n(reinterpret_cast<unsigned long long*>(recv + bytes), seq, __ATOMIC_RELEASE);
                }
                p->pending.pop_front(); moved = true; cv.notify_all();
            }
            if (!moved) cv.wait_for(lock, std::chrono::microseconds(50));
        }
    });
}
void stop() {
    { std::lock_guard<std::mutex> lock(mutex); quit = true; }
    cv.notify_all(); thread.join();
}
}
namespace g6 {
void Fabric::init(const FabricConfig& config) {
    insist(!impl_, "fresh fabric"); cfg_ = config; impl_ = new Impl;
    insist(fab_slot_layout(cfg_, cfg_.ncoll, &dev_), "fabric layout");
    dev_.rank = cfg_.rank; dev_.nr = 4; dev_.compact = 1;
    auto& mem = impl_->memory;
    const size_t ns = 2 * fab_slot_base(dev_, dev_.ncoll), nr = fab_recv_bytes(dev_);
    dev_.send = mem.make<char>(ns, true); dev_.recv = mem.make<char>(nr, true);
    dev_.doorbell = mem.make<unsigned long long>(dev_.ncoll, true);
    dev_.error = mem.make<unsigned>(1, true);
    std::memset(dev_.send, 0, ns); std::memset(dev_.recv, 0, nr);
    std::memset(dev_.doorbell, 0, dev_.ncoll * sizeof(*dev_.doorbell)); *dev_.error = 0;
    proxy::fabrics[cfg_.rank] = this; proxy::impls[cfg_.rank] = impl_;
}
Fabric::~Fabric() {
    if (!impl_) return;
    insist(impl_->pending.empty(), "fabric drained on release");
    proxy::fabrics[cfg_.rank] = nullptr; proxy::impls[cfg_.rank] = nullptr; delete impl_;
}
void Fabric::enqueue_step(unsigned long long epoch, const std::vector<uint32_t>& sizes, const Rs& rs) {
    insist(rs.order.size() == 1, "one collective per submission");
    { std::lock_guard<std::mutex> lock(proxy::mutex); impl_->pending.push_back({epoch, sizes, rs}); }
    proxy::cv.notify_all();
}
bool Fabric::wait_proxy_idle(double seconds) {
    std::unique_lock<std::mutex> lock(proxy::mutex);
    return proxy::cv.wait_for(lock, std::chrono::duration<double>(seconds), [&] { return impl_->pending.empty(); });
}
}
namespace {
template<class T> std::vector<T> read(const T* p, size_t n) {
    std::vector<T> v(n); check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
template<class T> void write(T* p, const std::vector<T>& v) {
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
std::vector<__nv_bfloat16> values(size_t n, unsigned seed) {
    std::vector<__nv_bfloat16> v(n);
    uint32_t state = seed;
    for (auto& x : v) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        // Signed, nonuniform products, with a mean keeping rounding-ULP checks
        // away from cancellation to zero. Inputs are not exact dot fixtures.
        x = __float2bfloat16_rn((float(state % 100003) / 100003.f - .25f) * .25f);
    }
    return v;
}
struct Error {
    double max_abs = 0, square = 0, norm = 0, product = 0, got_norm = 0;
    void add(double got, double ref) {
        insist(std::isfinite(got) && std::isfinite(ref), "finite outputs/reference");
        const double d = got - ref;
        max_abs = std::max(max_abs, std::abs(d)); square += d*d; norm += ref*ref;
        product += got*ref; got_norm += got*got;
    }
    double relative() const { return std::sqrt(square / std::max(norm, 1e-300)); }
    double cosine() const { return product / std::sqrt(std::max(norm * got_norm, 1e-300)); }
};
int ordered(__nv_bfloat16 x) {
    uint16_t b; std::memcpy(&b, &x, sizeof(b));
    return (b & 0x8000) ? 0x8000 - (b & 0x7fff) : 0x8000 + b;
}
std::vector<double> reference(const std::vector<__nv_bfloat16>& x, int ldx,
                              const std::vector<__nv_bfloat16>& w, int M, int N, int K) {
    // Every multiplication and addition is FP64 on exact BF16 inputs.
    std::vector<double> xd(x.size()), wd(w.size()), y(size_t(M)*N);
    for (size_t j=0; j<x.size(); ++j) xd[j] = __bfloat162float(x[j]);
    for (size_t j=0; j<w.size(); ++j) wd[j] = __bfloat162float(w[j]);
    for (int m=0; m<M; ++m) for (int n=0; n<N; ++n) {
        double sum=0;
        for (int k=0; k<K; ++k) sum += xd[size_t(m)*ldx+k]*wd[size_t(n)*K+k];
        y[size_t(m)*N+n]=sum;
    }
    return y;
}
template<class T> void padding(const std::vector<T>& y, int M, int N, int ldy) {
    // Two extra rows plus all per-row padding were poisoned before each call.
    for (int m=0; m<M+2; ++m) for (int n=0; n<ldy; ++n) if (m>=M || n>=N) {
        const auto* p = reinterpret_cast<const unsigned char*>(&y[size_t(m)*ldy+n]);
        for (size_t b=0; b<sizeof(T); ++b) insist(p[b]==0xa7, "row/column sentinel intact");
    }
}
Error float_error(const float* y, const std::vector<double>& ref, int M, int N, int ldy) {
    auto got=read(y,size_t(M+2)*ldy); padding(got,M,N,ldy); Error e;
    for (int m=0; m<M; ++m) for (int n=0; n<N; ++n) e.add(got[size_t(m)*ldy+n], ref[size_t(m)*N+n]);
    insist(e.max_abs < 5e-4 && e.relative() < 2e-5, "FP32 vs FP64 accuracy"); return e;
}
void direct(int M, int N, int K, bool strided) {
    Arena a; Inputs immutable;
    const int ldx=K+(strided?24:0), ldy=N+(strided?10:0);
    auto hx=values(size_t(M)*ldx,17), hw0=values(size_t(N)*K,29), hw1=values(size_t(N)*K,43);
    auto* x=a.make<__nv_bfloat16>(hx.size()); auto* w0=a.make<__nv_bfloat16>(hw0.size()); auto* w1=a.make<__nv_bfloat16>(hw1.size());
    write(x,hx); write(w0,hw0); write(w1,hw1);
    immutable.add(x,hx.size()*2); immutable.add(w0,hw0.size()*2); immutable.add(w1,hw1.size()*2);
    const size_t count=size_t(M+2)*ldy;
    auto* y0=a.make<float>(count); auto* y1=a.make<float>(count); auto* yb=a.make<__nv_bfloat16>(count);
    const auto ref0=reference(hx,ldx,hw0,M,N,K), ref1=reference(hx,ldx,hw1,M,N,K);
    check(cudaMemset(y0,0xa7,count*4));
    check(tc::launch_gemm_f32(x,ldx,w0,N,K,y0,ldy,M,nullptr));
    const Error f=float_error(y0,ref0,M,N,ldy);
    // A perturbed reference must not pass the accuracy comparison.
    auto bad=ref0; bad[0]+=1.; auto yhost=read(y0,count);
    insist(std::abs(yhost[0]-bad[0])>=5e-4,"projection negative control rejected");
    check(cudaMemset(y0,0xa7,count*4)); check(cudaMemset(y1,0xa7,count*4));
    check(tc::launch_gemm2_f32(x,ldx,w0,w1,N,K,y0,y1,ldy,M,nullptr));
    const Error p0=float_error(y0,ref0,M,N,ldy), p1=float_error(y1,ref1,M,N,ldy);
    check(cudaMemset(yb,0xa7,count*2));
    check(tc::launch_gemm_bf16(x,ldx,w0,N,K,yb,ldy,M,nullptr));
    auto b=read(yb,count); padding(b,M,N,ldy); Error rounded, unrounded; int ulp=0;
    for (int m=0; m<M; ++m) for (int n=0; n<N; ++n) {
        const double d=ref0[size_t(m)*N+n]; const auto ref=__float2bfloat16_rn(static_cast<float>(d));
        const auto got=b[size_t(m)*ldy+n];
        ulp=std::max(ulp,std::abs(ordered(got)-ordered(ref)));
        rounded.add(__bfloat162float(got),__bfloat162float(ref)); unrounded.add(__bfloat162float(got),d);
    }
    insist(ulp<=1 && rounded.relative()<5e-4 && unrounded.relative()<5e-3,"BF16 rounded ULP/L2 and unrounded L2");
    immutable.verify(); a.verify();
    printf("PASS direct M=%d N=%d K=%d ldx=%d ldy=%d max_abs=%.9g rel_l2=%.9g bf16_ulp=%d rounded_l2=%.9g unrounded_l2=%.9g\n",
           M,N,K,ldx,ldy,std::max({f.max_abs,p0.max_abs,p1.max_abs}),std::max({f.relative(),p0.relative(),p1.relative()}),ulp,rounded.relative(),unrounded.relative());
}
template<class T> T* offset2(T* p) { return reinterpret_cast<T*>(reinterpret_cast<char*>(p)+2); }
void refusal() {
    Arena a;
    auto* x=a.make<__nv_bfloat16>(16*136); auto* w=a.make<__nv_bfloat16>(34*128);
    auto* f=a.make<float>(16*34); auto* b=a.make<__nv_bfloat16>(16*34);
    // Perturb each structural input independently, including each paired output.
    for (int bad=0; bad<14; ++bad) {
        auto* xx=bad==0?offset2(x):x; auto* ww=bad==1?offset2(w):w;
        auto* ff=bad==2?offset2(f):f; auto* bb=bad==2?offset2(b):b;
        int N=bad==3?31:32, K=bad==4?127:128, ldx=bad==5?129:136, ldy=bad==6?33:34, M=16;
        if (bad==7) xx=nullptr;
        if (bad==8) ww=nullptr;
        if (bad==9) { ff=nullptr; bb=nullptr; }
        if (bad==10) M=0;
        if (bad==11) N=0;
        if (bad==12) ldx=120;
        if (bad==13) ldy=30;
        insist(!tc::can_launch_f32(xx,ldx,ww,N,K,ff,ldy,M),"float predicate refuses");
        insist(tc::launch_gemm_f32(xx,ldx,ww,N,K,ff,ldy,M,nullptr)==cudaErrorInvalidValue,"float launch refuses");
        insist(!tc::can_launch_bf16(xx,ldx,ww,N,K,bb,ldy,M),"BF16 predicate refuses");
        insist(tc::launch_gemm_bf16(xx,ldx,ww,N,K,bb,ldy,M,nullptr)==cudaErrorInvalidValue,"BF16 launch refuses");
        for (int side=0; side<2; ++side) {
            auto* w0=side?w:ww; auto* w1=side?ww:w; auto* f0=side?f:ff; auto* f1=side?ff:f;
            insist(!tc::can_launch_f32_pair(xx,ldx,w0,w1,N,K,f0,f1,ldy,M),"pair predicate refuses");
            insist(tc::launch_gemm2_f32(xx,ldx,w0,w1,N,K,f0,f1,ldy,M,nullptr)==cudaErrorInvalidValue,"pair launch refuses");
        }
    }
    puts("PASS negative pointer/alignment/shape predicates and all launch refusals");
}
template<class Kernel> void resources(const char* name, Kernel kernel) {
    cudaFuncAttributes attr{}; int active=0;
    check(cudaFuncGetAttributes(&attr,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,128,tc::TC_SMEM_BYTES));
    printf("RESOURCE %s registers=%d local_bytes=%zu dynamic_smem=%d active_blocks=%d\n",name,attr.numRegs,attr.localSizeBytes,tc::TC_SMEM_BYTES,active);
    insist(attr.numRegs<=128 && attr.localSizeBytes==0 && active==2,"register/local/occupancy gate");
}
constexpr int H=ffn::H, E=6;
ffn::Layer weights(Arena& a, Inputs& immutable, bool separated) {
    auto* gate=a.make<__nv_bfloat16>(size_t(ffn::NE)*H);
    auto hg=values(size_t(ffn::NE)*H,71);
    if (separated) {
        std::fill(hg.begin(),hg.end(),__float2bfloat16_rn(0.f));
        for (int e=0;e<E;++e) hg[size_t(e)*H+e]=__float2bfloat16_rn(float(e+1));
    }
    write(gate,hg);
    auto* bias=a.make<float>(ffn::NE); std::vector<float> hb(ffn::NE,-10000.f);
    for (int e=0;e<E;++e) hb[e]=separated?float(2*e):0.f;
    write(bias,hb);
    const size_t n1=size_t(E)*(ffn::I/16)*(H/64)*2*mx4::GR, n2=size_t(E)*(H/16)*(ffn::I/64)*mx4::GR;
    auto* fc1=a.make<uint8_t>(n1); auto* fc2=a.make<uint8_t>(n2);
    auto packed=[](uint8_t* p,size_t n,unsigned seed) {
        auto v=data(n,seed);
        for(size_t j=0;j<n;++j) v[j]=(j%mx4::GR<512)?uint8_t(v[j]&0xbb):uint8_t(116+v[j]%4);
        upload(p,v);
    };
    packed(fc1,n1,81); packed(fc2,n2,83);
    const size_t n13=size_t(2)*ffn::I*H, ndown=size_t(H)*ffn::KP;
    auto* w13=a.make<uint8_t>(n13); auto* s13=a.make<uint8_t>(n13/32);
    auto* w2=a.make<uint8_t>(ndown); auto* s2=a.make<uint8_t>(ndown/32);
    fp8(w13,n13,89); fp8(s13,n13/32,97,true); fp8(w2,ndown,101); fp8(s2,ndown/32,103,true);
    auto hw2=download(w2,ndown);
    for(size_t j=0;j<ndown;++j) if(j%ffn::KP>=ffn::I) hw2[j]=0;
    upload(w2,hw2);
    immutable.add(gate,hg.size()*2); immutable.add(bias,hb.size()*4);
    immutable.add(fc1,n1); immutable.add(fc2,n2); immutable.add(w13,n13); immutable.add(s13,n13/32);
    immutable.add(w2,ndown); immutable.add(s2,ndown/32);
    return {gate,bias,bias,fc1,fc2,w13,s13,w2,s2,10.f};
}
struct Snapshot {
    std::vector<float> logits;
    std::vector<int> indices;
    std::vector<__nv_bfloat16> output;
};
void integration(bool separated) {
    // 260 covers every requested M, without allocating unused wide planners.
    constexpr int CAP=260, Q=CAP/4;
    Arena a; Inputs immutable; auto w=weights(a,immutable,separated);
    g6::Fabric fabric[4]; State state[4]; ffn::Scratch scratch[4];
    for (int r=0;r<4;++r) {
        auto& s=state[r]; auto& q=scratch[r];
        tc_test_alloc::arena=&a; check(ffn::prepare(q,CAP)); tc_test_alloc::arena=nullptr;
        g6::FabricConfig cfg{}; cfg.rank=r; cfg.ips.resize(4); cfg.ncoll=int(Coll::Count); cfg.compact=true;
        // Only FfnIn and Ffn0..3 carry data here; unused attention slots
        // retain small valid capacities instead of multiplying the FFN payload.
        cfg.run1=int(Coll::FfnIn); cfg.run2=int(Coll::FfnIn)+1; cfg.run3=int(Coll::Ffn0);
        cfg.slot_bytes=256; cfg.slot_bytes1=size_t(Q)*(H+H/32+48)+64; cfg.slot_bytes2=256;
        cfg.slot_bytes3=4*aligned(size_t(Q)*ffn::COLS*2+8,256)-8;
        fabric[r].init(cfg); check(cudaStreamCreateWithFlags(&s.st,cudaStreamNonBlocking)); init(s,fabric[r],s.st);
        s.y=a.make<__nv_bfloat16>(Q*H); s.ffn_out=a.make<__nv_bfloat16>(Q*H);
        write(s.y,values(Q*H,107+unsigned(r))); immutable.add(s.y,size_t(Q)*H*2);
    }
    check(cudaFuncSetAttribute(dsr::k_gate_logits_ro<ffn::NE>,cudaFuncAttributeMaxDynamicSharedMemorySize,dsr::GRO_SMEM));
    proxy::start();
    for (int M:{60,64,68,132,260}) for (bool gate_ro:{false,true}) {
        std::array<Snapshot,4> baseline;
        for (int mask:{0,1}) {
            for (int r=0;r<4;++r) {
                auto& s=state[r]; auto& q=scratch[r];
                s.g=geometry(M,4,r); s.live=true; ++s.epoch;
                s.order={{int(Coll::FfnIn),int(Coll::Ffn0),int(Coll::Ffn1),int(Coll::Ffn2),int(Coll::Ffn3)},0};
                s.bf16_tc.mask=mask; q.gate_ro=gate_ro;
                check(cudaMemsetAsync(q.logits,0xa7,size_t(Q)*ffn::NE*4,s.st));
                check(cudaMemsetAsync(s.ffn_out,0xa7,size_t(Q)*H*2,s.st));
            }
            for (int r=0;r<4;++r) check(ffn::run(state[r],w,scratch[r]));
            Error logerr, outerr; size_t matches=0,total=0; bool bytes=true;
            for (int r=0;r<4;++r) {
                auto& s=state[r]; auto& q=scratch[r];
                end_layer(s); check(cudaStreamSynchronize(s.st)); check(cudaStreamSynchronize(s.cs));
                insist(fabric[r].wait_proxy_idle(30) && !*s.wire.error,"FFN transport drained without errors");
                Snapshot got{read(q.logits,size_t(s.g.owned)*ffn::NE),read(q.own_idx,size_t(s.g.owned)*ffn::TK),read(s.ffn_out,size_t(s.g.owned)*H)};
                // An independent stock RO invocation checks both SIMT selectors.
                dsr::k_gate_logits_ro<ffn::NE><<<ffn::NE/8,256,dsr::GRO_SMEM,s.st>>>(w.gate,s.y,s.g.owned,q.logits);
                check(cudaGetLastError()); check(cudaStreamSynchronize(s.st));
                auto ro=read(q.logits,got.logits.size());
                for(size_t j=0;j<ro.size();++j) logerr.add(got.logits[j],ro[j]);
                if (!mask) baseline[r]=std::move(got);
                else {
                    const auto& ref=baseline[r];
                    for(size_t j=0;j<got.indices.size();++j) { matches+=got.indices[j]==ref.indices[j]; ++total; }
                    for(size_t j=0;j<got.output.size();++j) outerr.add(__bfloat162float(got.output[j]),__bfloat162float(ref.output[j]));
                    bytes &= std::memcmp(got.output.data(),ref.output.data(),got.output.size()*2)==0;
                    if(M==60) insist(std::memcmp(got.logits.data(),ref.logits.data(),got.logits.size()*4)==0,"owned<16 logits byte-identical");
                }
                auto tail=download(s.ffn_out+size_t(s.g.owned)*H,size_t(Q-s.g.owned)*H*2);
                for(auto v:tail) insist(v==0xa7,"FFN unused rows untouched");
            }
            insist(logerr.max_abs<5e-4 && logerr.relative()<2e-5,"FFN gate logits vs stock RO");
            if(mask) {
                insist(outerr.cosine()>.9999 && outerr.relative()<5e-3,"FFN output cosine/L2");
                if(separated || M==60) insist(bytes && matches==total,"separated routes or fallback output exact");
                printf("PASS FFN M=%d owned=%d gate_ro=%d separated=%d max_abs=%.9g logits_l2=%.9g output_cos=%.12g output_l2=%.9g top6=%zu/%zu memcmp=%d\n",
                    M,(M+3)/4,int(gate_ro),int(separated),logerr.max_abs,logerr.relative(),outerr.cosine(),outerr.relative(),matches,total,bytes?0:1);
            }
        }
    }
    proxy::stop(); immutable.verify(); a.verify();
    for(int r=0;r<4;++r) { release(state[r]); check(cudaStreamDestroy(state[r].st)); ffn::release(scratch[r]); }
}
}
int main() {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1); setenv("CUDA_MODULE_LOADING","EAGER",1);
    setenv("GR8_DS_PF_V2","0",1); setenv("GR8_DS_PF_V2_TIMING","0",1);
    ds41_attn_set_tc(1); check(tc::prepare());
    resources("f32",tc::k_gemm_bf16_tc<float,false>);
    resources("pair",tc::k_gemm_bf16_tc<float,true>);
    resources("bf16",tc::k_gemm_bf16_tc<__nv_bfloat16,false>);
    refusal();
    for(int K:{128,5120}) for(int N:{32,384,512}) for(int M:{1,15,16,17,63,64,65,128,131})
        for(bool strided:{false,true}) direct(M,N,K,strided);
    insist(live==0,"direct scopes released");
    // Resolve peer-wait kernels before asynchronous four-rank launches.
    auto load=[](auto kernel) { cudaFuncAttributes attr{}; check(cudaFuncGetAttributes(&attr,kernel)); };
    load(detail::publish<false>); load(detail::wait<false>); load(detail::unpack<false>);
    load(detail::reduce<false>); load(detail::reduce_vector<false>);
    integration(false); insist(live==0,"general FFN released");
    integration(true); insist(live==0,"separated FFN released");
    summary("bf16_tc");
}
