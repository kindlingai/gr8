// Direct shared-expert finalizers, synthetic inputs, one GPU, no model or network.
// G6_STEERING_TEST_BASELINE builds this same stock harness against the parent tree.
#include "../prefill.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__); std::exit(2); } } while(0)
static int checks=0, failures=0;
static void check(bool ok,const char* what) { ++checks; if(!ok) { ++failures; std::printf("FAIL %s\n",what); } }
struct Arena {
    std::vector<void*> owned;
    template<class T> T* alloc(size_t n) { T* p; CK(cudaMalloc(&p,n*sizeof(T))); owned.push_back(p); CK(cudaMemset(p,0,n*sizeof(T))); return p; }
    ~Arena() { for(void* p:owned) cudaFree(p); }
};
template<class T> static void put(T* p,const std::vector<T>& v) { CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> static std::vector<T> get(const T* p,size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
static void append(FILE* f,const void* p,size_t n) { if(f && std::fwrite(p,1,n,f)!=n) std::exit(2); }
static float rounded_half(float x) { return __half2float(__float2half_rn(x)); }
static unsigned short encode(float x,bool bf) { unsigned short u; if(bf) { auto q=__float2bfloat16_rn(x); std::memcpy(&u,&q,2); } else { auto q=__float2half_rn(x); std::memcpy(&u,&q,2); } return u; }
static float decode(unsigned short u,bool bf) { if(bf) { __nv_bfloat16 q; std::memcpy(&q,&u,2); return __bfloat162float(q); } __half q; std::memcpy(&q,&u,2); return __half2float(q); }
// Independent scalar Walsh transform, normalized exactly as the production hidden rotation.
static void had(float* v) { for(int s=1;s<128;s*=2) for(int i=0;i<128;i+=2*s) for(int j=0;j<s;++j) { const float a=v[i+j],b=v[i+j+s]; v[i+j]=a+b; v[i+j+s]=a-b; } for(int j=0;j<128;++j) v[j]*=SQ128; }
static std::vector<float> direction() { std::vector<float> d(HID); for(int c=0;c<HID;++c) d[c]=(c/128%2 ? -1.f:1.f)/std::sqrt(float(HID)); return d; }
static bool close(const std::vector<unsigned short>& out,const std::vector<float>& want,bool bf,float& error) {
    bool ok=true; error=0;
    for(size_t i=0;i<want.size();++i) { const float a=decode(out[i],bf),e=std::fabs(a-want[i]); error=std::max(error,e); ok &= std::isfinite(a) && e<(bf ? .009f:.0012f)*(1+std::fabs(want[i])); }
    return ok;
}
#ifdef DEC_SLOT
// Observe the actual doorbell concurrently, then snapshot data before kernel completion can hide an early publish.
__global__ void observe(const unsigned long long* bell,unsigned long long seq,const unsigned short* src,unsigned short* dst,size_t n,int* timed_out) {
    __shared__ int ready;
    if(threadIdx.x==0) {
        const unsigned long long start=clock64(); unsigned long long v=0;
        do { asm volatile("ld.acquire.sys.global.u64 %0, [%1];":"=l"(v):"l"(bell):"memory"); } while(v!=seq && clock64()-start<2000000000ull);
        ready=v==seq; if(!ready) *timed_out=1;
    }
    __syncthreads();
    if(ready) for(size_t i=threadIdx.x;i<n;i+=blockDim.x) dst[i]=reinterpret_cast<const volatile unsigned short*>(src)[i];
}
#endif

static void decode_case(bool sh,bool bf,int M,bool slot,int mode,int repeats,FILE* dump) {
    Arena mem;
    constexpr int B=2, E=3, plans=2;
    const int nslots=2*((M+7)/8), nsh=(M+7)/8;
    const size_t N=size_t(M)*HID, guarded=N+HID;
    std::vector<long long> tab(E*6,0);
    for(int e=0;e<E;++e) { tab[e*6+2]=e*B; tab[e*6+3]=e==1 ? 1:B; }
    std::vector<int> experts(plans*SMAX,0), slots(plans*G6_MX*8,-1), cols(slots.size(),0);
    std::vector<float> weights(slots.size(),0);
    for(int p=0;p<plans;++p) for(int s=0;s<nslots;++s) experts[p*SMAX+s]=s%2;
    for(int p=0;p<plans;++p) for(int m=0;m<M;++m) for(int k=0;k<8;++k) {
        const int t=(p*G6_MX+m)*8+k;
        if((k+m)%5!=0) { slots[t]=2*(m/8)+k%2; cols[t]=m%8; weights[t]=float(1+(k+p)%3)/16.f; }
    }
    std::vector<__half> sg(2*HID), v3(E*B*HID), vb(E*B*48);
    for(int c=0;c<2*HID;++c) sg[c]=__float2half_rn(c%3==0 ? -1.f:1.f);
    for(int r=0;r<E*B;++r) for(int c=0;c<HID;++c) v3[size_t(r)*HID+c]=__float2half_rn((c%3==0 ? -1.f:1.f)*float(2+(r+c/128)%3)/8.f);
    for(int r=0;r<E*B;++r) for(int b=0;b<48;++b) vb[r*48+b]=__float2half_rn(float(2+(r+b)%3)/8.f);
    K1Args a{}; a.M=M; a.n_slots=nslots; a.n_plans=plans; a.B=B; a.sh_e=2; a.out_bf16=bf; a.ysh=1;
    a.ctl=mem.alloc<int>(3); a.slot_expert=mem.alloc<int>(experts.size()); put(const_cast<int*>(a.slot_expert),experts);
    a.tok_slot=mem.alloc<int>(slots.size()); put(const_cast<int*>(a.tok_slot),slots);
    a.tok_col=mem.alloc<int>(cols.size()); put(const_cast<int*>(a.tok_col),cols);
    a.tok_w=mem.alloc<float>(weights.size()); put(const_cast<float*>(a.tok_w),weights);
    a.t_tab=mem.alloc<long long>(tab.size()); put(const_cast<long long*>(a.t_tab),tab);
    a.t_sg=mem.alloc<__half>(sg.size()); put(const_cast<__half*>(a.t_sg),sg);
    a.t_v3=mem.alloc<__half>(v3.size()); put(const_cast<__half*>(a.t_v3),v3);
    a.t_v3b=mem.alloc<__half>(vb.size()); put(const_cast<__half*>(a.t_v3b),vb);
    a.z=mem.alloc<__half>(size_t(nslots+nsh)*B*8*HID);
    a.ypart=mem.alloc<__half>(size_t(M+1)*HID); a.out=mem.alloc<__half>(guarded);
    auto d=direction(); auto* dd=mem.alloc<float>(HID); put(dd,d); auto* ds=mem.alloc<float>(M);
    auto* parts=mem.alloc<float>(size_t(M+1)*48); auto* counts=mem.alloc<unsigned>(M+1);
    std::vector<float> ps(size_t(M+1)*48,-123.f); put(parts,ps);
    std::vector<unsigned> ct(M+1,0); ct[M]=0xabcdefu; put(counts,ct);
    unsigned short* target=reinterpret_cast<unsigned short*>(a.out);
#ifdef DEC_SLOT
    const size_t stride=guarded*2;
    auto* send=mem.alloc<char>(4*stride); CK(cudaMemset(send,0x5a,4*stride));
    auto* bell=mem.alloc<unsigned long long>(2); auto* ticket=mem.alloc<unsigned>(1); auto* step=mem.alloc<u64>(1);
    auto* snapshot=mem.alloc<unsigned short>(N); auto* timeout=mem.alloc<int>(1);
    cudaStream_t producer, watcher; CK(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&watcher,cudaStreamNonBlocking));
    if(slot) { a.slot={send,bell,step,stride,2,1,ticket,nullptr}; }
#else
    (void)slot;
    cudaStream_t producer; CK(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking));
#endif
    for(int rep=0;rep<repeats;++rep) {
        const int plan=rep%2;
        std::vector<int> ctl{0,0,plan}; put(a.ctl,ctl);
        std::vector<__half> z(size_t(nslots+nsh)*B*8*HID), ys(size_t(M+1)*HID);
        for(int s=0;s<nslots+nsh;++s) for(int b=0;b<B;++b) for(int m=0;m<8;++m) for(int c=0;c<HID;++c) {
            const float v=c%128==0 ? (c/128%2 ? -1.f:1.f)*(1.f+float((s+b+m+rep)%5)/8.f) : float((c*7+s*3+b+m+rep)%13-6)/512.f;
            z[((size_t(s)*B+b)*8+m)*HID+c]=__float2half_rn(v);
        }
        for(int m=0;m<M;++m) for(int c=0;c<HID;++c) ys[size_t(m+1)*HID+c]=__float2half_rn(float((c*3+m+rep)%17-8)/64.f + d[c]*2.f);
        put(a.z,z); put(a.ypart,ys); CK(cudaMemsetAsync(a.out,0x5a,guarded*2,producer));
        std::vector<float> scales(M,0.f);
        for(int m=0;m<M;++m) scales[m]=mode==1 ? .75f:mode==2 && m%2 ? 1.25f:0.f;
        if(mode==3 || mode==4 || mode==5) std::fill(scales.begin(),scales.end(),1.f);
        put(ds,scales);
        const float alpha=mode==5 ? 0.f:.625f;
#ifndef G6_STEERING_TEST_BASELINE
        a.steer_dir=mode==3 || mode==5 ? reinterpret_cast<const float*>(uintptr_t(1)):mode==4 ? nullptr:dd;
        a.steer_scale=mode==6 ? nullptr:ds; a.steer_alpha=alpha; a.steer_part=parts; a.steer_count=counts;
        if(mode==3) { std::fill(scales.begin(),scales.end(),0.f); put(ds,scales); }
#else
        std::fill(scales.begin(),scales.end(),0.f);
#endif
        K1Args reference=a;
#ifndef G6_STEERING_TEST_BASELINE
        reference.steer_dir=nullptr;
#endif
#ifdef DEC_SLOT
        reference.slot.send=nullptr;
#endif
        if(sh) k1_tr3_fin<true><<<M*48,160,0,producer>>>(reference); else k1_tr3_fin<false><<<M*48,128,0,producer>>>(reference);
        CK(cudaGetLastError()); CK(cudaStreamSynchronize(producer));
        const auto stock_output=get(reinterpret_cast<unsigned short*>(a.out),N);
        put(a.ctl,ctl); CK(cudaMemsetAsync(a.out,0x5a,guarded*2,producer));
#ifdef DEC_SLOT
        if(slot) {
            const u64 value=rep+1; CK(cudaMemcpy(step,&value,sizeof value,cudaMemcpyHostToDevice));
            target=reinterpret_cast<unsigned short*>(send+((value&1)*2+1)*stride);
            observe<<<1,256,0,watcher>>>(bell+1,value*2+2,target,snapshot,N,timeout);
        }
#endif
        if(sh) k1_tr3_fin<true><<<M*48,160,0,producer>>>(a); else k1_tr3_fin<false><<<M*48,128,0,producer>>>(a);
        CK(cudaGetLastError()); CK(cudaStreamSynchronize(producer));
#ifdef DEC_SLOT
        if(slot) {
            CK(cudaStreamSynchronize(watcher));
            check(get(timeout,1)[0]==0,"concurrent observer reaches actual decode doorbell");
            check(get(snapshot,N)==get(target,N),"doorbell observer sees every correction store");
            check(get(ticket,1)[0]==0 && get(bell,2)[1]==u64(rep+1)*2+2,"decode publication sequence and ticket reset");
        }
#endif
        auto out=get(target,guarded);
        std::vector<float> want(N), shared(HID), routed(HID);
        bool off_exact=true;
        for(int m=0;m<M;++m) {
            std::fill(routed.begin(),routed.end(),0.f); std::fill(shared.begin(),shared.end(),0.f);
            for(int group=0;group<5;++group) {
                std::vector<float> sum(HID,0.f);
                if(group==4 && !sh) for(int c=0;c<HID;++c) sum[c]=__half2float(ys[size_t(m+1)*HID+c]);
                else for(int u=0;u<(group==4 ? 1:2);++u) {
                    const int t=(plan*G6_MX+m)*8+group*2+u;
                    const int sl=group==4 ? nslots+m/8:slots[t];
                    if(sl<0) continue;
                    const int e=group==4 ? 2:experts[plan*SMAX+sl], col=group==4 ? m%8:cols[t];
                    const float w=group==4 ? 1.f:weights[t];
                    for(int b=0;b<tab[e*6+3];++b) for(int h=0;h<48;++h) {
                        float v[128]; for(int j=0;j<128;++j) v[j]=__half2float(z[((size_t(sl)*B+b)*8+col)*HID+h*128+j]); had(v);
                        for(int j=0;j<128;++j) {
                            const int c=h*128+j,r=e*B+b;
                            const float sv=sh ? rounded_half(__half2float(sg[HID+c])*__half2float(vb[r*48+h])):__half2float(v3[size_t(r)*HID+c]);
                            sum[c]+=w*(v[j]*sv);
                        }
                    }
                }
                for(int c=0;c<HID;++c) { if(group==4) shared[c]=sum[c]; else routed[c]+=sum[c]; }
            }
            double dot=0; for(int c=0;c<HID;++c) dot+=double(shared[c])*d[c];
            const bool selected=mode!=3 && mode!=4 && mode!=6 && scales[m]!=0.f && alpha!=0.f;
            if(!selected) off_exact &= std::memcmp(out.data()+size_t(m)*HID,stock_output.data()+size_t(m)*HID,HID*2)==0;
            for(int c=0;c<HID;++c) {
                const float stock=decode(encode(routed[c]+shared[c],bf),bf);
                want[size_t(m)*HID+c]=stock-(selected ? float(scales[m]*alpha*dot)*d[c]:0.f);
            }
        }
        float error=0; check(close(out,want,bf,error),"decode projects only shared contribution; host transform/projection oracle");
        check(off_exact,"decode zero/null/alpha-zero rows exactly match stock GPU bytes");
        bool guard=true; for(size_t i=N;i<guarded;++i) guard &= out[i]==0x5a5a;
        check(guard,"decode output guard row untouched");
        const auto got_count=get(counts,M+1); const auto got_parts=get(parts,ps.size());
        bool reset=got_count[M]==ct[M], scratch_guard=true;
        for(int m=0;m<M;++m) { reset &= got_count[m]==0; const bool active=mode!=3 && mode!=4 && mode!=6 && scales[m]!=0 && alpha!=0;
            if(!active) for(int b=0;b<48;++b) scratch_guard &= got_parts[m*48+b]==ps[m*48+b];
        }
        for(int b=0;b<48;++b) scratch_guard &= got_parts[M*48+b]==ps[M*48+b];
        check(reset && scratch_guard,"decode counters reset; disabled/guard scratch untouched");
        check(get(a.ctl,3)[0]==(plan+1)%plans,"decode plan rotation preserved");
        if(slot) { const auto stage=get(reinterpret_cast<unsigned short*>(a.out),guarded); check(std::all_of(stage.begin(),stage.end(),[](auto x){return x==0x5a5a;}),"slot output bypasses staging"); }
        if(dump) { append(dump,out.data(),out.size()*2); const auto c=get(a.ctl,3); append(dump,c.data(),c.size()*4); }
        std::printf("decode SH=%d bf16=%d M=%d slot=%d mode=%d rep=%d error=%g\n",sh,bf,M,slot,mode,rep,error);
        // Keep the actual previous scratch on selected rows; inactive rows must never mutate it.
        ps=got_parts;
    }
    CK(cudaStreamDestroy(producer));
#ifdef DEC_SLOT
    CK(cudaStreamDestroy(watcher));
#endif
}

static void prefill_case(bool shx,bool rs,bool block_scale,int M,int mode,FILE* dump) {
    Arena mem;
    const size_t N=size_t(M)*HID, guarded=N+HID;
    bindp::Expert a{}; a.M=M; a.on=1;
    std::vector<int> pos(M*8); std::vector<__half> yw(size_t(M)*8*HID), ysh(N), sv(HID), sc(48);
    for(int b=0;b<48;++b) sc[b]=__float2half_rn(float(1+b%3)/2.f);
    for(int c=0;c<HID;++c) sv[c]=__float2half_rn(c%3==0 ? -1.f:1.f);
    for(size_t i=0;i<yw.size();++i) yw[i]=__float2half_rn(float(int(i%29)-14)/128.f);
    for(size_t i=0;i<N;++i) ysh[i]=__float2half_rn(float(int(i%31)-15)/64.f);
    for(int m=0;m<M;++m) for(int k=0;k<8;++k) {
        pos[m*8+k]=m*8+k;
#ifdef PF_FINLOCAL
        if((m+k)%4==0) pos[m*8+k]=-1;
#endif
    }
    a.pos=mem.alloc<int>(pos.size()); put(a.pos,pos);
    a.yw=mem.alloc<__half>(yw.size()); put(a.yw,yw);
    a.ysh=mem.alloc<__half>(ysh.size()); put(const_cast<__half*>(a.ysh),ysh);
    auto* dsv=mem.alloc<__half>(HID); put(dsv,sv); a.sh_svh=shx ? dsv:nullptr;
    auto* dsc=mem.alloc<__half>(48); put(dsc,sc); a.sh_sc=block_scale ? dsc:nullptr;
    a.out=mem.alloc<uint16_t>(guarded); CK(cudaMemset(a.out,0x5a,guarded*2));
    const auto d=direction(); auto* dd=mem.alloc<float>(HID); put(dd,d);
    float scale=mode==1 ? 1.f:mode==2 ? .375f:0.f,alpha=.75f;
#ifndef G6_STEERING_TEST_BASELINE
    a.steer_dir=mode==3 || mode==5 ? reinterpret_cast<const float*>(uintptr_t(1)):mode==4 ? nullptr:dd;
    if(mode==4 || mode==5) scale=1.f;
    if(mode==5) alpha=0.f;
    a.steer_scale=scale; a.steer_alpha=alpha;
#endif
    const size_t ss=size_t((M+g6nr::NR-1)/g6nr::NR+1)*HID*2, stride=g6nr::NR*ss;
    auto* send=mem.alloc<char>(2*stride); CK(cudaMemset(send,0x5a,2*stride));
    auto* step=mem.alloc<u64>(1); const u64 one=1; CK(cudaMemcpy(step,&one,sizeof one,cudaMemcpyHostToDevice));
    if(rs) a.rs={send,stride,ss,step,M,g6nr::NR};
    bindp::Expert reference=a; reference.rs.base=nullptr;
#ifndef G6_STEERING_TEST_BASELINE
    reference.steer_dir=nullptr;
#endif
    bindp::launch_fin(reference,nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const auto stock_output=get(a.out,N); CK(cudaMemset(a.out,0x5a,guarded*2));
    bindp::launch_fin(a,nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const auto staging=get(a.out,guarded); const auto wire=get(send,2*stride);
    std::vector<unsigned short> out(guarded,0x5a5a); std::vector<char> expected_wire(2*stride,char(0x5a));
    for(int m=0;m<M;++m) {
        const int p=((m+1)*g6nr::NR-1)/M;
        const size_t offset=stride+p*ss+size_t(m-p*M/g6nr::NR)*HID*2;
        if(rs) std::memcpy(out.data()+size_t(m)*HID,wire.data()+offset,HID*2); else std::copy_n(staging.data()+size_t(m)*HID,HID,out.data()+size_t(m)*HID);
        std::memcpy(expected_wire.data()+offset,out.data()+size_t(m)*HID,HID*2);
    }
    std::vector<float> want(N), shared(N);
    for(int m=0;m<M;++m) {
        for(int b=0;b<48;++b) {
            float v[128]; for(int j=0;j<128;++j) v[j]=__half2float(ysh[size_t(m)*HID+b*128+j]); if(shx) had(v);
            for(int j=0;j<128;++j) { const int c=b*128+j; shared[size_t(m)*HID+c]=shx ? rounded_half(v[j]*__half2float(sv[c])*(block_scale ? __half2float(sc[b]):1.f)):v[j]; }
        }
        double dot=0; for(int c=0;c<HID;++c) dot+=double(shared[size_t(m)*HID+c])*d[c];
        for(int c=0;c<HID;++c) { float routed=0; for(int k=0;k<8;++k) if(pos[m*8+k]>=0) routed+=__half2float(yw[size_t(pos[m*8+k])*HID+c]);
            want[size_t(m)*HID+c]=routed+shared[size_t(m)*HID+c]-(mode!=4 ? float(scale*alpha*dot)*d[c]:0.f);
        }
    }
    float error=0; check(close(out,want,true,error),"prefill direct/SHX shared projection agrees with scalar oracle");
    check((mode==1 || mode==2) || std::equal(stock_output.begin(),stock_output.end(),out.begin()),"prefill zero/null/alpha-zero output exactly matches stock GPU bytes");
    check(!rs || wire==expected_wire,"prefill RS ownership, parity and guard bytes");
    check(rs ? std::all_of(staging.begin(),staging.end(),[](auto x){return x==0x5a5a;}):std::all_of(staging.begin()+N,staging.end(),[](auto x){return x==0x5a5a;}),"prefill staging/guard preservation");
    if(dump) { append(dump,staging.data(),staging.size()*2); append(dump,wire.data(),wire.size()); }
    std::printf("prefill SHX=%d RS=%d scales=%d M=%d mode=%d error=%g\n",shx,rs,block_scale,M,mode,error);
}

int main(int argc,char** argv) {
    FILE* dump=nullptr;
    if(argc==3 && !std::strcmp(argv[1],"--stock-dump")) { dump=std::fopen(argv[2],"wb"); if(!dump) return 2; }
    else if(argc!=1) return 2;
    for(int M:{1,8,9,G6_MX}) for(bool sh:{false,true}) for(bool bf:{false,true}) for(bool slot:{false,true}) {
#ifndef DEC_SLOT
        if(slot) continue;
#endif
        for(int mode=0;mode<(dump ? 1:7);++mode) decode_case(sh,bf,M,slot,mode,dump ? 2:mode==1 || mode==2 ? 5:1,dump);
    }
    for(int M:{1,3,8,17,32}) for(bool sh:{false,true}) for(bool rs:{false,true}) for(bool bs:{false,true}) {
        if(!sh && bs) continue;
        for(int mode=0;mode<(dump ? 1:6);++mode) prefill_case(sh,rs,bs,M,mode,dump);
    }
    if(dump && std::fclose(dump)) return 2;
    std::printf("steering_shared_test: %d checks, %d failures\n",checks,failures);
    return failures ? 1:0;
}
