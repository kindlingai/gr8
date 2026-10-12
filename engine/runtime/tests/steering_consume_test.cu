// Direct production consume kernels on one GPU, synthetic fabric slots, no model or network.
// Compile with G6_STEERING_TEST_BASELINE against the parent tree for --stock-dump byte comparisons.
#include "../prefill.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__); std::exit(2); } } while(0)
using namespace g6rt;
static int checks=0, failures=0;
static void check(bool b,const char* what) { ++checks; if(!b) { ++failures; std::printf("FAIL %s\n",what); } }
struct Arena {
    std::vector<void*> ptr;
    template<class T> T* alloc(size_t n) { T* p; CK(cudaMalloc(&p,n*sizeof(T))); ptr.push_back(p); CK(cudaMemset(p,0,n*sizeof(T))); return p; }
    ~Arena() { for(void* p:ptr) cudaFree(p); }
};
template<class T> static void put(T* dst,const std::vector<T>& v) { CK(cudaMemcpy(dst,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> static std::vector<T> get(const T* src,size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(),src,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
static float partial(int p,int m,int c) { return float((c*13+m*7+p*3)%97-48)/64.f; }
static void append(FILE* f,const void* p,size_t n) { if(f && std::fwrite(p,1,n,f)!=n) std::exit(2); }
// Each case covers either AR, RS cluster, RS wide, owner, or pair-owner (lo/hi in both orders).
static void run(int kind,int M,int nr,int rank,int mode,bool loop,FILE* dump) {
    Arena a;
    const int rows=kind==0 ? M : pf_rows(M,rank,nr), first=kind==0 ? 0 : pf_r0(M,rank,nr);
    const bool pair=kind>=4, owner=kind==3;
    const size_t N=size_t(M+1)*HID;
    const size_t slice=pf_ss_min(M,nr), stride=std::max(size_t(M)*HID*2+256,size_t(nr)*slice);
    g6::FabricDev f{}; f.nr=nr; f.rank=rank; f.ncoll=2; f.stride=stride;
    std::vector<char> send(4*stride,0), recv(4*nr*stride,0);
    f.send=a.alloc<char>(send.size()); f.recv=a.alloc<char>(recv.size());
    f.doorbell=a.alloc<unsigned long long>(2); f.error=a.alloc<unsigned>(1);
    auto* ticket=a.alloc<unsigned>(1); auto* step=a.alloc<u64>(1); const u64 one=1;
    CK(cudaMemcpy(step,&one,sizeof one,cudaMemcpyHostToDevice));
    std::vector<float> residual(N), writer(size_t(std::max(rows,1))*HID), direction(HID), scales(M,0);
    std::vector<__nv_bfloat16> weight(HID), mine(size_t(std::max(rows,1))*HID);
    for(size_t i=0;i<N;++i) residual[i]=float(int(i%53)-26)/128.f;
    const auto original=residual;
    const float alpha=.875f;
    for(int c=0;c<HID;++c) { direction[c]=float((c%7)-3)/std::sqrt(4.f*HID); weight[c]=__float2bfloat16(float(3+c%5)/4.f); }
    for(int m=0;m<M;++m) scales[m]=mode==0 ? 0.f : mode==1 ? 1.125f : (m%2 ? .625f : 0.f);
    if(mode==4) std::fill(scales.begin(),scales.end(),.75f);
    if(mode==3 || mode==5 || mode==6) std::fill(scales.begin(),scales.end(),0.f); // invalid direction pointer must never be dereferenced
    for(int j=0;j<rows;++j) for(int c=0;c<HID;++c) {
        const int m=first+j;
        writer[size_t(j)*HID+c]=float((c*5+m*11)%67-33)/128.f;
        mine[size_t(j)*HID+c]=__float2bfloat16(partial(0,m,c));
    }
    for(int p=0;p<nr;++p) for(int j=0;j<rows;++j) for(int c=0;c<HID;++c) {
        const int m=first+j;
        const auto value=__float2bfloat16(partial(pair ? 1 : p,m,c));
        const size_t off=(kind==0 ? 0 : (p==rank || loop ? rank*slice : 0))+size_t(j)*HID*2+c*2;
        char* dst=p==rank ? send.data()+2*stride+off : recv.data()+(2*nr+p)*stride+off;
        std::memcpy(dst,&value,sizeof value);
    }
    put(f.send,send); put(f.recv,recv);
    auto* dr=a.alloc<float>(N); put(dr,residual);
    auto* dw=a.alloc<__nv_bfloat16>(HID); put(dw,weight);
    auto* dd=a.alloc<float>(HID); put(dd,direction);
    auto* ds=a.alloc<float>(M); put(ds,scales);
    auto* dy=a.alloc<float>(writer.size()); put(dy,writer);
    auto* dm=a.alloc<__nv_bfloat16>(mine.size()); put(dm,mine);
    auto* out=a.alloc<__half>(N); CK(cudaMemset(out,0x5a,N*sizeof(__half)));
#ifdef DEC_SLOT
    const size_t frag_words=size_t((M+7)/8)*KSH*16*4;
    auto* xf=a.alloc<u32>(frag_words); auto* xb=a.alloc<u32>(frag_words);
#endif
    if(kind==0) {
        glue::ArNormArgs args{f,step,0,M,dr,dw,out,1e-5f};
#ifndef G6_STEERING_TEST_BASELINE
        args.steer={mode==6 ? nullptr : mode==3 || mode==5 ? reinterpret_cast<const float*>(uintptr_t(1)) : dd,
                    mode==4 || mode==5 ? nullptr : ds, mode==4 ? .75f : mode==5 ? 1.f : 0.f, mode==5 ? 0.f : alpha};
#endif
#ifdef DEC_SLOT
        args.xf16=xf; args.xbf16=xb;
#endif
        glue::k_ar_add_norm<<<dim3(glue::RN_CL,M),glue::RN_T>>>(args);
    } else {
        RsNormArgs args{f,step,0,M,loop,dr,dw,out,1e-5f,ticket};
#ifndef G6_STEERING_TEST_BASELINE
        args.steer={mode==6 ? nullptr : mode==3 || mode==5 ? reinterpret_cast<const float*>(uintptr_t(1)) : dd,
                    mode==4 || mode==5 ? nullptr : ds, mode==4 ? .75f : mode==5 ? 1.f : 0.f, mode==5 ? 0.f : alpha};
#endif
        if(kind==1) k_rs_add_norm<<<dim3(glue::RN_CL,std::max(rows,1)),glue::RN_T>>>(args);
#ifdef PF_RNWIDE
        if(kind==2) k_rs_add_norm_w<<<std::max(rows,1),glue::RN_CL*glue::RN_T>>>(args);
#endif
        if(owner) k_own_add_norm<<<dim3(glue::RN_CL,std::max(rows,1)),glue::RN_T>>>(args,dy);
#if G6_NR == 4
        if(pair) k_pair_add_norm<<<dim3(glue::RN_CL,std::max(rows,1)),glue::RN_T>>>(args,dm,rank^2,kind==4);
#endif
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    residual=get(dr,N); const auto output=get(out,N); const auto ag=get(f.send,send.size());
    float er=0,en=0; bool finite=true, untouched=true, exact_zero=true, ag_same=true;
    std::vector<float> y(HID);
    for(int m=0;m<M+1;++m) {
        const bool live=m>=first && m<first+rows;
        for(int c=0;c<HID;++c) {
            const size_t i=size_t(m)*HID+c;
            if(!live) {
                untouched &= std::memcmp(&residual[i],&original[i],sizeof(float))==0;
                unsigned short bits; std::memcpy(&bits,&output[i],2); untouched &= bits==0x5a5a;
            }
        }
        if(!live) continue;
        for(int c=0;c<HID;++c) {
            float v=0.f;
            if(owner) v=writer[size_t(m-first)*HID+c];
            else if(pair) { v+=partial(kind==4 ? 0 : 1,m,c); v+=partial(kind==4 ? 1 : 0,m,c); }
            else for(int p=0;p<nr;++p) v+=partial(p,m,c);
            y[c]=v;
        }
        const auto stock=y;
        g6rt::steering::project(y.data(),direction.data(),HID,scales[m]*alpha);
        double ss=0;
        for(int c=0;c<HID;++c) { y[c]+=original[size_t(m)*HID+c]; ss+=double(y[c])*y[c]; }
        const double inv=1/std::sqrt(ss/HID+1e-5);
        for(int c=0;c<HID;++c) {
            const size_t i=size_t(m)*HID+c;
            const float expected=float(double(y[c])*inv*__bfloat162float(weight[c]));
            finite &= std::isfinite(residual[i]) && std::isfinite(__half2float(output[i]));
            er=std::max(er,std::fabs(residual[i]-y[c]));
            en=std::max(en,std::fabs(__half2float(output[i])-expected));
            if(scales[m]==0.f) { const float v=stock[c]+original[i]; exact_zero &= std::memcmp(&v,&residual[i],4)==0; }
            if(kind!=0) {
                const size_t off=3*stride+size_t(m-first)*HID*2+c*2;
                ag_same &= std::memcmp(ag.data()+off,&output[i],2)==0;
            }
        }
    }
    const auto bell=get(f.doorbell,2); const auto tickets=get(ticket,1);
    check(finite && er<4e-5f && en<3e-3f,"consume agrees with host projection/residual/RMS oracle");
    check(untouched && exact_zero,"unowned/guard rows untouched; zero-scale residual exact");
    check(ag_same && tickets[0]==0 && bell[1]==(kind==0 ? 0 : 4),"AG bytes and publish/reset state");
#ifdef DEC_SLOT
    if(dump && kind==0 && mode==0) { const auto hxf=get(xf,frag_words), hxb=get(xb,frag_words); append(dump,hxf.data(),frag_words*4); append(dump,hxb.data(),frag_words*4); }
#endif
    if(dump && mode==0) { append(dump,residual.data(),N*4); append(dump,output.data(),N*2); append(dump,ag.data(),ag.size()); append(dump,bell.data(),bell.size()*8); }
    std::printf("consume kind=%d M=%d nr=%d rank=%d mode=%d loop=%d residual=%g norm=%g\n",kind,M,nr,rank,mode,int(loop),er,en);
}
#ifndef G6_STEERING_TEST_BASELINE
static void scope_selection() {
    Model m; Prefill p;
    check(pf_steer(m,p,0,true).dir==nullptr,"empty model has no directions");
    m.layers.resize(2); m.layers[0].moe=false; m.layers[1].moe=true;
    m.steering_dir.assign(2,reinterpret_cast<const float*>(uintptr_t(1)));
    m.steering_alpha.assign(2,.5f); p.steering_scale=.75f;
    m.steering_scope=steering::Scope::Shared;
    check(pf_steer(m,p,1,true).dir && !pf_steer(m,p,1,false).dir,"shared MoE excludes full MLP consume");
    check(pf_steer(m,p,0,false).dir && !pf_steer(m,p,2,true).dir,"dense MLP included; absent layer excluded");
    m.steering_scope=steering::Scope::Full;
    check(pf_steer(m,p,1,false).dir && pf_steer(m,p,1,false).scale==.75f,"full scope includes MLP with lane scale");
}
#endif
int main(int argc,char** argv) {
#ifndef G6_STEERING_TEST_BASELINE
    scope_selection();
#endif
    FILE* dump=nullptr;
    if(argc==3 && std::string(argv[1])=="--stock-dump") { dump=std::fopen(argv[2],"wb"); if(!dump)return 2; }
#ifdef G6_STEERING_TEST_BASELINE
    if(!dump) { std::fprintf(stderr,"baseline requires --stock-dump PATH\n"); return 2; }
#endif
    for(int nr : {3,4,6}) for(int M : {1,3,8,32}) for(int rank : {0,nr-1}) {
        for(int kind=0;kind<6;++kind) {
            if(kind>=4 && nr!=4)continue;
#ifndef PF_RNWIDE
            if(kind==2)continue;
#endif
#if G6_NR != 4
            if(kind>=4)continue;
#endif
            const int modes=dump ? 1 : 7;
            for(int mode=0;mode<modes;++mode) run(kind,M,nr,rank,mode,false,dump);
            if(kind==1 || kind>=4) run(kind,M,nr,rank,0,true,dump);
        }
    }
    if(dump && std::fclose(dump))return 2;
    std::printf("steering_consume_test: %d checks, %d FAIL\n",checks,failures); return failures ? 1 : 0;
}
