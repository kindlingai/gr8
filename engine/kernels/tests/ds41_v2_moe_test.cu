// Local synthetic adapter/grouped-down gate, no model or fabric, <100 MiB.
// nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -o /tmp/ds41_v2_moe_test engine/kernels/tests/ds41_v2_moe_test.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include "../ds41_v2_moe.cuh"
#include "../ds41_v2_fc2.cuh"
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));exit(2);} } while(0)
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
template<class T> struct Dev {
    T* p; size_t n;
    explicit Dev(size_t n_):n(n_){CK(cudaMalloc(&p,n*sizeof(T)));}
    ~Dev(){cudaFree(p);}
    void put(const T* v){CK(cudaMemcpy(p,v,n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
constexpr int TK=6, NE=32, MM=128, H=5120, I=576;
using Plan=dsr::PlanMT<TK,MM>;
void run(int M, int cut)
{
    const int padded=(M+3)/4*4;
    auto ref=std::make_unique<Plan>();
    std::vector<int> idx(M*TK), ex(M*8,-1);
    std::vector<float> wt(M*TK), ew(M*8,0);
    for(int m=0;m<M;++m) for(int k=0;k<TK;++k){
        int e=k<cut?(k*3+(m%11==0))%NE:-1; // hot slots exercise >64-row loops
        ex[m*8+k]=idx[m*TK+k]=e;
        ew[m*8+k]=wt[m*TK+k]=k<cut?1.5f/float(cut):0.f;
    }
    dsr::plan_ref<TK,MM>(M,reinterpret_cast<const int(*)[8]>(ex.data()),reinterpret_cast<const float(*)[8]>(ew.data()),*ref,cut);
    Dev<Plan> p(1); Dev<int> di(idx.size()), t0(NE); Dev<float> dw(wt.size()); Dev<unsigned> masks(NE*(MM/32));
    di.put(idx.data());dw.put(wt.data()); CK(cudaMemset(p.p,0xa5,sizeof(Plan))); CK(cudaMemset(masks.p,0,masks.n*4));
    ds41v2moe::k_import_routes<NE,TK,MM><<<(M*8+255)/256,256>>>(di.p,dw.p,M,p.p,masks.p);
    dsr::k_scan_ro<NE,TK,MM><<<1,NE>>>(p.p,masks.p,t0.p);
    dsr::k_place_ro<NE,TK,MM,TK,true><<<(M*TK+255)/256,256>>>(p.p,masks.p,t0.p,M);
    CK(cudaGetLastError()); auto got=p.get()[0];
    CHECK(got.n_slots==ref->n_slots && got.n_tiles==ref->n_tiles);
    for(int s=0;s<got.n_slots;++s) CHECK(got.slot_expert[s]==ref->slot_expert[s] && got.slot_tile0[s]==ref->slot_tile0[s] && got.slot_nt[s]==ref->slot_nt[s]);
    for(int t=0;t<got.n_tiles;++t) for(int c=0;c<8;++c) CHECK(got.tile_tok[t][c]==ref->tile_tok[t][c] && got.tile_w[t][c]==ref->tile_w[t][c]);
    for(int m=0;m<M;++m) for(int k=0;k<8;++k) CHECK(got.tok_slot[m][k]==ref->tok_slot[m][k] && got.tok_col[m][k]==ref->tok_col[m][k]);
    // Export including padding and a rank with no live rows.
    Dev<int> oi(padded*TK);Dev<float> ow(padded*TK);
    ds41v2moe::k_export_routes<TK,MM><<<(padded*TK+255)/256,256>>>(p.p,M,padded,oi.p,ow.p);
    auto hi=oi.get();auto hw=ow.get();
    for(int j=0;j<padded*TK;++j) CHECK(hi[j]==(j<M*TK?idx[j]:-1) && hw[j]==(j<M*TK?wt[j]:0.f));
    ds41v2moe::k_export_routes<TK,MM><<<(padded*TK+255)/256,256>>>(p.p,0,padded,oi.p,ow.p);
    hi=oi.get();hw=ow.get();for(int j=0;j<padded*TK;++j) CHECK(hi[j]==-1 && hw[j]==0.f);
    // Roundtrip plain/record quantized bytes: includes every record byte position.
    Dev<uint8_t> plain(M*I),record(M*I),back(M*I);
    std::vector<uint8_t> hp(plain.n);for(size_t j=0;j<hp.size();++j)hp[j]=(j*17+j/47)%256;plain.put(hp.data());
    ds41v2moe::k_relayout<true><<<(plain.n+255)/256,256>>>(plain.p,record.p,M,I);
    ds41v2moe::k_relayout<false><<<(plain.n+255)/256,256>>>(record.p,back.p,M,I);
    CHECK(back.get()==hp);
    const int nr=got.n_tiles*8;
    Dev<uint8_t> weights(NE*(H/16)*(I/64)*mx4::GR),aq(nr*I),as(nr*mxtc::ASP);
    std::vector<uint8_t> w(weights.n),a(aq.n),sc(as.n,121);
    for(size_t j=0;j<w.size();++j)w[j]=j%mx4::GR<512?uint8_t(0x12+(j*13)%0x55):uint8_t(118+j%3);
    for(size_t j=0;j<a.size();++j)a[j]=uint8_t((j*7+j/13)%0x78); // finite nonnegative fp8
    weights.put(w.data());aq.put(a.data());as.put(sc.data());
    Dev<__half> full(nr*H),grouped(nr*H);
    ds41v2moe::Args args{};args.fc2=weights.p;args.slot_expert=p.p->slot_expert;args.slot_tile0=p.p->slot_tile0;args.slot_nt=p.p->slot_nt;
    args.tile_mask=p.p->tile_mask;args.tile_w=&p.p->tile_w[0][0];args.tile_tok=&p.p->tile_tok[0][0];args.n_slots_dev=&p.p->n_slots;args.n_slots=got.n_slots;
    args.gy2=4;args.os2=16;args.aq=aq.p;args.as=as.p;args.ypart=full.p;
    constexpr size_t SM=mxtc::fc2_smem<4,8,8>();
    auto kfull=mxtc::k_fc2_tc<I,H,4,2,8,8>;auto kgroup=mxtc::k_fc2_tc<I,H,4,2,8,8,4,ds41v2moe::Args>;
    CK(cudaFuncSetAttribute(kfull,cudaFuncAttributeMaxDynamicSharedMemorySize,SM));
    CK(cudaFuncSetAttribute(kgroup,cudaFuncAttributeMaxDynamicSharedMemorySize,SM));
    kfull<<<args.n_slots*args.gy2*args.os2,128,SM>>>(args);CK(cudaGetLastError());
    auto f=full.get();
    const __half sentinel=__float2half(-123.f);std::vector<__half> init(grouped.n,sentinel);grouped.put(init.data());
    args.ypart=grouped.p;args.os2=5;
    Dev<__nv_bfloat16> shared(padded*1280),output(padded*1280);
    std::vector<__nv_bfloat16> sh(shared.n);for(size_t j=0;j<sh.size();++j)sh[j]=__float2bfloat16(float(int(j%17)-8)*0.01f);shared.put(sh.data());
    double maxdiff=0;
    for(int c=0;c<4;++c){
        args.col_group=c;
        kgroup<<<args.n_slots*args.gy2*args.os2,128,SM>>>(args);CK(cudaGetLastError());
        auto g=grouped.get();
        for(int t=0;t<got.n_tiles;++t)for(int r=0;r<8;++r)if(got.tile_mask[t]>>r&1)for(int n=0;n<H;++n){
            size_t j=(t*8+r)*H+n;
            if(n<(c+1)*1280){double d=fabs(__half2float(g[j])-__half2float(f[j]));maxdiff=std::max(maxdiff,d);CHECK(std::isfinite(d)&&d<=0.002*(1+fabs(__half2float(f[j]))));}
            else CHECK(__half2float(g[j])==-123.f);
        }
        ds41v2moe::k_combine_group<H,TK,MM,__nv_bfloat16><<<(padded*1280+255)/256,256>>>(p.p,grouped.p,shared.p,output.p,M,padded,c*1280,1280);
        auto out=output.get();
        for(int m=0;m<padded;++m)for(int n=0;n<1280;++n){float sum=0;
            if(m<M){for(int k=0;k<cut;++k)sum+=__half2float(f[(got.tok_slot[m][k]*8+got.tok_col[m][k])*H+c*1280+n]);sum+=__bfloat162float(sh[m*1280+n]);}
            float expect=__bfloat162float(__float2bfloat16(sum)),v=__bfloat162float(out[m*1280+n]);CHECK(std::isfinite(v)&&fabs(v-expect)<=0.008*(1+fabs(expect)));
        }
    }
    // Resident-weight variant, heavy-first ordering. Check against stock before
    // any timing claim. All four column launches reuse the same bucket data.
    Dev<int> order(NE);
    ds41v2moe::k_order_desc<NE,TK,MM><<<1,NE>>>(p.p,order.p);
    const auto ho=order.get();
    for(int j=1;j<got.n_slots;++j)CHECK(got.slot_nt[ho[j-1]]>=got.slot_nt[ho[j]]);
    constexpr size_t RS=ds41v2moe::resident_smem<4,8,I>();
    auto kr=ds41v2moe::k_fc2_resident<I,H>;
    CK(cudaFuncSetAttribute(kr,cudaFuncAttributeMaxDynamicSharedMemorySize,RS));
    grouped.put(init.data());
    for(int c=0;c<4;++c){args.col_group=c;kr<<<args.n_slots*10,128,RS>>>(args,order.p);CK(cudaGetLastError());}
    auto resident=grouped.get();double rmax=0;
    for(int t=0;t<got.n_tiles;++t)for(int r=0;r<8;++r)if(got.tile_mask[t]>>r&1)for(int n=0;n<H;++n){size_t j=(t*8+r)*H+n;
        double d=fabs(__half2float(resident[j])-__half2float(f[j]));rmax=std::max(rmax,d);CHECK(std::isfinite(d)&&d<=.002*(1+fabs(__half2float(f[j]))));}
    printf("PASS resident M=%d cut=%d max_abs=%g\n",M,cut,rmax);
    printf("PASS M=%d cut=%d slots=%d tiles=%d grouped_max_abs=%g plan/export/padding/relayout/combine\n",M,cut,got.n_slots,got.n_tiles,maxdiff);
}
void large_plan() {
    constexpr int LM=8192;using LP=dsr::PlanMT<TK,LM>;
    Dev<LP> p(1);Dev<int> idx(LM*TK),off(NE);Dev<float> wt(LM*TK);Dev<unsigned> masks(NE*(LM/32));
    std::vector<int> hi(idx.n);std::vector<float> hw(wt.n,0.25f);
    for(size_t i=0;i<hi.size();++i)hi[i]=i%TK;
    idx.put(hi.data());wt.put(hw.data());auto h=std::make_unique<LP>();
    for(int M:{4095,4096,8191,8192}) {
        CK(cudaMemset(masks.p,0,masks.n*sizeof(unsigned)));
        ds41v2moe::k_import_routes<NE,TK,LM><<<(M*8+255)/256,256>>>(idx.p,wt.p,M,p.p,masks.p);
        dsr::k_scan_ro<NE,TK,LM><<<1,NE>>>(p.p,masks.p,off.p);
        dsr::k_place_ro<NE,TK,LM,TK,true><<<(M*TK+255)/256,256>>>(p.p,masks.p,off.p,M);
        CK(cudaMemcpy(h.get(),p.p,sizeof(LP),cudaMemcpyDeviceToHost));
        const int nt=(M+7)/8;CHECK(h->n_slots==TK&&h->n_tiles==TK*nt);
        for(int m=0;m<M;++m)for(int k=0;k<TK;++k)CHECK(h->tok_slot[m][k]==k*nt+m/8&&h->tok_col[m][k]==m%8);
        printf("PASS plan capacity M=%d tiles=%d\n",M,h->n_tiles);
    }
}
int main(){for(int m:{1,5,33,71,97}){run(m,6);run(m,3);}large_plan();puts("DS41_V2_MOE_PASS");}
