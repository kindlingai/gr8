// Synthetic FFN numerical test: own-row routing/gather assembly -> production
// shared/routed up -> grouped down/compact combine, versus full-column down.
// No fabric or model; links the actual attention TU (shared expert entrypoints).
// nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -DG6_MODEL_DS41=1 \
//   engine/runtime/tests/ds41_v2_ffn_test.cu engine/attn/ds41_attn.cu \
//   engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm -o ds41_v2_ffn_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <cstring>
#include "../ds41_v2_ffn.cuh"
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));exit(2);} } while(0)
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
namespace f=ds41::v2::ffn;
template<class T> struct Dev {
    T* p; size_t n; explicit Dev(size_t nn):n(nn){CK(cudaMalloc(&p,n*sizeof(T)));}~Dev(){cudaFree(p);}
    void put(const T* v){CK(cudaMemcpy(p,v,n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
void mx4fill(Dev<uint8_t>& d){std::vector<uint8_t> v(d.n);for(size_t j=0;j<v.size();++j){uint32_t h=hash(j);v[j]=j%mx4::GR<512?uint8_t((h&0x33)|((h>>6)&0x88)):uint8_t(116+h%4);}d.put(v.data());}
// Exact gate comparison, independent of routing/FFN tolerances below. Run
// --gate-only to avoid allocating expert scratch on a shared test GPU.
void gate_equal() {
    constexpr int H=f::H, NE=f::NE, GUARD=32;
    CK(cudaFuncSetAttribute(dsr::k_gate_logits_ro<NE>,cudaFuncAttributeMaxDynamicSharedMemorySize,dsr::GRO_SMEM));
    Dev<__nv_bfloat16> gate(NE*H);
    std::vector<__nv_bfloat16> hg(gate.n);
    for(size_t j=0;j<hg.size();++j)hg[j]=__float2bfloat16(std::ldexp(float(int(hash(j)%511)-255),-10+int(hash(j+7)%5)));
    gate.put(hg.data());
    for(int M:{1,7,8,9,2047,2048,2049,4096,8192}) {
        Dev<__nv_bfloat16> x(size_t(M)*H);
        std::vector<__nv_bfloat16> hx(x.n);
        for(size_t j=0;j<hx.size();++j)hx[j]=__float2bfloat16(std::ldexp(float(int(hash(j+31)%511)-255),-8+int(hash(j+53)%5)));
        x.put(hx.data());
        Dev<float> mt(size_t(M)*NE+2*GUARD),ro(mt.n);
        CK(cudaMemset(mt.p,0xa5,mt.n*sizeof(float)));
        CK(cudaMemset(ro.p,0xa5,ro.n*sizeof(float)));
        dsr::k_gate_logits_mt<NE,0><<<NE/8,256>>>(gate.p,x.p,M,mt.p+GUARD); CK(cudaGetLastError());
        dsr::k_gate_logits_ro<NE><<<NE/8,256,dsr::GRO_SMEM>>>(gate.p,x.p,M,ro.p+GUARD); CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        auto a=mt.get(),b=ro.get();
        CHECK(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
        std::vector<unsigned char> sentinel(GUARD*sizeof(float),0xa5);
        CHECK(std::memcmp(a.data(),sentinel.data(),sentinel.size())==0);
        CHECK(std::memcmp(a.data()+GUARD+size_t(M)*NE,sentinel.data(),sentinel.size())==0);
        for(size_t j=GUARD;j<GUARD+size_t(M)*NE;++j)CHECK(std::isfinite(a[j]));
        printf("PASS gate M=%d NE=%d logits BYTE-EQUAL (memcmp), guards intact\n",M,NE);
    }
    puts("DS41_V2_GATE_RO_PASS");
}
void gate_log_test() {
    f::Scratch q;
    FILE* log=std::tmpfile(); CHECK(log);
    q.select_request(7,0,2,log); q.select_request(7,0,2,log);
    q.select_request(9,1,2,log); q.select_request(7,0,2,log);
    q.gate_ro=true; q.select_request(10,0,2,log);
    CHECK(q.gate_ro_seen.size()==2);
    std::rewind(log); char text[512]={}; CHECK(std::fread(text,1,sizeof(text)-1,log)>0);
    CHECK(std::strcmp(text,"# ds41_v2_gate_ro request=7 on=0 rank=2\n"
        "# ds41_v2_gate_ro request=9 on=0 rank=2\n"
        "# ds41_v2_gate_ro request=10 on=1 rank=2\n")==0);
    std::fclose(log); puts("PASS gate logging: off/on, repeated/interleaved chunks, slot reuse");
}
int main(int argc,char** argv){
    gate_log_test();
    gate_equal();
    if(argc==2 && std::strcmp(argv[1],"--gate-only")==0)return 0;
    constexpr int H=f::H, I=f::I, NE=f::NE, TK=f::TK, MM=f::MM, E=32, CAP=97;
    f::Scratch q;CK(f::prepare(q,CAP));
    printf("FFN resident_fc2=%d (V2=%d FC2=%d)\n",int(q.resident_fc2),int(knob::ds_pf_v2()),int(knob::ds_pf_v2_fc2()));
    CHECK(q.resident_fc2==(knob::ds_pf_v2()&&knob::ds_pf_v2_fc2()));
    CHECK(q.gate_ro==knob::ds_pf_v2_gate_ro());
    Dev<__nv_bfloat16> gate(NE*H),x(CAP*H),out(CAP*H);
    Dev<float> bias(NE),vl(NE);Dev<uint8_t> flags(CAP);
    std::vector<__nv_bfloat16> hg(gate.n),hx(x.n);std::vector<float> hb(NE),hv(NE);
    for(size_t j=0;j<hg.size();++j)hg[j]=__float2bfloat16(float(int(hash(j)%127)-63)/1024);
    for(size_t j=0;j<hx.size();++j)hx[j]=__float2bfloat16(float(int(hash(j+31)%511)-255)/128);
    for(int e=0;e<NE;++e){hb[e]=e<E?float(e%7)*.03f:-10000.f;hv[e]=e<E?float((31-e)%7)*.04f:-10000.f;}
    gate.put(hg.data());x.put(hx.data());bias.put(hb.data());vl.put(hv.data());
    std::vector<uint8_t> hf(CAP);for(int m=0;m<CAP;++m)hf[m]=m%3==1;flags.put(hf.data());
    Dev<uint8_t> fc1(size_t(E)*(I/16)*(H/64)*2*mx4::GR),fc2(size_t(E)*(H/16)*(I/64)*mx4::GR);
    mx4fill(fc1);mx4fill(fc2);
    Dev<uint8_t> w13(2*I*H),sw13(2*I*(H/32)),w2(H*f::KP),sw2(H*(f::KP/32));
    std::vector<uint8_t> ww13(w13.n),ww2(w2.n),ss13(sw13.n,118),ss2(sw2.n,118);
    for(size_t j=0;j<ww13.size();++j)ww13[j]=uint8_t(hash(j+19)%0x60 | (hash(j)&0x80));
    for(size_t j=0;j<ww2.size();++j)ww2[j]=j%f::KP<I?uint8_t(hash(j+37)%0x60 | (hash(j)&0x80)):0;
    w13.put(ww13.data());sw13.put(ss13.data());w2.put(ww2.data());sw2.put(ss2.data());
    f::Layer w{gate.p,bias.p,vl.p,fc1.p,fc2.p,w13.p,sw13.p,w2.p,sw2.p,10.f};
    Dev<f::Plan> rp(1);Dev<float> gl(CAP*NE);Dev<unsigned> masks(NE*(MM/32));
    Dev<__nv_bfloat16> shared(CAP*H);Dev<__half> full(size_t(q.tiles)*8*H);
    // Each own-row router exports into the assembled gather destinations before
    // the next call reuses its local scratch. This is a numerical gather model,
    // not a transport or whole-engine test.
    for(int M:{1,5,71,97})for(int cut:{3,6}) {
        for(int rank=0;rank<4;++rank){auto g=ds41::v2::geometry(M,4,rank);
            CK(f::router_own(g,w,q,x.p+size_t(g.row0)*H,flags.p+g.row0,cut,nullptr));
            if(!g.owned)continue;
#define COPY(field,own,width) CK(cudaMemcpyAsync(q.field+size_t(g.row0)*width,q.own,size_t(g.owned)*width*sizeof(*q.field),cudaMemcpyDeviceToDevice))
            COPY(all_q,own_q,H);COPY(all_s,own_s,H/32);COPY(all_idx,own_idx,TK);COPY(all_wt,own_wt,TK);
#undef COPY
        }
        dsr::k_gate_logits_mt<NE,0><<<NE/8,256>>>(gate.p,x.p,M,gl.p);
        CK(cudaMemset(masks.p,0,masks.n*sizeof(unsigned)));
        if(cut==6)dsr::k_route_ro<NE,TK,MM,6><<<(M+7)/8,256>>>(gl.p,bias.p,M,rp.p,masks.p,vl.p,flags.p);
        else dsr::k_route_ro<NE,TK,MM,3><<<(M+7)/8,256>>>(gl.p,bias.p,M,rp.p,masks.p,vl.p,flags.p);
        auto r=std::make_unique<f::Plan>();CK(cudaMemcpy(r.get(),rp.p,sizeof(f::Plan),cudaMemcpyDeviceToHost));
        std::vector<int> idx(M*TK);std::vector<float> wt(M*TK);
        CK(cudaMemcpy(idx.data(),q.all_idx,idx.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(wt.data(),q.all_wt,wt.size()*4,cudaMemcpyDeviceToHost));
        for(int m=0;m<M;++m)for(int k=0;k<TK;++k)CHECK(idx[m*TK+k]==r->tok_expert[m][k]&&fabs(wt[m*TK+k]-r->tok_w[m][k])<1e-6);
        auto g=ds41::v2::geometry(M,4,0);CK(f::up(g,w,q,nullptr));
        // Same quantized up output, stock full-column down and full shared down.
        auto a=f::args(w,q,M);a.ypart=full.p;a.os2=16;
        constexpr size_t SM=mxtc::fc2_smem<4,8,8>();auto kd=mxtc::k_fc2_tc<I,H,4,2,8,8>;
        CK(cudaFuncSetAttribute(kd,cudaFuncAttributeMaxDynamicSharedMemorySize,SM));kd<<<a.n_slots*a.gy2*a.os2,128,SM>>>(a);
        CHECK(!ds41_gemm_mx_tc(q.sh_q,f::KP,q.sh_s,f::KP/32,w.w2,w.sw2,shared.p,H,M,H,f::KP,nullptr));
        CK(cudaMemcpy(r.get(),q.plan,sizeof(f::Plan),cudaMemcpyDeviceToHost));auto pf=full.get();auto sh=shared.get();
        double maxerr=0;
        for(int c=0;c<4;++c){CK(f::down_group(g,w,q,c,nullptr));std::vector<__nv_bfloat16> y(g.padded()*f::COLS);
            CK(cudaMemcpy(y.data(),q.send[c],y.size()*2,cudaMemcpyDeviceToHost));
            for(int m=0;m<g.padded();++m)for(int n=0;n<f::COLS;++n){float v=0;if(m<M){
                for(int k=0;k<cut;++k)v+=__half2float(pf[(size_t(r->tok_slot[m][k])*8+r->tok_col[m][k])*H+c*f::COLS+n]);
                v+=__bfloat162float(sh[m*H+c*f::COLS+n]);}
                float expected=__bfloat162float(__float2bfloat16(v)),actual=__bfloat162float(y[m*f::COLS+n]);
                double d=fabs(actual-expected);maxerr=std::max(maxerr,d);CHECK(std::isfinite(actual)&&d<=.008*(1+fabs(expected)));
            }
            // Synthetic rank-0 receive to exercise group assembly (RS itself is C's test).
            CK(cudaMemcpy(q.recv[c],q.send[c],g.owned*f::COLS*2,cudaMemcpyDeviceToDevice));
        }
        f::k_assemble<<<(g.owned*H+255)/256,256>>>(q.recv[0],q.recv[1],q.recv[2],q.recv[3],out.p,g.owned);auto assembled=out.get();
        for(int c=0;c<4;++c){std::vector<__nv_bfloat16> y(g.owned*f::COLS);CK(cudaMemcpy(y.data(),q.send[c],y.size()*2,cudaMemcpyDeviceToHost));
            for(int m=0;m<g.owned;++m)for(int n=0;n<f::COLS;++n)CHECK(__bfloat162float(assembled[m*H+c*f::COLS+n])==__bfloat162float(y[m*f::COLS+n]));}
        printf("PASS FFN M=%d cut=%d router=image/text+ragged max_abs=%g full-vs-grouped shared/routed/combine/assemble\n",M,cut,maxerr);
    }
    CK(cudaDeviceSynchronize());f::release(q);puts("DS41_V2_FFN_PASS");
}
