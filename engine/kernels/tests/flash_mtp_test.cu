// Per-op GPU gate: every operation starts from golden inputs, not a drifting chain.
// BF16 expert matrices are streamed one at a time; K4 is a separate gate.
// nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=4 -DDEC_S2 \
//   engine/kernels/tests/flash_mtp_test.cu -o flash_mtp_test
// flash_mtp_test FIXTURE [BATCH=8]
#include "../flash_mtp.cuh"
#include "../../mhc/mhc.cuh"
#include "../../h2d.h"
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) {fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
static std::string dir;
static int fails=0;
static void require(bool b,const char* s) {if(!b){fprintf(stderr,"FAIL %s\n",s);exit(2);}}
template<class T> std::vector<T> read(const std::string& n,size_t expected=0) {
    std::ifstream f(dir+"/"+n,std::ios::binary|std::ios::ate); require(bool(f),n.c_str());
    auto bytes=f.tellg(); require(bytes>=0 && (size_t)bytes%sizeof(T)==0,"file alignment");
    std::vector<T> v((size_t)bytes/sizeof(T)); require(!expected || v.size()==expected,"file size");
    f.seekg(0);f.read((char*)v.data(),bytes);require(bool(f),"read");return v;
}
template<class T> T* alloc(size_t n) {T* p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> T* up(const std::vector<T>& v) {T* p=alloc<T>(v.size());CK(g6h2d::copy(p,v.data(),v.size()*sizeof(T)));return p;}
template<class T> T* load(const std::string& n,size_t count=0){return up(read<T>(n,count));}
static void gate(const char* name,const float* d,const std::vector<float>& ref,double bar=1e-4) {
    std::vector<float> got(ref.size());CK(cudaMemcpy(got.data(),d,got.size()*4,cudaMemcpyDeviceToHost));
    double a=0,b=0; bool finite=true;
    for(size_t i=0;i<ref.size();++i) {finite &= std::isfinite(got[i]);double e=(double)got[i]-ref[i];a+=e*e;b+=(double)ref[i]*ref[i];}
    double rel=sqrt(a/std::max(b,1e-300));bool ok=finite && rel<=bar; fails+=!ok;
    printf("%-22s rel_l2=%.9g gate=%.3g %s\n",name,rel,bar,ok?"PASS":"FAIL");fflush(stdout);
}
static void gate(const char* name,const float* d,const std::string& file,double bar=1e-4){gate(name,d,read<float>(file),bar);}
struct Array {const float* x;__device__ float operator()(size_t i)const{return x[i];}};
__global__ void k_weighted(float* sum,const float* expert,const int* ids,const float* weights,int e,int n) {
    int i=blockIdx.x*256+threadIdx.x;if(i>=n*geom::HID)return;
    int row=i/geom::HID;
    for(int k=0;k<geom::TOPK;++k)if(ids[row*geom::TOPK+k]==e)sum[i]+=expert[i]*weights[row*geom::TOPK+k];
}
static void linear(const float* x,const uint16_t* w,float* y,int rows,int K,int N,int batch) {
    for(int a=0;a<rows;a+=batch)flash_mlp::k_linear<<<dim3((N+7)/8,std::min(batch,rows-a)),256>>>(x+(size_t)a*K,w,y+(size_t)a*N,K,N);
}
int main(int argc,char**argv) {
    require(argc==2||argc==3,"usage FIXTURE [BATCH]");dir=argv[1];int batch=argc==3?atoi(argv[2]):8;
    require(batch>0 && batch<=G6_MX,"batch range");int n,ne;std::ifstream meta(dir+"/meta.txt");meta>>n>>ne;
    require(bool(meta)&&n>0&&n<=64&&ne>0&&ne<=geom::NEXP,"metadata");std::vector<int> experts(ne);for(int& e:experts)meta>>e;
    require(bool(meta),"expert list");constexpr int H=geom::HID,I=geom::EXP_I,V=geom::VOCAB;
    CK(bind::init());
    int* tok=load<int>("tok.i32",n),*pos=load<int>("pos.i32",n);
    auto emb=load<__nv_bfloat16>("emb.bf16",(size_t)n*H);
    auto en=load<__nv_bfloat16>("enorm.bf16",H),hn=load<__nv_bfloat16>("hnorm.bf16",H);
    auto prev=load<float>("MTP.prev_hidden.f32",(size_t)n*H),cat=alloc<float>((size_t)n*2*H),out=alloc<float>((size_t)n*H);
    for(int a=0;a<n;a+=batch)flash_mtp::input(tok+a,pos+a,emb,en,hn,prev+(size_t)a*H,cat+(size_t)a*2*H,std::min(batch,n-a),nullptr);
    gate("concat norms / pos0",cat,"cat.f32");
    // Restore golden concat before projection.
    CK(cudaFree(cat));cat=load<float>("cat.f32");auto eh=load<uint16_t>("eh.bf16",(size_t)H*2*H);
    linear(cat,eh,out,n,2*H,H,batch);gate("eh_proj",out,"MTP.eh.f32");CK(cudaFree(eh));
    auto ehgold=load<float>("MTP.eh.f32");auto an=load<__nv_bfloat16>("attn_norm.bf16",H);
    flash_mtp::norm(ehgold,an,out,nullptr,n,nullptr);gate("attention norm",out,"MTP.attn_in.f32");
    auto attn=load<float>("L45.attn.out.f32");
    flash_mtp::k_add<<<(n*H+255)/256,256>>>(ehgold,Array{attn},n*H);gate("attention residual",ehgold,"MTP.mid.f32");
    auto mid=load<float>("MTP.mid.f32");auto mn=load<__nv_bfloat16>("mlp_norm.bf16",H);
    flash_mtp::norm(mid,mn,out,nullptr,n,nullptr);gate("MLP norm",out,"MTP.mlp_in.f32");
    auto x=load<float>("MTP.mlp_in.f32");auto router=load<uint16_t>("router.bf16",(size_t)geom::NEXP*H);
    auto logits=alloc<float>((size_t)n*geom::NEXP);
    linear(x,router,logits,n,H,geom::NEXP,batch);gate("router logits",logits,"L45.moe.router_logits.f32");
    auto goldlog=load<float>("L45.moe.router_logits.f32"),bias=load<float>("router_bias.f32",geom::NEXP);
    auto ids=read<int>("topk.i32",n*geom::TOPK);auto weights=read<float>("L45.moe.topk_w.f32",n*geom::TOPK);
    auto plan=alloc<glue::RoutePlan>(1);double werr=0;int choices=0;
    for(int a=0;a<n;a+=batch) {
        int M=std::min(batch,n-a);glue::k_router_plan<<<1,geom::NEXP>>>(goldlog+(size_t)a*geom::NEXP,bias,M,plan);
        glue::RoutePlan p;CK(cudaMemcpy(&p,plan,sizeof p,cudaMemcpyDeviceToHost));
        for(int r=0;r<M;++r)for(int k=0;k<geom::TOPK;++k){choices+=p.tok_expert[r][k]!=ids[(a+r)*geom::TOPK+k];werr=std::max(werr,(double)fabs(p.tok_w[r][k]-weights[(a+r)*geom::TOPK+k]));}
    }
    bool route_ok=!choices&&werr<1e-4;fails+=!route_ok;printf("router plan choices=%d max_abs_weight=%.9g %s\n",choices,werr,route_ok?"PASS":"FAIL");
    auto dids=up(ids);auto dw=up(weights),act=alloc<float>((size_t)n*I),sum=alloc<float>((size_t)n*H);
    CK(cudaMemset(sum,0,(size_t)n*H*4));
    for(int j=0;j<=ne;++j) {
        std::string tag=j==ne?"shared":std::to_string(experts[j]);
        auto g=load<uint16_t>("e"+tag+".gate.bf16",(size_t)I*H),u=load<uint16_t>("e"+tag+".up.bf16",(size_t)I*H),d=load<uint16_t>("e"+tag+".down.bf16",(size_t)H*I);
        for(int a=0;a<n;a+=batch)flash_mlp::k_dense_gate<<<dim3((I+7)/8,std::min(batch,n-a)),256>>>(x+(size_t)a*H,g,u,act+(size_t)a*I,I);
        linear(act,d,out,n,I,H,batch);
        if(j<ne)k_weighted<<<(n*H+255)/256,256>>>(sum,out,dids,dw,experts[j],n);
        else {
            gate("BF16 routed experts",sum,"L45.moe.routed.f32");gate("BF16 shared expert",out,"L45.moe.shared.f32");
            flash_mtp::k_add<<<(n*H+255)/256,256>>>(sum,Array{out},n*H);gate("BF16 MoE output",sum,"L45.moe.out.f32");
        }
        CK(cudaFree(g));CK(cudaFree(u));CK(cudaFree(d));
    }
    CK(cudaFree(sum));sum=load<float>("L45.moe.out.f32");
    flash_mtp::k_add<<<(n*H+255)/256,256>>>(mid,Array{sum},n*H);gate("MLP residual",mid,"MTP.out.f32");
    auto final=load<float>("MTP.out.f32");auto norm=load<__nv_bfloat16>("head_norm.bf16",H);
    flash_mtp::norm(final,norm,out,nullptr,n,nullptr);gate("shared head norm",out,"head_in.f32");
    // Production BF16 .gv head path, one rank slice at a time, from identical golden hidden.
    auto hidden=load<float>("head_in.f32"),hlog=alloc<float>((size_t)n*V);
    auto fragments=alloc<u32>((size_t)H/16*32*2*n);
    for(int r=0;r<g6nr::NR;++r) {
        int lo,vn;require(bind::vocab_slice(r,lo,vn),"vocab rank");
        auto w=load<uint8_t>("head.r"+std::to_string(r)+".gv",(size_t)vn*H*2);
        bind::Tensor t;t.p=w;t.nd=3;t.shape[0]=vn/16;t.shape[1]=H/16;t.shape[2]=512;
        for(int a=0;a<n;a+=batch) {
            int M=std::min(batch,n-a);bind::Gv gv;
            flash_mhc::fragments(hidden+(size_t)a*H,M,H,nullptr,fragments,nullptr);
            auto tmp=alloc<float>((size_t)M*vn);
            const char* err=bind::make_gv(t,bind::BF16W,1,32,M,fragments,tmp,gv::O_F32,gv);require(!err,"head binding");
            bind::launch(gv,nullptr);
            CK(cudaMemcpy2D(hlog+(size_t)a*V+lo,V*4,tmp,vn*4,vn*4,M,cudaMemcpyDeviceToDevice));CK(cudaFree(tmp));
        }
        CK(cudaFree(w));
    }
    gate("shared lm_head",hlog,"MTP.logits.f32",1e-2);
    auto reference=read<float>("MTP.logits.f32");std::vector<float> got(reference.size());CK(cudaMemcpy(got.data(),hlog,got.size()*4,cudaMemcpyDeviceToHost));
    int agree=0;double kl=0;
    for(int row=0;row<n;++row) {
        auto a=got.data()+(size_t)row*V,b=reference.data()+(size_t)row*V;
        auto ia=std::max_element(a,a+V)-a,ib=std::max_element(b,b+V)-b;agree+=ia==ib;
        double sa=0,sb=0;for(int j=0;j<V;++j){sa+=exp(a[j]-a[ia]);sb+=exp(b[j]-b[ib]);}
        double la=a[ia]+log(sa),lb=b[ib]+log(sb);
        for(int j=0;j<V;++j)kl+=exp(b[j]-lb)*((b[j]-lb)-(a[j]-la));
    }
    kl/=n;bool ok=agree>=.99*n&&std::isfinite(kl)&&kl<=1e-3;fails+=!ok;
    printf("head top1=%d/%d KL=%.9g %s\n",agree,n,kl,ok?"PASS":"FAIL");
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    printf("MTP per-op BF16 batch=%d %s (%d failures)\n",batch,fails?"FAIL":"PASS",fails);return fails?1:0;
}
