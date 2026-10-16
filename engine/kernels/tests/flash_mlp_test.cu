// Standalone W3 per-op GPU gate. Python check_mlp.py prepares W1 rank safetensors + .meta offset maps.
// nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_FLASH=1 -DG6_NR=4 -DDEC_S2 \
//      engine/kernels/tests/flash_mlp_test.cu -o flash_mlp_test
// flash_mlp_test --router-check
// flash_mlp_test WEIGHTS.safetensors INPUT.f32 LAYER ROWS OUTPUT_PREFIX [FUSE]
#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
#include "../flash_mlp.cuh"
#include "../../h2d.h"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(1); } } while(0)
static void require(bool ok, const char* msg) { if (!ok) { fprintf(stderr,"FAIL %s\n",msg); exit(1); } }
static void checked(const char* err) { if (err) { fprintf(stderr,"FAIL %s\n",err); exit(1); } }
template<class T> static T* alloc(size_t n) { T* p; CK(cudaMalloc(&p,n*sizeof(T))); return p; }
template<class T> static T* copy(const std::vector<T>& v)
{ T* p=alloc<T>(v.size()); CK(g6h2d::copy(p,v.data(),v.size()*sizeof(T))); return p; }
static void dump(const std::string& path, const void* d, size_t n)
{
    std::vector<char> h(n); CK(cudaMemcpy(h.data(),d,n,cudaMemcpyDeviceToHost));
    std::ofstream f(path,std::ios::binary); f.write(h.data(),n); require(bool(f),"dump");
}
static std::vector<float> input(const std::string& path,int n)
{
    std::vector<float> h(n); std::ifstream f(path,std::ios::binary); f.read(reinterpret_cast<char*>(h.data()),n*4);
    require(bool(f),"input"); return h;
}
struct TensorRef
{
    const void* p=nullptr; long long shape[4]={}; int nd=0,fmt=bind::FMT_EXLR8_E4M3; bool bf16=false,fp32=false;
};
// .meta lines: NAME DTYPE ND DIMS... ABSOLUTE_OFFSET NBYTES (generated from safetensors header).
static std::map<std::string,TensorRef> load(const std::string& path)
{
    std::ifstream meta(path+".meta"), data(path,std::ios::binary); require(bool(meta)&&bool(data),"weights and .meta");
    std::map<std::string,TensorRef> out; std::string name,dt;
    while(meta>>name>>dt)
    {
        TensorRef t; meta>>t.nd; require(t.nd>=1&&t.nd<=4,"weight dimensions");
        for(int i=0;i<t.nd;++i)meta>>t.shape[i];
        size_t off,n;meta>>off>>n;require(bool(meta),"weight metadata");
        std::vector<char> h(n);data.seekg(off);data.read(h.data(),n);require(bool(data),"weight payload");
        t.bf16=dt=="BF16"; t.fp32=dt=="F32";
        if(name.find("router.bias")!=std::string::npos)require(dt=="F32","router.bias dtype");
        t.p=copy(h);out.emplace(name,t);
    }
    return out;
}
static void check_plan(const glue::RoutePlan& p, const std::vector<float>& logits, const std::vector<float>& bias,int M)
{
    int seen[glue::MMAX][glue::TOPK]={};
    require(p.n_slots>0&&p.n_slots<=M*glue::TOPK,"slot count");
    for(int m=0;m<M;++m)
    {
        std::vector<int> ids(geom::NEXP);std::iota(ids.begin(),ids.end(),0);
        std::vector<float> s(geom::NEXP);
        for(int e=0;e<geom::NEXP;++e)s[e]=1.f/(1.f+expf(-logits[m*geom::NEXP+e]));
        std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return s[a]+bias[a]>s[b]+bias[b];});
        float sum=0;for(int k=0;k<geom::TOPK;++k)sum+=s[ids[k]];
        for(int k=0;k<geom::TOPK;++k)
        {
            const int e=p.tok_expert[m][k],slot=p.tok_slot[m][k],col=p.tok_col[m][k];
            require(e==ids[k],"top-k (including experts 256..287 and ties)");
            require(std::isfinite(p.tok_w[m][k])&&fabsf(p.tok_w[m][k]-s[e]/(sum+1e-20f)*2.5f)<2e-6f,"router weights");
            require(slot>=0&&slot<p.n_slots&&col>=0&&col<8,"token slot bounds");
            require(p.slot_expert[slot]==e&&p.slot_tok[slot][col]==m,"slot round trip");
            require((p.slot_mask[slot]>>col)&1,"slot mask");
            require(p.slot_w[slot][col]==p.tok_w[m][k],"slot weight round trip");
        }
    }
    for(int slot=0;slot<p.n_slots;++slot)
    {
        if(slot)require(p.slot_expert[slot]>=p.slot_expert[slot-1],"ascending experts");
        for(int col=0;col<8;++col)
        {
            if(!((p.slot_mask[slot]>>col)&1)) { require(p.slot_tok[slot][col]==-1&&p.slot_w[slot][col]==0,"empty slot column");continue; }
            const int m=p.slot_tok[slot][col];require(m>=0&&m<M,"slot token bounds");
            int k=0;while(k<geom::TOPK&&p.tok_expert[m][k]!=p.slot_expert[slot])++k;
            require(k<geom::TOPK,"unexpected slot route"); ++seen[m][k];
        }
    }
    for(int m=0;m<M;++m)for(int k=0;k<geom::TOPK;++k)require(seen[m][k]==1,"each route once");
    for(int m=M;m<glue::MMAX;++m)for(int k=0;k<geom::TOPK;++k)
        require(p.tok_slot[m][k]==-1&&p.tok_expert[m][k]==-1&&p.tok_w[m][k]==0,"inactive token rows");
}
static void router_check()
{
    std::mt19937 rng(7301); std::uniform_real_distribution<float> noise(-3,3);
    auto* dp=alloc<glue::RoutePlan>(1);
    for(int M:{1,3,8,9,16,32})
    {
        if(M>G6_MX)continue;
        for(int mode=0;mode<4;++mode)
        {
            std::vector<float> x(M*geom::NEXP),b(geom::NEXP);
            for(int e=0;e<geom::NEXP;++e)b[e]=mode==0?e*.0031f:mode==1&&e>=256?20.f+e*.001f:0;
            for(float& v:x)v=mode<2?noise(rng):mode==2?0.f:-1000.f;
            auto* dx=copy(x);auto* db=copy(b);glue::k_router_plan<<<1,glue::NEXP>>>(dx,db,M,dp);
            CK(cudaGetLastError());glue::RoutePlan p;CK(cudaMemcpy(&p,dp,sizeof(p),cudaMemcpyDeviceToHost));
            check_plan(p,x,b,M);CK(cudaFree(dx));CK(cudaFree(db));
        }
        printf("router M=%d: random/tail/ties/underflow + all slot invariants PASS\n",M);
    }
    CK(cudaFree(dp));
}
static void binder_reject_test()
{
    // No CUDA calls: invalid types must fail before scan_widths reads a device table.
    auto find=[](const std::string& n) {
        TensorRef t;
        if(n=="L003.tr3.table") { t.p=reinterpret_cast<void*>(1);t.nd=2;t.shape[0]=geom::NEXP+1;t.shape[1]=6; }
        if(n=="L003.router.w") { t.p=reinterpret_cast<void*>(1);t.bf16=true;t.nd=2;t.shape[0]=geom::NEXP;t.shape[1]=HID; }
        if(n=="L003.router.bias") { t.p=reinterpret_cast<void*>(1);t.nd=1;t.shape[0]=geom::NEXP; }
        return t;
    };
    flash_mlp::Weights w;
    const char* e=flash_mlp::load(3,find,w);
    require(e&&std::string(e)=="Flash router.bias must be F32 [NEXP]","reject non-F32 router bias before CUDA");
    puts("CPU binder: non-BF16 is not implicitly FP32, invalid bias rejected PASS");
}
int main(int argc,char**argv)
{
    if(argc==2&&std::string(argv[1])=="--binder-reject-test"){binder_reject_test();return 0;}
    if(argc==2&&std::string(argv[1])=="--router-check"){router_check();return 0;}
    require(argc==6||argc==7,"usage WEIGHTS INPUT LAYER ROWS OUTPUT_PREFIX [FUSE]");
    const int layer=atoi(argv[3]),rows=atoi(argv[4]),fuse=argc==7?atoi(argv[6]):0;
    require(rows>=1&&rows<=G6_MX,"rows");
    const std::string output=argv[5]; auto tensors=load(argv[1]);
    auto find=[&](const std::string& name){auto it=tensors.find(name);return it==tensors.end()?TensorRef():it->second;};
    flash_mlp::Weights w;checked(flash_mlp::load(layer,find,w));CK(bind::init());
    void* scratch=alloc<char>(bind::scratch_bytes(rows));CK(cudaMemset(scratch,0,bind::scratch_bytes(rows)));
    auto sc=bind::scratch_at(scratch,rows); void* extra=alloc<char>(flash_mlp::workspace_bytes(rows));
    auto ws=flash_mlp::workspace_at(extra,rows);auto* plan=alloc<glue::RoutePlan>(1);
    auto* x=copy(input(argv[2],rows*HID));auto* out=alloc<uint16_t>(rows*HID);
    float* route_input = w.moe ? copy(input(std::string(argv[2])+".router.f32", rows*geom::NEXP)) : nullptr;
    for(int M:{1,3,8,9,16,32})
    {
        if(M>rows)continue;
        flash_mlp::Block b;checked(flash_mlp::make(w,M,x,out,plan,ws,sc,b,fuse));
        flash_mlp::launch(b,nullptr); CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        std::string pre=output+".m"+std::to_string(M);
        dump(pre+".out.bf16",out,static_cast<size_t>(M)*HID*2);
        if(w.moe)
        {
            dump(pre+".logits.f32",ws.logits,M*geom::NEXP*4);
            glue::RoutePlan hp;CK(cudaMemcpy(&hp,plan,sizeof(hp),cudaMemcpyDeviceToHost));
            dump(pre+".topk.i32",plan->tok_expert,M*geom::TOPK*4);
            dump(pre+".weights.f32",plan->tok_w,M*geom::TOPK*4);
            // Gate router and expert execution from the exact W0 logits, not a chain of near-tie decisions.
            glue::k_router_plan<<<1,glue::NEXP>>>(route_input,w.bias,M,plan);
            bind::launch_k1(b.k1,nullptr);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
            dump(pre+".route_input.topk.i32",plan->tok_expert,M*geom::TOPK*4);
            dump(pre+".route_input.weights.f32",plan->tok_w,M*geom::TOPK*4);
            dump(pre+".fixed_route.out.bf16",out,static_cast<size_t>(M)*HID*2);
        }
        else dump(pre+".out.f32",ws.dense_out,static_cast<size_t>(M)*HID*4);
        printf("layer=%d M=%d fuse=%d GPU output -> %s\n",layer,M,fuse,pre.c_str());
    }
    for(auto& pair:tensors)CK(cudaFree(const_cast<void*>(pair.second.p)));
    CK(cudaFree(route_input));CK(cudaFree(x));CK(cudaFree(out));CK(cudaFree(plan));CK(cudaFree(extra));CK(cudaFree(scratch));
}
