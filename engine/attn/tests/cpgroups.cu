// cpgroups: G6_CP_ONECHAIN on one GPU. Past R every layer of a CP step runs, on each rank, the other ranks' head groups
// over its own keys (rank.cuh launch_attn_cp, phase 1). Stock (G6_CP_S3=1, as served): per remote group k = 1..nr-1 a
// k_vget of its q / rope from the q all-gather slot, g6a_cp_group (scores, cp_softmax, context parts, merge) of 16
// heads, a k_vput of its partial. ONE: one k_vget_heads (the three payloads side by side), one g6_attn_cp_groups of 48
// heads (each partial into its own block), the same three k_vput. The fabric is device memory here (the recv slots hold
// what the peers' k_vput packed, the send slots are what this rank publishes; doorbells / tickets / error word in device
// memory), so the step is rank 0 of NR=4 at live context L in capture bucket T with the CP page map of R (pages < R/64
// replicated, the rest every NR-th page), indices = 2048 distinct positions <= the row's position (uniform; CPG_LOCAL=n:
// the n most recent positions first, as a local window), so about 1/NR of the slots are this rank's (printed).
//   EXACT  bytes of the three published partials (send slots, poisoned before every step) ONE vs stock, x chk.
//   TIME   one step's phase-1 body (the 7 / 9 launches above) inside a graph of K steps, median of TR alternating trials,
//          hot (no flush) and cold (a 96 MB L2 flush before every step, subtracted: the served case, the experts stream
//          between layers). Saved per layer, and per step x 78 layers (every layer takes the CP block past R).
// Arms: env CPG_ARMS="arm arm ..." with an arm = comma-separated KEY=VALUE attention flags (SC2 TR CTX2 CTX3 HG CTX4) or
// "stock" (all 0); default "SC2=4,TR=2,CTX2=4,CTX3=2 stock" (the measured attention set of docs/SETTINGS.md, then the
// kernels without it).
// build (from engine/attn/tests), with the engine's flags (scripts/build.sh K1H2_FLAGS, -DG6_NR=4 -DDEC_S2 -DPF_F12):
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $FLAGS -o cpgroups cpgroups.cu ../bind.cu -lcublas
// run: cpgroups T L M [R=65536] [reps=10] [K=16] [TR=7] [chk=3]
//   sweep: for M in 1 3 12; do ./cpgroups 131072 75000 $M; done; ./cpgroups 131072 70000 3; ./cpgroups 131072 80000 3
// Exit 3 on any byte difference, 1 on a CUDA / validation error.
#include "../attn.cu"
#include "../../kernels/cpglue.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#define CK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)
#define CKI(x) do{int e_=(x);if(e_){fprintf(stderr,"%s:%d status %d (%s)\n",__FILE__,__LINE__,e_,cudaGetErrorString((cudaError_t)e_));exit(1);}}while(0)

__device__ uint32_t hsh(uint64_t x){x^=x>>33;x*=0xff51afd7ed558ccdull;x^=x>>33;x*=0xc4ceb9fe1a85ec53ull;x^=x>>33;return (uint32_t)x;}
__device__ float unif(uint64_t seed,uint64_t i){return (float)(hsh(seed*0x9e3779b97f4a7c15ull+i)>>8)*(2.f/16777216.f)-1.f;}   // [-1, 1)
__global__ void fill_f32(float* p,size_t n,uint64_t seed,float s){
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)p[i]=s*unif(seed,i);}
__device__ uint8_t fin8(uint32_t v){uint8_t b=(uint8_t)v;if((b&0x7f)==0x7f)b^=1;return b;}   // finite e4m3 code
__global__ void fill_kv(uint8_t* c,size_t rows,uint64_t seed){   // latent rows (kvfmt.h) with finite codes, positive scales
    for(size_t r=blockIdx.x;r<rows;r+=gridDim.x){uint8_t* v=c+r*g6kv::ROW;const int t=threadIdx.x;
#if G6_KV4
        if(t<256)v[t]=(uint8_t)hsh(seed+r*1000+t);
        if(t<16)((__half*)(v+g6kv::SCALE_OFF))[t]=__float2half(0.02f*(1.5f+unif(seed+7,r*16+t)));
#else
        for(int d=t;d<512;d+=blockDim.x)v[d]=fin8(hsh(seed+r*1000+d));
        if(t<4)((float*)(v+g6kv::SCALE_OFF))[t]=0.004f*(1.5f+unif(seed+7,r*4+t));
#endif
        if(t<64)((__nv_bfloat16*)(v+g6kv::ROPE_OFF))[t]=__float2bfloat16(0.5f*unif(seed+9,r*64+t));}
}
__global__ void l2_flush(const float4* p,size_t n,float* sink){
    float acc=0;for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i];acc+=v.x+v.y+v.z+v.w;}
    if(acc==1234.5f)*sink=acc;   // never (the buffer is zero); keeps the loads
}
template<class T> static T* dalloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));CK(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T> static std::vector<T> host(const T* d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static float med(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
template<class F> static cudaGraphExec_t inst(cudaStream_t st,F body){
    cudaGraph_t g;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));body();CK(cudaStreamEndCapture(st,&g));
    cudaGraphExec_t x;CK(cudaGraphInstantiate(&x,g,0));CK(cudaGraphDestroy(g));return x;}
static float run_us(cudaGraphExec_t x,cudaStream_t st,int reps){   // us per graph launch
    for(int i=0;i<2;i++)CK(cudaGraphLaunch(x,st));
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,st));
    for(int i=0;i<reps;i++)CK(cudaGraphLaunch(x,st));
    CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
    cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.f/reps;
}
struct Arm{int sc,tr,ctx2,ctx3,hg,ctx4;};
static void set_arm(const Arm& x){ga::attn_sc2()=x.sc;ga::attn_sc2_tr()=x.tr;ga::attn_ctx2()=x.ctx2;ga::attn_ctx3()=x.ctx3;ga::attn_ctx3_hg()=x.hg;ga::attn_ctx4()=x.ctx4;}
static bool parse_arm(const std::string& s,Arm& x){
    x=Arm{0,0,0,0,0,0};if(s=="stock")return true;
    size_t i=0;while(i<s.size()){size_t e=s.find(',',i);if(e==std::string::npos)e=s.size();const std::string kv=s.substr(i,e-i);i=e+1;
        const size_t q=kv.find('=');if(q==std::string::npos)return false;const std::string k=kv.substr(0,q);const int v=atoi(kv.c_str()+q+1);
        if(k=="SC2")x.sc=v;else if(k=="TR")x.tr=v;else if(k=="CTX2")x.ctx2=v;else if(k=="CTX3")x.ctx3=v;else if(k=="HG")x.hg=v;
        else if(k=="CTX4")x.ctx4=v;else return false;}
    return true;
}

int main(int argc,char** argv){
    if(argc<4){fprintf(stderr,"usage: cpgroups T L M [R=65536] [reps=10] [K=16] [TR=7] [chk=3]\n");return 2;}
    const int T=atoi(argv[1]),L=atoi(argv[2]),M=atoi(argv[3]),CPR=argc>4?atoi(argv[4]):65536,REPS=argc>5?atoi(argv[5]):10,
        K=argc>6?atoi(argv[6]):16,TR=argc>7?atoi(argv[7]):7,CHK=argc>8?atoi(argv[8]):3;
    const int NR=g6nr::NR,H=g6nr::HMAX,NG=NR-1,HA=NG*H,QT=32,PS=64;
    if(L>T||L<ga::TOP||M<1||M>QT||M>L||CPR<0||CPR%PS||CPR>=L){fprintf(stderr,"bad sizes (M <= 32, 2048 <= L <= T, R < L, R %% 64 == 0)\n");return 2;}
    if(g6nr::HMIN!=H||!g6a_cp_groups_ok(H)){fprintf(stderr,"needs equal head groups of a multiple of 16 heads (G6_NR=4) and G6_ATTN_FLIP off\n");return 2;}
    std::vector<Arm> arms;std::vector<std::string> specs={"SC2=4,TR=2,CTX2=4,CTX3=2","stock"};
    if(const char* e=getenv("CPG_ARMS")){specs.clear();std::string s=e;size_t i=0;
        while(i<s.size()){while(i<s.size()&&s[i]==' ')i++;size_t e2=s.find(' ',i);if(e2==std::string::npos)e2=s.size();if(e2>i)specs.push_back(s.substr(i,e2-i));i=e2;}}
    for(auto& sp:specs){Arm x;if(!parse_arm(sp,x)){fprintf(stderr,"CPG_ARMS: bad arm %s\n",sp.c_str());return 2;}arms.push_back(x);}
    const int LOCAL=getenv("CPG_LOCAL")?atoi(getenv("CPG_LOCAL")):0;
    cudaStream_t st;CK(cudaStreamCreate(&st));
    // CP page map of rank 0: pages < R/64 replicated, then every NR-th page (bind.cu create)
    const int tpages=(T+PS-1)/PS;std::vector<int> table(tpages);int phys=0;
    for(int lp=0;lp<tpages;lp++)table[lp]=(lp<CPR/PS||lp%NR==0)?phys++:G6_CP_REMOTE;
    uint8_t* mla=dalloc<uint8_t>((size_t)phys*PS*g6kv::ROW);fill_kv<<<1024,256,0,st>>>(mla,(size_t)phys*PS,100);
    int* dtab=dalloc<int>(tpages);CK(cudaMemcpy(dtab,table.data(),tpages*4,cudaMemcpyHostToDevice));
    // rows at L - M + i; per row 2048 distinct positions <= its position, in random order (the merged list's order is the
    // score order, unrelated to the position); CPG_LOCAL=n: the n most recent first
    std::vector<int> pos(M),ind((size_t)M*ga::TOP);std::mt19937_64 rng(12345);int mine=0;
    for(int r=0;r<M;r++){pos[r]=L-M+r;std::vector<int> pick;std::vector<char> used(pos[r]+1,0);
        for(int k=0;k<LOCAL&&k<=pos[r]&&(int)pick.size()<ga::TOP;k++){pick.push_back(pos[r]-k);used[pos[r]-k]=1;}
        while((int)pick.size()<ga::TOP){const int p=(int)(rng()%(uint64_t)(pos[r]+1));if(!used[p]){used[p]=1;pick.push_back(p);}}
        std::shuffle(pick.begin()+std::min<size_t>(pick.size(),(size_t)LOCAL),pick.end(),rng);
        for(int k=0;k<ga::TOP;k++){ind[(size_t)r*ga::TOP+k]=pick[k];mine+=(pick[k]/PS)%NR==0;}}
    int *dpos=dalloc<int>(M),*dind=dalloc<int>((size_t)M*ga::TOP);unsigned* err=dalloc<unsigned>(1);
    CK(cudaMemcpy(dpos,pos.data(),M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dind,ind.data(),ind.size()*4,cudaMemcpyHostToDevice));
    G6Attn a{};a.rows=M;a.heads=H;a.context=T;a.page_size=PS;a.page_count=phys;a.table_size=tpages;a.query_tile=QT;
    a.positions=dpos;a.block_table=dtab;a.mla_cache=mla;a.indices=dind;a.error=err;a.cp_nr=NR;a.cp_rank=0;
    G6AttnStepCell c{};c.context=L;c.page_count=phys;c.table_size=tpages;c.reuse_rows=0;c.valid=1;
    G6AttnStepCell* cell;CK(cudaMalloc(&cell,sizeof c));CK(cudaMemcpy(cell,&c,sizeof c,cudaMemcpyHostToDevice));
    // the fabric in device memory: q all-gather recv slots (both parities: q [M][H][512] | rope [M][H][64] per source, as
    // k_vput packs them), the three partial send slots, doorbells, tickets, step (parity 0)
    const size_t qn=(size_t)M*H*512*4,rn=(size_t)M*H*64*4,pb=(size_t)M*H*514*4,tblk=((size_t)M*H*514+3)&~(size_t)3;
    unsigned long long* bells=dalloc<unsigned long long>(16);u64* stepp=dalloc<u64>(1);unsigned* tick=dalloc<unsigned>(16);
    g6::VSlotDev vq{};vq.doorbell=bells;vq.error=err;vq.c=0;vq.ncoll=16;vq.nsrc=NG;
    for(int j=0;j<NG;j++){char* s=dalloc<char>(qn+rn+256);fill_f32<<<256,256,0,st>>>((float*)s,(qn+rn)/4,300+j,0.5f);
        vq.recv[0][j]=vq.recv[1][j]=s;vq.src[j]=j+1;}
    g6::VSlotDev vp[7]{};char* sendb[7]={};
    for(int k=0;k<NG;k++){sendb[k]=dalloc<char>(pb+256);vp[k]=g6::VSlotDev{};vp[k].send[0]=vp[k].send[1]=sendb[k];vp[k].doorbell=bells;
        vp[k].error=err;vp[k].c=1+k;vp[k].ncoll=16;}
    // buffers: stock (qbuf of one group, scratch, one partial); ONE (all groups' q, its buffer, NG partial blocks)
    float *qbuf=dalloc<float>((size_t)M*H*576),*ptmp=dalloc<float>(tblk),*qall=dalloc<float>((size_t)M*HA*576),*pall=dalloc<float>(NG*tblk);
    G6Attn a1=a;   // each remote group's call: the 16-head shape (rank.cuh: G6Attn ad = *a; ad.heads = hd)
    G6Attn aa=a;aa.heads=HA;
    const size_t fl=(size_t)96<<20;float4* fb=dalloc<float4>(fl/16);float* sink=dalloc<float>(1);
    auto flushk=[&]{l2_flush<<<48*6,256,0,st>>>(fb,fl/16,sink);};
    CK(cudaStreamSynchronize(st));
    {int dev=0,sms=0,clk=0,l2=0;CK(cudaGetDevice(&dev));cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
     cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,dev);cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,dev);
     printf("GPU %d SMs, attr clock %d MHz, L2 %.1f MB | T=%d L=%d M=%d R=%d NR=%d H=%d (one call %d heads) | phys pages %d, slots of rank 0 %.1f %% "
            "(local %d) | K=%d reps=%d TR=%d chk=%d\n",sms,clk/1000,l2/1048576.,T,L,M,CPR,NR,H,HA,phys,100.*mine/ind.size(),LOCAL,K,REPS,TR,CHK);}
    int bad=0;
    for(size_t ai=0;ai<arms.size();ai++){
        set_arm(arms[ai]);const std::string tag=specs[ai];
        const size_t sb=g6a_scratch_bytes(&a1),ob=g6a_cp_groups_bytes(HA,M);
        char *scr=dalloc<char>(sb),*obuf=dalloc<char>(ob);
        auto stock=[&]{for(int k=1;k<NR;k++){
            cpglue::k_vget<<<16,256,0,st>>>(vq,k-1,stepp,qbuf,qn+rn);
            CKI(g6a_cp_group(&a1,cell,scr,sb,qbuf,qbuf+qn/4,ptmp,st));
            cpglue::k_vput<<<16,256,0,st>>>(ptmp,pb,nullptr,0,vp[k-1],stepp,tick+k);}};
        auto one=[&]{cpglue::VHeads s{};for(int k=0;k<NG;k++)s.j[k]=k;float* rall=qall+(size_t)M*HA*512;
            cpglue::k_vget_heads<<<dim3(16,NG),256,0,st>>>(vq,s,stepp,qall,rall,M,H);
            CKI(g6a_cp_groups(&aa,cell,obuf,ob,qall,rall,pall,H,(int64_t)tblk,st));
            for(int k=1;k<NR;k++)cpglue::k_vput<<<16,256,0,st>>>(pall+(size_t)(k-1)*tblk,pb,nullptr,0,vp[k-1],stepp,tick+k);};
        // EXACT: the published partials, every step from poisoned buffers
        auto take=[&](bool o){std::vector<uint8_t> out;
            for(int k=0;k<NG;k++)CK(cudaMemsetAsync(sendb[k],0xab,pb,st));
            CK(cudaMemsetAsync(scr,0xcd,sb,st));CK(cudaMemsetAsync(obuf,0xcd,ob,st));CK(cudaMemsetAsync(ptmp,0xcd,tblk*4,st));
            CK(cudaMemsetAsync(pall,0xcd,NG*tblk*4,st));CK(cudaMemsetAsync(qbuf,0xcd,(size_t)M*H*576*4,st));CK(cudaMemsetAsync(qall,0xcd,(size_t)M*HA*576*4,st));
            cudaGraphExec_t g=inst(st,[&]{if(o)one();else stock();});CK(cudaGraphLaunch(g,st));CK(cudaStreamSynchronize(st));cudaGraphExecDestroy(g);
            for(int k=0;k<NG;k++){auto v=host(sendb[k],pb);out.insert(out.end(),v.begin(),v.end());}return out;};
        const auto ref=take(false);
        size_t nd=0,live=0;{const float* f=(const float*)ref.data();   // m of each (group, row, head): -inf = no key of rank 0
            for(int k=0;k<NG;k++)for(int i=0;i<M*H;i++)live+=std::isfinite(f[k*pb/4+(size_t)M*H*512+i]);}
        for(int it=0;it<CHK;it++){const auto x=take(true);for(size_t i=0;i<ref.size();i++)nd+=ref[i]!=x[i];
            const auto y=take(false);for(size_t i=0;i<ref.size();i++)nd+=ref[i]!=y[i];}
        const unsigned e=host(err,1)[0];
        printf("EXACT %s T=%d L=%d M=%d x%d | published partials %zu B, bytes differing ONE vs stock (and stock vs stock) %zu | heads with a key of "
               "rank 0: %zu / %d | err %u | %s\n",tag.c_str(),T,L,M,CHK,ref.size(),nd,live,NG*M*H,e,nd||e?"DIFFER":"IDENTICAL");
        if(nd||e){bad++;cudaFree(scr);cudaFree(obuf);continue;}
        // TIME: alternating, hot / cold
        float tflush=0;{cudaGraphExec_t g=inst(st,[&]{for(int k=0;k<K;k++)flushk();});std::vector<float> v;for(int t=0;t<TR;t++)v.push_back(run_us(g,st,REPS));
            tflush=med(v)/K;cudaGraphExecDestroy(g);}
        cudaGraphExec_t g[4]={inst(st,[&]{for(int k=0;k<K;k++)stock();}),inst(st,[&]{for(int k=0;k<K;k++){flushk();stock();}}),
                             inst(st,[&]{for(int k=0;k<K;k++)one();}),inst(st,[&]{for(int k=0;k<K;k++){flushk();one();}})};
        std::vector<float> v[4];
        for(int t=0;t<TR;t++)for(int q=0;q<4;q++)v[q].push_back(run_us(g[q],st,REPS)/K-(q&1?tflush:0));
        for(auto x:g)cudaGraphExecDestroy(x);
        float o[4];for(int q=0;q<4;q++)o[q]=med(v[q]);
        const float sh=o[0]-o[2],sc=o[1]-o[3];
        printf("TIME %s T=%d L=%d M=%d | stock (3 x vget+group+put) hot %.1f cold %.1f us | ONE (vget_heads+groups+3 put) hot %.1f cold %.1f us | "
               "saved per layer hot %.1f cold %.1f us, per step (78 layers) hot %.2f cold %.2f ms | flush %.1f us | gate 20 us/layer cold: %s\n",
               tag.c_str(),T,L,M,o[0],o[1],o[2],o[3],sh,sc,sh*78/1000.,sc*78/1000.,tflush,sc>=20?"PASS":"below (reject)");
        cudaFree(scr);cudaFree(obuf);
    }
    CK(cudaGetLastError());
    return bad?3:0;
}
