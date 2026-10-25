// attnk: exact decode attention kernel arms against the stock decode attention chain of one layer, as the engine replays it
// (g6_attn_replay_run captured in a CUDA graph). Harness = an earlier attention-chain bench without the MK
// chain kernel. Flags under test (each a host-side static the bench assigns): G6_ATTN_SC2 / _SC2_TR (scores_sc), G6_ATTN_CTX2
// (context_parts_heads PB / context_pk), G6_ATTN_CTX3 / _CTX3_HG (context_pk3), G6_ATTN_SEL2 (rank_selected), G6_IDX_GRP
// (index_scores_g); round 3: G6_ATTN_QT (absorb_m2 with the lane-transposed q for scores_sc), G6_ATTN_FRONT (attn_front).
// Layers: a full-index layer (its own indexer + select) then a shared layer reusing its indices, M rows at live context L in
// capture bucket T, S2 batched cell with per-row tables (s2=1, as served). One call's GPU time inside a graph of K calls,
// median of TR alternating trials; "cold" = a 96 MB L2 flush before every call, flush subtracted (served case: the experts
// stream ~90 MB per rank between two layers' attention); "hot" = no flush.
// ATTNK_SESS=2: the M rows are two sessions (S2 batched step as served at c > 1): rows 0 .. M-M/2-1 at live context L on
// the identity page table, rows M-M/2 .. M-1 at live context L/2 on the reversed table (other physical pages, unequal
// lengths, a group boundary inside the step); each session on its own physical pages (pool of 2x the bucket's pages).
// Fixtures for the edge paths (exactness; timing as any cell): ATTNK_TIES=1 makes the indexer scores repeat every 7
// positions (exact ties: the selector's id tie-break), ATTNK_QSCALE=<s> scales the query (s = 400: most probabilities
// underflow to exactly 0, so the p == 0 skips and context_pk3's mixed batches run).
//   REF     the stock chain (every flag 0): outputs of both layers, indices, the indexer's scores and ids, the appended KV
//           rows and index pages, per layer the raw scores (chain prefix through the scores node) and the context parts
//           (prefix through the context node), the shared layer's normalized probabilities and merged context. Captured
//           once, before any arm is touched.
//   BASE    the serving kernels of rounds 1 + 2 (default SC2=4,TR=2,CTX2=4,CTX3=2,SEL2=16,IDX=16; ATTNK_BASE overrides,
//           "stock" = all 0): every arm's timing is against BASE, so a gain is on top of round 2. BASE itself is the first
//           arm (against stock).
//   CHAIN   per-kernel table of BASE (what bit 1) and of stock (bit 4): in-chain cost of every node = T(prefix k) -
//           T(prefix k-1), hot / cold / dead, DRAM floor.
//   ARM     one flag setting (BASE + changes):
//           EXACT  bytes differing vs REF over chk poisoned launches (raw scores, parts, probs, context, outputs, indices,
//                  index scores / ids, kv).
//           NODE   per layer, every node whose kernel or grid differs from BASE: its in-chain cost (prefix delta) BASE -> arm,
//                  hot / cold.
//           TIME   whole call BASE vs arm alternating, hot / cold, per layer, and the per-step projection over the main
//                  verify pass: 21 full-index + 57 shared layers (cold), ms/step.
//   Decision (written before running): adopt an arm if it is IDENTICAL and saves >= 0.3 ms/step cold over BASE at T=2048
//   L=1000 M=3, or >= 1 ms/step at M=12, or >= 1 ms/step at T=32768 (index / select arms: the long-context levers). Kill
//   otherwise, or on any byte difference.
// build (from engine/attn/tests), with the engine's flags (scripts/build.sh K1H2_FLAGS, -DG6_NR=4 -DDEC_S2 -DPF_F12;
// TP=6: -DG6_NR=6 -DDEC_S2, run with ATTNK_H=11 and ATTNK_H=10, the two head counts of a TP=6 rank):
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $FLAGS -o attnk attnk.cu ../bind.cu -lcublas
// run: attnk T L M [what=3] [s2=1] [reps=10] [K=16] [TR=7] [chk=5] [bw=250]
//   what: 1 CHAIN of BASE, 2 the arms, 4 CHAIN of stock. Arms: env ATTNK_ARMS="arm arm ..." with an arm = comma-separated
//   KEY=VALUE changes over BASE (keys SC2 TR CTX2 CTX3 HG SEL2 IDX QT FRONT; "base" = BASE alone, compared
//   against stock), default below. ATTNK_BASE="SC2=4,TR=2,CTX2=4,CTX3=2,SEL2=16,IDX=16" (the same syntax over stock).
//   sweep: for M in 1 3 8 12; do ./attnk 2048 1000 $M; ./attnk 8192 8000 $M; ./attnk 32768 32000 $M; done
// Exit 3 on any byte difference of an exact arm, 1 on a CUDA / validation error.
//
// Round 10, E1 arms (deterministic fp-order changes; keys SP and POS; ATTNK_E1=1 picks the default E1 arm list below):
//   SP=64 / 128   G6_ATTN_CTX3_SP: context_pk3 over 64 / 128-slot partitions, merge_parts over 32 / 16 partials (run_body)
//   POS=1         G6_ATTN_POSIDX: the full-index layer at capacity T <= 2048 takes the positional index (no indexer)
//   An E1 arm is not compared byte for byte against stock; instead (E1 lines):
//     DET    every byte of CHK repeated poisoned calls (a fresh graph each) identical to the arm's first call: deterministic
//     SAME   what must not change: the selected SET per row (full layer, as sorted ids), the raw scores BY KEY (each key's
//            score wherever its slot is: scores_sc's per-(key, head) arithmetic does not depend on the slot), the appended
//            KV / index rows, the error words. SP alone: indices, scores and probabilities byte-compared too
//     ERR    vs stock: max abs and max rel error of the attention outputs (fp16, both layers) and of the shared layer's
//            merged context (fp32): max |d|, max |d| / max |ref|, max |d| / |ref| over |ref| >= 1e-3 max |ref|, elements
//            differing; and of the normalized probabilities by key (POS)
//   An E1 arm the dispatch does not take (POS at T > 2048: those buckets run the indexer) is still byte-compared against
//   stock and must be IDENTICAL ("inert"): the bucket boundary. Then NODE / TIME as for an exact arm.
//   ATTNK_NANKEY=<pos>: the index key at <pos> (session 0) gets a NaN scale. Stock drops that key from every row that
//   sees it and raises error 4; POS keeps it (no indexer score): reported as the expected semantic difference, and the
//   SET / score checks then ignore that key.
//   E1 sweep on one GB10: attnk_r10a.sh next to this file.
#include "../attn.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cxxabi.h>
#define CK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)
#define CKI(x) do{int e_=(x);if(e_){fprintf(stderr,"%s:%d status %d (%s)\n",__FILE__,__LINE__,e_,cudaGetErrorString((cudaError_t)e_));exit(1);}}while(0)

__device__ uint32_t hsh(uint64_t x){x^=x>>33;x*=0xff51afd7ed558ccdull;x^=x>>33;x*=0xc4ceb9fe1a85ec53ull;x^=x>>33;return (uint32_t)x;}
__device__ float unif(uint64_t seed,uint64_t i){return (float)(hsh(seed*0x9e3779b97f4a7c15ull+i)>>8)*(2.f/16777216.f)-1.f;}   // [-1, 1)
__global__ void fill_bf16(__nv_bfloat16* p,size_t n,uint64_t seed,float s,float c){
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)p[i]=__float2bfloat16(c+s*unif(seed,i));}
__global__ void fill_half(__half* p,size_t n,uint64_t seed,float s){
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)p[i]=__float2half(s*unif(seed,i));}
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
// ties: the key and scale of row r depend on r % 7 only, so the indexer scores of a row repeat every 7 positions (exact ties,
// broken by id in the selector)
__global__ void fill_idx(uint8_t* c,size_t pages,int ps,uint64_t seed,int ties){
    for(size_t p=blockIdx.x;p<pages;p+=gridDim.x){uint8_t* b=c+p*ps*g6kv::IDX_ROW;
        for(int i=threadIdx.x;i<ps*128;i+=blockDim.x)b[i]=fin8(ties?hsh(seed+((p*ps+i/128)%7)*1000+i%128):hsh(seed+p*100000+i));
        for(int i=threadIdx.x;i<ps;i+=blockDim.x)((float*)(b+ps*128))[i]=ldexpf(1.f,-6-(int)(hsh(seed+3+(ties?(p*ps+i)%7:p*64+i))%3));}
}
__global__ void l2_flush(const float4* p,size_t n,float* sink){
    float acc=0;for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i];acc+=v.x+v.y+v.z+v.w;}
    if(acc==1234.5f)*sink=acc;   // never (the buffer is zero); keeps the loads
}

template<class T> static T* dalloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));CK(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T> static std::vector<T> host(const T* d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static size_t ndiff(const void* x,const void* y,size_t bytes){size_t k=0;for(size_t i=0;i<bytes;i++)k+=((const uint8_t*)x)[i]!=((const uint8_t*)y)[i];return k;}
template<class T> static size_t vdiff(const std::vector<T>& x,const std::vector<T>& y){return x.size()!=y.size()?~size_t(0):ndiff(x.data(),y.data(),x.size()*sizeof(T));}
static float med(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
// E1: x against the reference y: max |d|, max |d| / max |y|, max |d| / |y| over |y| >= 1e-3 max |y|, elements differing
// (bitwise), nonfinite elements of x
struct Err{double abs=0,nrel=0,erel=0;size_t ndiff=0,nonfin=0,n=0;};
static float tof(float v){return v;}
static float tof(__half v){return __half2float(v);}
template<class T> static Err err_of(const std::vector<T>& x,const std::vector<T>& y){
    Err e;e.n=x.size();if(x.size()!=y.size()){e.ndiff=~size_t(0);return e;}
    double ym=0;for(const T& v:y){const double a=fabs((double)tof(v));if(std::isfinite(a))ym=std::max(ym,a);}
    for(size_t i=0;i<x.size();i++){const double a=tof(x[i]),b=tof(y[i]);
        e.ndiff+=memcmp(&x[i],&y[i],sizeof(T))!=0;
        if(!std::isfinite(a)){e.nonfin++;continue;}
        const double d=fabs(a-b);if(!std::isfinite(d))continue;
        e.abs=std::max(e.abs,d);if(fabs(b)>=1e-3*ym&&b!=0)e.erel=std::max(e.erel,d/fabs(b));}
    e.nrel=ym>0?e.abs/ym:0;return e;}
static std::string err_str(const Err& e){char b[192];snprintf(b,sizeof b,"max abs %.3e, rel(max|ref|) %.3e, rel(elem) %.3e, %zu of %zu differ, %zu nonfinite",
    e.abs,e.nrel,e.erel,e.ndiff,e.n,e.nonfin);return b;}

// ---- graphs
template<class F> static cudaGraph_t capture(cudaStream_t st,F body){
    cudaGraph_t g;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));body();CK(cudaStreamEndCapture(st,&g));return g;}
template<class F> static cudaGraphExec_t inst(cudaStream_t st,F body){
    cudaGraph_t g=capture(st,body);cudaGraphExec_t x;CK(cudaGraphInstantiate(&x,g,0));CK(cudaGraphDestroy(g));return x;}
static float run_us(cudaGraphExec_t x,cudaStream_t st,int reps){   // us per graph launch
    for(int i=0;i<2;i++)CK(cudaGraphLaunch(x,st));
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,st));
    for(int i=0;i<reps;i++)CK(cudaGraphLaunch(x,st));
    CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
    cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000.f/reps;
}
// a captured call's kernel nodes in launch order (the capture is one stream: a chain)
struct Node{cudaKernelNodeParams p;std::string name;};
static std::string short_name(const void* f){
    const char* raw=nullptr;if(cudaFuncGetName(&raw,f)!=cudaSuccess||!raw)return "?";
    int st=0;char* dm=abi::__cxa_demangle(raw,nullptr,nullptr,&st);std::string s=st==0&&dm?dm:raw;free(dm);
    if(!s.compare(0,5,"void "))s=s.substr(5);
    const size_t par=s.find('(');if(par!=std::string::npos)s=s.substr(0,par);
    for(const char* cut:{"ga::small_decode::","ga::kv4::","ga::"}){size_t i;while((i=s.find(cut))!=std::string::npos)s.erase(i,strlen(cut));}
    {size_t i;while((i=s.find("ReplayInput"))!=std::string::npos)s.erase(i,11);
     while((i=s.find(", , "))!=std::string::npos)s.replace(i,4,", ");
     for(const char* f:{", >",",>","<, ","<,","<>"}){const char* r=f[0]==','?">":f[1]=='>'?"":"<";while((i=s.find(f))!=std::string::npos)s.replace(i,strlen(f),r);}}
    return s;
}
static std::vector<Node> chain_of(cudaGraph_t g){
    size_t nn=0;CK(cudaGraphGetNodes(g,nullptr,&nn));std::vector<cudaGraphNode_t> nodes(nn);CK(cudaGraphGetNodes(g,nodes.data(),&nn));
    size_t ne=0;CK(cudaGraphGetEdges(g,nullptr,nullptr,nullptr,&ne));
    std::vector<cudaGraphNode_t> from(ne),to(ne);std::vector<cudaGraphEdgeData> ed(ne);
    if(ne)CK(cudaGraphGetEdges(g,from.data(),to.data(),ed.data(),&ne));
    std::vector<int> indeg(nn,0),order;std::vector<bool> done(nn,false);
    auto idx=[&](cudaGraphNode_t x){for(size_t i=0;i<nn;i++)if(nodes[i]==x)return (int)i;return -1;};
    for(size_t e=0;e<ne;e++)indeg[idx(to[e])]++;
    for(size_t step=0;step<nn;step++){int pick=-1;for(size_t i=0;i<nn;i++)if(!done[i]&&!indeg[i]){pick=(int)i;break;}
        if(pick<0){fprintf(stderr,"graph is not a DAG?\n");exit(1);}
        done[pick]=true;order.push_back(pick);for(size_t e=0;e<ne;e++)if(from[e]==nodes[pick])indeg[idx(to[e])]--;}
    std::vector<Node> out;
    for(int i:order){cudaGraphNodeType t;CK(cudaGraphNodeGetType(nodes[i],&t));
        if(t!=cudaGraphNodeTypeKernel){fprintf(stderr,"node %d is not a kernel (type %d): not timed\n",i,(int)t);continue;}
        Node n{};CK(cudaGraphKernelNodeGetParams(nodes[i],&n.p));n.name=short_name(n.p.func);out.push_back(n);}
    return out;
}
static void relaunch(const Node& n,cudaStream_t st){
    cudaLaunchConfig_t c{};c.gridDim=n.p.gridDim;c.blockDim=n.p.blockDim;c.dynamicSmemBytes=n.p.sharedMemBytes;c.stream=st;
    CK(cudaLaunchKernelExC(&c,n.p.func,n.p.kernelParams));
}
static bool same_node(const Node& x,const Node& y){
    return x.p.func==y.p.func&&x.p.gridDim.x==y.p.gridDim.x&&x.p.gridDim.y==y.p.gridDim.y&&x.p.gridDim.z==y.p.gridDim.z&&x.p.blockDim.x==y.p.blockDim.x;}
// arms that differ only in a kernel argument (index_scores_g's group size G6_IDX_GRP): the node counts as changed
static bool arg_differs(const Node& x,int idx_a,int idx_b){return idx_a!=idx_b&&x.name.find("index_scores_g")!=std::string::npos;}
// the attention scores node (not the indexer's) / the context node of a chain; -1 if absent
static int node_of(const std::vector<Node>& c,const char* what){
    for(int i=0;i<(int)c.size();i++){const std::string& s=c[i].name;
        if(!strcmp(what,"scores")&&s.find("scores")!=std::string::npos&&s.find("index_scores")==std::string::npos)return i;
        if(!strcmp(what,"context")&&s.find("context")!=std::string::npos)return i;}
    return -1;
}
// DRAM floor of a node: the compulsory bytes it is the first to read
static double floor_bytes(const std::string& s,int H,int L){
    const double kv=(double)std::min(L,ga::TOP)*ga::KV_BYTES;
    if(s.find("absorb")!=std::string::npos)return (double)H*512*192*2;
    if(s.find("value_project")!=std::string::npos)return (double)H*256*512*2;
    if(s.find("index_scores")!=std::string::npos)return (double)L*(128+4);
    if(s.find("scores")!=std::string::npos)return kv;
    return 0;
}
struct Arm{int sc,tr,ctx2,ctx3,hg,sel,idx,qt,front,sp,pos;bool vs_stock;bool e1()const{return sp||pos;}};
static void set_arm(const Arm& x){ga::attn_sc2()=x.sc;ga::attn_sc2_tr()=x.tr;ga::attn_ctx2()=x.ctx2;ga::attn_ctx3()=x.ctx3;ga::attn_ctx3_hg()=x.hg;
    ga::attn_sel2()=x.sel;ga::attn_idx_grp()=x.idx;ga::attn_qt()=x.qt;ga::attn_front_on()=x.front;ga::attn_ctx3_sp()=x.sp;ga::attn_posidx()=x.pos;}
static const Arm STOCK{0,0,0,0,0,0,0,0,0,0,0,false};
static std::string tag_of(const Arm& x){char b[192];snprintf(b,sizeof b,"SC2=%d TR=%d CTX2=%d CTX3=%d HG=%d SEL2=%d IDX=%d QT=%d FRONT=%d%s",
    x.sc,x.tr,x.ctx2,x.ctx3,x.hg,x.sel,x.idx,x.qt,x.front,x.e1()?(" SP="+std::to_string(x.sp)+" POS="+std::to_string(x.pos)).c_str():"");return b;}
// "KEY=V,KEY=V" over `on`; false on an unknown key
static bool parse_arm(const std::string& s,Arm& x){
    if(s=="base"||s=="stock")return true;
    size_t i=0;while(i<s.size()){size_t e=s.find(',',i);if(e==std::string::npos)e=s.size();const std::string kv=s.substr(i,e-i);i=e+1;
        const size_t q=kv.find('=');if(q==std::string::npos)return false;const std::string k=kv.substr(0,q);const int v=atoi(kv.c_str()+q+1);
        if(k=="SC2")x.sc=v;else if(k=="TR")x.tr=v;else if(k=="CTX2")x.ctx2=v;else if(k=="CTX3")x.ctx3=v;else if(k=="HG")x.hg=v;
        else if(k=="SEL2")x.sel=v;else if(k=="IDX")x.idx=v;else if(k=="QT")x.qt=v;else if(k=="FRONT")x.front=v;
        else if(k=="SP")x.sp=v;else if(k=="POS")x.pos=v;else return false;}
    return true;
}

int main(int argc,char** argv){
    if(argc<4){fprintf(stderr,"usage: attnk T L M [what=3] [s2=1] [reps=10] [K=16] [TR=7] [chk=5] [bw=250]\n");return 2;}
    const int T=atoi(argv[1]),L=atoi(argv[2]),M=atoi(argv[3]),WHAT=argc>4?atoi(argv[4]):3,S2=argc>5?atoi(argv[5]):1,
        REPS=argc>6?atoi(argv[6]):10,K=argc>7?atoi(argv[7]):16,TR=argc>8?atoi(argv[8]):7,CHK=argc>9?atoi(argv[9]):5;
    const double BW=argc>10?atof(argv[10]):250.;
    if(L>T||M<1||M>G6_MX||M>L){fprintf(stderr,"bad sizes (M <= %d, M <= L <= T)\n",G6_MX);return 2;}
    if(CHK<1||TR<1||K<1||REPS<1){fprintf(stderr,"chk, TR, K, reps >= 1\n");return 2;}
#ifdef DEC_S2
#else
    if(S2){fprintf(stderr,"s2 needs -DDEC_S2\n");return 2;}
#endif
#if G6_KV4
    fprintf(stderr,"attnk: FP8 KV build only (the flags are inert under G6_KV4)\n");return 2;
#endif
    const int NSESS=getenv("ATTNK_SESS")&&atoi(getenv("ATTNK_SESS"))==2&&S2&&M>=2?2:1;
    const int TIES=getenv("ATTNK_TIES")&&atoi(getenv("ATTNK_TIES"))?1:0;
    const float QS=getenv("ATTNK_QSCALE")?(float)atof(getenv("ATTNK_QSCALE")):1.f;
    const int NANKEY=getenv("ATTNK_NANKEY")?atoi(getenv("ATTNK_NANKEY")):-1;
    // BASE and the arms
    Arm base{4,2,4,2,0,16,16,0,0,0,0,false};   // the serving flags (rounds 1 + 2)
    if(const char* e=getenv("ATTNK_BASE")){base=STOCK;if(!parse_arm(e,base)){fprintf(stderr,"ATTNK_BASE: KEY=V,...\n");return 2;}}
    // round 3: QT (S = 4 and S = 2), FRONT, then both (the serving arm)
    std::vector<std::string> specs={"base","QT=1","QT=1,SC2=2","FRONT=1","QT=1,FRONT=1"};
    if(getenv("ATTNK_E1")&&atoi(getenv("ATTNK_E1")))specs={"base","SP=64","SP=128","POS=1","POS=1,FRONT=0","SP=64,POS=1"};   // round 10
    if(const char* e=getenv("ATTNK_ARMS")){specs.clear();std::string s=e;size_t i=0;
        while(i<s.size()){while(i<s.size()&&s[i]==' ')i++;size_t e2=s.find(' ',i);if(e2==std::string::npos)e2=s.size();if(e2>i)specs.push_back(s.substr(i,e2-i));i=e2;}}
    std::vector<Arm> arms;
    for(auto& sp:specs){Arm x=base;x.vs_stock=sp=="base";if(!parse_arm(sp,x)){fprintf(stderr,"ATTNK_ARMS: bad arm %s\n",sp.c_str());return 2;}arms.push_back(x);}
    // ATTNK_H=<heads>: the rank's head count, g6nr::HMIN or HMAX (default HMAX). A -DG6_NR=6 build runs 11 or 10 heads, the
    // partial head groups of the SC2 / QT arms (G6_ATTN_SC2_PART); TP=4 builds have 16 either way
    const int H=getenv("ATTNK_H")?atoi(getenv("ATTNK_H")):g6nr::HMAX,pages=(T+63)/64,pool=NSESS*pages;   // session s: physical pages [s*pages, (s+1)*pages)
    if(H!=g6nr::HMIN&&H!=g6nr::HMAX){fprintf(stderr,"ATTNK_H=%d: %d or %d heads at G6_NR=%d\n",H,g6nr::HMIN,g6nr::HMAX,g6nr::NR);return 2;}
    cudaStream_t st;CK(cudaStreamCreate(&st));
    {int dev=0,sms=0,clk=0,l2=0;CK(cudaGetDevice(&dev));cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
     cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,dev);cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,dev);
     printf("GPU %d SMs, attr clock %d MHz, L2 %.1f MB | T=%d L=%d M=%d H=%d s2=%d sessions=%d ties=%d qscale=%g K=%d reps=%d TR=%d chk=%d bw=%.0f GB/s | BASE %s\n",
        sms,clk/1000,l2/1048576.,T,L,M,H,S2,NSESS,TIES,QS,K,REPS,TR,CHK,BW,tag_of(base).c_str());}
    CKI(g6a_rope_init(T,st));   // as a served boot: the decode RoPE table (absorb / append skip the FP64 sincos)
    // weights, per-layer caches (layer 0 full index, layer 1 shared), step inputs
    __nv_bfloat16 *uk=dalloc<__nv_bfloat16>((size_t)H*512*192),*uv=dalloc<__nv_bfloat16>((size_t)H*256*512),
        *kvn=dalloc<__nv_bfloat16>(512),*inw=dalloc<__nv_bfloat16>(128),*inb=dalloc<__nv_bfloat16>(128);
    fill_bf16<<<512,256,0,st>>>(uk,(size_t)H*512*192,1,0.05f,0);fill_bf16<<<512,256,0,st>>>(uv,(size_t)H*256*512,2,0.05f,0);
    fill_bf16<<<2,256,0,st>>>(kvn,512,3,0.2f,1);fill_bf16<<<1,128,0,st>>>(inw,128,4,0.2f,1);fill_bf16<<<1,128,0,st>>>(inb,128,5,0.1f,0);
    uint8_t *mla[2],*ixc=dalloc<uint8_t>((size_t)pool*64*g6kv::IDX_ROW);
    for(int l=0;l<2;l++){mla[l]=dalloc<uint8_t>((size_t)pool*64*g6kv::ROW);fill_kv<<<1024,256,0,st>>>(mla[l],(size_t)pool*64,100+l);}
    fill_idx<<<512,256,0,st>>>(ixc,pool,64,200,TIES);
    if(NANKEY>=0&&NANKEY<L){const float nan=NAN;   // session 0's identity table: page NANKEY / 64; a page's scales follow its keys
        CK(cudaMemcpy(ixc+(size_t)(NANKEY/64)*64*g6kv::IDX_ROW+64*128+(NANKEY%64)*4,&nan,4,cudaMemcpyHostToDevice));}
    __half *qb=dalloc<__half>((size_t)M*H*256),*kva=dalloc<__half>((size_t)M*576),*iq=dalloc<__half>((size_t)M*32*128),
        *ik=dalloc<__half>((size_t)M*128),*iw=dalloc<__half>((size_t)M*32);
    fill_half<<<64,256,0,st>>>(qb,(size_t)M*H*256,6,QS);fill_half<<<8,256,0,st>>>(kva,(size_t)M*576,7,1.f);
    fill_half<<<64,256,0,st>>>(iq,(size_t)M*32*128,8,1.f);fill_half<<<1,256,0,st>>>(ik,(size_t)M*128,9,1.f);fill_half<<<1,256,0,st>>>(iw,(size_t)M*32,10,0.1f);
    // sessions: rows [0, M0) at L on the identity table (pages [0, pages)); rows [M0, M) at L1 = L/2 on the reversed table
    // of pages [pages, 2 pages)
    const int M1=NSESS==2?M/2:0,M0=M-M1,L1=std::max(L/2,M1);
    std::vector<int> pos(M),table(pages),tableB(pages),rsess(M);
    for(int i=0;i<pages;i++){table[i]=i;tableB[i]=pool-1-i;}
    for(int i=0;i<M;i++){rsess[i]=i<M0?0:1;pos[i]=i<M0?L-M0+i:L1-M1+(i-M0);}
    int *dpos=dalloc<int>(M),*dtab=dalloc<int>(pages),*dtabB=dalloc<int>(pages),*ind[2]={dalloc<int>((size_t)M*ga::TOP),dalloc<int>((size_t)M*ga::TOP)};
    CK(cudaMemcpy(dpos,pos.data(),M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dtab,table.data(),pages*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtabB,tableB.data(),pages*4,cudaMemcpyHostToDevice));
    // physical row of each step row's appended KV / index key
    std::vector<size_t> prow(M);for(int i=0;i<M;i++)prow[i]=(size_t)(rsess[i]?tableB:table)[pos[i]/64]*64+pos[i]%64;
    __half* out[2]={dalloc<__half>((size_t)M*H*256),dalloc<__half>((size_t)M*H*256)};
    unsigned* err[2]={dalloc<unsigned>(1),dalloc<unsigned>(1)};
    G6Attn a[2]{};
    for(int l=0;l<2;l++){G6Attn& x=a[l];x.rows=M;x.heads=H;x.context=T;x.page_size=64;x.page_count=pool;x.table_size=pages;x.query_tile=std::max(M,32);
        x.q_b=qb;x.kv_a=kva;x.kv_norm=kvn;x.uk=uk;x.uv=uv;x.positions=dpos;x.block_table=dtab;x.mla_cache=mla[l];
        x.indices=ind[l];x.output=out[l];x.error=err[l];
#ifdef DEC_S2
        x.s2_batched=S2;
#endif
    }
    a[0].index_q=iq;a[0].index_k=ik;a[0].index_weight=iw;a[0].index_norm=inw;a[0].index_bias=inb;a[0].index_cache=ixc;
    a[1].reuse_indices=ind[0];a[1].reuse_rows=M;
    // cells: live (valid) and dead (context 0: validate_replay marks it invalid, every kernel returns after the cell read)
    G6AttnStepCell *dcell[2],*dead[2];
#ifdef DEC_S2
    G6AttnRow* drows=nullptr;
    if(S2){std::vector<G6AttnRow> rr(M);
        for(int i=0;i<M;i++){G6AttnRow& x=rr[i];x.table=rsess[i]?dtabB:dtab;x.context=rsess[i]?L1:L;x.page_count=pool;x.table_size=pages;x.ibase=0;x.sess=rsess[i];}
        drows=dalloc<G6AttnRow>(M);CK(cudaMemcpy(drows,rr.data(),M*sizeof(G6AttnRow),cudaMemcpyHostToDevice));}
#endif
    for(int l=0;l<2;l++){G6AttnStepCell c{};c.context=L;c.page_count=pool;c.table_size=pages;c.reuse_rows=l?M:0;
#ifdef DEC_S2
        c.rows=drows;
#endif
        CK(cudaMalloc(&dcell[l],sizeof c));CK(cudaMemcpy(dcell[l],&c,sizeof c,cudaMemcpyHostToDevice));
        c.context=0;CK(cudaMalloc(&dead[l],sizeof c));CK(cudaMemcpy(dead[l],&c,sizeof c,cudaMemcpyHostToDevice));}
    ga::attn_ctx3_sp()=64;   // size the scratch for the largest layout (SP=64: 32 partials), so every arm's layout fits it
    const size_t sb=std::max(g6a_scratch_bytes(&a[0]),g6a_scratch_bytes(&a[1]));
    ga::attn_ctx3_sp()=0;
    void* scr=dalloc<uint8_t>(sb);
    G6AttnReplay rp[2]={{a[0],dcell[0],scr,sb},{a[1],dcell[1],scr,sb}},rd[2]={{a[0],dead[0],scr,sb},{a[1],dead[1],scr,sb}};
    const size_t fl=(size_t)96<<20;float4* fb=dalloc<float4>(fl/16);float* sink=dalloc<float>(1);
    auto flushk=[&]{l2_flush<<<48*6,256,0,st>>>(fb,fl/16,sink);};
    CK(cudaStreamSynchronize(st));
    const char* kind[2]={"full-index","shared"};
    set_arm(STOCK);

    // ---- reference (stock): outputs, the shared layer's scratch, the indexer's scratch, per layer raw scores and parts
    const size_t no=(size_t)M*H*256,ni=(size_t)M*ga::TOP,nc=(size_t)M*H*512,np=(size_t)M*H*ga::TOP,npa=(size_t)M*H*ga::PART*512,nsi=(size_t)M*T;
    const size_t ipb=(size_t)64*g6kv::IDX_ROW;   // one index-cache page
    struct Snap{std::vector<__half> o0,o1;std::vector<int> i0,ii;std::vector<float> cx,pr,pa,si,sc[2],pl[2];std::vector<uint8_t> kv,ix;unsigned e0,e1,v0,v1;};
    auto call=[&](int l){CKI(g6a_replay_run(&rp[l],st));};
    auto both=[&]{call(0);call(1);};
    auto poison_w=[&](const ga::Work& W){CK(cudaMemsetAsync(W.context,0xab,nc*4,st));CK(cudaMemsetAsync(W.prob,0xab,np*4,st));CK(cudaMemsetAsync(W.parts,0xab,npa*4,st));
        CK(cudaMemsetAsync(W.si,0xab,nsi*4,st));CK(cudaMemsetAsync(W.ii,0xab,nsi*4,st));};
    auto poison=[&](const ga::Work& W){for(int l=0;l<2;l++){CK(cudaMemsetAsync(out[l],0xff,no*2,st));CK(cudaMemsetAsync(ind[l],0x5a,ni*4,st));}poison_w(W);};
    // the full call into s (both layers), then per layer the chain prefixes through the scores and the context node
    auto take=[&](Snap& s){
        const ga::Work W=ga::workspace(a[1],scr);   // the layers share the scratch (the same layout: rows, heads, context)
        {cudaGraphExec_t g=inst(st,both);poison(W);CK(cudaGraphLaunch(g,st));CK(cudaStreamSynchronize(st));cudaGraphExecDestroy(g);}
        s.o0=host(out[0],no);s.o1=host(out[1],no);s.i0=host(ind[0],ni);s.cx=host(W.context,nc);s.pr=host(W.prob,np);
        s.si=host(W.si,nsi);s.ii=host(W.ii,nsi);   // written by the full layer's indexer only (the shared layer reuses)
        s.pa=host(W.parts,npa);
        s.kv.clear();s.ix.clear();
        for(int i=0;i<M;i++){for(int l=0;l<2;l++){auto v=host(mla[l]+prow[i]*g6kv::ROW,(size_t)g6kv::ROW);s.kv.insert(s.kv.end(),v.begin(),v.end());}
            auto v=host(ixc+(prow[i]/64)*ipb,ipb);s.ix.insert(s.ix.end(),v.begin(),v.end());}
        s.e0=host(err[0],1)[0];s.e1=host(err[1],1)[0];s.v0=host(dcell[0],1)[0].valid;s.v1=host(dcell[1],1)[0].valid;
        for(int l=0;l<2;l++){cudaGraph_t g=capture(st,[&]{call(l);});auto c=chain_of(g);const int ks=node_of(c,"scores"),kc=node_of(c,"context");
            if(ks<0||kc<0){fprintf(stderr,"no scores / context node\n");exit(1);}
            // (the shared layer reads the full layer's indices: they are in place from the call above; scratch poisoned after)
            poison_w(W);for(int j=0;j<=ks;j++)relaunch(c[j],st);CK(cudaStreamSynchronize(st));s.sc[l]=host(W.prob,np);
            poison_w(W);for(int j=0;j<=kc;j++)relaunch(c[j],st);CK(cudaStreamSynchronize(st));s.pl[l]=host(W.parts,npa);
            cudaGraphDestroy(g);}
        CK(cudaGetLastError());};
    Snap ref;take(ref);
    {int live=0;for(int v:ref.i0)live+=v>=0;size_t nan=0;for(auto h:ref.o0)nan+=!std::isfinite(__half2float(h));for(auto h:ref.o1)nan+=!std::isfinite(__half2float(h));
     printf("REF selected %d, nonfinite out %zu, err %u/%u valid %u/%u\n",live,nan,ref.e0,ref.e1,ref.v0,ref.v1);
     if((ref.e0&&!(NANKEY>=0&&ref.e0==4))||ref.e1||!ref.v0||!ref.v1)return 1;}   // NANKEY: the indexer's error 4 expected

    // ---- timing helpers: per-call us of `body` (cold: behind a flush, the flush subtracted)
    float tflush=0;{cudaGraphExec_t g=inst(st,[&]{for(int k=0;k<K;k++)flushk();});std::vector<float> v;for(int t=0;t<TR;t++)v.push_back(run_us(g,st,REPS));tflush=med(v)/K;cudaGraphExecDestroy(g);}
    struct Timed{cudaGraphExec_t x;bool cold;std::vector<float> v;};
    auto timed_graph=[&](auto body,bool cold){return Timed{inst(st,[&]{for(int k=0;k<K;k++){if(cold)flushk();body();}}),cold,{}};};
    auto measure=[&](std::vector<Timed>& g){for(int t=0;t<TR;t++)for(auto& x:g)x.v.push_back(run_us(x.x,st,REPS)/K-(x.cold?tflush:0));
        for(auto& x:g)cudaGraphExecDestroy(x.x);};
    auto value=[](const Timed& x){return med(x.v);};
    auto once=[&](auto body,bool cold){Timed tg=timed_graph(body,cold);const float u=run_us(tg.x,st,REPS)/K-(cold?tflush:0);cudaGraphExecDestroy(tg.x);return u;};
    printf("flush %.1f us per call (subtracted from cold)\n",tflush);

    // ---- CHAIN: per-kernel prefix costs, live and dead, totals (BASE: what bit 1, stock: bit 4)
    for(int which=0;which<2;which++){
        if(!(WHAT&(which?4:1)))continue;
        const Arm cfg=which?STOCK:base;set_arm(cfg);
        for(int l=0;l<2;l++){
            cudaGraph_t gl=capture(st,[&]{call(l);}),gd=capture(st,[&]{CKI(g6a_replay_run(&rd[l],st));});
            std::vector<Node> nl=chain_of(gl),nd=chain_of(gd);
            if(nl.size()!=nd.size()){fprintf(stderr,"live / dead chains differ in length\n");exit(1);}
            const int n=(int)nl.size();std::vector<Timed> g;
            for(int k=1;k<=n;k++){g.push_back(timed_graph([&]{for(int j=0;j<k;j++)relaunch(nl[j],st);},false));
                g.push_back(timed_graph([&]{for(int j=0;j<k;j++)relaunch(nl[j],st);},true));
                g.push_back(timed_graph([&]{for(int j=0;j<k;j++)relaunch(nd[j],st);},false));}
            g.push_back(timed_graph([&]{call(l);},false));g.push_back(timed_graph([&]{call(l);},true));
            measure(g);
            printf("CHAIN %s (%s) %s layer T=%d L=%d M=%d: %d kernel nodes\n",which?"stock":"BASE",tag_of(cfg).c_str(),kind[l],T,L,M,n);
            printf("  %-3s %-34s %13s %9s %9s %9s %8s\n","#","kernel","grid x block","hot","cold","dead","floor");
            float ph=0,pc=0,pd=0;double fsum=0;
            for(int k=0;k<n;k++){const float h=value(g[3*k]),c=value(g[3*k+1]),d=value(g[3*k+2]);
                const double fbb=floor_bytes(nl[k].name,H,L);fsum+=fbb;
                char gb[48];snprintf(gb,sizeof gb,"%ux%ux%u x %u",nl[k].p.gridDim.x,nl[k].p.gridDim.y,nl[k].p.gridDim.z,nl[k].p.blockDim.x);
                printf("  %-3d %-34.34s %13s %9.2f %9.2f %9.2f %8.2f\n",k,nl[k].name.c_str(),gb,h-ph,c-pc,d-pd,fbb/BW/1e3);
                ph=h;pc=c;pd=d;}
            printf("  total hot %.1f cold %.1f us | dead %.1f us | DRAM floor %.1f us\n",value(g[3*n]),value(g[3*n+1]),pd,fsum/BW/1e3);
            cudaGraphDestroy(gl);cudaGraphDestroy(gd);}
    }
    set_arm(STOCK);

    // ---- arms
    int bad=0;
    if(WHAT&2)for(const Arm& x:arms){
        const Arm ref_arm=x.vs_stock?STOCK:base;   // timing reference
        const std::string tag=tag_of(x),rtag=x.vs_stock?"stock":"BASE";
        // the arm must reach this call's dispatch: some node of some layer differs from its timing reference
        {bool differs=false;
         for(int l=0;l<2&&!differs;l++){set_arm(ref_arm);cudaGraph_t g0=capture(st,[&]{call(l);});auto c0=chain_of(g0);
            set_arm(x);cudaGraph_t g1=capture(st,[&]{call(l);});auto c1=chain_of(g1);
            differs=c0.size()!=c1.size();for(size_t i=0;!differs&&i<c0.size();i++)differs=!same_node(c0[i],c1[i])||arg_differs(c1[i],ref_arm.idx,x.idx);
            cudaGraphDestroy(g0);cudaGraphDestroy(g1);}
         if(!differs&&!x.e1()){printf("ARM %s not taken by this call's dispatch (same chain as %s)\n",tag.c_str(),rtag.c_str());continue;}
         if(!differs){   // an E1 arm this call's dispatch does not take (POS past 2048): must be byte-identical to stock
            set_arm(x);size_t nd=0;bool ok=true;
            for(int it=0;it<CHK;it++){Snap s;take(s);
                nd+=vdiff(ref.o0,s.o0)+vdiff(ref.o1,s.o1)+vdiff(ref.i0,s.i0)+vdiff(ref.pr,s.pr)+vdiff(ref.cx,s.cx)+vdiff(ref.kv,s.kv)+vdiff(ref.ix,s.ix)+
                    vdiff(ref.sc[0],s.sc[0])+vdiff(ref.sc[1],s.sc[1])+vdiff(ref.si,s.si)+vdiff(ref.ii,s.ii);ok&=s.e0==ref.e0&&!s.e1&&s.v0&&s.v1;}
            printf("E1 %s T=%d L=%d M=%d: not taken by this call's dispatch (same chain as %s): bytes differing vs stock %zu x%d, err/valid %s | %s\n",
                tag.c_str(),T,L,M,rtag.c_str(),nd,CHK,ok?"ok":"BAD",nd||!ok?"DIFFER":"IDENTICAL (inert)");
            if(nd||!ok)bad++;
            continue;}}
        set_arm(x);
        if(x.e1()){
            // DET: CHK calls against the arm's first
            std::vector<Snap> ss(CHK);for(int it=0;it<CHK;it++)take(ss[it]);
            size_t det=0;bool ok=true;
            for(int it=0;it<CHK;it++){const Snap& s=ss[it];const Snap& f=ss[0];
                det+=vdiff(f.o0,s.o0)+vdiff(f.o1,s.o1)+vdiff(f.i0,s.i0)+vdiff(f.pr,s.pr)+vdiff(f.pa,s.pa)+vdiff(f.cx,s.cx)+vdiff(f.kv,s.kv)+vdiff(f.ix,s.ix)+
                    vdiff(f.sc[0],s.sc[0])+vdiff(f.sc[1],s.sc[1])+vdiff(f.pl[0],s.pl[0])+vdiff(f.pl[1],s.pl[1])+vdiff(f.si,s.si)+vdiff(f.ii,s.ii);
                ok&=!s.e1&&s.v0&&s.v1&&(x.pos?s.e0==0:s.e0==ref.e0);}
            const Snap& s=ss[0];
            // SAME: the set per row (NANKEY ignored), the raw scores by key, the kv / index rows
            size_t setbad=0,nan_kept=0,nan_rows=0;
            for(int r=0;r<M;r++){std::vector<int> u,v;
                for(int j=0;j<ga::TOP;j++){const int p=ref.i0[(size_t)r*ga::TOP+j],q=s.i0[(size_t)r*ga::TOP+j];
                    if(p>=0&&p!=NANKEY)u.push_back(p);if(q>=0&&q!=NANKEY)v.push_back(q);if(NANKEY>=0&&q==NANKEY)nan_kept++;}
                if(NANKEY>=0&&rsess[r]==0&&NANKEY<=pos[r])nan_rows++;
                std::sort(u.begin(),u.end());std::sort(v.begin(),v.end());setbad+=u!=v;}
            // the score of (row, head, key): REF's slot j holds key ref.i0[r][j], the arm's slot k holds s.i0[r][k]
            size_t scbad=0,sccmp=0;double pmax=0;
            for(int l=0;l<2;l++)for(int r=0;r<M;r++){std::vector<int> at(T,-1);
                for(int k=0;k<ga::TOP;k++){const int q=s.i0[(size_t)r*ga::TOP+k];if(q>=0&&q<T)at[q]=k;}
                for(int j=0;j<ga::TOP;j++){const int p=ref.i0[(size_t)r*ga::TOP+j];if(p<0||p==NANKEY)continue;const int k=at[p];if(k<0){scbad++;continue;}
                    for(int h=0;h<H;h++){const size_t o=((size_t)r*H+h)*ga::TOP;sccmp++;
                        scbad+=memcmp(&ref.sc[l][o+j],&s.sc[l][o+k],4)!=0;
                        if(l==1)pmax=std::max(pmax,(double)fabsf(ref.pr[o+j]-s.pr[o+k]));}}}
            const size_t kvbad=vdiff(ref.kv,s.kv)+vdiff(ref.ix,s.ix);
            const bool same=!setbad&&!scbad&&!kvbad&&(x.pos||(!vdiff(ref.i0,s.i0)&&!vdiff(ref.sc[0],s.sc[0])&&!vdiff(ref.sc[1],s.sc[1])&&!vdiff(ref.pr,s.pr)));
            printf("E1 %s T=%d L=%d M=%d x%d | DET bytes differing across calls %zu, err/valid %s (err %u/%u) | SAME set rows differing %zu, "
                   "scores by key %zu of %zu differ, kv+index cache %zu%s | %s\n",tag.c_str(),T,L,M,CHK,det,ok?"ok":"BAD",s.e0,s.e1,setbad,scbad,sccmp,kvbad,
                   x.pos?"":" (SP: indices, scores, probs byte-compared too)",!det&&ok&&same?"DETERMINISTIC, SAME":"FAIL");
            printf("E1 %s ERR vs stock | out full: %s\n",tag.c_str(),err_str(err_of(s.o0,ref.o0)).c_str());
            printf("E1 %s ERR vs stock | out shared: %s\n",tag.c_str(),err_str(err_of(s.o1,ref.o1)).c_str());
            printf("E1 %s ERR vs stock | context shared: %s\n",tag.c_str(),err_str(err_of(s.cx,ref.cx)).c_str());
            if(x.pos)printf("E1 %s ERR vs stock | probs shared by key: max abs %.3e\n",tag.c_str(),pmax);
            if(NANKEY>=0)printf("E1 %s NANKEY %d: stock dropped it (err %u); the arm lists it in %zu rows (of %zu that see it), err %u%s\n",tag.c_str(),NANKEY,ref.e0,
                nan_kept,nan_rows,s.e0,x.pos?" (POS: expected, no indexer score)":"");
            if(det||!ok||!same){bad++;continue;}
        }else{
        // EXACT (against stock)
        size_t w[14]={0};bool ok=true;
        for(int it=0;it<CHK;it++){Snap s;take(s);
            size_t d[14]={vdiff(ref.o0,s.o0),vdiff(ref.o1,s.o1),vdiff(ref.i0,s.i0),vdiff(ref.pr,s.pr),vdiff(ref.pa,s.pa),vdiff(ref.cx,s.cx),
                vdiff(ref.kv,s.kv)+vdiff(ref.ix,s.ix),vdiff(ref.sc[0],s.sc[0]),vdiff(ref.sc[1],s.sc[1]),
                vdiff(ref.pl[0],s.pl[0]),vdiff(ref.pl[1],s.pl[1]),vdiff(ref.si,s.si),vdiff(ref.ii,s.ii),0};
            for(int q=0;q<13;q++)w[q]=std::max(w[q],d[q]);ok&=s.e0==ref.e0&&!s.e1&&s.v0&&s.v1;}
        bool exact=ok;for(int q=0;q<13;q++)exact&=!w[q];
        printf("EXACT %s T=%d L=%d M=%d x%d | bytes differing vs stock: index scores %zu ids %zu, indices %zu, scores full %zu shared %zu, "
               "parts full %zu shared %zu, probs %zu, parts(call) %zu, context %zu, out full %zu shared %zu, kv+index cache %zu | err/valid %s | %s\n",
               tag.c_str(),T,L,M,CHK,w[11],w[12],w[2],w[7],w[8],w[9],w[10],w[3],w[4],w[5],w[0],w[1],w[6],ok?"ok":"BAD",exact?"IDENTICAL":"DIFFER");
        if(!exact){bad++;continue;}
        }
        // NODE: per layer, the nodes that differ from the timing reference: prefix deltas, reference vs arm
        for(int l=0;l<2;l++){
            std::vector<Node> nn[2];cudaGraph_t gg[2];
            for(int arm=0;arm<2;arm++){set_arm(arm?x:ref_arm);gg[arm]=capture(st,[&]{call(l);});nn[arm]=chain_of(gg[arm]);}
            set_arm(x);
            if(nn[0].size()!=nn[1].size()){printf("NODE %s %s layer: chains differ in length (%zu vs %zu): not compared per node\n",tag.c_str(),kind[l],nn[0].size(),nn[1].size());}
            else for(size_t k=0;k<nn[0].size();k++){if(same_node(nn[0][k],nn[1][k])&&!arg_differs(nn[1][k],ref_arm.idx,x.idx))continue;
                float v[2][2][2];std::vector<float> acc[2][2][2];   // [arm][end k-1 / k][cold]
                for(int t=0;t<TR;t++)for(int arm=0;arm<2;arm++)for(int e=0;e<2;e++)for(int cold=0;cold<2;cold++){
                    const int end=(int)k-1+e;acc[arm][e][cold].push_back(once([&]{for(int j=0;j<=end;j++)relaunch(nn[arm][j],st);},cold));}
                for(int arm=0;arm<2;arm++)for(int e=0;e<2;e++)for(int cold=0;cold<2;cold++)v[arm][e][cold]=med(acc[arm][e][cold]);
                char gb[2][48];for(int arm=0;arm<2;arm++)snprintf(gb[arm],48,"%ux%ux%u x %u",nn[arm][k].p.gridDim.x,nn[arm][k].p.gridDim.y,nn[arm][k].p.gridDim.z,nn[arm][k].p.blockDim.x);
                printf("NODE %s %s layer #%zu | %s [%s] -> %s [%s] | hot %.2f -> %.2f cold %.2f -> %.2f us\n",tag.c_str(),kind[l],k,
                    nn[0][k].name.c_str(),gb[0],nn[1][k].name.c_str(),gb[1],v[0][1][0]-v[0][0][0],v[1][1][0]-v[1][0][0],v[0][1][1]-v[0][0][1],v[1][1][1]-v[1][0][1]);}
            cudaGraphDestroy(gg[0]);cudaGraphDestroy(gg[1]);}
        // TIME: whole call, reference vs arm alternating
        std::vector<float> v[8];
        for(int t=0;t<TR;t++)for(int arm=0;arm<2;arm++){set_arm(arm?x:ref_arm);
            for(int l=0;l<2;l++){v[arm*4+l*2].push_back(once([&]{call(l);},false));v[arm*4+l*2+1].push_back(once([&]{call(l);},true));}}
        float o[8];for(int q=0;q<8;q++)o[q]=med(v[q]);
        const double step_cold=(21*(o[1]-o[5])+57*(o[3]-o[7]))/1000.,step_hot=(21*(o[0]-o[4])+57*(o[2]-o[6]))/1000.;
        const double gate=x.vs_stock?0.5:M>=12||T>=32768?1.0:0.3;
        printf("TIME %s vs %s T=%d L=%d M=%d | hot full %.1f -> %.1f, shared %.1f -> %.1f | cold full %.1f -> %.1f, shared %.1f -> %.1f us | "
               "saved per step (21 full + 57 shared): cold %.3f ms, hot %.3f ms | gate %.1f %s\n",tag.c_str(),rtag.c_str(),T,L,M,o[0],o[4],o[2],o[6],o[1],o[5],o[3],o[7],
               step_cold,step_hot,gate,step_cold>=gate?"PASS":"below");
    }
    set_arm(STOCK);
    if(bad)return 3;
    return 0;
}
