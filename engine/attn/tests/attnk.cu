// attnk: G6_ATTN_SC2 (scores_sc) and G6_ATTN_CTX2 (context_parts_heads PB / context_pk) against the stock decode attention
// chain of one layer, as the engine replays it (g6_attn_replay_run captured in a CUDA graph). Harness = an earlier
// attention-chain bench without the MK chain kernel.
// Layers: a full-index layer (its own indexer + select) then a shared layer reusing its indices, M rows at live context L in
// capture bucket T, S2 batched cell with per-row tables (s2=1, as served). One call's GPU time inside a graph of K calls,
// median of TR alternating trials; "cold" = a 96 MB L2 flush before every call, flush subtracted (served case: the experts
// stream ~90 MB per rank between two layers' attention); "hot" = no flush.
//   REF     the stock chain (every arm flag 0): outputs of both layers, indices, appended KV rows, per layer the raw scores
//           (chain prefix through the scores node) and the context parts (prefix through the context node), the shared
//           layer's normalized probabilities and merged context. Captured once, before any arm is touched.
//   CHAIN   stock per-kernel table: in-chain cost of every node = T(prefix k) - T(prefix k-1), hot / cold / dead, DRAM floor.
//   ARM     one (G6_ATTN_SC2, G6_ATTN_SC2_TR, G6_ATTN_CTX2) setting:
//           EXACT  bytes differing vs REF over chk poisoned launches (raw scores, parts, probs, context, outputs, indices, kv);
//                  ctx2=9 is E1: parts have another layout (not compared), outputs / context report max |diff|, the rest
//                  must still be identical.
//           NODE   the changed nodes' in-chain cost (prefix deltas) stock -> arm, hot / cold, per layer.
//           TIME   whole call stock vs arm alternating, hot / cold, per layer, and the per-step projection over the main
//                  verify pass: 21 full-index + 57 shared layers (cold), ms/step.
//   Decision (written before running): adopt an arm if it is IDENTICAL (E1: separate KL gate) and saves >= 0.5 ms/step cold at
//   T=2048 L=1000 M=3, or >= 2 ms/step at M=12. Kill otherwise, or on any byte difference.
// build (from engine/attn/tests), with the engine's flags (scripts/build.sh K1H2_FLAGS, -DG6_NR=4 -DDEC_S2 -DPF_F12):
//   nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $FLAGS -o attnk attnk.cu ../bind.cu -lcublas
// run: attnk T L M [what=3] [s2=1] [reps=10] [K=16] [TR=7] [chk=5] [bw=250]
//   what: 1 stock CHAIN table, 2 the arms. Arms: env ATTNK_ARMS="S.TR.CTX ..." (default below).
//   sweep: for M in 1 3 8 12; do ./attnk 2048 1000 $M; ./attnk 8192 8000 $M; ./attnk 32768 32000 $M; done
// Exit 3 on any byte difference of an exact arm, 1 on a CUDA / validation error.
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
__global__ void fill_idx(uint8_t* c,size_t pages,int ps,uint64_t seed){
    for(size_t p=blockIdx.x;p<pages;p+=gridDim.x){uint8_t* b=c+p*ps*g6kv::IDX_ROW;
        for(int i=threadIdx.x;i<ps*128;i+=blockDim.x)b[i]=fin8(hsh(seed+p*100000+i));
        for(int i=threadIdx.x;i<ps;i+=blockDim.x)((float*)(b+ps*128))[i]=ldexpf(1.f,-6-(int)(hsh(seed+3+p*64+i)%3));}
}
__global__ void l2_flush(const float4* p,size_t n,float* sink){
    float acc=0;for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i];acc+=v.x+v.y+v.z+v.w;}
    if(acc==1234.5f)*sink=acc;   // never (the buffer is zero); keeps the loads
}

template<class T> static T* dalloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));CK(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T> static std::vector<T> host(const T* d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static size_t ndiff(const void* x,const void* y,size_t bytes){size_t k=0;for(size_t i=0;i<bytes;i++)k+=((const uint8_t*)x)[i]!=((const uint8_t*)y)[i];return k;}
static float med(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

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
// the attention scores node (not the indexer's) / the context node / the merge node of a chain; -1 if absent
static int node_of(const std::vector<Node>& c,const char* what){
    for(int i=0;i<(int)c.size();i++){const std::string& s=c[i].name;
        if(!strcmp(what,"scores")&&s.find("scores")!=std::string::npos&&s.find("index_scores")==std::string::npos)return i;
        if(!strcmp(what,"context")&&s.find("context")!=std::string::npos)return i;
        if(!strcmp(what,"merge")&&s.find("merge_parts")!=std::string::npos)return i;}
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
struct Arm{int sc,tr,ctx;};
static void set_arm(const Arm& x){ga::attn_sc2()=x.sc;ga::attn_sc2_tr()=x.tr;ga::attn_ctx2()=x.ctx;}

int main(int argc,char** argv){
    if(argc<4){fprintf(stderr,"usage: attnk T L M [what=3] [s2=1] [reps=10] [K=16] [TR=7] [chk=5] [bw=250]\n");return 2;}
    const int T=atoi(argv[1]),L=atoi(argv[2]),M=atoi(argv[3]),WHAT=argc>4?atoi(argv[4]):3,S2=argc>5?atoi(argv[5]):1,
        REPS=argc>6?atoi(argv[6]):10,K=argc>7?atoi(argv[7]):16,TR=argc>8?atoi(argv[8]):7,CHK=argc>9?atoi(argv[9]):5;
    const double BW=argc>10?atof(argv[10]):250.;
    if(L>T||M<1||M>G6_MX||M>L){fprintf(stderr,"bad sizes (M <= %d, M <= L <= T)\n",G6_MX);return 2;}
#ifdef DEC_S2
#else
    if(S2){fprintf(stderr,"s2 needs -DDEC_S2\n");return 2;}
#endif
#if G6_KV4
    fprintf(stderr,"attnk: FP8 KV build only (the flags are inert under G6_KV4)\n");return 2;
#endif
    std::vector<Arm> arms={{1,0,0},{1,1,0},{2,0,0},{2,1,0},{4,0,0},{4,1,0},{0,0,1},{0,0,2},{0,0,3},{0,0,4},{0,0,9},{2,1,3},{4,1,4}};
    if(const char* e=getenv("ATTNK_ARMS")){arms.clear();const char* p=e;
        while(*p){while(*p==' '||*p==',')p++;if(!*p)break;Arm x{};if(sscanf(p,"%d.%d.%d",&x.sc,&x.tr,&x.ctx)!=3){fprintf(stderr,"ATTNK_ARMS: S.TR.CTX\n");return 2;}
            arms.push_back(x);while(*p&&*p!=' '&&*p!=',')p++;}}
    const int H=g6nr::HMAX,pages=(T+63)/64;
    cudaStream_t st;CK(cudaStreamCreate(&st));
    {int dev=0,sms=0,clk=0,l2=0;CK(cudaGetDevice(&dev));cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
     cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,dev);cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,dev);
     printf("GPU %d SMs, attr clock %d MHz, L2 %.1f MB | T=%d L=%d M=%d H=%d s2=%d K=%d reps=%d TR=%d chk=%d bw=%.0f GB/s\n",sms,clk/1000,l2/1048576.,T,L,M,H,S2,K,REPS,TR,CHK,BW);}
    CKI(g6_attn_rope_init(T,st));   // as a served boot: the decode RoPE table (absorb / append skip the FP64 sincos)
    // weights, per-layer caches (layer 0 full index, layer 1 shared), step inputs
    __nv_bfloat16 *uk=dalloc<__nv_bfloat16>((size_t)H*512*192),*uv=dalloc<__nv_bfloat16>((size_t)H*256*512),
        *kvn=dalloc<__nv_bfloat16>(512),*inw=dalloc<__nv_bfloat16>(128),*inb=dalloc<__nv_bfloat16>(128);
    fill_bf16<<<512,256,0,st>>>(uk,(size_t)H*512*192,1,0.05f,0);fill_bf16<<<512,256,0,st>>>(uv,(size_t)H*256*512,2,0.05f,0);
    fill_bf16<<<2,256,0,st>>>(kvn,512,3,0.2f,1);fill_bf16<<<1,128,0,st>>>(inw,128,4,0.2f,1);fill_bf16<<<1,128,0,st>>>(inb,128,5,0.1f,0);
    uint8_t *mla[2],*ixc=dalloc<uint8_t>((size_t)pages*64*g6kv::IDX_ROW);
    for(int l=0;l<2;l++){mla[l]=dalloc<uint8_t>((size_t)pages*64*g6kv::ROW);fill_kv<<<1024,256,0,st>>>(mla[l],(size_t)pages*64,100+l);}
    fill_idx<<<512,256,0,st>>>(ixc,pages,64,200);
    __half *qb=dalloc<__half>((size_t)M*H*256),*kva=dalloc<__half>((size_t)M*576),*iq=dalloc<__half>((size_t)M*32*128),
        *ik=dalloc<__half>((size_t)M*128),*iw=dalloc<__half>((size_t)M*32);
    fill_half<<<64,256,0,st>>>(qb,(size_t)M*H*256,6,1.f);fill_half<<<8,256,0,st>>>(kva,(size_t)M*576,7,1.f);
    fill_half<<<64,256,0,st>>>(iq,(size_t)M*32*128,8,1.f);fill_half<<<1,256,0,st>>>(ik,(size_t)M*128,9,1.f);fill_half<<<1,256,0,st>>>(iw,(size_t)M*32,10,0.1f);
    std::vector<int> pos(M),table(pages);for(int i=0;i<M;i++)pos[i]=L-M+i;for(int i=0;i<pages;i++)table[i]=i;
    int *dpos=dalloc<int>(M),*dtab=dalloc<int>(pages),*ind[2]={dalloc<int>((size_t)M*ga::TOP),dalloc<int>((size_t)M*ga::TOP)};
    CK(cudaMemcpy(dpos,pos.data(),M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dtab,table.data(),pages*4,cudaMemcpyHostToDevice));
    __half* out[2]={dalloc<__half>((size_t)M*H*256),dalloc<__half>((size_t)M*H*256)};
    unsigned* err[2]={dalloc<unsigned>(1),dalloc<unsigned>(1)};
    G6Attn a[2]{};
    for(int l=0;l<2;l++){G6Attn& x=a[l];x.rows=M;x.heads=H;x.context=T;x.page_size=64;x.page_count=x.table_size=pages;x.query_tile=std::max(M,32);
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
    if(S2){std::vector<G6AttnRow> rr(M);for(auto& x:rr){x.table=dtab;x.context=L;x.page_count=pages;x.table_size=pages;x.ibase=0;x.sess=0;}
        drows=dalloc<G6AttnRow>(M);CK(cudaMemcpy(drows,rr.data(),M*sizeof(G6AttnRow),cudaMemcpyHostToDevice));}
#endif
    for(int l=0;l<2;l++){G6AttnStepCell c{};c.context=L;c.page_count=pages;c.table_size=pages;c.reuse_rows=l?M:0;
#ifdef DEC_S2
        c.rows=drows;
#endif
        CK(cudaMalloc(&dcell[l],sizeof c));CK(cudaMemcpy(dcell[l],&c,sizeof c,cudaMemcpyHostToDevice));
        c.context=0;CK(cudaMalloc(&dead[l],sizeof c));CK(cudaMemcpy(dead[l],&c,sizeof c,cudaMemcpyHostToDevice));}
    // scratch for the largest layout (ctx2 = 9 widens the parts)
    size_t sb=0;for(int c:{0,9}){ga::attn_ctx2()=c;sb=std::max({sb,g6_attn_scratch_bytes(&a[0]),g6_attn_scratch_bytes(&a[1])});}
    void* scr=dalloc<uint8_t>(sb);
    G6AttnReplay rp[2]={{a[0],dcell[0],scr,sb},{a[1],dcell[1],scr,sb}},rd[2]={{a[0],dead[0],scr,sb},{a[1],dead[1],scr,sb}};
    const size_t fl=(size_t)96<<20;float4* fb=dalloc<float4>(fl/16);float* sink=dalloc<float>(1);
    auto flushk=[&]{l2_flush<<<48*6,256,0,st>>>(fb,fl/16,sink);};
    CK(cudaStreamSynchronize(st));
    const char* kind[2]={"full-index","shared"};
    set_arm({0,0,0});

    // ---- reference (stock): outputs, the shared layer's scratch, per layer raw scores and parts (chain prefixes)
    const size_t no=(size_t)M*H*256,ni=(size_t)M*ga::TOP,nc=(size_t)M*H*512,np=(size_t)M*H*ga::TOP,npa=(size_t)M*H*ga::PART*512;
    struct Snap{std::vector<__half> o0,o1;std::vector<int> i0;std::vector<float> cx,pr,pa,sc[2],pl[2];std::vector<uint8_t> kv0,kv1;unsigned e0,e1,v0,v1;};
    auto call=[&](int l){CKI(g6_attn_replay_run(&rp[l],st));};
    auto both=[&]{call(0);call(1);};
    auto poison_w=[&](const ga::Work& W){CK(cudaMemsetAsync(W.context,0xab,nc*4,st));CK(cudaMemsetAsync(W.prob,0xab,np*4,st));CK(cudaMemsetAsync(W.parts,0xab,npa*4,st));};
    auto poison=[&](const ga::Work& W){for(int l=0;l<2;l++){CK(cudaMemsetAsync(out[l],0xff,no*2,st));CK(cudaMemsetAsync(ind[l],0x5a,ni*4,st));}poison_w(W);};
    // the full call into s (both layers), then per layer the chain prefixes through the scores and the context node
    auto take=[&](Snap& s,bool e1){
        const ga::Work W=ga::workspace(a[1],scr);
        {cudaGraphExec_t g=inst(st,both);poison(W);CK(cudaGraphLaunch(g,st));CK(cudaStreamSynchronize(st));cudaGraphExecDestroy(g);}
        s.o0=host(out[0],no);s.o1=host(out[1],no);s.i0=host(ind[0],ni);s.cx=host(W.context,nc);s.pr=host(W.prob,np);
        s.pa=e1?std::vector<float>():host(W.parts,npa);
        const size_t r0=(size_t)(L-M);s.kv0=host(mla[0]+r0*g6kv::ROW,(size_t)M*g6kv::ROW);s.kv1=host(mla[1]+r0*g6kv::ROW,(size_t)M*g6kv::ROW);
        s.e0=host(err[0],1)[0];s.e1=host(err[1],1)[0];s.v0=host(dcell[0],1)[0].valid;s.v1=host(dcell[1],1)[0].valid;
        for(int l=0;l<2;l++){cudaGraph_t g=capture(st,[&]{call(l);});auto c=chain_of(g);const int ks=node_of(c,"scores"),kc=node_of(c,"context");
            if(ks<0||kc<0){fprintf(stderr,"no scores / context node\n");exit(1);}
            // (the shared layer reads the full layer's indices: they are in place from the call above; scratch poisoned after)
            poison_w(W);for(int j=0;j<=ks;j++)relaunch(c[j],st);CK(cudaStreamSynchronize(st));s.sc[l]=host(W.prob,np);
            poison_w(W);for(int j=0;j<=kc;j++)relaunch(c[j],st);CK(cudaStreamSynchronize(st));s.pl[l]=e1?std::vector<float>():host(W.parts,npa);
            cudaGraphDestroy(g);}
        CK(cudaGetLastError());};
    Snap ref;take(ref,false);
    {int live=0;for(int v:ref.i0)live+=v>=0;size_t nan=0;for(auto h:ref.o0)nan+=!std::isfinite(__half2float(h));for(auto h:ref.o1)nan+=!std::isfinite(__half2float(h));
     printf("REF selected %d, nonfinite out %zu, err %u/%u valid %u/%u\n",live,nan,ref.e0,ref.e1,ref.v0,ref.v1);
     if(ref.e0||ref.e1||!ref.v0||!ref.v1)return 1;}

    // ---- timing helpers: per-call us of `body` (cold: behind a flush, the flush subtracted)
    float tflush=0;{cudaGraphExec_t g=inst(st,[&]{for(int k=0;k<K;k++)flushk();});std::vector<float> v;for(int t=0;t<TR;t++)v.push_back(run_us(g,st,REPS));tflush=med(v)/K;cudaGraphExecDestroy(g);}
    struct Timed{cudaGraphExec_t x;bool cold;std::vector<float> v;};
    auto timed_graph=[&](auto body,bool cold){return Timed{inst(st,[&]{for(int k=0;k<K;k++){if(cold)flushk();body();}}),cold,{}};};
    auto measure=[&](std::vector<Timed>& g){for(int t=0;t<TR;t++)for(auto& x:g)x.v.push_back(run_us(x.x,st,REPS)/K-(x.cold?tflush:0));
        for(auto& x:g)cudaGraphExecDestroy(x.x);};
    auto value=[](const Timed& x){return med(x.v);};
    printf("flush %.1f us per call (subtracted from cold)\n",tflush);

    // ---- CHAIN (stock): per-kernel prefix costs, live and dead, totals
    if(WHAT&1)for(int l=0;l<2;l++){
        cudaGraph_t gl=capture(st,[&]{call(l);}),gd=capture(st,[&]{CKI(g6_attn_replay_run(&rd[l],st));});
        std::vector<Node> nl=chain_of(gl),nd=chain_of(gd);
        if(nl.size()!=nd.size()){fprintf(stderr,"live / dead chains differ in length\n");exit(1);}
        const int n=(int)nl.size();std::vector<Timed> g;
        for(int k=1;k<=n;k++){g.push_back(timed_graph([&]{for(int j=0;j<k;j++)relaunch(nl[j],st);},false));
            g.push_back(timed_graph([&]{for(int j=0;j<k;j++)relaunch(nl[j],st);},true));
            g.push_back(timed_graph([&]{for(int j=0;j<k;j++)relaunch(nd[j],st);},false));}
        g.push_back(timed_graph([&]{call(l);},false));g.push_back(timed_graph([&]{call(l);},true));
        measure(g);
        printf("CHAIN stock %s layer T=%d L=%d M=%d: %d kernel nodes\n",kind[l],T,L,M,n);
        printf("  %-3s %-34s %13s %9s %9s %9s %8s\n","#","kernel","grid x block","hot","cold","dead","floor");
        float ph=0,pc=0,pd=0;double fsum=0;
        for(int k=0;k<n;k++){const float h=value(g[3*k]),c=value(g[3*k+1]),d=value(g[3*k+2]);
            const double fbb=floor_bytes(nl[k].name,H,L);fsum+=fbb;
            char gb[48];snprintf(gb,sizeof gb,"%ux%ux%u x %u",nl[k].p.gridDim.x,nl[k].p.gridDim.y,nl[k].p.gridDim.z,nl[k].p.blockDim.x);
            printf("  %-3d %-34.34s %13s %9.2f %9.2f %9.2f %8.2f\n",k,nl[k].name.c_str(),gb,h-ph,c-pc,d-pd,fbb/BW/1e3);
            ph=h;pc=c;pd=d;}
        printf("  total hot %.1f cold %.1f us | dead %.1f us | DRAM floor %.1f us\n",value(g[3*n]),value(g[3*n+1]),pd,fsum/BW/1e3);
        cudaGraphDestroy(gl);cudaGraphDestroy(gd);
    }

    // ---- arms
    int bad=0;
    if(WHAT&2)for(const Arm& x:arms){
        char tag[48];snprintf(tag,sizeof tag,"SC2=%d TR=%d CTX2=%d",x.sc,x.tr,x.ctx);
        const bool e1=x.ctx==9;
        set_arm(x);
        // the arm must reach this call's dispatch: some node differs from stock
        {set_arm({0,0,0});cudaGraph_t g0=capture(st,[&]{call(1);});auto c0=chain_of(g0);set_arm(x);cudaGraph_t g1=capture(st,[&]{call(1);});auto c1=chain_of(g1);
         bool differs=c0.size()!=c1.size();for(size_t i=0;!differs&&i<c0.size();i++)differs=c0[i].p.func!=c1[i].p.func||c0[i].p.gridDim.x!=c1[i].p.gridDim.x;
         cudaGraphDestroy(g0);cudaGraphDestroy(g1);
         if(!differs){printf("ARM %s not taken by this call's dispatch\n",tag);continue;}}
        // EXACT
        size_t w[12]={0};bool ok=true;double dout=0,dcx=0,mout=0,mcx=0;
        for(int it=0;it<CHK;it++){Snap s;take(s,e1);
            size_t d[12]={ndiff(ref.o0.data(),s.o0.data(),no*2),ndiff(ref.o1.data(),s.o1.data(),no*2),ndiff(ref.i0.data(),s.i0.data(),ni*4),
                ndiff(ref.pr.data(),s.pr.data(),np*4),e1?0:ndiff(ref.pa.data(),s.pa.data(),npa*4),ndiff(ref.cx.data(),s.cx.data(),nc*4),
                ndiff(ref.kv0.data(),s.kv0.data(),s.kv0.size())+ndiff(ref.kv1.data(),s.kv1.data(),s.kv1.size()),
                ndiff(ref.sc[0].data(),s.sc[0].data(),np*4),ndiff(ref.sc[1].data(),s.sc[1].data(),np*4),
                e1?0:ndiff(ref.pl[0].data(),s.pl[0].data(),npa*4),e1?0:ndiff(ref.pl[1].data(),s.pl[1].data(),npa*4),0};
            for(int q=0;q<11;q++)w[q]=std::max(w[q],d[q]);ok&=!s.e0&&!s.e1&&s.v0&&s.v1;
            for(size_t i=0;i<no;i++){const double r0=__half2float(ref.o0[i]),r1=__half2float(ref.o1[i]);
                dout=std::max({dout,fabs(r0-__half2float(s.o0[i])),fabs(r1-__half2float(s.o1[i]))});mout=std::max({mout,fabs(r0),fabs(r1)});}
            for(size_t i=0;i<nc;i++){dcx=std::max(dcx,(double)fabs(ref.cx[i]-s.cx[i]));mcx=std::max(mcx,(double)fabs(ref.cx[i]));}}
        bool exact=ok;for(int q=0;q<11;q++)exact&=!w[q];
        printf("EXACT %s T=%d L=%d M=%d x%d | bytes differing: scores full %zu shared %zu, parts full %zu shared %zu, probs %zu, parts(call) %zu, "
               "context %zu, out full %zu shared %zu, indices %zu, kv %zu | err/valid %s | %s\n",tag,T,L,M,CHK,w[7],w[8],w[9],w[10],w[3],w[4],w[5],w[0],w[1],w[2],w[6],
               ok?"ok":"BAD",exact?"IDENTICAL":e1?"DIFFER (E1)":"DIFFER");
        if(e1)printf("  E1 max |diff|: out %.3g (max |out| %.3g), context %.3g (max |context| %.3g)%s\n",dout,mout,dcx,mcx,
            w[2]||w[3]||w[6]||w[7]||w[8]||!ok?" | indices / probs / scores / kv / err NOT identical: BAD":"");
        if(!exact&&!e1){bad++;continue;}
        if(e1&&(w[2]||w[3]||w[6]||w[7]||w[8]||!ok)){bad++;continue;}
        // NODE: the changed nodes' prefix deltas, stock vs arm, per layer
        for(int l=0;l<2;l++){
            std::vector<Node> nn[2];cudaGraph_t gg[2];
            for(int arm=0;arm<2;arm++){set_arm(arm?x:Arm{0,0,0});gg[arm]=capture(st,[&]{call(l);});nn[arm]=chain_of(gg[arm]);}
            set_arm(x);
            const int ks=node_of(nn[0],"scores"),kc=node_of(nn[0],"context"),km=node_of(nn[0],"merge");
            const int ends[6]={ks-1,ks,kc-1,kc,km-1,km};   // prefixes of length end+1
            std::vector<Timed> g;
            for(int t=0;t<TR;t++)for(int arm=0;arm<2;arm++)for(int e=0;e<6;e++)for(int cold=0;cold<2;cold++){
                Timed tg=timed_graph([&]{for(int j=0;j<=ends[e];j++)relaunch(nn[arm][j],st);},cold);
                tg.v.push_back(run_us(tg.x,st,REPS)/K-(cold?tflush:0));cudaGraphExecDestroy(tg.x);g.push_back(tg);}
            auto at=[&](int arm,int e,int cold){std::vector<float> v;for(int t=0;t<TR;t++)v.push_back(g[((t*2+arm)*6+e)*2+cold].v[0]);return med(v);};
            auto dl=[&](int arm,int k,int cold){return at(arm,2*k+1,cold)-at(arm,2*k,cold);};
            printf("NODE %s %s layer | scores %s -> %s: hot %.2f -> %.2f cold %.2f -> %.2f | context %s -> %s: hot %.2f -> %.2f cold %.2f -> %.2f | "
                   "merge hot %.2f -> %.2f cold %.2f -> %.2f us\n",tag,kind[l],nn[0][ks].name.c_str(),nn[1][ks].name.c_str(),dl(0,0,0),dl(1,0,0),dl(0,0,1),dl(1,0,1),
                   nn[0][kc].name.c_str(),nn[1][kc].name.c_str(),dl(0,1,0),dl(1,1,0),dl(0,1,1),dl(1,1,1),dl(0,2,0),dl(1,2,0),dl(0,2,1),dl(1,2,1));
            cudaGraphDestroy(gg[0]);cudaGraphDestroy(gg[1]);}
        // TIME: whole call, stock vs arm alternating
        std::vector<float> v[8];
        for(int t=0;t<TR;t++)for(int arm=0;arm<2;arm++){set_arm(arm?x:Arm{0,0,0});
            for(int l=0;l<2;l++){std::vector<Timed> g;g.push_back(timed_graph([&]{call(l);},false));g.push_back(timed_graph([&]{call(l);},true));
                for(auto& y:g){y.v.push_back(run_us(y.x,st,REPS)/K-(y.cold?tflush:0));cudaGraphExecDestroy(y.x);}
                v[arm*4+l*2].push_back(g[0].v[0]);v[arm*4+l*2+1].push_back(g[1].v[0]);}}
        float o[8];for(int q=0;q<8;q++)o[q]=med(v[q]);
        const double step_cold=(21*(o[1]-o[5])+57*(o[3]-o[7]))/1000.,step_hot=(21*(o[0]-o[4])+57*(o[2]-o[6]))/1000.;
        printf("TIME %s T=%d L=%d M=%d | hot full %.1f -> %.1f, shared %.1f -> %.1f | cold full %.1f -> %.1f, shared %.1f -> %.1f us | "
               "saved per step (21 full + 57 shared): cold %.3f ms, hot %.3f ms | gate %s\n",tag,T,L,M,o[0],o[4],o[2],o[6],o[1],o[5],o[3],o[7],
               step_cold,step_hot,(M>=12?step_cold>=2.0:step_cold>=0.5)?(e1?"PASS (E1: needs KL)":"PASS"):"below");
    }
    set_arm({0,0,0});
    if(bad)return 3;
    return 0;
}
