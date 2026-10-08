// #31 decode indexer microbenchmark, one GPU. Times the decode index stages of one full-index layer (prep_index,
// index_scores, the exact top-2048 select) for M rows at a live context L inside a graph bucket T, replicated (cp 0) or
// as CP rank 0 of cp_nr, each stage captured in its own CUDA graph and replayed. Random index cache and queries.
//   build (from engine/attn/tests): nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $FLAGS -o idxbench idxbench.cu
//   run: idxbench T L M cp_nr [reps] [ties]   one line: stage times in us per call (CP rank 0), exactness of the #31 path
//        checked on every CP rank (one EXACT line per rank; exit 3 if any rank differs)
//   ties = k > 0: index keys from k code values with one scale (many equal scores: the id tie-break is exercised)
#include "../attn.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>
#define CK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)
template<class F> static float timed(cudaStream_t st,int reps,F body){
    cudaGraph_t g;cudaGraphExec_t x;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));body();
    CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&x,g,0));
    for(int i=0;i<3;i++)CK(cudaGraphLaunch(x,st));
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,st));
    for(int i=0;i<reps;i++)CK(cudaGraphLaunch(x,st));
    CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
    cudaGraphExecDestroy(x);cudaGraphDestroy(g);return ms*1000.f/reps;
}
int main(int argc,char** argv){
    if(argc<5){fprintf(stderr,"usage: idxbench T L M cp_nr [reps]\n");return 2;}
    const int T=atoi(argv[1]),L=atoi(argv[2]),M=atoi(argv[3]),cp=atoi(argv[4]),reps=argc>5?atoi(argv[5]):50,ties=argc>6?atoi(argv[6]):0;
    if(L>T||M<1||M>8){fprintf(stderr,"bad sizes\n");return 2;}
    std::mt19937 rng(31);
    const int pages=(T+63)/64;
    std::vector<uint8_t> cache((size_t)pages*64*ga::IDX_BYTES);
    for(int p=0;p<pages;p++){uint8_t* base=cache.data()+(size_t)p*64*ga::IDX_BYTES;
        for(int i=0;i<64*128;i++){uint8_t v=rng()&0xff;if((v&0x7f)==0x7f)v^=1;base[i]=ties?(uint8_t)(0x30+(rng()%ties)):v;}   // finite e4m3; ties: few values
        float* sc=(float*)(base+64*128);for(int i=0;i<64;i++)sc[i]=ties?ldexpf(1.f,-6):ldexpf(1.f,-6-(int)(rng()%3));}
    std::vector<__half> q((size_t)M*32*128),w((size_t)M*32);
    std::normal_distribution<float> nd(0.f,1.f);
    for(auto& v:q)v=__float2half(nd(rng));for(auto& v:w)v=__float2half(nd(rng)*0.1f);
    std::vector<int> pos(M),table(pages);for(int i=0;i<M;i++)pos[i]=L-M+i;for(int i=0;i<pages;i++)table[i]=i;
    void *dcache,*dq,*dw;int *dpos,*dtab,*dind;unsigned* derr;
    CK(cudaMalloc(&dcache,cache.size()));CK(cudaMemcpy(dcache,cache.data(),cache.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dq,q.size()*2));CK(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dw,w.size()*2));CK(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dpos,M*4));CK(cudaMemcpy(dpos,pos.data(),M*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dtab,pages*4));CK(cudaMemcpy(dtab,table.data(),pages*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dind,(size_t)M*ga::TOP*4));CK(cudaMalloc(&derr,4));CK(cudaMemset(derr,0,4));
    G6Attn a{};a.rows=M;a.heads=g6nr::HMAX;a.context=T;a.page_size=64;a.page_count=a.table_size=pages;a.query_tile=M;
    a.index_q=dq;a.index_weight=dw;a.index_cache=dcache;a.positions=dpos;a.block_table=dtab;a.indices=dind;a.error=derr;
#if defined(G6_CP) && G6_CP
    a.cp_nr=cp;a.cp_rank=0;
#else
    if(cp){fprintf(stderr,"cp needs -DG6_CP=1\n");return 2;}
#endif
    // the live context arrives through the replay cell, as in the graph form (context = bucket capacity)
    G6AttnStepCell cell{L,pages,pages,0,1},*dcell;CK(cudaMalloc(&dcell,sizeof cell));CK(cudaMemcpy(dcell,&cell,sizeof cell,cudaMemcpyHostToDevice));
    ga::ReplayInput in{a,dcell};
    size_t bytes=g6_attn_scratch_bytes(&a);void* scr;CK(cudaMalloc(&scr,bytes));auto W=ga::workspace(a,scr);
    cudaStream_t st;CK(cudaStreamCreate(&st));
    auto hash=[&](std::vector<int>& ind){CK(cudaStreamSynchronize(st));CK(cudaMemcpy(ind.data(),dind,ind.size()*4,cudaMemcpyDeviceToHost));
        uint64_t h=1469598103934665603ull;for(int v:ind){h^=(uint32_t)v;h*=1099511628211ull;}return h;};
    std::vector<int> i0((size_t)M*ga::TOP),i1(i0.size());std::vector<float> s0((size_t)M*T),s1(s0.size());
    // scores and indices are poisoned before each path, so a stage that skips a write cannot pass on stale data
    auto poison=[&]{CK(cudaStreamSynchronize(st));CK(cudaMemset(W.si,0xff,s0.size()*4));CK(cudaMemset(dind,0xff,i0.size()*4));};
    float t_prep=0,t_scores=0,t_select=0,n_scores=0,n_select=0;int bad=0;
    for(int r=0;r<(cp?cp:1);r++){
#if defined(G6_CP) && G6_CP
        a.cp_rank=r;in.fixed=a;
#endif
        const int R=r?1:reps;   // time on rank 0 only
        // baseline path (index_scores + select_index)
        poison();float tp=timed(st,R,[&]{ga::prep_index<<<dim3(M,32),128,0,st>>>(in,0,W.iq,W.weights);});
        float ts=timed(st,R,[&]{ga::launch_index_scores(a,in,W,0,M,st);});
        CK(cudaStreamSynchronize(st));CK(cudaMemcpy(s0.data(),W.si,s0.size()*4,cudaMemcpyDeviceToHost));
        float tl=timed(st,R,[&]{CK(ga::select_index(a,in,W,0,M,st));});
        uint64_t h0=hash(i0);
        // #31 path (index_scores_m + select_index31)
        poison();timed(st,1,[&]{ga::prep_index<<<dim3(M,32),128,0,st>>>(in,0,W.iq,W.weights);});
        float ns=timed(st,R,[&]{ga::launch_index_scores_m(a,in,W,0,M,st);});
        CK(cudaStreamSynchronize(st));CK(cudaMemcpy(s1.data(),W.si,s1.size()*4,cudaMemcpyDeviceToHost));
        float nl=timed(st,R,[&]{CK(ga::select_index31(a,in,W,0,M,st));});
        uint64_t h1=hash(i1);
        size_t sdiff=0;for(size_t i=0;i<s0.size();i++)if(memcmp(&s0[i],&s1[i],4))sdiff++;
        int valid=0;for(int v:i0)valid+=v>=0;
        printf("EXACT T=%d L=%d M=%d cp=%d rank=%d | valid=%d score_diff=%zu ind_equal=%d hash=%016llx\n",T,L,M,cp,r,valid,sdiff,
            (int)(i0==i1),(unsigned long long)h0);
        if(sdiff||i0!=i1)bad++;
        if(!r){t_prep=tp;t_scores=ts;t_select=tl;n_scores=ns;n_select=nl;}
    }
#if defined(G6_CP) && G6_CP
    a.cp_rank=0;in.fixed=a;
#endif
    unsigned err;CK(cudaMemcpy(&err,derr,4,cudaMemcpyDeviceToHost));
    printf("T=%d L=%d M=%d cp=%d | base prep %.1f scores %.1f select %.1f | idx31 scores %.1f select %.1f us | err=%u ranks_bad=%d\n",
        T,L,M,cp,t_prep,t_scores,t_select,n_scores,n_select,err,bad);
    if(bad)return 3;
    // attention stage after the select (indices as selected above), heads = 11 (the widest TP6 group):
    //   replicated: mla scores + normalize + context parts + merge for my heads (run_body)
    //   CP: cp_group for each of the cp_nr head groups (cp_group_body: mla scores + cp_softmax + context parts + merge)
    {
        void* mla;CK(cudaMalloc(&mla,(size_t)pages*64*ga::KV_BYTES));
        std::vector<uint8_t> kv((size_t)pages*64*ga::KV_BYTES);
        for(size_t i=0;i<kv.size();i+=ga::KV_BYTES){for(int d=0;d<512;d++){uint8_t v=rng()&0xff;if((v&0x7f)==0x7f)v^=1;kv[i+d]=v;}
            float* sc=(float*)&kv[i+512];for(int g=0;g<4;g++)sc[g]=ldexpf(1.f,-7);
            __nv_bfloat16* r=(__nv_bfloat16*)&kv[i+528];for(int d=0;d<64;d++)r[d]=__float2bfloat16(nd(rng)*0.1f);}
        CK(cudaMemcpy(mla,kv.data(),kv.size(),cudaMemcpyHostToDevice));
        const int H=a.heads;std::vector<float> qh((size_t)M*H*576);for(auto& v:qh)v=nd(rng)*0.05f;
        float *dqa,*part;CK(cudaMalloc(&dqa,qh.size()*4));CK(cudaMemcpy(dqa,qh.data(),qh.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&part,((size_t)M*H*514+4)*4));
        G6Attn b=a;b.mla_cache=mla;
        ga::ReplayInput in2{b,dcell};auto W2=W;W2.q=dqa;W2.rope=dqa+(size_t)M*H*512;
        float t_rep=timed(st,reps,[&]{ga::launch_mla_scores(b,in2,W2,0,M,st);ga::normalize<<<M*H,256,0,st>>>(in2,W2.prob);
            ga::launch_context_parts(b,in2,W2,0,M,st);ga::merge_parts<<<M*H,256,0,st>>>(in2,W2.parts,W2.context);});
        float t_cp=0;
#if defined(G6_CP) && G6_CP
        if(cp)t_cp=timed(st,reps,[&]{for(int g=0;g<cp;g++)CK(ga::cp_group_body(b,in2,W2,W2.q,W2.rope,part,st));});
#endif
        printf("ATTN T=%d L=%d M=%d cp=%d heads=%d | replicated attention %.1f us | cp groups x%d %.1f us\n",T,L,M,cp,H,t_rep,cp,t_cp);
    }
    return err?1:0;
}
