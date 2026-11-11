// Synthetic exactness and kernel-only (NOT TTFT) screen for the INT4 attention paths.
// Build with scripts/build.sh's K1H2_FLAGS, G6_KV4=1, G6_NR=4, optionally DEC_S2;
// link bind.cu, libgr8knobs.a, cublas, pthread, dl and m. Run only on an allocated test GPU.
#include "../attn.cu"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#define CK(expr) do{const auto code=(expr);if(int(code)!=0){std::fprintf(stderr,"CUDA %s: %s at line %d\n",#expr,cudaGetErrorString(cudaError_t(code)),__LINE__);std::exit(2);}}while(0)
static void require(bool ok,const char* what){if(!ok){std::fprintf(stderr,"FAIL %s\n",what);std::exit(1);}}
static uint64_t hash_bytes(const std::vector<uint8_t>& v){uint64_t h=1469598103934665603ull;for(uint8_t c:v){h^=c;h*=1099511628211ull;}return h;}
struct Buffer {
    uint8_t* base=nullptr;size_t size=0;bool immutable=false;uint64_t before=0;
    explicit Buffer(size_t n):size(n){CK(cudaMalloc(&base,n+512));CK(cudaMemset(base,0xa7,n+512));}
    ~Buffer(){cudaFree(base);}
    void* data()const{return base+256;}
    template<class T>T* ptr()const{return static_cast<T*>(data());}
    std::vector<uint8_t> read()const{std::vector<uint8_t> v(size);CK(cudaMemcpy(v.data(),data(),size,cudaMemcpyDeviceToHost));return v;}
    void upload(const void* p){CK(cudaMemcpy(data(),p,size,cudaMemcpyHostToDevice));}
    void freeze(){before=hash_bytes(read());immutable=true;}
    void check()const{
        std::array<uint8_t,512> edges{};CK(cudaMemcpy(edges.data(),base,256,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(edges.data()+256,base+256+size,256,cudaMemcpyDeviceToHost));
        for(auto v:edges)require(v==0xa7,"red zone modified");
        if(immutable)require(hash_bytes(read())==before,"immutable input modified");
    }
};
struct Profile {int sc=0,tr=2,qt=0,ctx=0,rot=0,hg=0;};
static void set(Profile p){ga::kv4_sc()=p.sc;ga::kv4_sc_tr()=p.tr;ga::kv4_qt_on()=p.qt;ga::kv4_ctx()=p.ctx;ga::kv4_fuse_rot()=p.rot;ga::attn_ctx3_hg()=p.hg;}
static std::vector<Profile> profiles(){
    std::vector<Profile> p={{}};
    for(int s:{1,2,4})for(int tr:{0,1,2})for(int qt:{0,1})p.push_back({s,tr,qt,0,0,0});
    for(int ctx:{1,2})for(int hg:{0,2,4,8})p.push_back({0,2,0,ctx,0,hg});
    p.push_back({0,2,1,0,0,0}); // QT alone must remain disabled without SC.
    p.push_back({0,2,0,0,1,0});p.push_back({4,2,1,2,1,0});return p;
}
static uint32_t rng=0x613e21u;
static float random_small(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return float(int(rng%129)-64)/256.f;}

struct Fixture {
    G6Attn a{};int M,H,L;cudaStream_t stream{};
    std::vector<std::unique_ptr<Buffer>> buffers;
    Buffer *cache,*scratch,*output,*indices,*error,*cellbuf,*shifted,*allparts,*groupbuf,*groupout;
    std::vector<uint8_t> cache_initial;
    G6AttnStepCell cell{};
#ifdef DEC_S2
    Buffer* rowbuf;
#endif
    ga::Work w{};size_t blk,gblk;
    Buffer* alloc(size_t n){buffers.emplace_back(new Buffer(n));return buffers.back().get();}
    template<class T>T* input(const std::vector<T>& v){auto* b=alloc(v.size()*sizeof(T));b->upload(v.data());b->freeze();return b->ptr<T>();}
    Fixture(int m,int h,int l,bool sparse=false,bool remote=false,bool badpage=false):M(m),H(h),L(l){
        CK(cudaStreamCreate(&stream));a.rows=M;a.heads=H;a.context=L;a.page_size=64;a.page_count=(L+63)/64;a.table_size=a.page_count;a.query_tile=std::min(8,M);
        std::vector<int> pos(M),table(a.table_size),ids(size_t(M)*ga::TOP,-1);
        std::iota(table.begin(),table.end(),0);
        if(badpage)table[0]=a.page_count+1;
        for(int r=0;r<M;r++){
            pos[r]=L==37?(M==1?36:(r*36)/(M-1)):L-M+r;
            for(int j=0;j<std::min(ga::TOP,L);j++)ids[size_t(r)*ga::TOP+j]=j;
        }
        a.positions=input(pos);a.block_table=input(table);a.reuse_indices=input(ids);a.reuse_rows=M;
        std::vector<ga::half> qb(size_t(M)*H*256),kva(size_t(M)*576);
        std::vector<ga::bf16> uk(size_t(H)*512*192),uv(size_t(H)*256*512),norm(512,__float2bfloat16(1.f));
        for(size_t i=0;i<qb.size();i++){float v=random_small();if(sparse&&((i/256)%H)%2==0)v*=256.f;qb[i]=__float2half(v);}
        for(auto& v:kva)v=__float2half(random_small());
        for(auto& v:uk)v=__float2bfloat16(random_small()/16.f);
        for(auto& v:uv)v=__float2bfloat16(random_small()/16.f);
        a.q_b=input(qb);a.kv_a=input(kva);a.uk=input(uk);a.uv=input(uv);a.kv_norm=input(norm);
        cache=alloc(size_t(a.page_count)*64*g6kv::ROW);cache_initial.resize(cache->size);
        for(int j=0;j<a.page_count*64;j++){
            auto* row=cache_initial.data()+size_t(j)*g6kv::ROW;
            for(int b=0;b<256;b++){rng=rng*1664525u+1013904223u;row[b]=uint8_t(1+rng%15)|uint8_t((1+(rng>>8)%15)<<4);}
            for(int g=0;g<16;g++){const ga::half x=__float2half(.03125f+float((j+g)%7)/128.f);std::memcpy(row+g6kv::SCALE_OFF+g*2,&x,2);}
            for(int k=0;k<64;k++){const ga::bf16 x=__float2bfloat16(random_small()*(sparse?256.f:1.f));std::memcpy(row+g6kv::ROPE_OFF+k*2,&x,2);}
        }
        a.mla_cache=cache->data();indices=alloc(ids.size()*4);a.indices=indices->ptr<int>();
        output=alloc(size_t(M)*H*256*2);a.output=output->data();error=alloc(4);a.error=error->ptr<unsigned>();
        if(remote){std::vector<unsigned char> pages(a.page_count,1);std::vector<int> slots(a.page_count*64);std::iota(slots.begin(),slots.end(),0);
            a.rt_page=input(pages);a.rt_table=input(slots);a.rt_rows=input(cache_initial);a.rt_stride=g6kv::ROW;}
        // Allocate room for a single CP tile of M rows, even when replicated decode uses tiles of eight.
        G6Attn wide=a;wide.query_tile=M;const size_t need=ga::workspace(wide,nullptr).total;require(need>0,"workspace sizing");
        scratch=alloc(need);w=ga::workspace(a,scratch->data());
        shifted=alloc(size_t(M)*H*512*4+4);
        blk=(size_t(M)*H*514+3)&~size_t(3);gblk=(size_t(M)*16*514+3)&~size_t(3);
        allparts=alloc(8*blk*4);groupout=alloc(3*gblk*4);groupbuf=alloc(g6a_cp_groups_bytes(H,M));
        cell.context=L;cell.page_count=a.page_count;cell.table_size=a.table_size;cell.reuse_rows=M;cell.valid=1;
#ifdef DEC_S2
        // Alternate session tables point to different physical pages, not just the same table pointer.
        std::vector<int> alternate=table;if(!badpage&&a.page_count>1)std::reverse(alternate.begin(),alternate.end());
        const int* alt=input(alternate);std::vector<G6AttnRow> rows(M);
        for(int r=0;r<M;r++){rows[r].table=(r&1)?alt:a.block_table;rows[r].context=L;rows[r].page_count=a.page_count;rows[r].table_size=a.table_size;rows[r].sess=r&1;}
        rowbuf=alloc(rows.size()*sizeof(G6AttnRow));rowbuf->upload(rows.data());rowbuf->freeze();cell.rows=rowbuf->ptr<G6AttnRow>();a.s2_batched=1;
#endif
        cellbuf=alloc(sizeof cell);reset(true);
    }
    ~Fixture(){CK(cudaStreamSynchronize(stream));for(auto& b:buffers)b->check();cudaStreamDestroy(stream);}
    void reset(bool active){
        CK(cudaStreamSynchronize(stream));cache->upload(cache_initial.data());
        CK(cudaMemcpy(indices->data(),a.reuse_indices,indices->size,cudaMemcpyDeviceToDevice));
        CK(cudaMemset(output->data(),0x5b,output->size));CK(cudaMemset(scratch->data(),0x69,scratch->size));
        CK(cudaMemset(error->data(),0,4));CK(cudaMemset(allparts->data(),0x5b,allparts->size));CK(cudaMemset(groupout->data(),0x5b,groupout->size));
        cell.valid=active?1:0;cell.context=active?L:0;cellbuf->upload(&cell);
    }
    void check(){CK(cudaStreamSynchronize(stream));for(auto& b:buffers)b->check();}
    std::vector<uint8_t> result(bool cp=false){
        CK(cudaStreamSynchronize(stream));auto v=output->read();auto e=error->read();v.insert(v.end(),e.begin(),e.end());
        // Final scratch intermediate is checked too; query storage intentionally changes layout.
        const int n=cp?M:std::min(M,8);
        for(auto pair:{std::pair<const void*,size_t>{w.prob,size_t(n)*H*ga::TOP*4},
                       {w.parts,size_t(n)*H*ga::PART*512*4},{w.context,size_t(n)*H*512*4}}){
            size_t old=v.size();v.resize(old+pair.second);CK(cudaMemcpy(v.data()+old,pair.first,pair.second,cudaMemcpyDeviceToHost));
        }
        return v;
    }
};
static size_t comparisons=0;
static size_t differences(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){
    if(a.size()!=b.size())return size_t(-1);size_t n=0;for(size_t i=0;i<a.size();i++)n+=a[i]!=b[i];return n;
}
static void compare(const std::vector<uint8_t>& ref,const std::vector<uint8_t>& got,const char* label,Fixture& f,Profile p){
    size_t n=differences(ref,got);if(n){std::fprintf(stderr,"FAIL %s M=%d H=%d L=%d sc=%d tr=%d qt=%d ctx=%d rot=%d hg=%d differing_bytes=%zu\n",label,f.M,f.H,f.L,p.sc,p.tr,p.qt,p.ctx,p.rot,p.hg,n);std::exit(1);}++comparisons;
}
// Replay active=0 may alter scratch via the unconditional rotation, but never output.
static void inactive(Fixture& f){const auto v=f.output->read();for(auto x:v)require(x==0x5b,"inactive replay wrote output");}
static void run_rep(Fixture& f,int mode,bool active,bool graph=false){
    f.a.cp_nr=0;f.a.query_tile=std::min(f.M,8);f.w=ga::workspace(f.a,f.scratch->data());
    auto launch=[&](){
        if(mode==0)CK(ga::run_body(f.a,f.a,f.w,f.stream));
        else if(mode==1)CK(ga::run_body(f.a,ga::ReplayInput{f.a,f.cellbuf->ptr<G6AttnStepCell>()},f.w,f.stream));
        else if(mode==2)CK(g6a_run(&f.a,f.scratch->data(),f.scratch->size,f.stream));
        else{G6AttnReplay r{f.a,f.cellbuf->ptr<G6AttnStepCell>(),f.scratch->data(),f.scratch->size};CK(g6a_replay_run(&r,f.stream));}
    };
    if(graph){cudaGraph_t g;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(f.stream,cudaStreamCaptureModeThreadLocal));launch();CK(cudaStreamEndCapture(f.stream,&g));CK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));CK(cudaGraphLaunch(exec,f.stream));CK(cudaStreamSynchronize(f.stream));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(g));}
    else launch();
    CK(cudaStreamSynchronize(f.stream));if(!active)inactive(f);
}
// Change grouped [group][row][head] partials into cp_lse's [row][all heads] layout.
__global__ void ungroup(const float* groups,float* dst,int M,int H,int64_t gblk){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*H)return;
    const int r=i/H,h=i%H,g=h/16,local=r*16+h%16;const int64_t base=g*gblk;
    for(int d=0;d<512;d++)dst[(int64_t)i*512+d]=groups[base+(int64_t)local*512+d];
    dst[(int64_t)M*H*512+i]=groups[base+(int64_t)M*16*512+local];
    dst[(int64_t)M*H*513+i]=groups[base+(int64_t)M*16*513+local];
}
static void run_cp(Fixture& f,int nr,int mode,bool active,bool misalign=false,bool grouped=false){
    f.a.cp_nr=nr;f.a.cp_rank=0;f.a.query_tile=f.M;f.w=ga::workspace(f.a,f.scratch->data());
    const bool pub=mode==2||mode==3;const bool replay=mode==1||mode==3;
    const auto* cell=replay?f.cellbuf->ptr<G6AttnStepCell>():nullptr;
    auto dispatch=[&](auto in){
        if(pub&&f.H==16)CK(g6a_cp_front(&f.a,cell,f.scratch->data(),f.scratch->size,f.stream));
        else CK(ga::cp_front_body(f.a,in,f.w,f.stream));
    };
    if(replay)dispatch(ga::ReplayInput{f.a,cell});else dispatch(f.a);
    const float* q=f.w.q;
    if(misalign){q=f.shifted->ptr<float>()+1;CK(cudaMemcpyAsync(const_cast<float*>(q),f.w.q,size_t(f.M)*f.H*512*4,cudaMemcpyDeviceToDevice,f.stream));}
    for(int sh=0;sh<nr;sh++){
        f.a.cp_rank=sh;float* part=f.allparts->ptr<float>()+sh*f.blk;
        auto group=[&](auto in){
            if(grouped){
                if(pub)CK(g6a_cp_groups(&f.a,cell,f.groupbuf->data(),f.groupbuf->size,q,f.w.rope,f.groupout->ptr<float>(),16,f.gblk,f.stream));
                else{ga::Work gw{};char* b=f.groupbuf->ptr<char>();gw.prob=(float*)(b+cp_groups_at(f.H,f.M,0));gw.parts=(float*)(b+cp_groups_at(f.H,f.M,1));float* ml=(float*)(b+cp_groups_at(f.H,f.M,2));CK(ga::cp_groups_body(f.a,in,gw,ml,q,f.w.rope,f.groupout->ptr<float>(),16,f.gblk,f.stream));}
                if(active)ungroup<<<(f.M*f.H+255)/256,256,0,f.stream>>>(f.groupout->ptr<float>(),part,f.M,f.H,f.gblk);
            }else if(pub)CK(g6a_cp_group(&f.a,cell,f.scratch->data(),f.scratch->size,q,f.w.rope,part,f.stream));
            else CK(ga::cp_group_body(f.a,in,f.w,q,f.w.rope,part,f.stream));
        };
        if(replay)group(ga::ReplayInput{f.a,cell});else group(f.a);
    }
    f.a.cp_rank=0;
    auto finish=[&](auto in){
        if(pub&&f.H==16)CK(g6a_cp_finish(&f.a,cell,f.scratch->data(),f.scratch->size,f.allparts->ptr<float>(),f.stream));
        else CK(ga::cp_finish_body(f.a,in,f.w,f.allparts->ptr<float>(),f.stream));
    };
    if(replay)finish(ga::ReplayInput{f.a,cell});else finish(f.a);
    CK(cudaStreamSynchronize(f.stream));if(!active)inactive(f);
}
static std::vector<uint8_t> cp_result(Fixture& f,bool grouped){
    auto v=f.result(true);auto parts=f.allparts->read();v.insert(v.end(),parts.begin(),parts.end());
    if(grouped){auto p=f.groupout->read();v.insert(v.end(),p.begin(),p.end());}return v;
}
static void parity_case(Fixture& f,bool cp,int nr,int mode,bool active,bool grouped=false,bool unaligned=false,const std::vector<Profile>& cover=profiles()){
    std::vector<uint8_t> ref;
    for(auto p:cover){
        set(p);f.reset(active);
        // The stock scores kernel assumes aligned float4 loads. Its logical input is identical.
        const bool shift=unaligned&&p.sc!=0;
        if(cp)run_cp(f,nr,mode,active,shift,grouped);else run_rep(f,mode,active,mode==3);
        auto got=cp?cp_result(f,grouped):f.result();
        if(!active){got=f.output->read();auto e=f.error->read();got.insert(got.end(),e.begin(),e.end());}
        if(ref.empty())ref=got;else compare(ref,got,cp?"cp":"rep",f,p);
    }
    f.check();
    auto mutant=ref;mutant[0]^=1;require(differences(ref,mutant)==1,"negative control not detected");
}
static void baseline_trace(){
    Fixture f(8,16,37);G6AttnBinding b{};b.args=f.a;b.storage=f.cache->data();b.scratch=f.scratch->data();b.scratch_bytes=f.scratch->size;b.capacity=f.L;b.max_rows=f.M;
    G6AttnStep step{};step.rows=f.M;step.context=f.L;step.q_b=f.a.q_b;step.kv_a=f.a.kv_a;step.positions=f.a.positions;step.reuse_indices=f.a.reuse_indices;step.reuse_rows=f.M;
    std::vector<uint8_t> ref;
    for(Profile p:std::vector<Profile>{{},{4,2,1,2,1,0}}){set(p);f.reset(true);CK(g6a_bind_baseline_trace(&b,&step,nullptr,f.stream));auto got=f.result();if(ref.empty())ref=got;else compare(ref,got,"baseline_trace",f,p);}
}
static void rotation_parity(){
    for(int rows:{1,7,8,9}){
        std::vector<float> q(size_t(rows)*512);for(auto& x:q)x=random_small();
        Buffer reference(q.size()*4);reference.upload(q.data());
        ga::fwht_rows<<<(rows+7)/8,256>>>(reference.ptr<float>(),rows);CK(cudaDeviceSynchronize());
        const auto expected=reference.read();
        for(int offset:{0,4}){
            Buffer candidate(q.size()*4+offset);float* p=reinterpret_cast<float*>(candidate.base+256+offset);
            CK(cudaMemcpy(p,q.data(),q.size()*4,cudaMemcpyHostToDevice));
            ga::fwht_rows_qt<<<(rows+7)/8,256>>>(p,rows);CK(cudaDeviceSynchronize());
            std::vector<float> got(q.size());CK(cudaMemcpy(got.data(),p,q.size()*4,cudaMemcpyDeviceToHost));
            for(int row=0;row<rows;row++)for(int lane=0;lane<32;lane++)for(int c=0;c<4;c++)for(int e=0;e<4;e++){
                const int src=row*512+16*lane+4*c+e,dst=row*512+(c*32+lane)*4+e;
                require(!std::memcmp(expected.data()+src*4,&got[dst],4),"rotation layout mismatch");
            }
            candidate.check();++comparisons;
        }
        reference.check();
    }
}
static void parity(){
    rotation_parity();
    for(int H:{10,11,16,21,22,48})for(int M:{1,2,4,5,8,16})for(int L:{37,2048}){
        Fixture f(M,H,L);
        parity_case(f,false,0,0,true);parity_case(f,false,0,1,true);parity_case(f,false,0,1,false);
        const int nr=(H==10||H==11)?6:4;
        parity_case(f,true,nr,0,true);parity_case(f,true,nr,1,true);parity_case(f,true,nr,1,false);
        if(H==16){parity_case(f,false,0,2,true);parity_case(f,false,0,3,true);parity_case(f,false,0,3,false);
            parity_case(f,true,4,2,true);parity_case(f,true,4,3,true);parity_case(f,true,4,3,false);}
        if(H==48){parity_case(f,true,4,2,true,true);parity_case(f,true,4,1,true,true);parity_case(f,true,4,1,false,true);}
        std::printf("PASS matrix H=%d M=%d L=%d comparisons=%zu\n",H,M,L,comparisons);std::fflush(stdout);
    }
    for(int kind=0;kind<4;kind++){
        Fixture f(8,16,kind==2?37:2048,kind==0,kind==1,kind==2);
        parity_case(f,false,0,0,true);parity_case(f,false,0,1,true);
        if(kind==0){
            std::vector<float> p(size_t(f.M)*f.H*ga::TOP);CK(cudaMemcpy(p.data(),f.w.prob,p.size()*4,cudaMemcpyDeviceToHost));
            size_t zeros[2]={};for(int h=0;h<f.H;h++)for(int j=0;j<ga::TOP;j++)zeros[h&1]+=p[(size_t(f.M-1)*f.H+h)*ga::TOP+j]==0.f;
            require(zeros[0]>0&&zeros[1]<zeros[0],"softmax underflow subset not exercised");
        }
        parity_case(f,true,4,0,true,false,kind==3);parity_case(f,true,4,1,true,false,kind==3);
        if(kind==2){unsigned e=0;CK(cudaMemcpy(&e,f.a.error,4,cudaMemcpyDeviceToHost));require((e&2)!=0,"invalid page error(2) not exercised");}
    }
    baseline_trace();std::printf("PASS exactness comparisons=%zu differing_bytes=0 error_words=identical negative_control=detected guards=PASS inputs=PASS\n",comparisons);
}
// Focused sanitizer cover; the no-argument exhaustive matrix above is unchanged.
static void memcheck(){
    const std::vector<Profile> cover={{},{1,0,0,0,0,0},{2,1,1,1,0,4},{4,2,0,2,0,8},{0,2,0,0,1,0},{4,2,1,2,1,0}};
    rotation_parity();
    for(int H:{11,16,48})for(int M:{1,8})for(int L:{37,2048}){
        Fixture f(M,H,L);const int nr=H==11?6:4;
        const int mode=H==16?(M==1?2:3):(M==1?0:1);
        parity_case(f,false,0,mode,true,false,false,cover);
        const int cp_mode=H==48?(M==1?2:3):mode;
        parity_case(f,true,nr,cp_mode,true,H==48,false,cover);
        if(M==8&&L==37){
            const std::vector<Profile> inactive_cover={{},{4,2,1,2,1,0}};
            parity_case(f,false,0,H==16?3:1,false,false,false,inactive_cover);
            parity_case(f,true,nr,H==11?1:3,false,H==48,false,inactive_cover);
        }
    }
    for(int kind=0;kind<4;kind++){
        Fixture f(8,16,kind==2?37:2048,kind==0,kind==1,kind==2);
        parity_case(f,false,0,1,true,false,false,cover);
        if(kind==0){
            std::vector<float> p(size_t(f.M)*f.H*ga::TOP);CK(cudaMemcpy(p.data(),f.w.prob,p.size()*4,cudaMemcpyDeviceToHost));
            size_t zeros[2]={};for(int h=0;h<f.H;h++)for(int j=0;j<ga::TOP;j++)zeros[h&1]+=p[(size_t(f.M-1)*f.H+h)*ga::TOP+j]==0.f;
            require(zeros[0]>0&&zeros[1]<zeros[0],"softmax underflow subset not exercised");
        }
        parity_case(f,true,4,1,true,false,kind==3,cover);
        if(kind==2){unsigned e=0;CK(cudaMemcpy(&e,f.a.error,4,cudaMemcpyDeviceToHost));require((e&2)!=0,"invalid page error(2) not exercised");}
    }
    baseline_trace();std::printf("PASS memcheck comparisons=%zu differing_bytes=0 error_words=identical\n",comparisons);
}
// One rank's CP work: front + its local head group + the three remote groups + LSE finish.
static void bench_launch(Fixture& f,Fixture* remote){
    if(!remote){CK(ga::run_body(f.a,f.a,f.w,f.stream));return;}
    CK(ga::cp_front_body(f.a,f.a,f.w,f.stream));
    CK(ga::cp_group_body(f.a,f.a,f.w,f.w.q,f.w.rope,f.allparts->ptr<float>(),f.stream));
    CK(g6a_cp_groups(&remote->a,nullptr,remote->groupbuf->data(),remote->groupbuf->size,remote->w.q,remote->w.rope,
                    remote->groupout->ptr<float>(),16,remote->gblk,f.stream));
    CK(ga::cp_finish_body(f.a,f.a,f.w,f.allparts->ptr<float>(),f.stream));
}
static void bench_remote_prepare(Fixture& remote,Profile p){
    set(p);remote.reset(true);remote.a.cp_nr=4;remote.a.cp_rank=0;remote.a.query_tile=remote.M;
    remote.w=ga::workspace(remote.a,remote.scratch->data());
    CK(ga::cp_front_body(remote.a,remote.a,remote.w,remote.stream));CK(cudaStreamSynchronize(remote.stream));
}
static void bench_shape(int M,int L,bool cp){
    Fixture f(M,16,L);std::unique_ptr<Fixture> remote_stock,remote_arm;
    if(cp){const uint32_t seed=rng;remote_stock.reset(new Fixture(M,48,L));rng=seed;remote_arm.reset(new Fixture(M,48,L));}
    const Profile stock{};const std::vector<Profile> arms={{},{4,2,1,0,0,0},{0,2,0,2,0,0},{0,2,0,0,1,0},{4,2,1,2,1,0}};
    const char* labels[]={"stock-null","scores-only","ctx-only","rot-only","full"};
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    set(stock);f.reset(true);
    if(cp){run_cp(f,4,0,true);f.a.cp_rank=0;f.a.query_tile=f.M;bench_remote_prepare(*remote_stock,stock);}
    else f.a.cp_nr=0;
    f.w=ga::workspace(f.a,f.scratch->data());
    bench_launch(f,remote_stock.get());CK(cudaStreamSynchronize(f.stream));
    const auto reference=f.output->read();const auto ref_error=f.error->read();
    auto poison=[&](Fixture* remote){CK(cudaMemsetAsync(f.output->data(),0x5b,f.output->size,f.stream));CK(cudaMemsetAsync(f.error->data(),0,4,f.stream));
        if(remote)CK(cudaMemsetAsync(remote->groupout->data(),0x5b,remote->groupout->size,f.stream));};
    auto check=[&](){require(!differences(reference,f.output->read()),"timed output mismatch");require(!differences(ref_error,f.error->read()),"timed error mismatch");};
    // A compute-once/skipped-work mutant must be detected with the same fresh poison.
    poison(remote_stock.get());CK(cudaStreamSynchronize(f.stream));require(differences(reference,f.output->read())>0,"compute-once negative control not detected");
    for(size_t arm=0;arm<arms.size();arm++){
        if(cp)bench_remote_prepare(*remote_arm,arms[arm]);
        std::vector<float> times[2];
        for(int warm=0;warm<4;warm++)for(int k=0;k<2;k++){
            set(k?arms[arm]:stock);Fixture* remote=k?remote_arm.get():remote_stock.get();
            poison(remote);bench_launch(f,remote);CK(cudaStreamSynchronize(f.stream));check();
        }
        // Counterbalanced AB/BA; immutable uploads and query preparation stay outside timing.
        for(int it=0;it<200;it++)for(int order=0;order<2;order++){
            const int k=order^(it&1);set(k?arms[arm]:stock);Fixture* remote=k?remote_arm.get():remote_stock.get();poison(remote);
            CK(cudaEventRecord(begin,f.stream));bench_launch(f,remote);CK(cudaEventRecord(end,f.stream));CK(cudaEventSynchronize(end));
            float ms=0;CK(cudaEventElapsedTime(&ms,begin,end));times[k].push_back(ms*1000.f);if(it==199)check();
        }
        auto mean=[](const std::vector<float>& v){return std::accumulate(v.begin(),v.end(),0.)/v.size();};
        const double a=mean(times[0]),b=mean(times[1]);double min_delta=1e30,max_delta=-1e30;
        for(int i=0;i<200;i++){const double d=times[0][i]-times[1][i];min_delta=std::min(min_delta,d);max_delta=std::max(max_delta,d);}
        std::printf("SYNTHETIC kernel-only, NOT TTFT M=%d L=%d CP=%d local_H=16 remote_H=%d profile=%s n=200 stock_us=%.4f candidate_us=%.4f x79_saved_ms=%.4f paired_delta_range_us=[%.4f,%.4f] keep_drop=DISABLED_pending_capture_and_calibration\n",
                    M,L,cp?4:0,cp?48:0,labels[arm],a,b,(a-b)*79/1000,min_delta,max_delta);
        f.check();if(cp){remote_stock->check();remote_arm->check();}
    }
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
}
int main(int argc,char** argv){
    if(argc>2||(argc==2&&std::strcmp(argv[1],"--bench")&&std::strcmp(argv[1],"--memcheck"))){std::fprintf(stderr,"usage: s6_kv4_test [--bench|--memcheck]\n");return 2;}
    // Tests never consult ambient tuning profiles for the paths being compared.
    setenv("GR8_ATTN_HEADS","1",1);setenv("GR8_ATTN_FLIP","0",1);
    ga::avp1_on()=0;ga::attn_front_on()=0;ga::attn_posidx()=0;ga::attn_sel2()=0;ga::attn_idx_grp()=0;
    set({});cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));
    std::printf("SYNTHETIC device=%s sm=%d%d runtime_clock_telemetry=external-required keep_drop=DISABLED_pending_capture_and_calibration\n",prop.name,prop.major,prop.minor);
    if(argc==2&&!std::strcmp(argv[1],"--memcheck"))memcheck();
    else if(argc==2){for(int m:{1,8}){bench_shape(m,512,false);bench_shape(m,4096,false);bench_shape(m,4096,true);}}
    else parity();
    CK(cudaDeviceSynchronize());return 0;
}
