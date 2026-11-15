// Included inside namespace test. Candidate projections are always owner-local;
// only the independent reference takes a full-M projection. The wire is the
// production multi-segment pack/publish/wait/unpack implementation.
namespace wire = ds41::v2;
using Bytes = v2test::Bytes;
size_t bit_diff(const Bytes& a,const Bytes& b) {
    require(a.size()==b.size(),"byte sizes"); size_t n=0;
    for(size_t i=0;i<a.size();++i)n+=__builtin_popcount(unsigned(a[i]^b[i])); return n;
}
void byte_check(const char* label,const void* got,const void* ref,size_t bytes,bool exact=true) {
    auto a=v2test::download(got,bytes),b=v2test::download(ref,bytes);
    size_t n=bit_diff(a,b); printf("stage=%s bit_mismatches=%zu exact=%d\n",label,n,exact);
    if(exact)require(n==0,label);
}
template<class T> void numeric(const char* label,const T* got,const T* ref,size_t n,bool exact,float tol=.005f) {
    byte_check(label,got,ref,n*sizeof(T),exact);
    close(label,host(got,n),host(ref,n),exact?0:tol);
}
std::vector<wire::Segment> segments(v2::InputBuf& own,v2::InputBuf& in,int mode) {
    if(mode==3)return {{own.ckv1,in.ckv1,HD*2}}; // L20 hoist only
    std::vector<wire::Segment> s{{own.qa,in.qa,QA*2},{own.kv0,in.kv0,HD*2}};
    if(mode==1)s.push_back({own.ckv1,in.ckv1,HD*2});
    if(mode==2){s.push_back({own.ckv,in.ckv,HD*4});s.push_back({own.csc,in.csc,HD*4});}
    return s;
}
// Preserve tail sentinels in every gathered array; the production allocation has
// only M rows, so unpack must not write padding. Pack padding must be zero.
template<class Producer> void gather(Arena&,v2::InputBuf& own,v2::InputBuf& in,int M,int mode,Producer produce,bool negative) {
    const int Q=(M+3)/4,P=Q*4; auto seg=segments(own,in,mode);
    const auto layout=wire::wire_layout(Q,seg.data(),seg.size());
    require(layout.payload==(mode==3?size_t(Q)*HD*2:wire::attn_in_wire_layout(Q,mode).payload),"host wire helper matches kernels");
    v2test::Arena mapped; v2test::Wire net(mapped,layout.stride);
    std::array<Bytes,4> sent; std::vector<Bytes> expected(seg.size());
    for(size_t j=0;j<seg.size();++j)expected[j].resize(size_t(M)*seg[j].row_bytes);
    for(int r=0;r<4;++r) {
        auto g=wire::geometry(M,4,r); produce(r,g.owned);
        v2test::Inputs immutable;
        for(size_t j=0;j<seg.size();++j) {
            size_t count=size_t(g.owned)*seg[j].row_bytes;
            auto raw=v2test::download(seg[j].src,count);
            if(count)memcpy(expected[j].data()+size_t(g.row0)*seg[j].row_bytes,raw.data(),count);
            immutable.add(seg[j].src,size_t(Q)*seg[j].row_bytes);
        }
        net.clear(r); auto args=wire::detail::segments(g,seg.data(),seg.size());
        wire::detail::pack<<<8,256>>>(net.f,1,0,g,args,false);
        wire::detail::publish<><<<1,32>>>(net.f,1,0); sent[r]=net.snapshot(); immutable.verify();
        for(size_t j=0;j<seg.size();++j)
            for(size_t k=size_t(g.owned)*seg[j].row_bytes;k<size_t(Q)*seg[j].row_bytes;++k)
                require(sent[r][layout.offsets[j]+k]==0,"zero packed ragged tail");
    }
    auto receive=[&](int r) {
        net.clear(r); net.deliver(sent,layout,false);
        for(auto s:seg)CK(cudaMemset(s.dst,0xa7,size_t(P)*s.row_bytes));
        auto g=wire::geometry(M,4,r); auto args=wire::detail::segments(g,seg.data(),seg.size());
        wire::detail::wait<><<<1,32>>>(net.f,1,0,layout.payload);
        wire::detail::unpack<><<<8,256>>>(net.f,1,0,g,args,false);
        CK(cudaDeviceSynchronize()); require(*net.f.error==0,"wire error word");
    };
    for(int r=0;r<4;++r) {
        receive(r);
        for(size_t j=0;j<seg.size();++j) {
            auto raw=v2test::download(seg[j].dst,size_t(P)*seg[j].row_bytes);
            require(std::equal(expected[j].begin(),expected[j].end(),raw.begin()),"native wire bit_mismatches == 0");
            for(size_t k=expected[j].size();k<raw.size();++k)require(raw[k]==0xa7,"unpack tail untouched");
        }
    }
    if(negative) {
        sent[0][layout.offsets[0]]^=1; receive(3);
        require(bit_diff(v2test::download(seg[0].dst,size_t(M)*seg[0].row_bytes),expected[0])==1,"one wire bit detected");
        sent[0][layout.offsets[0]]^=1; receive(3);
        require(bit_diff(v2test::download(seg[0].dst,size_t(M)*seg[0].row_bytes),expected[0])==0,"negative control restored");
    }
    printf("WIRE_PASS M=%d mode=%d ranks=4 bit_mismatches=0 padding=PASS negative=%d\n",M,mode,negative);
}
void wire_test(int M,int mode) {
    Arena a; const int Q=(M+3)/4; auto own=input(a,Q,Q),in=input(a,Q*4,0);
    gather(a,own,in,M,mode,[&](int r,int n) {
        // Distinct deterministic native values for every owner and segment.
        rng=1017+71*r; fill(own.qa,size_t(n)*QA,3);fill(own.kv0,size_t(n)*HD,2);
        fill(own.ckv1,size_t(n)*HD,5);fill(own.ckv,size_t(n)*HD,7);fill(own.csc,size_t(n)*HD,9);
    },true);
}
// Same MXFP8 record layout as the loader and the standalone stock attention test.
__global__ void make_gv(const uint8_t* codes,const uint8_t* scales,int K,uint8_t* out) {
    int ng=K/32,t=blockIdx.x/ng,g=blockIdx.x%ng,i=threadIdx.x;
    auto* rec=out+size_t(blockIdx.x)*528;
    int l=i>>3,e=i&7,r=l>>2,c=l&3,row=r+((e&2)?8:0),col=2*c+(e&1)+((e&4)?8:0);
    for(int h=0;h<2;++h)rec[h*256+i]=codes[size_t(16*t+row)*K+32*g+16*h+col];
    if(i<8){rec[512+2*i]=scales[size_t(16*t+i)*ng+g];rec[513+2*i]=scales[size_t(16*t+i+8)*ng+g];}
}
void records(Arena& a,Fp8Mat& w) {
    size_t n=size_t(w.N/16)*(w.K/32); auto* p=a.alloc<uint8_t>(n*528);
    make_gv<<<n,256>>>(w.w,w.s,w.K,p);w.gv=p;
}
bf16* ones(Arena& a,int n) {
    auto* p=a.alloc<bf16>(n);std::vector<bf16> v(n,bf16(1));
    CK(cudaMemcpy(p,v.data(),n*2,cudaMemcpyHostToDevice));return p;
}
struct Stage {
    Scratch s; Cache c{}; uint8_t *ca,*ica; bf16 *pre,*shadow;
    Stage(Arena& a,int M,int heads,int cap,int ring):s(scratch(a,M,heads)) {
        c.ring_rows=ring;c.ring=a.alloc<bf16>(size_t(ring)*HD);
        c.comp=a.alloc<bf16>(size_t(cap)*HD);c.compq=a.alloc<uint8_t>(size_t(cap)*CQ_ROW);
        c.kvs=a.alloc<float>(2*HD);c.scs=a.alloc<float>(2*HD);
        ca=a.alloc<uint8_t>(size_t(cap)*ds41i::CA_ROW);ica=a.alloc<uint8_t>(size_t(cap)*ds41i::ICA_ROW);
        pre=a.alloc<bf16>(size_t(M)*HD);shadow=a.alloc<bf16>(size_t(cap)*HD);
    }
};
void index_rows(Stage& x,const Weights& w,const bf16* ckv1,const bf16* wk,const bf16* kn,
                const int* pos,const int* pages,int cap,int M,int mode) {
    const auto inv=ds41i::rope_inv_ds41();
    if(mode==1) {
        ds41i::CaR1P p{};p.y=ckv1;p.norm_w=w.c_norm;p.inv=inv;p.dpos=pos;
        p.ca=ds41i::ca_map(x.ca,pages,cap,size_t(cap)*ds41i::CA_ROW);p.pre=x.pre;p.shadow=x.shadow;
        ds41i::launch_ca_row_r1(p,M,0);
    }
    ds41i::IcaP p{};p.pre=mode==1?x.pre:x.s.cnorm;p.wk=wk;p.knorm_w=kn;p.inv=inv;
    p.ratio=mode==1?1:2;p.dpos=pos;p.ica=ds41i::ica_map(x.ica,pages,cap,size_t(cap)*ds41i::ICA_ROW);
    ds41i::launch_ica_row(p,M,0);
}
float fp4_value(uint8_t c) {
    constexpr float values[]={0,.5,1,1.5,2,3,4,6};return (c&8?-1:1)*values[c&7];
}
std::vector<float> unpack_cache(const uint8_t* p,int cap,bool ica) {
    const int D=ica?128:512;auto raw=v2test::download(p,size_t(cap)*(ica?68:288));
    std::vector<float> out(size_t(cap)*D);
    for(int r=0;r<cap;++r)for(int k=0;k<D;++k) {
        uint8_t code=raw[ica?size_t(r)*64+k/2:size_t(r)*288+k/2];code=(code>>(4*(k&1)))&15;
        const uint8_t sb=raw[ica?size_t(cap)*64+r*4+k/32:size_t(r)*288+256+k/16];
        float scale;
        if(ica)scale=std::ldexp(1.f,int(sb)-127);
        else {__nv_fp8_e4m3 x;x.__x=sb;scale=float(x);}
        out[size_t(r)*D+k]=fp4_value(code)*scale;
    }
    return out;
}
void packed_check(const char* label,const uint8_t* got,const uint8_t* ref,int cap,bool ica,bool exact) {
    byte_check(label,got,ref,size_t(cap)*(ica?68:288),exact);
    close(label,unpack_cache(got,cap,ica),unpack_cache(ref,cap,ica),exact?0:.02f);
}
void stage_test(int M,int start,int mode,bool tc,int mask) {
    Arena a; rng=1701; const int heads=16,Q=(M+3)/4,P=4*Q,cap=start+M+2,ring=M+WIN-1;
    // GV records and real launch hooks are present, including small-owner cases.
    g_use_tc=tc;g_use_gv=true;g_fn_xfrag=gv_xfrag_launch;g_fn_gemv=gv_gemv_launch;g_src_fast=3;
    Weights w{};w.wq_a=weight(a,QA,ds41a::HID);w.wkv=weight(a,HD,ds41a::HID);w.wq_b=weight(a,heads*HD,QA);
    records(a,w.wq_a);records(a,w.wkv);records(a,w.wq_b);
    auto* cw=a.alloc<bf16>(size_t(HD)*ds41a::HID);auto* cg=a.alloc<bf16>(size_t(HD)*ds41a::HID);
    fill(cw,size_t(HD)*ds41a::HID,.02f);fill(cg,size_t(HD)*ds41a::HID,.02f);w.c_wkv=cw;w.c_wgate=cg;
    w.q_norm=ones(a,QA);w.kv_norm=ones(a,HD);w.c_norm=ones(a,HD);
    auto inv=ds41i::rope_inv_ds41();auto* d_inv=a.alloc<float>(32);
    CK(cudaMemcpy(d_inv,inv.v,128,cudaMemcpyHostToDevice));w.inv=d_inv;w.sink=a.alloc<float>(heads);
    auto* wk=a.alloc<bf16>(128*HD);fill(wk,128*HD,.02f);auto* kn=ones(a,128);
    auto* y=a.alloc<bf16>(size_t(M)*ds41a::HID);fill(y,size_t(M)*ds41a::HID);
    auto* pos=a.alloc<int>(M);std::vector<int> hp(M);for(int t=0;t<M;++t)hp[t]=start+t;
    CK(cudaMemcpy(pos,hp.data(),M*4,cudaMemcpyHostToDevice));auto* pages=a.alloc<int>(1);
    v2test::Inputs immutable;immutable.add(y,size_t(M)*ds41a::HID*2);immutable.add(cw,size_t(HD)*ds41a::HID*2);
    immutable.add(cg,size_t(HD)*ds41a::HID*2);immutable.add(wk,128*HD*2);immutable.add(pos,M*4);
    for(auto ww:{w.wq_a,w.wq_b,w.wkv}) {
        immutable.add(ww.w,size_t(ww.N)*ww.K);immutable.add(ww.s,size_t(ww.N)*ww.K/32);
        immutable.add(ww.gv,size_t(ww.N/16)*(ww.K/32)*528);
    }
    Stage ref(a,M,heads,cap,ring),got(a,M,heads,cap,ring);
    // Prior-chunk ring and compressor carry must be nonzero and identical.
    auto history=fill(ref.c.ring,size_t(ring)*HD,.2f);
    CK(cudaMemcpy(got.c.ring,history.data(),history.size()*2,cudaMemcpyHostToDevice));
    for(auto pp:{std::pair<float*,float*>{ref.c.kvs,got.c.kvs},{ref.c.scs,got.c.scs}}) {
        auto h=fill(pp.first,2*HD,.2f);CK(cudaMemcpy(pp.second,h.data(),h.size()*4,cudaMemcpyHostToDevice));
    }
    auto own=input(a,Q,Q),in=input(a,P,0); wire::Bf16Tc selector;selector.init({mask});
    bool comp_on=false,idx_on=false;
    auto produce=[&](int r,int n) {
        if(!n)return;const auto* src=y+size_t(r)*Q*ds41a::HID;
        const bool comp=mode==2&&selector.comp(n);comp_on|=comp;
        v2::attn_proj_own(w,own,src,nullptr,nullptr,0,n,mode==2&&!comp,0);
        if(comp)CK(ds41v2tc::launch_gemm2_f32(src,ds41a::HID,cw,cg,HD,ds41a::HID,own.ckv,own.csc,HD,n,0));
        if(mode==1) {
            const bool idx=selector.idx(n);idx_on|=idx;
            if(idx)CK(ds41v2tc::launch_gemm_bf16(src,ds41a::HID,cw,HD,ds41a::HID,own.ckv1,HD,n,0));
            else ds41i::launch_gemv_bf16(src,ds41a::HID,cw,HD,ds41a::HID,own.ckv1,HD,n,0);
        }
    };
    gather(a,own,in,M,mode,produce,true);
    if(mode==1) {
        auto normal=v2test::download(in.ckv1,size_t(M)*HD*2);
        gather(a,own,in,M,3,produce,true);
        require(bit_diff(v2test::download(in.ckv1,size_t(M)*HD*2),normal)==0,"L20 hoist uses own ckv1");
        printf("HOIST_PASS M=%d bf16_tc=%d bit_mismatches=0\n",M,mask);
    }
    // Full-M stock path; never provides any candidate producer input.
    attn_proj(w,ref.c,ref.s,y,M,0,heads,0,pos);
    v2::attn_proj_rest(w,got.c,got.s,in,M,heads,0,pos);
    // Only the explicitly exercised GV-vs-GEMM family boundary is non-exact.
    const bool family=(M==17||M==33);const bool projection_exact=!family;
    numeric("qa",in.qa,ref.s.qa,size_t(M)*QA,projection_exact);
    numeric("kv0",in.kv0,ref.s.kv0,size_t(M)*HD,projection_exact);
    numeric("qr",got.s.qr,ref.s.qr,size_t(M)*QA,projection_exact);
    numeric("q",got.s.q,ref.s.q,size_t(M)*heads*HD,projection_exact);
    numeric("kvrows",got.s.kvrows,ref.s.kvrows,size_t(M)*HD,projection_exact);
    numeric("ring",got.c.ring,ref.c.ring,size_t(ring)*HD,projection_exact);
    bool exact_cache=true;
    if(mode==2) {
        // Pool twice from identical incoming carry: BF16 and FP4 destinations.
        auto* rq=ref.c.compq;auto* gq=got.c.compq;ref.c.compq=got.c.compq=nullptr;
        auto rk=v2test::download(ref.c.kvs,2*HD*4),rs=v2test::download(ref.c.scs,2*HD*4);
        comp_row(w,ref.c,ref.s,y,M,0,0,pos);v2::comp_row_rest(w,got.c,got.s,in,M,0,0,pos);
        exact_cache=!comp_on;
        numeric("ckv",in.ckv,ref.s.ckv,size_t(M)*HD,exact_cache,2e-5f);
        numeric("csc",in.csc,ref.s.csc,size_t(M)*HD,exact_cache,2e-5f);
        numeric("comp",got.c.comp,ref.c.comp,size_t(cap)*HD,exact_cache,.02f);
        numeric("kvs",got.c.kvs,ref.c.kvs,2*HD,exact_cache,2e-5f);
        numeric("scs",got.c.scs,ref.c.scs,2*HD,exact_cache,2e-5f);
        const int completed=(M+(start&1))/2;
        numeric("cnorm",got.s.cnorm,ref.s.cnorm,size_t(completed)*HD,exact_cache);
        ref.c.compq=rq;got.c.compq=gq;
        for(auto* p:{ref.c.kvs,got.c.kvs})v2test::upload(p,rk);
        for(auto* p:{ref.c.scs,got.c.scs})v2test::upload(p,rs);
        auto* rb=ref.c.comp;auto* gb=got.c.comp;ref.c.comp=got.c.comp=nullptr;
        comp_row(w,ref.c,ref.s,y,M,0,0,pos);v2::comp_row_rest(w,got.c,got.s,in,M,0,0,pos);
        packed_check("compq",got.c.compq,ref.c.compq,cap,false,exact_cache);
        ref.c.comp=rb;got.c.comp=gb;
        index_rows(ref,w,nullptr,wk,kn,pos,pages,cap,M,2);index_rows(got,w,nullptr,wk,kn,pos,pages,cap,M,2);
    } else if(mode==1) {
        auto* c1=a.alloc<bf16>(size_t(M)*HD);
        ds41i::launch_gemv_bf16(y,ds41a::HID,cw,HD,ds41a::HID,c1,HD,M,0);exact_cache=!idx_on;
        numeric("ckv1",in.ckv1,c1,size_t(M)*HD,exact_cache);
        index_rows(ref,w,c1,wk,kn,pos,pages,cap,M,1);index_rows(got,w,in.ckv1,wk,kn,pos,pages,cap,M,1);
        packed_check("ca",got.ca,ref.ca,cap,false,exact_cache);
        numeric("shadow",got.shadow,ref.shadow,size_t(cap)*HD,exact_cache,.02f);
        numeric("pre",got.pre,ref.pre,size_t(M)*HD,exact_cache);
    }
    if(mode)packed_check("ica",got.ica,ref.ica,cap,true,exact_cache);
    // Same visible causal key IDs on both sides, including the prior chunk.
    auto* topk=a.alloc<int>(size_t(M)*512);std::vector<int> ht(size_t(M)*512,-1);
    for(int t=0;t<M;++t)for(int j=0;j<std::min(512,(start+t+1)/(mode==1?1:2));++j)ht[size_t(t)*512+j]=j;
    CK(cudaMemcpy(topk,ht.data(),ht.size()*4,cudaMemcpyHostToDevice));
    for(int kind=0;kind<3;++kind) {
        g_attn_cpa=kind!=0;g_attn_pipe=kind==2;
        ref.c.src_q=got.c.src_q=mode!=0;ref.c.src_ss=got.c.src_ss=size_t(cap)*CQ_ROW;
        const auto* rc=mode==1?ref.ca:ref.c.compq;const auto* gc=mode==1?got.ca:got.c.compq;
        CK(cudaMemset(ref.s.o,0xff,size_t(M)*heads*HD*2));CK(cudaMemset(got.s.o,0xfe,size_t(M)*heads*HD*2));
        attn_core(w,ref.c,(const bf16*)rc,ref.s,M,0,heads,topk,512,mode?512:0,0,pos);
        attn_core(w,got.c,(const bf16*)gc,got.s,M,0,heads,topk,512,mode?512:0,0,pos);
        numeric(kind==2?"o_pipe":kind==1?"o_cpa":"o",got.s.o,ref.s.o,size_t(M)*heads*HD,projection_exact&&exact_cache,.02f);
    }
    g_attn_pipe=false;immutable.verify();
    printf("STAGE_PASS M=%d start=%d mode=%d tc=%d bf16_tc=%d comp_tc=%d idx_tc=%d gv_boundary=%d wire_bit_mismatches=0\n",
           M,start,mode,tc,mask,comp_on,idx_on,family);
}
