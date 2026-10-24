// Flash-only main-model residual and layer loop. Included by rank.cuh inside g6rt,
// after its legacy helpers. The MTP model does not use this four-stream path.
#pragma once
#if G6_MODEL_FLASH
inline flash_mhc::Buffers mhc_buffers(Model& m) {
    flash_mhc::Buffers b{m.resid4,m.hc_inv,m.hc_mix,m.hc_pre,m.hc_post,m.hc_comb,m.hc_done};
    b.bf16=flash_mhc::bf16_env()==3;   // un-sided launches: only when both sides round
    return b;
}
inline flash_mhc::Weights mhc_weights(const Find& find,int layer,const char* block) {
    const std::string p=lname(layer,(std::string("hc_")+block+"_").c_str());
    auto get=[&](const std::string& n,bool bf16,std::initializer_list<long long> shape) {
        TensorRef t=find(n);
        if (!t.p || t.nd!=(int)shape.size() || (bf16 ? !t.bf16 : !t.fp32)) die("bad mHC dtype/shape: "+n);
        int i=0; for (long long s:shape) if(t.shape[i++]!=s) die("bad mHC shape: "+n);
        return t.p;
    };
    const auto fn=(const __nv_bfloat16*)get(p+"fn",true,{geom::HC_MIX,geom::HC*HID});
    const auto base=(const float*)get(p+"base",false,{geom::HC_MIX});
    const auto scale=(const float*)get(p+"scale",false,{3});
    const auto norm=(const __nv_bfloat16*)get(lname(layer,!strcmp(block,"attn")?"attn_norm":"mlp_norm"),true,{HID});
    return {fn,base,scale,norm};
}
// W3/W4/W5 allocate and bind outside graph capture. Every main-model layer
// owns an attention error word, including KDA layers (which leave it zero).
inline void build_flash(Model& m,int rank,const std::vector<int>& ids,const Find& find,int capacity,
                         const g6::FabricDev& f,u64* stepp,cudaStream_t st,int max_rows) {
    if(ids.empty() || capacity<1 || rank<0 || rank>=g6nr::NR || f.nr!=g6nr::NR)
        die("Flash build: invalid layers/capacity/rank/fabric");
    m.flash_draft=ids.size()==1 && ids[0]==geom::MTP_LAYER;
    for(int l:ids) if(!m.flash_draft && (l<0 || l>=geom::L_MAIN)) die("Flash main model layer outside 0..44");
#if defined(G6_CP) && G6_CP
    if(m.cp.on) die("Flash CP is not implemented");
#endif
    if(max_rows>MX) die("Flash main model currently supports decode rows only (no prefill)");
    m.rank=rank; m.nl=(int)ids.size(); m.f=f; m.stepp=stepp;
    if(f.ncoll!=m.ncoll()) die("Flash fabric ncoll must be 2 * layers + 1");
    const char* be=getenv("G6_DEC_BK");
    m.nbk=context_buckets(m.bk,be?atoi(be):2048,capacity,0);
#ifdef DEC_S2
    m.s2_run_pages=(capacity+63)/64;
#endif
    m.cells=dmalloc<G6AttnStepCell>(m.nl); m.shared=dmalloc<int>(m.nl);
    m.errs=dmalloc<unsigned*>(m.nl); m.tickets=dmalloc<unsigned>(m.ncoll());
    m.resid=dmalloc<float>((size_t)MX*HID);
    if(!m.flash_draft) m.resid4=dmalloc<float>((size_t)MX*geom::HC*HID);
    m.hc_inv=dmalloc<float>(MX); m.hc_mix=dmalloc<float>(MX*geom::HC_MIX);
    m.hc_pre=dmalloc<float>(MX*geom::HC); m.hc_post=dmalloc<float>(MX*geom::HC);
    m.hc_comb=dmalloc<float>(MX*geom::HC*geom::HC); m.hc_done=dmalloc<unsigned>(64);
    G6RT_CK(cudaMemsetAsync(m.hc_done,0,64*sizeof(unsigned),st));
    m.x=dmalloc<__half>((size_t)MX*HID); m.x2=dmalloc<__half>((size_t)MX*HID);
    m.xn=dmalloc<float>((size_t)MX*HID); m.xn2=dmalloc<float>((size_t)MX*HID);
    m.row_slot=dmalloc<int>(MX); m.kda_src=dmalloc<int>(MX);
    m.q_a=dmalloc<__half>((size_t)MX*QA); m.q_a_n=dmalloc<__half>((size_t)MX*QA);
    m.kv_a=dmalloc<__half>((size_t)MX*KVA); m.q_b=dmalloc<__half>((size_t)MX*g6nr::QBW);
    m.idx_q=dmalloc<__half>((size_t)MX*IDXQ); m.idx_k=dmalloc<__half>((size_t)MX*IDXK);
    m.idx_w=dmalloc<__half>((size_t)MX*IDXW); m.idx_g=dmalloc<__half>((size_t)MX*geom::IDX_DIM);
    m.logits=dmalloc<float>((size_t)MX*geom::NEXP); m.plan=dmalloc<glue::RoutePlan>(1);
    m.srow=dmalloc<glue::SampleRow>(MX); m.tk=dmalloc<glue::TkRes>(1); m.topp_miss=dmalloc<unsigned>(1);
    m.tok=dmalloc<int>(MX); m.pos=dmalloc<int>(MX); m.tok_out=dmalloc<int>(MX); m.tgt=dmalloc<int>(MX);
    m.steering_scale=dmalloc<float>(MX);   // k_step_in copies StepIn::steering_scale here every step
    m.lp=dmalloc<float>(MX); m.lpt=dmalloc<float>(MX);
    m.scratch=dmalloc<char>(bind::scratch_bytes()); m.sc=bind::scratch_at(m.scratch);
    m.flash_mlp_sc=flash_mlp::workspace_at(dmalloc<char>(flash_mlp::workspace_bytes(MX)),MX);
    bool any_kda=false;
    for(int l:ids) any_kda|=geom::is_kda(l);
    if(any_kda) {
        if(const char* e=kda::build_state(m.kda_st,std::max(1,m.kda_ring)*std::max(1,m.kda_sessions),g6nr::NR)) die(e);
        if(m.kda_ring>1) m.kda_st.src=m.kda_src;
        if(const char* e=kda::build_scratch(m.kda_sc,MX,g6nr::NR)) die(e);
    }
    m.layers=std::vector<Layer>(m.nl);
    std::vector<unsigned*> errors(m.nl);
    for(int i=0;i<m.nl;++i) {
        Layer& y=m.layers[i]; y.l=ids[i]; y.kda=geom::is_kda(y.l); y.moe=geom::is_moe(y.l); y.indexer=!y.kda;
        if(!m.flash_draft) {
        y.hc[0]=mhc_weights(find,y.l,"attn"); y.hc[1]=mhc_weights(find,y.l,"ffn");
        y.attn_norm=y.hc[0].norm; y.mlp_norm=y.hc[1].norm;
        } else {
            y.attn_norm=(const __nv_bfloat16*)need(find,lname(y.l,"attn_norm"));
            y.mlp_norm=(const __nv_bfloat16*)need(find,lname(y.l,"mlp_norm"));
        }
        if(y.kda) {
            if(const char* e=kda::bind_kda(y.kda_w,y.l,find,rank,g6nr::NR)) die(e);
            y.ab.args.error=dmalloc<unsigned>(1);
        } else bind_dsa_layer(m,i,find,capacity,st);
        if(const char* e=flash_mlp::load(y.l,find,y.flash_mlp_w)) die(e);
        if(y.flash_mlp_w.moe!=y.moe) die("Flash MLP format disagrees with layer schedule");
        if(y.moe && !find(lname(y.l,"router.bias")).fp32) die("Flash router.bias must be FP32");
        y.T=y.flash_mlp_w.experts;
        errors[i]=y.ab.args.error;
        check_headroom(rank,capacity,false);
    }
    G6RT_CK(bind::init());
    G6RT_CK(g6h2d::copy(m.errs,errors.data(),errors.size()*sizeof(unsigned*)));
    auto emb=find("G.embed"),norm=find("G.final_norm"),head=find("G.lm_head.r8.gv"),hscale=find("G.lm_head.scale");
    if(!head.p) head=find("G.lm_head.mx.gv");
    if(!head.p) head=find("G.lm_head.gv");
    if(!emb.p || !emb.bf16 || emb.nd!=2 || emb.shape[0]!=geom::VOCAB || emb.shape[1]!=HID ||
       !norm.p || !norm.bf16 || norm.nd!=1 || norm.shape[0]!=HID || !head.p) die("Flash vocabulary tensor dtype/shape");
    if(!bind::vocab_slice(rank,m.vlo,m.vn)) die("Flash bad vocab rank");
    if(hscale.p && (!hscale.fp32 || hscale.nd!=1 || hscale.shape[0]!=m.vn)) die("Flash G.lm_head.scale dtype/shape");
    m.emb=(const __nv_bfloat16*)emb.p; m.final_norm=(const __nv_bfloat16*)norm.p;
    m.head_t.lm_head=as_bind(head);
    m.head_t.lm_head_scale=as_bind(hscale);
    m.hlog=dmalloc<float>((size_t)MX*m.vn);
    check_headroom(rank,capacity,true);
}
inline void prepare_flash(Model& m,int M,cudaStream_t st) {
    if(M<1 || M>MX) die("Flash prepare: M outside 1..MX");
    head_for(m,M);
    for(int i=0;i<m.nl;++i) {
        Layer& y=m.layers[i];
        if(!y.kda) prepare_dsa_layer(m,i,M,st);
        if(!y.flash_mlp[M].M) {
            if(const char* e=flash_mlp::make(y.flash_mlp_w,M,m.xn2,m.sc.part[1],m.plan,m.flash_mlp_sc,m.sc,y.flash_mlp[M],-1,m.x2)) die(e);
            y.flash_mlp[M].topk=m.flash_draft ? geom::TOPK : moe_topk_cut(geom::TOPK);   // G6_MOE_TOPK: main model only, the drafter keeps the model's top-k
#ifdef DEC_SLOT
            const dslot::Slot mlp_slot=make_flash_slot(m,2*i+1);
            y.flash_mlp[M].k1.a.slot=mlp_slot;
            y.flash_mlp[M].down_t.a.slot=mlp_slot;
#endif
        }
    }
}
inline void launch_mhc_input(Model& m,int i,bool ffn,int M,cudaStream_t st) {
    flash_mhc::prepare(mhc_buffers(m),m.layers[i].hc[ffn?1:0],M,HID,geom::RMS_EPS_FLASH,
                       ffn?m.x2:m.x,st,ffn?m.xn2:m.xn,
                       ffn?m.sc.xf_x2:m.sc.xf_x,ffn?m.sc.xb_x2:m.sc.xb_x);
}
inline void launch_embed_flash(Model& m,int M,cudaStream_t st) {
    flash_mhc::k_mhc_embed<<<M,flash_mhc::THREADS,0,st>>>(mhc_buffers(m),m.tok,m.emb,HID,geom::RMS_EPS_FLASH);
    launch_mhc_input(m,0,false,M,st);
}
// G6_PF_XTRACE=1 (prefill measurement only): one record of PX_W u64 per (step % PX_STEPS, collective) in device memory.
// globaltimer ns: [PX_PROD] producer end (before the put), [PX_PUT] put + doorbell done, [PX_ENTRY] first consumer CTA
// entry, [PX_WMIN]/[PX_WMAX] first/last consumer CTA past the wait, [PX_END] mHC end; [PX_M] rows, [PX_STEP] step + 1.
namespace pxt {
constexpr int PX_STEPS = 8192, PX_W = 8;
enum { PX_PROD = 0, PX_PUT, PX_ENTRY, PX_WMIN, PX_WMAX, PX_END, PX_M, PX_STEP };
__device__ __forceinline__ unsigned long long now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned long long* rec(unsigned long long* p, u64 step, int c) {
    return p + ((size_t)(step % PX_STEPS) * 2 + c) * PX_W;
}
}
// BF16 loads are volatile with respect to system-memory DMA. wait_peers can run
// either as a standalone launch or fused inside k_mhc_out_inv / k_mhc_consume_fused.
struct MhcFabricBlock {
    g6::FabricDev f;
    const u64* stepp;
    int c;
    size_t wait_payload=0;
    const __nv_bfloat16* local_part=nullptr;
    unsigned long long* pxt=nullptr;      // G6_PF_XTRACE records (prefill), null = off
    const __nv_bfloat16* peer_part=nullptr; // NR == 2 prefill: the peer's partial already copied to device memory
    unsigned long long* xr=nullptr;
    const unsigned short* base[g6nr::NR]={};
    __device__ __forceinline__ u64 init() {
        const u64 step=*stepp;
        if(pxt) {
            xr=pxt::rec(pxt,step,c);
            if(!threadIdx.x) atomicMin(xr+pxt::PX_ENTRY,pxt::now());
        }
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p)
            base[p]=(local_part && p==f.rank) ? reinterpret_cast<const unsigned short*>(local_part)
                  : (peer_part && p!=f.rank) ? reinterpret_cast<const unsigned short*>(peer_part)
                                              : (const unsigned short*)glue::peer_slot(f,step,c,p);
        return step;
    }
    __device__ __forceinline__ void wait(u64 step) const {
        if(wait_payload)
            glue::wait_peers(f,step,c,g6::coll_seq(step,c,f.ncoll),wait_payload,glue::ERR_AR);
        if(xr && !threadIdx.x) {
            const unsigned long long t=pxt::now();
            atomicMin(xr+pxt::PX_WMIN,t);
            atomicMax(xr+pxt::PX_WMAX,t);
        }
    }
    __device__ __forceinline__ unsigned seq32(u64 step) const {
        return (unsigned)g6::coll_seq(step,c,f.ncoll);
    }
    __device__ __forceinline__ float operator()(size_t k) const {
        unsigned short v[g6nr::NR];
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p)
            v[p]=((local_part && p==f.rank) || (peer_part && p!=f.rank)) ? __ldg(base[p]+k) : __ldcv(base[p]+k);
        float sum=0.f;
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p) sum+=__uint_as_float((unsigned)v[p]<<16);
        return sum;
    }
};
// Async prefill only. Keep MhcFabricBlock unchanged so the OFF kernel type and
// its parameter layout stay identical. Every peer has its own device copy.
struct MhcFabricPeerBlock : MhcFabricBlock {
    const __nv_bfloat16* peers[g6nr::NR] = {};
    __device__ __forceinline__ u64 init() {
        const u64 step = MhcFabricBlock::init();
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p)
            if (p != f.rank && peers[p])
                base[p] = reinterpret_cast<const unsigned short*>(peers[p]);
        return step;
    }
    __device__ __forceinline__ float operator()(size_t k) const {
        float sum = 0.f;
        #pragma unroll
        for (int p = 0; p < g6nr::NR; ++p) {
            const unsigned short v = ((local_part && p == f.rank) || (p != f.rank && peers[p]))
                ? __ldg(base[p] + k) : __ldcv(base[p] + k);
            sum += __uint_as_float((unsigned)v << 16);
        }
        return sum;
    }
};
// Decode-only pinned-partial stage. The base wait still establishes peer visibility;
// no slot is read until that wait returns. CTA 0 alone calls wait in consume_fused.
// Keep each BF16 word intact and retain the base rank-order FP32 summation.
struct MhcFabricStagedBlock : MhcFabricBlock {
    const unsigned short* staged=nullptr;
    __device__ __forceinline__ void wait(u64 step) {
        MhcFabricBlock::wait(step);
        __shared__ uint4 slots[g6nr::NR][512];
        const int m=blockIdx.y;
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p) {
            const uint4* src=reinterpret_cast<const uint4*>(base[p]+(size_t)m*4096);
            #pragma unroll
            for(int j=0;j<2;++j) {
                const int k=threadIdx.x+j*flash_mhc::THREADS;
                slots[p][k]=__ldcv(src+k);
            }
        }
        __syncthreads();
        staged=reinterpret_cast<const unsigned short*>(slots);
    }
    __device__ __forceinline__ float operator()(size_t k) const {
        const int h=k%4096;
        float sum=0.f;
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p)
            sum+=__uint_as_float((unsigned)staged[p*4096+h]<<16);
        return sum;
    }
};
// marks from the Flash-only kernels (kda.cuh, flash_mlp.cuh) -> DecStage (order = FlashMark in flash_mark.h)
inline void flash_mark(int fmk,void* p) {
    static const int dp[]={DP_K_PROJ,DP_K_PREP,DP_K_STEP,DP_K_OUT,DP_K_OPROJ,DP_M_ROUTER,DP_M_PLAN,DP_M_XHALF,
                           DP_M_XFRAG,DP_M_FC1,DP_M_FC2,DP_M_FIN,DP_D_GATE,DP_D_DOWN,DP_D_OUT};
    if(fmk<0 || fmk>=(int)(sizeof dp/sizeof dp[0])) return;
    dmark(dp[fmk],(cudaStream_t)p);
}
inline void launch_consume_mhc(Model& m,int c,int M,cudaStream_t st) {
    static const int mhc_fuse_m1 = getenv("G6_MHC_FUSE_M1") ? atoi(getenv("G6_MHC_FUSE_M1")) : (getenv("G6_MHC_FUSE") ? atoi(getenv("G6_MHC_FUSE")) : 2);
    static const int mhc_fuse_m4 = getenv("G6_MHC_FUSE_M4") ? atoi(getenv("G6_MHC_FUSE_M4")) : (getenv("G6_MHC_FUSE") ? atoi(getenv("G6_MHC_FUSE")) : 2);
    const int mhc_fuse = M == 1 ? mhc_fuse_m1 : mhc_fuse_m4;
    static const bool stage_on = !getenv("G6_DEC_MHC_STAGE") || atoi(getenv("G6_DEC_MHC_STAGE"));
    const bool stage = stage_on && (M == 1 || M == 4);
    const size_t payload = (size_t)M*HID*2;
    dmark(DP_OTHER,st);
    if (mhc_fuse == 0)
        glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,payload,glue::ERR_AR);
    dmark(c%2==0?DP_WAIT_A:DP_WAIT_M,st);
    const size_t wp = mhc_fuse ? payload : 0;
    const int i=c/2;
    if (!(c&1)) {
        if (mhc_fuse == 2 && HID == 4096) {
            dmark(DP_MHC_OI,st);
            if(stage) {
            flash_mhc::k_mhc_consume_fused<<<dim3(geom::HC_MIX,M),flash_mhc::THREADS,0,st>>>(
                flash_mhc::side(mhc_buffers(m),c&1),MhcFabricStagedBlock{MhcFabricBlock{m.f,m.stepp,c,wp}},m.layers[i].hc[1],HID,geom::RMS_EPS_FLASH,
                m.x2,m.xn2,m.sc.xf_x2,m.sc.xb_x2);
            } else {
            flash_mhc::k_mhc_consume_fused<<<dim3(geom::HC_MIX,M),flash_mhc::THREADS,0,st>>>(
                flash_mhc::side(mhc_buffers(m),c&1),MhcFabricBlock{m.f,m.stepp,c,wp},m.layers[i].hc[1],HID,geom::RMS_EPS_FLASH,
                m.x2,m.xn2,m.sc.xf_x2,m.sc.xb_x2);
            }
        } else {
            flash_mhc::k_mhc_out_inv<<<M,flash_mhc::THREADS,0,st>>>(flash_mhc::side(mhc_buffers(m),c&1),MhcFabricBlock{m.f,m.stepp,c,wp},HID,geom::RMS_EPS_FLASH);
            dmark(DP_MHC_OI,st);
            launch_mhc_input(m,i,true,M,st);
        }
    } else if(i+1<m.nl) {
        if (mhc_fuse == 2 && HID == 4096) {
            dmark(DP_MHC_OI,st);
            if(stage) {
            flash_mhc::k_mhc_consume_fused<<<dim3(geom::HC_MIX,M),flash_mhc::THREADS,0,st>>>(
                flash_mhc::side(mhc_buffers(m),c&1),MhcFabricStagedBlock{MhcFabricBlock{m.f,m.stepp,c,wp}},m.layers[i+1].hc[0],HID,geom::RMS_EPS_FLASH,
                m.x,m.xn,m.sc.xf_x,m.sc.xb_x);
            } else {
            flash_mhc::k_mhc_consume_fused<<<dim3(geom::HC_MIX,M),flash_mhc::THREADS,0,st>>>(
                flash_mhc::side(mhc_buffers(m),c&1),MhcFabricBlock{m.f,m.stepp,c,wp},m.layers[i+1].hc[0],HID,geom::RMS_EPS_FLASH,
                m.x,m.xn,m.sc.xf_x,m.sc.xb_x);
            }
        } else {
            flash_mhc::k_mhc_out_inv<<<M,flash_mhc::THREADS,0,st>>>(flash_mhc::side(mhc_buffers(m),c&1),MhcFabricBlock{m.f,m.stepp,c,wp},HID,geom::RMS_EPS_FLASH);
            dmark(DP_MHC_OI,st);
            launch_mhc_input(m,i+1,false,M,st);
        }
    } else {
        dmark(DP_MHC_OI,st);
        const int hcodec=head_for(m,M).lm_head.codec;
        const bool f16_in=hcodec==bind::MXFP8 || hcodec==bind::FP8R;
        flash_mhc::k_mhc_out_inv_final<<<M,flash_mhc::THREADS,0,st>>>(
            flash_mhc::side(mhc_buffers(m),c&1),MhcFabricBlock{m.f,m.stepp,c,wp},m.final_norm,HID,geom::RMS_EPS_FLASH,m.x,m.xn,m.resid,
            f16_in?m.sc.xf_head:nullptr,f16_in?nullptr:m.sc.xf_head);
    }
    dmark(c%2==0?DP_NORM_A:DP_NORM_M,st);
}
// MTP has one FP32 residual, no mHC. Rank-ordered BF16 partials match the main collectives.
inline void launch_consume_flash_draft(Model& m,int c,int M,cudaStream_t st) {
    glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,(size_t)M*HID*2,glue::ERR_AR);
    flash_mtp::k_add<<<(M*HID+255)/256,256,0,st>>>(m.resid,MhcFabricBlock{m.f,m.stepp,c},M*HID);
    const bool last=c==1;
    flash_mtp::norm(m.resid,last?m.final_norm:m.layers[0].mlp_norm,last?m.xn:m.xn2,last?m.x:m.x2,M,st);
    if(!last) flash_mhc::fragments(m.xn2,M,HID,m.sc.xf_x2,m.sc.xb_x2,st);
}
inline void launch_kda_block(Model& m,int i,int M,cudaStream_t st) {
    dmark(DP_OTHER,st);
#ifdef DEC_SLOT
    const bool use_slot=m.layers[i].kda_w.ot.w!=nullptr;
    const dslot::Slot slot=make_flash_slot(m,2*i);
    kda::launch(m.layers[i].kda_w,m.kda_st,m.kda_sc,m.sc.xb_x,m.row_slot,M,m.sc.part[0],st,gv::O_BF16,m.x,m.sc.xf_x,&m.sc.dt,use_slot?&slot:nullptr);
    dmark(DP_ATTN,st);
    if(use_slot) slot_publish(m,2*i,st);
    else put_publish(m,m.sc.part[0],2*i,M,st);
#else
    kda::launch(m.layers[i].kda_w,m.kda_st,m.kda_sc,m.sc.xb_x,m.row_slot,M,m.sc.part[0],st,gv::O_BF16,m.x,m.sc.xf_x,&m.sc.dt);
    dmark(DP_ATTN,st);
    put_publish(m,m.sc.part[0],2*i,M,st);
#endif
    dmark(DP_OPROJ,st);
}
inline void launch_moe_block(Model& m,int i,int M,cudaStream_t st) {
    const flash_mlp::Block& b=m.layers[i].flash_mlp[M];
    flash_mlp::launch(b,st);
    // Diagnostic only: save this layer's plan before the next MoE layer reuses m.plan.
    if(m.rlog && b.w.moe)
        G6RT_CK(cudaMemcpyAsync(m.rlog+(size_t)i*RLOG,b.plan,RLOG*sizeof(int),cudaMemcpyDeviceToHost,st));
    dmark(DP_EXPERTS,st);
#ifdef DEC_SLOT
    if(b.w.moe || b.glu_t.on) slot_publish(m,2*i+1,st);
    else put_publish(m,m.sc.part[1],2*i+1,M,st);
#else
    put_publish(m,m.sc.part[1],2*i+1,M,st);
#endif
    dmark(DP_PUT_M,st);
}
inline void launch_dense_block(Model& m,int i,int M,cudaStream_t st) { launch_moe_block(m,i,M,st); }
inline void launch_step_flash(Model& m,int M,int context,cudaStream_t st,int part,int ls,const std::function<void(int,int)>& residual_copy) {
    EvProf& ep=ev_prof();
    if(part!=1) ep.n=0;
    ep.rec=ep.every>0;
    if(ep.rec || getenv("G6_DEC_PROF")) { g6_attn_mark=attn_mark; g6_flash_mark=flash_mark; }
    if(part!=1) { dmark(DP_OTHER,st); launch_step_in(m,st); launch_embed_flash(m,M,st); if(residual_copy) residual_copy(0,M); dmark(DP_STEP,st); }
    for(int i=part==1?ls:0;i<(part==0?ls:m.nl);++i) {
        if(geom::is_kda(m.layers[i].l)) launch_kda_block(m,i,M,st);
        else launch_dsa_block(m,i,M,context,st);
        launch_consume_mhc(m,2*i,M,st);
        if(residual_copy) residual_copy(2*i+1,M);
        if(geom::is_moe(m.layers[i].l)) launch_moe_block(m,i,M,st);
        else launch_dense_block(m,i,M,st);
        launch_consume_mhc(m,2*i+1,M,st);
        if(residual_copy) residual_copy(2*i+2,M);
    }
    if(part!=0) { launch_head_local(m,M,st); launch_head_merge(m,M,st); launch_step_out(m,M,st); dmark(DP_STEP,st); }
    ep.rec=false;
}
#endif
