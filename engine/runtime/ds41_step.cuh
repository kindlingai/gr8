// DeepSeek-V4.1-Flash-1017 main-model decode step. Included by rank.cuh inside g6rt, after its helpers (the flash_step.cuh analog).
// The step is a plan (ds41_ops.h: plan() = every stage of every layer, in order, grouped into the units the m4sim wrappers
// launch) and an op table. This file fills the table's driver and mHC slots and leaves the attention, indexer, MoE and Engram slots
// to their owners (null = the step stops there with a NOTSUP message that names the stage and its owner). docs/DEEPSEEK-STEP.md.
#pragma once
#if G6_MODEL_DS41
#include "ds41_prefill_rows.h"
#include "ds41_prefill_capacity.h"
#include "ds41_slots.h"
#include "ds41_knobs.h"
extern "C" int ds41_sh_act_quant(const void*,int,uint8_t*,int,uint8_t*,int,int,int,void*);
// B4's Engram projection hook (ds41_attn_ops.cuh, included below); the Engram wiring (reduce collective, token window, op, knobs) is B11's
inline int ds41_engram_project(void* model, int e, int rows, void* stream, const __nv_bfloat16* emb, __nv_bfloat16* kv);
#include "ds41_engram_ops.cuh"
#include <map>
// BF16 partials of the NR ranks, summed in rank order as FP32. The ds_mhc kernels call init(), wait(step) and operator()(k)
// with k = row * H + h. A DS-local copy of the part of flash's MhcFabricBlock that the decode step uses (no prefill hooks).
struct Ds41Block {
    g6::FabricDev f;
    const u64* stepp;
    int c;
    size_t wait_payload=0;
    const unsigned short* base[g6nr::NR]={};
    __device__ __forceinline__ u64 init() {
        const u64 step=*stepp;
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p) base[p]=(const unsigned short*)glue::peer_slot(f,step,c,p);
        return step;
    }
    __device__ __forceinline__ void wait(u64 step) const {
        if(wait_payload) glue::wait_peers(f,step,c,g6::coll_seq(step,c,f.ncoll),wait_payload,glue::ERR_AR);
    }
    __device__ __forceinline__ float operator()(size_t k) const {
        unsigned short v[g6nr::NR];
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p) v[p]=__ldcv(base[p]+k);
        float sum=0.f;
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p) sum+=__uint_as_float((unsigned)v[p]<<16);
        return sum;
    }
};

// K=2 (GR8_DS_WAIT_K=2): the relay wait. The consume kernel starts right behind the publish kernel like with K=0, but only CTA (0,0) polls the peers'
// tails (glue::wait_peers: 3 threads on 3 host-memory lines); when they are in, it stores coll_seq into the device word `relay` (release, gpu scope).
// Every other CTA polls that word (acquire, gpu scope; one thread, then __syncthreads) instead of the three host-memory lines. The payload reads that
// follow are the unchanged __ldcv loads. No arithmetic changes. The test is `!= seq`, as the tails' own (a unique coll_seq per collective; the next consume
// kernel starts only after this one has ended, so the word cannot move on while a CTA of this kernel still reads it). Progress: CTA (0,0) is the first block of the grid, so it is resident before any CTA that waits for it; the host
// turns K=2 off (back to K=1) for GR8_DRAFT_PAR and for the draft Model (ds41::wait_mode).
__device__ __forceinline__ u64 ds41_ld_acquire_gpu(const u64* p) {
    u64 v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void ds41_st_release_gpu(u64* p, u64 v) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
struct Ds41RelayBlock : Ds41Block {
    u64* relay=nullptr;
    __device__ __forceinline__ void wait(u64 step) const {
        if(!wait_payload) return;
        const u64 seq=g6::coll_seq(step,c,f.ncoll);
        if(blockIdx.x==0 && blockIdx.y==0) {
            glue::wait_peers(f,step,c,seq,wait_payload,glue::ERR_AR);     // ends with __syncthreads; on a timeout it has set the error word
            if(threadIdx.x==0) ds41_st_release_gpu(relay,seq);            // also after a timeout, so the other CTAs leave their spin
        } else {
            if(threadIdx.x==0) {
                const long long t0=clock64();
                unsigned it=0;
                while(ds41_ld_acquire_gpu(relay)!=seq) {
                    if((++it&1023u)==0 && (*reinterpret_cast<volatile unsigned*>(f.error)!=0 || clock64()-t0>2*G6_SPIN_TIMEOUT_CYC)) {
                        atomicCAS(f.error,0u,glue::ERR_AR|((unsigned)c<<8)|(unsigned)f.rank);   // peer = this rank: the relay word never came
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }
};

#include "ds41_stage.cuh"      // F7 k16 GR8_DS_DEC_STAGE: Ds41Staged<Block> (no includes of its own; the test includes it at global scope)
#include "ds41_gather.cuh"     // F7 GR8_DS_PF_GATHER: k_ds41_gather + Ds41DevBlock (no includes of its own; the test includes it at global scope)
#include "ds41_rs.cuh"         // F10 GR8_DS_PF_RS: k_put_publish_rs, k_rs_reduce, k_rs_gather (rank.cuh includes ds41_rs.h at global scope)

inline ds_mhc::Buffers ds41_buffers(Model& m) {
    ds_mhc::Buffers b{};
    b.streams=m.resid4; b.mix=m.hc_mix; b.pre=m.hc_pre; b.post=m.hc_post; b.comb=m.hc_comb; b.done=m.hc_done; b.part=m.hc_part;
    return b;
}
inline ds_mhc::Buffers ds41_buffers_slice(Model& m,int row0) {
    auto b=ds41_buffers(m);
    b.streams+=(size_t)row0*4*HID; b.mix+=(size_t)row0*24;
    b.pre+=(size_t)row0*4; b.post+=(size_t)row0*4; b.comb+=(size_t)row0*16;
    // done/part are scratch indexed from local row zero and reused in stream order.
    return b;
}
inline Layer::Hc ds41_hc_weights(const Find& find,int layer,const char* block) {
    const std::string p=lname(layer,(std::string("hc_")+block+".").c_str());
    auto get=[&](const std::string& n,bool want_f32,std::initializer_list<long long> shape) {
        const TensorRef t=find(n);
        if(!t.p || t.nd!=(int)shape.size() || (want_f32 ? !t.fp32 : !t.fp32 && !t.bf16)) die("bad mHC dtype/shape: "+n);
        int i=0; for(long long s:shape) if(t.shape[i++]!=s) die("bad mHC shape: "+n);
        return t;
    };
    Layer::Hc h;
    const TensorRef fn=get(p+"fn",false,{geom::HC_MIX,geom::HC*HID});   // F32 in the checkpoint; BF16 is the precision knob
    h.fn=fn.p; h.fn_f32=fn.fp32;
    h.base=(const float*)get(p+"base",true,{geom::HC_MIX}).p;
    h.scale=(const float*)get(p+"scale",true,{3}).p;
    const TensorRef norm=find(lname(layer,!strcmp(block,"attn")?"attn_norm":"ffn_norm"));
    if(!norm.p || !norm.bf16 || norm.nd!=1 || norm.shape[0]!=HID) die("bad mHC norm dtype/shape: layer "+std::to_string(layer)+" "+block);
    h.norm=(const __nv_bfloat16*)norm.p;
    return h;
}

// The model, outside graph capture. Refused for now: the DSpark layers 40..42, S2, CP, Engram, GR8_DRAFT_HEAD2, prefix reuse.
inline void build_ds41(Model& m,int rank,const std::vector<int>& ids,const Find& find,int capacity,
                       const g6::FabricDev& f,u64* stepp,cudaStream_t st,int max_rows,int dspark,int ag,bool rowshard) {   // ag: F10 AGs; rowshard: B12 state collective
    if(ids.empty() || capacity<1 || rank<0 || rank>=g6nr::NR || f.nr!=g6nr::NR) die("DS41 build: invalid layers/capacity/rank/fabric");
    const bool draft=dspark==2;   // B12: the DSpark draft Model holds exactly the layers 40..42 (docs/DEEPSEEK-DSPARK.md section 3)
    if(draft) {
        if(ids!=std::vector<int>{geom::L_MAIN,geom::L_MAIN+1,geom::L_MAIN+2}) die("DS41 build: the DSpark draft Model holds the layers 40, 41, 42");
    }
    else for(int l:ids) if(l<0 || l>=geom::L_MAIN) die("DS41 build: layer "+std::to_string(l)+" is outside the main model 0.."+std::to_string(geom::L_MAIN-1)+
                                                  " (DSpark layers 40..42 are the second wave: GR8_DS_MTP=0)");
#if defined(G6_CP) && G6_CP
    if(m.cp.on) die("DS41 build: CP is not implemented");
#endif
    if(max_rows>MX) die("DS41 build: decode rows only (no prefill) -- max_rows above MX");
    // W is a separate future prefill row budget. Decode/verify graphs stay MX.
    // Until the prefill hook is wired, this only widens caller-owned buffers;
    // prepare_ds41 still refuses M>MX. The attention and indexer row budgets
    // must match before a W-row hook can use any of them.
    const int prefill_rows=ds41_prefill_rowcap(DS41_TEXT(knob::ds_prefill_w()),MX);
    if(!prefill_rows) die("DS41 build: GR8_DS_PREFILL_W must be MX..4096");
    if(prefill_rows>MX) {
        const int aw=std::max<int>(MX,(int)knob::ds_attn_pw()),iw=std::max<int>(MX,(int)knob::ds41_idx_pw().value_or(MX));
        if(aw!=prefill_rows || iw!=prefill_rows)
            die("DS41 build: GR8_DS_PREFILL_W requires matching GR8_DS_ATTN_PW and GR8_DS41_IDX_PW");
    }
    if(knob::draft_head2()) die("DS41 build: GR8_DRAFT_HEAD2 is a drafter feature");
    const ds41::engram::Knobs ek=ds41_engram_knobs();   // GR8_DS_ENGRAM (default 0); off reads no other Engram knob
    const bool engram_on=ek.on && !draft;               // the DSpark draft block (layers 40..42) carries no Engram layer
    m.ds41=true; m.ds41_draft=draft;
    m.rank=rank; m.nl=(int)ids.size(); m.f=f; m.stepp=stepp;
    m.ecoll=draft?0:ds41_engram_colls(ids);             // 0 off; 2 on (also checks that layers 1 and 14 are served)
    if(const char* e=::ds41::rs::fabric_ncoll_error(f.ncoll,m.ncoll(),ag,draft,rowshard)) die(e);
    // slots of prefill_rows * 6144 * 2 B + the doorbell word: MX rows in the decode build, W rows with GR8_DS_PREFILL_W (B1's slot_bytes; the engram
    // collective is wider than the hidden one: 12,288 B per row against 10,240)
    if(engram_on && g6::fab_slot_stride(f,2*m.nl+1)<(size_t)prefill_rows*ds41::engram::kEmbDim*2+8) die("DS41 Engram: the fabric slots are smaller than prefill rows (MX or GR8_DS_PREFILL_W) * 6144 * 2 bytes + 8");
    m.nbk=context_buckets(m.bk,(int)knob::dec_bk(),capacity,0);
#ifdef DEC_S2
    m.s2_run_pages=(capacity+63)/64;
#endif
    m.cells=dmalloc<G6AttnStepCell>(m.nl); m.shared=dmalloc<int>(m.nl);
    G6RT_CK(cudaMemsetAsync(m.shared,0,(size_t)m.nl*sizeof(int),st)); // DS41 has no S2 shared-index page mapping.
    m.errs=dmalloc<unsigned*>(m.nl); m.tickets=dmalloc<unsigned>(m.ncoll());
    m.resid=dmalloc<float>((size_t)MX*HID);
    // W-row prefill has a separate mHC capacity; decode still uses MX. The trunk stores BF16-valued rows.
    m.hc_rows=prefill_rows;
    const size_t HR=(size_t)m.hc_rows; const int hc_dw=ds_mhc::done_words(m.hc_rows);
    m.resid4=dmalloc<ds_mhc::trunk_t>(HR*geom::HC*HID);
    m.hc_mix=dmalloc<float>(HR*geom::HC_MIX);
    m.hc_pre=dmalloc<float>(HR*geom::HC); m.hc_post=dmalloc<float>(HR*geom::HC);
    m.hc_comb=dmalloc<float>(HR*geom::HC*geom::HC); m.hc_done=dmalloc<unsigned>(hc_dw);
    G6RT_CK(cudaMemsetAsync(m.hc_done,0,hc_dw*sizeof(unsigned),st));
    m.hc_part=dmalloc<float>(HR*(geom::HC_MIX+2)*(HID/ds_mhc::PW));
    m.hc_y=dmalloc<__nv_bfloat16>(HR*HID);
    m.x=dmalloc<__half>((size_t)MX*HID); m.x2=dmalloc<__half>((size_t)MX*HID);
    m.srow=dmalloc<glue::SampleRow>(MX); m.tk=dmalloc<glue::TkRes>(1); m.topp_miss=dmalloc<unsigned>(1);
    m.tok=dmalloc<int>(prefill_rows); m.pos=dmalloc<int>(prefill_rows); m.tok_out=dmalloc<int>(prefill_rows); m.tgt=dmalloc<int>(prefill_rows);
    if(const char* e=ds41slots::slots_error(m.ds_slots)) die(e);   // main and draft Model alike: one ring slab per S2 slot (ds41dsp_ops::build checks draft == main)
    m.sess=dmalloc<int>(prefill_rows);
    G6RT_CK(cudaMemsetAsync(m.sess,0,(size_t)prefill_rows*sizeof(int),st));   // slot 0 until k_step_in / a prefill chunk writes it
    m.steering_scale=dmalloc<float>(MX);   // k_step_in copies StepIn::steering_scale here every step
    m.lp=dmalloc<float>(MX); m.lpt=dmalloc<float>(MX);
    m.scratch=dmalloc<char>(bind::scratch_bytes()); m.sc=bind::scratch_at(m.scratch);
    if(prefill_rows>MX) { // bind::scratch_layout intentionally remains MX; only the two fabric partials need W rows.
        m.sc.part[0]=dmalloc<uint16_t>((size_t)prefill_rows*HID);
        m.sc.part[1]=dmalloc<uint16_t>((size_t)prefill_rows*HID);
    }
    m.layers=std::vector<Layer>(m.nl);
    std::vector<unsigned*> errors(m.nl);
    const ds41::Table& table=ds41_table();
    m.ds41_find=find;
    for(int i=0;i<m.nl;++i) {
        Layer& y=m.layers[i]; y.l=ids[i]; y.moe=geom::is_moe(y.l); y.indexer=geom::is_index_source(y.l);
        y.hc[0]=ds41_hc_weights(find,y.l,"attn"); y.hc[1]=ds41_hc_weights(find,y.l,"ffn");
        y.attn_norm=y.hc[0].norm; y.mlp_norm=y.hc[1].norm;
        y.ab.args.error=dmalloc<unsigned>(1);     // every layer owns an attention error word (k_step_out ORs them)
        errors[i]=y.ab.args.error;
        if(table.bind_attn) if(int e=table.bind_attn(&m,i,capacity,(void*)st)) die("DS41 bind_attn layer "+std::to_string(y.l)+": code "+std::to_string(e));
        if(table.bind_moe) if(int e=table.bind_moe(&m,i,0,(void*)st)) die("DS41 bind_moe layer "+std::to_string(y.l)+": code "+std::to_string(e));
        check_headroom(rank,capacity,false);
    }
    G6RT_CK(bind::init());
    G6RT_CK(g6h2d::copy(m.errs,errors.data(),errors.size()*sizeof(unsigned*)));
    auto emb=find("G.embed"),norm=find("G.final_norm"),head=find("G.lm_head.r8.gv"),hscale=find("G.lm_head.scale");
    if(!head.p) head=find("G.lm_head.mx.gv");
    if(!head.p) head=find("G.lm_head.gv");
    if(!emb.p || !emb.bf16 || emb.nd!=2 || emb.shape[0]!=geom::VOCAB || emb.shape[1]!=HID ||
       !norm.p || !norm.bf16 || norm.nd!=1 || norm.shape[0]!=HID || !head.p) die("DS41 vocabulary tensor dtype/shape");
    if(!bind::vocab_slice(rank,m.vlo,m.vn)) die("DS41 bad vocab rank");
    if(hscale.p && (!hscale.fp32 || hscale.nd!=1 || hscale.shape[0]!=m.vn)) die("DS41 G.lm_head.scale dtype/shape");
    m.emb=(const __nv_bfloat16*)emb.p; m.final_norm=(const __nv_bfloat16*)norm.p;
    m.head_t.lm_head=as_bind(head);
    m.head_t.lm_head_scale=as_bind(hscale);
    m.hlog=dmalloc<float>((size_t)MX*m.vn);
    // the plan: all stages of all layers; consume(s-1)+prepare(s) fused (one trunk pass) unless GR8_DS_MHC_FUSE=0 or Engram sits between them
    // engram_on (GR8_DS_ENGRAM=1, read above) is the plan's engram flag and the fusion rule below follows it
    m.ds41_plan=draft?ds41::plan_draft(ids):ds41::plan(ids,0,m.nl,true,true,engram_on,/*dspark=*/dspark==1);
    m.ds41_idx=ds41::index(m.ds41_plan);
    m.ds41_fuse=knob::ds_mhc_fuse();
    const int b12_mode=knob::ds_mhc_decomp()?1:0;
    m.ds41_b12_mhc=b12_mode==1 && !draft;
    if(!draft) fprintf(stderr,"# rank %d: B12_MHC_BOOT GR8_DS_MHC_DECOMP=%d candidate_rows=%d..%d\n",
                       rank,b12_mode,MX+1,m.hc_rows);
    const int wkv=(int)knob::ds_wait_k(), wmode=ds41::wait_mode(wkv,draft,knob::draft_par().value_or(0)!=0);
    m.ds41_waitk=wmode==1;
    if(wmode==2) {
        m.ds41_relay=dmalloc<u64>(1);
        fprintf(stderr,"# rank %d: DS41 relay wait: GR8_DS_WAIT_K=2 (CTA (0,0) of each consume kernel waits for the peers, the others wait for a device word)\n",rank);
    }
    else if(wkv==2) fprintf(stderr,"# rank %d: GR8_DS_WAIT_K=2 is off for %s: using K=1 (the wait kernel)\n",rank,draft?"the draft Model":"GR8_DRAFT_PAR");
    if(knob::ds_pf_gather()) {
        if(HID%(8*DS41_GATHER_T)!=0) die("GR8_DS_PF_GATHER: HID must be a multiple of 1024");
        if(!draft && m.hc_rows>MX) {
            m.ds41_gsum=dmalloc<__nv_bfloat16>((size_t)m.hc_rows*HID);
            fprintf(stderr,"# rank %d: DS41 prefill gather: GR8_DS_PF_GATHER=1 (consumes of %d..%d rows read one device BF16 sum; %zu B)\n",rank,MX+1,m.hc_rows,(size_t)m.hc_rows*HID*2);
        }
    }
    if(knob::ds_dec_stage()) {   // decode consumes (k_step / k_consume / k_finish) read the slots with 16-byte loads through smem
        if(HID%ds_mhc::FCH!=0) die("GR8_DS_DEC_STAGE: HID must be a multiple of 1024");
        if(!draft) {
            m.ds41_stage=true;
            fprintf(stderr,"# rank %d: DS41 staged slot reads: GR8_DS_DEC_STAGE=1 (main-Model consumes of 1..%d rows read the %d slots with 16-byte loads through smem)\n",rank,MX,g6nr::NR);
        }
    }
    // hc_attn(i) runs inside consume_ffn(i-1) -- except at an Engram layer: Engram edits the trunk between consume and prepare (B11 contract)
    for(int i=0;i<m.nl;++i) m.layers[i].hc_in_fused=m.ds41_fuse && i>0 && !(engram_on && geom::is_engram(ids[i]));
    const std::string note=ds41::synthetic_note(m.ds41_plan);
    if(!note.empty()) fprintf(stderr,"# rank %d: %s\n",rank,note.c_str());
    // PLAN step 9: with Engram off the layers that carry it (1, 14) serve wrong logits -- every row from such a run is "diag", never a quality gate
    if(!engram_on) for(int l:ids) if(geom::is_engram(l)) {
        fprintf(stderr,"# rank %d: DIAG: Engram is off (layer %d carries it) -- logits are wrong by design; label every row from this run \"diag\"\n",rank,l);
        break;
    }
    if(engram_on) {
        // B11: the Engram engine. Fail closed on a missing or corrupt sidecar, a wrong rank/tp or a wrong cmap sha (docs/DEEPSEEK-ENGRAM.md)
        m.win=dmalloc<int32_t>((size_t)prefill_rows*4); m.ehist=dmalloc<int32_t>((size_t)m.ds_slots*ENG_RING);
        ds41::engram::Config ec; std::string eerr;
        ec.prefix=ek.prefix; ec.rank=rank; ec.nr=g6nr::NR; ec.max_rows=prefill_rows; ec.knobs=ek;
        for(int e=0;e<ds41::engram::kNE;++e) {
            const int l=ds41::engram::kLayerIds[e];
            const TensorRef q=find(lname(l,"engram.q_weight")),k=find(lname(l,"engram.k_weight"));
            if(!q.p||!q.bf16||q.nd!=2||q.shape[0]!=4||q.shape[1]!=HID||!k.p||!k.bf16||k.nd!=2||k.shape[0]!=4||k.shape[1]!=HID)
                die("DS41 Engram: bad engram.q_weight / engram.k_weight (bf16 [4][5120]) of layer "+std::to_string(l));
            ec.q_w[e]=(const __nv_bfloat16*)q.p; ec.k_w[e]=(const __nv_bfloat16*)k.p;
            if(prefill_rows>MX) {   // the wkv of W-row chunks goes through B4's tensor-core GEMM: the pack's MXFP8 pair [25600][6144] + [25600][192] (B3's form)
                const TensorRef wc=find(lname(l,"engram.wkv.mx.codes")),ws=find(lname(l,"engram.wkv.mx.scales"));
                if(wc.p&&ws.p) {
                    if(wc.nd!=2||wc.shape[0]!=ds41::engram::kKvDim||wc.shape[1]!=ds41::engram::kEmbDim||ws.nd!=2||ws.shape[0]!=ds41::engram::kKvDim||ws.shape[1]!=ds41::engram::kEmbDim/32)
                        die("DS41 Engram: bad engram.wkv.mx.codes / .mx.scales (u8 [25600][6144] / [25600][192]) of layer "+std::to_string(l));
                    ec.wkv_codes[e]=(const uint8_t*)wc.p; ec.wkv_scales[e]=(const uint8_t*)ws.p;
                }
            }
        }
        ec.hooks.reduce=&ds41_engram_reduce; ec.hooks.project=&ds41_engram_project;
        m.engram=std::make_shared<ds41::engram::Engram>();
        if(!m.engram->init(ec,&eerr)) die("DS41 Engram: "+eerr);
        fprintf(stderr,"# rank %d: Engram on (layers %d, %d; prefix %s; cache %zu MiB, %d ways, %d io threads)\n",rank,ds41::engram::kLayerIds[0],
                ds41::engram::kLayerIds[1],ek.prefix.c_str(),ek.store.cache_bytes>>20,ek.store.ways,ek.store.io_threads);
    }
    check_headroom(rank,capacity,true);
}
inline void prepare_ds41(Model& m,int M,cudaStream_t st) {
    if(!ds41_prepare_rows_supported(M)) die("DS41 prepare: M outside 1..MX");
    const ds41::Table& table=ds41_table();
    // B6's multi-tile MoE path covers M=9..32. Each op retains its own range check.
    head_for(m,M);
    for(int i=0;i<m.nl;++i) {
        if(table.prepare_attn) if(int e=table.prepare_attn(&m,i,M,(void*)st)) die("DS41 prepare_attn layer "+std::to_string(m.layers[i].l)+": code "+std::to_string(e));
        if(table.prepare_moe) if(int e=table.prepare_moe(&m,i,M,(void*)st)) die("DS41 prepare_moe layer "+std::to_string(m.layers[i].l)+": code "+std::to_string(e));
    }
}
// Headless W-row prepare: do not ask head_for for a shape larger than decode MX. Layers [i0, i1) (default all): bounded replay prepares range A
// at the chunk's rows and range B (the layers >= 21) at its tail rows.
inline void prepare_ds41_prefill(Model& m,int M,cudaStream_t st,int i0=0,int i1=-1) {
    if(M < 1 || M > m.hc_rows) die("DS41 prefill prepare: rows outside W");
    const ds41::Table& table=ds41_table();
    if(i1<0) i1=m.nl;
    for(int i=i0;i<i1;++i) {
        if(table.prepare_attn) if(int e=table.prepare_attn(&m,i,M,(void*)st)) die("DS41 prefill prepare_attn layer "+std::to_string(m.layers[i].l)+": code "+std::to_string(e));
        if(table.prepare_moe) if(int e=table.prepare_moe(&m,i,M,(void*)st)) die("DS41 prefill prepare_moe layer "+std::to_string(m.layers[i].l)+": code "+std::to_string(e));
    }
}

// ---- the driver's own ops (Fn = int(void* model, int i, int rows, void* stream)) and the mHC ops (B7's ds_mhc kernels)
inline bool ds41_b12_on(const Model& m,int rows) {
    return ::ds41::b12::on(m.ds41_b12_mhc,m.ds41_draft,rows,MX,m.hc_rows);
}
inline void ds41_rowshard_finish_ag(Model& m,int c,int rows,cudaStream_t st);
inline void ds41_hc_prepare_slice(Model& m,int i,bool ffn,int total_rows,int row0,int n,cudaStream_t st) {
    const Layer::Hc& h=m.layers[i].hc[ffn?1:0];
    const auto b=ds41_buffers_slice(m,row0);
    __nv_bfloat16* y=m.hc_y+(size_t)row0*HID;
    const bool b12=ds41_b12_on(m,total_rows);
    if(h.fn_f32) {
        const ds_mhc::Weights<float> w{(const float*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::prepare_decomp<float>(b,w,n,HID,geom::RMS_EPS,y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::prepare<float>(b,w,n,HID,geom::RMS_EPS,y,st);
    } else {
        const ds_mhc::Weights<__nv_bfloat16> w{(const __nv_bfloat16*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::prepare_decomp<__nv_bfloat16>(b,w,n,HID,geom::RMS_EPS,y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::prepare<__nv_bfloat16>(b,w,n,HID,geom::RMS_EPS,y,st);
    }
}
inline void ds41_hc_prepare(Model& m,int i,bool ffn,int M,cudaStream_t st) {
    if(m.ds41_rowshard_pending_c>=0) {
        const int c=m.ds41_rowshard_pending_c,q=::ds41::rs::quarter(M,g6nr::NR),row0=m.rank*q;
        ds41_hc_prepare_slice(m,i,ffn,M,row0,q,st);
        ds41_rowshard_finish_ag(m,c,M,st);
        m.ds41_rowshard_pending_c=-1;
        return;
    }
    const Layer::Hc& h=m.layers[i].hc[ffn?1:0];
    const bool b12=ds41_b12_on(m,M);
    if(h.fn_f32) {
        const ds_mhc::Weights<float> w{(const float*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::prepare_decomp<float>(ds41_buffers(m),w,M,HID,geom::RMS_EPS,m.hc_y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::prepare<float>(ds41_buffers(m),w,M,HID,geom::RMS_EPS,m.hc_y,st);
    } else {
        const ds_mhc::Weights<__nv_bfloat16> w{(const __nv_bfloat16*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::prepare_decomp<__nv_bfloat16>(ds41_buffers(m),w,M,HID,geom::RMS_EPS,m.hc_y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::prepare<__nv_bfloat16>(ds41_buffers(m),w,M,HID,geom::RMS_EPS,m.hc_y,st);
    }
}
// F1 (GR8_DS_WAIT_K=1): the peers' tails are awaited by ONE CTA (k_wait_peers, the GLM / Flash pattern) ahead of the consume kernel, so the 80..160
// CTAs of that kernel do not spin on the same three host-memory lines (B1 XTRACE: publish -> tails seen 86 us on the fused consume kernels vs 13 us on
// the finish kernel, same payload). Returns the payload the Ds41Block must still wait for inside the kernel: 0 = already here (Ds41Block::wait skips).
// Off (default): the block waits itself, as before. Call it right before the consume launch, in stream order.
inline size_t ds41_consume_wait(Model& m,int c,int rows,cudaStream_t st) {
    const ds41::ConsumeWait w=ds41::consume_wait_mode(m.ds41_relay?2:m.ds41_waitk?1:0,rows,HID);   // host rule + test: ds41_wait.h
    if(w.kernel) glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,w.kernel_payload,glue::ERR_AR);
    return w.block_payload;
}
// consume of collective c, then prepare of layer `next` (ffn: its hc_ffn, else its hc_attn) in one trunk pass
template<class Blk>
inline void ds41_hc_consume_prepare_b(Model& m,const Blk& blk,int next,bool ffn,int M,cudaStream_t st) {
    const Layer::Hc& h=m.layers[next].hc[ffn?1:0];
    const bool b12=ds41_b12_on(m,M);
    if(h.fn_f32) {
        const ds_mhc::Weights<float> w{(const float*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::consume_prepare_decomp<float>(ds41_buffers(m),blk,w,M,HID,geom::RMS_EPS,m.hc_y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::consume_prepare<float>(ds41_buffers(m),blk,w,M,HID,geom::RMS_EPS,m.hc_y,st);
    } else {
        const ds_mhc::Weights<__nv_bfloat16> w{(const __nv_bfloat16*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::consume_prepare_decomp<__nv_bfloat16>(ds41_buffers(m),blk,w,M,HID,geom::RMS_EPS,m.hc_y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::consume_prepare<__nv_bfloat16>(ds41_buffers(m),blk,w,M,HID,geom::RMS_EPS,m.hc_y,st);
    }
}
template<class Blk>
inline void ds41_hc_consume_prepare_b_slice(Model& m,const Blk& blk,int next,bool ffn,int total_rows,int row0,int n,cudaStream_t st) {
    const Layer::Hc& h=m.layers[next].hc[ffn?1:0];
    const auto b=ds41_buffers_slice(m,row0);
    __nv_bfloat16* y=m.hc_y+(size_t)row0*HID;
    const bool b12=ds41_b12_on(m,total_rows);
    if(h.fn_f32) {
        const ds_mhc::Weights<float> w{(const float*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::consume_prepare_decomp<float>(b,blk,w,n,HID,geom::RMS_EPS,y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::consume_prepare<float>(b,blk,w,n,HID,geom::RMS_EPS,y,st);
    } else {
        const ds_mhc::Weights<__nv_bfloat16> w{(const __nv_bfloat16*)h.fn,h.base,h.scale,h.norm};
        if(b12) ds_mhc::consume_prepare_decomp<__nv_bfloat16>(b,blk,w,n,HID,geom::RMS_EPS,y,st,ds_mhc::DECOMP_STRIP_M,&m.ds41_b12_kernel_launches);
        else ds_mhc::consume_prepare<__nv_bfloat16>(b,blk,w,n,HID,geom::RMS_EPS,y,st);
    }
}
// F7: a consume of collective c through the gather (GR8_DS_PF_GATHER=1 and more than MX rows: a prefill chunk, never a decode step)
inline bool ds41_gather_on(const Model& m,int rows) { return m.ds41_gsum && rows>MX; }
// F7 k16: a decode consume (1..MX rows) of the main Model with GR8_DS_DEC_STAGE=1 reads the slots through the staged kernels (Ds41Staged)
inline bool ds41_stage_on(const Model& m,int rows) { return m.ds41_stage && rows<=MX; }
inline Ds41DevBlock ds41_gather(Model& m,int c,int rows,cudaStream_t st) {
    glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,(size_t)rows*HID*2,glue::ERR_AR);
    k_ds41_gather<<<dim3(HID/8/DS41_GATHER_T,rows),DS41_GATHER_T,0,st>>>(m.f,m.stepp,c,HID,m.ds41_gsum);
    return Ds41DevBlock{m.ds41_gsum,m.stepp};
}
// F10 GR8_DS_PF_RS=1: collective c of a `rows`-row step runs as RS+AG (ds41_rs.h rule; the host enqueue uses the same rule)
inline bool ds41_rs_on(const Model& m,int c,int rows) { return ::ds41::rs::on(m.ds41_rs_k>0,c,m.ds41_rs_k,rows,MX,g6nr::NR); }
// the payload whose tail says collective c has arrived (Ds41Block's wait, the L2PF stop): Q rows under RS (each slice), else all rows
inline size_t ds41_coll_payload(const Model& m,int c,int rows) {
    return ds41_rs_on(m,c,rows) ? (size_t)::ds41::rs::quarter(rows,g6nr::NR)*HID*2 : (size_t)rows*HID*2;
}
inline void put_publish_rs(Model& m,const uint16_t* staging,int c,int rows,cudaStream_t st) {
    const int q=::ds41::rs::quarter(rows,g6nr::NR);
    k_put_publish_rs<<<16,128,0,st>>>(reinterpret_cast<const uint4*>(staging),m.f,m.stepp,c,rows,HID,q,::ds41::rs::slice_stride(q,(size_t)HID*2),m.tickets+c);
}
// F10 #8 GR8_DS_PF_DIRECT=1: this enqueue is an eager W-row prefill chunk whose send slots are flat [rows][HID] (ds41_dir_dst)
inline bool ds41_direct_chunk(const Model& m) { return m.ds41_direct && m.ds41_dchunk; }
inline uint16_t* ds41_dir_dst(const Model& m,int c) { return reinterpret_cast<uint16_t*>(m.f.send+g6::fab_send_off(m.f,(int)(m.ds41_dseq&1),c)); }
// the publish of layer exchange c (k: 0 attention, 1 MoE): RS layout (one slice per owner) or the plain copy. Direct chunk: the
// producer wrote the slot (ring only), else the plain copy, flat as well (never the RS layout: the proxy sends flat slices)
inline void ds41_put_publish(Model& m,const uint16_t* staging,int c,int rows,cudaStream_t st,int k) {
    if(ds41_direct_chunk(m)) {
        if(m.ds41_dir_used[k]) { m.ds41_dir_used[k]=false; k_put_publish<<<1,128,0,st>>>(nullptr,m.f,m.stepp,c,0,m.tickets+c); }
        else put_publish(m,staging,c,rows,st);
        return;
    }
    if(ds41_rs_on(m,c,rows)) put_publish_rs(m,staging,c,rows,st);
    else put_publish(m,staging,c,rows,st);
}
// RS+AG consume of c: wait the peers' slices, reduce my Q rows (-> device sum + the AG send slot, ring c_ag), wait the peers' AG payloads,
// gather their rows into the device sum. The consumer reads the device sum (Ds41DevBlock), as the F7 gather.
inline Ds41DevBlock ds41_rs_consume(Model& m,int c,int rows,cudaStream_t st) {
    const int q=::ds41::rs::quarter(rows,g6nr::NR),cag=::ds41::rs::c_ag(m.ds41_rs_nc,c);
    const size_t pay=(size_t)q*HID*2;
    glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,pay,glue::ERR_AR);
    const size_t own_ss=ds41_direct_chunk(m) ? pay : ::ds41::rs::slice_stride(q,(size_t)HID*2);   // F10 #8: a flat slot's slices are Q rows apart
    k_rs_reduce<<<dim3(HID/8/DS41_GATHER_T,q),DS41_GATHER_T,0,st>>>(m.f,m.stepp,c,cag,HID,rows,q,own_ss,m.ds41_gsum,m.ds41_rs_tickets+c);
    glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,cag,pay,glue::ERR_AR);
    k_rs_gather<<<dim3(HID/8/DS41_GATHER_T,q,g6nr::NR-1),DS41_GATHER_T,0,st>>>(m.f,m.stepp,cag,HID,rows,q,m.ds41_gsum);
    return Ds41DevBlock{m.ds41_gsum,m.stepp};
}
// F10: GR8_DS_PF_RS=1 on the main Model (g6m4.cu, after build_ds41): RS collectives 0..k-1, their AG collectives nc..nc+k-1 on the main
// fabric (ncoll = nc + k), the AG publish tickets, the device sum (shared with the F7 gather; the consumes run one after another)
inline void ds41_rs_setup(Model& m,int rank,int nc,int k,bool rowshard=false,bool y_mxfp8=false) {
    if(HID%(8*DS41_GATHER_T)!=0) die("GR8_DS_PF_RS: HID must be a multiple of 1024");
    if(m.hc_rows<=MX) die("GR8_DS_PF_RS=1 needs GR8_DS_PREFILL_W > MX");
    if(k<1 || nc!=m.ncoll() || m.f.ncoll!=::ds41::rs::main_ncoll(nc,k,rowshard)) die("GR8_DS_PF_RS: the main fabric does not hold the AG/state collectives");
    m.ds41_rs_k=k; m.ds41_rs_nc=nc;
    m.ds41_rowshard=rowshard;
    m.ds41_y_mxfp8=y_mxfp8;
    if(y_mxfp8) {
        m.ds41_y_codes=dmalloc<uint8_t>((size_t)m.hc_rows*HID);
        m.ds41_y_scales=dmalloc<uint8_t>((size_t)m.hc_rows*HID/32);
    }
    if(rowshard) { m.ds41_state_c=nc+k; m.ds41_state_ticket=dmalloc<unsigned>(1); }
    m.ds41_rs_tickets=dmalloc<unsigned>((size_t)k);
    if(!m.ds41_gsum) m.ds41_gsum=dmalloc<__nv_bfloat16>((size_t)m.hc_rows*HID);
    fprintf(stderr,"# rank %d: DS41 prefill RS+AG: GR8_DS_PF_RS=1 (layer exchanges c < %d of %d..%d-row chunks: reduce-scatter + all-gather, AG collectives %d..%d)\n",
            rank,k,MX+1,m.hc_rows,nc,nc+k-1);
}
inline Ds41DevBlock ds41_rs_reduce_local(Model& m,int c,int rows,cudaStream_t st) {
    if(m.ds41_rowshard_pending_c>=0) die("DS41 row-shard: prior Engram prepare has not published y");
    const int q=::ds41::rs::quarter(rows,g6nr::NR);
    const size_t pay=(size_t)q*HID*2;
    glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,pay,glue::ERR_AR);
    const size_t own_ss=ds41_direct_chunk(m) ? pay : ::ds41::rs::slice_stride(q,(size_t)HID*2);
    k_rs_reduce_local<<<dim3(HID/8/DS41_GATHER_T,q),DS41_GATHER_T,0,st>>>(m.f,m.stepp,c,HID,rows,q,own_ss,m.ds41_gsum);
    return Ds41DevBlock{m.ds41_gsum+(size_t)m.rank*q*HID,m.stepp};
}
inline void ds41_rowshard_finish_ag(Model& m,int c,int rows,cudaStream_t st) {
    const int q=::ds41::rs::quarter(rows,g6nr::NR),cag=::ds41::rs::c_ag(m.ds41_rs_nc,c);
    const size_t pay=(size_t)q*HID*2;
    if(m.ds41_y_mxfp8) {
        const int row0=m.rank*q;
        if(ds41_sh_act_quant(m.hc_y+(size_t)row0*HID,HID,m.ds41_y_codes+(size_t)row0*HID,HID,
                             m.ds41_y_scales+(size_t)row0*HID/32,HID/32,q,HID,st))
            die("DS41 MXFP8 y: owner act-quant refused");
        k_rs_copy_y_mx<<<16,128,0,st>>>(m.f,m.stepp,cag,m.ds41_y_codes,m.ds41_y_scales,row0,q,HID);
        k_put_publish<<<1,128,0,st>>>(nullptr,m.f,m.stepp,cag,0,m.ds41_rs_tickets+c);
        glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,cag,::ds41::rowshard::y_mxfp8_bytes(q,HID),glue::ERR_AR);
        k_rs_gather_y_mx<<<dim3(16,g6nr::NR),128,0,st>>>(m.f,m.stepp,cag,m.ds41_y_codes,m.ds41_y_scales,q,HID);
        k_rs_dequant_y<<<256,256,0,st>>>(m.ds41_y_codes,m.ds41_y_scales,m.hc_y,(size_t)rows*HID);
        m.ds41_y_quant_ready=true;
        return;
    }
    k_rs_copy_y<<<16,128,0,st>>>(m.f,m.stepp,cag,m.hc_y,m.rank*q,q,HID);
    k_put_publish<<<1,128,0,st>>>(nullptr,m.f,m.stepp,cag,0,m.ds41_rs_tickets+c);
    glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,cag,pay,glue::ERR_AR);
    k_rs_gather<<<dim3(HID/8/DS41_GATHER_T,q,g6nr::NR-1),DS41_GATHER_T,0,st>>>(m.f,m.stepp,cag,HID,rows,q,m.hc_y);
}
inline void ds41_hc_consume_prepare(Model& m,int c,int next,bool ffn,int M,cudaStream_t st) {
    if(m.ds41_rowshard_chunk && M==4096 && ds41_rs_on(m,c,M)) {
        const int q=::ds41::rs::quarter(M,g6nr::NR),row0=m.rank*q;
        const auto blk=ds41_rs_reduce_local(m,c,M,st);
        ds41_hc_consume_prepare_b_slice(m,blk,next,ffn,M,row0,q,st);
        ds41_rowshard_finish_ag(m,c,M,st);
        return;
    }
    if(ds41_rs_on(m,c,M)) { ds41_hc_consume_prepare_b(m,ds41_rs_consume(m,c,M,st),next,ffn,M,st); return; }
    if(ds41_gather_on(m,M)) { ds41_hc_consume_prepare_b(m,ds41_gather(m,c,M,st),next,ffn,M,st); return; }
    const size_t wp=ds41_consume_wait(m,c,M,st);
    if(ds41_stage_on(m,M)) {
        if(m.ds41_relay) { Ds41RelayBlock blk; blk.f=m.f; blk.stepp=m.stepp; blk.c=c; blk.wait_payload=wp; blk.relay=m.ds41_relay; ds41_hc_consume_prepare_b(m,Ds41Staged<Ds41RelayBlock>{blk},next,ffn,M,st); }
        else ds41_hc_consume_prepare_b(m,Ds41Staged<Ds41Block>{Ds41Block{m.f,m.stepp,c,wp}},next,ffn,M,st);
        return;
    }
    if(m.ds41_relay) { Ds41RelayBlock blk; blk.f=m.f; blk.stepp=m.stepp; blk.c=c; blk.wait_payload=wp; blk.relay=m.ds41_relay; ds41_hc_consume_prepare_b(m,blk,next,ffn,M,st); }
    else ds41_hc_consume_prepare_b(m,Ds41Block{m.f,m.stepp,c,wp},next,ffn,M,st);
}
// the plain consume of collective c (no prepare behind it)
inline void ds41_consume_plain(Model& m,int c,int rows,cudaStream_t st) {
    if(m.ds41_rowshard_chunk && rows==4096 && ds41_rs_on(m,c,rows)) {
        const int q=::ds41::rs::quarter(rows,g6nr::NR),row0=m.rank*q;
        const auto blk=ds41_rs_reduce_local(m,c,rows,st);
        ds_mhc::consume(ds41_buffers_slice(m,row0),blk,q,HID,st);
        m.ds41_rowshard_pending_c=c;   // Engram op then local prepare publishes y through the AG slot
        return;
    }
    if(ds41_rs_on(m,c,rows)) { ds_mhc::consume(ds41_buffers(m),ds41_rs_consume(m,c,rows,st),rows,HID,st); return; }
    if(ds41_gather_on(m,rows)) { ds_mhc::consume(ds41_buffers(m),ds41_gather(m,c,rows,st),rows,HID,st); return; }
    const size_t wp=ds41_consume_wait(m,c,rows,st);
    if(ds41_stage_on(m,rows)) {
        if(m.ds41_relay) { Ds41RelayBlock blk; blk.f=m.f; blk.stepp=m.stepp; blk.c=c; blk.wait_payload=wp; blk.relay=m.ds41_relay; ds_mhc::consume(ds41_buffers(m),Ds41Staged<Ds41RelayBlock>{blk},rows,HID,st); }
        else ds_mhc::consume(ds41_buffers(m),Ds41Staged<Ds41Block>{Ds41Block{m.f,m.stepp,c,wp}},rows,HID,st);
        return;
    }
    if(m.ds41_relay) { Ds41RelayBlock blk; blk.f=m.f; blk.stepp=m.stepp; blk.c=c; blk.wait_payload=wp; blk.relay=m.ds41_relay; ds_mhc::consume(ds41_buffers(m),blk,rows,HID,st); }
    else ds_mhc::consume(ds41_buffers(m),Ds41Block{m.f,m.stepp,c,wp},rows,HID,st);
}
inline int ds41_op_step_in(void* model,int,int,void* stream) {
    launch_step_in(*(Model*)model,(cudaStream_t)stream); return 0;
}
// the mHC ops hold rows up to Model::hc_rows (the buffers of build_ds41): more answers 1, "the op failed", before any launch
inline bool ds41_hc_fits(const Model& m,int rows) { return rows>=1 && rows<=m.hc_rows; }
inline int ds41_op_embed(void* model,int,int rows,void* stream) {
    Model& m=*(Model*)model;
    if(!ds41_hc_fits(m,rows)) return 1;
    ds_mhc::embed(ds41_buffers(m),m.tok,m.emb,rows,HID,(cudaStream_t)stream);
    return 0;
}
inline int ds41_op_hc_attn(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model;
    if(!ds41_hc_fits(m,rows)) return 1;
    if(m.layers[i].hc_in_fused) return 0;     // ran inside consume_ffn(i-1)
    ds41_hc_prepare(m,i,false,rows,(cudaStream_t)stream);
    return 0;
}
inline int ds41_op_hc_ffn(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model;
    if(!ds41_hc_fits(m,rows)) return 1;
    if(m.ds41_fuse) return 0;                 // ran inside consume_attn(i)
    ds41_hc_prepare(m,i,true,rows,(cudaStream_t)stream);
    return 0;
}
inline int ds41_op_publish_attn(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model; cudaStream_t st=(cudaStream_t)stream;
    ds41_put_publish(m,m.sc.part[0],2*i,rows,st,0);
    dmark(DP_OPROJ,st);
    return 0;
}
inline int ds41_op_publish_ffn(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model; cudaStream_t st=(cudaStream_t)stream;
    ds41_put_publish(m,m.sc.part[1],2*i+1,rows,st,1);
    dmark(DP_PUT_M,st);
    return 0;
}
// G6_L2PF (kernels/l2pf.cuh; defined after the MoE include below): window w of layer i's gate + shared-expert prefetch, ended when
// collective c has arrived. A no-op unless launch_step_ds41 armed the prefetch (G6_L2PF set); never launches a kernel of the step.
inline void ds41_l2pf_window(Model& m,int w,int i,int c,int rows,cudaStream_t st);
inline int ds41_op_consume_attn(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model; cudaStream_t st=(cudaStream_t)stream;
    if(!ds41_hc_fits(m,rows)) return 1;
    dmark(DP_OTHER,st);
    ds41_l2pf_window(m,l2pf::W_A,i,2*i,rows,st);          // G6_L2PF bit 8: this layer's gate + shared expert while the attention partials are awaited
    if(m.ds41_fuse) ds41_hc_consume_prepare(m,2*i,i,true,rows,st);
    else ds41_consume_plain(m,2*i,rows,st);
    dmark(DP_NORM_A,st);
    return 0;
}
inline int ds41_op_consume_ffn(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model; cudaStream_t st=(cudaStream_t)stream;
    if(!ds41_hc_fits(m,rows)) return 1;
    dmark(DP_OTHER,st);
    if(i+1<m.nl) ds41_l2pf_window(m,l2pf::W_M,i+1,2*i+1,rows,st);   // G6_L2PF bit 1: the next layer's gate + shared expert while the MLP partials are awaited
    if(i+1<m.nl && m.layers[i+1].hc_in_fused) ds41_hc_consume_prepare(m,2*i+1,i+1,false,rows,st);
    else ds41_consume_plain(m,2*i+1,rows,st);
    dmark(DP_NORM_M,st);
    return 0;
}
// the last consume + final collapse + final norm: hc_y = lm_head input row, the head fragment table for the codec of the head
inline int ds41_op_finish(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model; cudaStream_t st=(cudaStream_t)stream;
    if(!ds41_hc_fits(m,rows)) return 1;
    dmark(DP_OTHER,st);
    const int hcodec=head_for(m,rows).lm_head.codec;
    const bool f16_in=hcodec==bind::MXFP8 || hcodec==bind::FP8R;
    if(ds41_stage_on(m,rows))
        ds_mhc::finish(ds41_buffers(m),Ds41Staged<Ds41Block>{Ds41Block{m.f,m.stepp,2*i+1,(size_t)rows*HID*2}},m.final_norm,rows,HID,geom::RMS_EPS,/*store=*/true,m.hc_y,st,
                       f16_in?m.sc.xf_head:nullptr,f16_in?nullptr:m.sc.xf_head);
    else
    ds_mhc::finish(ds41_buffers(m),Ds41Block{m.f,m.stepp,2*i+1,(size_t)rows*HID*2},m.final_norm,rows,HID,geom::RMS_EPS,/*store=*/true,m.hc_y,st,
                   f16_in?m.sc.xf_head:nullptr,f16_in?nullptr:m.sc.xf_head);
    dmark(DP_NORM_M,st);
    return 0;
}
inline int ds41_op_head_local(void* model,int,int rows,void* stream) { launch_head_local(*(Model*)model,rows,(cudaStream_t)stream); return 0; }
inline int ds41_op_head_merge(void* model,int,int rows,void* stream) { launch_head_merge(*(Model*)model,rows,(cudaStream_t)stream); return 0; }
inline int ds41_op_step_out(void* model,int,int rows,void* stream) { launch_step_out(*(Model*)model,rows,(cudaStream_t)stream); return 0; }
#include "../attn/ds41_attn_ops.cuh"   // B4: ST_ATTN_PROJ / ST_ATTN_CORE / ST_ATTN_OUT, bind_attn, prepare_attn
#include "../attn/ds41_idx_ops.cuh"     // B5: ST_CA_ROW / ST_INDEX_TOPK / ST_INDEX_BORROW / ST_CAND_BLOCKS; ds41_idx_bind / _prepare are called by B4's bind/prepare
#include "../kernels/ds41_moe_ops.cuh"   // B6: ST_ROUTER / ST_SHARED / ST_EXPERTS, bind_moe, prepare_moe
#include "ds41_dspark_ops.cuh"   // B12: ST_DS_*, draft-layer attention dispatch (overrides B4's ATTN slots for layers >= 40)

// ---- bounded replay (runtime/ds41_pf_split.h): ST_PF_COMPACT, the last op of range A of a prefill chunk
// rows [rows - late, rows) of a [rows][row_bytes] device array -> rows [0, late). Copies of at most rows - late rows, ascending, never overlap
// (dst ends where its src starts) and run in stream order, so a later copy's dst was read by an earlier one already.
inline void ds41_rows_shift(void* base,size_t row_bytes,int rows,int late,cudaStream_t st) {
    const int off=rows-late;
    if(!base || off<=0 || late<=0) return;
    char* b=(char*)base;
    for(int r=0;r<late;r+=off) {
        const int n=std::min(off,late-r);
        G6RT_CK(cudaMemcpyAsync(b+(size_t)r*row_bytes,b+(size_t)(r+off)*row_bytes,(size_t)n*row_bytes,cudaMemcpyDeviceToDevice,st));
    }
}
// Moves the per-row state range B reads and range A made: the trunk, the mHC coefficients and hc_y of layer 21's prepare (fused into
// consume_ffn(20)), pos / tok / tgt, and B5's layer-20 top-k and candidate lists. late = Model::ds41_pf_late (the hook sets it); late == rows
// (tail >= rows) moves nothing. Engram's token window (layers 1, 14) and the MoE / attention scratch are not read across the boundary.
inline int ds41_op_pf_compact(void* model,int,int rows,void* stream) {
    Model& m=*(Model*)model; cudaStream_t st=(cudaStream_t)stream;
    if(!ds41_hc_fits(m,rows)) return 1;
    const int late=m.ds41_pf_late;
    m.ds41_y_quant_ready=false; // late-range prepare/consume returns to the uncompressed path
    if(late<1 || late>rows) return 1;
    if(m.ds41_rowshard_pending_c>=0) die("DS41 row-shard: missing y all-gather at compact");
    if(m.ds41_rowshard_state) {
        const int first=rows-late,owner=g6nr::NR-1,c=m.ds41_state_c;
        const size_t pay=::ds41::rowshard::state_bytes(late,HID,sizeof(ds_mhc::trunk_t));
        k_rs_tail_state<false><<<16,128,0,st>>>(m.f,m.stepp,c,owner,first,late,HID,ds41_buffers(m),m.hc_y);
        k_put_publish<<<1,128,0,st>>>(nullptr,m.f,m.stepp,c,0,m.ds41_state_ticket);
        glue::k_wait_peers<<<1,32,0,st>>>(m.f,m.stepp,c,pay,glue::ERR_AR);
        k_rs_tail_state<true><<<16,128,0,st>>>(m.f,m.stepp,c,owner,first,late,HID,ds41_buffers(m),m.hc_y);
    }
    if(late==rows) return 0;
    auto sh=[&](void* p,size_t rb) { ds41_rows_shift(p,rb,rows,late,st); };
    sh(m.resid4,(size_t)geom::HC*HID*sizeof(ds_mhc::trunk_t));
    sh(m.hc_mix,(size_t)geom::HC_MIX*sizeof(float));
    sh(m.hc_pre,(size_t)geom::HC*sizeof(float)); sh(m.hc_post,(size_t)geom::HC*sizeof(float)); sh(m.hc_comb,(size_t)geom::HC*geom::HC*sizeof(float));
    sh(m.hc_y,(size_t)HID*sizeof(__nv_bfloat16));
    sh(m.pos,sizeof(int)); sh(m.tok,sizeof(int)); sh(m.tgt,sizeof(int));
    ds41_idx_pf_compact(model,sh);   // 1 = no indexer state (a subset without ratio layers): nothing to move
    return 0;
}

// G6_L2PF for the DS41 main-model step (the GLM step's l2pf_window, rank.cuh): layer i's queue holds, in the order the step reads them,
// q_a + kv (T_XIN), q_b (T_QB), the gate (T_ROUTER) and the shared expert (T_SHARED). The MLP window (W_M, issued in layer i-1's MLP wait)
// starts at T_XIN: the attention projections; the attention window (W_A, layer i's own attention wait) starts at T_ROUTER: gate + shared
// expert. Each window continues where the layer's earlier one stopped (state keyed by the layer index). G6_L2PF_T picks the targets (bit =
// l2pf::T_*: 0x05 attention only, 0xc0 gate + shared only = the first version of this patch). It forks a side stream (l2pf::window), issues paced
// cp.async.bulk.prefetch.L2 until collective c's peer tails arrive or the window's budget (G6_L2PF_US[w], G6_L2PF_MB) is spent,
// and is joined once at the end of launch_step_ds41. Only between l2pf::arm and l2pf::join (launch_step_ds41).
inline void ds41_l2pf_window(Model& m,int w,int i,int c,int rows,cudaStream_t st) {
    const l2pf::Cfg& g=l2pf::cfg();
    if(!g.armed || !(g.win>>w&1)) return;
    l2pf::Plan P;
    if(const Ds41AttnLayer* a=ds41_attn_state(m,i)) {
        const ds41a::Weights& W=a->W;
        ds41l2pf::add_attn(P,W.wq_a.gv,W.wq_a.N,W.wq_a.K,W.wkv.gv,W.wkv.N,W.wkv.K,W.wq_b.gv,W.wq_b.N,W.wq_b.K);
    }
    ds41moe::l2pf_add(m,i,P);
    l2pf::Stop s; s.f=m.f; s.c=c; s.payload=ds41_coll_payload(m,c,rows); s.stepp=m.stepp;   // as Ds41Block's wait: the tail sits after rows*HID*2 bytes (F10 RS: Q*HID*2)
    l2pf::window(w,w==l2pf::W_M ? l2pf::T_XIN : l2pf::T_ROUTER,i,P,s,st);
}

// The op table. A null slot is a stage nobody has written yet: the step stops there (NOTSUP) and says whose it is
// (ds41::stage_owner). To fill a slot: include the owner's ops header at the marked line below and set the slot. Every op reads
// its input from Model::hc_y (the sub-layer input row, BF16 [rows][HID]) and writes its output to Model::sc.part[0] (attention)
// or Model::sc.part[1] (MoE), a BF16 [rows][HID] partial of this rank; the publish and consume ops are the driver's and B7's.
// F10 #8 GR8_DS_PF_DIRECT=1: the two producers of a layer's partials (K 0: ST_ATTN_OUT -> sc.part[0], c = 2i; K 1: ST_EXPERTS -> the MoE
// combine into sc.part[1], c = 2i+1), wrapped. On a direct chunk, sc.part[K] points at the pinned send slot of c while the op runs,
// then back; the publish then only rings. The ops write sc.part[K] once, as their output (B4 / B6; no op reads it before).
inline ds41::Fn g_ds41_dir_inner[2]={};
template<int K> inline int ds41_op_direct(void* model,int i,int rows,void* stream) {
    Model& m=*(Model*)model;
    if(!ds41_direct_chunk(m)) return g_ds41_dir_inner[K](model,i,rows,stream);
    uint16_t* keep=m.sc.part[K];
    m.sc.part[K]=ds41_dir_dst(m,2*i+K);
    const int r=g_ds41_dir_inner[K](model,i,rows,stream);
    m.sc.part[K]=keep;
    m.ds41_dir_used[K]=r==0;
    return r;
}
inline const ds41::Table& ds41_table() {
    static const ds41::Table table=[] {
        ds41::Table t;
        t.fn[ds41::ST_STEP_IN]=&ds41_op_step_in;
        t.fn[ds41::ST_EMBED]=&ds41_op_embed;
        t.fn[ds41::ST_HC_ATTN]=&ds41_op_hc_attn;
        t.fn[ds41::ST_PUBLISH_ATTN]=&ds41_op_publish_attn;
        t.fn[ds41::ST_CONSUME_ATTN]=&ds41_op_consume_attn;
        t.fn[ds41::ST_HC_FFN]=&ds41_op_hc_ffn;
        t.fn[ds41::ST_PUBLISH_FFN]=&ds41_op_publish_ffn;
        t.fn[ds41::ST_CONSUME_FFN]=&ds41_op_consume_ffn;
        t.fn[ds41::ST_FINISH]=&ds41_op_finish;
        t.fn[ds41::ST_HEAD_LOCAL]=&ds41_op_head_local;
        t.fn[ds41::ST_HEAD_MERGE]=&ds41_op_head_merge;
        t.fn[ds41::ST_STEP_OUT]=&ds41_op_step_out;
        // B4 (attention A, engine/attn/ds41_attn_ops.cuh): layers 0..19; ST_CA_ROW at 2/8/14 is B5's fn, which calls ds41_attn_op_comp_row
        t.fn[ds41::ST_ATTN_PROJ]=&ds41_attn_op_proj;
        t.fn[ds41::ST_ATTN_CORE]=&ds41_attn_op_core;
        t.fn[ds41::ST_ATTN_OUT]=&ds41_attn_op_out;
        t.bind_attn=&ds41_attn_bind;          // the ONE writer of both slots; it calls B5's ds41_idx_bind / ds41_idx_prepare at its end
        t.prepare_attn=&ds41_attn_prepare;
        // B5 (attention B, engine/attn/ds41_idx_ops.cuh)
        t.fn[ds41::ST_CA_ROW]=&ds41_idx_op_ca_row;
        t.fn[ds41::ST_INDEX_TOPK]=&ds41_idx_op_index_topk;
        t.fn[ds41::ST_INDEX_BORROW]=&ds41_idx_op_index_borrow;
        t.fn[ds41::ST_CAND_BLOCKS]=&ds41_idx_op_cand_blocks;
        // B6 (MoE, engine/kernels/ds41_moe_ops.cuh): ST_ROUTER, ST_SHARED, ST_EXPERTS; per-layer state behind Layer::ds41_moe via bind_moe / prepare_moe
        ds41moe::fill(t);
        // B11 (Engram, engine/runtime/ds41_engram_ops.cuh): the op answers NOTSUP when GR8_DS_ENGRAM=0 (plan(engram=false) never reaches it)
        t.fn[ds41::ST_ENGRAM]=&ds41_op_engram;
        // B12 (DSpark): ST_DS_TAP, ST_DS_MAIN (main model); ST_DS_DRAFT_IN, ST_DS_HEAD_LOCAL, ST_DS_MARKOV (draft model); also wraps ST_ATTN_* (after B4/B6)
        ds41dsp_ops::fill(t);
        // F10 #8: wrap whatever B4 / B6 / B12 put in the two producer stages (ds41_op_direct passes through off a direct chunk)
        if((g_ds41_dir_inner[0]=t.fn[ds41::ST_ATTN_OUT])) t.fn[ds41::ST_ATTN_OUT]=&ds41_op_direct<0>;
        if((g_ds41_dir_inner[1]=t.fn[ds41::ST_EXPERTS])) t.fn[ds41::ST_EXPERTS]=&ds41_op_direct<1>;
        t.fn[ds41::ST_PF_COMPACT]=&ds41_op_pf_compact;   // bounded replay: only in a split prefill plan (ds41_pf_split.h)
        return t;
    }();
    return table;
}

// ---- first-light diagnostics (both off unless their env var is set; M must be 6 = the oracle's p6 rows; eager steps only)
// GR8_DS_FEED_DIR=<B2 oracle dir, p6-faithful>: TEACHER FORCING. Before the attention stage of every layer the trunk and the mHC state are
// overwritten with the oracle's (x_in -> resid4, attn_in -> hc_y, attn_pre/post/comb -> hc_pre/post/comb); before the MLP stage the same
// with resid_attn / ffn_in / ffn_* . Each stage then runs on the oracle's input, so its output (residual slots 2i+1, 2i+2) is that stage's own
// error, not chain drift. Compare resid_attn with the faithful arm, resid with the w4a16 arm (the engine MoE is W4A16).
// GR8_DS_SUBDUMP=<dir>: every rank's BF16 attention / MLP partial per layer (r<rank>.L<l>.attn_part.bin / ffn_part.bin, [M][5120] bf16);
// the sum over the 4 ranks (FP32, one bf16 round) is the oracle's L<l>.attn_out / ffn_out. Also hc_y at the MLP stage (r0 .ffn_in.bin).
struct Ds41FeedDev { float *x_in=nullptr,*resid_attn=nullptr,*apre=nullptr,*apost=nullptr,*acomb=nullptr,*fpre=nullptr,*fpost=nullptr,*fcomb=nullptr;
                     __nv_bfloat16 *attn_in=nullptr,*ffn_in=nullptr; };
inline const char* ds41_feed_dir() { return DS41_TEXT(knob::ds_feed_dir()); }
inline const char* ds41_subdump_dir() { return DS41_TEXT(knob::ds_subdump()); }
inline bool ds41_diag_eager(cudaStream_t st,const char* who) {
    cudaStreamCaptureStatus cs=cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(st,&cs)!=cudaSuccess || cs!=cudaStreamCaptureStatusNone) die(std::string(who)+": needs eager steps, the stream is capturing");
    return true;
}
inline std::vector<uint8_t> ds41_feed_read(const std::string& dir,int l,const char* name,size_t want) {
    const std::string p=dir+"/L"+std::to_string(l)+"."+name+".bin";
    FILE* f=fopen(p.c_str(),"rb"); if(!f) die("GR8_DS_FEED_DIR: cannot open "+p);
    std::vector<uint8_t> b(want); const size_t n=fread(b.data(),1,want,f); const int extra=fgetc(f); fclose(f);
    if(n!=want || extra!=EOF) die("GR8_DS_FEED_DIR: "+p+" must be exactly "+std::to_string(want)+" bytes");
    return b;
}
inline float* ds41_feed_f32(const std::string& dir,int l,const char* name,size_t count,bool bf16) {
    const std::vector<uint8_t> b=ds41_feed_read(dir,l,name,count*(bf16?2:4));
    std::vector<float> h(count);
    if(bf16) for(size_t k=0;k<count;++k) { const uint32_t u=(uint32_t)((const uint16_t*)b.data())[k]<<16; memcpy(&h[k],&u,4); }
    else memcpy(h.data(),b.data(),count*4);
    float* d=dmalloc<float>(count);
    if(cudaMemcpy(d,h.data(),count*4,cudaMemcpyHostToDevice)!=cudaSuccess) die("GR8_DS_FEED_DIR: upload failed");
    return d;
}
inline __nv_bfloat16* ds41_feed_bf16(const std::string& dir,int l,const char* name,size_t count) {
    const std::vector<uint8_t> b=ds41_feed_read(dir,l,name,count*2);
    __nv_bfloat16* d=dmalloc<__nv_bfloat16>(count);
    if(cudaMemcpy(d,b.data(),count*2,cudaMemcpyHostToDevice)!=cudaSuccess) die("GR8_DS_FEED_DIR: upload failed");
    return d;
}
inline void ds41_feed_stage(Model& m,int i,bool ffn,int M,cudaStream_t st) {
    if(!ds41_feed_dir()) return;
    if(M!=6) die("GR8_DS_FEED_DIR: the oracle fixtures have 6 rows, this step has M="+std::to_string(M));
    static std::map<std::pair<const Model*,int>,Ds41FeedDev> all;
    const int l=m.layers[i].l;
    auto it=all.find({&m,l});
    if(it==all.end()) {
        ds41_diag_eager(st,"GR8_DS_FEED_DIR");
        const std::string dir=ds41_feed_dir(); const size_t R=6;
        Ds41FeedDev d;
        d.x_in=ds41_feed_f32(dir,l,"x_in",R*4*HID,true);      d.resid_attn=ds41_feed_f32(dir,l,"resid_attn",R*4*HID,true);
        d.attn_in=ds41_feed_bf16(dir,l,"attn_in",R*HID);       d.ffn_in=ds41_feed_bf16(dir,l,"ffn_in",R*HID);
        d.apre=ds41_feed_f32(dir,l,"attn_pre",R*4,false);      d.apost=ds41_feed_f32(dir,l,"attn_post",R*4,false);  d.acomb=ds41_feed_f32(dir,l,"attn_comb",R*16,false);
        d.fpre=ds41_feed_f32(dir,l,"ffn_pre",R*4,false);       d.fpost=ds41_feed_f32(dir,l,"ffn_post",R*4,false);   d.fcomb=ds41_feed_f32(dir,l,"ffn_comb",R*16,false);
        it=all.emplace(std::make_pair(std::make_pair((const Model*)&m,l),d)).first;
    }
    const Ds41FeedDev& d=it->second;
    auto cp=[&](void* dst,const void* src,size_t n) { if(cudaMemcpyAsync(dst,src,n,cudaMemcpyDeviceToDevice,st)!=cudaSuccess) die("GR8_DS_FEED_DIR: copy failed"); };
    ds_mhc::f32_to_trunk(ffn?d.resid_attn:d.x_in,m.resid4,(size_t)M*4*HID,st);   // FP32 oracle rows -> trunk_t (a plain copy unless DS_MHC_TRUNK_BF16)
    cp(m.hc_y,ffn?d.ffn_in:d.attn_in,(size_t)M*HID*2);
    cp(m.hc_pre,ffn?d.fpre:d.apre,(size_t)M*4*4);
    cp(m.hc_post,ffn?d.fpost:d.apost,(size_t)M*4*4);
    cp(m.hc_comb,ffn?d.fcomb:d.acomb,(size_t)M*16*4);
}
inline void ds41_subdump(Model& m,int i,const char* what,const void* dev,int M,cudaStream_t st) {
    const char* dir=ds41_subdump_dir(); if(!dir) return;
    ds41_diag_eager(st,"GR8_DS_SUBDUMP");
    if(cudaStreamSynchronize(st)!=cudaSuccess) die("GR8_DS_SUBDUMP: the stream failed");
    std::vector<uint8_t> h((size_t)M*HID*2);
    if(cudaMemcpy(h.data(),dev,h.size(),cudaMemcpyDeviceToHost)!=cudaSuccess) die("GR8_DS_SUBDUMP: copy failed");
    const std::string p=std::string(dir)+"/r"+std::to_string(m.rank)+".L"+std::to_string(m.layers[i].l)+"."+what+".bin";
    FILE* f=fopen(p.c_str(),"wb"); if(!f) die("GR8_DS_SUBDUMP: cannot open "+p);
    fwrite(h.data(),1,h.size(),f); fclose(f);
}
// GR8_DS_SPAN=1: per-unit GPU time spans (cudaEvent pair around each plan unit on the rank's stream), summed over the layers and printed once
// per step on stderr: "# span step S rank R: attn x ms cons_a y ms mlp z ms cons_f w ms ..." (head_local + head_merge = head, step_out = deliver).
// A span is the stream time between its two events: a consume unit includes the wait for the other ranks' partials (the simulated ranks share
// one GPU). Eager steps only; the events of step S are read at the start of step S+1 and by ds41_span_flush() at the end of the run.
struct Ds41Span { int rank,layer,unit; cudaEvent_t a,b; };
inline std::vector<Ds41Span>& ds41_spans() { static std::vector<Ds41Span> v; return v; }
inline bool ds41_span_on() { return knob::ds_span(); }
inline void ds41_span_flush() {
    std::vector<Ds41Span>& v=ds41_spans(); if(v.empty()) return;
    static int step=0; double tot[16][ds41::U_COUNT]={}; int maxr=0;
    for(const Ds41Span& s:v) {
        float ms=0; cudaEventSynchronize(s.b); cudaEventElapsedTime(&ms,s.a,s.b);
        if(s.rank<16) { tot[s.rank][s.unit]+=ms; if(s.rank>maxr) maxr=s.rank; }
        cudaEventDestroy(s.a); cudaEventDestroy(s.b);
    }
    v.clear();
    for(int r=0;r<=maxr;++r) {
        fprintf(stderr,"# span step %d rank %d:",step,r);
        for(int u=0;u<ds41::U_COUNT;++u) if(tot[r][u]>0) fprintf(stderr," %s %.3f ms",ds41::unit_name((ds41::Unit)u),tot[r][u]);
        fprintf(stderr,"\n");
    }
    ++step;
}
// Run the ops of one unit (the plan span of (u, ui)). A stop (null slot or a refusing op) ends the process with the plan's own message.
// Stop hook (m4sim sets it when GR8_RESID_DUMP is on): runs just before a stop dies, so the residual slots reached are written.
inline void (*g_ds41_stop_hook)(int M,cudaStream_t st)=nullptr;
inline void ds41_unit_run(Model& m,ds41::Unit u,int ui,int M,cudaStream_t st) {
    const ds41::Span s=m.ds41_idx.get(u,ui);
    if(s.empty()) die(std::string("DS41 step: the plan has no ")+ds41::unit_name(u)+" unit "+std::to_string(ui));
    const ds41::Stop stop=ds41::run(m.ds41_plan,s.b,s.e,ds41_table(),&m,M,(void*)st,[](const ds41::Op&){});
    if(stop.stopped) { if(g_ds41_stop_hook) g_ds41_stop_hook(M,st); die(ds41::stop_message(stop)); }
}
// Every caller of a unit goes through here: launch_step_ds41 (the g6m4 decode) and the per-unit wrappers in rank.cuh that m4sim drives
// (launch_attention / launch_consume / launch_mlp_block). The three diagnostics hook at unit boundaries, so they see both paths, and the
// feed of a stage sits at the START of the stage's first unit (after the residual-slot copy that follows the previous consume).
// Off (no env var): one predictable branch, nothing else.
inline bool ds41_diag_any() { return ds41_feed_dir() || ds41_subdump_dir() || ds41_span_on(); }
inline void ds41_unit(Model& m,ds41::Unit u,int ui,int M,cudaStream_t st) {
    if(!ds41_diag_any()) { ds41_unit_run(m,u,ui,M,st); return; }
    if(u==ds41::U_EMBED && m.rank==0 && ds41_span_on()) ds41_span_flush();   // the previous step is complete (m4sim syncs after every step)
    if(u==ds41::U_ATTN) ds41_feed_stage(m,ui,false,M,st);
    if(u==ds41::U_MLP) {
        if(m.rank==0) ds41_subdump(m,ui,"ffn_in",m.hc_y,M,st);   // the engine's own, before a feed replaces it
        ds41_feed_stage(m,ui,true,M,st);
    }
    Ds41Span sp{m.rank,ui,(int)u,nullptr,nullptr};
    const bool span=ds41_span_on();
    if(span) { ds41_diag_eager(st,"GR8_DS_SPAN"); cudaEventCreate(&sp.a); cudaEventCreate(&sp.b); cudaEventRecord(sp.a,st); }
    ds41_unit_run(m,u,ui,M,st);
    if(span) { cudaEventRecord(sp.b,st); ds41_spans().push_back(sp); }
    if(u==ds41::U_ATTN) ds41_subdump(m,ui,"attn_part",m.sc.part[0],M,st);
    if(u==ds41::U_MLP) ds41_subdump(m,ui,"ffn_part",m.sc.part[1],M,st);
}
inline void launch_step_ds41(Model& m,int M,int context,cudaStream_t st,int part,int ls,const std::function<void(int,int)>& residual_copy) {
    (void)context;   // the layers read the context from their step cells (k_step_in)
    EvProf& ep=ev_prof();
    if(part!=1) ep.n=0;
    ep.rec=ep.every>0;
    if(ep.rec || knob::dec_prof()) g6_attn_mark=attn_mark;
    l2pf::arm();                      // G6_L2PF: the consume ops' windows may fork; joined at the end of this part (a no-op when G6_L2PF is unset)
    if(part!=1) {
        dmark(DP_OTHER,st);
        ds41_unit(m,ds41::U_STEP_IN,0,M,st);
        if(m.engram) ds41_engram_begin(m,M,st);     // the token window + the hash + the NVMe prefetch of layers 1 and 14 (they overlap layers 0..13)
        ds41_unit(m,ds41::U_EMBED,0,M,st);
        if(residual_copy) residual_copy(0,M);
        dmark(DP_STEP,st);
    }
    for(int i=part==1?ls:0;i<(part==0?ls:m.nl);++i) {
        ds41_unit(m,ds41::U_ATTN,i,M,st);
        ds41_unit(m,ds41::U_CONS_A,i,M,st);
        if(residual_copy) residual_copy(2*i+1,M);
        ds41_unit(m,ds41::U_MLP,i,M,st);
        ds41_unit(m,ds41::U_CONS_F,i,M,st);
        if(residual_copy) residual_copy(2*i+2,M);
    }
    if(part!=0) {
        ds41_unit(m,ds41::U_HEAD_LOCAL,0,M,st); ds41_unit(m,ds41::U_HEAD_MERGE,0,M,st); ds41_unit(m,ds41::U_STEP_OUT,0,M,st);
        dmark(DP_STEP,st);
    }
    l2pf::join(st);
    ep.rec=false;
}
#endif
