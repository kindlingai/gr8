// Own-row range-A conductor. Included inside g6rt AFTER stock rank operators
// and ds41_v2_attn.cuh; CUDA collective / FFN headers are included globally.
#pragma once
#if G6_MODEL_DS41
struct Ds41V2Engine {
    ::ds41::v2::State v;
    ::ds41::v2::ffn::Scratch moe;
    ds_mhc::Buffers b{};
    int capacity=0;
    std::vector<void*> allocations;
    uint8_t *tail_send=nullptr,*tail_recv=nullptr,*eng_send=nullptr,*eng_recv=nullptr;
    template<class T> T* alloc(size_t n) {
        T* p=dmalloc<T>(n); allocations.push_back(p); return p;
    }
    static size_t tail_row_bytes() { return 4*HID*sizeof(ds_mhc::trunk_t)+(24+4+4+16)*sizeof(float)+HID*2; }
    // nullptr = this Model's layers suit v2; otherwise the reason the boot runs the old range-A path instead
    // (a trellis release, a layer subset). boot() still refuses them: a caller that skips this check is a bug.
    static const char* model_reason(const Model& m) {
        if(m.nl<=::ds41::v2::FIRST_LATE)return "the engine does not hold layers 0..21";
        for(int i=0;i<=::ds41::v2::FIRST_LATE;++i)
            if(m.layers[i].l!=i)return "the engine does not hold layers 0..21";
        for(int i=0;i<::ds41::v2::FIRST_LATE;++i) {
            auto* w=static_cast<ds41moe::LayerState*>(m.layers[i].ds41_moe);
            if(!w || !::ds41::v2::ffn::supported(::ds41::v2::ffn::layer_view(*w)))
                return "layers 0..20 hold trellis experts or no SH_TC shared weights (v2 needs native MXFP4 experts)";
        }
        return nullptr;
    }
    void boot(Model& m,g6::Fabric& f,cudaStream_t st) {
        capacity=m.hc_rows;
        // This schedule implements layers 0..20; borrowed index sources of a
        // subset would require a different schedule, never silently skip them.
        if(m.nl<=::ds41::v2::FIRST_LATE)die("DS41 v2 needs range A and stock late layers");
        for(int i=0;i<=::ds41::v2::FIRST_LATE;++i)
            if(m.layers[i].l!=i)die("DS41 v2 needs contiguous layers 0..21");
        for(int i=0;i<::ds41::v2::FIRST_LATE;++i) {
            auto* w=static_cast<ds41moe::LayerState*>(m.layers[i].ds41_moe);
            if(!w || !::ds41::v2::ffn::supported(::ds41::v2::ffn::layer_view(*w)))
                die("DS41 v2 FFN needs native MXFP4 experts and SH_TC weights");
        }
        ::ds41::v2::init(v,f,st);
        const size_t Q=(capacity+g6nr::NR-1)/g6nr::NR;
        b.streams=alloc<ds_mhc::trunk_t>(Q*4*HID); b.mix=alloc<float>(Q*24);
        b.pre=alloc<float>(Q*4); b.post=alloc<float>(Q*4); b.comb=alloc<float>(Q*16);
        b.done=alloc<unsigned>(ds_mhc::done_words(Q)); b.part=alloc<float>(Q*ds_mhc::NV*(HID/ds_mhc::PW));
        G6RT_CK(cudaMemsetAsync(b.done,0,ds_mhc::done_words(Q)*sizeof(unsigned),st));
        v.streams=b.streams; v.mix=b.mix; v.pre=b.pre; v.post=b.post; v.comb=b.comb;
        v.y=alloc<__nv_bfloat16>(Q*HID); v.attn_out=alloc<__nv_bfloat16>(Q*HID); v.ffn_out=alloc<__nv_bfloat16>(Q*HID);
        tail_send=alloc<uint8_t>(128*tail_row_bytes()); tail_recv=alloc<uint8_t>(g6nr::NR*128*tail_row_bytes());
        if(m.engram) {
            eng_send=alloc<uint8_t>(Q*g6nr::NR*::ds41::v2::engram_wire::ROW_BYTES);
            eng_recv=alloc<uint8_t>(Q*g6nr::NR*::ds41::v2::engram_wire::ROW_BYTES);
        }
        G6RT_CK(::ds41::v2::ffn::prepare(moe,capacity)); v.ffn=&moe;
        ds41_v2_attn_boot(v,m,capacity);
        G6RT_CK(cudaStreamSynchronize(st)); // boot only
    }
    void release() {
        if(!v.live)return;
        ::ds41::v2::release(v); // drains both streams/proxy before scratch dies
        ds41_v2_attn_release(v); ::ds41::v2::ffn::release(moe);
        for(void* p:allocations)G6RT_CK(cudaFree(p));
        allocations.clear(); v.ffn=nullptr;
    }
    void prepare(Model& m,int i,bool ffn,const __nv_bfloat16* output=nullptr) {
        if(!v.g.owned)return;
        const auto& h=m.layers[i].hc[ffn?1:0];
        const Ds41DevBlock blk{output,m.stepp};
        if(h.fn_f32) {
            const ds_mhc::Weights<float> w{static_cast<const float*>(h.fn),h.base,h.scale,h.norm};
            if(output)ds_mhc::consume_prepare<float>(b,blk,w,v.g.owned,HID,geom::RMS_EPS,v.y,v.st);
            else ds_mhc::prepare<float>(b,w,v.g.owned,HID,geom::RMS_EPS,v.y,v.st);
        } else {
            const ds_mhc::Weights<__nv_bfloat16> w{static_cast<const __nv_bfloat16*>(h.fn),h.base,h.scale,h.norm};
            if(output)ds_mhc::consume_prepare<__nv_bfloat16>(b,blk,w,v.g.owned,HID,geom::RMS_EPS,v.y,v.st);
            else ds_mhc::prepare<__nv_bfloat16>(b,w,v.g.owned,HID,geom::RMS_EPS,v.y,v.st);
        }
    }
    void consume(Model& m,const __nv_bfloat16* output) {
        if(v.g.owned)ds_mhc::consume(b,Ds41DevBlock{output,m.stepp},v.g.owned,HID,v.st);
    }
    void engram(Model& m,int i) {
        using namespace ::ds41::v2;
        TimingDetailSpan timer(v,TimingStage::Engram,v.st);
        m.engram->set_dead(m.vis_n>0?m.vis_flag:nullptr);
        const int e=m.layers[i].l==geom::ENGRAM_L0?0:1;
        const int rc=m.engram->op_owned(&m,e,v.g.M,v.g.row0,v.g.owned,b.streams,v.st,
            [&](const uint32_t* ids,const uint8_t* stage,const uint8_t* scales,int64_t first,int64_t valid,__nv_bfloat16* emb) {
                engram_wire::pack<<<v.g.M,256,0,v.st>>>(ids,stage,scales,first,valid,v.g.M,eng_send);
                const Segment seg{eng_send,eng_recv,engram_wire::ROW_BYTES};
                fork(v,Coll::Engram); all_to_all(v,Coll::Engram,&seg,1,v.cs); record_received(v,Coll::Engram); join(v,Coll::Engram);
                if(v.g.owned)engram_wire::unpack<<<dim3(v.g.owned,24),256,0,v.st>>>(eng_recv,v.g.Q,v.g.owned,v.g.nr,emb);
                return int(cudaGetLastError());
            });
        if(rc)die("DS41 v2 Engram: code "+std::to_string(rc));
    }
    // Bounded handover: each rank stages only its actual portion of the final
    // 128 rows, at most128 padded rows. Six arrays travel as one collective.
    void tail(Model& m) {
        using namespace ::ds41::v2;
        const Geometry saved=v.g;
        const auto mine=tail_slice(saved);
        const size_t rb[]={4*HID*sizeof(ds_mhc::trunk_t),24*sizeof(float),4*sizeof(float),4*sizeof(float),16*sizeof(float),HID*2};
        const void* own[]={b.streams,b.mix,b.pre,b.post,b.comb,v.y};
        void* stock[]={m.resid4,m.hc_mix,m.hc_pre,m.hc_post,m.hc_comb,m.hc_y};
        Segment seg[6]; size_t off=0;
        G6RT_CK(cudaMemsetAsync(tail_send,0,128*tail_row_bytes(),v.st));
        for(int k=0;k<6;++k) {
            seg[k]={tail_send+off,tail_recv+g6nr::NR*off,rb[k]};
            if(mine.count)G6RT_CK(cudaMemcpyAsync(tail_send+off,static_cast<const char*>(own[k])+size_t(mine.local)*rb[k],size_t(mine.count)*rb[k],cudaMemcpyDeviceToDevice,v.st));
            off+=128*rb[k];
        }
        v.g=geometry(128*g6nr::NR,g6nr::NR,m.rank);
        all_gather(v,Coll::Tail,seg,6,v.st); v.g=saved;
        for(int p=0;p<g6nr::NR;++p) {
            const auto part=tail_slice(geometry(saved.M,g6nr::NR,p));
            if(!part.count)continue;
            const int row=std::max(0,saved.M-128)+part.destination;
            for(int k=0;k<6;++k)G6RT_CK(cudaMemcpyAsync(static_cast<char*>(stock[k])+size_t(row)*rb[k],static_cast<const char*>(seg[k].dst)+size_t(p)*128*rb[k],size_t(part.count)*rb[k],cudaMemcpyDeviceToDevice,v.st));
        }
        m.ds41_rowshard_state=false; m.ds41_rowshard_pending_c=-1;
        if(ds41_op_pf_compact(&m,0,saved.M,v.st))die("DS41 v2 tail compaction failed");
    }
    void run(Model& m,int rows,int late_i,bool skip_late) {
        using namespace ::ds41::v2;
        if(rows<1 || rows>capacity || late_i!=FIRST_LATE || m.ds41_pack)die("DS41 v2 invalid chunk");
        const Geometry g=geometry(rows,g6nr::NR,m.rank); v.g=g;
        if(g.owned) {
            ds_mhc::embed(b,m.tok+g.row0,m.emb,g.owned,HID,v.st);
            if(m.vis_n>0)k_vis_embed<<<dim3((HID+ds_mhc::THREADS-1)/ds_mhc::THREADS,g.owned),ds_mhc::THREADS,0,v.st>>>(b,m.pos+g.row0,m.ds_slots>1?m.sess+g.row0:nullptr,m.vis_tab,m.vis_n,m.vis_pool,HID);
        }
        bool prepared=false;
        for(int i=0;i<late_i;++i) {
            const int l=m.layers[i].l;
            const bool eg=bool(m.engram) && geom::is_engram(l),last=i+1==late_i;
            auto* idx=ds41_idx_detail::find(m,false);
            const bool cand=l==geom::CAND_SOURCE && idx && !ds41_idx_detail::cand_skipped(*idx,3);
            timing_layer(v,l);
            begin_layer(v,g,eg,geom::is_index_source(l),last&&!skip_late,cand);
            TimingDetailSpan layer_timer(v,TimingStage::Layer,v.st);
            if(eg)engram(m,i);
            if(!prepared)prepare(m,i,false);
            ds41_v2_attention(v,m,i,v.y,v.attn_out);
            prepare(m,i,true,v.attn_out);
            const auto w=ffn::layer_view(*static_cast<ds41moe::LayerState*>(m.layers[i].ds41_moe));
            G6RT_CK(ffn::run(v,w,moe,ds41moe::topk_cut(),m.vis_n>0?m.vis_flag+g.row0:nullptr));
            prepared=(!last || !skip_late) && !(m.engram && geom::is_engram(m.layers[i+1].l));
            if(prepared)prepare(m,i+1,false,v.ffn_out); else consume(m,v.ffn_out);
            if(last&&!skip_late)tail(m);
            end_layer(v);
        }
    }
};
#endif
