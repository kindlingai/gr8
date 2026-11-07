// gr8 sequence-parallel prefill FFN. Include at global scope, never inside g6rt.
// Caller gates this path with GR8_DS_PF_V2; stock operators/scratch are untouched.
#pragma once
#include "ds41_v2.h"
#include "ds41_v2_timing.cuh"
#include "../kernels/ds41_v2_moe.cuh"
#include "../kernels/ds41_v2_fc2.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include "../kernels/ds_ro_cfg.h"
#include <vector>

// Existing attention-TU shared-expert entries; no replacement GEMM/quantizer.
extern "C" int ds41_gemm_mx_tc(const uint8_t*,int,const uint8_t*,int,const uint8_t*,const uint8_t*,void*,int,int,int,int,void*);
extern "C" int ds41_sh_act_quant(const void*,int,uint8_t*,int,uint8_t*,int,int,int,void*);
extern "C" int ds41_sh_swiglu_q(const void*,int,int,float,uint8_t*,int,uint8_t*,int,int,int,void*);

namespace ds41::v2::ffn {
constexpr int H = 5120, I = 576, NE = 384, TK = 6, COLS = H / COLUMN_GROUPS;
constexpr int KP = (I + 127) / 128 * 128, MM = MAX_ROWS;
using Plan = dsr::PlanMT<TK, MM>;
struct Layer {
    const __nv_bfloat16* gate = nullptr;  // full [NE][H], already replicated at stock bind
    const float *bias = nullptr, *bias_vl = nullptr;
    const uint8_t *fc1 = nullptr, *fc2 = nullptr;
    const uint8_t *w13 = nullptr, *sw13 = nullptr, *w2 = nullptr, *sw2 = nullptr;
    float limit = 10.f;
};
// The existing SH_TC boot path supplies these plain shared weights. Trellis
// and draft layers are deliberately not interpreted as native MXFP4 records.
template<class StockLayer>
inline Layer layer_view(const StockLayer& s) {
    if (s.tr || s.ne != NE || s.tk != TK) return {};
    return {s.gate,s.bias,s.bias_vl,s.fc1,s.fc2,s.tc_w13,s.tc_sw13,s.tc_w2,s.tc_sw2,10.f};
}
inline bool supported(const Layer& w) {
    return w.gate && w.bias && w.fc1 && w.fc2 && w.w13 && w.sw13 && w.w2 && w.sw2;
}
struct Scratch {
    int capacity = 0, own_capacity = 0, tiles = 0;
    bool resident_fc2 = false;
    int* order = nullptr;
    std::vector<void*> allocations;
    Plan* plan = nullptr;
    unsigned* masks = nullptr;
    int* tile0 = nullptr;
    float* logits = nullptr;
    uint8_t *own_q = nullptr, *own_s = nullptr, *all_q = nullptr, *all_s = nullptr, *record_q = nullptr;
    int *own_idx = nullptr, *all_idx = nullptr;
    float *own_wt = nullptr, *all_wt = nullptr;
    uint8_t *aq = nullptr, *as = nullptr, *sh_q = nullptr, *sh_s = nullptr;
    __half* partial = nullptr;
    __nv_bfloat16 *sh13 = nullptr, *shared = nullptr;
    __nv_bfloat16 *send[COLUMN_GROUPS] = {}, *recv[COLUMN_GROUPS] = {};
};
// Only after all users of scratch (including cs) are drained by the owner.
inline void release(Scratch& q) {
    for (void* p : q.allocations) cudaFree(p);
    q = {};
}
template<class T> inline cudaError_t alloc(Scratch& q, T*& p, size_t n) {
    cudaError_t e = cudaMalloc(&p, n * sizeof(T));
    if (e == cudaSuccess) q.allocations.push_back(p);
    return e;
}
inline cudaError_t prepare(Scratch& q, int capacity) {
    if (capacity < 1 || capacity > MM || q.capacity || !q.allocations.empty()) return cudaErrorInvalidValue;
    q.resident_fc2 = knob::ds_pf_v2() && knob::ds_pf_v2_fc2();
    q.own_capacity = (capacity + 3) / 4;
    q.capacity = q.own_capacity * 4;
    q.tiles = dsr::pw_tile_bound(q.capacity * TK, NE);
    const size_t M = q.capacity, Q = q.own_capacity, R = size_t(q.tiles) * 8;
#define DS41_V2_ALLOC(p,n) do { cudaError_t e=alloc(q,q.p,n); if(e!=cudaSuccess){release(q);return e;} } while(0)
    if(q.resident_fc2) { DS41_V2_ALLOC(order,NE); }
    DS41_V2_ALLOC(plan,1); DS41_V2_ALLOC(masks,NE*(MM/32)); DS41_V2_ALLOC(tile0,NE); DS41_V2_ALLOC(logits,Q*NE);
    DS41_V2_ALLOC(own_q,Q*H); DS41_V2_ALLOC(own_s,Q*(H/32)); DS41_V2_ALLOC(own_idx,Q*TK); DS41_V2_ALLOC(own_wt,Q*TK);
    DS41_V2_ALLOC(all_q,M*H); DS41_V2_ALLOC(all_s,M*(H/32)); DS41_V2_ALLOC(all_idx,M*TK); DS41_V2_ALLOC(all_wt,M*TK);
    DS41_V2_ALLOC(record_q,M*H); DS41_V2_ALLOC(aq,R*I); DS41_V2_ALLOC(as,R*mxtc::ASP); DS41_V2_ALLOC(partial,R*H);
    DS41_V2_ALLOC(sh13,M*2*I); DS41_V2_ALLOC(sh_q,M*KP); DS41_V2_ALLOC(sh_s,M*(KP/32)); DS41_V2_ALLOC(shared,M*COLS);
    for (int c=0;c<COLUMN_GROUPS;++c) { DS41_V2_ALLOC(send[c],M*COLS); DS41_V2_ALLOC(recv[c],Q*COLS); }
#undef DS41_V2_ALLOC
    constexpr size_t S1=mxtc::fc1_smem<H,DS41_TC_CW,DS41_TC_NTG1,DS41_TC_ST1>();
    constexpr size_t S2=mxtc::fc2_smem<DS41_TC_W2,DS41_TC_TPC,DS41_TC_ST2>();
    cudaError_t e=cudaFuncSetAttribute(mxtc::k_fc1_tc<H,I,DS41_TC_CW,DS41_TC_NTG1,DS41_TC_ST1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,S1);
    if(e==cudaSuccess)e=cudaFuncSetAttribute(mxtc::k_fc2_tc<I,H,DS41_TC_W2,DS41_TC_WT,DS41_TC_TPC,DS41_TC_ST2,COLUMN_GROUPS>,cudaFuncAttributeMaxDynamicSharedMemorySize,S2);
    if(e==cudaSuccess && q.resident_fc2)e=cudaFuncSetAttribute(ds41v2moe::k_fc2_resident<I,H>,cudaFuncAttributeMaxDynamicSharedMemorySize,ds41v2moe::resident_smem<4,8,I>());
    if(e!=cudaSuccess)release(q);
    return e;
}
// Input is only [g.owned][H], never full-row y. vflag, if present, is also own-row.
inline cudaError_t router_own(const Geometry& g, const Layer& w, Scratch& q,
                             const __nv_bfloat16* y, const uint8_t* vflag, int kc, cudaStream_t st) {
    if (!supported(w) || g.M<1 || g.M>q.capacity || g.Q>q.own_capacity || kc<1 || kc>TK || (vflag&&!w.bias_vl)) return cudaErrorInvalidValue;
    cudaError_t e;
    if (g.owned) {
        dsr::k_gate_logits_mt<NE,0><<<NE/8,256,0,st>>>(w.gate,y,g.owned,q.logits);
        if((e=cudaGetLastError())!=cudaSuccess)return e;
        if((e=cudaMemsetAsync(q.masks,0,NE*(MM/32)*sizeof(unsigned),st))!=cudaSuccess)return e;
#define DS41_V2_ROUTE(K) case K: dsr::k_route_ro<NE,TK,MM,K><<<(g.owned+7)/8,256,0,st>>>(q.logits,w.bias,g.owned,q.plan,q.masks,w.bias_vl,vflag); break
        switch(kc) { DS41_V2_ROUTE(1); DS41_V2_ROUTE(2); DS41_V2_ROUTE(3); DS41_V2_ROUTE(4); DS41_V2_ROUTE(5); DS41_V2_ROUTE(6); }
#undef DS41_V2_ROUTE
        if((e=cudaGetLastError())!=cudaSuccess)return e;
        if(ds41_sh_act_quant(y,H,q.own_q,H,q.own_s,H/32,g.owned,H,st))return cudaErrorInvalidValue;
        if((e=cudaGetLastError())!=cudaSuccess)return e;
    }
    ds41v2moe::k_export_routes<TK,MM><<<(g.Q*TK+255)/256,256,0,st>>>(q.plan,g.owned,g.Q,q.own_idx,q.own_wt);
    return cudaGetLastError();
}
inline mxtc::Args args(const Layer& w, Scratch& q, int M) {
    mxtc::Args a{};
    a.fc1=w.fc1; a.fc2=w.fc2; a.xq=q.record_q; a.xs=q.all_s;
    a.slot_expert=q.plan->slot_expert; a.slot_tile0=q.plan->slot_tile0; a.slot_nt=q.plan->slot_nt;
    a.tile_mask=q.plan->tile_mask; a.tile_w=&q.plan->tile_w[0][0]; a.tile_tok=&q.plan->tile_tok[0][0];
    a.n_slots_dev=&q.plan->n_slots; a.n_tiles_dev=&q.plan->n_tiles; a.n_slots=dsr::ro_slots_bound(M*TK,NE);
    a.gy1=DS41_TC_GY1; a.gy2=DS41_TC_GY2; a.os2=5; // ten 128-column blocks/group, two blocks per CTA slice
    a.limit=w.limit; a.aq=q.aq; a.as=q.as; a.ypart=q.partial;
    return a;
}
inline cudaError_t up(const Geometry& g,const Layer& w,Scratch& q,cudaStream_t st) {
    cudaError_t e=cudaMemsetAsync(q.masks,0,NE*(MM/32)*sizeof(unsigned),st);
    if(e!=cudaSuccess)return e;
    ds41v2moe::k_import_routes<NE,TK,MM><<<(g.M*8+255)/256,256,0,st>>>(q.all_idx,q.all_wt,g.M,q.plan,q.masks);
    dsr::k_scan_ro<NE,TK,MM><<<1,NE,0,st>>>(q.plan,q.masks,q.tile0);
    ds41v2moe::k_place_routes<TK,MM><<<(g.M*TK+255)/256,256,0,st>>>(q.plan,q.masks,q.tile0,g.M);
    if(q.resident_fc2)ds41v2moe::k_order_desc<NE,TK,MM><<<1,NE,0,st>>>(q.plan,q.order);
    ds41v2moe::k_relayout<true><<<(size_t(g.M)*H+255)/256,256,0,st>>>(q.all_q,q.record_q,g.M,H);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    if(ds41_gemm_mx_tc(q.all_q,H,q.all_s,H/32,w.w13,w.sw13,q.sh13,2*I,g.M,2*I,H,st))return cudaErrorInvalidValue;
    if(ds41_sh_swiglu_q(q.sh13,2*I,I,w.limit,q.sh_q,KP,q.sh_s,KP/32,g.M,KP,st))return cudaErrorInvalidValue;
    const auto a=args(w,q,g.M);
    constexpr size_t SM=mxtc::fc1_smem<H,DS41_TC_CW,DS41_TC_NTG1,DS41_TC_ST1>();
    mxtc::k_fc1_tc<H,I,DS41_TC_CW,DS41_TC_NTG1,DS41_TC_ST1,true><<<a.n_slots*(I/16/DS41_TC_CW)*a.gy1,DS41_TC_CW*32,SM,st>>>(a);
    return cudaGetLastError();
}
inline cudaError_t down_group(const Geometry& g,const Layer& w,Scratch& q,int c,cudaStream_t st) {
    if(c<0||c>=COLUMN_GROUPS)return cudaErrorInvalidValue;
    const int col0=c*COLS;
    if(ds41_gemm_mx_tc(q.sh_q,KP,q.sh_s,KP/32,w.w2+size_t(col0)*KP,w.sw2+size_t(col0)*(KP/32),q.shared,COLS,g.M,COLS,KP,st))return cudaErrorInvalidValue;
    auto a=args(w,q,g.M); a.col_group=c;
    constexpr size_t SM=mxtc::fc2_smem<DS41_TC_W2,DS41_TC_TPC,DS41_TC_ST2>();
    if(q.resident_fc2) {
        constexpr size_t RS=ds41v2moe::resident_smem<4,8,I>();
        ds41v2moe::k_fc2_resident<I,H><<<a.n_slots*(COLS/128),128,RS,st>>>(a,q.order);
    } else mxtc::k_fc2_tc<I,H,DS41_TC_W2,DS41_TC_WT,DS41_TC_TPC,DS41_TC_ST2,COLUMN_GROUPS><<<a.n_slots*a.gy2*a.os2,DS41_TC_W2*32,SM,st>>>(a);
    ds41v2moe::k_combine_group<H,TK,MM,__nv_bfloat16><<<(size_t(g.padded())*COLS+255)/256,256,0,st>>>(q.plan,q.partial,q.shared,q.send[c],g.M,g.padded(),col0,COLS);
    return cudaGetLastError();
}
__global__ void k_assemble(const __nv_bfloat16* a,const __nv_bfloat16* b,const __nv_bfloat16* c,const __nv_bfloat16* d,
                            __nv_bfloat16* out,int owned) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=owned*H)return;
    const int col=i%H,group=col/COLS;
    const __nv_bfloat16* src=group==0?a:group==1?b:group==2?c:d;
    out[i]=src[(i/H)*COLS+col%COLS];
}
// One enqueue-only chain. All ranks call even if owned==0. The four send/recv
// buffers are independent: cs never consumes a buffer st may overwrite next.
inline cudaError_t run(State& s,const Layer& w,Scratch& q,int kc=TK,const uint8_t* own_vflag=nullptr) {
    if(s.g.nr!=4 || !s.ffn_out || !s.y || !supported(w))return cudaErrorInvalidValue;
    cudaError_t e=router_own(s.g,w,q,s.y,own_vflag,kc,s.st);
    if(e!=cudaSuccess)return e;
    const Segment seg[]={{q.own_q,q.all_q,H},{q.own_s,q.all_s,H/32},{q.own_idx,q.all_idx,TK*sizeof(int)},{q.own_wt,q.all_wt,TK*sizeof(float)}};
    all_gather(s,Coll::FfnIn,seg,4,s.st);
    {
        TimingSpan timer(s,TimingStage::FfnUp,s.st);
        if((e=up(s.g,w,q,s.st))!=cudaSuccess)return e;
    }
    for(int c=0;c<COLUMN_GROUPS;++c) {
        {
            TimingSpan timer(s,TimingStage::FfnDown,s.st,c);
            if((e=down_group(s.g,w,q,c,s.st))!=cudaSuccess)return e;
        }
        fork(s,ffn_group(c));
        reduce_scatter(s,c,q.send[c],q.recv[c],COLS,s.cs);
        record_received(s,ffn_group(c));
    }
    join(s,ffn_group(COLUMN_GROUPS-1));
    if(s.g.owned)k_assemble<<<(s.g.owned*H+255)/256,256,0,s.st>>>(q.recv[0],q.recv[1],q.recv[2],q.recv[3],s.ffn_out,s.g.owned);
    return cudaGetLastError();
}
} // namespace ds41::v2::ffn
