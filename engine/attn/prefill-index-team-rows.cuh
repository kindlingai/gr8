#pragma once
// Row-owner phase1b: reuse the current index arithmetic, without latent-KV/MLA.
namespace gp {
__global__ void expand_index_rows(G6Attn fixed,half* ikey,float* scales,int count){
    auto a=ga::resolve(fixed);int pos=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;if(pos>=count)return;
    float s=0;const uint8_t* k=pos<a.context?ga::ik_at(a,pos,s):nullptr;
    for(int d=lane;d<128;d+=32)ikey[(int64_t)pos*128+d]=__float2half_rn(k?ga::e4(k[d]):0);if(lane==0)scales[pos]=s;
}
// Created binding metadata remains immutable. Device positions/pages retain
// the existing caller contract and asynchronous diagnostics; no host readback.
static bool index_rows_shape(const G6AttnBinding* b,const G6AttnStep* step){
    return b&&b->storage&&step&&step->rows>=0&&step->rows<=b->max_rows;
}
static bool index_rows_bounds(const G6AttnBinding* b,const G6AttnStep* step){
    return step->context>0&&step->context<=b->capacity&&step->rows<=step->context&&
        step->positions&&b->args.block_table&&b->args.error;
}
}
extern "C" int g6a_bind_index_append(G6AttnBinding* b,const G6AttnStep* step,void* s){
    if(!gp::index_rows_shape(b,step))return int(cudaErrorInvalidValue);
    if(!step->rows)return 0;
    if(!gp::index_rows_bounds(b,step)||!b->full_index||!step->index_k||
       !b->args.index_cache||!b->args.index_norm||!b->args.index_bias)return int(cudaErrorInvalidValue);
    G6Attn a=b->args;a.rows=step->rows;a.context=step->context;
    a.positions=step->positions;a.index_k=step->index_k;
    auto st=(cudaStream_t)s;auto e=cudaMemsetAsync(a.error,0,sizeof(unsigned),st);
    if(e!=cudaSuccess)return int(e);
    ga::append_index<<<a.rows,128,0,st>>>(a);return int(cudaGetLastError());
}
extern "C" int g6a_bind_prefill_index_rows(G6AttnPrefill* p,G6AttnBinding* b,
    const G6AttnStep* step,int* out_ids,float* out_scores,void* s){
    if(!p)return int(cudaErrorInvalidValue);
    gp::profile_reset(p->profile); // No whole-prefill timing is produced here.
    if(p->profile.events)p->profile.report.status=G6_PF_PROFILE_EMPTY;
    if(p->owner!=b||!gp::index_rows_shape(b,step)||step->rows>p->maxrows)return int(cudaErrorInvalidValue);
    if(!step->rows)return 0;
    if(!gp::index_rows_bounds(b,step)||step->context>p->cap||!out_ids)return int(cudaErrorInvalidValue);
    if(step->reuse_indices){
        if(step->reuse_rows<=0||(!step->reuse_row_map&&step->reuse_rows<step->rows)||
           (step->reuse_row_map&&step->reuse_indices==out_ids))return int(cudaErrorInvalidValue);
    }else if(!b->full_index||!p->full_index||!step->index_q||!step->index_weight||
             !b->args.index_cache||step->reuse_row_map||step->reuse_rows)return int(cudaErrorInvalidValue);
    G6Attn a=b->args;a.rows=step->rows;a.context=step->context;a.query_tile=p->R;
    a.positions=step->positions;a.index_q=step->index_q;a.index_weight=step->index_weight;
    a.indices=out_ids;a.reuse_indices=step->reuse_indices;a.reuse_row_map=step->reuse_row_map;a.reuse_rows=step->reuse_rows;
    auto st=(cudaStream_t)s;
    if(a.reuse_indices){
        for(int start=0;start<a.rows;start+=p->R){int n=std::min(p->R,a.rows-start);ga::reuse<<<n,256,0,st>>>(a,start);}
        return int(cudaGetLastError());
    }
    auto w=gp::index_workspace(a,p->indexwork,true);if(!w.total)return int(cudaErrorInvalidValue);
    int expanded=(a.context+31)&~31;
    gp::expand_index_rows<<<(expanded+7)/8,256,0,st>>>(a,p->ikey,p->kscale,expanded);
    for(int start=0;start<a.rows;start+=p->R){int n=std::min(p->R,a.rows-start);
        ga::prep_index<<<dim3(n,32),128,0,st>>>(a,start,w.iq,w.weights);
#if defined(PF_ISGRT) && PF_ISGRT
        gp::launch_index_scores_rg(a,start,n,w.iq,w.weights,p->ikey,p->kscale,w.si,w.ii,st);
#else
        gp::index_scores_grouped<<<dim3((a.context+63)/64,(n+3)/4),256,0,st>>>(a,start,n,w.iq,w.weights,p->ikey,p->kscale,w.si,w.ii);
#endif
        if(out_scores){auto e=cudaMemcpyAsync(out_scores+(int64_t)start*a.context,w.si,(size_t)n*a.context*4,cudaMemcpyDeviceToDevice,st);if(e!=cudaSuccess)return int(e);}
        auto e=gp::select_prefill(a,w,start,n,st);if(e!=cudaSuccess)return int(e);
    }
    return int(cudaGetLastError());
}
