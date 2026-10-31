// DS41 prefill RS+AG kernels (GR8_DS_PF_RS=1, default off). Rules and slot math: ds41_rs.h (the host test runs the same math).
// Needs kernels/glue.cuh, g6nr.h, ds41_gather.cuh (DS41_GATHER_T, Ds41DevBlock) and ds41_rs.h (at global scope) before it; no includes,
// because ds41_step.cuh includes it inside namespace g6rt.
#pragma once
// the publish of an RS collective: device staging [rows][H] BF16 -> my send slot of c, row m at ds41::rs::pub_off (slice owner(m),
// local row), 16-byte stores; then the doorbell of c. grid 16, block 128 (as k_put_publish).
__global__ void __launch_bounds__(128) k_put_publish_rs(const uint4* src,g6::FabricDev f,const u64* stepp,int c,int rows,int H,int q,size_t ss,
                                                        unsigned* ticket) {
    const u64 step=*stepp;
    char* dst=glue::send_slot(f,step,c);
    const int per=H/8;
    const size_t n16=(size_t)rows*per;
    for(size_t i=(size_t)blockIdx.x*128+threadIdx.x;i<n16;i+=(size_t)gridDim.x*128) {
        const int m=(int)(i/per),k=(int)(i%per);
        *reinterpret_cast<uint4*>(dst+::ds41::rs::pub_off(m,q,ss,(size_t)H*2)+(size_t)k*16)=src[i];
    }
    glue::publish(f,c,g6::coll_seq(step,c,f.ncoll),ticket);
}
// the reduce of my Q rows (after k_wait_peers(c, Q H 2): the peers' slices are in): 16-byte loads of my own slice (send slot of c at
// me * ss) and of recv (c, p), every load in flight before the adds, sum p = 0..NR-1 in FP32 from 0.f (the order of Ds41Block and
// k_ds41_gather), rn16, store to the device sum (row me Q + l) and to my send slot of cag; then the doorbell of cag.
// grid (H / 8 / DS41_GATHER_T, Q), block DS41_GATHER_T. Rows past the chunk (the last ranks' short slices) store nothing.
__global__ void __launch_bounds__(DS41_GATHER_T) k_rs_reduce(g6::FabricDev f,const u64* stepp,int c,int cag,int H,int rows,int q,size_t ss,
                                                             __nv_bfloat16* out,unsigned* ticket) {
    const u64 step=*stepp;
    const int me=f.rank,l=blockIdx.y,m=me*q+l,col0=(blockIdx.x*DS41_GATHER_T+threadIdx.x)*8;
    if(m<rows && col0<H) {
        const size_t off=((size_t)l*H+col0)*2;
        uint4 v[g6nr::NR];
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p)
            v[p]=__ldcv(reinterpret_cast<const uint4*>(p==me ? glue::send_slot(f,step,c)+(size_t)me*ss+off : glue::peer_slot(f,step,c,p)+off));
        float x[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int p=0;p<g6nr::NR;++p) {
            float b[8];
            glue::bf16x8_to_f(v[p],b);
            #pragma unroll
            for(int k=0;k<8;++k) x[k]+=b[k];
        }
        uint4 o;
        __nv_bfloat16* ob=reinterpret_cast<__nv_bfloat16*>(&o);
        #pragma unroll
        for(int k=0;k<8;++k) ob[k]=__float2bfloat16_rn(x[k]);
        *reinterpret_cast<uint4*>(glue::send_slot(f,step,cag)+off)=o;
        *reinterpret_cast<uint4*>(out+(size_t)m*H+col0)=o;
    }
    glue::publish(f,cag,g6::coll_seq(step,cag,f.ncoll),ticket);
}
// the gather of the peers' reduced rows (after k_wait_peers(cag, Q H 2)): recv (cag, p) row l -> device sum row p Q + l, 16-byte copies.
// grid (H / 8 / DS41_GATHER_T, Q, NR - 1), block DS41_GATHER_T; z = the peers in rank order without me.
__global__ void __launch_bounds__(DS41_GATHER_T) k_rs_gather(g6::FabricDev f,const u64* stepp,int cag,int H,int rows,int q,__nv_bfloat16* out) {
    const u64 step=*stepp;
    const int me=f.rank,p=(int)blockIdx.z<me ? (int)blockIdx.z : (int)blockIdx.z+1,l=blockIdx.y,m=p*q+l,col0=(blockIdx.x*DS41_GATHER_T+threadIdx.x)*8;
    if(m>=rows || col0>=H) return;
    *reinterpret_cast<uint4*>(out+(size_t)m*H+col0)=__ldcv(reinterpret_cast<const uint4*>(glue::peer_slot(f,step,cag,p)+((size_t)l*H+col0)*2));
}

// Row-shard variant of k_rs_reduce (k_rs_reduce serves the row shard off): it stops after the
// rank-owned BF16 sum; the AG slot is published only after the local consume_prepare writes y.
__global__ void __launch_bounds__(DS41_GATHER_T) k_rs_reduce_local(g6::FabricDev f,const u64* stepp,int c,int H,int rows,int q,size_t ss,
                                                                   __nv_bfloat16* out) {
    const u64 step=*stepp;
    const int me=f.rank,l=blockIdx.y,m=me*q+l,col0=(blockIdx.x*DS41_GATHER_T+threadIdx.x)*8;
    if(m>=rows || col0>=H) return;
    const size_t off=((size_t)l*H+col0)*2;
    uint4 v[g6nr::NR];
    #pragma unroll
    for(int p=0;p<g6nr::NR;++p)
        v[p]=__ldcv(reinterpret_cast<const uint4*>(p==me ? glue::send_slot(f,step,c)+(size_t)me*ss+off : glue::peer_slot(f,step,c,p)+off));
    float x[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int p=0;p<g6nr::NR;++p) {
        float b[8]; glue::bf16x8_to_f(v[p],b);
        #pragma unroll
        for(int k=0;k<8;++k) x[k]+=b[k];
    }
    uint4 o; __nv_bfloat16* ob=reinterpret_cast<__nv_bfloat16*>(&o);
    #pragma unroll
    for(int k=0;k<8;++k) ob[k]=__float2bfloat16_rn(x[k]);
    *reinterpret_cast<uint4*>(out+(size_t)m*H+col0)=o;
}
// Separate copy and doorbell launches: a multi-CTA copy cannot publish safely from one CTA.
__global__ void __launch_bounds__(128) k_rs_copy_y(g6::FabricDev f,const u64* stepp,int cag,const __nv_bfloat16* y,int row0,int owned,int H) {
    char* dst=glue::send_slot(f,*stepp,cag);
    const uint4* src=reinterpret_cast<const uint4*>(y+(size_t)row0*H);
    const size_t n16=(size_t)owned*H/8;
    for(size_t i=(size_t)blockIdx.x*128+threadIdx.x;i<n16;i+=(size_t)gridDim.x*128)
        reinterpret_cast<uint4*>(dst)[i]=src[i];
}

// Lossy optional wire form: owner rows' E4M3 bytes then E8M0 bytes, both 16-byte aligned.
__global__ void __launch_bounds__(128) k_rs_copy_y_mx(g6::FabricDev f,const u64* stepp,int cag,
                                                        const uint8_t* codes,const uint8_t* scales,int row0,int q,int H) {
    char* dst=glue::send_slot(f,*stepp,cag);
    const size_t nc=(size_t)q*H, ns=nc/32;
    for(size_t i=(size_t)blockIdx.x*128+threadIdx.x;i<nc/16;i+=(size_t)gridDim.x*128)
        reinterpret_cast<uint4*>(dst)[i]=reinterpret_cast<const uint4*>(codes+(size_t)row0*H)[i];
    for(size_t i=(size_t)blockIdx.x*128+threadIdx.x;i<ns/16;i+=(size_t)gridDim.x*128)
        reinterpret_cast<uint4*>(dst+nc)[i]=reinterpret_cast<const uint4*>(scales+(size_t)row0*H/32)[i];
}
__global__ void __launch_bounds__(128) k_rs_gather_y_mx(g6::FabricDev f,const u64* stepp,int cag,
                                                          uint8_t* codes,uint8_t* scales,int q,int H) {
    const int p=(int)blockIdx.y;
    if(p==f.rank) return;
    const char* src=glue::peer_slot(f,*stepp,cag,p);
    const size_t nc=(size_t)q*H, ns=nc/32;
    for(size_t i=(size_t)blockIdx.x*128+threadIdx.x;i<nc/16;i+=(size_t)gridDim.x*128)
        reinterpret_cast<uint4*>(codes+(size_t)p*nc)[i]=reinterpret_cast<const uint4*>(src)[i];
    for(size_t i=(size_t)blockIdx.x*128+threadIdx.x;i<ns/16;i+=(size_t)gridDim.x*128)
        reinterpret_cast<uint4*>(scales+(size_t)p*ns)[i]=reinterpret_cast<const uint4*>(src+nc)[i];
}
// The BF16 readers intentionally see local dequant, including this rank's own rows.
__global__ void __launch_bounds__(256) k_rs_dequant_y(const uint8_t* codes,const uint8_t* scales,__nv_bfloat16* y,size_t n) {
    for(size_t i=(size_t)blockIdx.x*256+threadIdx.x;i<n;i+=(size_t)gridDim.x*256) {
        const __half_raw h=__nv_cvt_fp8_to_halfraw(codes[i],__NV_E4M3);
        const float v=__half2float(__half(h));
        const uint8_t b=scales[i/32];
        const float scale=__uint_as_float(b ? (uint32_t)b<<23 : 0x00400000u);
        y[i]=__float2bfloat16_rn(v*scale);
    }
}

// One-time boundary transfer of the late-row state from its owner to every rank before
// ST_PF_COMPACT. The AG of y cannot carry this state. Raw uint4 copy preserves every bit.
template<bool Import>
__global__ void __launch_bounds__(128) k_rs_tail_state(g6::FabricDev f,const u64* stepp,int c,int owner,int first,int late,int H,
                                                       ds_mhc::Buffers b,__nv_bfloat16* y) {
    if constexpr (Import) if(f.rank==owner) return;
    const u64 step=*stepp;
    const char* slot=Import ? glue::peer_slot(f,step,c,owner) : glue::send_slot(f,step,c);
    const size_t n0=(size_t)late*4*H*sizeof(ds_mhc::trunk_t), n1=(size_t)late*24*sizeof(float);
    const size_t n2=(size_t)late*4*sizeof(float), n3=n2, n4=(size_t)late*16*sizeof(float), n5=(size_t)late*H*2;
    const size_t n=n0+n1+n2+n3+n4+n5;
    for(size_t i=((size_t)blockIdx.x*128+threadIdx.x)*16;i<n;i+=(size_t)gridDim.x*128*16) {
        size_t j=i; char* local;
        if(j<n0) local=reinterpret_cast<char*>(b.streams+(size_t)first*4*H)+j;
        else if((j-=n0)<n1) local=reinterpret_cast<char*>(b.mix+(size_t)first*24)+j;
        else if((j-=n1)<n2) local=reinterpret_cast<char*>(b.pre+(size_t)first*4)+j;
        else if((j-=n2)<n3) local=reinterpret_cast<char*>(b.post+(size_t)first*4)+j;
        else if((j-=n3)<n4) local=reinterpret_cast<char*>(b.comb+(size_t)first*16)+j;
        else { j-=n4; local=reinterpret_cast<char*>(y+(size_t)first*H)+j; }
        if constexpr (Import) *reinterpret_cast<uint4*>(local)=*reinterpret_cast<const uint4*>(slot+i);
        else *reinterpret_cast<uint4*>(const_cast<char*>(slot)+i)=*reinterpret_cast<const uint4*>(local);
    }
}
