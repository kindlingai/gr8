// kpool indexer of GLM 5.3 Flash (included by flash_attn.cu, inside namespace fa, after the cache helpers).
// Cache layout: attn/kvfmt.h (g6kv IDX_*).
//   append  kp_pool     the row that completes pool p (pos % 4 == 3) forms pooled_key[p] in FP32 from the raw rows of
//                       its pool (its call's own, or the session's raw ring)
//           kp_ring     the call's last RAW_RING rows write their raw key (wk x, FP16, before k_norm) and pool gate
//                       (FP16) to the session's raw ring
//   select  kp_scores   score[p] = 1/64 * sum_h w_h * relu(q_h . pooled_key[p]) for the pools visible to the row
//                       (complete, last token <= the row's position); 1/64 = 128^-0.5 * 32^-0.5
//           kp_select   the top min(512, visible) pools, ties to the lower pool index, written as their 4 positions in
//                       ascending order, then the incomplete tail (positions (pos+1) - (pos+1) % 4 .. pos), -1 after
//           kp_select_tile  the same list from key tiles of KP_TILE pools and a running top-512 (below), for a call
//                       whose pool stride passes KP_TILE: its scratch is one tile per row, not the context
// CP (a CP binding): a pool and its key live on the rank that stores its page; every rank writes the raw ring, which
// stays replicated. kp_scores<true> scores only the pools this rank owns (cp_mine), for flash_attn.cu's cp_local.
// Every product and sum is FP32 from the FP16 inputs, so the selection equals the float64 reference on the same inputs
// up to pools whose scores differ by FP32 rounding.

__device__ __forceinline__ uint8_t* ix_page(const DeviceAttn& a,int pos){
    const int p=page(a,pos);return p<0?nullptr:(uint8_t*)a.index_cache+(int64_t)p*g6kv::IDX_PAGE;
}
// the S2 session of row r (its raw ring); a session outside the ring is error 64
__device__ __forceinline__ int raw_sess(const DeviceAttn& a,int r){
    const int q=a.raw_sess?a.raw_sess[r]:0;
    if(q<0||q>=g6kv::RAW_SESS){error(a,64);return -1;}
    return q;
}
// 128 threads, thread t = channel t. The row that completes pool p (pos % 4 == 3) forms pooled_key[p] in FP32 from the
// raw rows of positions pos - 3 .. pos: row - d of this call when it holds position pos - d of the same session, else
// the session's raw ring (kp_ring wrote it in an earlier call)
template<class Input>
__global__ void kp_pool(Input input){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,pos=a.positions[row],w=t>>5,lane=t&31;FOR_ROW(a,input,row);
    if(pos%KP!=KP-1||cp_remote(a,pos))return;
    uint8_t* base=ix_page(a,pos);if(!base)return;
    const int sess=raw_sess(a,row);if(sess<0)return;
    __shared__ float red[KP][4];
    float k[KP],g[KP],mean[KP];
    #pragma unroll
    for(int j=0;j<KP;j++){
        const int d=KP-1-j,q=pos-d,r2=row-d;
        const half *kr,*gr;
        if(r2>=0&&a.positions[r2]==q&&raw_sess(a,r2)==sess){kr=(const half*)a.index_k+(int64_t)r2*IDIM;gr=(const half*)a.index_gate+(int64_t)r2*IDIM;}
        else{kr=(const half*)((const uint8_t*)a.index_raw+((int64_t)sess*g6kv::RAW_RING+q%g6kv::RAW_RING)*g6kv::IDX_RAW);gr=kr+IDIM;}
        k[j]=__half2float(kr[t]);g[j]=__half2float(gr[t])+bfloat(a.index_ape,j*IDIM+t);
    }
    #pragma unroll
    for(int j=0;j<KP;j++){const float s=warp_sum(k[j]);if(lane==0)red[j][w]=s;}
    __syncthreads();
    #pragma unroll
    for(int j=0;j<KP;j++)mean[j]=(red[j][0]+red[j][1]+red[j][2]+red[j][3])*(1.f/IDIM);
    __syncthreads();
    #pragma unroll
    for(int j=0;j<KP;j++){k[j]-=mean[j];const float s=warp_sum(k[j]*k[j]);if(lane==0)red[j][w]=s;}
    __syncthreads();
    const float nw=bfloat(a.index_norm,t),nb=bfloat(a.index_bias,t);
    float m=-INFINITY;
    #pragma unroll
    for(int j=0;j<KP;j++){const float var=(red[j][0]+red[j][1]+red[j][2]+red[j][3])*(1.f/IDIM);
        k[j]=k[j]/sqrtf(var+1e-6f)*nw+nb;m=fmaxf(m,g[j]);}
    float den=0,acc=0;
    #pragma unroll
    for(int j=0;j<KP;j++){const float e=expf(g[j]-m);den+=e;acc=fmaf(e,k[j],acc);}
    const float v=acc/den;if(!isfinite(v))error(a,4);
    ((float*)base)[((pos/KP)%g6kv::PAGE_POOLS)*IDIM+t]=v;
}
// grid min(rows, RAW_RING), 256 threads: row rows - gridDim.x + blockIdx.x writes its raw key (wk x, FP16, before
// k_norm) and pool gate (FP16) to its session's ring slot pos % RAW_RING, after kp_pool read the ring. Only the last
// RAW_RING rows of a call write, so no two rows of one session share a slot. CP: a row on another rank's page writes too
template<class Input>
__global__ void kp_ring(Input input){
    auto a=resolve(input);if(!a.active)return;
    const int row=a.rows-(int)gridDim.x+blockIdx.x,t=threadIdx.x,pos=a.positions[row];FOR_ROW(a,input,row);
    if(cp_remote(a,pos)){if(pos>=a.live_context){error(a,1);return;}}
    else if(page(a,pos)<0)return;
    const int sess=raw_sess(a,row);if(sess<0)return;
    half* raw=(half*)((uint8_t*)a.index_raw+((int64_t)sess*g6kv::RAW_RING+pos%g6kv::RAW_RING)*g6kv::IDX_RAW);
    raw[t]=t<IDIM?((const half*)a.index_k)[(int64_t)row*IDIM+t]:((const half*)a.index_gate)[(int64_t)row*IDIM+t-IDIM];
}
// pools a row may select: complete and ending at or before its position
__device__ __forceinline__ int kp_visible(const DeviceAttn& a,int r){
    const int cnt=min(a.positions[r]+1,a.live_context);return cnt>0?cnt/KP:0;
}
// grid (pools / 32, rows), 256 threads: 32 pools x 32 heads per CTA; thread = (pool t / 8, heads t % 8 + 8 k). Pools
// from p_off (a key tile, kp_select_tile; 0 = all of them): pool p's score at scores[row * pstride + p - p_off].
// CP: the pools of this rank only (the others are neither read nor written)
template<bool CP=false,class Input>
__global__ void kp_scores(Input input,int start,float* scores,int pstride,int p_off=0){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,r=start+row,p0=p_off+blockIdx.x*32,t=threadIdx.x;FOR_ROW(a,input,r);
    const int nvis=kp_visible(a,r);
    if(p0>=nvis)return;
    const int npools=min(32,nvis-p0);
    if(CP&&!cp_mine(a,p0*KP)&&!cp_mine(a,(p0+npools-1)*KP))return;   // 32 pools span two pages
    __shared__ float qs[IHEADS][IDIM+1],ks[32][IDIM+1],ws[IHEADS];
    for(int i=t;i<IHEADS*IDIM;i+=256)qs[i/IDIM][i%IDIM]=hfloat(a.index_q,(int64_t)r*IHEADS*IDIM+i);
    if(t<IHEADS)ws[t]=hfloat(a.index_weight,(int64_t)r*IHEADS+t);
    for(int i=t;i<npools*IDIM;i+=256){const int pp=i/IDIM,d=i%IDIM,p=p0+pp;float v=0;
        const uint8_t* b=(!CP||cp_mine(a,p*KP))?ix_page(a,p*KP):nullptr;if(b)v=__ldg(((const float*)b)+((p%g6kv::PAGE_POOLS)*IDIM+d));
        ks[pp][d]=v;}
    __syncthreads();
    const int pp=t>>3,hl=t&7;
    if((pp&~3)>=npools)return;
    float sum=0;
    if(pp<npools){
        #pragma unroll
        for(int k=0;k<IHEADS/8;k++){const int h=hl+8*k;float dot=0;
            #pragma unroll 8
            for(int d=0;d<IDIM;d++)dot=fmaf(qs[h][d],ks[pp][d],dot);
            sum=fmaf(fmaxf(dot,0.f),ws[h],sum);}
    }
    sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);sum+=__shfl_xor_sync(0xffffffff,sum,4);
    if(hl==0&&pp<npools&&(!CP||cp_mine(a,(p0+pp)*KP))){
        float s=sum*(1.f/64)+0.f;   // + 0: -0 (all relu 0, negative weights) ties with +0, as in the reference
        if(!isfinite(s)){error(a,4);s=-INFINITY;}
        scores[(int64_t)row*pstride+p0-p_off+pp]=s;}
}
// descending float order as unsigned (larger float -> larger key)
__device__ __forceinline__ unsigned kp_key(float f){const unsigned u=__float_as_uint(f);return (u&0x80000000u)?~u:(u|0x80000000u);}
#ifndef KP_SELT
#define KP_SELT 512   // threads per row of kp_select (same selection at any value; 512 fastest 2K..131K on GB10)
#endif
constexpr int SELT=KP_SELT, SELR=8192/SELT;   // SELR keys per thread in registers: rows with <= 8192 visible pools
static_assert(SELT>=256&&SELT<=1024&&8192%SELT==0,"kp_select geometry");
// warp 0 of kp_select: b = the largest bin in 1..255 with sum(hist[b..255]) >= k, else 0; need = k - sum(hist[b+1..255]).
// Equal to the serial descending scan; lane l holds bins 255-8l .. 248-8l.
__device__ __forceinline__ void kp_pick(const int* hist,int k,unsigned prefix,int shift,unsigned* prefix_s,int* need_s){
    const int lane=threadIdx.x&31;
    int h[8],loc=0;
    #pragma unroll
    for(int j=0;j<8;j++){h[j]=hist[255-8*lane-j];loc+=h[j];}
    int inc=loc;
    #pragma unroll
    for(int d=1;d<32;d<<=1){const int v=__shfl_up_sync(0xffffffffu,inc,d);if(lane>=d)inc+=v;}
    int acc=inc-loc,jb=-1;   // acc: the bins above this lane's 8
    #pragma unroll
    for(int j=0;j<8;j++)if(jb<0&&255-8*lane-j>0){if(acc+h[j]>=k)jb=j;else acc+=h[j];}
    const unsigned found=__ballot_sync(0xffffffffu,jb>=0);
    const int src=found?__ffs(found)-1:31;
    if(lane==src){*prefix_s=prefix|((unsigned)(found?255-8*lane-jb:0)<<shift);*need_s=k-acc;}
}
template<class Input>
__global__ void __launch_bounds__(SELT) kp_select(Input input,int start,const float* scores,int pstride){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,r=start+row,t=threadIdx.x;FOR_ROW(a,input,r);
    const int nvis=kp_visible(a,r),cnt=a.positions[r]+1;
    const float* s=scores+(int64_t)row*pstride;
    int* out=a.indices+(int64_t)r*TOP;
    const bool all=nvis<=g6kv::TOP_POOLS;
    if(all){
        for(int p=t;p<nvis;p+=SELT){
            #pragma unroll
            for(int j=0;j<KP;j++)out[p*KP+j]=p*KP+j;
        }
        const int nsel=nvis*KP,tail0=cnt-cnt%KP;
        for(int j=nsel+t;j<TOP;j+=SELT){const int x=tail0+(j-nsel);out[j]=x<cnt?x:-1;}
        return;
    }
    __shared__ int hist[256],need_s,base_s,eqbase_s;__shared__ unsigned prefix_s;
    unsigned thr=0;int need=0;
    if(nvis<=SELR*SELT){
        unsigned rk[SELR];
        #pragma unroll
        for(int j=0;j<SELR;j++){const int p=t+j*SELT;rk[j]=(p<nvis)?kp_key(__ldg(s+p)):0u;}
        unsigned prefix=0,mask=0;int k=g6kv::TOP_POOLS;
        for(int pass=0;pass<4;pass++){const int shift=24-8*pass;
            for(int i=t;i<256;i+=SELT)hist[i]=0;
            __syncthreads();
            #pragma unroll
            for(int j=0;j<SELR;j++){
                if(j*SELT>=nvis)break;   // uniform over the CTA; no key past nvis is active
                const int p=t+j*SELT;
                const bool act=(p<nvis)&&((rk[j]&mask)==prefix);
                const unsigned m=__ballot_sync(0xffffffff,act);
                if(act){
                    const int b=(rk[j]>>shift)&255;
                    const unsigned peers=__match_any_sync(m,b);
                    if((t&31)==(__ffs(peers)-1))atomicAdd(&hist[b],__popc(peers));
                }
            }
            __syncthreads();
            if(t<32)kp_pick(hist,k,prefix,shift,&prefix_s,&need_s);
            __syncthreads();
            prefix=prefix_s;k=need_s;mask|=255u<<shift;
            __syncthreads();
        }
        thr=prefix;need=k;
        typedef cub::BlockScan<int,SELT> Scan;
        __shared__ typename Scan::TempStorage ts;
        if(t==0){base_s=0;eqbase_s=0;}
        __syncthreads();
        #pragma unroll
        for(int j=0;j<SELR;j++){
            if(j*SELT>=nvis)break;
            const int p=j*SELT+t;const bool in=p<nvis;const unsigned key=rk[j];
            const bool gt=in&&key>thr,eq=in&&key==thr;
            int ex,etot;Scan(ts).ExclusiveSum(eq?1:0,ex,etot);
            __syncthreads();
            const bool sel=gt||(eq&&eqbase_s+ex<need);
            int sx,stot;Scan(ts).ExclusiveSum(sel?1:0,sx,stot);
            if(sel){const int o=(base_s+sx)*KP;
                #pragma unroll
                for(int q=0;q<KP;q++)out[o+q]=p*KP+q;}
            __syncthreads();
            if(t==0){base_s+=stot;eqbase_s+=etot;}
            __syncthreads();
        }
    }else{
        unsigned prefix=0,mask=0;int k=g6kv::TOP_POOLS;
        for(int pass=0;pass<4;pass++){const int shift=24-8*pass;
            for(int i=t;i<256;i+=SELT)hist[i]=0;
            __syncthreads();
            for(int p=t;p<nvis;p+=SELT){const unsigned key=kp_key(s[p]);if((key&mask)==prefix)atomicAdd(&hist[(key>>shift)&255],1);}
            __syncthreads();
            if(t<32)kp_pick(hist,k,prefix,shift,&prefix_s,&need_s);
            __syncthreads();
            prefix=prefix_s;k=need_s;mask|=255u<<shift;
            __syncthreads();
        }
        thr=prefix;need=k;
        typedef cub::BlockScan<int,SELT> Scan;
        __shared__ typename Scan::TempStorage ts;
        if(t==0){base_s=0;eqbase_s=0;}
        __syncthreads();
        for(int c=0;c<nvis;c+=SELT){
            const int p=c+t;const bool in=p<nvis;const unsigned key=in?kp_key(s[p]):0u;
            const bool gt=in&&key>thr,eq=in&&key==thr;
            int ex,etot;Scan(ts).ExclusiveSum(eq?1:0,ex,etot);
            __syncthreads();
            const bool sel=gt||(eq&&eqbase_s+ex<need);
            int sx,stot;Scan(ts).ExclusiveSum(sel?1:0,sx,stot);
            if(sel){const int o=(base_s+sx)*KP;
                #pragma unroll
                for(int j=0;j<KP;j++)out[o+j]=p*KP+j;}
            __syncthreads();
            if(t==0){base_s+=stot;eqbase_s+=etot;}
            __syncthreads();
        }
    }
    const int nsel=base_s*KP,tail0=cnt-cnt%KP;
    for(int j=nsel+t;j<TOP;j+=SELT){const int x=tail0+(j-nsel);out[j]=x<cnt?x:-1;}
}
// A call whose pool stride passes KP_TILE scores its pools in key tiles of KP_TILE and keeps a running top-512 per row
// (kp_select_tile), so its scratch holds one tile, not the context. Tile k reads the carry of tiles < k: the top
// min(512, pools before the tile) pools so far in ascending pool id, with their scores. The candidates are the carry,
// then the tile's visible pools; every carry id is below the tile's, so ascending candidate order is ascending pool id,
// and the same radix select (ties to the lower candidate) keeps the top 512 of the pools seen. The top 512 of a set
// under (score descending, id ascending) is the top 512 of (the top 512 of its first part, the rest), so the last tile's
// list is kp_select's, ties included. The last tile writes the row's indices; a row with no pool in a tile copies its
// carry to the other buffer, and one with at most 512 visible pools takes them all at the last tile, as kp_select.
#ifndef KP_TILE_POOLS
#define KP_TILE_POOLS 16384   // 64K tokens: one tile at the contexts the benches run; a smaller value tests the tiles
#endif
constexpr int KP_TILE=KP_TILE_POOLS;
static_assert(KP_TILE%32==0&&KP_TILE>=g6kv::TOP_POOLS,"KP_TILE_POOLS: a multiple of 32, at least 512");
template<class Input>
__global__ void __launch_bounds__(SELT) kp_select_tile(Input input,int start,const float* scores,int tstride,
                                                       const float* cs_in,const int* ci_in,float* cs_out,int* ci_out,
                                                       int t0,int nt,int last){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,r=start+row,t=threadIdx.x;FOR_ROW(a,input,r);
    const int nvis=kp_visible(a,r),cnt=a.positions[r]+1,tail0=cnt-cnt%KP;
    int* out=a.indices+(int64_t)r*TOP;
    if(nvis<=g6kv::TOP_POOLS){
        if(!last)return;
        for(int p=t;p<nvis;p+=SELT){
            #pragma unroll
            for(int j=0;j<KP;j++)out[p*KP+j]=p*KP+j;
        }
        const int nsel=nvis*KP;
        for(int j=nsel+t;j<TOP;j+=SELT){const int x=tail0+(j-nsel);out[j]=x<cnt?x:-1;}
        return;
    }
    const int cc=min(g6kv::TOP_POOLS,min(nvis,t0)),vt=max(0,min(nvis-t0,nt));
    const float* s=scores+(int64_t)row*tstride;
    const float* cs=cs_in+(int64_t)row*g6kv::TOP_POOLS;
    const int* ci=ci_in+(int64_t)row*g6kv::TOP_POOLS;
    if(!vt){
        if(!last){   // no pool of the row in this tile: its carry moves to the buffer the next tile reads
            for(int i=t;i<cc;i+=SELT){cs_out[(int64_t)row*g6kv::TOP_POOLS+i]=cs[i];ci_out[(int64_t)row*g6kv::TOP_POOLS+i]=ci[i];}
            return;
        }
        for(int i=t;i<cc;i+=SELT){
            const int p=ci[i];
            #pragma unroll
            for(int j=0;j<KP;j++)out[i*KP+j]=p*KP+j;
        }
        const int nsel=cc*KP;
        for(int j=nsel+t;j<TOP;j+=SELT){const int x=tail0+(j-nsel);out[j]=x<cnt?x:-1;}
        return;
    }
    const int n=cc+vt;
    auto score=[&](int i){return i<cc?cs[i]:s[i-cc];};
    __shared__ int hist[256],need_s,base_s,eqbase_s;__shared__ unsigned prefix_s;
    unsigned prefix=0,mask=0;int k=g6kv::TOP_POOLS;
    for(int pass=0;pass<4;pass++){const int shift=24-8*pass;
        for(int i=t;i<256;i+=SELT)hist[i]=0;
        __syncthreads();
        for(int i=t;i<n;i+=SELT){const unsigned key=kp_key(score(i));if((key&mask)==prefix)atomicAdd(&hist[(key>>shift)&255],1);}
        __syncthreads();
        if(t<32)kp_pick(hist,k,prefix,shift,&prefix_s,&need_s);
        __syncthreads();
        prefix=prefix_s;k=need_s;mask|=255u<<shift;
        __syncthreads();
    }
    const unsigned thr=prefix;const int need=k;
    typedef cub::BlockScan<int,SELT> Scan;
    __shared__ typename Scan::TempStorage ts;
    if(t==0){base_s=0;eqbase_s=0;}
    __syncthreads();
    for(int c=0;c<n;c+=SELT){
        const int i=c+t;const bool in=i<n;const float v=in?score(i):0.f;const unsigned key=in?kp_key(v):0u;
        const bool gt=in&&key>thr,eq=in&&key==thr;
        int ex,etot;Scan(ts).ExclusiveSum(eq?1:0,ex,etot);
        __syncthreads();
        const bool sel=gt||(eq&&eqbase_s+ex<need);
        int sx,stot;Scan(ts).ExclusiveSum(sel?1:0,sx,stot);
        if(sel){
            const int o=base_s+sx,p=i<cc?ci[i]:t0+i-cc;
            if(last){
                #pragma unroll
                for(int j=0;j<KP;j++)out[o*KP+j]=p*KP+j;
            }else{cs_out[(int64_t)row*g6kv::TOP_POOLS+o]=v;ci_out[(int64_t)row*g6kv::TOP_POOLS+o]=p;}
        }
        __syncthreads();
        if(t==0){base_s+=stot;eqbase_s+=etot;}
        __syncthreads();
    }
    if(!last)return;
    const int nsel=base_s*KP;
    for(int j=nsel+t;j<TOP;j+=SELT){const int x=tail0+(j-nsel);out[j]=x<cnt?x:-1;}
}
