// kpool indexer of GLM 5.3 Flash (included by flash_attn.cu, inside namespace fa, after the cache helpers).
// Cache layout: attn/kvfmt.h (g6kv IDX_*).
//   append  kp_append   every row writes its raw key (wk x, FP16, before k_norm) and pool gate (FP16) to its slot
//           kp_pool     the row that completes pool p (pos % 4 == 3) forms pooled_key[p] in FP32
//   select  kp_scores   score[p] = 1/64 * sum_h w_h * relu(q_h . pooled_key[p]) for the pools visible to the row
//                       (complete, last token <= the row's position); 1/64 = 128^-0.5 * 32^-0.5
//           kp_select   the top min(512, visible) pools, ties to the lower pool index, written as their 4 positions in
//                       ascending order, then the incomplete tail (positions (pos+1) - (pos+1) % 4 .. pos), -1 after
// Every product and sum is FP32 from the FP16 inputs, so the selection equals the float64 reference on the same inputs
// up to pools whose scores differ by FP32 rounding.

__device__ __forceinline__ uint8_t* ix_page(const DeviceAttn& a,int pos){
    const int p=page(a,pos);return p<0?nullptr:(uint8_t*)a.index_cache+(int64_t)p*g6kv::IDX_PAGE;
}
template<class Input>
__global__ void kp_append(Input input){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,pos=a.positions[row];FOR_ROW(a,input,row);
    uint8_t* base=ix_page(a,pos);if(!base)return;
    half* raw=(half*)(base+(pos%g6kv::PAGE)*g6kv::IDX_RAW);
    raw[t]=t<IDIM?((const half*)a.index_k)[(int64_t)row*IDIM+t]:((const half*)a.index_gate)[(int64_t)row*IDIM+t-IDIM];
}
// 128 threads, thread t = channel t
template<class Input>
__global__ void kp_pool(Input input){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,pos=a.positions[row],w=t>>5,lane=t&31;FOR_ROW(a,input,row);
    if(pos%KP!=KP-1)return;
    uint8_t* base=ix_page(a,pos);if(!base)return;
    const half* raw=(const half*)(base+((pos-(KP-1))%g6kv::PAGE)*g6kv::IDX_RAW);   // the pool's 4 slots, same page
    __shared__ float red[KP][4];
    float k[KP],g[KP],mean[KP];
    #pragma unroll
    for(int j=0;j<KP;j++){k[j]=__half2float(raw[j*2*IDIM+t]);g[j]=__half2float(raw[j*2*IDIM+IDIM+t])+bfloat(a.index_ape,j*IDIM+t);}
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
    ((float*)(base+g6kv::IDX_POOL_OFF))[((pos/KP)%g6kv::PAGE_POOLS)*IDIM+t]=v;
}
// pools a row may select: complete and ending at or before its position
__device__ __forceinline__ int kp_visible(const DeviceAttn& a,int r){
    const int cnt=min(a.positions[r]+1,a.live_context);return cnt>0?cnt/KP:0;
}
// grid (pstride / 32, rows), 256 threads: 32 pools x 32 heads per CTA; thread = (pool t / 8, heads t % 8 + 8 k)
template<class Input>
__global__ void kp_scores(Input input,int start,float* scores,int pstride){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,r=start+row,p0=blockIdx.x*32,t=threadIdx.x;FOR_ROW(a,input,r);
    const int nvis=kp_visible(a,r);
    if(p0>=nvis)return;
    const int npools=min(32,nvis-p0);
    __shared__ float qs[IHEADS][IDIM+1],ks[32][IDIM+1],ws[IHEADS];
    for(int i=t;i<IHEADS*IDIM;i+=256)qs[i/IDIM][i%IDIM]=hfloat(a.index_q,(int64_t)r*IHEADS*IDIM+i);
    if(t<IHEADS)ws[t]=hfloat(a.index_weight,(int64_t)r*IHEADS+t);
    for(int i=t;i<npools*IDIM;i+=256){const int pp=i/IDIM,d=i%IDIM,p=p0+pp;float v=0;
        const uint8_t* b=ix_page(a,p*KP);if(b)v=__ldg(((const float*)(b+g6kv::IDX_POOL_OFF))+((p%g6kv::PAGE_POOLS)*IDIM+d));
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
    if(hl==0&&pp<npools){
        float s=sum*(1.f/64)+0.f;   // + 0: -0 (all relu 0, negative weights) ties with +0, as in the reference
        if(!isfinite(s)){error(a,4);s=-INFINITY;}
        scores[(int64_t)row*pstride+p0+pp]=s;}
}
// descending float order as unsigned (larger float -> larger key)
__device__ __forceinline__ unsigned kp_key(float f){const unsigned u=__float_as_uint(f);return (u&0x80000000u)?~u:(u|0x80000000u);}
constexpr int SELT=1024;
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
    if(nvis<=8*SELT){
        unsigned rk[8];
        #pragma unroll
        for(int j=0;j<8;j++){const int p=t+j*SELT;rk[j]=(p<nvis)?kp_key(__ldg(s+p)):0u;}
        unsigned prefix=0,mask=0;int k=g6kv::TOP_POOLS;
        for(int pass=0;pass<4;pass++){const int shift=24-8*pass;
            for(int i=t;i<256;i+=SELT)hist[i]=0;
            __syncthreads();
            #pragma unroll
            for(int j=0;j<8;j++){
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
            if(t==0){int acc=0,b=255;for(;b>0;b--){if(acc+hist[b]>=k)break;acc+=hist[b];}
                prefix_s=prefix|((unsigned)b<<shift);need_s=k-acc;}
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
        for(int j=0;j<8;j++){
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
            if(t==0){int acc=0,b=255;for(;b>0;b--){if(acc+hist[b]>=k)break;acc+=hist[b];}
                prefix_s=prefix|((unsigned)b<<shift);need_s=k-acc;}
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
