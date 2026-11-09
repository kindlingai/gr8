// The DEC_IDX31 decode indexer, exact variants. Included inside ga after select.cuh. Build flag DEC_IDX31=1 routes decode
// make_index through them; tests call them regardless of the flag.
// - index_scores_m: one CTA per 32-key tile scores ALL n rows of the query tile, so each key is read and converted once
//   per step instead of once per row. Per row the arithmetic is index_scores' (same fp16 key values, same wmma tiles,
//   same head-order ReLU-weighted sum, same scale), so every score is bit-identical. A tile no row can see (t0 > every
//   row's position) or whose page another CP rank scores is written -inf without loading keys (index_scores computes
//   zeros there and then writes -inf). ids are not written: no SEL32 select path reads them (only the full sort does).
// - radix_*32_live: chunks at or past the live context hold only -inf scores (key 0), so they are not read: their
//   histogram is known (key 0 lands in digit 0 when the prefix is 0, nowhere otherwise; the update adds it) and compact
//   takes nothing from them. Same counts, same state, same selection as radix_histogram32 / update32 / compact32.
template<class Input>
__global__ void index_scores_m(Input input,int start,int n,const half* q,const float* weights,float* scores){
    auto a=resolve(input);if(!a.active)return;
    using namespace nvcuda;
    const int t0=blockIdx.x*32,t=threadIdx.x,warp=t/32;
    __shared__ __align__(32) half key[32*128];
    __shared__ __align__(32) float products[32*32];
    __shared__ float scales[32];
    int pmax=-1;for(int row=0;row<n;row++)pmax=max(pmax,a.positions[start+row]);
    const bool skip=t0>pmax||(a.page_size>=32&&!CP_MINE(a,t0));   // page_size: power of two (valid), so one page per tile
    if(skip){if(t<32&&t0+t<a.context)for(int row=0;row<n;row++)scores[(int64_t)row*a.context+t0+t]=-INFINITY;return;}
    if(a.page_size>=32){
        // the tile's 32 keys are one contiguous 4 KB run of one page (and its 32 scales one run after the page's codes);
        // fp8x2 -> half2 gives the same fp16 values as e4 -> float -> half (every e4m3 value is exact in fp16)
        const int live=min(32,a.live_context-t0);float scale=0;const uint8_t* src=ik_at(a,t0,scale);
        if(t<32)scales[t]=src&&t<live?((const float*)(src-(t0%a.page_size)*128+a.page_size*128))[t0%a.page_size+t]:0.f;
        #pragma unroll
        for(int u=0;u<2;u++){const int i=t+u*128,col=i/8;   // 256 x 16 B
            uint4 v=src&&col<live?((const uint4*)src)[i]:make_uint4(0,0,0,0);const uint16_t* c=(const uint16_t*)&v;
            __half2* dst=(__half2*)key+i*8;
            #pragma unroll
            for(int k=0;k<8;k++)dst[k]=__half2(__nv_cvt_fp8x2_to_halfraw2(c[k],__NV_E4M3));}
    }else
    for(int i=t;i<32*128;i+=128){int col=i/128,dim=i%128,pos=t0+col;float scale=0;const uint8_t* src=pos<a.live_context&&CP_MINE(a,pos)?ik_at(a,pos,scale):nullptr;
        key[i]=__float2half_rn(src?e4(src[dim]):0);if(dim==0)scales[col]=scale;}
    __syncthreads();
    const int hs=(warp/2)*16,ts=(warp%2)*16;
    for(int row=0;row<n;row++){
        const int r=start+row;float* out=scores+(int64_t)row*a.context;
        if(t0>a.positions[r]){if(t<32&&t0+t<a.context)out[t0+t]=-INFINITY;continue;}   // uniform over the CTA
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> A;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> B;
        wmma::fragment<wmma::accumulator,16,16,16,float> C;wmma::fill_fragment(C,0.f);
        for(int k=0;k<128;k+=16){wmma::load_matrix_sync(A,q+((int64_t)row*32+hs)*128+k,128);wmma::load_matrix_sync(B,key+ts*128+k,128);wmma::mma_sync(C,A,B,C);}
        wmma::store_matrix_sync(products+hs*32+ts,C,32,wmma::mem_row_major);__syncthreads();
        if(t<32&&t0+t<a.context){float score=0;for(int h=0;h<32;h++)score+=fmaxf(products[h*32+t],0.f)*weights[row*32+h];score*=scales[t];
            int pos=t0+t;if(pos>a.positions[r]||!CP_MINE(a,pos))score=-INFINITY;else if(!isfinite(score)){error(a,4);score=-INFINITY;}
            out[pos]=score;}
        __syncthreads();
    }
}
template<class Input>
static void launch_index_scores_m(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st){
    index_scores_m<<<(a.context+31)/32,128,0,st>>>(input,start,n,w.iq,w.weights,w.si);
}
// GR8_IDX_GRP=<G> (attn_idx_grp; decode run_body, the per-row indexer calls, i.e. S2 batched steps): index_scores with the
// key tile read once for a group of rows of ONE session (rows bound to the same G6AttnRow bounds and table; step rows of a
// session are contiguous), at most G rows: the groups are the session runs cut at rows that are multiples of G, and the CTA
// of a group's first row runs the group (the others return). Tile load as index_scores<1> (16 B loads when page_size >= 32
// and the tile is this CP rank's; else index_scores' scalar loop: the same fp16 key values and scales; the same ik_at
// errors, which depend on the tile's page only), skipped when the tile is past every row of the group. Then warp w runs
// rows w, w + 4, ... of the group: per row index_scores' four 16x16 wmma tiles (the same fragments, k order and
// accumulator chain per tile; one warp computes all four), stored to the warp's own products, and index_scores' epilogue
// (the same head-order ReLU-weighted sum, scale, masks, error 4, score and id stores). A row the tile is past writes
// index_scores' early-return values. Per row bit-identical to index_scores (and small_decode::index_scores).
#ifdef DEC_S2
__device__ __forceinline__ bool same_session(const G6Attn&,int,int){return true;}
__device__ __forceinline__ bool same_session(const ReplayInput& in,int r1,int r2){
    const G6AttnRow* rows=in.cell->rows;if(!rows)return true;
    const G6AttnRow x=rows[r1],y=rows[r2];
    return x.table==y.table&&x.context==y.context&&x.page_count==y.page_count&&x.table_size==y.table_size&&x.ibase==y.ibase;
}
#else
template<class Input> __device__ __forceinline__ bool same_session(const Input&,int,int){return true;}
#endif
template<class Input>
__global__ void __launch_bounds__(128) index_scores_g(Input input,int start,int n,int G,const half* q,const float* weights,float* scores,int* ids){
    auto a=resolve(input);if(!a.active)return;
    using namespace nvcuda;
    const int t0=blockIdx.x*32,row0=blockIdx.y,r0=start+row0,t=threadIdx.x,warp=t/32,lane=t&31;
    if(row0%G&&same_session(input,r0-1,r0))return;   // not the first row of its group
    int g=1;while(row0+g<n&&(row0+g)%G&&same_session(input,r0,r0+g))g++;
    FOR_ROW(a,input,r0);   // the group's bounds and table
    __shared__ __align__(32) half key[32*128];
    __shared__ __align__(32) float products[4][32*32];
    __shared__ float scales[32];
    int pmax=-1;for(int k=0;k<g;k++)pmax=max(pmax,a.positions[r0+k]);
    if(t0<=pmax){
        if(a.page_size>=32&&CP_MINE(a,t0)){
            const int live=min(32,a.live_context-t0);float scale=0;const uint8_t* src=live>0?ik_at(a,t0,scale):nullptr;
            if(t<32)scales[t]=src&&t<live?((const float*)(src-(t0%a.page_size)*128+a.page_size*128))[t0%a.page_size+t]:0.f;
            #pragma unroll
            for(int u=0;u<2;u++){const int i=t+u*128,col=i/8;   // 256 x 16 B
                uint4 v=src&&col<live?((const uint4*)src)[i]:make_uint4(0,0,0,0);const uint16_t* c=(const uint16_t*)&v;
                __half2* dst=(__half2*)key+i*8;
                #pragma unroll
                for(int k=0;k<8;k++)dst[k]=__half2(__nv_cvt_fp8x2_to_halfraw2(c[k],__NV_E4M3));}
        }else
        for(int i=t;i<32*128;i+=128){int col=i/128,dim=i%128,pos=t0+col;float scale=0;const uint8_t* src=pos<a.live_context&&CP_MINE(a,pos)?ik_at(a,pos,scale):nullptr;
            key[i]=__float2half_rn(src?e4(src[dim]):0);if(dim==0)scales[col]=scale;}
    }
    __syncthreads();
    float* prod=products[warp];
    for(int k=warp;k<g;k+=4){
        const int row=row0+k,r=start+row;
        if(t0>a.positions[r]){if(t0+lane<a.context){scores[(int64_t)row*a.context+t0+lane]=-INFINITY;ids[(int64_t)row*a.context+t0+lane]=t0+lane;}continue;}
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> A[2];
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> B[2];
        wmma::fragment<wmma::accumulator,16,16,16,float> C[2][2];
        #pragma unroll
        for(int x=0;x<2;x++)
            #pragma unroll
            for(int y=0;y<2;y++)wmma::fill_fragment(C[x][y],0.f);
        for(int kk=0;kk<128;kk+=16){
            #pragma unroll
            for(int x=0;x<2;x++){wmma::load_matrix_sync(A[x],q+((int64_t)row*32+x*16)*128+kk,128);wmma::load_matrix_sync(B[x],key+x*16*128+kk,128);}
            #pragma unroll
            for(int x=0;x<2;x++)
                #pragma unroll
                for(int y=0;y<2;y++)wmma::mma_sync(C[x][y],A[x],B[y],C[x][y]);}
        #pragma unroll
        for(int x=0;x<2;x++)
            #pragma unroll
            for(int y=0;y<2;y++)wmma::store_matrix_sync(prod+x*16*32+y*16,C[x][y],32,wmma::mem_row_major);
        __syncwarp();
        if(t0+lane<a.context){float score=0;for(int h=0;h<32;h++)score+=fmaxf(prod[h*32+lane],0.f)*weights[row*32+h];score*=scales[lane];
            int pos=t0+lane;if(pos>a.positions[r]||!CP_MINE(a,pos))score=-INFINITY;else if(!isfinite(score)){error(a,4);score=-INFINITY;}
            scores[(int64_t)row*a.context+pos]=score;ids[(int64_t)row*a.context+pos]=pos;}
        __syncwarp();
    }
}
template<class Input>
static void launch_index_scores_g(const G6Attn& a,Input input,Work& w,int start,int n,int G,cudaStream_t st){
    index_scores_g<<<dim3((a.context+31)/32,n),128,0,st>>>(input,start,n,G,w.iq,w.weights,w.si,w.ii);
}
#ifdef SEL32
template<class Input>
__global__ void radix_histogram32_live(Input input,const float* scores,SelectionState* state,int* bins,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x;
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;
    if(pass&&state[row].complete)return;
    if(chunk*RADIX_CHUNK>=a.live_context)return;   // dead chunk: radix_update32_live accounts for its keys (all 0)
    const int lane=t&31,warp=t>>5,shift=56-pass*8;
    uint64_t prefix=pass?state[row].cutoff:0,mask=pass?~0ull<<(shift+8):0;
    __shared__ int hist[8*256];
    for(int j=t;j<8*256;j+=256)hist[j]=0;
    __syncthreads();
    #pragma unroll
    for(int j=0;j<4;j++){
        int id=chunk*RADIX_CHUNK+j*256+t;
        uint64_t key=id<a.context?selection_key32(ordered_score(scores[(int64_t)row*a.context+id]),id):0;
        bool active=id<a.context&&(key&mask)==prefix;
        unsigned digit=active?unsigned((key>>shift)&255):256;
        unsigned peers=__match_any_sync(0xffffffffu,digit);
        if(active&&lane==__ffs(peers)-1)atomicAdd(hist+warp*256+digit,__popc(peers));
    }
    __syncthreads();int total=0;
    #pragma unroll
    for(int w=0;w<8;w++)total+=hist[w*256+t];
    bins[((int64_t)row*chunks+chunk)*256+t]=total;
}
// radix_update32 over the live chunks only; the dead chunks' keys (all 0: digit 0 while the prefix is 0, else none) are
// added to digit 0 without reading their (unwritten) bins. Same counts as radix_update32 over all chunks.
template<class Input>
__global__ void radix_update32_live(Input input,SelectionState* state,const int* bins,uint64_t* selected,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;
    const int live=min(chunks,(a.live_context+RADIX_CHUNK-1)/RADIX_CHUNK);
    if(pass&&state[row].complete)return;
    const uint64_t prefix=pass?state[row].cutoff:0;
    const int need=pass?state[row].rank:TOP;
    if(pass==0){for(int j=t;j<TOP;j+=256)selected[(int64_t)row*TOP+j]=0;if(t==0)state[row].used=0;}
    int count=0;
    for(int c=0;c<live;c++)count+=bins[((int64_t)row*chunks+c)*256+255-t];
    if(t==255&&prefix==0)count+=max(0,a.context-live*RADIX_CHUNK);   // digit 0 (bin 255 - t)
    using Scan=cub::BlockScan<int,256>;__shared__ typename Scan::TempStorage temp;
    int before;Scan(temp).ExclusiveSum(count,before);
    if(before<need&&before+count>=need){
        state[row].cutoff=prefix|((uint64_t)(255-t)<<(56-pass*8));
        state[row].rank=need-before;state[row].complete=(need-before)==count;
    }
}
template<class Input>
__global__ void radix_compact32_live(Input input,const float* scores,SelectionState* state,uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
    if(blockIdx.x*RADIX_CHUNK>=a.live_context)return;   // key 0 is never taken
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x,lane=t&31;
    const uint64_t threshold=state[row].cutoff;
    #pragma unroll
    for(int j=0;j<4;j++){
        int id=chunk*RADIX_CHUNK+j*256+t;
        uint64_t key=id<a.context?selection_key32(ordered_score(scores[(int64_t)row*a.context+id]),id):0;
        bool take=id<a.context&&key!=0&&key>=threshold;
        unsigned votes=__ballot_sync(0xffffffffu,take);int base=0;
        if(lane==0&&votes)base=atomicAdd(&state[row].used,__popc(votes));
        base=__shfl_sync(0xffffffffu,base,0);
        if(take){int slot=base+__popc(votes&((1u<<lane)-1));
            if(slot<TOP)selected[(int64_t)row*TOP+slot]=key;
            else error(a,4);
        }
    }
}
template<class Input>
static void threshold_compact32_live(const G6Attn& a,Input input,Work& w,int count,cudaStream_t stream){
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;int* bins=(int*)w.so;
    for(int pass=0;pass<8;pass++){
        radix_histogram32_live<<<dim3(chunks,count),256,0,stream>>>(input,w.si,w.selection,bins,pass);
        radix_update32_live<<<count,256,0,stream>>>(input,w.selection,bins,w.selected,pass);
    }
    radix_compact32_live<<<dim3(chunks,count),256,0,stream>>>(input,w.si,w.selection,w.selected);
}
// the DEC_IDX31 select: as select_index, with the live-context chunk bound past 32768
template<class Input>
static cudaError_t select_index31(const G6Attn& a,Input input,Work& w,int start,int count,cudaStream_t stream,int rank=0){
    if(a.context<=32768)return select_index(a,input,w,start,count,stream,rank);
    threshold_compact32_live(a,input,w,count,stream);
    if(rank)launch_rank_selected<2>(rank,input,start,w.si,w.selected,count,stream);
    else sort_selected32<<<count,256,0,stream>>>(input,start,w.selected);
    return cudaGetLastError();
}
#endif
