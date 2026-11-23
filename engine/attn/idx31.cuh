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
template<bool WRITE_TAIL=true,class Input>
__global__ void index_scores_m(Input input,int start,int n,const half* q,const float* weights,float* scores){
    auto a=resolve(input);if(!a.active)return;
    using namespace nvcuda;
    const int t0=blockIdx.x*32,t=threadIdx.x,warp=t/32;
    __shared__ __align__(32) half key[32*128];
    __shared__ __align__(32) float products[32*32];
    __shared__ float scales[32];
    int pmax=-1;for(int row=0;row<n;row++)pmax=max(pmax,a.positions[start+row]);
    const bool skip=t0>pmax||(a.page_size>=32&&!CP_MINE(a,t0));   // page_size: power of two (valid), so one page per tile
    if(skip){if constexpr(WRITE_TAIL){if(t<32&&t0+t<a.context)for(int row=0;row<n;row++)scores[(int64_t)row*a.context+t0+t]=-INFINITY;}return;}
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
        if(t0>a.positions[r]){if constexpr(WRITE_TAIL){if(t<32&&t0+t<a.context)out[t0+t]=-INFINITY;}continue;}   // uniform over the CTA
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
static void launch_index_scores_m(const G6Attn& a,Input input,Work& w,int start,int n,cudaStream_t st,bool write_tail=true){
    if(write_tail)index_scores_m<true><<<(a.context+31)/32,128,0,st>>>(input,start,n,w.iq,w.weights,w.si);
    else index_scores_m<false><<<(a.context+31)/32,128,0,st>>>(input,start,n,w.iq,w.weights,w.si);
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
template<bool WRITE_IDS=true,bool WRITE_TAIL=true,class Input>
__global__ void __launch_bounds__(128) index_scores_g(Input input,int start,int n,int G,const half* q,const float* weights,float* scores,int* ids){
    auto a=resolve(input);if(!a.active)return;
    using namespace nvcuda;
    const int t0=blockIdx.x*32,row0=blockIdx.y,r0=start+row0,t=threadIdx.x,warp=t/32,lane=t&31;
    if(row0%G&&same_session(input,r0-1,r0))return;   // not the first row of its group
    int g=1;while(row0+g<n&&(row0+g)%G&&same_session(input,r0,r0+g))g++;
    FOR_ROW(a,input,r0);   // the group's bounds and table
    int pmax=-1;for(int k=0;k<g;k++)pmax=max(pmax,a.positions[r0+k]);
    const bool skip=(t0>pmax)||(a.page_size>=32&&!CP_MINE(a,t0));
    if(skip){
        if constexpr(WRITE_TAIL){
            if(t0+lane<a.context){
                for(int k=warp;k<g;k+=4){
                    const int row=row0+k;
                    scores[(int64_t)row*a.context+t0+lane]=-INFINITY;
                    if constexpr(WRITE_IDS)ids[(int64_t)row*a.context+t0+lane]=t0+lane;
                }
            }
        }
        return;
    }
    __shared__ __align__(32) half key[32*128];
    __shared__ __align__(32) float products[4][32*32];
    __shared__ float scales[32];
    if(a.page_size>=32){
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
    __syncthreads();
    float* prod=products[warp];
    for(int k=warp;k<g;k+=4){
        const int row=row0+k,r=start+row;
        if(t0>a.positions[r]){
            if constexpr(WRITE_TAIL){
                if(t0+lane<a.context){scores[(int64_t)row*a.context+t0+lane]=-INFINITY;if constexpr(WRITE_IDS)ids[(int64_t)row*a.context+t0+lane]=t0+lane;}
            }
            continue;
        }
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
            scores[(int64_t)row*a.context+pos]=score;if constexpr(WRITE_IDS)ids[(int64_t)row*a.context+pos]=pos;}
        __syncwarp();
    }
}
template<class Input>
static void launch_index_scores_g(const G6Attn& a,Input input,Work& w,int start,int n,int G,cudaStream_t st,bool write_ids=true,bool write_tail=true){
    const dim3 grid((a.context+31)/32,n);
    if(write_ids){
        if(write_tail)index_scores_g<true,true><<<grid,128,0,st>>>(input,start,n,G,w.iq,w.weights,w.si,w.ii);
        else index_scores_g<true,false><<<grid,128,0,st>>>(input,start,n,G,w.iq,w.weights,w.si,w.ii);
    }else{
        if(write_tail)index_scores_g<false,true><<<grid,128,0,st>>>(input,start,n,G,w.iq,w.weights,w.si,w.ii);
        else index_scores_g<false,false><<<grid,128,0,st>>>(input,start,n,G,w.iq,w.weights,w.si,w.ii);
    }
}
#ifdef SEL32
template<class Input>
__global__ void radix_histogram32_live(Input input,int start,const float* scores,SelectionState* state,int* bins,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x;
    FOR_ROW(a,input,start+row);
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
__global__ void radix_update32_live(Input input,int start,SelectionState* state,const int* bins,uint64_t* selected,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;
    FOR_ROW(a,input,start+row);
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
__global__ void radix_compact32_live(Input input,int start,const float* scores,SelectionState* state,uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x,lane=t&31;
    FOR_ROW(a,input,start+row);
    if(chunk*RADIX_CHUNK>=a.live_context)return;   // key 0 is never taken
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
static void threshold_compact32_live(const G6Attn& a,Input input,Work& w,int start,int count,cudaStream_t stream){
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;int* bins=(int*)w.so;
    for(int pass=0;pass<8;pass++){
        radix_histogram32_live<<<dim3(chunks,count),256,0,stream>>>(input,start,w.si,w.selection,bins,pass);
        radix_update32_live<<<count,256,0,stream>>>(input,start,w.selection,bins,w.selected,pass);
    }
    radix_compact32_live<<<dim3(chunks,count),256,0,stream>>>(input,start,w.si,w.selection,w.selected);
}
// -----------------------------------------------------------------------------
// Single-launch exact top-2048 decode selector (per-row live_context + CP-compact):
// Replaces 13-18 serial radix launches with 1 CTA/row (fits 2 CTAs/SM), bounded to
// each row's exact min(context, live_context, pos + 1) via FOR_ROW and iterating
// strictly over CP_MINE pages when cp_nr > 0. Optionally fuses cp_cand_pack into
// the output store (cand_out != nullptr). Bit-exact to select_index / cp_cand_pack.
// -----------------------------------------------------------------------------
namespace decsel2 {
constexpr int W = 11, NB = 1 << W, CAP_SMEM = 3072;

__device__ __forceinline__ float inv_ord(uint32_t o) {
    if (o <= 0x00800000u) return -3.4028234663852886e+38f;
    if (o >= 0xff800000u) return  3.4028234663852886e+38f;
    const uint32_t b = (o & 0x80000000u) ? (o ^ 0x80000000u) : ~o;
    return __uint_as_float(b);
}
__device__ __forceinline__ uint64_t make_key(float s, int id) {
    return selection_key32(ordered_score(s), id);
}
struct St { uint64_t pp, p; int lp, l, need, fin, nout, ncand, B, above, hb, total; };

template<int BT>
struct SmemSel {
    uint64_t out[TOP];
    uint64_t cand[CAP_SMEM];
    int hist[NB];
    typename cub::BlockScan<int, BT, cub::BLOCK_SCAN_WARP_SCANS>::TempStorage scan;
    St s;
};
template<int BT>
union Smem {
    SmemSel<BT> sel;
    typename cub::BlockRadixSort<uint64_t, BT, TOP / BT>::TempStorage sort;
};
static_assert(sizeof(Smem<256>) <= 50688, "decsel2::Smem<256> fits 2 CTAs/SM");

__device__ __forceinline__ int c2pos(int c, int cp_nr, int cp_rank, int ps_shift, int ps_mask) {
    return cp_nr ? ((((c >> ps_shift) * cp_nr + cp_rank) << ps_shift) + (c & ps_mask)) : c;
}
__device__ __forceinline__ void load_row4(const float* x, int len, bool v4, int c0,
                                          int cp_nr, int cp_rank, int ps_shift, int ps_mask, uint64_t* k) {
    if (v4 && c0 + 3 < len) {
        const int p0 = c2pos(c0, cp_nr, cp_rank, ps_shift, ps_mask);
        const float4 q = __ldg(reinterpret_cast<const float4*>(x + p0));
        k[0] = make_key(q.x, p0);     k[1] = make_key(q.y, p0 + 1);
        k[2] = make_key(q.z, p0 + 2); k[3] = make_key(q.w, p0 + 3);
    } else {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (c0 + e < len) {
                const int p = c2pos(c0 + e, cp_nr, cp_rank, ps_shift, ps_mask);
                k[e] = make_key(__ldg(x + p), p);
            } else k[e] = 0ull;
        }
    }
}
__device__ __forceinline__ void append_out(bool take, uint64_t k, int* cnt, uint64_t* buf) {
    const unsigned m = __ballot_sync(0xffffffffu, take);
    if (!m) return;
    const int lane = threadIdx.x & 31, lead = __ffs(m) - 1;
    int base = 0;
    if (lane == lead) base = atomicAdd(cnt, __popc(m));
    base = __shfl_sync(0xffffffffu, base, lead);
    if (take) { const int slot = base + __popc(m & ((1u << lane) - 1)); if (slot < TOP) buf[slot] = k; }
}
__device__ __forceinline__ void append_cand(bool take, uint64_t k, int* cnt, uint64_t* sc, uint64_t* gc, int cap_tot) {
    const unsigned m = __ballot_sync(0xffffffffu, take);
    if (!m) return;
    const int lane = threadIdx.x & 31, lead = __ffs(m) - 1;
    int base = 0;
    if (lane == lead) base = atomicAdd(cnt, __popc(m));
    base = __shfl_sync(0xffffffffu, base, lead);
    if (take) {
        const int slot = base + __popc(m & ((1u << lane) - 1));
        if (slot < CAP_SMEM) sc[slot] = k;
        else if (slot < cap_tot) gc[slot - CAP_SMEM] = k;
    }
}
template<int BT, class Load, class F>
__device__ __forceinline__ void each4(int n4, Load load, F f) {
    constexpr int U = 4;
    for (int b = 0; b < n4; b += BT * U) {
        uint64_t k[U][4];
        #pragma unroll
        for (int u = 0; u < U; ++u) load(b + u * BT + (int)threadIdx.x, k[u]);
        #pragma unroll
        for (int u = 0; u < U; ++u)
            #pragma unroll
            for (int e = 0; e < 4; ++e) f(k[u][e]);
    }
}
template<int BT>
struct Row {
    SmemSel<BT>& m; const float* x; int len; bool v4;
    int cp_nr, cp_rank, ps_shift, ps_mask;
    uint64_t* gc; int cap_tot;

    __device__ void clear(bool zero_cand) {
        for (int j = threadIdx.x; j < NB; j += BT) m.hist[j] = 0;
        if (threadIdx.x == 0 && zero_cand) m.s.ncand = 0;
        __syncthreads();
    }
    __device__ void choose() {
        constexpr int G = NB / BT;
        const int t = threadIdx.x;
        int c = 0;
        #pragma unroll
        for (int j = 0; j < G; ++j) c += m.hist[NB - 1 - t * G - j];
        int before, total;
        cub::BlockScan<int, BT, cub::BLOCK_SCAN_WARP_SCANS>(m.scan).ExclusiveSum(c, before, total);
        const int need = m.s.need;
        if (before < need && before + c >= need) {
            int acc = before;
            for (int j = 0; j < G; ++j) {
                const int b = NB - 1 - t * G - j, h = m.hist[b];
                if (acc + h >= need) { m.s.B = b; m.s.above = acc; m.s.hb = h; break; }
                acc += h;
            }
        }
        if (t == 0) m.s.total = total;
        __syncthreads();
    }
    __device__ void descend() {
        St& s = m.s;
        s.pp = s.p; s.lp = s.l;
        if (s.total < s.need) { s.fin = 1; return; }
        const int w = min(W, 64 - s.l);
        s.p = s.l ? (s.p << w) | (uint64_t)s.B : (uint64_t)s.B;
        s.l += w; s.need -= s.above; s.fin = (s.hb == s.need);
    }
    __device__ void pass(int src, bool emit_ok, bool collect, int samp_stride, int sb) {
        const uint64_t pp = m.s.pp, p = m.s.p;
        const int lp = m.s.lp, l = m.s.l, fin = m.s.fin;
        const int w = min(W, 64 - l), sh = 64 - l - w;
        const uint32_t dmask = (1u << w) - 1u;
        auto f = [&](uint64_t k) {
            const bool in = (k != 0ull) && (lp == 0 || (k >> (64 - lp)) == pp);
            bool em = false, hs = false;
            if (in) {
                const uint64_t u = l ? (k >> (64 - l)) : 0ull;
                if (fin) em = (l == lp) || (u >= p);
                else { em = (l != 0) && (u > p); hs = (l == 0) || (u == p); }
                em &= emit_ok;
            }
            append_out(em, k, &m.s.nout, m.out);
            if (__any_sync(0xffffffffu, hs)) {
                const unsigned d = hs ? (unsigned(k >> sh) & dmask) : (unsigned)NB;
                const unsigned peers = __match_any_sync(0xffffffffu, d);
                if (hs && (threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(m.hist + d, __popc(peers));
                if (collect) append_cand(hs, k, &m.s.ncand, m.cand, gc, cap_tot);
            }
        };
        if (src == 0) {
            each4<BT>((len + 3) / 4, [&](int i4, uint64_t* k) {
                load_row4(x, len, v4, 4 * i4, cp_nr, cp_rank, ps_shift, ps_mask, k);
            }, f);
        } else if (src == 1) {
            const int sb4 = sb >> 2;
            const int nblk = (len + sb - 1) / sb;
            const int ns = (nblk + samp_stride - 1) / samp_stride;
            each4<BT>(ns * sb4, [&](int i4, uint64_t* k) {
                const int c0 = (i4 / sb4) * sb * samp_stride + (i4 % sb4) * 4;
                load_row4(x, len, v4, c0, cp_nr, cp_rank, ps_shift, ps_mask, k);
            }, f);
        } else {
            const int n = m.s.ncand;
            each4<BT>((n + 3) / 4, [&](int i4, uint64_t* k) {
                const int b = 4 * i4;
                if (b + 3 < n && b + 3 < CAP_SMEM) {
                    const ulonglong2 v0 = *reinterpret_cast<const ulonglong2*>(m.cand + b);
                    const ulonglong2 v1 = *reinterpret_cast<const ulonglong2*>(m.cand + b + 2);
                    k[0] = v0.x; k[1] = v0.y; k[2] = v1.x; k[3] = v1.y;
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int idx = b + e;
                        if (idx >= n) k[e] = 0ull;
                        else if (idx < CAP_SMEM) k[e] = m.cand[idx];
                        else k[e] = gc[idx - CAP_SMEM];
                    }
                }
            }, f);
        }
    }
    __device__ void levels(int src) {
        for (;;) {
            const bool collect = (src == 0) && (m.s.l > 0);
            clear(collect);
            pass(src, true, collect, 1, 256);
            __syncthreads();
            if (m.s.fin) return;
            choose();
            const bool to_cand = collect && (m.s.total <= cap_tot);
            __syncthreads();
            if (threadIdx.x == 0) descend();
            __syncthreads();
            if (to_cand) src = 2;
        }
    }
};

template<int BT, class Input>
__global__ void __launch_bounds__(BT, 2) select_decode_v2(
    Input input, int start, const float* __restrict__ scores,
    uint64_t* __restrict__ cand_g, int capg, uint64_t* __restrict__ cand_out)
{
    auto a = resolve(input);
    if (!a.active) return;
    const int r = blockIdx.x, row_abs = start + r, t = threadIdx.x;
    FOR_ROW(a, input, row_abs);

    constexpr int ITEMS = TOP / BT;
    using SortT = cub::BlockRadixSort<uint64_t, BT, ITEMS>;
    extern __shared__ __align__(16) unsigned char raw_smem[];
    Smem<BT>& smu = *reinterpret_cast<Smem<BT>*>(raw_smem);
    SmemSel<BT>& m = smu.sel;

    const int n_pos = max(0, min(min(a.context, a.live_context), a.positions[row_abs] + 1));
#if defined(G6_CP) && G6_CP
    const int cp_nr = a.cp_nr, cp_rank = a.cp_rank;
#else
    const int cp_nr = 0, cp_rank = 0;
#endif
    const int ps = a.page_size, ps_mask = ps - 1, ps_shift = __ffs(ps) - 1;
    int n = n_pos;
    if (cp_nr > 0) {
        const int cyc = cp_nr << ps_shift;
        const int full = n_pos / cyc;
        const int rem = n_pos - full * cyc;
        n = (full << ps_shift) + max(0, min(ps, rem - (cp_rank << ps_shift)));
    }

    const float* x = scores + (int64_t)r * a.context;
    const bool v4 = (((uintptr_t)x & 15u) == 0u) && (!cp_nr || ps >= 4);
    uint64_t* gc = cand_g + (int64_t)r * capg;
    const int cap_tot = CAP_SMEM + capg;

    int* o = a.indices + (int64_t)row_abs * TOP;
    uint64_t* co = cand_out ? (cand_out + (int64_t)row_abs * TOP) : nullptr;

    if (n <= TOP) {
        uint64_t keys[ITEMS];
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int c = t * ITEMS + j;
            if (c < n) {
                const int p = c2pos(c, cp_nr, cp_rank, ps_shift, ps_mask);
                keys[j] = make_key(__ldg(x + p), p);
            } else keys[j] = 0ull;
        }
        SortT(smu.sort).SortDescending(keys);
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const uint64_t k = keys[j];
            o[t * ITEMS + j] = k ? int(0xffffffffu - uint32_t(k)) : -1;
            if (co) co[t * ITEMS + j] = k;
        }
        return;
    }

    Row<BT> rw{m, x, n, v4, cp_nr, cp_rank, ps_shift, ps_mask, gc, cap_tot};
    if (t == 0) { m.s = St{}; m.s.need = TOP; }
    __syncthreads();

    const int samp = (n >= 98304) ? 32 : (n >= 49152 ? 16 : (n >= 20480 ? 8 : 4));
    constexpr int SB = 256;
    const int mu = TOP / samp;
    const int k_samp = mu + ((samp == 32) ? 32 : (samp == 16 ? 48 : (samp == 8 ? 64 : 128)));

    if (t == 0) m.s.need = k_samp;
    __syncthreads();

    uint64_t bound = 1ull;
    rw.clear(false);
    rw.pass(1, false, false, samp, SB);
    __syncthreads();
    rw.choose();
    if (m.s.total >= k_samp) {
        const uint64_t B0 = m.s.B;
        const int need1 = k_samp - m.s.above;
        bound = B0 << (64 - W);
        if (m.s.hb != need1) {
            __syncthreads();
            if (t == 0) { m.s.p = B0; m.s.l = W; m.s.need = need1; }
            rw.clear(false);
            rw.pass(1, false, false, samp, SB);
            __syncthreads();
            rw.choose();
            bound = ((B0 << W) | (uint64_t)m.s.B) << (64 - 2 * W);
        }
    }
    __syncthreads();
    if (t == 0) { m.s = St{}; m.s.need = TOP; }
    __syncthreads();

    const float bound_f = (bound == 1ull) ? -3.4028234663852886e+38f : inv_ord((uint32_t)(bound >> 32));
    const int n4 = (n + 3) / 4;
    constexpr int UF = 4;
    for (int b = 0; b < n4; b += BT * UF) {
        float4 qv[UF];
        int p0s[UF];
        bool fast4[UF];
        #pragma unroll
        for (int u = 0; u < UF; ++u) {
            const int c0 = 4 * (b + u * BT + t);
            if (v4 && c0 + 3 < n) {
                const int p0 = c2pos(c0, cp_nr, cp_rank, ps_shift, ps_mask);
                p0s[u] = p0;
                fast4[u] = true;
                qv[u] = __ldg(reinterpret_cast<const float4*>(x + p0));
            } else {
                p0s[u] = c0;
                fast4[u] = false;
                qv[u].x = (c0 + 0 < n) ? __ldg(x + c2pos(c0 + 0, cp_nr, cp_rank, ps_shift, ps_mask)) : -INFINITY;
                qv[u].y = (c0 + 1 < n) ? __ldg(x + c2pos(c0 + 1, cp_nr, cp_rank, ps_shift, ps_mask)) : -INFINITY;
                qv[u].z = (c0 + 2 < n) ? __ldg(x + c2pos(c0 + 2, cp_nr, cp_rank, ps_shift, ps_mask)) : -INFINITY;
                qv[u].w = (c0 + 3 < n) ? __ldg(x + c2pos(c0 + 3, cp_nr, cp_rank, ps_shift, ps_mask)) : -INFINITY;
            }
        }
        #pragma unroll
        for (int u = 0; u < UF; ++u) {
            const float4 q = qv[u];
            const bool p0 = (q.x >= bound_f) && (q.x < INFINITY);
            const bool p1 = (q.y >= bound_f) && (q.y < INFINITY);
            const bool p2 = (q.z >= bound_f) && (q.z < INFINITY);
            const bool p3 = (q.w >= bound_f) && (q.w < INFINITY);
            if (__any_sync(0xffffffffu, p0 | p1 | p2 | p3)) {
                const int base_i = p0s[u];
                const bool f4 = fast4[u];
                if (__any_sync(0xffffffffu, p0)) {
                    const int id = f4 ? (base_i + 0) : c2pos(base_i + 0, cp_nr, cp_rank, ps_shift, ps_mask);
                    append_cand(p0, p0 ? make_key(q.x, id) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                }
                if (__any_sync(0xffffffffu, p1)) {
                    const int id = f4 ? (base_i + 1) : c2pos(base_i + 1, cp_nr, cp_rank, ps_shift, ps_mask);
                    append_cand(p1, p1 ? make_key(q.y, id) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                }
                if (__any_sync(0xffffffffu, p2)) {
                    const int id = f4 ? (base_i + 2) : c2pos(base_i + 2, cp_nr, cp_rank, ps_shift, ps_mask);
                    append_cand(p2, p2 ? make_key(q.z, id) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                }
                if (__any_sync(0xffffffffu, p3)) {
                    const int id = f4 ? (base_i + 3) : c2pos(base_i + 3, cp_nr, cp_rank, ps_shift, ps_mask);
                    append_cand(p3, p3 ? make_key(q.w, id) : 0ull, &m.s.ncand, m.cand, gc, cap_tot);
                }
            }
        }
    }
    __syncthreads();

    if (m.s.ncand <= cap_tot && (m.s.ncand >= TOP || bound == 1ull)) {
        if (m.s.ncand <= TOP) {
            for (int j = t; j < m.s.ncand; j += BT) m.out[j] = (j < CAP_SMEM) ? m.cand[j] : gc[j - CAP_SMEM];
            if (t == 0) m.s.nout = m.s.ncand;
        } else {
            rw.levels(2);
        }
    } else {
        __syncthreads();
        if (t == 0) { m.s = St{}; m.s.need = TOP; }
        __syncthreads();
        rw.levels(0);
    }
    __syncthreads();
    const int nout = min(m.s.nout, TOP);

    uint64_t keys[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int idx = t * ITEMS + j;
        keys[j] = (idx < nout) ? m.out[idx] : 0ull;
    }
    __syncthreads();
    SortT(smu.sort).SortDescending(keys);

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const uint64_t k = keys[j];
        o[t * ITEMS + j] = k ? int(0xffffffffu - uint32_t(k)) : -1;
        if (co) co[t * ITEMS + j] = k;
    }
}

template<class Input>
static inline cudaError_t launch(const G6Attn& a, Input input, Work& w, int start, int count,
                                 uint64_t* cand_out, cudaStream_t stream)
{
    if (count <= 0) return cudaSuccess;
    static const bool once =
        cudaFuncSetAttribute(select_decode_v2<256, Input>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem<256>)) == cudaSuccess;
    (void)once;
    // Under SEL32, w.ii (R * context * 4 B = R * (context/2) uint64s) is unused and serves as overflow cand_g.
    uint64_t* cand_g = reinterpret_cast<uint64_t*>(w.ii);
    const int capg = std::min(65536, a.context / 2);
    select_decode_v2<256, Input><<<count, 256, sizeof(Smem<256>), stream>>>(input, start, w.si, cand_g, capg, cand_out);
    return cudaGetLastError();
}
} // namespace decsel2

// the DEC_IDX31 select: live-context-bounded 1-launch exact selector past TOP (works for both solo and S2 batched steps)
template<class Input>
static cudaError_t select_index31(const G6Attn& a,Input input,Work& w,int start,int count,cudaStream_t stream,int rank=0,uint64_t* cand_out=nullptr){
    if(a.context<=TOP)return select_index(a,input,w,start,count,stream,rank);
    return decsel2::launch(a,input,w,start,count,cand_out,stream);
}
#endif
