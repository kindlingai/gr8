// Independently written exact selector. Included inside namespace ga.
// Finite FP32 -> monotonically increasing unsigned key; both zeros canonical.
// Nonfinite scores are masked by index_scores and never selected here.
__device__ uint32_t ordered_score(float score){
    uint32_t b=__float_as_uint(score);
    if((b&0x7fffffffu)>=0x7f800000u)return 0;
    if((b&0x7fffffffu)==0)b=0;
    return (b&0x80000000u)?~b:(b^0x80000000u);
}
__device__ uint64_t selection_key(uint32_t score,int id){
    // This fast path is bounded to context<=32768; 16 ID bits are sufficient.
    return score?((uint64_t)score<<16)|(0xffffu-(uint32_t)id):0;
}
struct SelectionState {uint64_t cutoff;int rank,complete,used;};
constexpr int RADIX_CHUNK=1024;
// Cooperative within each CTA, independent across score chunks. Kernel boundaries
// provide global ordering; there are no cross-CTA spin waits.
template<class Input>
__global__ void radix_histogram(Input input,const float* scores,SelectionState* state,int* bins,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK,shift=40-pass*8;
    if(pass&&state[row].complete)return;
    uint64_t prefix=pass?state[row].cutoff:0,mask=~0ull<<(shift+8);
    __shared__ int hist[8*256];
    for(int j=t;j<8*256;j+=256)hist[j]=0;
    __syncthreads();
    #pragma unroll
    for(int j=0;j<4;j++){
        int id=chunk*RADIX_CHUNK+j*256+t;
        uint64_t key=id<a.context?selection_key(ordered_score(scores[(int64_t)row*a.context+id]),id):0;
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
template<class Input>
__global__ void radix_update(Input input,SelectionState* state,const int* bins,uint64_t* selected,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;
    if(pass&&state[row].complete)return;
    const uint64_t prefix=pass?state[row].cutoff:0;
    const int need=pass?state[row].rank:TOP;
    if(pass==0){for(int j=t;j<TOP;j+=256)selected[(int64_t)row*TOP+j]=0;if(t==0)state[row].used=0;}
    int count=0;
    for(int c=0;c<chunks;c++)count+=bins[((int64_t)row*chunks+c)*256+255-t];
    using Scan=cub::BlockScan<int,256>;__shared__ typename Scan::TempStorage temp;
    int before;Scan(temp).ExclusiveSum(count,before);
    if(before<need&&before+count>=need){
        state[row].cutoff=prefix|((uint64_t)(255-t)<<(40-pass*8));
        state[row].rank=need-before;state[row].complete=(need-before)==count;
    }
}
template<class Input>
__global__ void radix_compact(Input input,const float* scores,SelectionState* state,uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x,lane=t&31;
    const uint64_t threshold=state[row].cutoff;
    #pragma unroll
    for(int j=0;j<4;j++){
        int id=chunk*RADIX_CHUNK+j*256+t;
        uint64_t key=id<a.context?selection_key(ordered_score(scores[(int64_t)row*a.context+id]),id):0;
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
// Sorting only TOP composite keys makes arbitrary compaction order irrelevant.
// For short rows, bypass selection and sort the whole (padded) row in one CTA.
template<bool COMPACT,class Input>
__global__ void sort_selected(Input input,int start,const float* scores,const uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
    constexpr int BT=256,ITEMS=8;using Sort=cub::BlockRadixSort<uint64_t,BT,ITEMS>;
    __shared__ typename Sort::TempStorage temp;
    int row=blockIdx.x,t=threadIdx.x;uint64_t keys[ITEMS];
    #pragma unroll
    for(int j=0;j<ITEMS;j++){
        int id=t*ITEMS+j;
        if(COMPACT)keys[j]=selected[(int64_t)row*TOP+id];
        else keys[j]=id<a.context?selection_key(ordered_score(scores[(int64_t)row*a.context+id]),id):0;
    }
    Sort(temp).SortDescending(keys,0,48);
    #pragma unroll
    for(int j=0;j<ITEMS;j++)a.indices[(int64_t)(start+row)*TOP+t*ITEMS+j]=keys[j]?int(0xffffu-uint32_t(keys[j]&0xffffu)):-1;
}
// GR8_ATTN_SEL2=<CH> (attn_sel2; decode run_body): the selector's final sort (sort_selected<false> / <true>, sort_selected32)
// as a rank by count over CH CTAs per row. The nonzero keys of a row are distinct (their low bits hold the token id), so a
// nonzero key's place in the descending sort is the number of larger keys of the row, and the nonzero keys fill places
// 0 .. nnz - 1; zero keys (nonfinite / past the context / unfilled) sort below them and all map to -1. Every CTA stages the
// row's TOP keys in shared memory and counts for its KC = TOP/CH own keys, each over NP = 256/KC shares of the row (summed),
// then writes the own key's id at its rank and -1 at its own places >= nnz: every place written once, the bytes of the sort.
// MODE 0 (<false>): the key of place j is ordered_score << 16 | (0xffff - j), so key_j > key_i <=> s_j > s_i, or s_j == s_i
// and j < i: counted in 32 bits as s_j > (j < i ? s_i - 1 : s_i) (s_i >= 1 for a nonzero key; a zero key counts nothing
// and writes nothing); a warp's own keys are 32 consecutive places, so outside the warp's own block of 32 the threshold is
// warp-uniform. MODE 1 (<true>, 48-bit keys) / 2 (sort_selected32, 64-bit): the keys compared. grid dim3(CH, count), 256.
template<int MODE,int CH,class Input>
__global__ void __launch_bounds__(256) rank_selected(Input input,int start,const float* scores,const uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
    static_assert(CH==8||CH==16||CH==32||CH==64,"CH in {8,16,32,64}");
    constexpr int KC=TOP/CH,NP=256/KC,SH=TOP/NP;   // own keys per CTA, shares per key, comparands per share
    const int row=blockIdx.y,c=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
    __shared__ __align__(16) uint32_t s32[MODE==0?TOP:4];
    __shared__ __align__(16) uint64_t k64[MODE==0?2:TOP];
    __shared__ int cnt[256],nzw[8];
    int nz=0;
    #pragma unroll
    for(int u=0;u<TOP/256;u++){const int j=u*256+t;
        if constexpr(MODE==0){const uint32_t s=j<a.context?ordered_score(scores[(int64_t)row*a.context+j]):0u;s32[j]=s;nz+=s!=0;}
        else{const uint64_t k=selected[(int64_t)row*TOP+j];k64[j]=k;nz+=k!=0;}}
    nz=__reduce_add_sync(0xffffffffu,nz);if(lane==0)nzw[warp]=nz;
    __syncthreads();
    int nnz=0;
    #pragma unroll
    for(int w=0;w<8;w++)nnz+=nzw[w];
    const int k=t%KC,i=c*KC+k,j0=(t/KC)*SH;   // own key i, comparands j0 .. j0 + SH - 1
    int n0=0,n1=0,n2=0,n3=0;   // four independent counts (one serial add chain each)
    if constexpr(MODE==0){
        const uint32_t si=s32[i],ib=i&~31;
        for(int jb=j0;jb<j0+SH;jb+=32){
            if(jb==ib){
                #pragma unroll
                for(int e=0;e<32;e+=4){const uint4 v=*(const uint4*)&s32[jb+e];
                    n0+=v.x>(jb+e<i?si-1:si);n1+=v.y>(jb+e+1<i?si-1:si);n2+=v.z>(jb+e+2<i?si-1:si);n3+=v.w>(jb+e+3<i?si-1:si);}}
            else{const uint32_t th=jb<ib?si-1:si;
                #pragma unroll
                for(int e=0;e<32;e+=4){const uint4 v=*(const uint4*)&s32[jb+e];n0+=v.x>th;n1+=v.y>th;n2+=v.z>th;n3+=v.w>th;}}}}
    else{const uint64_t ki=k64[i];
        #pragma unroll 4
        for(int j=j0;j<j0+SH;j+=4){const ulonglong2 v=*(const ulonglong2*)&k64[j],w=*(const ulonglong2*)&k64[j+2];
            n0+=v.x>ki;n1+=v.y>ki;n2+=w.x>ki;n3+=w.y>ki;}}
    cnt[t]=n0+n1+n2+n3;
    __syncthreads();
    if(t<KC){int rk=0;
        #pragma unroll
        for(int p=0;p<NP;p++)rk+=cnt[p*KC+t];
        int* out=a.indices+(int64_t)(start+row)*TOP;
        if constexpr(MODE==0){if(s32[i])out[rk]=i;}
        else if constexpr(MODE==1){const uint64_t ki=k64[i];if(ki)out[rk]=int(0xffffu-uint32_t(ki&0xffffu));}
        else{const uint64_t ki=k64[i];if(ki)out[rk]=int(0xffffffffu-uint32_t(ki));}
        if(i>=nnz)out[i]=-1;}
}
#ifdef SEL32
// SEL32: the same exact selector past context 32768, so no context size runs
// the full CUB sort. 64-bit key = ordered_score (CUB's float order, both zeros equal as CUB treats them) in the high
// word, 32 ID bits (0xffffffff - id) in the low word -> the order of the stable descending full sort: score desc, then
// id asc. Nonfinite (masked) scores get key 0 and are never selected, as top_from_sort maps them to -1.
__device__ uint64_t selection_key32(uint32_t score,int id){
    return score?((uint64_t)score<<32)|(0xffffffffu-(uint32_t)id):0;
}
template<class Input>
__global__ void radix_histogram32(Input input,const float* scores,SelectionState* state,int* bins,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
    const int chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK,shift=56-pass*8;
    if(pass&&state[row].complete)return;
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
template<class Input>
__global__ void radix_update32(Input input,SelectionState* state,const int* bins,uint64_t* selected,int pass){
    auto a=resolve(input);if(!a.active)return;
    const int row=blockIdx.x,t=threadIdx.x,chunks=(a.context+RADIX_CHUNK-1)/RADIX_CHUNK;
    if(pass&&state[row].complete)return;
    const uint64_t prefix=pass?state[row].cutoff:0;
    const int need=pass?state[row].rank:TOP;
    if(pass==0){for(int j=t;j<TOP;j+=256)selected[(int64_t)row*TOP+j]=0;if(t==0)state[row].used=0;}
    int count=0;
    for(int c=0;c<chunks;c++)count+=bins[((int64_t)row*chunks+c)*256+255-t];
    using Scan=cub::BlockScan<int,256>;__shared__ typename Scan::TempStorage temp;
    int before;Scan(temp).ExclusiveSum(count,before);
    if(before<need&&before+count>=need){
        state[row].cutoff=prefix|((uint64_t)(255-t)<<(56-pass*8));
        state[row].rank=need-before;state[row].complete=(need-before)==count;
    }
}
template<class Input>
__global__ void radix_compact32(Input input,const float* scores,SelectionState* state,uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
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
__global__ void sort_selected32(Input input,int start,const uint64_t* selected){
    auto a=resolve(input);if(!a.active)return;
    constexpr int BT=256,ITEMS=8;using Sort=cub::BlockRadixSort<uint64_t,BT,ITEMS>;
    __shared__ typename Sort::TempStorage temp;
    int row=blockIdx.x,t=threadIdx.x;uint64_t keys[ITEMS];
    #pragma unroll
    for(int j=0;j<ITEMS;j++)keys[j]=selected[(int64_t)row*TOP+t*ITEMS+j];
    Sort(temp).SortDescending(keys,0,64);
    #pragma unroll
    for(int j=0;j<ITEMS;j++)a.indices[(int64_t)(start+row)*TOP+t*ITEMS+j]=keys[j]?int(0xffffffffu-uint32_t(keys[j])):-1;
}
#endif
