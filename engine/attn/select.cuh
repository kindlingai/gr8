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
