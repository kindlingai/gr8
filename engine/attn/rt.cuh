// Remote attention rows (attn.h G6RtGroup, runtime/rt_rows.cuh): the resolve after a group's selection and the wait
// before its scores. Included inside namespace ga by attn.cu.
#pragma once

struct RtEnt { unsigned long long off; unsigned slot, pad; };   // kvman-rs g6kv_rows_ent
struct RtReq { unsigned long long seq; unsigned n, pad; };      // g6kv_rows_req, cap RtEnt follow

__device__ __forceinline__ unsigned long long rt_ld_acquire(const unsigned long long* p){
    unsigned long long v;asm volatile("ld.acquire.sys.global.u64 %0, [%1];":"=l"(v):"l"(p));return v;
}
__device__ __forceinline__ void rt_st_release(unsigned long long* p,unsigned long long v){
    asm volatile("st.release.sys.global.u64 [%0], %1;"::"l"(p),"l"(v):"memory");
}

// grid 1, block 1024: rows 0..n-1 of a.indices (this layer's own selection), each with its own table (S2). Hits stamp
// their slot; each missed row takes a slot not used this step, in clock order from the group's hand (its old row, if
// any, is evicted), and goes to the request ring. Then the doorbell (seq), only when something missed. Misses past
// the pool's room stay absent (their reads take the local copy, or fail on a released page) and count in g.local.
template<class Input>
__global__ void __launch_bounds__(1024) rt_resolve(Input input,int n){
    auto a=resolve(input);G6RtGroup& g=*a.rt_group;const int t=threadIdx.x;
    if(!a.active){if(t==0)g.n=0;return;}
    using Scan=cub::BlockScan<int,1024>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int nm,got,adv;
    const unsigned long long cur=*g.step;
    if(t==0){nm=0;got=0;adv=0;}
    __syncthreads();
    for(int i=t;i<n*TOP;i+=1024){
        const int r=i/TOP,id=a.indices[(int64_t)r*TOP+i%TOP];
        DeviceAttn ar=a;FOR_ROW(ar,input,r);
        if(id<0||id>a.positions[r]||id>=ar.live_context||id/ar.page_size>=ar.table_size)continue;
        const int p=ar.block_table[id/ar.page_size];
        if(p<0||p>=ar.page_count+a.rt_vpages||!a.rt_page[p])continue;
        const int k=p*ar.page_size+id%ar.page_size,s=g.pos2slot[k];
        if(s>=0){g.last[s]=cur;continue;}
        if(s==-1&&atomicCAS(&g.pos2slot[k],-1,-2)==-1){const int m=atomicAdd(&nm,1);if(m<g.cap)g.miss[m]=k;else g.pos2slot[k]=-1;}
    }
    __syncthreads();   // every hit is stamped before any slot is taken
    const int need=min(nm,g.cap);
    RtReq* req=(RtReq*)g.req;RtEnt* ent=(RtEnt*)(req+1);
    for(int b=0;b<g.slots&&got<need;b+=1024){
        const int s=(g.hand+b+t)%g.slots;
        const bool free=b+t<g.slots&&g.last[s]!=cur;
        int rank,total;Scan(scan).ExclusiveSum(free?1:0,rank,total);
        const int j=got+rank;
        if(free&&j<need){
            const int k=g.miss[j],old=g.slot2pos[s];
            if(old>=0)g.pos2slot[old]=-1;
            g.slot2pos[s]=k;g.pos2slot[k]=s;g.last[s]=cur;
            ent[j]=RtEnt{g.base+(unsigned long long)(k-g.kbase)*g.rec,(unsigned)s,0u};
            atomicMax(&adv,b+t+1);
        }
        __syncthreads();
        if(t==0)got=min(need,got+total);
        __syncthreads();
    }
    for(int j=got+t;j<min(nm,g.cap);j+=1024)g.pos2slot[g.miss[j]]=-1;
    if(t==0&&got<nm)g.local+=nm-got;
    __threadfence_system();
    __syncthreads();
    if(t==0){
        g.hand=(g.hand+adv)%g.slots;g.n=got;
        if(got){req->n=(unsigned)got;g.seq++;__threadfence_system();rt_st_release(&req->seq,g.seq);}
    }
}
// grid 1, block 32: the helper has written every row of the group's last request (error 0x10000: a read failed,
// 0x20000: timeout)
__global__ void __launch_bounds__(32) rt_wait(const G6RtGroup* gp,unsigned* err){
    if(threadIdx.x)return;
    const G6RtGroup& g=*gp;
    if(!g.n)return;
    const long long t0=clock64();
    for(unsigned it=0;;++it){
        const unsigned long long d=rt_ld_acquire(g.done);
        if(d==g.seq)return;
        if(d==~0ull){atomicOr(err,0x10000u);return;}
        if((it&1023u)==0&&clock64()-t0>20000000000LL){atomicOr(err,0x20000u);return;}
    }
}
#if !G6_KV4
// G6_RT_FMT=int4: the store's records hold rows in the INT4 layout (kvfmt.h int4row), while the pages and the slots stay
// FP8, so no attention kernel changes. Warp per row; x[k] is dim lane + 32 k (fwht16's order), so a lane's x[k] has INT4
// scale k and FP8 scale k / 4. The rope key is copied, 4 B a lane.
namespace f8=g6kv::fp8row;namespace i4=g6kv::int4row;
// FP8 -> f32 -> H x -> INT4 per 32 dims, as append_kv's G6_KV4 branch quantizes
__device__ __forceinline__ void rt_pack_row(const uint8_t* s,uint8_t* d,int lane){
    const float* sc=(const float*)(s+f8::SCALE_OFF);float x[16];
    #pragma unroll
    for(int k=0;k<16;k++)x[k]=e4(s[lane+32*k])*sc[k>>2];
    fwht16(x,lane);
    #pragma unroll
    for(int k=0;k<16;k++){
        half h;const int c=kv4_code(x[k]*HSCALE,h),hi=__shfl_down_sync(0xffffffff,c,1);
        if(!(lane&1))d[(lane>>1)+16*k]=(uint8_t)(c|(hi<<4));
        if(lane==0)((half*)(d+i4::SCALE_OFF))[k]=h;
    }
    ((uint32_t*)(d+i4::ROPE_OFF))[lane]=((const uint32_t*)(s+f8::ROPE_OFF))[lane];
}
// INT4 -> f32 -> H x (H is its own inverse) -> FP8 per 128 dims, as append_kv's FP8 branch quantizes
__device__ __forceinline__ void rt_unpack_row(const uint8_t* s,uint8_t* d,int lane){
    float x[16];
    #pragma unroll
    for(int k=0;k<16;k++){const int dd=lane+32*k;
        x[k]=(float)(((s[dd>>1]>>((dd&1)*4))&15)-8)*__half2float(((const half*)(s+i4::SCALE_OFF))[k]);}
    fwht16(x,lane);
    float sc[4];
    #pragma unroll
    for(int g=0;g<4;g++){float m=0;
        #pragma unroll
        for(int k=4*g;k<4*g+4;k++){x[k]*=HSCALE;m=fmaxf(m,fabsf(x[k]));}
        sc[g]=fp8_scale(warp_max(m));}
    #pragma unroll
    for(int k=0;k<16;k++)d[lane+32*k]=q4(x[k]/sc[k>>2]);
    if(lane<4)((float*)(d+f8::SCALE_OFF))[lane]=lane==0?sc[0]:lane==1?sc[1]:lane==2?sc[2]:sc[3];
    ((uint32_t*)(d+f8::ROPE_OFF))[lane]=((const uint32_t*)(s+i4::ROPE_OFF))[lane];
}
template<bool PACK>
__global__ void __launch_bounds__(256) rt_convert(const uint8_t* src,size_t ss,uint8_t* dst,size_t ds,int n){
    const int r=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;if(r>=n)return;
    if(PACK)rt_pack_row(src+r*ss,dst+r*ds,lane);else rt_unpack_row(src+r*ss,dst+r*ds,lane);
}
// after rt_wait, before any scores: every slot of the group's last request, its INT4 record (at the slot's tail) -> its
// FP8 rows, in place. A block per slot at a time: the record goes to shared memory first, since the FP8 rows it becomes
// cover it. A fixed grid, for capture; nothing to do when the resolve missed nothing.
constexpr int RT_NLMAX=8;   // layers of a group the shared record holds (rt_rows.cuh rt_build checks)
__global__ void __launch_bounds__(256) rt_unpack(const G6RtGroup* gp){
    const G6RtGroup& g=*gp;const int n=g.n;if(!n)return;
    __shared__ __align__(16) uint8_t rec[RT_NLMAX*i4::ROW];
    const RtEnt* ent=(const RtEnt*)((const RtReq*)g.req+1);
    const int lane=threadIdx.x&31,w=threadIdx.x/32;
    const size_t srec=(size_t)g.nl*f8::ROW;
    for(int j=blockIdx.x;j<n;j+=gridDim.x){
        uint8_t* slot=g.out+ent[j].slot*srec;
        const uint4* src=(const uint4*)(slot+g.soff);
        for(int k=threadIdx.x;k<g.nl*i4::ROW/16;k+=blockDim.x)((uint4*)rec)[k]=src[k];
        __syncthreads();
        for(int l=w;l<g.nl;l+=blockDim.x/32)rt_unpack_row(rec+l*i4::ROW,slot+l*f8::ROW,lane);
        __syncthreads();
    }
}
#endif
