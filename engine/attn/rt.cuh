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
// the pool's room stay absent (their reads take the local copy) and count in g.local.
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
        if(p<0||p>=ar.page_count||!a.rt_page[p])continue;
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
            ent[j]=RtEnt{g.base+(unsigned long long)k*g.rec,(unsigned)s,0u};
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
