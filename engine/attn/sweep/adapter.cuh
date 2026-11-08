#pragma once
#include "fixtures.cuh"
#include <stdexcept>
#include <memory>
namespace sweep_adapter {
using Params = ds41a::AttnP;
struct Shape { int n, pos, ratio, floor, wl, keys; };
inline Shape shape(const std::vector<std::string>& a) {
    if (a.size()!=6) throw std::runtime_error("attention needs rows,pos,ratio,floor,WL,key_rows");
    Shape s{std::stoi(a[0]),std::stoi(a[1]),std::stoi(a[2]),std::stoi(a[3]),std::stoi(a[4]),std::stoi(a[5])};
    if (s.n<1 || s.n>8192 || s.pos<0 || s.pos>1048576-s.n || s.ratio<0 || s.ratio>2 ||
        s.floor<0 || s.floor>s.pos || s.wl<1 || s.wl>128 || s.keys<1 || s.keys>1048576 ||
        (s.floor && !s.ratio) || (s.ratio && s.keys<(s.pos+s.n)/s.ratio))
        throw std::runtime_error("invalid attention shape");
    return s;
}
inline size_t required_bytes(const std::vector<std::string>& a) {
    const auto s=shape(a);
    // Nine allocations: Q, two outputs, both key forms, window, top-k,
    // positions, sink, plus guarded GPU hash workspace. Host registry/snapshot
    // stores only small per-chunk digests. No split buffer in this adapter.
    const size_t largest_input=std::max({size_t(s.n)*16*512*2,
        size_t(s.keys)*512*2,size_t(128+s.n-1)*512*2});
    return size_t(s.n)*16*512*2*3 + size_t(s.keys)*(512*2+288) +
           size_t(128+s.n-1)*512*2 + size_t(s.n)*(512*4+4) +
           16*sizeof(float) + 10*2*sweep_memory::redzone_bytes + sweep_memory::digest_bytes(largest_input);
}
struct Fixture {
    Built b{};
    Keys keys{};
    uint64_t seed = 0;
    std::unique_ptr<sweep_memory::Registry> memory;
};
inline Fixture make_case(const std::vector<std::string>& a, uint64_t seed) {
    const auto s=shape(a); rng.seed(seed);
    Fixture f; f.seed=seed;
    f.memory=std::make_unique<sweep_memory::Registry>();
    sweep_memory::AllocationScope allocation_scope(*f.memory);
    f.keys=make_keys(s.keys,1);
    Case c{"sweep",s.ratio ? s.ratio : 2,s.n,1,s.pos,s.floor,s.wl};
    f.b=build(c,16,f.keys,false);
    if (!s.ratio) f.b.p.TK=0;
    set_rows(f.b.p,f.keys,s.ratio!=0,f.b.out_b);
    auto& m=*f.memory;
    m.describe(f.keys.pk,"keys_packed"); m.describe(f.keys.bf,"keys_bf16");
    m.describe(f.b.q,"q"); m.describe(f.b.kvw,"window_kv");
    m.describe(f.b.sink,"sink"); m.describe(f.b.d_topk,"topk");
    m.describe(f.b.d_pos,"positions"); m.describe(f.b.d_sess,"sessions");
    m.describe(f.b.out_a,"reference_output",false);
    m.describe(f.b.out_b,"candidate_output",false);
    m.describe(f.b.split,"adapter_split_scratch",false);
    m.prepare();
    return f;
}
inline Params params(Fixture& f,bool ref) { auto p=f.b.p; p.out=ref?f.b.out_a:f.b.out_b; return p; }
inline Params params_with_output(Fixture& inputs, Fixture& outputs, bool reference) {
    auto p=params(inputs,reference);
    p.out=reference ? outputs.b.out_a : outputs.b.out_b;
    return p;
}
inline void* output(Fixture& f,bool ref) { return ref?f.b.out_a:f.b.out_b; }
inline size_t output_bytes(const Fixture& f) { return f.b.out_n*2; }
// Refresh is SYNTHETIC only. Captured-data callers must not use this hook and
// claim captured values survived. Pointers/top-k/positions never change; launch
// on the caller's stream, outside timing, then snapshot before any candidate.
__device__ inline uint64_t refresh_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static __global__ void refresh_bf16(bf16* out, size_t n, uint64_t nonce, float scale) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<n) out[i]=__float2bfloat16_rn(float(int(refresh_mix(nonce+i)&65535)-32768)*scale);
}
static __global__ void refresh_sink(float* out, int n, uint64_t nonce) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if (i<n) out[i]=float(int(refresh_mix(nonce+i)&65535)-32768)*(1.0f/16384.0f);
}
static __global__ void refresh_keys(uint8_t* pk, bf16* bf, size_t groups, uint64_t nonce) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=groups) return;
    const size_t row=i/32, group=i%32;
    uint64_t random=refresh_mix(nonce+i);
    const uint8_t scale=1+random%0x47;
    const int e=(scale>>3)&15, m=scale&7;
    const float factor=e ? ldexpf(1.0f+float(m)/8.0f,e-7) : ldexpf(float(m),-9);
    pk[row*CQ_ROW+CQ_CODES+group]=scale;
    for (int j=0;j<8;++j) {
        random=refresh_mix(random);
        const uint8_t packed=uint8_t(random);
        pk[row*CQ_ROW+group*8+j]=packed;
        for (int half=0;half<2;++half) {
            const int code=(packed>>(4*half))&15, mag=code&7;
            const float v=mag<2 ? float(mag)*0.5f : mag<4 ? 1.0f+float(mag-2)*0.5f :
                          mag<6 ? float(mag-2) : float(2*mag-8);
            bf[row*HD+group*16+j*2+half]=__float2bfloat16_rn((code&8 ? -v : v)*factor);
        }
    }
}
inline void refresh_inputs(Fixture& f, uint64_t nonce, cudaStream_t st) {
    f.memory->invalidate_snapshot();
    const uint64_t key=nonce ^ (f.seed*0x9e3779b97f4a7c15ULL);
    const size_t qn=f.b.out_n;
    const size_t wn=size_t(f.b.p.ring_rows ? f.b.p.ring_rows : f.b.p.n)*HD*f.keys.slots;
    const size_t groups=f.keys.rows*f.keys.slots*32;
    refresh_bf16<<<(qn+255)/256,256,0,st>>>(f.b.q,qn,key,1.0f/65536.0f);
    refresh_bf16<<<(wn+255)/256,256,0,st>>>(f.b.kvw,wn,key^0x243f6a8885a308d3ULL,0.3f/32768.0f);
    refresh_sink<<<1,32,0,st>>>(f.b.sink,f.b.p.heads,key^0x13198a2e03707344ULL);
    refresh_keys<<<(groups+255)/256,256,0,st>>>(f.keys.pk,f.keys.bf,groups,key^0xa4093822299f31d0ULL);
    CK(cudaGetLastError());
}
inline void snapshot_inputs(Fixture& f, cudaStream_t st) { f.memory->snapshot(st); }
inline size_t check_integrity(Fixture& f, cudaStream_t st) { return f.memory->check(st); }
inline const sweep_memory::Integrity& integrity_counts(const Fixture& f) { return f.memory->last; }
inline void destroy(Fixture& f) {
    free_built(f.b); free_keys(f.keys);
    if (f.memory) f.memory->release_workspace();
    f.memory.reset();
}
}
