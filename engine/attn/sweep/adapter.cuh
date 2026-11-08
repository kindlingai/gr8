#pragma once
#include "fixtures.cuh"
#include "capture_io.h"
#define SWEEP_ADAPTER_HAS_METADATA 1
#include <stdexcept>
#include <memory>
namespace sweep_adapter {
using Params = ds41a::AttnP;
struct Shape {
    int n, pos, ratio, floor, wl, keys;
    int ring_rows=8319, slots=1, slot=0;
    bool shadow=false;
    std::string phase, pattern="uniform", capture;
    std::shared_ptr<sweep_capture::Record> record;
};
static std::string active_phase;
inline bool pipe_enabled(const Params&) { return active_phase=="prefill_v2"; }
inline Shape shape(const std::vector<std::string>& a) {
    if (a.size()<6 || (a.size()-6)%2) throw std::runtime_error("attention needs six shape args followed by option/value pairs");
    Shape s{std::stoi(a[0]),std::stoi(a[1]),std::stoi(a[2]),std::stoi(a[3]),std::stoi(a[4]),std::stoi(a[5])};
    s.phase=s.floor?"late_replay":s.n==1?"main":"prefill_v2";
    for(size_t i=6;i<a.size();i+=2) {
        const auto& k=a[i];const auto& v=a[i+1];
        if(k=="--shadow-keys") {if(v!="0"&&v!="1")throw std::runtime_error("shadow-keys must be0 or1");s.shadow=v=="1";}
        else if(k=="--ring-rows")s.ring_rows=std::stoi(v);
        else if(k=="--slots")s.slots=std::stoi(v);
        else if(k=="--slot")s.slot=std::stoi(v);
        else if(k=="--phase")s.phase=v;
        else if(k=="--topk-pattern")s.pattern=v;
        else if(k=="--capture")s.capture=v;
        else throw std::runtime_error("unknown attention option "+k);
    }
    if(!s.capture.empty()) {
        if(a.size()!=8 || a[6]!="--capture")throw std::runtime_error("capture geometry cannot be overridden");
        s.record=std::make_shared<sweep_capture::Record>(sweep_capture::load(s.capture));
        const auto& m=s.record->metadata;
        if(m.n!=s.n||m.chunk_position!=s.pos||m.ratio!=s.ratio||m.floor!=s.floor||m.wl!=s.wl||m.key_rows!=s.keys||m.heads!=16)
            throw std::runtime_error("capture shape does not match six arguments");
        if(m.format!="fp4_e2m1_e4m3_288"&&m.format!="window_only")throw std::runtime_error("CQ adapter needs packed or window-only capture");
        s.ring_rows=m.ring_rows;s.slots=m.slot_count;s.phase=m.dispatch_phase;
        s.pattern="captured_indices";
    }
    if (s.n<1 || s.n>8192 || s.pos<0 || s.pos>1048576-s.n || s.ratio<0 || s.ratio>2 ||
        s.floor<0 || s.floor>s.pos || s.wl<1 || s.wl>128 || s.keys<0 || s.keys>1048576 ||
        (s.floor && !s.ratio) || (s.ratio && !s.keys) ||
        s.ring_rows<s.wl || s.ring_rows>1048704 || s.slots<1 || s.slots>65536 || s.slot<0 || s.slot>=s.slots)
        throw std::runtime_error("invalid attention shape");
    if(s.phase!="prefill_v2"&&s.phase!="prefill_legacy"&&s.phase!="late_replay"&&s.phase!="main")throw std::runtime_error("invalid dispatch phase");
    if(s.capture.empty()&&s.pattern!="uniform"&&s.pattern!="shared"&&s.pattern!="local")throw std::runtime_error("invalid top-k pattern");
    return s;
}
inline size_t required_bytes(const std::vector<std::string>& a) {
    const auto s=shape(a);const size_t keys=std::max(1,s.keys);
    const size_t largest_input=std::max({size_t(s.n)*16*512*2,
        keys*s.slots*(s.shadow?512*2:288),size_t(s.ring_rows)*s.slots*512*2});
    const bool per_row=s.record?s.record->metadata.session_form=="per_row_slots":s.slots>1;
    const size_t allocations=9+size_t(s.shadow)+size_t(per_row); // includes workspace
    return size_t(s.n)*16*512*2*3 + keys*s.slots*((s.shadow?512*2:0)+288) +
           size_t(s.ring_rows)*s.slots*512*2 + size_t(s.n)*(512*4+4+(per_row?4:0)) +
           16*sizeof(float) + allocations*2*sweep_memory::redzone_bytes + sweep_memory::digest_bytes(largest_input);
}
struct Fixture {
    Built b{};
    Keys keys{};
    uint64_t seed = 0;
    Shape geometry;
    sweep_topk::Metrics metrics;
    std::unique_ptr<sweep_memory::Registry> memory;
};
inline Fixture make_case(const std::vector<std::string>& a, uint64_t seed) {
    const auto s=shape(a); rng.seed(seed);
    Fixture f; f.seed=seed;
    f.memory=std::make_unique<sweep_memory::Registry>();
    sweep_memory::AllocationScope allocation_scope(*f.memory);
    f.geometry=s;active_phase=s.phase;
    f.keys=make_keys(std::max(1,s.keys),s.slots,s.shadow);
    auto& b=f.b;auto& p=b.p;
    b.q=(bf16*)dup(randbf(size_t(s.n)*16*HD,0.5f));b.sink=dup(randf(16,1.0f));
    b.out_n=size_t(s.n)*16*HD;b.out_a=dalloc<bf16>(b.out_n);b.out_b=dalloc<bf16>(b.out_n);
    b.kvw=dalloc<bf16>(size_t(s.ring_rows)*s.slots*HD);
    std::vector<int32_t> positions(s.n),sessions(s.n,s.slot),topk(size_t(s.n)*512,-1);
    for(int i=0;i<s.n;++i)positions[i]=s.pos+i;
    if(s.record){positions=s.record->positions;sessions=s.record->sessions;topk=s.record->topk;}
    else if(s.ratio)topk=sweep_topk::make(positions,s.ratio,s.keys,s.pattern,seed);
    f.metrics=sweep_topk::metrics(topk,sessions);
    b.d_pos=dup(positions);b.d_topk=dup(topk);
    const bool per_row=s.record?s.record->metadata.session_form=="per_row_slots":s.slots>1;
    if(per_row)b.d_sess=dup(sessions);
    p.q=b.q;p.kvw=b.kvw;p.topk=b.d_topk;p.sink=b.sink;p.dstart=b.d_pos;p.dsess=b.d_sess;
    p.n=s.n;p.heads=16;p.ldtk=512;p.decode=true;p.WL=s.wl;p.ring_rows=s.ring_rows;p.TK=s.ratio?512:0;
    p.scale=0.04419417382415922f;p.floor=s.floor;
    set_rows(p,f.keys,true,b.out_b);
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
static __global__ void refresh_window(bf16* out, size_t n, uint64_t nonce) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) {
        const uint8_t code=uint8_t(refresh_mix(nonce+i)%120); // finite magnitudes, synthetic distribution
        const int e=(code>>3)&15,m=code&7;
        float v=e?ldexpf(1.0f+float(m)/8,e-7):ldexpf(float(m),-9);
        if(refresh_mix(nonce+i)&128)v=-v;
        out[i]=__float2bfloat16_rn(v);
    }
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
            if(bf)bf[row*HD+group*16+j*2+half]=__float2bfloat16_rn((code&8 ? -v : v)*factor);
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
    refresh_window<<<(wn+255)/256,256,0,st>>>(f.b.kvw,wn,key^0x243f6a8885a308d3ULL);
    refresh_sink<<<1,32,0,st>>>(f.b.sink,f.b.p.heads,key^0x13198a2e03707344ULL);
    refresh_keys<<<(groups+255)/256,256,0,st>>>(f.keys.pk,f.keys.bf,groups,key^0xa4093822299f31d0ULL);
    CK(cudaGetLastError());
}
inline void snapshot_inputs(Fixture& f, cudaStream_t st) { f.memory->snapshot(st); }
inline size_t check_integrity(Fixture& f, cudaStream_t st) { return f.memory->check(st); }
inline const sweep_memory::Integrity& integrity_counts(const Fixture& f) { return f.memory->last; }
inline std::string metadata_json(const Fixture& f) {
    const auto& s=f.geometry;
    std::string source=s.record?(s.record->is_captured?"captured_indices_synthetic_operands":"synthetic_capture_test"):"synthetic";
    std::ostringstream o;
    o<<"{\"input_provenance\":"<<sweep_capture::quote(source)<<",\"dispatch_phase\":"<<sweep_capture::quote(s.phase)
     <<",\"ring_rows\":"<<s.ring_rows<<",\"slot_count\":"<<s.slots<<",\"key_rows\":"<<s.keys
     <<",\"topk_pattern\":"<<sweep_capture::quote(s.pattern)<<",\"within_row_unique_fraction\":"<<f.metrics.within_row_unique_fraction
     <<",\"adjacent_overlap_fraction\":"<<f.metrics.adjacent_overlap_fraction
     <<",\"capture_sha256\":"<<sweep_capture::quote(s.record?s.record->source_hash:"")
     <<",\"operands\":\"synthetic_fp8_valued_window_bf16_q_fp4_keys\",\"bf16_shadow_allocated\":"<<(s.shadow?"true":"false")<<",\"decision_rows\":false}";
    return o.str();
}
inline void destroy(Fixture& f) {
    free_built(f.b); free_keys(f.keys);
    if (f.memory) f.memory->release_workspace();
    f.memory.reset();
}
}
