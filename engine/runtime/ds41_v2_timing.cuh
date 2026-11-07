// Optional v2 GPU timeline. Include globally. All events are allocated at boot;
// request-time calls only record/query them. No host fence or extra stream wait.
#pragma once
#include "ds41_v2.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace ds41::v2 {
enum class TimingStage {
    Chunk, AttnInAg, IndexAg, CandidatesAg, TailAg, EngramA2A, AttnA2A,
    WoBOwn, WoBRemote, FfnAg, FfnUp, FfnDown, Rs, Join,
    Layer, Attention, Engram, Dispatch, ExpertFc1, ExpertFc2, Combine, Assemble,
    PackPublish, Wait, Unpack, Reduce
};
inline const char* timing_stage_name(TimingStage s) {
    switch(s) {
    case TimingStage::Chunk:return "chunk";
    case TimingStage::AttnInAg:return "attn_in_ag";
    case TimingStage::IndexAg:return "index_ag";
    case TimingStage::CandidatesAg:return "candidates_ag";
    case TimingStage::TailAg:return "tail_ag";
    case TimingStage::EngramA2A:return "engram_a2a";
    case TimingStage::AttnA2A:return "attn_a2a";
    case TimingStage::WoBOwn:return "wo_b_own";
    case TimingStage::WoBRemote:return "wo_b_remote";
    case TimingStage::FfnAg:return "ffn_ag";
    case TimingStage::FfnUp:return "ffn_up";
    case TimingStage::FfnDown:return "ffn_down";
    case TimingStage::Rs:return "rs";
    case TimingStage::Join:return "join";
    case TimingStage::Layer:return "layer";
    case TimingStage::Attention:return "attention";
    case TimingStage::Engram:return "engram";
    case TimingStage::Dispatch:return "dispatch";
    case TimingStage::ExpertFc1:return "expert_fc1";
    case TimingStage::ExpertFc2:return "expert_fc2";
    case TimingStage::Combine:return "combine";
    case TimingStage::Assemble:return "assemble";
    case TimingStage::PackPublish:return "pack_publish";
    case TimingStage::Wait:return "wait";
    case TimingStage::Unpack:return "unpack";
    case TimingStage::Reduce:return "reduce";
    }
    return "unknown";
}
inline void timing_check(cudaError_t e) {
    if(e!=cudaSuccess) { std::fprintf(stderr,"DS41 v2 timing: %s\n",cudaGetErrorString(e)); std::abort(); }
}
struct Timing {
    // Bounded storage: overflow drops new samples, never waits or overwrites an
    // unreported event. A request may span chunks and interleave with others.
    static constexpr size_t CAPACITY=8192;
    struct Entry {
        cudaEvent_t begin=nullptr,end=nullptr;
        cudaStream_t stream=nullptr;
        uint64_t request=0,serial=0;
        uint32_t chunk=0,rows=0;
        int layer=-1,group=-1,anchor=-1;
        TimingStage stage=TimingStage::Chunk;
        bool used=false,ended=false,reported=false,comm=false;
    };
    struct Request { size_t chunks=0,dropped=0; uint32_t first_chunk=0,last_chunk=0; };
    std::vector<Entry> ring;
    std::map<uint64_t,Request> requests;
    size_t cursor=0;
    uint64_t serial=0,request=0;
    uint32_t chunk=0,rows=0;
    int layer=-1,anchor=-1,rank=0;
    bool active=false;
    explicit Timing(int rank_,size_t capacity=CAPACITY):ring(capacity),rank(rank_) {
        for(auto& e:ring) {
            timing_check(cudaEventCreateWithFlags(&e.begin,cudaEventDefault));
            timing_check(cudaEventCreateWithFlags(&e.end,cudaEventDefault));
        }
    }
    Timing(const Timing&)=delete;
    Timing& operator=(const Timing&)=delete;
    ~Timing() {
        for(auto& e:ring) { timing_check(cudaEventDestroy(e.begin)); timing_check(cudaEventDestroy(e.end)); }
    }
    static bool ready(cudaEvent_t ev) {
        const auto e=cudaEventQuery(ev);
        if(e==cudaErrorNotReady)return false;
        timing_check(e); return true;
    }
    int begin(TimingStage stage,cudaStream_t stream,bool comm,int group=-1) {
        if(!active)return -1;
        if((stage!=TimingStage::Chunk && anchor<0) || ring.empty()) { ++requests[request].dropped; return -1; }
        const size_t i=cursor;
        Entry& e=ring[i];
        // Retired, not-ready entries can only be reused after the end has
        // completed. Check once, never poll. The write cursor stays on a full
        // slot until a report frees it: no scan in the enqueue hot path.
        if(e.used && (!e.reported || !e.ended || !ready(e.end))) { ++requests[request].dropped; return -1; }
        e.stream=stream; e.request=request; e.serial=++serial;
        e.chunk=chunk; e.rows=rows; e.layer=layer; e.group=group;
        e.anchor=stage==TimingStage::Chunk?int(i):anchor;
        e.stage=stage; e.used=true; e.ended=false; e.reported=false; e.comm=comm;
        timing_check(cudaEventRecord(e.begin,stream));
        cursor=(cursor+1)%ring.size();
        return int(i);
    }
    void end(int i) {
        if(i<0)return;
        auto& e=ring[size_t(i)];
        timing_check(cudaEventRecord(e.end,e.stream)); e.ended=true;
    }
    void begin_chunk(uint64_t req,uint32_t first,uint32_t n,cudaStream_t st) {
        if(active) { std::fprintf(stderr,"DS41 v2 timing: nested chunk\n"); std::abort(); }
        request=req; chunk=first; rows=n; layer=-1; active=true; anchor=-1;
        auto& r=requests[request];
        if(!r.chunks)r.first_chunk=first;
        r.last_chunk=first; ++r.chunks;
        anchor=begin(TimingStage::Chunk,st,false);
    }
    void end_chunk() { if(active)end(anchor); active=false; anchor=-1; }
    // Call after each chunk at the caller's EXISTING completion fence. Chunk
    // labels are first-token offsets, not ordinal counters. Query only: pending entries are
    // labelled, not waited for or presented as measurements. Both streams use
    // the SAME chunk-begin event as origin; no profiling dependency is added.
    // A join interval measures the compute stream's wait, not CPU enqueue time.
    // Retiring each chunk lets arbitrarily long requests reuse the fixed ring;
    // a later begin_chunk with the same request id starts fresh stats. Multiple
    // chunks can still be aggregated if a caller elects to report less often.
    void report(uint64_t req,FILE* out=stderr) {
        const auto it=requests.find(req);
        if(it==requests.end())return; // idempotent, including default-off callers
        if(active && request==req) { std::fprintf(stderr,"DS41 v2 timing: report inside chunk\n"); std::abort(); }
        std::vector<size_t> ids;
        for(size_t i=0;i<ring.size();++i)if(ring[i].used && !ring[i].reported && ring[i].request==req)ids.push_back(i);
        std::sort(ids.begin(),ids.end(),[&](size_t a,size_t b){return ring[a].serial<ring[b].serial;});
        size_t measured=0,pending=0;
        std::string body;
        char line[512];
        for(size_t i:ids) {
            auto& e=ring[i]; const auto& origin=ring[size_t(e.anchor)];
            const bool done=e.ended && ready(e.end) && ready(e.begin) && ready(origin.begin);
            int n;
            if(done) {
                float t0=0,t1=0;
                timing_check(cudaEventElapsedTime(&t0,origin.begin,e.begin));
                timing_check(cudaEventElapsedTime(&t1,origin.begin,e.end));
                n=std::snprintf(line,sizeof(line),"# ds41_v2_timing rank=%d request=%llu chunk=%u rows=%u layer=%d stage=%s group=%d stream=%s t0_ms=%.6f t1_ms=%.6f ms=%.6f\n",
                    rank,(unsigned long long)req,e.chunk,e.rows,e.layer,timing_stage_name(e.stage),e.group,e.comm?"cs":"st",t0,t1,t1-t0);
                ++measured;
            } else {
                n=std::snprintf(line,sizeof(line),"# ds41_v2_timing rank=%d request=%llu chunk=%u rows=%u layer=%d stage=%s group=%d stream=%s pending=1\n",
                    rank,(unsigned long long)req,e.chunk,e.rows,e.layer,timing_stage_name(e.stage),e.group,e.comm?"cs":"st");
                ++pending;
            }
            body.append(line,size_t(n));
            e.reported=true;
            // Do not free anchors until EVERY interval has been read. Marking
            // reported preserves event objects and prevents duplicate output.
        }
        const int n=std::snprintf(line,sizeof(line),"# ds41_v2_timing_summary rank=%d request=%llu chunks=%zu measured=%zu pending=%zu dropped=%zu first_chunk=%u last_chunk=%u origin=chunk_begin\n",
            rank,(unsigned long long)req,it->second.chunks,measured,pending,it->second.dropped,it->second.first_chunk,it->second.last_chunk);
        body.append(line,size_t(n));
        std::fwrite(body.data(),1,body.size(),out);
        requests.erase(it);
    }
};

inline void timing_init(State& s,bool enabled,size_t capacity=Timing::CAPACITY) {
    if(enabled && !s.timing)s.timing=new Timing(s.wire.rank,capacity);
}
inline void timing_release(State& s) { delete s.timing; s.timing=nullptr; }
inline void timing_begin_chunk(State& s,uint64_t request,uint32_t first,uint32_t rows) {
    if(s.timing)s.timing->begin_chunk(request,first,rows,s.st);
}
inline void timing_layer(State& s,int layer) { if(s.timing)s.timing->layer=layer; }
inline void timing_end_chunk(State& s) { if(s.timing)s.timing->end_chunk(); }
inline void timing_report(State& s,uint64_t request,FILE* out=stderr) { if(s.timing)s.timing->report(request,out); }
struct TimingSpan {
    Timing* timing=nullptr;
    int slot=-1;
    TimingSpan(Timing* t,TimingStage stage,cudaStream_t stream,bool comm,int group=-1):timing(t) {
        if(timing)slot=timing->begin(stage,stream,comm,group);
    }
    TimingSpan(State& s,TimingStage stage,cudaStream_t stream,int group=-1)
        :TimingSpan(s.timing,stage,stream,stream==s.cs,group) {}
    void finish() { if(timing && slot>=0) { timing->end(slot); slot=-1; } }
    ~TimingSpan() { finish(); }
    TimingSpan(const TimingSpan&)=delete;
    TimingSpan& operator=(const TimingSpan&)=delete;
};
// Extra stage boundaries are enabled only by the v2 transport trace, not by
// the pre-existing timing knob alone. No new event launches on the off path.
struct TimingDetailSpan : TimingSpan {
    TimingDetailSpan(State* s,TimingStage stage,cudaStream_t stream,int group=-1)
        :TimingSpan(s && s->wire.xt ? s->timing : nullptr,stage,stream,s && stream==s->cs,group) {}
    TimingDetailSpan(State& s,TimingStage stage,cudaStream_t stream,int group=-1)
        :TimingDetailSpan(&s,stage,stream,group) {}
};
} // namespace ds41::v2
