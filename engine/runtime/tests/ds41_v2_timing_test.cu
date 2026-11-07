// Standalone event-ring tests; the test harness fences, the instrument never does.
#include "../ds41_v2_timing.cuh"
#include "../../knobs-rs/gr8_knobs.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>
using namespace ds41::v2;
#define CK(call) timing_check(call)
static std::string read(FILE* f) {
    std::fflush(f); std::rewind(f); std::string s; char b[2048]; size_t n;
    while((n=std::fread(b,1,sizeof(b),f)))s.append(b,n);
    return s;
}
static std::string report(State& s,uint64_t id) {
    FILE* f=std::tmpfile(); assert(f); timing_report(s,id,f); auto out=read(f); std::fclose(f); return out;
}
static size_t occurrences(const std::string& s,const char* needle) {
    size_t n=0,pos=0;
    while((pos=s.find(needle,pos))!=std::string::npos) { ++n; pos+=std::strlen(needle); }
    return n;
}
static void CUDART_CB block(void* p) {
    auto& gate=*static_cast<std::atomic<bool>*>(p);
    while(!gate.load(std::memory_order_acquire))std::this_thread::yield();
}
__global__ void work(unsigned long long clocks) {
    const auto start=clock64();
    while(clock64()-start<clocks)asm volatile("");
}
int main(int argc,char**) {
    if(argc>1) {
        assert(knob::ds_pf_v2_timing());
        std::puts("PASS knob GR8_DS_PF_V2_TIMING=1"); return 0;
    }
    assert(!knob::ds_pf_v2_timing());
    State s; s.wire.rank=2;
    CK(cudaStreamCreateWithFlags(&s.st,cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s.cs,cudaStreamNonBlocking));
    timing_init(s,false,0); assert(!s.timing);
    timing_begin_chunk(s,1,0,256); timing_layer(s,3);
    { TimingSpan span(s,TimingStage::AttnInAg,s.st); }
    timing_end_chunk(s); assert(report(s,1).empty());
    std::puts("PASS default off: no ring/events/output");

    timing_init(s,true,16);
    timing_begin_chunk(s,9,4096,4096); timing_layer(s,3);
    const auto serial=s.timing->serial;
    { TimingDetailSpan span(s,TimingStage::Wait,s.st,0); }
    assert(s.timing->serial==serial); // old TIMING alone has no new boundaries
    s.wire.xt=reinterpret_cast<unsigned long long*>(uintptr_t(1)); // gate only; never dereferenced
    { TimingDetailSpan span(s,TimingStage::Wait,s.st,0); work<<<1,32,0,s.st>>>(1000); span.finish(); span.finish(); }
    { TimingDetailSpan span(s,TimingStage::Unpack,s.st,0); work<<<1,32,0,s.st>>>(1000); }
    s.wire.xt=nullptr;
    timing_end_chunk(s); CK(cudaStreamSynchronize(s.st));
    auto detailed=report(s,9);
    assert(detailed.find("chunks=1 measured=3 pending=0 dropped=0")!=std::string::npos);
    assert(detailed.find("stage=wait group=0 stream=st")!=std::string::npos);
    assert(detailed.find("stage=unpack group=0 stream=st")!=std::string::npos);
    std::puts("PASS detailed boundaries need XTRACE; wait/unpack separate; finish idempotent");
    timing_release(s);

    timing_init(s,true,8);
    std::atomic<bool> gate{false};
    CK(cudaLaunchHostFunc(s.st,block,&gate));
    timing_begin_chunk(s,2,128,256); timing_layer(s,7);
    { TimingSpan span(s,TimingStage::WoBOwn,s.st,2); }
    timing_end_chunk(s);
    const auto start=std::chrono::steady_clock::now();
    const auto pending=report(s,2);
    const auto elapsed=std::chrono::steady_clock::now()-start;
    assert(elapsed<std::chrono::seconds(2));
    assert(pending.find("measured=0 pending=2 dropped=0")!=std::string::npos);
    assert(pending.find("t0_ms=")==std::string::npos);
    assert(report(s,2).empty());
    gate.store(true,std::memory_order_release);
    CK(cudaStreamSynchronize(s.st));
    std::puts("PASS not-ready query returns while stream blocked; pending labelled; report idempotent");
    timing_release(s);

    timing_init(s,true,3);
    timing_begin_chunk(s,3,0,256); timing_layer(s,0);
    { TimingSpan span(s,TimingStage::AttnInAg,s.st); }
    { TimingSpan span(s,TimingStage::WoBOwn,s.st); }
    { TimingSpan span(s,TimingStage::WoBRemote,s.st,1); }
    timing_end_chunk(s); CK(cudaStreamSynchronize(s.st));
    auto full=report(s,3);
    assert(full.find("chunks=1 measured=3 pending=0 dropped=1")!=std::string::npos);
    timing_begin_chunk(s,4,256,7); timing_layer(s,20);
    { TimingSpan span(s,TimingStage::FfnAg,s.st); }
    timing_end_chunk(s); CK(cudaStreamSynchronize(s.st));
    auto wrap=report(s,4);
    assert(wrap.find("chunks=1 measured=2 pending=0 dropped=0")!=std::string::npos);
    assert(wrap.find("chunk=256 rows=7 layer=20 stage=ffn_ag")!=std::string::npos);
    std::puts("PASS full ring drops without overwrite; completed slots wrap and retain metadata");
    timing_release(s);

    timing_init(s,true,16);
    for(int c=0;c<3;++c) {
        timing_begin_chunk(s,c==1?6:5,c==2?256:0,256); timing_layer(s,c);
        { TimingSpan span(s,TimingStage::FfnDown,s.st,c); }
        timing_end_chunk(s);
    }
    CK(cudaStreamSynchronize(s.st));
    const auto multi=report(s,5),other=report(s,6);
    assert(multi.find("chunks=2 measured=4 pending=0 dropped=0")!=std::string::npos);
    assert(occurrences(multi,"stage=ffn_down")==2 && multi.find("request=6 ")==std::string::npos);
    assert(other.find("chunks=1 measured=2 pending=0 dropped=0")!=std::string::npos);
    assert(report(s,5).empty() && report(s,6).empty());
    std::puts("PASS optional interleaved multi-chunk aggregation reports each sample once");
    timing_release(s);

    timing_init(s,true,8);
    for(uint32_t c=0;c<32;++c) {
        timing_begin_chunk(s,8,c*4096,4096); timing_layer(s,20);
        { TimingSpan span(s,TimingStage::AttnInAg,s.st); }
        { TimingSpan span(s,TimingStage::FfnAg,s.st); }
        timing_end_chunk(s); CK(cudaStreamSynchronize(s.st)); // harness fence only
        const auto chunk_report=report(s,8);
        assert(chunk_report.find("request=8 ")!=std::string::npos);
        assert(chunk_report.find("chunks=1 measured=3 pending=0 dropped=0")!=std::string::npos);
        assert(chunk_report.find("first_chunk="+std::to_string(c*4096)+" last_chunk="+std::to_string(c*4096))!=std::string::npos);
        assert(report(s,8).empty());
    }
    std::puts("PASS 32 per-chunk reports reuse one request id and bounded ring without loss (synthetic metadata)");
    timing_release(s);

    timing_init(s,true,8);
    cudaEvent_t forked,received;
    CK(cudaEventCreateWithFlags(&forked,cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&received,cudaEventDisableTiming));
    timing_begin_chunk(s,7,0,8192); timing_layer(s,14);
    CK(cudaEventRecord(forked,s.st)); CK(cudaStreamWaitEvent(s.cs,forked,0));
    { TimingSpan span(s,TimingStage::Rs,s.cs,0); work<<<1,32,0,s.cs>>>(10000000); }
    CK(cudaEventRecord(received,s.cs));
    { TimingSpan span(s,TimingStage::FfnDown,s.st,1); work<<<1,32,0,s.st>>>(10000000); }
    { TimingSpan span(s,TimingStage::Join,s.st,0); CK(cudaStreamWaitEvent(s.st,received,0)); }
    timing_end_chunk(s); CK(cudaStreamSynchronize(s.st)); CK(cudaStreamSynchronize(s.cs));
    const auto timeline=report(s,7);
    assert(timeline.find("chunks=1 measured=4 pending=0 dropped=0")!=std::string::npos);
    assert(timeline.find("stage=rs group=0 stream=cs t0_ms=")!=std::string::npos);
    assert(timeline.find("stage=ffn_down group=1 stream=st t0_ms=")!=std::string::npos);
    // Validate common-origin intervals directly, not an assumption that adding
    // independent per-stream durations can establish overlap.
    auto& ring=s.timing->ring;
    float rs0,rs1,fc0,fc1;
    CK(cudaEventElapsedTime(&rs0,ring[0].begin,ring[1].begin));
    CK(cudaEventElapsedTime(&rs1,ring[0].begin,ring[1].end));
    CK(cudaEventElapsedTime(&fc0,ring[0].begin,ring[2].begin));
    CK(cudaEventElapsedTime(&fc1,ring[0].begin,ring[2].end));
    assert(rs0>=0 && fc0>=0 && rs1>rs0 && fc1>fc0);
    std::fputs(timeline.c_str(),stdout);
    std::printf("PASS common-origin st/cs timeline; observed overlap_ms=%.6f (synthetic, not model performance)\n",std::max(0.f,std::min(rs1,fc1)-std::max(rs0,fc0)));
    timing_release(s); assert(!s.timing);
    CK(cudaEventDestroy(forked)); CK(cudaEventDestroy(received));
    CK(cudaStreamDestroy(s.st)); CK(cudaStreamDestroy(s.cs));
    std::puts("PASS all v2 timing tests");
}
