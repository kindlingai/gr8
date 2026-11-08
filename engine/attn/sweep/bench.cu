// Generic CUDA paired timing and fresh-work checker. No kernel-specific shapes.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include SWEEP_ADAPTER_HEADER
#define sweep_variant sweep_reference
#include SWEEP_REFERENCE_HEADER
#undef sweep_variant
#if defined(SWEEP_VARIANT_HAS_SETUP) && SWEEP_VARIANT_HAS_SETUP
#define SWEEP_REFERENCE_HAS_SETUP 1
#else
#define SWEEP_REFERENCE_HAS_SETUP 0
#endif
#undef SWEEP_VARIANT_HAS_SETUP
#if SWEEP_SAME_VARIANT
namespace sweep_variant = sweep_reference;
#define SWEEP_VARIANT_HAS_SETUP SWEEP_REFERENCE_HAS_SETUP
#else
#include SWEEP_VARIANT_HEADER
#endif
#ifndef SWEEP_VARIANT_HAS_SETUP
#define SWEEP_VARIANT_HAS_SETUP 0
#endif
static void check(cudaError_t e) {
    if(e!=cudaSuccess) {
        fprintf(stderr,"%s: %s\n",e==cudaErrorMemoryAllocation?"LOW_MEMORY":"CUDA_ERROR",cudaGetErrorString(e));
        exit(e==cudaErrorMemoryAllocation?77:2);
    }
}
// Harness scratch has its own red zones too; adapter guards its allocations.
struct GuardedScratch {
    unsigned char* base=nullptr; size_t bytes=0;
    void alloc(size_t n) { bytes=n; check(cudaMalloc(&base,n+512));check(cudaMemset(base,0xD3,n+512)); }
    unsigned char* data() const { return base+256; }
    size_t errors() const {
        unsigned char a[256],b[256];size_t n=0;
        check(cudaMemcpy(a,base,256,cudaMemcpyDeviceToHost));check(cudaMemcpy(b,data()+bytes,256,cudaMemcpyDeviceToHost));
        for(int i=0;i<256;++i)n+=(a[i]!=0xD3)+(b[i]!=0xD3);return n;
    }
    void release() { if(base)check(cudaFree(base));base=nullptr; }
};
__global__ void sweep_count_bytes(const unsigned char* a,const unsigned char* b,size_t n,unsigned long long* bad) {
    unsigned long long count=0;__shared__ unsigned long long sums[256];
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)count+=a[i]!=b[i];
    sums[threadIdx.x]=count;__syncthreads();
    for(int step=128;step;step>>=1){if(threadIdx.x<step)sums[threadIdx.x]+=sums[threadIdx.x+step];__syncthreads();}
    if(!threadIdx.x && sums[0])atomicAdd(bad,sums[0]);
}
static double avg(const std::vector<double>& x) { double s=0;for(double v:x)s+=v;return s/x.size(); }
static double sd(const std::vector<double>& x) { if(x.size()<2)return 0;double m=avg(x),s=0;for(double v:x)s+=(v-m)*(v-m);return sqrt(s/(x.size()-1)); }
static void array_json(const char* name,const std::vector<double>& a) {
    printf(",\"%s\":[",name);for(size_t i=0;i<a.size();++i)printf("%s%.9f",i?",":"",a[i]);printf("]");
}
int main(int argc,char** argv) {
  try {
    int warmup=3,reps=5,iters=20,pools=3,amortize=1,flush_mb=0; uint64_t seed=12345;std::vector<std::string> args;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--"){while(++i<argc)args.emplace_back(argv[i]);break;}
        if(i+1>=argc)throw std::runtime_error("missing option value");
        if(a=="--warmup")warmup=std::stoi(argv[++i]);
        else if(a=="--reps")reps=std::stoi(argv[++i]);
        else if(a=="--iters")iters=std::stoi(argv[++i]);
        else if(a=="--seed")seed=std::stoull(argv[++i]);
        else if(a=="--pools")pools=std::stoi(argv[++i]);
        else if(a=="--amortize")amortize=std::stoi(argv[++i]);
        else if(a=="--flush-mb")flush_mb=std::stoi(argv[++i]);
        else throw std::runtime_error("unknown option");
    }
    if(warmup<0||reps<2||iters<1||pools<1||pools>64||amortize<1||amortize>1024||flush_mb<0||flush_mb>16384)
        throw std::runtime_error("need warmup>=0,reps>=2,iters>=1,pools=1..64,amortize=1..1024,flush-mb=0..16384");
    const int slots=std::max(3,pools);
    const size_t flush_bytes=size_t(flush_mb)<<20;
    size_t free=0,total=0,required=slots*sweep_adapter::required_bytes(args)+flush_bytes+1024,reserve=256ul<<20;
    check(cudaMemGetInfo(&free,&total));
    fprintf(stderr,"MEMORY free_bytes=%zu total_bytes=%zu required_bytes=%zu reserve_bytes=%zu pools=%d\n",free,total,required,reserve,pools);
    if(free<required+reserve){fprintf(stderr,"LOW_MEMORY: rotating fixtures need more free memory\n");return 77;}
    std::vector<sweep_adapter::Fixture> fixtures;fixtures.reserve(slots);
    for(int i=0;i<slots;++i)fixtures.push_back(sweep_adapter::make_case(args,seed+104729ull*i));
    auto p0=sweep_adapter::params(fixtures[0],false),r0=sweep_adapter::params(fixtures[0],true);
    // Exactly once per initialized shape: a stale-input init cache must not be reset by the checker.
    check(sweep_reference::init(r0));check(sweep_variant::init(p0));
    cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,sweep_variant::kernel(p0)));
    cudaStream_t st;check(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    GuardedScratch bad,flush;bad.alloc(sizeof(unsigned long long));if(flush_bytes)flush.alloc(flush_bytes);
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    size_t mismatch=0,timed_mismatch=0,integrity_errors=0,checks=0,timed_checks=0,stream_ready=0,stream_pending=0;
    const size_t bytes=sweep_adapter::output_bytes(fixtures[0]);
    auto query=[&] {
        const cudaError_t q=cudaStreamQuery(st);
        // A short kernel can legitimately finish before the host queries it. Enqueue proof is graph validation below.
        if(q==cudaSuccess)++stream_ready;else if(q==cudaErrorNotReady)++stream_pending;else check(q);
    };
    auto setup=[&](bool reference,const sweep_adapter::Params& p) {
        if(reference) {
#if SWEEP_REFERENCE_HAS_SETUP
            sweep_reference::setup_chunk(p,st);
#endif
        } else {
#if SWEEP_VARIANT_HAS_SETUP
            sweep_variant::setup_chunk(p,st);
#endif
        }
    };
    auto launch=[&](bool reference,const sweep_adapter::Params& p) {
        if(reference)sweep_reference::launch(p,st);else sweep_variant::launch(p,st);
        check(cudaGetLastError());query();
    };
    auto prepare=[&](int input,uint64_t nonce) {
        sweep_adapter::refresh_inputs(fixtures[input],nonce,st);
        sweep_adapter::snapshot_inputs(fixtures[input],st);
    };
    auto verify=[&](int input,int output,bool timed) {
        auto& f=fixtures[input];auto& dst=fixtures[output];
        auto ref=sweep_adapter::params_with_output(f,dst,true);
        check(cudaMemsetAsync(sweep_adapter::output(dst,true),0xA5,bytes,st));
        setup(true,ref);launch(true,ref); // Fresh reference on the current immutable inputs, never a saved first result.
        check(cudaMemsetAsync(bad.data(),0,sizeof(unsigned long long),st));
        sweep_count_bytes<<<256,256,0,st>>>((const unsigned char*)sweep_adapter::output(dst,true),
            (const unsigned char*)sweep_adapter::output(dst,false),bytes,(unsigned long long*)bad.data());
        check(cudaGetLastError());check(cudaStreamSynchronize(st));
        unsigned long long n=0;check(cudaMemcpy(&n,bad.data(),sizeof(n),cudaMemcpyDeviceToHost));
        mismatch=std::max(mismatch,size_t(n));if(timed)timed_mismatch=std::max(timed_mismatch,size_t(n));
        ++checks;if(timed)++timed_checks;
        integrity_errors+=sweep_adapter::check_integrity(f,st);
        if(output!=input)integrity_errors+=sweep_adapter::check_integrity(dst,st);
    };
    // Every slot has expected immutable state even before it is used as an output-only slot.
    for(int i=0;i<slots;++i)sweep_adapter::snapshot_inputs(fixtures[i],st);
    for(int call=0;call<2*slots;++call) {
        int in=call%pools,out=(call+1)%slots;prepare(in,seed+call+1);
        auto p=sweep_adapter::params_with_output(fixtures[in],fixtures[out],false);
        check(cudaMemsetAsync(sweep_adapter::output(fixtures[out],false),0x5A,bytes,st));
        setup(false,p);launch(false,p);verify(in,out,false);
    }
    // Validate actual enqueued kernel identity on the supplied nonblocking stream.
    // No-op/off-stream wrappers cannot satisfy this even if StreamQuery sees a completed short kernel.
    cudaGraph_t graph{};check(cudaStreamSynchronize(st));check(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    setup(false,p0);sweep_variant::launch(p0,st);
    cudaError_t capture_error=cudaStreamEndCapture(st,&graph);
    size_t kernel_nodes=0,main_nodes=0;bool launch_valid=capture_error==cudaSuccess;
    if(launch_valid) {
        size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeKernel){
            cudaKernelNodeParams kp{};check(cudaGraphKernelNodeGetParams(node,&kp));++kernel_nodes;
            main_nodes+=(const void*)kp.func==sweep_variant::kernel(p0);
        }}
        launch_valid=main_nodes>0;check(cudaGraphDestroy(graph));
    }
    check(cudaDeviceSynchronize()); // catches any invalid off-stream enqueue before timing
    uint64_t sequence=2*slots+100;
    auto flush_cache=[&] {
        if(flush_bytes){check(cudaMemsetAsync(flush.data(),int(sequence&255),flush_bytes,st));check(cudaStreamSynchronize(st));}
    };
    auto elapsed=[&](){check(cudaEventRecord(end,st));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));return 1000.0*ms;};
    // Setup charged once per M-call chunk, kernel calls charged individually; all validation/poison/flush excluded and labelled.
    auto timed_group=[&](bool reference,int in,int out0,bool sample)->double {
        double us=0;auto p=sweep_adapter::params_with_output(fixtures[in],fixtures[out0],reference);
        flush_cache();
        if(reference ? bool(SWEEP_REFERENCE_HAS_SETUP) : bool(SWEEP_VARIANT_HAS_SETUP)) {
            check(cudaEventRecord(begin,st));setup(reference,p);us+=elapsed();
        }
        for(int j=0;j<amortize;++j){int out=(out0+j)%slots;p=sweep_adapter::params_with_output(fixtures[in],fixtures[out],reference);
            check(cudaMemsetAsync(sweep_adapter::output(fixtures[out],reference),reference?0xA5:0x5A,bytes,st));
            flush_cache();check(cudaEventRecord(begin,st));launch(reference,p);us+=elapsed();
            if(!reference && sample)verify(in,out,true);
        }
        return us/amortize;
    };
    for(int i=0;i<warmup;++i){int in=i%pools;prepare(in,seed+(++sequence));auto p=sweep_adapter::params(fixtures[in],false);
        check(cudaMemsetAsync(sweep_adapter::output(fixtures[in],false),0x5A,bytes,st));setup(false,p);launch(false,p);}
    check(cudaStreamSynchronize(st));
    std::vector<double> samples,reference_samples,paired,nulls;
    for(int r=0;r<reps;++r){double cand=0,ref=0;
        for(int i=0;i<iters;++i){int in=int(sequence%pools),out=(in+1)%slots;prepare(in,seed+(++sequence));
            const bool sample=i==0||i==iters-1||i%7==0;
            if((r+i)&1){cand+=timed_group(false,in,out,sample);ref+=timed_group(true,in,out,false);}
            else{ref+=timed_group(true,in,out,false);cand+=timed_group(false,in,out,sample);}
        }
        samples.push_back(cand/iters);reference_samples.push_back(ref/iters);paired.push_back(100*(cand/ref-1));
        int in=int(sequence%pools);prepare(in,seed+(++sequence));
        double a=timed_group(true,in,(in+1)%slots,false),b=timed_group(true,in,(in+2)%slots,false);
        nulls.push_back(100*(b/a-1));
    }
    check(cudaStreamSynchronize(st));for(auto& f:fixtures)integrity_errors+=sweep_adapter::check_integrity(f,st);
    integrity_errors+=bad.errors();if(flush_bytes)integrity_errors+=flush.errors();
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    const char* status=integrity_errors?"MEMORY_CORRUPTION":mismatch?"MISMATCH":!launch_valid?"INVALID_LAUNCH":"EXACT";
    printf("{\"status\":\"%s\",\"mean_us\":%.9f,\"sd_us\":%.9f,\"reference_mean_us\":%.9f,\"paired_delta_pct_mean\":%.9f,\"null_delta_pct_mean\":%.9f,\"null_band_pct\":3,\"mismatch_bytes\":%zu,\"timed_mismatch_bytes\":%zu,\"integrity_errors\":%zu,\"output_bytes\":%zu,\"regs\":%d,\"local_bytes\":%zu,\"static_smem\":%zu,\"dynamic_smem\":%zu,\"ptxas_key\":\"%s\",\"gpu\":\"%s\",\"cc\":\"%d.%d\",\"input_slots\":%d,\"output_slots\":%d,\"checks\":%zu,\"timed_checks\":%zu,\"stream_ready_queries\":%zu,\"stream_pending_queries\":%zu,\"launch_valid\":%s,\"kernel_nodes\":%zu,\"reported_main_nodes\":%zu,\"flush_bytes\":%zu,\"l2_bytes\":%d,\"amortize\":%d,\"has_chunk_setup\":%s,\"timing_regime\":\"rotating-fresh-input-poisoned-output%s\"",
        status,avg(samples),sd(samples),avg(reference_samples),avg(paired),avg(nulls),mismatch,timed_mismatch,integrity_errors,bytes,
        attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,sweep_variant::dynamic_smem(p0),sweep_variant::ptxas_key(p0),prop.name,prop.major,prop.minor,
        pools,slots,checks,timed_checks,stream_ready,stream_pending,launch_valid?"true":"false",kernel_nodes,main_nodes,flush_bytes,prop.l2CacheSize,amortize,
        SWEEP_VARIANT_HAS_SETUP?"true":"false",flush_bytes?"-explicit-flush":"");
    array_json("samples_us",samples);array_json("reference_samples_us",reference_samples);array_json("paired_delta_pct_samples",paired);array_json("null_delta_pct_samples",nulls);printf("}\n");
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaStreamDestroy(st));bad.release();flush.release();
    for(auto& f:fixtures)sweep_adapter::destroy(f);
    return std::string(status)=="EXACT"?0:1;
  }catch(const std::exception& e){fprintf(stderr,"INVALID_CASE: %s\n",e.what());return 2;}
}
