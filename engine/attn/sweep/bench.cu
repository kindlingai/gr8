// Generic CUDA timing/byte-exactness harness. Shapes and inputs belong to adapters.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include SWEEP_ADAPTER_HEADER
#define sweep_variant sweep_reference
#include SWEEP_REFERENCE_HEADER
#undef sweep_variant
#include SWEEP_VARIANT_HEADER
static void check(cudaError_t e) {
    if(e!=cudaSuccess) { fprintf(stderr,"CUDA_ERROR: %s\n",cudaGetErrorString(e)); exit(e==cudaErrorMemoryAllocation?77:2); }
}
int main(int argc,char** argv) {
  try {
    int warmup=3,reps=5,iters=20; uint64_t seed=12345; std::vector<std::string> args;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--") { while(++i<argc) args.emplace_back(argv[i]); break; }
        if(i+1>=argc) throw std::runtime_error("missing option value");
        if(a=="--warmup") warmup=std::stoi(argv[++i]);
        else if(a=="--reps") reps=std::stoi(argv[++i]);
        else if(a=="--iters") iters=std::stoi(argv[++i]);
        else if(a=="--seed") seed=std::stoull(argv[++i]);
        else throw std::runtime_error("unknown option");
    }
    if(warmup<0 || reps<2 || iters<1) throw std::runtime_error("need warmup>=0 reps>=2 iters>=1");
    size_t free=0,total=0,required=sweep_adapter::required_bytes(args),reserve=256ul<<20;
    check(cudaMemGetInfo(&free,&total));
    fprintf(stderr,"MEMORY free_bytes=%zu total_bytes=%zu required_bytes=%zu reserve_bytes=%zu\n",free,total,required,reserve);
    if(free<required+reserve) {
        fprintf(stderr,"LOW_MEMORY: insufficient free GPU memory; retry in a cleared slot\n"); return 77;
    }
    auto f=sweep_adapter::make_case(args,seed);
    auto ref=sweep_adapter::params(f,true),p=sweep_adapter::params(f,false);
    check(sweep_reference::init(ref)); check(sweep_variant::init(p));
    cudaFuncAttributes attr{}; check(cudaFuncGetAttributes(&attr,sweep_variant::kernel(p)));
    const size_t bytes=sweep_adapter::output_bytes(f);
    check(cudaMemset(sweep_adapter::output(f,true),0xA5,bytes));
    check(cudaMemset(sweep_adapter::output(f,false),0x5A,bytes));
    sweep_reference::launch(ref,0); check(cudaGetLastError());
    sweep_variant::launch(p,0); check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<unsigned char> a(bytes),b(bytes);
    check(cudaMemcpy(a.data(),sweep_adapter::output(f,true),bytes,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(),sweep_adapter::output(f,false),bytes,cudaMemcpyDeviceToHost));
    size_t mismatch=0; if(memcmp(a.data(),b.data(),bytes)) for(size_t i=0;i<bytes;++i) mismatch+=a[i]!=b[i];
    for(int i=0;i<warmup;++i) sweep_variant::launch(p,0);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    cudaEvent_t begin,end; check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    std::vector<double> samples; double mean=0,ss=0;
    for(int r=0;r<reps;++r) {
        check(cudaEventRecord(begin));
        for(int i=0;i<iters;++i) sweep_variant::launch(p,0);
        check(cudaGetLastError()); check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float ms=0;check(cudaEventElapsedTime(&ms,begin,end));
        samples.push_back(1000.0*ms/iters); mean+=samples.back()/reps;
    }
    for(double v:samples)ss+=(v-mean)*(v-mean);
    // Recheck after timing as well: kernels cannot pass once then corrupt output.
    check(cudaMemcpy(b.data(),sweep_adapter::output(f,false),bytes,cudaMemcpyDeviceToHost));
    size_t after=0; if(memcmp(a.data(),b.data(),bytes)) for(size_t i=0;i<bytes;++i) after+=a[i]!=b[i];
    mismatch=std::max(mismatch,after);
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    printf("{\"status\":\"%s\",\"mean_us\":%.6f,\"sd_us\":%.6f,\"mismatch_bytes\":%zu,\"output_bytes\":%zu,\"regs\":%d,\"local_bytes\":%zu,\"static_smem\":%zu,\"dynamic_smem\":%zu,\"ptxas_key\":\"%s\",\"gpu\":\"%s\",\"cc\":\"%d.%d\",\"samples_us\":[",
        mismatch?"MISMATCH":"EXACT",mean,sqrt(ss/(reps-1)),mismatch,bytes,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,sweep_variant::dynamic_smem(p),sweep_variant::ptxas_key(p),prop.name,prop.major,prop.minor);
    for(size_t i=0;i<samples.size();++i)printf("%s%.6f",i?",":"",samples[i]);
    printf("]}\n");
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));sweep_adapter::destroy(f);
    return mismatch?1:0;
  } catch(const std::exception& e) { fprintf(stderr,"INVALID_CASE: %s\n",e.what());return 2; }
}
