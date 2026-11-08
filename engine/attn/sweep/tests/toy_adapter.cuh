#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
namespace sweep_adapter {
struct Params {const uint32_t* x;uint32_t* out;int n;};
struct Fixture {uint32_t *x,*a,*b;int n;std::vector<uint32_t> expected;};
inline void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline size_t required_bytes(const std::vector<std::string>&){return 3*4096*sizeof(uint32_t);}
inline Fixture make_case(const std::vector<std::string>&,uint64_t){
    Fixture f{};f.n=4096;ck(cudaMalloc(&f.x,f.n*4));ck(cudaMalloc(&f.a,f.n*4));ck(cudaMalloc(&f.b,f.n*4));
    f.expected.resize(f.n);return f;
}
inline Params params(Fixture& f,bool ref){return {f.x,ref?f.a:f.b,f.n};}
inline Params params_with_output(Fixture& f,Fixture& o,bool ref){return {f.x,ref?o.a:o.b,f.n};}
inline void* output(Fixture& f,bool ref){return ref?f.a:f.b;}
inline size_t output_bytes(const Fixture& f){return f.n*4;}
inline void refresh_inputs(Fixture& f,uint64_t nonce,cudaStream_t st){
    for(int i=0;i<f.n;++i)f.expected[i]=uint32_t(nonce*104729+i*17);
    ck(cudaMemcpyAsync(f.x,f.expected.data(),f.n*4,cudaMemcpyHostToDevice,st));ck(cudaStreamSynchronize(st));
}
inline void snapshot_inputs(Fixture& f,cudaStream_t st){ck(cudaStreamSynchronize(st));ck(cudaMemcpy(f.expected.data(),f.x,f.n*4,cudaMemcpyDeviceToHost));}
inline size_t check_integrity(Fixture& f,cudaStream_t st){
    ck(cudaStreamSynchronize(st));std::vector<uint32_t> now(f.n);ck(cudaMemcpy(now.data(),f.x,f.n*4,cudaMemcpyDeviceToHost));
    size_t bad=0;for(int i=0;i<f.n;++i)bad+=now[i]!=f.expected[i];return bad;
}
inline void destroy(Fixture& f){ck(cudaFree(f.x));ck(cudaFree(f.a));ck(cudaFree(f.b));}
__global__ void toy(Params p){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<p.n)p.out[i]=p.x[i]*3u+17u;}
}
