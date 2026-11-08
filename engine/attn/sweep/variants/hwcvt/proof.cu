// Exhaustive proof of all four-code words against every scale byte.
#include "../../../ds41_attn.cuh"
#include "widen.cuh"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s: %s\n", #x, cudaGetErrorString(e)); return 2; } } while(0)
__global__ void prove(unsigned long long* bad, unsigned int* first)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t codes = i & 65535u, scale = i >> 16;
    if (scale >= 256) return;
    const __nv_bfloat162 s = __float2bfloat162_rn(ds41a::f_e4m3((uint8_t)scale));
    uint32_t a,b,c,d;
    ds41a::cq_dq4(codes,s,a,b);
    hwcvt::dq4(codes,s,c,d);
    if(a != c || b != d) { atomicAdd(bad,1ull); atomicMin(first,i); }
}
int main()
{
    unsigned long long *d_bad, bad=0; unsigned int *d_first, first=~0u;
    CHECK(cudaMalloc(&d_bad,sizeof(bad))); CHECK(cudaMalloc(&d_first,sizeof(first)));
    CHECK(cudaMemcpy(d_bad,&bad,sizeof(bad),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_first,&first,sizeof(first),cudaMemcpyHostToDevice));
    prove<<<65536,256>>>(d_bad,d_first); CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(&bad,d_bad,sizeof(bad),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&first,d_first,sizeof(first),cudaMemcpyDeviceToHost));
    printf("words=65536 scales=256 comparisons=16777216 mismatch_words=%llu first=0x%08x\n",bad,first);
    CHECK(cudaFree(d_bad)); CHECK(cudaFree(d_first)); return bad ? 1 : 0;
}
