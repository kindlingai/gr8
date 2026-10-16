// Decode mHC pinned-partial stage byte gate and isolated timing. No fabric-delay simulation.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <random>
#include <vector>
#include "../../mhc/mhc.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// Both rank slots are pinned, as in actual decode. Rank order is fixed 0,1.
// The staged arm changes only pinned-load width and the shared-memory transpose.
// The ready flag excludes transport delay; full fabric visibility still needs the engine gate.
template <bool STAGE> struct ReadyBlockT {
    const volatile unsigned* flag;
    const unsigned short* rank0;
    const unsigned short* rank1;
    const unsigned short* staged=nullptr;
    __device__ __forceinline__ unsigned long long init() { return 0; }
    __device__ __forceinline__ unsigned seq32(unsigned long long) const { return 1; }
    __device__ __forceinline__ void wait(unsigned long long) {
        if (flag) {
            if (!threadIdx.x) { unsigned v; do v = *flag; while (!v); }
            __syncthreads();
        }
        if constexpr (STAGE) {
            __shared__ uint4 slots[2][512];
            const int m=blockIdx.y;
            #pragma unroll
            for(int p=0;p<2;++p) {
                const uint4* src=reinterpret_cast<const uint4*>((p ? rank1 : rank0)+m*4096);
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const int k=threadIdx.x+j*256;
                    slots[p][k]=__ldcv(src+k);
                }
            }
            __syncthreads();
            staged=reinterpret_cast<const unsigned short*>(slots);
        }
    }
    __device__ __forceinline__ float operator()(size_t k) const {
        unsigned short a,b;
        if constexpr(STAGE) {
            const int h=k%4096;
            a=staged[h]; b=staged[4096+h];
        } else {
            a=__ldcv(rank0+k); b=__ldcv(rank1+k);
        }
        float sum=0.f;
        sum+=__uint_as_float((unsigned)a<<16);
        sum+=__uint_as_float((unsigned)b<<16);
        return sum;
    }
};

static std::mt19937 rng(1234);
template <class T> T* dev_fill(size_t n, float lo, float hi, bool bf16 = false)
{
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<unsigned char> h(n * sizeof(T));
    for (size_t i = 0; i < n; ++i) {
        const float f = u(rng);
        if (bf16) { unsigned w; memcpy(&w, &f, 4); unsigned short s = (unsigned short)(w >> 16); memcpy(&h[i * sizeof(T)], &s, 2); }
        else memcpy(&h[i * sizeof(T)], &f, sizeof(T));
    }
    T* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    return d;
}

__global__ void evict_decode_fn(unsigned* p, size_t n) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x)
        p[i]=p[i]*1664525u+1013904223u;
}
int main(int argc, char** argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    std::vector<int> Ms;
    for (int i = 2; i < argc; ++i) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {1, 4};
    if (reps < 1) return 2;
    for (int m : Ms) if (m < 1 || m > 32) return 2;
    int MX = 0;
    for (int m : Ms) MX = m > MX ? m : MX;
    const int H = 4096;
    const float eps = 1e-6f;
    flash_mhc::Weights w;
    w.fn = reinterpret_cast<const __nv_bfloat16*>(dev_fill<unsigned short>((size_t)24 * 4 * H, -0.02f, 0.02f, true));
    w.base = dev_fill<float>(24, -0.5f, 0.5f);
    w.scale = dev_fill<float>(3, 0.5f, 1.5f);
    w.norm = reinterpret_cast<const __nv_bfloat16*>(dev_fill<unsigned short>(H, 0.5f, 1.5f, true));
    const size_t ns = (size_t)MX * 4 * H;
    float* streams0 = dev_fill<float>(ns, -1.f, 1.f);
    float* post0 = dev_fill<float>((size_t)MX * 4, 0.5f, 1.5f);
    float* comb0 = dev_fill<float>((size_t)MX * 16, 0.2f, 0.3f);
    flash_mhc::Buffers b{};
    CK(cudaMalloc(&b.streams, ns * 4)); CK(cudaMalloc(&b.inv, (size_t)MX * 4)); CK(cudaMalloc(&b.mix, (size_t)MX * 24 * 4));
    CK(cudaMalloc(&b.pre, (size_t)MX * 16)); CK(cudaMalloc(&b.post, (size_t)MX * 16)); CK(cudaMalloc(&b.comb, (size_t)MX * 64));
    CK(cudaMalloc(&b.done,64*sizeof(unsigned)));
    unsigned short* own_dev = dev_fill<unsigned short>((size_t)MX * H, -1.f, 1.f, true);
    unsigned short* peer_dev = dev_fill<unsigned short>((size_t)MX * H, -1.f, 1.f, true);
    unsigned short *own_host, *own, *peer_host, *peer_host_d;
    CK(cudaHostAlloc(&own_host, (size_t)MX * H * 2, cudaHostAllocMapped));
    CK(cudaMemcpy(own_host, own_dev, (size_t)MX * H * 2, cudaMemcpyDeviceToHost));
    CK(cudaHostGetDevicePointer(&own, own_host, 0));
    CK(cudaHostAlloc(&peer_host, (size_t)MX * H * 2, cudaHostAllocMapped));
    CK(cudaMemcpy(peer_host, peer_dev, (size_t)MX * H * 2, cudaMemcpyDeviceToHost));
    CK(cudaHostGetDevicePointer(&peer_host_d, peer_host, 0));
    unsigned *flag, *flag_d;
    CK(cudaHostAlloc(&flag, 4, cudaHostAllocMapped)); *flag = 1;
    CK(cudaHostGetDevicePointer(&flag_d, flag, 0));
    __half* out; float* xn;
    CK(cudaMalloc(&out, (size_t)MX * H * 2)); CK(cudaMalloc(&xn, (size_t)MX * H * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto reset = [&](int M) {
        CK(cudaMemsetAsync(b.done,0,64*sizeof(unsigned)));
        CK(cudaMemcpyAsync(b.streams, streams0, (size_t)M * 4 * H * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(b.post, post0, (size_t)M * 16, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(b.comb, comb0, (size_t)M * 64, cudaMemcpyDeviceToDevice));
    };
    unsigned *f16,*bf16;
    const size_t frag_n=(size_t)((MX+7)/8)*256*64;
    CK(cudaMalloc(&f16,frag_n*4)); CK(cudaMalloc(&bf16,frag_n*4));
    auto launch = [&](int kind, int M) {
        if (kind == 0) flash_mhc::k_mhc_consume_fused<<<dim3(24,M),256>>>(b,ReadyBlockT<false>{flag_d,own,peer_host_d},w,H,eps,out,xn,f16,bf16);
        else flash_mhc::k_mhc_consume_fused<<<dim3(24,M),256>>>(b,ReadyBlockT<true>{flag_d,own,peer_host_d},w,H,eps,out,xn,f16,bf16);
        CK(cudaGetLastError());
    };
    // all outputs of one launch from reset buffers, as bytes
    auto snap = [&](int M) {
        std::vector<unsigned char> v;
        auto add = [&](const void* d, size_t n) { size_t o = v.size(); v.resize(o + n); CK(cudaMemcpy(v.data() + o, d, n, cudaMemcpyDeviceToHost)); };
        add(b.streams, (size_t)M * 4 * H * 4); add(b.inv, (size_t)M * 4); add(b.mix, (size_t)M * 24 * 4);
        add(b.pre, (size_t)M * 16); add(b.post, (size_t)M * 16); add(b.comb, (size_t)M * 64);
        add(out, (size_t)M * H * 2); add(xn, (size_t)M * H * 4);
        add(f16,(size_t)((M+7)/8)*256*64*4); add(bf16,(size_t)((M+7)/8)*256*64*4);
        return v;
    };
    unsigned* trash;
    const size_t trash_n=(128ull<<20)/4;
    CK(cudaMalloc(&trash,trash_n*4)); CK(cudaMemset(trash,0,trash_n*4));
    int failures=0;
    for (int M : Ms) {
        reset(M); launch(0,M); CK(cudaDeviceSynchronize());
        const auto ref=snap(M);
        reset(M); launch(1,M); CK(cudaDeviceSynchronize());
        const auto got=snap(M);
        size_t diff=0; for(size_t i=0;i<got.size();++i) diff+=got[i]!=ref[i];
        printf("EXACT M=%d bytes=%zu different=%zu\n",M,got.size(),diff);
        if(diff) {failures++; continue;}
        for(int cold=0;cold<2;++cold) for(int rep=0;rep<5;++rep) {
            std::vector<float> times[2];
            for(int j=0;j<reps+2;++j) for(int k=0;k<2;++k) {
                int arm=(j+k+rep)&1;
                reset(M);
                if(cold) evict_decode_fn<<<512,256>>>(trash,trash_n);
                CK(cudaEventRecord(e0)); launch(arm,M); CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms,e0,e1));
                if(j>=2) times[arm].push_back(ms*1000);
            }
            for(auto& v:times) std::sort(v.begin(),v.end());
            const double c=times[0][times[0].size()/2],v=times[1][times[1].size()/2];
            printf("TIME M=%d cold=%d rep=%d control_us=%.3f staged_us=%.3f gain_pct=%.3f\n",M,cold,rep,c,v,100*(c-v)/c);
        }
    }
    puts(failures ? "MHC_DECODE_STAGE_FAIL" : "MHC_DECODE_STAGE_PASS");
    return failures ? 1 : 0;
}
