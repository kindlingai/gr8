// gpuprobe: the GPU part of the rank health probe (tools/health/rank_probe.py runs it on each node before a boot).
// Measures, on device 0, for about --seconds seconds:
//   hbm_gbs   sustained device-memory read rate (grid-stride read of a 1 GiB buffer; the GB10 LPDDR is the HBM here)
//   h2d_gbs   host-to-device copy rate from pinned memory (256 MiB copies)
// and prints one JSON line. The read kernel is the load the probe samples the GPU clock under, so the tool writes
// "LOAD" on stdout when the load phase starts and "DONE" when it ends (rank_probe.py samples nvidia-smi in between).
// It never leaves memory allocated after it exits. Needs no other files; build: nvcc -O2 -arch=sm_121a gpuprobe.cu.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"error\":\"%s: %s\"}\n", #x, cudaGetErrorString(e_)); return 2; } } while (0)

__global__ void read_sum(const uint4* p, size_t n, unsigned long long* out)
{
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        const uint4 v = p[i];
        s += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0x123456789abcdefULL) out[0] = s;        // keeps the loads alive
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; }

int main(int argc, char** argv)
{
    double seconds = 4.0;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { printf("{\"error\":\"no CUDA device\"}\n"); return 2; }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const size_t bytes = (size_t)1 << 30, hbytes = (size_t)256 << 20;
    uint4* d = nullptr;
    unsigned long long* sink = nullptr;
    CK(cudaMalloc((void**)&d, bytes));
    CK(cudaMalloc((void**)&sink, sizeof *sink));
    CK(cudaMemset(d, 0x5a, bytes));
    void* h = nullptr;
    CK(cudaHostAlloc(&h, hbytes, cudaHostAllocDefault));
    memset(h, 1, hbytes);
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    int sms = prop.multiProcessorCount;
    const size_t n = bytes / sizeof(uint4);

    // copies first (short), then the read load: rank_probe.py samples the clock during the read phase
    std::vector<double> h2d;
    for (int i = 0; i < 12; ++i)
    {
        const double t0 = now_s();
        CK(cudaMemcpyAsync(d, h, hbytes, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
        if (i >= 2) h2d.push_back(hbytes / (now_s() - t0) / 1e9);     // the first two are warm-up
    }
    printf("LOAD\n");
    std::vector<double> rd;
    const double end = now_s() + seconds;
    for (int i = 0; now_s() < end || i < 4; ++i)
    {
        const double t0 = now_s();
        read_sum<<<sms * 8, 256, 0, st>>>(d, n, sink);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(st));
        if (i >= 2) rd.push_back(bytes / (now_s() - t0) / 1e9);
    }
    printf("DONE\n");
    printf("{\"device\":\"%s\",\"sms\":%d,\"hbm_gbs\":%.2f,\"h2d_gbs\":%.2f,\"read_passes\":%zu,\"copy_passes\":%zu}\n",
           prop.name, sms, median(rd), median(h2d), rd.size(), h2d.size());
    cudaFreeHost(h);
    cudaFree(d);
    cudaFree(sink);
    return 0;
}
