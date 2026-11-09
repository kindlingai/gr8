// nccl_probe: NCCL all-reduce timings of one rank, for tools/perftest/perftest.py.
//
//   nccl_probe --rank R --nranks N [--uid HEX] [--sizes 8,1048576,...] [--iters 20] [--warmup 5]
//
// Rank 0 without --uid creates the NCCL unique id, prints "uid <hex>" as its first line and keeps the id's bootstrap
// socket open while it initializes; the other ranks get that line through perftest.py's rendezvous. Every rank then
// runs in-place float sum all-reduces of each size on GPU 0 and prints one line per size:
// "size <bytes> us <time per op> algbw <Gb/s> busbw <Gb/s>", busbw = algbw x 2(N-1)/N as nccl-tests reports it.
// NCCL reads its own environment (NCCL_ALGO, NCCL_IB_HCA, ...) once per process, so each algorithm is its own run.
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const std::string& m)
{
    fprintf(stderr, "nccl_probe: %s\n", m.c_str());
    exit(1);
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) die(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)
#define NK(x) do { ncclResult_t r = (x); if (r != ncclSuccess) die(std::string(#x) + ": " + ncclGetErrorString(r)); } while (0)

int main(int argc, char** argv)
{
    int rank = -1, nranks = 0, iters = 20, warmup = 5;
    std::string uid_hex;
    std::vector<size_t> sizes = {8, 1 << 20, 16 << 20, 134217728};
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--rank") rank = atoi(v.c_str());
        else if (k == "--nranks") nranks = atoi(v.c_str());
        else if (k == "--uid") uid_hex = v;
        else if (k == "--iters") iters = atoi(v.c_str());
        else if (k == "--warmup") warmup = atoi(v.c_str());
        else if (k == "--sizes")
        {
            sizes.clear();
            for (const char* p = v.c_str(); *p;)
            {
                char* e;
                sizes.push_back(strtoull(p, &e, 10));
                p = *e == ',' ? e + 1 : e;
            }
        }
        else die("unknown option " + k);
    }
    if (rank < 0 || nranks < 1 || rank >= nranks || iters < 1 || sizes.empty()) die("needs --rank R --nranks N with 0 <= R < N");

    ncclUniqueId id;
    if (uid_hex.empty())
    {
        if (rank != 0) die("ranks other than 0 need --uid");
        NK(ncclGetUniqueId(&id));
        printf("uid ");
        for (size_t i = 0; i < sizeof id; ++i) printf("%02x", (unsigned char)id.internal[i]);
        printf("\n");
        fflush(stdout);
    }
    else
    {
        if (uid_hex.size() != 2 * sizeof id) die("--uid is not " + std::to_string(2 * sizeof id) + " hex digits");
        for (size_t i = 0; i < sizeof id; ++i) id.internal[i] = (char)strtol(uid_hex.substr(2 * i, 2).c_str(), nullptr, 16);
    }

    CK(cudaSetDevice(0));
    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, nranks, id, rank));
    size_t maxb = 0;
    for (size_t s : sizes) maxb = s > maxb ? s : maxb;
    float* buf;
    CK(cudaMalloc(&buf, maxb));
    CK(cudaMemset(buf, 0, maxb));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (size_t bytes : sizes)
    {
        const size_t n = bytes / sizeof(float) ? bytes / sizeof(float) : 1;
        for (int i = 0; i < warmup; ++i) NK(ncclAllReduce(buf, buf, n, ncclFloat, ncclSum, comm, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < iters; ++i) NK(ncclAllReduce(buf, buf, n, ncclFloat, ncclSum, comm, st));
        CK(cudaEventRecord(b, st));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        const double us = ms * 1e3 / iters, algbw = n * sizeof(float) * 8 / (us * 1e3);
        printf("size %zu us %.2f algbw %.3f busbw %.3f\n", n * sizeof(float), us, algbw, algbw * 2.0 * (nranks - 1) / nranks);
        fflush(stdout);
    }
    NK(ncclCommDestroy(comm));
    return 0;
}
