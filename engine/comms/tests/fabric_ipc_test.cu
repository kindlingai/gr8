// GPU-driven intra-node fabric test (NVLink CUDA-IPC 127.0.0.1 and POSIX SHM 127.0.0.2).
// Exercises the exact GPU-in-the-loop contract used by g6m4:
//   - GPU kernel writes send slots in d.send, executes __threadfence_system(), and rings d.doorbell[c]
//   - Proxy transfers payload + tail to peers' d.recv
//   - GPU kernel spins on peers' tails with ld.acquire.sys.global.u64 and verifies every 64-bit word on device
// Tests uniform all-peer collectives (decode 8 KB / 64 KB) and flat + non-flat reduce-scatter slices (prefill).
#include "fabric.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace g6;

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v; size_t a = 0, b;
  while ((b = s.find(',', a)) != std::string::npos) { v.push_back(s.substr(a, b - a)); a = b + 1; }
  if (a < s.size()) v.push_back(s.substr(a));
  return v;
}

static void die(const std::string& s) { fprintf(stderr, "fabric_ipc_test: FAIL %s\n", s.c_str()); exit(1); }

#define CU(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) die(std::string(#x ": ") + cudaGetErrorName(_e) + ": " + cudaGetErrorString(_e)); } while (0)

__device__ __forceinline__ uint64_t payload_word(uint64_t step, int c, int from, uint64_t k) {
  uint64_t x = step * 0x9E3779B97F4A7C15ull ^ ((uint64_t)c << 48) ^ ((uint64_t)from << 40) ^ k;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
  return x | 1ull;
}

__device__ __forceinline__ uint64_t ld_acq_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_rel_sys_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Write uniform send slot, fence, and ring doorbell
__global__ void k_send_uniform(FabricDev d, int par, int c, uint64_t step, uint32_t bytes, unsigned long long seq) {
  uint64_t* dst = reinterpret_cast<uint64_t*>(d.send + fab_send_off(d, par, c));
  const size_t nw = bytes / 8;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nw; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = payload_word(step, c, d.rank, i);
  __syncthreads();
  // single-block kernel so __syncthreads guarantees all payload writes are issued before fence + doorbell
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_rel_sys_u64(d.doorbell + c, seq);
  }
}

// Write RS send slot (flat or spaced), fence, and ring doorbell
__global__ void k_send_rs(FabricDev d, int par, int c, uint64_t step, int rows, int q, size_t rb, size_t ss, int flat, unsigned long long seq) {
  char* s0 = d.send + fab_send_off(d, par, c);
  const size_t total_w = (size_t)rows * rb / 8;
  const size_t slice_w = (size_t)q * rb / 8;
  for (size_t k = threadIdx.x; k < total_w; k += blockDim.x) {
    if (flat) {
      reinterpret_cast<uint64_t*>(s0)[k] = payload_word(step, c, d.rank, k);
    } else {
      const int p = (int)(k / slice_w);
      const size_t j = k - (size_t)p * slice_w;
      reinterpret_cast<uint64_t*>(s0 + fab_rs_slice_off(ss, p))[j] = payload_word(step, c, d.rank, k);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_rel_sys_u64(d.doorbell + c, seq);
  }
}

// Spin on all peers' recv tails and verify every payload word on GPU
__global__ void k_recv_verify(FabricDev d, int par, int c, uint64_t step, uint32_t nb, uint64_t base_word, unsigned long long want, unsigned int* bad_cnt) {
  __shared__ int timed_out;
  if (threadIdx.x == 0) {
    timed_out = 0;
    for (int p = 0; p < d.nr; ++p) {
      if (p == d.rank) continue;
      const uint64_t* tail = reinterpret_cast<const uint64_t*>(d.recv + fab_recv_off(d, par, c, p) + nb);
      unsigned long long spin = 0;
      while (ld_acq_sys_u64(tail) != want) {
        if (*reinterpret_cast<const volatile unsigned int*>(d.error) != 0) { timed_out = 1; break; }
        if (++spin > 2000000000ull) { timed_out = 2; atomicCAS(d.error, 0u, 0x80000000u | (unsigned)p); break; }
      }
      if (timed_out) break;
    }
  }
  __syncthreads();
  if (timed_out) {
    if (threadIdx.x == 0) atomicAdd(bad_cnt, 1000000u + (unsigned)timed_out);
    return;
  }
  const size_t nw = nb / 8;
  for (int p = 0; p < d.nr; ++p) {
    if (p == d.rank) continue;
    const uint64_t* slot = reinterpret_cast<const uint64_t*>(d.recv + fab_recv_off(d, par, c, p));
    for (size_t k = threadIdx.x; k < nw; k += blockDim.x) {
      const uint64_t got = slot[k];
      const uint64_t exp = payload_word(step, c, p, base_word + k);
      if (got != exp) atomicAdd(bad_cnt, 1u);
    }
  }
}

// Latency benchmark: N back-to-back uniform all-gather collectives in one stream
__global__ void k_ping_step(FabricDev d, int par, int c, uint32_t nb, unsigned long long seq) {
  uint64_t* dst = reinterpret_cast<uint64_t*>(d.send + fab_send_off(d, par, c));
  const size_t nw = nb / 8;
  for (size_t i = threadIdx.x; i < nw; i += blockDim.x) dst[i] = seq ^ i;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_rel_sys_u64(d.doorbell + c, seq);
    for (int p = 0; p < d.nr; ++p) {
      if (p == d.rank) continue;
      const uint64_t* tail = reinterpret_cast<const uint64_t*>(d.recv + fab_recv_off(d, par, c, p) + nb);
      while (ld_acq_sys_u64(tail) != seq) {
        if (*reinterpret_cast<const volatile unsigned int*>(d.error) != 0) return;
      }
    }
  }
}

int main(int argc, char** argv) {
  FabricConfig cfg;
  int steps = 20, port = 18820, bench_iters = 200;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
    if (a == "--index") cfg.rank = atoi(next().c_str());
    else if (a == "--ips") cfg.ips = split(next());
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--port") port = atoi(next().c_str());
    else if (a == "--bench-iters") bench_iters = atoi(next().c_str());
    else die("unknown argument " + a);
  }
  const int nr = (int)cfg.ips.size(), me = cfg.rank;
  if (nr < 2 || me < 0 || me >= nr) die("need --index I --ips A0,A1,...");

  int ndev = 0;
  CU(cudaGetDeviceCount(&ndev));
  if (me >= ndev) die("rank >= cudaGetDeviceCount");
  CU(cudaSetDevice(me));
  CU(cudaSetDeviceFlags(cudaDeviceMapHost));

  const size_t rb = 8192;            // Flash HID (4096) * 2 B = 8192 B/row
  const int rows_max = 256;          // 2 MiB RS payload
  const int qmax = (rows_max + nr - 1) / nr;
  const size_t ss_max = (size_t)qmax * rb + 8;

  cfg.ncoll = 2;
  cfg.port = port;
  cfg.sge2 = true;
  cfg.compact = true;
  cfg.slot_bytes = (size_t)nr * ss_max + 8 * (size_t)nr;

  Fabric f;
  try { f.init(cfg); } catch (const std::exception& e) { die(e.what()); }
  f.start_proxy();
  const FabricDev d = f.dev();

  cudaStream_t st = nullptr;
  CU(cudaStreamCreate(&st));
  unsigned int* d_bad = nullptr;
  CU(cudaMalloc(&d_bad, sizeof(unsigned int)));
  CU(cudaMemset(d_bad, 0, sizeof(unsigned int)));

  // 1. Correctness: uniform + RS (flat and spaced), GPU write -> proxy -> GPU spin & verify
  const int rows_cycle[] = {1, 3, 7, 32, 128, 255, 256};
  for (int s = 0; s < steps; ++s) {
    const int par = s & 1;
    const bool flat = ((s >> 1) & 1) == 0;
    const int rows = rows_cycle[s % (sizeof(rows_cycle) / sizeof(rows_cycle[0]))];
    const int q = (rows + nr - 1) / nr;
    const size_t ss = flat ? (size_t)q * rb : (size_t)q * rb + 8;
    Fabric::Rs rs; rs.mask = 1u; rs.ss = ss; if (flat) rs.flat = 1u;
    for (int p = 0; p < nr; ++p) {
      const int r0 = p * q;
      const int n = rows - r0 < 0 ? 0 : (rows - r0 < q ? rows - r0 : q);
      rs.bytes.push_back((uint32_t)((size_t)n * rb));
    }
    const uint32_t plain = (uint32_t)(rb * 4);   // 32 KiB uniform decode/small-batch
    f.enqueue_step((unsigned long long)s, std::vector<uint32_t>(2, plain), rs);

    const unsigned long long seq0 = coll_seq(s, 0, 2);
    const unsigned long long seq1 = coll_seq(s, 1, 2);
    k_send_rs<<<1, 256, 0, st>>>(d, par, 0, (uint64_t)s, rows, q, rb, ss, flat ? 1 : 0, seq0);
    k_send_uniform<<<1, 256, 0, st>>>(d, par, 1, (uint64_t)s, plain, seq1);
    k_recv_verify<<<1, 256, 0, st>>>(d, par, 0, (uint64_t)s, rs.bytes[me], (uint64_t)me * q * rb / 8, seq0, d_bad);
    k_recv_verify<<<1, 256, 0, st>>>(d, par, 1, (uint64_t)s, plain, 0ull, seq1, d_bad);
    CU(cudaStreamSynchronize(st));
    if (!f.wait_proxy_idle(30)) die("proxy idle timeout step " + std::to_string(s));
  }
  unsigned int h_bad = 0;
  CU(cudaMemcpy(&h_bad, d_bad, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  if (h_bad != 0) die("GPU verification errors: " + std::to_string(h_bad));

  // 2. Latency & bandwidth microbenchmark at decode (8 KB, 64 KB) and prefill (1 MB) sizes
  const uint32_t sizes[] = {8192u, 65536u, 1048576u};
  unsigned long long step_base = (unsigned long long)steps + 2;
  for (uint32_t nb : sizes) {
    f.barrier();
    // warmup 10
    for (int i = 0; i < 10; ++i) {
      const unsigned long long stp = step_base++;
      f.enqueue_step(stp, std::vector<uint32_t>{nb, 8u});
      k_ping_step<<<1, 256, 0, st>>>(d, (int)(stp & 1), 0, nb, coll_seq(stp, 0, 2));
      k_ping_step<<<1, 256, 0, st>>>(d, (int)(stp & 1), 1, 8u, coll_seq(stp, 1, 2));
    }
    CU(cudaStreamSynchronize(st));
    f.wait_proxy_idle(30);
    f.barrier();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      const unsigned long long stp = step_base++;
      f.enqueue_step(stp, std::vector<uint32_t>{nb, nb});
      k_ping_step<<<1, 256, 0, st>>>(d, (int)(stp & 1), 0, nb, coll_seq(stp, 0, 2));
      k_ping_step<<<1, 256, 0, st>>>(d, (int)(stp & 1), 1, nb, coll_seq(stp, 1, 2));
    }
    CU(cudaStreamSynchronize(st));
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    f.wait_proxy_idle(30);
    const double per_coll_us = us / (2.0 * bench_iters);
    const double peer_gbps = ((double)nb * (nr - 1) / 1e9) / (per_coll_us * 1e-6);
    if (me == 0) {
      printf("fabric_ipc_test: size=%7u B  avg=%7.2f us/coll  tx=%.2f GB/s/rank (x%d peers)\n",
             nb, per_coll_us, peer_gbps, nr - 1);
      fflush(stdout);
    }
  }

  CU(cudaFree(d_bad));
  CU(cudaStreamDestroy(st));
  f.barrier();
  f.close();
  if (me == 0) printf("fabric_ipc_test: PASS nr=%d steps=%d (0 GPU errors)\n", nr, steps);
  return 0;
}
