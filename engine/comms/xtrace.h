// g6 stage-0 transport trace (G6_XTRACE=1; default off). Measurement only: nothing here feeds a compute result.
// GPU side: one record of XT_W u64 per (step % XT_STEPS, collective) in DEVICE memory (FabricDev::xt; null = off).
//   [0] globaltimer before the final publish work (last CTA, before its system fence + doorbell store)
//   [1] globaltimer right after the doorbell store       [2 + p] globaltimer when peer p's tail was seen (own = 0)
//   [10] globaltimer at reduce end (k_ar_add_norm, max over CTAs)   [11] seq of the collective that wrote [1]
// Host proxy side (fabric.cpp): per (step % XT_STEPS, collective, rail) 4 ticks of the CPU counter (CNTVCT_EL0 on
// ARM, rdtsc on x86): doorbell seen, tail stored, first post, last post on that rail. Per-QP WC error counters.
// Everything is dumped OUTSIDE the timed loop: at stop_proxy / exit, or once when the file G6_XTRACE_QUIT appears.
// Only local intervals are meaningful across ranks (each clock is per node).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace g6 {

constexpr int XT_STEPS = 64, XT_W = 16;
enum { XT_PUB = 0, XT_DB = 1, XT_TAIL = 2, XT_RED = 10, XT_SEQ = 11 };

inline bool xtrace_env() { const char* e = getenv("G6_XTRACE"); return e && atoi(e) == 1; }

// device buffer for one rank's GPU records (zeroed); nullptr when G6_XTRACE is off
inline unsigned long long* xt_alloc(int ncoll) {
  if (!xtrace_env()) return nullptr;
  unsigned long long* p = nullptr;
  const size_t n = (size_t)XT_STEPS * ncoll * XT_W * 8;
  if (cudaMalloc((void**)&p, n) != cudaSuccess || cudaMemset(p, 0, n) != cudaSuccess) { fprintf(stderr, "g6xtrace: alloc failed\n"); return nullptr; }
  return p;
}

// copy the GPU records out on a private non-blocking stream and print them.
// Format: "G step c pub_ns db_ns tail0_ns .. tail{nr-1}_ns red_ns | db-pub tail_p-db .. red-maxtail" (ns, local clock)
inline void xt_dump_gpu(FILE* o, const unsigned long long* dev, int ncoll, int nr, int rank) {
  if (!dev || !o) return;
  const size_t n = (size_t)XT_STEPS * ncoll * XT_W;
  unsigned long long* h = (unsigned long long*)malloc(n * 8);
  cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaMemcpyAsync(h, dev, n * 8, cudaMemcpyDeviceToHost, s); cudaStreamSynchronize(s); cudaStreamDestroy(s);
  fprintf(o, "# G rank %d: step c pub db tail[0..%d] red | d_db_pub d_tail[p]_db.. d_red_lasttail (ns, globaltimer)\n", rank, nr - 1);
  for (int i = 0; i < XT_STEPS * ncoll; i++) {
    const unsigned long long* r = h + (size_t)i * XT_W;
    if (!r[XT_SEQ]) continue;
    const unsigned long long step = (r[XT_SEQ] - 1) / ncoll; const int c = (int)((r[XT_SEQ] - 1) % ncoll);
    fprintf(o, "G %llu %d %llu %llu", step, c, r[XT_PUB], r[XT_DB]);
    unsigned long long mx = 0;
    for (int p = 0; p < nr && p < 8; p++) { fprintf(o, " %llu", r[XT_TAIL + p]); if (r[XT_TAIL + p] > mx) mx = r[XT_TAIL + p]; }
    fprintf(o, " %llu | %lld", r[XT_RED], (long long)(r[XT_DB] - r[XT_PUB]));
    for (int p = 0; p < nr && p < 8; p++) fprintf(o, " %lld", r[XT_TAIL + p] ? (long long)(r[XT_TAIL + p] - r[XT_DB]) : 0LL);
    fprintf(o, " %lld\n", (r[XT_RED] && mx) ? (long long)(r[XT_RED] - mx) : 0LL);
  }
  free(h);
}

}  // namespace g6
