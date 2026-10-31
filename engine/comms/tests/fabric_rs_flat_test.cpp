// reduce-scatter slices over real verbs, flat (Fabric::Rs::flat, FabricConfig::sge2: payload + tail
// word as two SGEs) and not flat (the RS publish layout, one SGE), mesh or ring, striped or not. One process per box: it runs
// on the cluster (RoCE v2 or the cabled ring), not in the host tests.
//   nvcc -O2 -std=c++17 -I engine/comms engine/comms/tests/fabric_rs_flat_test.cpp engine/comms/fabric.cpp -libverbs -lpthread -o fabric_rs_flat_test
//   fabric_rs_flat_test --index I --ips A0,A1,.. [--ips2 B0,..] [--stripe-min N] [--ring] [--ring-devs d0,d1,d2,d3]
//                       [--steps 48] [--rows-max 2048] [--row 14336] [--port 18760]
// Same command on every box. Host-driven, as fabric_group_test: the host writes the payload and rings the doorbell. Two
// collectives a step: c0 reduce-scatter (Q = ceil(rows / NR) rows a slice, the last slices short or empty), c1 a plain
// all-peer collective. Steps alternate in pairs (both parities) between flat and not flat; rows cycle through sizes that give
// empty, short and full slices, up to rows-max (rows-max x row = the DS41 W 2048 prefill chunk: slices over the 1 MiB stripe).
// Checks: every byte of the slice each peer sent me (the payload of rank p's slice `me`), its tail = coll_seq at bytes[me] of
// my recv slot [c0][p]; the plain collective's bytes; and on a flat step, after the proxy drained, that my own send slot
// still holds every row I wrote (a tail word placed inside the payload overwrites a row) and the NR - 1 tail words at
// NR ss + 8 p. The head prints one PASS line, or the box exits non-zero with what differed.
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
static void die(const std::string& s) { fprintf(stderr, "fabric_rs_flat_test: FAIL %s\n", s.c_str()); exit(1); }
static void stall(Fabric& f, const std::string& s) {
  fprintf(stderr, "fabric_rs_flat_test: %s; closing the fabric for its QP and completion report\n", s.c_str());
  f.close();
  die(s);
}
static uint64_t word(uint64_t step, int c, int from, uint64_t k) {   // payload word k of collective c from rank `from`
  uint64_t x = step * 0x9E3779B97F4A7C15ull ^ (uint64_t)c << 48 ^ (uint64_t)from << 40 ^ k;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
  return x | 1;                                        // never 0
}
static const uint64_t kPoison = 0xDEADBEEFDEADBEEFull;

int main(int argc, char** argv) {
  FabricConfig cfg;
  int steps = 48, rows_max = 2048, port = 18760;
  size_t rb = 14336;   // DS41 HID x 2 (bf16 row)
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
    if (a == "--index") cfg.rank = atoi(next().c_str());
    else if (a == "--ips") cfg.ips = split(next());
    else if (a == "--ips2") cfg.ips2 = split(next());
    else if (a == "--stripe-min") cfg.stripe_min = (size_t)atoll(next().c_str());
    else if (a == "--ring") cfg.ring = true;
    else if (a == "--ring-devs") cfg.ring_devs = split(next());
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--rows-max") rows_max = atoi(next().c_str());
    else if (a == "--row") rb = (size_t)atoll(next().c_str());
    else if (a == "--port") port = atoi(next().c_str());
    else die("unknown argument " + a);
  }
  const int nr = (int)cfg.ips.size(), me = cfg.rank;
  if (nr < 2 || me < 0 || me >= nr) die("--index I --ips A0,A1,..: at least two boxes");
  if (rb % 8 || rows_max < 1 || steps < 4) die("--row multiple of 8, --rows-max >= 1, --steps >= 4");
  // every CUDA failure names its cause (cudaGetErrorName + cudaGetErrorString); fabric.cpp's CUCHECK prints the same for its own calls
  if (const cudaError_t e = cudaSetDeviceFlags(cudaDeviceMapHost); e != cudaSuccess)
    die(std::string("cudaSetDeviceFlags(cudaDeviceMapHost): ") + cudaGetErrorName(e) + ": " + cudaGetErrorString(e));
  const int qmax = (rows_max + nr - 1) / nr;
  const size_t ss_max = (size_t)qmax * rb + 8;          // the not-flat slice stride (payload + tail gap)
  cfg.ncoll = 2; cfg.port = port; cfg.sge2 = true;
  cfg.slot_bytes = (size_t)nr * ss_max + 8 * (size_t)nr;
  Fabric f;
  try { f.init(cfg); } catch (const std::exception& e) { die(e.what()); }
  f.start_proxy();
  const FabricDev d = f.dev();
  // rows per step: empty / short / full slices (rows < NR leaves empty slices), the 846-token tail chunk, a full W chunk
  const int rows_cycle[] = {1, 3, 5, 7, 64, 255, 846, rows_max, rows_max - 1};
  long checked = 0, flat_steps = 0;
  for (int s = 0; s < steps; s++) {
    const int par = s & 1, nc = d.ncoll;
    const bool flat = ((s >> 1) & 1) == 0;
    int rows = rows_cycle[(s >> 1) % (sizeof rows_cycle / sizeof rows_cycle[0])];
    if (rows > rows_max) rows = rows_max;
    const int q = (rows + nr - 1) / nr;
    const size_t ss = flat ? (size_t)q * rb : (size_t)q * rb + 8;
    Fabric::Rs rs; rs.mask = 1u; rs.ss = ss; if (flat) rs.flat = 1u;
    for (int p = 0; p < nr; p++) { const int r0 = p * q, n = rows - r0 < 0 ? 0 : (rows - r0 < q ? rows - r0 : q); rs.bytes.push_back((uint32_t)((size_t)n * rb)); }
    const uint32_t plain = (uint32_t)(rb * 4);
    if (s > 0 && !f.wait_proxy_idle(30)) stall(f, "proxy did not post step " + std::to_string(s - 1));
    // write both send slots, then ring
    char* s0 = d.send + fab_send_off(d, par, 0);
    for (size_t k = 0; k < fab_slot_stride(d, 0) / 8; k++) ((volatile uint64_t*)s0)[k] = kPoison;
    if (flat) {   // the producer's own [rows][row] store at 0 of the send slot
      for (size_t k = 0; k < (size_t)rows * rb / 8; k++) ((volatile uint64_t*)s0)[k] = word(s, 0, me, k);
    } else {      // the RS publish layout: slice p at p ss
      for (int p = 0; p < nr; p++)
        for (size_t k = 0; k < rs.bytes[p] / 8; k++) ((volatile uint64_t*)(s0 + fab_rs_slice_off(ss, p)))[k] = word(s, 0, me, (size_t)p * q * rb / 8 + k);
    }
    char* s1 = d.send + fab_send_off(d, par, 1);
    for (size_t k = 0; k < plain / 8; k++) ((volatile uint64_t*)s1)[k] = word(s, 1, me, k);
    f.enqueue_step((unsigned long long)s, std::vector<uint32_t>(nc, plain), rs);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    for (int c = 0; c < nc; c++) *(volatile unsigned long long*)(d.doorbell + c) = coll_seq(s, c, nc);
    for (int c = 0; c < nc; c++)
      for (int p = 0; p < nr; p++) {
        if (p == me) continue;
        volatile uint64_t* slot = (volatile uint64_t*)(d.recv + fab_recv_off(d, par, c, p));
        const size_t nb = c == 0 ? rs.bytes[me] : plain;
        const unsigned long long want = coll_seq(s, c, nc);
        const auto t0 = std::chrono::steady_clock::now();
        for (long spin = 0; slot[nb / 8] != want; spin++) {
          if (*(volatile unsigned*)d.error) stall(f, "fabric error word " + std::to_string(*d.error));
          if ((spin & 0xfffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
            char b[240];
            snprintf(b, sizeof b, "no tail from rank %d (step %d %s, collective %d, rows %d, nb %zu: slot holds %llu, want %llu)", p, s,
                     flat ? "flat" : "not flat", c, rows, nb, (unsigned long long)slot[nb / 8], want);
            stall(f, b);
          }
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const size_t base = c == 0 ? (size_t)me * q * rb / 8 : 0;   // my slice's first word in the sender's row order
        for (size_t k = 0; k < nb / 8; k++)
          if (slot[k] != word(s, c, p, base + k)) {
            char b[240];
            snprintf(b, sizeof b, "step %d %s collective %d rows %d: wrong word %zu of %zu from rank %d", s, flat ? "flat" : "not flat", c, rows, k, nb / 8, p);
            die(b);
          }
        checked += (long)(nb / 8) + 1;
      }
    if (flat) {   // my own send slot after the proxy posted: the rows intact, the tail words after the NR slices
      if (!f.wait_proxy_idle(30)) stall(f, "proxy did not post step " + std::to_string(s));
      for (size_t k = 0; k < (size_t)rows * rb / 8; k++)
        if (((volatile uint64_t*)s0)[k] != word(s, 0, me, k)) die("step " + std::to_string(s) + " flat: my send slot row word " + std::to_string(k) + " overwritten");
      for (int p = 0; p < nr; p++) {
        const uint64_t t = *(volatile uint64_t*)(s0 + fab_rs_tail_off(ss, p, rs.bytes[p], true, nr));
        if (p != me && t != coll_seq(s, 0, nc)) die("step " + std::to_string(s) + " flat: tail word " + std::to_string(p) + " not at NR ss + 8 p");
      }
      flat_steps++;
    }
  }
  if (!f.wait_proxy_idle(60)) die("proxy did not drain");
  f.barrier();
  f.close();
  if (me == 0)
    printf("fabric_rs_flat_test: PASS %s nr %d steps %d (flat %ld) rows-max %d row %zu stripe_min %zu words %ld\n", cfg.ring ? "ring" : "mesh", nr, steps,
           flat_steps, rows_max, rb, cfg.stripe_min, checked);
  else
    printf("fabric_rs_flat_test: rank %d done, words %ld\n", me, checked);
  return 0;
}
