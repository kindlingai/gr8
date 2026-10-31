// Fabric groups over real verbs (#154 gates), one process per pool box: needs RoCE v2 between the boxes, so it runs on
// the cluster, not in the host tests.
//   nvcc -O2 -std=c++17 -I engine/comms engine/comms/tests/fabric_group_test.cpp engine/comms/fabric.cpp -libverbs \
//        -o fabric_group_test
//   fabric_group_test --index I --ips A0,A1,.. [--ips2 B0,B1,..] [--loops 200] [--steps 64] [--port 18660]
// Same command on every box, I = the box's position in --ips (index 0 = the pool head). Steps:
//  1. today's fabric (Fabric::init with every box, rank = index) runs `steps` collectives with fixed payloads; each box
//     keeps a digest of every byte it received, then closes it;
//  2. the group of every box in pool order (fabric_pool.h) runs the same collectives: its digest must equal step 1's
//     (bit-identical), and its received bytes must be the senders' payloads;
//  3. `loops` groups (every box, then subsets in other orders, two fabrics a group), each created, used and dropped;
//     after each, every box's live verbs objects, pinned bytes (fabric_live) and open descriptors are back where they
//     were after step 1 (step 1 also loads CUDA and the verbs provider, which keep descriptors for the process), members
//     and idle boxes alike. Where rdma(8) is installed, the kernel's own count of this process's QPs, MRs, CQs and PDs
//     (`rdma res show`) must be zero after every group too: that check does not trust fabric.cpp's counters.
// Steps 1 and 2 show that a group's config (group_fabric) delivers exactly the bytes today's fabric does: the payloads
// are fixed, so the digests match whenever delivery is correct. The decode gate needs the engine on a group (#156).
// The head prints one PASS line, or the box exits non-zero with what differed.
#include "fabric.h"
#include "fabric_pool.h"

#include <cuda_runtime.h>
#include <dirent.h>

#include <atomic>
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
static int open_fds() {
  int n = 0; DIR* d = opendir("/proc/self/fd"); if (!d) return -1;
  while (auto* e = readdir(d)) if (e->d_name[0] != '.') n++;
  closedir(d); return n - 1;                           // the DIR's own descriptor
}
// QPs, MRs, CQs or PDs the kernel holds for this process (rdma(8) output, counted here); -1 = rdma failed or missing
static long rdma_res(const char* what) {
  const std::string cmd = std::string("rdma res show ") + what + " 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r"); if (!p) return -1;
  const std::string mine = " pid " + std::to_string((int)getpid()) + " ";
  long n = 0; char line[1024];
  while (fgets(line, sizeof line, p)) if (strstr(line, mine.c_str())) n++;
  return pclose(p) == 0 ? n : -1;
}
static void die(const std::string& s) { fprintf(stderr, "fabric_group_test: FAIL %s\n", s.c_str()); exit(1); }
// a run that stopped: close the fabric first, so it prints any QP that left RTS and any failed completion still in its
// CQs (the proxy polls only while it posts), then fail
static void stall(Fabric& f, const std::string& s) {
  fprintf(stderr, "fabric_group_test: %s; closing the fabric for its QP and completion report\n", s.c_str());
  f.close();
  die(s);
}
static uint64_t word(uint64_t step, int c, int from, uint64_t k) {   // payload word k of collective c, sent by pool box `from`
  uint64_t x = step * 0x9E3779B97F4A7C15ull ^ (uint64_t)c << 48 ^ (uint64_t)from << 40 ^ k;
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
  return x | 1;                                        // never 0 (a zero word would hide a missing write)
}
static uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); return h; }
static bool same(const FabricLive& a, const FabricLive& b) {
  return a.fabrics == b.fabrics && a.qps == b.qps && a.mrs == b.mrs && a.cqs == b.cqs && a.pds == b.pds && a.ctxs == b.ctxs &&
         a.pinned_bytes == b.pinned_bytes;
}
static std::string str(const FabricLive& l) {
  char b[200]; snprintf(b, sizeof b, "fabrics %ld qps %ld mrs %ld cqs %ld pds %ld ctxs %ld pinned %ld", l.fabrics, l.qps, l.mrs, l.cqs, l.pds, l.ctxs, l.pinned_bytes);
  return b;
}

// `steps` uniform collectives (all peers) on a started fabric; members[r] = pool index of rank r. Host-driven: the
// host writes the payload and rings the doorbell, as the GPU does in the engine. Returns the digest of what arrived.
static uint64_t run(Fabric& f, const std::vector<int>& members, int steps, int bytes) {
  const FabricDev d = f.dev();
  const int nr = d.nr, me = d.rank, nc = d.ncoll;
  uint64_t h = 0;
  for (int s = 0; s < steps; s++) {
    const int par = s & 1;
    // The proxy waits for doorbell[c] == coll_seq(s, c) exactly, and doorbell[c] is one word for both parities. This
    // loop rings all of a step's doorbells before it waits for any peer, so it can finish step s-1's receives (the
    // peers' proxies posted them) before its own proxy has read step s-1's doorbells. Ringing step s then overwrites a
    // value the proxy never saw, and the proxy waits for it until its 30 s timeout ("have" = the next step's seq). The
    // engine has a layer of GPU work between two rings of one doorbell, so this is timing there, as on main; here the
    // rings are back to back, so the loop lets its proxy drain first.
    if (s > 0 && !f.wait_proxy_idle(30)) {
      char b[160];
      snprintf(b, sizeof b, "proxy did not post step %d (port %d)", s - 1, f.port());
      stall(f, b);
    }
    f.enqueue_step((unsigned long long)s, std::vector<uint32_t>(nc, (uint32_t)bytes));
    for (int c = 0; c < nc; c++) {
      volatile uint64_t* src = (volatile uint64_t*)(d.send + ((size_t)par * nc + c) * d.stride);
      for (int k = 0; k < bytes / 8; k++) src[k] = word(s, c, members[me], k);
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
      *(volatile unsigned long long*)(d.doorbell + c) = coll_seq(s, c, nc);
    }
    for (int c = 0; c < nc; c++)
      for (int p = 0; p < nr; p++) {
        if (p == me) continue;
        volatile uint64_t* slot = (volatile uint64_t*)(d.recv + fab_recv_off(d, par, c, p));
        const unsigned long long want = coll_seq(s, c, nc);
        const auto t0 = std::chrono::steady_clock::now();
        for (long spin = 0; slot[bytes / 8] != want; spin++) {
          if (*(volatile unsigned*)d.error) stall(f, "fabric error word " + std::to_string(*d.error));
          // 30 s, the proxy's own doorbell timeout (a spin count was a few seconds on a loaded box)
          if ((spin & 0xfffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
            char b[200];
            snprintf(b, sizeof b, "no tail from rank %d (port %d, step %d, collective %d: slot holds seq %llu, want %llu)", p,
                     f.port(), s, c, (unsigned long long)slot[bytes / 8], want);
            stall(f, b);
          }
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        for (int k = 0; k < bytes / 8; k++) {
          const uint64_t v = slot[k];
          if (v != word(s, c, members[p], k)) die("step " + std::to_string(s) + " collective " + std::to_string(c) + ": wrong bytes from rank " + std::to_string(p));
          h = mix(h, v);
        }
      }
  }
  if (!f.wait_proxy_idle(60)) die("proxy did not drain");
  return h;
}

int main(int argc, char** argv) {
  PoolConfig pc;
  int loops = 200, steps = 64;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
    if (a == "--index") pc.index = atoi(next().c_str());
    else if (a == "--ips") pc.ips = split(next());
    else if (a == "--ips2") pc.ips2 = split(next());
    else if (a == "--loops") loops = atoi(next().c_str());
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--port") pc.port = atoi(next().c_str());
    else die("unknown argument " + a);
  }
  if (cudaSetDeviceFlags(cudaDeviceMapHost) != cudaSuccess) die("cudaSetDeviceFlags");
  const int n = (int)pc.ips.size();
  if (n < 2 || pc.index < 0 || pc.index >= n) die("--index I --ips A0,A1,..: at least two boxes");
  FabricPool pool;
  try { pool.init(pc); } catch (const std::exception& e) { die(e.what()); }
  FabricConfig base; base.slot_bytes = 6144 * 2 * 8; base.ncoll = 5;   // decode-sized slots (MX x HID x 2 at MX 8)
  const int bytes = 6144 * 2;                                          // one row of a decode all-reduce
  std::vector<int> all(n); for (int i = 0; i < n; i++) all[i] = i;

  // 1. today's fabric, every box
  uint64_t h_fixed = 0;
  {
    FabricConfig c = base; c.rank = pc.index; c.ips = pc.ips; c.ips2 = pc.ips2; c.port = pc.group_port[0];
    Fabric f; f.init(c); f.start_proxy();
    h_fixed = run(f, all, steps, bytes);
    f.barrier(); f.close();
  }
  const FabricLive live0 = fabric_live();             // after step 1: CUDA and the verbs provider are loaded
  const int fds0 = open_fds();
  if (live0.fabrics || live0.qps || live0.mrs || live0.cqs || live0.pds || live0.ctxs || live0.pinned_bytes) die("today's fabric left " + str(live0));
  bool kernel_check = rdma_res("qp") == 0;
  std::string kernel_why = kernel_check ? "" : "rdma res show failed or is missing";
  // 2. the group of every box in pool order
  uint64_t h_group = 0;
  {
    Group g;
    try { g = pool.head() ? pool.announce(all) : pool.follow(600000); } catch (const std::exception& e) { die(e.what()); }
    Fabric f; f.init(group_fabric(pc, g, base)); f.start_proxy();
    if (kernel_check && rdma_res("qp") <= 0) { kernel_check = false; kernel_why = "rdma res show does not list this process's QPs"; }
    h_group = run(f, g.members, steps, bytes);
    f.barrier(); f.close();
  }
  if (h_group != h_fixed) die("the group of every box received other bytes than today's fabric");
  if (!same(fabric_live(), live0)) die("after two fabrics: " + str(fabric_live()) + ", before " + str(live0));
  // 3. groups in a loop: every box, then subsets in rotating orders; two fabrics per group (decode and a second region)
  long groups_in = 0;
  for (int k = 0; k < loops; k++) {
    std::vector<int> L;
    const int sz = k % 3 == 0 ? n : 2 + k % (n - 1);    // n, then 2..n
    for (int j = 0; j < sz; j++) L.push_back((k + j * (1 + k % 2)) % n);
    if (group_refusal(L, n) != "") { L.clear(); for (int j = 0; j < sz; j++) L.push_back((k + j) % n); }
    Group g;
    try { g = pool.head() ? pool.announce(L) : pool.follow(600000); } catch (const std::exception& e) { die(e.what()); }
    if (g.members != L && pool.head()) die("announce returned another group");
    if (g.member()) {
      Fabric f1, f2;
      FabricConfig c1 = group_fabric(pc, g, base), c2 = c1;
      c2.ncoll = 2; c2.port = c1.port + 8;
      f1.init(c1); f1.start_proxy(); f2.init(c2); f2.start_proxy();
      run(f1, g.members, 4, bytes); run(f2, g.members, 4, bytes);
      f1.barrier();                                   // no member still writes into either fabric
      f1.close(); f2.close();
      groups_in++;
    }
    const FabricLive l = fabric_live();
    if (!same(l, live0)) die("group " + std::to_string(g.id) + ": " + str(l) + ", before " + str(live0));
    const int fds = open_fds();
    if (fds != fds0) die("group " + std::to_string(g.id) + ": " + std::to_string(fds) + " open descriptors, before " + std::to_string(fds0));
    if (kernel_check && (k < 8 || k % 25 == 0))
      for (const char* w : {"qp", "mr", "cq", "pd"})
        if (const long n = rdma_res(w)) die("group " + std::to_string(g.id) + (n < 0 ? std::string(": rdma res show failed") : ": the kernel still holds " + std::to_string(n) + " " + w + " of this process"));
  }
  // one last group so every box has checked its counters before the head reports
  try { if (pool.head()) pool.announce(all); else pool.follow(600000); } catch (const std::exception& e) { die(e.what()); }
  printf("fabric_group_test box %d: digest today %016llx group %016llx (identical), %d groups (%ld as a member), "
         "live objects and %d descriptors unchanged, kernel rdma resources %s PASS\n", pc.index, (unsigned long long)h_fixed,
         (unsigned long long)h_group, loops, groups_in, fds0, kernel_check ? "zero" : ("not checked: " + kernel_why).c_str());
  pool.close();
  return 0;
}
