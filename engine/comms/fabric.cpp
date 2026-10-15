// gr8 fabric implementation (fabric.h).
#include "fabric.h"
#include "bulk_proxy.h"
#include "progress_cpus.h"
#include "relay.h"
#include "xtrace.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#if !defined(__aarch64__)
#include <x86intrin.h>
#endif

namespace g6 {

#define FCHECK(x) do { if (!(x)) { fprintf(stderr, "[fabric r%d] FAIL %s:%d %s (errno %d %s)\n", g_rank, __FILE__, __LINE__, #x, errno, strerror(errno)); abort(); } } while (0)
#define CUCHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "[fabric r%d] CUDA %s at %s:%d\n", g_rank, cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
static int g_rank = -1;

namespace {
// bit c of a 32-bit collective mask; c >= 32 is never set (a plain `m >> c` is undefined there: ncoll = 2 NL + 1)
inline bool bit(unsigned m, size_t c) { return c < 32 && (m >> c & 1u); }

std::string slurp(const std::string& p) { std::ifstream f(p); std::string s; std::getline(f, s); return s; }

struct DevInfo { std::string dev, netdev; int gid_index = -1; };
DevInfo find_dev(const std::string& ip) {
  DevInfo d;
  struct ifaddrs* ifa; FCHECK(getifaddrs(&ifa) == 0);
  for (auto* p = ifa; p; p = p->ifa_next) {
    if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
    char buf[64]; inet_ntop(AF_INET, &((sockaddr_in*)p->ifa_addr)->sin_addr, buf, sizeof buf);
    if (ip == buf) d.netdev = p->ifa_name;
  }
  freeifaddrs(ifa);
  FCHECK(!d.netdev.empty());
  DIR* dir = opendir("/sys/class/infiniband"); FCHECK(dir);
  while (auto* e = readdir(dir)) {
    if (e->d_name[0] == '.') continue;
    std::string base = std::string("/sys/class/infiniband/") + e->d_name;
    DIR* nd = opendir((base + "/device/net").c_str()); if (!nd) continue;
    while (auto* n = readdir(nd)) if (d.netdev == n->d_name) d.dev = e->d_name;
    closedir(nd);
  }
  closedir(dir);
  FCHECK(!d.dev.empty());
  in_addr a; inet_pton(AF_INET, ip.c_str(), &a);
  const uint8_t* b = (const uint8_t*)&a;
  char want[64]; snprintf(want, sizeof want, "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x", b[0], b[1], b[2], b[3]);
  for (int i = 0; i < 64; i++) {
    std::string base = "/sys/class/infiniband/" + d.dev + "/ports/1/";
    if (slurp(base + "gids/" + std::to_string(i)) == want && slurp(base + "gid_attrs/types/" + std::to_string(i)) == "RoCE v2") { d.gid_index = i; break; }
  }
  FCHECK(d.gid_index >= 0);
  return d;
}

int tcp_connect(const std::string& ip, int port) {
  for (int tries = 0; tries < 1200; tries++) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_port = htons(port); inet_pton(AF_INET, ip.c_str(), &sa.sin_addr);
    if (connect(s, (sockaddr*)&sa, sizeof sa) == 0) { int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); return s; }
    close(s); usleep(100000);
  }
  FCHECK(false); return -1;
}
void xfer(int s, void* p, size_t n, bool snd) {
  char* c = (char*)p;
  while (n) { ssize_t k = snd ? send(s, c, n, 0) : recv(s, c, n, MSG_WAITALL); FCHECK(k > 0); c += k; n -= k; }
}
std::vector<char> allgather(int rank, int nr, const std::string& ip0, int port, const void* mine, size_t n) {
  std::vector<char> all(n * nr);
  memcpy(all.data() + n * rank, mine, n);
  if (rank == 0) {
    int ls = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_port = htons(port); sa.sin_addr.s_addr = INADDR_ANY;
    FCHECK(bind(ls, (sockaddr*)&sa, sizeof sa) == 0); FCHECK(listen(ls, 64) == 0);
    std::vector<int> cs(nr, -1);
    for (int i = 1; i < nr; i++) { int c = accept(ls, nullptr, nullptr); FCHECK(c >= 0); int r; xfer(c, &r, 4, false); FCHECK(r > 0 && r < nr); cs[r] = c; xfer(c, all.data() + n * r, n, false); }
    for (int i = 1; i < nr; i++) { xfer(cs[i], all.data(), n * nr, true); close(cs[i]); }
    close(ls);
  } else {
    int s = tcp_connect(ip0, port); xfer(s, &rank, 4, true); xfer(s, (void*)mine, n, true); xfer(s, all.data(), n * nr, false); close(s);
  }
  return all;
}

// G6_XTRACE proxy clock: CNTVCT_EL0 on ARM (GB10), rdtsc elsewhere; read only when the trace is on
inline uint64_t xt_tick() {
#if defined(__aarch64__)
  uint64_t v; asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v) :: "memory"); return v;
#else
  return __rdtsc();
#endif
}
inline uint64_t xt_freq() {
#if defined(__aarch64__)
  uint64_t v; asm volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
#else
  return 0;   // unknown: ticks only
#endif
}

struct QPInfo { uint32_t qpn, psn, rkey; uint64_t addr; uint8_t gid[16]; };
struct Rail { DevInfo di; ibv_context* ctx = nullptr; ibv_pd* pd = nullptr; ibv_cq* cq = nullptr; ibv_mr* mr = nullptr; ibv_gid gid{}; };

// ---- switchless ring (FabricConfig::ring) ---------------------------------------------------------------------------
// Ranks sit on a cable ring: port f0 of rank r is cabled to port f1 of rank r+1, and each port shows up as two RDMA
// devices (one per PCIe root). FabricConfig::ring_devs names this box's four devices, in the order
//   relay-out (f0, root 0), direct-next (f0, root 1), relay-in (f1, root 0), direct-prev (f1, root 1).
// Neighbours are written directly. With four ranks, the diagonal peer (r+2) is reached through the next rank: the
// payload goes into a relay ring on the next rank (relay.h) and a SEND gives its ring position and destination offset.
// That rank's relay thread writes it on to its own next rank, at the offset a direct write would have used, and writes
// the origin a credit once the forward has completed. One QP chain per origin keeps a payload ahead of its tail and the
// bulk ranges ahead of their flags. Writes of up to kRelayInline bytes (the bulk flags) travel inside the SEND.
struct RelayHdr { uint64_t off, pos; uint32_t len, kind; };   // kind 0: len bytes at ring position pos; 1: inline
constexpr int kRelayRecv = 4096, kRelayBuf = 64, kRelayInline = kRelayBuf - (int)sizeof(RelayHdr);
struct RingInfo { QPInfo dn, dp, orq, ri, fwd, fin; };        // my QPs; addr/rkey = where the paired QP writes
struct RelayCtx {
  ibv_qp* ri = nullptr; ibv_cq* ri_cq = nullptr; ibv_qp* fwd = nullptr; ibv_cq* fwd_cq = nullptr;
  char* rbuf = nullptr; char* ring = nullptr; uint32_t ring_lkey = 0;
  uint64_t fin_addr = 0; uint32_t fin_rkey = 0, rbuf_lkey = 0; volatile unsigned* err = nullptr; int rank = -1;
  uint64_t cred_addr = 0; uint32_t cred_rkey = 0;         // the origin's credit word
  uint64_t done = 0, sent = 0;                            // ring position forwarded, and the last credit written
  int unsig = 0, inflight = 0; std::deque<int> batches;
  RelayCreditQueue credits; unsigned credit_sq_capacity = 0;
  std::string ri_dev;
};
std::mutex g_relay_mu;
RelayCtx* g_relay[32]; std::atomic<int> g_nrelay{0};
bool g_relay_started = false;

int ring_gid_index(const std::string& dev) {
  for (int i = 0; i < 64; i++) {
    const std::string base = "/sys/class/infiniband/" + dev + "/ports/1/";
    if (slurp(base + "gids/" + std::to_string(i)).rfind("0000:0000:0000:0000:0000:ffff:", 0) == 0 &&
        slurp(base + "gid_attrs/types/" + std::to_string(i)) == "RoCE v2") return i;
  }
  return -1;
}

// One relay thread for every ring Fabric of the process: busy-polls each relay-in CQ and forwards in arrival order.
void relay_loop(int cpu) {
  if (cpu >= 0) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof cs, &cs) == 0) { progress_cpu_add(cpu); progress_tid_add((int)syscall(SYS_gettid)); }
  }
  auto fail = [](RelayCtx& c, const char* what, int st) {
    fprintf(stderr, "[fabric r%d] relay %s (%s)\n", c.rank, what, st >= 0 ? ibv_wc_status_str((ibv_wc_status)st) : "-");
    __sync_val_compare_and_swap(c.err, 0u, 0x75000000u);
  };
  auto credit_fail = [](RelayCtx& c, int rc, int saved_errno, const ibv_send_wr* bad, const ibv_send_wr* wr) {
    ibv_qp_attr at{}; ibv_qp_init_attr ia{};
    const int qrc = ibv_query_qp(c.ri, &at, IBV_QP_STATE, &ia);
    const std::string rc_msg = strerror(rc);                  // strerror has one buffer: copy before the second call
    const std::string errno_msg = strerror(saved_errno);
    fprintf(stderr, "[fabric r%d] relay credit post (-): rc=%d (%s) errno=%d (%s) dev=%s "
                    "qpn=%u state=%d query_rc=%d sq_capacity=%u sq_limit=%u credits_in_flight=%u "
                    "credit_batches=%zu credit_unsignaled=%u done=%llu sent=%llu bad_wr=%d\n",
            c.rank, rc, rc_msg.c_str(), saved_errno, errno_msg.c_str(), c.ri_dev.c_str(),
            c.ri->qp_num, qrc ? -1 : (int)at.qp_state, qrc, c.credit_sq_capacity, c.credits.capacity,
            c.credits.in_flight, c.credits.batches.size(), c.credits.unsignaled,
            (unsigned long long)c.done, (unsigned long long)c.sent, bad == wr);
    __sync_val_compare_and_swap(c.err, 0u, 0x75000000u);
  };
  for (;;) {
    const int n = g_nrelay.load(std::memory_order_acquire);
    for (int k = 0; k < n; k++) {
      RelayCtx& c = *g_relay[k];
      if (*c.err) continue;
      ibv_wc fc[32]; int m;
      while ((m = ibv_poll_cq(c.fwd_cq, 32, fc)) > 0)
        for (int i = 0; i < m; i++) {
          if (fc[i].status != IBV_WC_SUCCESS) { fail(c, "forward completion", fc[i].status); break; }
          relay_forward_completed(c.inflight, c.batches, fc[i].wr_id, c.done);
        }
      if (m < 0) { fail(c, "forward CQ poll", -1); continue; }
      if (c.done != c.sent && c.credits.available() && !*c.err) { // credit: the origin may reuse the ring below done
        ibv_sge cs{(uint64_t)&c.done, 8, 0};
        ibv_send_wr w{}, *bad = nullptr; w.sg_list = &cs; w.num_sge = 1; w.opcode = IBV_WR_RDMA_WRITE;
        w.wr.rdma.remote_addr = c.cred_addr; w.wr.rdma.rkey = c.cred_rkey;
        const bool sig = c.credits.signal_next();
        w.send_flags = IBV_SEND_INLINE | (sig ? IBV_SEND_SIGNALED : 0);
        const int rc = ibv_post_send(c.ri, &w, &bad);
        if (rc != 0) { const int e = errno; credit_fail(c, rc, e, bad, &w); continue; }
        c.credits.posted(sig);
        c.sent = c.done;
      }
      ibv_wc wc[32];
      const int got = ibv_poll_cq(c.ri_cq, 32, wc);
      if (got < 0) { fail(c, "relay-in CQ poll", -1); continue; }
      for (int i = 0; i < got && !*c.err; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) { fail(c, "relay-in completion", wc[i].status); break; }
        if (wc[i].opcode != IBV_WC_RECV) {                  // a signaled credit write
          if (!c.credits.completed()) { fail(c, "unexpected credit completion", -1); break; }
          continue;
        }
        const int slot = (int)wc[i].wr_id;
        char* buf = c.rbuf + (size_t)slot * kRelayBuf;
        RelayHdr h; memcpy(&h, buf, sizeof h);
        // an inline write is at most what follows the header in its 64-byte slot; more would read past the slot
        if (h.kind > 1 || (h.kind == 1 && h.len > (uint32_t)kRelayInline)) {
          fail(c, "relay header", (int)h.len); break;
        }
        while (c.inflight >= 1800 && !*c.err) {            // forward queue full: reap
          ibv_wc f; const int r = ibv_poll_cq(c.fwd_cq, 1, &f);
          if (r < 0 || (r == 1 && f.status != IBV_WC_SUCCESS)) { fail(c, "forward drain", r < 0 ? -1 : f.status); break; }
          if (r == 1) relay_forward_completed(c.inflight, c.batches, f.wr_id, c.done);
        }
        ibv_sge sge;
        ibv_send_wr w{}, *bad = nullptr; w.sg_list = &sge; w.num_sge = 1; w.opcode = IBV_WR_RDMA_WRITE;
        w.wr.rdma.remote_addr = c.fin_addr + h.off; w.wr.rdma.rkey = c.fin_rkey;
        if (h.kind == 1) { sge = ibv_sge{(uint64_t)(buf + sizeof h), h.len, 0}; w.send_flags = IBV_SEND_INLINE; }
        else { sge = ibv_sge{(uint64_t)(c.ring + h.pos % kRelayRing), h.len, c.ring_lkey}; w.wr_id = h.pos + relay_pad(h.len); }
        const bool sig = ++c.unsig >= 64 || h.kind == 0;     // every ring forward reports its end position
        if (sig) { w.send_flags |= IBV_SEND_SIGNALED; c.batches.push_back(c.unsig); c.unsig = 0; }
        ++c.inflight;
        if (ibv_post_send(c.fwd, &w, &bad) != 0) { fail(c, "forward post", -1); break; }
        ibv_sge rs{(uint64_t)buf, kRelayBuf, c.rbuf_lkey};
        ibv_recv_wr rw{}, *rbad = nullptr; rw.wr_id = (uint64_t)slot; rw.sg_list = &rs; rw.num_sge = 1;
        if (ibv_post_recv(c.ri, &rw, &rbad) != 0) { fail(c, "recv repost", -1); break; }
      }
    }
  }
}
struct Job { unsigned long long step; std::vector<uint32_t> sizes; unsigned rs_mask = 0; size_t rs_ss = 0; std::vector<uint32_t> rs_bytes; unsigned skip = 0;
             unsigned p2p = 0; int peer = -1; bool bulk = false; Fabric::BulkJob b; };
}  // namespace

struct Fabric::Impl {
  int nr = 0, nrails = 1;
  Rail rails[4];
  std::vector<ibv_qp*> qp;
  std::vector<QPInfo> peer;  // peer[p] = p's QP toward me (its addr/rkey are for its recv region)
  char* region = nullptr; size_t region_bytes = 0;  // one pinned region: send | recv | doorbell | error
  std::vector<int> unsignaled;
  bool var = false; VLayout L;  // variable layout (cfg.vcap non-empty)
  bool bulk = false; BulkDev bd{};                    // bulk mode (cfg.bulk_bytes > 0)
  std::vector<std::deque<int>> batches;               // bulk: per peer, WR counts of the signaled batches in flight
  std::vector<int> inflight;                          // bulk: per peer, WRs posted and not yet known complete
  std::thread th;
  std::atomic<bool> stop{false};
  std::mutex mu; std::condition_variable cv; std::deque<Job> q; std::atomic<long> pending{0};
  std::atomic<bool> exited{false};                   // the proxy thread has returned (failure or stop)
  int pinned_cpu = -1;                               // CPU registered with progress_cpus.h (-1 = none)
  std::atomic<int> proxy_tid{0};                     // thread id of the running proxy, registered with progress_cpus.h (0 = none)
  std::atomic<bool> proxy_started{false};            // the proxy thread has pinned itself and registered its tid
  int rail_of(int p, int me) const { return ring ? ring_rail[p] : nrails == 1 ? 0 : (p + me) % 2; }
  // ring mode: rails[0] direct-next, [1] direct-prev, [2] relay-out (also carries the diagonal's QP), [3] relay-in
  bool ring = false; int nxt = -1, prv = -1, dia = -1; int ring_rail[8] = {};
  RelayCtx relay;
  RelayOut rout;                                     // my side of the relay ring on the next rank
  uint64_t* credit = nullptr;                        // the relay's credit: ring position forwarded (written by its NIC)
  uint64_t ring_addr = 0; uint32_t ring_rkey = 0;
  volatile unsigned* err = nullptr;
  // the number of WRs post(p, wr) puts on the send queue
  int wrs(int p, const ibv_send_wr& wr) const {
    const uint32_t len = wr.sg_list[0].length;
    return ring && p == dia && len > (uint32_t)kRelayInline ? 2 * (int)((len + kRelayChunk - 1) / kRelayChunk) : 1;
  }
  // post one write to peer p. The diagonal peer: each chunk of the payload goes into the relay ring, then a SEND gives
  // its position and destination offset (or the SEND carries a short payload itself). The last SEND takes the write's
  // signaled flag and wr_id. With the ring full, this waits for the relay's credit, and gives up after 30 s with the
  // error word set.
  void post(int p, ibv_send_wr& wr) {
    ibv_send_wr* bad = nullptr;
    if (!ring || p != dia) { FCHECK(ibv_post_send(qp[p], &wr, &bad) == 0); return; }
    FCHECK(wr.num_sge == 1 && wr.opcode == IBV_WR_RDMA_WRITE);
    const uint64_t off = wr.wr.rdma.remote_addr - peer[p].addr, src = wr.sg_list[0].addr;
    const uint32_t len = wr.sg_list[0].length, lkey = wr.sg_list[0].lkey;
    RelayHdr h{off, 0, len, 1};
    ibv_sge hs[2]; hs[0] = ibv_sge{(uint64_t)&h, (uint32_t)sizeof h, 0};
    ibv_send_wr s{}; s.opcode = IBV_WR_SEND; s.sg_list = hs; s.num_sge = 1;
    if (len <= (uint32_t)kRelayInline) {
      hs[1] = ibv_sge{src, len, 0}; s.num_sge = 2;
      s.wr_id = wr.wr_id; s.send_flags = IBV_SEND_INLINE | (wr.send_flags & IBV_SEND_SIGNALED);
      FCHECK(ibv_post_send(qp[p], &s, &bad) == 0); return;
    }
    for (uint32_t at = 0; at < len; at += kRelayChunk) {
      const uint32_t n = len - at < kRelayChunk ? len - at : kRelayChunk;
      uint64_t pos;
      const auto t0 = std::chrono::steady_clock::now();
      while (!rout.reserve(n, kRelayRing, __atomic_load_n(credit, __ATOMIC_ACQUIRE), &pos)) {
        if (*err) return;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
          fprintf(stderr, "[fabric] relay ring: no credit for 30 s (ring at %llu, forwarded %llu)\n",
                  (unsigned long long)rout.pos, (unsigned long long)__atomic_load_n(credit, __ATOMIC_ACQUIRE));
          __sync_val_compare_and_swap(err, 0u, 0x75000001u);
          return;
        }
      }
      ibv_sge ds{src + at, n, lkey};
      ibv_send_wr w{}; w.opcode = IBV_WR_RDMA_WRITE; w.sg_list = &ds; w.num_sge = 1; w.next = &s;
      w.wr.rdma.remote_addr = ring_addr + pos % kRelayRing; w.wr.rdma.rkey = ring_rkey;
      h = RelayHdr{off + at, pos, n, 0};
      const bool last = at + n == len;
      s.wr_id = last ? wr.wr_id : 0; s.send_flags = IBV_SEND_INLINE | (last ? (wr.send_flags & IBV_SEND_SIGNALED) : 0);
      FCHECK(ibv_post_send(qp[p], &w, &bad) == 0);
    }
  }
  // G6_XTRACE (xtrace.h). xenv: env on (WC error counters dumped for every fabric); xt: proxy stamps on this fabric
  bool xenv = false, xt = false, xdumped = false;
  unsigned long long* xgpu = nullptr;                // device records (dev_.xt), cudaMalloc'd
  std::vector<uint64_t> xp;                          // [XT_STEPS][ncoll][2 rails][4]: db seen, tail stored, first, last post
  std::vector<unsigned long long> xpseq;             // [XT_STEPS][ncoll] seq of the proxy record
  unsigned long long xwc[2][8][3] = {};              // [rail][peer][retry-exceeded, rnr-exceeded, other bad status]
  std::thread xquit;                                 // G6_XTRACE_QUIT watcher
  void wc_count(int r, int peer, int st) {
    if (peer < 0 || peer >= 8 || r < 0 || r > 1) return;
    xwc[r][peer][st == IBV_WC_RETRY_EXC_ERR ? 0 : st == IBV_WC_RNR_RETRY_EXC_ERR ? 1 : 2]++;
  }
};

Fabric::~Fabric() { stop_proxy(); }

void Fabric::init(const FabricConfig& cfg) {
  cfg_ = cfg; g_rank = cfg.rank;
  impl_ = new Impl();
  Impl& I = *impl_;
  const int nr = (int)cfg.ips.size(); I.nr = nr;
  I.bulk = cfg.bulk_bytes > 0;
  I.var = !I.bulk && !cfg.vcap.empty();
  if (I.bulk) {
    FCHECK(cfg.bulk_flags > 0 && cfg.bulk_bytes % 8 == 0 && nr <= 8);
    cfg_.ncoll = 0; cfg_.slot_bytes = 0;
  } else if (I.var) {
    FCHECK(cfg.vdst_k.size() == cfg.vcap.size() && nr <= 8);
    for (size_t c = 0; c < cfg.vcap.size(); c++) FCHECK(cfg.vcap[c] > 0 && cfg.vcap[c] % 8 == 0 && cfg.vdst_k[c] < nr);
    cfg_.ncoll = (int)cfg.vcap.size(); cfg_.slot_bytes = 0;
    I.L.make(nr, cfg.vcap, cfg.vdst_k);
  }
  FCHECK(cfg_.rank >= 0 && cfg_.rank < nr && nr >= 2 && (I.bulk || (cfg_.ncoll > 0 && (I.var || cfg_.slot_bytes > 0))));
  I.nrails = cfg.ips2.empty() ? 1 : 2;
  if (I.bulk) {
    const size_t fl = (size_t)cfg.bulk_flags * nr * 16, db = (size_t)cfg.bulk_flags * 8;
    I.region_bytes = ((cfg.bulk_bytes + fl + 2 * db + 64) + 4095) & ~size_t(4095);
    CUCHECK(cudaHostAlloc((void**)&I.region, I.region_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(I.region, 0, I.region_bytes);
    char* bdev = nullptr; CUCHECK(cudaHostGetDevicePointer((void**)&bdev, I.region, 0));
    FCHECK(bdev == I.region);
    I.bd.bulk = I.region;
    I.bd.flags = (unsigned long long*)(I.region + cfg.bulk_bytes);
    I.bd.doorbell = (unsigned long long*)(I.region + cfg.bulk_bytes + fl);
    I.bd.aux_out = (unsigned long long*)(I.region + cfg.bulk_bytes + fl + db);
    I.bd.error = (unsigned int*)(I.region + cfg.bulk_bytes + fl + 2 * db);
    I.bd.rank = cfg.rank; I.bd.nr = nr; I.bd.nflags = cfg.bulk_flags;
    dev_ = FabricDev{}; dev_.error = I.bd.error; dev_.rank = cfg.rank; dev_.nr = nr;
    dev_.recv = I.region;                               // the advertised remote address = the region base
    I.batches.assign(nr, {}); I.inflight.assign(nr, 0);
  } else {
  const size_t stride = I.var ? 0 : (cfg.slot_bytes + 8 + 255) & ~size_t(255);
  const size_t send_bytes = I.var ? I.L.send_bytes() : 2 * (size_t)cfg.ncoll * stride;
  const size_t recv_bytes = I.var ? I.L.recv_bytes() : 2 * (size_t)cfg.ncoll * nr * stride;
  const size_t db_bytes = (size_t)cfg_.ncoll * 8;
  I.region_bytes = ((send_bytes + recv_bytes + db_bytes + 64) + 4095) & ~size_t(4095);
  CUCHECK(cudaHostAlloc((void**)&I.region, I.region_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  memset(I.region, 0, I.region_bytes);
  char* base_dev = nullptr; CUCHECK(cudaHostGetDevicePointer((void**)&base_dev, I.region, 0));
  FCHECK(base_dev == I.region);  // unified addressing: host pointer == device pointer (true on GB10 / sm_120 UVA)
  dev_.send = I.region;
  dev_.recv = I.region + send_bytes;
  dev_.doorbell = (unsigned long long*)(I.region + send_bytes + recv_bytes);
  dev_.error = (unsigned int*)(I.region + send_bytes + recv_bytes + db_bytes);
  dev_.rank = cfg.rank; dev_.nr = nr; dev_.ncoll = cfg_.ncoll; dev_.stride = stride;
  if (cfg.xtrace && !I.var && xtrace_env()) {        // stage-0 trace (default off)
    I.xgpu = xt_alloc(cfg_.ncoll); dev_.xt = I.xgpu;
    I.xt = I.xgpu != nullptr;
    I.xp.assign((size_t)XT_STEPS * cfg_.ncoll * 2 * 4, 0); I.xpseq.assign((size_t)XT_STEPS * cfg_.ncoll, 0);
    fprintf(stderr, "[fabric r%d] G6_XTRACE on: %d steps x %d collectives, GPU %s\n", cfg.rank, XT_STEPS, cfg_.ncoll, I.xt ? "ok" : "FAILED");
  }
  }
  I.xenv = xtrace_env();

  if (cfg.ring) {                                     // switchless ring (see RelayHdr above)
    FCHECK(nr == 3 || nr == 4);
    const int me = cfg.rank;
    I.ring = true; I.nxt = (me + 1) % nr; I.prv = (me + nr - 1) % nr; I.dia = nr == 4 ? (me + 2) % nr : -1;
    I.ring_rail[I.nxt] = 0; I.ring_rail[I.prv] = 1; if (I.dia >= 0) I.ring_rail[I.dia] = 2;
    I.nrails = nr == 4 ? 3 : 2;                         // the rails whose CQs the proxy polls
    const std::vector<std::string> dn = cfg.ring_devs.empty()
        ? std::vector<std::string>{"rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"} : cfg.ring_devs;
    FCHECK(dn.size() == 4);
    const int map[4] = {2, 0, 3, 1};                    // ring_devs order -> rails index
    const int nopen = nr == 4 ? 4 : 2;
    for (int k = 0; k < 4; k++) {
      const int r = map[k]; if (r >= nopen) continue;
      Rail& R = I.rails[r]; R.di.dev = dn[k]; R.di.gid_index = ring_gid_index(dn[k]); FCHECK(R.di.gid_index >= 0);
      int n; ibv_device** list = ibv_get_device_list(&n); FCHECK(list);
      for (int i = 0; i < n; i++) if (R.di.dev == ibv_get_device_name(list[i])) R.ctx = ibv_open_device(list[i]);
      ibv_free_device_list(list); FCHECK(R.ctx);
      R.pd = ibv_alloc_pd(R.ctx); FCHECK(R.pd);
      R.cq = ibv_create_cq(R.ctx, 8192, nullptr, nullptr, 0); FCHECK(R.cq);
      FCHECK(ibv_query_gid(R.ctx, 1, R.di.gid_index, &R.gid) == 0);
      R.mr = ibv_reg_mr(R.pd, I.region, I.region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE); FCHECK(R.mr);
    }
    auto mkqp = [&](Rail& R, ibv_cq* rcq, int send_wr, int recv_wr, int sge, int inl, ibv_qp_cap* actual) {
      ibv_qp_init_attr ia{}; ia.send_cq = ia.recv_cq = rcq ? rcq : R.cq; ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = send_wr; ia.cap.max_recv_wr = recv_wr; ia.cap.max_send_sge = sge; ia.cap.max_recv_sge = 1;
      ia.cap.max_inline_data = inl;
      ibv_qp* q = ibv_create_qp(R.pd, &ia); FCHECK(q);
      if (actual) *actual = ia.cap;                         // ibv_create_qp returns the provider's actual capacities
      ibv_qp_attr at{}; at.qp_state = IBV_QPS_INIT; at.port_num = 1; at.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
      FCHECK(ibv_modify_qp(q, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0);
      return q;
    };
    auto info = [&](ibv_qp* q, Rail& R, uint32_t rkey, uint64_t addr, int tag) {
      QPInfo x{q->qp_num, (uint32_t)((me * 131 + tag * 7 + 1) & 0xffffff), rkey, addr, {}}; memcpy(x.gid, R.gid.raw, 16); return x;
    };
    auto connect = [&](ibv_qp* q, Rail& R, const QPInfo& mine, const QPInfo& pi) {
      ibv_qp_attr at{}; at.qp_state = IBV_QPS_RTR; at.path_mtu = IBV_MTU_4096; at.dest_qp_num = pi.qpn; at.rq_psn = pi.psn;
      at.max_dest_rd_atomic = 1; at.min_rnr_timer = 12;
      at.ah_attr.is_global = 1; memcpy(at.ah_attr.grh.dgid.raw, pi.gid, 16); at.ah_attr.grh.sgid_index = R.di.gid_index;
      at.ah_attr.grh.hop_limit = 64; at.ah_attr.port_num = 1;
      FCHECK(ibv_modify_qp(q, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
      ibv_qp_attr as{}; as.qp_state = IBV_QPS_RTS; as.timeout = 14; as.retry_cnt = 7; as.rnr_retry = 7; as.sq_psn = mine.psn; as.max_rd_atomic = 1;
      FCHECK(ibv_modify_qp(q, &as, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
    };
    const uint64_t recv = (uint64_t)dev_.recv;
    const int inl = I.bulk ? 64 : 0;
    I.qp.assign(nr, nullptr); I.unsignaled.assign(nr, 0); I.peer.resize(nr);
    RingInfo mi{};
    ibv_qp* qdn = mkqp(I.rails[0], nullptr, 2048, 1, 1, inl, nullptr);
    ibv_qp* qdp = mkqp(I.rails[1], nullptr, 2048, 1, 1, inl, nullptr);
    mi.dn = info(qdn, I.rails[0], I.rails[0].mr->rkey, recv, 1);
    mi.dp = info(qdp, I.rails[1], I.rails[1].mr->rkey, recv, 2);
    ibv_qp *qor = nullptr, *qri = nullptr, *qfwd = nullptr, *qfin = nullptr;
    RelayCtx& X = I.relay;
    if (I.dia >= 0) {
      CUCHECK(cudaHostAlloc((void**)&X.ring, kRelayRing, cudaHostAllocPortable));
      ibv_mr* rin = ibv_reg_mr(I.rails[3].pd, X.ring, kRelayRing, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE); FCHECK(rin);
      ibv_mr* rout = ibv_reg_mr(I.rails[2].pd, X.ring, kRelayRing, IBV_ACCESS_LOCAL_WRITE); FCHECK(rout);
      X.ring_lkey = rout->lkey;
      CUCHECK(cudaHostAlloc((void**)&I.credit, 64, cudaHostAllocPortable)); memset(I.credit, 0, 64);
      ibv_mr* cmr = ibv_reg_mr(I.rails[2].pd, I.credit, 64, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE); FCHECK(cmr);
      CUCHECK(cudaHostAlloc((void**)&X.rbuf, (size_t)kRelayRecv * kRelayBuf, cudaHostAllocPortable));
      ibv_mr* rmr = ibv_reg_mr(I.rails[3].pd, X.rbuf, (size_t)kRelayRecv * kRelayBuf, IBV_ACCESS_LOCAL_WRITE); FCHECK(rmr);
      X.rbuf_lkey = rmr->lkey;
      X.ri_cq = ibv_create_cq(I.rails[3].ctx, kRelayRecv + (int)kRelayCredit, nullptr, nullptr, 0); FCHECK(X.ri_cq);
      X.fwd_cq = ibv_create_cq(I.rails[2].ctx, 4096, nullptr, nullptr, 0); FCHECK(X.fwd_cq);
      qor = mkqp(I.rails[2], nullptr, 4096, 1, 2, 128, nullptr);
      ibv_qp_cap credit_cap{};
      qri = mkqp(I.rails[3], X.ri_cq, (int)kRelayCredit, kRelayRecv, 1, 64, &credit_cap); // its sends are the credits
      X.credit_sq_capacity = credit_cap.max_send_wr;
      // A provider may grant more than it was asked for, but the CQ only reserves kRelayCredit beyond the receives.
      X.credits.capacity = std::min<unsigned>(kRelayCredit, X.credit_sq_capacity);
      FCHECK(X.credits.capacity > 0);
      { ibv_qp_init_attr ia{}; ia.send_cq = X.fwd_cq; ia.recv_cq = X.fwd_cq; ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = 2048; ia.cap.max_recv_wr = 1; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1; ia.cap.max_inline_data = 64;
        qfwd = ibv_create_qp(I.rails[2].pd, &ia); FCHECK(qfwd);
        ibv_qp_attr at{}; at.qp_state = IBV_QPS_INIT; at.port_num = 1; at.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        FCHECK(ibv_modify_qp(qfwd, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0); }
      qfin = mkqp(I.rails[3], nullptr, 1, 1, 1, 0, nullptr);
      for (int i = 0; i < kRelayRecv; i++) {
        ibv_sge rs{(uint64_t)(X.rbuf + (size_t)i * kRelayBuf), kRelayBuf, X.rbuf_lkey};
        ibv_recv_wr rw{}, *bad = nullptr; rw.wr_id = (uint64_t)i; rw.sg_list = &rs; rw.num_sge = 1;
        FCHECK(ibv_post_recv(qri, &rw, &bad) == 0);
      }
      mi.orq = info(qor, I.rails[2], cmr->rkey, (uint64_t)I.credit, 3);
      mi.ri = info(qri, I.rails[3], rin->rkey, (uint64_t)X.ring, 4);
      mi.fwd = info(qfwd, I.rails[2], 0, 0, 5);
      mi.fin = info(qfin, I.rails[3], I.rails[3].mr->rkey, recv, 6);
      fprintf(stderr, "[fabric r%d] ring: diagonal %d through %d, relay ring %llu MB\n", me, I.dia, I.nxt, (unsigned long long)(kRelayRing >> 20));
    }
    auto all = allgather(me, nr, cfg.ips[0], cfg.port, &mi, sizeof mi);
    const RingInfo* A = (const RingInfo*)all.data();
    connect(qdn, I.rails[0], mi.dn, A[I.nxt].dp); I.qp[I.nxt] = qdn; I.peer[I.nxt] = A[I.nxt].dp;
    connect(qdp, I.rails[1], mi.dp, A[I.prv].dn); I.qp[I.prv] = qdp; I.peer[I.prv] = A[I.prv].dn;
    if (I.dia >= 0) {
      connect(qor, I.rails[2], mi.orq, A[I.nxt].ri); I.qp[I.dia] = qor; I.peer[I.dia] = A[I.nxt].ri;
      connect(qri, I.rails[3], mi.ri, A[I.prv].orq);
      connect(qfwd, I.rails[2], mi.fwd, A[I.nxt].fin);
      connect(qfin, I.rails[3], mi.fin, A[I.prv].fwd);
      X.ri = qri; X.fwd = qfwd; X.fin_addr = A[I.nxt].fin.addr; X.fin_rkey = A[I.nxt].fin.rkey;
      X.cred_addr = A[I.prv].orq.addr; X.cred_rkey = A[I.prv].orq.rkey;
      I.ring_addr = A[I.nxt].ri.addr; I.ring_rkey = A[I.nxt].ri.rkey;
      X.err = I.err = dev_.error; X.rank = me;
      X.ri_dev = I.rails[3].di.dev;
      std::lock_guard<std::mutex> lk(g_relay_mu);
      const int k = g_nrelay.load(); FCHECK(k < 32);
      g_relay[k] = &X; g_nrelay.store(k + 1, std::memory_order_release);
      if (!g_relay_started) { g_relay_started = true; std::thread(relay_loop, cfg.relay_cpu).detach(); }
    }
    barrier();
    return;
  }
  for (int r = 0; r < I.nrails; r++) {
    Rail& R = I.rails[r];
    R.di = find_dev(r == 0 ? cfg.ips[cfg.rank] : cfg.ips2[cfg.rank]);
    int n; ibv_device** list = ibv_get_device_list(&n); FCHECK(list);
    for (int i = 0; i < n; i++) if (R.di.dev == ibv_get_device_name(list[i])) R.ctx = ibv_open_device(list[i]);
    ibv_free_device_list(list); FCHECK(R.ctx);
    R.pd = ibv_alloc_pd(R.ctx); FCHECK(R.pd);
    R.cq = ibv_create_cq(R.ctx, 8192, nullptr, nullptr, 0); FCHECK(R.cq);
    FCHECK(ibv_query_gid(R.ctx, 1, R.di.gid_index, &R.gid) == 0);
    R.mr = ibv_reg_mr(R.pd, I.region, I.region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE); FCHECK(R.mr);
  }
  I.qp.assign(nr, nullptr); I.unsignaled.assign(nr, 0);
  std::vector<QPInfo> mine(nr);
  for (int p = 0; p < nr; p++) {
    if (p == cfg.rank) continue;
    Rail& R = I.rails[I.rail_of(p, cfg.rank)];
    ibv_qp_init_attr ia{}; ia.send_cq = R.cq; ia.recv_cq = R.cq; ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = 2048; ia.cap.max_recv_wr = 1; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    if (I.bulk) ia.cap.max_inline_data = 64;            // bulk: 8-B flag writes go inline
    I.qp[p] = ibv_create_qp(R.pd, &ia); FCHECK(I.qp[p]);
    ibv_qp_attr at{}; at.qp_state = IBV_QPS_INIT; at.port_num = 1; at.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    FCHECK(ibv_modify_qp(I.qp[p], &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0);
    mine[p] = QPInfo{I.qp[p]->qp_num, (uint32_t)((cfg.rank * 131 + p * 7) & 0xffffff), R.mr->rkey, (uint64_t)dev_.recv, {}};
    memcpy(mine[p].gid, R.gid.raw, 16);
  }
  auto all = allgather(cfg.rank, nr, cfg.ips[0], cfg.port, mine.data(), sizeof(QPInfo) * nr);
  I.peer.resize(nr);
  for (int p = 0; p < nr; p++) {
    if (p == cfg.rank) continue;
    QPInfo pi = ((QPInfo*)(all.data() + sizeof(QPInfo) * nr * p))[cfg.rank]; I.peer[p] = pi;
    Rail& R = I.rails[I.rail_of(p, cfg.rank)];
    ibv_qp_attr at{}; at.qp_state = IBV_QPS_RTR; at.path_mtu = IBV_MTU_4096; at.dest_qp_num = pi.qpn; at.rq_psn = pi.psn;
    at.max_dest_rd_atomic = 1; at.min_rnr_timer = 12;
    at.ah_attr.is_global = 1; memcpy(at.ah_attr.grh.dgid.raw, pi.gid, 16); at.ah_attr.grh.sgid_index = R.di.gid_index;
    at.ah_attr.grh.hop_limit = 64; at.ah_attr.port_num = 1;
    FCHECK(ibv_modify_qp(I.qp[p], &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
    ibv_qp_attr as{}; as.qp_state = IBV_QPS_RTS; as.timeout = 14; as.retry_cnt = 7; as.rnr_retry = 7; as.sq_psn = mine[p].psn; as.max_rd_atomic = 1;
    FCHECK(ibv_modify_qp(I.qp[p], &as, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
  }
  barrier();
}

void Fabric::barrier() {
  static int bump = 1;
  char z = 0; allgather(cfg_.rank, nranks(), cfg_.ips[0], cfg_.port + (bump++ % 3) + 1, &z, 1);
}

void Fabric::enqueue_step(unsigned long long step, const std::vector<uint32_t>& sizes) { enqueue_step(step, sizes, Rs{}); }

const VLayout& Fabric::vlayout() const { FCHECK(impl_ && impl_->var); return impl_->L; }

VSlotDev Fabric::vslot(int c) const {
  const VLayout& L = vlayout();
  FCHECK(c >= 0 && c < L.ncoll);
  return L.vslot(c, cfg_.rank, dev_.send, dev_.recv, dev_.doorbell, dev_.error);
}

void Fabric::enqueue_step(unsigned long long step, const std::vector<uint32_t>& sizes, const Rs& rs) {
  FCHECK((int)sizes.size() <= cfg_.ncoll);
  if (impl_->var) {
    FCHECK(!rs.mask && !rs.skip);
    for (size_t c = 0; c < sizes.size(); c++) FCHECK(sizes[c] <= impl_->L.cap[c] && sizes[c] % 8 == 0);
  } else
  for (size_t c = 0; c < sizes.size(); c++) if (!bit(rs.mask, c)) FCHECK(sizes[c] + 8 <= dev_.stride && sizes[c] % 8 == 0);
  if (rs.mask) {
    FCHECK((int)rs.bytes.size() == nranks() && rs.ss % 8 == 0);
    for (uint32_t b : rs.bytes) FCHECK(b % 8 == 0 && b + 8 <= rs.ss && rs.ss * nranks() <= dev_.stride);
  }
  FCHECK((rs.p2p & ~rs.mask) == 0 && (!rs.p2p || (rs.peer >= 0 && rs.peer < nranks() && rs.peer != cfg_.rank)));
  { std::lock_guard<std::mutex> lk(impl_->mu); impl_->q.push_back(Job{step, sizes, rs.mask, rs.ss, rs.bytes, rs.skip, rs.p2p, rs.peer}); impl_->pending++; }
  impl_->cv.notify_one();
}

void Fabric::enqueue_bulk(BulkJob job) {
  Impl& I = *impl_;
  FCHECK(I.bulk && job.db < I.bd.nflags && job.wait_flag < I.bd.nflags && job.post_flag < I.bd.nflags);
  for (const auto& r : job.ranges) FCHECK(r.first % 8 == 0 && r.second % 8 == 0 && r.second > 0 && r.first + r.second <= cfg_.bulk_bytes);
  FCHECK(job.peer == -1 || (job.peer >= 0 && job.peer < I.bd.nr && job.peer != I.bd.rank));
  Job j{}; j.bulk = true; j.b = std::move(job);
  { std::lock_guard<std::mutex> lk(I.mu); I.q.push_back(std::move(j)); I.pending++; }
  I.cv.notify_one();
}

BulkDev Fabric::bulk_dev() const { FCHECK(impl_ && impl_->bulk); return impl_->bd; }

bool Fabric::wait_proxy_idle(double stall_s) {
  Impl& I = *impl_;
  const unsigned e = drain_wait([&] { return I.pending.load(std::memory_order_acquire); },
                                [&] { return I.exited.load(std::memory_order_acquire); },
                                [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); },
                                [] { std::this_thread::yield(); }, stall_s);
  if (!e) return true;
  fprintf(stderr, "[fabric r%d] drain FAILED %08x (%s), pending %ld, error word %08x\n", cfg_.rank, e,
          e == kDrainDead ? "proxy exited with work pending" : "no progress", I.pending.load(), *dev_.error);
  if (e == kDrainStall) __sync_val_compare_and_swap(dev_.error, 0u, e);
  return false;
}

void Fabric::start_proxy() {
  Impl& I = *impl_;
  I.stop = false; I.exited = false; I.proxy_started = false;
  if (cfg_.proxy_cpu >= 0 && I.pinned_cpu < 0) { I.pinned_cpu = cfg_.proxy_cpu; progress_cpu_add(I.pinned_cpu); }
  I.th = std::thread([this]() {
    Impl& I = *impl_;
    struct Exit { Impl& I; ~Exit() { if (int t = I.proxy_tid.exchange(0)) progress_tid_remove(t); I.exited.store(true, std::memory_order_release); } } on_exit{I};
    if (cfg_.proxy_cpu >= 0) {
      cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cfg_.proxy_cpu, &cs);
      const int rc = pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
      if (rc != 0) fprintf(stderr, "[fabric r%d] WARNING: proxy thread not pinned to CPU %d (%s); the progress thread can be starved\n", cfg_.rank, cfg_.proxy_cpu, strerror(rc));
      const int tid = (int)syscall(SYS_gettid);
      I.proxy_tid = tid; progress_tid_add(tid);        // the isolation exempts exactly this thread
    }
    I.proxy_started.store(true, std::memory_order_release);
    const int me = cfg_.rank, nr = I.nr;
    // bulk mode: drain completions until peer p has room for one more WR (signaled every 64 WRs, wr_id = peer);
    // CQ error / bad completion / 60 s without a completion -> error word (bulk_proxy.h)
    auto cqpoll = [&](int r, BulkCompl* out, int max) -> int {
      ibv_wc wc[32]; const int n = ibv_poll_cq(I.rails[r].cq, max < 32 ? max : 32, wc);
      for (int i = 0; i < n; i++) {
        out[i] = BulkCompl{wc[i].wr_id, wc[i].status == IBV_WC_SUCCESS};
        if (wc[i].status != IBV_WC_SUCCESS) { I.wc_count(r, (int)wc[i].wr_id, wc[i].status); fprintf(stderr, "[fabric r%d] bulk WC error %s (peer %d)\n", me, ibv_wc_status_str(wc[i].status), (int)wc[i].wr_id); }
      }
      return n;
    };
    auto bulk_room_t = [&](int p) -> unsigned {
      return g6::bulk_room(p, 1792, I.nrails, I.inflight, I.batches, cqpoll,
        [&] { return I.stop.load(std::memory_order_relaxed) || *(volatile unsigned*)I.bd.error != 0; },
        [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }, 60.0);
    };
    auto bulk_room = [&](int p) -> bool {
      const unsigned e = bulk_room_t(p);
      if (e && e != kBulkStopped) { fprintf(stderr, "[fabric r%d] bulk proxy CQ failure %08x (peer %d, %d WRs in flight)\n", me, e, p, I.inflight[p]); __sync_val_compare_and_swap(I.bd.error, 0u, e); }
      return e == 0;
    };
    auto bulk_post = [&](int p, ibv_send_wr& wr) -> bool {
      if (!bulk_room(p)) return false;
      const int k = I.wrs(p, wr);
      const bool sig = (I.unsignaled[p] += k) >= 64;
      wr.wr_id = (uint64_t)p; wr.send_flags |= sig ? IBV_SEND_SIGNALED : 0;
      I.post(p, wr);
      I.inflight[p] += k;
      if (sig) { I.batches[p].push_back(I.unsignaled[p]); I.unsignaled[p] = 0; }
      return true;
    };
    auto spin = [&](volatile unsigned long long* w, unsigned long long want, unsigned code) -> bool {
      auto t0 = std::chrono::steady_clock::now();
      while (*w < want) {
        if (I.stop.load(std::memory_order_relaxed) || *I.bd.error) return false;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) {
          fprintf(stderr, "[fabric r%d] bulk proxy TIMEOUT code %08x (want %llu have %llu)\n", me, code, want, (unsigned long long)*w);
          *I.bd.error = code; return false;
        }
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      return true;
    };
    auto bulk_job = [&](const BulkJob& b) -> bool {
      {  // non-blocking reap first: a bad completion from an earlier job is caught here, not only under back-pressure
        int got = 0, bad = -1;
        if (const unsigned e = bulk_reap(I.nrails, I.inflight, I.batches, cqpoll, &got, &bad)) {
          fprintf(stderr, "[fabric r%d] bulk proxy CQ failure %08x at job start (peer %d)\n", me, e, bad);
          __sync_val_compare_and_swap(I.bd.error, 0u, e); return false;
        }
      }
      if (b.db >= 0 && !spin(I.bd.doorbell + b.db, b.seq, 0x72000000u | (unsigned)b.db)) return false;
      if (b.wait_flag >= 0)
        for (int p = 0; p < nr; p++)
          if (p != me && bulk_to(b.peer, p) && !spin(I.bd.flags + ((size_t)b.wait_flag * nr + p) * 2, b.wait_min, 0x73000000u | (unsigned)(b.wait_flag << 8) | (unsigned)p)) return false;
      FlagWrite fw[2];
      flag_writes(b.post_val, b.post_flag >= 0 ? *(volatile unsigned long long*)(I.bd.aux_out + b.post_flag) : 0ull, fw);
      for (int k = 1; k < nr; k++) {
        const int p = (me + k) % nr;                    // stagger the peer order across ranks
        if (!bulk_to(b.peer, p)) continue;              // one-peer job (BulkJob::peer)
        Rail& R = I.rails[I.rail_of(p, me)];
        for (const auto& rg : b.ranges) {
          ibv_sge sge{(uint64_t)(I.bd.bulk + rg.first), rg.second, R.mr->lkey};
          ibv_send_wr wr{}; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE;
          wr.wr.rdma.remote_addr = I.peer[p].addr + rg.first; wr.wr.rdma.rkey = I.peer[p].rkey;
          if (!bulk_post(p, wr)) return false;
        }
        if (b.post_flag >= 0)
          for (const FlagWrite& w : fw) {                // aux, then val: two ordered 8-B inline writes (bulk_proxy.h)
            ibv_sge sge{(uint64_t)&w.value, 8, 0};
            ibv_send_wr wr{}; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_INLINE;
            wr.wr.rdma.remote_addr = I.peer[p].addr + (uint64_t)((char*)(I.bd.flags + ((size_t)b.post_flag * nr + me) * 2 + w.word) - I.bd.bulk);
            wr.wr.rdma.rkey = I.peer[p].rkey;
            if (!bulk_post(p, wr)) return false;
          }
      }
      return true;
    };
    while (!I.stop.load(std::memory_order_relaxed)) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(I.mu);
        // busy-ish wait: spin briefly on the queue so the next step is picked up without a futex wake
        if (I.q.empty()) I.cv.wait_for(lk, std::chrono::microseconds(200), [&] { return !I.q.empty() || I.stop.load(); });
        if (I.q.empty()) continue;
        job = std::move(I.q.front()); I.q.pop_front();
      }
      if (job.bulk) { if (!bulk_job(job.b)) return; I.pending.fetch_sub(1, std::memory_order_release); continue; }
      const int par = (int)(job.step & 1);
      for (int c = 0; c < (int)job.sizes.size(); c++) {
        if (bit(job.skip, c)) continue;
        const unsigned long long seq = coll_seq(job.step, c, cfg_.ncoll);
        volatile unsigned long long* db = dev_.doorbell + c;
        auto t0 = std::chrono::steady_clock::now();
        while (*db != seq) {
          if (I.stop.load(std::memory_order_relaxed)) return;
          if (*dev_.error) return;
          if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
            fprintf(stderr, "[fabric r%d] proxy TIMEOUT waiting doorbell step %llu coll %d (have %llu)\n", me, job.step, c, (unsigned long long)*db);
            *dev_.error = 0x70000000u | c; return;
          }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t* xr = nullptr;                         // G6_XTRACE proxy record of (step, c): [rail][4]
        if (I.xt) {
          const size_t k = (size_t)(job.step % XT_STEPS) * cfg_.ncoll + c;
          xr = &I.xp[k * 8]; for (int z = 0; z < 8; z++) xr[z] = 0;
          xr[0] = xr[4] = xt_tick(); I.xpseq[k] = seq;
        }
        if (I.var) {  // variable layout: own slot, all peers (compact slots) or one destination
          const VLayout& L = I.L;
          const uint32_t bytes = job.sizes[c];
          char* src = dev_.send + L.send_at(c, par);
          *(volatile unsigned long long*)(src + bytes) = seq;
          for (int p = 0; p < nr; p++) {
            const int j = L.slot_of(c, p, me);
            if (j < 0) continue;
            Rail& R = I.rails[I.rail_of(p, me)];
            ibv_sge sge{(uint64_t)src, bytes + 8, R.mr->lkey};
            ibv_send_wr wr{}, *bad; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = I.peer[p].addr + L.recv_at(c, par, j);
            wr.wr.rdma.rkey = I.peer[p].rkey;
            bool sig = (I.unsignaled[p] += I.wrs(p, wr)) >= 256; if (sig) I.unsignaled[p] = 0;
            wr.send_flags = sig ? IBV_SEND_SIGNALED : 0;
            (void)bad; I.post(p, wr);
          }
        } else {
        const bool rs = bit(job.rs_mask, c);
        const uint32_t bytes = job.sizes[c];
        char* src = dev_.send + ((size_t)par * cfg_.ncoll + c) * dev_.stride;
        if (!rs) *(volatile unsigned long long*)(src + bytes) = seq;  // tail travels with the payload in one write
        if (xr) xr[1] = xr[5] = xt_tick();
        for (int p = 0; p < nr; p++) {
          if (p == me || (rs && bit(job.p2p, c) && p != job.peer)) continue;   // p2p: one peer only
          Rail& R = I.rails[I.rail_of(p, me)];
          char* sp = rs ? src + p * job.rs_ss : src;             // reduce-scatter: peer p gets slice p only
          const uint32_t nb = rs ? job.rs_bytes[p] : bytes;
          if (rs) *(volatile unsigned long long*)(sp + nb) = seq;
          ibv_sge sge{(uint64_t)sp, nb + 8, R.mr->lkey};
          ibv_send_wr wr{}, *bad; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE;
          wr.wr.rdma.remote_addr = I.peer[p].addr + (((size_t)par * cfg_.ncoll + c) * nr + me) * dev_.stride;
          wr.wr.rdma.rkey = I.peer[p].rkey;
          bool sig = (I.unsignaled[p] += I.wrs(p, wr)) >= 256; if (sig) I.unsignaled[p] = 0;
          wr.send_flags = sig ? IBV_SEND_SIGNALED : 0;
          wr.wr_id = (uint64_t)p;                       // peer, for the per-QP WC error counters
          const int xrl = I.rail_of(p, me) > 1 ? 1 : I.rail_of(p, me);   // trace keeps two rails; ring mode folds the relay rail into 1
          if (xr && !xr[xrl * 4 + 2]) xr[xrl * 4 + 2] = xt_tick();
          (void)bad; I.post(p, wr);
          if (xr) xr[xrl * 4 + 3] = xt_tick();
        }
        }
        for (int r = 0; r < I.nrails; r++) {
          ibv_wc wc[32]; int n;
          while ((n = ibv_poll_cq(I.rails[r].cq, 32, wc)) > 0)
            for (int i = 0; i < n; i++) if (wc[i].status != IBV_WC_SUCCESS) { I.wc_count(r, (int)wc[i].wr_id, wc[i].status); fprintf(stderr, "[fabric r%d] WC error %s\n", me, ibv_wc_status_str(wc[i].status)); *dev_.error = 0x71000000u; return; }
          if (n < 0) { fprintf(stderr, "[fabric r%d] ibv_poll_cq failed (%d)\n", me, n); *dev_.error = kBulkCqError; return; }   // fail closed, as the bulk path
        }
      }
      I.pending.fetch_sub(1, std::memory_order_release);
    }
  });
  if (I.xt && getenv("G6_XTRACE_QUIT") && !I.xquit.joinable())   // dump once when the quit file appears (outside the timed loop)
    I.xquit = std::thread([this]() {
      const std::string qf = getenv("G6_XTRACE_QUIT");
      while (!impl_->stop.load()) {
        if (access(qf.c_str(), F_OK) == 0) { xtrace_dump("quit-file"); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    });
  // the isolation needs the proxy's tid: wait until the thread has pinned itself and registered
  for (int i = 0; i < 5000 && !I.proxy_started.load(std::memory_order_acquire); i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!I.proxy_started.load(std::memory_order_acquire)) fprintf(stderr, "[fabric r%d] WARNING: proxy thread did not start within 5 s\n", cfg_.rank);
}

void Fabric::stop_proxy() {
  if (!impl_) return;
  impl_->stop = true; impl_->cv.notify_all();
  if (impl_->th.joinable()) impl_->th.join();
  if (impl_->xquit.joinable()) impl_->xquit.join();
  if (impl_->xenv) xtrace_dump("stop");
  if (impl_->pinned_cpu >= 0) { progress_cpu_remove(impl_->pinned_cpu); impl_->pinned_cpu = -1; }
}

// G6_XTRACE dump: $G6_XTRACE_OUT (default /tmp) / g6xtrace.r<rank>.f<n>.txt, n = fabric instance number on this rank.
// Lines: "# header", "G ..." (xtrace.h), "P step c rail dbseen tailst first last | tail-db first-db last-db (ns)",
// "W rail peer retry_exc rnr_exc other" (nonzero rows only; a WC error stops the proxy, so expect 0 or 1 per run).
void Fabric::xtrace_dump(const char* why) {
  static std::atomic<int> inst{0};
  static std::mutex dmu;
  std::lock_guard<std::mutex> lk(dmu);
  Impl& I = *impl_;
  if (!I.xenv || I.xdumped) return;
  I.xdumped = true;
  unsigned long long nw = 0;
  for (auto& a : I.xwc) for (auto& b : a) for (auto v : b) nw += v;
  if (!I.xt && !nw) return;                          // fabrics without stamps write a file only when a QP error was counted
  const char* d = getenv("G6_XTRACE_OUT");
  char path[512]; snprintf(path, sizeof path, "%s/g6xtrace.r%d.f%d.txt", d ? d : "/tmp", cfg_.rank, inst++);
  FILE* o = fopen(path, "w");
  if (!o) { fprintf(stderr, "[fabric r%d] G6_XTRACE: cannot open %s\n", cfg_.rank, path); return; }
  const uint64_t fq = xt_freq();
  fprintf(o, "# g6xtrace v1 why=%s rank=%d nr=%d ncoll=%d nrails=%d bulk=%d steps=%d cntfrq=%llu stamps=%d\n", why, cfg_.rank, I.nr,
          cfg_.ncoll, I.nrails, (int)I.bulk, XT_STEPS, (unsigned long long)fq, (int)I.xt);
  fprintf(o, "# W rail peer retry_exc rnr_exc other\n");
  for (int r = 0; r < 2; r++) for (int p = 0; p < 8; p++)
    if (I.xwc[r][p][0] | I.xwc[r][p][1] | I.xwc[r][p][2]) fprintf(o, "W %d %d %llu %llu %llu\n", r, p, I.xwc[r][p][0], I.xwc[r][p][1], I.xwc[r][p][2]);
  if (I.xt) {
    auto ns = [&](uint64_t a, uint64_t b) -> long long { return (a && b && fq) ? (long long)((double)(int64_t)(a - b) * 1e9 / (double)fq) : 0LL; };
    fprintf(o, "# P step c rail dbseen tailst first last | tail-db first-db last-db (ns; raw = proxy ticks)\n");
    for (size_t k = 0; k < I.xpseq.size(); k++) {
      if (!I.xpseq[k]) continue;
      const unsigned long long step = (I.xpseq[k] - 1) / cfg_.ncoll; const int c = (int)((I.xpseq[k] - 1) % cfg_.ncoll);
      for (int r = 0; r < I.nrails; r++) {
        const uint64_t* x = &I.xp[k * 8 + r * 4];
        if (!x[2]) continue;                         // no post on this rail
        fprintf(o, "P %llu %d %d %llu %llu %llu %llu | %lld %lld %lld\n", step, c, r, (unsigned long long)x[0], (unsigned long long)x[1],
                (unsigned long long)x[2], (unsigned long long)x[3], ns(x[1], x[0]), ns(x[2], x[0]), ns(x[3], x[0]));
      }
    }
    xt_dump_gpu(o, I.xgpu, cfg_.ncoll, I.nr, cfg_.rank);
  }
  fclose(o);
  fprintf(stderr, "[fabric r%d] G6_XTRACE dump (%s) -> %s\n", cfg_.rank, why, path);
}

}  // namespace g6
