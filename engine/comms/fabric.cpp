// g6 fabric implementation — see fabric.h and RUNTIME.md §3.
#include "fabric.h"
#include "bulk_proxy.h"

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
#include <unistd.h>

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

struct QPInfo { uint32_t qpn, psn, rkey; uint64_t addr; uint8_t gid[16]; };
struct Rail { DevInfo di; ibv_context* ctx = nullptr; ibv_pd* pd = nullptr; ibv_cq* cq = nullptr; ibv_mr* mr = nullptr; ibv_gid gid{}; };
struct Job { unsigned long long step; std::vector<uint32_t> sizes; unsigned rs_mask = 0; size_t rs_ss = 0; std::vector<uint32_t> rs_bytes; unsigned skip = 0;
             unsigned p2p = 0; int peer = -1; bool bulk = false; Fabric::BulkJob b; };
}  // namespace

struct Fabric::Impl {
  int nr = 0, nrails = 1;
  Rail rails[2];
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
  int rail_of(int p, int me) const { return nrails == 1 ? 0 : (p + me) % 2; }
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
  I.stop = false; I.exited = false;
  I.th = std::thread([this]() {
    Impl& I = *impl_;
    struct Exit { std::atomic<bool>& f; ~Exit() { f.store(true, std::memory_order_release); } } on_exit{I.exited};
    if (cfg_.proxy_cpu >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cfg_.proxy_cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
    const int me = cfg_.rank, nr = I.nr;
    // bulk mode: drain completions until peer p has room for one more WR (signaled every 64 WRs, wr_id = peer);
    // CQ error / bad completion / 60 s without a completion -> error word (bulk_proxy.h)
    auto cqpoll = [&](int r, BulkCompl* out, int max) -> int {
      ibv_wc wc[32]; const int n = ibv_poll_cq(I.rails[r].cq, max < 32 ? max : 32, wc);
      for (int i = 0; i < n; i++) {
        out[i] = BulkCompl{wc[i].wr_id, wc[i].status == IBV_WC_SUCCESS};
        if (wc[i].status != IBV_WC_SUCCESS) fprintf(stderr, "[fabric r%d] bulk WC error %s (peer %d)\n", me, ibv_wc_status_str(wc[i].status), (int)wc[i].wr_id);
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
      const bool sig = ++I.unsignaled[p] >= 64;
      wr.wr_id = (uint64_t)p; wr.send_flags |= sig ? IBV_SEND_SIGNALED : 0;
      ibv_send_wr* bad = nullptr;
      FCHECK(ibv_post_send(I.qp[p], &wr, &bad) == 0);
      ++I.inflight[p];
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
            bool sig = ++I.unsignaled[p] >= 256; if (sig) I.unsignaled[p] = 0;
            wr.send_flags = sig ? IBV_SEND_SIGNALED : 0;
            FCHECK(ibv_post_send(I.qp[p], &wr, &bad) == 0);
          }
        } else {
        const bool rs = bit(job.rs_mask, c);
        const uint32_t bytes = job.sizes[c];
        char* src = dev_.send + ((size_t)par * cfg_.ncoll + c) * dev_.stride;
        if (!rs) *(volatile unsigned long long*)(src + bytes) = seq;  // tail travels with the payload in one write
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
          bool sig = ++I.unsignaled[p] >= 256; if (sig) I.unsignaled[p] = 0;
          wr.send_flags = sig ? IBV_SEND_SIGNALED : 0;
          FCHECK(ibv_post_send(I.qp[p], &wr, &bad) == 0);
        }
        }
        for (int r = 0; r < I.nrails; r++) {
          ibv_wc wc[32]; int n;
          while ((n = ibv_poll_cq(I.rails[r].cq, 32, wc)) > 0)
            for (int i = 0; i < n; i++) if (wc[i].status != IBV_WC_SUCCESS) { fprintf(stderr, "[fabric r%d] WC error %s\n", me, ibv_wc_status_str(wc[i].status)); *dev_.error = 0x71000000u; return; }
          if (n < 0) { fprintf(stderr, "[fabric r%d] ibv_poll_cq failed (%d)\n", me, n); *dev_.error = kBulkCqError; return; }   // fail closed, as the bulk path
        }
      }
      I.pending.fetch_sub(1, std::memory_order_release);
    }
  });
}

void Fabric::stop_proxy() {
  if (!impl_) return;
  impl_->stop = true; impl_->cv.notify_all();
  if (impl_->th.joinable()) impl_->th.join();
}

}  // namespace g6
