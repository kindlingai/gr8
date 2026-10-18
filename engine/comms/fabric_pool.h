// gr8 fabric pool (#154): connections to every box of a deployment at boot, TP groups at runtime.
//
// The pool is every box that may join a layout of this deployment, in a fixed order (pool index = position in
// PoolConfig::ips). Boot connects it once: every box opens one TCP control link to the pool head (index 0), which
// checks that all of them were started with the same pool. The links stay open for the life of the engine, so a box
// outside the current group stays connected and idle (or loads pieces for the next layout, #156).
//
// A group is an ordered list of pool indices: its first member is rank 0 of the group. The head announces a group over
// the pool links (FabricPool::announce); every box, member or not, gets the same Group and acknowledges it before
// announce returns (FabricPool::follow on the others). A member then builds its fabrics from the group
// (group_fabric(): rank = its position, ips/ips2 = the members' addresses in group order, the group's own port range)
// with Fabric::init, exactly as today's fabric with the same members is built. Collectives, regions, doorbells and the
// S2 control links are per group. To drop a group, every member runs Fabric::barrier() on one of its fabrics (no peer
// still writes) and then Fabric::close() on each, which frees every QP, MR, CQ, PD, device context and pinned region.
//
// Mesh only: a cabled ring (FabricConfig::ring) cannot regroup, and group_fabric() refuses it.
// Any exception from init, announce or follow leaves the pool broken (some boxes may have taken a group the others did
// not): up() turns false and every later call throws. As with a fabric error, the engine then stops.
// The links keep TCP keepalive on (10 s idle, 5 s apart, 4 probes, as the S2 control links), so a dead box is noticed.
// Two groups may exist at once (the current layout and the one being built for a switch): group ids alternate between
// two port ranges, so the next group never collides with the current one's bootstrap ports.
//
// No CUDA and no verbs here: the host tests run it over loopback (scripts/host_tests.sh, tests/fabric_pool_test.cpp).
#pragma once
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fabric.h"

namespace g6 {

struct PoolConfig {
  int index = -1;                     // this box's pool index (position in ips)
  std::vector<std::string> ips;       // rail 0 address of every pool box; ips[0] is the pool head
  std::vector<std::string> ips2;      // rail 1 addresses (empty = single rail; else one per pool box)
  int port = 18660;                   // pool control port on the head
  int group_port[2] = {18600, 18700}; // fabric bootstrap base of even / odd group ids (a group's fabrics use base..base+59)
  int s2_port[2] = {7341, 7342};      // S2 control port of even / odd group ids, on the group's rank 0
  unsigned timeout_s = 120;           // boot: how long the head waits for every box, and a box for the head
};

struct Group {
  uint64_t id = 0;                    // announce order, from 1
  std::vector<int> members;           // pool indices; members[r] is rank r of the group
  int rank = -1;                      // this box's rank in the group, -1 = idle (outside the group)
  int fabric_port = 0;                // FabricConfig::port of the group's first fabric (the engine adds its offsets)
  int s2_port = 0;                    // S2 control port of the group (BootstrapConfig::port)
  std::string s2_head_ip;             // S2 head of the group (BootstrapConfig::head_ip): its rank 0
  bool member() const { return rank >= 0; }
  int size() const { return (int)members.size(); }
};

namespace pool_detail {
inline void fail(const std::string& what) { throw std::runtime_error("fabric pool: " + what); }
inline void need(bool ok, const std::string& what) { if (!ok) fail(what); }
using Clock = std::chrono::steady_clock;
inline int left_ms(Clock::time_point end) {
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
  return ms < 0 ? 0 : ms > 1000000000 ? 1000000000 : (int)ms;
}
inline sockaddr_in addr(const std::string& ip, int port) {
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
  need(port > 0 && port < 65536 && inet_pton(AF_INET, ip.c_str(), &a.sin_addr) == 1, "bad address or port");
  return a;
}
// one frame: u32 length (network order), then the bytes
inline void put(int fd, const std::string& s) {
  need(s.size() < (1u << 20), "frame too long");
  const uint32_t n = htonl((uint32_t)s.size());
  std::string b((const char*)&n, 4); b += s;
  for (size_t at = 0; at < b.size();) {
    const ssize_t k = ::send(fd, b.data() + at, b.size() - at, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    need(k > 0, "control link send failed (" + std::string(strerror(errno)) + ")");
    at += (size_t)k;
  }
}
inline void get_n(int fd, char* p, size_t n, int timeout_ms) {
  const auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (n) {
    if (timeout_ms >= 0) {
      pollfd q{fd, POLLIN, 0};
      const int r = ::poll(&q, 1, left_ms(end));
      if (r < 0 && errno == EINTR) continue;
      need(r > 0, "control link timed out");
    }
    const ssize_t k = ::recv(fd, p, n, 0);
    if (k < 0 && errno == EINTR) continue;
    need(k > 0, "control link closed");
    p += k; n -= (size_t)k;
  }
}
inline std::string get(int fd, int timeout_ms = -1) {
  uint32_t n; get_n(fd, (char*)&n, 4, timeout_ms); n = ntohl(n);
  need(n < (1u << 20), "frame too long");
  std::string s(n, '\0'); if (n) get_n(fd, &s[0], n, timeout_ms);
  return s;
}
inline std::string join(const std::vector<std::string>& v) { std::string s; for (auto& x : v) s += x + ","; return s; }
inline void keepalive(int fd) {
  int one = 1, idle = 10, intvl = 5, cnt = 4;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
}
// connect with a deadline (a filtered port would otherwise block for the kernel's SYN retries); -1 = not yet
inline int connect_by(const std::string& ip, int port, Clock::time_point end) {
  sockaddr_in a = addr(ip, port);
  int s = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0); need(s >= 0, "socket");
  int rc = ::connect(s, (sockaddr*)&a, sizeof a), err = 0;
  if (rc != 0 && errno == EINPROGRESS) {
    pollfd q{s, POLLOUT, 0};
    const int w = ::poll(&q, 1, left_ms(end) < 1000 ? left_ms(end) : 1000);
    socklen_t n = sizeof err;
    if (w > 0 && getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &n) == 0 && err == 0) rc = 0;
  }
  if (rc != 0) { ::close(s); return -1; }
  const int fl = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, fl & ~O_NONBLOCK);
  return s;
}
// a list of non-negative integers separated by commas, nothing else
inline bool parse_ints(const std::string& s, std::vector<int>* out) {
  out->clear();
  for (size_t a = 0; a <= s.size();) {
    size_t e = s.find(',', a); if (e == std::string::npos) e = s.size();
    if (e == a || e - a > 6) return false;
    for (size_t k = a; k < e; k++) if (s[k] < '0' || s[k] > '9') return false;
    out->push_back(atoi(s.substr(a, e - a).c_str()));
    a = e + 1;
  }
  return true;
}
inline std::string join(const std::vector<int>& v) { std::string s; for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x); return s; }
}  // namespace pool_detail

// The pool's rule for a group: 2..8 members (the fabric's limit) and at most the pool, each a pool index, none twice.
// Which widths the engine serves (S2 takes 3, 4 and 6 today) is the caller's check (#156).
inline std::string group_refusal(const std::vector<int>& members, int pool_size) {
  if (members.size() < 2) return "a group needs at least two members";
  if (members.size() > 8) return "a group has more than 8 members (the fabric's limit)";
  if ((int)members.size() > pool_size) return "a group has more members than the pool";
  std::vector<char> seen(pool_size, 0);
  for (int m : members) {
    if (m < 0 || m >= pool_size) return "group member " + std::to_string(m) + " is not in the pool";
    if (seen[m]++) return "group member " + std::to_string(m) + " appears twice";
  }
  return "";
}

// A member's fabric config in group g: base (slot sizes, layout, proxy CPU, ...) with the group's rank, addresses and
// ports. With every pool box in pool order this is the FabricConfig a fixed fabric of the same boxes gets today, so its
// collectives are the same bytes (the only differences are the bootstrap ports).
inline FabricConfig group_fabric(const PoolConfig& pc, const Group& g, const FabricConfig& base) {
  using namespace pool_detail;
  need(g.member(), "group_fabric on a box outside the group");
  need(!base.ring, "a ring fabric (--shape ring) cannot regroup: groups are mesh only");
  FabricConfig c = base;
  c.rank = g.rank;
  c.ips.clear(); c.ips2.clear();
  for (int m : g.members) {
    c.ips.push_back(pc.ips[m]);
    if (!pc.ips2.empty()) c.ips2.push_back(pc.ips2[m]);
  }
  c.port = g.fabric_port;
  return c;
}

class FabricPool {
 public:
  FabricPool() = default;
  FabricPool(const FabricPool&) = delete;
  FabricPool& operator=(const FabricPool&) = delete;
  ~FabricPool() { close(); }

  // Boot: the head accepts one link from every other pool box and checks that it has the same pool; every box returns
  // once all of them are connected. Throws on a mismatch, a duplicate index or a timeout.
  void init(const PoolConfig& pc) {
    using namespace pool_detail;
    const int n = (int)pc.ips.size();
    need(n >= 2 && pc.index >= 0 && pc.index < n, "pool index out of range");
    need(pc.ips2.empty() || pc.ips2.size() == pc.ips.size(), "ips2 must name every pool box");
    for (int k = 0; k < 2; k++)
      need(pc.group_port[k] > 0 && pc.group_port[k] + 60 < 65536 && pc.s2_port[k] > 0 && pc.s2_port[k] < 65536, "bad group port");
    need(pc.group_port[0] + 60 <= pc.group_port[1] || pc.group_port[1] + 60 <= pc.group_port[0], "the two group port ranges overlap");
    for (int k = 0; k < 2; k++)
      need(pc.port < pc.group_port[k] || pc.port > pc.group_port[k] + 59, "the pool port is inside a group port range");
    pc_ = pc;
    const std::string agree = agreement();
    const auto end = Clock::now() + std::chrono::seconds(pc.timeout_s);
    if (pc.index == 0) {
      int ls = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0); need(ls >= 0, "socket");
      int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
      sockaddr_in a = addr(pc.ips[0], pc.port);
      if (::bind(ls, (sockaddr*)&a, sizeof a) != 0 || ::listen(ls, 64) != 0) { ::close(ls); fail("cannot listen on the pool port"); }
      links_.assign(n, -1);
      try {
        for (int got = 1; got < n;) {
          pollfd q{ls, POLLIN, 0};
          const int r = ::poll(&q, 1, left_ms(end));
          if (r < 0 && errno == EINTR) continue;
          need(r > 0, "timed out waiting for every pool box (" + std::to_string(got - 1) + " of " + std::to_string(n - 1) + " connected)");
          int c = ::accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
          if (c < 0) continue;
          keepalive(c);
          std::string hello;
          try { hello = get(c, std::min(left_ms(end), 10000)); }   // a box sends its hello at once
          catch (const std::exception& e) { ::close(c); fprintf(stderr, "fabric pool: dropped a connection without a hello (%s)\n", e.what()); continue; }
          const size_t eol = hello.find('\n');
          int idx = -1;
          if (hello.compare(0, 7, "G6POOL ") == 0 && eol != std::string::npos) idx = atoi(hello.substr(7, eol - 7).c_str());
          if (idx < 1 || idx >= n || links_[idx] >= 0 || hello.substr(eol + 1) != agree) {
            const std::string why = idx < 1 || idx >= n ? "bad pool index" : links_[idx] >= 0 ? "pool index connected twice" : "pool config mismatch";
            try { put(c, "G6POOL REFUSED " + why); } catch (...) {}
            ::close(c);
            fail(why + " (from pool box " + std::to_string(idx) + ")");
          }
          links_[idx] = c; ++got;
        }
        for (int i = 1; i < n; i++) put(links_[i], "G6POOL UP\n" + agree);
      } catch (...) { ::close(ls); close(); throw; }
      ::close(ls);
    } else {
      int s = -1;
      for (;;) {
        s = connect_by(pc.ips[0], pc.port, end);
        if (s >= 0) break;
        need(Clock::now() < end, "timed out connecting to the pool head");
        usleep(100000);
      }
      keepalive(s);
      links_.assign(1, s);
      try {
        put(s, "G6POOL " + std::to_string(pc.index) + "\n" + agree);
        const std::string up = get(s, left_ms(end));
        need(up == "G6POOL UP\n" + agree, up.compare(0, 15, "G6POOL REFUSED ") == 0 ? "the head refused this box: " + up.substr(15)
                                         : up == "G6POOL QUIT" ? "the pool head closed the pool" : "bad pool reply");
      } catch (...) { close(); throw; }
    }
  }

  int size() const { return (int)pc_.ips.size(); }
  int index() const { return pc_.index; }
  bool head() const { return pc_.index == 0; }
  const PoolConfig& config() const { return pc_; }
  bool up() const { return !links_.empty() && !broken_; }

  // Head: send group `members` to every pool box and wait until each has acknowledged it; returns the head's view.
  // Members are pool indices in rank order (the head need not be one of them).
  Group announce(const std::vector<int>& members) {
    using namespace pool_detail;
    need(up(), "the pool is not up (closed, or broken by an earlier error)");
    need(head(), "announce is for the pool head");
    const std::string why = group_refusal(members, size());
    need(why.empty(), why);                           // a refused group leaves the pool as it was
    broken_ = true;                                   // until every box has acknowledged
    const uint64_t id = ++last_;
    const std::string f = "G6GROUP " + std::to_string(id) + " " + join(members);
    for (int i = 1; i < size(); i++) put(links_[i], f);
    const auto end = Clock::now() + std::chrono::seconds(pc_.timeout_s);   // one deadline for all the acknowledgements
    for (int i = 1; i < size(); i++)
      need(get(links_[i], left_ms(end)) == "G6GROUP OK " + std::to_string(id), "pool box " + std::to_string(i) + " did not take group " + std::to_string(id));
    broken_ = false;
    return make(id, members);
  }

  // Every other box: wait for the head's next group (timeout_ms < 0: no limit) and acknowledge it. Nothing arriving
  // within timeout_ms throws and leaves the pool usable; any other failure breaks it.
  Group follow(int timeout_ms = -1) {
    using namespace pool_detail;
    need(up(), "the pool is not up (closed, or broken by an earlier error)");
    need(!head(), "follow is for a box other than the pool head");
    for (;;) {                                        // wait for the frame's first byte: a timeout here leaves the pool as it was
      pollfd q{links_[0], POLLIN, 0};
      const int w = ::poll(&q, 1, timeout_ms);
      if (w < 0 && errno == EINTR) continue;
      need(w != 0, "no group from the pool head within the timeout");
      need(w > 0, "poll failed");
      break;
    }
    broken_ = true;                                   // until the group is acknowledged
    const std::string f = get(links_[0], (int)pc_.timeout_s * 1000);   // the rest of a frame follows at once
    need(f.compare(0, 8, "G6GROUP ") == 0, f == "G6POOL QUIT" ? "the pool head closed the pool" : "bad group frame");
    const size_t sp = f.find(' ', 8);
    need(sp != std::string::npos && sp > 8 && f.find_first_not_of("0123456789", 8) == sp, "bad group frame");
    const uint64_t id = strtoull(f.substr(8, sp - 8).c_str(), nullptr, 10);
    std::vector<int> members;
    need(parse_ints(f.substr(sp + 1), &members), "bad group frame");
    const std::string why = group_refusal(members, size());
    need(why.empty() && id == last_ + 1, why.empty() ? "group id out of order" : why);
    last_ = id;
    put(links_[0], "G6GROUP OK " + std::to_string(id));
    broken_ = false;
    return make(id, members);
  }

  // Head: tell every box the pool is closing (follow() then throws), then drop the links. Others: drop the link.
  void close() {
    if (links_.empty()) return;
    if (head()) for (size_t i = 1; i < links_.size(); i++) if (links_[i] >= 0) { try { pool_detail::put(links_[i], "G6POOL QUIT"); } catch (...) {} }
    for (int fd : links_) if (fd >= 0) ::close(fd);
    links_.clear();
  }

 private:
  std::string agreement() const {
    return pool_detail::join(pc_.ips) + "|" + pool_detail::join(pc_.ips2) + "|" + std::to_string(pc_.port) + "|" +
           std::to_string(pc_.group_port[0]) + "," + std::to_string(pc_.group_port[1]) + "|" +
           std::to_string(pc_.s2_port[0]) + "," + std::to_string(pc_.s2_port[1]);
  }
  Group make(uint64_t id, const std::vector<int>& members) const {
    Group g; g.id = id; g.members = members;
    for (int r = 0; r < (int)members.size(); r++) if (members[r] == pc_.index) g.rank = r;
    g.fabric_port = pc_.group_port[id & 1];
    g.s2_port = pc_.s2_port[id & 1];
    g.s2_head_ip = pc_.ips[members[0]];
    return g;
  }
  PoolConfig pc_;
  std::vector<int> links_;            // head: [pool index] -> link (index 0 unused); others: [0] = link to the head
  uint64_t last_ = 0;
  bool broken_ = false;               // an announce or follow failed part way (or one is in progress)
};

}  // namespace g6
