// fabric_pool.h over loopback, one thread per pool box: boot, groups announced and taken by every box, the group's
// ranks, ports and S2 head, group_fabric() against today's fixed-fabric config, the refusals, a pool mismatch, a few
// thousand groups without a leaked descriptor, and close.
//   g++ -std=c++17 -O2 -Wall -pthread -I engine/comms engine/comms/tests/fabric_pool_test.cpp -o fabric_pool_test && ./fabric_pool_test
#include "fabric_pool.h"

#include <dirent.h>

#include <atomic>
#include <cstdio>
#include <functional>
#include <thread>

using namespace g6;

static std::atomic<int> fails{0};
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)

static int open_fds() {
  int n = 0; DIR* d = opendir("/proc/self/fd"); if (!d) return -1;
  while (auto* e = readdir(d)) if (e->d_name[0] != '.') n++;
  closedir(d); return n;
}

static PoolConfig pool(int n, int index, int port) {
  PoolConfig pc; pc.index = index; pc.port = port; pc.timeout_s = 20;
  pc.group_port[0] = port + 100; pc.group_port[1] = port + 200; pc.s2_port[0] = port + 300; pc.s2_port[1] = port + 301;
  for (int i = 0; i < n; i++) { pc.ips.push_back("127.0.0.1"); pc.ips2.push_back("127.0.1." + std::to_string(i + 1)); }
  return pc;
}

// run fn(box) on n threads, one per pool box
static void boxes(int n, const std::function<void(int)>& fn) {
  std::vector<std::thread> t;
  for (int i = 0; i < n; i++) t.emplace_back([&, i] { try { fn(i); } catch (const std::exception& e) { fprintf(stderr, "FAIL box %d: %s\n", i, e.what()); fails++; } });
  for (auto& x : t) x.join();
}

int main() {
  const int base_port = 21000 + (int)(getpid() % 2000) * 2;
  // 1. boot a pool of six, then groups: every box sees the same group, its own rank or idle, and the group's ports
  {
    const int n = 6;
    const std::vector<std::vector<int>> layouts = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3}, {2, 3, 4}, {5, 4, 3, 2}, {1, 3}, {0, 1, 2, 3, 4, 5}};
    std::vector<std::vector<Group>> seen(n);
    boxes(n, [&](int i) {
      FabricPool p; p.init(pool(n, i, base_port));
      CHECK(p.up() && p.size() == n && p.index() == i);
      for (const auto& L : layouts) seen[i].push_back(i == 0 ? p.announce(L) : p.follow(20000));
      p.close();
    });
    for (int i = 0; i < n; i++) {
      CHECK(seen[i].size() == layouts.size());
      for (size_t k = 0; k < seen[i].size() && k < layouts.size(); k++) {
        const Group& g = seen[i][k];
        CHECK(g.id == k + 1 && g.members == layouts[k]);
        int want = -1; for (int r = 0; r < (int)layouts[k].size(); r++) if (layouts[k][r] == i) want = r;
        CHECK(g.rank == want && g.member() == (want >= 0));
        CHECK(g.fabric_port == base_port + ((k + 1) & 1 ? 200 : 100));           // ids alternate the two port ranges
        CHECK(g.s2_port == base_port + ((k + 1) & 1 ? 301 : 300));
        CHECK(g.s2_head_ip == "127.0.0.1");
      }
    }
  }
  // 2. group_fabric: with the whole pool in pool order it is today's config of the same boxes (rank, ips, ips2, layout),
  //    so the fabric it builds is the same one. A subset gets the members' addresses in group order.
  {
    PoolConfig pc = pool(4, 2, base_port);
    FabricConfig base; base.slot_bytes = 49152; base.ncoll = 157; base.proxy_cpu = 19; base.vcap = {64, 128}; base.vdst_k = {0, 1};
    FabricConfig today = base; today.rank = 2; today.ips = pc.ips; today.ips2 = pc.ips2; today.port = pc.group_port[1];
    Group g; g.id = 1; g.members = {0, 1, 2, 3}; g.rank = 2; g.fabric_port = pc.group_port[1];
    const FabricConfig c = group_fabric(pc, g, base);
    CHECK(c.rank == today.rank && c.ips == today.ips && c.ips2 == today.ips2 && c.port == today.port);
    CHECK(c.slot_bytes == today.slot_bytes && c.ncoll == today.ncoll && c.proxy_cpu == today.proxy_cpu && c.vcap == today.vcap && c.vdst_k == today.vdst_k);
    Group s; s.members = {3, 2}; s.rank = 1; s.fabric_port = pc.group_port[0];
    const FabricConfig d = group_fabric(pc, s, base);
    CHECK(d.rank == 1 && d.ips.size() == 2 && d.ips2.size() == 2 && d.ips2[0] == pc.ips2[3] && d.ips2[1] == pc.ips2[2] && d.port == pc.group_port[0]);
    bool threw = false;
    FabricConfig ring = base; ring.ring = true;
    try { group_fabric(pc, g, ring); } catch (const std::exception&) { threw = true; }
    CHECK(threw);                                       // mesh only
    threw = false; Group idle; idle.members = {0, 1}; idle.rank = -1;
    try { group_fabric(pc, idle, base); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
  }
  // 3. refusals: too small, too big, outside the pool, a member twice
  CHECK(!group_refusal({0}, 4).empty());
  CHECK(!group_refusal({0, 1, 2, 3, 0}, 4).empty());
  CHECK(!group_refusal({0, 4}, 4).empty());
  CHECK(!group_refusal({-1, 1}, 4).empty());
  CHECK(!group_refusal({1, 2, 1}, 4).empty());
  CHECK(group_refusal({3, 1, 0}, 4).empty());
  CHECK(!group_refusal({0, 1, 2, 3, 4, 5, 6, 7, 8}, 12).empty());   // the fabric's limit is 8
  CHECK(group_refusal({0, 1, 2, 3, 4, 5, 6, 7}, 12).empty());
  // 4. a box started with another pool is refused, and the head fails too
  {
    std::atomic<int> threw{0};
    boxes(3, [&](int i) {
      PoolConfig pc = pool(3, i, base_port + 2);
      pc.timeout_s = 5;
      if (i == 2) pc.ips2[1] = "127.0.9.9";
      FabricPool p;
      try { p.init(pc); } catch (const std::exception&) { threw++; return; }
    });
    CHECK(threw == 3);                                  // the odd box is refused, the head stops, box 1 hears QUIT or times out
  }
  // 5. groups created and dropped in a loop: no descriptor grows while the pool runs, none is left after close, and the
  //    announce and the follow stay in step. The head samples while every other box waits for the next group (it has
  //    taken the last one, or announce would not have returned).
  {
    const int n = 4, loops = 2000;
    const int fds_start = open_fds();
    int fds_early = 0, fds_late = 0;
    boxes(n, [&](int i) {
      FabricPool p; p.init(pool(n, i, base_port + 4));
      for (int k = 0; k < loops; k++) {
        const std::vector<int> L = k % 3 == 0 ? std::vector<int>{0, 1, 2, 3} : k % 3 == 1 ? std::vector<int>{1, 2, 3} : std::vector<int>{3, 0};
        const Group g = i == 0 ? p.announce(L) : p.follow(20000);
        if (g.members != L || g.id != (uint64_t)k + 1) { fails++; return; }
        if (i == 0 && k == 10) fds_early = open_fds();
        if (i == 0 && k == loops - 2) fds_late = open_fds();
      }
      p.close();
    });
    CHECK(fds_early > fds_start && fds_late == fds_early);   // the pool's links, and nothing more per group
    CHECK(open_fds() == fds_start);                          // close drops every link
  }
  // 6. close: a box waiting for a group learns that the pool closed
  {
    std::atomic<int> told{0};
    boxes(3, [&](int i) {
      FabricPool p; p.init(pool(3, i, base_port + 6));
      if (i == 0) { p.announce({0, 2}); p.close(); return; }
      p.follow(20000);
      try { p.follow(20000); } catch (const std::exception& e) { if (std::string(e.what()).find("closed the pool") != std::string::npos) told++; }
    });
    CHECK(told == 2);
  }
  // 7. a follow that times out with nothing sent leaves the pool usable; a refused announce leaves it as it was
  {
    std::atomic<bool> timed_out{false};
    boxes(2, [&](int i) {
      FabricPool p; p.init(pool(2, i, base_port + 8));
      if (i == 0) {
        bool threw = false;
        try { p.announce({0, 0}); } catch (const std::exception&) { threw = true; }
        CHECK(threw && p.up());
        while (!timed_out.load()) usleep(1000);         // box 1 has timed out once
        CHECK(p.announce({1, 0}).id == 1);
      } else {
        bool threw = false;
        try { p.follow(100); } catch (const std::exception&) { threw = true; }
        CHECK(threw && p.up());
        timed_out = true;
        const Group g = p.follow(20000);
        CHECK(g.id == 1 && g.rank == 0 && g.s2_head_ip == "127.0.0.1");
      }
    });
  }
  // 8. a broken pool refuses further use: the head dies mid-life, a waiting box sees the link close, then up() is false
  {
    boxes(2, [&](int i) {
      FabricPool p; p.init(pool(2, i, base_port + 10));
      if (i == 0) { p.announce({0, 1}); usleep(100000); return; }   // the destructor sends QUIT
      p.follow(20000);
      bool threw = false;
      try { p.follow(20000); } catch (const std::exception&) { threw = true; }
      CHECK(threw && !p.up());
      threw = false;
      try { p.follow(100); } catch (const std::exception&) { threw = true; }
      CHECK(threw);
    });
  }
  if (fails) { fprintf(stderr, "fabric pool: %d FAILED\n", fails.load()); return 1; }
  puts("fabric pool: boot, 6 layouts on 6 boxes, group_fabric = the fixed config, refusals, pool mismatch, 2000 groups without a leaked fd, close, timeouts and a broken pool PASS");
}
