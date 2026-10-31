// F14 P-A slot pool (fabric.h FabricConfig::pool, G6_DS_PF_POOL). Host only. DS41 main fabric shapes of fabric_compact_mem.cpp
// (NL 40, i_late 21: run 0 = 42 layer exchanges, Engram 2, RS on: run 3 = 42 AG collectives), W 2048 and 4096, P 2..4.
//  1. pool 0: fab_slot_base equals the formula before F14 for every c (the layout byte for byte).
//  2. layout: run 0 / run 3 collectives share a slot exactly when c = c' mod P; distinct slots never overlap; bytes per rank equal
//     f14/pool_mem.out (W2048 P4 1650583552 B, W4096 P4 2892097536 B); the guards refuse P = 1, P > run 0,
//     run 3 shorter than run 0 or longer than run 0 + 1, and no runs.
//  3. mesh and ring: every write (sender, parity, c) lands in the receiver's own slot of (c, sender) (fab_recv_off with owner = the
//     receiver), and for the ring diagonal every relay chunk too; the receiver reads the same offset.
//  4. credits: the RS prefill order (ds41::rs::job over coll_exec_order, Engram mid-step) gives RS(c) <- AG(c - P), AG(c) <- RS(c - P + 1);
//     the decode order gives c <- c - P + 1; each credit is posted after the slot's last user and before c. Mutants: P = 1 (forced past
//     the guard) must report no credit; a skip of every run 3 AG still gives credits.
//  5. the 0x76 error word has a name (ds41_fabric_error.h).
#include "fabric.h"
#include "relay.h"
#include "../runtime/ds41_rs.h"
#include "../runtime/ds41_coll_order.h"
#include "../runtime/ds41_fabric_error.h"
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace g6;
static int fails = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++fails; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); } } while (0)

static const int NL = 40, IL = 21, NX = 2, NC = 2 * NL + 1 + NX, H = 5120, MX = 48;

static FabricConfig cfg_of(int W, int nr, bool rs, int pool)
{
  FabricConfig c;
  const size_t dec = (size_t)MX * 6144 * 2;
  const int q = ds41::rs::quarter(W, nr);
  c.slot_bytes = std::max(dec, (size_t)W * H * 2);
  if (rs) c.slot_bytes = std::max(c.slot_bytes, (size_t)nr * ds41::rs::slice_stride(q, (size_t)H * 2) - 8);
  c.run1 = 2 * IL; c.run2 = 2 * NL + 1; c.slot_bytes1 = std::max(dec, (size_t)128 * H * 2); c.slot_bytes2 = std::max(dec, (size_t)W * 6144 * 2);
  if (rs) { c.run3 = NC; c.slot_bytes3 = (size_t)q * H * 2; }
  c.pool = pool;
  return c;
}

// fab_slot_base before F14 (one slot per collective)
static size_t old_base(const FabricDev& f, int c)
{
  if (c <= f.c1) return (size_t)c * f.stride;
  if (c <= f.c2) return (size_t)f.c1 * f.stride + (size_t)(c - f.c1) * f.stride1;
  if (c <= f.c3) return (size_t)f.c1 * f.stride + (size_t)(f.c2 - f.c1) * f.stride1 + (size_t)(c - f.c2) * f.stride2;
  return (size_t)f.c1 * f.stride + (size_t)(f.c2 - f.c1) * f.stride1 + (size_t)(f.c3 - f.c2) * f.stride2 + (size_t)(c - f.c3) * f.stride3;
}

static void layout(int W, int nr, bool rs, int P)
{
  const int n = ds41::rs::main_ncoll(NC, rs ? 2 * IL : 0);
  FabricDev d{}; d.nr = nr; d.compact = 1;
  CHECK(fab_slot_layout(cfg_of(W, nr, rs, P), n, &d));
  CHECK(d.pool == P);
  if (!P) { for (int c = 0; c <= n; ++c) CHECK(fab_slot_base(d, c) == old_base(d, c)); return; }
  const size_t half = fab_slot_base(d, n);
  for (int c = 0; c < n; ++c)
    for (int c2 = c + 1; c2 < n; ++c2) {
      const size_t a = fab_slot_base(d, c), b = fab_slot_base(d, c2);
      const int sa = fab_pool_slot(d, c), sb = fab_pool_slot(d, c2);
      if (sa >= 0 && sa == sb) { CHECK(a == b); CHECK(fab_slot_stride(d, c) == fab_slot_stride(d, c2)); }
      else CHECK(a + fab_slot_stride(d, c) <= b || b + fab_slot_stride(d, c2) <= a);   // no overlap
      CHECK((sa >= 0 && sa == sb) == ((c < 2 * IL && c2 < 2 * IL && c % P == c2 % P) || (c >= NC && c2 >= NC && (c - NC) % P == (c2 - NC) % P)));
    }
  for (int c = 0; c < n; ++c) CHECK(fab_slot_base(d, c) + fab_slot_stride(d, c) <= half);
  const size_t bytes = 2 * half + fab_recv_bytes(d);
  if (rs && nr == 4 && P == 4) CHECK(bytes == (W == 2048 ? 1650583552ull : 2892097536ull));
  if (rs && nr == 4 && P == 2 && W == 4096) CHECK(bytes == 2053216256ull);
  if (rs && nr == 4) printf("| %d | %d | %zu B (%.3f GiB) |\n", W, P, bytes, bytes / double(1 << 30));
}

// every write of every sender lands in the receiver's slot of (c, sender); the ring diagonal as relay chunks at roff + at
static void wires(int W, int nr, bool ring, int P)
{
  const int n = ds41::rs::main_ncoll(NC, 2 * IL);
  FabricConfig cfg = cfg_of(W, nr, true, P); cfg.ring = ring;
  CHECK(fab_runs_allowed(cfg, false));
  FabricDev d{}; d.nr = nr; d.compact = 1;
  CHECK(fab_slot_layout(cfg, n, &d));
  const size_t area = fab_recv_bytes(d);
  for (int me = 0; me < nr; ++me) {
    const int dia = nr == 4 ? (me + 2) % nr : -1;
    for (int par = 0; par < 2; ++par)
      for (int c = 0; c < n; ++c) {
        const size_t len = fab_slot_stride(d, c);
        for (int p = 0; p < nr; ++p) {
          if (p == me) continue;
          FabricDev dp = d; dp.rank = p;
          const size_t roff = fab_recv_off_at(d, par, c, p, me), want = fab_recv_off(dp, par, c, me);   // sender view == reader view
          CHECK(roff == want);
          CHECK(roff + len <= area);
          if (ring && p == dia)
            for (uint32_t at = 0; at < len; at += kRelayChunk) {
              const uint32_t k = len - at < kRelayChunk ? (uint32_t)(len - at) : kRelayChunk;
              CHECK(roff + at + k <= want + len);
            }
          // the slot of (c, me) in p's area is shared only with (c', me), c' = c mod P of the same run, never with another sender
          for (int q = 0; q < nr; ++q) {
            if (q == p || q == me) continue;
            const size_t o = fab_recv_off(dp, par, c, q);
            CHECK(o + len <= roff || roff + len <= o);
          }
        }
      }
  }
}

static std::vector<int> walk_of(const std::vector<int>& order, size_t nsizes, const CollMask& skip)
{
  std::vector<int> w;
  const size_t n = order.empty() ? nsizes : order.size();
  for (size_t k = 0; k < n; ++k) { const int c = order.empty() ? (int)k : order[k]; if (!skip.bit((size_t)c)) w.push_back(c); }
  return w;
}

static void credits(int P)
{
  const int n = ds41::rs::main_ncoll(NC, 2 * IL), K = 2 * IL;
  FabricDev d{}; d.nr = 4; d.compact = 1;
  CHECK(fab_slot_layout(cfg_of(2048, 4, true, P), n, &d));
  std::vector<int> ids(NL); for (int i = 0; i < NL; ++i) ids[i] = i;
  // RS prefill chunk (head skipped, Engram mid-step), as ds41_step builds the proxy job
  const std::vector<int> base = ds41::coll_exec_order(ids, NX, false);
  std::vector<uint32_t> sizes((size_t)NC, (uint32_t)(2048 * H * 2));
  const ds41::rs::Job j = ds41::rs::job(sizes, base, NC, K, 2048, 4, (size_t)H * 2);
  for (int pass = 0; pass < 2; ++pass) {
    const std::vector<int> walk = pass ? walk_of(std::vector<int>(), (size_t)NC, CollMask()) : walk_of(j.order, j.sizes.size(), CollMask());
    std::vector<int> cr;
    CHECK(fab_pool_credits(d, walk, &cr));
    std::vector<int> pos((size_t)n, -1);
    for (size_t k = 0; k < walk.size(); ++k) pos[(size_t)walk[k]] = (int)k;
    for (int c : walk) {
      const int s = fab_pool_slot(d, c);
      if (s < 0) { CHECK(cr[(size_t)c] == -1); continue; }
      if (!pass) {   // RS: RS(c) <- AG(c - P); AG(c) <- RS(c - P + 1)
        if (c < K) CHECK(cr[(size_t)c] == (c >= P ? ds41::rs::c_ag(NC, c - P) : -1));
        else { const int r = c - NC; CHECK(cr[(size_t)c] == (r >= P ? r - P + 1 : -1)); }
      } else CHECK(cr[(size_t)c] == (c < K && c >= P ? c - P + 1 : -1));   // decode: plain all-reduce, no run 3
      if (cr[(size_t)c] >= 0) {
        const int k = cr[(size_t)c], u = pass ? c - P : (c < K ? c - P : NC + (c - NC) - P);
        CHECK(pos[(size_t)u] < pos[(size_t)k] && pos[(size_t)k] < pos[(size_t)c]);   // u, then its credit, then c
        CHECK(fab_slot_base(d, u) == fab_slot_base(d, c));
      }
    }
  }
  // skip every AG: run 0 still pools (credit = the next pooled collective after c - P, here c - P + 1)
  CollMask skip; for (int c = NC; c < n; ++c) skip.set((size_t)c);
  std::vector<int> cr;
  CHECK(fab_pool_credits(d, walk_of(j.order, j.sizes.size(), skip), &cr));
  for (int c = P; c < K; ++c) CHECK(cr[(size_t)c] == c - P + 1);
}

int main()
{
  printf("| W | P | send+recv / rank (RS on, nr 4, compact) |\n|---|---|---|\n");
  for (int W : {2048, 4096})
    for (int nr : {4, 3})
      for (bool rs : {false, true})
        for (int P : {0, 2, 3, 4}) layout(W, nr, rs, P);
  for (int W : {2048, 4096})
    for (int nr : {4, 3})
      for (bool ring : {false, true})
        for (int P : {2, 4}) wires(W, nr, ring, P);
  for (int P : {2, 3, 4}) credits(P);
  // guards
  { FabricDev d{}; const int n = ds41::rs::main_ncoll(NC, 2 * IL);
    CHECK(!fab_slot_layout(cfg_of(2048, 4, true, 1), n, &d));                       // P = 1
    CHECK(!fab_slot_layout(cfg_of(2048, 4, true, 2 * IL + 1), n, &d));              // P > run 0
    CHECK(fab_slot_layout(cfg_of(2048, 4, true, 2 * IL), n, &d));                   // P = run 0: no reuse
    CHECK(!fab_slot_layout(cfg_of(2048, 4, true, 4), n - 1, &d));                   // run 3 != run 0
    CHECK(fab_slot_layout(cfg_of(2048, 4, true, 4), n + 1, &d));                    // B12 state collective after AGs
    CHECK(!fab_slot_layout(cfg_of(2048, 4, true, 4), n + 2, &d));                   // only one extra supported
    FabricConfig u; u.slot_bytes = 4096; u.pool = 2; CHECK(!fab_slot_layout(u, 8, &d));   // no runs
    u.pool = 0; CHECK(fab_slot_layout(u, 8, &d) && d.pool == 0); }
  // mutant: a pool of 1 (forced past the guard) has no credit collective between two users of a slot
  { FabricDev d{}; d.nr = 4; const int n = ds41::rs::main_ncoll(NC, 2 * IL);
    CHECK(fab_slot_layout(cfg_of(2048, 4, true, 2), n, &d)); d.pool = 1;
    std::vector<int> cr, w; for (int c = 0; c < NC; ++c) w.push_back(c);
    CHECK(!fab_pool_credits(d, w, &cr)); }
  { std::vector<int> ids(NL); for (int i = 0; i < NL; ++i) ids[i] = i;
    const std::string t = ds41::fabric_error_text("x", 0x76000502u, ids, NX, nullptr);
    CHECK(t.find("slot pool") != std::string::npos && t.find("rank 2") != std::string::npos && t.find("collective 5 ") != std::string::npos); }
  printf("fabric slot pool: %d checks, %d failed: %s\n", checks, fails, fails ? "FAIL" : "OK");
  return fails ? 1 : 0;
}
