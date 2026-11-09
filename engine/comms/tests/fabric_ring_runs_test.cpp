// Slot runs on the switchless ring (fabric.h FabricConfig::run1 with ring = true, as DS41 prefill uses them).
// The ring proxy sends like the mesh one: direct peers (next, prev) get one RDMA write at fab_recv_off(par, c, me); the diagonal
// peer (TP=4) gets the same write as kRelayChunk chunks through the relay ring (relay.h RelayOut), each forwarded to
// RelayHdr.off = roff + at (fabric.cpp Impl::post, relay_loop). Model: every write of every (sender, parity, collective) at its
// full slot (payload + tail = stride) lands, per receiving peer, inside its own slot [fab_recv_off, + fab_slot_stride) and inside
// the recv area [0, 2 half nr); the slots of a peer's recv area are covered once, no gap, no overlap. nr 4 and 3, DS41 run shapes.
// Also: Fabric::init's guard (fab_runs_allowed) takes run1 on ring and mesh, refuses it on the var layout.
// every shape runs twice, nr slots and compact (FabricDev::compact: no own-rank slot, area [0, 2 half (nr - 1))). The sender
// names the receiving peer as the area owner (fabric.cpp: fab_recv_off_at(dev, par, c, p, me)); a mutant that computes the offset
// once with the sender as owner (the nr-slot formula's habit) must fail on the compact area.
#include "fabric.h"
#include "relay.h"
#include <algorithm>
#include <cstdio>
#include <tuple>
#include <vector>

using namespace g6;
static int fails = 0, checks = 0;
static bool quiet = false;   // the in-test mutant run: count, do not print
static bool old_sender = false;   // mutant: roff with owner = me, as before the compact layout
#define CHECK(x) do { ++checks; if (!(x)) { ++fails; if (!quiet) fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); } } while (0)

struct W { size_t off, len; int par, c, src; };   // one landed write (or relay chunk) in a peer's recv area

static void ring1(int compact, int nr, int ncoll, int r1, int r2, size_t s0, size_t s1, size_t s2, int r3, size_t s3)
{
  FabricConfig cfg; cfg.ring = true; cfg.slot_bytes = s0; cfg.run1 = r1; cfg.run2 = r2; cfg.slot_bytes1 = s1; cfg.slot_bytes2 = s2;
  cfg.run3 = r3; cfg.slot_bytes3 = s3;
  CHECK(fab_runs_allowed(cfg, false));
  FabricDev d{}; d.nr = nr; d.compact = compact;
  CHECK(fab_slot_layout(cfg, ncoll, &d));
  const size_t recv = fab_recv_bytes(d);
  CHECK(recv == 2 * fab_slot_base(d, ncoll) * (size_t)(nr - compact));
  std::vector<std::vector<W>> got(nr);
  size_t chunks = 0;
  for (int me = 0; me < nr; ++me) {
    const int nxt = (me + 1) % nr, prv = (me + nr - 1) % nr, dia = nr == 4 ? (me + 2) % nr : -1;
    RelayOut out;                                    // this sender's relay ring on nxt (forwarded at once: done = pos)
    for (int par = 0; par < 2; ++par)
      for (int c = 0; c < ncoll; ++c) {
        const size_t len = fab_slot_stride(d, c);    // payload stride - 8 + the 8-byte tail
        CHECK(fab_slot_fits(d, c, len - 8));
        for (int p = 0; p < nr; ++p) {
          if (p == me) continue;
          const size_t roff = fab_recv_off_at(d, par, c, old_sender ? me : p, me);   // my slot in p's area
          CHECK(p == nxt || p == prv || p == dia);
          if (p != dia) { got[p].push_back({roff, len, par, c, me}); continue; }
          for (uint32_t at = 0; at < len; at += kRelayChunk) {            // Impl::post, diagonal
            const uint32_t n = len - at < kRelayChunk ? (uint32_t)(len - at) : kRelayChunk;
            uint64_t pos;
            CHECK(out.reserve(n, kRelayRing, out.pos, &pos));
            CHECK(pos % kRelayRing + n <= kRelayRing);
            got[p].push_back({roff + at, n, par, c, me});                 // relay_loop: fin_addr + h.off
            ++chunks;
          }
        }
      }
  }
  for (int p = 0; p < nr; ++p) {
    std::vector<W>& v = got[p];
    std::sort(v.begin(), v.end(), [](const W& a, const W& b) { return a.off < b.off; });
    for (const W& w : v) {                           // own slot, inside the area
      const size_t so = fab_recv_off_at(d, w.par, w.c, p, w.src);
      CHECK(w.off >= so && w.off + w.len <= so + fab_slot_stride(d, w.c) && w.off + w.len <= recv);
    }
    // the peer's recv area: every slot of the other ranks written once (its own slots get no write)
    std::vector<std::tuple<size_t, size_t>> slots;
    for (int par = 0; par < 2; ++par)
      for (int c = 0; c < ncoll; ++c)
        for (int s = 0; s < nr; ++s) if (s != p) slots.push_back({fab_recv_off_at(d, par, c, p, s), fab_slot_stride(d, c)});
    std::sort(slots.begin(), slots.end());
    size_t i = 0;
    for (auto& [so, st] : slots) {                   // the writes into this slot tile it exactly
      size_t at = so;
      while (i < v.size() && v[i].off < so + st) { CHECK(v[i].off == at); at += v[i].len; ++i; }
      CHECK(at == so + st);
    }
    CHECK(i == v.size());
  }
  printf("ring nr%d ncoll %d runs [%d, %d, %d): strides %zu / %zu / %zu / %zu, recv %zu B, %zu relay chunks\n", nr, ncoll, r1, r2, r3,
         d.stride, d.stride1, d.stride2, d.stride3, recv, chunks);
}

// relay_plan (relay.h), the diagonal post plan of Impl::post and Impl::wrs.
// 1. nsge 1 = the old post, segment for segment: inline iff len <= inline_max, else 1 MiB chunks; the last one takes wr_id / signal;
//    WR count = the old Impl::wrs formula.
// 2. nsge 2 (a flat RS slice): the payload segments tile [off, off + len0) in post order, then ONE inline 8-B segment from the tail
//    word to off + len0, last; nothing before it is last. The relay forwards in arrival order, so the tail lands after every payload byte.
// 3. the flat RS slices of the DS41 W2048 shape (Q 512 rows): every sender's slice to every peer (direct: one 2-SGE write; diagonal:
//    the plan) lands as [roff, roff + nb + 8) exactly, payload bytes from [p ss, p ss + nb) of the send slot, the tail from NR ss + 8 p,
//    and the tail word never overlaps any slice's rows; all of it inside the send slot stride.
static const uint32_t kInl = 40;   // fabric.cpp kRelayInline = 64 - sizeof(RelayHdr) (24)
using PlanFn = int (*)(uint64_t, const uint64_t*, const uint32_t*, int, uint32_t, RelaySeg*, int);
// a write's landed bytes as (dst off, src addr, len) in landing order
static std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> land(PlanFn f, uint64_t off, const uint64_t* src, const uint32_t* len, int nsge, int* nseg, int* wrs, int* lastpos)
{
  static std::vector<RelaySeg> sg(kRelayPlanMax);
  const int n = f(off, src, len, nsge, kInl, sg.data(), kRelayPlanMax);
  std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> v;
  *nseg = n; *wrs = n > 0 ? relay_plan_wrs(sg.data(), n) : 0; *lastpos = -1;
  for (int i = 0; i < n; ++i) {
    v.push_back({sg[i].off, sg[i].src, sg[i].len});
    if (sg[i].last) { CHECK(*lastpos < 0); *lastpos = i; }
    const bool tail = nsge == 2 && i == n - 1;                     // inline exactly: the tail word, or a whole payload <= inline_max
    CHECK(sg[i].inl == (tail || len[0] <= kInl));
    CHECK(sg[i].len <= kRelayChunk && sg[i].len > 0);
  }
  return v;
}
static void plan_checks(PlanFn f)
{
  for (uint32_t len : {8u, 40u, 41u, 4096u, kRelayChunk - 8, kRelayChunk, kRelayChunk + 8, 3 * kRelayChunk + 520, 10485768u, 25166088u}) {
    const uint64_t src = 0x100000, off = 0x7000;
    int n, w, last;
    auto v = land(f, off, &src, &len, 1, &n, &w, &last);
    const int old_wrs = len > kInl ? 2 * (int)((len + kRelayChunk - 1) / kRelayChunk) : 1;   // the old Impl::wrs
    CHECK(w == old_wrs && n > 0 && last == n - 1);
    uint64_t at = 0;
    for (auto& [o, s, l] : v) { CHECK(o == off + at && s == src + at); at += l; }
    CHECK(at == len);
    // 2-sge: payload len, then the tail word from elsewhere
    const uint64_t s2[2] = {src, 0x900000}; const uint32_t l2[2] = {len, 8};
    auto v2 = land(f, off, s2, l2, 2, &n, &w, &last);
    CHECK(n >= 2 && last == n - 1 && w == old_wrs + 1);
    at = 0;
    for (int i = 0; i + 1 < (int)v2.size(); ++i) { auto& [o, s, l] = v2[i]; CHECK(o == off + at && s == src + at); at += l; }
    CHECK(at == len);
    if (!v2.empty()) { auto& [o, s, l] = v2.back(); CHECK(o == off + len && s == 0x900000 && l == 8); }
  }
  uint64_t s3[2] = {0, 0}; uint32_t l3[2] = {64, 64};
  RelaySeg sg[8];
  CHECK(f(0, s3, l3, 2, kInl, sg, 8) < 0);                    // a tail longer than inline_max is refused
  CHECK(f(0, s3, l3, 3, kInl, sg, 8) < 0);                    // 3 SGEs are refused
}
static void flat_rs(int nr, int rows, size_t row_bytes)
{
  const int q = (rows + nr - 1) / nr;
  const uint32_t nb = (uint32_t)(q * row_bytes);
  const size_t ss = nb, stride = 4 * (((size_t)nb + 8 + 255) & ~size_t(255));   // run 0 of the RS fabric (DS41 reduce-scatter prefill): 4 aligned slices
  CHECK(nr * ss + 8 * nr <= stride);
  const uint64_t send = 0x10000000, roff = 0x40000;
  for (int me = 0; me < nr; ++me)
    for (int p = 0; p < nr; ++p) {
      if (p == me) continue;
      const bool dia = nr == 4 && p == (me + 2) % nr;
      const uint64_t sp = send + fab_rs_slice_off(ss, p), tw = send + fab_rs_tail_off(ss, p, nb, true, nr);   // fabric.cpp's proxy math
      CHECK(tw >= send + nr * ss && tw + 8 <= send + stride);                  // the tail word is past every slice's rows
      std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> v;
      if (dia) { int n, w, last; const uint64_t s2[2] = {sp, tw}; const uint32_t l2[2] = {nb, 8}; v = land(&relay_plan, roff, s2, l2, 2, &n, &w, &last); }
      else { v.push_back({roff, sp, nb}); v.push_back({roff + nb, tw, 8}); }   // direct: one RC write, 2 SGEs gathered in order
      uint64_t at = 0;
      for (size_t i = 0; i < v.size(); ++i) {
        auto& [o, s, l] = v[i];
        CHECK(o == roff + at);
        if (i + 1 < v.size()) CHECK(s == sp + at && s + l <= sp + nb); else CHECK(s == tw && l == 8);
        at += l;
      }
      CHECK(at == (uint64_t)nb + 8);
    }
}

static void ring(int nr, int ncoll, int r1, int r2, size_t s0, size_t s1, size_t s2, int r3 = 0, size_t s3 = 0)
{
  for (int compact : {0, 1}) ring1(compact, nr, ncoll, r1, r2, s0, s1, s2, r3, s3);
}

int main()
{
  // the guard: run1 on ring and mesh yes, on var no; no runs always
  {
    FabricConfig c; c.run1 = 42; c.ring = true; CHECK(fab_runs_allowed(c, false)); CHECK(!fab_runs_allowed(c, true));
    c.ring = false; CHECK(fab_runs_allowed(c, false)); CHECK(!fab_runs_allowed(c, true));
    c.run1 = 0; CHECK(fab_runs_allowed(c, true) && fab_runs_allowed(c, false));
  }
  const size_t dec = 48 * 6144 * 2;
  for (int nr : {4, 3}) {
    ring(nr, 83, 42, 81, 2048ull * 5120 * 2, std::max(dec, (size_t)128 * 5120 * 2), 2048ull * 6144 * 2);   // DS41 W2048, tail 128, Engram
    ring(nr, 83, 80, 81, 2048ull * 5120 * 2, dec, 2048ull * 6144 * 2);                                      // no bounded replay
    ring(nr, 81, 42, 81, 2048ull * 5120 * 2, std::max(dec, (size_t)128 * 5120 * 2), 0);                     // Engram off
    // reduce-scatter + all-gather: run 0 raised to 4 RS slices (4 x round_up(512 x 5120 x 2 + 8, 256) - 8), 42 AG collectives
    // [83, 125) of one quarter (run 3); with Engram and without it (run2 == run3)
    ring(nr, 125, 42, 81, 4 * 5243136ull - 8, std::max(dec, (size_t)128 * 5120 * 2), 2048ull * 6144 * 2, 83, 512ull * 5120 * 2);
    ring(nr, 123, 42, 81, 4 * 5243136ull - 8, std::max(dec, (size_t)128 * 5120 * 2), 0, 81, 512ull * 5120 * 2);
    ring(nr, 9, 2, 4, 40, 3000000, 8, 6, 2100000);                                                       // tiny 4-run, relay chunks
    ring(nr, 11, 0, 0, dec, 0, 0);                                                                         // DSpark draft: one run
    ring(nr, 7, 3, 5, 40, 3000000, 8);                                                                     // tiny + a 3-chunk run
  }
  // the relay plan, and the flat RS slices at the DS41 shapes; then the same checks on a plan that drops the tail must fail
  plan_checks(&relay_plan);
  for (int nr : {4, 3}) { flat_rs(nr, 2048, 5120 * 2); flat_rs(nr, 846, 5120 * 2); flat_rs(nr, 1025, 5120 * 2); flat_rs(nr, 33, 5120 * 2); }
  {
    const int f0 = fails, c0 = checks;
    quiet = true;
    plan_checks([](uint64_t off, const uint64_t* src, const uint32_t* len, int nsge, uint32_t im, RelaySeg* out, int max) {
      return relay_plan(off, src, len, nsge > 1 ? 1 : nsge, im, out, max);   // mutant: the tail SGE dropped
    });
    const int caught = fails - f0;
    fails = f0; checks = c0;
    quiet = false;
    CHECK(caught > 0);
    printf("relay plan: tail-drop mutant caught by %d checks\n", caught);
  }
  {   // mutant: the sender's offset computed with itself as owner lands outside / on the wrong slot of the compact area
    const int f0 = fails, c0 = checks;
    quiet = true; old_sender = true;
    for (int nr : {4, 3}) ring1(1, nr, 83, 42, 81, 2048ull * 5120 * 2, std::max(dec, (size_t)128 * 5120 * 2), 2048ull * 6144 * 2, 0, 0);
    const int caught = fails - f0;
    fails = f0; checks = c0;
    quiet = false; old_sender = false;
    CHECK(caught > 0);
    printf("compact recv: old-sender-offset mutant caught by %d checks\n", caught);
  }
  printf("fabric ring runs: %d checks, %d failed -> %s\n", checks, fails, fails ? "FAIL" : "OK");
  return fails ? 1 : 0;
}
