// Slot runs of a uniform fabric (fabric.h FabricConfig::run1, fab_slot_layout / fab_send_off / fab_recv_off, as DS41 prefill uses them).
// 1. one run: every offset equals the old formula ((par * ncoll + c) * stride, ((par * ncoll + c) * nr + p) * stride);
// 2. three runs (the DS41 shapes): the send slots of a parity and the recv slots tile [0, 2 half) and [0, 2 half nr) with no gap or
//    overlap, every slot 256-aligned; 3. capacity: a payload fits its own collective's slot only (enqueue_step's check);
// 4. bad run configs are refused; 5. the DS41 W2048 byte count;
// compact recv (FabricDev::compact, no own-rank slot): 1c. one run = ((par * ncoll + c) * (nr - 1) + q) * stride, q = p - (p > rank);
//    every layout above tiles [0, 2 half (nr - 1)) for every owner rank, and the sender's offset (fab_recv_off_at, owner = the peer)
//    equals the owner's reader offset (fab_recv_off) for every pair; 5c. the bytes it saves.
#include "fabric.h"
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

using namespace g6;
static int fails = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++fails; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); } } while (0)

static FabricDev layout(size_t s0, int ncoll, int nr, int r1 = 0, int r2 = 0, size_t s1 = 0, size_t s2 = 0, int r3 = 0, size_t s3 = 0)
{
  FabricConfig c; c.slot_bytes = s0; c.run1 = r1; c.run2 = r2; c.slot_bytes1 = s1; c.slot_bytes2 = s2; c.run3 = r3; c.slot_bytes3 = s3;
  FabricDev d{}; d.nr = nr;
  CHECK(fab_slot_layout(c, ncoll, &d));
  return d;
}

// every slot [off, off + stride) of the send area and of the recv area, sorted: they must tile the area exactly
static void tiles(const FabricDev& d)
{
  const size_t half = fab_slot_base(d, d.ncoll);
  std::vector<std::pair<size_t, size_t>> s, r;
  for (int par = 0; par < 2; ++par)
    for (int c = 0; c < d.ncoll; ++c) {
      const size_t st = fab_slot_stride(d, c);
      CHECK(st % 256 == 0 && st > 0);
      s.push_back({fab_send_off(d, par, c), st});
      for (int p = 0; p < d.nr; ++p) if (!d.compact || p != d.rank) r.push_back({fab_recv_off(d, par, c, p), st});
    }
  for (auto* v : {&s, &r}) {
    std::sort(v->begin(), v->end());
    size_t at = 0;
    for (auto& x : *v) { CHECK(x.first == at && x.first % 256 == 0); at = x.first + x.second; }
    CHECK(at == (v == &s ? 2 * half : 2 * half * fab_recv_slots(d)));
  }
  CHECK(fab_recv_bytes(d) == 2 * half * (size_t)fab_recv_slots(d));
}
// d as every rank sees it with the compact recv area: tiles for each owner, and sender view == reader view for every pair
static void compact_all(FabricDev d)
{
  d.compact = 1;
  CHECK(fab_recv_slots(d) == d.nr - 1);
  for (int r = 0; r < d.nr; ++r) { d.rank = r; tiles(d); }
  for (int owner = 0; owner < d.nr; ++owner)
    for (int p = 0; p < d.nr; ++p) {
      if (p == owner) continue;
      FabricDev rd = d, sd = d; rd.rank = owner; sd.rank = p;
      for (int par = 0; par < 2; ++par)
        for (int c = 0; c < d.ncoll; ++c) CHECK(fab_recv_off_at(sd, par, c, owner, p) == fab_recv_off(rd, par, c, p));
    }
}

int main()
{
  // 1. one run = the old formulas, byte for byte
  for (int nr : {2, 3, 4, 6})
    for (int ncoll : {1, 3, 11, 83})
      for (size_t sb : {(size_t)8, (size_t)491520, (size_t)589824, (size_t)25165824}) {
        const FabricDev d = layout(sb, ncoll, nr);
        CHECK(d.c1 == 0x7fffffff && d.c2 == 0x7fffffff);
        CHECK(fab_slot_base(d, ncoll) == (size_t)ncoll * d.stride);
        for (int par = 0; par < 2; ++par)
          for (int c = 0; c < ncoll; ++c) {
            CHECK(fab_slot_stride(d, c) == d.stride);
            CHECK(fab_send_off(d, par, c) == ((size_t)par * ncoll + c) * d.stride);
            for (int p = 0; p < nr; ++p) CHECK(fab_recv_off(d, par, c, p) == (((size_t)par * ncoll + c) * nr + p) * d.stride);
            for (int r = 0; r < nr; ++r) {               // 1c. compact, owner r
              FabricDev e = d; e.compact = 1; e.rank = r;
              for (int p = 0; p < nr; ++p)
                if (p != r) CHECK(fab_recv_off(e, par, c, p) == (((size_t)par * ncoll + c) * (nr - 1) + (p - (p > r))) * d.stride);
            }
          }
        tiles(d); compact_all(d);
      }
  // 2. three runs: DS41 40 layers, i_late 21 (c1 = 42), head c = 80, Engram 81, 82; also no split (c1 = 80) and no Engram (run2 = ncoll)
  const size_t dec = 48 * 6144 * 2;
  struct Shape { int ncoll, r1, r2; size_t s0, s1, s2; };
  const Shape shapes[] = {
      {83, 42, 81, 2048ull * 5120 * 2, 128ull * 5120 * 2, 2048ull * 6144 * 2},   // W2048, tail 128, Engram
      {83, 42, 81, 1024ull * 5120 * 2, 128ull * 5120 * 2, 1024ull * 6144 * 2},   // W1024
      {83, 80, 81, 2048ull * 5120 * 2, dec, 2048ull * 6144 * 2},                 // no bounded replay: run 1 = the head only
      {81, 42, 81, 2048ull * 5120 * 2, 128ull * 5120 * 2, 0},                    // Engram off: no run 2
      {11, 6, 6, 409600, 409600, 409600},                                        // run1 == run2
  };
  for (const Shape& x : shapes)
    for (int nr : {2, 4}) {
      const FabricDev d = layout(std::max(dec, x.s0), x.ncoll, nr, x.r1, x.r2, std::max(dec, x.s1), x.s2 ? std::max(dec, x.s2) : 0);
      CHECK(d.c1 == x.r1 && d.c2 == x.r2);
      tiles(d); compact_all(d);
      // 3. capacity per collective: the largest payload of each run fits there and is refused one run over (when the run is smaller)
      for (int c = 0; c < x.ncoll; ++c) {
        const size_t st = fab_slot_stride(d, c);
        CHECK(fab_slot_fits(d, c, st - 8) && !fab_slot_fits(d, c, st) && !fab_slot_fits(d, c, 4));
      }
      if (x.r1 < x.r2 && x.s1 < x.s0) CHECK(!fab_slot_fits(d, x.r1, x.s0));
      if (x.r2 < x.ncoll) CHECK(fab_slot_fits(d, x.ncoll - 1, x.s2) && fab_slot_stride(d, x.ncoll - 1) == d.stride2);
      CHECK(fab_slot_stride(d, 0) == d.stride && fab_slot_stride(d, x.r1 - 1) == d.stride);
    }
  // 4. bad runs are refused
  {
    FabricConfig c; c.slot_bytes = 64; FabricDev d{};
    c.run1 = 5; c.run2 = 4; c.slot_bytes1 = 64; CHECK(!fab_slot_layout(c, 10, &d));          // run1 > run2
    c.run1 = 4; c.run2 = 11; CHECK(!fab_slot_layout(c, 10, &d));                              // run2 > ncoll
    c.run1 = 4; c.run2 = 8; c.slot_bytes2 = 0; CHECK(!fab_slot_layout(c, 10, &d));            // run 2 has no size
    c.run1 = 4; c.run2 = 10; c.slot_bytes1 = 0; CHECK(!fab_slot_layout(c, 10, &d));           // run 1 has no size
    c.run1 = -1; c.run2 = 10; c.slot_bytes1 = 64; CHECK(!fab_slot_layout(c, 10, &d));
  }
  // 5. DS41 W2048 per rank (nr 4): the region's send + recv bytes (Fabric::init: 2 half + 2 half nr) against the uniform W slot
  {
    const FabricDev d = layout(2048ull * 5120 * 2, 83, 4, 42, 81, 128ull * 5120 * 2, 2048ull * 6144 * 2);
    const FabricDev u = layout(2048ull * 6144 * 2, 83, 4);
    const FabricDev dd = layout(dec, 11, 4);
    const size_t h = fab_slot_base(d, 83), hu = fab_slot_base(u, 83), hd = fab_slot_base(dd, 11);
    printf("DS41 W2048 nr4 main: strides %zu / %zu / %zu, half %zu, send+recv %zu B (uniform W slot: %zu B); draft (11 x decode slot) %zu B\n",
           d.stride, d.stride1, d.stride2, h, 10 * h, 10 * hu, 10 * hd);
    CHECK(h == 42ull * 20971776 + 39ull * 1310976 + 2ull * 25166080);
    CHECK(10 * h < 10 * hu / 2);
    FabricDev c = d; c.compact = 1;                  // 5c. send 2 half + recv 2 half (nr - 1) = 8 half
    printf("DS41 W2048 nr4 main compact recv: send+recv %zu B, saves %zu B (one parity pair of nr slots -> nr - 1)\n",
           2 * h + fab_recv_bytes(c), 10 * h - (2 * h + fab_recv_bytes(c)));
    CHECK(2 * h + fab_recv_bytes(c) == 8 * h);
  }
  // 6. reduce-scatter + all-gather: a 4th run [run3, ncoll) for the 42 AG collectives. Without run3 every offset is the 3-run one;
  //    with it the areas tile, each run has its own stride, and the RS run-0 slot holds 4 slices of ss = round_up(Q H 2 + 8, 256).
  {
    const size_t ss = (512ull * 5120 * 2 + 8 + 255) & ~size_t(255), q = 512ull * 5120 * 2;
    CHECK(ss == 5243136);
    for (int nr : {2, 4}) {
      const FabricDev a = layout(2048ull * 5120 * 2, 83, nr, 42, 81, 128ull * 5120 * 2, 2048ull * 6144 * 2);
      const FabricDev b = layout(4 * ss - 8, 125, nr, 42, 81, 128ull * 5120 * 2, 2048ull * 6144 * 2, 83, q);
      CHECK(b.c3 == 83 && b.stride3 == ss && b.stride == 4 * ss && a.c3 == 0x7fffffff && a.stride3 == 0);
      tiles(b); compact_all(b);
      for (int c = 0; c < 83; ++c) CHECK(fab_slot_stride(a, c) == fab_slot_stride(b, c) || c < 42);
      for (int c = 83; c < 125; ++c) CHECK(fab_slot_stride(b, c) == ss && fab_slot_fits(b, c, q) && !fab_slot_fits(b, c, q + 256));
      for (int c = 0; c < 42; ++c) CHECK(4 * ss <= fab_slot_stride(b, c));
      const FabricDev e = layout(4 * ss - 8, 123, nr, 42, 81, 128ull * 5120 * 2, 0, 81, q);   // Engram off: run2 == run3
      tiles(e); compact_all(e); CHECK(e.c2 == 81 && e.c3 == 81 && fab_slot_stride(e, 81) == ss);
      if (nr == 4) {
        const size_t ha = fab_slot_base(a, 83), hb = fab_slot_base(b, 125);
        printf("DS41 W2048 nr4 RS+AG: half %zu -> %zu, send+recv %zu -> %zu B (+%zu B)\n", ha, hb, 10 * ha, 10 * hb, 10 * (hb - ha));
        CHECK(hb - ha == 42ull * (4 * ss - 20971776) + 42ull * ss);
      }
    }
    FabricConfig c; c.slot_bytes = 64; FabricDev d{};
    c.run3 = 5; c.slot_bytes3 = 64; CHECK(!fab_slot_layout(c, 10, &d));                              // run3 without run1
    c.run1 = 2; c.run2 = 4; c.slot_bytes1 = 64; c.slot_bytes2 = 64; c.run3 = 3; CHECK(!fab_slot_layout(c, 10, &d));   // run3 < run2
    c.run3 = 10; CHECK(!fab_slot_layout(c, 10, &d));                                                // run3 == ncoll (empty run)
    c.run3 = 6; c.slot_bytes3 = 0; CHECK(!fab_slot_layout(c, 10, &d));                              // run 3 has no size
    c.slot_bytes3 = 64; CHECK(fab_slot_layout(c, 10, &d) && d.c3 == 6);
    c.run2 = 6; c.slot_bytes2 = 0; CHECK(fab_slot_layout(c, 10, &d) && d.c2 == 6 && d.stride2 == 0); // run2 == run3: no run 2
  }
  printf("fabric runs: %d checks, %d failed -> %s\n", checks, fails, fails ? "FAIL" : "OK");
  return fails ? 1 : 0;
}
