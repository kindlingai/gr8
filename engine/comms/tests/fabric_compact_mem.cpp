// F11 A: bytes per rank of the DS41 main fabric region (send + recv) with nr recv slots per collective and with the compact recv
// area (FabricDev::compact, no own-rank slot), from the real fab_slot_layout / fab_recv_bytes. Config = g6m4.cu's DS41 serve main
// fabric (g6m4.cu:684-702, 745-746): MX 48, HID 5120, Engram 6144 (2 reduces), 40 layers, G6_DS_PF_TAIL 128 (i_late 21, run1 = 42,
// late slot 128 rows), run2 = 2 NL + 1; RS (G6_DS_PF_RS=1): run 0 >= NR slices, AG run3 = [NC, NC + 2 i_late) of one quarter each.
// The head slot (glue::head_bytes_tk(48, ...)) is below the 128-row late slot, so run 1 = max(decode slot, late slot) as here.
// Prints; checks only that compact = send + recv (nr - 1) and saves exactly one recv slot set (= the send area).
#include "fabric.h"
#include "../runtime/ds41_rs.h"
#include <algorithm>
#include <cstdio>

int main()
{
  int fails = 0;
  const int NL = 40, il = 21, NX = 2, NC = 2 * NL + 1 + NX, nr = 4, H = 5120, MX = 48;
  const size_t dec = (size_t)MX * 6144 * 2;
  const double G = 1 << 30;
  printf("| W | RS | ncoll | send+recv now (nr slots) | compact (nr-1) | saved |\n|---|---|---|---|---|---|\n");
  for (int W : {2048, 4096})
    for (int rs : {0, 1}) {
      g6::FabricConfig c; g6::FabricDev d{};
      const int q = ds41::rs::quarter(W, nr), k = rs ? 2 * il : 0;
      c.slot_bytes = std::max(dec, (size_t)W * H * 2);
      if (rs) c.slot_bytes = std::max(c.slot_bytes, (size_t)nr * ds41::rs::slice_stride(q, (size_t)H * 2) - 8);
      c.run1 = 2 * il; c.run2 = 2 * NL + 1; c.slot_bytes1 = std::max(dec, (size_t)128 * H * 2); c.slot_bytes2 = std::max(dec, (size_t)W * 6144 * 2);
      if (rs) { c.run3 = NC; c.slot_bytes3 = (size_t)q * H * 2; }
      const int n = ds41::rs::main_ncoll(NC, k);
      d.nr = nr;
      if (!g6::fab_slot_layout(c, n, &d)) { printf("layout refused\n"); return 1; }
      const size_t send = 2 * g6::fab_slot_base(d, n), now = send + g6::fab_recv_bytes(d);
      g6::FabricDev e = d; e.compact = 1;
      const size_t cmp = send + g6::fab_recv_bytes(e);
      if (cmp != send * nr || now - cmp != send) ++fails;
      printf("| %d | %s | %d | %zu B (%.3f GiB) | %zu B (%.3f GiB) | %zu B (%.3f GiB, %.2f GB) |\n", W, rs ? "on" : "off", n,
             now, now / G, cmp, cmp / G, now - cmp, (now - cmp) / G, (now - cmp) / 1e9);
    }
  printf("fabric compact mem: %s\n", fails ? "FAIL" : "OK");
  return fails ? 1 : 0;
}
