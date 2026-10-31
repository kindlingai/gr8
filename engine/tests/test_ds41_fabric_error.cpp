// Host test of the DS41 serve fabric error text (engine/runtime/ds41_fabric_error.h): no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -I engine engine/tests/test_ds41_fabric_error.cpp -o /tmp/test_ds41_fabric_error && /tmp/test_ds41_fabric_error
// The words are the ones the kernels and the proxy build (glue.cuh wait_peers: code | c << 8 | peer; fabric.cpp: 0x70 << 24 | c).
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include "runtime/ds41_fabric_error.h"
#define EXPECT_EQ(a, b) do { const std::string x = (a), y = (b); if (x != y) { std::printf("FAIL %s:%d\n  got  %s\n  want %s\n", __FILE__, __LINE__, x.c_str(), y.c_str()); std::exit(1); } } while (0)
int main()
{
    std::vector<int> all(40); std::iota(all.begin(), all.end(), 0);
    const int eg[2] = {1, 14};
    const char* P = "DS41 serve fabric error";
    // the layer exchanges: c = 2i attention, 2i+1 MLP (layer ids[i])
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61000201u, all, 2, eg), "DS41 serve fabric error 0x61000201: layer-exchange wait timed out, waiting for rank 1 after ~10 s: collective 2 (attention exchange of layer 1)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61001b03u, all, 2, eg), "DS41 serve fabric error 0x61001b03: layer-exchange wait timed out, waiting for rank 3 after ~10 s: collective 27 (MLP exchange of layer 13)");
    // the head merge is c = 2*nl = 80; the two Engram collectives come after it: 81 (layer 1), 82 (layer 14)
    EXPECT_EQ(ds41::fabric_error_text(P, 0x62005002u, all, 2, eg), "DS41 serve fabric error 0x62005002: head-merge wait timed out, waiting for rank 2 after ~10 s: collective 80 (head merge)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61005101u, all, 2, eg), "DS41 serve fabric error 0x61005101: layer-exchange wait timed out, waiting for rank 1 after ~10 s: collective 81 (Engram 0, layer 1)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61005202u, all, 2, eg), "DS41 serve fabric error 0x61005202: layer-exchange wait timed out, waiting for rank 2 after ~10 s: collective 82 (Engram 1, layer 14)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61005301u, all, 2, eg), "DS41 serve fabric error 0x61005301: layer-exchange wait timed out, waiting for rank 1 after ~10 s: collective 83 (outside this fabric)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61005101u, all, 0, nullptr), "DS41 serve fabric error 0x61005101: layer-exchange wait timed out, waiting for rank 1 after ~10 s: collective 81 (outside this fabric)");
    // a layer subset: collective 3 is the MLP exchange of the second served layer (geom layer 1)
    EXPECT_EQ(ds41::fabric_error_text(P, 0x61000300u, {0, 1, 14, 20}, 2, eg), "DS41 serve fabric error 0x61000300: layer-exchange wait timed out, waiting for rank 0 after ~10 s: collective 3 (MLP exchange of layer 1)");
    // the proxy
    EXPECT_EQ(ds41::fabric_error_text(P, 0x70000002u, all, 2, eg), "DS41 serve fabric error 0x70000002: proxy gave up after 30 s waiting for the doorbell of collective 2 (attention exchange of layer 1)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x71000000u, all, 2, eg), "DS41 serve fabric error 0x71000000: RDMA work completion error (see the '[fabric rN] WC error' line)");
    // the Engram engine's own word, and a word nobody decodes
    EXPECT_EQ(ds41::fabric_error_text(P, 0x80000000u, all, 2, eg), "DS41 serve fabric error 0x80000000: the Engram engine failed (NVMe read or window token; the 'DS41 Engram:' line above says which)");
    EXPECT_EQ(ds41::fabric_error_text(P, 0x00000001u, all, 2, eg), "DS41 serve fabric error 0x00000001");
    std::printf("test_ds41_fabric_error: all checks passed\n");
    return 0;
}
