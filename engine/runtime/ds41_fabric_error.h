// The text of the DS41 serve step's fabric error word (host-only, no CUDA): it names the code, the collective and the peer.
// Layouts: kernels/glue.cuh wait_peers sets code << 24 | c << 8 | peer (0x61 layer exchange or Engram sum, 0x62 head merge; peer = the rank
// whose tail never arrived, after ~10 s); comms/fabric.cpp's proxy sets 0x70 << 24 | c (30 s without the doorbell of collective c),
// 0x71 << 24 (work completion error), 0x72 << 24 (stripe or bulk wait), 0x76 << 24 | c << 8 | peer (slot pool credit, 30 s). rank.cuh ERR_QS 0x66 (gather wait). Bit 31 alone, 0x80000000,
// is ds41_engram_error (the Engram engine failed closed: a failed NVMe read or a window token outside the vocabulary).
// c is named from the served layers: c < 2*nl is layer ids[c/2], attention (even c) or MLP (odd c) exchange; c == 2*nl is the head merge;
// c > 2*nl is Engram collective e = c - 2*nl - 1 (its layer is engram_layers[e]). The word is the OR of several sources
// (main fabric, CP, gather proxies, Engram); the raw hex is always printed so a mixed word can be read by hand.
#pragma once
#include <cstdio>
#include <string>
#include <vector>
namespace ds41 {

inline std::string coll_name(unsigned c, const std::vector<int>& layer_ids, int ecoll, const int* engram_layers)
{
    char b[96];
    const unsigned nl = (unsigned)layer_ids.size();
    if (c < 2 * nl)
        std::snprintf(b, sizeof b, "collective %u (%s exchange of layer %d)", c, (c & 1) ? "MLP" : "attention", layer_ids[c / 2]);
    else if (c == 2 * nl)
        std::snprintf(b, sizeof b, "collective %u (head merge)", c);
    else if (ecoll > 0 && c - 2 * nl - 1 < (unsigned)ecoll)
        std::snprintf(b, sizeof b, "collective %u (Engram %u, layer %d)", c, c - 2 * nl - 1, engram_layers ? engram_layers[c - 2 * nl - 1] : -1);
    else
        std::snprintf(b, sizeof b, "collective %u (outside this fabric)", c);
    return b;
}

// prefix: the message head ("DS41 serve fabric error"). e != 0.
inline std::string fabric_error_text(const char* prefix, unsigned e, const std::vector<int>& layer_ids, int ecoll, const int* engram_layers)
{
    char h[64];
    std::snprintf(h, sizeof h, " 0x%08x", e);
    std::string s = std::string(prefix) + h;
    if (e == 0x80000000u) return s + ": the Engram engine failed (NVMe read or window token; the 'DS41 Engram:' line above says which)";
    const unsigned code = e >> 24;
    char b[64];
    switch (code)
    {
    case 0x61: case 0x62:
        std::snprintf(b, sizeof b, ", waiting for rank %u after ~10 s: ", e & 0xffu);
        return s + ": " + (code == 0x61 ? "layer-exchange" : "head-merge") + " wait timed out" + b + coll_name((e >> 8) & 0xffu, layer_ids, ecoll, engram_layers);
    case 0x66:
        std::snprintf(b, sizeof b, "gather wait timed out, collective %u, peer rank %u", (e >> 8) & 0xffu, e & 0xffu);
        return s + ": " + b;
    case 0x70:
        return s + ": proxy gave up after 30 s waiting for the doorbell of " + coll_name(e & 0xffffu, layer_ids, ecoll, engram_layers);
    case 0x71: return s + ": RDMA work completion error (see the '[fabric rN] WC error' line)";
    case 0x72: case 0x73: return s + ": bulk or stripe wait timed out";
    case 0x74: case 0x75: return s + ": proxy drain or transport error";
    case 0x76:
        std::snprintf(b, sizeof b, ", rank %u: ", e & 0xffu);
        return s + ": slot pool (GR8_DS_PF_POOL) credit wait timed out after 30 s" + b + coll_name((e >> 8) & 0xffu, layer_ids, ecoll, engram_layers);
    default: return s;
    }
}
}
