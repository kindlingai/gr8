// Two-rank fabric test of FabricConfig::stripe_min (dual-rail striping), host-driven (no kernels).
// Both ranks exchange one uniform-layout collective of BYTES per step, as the prefill partial exchange does: write a
// pattern of (rank, step) into the send slot, ring the doorbell, wait for the peer's tail, check every payload word.
// Prints the per-step exchange time (doorbell -> peer tail seen) p10 / p50 / p90.
//   stripe_test RANK IP0,IP1 IP2_0,IP2_1|- BYTES STEPS STRIPE_MIN(0 = off) [PORT]
// Build (on a Spark): nvcc -O2 -std=c++17 -I.. -o stripe_test stripe_test.cpp ../fabric.cpp -libverbs -lpthread
#include "../fabric.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(t);
    return v;
}
static unsigned long long word(int rank, unsigned long long step, size_t i)
{
    return (static_cast<unsigned long long>(rank + 1) << 56) ^ (step << 32) ^ (i * 0x9e3779b97f4a7c15ull);
}

int main(int argc, char** argv)
{
    if (argc < 7) { fprintf(stderr, "usage: stripe_test RANK IPS IPS2|- BYTES STEPS STRIPE_MIN [PORT]\n"); return 2; }
    g6::FabricConfig cfg;
    cfg.rank = atoi(argv[1]);
    cfg.ips = split(argv[2]);
    if (strcmp(argv[3], "-") != 0) cfg.ips2 = split(argv[3]);
    const size_t bytes = static_cast<size_t>(atoll(argv[4]));
    const int steps = atoi(argv[5]);
    cfg.stripe_min = static_cast<size_t>(atoll(argv[6]));
    cfg.port = argc > 7 ? atoi(argv[7]) : 18700;
    cfg.slot_bytes = bytes;
    cfg.ncoll = 1;
    if (bytes % 8 || cfg.ips.size() != 2) { fprintf(stderr, "BYTES must be a multiple of 8, two ranks\n"); return 2; }
    g6::Fabric f;
    f.init(cfg);
    f.start_proxy();
    const g6::FabricDev d = f.dev();
    const int me = cfg.rank, peer = 1 - me;
    std::vector<double> ms;
    size_t bad = 0;
    for (int s = 0; s < steps; ++s)
    {
        const unsigned long long step = s, seq = g6::coll_seq(step, 0, 1);
        const int par = static_cast<int>(step & 1);
        unsigned long long* src = reinterpret_cast<unsigned long long*>(d.send + static_cast<size_t>(par) * d.stride);
        for (size_t i = 0; i < bytes / 8; ++i) src[i] = word(me, step, i);
        f.enqueue_step(step, {static_cast<uint32_t>(bytes)});
        f.barrier();                                         // both ranks ring at about the same time
        const auto t0 = std::chrono::steady_clock::now();
        __atomic_store_n(d.doorbell, seq, __ATOMIC_RELEASE);
        const char* rs = d.recv + (static_cast<size_t>(par) * d.nr + peer) * d.stride;
        volatile const unsigned long long* tail = reinterpret_cast<volatile const unsigned long long*>(rs + bytes);
        while (*tail != seq)
            if (*d.error) { fprintf(stderr, "rank %d: fabric error 0x%x at step %d\n", me, *d.error, s); return 1; }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const auto t1 = std::chrono::steady_clock::now();
        const volatile unsigned long long* got = reinterpret_cast<const volatile unsigned long long*>(rs);
        for (size_t i = 0; i < bytes / 8; ++i) if (got[i] != word(peer, step, i)) ++bad;
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        f.barrier();                                         // the peer has read my slot of this parity before reuse
    }
    f.stop_proxy();
    std::vector<double> v(ms.begin() + std::min<size_t>(2, ms.size()), ms.end());   // first steps warm the QPs
    std::sort(v.begin(), v.end());
    auto pct = [&](double q) { return v.empty() ? 0.0 : v[static_cast<size_t>(q * (v.size() - 1))]; };
    printf("rank %d bytes %zu steps %d stripe_min %zu: exchange ms p10 %.3f p50 %.3f p90 %.3f, %.1f Gb/s at p50, bad words %zu %s\n",
           me, bytes, steps, cfg.stripe_min, pct(0.1), pct(0.5), pct(0.9), bytes * 8 / (pct(0.5) * 1e6), bad, bad ? "FAIL" : "BYTES OK");
    return bad ? 1 : 0;
}
