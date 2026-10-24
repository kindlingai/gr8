// KvTiers' store failures (--kv-store-fail, kv_tier.cuh): host-only, so the host tests run it without CUDA.
// hard: a failed store call throws, and the serve stops (the behaviour before this knob). soft: the tier is dead on this
// rank from that call on (its later calls are skipped, so a frozen node costs one timeout, not one per call), and the
// call's bit goes into round. The ranks OR their rounds at the barrier after the drain (service.h settle), and every
// rank then drops the agreed tiers from the pool alike (pool.h tier_dead) and redoes the grants whose restores those
// copies were: one rank's failed read leaves its pages wrong while the others' are right.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace g6s2 {
namespace kvt {
struct Faults {
    bool soft = false;
    int rank = 0;
    std::atomic<uint64_t> round{0};   // bit t: tier t failed on this rank since the last take()
    std::atomic<uint64_t> dead{0};    // the tiers this rank no longer calls: failed here, or agreed dead
    std::atomic<uint64_t> failed{0}, timeouts{0};   // store calls that failed here, and those that timed out
    // one store call of tier t (rc: 0 = done, else err says why). true: it ran
    template <class F> bool call(uint32_t t, const char* what, F&& f, const char* (*err)()) {
        const uint64_t bit = t < 64 ? 1ull << t : 0;
        if (dead.load() & bit) return false;
        if (f() == 0) return true;
        const std::string why = err();
        if (!soft || !bit) throw std::runtime_error(std::string("KV store ") + what + ": " + why);
        ++failed;
        if (why.find("timed out") != std::string::npos) ++timeouts;
        dead |= bit;
        round |= bit;
        fprintf(stderr, "# rank %d: s2 store %u failed (--kv-store-fail soft): %s %s; its copies are misses from now on\n", rank, t,
                what, why.c_str());
        return false;
    }
    // this rank's failures since the last take(), for the barrier
    uint64_t take() { return round.exchange(0); }
    // the tiers every rank agreed are dead
    void agreed(uint64_t mask) { dead |= mask; }
};
} // namespace kvt
} // namespace g6s2
