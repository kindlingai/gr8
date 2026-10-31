// gr8 fabric proxy: the order in which one step's collectives are waited for and sent. No verbs, no CUDA, so a host test
// (tests/fabric_order_test.cpp) drives the same loop fabric.cpp runs, with a fake doorbell array and a fake clock.
//
// The proxy waits for doorbell[c] and then sends collective c, one collective at a time. By default c runs 0..n-1 in
// numeric order. A step whose GPU work publishes the collectives in another order (DS41 with Engram: the Engram reduce
// c = 2 NL + 1 + e runs between layer e's ffn publish and the next layer's attn publish, not after the head) must hand the
// proxy that order: Fabric::Rs::order. Every rank passes the same order, so the sends stay in the same sequence on every rank.
#pragma once
#include <atomic>
#include <cstddef>
#include <vector>

#include "fabric.h"

namespace g6 {

// bit c of a collective mask (CollMask, fabric.h: c < 128; a 32-bit `unsigned` converts). c >= 128 is never set.
inline bool mask_bit(const CollMask& m, size_t c) { return m.bit(c); }

// Why a step's order is refused (nullptr = fine). Empty `order` = collectives 0..nsizes-1 in numeric order. A non-empty
// order names exactly the collectives the proxy handles this step, each once, each below nsizes (sizes[c] exists); a
// collective left out is neither waited for nor sent (a prefill chunk that publishes no head).
inline const char* order_error(const std::vector<int>& order, size_t nsizes) {
  std::vector<char> seen(nsizes, 0);
  for (int c : order) {
    if (c < 0 || (size_t)c >= nsizes) return "order names a collective with no size";
    if (seen[(size_t)c]++) return "order names a collective twice";
  }
  return nullptr;
}

enum : unsigned { kWalkOk = 0, kWalkHalt = 1, kWalkTimeout = 2, kWalkPostFailed = 3 };
struct StepWalk { unsigned rc; int c; unsigned long long have; };   // c / have: where it stopped and what the doorbell held

// Walk one step: for each collective in execution order (not in skip), wait until db(c) == coll_seq(step, c, ncoll), then
// post(c, seq). db(c) -> doorbell word; halt() -> stop requested or error word set; now() -> seconds (monotonic);
// post(c, seq) -> false = the send failed (it has set the error word). A doorbell that stays short for timeout_s returns
// kWalkTimeout (the caller publishes the error word).
template <class Db, class Halt, class Now, class Post>
StepWalk walk_step(unsigned long long step, int ncoll, size_t nsizes, const std::vector<int>& order, const CollMask& skip, Db&& db,
                   Halt&& halt, Now&& now, double timeout_s, Post&& post) {
  const size_t n = order.empty() ? nsizes : order.size();
  for (size_t k = 0; k < n; k++) {
    const int c = order.empty() ? (int)k : order[k];
    if (mask_bit(skip, (size_t)c)) continue;
    const unsigned long long seq = coll_seq(step, c, ncoll);
    const double t0 = now();
    while (db(c) != seq) {
      if (halt()) return StepWalk{kWalkHalt, c, db(c)};
      if (now() - t0 > timeout_s) return StepWalk{kWalkTimeout, c, db(c)};
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (!post(c, seq)) return StepWalk{kWalkPostFailed, c, seq};
  }
  return StepWalk{kWalkOk, -1, 0};
}

}  // namespace g6
