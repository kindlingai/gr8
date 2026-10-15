// Ring fabric relay ring (fabric.cpp, FabricConfig::ring): the origin's side of the bounded buffer on the relay rank that
// a diagonal write passes through. This file touches neither verbs nor CUDA, so engine/comms/tests/relay_test.cpp can
// drive it.
#pragma once
#include <cstdint>
#include <deque>

namespace g6 {

constexpr uint64_t kRelayRing = 32ull << 20;   // relay ring bytes per ring Fabric (at most the region's receive side)
constexpr uint32_t kRelayChunk = 1u << 20;     // a longer write crosses the relay in chunks of this size, in order
constexpr unsigned kRelayCredit = 64;          // credit writes the relay-in send queue holds, and what its CQ reserves
constexpr unsigned kRelayCreditSignal = 32;    // one credit CQE retires this batch of ordered writes

inline uint64_t relay_pad(uint64_t len) { return (len + 63) & ~63ull; }

// Positions are monotonic byte counts, and position % ring is the offset in the ring. done is the relay's credit: every
// byte below it has been forwarded, so it may be overwritten.
struct RelayOut {
  uint64_t pos = 0;   // the ring position after the last payload
  // Room for len bytes (at most ring)? Yes: *at = the payload's position, and pos advances. A payload never wraps: one
  // that would cross the end of the ring starts at the next ring start. The bytes in [done, pos) are not forwarded yet.
  bool reserve(uint32_t len, uint64_t ring, uint64_t done, uint64_t* at) {
    const uint64_t n = relay_pad(len);
    uint64_t p = pos;
    if (p % ring + n > ring) p += ring - p % ring;
    if (done != pos && p + n - done > ring) return false;
    *at = p;
    pos = p + n;
    return true;
  }
};

// The relay-in QP receives payload notices and sends ring credits on one CQ. A receive backlog can therefore hide
// credit CQEs. Keep the send queue bounded while the relay continues polling that CQ; `done` may advance meanwhile,
// and the caller coalesces it into the next credit write. A signaled completion retires its whole ordered batch.
struct RelayCreditQueue {
  unsigned capacity = 0, in_flight = 0, unsignaled = 0;
  std::deque<unsigned> batches;

  bool available() const { return in_flight < capacity; }
  bool signal_next() const {
    return unsignaled + 1 >= kRelayCreditSignal || in_flight + 1 >= capacity;
  }
  void posted(bool signaled) {
    ++in_flight;
    ++unsignaled;
    if (signaled) { batches.push_back(unsignaled); unsignaled = 0; }
  }
  bool completed() {
    if (batches.empty() || batches.front() > in_flight) return false;
    in_flight -= batches.front();
    batches.pop_front();
    return true;
  }
};

// Both the ordinary poll and the queue-full drain consume the same signaled forward completions. Keep their credit
// advance identical: a payload completion's wr_id is the first ring position the origin may reuse.
inline void relay_forward_completed(int& in_flight, std::deque<int>& batches, uint64_t wr_id, uint64_t& done) {
  if (!batches.empty()) { in_flight -= batches.front(); batches.pop_front(); }
  if (wr_id) done = wr_id;
}

}  // namespace g6
