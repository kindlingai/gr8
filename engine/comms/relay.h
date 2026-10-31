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

// One diagonal write as the origin posts it (fabric.cpp Impl::post) and the relay forwards it: a list of segments, in post order.
// inl: the bytes ride inside the notice SEND (at most inline_max); else a ring chunk (RDMA write into the relay ring + a notice).
// Every segment lands at its off, in this order (the relay forwards notices in arrival order on one QP). The last segment takes
// the write's wr_id and signaled flag.
// nsge 1: the payload [src, len) (len includes the tail when the tail follows the payload): inline if len <= inline_max, else
//   chunks of kRelayChunk. nsge 2 (a flat reduce-scatter slice, Fabric::Rs::flat): payload sge 0 as nsge 1 does it but never last, then
//   the 8-byte tail word sge 1 as one inline segment at off + len0, last.
struct RelaySeg { uint64_t src; uint32_t len; uint64_t off; bool inl; bool last; };
constexpr int kRelayPlanMax = 4097;   // a 32-bit payload in 1 MiB chunks (<= 4096) + the tail: every valid write fits
inline int relay_plan(uint64_t off, const uint64_t* src, const uint32_t* len, int nsge, uint32_t inline_max, RelaySeg* out, int max) {
  if (nsge < 1 || nsge > 2 || (nsge == 2 && len[1] > inline_max)) return -1;
  int n = 0;
  if (len[0] <= inline_max) { if (n >= max) return -1; out[n++] = RelaySeg{src[0], len[0], off, true, false}; }
  else
    for (uint32_t at = 0; at < len[0]; at += kRelayChunk) {
      if (n >= max) return -1;
      out[n++] = RelaySeg{src[0] + at, len[0] - at < kRelayChunk ? len[0] - at : kRelayChunk, off + at, false, false};
    }
  if (nsge == 2) { if (n >= max) return -1; out[n++] = RelaySeg{src[1], len[1], off + len[0], true, false}; }
  out[n - 1].last = true;
  return n;
}
// the send WRs a plan puts on the queue: a ring chunk is a write + a notice, an inline segment one notice
inline int relay_plan_wrs(const RelaySeg* s, int n) { int w = 0; for (int i = 0; i < n; ++i) w += s[i].inl ? 1 : 2; return w; }

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
