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

// The notice SEND that follows a ring payload (kind 0: len bytes at ring position pos) or carries a short write itself
// (kind 1). The legacy ring writes kind exactly 0 or 1. A hybrid route (FabricConfig::hybrid) packs the route into it:
// kind = k | dst << 8 | src << 16, bits 24..31 zero, so a miswired relay fails at once instead of writing into the wrong
// rank's region. Every kind test of the relay uses the decoded k.
struct RelayHdr { uint64_t off, pos; uint32_t len, kind; };
static_assert(sizeof(RelayHdr) == 24, "RelayHdr is 24 bytes: kRelayInline = 64 - 24 depends on it");
constexpr int kRelayMaxRanks = 8;
constexpr unsigned kRelayMaxContexts = 128;   // relay contexts per process (6 TP8 Fabrics x 10 routes on a via rank = 60)
inline uint32_t relay_kind(uint32_t k, int dst, int src) { return k | (uint32_t)dst << 8 | (uint32_t)src << 16; }

// Check a received header against the context it arrived on. hybrid: the packed route must be exactly (src, dst) of the
// context; legacy: kind is 0 or 1. Then: k = 1 carries at most `inline_max` bytes; k = 0 is 1..chunk bytes. byte_len (the
// received SEND length) is checked in hybrid mode only: header + inline bytes, or the header alone. *k = the decoded kind.
inline bool relay_header_ok(const RelayHdr& h, bool hybrid, int src, int dst, uint32_t byte_len, uint32_t inline_max,
                            uint32_t chunk, uint32_t* k) {
  if (!hybrid) {
    *k = h.kind;
    return !(h.kind > 1 || (h.kind == 1 && h.len > inline_max));
  }
  const uint32_t kk = h.kind & 0xff, d = h.kind >> 8 & 0xff, s = h.kind >> 16 & 0xff;
  *k = kk;
  if (h.kind >> 24 || kk > 1) return false;
  if (d >= (uint32_t)kRelayMaxRanks || s >= (uint32_t)kRelayMaxRanks || d == s) return false;
  if ((int)d != dst || (int)s != src) return false;
  if (kk == 1) return h.len <= inline_max && byte_len == sizeof(RelayHdr) + h.len;
  return h.len > 0 && h.len <= chunk && byte_len == sizeof(RelayHdr);
}

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
// A relay's forward queue that stays full with no completion for this long fails its Fabric (error 0x75000002), so the
// one relay thread returns to the other contexts. Not a timeout of healthy traffic: a forward completes on the
// destination NIC's ACK alone.
inline double relay_drain_stall_s = 10.0;

inline void relay_forward_completed(int& in_flight, std::deque<int>& batches, uint64_t wr_id, uint64_t& done) {
  if (!batches.empty()) { in_flight -= batches.front(); batches.pop_front(); }
  if (wr_id) done = wr_id;
}

}  // namespace g6
