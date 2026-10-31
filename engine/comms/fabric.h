// gr8 fabric: NR-rank (six or four) one-shot all-reduce over raw ibverbs with the GPU in the loop.
// Buffers are cudaHostAlloc'd (pinned, device-mapped) and registered on both rails: the GPU writes partials and
// reads peers' partials directly; a host proxy thread posts the RDMA writes when the GPU rings a doorbell.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "cplayout.h"

namespace g6 {

struct FabricConfig {
  int rank = -1;
  std::vector<std::string> ips;   // rail 0 addresses, index = rank (also used for the TCP bootstrap)
  std::vector<std::string> ips2;  // rail 1 addresses (empty = single rail)
  // dual-rail striping (uniform layout, ips2 set, not ring/bulk): a payload write of >= stripe_min bytes goes out as two
  // halves, the first on a second QP over the other rail, the second (+ tail) on the peer's QP; the proxy posts the
  // tail write only after the first half completed, so the tail still means "the whole payload landed". 0 = off.
  // A three-rank ring stripes too: the second QP is on the same port's device on the other PCIe root (no ips2).
  size_t stripe_min = 0;
  size_t slot_bytes = 0;          // max payload bytes per collective (tail is appended)
  // slot runs (uniform layout only, mesh or ring): collectives [0, run1) hold slot_bytes, [run1, run2) slot_bytes1,
  // [run2, ncoll) slot_bytes2, each slot round_up(bytes + 8, 256), the runs contiguous in every parity and recv block
  // (the fab_* slot helpers below). run1 = 0: one run. DS41 prefill uses runs for its large per-layer slots.
  int run1 = 0, run2 = 0;
  size_t slot_bytes1 = 0, slot_bytes2 = 0;
  // a 4th run: with run3 > 0, [run2, run3) hold slot_bytes2 and [run3, ncoll) slot_bytes3 (DS41 prefill's all-gathers)
  int run3 = 0;
  size_t slot_bytes3 = 0;
  // slot pool (needs run1 > 0): run 0 [0, run1) and run 3 [run3, ncoll) keep only `pool` slots each, and collective c uses
  // slot c mod pool of its run. The proxy posts a pooled c only after every peer's tail of its credit collective is in
  // (fab_pool_credits; error 0x76). Run 3 may hold one collective more than run 0. 0 = one slot per collective;
  // otherwise 2 <= pool <= each pooled run's length.
  int pool = 0;
  // the direct QPs take 2 send SGEs, so a flat reduce-scatter slice (Rs::flat) sends its payload and tail word in one write
  bool sge2 = false;
  // 1 = the recv area has no slot for this rank's own contribution (FabricDev::compact). 0 = nr slots per collective, the
  // layout the hand-indexed readers (prefill.cuh, flash_prefill.cuh) assume.
  bool compact = false;
  int ncoll = 0;                  // collectives per step (slot count per parity)
  int port = 18600;               // TCP bootstrap port on rank 0 (uses port..port+3)
  int proxy_cpu = -1;             // pin the proxy thread to this CPU (-1 = no pinning)
  // variable layout (cplayout.h; CP fabric): non-empty -> ncoll = vcap.size(), slot_bytes ignored, per-collective
  // capacity vcap[c] and destination vdst_k[c] (0 = all peers, k = peer (rank + k) % nr only). Empty = uniform layout.
  std::vector<uint32_t> vcap;
  std::vector<uint8_t> vdst_k;
  // bulk mode (CP prefill staging): bulk_bytes > 0 -> no collectives (ncoll / slot_bytes / vcap
  // ignored); the registered region is [bulk | flags [bulk_flags][nr] x {val, aux} u64 | doorbell [bulk_flags] u64 |
  // aux_out [bulk_flags] u64 | error], the same layout on every rank, so a range of my bulk area lands at the same offset
  // of a peer's (identity placement)
  size_t bulk_bytes = 0;
  int bulk_flags = 0;
  int xtrace = 0;                 // 1 = stage-0 trace records on this fabric when env GR8_XTRACE=1 (xtrace.h); uniform layout only
  // switchless ring (fabric.cpp, RelayHdr): QPs on four local devices to the two cabled neighbours, and at TP=4 the
  // diagonal relayed by the next rank. ips then only says where the TCP bootstrap goes (the LAN).
  bool ring = false;
  std::vector<std::string> ring_devs;  // relay-out, direct-next, relay-in, direct-prev (empty = the GB10 CX-7 names)
  int relay_cpu = -1;                  // pin the relay thread to this CPU (-1 = no pinning)
};

// bulk mode kernel view (pinned host, device-mapped). flags[(f * nr + p) * 2 + {0, 1}] = {val, aux} written by peer p as
// two ordered 8-B writes, aux first, then val (my own slot unused): read val (acquire), then aux. aux = the sender's
// aux_out[f] when the job was posted (a payload checksum in GR8_CP_PF_VERIFY mode, else whatever the GPU left there)
struct BulkDev {
  char* bulk;
  unsigned long long* flags;     // [bulk_flags][nr][2]
  unsigned long long* doorbell;  // [bulk_flags] GPU -> proxy
  unsigned long long* aux_out;   // [bulk_flags] GPU -> proxy (read after the doorbell)
  unsigned int* error;
  int rank, nr, nflags;
};

// Plain-old-data view handed to kernels. All pointers are device-accessible pinned host memory.
struct FabricDev {
  char* send;                  // [2 parity][ncoll][stride]
  char* recv;                  // [2 parity][ncoll][nr][stride]; compact: [2 parity][ncoll][nr - 1][stride] (no slot for my own rank)
  unsigned long long* doorbell;  // [ncoll] GPU writes seq here when send slot c is complete
  unsigned int* error;         // nonzero = a kernel timed out waiting (code identifies where)
  int rank, nr, ncoll;
  size_t stride;               // slot stride = round_up(slot_bytes + 8, 256); tail = u64 seq at [payload_bytes]
                               // (0 for a variable-layout fabric: use Fabric::vslot). With slot runs: run 0's stride only
  unsigned long long* xt = nullptr;   // GR8_XTRACE stage-0 GPU records (device memory, xtrace.h); null = off (default)
  // slot runs (FabricConfig::run1): collectives [c1, c2) have stride1, [c2, c3) stride2, [c3, ncoll) stride3.
  // c1 = c2 = c3 = INT_MAX: one run. c3 = INT_MAX: no 4th run.
  int c1 = 0x7fffffff, c2 = 0x7fffffff;
  size_t stride1 = 0, stride2 = 0;
  int c3 = 0x7fffffff;
  size_t stride3 = 0;
  // 1 = the recv area has no slot for its owner's rank (FabricConfig::compact). Nobody writes that slot: the proxy skips
  // p == me and glue::peer_slot reads this rank's contribution from its send slot.
  int compact = 0;
  int pool = 0;                // FabricConfig::pool
};

// A set of collectives (Fabric::Rs mask / skip / p2p / flat): bit c for c in [0, 128). It converts from a 32-bit
// `unsigned`, so callers that pass 0x5 compile unchanged. DS41's 40 layers need more than 32 collectives.
struct CollMask {
  uint64_t w[2] = {0, 0};
  CollMask() = default;
  CollMask(unsigned m) { w[0] = m; }
  static constexpr size_t kBits = 128;
  bool bit(size_t c) const { return c < kBits && (w[c >> 6] >> (c & 63) & 1u); }
  void set(size_t c) { if (c < kBits) w[c >> 6] |= uint64_t(1) << (c & 63); }
  bool any() const { return w[0] || w[1]; }
  explicit operator bool() const { return any(); }
  CollMask operator&(const CollMask& o) const { CollMask r; r.w[0] = w[0] & o.w[0]; r.w[1] = w[1] & o.w[1]; return r; }
  CollMask operator~() const { CollMask r; r.w[0] = ~w[0]; r.w[1] = ~w[1]; return r; }
  bool operator!() const { return !any(); }
};

#ifdef __CUDACC__
#define G6_FAB_HD __host__ __device__
#else
#define G6_FAB_HD
#endif
// Slot addressing of a uniform-layout fabric. Use these, not c * stride: with runs the slots differ in size, and a
// compact recv area has nr - 1 slots per collective.
// send [2 parity][ncoll slots], recv [2 parity][ncoll][nr slots of that collective's stride].
G6_FAB_HD inline size_t fab_slot_stride(const FabricDev& f, int c) { return c < f.c1 ? f.stride : c < f.c2 ? f.stride1 : c < f.c3 ? f.stride2 : f.stride3; }
// bytes of the slots of collectives [0, c) in one parity of the send area (one slot each)
G6_FAB_HD inline size_t fab_slot_base(const FabricDev& f, int c)
{
  if (f.pool) {   // run 0 and run 3 are pools of f.pool slots (fab_slot_layout checked c1 <= ncoll, pool <= each run)
    const int P = f.pool;
    if (c < f.c1) return (size_t)(c % P) * f.stride;
    size_t b = (size_t)P * f.stride;
    if (c <= f.c2) return b + (size_t)(c - f.c1) * f.stride1;
    b += (size_t)(f.c2 - f.c1) * f.stride1;
    if (c <= f.c3) return b + (size_t)(c - f.c2) * f.stride2;
    b += (size_t)(f.c3 - f.c2) * f.stride2;
    return b + (size_t)(c >= f.ncoll ? P : (c - f.c3) % P) * f.stride3;
  }
  if (c <= f.c1) return (size_t)c * f.stride;
  if (c <= f.c2) return (size_t)f.c1 * f.stride + (size_t)(c - f.c1) * f.stride1;
  if (c <= f.c3) return (size_t)f.c1 * f.stride + (size_t)(f.c2 - f.c1) * f.stride1 + (size_t)(c - f.c2) * f.stride2;
  return (size_t)f.c1 * f.stride + (size_t)(f.c2 - f.c1) * f.stride1 + (size_t)(f.c3 - f.c2) * f.stride2 + (size_t)(c - f.c3) * f.stride3;
}
// offset of my send slot of collective c, parity par (= ((par * ncoll + c) * stride with one run)
G6_FAB_HD inline size_t fab_send_off(const FabricDev& f, int par, int c) { return (size_t)par * fab_slot_base(f, f.ncoll) + fab_slot_base(f, c); }
// slots per collective in one recv area: nr, or nr - 1 when compact
G6_FAB_HD inline int fab_recv_slots(const FabricDev& f) { return f.nr - (f.compact ? 1 : 0); }
// offset in the recv area of rank `owner` of the slot rank p (p != owner) writes for collective c, parity par
// (= ((par * ncoll + c) * S + q) * stride with one run; S = fab_recv_slots, q = p, or p - (p > owner) when compact).
// The sender (fabric.cpp proxy) names the peer it writes to as owner; a reader of its own area uses fab_recv_off.
G6_FAB_HD inline size_t fab_recv_off_at(const FabricDev& f, int par, int c, int owner, int p)
{
  return fab_send_off(f, par, c) * fab_recv_slots(f) + (size_t)(f.compact && p > owner ? p - 1 : p) * fab_slot_stride(f, c);
}
// offset in my recv area (owner = f.rank) of the slot rank p writes for collective c, parity par
G6_FAB_HD inline size_t fab_recv_off(const FabricDev& f, int par, int c, int p) { return fab_recv_off_at(f, par, c, f.rank, p); }
// bytes of one recv area: both parities, fab_recv_slots slots of every collective
G6_FAB_HD inline size_t fab_recv_bytes(const FabricDev& f) { return 2 * fab_slot_base(f, f.ncoll) * (size_t)fab_recv_slots(f); }
// reduce-scatter send: slice p of an RS send slot starts at p ss and its 8-byte tail word follows it (sp + nb). In a flat
// slot (Fabric::Rs::flat: the producer wrote [rows][row] in place, so sp + nb is the next slice's first row) the tail words
// sit after the NR slices, at NR ss + 8 p. Byte offsets from the send slot.
G6_FAB_HD inline size_t fab_rs_slice_off(size_t ss, int p) { return (size_t)p * ss; }
G6_FAB_HD inline size_t fab_rs_tail_off(size_t ss, int p, uint32_t nb, bool flat, int nr) { return flat ? (size_t)nr * ss + 8 * (size_t)p : (size_t)p * ss + nb; }
// payload `bytes` + the u64 tail fits collective c's slot (Fabric::enqueue_step's check)
inline bool fab_slot_fits(const FabricDev& f, int c, size_t bytes) { return bytes % 8 == 0 && bytes + 8 <= fab_slot_stride(f, c); }
// slot runs need the uniform layout. Mesh and ring both send through fab_send_off / fab_recv_off (the ring's diagonal relays
// by the same offset).
inline bool fab_runs_allowed(const FabricConfig& cfg, bool var) { return !cfg.run1 || !var; }
// Fabric::init's slot layout of a uniform fabric: ncoll, stride and the runs of cfg (run1 = 0: one run) into d. false = the
// runs or the pool are malformed.

inline bool fab_slot_layout(const FabricConfig& cfg, int ncoll, FabricDev* d)
{
  d->ncoll = ncoll; d->stride = (cfg.slot_bytes + 8 + 255) & ~size_t(255);
  d->c1 = d->c2 = d->c3 = 0x7fffffff; d->stride1 = d->stride2 = d->stride3 = 0; d->pool = 0;
  if (!cfg.run1) return !cfg.run3 && !cfg.pool;
  // pool: 2 <= P <= run 0's length, and run 3 as long as run 0 or one longer
  if (cfg.pool && !(cfg.pool >= 2 && cfg.pool <= cfg.run1 &&
                    (!cfg.run3 || ncoll - cfg.run3 == cfg.run1 || ncoll - cfg.run3 == cfg.run1 + 1))) return false;
  const int end2 = cfg.run3 ? cfg.run3 : ncoll;   // run 2 ends at the 4th run, if any
  if (!(0 < cfg.run1 && cfg.run1 <= cfg.run2 && cfg.run2 <= end2 && end2 <= ncoll && cfg.slot_bytes1 > 0 && (cfg.run2 == end2 || cfg.slot_bytes2 > 0))) return false;
  if (cfg.run3 && !(cfg.run3 < ncoll && cfg.slot_bytes3 > 0)) return false;
  d->c1 = cfg.run1; d->c2 = cfg.run2;
  d->stride1 = (cfg.slot_bytes1 + 8 + 255) & ~size_t(255);
  d->stride2 = cfg.run2 < end2 ? (cfg.slot_bytes2 + 8 + 255) & ~size_t(255) : 0;
  if (cfg.run3) { d->c3 = cfg.run3; d->stride3 = (cfg.slot_bytes3 + 8 + 255) & ~size_t(255); }
  d->pool = cfg.pool;
  return true;
}

// the pool slot of collective c (0..P-1 run 0, P..2P-1 run 3); -1 = c has its own slot (pool off, or runs 1 / 2)
G6_FAB_HD inline int fab_pool_slot(const FabricDev& f, int c)
{
  if (!f.pool || c < 0 || c >= f.ncoll) return -1;
  if (c < f.c1) return c % f.pool;
  if (c >= f.c3) return f.pool + (c - f.c3) % f.pool;
  return -1;
}
// Slot pool credits of one proxy step. walk = the collectives the proxy posts this step, in its order (Rs::order or 0..n-1, skip bits
// removed). credit[c] = -1: c's pool slot is unused before c in this step (or c is not pooled). Else credit[c] = the first pooled
// collective posted after u, the previous user of c's pool slot: each peer sends it only after it consumed u (it reads the slot of u
// before it produces the next pooled collective: RS(j) -> AG(j) -> RS(j+1), or decode c -> c+1), so when that peer's tail of
// credit[c] is in my recv area, the peer's slot that c overwrites is free. A credit that is not posted before c (P = 1) is an error:
// returns false. ncoll = the fabric's collective count (credit has ncoll entries).
inline bool fab_pool_credits(const FabricDev& f, const std::vector<int>& walk, std::vector<int>* credit)
{
  credit->assign((size_t)f.ncoll, -1);
  if (!f.pool) return true;
  std::vector<int> last((size_t)2 * f.pool, -1);    // pool slot -> walk index of its last user
  std::vector<int> pooled_at(walk.size(), 0);       // walk index -> 1 if pooled
  for (size_t k = 0; k < walk.size(); k++) {
    const int c = walk[k], s = fab_pool_slot(f, c);
    if (c < 0 || c >= f.ncoll) return false;
    if (s < 0) continue;
    pooled_at[k] = 1;
    if (last[(size_t)s] >= 0) {
      int cr = -1;
      for (size_t j = (size_t)last[(size_t)s] + 1; j < k; j++) if (pooled_at[j]) { cr = walk[j]; break; }
      if (cr < 0) return false;                     // nothing pooled between two users of one slot: no credit can exist
      (*credit)[(size_t)c] = cr;
    }
    last[(size_t)s] = (int)k;
  }
  return true;
}

// seq for collective c of step s (unique, never 0)
#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned long long coll_seq(unsigned long long step, int c, int ncoll) { return step * (unsigned long long)ncoll + c + 1; }

// Verbs objects and pinned bytes held by the live mesh fabrics of this process (each counter goes up in init and back
// down in close). A group created and dropped (fabric_pool.h) must leave every counter where it was. ring_fabrics counts
// ring fabrics, which are never freed.
struct FabricLive { long fabrics = 0, qps = 0, mrs = 0, cqs = 0, pds = 0, ctxs = 0, pinned_bytes = 0, ring_fabrics = 0; };
FabricLive fabric_live();

class Fabric {
 public:
  Fabric() = default;
  ~Fabric();
  void init(const FabricConfig& cfg);
  // Stop the proxy and free everything init made (mesh): QPs, MRs, CQs, PDs, device contexts, the pinned region. The
  // caller first lets the proxy post what it was given (wait_proxy_idle; close waits up to 60 s itself and says so if
  // jobs were left) and runs a barrier on a fabric of the group, so no peer still writes into this one. The Fabric can
  // then be init'ed again; enqueue, barrier and the proxy calls abort before that, and dev() is empty. A thread that
  // watches dev().error must stop before close (the word is in the freed region). A ring fabric (never regrouped) only stops its proxy and
  // cannot be init'ed again: its relay is shared by the process. The destructor only stops the proxy, as before (at
  // exit a peer may still be writing).
  void close();
  FabricDev dev() const { return dev_; }
  int rank() const { return cfg_.rank; }
  int nranks() const { return (int)cfg_.ips.size(); }
  int port() const { return cfg_.port; }   // bootstrap base (FabricConfig::port), for messages
  // variable-layout fabric only: kernel view of collective c (both parities)
  VSlotDev vslot(int c) const;
  const VLayout& vlayout() const;

  // Proxy control. enqueue_step() hands the proxy the payload sizes of collectives 0..n-1 of `step`;
  // the proxy then waits for each doorbell in order and posts the writes (numeric order, or Rs::order). Call before
  // launching the step's GPU work.
  void start_proxy();
  void stop_proxy();
  void enqueue_step(unsigned long long step, const std::vector<uint32_t>& sizes);
  // Variable layout: sizes[c] <= vcap[c] (multiple of 8); Rs must be empty. A step may enqueue fewer collectives
  // than ncoll (the proxy handles 0..sizes.size()-1 in order, or the collectives Rs::order names).
  // reduce-scatter form: collectives whose bit is set in rs_mask send slice p (at send + p * ss, rs_bytes[p] bytes)
  // to peer p only, into p's recv slot [c][me] at offset 0, tail right after it; sizes[c] is ignored for them.
  // skip: collectives the proxy neither waits for nor sends this step (GR8_PF_OWN=3 has no collective 0)
  // mask / skip / p2p are CollMask: collectives 0..127 (c >= 128 reads as 0); a plain `unsigned` converts (bits 0..31).
  // p2p: reduce-scatter collectives (bits also set in mask) whose slice goes to `peer` only; the other peers get no
  // write (GR8_PF_OWN=4: the attention partial of the counterpart's rows, collective 0)
  // order: the collectives the proxy handles this step, in the order it waits for and sends them (fabric_order.h). Empty =
  // 0..sizes.size()-1 in numeric order. Use it when the GPU publishes in another order than c (DS41 with Engram: the
  // Engram reduce runs mid-step, so the proxy must wait for it there, not after the head). Each entry
  // < sizes.size() and distinct, else enqueue_step aborts; a collective left out is neither waited for nor sent (a prefill
  // chunk with no head: sizes keeps one entry per c, order leaves the head out). The skip bits still apply (c < 128). Every
  // rank passes the same order; the doorbell sequence number stays coll_seq(step, c, ncoll).
  struct Rs { CollMask mask; size_t ss = 0; std::vector<uint32_t> bytes; CollMask skip; CollMask p2p; int peer = -1;
              std::vector<int> order;
              // needs FabricConfig::sge2: RS collectives whose send slot is flat (the producer wrote [rows][row] in place):
              // slice p = [p ss, p ss + bytes[p]) with ss = the slice payload (no tail gap); its tail word sits after the NR slices at
              // NR ss + 8 p, and travels as the second SGE of the slice's write (the receiver's layout is unchanged). flat is a subset of mask.
              CollMask flat; };
  void enqueue_step(unsigned long long step, const std::vector<uint32_t>& sizes, const Rs& rs);
  // bulk mode: one proxy job, in enqueue order: wait doorbell[db] >= seq; if wait_flag >= 0 wait flags[wait_flag][p] >=
  // wait_min for every peer p; RDMA-write every (offset, bytes) range of my bulk area to every peer at the same offset;
  // then (post_flag >= 0) write aux_out[post_flag] then post_val into flags[post_flag][me] of every peer (two 8-B inline
  // writes on the same QP after the ranges; in-order RC delivery is ASSUMED, checked by GR8_CP_PF_VERIFY).
  // A CQ error, a bad completion, or 60 s without a completion while a send queue is full sets the error word
  // (bulk_proxy.h codes) and stops the proxy.
  // wait_flag compares the val word. Ranges: offset and bytes multiples of 8.
  // peer: -1 = every peer (as above); k = peer k only: the wait_flag spin reads only flags[wait_flag][k], and the ranges
  // and the post_flag writes go to k only (GR8_PF_QP_ROLL: the pair partner's half image, qp_wire.h)
  struct BulkJob { int db = -1; unsigned long long seq = 0; int wait_flag = -1; unsigned long long wait_min = 0;
                   std::vector<std::pair<uint64_t, uint32_t>> ranges; int post_flag = -1; unsigned long long post_val = 0;
                   int peer = -1; };
  void enqueue_bulk(BulkJob job);
  BulkDev bulk_dev() const;
  // wait until the proxy has posted every job enqueued so far. false (message printed) = the proxy exited with work
  // pending, or pending did not move for stall_s (error word set to kDrainStall if it was 0); the unfinished work is not
  // counted as done. Callers that care must check it (g6m4 CP staging drain does).
  bool wait_proxy_idle(double stall_s = 180.0);
  // host-side barrier over TCP (slow; setup/teardown only)
  void barrier();
  // GR8_XTRACE: write the GPU + proxy records and the per-QP WC error counters (stop_proxy calls it; once per fabric)
  void xtrace_dump(const char* why);

  struct Impl;
 private:
  FabricConfig cfg_;
  FabricDev dev_{};
  Impl* impl_ = nullptr;
};

}  // namespace g6
