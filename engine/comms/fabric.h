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
  char* recv;                  // [2 parity][ncoll][nr][stride]
  unsigned long long* doorbell;  // [ncoll] GPU writes seq here when send slot c is complete
  unsigned int* error;         // nonzero = a kernel timed out waiting (code identifies where)
  int rank, nr, ncoll;
  size_t stride;               // slot stride = round_up(slot_bytes + 8, 256); tail = u64 seq at [payload_bytes]
                               // (0 for a variable-layout fabric: use Fabric::vslot)
  unsigned long long* xt = nullptr;   // GR8_XTRACE stage-0 GPU records (device memory, xtrace.h); null = off (default)
};

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
  // the proxy then waits for each doorbell in order and posts the writes. Call before launching the step's GPU work.
  void start_proxy();
  void stop_proxy();
  void enqueue_step(unsigned long long step, const std::vector<uint32_t>& sizes);
  // Variable layout: sizes[c] <= vcap[c] (multiple of 8); Rs must be empty. A step may enqueue fewer collectives
  // than ncoll (the proxy handles 0..sizes.size()-1 in order).
  // reduce-scatter form: collectives whose bit is set in rs_mask send slice p (at send + p * ss, rs_bytes[p] bytes)
  // to peer p only, into p's recv slot [c][me] at offset 0, tail right after it; sizes[c] is ignored for them.
  // skip: collectives the proxy neither waits for nor sends this step (GR8_PF_OWN=3 has no collective 0)
  // mask / skip bits address collectives 0..31 only; bit c >= 32 reads as 0 (fabric.cpp bit()).
  // p2p: reduce-scatter collectives (bits also set in mask) whose slice goes to `peer` only; the other peers get no
  // write (GR8_PF_OWN=4: the attention partial of the counterpart's rows, collective 0)
  struct Rs { unsigned mask = 0; size_t ss = 0; std::vector<uint32_t> bytes; unsigned skip = 0; unsigned p2p = 0; int peer = -1; };
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
