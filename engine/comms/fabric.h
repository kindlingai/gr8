// g6 fabric: NR-rank (six or four) one-shot all-reduce over raw ibverbs with the GPU in the loop (RUNTIME.md §3).
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
  size_t slot_bytes = 0;          // max payload bytes per collective (tail is appended)
  int ncoll = 0;                  // collectives per step (slot count per parity)
  int port = 18600;               // TCP bootstrap port on rank 0 (uses port..port+3)
  int proxy_cpu = -1;             // pin the proxy thread to this CPU (-1 = no pinning)
  // variable layout (cplayout.h; J4 CP fabric): non-empty -> ncoll = vcap.size(), slot_bytes ignored, per-collective
  // capacity vcap[c] and destination vdst_k[c] (0 = all peers, k = peer (rank + k) % nr only). Empty = uniform layout.
  std::vector<uint32_t> vcap;
  std::vector<uint8_t> vdst_k;
  // bulk mode (J4 CP prefill staging, BUILD-CPE step 4d): bulk_bytes > 0 -> no collectives (ncoll / slot_bytes / vcap
  // ignored); the registered region is [bulk | flags [bulk_flags][nr] x {val, aux} u64 | doorbell [bulk_flags] u64 |
  // aux_out [bulk_flags] u64 | error], the same layout on every rank, so a range of my bulk area lands at the same offset
  // of a peer's (identity placement)
  size_t bulk_bytes = 0;
  int bulk_flags = 0;
  int xtrace = 0;                 // 1 = stage-0 trace records on this fabric when env G6_XTRACE=1 (xtrace.h); uniform layout only
};

// bulk mode kernel view (pinned host, device-mapped). flags[(f * nr + p) * 2 + {0, 1}] = {val, aux} written by peer p as
// two ordered 8-B writes, aux first, then val (my own slot unused): read val (acquire), then aux. aux = the sender's
// aux_out[f] when the job was posted (a payload checksum in G6_CP_PF_VERIFY mode, else whatever the GPU left there)
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
  unsigned long long* xt = nullptr;   // G6_XTRACE stage-0 GPU records (device memory, xtrace.h); null = off (default)
};

// seq for collective c of step s (unique, never 0)
#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned long long coll_seq(unsigned long long step, int c, int ncoll) { return step * (unsigned long long)ncoll + c + 1; }

class Fabric {
 public:
  Fabric() = default;
  ~Fabric();
  void init(const FabricConfig& cfg);
  FabricDev dev() const { return dev_; }
  int rank() const { return cfg_.rank; }
  int nranks() const { return (int)cfg_.ips.size(); }
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
  // skip: collectives the proxy neither waits for nor sends this step (G6_PF_OWN=3 has no collective 0)
  // mask / skip bits address collectives 0..31 only; bit c >= 32 reads as 0 (fabric.cpp bit()).
  // p2p: reduce-scatter collectives (bits also set in mask) whose slice goes to `peer` only; the other peers get no
  // write (G6_PF_OWN=4: the attention partial of the counterpart's rows, collective 0)
  struct Rs { unsigned mask = 0; size_t ss = 0; std::vector<uint32_t> bytes; unsigned skip = 0; unsigned p2p = 0; int peer = -1; };
  void enqueue_step(unsigned long long step, const std::vector<uint32_t>& sizes, const Rs& rs);
  // bulk mode: one proxy job, in enqueue order: wait doorbell[db] >= seq; if wait_flag >= 0 wait flags[wait_flag][p] >=
  // wait_min for every peer p; RDMA-write every (offset, bytes) range of my bulk area to every peer at the same offset;
  // then (post_flag >= 0) write aux_out[post_flag] then post_val into flags[post_flag][me] of every peer (two 8-B inline
  // writes on the same QP after the ranges; in-order RC delivery is ASSUMED, checked by G6_CP_PF_VERIFY).
  // A CQ error, a bad completion, or 60 s without a completion while a send queue is full sets the error word
  // (bulk_proxy.h codes) and stops the proxy.
  // wait_flag compares the val word. Ranges: offset and bytes multiples of 8.
  // peer: -1 = every peer (as above); k = peer k only: the wait_flag spin reads only flags[wait_flag][k], and the ranges
  // and the post_flag writes go to k only (G6_PF_QP_ROLL: the pair partner's half image, qp_wire.h)
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
  // G6_XTRACE: write the GPU + proxy records and the per-QP WC error counters (stop_proxy calls it; once per fabric)
  void xtrace_dump(const char* why);

  struct Impl;
 private:
  FabricConfig cfg_;
  FabricDev dev_{};
  Impl* impl_ = nullptr;
};

}  // namespace g6
