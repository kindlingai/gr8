// gr8 fabric bulk-mode proxy logic that does not touch verbs or CUDA, so a host test can drive it with a fake CQ and a
// fake clock (tools/cp/proxytest.cpp). fabric.cpp adapts ibv_poll_cq / steady_clock to these templates.
#pragma once
#include <cstdint>
#include <deque>
#include <vector>

namespace g6 {

// error words the bulk proxy / drain publish (fabric.h BulkDev::error)
enum : unsigned {
  kBulkWcError   = 0x71000001u,  // a completion with status != success
  kBulkCqError   = 0x71000002u,  // ibv_poll_cq < 0
  kBulkCqStall   = 0x71000003u,  // no completion for stall_s while a peer's send queue is full
  kDrainStall    = 0x74000001u,  // wait_proxy_idle: pending did not move for stall_s with the proxy alive
  kDrainDead     = 0x74000002u,  // wait_proxy_idle: the proxy exited with work pending
};
// bulk_room result: 0 = room, kBulkStopped = stop requested or the error word was already set (no new code to publish)
constexpr unsigned kBulkStopped = 1u;

struct BulkCompl { uint64_t wr_id; bool ok; };

// fabric.h BulkJob::peer: does a job with this peer field address peer p? (-1 = every peer)
inline bool bulk_to(int job_peer, int p) { return job_peer < 0 || job_peer == p; }

// Reap every completion available now (non-blocking): each signaled completion with wr_id q retires the oldest batch of q.
// Returns kBulkCqError (poll < 0), kBulkWcError (bad status; *bad_peer = its wr_id), else 0; *got = completions reaped.
template <class Poll>
unsigned bulk_reap(int nrails, std::vector<int>& inflight, std::vector<std::deque<int>>& batches, Poll&& poll, int* got, int* bad_peer) {
  const int nr = (int)inflight.size();
  *got = 0;
  for (int r = 0; r < nrails; r++) {
    BulkCompl wc[32]; int n;
    while ((n = poll(r, wc, 32)) > 0) {
      *got += n;
      for (int i = 0; i < n; i++) {
        if (!wc[i].ok) { *bad_peer = (int)wc[i].wr_id; return kBulkWcError; }
        const int q = (int)wc[i].wr_id;
        if (q >= 0 && q < nr && !batches[q].empty()) { inflight[q] -= batches[q].front(); batches[q].pop_front(); }
      }
    }
    if (n < 0) return kBulkCqError;
  }
  return 0;
}

// Drain completions until peer p has fewer than `cap` WRs in flight. poll(rail, out, max) -> n completions (< 0 = CQ
// error); halt() -> stop requested or error word set; now() -> seconds (monotonic). Each signaled completion with wr_id q
// retires the oldest batch of q. Fails closed: a CQ error, a bad completion, or stall_s without any completion returns
// the code for the caller to publish, instead of polling forever.
template <class Poll, class Halt, class Now>
unsigned bulk_room(int p, int cap, int nrails, std::vector<int>& inflight, std::vector<std::deque<int>>& batches,
                   Poll&& poll, Halt&& halt, Now&& now, double stall_s) {
  double last = now();
  while (inflight[p] >= cap) {
    int got = 0, bad = -1;
    if (const unsigned e = bulk_reap(nrails, inflight, batches, poll, &got, &bad)) return e;
    if (got) last = now();
    if (halt()) return kBulkStopped;
    if (now() - last > stall_s) return kBulkCqStall;
  }
  return 0;
}

// Host drain: wait until pending == 0. Returns 0, or kDrainDead (the proxy exited while work was pending: that work was
// not done and is not counted as done) or kDrainStall (pending did not move for stall_s). pending() / exited() / now()
// are read each round; yield() between rounds.
template <class Pending, class Exited, class Now, class Yield>
unsigned drain_wait(Pending&& pending, Exited&& exited, Now&& now, Yield&& yield, double stall_s) {
  long last_n = pending();
  double last = now();
  for (;;) {
    const long n = pending();
    if (n <= 0) return 0;
    if (exited()) return pending() > 0 ? kDrainDead : 0;
    if (n != last_n) { last_n = n; last = now(); }
    else if (now() - last > stall_s) return kDrainStall;
    yield();
  }
}

// Flag publication (fabric.h BulkJob::post_flag): the 16-B slot {val, aux} is written as TWO ordered 8-B RDMA writes on
// the same QP, aux first, then val. The receiver acquires val and only then reads aux, so val >= v implies the aux of
// that post has landed (in-order RC writes on an MR without relaxed ordering); no 16-B atomicity is assumed.
struct FlagWrite { int word; unsigned long long value; };  // word 0 = val (seq), 1 = aux
inline void flag_writes(unsigned long long val, unsigned long long aux, FlagWrite out[2]) {
  out[0] = {1, aux};
  out[1] = {0, val};
}

}  // namespace g6
