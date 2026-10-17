#!/usr/bin/env python3
"""Compile the production relay loop against mocked verbs and force its two credit-pressure edge cases."""
from pathlib import Path
import subprocess
import tempfile


repo = Path(__file__).resolve().parents[3]
source = (repo / 'engine/comms/fabric.cpp').read_text()
ctx = source[source.index('constexpr int kRelayRecv'):source.index('int ring_gid_index')]
loop = source[source.index('void relay_loop(int cpu)'):source.index('struct Job')]
loop_production = loop
# The real process leaves a failed context parked forever; make an integration-test failure observable.
loop = loop.replace('if (*c.err) continue;', 'if (*c.err) throw std::runtime_error("relay failed");')

preamble = r'''
#include "relay.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
using namespace g6;
void progress_cpu_add(int) {} void progress_tid_add(int) {}
struct QPInfo {};
enum ibv_wc_status { IBV_WC_SUCCESS };
enum { IBV_WC_RECV=1, IBV_WR_RDMA_WRITE=2, IBV_SEND_INLINE=4, IBV_SEND_SIGNALED=8,
       IBV_QP_STATE=16, IBV_QPS_RTS=3 };
struct ibv_wc { uint64_t wr_id; ibv_wc_status status; int opcode; uint32_t byte_len = 0; };
struct ibv_cq { std::deque<ibv_wc> q; bool credit=false; };
struct ibv_qp { ibv_cq* cq; bool credit; uint32_t qp_num; };
struct ibv_qp_attr { int qp_state=IBV_QPS_RTS; };
struct ibv_qp_init_attr {};
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
    uint64_t wr_id=0; ibv_sge* sg_list=nullptr; int num_sge=0, opcode=0, send_flags=0;
    struct { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr;
};
struct ibv_recv_wr { uint64_t wr_id; ibv_sge* sg_list; int num_sge; };
const char* ibv_wc_status_str(ibv_wc_status) { return "success"; }
int ibv_query_qp(ibv_qp*, ibv_qp_attr* at, int, ibv_qp_init_attr*) { at->qp_state=IBV_QPS_RTS; return 0; }
int credit_outstanding=0, credit_posts=0, credit_reaped=0, max_credit_outstanding=0;
int forward_drain_reaped=0; bool post_failed=false; unsigned error=0;
int ibv_poll_cq(ibv_cq* cq, int max, ibv_wc* out) {
    if (post_failed) throw std::runtime_error("credit post reached a full SQ");
    if (cq->credit && credit_posts > (int)kRelayCredit && credit_outstanding == 1 && cq->q.empty())
        throw std::runtime_error("expected success");
    if (!cq->credit && max > 1) return 0; // force forward completions through the queue-full drain
    int n=0;
    while (n<max && !cq->q.empty()) {
        out[n]=cq->q.front(); cq->q.pop_front();
        if (cq->credit && out[n].opcode != IBV_WC_RECV) { credit_outstanding-=(int)kRelayCreditSignal; ++credit_reaped; }
        if (!cq->credit) ++forward_drain_reaped;
        ++n;
    }
    return n;
}
int ibv_post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
    if (qp->credit) {
        ++credit_posts;
        if (credit_outstanding==(int)kRelayCredit) { post_failed=true; errno=ENOMEM; return ENOMEM; }
        ++credit_outstanding;
        if (credit_outstanding > max_credit_outstanding) max_credit_outstanding=credit_outstanding;
    }
    if (wr->send_flags & IBV_SEND_SIGNALED) qp->cq->q.push_back({wr->wr_id, IBV_WC_SUCCESS, 0});
    return 0;
}
int ibv_post_recv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) { return 0; }
'''

main = r'''
int main() {
    ibv_cq incoming, forward; incoming.credit=true;
    ibv_qp ri{&incoming,true,17}, fwd{&forward,false,18};
    std::vector<char> recv((size_t)kRelayRecv*kRelayBuf), ring(1<<24);
    for (int i=0; i<kRelayRecv; ++i) {
        RelayHdr h{0,uint64_t(i)*1024,1024,0};
        memcpy(recv.data()+(size_t)i*kRelayBuf,&h,sizeof h);
        incoming.q.push_back({uint64_t(i),IBV_WC_SUCCESS,IBV_WC_RECV});
    }
    RelayCtx c; c.ri=&ri; c.ri_cq=&incoming; c.fwd=&fwd; c.fwd_cq=&forward;
    c.rbuf=recv.data(); c.ring=ring.data(); c.err=&error; c.rank=1;
    c.credits.capacity=c.credit_sq_capacity=kRelayCredit; c.ri_dev="mock0";
    g_relay[0]=&c; g_nrelay=1;
    try { relay_loop(-1); }
    catch (const std::runtime_error& e) {
        if (strcmp(e.what(),"expected success")) { fprintf(stderr,"relay_loop_test: %s\n",e.what()); return 1; }
        if (error || post_failed || credit_posts != (int)kRelayCredit+1 ||
                credit_reaped != (int)(kRelayCredit/kRelayCreditSignal) ||
                max_credit_outstanding != (int)kRelayCredit ||
                forward_drain_reaped == 0 || c.sent != c.done) return 1;
        printf("relay_loop_test: OK (%u posts, max %u in flight, %d forward-drain completions)\n",
               kRelayCredit+1, kRelayCredit, forward_drain_reaped);
        return 0;
    }
    return 1;
}
'''

# Hybrid relay contexts (FabricConfig::hybrid): many contexts on the one relay thread, the packed route header, and the
# forward-queue progress deadline. The mock completes every signaled write at once unless its CQ is stalled.
multi_preamble = r"""
#include "relay.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
using namespace g6;
void progress_cpu_add(int) {} void progress_tid_add(int) {}
struct QPInfo {};
enum ibv_wc_status { IBV_WC_SUCCESS };
enum { IBV_WC_RECV=1, IBV_WR_RDMA_WRITE=2, IBV_SEND_INLINE=4, IBV_SEND_SIGNALED=8,
       IBV_QP_STATE=16, IBV_QPS_RTS=3 };
struct ibv_wc { uint64_t wr_id; ibv_wc_status status; int opcode; uint32_t byte_len = 0; };
struct ibv_cq { std::deque<ibv_wc> q; bool stall = false; };
struct Fwd { uint64_t remote_addr, wr_id; uint32_t len; bool inl; uint64_t inline_word; };
struct ibv_qp { ibv_cq* cq; bool credit; uint32_t qp_num; std::vector<uint64_t> credits; std::vector<Fwd> fwd; };
struct ibv_qp_attr { int qp_state=IBV_QPS_RTS; };
struct ibv_qp_init_attr {};
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
    uint64_t wr_id=0; ibv_sge* sg_list=nullptr; int num_sge=0, opcode=0, send_flags=0;
    struct { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr;
};
struct ibv_recv_wr { uint64_t wr_id; ibv_sge* sg_list; int num_sge; };
const char* ibv_wc_status_str(ibv_wc_status) { return "success"; }
int ibv_query_qp(ibv_qp*, ibv_qp_attr* at, int, ibv_qp_init_attr*) { at->qp_state=IBV_QPS_RTS; return 0; }
long polls = 0, poll_limit = 0; bool (*finished)() = nullptr;
int ibv_poll_cq(ibv_cq* cq, int max, ibv_wc* out) {
    if (++polls > poll_limit) throw std::runtime_error("poll limit: no progress to completion");
    if (finished && finished()) throw std::runtime_error("finished");
    if (cq->stall) return 0;
    int n = 0;
    while (n < max && !cq->q.empty()) { out[n++] = cq->q.front(); cq->q.pop_front(); }
    return n;
}
int ibv_post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
    if (qp->credit) qp->credits.push_back(*(const uint64_t*)wr->sg_list[0].addr);
    else {
        uint64_t word = 0;
        if (wr->send_flags & IBV_SEND_INLINE) memcpy(&word, (const void*)wr->sg_list[0].addr, wr->sg_list[0].length < 8 ? wr->sg_list[0].length : 8);
        qp->fwd.push_back({wr->wr.rdma.remote_addr, wr->wr_id, wr->sg_list[0].length, (wr->send_flags & IBV_SEND_INLINE) != 0, word});
    }
    if (wr->send_flags & IBV_SEND_SIGNALED) qp->cq->q.push_back({wr->wr_id, IBV_WC_SUCCESS, 0});
    return 0;
}
int reposts = 0;
int ibv_post_recv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) { ++reposts; return 0; }
"""

multi_main = r"""
static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++fails; } } while (0)
static char ring_dummy[1];                          // ring addresses are computed, never read, by the mock
struct Route {
    RelayCtx c; ibv_cq in, fwdcq; ibv_qp ri{nullptr, true, 1, {}, {}}, fwd{nullptr, false, 2, {}, {}};
    std::vector<char> rbuf; std::vector<RelayHdr> sent; uint64_t end = 0; int notices = 0;
};
std::vector<std::unique_ptr<Route>> routes;
// a context for route src->dst on the Fabric whose error word is err; notices: payload (k 0) and inline (k 1) headers
static Route* add(int src, int dst, unsigned* err, int payloads, int inlines, bool hybrid = true) {
    auto R = std::make_unique<Route>();
    R->ri.cq = &R->in; R->fwd.cq = &R->fwdcq;
    R->rbuf.resize((size_t)kRelayRecv * kRelayBuf);
    RelayCtx& c = R->c;
    c.ri = &R->ri; c.ri_cq = &R->in; c.fwd = &R->fwd; c.fwd_cq = &R->fwdcq; c.rbuf = R->rbuf.data(); c.ring = ring_dummy;
    c.err = err; c.rank = 9; c.credits.capacity = c.credit_sq_capacity = kRelayCredit; c.ri_dev = "mock";
    c.fin_addr = 0x100000ull * (uint64_t)(routes.size() + 1); c.hybrid = hybrid; c.src = src; c.dst = dst;
    uint64_t pos = 0; int slot = 0;
    for (int i = 0; i < payloads + inlines; i++) {
        const bool inl = i % 5 == 4 && inlines > 0 ? true : i >= payloads;
        RelayHdr h{};
        uint32_t bl = sizeof(RelayHdr);
        if (inl) { h = RelayHdr{(uint64_t)i * 64, 0, 16, hybrid ? relay_kind(1, dst, src) : 1u}; bl += 16; }
        else { const uint32_t len = 4096 * (1 + i % 7); h = RelayHdr{(uint64_t)i * 64, pos, len, hybrid ? relay_kind(0, dst, src) : 0u}; pos += relay_pad(len); }
        memcpy(R->rbuf.data() + (size_t)slot * kRelayBuf, &h, sizeof h);
        if (inl) { const uint64_t word = 0xC0DE0000ull + (uint64_t)i; memcpy(R->rbuf.data() + (size_t)slot * kRelayBuf + sizeof h, &word, 8); }
        R->in.q.push_back({(uint64_t)slot, IBV_WC_SUCCESS, IBV_WC_RECV, bl});
        R->sent.push_back(h); slot = (slot + 1) % kRelayRecv;
    }
    R->end = pos; R->notices = payloads + inlines;
    routes.push_back(std::move(R));
    return routes.back().get();
}
static void reg() { for (size_t i = 0; i < routes.size(); i++) g_relay[i] = &routes[i]->c; g_nrelay = (int)routes.size(); }
static void reset() { routes.clear(); g_nrelay = 0; polls = 0; reposts = 0; }
static bool all_done() {
    for (auto& R : routes) {
        if (*R->c.err) continue;
        if (!R->in.q.empty() || (int)R->fwd.fwd.size() != R->notices || R->c.sent != R->end || R->c.done != R->end) return false;
    }
    return true;
}
static void run(long limit) {
    finished = all_done; poll_limit = limit;
    try { relay_loop(-1); } catch (const std::runtime_error& e) { if (strcmp(e.what(), "finished")) { fprintf(stderr, "%s\n", e.what()); ++fails; } }
}
// the forwards of one route: in order, at fin_addr + off, payloads by ring position and inline words intact
static void check_route(Route& R) {
    CHECK(R.fwd.fwd.size() == R.sent.size());
    for (size_t i = 0; i < R.fwd.fwd.size() && i < R.sent.size(); i++) {
        const Fwd& f = R.fwd.fwd[i]; const RelayHdr& h = R.sent[i];
        CHECK(f.remote_addr == R.c.fin_addr + h.off && f.len == h.len);
        const uint32_t k = h.kind & 0xff;
        CHECK(f.inl == (k == 1));
        if (k == 1) CHECK(f.inline_word == 0xC0DE0000ull + i);
        else CHECK(f.wr_id == h.pos + relay_pad(h.len));
    }
    for (size_t i = 1; i < R.ri.credits.size(); i++) CHECK(R.ri.credits[i] > R.ri.credits[i - 1]);   // monotonic
    CHECK(!R.ri.credits.empty() && R.ri.credits.back() == R.end);
}

// 60 contexts (6 Fabrics x 10 routes, as on a TP8 via rank) share the relay thread; every route completes on its own
static void sixty() {
    reset();
    unsigned err[6] = {};
    for (int f = 0; f < 6; f++)
        for (int k = 0; k < 10; k++) {
            const int far = 1 + k % 5, near = k < 5 ? 6 : 7;          // 5 far peers x 2 directions per Fabric
            if (k < 5) add(near, far == 5 ? 0 : far, &err[f], 300 + 10 * k, 60);
            else add(far == 5 ? 0 : far, near, &err[f], 300 + 10 * k, 60);
        }
    reg();
    run(50000000);
    for (unsigned e : err) CHECK(e == 0);
    for (auto& R : routes) check_route(*R);
    CHECK(routes.size() == 60);
    printf("  60 contexts: every route forwarded in order, credits monotonic to the ring end, no error\n");
}

// a header that names another route fails its own Fabric only; a legacy ring context still takes kinds 0/1
static void mismatch() {
    reset();
    unsigned bad = 0, good = 0, legacy = 0;
    Route* a = add(2, 3, &bad, 10, 0);
    Route* b = add(2, 4, &good, 10, 0);
    Route* l = add(-1, -1, &legacy, 10, 4, false);
    RelayHdr h; memcpy(&h, a->rbuf.data() + 3 * kRelayBuf, sizeof h);
    h.kind = relay_kind(0, 4, 2);                                     // dst 4 on the 2->3 context
    memcpy(a->rbuf.data() + 3 * kRelayBuf, &h, sizeof h);
    reg(); run(5000000);
    CHECK(bad == 0x75000000u && good == 0 && legacy == 0);
    CHECK(a->fwd.fwd.size() == 3);                                    // the three before the bad header, none after
    check_route(*b); check_route(*l);
    // the reverse case: a legacy kind on a hybrid context, and a src swap
    reset(); bad = good = 0;
    a = add(5, 1, &bad, 6, 0); b = add(1, 5, &good, 6, 0);
    memcpy(&h, a->rbuf.data(), sizeof h); h.kind = 0; memcpy(a->rbuf.data(), &h, sizeof h);
    reg(); run(5000000);
    CHECK(bad == 0x75000000u && good == 0 && a->fwd.fwd.empty());
    check_route(*b);
    reset(); bad = good = 0;
    a = add(5, 1, &bad, 6, 0); b = add(1, 5, &good, 6, 0);
    memcpy(&h, a->rbuf.data(), sizeof h); h.kind = relay_kind(0, 5, 1); memcpy(a->rbuf.data(), &h, sizeof h);
    reg(); run(5000000);
    CHECK(bad == 0x75000000u && good == 0 && a->fwd.fwd.empty());
    printf("  dst/src mismatch: error word 0x75000000 on that Fabric, the others forward on\n");
}

// a zero-progress forward CQ: context A fills its forward queue and never completes. The drain deadline fails A's Fabric
// and the one relay thread goes on to serve every other context
static void zero_progress() {
    reset();
    unsigned ea = 0, eb = 0;
    Route* A = add(6, 1, &ea, 2000, 0);
    A->fwdcq.stall = true;
    for (int k = 0; k < 9; k++) add(k < 5 ? 6 : 1 + k % 5, k < 5 ? 1 + k : 6, &eb, 400, 20);
    relay_drain_stall_s = 0.2;
    reg();
    const auto t0 = std::chrono::steady_clock::now();
    run(200000000);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ea == 0x75000002u && eb == 0);
    CHECK(A->c.inflight >= 1800 && A->fwd.fwd.size() == 1800);
    for (size_t i = 1; i < routes.size(); i++) check_route(*routes[i]);
    CHECK(s < 5.0);
    relay_drain_stall_s = 10.0;
    printf("  zero-progress forward CQ: its Fabric fails 0x75000002 after the %.1f s deadline (%.2f s run), 9 other routes complete\n", 0.2, s);
}

// Negative test, the reason credits are per route: one aggregate ring shared by two destinations, whose two forward QPs
// complete out of order. done = the wr_id of each completion, so it jumps past bytes not yet forwarded, then backwards.
static void aggregate_hazard() {
    const uint64_t A_end = relay_pad(1u << 20), B_end = A_end + relay_pad(1u << 20);   // A at [0, 1M) to d1, B after it to d2
    int inflight = 2; std::deque<int> batches{1, 1}; uint64_t done = 0;
    relay_forward_completed(inflight, batches, B_end, done);   // d2's QP completes first
    const bool past_live = done > 0 && done == B_end;          // A's bytes [0, A_end) are below done but not forwarded
    relay_forward_completed(inflight, batches, A_end, done);
    const bool backwards = done == A_end && A_end < B_end;
    CHECK(past_live && backwards);                             // the hazard is real for an aggregate ring
    // per route: each ring sees its own completions in its one QP's order, so done only grows
    int ia = 1, ib = 1; std::deque<int> ba{1}, bb{1}; uint64_t da = 0, db = 0;
    relay_forward_completed(ib, bb, relay_pad(1u << 20), db);
    relay_forward_completed(ia, ba, relay_pad(1u << 20), da);
    CHECK(da == relay_pad(1u << 20) && db == relay_pad(1u << 20));
    printf("  aggregate-ring hazard: out-of-order completions move done past live bytes, then backwards (per-route: no)\n");
}

int main() {
    sixty();
    mismatch();
    zero_progress();
    aggregate_hazard();
    if (fails) { fprintf(stderr, "relay_loop_test (hybrid): %d failures\n", fails); return 1; }
    printf("relay_loop_test (hybrid): OK\n");
    return 0;
}
"""

with tempfile.TemporaryDirectory(prefix='gr8-relay-loop-') as d:
    for name, pre, lp, body, tmo in (('relay_loop_test', preamble, loop, main, 5),
                                     ('relay_loop_hybrid_test', multi_preamble, loop_production, multi_main, 60)):
        cpp = Path(d) / (name + '.cpp')
        exe = Path(d) / name
        cpp.write_text(pre + ctx + lp + body)
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-pthread',
                        '-I', str(repo / 'engine/comms'), str(cpp), '-o', str(exe)], check=True)
        try:
            subprocess.run([str(exe)], check=True, timeout=tmo)
        except subprocess.TimeoutExpired:
            raise SystemExit(f'{name}: no progress to completion within {tmo} seconds') from None
