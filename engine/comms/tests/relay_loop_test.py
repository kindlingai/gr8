#!/usr/bin/env python3
"""Compile the production relay loop against mocked verbs and force its two credit-pressure edge cases."""
from pathlib import Path
import subprocess
import tempfile


repo = Path(__file__).resolve().parents[3]
source = (repo / 'engine/comms/fabric.cpp').read_text()
ctx = source[source.index('struct RelayHdr'):source.index('int ring_gid_index')]
loop = source[source.index('void relay_loop(int cpu)'):source.index('struct Job')]
# The real process leaves a failed context parked forever; make an integration-test failure observable.
loop = loop.replace('if (*c.err) continue;', 'if (*c.err) throw std::runtime_error("relay failed");')

preamble = r'''
#include "relay.h"
#include <atomic>
#include <cerrno>
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
struct ibv_wc { uint64_t wr_id; ibv_wc_status status; int opcode; };
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

with tempfile.TemporaryDirectory(prefix='gr8-relay-loop-') as d:
    cpp = Path(d) / 'relay_loop_test.cpp'
    exe = Path(d) / 'relay_loop_test'
    cpp.write_text(preamble + ctx + loop + main)
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-pthread',
                    '-I', str(repo / 'engine/comms'), str(cpp), '-o', str(exe)], check=True)
    try:
        subprocess.run([str(exe)], check=True, timeout=5)
    except subprocess.TimeoutExpired:
        raise SystemExit('relay_loop_test: no progress to completion within 5 seconds') from None
