#!/usr/bin/env python3
"""Compile the production Fabric::Impl relayed()/wrs()/post() against mocked verbs (review M5).

Hybrid origin: 5 relayed peers, ZERO relay contexts, the error word still set; each relayed destination has its own
ring, credit and packed header (relay_kind(k, p, me)). Ring mode: the diagonal's headers are exactly 0/1."""
from pathlib import Path
import subprocess
import tempfile


repo = Path(__file__).resolve().parents[3]
source = (repo / 'engine/comms/fabric.cpp').read_text()
body = source[source.index('struct HybDst'):source.index('// G6_XTRACE (xtrace.h). xenv')]

program = r'''
#include "relay.h"
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace g6;
#define FCHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); abort(); } } while (0)
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "CHECK FAILED %s:%d %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
enum { IBV_WR_RDMA_WRITE = 1, IBV_WR_SEND = 2, IBV_SEND_INLINE = 4, IBV_SEND_SIGNALED = 8 };
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
  uint64_t wr_id = 0; ibv_send_wr* next = nullptr; ibv_sge* sg_list = nullptr; int num_sge = 0, opcode = 0, send_flags = 0;
  struct { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr{};
};
struct ibv_qp { int peer; };
struct Posted { int peer, opcode, flags; uint64_t wr_id, raddr; uint32_t rkey, len; bool has_hdr; RelayHdr h; };
static std::vector<Posted> posted;
int ibv_post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) {
  *bad = nullptr;
  for (; wr; wr = wr->next) {
    Posted p{qp->peer, wr->opcode, wr->send_flags, wr->wr_id, wr->wr.rdma.remote_addr, wr->wr.rdma.rkey,
             wr->sg_list[0].length, false, {}};
    if (wr->opcode == IBV_WR_SEND) { p.has_hdr = true; memcpy(&p.h, (void*)wr->sg_list[0].addr, sizeof(RelayHdr)); }
    posted.push_back(p);
  }
  return 0;
}
constexpr int kRelayRecv = 4096, kRelayBuf = 64, kRelayInline = kRelayBuf - (int)sizeof(RelayHdr);
struct Peer { uint64_t addr; };
struct Impl {
  ibv_qp* qp[8] = {};
  Peer peer[8] = {};
  bool ring = false; int nxt = -1, prv = -1, dia = -1;
  RelayOut rout;
  uint64_t* credit = nullptr;
  uint64_t ring_addr = 0; uint32_t ring_rkey = 0;
  volatile unsigned* err = nullptr;
''' + body + r'''
};
static std::vector<char> src(4u << 20);
static ibv_send_wr write_to(Impl& I, int p, uint64_t off, uint32_t len, uint64_t id, ibv_sge& sge) {
  sge = ibv_sge{(uint64_t)src.data(), len, 7};
  ibv_send_wr w{}; w.opcode = IBV_WR_RDMA_WRITE; w.sg_list = &sge; w.num_sge = 1; w.wr_id = id;
  w.send_flags = IBV_SEND_SIGNALED; w.wr.rdma.remote_addr = I.peer[p].addr + off; w.wr.rdma.rkey = 99;
  return w;
}
int main() {
  ibv_qp qps[8]; for (int i = 0; i < 8; ++i) qps[i].peer = i;
  // ---- hybrid origin: rank 0 of 8, peers 1 and 2 direct, 3..7 relayed (via 1 or 2); this process has NO RelayCtx
  {
    Impl I; I.hybrid = true; I.me = 0; I.hd.resize(8);
    unsigned errword = 0; I.err = &errword;
    uint64_t credits[8] = {};
    for (int p = 0; p < 8; ++p) { I.qp[p] = &qps[p]; I.peer[p].addr = 0x100000000ull * (p + 1); }
    for (int p = 3; p < 8; ++p) {
      I.hd[p].relayed = true; I.hd[p].via = 1 + p % 2; I.hd[p].credit = &credits[p];
      I.hd[p].ring_addr = 0x9000000000ull + 0x10000000ull * p; I.hd[p].ring_rkey = 500 + p;
    }
    CHECK(I.err != nullptr);
    int nrel = 0; for (int p = 0; p < 8; ++p) nrel += I.relayed(p);
    CHECK(nrel == 5 && !I.relayed(0) && !I.relayed(1) && !I.relayed(2));
    ibv_sge sge;
    // a direct peer: one plain write
    { auto w = write_to(I, 1, 64, 3u << 20, 11, sge); CHECK(I.wrs(1, w) == 1); posted.clear(); I.post(1, w);
      CHECK(posted.size() == 1 && posted[0].opcode == IBV_WR_RDMA_WRITE && posted[0].raddr == I.peer[1].addr + 64); }
    // an inline write to relayed peer 5: one SEND, packed header kind 1
    { auto w = write_to(I, 5, 128, 16, 12, sge); CHECK(I.wrs(5, w) == 1); posted.clear(); I.post(5, w);
      CHECK(posted.size() == 1 && posted[0].opcode == IBV_WR_SEND && posted[0].has_hdr);
      CHECK(posted[0].h.kind == relay_kind(1, 5, 0) && posted[0].h.off == 128 && posted[0].h.len == 16);
      CHECK(posted[0].wr_id == 12 && (posted[0].flags & IBV_SEND_SIGNALED)); }
    // a 2.5 MiB write to each relayed peer: 3 chunks into its own ring, kind 0 packed with its own route
    for (int p = 3; p < 8; ++p) {
      const uint32_t len = p == 3 ? 3u << 20 : (5u << 20) / 2;  // peer 3: whole chunks, so it can fill exactly
      auto w = write_to(I, p, 4096, len, 100 + p, sge);
      CHECK(I.wrs(p, w) == 6); posted.clear(); I.post(p, w);
      CHECK(posted.size() == 6);
      for (int c = 0; c < 3; ++c) {
        const Posted& d = posted[2 * c]; const Posted& s = posted[2 * c + 1];
        CHECK(d.opcode == IBV_WR_RDMA_WRITE && d.rkey == I.hd[p].ring_rkey);
        CHECK(d.raddr == I.hd[p].ring_addr + (uint64_t)c * kRelayChunk);
        CHECK(s.opcode == IBV_WR_SEND && s.h.kind == relay_kind(0, p, 0) && s.h.pos == (uint64_t)c * kRelayChunk);
        CHECK(s.h.off == 4096 + (uint64_t)c * kRelayChunk);
        CHECK(s.wr_id == (c == 2 ? 100u + p : 0u));
      }
      CHECK(I.hd[p].rout.pos == 2ull * kRelayChunk + relay_pad(len - 2ull * kRelayChunk));
    }
    // fill peer 3's ring to exactly full (credit 0): peer 4 is not affected
    { auto w = write_to(I, 3, 0, kRelayChunk, 0, sge);
      for (uint64_t n = 3; n < kRelayRing / kRelayChunk; ++n) I.post(3, w);
      CHECK(I.hd[3].rout.pos == kRelayRing);
      posted.clear(); auto w4 = write_to(I, 4, 0, kRelayChunk, 44, sge); I.post(4, w4);
      CHECK(posted.size() == 2 && posted[1].h.kind == relay_kind(0, 4, 0) && I.hd[4].rout.pos == 3ull * kRelayChunk + relay_pad(kRelayChunk / 2));
      CHECK(errword == 0);
      // peer 3 has no room: with the error word set, post returns at once (no 30 s credit wait) and posts nothing
      errword = 0x75000002u; posted.clear();
      const auto t0 = std::chrono::steady_clock::now();
      I.post(3, w);
      CHECK(posted.empty() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
      // the relay forwards (credit advances) and the error clears: peer 3 continues where its ring left off
      errword = 0; credits[3] = 4ull * kRelayChunk; posted.clear(); I.post(3, w);
      CHECK(posted.size() == 2 && posted[1].h.pos == kRelayRing && posted[1].h.kind == relay_kind(0, 3, 0)); }
    // the packed kinds of distinct routes differ and keep bits 24..31 zero
    for (int p = 3; p < 8; ++p) { CHECK(relay_kind(0, p, 0) >> 24 == 0); CHECK(relay_kind(0, p, 0) != relay_kind(0, 3 + (p - 2) % 5, 0)); }
  }
  // ---- ring mode (hybrid = false): only the diagonal is relayed, headers are exactly 0 / 1
  {
    Impl I; I.ring = true; I.dia = 2; I.me = 0;
    unsigned errword = 0; I.err = &errword; uint64_t credit = 0; I.credit = &credit; I.ring_addr = 0x7000; I.ring_rkey = 3;
    for (int p = 0; p < 4; ++p) { I.qp[p] = &qps[p]; I.peer[p].addr = 0x100000000ull * (p + 1); }
    CHECK(I.relayed(2) && !I.relayed(1) && !I.relayed(3));
    ibv_sge sge;
    auto wi = write_to(I, 2, 8, 8, 1, sge); posted.clear(); I.post(2, wi);
    CHECK(posted.size() == 1 && posted[0].h.kind == 1u);
    auto wb = write_to(I, 2, 0, 2u << 20, 2, sge); posted.clear(); I.post(2, wb);
    CHECK(posted.size() == 4 && posted[1].h.kind == 0u && posted[3].h.kind == 0u && posted[3].wr_id == 2);
    CHECK(I.rout.pos == 2ull * kRelayChunk);
  }
  printf("relay_post_test OK: hybrid origin 5 relayed peers, 0 relay contexts, per-dest rings; ring headers 0/1\n");
  return 0;
}
'''

with tempfile.TemporaryDirectory() as d:
    cpp = Path(d) / 'relay_post_test.cpp'
    cpp.write_text(program)
    exe = Path(d) / 'relay_post_test'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-I', str(repo / 'engine/comms'), str(cpp), '-o', str(exe)],
                   check=True)
    subprocess.run([str(exe)], check=True)
