// Four-rank REAL-fabric byte gate of the Flash SP prefill (G6_PF_RS=2), TP4-SP-MHC-RS-AG-BUILD.md "CPU four-rank gate"
// (its last item, run in the lead's slot). Host-driven like stripe_test.cpp: no kernels, no GPU memory; the real
// g6::Fabric proxy and NICs move the bytes. Every lane runs the SP layer-step plan of pf_layout.h exactly as
// pf_enqueue_flash_sp / pf_consume_flash_sp do it: one enqueue_step per layer-step (Rs mask 0x5, ss = pf_ss, bytes =
// sp_rs_bytes, sizes = sp_sizes, c4 on the last layer-step only), then per boundary
//   RS  (c0 / c2): pack my BF16-sized partial of owner p's rows into slice p (send + p * ss), ring, wait the three peer
//                  slices of MY rows (recv[c][p] offset 0, tail at sp_rs_bytes(M, me)), check every byte;
//   AG  (c1 / c3 / last c4): pack my FP32-sized owner rows at send offset 0, ring, wait every peer's rows (tail at
//                  sp_ag_bytes(M, p)), check every byte, and check the gathered view: global row r comes from
//                  pf_owner_of(M, r) at row r - pf_r0(M, owner).
// Checks: payload words = the sender's source words (a function of lane, sender, step, c, global row, word), each tail =
// coll_seq(step, c, 5), guard bytes after each tail untouched (receive slots are re-poisoned one step ahead, at the
// point the protocol makes it safe), my own RS slice and its tail spot never written by the proxy, c4 slots untouched
// on non-last layer-steps, owner intervals disjoint and covering [0, M). M = {1, 2, 3, 4, 5, 63, 64, 8191, 8192} (<= MP)
// with zero-row owners, both lane fabrics at once (ports PORT and PORT + 16, as g6m4's +8 / +24), optional random
// doorbell delay on one rank (two-parity reuse with a late rank), optional dual-rail striping. Per-collective doorbell ->
// last tail times are printed; they are host-driven transport times, not a prefill result.
// What it does not prove: CUDA visibility of the pinned buffers, kernel arithmetic, or speed of the GPU path.
//   sp4_test RANK IP0,IP1,IP2,IP3 IPS2|- [MP 8192] [LANES 2] [NL 3] [REPS 2] [DELAY_RANK -1] [STRIPE_MIN 0] [PORT 18760]
// Build (on a Spark, in engine/comms/tests):
//   nvcc -O2 -std=c++17 -I.. -I../../runtime -o sp4_test sp4_test.cpp ../fabric.cpp -libverbs -lpthread
// Loopback self-check of THIS program (DEV box, no NIC). LoopFab is a model of the proxy's uniform/Rs path, not the
// fabric; it proves the test logic and that its checks fire (SP4_LOOP_FAULT=corrupt|overrun|selftail must FAIL; set
// SP4_TIMEOUT_S=3 so the ranks a fault starves give up fast):
//   g++ -std=c++17 -O2 -Wall -DSP4_LOOP -I engine/comms -I engine/runtime engine/comms/tests/sp4_test.cpp -o sp4_loop -lpthread
//   sp4_loop [MP 8192] [HID 64] [LANES 2] [NL 3] [REPS 2] [DELAY_RANK 3]
#include "pf_layout.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#ifdef SP4_LOOP
#include <condition_variable>
#include <deque>
#include <mutex>
#endif

using namespace g6rt;
using u64 = unsigned long long;

namespace
{
constexpr int NR = 4, NC = sp_ncoll;
constexpr unsigned char POISON = 0xA5;
constexpr u64 POISON64 = 0xA5A5A5A5A5A5A5A5ull;
constexpr size_t GUARD = 4096;   // guard bytes checked after each tail (plus the last 256 B of the slot)

struct Opt
{
    int rank = 0, mp = 8192, hid = 4096, lanes = 2, nl = 3, reps = 2, delay_rank = -1, port = 18760;
    size_t stripe_min = 0;
    std::vector<std::string> ips, ips2;
};

// source word of (lane, sender, step, c, global row, word in row); a bijective mix of disjoint bit fields
inline u64 word(int lane, int sender, u64 step, int c, int row, int w)
{
    u64 x = ((u64)lane << 62) ^ ((u64)(sender + 1) << 58) ^ ((u64)c << 55) ^ (step << 32) ^ ((u64)row << 12) ^ (u64)w;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}

// a host wait gives up after SP4_TIMEOUT_S seconds (default 30, the proxy's doorbell timeout)
double timeout_ms()
{
    static const double t = getenv("SP4_TIMEOUT_S") ? atof(getenv("SP4_TIMEOUT_S")) * 1000 : 30000;
    return t;
}
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int> m_list(int mp)
{
    std::vector<int> v;
    for (int m : {1, 2, 3, 4, 5, 63, 64, 8191, 8192}) if (m <= mp) v.push_back(m);
    if (v.back() != mp) v.push_back(mp);
    return v;
}

template <class Fab>
struct Lane
{
    Fab& f;
    g6::FabricDev d;
    Opt o;
    int lane, me;
    size_t ss;
    u64 step = 0;
    std::string err;
    bool stuck = false;   // a wait timed out or the fabric set its error word: no end barrier
    size_t words = 0, bad = 0, guard_bad = 0, tail_bad = 0, struct_bad = 0;
    std::vector<double> ms[NC];
    size_t ext[2][NC][NR] = {};   // per (parity, c, sender): extent written by the proxy into my receive slot since poison
    std::mt19937 rng;

    Lane(Fab& f_, const Opt& o_, int lane_) : f(f_), d(f_.dev()), o(o_), lane(lane_), me(o_.rank), ss(pf_ss(d)),
                                              rng(1234u + 17u * o_.rank + lane_) {}
    char* send_slot(u64 s, int c) { return d.send + (((s & 1) * NC + c) * d.stride); }
    char* recv_slot(u64 s, int c, int p) { return d.recv + ((((s & 1) * NC + c) * NR + p) * d.stride); }
    u64* w64(char* p) { return reinterpret_cast<u64*>(p); }

    void fail(const char* fmt, ...) __attribute__((format(printf, 2, 3)))
    {
        if (!err.empty()) return;
        char b[512];
        va_list a; va_start(a, fmt); vsnprintf(b, sizeof b, fmt, a); va_end(a);
        err = b;
    }
    void ring(int c, u64 seq)
    {
        if (me == o.delay_rank) usleep(rng() % 1500);
        __atomic_store_n(d.doorbell + c, seq, __ATOMIC_RELEASE);
    }
    bool wait_tail(const char* slot, size_t off, u64 seq, int c, int p)
    {
        volatile const u64* t = reinterpret_cast<volatile const u64*>(slot + off);
        const double t0 = now_ms();
        while (*t != seq)
        {
            if (*d.error) { stuck = true; fail("fabric error 0x%08x waiting step %llu c%d from rank %d", *d.error, step, c, p); return false; }
            if (now_ms() - t0 > timeout_ms())
            {
                stuck = true;
                fail("timeout: step %llu c%d from rank %d, tail at +%zu is %016llx, want %016llx", step, c, p, off, (u64)*t, seq);
                return false;
            }
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        return true;
    }
    // bytes [nb + 8, nb + 8 + GUARD) and the slot's last 256 B still poison
    void check_guard(const char* slot, size_t nb, int c, int p)
    {
        auto scan = [&](size_t a, size_t b) {
            for (size_t i = a; i < b && i < d.stride; ++i)
                if ((unsigned char)slot[i] != POISON) { ++guard_bad; fail("guard: step %llu c%d from rank %d byte +%zu = %02x", step, c, p, i, (unsigned char)slot[i]); return; }
        };
        scan(nb + 8, nb + 8 + GUARD);
        scan(d.stride - 256, d.stride);
    }

    bool rs(int M, int c)
    {
        const u64 seq = g6::coll_seq(step, c, NC);
        char* s = send_slot(step, c);
        const int wr = o.hid / 4;   // u64 words per BF16 row
        for (int p = 0; p < NR; ++p)
        {
            u64* sl = w64(s + p * ss);
            const int r0 = pf_r0(M, p, NR), n = pf_rows(M, p, NR);
            for (int r = 0; r < n; ++r)
                for (int w = 0; w < wr; ++w) sl[(size_t)r * wr + w] = word(lane, me, step, c, r0 + r, w);
            *w64(s + p * ss + sp_rs_bytes(M, p, NR, o.hid)) = POISON64;   // the proxy writes peer tails here, never mine
        }
        const double t0 = now_ms();
        ring(c, seq);
        const size_t nb = sp_rs_bytes(M, me, NR, o.hid);
        const int r0 = pf_r0(M, me, NR), n = pf_rows(M, me, NR);
        for (int p = 0; p < NR; ++p)
            if (p != me && !wait_tail(recv_slot(step, c, p), nb, seq, c, p)) return false;
        ms[c].push_back(now_ms() - t0);
        for (int p = 0; p < NR; ++p)
        {
            const char* slot = p == me ? s + me * ss : recv_slot(step, c, p);   // rs_slice(loop = false)
            const u64* g = reinterpret_cast<const u64*>(slot);
            for (int r = 0; r < n; ++r)
                for (int w = 0; w < wr; ++w, ++words)
                    if (g[(size_t)r * wr + w] != word(lane, p, step, c, r0 + r, w))
                    {
                        if (!bad++) fail("RS bytes: step %llu c%d M %d from rank %d row %d word %d", step, c, M, p, r0 + r, w);
                    }
            if (p == me)
            {
                if (*reinterpret_cast<const u64*>(slot + nb) != POISON64) { ++tail_bad; fail("RS: proxy wrote my own slice tail, step %llu c%d", step, c); }
                continue;
            }
            check_guard(slot, nb, c, p);
            ext[step & 1][c][p] = std::max(ext[step & 1][c][p], nb + 8);
        }
        return true;   // byte failures are counted; only a stuck wait stops the lane
    }

    bool ag(int M, int c)
    {
        const u64 seq = g6::coll_seq(step, c, NC);
        char* s = send_slot(step, c);
        const int wr = o.hid / 2;   // u64 words per FP32 row
        const int r0 = pf_r0(M, me, NR), n = pf_rows(M, me, NR);
        for (int r = 0; r < n; ++r)
            for (int w = 0; w < wr; ++w) w64(s)[(size_t)r * wr + w] = word(lane, me, step, c, r0 + r, w);
        const double t0 = now_ms();
        ring(c, seq);   // a zero-row owner still rings: the tail alone travels
        for (int p = 0; p < NR; ++p)
            if (p != me && !wait_tail(recv_slot(step, c, p), sp_ag_bytes(M, p, NR, o.hid), seq, c, p)) return false;
        ms[c].push_back(now_ms() - t0);
        // gathered view: every global row from its owner, at row - pf_r0(owner)
        for (int r = 0; r < M; ++r)
        {
            const int q = pf_owner_of(M, r, NR);
            const int q0 = pf_r0(M, q, NR);
            if (q < 0 || q >= NR || r < q0 || r >= q0 + pf_rows(M, q, NR)) { ++struct_bad; fail("owner of row %d of M %d = %d outside its interval", r, M, q); return false; }
            const u64* g = reinterpret_cast<const u64*>(q == me ? s : recv_slot(step, c, q)) + (size_t)(r - q0) * wr;
            for (int w = 0; w < wr; ++w, ++words)
                if (g[w] != word(lane, q, step, c, r, w))
                {
                    if (!bad++) fail("AG bytes: step %llu c%d M %d row %d (owner %d) word %d", step, c, M, r, q, w);
                }
        }
        for (int p = 0; p < NR; ++p)
            if (p != me)
            {
                const size_t nb = sp_ag_bytes(M, p, NR, o.hid);
                check_guard(recv_slot(step, c, p), nb, c, p);
                ext[step & 1][c][p] = std::max(ext[step & 1][c][p], nb + 8);
            }
        return true;
    }

    // re-poison the receive slots of the NEXT step's parity. Safe after my c2 wait of step s and before my c3 ring:
    // a peer writes step s + 1 only after it has my step s c3, and step s - 1 (same parity) is fully consumed.
    void poison_next()
    {
        const int par = (int)((step + 1) & 1);
        for (int c = 0; c < NC; ++c)
            for (int p = 0; p < NR; ++p)
                if (ext[par][c][p]) { memset(recv_slot(step + 1, c, p), POISON, ext[par][c][p]); ext[par][c][p] = 0; }
    }

    bool sub_batch(int M)
    {
        int cover = 0;   // owner intervals disjoint and covering [0, M), zero-row owners allowed
        for (int p = 0; p < NR; ++p)
        {
            if (pf_r0(M, p, NR) != cover || pf_rows(M, p, NR) < 0) { ++struct_bad; fail("owner intervals of M %d broken at rank %d", M, p); return false; }
            cover += pf_rows(M, p, NR);
        }
        if (cover != M) { ++struct_bad; fail("owner intervals of M %d cover %d rows", M, cover); return false; }
        // = pf_enqueue_flash_sp (flash_prefill.cuh)
        typename Fab::Rs rsj;
        rsj.mask = sp_rs_mask;
        rsj.ss = pf_ss(d);
        for (int p = 0; p < NR; ++p) rsj.bytes.push_back(sp_rs_bytes(M, p, NR, o.hid));
        for (int i = 0; i < o.nl; ++i) f.enqueue_step(step + i, sp_sizes(M, me, NR, o.hid, i == o.nl - 1), rsj);
        for (int i = 0; i < o.nl; ++i, ++step)
        {
            const bool last = i == o.nl - 1;
            if (!rs(M, sp_c_attn_rs) || !ag(M, sp_c_attn_ag) || !rs(M, sp_c_mlp_rs)) return false;
            poison_next();
            if (!ag(M, sp_c_mlp_ag)) return false;
            if (last && !ag(M, sp_c_resid_ag)) return false;
            if (!last)   // c4 only on the last layer-step: this parity's c4 receive slots stay poison
                for (int p = 0; p < NR; ++p)
                    if (p != me && (ext[step & 1][sp_c_resid_ag][p] ||
                                    *reinterpret_cast<const u64*>(recv_slot(step, sp_c_resid_ag, p) + sp_ag_bytes(M, p, NR, o.hid)) != POISON64))
                    {
                        ++tail_bad; fail("c4 written on non-last layer-step %llu by rank %d", step, p);
                    }
        }
        return true;
    }

    // config checks + poison (before the start barrier)
    int prepare()
    {
        if (d.ncoll != NC || d.nr != NR) { fprintf(stderr, "lane %d: fabric has ncoll %d nr %d, want 5 / 4\n", lane, d.ncoll, d.nr); return 1; }
        if (ss < pf_ss_min_rb(o.mp, NR, (size_t)o.hid * 2) || (size_t)(o.mp + NR - 1) / NR * o.hid * 4 + 8 > d.stride)
        { fprintf(stderr, "lane %d: slot too small (stride %zu ss %zu)\n", lane, d.stride, ss); return 1; }
        memset(d.send, POISON, 2 * NC * d.stride);
        memset(d.recv, POISON, 2 * NC * NR * d.stride);
        return 0;
    }
    std::vector<std::pair<int, std::vector<double>>> per_m;
    void body()
    {
        const std::vector<int> ms_list = m_list(o.mp);
        for (int rep = 0; rep < o.reps && !stuck; ++rep)
            for (size_t k = 0; k < ms_list.size() && !stuck; ++k)
            {
                const int M = ms_list[lane & 1 ? ms_list.size() - 1 - k : k];   // lane 1 runs the list backwards
                for (auto& v : ms) v.clear();
                if (!sub_batch(M)) return;
                if (rep == o.reps - 1)
                {
                    std::vector<double> med(NC, 0.0);
                    for (int c = 0; c < NC; ++c)
                        if (!ms[c].empty()) { std::sort(ms[c].begin(), ms[c].end()); med[c] = ms[c][ms[c].size() / 2]; }
                    per_m.push_back({M, med});
                }
            }
    }
    int report()
    {
        for (auto& [M, med] : per_m)
            printf("rank %d lane %d M %5d: doorbell->last tail ms p50 c0 %.3f c1 %.3f c2 %.3f c3 %.3f c4 %.3f\n",
                   me, lane, M, med[0], med[1], med[2], med[3], med[4]);
        const bool ok = err.empty() && !bad && !guard_bad && !tail_bad && !struct_bad;
        printf("rank %d lane %d: %s steps %llu words %zu bad %zu guard %zu tail %zu struct %zu%s%s\n", me, lane,
               ok ? "BYTES OK" : "FAIL", step, words, bad, guard_bad, tail_bad, struct_bad, err.empty() ? "" : " | ", err.c_str());
        return ok ? 0 : 1;
    }
};

// one rank: its lanes run on their own threads. Fabric::barrier takes its TCP port from a process-wide counter
// (fabric.cpp), so the barriers are called from this one thread in lane order, never from two lane threads at once.
template <class Fab>
int rank_main(const std::vector<Fab*>& fabs, const Opt& o)
{
    std::vector<std::unique_ptr<Lane<Fab>>> lanes;
    for (size_t l = 0; l < fabs.size(); ++l) lanes.emplace_back(new Lane<Fab>(*fabs[l], o, (int)l));
    for (auto& L : lanes) if (L->prepare()) return 1;
    for (auto& L : lanes) L->f.barrier();   // everyone poisoned before anyone writes
    std::vector<std::thread> th;
    for (auto& L : lanes) th.emplace_back([&L] { L->body(); });
    for (auto& t : th) t.join();
    for (auto& L : lanes) if (!L->stuck) L->f.barrier();   // peers' last writes into my slots are done
    int rc = 0;
    for (auto& L : lanes) rc |= L->report();
    return rc;
}

[[maybe_unused]] std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(t);
    return v;
}
}   // namespace

#ifndef SP4_LOOP
int main(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: sp4_test RANK IP0,IP1,IP2,IP3 IPS2|- [MP] [LANES] [NL] [REPS] [DELAY_RANK] [STRIPE_MIN] [PORT]\n"); return 2; }
    Opt o;
    o.rank = atoi(argv[1]);
    o.ips = split(argv[2]);
    if (strcmp(argv[3], "-") != 0) o.ips2 = split(argv[3]);
    if (argc > 4) o.mp = atoi(argv[4]);
    if (argc > 5) o.lanes = atoi(argv[5]);
    if (argc > 6) o.nl = atoi(argv[6]);
    if (argc > 7) o.reps = atoi(argv[7]);
    if (argc > 8) o.delay_rank = atoi(argv[8]);
    if (argc > 9) o.stripe_min = (size_t)atoll(argv[9]);
    if (argc > 10) o.port = atoi(argv[10]);
    if (o.ips.size() != NR || o.rank < 0 || o.rank >= NR || o.mp < 1 || o.mp > 8192 || o.lanes < 1 || o.lanes > 2 || o.nl < 1 || o.reps < 1)
    { fprintf(stderr, "sp4_test: four ranks, 1 <= MP <= 8192, LANES 1 or 2, NL >= 1, REPS >= 1\n"); return 2; }
    std::vector<std::unique_ptr<g6::Fabric>> fabs;
    for (int l = 0; l < o.lanes; ++l)
    {
        g6::FabricConfig cfg;
        cfg.rank = o.rank; cfg.ips = o.ips; cfg.ips2 = o.ips2; cfg.stripe_min = o.stripe_min;
        cfg.slot_bytes = pf_slot_bytes_rb(o.mp, true, NR, (size_t)o.hid * 2);   // = g6m4 pf_slot_bytes(MP, true)
        cfg.ncoll = NC;
        cfg.port = o.port + 16 * l;
        fabs.emplace_back(new g6::Fabric);
        fabs.back()->init(cfg);
        fabs.back()->start_proxy();
    }
    printf("rank %d: MP %d HID %d lanes %d NL %d reps %d delay_rank %d stripe_min %zu stride %zu ss %zu\n", o.rank, o.mp, o.hid,
           o.lanes, o.nl, o.reps, o.delay_rank, o.stripe_min, fabs[0]->dev().stride, pf_ss(fabs[0]->dev()));
    std::vector<g6::Fabric*> fp;
    for (auto& f : fabs) fp.push_back(f.get());
    const bool ok = rank_main(fp, o) == 0;
    for (auto& f : fabs) f->stop_proxy();
    printf("sp4_test rank %d: %s\n", o.rank, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
#else
// ------------------------------------------------------------------ loopback model (self-check of this program only)
namespace
{
struct LoopBarrier
{
    std::mutex mu; std::condition_variable cv; int n = 0, gen = 0;
    void wait()
    {
        std::unique_lock<std::mutex> lk(mu);
        const int g = gen;
        if (++n == NR) { n = 0; ++gen; cv.notify_all(); }
        else cv.wait(lk, [&] { return gen != g; });
    }
};
struct LoopFab;
struct LoopWorld { LoopFab* fab[NR] = {}; LoopBarrier bar; };
struct LoopFab
{
    using Rs = g6::Fabric::Rs;
    struct Job { u64 step; std::vector<uint32_t> sizes; Rs rs; };
    LoopWorld* w = nullptr;
    g6::FabricDev d{};
    std::vector<u64> mem;
    std::deque<Job> q;
    std::mutex mu;
    std::thread th;
    std::atomic<bool> stop{false};
    std::string fault;   // SP4_LOOP_FAULT: corrupt | overrun | selftail (one fault, rank 1 -> rank 2, step 4)
    void init(int rank, size_t slot_bytes, LoopWorld* world)
    {
        w = world;
        w->fab[rank] = this;
        const size_t stride = (slot_bytes + 8 + 255) & ~size_t(255);
        const size_t send = 2 * NC * stride, recv = 2 * NC * NR * stride;
        mem.assign((send + recv) / 8 + NC + 1, 0);
        d.send = reinterpret_cast<char*>(mem.data());
        d.recv = d.send + send;
        d.doorbell = reinterpret_cast<unsigned long long*>(d.recv + recv);
        d.error = reinterpret_cast<unsigned int*>(d.doorbell + NC);
        d.rank = rank; d.nr = NR; d.ncoll = NC; d.stride = stride;
        if (const char* f = getenv("SP4_LOOP_FAULT")) fault = f;
    }
    g6::FabricDev dev() const { return d; }
    void enqueue_step(u64 step, const std::vector<uint32_t>& sizes, const Rs& rs)
    {
        std::lock_guard<std::mutex> lk(mu);
        q.push_back({step, sizes, rs});
    }
    void start_proxy() { th = std::thread([this] { proxy(); }); }
    void stop_proxy() { stop = true; th.join(); }
    void barrier() { w->bar.wait(); }
    // fabric.cpp uniform path: per job, collectives in order; wait doorbell == seq; Rs: slice p -> peer p, tail after it
    // (never my own slice); else the payload + tail from offset 0 to every peer; into peer recv[par][c][me] offset 0
    void proxy()
    {
        const int me = d.rank;
        while (!stop)
        {
            Job job;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!q.empty()) { job = std::move(q.front()); q.pop_front(); }
            }
            if (job.sizes.empty()) { usleep(20); continue; }
            const int par = (int)(job.step & 1);
            for (int c = 0; c < (int)job.sizes.size(); ++c)
            {
                const u64 seq = g6::coll_seq(job.step, c, NC);
                const double t0 = now_ms();
                while (__atomic_load_n(d.doorbell + c, __ATOMIC_ACQUIRE) != seq)
                {
                    if (stop) return;
                    if (now_ms() - t0 > timeout_ms()) { *d.error = 0x70000000u | c; return; }
                }
                const bool rs = (job.rs.mask >> c) & 1;
                char* src = d.send + ((size_t)par * NC + c) * d.stride;
                if (!rs) *reinterpret_cast<u64*>(src + job.sizes[c]) = seq;
                for (int p = 0; p < NR; ++p)
                {
                    if (p == me) continue;
                    char* sp = rs ? src + p * job.rs.ss : src;
                    const uint32_t nb = rs ? job.rs.bytes[p] : job.sizes[c];
                    if (rs) *reinterpret_cast<u64*>(sp + nb) = seq;
                    char* dst = w->fab[p]->d.recv + (((size_t)par * NC + c) * NR + me) * d.stride;
                    memcpy(dst, sp, nb);
                    const bool hit = me == 1 && p == 2 && job.step == 4;
                    if (hit && fault == "corrupt" && nb) dst[nb / 2] ^= 0x10;
                    if (hit && fault == "overrun") memset(dst + nb + 8, 0x3C, 8);
                    if (hit && fault == "selftail" && rs) *reinterpret_cast<u64*>(src + me * job.rs.ss + job.rs.bytes[me]) = seq;
                    __atomic_store_n(reinterpret_cast<u64*>(dst + nb), *reinterpret_cast<u64*>(sp + nb), __ATOMIC_RELEASE);
                }
            }
        }
    }
};
}   // namespace

int main(int argc, char** argv)
{
    Opt o;
    o.hid = 64;
    o.delay_rank = 3;
    if (argc > 1) o.mp = atoi(argv[1]);
    if (argc > 2) o.hid = atoi(argv[2]);
    if (argc > 3) o.lanes = atoi(argv[3]);
    if (argc > 4) o.nl = atoi(argv[4]);
    if (argc > 5) o.reps = atoi(argv[5]);
    if (argc > 6) o.delay_rank = atoi(argv[6]);
    if (o.hid % 4 || o.lanes < 1 || o.lanes > 2) { fprintf(stderr, "HID multiple of 4, LANES 1 or 2\n"); return 2; }
    std::vector<std::unique_ptr<LoopWorld>> worlds;
    std::vector<std::unique_ptr<LoopFab>> fabs;
    for (int l = 0; l < o.lanes; ++l)
    {
        worlds.emplace_back(new LoopWorld);
        for (int r = 0; r < NR; ++r)
        {
            fabs.emplace_back(new LoopFab);
            fabs.back()->init(r, pf_slot_bytes_rb(o.mp, true, NR, (size_t)o.hid * 2), worlds.back().get());
        }
    }
    for (auto& f : fabs) f->start_proxy();
    std::vector<int> rc(NR, 1);
    std::vector<std::thread> th;
    for (int r = 0; r < NR; ++r)   // one thread per rank = one sp4_test process
        th.emplace_back([&, r] {
            Opt ro = o;
            ro.rank = r;
            std::vector<LoopFab*> fp;
            for (int l = 0; l < o.lanes; ++l) fp.push_back(fabs[l * NR + r].get());
            rc[r] = rank_main(fp, ro);
        });
    for (auto& t : th) t.join();
    for (auto& f : fabs) f->stop_proxy();
    const bool ok = std::all_of(rc.begin(), rc.end(), [](int r) { return r == 0; });
    printf("sp4_loop (loopback model, not the fabric): %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
#endif
