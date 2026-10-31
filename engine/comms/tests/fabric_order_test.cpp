// Host test (no verbs, no CUDA) of the order in which the fabric proxy walks one step's collectives (fabric_order.h
// walk_step, the loop fabric.cpp runs). The GPU is modelled by a thread that publishes collectives in PROGRAM order: it
// rings doorbell[c] = coll_seq(step, c, ncoll), then blocks until the proxy has sent c (a real step uses the reduced
// result of c before it publishes the next one). The proxy thread runs walk_step with a fake doorbell array; its timeout
// and the device's spin timeout are 1 s (fabric.cpp uses 30 s, G6_SPIN_TIMEOUT_CYC the device side).
//   g++ -std=c++17 -O2 -Wall -Wextra -pthread -I engine/comms engine/comms/tests/fabric_order_test.cpp -o fabric_order_test
// The DS41 + Engram plan (83 collectives: 2 NL + 1 = 81 plus the Engram reduces c = 81, 82) publishes c = 81 between c = 1
// and c = 2 and c = 82 between c = 27 and c = 28, so numeric order deadlocks; the same plan with the execution order must pass.
#include "fabric_order.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace g6;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("  FAIL %s:%d %s: ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); g_fail++; } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Run {
    bool proxy_ok = false, device_ok = false;
    unsigned proxy_rc = 0; int proxy_c = -1;          // where the proxy gave up
    int device_c = -1;                                // the collective the device was blocked on
    std::vector<std::vector<int>> sent;               // per step, the collectives the proxy sent, in order
    double secs = 0;
};

// ncoll slots, `steps` steps. program: the collectives the device publishes each step, in its order. order/skip: what the
// proxy is given. nsizes: sizes.size() the proxy is given (collectives 0..nsizes-1 have a size).
static Run run_plan(int ncoll, size_t nsizes, const std::vector<int>& program, const std::vector<int>& order, const CollMask& skip, int steps,
                    double timeout_s = 1.0)
{
    Run r;
    std::vector<std::atomic<unsigned long long>> db(ncoll), done(ncoll);   // doorbell (GPU -> proxy); sent seq (proxy -> GPU)
    for (auto& a : db) a = 0;
    for (auto& a : done) a = 0;
    std::atomic<bool> halt{false};
    r.sent.resize(steps);
    const double t0 = now_s();
    std::thread dev([&] {
        for (int s = 0; s < steps; s++)
            for (int c : program)
            {
                const unsigned long long seq = coll_seq(s, c, ncoll);
                db[c].store(seq, std::memory_order_release);
                const double w0 = now_s();
                while (done[c].load(std::memory_order_acquire) != seq)
                {
                    if (halt.load()) { r.device_c = c; return; }   // the proxy gave up while the device waited for c
                    if (now_s() - w0 > timeout_s) { r.device_c = c; halt = true; return; }
                    std::this_thread::yield();
                }
            }
        r.device_ok = true;
    });
    std::thread proxy([&] {
        for (int s = 0; s < steps; s++)
        {
            const StepWalk w = walk_step(s, ncoll, nsizes, order, skip,
                [&](int c) { return db[c].load(std::memory_order_acquire); },
                [&] { return halt.load(); }, now_s, timeout_s,
                [&](int c, unsigned long long seq) { r.sent[s].push_back(c); done[c].store(seq, std::memory_order_release); return true; });
            if (w.rc != kWalkOk) { r.proxy_rc = w.rc; r.proxy_c = w.c; halt = true; return; }
        }
        r.proxy_ok = true;
    });
    dev.join(); proxy.join();
    r.secs = now_s() - t0;
    return r;
}

static std::vector<int> iota_v(int a, int b) { std::vector<int> v; for (int c = a; c < b; c++) v.push_back(c); return v; }
// [0, 1, 81, 2..27, 82, 28..last): the DS41 + Engram plan. last = 81 for a full step (head c = 80 included), 80 for a chunk with no head
static std::vector<int> engram_program(bool head)
{
    std::vector<int> v = {0, 1, 81};
    for (int c = 2; c <= 27; c++) v.push_back(c);
    v.push_back(82);
    for (int c = 28; c <= (head ? 80 : 79); c++) v.push_back(c);
    return v;
}

static void expect_pass(const char* what, const Run& r, const std::vector<int>& want, int steps)
{
    std::printf("%s: %s (%.3f s)\n", what, r.proxy_ok && r.device_ok ? "PASS" : "FAIL", r.secs);
    CHECK(r.proxy_ok && r.device_ok, "proxy rc %u at c=%d, device blocked at c=%d", r.proxy_rc, r.proxy_c, r.device_c);
    for (int s = 0; s < steps && r.proxy_ok; s++) CHECK(r.sent[s] == want, "step %d sent a different order (%zu collectives)", s, r.sent[s].size());
}
static void expect_stall(const char* what, const Run& r, int proxy_c, int device_c)
{
    std::printf("%s: FAIL as expected (proxy %s at c=%d, device blocked at c=%d, %.2f s)\n", what,
                r.proxy_rc == kWalkTimeout ? "timeout" : "other", r.proxy_c, r.device_c, r.secs);
    CHECK(!r.proxy_ok && !r.device_ok, "the plan was supposed to stall");
    CHECK(r.proxy_rc == kWalkTimeout && r.proxy_c == proxy_c, "proxy rc %u at c=%d, want timeout at c=%d", r.proxy_rc, r.proxy_c, proxy_c);
    CHECK(r.device_c == device_c, "device blocked at c=%d, want c=%d", r.device_c, device_c);
}

int main()
{
    const int steps = 3;
    // 1. Engram ON, full step, 83 collectives. WITHOUT the order vector the proxy waits for doorbell[2] while the device
    //    waits for the send of c=81; WITH it the same plan passes on three consecutive steps.
    {
        const std::vector<int> prog = engram_program(true);
        CHECK(prog.size() == 83, "plan size %zu", prog.size());
        expect_stall("Engram ON  full step, numeric order (no vector)  ", run_plan(83, 83, prog, {}, 0, steps), 2, 81);
        expect_pass("Engram ON  full step, order vector            ", run_plan(83, 83, prog, prog, 0, steps), prog, steps);
    }
    // 2. Engram ON, a chunk with no head: sizes keep one entry per c (83), the order leaves c=80 out. Numeric order waits
    //    for c=2 (stall); a numeric order that only drops the head still stalls; the order vector passes.
    {
        const std::vector<int> prog = engram_program(false);
        expect_stall("Engram ON  no-head chunk, numeric order         ", run_plan(83, 83, prog, {}, 0, steps), 2, 81);
        std::vector<int> num = iota_v(0, 80); num.push_back(81); num.push_back(82);
        expect_stall("Engram ON  no-head chunk, numeric order minus 80", run_plan(83, 83, prog, num, 0, steps), 2, 81);
        expect_pass("Engram ON  no-head chunk, order vector        ", run_plan(83, 83, prog, prog, 0, steps), prog, steps);
    }
    // 3. Engram OFF (81 collectives, program order = numeric order): no vector and the identity vector send the same order.
    {
        const std::vector<int> prog = iota_v(0, 81);
        expect_pass("Engram OFF no vector                          ", run_plan(81, 81, prog, {}, 0, steps), prog, steps);
        expect_pass("Engram OFF identity vector                    ", run_plan(81, 81, prog, prog, 0, steps), prog, steps);
    }
    // 4. skip bits still apply to c < 32 inside an order (GR8_PF_OWN=3 has no collective 0), and never to c >= 32.
    {
        std::vector<int> prog = engram_program(true); prog.erase(prog.begin());   // the device never publishes c=0
        std::vector<int> want = prog;
        expect_pass("Engram ON  skip c=0, order vector (incl. 0)   ", run_plan(83, 83, prog, engram_program(true), 1u, steps), want, steps);
        const std::vector<int> hi = iota_v(32, 83);                                // skip = all 32 bits: c >= 32 is never skipped
        expect_pass("skip mask 0xFFFFFFFF, c>=32 still waited      ", run_plan(83, 83, hi, {}, 0xFFFFFFFFu, steps), hi, steps);
    }
    // 4b. the mask is 128 bits (CollMask). The 32-bit form converts unchanged; bits 32..127 work (DS41 reduce-scatter + all-gather: the
    //     big collectives 0..41 and the AG collectives 83..124 of ncoll 125); c >= 128 reads as 0.
    {
        CollMask m(0x5u);
        CHECK(m.bit(0) && !m.bit(1) && m.bit(2) && !m.bit(32), "unsigned 0x5 converts to bits 0, 2");
        CollMask h; for (int c = 32; c < 128; c += 3) h.set((size_t)c);
        bool ok = true;
        for (int c = 0; c < 160; ++c) ok = ok && (mask_bit(h, (size_t)c) == (c >= 32 && c < 128 && (c - 32) % 3 == 0));
        CHECK(ok, "bits 32..127 set and read back; c >= 128 reads 0");
        h.set(200); CHECK(!h.bit(200), "set(c >= 128) is a no-op");
        CollMask a; a.set(5); a.set(100); CollMask b; b.set(100);
        CHECK(!(b & ~a) && (a & ~b).bit(5) && !(a & ~b).bit(100), "p2p & ~mask over 128 bits");
        CHECK(!CollMask() && CollMask(1u) && a.any(), "bool form");
        std::vector<int> all = iota_v(0, 125), want;
        CollMask sk; for (int c = 83; c < 125; c += 2) sk.set((size_t)c);
        for (int c : all) if (!(c >= 83 && (c - 83) % 2 == 0)) want.push_back(c);
        expect_pass("skip bits 83..124 (128-bit mask), ncoll 125   ", run_plan(125, 125, want, {}, sk, steps), want, steps);
    }
    // 5. walk order without any waiting (every doorbell already rung): the visit sequence is exactly the order, minus skipped
    //    bits; with no order it is the old loop `for c in 0..n-1: if bit(skip, c) continue` for random n and skip masks.
    {
        std::mt19937 rng(12345);
        int checked = 0;
        for (int t = 0; t < 2000; t++)
        {
            const int ncoll = 1 + (int)(rng() % 90);
            const size_t n = 1 + rng() % ncoll;
            const unsigned skip = (t % 3 == 0) ? 0 : (unsigned)rng();
            std::vector<int> order;
            if (t % 2)
            {
                std::vector<int> all = iota_v(0, (int)n);
                std::shuffle(all.begin(), all.end(), rng);
                all.resize(1 + rng() % all.size());
                order = all;
            }
            std::vector<int> got, want;
            const StepWalk w = walk_step(7, ncoll, n, order, skip, [&](int c) { return coll_seq(7, c, ncoll); }, [] { return false; }, now_s, 1.0,
                                         [&](int c, unsigned long long seq) { CHECK(seq == coll_seq(7, c, ncoll), "seq"); got.push_back(c); return true; });
            if (order.empty()) for (size_t c = 0; c < n; c++) { if (!(c < 32 && (skip >> c & 1u))) want.push_back((int)c); }   // the pre-change loop
            else for (int c : order) { if (!(c < 32 && (skip >> c & 1u))) want.push_back(c); }
            CHECK(w.rc == kWalkOk && got == want, "t=%d ncoll=%d n=%zu: visit sequence differs", t, ncoll, n);
            checked++;
        }
        std::printf("random walks (identity == pre-change loop, order == order minus skipped): %d checked\n", checked);
    }
    // 6. order validation (Fabric::enqueue_step aborts on these)
    CHECK(order_error({}, 83) == nullptr, "empty order is fine");
    CHECK(order_error(engram_program(true), 83) == nullptr, "engram order is fine");
    CHECK(order_error(engram_program(false), 83) == nullptr, "no-head order is fine");
    CHECK(order_error({0, 1, 1}, 83) != nullptr, "a repeated collective is refused");
    CHECK(order_error({0, 83}, 83) != nullptr, "c = nsizes is refused");
    CHECK(order_error({-1}, 83) != nullptr, "negative c is refused");
    CHECK(order_error({0, 5}, 3) != nullptr, "c without a size is refused");
    std::printf("order validation: checked\n");

    if (g_fail) { std::printf("fabric_order_test: %d FAILED\n", g_fail); return 1; }
    std::printf("fabric_order_test: PASS (a plan whose publish order differs from c stalls without the order vector and passes with it)\n");
    return 0;
}
