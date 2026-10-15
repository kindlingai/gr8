// The barrier between rounds (EventEndpoint::sync) on three ranks over socket pairs: no rank leaves a barrier before
// every rank has reached it; a worker's barrier arrival and its event ACK share one link, in either thread's read; a
// failed head wakes a worker that waits at a barrier.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime/s2 engine/runtime/s2/tests/control_sync_test.cpp -o /tmp/control_sync_test
#include "control_channel.h"
#include <sys/socket.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <thread>

using namespace g6s2;
using Clock = std::chrono::steady_clock;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    constexpr uint64_t epoch = 9;
    constexpr int N = 20;
    int s1[2], s2[2];
    CHECK(!::socketpair(AF_UNIX, SOCK_STREAM, 0, s1) && !::socketpair(AF_UNIX, SOCK_STREAM, 0, s2));
    auto h1 = std::make_unique<ControlIO>(s1[0]), h2 = std::make_unique<ControlIO>(s2[0]);
    ControlIO w1io(s1[1]), w2io(s2[1]);
    h1->timeout_seconds(10); h2->timeout_seconds(10);
    LeaderChannel head(epoch, {h1.get(), h2.get()});
    WorkerChannel w1(epoch, 1, w1io), w2(epoch, 2, w2io);
    std::thread r1([&] { try { w1.receive_events(); } catch (...) {} });
    std::thread r2([&] { try { w2.receive_events(); } catch (...) {} });
    EventEndpoint* ep[3] = {&head, &w1, &w2};

    // every rank arrives late by a different random time at each barrier: nobody leaves before the last one arrives
    Clock::time_point arrive[3][N], leave[3][N];
    {
        std::thread t[3];
        for (int r = 0; r < 3; ++r)
            t[r] = std::thread([&, r] {
                std::mt19937 g(unsigned(r) * 7 + 1);
                for (int k = 0; k < N; ++k) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(g() % 20));
                    arrive[r][k] = Clock::now(); ep[r]->sync(uint64_t(k + 1)); leave[r][k] = Clock::now();
                }
            });
        for (auto& x : t) x.join();
    }
    for (int k = 0; k < N; ++k)
        for (int r = 0; r < 3; ++r)
            for (int q = 0; q < 3; ++q) CHECK(leave[r][k] >= arrive[q][k]);

    // an event while the workers pass a barrier first: each worker's link carries its barrier arrival, then its ACK. The
    // control thread waits for the ACKs and the head's barrier waits for the arrivals, both reading the same links
    {
        const Notice n{epoch, 1, 5, 7, 0, EventKind::Cancel};
        std::atomic<int> applied{0};
        auto compute = [&](EventEndpoint& e, bool barrier) {
            e.wait_notice();
            if (barrier) e.sync(N + 1);
            for (uint64_t b = 1;; ++b)
                if (auto cm = e.boundary(b, [] { return uint64_t(42); })) { e.applied(); ++applied; return; }
        };
        std::thread c1([&] { compute(w1, true); }), c2([&] { compute(w2, true); });
        std::thread c0([&] { compute(head, false); });
        std::thread hb([&] { head.wait_notice(); std::this_thread::sleep_for(std::chrono::milliseconds(50)); head.sync(N + 1); });
        head.event(n, [&](uint64_t hash) { return Commit{n.epoch, n.event, n.boundary, hash, {}, {}}; });
        c0.join(); c1.join(); c2.join(); hb.join();
        CHECK(applied.load() == 3);
    }

    // a worker at a barrier the head never reaches: the head's links close and the worker's barrier throws
    {
        std::atomic<bool> threw{false};
        std::thread w([&] { try { w1.sync(N + 2); } catch (...) { threw = true; } });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(!threw.load());
        head.fail();
        w.join();
        CHECK(threw.load());
    }
    h1.reset(); h2.reset();
    r1.join(); r2.join();
    std::printf(fails ? "control_sync_test: %d FAIL\n" : "control_sync_test: PASS\n", fails);
    return fails != 0;
}
