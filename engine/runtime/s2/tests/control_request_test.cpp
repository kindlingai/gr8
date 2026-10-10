// The head sends an ADMIT's request bytes to every worker over the control links, just before the notice, and passes
// its quit on. Three ranks on socket pairs, the whole event protocol: notice, ACKs, commit, apply.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime/s2 engine/runtime/s2/tests/control_request_test.cpp -o /tmp/control_request_test
#include "control_channel.h"
#include <sys/socket.h>
#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    constexpr uint64_t epoch = 9;
    int s1[2], s2[2];
    CHECK(!::socketpair(AF_UNIX, SOCK_STREAM, 0, s1) && !::socketpair(AF_UNIX, SOCK_STREAM, 0, s2));
    auto h1 = std::make_unique<ControlIO>(s1[0]), h2 = std::make_unique<ControlIO>(s2[0]);
    ControlIO w1io(s1[1]), w2io(s2[1]);
    LeaderChannel head(epoch, {h1.get(), h2.get()});
    WorkerChannel w1(epoch, 1, w1io), w2(epoch, 2, w2io);
    std::atomic<int> quits{0};
    w1.on_quit = w2.on_quit = [&] { ++quits; };
    std::thread r1([&] { try { w1.receive_events(); } catch (...) {} });
    std::thread r2([&] { try { w2.receive_events(); } catch (...) {} });

    // 4 MB of request bytes: a long prompt fits in one frame
    std::string bytes(4u << 20, 'x');
    for (size_t i = 0; i < bytes.size(); i += 4096) bytes[i] = char('a' + i / 4096 % 26);
    std::string seen[2];
    auto compute = [&](EventEndpoint& ep, WorkerChannel* w, std::string* out) {
        const Notice n = ep.wait_notice();
        for (uint64_t b = 1;; ++b) {
            if (auto cm = ep.boundary(b, [] { return uint64_t(42); })) {
                if (w) { *out = w->request(n.request); w->admitted(n.request); }
                ep.applied();
                return;
            }
        }
    };
    std::thread c0([&] { compute(head, nullptr, nullptr); });
    std::thread c1([&] { compute(w1, &w1, &seen[0]); });
    std::thread c2([&] { compute(w2, &w2, &seen[1]); });
    const Notice n{epoch, 1, 20, 7, 0x1234, EventKind::Admit};
    head.event(n, [&](uint64_t hash) { return Commit{n.epoch, n.event, n.boundary, hash, {}, {}}; }, &bytes);
    c0.join(); c1.join(); c2.join();
    CHECK(seen[0] == bytes && seen[1] == bytes);

    bool threw = false;   // admitted: the bytes are gone
    try { w1.request(7); } catch (...) { threw = true; }
    CHECK(threw);

    head.quit();          // the head's quit reaches every worker
    for (int i = 0; i < 200 && quits.load() < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(quits.load() == 2);

    threw = false;        // an ADMIT notice without its bytes fails the epoch (and closes the head's links)
    try { head.event(Notice{epoch, 2, 40, 8, 0, EventKind::Admit}, [](uint64_t) { return Commit{}; }); } catch (...) { threw = true; }
    CHECK(threw);

    h1.reset(); h2.reset();   // the head closes its links: the workers' readers end
    r1.join(); r2.join();
    std::printf(fails ? "control_request_test: %d FAIL\n" : "control_request_test: PASS\n", fails);
    return fails != 0;
}
