// TP2 must admit an event through the same control protocol as larger rings.
#include "control_channel.h"
#include <sys/socket.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
using namespace g6s2;
int main() {
    int fd[2]; assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fd));
    ControlIO hi(fd[0]), wi(fd[1]);
    LeaderChannel head(1, {&hi}); WorkerChannel worker(1, 1, wi);
    std::atomic<int> applied{0};
    std::thread reader([&] { try { worker.receive_events(); } catch (...) {} });
    auto compute = [&](EventEndpoint& e) {
        e.wait_notice();
        for (uint64_t b = 1;; ++b) {
            if (auto c = e.boundary(b, [] { return uint64_t(42); })) {
                e.applied(); ++applied; return;
            }
        }
    };
    std::thread h([&] { compute(head); }), w([&] { compute(worker); });
    Notice n{1, 1, 20, 7, 0, EventKind::Cancel};
    head.event(n, [&](uint64_t hash) { return Commit{n.epoch, n.event, n.boundary, hash, {}, {}}; });
    h.join(); w.join(); assert(applied == 2);
    head.fail(); reader.join();
    puts("control_tp2_test: PASS");
}
