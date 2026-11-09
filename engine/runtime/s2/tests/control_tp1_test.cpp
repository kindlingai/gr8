// TP1: a head with no workers admits an event through the same control protocol, agreeing with itself.
#include "control_channel.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
using namespace g6s2;
int main() {
    LeaderChannel head(1, {});
    std::atomic<int> applied{0};
    std::thread h([&] {
        head.wait_notice();
        for (uint64_t b = 1;; ++b)
            if (auto c = head.boundary(b, [] { return uint64_t(42); })) { head.applied(); ++applied; return; }
    });
    Notice n{1, 1, 20, 7, 0, EventKind::Cancel};
    head.event(n, [&](uint64_t hash) { assert(hash == 42); return Commit{n.epoch, n.event, n.boundary, hash, {}, {}}; });
    h.join(); assert(applied == 1);
    head.sync(1);
    puts("control_tp1_test: PASS");
}
