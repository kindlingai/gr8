// A held boundary (Commit::hold, event_gate.h EventGate::applied): after an admit with hold every rank stays stopped at
// the event's boundary, the next notice names that SAME boundary, and a commit without hold releases it. Gate level: a
// held boundary is not departed, a notice for any other boundary is refused, and the release departs. Three ranks on
// socket pairs: three admissions chained at one boundary (hold, hold, release), the hold flag carried by the commit
// frame, and every rank departs the boundary only after the release.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime/s2 engine/runtime/s2/tests/control_hold_test.cpp -o /tmp/control_hold_test
#include "control_channel.h"
#include <sys/socket.h>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
template<class F> static bool throws(F f) { try { f(); } catch (...) { return true; } return false; }

static void gate_holds() {
    constexpr uint64_t epoch = 3;
    EventGate g(epoch, 0);
    CHECK(!g.before_work(1));
    const Notice n1{epoch, 1, 5, 11, 0, EventKind::Admit};
    g.announce(n1);
    CHECK(!g.before_work(4));
    CHECK(g.before_work(5));
    g.arrive(5, 1); g.commit(n1); g.applied(n1, true);
    CHECK(g.before_work(5));              // held: still stopped at 5, not departed
    CHECK(g.before_work(5));
    const Notice n2{epoch, 2, 5, 12, 0, EventKind::Admit};
    g.announce(n2);                       // the same boundary: accepted
    CHECK(g.before_work(5));
    g.arrive(5, 2); g.commit(n2); g.applied(n2);
    CHECK(!g.before_work(5));             // released: departs
    CHECK(!g.before_work(6));

    EventGate h(epoch, 0);                // a notice for another boundary while held is refused
    const Notice m1{epoch, 1, 5, 11, 0, EventKind::Admit};
    h.announce(m1); CHECK(h.before_work(5)); h.arrive(5, 1); h.commit(m1); h.applied(m1, true);
    CHECK(throws([&] { h.announce(Notice{epoch, 2, 7, 12, 0, EventKind::Admit}); }));
}

static void frame_carries_hold() {
    Grant gr; gr.key = Key{3, 1, 1}; gr.req = Request{11, 100, 10}; gr.run = Run{0, 2, 1, 77};
    for (bool hold : {false, true}) {
        EventCommit e{Notice{3, 1, 5, 11, 0, EventKind::Admit}, Commit{3, 1, 5, 99, {}, gr, hold}};
        const auto d = decode_commit(encode(e));
        CHECK(d.commit.hold == hold && d.commit.admit && d.commit.admit->req.id == 11);
    }
    // hold without a grant is refused at both ends
    EventCommit bad{Notice{3, 1, 5, 11, 0, EventKind::Admit}, Commit{3, 1, 5, 99, {}, {}, true}};
    CHECK(throws([&] { encode(bad); }));
}

static void chain_three_ranks() {
    constexpr uint64_t epoch = 9, B = 20;
    int s1[2], s2[2];
    CHECK(!::socketpair(AF_UNIX, SOCK_STREAM, 0, s1) && !::socketpair(AF_UNIX, SOCK_STREAM, 0, s2));
    auto h1 = std::make_unique<ControlIO>(s1[0]), h2 = std::make_unique<ControlIO>(s2[0]);
    ControlIO w1io(s1[1]), w2io(s2[1]);
    h1->timeout_seconds(10); h2->timeout_seconds(10);
    LeaderChannel head(epoch, {h1.get(), h2.get()});
    WorkerChannel w1(epoch, 1, w1io), w2(epoch, 2, w2io);
    std::thread r1([&] { try { w1.receive_events(); } catch (...) {} });
    std::thread r2([&] { try { w2.receive_events(); } catch (...) {} });
    const std::string bytes = "req";

    // the service compute loop (service.h compute), cut down: depart boundaries until one is due, apply its commit, and
    // re-enter the SAME boundary (the loop's continue); the first boundary departed after a commit ends the run
    struct Seen { std::vector<uint64_t> requests, boundaries; std::vector<bool> holds; uint64_t departed = 0;
                 std::vector<std::pair<long long, long long>> stamps; };   // GR8_S2_EVENTLOG installed/arrived
    Seen seen[3];
    auto compute = [&](EventEndpoint& ep, WorkerChannel* w, Seen& out) {
        (void)ep.wait_notice();
        for (uint64_t b = 1;;) {
            Notice noticed{};
            auto cm = ep.boundary(b, [&] { noticed = ep.wait_notice(); return uint64_t(42); });
            if (cm) {
                out.requests.push_back(noticed.request); out.boundaries.push_back(cm->boundary); out.holds.push_back(cm->hold);
                out.stamps.push_back(ep.stamps());
                if (w) w->admitted(noticed.request);
                ep.applied();
                continue;                             // same boundary again
            }
            if (!out.requests.empty()) { out.departed = b; return; }
            ++b;
        }
    };
    std::thread c0([&] { compute(head, nullptr, seen[0]); });
    std::thread c1([&] { compute(w1, &w1, seen[1]); });
    std::thread c2([&] { compute(w2, &w2, seen[2]); });
    const bool holds[3] = {true, true, false};
    for (uint64_t i = 0; i < 3; ++i) {
        const Notice n{epoch, i + 1, B, 100 + i, 0, EventKind::Admit};
        head.event(n, [&](uint64_t hash) {
            Grant g; g.key = Key{epoch, unsigned(i), 1}; g.req = Request{100 + i, 64, 8}; g.run = Run{unsigned(i) * 4, 4, 1, 5 + i};
            return Commit{n.epoch, n.event, n.boundary, hash, {}, g, holds[i]};
        }, &bytes);
    }
    c0.join(); c1.join(); c2.join();
    for (const auto& s : seen) {
        CHECK((s.requests == std::vector<uint64_t>{100, 101, 102}));
        CHECK((s.boundaries == std::vector<uint64_t>{B, B, B}));
        CHECK((s.holds == std::vector<bool>{true, true, false}));
        CHECK(s.departed == B);                       // departed only after the release
        CHECK(s.stamps.size() == 3);
        for (const auto& t : s.stamps) CHECK(t.first > 0 && t.second >= t.first);   // installed, then arrived at B
    }
    h1.reset(); h2.reset();
    r1.join(); r2.join();
}

int main() {
    gate_holds();
    frame_carries_hold();
    chain_three_ranks();
    std::printf(fails ? "control_hold_test: %d FAIL\n" : "control_hold_test: PASS\n", fails);
    return fails != 0;
}
