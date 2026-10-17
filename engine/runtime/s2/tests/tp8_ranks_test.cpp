// S2 at TP8: the rank-count guards accept 8 ranks / 7 peers and still refuse other counts; the bootstrap and the capture
// barrier run over 8 loopback sockets; one ADMIT-style event goes through LeaderChannel with 7 peers, every worker ACKs,
// and Agreement(8) needs all 8 ACKs.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/s2/tests/tp8_ranks_test.cpp -o /tmp/tp8_ranks_test
#include "s2/bootstrap.h"
#include "s2/control_channel.h"
#include <sys/socket.h>
#include <sys/stat.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
template <class F> static bool rejects(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static void guards() {
    Config c;
    CHECK(capability(c, 8).find("G6S2 3 1 8 ") == 0);
    for (unsigned n : {3u, 4u, 6u, 8u}) CHECK(!rejects([&] { capability(c, n); }));
    for (unsigned n : {0u, 1u, 2u, 5u, 7u, 9u, 16u}) CHECK(rejects([&] { capability(c, n); }));
    BootstrapConfig b; b.ranks = 8; b.rank = 7; b.head_ip = "127.0.0.1"; b.port = 1; b.directory = "/x"; b.runtime_signature = "s";
    CHECK(!rejects([&] { b.validate(); }));
    b.rank = 8; CHECK(rejects([&] { b.validate(); }));
    b.rank = 0; b.ranks = 7; CHECK(rejects([&] { b.validate(); }));
    const Notice n{1, 1, 5, 7, 0, EventKind::Cancel};
    CHECK(!rejects([&] { Agreement a(8, n); }));
    CHECK(rejects([&] { Agreement a(7, n); }));
    CHECK(rejects([&] { Agreement a(9, n); }));
    // Agreement(8): ready only after all 8 ranks ACK with one state hash; a 9th rank id or a duplicate is refused
    Agreement a(8, n);
    for (unsigned r = 0; r < 8; ++r) { CHECK(!a.ready()); a.ack(Ack{n, r, 77}); }
    CHECK(a.ready() && a.agreed_hash() == 77);
    Agreement d(8, n); d.ack(Ack{n, 3, 1});
    CHECK(rejects([&] { d.ack(Ack{n, 3, 1}); }));
    CHECK(rejects([&] { d.ack(Ack{n, 8, 1}); }));
    CHECK(rejects([&] { d.ack(Ack{n, 4, 2}); }));
    // LeaderChannel peer count: 7 accepted, 6 refused
    std::vector<std::unique_ptr<ControlIO>> io; std::vector<ControlIO*> p;
    std::vector<int> other;
    for (int i = 0; i < 7; ++i) { int s[2]; CHECK(!::socketpair(AF_UNIX, SOCK_STREAM, 0, s)); io.emplace_back(new ControlIO(s[0])); other.push_back(s[1]); p.push_back(io.back().get()); }
    CHECK(!rejects([&] { LeaderChannel h(1, p); }));
    std::vector<ControlIO*> six(p.begin(), p.begin() + 6);
    CHECK(rejects([&] { LeaderChannel h(1, six); }));
    for (int fd : other) ::close(fd);
}

static BootstrapConfig config(unsigned rank, uint16_t port, const std::string& dir) {
    BootstrapConfig c; c.rank = rank; c.ranks = 8; c.timeout = 10; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(rank); c.runtime_signature = "tp8-test";
    ::mkdir(c.directory.c_str(), 0700);
    return c;
}

// 8 ranks bootstrap on loopback, pass the capture barrier, and the head publishes ready.s2 with "8" as the rank count
static void bootstrap8(uint16_t port, const std::string& dir) {
    std::vector<std::thread> w; std::atomic<int> bad{0};
    for (unsigned r = 1; r < 8; ++r) w.emplace_back([&, r] {
        try { auto c = config(r, port, dir); auto links = bootstrap(c); CHECK(links.links.size() == 1); capture_barrier(c, links); }
        catch (const std::exception& e) { std::fprintf(stderr, "worker %u: %s\n", r, e.what()); ++bad; }
    });
    try {
        auto c = config(0, port, dir); auto links = bootstrap(c);
        CHECK(links.links.size() == 7);
        capture_barrier(c, links);
        publish_ready(c, links);
        const std::string ready = read_file(c.directory + "/ready.s2");
        CHECK(ready.find("G6S2 3 1 8 ") == 0);
    } catch (const std::exception& e) { std::fprintf(stderr, "head: %s\n", e.what()); ++bad; }
    for (auto& t : w) t.join();
    CHECK(bad == 0);
}

// one event through LeaderChannel with 7 peers over socket pairs: every one of 8 ranks applies the commit
static void event8() {
    constexpr uint64_t epoch = 11;
    std::vector<std::unique_ptr<ControlIO>> hio; std::vector<std::unique_ptr<ControlIO>> wio;
    std::vector<ControlIO*> peers;
    for (int i = 0; i < 7; ++i) {
        int s[2]; CHECK(!::socketpair(AF_UNIX, SOCK_STREAM, 0, s));
        hio.emplace_back(new ControlIO(s[0])); wio.emplace_back(new ControlIO(s[1]));
        hio.back()->timeout_seconds(10); peers.push_back(hio.back().get());
    }
    LeaderChannel head(epoch, peers);
    std::vector<std::unique_ptr<WorkerChannel>> wk;
    for (unsigned r = 1; r < 8; ++r) wk.emplace_back(new WorkerChannel(epoch, r, *wio[r - 1]));
    std::vector<std::thread> readers;
    for (auto& w : wk) readers.emplace_back([&w] { try { w->receive_events(); } catch (...) {} });
    const Notice n{epoch, 1, 5, 7, 0, EventKind::Cancel};
    std::atomic<int> applied{0};
    auto compute = [&](EventEndpoint& e) {
        e.wait_notice();
        for (uint64_t b = 1;; ++b)
            if (auto cm = e.boundary(b, [] { return uint64_t(42); })) { e.applied(); ++applied; return; }
    };
    std::vector<std::thread> c;
    c.emplace_back([&] { compute(head); });
    for (auto& w : wk) c.emplace_back([&] { compute(*w); });
    head.event(n, [&](uint64_t hash) { CHECK(hash == 42); return Commit{n.epoch, n.event, n.boundary, hash, {}, {}}; });
    for (auto& t : c) t.join();
    CHECK(applied.load() == 8);
    head.fail();
    hio.clear();
    for (auto& t : readers) t.join();
}

int main() {
    guards();
    char tmpl[] = "/tmp/tp8s2XXXXXX"; const std::string dir = mkdtemp(tmpl);
    bootstrap8(uint16_t(20000 + getpid() % 20000), dir);
    event8();
    std::printf(fails ? "tp8_ranks_test: %d FAIL\n" : "tp8_ranks_test: PASS\n", fails);
    return fails != 0;
}
