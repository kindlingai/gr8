// The bytes on the control links of a scripted TP3 boot, against hashes recorded from the C++ that engine/scheduler-rs's
// control code replaced (git 5381433c). The workers reach the head through a relay on loopback that records each link's
// two directions: the bootstrap's hello, release and ACK, a store agreement, the capture barrier, then the event channel
// (an admission held at its boundary, a second one that releases it, a cancel and a resize, each with its request bytes
// and ACKs, a barrier after each, and the head's quit). Every step waits for the one before, so each direction carries
// its frames in one order. Host only.
#include "../bootstrap.h"
#include "../control_channel.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;
namespace {
int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

uint64_t fnv(const std::vector<uint8_t>& b) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
    return h;
}
int tcp_listen(uint16_t port) {
    const int l = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    ::setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (posix_bind(l, reinterpret_cast<sockaddr*>(&a), sizeof a) || ::listen(l, 8)) { std::perror("relay listen"); std::exit(1); }
    return l;
}
int tcp_connect(uint16_t port) {
    for (;;) {
        const int c = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (!::connect(c, reinterpret_cast<sockaddr*>(&a), sizeof a)) return c;
        ::close(c);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
// one worker's link through the relay: what each direction carried
struct Tap {
    int worker = -1, head = -1;
    std::vector<uint8_t> up, down;   // worker to head, head to worker
    std::thread a, b;
    static void pump(int from, int to, std::vector<uint8_t>& out) {
        uint8_t buf[65536];
        for (;;) {
            const ssize_t k = ::recv(from, buf, sizeof buf, 0);
            if (k <= 0) break;
            out.insert(out.end(), buf, buf + k);
            for (ssize_t o = 0; o < k;) {
                const ssize_t w = ::send(to, buf + o, size_t(k - o), SEND_FLAGS);
                if (w <= 0) return;
                o += w;
            }
        }
        ::shutdown(to, SHUT_WR);
    }
    void start() {
        a = std::thread([this] { pump(worker, head, up); });
        b = std::thread([this] { pump(head, worker, down); });
    }
    void join() { a.join(); b.join(); ::close(worker); ::close(head); }
};
// the rank a tap carried: its first frame is the worker's hello, "G6S2 <rank>\n..."
unsigned rank_of(const Tap& t) { return t.up.size() > 9 ? unsigned(t.up[9] - '0') : 0; }

BootstrapConfig config(unsigned rank, uint16_t port, const std::string& dir) {
    BootstrapConfig c; c.rank = rank; c.ranks = 3; c.timeout = 5; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(rank); c.runtime_signature = "wire test signature";
    ::mkdir(c.directory.c_str(), 0700);
    return c;
}
// a count every compute thread bumps once a step is done, so the head starts the next one after all of them
struct Steps {
    std::mutex mu; std::condition_variable cv; int done = 0;
    void bump() { std::lock_guard<std::mutex> l(mu); ++done; cv.notify_all(); }
    void wait(int n) { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return done >= n; }); }
};
} // namespace

int main() {
    char tmpl[] = "/tmp/ctlwireXXXXXX"; const std::string dir = mkdtemp(tmpl);
    const uint16_t head_port = uint16_t(20000 + getpid() % 20000), relay_port = uint16_t(head_port + 1);
    const int relay = tcp_listen(relay_port);
    Tap taps[2];
    std::thread relay_thread([&] {
        for (auto& t : taps) {
            t.worker = ::accept(relay, nullptr, nullptr);
            t.head = tcp_connect(head_port);
            t.start();
        }
    });
    constexpr uint64_t epoch = 1;
    const std::string lines[3] = {"a\nb\nc\n", "b\nc\nd\n", "c\nb\n"};
    std::string agreed[3];
    ControlLinks links[3];
    BootstrapConfig cfg[3];
    std::thread boot[3];
    for (unsigned r = 0; r < 3; ++r)
        boot[r] = std::thread([&, r] {
            try {
                cfg[r] = config(r, r ? relay_port : head_port, dir);
                links[r] = bootstrap(cfg[r]);
                agreed[r] = agree_lines(cfg[r], links[r], lines[r], r == 2);
                capture_barrier(cfg[r], links[r]);
            } catch (const std::exception& e) { std::fprintf(stderr, "rank %u: %s\n", r, e.what()); ++fails; }
        });
    for (auto& t : boot) t.join();
    relay_thread.join();
    for (const auto& a : agreed) CHECK(a == "b\nc\n");

    LeaderChannel head(epoch, links[0].peers());
    WorkerChannel w1(epoch, 1, *links[1].links[0]), w2(epoch, 2, *links[2].links[0]);
    head.room = [] { return uint32_t(17); };
    w1.room = [] { return uint32_t(5); };
    std::atomic<int> quits{0};
    w1.on_quit = [&] { ++quits; };
    w2.on_quit = [&] { ++quits; };
    std::thread r1([&] { try { w1.receive_events(); } catch (...) {} });
    std::thread r2([&] { try { w2.receive_events(); } catch (...) {} });

    struct Ev { EventKind kind; uint64_t boundary; bool hold; };
    const Ev evs[] = {{EventKind::Admit, 20, true}, {EventKind::Admit, 20, false}, {EventKind::Cancel, 30, false},
                      {EventKind::Resize, 40, false}};
    constexpr int NE = 4;
    Steps steps;
    std::string got[3];
    auto compute = [&](EventEndpoint& ep, WorkerChannel* w, unsigned rank) {
        uint64_t b = 1;
        for (int i = 0; i < NE; ++i) {
            (void)ep.wait_notice();
            for (;;) {
                Notice n{};
                auto cm = ep.boundary(b, [&] {
                    n = ep.wait_notice();
                    if (w && n.kind == EventKind::Admit) got[rank] += w->request(n.request) + ";";
                    return uint64_t(0xD16E57 + i);
                });
                if (cm) {
                    if (w && cm->admit) w->admitted(cm->admit->req.id);
                    ep.applied();
                    break;
                }
                ++b;
            }
            ep.sync(uint64_t(i + 1));
            steps.bump();
        }
    };
    std::thread c0([&] { compute(head, nullptr, 0); }), c1([&] { compute(w1, &w1, 1); }), c2([&] { compute(w2, &w2, 2); });
    for (int i = 0; i < NE; ++i) {
        const Ev& e = evs[i];
        const std::string payload = "request " + std::to_string(i);
        const Notice n{epoch, uint64_t(i + 1), e.boundary, uint64_t(100 + i), payload_hash(payload), e.kind};
        head.event(n, [&](uint64_t hash) {
            Commit c{n.epoch, n.event, n.boundary, hash, {}, {}, e.hold, {}, {}};
            if (e.kind == EventKind::Admit) {
                Grant g; g.key = Key{epoch, unsigned(i), 1}; g.req = Request{n.request, 64, 8}; g.run = Run{unsigned(i) * 4, 4, 1, 5u + i};
                g.req.r_rows = PAGE * 4;
                c.admit = g;
                if (i == 1) c.shrink = Shrink{Key{epoch, 0, 1}, PAGE * 2, Run{0, 4, 1, 99}};
            } else if (e.kind == EventKind::Cancel) {
                c.cancel = Key{epoch, 0, 1};
            } else {
                c.resize = Resize{2, 7, 896, 31};
            }
            return c;
        }, e.kind == EventKind::Admit ? &payload : nullptr);
        steps.wait(3 * (i + 1));
    }
    c0.join(); c1.join(); c2.join();
    CHECK(got[1] == "request 0;request 1;" && got[2] == got[1]);
    head.quit();
    while (quits.load() < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    head.fail();
    r1.join(); r2.join();
    for (auto& l : links) l.links.clear();   // closed: macOS may not pass on a shutdown after the peer's own
    for (auto& t : taps) t.join();
    ::close(relay);

    // per rank: worker to head, head to worker (recorded from git 5381433c)
    const uint64_t want[2][2] = {{0x792fe30be6938149ull, 0x4878724697c93fbdull}, {0x8c244dbd1c2adfdaull, 0x4878724697c93fbdull}};
    for (const auto& t : taps) {
        const unsigned r = rank_of(t);
        CHECK(r == 1 || r == 2);
        if (r != 1 && r != 2) continue;
        const uint64_t up = fnv(t.up), down = fnv(t.down);
        std::printf("rank %u: up %zu bytes 0x%016llxull, down %zu bytes 0x%016llxull\n", r, t.up.size(), (unsigned long long)up,
                    t.down.size(), (unsigned long long)down);
        CHECK(up == want[r - 1][0] && down == want[r - 1][1]);
    }
    std::printf(fails ? "control_wire_test: %d FAIL\n" : "control_wire_test: PASS\n", fails);
    return fails != 0;
}
