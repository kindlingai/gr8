// #26: a worker's event reader must survive a silent control link longer than the socket timeout,
// while a frame that stalls mid-way still times out and a closed link still ends the reader.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime/s2 engine/runtime/s2/tests/control_idle_test.cpp -o /tmp/control_idle_test
// Run:   /tmp/control_idle_test                      (socketpair cases; the silent-peer case is skipped)
//        unshare -rn /tmp/control_idle_test netns    (also the silent, still-open TCP peer: a private network namespace
//                                                     where the test may drop all loopback packets with tc netem)
#include "control_io.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static void pair(int sv[2]) { CHECK(!::socketpair(AF_UNIX, SOCK_STREAM, 0, sv)); }

int main(int argc, char** argv) {
    const bool netns = argc > 1 && !std::strcmp(argv[1], "netns");
    { // idle longer than the timeout, then a frame: received
        int sv[2]; pair(sv); ControlIO a(sv[0]), b(sv[1]); b.timeout_seconds(1);
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(2500)); a.send({1, 2, 3}); });
        bool ok = false;
        try { auto f = b.receive(true); ok = f.size() == 3 && f[2] == 3; } catch (...) {}
        t.join(); CHECK(ok);
    }
    { // same silence without idle: times out (the pre-fix behaviour, kept for ACK/bootstrap reads)
        int sv[2]; pair(sv); ControlIO a(sv[0]), b(sv[1]); b.timeout_seconds(1);
        bool threw = false; try { b.receive(false); } catch (...) { threw = true; }
        CHECK(threw);
    }
    { // idle, but the frame stalls after its header: still times out
        int sv[2]; pair(sv); ControlIO b(sv[1]); b.timeout_seconds(1);
        const uint8_t h[4] = {0, 0, 0, 8}; CHECK(::send(sv[0], h, 4, 0) == 4);
        bool threw = false; try { b.receive(true); } catch (...) { threw = true; }
        CHECK(threw); ::close(sv[0]);
    }
    { // idle, but only part of the header arrives: times out
        int sv[2]; pair(sv); ControlIO b(sv[1]); b.timeout_seconds(1);
        const uint8_t h[1] = {0}; CHECK(::send(sv[0], h, 1, 0) == 1);
        bool threw = false; try { b.receive(true); } catch (...) { threw = true; }
        CHECK(threw); ::close(sv[0]);
    }
    { // idle reader is woken by shutdown() (epoch failure path)
        int sv[2]; pair(sv); ControlIO a(sv[0]), b(sv[1]); b.timeout_seconds(1);
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(1500)); b.shutdown(); });
        bool threw = false; try { b.receive(true); } catch (...) { threw = true; }
        t.join(); CHECK(threw);
    }
    { // idle reader ends when the peer closes
        int sv[2]; pair(sv); ControlIO b(sv[1]); b.timeout_seconds(1);
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(1500)); ::close(sv[0]); });
        bool threw = false; try { b.receive(true); } catch (...) { threw = true; }
        t.join(); CHECK(threw);
    }
    if (netns) { // silent, still-open TCP peer (no FIN/RST): keepalive must end the idle wait
        CHECK(!std::system("ip link set lo up"));
        int l = ::socket(AF_INET, SOCK_STREAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t al = sizeof a; const int one = 1; setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        CHECK(!posix_bind(l, reinterpret_cast<sockaddr*>(&a), sizeof a) && !::listen(l, 1) && !getsockname(l, reinterpret_cast<sockaddr*>(&a), &al));
        int c = ::socket(AF_INET, SOCK_STREAM, 0); CHECK(!::connect(c, reinterpret_cast<sockaddr*>(&a), sizeof a));
        int s = ::accept(l, nullptr, nullptr); CHECK(s >= 0);
        ControlIO peer(s), b(c); b.timeout_seconds(1); b.keepalive(1, 1, 2);
        CHECK(!std::system("tc qdisc add dev lo root netem loss 100%")); // the peer stays open but nothing gets through
        const auto t0 = std::chrono::steady_clock::now(); bool threw = false;
        std::thread guard([&] { std::this_thread::sleep_for(std::chrono::seconds(20)); b.shutdown(); }); // test bound only
        try { b.receive(true); } catch (...) { threw = true; }
        const double s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        guard.detach(); std::printf("silent peer detected after %.1f s\n", s_);
        CHECK(threw && s_ < 10.0); ::close(l);
    } else std::printf("silent-peer case skipped (run under: unshare -rn <test> netns)\n");
    std::printf(fails ? "control_idle_test: %d FAIL\n" : "control_idle_test: PASS\n", fails);
    return fails != 0;
}
