// Ranks of different control builds refuse each other at bootstrap (s2/bootstrap.h agreement carries CONTROL_VERSION,
// scheduler.h) and a commit frame of another version is refused (control_protocol.h magic "G6S2E" + the version). Three
// ranks on loopback: all at CONTROL_VERSION boot; a worker acting as the previous build (control_version 6, the frame
// before Commit::resize) makes the head refuse with "bootstrap config mismatch". Host only.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/s2/tests/control_version_test.cpp -o /tmp/control_version_test
#include "s2/bootstrap.h"
#include "s2/control_protocol.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static BootstrapConfig config(unsigned rank, uint16_t port, const std::string& dir, unsigned version) {
    BootstrapConfig c; c.rank = rank; c.ranks = 3; c.timeout = 2; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(rank); c.runtime_signature = "test"; c.control_version = version;
    ::mkdir(c.directory.c_str(), 0700);
    return c;
}

// worker 2 runs `version`; returns whether the head booted, and the head's error text
static bool boot(uint16_t port, const std::string& dir, unsigned version, std::string& why) {
    ::mkdir(dir.c_str(), 0700);
    std::vector<std::thread> w;
    for (unsigned r = 1; r < 3; ++r) w.emplace_back([&, r] {
        try { auto links = bootstrap(config(r, port, dir, r == 2 ? version : CONTROL_VERSION)); } catch (...) {}
    });
    bool ok = true;
    try { auto links = bootstrap(config(0, port, dir, CONTROL_VERSION)); }
    catch (const std::exception& e) { ok = false; why = e.what(); }
    for (auto& t : w) t.join();
    return ok;
}

int main() {
    CHECK(CONTROL_VERSION == 9);
    BootstrapConfig c; c.runtime_signature = "x";
    CHECK(c.agreement().find(" ctl 9\n") != std::string::npos);

    char tmpl[] = "/tmp/ctlverXXXXXX"; const std::string dir = mkdtemp(tmpl);
    const uint16_t port = uint16_t(20000 + getpid() % 20000);
    std::string why;
    CHECK(boot(port, dir + "/same", CONTROL_VERSION, why));
    const bool mixed = boot(uint16_t(port + 1), dir + "/mixed", 6, why);
    CHECK(!mixed);
    CHECK(why.find("bootstrap config mismatch") != std::string::npos);
    std::printf("control_version_test: mixed boot refused: %s\n", why.c_str());

    // a frame with the previous magic (G6S2E6) is refused before any field is read
    auto b = encode(Notice{9, 1, 5, 11, 0, EventKind::Admit});
    b[7] = uint8_t(b[7] - 1);
    bool threw = false;
    try { (void)decode_notice(b); } catch (const std::exception& e) { threw = std::strstr(e.what(), "control version mismatch") != nullptr; }
    CHECK(threw);

    if (!fails) std::printf("control_version_test: PASS\n");
    return fails ? 1 : 0;
}
