// The all-rank barrier after the boot graph capture: the head returns only after every worker has reported, even when a
// worker is later than the bootstrap/socket timeout; a worker that dies before reporting fails the head.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/s2/tests/capture_barrier_test.cpp -o /tmp/capture_barrier_test
#include "s2/bootstrap.h"
#include <atomic>
#include <cstdio>
#include <sys/stat.h>
#include <vector>
#include <cstdlib>
#include <thread>
#include <unistd.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static BootstrapConfig config(unsigned rank, uint16_t port, const std::string& dir) {
    BootstrapConfig c; c.rank = rank; c.ranks = 4; c.timeout = 1; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(rank); c.runtime_signature = "test";
    ::mkdir(c.directory.c_str(), 0700);
    return c;
}

// delay_ms: capture time of worker 3; die: worker 3 closes its link without reporting
static bool run(uint16_t port, const std::string& dir, int delay_ms, bool die, long& head_ms) {
    std::vector<std::thread> w; std::atomic<bool> bad{false};
    for (unsigned r = 1; r < 4; ++r) w.emplace_back([&, r] {
        try {
            auto c = config(r, port, dir); auto links = bootstrap(c);
            if (r == 3) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            if (r == 3 && die) { links.shutdown(); return; }
            capture_barrier(c, links);
        } catch (...) { bad = true; }
    });
    bool ok = true;
    const auto t0 = std::chrono::steady_clock::now();
    try { auto c = config(0, port, dir); auto links = bootstrap(c); capture_barrier(c, links); }
    catch (...) { ok = false; }
    head_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    for (auto& t : w) t.join();
    return ok && !bad;
}

int main() {
    char tmpl[] = "/tmp/capbarXXXXXX"; const std::string dir = mkdtemp(tmpl);
    const uint16_t port = uint16_t(20000 + getpid() % 20000);
    long ms = 0;
    CHECK(run(port, dir + "", 3000, false, ms));        // worker later than the 1 s socket timeout
    CHECK(ms >= 3000);
    const std::string dir2 = dir + "/b"; ::mkdir(dir2.c_str(), 0700);
    CHECK(!run(uint16_t(port + 1), dir2, 200, true, ms)); // a worker that dies before reporting
    if (!fails) std::printf("capture_barrier_test: PASS\n");
    return fails ? 1 : 0;
}
