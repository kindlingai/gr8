// DS41 ranks that disagree on an agree-scope knob refuse at bootstrap: the DS41 runtime signature in g6m4.cu ends with
// knob::signature() (as Flash and GLM do), and four ranks on loopback with GR8_DS_COMP_FP4 [1,1,0,1] refuse with
// "bootstrap config mismatch" while [1,1,1,1] and [0,0,0,0] boot. Each rank is its own process (the knob table reads
// the environment once). Host only.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/s2/tests/ds41_agree_knobs_test.cpp
//          engine/knobs-rs/target/release/libgr8knobs.a -ldl -lm -o /tmp/ds41_agree_knobs_test
// Run:   /tmp/ds41_agree_knobs_test engine/runtime/g6m4.cu
#include "s2/bootstrap.h"
#include "../knobs-rs/gr8_knobs.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// the DS41 signature block of g6m4.cu, from its first line to the bootstrap call that sends it
static std::string ds41_block(const char* path) {
    std::ifstream f(path); std::stringstream ss; ss << f.rdbuf(); const std::string s = ss.str();
    const size_t a = s.find("bc.runtime_signature = std::string(s2sess > 1 ? \"ds41 sessions ");
    if (a == std::string::npos) return "";
    const size_t b = s.find("g6s2::bootstrap(bc)", a);
    return b == std::string::npos ? "" : s.substr(a, b - a);
}

// rank `r` of 4 with GR8_DS_COMP_FP4=fp4; exits 0 when it boots, 3 when it refuses (the head writes its error to `why`)
[[noreturn]] static void rank(unsigned r, int fp4, uint16_t port, const std::string& dir, const std::string& why) {
    if (r) (void)!std::freopen("/dev/null", "w", stderr);   // a refused worker reports its closed control link
    ::setenv("GR8_DS_COMP_FP4", fp4 ? "1" : "0", 1);
    if (knob::init()) std::_Exit(2);
    BootstrapConfig c; c.rank = r; c.ranks = 4; c.timeout = 3; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(r); ::mkdir(c.directory.c_str(), 0700);
    c.runtime_signature = "ds41 solo cap 262144 layers all mtp 0" + knob::signature();
    try { auto links = bootstrap(c); std::_Exit(0); }
    catch (const std::exception& e) {
        if (r == 0) { std::ofstream(why) << e.what(); }
        std::_Exit(3);
    }
}

// the head's exit code and error text for ranks with these GR8_DS_COMP_FP4 values
static int boot(const std::vector<int>& fp4, uint16_t port, const std::string& dir, std::string& why) {
    ::mkdir(dir.c_str(), 0700);
    const std::string wf = dir + "/why";
    std::vector<pid_t> ps;
    for (unsigned r = 0; r < fp4.size(); ++r) {
        const pid_t p = ::fork();
        if (p == 0) rank(r, fp4[r], port, dir, wf);
        ps.push_back(p);
    }
    int head = -1;
    for (size_t r = 0; r < ps.size(); ++r) {
        int st = 0; ::waitpid(ps[r], &st, 0);
        if (r == 0) head = WIFEXITED(st) ? WEXITSTATUS(st) : 128;
    }
    std::ifstream f(wf); std::stringstream ss; ss << f.rdbuf(); why = ss.str();
    return head;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s engine/runtime/g6m4.cu\n", argv[0]); return 2; }
    const std::string blk = ds41_block(argv[1]);
    CHECK(!blk.empty());
    CHECK(blk.find("bc.runtime_signature += knob::signature();") != std::string::npos);

    char tmpl[] = "/tmp/ds41agreeXXXXXX"; const std::string dir = mkdtemp(tmpl);
    const uint16_t port = uint16_t(20000 + getpid() % 20000);
    std::string why;
    CHECK(boot({1, 1, 1, 1}, port, dir + "/on", why) == 0);
    CHECK(boot({0, 0, 0, 0}, uint16_t(port + 1), dir + "/off", why) == 0);
    CHECK(boot({1, 1, 0, 1}, uint16_t(port + 2), dir + "/mixed", why) == 3);
    // the head names the first word that differs: rank 2's signature has no " env DS_COMP_FP4=1" at its end
    CHECK(why.find("bootstrap config mismatch: rank 2 has '', the head has 'env'") != std::string::npos);
    std::printf("ds41_agree_knobs_test: mixed FP4 boot refused: %s\n", why.c_str());

    if (!fails) std::printf("ds41_agree_knobs_test: PASS\n");
    return fails ? 1 : 0;
}
