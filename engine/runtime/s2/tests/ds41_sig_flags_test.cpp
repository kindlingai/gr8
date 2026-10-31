// Ranks that disagree on GR8_DS_SH_TC or GR8_DS_MOE_L2HINT refuse each other at bootstrap. The DS41 runtime_signature (g6m4.cu,
// bc.runtime_signature) ends "... set <id> shtc <0|1>" + optional l2h/stg and, when set, tkmain/tkdecode; BootstrapConfig::agreement (s2/bootstrap.h)
// carries it, so a mixed boot fails with "bootstrap config mismatch". The suffix below is the g6m4.cu expression with the flag readers
// stubbed (host only); argv[1] = g6m4.cu, whose signature expression must equal SIG_LINES byte for byte (drift fails the test).
// Three ranks on loopback (as control_version_test). Build: g++ -std=c++17 -O2 -pthread -I engine/runtime <this> -o /tmp/ds41_sig_flags_test
#include "s2/bootstrap.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace ds41moe { static int tc_v = 0, h_v = 0, g_v = 0, b_v = 0; inline bool sh_tc_on() { return tc_v != 0; } inline int ro_l2hint() { return h_v; }
                    inline bool ro_fc2stg() { return g_v != 0; } inline bool ro_fc1bstg() { return b_v != 0; } }

static const char* SIG_LINES =
"            \" shtc \" + std::to_string(ds41moe::sh_tc_on() ? 1 : 0) +   // GR8_DS_SH_TC: ranks must agree (W8A8 shared expert, not bit-identical to 0)\n"
"            (ds41moe::ro_l2hint() ? \" l2h \" + std::to_string(ds41moe::ro_l2hint()) : std::string()) +   // GR8_DS_MOE_L2HINT: 0 adds nothing; ranks that disagree on 1 / 2 refuse to join\n"
"            (ds41moe::ro_fc2stg() ? std::string(\" stg 1\") : std::string()) +   // GR8_DS_MOE_FC2STG: 0 adds nothing; ranks that disagree refuse to join\n"
"            (ds41moe::ro_fc1bstg() ? std::string(\" fc1bstg 1\") : std::string()); // off adds nothing; ranks with the staged B arm must agree\n";

static std::string sig(int tc, int h, int g = 0, int b = 0) {   // the g6m4.cu tail, from " set " + set_id on
    ds41moe::tc_v = tc; ds41moe::h_v = h; ds41moe::g_v = g; ds41moe::b_v = b;
    const std::string set_id = "s";
    return std::string("ds41 sessions 8 cap 131072") + " set " + set_id +
            " shtc " + std::to_string(ds41moe::sh_tc_on() ? 1 : 0) +
            (ds41moe::ro_l2hint() ? " l2h " + std::to_string(ds41moe::ro_l2hint()) : std::string()) +
            (ds41moe::ro_fc2stg() ? std::string(" stg 1") : std::string()) +
            (ds41moe::ro_fc1bstg() ? std::string(" fc1bstg 1") : std::string());
}

static BootstrapConfig config(unsigned rank, uint16_t port, const std::string& dir, const std::string& s) {
    BootstrapConfig c; c.rank = rank; c.ranks = 3; c.timeout = 2; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(rank); c.runtime_signature = s;
    ::mkdir(c.directory.c_str(), 0700);
    return c;
}

// head and worker 1 run `a`, worker 2 runs `b`; returns whether the head booted, and the head's error text
static bool boot(uint16_t port, const std::string& dir, const std::string& a, const std::string& b, std::string& why) {
    ::mkdir(dir.c_str(), 0700);
    std::vector<std::thread> w;
    for (unsigned r = 1; r < 3; ++r) w.emplace_back([&, r] {
        try { auto links = bootstrap(config(r, port, dir, r == 2 ? b : a)); } catch (...) {}
    });
    bool ok = true;
    try { auto links = bootstrap(config(0, port, dir, a)); }
    catch (const std::exception& e) { ok = false; why = e.what(); }
    for (auto& t : w) t.join();
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: ds41_sig_flags_test <engine/runtime/g6m4.cu>\n"); return 2; }
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
    CHECK(f.good() || f.eof());
    CHECK(ss.str().find(SIG_LINES) != std::string::npos);   // the stubbed expression is g6m4.cu's
    CHECK(ss.str().find("bc.runtime_signature += \" tkmain \" + std::to_string(ds_moe_main_topk)") != std::string::npos);
    CHECK(sig(0, 0) == "ds41 sessions 8 cap 131072 set s shtc 0");
    CHECK(sig(1, 0) == "ds41 sessions 8 cap 131072 set s shtc 1");
    CHECK(sig(0, 1) == "ds41 sessions 8 cap 131072 set s shtc 0 l2h 1");
    CHECK(sig(1, 1) == "ds41 sessions 8 cap 131072 set s shtc 1 l2h 1");
    CHECK(sig(1, 2) == "ds41 sessions 8 cap 131072 set s shtc 1 l2h 2");
    CHECK(sig(1, 2, 1) == "ds41 sessions 8 cap 131072 set s shtc 1 l2h 2 stg 1");
    CHECK(sig(0, 0, 1) == "ds41 sessions 8 cap 131072 set s shtc 0 stg 1");
    CHECK(sig(0, 0, 0, 1) == "ds41 sessions 8 cap 131072 set s shtc 0 fc1bstg 1");
    CHECK(sig(1, 2, 1, 1) == "ds41 sessions 8 cap 131072 set s shtc 1 l2h 2 stg 1 fc1bstg 1");

    char tmpl[] = "/tmp/sigflagsXXXXXX"; const std::string dir = mkdtemp(tmpl);
    // Stay below Linux's ephemeral source-port range: prior loopback connections
    // can otherwise claim a later case's listen port before its bind.
    const uint16_t port = uint16_t(10000 + (getpid() * 7) % 10000);
    struct Case { const char* name; int tc_a, h_a, tc_b, h_b; bool want; int g_a = 0, g_b = 0, b_a = 0, b_b = 0; };
    const Case cases[] = {
        {"same shtc1 l2h1", 1, 1, 1, 1, true},  {"same shtc0 l2h0", 0, 0, 0, 0, true},
        {"shtc 1 vs 0", 1, 1, 0, 1, false},     {"l2h 1 vs 0", 1, 1, 1, 0, false},
        {"l2h 1 vs 2", 0, 1, 0, 2, false},      {"both differ", 1, 1, 0, 0, false},
        {"same stg1", 1, 2, 1, 2, true, 1, 1},  {"stg 1 vs 0", 1, 2, 1, 2, false, 1, 0},
        {"same fc1bstg1", 1, 2, 1, 2, true, 0, 0, 1, 1},
        {"fc1bstg 1 vs 0", 1, 2, 1, 2, false, 0, 0, 1, 0},
    };
    unsigned i = 0;
    for (const auto& c : cases) {
        std::string why;
        const std::string a = sig(c.tc_a, c.h_a, c.g_a, c.b_a), b = sig(c.tc_b, c.h_b, c.g_b, c.b_b);
        const bool ok = boot(uint16_t(port + i), dir + "/c" + std::to_string(i), a, b, why); ++i;
        CHECK(ok == c.want);
        if (!c.want) CHECK(why.find("bootstrap config mismatch") != std::string::npos);
        std::printf("ds41_sig_flags_test: %-16s '%s' vs '%s': %s%s\n", c.name, a.c_str() + 27, b.c_str() + 27,
                    ok ? "booted" : "refused: ", ok ? "" : why.c_str());
    }
    {
        std::string why;
        const bool ok = boot(uint16_t(port + i), dir + "/tkdecode", sig(0, 0) + " tkmain 4 tkdecode 2",
                             sig(0, 0) + " tkmain 4 tkdecode 3", why);
        CHECK(!ok && why.find("bootstrap config mismatch") != std::string::npos);
    }
    if (!fails) std::printf("ds41_sig_flags_test: PASS\n");
    return fails ? 1 : 0;
}
