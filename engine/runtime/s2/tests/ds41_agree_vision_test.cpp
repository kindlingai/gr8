// DS41 ranks that differ in the vision tower they loaded refuse at bootstrap (vision_id.h; g6m4.cu passes vis_on and
// vis_id to the DS41 runtime signature, s2/signature.h). Four ranks on loopback, each its own process: every rank
// with the same tower file boots, every rank without one boots (text-only), a rank without the file among ranks with it
// refuses, and a rank with another tower file refuses. Host only. Model: ds41_agree_knobs_test.cpp.
// Run: ds41_agree_vision_test engine/runtime/g6m4.cu
#include "s2/bootstrap.h"
#include "vision_id.h"
#include "s2/signature.h"
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

static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }

// a safetensors file: the 8-byte header length, the header, `body` payload bytes
static void tower(const std::string& path, const std::string& header, size_t body) {
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = header.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f << header << std::string(body, '\0');
}

// rank `r` of 4 with the tower file `file` ("" = no file); exits 0 when it boots, 3 when it refuses (the head writes `why`)
[[noreturn]] static void rank(unsigned r, const std::string& file, uint16_t port, const std::string& dir, const std::string& why) {
    if (r) (void)!std::freopen("/dev/null", "w", stderr);
    const std::string id = file.empty() ? "" : g6vis::file_id(file);
    if (!file.empty() && id.empty()) std::_Exit(2);
    BootstrapConfig c; c.rank = r; c.ranks = 4; c.timeout = 3; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(r); ::mkdir(c.directory.c_str(), 0700);
    SigInputs s;
    s.model = SigModel::Ds41; s.s2sess = 32; s.cap = 327680; s.layers_s = "all"; s.set_id = "ab12";
    s.vis_on = !file.empty(); s.vis_id = id;
    c.runtime_signature = runtime_signature(s);
    try { auto links = bootstrap(c); std::_Exit(0); }
    catch (const std::exception& e) {
        if (r == 0) { std::ofstream(why) << e.what(); }
        std::_Exit(3);
    }
}

static int boot(const std::vector<std::string>& files, uint16_t port, const std::string& dir, std::string& why) {
    ::mkdir(dir.c_str(), 0700);
    const std::string wf = dir + "/why";
    std::vector<pid_t> ps;
    for (unsigned r = 0; r < files.size(); ++r) {
        const pid_t p = ::fork();
        if (p == 0) rank(r, files[r], port, dir, wf);
        ps.push_back(p);
    }
    int head = -1;
    for (size_t r = 0; r < ps.size(); ++r) {
        int st = 0; ::waitpid(ps[r], &st, 0);
        if (r == 0) head = WIFEXITED(st) ? WEXITSTATUS(st) : 128;
    }
    why = slurp(wf);
    return head;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s engine/runtime/g6m4.cu\n", argv[0]); return 2; }
    // g6m4.cu: the effective vision state and the file id go into the DS41 signature (not the knob)
    const std::string src = slurp(argv[1]);
    const size_t a = src.find("sig.model = g6s2::SigModel::Ds41;");
    const size_t b = a == std::string::npos ? a : src.find("g6s2::serve_ds41(dc, bc", a);
    CHECK(a != std::string::npos && b != std::string::npos);
    if (a != std::string::npos && b != std::string::npos)
        CHECK(src.substr(a, b - a).find("sig.vis_on = vis_on; sig.vis_id = vis_id;") != std::string::npos);
    CHECK(src.find("vis_id = g6vis::file_id(vpath);") != std::string::npos);

    char tmpl[] = "/tmp/ds41agreevisXXXXXX"; const std::string dir = mkdtemp(tmpl);
    const std::string t1 = dir + "/t1.safetensors", t1b = dir + "/t1-copy.safetensors", t2 = dir + "/t2.safetensors";
    const std::string h1 = "{\"vision.blocks.0.norm1.weight\":{\"dtype\":\"BF16\",\"shape\":[1024],\"data_offsets\":[0,2048]}}";
    const std::string h2 = "{\"vision.blocks.0.norm1.weight\":{\"dtype\":\"BF16\",\"shape\":[1024],\"data_offsets\":[0,2049]}}";
    tower(t1, h1, 2048); tower(t1b, h1, 2048); tower(t2, h2, 2049);
    CHECK(g6vis::file_id(t1) == g6vis::file_id(t1b));            // same bytes on another box: the same id
    CHECK(!g6vis::file_id(t1).empty() && g6vis::file_id(t1) != g6vis::file_id(t2));
    CHECK(g6vis::file_id(dir + "/none").empty());
    CHECK(g6vis::signature(false, "x").empty());

    const uint16_t port = uint16_t(21000 + getpid() % 20000);
    std::string why;
    CHECK(boot({t1, t1b, t1, t1b}, port, dir + "/same", why) == 0);
    CHECK(boot({"", "", "", ""}, uint16_t(port + 1), dir + "/none", why) == 0);
    CHECK(boot({t1, t1, "", t1}, uint16_t(port + 2), dir + "/missing", why) == 3);
    CHECK(why.find("bootstrap config mismatch: rank 2") != std::string::npos);
    std::printf("ds41_agree_vision_test: rank 2 without the tower file refused: %s\n", why.c_str());
    CHECK(boot({t1, t1, t1, t2}, uint16_t(port + 3), dir + "/other", why) == 3);
    CHECK(why.find("bootstrap config mismatch: rank 3") != std::string::npos);
    std::printf("ds41_agree_vision_test: rank 3 with another tower file refused: %s\n", why.c_str());
    for (const auto& f : {t1, t1b, t2}) std::remove(f.c_str());

    if (!fails) std::printf("ds41_agree_vision_test: PASS\n");
    return fails ? 1 : 0;
}
