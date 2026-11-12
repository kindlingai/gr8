// g6m4's option parse (engine_options.h, the facade over scheduler-rs options.rs) over the cases scripts/check_args.py
// writes (options_cases.txt): the launcher's argument lists per model and variant, and one mutation per refusal (a
// MODEL@NR case runs only at that width). The knob table is read once a process, so each case runs in a child with its
// environment alone, as g6m4's main runs it: knob::init, then the parse. The child's exit code, its stderr and either
// the refusal or describe() make the case's transcript; each shape's hash must equal the recorded one. Rerecord only
// for a change meant to alter a check, and say so in the commit. scripts/check_options_coverage.py reads the --dump of
// every shape and checks each refusal is reached.
//   options_vectors_test CASES [--dump]   --dump prints every case's transcript
// Host only, Linux only (clearenv; scripts/host_tests.sh runs it there).
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include "../../engine_options.h"
#include "../flash_ring.h"

namespace {
struct Case { std::string model, name; std::vector<std::string> env, args; };

std::vector<std::string> words(const std::string& s)
{
    std::vector<std::string> v;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(' ', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return v;
}

std::vector<Case> read_cases(const char* path, const char* model)
{
    std::vector<Case> v;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1), t3 = line.find('\t', t2 + 1);
        if (t3 == std::string::npos) { fprintf(stderr, "bad case line: %s\n", line.c_str()); exit(2); }
        Case c{line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1), words(line.substr(t2 + 1, t3 - t2 - 1)),
               words(line.substr(t3 + 1))};
        if (c.model == model || c.model == std::string(model) + "@" + std::to_string(g6nr::NR)) v.push_back(c);
    }
    return v;
}

std::string addresses(const char* net)
{
    std::string s;
    for (int i = 0; i < g6nr::NR; ++i) s += (i ? "," : "") + std::string(net) + std::to_string(i + 1);
    return s;
}

// the child: exits with main's code, the transcript on fd 1
[[noreturn]] void child(const Case& c)
{
    ::clearenv();
    for (const std::string& kv : c.env) {
        const size_t eq = kv.find('=');
        setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
    }
    std::vector<std::string> args{"g6m4", "--check-args"};
    for (const std::string& a : c.args)
        args.push_back(a == "@IPS" ? addresses("10.0.0.") : a == "@IPS2" ? addresses("10.1.0.") : a == "@EMPTY" ? "" : a);
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    if (knob::init()) _exit(2);
    g6rt::EngineOptions o;
    const std::string e = g6rt::parse_engine_options((int)args.size(), argv.data(), o);
    if (!e.empty()) { fprintf(stderr, "g6: %s\n", e.c_str()); _exit(2); }
    fputs(g6rt::describe(o).c_str(), stdout);
    fflush(stdout);
    _exit(0);
}

std::string run(const Case& c)
{
    int fd[2];
    if (pipe(fd)) { perror("pipe"); exit(2); }
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(2); }
    if (pid == 0) {
        close(fd[0]);
        dup2(fd[1], 1);
        dup2(fd[1], 2);
        close(fd[1]);
        child(c);
    }
    close(fd[1]);
    std::string out;
    char b[65536];
    for (ssize_t n; (n = read(fd[0], b, sizeof b)) > 0;) out.append(b, size_t(n));
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    const std::string rc = WIFEXITED(st) ? std::to_string(WEXITSTATUS(st)) : "signal " + std::to_string(WTERMSIG(st));
    return "== " + c.name + "\nrc " + rc + "\n" + out;
}
}   // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s options_cases.txt [--dump]\n", argv[0]); return 2; }
    const bool dump = argc > 2 && !strcmp(argv[2], "--dump");
#if G6_MODEL_FLASH
    const char* model = "glm53f";
#elif G6_MODEL_DS41
    const char* model = "ds41";
#else
    const char* model = "glm53";
#endif
#if G6_KV4 && !G6_MODEL_FLASH && !G6_MODEL_DS41
    const char* model_tag = "glm53-kv4";
#else
    const char* model_tag = model;
#endif
    const std::string shape = std::string(model_tag) + " nr" + std::to_string(g6nr::NR) + " mx" + std::to_string(g6mx::MX);
    struct Want { const char* shape; size_t n; uint64_t hash; };
    const Want wants[] = {
        {"glm53 nr6 mx8", 122, 0xe80ac959f044daf8ull},
        {"glm53 nr4 mx32", 122, 0x90c53302b108f89cull},
        {"glm53-kv4 nr4 mx32", 122, 0xeb128b40cbc92c4full},
        {"glm53f nr4 mx32", 116, 0x2920b0d6c121e568ull},
        {"ds41 nr4 mx48", 116, 0x07ec2d75cdd0d055ull},
    };
    const std::vector<Case> cs = read_cases(argv[1], model);
    uint64_t h = 14695981039346656037ull;
    for (const Case& c : cs) {
        const std::string t = run(c);
        for (unsigned char ch : t) h = (h ^ ch) * 1099511628211ull;
        if (dump) fputs(t.c_str(), stdout);
    }
    // Waterfill probes are checked independently of the launcher-default transcript goldens.
    bool wf_ok = true;
    auto probe = [&](const char* name, const std::vector<std::string>& env,
                     const std::vector<std::string>& args, const char* expected) {
        const std::string t = run(Case{model, name, env, args});
        if (dump) fputs(t.c_str(), stdout);
        if (t.find(expected) == std::string::npos) { fprintf(stderr, "waterfill probe failed: %s", t.c_str()); wf_ok = false; }
    };
    std::vector<std::string> plain{"--rank", "0", "--ips", "@IPS", "--manifest", "m", "--pack", "p"};
#if G6_KV4
    // Every INT4-build probe must pass the format check before reaching its intended guard.
    plain.insert(plain.end(), {"--kv-format", "int4"});
#endif
    probe("waterfill-margin", {"GR8_S2_SHARED_WF_MARGIN=-1"}, plain,
          "g6: GR8_S2_SHARED_WF_MARGIN: want a finite nonnegative value");
#if G6_MODEL_DS41
    probe("waterfill-uniform-refusal", {"GR8_S2_SHARED_WATERFILL=1"}, plain,
          "g6: GR8_S2_SHARED_WATERFILL: DS41 shared steps require uniform rows; ragged rows are unsupported");
#else
    probe("waterfill-serve-refusal", {"GR8_S2_SHARED_WATERFILL=1"}, plain,
          "g6: GR8_S2_SHARED_WATERFILL needs --serve, --mtp >= 1 and max rows <= 64");
    if (g6mx::MX >= 32) {
        std::vector<std::string> args = plain;
        for (const char* v : {"--serve", "session", "--graph", "--mtp", "3", "--prefill", "64",
                              "--s2-sessions", "4", "--s2-port", "1"}) args.push_back(v);
        probe("waterfill-boot", {"GR8_S2_SHARED_WATERFILL=1"}, args, "rc 0\n");
        probe("waterfill-ngram-boot", {"GR8_S2_SHARED_WATERFILL=1", "GR8_NGRAM=table"}, args, "rc 0\n");
    }
#endif
    for (int n : {2, 4, 8}) for (bool ng : {false, true}) {
        const auto f = g6s2::flash_shapes(n, 3, 4, false, 32, true, ng);
        for (int m = 1; m <= std::min(32, n * (ng ? 8 : 4)); ++m)
            wf_ok = wf_ok && f.mains.count(m) && f.drafts.count(m);
        for (int m = 1; m <= n; ++m) wf_ok = wf_ok && f.chains.count(m);
        // N-gram verify rows fit independently in every Flash session's rollback ring.
        g6s2::FlashSlots slots(n, ng ? 8 : 4);
        for (int q = 0; q < n; ++q) {
            int out[8]{};
            wf_ok = wf_ok && slots.rows(q, 0, ng ? 8 : 4, true, out) == slots.base(q);
        }
    }
    const Want* w = nullptr;
    for (const Want& x : wants) if (shape == x.shape) w = &x;
    const bool ok = wf_ok && w && w->n == cs.size() && w->hash == h;
    printf("options %s: %zu cases, hash %016" PRIx64 "%s\n", shape.c_str(), cs.size(), h,
           ok ? "" : w ? " (MISMATCH)" : " (no recorded hash for this shape)");
    return ok ? 0 : 1;
}
