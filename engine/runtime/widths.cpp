// The main of g6m4 and of m4sim (-DM4SIM): one binary for every TP width. build_m4.sh compiles the engine once
// per width in WIDTHS into an instance whose only global symbol is its main, renamed g6w<N>_main; a width the build left
// out is a null weak symbol here. g6m4 runs the instance of its --ips count. m4sim runs the one --tp N names, else the
// one of its --packs DIR (rank0.manifest.json, rank1.. ). A binary with one width runs it, as the one-width builds did.
#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef M4SIM
#include <sys/stat.h>
#include <vector>
#endif

extern "C" {
int g6w1_main(int, char**) __attribute__((weak));
int g6w2_main(int, char**) __attribute__((weak));
int g6w3_main(int, char**) __attribute__((weak));
int g6w4_main(int, char**) __attribute__((weak));
int g6w6_main(int, char**) __attribute__((weak));
int g6w8_main(int, char**) __attribute__((weak));
}

using Main = int (*)(int, char**);
static const struct { int nr; Main main; } WIDTHS[] = {
    {1, g6w1_main}, {2, g6w2_main}, {3, g6w3_main}, {4, g6w4_main}, {6, g6w6_main}, {8, g6w8_main}};

[[noreturn]] static void die(const std::string& m) { fprintf(stderr, "g6: %s\n", m.c_str()); exit(2); }

#ifndef M4SIM
static int count(const std::string& s)   // the entries g6m4.cu's split(s, ',') returns
{
    int n = 0;
    size_t a = 0, b;
    while ((b = s.find(',', a)) != std::string::npos) { ++n; a = b + 1; }
    return n + (a < s.size() ? 1 : 0);
}
#endif

int main(int argc, char** argv)
{
    std::string built;
    Main only = nullptr;
    int n = 0;
    for (const auto& w : WIDTHS)
        if (w.main) { built += (n++ ? ", " : "") + std::to_string(w.nr); only = w.main; }
    int nr = 0;
#ifdef M4SIM
    // --tp N is this main's flag: it leaves argv before the instance parses the rest
    std::vector<char*> av;
    std::string packs;
    bool tp = false;
    for (int i = 0; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (i && a == "--tp")
        {
            if (i + 1 >= argc || (nr = atoi(argv[++i])) < 1) die("--tp needs the width (this binary: TP = " + built + ")");
            tp = true;
            continue;
        }
        if (a == "--packs" && i + 1 < argc) packs = argv[i + 1];
        av.push_back(argv[i]);
    }
    av.push_back(nullptr);
    argc = (int)av.size() - 1;
    argv = av.data();
    if (n == 1 && !tp) return only(argc, argv);
    struct stat sb;
    if (!tp && !packs.empty())
        while (stat((packs + "/rank" + std::to_string(nr) + ".manifest.json").c_str(), &sb) == 0) ++nr;
    const char* how = "--tp N, or --packs DIR with its rank<r>.manifest.json files,";
#else
    if (n == 1) return only(argc, argv);
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--ips") nr = count(argv[i + 1]);
    const char* how = "--ips, one address per rank,";
#endif
    for (const auto& w : WIDTHS)
        if (w.nr == nr && w.main) return w.main(argc, argv);
    if (!n) die("this binary has no TP width (build_m4.sh WIDTHS)");
    die(std::string(how) + " names the TP width: " + (nr ? std::to_string(nr) : "none") + " here, and this binary has TP = " + built);
}
