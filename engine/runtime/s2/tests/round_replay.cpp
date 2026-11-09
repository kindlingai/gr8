// round_replay [--steps-only] [--dump-learn LINE] [--time] LOG [NGRAM_TABLE]: rerun the planner from a round log
// (GR8_S2_REPLAY_LOG) and check that this build asks for every step the logged planner did and, unless --steps-only,
// keeps the same learned state (s2/round_replay.h). --dump-learn: write out the replaying planner's learned state at
// that learn record's line. --time: the planner's own time per call (hook answers excluded). Build it with the logged engine's model flag (none, -DG6_MODEL_FLASH=1 or
// -DG6_MODEL_DS41=1); a Flash log also needs GR8_GLM53F_S2_COST as the logging engine had it. Exit 0 when every call
// and step matches.
#include "s2/round_replay.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>

int main(int argc, char** argv)
{
    bool steps_only = false, time = false;
    size_t dump = 0;
    while (argc > 1 && argv[1][0] == '-' && argv[1][1] == '-')
    {
        const std::string o = argv[1];
        if (o == "--steps-only") steps_only = true;
        else if (o == "--time") time = true;
        else if (o == "--dump-learn" && argc > 2) { dump = std::stoul(argv[2]); --argc; ++argv; }
        else { std::fprintf(stderr, "round_replay: unknown option %s\n", o.c_str()); return 2; }
        --argc;
        ++argv;
    }
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: round_replay [--steps-only] [--dump-learn LINE] [--time] LOG [NGRAM_TABLE]\n"); return 2; }
    std::FILE* f = std::fopen(argv[1], "r");
    if (!f) { std::fprintf(stderr, "round_replay: cannot read %s\n", argv[1]); return 2; }
    const std::vector<std::string> lines = s2::read_lines(f);
    std::fclose(f);
    g6rt::NgramTable table;
    if (argc == 3)
    {
        std::ifstream t(argv[2]);
        const std::string text((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        std::string err;
        if (!t || !table.parse(text, &err)) { std::fprintf(stderr, "round_replay: %s: %s\n", argv[2], err.c_str()); return 2; }
    }
    const s2::ReplayResult r = s2::replay(lines, argc == 3 ? &table : nullptr, !steps_only, dump, time);
    for (auto [tag, v] : r.time_us)
    {
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (double x : v) sum += x;
        std::printf("time %-18s %6zu calls  mean %8.2f us  p50 %8.2f  p99 %8.2f  max %8.2f\n", tag.c_str(), v.size(),
                    sum / v.size(), v[v.size() / 2], v[std::min(v.size() - 1, v.size() * 99 / 100)], v.back());
    }
    if (dump) std::printf("%s\n", r.learn_text.empty() ? "round_replay: no learn record at that line was reached" : r.learn_text.c_str());
    if (!r.ok)
    {
        std::printf("round_replay: FAIL at line %zu of %zu: %s\n", r.line, lines.size(), r.error.c_str());
        return 1;
    }
    std::printf("round_replay: PASS: %ld calls, %ld steps, %ld hook answers (log v%d%s%s%s, replayed on %s)\n", r.calls, r.steps,
                r.hooks, r.version, steps_only ? ", steps only" : "", r.libc.empty() ? "" : ", logged on ", r.libc.c_str(), s2::libc_version().c_str());
    return 0;
}
