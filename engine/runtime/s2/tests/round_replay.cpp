// round_replay [--steps-only] LOG [NGRAM_TABLE]: rerun the planner from a round log (GR8_S2_REPLAY_LOG) and check that
// this build asks for every step the logged planner did and, unless --steps-only, keeps the same learned state
// (s2/round_replay.h). Build it with the logged engine's model flag (none,
// -DG6_MODEL_FLASH=1 or -DG6_MODEL_DS41=1); a Flash log also needs GR8_GLM53F_S2_COST as the logging engine had it.
// Exit 0 when every call and step matches.
#include "s2/round_replay.h"
#include <cstdio>
#include <fstream>
#include <iterator>

int main(int argc, char** argv)
{
    const bool steps_only = argc > 1 && std::string(argv[1]) == "--steps-only";
    if (steps_only) { --argc; ++argv; }
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: round_replay [--steps-only] LOG [NGRAM_TABLE]\n"); return 2; }
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
    const s2::ReplayResult r = s2::replay(lines, argc == 3 ? &table : nullptr, !steps_only);
    if (!r.ok)
    {
        std::printf("round_replay: FAIL at line %zu of %zu: %s\n", r.line, lines.size(), r.error.c_str());
        return 1;
    }
    std::printf("round_replay: PASS: %ld calls, %ld steps, %ld hook answers (log v%d%s%s%s, replayed on %s)\n", r.calls, r.steps,
                r.hooks, r.version, steps_only ? ", steps only" : "", r.libc.empty() ? "" : ", logged on ", r.libc.c_str(), s2::libc_version().c_str());
    return 0;
}
