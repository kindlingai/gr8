// trace.h: the events of a GR8_S2_TRACE file, which tools/trace/merge.py then closes and parses (scripts/host_tests.sh).
//   trace_test PATH    writes PATH.r3.json
#include "../trace.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char** argv)
{
    assert(argc == 2);
    {
        g6s2::Trace off;
        off.open("", 0);
        assert(!off);
        g6s2::Trace t;
        t.open(argv[1], 3);
        assert(t);
        const auto t0 = g6s2::Trace::now();
        t.span("prefill", g6s2::Trace::STEPS, t0, t0 + 1500, g6s2::targ("rows", 8192) + ",\"requests\":[7]");
        t.instant("arrive", g6s2::Trace::CONTROL, g6s2::targ("request", 7) + "," + g6s2::targ("why", "slots"));
    }
    std::ifstream f(std::string(argv[1]) + ".r3.json");
    std::stringstream s;
    s << f.rdbuf();
    const std::string text = s.str();
    assert(text.rfind("[\n", 0) == 0 && text.size() > 2 && text.substr(text.size() - 2) == ",\n");
    assert(text.find("\"name\":\"prefill\",\"ph\":\"X\",\"pid\":3,\"tid\":0") != std::string::npos);
    assert(text.find("\"dur\":1500,\"args\":{\"rows\":8192,\"requests\":[7]}") != std::string::npos);
    assert(text.find("\"name\":\"rank 3\"") != std::string::npos);
    printf("trace_test: PASS\n");
}
