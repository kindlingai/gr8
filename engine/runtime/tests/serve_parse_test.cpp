// serve.h's request parser (the --serve request file, frontend/g6shim.py writes it), host only.
//   g++ -std=c++17 -O1 -fsanitize=address,undefined -I engine/runtime engine/runtime/tests/serve_parse_test.cpp -o /tmp/serve_parse_test
#include "serve.h"

#include <cstdio>
#include <string>

using namespace g6rt;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr int V = 154880;

static ServeReq parse(const std::string& text)
{
    ServeReq r;
    serve_parse(text, V, r);
    return r;
}

static bool refused(const std::string& text) { return parse(text).n_gen == 0; }

int main()
{
    // a request as g6shim.py writes it
    ServeReq r = parse("n_gen 16 temp 0.7 seed 1235 min 2 stop 154820,154827 top_p 0.9 lp 5\n5,6,154879\n");
    CHECK(r.n_gen == 16 && r.min_tokens == 2 && r.seed == 1235u && r.temp > 0.69f && r.top_p > 0.89f && r.lp == 5);
    CHECK(r.plp == -1 && r.stop == (std::vector<int>{154820, 154827}) && r.prompt == (std::vector<int>{5, 6, 154879}));
    r = parse("n_gen 4 temp 0 seed 1 min 0 stop -\n7\n");            // no stop ids, no optional keys (older shims)
    CHECK(r.n_gen == 4 && r.stop.empty() && r.top_p == 0.f && r.prompt == std::vector<int>{7});
    CHECK(parse("n_gen 4 temp 0 seed 1 min 0 stop - lp 99\n7\n").lp == LP_MAX);   // clamped, as before

    // a stop field longer than the old 65535-char buffer: read whole, and nothing of it reaches the prompt
    std::string stops;
    while (stops.size() < 70000) stops += "154879,";
    stops.pop_back();
    r = parse("n_gen 4 temp 0 seed 1 min 0 stop " + stops + " top_p 1\n5,6\n");
    CHECK(r.n_gen == 4 && r.prompt == (std::vector<int>{5, 6}) && r.stop.size() == (stops.size() + 1) / 7);
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop " + stops + "9\n5,6\n"));   // its last id out of range

    // ids outside the vocabulary, negative ids and malformed numbers are refused
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop 154880\n5\n"));
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop -1\n5\n"));
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop -\n5,154880\n"));
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop -\n5,-3\n"));
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop -\n5,99999999999999999999\n"));
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop -\n5,,6\n"));
    CHECK(refused("n_gen 4x temp 0 seed 1 min 0 stop -\n5\n"));
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0 stop -\n\n"));          // no prompt
    CHECK(refused("n_gen 4 temp 0 seed 1 min 0\n5\n"));               // no stop field
    CHECK(refused(""));

    // only line 2 is the prompt
    CHECK(parse("n_gen 4 temp 0 seed 1 min 0 stop -\n5\n6,7\n").prompt == std::vector<int>{5});

    std::printf(fails ? "serve_parse_test: %d FAIL\n" : "serve_parse_test: PASS\n", fails);
    return fails != 0;
}
