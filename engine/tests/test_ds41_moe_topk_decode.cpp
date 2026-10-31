#include "ds41_moe_topk.h"
#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)
static int parse(const char* value, int main_cut, bool good) {
    if (value) setenv("G6_DS_MOE_TOPK_DECODE", value, 1);
    else unsetenv("G6_DS_MOE_TOPK_DECODE");
    std::string err;
    const int got = ds41_moe_topk::decode_env(main_cut, 6, &err);
    CHECK((got > 0) == good);
    if (!good) CHECK(err.find("G6_DS_MOE_TOPK_DECODE") != std::string::npos);
    return got;
}
int main() {
    CHECK(parse(nullptr, 6, true) == 6);
    CHECK(parse(nullptr, 4, true) == 4);
    for (int k = 1; k <= 6; ++k) {
        const std::string v = std::to_string(k);
        CHECK(parse(v.c_str(), 4, true) == k);
        CHECK(ds41_moe_topk::for_step(false, 4, k) == 4);
        CHECK(ds41_moe_topk::for_step(true, 4, k) == k);
        CHECK(ds41_moe_topk::graph_kind(false, 4, k) == 0);
        CHECK(ds41_moe_topk::graph_kind(true, 4, k) == (k != 4));
    }
    for (const char* bad : {"0", "7", "", "01", "-1", "1x", " 1", "1 "})
        CHECK(parse(bad, 4, false) == -1);
    unsetenv("G6_DS_MOE_TOPK_DECODE");
    if (fails) return 1;
    std::puts("DS41_MOE_TOPK_DECODE_HOST_PASS");
}
