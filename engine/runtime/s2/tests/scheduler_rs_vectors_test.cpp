// engine/scheduler-rs through its C++ headers against transcripts recorded from the C++ it replaced: each section of
// scheduler_rs_vectors.h runs a seeded command stream and must hash to the value the C++ gave (recorded with
// libstdc++, where the request parser's number rules come from). The scheduler's streams depend on the module's shape
// (slots and caps), so it has one value per shape. A changed rule changes a hash: rerecord only for a change meant to
// alter behaviour, and say so in the commit. Host only.
#include "../kv_plan.h"
#include "../flash_lend.h"
#include "../flash_ring.h"
#include "../cp_rows.h"
#include "../scheduler.h"
#include "../kv_share.h"
#include "../sessions.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>

namespace vec {
namespace impl {
using namespace g6s2;
inline void validate(const Config& c) { c.validate(); }
} // namespace impl
#include "scheduler_rs_vectors.h"
} // namespace vec

int main()
{
    struct Shape { int max_sessions; bool mtp0, pack, park; uint64_t scheduler; };
    const Shape shapes[] = {
        {8, false, false, false, 0x516723cb09cd4014ull},   // GLM-5.3 and Flash
        {32, true, true, true, 0x6ab1879c4a43bb14ull},     // DS41
    };
    const int ms = int(g6mod::INFO.max_sessions);
    const bool mtp0 = g6mod::has(G6_CAP_MTP0), pack = g6mod::has(G6_CAP_PACK), park = g6mod::has(G6_CAP_PARK);
    const Shape* shape = nullptr;
    for (const auto& s : shapes)
        if (s.max_sessions == ms && s.mtp0 == mtp0 && s.pack == pack && s.park == park) shape = &s;
    struct Section { const char* name; uint64_t seed; uint64_t (*run)(uint64_t, std::vector<std::string>*); uint64_t want; };
    const Section sections[] = {
        {"kv_plan", 1, vec::kv_plan, 0x1284d2c7525aca22ull},
        {"flash_lend", 2, vec::flash_lend, 0xcd000bb5bc3e58b8ull},
        {"flash_ring", 3, vec::flash_ring, 0x2f02e4d6c3bd7a5full},
        {"cp_rows", 4, vec::cp_rows, 0xfe9fb2d531cb05a9ull},
        {"scheduler", 5, vec::scheduler, shape ? shape->scheduler : 0},
        {"share", 6, vec::share, 0x5a98bacc9f262ab7ull},
        {"queue", 7, vec::queue, 0xaf92603441c49eaaull},
        {"request", 8, vec::request, 0xaaab3710d1030662ull},
    };
    int fails = 0;
    for (const auto& s : sections)
    {
        const uint64_t h = s.run(s.seed, nullptr);
        if (h == s.want) continue;
        ++fails;
        std::printf("FAIL %s: transcript %016llx, recorded %016llx%s\n", s.name, (unsigned long long)h, (unsigned long long)s.want,
                    s.want ? "" : " (no value recorded for this shape)");
    }
    std::printf("scheduler_rs_vectors_test (%d slots%s%s%s): %s\n", ms, mtp0 ? ", mtp0" : "", pack ? ", pack" : "", park ? ", park" : "",
                fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
