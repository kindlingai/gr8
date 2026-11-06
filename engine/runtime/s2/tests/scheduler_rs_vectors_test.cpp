// engine/scheduler-rs through its C++ headers against transcripts recorded from the C++ it replaced: each section of
// scheduler_rs_vectors.h runs a seeded command stream and must hash to the value the C++ gave (recorded with
// libstdc++, where the request parser's number rules come from). The scheduler's streams depend on the module's shape
// (slots and caps), so it has one value per shape. A changed rule changes a hash: rerecord only for a change meant to
// alter behaviour, and say so in the commit. The first eight were recorded at R2d (kindlingai/gr8#353) from the C++ then
// in s2/tests/ref, the driver's from the C++ driver, driver_pool's from the service's pool hooks; --dump SECTION prints
// a section's transcript, to diff two builds line by line. Host only.
#include "../kv_plan.h"
#include "../flash_lend.h"
#include "../flash_ring.h"
#include "../cp_rows.h"
#include "../scheduler.h"
#include "../kv_share.h"
#include "../sessions.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
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

// --dump SECTION: write that section's transcript to stdout instead of checking the hashes (to diff two builds)
int main(int argc, char** argv)
{
    struct Shape { int max_sessions; bool mtp0, pack, park; uint64_t scheduler, driver, driver_pool; };
    const Shape shapes[] = {
        {8, false, false, false, 0x516723cb09cd4014ull, 0x3508037ae19f733cull, 0xaebb24ec50c72c50ull},   // GLM-5.3 and Flash
        {32, true, true, true, 0x6ab1879c4a43bb14ull, 0xaa629dc3f6a951dbull, 0x75108cfa718bede4ull},     // DS41
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
        {"request", 8, [](uint64_t s, std::vector<std::string>* l) { return vec::request(s, l, false); }, 0xaaab3710d1030662ull},
        // recorded from the Rust after the old C++ parser (git c74d024, libstdc++) gave the same transcript
        {"request_gram", 9, [](uint64_t s, std::vector<std::string>* l) { return vec::request(s, l, true); }, 0x18a523b6c0f716ceull},
        {"share_resize", 10, vec::share_resize, 0xd99c642674175c07ull},
        // recorded from the C++ driver it replaced (git bd1bb6a), in each shape
        {"driver", 11, vec::driver, shape ? shape->driver : 0},
        // recorded from the service's C++ pool hooks before the driver made these calls (git 7440bda), in each shape
        {"driver_pool", 12, vec::driver_pool, shape ? shape->driver_pool : 0},
    };
    if (argc == 3 && std::string(argv[1]) == "--dump")
    {
        for (const auto& s : sections)
            if (s.name == std::string(argv[2]))
            {
                std::vector<std::string> lines;
                s.run(s.seed, &lines);
                for (const auto& l : lines) std::printf("%s\n", l.c_str());
                return 0;
            }
        std::fprintf(stderr, "no section %s\n", argv[2]);
        return 2;
    }
    int fails = 0;
    for (const auto& s : sections)
    {
        const uint64_t h = s.run(s.seed, nullptr);
        if (h == s.want) continue;
        ++fails;
        std::printf("FAIL %s: transcript %016llx, recorded %016llx%s\n", s.name, (unsigned long long)h, (unsigned long long)s.want,
                    s.want ? "" : " (no value recorded)");
    }
    std::printf("scheduler_rs_vectors_test (%d slots%s%s%s): %s\n", ms, mtp0 ? ", mtp0" : "", pack ? ", pack" : "", park ? ", park" : "",
                fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
