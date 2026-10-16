// engine_opts.h: the flags g6m4 and m4sim both take. Until these moved into one place the two mains held the same 29
// parse lines twice, and nothing checked that they agreed.
//   g++ -std=c++17 -O2 -Wall -I engine/runtime engine/runtime/tests/engine_opts_test.cpp && ./a.out
#include "engine_opts.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace g6rt;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// argv as the mains hand it over: `next` yields the following entry, or throws where they would die
struct Args
{
    std::vector<std::string> v;
    size_t i = 0;
    std::string next() { if (i + 1 >= v.size()) throw std::runtime_error("missing value"); return v[++i]; }
};

/// Parse a whole command line of shared flags. Returns the arguments common_arg did not take.
static std::vector<std::string> parse(const std::vector<std::string>& argv, CommonOpts& o, std::string& err)
{
    Args a{argv};
    std::vector<std::string> rest;
    for (a.i = 0; a.i < a.v.size(); ++a.i)
    {
        std::string e;
        if (common_arg(a.v[a.i], [&] { return a.next(); }, o, e))
        {
            if (!e.empty() && err.empty()) err = e;
        }
        else rest.push_back(a.v[a.i]);
    }
    return rest;
}

static void defaults_are_the_mains()
{
    CommonOpts o;
    CHECK(o.layers_s == "0,3,40,77");            // both mains declared this one
    CHECK(o.cap == 64 && o.lanes == 1 && o.mtp == 0 && o.nc == 0);
    CHECK(o.MP == 0 && o.top == 0 && o.pfrows == 0 && o.temp == 0.f);
    CHECK(!o.graph && !o.wide && !o.pfv2 && !o.adapt && o.v2_trellis);
    CHECK(o.prompts.empty() && o.levels_s.empty());
    // the Plan-bound flags say nothing until the command line does, so Plan keeps its defaults
    CHECK(!o.n_gen && !o.chunk && !o.seed && !o.top_p && !o.rs);
    CHECK(common_check(o, 32, 8).empty());
}

static void every_shared_flag_lands()
{
    CommonOpts o;
    std::string err;
    const auto rest = parse({"--v2", "/w", "--v2-slices", "/s", "--v2-widths", "/m.json", "--v2-dense", "/r",
                             "--v2-dense-form", "rounded", "--qo-manifest", "/q.json", "--qo-pack", "/q.pack",
                             "--spec-adaptive", "--spec-levels", "2,3,5", "--layers", "0,3", "--prompt", "1,2",
                             "--prompt", "3,4", "--gen", "7", "--temp", "0.7", "--seed", "99", "--cap", "4096",
                             "--chunk", "8", "--graph", "--prefill", "4096", "--top", "5", "--wide", "--pfv2",
                             "--pf-rows", "-2", "--mtp", "3", "--pf-lanes", "2", "--top-p", "0.9",
                             "--top-p-cands", "256", "--spec-accept", "rejection"}, o, err);
    CHECK(err.empty());
    CHECK(rest.empty());                          // all 29 belong to the shared set
    CHECK(o.v2_dir == "/w" && o.v2_slices == "/s" && o.v2_widths == "/m.json" && o.v2_dense == "/r");
    CHECK(!o.v2_trellis);                         // "rounded"
    CHECK(o.qo_man == "/q.json" && o.qo_file == "/q.pack");
    CHECK(o.adapt && o.levels_s == "2,3,5" && o.layers_s == "0,3");
    CHECK(o.prompts.size() == 2 && o.prompts[0] == "1,2" && o.prompts[1] == "3,4");
    CHECK(o.cap == 4096 && o.MP == 4096 && o.top == 5 && o.pfrows == -2 && o.mtp == 3 && o.lanes == 2 && o.nc == 256);
    CHECK(o.graph && o.wide && o.pfv2);
    CHECK(o.temp == 0.7f);
    CHECK(o.n_gen && *o.n_gen == 7);
    CHECK(o.seed && *o.seed == 99u);
    CHECK(o.chunk && *o.chunk == 8);
    CHECK(o.top_p && *o.top_p == 0.9f);
    CHECK(o.rs && *o.rs);                         // "rejection"
    CHECK(common_check(o, 32, 8).empty());
}

static void trellis_and_coupled_are_the_other_choices()
{
    CommonOpts o;
    std::string err;
    parse({"--v2-dense-form", "trellis", "--spec-accept", "coupled"}, o, err);
    CHECK(err.empty() && o.v2_trellis);
    CHECK(o.rs && !*o.rs);
}

static void a_flag_this_main_does_not_know_is_left_alone()
{
    CommonOpts o;
    std::string err;
    // g6m4's own flags, and m4sim's: common_arg must not claim them, or one main would swallow the other's
    const auto rest = parse({"--rank", "--ips", "--serve", "--packs", "--s2", "--mtp-dump", "--set-id", "--zzz"}, o, err);
    CHECK(err.empty());
    CHECK(rest.size() == 8);
}

static void a_bad_value_names_itself_and_is_still_consumed()
{
    for (const auto& c : std::vector<std::pair<std::vector<std::string>, const char*>>{
             {{"--v2-dense-form", "halfway"}, "--v2-dense-form"},
             {{"--spec-accept", "maybe"}, "--spec-accept"}})
    {
        CommonOpts o;
        std::string err;
        const auto rest = parse(c.first, o, err);
        CHECK(!err.empty());
        CHECK(err.find(c.second) != std::string::npos);
        CHECK(rest.empty());                      // consumed, so the caller dies on err rather than "unknown argument"
    }
}

static void a_missing_value_is_the_callers_problem()
{
    CommonOpts o;
    std::string err;
    bool threw = false;
    try { parse({"--layers"}, o, err); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);                                 // `next` dies in the mains; here it throws
}

static void the_bounds_are_the_ones_the_mains_checked()
{
    CommonOpts o;
    CHECK(common_check(o, 32, 0) == "--chunk must be 1..32");       // the default is checked too, not only a given one
    CHECK(common_check(o, 32, 33) == "--chunk must be 1..32");
    CHECK(common_check(o, 32, 32).empty());
    CHECK(common_check(o, 8, 9) == "--chunk must be 1..8");         // mx follows the build
    o.mtp = 8;
    CHECK(common_check(o, 32, 8) == "--mtp must be 1..7 drafts");   // 7 drafts whatever MX is
    o.mtp = 7;
    CHECK(common_check(o, 32, 8).empty());
    o.mtp = -1;
    CHECK(common_check(o, 32, 8) == "--mtp must be 1..7 drafts");
    o.mtp = 4;
    CHECK(common_check(o, 4, 4) == "--mtp must be 1..7 drafts");    // and at most MX - 1
}

static void inv_temp_is_the_reciprocal_or_zero()
{
    CHECK(inv_temp(0.f) == 0.f);
    CHECK(inv_temp(-1.f) == 0.f);
    CHECK(inv_temp(0.5f) == 2.f);
    CHECK(strlen(DEFAULT_PROMPT) > 0);
}

int main()
{
    defaults_are_the_mains();
    every_shared_flag_lands();
    trellis_and_coupled_are_the_other_choices();
    a_flag_this_main_does_not_know_is_left_alone();
    a_bad_value_names_itself_and_is_still_consumed();
    a_missing_value_is_the_callers_problem();
    the_bounds_are_the_ones_the_mains_checked();
    inv_temp_is_the_reciprocal_or_zero();
    if (fails) { fprintf(stderr, "engine_opts_test: %d FAILED\n", fails); return 1; }
    printf("engine_opts_test: ok (29 shared flags, bounds, and the two bad values)\n");
    return 0;
}
