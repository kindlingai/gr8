#pragma once
// The command line g6m4 and m4sim both take. The real engine and the reference driver must agree on what every shared
// flag means, because m4sim's output is the oracle g6m4 is held to (g6m4.cu: "all six (NR) must equal m4sim's on the
// same packs"), and until now they agreed by holding the same 29 parse lines twice.
//
// Host only, no CUDA, so tests/engine_opts_test.cpp builds it with g++ (scripts/host_tests.sh). Two consequences:
// the flags that land in rank.cuh's Plan are optionals here, so Plan keeps its own defaults and a caller applies only
// what the command line set; and a bad value comes back as a message instead of calling die(), which lives in rank.cuh.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace g6rt
{

/// The prompt both mains run when none is given.
inline const char* const DEFAULT_PROMPT = "154822,154824,154826,25062";

/// The shared flags. Every default here is the one both mains declared.
struct CommonOpts
{
    std::string v2_dir, v2_slices, v2_widths, v2_dense;   // --v2, --v2-slices, --v2-widths, --v2-dense
    bool v2_trellis = true;                               // --v2-dense-form
    std::string qo_man, qo_file;                          // --qo-manifest, --qo-pack
    std::string layers_s = "0,3,40,77";                   // --layers: the mini model both mains default to
    std::string levels_s;                                 // --spec-levels
    std::vector<std::string> prompts;                     // --prompt, repeatable
    int cap = 64, MP = 0, top = 0, pfrows = 0, mtp = 0, lanes = 1, nc = 0;
    float temp = 0.f;
    bool graph = false, wide = false, pfv2 = false, adapt = false;
    // these land in Plan, which owns their defaults: unset means the command line did not name them
    std::optional<int> n_gen, chunk;
    std::optional<uint32_t> seed;
    std::optional<float> top_p;
    std::optional<bool> rs;                               // --spec-accept rejection
};

/// One argument. True when `a` is a shared flag, `next` giving its value as each main's own lambda does (which dies
/// when a value is missing). A bad value leaves `err` set and the caller dies with it; `a` is still consumed.
template <class Next>
bool common_arg(const std::string& a, Next&& next, CommonOpts& o, std::string& err)
{
    if (a == "--v2") o.v2_dir = next();
    else if (a == "--v2-slices") o.v2_slices = next();
    else if (a == "--v2-widths") o.v2_widths = next();
    else if (a == "--v2-dense") o.v2_dense = next();
    else if (a == "--v2-dense-form")
    {
        const std::string f = next();
        if (f != "trellis" && f != "rounded") err = "--v2-dense-form: trellis or rounded";
        else o.v2_trellis = f == "trellis";
    }
    else if (a == "--qo-manifest") o.qo_man = next();
    else if (a == "--qo-pack") o.qo_file = next();
    else if (a == "--spec-adaptive") o.adapt = true;
    else if (a == "--spec-levels") o.levels_s = next();
    else if (a == "--layers") o.layers_s = next();
    else if (a == "--prompt") o.prompts.push_back(next());
    else if (a == "--gen") o.n_gen = atoi(next().c_str());
    else if (a == "--temp") o.temp = (float)atof(next().c_str());
    else if (a == "--seed") o.seed = (uint32_t)strtoul(next().c_str(), nullptr, 10);
    else if (a == "--cap") o.cap = atoi(next().c_str());
    else if (a == "--chunk") o.chunk = atoi(next().c_str());
    else if (a == "--graph") o.graph = true;
    else if (a == "--prefill") o.MP = atoi(next().c_str());
    else if (a == "--top") o.top = atoi(next().c_str());
    else if (a == "--wide") o.wide = true;
    else if (a == "--pfv2") o.pfv2 = true;
    else if (a == "--pf-rows") o.pfrows = atoi(next().c_str());
    else if (a == "--mtp") o.mtp = atoi(next().c_str());
    else if (a == "--pf-lanes") o.lanes = atoi(next().c_str());
    else if (a == "--top-p") o.top_p = (float)atof(next().c_str());
    else if (a == "--top-p-cands") o.nc = atoi(next().c_str());
    else if (a == "--spec-accept")
    {
        const std::string v = next();
        if (v == "coupled") o.rs = false;
        else if (v == "rejection") o.rs = true;
        else err = "--spec-accept must be coupled or rejection";
    }
    else return false;
    return true;
}

/// The bounds both mains check once the options are in. `mx` is rank.cuh's MX and `chunk` the value Plan ended up
/// with, so the default is checked as well as a given one. Empty return: the options are good.
inline std::string common_check(const CommonOpts& o, int mx, int chunk)
{
    if (chunk < 1 || chunk > mx) return "--chunk must be 1.." + std::to_string(mx);
    if (o.mtp < 0 || o.mtp > std::min<int>(mx - 1, 7)) return "--mtp must be 1..7 drafts";
    return {};
}

/// Plan::inv_temp from --temp.
inline float inv_temp(float temp) { return temp > 0.f ? 1.f / temp : 0.f; }

}   // namespace g6rt
