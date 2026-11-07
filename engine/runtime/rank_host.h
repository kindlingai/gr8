// The host-only pieces of rank.cuh and packfind.h that g6m4's option parser (engine_options.h) reads, so the parser
// builds without CUDA: die(), the M4 test sequence's Plan, the --v2 sources and parse_ints.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../model_geom.h"
#include "steering.h"

namespace g6rt
{

[[noreturn]] inline void die(const std::string& m) { fprintf(stderr, "g6: %s\n", m.c_str()); exit(2); }

// ---------------------------------------------------------------- the M4 test sequence, shared by both drivers
// The prompt goes in first, `chunk` rows per step at consecutive positions (the last prompt step may be shorter);
// then n_gen single-row steps, each feeding the token sampled from the previous step's last row.
struct Plan
{
    std::vector<int> prompt;
    int first = 0;                    // prompt tokens before this index were taken by the prefill path
    int chunk = 4;                    // rows per prompt step, 1..MX
    int n_gen = 16;
    float inv_temp = 0.f;             // 0 = greedy
    uint32_t seed = 1234u;
    float top_p = 0.f;                // (0,1) = nucleus sampling (needs Model::nc > 0); else off
    int lp = -1, plp = -1;            // --serve: top-k logprobs of the generated / the prompt tokens (k > 0 = top lists; 0 = only
                                      // the token's own logprob; -1 = off). Rows of a step with k > 0 carry SampleRow::topk.
    bool rs = false;                  // --spec-accept rejection: drafts are greedy, the target accepts them by rejection sampling
    float steering_scale = 0.f;       // target model only; exact canonical bits from the effective request identity
    int steering_slot = 0;            // steering.slot
    steering::Identity steering{};    // full cache identity (profile/scope/schedule/exact scale)
};
inline int prompt_steps(const Plan& p) { return ((int)p.prompt.size() - p.first + p.chunk - 1) / p.chunk; }
inline int total_steps(const Plan& p) { return prompt_steps(p) + p.n_gen; }
inline int step_rows(const Plan& p, int s)
{
    const int left = (int)p.prompt.size() - p.first - s * p.chunk;
    return s >= prompt_steps(p) ? 1 : left < p.chunk ? left : p.chunk;
}
// the distinct row counts of the plan (each needs prepare() and, in the graph form, its own graph)
inline std::vector<int> plan_rows(const Plan& p)
{
    std::vector<int> v;
    for (int s = 0; s < total_steps(p); ++s)
    {
        const int M = step_rows(p, s);
        bool seen = false;
        for (int x : v) seen |= x == M;
        if (!seen) v.push_back(M);
    }
    return v;
}
// exlr8 v2 experts from the raw width files (--v2 DIR --v2-slices DIR --v2-widths FILE). With dense (--v2-dense DIR),
// every other tensor comes from the release's dense and vocab files too, with no base pack. trellis keeps the trellis
// matrices as the release's entries (engine/kernels/dtr.cuh, --v2-dense-form trellis); false rounds them to a
// converted pack's MXFP8 / int4 forms. qo adds the GR8_PF_OWN=3 full-head tensors to the image. layout (--layout FILE)
// says which pieces each rank holds (engine/loader-rs/src/layout.rs); empty: the formulas.
struct V2Src
{
    std::string dir, slices, widths, dense, layout;
    bool trellis = true, qo = false;
};

// comma-separated integers; "@file" reads them (comma or white space separated) from a file; "all" = 0..77
inline std::vector<int> parse_ints(std::string s)
{
    std::vector<int> v;
    if (s == "all") { for (int i = 0; i < geom::L_MAIN; ++i) v.push_back(i); return v; }
    if (!s.empty() && s[0] == '@')
    {
        FILE* f = fopen(s.c_str() + 1, "r");
        if (!f) die("cannot read " + s.substr(1));
        int x;
        while (fscanf(f, "%d%*[, \n\t]", &x) == 1) v.push_back(x);
        fclose(f);
        return v;
    }
    for (size_t p = 0; p < s.size();)
    {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        v.push_back(atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

}   // namespace g6rt
