// CPU model of the KDA decode kernels (kda.cuh): the same phase functions (math.h), run CTA by CTA, phase by phase,
// thread by thread, with the barriers where the kernels have them.
#pragma once
#include <vector>
#include "../math.h"
#include "../prefill.h"

namespace kda_cpu
{
using namespace kda;

inline void prep(const Args& a)
{
    std::vector<PrepThread> th(PREP_T);
    PrepSmem* s = new PrepSmem;
    for (int h = 0; h < a.nh; ++h)
        for (int hf = 0; hf < 2; ++hf)
        {
            for (auto& x : th) x = PrepThread();
            for (int t = 0; t < PREP_T; ++t) prep_stage(a, *s, h, hf, t);
            for (int m = 0; m < a.M; ++m)
            {
                for (int t = 0; t < PREP_T; ++t) prep_row_a(a, *s, th[t], m, h, hf, t);
                for (int t = 0; t < PREP_T; ++t) prep_row_b(a, *s, m, h, hf, t);
            }
            for (int t = 0; t < PREP_T; ++t) prep_end(a, th[t], h, hf, t);
        }
    delete s;
}

inline void step(const Args& a)
{
    std::vector<StepThread> th(STEP_T);
    std::vector<float> o(STEP_T);
    StepSmem s;
    for (int h = 0; h < a.nh; ++h)
        for (int vb = 0; vb < DV / VB; ++vb)
        {
            for (auto& x : th) x = StepThread();
            for (int m = 0; m < a.M; ++m)
            {
                for (int t = 0; t < STEP_T; ++t) step_row_0(a, s, th[t], m, h, vb, t);
                for (int t = 0; t < STEP_T; ++t) step_row_1(a, s, m, h, t);
                for (int t = 0; t < STEP_T; ++t) step_row_2(s, th[t], t);
                for (int t = 0; t < STEP_T; ++t) o[t] = step_row_3(a, s, th[t], m, h, vb, t);
                for (int t = 0; t < STEP_T; ++t) step_row_3b(s, t, o[t]);
                for (int t = 0; t < STEP_T; ++t) step_row_4(a, s, m, h, vb, t);
            }
            for (int t = 0; t < STEP_T; ++t) step_end(a, th[t], h, vb, t);
        }
}

inline void out(const Args& a)
{
    OutSmem s;
    for (int h = 0; h < a.nh; ++h)
        for (int m = 0; m < a.M; ++m)
        {
            for (int t = 0; t < OUT_T; ++t) out_0(a, s, m, h, t);
            for (int t = 0; t < OUT_T; ++t) out_1(s, t);
            for (int t = 0; t < OUT_T; ++t) out_2(a, s, m, h, t);
        }
}
// chunked prefill (prefill.h): intra per (head, chunk), then state per (head, v block); replaces step()
inline void prefill(const PfArgs& p)
{
    IntraSmem* s = new IntraSmem;
    for (int h = 0; h < p.a.nh; ++h)
        for (int c = 0; c < p.nchunk(); ++c)
        {
            for (int t = 0; t < INTRA_T; ++t) intra_0(p, *s, h, c, t);
            for (int t = 0; t < INTRA_T; ++t) intra_1(p, *s, h, c, t);
            for (int t = 0; t < INTRA_T; ++t) intra_2(p, *s, h, c, t);
            for (int t = 0; t < INTRA_T; ++t) intra_3(*s, t);
            for (int t = 0; t < INTRA_T; ++t) intra_4(p, *s, h, c, t);
            for (int t = 0; t < INTRA_T; ++t) intra_5(p, h, c, t);
        }
    delete s;
    StSmem* ss = new StSmem;
    std::vector<float> o(static_cast<size_t>(ST_T) * ST_R);
    for (int h = 0; h < p.a.nh; ++h)
        for (int vb = 0; vb < DV / VB; ++vb)
        {
            for (int t = 0; t < ST_T; ++t) st_load(p, *ss, h, vb, t);
            for (int c = 0; c < p.nchunk(); ++c)
            {
                for (int t = 0; t < ST_T; ++t) st_a(p, *ss, h, vb, c, t, &o[static_cast<size_t>(t) * ST_R]);
                for (int t = 0; t < ST_T; ++t) st_b(p, *ss, h, vb, c, t, &o[static_cast<size_t>(t) * ST_R]);
            }
            for (int t = 0; t < ST_T; ++t) st_store(p, *ss, h, vb, t);
        }
    delete ss;
}
}   // namespace kda_cpu
