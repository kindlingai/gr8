// CPU check of the KDA decode kernels' math (math.h phases through kda_cpu.h) against the W0 fp32 goldens, from
// identical inputs: q|k|v projections = the golden `mixed`, f_a/g_a/b projections = x @ W^T in double from the golden
// block input. Gates (ENGINE-PLAN.md): fp32 ops rel-L2 <= 1e-4; `out` (BF16 o_gated into o_proj) <= 1e-2.
//
//   g++ -O3 -march=native -fopenmp -std=c++17 -DG6_MODEL_FLASH=1 -o kda_cpu_test kda_cpu_test.cpp
//   kda_cpu_test --dense FLASH/full-dense/dense [--prompt golden/prompt-bf16.safetensors] [--step golden/step-bf16.safetensors]
//                [--long golden/kda-L00-4096.safetensors] [--layers 0,1,2]
//
//   prompt: 64 tokens from zero state, TP 1 and TP 4 head slices, M = 1 rows; M = 8 rows of one slot and 2 interleaved
//           slots must give the same bits as M = 1.
//   step:   one decode step from the golden state_in (position 64).
//   long:   4096 tokens from zero state; state after 1, 64, 4096 tokens and every op of every token.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "common.h"
#include "kda_cpu.h"

using namespace kda;
using namespace kda_test;

// one rank's KDA layer on the CPU model
struct Rank
{
    int h0 = 0, nh = 0, nhmax = 0, slots = 2;
    std::vector<float> S, cst, P, C, G, Z, O, OG;
    std::vector<float> QE, KD, PW, PU, GC, AQ;   // chunked prefill scratch
    std::vector<uint16_t> OF;
    std::vector<int> rs;
    const W* w = nullptr;
    bool chunked = false;
    void init(const W& w_, int h0_, int nh_, int nhmax_, int rows, bool chunked_ = false)
    {
        chunked = chunked_;
        w = &w_; h0 = h0_; nh = nh_; nhmax = nhmax_;
        S.assign(static_cast<size_t>(slots) * nhmax * DK * DV, 0.f);
        cst.assign(static_cast<size_t>(slots) * NHIST * 3 * nhmax * DK, 0.f);
        const int dl = nh * DK;
        PLayout pl; pl.dl = dl;
        P.assign(static_cast<size_t>(rows) * pl.ld(), 0.f);
        C.assign(static_cast<size_t>(rows) * 3 * dl, 0.f);
        G.assign(static_cast<size_t>(rows) * dl, 0.f); Z = G; O = G; OG = G;
        OF.assign(static_cast<size_t>((rows + 7) / 8) * dl * 8, 0);
        rs.assign(rows, 0);
        if (chunked)
        {
            QE.assign(static_cast<size_t>(rows) * dl, 0.f); KD = QE; PW = QE; PU = QE; GC = QE;
            AQ.assign(static_cast<size_t>(rows) * nh * CH, 0.f);
        }
    }
    // state from a golden [64][128][128] / [3][3 * 8192] into slot
    void set_state(int slot, const float* gS, const float* gconv)
    {
        for (int h = 0; h < nh; ++h)
            memcpy(&S[(static_cast<size_t>(slot) * nhmax + h) * DK * DV], gS + static_cast<size_t>(h0 + h) * DK * DV, DK * DV * 4);
        Args a; a.nhmax = nhmax;
        for (int j = 0; j < NHIST; ++j)
            for (int s = 0; s < 3; ++s)
                for (int c = 0; c < nh * DK; ++c) cst[a.c_off(slot, j, s, c)] = gconv[static_cast<size_t>(j) * 3 * D + s * D + h0 * DK + c];
    }
    // rows m: golden mixed row, small projections row, slot
    void run(const float* const* mixed, const float* const* sp, const int* slot, int M)
    {
        const int dl = nh * DK;
        PLayout pl; pl.dl = dl;
        for (int m = 0; m < M; ++m)
        {
            float* p = &P[static_cast<size_t>(m) * pl.ld()];
            for (int s = 0; s < 3; ++s)
                for (int c = 0; c < dl; ++c) p[s * dl + c] = mixed[m][s * D + h0 * DK + c];
            for (int r = 0; r < 320; ++r) p[pl.fa() + r] = sp[m][r];
            rs[m] = slot[m];
        }
        Args a;
        a.nh = nh; a.h0 = h0; a.M = M; a.nhmax = nhmax;
        for (int i = 0; i < 3; ++i) a.conv_w[i] = w->conv[i] + static_cast<size_t>(h0) * DK * CONV;
        a.fb = w->fb + static_cast<size_t>(h0) * DK * DK; a.gb = w->gb + static_cast<size_t>(h0) * DK * DK;
        a.A_log = w->A_log + h0; a.dt_bias = w->dt_bias + h0 * DK; a.onorm = w->onorm;
        a.S = S.data(); a.cst = cst.data(); a.row_slot = rs.data();
        a.P = P.data(); a.C = C.data(); a.G = G.data(); a.Z = Z.data(); a.O = O.data(); a.OG = OG.data(); a.OF = OF.data();
        kda_cpu::prep(a);
        if (chunked)
        {
            PfArgs p;
            p.a = a;
            p.QE = QE.data(); p.KD = KD.data(); p.W = PW.data(); p.U = PU.data(); p.GC = GC.data(); p.AQ = AQ.data();
            kda_cpu::prefill(p);
        }
        else kda_cpu::step(a);
        kda_cpu::out(a);
    }
    // BF16 o_gated row m (from the fragment table) @ own o_proj columns -> acc [HID] (double)
    void oproj(int m, std::vector<double>& acc) const
    {
        const int dl = nh * DK;
        std::vector<float> ob(dl);
        for (int c = 0; c < dl; ++c) ob[c] = bf2f(OF[frag_u16(m, c, dl)]);
        for (int r = 0; r < HID_; ++r)
        {
            const uint16_t* row = w->o + static_cast<size_t>(r) * D + h0 * DK;
            double d = 0;
            for (int c = 0; c < dl; ++c) d += static_cast<double>(bf2f(row[c])) * ob[c];
            acc[r] += static_cast<float>(d);   // each rank's partial is BF16-free fp32 here; the sum in rank order
        }
    }
};

struct Run   // T tokens of one golden sequence through nr ranks, rows per call M (one slot), outputs gathered
{
    std::vector<float> g, q, k, v, o, og, out;   // [T][...] full heads
    std::vector<int> out_rows;
};

// gather one rank's rows into the full-head outputs
static void gather(const Rank& R, Run& out, int t0, int M)
{
    const int dl = R.nh * DK;
    for (int m = 0; m < M; ++m)
    {
        const size_t t = static_cast<size_t>(t0 + m);
        for (int c = 0; c < dl; ++c)
        {
            const size_t gi = t * D + R.h0 * DK + c, li = static_cast<size_t>(m) * dl + c;
            out.g[gi] = R.G[li]; out.o[gi] = R.O[li]; out.og[gi] = R.OG[li];
            out.v[gi] = R.C[static_cast<size_t>(m) * 3 * dl + 2 * dl + c];
        }
        for (int h = 0; h < R.nh; ++h)
            for (int s = 0; s < 2; ++s)
            {
                const float* x = &R.C[static_cast<size_t>(m) * 3 * dl + s * dl + h * DK];
                double ss = 0;
                for (int i = 0; i < DK; ++i) ss += static_cast<double>(x[i]) * x[i];
                const double dn = std::sqrt(ss + L2_EPS);
                float* dst = (s ? out.k : out.q).data() + t * D + (R.h0 + h) * DK;
                for (int i = 0; i < DK; ++i) dst[i] = static_cast<float>(x[i] / dn * (s ? 1.0 : QSCALE));
            }
    }
}

// run tokens [0, T) of a sequence, M rows per call, ranks of TP nr; states before token 0 from st_in (or zero);
// snap(t, ranks) is called after t tokens for every t in marks; chunked: the prefill path instead of step, with
// call sizes `sizes` (repeated cyclically; empty = M)
template <class Snap>
static Run run_seq(const W& w, int l, int nr, int T, int M, const float* mixed, const std::vector<float>& sp, const float* S_in,
                   const float* conv_in, const std::vector<int>& out_rows, const std::vector<int>& marks, Snap snap,
                   bool chunked = false, std::vector<int> sizes = {})
{
    if (sizes.empty()) sizes = {M};
    for (int z : sizes) M = std::max(M, z);
    Run out;
    const size_t n = static_cast<size_t>(T) * D;
    out.g.assign(n, 0); out.q = out.g; out.k = out.g; out.v = out.g; out.o = out.g; out.og = out.g;
    out.out.assign(out_rows.size() * HID_, 0);
    out.out_rows = out_rows;
    std::vector<Rank> R(nr);
    const int nhm = (HEADS + nr - 1) / nr;
    for (int r = 0; r < nr; ++r)
    {
        int h0, nh;
        head_rows(l, nr, r, h0, nh);
        R[r].init(w, h0, nh, nhm, M, chunked);
        if (S_in) R[r].set_state(0, S_in, conv_in);
    }
    std::vector<const float*> mx(M), spp(M);
    std::vector<int> slot(M, 0);
    size_t oi = 0;
    for (int t0 = 0, call = 0; t0 < T; t0 += sizes[call++ % sizes.size()])
    {
        const int m = std::min(sizes[call % sizes.size()], T - t0);
        for (int i = 0; i < m; ++i) { mx[i] = mixed + static_cast<size_t>(t0 + i) * 3 * D; spp[i] = &sp[static_cast<size_t>(t0 + i) * 320]; }
#pragma omp parallel for
        for (int r = 0; r < nr; ++r) R[r].run(mx.data(), spp.data(), slot.data(), m);
        for (int r = 0; r < nr; ++r) gather(R[r], out, t0, m);
        for (int i = 0; i < m; ++i)
            if (oi < out_rows.size() && out_rows[oi] == t0 + i)
            {
                std::vector<double> acc(HID_, 0.0);
                for (int r = 0; r < nr; ++r) R[r].oproj(i, acc);
                for (int j = 0; j < HID_; ++j) out.out[oi * HID_ + j] = static_cast<float>(acc[j]);
                ++oi;
            }
        for (int mk : marks)
            if (mk > t0 && mk <= t0 + m)
            {
                if (mk != t0 + m) { fprintf(stderr, "mark %d inside a call of %d rows\n", mk, m); exit(2); }
                snap(mk, R);
            }
    }
    return out;
}

// full-head state of all ranks (slot 0) -> [64][128][128], [3][3 * 8192]
static void full_state(const std::vector<Rank>& R, std::vector<float>& S, std::vector<float>& conv)
{
    S.assign(static_cast<size_t>(HEADS) * DK * DV, 0);
    conv.assign(static_cast<size_t>(NHIST) * 3 * D, 0);
    for (const Rank& r : R)
    {
        for (int h = 0; h < r.nh; ++h)
            memcpy(&S[static_cast<size_t>(r.h0 + h) * DK * DV], &r.S[static_cast<size_t>(h) * DK * DV], DK * DV * 4);
        Args a; a.nhmax = r.nhmax;
        for (int j = 0; j < NHIST; ++j)
            for (int s = 0; s < 3; ++s)
                for (int c = 0; c < r.nh * DK; ++c) conv[static_cast<size_t>(j) * 3 * D + s * D + r.h0 * DK + c] = r.cst[a.c_off(0, j, s, c)];
    }
}

static void compare_ops(const st::File& g, const std::string& L, const Run& r, int t0, int T, const std::string& tag)
{
    const size_t off = static_cast<size_t>(t0) * D, n = static_cast<size_t>(T) * D;
    gate(tag + " g", rel(r.g.data(), g[L + "kda.g"].f32() + off, n), 1e-4);
    gate(tag + " q", rel(r.q.data(), g[L + "kda.q"].f32() + off, n), 1e-4);
    gate(tag + " k", rel(r.k.data(), g[L + "kda.k"].f32() + off, n), 1e-4);
    gate(tag + " v", rel(r.v.data(), g[L + "kda.v"].f32() + off, n), 1e-4);
    gate(tag + " o", rel(r.o.data(), g[L + "kda.o"].f32() + off, n), 1e-4);
    gate(tag + " o_gated", rel(r.og.data(), g[L + "kda.o_gated"].f32() + off, n), 1e-4);
    std::vector<float> ref(r.out_rows.size() * HID_);
    for (size_t i = 0; i < r.out_rows.size(); ++i)
        memcpy(&ref[i * HID_], g[L + "kda.out"].f32() + static_cast<size_t>(t0 + r.out_rows[i]) * HID_, HID_ * 4);
    gate(tag + " out (BF16 o_gated @ o_proj, " + std::to_string(r.out_rows.size()) + " rows)", rel(r.out.data(), ref.data(), ref.size()), 1e-2);
}

int main(int argc, char** argv)
{
    std::string dense, prompt, step, lng, layers = "0,1,2";
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string k = argv[i];
        if (k == "--dense") dense = argv[i + 1];
        else if (k == "--prompt") prompt = argv[i + 1];
        else if (k == "--step") step = argv[i + 1];
        else if (k == "--long") lng = argv[i + 1];
        else if (k == "--layers") layers = argv[i + 1];
    }
    if (dense.empty()) { fprintf(stderr, "usage: see header\n"); return 2; }
    std::vector<int> ls;
    for (const char* p = layers.c_str(); *p;) { ls.push_back(strtol(p, const_cast<char**>(&p), 10)); if (*p == ',') ++p; }

    for (int l : ls)
    {
        W w;
        w.load(dense, l);
        char Lb[8];
        snprintf(Lb, sizeof Lb, "L%02d.", l);
        const std::string L = Lb;
        if (!prompt.empty())
        {
            st::File g;
            g.open(prompt);
            const int T = static_cast<int>(g[L + "attn_in"].shape[0]);
            printf("prompt %s: %d tokens from zero state\n", L.c_str(), T);
            const auto sp = small_proj(w, g[L + "attn_in"].f32(), T);
            std::vector<int> rows;
            for (int t = 0; t < T; ++t) rows.push_back(t);
            std::vector<float> S, conv;
            Run r1 = run_seq(w, l, 1, T, 1, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, {T},
                             [&](int, const std::vector<Rank>& R) { full_state(R, S, conv); });
            compare_ops(g, L, r1, 0, T, "TP1 M1");
            gate("TP1 M1 S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
            Run r4 = run_seq(w, l, 4, T, 1, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, {T},
                             [&](int, const std::vector<Rank>& R) { full_state(R, S, conv); });
            compare_ops(g, L, r4, 0, T, "TP4 M1");
            gate("TP4 M1 S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
            same("TP4 o_gated == TP1 o_gated", r4.og, r1.og);
            Run r8 = run_seq(w, l, 4, T, 8, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, {T}, [](int, const std::vector<Rank>&) {});
            same("TP4 M8 (one slot) o_gated == M1", r8.og, r4.og);
            same("TP4 M8 (one slot) out == M1", r8.out, r4.out);
            // two sessions interleaved: rows [A t, B t]; B is the same sequence, both must equal M1
            {
                int h0, nh;
                head_rows(l, 4, 1, h0, nh);
                Rank R;
                R.init(w, h0, nh, 16, 2);
                std::vector<float> ogA(static_cast<size_t>(T) * nh * DK), ogB(ogA.size()), ref(ogA.size());
                for (int t = 0; t < T; ++t)
                {
                    const float* mx[2] = {g[L + "kda.mixed"].f32() + static_cast<size_t>(t) * 3 * D, g[L + "kda.mixed"].f32() + static_cast<size_t>(t) * 3 * D};
                    const float* spp[2] = {&sp[static_cast<size_t>(t) * 320], &sp[static_cast<size_t>(t) * 320]};
                    const int slot[2] = {0, 1};
                    R.run(mx, spp, slot, 2);
                    memcpy(&ogA[static_cast<size_t>(t) * nh * DK], &R.OG[0], nh * DK * 4);
                    memcpy(&ogB[static_cast<size_t>(t) * nh * DK], &R.OG[nh * DK], nh * DK * 4);
                    for (int c = 0; c < nh * DK; ++c) ref[static_cast<size_t>(t) * nh * DK + c] = r4.og[static_cast<size_t>(t) * D + h0 * DK + c];
                }
                same("2 slots interleaved: slot 0 == M1 (rank 1)", ogA, ref);
                same("2 slots interleaved: slot 1 == M1 (rank 1)", ogB, ref);
            }
            // chunked prefill: one call of T rows; calls of 5 and 59 rows (partial chunks, state carried)
            for (int nr : {1, 4})
            {
                Run rc = run_seq(w, l, nr, T, T, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, {T},
                                 [&](int, const std::vector<Rank>& R) { full_state(R, S, conv); }, true);
                const std::string tag = "chunk TP" + std::to_string(nr) + " M" + std::to_string(T);
                compare_ops(g, L, rc, 0, T, tag);
                gate(tag + " S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
                gate(tag + " o vs recurrent kernels", rel(rc.o.data(), r4.o.data(), rc.o.size()), 1e-4);
            }
            {
                Run rc = run_seq(w, l, 4, T, 64, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, {T},
                                 [&](int, const std::vector<Rank>& R) { full_state(R, S, conv); }, true, {5, 59});
                compare_ops(g, L, rc, 0, T, "chunk TP4 calls 5,59");
                gate("chunk TP4 calls 5,59 S after " + std::to_string(T), rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
            }
        }
        if (!step.empty())
        {
            st::File g;
            g.open(step);
            if (g.has(L + "state_in.S"))
            {
                printf("step %s: one token from the golden state_in\n", L.c_str());
                const auto sp = small_proj(w, g[L + "attn_in"].f32(), 1);
                for (int nr : {1, 4})
                {
                    std::vector<float> S, conv;
                    Run r = run_seq(w, l, nr, 1, 1, g[L + "kda.mixed"].f32(), sp, g[L + "state_in.S"].f32(), g[L + "state_in.conv"].f32(),
                                    {0}, {1}, [&](int, const std::vector<Rank>& R) { full_state(R, S, conv); });
                    const std::string tag = "TP" + std::to_string(nr) + " step";
                    compare_ops(g, L, r, 0, 1, tag);
                    gate(tag + " S", rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
                    Run rc = run_seq(w, l, nr, 1, 1, g[L + "kda.mixed"].f32(), sp, g[L + "state_in.S"].f32(), g[L + "state_in.conv"].f32(),
                                     {0}, {1}, [&](int, const std::vector<Rank>& R) { full_state(R, S, conv); }, true);
                    compare_ops(g, L, rc, 0, 1, "chunk " + tag);
                    gate("chunk " + tag + " S", rel(S.data(), g[L + "kda.S"].f32(), S.size()), 1e-4);
                }
            }
        }
        if (!lng.empty())
        {
            st::File g;
            g.open(lng);
            if (g.has(L + "attn_in"))
            {
                const int T = static_cast<int>(g[L + "attn_in"].shape[0]);
                printf("long %s: %d tokens from zero state (TP4, M1)\n", L.c_str(), T);
                const auto sp = small_proj(w, g[L + "attn_in"].f32(), T);
                std::vector<int> rows = {0, 1, 62, 63, 64};
                for (int t = 256; t < T; t += 256) rows.push_back(t - 1);
                if (rows.back() != T - 1) rows.push_back(T - 1);
                std::vector<int> marks = {1, 63, 64, T - 1, T};
                std::vector<std::pair<int, std::pair<std::vector<float>, std::vector<float>>>> snaps;
                Run r = run_seq(w, l, 4, T, 1, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, marks,
                                [&](int t, const std::vector<Rank>& R) {
                                    std::vector<float> S, conv;
                                    full_state(R, S, conv);
                                    snaps.push_back({t, {S, conv}});
                                });
                compare_ops(g, L, r, 0, T, "long TP4");
                for (auto& s : snaps)
                {
                    const std::string t = std::to_string(s.first);
                    gate("long S after " + t, rel(s.second.first.data(), g[L + "S." + t].f32(), s.second.first.size()), 1e-4);
                    gate("long conv after " + t, rel(s.second.second.data(), g[L + "conv." + t].f32(), s.second.second.size()), 1e-4);
                }
                for (int t : {1, 64, T})   // per-window rel of o, so late tokens are not hidden by the early ones
                {
                    const int a = t == 1 ? 0 : t - 64 < 0 ? 0 : t - 64;
                    const size_t off = static_cast<size_t>(a) * D, n = static_cast<size_t>(t - a) * D;
                    gate("long o, tokens " + std::to_string(a) + ".." + std::to_string(t - 1), rel(r.o.data() + off, g[L + "kda.o"].f32() + off, n), 1e-4);
                }
                // chunked prefill, calls of 1, 62, 1, T - 65, 1 rows (state marks at the call ends)
                printf("long %s chunked: calls 1, 62, 1, %d, 1 (TP4)\n", L.c_str(), T - 65);
                snaps.clear();
                Run rc = run_seq(w, l, 4, T, 1, g[L + "kda.mixed"].f32(), sp, nullptr, nullptr, rows, marks,
                                 [&](int t, const std::vector<Rank>& R) {
                                     std::vector<float> S, conv;
                                     full_state(R, S, conv);
                                     snaps.push_back({t, {S, conv}});
                                 }, true, {1, 62, 1, T - 65, 1});
                compare_ops(g, L, rc, 0, T, "long chunk TP4");
                for (auto& s : snaps)
                {
                    const std::string t = std::to_string(s.first);
                    gate("long chunk S after " + t, rel(s.second.first.data(), g[L + "S." + t].f32(), s.second.first.size()), 1e-4);
                    gate("long chunk conv after " + t, rel(s.second.second.data(), g[L + "conv." + t].f32(), s.second.second.size()), 1e-4);
                }
                for (int t : {64, 2048, T})
                {
                    const size_t off = static_cast<size_t>(t - 64) * D, n = static_cast<size_t>(64) * D;
                    gate("long chunk o, tokens " + std::to_string(t - 64) + ".." + std::to_string(t - 1), rel(rc.o.data() + off, g[L + "kda.o"].f32() + off, n), 1e-4);
                }
                gate("long chunk o == recurrent kernels o", rel(rc.o.data(), r.o.data(), rc.o.size()), 1e-4);
                gate("long chunk o vs ref kda_chunk o", rel(rc.o.data(), g[L + "chunk.o"].f32(), rc.o.size()), 1e-4);
                gate("long chunk S vs ref kda_chunk S", rel(snaps.back().second.first.data(), g[L + "chunk.S"].f32(), snaps.back().second.first.size()), 1e-4);
            }
        }
    }
    printf("%s: %d gate failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
