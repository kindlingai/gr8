// GR8_DS_SESS_DRAFTS (s2/ds41_sess_drafts.h, Batch::set_sess_drafts) on the real s2::Batch with a fake engine, DS41 shape (K = 5 DSpark
// drafts, 32 slots), at each row cap the DS41 build takes: G6_MX = 32 (today) and DS41_MX = 40 / 48 (build_m4.sh, opt-in):
//   (a) the rule: kv(n) for n = 1..32 active sessions against a literal table per row cap, and one table line per n: rows a session, main
//       rows, draft launches of at most chunk sessions (sess_drafts_chunk: 6 at 32 rows, 8 at 40 / 48), and the same n with the knob off
//   (b) every decode-round MAIN step of n sessions gives each session kv(n) + 1 rows and fits the row cap; every DRAFT step goes through
//       g6s2::draft_launches (the host part of Ds41Round::draft_step_sessions, per-slot Ds41RoundLogic) with the engine's cap: launches
//       of at most chunk sessions (5 draft rows each, <= the row cap), covering the drafted sessions once, in step order, each input from its own rows
//   (c) knob on while every session keeps K drafts (1..5 at 32 rows, 1..6 at 40, 1..8 at 48): the same steps (kind, rows, every row's
//       session / token / position) and the same tokens as knob off
//   (d) every committed token is the target's, knob on, at every n
// Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I engine/runtime engine/runtime/s2/tests/batch_ds41_sessdrafts_test.cpp <knobs lib>
#include "s2_batch.h"
#include "s2/ds41_dspark_round_logic.h"
#include <cstdio>
#include <string>
#include <tuple>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 30) std::printf("FAIL %s:%d %s  [%s]\n", __FILE__, __LINE__, #c, ctx.c_str()); ++fails; } } while (0)
static std::string ctx;

constexpr int K = 5, BLOCK = 5, MAXS = 8;
static int target(int q, int pos) { return int((uint32_t(pos) * 37u + uint32_t(q) * 7919u + 11u) % 997u); }   // token at pos + 1 of session q
static bool good(int q, int spos) { return ((uint32_t(spos) * 2654435761u + uint32_t(q) * 40503u) >> 7) % 4 != 0; }

struct Fake
{
    std::vector<std::tuple<int, int, std::vector<int>>> trace;   // (kind, round, rows as sess / tok / pos triples)
    std::vector<Step> mains, drafts;                              // decode-round main steps; every draft step
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            std::vector<int> out, rows;
            for (const Row& r : s.r)
            {
                const int q = r.sess;
                out.push_back(s.kind == Step::MAIN || good(q, r.spos) ? target(q, r.spos - 1) : -2);
                rows.insert(rows.end(), {q, r.tok, r.pos});
            }
            trace.emplace_back(int(s.kind), int(s.round), rows);
            if (s.kind == Step::MAIN && s.round >= 0) mains.push_back(s);
            if (s.kind == Step::DRAFT) drafts.push_back(s);
            return out;
        };
        h.emit = [](int, const int*, int, int) { return false; };
        return h;
    }
};

struct Run { Fake f; std::vector<std::vector<int>> out; };
static Run run(int n, bool on, int MXR)
{
    Run r;
    Batch b(K, MXR, r.f.hooks());
    b.set_mtp_sessions(std::min(MXR / BLOCK, 8));   // g6m4.cu's Config::mtp_sessions for MXR rows
    b.set_sess_drafts(on);
    std::vector<int> qs;
    std::vector<Session> ss;
    for (int q = 0; q < n; ++q)
    {
        Session s;
        for (int i = 0; i < 9 + q % 3; ++i) s.prompt.push_back(i == 0 ? 1 : target(q, i - 1));
        s.chunk = 4;
        s.n_gen = 48;
        qs.push_back(q); ss.push_back(s);
    }
    b.admit(qs, ss);
    while (b.round(qs)) {}
    for (int q = 0; q < n; ++q) r.out.push_back(b.result(q));
    return r;
}

// the checks at row cap MXR: want[n] = the drafts a session at n active (n = 1..32), the most drafting sessions of 32 slots, the draft chunk
static void check_mx(int MXR, const int (&want)[33], int max32, int chunk)
{
    // (a) the rule
    for (int n = 1; n <= 32; ++n)
    {
        ctx = "MX " + std::to_string(MXR) + " kv n " + std::to_string(n);
        CHECK(sess_drafts_kv(K, MXR, n) == want[n]);
        CHECK(n * (want[n] + 1) <= MXR || want[n] == 0);
    }
    ctx = "MX " + std::to_string(MXR) + " max"; CHECK(sess_drafts_max(K, MXR, 32) == max32); CHECK(sess_drafts_max(K, MXR, 8) == 8);
    ctx = "MX " + std::to_string(MXR) + " chunk"; CHECK(sess_drafts_chunk(MXR, BLOCK, MAXS) == chunk); CHECK(chunk * BLOCK <= MXR);
    for (int S = 1; S <= 32; ++S)
    {
        ctx = "MX " + std::to_string(MXR) + " chunks S " + std::to_string(S);
        const auto c = draft_chunks(S, chunk);
        CHECK((int)c.size() == (S + chunk - 1) / chunk);
        int next = 0;
        for (const auto& p : c) { CHECK(p.first == next && p.second >= 1 && p.second <= chunk); next += p.second; }
        CHECK(next == S);
    }

    std::printf("MX %d: active  rows/session  main rows  draft launches | knob off: rows/session\n", MXR);
    for (int n : {1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20, 24})
    {
        const Run on = run(n, true, MXR), off = run(n, false, MXR);
        // (b) every decode-round main step: kv(its sessions) + 1 rows a session, at most MXR rows; the full-n rounds for the table
        int full_rps = -1, full_off = -1, full_rounds = 0;
        for (const Step& s : on.f.mains)
        {
            const int ns = (int)s.sess.size(), kv = sess_drafts_kv(K, MXR, ns);
            ctx = "MX " + std::to_string(MXR) + " on n " + std::to_string(n) + " ns " + std::to_string(ns);
            CHECK(s.M() <= MXR);
            for (size_t j = 0; j < s.sess.size(); ++j) CHECK(s.rows[j] == kv + 1);
            if (ns > 1) for (size_t j = 0; j < s.sess.size(); ++j) CHECK(s.rows[j] <= index_sess_rows(true, K));
            if (ns == n) { ++full_rounds; if (full_rps < 0) full_rps = s.rows[0]; CHECK(s.rows[0] == full_rps); }
        }
        for (const Step& s : off.f.mains) if ((int)s.sess.size() == n) { full_off = s.rows[0]; break; }
        for (const Step& s : off.f.mains)
            if (s.sess.size() > 1)
                for (size_t j = 0; j < s.sess.size(); ++j) { ctx = "MX " + std::to_string(MXR) + " off n " + std::to_string(n); CHECK(s.rows[j] <= index_sess_rows(true, K)); }
        ctx = "MX " + std::to_string(MXR) + " on n " + std::to_string(n);
        CHECK(full_rounds > 0);
        CHECK(full_rps == want[n] + 1);
        size_t full_launches = 0;
        std::vector<g6s2::Ds41RoundLogic> logic(32, g6s2::Ds41RoundLogic(K));
        for (const Step& d : on.f.drafts)
        {
            CHECK(d.M() <= MXR);
            const auto ls = g6s2::draft_launches(d, logic, chunk);
            std::vector<size_t> seen;
            for (const auto& L : ls)
            {
                CHECK(!L.js.empty() && (int)L.js.size() <= chunk && L.js.size() == L.ins.size() && (int)L.js.size() * BLOCK <= MXR);
                for (size_t i = 0; i < L.js.size(); ++i)
                {
                    const size_t j = L.js[i];
                    seen.push_back(j);
                    const Row& last = d.r[size_t(d.row0[j] + d.rows[j] - 1)];
                    CHECK(L.ins[i].q == last.pos && L.ins[i].golden == last.tok && L.ins[i].acc == d.rows[j] - 1);
                }
            }
            if (d.round >= 0) { CHECK(seen.size() == d.sess.size()); for (size_t j = 0; j < seen.size(); ++j) CHECK(seen[j] == j); }
            CHECK(ls.size() == (seen.size() + chunk - 1) / chunk);
            if ((int)d.sess.size() == n && d.round >= 0) full_launches = std::max(full_launches, ls.size());
        }
        CHECK(full_launches == (want[n] ? size_t((n + chunk - 1) / chunk) : 0));
        // (c) every session keeps K drafts: the same steps and tokens as knob off
        if (want[n] == K)
        {
            ctx = "MX " + std::to_string(MXR) + " same n " + std::to_string(n);
            CHECK(on.f.trace == off.f.trace); CHECK(on.out == off.out); CHECK(full_off == full_rps);
        }
        // (d) every committed token is the target's
        for (int q = 0; q < n; ++q)
        {
            ctx = "MX " + std::to_string(MXR) + " tokens n " + std::to_string(n) + " q " + std::to_string(q);
            const int plen = 9 + q % 3;
            CHECK((int)on.out[size_t(q)].size() == 48);
            for (size_t i = 0; i < on.out[size_t(q)].size(); ++i) CHECK(on.out[size_t(q)][i] == target(q, plen - 1 + int(i)));
            CHECK(on.out[size_t(q)] == off.out[size_t(q)]);
        }
        std::printf("%6d  %12d  %9d  %14zu | %d\n", n, full_rps, n * full_rps, full_launches, full_off);
    }
}

int main()
{
    //                        n: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25..32
    const int want32[33] = {0, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int want40[33] = {0, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int want48[33] = {0, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    // the per-session index scratch (Model::ds41_sess_rows = index_sess_rows) holds every captured ns-session main width, at every K and MX
    for (int k = 0; k <= 5; ++k)
        for (int mx : {32, 40, 48})
            for (int ns = 2; ns <= 32; ++ns)
                for (bool sd : {false, true})
                {
                    ctx = "sess rows k " + std::to_string(k) + " mx " + std::to_string(mx) + " ns " + std::to_string(ns) + (sd ? " sd" : "");
                    const bool on = k > 0;
                    CHECK(ms_rps(on, sd, k, mx, ns) >= 1);
                    CHECK(ms_rps(on, sd, k, mx, ns) <= index_sess_rows(on, k));
                }
    ctx = "sess rows"; CHECK(index_sess_rows(true, K) == 6); CHECK(index_sess_rows(false, 0) == 1);
    check_mx(32, want32, 16, 6);
    check_mx(40, want40, 20, 8);
    check_mx(48, want48, 24, 8);
    if (fails) { std::printf("batch_ds41_sessdrafts_test: %d FAILED\n", fails); return 1; }
    std::printf("batch_ds41_sessdrafts_test: OK\n");
    return 0;
}
