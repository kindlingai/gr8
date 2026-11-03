// GR8_DS_SESS_DRAFTS (s2/ds41_sess_drafts.h, Batch::set_sess_drafts) on the real s2::Batch with a fake engine, DS41 shape (K = 5 DSpark
// drafts, G6_MX = 32 rows a step, 32 slots):
//   (a) the rule: kv(n) for n = 1..32 active sessions, and one table line per n = 1, 5, 6, 8, 9, 16: rows a session, main rows, draft
//       launches of at most 6 sessions (sess_drafts_chunk), and the same n with the knob off (today)
//   (b) every decode-round MAIN step of n sessions gives each session kv(n) + 1 rows and fits 32 rows; every DRAFT step's sessions split
//       into launches of at most 6 that cover them once, in order (draft_chunks)
//   (c) knob on at 1..5 sessions: the same steps (kind, rows, every row's session / token / position) and the same tokens as knob off
//   (d) every committed token is the target's, knob on, at every n
// Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -Wall -DG6_MODEL_DS41=1 -I engine/runtime engine/runtime/s2/tests/batch_ds41_sessdrafts_test.cpp <knobs lib>
#include "s2_batch.h"
#include <cstdio>
#include <string>
#include <tuple>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 30) std::printf("FAIL %s:%d %s  [%s]\n", __FILE__, __LINE__, #c, ctx.c_str()); ++fails; } } while (0)
static std::string ctx;

constexpr int K = 5, MXR = 32, BLOCK = 5, MAXS = 8;
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
static Run run(int n, bool on)
{
    Run r;
    Batch b(K, MXR, r.f.hooks());
    b.set_mtp_sessions(std::min(MXR / BLOCK, 8));   // g6m4.cu's Config::mtp_sessions for 32 rows
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

int main()
{
    // (a) the rule
    const int want[17] = {0, 5, 5, 5, 5, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
    for (int n = 1; n <= 16; ++n) { ctx = "kv n " + std::to_string(n); CHECK(sess_drafts_kv(K, MXR, n) == want[n]); CHECK(n * (want[n] + 1) <= MXR); }
    for (int n = 17; n <= 32; ++n) { ctx = "kv n " + std::to_string(n); CHECK(sess_drafts_kv(K, MXR, n) == 0); }
    ctx = "max"; CHECK(sess_drafts_max(K, MXR, 32) == 16); CHECK(sess_drafts_max(K, MXR, 8) == 8);
    ctx = "chunk"; CHECK(sess_drafts_chunk(MXR, BLOCK, MAXS) == 6);
    for (int S = 1; S <= 16; ++S)
    {
        ctx = "chunks S " + std::to_string(S);
        const auto c = draft_chunks(S, 6);
        CHECK((int)c.size() == (S + 5) / 6);
        int next = 0;
        for (const auto& p : c) { CHECK(p.first == next && p.second >= 1 && p.second <= 6); next += p.second; }
        CHECK(next == S);
    }

    std::printf("active  rows/session  main rows  draft launches | knob off (today): rows/session\n");
    for (int n : {1, 5, 6, 8, 9, 16})
    {
        const Run on = run(n, true), off = run(n, false);
        // (b) every decode-round main step: kv(its sessions) + 1 rows a session, at most 32 rows; the full-n rounds for the table
        int full_rps = -1, full_off = -1, full_rounds = 0;
        for (const Step& s : on.f.mains)
        {
            const int ns = (int)s.sess.size(), kv = sess_drafts_kv(K, MXR, ns);
            ctx = "on n " + std::to_string(n) + " ns " + std::to_string(ns);
            CHECK(s.M() <= MXR);
            for (size_t j = 0; j < s.sess.size(); ++j) CHECK(s.rows[j] == kv + 1);
            if (ns == n) { ++full_rounds; if (full_rps < 0) full_rps = s.rows[0]; CHECK(s.rows[0] == full_rps); }
        }
        for (const Step& s : off.f.mains) if ((int)s.sess.size() == n) { full_off = s.rows[0]; break; }
        ctx = "on n " + std::to_string(n);
        CHECK(full_rounds > 0);
        CHECK(full_rps == want[n] + 1);
        for (const Step& d : on.f.drafts)
        {
            CHECK(d.M() <= MXR);
            const auto c = draft_chunks((int)d.sess.size(), 6);
            CHECK((int)c.size() == ((int)d.sess.size() + 5) / 6);
        }
        // (c) 1..5: the same steps and tokens as knob off
        if (n <= 5) { ctx = "same n " + std::to_string(n); CHECK(on.f.trace == off.f.trace); CHECK(on.out == off.out); CHECK(full_off == full_rps); }
        // (d) every committed token is the target's
        for (int q = 0; q < n; ++q)
        {
            ctx = "tokens n " + std::to_string(n) + " q " + std::to_string(q);
            const int plen = 9 + q % 3;
            CHECK((int)on.out[size_t(q)].size() == 48);
            for (size_t i = 0; i < on.out[size_t(q)].size(); ++i) CHECK(on.out[size_t(q)][i] == target(q, plen - 1 + int(i)));
            CHECK(on.out[size_t(q)] == off.out[size_t(q)]);
        }
        std::printf("%6d  %12d  %9d  %14zu | %d\n", n, full_rps, n * full_rps, draft_chunks(n, 6).size(), full_off);
    }
    if (fails) { std::printf("batch_ds41_sessdrafts_test: %d FAILED\n", fails); return 1; }
    std::printf("batch_ds41_sessdrafts_test: OK\n");
    return 0;
}
