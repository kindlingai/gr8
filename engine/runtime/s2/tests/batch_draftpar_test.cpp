// GR8_DRAFT_PAR (s2_batch.h Batch::draft_par): a solo DFlash2 round launches the drafter step (phase 1) before the MTP first
// form and waits for it (phase 2) at the end of the round. Against the serial batch, every scenario must give the same
// committed tokens, the same drafter requests, the same target / first-form / chain steps and the same stats; and no
// target step or other drafter step may start while one is in flight. Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_draftpar_test.cpp \
//          engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm -o /tmp/batch_draftpar_test
#include "s2_batch.h"
#include <cstdio>
#include <string>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1

struct Fake
{
    int mtp = 0, dfl = 0;             // drafting quality: MTP 0 all right, 1 two in three, 2 none; DFlash2 depth = 7 - dfl
    std::vector<std::string> steps, reqs;
    std::vector<int> got, last;       // committed tokens; the last draft or chain step's tokens (lp)
    bool pending = false;             // a phase 1 drafter step without its phase 2
    DflDrafts stash{};
    long overlapped = 0, launches = 0, errors = 0;
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            if (s.kind == Step::MAIN && pending) ++errors;     // a target step while the drafter step runs
            if (pending) ++overlapped;
            std::string l = s.kind == Step::MAIN ? "M" : s.kind == Step::CHAIN ? "C" : "D";
            std::vector<int> out;
            for (const Row& r : s.r)
            {
                const int p = r.spos - 1;
                const bool ok = s.kind == Step::MAIN || mtp == 0 || (mtp == 1 && r.spos % 3 != 0);
                out.push_back(ok ? target(p) : -2);
                l += " " + std::to_string(r.tok) + "@" + std::to_string(r.pos);
            }
            steps.push_back(l);
            if (s.kind != Step::MAIN) last = out;
            return out;
        };
        h.dfl_start = [](int) { return 0; };
        h.dfl = [this](const DflReq& r) {
            if (r.phase == 2)
            {
                if (!pending) ++errors;
                pending = false;
                return stash;
            }
            if (pending) ++errors;
            ++launches;
            reqs.push_back(std::string(r.query ? "Q" : "C") + std::to_string(r.rows) + "@" + std::to_string(r.pos0) + " t" +
                           std::to_string(r.tok0) + " lo" + std::to_string(r.lo));
            DflDrafts o{};
            const int q0 = r.pos0 + r.rows;
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = i <= 7 - dfl ? target(q0 + i - 1) : -1; o.lp[i] = -0.05f * i - 0.01f * (q0 % 7); }
            if (r.phase == 1)
            {
                pending = true;
                stash = o;
                DflDrafts g{};
                for (int i = 0; i < DFL_B; ++i) { g.tok[i] = -9; g.lp[i] = 1.f; }   // must not be used
                return g;
            }
            return o;
        };
        h.emit = [this](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };
        h.lp = [this](int row) { return row < (int)last.size() ? -0.02f * (float)((last[row] + 3) % 11) : 0.f; };
        return h;
    }
};

struct Run
{
    std::vector<std::string> steps, reqs;
    std::vector<int> got;
    Stats st;
    long overlapped = 0, launches = 0, errors = 0;
};

// one scenario: dsel on, kstop mode km; phases of (mtp, dfl) quality, n rounds each
static Run scenario(bool par, int km, bool insert, int reuse)
{
    Fake f;
    Batch b(2, 32, f.hooks());
    b.draft_par(par);
    b.dflash(7, g6rt::SpecLen());
    DSel sel;
    sel.on = true;
    sel.dwell = 3;
    sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
    b.dsel(sel);
    if (km)
    {
        KStop k;
        k.mode = km; k.kmax = 3;
        b.kstop(k);
    }
    if (reuse) { DflReuse r; r.mode = reuse; r.min = 6; b.dfl_reuse(r); }
    Session s; s.prompt.assign(12, 5); s.first = 11; s.chunk = 8; s.n_gen = 100000; s.dflash = !reuse;
    if (insert) { s.ins.after = 40; s.ins.tokens = {900, 901, 902}; }
    b.admit({0}, {s});
    if (reuse) b.promote(0);
    const int phase[][3] = {{0, 5, 25}, {0, 0, 25}, {1, 4, 30}, {2, 1, 30}, {0, 6, 30}, {1, 2, 30}};
    for (const auto& p : phase)
    {
        f.mtp = p[0]; f.dfl = p[1];
        for (int i = 0; i < p[2]; ++i) CHECK(b.round({0}));
    }
    Run r;
    r.steps = f.steps; r.reqs = f.reqs; r.got = f.got; r.st = b.stats(0);
    r.overlapped = f.overlapped; r.launches = f.launches; r.errors = f.errors + (f.pending ? 1 : 0);
    return r;
}

int main()
{
    long overlapped = 0, mtp_rounds = 0, dfl_rounds = 0, switches = 0, inserted = 0, cases = 0;
    for (int km : {0, 1, 2})
        for (int insert : {0, 1})
            for (int reuse : {0, 2})
            {
                const Run a = scenario(false, km, insert, reuse), c = scenario(true, km, insert, reuse);
                const bool same = a.got == c.got && a.reqs == c.reqs && a.steps == c.steps && a.st.rounds == c.st.rounds &&
                                  a.st.tokens == c.st.tokens && a.st.mtp_rounds == c.st.mtp_rounds && a.st.dfl_rounds == c.st.dfl_rounds &&
                                  a.st.switches == c.st.switches && a.st.acc == c.st.acc && a.st.inserted == c.st.inserted;
                CHECK(same);
                CHECK(a.errors == 0 && c.errors == 0 && a.overlapped == 0);
                bool exact = true;   // the session's own tokens are the target's (the insert's are its own)
                for (size_t i = 0; i < c.got.size(); ++i) exact &= c.got[i] == target(11 + (int)i) || c.got[i] >= 900;
                CHECK(exact);
                if (!same || c.errors)
                    std::printf("  km %d insert %d reuse %d: got %zu/%zu reqs %zu/%zu steps %zu/%zu errors %ld\n", km,
                                insert, reuse, a.got.size(), c.got.size(), a.reqs.size(), c.reqs.size(), a.steps.size(), c.steps.size(), c.errors);
                overlapped += c.overlapped; mtp_rounds += c.st.mtp_rounds; dfl_rounds += c.st.dfl_rounds; ++cases;
                switches += c.st.switches; inserted += c.st.inserted;
            }
    CHECK(mtp_rounds > 0 && dfl_rounds > 0 && overlapped > 0 && switches > 0 && inserted > 0);
    std::printf("batch_draftpar_test: %s (%ld cases, %ld MTP / %ld DFlash2 rounds, %ld switches, %ld inserted, %ld steps under a "
                "drafter step)\n", fails ? "FAIL" : "PASS", cases, mtp_rounds, dfl_rounds, switches, inserted, overlapped);
    return fails ? 1 : 0;
}
