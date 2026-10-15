// Drafter choice (s2_batch.h Batch::dsel): a DFlash2 session moves to MTP while MTP drafts better and back when DFlash2
// does, the drafter keeps its context on MTP rounds, and every committed token is the target's. Host only, a fake engine.
#include "s2_batch.h"
#include <algorithm>
#include <cstdio>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1

int main()
{
    bool mtp_good = true;     // phase 1: MTP drafts right, DFlash2 only its first draft; phase 2: the reverse
    bool both = false;        // phase 3, a new request: both drafters right (structured text)
    std::vector<char> rounds; // per decode round: 'm' (MTP verify of K + 1 rows) or 'd' (DFlash2, 8 rows)
    int ctx = 0, query = 0, chains = 0;
    Hooks h;
    h.run = [&](const Step& s) {
        std::vector<int> out;
        for (const Row& r : s.r)  // a verify row samples position pos + 1; an MTP draft row drafts spos (wrong in phase 2)
            out.push_back(s.kind == Step::MAIN || mtp_good ? target(r.spos - 1) : -2);
        if (s.kind == Step::MAIN && s.M() > 1) rounds.push_back(s.M() == 8 ? 'd' : 'm');
        if (s.kind == Step::CHAIN) ++chains;
        return out;
    };
    h.dfl_start = [&](int) { return 0; };
    h.dfl = [&](const DflReq& r) {
        ++(r.query ? query : ctx);
        DflDrafts o{};
        const int q0 = r.pos0 + r.rows;
        for (int i = 1; i < DFL_B; ++i) { o.tok[i] = i == 1 || !mtp_good || both ? target(q0 + i - 1) : -1; o.lp[i] = -0.1f; }
        return o;
    };
    std::vector<int> got;
    h.emit = [&](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };

    Batch b(3, 32, h);
    b.dflash(7, g6rt::SpecLen());
    DSel sel;
    sel.on = true;
    sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
    b.dsel(sel);
    Session s; s.prompt.assign(10, 5); s.first = 9; s.chunk = 8; s.n_gen = 4000; s.dflash = true;
    b.admit({0}, {s});
    for (int i = 0; i < 40; ++i) CHECK(b.round({0}));
    CHECK(rounds.size() == 40 && rounds[0] == 'd');
    CHECK(rounds.back() == 'm');                                  // MTP commits 4 a round against DFlash2's 2
    const int ctx1 = ctx;
    CHECK(ctx1 > 20 && query > 0 && chains > 0);                  // context-only drafter steps on the MTP rounds
    mtp_good = false;
    for (int i = 0; i < 200; ++i) CHECK(b.round({0}));
    CHECK(rounds.back() == 'd');                                  // back on DFlash2 once its probes score well
    CHECK(b.stats(0).switches >= 2 && b.stats(0).mtp_rounds > 0);
    bool exact = true;                                            // every committed token is the target's
    for (size_t i = 0; i < got.size(); ++i) exact &= got[i] == target(9 + (int)i);
    CHECK(exact);
    // phase 3: the request ends on MTP; the next request starts with fresh estimates and stays on DFlash2, which commits 8 a
    // round against MTP's 3
    mtp_good = true;
    for (int i = 0; i < 40; ++i) CHECK(b.round({0}));
    CHECK(rounds.back() == 'm');
    b.cancel(0); b.release(0);
    both = true;
    rounds.clear();
    b.admit({0}, {s});
    for (int i = 0; i < 40; ++i) CHECK(b.round({0}));
    CHECK(std::count(rounds.begin(), rounds.end(), 'm') == 0);
    // A session reset must preserve custom verify costs while clearing learned acceptance.
    // With both drafters right, the defaults above prefer DFlash2. This table makes MTP cheaper.
    sel.d.table[8] = 1000000;
    sel.m.table[4] = 1;
    b.dsel(sel);
    for (int request = 0; request < 2; ++request)
    {
        b.cancel(0); b.release(0);
        rounds.clear();
        b.admit({0}, {s});
        for (int i = 0; i < 40; ++i) CHECK(b.round({0}));
        CHECK(!rounds.empty() && rounds.back() == 'm');
        CHECK(b.stats(0).mtp_rounds > 0);
    }
    std::printf("batch_dsel_test: %s (%ld rounds, %ld on MTP, %ld switches)\n", fails ? "FAIL" : "PASS", b.stats(0).rounds,
                b.stats(0).mtp_rounds, b.stats(0).switches);
    return fails ? 1 : 0;
}
