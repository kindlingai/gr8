// Batch with a DFlash2 session (s2_batch.h): the prompt and round steps of rank.cuh / dflash.cuh dflash_sequence, the MTP
// first-form pass beside them, and demote() handing the session to the MTP rounds. Host only, a fake engine.
#include "s2_batch.h"
#include <cstdio>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// the target's sample at position pos + 1 and the drafter's guesses: drafts 1..3 right, the rest wrong
static int target(int pos) { return (pos * 37 + 11) % 997; }

int main()
{
    std::vector<std::string> log;
    Hooks h;
    h.run = [&](const Step& s) {
        std::vector<int> out;
        for (const Row& r : s.r) out.push_back(target(r.pos));
        log.push_back(std::string(s.kind == Step::MAIN ? "m" : s.kind == Step::DRAFT ? "d" : "c") + std::to_string(s.M()) + "@" + std::to_string(s.r[0].pos));
        return out;
    };
    int starts = 0;
    h.dfl_start = [&](int) { ++starts; return 100; };
    h.dfl = [&](const DflReq& r) {
        log.push_back(std::string(r.query ? "Q" : "C") + std::to_string(r.rows) + "@" + std::to_string(r.pos0));
        DflDrafts o{};
        const int q0 = r.pos0 + r.rows;
        for (int i = 1; i < DFL_B; ++i) { o.tok[i] = i <= 3 ? target(q0 + i - 1) : -1; o.lp[i] = -0.1f * i; }
        return o;
    };
    std::vector<int> emitted;
    h.emit = [&](int, const int* t, int k, int) { emitted.insert(emitted.end(), t, t + k); return false; };

    Batch b(2, 24, h);
    b.dflash(7, g6rt::SpecLen());           // no adaptive policy: every round verifies 7
    Session s; s.prompt.assign(300, 5); s.first = 299; s.chunk = 8; s.n_gen = 20; s.dflash = true;
    b.admit({0}, {s});
    CHECK(starts == 1);
    // prompt tail: main 1 row at 299, MTP first form, drafter context + query; no MTP chain for a DFlash2 session
    CHECK((log == std::vector<std::string>{"m1@299", "d1@299", "Q1@299"}));
    log.clear();
    CHECK(b.round({0}));
    // a round: verify 8 rows at 300 (bonus + 7 drafts), accept 3, MTP first form over 4 rows, drafter over the same 4
    CHECK((log == std::vector<std::string>{"m8@300", "d4@300", "Q4@300"}));
    CHECK(emitted.size() == 1 + 4);
    log.clear();
    b.demote();                              // a second session is resident: K - 1 = 1 chain step
    CHECK((log == std::vector<std::string>{"c1@304"}));
    log.clear();
    CHECK(b.round({0}));
    CHECK(!log.empty() && log[0] == "m3@304");   // MTP from here: 3 verify rows
    CHECK(b.stats(0).acc.size() == 8 && b.stats(0).rounds == 2 && b.stats(0).drafted == 7 + 2);
    std::printf("batch_dflash_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
