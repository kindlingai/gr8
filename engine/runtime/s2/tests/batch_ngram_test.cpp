// N-gram lookup as a draft source (s2_batch.h Batch::ngram, ngram_lookup.h; S5, #56): the table file parses and chains
// with backoff; a solo DFlash2 round verifies the table's chain when it promises more tokens per verify ms than the
// drafter's drafts and the drafter's otherwise; every committed token is the target's, the same with the table on and
// off. Host only, a fake engine.
#include "s2_batch.h"
#include <cstdio>
#include <string>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// the target's text: a cycle of 50 tokens (ids 1000..1049), so the table can know every next token
static int target(int pos) { return 1000 + (pos + 1) % 50; }    // the token at position pos + 1

static std::string cycle_table(const char* cal, double p)
{   // every 3-token window of the cycle -> its next token at probability p
    std::string s = "# test\ng6ngram 1\nns 3\ncal " + std::string(cal) + "\n";
    for (int i = 0; i < 50; ++i)
    {
        char b[96];
        snprintf(b, sizeof b, "e 3 %d %d %d %d %.3f\n", 1000 + i, 1000 + (i + 1) % 50, 1000 + (i + 2) % 50, 1000 + (i + 3) % 50, p);
        s += b;
    }
    return s;
}

struct Run
{
    std::vector<int> got;
    std::vector<int> widths;          // main steps' rows after the prompt
    Stats st;
};

// one solo DFlash2 request of n_gen tokens; dfl_right: how many of the drafter's 7 drafts are right
static Run run(const NgramSrc* ng, int dfl_right)
{
    Run o;
    bool prompt = true;
    Hooks h;
    h.run = [&](const Step& s) {
        std::vector<int> out;
        for (const Row& r : s.r) out.push_back(s.kind == Step::MAIN ? target(r.pos) : target(r.spos - 1));
        if (s.kind == Step::MAIN && !prompt) o.widths.push_back(s.M());
        return out;
    };
    h.dfl_start = [&](int) { return 0; };
    h.dfl = [&](const DflReq& r) {
        DflDrafts d{};
        const int q0 = r.pos0 + r.rows;
        for (int i = 1; i < DFL_B; ++i) { d.tok[i] = i <= dfl_right ? target(q0 + i - 1) : 7; d.lp[i] = i <= dfl_right ? -0.01f : -3.f; }
        return d;
    };
    h.emit = [&](int, const int* t, int k, int) { o.got.insert(o.got.end(), t, t + k); return false; };
    Batch b(3, 32, h);
    b.dflash(7, g6rt::SpecLen());
    if (ng) b.ngram(*ng);
    Session s;
    for (int i = 0; i < 10; ++i) s.prompt.push_back(i < 7 ? 5 : target(i - 1));   // its last 3 tokens are the cycle's
    s.first = 9; s.chunk = 8; s.n_gen = 200; s.dflash = true;
    b.admit({0}, {s});
    prompt = false;
    while (b.round({0})) {}
    o.got.resize(std::min<size_t>(o.got.size(), 200));
    o.st = b.stats(0);
    return o;
}

int main()
{
    // the parser
    {
        g6rt::NgramTable t;
        std::string err;
        CHECK(t.parse("g6ngram 1\nns 2 4\ncal none\ne 4 1 2 3 4 9 0.9\ne 2 3 4 8 0.5\ne 2 4 9 5 0.8\n", &err));
        CHECK(t.ns.size() == 2 && t.ns[0] == 4 && !t.has_cal && t.t.size() == 3);
        int tok; float p;
        const int c1[] = {1, 2, 3, 4};
        CHECK(t.next(c1, 4, &tok, &p) && tok == 9 && p > 0.89f);          // the 4-gram first
        const int c2[] = {7, 7, 3, 4};
        CHECK(t.next(c2, 4, &tok, &p) && tok == 8);                       // backoff to the 2-gram
        const int c3[] = {7, 7, 7, 7};
        CHECK(!t.next(c3, 4, &tok, &p));
        int out[8]; double c[8];
        CHECK(t.chain(c1, 4, 7, out, c) == 2 && out[1] == 9 && out[2] == 5 && std::abs(c[2] - 0.8) < 1e-6);   // 4 9 -> 5, then none
        CHECK(t.chain(c1, 4, 1, out, c) == 1);
        CHECK(t.parse("g6ngram 1\nns 2\ncal 0 .1 .2 .3 .4 .5 .6 .7\ne 2 1 2 3 0.99\n", &err) && t.has_cal && t.conf(0.99) == 0.7 && t.conf(0.1) == 0);
        const uint64_t d = t.digest;
        CHECK(t.parse("g6ngram 1\nns 2\ncal 0 .1 .2 .3 .4 .5 .6 .7\ne 2 1 2 3 0.98\n", &err) && t.digest != d);
        CHECK(!t.parse("g6ngram 2\nns 2\ncal none\n", &err));
        CHECK(!t.parse("g6ngram 1\nns 2\ncal 0.1 0.2\n", &err));
        CHECK(!t.parse("g6ngram 1\nns 2\ncal none\ne 3 1 2 3 4 0.5\n", &err));     // length not in ns
        CHECK(!t.parse("g6ngram 1\nns 2\ncal none\ne 2 1 2 3 1.5\n", &err));
        CHECK(!t.parse("g6ngram 1\nns 2\n", &err));                               // no cal
    }
    // the drafter alone: 3 of 7 right
    const Run base = run(nullptr, 3);
    CHECK(base.got.size() == 200 && base.st.ngram_rounds == 0);
    bool exact = true;
    for (int i = 0; i < 200; ++i) exact &= base.got[i] == target(9 + i);
    CHECK(exact);
    // a confident table on the same text: its 7-token chains win every round, and the output is the same
    g6rt::NgramTable good;
    std::string err;
    CHECK(good.parse(cycle_table("0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.98", 0.995), &err));
    NgramSrc ng; ng.t = &good;
    const Run on = run(&ng, 3);
    CHECK(on.got == base.got);
    CHECK(on.st.ngram_rounds > 0 && on.st.ngram_rounds == on.st.rounds && on.st.ngram_acc == on.st.ngram_drafted);
    CHECK(on.st.rounds < base.st.rounds);                                  // 8 tokens a round against 4
    for (int w : on.widths) CHECK(w == 8);
    // kmax 3: chains of 3, verify width 4
    ng.kmax = 3;
    const Run k3 = run(&ng, 3);
    CHECK(k3.got == base.got);
    for (int w : k3.widths) CHECK(w == 4);
    // the drafter right on all 7: the table's chain of 7 at calibrated 0.98 each no longer beats it by the margin
    ng.kmax = 7;
    const Run dfl7 = run(&ng, 7);
    CHECK(dfl7.st.ngram_rounds == 0);
    // a calibration that says the table is often wrong: the drafter keeps every round
    g6rt::NgramTable weak;
    CHECK(weak.parse(cycle_table("0.05 0.1 0.15 0.2 0.25 0.3 0.3 0.3", 0.995), &err));
    ng.t = &weak;
    const Run w = run(&ng, 3);
    CHECK(w.st.ngram_rounds == 0 && w.got == base.got);
    // a wrong table (every entry proposes a token the target never samples): chains are verified and rejected, the
    // output still the target's
    std::string bad = "g6ngram 1\nns 3\ncal none\n";
    for (int i = 0; i < 50; ++i)
    {
        char b[96];
        snprintf(b, sizeof b, "e 3 %d %d %d 3 0.999\n", 1000 + i, 1000 + (i + 1) % 50, 1000 + (i + 2) % 50);
        bad += b;
    }
    g6rt::NgramTable wrong;
    CHECK(wrong.parse(bad, &err));
    ng.t = &wrong; ng.kmin = 1;
    const Run wr = run(&ng, 3);
    CHECK(wr.got == base.got && wr.st.ngram_rounds > 0 && wr.st.ngram_acc == 0);
    CHECK(ngram_sig(NgramSrc()).empty() && !ngram_sig(ng).empty());
    std::printf("batch_ngram_test: %s (rounds: drafter %ld, table %ld)\n", fails ? "FAIL" : "PASS", base.st.rounds, on.st.rounds);
    return fails ? 1 : 0;
}
