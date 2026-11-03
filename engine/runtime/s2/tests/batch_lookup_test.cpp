// Context lookup as a draft source (s2_batch.h Batch::lookup, ctx_lookup.h, GR8_LOOKUP): the index finds the longest
// earlier match; a solo DFlash2 round verifies the lookup's chain when the reply repeats earlier text and the drafter's
// drafts otherwise; every committed token is the target's, the same with the lookup on and off. Host only, a fake engine.
#include "s2_batch.h"
#include <cstdio>
#include <functional>
#include <string>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

struct Run
{
    std::vector<int> got;
    std::vector<int> widths;          // main steps' rows after the prompt
    Stats st;
};

// one solo DFlash2 request: tok(p) is the text's token at position p (prompt 0..59, then the reply); the drafter's
// first dfl_right drafts are right
static Run run(const std::function<int(int)>& tok, const LookupSrc* lk, const NgramSrc* ng, int dfl_right, const g6rt::InsertRule& ins = {})
{
    Run o;
    bool prompt = true;
    Hooks h;
    h.run = [&](const Step& s) {
        std::vector<int> out;
        for (const Row& r : s.r) out.push_back(s.kind == Step::MAIN ? tok(r.pos + 1) : tok(r.spos));
        if (s.kind == Step::MAIN && !prompt) o.widths.push_back(s.M());
        return out;
    };
    h.dfl_start = [&](int) { return 0; };
    h.dfl = [&](const DflReq& r) {
        DflDrafts d{};
        const int q0 = r.pos0 + r.rows;
        for (int i = 1; i < DFL_B; ++i) { d.tok[i] = i <= dfl_right ? tok(q0 + i) : 7; d.lp[i] = i <= dfl_right ? -0.01f : -3.f; }
        return d;
    };
    h.emit = [&](int, const int* t, int k, int) { o.got.insert(o.got.end(), t, t + k); return false; };
    Batch b(3, 32, h);
    g6rt::SpecLen sl;
    b.dflash(7, sl);
    if (lk) b.lookup(*lk);
    if (ng) b.ngram(*ng);
    Session s;
    for (int i = 0; i < 60; ++i) s.prompt.push_back(tok(i));
    s.first = 59; s.chunk = 8; s.n_gen = 200; s.dflash = true; s.ins = ins;
    b.admit({0}, {s});
    prompt = false;
    while (b.round({0})) {}
    o.got.resize(std::min<size_t>(o.got.size(), 200));
    o.st = b.stats(0);
    return o;
}

int main()
{
    // the index
    {
        g6rt::CtxLookup l;
        int out[8], ml;
        double c[8];
        // 1 2 3 4 5 9 3 4 6 | 2 3 4: the tail "3 4" occurs at 3 and 7; "2 3 4" matches the first, which wins
        const std::vector<int> t = {1, 2, 3, 4, 5, 9, 3, 4, 6, 2, 3, 4};
        CHECK(l.chain(t.data(), (int)t.size(), 7, out, c, &ml) == 7 && ml == 3 && out[1] == 5 && out[2] == 9 && out[7] == 3);
        CHECK(std::abs(c[1] - g6rt::CtxLookup::seed[1]) < 1e-12);
        // equal match lengths: the latest; a chain stops at the tail
        const std::vector<int> u = {8, 3, 4, 5, 8, 3, 4, 6, 8, 3, 4};
        g6rt::CtxLookup l2;
        CHECK(l2.chain(u.data(), (int)u.size(), 7, out, c, &ml) == 4 && ml == 3 && out[1] == 6 && out[4] == 4);
        CHECK(l2.chain(u.data(), (int)u.size(), 2, out, c, &ml) == 2);
        // no earlier place
        g6rt::CtxLookup l3;
        const std::vector<int> v = {1, 2, 3, 4, 5};
        CHECK(l3.chain(v.data(), 5, 7, out, c, &ml) == 0 && ml == 0);
        // the estimate learns: misses lower a bucket's chance, hits raise it
        g6rt::CtxLookup e;
        const double p0 = e.p(4);
        for (int i = 0; i < 20; ++i) e.observe(20, 7, 0);
        CHECK(e.p(4) < 0.2 && e.p(4) < p0 && std::abs(e.p(0) - g6rt::CtxLookup::seed[0]) < 1e-12);
        for (int i = 0; i < 40; ++i) e.observe(20, 7, 7);
        CHECK(e.p(4) > 0.85);
    }
    // repeated text: a cycle of 50 tokens, so the prompt holds every continuation
    const auto cyc = [](int p) { return 1000 + p % 50; };
    const Run base = run(cyc, nullptr, nullptr, 3);
    bool exact = base.got.size() == 200;
    for (int i = 0; i < 200 && exact; ++i) exact = base.got[i] == cyc(60 + i);
    CHECK(exact && base.st.lookup_rounds == 0);
    LookupSrc lk; lk.on = true;
    const Run on = run(cyc, &lk, nullptr, 3);
    CHECK(on.got == base.got);
    CHECK(on.st.lookup_rounds > 0 && on.st.lookup_rounds == on.st.rounds && on.st.lookup_acc == on.st.lookup_drafted);
    CHECK(on.st.rounds < base.st.rounds);
    // the first rounds start from the seed chance of a medium match and take shorter chains, then 7 drafts each
    int full = 0;
    for (int w : on.widths) full += w == 8;
    CHECK(full >= (int)on.widths.size() - 3);
    lk.kmax = 3;
    const Run k3 = run(cyc, &lk, nullptr, 3);
    CHECK(k3.got == base.got);
    for (int w : k3.widths) CHECK(w == 4);
    lk.kmax = 7;
    // text with no repeated bigram: the lookup never proposes
    const auto uniq = [](int p) { return 2000 + p; };
    const Run u0 = run(uniq, nullptr, nullptr, 3), u1 = run(uniq, &lk, nullptr, 3);
    CHECK(u1.got == u0.got && u1.st.lookup_rounds == 0 && u1.st.rounds == u0.st.rounds);
    // short matches that go wrong ("a b" then a new token each time): match length 2 never beats the drafter
    const auto ab = [](int p) { return p % 3 == 0 ? 3000 : p % 3 == 1 ? 3001 : 4000 + p; };
    const Run a0 = run(ab, nullptr, nullptr, 3), a1 = run(ab, &lk, nullptr, 3);
    CHECK(a1.got == a0.got && a1.st.lookup_rounds == 0);
    // a strong drafter (6 of 7 right) against a lookup that is always right: the lookup's estimate starts below the
    // drafter's, learns from the rounds the drafter verifies, and then takes the rounds
    {
        const Run d6 = run(cyc, nullptr, nullptr, 6), l6 = run(cyc, &lk, nullptr, 6);
        CHECK(l6.got == d6.got && l6.st.lookup_rounds > l6.st.rounds / 2 && l6.st.rounds < d6.st.rounds);
    }
    // the drafter right on all 7: the lookup's 7 can only tie it, so the drafter keeps the rounds it wins on cost
    const Run d7 = run(cyc, &lk, nullptr, 7);
    CHECK(d7.got == base.got);
    // an insert rule that cuts inside a lookup chain: the reply is the same as without the lookup
    {
        g6rt::InsertRule ins;
        ins.after = 37; ins.tokens = {7, 8, 9}; ins.unless = -1;
        const Run plain = run(cyc, nullptr, nullptr, 3, ins);
        const Run look = run(cyc, &lk, nullptr, 3, ins);
        CHECK(plain.got.size() == 200 && look.got == plain.got && look.st.lookup_rounds > 0);
        CHECK(plain.got[37] == 7 && plain.got[39] == 9);
    }
    // the table on too: every round verifies one source's chain, the output the same
    {
        std::string tab = "g6ngram 1\nns 3\ncal none\n";
        for (int i = 0; i < 50; ++i)
        {
            char l[96];
            snprintf(l, sizeof l, "e 3 %d %d %d %d 0.995\n", cyc(i), cyc(i + 1), cyc(i + 2), cyc(i + 3));
            tab += l;
        }
        g6rt::NgramTable t;
        std::string err;
        CHECK(t.parse(tab, &err));
        NgramSrc ng; ng.t = &t;
        const Run both = run(cyc, &lk, &ng, 3);
        CHECK(both.got == base.got && both.st.lookup_rounds + both.st.ngram_rounds == both.st.rounds && both.st.ngram_rounds > 0);
    }
    // bounds
    {
        LookupSrc bad; bad.on = true; bad.kmax = 8;
        Hooks h;
        Batch b(3, 32, h);
        bool threw = false;
        try { b.lookup(bad); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    std::printf("batch_lookup_test: %s (rounds: drafter %ld, lookup %ld)\n", fails ? "FAIL" : "PASS", base.st.rounds, on.st.rounds);
    return fails ? 1 : 0;
}
