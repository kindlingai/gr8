// Drafter choice cost model (s2_batch.h DSelTable, DSel::model, G6_S2_DSEL_MODEL): the table parses to one canonical
// form whatever its spelling, survives each request's fresh estimates, prices MTP rounds as the confidence stop runs
// them (so a K-stop MTP that verifies three confident drafts wins where the fixed-K pricing keeps DFlash2, and one that
// verifies one draft loses to a better DFlash2), ranks reading the same table make the same choices bit for bit, and
// with the model off the steps are the ones without it. Host only, a fake engine.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_dselmodel_test.cpp -o /tmp/batch_dselmodel_test
#include "s2_batch.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <tuple>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1

// MTP drafts right while mtp_ok(position in the round), with probability p_ok / p_bad; DFlash2's first dright drafts right
struct Fake
{
    int dright = 3;
    float p_ok = 0.95f, p_bad = 0.6f;
    int mtp_depth = 99;       // MTP drafts d[1..mtp_depth] right, the later ones wrong
    int idx = 0;              // the draft index of the last DRAFT (1) or CHAIN (2, 3) step
    std::vector<float> lp;
    std::vector<std::tuple<int, int, long>> trace;   // (kind, M, round) per step
    std::vector<int> got;
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            std::vector<int> out;
            if (s.kind != Step::MAIN) { lp.clear(); idx = s.kind == Step::DRAFT ? 1 : idx + 1; }
            for (const Row& r : s.r)
            {
                if (s.kind == Step::MAIN) { out.push_back(target(r.spos - 1)); continue; }
                const bool ok = idx <= mtp_depth;
                out.push_back(ok ? target(r.spos - 1) : -2);
                lp.push_back(std::log(ok ? p_ok : p_bad));
            }
            trace.emplace_back((int)s.kind, s.M(), s.round);
            return out;
        };
        h.dfl_start = [](int) { return 0; };
        h.dfl = [this](const DflReq& r) {
            DflDrafts o{};
            const int q0 = r.pos0 + r.rows;
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = i <= dright ? target(q0 + i - 1) : -1; o.lp[i] = i <= dright ? -0.05f : -3.f; }
            return o;
        };
        h.emit = [this](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };
        h.lp = [this](int r) { return lp.at(r); };
        return h;
    }
};
static Session session()
{
    Session s; s.prompt.assign(10, 5); s.first = 9; s.chunk = 8; s.n_gen = 100000; s.dflash = true;
    return s;
}
static bool exact(const std::vector<int>& got) { bool e = true; for (size_t i = 0; i < got.size(); ++i) e &= got[i] == target(9 + (int)i); return e; }
static DSelTable table(const char* s)
{
    DSelTable t;
    std::string err;
    if (!t.parse(s, &err)) { std::printf("FAIL parse %s: %s\n", s, err.c_str()); ++fails; }
    return t;
}
static const char* TAB = "v2:49.6,v3:57,v4:64,v8:104,d:6.9,m:2,e:1.17,p:4.9,g:2,k:3,n:3";
static const char* TABW = "v2:49.6,v3:57,v4:64,v8:104,d:6.9,m:2,e:1.17,p:4.9,g:2,k:3,n:3,w:1";   // the estimates' prior at 1 round
static DSel dsel(int model, const DSelTable& t, bool start_mtp = false)
{
    DSel sel;
    sel.on = true; sel.start_mtp = start_mtp; sel.probe = 8;
    sel.d.draft_round = 8.1; sel.m.draft_round = 2.9; sel.m.draft_each = 2.1;   // the fixed-K serving MCOST
    sel.model = model; sel.tab = t;
    return sel;
}
static KStop kstop3()
{
    KStop ks; ks.mode = 1; ks.kmax = 3; ks.tau[1] = 0; ks.tau[2] = 0.5; ks.tau[3] = 0.75;
    return ks;
}
// rounds by drafter from the trace: a DFlash2 verify has 8 rows here (no --spec-adaptive), an MTP one 2..4
static int count(const Fake& f, int M_lo, int M_hi, long from = 0)
{
    int n = 0;
    for (const auto& t : f.trace) n += std::get<0>(t) == Step::MAIN && std::get<2>(t) >= from && std::get<1>(t) >= M_lo && std::get<1>(t) <= M_hi;
    return n;
}

int main()
{
    // 1. the table: separators, comments, interpolation, extrapolation, the canonical form, refusals
    {
        const DSelTable a = table(TAB);
        const DSelTable b = table("# a boot's table\n n:3  k:3 g:2\tp:4.9, e:1.17 m:2 d:6.9  v8:104 v4:64 # rows\n v3:57,,v2:49.6\n");
        CHECK(a.str() == b.str());
        CHECK(table(a.str().c_str()).str() == a.str());
        CHECK(table("v2:49.60001,v3:57,v4:64,v8:104,d:6.9,m:2,e:1.17,p:4.9,g:2,k:3,n:3").str() == a.str());   // the 1e-4 grid
        CHECK(table("v2:49.6001,v3:57,v4:64,v8:104,d:6.9,m:2,e:1.17,p:4.9,g:2,k:3,n:3").str() != a.str());
        CHECK(std::fabs(a.v[1] - 42.2) < 1e-9);                  // below the first row: the v2-v3 slope
        CHECK(std::fabs(a.v[6] - 84) < 1e-9);                    // between v4 and v8
        CHECK(std::fabs(a.verify(3.5) - 60.5) < 1e-9 && std::fabs(a.verify(20) - 104) < 1e-9);
        CHECK(a.str().find(' ') == std::string::npos && a.str().find(';') == std::string::npos);
        for (const char* bad : {"", "v2:49.6,d:6.9,m:2,e:1", "v2:49.6,v3:57,m:2,e:1", "v2:49.6,v3:57,d:6.9,m:2,e:1,x:3",
                                "v2:49.6,v3:57,d:6.9,m:2,e:-1", "v2:49.6,v3:57,d:6.9,m:2,e:1,v9:3", "v2:49.6,v3:57,d:6.9,m:2,e:nan",
                                "v2:49.6,v3:57,d:6.9,m:2,e:1,k:3,n:2", "v2:49.6,v3:57,d:6.9,m:2,e", "v2:10,v3:57,d:1,m:1,e:1",
                                "v2:0,v3:57,d:1,m:1,e:1", "v2:49.6,v3:57,d:6.9,m:2,e:1,w:0"})
        {
            DSelTable t;
            std::string err;
            CHECK(!t.parse(bad, &err) && !err.empty());
        }
    }
    // 2. MTP under the stop verifies three confident right drafts (4 tokens per ~68 ms) against DFlash2's 3 of 7 (4 per
    //    ~111 ms): the model moves to MTP and stays; the fixed-K pricing (MTP's second position at its seed, MCOST
    //    2.9) keeps DFlash2. Every committed token is the target's either way
    for (int model : {0, 1})
    {
        Fake f;
        Batch b(2, 32, f.hooks());
        b.dflash(7, g6rt::SpecLen());
        b.dsel(dsel(model, table(TAB)));
        b.kstop(kstop3());
        b.admit({0}, {session()});
        for (int i = 0; i < 200; ++i) CHECK(b.round({0}));
        const int mtp = count(f, 2, 4, 100), dfl = count(f, 8, 8, 100);
        if (model) CHECK(mtp > 90 && b.stats(0).switches >= 1);
        else CHECK(dfl == 100 && b.stats(0).mtp_rounds == 0);
        CHECK(exact(f.got));
        if (model)
        {   // the estimate is the closed form over what the MTP rounds were: 4 tokens at v4 + m + 2e, plus a probe share
            const auto [rm, rd] = b.dsel_rates();
            const DSel& s = b.dsel();
            const double c = (s.rc + s.prior * (64 + 2 + 2 * 1.17)) / (s.rn + s.prior);
            CHECK(std::fabs(c - (64 + 2 + 2 * 1.17)) < 1e-9);
            CHECK(std::fabs(rm - (1 + s.m.rate(1) * 3) / (c + 4.9 / 8)) < 1e-12 && rm > 1.15 * rd);
        }
    }
    // 3. MTP under the stop drafts d2 wrong and unsure (0.95 x 0.3 < 0.5), so it verifies one draft (2 tokens per ~53 ms)
    //    against DFlash2's 5 of 7 (6 per ~111 ms): a request starting on MTP moves to DFlash2 and stays, although the
    //    table's prior (g 2, three drafts) first prices MTP high
    {
        Fake f;
        f.mtp_depth = 1;
        f.p_bad = 0.3f;
        f.dright = 5;
        Batch b(2, 32, f.hooks());
        b.dflash(7, g6rt::SpecLen());
        b.dsel(dsel(1, table(TAB), true));
        b.kstop(kstop3());
        b.admit({0}, {session()});
        for (int i = 0; i < 300; ++i) CHECK(b.round({0}));
        CHECK(count(f, 2, 2, 0) > 0 && count(f, 3, 4, 0) == 0);  // MTP rounds verified one draft
        CHECK(count(f, 8, 8, 200) >= 95);
        CHECK(exact(f.got));
    }
    // 4. the table and the estimate's settings survive each request's fresh start (fresh() used to drop the table)
    {
        Fake f;
        Batch b(2, 32, f.hooks());
        b.dflash(7, g6rt::SpecLen());
        const DSelTable t = table(TABW);
        b.dsel(dsel(1, t));
        b.kstop(kstop3());
        std::vector<char> seq[2];
        for (int req = 0; req < 2; ++req)
        {
            f.trace.clear();
            b.admit({0}, {session()});
            for (int i = 0; i < 60; ++i) CHECK(b.round({0}));
            for (int j = 1; j <= DSelTable::VX; ++j) CHECK(b.dsel().m.table[j] == t.v[j] && b.dsel().d.table[j] == t.v[j]);
            CHECK(b.dsel().d.draft_round == 6.9 && b.dsel().m.draft_round == 2 && b.dsel().m.draft_each == 1.17);
            CHECK(b.dsel().d.prior == 1 && b.dsel().m.prior == 1);   // w
            for (const auto& x : f.trace) if (std::get<0>(x) == Step::MAIN && std::get<2>(x) >= 0) seq[req].push_back(std::get<1>(x) == 8 ? 'd' : 'm');
            b.cancel(0); b.release(0);
        }
        CHECK(seq[0] == seq[1] && !seq[0].empty());                     // the second request decides as the first did
        DSel off = dsel(0, t);                                          // model off: the constants, the default table
        Batch c(2, 32, f.hooks());
        c.dflash(7, g6rt::SpecLen());
        c.dsel(off);
        CHECK(c.dsel().m.table[3] == g6rt::SpecLen().table[3] && c.dsel().m.draft_round == 2.9 && c.dsel().m.prior == 4);
    }
    // 5. four ranks, each with the table spelled differently (inline, reordered, from a file with a digit past the grid,
    //    canonical): the same steps and the same estimates, bit for bit, every round. MTP verifies two of three drafts,
    //    then DFlash2 drafts six right: with the prior at one round (w) the choice moves to MTP and back
    {
        const std::string path = "batch_dselmodel_test.table";
        { std::ofstream o(path); o << "# from a file\nv8:104\nv4:64\nv3:57.00001\nv2:49.6\nd:6.9 m:2 e:1.17 p:4.9\ng:2 k:3 n:3 w:1 # prior\n"; }
        std::ifstream in(path);
        const std::string file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::remove(path.c_str());
        const DSelTable t0 = table(TABW);
        const std::string spell[4] = {TABW, "w:1,n:3,k:3,g:2,p:4.9,e:1.17,m:2,d:6.9,v8:104,v4:64,v3:57,v2:49.6", file, t0.str()};
        Fake f[4];
        std::vector<Batch> r;
        for (int k = 0; k < 4; ++k)
        {
            f[k].dright = 3; f[k].mtp_depth = 2;   // MTP: two right, the third wrong; the choice moves both ways
            r.emplace_back(2, 32, f[k].hooks());
            r[k].dflash(7, g6rt::SpecLen());
            r[k].dsel(dsel(1, table(spell[k].c_str())));
            r[k].kstop(kstop3());
            r[k].admit({0}, {session()});
            CHECK(r[k].dsel().tab.str() == t0.str());
        }
        bool same = true;
        for (int i = 0; i < 500; ++i)
        {
            if (i == 150) for (auto& x : f) x.dright = 6;          // the text changes: DFlash2 drafts better
            for (int k = 0; k < 4; ++k) CHECK(r[k].round({0}));
            const auto a = r[0].dsel_rates();
            for (int k = 1; k < 4; ++k)
            {
                const auto b = r[k].dsel_rates();
                same &= std::memcmp(&a.first, &b.first, sizeof a.first) == 0 && std::memcmp(&a.second, &b.second, sizeof a.second) == 0;
            }
        }
        CHECK(same);
        for (int k = 1; k < 4; ++k) CHECK(f[k].trace == f[0].trace && r[k].stats(0).switches == r[0].stats(0).switches);
        CHECK(r[0].stats(0).switches >= 2 && r[0].stats(0).mtp_rounds > 20 && count(f[0], 8, 8, 450) > 45);
        for (const auto& x : f) CHECK(exact(x.got));
    }
    // 6. model off with a table given: the steps of a DSel without one (byte-identical to before the model)
    {
        std::vector<std::tuple<int, int, long>> ref;
        for (int arm = 0; arm < 2; ++arm)
        {
            Fake f;
            f.mtp_depth = 2;
            Batch b(2, 32, f.hooks());
            b.dflash(7, g6rt::SpecLen());
            b.dsel(dsel(0, arm ? table(TAB) : DSelTable()));
            b.kstop(kstop3());
            b.admit({0}, {session()});
            for (int i = 0; i < 200; ++i) { CHECK(b.round({0})); if (i == 100) f.dright = 6; }
            if (!arm) ref = f.trace;
            else CHECK(f.trace == ref);
        }
    }
    // 7. model 2: the DFlash2 rounds' own draft count (--spec-adaptive) prices its verify from the table too. With an
    //    8-row verify at 1000 ms it stops verifying seven drafts; model 1 keeps spec_len.h's table and verifies seven
    for (int model : {1, 2})
    {
        Fake f;
        f.dright = 7; f.mtp_depth = 0;              // DFlash2 drafts all right, MTP's first draft wrong: no MTP rounds
        g6rt::SpecLen sl;
        sl.on = true; sl.set_levels({1, 2, 3, 4, 5, 6, 7}); sl.draft_round = 5.6;
        Batch b(2, 32, f.hooks());
        b.dflash(7, sl);
        b.dsel(dsel(model, table("v2:49.6,v3:57,v4:64,v7:96,v8:1000,d:6.9,m:2,e:1.17,p:4.9")));
        b.kstop(kstop3());
        b.admit({0}, {session()});
        for (int i = 0; i < 60; ++i) CHECK(b.round({0}));
        const int full = count(f, 8, 8, 20), seven = count(f, 7, 7, 20);
        if (model == 1) CHECK(full == 40);
        else CHECK(full == 0 && seven == 40);
        CHECK(exact(f.got));
    }
    std::printf("batch_dselmodel_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
