// GR8_DFL_REUSE (s2_batch.h Batch::dfl_reuse): a DFlash2 session whose prefill a prefix-reuse hit cut into the drafter's
// window keeps the drafter, with the ring's rows of the last DFlash2 session when its tokens are this prompt's; a short
// context warms up on MTP rounds; mode 2 promotes a session alone on MTP. Mode 0 plans the steps as before. Every
// committed token is the target's. With GR8_MTP_KSTOP (Batch::kstop) the warm-up and promoted MTP rounds verify the
// confidence stop's 1..kmax drafts while the drafter's context stays contiguous. Host only, a fake engine.
#include "s2_batch.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1

struct Fake
{
    std::vector<std::string> log;     // steps and drafter calls, in order
    std::vector<DflReq> reqs;         // drafter calls
    std::vector<char> rounds;         // per decode round: 'm' (MTP verify of K + 1 = 3 rows) or 'd' (8 rows)
    std::vector<int> got;             // committed tokens
    int start = 0;                    // what dfl_start returns (the fill's first position)
    int starts = 0;
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            std::vector<int> out;
            for (const Row& r : s.r) out.push_back(target(r.spos - 1));
            if (s.kind == Step::MAIN && s.M() > 1) rounds.push_back(s.M() == 8 ? 'd' : 'm');
            log.push_back(std::string(s.kind == Step::MAIN ? "m" : s.kind == Step::DRAFT ? "d" : "c") + std::to_string(s.M()) + "@" +
                          std::to_string(s.r[0].pos));
            return out;
        };
        h.dfl_start = [this](int) { ++starts; return start; };
        h.dfl = [this](const DflReq& r) {
            reqs.push_back(r);
            log.push_back(std::string(r.query ? "Q" : "C") + std::to_string(r.rows) + "@" + std::to_string(r.pos0) + "lo" + std::to_string(r.lo));
            DflDrafts o{};
            const int q0 = r.pos0 + r.rows;
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = target(q0 + i - 1); o.lp[i] = -0.1f; }
            return o;
        };
        h.emit = [this](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };
        return h;
    }
    void clear() { log.clear(); reqs.clear(); rounds.clear(); got.clear(); }
};

static Session make(const std::vector<int>& prompt, int n_gen, bool dflash)
{
    Session s;
    s.prompt = prompt; s.first = (int)prompt.size() - 1; s.chunk = 8; s.n_gen = n_gen; s.dflash = dflash;
    return s;
}
// every committed token of a session with prompt size n is the target's
static bool exact(const std::vector<int>& got, int n)
{
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) ok &= got[i] == target(n - 1 + (int)i);
    return ok;
}
static void run(Batch& b, int q) { while (b.round({q})) {} }

// GR8_MTP_KSTOP with the reuse: MTP drafts right where good(), with the MTP head's probability p() (batch_kstop_test's)
static bool good(int spos) { return (spos * 2654435761u >> 7) % 4 != 0; }
static float prob(int spos) { return !good(spos) ? 0.15f : spos % 4 ? 0.97f : 0.8f; }
struct FakeK
{
    std::vector<DflReq> reqs;
    std::vector<int> got;
    std::vector<float> lp;            // the last draft or chain step's rows
    int start = 0, nchain = 0;
    KStop ks;
    int fixed_nd = 0;                 // > 0: the next MTP verify's drafts were chained by the shared path (nd fixed)
    int bad = 0, mtp = 0, dfl = 0, by[9] = {};
    // KStop mode 1 on prob(): (drafts chained, drafts verified) for a round at base; nd > 0: the drafts already held
    std::pair<int, int> rule(int base, int nd = 0) const
    {
        auto P = [&](int j) { double p = 1; for (int i = 1; i <= j; ++i) p *= std::exp((double)std::log(prob(base + i))); return p; };
        if (!nd)
        {
            nd = 1;
            for (;;)
            {
                if (nd >= ks.kmax) break;
                bool ok = true;
                for (int j = 1; j <= nd; ++j) ok &= P(j) >= ks.tau[j];
                if (!ok || P(nd) < ks.tau[nd + 1]) break;
                ++nd;
            }
        }
        int k = 0;
        while (k < nd && P(k + 1) >= ks.tau[k + 1]) ++k;
        return {nd, std::max(k, 1)};
    }
    Hooks hooks(bool solo_rounds_only)
    {
        Hooks h;
        h.run = [this, solo_rounds_only](const Step& s) {
            std::vector<int> out;
            if (s.kind != Step::MAIN) lp.clear();
            if (s.kind == Step::CHAIN) ++nchain;
            if (s.kind == Step::MAIN && s.round >= 0 && (!solo_rounds_only || s.sess.size() == 1))
            {
                if (s.M() == 8) ++dfl;
                else if (solo_rounds_only)
                {   // a solo MTP verify: the rule's k after the rule's chain steps (promoted: K drafts held)
                    const auto r = rule(s.r[0].pos, fixed_nd);
                    bad += s.M() - 1 != r.second || (!fixed_nd && nchain != r.first - 1);
                    fixed_nd = 0;
                    ++mtp; ++by[s.M()];
                }
            }
            if (s.kind == Step::MAIN) nchain = 0;
            for (const Row& r : s.r)
            {
                if (s.kind == Step::MAIN) out.push_back(target(r.spos - 1));
                else { out.push_back(good(r.spos) ? target(r.spos - 1) : -2); lp.push_back(std::log(prob(r.spos))); }
            }
            return out;
        };
        h.dfl_start = [this](int) { return start; };
        h.dfl = [this](const DflReq& r) {
            reqs.push_back(r);
            DflDrafts o{};
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = target(r.pos0 + r.rows + i - 1); o.lp[i] = -0.1f; }
            return o;
        };
        h.emit = [this](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };
        h.lp = [this](int r) { return lp.at(r); };
        return h;
    }
    // the drafter's context rows from `from` on, each call right after the last (the ring tracker's input)
    bool contiguous(size_t from) const
    {
        bool ok = true;
        for (size_t i = from + 1; i < reqs.size(); ++i) ok &= reqs[i].pos0 == reqs[i - 1].pos0 + reqs[i - 1].rows;
        return ok;
    }
};

int main()
{
    std::vector<int> p1(5000);
    for (int i = 0; i < 5000; ++i) p1[i] = (i * 13 + 7) % 991;
    const int n1 = 5000, w1 = n1 - 1 - DFL_WIN;
    std::vector<std::string> base_log;      // turn 2 with the reuse off (mode 0), for the comparison below
    for (int mode : {0, 1})
    {
        Fake f;
        Batch b(2, 32, f.hooks());
        b.dflash(7, g6rt::SpecLen());
        DflReuse r; r.mode = mode;
        b.dfl_reuse(r);
        // turn 1: a fresh prompt, the fill covers the window
        f.start = w1;
        b.admit({0}, {make(p1, 60, true)});
        run(b, 0);
        CHECK(exact(f.got, n1) && !f.reqs.empty() && f.reqs[0].lo == w1);
        CHECK(std::count(f.rounds.begin(), f.rounds.end(), 'm') == 0);
        if (mode) CHECK(b.stats(0).dfl_from == w1 && b.stats(0).dfl_lo == w1 && b.stats(0).dfl_warm == 0);
        else CHECK(b.stats(0).dfl_from == -1);
        // turn 2 on the same slot: turn 1's prompt + reply + 300 new tokens; the hit skipped the first 4096 rows, into the window
        std::vector<int> p2 = p1;
        const std::vector<int> r1 = b.result(0);
        p2.insert(p2.end(), r1.begin(), r1.end());
        for (int i = 0; i < 300; ++i) p2.push_back((i * 7 + 3) % 977);
        const int n2 = (int)p2.size(), w2 = n2 - 1 - DFL_WIN;
        b.release(0);
        f.clear();
        f.start = 4096;
        b.admit({0}, {make(p2, 40, true)});
        run(b, 0);
        CHECK(exact(f.got, n2));
        // mode 1: the drafter keeps turn 1's ring rows [w2, 4096), so the context is the whole window; mode 0: from 4096
        CHECK(!f.reqs.empty() && f.reqs[0].lo == (mode ? w2 : 4096));
        CHECK(std::count(f.rounds.begin(), f.rounds.end(), 'm') == 0);
        if (mode) CHECK(b.stats(0).dfl_from == 4096 && b.stats(0).dfl_lo == w2 && b.stats(0).dfl_rounds == b.stats(0).rounds);
        if (!mode) base_log = f.log;
        else
        {   // the same steps as mode 0; only the drafter's lo differs
            CHECK(f.log.size() == base_log.size());
            for (size_t i = 0; i < std::min(f.log.size(), base_log.size()); ++i)
            {
                const std::string a = f.log[i].substr(0, f.log[i].find("lo")), c = base_log[i].substr(0, base_log[i].find("lo"));
                CHECK(a == c);
            }
        }
        if (!mode) continue;
        // turn 3 on slot 1, another conversation's prompt (differs at 100): the ring is not its context. The hit skipped to
        // 8192 of a 8300-token prompt: 108 rows < 256, so MTP rounds until the drafter holds 256 rows, then DFlash2
        std::vector<int> p3(8300);
        for (int i = 0; i < 8300; ++i) p3[i] = i < 100 ? p1[i] : (i * 11 + 5) % 983;
        b.release(0);
        f.clear();
        f.start = 8192;
        b.admit({1}, {make(p3, 400, true)});
        CHECK(!f.reqs.empty() && !f.reqs[0].query && f.reqs[0].lo == 8192);   // the tail step: context only, no probe
        {   // the warm-up starts on MTP: admit chained d[2..K] for its first round, which verifies K + 1 = 3 rows
            size_t j = f.log.size();
            while (j > 0 && (f.log[j - 1][0] == 'C' || f.log[j - 1][0] == 'Q')) --j;
            CHECK(j > 0 && f.log[j - 1][0] == 'c');
        }
        CHECK(b.round({1}) && f.log.size() > 1 && f.rounds.size() == 1 && f.rounds[0] == 'm');
        run(b, 1);
        CHECK(exact(f.got, 8300));
        CHECK(b.stats(1).dfl_lo == 8192 && b.stats(1).dfl_warm == 8192 + 256 && b.stats(1).mtp_rounds > 0 && b.stats(1).dfl_rounds > 0);
        CHECK(!f.rounds.empty() && f.rounds[0] == 'm' && f.rounds.back() == 'd');
        {   // the switch: the first DFlash2 query comes once the context reaches 8192 + 256
            const auto it = std::find_if(f.reqs.begin(), f.reqs.end(), [](const DflReq& x) { return x.query; });
            CHECK(it != f.reqs.end() && it->pos0 + it->rows >= 8192 + 256 && it->pos0 < 8192 + 256);
            bool lo = true;
            for (const DflReq& x : f.reqs) lo &= x.lo == 8192;
            CHECK(lo);
        }
        // turn 4 on slot 0: turn 3's conversation continued (slot 1's tokens are still the ring's), a hit to 8192 of
        // 8800 + 300: the ring holds [8192 .. ) only from turn 3's own start, so the context starts at 8192 (no warm-up)
        std::vector<int> p4 = p3;
        const std::vector<int> r3 = b.result(1);
        p4.insert(p4.end(), r3.begin(), r3.end());
        for (int i = 0; i < 300; ++i) p4.push_back((i * 3 + 1) % 971);
        f.clear();
        f.start = 8192;
        b.admit({0}, {make(p4, 20, true)});
        CHECK(!f.reqs.empty() && f.reqs[0].lo == 8192 && b.stats(0).dfl_warm == 0);
        run(b, 0);
        CHECK(exact(f.got, (int)p4.size()));
    }
    {   // mode 2: a session admitted next to another (MTP only) takes the drafter when promoted; mode 1 does not promote
        for (int mode : {1, 2})
        {
            Fake f;
            Batch b(2, 32, f.hooks());
            b.dflash(7, g6rt::SpecLen());
            DSel sel;
            sel.on = true;
            sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
            b.dsel(sel);
            DflReuse r; r.mode = mode; r.min = 64;
            b.dfl_reuse(r);
            b.admit({0}, {make(p1, 600, false)});
            for (int i = 0; i < 5; ++i) CHECK(b.round({0}));
            CHECK(f.reqs.empty() && f.rounds.size() == 5 && b.stats(0).acc.size() == 3);
            const int at = b.length(0) - 1;
            b.promote(0);
            run(b, 0);
            CHECK(exact(f.got, n1));
            if (mode == 1) { CHECK(f.reqs.empty() && b.stats(0).promoted == -1); continue; }
            CHECK(b.stats(0).promoted == at && b.stats(0).dfl_warm == at + 64 && b.stats(0).acc.size() == 8);
            CHECK(!f.reqs.empty() && f.reqs[0].pos0 == at && !f.reqs[0].query);
            bool lo = true;
            for (const DflReq& x : f.reqs) lo &= x.lo == at;
            CHECK(lo);
            CHECK(f.rounds.back() == 'd' && b.stats(0).dfl_rounds > 0);
        }
    }
    {   // GR8_MTP_KSTOP=1 (kmax 3, K 2) with the reuse. Warm-up: the MTP rounds verify the rule's 1..3 drafts, the drafter
        // takes each round's accepted rows as context (contiguous), DFlash2 starts once it holds GR8_DFL_REUSE_MIN rows.
        // The next turn's hit finds those rows in the ring
        std::vector<int> p3(8300);
        for (int i = 0; i < 8300; ++i) p3[i] = (i * 11 + 5) % 983;
        FakeK f;
        f.ks.mode = 1; f.ks.kmax = 3; f.ks.tau[2] = 0.3; f.ks.tau[3] = 0.8;
        Batch b(2, 32, f.hooks(true));
        b.dflash(7, g6rt::SpecLen());
        DSel sel;
        sel.on = true;
        sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
        b.dsel(sel);
        b.kstop(f.ks);
        DflReuse r; r.mode = 1; r.min = 256;
        b.dfl_reuse(r);
        f.start = 8192;
        b.admit({1}, {make(p3, 400, true)});
        run(b, 1);
        CHECK(exact(f.got, 8300));
        CHECK(f.bad == 0 && f.mtp > 20 && f.dfl > 0 && b.stats(1).mtp_rounds == f.mtp && b.stats(1).dfl_warm == 8192 + 256);
        CHECK(f.by[2] > 0 && f.by[3] > 0 && f.by[4] > 0);   // every draft count during the warm-up
        CHECK(f.contiguous(0));
        {
            const auto it = std::find_if(f.reqs.begin(), f.reqs.end(), [](const DflReq& x) { return x.query; });
            CHECK(it != f.reqs.end() && it->pos0 < 8192 + 256 && it->pos0 + it->rows >= 8192 + 256);
        }
        // turn 2 on slot 0, turn 1 continued: a hit to 8600 (past the warm-up); the ring holds [8192, end) from turn 1's
        // K-stop rounds, so the context starts at 8192 with no warm-up
        std::vector<int> p4 = p3;
        const std::vector<int> r3 = b.result(1);
        p4.insert(p4.end(), r3.begin(), r3.end());
        for (int i = 0; i < 300; ++i) p4.push_back((i * 3 + 1) % 971);
        f.reqs.clear(); f.got.clear();
        f.start = 8600;
        b.admit({0}, {make(p4, 60, true)});
        CHECK(!f.reqs.empty() && f.reqs[0].lo == 8192 && b.stats(0).dfl_lo == 8192 && b.stats(0).dfl_warm == 0);
        run(b, 0);
        CHECK(exact(f.got, (int)p4.size()) && f.bad == 0);
    }
    for (int pre = 3; pre <= 12; ++pre)
    {   // GR8_DFL_REUSE=2 promotes a shared-path session after `pre` rounds: its first MTP verify takes the rule's k of the K
        // drafts the shared chain left (their log-probabilities from that round's steps), the later ones the rule's chain
        std::vector<int> p1(5000);
        for (int i = 0; i < 5000; ++i) p1[i] = (i * 13 + 7) % 991;
        FakeK f;
        f.ks.mode = 1; f.ks.kmax = 3; f.ks.tau[2] = 0.3; f.ks.tau[3] = 0.8;
        bool solo = false;
        Hooks h = f.hooks(true);
        auto run0 = h.run;
        h.run = [&](const Step& s) { return solo ? run0(s) : f.hooks(false).run(s); };
        Batch b(2, 32, h);
        b.dflash(7, g6rt::SpecLen());
        DSel sel;
        sel.on = true;
        sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
        b.dsel(sel);
        b.kstop(f.ks);
        DflReuse r; r.mode = 2; r.min = 64;
        b.dfl_reuse(r);
        b.admit({0}, {make(p1, 600, false)});
        for (int i = 0; i < pre; ++i) CHECK(b.round({0}));
        const int at = b.length(0) - 1;
        b.promote(0);
        solo = true; f.fixed_nd = 2; f.nchain = 0;
        run(b, 0);
        CHECK(exact(f.got, 5000));
        CHECK(f.bad == 0 && f.mtp > 0 && f.dfl > 0 && b.stats(0).promoted == at && b.stats(0).dfl_warm == at + 64);
        CHECK(!f.reqs.empty() && f.reqs[0].pos0 == at && f.contiguous(0));
    }
    std::printf("batch_dflreuse_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
