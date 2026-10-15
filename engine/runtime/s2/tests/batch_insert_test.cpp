// The insert rule in Batch (s2_batch.h): a session whose rule fires must, after the inserted tokens, generate what a
// session generates whose prompt already holds them (prompt + reply up to the rule + the inserted tokens), and every
// committed position must hold its token in the fake KV cache. Sessions without a rule must not notice one beside them.
// Host only. The fake model reads its KV cache: a main row writes its token at its position, and samples a hash of the
// cache up to that position, so a row the Batch forgets to feed, or feeds the wrong token, changes what follows.
#include "s2_batch.h"
#include <cstdio>
#include <algorithm>
#include <map>

using namespace s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr int V = 5;                  // a small vocabulary, so the unless token shows up
static uint64_t h64(uint64_t h, int v) { h ^= uint64_t(v) + 0x9E3779B97F4A7C15ull; h *= 0xBF58476D1CE4E5B9ull; return h ^ (h >> 31); }

struct Fake
{
    std::map<int, std::vector<int>> kv, mkv;   // per session: the main and the MTP layer's cache, token by position
    std::map<int, uint64_t> salt;
    long steps = 0;
    static void put(std::vector<int>& c, int pos, int tok) { if ((int)c.size() <= pos) c.resize(pos + 1, -1); c[pos] = tok; }
    uint64_t prefix(int q, int pos) const
    {
        uint64_t h = salt.count(q) ? salt.at(q) : 1;
        const auto& c = kv.at(q);
        for (int p = 0; p <= pos; ++p) h = h64(h, p < (int)c.size() ? c[p] : -7);
        return h;
    }
    Hooks hooks(std::map<int, std::vector<int>>* emitted)
    {
        Hooks h;
        h.run = [this](const Step& s) {
            ++steps;
            if (s.kind == Step::MAIN) for (const Row& r : s.r) put(kv[r.sess], r.pos, r.tok);
            else for (const Row& r : s.r) put(mkv[r.sess], r.pos, r.tok);
            std::vector<int> out;
            for (const Row& r : s.r)
            {
                if (s.kind == Step::MAIN) { out.push_back(int(prefix(r.sess, r.pos) % V)); continue; }
                // a draft: the main model's next token given the cache to pos and the row's token, wrong a quarter of the time
                const uint64_t g = h64(prefix(r.sess, r.pos), r.tok);
                out.push_back(g % 4 == 0 ? int((g >> 8) % V) : int(h64(h64(prefix(r.sess, r.pos), r.tok), 0) % V));
            }
            return out;
        };
        if (emitted) h.emit = [emitted](int q, const int* t, int k, int) { (*emitted)[q].insert((*emitted)[q].end(), t, t + k); return false; };
        h.dfl_start = [](int) { return 0; };
        h.dfl = [](const DflReq& r) { DflDrafts o{}; for (int i = 1; i < DFL_B; ++i) o.tok[i] = (r.tok0 + i) % V; return o; };
        return h;
    }
};

struct Run { std::vector<std::vector<int>> out; std::map<int, std::vector<int>> emitted; Fake f; std::vector<Stats> st; };
// sessions ss in one Batch (DFlash2 sessions alone first, as the driver demotes them when a second arrives)
static Run run(const std::vector<Session>& ss, int K, bool demote_after_first = false)
{
    Run r;
    for (size_t q = 0; q < ss.size(); ++q)   // the prefill path wrote the prompt rows before first
    {
        r.f.salt[(int)q] = 1;
        r.f.kv[(int)q].assign(ss[q].prompt.begin(), ss[q].prompt.begin() + ss[q].first);
    }
    Batch b(K, 32, r.f.hooks(&r.emitted));
    b.dflash(7, g6rt::SpecLen());
    std::vector<int> qs;
    for (size_t q = 0; q < ss.size(); ++q) qs.push_back((int)q);
    if (demote_after_first)
    {
        b.admit({0}, {ss[0]});
        r.f.kv[1].assign(ss[1].prompt.begin(), ss[1].prompt.begin() + ss[1].first);
        b.round({0});
        std::vector<int> rest(qs.begin() + 1, qs.end());
        b.admit(rest, std::vector<Session>(ss.begin() + 1, ss.end()));
        b.demote();
    }
    else b.admit(qs, ss);
    while (b.round(qs)) {}
    for (int q : qs) { r.out.push_back(b.result(q)); r.st.push_back(b.stats(q)); }
    return r;
}

static Session sess(int len, int first, int chunk, int n_gen, uint32_t seed)
{
    Session s;
    for (int i = 0; i < len; ++i) s.prompt.push_back(int(h64(seed, i) % V));
    s.first = first < 0 ? len - 1 : first; s.chunk = chunk; s.n_gen = n_gen;
    return s;
}

// a's tokens after its insert must be a continuation prompt's tokens, and every committed position must hold its token
static void check_session(const Session& a, const std::vector<int>& out, const Fake& f, int q, int K, bool dfl)
{
    const auto& ins = a.ins;
    std::vector<int> reply_before;
    bool fires = ins.on() && ins.after < a.n_gen;
    for (int i = 0; fires && i < ins.after; ++i) if (i >= (int)out.size() || out[i] == ins.unless) fires = false;
    if (!fires)
    {
        Session plain = a; plain.ins = g6rt::InsertRule();
        CHECK(run({plain}, K).out[0] == out);
        return;
    }
    const int m = std::min((int)ins.tokens.size(), a.n_gen - ins.after);
    CHECK((int)out.size() == a.n_gen);
    CHECK(std::equal(ins.tokens.begin(), ins.tokens.begin() + m, out.begin() + ins.after));
    if (ins.after + m < a.n_gen)
    {
        Session c = a; c.ins = g6rt::InsertRule(); c.dflash = dfl;
        c.prompt.insert(c.prompt.end(), out.begin(), out.begin() + ins.after + m);
        c.first = int(c.prompt.size()) - 1;
        c.n_gen = a.n_gen - ins.after - m;
        const auto cont = run({c}, K).out[0];
        CHECK(std::equal(cont.begin(), cont.end(), out.begin() + ins.after + m));
        if (!std::equal(cont.begin(), cont.end(), out.begin() + ins.after + m))
            std::printf("  after %d m %d K %d chunk %d first %d dfl %d\n", ins.after, m, K, a.chunk, a.first, dfl);
    }
    // inserted tokens that end the reply are never fed
    std::vector<int> all = a.prompt;
    all.insert(all.end(), out.begin(), out.begin() + (ins.after + m < a.n_gen ? (int)out.size() : ins.after + 1));
    const auto& c = f.kv.at(q);
    bool fed = true;
    for (int p = a.first; p + 1 < (int)all.size(); ++p) fed &= p < (int)c.size() && c[p] == all[p];
    // and the MTP layer's row at each fed position holds the next token
    const auto& d = f.mkv.at(q);
    for (int p = a.first; p + 1 < (int)all.size(); ++p) fed &= p < (int)d.size() && d[p] == all[p + 1];
    CHECK(fed);
    if (!fed) std::printf("  fed: after %d m %d K %d chunk %d first %d dfl %d\n", ins.after, m, K, a.chunk, a.first, dfl);
}

int main()
{
    long fired = 0, cases = 0;
    for (int K = 1; K <= 3; ++K)
        for (int chunk : {1, 2, 4})
            for (int after = 0; after <= 14; ++after)
                for (int m : {1, 3, 5})
                {
                    Session a = sess(9 + after % 4, after % 3 == 0 ? 0 : -1, chunk, 18, 1000 + after * 7 + m);
                    a.ins.after = after;
                    a.ins.unless = after % 5 == 4 ? 2 : -1;      // sometimes a token that may show up first
                    for (int i = 0; i < m; ++i) a.ins.tokens.push_back(V + 10 + i);
                    const Run r = run({a}, K);
                    ++cases;
                    fired += r.st[0].inserted > 0;
                    check_session(a, r.out[0], r.f, 0, K, false);
                    CHECK(r.emitted.at(0) == std::vector<int>(r.out[0].begin(), r.out[0].end()) ||
                          std::equal(r.out[0].begin(), r.out[0].end(), r.emitted.at(0).begin()));
                    // beside two sessions without a rule: each of them as alone, the rule's session as alone
                    Session b = sess(13, -1, chunk, 21, 77 + after), c = sess(6, 0, chunk, 9, 99 + m);
                    const Run three = run({b, a, c}, K);
                    CHECK(three.out[0] == run({b}, K).out[0]);
                    CHECK(three.out[1] == r.out[0]);
                    CHECK(three.out[2] == run({c}, K).out[0]);
                    // a DFlash2 session alone, and one demoted when a second session arrives
                    Session d = a; d.dflash = true;
                    const Run rd = run({d}, K);
                    check_session(d, rd.out[0], rd.f, 0, K, true);
                    const Run dm = run({d, b}, K, true);
                    CHECK(dm.out[0] == rd.out[0]);
                    CHECK(dm.out[1] == run({b}, K).out[0]);
                }
    // the rule's cut is the n_gen-th token: the inserted tokens are cut to n_gen and nothing follows
    {
        Session a = sess(8, -1, 4, 6, 5);
        a.ins.after = 4; a.ins.tokens = {20, 21, 22};
        const Run r = run({a}, 2);
        CHECK((int)r.out[0].size() == 6 && r.out[0][4] == 20 && r.out[0][5] == 21);
    }
    // the token at the rule's place is the unless token itself: it is replaced all the same (TensorFold's budget)
    for (int K = 1; K <= 3; ++K)
    {
        int found = 0;
        for (uint32_t seed = 1; seed < 400 && found < 3; ++seed)
        {
            Session a = sess(10, -1, 4, 16, seed);
            const auto plain = run({a}, K).out[0];
            const int after = 2 + int(seed % 5);
            if (std::find(plain.begin(), plain.begin() + after, plain[after]) != plain.begin() + after) continue;
            ++found;
            a.ins.after = after; a.ins.unless = plain[after]; a.ins.tokens = {30, 31};
            const auto out = run({a}, K).out[0];
            CHECK(out[after] == 30 && out[after + 1] == 31);
            a.ins.unless = plain[after - 1];               // one of the first `after`: the rule never fires
            CHECK(run({a}, K).out[0] == plain);
        }
        CHECK(found == 3);
    }
    CHECK(fired > cases / 2);
    std::printf("batch_insert_test: %s (%ld cases, %ld with the insert)\n", fails ? "FAIL" : "PASS", cases, fired);
    return fails ? 1 : 0;
}
