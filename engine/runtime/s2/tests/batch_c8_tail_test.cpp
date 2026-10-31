// One packed prompt-tail MAIN, then the batched DRAFT and CHAIN.
// Host-only fake engine: each slot's sampled value depends on its own row, not batch position.
#include "s2_batch.h"
#include <cstdio>
#include <vector>

using namespace s2;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)
template<class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static Session session(int n) {
    Session s; s.prompt.resize(size_t(n));
    for (int i = 0; i < n; ++i) s.prompt[size_t(i)] = 10 + i;
    s.first = n - 1; s.chunk = 4; s.n_gen = 16;
    return s;
}
struct Recorder {
    std::vector<Step> steps;
    std::vector<int> emitted = std::vector<int>(8, 0), prompted = std::vector<int>(8, 0);
    int finish = -1;
    Hooks hooks() {
        Hooks h;
        h.run = [this](const Step& s) {
            steps.push_back(s);
            std::vector<int> t;
            for (const auto& r : s.r) t.push_back(10000 + r.sess * 100 + r.pos);
            return t;
        };
        h.prompt = [this](int q, int m, int pos) { CHECK(m == 1 && pos >= 1); ++prompted[size_t(q)]; };
        h.emit = [this](int q, const int* ids, int n, int row) {
            CHECK(n >= 1 && ids && row >= 0 && row < 8);
            emitted[size_t(q)] += n;
            return q == finish;
        };
        return h;
    }
};
static void full_pack(int K) {
    const int lens[8] = {56,58,59,59,59,61,62,65};
    std::vector<int> qs; std::vector<Session> ss;
    for (int i = 0; i < 8; ++i) { qs.push_back(i); ss.push_back(session(lens[i])); }
    Recorder rec; Batch b(K, 48, rec.hooks());
    b.admit_packed_tail(qs, ss);
    CHECK(rec.steps.size() == size_t(K ? K + 1 : 1));
    for (size_t i = 0; i < rec.steps.size(); ++i) {
        const auto& s = rec.steps[i];
        CHECK(s.kind == (i == 0 ? Step::MAIN : i == 1 ? Step::DRAFT : Step::CHAIN));
        CHECK(s.M() == 8 && s.sess.size() == 8 && s.cls == row_class(8));
        for (int q = 0; q < 8; ++q) {
            CHECK(s.sess[size_t(q)] == q && s.row0[size_t(q)] == q && s.rows[size_t(q)] == 1);
            CHECK(s.ev[size_t(q)] == Step::TAIL);
        }
    }
    for (int q = 0; q < 8; ++q) {
        const auto r = b.result(q);
        CHECK(r.size() == 1 && r[0] == 10000 + q * 100 + lens[q] - 1);
        CHECK(rec.prompted[size_t(q)] == 1 && rec.emitted[size_t(q)] == 1 && b.active(q));
    }
    std::printf("batch_c8_tail_test: K=%d, one 8-row MAIN%s, %d shared CHAINs, eight independent outputs\n",
                K, K ? "/DRAFT" : "", K ? K - 1 : 0);
}
static void stop_and_insert() {
    std::vector<int> qs{0,1,2}; std::vector<Session> ss{session(39),session(32),session(65)};
    ss[0].ins.after = 0; ss[0].ins.tokens = {42,43};
    Recorder rec; rec.finish = 2;
    Batch b(5, 48, rec.hooks()); b.admit_packed_tail(qs, ss);
    CHECK(rec.steps.size() == 6);
    CHECK(rec.steps[0].kind == Step::MAIN && rec.steps[0].M() == 3);
    CHECK(rec.steps[1].kind == Step::DRAFT && rec.steps[1].M() == 2); // insert-forced still gets first DRAFT
    for (size_t i = 2; i < rec.steps.size(); ++i)
        CHECK(rec.steps[i].kind == Step::CHAIN && rec.steps[i].M() == 1 && rec.steps[i].sess[0] == 1);
    CHECK(b.result(0) == std::vector<int>({42,43}) && b.stats(0).inserted == 2);
    CHECK(b.finished(2) && !b.active(2));
    CHECK(rec.emitted[0] == 2 && rec.emitted[1] == 1 && rec.emitted[2] == 1);
    std::printf("batch_c8_tail_test: stopped session omits DRAFT/CHAIN; insert-forced session drafts once but omits CHAIN\n");
}
static void refusals() {
    Recorder rec;
    const std::vector<int> qs{0,1}; const std::vector<Session> good{session(39),session(32)};
    CHECK(throws([&] { Batch b(5,48,rec.hooks()); b.admit_packed_tail({0}, {good[0]}); }));
    CHECK(throws([&] { Batch b(5,48,rec.hooks()); b.admit_packed_tail({0,0}, good); }));
    CHECK(throws([&] { Batch b(5,48,rec.hooks()); auto bad=good; bad[1].first=0; b.admit_packed_tail(qs,bad); }));
    CHECK(throws([&] { Batch b(5,48,rec.hooks()); auto bad=good; bad[1].dflash=true; b.admit_packed_tail(qs,bad); }));
    CHECK(throws([&] { Batch b(5,1,rec.hooks()); b.admit_packed_tail(qs,good); }));
    CHECK(rec.steps.empty()); // prevalidation rejected every bad shape before any engine callback
    std::printf("batch_c8_tail_test: invalid count/slot/first/drafter/width all refused before launch\n");
}
int main() {
    full_pack(0); full_pack(5); stop_and_insert(); refusals();
    if (fails) std::fprintf(stderr, "batch_c8_tail_test: %d failures\n", fails);
    else std::printf("batch_c8_tail_test: PASS\n");
    return fails ? 1 : 0;
}
