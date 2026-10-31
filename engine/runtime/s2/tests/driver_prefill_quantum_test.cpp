// The prefill quantum through Driver::step (driver.h) and the scheduler (scheduler.h): what the DS41 first-light prefill
// config (prefill_rows = W = 1024, one session at a time) asks of the host side, with no engine and no pool. Host only.
//  1. a prompt's prefill hook calls are contiguous from row 0, none above the quantum cap, they sum to prompt - 1, the
//     prompt tail is ONE row at position prompt - 1 and is the session's first MAIN row (no prefill chunk needs the
//     head: the last token runs as a MAIN step), and the tokens are the solo run's, and the decode_prompt run's
//  2. a decoding session beside a prefilling one: Prefill quanta take at most decode_prefill_rows, and decode_rounds
//     rounds run per quantum (D P D P ..., D D D P ...), every output the solo run's
//  3. replicated ranks agree on every quantum and digest; a rank with another prefill_rows or two_lanes, or one that
//     skipped the prefill (decode_prompt), is refused at the next apply
//  4. a cancel between quanta: retire runs once, no further prefill call, the other session keeps decoding, then Idle
//  5. the first-light config (max_sessions 1): limit() == 1, a second admission is refused; the notice horizon
// The fake model is a pure function of (request, row position, row token), as driver_grow_test's.
#include "s2/driver.h"
#include <cstdio>
#include <map>
#include <memory>
#include <string>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 0x1234567ull) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
constexpr int V = 2;
static int model(const s2::Row& r, int kind, uint32_t salt)
{
    int t = (int)(mix(salt, kind == s2::Step::MAIN ? r.pos : r.pos + 1, r.tok) % V);
    if (kind != s2::Step::MAIN && mix(salt, r.pos, r.tok + 99) % 3 == 0) t = (t + 1) % V;
    return t;
}
struct Req { uint64_t id; std::vector<int> prompt; uint32_t max_tokens; };
static Req req(uint64_t id, int n, uint32_t max_tokens)
{
    Req r{id, {}, max_tokens};
    for (int k = 0; k < n; ++k) r.prompt.push_back(int(mix(uint32_t(id), uint32_t(k), 7) % V));
    return r;
}

// the solo tokens of a request: `first` prompt rows taken by a prefill (n - 1), or none (the decode_prompt run: 0)
static std::vector<int> solo(const Req& r, int K, int first)
{
    s2::Hooks h;
    h.run = [&](const s2::Step& st) {
        std::vector<int> ids;
        for (const auto& x : st.r) ids.push_back(model(x, st.kind, uint32_t(r.id)));
        return ids;
    };
    s2::Session s;
    s.prompt = r.prompt; s.first = first; s.chunk = 4; s.n_gen = int(r.max_tokens);
    std::vector<s2::Stats> st;
    return s2::mtp_batch({s}, K, 8, h, st)[0];
}

struct Pf { uint64_t id; uint32_t first, rows; size_t prompt; };

// one rank: a scheduler/driver and what its hooks saw
struct Rank {
    Config c;
    std::unique_ptr<Driver> d;
    std::map<uint64_t, std::vector<int>> out;   // request id -> emitted tokens
    std::map<uint64_t, int> first_main;         // request id -> position of the first MAIN row it ran
    std::vector<Pf> pf;                         // every prefill hook call
    std::vector<Quantum> q;                     // every quantum step() returned
    std::vector<uint64_t> retired;              // request ids the retire hook saw
    long outside = 0;
    uint32_t next_page = 0;
    explicit Rank(const Config& cfg) : c(cfg) {
        DriverHooks h;
        h.bind = [](const Grant&) {};
        h.retire = [this](const Slot& s) { retired.push_back(s.req.id); };
        h.prefill = [this](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
            pf.push_back({s.req.id, first, rows, p.prompt.size()});
            if (uint64_t(first) + rows > uint64_t(s.run.pages) * PAGE) ++outside;
        };
        h.run = [this](const s2::Step& st) {
            std::vector<int> ids;
            for (const auto& r : st.r) {
                const Slot& s = d->state().slots()[r.sess];
                if (uint64_t(r.pos) >= uint64_t(s.run.pages) * PAGE) ++outside;   // a row past the session's pages
                if (st.kind == s2::Step::MAIN && !first_main.count(s.req.id)) first_main[s.req.id] = r.pos;
                ids.push_back(model(r, st.kind, uint32_t(s.req.id)));
            }
            return ids;
        };
        h.emit = [this](Key k, const int* ids, int n, int) {
            const Slot& s = d->state().slots()[k.slot];
            auto& o = out[s.req.id];
            const int cnt = std::min<int>(n, int(s.req.max_tokens - o.size()));
            o.insert(o.end(), ids, ids + cnt);
            return Emission{uint32_t(cnt), o.size() == s.req.max_tokens};
        };
        d = std::make_unique<Driver>(c, h);
    }
    const Scheduler& s() const { return d->state(); }
    std::vector<uint32_t> rows_of(uint64_t id) const {
        std::vector<uint32_t> v;
        for (const auto& x : pf) if (x.id == id) v.push_back(x.rows);
        return v;
    }
};

static Config cfg(uint32_t mtp, uint32_t prefill_rows, bool two_lanes, uint32_t dpr = 0, uint32_t rounds = 0, uint32_t max_sessions = 4)
{
    Config c;
    c.epoch = 1; c.mtp = mtp; c.max_rows = 24; c.max_sessions = max_sessions;
    c.prefill_rows = prefill_rows; c.two_lanes = two_lanes; c.decode_prefill_rows = dpr; c.decode_rounds = rounds;
    return c;
}

// the FIFO head on a run of its worst-case pages (so no growth hook is needed); the same grant goes to every rank
static Grant grant_for(const Rank& head, const Req& r, uint32_t& page)
{
    const Request rq{r.id, uint32_t(r.prompt.size()), r.max_tokens};
    Run run;
    run.first_page = page; run.pages = reserve_pages(head.c, rq); run.table_version = 1; run.list_hash = mix(uint32_t(r.id), page, 3);
    page += run.pages;
    return Grant{head.s().next_key(), rq, run, 0};
}
static Commit commit_for(const Scheduler& s)
{
    Commit cm;
    cm.epoch = s.config().epoch; cm.event = s.next_event(); cm.boundary = s.boundary(); cm.before_hash = s.digest();
    return cm;
}
static Payload payload_of(const Req& r, bool decode_prompt = false)
{
    Payload p; p.prompt = r.prompt; p.chunk = 4; p.decode_prompt = decode_prompt;
    return p;
}
static Key admit(Rank& r, const Req& q, bool decode_prompt = false)
{
    Commit cm = commit_for(r.s());
    cm.admit = grant_for(r, q, r.next_page);
    const Payload p = payload_of(q, decode_prompt);
    r.d->apply(cm, &p);
    return cm.admit->key;
}
static Quantum step(Rank& r)
{
    const Quantum q = r.d->step();
    r.q.push_back(q);
    const uint64_t h = r.s().horizon();   // a notice target is always inside [boundary + NOTICE_MIN, boundary + NOTICE_MAX]
    CHECK(h >= r.s().boundary() + NOTICE_MIN && h <= r.s().boundary() + NOTICE_MAX);
    return q;
}
static void drain(Rank& r)
{
    for (long g = 0; r.s().resident(); ++g) {
        const Quantum q = step(r);
        CHECK(q.kind != Kind::Idle);          // a resident session always has something to run
        if (q.kind == Kind::Idle || g > 200000) { CHECK(!"no progress"); fails += 1000; break; }
    }
}
static char letter(Kind k) { return k == Kind::Prefill ? 'P' : k == Kind::PromptTail ? 'T' : k == Kind::Decode ? 'D' : 'I'; }
static std::string kinds(const std::vector<Quantum>& v, size_t from = 0)
{
    std::string s;
    for (size_t i = from; i < v.size(); ++i) s += letter(v[i].kind);
    return s;
}
// rows the prefill takes per quantum for a prompt of n with nothing else running: scheduler.h plan()
static std::vector<uint32_t> expect_rows(uint32_t n, uint32_t W, bool two)
{
    std::vector<uint32_t> v;
    for (uint32_t left = n - 1; left;) {
        const uint32_t cap = W * (two && left > W ? 2 : 1), r = std::min(left, cap);
        v.push_back(r); left -= r;
    }
    return v;
}
static std::string repeat(const std::string& s, size_t n) { std::string o; while (n--) o += s; return o; }

// 1. one prompt alone: the hook calls, the tail, the first MAIN row, the tokens
static void check_alone(uint32_t mtp, uint32_t N, uint32_t W, bool two, long& cases)
{
    const Req r = req(7, int(N), 30);
    Rank a(cfg(mtp, W, two));
    admit(a, r);
    drain(a);
    ++cases;
    const auto want = expect_rows(N, W, two);
    CHECK(a.rows_of(r.id) == want);
    uint32_t next = 0;
    for (const auto& x : a.pf) {
        CHECK(x.id == r.id && x.first == next && x.prompt == N);   // contiguous from row 0, called with its own payload
        CHECK(x.rows <= W * (two ? 2 : 1));
        next += x.rows;
    }
    CHECK(next == N - 1);
    // quanta: the prefill ones, ONE tail row at N - 1, then decode rounds only
    CHECK(a.q.size() > want.size() + 1);
    size_t tails = 0;
    for (size_t i = 0; i < a.q.size(); ++i) {
        const Quantum& q = a.q[i];
        if (i < want.size()) CHECK(q.kind == Kind::Prefill && q.rows == want[i]);
        else if (i == want.size()) { CHECK(q.kind == Kind::PromptTail && q.first == N - 1 && q.rows == 1); ++tails; }
        else CHECK(q.kind == Kind::Decode);
    }
    CHECK(tails == 1);
    CHECK(a.first_main[r.id] == int(N) - 1);                     // no prefill row needs the head: the tail is the first MAIN row
    CHECK(a.out[r.id] == solo(r, int(mtp), int(N) - 1));
    CHECK(a.outside == 0);
    // the same request with no tensor prefill: every prompt row a decode step, the same tokens
    Rank b(cfg(mtp, W, two));
    admit(b, r, true);
    drain(b);
    CHECK(b.pf.empty() && b.first_main[r.id] == 0);
    CHECK(b.out[r.id] == a.out[r.id]);
    CHECK(b.out[r.id] == solo(r, int(mtp), 0));
}

// 2. A decodes while B prefills: the quantum sequence, the cap, the tokens
static void check_interleave(uint32_t mtp, uint32_t dpr, uint32_t rounds, long& cases)
{
    const uint32_t W = 128, NB = 1000;
    const Req A = req(1, 20, 600), B = req(2, int(NB), 40);
    Rank r(cfg(mtp, W, false, dpr, rounds));
    admit(r, A);
    while (r.s().slots()[0].phase != Phase::Decode) step(r);
    CHECK(r.q.back().kind == Kind::PromptTail);
    const size_t from = r.q.size();
    admit(r, B);
    while (r.s().slots()[1].phase != Phase::Decode) step(r);
    ++cases;
    // while A decodes, a prefill quantum takes at most decode_prefill_rows (0 = off: prefill_rows), one lane
    const uint32_t cap = dpr ? dpr : W, nP = (NB - 1 + cap - 1) / cap, D = rounds > 1 ? rounds : 1;
    const auto rows = r.rows_of(B.id);
    CHECK(rows.size() == nP);
    uint32_t sum = 0;
    for (uint32_t x : rows) { CHECK(x <= cap); sum += x; }
    CHECK(sum == NB - 1 && rows.back() == (NB - 1) % cap);
    // D D D P ... : decode_rounds rounds before each quantum of the prefill, and before the tail
    const std::string seq = kinds(r.q, from);
    CHECK(seq == repeat(repeat("D", D) + "P", nP) + repeat("D", D) + "T");
    for (size_t i = from; i < r.q.size(); ++i) if (r.q[i].kind == Kind::Decode) CHECK(r.q[i].count == 1);   // only A decodes
    drain(r);
    CHECK(r.out[A.id] == solo(A, int(mtp), int(A.prompt.size()) - 1));
    CHECK(r.out[B.id] == solo(B, int(mtp), int(NB) - 1));
    CHECK(r.first_main[B.id] == int(NB) - 1 && r.outside == 0);
}

// 3. replicated ranks, and the ranks that must be refused
static void check_replicated(uint32_t mtp, long& cases)
{
    const Config c = cfg(mtp, 128, false, 64, 2);
    std::vector<std::unique_ptr<Rank>> rk;
    for (int i = 0; i < 4; ++i) rk.push_back(std::make_unique<Rank>(c));
    Rank& head = *rk[0];
    const Req A = req(1, 20, 300), B = req(2, 700, 30);
    auto event_all = [&](Commit cm, const Payload* p) {
        cm.before_hash = head.s().digest();   // the head's commit goes to every rank
        for (auto& x : rk) x->d->apply(cm, p);
    };
    auto step_all = [&] {
        std::vector<Quantum> q;
        for (auto& x : rk) q.push_back(step(*x));
        for (size_t i = 1; i < q.size(); ++i) {
            CHECK(q[i].kind == q[0].kind && q[i].count == q[0].count && q[i].first == q[0].first && q[i].rows == q[0].rows &&
                  q[i].boundary == q[0].boundary && q[i].keys[0] == q[0].keys[0]);
            CHECK(rk[i]->s().digest() == head.s().digest());
        }
        return q[0];
    };
    {
        Commit cm = commit_for(head.s()); cm.admit = grant_for(head, A, head.next_page);
        const Payload p = payload_of(A);
        event_all(cm, &p);
    }
    while (head.s().slots()[0].phase != Phase::Decode) step_all();
    Commit cb = commit_for(head.s()); cb.admit = grant_for(head, B, head.next_page);
    const Payload pb = payload_of(B);
    // ranks that hold another config refuse the head's commit at the apply: the digest names prefill_rows and two_lanes
    const uint32_t other_W[] = {64, 128, 1024};
    for (int i = 0; i < 3; ++i) {
        Config oc = c; oc.prefill_rows = other_W[i]; oc.two_lanes = i == 1;
        if (oc.prefill_rows == c.prefill_rows && oc.two_lanes == c.two_lanes) continue;
        Rank odd(oc);
        Commit mine = commit_for(head.s());   // the head's event 1 would be this rank's too if its digest agreed
        bool threw = false;
        try { odd.d->apply(mine); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "event/state mismatch"; }
        CHECK(threw);
        CHECK(odd.s().resident() == 0 && odd.s().next_event() == 1);   // refused whole: nothing changed
    }
    // a rank that skipped the prefill (decode_prompt) holds another `prefilled` from the grant on
    {
        Rank dp(c);
        dp.next_page = head.next_page - reserve_pages(c, Request{B.id, uint32_t(B.prompt.size()), B.max_tokens}) - reserve_pages(c, Request{A.id, uint32_t(A.prompt.size()), A.max_tokens});
        Commit ca = commit_for(dp.s()); ca.admit = grant_for(dp, A, dp.next_page);
        const Payload pa = payload_of(A, true);
        dp.d->apply(ca, &pa);
        Rank ok(c);
        ok.next_page = dp.next_page - reserve_pages(c, Request{A.id, uint32_t(A.prompt.size()), A.max_tokens});
        Commit cc = commit_for(ok.s()); cc.admit = grant_for(ok, A, ok.next_page);
        const Payload pn = payload_of(A);
        ok.d->apply(cc, &pn);
        CHECK(dp.s().digest() != ok.s().digest());
        CHECK(dp.s().slots()[0].prefilled == A.prompt.size() - 1 && ok.s().slots()[0].prefilled == 0);
        bool threw = false;
        try { dp.d->apply(commit_for(ok.s())); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "event/state mismatch"; }
        CHECK(threw);
    }
    event_all(cb, &pb);
    while (head.s().resident()) {
        const Quantum q = step_all();
        CHECK(q.kind != Kind::Idle);
    }
    ++cases;
    for (auto& x : rk) {
        CHECK(x->out[A.id] == solo(A, int(mtp), int(A.prompt.size()) - 1));
        CHECK(x->out[B.id] == solo(B, int(mtp), int(B.prompt.size()) - 1));
        CHECK(x->rows_of(B.id) == head.rows_of(B.id) && kinds(x->q) == kinds(head.q));
    }
}

// 4. a cancel between quanta
static void check_cancel(uint32_t mtp, long& cases)
{
    const Req A = req(1, 20, 600), B = req(2, 1000, 40);
    // B cancelled after three quanta of its prefill, while A decodes
    {
        Rank r(cfg(mtp, 128, false, 64, 0));
        admit(r, A);
        while (r.s().slots()[0].phase != Phase::Decode) step(r);
        const Key kb = admit(r, B);
        while (r.rows_of(B.id).size() < 3) step(r);
        const size_t done = r.pf.size(), seen = r.q.size();
        Commit cm = commit_for(r.s()); cm.cancel = kb;
        const Run run = r.s().slots()[kb.slot].run;
        const Changes ch = r.d->apply(cm);
        ++cases;
        CHECK(ch.freed.size() == 1 && ch.freed[0] == run);
        CHECK(r.retired == std::vector<uint64_t>{B.id});
        CHECK(r.s().resident() == 1 && r.s().slots()[kb.slot].phase == Phase::Free);
        drain(r);
        CHECK(r.pf.size() == done);                                                   // no prefill call after the cancel
        CHECK(kinds(r.q, seen).find_first_not_of('D') == std::string::npos);          // only A's decode rounds
        CHECK(r.retired == std::vector<uint64_t>{B.id} && r.s().slots()[0].phase == Phase::Retiring);   // A retires at the next event
        r.d->apply(commit_for(r.s()));
        CHECK(r.retired == (std::vector<uint64_t>{B.id, A.id}));                       // each once
        CHECK(r.out[A.id] == solo(A, int(mtp), int(A.prompt.size()) - 1));
        CHECK(r.first_main.count(B.id) == 0);                                          // B never reached its tail
    }
    // B alone, cancelled mid-prefill: the next step is Idle
    {
        Rank r(cfg(mtp, 128, false));
        const Key kb = admit(r, B);
        step(r); step(r);
        const size_t done = r.pf.size();
        Commit cm = commit_for(r.s()); cm.cancel = kb;
        r.d->apply(cm);
        CHECK(r.retired == std::vector<uint64_t>{B.id} && r.s().resident() == 0);
        const Quantum q = step(r);
        CHECK(q.kind == Kind::Idle && q.count == 0 && r.pf.size() == done);
    }
    // A cancelled while it decodes alone (Batch session released): Idle, and a new request on the slot runs clean
    {
        Rank r(cfg(mtp, 128, false));
        const Key ka = admit(r, A);
        while (r.s().slots()[0].phase != Phase::Decode) step(r);
        for (int i = 0; i < 5; ++i) step(r);
        Commit cm = commit_for(r.s()); cm.cancel = ka;
        r.d->apply(cm);
        CHECK(r.retired == std::vector<uint64_t>{A.id} && step(r).kind == Kind::Idle);
        const Req C = req(3, 300, 25);
        admit(r, C);
        drain(r);
        CHECK(r.out[C.id] == solo(C, int(mtp), int(C.prompt.size()) - 1));
    }
}

// 5. the first-light config: one session, W = 1024, an 8192-token prompt; the notice horizon
static void check_first_light(uint32_t mtp, long& cases)
{
    const Config c = cfg(mtp, 1024, false, 0, 0, 1);
    CHECK(c.limit() == 1);
    const Req r = req(1, 8192, 20);
    Rank a(c);
    admit(a, r);
    ++cases;
    CHECK(a.s().horizon() == a.s().boundary() + 3);   // 1024 rows a quantum: 2 x 1024 reaches NOTICE_ROWS (2048) at the third boundary
    // a second request is refused while the first is resident, whatever the free slots
    {
        const Req r2 = req(2, 100, 10);
        Commit cm = commit_for(a.s()); cm.admit = grant_for(a, r2, a.next_page);
        const Payload p = payload_of(r2);
        bool threw = false;
        try { a.d->apply(cm, &p); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "active session/row limit"; }
        CHECK(threw && a.s().resident() == 1);
    }
    drain(a);
    CHECK(a.rows_of(r.id) == (std::vector<uint32_t>{1024, 1024, 1024, 1024, 1024, 1024, 1024, 1023}));
    CHECK(a.first_main[r.id] == 8191 && a.out[r.id] == solo(r, int(mtp), 8191));
    // the same prompt with W = 2048 and with two lanes of 1024: the target is one boundary nearer
    Rank b(cfg(mtp, 2048, false)), e(cfg(mtp, 1024, true));
    admit(b, r); admit(e, r);
    CHECK(b.s().horizon() == b.s().boundary() + 2 && e.s().horizon() == e.s().boundary() + 2);
    drain(b); drain(e);
    CHECK(b.rows_of(r.id) == (std::vector<uint32_t>{2048, 2048, 2048, 2047}));
    CHECK(e.rows_of(r.id) == (std::vector<uint32_t>{2048, 2048, 2048, 2047}));
    CHECK(b.out[r.id] == a.out[r.id] && e.out[r.id] == a.out[r.id]);
}

int main()
{
    std::vector<uint32_t> mtps{2};
#if G6_MODEL_DS41
    mtps.push_back(0);   // DS41 with no drafter (Config::validate)
#endif
    long alone = 0, inter = 0, repl = 0, canc = 0, light = 0;
    for (uint32_t mtp : mtps) {
        for (uint32_t N : {1u, 2u, 64u, 65u, 129u, 1000u, 1025u, 2049u})
            for (uint32_t W : {64u, 128u, 1024u})
                for (bool two : {false, true}) check_alone(mtp, N, W, two, alone);
        check_alone(mtp, 1000, 128, false, alone);
        {   // the literal pins: 999 prompt rows at 128 a quantum, one lane and two
            Rank a(cfg(mtp, 128, false)), b(cfg(mtp, 128, true));
            const Req r = req(7, 1000, 8);
            admit(a, r); admit(b, r); drain(a); drain(b);
            CHECK(a.rows_of(7) == (std::vector<uint32_t>{128, 128, 128, 128, 128, 128, 128, 103}));
            CHECK(b.rows_of(7) == (std::vector<uint32_t>{256, 256, 256, 231}));
        }
        for (uint32_t dpr : {0u, 64u}) for (uint32_t rounds : {0u, 1u, 3u}) check_interleave(mtp, dpr, rounds, inter);
        check_replicated(mtp, repl);
        check_cancel(mtp, canc);
        check_first_light(mtp, light);
    }
    CHECK(alone > 0 && inter == long(mtps.size()) * 6 && repl == long(mtps.size()) && canc == long(mtps.size()) && light == long(mtps.size()));
    std::printf("driver_prefill_quantum_test: %s (%zu mtp settings; %ld alone, %ld interleave, %ld replicated, %ld cancel, "
                "%ld first-light cases)\n", fails ? "FAIL" : "PASS", mtps.size(), alone, inter, repl, canc, light);
    return fails ? 1 : 0;
}
