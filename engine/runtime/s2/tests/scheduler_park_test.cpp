// Config::park (GR8_DS_PARK=1, s2/scheduler.h Phase::Parked, driver.h apply): a finished DS41 session parks its slot
// (KV kept) instead of freeing it, a follow-up that extends it is granted that slot and prefills only the new rows, and a
// new chat with no free slot evicts the oldest parked slot. The properties here, through Driver and Scheduler with a
// fake model (a pure function of request, row position and row token) and a test-side "native" layer that plays
// ds41_native.cuh's part (parked tokens per slot, the capture of an evicted slot):
//  1. knob off: a scripted mixed workload (admits, cancels, finishes over three slots) gives the same trace (every
//     quantum, digest, bind, retire, token) as the scheduler before this change: TRACE_OFF is that trace's hash, built
//     against origin/main's s2 headers (build this file with -DGOLDEN_ONLY against those to reproduce it)
//  2. park at request end: the finished session's slot is Parked at the next commit, its session retired once, it is
//     not resident (idle_boundary still advances), a cancel never names it
//  3. a follow-up that extends the parked tokens takes that slot with reuse = the parked position and prefills only the
//     delta; its tokens and prefill calls equal the knob-off run that restores the same position from a cache
//  4. a prompt that does not extend the parked tokens does not take that slot (a free one first)
//  5. a new chat with no free slot evicts the oldest parked slot, captured (native) before the bind
//  6. a parked slot never shrinks, and a parked slot does not count toward limit()
//  7. two replicas driven by the same events agree on every digest; the digest names the knob
// Host only. Build and run: see scripts/host_tests.sh (with -DG6_MODEL_DS41=1).
#include "s2/driver.h"
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 0x1234567ull) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
constexpr int V = 5;
// the fake model: a token from (salt, position, token); salt is the chat, so a follow-up continues its chat's stream
static int model(const s2::Row& r, int kind, uint32_t salt)
{
    int t = (int)(mix(salt, kind == s2::Step::MAIN ? r.pos : r.pos + 1, r.tok) % V);
    if (kind != s2::Step::MAIN && mix(salt, r.pos, r.tok + 99) % 3 == 0) t = (t + 1) % V;
    return t;
}
struct Req { uint64_t id; uint32_t chat; std::vector<int> prompt; uint32_t max_tokens; };
static Req req(uint64_t id, uint32_t chat, int n, uint32_t max_tokens)
{
    Req r{id, chat, {}, max_tokens};
    for (int k = 0; k < n; ++k) r.prompt.push_back(int(mix(chat, uint32_t(k), 7) % V));
    return r;
}

struct Trace {
    uint64_t h = 14695981039346656037ull;
    void add(uint64_t v) { for (unsigned i = 0; i < 8; ++i) { h ^= uint8_t(v >> (8 * i)); h *= 1099511628211ull; } }
};

struct Pf { uint64_t id; uint32_t first, rows; };
struct Rank {
    Config c;
    std::unique_ptr<Driver> d;
    std::map<uint64_t, uint32_t> chat;          // request id -> its chat (the model's salt)
    std::map<uint64_t, std::vector<int>> out;   // request id -> emitted tokens
    std::vector<Pf> pf;
    std::vector<uint64_t> retired, bound;
    std::vector<std::string> native;            // the test's native layer: "park s id", "capture s id", "resume s id"
    // the native parked table (ds41_native.cuh's): slot -> {request, tokens it holds, position}
    struct Parked { bool on = false; uint64_t id = 0; std::vector<int> tokens; uint32_t pos = 0; };
    std::array<Parked, MAX_SESSIONS> parked{};
    int claim = -1;                             // the parked slot the next grant resumes (the take hook's claim)
    Trace t;
    explicit Rank(const Config& cfg) : c(cfg) {
        DriverHooks h;
        h.bind = [this](const Grant& g) {
            bound.push_back(g.req.id);
            t.add(0xB1); t.add(g.key.slot); t.add(g.key.generation); t.add(g.req.id); t.add(g.reuse);
#if !GOLDEN_ONLY
            Parked& p = parked[g.key.slot];
            if (claim >= 0) {
                CHECK(int(g.key.slot) == claim && p.on && p.pos == g.reuse);
                native.push_back("resume " + std::to_string(g.key.slot) + " " + std::to_string(p.id));
            } else if (p.on)   // a new chat on a parked slot: capture it first (today's path), then the slot is the chat's
                native.push_back("capture " + std::to_string(g.key.slot) + " " + std::to_string(p.id));
            p = Parked{};
            claim = -1;
#endif
        };
        h.retire = [this](const Slot& s) { retired.push_back(s.req.id); t.add(0x7E); t.add(s.key.slot); t.add(s.req.id); };
        h.prefill = [this](const Slot& s, const Payload&, uint32_t first, uint32_t rows) {
            pf.push_back({s.req.id, first, rows}); t.add(0x9F); t.add(s.req.id); t.add(first); t.add(rows);
        };
        h.run = [this](const s2::Step& st) {
            std::vector<int> ids;
            t.add(0x51); t.add(st.kind); t.add(st.r.size());
            for (const auto& r : st.r) {
                const Slot& s = d->state().slots()[r.sess];
                ids.push_back(model(r, st.kind, chat.at(s.req.id)));
                t.add(r.sess); t.add(uint64_t(r.pos)); t.add(uint64_t(r.tok)); t.add(uint64_t(ids.back()));
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
#if !GOLDEN_ONLY
        // service.h retire_run on a finished session: with park the native layer parks it (ds41_native.cuh park hook)
        h.finished = [this](const Slot& s) {
            if (!c.park) return;
            Parked& p = parked[s.key.slot];
            p.on = true; p.id = s.req.id; p.tokens = prompt_of.at(s.req.id);
            const auto& o = out.at(s.req.id);
            p.tokens.insert(p.tokens.end(), o.begin(), o.end());
            p.pos = uint32_t(p.tokens.size()) - 1;   // the last emitted token is not in the KV yet
            native.push_back("park " + std::to_string(s.key.slot) + " " + std::to_string(s.req.id));
        };
#endif
        d = std::make_unique<Driver>(c, h);
    }
    std::map<uint64_t, std::vector<int>> prompt_of;
    const Scheduler& s() const { return d->state(); }
    // the head's match over the parked table (ds41_native.cuh): the slot whose tokens the prompt extends, longest first
    int match(const std::vector<int>& prompt) const {
        int best = -1;
        for (unsigned q = 0; q < MAX_SESSIONS; ++q) {
            const Parked& p = parked[q];
            if (!p.on || !p.pos || p.pos > prompt.size() - 1 || !std::equal(p.tokens.begin(), p.tokens.begin() + p.pos, prompt.begin())) continue;
            if (best < 0 || p.pos > parked[size_t(best)].pos) best = int(q);
        }
        return best;
    }
};

static Config cfg(bool park, uint32_t sessions = 3)
{
    Config c;
    c.epoch = 1; c.mtp = 1; c.max_rows = 8; c.max_sessions = sessions; c.prefill_rows = 64; c.two_lanes = false;
#if !GOLDEN_ONLY
    c.park = park;
#else
    (void)park;
#endif
    return c;
}
static Commit commit_for(const Scheduler& s)
{
    Commit cm;
    cm.epoch = s.config().epoch; cm.event = s.next_event(); cm.boundary = s.boundary(); cm.before_hash = s.digest();
    return cm;
}
// one admission: reuse rows from a parked slot (park) or a cache (reuse > 0 without park), the same grant to every rank
static Grant grant(Rank& r, const Req& q, uint32_t reuse = 0, int park = -1)
{
    const Request rq{q.id, uint32_t(q.prompt.size()), q.max_tokens};
    Run run{0, reserve_pages(r.c, rq), 1, mix(uint32_t(q.id), 1, 3)};
#if !GOLDEN_ONLY
    Key k = park >= 0 ? r.s().next_key({}, unsigned(park)) : r.s().next_key();
#else
    (void)park;
    Key k = r.s().next_key();
#endif
    return Grant{k, rq, run, reuse};
}
static void apply(Rank& r, Commit cm, const Req* q = nullptr, int park = -1)
{
    r.t.add(0xEE); r.t.add(cm.before_hash);
    Payload p;
    if (q) { p.prompt = q->prompt; p.chunk = 4; r.chat[q->id] = q->chat; r.prompt_of[q->id] = q->prompt; }
    r.claim = park;
    r.d->apply(cm, q ? &p : nullptr);
    r.t.add(r.s().digest());
}
static void step(Rank& r)
{
    const Quantum q = r.d->step();
    r.t.add(0x5E); r.t.add(uint64_t(q.kind)); r.t.add(q.count); r.t.add(q.first); r.t.add(q.rows); r.t.add(q.boundary);
    for (unsigned i = 0; i < q.count; ++i) { r.t.add(q.keys[i].slot); r.t.add(q.keys[i].generation); }
}
static void drain(Rank& r)
{
    for (long g = 0; r.s().resident(); ++g) {
        step(r);
        if (g > 100000) { CHECK(!"no progress"); fails += 1000; break; }
    }
}
// an idle boundary (no resident session): a commit's notice lands there, as service.h compute does
static void idle(Rank& r) { if (!r.s().resident()) r.d->idle_boundary(r.s().boundary() + NOTICE_MIN); }

// 1. knob off: the scripted workload's trace hash against the scheduler before parking existed
// TRACE_OFF: scripted() on origin/main 722a6dd's scheduler.h/driver.h (g++ -DGOLDEN_ONLY=1 -DG6_MODEL_DS41=1 -I <main>/engine/runtime)
static constexpr uint64_t TRACE_OFF = 0x65c7f53e15dcb45dull;
static uint64_t scripted(bool park)
{
    Rank r(cfg(park));
    std::vector<Req> q;
    for (uint64_t i = 1; i <= 14; ++i) q.push_back(req(i, uint32_t(i), int(3 + mix(uint32_t(i), 5, 5) % 150), 2 + mix(uint32_t(i), 6, 6) % 18));
    size_t next = 0;
    for (int it = 0; it < 400 && (next < q.size() || r.s().resident()); ++it) {
        idle(r);
        if (it % 7 == 3 && r.s().resident()) {   // cancel the lowest live request
            uint64_t id = UINT64_MAX;
            for (const auto& sl : r.s().slots())
                if ((sl.phase == Phase::Prefill || sl.phase == Phase::Decode) && sl.req.id < id) id = sl.req.id;
            Commit cm = commit_for(r.s());
            cm.cancel = r.s().find(id);
            apply(r, cm);
        } else if (next < q.size() && r.s().resident() < r.c.limit()) {
            Commit cm = commit_for(r.s());
            cm.admit = grant(r, q[next]);
            apply(r, cm, &q[next]);
            ++next;
        }
        for (int k = 0; k < 1 + it % 5 && r.s().resident(); ++k) step(r);
    }
    drain(r);
    for (const auto& [id, o] : r.out) { r.t.add(id); for (int x : o) r.t.add(uint64_t(x)); }
    return r.t.h;
}

#if !GOLDEN_ONLY
// 2, 3, 4: park at request end; a follow-up that extends it resumes the slot; one that does not extend it does not
static void follow_up(long& cases)
{
    // chat 50: turn 1 (request 1), then turn 2 = turn 1's prompt + its reply + 7 new tokens (request 3); chat 60 between
    const Req a = req(1, 50, 40, 9);
    Rank on(cfg(true)), off(cfg(false));
    for (Rank* r : {&on, &off}) {
        apply(*r, [&] { Commit cm = commit_for(r->s()); cm.admit = grant(*r, a); return cm; }(), &a);
        drain(*r);
    }
    CHECK(on.s().slots()[0].phase == Phase::Retiring && on.native.size() == 1 && on.native[0] == "park 0 1");
    // the next commit (a new chat 60, request 2) parks slot 0: request 1 is retired once, slot 0 not resident
    const Req b = req(2, 60, 20, 4);
    for (Rank* r : {&on, &off}) {
        idle(*r);
        apply(*r, [&] { Commit cm = commit_for(r->s()); cm.admit = grant(*r, b); return cm; }(), &b);
    }
    CHECK(on.s().slots()[0].phase == Phase::Parked && on.s().slots()[0].parked == 1 && on.s().parked() == 1);
    CHECK(off.s().slots()[0].phase == Phase::Prefill && off.s().slots()[0].req.id == 2);   // knob off: the freed slot, as before
    CHECK(on.retired == std::vector<uint64_t>{1} && off.retired == std::vector<uint64_t>{1});
    CHECK(on.s().slots()[1].key.slot == 1 && on.s().slots()[1].phase == Phase::Prefill);   // chat 60 took a free slot, not the parked one
    CHECK(on.match(b.prompt) < 0);            // 4: chat 60 does not extend chat 50's tokens
    CHECK(!on.s().find(1));                   // a cancel of a finished request names nothing
    drain(on); drain(off);
    CHECK(on.s().resident() == 0 && on.s().slots()[0].phase == Phase::Parked);
    const uint64_t b0 = on.s().boundary();
    on.d->idle_boundary(b0 + 3);              // a parked slot is not resident: the idle boundary advances
    CHECK(on.s().boundary() == b0 + 3);
    off.d->idle_boundary(off.s().boundary() + 3);
    // turn 2 of chat 50
    Req c = a; c.id = 3; c.max_tokens = 6;
    c.prompt.insert(c.prompt.end(), on.out.at(1).begin(), on.out.at(1).end());
    for (int k = 0; k < 7; ++k) c.prompt.push_back(int(mix(50, 1000 + k, 7) % V));
    const int q = on.match(c.prompt);
    CHECK(q == 0);
    const uint32_t pos = on.parked[0].pos;
    CHECK(pos == a.prompt.size() + a.max_tokens - 1);
    const size_t pf_on = on.pf.size(), pf_off = off.pf.size(), ret_on = on.retired.size();
    apply(on, [&] { Commit cm = commit_for(on.s()); cm.admit = grant(on, c, pos, q); return cm; }(), &c, q);
    // knob off: the same rows come from a cache restore (reuse = pos) on whatever slot is free
    apply(off, [&] { Commit cm = commit_for(off.s()); cm.admit = grant(off, c, pos); return cm; }(), &c);
    CHECK(on.s().slots()[0].phase == Phase::Prefill && on.s().slots()[0].req.id == 3 && on.s().slots()[0].prefilled == pos &&
          on.s().slots()[0].key.generation == 2);
    CHECK(on.native.back() == "resume 0 1");
    CHECK(on.retired.size() == ret_on + 1 && on.retired.back() == 2);   // request 2 parked at this commit; request 1 is not retired twice
    drain(on); drain(off);
    // only the delta prefilled, from the parked position, and the same calls and tokens as the cache restore
    uint32_t rows = 0; bool from = true;
    for (size_t i = pf_on; i < on.pf.size(); ++i) { rows += on.pf[i].rows; if (i == pf_on) from = on.pf[i].first == pos; }
    CHECK(from && rows == c.prompt.size() - 1 - pos);
    CHECK(on.pf.size() - pf_on == off.pf.size() - pf_off);
    for (size_t i = 0; i < on.pf.size() - pf_on; ++i)
        CHECK(on.pf[pf_on + i].first == off.pf[pf_off + i].first && on.pf[pf_on + i].rows == off.pf[pf_off + i].rows);
    CHECK(on.out == off.out);
    cases += 1;
}

// 5. no free slot: a new chat evicts the oldest parked slot (captured first); 6. parked never shrinks or counts
static void evict(long& cases)
{
    Config c2 = cfg(true, 2);
    c2.r2_rows = 64; c2.window_pages = 1;      // a shrink (the smaller R) is possible, so only the phase can refuse it
    Rank r(c2);
    CHECK(r.c.limit() == 2);
    const Req a = req(1, 70, 12, 3), b = req(2, 71, 12, 5);
    Commit cm = commit_for(r.s()); cm.admit = grant(r, a); apply(r, cm, &a);
    idle(r);
    cm = commit_for(r.s()); cm.admit = grant(r, b); apply(r, cm, &b);
    drain(r);                                  // both finish: request 1 first (fewer tokens)
    // the next commit parks both (1 before 2: lower slot first among the same commit)
    const Req c = req(3, 72, 30, 4);
    idle(r);
    cm = commit_for(r.s());
    cm.admit = grant(r, c);                    // no free slot: the oldest Retiring/parked one, slot 0
    CHECK(cm.admit->key.slot == 0);
    apply(r, cm, &c);
    CHECK(r.s().slots()[1].phase == Phase::Parked && r.s().slots()[1].req.id == 2 && r.s().slots()[1].parked == 2);
    CHECK(r.native.size() == 3 && r.native[2] == "capture 0 1");   // evicted request 1 was captured before its slot was bound
    CHECK(r.s().resident() == 1);
    // 6: a parked slot never shrinks (Scheduler::apply validates before it changes anything, so r goes on)
    Commit sh = commit_for(r.s());
    sh.shrink = Shrink{r.s().slots()[1].key, 64, r.s().slots()[1].run};
    bool threw = false;
    try { r.d->apply(sh); } catch (const std::exception& e) { threw = std::string(e.what()) == "bad shrink"; }
    CHECK(threw && r.s().slots()[1].phase == Phase::Parked && !r.s().slots()[1].req.r_rows);
    // and does not count toward limit(): live 1 < limit 2 with no free slot, so a new chat takes the parked one
    const Req d = req(4, 73, 9, 2);
    cm = commit_for(r.s()); cm.admit = grant(r, d);
    CHECK(cm.admit->key.slot == 1);
    apply(r, cm, &d);
    CHECK(r.native.back() == "capture 1 2" && r.s().resident() == 2 && r.s().parked() == 0);
    drain(r);
    CHECK(r.out.at(3).size() == 4 && r.out.at(4).size() == 2);
    cases += 1;
}

// 7. replicas: the same events give the same digest at every commit and step; the digest names the knob
static void replicas(long& cases)
{
    auto run = [](Rank& r, std::vector<uint64_t>& dig) {
        // ten requests over chats 100..102: a chat's next turn is its last turn's prompt + reply + a few new tokens
        std::map<uint32_t, uint64_t> last;    // chat -> its last request
        uint64_t next = 1;
        for (int it = 0; it < 400 && (next <= 10 || r.s().resident()); ++it) {
            idle(r);
            if (next <= 10 && r.s().resident() < r.c.limit()) {
                const uint32_t chat = uint32_t(100 + next % 3);
                Req x = req(next, chat, int(5 + mix(uint32_t(next), 2, 2) % 60), 2 + mix(uint32_t(next), 3, 3) % 9);
                if (last.count(chat) && r.out.count(last[chat])) {
                    const uint64_t l = last[chat];
                    bool done = true;
                    for (const auto& sl : r.s().slots()) done &= !(sl.req.id == l && (sl.phase == Phase::Prefill || sl.phase == Phase::Decode));
                    if (done) {
                        x.prompt = r.prompt_of.at(l);
                        x.prompt.insert(x.prompt.end(), r.out.at(l).begin(), r.out.at(l).end());
                        for (int k = 0; k < 3 + int(next % 4); ++k) x.prompt.push_back(int(mix(chat, uint32_t(5000 + 10 * next + k), 7) % V));
                    }
                }
                last[chat] = next;
                const int p = r.match(x.prompt);
                Commit cm = commit_for(r.s());
                cm.admit = grant(r, x, p >= 0 ? r.parked[size_t(p)].pos : 0, p);
                apply(r, cm, &x, p);
                ++next;
            }
            dig.push_back(r.s().digest());
            for (int k = 0; k < 1 + it % 3 && r.s().resident(); ++k) { step(r); dig.push_back(r.s().digest()); }
        }
        drain(r);
        dig.push_back(r.s().digest());
    };
    Rank a(cfg(true)), b(cfg(true)), off(cfg(false));
    std::vector<uint64_t> da, db, doff;
    run(a, da); run(b, db); run(off, doff);
    CHECK(da == db && !da.empty());
    CHECK(a.out == b.out && a.native == b.native && a.out.size() == 10);
    long resumes = 0, captures = 0;
    for (const auto& n : a.native) { resumes += n.rfind("resume", 0) == 0; captures += n.rfind("capture", 0) == 0; }
    std::printf("replicas: %zu digests, %ld resumes, %ld captures\n", da.size(), resumes, captures);
    CHECK(resumes >= 2 && captures >= 1);
    CHECK(da.front() != doff.front());          // a rank booted with the other setting is refused at the first commit
    cases += 1;
}
#endif

int main()
{
    const uint64_t off = scripted(false);
    std::printf("TRACE_OFF 0x%016llx\n", (unsigned long long)off);
#if !GOLDEN_ONLY
    CHECK(off == TRACE_OFF);
    long cases = 0;
    follow_up(cases);
    evict(cases);
    replicas(cases);
    const uint64_t on = scripted(true);   // knob on, no follow-ups: every chat still served (parks, evictions)
    (void)on;
    if (fails) { std::fprintf(stderr, "scheduler_park_test: %d failures\n", fails); return 1; }
    std::printf("scheduler_park_test: ok (%ld cases)\n", cases);
#endif
    return 0;
}
