// Prefix reuse rows (Dispatch::track_rows, prefix_reuse.h) in the DS41 build, with drafts off and on. Host only.
//  1. drafts off (MTP 0): a Driver serves requests through a Dispatch the way Service does. Each request has an insert
//     rule with more tokens than one prompt chunk, as a forced tool call does. The forced tokens run as a main step of
//     several rows inside a round, and no draft step follows it. The draft rows must still have no hole: the next
//     one-row step writes its draft row right after them (PrefixReuse::written asserts otherwise). At the end the
//     retained prefix covers every row the request wrote.
//  2. drafts on and off, hand-built steps: with drafts on, a verify step of several rows leaves the draft rows to the
//     draft step that follows it; with drafts off, its rows count as draft rows too.
// The fake model is a pure function of (request, row position, row token).
#include "s2/dispatch.h"
#include "s2/driver.h"
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
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
constexpr int V = 7;

static Config config(uint32_t mtp)
{
    Config c;
    c.epoch = 1; c.mtp = mtp; c.max_rows = 8; c.max_sessions = 1; c.prefill_rows = 64; c.pool_rows = 64 * 8; c.two_lanes = false;
    c.validate();
    return c;
}
static g6rt::ServeReq serve_req(uint32_t seed, int plen, int n_gen, const std::vector<int>& forced)
{
    g6rt::ServeReq r;
    for (int k = 0; k < plen; ++k) r.prompt.push_back(int(mix(seed, uint32_t(k), 7) % V));
    r.n_gen = n_gen;
    r.ins.after = 0; r.ins.unless = -1; r.ins.tokens = forced;   // the forced text takes the reply's first token's place
    return r;
}

// 1. A Driver at MTP 0 with a Dispatch that tracks rows, as Service wires them (bind -> Dispatch::begin, run ->
// Dispatch::run, prefill -> Dispatch::prefill, finished -> the reuse state Service::retire_run retains).
static std::map<uint64_t, std::vector<int>> serve_mtp0(bool decode_prompt, int reuse_mode = 0)
{
    const Config c = config(0);
    Sessions ss(c, "", false, 4);
    DispatchHooks dh;
    dh.bucket = [](bool, int) { return 0; };
    dh.mode = [](int, int) {};
    dh.launch = [&](const Launch& l) {
        CHECK(l.step.kind == s2::Step::MAIN);   // MTP 0: no draft or chain step
        std::vector<int> ids;
        for (const auto& r : l.step.r) ids.push_back(int(mix(uint32_t(l.requests[0]), uint32_t(r.pos), uint32_t(r.tok)) % V));
        return ids;
    };
    dh.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    Dispatch dp(ss, c.max_rows, dh);
    dp.track_rows(c.mtp > 0);

    const g6rt::ServeReq* cur = nullptr;
    std::map<uint64_t, std::vector<int>> out;
    std::map<uint64_t, g6rt::PrefixReuse> kept;
    std::map<uint64_t, int> first_main;
    std::map<uint64_t, std::vector<int>> main_ends;
    long forced_steps = 0;   // main steps of several rows inside a round
    std::unique_ptr<Driver> d;
    DriverHooks h;
    h.bind = [&](const Grant& g) { ss.bind(g, *cur); dp.begin(g.key.slot, int(g.reuse)); };
    h.retire = [&](const Slot& s) { ss.retire(s); dp.release(s.key.slot); };
    h.prefill = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) { dp.prefill(s, p, first, rows); };
    h.run = [&](const s2::Step& st) {
        if (st.kind == s2::Step::MAIN) {
            const auto id = d->state().slots()[st.r.front().sess].req.id;
            if (!first_main.count(id)) first_main[id] = st.r.front().pos;
            main_ends[id].push_back(st.r.back().pos + 1);
        }
        if (st.kind == s2::Step::MAIN && st.round >= 0 && st.M() > 1) ++forced_steps;
        return dp.run(st);
    };
    h.emit = [&](Key k, const int* ids, int n, int) {
        const Slot& s = d->state().slots()[k.slot];
        auto& o = out[s.req.id];
        const int cnt = std::min<int>(n, int(s.req.max_tokens - o.size()));
        o.insert(o.end(), ids, ids + cnt);
        return Emission{uint32_t(cnt), o.size() == s.req.max_tokens};
    };
    h.finished = [&](const Slot& s) {
        g6rt::PrefixReuse st = dp.rows(s.key.slot);
        const auto& o = out[s.req.id];
        st.finish(ss.at(s.key.slot).payload.prompt, o, int(o.size()), true);
        kept[s.req.id] = st;
    };
    d = std::make_unique<Driver>(c, h);

    // forced texts of 5 tokens (chunks of 4 and 1 rows), 2 tokens (one chunk of 2) and 9 tokens (4, 4 and 1)
    const std::vector<g6rt::ServeReq> reqs = {serve_req(1, 11, 12, {3, 1, 4, 1, 5}), serve_req(2, 6, 9, {2, 6}),
                                             serve_req(3, 20, 14, {5, 5, 5, 5, 5, 5, 5, 5, 5})};
    uint32_t page = 0;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const g6rt::ServeReq& r = reqs[i];
        const uint64_t id = i + 1;
        cur = &r;
        Commit cm;
        cm.epoch = c.epoch; cm.event = d->state().next_event(); cm.boundary = d->state().boundary(); cm.before_hash = d->state().digest();
        Request rq = Sessions::request(id, r);
        const uint32_t reuse = reuse_mode == 2 ? uint32_t(r.prompt.size() - 1) : reuse_mode == 1 ? 2u : 0u;
        // chunk=4 does not divide the distance from reuse=2 to this checkpoint.
        if (reuse_mode == 1) rq.ckpt_at = 4;
        Run run;
        run.first_page = page; run.pages = reserve_pages(c, rq); run.table_version = 1; run.list_hash = mix(uint32_t(id), page, 3);
        cm.admit = Grant{d->state().next_key(), rq, run, reuse};
        Payload p;
        p.prompt = r.prompt; p.chunk = 4; p.decode_prompt = decode_prompt; p.ins = r.ins;
        d->apply(cm, &p);
        for (long g = 0; d->state().resident(); ++g) {
            const Quantum q = d->step();
            CHECK(q.kind != Kind::Idle);
            if (decode_prompt) CHECK(q.kind != Kind::Prefill);
            if (q.kind == Kind::Idle || g > 10000) { CHECK(!"no progress"); return {}; }
        }
        CHECK(kept.count(id));
        CHECK(first_main[id] == int(decode_prompt ? reuse : r.prompt.size() - 1));
        if (decode_prompt && rq.ckpt_at > reuse)
            CHECK(std::find(main_ends[id].begin(), main_ends[id].end(), int(rq.ckpt_at)) != main_ends[id].end());
        const auto& o = out[id];
        const auto& st = kept[id];
        const int n = int(r.prompt.size()), t = int(o.size());
        CHECK(t == r.n_gen);
        CHECK(std::equal(r.ins.tokens.begin(), r.ins.tokens.end(), o.begin()));
        // every row but the last token's: main and draft rows alike, so the retained prefix is all of them
        CHECK(st.main_end == n + t - 1);
        CHECK(st.draft_end == st.main_end);
        CHECK(st.written_end == st.main_end);
        CHECK(st.valid == n + t - 1 && int(st.history.size()) == n + t);
        page += run.pages;
    }
    CHECK(forced_steps == (reuse_mode == 1 ? 7 : 4));   // checkpoint alignment reduces chunks from 4 to 2
    std::printf("serve mtp0 (%s prompt): %zu requests, %ld forced steps of several rows\n",
                decode_prompt ? "decode" : "prefilled", reqs.size(), forced_steps);
    return out;
}

// 2. Hand-built steps through a Dispatch on one bound slot
static s2::Step step(s2::Step::Kind k, long round, int pos0, int n)
{
    s2::Step s;
    s.kind = k; s.round = round; s.sess = {0}; s.row0 = {0}; s.rows = {n}; s.ctx = {pos0 + n}; s.cls = s2::row_class(n);
    for (int i = 0; i < n; ++i) s.r.push_back({0, 1, pos0 + i, pos0 + i + 1, -1});
    return s;
}
static void steps(bool drafts)
{
    const Config c = config(drafts ? 5 : 0);
    Sessions ss(c, "", false, 4);
    DispatchHooks dh;
    dh.bucket = [](bool, int) { return 0; };
    dh.mode = [](int, int) {};
    dh.launch = [](const Launch& l) { return std::vector<int>(size_t(l.step.M()), 1); };
    dh.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    Dispatch dp(ss, c.max_rows, dh);
    dp.track_rows(drafts);
    const g6rt::ServeReq r = serve_req(9, 9, 16, {1});
    const Request rq = Sessions::request(1, r);
    Run run;
    run.first_page = 0; run.pages = reserve_pages(c, rq); run.table_version = 1;
    const Grant g{{c.epoch, 0, 1}, rq, run, 0};
    ss.bind(g, r);
    dp.begin(0, 0);
    Slot slot; slot.key = g.key; slot.run = run;
    dp.prefill(slot, ss.at(0).payload, 0, 8);
    const auto& w = dp.rows(0);
    CHECK(w.main_end == 8 && w.draft_end == 8);
    dp.run(step(s2::Step::MAIN, -1, 8, 1));   // the prompt tail
    CHECK(w.main_end == 9 && w.draft_end == 9);
    if (drafts) {
        dp.run(step(s2::Step::MAIN, 0, 9, 5));   // a verify of 4 drafts: the draft step writes their draft rows
        CHECK(w.main_end == 14 && w.draft_end == 9);
        dp.run(step(s2::Step::DRAFT, 0, 9, 5));
        CHECK(w.main_end == 14 && w.draft_end == 14);
    } else {
        dp.run(step(s2::Step::MAIN, 0, 9, 4));   // a forced chunk: no draft step follows
        CHECK(w.main_end == 13 && w.draft_end == 13);
        dp.run(step(s2::Step::MAIN, 0, 13, 1));
        CHECK(w.main_end == 14 && w.draft_end == 14);
    }
    dp.run(step(s2::Step::MAIN, 1, 14, 1));
    CHECK(w.main_end == 15 && w.draft_end == 15 && w.written_end == 15);
}

int main()
{
    const auto reference = serve_mtp0(false);
    CHECK(serve_mtp0(true) == reference);
    CHECK(serve_mtp0(true, 1) == reference);
    CHECK(serve_mtp0(true, 2) == reference);
    steps(true);
    steps(false);
    if (fails) { std::fprintf(stderr, "ds41_reuse_mtp0_test: %d FAILED\n", fails); return 1; }
    std::printf("ds41_reuse_mtp0_test ok\n");
    return 0;
}
