// F12 stage 1, host only: DS41 c8 W-pack planning, held-tail membership, fairness, horizon and
// one host/fabric sequence. No CUDA path is exercised. Build with -DG6_MODEL_DS41=1.
#include "s2/bootstrap.h"
#include "s2/dispatch.h"
#include "s2/ds41_prefill.h"
#include "e2e_c8_prompt_fixture.h"
#include <cstdio>
#include <numeric>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
template<class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static Config cfg(bool on = true) {
    Config c; c.pool_rows = 614400; c.max_rows = 48; c.max_sessions = 8; c.mtp = 5;
    c.prefill_rows = 4096; c.two_lanes = false; c.tail_first = true; c.c8_pack = on;
    return c;
}
static Key admit(Scheduler& s, uint64_t id, uint32_t prompt, uint32_t tokens = 256,
                 uint32_t reuse = 0, uint32_t restored = 0) {
    const Request r{id, prompt, tokens};
    const Key k = s.next_key();
    Commit e; e.epoch = s.config().epoch; e.event = s.next_event(); e.boundary = s.boundary();
    e.before_hash = s.digest();
    e.admit = Grant{k, r, Run{0, grant_pages(s.config(), r), 1, id + 7}, reuse, restored};
    s.apply(e); return k;
}
static void cancel(Scheduler& s, Key k) {
    Commit e; e.epoch = s.config().epoch; e.event = s.next_event(); e.boundary = s.boundary();
    e.before_hash = s.digest(); e.cancel = k; s.apply(e);
}
static Quantum step(Scheduler& s) {
    const Quantum q = s.next();
    if (q.kind == Kind::Prefill) s.complete_prefill();
    else if (q.kind == Kind::PrefillPack) s.complete_prefill_pack();
    else if (q.kind == Kind::PromptTail) s.complete_tail({q.keys[0], 1, false, 0});
    else if (q.kind == Kind::PromptTailPack || q.kind == Kind::Decode) {
        std::vector<Completion> out;
        for (unsigned i = 0; i < q.count; ++i) out.push_back({q.keys[i], 1, false, 0});
        if (q.kind == Kind::PromptTailPack) s.complete_tail_pack(out); else s.complete_decode(out);
    }
    return q;
}
static uint64_t projected_horizon(const Scheduler& s, bool want) {
    auto sim = s.slots(); uint32_t run = 0;
    const Config& c = s.config();
    const Quantum first = Scheduler::plan(c, sim, want);
    if (first.kind == Kind::Idle) return s.boundary() + NOTICE_MIN;
    Scheduler::advance(sim, first); Scheduler::alternate(c, first.kind, want, run);
    uint32_t covered = 0;
    for (uint64_t k = NOTICE_MIN; k < c.notice_max; ++k) {
        const Quantum q = Scheduler::plan(c, sim, want);
        if (q.kind == Kind::Idle) return s.boundary() + k;
        Scheduler::advance(sim, q); Scheduler::alternate(c, q.kind, want, run);
        if (q.kind == Kind::PrefillPack) {
            for (unsigned i = 0; i < q.count; ++i) covered += q.row_counts[i];
        } else if (q.kind == Kind::Prefill) covered += q.rows;
        else if (q.kind == Kind::PromptTailPack) covered += q.count;
        else if (q.kind == Kind::PromptTail) ++covered;
        if (covered >= NOTICE_ROWS) return s.boundary() + k;
    }
    return s.boundary() + c.notice_max;
}
static void code_pack_and_off() {
    const uint32_t lens[] = {56,58,59,59,59,61,62,65};
    Scheduler a(cfg()), b(cfg()), off(cfg(false));
    for (unsigned i = 0; i < 8; ++i) { admit(a, i + 1, lens[i]); admit(b, i + 1, lens[i]); admit(off, i + 1, lens[i]); }
    CHECK(a.digest() == b.digest() && a.horizon() == b.horizon());
    CHECK(a.horizon() == projected_horizon(a, true));
    CHECK(a.digest() != off.digest());
    const Quantum qo = off.next(); CHECK(qo.kind == Kind::Prefill && qo.count == 1);
    off.complete_prefill();
    const Quantum q = a.next();
    CHECK(q.kind == Kind::PrefillPack && q.count == 8 && q.rows == 471);
    for (unsigned i = 0; i < 8; ++i) CHECK(q.keys[i].slot == i && q.firsts[i] == 0 && q.row_counts[i] == lens[i] - 1);
    a.complete_prefill_pack();
    CHECK(a.horizon() == projected_horizon(a, true));
    const Quantum qb = b.next(); CHECK(qb.kind == q.kind && qb.rows == q.rows);
    b.complete_prefill_pack();
    CHECK(a.digest() == b.digest());
    // MAX_SESSIONS is 32 since S32: the 8 admitted slots are checked, the other 24 stay Free
    unsigned live = 0;
    for (const auto& x : a.slots()) if (x.phase != Phase::Free) { ++live; CHECK(x.phase == Phase::Prefill && x.pack_tail && x.deferred == 0 && prefill_left(x) == 0); }
    CHECK(live == 8);
    const Quantum tail = a.next(); CHECK(tail.kind == Kind::PromptTailPack && tail.count == 8 && tail.rows == 8);
    std::vector<Completion> result;
    for (unsigned i = 0; i < tail.count; ++i) result.push_back({tail.keys[i], 1, false, 0});
    const uint64_t rev = a.revision();
    CHECK(throws([&] { auto bad = result; bad[7].key = bad[0].key; a.complete_tail_pack(bad); }));
    CHECK(a.revision() == rev && a.slots()[0].phase == Phase::Prefill);
    a.complete_tail_pack(result);
    CHECK(step(b).kind == Kind::PromptTailPack && a.digest() == b.digest());
    CHECK(step(a).kind == Kind::Decode);
    CHECK(CONTROL_VERSION == 10);
    BootstrapConfig ba, bo; ba.scheduler = cfg(); bo.scheduler = cfg(false);
    ba.runtime_signature = bo.runtime_signature = "c8";
    CHECK(ba.agreement().find(" c8pack ctl 10") != std::string::npos);
    CHECK(bo.agreement().find(" c8pack") == std::string::npos);
    CHECK(ba.agreement() != bo.agreement());
    std::printf("scheduler_c8_pack_test: code 471 rows -> one W pack, one 8-tail pack; OFF solo; agreement differs\n");
}
static void class_totals_and_single() {
    for (const auto& x : std::vector<std::pair<uint32_t,uint32_t>>{{39,304},{32,248}}) {
        Scheduler s(cfg());
        for (unsigned i = 0; i < 8; ++i) admit(s, i + 1, x.first);
        const Quantum q = s.next(); CHECK(q.kind == Kind::PrefillPack && q.rows == x.second); s.complete_prefill_pack();
        CHECK(step(s).kind == Kind::PromptTailPack);
    }
    Scheduler one(cfg()); admit(one, 1, 65);
    CHECK(step(one).kind == Kind::Prefill); CHECK(step(one).kind == Kind::PromptTail);
    std::printf("scheduler_c8_pack_test: prose 304, structured 248; one session stays solo\n");
}
static void cap_and_fallback() {
    for (uint32_t cap : {64u, 128u}) {
        Config c = cfg(); c.decode_prefill_rows = cap;
        Scheduler s(c); admit(s, 1, 39);
        CHECK(step(s).kind == Kind::Prefill); CHECK(step(s).kind == Kind::PromptTail);
        for (unsigned i = 0; i < 3; ++i) admit(s, i + 2, 61);
        CHECK(step(s).kind == Kind::Decode); // alternation first offers the old decoder its round
        const Quantum q = s.next();
        CHECK(q.kind == (cap == 64 ? Kind::Prefill : Kind::PrefillPack));
        CHECK(q.rows == (cap == 64 ? 60u : 120u));
        if (q.kind == Kind::Prefill) s.complete_prefill(); else s.complete_prefill_pack();
    }
    Scheduler s(cfg()); admit(s, 1, 150); admit(s, 2, 39);
    CHECK(step(s).kind == Kind::Prefill); // the 149-row segment is never packed
    std::printf("scheduler_c8_pack_test: decode cap 64 solo, 128 two-member pack; >64 segment ineligible\n");
}
static void held_membership_cancel() {
    Scheduler s(cfg()); for (unsigned i = 0; i < 8; ++i) admit(s, i + 1, 39);
    CHECK(step(s).kind == Kind::PrefillPack);
    cancel(s, s.slots()[3].key);
    const Key newcomer = admit(s, 9, 39); // slot 3 is reused, but its new generation did not share the W pack
    CHECK(newcomer.slot == 3);
    const Quantum q = s.next();
    CHECK(q.kind == Kind::PromptTailPack && q.count == 7);
    for (unsigned i = 0; i < q.count; ++i) CHECK(q.keys[i] != newcomer);
    std::printf("scheduler_c8_pack_test: canceled member removed; new grant cannot join held tail\n");
}
static void horizon_counts_all_pack_rows() {
    Scheduler s(cfg());
    admit(s, 1, 10000);
    for (unsigned i = 0; i < 7; ++i) admit(s, i + 2, 39);
    CHECK(s.horizon() == projected_horizon(s, true));
    CHECK(s.horizon() == s.boundary() + 4); // pack in flight; tail, decode, then long W covers NOTICE_ROWS
    CHECK(step(s).kind == Kind::PrefillPack);
    CHECK(s.horizon() == projected_horizon(s, true));
    CHECK(s.horizon() == s.boundary() + 3);
    // The horizon excludes the first in-flight W pack. The following 7-row tail plus one
    // 2041-row solo W must reach NOTICE_ROWS=2048 exactly; counting the tail as one does not.
    Config c = cfg(); c.prefill_rows = 2041;
    Scheduler exact(c); admit(exact, 1, 2042);
    for (unsigned i = 0; i < 7; ++i) admit(exact, i + 2, 39);
    CHECK(exact.horizon() == exact.boundary() + 4); // held tail, one decode round, then 2041-row W
    auto sim = exact.slots(); bool want = true; uint32_t run = 0;
    const Quantum pack = Scheduler::plan(c, sim, want);
    CHECK(pack.kind == Kind::PrefillPack && pack.count == 7);
    Scheduler::advance(sim, pack); Scheduler::alternate(c, pack.kind, want, run);
    CHECK(sim[0].deferred == 1); // long prefill was passed over by the projected W pack
    for (unsigned i = 1; i < 8; ++i) CHECK(sim[i].deferred == 0);
    const Quantum tail = Scheduler::plan(c, sim, want);
    CHECK(tail.kind == Kind::PromptTailPack && tail.count == 7);
    Scheduler::advance(sim, tail);
    CHECK(sim[0].deferred == 2); // and by its held tail; projection must match completion
    std::printf("scheduler_c8_pack_test: horizon counts packed W and all tail rows; projected deferred increments\n");
}
static void driver_and_dispatch() {
    Config c = cfg();
    unsigned called = 0; uint32_t total = 0;
    std::vector<s2::Step> seen;
    unsigned emitted = 0, prompted = 0;
    DriverHooks h;
    h.bind = [](const Grant&) {}; h.retire = [](const Slot&) {};
    h.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    h.run = [&](const s2::Step& st) { seen.push_back(st); return std::vector<int>(st.M(), 1); };
    h.prompt = [&](Key, int rows, int pos) { CHECK(rows == 1 && pos == 38); ++prompted; };
    h.emit = [&](Key, const int*, int n, int row) {
        if (emitted < 8) CHECK(n == 1 && row >= 0 && row < 8);
        else CHECK(n >= 1 && row >= 0 && row < 48);
        ++emitted; return Emission{uint32_t(n), false};
    };
    h.prefill_pack = [&](const std::vector<PrefillPart>& parts) {
        ++called; CHECK(parts.size() == 8);
        for (const auto& p : parts) total += p.rows;
    };
    Driver d(c, h);
    for (unsigned i = 0; i < 8; ++i) {
        Payload p; p.prompt.assign(39, int(i)); p.chunk = 4;
        const Request r{i + 1, 39, 16};
        const auto& st = d.state();
        Commit e; e.epoch = c.epoch; e.event = st.next_event(); e.boundary = st.boundary(); e.before_hash = st.digest();
        e.admit = Grant{st.next_key(), r, Run{0, grant_pages(c,r), 1, i + 11}};
        d.apply(e, &p);
    }
    CHECK(d.step().kind == Kind::PrefillPack && called == 1 && total == 304);
    CHECK(d.step().kind == Kind::PromptTailPack && prompted == 8 && emitted == 8);
    CHECK(seen.size() == 6); // one MAIN, one existing first-form DRAFT, four existing CHAINs at K=5
    for (size_t i = 0; i < seen.size(); ++i) {
        CHECK(!seen[i].scheduler_decode); // PromptTailPack remains on the main/prefill cut
        CHECK(seen[i].kind == (i == 0 ? s2::Step::MAIN : i == 1 ? s2::Step::DRAFT : s2::Step::CHAIN));
        CHECK(seen[i].M() == 8 && seen[i].sess.size() == 8);
        if (i < 2) CHECK(seen[i].event == s2::Step::TAIL);
        for (int j = 0; j < 8; ++j)
            CHECK(seen[i].row0[size_t(j)] == j && seen[i].rows[size_t(j)] == 1 && seen[i].ev[size_t(j)] == s2::Step::TAIL);
    }
    unsigned live_d = 0;
    for (const auto& s : d.state().slots()) if (s.phase != Phase::Free) { ++live_d; CHECK(s.phase == Phase::Decode && s.emitted == 1); }
    CHECK(live_d == 8);
    const size_t tail_steps = seen.size();
    CHECK(d.step().kind == Kind::Decode);
    CHECK(seen.size() > tail_steps);
    for (size_t i = tail_steps; i < seen.size(); ++i) CHECK(seen[i].scheduler_decode);

    Sessions sessions(c, "", false, 4, 2);
    std::vector<uint64_t> seq;
    DispatchHooks dh;
    dh.bucket = [](bool, int) { return 0; }; dh.mode = [](int, int) {};
    dh.launch = [](const Launch&) { return std::vector<int>(); };
    dh.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    dh.prefill_pack_seq = [&](const std::vector<PrefillPart>& parts, uint64_t n) { CHECK(parts.size() == 2); seq.push_back(n); };
    Dispatch dispatch(sessions, 48, dh);
    Slot slots[2]; Payload p[2];
    for (unsigned i = 0; i < 2; ++i) {
        slots[i].key = Key{1,i,1}; slots[i].req = Request{i+1,39,16}; slots[i].run = Run{0,1,1,i+1};
        slots[i].phase = Phase::Prefill; slots[i].packable = true; p[i].prompt.assign(39, int(i));
    }
    std::vector<PrefillPart> parts{{&slots[0], &p[0], 0, 38}, {&slots[1], &p[1], 0, 38}};
    dispatch.prefill_pack(parts); dispatch.prefill_pack(parts);
    CHECK(seq.size() == 2 && seq[0] == 1 && seq[1] == 2);
    CHECK(throws([&] { auto bad = parts; bad[1] = bad[0]; dispatch.prefill_pack(bad); }));
    CHECK(throws([&] { auto bad = parts; Slot dup = slots[1]; dup.req.id = 1; bad[1].slot = &dup; dispatch.prefill_pack(bad); }));
    CHECK(throws([&] { auto bad = parts; Slot lower = slots[1]; lower.req.id = 0; bad[1].slot = &lower; dispatch.prefill_pack(bad); }));
    CHECK(throws([&] { auto bad = parts; bad[1].rows = 37; dispatch.prefill_pack(bad); }));
    std::printf("scheduler_c8_pack_test: driver one W callback, one 8-row MAIN/DRAFT, four shared CHAINs; dispatch one sequence and negative slot/id/row guards\n");
}
static void e2e_request_zero_replay() {
    // F12C OFF-k6 E2E reference, eight exact prompt-token vectors (56..65 IDs).
    // A fresh native boot grants ordinals 0..7; all eight land at boundary 2.
    auto prompts = e2e_c8_prompts();
    Config c = cfg(); c.prefill_rows = 2048;
    Scheduler scheduler(c);
    std::array<Payload, 8> payload{};
    for (unsigned i = 0; i < 8; ++i) {
        payload[i].prompt = prompts[i]; payload[i].chunk = 4; payload[i].decode_prompt = false;
        const Request req{i, uint32_t(prompts[i].size()), 64};
        Commit e; e.epoch = c.epoch; e.event = scheduler.next_event(); e.boundary = scheduler.boundary();
        e.before_hash = scheduler.digest();
        e.admit = Grant{scheduler.next_key(), req, Run{0, grant_pages(c, req), 1, i + 1}};
        scheduler.apply(e);
    }
    const Quantum q = scheduler.next();
    CHECK(q.kind == Kind::PrefillPack && q.count == 8 && q.rows == 476);
    std::vector<PrefillPart> parts;
    for (unsigned i = 0; i < q.count; ++i) {
        const auto slot = q.keys[i].slot;
        parts.push_back({&scheduler.slots()[slot], &payload[slot], q.firsts[i], q.row_counts[i]});
    }
    uint64_t last_id = 0; std::array<bool, MAX_SESSIONS> seen{};
    for (const auto& p : parts) {
        const auto& s = *p.slot;
        const bool other = s.key.slot < MAX_SESSIONS && !seen[s.key.slot] && s.packable &&
            s.phase == Phase::Prefill && !s.pack_tail && !p.payload->decode_prompt &&
            p.payload->prompt.size() == s.req.prompt_tokens && p.first == 0 &&
            p.rows >= 1 && p.rows <= 64 && uint64_t(p.first) + p.rows + 1 == p.payload->prompt.size() &&
            uint64_t(p.first) + p.rows <= uint64_t(s.run.pages) * PAGE;
        std::printf("REPLAY request %llu slot %u prompt %zu rows %u id_ascending %d other_guards %d\n",
            (unsigned long long)s.req.id, s.key.slot, p.payload->prompt.size(), p.rows,
            int(s.req.id > last_id), int(other));
        CHECK(other);
        seen[s.key.slot] = true; last_id = s.req.id;
    }
    Sessions sessions(c, "", false, 4, 2);
    DispatchHooks hooks;
    hooks.bucket = [](bool, int) { return 0; }; hooks.mode = [](int, int) {};
    hooks.launch = [](const Launch&) { return std::vector<int>(); };
    hooks.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
    unsigned calls = 0;
    hooks.prefill_pack_seq = [&](const std::vector<PrefillPart>& got, uint64_t) { CHECK(got.size() == 8); ++calls; };
    Dispatch dispatch(sessions, c.max_rows, hooks);
    try { dispatch.prefill_pack(parts); }
    catch (const std::exception& e) { std::fprintf(stderr, "REPLAY dispatch rejected: %s\n", e.what()); CHECK(false); }
    CHECK(calls == 1);
    std::vector<Ds41PrefillChunk> chunks;
    for (const auto& p : parts)
        chunks.push_back(ds41_make_chunk(p.payload->prompt, p.first, p.rows, c.prefill_rows, 1,
                                         p.slot->req.id, p.slot->key.slot, p.slot->run));
    try { const auto pack = ds41_make_pack(chunks, c.prefill_rows, 8); CHECK(pack.layout.rows == 476); }
    catch (const std::exception& e) { std::fprintf(stderr, "REPLAY layout rejected: %s\n", e.what()); CHECK(false); }
}
static void mixed_flag_boot_refuses() {
    char temp[] = "/tmp/c8packbootXXXXXX";
    const char* root = ::mkdtemp(temp);
    CHECK(root != nullptr);
    if (!root) return;
    const std::string a = std::string(root) + "/a", b = std::string(root) + "/b";
    CHECK(::mkdir(a.c_str(), 0700) == 0 && ::mkdir(b.c_str(), 0700) == 0);
    const uint16_t port = uint16_t(25000 + ::getpid() % 10000);
    auto boot = [&](unsigned rank, bool pack, const std::string& dir) {
        BootstrapConfig bc; bc.scheduler = cfg(pack); bc.rank = rank; bc.ranks = 2;
        bc.timeout = 2; bc.head_ip = "127.0.0.1"; bc.port = port;
        bc.directory = dir; bc.runtime_signature = "same-weights";
        return bootstrap(bc);
    };
    std::string why;
    std::thread worker([&] { try { auto links = boot(1, false, b); } catch (...) {} });
    try { auto links = boot(0, true, a); }
    catch (const std::exception& e) { why = e.what(); }
    worker.join();
    CHECK(why.find("bootstrap config mismatch") != std::string::npos);
    std::printf("scheduler_c8_pack_test: mixed-flag boot refused: %s\n", why.c_str());
}
// r2-d4 ring crash (SESS=32): every W pack must have at most 8 members, and the second wave packs too.
static void s32_member_cap() {
    Config c = cfg(); c.max_sessions = 32; c.dynamic_mtp_off = true;
    Scheduler s(c);
    for (unsigned i = 0; i < 13; ++i) admit(s, i + 9, 39);
    const Quantum q = s.next();
    CHECK(q.kind == Kind::PrefillPack && q.count == PACK_MAX_MEMBERS && q.rows == 8 * 38);
    for (unsigned i = 1; i < q.count; ++i) CHECK(q.keys[i - 1].slot != q.keys[i].slot);
    s.complete_prefill_pack();
    const Quantum t = step(s); CHECK(t.kind == Kind::PromptTailPack && t.count == PACK_MAX_MEMBERS);
    bool second = false;
    for (int k = 0; k < 8 && !second; ++k) { const Quantum x = step(s); if (x.kind == Kind::PrefillPack) { second = true; CHECK(x.count == 5 && x.rows == 5 * 38); } }
    CHECK(second);
    std::printf("scheduler_c8_pack_test: SESS=32, 13 short prompts -> W pack of 8, tail of 8, then a W pack of 5\n");
}
int main() {
    s32_member_cap();
    code_pack_and_off(); class_totals_and_single(); cap_and_fallback(); held_membership_cancel(); horizon_counts_all_pack_rows();
    driver_and_dispatch(); e2e_request_zero_replay(); mixed_flag_boot_refuses();
    if (fails) std::fprintf(stderr, "scheduler_c8_pack_test: %d failures\n", fails);
    else std::printf("scheduler_c8_pack_test: PASS\n");
    return fails ? 1 : 0;
}
