// engine/scheduler-rs against the C++ it replaced (tests/ref/*_ref.h): seeded random inputs through both, every result
// and every refusal (exception type and message) the same. Host only.
#include "../kv_plan.h"
#include "../flash_lend.h"
#include "../flash_ring.h"
#include "../cp_rows.h"
#include "../scheduler.h"
#include "ref/kv_plan_ref.h"
#include "ref/flash_lend_ref.h"
#include "ref/flash_ring_ref.h"
#include "ref/cp_rows_ref.h"
#include "ref/scheduler_ref.h"
#include <cstdio>
#include <random>
#include <string>

static int fails = 0;
static long cases = 0;
#define CHECK(c) do { ++cases; if (!(c)) { if (++fails < 20) std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); } } while (0)

// a call's outcome: its value printed, or the exception's type and message
template <class F> static std::string outcome(F f)
{
    try { return "ok " + f(); }
    catch (const std::out_of_range& e) { return std::string("out_of_range ") + e.what(); }
    catch (const std::runtime_error& e) { return std::string("runtime_error ") + e.what(); }
    catch (const std::exception& e) { return std::string("exception ") + e.what(); }
}
template <class T> static std::string str(const std::vector<T>& v)
{
    std::string s;
    for (const T& x : v) s += std::to_string(x) + ",";
    return s;
}
static std::string str(const std::set<int>& v) { return str(std::vector<int>(v.begin(), v.end())); }


// the scheduler's state, field by field (padding bytes differ)
static std::string str(const g6s2::Key& k) { return std::to_string(k.epoch) + "." + std::to_string(k.slot) + "." + std::to_string(k.generation); }
static std::string str(const g6s2::Run& r)
{
    return std::to_string(r.first_page) + "+" + std::to_string(r.pages) + "v" + std::to_string(r.table_version) + "#" + std::to_string(r.list_hash);
}
static std::string str(const g6s2::Quantum& q)
{
    std::string s = std::to_string(int(q.kind)) + "@" + std::to_string(q.boundary) + " n" + std::to_string(q.count) + " " +
                    std::to_string(q.first) + "+" + std::to_string(q.rows) + ":";
    for (unsigned i = 0; i < g6s2::MAX_SESSIONS; ++i)
        s += str(q.keys[i]) + "/" + std::to_string(q.firsts[i]) + "/" + std::to_string(q.row_counts[i]) + ",";
    return s;
}
template <class S> static std::string state(const S& x)
{
    std::string s = std::to_string(x.boundary()) + " " + std::to_string(x.next_event()) + " " + std::to_string(x.revision()) + " " +
                    std::to_string(x.busy()) + " " + std::to_string(x.resident()) + " " + std::to_string(x.retiring()) + " " +
                    std::to_string(x.parked()) + " h" + std::to_string(x.horizon()) + " d" + std::to_string(x.digest()) + "|";
    for (const auto& t : x.slots())
        s += str(t.key) + " " + std::to_string(t.req.id) + " " + std::to_string(t.req.prompt_tokens) + " " + std::to_string(t.req.max_tokens) +
             " " + std::to_string(t.req.steering.enabled) + " " + std::to_string(t.req.r_rows) + " " + str(t.run) + " " +
             std::to_string(int(t.phase)) + " " + std::to_string(t.prefilled) + " " + std::to_string(t.emitted) + " " +
             std::to_string(t.inserted) + " " + std::to_string(t.deferred) + " " + std::to_string(t.packable) +
             std::to_string(t.pack_tail) + " " + std::to_string(t.parked) + ";";
    return s;
}

// both schedulers through one random command stream: every call's outcome and the whole state after it the same
static void scheduler_diff(std::mt19937_64& rng)
{
    auto U = [&](int lo, int hi) { return int(std::uniform_int_distribution<int>(lo, hi)(rng)); };
    using namespace g6s2;
    for (int t = 0; t < 600; ++t)
    {
        Config c;
        c.epoch = uint64_t(U(0, 20) ? U(1, 3) : 0);
        c.pool_rows = uint32_t(U(0, 20) ? PAGE * U(8, 400) : U(0, 9000));
        c.max_rows = uint32_t(U(1, 40));
        c.max_sessions = uint32_t(U(0, int(MAX_SESSIONS) + 1));
        c.mtp = uint32_t(U(0, 4));
        c.guard_tokens = U(0, 3) ? UINT32_MAX : uint32_t(U(0, 16));
        c.prefill_rows = uint32_t(U(0, 20) ? U(1, 16) * 64 : 0);
        c.two_lanes = U(0, 1);
        c.decode_prefill_rows = uint32_t(U(0, 2) ? 0 : U(0, 3) ? 64 * U(1, 2) : U(1, 100));
        c.decode_rounds = uint32_t(U(0, 2) ? 0 : U(0, 70));
        c.window_pages = uint32_t(U(0, 2) ? 0 : U(1, 4));
        c.cp_step_rows = uint32_t(U(0, 2) ? 0 : U(1, 12));
        c.cp_after = U(0, 1) ? UINT64_MAX : uint64_t(U(0, 4096));
        c.cp_seg = !U(0, 3);
        c.r2_rows = uint32_t(U(0, 2) ? 0 : U(0, 4) ? 64 * U(1, 4) : U(1, 100));
        c.tail_first = U(0, 1);
        c.notice_max = uint32_t(U(0, 4) ? 8 : U(0, 10));
        c.c8_pack = U(0, 1) && (U(0, 6) || g6mod::has(G6_CAP_PACK));
        c.mtp_sessions = uint32_t(U(0, 2) ? 0 : U(1, 4));
        c.park = U(0, 1) && (U(0, 6) || g6mod::has(G6_CAP_PARK));
        const std::string va = outcome([&] { c.validate(); return std::string(); });
        const std::string vr = outcome([&] { g6s2ref::validate(c); return std::string(); });
        CHECK(va == vr);
        if (va != "ok ") continue;
        Scheduler a(c);
        g6s2ref::Scheduler r(c);
        uint64_t id = 0;
        Quantum work;   // the last quantum next() returned
        for (int k = 0; k < 300; ++k)
        {
            const int op = U(0, 9);
            std::string oa, orf;
            auto both = [&](auto fa, auto fr) { oa = outcome(fa); orf = outcome(fr); };
            if (op <= 2)
            {
                // a commit: maybe a cancel, a grant at the next key, a shrink; the digest right or wrong
                Commit e;
                e.epoch = U(0, 30) ? c.epoch : c.epoch + 1;
                e.event = r.next_event() - uint64_t(!U(0, 30));
                e.boundary = r.boundary() + uint64_t(!U(0, 30));
                e.before_hash = U(0, 30) ? r.digest() : 7;
                const auto& sl = r.slots();
                if (!U(0, 3)) e.cancel = sl[size_t(U(0, int(MAX_SESSIONS) - 1))].key;
                if (U(0, 3))
                {
                    std::optional<unsigned> park;
                    if (c.park && !U(0, 2)) park = unsigned(U(0, int(MAX_SESSIONS)));
                    Key key{};
                    const std::string ka = outcome([&] { key = a.next_key(e.cancel, park); return str(key); });
                    const std::string kr = outcome([&] { return str(r.next_key(e.cancel, park)); });
                    CHECK(ka == kr);
                    if (ka.rfind("ok", 0) == 0)
                    {
                        Grant g;
                        g.key = key;
                        if (!U(0, 20)) g.key.generation++;
                        g.req.id = U(0, 20) ? (id += uint64_t(U(1, 3))) : id;
                        g.req.prompt_tokens = uint32_t(U(0, 30) ? (U(0, 3) ? U(1, 400) : U(1, 6000)) : 0);
                        g.req.max_tokens = uint32_t(U(0, 30) ? U(1, 60) : 0);
                        g.req.steering.enabled = !U(0, 8);
                        if (g.req.steering.enabled) g.req.steering.profile_digest[size_t(U(0, 31))] = uint8_t(U(0, 255));
                        g.req.r_rows = U(0, 4) ? 0 : c.r2_rows;
                        const std::string pa = outcome([&] { return std::to_string(grant_pages(c, g.req)) + "/" + std::to_string(reserve_pages(c, g.req)); });
                        const std::string pr = outcome([&] { return std::to_string(g6s2ref::grant_pages(c, g.req)) + "/" + std::to_string(g6s2ref::reserve_pages(c, g.req)); });
                        CHECK(pa == pr);
                        uint32_t pages = 1;
                        try { pages = U(0, 1) ? grant_pages(c, g.req) : reserve_pages(c, g.req) + (U(0, 2) ? 0 : c.window_pages); } catch (...) {}
                        if (!U(0, 20)) pages += 1;
                        g.run = Run{uint32_t(U(0, int(c.pool_rows / PAGE))), pages, uint32_t(U(0, 20) ? U(1, 9) : 0), uint64_t(U(0, 1 << 20))};
                        g.reuse = uint32_t(U(0, 2) ? 0 : U(0, int(g.req.prompt_tokens) + 1));
                        g.restored = uint32_t(U(0, 3) ? 0 : U(1, 3));
                        g.counted = uint32_t(U(0, 2));
                        CHECK(outcome([&] { return std::to_string(grant_pages_ok(c, g.req, pages)); }) ==
                              outcome([&] { return std::to_string(g6s2ref::grant_pages_ok(c, g.req, pages)); }));
                        e.admit = g;
                    }
                }
                if (c.r2_rows && !U(0, 3))
                {
                    const auto& x = sl[size_t(U(0, int(MAX_SESSIONS) - 1))];
                    e.shrink = Shrink{x.key, U(0, 8) ? c.r2_rows : 0, Run{x.run.first_page + uint32_t(!U(0, 4)), x.run.pages, x.run.table_version, 99}};
                }
                e.hold = U(0, 1);
                auto changes = [](const Changes& d) {
                    std::string s;
                    for (const auto& x : d.freed) s += str(x) + ",";
                    return s + "|" + (d.admitted ? str(d.admitted->key) : "-");
                };
                both([&] { return changes(a.apply(e)); }, [&] { return changes(r.apply(e)); });
            }
            else if (op == 3)
            {
                std::array<bool, MAX_SESSIONS> stalled{};
                for (auto& x : stalled) x = !U(0, 5);
                both([&] { const auto q = a.next(stalled); if (q.kind != Kind::Idle) work = q; return str(q); }, [&] { return str(r.next(stalled)); });
                for (int w = 0; w < 2; ++w)
                {
                    const bool want = U(0, 1);
                    CHECK(str(Scheduler::plan(c, a.slots(), want, stalled)) == str(g6s2ref::Scheduler::plan(c, r.slots(), want, stalled)));
                    auto sa = a.slots();
                    auto sr = r.slots();
                    const Quantum q = g6s2ref::Scheduler::plan(c, sr, want);
                    Scheduler::advance(sa, q);
                    g6s2ref::Scheduler::advance(sr, q);
                    bool wa = want, wr = want;
                    uint32_t ra = uint32_t(U(0, 70)), rr = ra;
                    Scheduler::alternate(c, q.kind, wa, ra);
                    g6s2ref::Scheduler::alternate(c, q.kind, wr, rr);
                    CHECK(wa == wr && ra == rr);
                    std::string la, lr;
                    for (unsigned i = 0; i < MAX_SESSIONS; ++i)
                    {
                        la += std::to_string(sa[i].prefilled) + std::to_string(int(sa[i].phase)) + std::to_string(sa[i].deferred) + std::to_string(sa[i].pack_tail) + ",";
                        lr += std::to_string(sr[i].prefilled) + std::to_string(int(sr[i].phase)) + std::to_string(sr[i].deferred) + std::to_string(sr[i].pack_tail) + ",";
                        for (unsigned j = 0; j < MAX_SESSIONS; ++j)
                        {
                            CHECK(prefill_first(sa[i], sa[j]) == g6s2ref::prefill_first(sa[i], sa[j]));
                            CHECK(prefill_left(sa[i]) == g6s2ref::prefill_left(sa[i]));
                        }
                    }
                    CHECK(la == lr);
                }
            }
            else if (op <= 6)
            {
                // complete the quantum in flight, mostly with valid completions
                const auto& sl = r.slots();
                auto comps = [&](unsigned n, const auto& keys) {
                    std::vector<Completion> v;
                    for (unsigned i = 0; i < n + unsigned(!U(0, 30)); ++i)
                    {
                        const Key key = U(0, 30) ? keys[i % MAX_SESSIONS] : sl[size_t(U(0, int(MAX_SESSIONS) - 1))].key;
                        const auto& x = sl[key.slot % MAX_SESSIONS];
                        const uint32_t left = x.req.max_tokens - x.emitted;
                        v.push_back({key, uint32_t(U(0, 30) ? U(0, int(std::min(left, 4u))) : left + 1), !U(0, 6),
                                     x.inserted + uint32_t(U(0, 30) ? U(0, 2) : 0) - uint32_t(!U(0, 40))});
                    }
                    return v;
                };
                const int pick = U(0, 7) ? int(work.kind) : U(1, 5);
                if (pick == int(Kind::Prefill)) both([&] { a.complete_prefill(); return std::string(); }, [&] { r.complete_prefill(); return std::string(); });
                else if (pick == int(Kind::PrefillPack))
                    both([&] { a.complete_prefill_pack(); return std::string(); }, [&] { r.complete_prefill_pack(); return std::string(); });
                else
                {
                    const auto v = comps(work.count, work.keys);
                    if (pick == int(Kind::PromptTail) && !v.empty())
                        both([&] { a.complete_tail(v[0]); return std::string(); }, [&] { r.complete_tail(v[0]); return std::string(); });
                    else if (pick == int(Kind::PromptTailPack))
                        both([&] { a.complete_tail_pack(v); return std::string(); }, [&] { r.complete_tail_pack(v); return std::string(); });
                    else both([&] { a.complete_decode(v); return std::string(); }, [&] { r.complete_decode(v); return std::string(); });
                }
            }
            else if (op == 7)
            {
                const auto& x = r.slots()[size_t(U(0, int(MAX_SESSIONS) - 1))];
                const Run g{x.run.first_page + uint32_t(!U(0, 10)), x.run.pages + uint32_t(U(0, 3)), x.run.table_version, uint64_t(U(0, 99))};
                both([&] { a.grow(x.key, g); return std::string(); }, [&] { r.grow(x.key, g); return std::string(); });
            }
            else if (op == 8)
            {
                const auto& x = r.slots()[size_t(U(0, int(MAX_SESSIONS) - 1))];
                both([&] { a.skip_prefill(x.key); return std::string(); }, [&] { r.skip_prefill(x.key); return std::string(); });
            }
            else
            {
                const uint64_t to = r.boundary() + uint64_t(U(0, 3)) - uint64_t(!U(0, 6));
                both([&] { a.idle_boundary(to); return std::string(); }, [&] { r.idle_boundary(to); return std::string(); });
                const uint64_t pd = uint64_t(U(0, 9));
                a.pool_digest(pd);
                r.pool_digest(pd);
                const uint64_t want = uint64_t(U(0, 40));
                const auto fa = a.find(want), fr = r.find(want);
                CHECK(fa.has_value() == fr.has_value() && (!fa || *fa == *fr));
                std::optional<Key> cancel;
                if (U(0, 1)) cancel = r.slots()[size_t(U(0, int(MAX_SESSIONS) - 1))].key;
                std::string ra, rr;
                for (const auto& x : a.reclaimable(cancel)) ra += str(x) + ",";
                for (const auto& x : r.reclaimable(cancel)) rr += str(x) + ",";
                CHECK(ra == rr);
                std::string da, dr;
                for (const auto& x : a.decoding()) da += str(x) + ",";
                for (const auto& x : r.decoding()) dr += str(x) + ",";
                CHECK(da == dr);
                if (!U(0, 200)) { a.fail(); r.fail(); }
            }
            CHECK(oa == orf);
            if (oa != orf && fails < 20) std::printf("  op %d: %s | %s\n", op, oa.c_str(), orf.c_str());
            const std::string xa = state(a), xr = state(r);
            CHECK(xa == xr);
            if (xa != xr) break;   // diverged: the rest of the stream says nothing more
        }
    }
}

int main()
{
    std::mt19937_64 rng(20261105);
    auto U = [&](int lo, int hi) { return int(std::uniform_int_distribution<int>(lo, hi)(rng)); };
    // kv_plan: copy lists in runs of one kind, a few tiers and slots, parts and subs, and every batch to the end
    for (int t = 0; t < 4000; ++t)
    {
        std::vector<g6kv_io> io;
        const int n = U(0, 40);
        uint32_t kind = uint32_t(U(0, 2));
        for (int k = 0; k < n; ++k)
        {
            if (!U(0, 6)) kind = uint32_t(U(0, 2));
            io.push_back({kind, uint32_t(U(0, 30)), uint32_t(U(0, 2)), uint32_t(U(0, 2)), uint64_t(U(0, 6)), uint64_t(U(0, 6)),
                          uint32_t(U(0, 2)), uint32_t(U(0, 3))});
        }
        const int batch = U(-1, 8), items_cap = U(-1, 12);
        const uint32_t per_slot = uint32_t(U(0, 4));
        size_t i1 = 0, i2 = 0;
        for (int guard = 0; guard < 64 && (i1 < io.size() || guard == 0); ++guard)
        {
            const std::string a = outcome([&] {
                const auto b = g6s2::kvt::next_batch(io, i1, batch, per_slot, items_cap);
                std::string s = std::to_string(b.kind) + "|" + str(b.from) + "|" + str(b.to) + "|" + str(b.slots) + "|" + str(b.to_slots) + "|";
                for (const auto& x : b.items) s += std::to_string(x.page) + ":" + std::to_string(x.part) + ":" + std::to_string(x.sub) + ":" + std::to_string(x.hslot) + ",";
                return s;
            });
            const std::string r = outcome([&] {
                const auto b = g6s2ref::kvt::next_batch(io, i2, batch, per_slot, items_cap);
                std::string s = std::to_string(b.kind) + "|" + str(b.from) + "|" + str(b.to) + "|" + str(b.slots) + "|" + str(b.to_slots) + "|";
                for (const auto& x : b.items) s += std::to_string(x.page) + ":" + std::to_string(x.part) + ":" + std::to_string(x.sub) + ":" + std::to_string(x.hslot) + ",";
                return s;
            });
            CHECK(a == r && i1 == i2);
            if (a != r || a.rfind("ok", 0) != 0) break;
        }
    }
    // flash_lend: settings, then polls at rising times with idle on and off, and admissions
    for (int t = 0; t < 2000; ++t)
    {
        g6s2::FlashLend a;
        g6s2ref::FlashLend r;
        a.idle_ms = r.idle_ms = U(0, 3) ? U(0, 50) : 0;
        a.keep = r.keep = uint32_t(U(0, 5));
        a.chunk_rows = r.chunk_rows = uint64_t(U(0, 3) ? U(1, 4096) : 0);
        a.chunks = r.chunks = uint32_t(U(0, 8));
        auto now = std::chrono::steady_clock::now();
        for (int k = 0; k < 60; ++k)
        {
            now += std::chrono::microseconds(U(0, 20000));
            const bool idle = U(0, 4) != 0;
            const auto pa = a.poll(idle, now), pr = r.poll(idle, now);
            CHECK(pa == pr);
            const uint64_t rows = uint64_t(U(0, 40000));
            const uint32_t room = uint32_t(U(0, 9));
            CHECK(a.need(rows) == r.need(rows) && a.recall(rows, room) == r.recall(rows, room));
        }
    }
    // flash_ring: steps that follow the accepted rows, and steps that do not; the capture set over its arguments
    for (int t = 0; t < 3000; ++t)
    {
        const int sessions = U(1, 4), per = U(1, 5);
        g6s2::FlashSlots a(sessions, per);
        g6s2ref::FlashSlots r(sessions, per);
        std::vector<int> pos(size_t(sessions), 100);
        for (int k = 0; k < 40; ++k)
        {
            const int q = U(-1, sessions), n = U(0, per + 1);
            const bool verify = U(0, 1);
            if (q >= 0 && q < sessions && U(0, 3)) pos[size_t(q)] += U(0, per + 1); else if (q >= 0 && q < sessions) pos[size_t(q)] += U(-3, 9);
            const int p = q >= 0 && q < sessions ? pos[size_t(q)] : 0;
            int ka[16] = {}, kr[16] = {};
            const std::string oa = outcome([&] { return std::to_string(a.rows(q, p, n, verify, ka)); });
            const std::string orr = outcome([&] { return std::to_string(r.rows(q, p, n, verify, kr)); });
            CHECK(oa == orr && std::equal(ka, ka + 16, kr));
            if (!U(0, 9) && q >= 0 && q < sessions) { a.reset(q); r.reset(q); }
        }
        const int s2 = U(0, 9), mtp = U(0, 7), chunk = U(0, 40), mx = U(8, 48);
        const bool adapt = U(0, 1);
        CHECK(outcome([&] { const auto f = g6s2::flash_shapes(s2, mtp, chunk, adapt, mx); return str(f.mains) + "|" + str(f.drafts) + "|" + str(f.chains); }) ==
              outcome([&] { const auto f = g6s2ref::flash_shapes(s2, mtp, chunk, adapt, mx); return str(f.mains) + "|" + str(f.drafts) + "|" + str(f.chains); }));
    }
    // cp_rows: the CP boundary, the boot value, and rounds over cycles with sessions past R and below it
    for (int t = 0; t < 3000; ++t)
    {
        std::vector<int> bk;
        int v = 0;
        for (int k = U(1, 8); k > 0; --k) bk.push_back(v += U(1, 4096));
        const int R = U(-2, 20000);
        CHECK(g6s2::cp_after(bk.data(), int(bk.size()), R) == g6s2ref::cp_after(bk.data(), int(bk.size()), R));
        const int64_t val = U(0, 3) ? U(-5, 40) : 0;
        const int solo = U(1, 12), full = U(1, 40);
        std::string ea, er;
        CHECK(g6s2::cp_step_rows(val, solo, full, &ea) == g6s2ref::cp_step_rows(val, solo, full, &er) && ea == er);
        g6s2::CpRound a;
        g6s2ref::CpRound r;
        const uint32_t rows = uint32_t(U(0, 3) ? U(1, 30) : 0), per = uint32_t(U(0, 8));
        const bool seg = U(0, 1);
        for (int k = 0; k < 30; ++k)
        {
            std::vector<g6s2::CpCand> ca;
            std::vector<g6s2ref::CpCand> cr;
            for (int j = U(0, 8), id = 0; j > 0; --j) { id += U(1, 3); const bool past = U(0, 2) == 0; ca.push_back({uint64_t(id), past}); cr.push_back({uint64_t(id), past}); }
            const auto pa = a.pick(ca, rows, per, seg);
            const auto pr = r.pick(cr, rows, per, seg);
            CHECK(pa.in == pr.in && pa.capped == pr.capped && pa.fresh == pr.fresh);
            if (U(0, 4)) { a.commit(pa, ca); r.commit(pr, cr); }
            CHECK(a.cycle() == r.cycle());
        }
    }
    scheduler_diff(rng);
    std::printf("scheduler_rs_diff_test: %s (%ld comparisons)\n", fails ? "FAIL" : "PASS", cases);
    return fails ? 1 : 0;
}
