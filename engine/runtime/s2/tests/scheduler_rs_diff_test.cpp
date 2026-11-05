// engine/scheduler-rs against the C++ it replaced (tests/ref/*_ref.h): seeded random inputs through both, every result
// and every refusal (exception type and message) the same. Host only.
#include "../kv_plan.h"
#include "../flash_lend.h"
#include "../flash_ring.h"
#include "../cp_rows.h"
#include "ref/kv_plan_ref.h"
#include "ref/flash_lend_ref.h"
#include "ref/flash_ring_ref.h"
#include "ref/cp_rows_ref.h"
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
    std::printf("scheduler_rs_diff_test: %s (%ld comparisons)\n", fails ? "FAIL" : "PASS", cases);
    return fails ? 1 : 0;
}
