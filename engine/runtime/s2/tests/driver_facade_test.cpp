// The C++ facade over the Rust driver (driver.h Driver, engine/scheduler-rs src/driver.rs): a hook's exception comes
// out of apply() or step() as itself and fails the scheduler, as do the driver's own refusals after the scheduler
// changed; a refusal before it leaves the state as it was. Hooks may read the driver back (the planner's own getters
// only from the driver's hooks) but not call apply(), step(), idle_boundary() or pool_digest(); a hook that calls
// apply() and catches the refusal leaves the outer apply() whole. The scheduler's state is the Rust's, read through
// state(), which stays the same object when the Driver moves. Host only.
#include "s2/driver.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

struct Boom : std::exception { const char* what() const noexcept override { return "boom"; } };

// a driver whose hooks each do what the test sets
struct Fix {
    Config c;
    std::function<void()> on_bind, on_run, on_emit, on_retire;
    uint32_t emit_extra = 0;   // the emit hook reports this many tokens more than it got
    std::unique_ptr<Driver> d;
    uint64_t event = 0;
    Fix() {
        c.max_sessions = 2;
        DriverHooks h;
        h.bind = [this](const Grant&) { if (on_bind) on_bind(); };
        h.retire = [this](const Slot&) { if (on_retire) on_retire(); };
        h.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
        h.run = [this](const s2::Step& s) { if (on_run) on_run(); return std::vector<int>(s.M(), 1); };
        h.emit = [this](Key, const int*, int n, int) { if (on_emit) on_emit(); return Emission{uint32_t(n) + emit_extra, false}; };
        d = std::make_unique<Driver>(c, h);
    }
    Key admit(uint64_t id, uint32_t max_tokens = 64) {
        const Key k = d->state().next_key();
        const Request r{id, 4, max_tokens, {}, 0};
        const Commit cm{c.epoch, ++event, d->state().boundary(), d->state().digest(), {}, Grant{k, r, {0, 1, 1}, 0}, false, {}, {}};
        Payload p{{5, 6, 7, 8}, 4, false, {}, {}, {}, {}};
        d->apply(cm, &p);
        return k;
    }
    void cancel(Key k) {
        const Commit cm{c.epoch, ++event, d->state().boundary(), d->state().digest(), k, {}, false, {}, {}};
        d->apply(cm);
    }
    // steps until a decode round
    void to_decode() { for (int i = 0; i < 8 && d->step().kind != Kind::Decode; ++i) {} }
};
template <class E, class F> static bool throws(F f, const char* what = nullptr) {
    try { f(); } catch (const E& e) { return !what || std::string(e.what()) == what; } catch (...) { return false; }
    return false;
}
static bool failed(Driver& d) { return throws<std::runtime_error>([&] { d.step(); }, "scheduler failed or work in flight"); }

int main() {
    {   // a hook's exception comes out of step() as itself and fails the scheduler
        Fix f;
        f.admit(1);
        f.to_decode();
        f.on_run = [] { throw Boom(); };
        CHECK(throws<Boom>([&] { f.d->step(); }));
        CHECK(failed(*f.d));
    }
    {   // the same from apply(): the bind hook
        Fix f;
        f.on_bind = [] { throw Boom(); };
        CHECK(throws<Boom>([&] { f.admit(1); }));
        CHECK(failed(*f.d));
    }
    {   // the driver's own refusal inside a round (the emit hook's count) fails it too
        Fix f;
        f.admit(1);
        f.to_decode();
        f.emit_extra = 1;
        CHECK(throws<std::runtime_error>([&] { f.d->step(); }, "bad emitter count"));
        CHECK(failed(*f.d));
    }
    {   // a refusal before the scheduler changes leaves it as it was
        Fix f;
        const auto before = f.d->state().digest();
        const Request r{1, 4, 64, {}, 0};
        const Commit cm{f.c.epoch, 1, 0, before, {}, Grant{f.d->state().next_key(), r, {0, 1, 1}, 0}, false, {}, {}};
        Payload p{{5, 6, 7}, 4, false, {}, {}, {}, {}};
        CHECK(throws<std::runtime_error>([&] { f.d->apply(cm, &p); }, "admit payload mismatch"));
        CHECK(throws<std::runtime_error>([&] { f.d->apply(Commit{f.c.epoch, 1, 0, before, {}, {}, false, {}, {}}, &p); }, "payload without grant"));
        CHECK(f.d->state().digest() == before);
        f.admit(1);
        f.to_decode();
    }
    {   // a planner hook reads what the driver keeps; apply() and step() from a hook are refused
        Fix f;
        const Key k = f.admit(1);
        f.to_decode();
        size_t prompt = 0;
        bool decoding = false;
        f.on_emit = [&] { prompt = f.d->prompt(k.slot).size(); decoding = f.d->state().slots()[k.slot].phase == Phase::Decode; };
        f.d->step();
        CHECK(prompt == 4 && decoding);
        f.on_emit = [&] { f.d->step(); };
        CHECK(throws<std::runtime_error>([&] { f.d->step(); }, "s2: a driver hook called the driver back"));
        CHECK(failed(*f.d));
    }
    {   // the retire hook reads the cancelled session's tokens; the slot is clean after it
        Fix f;
        const Key k = f.admit(1);
        f.to_decode();
        f.d->step();
        const size_t n = f.d->emitted(k.slot).size();
        size_t at_retire = 0;
        f.on_retire = [&] { at_retire = f.d->emitted(k.slot).size(); };
        f.cancel(k);
        CHECK(n > 0 && at_retire == n && f.d->emitted(k.slot).empty() && f.d->prompt(k.slot).empty());
        CHECK(f.d->stats_if_initialized(f.admit(2)) == nullptr);
    }
    {   // the retire hook calls apply() and catches the refusal: the outer commit (a cancel and a grant) still admits with
        // its own payload
        Fix f;
        const Key k = f.admit(1);
        f.to_decode();
        std::string refused;
        f.on_retire = [&] {
            const Request r{9, 4, 64, {}, 0};
            const Commit inner{f.c.epoch, f.event + 1, f.d->state().boundary(), f.d->state().digest(), {},
                               Grant{f.d->state().next_key(), r, {0, 1, 1}, 0}, false, {}, {}};
            Payload p{{1, 1, 1, 1}, 4, false, {}, {}, {}, {}};
            try { f.d->apply(inner, &p); } catch (const std::runtime_error& e) { refused = e.what(); }
        };
        const Key k2 = f.d->state().next_key(k);
        const Request r{2, 4, 64, {}, 0};
        const Commit cm{f.c.epoch, ++f.event, f.d->state().boundary(), f.d->state().digest(), k, Grant{k2, r, {0, 1, 1}, 0},
                        false, {}, {}};
        Payload p{{9, 8, 7, 6}, 4, false, {}, {}, {}, {}};
        const Changes ch = f.d->apply(cm, &p);
        CHECK(refused == "s2: a driver hook called the driver back");
        CHECK(ch.admitted && ch.admitted->key == k2 && f.d->prompt(k2.slot) == p.prompt);
        f.on_retire = nullptr;
        f.to_decode();
        f.d->step();
        CHECK(!f.d->emitted(k2.slot).empty() && f.d->stats(k2).rounds >= 1);
    }
    {   // a hook that sets the pool digest or an idle boundary is refused, and the state is as it was
        Fix f;
        const Key k = f.admit(1);
        f.to_decode();
        std::string pool, idle;
        uint64_t before = 0, after = 0;
        f.on_emit = [&] {
            before = f.d->state().digest();
            try { f.d->pool_digest(7); } catch (const std::runtime_error& e) { pool = e.what(); }
            try { f.d->idle_boundary(f.d->state().boundary() + 1); } catch (const std::runtime_error& e) { idle = e.what(); }
            after = f.d->state().digest();
        };
        f.d->step();
        CHECK(pool == "s2: a driver hook called the driver back" && idle == pool && before == after);
        f.on_emit = nullptr;
        f.d->step();
        CHECK(f.d->stats(k).rounds >= 2);
        f.cancel(k);   // the engine is idle
        const uint64_t d0 = f.d->state().digest(), b0 = f.d->state().boundary();
        f.d->pool_digest(42);
        CHECK(f.d->state().digest() != d0);
        f.d->idle_boundary(b0 + 3);
        CHECK(f.d->state().boundary() == b0 + 3);
        f.d->fail();
        CHECK(failed(*f.d));
    }
    {   // a moved Driver keeps its hooks and state, and state() is the same view
        Fix f;
        const Key k = f.admit(1);
        const SchedulerView& v = f.d->state();
        Driver moved = std::move(*f.d);
        f.d.reset();
        CHECK(&v == &moved.state());
        for (int i = 0; i < 8 && moved.step().kind != Kind::Decode; ++i) {}
        moved.step();
        CHECK(!moved.emitted(k.slot).empty() && moved.stats(k).rounds >= 1);
        CHECK(v.slots()[k.slot].phase == Phase::Decode && v.slots()[k.slot].emitted > 0 && v.resident() == 1);
    }
    if (fails) { std::fprintf(stderr, "driver_facade_test: %d FAIL\n", fails); return 1; }
    std::printf("driver_facade_test: PASS\n");
    return 0;
}
