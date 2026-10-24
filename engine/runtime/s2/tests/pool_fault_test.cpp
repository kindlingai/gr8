// --kv-store-fail soft (s2/kv_fault.h) over the pool's block stores (s2/pool.h, kvman-rs src/stash.rs tier_dead), in
// lock-step on two ranks as in pool_stash_test: a store call fails on one rank (an injected failure, or a real KV node
// killed or frozen under it). That rank stops calling the tier; the ranks OR their failed tiers at the barrier after each
// drain (service.h settle) and both drop them (RunPool::tier_dead, with 0 when none failed), and a grant whose restores
// were among those copies prefills from row 0 again (driver.h rebind). After every grant, every row the grant does not
// prefill must hold the prompt's token on both ranks; the ranks must grant and copy alike, agree on the digest, and never
// copy to or from a dead tier. Host only.
//   pool_fault_test SPEC [SPEC ..]          (default mem:) seeded injected failures, the store under test as tier 1
//   pool_fault_test --kill PID remote:H:P   rank 1's tier 1 is that KV node; the test kills it (SIGKILL) mid-run
//   pool_fault_test --stop PID remote:H:P   the same, frozen (SIGSTOP) under a 1 s store timeout
#include "s2/pool.h"
#include "s2/kv_fault.h"
#include <signal.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
using namespace g6s2;

static long fails = 0;
static const char* spec = "";
static uint64_t seed = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 20) std::printf("FAIL %s seed %llu: %s\n", spec, (unsigned long long)seed, #c); ++fails; } } while (0)

struct Rng {
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    int range(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
    bool one_in(int n) { return next() % uint64_t(n) == 0; }
};

static bool injected = false;
static const char* fault_error() { return injected ? "injected fault" : g6kv_tier_error(); }

using Page = std::array<int32_t, 64>;
struct Tier {
    g6kv_tier* t = nullptr;
    std::vector<uint8_t> buf;
    Tier(const char* spec, const std::string& name, uint64_t blocks) {
        require(g6kv_tier_open(spec, name.c_str(), sizeof(Page), blocks, &t) == 0, g6kv_tier_error());
        buf.resize(g6kv_tier_block_bytes(t));
    }
    ~Tier() { g6kv_tier_free(t); }
};
struct Rank {
    RunPool pool;
    std::vector<Page> dev;
    std::vector<std::unique_ptr<Tier>> tiers;
    kvt::Faults faults;
    long calls[2] = {}, fail_at[2] = {-1, -1};   // fail_at[t]: tier t's call that fails here (an injected fault)
    Rank(int rank, uint32_t pages, uint32_t place, uint32_t below, uint64_t slots, const char* spec, const std::string& name)
        : pool(pages, place, below), dev(pages) {
        g6rt::PrefixReuse rules; rules.strict = false; rules.noend = true; rules.grid = 128;
        pool.keep(rules); pool.stores(2);
        tiers.push_back(std::make_unique<Tier>("mem:", name + "-t0", 4));
        tiers.push_back(std::make_unique<Tier>(spec, name + "-t1", slots));
        pool.add_tier(4, 0, "", "");
        pool.add_tier(slots, 2, "", "");
        pool.policy(1, 0, 1, sizeof(Page));
        pool.write_limit(500);
        pool.tail(2, 1);
        faults.soft = true; faults.rank = rank;
        for (auto& p : dev) p.fill(-1);
    }
    // one store call through the fault policy, as KvTiers does (kv_tier.cuh): true when it ran
    template <class F> bool call(uint32_t t, const char* what, F&& f) {
        return faults.call(t, what, [&] {
            injected = ++calls[t] == fail_at[t];
            return injected ? -1 : f();
        }, fault_error);
    }
    // the engine's part: every copy, in order, then io_done. A copy of a tier that failed here is skipped: a spill or a
    // move leaves the store wrong, a restore leaves the device page wrong
    std::vector<g6kv_io> run_io() {
        const auto io = pool.io();
        for (const auto& x : io) {
            const uint64_t dead = faults.dead.load();
            CHECK(!(dead >> x.tier & 1) || faults.round.load() >> x.tier & 1);   // agreed dead: the pool copies nothing there
            if (x.kind == 0) {
                Tier& t = *tiers.at(x.tier);
                std::memcpy(t.buf.data(), dev[x.page].data(), sizeof(Page));
                call(x.tier, "write", [&] { return g6kv_tier_write(t.t, x.slot, t.buf.data()); });
            } else if (x.kind == 1) {
                Tier &a = *tiers.at(x.tier), &b = *tiers.at(x.to);
                CHECK(!(dead >> x.to & 1) || faults.round.load() >> x.to & 1);
                if (call(x.tier, "read", [&] { return g6kv_tier_read(a.t, x.slot, a.buf.data()); })) {
                    std::memcpy(b.buf.data(), a.buf.data(), sizeof(Page));
                    call(x.to, "write", [&] { return g6kv_tier_write(b.t, x.to_slot, b.buf.data()); });
                }
            } else {
                Tier& t = *tiers.at(x.tier);
                if (call(x.tier, "read", [&] { return g6kv_tier_read(t.t, x.slot, t.buf.data()); }))
                    std::memcpy(dev[x.page].data(), t.buf.data(), sizeof(Page));
            }
        }
        require(pool.io_done() == nullptr, "io_done failed");
        return io;
    }
    void write(const std::vector<int>& list, size_t from, const std::vector<int>& tokens) {
        for (size_t r = from; r < tokens.size(); ++r) dev[list[r / 64]][r % 64] = tokens[r];
    }
};
struct Live { Run run; std::vector<int> tokens; };

// the barrier after a drain: the ranks' failed tiers ORed, dropped alike on both. The agreed mask
static uint64_t settle(Rank (&ranks)[2], long& dropped) {
    const uint64_t mask = ranks[0].faults.take() | ranks[1].faults.take();
    uint64_t n[2];
    for (int k = 0; k < 2; ++k) { ranks[k].faults.agreed(mask); n[k] = ranks[k].pool.tier_dead(mask); }
    CHECK(n[0] == n[1]);
    dropped += long(n[0]);
    return mask;
}

int main(int argc, char** argv)
{
    std::vector<const char*> specs;
    int kill_pid = 0, kill_sig = 0;
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "--kill") || !std::strcmp(argv[i], "--stop")) && i + 1 < argc) {
            kill_sig = argv[i][2] == 'k' ? SIGKILL : SIGSTOP; kill_pid = std::atoi(argv[++i]);
        } else specs.push_back(argv[i]);
    }
    if (specs.empty()) specs.push_back("mem:");
    if (kill_sig == SIGSTOP) g6kv_tier_timeout(1);   // a frozen node costs one 1 s call, then the tier is dead
    for (const char* sp : specs) {
        spec = sp;
        long grants = 0, redone = 0, restored = 0, failures = 0, timeouts = 0, dropped = 0, dead_seeds = 0;
        const uint64_t seeds = kill_pid ? 1 : getenv("POOL_FAULT_SEEDS") ? std::strtoull(getenv("POOL_FAULT_SEEDS"), nullptr, 10) : 40;
        for (seed = 1; seed <= seeds; ++seed) {
            Rng rng{seed * 7919};
            const uint32_t pages = rng.range(6, 30), place = rng.one_in(2) ? pages : rng.range(4, pages);
            const uint32_t below = rng.range(3, place) * 64;
            const uint64_t slots = rng.range(8, 40);
            const std::string name = "pool-fault-test-" + std::to_string(seed) + "-r";
            const std::string own = std::string(sp).rfind("ssd:", 0) == 0 ? ".r" : "";
            // with --kill/--stop only rank 1 is on the node: rank 0's tier 1 is memory
            const std::string s0 = kill_pid ? "mem:" : sp + (own.empty() ? "" : own + "0"), s1 = sp + (own.empty() ? "" : own + "1");
            Rank ranks[2] = {Rank(0, pages, place, below, slots, s0.c_str(), name + "0"), Rank(1, pages, place, below, slots, s1.c_str(), name + "1")};
            // the faults: none (1 in 5 seeds), one rank's call on one tier, or one on each rank
            const int plan = kill_pid ? -1 : int(seed % 5);
            if (plan >= 1) ranks[plan & 1].fail_at[rng.range(0, 1)] = rng.range(1, 300);
            if (plan >= 3) ranks[(plan & 1) ^ 1].fail_at[rng.range(0, 1)] = rng.range(1, 300);
            const int kill_op = 120;
            std::vector<Live> live;
            std::vector<std::vector<int>> hist;
            for (int op = 0; op < 250; ++op) {
                if (kill_pid && op == kill_op) { CHECK(::kill(kill_pid, kill_sig) == 0); std::printf("pool_fault_test: signal %d to the node at op %d\n", kill_sig, op); }
                if (!live.empty() && (rng.one_in(2) || live.size() > 3)) {
                    const size_t i = rng.range(0, int(live.size()) - 1);
                    Live s = live[i]; live.erase(live.begin() + long(i));
                    g6rt::PrefixReuse st;
                    st.history = s.tokens; st.valid = int(s.tokens.size()) - 1; st.written_end = st.valid; st.pf_end = st.valid;
                    std::vector<g6kv_io> io[2];
                    for (int k = 0; k < 2; ++k) { ranks[k].pool.retain(s.run, st); io[k] = ranks[k].run_io(); }
                    CHECK(io[0].size() == io[1].size());
                    // odd seeds: the retire's copies and the next grant's meet at one barrier, as a retire and a bind at
                    // one event do (the pool keeps the moves of both for tier_dead)
                    if (!(seed & 1)) settle(ranks, dropped);
                    hist.push_back(s.tokens);
                    continue;
                }
                std::vector<int> prompt;
                int64_t span = -1;
                if (rng.one_in(3)) {
                    for (int n = 0; n < 300; ++n) prompt.push_back(7000 + n);
                    if (rng.one_in(2)) span = 300;
                }
                else if (!hist.empty() && !rng.one_in(4)) {
                    const auto& h = hist[rng.range(0, int(hist.size()) - 1)];
                    prompt.assign(h.begin(), h.begin() + rng.range(1, int(h.size())));
                }
                for (int n = rng.range(1, 192); n > 0; --n) prompt.push_back(rng.range(0, 1 << 20));
                const int gen = rng.range(1, 64);
                const uint64_t worst = prompt.size() + gen + 1;
                const uint64_t now = ranks[0].pool.exclusive(worst) ? worst : prompt.size() + 1;
                if (worst > uint64_t(pages) * 64 || !ranks[0].pool.fits(now, worst)) continue;
                const int limit = int(prompt.size()) - 1;
                CHECK(ranks[0].pool.count(prompt, span) == ranks[1].pool.count(prompt, span));
                const PoolHit h = ranks[0].pool.match(prompt, now, worst, limit), h1 = ranks[1].pool.match(prompt, now, worst, limit);
                CHECK(h.skip == h1.skip && h.stored == h1.stored && h.tier == h1.tier);
                std::vector<int> tokens = prompt;
                for (int n = 0; n < gen; ++n) tokens.push_back(rng.range(0, 1 << 20));
                Run got[2]; std::vector<g6kv_io> ios[2];
                for (int k = 0; k < 2; ++k) {
                    Rank& r = ranks[k];
                    got[k] = h.skip > 0 ? r.pool.take(h, now, worst) : r.pool.reserve(now, worst);
                    ios[k] = r.run_io();
                }
                CHECK(got[0] == got[1] && ios[0].size() == ios[1].size());
                bool restores = false;
                for (const auto& x : ios[0]) restores |= x.kind == 2;
                // the step's barrier, then the prefill: from row 0 again when a failed store was under its restores
                const uint64_t mask = settle(ranks, dropped);
                const int skip = mask && restores ? 0 : h.skip;
                redone += skip != h.skip; restored += restores; ++grants;
                size_t end = prompt.size();
                for (int k = 0; k < 2; ++k) {
                    Rank& r = ranks[k];
                    const std::vector<int> list = r.pool.list(got[k]);
                    for (int row = 0; row < skip; ++row) CHECK(r.dev[list[size_t(row) / 64]][size_t(row) % 64] == prompt[size_t(row)]);
                    r.write(list, size_t(skip), prompt);
                }
                // the decode rows, in a later step (its own barrier)
                std::optional<Run> g[2];
                for (int k = 0; k < 2; ++k) if ((g[k] = ranks[k].pool.grow(got[k], worst))) { got[k] = *g[k]; ranks[k].run_io(); }
                CHECK(bool(g[0]) == bool(g[1]));
                settle(ranks, dropped);
                if (g[0]) {
                    for (int k = 0; k < 2; ++k) ranks[k].write(ranks[k].pool.list(got[k]), prompt.size(), tokens);
                    end = tokens.size();
                }
                CHECK(got[0] == got[1] && ranks[0].pool.digest() == ranks[1].pool.digest());
                tokens.resize(end);
                live.push_back(Live{got[0], tokens});
            }
            settle(ranks, dropped);   // the last retire's barrier
            const auto t0 = ranks[0].pool.tiers(), t1 = ranks[1].pool.tiers();
            const uint64_t dead = ranks[0].faults.dead.load();
            CHECK(dead == ranks[1].faults.dead.load());
            for (size_t t = 0; t < t0.size() && t < t1.size(); ++t) {
                CHECK(t0[t].dead == t1[t].dead && t0[t].dead == (dead >> t & 1));
                if (t0[t].dead) CHECK(t0[t].entries == 0 && t1[t].entries == 0);
            }
            CHECK(dead == 0 || plan >= 1 || kill_pid);   // (a fault drawn past the seed's last call fails nothing)
            dead_seeds += dead != 0;
            failures += long(ranks[0].faults.failed + ranks[1].faults.failed);
            timeouts += long(ranks[0].faults.timeouts + ranks[1].faults.timeouts);
        }
        if (kill_pid) CHECK(failures == 1 && dead_seeds == 1 && (kill_sig != SIGSTOP || timeouts == 1));
        else CHECK(dead_seeds >= 16 && redone > 0 && dropped > 0);
        std::printf("pool_fault_test %s: %ld grants (%ld restored, %ld prefilled again), %ld failed store calls (%ld timed out), "
                    "%ld seeds with a dead store, %ld entries dropped\n", sp, grants, restored, redone, failures, timeouts, dead_seeds, dropped);
    }
    std::printf("pool_fault_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
