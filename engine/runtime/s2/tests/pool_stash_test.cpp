// The pool's block stores (s2/pool.h add_tier, kvman-rs src/stash.rs) through the C interface, in lock-step on two ranks:
// a small memory tier for evicted runs, then a tier of the store under test (g6kv_tier: mem:, ssd:PATH, or
// remote:HOST:PORT on a KV node) that keeps both evicted runs and shared prefixes. Each rank models its device pages as
// rows of tokens. After a grant and its copies (spills, moves between the tiers, restores), every row below the
// grant's skip must hold the prompt's token. The ranks must count, grant and copy alike and agree on the digest.
// Host only.
//   pool_stash_test SPEC [SPEC ..]   (default mem:)
#include "s2/pool.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
using namespace g6s2;

static long fails = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 20) std::printf("FAIL %s seed %llu: %s\n", spec, (unsigned long long)seed, #c); ++fails; } } while (0)

struct Rng {
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    int range(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
    bool one_in(int n) { return next() % uint64_t(n) == 0; }
};

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
    Rank(uint32_t pages, uint32_t place, uint32_t below, uint64_t slots, const char* spec, const std::string& name)
        : pool(pages, place, below), dev(pages) {
        g6rt::PrefixReuse rules; rules.strict = false; rules.noend = true; rules.grid = 128;
        pool.keep(rules); pool.stores(2);
        tiers.push_back(std::make_unique<Tier>("mem:", name + "-t0", 4));
        tiers.push_back(std::make_unique<Tier>(spec, name + "-t1", slots));
        pool.add_tier(4, 0, "", "");
        pool.add_tier(slots, 2, "", "");
        // the policies --kv-keep sessions and --kv-write-limit set, through the C interface
        pool.policy(1, 0, 1, sizeof(Page));
        pool.write_limit(500);
        // runs past R keep their tails. Two ranks share the tail: a run past R takes its own pages (its prefix and
        // shard below place, the drafter's copy above it, pool.rs page classes), and each rank spills its shard (the
        // (below + j)-th page of its list below place) and every tail page's replicated parts — with per_slot 1, a
        // block holds one page's parts, the whole page again
        pool.tail(2, 1);
        for (auto& p : dev) p.fill(-1);
    }
    // the engine's part: every copy, in order, then io_done
    long tail_spills = 0, tail_restores = 0;
    std::vector<g6kv_io> run_io() {
        const auto io = pool.io();
        bool restoring = false;
        for (const auto& x : io) {
            require(x.part <= 2 && (x.part == 2 || x.sub == 0) && (x.part != 2 || x.sub < 1), "a copy's part");
            if (x.part) ++(x.kind == 0 ? tail_spills : tail_restores);
            if (x.kind == 0) {
                require(!restoring, "a spill after a restore");
                Tier& t = *tiers.at(x.tier);
                std::memcpy(t.buf.data(), dev[x.page].data(), sizeof(Page));
                require(g6kv_tier_write(t.t, x.slot, t.buf.data()) == 0, g6kv_tier_error());
            } else if (x.kind == 1) {
                require(!restoring && x.to > x.tier, "a move after a restore, or backwards");
                Tier &a = *tiers.at(x.tier), &b = *tiers.at(x.to);
                require(g6kv_tier_read(a.t, x.slot, a.buf.data()) == 0, g6kv_tier_error());
                std::memcpy(b.buf.data(), a.buf.data(), sizeof(Page));
                require(g6kv_tier_write(b.t, x.to_slot, b.buf.data()) == 0, g6kv_tier_error());
            } else {
                restoring = true;
                Tier& t = *tiers.at(x.tier);
                require(g6kv_tier_read(t.t, x.slot, t.buf.data()) == 0, g6kv_tier_error());
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

int main(int argc, char** argv)
{
    std::vector<const char*> specs(argv + 1, argv + argc);
    if (specs.empty()) specs.push_back("mem:");
    for (const char* spec : specs) {
        long hits = 0, stored = 0, spills = 0, moves = 0, pending = 0, counted = 0, tail_out = 0, tail_back = 0;
        for (uint64_t seed = 1; seed <= 25; ++seed) {
            Rng rng{seed * 7907};
            const uint32_t pages = rng.range(6, 30), place = rng.one_in(2) ? pages : rng.range(4, pages);
            const uint32_t below = rng.range(3, place) * 64;
            const uint64_t slots = rng.range(1, 40);
            const std::string name = "pool-stash-test-" + std::to_string(seed) + "-r";
            // each rank has its own store, as on separate boxes: a store shared by both would see one rank's later spills
            // land in slots the other has yet to restore from
            const std::string own = std::string(spec).rfind("ssd:", 0) == 0 ? ".r" : "";
            const std::string s0 = spec + (own.empty() ? "" : own + "0"), s1 = spec + (own.empty() ? "" : own + "1");
            Rank ranks[2] = {Rank(pages, place, below, slots, s0.c_str(), name + "0"), Rank(pages, place, below, slots, s1.c_str(), name + "1")};
            std::vector<Live> live;
            std::vector<std::vector<int>> hist;
            for (int op = 0; op < 250; ++op) {
                if (!live.empty() && (rng.one_in(2) || live.size() > 3)) {
                    const size_t i = rng.range(0, int(live.size()) - 1);
                    Live s = live[i]; live.erase(live.begin() + long(i));
                    g6rt::PrefixReuse st;
                    st.history = s.tokens; st.valid = int(s.tokens.size()) - 1; st.written_end = st.valid; st.pf_end = st.valid;
                    std::vector<g6kv_io> io[2];
                    for (int k = 0; k < 2; ++k) { ranks[k].pool.retain(s.run, st); io[k] = ranks[k].run_io(); }
                    CHECK(io[0].size() == io[1].size());
                    hist.push_back(s.tokens);
                    continue;
                }
                std::vector<int> prompt;
                int64_t span = -1;   // the front end's system span, sent for half the prompts on the system prompt
                if (rng.one_in(3)) {
                    for (int n = 0; n < 300; ++n) prompt.push_back(7000 + n);   // a shared system prompt
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
                const uint32_t c0 = ranks[0].pool.count(prompt, span), c1 = ranks[1].pool.count(prompt, span);
                CHECK(c0 == c1);
                counted += c0;
                const PoolHit h = ranks[0].pool.match(prompt, now, worst, limit), h1 = ranks[1].pool.match(prompt, now, worst, limit);
                CHECK(h.skip == h1.skip && h.age == h1.age && h.stored == h1.stored && h.tier == h1.tier && std::string(h.why) == h1.why);
                std::vector<int> tokens = prompt;
                for (int n = 0; n < gen; ++n) tokens.push_back(rng.range(0, 1 << 20));
                Run got[2]; std::vector<g6kv_io> ios[2]; uint64_t digests[2] = {}; size_t ends[2] = {};
                for (int k = 0; k < 2; ++k) {
                    Rank& r = ranks[k];
                    Run run = h.skip > 0 ? r.pool.take(h, now, worst) : r.pool.reserve(now, worst);
                    if (r.pool.io_pending()) {   // the next claim waits for these copies
                        bool refused = false;
                        try { r.pool.reserve(1, 1); } catch (const std::exception&) { refused = true; }
                        CHECK(refused);
                        pending += refused;
                    }
                    ios[k] = r.run_io();
                    std::vector<int> list = r.pool.list(run);
                    for (int row = 0; row < h.skip; ++row) CHECK(r.dev[list[size_t(row) / 64]][size_t(row) % 64] == prompt[size_t(row)]);
                    r.write(list, size_t(h.skip), prompt);
                    ends[k] = prompt.size();
                    if (const auto g = r.pool.grow(run, worst)) {   // the decode rows, when the pool lets the run grow
                        run = *g;
                        const auto more = r.run_io();
                        ios[k].insert(ios[k].end(), more.begin(), more.end());
                        r.write(r.pool.list(run), prompt.size(), tokens);
                        ends[k] = tokens.size();
                    }
                    got[k] = run; digests[k] = r.pool.digest();
                }
                CHECK(got[0] == got[1] && digests[0] == digests[1] && ends[0] == ends[1] && ios[0].size() == ios[1].size());
                for (size_t i = 0; i < ios[0].size() && i < ios[1].size(); ++i)
                    CHECK(ios[0][i].kind == ios[1][i].kind && ios[0][i].page == ios[1][i].page && ios[0][i].slot == ios[1][i].slot &&
                          ios[0][i].part == ios[1][i].part && ios[0][i].sub == ios[1][i].sub);
                for (const auto& x : ios[0]) { spills += x.kind == 0; moves += x.kind == 1; }
                hits += h.skip > 0; stored += h.stored;
                tokens.resize(ends[0]);
                live.push_back(Live{got[0], tokens});
            }
            CHECK(ranks[0].pool.tiers().size() == 2);
            CHECK(ranks[0].tail_spills == ranks[1].tail_spills && ranks[0].tail_restores == ranks[1].tail_restores);
            tail_out += ranks[0].tail_spills; tail_back += ranks[0].tail_restores;
        }
        const uint64_t seed = 0;
        CHECK(stored > 10 && spills > 50 && moves > 10 && pending > 0 && counted > 10 && tail_out > 0);
        std::printf("pool_stash_test %s: %ld hits (%ld stored), %ld spills, %ld moves, %ld counted, %ld refused claims, %ld tail "
                    "copies out and %ld back\n", spec, hits, stored, spills, moves, counted, pending, tail_out, tail_back);
    }
    std::printf("pool_stash_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
