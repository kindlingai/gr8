// A pool tier on a group lease of a KV node, resized through the C interface (s2/kv_share.h over kvman-rs blocks.rs
// g6kv_tier_open_group / g6kv_store_*, ffi.rs g6kv_tier_resize / g6kv_tier_coldest). Cluster A (two ranks in lock-step,
// as pool_stash_test) fills its tier, its head reports and grows into the node's grant, cluster B (one rank) wants
// room so the node asks A to give a segment back, A shrinks, then shrinks again to its floor. Every rank applies each
// resize at the same point. After every grant and its copies the rows below the grant's skip must hold the prompt's
// tokens, and the ranks must agree on the digest. Host only; needs a node with small segments:
//   kvman-remote --gb 0.05 --seg-mb 2 --margin-s 0 --no-lock --no-rdma ...
//   kv_share_test remote:HOST:PORT
#include "s2/kv_share.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
using namespace g6s2;

static long fails = 0;
#define CHECK(c) do { if (!(c)) { if (fails < 20) std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

struct Rng {
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    int range(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); }
    bool one_in(int n) { return next() % uint64_t(n) == 0; }
};

using Page = std::array<int32_t, 64>;
constexpr uint64_t BLOCK = 256 << 10;   // 8 blocks a 2 MiB segment
constexpr uint64_t SEG = 2 << 20;

struct Rank {
    RunPool pool;
    std::vector<Page> dev;
    g6kv_tier* store = nullptr;
    std::vector<uint8_t> buf;
    long moves = 0;
    Rank(const char* spec, uint32_t nr, uint32_t rank, uint64_t cap_segs) : pool(24, 24, 12 * 64), dev(24) {
        require(g6kv_tier_open_group(spec, ("kv-share-test-a" + std::to_string(rank)).c_str(), "kv-share-test-a", nr, rank, SEG,
                                     cap_segs * SEG, 1000, BLOCK, &store) == 0, g6kv_tier_error());
        buf.resize(g6kv_tier_block_bytes(store));
        g6rt::PrefixReuse rules; rules.strict = false; rules.noend = true; rules.grid = 128;
        pool.keep(rules); pool.stores(2);
        pool.add_tier(g6kv_store_blocks(store), 2, "", "");
        pool.policy(0, 0, 0, BLOCK);
        for (auto& p : dev) p.fill(-1);
    }
    ~Rank() { g6kv_tier_free(store); }
    // the engine's part: every copy, in order, then io_done
    std::vector<g6kv_io> run_io() {
        const auto io = pool.io();
        for (const auto& x : io) {
            require(x.tier == 0, "one tier");
            if (x.kind == 0) {
                std::memcpy(buf.data(), dev[x.page].data(), sizeof(Page));
                require(g6kv_tier_write(store, x.slot, buf.data()) == 0, g6kv_tier_error());
            } else if (x.kind == 1) {   // within the tier: a shrink's moves into the slots that stay
                require(x.to == 0, "a move within the tier");
                require(g6kv_tier_read(store, x.slot, buf.data()) == 0, g6kv_tier_error());
                require(g6kv_tier_write(store, x.to_slot, buf.data()) == 0, g6kv_tier_error());
                ++moves;
            } else {
                require(g6kv_tier_read(store, x.slot, buf.data()) == 0, g6kv_tier_error());
                std::memcpy(dev[x.page].data(), buf.data(), sizeof(Page));
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

struct Cluster {
    std::unique_ptr<Rank> r[2];
    Rng rng{4242};
    std::vector<Live> live;
    std::vector<std::vector<int>> hist;
    long hits = 0, stored = 0, spills = 0;
    // one conversation step on both ranks, as pool_stash_test: retain a live run, or admit a prompt and check its rows
    void step() {
        if (!live.empty() && (rng.one_in(2) || live.size() > 3)) {
            const size_t i = rng.range(0, int(live.size()) - 1);
            Live s = live[i]; live.erase(live.begin() + long(i));
            g6rt::PrefixReuse st;
            st.history = s.tokens; st.valid = int(s.tokens.size()) - 1; st.written_end = st.valid; st.pf_end = st.valid;
            size_t n[2];
            for (int k = 0; k < 2; ++k) { r[k]->pool.retain(s.run, st); n[k] = r[k]->run_io().size(); }
            CHECK(n[0] == n[1]);
            hist.push_back(s.tokens);
            return;
        }
        std::vector<int> prompt;
        if (rng.one_in(3)) for (int n = 0; n < 300; ++n) prompt.push_back(7000 + n);
        else if (!hist.empty() && !rng.one_in(4)) {
            const auto& h = hist[rng.range(0, int(hist.size()) - 1)];
            prompt.assign(h.begin(), h.begin() + rng.range(1, int(h.size())));
        }
        for (int n = rng.range(1, 192); n > 0; --n) prompt.push_back(rng.range(0, 1 << 20));
        const int gen = rng.range(1, 64);
        const uint64_t worst = prompt.size() + gen + 1;
        const uint64_t now = r[0]->pool.exclusive(worst) ? worst : prompt.size() + 1;
        if (worst > 24 * 64 || !r[0]->pool.fits(now, worst)) return;
        const int limit = int(prompt.size()) - 1;
        CHECK(r[0]->pool.count(prompt, -1) == r[1]->pool.count(prompt, -1));
        const PoolHit h = r[0]->pool.match(prompt, now, worst, limit), h1 = r[1]->pool.match(prompt, now, worst, limit);
        CHECK(h.skip == h1.skip && h.stored == h1.stored);
        std::vector<int> tokens = prompt;
        for (int n = 0; n < gen; ++n) tokens.push_back(rng.range(0, 1 << 20));
        Run got[2]; uint64_t digests[2] = {}; size_t ends[2] = {}, ios[2] = {};
        for (int k = 0; k < 2; ++k) {
            Rank& x = *r[k];
            Run run = h.skip > 0 ? x.pool.take(h, now, worst) : x.pool.reserve(now, worst);
            const auto io = x.run_io();
            for (const auto& c : io) spills += k == 0 && c.kind == 0;
            ios[k] = io.size();
            std::vector<int> list = x.pool.list(run);
            for (int row = 0; row < h.skip; ++row) CHECK(x.dev[list[size_t(row) / 64]][size_t(row) % 64] == prompt[size_t(row)]);
            x.write(list, size_t(h.skip), prompt);
            ends[k] = prompt.size();
            if (const auto g = x.pool.grow(run, worst)) {
                run = *g;
                ios[k] += x.run_io().size();
                x.write(x.pool.list(run), prompt.size(), tokens);
                ends[k] = tokens.size();
            }
            got[k] = run; digests[k] = x.pool.digest();
        }
        CHECK(got[0] == got[1] && digests[0] == digests[1] && ends[0] == ends[1] && ios[0] == ios[1]);
        hits += h.skip > 0; stored += h.stored;
        tokens.resize(ends[0]);
        live.push_back(Live{got[0], tokens});
    }
    void steps(int n) { for (int i = 0; i < n; ++i) step(); }
    // every rank at the same point
    bool apply(const Resize& e) {
        bool ok[2];
        for (int k = 0; k < 2; ++k) { Rank* x = r[k].get(); ok[k] = share_apply(x->pool, x->store, e, [x] { x->run_io(); }); }
        CHECK(ok[0] == ok[1] && r[0]->pool.digest() == r[1]->pool.digest());
        return ok[0];
    }
    uint64_t slots(int k) const { return r[k]->pool.tiers().at(0).slots; }
};

int main(int argc, char** argv)
{
    if (argc != 2) { std::printf("usage: kv_share_test remote:HOST:PORT\n"); return 2; }
    const char* spec = argv[1];
    // share_want: what the head asks for from the counters at the last report and now
    {
        g6kv_tier_totals l{}, n{};
        const uint64_t p = 306;                             // blocks a segment
        n.slots = 612; n.free_slots = 612;
        CHECK(share_want(l, n, p) == 0);                    // empty, no spill
        n.free_slots = 233; n.entries = 1; n.spilled_pages = 379;
        CHECK(share_want(l, n, p) == 1);                    // one 379-block entry, 233 free: the next would evict
        n.free_slots = 400;
        CHECK(share_want(l, n, p) == 0);                    // room for another of that size
        l = n; n.free_slots = 100;
        CHECK(share_want(l, n, p) == 0);                    // low but no spill since the last report: an idle tier asks nothing
        n.full = 1; n.dropped_pages = 400;
        CHECK(share_want(l, n, p) == 2);                    // it filled since the last report: room for the 400 pages it dropped
        l = n; n.oversize = 1; n.oversize_last = 1500;
        CHECK(share_want(l, n, p) == 5);                    // a 1500-block entry did not fit 612 slots: 1400 more than free
        CHECK(share_want(l, n, 0) == 0);                    // not a group lease
    }
    // B first: one rank at its floor of one segment
    g6kv_tier* b = nullptr;
    require(g6kv_tier_open_group(spec, "kv-share-test-b0", "kv-share-test-b", 1, 0, SEG, 4 * SEG, 1000, BLOCK, &b) == 0, g6kv_tier_error());
    Cluster a;
    for (uint32_t k = 0; k < 2; ++k) a.r[k] = std::make_unique<Rank>(spec, 2, k, 12);
    const uint64_t per = g6kv_store_seg_blocks(a.r[0]->store);
    CHECK(per == SEG / BLOCK && g6kv_store_segs(a.r[0]->store) == 1 && a.slots(0) == per);

    // the head reports after every step as the engine does between rounds: it asks (share_want) before the tier
    // first fills. The node is not asked here (A stays at its floor for the checks below), only the want is read.
    long asked_at = -1, full_at = -1;
    {
        g6kv_tier_totals seen{};
        for (long i = 0; i < 150; ++i) {
            a.step();
            const g6kv_tier_totals now = a.r[0]->pool.tiers().at(0);
            if (asked_at < 0 && share_want(seen, now, per) && now.full == 0) asked_at = i;
            if (full_at < 0 && now.full > 0) full_at = i;
            seen = now;
        }
    }
    std::printf("kv_share_test: first ask ahead of a drop at step %ld, tier first full at step %ld\n", asked_at, full_at);
    CHECK(asked_at >= 0 && (full_at < 0 || asked_at <= full_at));
    const auto full = a.r[0]->pool.tiers().at(0);
    CHECK(full.full > 0);   // the tier had to make room
    const long hits0 = a.hits, stored0 = a.stored;

    // the node holds 25 segments: B 1, A 1 a rank. A's head wants 11 a rank, up to its cap: the node grants all 22
    // that nobody holds
    const auto g = share_poll(a.r[0]->pool, 0, a.r[0]->store, 11);
    CHECK(g && g->segs == 12 && g->slots == 12 * per);
    if (g) CHECK(a.apply(*g));
    CHECK(a.slots(0) == 12 * per && a.slots(1) == 12 * per && g6kv_store_segs(a.r[1]->store) == 12);
    a.steps(300);
    const long hits1 = a.hits, stored1 = a.stored;

    // A's head reports its ages (every block at least 20 ms old) and wants nothing. Then B wants 2: none is left, so
    // the node asks A (its blocks colder than B's by more than the margin, 0) to give one a rank
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!share_poll(a.r[0]->pool, 0, a.r[0]->store, 0));
    g6kv_report rb{};
    CHECK(g6kv_store_report(b, 2, nullptr, 0, &rb) == 0 && rb.grant == 0 && rb.unclaimed == 0);
    const auto give = share_poll(a.r[0]->pool, 0, a.r[0]->store, 0);
    CHECK(give && give->segs == 11 && give->slots == 11 * per);
    if (give) CHECK(a.apply(*give));
    CHECK(a.slots(0) == 11 * per && g6kv_store_segs(a.r[0]->store) == 11 && g6kv_store_segs(a.r[1]->store) == 11);
    // the two segments A gave are B's to take now
    uint64_t bb = 0;
    CHECK(g6kv_store_report(b, 2, nullptr, 0, &rb) == 0 && rb.grant == 2);
    CHECK(g6kv_store_grow(b, 1 + rb.grant, rb.txn, &bb) == 0 && bb == 3 * per);
    a.steps(300);
    const long hits2 = a.hits, stored2 = a.stored;

    // back to the floor on the group's own (transaction 0): the tier evicts its coldest entries and moves the rest
    // below one segment
    CHECK(a.apply(Resize{0, 1, per, 0}));
    CHECK(a.slots(0) == per && g6kv_store_segs(a.r[0]->store) == 1 && g6kv_store_blocks(a.r[1]->store) == per);
    a.steps(200);
    g6kv_tier_free(b);

    std::printf("kv_share_test: stored hits %ld / %ld / %ld / %ld (before, grown, after the give, at the floor), %ld spills, "
                "%ld moves in shrinks\n", stored0, stored1 - stored0, stored2 - stored1, a.stored - stored2, a.spills, a.r[0]->moves);
    CHECK(stored1 > stored0 && stored2 > stored1 && a.stored > stored2 && a.r[0]->moves > 0 && a.r[0]->moves == a.r[1]->moves);
    (void)hits0; (void)hits1; (void)hits2;
    std::printf("kv_share_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
