// persist=lease through the C interface (D-331 M4): an engine stores a session on a KV-node lease opened with
// "?persist", quits in order (the pool's runs go to the store, as g6m4 does after Service::run), and closes the lease;
// the node parks it (kvman-remote --holdover-s). A new engine with a larger pool (a resize of CAP) opens the same name,
// gets the lease back, loads the whole index (sessions too), and the next turn of the session restores from the store,
// row for row, instead of a prefill. Host only.
//   kv_lease_test remote:HOST:PORT INDEX_DIR   (the node needs --holdover-s > 0)
#include "s2/pool.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
using namespace g6s2;

static long fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s\n", #c); ++fails; } } while (0)

using Page = std::array<int32_t, 64>;
static const uint64_t SLOTS = 64;

struct Engine {
    g6kv_tier* t = nullptr;
    std::vector<uint8_t> buf;
    RunPool pool;
    std::vector<Page> dev;
    bool back = false;
    Engine(const std::string& spec, const std::string& index, uint32_t pages) : pool(pages, pages, pages * 64), dev(pages) {
        require(g6kv_tier_open((spec + "?persist").c_str(), "kv-lease-test-r0-t0", sizeof(Page), SLOTS, &t) == 0, g6kv_tier_error());
        buf.resize(g6kv_tier_block_bytes(t));
        back = g6kv_tier_reattached(t) == 1;
        // as g6m4: a fresh lease starts its index empty
        if (!back) std::filesystem::remove_all(index);
        g6rt::PrefixReuse rules; rules.strict = false; rules.noend = true; rules.grid = 128;
        pool.keep(rules); pool.stores(2);
        pool.add_tier(SLOTS, 2 | 4, index, "kv-lease-test nr 1");   // + 4: the whole index
        for (auto& p : dev) p.fill(-1);
    }
    ~Engine() { g6kv_tier_free(t); }
    void run_io() {
        for (const auto& x : pool.io()) {
            require(x.tier == 0 && x.part == 0 && x.kind != 1, "one tier, no tails, no moves");
            if (x.kind == 0) {
                std::memcpy(buf.data(), dev[x.page].data(), sizeof(Page));
                require(g6kv_tier_write(t, x.slot, buf.data()) == 0, g6kv_tier_error());
            } else {
                require(g6kv_tier_read(t, x.slot, buf.data()) == 0, g6kv_tier_error());
                std::memcpy(dev[x.page].data(), buf.data(), sizeof(Page));
            }
        }
        require(pool.io_done() == nullptr, "io_done failed");
    }
    // a turn: match, take or reserve, the copies, check the restored rows, write the rest, retain. Returns the skip.
    int turn(const std::vector<int>& prompt, std::vector<int>& tokens, int gen) {
        const uint64_t now = prompt.size() + gen + 1;
        pool.count(prompt, -1);
        const PoolHit h = pool.match(prompt, now, now, int(prompt.size()) - 1);
        Run run = h.skip > 0 ? pool.take(h, now, now) : pool.reserve(now, now);
        run_io();
        const std::vector<int> list = pool.list(run);
        for (int row = 0; row < h.skip; ++row) CHECK(dev[list[size_t(row) / 64]][size_t(row) % 64] == prompt[size_t(row)]);
        tokens = prompt;
        for (int n = 0; n < gen; ++n) tokens.push_back(900000 + int(prompt.size()) + n);
        for (size_t r = size_t(h.skip); r < tokens.size(); ++r) dev[list[r / 64]][r % 64] = tokens[r];
        g6rt::PrefixReuse st;
        st.history = tokens; st.valid = int(tokens.size()) - 1; st.written_end = st.valid; st.pf_end = st.valid;
        pool.retain(run, st);
        run_io();
        if (h.stored) ++stored;
        return h.skip;
    }
    // g6m4's orderly quit with a lease store: every retained run goes to the store
    void quit() {
        const auto all = pool.reserve(uint64_t(dev.size()) * 64, uint64_t(dev.size()) * 64);
        run_io();
        pool.release(all);
        run_io();
    }
    long stored = 0;
};

int main(int argc, char** argv)
{
    if (argc != 3) { std::fprintf(stderr, "usage: kv_lease_test remote:HOST:PORT INDEX_DIR\n"); return 2; }
    const std::string spec = argv[1], index = argv[2];
    std::vector<int> system, tokens;
    for (int n = 0; n < 200; ++n) system.push_back(1000 + n);
    size_t entries = 0;
    {
        Engine a(spec, index, 12);
        CHECK(!a.back);
        for (int c = 2; c < 4; ++c) {
            std::vector<int> p = system;
            for (int n = 0; n < 150; ++n) p.push_back(c * 10000 + n);
            a.turn(p, tokens, 20);
        }
        a.quit();
        for (const auto& t : a.pool.tiers()) entries += t.entries;
        CHECK(entries >= 2);   // the system prefix and the last session on it
    }
    // the resize: a larger pool, the same lease name, within the node's holdover
    Engine b(spec, index, 16);
    CHECK(b.back);
    size_t loaded = 0;
    for (const auto& t : b.pool.tiers()) loaded += t.entries;
    CHECK(loaded == entries);
    std::vector<int> next = tokens, out;
    for (int n = 0; n < 40; ++n) next.push_back(777000 + n);
    const int skip = b.turn(next, out, 5);
    CHECK(b.stored == 1 && skip > int(system.size()));
    std::printf("kv_lease_test: %zu entries kept over the restart, the next turn restored %d of %zu prompt rows from the lease\n",
                entries, skip, next.size());
    std::printf("kv_lease_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
