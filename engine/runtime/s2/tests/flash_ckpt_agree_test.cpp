// persist=on for Flash checkpoints: ranks that took back different persisted entries agree at the bootstrap
// (s2/flash_ckpt_index.h flash_agree_summary / flash_keep_agreed over bootstrap.h agree_lines), and the boot goes on:
// every rank keeps the entries every rank has, with the same clocks, so the index digests match. TP=1 agrees with
// itself; TP=2 and TP=4 over loopback. The index rewritten after the agreement loads back to the same digest.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/s2/tests/flash_ckpt_agree_test.cpp <knobs libs> -o /tmp/flash_ckpt_agree_test
#include "s2/bootstrap.h"
#include "s2/flash_ckpt_index.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static const uint64_t COMPAT = 0xA11CEull;

// rank r's index after the same captures: entries of 1 + 10 blocks at 64 x (gen) rows
static FlashIndex captures(int n) {
    FlashIndex ix(1000);
    for (int g = 1; g <= n; ++g) {
        FlashEntry x;
        x.blocks = 10;
        if (!ix.reserve(x.blocks, x.base)) throw std::runtime_error("no room");
        x.generation = ix.next_generation(); x.position = uint32_t(64 * g);
        for (uint32_t i = 0; i <= x.position; ++i) x.tokens.push_back(int(i + 100 * g));
        x.manifest = flash_manifest(x.generation, COMPAT, x.position, x.tokens, x.ns, uint64_t(g), 4096, 8192);
        x.manifest.seal();
        ix.add(x);
    }
    return ix;
}
static std::vector<uint64_t> gens(const FlashIndex& ix) {
    std::vector<uint64_t> v;
    for (const auto& x : ix.entries()) v.push_back(x.generation);
    return v;
}

static BootstrapConfig config(unsigned rank, unsigned nr, uint16_t port, const std::string& dir) {
    BootstrapConfig c; c.rank = rank; c.ranks = nr; c.timeout = 1; c.head_ip = "127.0.0.1"; c.port = port;
    c.directory = dir + "/r" + std::to_string(rank); c.runtime_signature = "flash test ckpt aligned blocks 1000 persist";
    ::mkdir(c.directory.c_str(), 0700);
    return c;
}
// every rank agrees over loopback (nr > 1) or alone (nr = 1: agree_lines on the head with no links, as at TP=1);
// returns each rank's dropped generations
static std::vector<std::vector<uint64_t>> agree(std::vector<FlashIndex>& ix, uint16_t port, const std::string& dir) {
    const unsigned nr = unsigned(ix.size());
    std::vector<std::vector<uint64_t>> gone(nr);
    std::atomic<bool> bad{false};
    if (nr == 1) {
        BootstrapConfig c = config(0, 1, port, dir);
        ControlLinks none;
        gone[0] = flash_keep_agreed(ix[0], agree_lines(c, none, flash_agree_summary(ix[0]), true));
        return gone;
    }
    std::vector<std::thread> w;
    for (unsigned r = 0; r < nr; ++r) w.emplace_back([&, r] {
        try {
            auto c = config(r, nr, port, dir);
            auto links = bootstrap(c);
            // the last rank checks its entries for longer than the 1 s socket timeout, and the other ranks wait
            if (r == nr - 1) std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            gone[r] = flash_keep_agreed(ix[r], agree_lines(c, links, flash_agree_summary(ix[r]), true));
            links.shutdown();
        } catch (const std::exception& e) { std::fprintf(stderr, "rank %u: %s\n", r, e.what()); bad = true; }
    });
    for (auto& t : w) t.join();
    CHECK(!bad);
    return gone;
}
static bool same_digest(const std::vector<FlashIndex>& ix) {
    for (const auto& x : ix) if (x.digest() != ix[0].digest()) return false;
    return true;
}

int main() {
    const char* base = getenv("TMPDIR");
    std::string tmpl = std::string(base && *base ? base : "/tmp") + "/fckagrXXXXXX";
    if (!mkdtemp(&tmpl[0])) { std::perror("mkdtemp"); return 1; }
    const std::string dir = tmpl;
    uint16_t port = uint16_t(20000 + getpid() % 20000);
    auto sub = [&](const std::string& n) { const std::string d = dir + "/" + n; std::filesystem::create_directories(d); return d; };

    // TP=1: the agreement is with this rank alone, nothing dropped, the clocks kept
    {
        std::vector<FlashIndex> ix{captures(4)};
        const uint64_t d = ix[0].digest();
        auto gone = agree(ix, port++, sub("tp1"));
        CHECK(gone[0].empty() && ix[0].size() == 4 && ix[0].digest() == d);
    }
    // TP=2: rank 1's check dropped entry 2 (a page checksum); both keep 1, 3, 4 and the clocks (they agree)
    {
        std::vector<FlashIndex> ix{captures(4), captures(4)};
        ix[1].drop(2);
        auto gone = agree(ix, port++, sub("tp2"));
        CHECK(gone[0] == std::vector<uint64_t>{2} && gone[1].empty());
        CHECK(gens(ix[0]) == (std::vector<uint64_t>{1, 3, 4}) && same_digest(ix));
        CHECK(ix[0].next_generation() == 5 && ix[1].next_generation() == 5);
    }
    // TP=4: rank 2 lost its index (a power loss, or a failed write or restore removed it): every rank starts empty,
    // from generation 1
    {
        std::vector<FlashIndex> ix{captures(4), captures(4), FlashIndex(1000), captures(4)};
        auto gone = agree(ix, port++, sub("tp4-lost"));
        CHECK(gone[0].size() == 4 && gone[2].empty() && gone[3].size() == 4);
        for (const auto& x : ix) CHECK(x.size() == 0 && x.next_generation() == 1);
        CHECK(same_digest(ix));
    }
    // TP=4: rank 3 kept one more capture (the others' index writes failed after it) and touched entry 1 since. The
    // clocks differ, so every rank continues one past the newest entry kept, and entry 1 (another use) goes
    {
        std::vector<FlashIndex> ix{captures(4), captures(4), captures(4), captures(5)};
        ix[3].touch(1);
        auto gone = agree(ix, port++, sub("tp4-clock"));
        CHECK(gens(ix[0]) == (std::vector<uint64_t>{2, 3, 4}) && same_digest(ix));
        CHECK(gone[3] == (std::vector<uint64_t>{1, 5}));
        for (const auto& x : ix) CHECK(x.next_generation() == 5 && x.clock() == 4);
    }
    // TP=2: the same generation at another base on one rank (its own capture order went another way) is dropped
    {
        std::vector<FlashIndex> ix{captures(3), captures(3)};
        std::vector<FlashEntry> moved = ix[1].entries();
        moved[1].base = 500;
        ix[1].adopt(ix[1].next_generation(), ix[1].clock(), moved);
        agree(ix, port++, sub("tp2-base"));
        CHECK(gens(ix[0]) == (std::vector<uint64_t>{1, 3}) && same_digest(ix));
    }
    // the index rewritten after the agreement loads back to the agreed digest, and the dropped entry's tokens go
    {
        const std::string d = sub("files");
        FlashIndex ix = captures(4);
        FlashIndexFiles f(d + "/flash-ckpt.index");
        f.save(COMPAT, ix);
        FlashIndex other = captures(4);
        other.drop(3);
        flash_keep_agreed(ix, flash_agree_summary(other));
        f.save(COMPAT, ix);
        FlashIndexImage img;
        uint64_t stale = 0;
        FlashIndexFiles g(d + "/flash-ckpt.index");
        CHECK(!g.load(COMPAT, 1000, img, stale) && img.entries.size() == 3);
        FlashIndex back(1000);
        back.adopt(img.next, img.clock, img.entries);
        CHECK(back.digest() == ix.digest() && back.digest() == other.digest());
        CHECK(access((d + "/flash-ckpt.tokens/3").c_str(), F_OK) != 0);
    }
    std::filesystem::remove_all(dir);
    if (!fails) std::printf("flash_ckpt_agree_test: PASS\n");
    return fails ? 1 : 0;
}
