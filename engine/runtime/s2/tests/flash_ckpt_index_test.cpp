// persist=on for Flash checkpoints (s2/flash_ckpt_index.h, s2/flash_checkpoint.h): the index bytes and files round
// trip. A cut, a flipped byte, another compat or store size, or two entries of one generation start empty.
// flash_entry_check refuses stored bytes that are not the entry's. FlashIndex::adopt keeps base order and the LRU order
// and refuses entries that do not fit. After a crash, the index lacks the entry a capture's reserve dropped. `bench`: the cost of one capture's save at 64 entries of 262,145 tokens.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/flash_ckpt_index_test.cpp -o /tmp/flash_ckpt_index_test
#include "s2/flash_ckpt_index.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static const uint64_t COMPAT = 0xC0FFEEull;
static uint64_t blocks_for(uint32_t position) { return 1 + (position + 63) / 64 + 2; }   // manifest, pages, 2 state blocks

// a capture's index steps (FlashCheckpoints::capture): reserve, then add. 0: the store cannot hold it
static uint64_t capture(FlashIndex& ix, uint32_t position, int seed, std::vector<uint64_t>* gone = nullptr) {
    uint64_t base = 0;
    const uint64_t need = blocks_for(position);
    if (!ix.reserve(need, base, gone)) return 0;
    FlashEntry x;
    x.generation = ix.next_generation(); x.base = base; x.blocks = need; x.position = position;
    for (uint32_t i = 0; i <= position; ++i) x.tokens.push_back(int(i * 7 + seed));
    x.ns[0] = uint8_t(seed);
    x.manifest = flash_manifest(x.generation, COMPAT, position, x.tokens, x.ns, (position + 63) / 64, 4096, 8192);
    x.manifest.page_sum = 11 * x.generation; x.manifest.state_sum = 13 * x.generation;
    x.manifest.seal();
    ix.add(x);
    return x.generation;
}
static bool same_entries(const std::vector<FlashEntry>& a, const std::vector<FlashEntry>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].generation != b[i].generation || a[i].used != b[i].used || a[i].base != b[i].base || a[i].blocks != b[i].blocks ||
            a[i].position != b[i].position || a[i].terminal != b[i].terminal || a[i].tokens != b[i].tokens || a[i].ns != b[i].ns ||
            std::memcmp(&a[i].manifest, &b[i].manifest, sizeof(FlashManifest)))
            return false;
    return true;
}
static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void spit(const std::string& p, const std::string& b) { std::ofstream(p, std::ios::binary | std::ios::trunc) << b; }
static void reseal(std::string& b) { const uint64_t s = fnv64(b.data(), b.size() - 8); std::memcpy(&b[b.size() - 8], &s, 8); }

static void bytes_round_trip() {
    FlashIndex ix(100);
    capture(ix, 64, 1); capture(ix, 128, 2); capture(ix, 640, 3);
    ix.touch(1);
    FlashIndexImage img;
    const std::string b = flash_index_encode(COMPAT, ix);
    CHECK(b.size() == fidx::HEADER + 3 * fidx::ENTRY + 8);
    CHECK(!flash_index_decode(b, img));
    CHECK(img.compat == COMPAT && img.blocks == 100 && img.next == ix.next_generation() && img.clock == ix.clock());
    CHECK(img.entries.size() == 3);
    for (size_t i = 0; i < 3 && i < img.entries.size(); ++i) {
        const FlashEntry &x = img.entries[i], &y = ix.entries()[i];
        CHECK(x.generation == y.generation && x.used == y.used && x.base == y.base && x.blocks == y.blocks &&
              x.position == y.position && x.ns == y.ns && x.tokens.empty() && !std::memcmp(&x.manifest, &y.manifest, sizeof x.manifest));
    }
    // every cut and every flipped byte is refused
    bool all = true;
    for (size_t n = 0; n < b.size(); ++n) all &= flash_index_decode(b.substr(0, n), img) != nullptr;
    CHECK(all);
    all = true;
    for (size_t i = 0; i < b.size(); ++i) { std::string c = b; c[i] ^= 0x10; all &= flash_index_decode(c, img) != nullptr; }
    CHECK(all);
    // a longer file with a good checksum (one entry too many for its count)
    { std::string c = b.substr(0, b.size() - 8) + b.substr(fidx::HEADER, fidx::ENTRY) + "12345678"; reseal(c);
      CHECK(flash_index_decode(c, img) != nullptr); }
    // two entries of one generation, the checksum made good
    { std::string c = b; std::memcpy(&c[fidx::HEADER + fidx::ENTRY], &c[fidx::HEADER], 8); reseal(c);
      const char* why = flash_index_decode(c, img);
      CHECK(why && std::string(why).find("generation") != std::string::npos); }
    // a token count that is not position + 1
    { std::string c = b; uint64_t bad = 5; std::memcpy(&c[fidx::HEADER + 4 * 8 + 2 * 4 + G6KV_NAMESPACE_BYTES], &bad, 8); reseal(c);
      CHECK(flash_index_decode(c, img) != nullptr); }
    // the old inline-token format is not this one
    { std::string c = b; const uint64_t old = 0x31584449484c4646ull; std::memcpy(&c[0], &old, 8); reseal(c);
      CHECK(flash_index_decode(c, img) != nullptr); }
}

static void files_round_trip(const std::string& dir) {
    const std::string path = dir + "/flash-ckpt.index";
    FlashIndexFiles f(path);
    FlashIndexImage img;
    uint64_t stale = 9;
    CHECK(f.load(COMPAT, 100, img, stale) && std::string(f.load(COMPAT, 100, img, stale)) == "no index" && img.entries.empty());
    FlashIndex ix(100);
    capture(ix, 64, 1); capture(ix, 128, 2); capture(ix, 640, 3);
    ix.touch(2);
    f.save(COMPAT, ix);
    CHECK(access((dir + "/flash-ckpt.tokens/1").c_str(), F_OK) == 0 && access((dir + "/flash-ckpt.tokens/3").c_str(), F_OK) == 0);
    CHECK(access((path + ".tmp").c_str(), F_OK) != 0);
    FlashIndexFiles g(path);
    CHECK(!g.load(COMPAT, 100, img, stale) && stale == 0);
    CHECK(same_entries(img.entries, ix.entries()) && img.next == ix.next_generation() && img.clock == ix.clock());
    // back into an index: the same digest, and the next capture continues the generations and the clock
    FlashIndex back(100);
    back.adopt(img.next, img.clock, img.entries);
    CHECK(back.digest() == ix.digest());
    // a token file is written once: a later save leaves it alone (a rewrite renames a new file over it: a new inode)
    auto inode = [&](uint64_t g) { struct stat st{}; return ::stat((dir + "/flash-ckpt.tokens/" + std::to_string(g)).c_str(), &st) ? 0 : st.st_ino; };
    const auto i1 = inode(1);
    capture(ix, 64, 4);
    f.save(COMPAT, ix);
    CHECK(i1 && inode(1) == i1);
    CHECK(access((dir + "/flash-ckpt.tokens/4").c_str(), F_OK) == 0);
    // a dropped entry's token file goes with the save
    ix.drop(4);
    f.save(COMPAT, ix);
    CHECK(access((dir + "/flash-ckpt.tokens/4").c_str(), F_OK) != 0);
    // another compat or store size starts empty, and its token files go
    { FlashIndexFiles h(path); const char* why = h.load(COMPAT + 1, 100, img, stale);
      CHECK(why && std::string(why) == "another engine or layout" && img.entries.empty()); }
    f.save(COMPAT, ix);
    { FlashIndexFiles h(path); const char* why = h.load(COMPAT, 101, img, stale);
      CHECK(why && std::string(why) == "another store size" && img.entries.empty()); }
    f = FlashIndexFiles(path);
    f.save(COMPAT, ix);
    // a cut or flipped index file on disk starts empty
    const std::string good = slurp(path);
    spit(path, good.substr(0, good.size() / 2));
    { FlashIndexFiles h(path); CHECK(h.load(COMPAT, 100, img, stale) != nullptr && img.entries.empty()); }
    std::string flip = good; flip[fidx::HEADER + 3] ^= 1; spit(path, flip);
    { FlashIndexFiles h(path); CHECK(h.load(COMPAT, 100, img, stale) != nullptr && img.entries.empty()); }
    // a lost or changed token file drops that entry only
    f = FlashIndexFiles(path);
    f.save(COMPAT, ix);
    std::filesystem::remove(dir + "/flash-ckpt.tokens/2");
    { std::string t = slurp(dir + "/flash-ckpt.tokens/3"); t[0] ^= 1; spit(dir + "/flash-ckpt.tokens/3", t); }
    spit(dir + "/flash-ckpt.tokens/77", "orphan");
    { FlashIndexFiles h(path); CHECK(!h.load(COMPAT, 100, img, stale) && stale == 2 && img.entries.size() == 1 &&
                                     img.entries[0].generation == 1);
      CHECK(access((dir + "/flash-ckpt.tokens/77").c_str(), F_OK) != 0 && access((dir + "/flash-ckpt.tokens/3").c_str(), F_OK) != 0); }
    // remove(): no index, and the next load cleans the token folder
    f.remove();
    { FlashIndexFiles h(path); const char* why = h.load(COMPAT, 100, img, stale);
      CHECK(why && std::string(why) == "no index" && std::filesystem::is_empty(dir + "/flash-ckpt.tokens")); }
}

static void adopt_rules() {
    auto entry = [](uint64_t gen, uint64_t base, uint64_t blocks, uint64_t used) {
        FlashEntry x; x.generation = gen; x.base = base; x.blocks = blocks; x.used = used; return x;
    };
    // any order in, base order out
    FlashIndex ix(100);
    ix.adopt(10, 20, {entry(3, 50, 10, 5), entry(1, 0, 10, 9), entry(2, 20, 10, 7)});
    CHECK(ix.size() == 3 && ix.entries()[0].base == 0 && ix.entries()[1].base == 20 && ix.entries()[2].base == 50);
    CHECK(ix.next_generation() == 10 && ix.clock() == 20);
    // the least recently used goes first: gen 3 (used 5), then gen 2 (used 7)
    uint64_t at = 0;
    std::vector<uint64_t> gone;
    CHECK(ix.reserve(65, at, &gone) && gone.size() == 1 && gone[0] == 3 && at == 30);
    gone.clear();
    CHECK(ix.reserve(85, at, &gone) && gone.size() == 1 && gone[0] == 2 && at == 10);
    // a capture after adopt continues the generations, and a touch continues the clock
    FlashIndex iy(100);
    iy.adopt(10, 20, {entry(1, 0, 10, 9)});
    iy.touch(1);
    CHECK(iy.entries()[0].used == 21);
    FlashEntry n = entry(10, 10, 5, 0);
    iy.add(n);
    CHECK(iy.next_generation() == 11 && iy.find(10));
    // refused: overlaps, past the store, generation >= next, use past the clock, two of one generation, no blocks
    auto refused = [](uint64_t next, uint64_t clock, std::vector<FlashEntry> v) {
        FlashIndex z(100);
        try { z.adopt(next, clock, std::move(v)); } catch (const std::exception&) { return z.size() == 0; }
        return false;
    };
    CHECK(refused(10, 20, {entry(1, 0, 10, 1), entry(2, 9, 10, 2)}));
    CHECK(refused(10, 20, {entry(1, 95, 10, 1)}));
    CHECK(refused(10, 20, {entry(1, 0, 101, 1)}));
    CHECK(refused(10, 20, {entry(1, UINT64_MAX - 2, 5, 1)}));
    CHECK(refused(10, 20, {entry(10, 0, 10, 1)}));
    CHECK(refused(10, 20, {entry(1, 0, 10, 21)}));
    CHECK(refused(10, 20, {entry(1, 0, 10, 1), entry(1, 20, 10, 2)}));
    CHECK(refused(10, 20, {entry(1, 0, 0, 1)}));
    CHECK(!refused(10, 20, {entry(1, 90, 10, 20)}));
}

// The crash case: a capture's reserve drops the least recently used entry, and the new entry's blocks then overwrite
// its range. The index on disk must lose the dropped entry before those writes (capture saves when reserve dropped),
// so a crash during the writes cannot bring it back.
static void crash_after_reserve(const std::string& dir) {
    const std::string path = dir + "/flash-ckpt.index";
    FlashIndexFiles f(path);
    FlashIndex ix(blocks_for(640) * 2);
    const uint64_t a = capture(ix, 640, 1), b = capture(ix, 640, 2);
    f.save(COMPAT, ix);
    ix.touch(b);
    // the next capture needs a's blocks: reserve drops a, capture saves, then writes over a's range ... and crashes
    uint64_t base = 0;
    std::vector<uint64_t> gone;
    CHECK(ix.reserve(blocks_for(640), base, &gone) && gone.size() == 1 && gone[0] == a && base == 0);
    if (!gone.empty()) f.save(COMPAT, ix);
    FlashIndexImage img;
    uint64_t stale = 0;
    FlashIndexFiles g(path);
    CHECK(!g.load(COMPAT, ix.blocks(), img, stale));
    CHECK(img.entries.size() == 1 && img.entries[0].generation == b);
    CHECK(access((dir + "/flash-ckpt.tokens/" + std::to_string(a)).c_str(), F_OK) != 0);
    // after the reload the next capture takes the next generation (b + 1)
    FlashIndex back(ix.blocks());
    back.adopt(img.next, img.clock, img.entries);
    CHECK(capture(back, 640, 3) == b + 1);
}

// A store of blocks in memory, written as FlashCheckpoints::capture writes (flash_checkpoint.cuh): the manifest last,
// page i of every binding (latent rows, then indexer rows) in block base + 1 + i, then the KDA state stream. The
// test computes the manifest sums as the device does, from each cache's pages end to end.
struct FakeStore {
    FlashLayout L;
    std::vector<std::vector<uint8_t>> b;
    FakeStore(uint64_t n) {
        L.slot = 4096; L.parts = {{1024, 512}, {2048, 0}}; L.state_bytes = 3 * 4096 + 1000;
        b.assign(n, std::vector<uint8_t>(L.slot, 0xEE));
    }
    FlashRead read() {
        return [this](uint64_t first, int n, uint8_t* dst) {
            for (int i = 0; i < n; ++i) {
                if (first + i >= b.size()) throw std::runtime_error("read past the store");
                std::memcpy(dst + size_t(i) * L.slot, b[first + i].data(), L.slot);
            }
        };
    }
    // capture at `position` from caches filled from `seed`; manifest: false = crash before it is written
    FlashEntry write(FlashIndex& ix, uint32_t position, int seed, bool manifest = true, std::vector<uint64_t>* gone = nullptr) {
        const uint64_t np = (position + 63) / 64;
        FlashEntry x;
        x.blocks = L.entry_blocks(position);
        if (!ix.reserve(x.blocks, x.base, gone)) throw std::runtime_error("no room");
        x.generation = ix.next_generation(); x.position = position;
        for (uint32_t i = 0; i <= position; ++i) x.tokens.push_back(int(i * 3 + seed));
        x.manifest = flash_manifest(x.generation, COMPAT, position, x.tokens, x.ns, np, L.page_bytes(), L.state_bytes);
        uint64_t rng = 0x9E3779B97F4A7C15ull * uint64_t(seed + 1);
        auto fill = [&](size_t bytes) { std::vector<uint64_t> v(bytes / 8); for (auto& w : v) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; w = rng; } return v; };
        uint64_t first = 0, page_sum = 0;
        for (size_t k = 0; k < L.parts.size(); ++k) {
            for (int c = 0; c < 2; ++c) {
                const size_t bytes = c ? L.parts[k].second : L.parts[k].first;
                if (!bytes) continue;
                const auto cache = fill(bytes * np);             // the device cache: pages [0, np) end to end
                page_sum += ckpt_sum_host(cache.data(), bytes * np, first);
                first += np * bytes / 8;
                size_t off = c ? L.parts[k].first : 0;
                for (size_t j = 0; j < k; ++j) off += L.parts[j].first + L.parts[j].second;
                for (uint64_t i = 0; i < np; ++i)
                    std::memcpy(b[x.base + 1 + i].data() + off, reinterpret_cast<const uint8_t*>(cache.data()) + i * bytes, bytes);
            }
        }
        const auto state = fill(L.state_bytes);
        x.manifest.page_sum = page_sum;
        x.manifest.state_sum = ckpt_sum_host(state.data(), L.state_bytes);
        for (uint64_t j = 0; j < L.state_blocks(); ++j) {
            auto& blk = b[x.base + 1 + np + j];
            std::fill(blk.begin(), blk.end(), 0);
            std::memcpy(blk.data(), reinterpret_cast<const uint8_t*>(state.data()) + j * L.slot, std::min<size_t>(L.slot, L.state_bytes - j * L.slot));
        }
        x.manifest.seal();
        if (manifest) {
            std::fill(b[x.base].begin(), b[x.base].end(), 0);
            std::memcpy(b[x.base].data(), &x.manifest, sizeof x.manifest);
            ix.add(x);
        }
        return x;
    }
};
static std::string check(const FlashEntry& x, FakeStore& st, int threads = 4) {
    const char* why = flash_entry_check(x, COMPAT, st.b.size(), st.L, st.read(), 3, threads);
    return why ? why : "ok";
}

static void entry_checks() {
    FakeStore st(200);
    FlashIndex ix(200);
    const FlashEntry a = st.write(ix, 640, 1), c = st.write(ix, 64, 2);
    CHECK(check(a, st) == "ok" && check(c, st) == "ok" && check(a, st, 1) == "ok");
    CHECK(flash_entry_check(a, COMPAT + 1, st.b.size(), st.L, st.read()) != nullptr);
    { FlashEntry y = a; y.tokens[5] ^= 1; CHECK(check(y, st) == "other tokens"); }
    { FlashEntry y = a; y.blocks += 1; CHECK(check(y, st) == "other sizes"); }
    { FlashEntry y = a; y.manifest.self_sum ^= 1; CHECK(check(y, st) == "manifest checksum"); }
    { FlashEntry y = a; y.terminal = true; CHECK(check(y, st) == "a terminal entry"); }
    { FlashEntry y = a; y.base = 195; CHECK(check(y, st) == "past the store"); }
    // a byte of a page, of the KDA state, or of the manifest block changed on disk
    auto flipped = [&](uint64_t block, size_t at) {
        st.b[block][at] ^= 0x40; const std::string r = check(a, st); st.b[block][at] ^= 0x40; return r;
    };
    CHECK(flipped(a.base + 1, 0) == "page checksum");
    CHECK(flipped(a.base + 1 + 9, 1024 + 511) == "page checksum");      // the last page's indexer rows
    CHECK(flipped(a.base + 1 + 3, 1536 + 2047) == "page checksum");     // the second binding
    CHECK(flipped(a.base + 1 + 10, 7) == "KDA state checksum");
    CHECK(flipped(a.base + 1 + 10 + 3, 999) == "KDA state checksum");   // the last state block's last byte
    CHECK(flipped(a.base + 1 + 2, 4000) == "ok");                       // a page block's padding is not summed
    CHECK(flipped(a.base, 8) != "ok");
    // two pages swapped: the sum numbers every word by its place
    std::swap(st.b[a.base + 1], st.b[a.base + 2]);
    CHECK(check(a, st) == "page checksum");
    std::swap(st.b[a.base + 1], st.b[a.base + 2]);
    // a read error reaches the caller (FlashCheckpoints::verify drops the entry with it)
    bool threw = false;
    const FlashRead bad = [](uint64_t, int, uint8_t*) { throw std::runtime_error("read"); };
    try { flash_entry_check(a, COMPAT, st.b.size(), st.L, bad); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

// The crash case again, with an index that still names the dropped entry (as the first version wrote it, before a
// capture saved after reserve): the new entry starts at the dropped one's base and crashes before its manifest, so the
// old manifest is still in place, but the pages under it are not its own and the page checksum drops it.
static void crash_with_stale_index(const std::string& dir) {
    FakeStore st(2 * FakeStore(0).L.entry_blocks(640));
    FlashIndex ix(st.b.size());
    const FlashEntry a = st.write(ix, 640, 1), b = st.write(ix, 640, 2);
    FlashIndexFiles f(dir + "/flash-ckpt.index");
    f.save(COMPAT, ix);   // names a and b
    ix.touch(b.generation);
    std::vector<uint64_t> gone;
    const FlashEntry c = st.write(ix, 640, 3, false, &gone);   // drops a, writes over it, no manifest yet: the crash
    CHECK(gone.size() == 1 && gone[0] == a.generation && c.base == a.base);
    FlashIndexImage img;
    uint64_t stale = 0;
    FlashIndexFiles g(dir + "/flash-ckpt.index");
    CHECK(!g.load(COMPAT, ix.blocks(), img, stale) && img.entries.size() == 2);
    for (const auto& x : img.entries) CHECK(check(x, st) == (x.generation == a.generation ? "page checksum" : "ok"));
}

// one capture's save at 64 entries of 262,145 tokens: the new entry's token file and the index, against the whole
// index with every entry's tokens inline (the first format)
static void bench(const std::string& dir) {
    const uint32_t P = 262144;
    FlashIndex ix(64 * blocks_for(P) + blocks_for(P));
    for (int i = 0; i < 64; ++i) capture(ix, P, i);
    FlashIndexFiles f(dir + "/flash-ckpt.index");
    f.save(COMPAT, ix);
    capture(ix, P, 99);
    auto t0 = std::chrono::steady_clock::now();
    f.save(COMPAT, ix);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string whole = flash_index_encode(COMPAT, ix);
    for (const auto& x : ix.entries()) whole.append(reinterpret_cast<const char*>(x.tokens.data()), x.tokens.size() * 4);
    t0 = std::chrono::steady_clock::now();
    fidx::write_durable(dir + "/whole.index", whole.data(), whole.size());
    const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("flash_ckpt_index_test bench: 65 entries of %u tokens: one capture's save %.2f ms (index %zu B + one %zu B token "
                "file); the whole index with tokens inline (%.1f MiB) %.2f ms\n", P + 1, ms, flash_index_encode(COMPAT, ix).size(),
                size_t(P + 1) * 4, whole.size() / 1048576.0, wms);
}

int main(int argc, char** argv) {
    std::string tmpl = (std::filesystem::temp_directory_path() / "fckidxXXXXXX").string();
    if (!mkdtemp(&tmpl[0])) { std::perror("mkdtemp"); return 1; }
    const std::string dir = tmpl;
    auto sub = [&](const char* n) { const std::string d = dir + "/" + n; std::filesystem::create_directories(d); return d; };
    if (argc > 1 && std::string(argv[1]) == "bench") {
        bench(argc > 2 ? std::string(argv[2]) : sub("bench"));
        std::filesystem::remove_all(dir);
        return 0;
    }
    bytes_round_trip();
    files_round_trip(sub("files"));
    adopt_rules();
    crash_after_reserve(sub("crash"));
    entry_checks();
    crash_with_stale_index(sub("crash2"));
    std::filesystem::remove_all(dir);
    if (!fails) std::printf("flash_ckpt_index_test: PASS\n");
    return fails ? 1 : 0;
}
