// GR8_KV_REWIDTH for GLM-5.3-Flash (s2/flash_ckpt_rewidth.h): a canonical KDA state (a value pattern per layer and
// head), sharded to 4 ranks' persisted checkpoint stores, rewritten to TP=2, then 1, then back to 4 over loopback
// kvman-peer servers. Every rank's entries pass flash_entry_check at the new layout, the agreement lines are equal
// across ranks, and each rank's state stream equals the canonical state sharded directly at the new width.
// Build: g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/s2/tests/flash_rewidth_test.cpp
//        engine/target/release/libg6host.a -lpthread -ldl -lm -o /tmp/flash_rewidth_test
// Run:   /tmp/flash_rewidth_test engine/target/release/kvman-peer
#include "s2/flash_ckpt_rewidth.h"
#include <csignal>
#include <cstdio>
#include <thread>
#include <sys/wait.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static const size_t RINGS = 3 * 4096;   // the raw rings' bytes
static const uint32_t GRID = 128;
static const uint64_t BLOCKS = 80000;   // a store's blocks: two entries at TP=1 (37,000 blocks each)
static const int ENTRIES = 2;

static size_t s_of(int nr) { return size_t(kda::nhmax(nr)) * kda::DK * kda::DV; }
static size_t c_of(int nr) { return size_t(kda::NHIST) * 3 * kda::nhmax(nr) * kda::DK; }
static FlashLayout layout(int nr) {
    FlashLayout L;
    L.slot = 4096;
    L.parts = {{2048, 0}, {1024, 1024}};
    L.state_bytes = size_t(kda::N_KDA) * (s_of(nr) + c_of(nr)) * 4 + RINGS;
    return L;
}
// the model set id names the width, as the launcher's does (modelset.py describe_release)
static std::string set_of(int nr) { return "set-tp" + std::to_string(nr); }
static std::string compat(int nr, int rank, size_t state_bytes, const std::string& set) {
    return "flash set " + set + " nr " + std::to_string(nr) + " rank " + std::to_string(rank) + " state " + std::to_string(state_bytes) + " engine e";
}
static uint64_t h64(const std::string& s) { return fnv64(s.data(), s.size()); }

// the canonical state: one value per (entry, layer slot, head, word)
static float sval(uint64_t g, int li, int h, size_t i) { return float((g * 7919 + uint64_t(li) * 131 + uint64_t(h) * 17 + i) % 65521); }
static float cval(uint64_t g, int li, int h, int k, int d) { return -float((g * 31 + uint64_t(li) * 7 + uint64_t(h) * 3 + uint64_t(k) * 1009 + uint64_t(d)) % 4099); }
// rank `rank` of `nr`'s state stream, sharded directly from the canonical state
static std::vector<uint8_t> shard(uint64_t g, int nr, int rank) {
    const FlashLayout L = layout(nr);
    std::vector<uint8_t> st(L.state_bytes, 0);
    const size_t s = s_of(nr), c = c_of(nr);
    for (int li = 0; li < kda::N_KDA; ++li) {
        float* S = reinterpret_cast<float*>(st.data() + size_t(li) * (s + c) * 4);
        float* C = S + s;
        int h0 = 0, nh = 0;
        kda::heads(li + li / 3, rank, nr, h0, nh);
        for (int hl = 0; hl < nh; ++hl) {
            for (size_t i = 0; i < size_t(kda::DK) * kda::DV; ++i) S[size_t(hl) * kda::DK * kda::DV + i] = sval(g, li, h0 + hl, i);
            for (int k = 0; k < kda::NHIST * 3; ++k)
                for (int d = 0; d < kda::DK; ++d) C[(size_t(k) * kda::nhmax(nr) + hl) * kda::DK + d] = cval(g, li, h0 + hl, k, d);
        }
    }
    for (size_t i = 0; i < RINGS; ++i) st[st.size() - RINGS + i] = uint8_t(g * 29 + i * 7);
    return st;
}
static std::vector<uint8_t> page(uint64_t g, uint64_t i, const FlashLayout& L) {
    std::vector<uint8_t> b(L.slot, 0);
    for (size_t k = 0; k < L.page_bytes(); ++k) b[k] = uint8_t(g * 13 + i * 5 + k);
    return b;
}
// the page checksum in flash_entry_check's word numbering
static uint64_t page_sum(uint64_t g, uint64_t np, const FlashLayout& L) {
    std::vector<uint64_t> lf, xf;
    uint64_t first = 0;
    for (const auto& p : L.parts) { lf.push_back(first); first += np * (p.first / 8); xf.push_back(first); first += np * (p.second / 8); }
    uint64_t s = 0;
    for (uint64_t i = 0; i < np; ++i) {
        const auto b = page(g, i, L);
        const uint8_t* q = b.data();
        for (size_t k = 0; k < L.parts.size(); ++k) {
            s += ckpt_sum_host(q, L.parts[k].first, lf[k] + i * (L.parts[k].first / 8)); q += L.parts[k].first;
            if (L.parts[k].second) { s += ckpt_sum_host(q, L.parts[k].second, xf[k] + i * (L.parts[k].second / 8)); q += L.parts[k].second; }
        }
    }
    return s;
}
static void pwrite_all(const std::string& path, const std::vector<uint8_t>& b, uint64_t off) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    frw::pwrite_all(fd, b.data(), b.size(), off, path);
    ::close(fd);
}
static void prev(const std::string& d) {   // what the launcher's store prep does on a member
    std::error_code ec;
    for (const char* n : {"flash-ckpt", "flash-ckpt.index", "flash-ckpt.tokens"}) {
        std::filesystem::remove_all(d + "/" + n + ".prev", ec);
        std::filesystem::rename(d + "/" + n, d + "/" + n + ".prev");
    }
}
// rank q of nr's store at TP nr, as a boot that captured ENTRIES entries left it
static void write_old(const std::string& d, int nr, int q) {
    std::filesystem::create_directories(d);
    const FlashLayout L = layout(nr);
    const std::string c = compat(nr, q, L.state_bytes, set_of(nr));
    FlashIndex ix(BLOCKS);
    for (int k = 1; k <= ENTRIES; ++k) {
        FlashEntry x;
        x.position = GRID * uint32_t(k);
        x.blocks = L.entry_blocks(x.position);
        if (!ix.reserve(x.blocks, x.base)) throw std::runtime_error("no room");
        x.generation = ix.next_generation();
        for (uint32_t i = 0; i <= x.position; ++i) x.tokens.push_back(int(i * 3 + k));
        const uint64_t np = (x.position + 63) / 64;
        const auto st = shard(x.generation, nr, q);
        x.manifest = flash_manifest(x.generation, h64(c), x.position, x.tokens, x.ns, np, L.page_bytes(), L.state_bytes);
        x.manifest.page_sum = page_sum(x.generation, np, L);
        x.manifest.state_sum = ckpt_sum_host(st.data(), L.state_bytes);
        x.manifest.seal();
        for (uint64_t i = 0; i < np; ++i) pwrite_all(d + "/flash-ckpt", page(x.generation, i, L), (x.base + 1 + i) * L.slot);
        pwrite_all(d + "/flash-ckpt", st, (x.base + 1 + np) * L.slot);
        std::vector<uint8_t> mb(L.slot, 0);
        std::memcpy(mb.data(), &x.manifest, sizeof x.manifest);
        pwrite_all(d + "/flash-ckpt", mb, x.base * L.slot);
        ix.add(x);
    }
    FlashIndexFiles(d + "/flash-ckpt.index").save(h64(c), ix);
    prev(d);
}

struct PeerProc { pid_t pid = -1; std::string addr; };
static PeerProc start_peer(const std::string& bin, const std::string& dir, const std::string& token) {
    int fds[2];
    if (::pipe(fds)) throw std::runtime_error("pipe");
    PeerProc p;
    p.pid = ::fork();
    if (p.pid == 0) {
        ::dup2(fds[1], 1);
        ::close(fds[0]);
        ::execl(bin.c_str(), "kvman-peer", "--dir", dir.c_str(), "--listen", "127.0.0.1:0", "--token-file", token.c_str(), (char*)nullptr);
        ::_exit(127);
    }
    ::close(fds[1]);
    FILE* f = ::fdopen(fds[0], "r");
    char line[512] = {};
    if (!std::fgets(line, sizeof line, f)) throw std::runtime_error("kvman-peer did not start: " + bin);
    std::fclose(f);
    std::string s = line;
    p.addr = s.substr(s.rfind(' ') + 1);
    while (!p.addr.empty() && std::isspace((unsigned char)p.addr.back())) p.addr.pop_back();
    return p;
}

// one scale: the old ranks' stores in old_dirs (served), new rank r in new_dirs[r], with its old rank from[r] (-1: a
// joiner); every new rank at once, as the boxes do. Then the checks.
static void scale(const std::string& bin, const std::string& token, const std::vector<std::string>& old_dirs,
                  const std::vector<std::string>& new_dirs, const std::vector<int>& from) {
    const int onr = int(old_dirs.size()), nr = int(new_dirs.size());
    std::vector<PeerProc> peers;
    for (const auto& d : old_dirs) peers.push_back(start_peer(bin, d, token));
    std::string list;
    for (const auto& p : peers) list += "," + p.addr;
    const FlashLayout L = layout(nr);
    std::vector<int64_t> got(static_cast<size_t>(nr), -2);
    std::vector<std::string> why(static_cast<size_t>(nr));
    std::vector<std::thread> w;
    for (int r = 0; r < nr; ++r) {
        std::filesystem::create_directories(new_dirs[size_t(r)]);
        const std::string spec = std::to_string(onr) + "," + (from[size_t(r)] < 0 ? "x" : std::to_string(from[size_t(r)])) + "," + token +
                                 "," + set_of(onr) + list;
        w.emplace_back([&, r, spec] {
            got[size_t(r)] = flash_rewidth(new_dirs[size_t(r)], spec, compat(nr, r, L.state_bytes, set_of(nr)), compat, L, nr, r, BLOCKS,
                                           GRID, 1 << 20, why[size_t(r)]);
        });
    }
    for (auto& t : w) t.join();
    for (const auto& p : peers) { ::kill(p.pid, SIGTERM); ::waitpid(p.pid, nullptr, 0); }
    std::vector<std::string> agree;
    for (int r = 0; r < nr; ++r) {
        if (got[size_t(r)] != ENTRIES) std::fprintf(stderr, "%d -> %d rank %d: %lld (%s)\n", onr, nr, r, (long long)got[size_t(r)], why[size_t(r)].c_str());
        CHECK(got[size_t(r)] == ENTRIES);
        const std::string d = new_dirs[size_t(r)];
        const uint64_t c = h64(compat(nr, r, L.state_bytes, set_of(nr)));
        FlashIndexImage img;
        uint64_t stale = 0;
        CHECK(FlashIndexFiles(d + "/flash-ckpt.index").load(c, BLOCKS, img, stale) == nullptr);
        CHECK(img.entries.size() == size_t(ENTRIES) && stale == 0);
        const FlashRead read = [&](uint64_t first, int n, uint8_t* dst) {
            const int fd = ::open((d + "/flash-ckpt").c_str(), O_RDONLY | O_CLOEXEC);
            const ssize_t want = ssize_t(size_t(n) * L.slot);
            const bool ok = fd >= 0 && ::pread(fd, dst, size_t(want), off_t(first * L.slot)) == want;
            ::close(fd);
            if (!ok) throw std::runtime_error("read");
        };
        for (const auto& x : img.entries) {
            const char* bad = flash_entry_check(x, c, BLOCKS, L, read);
            if (bad) std::fprintf(stderr, "%d -> %d rank %d entry %llu: %s\n", onr, nr, r, (unsigned long long)x.generation, bad);
            CHECK(!bad);
            std::vector<uint8_t> st(size_t(L.state_blocks() * L.slot));
            read(x.base + 1 + (x.position + 63) / 64, int(L.state_blocks()), st.data());
            st.resize(L.state_bytes);
            CHECK(st == shard(x.generation, nr, r));
        }
        FlashIndex ix(BLOCKS);
        ix.adopt(img.next, img.clock, img.entries);
        agree.push_back(flash_agree_summary(ix));
    }
    for (const auto& a : agree) CHECK(a == agree[0]);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: flash_rewidth_test KVMAN_PEER\n"); return 2; }
    const std::string bin = argv[1];
    const std::string root = (std::filesystem::temp_directory_path() / ("flash_rewidth_" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(root);
    const std::string token = root + "/token";
    { std::ofstream(token) << "00ff12ab\n"; }
    std::vector<std::string> box;
    for (int b = 0; b < 4; ++b) box.push_back(root + "/box" + std::to_string(b));
    for (int q = 0; q < 4; ++q) write_old(box[size_t(q)], 4, q);
    // 4 -> 2: boxes 0 and 1 stay (in place); 2 and 3 leave
    scale(bin, token, box, {box[0], box[1]}, {0, 1});
    for (int b = 0; b < 2; ++b) prev(box[size_t(b)]);
    // 2 -> 1: box 0 stays
    scale(bin, token, {box[0], box[1]}, {box[0]}, {0});
    prev(box[0]);
    // 1 -> 4: box 0 stays, boxes 1-3 join with their stale folders
    scale(bin, token, {box[0]}, box, {0, -1, -1, -1});
    // a wrong token: every rank fails and leaves no files
    prev(box[0]);
    {
        const std::string bad = root + "/bad";
        { std::ofstream(bad) << "0bad\n"; }
        PeerProc p = start_peer(bin, box[0], token);
        std::string why;
        const std::string fresh = root + "/fresh";
        std::filesystem::create_directories(fresh);
        const FlashLayout L = layout(2);
        const std::string peers4 = "," + p.addr + "," + p.addr + "," + p.addr + "," + p.addr;
        CHECK(flash_rewidth(fresh, "4,x," + bad + "," + set_of(4) + peers4, compat(2, 0, L.state_bytes, set_of(2)), compat, L,
                            2, 0, BLOCKS, GRID, 1 << 20, why) == -1);
        CHECK(!std::filesystem::exists(fresh + "/flash-ckpt.index") && !std::filesystem::exists(fresh + "/flash-ckpt"));
        // a store written by another engine, or under another model set than the old width's: nothing carried
        CHECK(flash_rewidth(fresh, "4,x," + token + "," + set_of(4) + peers4, compat(2, 0, L.state_bytes, set_of(2)),
                            [](int n, int q, size_t sb, const std::string& s) { return compat(n, q, sb, s) + " other"; }, L, 2, 0,
                            BLOCKS, GRID, 1 << 20, why) == 0);
        CHECK(why == "another engine or layout");
        CHECK(flash_rewidth(fresh, "4,x," + token + ",set-other" + peers4, compat(2, 0, L.state_bytes, set_of(2)), compat, L, 2,
                            0, BLOCKS, GRID, 1 << 20, why) == 0);
        CHECK(why == "another engine or layout");
        ::kill(p.pid, SIGTERM); ::waitpid(p.pid, nullptr, 0);
    }
    std::filesystem::remove_all(root);
    if (fails) { std::fprintf(stderr, "flash_rewidth_test: %d failed\n", fails); return 1; }
    std::printf("flash_rewidth_test: ok\n");
    return 0;
}
