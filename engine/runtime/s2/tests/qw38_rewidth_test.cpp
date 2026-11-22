// GR8_KV_REWIDTH for Qwen3.8-Flash-Next (s2/qw38_ckpt_rewidth.h): builds a canonical Qwen3.8 state
// (full-attention KV pages, QSA index pages, GDN FP32 S + conv rings, PLE conv history), shards it at TP=2,
// rewrites to TP=1, back to TP=2, then to TP=3 and TP=4 and back to TP=2 over loopback kvman-peer servers.
// Asserts every rank's carried pages and state streams are byte-identical to canonical sharding at the new width.
#include "s2/qw38_ckpt_rewidth.h"
#include <csignal>
#include <cstdio>
#include <thread>
#include <sys/wait.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static const uint32_t GRID = 64;
static const uint64_t BLOCKS = 200000;
static const int ENTRIES = 2;

static std::string set_of(int nr) { return "qw38-set-tp" + std::to_string(nr); }
static std::string compat(int nr, int rank, size_t pb, size_t sb, const std::string& set) {
    return "qw38 set " + set + " nr " + std::to_string(nr) + " rank " + std::to_string(rank)
        + " pb " + std::to_string(pb) + " sb " + std::to_string(sb) + " engine e";
}
static uint64_t h64(const std::string& s) { return fnv64(s.data(), s.size()); }

// Canonical page & state generators (width-independent global head indexing)
static std::vector<uint8_t> shard_page(uint64_t g, uint64_t p, const Qw38CkptLayout& L) {
    std::vector<uint8_t> out(L.page_bytes(), 0);
    uint8_t* dst = out.data();
    const size_t hb = size_t(geom::HEAD_DIM) * 2;
    const size_t idx_b = Qw38CkptLayout::idx_layer_page_bytes();
    for (int l = 0; l < L.n_full; ++l) {
        for (int kv_plane = 0; kv_plane < 2; ++kv_plane)
            for (int tok = 0; tok < 64; ++tok)
                for (int kvl = 0; kvl < L.nh_kv(); ++kvl) {
                    const int gkv = L.kv0() + kvl;
                    const size_t off = ((size_t(kv_plane) * 64 + size_t(tok)) * size_t(L.nh_kv()) + size_t(kvl)) * hb;
                    for (size_t b = 0; b < hb; ++b)
                        dst[off + b] = uint8_t((g * 13 + p * 7 + uint64_t(l) * 31 + uint64_t(kv_plane) * 5 + uint64_t(tok) * 3 + uint64_t(gkv) * 97 + b) & 0xff);
                }
        uint8_t* idst = dst + L.kv_layer_page_bytes();
        for (size_t b = 0; b < idx_b; ++b)
            idst[b] = uint8_t((g * 23 + p * 11 + uint64_t(l) * 19 + b) & 0xff);
        dst += L.kv_layer_page_bytes() + idx_b;
    }
    return out;
}

static std::vector<uint8_t> shard_state(uint64_t g, const Qw38CkptLayout& L) {
    std::vector<uint8_t> out(L.state_bytes(), 0);
    const size_t dk_dv = size_t(geom::GDN_DK) * geom::GDN_DV;
    const size_t hb = size_t(geom::GDN_DK) * 2;
    for (int li = 0; li < L.n_gdn; ++li) {
        uint8_t* dst_s = out.data() + size_t(li) * (L.gdn_s_bytes() + L.gdn_c_bytes());
        float* sf = reinterpret_cast<float*>(dst_s);
        for (int lv = 0; lv < L.nh_v(); ++lv) {
            const int gv = L.v0() + lv;
            for (size_t i = 0; i < dk_dv; ++i)
                sf[size_t(lv) * dk_dv + i] = float((g * 7919 + uint64_t(li) * 131 + uint64_t(gv) * 17 + i) % 65521);
        }
        uint8_t* dst_c = dst_s + L.gdn_s_bytes();
        for (int tap = 0; tap < geom::GDN_NHIST; ++tap) {
            uint8_t* dt = dst_c + size_t(tap) * size_t(L.qkv_local()) * 2;
            for (int lk = 0; lk < L.nh_k(); ++lk) {
                const int gk = L.k0() + lk;
                for (size_t b = 0; b < hb; ++b) {
                    dt[size_t(lk) * hb + b] = uint8_t((g * 3 + uint64_t(li) * 5 + uint64_t(tap) * 7 + uint64_t(gk) * 11 + b) & 0xff);
                    dt[size_t(L.nh_k() + lk) * hb + b] = uint8_t((g * 13 + uint64_t(li) * 17 + uint64_t(tap) * 19 + uint64_t(gk) * 23 + b) & 0xff);
                }
            }
            for (int lv = 0; lv < L.nh_v(); ++lv) {
                const int gv = L.v0() + lv;
                for (size_t b = 0; b < hb; ++b)
                    dt[size_t(2 * L.nh_k() + lv) * hb + b] = uint8_t((g * 29 + uint64_t(li) * 31 + uint64_t(tap) * 37 + uint64_t(gv) * 41 + b) & 0xff);
            }
        }
    }
    const size_t ple_b = Qw38CkptLayout::ple_conv_bytes();
    uint8_t* dp = out.data() + L.state_bytes() - ple_b;
    for (size_t i = 0; i < ple_b; ++i) dp[i] = uint8_t((g * 53 + i * 7) & 0xff);
    return out;
}

static void pwrite_vec(const std::string& path, const std::vector<uint8_t>& b, uint64_t off) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    frw::pwrite_all(fd, b.data(), b.size(), off, path);
    ::close(fd);
}

static void write_store(const std::string& d, int nr, int rank) {
    std::filesystem::create_directories(d + "/flash-ckpt.tokens");
    const Qw38CkptLayout L{nr, rank, geom::N_FULL, geom::N_GDN, 4096};
    const std::string c = compat(nr, rank, L.page_bytes(), L.state_bytes(), set_of(nr));
    FlashIndex ix(BLOCKS);
    for (int k = 1; k <= ENTRIES; ++k) {
        FlashEntry x;
        x.position = GRID * uint32_t(k);
        x.blocks = L.entry_blocks(x.position);
        CHECK(ix.reserve(x.blocks, x.base));
        x.generation = ix.next_generation();
        for (uint32_t i = 0; i <= x.position; ++i) x.tokens.push_back(int(i * 3 + k));
        const uint64_t np = (uint64_t(x.position) + 63) / 64;
        const auto st = shard_state(x.generation, L);
        uint64_t psum = 0;
        for (uint64_t p = 0; p < np; ++p) {
            auto pg = shard_page(x.generation, p, L);
            psum += ckpt_sum_host(pg.data(), L.page_bytes(), p * (L.page_bytes() / 8));
            pg.resize(size_t(L.page_slot_blocks() * L.slot), 0);
            pwrite_vec(d + "/flash-ckpt", pg, (x.base + 1 + p * L.page_slot_blocks()) * L.slot);
        }
        auto st_pad = st;
        st_pad.resize(size_t(L.state_blocks() * L.slot), 0);
        pwrite_vec(d + "/flash-ckpt", st_pad, (x.base + 1 + np * L.page_slot_blocks()) * L.slot);
        x.manifest = flash_manifest(x.generation, h64(c), x.position, x.tokens, x.ns, np, L.page_bytes(), L.state_bytes());
        x.manifest.page_sum = psum;
        x.manifest.state_sum = ckpt_sum_host(st.data(), L.state_bytes());
        x.manifest.seal();
        std::vector<uint8_t> mb(L.slot, 0);
        std::memcpy(mb.data(), &x.manifest, sizeof x.manifest);
        pwrite_vec(d + "/flash-ckpt", mb, x.base * L.slot);
        fidx::write_durable(d + "/flash-ckpt.tokens/" + std::to_string(x.generation), x.tokens.data(), x.tokens.size() * sizeof(int));
        ix.add(x);
    }
    FlashIndexFiles(d + "/flash-ckpt.index").save(h64(c), ix);
}

static void move_to_prev(const std::string& d) {
    std::error_code ec;
    for (const char* n : {"flash-ckpt", "flash-ckpt.index", "flash-ckpt.tokens"}) {
        std::filesystem::remove_all(d + "/" + n + ".prev", ec);
        if (std::filesystem::exists(d + "/" + n))
            std::filesystem::rename(d + "/" + n, d + "/" + n + ".prev");
    }
}

struct PeerProc {
    pid_t pid = -1;
    std::string addr;
    ~PeerProc() {
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            int st = 0;
            ::waitpid(pid, &st, 0);
        }
    }
};

static void run_rewidth(const std::string& root, const std::string& peer_bin, const std::string& tok_file,
                        int old_nr, int new_nr, int port_base) {
    std::vector<std::unique_ptr<PeerProc>> peers;
    std::string peer_list;
    for (int q = 0; q < old_nr; ++q) {
        const std::string dir = root + "/tp" + std::to_string(old_nr) + "_r" + std::to_string(q);
        move_to_prev(dir);
        auto p = std::make_unique<PeerProc>();
        p->addr = "127.0.0.1:" + std::to_string(port_base + q);
        p->pid = ::fork();
        if (p->pid == 0) {
            ::execl(peer_bin.c_str(), peer_bin.c_str(), "--dir", dir.c_str(), "--listen", p->addr.c_str(),
                    "--token-file", tok_file.c_str(), (char*)nullptr);
            _exit(127);
        }
        if (q) peer_list += ",";
        peer_list += p->addr;
        peers.push_back(std::move(p));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    for (int r = 0; r < new_nr; ++r) {
        const std::string dir = root + "/tp" + std::to_string(new_nr) + "_r" + std::to_string(r);
        std::filesystem::create_directories(dir);
        const std::string spec = std::to_string(old_nr) + ",x," + tok_file + "," + set_of(old_nr) + "," + peer_list;
        const Qw38CkptLayout L{new_nr, r, geom::N_FULL, geom::N_GDN, 4096};
        const std::string c = compat(new_nr, r, L.page_bytes(), L.state_bytes(), set_of(new_nr));
        std::string why;
        const int64_t n = qw38_rewidth(dir, spec, c, compat, L, BLOCKS, GRID, 4096, why);
        if (n != ENTRIES) std::fprintf(stderr, "rewidth %d->%d r%d failed: n=%lld why=%s\n", old_nr, new_nr, r, (long long)n, why.c_str());
        CHECK(n == ENTRIES);

        // Verify index & byte-identical pages and states against direct canonical sharding at new_nr!
        std::ifstream ifs(dir + "/flash-ckpt.index", std::ios::binary);
        std::string ib((std::istreambuf_iterator<char>(ifs)), {});
        FlashIndexImage img;
        CHECK(flash_index_decode(ib, img) == nullptr);
        CHECK(img.compat == h64(c));
        CHECK(img.entries.size() == size_t(ENTRIES));
        const int fd = ::open((dir + "/flash-ckpt").c_str(), O_RDONLY);
        CHECK(fd >= 0);
        for (const auto& e : img.entries) {
            const uint64_t np = (uint64_t(e.position) + 63) / 64;
            for (uint64_t p = 0; p < np; ++p) {
                std::vector<uint8_t> got(L.page_bytes());
                CHECK(::pread(fd, got.data(), got.size(), off_t((e.base + 1 + p * L.page_slot_blocks()) * L.slot)) == ssize_t(got.size()));
                CHECK(got == shard_page(e.generation, p, L));
            }
            std::vector<uint8_t> got_s(L.state_bytes());
            CHECK(::pread(fd, got_s.data(), got_s.size(), off_t((e.base + 1 + np * L.page_slot_blocks()) * L.slot)) == ssize_t(got_s.size()));
            CHECK(got_s == shard_state(e.generation, L));
        }
        ::close(fd);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <kvman-peer>\n", argv[0]);
        return 2;
    }
    const std::string peer_bin = argv[1];
    const std::string root = "/tmp/qw38-rewidth-test-" + std::to_string(getpid());
    std::filesystem::create_directories(root);
    const std::string tok_file = root + "/token";
    const std::string tok_body = "deadbeef0123456789abcdef\n";
    fidx::write_durable(tok_file, tok_body.data(), tok_body.size());

    // Start at TP=2 -> rewrite to TP=1 -> TP=2 -> TP=3 -> TP=4 -> TP=2
    for (int r = 0; r < 2; ++r) write_store(root + "/tp2_r" + std::to_string(r), 2, r);
    run_rewidth(root, peer_bin, tok_file, 2, 1, 19300);
    run_rewidth(root, peer_bin, tok_file, 1, 2, 19310);
    run_rewidth(root, peer_bin, tok_file, 2, 3, 19320);
    run_rewidth(root, peer_bin, tok_file, 3, 4, 19330);
    run_rewidth(root, peer_bin, tok_file, 4, 2, 19340);

    std::filesystem::remove_all(root);
    std::printf("qw38_rewidth_test (2->1->2->3->4->2): %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
