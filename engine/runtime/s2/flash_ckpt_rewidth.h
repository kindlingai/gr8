// GR8_KV_REWIDTH for GLM-5.3-Flash (`./gr8 glm53f scale`): the persisted aligned checkpoint store the old width left,
// rewritten for this rank before FlashCheckpoints opens it. Host only.
//
// Without CP an entry's pages and raw index rings are the same bytes on every rank; only the KDA state is split by head
// (kda::heads: layer l's heads rotate with l). So each carried entry keeps its manifest, tokens and pages, and its state
// stream is rebuilt for this rank: per KDA layer, S [nhmax][DK][DV] then conv [NHIST][3][nhmax*DK] (fp32, slots past
// this rank's heads zero), each head taken from the old rank that held it, then the rings as they were. Every old
// rank's state is checked against its own manifest first.
//
// The launcher renamed the old files to *.prev (flash-ckpt.prev, flash-ckpt.index.prev, flash-ckpt.tokens.prev): this
// box's own when it held an old rank, the others' through their kvman-peer (engine/kvman-rs/src/peer.rs). New files go
// to the usual names; the boot then checks every entry (flash_entry_check) and the ranks agree on the kept set.
#pragma once
#include "flash_ckpt_index.h"
#include "../../kda/math.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace g6s2 {
// GR8_KV_REWIDTH: <old_nr>,<old_rank|x>,<token_file>,<peer0>,...,<peer{old_nr-1}> (x: this box held no old rank)
struct FlashRewidthSpec {
    int old_nr = 0, old_rank = -1;
    std::string token_file;
    std::vector<std::string> peers;
};
inline const char* flash_rewidth_spec(const std::string& s, FlashRewidthSpec& out) {
    out = FlashRewidthSpec{};
    std::vector<std::string> f;
    for (size_t a = 0;;) {
        const size_t e = s.find(',', a);
        f.push_back(s.substr(a, e == std::string::npos ? std::string::npos : e - a));
        if (e == std::string::npos) break;
        a = e + 1;
    }
    const char* bad = "GR8_KV_REWIDTH: want <old_nr>,<old_rank|x>,<token_file>,<peer0>,...";
    if (f.size() < 4) return bad;
    char* e = nullptr;
    out.old_nr = int(std::strtol(f[0].c_str(), &e, 10));
    if (*e || out.old_nr <= 0 || f.size() != size_t(3 + out.old_nr)) return bad;
    if (f[1] != "x") {
        out.old_rank = int(std::strtol(f[1].c_str(), &e, 10));
        if (f[1].empty() || *e || out.old_rank < 0 || out.old_rank >= out.old_nr) return bad;
    }
    out.token_file = f[2];
    out.peers.assign(f.begin() + 3, f.end());
    return out.token_file.empty() ? bad : nullptr;
}

namespace frw {
[[noreturn]] inline void fail(const std::string& what) { throw std::runtime_error(what); }
// the old ranks' files: this box's own (the old rank it held, in dir) and the others' through their kvman-peer
class Src {
    int old_rank_;
    std::string dir_, token_;
    std::vector<std::string> peers_;
    std::vector<g6kv_peer*> conn_;
    g6kv_peer* peer(int q) {
        if (!conn_[size_t(q)]) {
            conn_[size_t(q)] = g6kv_peer_open(peers_[size_t(q)].c_str(), token_.c_str());
            if (!conn_[size_t(q)]) fail("peer " + std::to_string(q) + ": " + g6kv_peer_error());
        }
        return conn_[size_t(q)];
    }
public:
    Src(const FlashRewidthSpec& s, const std::string& dir, const std::string& token)
        : old_rank_(s.old_rank), dir_(dir), token_(token), peers_(s.peers), conn_(s.peers.size(), nullptr) {}
    ~Src() { for (g6kv_peer* p : conn_) g6kv_peer_close(p); }
    Src(const Src&) = delete;
    Src& operator=(const Src&) = delete;
    // n bytes of `name` at off, of old rank q's store
    void read(int q, const std::string& name, uint64_t off, uint64_t n, uint8_t* dst) {
        if (q != old_rank_) {
            if (g6kv_peer_read(peer(q), name.c_str(), off, n, dst)) fail(std::string("peer read: ") + g6kv_peer_error());
            return;
        }
        const std::string path = dir_ + "/" + name;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) fail("open " + path + ": " + std::strerror(errno));
        for (uint64_t at = 0; at < n;) {
            const ssize_t r = ::pread(fd, dst + at, size_t(n - at), off_t(off + at));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { ::close(fd); fail("read " + path + (r ? std::string(": ") + std::strerror(errno) : ": short")); }
            at += uint64_t(r);
        }
        ::close(fd);
    }
    // the whole of `name` (empty with missing = true when it does not exist)
    std::string read_all(int q, const std::string& name, bool* missing = nullptr) {
        uint64_t n = 0;
        if (q == old_rank_) {
            struct stat st{};
            if (::stat((dir_ + "/" + name).c_str(), &st)) {
                if (missing && errno == ENOENT) { *missing = true; return {}; }
                fail("stat " + dir_ + "/" + name + ": " + std::strerror(errno));
            }
            n = uint64_t(st.st_size);
        } else {
            const int64_t s = g6kv_peer_size(peer(q), name.c_str());
            if (s < 0) fail("peer " + std::to_string(q) + ": " + g6kv_peer_error());
            n = uint64_t(s);
        }
        std::string b(size_t(n), '\0');
        if (n) read(q, name, 0, n, reinterpret_cast<uint8_t*>(&b[0]));
        return b;
    }
};
inline void pwrite_all(int fd, const uint8_t* p, size_t n, uint64_t off, const std::string& path) {
    for (size_t at = 0; at < n;) {
        const ssize_t w = ::pwrite(fd, p + at, n - at, off_t(off + at));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) fail("write " + path + ": " + std::strerror(w ? errno : EIO));
        at += size_t(w);
    }
}
} // namespace frw

// Rewrites dir's Flash checkpoint store for rank `rank` of `nr`. compat: this store's compat line; old_compat(old_nr,
// old_rank, state_bytes): an old rank's. L: this rank's entry layout (FlashCheckpoints::layout_of), store_blocks its
// store's blocks, grid the cold prefill grid, pool_pages the pages a run may hold. >0: entries carried; 0: nothing
// carried (why); -1: failed (why), and every file it wrote is gone (the store starts empty).
inline int64_t flash_rewidth(const std::string& dir, const std::string& spec_text, const std::string& compat,
                             const std::function<std::string(int, int, size_t)>& old_compat, const FlashLayout& L, int nr,
                             int rank, uint64_t store_blocks, uint32_t grid, uint64_t pool_pages, std::string& why) {
    using namespace kda;
    FlashRewidthSpec spec;
    if (const char* e = flash_rewidth_spec(spec_text, spec)) { why = e; return -1; }
    const std::string out_blocks = dir + "/flash-ckpt", out_index = dir + "/flash-ckpt.index", out_tokens = dir + "/flash-ckpt.tokens";
    auto cleanup = [&] {
        std::error_code ec;
        std::filesystem::remove(out_blocks, ec);
        std::filesystem::remove(out_index, ec);
        std::filesystem::remove_all(out_tokens, ec);
    };
    try {
        std::string token;
        {
            std::ifstream t(spec.token_file);
            if (!t || !std::getline(t, token) || token.empty()) frw::fail("no token in " + spec.token_file);
            while (!token.empty() && std::isspace((unsigned char)token.back())) token.pop_back();
        }
        // one slot's KDA state per layer, at each width (floats)
        const size_t s_new = size_t(nhmax(nr)) * DK * DV, c_new = size_t(NHIST) * 3 * nhmax(nr) * DK;
        const size_t s_old = size_t(nhmax(spec.old_nr)) * DK * DV, c_old = size_t(NHIST) * 3 * nhmax(spec.old_nr) * DK;
        const size_t kda_new = size_t(N_KDA) * (s_new + c_new) * 4, kda_old = size_t(N_KDA) * (s_old + c_old) * 4;
        if (L.state_bytes < kda_new) frw::fail("the state stream is smaller than the KDA state");
        const size_t rings = L.state_bytes - kda_new, old_state = kda_old + rings;
        const uint64_t old_state_blocks = (old_state + L.slot - 1) / L.slot;
        frw::Src src(spec, dir, token);
        const int from = spec.old_rank >= 0 ? spec.old_rank : 0;   // the index, tokens, pages and rings: one old rank's
        bool missing = false;
        const std::string ib = src.read_all(from, "flash-ckpt.index.prev", &missing);
        if (missing) { why = "no flash-ckpt.index.prev"; return 0; }
        FlashIndexImage img;
        if (const char* e = flash_index_decode(ib, img)) { why = std::string("old index: ") + e; return 0; }
        std::vector<uint64_t> old_compats;
        for (int q = 0; q < spec.old_nr; ++q) {
            const std::string c = old_compat(spec.old_nr, q, old_state);
            old_compats.push_back(fnv64(c.data(), c.size()));
        }
        if (img.compat != old_compats[size_t(from)]) { why = "another engine or layout"; return 0; }

        // the entries to carry: aligned ones on this width's grid that fit its pool, most recently used first, packed from
        // block 0 until one does not fit
        std::vector<FlashEntry> want;
        for (const auto& x : img.entries)
            if (!x.terminal && grid && x.position % grid == 0 && (x.position + 63) / 64 <= pool_pages &&
                x.manifest.page_bytes == L.page_bytes() && x.manifest.state_bytes == old_state)
                want.push_back(x);
        std::sort(want.begin(), want.end(), [](const FlashEntry& a, const FlashEntry& b) { return a.used > b.used; });
        uint64_t at = 0;
        size_t fit = 0;
        for (; fit < want.size(); ++fit) {
            const uint64_t need = L.entry_blocks(want[fit].position);
            if (at + need > store_blocks) break;
            at += need;
        }
        want.resize(fit);

        const int fd = ::open(out_blocks.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) frw::fail("open " + out_blocks + ": " + std::strerror(errno));
        struct Close { int fd; ~Close() { ::close(fd); } } closer{fd};
        std::vector<FlashEntry> kept;
        std::string dropped;
        std::vector<uint8_t> buf(32 * L.slot), state(size_t(L.state_blocks() * L.slot));
        std::vector<std::vector<uint8_t>> old(size_t(spec.old_nr));
        at = 0;
        for (const FlashEntry& x : want) {
            const uint64_t np = (x.position + 63) / 64, base = at;
            at += L.entry_blocks(x.position);
            const std::string gen = std::to_string(x.generation);
            FlashEntry y = x;
            y.base = base; y.blocks = L.entry_blocks(x.position); y.used_ms = 0;
            const std::string tb = src.read_all(from, "flash-ckpt.tokens.prev/" + gen);
            if (tb.size() != x.manifest.tokens * sizeof(int) || fnv64(tb.data(), tb.size()) != x.manifest.token_hash) {
                dropped += " " + gen + " (tokens)";
                continue;
            }
            y.tokens.resize(x.manifest.tokens);
            std::memcpy(y.tokens.data(), tb.data(), tb.size());
            // every old rank's state, checked against its own manifest
            const char* bad = nullptr;
            for (int q = 0; q < spec.old_nr && !bad; ++q) {
                auto& o = old[size_t(q)];
                o.resize(size_t((1 + old_state_blocks) * L.slot));
                src.read(q, "flash-ckpt.prev", x.base * L.slot, L.slot, o.data());
                src.read(q, "flash-ckpt.prev", (x.base + 1 + np) * L.slot, old_state_blocks * L.slot, o.data() + L.slot);
                FlashManifest m;
                std::memcpy(&m, o.data(), sizeof m);
                if (m.unsealed() || m.generation != x.generation || m.compat != old_compats[size_t(q)] ||
                    m.position != x.position || m.state_bytes != old_state)
                    bad = "an old rank's manifest";
                else if (ckpt_sum_host(o.data() + L.slot, old_state) != m.state_sum)
                    bad = "an old rank's KDA state checksum";
            }
            if (bad) { dropped += " " + gen + " (" + bad + ")"; continue; }
            // this rank's heads of every KDA layer, each from the old rank that held it
            std::fill(state.begin(), state.end(), uint8_t(0));
            for (int li = 0; li < N_KDA; ++li) {
                const int l = li + li / 3;   // kda_slot's inverse: every fourth layer is attention
                float* S = reinterpret_cast<float*>(state.data() + size_t(li) * (s_new + c_new) * 4);
                float* C = S + s_new;
                int h0 = 0, nh = 0;
                heads(l, rank, nr, h0, nh);
                for (int hl = 0; hl < nh; ++hl) {
                    const int h = h0 + hl;
                    int q = 0, q0 = 0, qn = 0;
                    for (; q < spec.old_nr; ++q) { heads(l, q, spec.old_nr, q0, qn); if (h >= q0 && h < q0 + qn) break; }
                    const float* oS = reinterpret_cast<const float*>(old[size_t(q)].data() + L.slot + size_t(li) * (s_old + c_old) * 4);
                    const float* oC = oS + s_old;
                    std::memcpy(S + size_t(hl) * DK * DV, oS + size_t(h - q0) * DK * DV, size_t(DK) * DV * 4);
                    for (int k = 0; k < NHIST * 3; ++k)
                        std::memcpy(C + (size_t(k) * nhmax(nr) + hl) * DK, oC + (size_t(k) * nhmax(spec.old_nr) + (h - q0)) * DK, size_t(DK) * 4);
                }
            }
            std::memcpy(state.data() + kda_new, old[size_t(from)].data() + L.slot + kda_old, rings);
            // the pages, verbatim
            for (uint64_t i = 0; i < np; i += 32) {
                const uint64_t n = std::min<uint64_t>(32, np - i);
                src.read(from, "flash-ckpt.prev", (x.base + 1 + i) * L.slot, n * L.slot, buf.data());
                frw::pwrite_all(fd, buf.data(), size_t(n * L.slot), (base + 1 + i) * L.slot, out_blocks);
            }
            frw::pwrite_all(fd, state.data(), state.size(), (base + 1 + np) * L.slot, out_blocks);
            y.manifest.compat = fnv64(compat.data(), compat.size());
            y.manifest.state_bytes = L.state_bytes;
            y.manifest.state_sum = ckpt_sum_host(state.data(), L.state_bytes);
            y.manifest.seal();
            std::fill(buf.begin(), buf.begin() + L.slot, uint8_t(0));
            std::memcpy(buf.data(), &y.manifest, sizeof y.manifest);
            frw::pwrite_all(fd, buf.data(), L.slot, base * L.slot, out_blocks);
            kept.push_back(std::move(y));
        }
        if (::fsync(fd)) frw::fail("fsync " + out_blocks + ": " + std::strerror(errno));
        std::filesystem::create_directories(out_tokens);
        for (const auto& y : kept)
            fidx::write_durable(out_tokens + "/" + std::to_string(y.generation), y.tokens.data(), y.tokens.size() * sizeof(int), false);
        fidx::fsync_path(out_tokens, true);
        FlashIndex ix(store_blocks);
        ix.adopt(img.next, img.clock, kept);
        const std::string b = flash_index_encode(fnv64(compat.data(), compat.size()), ix);
        fidx::write_durable(out_index, b.data(), b.size());
        why = kept.empty() ? "no entries" : "";
        if (!dropped.empty()) why += (why.empty() ? "" : "; ") + std::string("dropped") + dropped;
        return int64_t(kept.size());
    } catch (const std::exception& e) {
        cleanup();
        why = e.what();
        return -1;
    }
}
} // namespace g6s2
