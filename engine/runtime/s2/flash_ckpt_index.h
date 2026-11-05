// persist=on for Flash session checkpoints (flash_checkpoint.h), the files beside an ssd: store's blocks. Host only.
//
//   flash-ckpt.index          the index: a header, each entry without its tokens, then the fnv64 of everything before
//                             it. Small (about 200 bytes an entry), so it is rewritten whole after every change.
//   flash-ckpt.tokens/<gen>   one entry's p + 1 tokens, written once, before the first index that names it. The
//                             entry's manifest holds their count and fnv64.
//
// Every file is written to a temporary name, synced, renamed over the old one, and the folder synced, so a crash or a
// power loss leaves the old file or the new one. A token file no index names is removed at the next load or save.
#pragma once
#include "flash_checkpoint.h"
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace g6s2 {
// a persisted index: the store it belongs to (compat, blocks), the index clocks and the entries (tokens filled in by
// FlashIndexFiles::load from their files)
struct FlashIndexImage {
    uint64_t compat = 0, blocks = 0, next = 1, clock = 0;
    std::vector<FlashEntry> entries;
};

namespace fidx {
constexpr uint64_t MAGIC = 0x32584449484c4646ull;   // "FFLHIDX2": entries without tokens (FFLHIDX1 kept them inline)
constexpr size_t HEADER = 6 * 8;                     // magic, compat, blocks, next, clock, entries
constexpr size_t ENTRY = 4 * 8 + 2 * 4 + G6KV_NAMESPACE_BYTES + 8 + sizeof(FlashManifest);
template<class T> void put(std::string& b, const T& x) { b.append(reinterpret_cast<const char*>(&x), sizeof x); }
template<class T> bool get(const std::string& b, size_t& at, T& x) {
    if (b.size() - at < sizeof x) return false;
    std::memcpy(&x, b.data() + at, sizeof x); at += sizeof x; return true;
}
[[noreturn]] inline void fail(const std::string& what) { throw std::system_error(errno, std::generic_category(), what); }
inline void fsync_path(const std::string& path, bool dir) {
    const int fd = ::open(path.c_str(), (dir ? O_RDONLY | O_DIRECTORY : O_RDONLY) | O_CLOEXEC);
    if (fd < 0) fail("open " + path);
    const int rc = ::fsync(fd);
    const int e = errno;
    ::close(fd);
    if (rc) { errno = e; fail("fsync " + path); }
}
inline std::string parent(const std::string& path) {
    const std::string p = std::filesystem::path(path).parent_path().string();
    return p.empty() ? "." : p;
}
// `path` holds `bytes` after a crash at any point, or its old contents. sync_dir false: the caller syncs the folder
// once after several files.
inline void write_durable(const std::string& path, const void* bytes, size_t n, bool sync_dir = true) {
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) fail("open " + tmp);
    const char* p = static_cast<const char*>(bytes);
    for (size_t left = n; left;) {
        const ssize_t w = ::write(fd, p, left);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { const int e = errno; ::close(fd); ::unlink(tmp.c_str()); errno = w ? e : EIO; fail("write " + tmp); }
        p += w; left -= size_t(w);
    }
    if (::fsync(fd)) { const int e = errno; ::close(fd); ::unlink(tmp.c_str()); errno = e; fail("fsync " + tmp); }
    if (::close(fd)) { const int e = errno; ::unlink(tmp.c_str()); errno = e; fail("close " + tmp); }
    if (::rename(tmp.c_str(), path.c_str())) { const int e = errno; ::unlink(tmp.c_str()); errno = e; fail("rename " + tmp); }
    if (sync_dir) fsync_path(parent(path), true);
}
} // namespace fidx

// the index bytes of `ix` for the store `compat` (tokens not included)
inline std::string flash_index_encode(uint64_t compat, const FlashIndex& ix) {
    std::string b;
    b.reserve(fidx::HEADER + ix.size() * fidx::ENTRY + 8);
    fidx::put(b, fidx::MAGIC); fidx::put(b, compat); fidx::put(b, ix.blocks()); fidx::put(b, ix.next_generation());
    fidx::put(b, ix.clock()); fidx::put(b, uint64_t(ix.size()));
    for (const auto& x : ix.entries()) {
        fidx::put(b, x.generation); fidx::put(b, x.used); fidx::put(b, x.base); fidx::put(b, x.blocks);
        fidx::put(b, x.position); fidx::put(b, uint32_t(x.terminal));
        b.append(reinterpret_cast<const char*>(x.ns.data()), x.ns.size());
        fidx::put(b, uint64_t(x.tokens.size()));
        fidx::put(b, x.manifest);
    }
    fidx::put(b, fnv64(b.data(), b.size()));
    return b;
}
// the index in `b` into `out` (entries without tokens). nullptr, or why it is not an index: its checksum differs, it
// is cut short or too long, an entry's token count is not its position + 1, or two entries share a generation.
inline const char* flash_index_decode(const std::string& b, FlashIndexImage& out) {
    out = FlashIndexImage{};
    if (b.size() < fidx::HEADER + 8) return "not an index";
    uint64_t sum = 0;
    std::memcpy(&sum, b.data() + b.size() - 8, 8);
    if (fnv64(b.data(), b.size() - 8) != sum) return "index checksum differs";
    const std::string body = b.substr(0, b.size() - 8);
    size_t at = 0;
    uint64_t magic = 0, n = 0;
    fidx::get(body, at, magic);
    if (magic != fidx::MAGIC) return "not an index";
    fidx::get(body, at, out.compat); fidx::get(body, at, out.blocks); fidx::get(body, at, out.next);
    fidx::get(body, at, out.clock); fidx::get(body, at, n);
    if (n > (body.size() - at) / fidx::ENTRY || body.size() - at != n * fidx::ENTRY) return "index length differs";
    std::set<uint64_t> gens;
    for (uint64_t i = 0; i < n; ++i) {
        FlashEntry x;
        uint32_t terminal = 0;
        uint64_t ntok = 0;
        fidx::get(body, at, x.generation); fidx::get(body, at, x.used); fidx::get(body, at, x.base); fidx::get(body, at, x.blocks);
        fidx::get(body, at, x.position); fidx::get(body, at, terminal);
        std::memcpy(x.ns.data(), body.data() + at, x.ns.size()); at += x.ns.size();
        fidx::get(body, at, ntok); fidx::get(body, at, x.manifest);
        if (ntok != uint64_t(x.position) + 1 || x.manifest.tokens != ntok) return "index entry token count differs";
        if (!gens.insert(x.generation).second) return "two index entries share a generation";
        x.terminal = terminal != 0;
        out.entries.push_back(std::move(x));
    }
    return nullptr;
}

// The index file and the token files of one store (persist=on). Not thread safe: FlashCheckpoints calls it under the
// lock that guards its index.
class FlashIndexFiles {
    std::string path_, dir_;      // the index; the token folder
    std::set<uint64_t> written_;  // generations whose token file is on disk
    void clean(const std::set<uint64_t>& keep) {   // token files of no kept entry (best effort: the next load retries)
        std::error_code ec;
        for (std::filesystem::directory_iterator it(dir_, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            char* e = nullptr;
            const uint64_t g = std::strtoull(name.c_str(), &e, 10);
            if (name.empty() || *e || !keep.count(g)) std::filesystem::remove(it->path(), ec);
        }
    }
public:
    FlashIndexFiles() = default;
    explicit FlashIndexFiles(const std::string& path)
        : path_(path), dir_(std::filesystem::path(path).replace_extension(".tokens").string()) {}
    bool on() const { return !path_.empty(); }
    const std::string& path() const { return path_; }
    std::string tokens_path(uint64_t generation) const { return dir_ + "/" + std::to_string(generation); }
    // the files for `ix`: each new entry's tokens, then the index, then the token files of entries it no longer holds.
    // Throws std::system_error (the caller turns persistence off).
    void save(uint64_t compat, const FlashIndex& ix) {
        if (!std::filesystem::exists(dir_)) {
            std::filesystem::create_directories(dir_);
            fidx::fsync_path(fidx::parent(dir_), true);
        }
        bool wrote = false;
        for (const auto& x : ix.entries())
            if (!written_.count(x.generation)) {
                fidx::write_durable(tokens_path(x.generation), x.tokens.data(), x.tokens.size() * sizeof(int), false);
                written_.insert(x.generation);
                wrote = true;
            }
        if (wrote) fidx::fsync_path(dir_, true);
        const std::string b = flash_index_encode(compat, ix);
        fidx::write_durable(path_, b.data(), b.size());
        std::set<uint64_t> keep;
        for (const auto& x : ix.entries()) keep.insert(x.generation);
        for (auto it = written_.begin(); it != written_.end();) {
            if (keep.count(*it)) { ++it; continue; }
            std::error_code ec;
            std::filesystem::remove(tokens_path(*it), ec);
            it = written_.erase(it);
        }
    }
    // the index on disk, if it is the store's (compat and blocks), with each entry's tokens. nullptr, or why it starts
    // empty. stale: entries left out because their token file is missing or not theirs.
    const char* load(uint64_t compat, uint64_t blocks, FlashIndexImage& out, uint64_t& stale) {
        out = FlashIndexImage{};
        stale = 0;
        written_.clear();
        std::ifstream f(path_, std::ios::binary);
        if (!f) { clean({}); return "no index"; }
        const std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        FlashIndexImage img;
        const char* why = flash_index_decode(b, img);
        if (!why && img.compat != compat) why = "another engine or layout";
        if (!why && img.blocks != blocks) why = "another store size";
        if (why) { clean({}); return why; }
        out.compat = img.compat; out.blocks = img.blocks; out.next = img.next; out.clock = img.clock;
        for (auto& x : img.entries) {
            std::ifstream t(tokens_path(x.generation), std::ios::binary);
            const std::string tb = t ? std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>()) : std::string();
            if (!t || tb.size() != x.manifest.tokens * sizeof(int) || fnv64(tb.data(), tb.size()) != x.manifest.token_hash) { ++stale; continue; }
            x.tokens.resize(x.manifest.tokens);
            std::memcpy(x.tokens.data(), tb.data(), tb.size());
            written_.insert(x.generation);
            out.entries.push_back(std::move(x));
        }
        clean(written_);
        return nullptr;
    }
    // the index goes (a restore found an entry that is not its own): the next boot starts with no index, and the
    // bootstrap's agreement drops on every rank what this rank no longer has
    void remove() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        std::filesystem::remove(path_ + ".tmp", ec);
        try { fidx::fsync_path(fidx::parent(path_), true); } catch (const std::exception&) {}
    }
};
} // namespace g6s2
