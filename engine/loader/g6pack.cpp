// gr8 pack loader implementation (see g6pack.h). Host-only; no CUDA dependency.
#include "g6pack.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#if defined(__aarch64__) && (defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO))
#include <arm_neon.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>
#define G6_ARM_SHA 1
#endif

namespace g6
{

// ------------------------------------------------------------------------------------------------ SHA-256
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void compress_portable(uint32_t st[8], const uint8_t* p, size_t nblocks)
{
    for (; nblocks--; p += 64)
    {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i)
        {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
        for (int i = 0; i < 64; ++i)
        {
            const uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
    }
}

#ifdef G6_ARM_SHA
static void compress_arm(uint32_t st[8], const uint8_t* p, size_t nblocks)
{
    uint32x4_t s0 = vld1q_u32(&st[0]), s1 = vld1q_u32(&st[4]);
    for (; nblocks--; p += 64)
    {
        const uint32x4_t save0 = s0, save1 = s1;
        uint32x4_t w[4];
        for (int i = 0; i < 4; ++i) w[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 16 * i)));
        for (int i = 0; i < 16; ++i)   // 16 groups of 4 rounds; w[i%4] holds message words 4i..4i+3
        {
            const uint32x4_t wk = vaddq_u32(w[i & 3], vld1q_u32(&K256[4 * i]));
            const uint32x4_t t = s0;
            s0 = vsha256hq_u32(s0, s1, wk);
            s1 = vsha256h2q_u32(s1, t, wk);
            if (i < 12) w[i & 3] = vsha256su1q_u32(vsha256su0q_u32(w[i & 3], w[(i + 1) & 3]), w[(i + 2) & 3], w[(i + 3) & 3]);
        }
        s0 = vaddq_u32(s0, save0);
        s1 = vaddq_u32(s1, save1);
    }
    vst1q_u32(&st[0], s0);
    vst1q_u32(&st[4], s1);
}
static const bool kHwSha = (getauxval(AT_HWCAP) & HWCAP_SHA2) != 0;
#endif

static inline void compress(uint32_t st[8], const uint8_t* p, size_t nblocks)
{
#ifdef G6_ARM_SHA
    if (kHwSha) { compress_arm(st, p, nblocks); return; }
#endif
    compress_portable(st, p, nblocks);
}

Sha256::Sha256()
{
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h, iv, sizeof(h));
}

void Sha256::update(const uint8_t* p, size_t n)
{
    total += n;
    if (fill)
    {
        const size_t k = std::min(n, 64 - fill);
        memcpy(buf + fill, p, k);
        fill += k; p += k; n -= k;
        if (fill < 64) return;
        compress(h, buf, 1);
        fill = 0;
    }
    if (n >= 64) { compress(h, p, n / 64); p += n / 64 * 64; n %= 64; }
    if (n) { memcpy(buf, p, n); fill = n; }
}

void Sha256::final(uint8_t out[32])
{
    const uint64_t bits = total * 8;
    uint8_t pad[72] = {0x80};
    const size_t padlen = (fill < 56 ? 56 - fill : 120 - fill);
    update(pad, padlen);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(len, 8);
    for (int i = 0; i < 8; ++i) { out[4 * i] = h[i] >> 24; out[4 * i + 1] = h[i] >> 16; out[4 * i + 2] = h[i] >> 8; out[4 * i + 3] = h[i]; }
}

std::string hex(const uint8_t* p, size_t n)
{
    static const char* d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; ++i) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}

// ------------------------------------------------------------------------------------------------ JSON (manifest subset)
namespace
{
struct J
{
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false, is_int = true;
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::vector<J> a;
    std::vector<std::pair<std::string, J>> o;
    const J* get(const char* k) const
    {
        for (auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct Parser
{
    const char* p;
    const char* e;
    std::string err;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    bool fail(const char* m) { if (err.empty()) err = m; return false; }
    static void utf8(std::string& s, uint32_t c)
    {
        if (c < 0x80) s += (char)c;
        else if (c < 0x800) { s += (char)(0xC0 | c >> 6); s += (char)(0x80 | (c & 63)); }
        else { s += (char)(0xE0 | c >> 12); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
    }
    bool str(std::string& s)
    {
        if (p >= e || *p != '"') return fail("expected string");
        ++p;
        while (p < e && *p != '"')
        {
            if (*p == '\\')
            {
                if (++p >= e) return fail("bad escape");
                switch (*p)
                {
                case 'n': s += '\n'; break; case 't': s += '\t'; break; case 'r': s += '\r'; break;
                case 'b': s += '\b'; break; case 'f': s += '\f'; break;
                case 'u':
                {
                    if (e - p < 5) return fail("bad \\u");
                    utf8(s, (uint32_t)strtoul(std::string(p + 1, 4).c_str(), nullptr, 16));
                    p += 4;
                    break;
                }
                default: s += *p;
                }
                ++p;
            }
            else s += *p++;
        }
        if (p >= e) return fail("unterminated string");
        ++p;
        return true;
    }
    bool val(J& v)
    {
        ws();
        if (p >= e) return fail("unexpected end");
        if (*p == '{')
        {
            v.t = J::Obj; ++p; ws();
            if (p < e && *p == '}') { ++p; return true; }
            for (;;)
            {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (p >= e || *p != ':') return fail("expected :");
                ++p;
                v.o.emplace_back(std::move(k), J());
                if (!val(v.o.back().second)) return false;
                ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == '}') { ++p; return true; }
                return fail("expected , or }");
            }
        }
        if (*p == '[')
        {
            v.t = J::Arr; ++p; ws();
            if (p < e && *p == ']') { ++p; return true; }
            for (;;)
            {
                v.a.emplace_back();
                if (!val(v.a.back())) return false;
                ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == ']') { ++p; return true; }
                return fail("expected , or ]");
            }
        }
        if (*p == '"') { v.t = J::Str; return str(v.s); }
        if (!strncmp(p, "true", 4)) { v.t = J::Bool; v.b = true; p += 4; return true; }
        if (!strncmp(p, "false", 5)) { v.t = J::Bool; p += 5; return true; }
        if (!strncmp(p, "null", 4)) { p += 4; return true; }
        const char* q = p;
        if (*q == '-') ++q;
        while (q < e && ((*q >= '0' && *q <= '9') || *q == '.' || *q == 'e' || *q == 'E' || *q == '+' || *q == '-')) ++q;
        if (q == p) return fail("bad value");
        std::string num(p, q);
        v.t = J::Num;
        v.is_int = num.find_first_of(".eE") == std::string::npos;
        if (v.is_int) v.i = strtoll(num.c_str(), nullptr, 10); else v.d = strtod(num.c_str(), nullptr);
        p = q;
        return true;
    }
};

bool dtype_of(const std::string& s, DType& d)
{
    static const std::pair<const char*, DType> m[] = {{"u8", DType::U8}, {"i16", DType::I16}, {"f16", DType::F16},
        {"bf16", DType::BF16}, {"f32", DType::F32}, {"i32", DType::I32}, {"i64", DType::I64}};
    for (auto& kv : m) if (s == kv.first) { d = kv.second; return true; }
    return false;
}

bool unhex(const std::string& s, uint8_t out[32])
{
    if (s.size() != 64) return false;
    for (int i = 0; i < 32; ++i)
    {
        auto nib = [](char c) -> int { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
        const int a = nib(s[2 * i]), b = nib(s[2 * i + 1]);
        if (a < 0 || b < 0) return false;
        out[i] = (uint8_t)(a << 4 | b);
    }
    return true;
}
}  // namespace

size_t dtype_size(DType d)
{
    switch (d)
    {
    case DType::U8: return 1;
    case DType::I16: case DType::F16: case DType::BF16: return 2;
    case DType::F32: case DType::I32: return 4;
    case DType::I64: return 8;
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ Pack
Pack::~Pack()
{
    if (base_) free(base_);
}

const Entry* Pack::find(const std::string& name) const
{
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &entries_[it->second];
}

bool Pack::open_manifest(const std::string& path, std::string& err)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open manifest " + path; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    Parser ps{text.data(), text.data() + text.size(), ""};
    J root;
    if (!ps.val(root) || root.t != J::Obj) { err = "manifest JSON: " + ps.err; return false; }
    const J* fmt = root.get("format");
    const J* rk = root.get("rank");
    const J* al = root.get("align");
    const J* pb = root.get("pack_bytes");
    const J* ts = root.get("tensors");
    if (!fmt || fmt->s != "g6pack-v1") { err = "manifest format is not g6pack-v1"; return false; }
    if (!rk || !al || !pb || !ts || ts->t != J::Arr) { err = "manifest missing rank/align/pack_bytes/tensors"; return false; }
    rank_ = (int)rk->i;
    align_ = (uint64_t)al->i;
    pack_bytes_ = (uint64_t)pb->i;
    if (align_ == 0 || pack_bytes_ % align_) { err = "pack_bytes not a multiple of align"; return false; }
    entries_.clear();
    index_.clear();
    for (const J& t : ts->a)
    {
        Entry e;
        const J *n = t.get("name"), *dt = t.get("dtype"), *sh = t.get("shape"), *of = t.get("offset"), *nb = t.get("nbytes"),
                *h = t.get("sha256"), *lay = t.get("layout"), *xf = t.get("expert_format");
        if (!n || !dt || !sh || !of || !nb || !h) { err = "tensor entry missing a field"; return false; }
        e.name = n->s;
        if (lay) e.layout = lay->s;
        if (xf)
        {
            if (xf->s == "tr3") e.expert_format = 0;
            else if (xf->s == "exlr8-e4m3") e.expert_format = 1;
            else { err = e.name + ": unknown expert_format " + xf->s; return false; }
        }
        if (!dtype_of(dt->s, e.dtype)) { err = e.name + ": unknown dtype " + dt->s; return false; }
        uint64_t count = 1, bytes = 0;
        for (const J& d : sh->a)
        {
            e.shape.push_back(d.i);
            if (d.i < 0 || __builtin_mul_overflow(count, (uint64_t)d.i, &count))
            { err = e.name + ": bad shape"; return false; }
        }
        if (of->i < 0 || nb->i < 0) { err = e.name + ": negative offset or nbytes"; return false; }
        e.offset = (uint64_t)of->i;
        e.nbytes = (uint64_t)nb->i;
        if (__builtin_mul_overflow(count, (uint64_t)dtype_size(e.dtype), &bytes) || bytes != e.nbytes)
        { err = e.name + ": nbytes != prod(shape) * dtype size"; return false; }
        if (e.offset % align_) { err = e.name + ": offset not aligned"; return false; }
        if (!unhex(h->s, e.sha256)) { err = e.name + ": bad sha256"; return false; }
        entries_.push_back(std::move(e));
    }
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.offset < b.offset; });
    uint64_t end = 0;
    for (size_t i = 0; i < entries_.size(); ++i)
    {
        const Entry& e = entries_[i];
        if (e.offset < end) { err = e.name + ": overlaps previous tensor"; return false; }
        if (__builtin_add_overflow(e.offset, e.nbytes, &end))
        { err = e.name + ": offset + nbytes overflows"; return false; }
        if (!index_.emplace(e.name, i).second) { err = e.name + ": duplicate name"; return false; }
    }
    if (end > pack_bytes_) { err = "tensors run past pack_bytes"; return false; }
    return true;
}

bool Pack::load(const std::string& path, const LoadOpts& opt, LoadStats& st, std::string& err)
{
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (rank_ < 0) { err = "open_manifest first"; return false; }
    const bool is_stdin = path == "-";
    int fd = 0;
    st.used_direct_io = false;
    if (!is_stdin)
    {
        fd = opt.direct_io ? open(path.c_str(), O_RDONLY | O_DIRECT) : -1;
        if (fd >= 0) st.used_direct_io = true;
        else fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open " + path + ": " + strerror(errno); return false; }
        struct stat sb;
        if (fstat(fd, &sb) == 0 && S_ISREG(sb.st_mode) && (uint64_t)sb.st_size != pack_bytes_)
        {
            close(fd);
            err = "pack size " + std::to_string(sb.st_size) + " != manifest pack_bytes " + std::to_string(pack_bytes_);
            return false;
        }
        if (!st.used_direct_io) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
    if (is_stdin) fcntl(fd, F_SETPIPE_SZ, 1 << 20);   // fewer, larger pipe reads when streamed over the fabric
    constexpr size_t HUGE2M = 2ull << 20;
    if (base_) { free(base_); base_ = nullptr; }
    alloc_bytes_ = (pack_bytes_ + HUGE2M - 1) / HUGE2M * HUGE2M;
    void* mem = nullptr;
    if (posix_memalign(&mem, HUGE2M, std::max<size_t>(alloc_bytes_, HUGE2M)))
    {
        if (!is_stdin) close(fd);
        err = "allocation of " + std::to_string(alloc_bytes_) + " bytes failed";
        return false;
    }
    base_ = (uint8_t*)mem;
    madvise(base_, alloc_bytes_, MADV_HUGEPAGE);

    std::atomic<uint64_t> done{0};
    std::atomic<bool> abort{false};
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<size_t> next{0};
    std::vector<uint8_t> ok(entries_.size(), 0);

    auto hasher = [&]() {
        for (;;)
        {
            const size_t i = next.fetch_add(1);
            if (i >= entries_.size()) return;
            const Entry& e = entries_[i];
            Sha256 h;
            uint64_t pos = e.offset;
            const uint64_t end = e.offset + e.nbytes;
            while (pos < end)
            {
                uint64_t avail = done.load(std::memory_order_acquire);
                if (avail <= pos)
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return done.load(std::memory_order_acquire) > pos || abort.load(); });
                    if (abort.load()) return;
                    continue;
                }
                const uint64_t to = std::min(avail, end);
                h.update(base_ + pos, to - pos);
                pos = to;
            }
            uint8_t d[32];
            h.final(d);
            ok[i] = memcmp(d, e.sha256, 32) == 0;
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < opt.hash_threads; ++i) pool.emplace_back(hasher);

    uint64_t pos = 0;
    bool read_ok = true;
    const size_t chunk = std::max<size_t>(opt.chunk / 4096 * 4096, 4096);
    uint64_t since_drop = 0, since_note = 0;
    while (pos < pack_bytes_)
    {
        const size_t want = (size_t)std::min<uint64_t>(chunk, pack_bytes_ - pos);
        const ssize_t n = read(fd, base_ + pos, want);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {   // a pipe someone else set O_NONBLOCK on (e.g. an ssh sharing the description): wait for data
            pollfd p{fd, POLLIN, 0};
            poll(&p, 1, -1);
            continue;
        }
        if (n < 0 && st.used_direct_io && errno == EINVAL && pos % 4096 == 0)
        {   // filesystem refused O_DIRECT: continue buffered from here
            const int fd2 = open(path.c_str(), O_RDONLY);
            if (fd2 < 0 || lseek(fd2, (off_t)pos, SEEK_SET) != (off_t)pos) { read_ok = false; err = "O_DIRECT fallback failed"; break; }
            close(fd);
            fd = fd2;
            st.used_direct_io = false;
            continue;
        }
        if (n <= 0)
        {
            read_ok = false;
            err = n == 0 ? "stream ended at byte " + std::to_string(pos) + " of " + std::to_string(pack_bytes_)
                         : std::string("read error: ") + strerror(errno);
            break;
        }
        pos += (uint64_t)n;
        since_drop += (uint64_t)n;
        since_note += (uint64_t)n;
        if (!is_stdin && !st.used_direct_io && since_drop >= (1ull << 30))
        {
            posix_fadvise(fd, 0, (off_t)pos, POSIX_FADV_DONTNEED);
            since_drop = 0;
        }
        if (since_note >= (4ull << 20) || pos == pack_bytes_)
        {
            {
                std::lock_guard<std::mutex> lk(mu);
                done.store(pos, std::memory_order_release);
            }
            cv.notify_all();
            since_note = 0;
        }
    }
    const double read_s = std::chrono::duration<double>(clk::now() - t0).count();
    if (read_ok && is_stdin)
    {   // the stream must end exactly at pack_bytes
        char extra;
        const ssize_t n = read(fd, &extra, 1);
        if (n != 0) { read_ok = false; err = "stream has bytes past pack_bytes"; }
    }
    if (!is_stdin)
    {
        if (!st.used_direct_io) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
    if (!read_ok)
    {
        {
            std::lock_guard<std::mutex> lk(mu);
            abort.store(true);
        }
        cv.notify_all();
    }
    for (auto& t : pool) t.join();
    if (!read_ok) return false;

    std::string bad;
    size_t nbad = 0;
    if (opt.hash_threads > 0)
        for (size_t i = 0; i < entries_.size(); ++i)
            if (!ok[i]) { if (nbad++ < 8) bad += (bad.empty() ? "" : ", ") + entries_[i].name; }
    // padding (and tail) must be zero
    uint64_t end = 0;
    size_t nonzero_pad = 0;
    auto zero = [&](uint64_t a, uint64_t b) { for (uint64_t i = a; i < b; ++i) if (base_[i]) return false; return true; };
    for (const Entry& e : entries_) { if (!zero(end, e.offset)) ++nonzero_pad; end = e.offset + e.nbytes; }
    if (!zero(end, pack_bytes_)) ++nonzero_pad;

    st.bytes = pack_bytes_;
    st.read_seconds = read_s;
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    st.gb_per_s = pack_bytes_ / 1e9 / st.seconds;
    if (nbad || nonzero_pad)
    {
        err = "checksum mismatch in " + std::to_string(nbad) + " tensor(s)" + (bad.empty() ? "" : " [" + bad + "]") +
              (nonzero_pad ? "; " + std::to_string(nonzero_pad) + " non-zero padding gap(s)" : "");
        return false;
    }
    return true;
}

}  // namespace g6
