// The control link's frames, the event gate and the bootstrap's word diff against transcripts recorded from the C++
// that engine/scheduler-rs's control code replaced (git 5381433c). Each section runs a seeded stream through the C++
// API (encode, decode_*, Encoder, Decoder, match, EventGate, Agreement, agreement_difference) and writes every frame's
// bytes, every decoded field and every refusal to a transcript, whose hash must equal the recorded one. A changed rule
// changes a hash: rerecord only for a change meant to alter the wire or a refusal, and say so in the commit.
//   --dump SECTION   print that section's transcript (to diff two builds line by line)
//   --write FILE     write every frame the frames section encodes, length-prefixed
//   --read FILE      decode the frames in FILE as the frames section does and print the transcript's hash; a build's
//                    own file and another build's file must give the same hash
// The generator is mt19937_64 with modulo ranges, and no call takes two draws as arguments. Host only.
#include "../bootstrap.h"
#include "../control_protocol.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace g6s2;
namespace {
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint64_t U(uint64_t lo, uint64_t hi) { return hi - lo == UINT64_MAX ? g() : lo + g() % (hi - lo + 1); }
    bool coin(unsigned pct) { return U(0, 99) < pct; }
    uint64_t raw() { return g(); }
};
struct Tx {
    uint64_t h = 14695981039346656037ull;
    std::vector<std::string>* lines = nullptr;
    void operator()(const std::string& s) {
        for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
        h = (h ^ '\n') * 1099511628211ull;
        if (lines) lines->push_back(s);
    }
};
template <class F> std::string outcome(F f) {
    try { return "ok " + f(); }
    catch (const std::runtime_error& e) { return std::string("runtime_error ") + e.what(); }
    catch (const std::exception& e) { return std::string("exception ") + e.what(); }
}
std::string u(uint64_t x) { return std::to_string(x); }
std::string bytes(const std::vector<uint8_t>& b) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
    return "[" + u(b.size()) + "#" + u(h) + "]";
}
std::string digest(const std::array<uint8_t, 32>& d) { std::string s; for (auto c : d) s += u(c) + "."; return s; }
std::string str(const Notice& n) {
    return u(n.epoch) + " " + u(n.event) + " " + u(n.boundary) + " " + u(n.request) + " " + u(n.payload_hash) + " k" + u(uint32_t(n.kind));
}
std::string str(const Key& k) { return u(k.epoch) + "." + u(k.slot) + "." + u(k.generation); }
std::string str(const Run& r) { return u(r.first_page) + "+" + u(r.pages) + "v" + u(r.table_version) + "#" + u(r.list_hash); }
std::string str(const g6rt::steering::Identity& s) {
    return u(s.enabled) + u(s.images) + " " + digest(s.image_digest) + " " + digest(s.profile_digest) + " s" + u(uint8_t(s.scope)) +
           " " + digest(s.schedule_digest) + " " + u(s.scale_bits) + " " + u(s.slot);
}
std::string str(const Ack& a) { return str(a.notice) + " r" + u(a.rank) + " h" + u(a.state_hash) + " room" + u(a.room); }
std::string str(const Commit& c) {
    std::string s = u(c.epoch) + " " + u(c.event) + " " + u(c.boundary) + " bh" + u(c.before_hash) + " hold" + u(c.hold);
    if (c.cancel) s += " cancel " + str(*c.cancel);
    if (c.admit) {
        const auto& g = *c.admit;
        s += " admit " + str(g.key) + " " + u(g.req.id) + " " + u(g.req.prompt_tokens) + " " + u(g.req.max_tokens) + " " +
             str(g.req.steering) + " " + str(g.run) + " " + u(g.reuse) + " " + u(g.restored) + " " + u(g.counted) + " " + u(g.req.r_rows);
    }
    if (c.resize) s += " resize " + u(c.resize->tier) + " " + u(c.resize->segs) + " " + u(c.resize->slots) + " " + u(c.resize->txn);
    if (c.shrink) s += " shrink " + str(c.shrink->key) + " " + u(c.shrink->r_rows) + " " + str(c.shrink->run);
    return s;
}
std::string str(const EventCommit& e) { return str(e.notice) + " | " + str(e.commit); }

std::array<uint8_t, 32> rdigest(Rng& r) {
    std::array<uint8_t, 32> d{};
    if (r.coin(20)) return d;
    for (auto& c : d) c = uint8_t(r.raw());
    return d;
}
uint64_t pick(Rng& r, std::initializer_list<uint64_t> v) { return *(v.begin() + r.U(0, v.size() - 1)); }
// valid most of the time; else a scope, scale or slot the decoder refuses
g6rt::steering::Identity identity(Rng& r) {
    g6rt::steering::Identity s;
    s.enabled = r.coin(50);
    s.images = r.coin(30);
    s.image_digest = rdigest(r);
    s.profile_digest = rdigest(r);
    s.scope = g6rt::steering::Scope(r.coin(90) ? r.U(1, 2) : pick(r, {0, 3, 255}));
    s.schedule_digest = rdigest(r);
    s.scale_bits = uint32_t(r.coin(85) ? pick(r, {0x3f800000, 0x3f8ccccd, 0x40800000, 0x00000001, 0x3e800000})
                                        : pick(r, {0, 0x80000000, 0x7fc00000, 0x7f800000, 0xbf800000, 0x40800001}));
    s.slot = uint32_t(r.coin(90) ? r.U(0, 7) : pick(r, {8, 9, 0xffffffff}));
    return s;
}
uint64_t wide(Rng& r) { return r.coin(70) ? r.U(0, 1000) : r.raw(); }
uint32_t wide32(Rng& r) { return uint32_t(r.coin(70) ? r.U(0, 1000) : r.raw()); }
Notice notice(Rng& r) {
    Notice n;
    n.epoch = wide(r); n.event = wide(r); n.boundary = wide(r); n.request = wide(r); n.payload_hash = r.raw();
    n.kind = EventKind(r.coin(95) ? r.U(1, 3) : pick(r, {0, 4, 0xffffffff}));
    return n;
}
Key key(Rng& r) { Key k; k.epoch = wide(r); k.slot = wide32(r); k.generation = wide32(r); return k; }
Run run(Rng& r) { Run x; x.first_page = wide32(r); x.pages = wide32(r); x.table_version = wide32(r); x.list_hash = r.raw(); return x; }
uint32_t rows(Rng& r) { return r.coin(90) ? uint32_t(PAGE * r.U(0, 4096)) : uint32_t(r.U(1, 63) + PAGE * r.U(0, 100)); }
// a commit for n: consistent with it (match) most of the time
EventCommit commit(Rng& r) {
    EventCommit e;
    e.notice = notice(r);
    if (r.coin(85)) e.notice.kind = EventKind(r.U(1, 3));
    auto& c = e.commit;
    c.epoch = r.coin(95) ? e.notice.epoch : wide(r);
    c.event = r.coin(95) ? e.notice.event : wide(r);
    c.boundary = r.coin(95) ? e.notice.boundary : wide(r);
    c.before_hash = r.raw();
    const bool admit = e.notice.kind == EventKind::Admit, resize = e.notice.kind == EventKind::Resize;
    if (r.coin(resize ? 5 : 40)) c.cancel = key(r);
    if (r.coin(admit ? 80 : 5)) {
        Grant g; g.key = key(r);
        g.req.id = r.coin(95) ? e.notice.request : wide(r);
        g.req.prompt_tokens = wide32(r); g.req.max_tokens = wide32(r); g.req.steering = identity(r);
        g.run = run(r);
        g.reuse = r.coin(50) ? 0 : wide32(r);
        g.restored = r.coin(60) ? 0 : (r.coin(90) && g.reuse ? uint32_t(r.U(1, 3)) : wide32(r));
        g.counted = r.coin(50) ? 0 : wide32(r);
        g.req.r_rows = rows(r);
        c.admit = g;
    }
    c.hold = c.admit ? r.coin(30) : r.coin(3);
    if (resize ? r.coin(95) : r.coin(3)) {
        Resize z; z.tier = uint32_t(r.coin(70) ? r.U(0, 3) : pick(r, {0xFFFFFFFF, 0xFFFFFFFE})); z.segs = wide(r); z.slots = wide(r); z.txn = r.raw();
        c.resize = z;
    }
    if (r.coin(admit ? 25 : 3)) {
        Shrink k; k.key = key(r); k.r_rows = r.coin(5) ? 0 : rows(r); k.run = run(r);
        c.shrink = k;
    }
    return e;
}
std::string frame_name(const std::vector<uint8_t>& b) {
    return outcome([&] { Decoder d(b); return u(uint64_t(d.tag)); });
}
// every decoder on b, each one's result or refusal
void decode_all(const std::vector<uint8_t>& b, Tx& tx) {
    tx("tag " + frame_name(b));
    tx(" notice " + outcome([&] { return str(decode_notice(b)); }));
    tx(" ack " + outcome([&] { return str(decode_ack(b)); }));
    tx(" commit " + outcome([&] { return str(decode_commit(b)); }));
    tx(" request " + outcome([&] { auto x = decode_request(b); return u(x.epoch) + " " + u(x.request) + " " + bytes({x.bytes.begin(), x.bytes.end()}); }));
    tx(" quit " + outcome([&] { return u(decode_quit(b)); }));
    tx(" sync " + outcome([&] { auto s = decode_sync(b); return u(s.epoch) + " " + u(s.rank) + " " + u(s.seq); }));
    tx(" go " + outcome([&] { auto g = decode_go(b); return u(g.first) + " " + u(g.second); }));
}
// the frames section's frames, in order: each kind once per step
std::vector<std::vector<uint8_t>> frames(uint64_t seed, Tx& tx) {
    Rng r(seed);
    std::vector<std::vector<uint8_t>> out;
    auto keep = [&](const std::string& what, const std::function<std::vector<uint8_t>()>& f) {
        std::vector<uint8_t> b;
        tx(what + " " + outcome([&] { b = f(); return bytes(b); }));
        if (!b.empty()) out.push_back(b);
    };
    for (int step = 0; step < 400; ++step) {
        const Notice n = notice(r);
        keep("notice " + str(n), [&] { return encode(n); });
        Ack a; a.notice = notice(r); a.rank = unsigned(r.U(0, 7)); a.state_hash = r.raw(); a.room = r.coin(50) ? UINT32_MAX : wide32(r);
        keep("ack " + str(a), [&] { return encode(a); });
        const EventCommit e = commit(r);
        keep("commit " + str(e), [&] { return encode(e); });
        RequestBytes q; q.epoch = wide(r); q.request = wide(r);
        const uint64_t len = r.coin(10) ? 0 : r.U(1, 300);
        for (uint64_t i = 0; i < len; ++i) q.bytes.push_back(char(r.raw()));
        keep("request " + u(q.epoch) + " " + u(q.request) + " " + bytes({q.bytes.begin(), q.bytes.end()}), [&] { return encode(q); });
        const uint64_t qe = wide(r);
        keep("quit " + u(qe), [&] { return encode_quit(qe); });
        SyncFrame s; s.epoch = wide(r); s.rank = unsigned(r.U(0, 7)); s.seq = wide(r);
        keep("sync " + u(s.epoch) + " " + u(s.rank) + " " + u(s.seq), [&] { return encode(s); });
        const uint64_t ge = wide(r), gs = wide(r);
        keep("go " + u(ge) + " " + u(gs), [&] { return encode_go(ge, gs); });
    }
    return out;
}
// how each frame decodes, and how its damaged copies are refused
void decode_frames(const std::vector<std::vector<uint8_t>>& all, uint64_t seed, Tx& tx) {
    Rng r(seed);
    for (const auto& b : all) {
        decode_all(b, tx);
        auto cut = b; cut.resize(size_t(r.U(0, b.size() - 1)));
        tx("cut " + u(cut.size()));
        decode_all(cut, tx);
        auto flip = b; const size_t at = size_t(r.U(0, b.size() - 1)); flip[at] ^= uint8_t(r.U(1, 255));
        tx("flip " + u(at));
        decode_all(flip, tx);
        auto more = b; more.push_back(uint8_t(r.raw()));
        tx("trailing");
        decode_all(more, tx);
    }
}
uint64_t frames_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    const auto all = frames(seed, tx);
    decode_frames(all, seed + 1000, tx);
    return tx.h;
}
// Encoder and Decoder piece by piece: steering identities, keys, notices and raw words, and their refusals
uint64_t pieces_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int step = 0; step < 2000; ++step) {
        const auto tag = Tag(r.U(1, 7));
        Encoder e(tag);
        const unsigned parts = unsigned(r.U(1, 4));
        std::vector<unsigned> kinds;
        for (unsigned p = 0; p < parts; ++p) {
            const unsigned k = unsigned(r.U(0, 3));
            kinds.push_back(k);
            if (k == 0) { const uint64_t x = r.coin(50) ? r.raw() : r.U(0, 0x1ffffffffull); e.u64(x); tx("u64 " + u(x)); }
            else if (k == 1) { const auto s = identity(r); e.steering(s); tx("steering " + str(s)); }
            else if (k == 2) { const auto kk = key(r); e.key(kk); tx("key " + str(kk)); }
            else { const auto n = notice(r); e.notice(n); tx("notice " + str(n)); }
        }
        tx("bytes " + bytes(e.b));
        if (r.coin(20)) e.b.resize(size_t(r.U(0, e.b.size())));
        if (r.coin(10)) e.b.push_back(uint8_t(r.raw()));
        tx(outcome([&] {
            Decoder d(e.b);
            std::string s = "tag " + u(uint64_t(d.tag));
            for (unsigned k : kinds) {
                if (k == 0) s += " " + (r.coin(50) ? u(d.u64()) : u(d.u32()));
                else if (k == 1) s += " " + str(d.steering());
                else if (k == 2) s += " " + str(d.key());
                else s += " " + str(d.notice());
            }
            d.end();
            return s + " i" + u(d.i);
        }));
    }
    return tx.h;
}
// match() on commits that agree with their notice, and on ones that do not
uint64_t match_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int step = 0; step < 3000; ++step) {
        const auto e = commit(r);
        tx(str(e) + " -> " + outcome([&] { match(e); return std::string(); }));
    }
    return tx.h;
}
// the event gate single-threaded: announce, before_work, arrive, commit, applied, fail, wait_commit only where it
// cannot block
uint64_t gate_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int life = 0; life < 300; ++life) {
        const uint64_t epoch = r.coin(97) ? r.U(1, 5) : 0;
        const unsigned rank = unsigned(r.U(0, 5));
        std::unique_ptr<EventGate> g;
        tx("gate " + u(epoch) + " " + u(rank) + " " + outcome([&] { g = std::make_unique<EventGate>(epoch, rank); return std::string(); }));
        if (!g) continue;
        uint64_t b = 0, ev = 1;
        std::optional<Notice> cur;
        bool arrived = false, committed = false, failed = false;
        for (int op = 0; op < 60; ++op) {
            const unsigned k = unsigned(r.U(0, 99));
            if (k < 30) {
                b += r.U(0, 2);
                const uint64_t at = r.coin(10) ? r.U(0, b + 2) : b;
                const auto o = outcome([&] { return u(g->before_work(at)); });
                tx("before_work " + u(at) + " " + o);
                if (o.find("raced") != std::string::npos) failed = true;
            } else if (k < 50) {
                Notice n{r.coin(95) ? epoch : r.U(0, 6), r.coin(90) ? ev : r.U(0, ev + 2), b + r.U(0, 4), r.U(0, 9), r.raw(),
                         EventKind(r.coin(95) ? r.U(1, 3) : r.U(0, 5))};
                const auto o = outcome([&] { g->announce(n); return std::string(); });
                tx("announce " + str(n) + " " + o);
                if (o == "ok ") { cur = n; arrived = committed = false; }
                if (o.find("raced") != std::string::npos) failed = true;
            } else if (k < 65) {
                const uint64_t at = cur && r.coin(85) ? cur->boundary : b + r.U(0, 3);
                const uint64_t h = r.U(0, 3);
                const uint32_t room = r.coin(50) ? UINT32_MAX : uint32_t(r.U(0, 9));
                const auto o = outcome([&] { return str(g->arrive(at, h, room)); });
                tx("arrive " + u(at) + " " + u(h) + " " + u(room) + " " + o);
                if (o.rfind("ok ", 0) == 0) arrived = true;
            } else if (k < 78) {
                Notice n = cur && r.coin(85) ? *cur : notice(r);
                const auto o = outcome([&] { g->commit(n); return std::string(); });
                tx("commit " + str(n) + " " + o);
                if (o == "ok ") committed = true;
            } else if (k < 90) {
                Notice n = cur && r.coin(85) ? *cur : notice(r);
                const bool hold = r.coin(30);
                const auto o = outcome([&] { g->applied(n, hold); return std::string(); });
                tx("applied " + str(n) + " " + u(hold) + " " + o);
                if (o == "ok ") { cur.reset(); arrived = committed = false; ++ev; }
            } else if (k < 97) {
                if (failed || !cur || !arrived || committed)
                    tx("wait_commit " + outcome([&] { g->wait_commit(); return std::string(); }));
            } else {
                g->fail(); failed = true; tx("fail");
            }
        }
    }
    return tx.h;
}
// the head's per-event ACK set
uint64_t agreement_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int life = 0; life < 600; ++life) {
        const unsigned nr = unsigned(r.U(0, 7));
        const Notice n = notice(r);
        std::unique_ptr<Agreement> a;
        tx("agreement " + u(nr) + " " + str(n) + " " + outcome([&] { a = std::make_unique<Agreement>(nr, n); return std::string(); }));
        if (!a) continue;
        const uint64_t h = r.U(0, 2);
        for (int op = 0; op < 12; ++op) {
            Ack k; k.notice = r.coin(90) ? n : notice(r); k.rank = unsigned(r.U(0, nr)); k.state_hash = r.coin(90) ? h : r.U(0, 3);
            k.room = r.coin(50) ? UINT32_MAX : uint32_t(r.U(0, 99));
            tx("ack " + str(k) + " " + outcome([&] { a->ack(k); return std::string(); }));
            tx("ready " + u(a->ready()) + " hash " + outcome([&] { return u(a->agreed_hash()); }) + " room " +
               outcome([&] { return u(a->room()); }));
        }
    }
    return tx.h;
}
// Check every ACK subset through the C ABI, including a complete six-rank set. A Rust optimizer regression can
// return 255 for true: branching may work, but the bool representation and promotion used by the transcript do not.
void agreement_ready_test() {
    for (unsigned nr = 1; nr <= 6; ++nr) {
        for (unsigned mask = 0; mask < (1u << nr); ++mask) {
            Notice n;
            Agreement a(nr, n);
            for (unsigned rank = 0; rank < nr; ++rank) {
                if (mask & (1u << rank)) {
                    Ack k; k.notice = n; k.rank = rank; k.state_hash = 7;
                    a.ack(k);
                }
            }
            const bool ready = a.ready();
            unsigned char raw = 0;
            static_assert(sizeof ready == sizeof raw, "control bool width");
            std::memcpy(&raw, &ready, sizeof raw);
            require(raw == unsigned(mask == (1u << nr) - 1), "noncanonical or incorrect agreement readiness");
        }
    }
}
// the bootstrap's first differing word
uint64_t difference_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    auto text = [&] {
        std::string s;
        const uint64_t n = r.U(0, 12);
        for (uint64_t i = 0; i < n; ++i) s += "ab \n\nG6_X=1"[r.U(0, 11)];
        return s;
    };
    for (int step = 0; step < 3000; ++step) {
        const std::string h = text(), w = r.coin(30) ? h : text();
        const unsigned rank = unsigned(r.U(1, 5));
        tx("[" + h + "] [" + w + "] " + agreement_difference(rank, h, w));
    }
    return tx.h;
}
std::vector<std::vector<uint8_t>> read_frames(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::vector<uint8_t>> out;
    uint8_t h[4];
    while (f.read(reinterpret_cast<char*>(h), 4)) {
        const uint32_t n = uint32_t(h[0]) << 24 | uint32_t(h[1]) << 16 | uint32_t(h[2]) << 8 | h[3];
        std::vector<uint8_t> b(n);
        f.read(reinterpret_cast<char*>(b.data()), n);
        out.push_back(b);
    }
    return out;
}
} // namespace

int main(int argc, char** argv) {
    struct Section { const char* name; uint64_t seed; uint64_t (*run)(uint64_t, std::vector<std::string>*); uint64_t want; };
    const Section sections[] = {
        // Control v15 adds Request::ckpt_at; re-recorded frame and piece transcripts.
        {"frames", 1, frames_section, 0x8635e01ab1611e51ull},
        {"pieces", 2, pieces_section, 0xb60cde07eb13e1f2ull},
        {"match", 3, match_section, 0x060f21b6d71b621eull},
        {"gate", 4, gate_section, 0xf485e0fbea73dd2dull},
        {"agreement", 5, agreement_section, 0x2475fad827f09bfbull},
        {"difference", 6, difference_section, 0x324815411f8dcfbcull},
    };
    if (argc == 3 && !std::strcmp(argv[1], "--write")) {
        Tx tx;
        const auto all = frames(1, tx);
        std::ofstream f(argv[2], std::ios::binary);
        for (const auto& b : all) {
            const uint32_t n = uint32_t(b.size());
            const uint8_t h[4] = {uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8), uint8_t(n)};
            f.write(reinterpret_cast<const char*>(h), 4);
            f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
        }
        std::printf("%zu frames\n", all.size());
        return f ? 0 : 1;
    }
    if (argc == 3 && !std::strcmp(argv[1], "--read")) {
        Tx tx;
        decode_frames(read_frames(argv[2]), 1 + 1000, tx);
        const bool ok = tx.h == 0x776b8289ffd0d408ull;   // git 5381433c's frames decoded by git 5381433c
        std::printf("decoded: 0x%016llxull %s\n", (unsigned long long)tx.h, ok ? "ok" : "MISMATCH");
        return ok ? 0 : 1;
    }
    if (argc == 3 && !std::strcmp(argv[1], "--dump")) {
        for (const auto& s : sections)
            if (!std::strcmp(s.name, argv[2])) {
                std::vector<std::string> lines;
                s.run(s.seed, &lines);
                for (const auto& l : lines) std::printf("%s\n", l.c_str());
                return 0;
            }
        return 2;
    }
    agreement_ready_test();
    int fails = 0;
    for (const auto& s : sections) {
        const uint64_t h = s.run(s.seed, nullptr);
        const bool ok = h == s.want;
        if (!ok) ++fails;
        std::printf("%-10s 0x%016llxull %s\n", s.name, (unsigned long long)h, ok ? "ok" : "MISMATCH");
    }
    std::printf(fails ? "control_vectors_test: %d FAIL\n" : "control_vectors_test: PASS\n", fails);
    return fails != 0;
}
