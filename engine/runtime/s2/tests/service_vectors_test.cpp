// The S2 file protocol and the service's status and trace output against transcripts recorded from the C++ that
// engine/scheduler-rs's service code replaced (git 05425dab). Each section runs a seeded stream through the C++ API
// (number, parse_command, Ingress, capability, BootstrapConfig::agreement, atomic_file, publish_ready, Trace and the
// service's status.s2) and writes every result, file and refusal to a transcript, whose hash must equal the recorded
// one. A changed rule changes a hash: rerecord only for a change meant to alter the files or a refusal, and say so in
// the commit.
//   --dump SECTION   print that section's transcript (to diff two builds line by line)
// The generator is mt19937_64 with modulo ranges. No call or + takes two draws, or a write and its read, as
// operands: their order is unspecified, and GCC on x86_64 takes them right to left. Host only.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../service.h"

using namespace g6s2;
namespace {
// status.s2 is written by a private Service member: an explicit instantiation may name private members
template <class Tag, typename Tag::type M> struct Rob { friend typename Tag::type get(Tag) { return M; } };
struct QueueOf { using type = AdmissionQueue Service::*; friend type get(QueueOf); };
struct ControlOf { using type = ControlHandle Service::*; friend type get(ControlOf); };
struct StatusOf { using type = void (Service::*)(const PoolStatus&); friend type get(StatusOf); };
template struct Rob<QueueOf, &Service::queue_>;
template struct Rob<ControlOf, &Service::ctl_>;
template struct Rob<StatusOf, &Service::write_status>;
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
uint64_t pick(Rng& r, std::initializer_list<uint64_t> v) { return *(v.begin() + r.U(0, v.size() - 1)); }
std::string text(Rng& r, const char* alphabet, uint64_t max) {
    std::string s;
    const uint64_t n = r.U(0, max), k = std::strlen(alphabet);
    for (uint64_t i = 0; i < n; ++i) s += alphabet[r.U(0, k - 1)];
    return s;
}
std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return "<none>";
    std::ostringstream o; o << f.rdbuf(); return o.str();
}
std::string fresh_dir() {
    char t[] = "/tmp/service_vectors_XXXXXX";
    require(::mkdtemp(t) != nullptr, "mkdtemp");
    return t;
}
void clear_dir(const std::string& d) {
    // the names the sections write
    for (const char* f : {"abort.s2", "ready.s2", "status.s2", "status.s2.tmp", "quit", "x", "x.tmp", "t.r0.json",
                          "t.r1.json", "t.r5.json"})
        ::unlink((d + "/" + f).c_str());
    for (int i = 0; i < 64; ++i) { ::unlink((d + "/cmd." + u(i)).c_str()); ::unlink((d + "/req." + u(i)).c_str()); }
}
std::string str(const Command& c) {
    return "admit" + u(c.admit) + " " + u(c.epoch) + " " + u(c.request) + " " + u(c.hash) + " resize" + u(bool(c.resize));
}

// number(): S2 decimal fields
uint64_t number_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (const char* s : {"", "0", "00", "18446744073709551615", "18446744073709551616", "99999999999999999999", "-1", "+1", " 1"})
        tx(std::string("[") + s + "] " + outcome([&] { return u(number(s)); }));
    for (int step = 0; step < 4000; ++step) {
        const std::string s = r.coin(70) ? text(r, "0123456789", 21) : text(r, "0123456789 -a+\t", 8);
        tx("[" + s + "] " + outcome([&] { return u(number(s)); }));
    }
    return tx.h;
}
// a command line, valid most of the time
std::string command_text(Rng& r, uint64_t epoch) {
    if (r.coin(5)) return text(r, "ADMITCNL 0123\n\t", 30);
    std::string kind = r.coin(85) ? (r.coin(60) ? "ADMIT" : "CANCEL")
                                  : std::string(pick(r, {0, 1, 2, 3}) == 0 ? "admit" : r.coin(50) ? "RESIZE" : "");
    auto num = [&] {
        if (r.coin(90)) return u(r.coin(70) ? r.U(0, 1000) : r.raw());
        return std::string(r.coin(50) ? "x" : r.coin(50) ? "-3" : "18446744073709551616");
    };
    const std::string e = r.coin(85) ? u(epoch) : num();
    const std::string req = num();
    const std::string hash = kind == "CANCEL" && r.coin(80) ? "0" : num();
    const char* sep[] = {" ", "  ", "\t", "\n"};
    const char* s1 = sep[r.U(0, 3)];   // one draw a statement: operands of + have no order
    const char* s2 = sep[r.U(0, 3)];
    std::string s = kind + s1 + e + s2 + req;
    if (r.coin(97)) s += sep[r.U(0, 3)] + hash;
    if (r.coin(5)) s += " " + num();
    if (r.coin(80)) s += "\n";
    return s;
}
uint64_t command_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int step = 0; step < 4000; ++step) {
        const uint64_t epoch = r.coin(80) ? r.U(1, 9) : r.raw();
        const std::string t = command_text(r, epoch);
        tx("[" + t + "] " + u(epoch) + " " + outcome([&] { return str(parse_command(t, epoch)); }));
    }
    return tx.h;
}
// the shim's files read through Ingress: commands in order, requests and their hashes, abort.s2
uint64_t ingress_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    const std::string d = fresh_dir();
    for (int round = 0; round < 60; ++round) {
        clear_dir(d);
        const uint64_t epoch = r.coin(90) ? r.U(1, 5) : 0;
        std::optional<Ingress> in;
        tx("new " + u(epoch) + " " + outcome([&] { in.emplace(d, epoch); return std::string(); }));
        if (!in) continue;
        int written = 0;
        for (int op = 0; op < 60; ++op) {
            const uint64_t k = r.U(0, 9);
            if (k <= 2 && written < 60) {   // the next cmd file
                const uint64_t id = r.U(0, 40);
                const uint64_t n_gen = r.U(1, 9);
                const uint64_t tok = r.U(0, 99);
                std::string bytes = "n_gen " + u(n_gen) + " temp 0 seed 1 min 0 stop -\n" + u(tok) + "\n";
                const bool good_hash = r.coin(85);
                const uint64_t h = good_hash ? payload_hash(bytes) : r.raw();
                if (r.coin(85)) atomic_file(d + "/req." + u(id), bytes);
                std::string c;
                if (r.coin(85)) {
                    const char* verb = r.coin(70) ? "ADMIT " : "CANCEL ";   // one draw a statement
                    const uint64_t e = r.coin(90) ? epoch : epoch + 1;
                    const std::string hs = r.coin(80) ? u(h) : "0";
                    c = verb + u(e) + " " + u(id) + " " + hs + "\n";
                } else c = command_text(r, epoch);
                const int at = r.coin(95) ? written : written + 1;   // a gap leaves the next command unread
                atomic_file(d + "/cmd." + u(at), c);
                tx("cmd." + u(at) + " [" + c + "]");
                written = std::max(written, at + 1);
            } else if (k <= 5) {
                tx("next " + outcome([&] { const auto c = in->next(); return c ? str(*c) : std::string("none"); }));
            } else if (k <= 7) {
                tx("peek " + outcome([&] { const auto c = in->peek(); return c ? str(*c) : std::string("none"); }));
            } else if (k == 8) {
                Command c;
                c.admit = r.coin(90); c.epoch = r.coin(90) ? epoch : epoch + 1; c.request = r.U(0, 40);
                const std::string bytes = slurp(d + "/req." + u(c.request));
                c.hash = r.coin(85) ? payload_hash(bytes) : r.raw();
                tx("request " + str(c) + " " + outcome([&] { return "[" + in->request(c) + "]"; }));
            } else {
                if (r.coin(20)) { atomic_file(d + "/abort.s2", "x\n"); tx("abort"); }
                else if (r.coin(50)) { ::unlink((d + "/abort.s2").c_str()); tx("unabort"); }
            }
        }
    }
    clear_dir(d); ::rmdir(d.c_str());
    for (const std::string& s : std::vector<std::string>{"", "a", "ab", std::string(1, '\0'), std::string("\xff\x00\x01", 3),
                                                        "n_gen 1 temp 0 seed 1 min 0 stop -\n1\n"})
        tx("hash " + u(s.size()) + " " + u(payload_hash(s)));
    for (int step = 0; step < 500; ++step) {
        std::string s;
        const uint64_t n = r.U(0, 300);
        for (uint64_t i = 0; i < n; ++i) s += char(r.U(0, 255));
        tx("hash " + u(n) + " " + u(payload_hash(s)));
    }
    return tx.h;
}
std::array<uint8_t, 32> digest(Rng& r) {
    std::array<uint8_t, 32> d{};
    for (auto& c : d) c = uint8_t(r.raw());
    return d;
}
// a steering receipt, loaded and valid most of the time
g6rt::steering::BootConfig steering(Rng& r) {
    g6rt::steering::BootConfig s;
    if (r.coin(40)) return s;
    const uint64_t n = r.coin(90) ? r.U(1, 3) : pick(r, {0, 9});
    s.loaded = r.coin(95) ? n > 0 : n == 0;
    s.scope = g6rt::steering::Scope(r.coin(95) ? r.U(1, 2) : 3);
    for (uint64_t i = 0; i < n; ++i) {
        g6rt::steering::BootConfig::Slot p;
        p.name = r.coin(95) ? "p" + text(r, "abcXYZ019._-", 12) : text(r, "ab ./", 4);
        p.profile_digest = digest(r); p.schedule_digest = digest(r);
        p.active_layers = uint32_t(r.coin(95) ? r.U(1, 92) : 0);
        s.profiles.push_back(p);
    }
    s.forced = r.coin(20);
    s.forced_scale_bits = uint32_t(s.forced ? pick(r, {0x3f800000, 0x40800000, 0x3e800000, 0x40800001, 0}) : r.coin(95) ? 0 : 1);
    return s;
}
// a scheduler config, valid in every model shape most of the time
Config config(Rng& r) {
    Config c;
    c.epoch = r.coin(95) ? r.U(1, 1000) : pick(r, {0, UINT64_MAX});
    c.pool_rows = uint32_t(r.coin(95) ? PAGE * r.U(1, 20000) : r.U(0, 100));
    c.mtp = uint32_t(r.coin(95) ? r.U(1, 3) : pick(r, {0, 8}));
    c.max_rows = uint32_t(r.coin(95) ? r.U(8, 48) : 0);
    c.max_sessions = uint32_t(r.coin(95) ? r.U(1, 8) : 0);
    c.guard_tokens = r.coin(70) ? UINT32_MAX : uint32_t(r.U(0, 20));
    c.prefill_rows = uint32_t(r.coin(95) ? PAGE * r.U(1, 128) : 0);
    c.two_lanes = r.coin(50);
    if (r.coin(20)) c.decode_prefill_rows = uint32_t(PAGE * r.U(0, 4));
    if (r.coin(20)) c.decode_rounds = uint32_t(r.U(0, 70));
    if (r.coin(20)) { c.cp_step_rows = uint32_t(r.U(0, 40)); c.cp_seg = r.coin(50); c.cp_after = r.U(0, 100000); }
    if (r.coin(20)) { c.window_pages = uint32_t(r.U(0, 100)); c.r2_rows = uint32_t(PAGE * r.U(0, 100)); }
    c.tail_first = r.coin(15);
    c.c8_pack = r.coin(5);
    c.park = r.coin(5);
    if (r.coin(20)) c.notice_max = uint32_t(r.U(1, 9));
    c.steering = steering(r);
    return c;
}
// ready.s2's capability line and the bootstrap's agreement
uint64_t capability_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int step = 0; step < 3000; ++step) {
        const Config c = config(r);
        const unsigned ranks = unsigned(r.coin(95) ? r.U(1, 6) : pick(r, {0, 7}));
        tx("cap " + u(ranks) + " " + outcome([&] { return capability(c, ranks); }));
        BootstrapConfig b;
        b.scheduler = c; b.ranks = ranks;
        if (r.coin(30)) b.prefill_min = uint32_t(r.U(0, 9000));
        if (r.coin(10)) b.control_version = unsigned(r.U(0, 20));
        b.runtime_signature = text(r, "G6_X=1 \n", 20);
        tx("agree " + outcome([&] { return b.agreement(); }));
    }
    return tx.h;
}
// atomic_file and publish_ready: the bytes on disk and the refusals
uint64_t files_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    const std::string d = fresh_dir();
    for (int step = 0; step < 600; ++step) {
        clear_dir(d);
        std::string bytes;
        const uint64_t n = r.U(0, 200);
        for (uint64_t i = 0; i < n; ++i) bytes += char(r.U(0, 255));
        const std::string p = r.coin(95) ? d + "/x" : d + "/missing/x";
        const std::string wrote = outcome([&] { atomic_file(p, bytes); return std::string(); });   // write before the reads
        tx("atomic " + wrote + " " + u(payload_hash(slurp(p))) + " tmp" + u(::access((p + ".tmp").c_str(), F_OK) == 0));
        BootstrapConfig b;
        b.scheduler = config(r);
        b.ranks = unsigned(r.U(1, 3)); b.rank = r.coin(90) ? 0 : 1;
        b.directory = d;
        ControlLinks links;
        const uint64_t nl = r.coin(85) ? b.ranks - 1 : r.U(0, 3);
        for (uint64_t i = 0; i < nl; ++i) {
            if (r.coin(5)) { links.links.push_back(nullptr); continue; }
            int sp[2];
            require(!::socketpair(AF_UNIX, SOCK_STREAM, 0, sp), "socketpair");
            links.links.push_back(std::make_unique<ControlIO>(sp[0]));
            ::close(sp[1]);
        }
        if (r.coin(10)) atomic_file(d + "/abort.s2", "x\n");
        const std::string ready = outcome([&] { publish_ready(b, links); return std::string(); });   // write before the read
        tx("ready " + ready + " [" + slurp(d + "/ready.s2") + "]");
    }
    clear_dir(d); ::rmdir(d.c_str());
    return tx.h;
}
// The trace's file. An instant's timestamp is the wall clock, so the transcript replaces each with "T".
std::string mask_ts(const std::string& s) {
    std::string out;
    size_t i = 0;
    for (;;) {
        const size_t line_end = s.find('\n', i);
        std::string line = s.substr(i, line_end == std::string::npos ? std::string::npos : line_end - i);
        if (line.find("\"ph\":\"i\"") != std::string::npos) {
            const size_t t = line.find("\"ts\":");
            if (t != std::string::npos) {
                size_t e = t + 5;
                while (e < line.size() && line[e] >= '0' && line[e] <= '9') ++e;
                line = line.substr(0, t + 5) + "T" + line.substr(e);
            }
        }
        out += line;
        if (line_end == std::string::npos) break;
        out += '\n';
        i = line_end + 1;
    }
    return out;
}
uint64_t trace_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    const std::string d = fresh_dir();
    const char* names[] = {"arrive", "notice", "shrink", "wait", "admit", "commit", "first token", "finish", "prefill",
                           "prompt tail", "decode"};
    for (int round = 0; round < 40; ++round) {
        clear_dir(d);
        const unsigned rank = unsigned(pick(r, {0, 1, 5}));
        const std::string path = r.coin(90) ? d + "/t" : r.coin(50) ? "" : d + "/missing/t";
        const uint64_t before = Trace::now();
        {
            Trace t;
            tx("open " + u(rank) + " [" + (path.empty() ? "" : path.substr(d.size())) + "]");
            t.open(path, rank);
            tx("on " + u(bool(t)));
            for (int ev = 0; bool(t) && ev < 40; ++ev) {
                const char* name = names[r.U(0, 10)];
                const int tid = int(r.U(0, 1));
                std::string args;
                const uint64_t na = r.U(0, 3);
                for (uint64_t a = 0; a < na; ++a) {
                    if (a) args += ",";
                    args += r.coin(70) ? targ("request", r.coin(80) ? r.U(0, 1000) : r.raw()) : targ("kind", r.coin(50) ? "admit" : "cancel");
                }
                if (r.coin(50)) {
                    const uint64_t t0 = r.coin(90) ? r.U(0, 1u << 30) : r.raw();
                    const uint64_t t1 = r.coin(80) ? t0 + r.U(0, 100000) : r.U(0, 1u << 30);
                    t.span(name, tid, t0, t1, args);
                } else t.instant(name, tid, args);
            }
        }
        const uint64_t after = Trace::now();
        tx("clock " + u(after >= before));
        tx("file [" + mask_ts(slurp(d + "/t.r" + u(rank) + ".json")) + "]");
    }
    clear_dir(d); ::rmdir(d.c_str());
    return tx.h;
}
// status.s2 as the head's control thread writes it: a Service built and not run (ranks 1), given seeded pool values
uint64_t status_section(uint64_t seed, std::vector<std::string>* lines) {
    Tx tx; tx.lines = lines;
    Rng r(seed);
    for (int round = 0; round < 12; ++round) {
        const std::string d = fresh_dir();
        Config c;
        c.epoch = r.U(1, 99); c.pool_rows = uint32_t(PAGE * r.U(1, 4000)); c.max_rows = 16; c.max_sessions = 8; c.mtp = 1;
        const uint32_t pages = c.pool_rows / PAGE;
        BootstrapConfig b;
        b.scheduler = c; b.ranks = 1; b.rank = 0; b.head_ip = "127.0.0.1"; b.port = 1; b.directory = d;
        b.runtime_signature = "service_vectors_test";
        ControlLinks links;
        RunPool pool{pages, pages, c.pool_rows};
        NativeHooks h;
        h.pool.pool = &pool;
        h.bind = [](const Grant&) {};
        h.retire = [](const Slot&) {};
        h.fence = [] {};
        h.dispatch.bucket = [](bool, int) { return 0; };
        h.dispatch.mode = [](int, int) {};
        h.dispatch.launch = [](const Launch& l) { return std::vector<int>(size_t(l.step.M()), 7); };
        h.dispatch.prefill = [](const Slot&, const Payload&, uint32_t, uint32_t) {};
        Service s(b, links, std::move(h), 1);
        uint64_t next_id = 1, revision = 0;
        for (int step = 0; step < 150; ++step) {
            PoolStatus in;
            in.free = uint32_t(r.coin(95) ? r.U(0, pages) : r.U(0, pages + 3));
            in.room = uint32_t(r.coin(95) ? r.U(0, in.free) : r.U(0, pages + 3));
            if (r.coin(30)) {
                const uint64_t n = r.U(0, 3);
                for (uint64_t i = 0; i < n; ++i) (s.*get(QueueOf{})).push(Request{next_id++, uint32_t(r.U(1, 300)), uint32_t(r.U(1, 300)), {}, 0});
            }
            if (r.coin(20) && (s.*get(QueueOf{})).size()) {
                const bool a = r.coin(50), b = r.coin(50);
                const uint32_t room = uint32_t(r.U(0, pages));
                (s.*get(QueueOf{})).check(revision++, a, b, room);
            }
            if (r.coin(10) && (s.*get(QueueOf{})).front()) (s.*get(QueueOf{})).cancel((s.*get(QueueOf{})).front()->id);
            const unsigned resident = unsigned(r.U(0, 8));
            const unsigned retiring = unsigned(r.U(0, 8));
            g6svc_control_publish((s.*get(ControlOf{})).get(), 0, 0, 0, resident, retiring);
            auto big = [&] { return r.coin(80) ? r.U(0, 5000) : r.raw(); };
            if (r.coin(50)) {
                g6kv_totals t{};
                t.live_runs = uint32_t(r.U(0, 8)); t.held_pages = uint32_t(big()); t.worst_pages = uint32_t(big());
                t.grows = big(); t.stalls = big();
                in.stats = t;
            }
            if (r.coin(50)) {
                std::vector<g6kv_tier_totals> ts(r.U(0, 3));
                for (auto& t : ts) {
                    t.slots = big(); t.free_slots = big(); t.entries = uint32_t(big()); t.shared = uint32_t(big());
                    t.spilled_pages = big(); t.moved_in_pages = big(); t.restored_pages = big(); t.hits = big();
                    t.dropped = big(); t.dropped_pages = big(); t.moved_out = big(); t.moved_out_pages = big();
                    t.refused = big(); t.refused_pages = big(); t.full = big();
                    t.block_bytes = r.coin(80) ? pick(r, {0, 4096, 1 << 20, 2359296}) : r.raw();
                    t.credit = int64_t(r.coin(80) ? r.U(0, 100) : r.raw());
                    t.oversize = big(); t.oversize_last = big();
                }
                in.tiers = ts;
            }
            if (r.coin(50)) {
                std::vector<PoolUse> us(r.U(0, 8));
                for (auto& x : us) { x.table_version = uint32_t(r.U(0, 9)); x.held = uint32_t(big()); x.worst = uint32_t(big()); }
                in.runs = us;
            }
            ::unlink((d + "/status.s2").c_str());
            const std::string wrote = outcome([&] { (s.*get(StatusOf{}))(in); return std::string(); });   // write before the read
            tx(wrote + " [" + slurp(d + "/status.s2") + "]");
        }
        ::unlink((d + "/status.s2").c_str()); ::unlink((d + "/status.s2.tmp").c_str());
        ::rmdir(d.c_str());
    }
    return tx.h;
}
} // namespace

int main(int argc, char** argv) {
    struct Section { const char* name; uint64_t seed; uint64_t (*run)(uint64_t, std::vector<std::string>*); uint64_t want; };
    // DS41 takes configs the other shapes refuse (mtp 0, park, c8 packs)
    const bool ds41 = g6mod::has(G6_CAP_PARK);
    const Section sections[] = {
        {"number", 1, number_section, 0x33740fd68ae4b673ull},
        {"command", 2, command_section, 0x7830189284d224c3ull},
        {"ingress", 3, ingress_section, 0x16ff191341caef03ull},
        {"capability", 4, capability_section, ds41 ? 0x15bd2028362a5baeull : 0x4aee24eb01c384c0ull},
        {"files", 5, files_section, ds41 ? 0x816facccb407eb3cull : 0x41d519b6797d1bdaull},
        {"trace", 6, trace_section, 0x8a798dcc6c22b7adull},
        {"status", 7, status_section, 0x25ab880d0fafd229ull},
    };
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
    int fails = 0;
    for (const auto& s : sections) {
        const uint64_t h = s.run(s.seed, nullptr);
        const bool ok = h == s.want;
        if (!ok) ++fails;
        std::printf("%-10s 0x%016llxull %s\n", s.name, (unsigned long long)h, ok ? "ok" : "MISMATCH");
    }
    std::printf(fails ? "service_vectors_test: %d FAIL\n" : "service_vectors_test: PASS\n", fails);
    return fails != 0;
}
